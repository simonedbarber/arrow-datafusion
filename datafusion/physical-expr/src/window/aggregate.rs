// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Physical exec for aggregate window function expressions.

use crate::window::partition_evaluator::find_ranges_in_range;
use crate::{expressions::PhysicalSortExpr, PhysicalExpr};
use crate::{window::WindowExpr, AggregateExpr};
use arrow::array::new_empty_array;
use arrow::compute::concat;
use arrow::record_batch::RecordBatch;
use arrow::{array::ArrayRef, datatypes::Field};
use datafusion_common::DataFusionError;
use datafusion_common::Result;
use datafusion_expr::Accumulator;
use datafusion_expr::{WindowFrame, WindowFrameBound, WindowFrameUnits};
use std::any::Any;
use std::iter::IntoIterator;
use std::ops::Range;
use std::sync::Arc;

/// A window expr that takes the form of an aggregate function
#[derive(Debug)]
pub struct AggregateWindowExpr {
    aggregate: Arc<dyn AggregateExpr>,
    partition_by: Vec<Arc<dyn PhysicalExpr>>,
    order_by: Vec<PhysicalSortExpr>,
    window_frame: Option<WindowFrame>,
}

impl AggregateWindowExpr {
    /// create a new aggregate window function expression
    pub fn new(
        aggregate: Arc<dyn AggregateExpr>,
        partition_by: &[Arc<dyn PhysicalExpr>],
        order_by: &[PhysicalSortExpr],
        window_frame: Option<WindowFrame>,
    ) -> Self {
        Self {
            aggregate,
            partition_by: partition_by.to_vec(),
            order_by: order_by.to_vec(),
            window_frame,
        }
    }

    /// the aggregate window function operates based on window frame, and by default the mode is
    /// "range".
    fn evaluation_mode(&self) -> WindowFrameUnits {
        self.window_frame.unwrap_or_default().units
    }

    /// create a new accumulator based on the underlying aggregation function
    fn create_accumulator(&self) -> Result<AggregateWindowAccumulator> {
        let accumulator = self.aggregate.create_accumulator()?;
        Ok(AggregateWindowAccumulator { accumulator })
    }

    /// peer based evaluation based on the fact that batch is pre-sorted given the sort columns
    /// and then per partition point we'll evaluate the peer group (e.g. SUM or MAX gives the same
    /// results for peers) and concatenate the results.
    fn peer_based_evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            // An empty batch with no PARTITION BY would otherwise produce a single
            // empty peer range, which scan_peers rejects.
            return Ok(new_empty_array(self.aggregate.field()?.data_type()));
        }
        let partition_points =
            self.evaluate_partition_points(num_rows, &self.partition_columns(batch)?)?;
        let sort_partition_points =
            self.evaluate_partition_points(num_rows, &self.sort_columns(batch)?)?;
        let values = self.evaluate_args(batch)?;
        let results = partition_points
            .iter()
            .map(|partition_range| {
                let sort_partition_points =
                    find_ranges_in_range(partition_range, &sort_partition_points);
                let mut window_accumulators = self.create_accumulator()?;
                sort_partition_points
                    .iter()
                    .map(|range| window_accumulators.scan_peers(&values, range))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<Vec<ArrayRef>>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<ArrayRef>>();
        let results = results.iter().map(|i| i.as_ref()).collect::<Vec<_>>();
        concat(&results).map_err(DataFusionError::ArrowError)
    }

    fn group_based_evaluate(&self, _batch: &RecordBatch) -> Result<ArrayRef> {
        Err(DataFusionError::NotImplemented(format!(
            "Group based evaluation for {} is not yet implemented",
            self.name()
        )))
    }

    fn row_based_evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        // This accumulator only grows: bounded or following frames require a
        // different algorithm. Do not silently reinterpret them as cumulative.
        match self.window_frame {
            Some(WindowFrame {
                units: WindowFrameUnits::Rows,
                start_bound: WindowFrameBound::Preceding(None),
                end_bound: WindowFrameBound::CurrentRow,
            }) => (),
            _ => return Err(DataFusionError::NotImplemented(format!(
                "Row based evaluation for {} supports only ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
                self.name()
            ))),
        }
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(new_empty_array(self.aggregate.field()?.data_type()));
        }
        let partition_points =
            self.evaluate_partition_points(num_rows, &self.partition_columns(batch)?)?;
        let values = self.evaluate_args(batch)?;
        let mut results = Vec::with_capacity(num_rows);
        for partition in partition_points {
            let mut accumulator = self.create_accumulator()?;
            for row in partition {
                // ROWS advances once per row, including duplicate ORDER BY peers.
                results.push(accumulator.scan_peers(&values, &(row..row + 1))?);
            }
        }
        let results = results.iter().map(|array| array.as_ref()).collect::<Vec<_>>();
        concat(&results).map_err(DataFusionError::ArrowError)
    }
}

impl WindowExpr for AggregateWindowExpr {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        self.aggregate.name()
    }

    fn field(&self) -> Result<Field> {
        self.aggregate.field()
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        self.aggregate.expressions()
    }

    fn partition_by(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.partition_by
    }

    fn order_by(&self) -> &[PhysicalSortExpr] {
        &self.order_by
    }

    /// evaluate the window function values against the batch
    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        match self.evaluation_mode() {
            WindowFrameUnits::Range => self.peer_based_evaluate(batch),
            WindowFrameUnits::Rows => self.row_based_evaluate(batch),
            WindowFrameUnits::Groups => self.group_based_evaluate(batch),
        }
    }
}

/// Aggregate window accumulator utilizes the accumulator from aggregation and do a accumulative sum
/// across evaluation arguments based on peer equivalences.
#[derive(Debug)]
struct AggregateWindowAccumulator {
    accumulator: Box<dyn Accumulator>,
}

impl AggregateWindowAccumulator {
    /// scan one peer group of values (as arguments to window function) given by the value_range
    /// and return evaluation result that are of the same number of rows.
    fn scan_peers(
        &mut self,
        values: &[ArrayRef],
        value_range: &Range<usize>,
    ) -> Result<ArrayRef> {
        if value_range.is_empty() {
            return Err(DataFusionError::Internal(
                "Value range cannot be empty".to_owned(),
            ));
        }
        let len = value_range.end - value_range.start;
        let values = values
            .iter()
            .map(|v| v.slice(value_range.start, len))
            .collect::<Vec<_>>();
        self.accumulator.update_batch(&values)?;
        let value = self.accumulator.evaluate()?;
        Ok(value.to_array_of_size(len))
    }
}


#[cfg(test)]
mod cumulative_rows_tests {
    use super::*;
    use crate::expressions::{Column, Sum};
    use arrow::array::{Array, Int64Array};
    use arrow::compute::SortOptions;
    use arrow::datatypes::{DataType, Schema};

    fn batch(partitions: Vec<i64>, keys: Vec<i64>, values: Vec<Option<i64>>) -> RecordBatch {
        RecordBatch::try_new(Arc::new(Schema::new(vec![
            Field::new("partition", DataType::Int64, false),
            Field::new("key", DataType::Int64, false),
            Field::new("value", DataType::Int64, true),
        ])), vec![Arc::new(Int64Array::from(partitions)), Arc::new(Int64Array::from(keys)),
            Arc::new(Int64Array::from(values))]).unwrap()
    }

    fn window(units: WindowFrameUnits, start_bound: WindowFrameBound, end_bound: WindowFrameBound) -> AggregateWindowExpr {
        AggregateWindowExpr::new(
            Arc::new(Sum::new(Arc::new(Column::new("value", 2)), "SUM(value)", DataType::Int64)),
            &[Arc::new(Column::new("partition", 0))],
            &[PhysicalSortExpr { expr: Arc::new(Column::new("key", 1)), options: SortOptions::default() }],
            Some(WindowFrame { units, start_bound, end_bound }),
        )
    }

    fn values(array: &ArrayRef) -> Vec<Option<i64>> {
        array.as_any().downcast_ref::<Int64Array>().unwrap().iter().collect()
    }

    #[test]
    fn cumulative_rows_distinguishes_peers_nulls_and_partition_resets_from_range() {
        let input = batch(vec![1, 1, 1, 1, 2, 2], vec![1, 1, 2, 3, 1, 1],
            vec![None, Some(2), None, Some(3), Some(7), None]);
        let rows = window(WindowFrameUnits::Rows, WindowFrameBound::Preceding(None), WindowFrameBound::CurrentRow);
        assert_eq!(values(&rows.evaluate(&input).unwrap()), vec![None, Some(2), Some(2), Some(5), Some(7), Some(7)]);
        let range = window(WindowFrameUnits::Range, WindowFrameBound::Preceding(None), WindowFrameBound::CurrentRow);
        assert_eq!(values(&range.evaluate(&input).unwrap()), vec![Some(2), Some(2), Some(2), Some(5), Some(7), Some(7)]);
    }

    #[test]
    fn cumulative_rows_handles_empty_and_all_null_partitions() {
        let rows = window(WindowFrameUnits::Rows, WindowFrameBound::Preceding(None), WindowFrameBound::CurrentRow);
        let empty = rows.evaluate(&batch(vec![], vec![], vec![])).unwrap();
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.data_type(), &DataType::Int64);
        assert_eq!(values(&rows.evaluate(&batch(vec![1, 1, 2], vec![1, 2, 1], vec![None, None, None])).unwrap()), vec![None, None, None]);
    }

    #[test]
    fn cumulative_rows_evaluates_full_input_before_late_selection_and_page() {
        let input = batch(vec![1; 5], vec![1, 2, 3, 4, 5], vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
        let rows = window(WindowFrameUnits::Rows, WindowFrameBound::Preceding(None), WindowFrameBound::CurrentRow);
        let result = values(&rows.evaluate(&input).unwrap());
        // A consumer selecting the final two rows sees prior source history,
        // and paging after this evaluation retains the full cumulative state.
        assert_eq!(&result[3..], &[Some(10), Some(15)]);
        assert_eq!(result[4], Some(15));
    }

    #[test]
    fn cumulative_rows_refuses_other_frames_without_reinterpreting_them() {
        let input = batch(vec![1], vec![1], vec![Some(1)]);
        for (start, end) in [
            (WindowFrameBound::Preceding(Some(1)), WindowFrameBound::CurrentRow),
            (WindowFrameBound::Preceding(None), WindowFrameBound::Following(Some(1))),
            (WindowFrameBound::CurrentRow, WindowFrameBound::Following(None)),
        ] {
            let rows = window(WindowFrameUnits::Rows, start, end);
            assert!(matches!(rows.evaluate(&input), Err(DataFusionError::NotImplemented(_))));
        }
        let groups = window(WindowFrameUnits::Groups, WindowFrameBound::Preceding(None), WindowFrameBound::CurrentRow);
        assert!(matches!(groups.evaluate(&input), Err(DataFusionError::NotImplemented(_))));
    }
}

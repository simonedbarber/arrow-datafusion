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

use crate::expressions::{Count, Sum};
use crate::window::partition_evaluator::find_ranges_in_range;
use crate::{expressions::PhysicalSortExpr, PhysicalExpr};
use crate::{window::WindowExpr, AggregateExpr};
use arrow::array::new_empty_array;
use arrow::compute::concat;
use arrow::record_batch::RecordBatch;
use arrow::{
    array::ArrayRef,
    datatypes::{DataType, Field},
};
use datafusion_common::DataFusionError;
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::Accumulator;
use datafusion_expr::{WindowFrame, WindowFrameBound, WindowFrameUnits};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use std::any::Any;
use std::convert::TryFrom;
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
        let frame = self.window_frame.unwrap_or_default();
        if frame.units != WindowFrameUnits::Rows
            || frame.start_bound == WindowFrameBound::Following(None)
            || frame.end_bound == WindowFrameBound::Preceding(None)
            || frame.start_bound > frame.end_bound
        {
            return Err(DataFusionError::Execution(format!(
                "Invalid ROWS frame: {}",
                frame
            )));
        }
        let cumulative = frame.start_bound == WindowFrameBound::Preceding(None)
            && frame.end_bound == WindowFrameBound::CurrentRow;
        // Other aggregate states may have distinct/order-dependent merge contracts.
        // Extend finite/following frames only for the ordinary SUM/COUNT owners.
        if !cumulative
            && !self.aggregate.as_any().is::<Sum>()
            && !self.aggregate.as_any().is::<Count>()
        {
            return Err(DataFusionError::NotImplemented(format!(
                "Finite/following ROWS evaluation for {} requires a qualified aggregate state",
                self.name()
            )));
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
            if cumulative {
                let mut accumulator = self.create_accumulator()?;
                for row in partition {
                    // Preserve the existing cumulative path, including peer semantics.
                    results.push(accumulator.scan_peers(&values, &(row..row + 1))?);
                }
                continue;
            }
            let mut queue = RowFrameQueue::new(self.aggregate.as_ref())?;
            let mut loaded = partition.start;
            let mut removed = partition.start;
            for row in partition.clone() {
                let relative = row - partition.start;
                let start = partition.start
                    + row_frame_boundary(
                        frame.start_bound,
                        relative,
                        partition.len(),
                        false,
                    );
                let end = partition.start
                    + row_frame_boundary(
                        frame.end_bound,
                        relative,
                        partition.len(),
                        true,
                    );
                // Both boundaries advance monotonically in physical (already sorted)
                // row order. An empty frame must evaluate SUM=NULL and COUNT=0.
                // Evict before appending: the union of consecutive frames may
                // overflow even though each actual frame is representable.
                while removed < start.min(loaded) {
                    queue.pop()?;
                    removed += 1;
                }
                if loaded < start {
                    // Skipped rows cannot enter any future frame because both
                    // bounds advance monotonically. Do not aggregate them.
                    loaded = start;
                    removed = start;
                }
                while loaded < end {
                    let row_values = values
                        .iter()
                        .map(|v| v.slice(loaded, 1))
                        .collect::<Vec<_>>();
                    queue.push(&row_values)?;
                    loaded += 1;
                }
                results.push(queue.evaluate()?.to_array_of_size(1));
            }
        }
        let results = results
            .iter()
            .map(|array| array.as_ref())
            .collect::<Vec<_>>();
        concat(&results).map_err(DataFusionError::ArrowError)
    }
}

/// Clipped half-open ROWS boundaries. Saturating arithmetic also handles bounds
/// larger than the partition (including u64::MAX) without wrapping or losing rows.
fn row_frame_boundary(
    bound: WindowFrameBound,
    row: usize,
    len: usize,
    end: bool,
) -> usize {
    let current = row + usize::from(end);
    match bound {
        WindowFrameBound::Preceding(None) => 0,
        WindowFrameBound::Following(None) => len,
        WindowFrameBound::CurrentRow => current,
        WindowFrameBound::Preceding(Some(n)) => {
            current.saturating_sub(usize::try_from(n).unwrap_or(usize::MAX))
        }
        WindowFrameBound::Following(Some(n)) => current
            .saturating_add(usize::try_from(n).unwrap_or(usize::MAX))
            .min(len),
    }
}

/// Sliding aggregation with two stacks of mergeable states. Each input is
/// accumulated once and transferred at most once, giving O(n) aggregate work
/// and O(frame width) retained states. No subtraction (which loses NULL counts,
/// decimal types and finite values after an infinity leaves the frame) is used.
/// Exact SUM stack states need more range than their output type: a suffix
/// can exceed Decimal128 even when cancellation makes every complete frame fit.
/// This state stays internal to the existing native window owner, never in a
/// result field or serialized query document.
enum RowFrameState {
    Exact(Option<BigInt>),
    Aggregate(Vec<ArrayRef>),
}

struct RowFrameQueue<'a> {
    aggregate: &'a dyn AggregateExpr,
    exact_sum_type: Option<DataType>,
    back: Vec<(RowFrameState, RowFrameState)>,
    front: Vec<RowFrameState>,
}

impl<'a> RowFrameQueue<'a> {
    fn new(aggregate: &'a dyn AggregateExpr) -> Result<Self> {
        let data_type = aggregate.field()?.data_type().clone();
        let exact_sum_type = if aggregate.as_any().is::<Sum>()
            && matches!(
                data_type,
                DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64
                    | DataType::Decimal(_, _)
            ) {
            Some(data_type)
        } else {
            None
        };
        Ok(Self {
            aggregate,
            exact_sum_type,
            back: Vec::new(),
            front: Vec::new(),
        })
    }

    fn states(accumulator: &dyn Accumulator) -> Result<RowFrameState> {
        Ok(RowFrameState::Aggregate(
            accumulator.state()?.iter().map(|v| v.to_array()).collect(),
        ))
    }

    fn row_state(&self, values: &[ArrayRef]) -> Result<RowFrameState> {
        if let Some(data_type) = &self.exact_sum_type {
            let value = ScalarValue::try_from_array(&values[0], 0)?;
            if value.is_null() {
                return Ok(RowFrameState::Exact(None));
            }
            let coefficient =
                match value {
                    ScalarValue::Int8(Some(v)) => BigInt::from(v),
                    ScalarValue::Int16(Some(v)) => BigInt::from(v),
                    ScalarValue::Int32(Some(v)) => BigInt::from(v),
                    ScalarValue::Int64(Some(v)) => BigInt::from(v),
                    ScalarValue::UInt8(Some(v)) => BigInt::from(v),
                    ScalarValue::UInt16(Some(v)) => BigInt::from(v),
                    ScalarValue::UInt32(Some(v)) => BigInt::from(v),
                    ScalarValue::UInt64(Some(v)) => BigInt::from(v),
                    ScalarValue::Decimal128(Some(v), _, scale) => {
                        let target_scale = match data_type {
                            DataType::Decimal(_, s) => *s,
                            _ => 0,
                        };
                        if scale > target_scale {
                            return Err(DataFusionError::Execution(
                                "ROWS SUM cannot reduce an exact input scale".to_owned(),
                            ));
                        }
                        // Decimal coefficients are already scaled integers. Never
                        // convert them through f64 or decimal text.
                        return Ok(RowFrameState::Exact(Some(
                            BigInt::from(v)
                                * BigInt::from(10_u8).pow((target_scale - scale) as u32),
                        )));
                    }
                    _ => return Err(DataFusionError::Execution(
                        "ROWS SUM requires an exact numeric input for its declared type"
                            .to_owned(),
                    )),
                };
            let scale = match data_type {
                DataType::Decimal(_, s) => *s,
                _ => 0,
            };
            return Ok(RowFrameState::Exact(Some(
                coefficient * BigInt::from(10_u8).pow(scale as u32),
            )));
        }
        let mut row = self.aggregate.create_accumulator()?;
        row.update_batch(values)?;
        Self::states(row.as_ref())
    }

    fn merge(&self, states: &[&RowFrameState]) -> Result<RowFrameState> {
        if self.exact_sum_type.is_some() {
            let mut sum: Option<BigInt> = None;
            for state in states {
                match state {
                    RowFrameState::Exact(Some(value)) => match &mut sum {
                        Some(sum) => *sum += value,
                        None => sum = Some(value.clone()),
                    },
                    RowFrameState::Exact(None) => (),
                    _ => {
                        return Err(DataFusionError::Internal(
                            "Invalid exact ROWS aggregate state".to_owned(),
                        ))
                    }
                }
            }
            return Ok(RowFrameState::Exact(sum));
        }
        let mut merged = self.aggregate.create_accumulator()?;
        for state in states {
            match state {
                RowFrameState::Aggregate(state) => merged.merge_batch(state)?,
                _ => {
                    return Err(DataFusionError::Internal(
                        "Invalid ROWS aggregate state".to_owned(),
                    ))
                }
            }
        }
        Self::states(merged.as_ref())
    }

    fn push(&mut self, values: &[ArrayRef]) -> Result<()> {
        let row = self.row_state(values)?;
        let merged = match self.back.last() {
            Some((_, state)) => self.merge(&[state, &row])?,
            None => self.merge(&[&row])?,
        };
        self.back.push((row, merged));
        Ok(())
    }

    fn pop(&mut self) -> Result<()> {
        if self.front.is_empty() {
            while let Some((row, _)) = self.back.pop() {
                let merged = match self.front.last() {
                    Some(state) => self.merge(&[&row, state])?,
                    None => self.merge(&[&row])?,
                };
                self.front.push(merged);
            }
        }
        self.front.pop().ok_or_else(|| {
            DataFusionError::Internal("Empty ROWS aggregate queue".to_owned())
        })?;
        Ok(())
    }

    fn exact_result(&self, sum: Option<BigInt>) -> Result<ScalarValue> {
        let data_type = self.exact_sum_type.as_ref().ok_or_else(|| {
            DataFusionError::Internal("Missing exact ROWS SUM type".to_owned())
        })?;
        let value = match sum {
            Some(value) => value,
            None => return ScalarValue::try_from(data_type),
        };
        let overflow = || {
            DataFusionError::Execution(format!("ROWS SUM result exceeds {:?}", data_type))
        };
        macro_rules! integer_result {
            ($convert:ident, $kind:ident) => {
                value
                    .$convert()
                    .map(|v| ScalarValue::$kind(Some(v)))
                    .ok_or_else(overflow)
            };
        }
        match data_type {
            DataType::Int8 => integer_result!(to_i8, Int8),
            DataType::Int16 => integer_result!(to_i16, Int16),
            DataType::Int32 => integer_result!(to_i32, Int32),
            DataType::Int64 => integer_result!(to_i64, Int64),
            DataType::UInt8 => integer_result!(to_u8, UInt8),
            DataType::UInt16 => integer_result!(to_u16, UInt16),
            DataType::UInt32 => integer_result!(to_u32, UInt32),
            DataType::UInt64 => integer_result!(to_u64, UInt64),
            DataType::Decimal(precision, scale) => {
                let bound = BigInt::from(10_u8).pow(*precision as u32);
                if value >= bound || value <= -bound {
                    return Err(overflow());
                }
                value
                    .to_i128()
                    .map(|v| ScalarValue::Decimal128(Some(v), *precision, *scale))
                    .ok_or_else(overflow)
            }
            _ => Err(DataFusionError::Internal(
                "Invalid exact ROWS SUM type".to_owned(),
            )),
        }
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        let states = self
            .front
            .last()
            .into_iter()
            .chain(self.back.last().map(|(_, state)| state))
            .collect::<Vec<_>>();
        if self.exact_sum_type.is_some() {
            return match self.merge(&states)? {
                RowFrameState::Exact(sum) => self.exact_result(sum),
                _ => Err(DataFusionError::Internal(
                    "Invalid exact ROWS SUM state".to_owned(),
                )),
            };
        }
        let mut merged = self.aggregate.create_accumulator()?;
        for state in states {
            match state {
                RowFrameState::Aggregate(state) => merged.merge_batch(state)?,
                _ => {
                    return Err(DataFusionError::Internal(
                        "Invalid ROWS SUM state".to_owned(),
                    ))
                }
            }
        }
        merged.evaluate()
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

    fn batch(
        partitions: Vec<i64>,
        keys: Vec<i64>,
        values: Vec<Option<i64>>,
    ) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("partition", DataType::Int64, false),
                Field::new("key", DataType::Int64, false),
                Field::new("value", DataType::Int64, true),
            ])),
            vec![
                Arc::new(Int64Array::from(partitions)),
                Arc::new(Int64Array::from(keys)),
                Arc::new(Int64Array::from(values)),
            ],
        )
        .unwrap()
    }

    fn window(
        units: WindowFrameUnits,
        start_bound: WindowFrameBound,
        end_bound: WindowFrameBound,
    ) -> AggregateWindowExpr {
        AggregateWindowExpr::new(
            Arc::new(Sum::new(
                Arc::new(Column::new("value", 2)),
                "SUM(value)",
                DataType::Int64,
            )),
            &[Arc::new(Column::new("partition", 0))],
            &[PhysicalSortExpr {
                expr: Arc::new(Column::new("key", 1)),
                options: SortOptions::default(),
            }],
            Some(WindowFrame {
                units,
                start_bound,
                end_bound,
            }),
        )
    }

    fn values(array: &ArrayRef) -> Vec<Option<i64>> {
        array
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .iter()
            .collect()
    }

    #[test]
    fn cumulative_rows_distinguishes_peers_nulls_and_partition_resets_from_range() {
        let input = batch(
            vec![1, 1, 1, 1, 2, 2],
            vec![1, 1, 2, 3, 1, 1],
            vec![None, Some(2), None, Some(3), Some(7), None],
        );
        let rows = window(
            WindowFrameUnits::Rows,
            WindowFrameBound::Preceding(None),
            WindowFrameBound::CurrentRow,
        );
        assert_eq!(
            values(&rows.evaluate(&input).unwrap()),
            vec![None, Some(2), Some(2), Some(5), Some(7), Some(7)]
        );
        let range = window(
            WindowFrameUnits::Range,
            WindowFrameBound::Preceding(None),
            WindowFrameBound::CurrentRow,
        );
        assert_eq!(
            values(&range.evaluate(&input).unwrap()),
            vec![Some(2), Some(2), Some(2), Some(5), Some(7), Some(7)]
        );
    }

    #[test]
    fn cumulative_rows_handles_empty_and_all_null_partitions() {
        let rows = window(
            WindowFrameUnits::Rows,
            WindowFrameBound::Preceding(None),
            WindowFrameBound::CurrentRow,
        );
        let empty = rows.evaluate(&batch(vec![], vec![], vec![])).unwrap();
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.data_type(), &DataType::Int64);
        assert_eq!(
            values(
                &rows
                    .evaluate(&batch(
                        vec![1, 1, 2],
                        vec![1, 2, 1],
                        vec![None, None, None]
                    ))
                    .unwrap()
            ),
            vec![None, None, None]
        );
    }

    #[test]
    fn cumulative_rows_evaluates_full_input_before_late_selection_and_page() {
        let input = batch(
            vec![1; 5],
            vec![1, 2, 3, 4, 5],
            vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
        );
        let rows = window(
            WindowFrameUnits::Rows,
            WindowFrameBound::Preceding(None),
            WindowFrameBound::CurrentRow,
        );
        let result = values(&rows.evaluate(&input).unwrap());
        // A consumer selecting the final two rows sees prior source history,
        // and paging after this evaluation retains the full cumulative state.
        assert_eq!(&result[3..], &[Some(10), Some(15)]);
        assert_eq!(result[4], Some(15));
    }

    #[test]
    fn rows_frames_match_independent_slices_at_partition_edges_and_nulls() {
        use WindowFrameBound::{CurrentRow, Following, Preceding};
        let input = batch(
            vec![1, 1, 1, 1, 1, 2, 2, 2],
            vec![1, 1, 2, 3, 4, 1, 2, 3],
            vec![Some(5), None, Some(-5), None, Some(7), None, None, None],
        );
        let source = [Some(5), None, Some(-5), None, Some(7), None, None, None];
        let bounds = [
            Preceding(None),
            Preceding(Some(u64::MAX)),
            Preceding(Some(2)),
            Preceding(Some(0)),
            CurrentRow,
            Following(Some(0)),
            Following(Some(2)),
            Following(Some(u64::MAX)),
            Following(None),
        ];
        for start in bounds {
            for end in bounds {
                if start == Following(None) || end == Preceding(None) || start > end {
                    continue;
                }
                let rows = window(WindowFrameUnits::Rows, start, end);
                let actual = values(&rows.evaluate(&input).unwrap());
                let mut expected = Vec::new();
                for partition in [0..5, 5..8] {
                    for row in partition.clone() {
                        // Independent signed arithmetic; no production boundary helper.
                        let boundary = |b, inclusive_end| -> usize {
                            let relative = (row - partition.start) as i128
                                + if inclusive_end { 1 } else { 0 };
                            let offset = match b {
                                Preceding(None) => 0,
                                Following(None) => partition.len() as i128,
                                Preceding(Some(n)) => relative - n as i128,
                                Following(Some(n)) => relative + n as i128,
                                CurrentRow => relative,
                            };
                            partition.start
                                + offset.max(0).min(partition.len() as i128) as usize
                        };
                        let a = boundary(start, false);
                        let b = boundary(end, true);
                        let non_null = source[a..b.max(a)]
                            .iter()
                            .flatten()
                            .copied()
                            .collect::<Vec<_>>();
                        expected.push(if non_null.is_empty() {
                            None
                        } else {
                            Some(non_null.iter().sum())
                        });
                    }
                }
                assert_eq!(actual, expected, "{:?} to {:?}", start, end);
            }
        }
    }

    #[test]
    fn finite_count_counts_observations_and_empty_frames_without_partition_leaks() {
        let input = batch(
            vec![1, 1, 1, 2, 2],
            vec![1, 2, 3, 1, 2],
            vec![Some(5), None, Some(7), None, None],
        );
        let count = AggregateWindowExpr::new(
            Arc::new(Count::new(
                Arc::new(Column::new("value", 2)),
                "COUNT(value)",
                DataType::Int64,
            )),
            &[Arc::new(Column::new("partition", 0))],
            &[],
            Some(WindowFrame {
                units: WindowFrameUnits::Rows,
                start_bound: WindowFrameBound::Following(Some(1)),
                end_bound: WindowFrameBound::Following(Some(2)),
            }),
        );
        assert_eq!(
            values(&count.evaluate(&input).unwrap()),
            vec![Some(1), Some(1), Some(0), Some(0), Some(0)]
        );
        let empty = count.evaluate(&batch(vec![], vec![], vec![])).unwrap();
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.data_type(), &DataType::Int64);
    }

    #[test]
    fn rows_rejects_invalid_frames_and_keeps_groups_unsupported() {
        let input = batch(vec![1], vec![1], vec![Some(1)]);
        for (start, end) in [
            (
                WindowFrameBound::Following(None),
                WindowFrameBound::Following(None),
            ),
            (
                WindowFrameBound::Preceding(None),
                WindowFrameBound::Preceding(None),
            ),
            (
                WindowFrameBound::Following(Some(1)),
                WindowFrameBound::CurrentRow,
            ),
        ] {
            assert!(matches!(
                window(WindowFrameUnits::Rows, start, end).evaluate(&input),
                Err(DataFusionError::Execution(_))
            ));
        }
        assert!(matches!(
            window(
                WindowFrameUnits::Groups,
                WindowFrameBound::Preceding(None),
                WindowFrameBound::CurrentRow
            )
            .evaluate(&input),
            Err(DataFusionError::NotImplemented(_))
        ));
    }

    #[test]
    fn finite_sum_preserves_decimal_integer_and_float_null_types() {
        use arrow::array::Float64Array;
        use datafusion_common::ScalarValue;
        let evaluate = |array: ArrayRef| {
            let data_type = array.data_type().clone();
            let input = RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "value",
                    data_type.clone(),
                    true,
                )])),
                vec![array],
            )
            .unwrap();
            let expr = AggregateWindowExpr::new(
                Arc::new(Sum::new(
                    Arc::new(Column::new("value", 0)),
                    "SUM(value)",
                    data_type,
                )),
                &[],
                &[],
                Some(WindowFrame {
                    units: WindowFrameUnits::Rows,
                    start_bound: WindowFrameBound::Preceding(Some(1)),
                    end_bound: WindowFrameBound::CurrentRow,
                }),
            );
            expr.evaluate(&input).unwrap()
        };
        let large = 9007199254740993;
        let ints = evaluate(Arc::new(Int64Array::from(vec![
            None,
            Some(large),
            Some(2),
            None,
            None,
        ])));
        assert_eq!(
            values(&ints),
            vec![None, Some(large), Some(large + 2), Some(2), None]
        );
        let decimals = ScalarValue::iter_to_array(
            vec![
                ScalarValue::Decimal128(None, 38, 6),
                ScalarValue::Decimal128(Some(large as i128), 38, 6),
                ScalarValue::Decimal128(Some(2), 38, 6),
                ScalarValue::Decimal128(None, 38, 6),
                ScalarValue::Decimal128(None, 38, 6),
            ]
            .into_iter(),
        )
        .unwrap();
        let decimal_result = evaluate(decimals);
        assert_eq!(decimal_result.data_type(), &DataType::Decimal(38, 6));
        for (row, target) in [
            None,
            Some(large as i128),
            Some(large as i128 + 2),
            Some(2),
            None,
        ]
        .iter()
        .enumerate()
        {
            assert_eq!(
                ScalarValue::try_from_array(&decimal_result, row).unwrap(),
                ScalarValue::Decimal128(*target, 38, 6)
            );
        }
        let floats = evaluate(Arc::new(Float64Array::from(vec![
            Some(f64::INFINITY),
            Some(2.5),
            Some(-0.5),
            None,
            None,
        ])));
        assert_eq!(floats.data_type(), &DataType::Float64);
        let floats = floats.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(
            floats.iter().collect::<Vec<_>>(),
            vec![
                Some(f64::INFINITY),
                Some(f64::INFINITY),
                Some(2.0),
                Some(-0.5),
                None
            ]
        );
    }

    #[test]
    fn adjacent_frames_do_not_aggregate_their_unrequested_union() {
        let input = batch(
            vec![1; 4],
            vec![1, 2, 3, 4],
            vec![
                Some(i64::MAX),
                Some(i64::MAX),
                Some(i64::MIN),
                Some(i64::MIN),
            ],
        );
        let current = window(
            WindowFrameUnits::Rows,
            WindowFrameBound::CurrentRow,
            WindowFrameBound::CurrentRow,
        );
        assert_eq!(
            values(&current.evaluate(&input).unwrap()),
            vec![
                Some(i64::MAX),
                Some(i64::MAX),
                Some(i64::MIN),
                Some(i64::MIN)
            ]
        );
        // An empty past frame must not aggregate unrelated (overflowing) rows.
        let distant = window(
            WindowFrameUnits::Rows,
            WindowFrameBound::Preceding(Some(100)),
            WindowFrameBound::Preceding(Some(50)),
        );
        assert_eq!(values(&distant.evaluate(&input).unwrap()), vec![None; 4]);
    }

    #[test]
    fn integer_stack_partials_can_exceed_int64_without_corrupting_valid_frames() {
        let m = i64::MAX;
        let input = batch(
            vec![1; 6],
            vec![1, 2, 3, 4, 5, 6],
            vec![Some(-m), Some(m), Some(m), Some(-m), Some(-m), Some(m)],
        );
        let rows = window(
            WindowFrameUnits::Rows,
            WindowFrameBound::Preceding(Some(2)),
            WindowFrameBound::CurrentRow,
        );
        assert_eq!(
            values(&rows.evaluate(&input).unwrap()),
            vec![Some(-m), Some(0), Some(m), Some(m), Some(-m), Some(-m)]
        );
        let overflow = batch(vec![1; 2], vec![1, 2], vec![Some(m), Some(m)]);
        assert!(matches!(
            rows.evaluate(&overflow),
            Err(DataFusionError::Execution(_))
        ));
    }

    fn exact_rows_sum(array: ArrayRef, preceding: u64) -> Result<ArrayRef> {
        let data_type = array.data_type().clone();
        exact_rows_sum_as(array, preceding, data_type)
    }

    fn exact_rows_sum_as(
        array: ArrayRef,
        preceding: u64,
        output_type: DataType,
    ) -> Result<ArrayRef> {
        let data_type = array.data_type().clone();
        let input = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "value",
                data_type.clone(),
                true,
            )])),
            vec![array],
        )?;
        AggregateWindowExpr::new(
            Arc::new(Sum::new(
                Arc::new(Column::new("value", 0)),
                "SUM(value)",
                output_type,
            )),
            &[],
            &[],
            Some(WindowFrame {
                units: WindowFrameUnits::Rows,
                start_bound: WindowFrameBound::Preceding(Some(preceding)),
                end_bound: WindowFrameBound::CurrentRow,
            }),
        )
        .evaluate(&input)
    }

    #[test]
    fn exact_rows_decimal_stack_partials_exceed_i128_but_frames_still_fit() {
        let maximum = 10_i128.pow(38) - 1;
        let coefficients = [-maximum, maximum, maximum, -maximum, -maximum, maximum];
        let input = ScalarValue::iter_to_array(
            coefficients
                .iter()
                .map(|v| ScalarValue::Decimal128(Some(*v), 38, 6)),
        )
        .unwrap();
        let result = exact_rows_sum(input, 2).unwrap();
        assert_eq!(result.data_type(), &DataType::Decimal(38, 6));
        for (row, expected) in [-maximum, 0, maximum, maximum, -maximum, -maximum]
            .iter()
            .enumerate()
        {
            assert_eq!(
                ScalarValue::try_from_array(&result, row).unwrap(),
                ScalarValue::Decimal128(Some(*expected), 38, 6)
            );
        }
    }

    #[test]
    fn exact_rows_unsigned_true_overflow_is_a_typed_error_not_wrap_or_panic() {
        use arrow::array::UInt64Array;
        let input: ArrayRef =
            Arc::new(UInt64Array::from(vec![Some(u64::MAX), Some(u64::MAX)]));
        assert!(matches!(
            exact_rows_sum(input, 1),
            Err(DataFusionError::Execution(_))
        ));
    }

    #[test]
    fn exact_rows_decimal_precision_overflow_is_rejected_for_both_signs() {
        // Both totals fit i128, but exceed the declared Decimal(38,6).
        // Arrow storage capacity alone is not the result's precision contract.
        let maximum = 10_i128.pow(38) - 1;
        for sign in [-1, 1] {
            let input = ScalarValue::iter_to_array(
                [maximum * sign, sign]
                    .iter()
                    .map(|value| ScalarValue::Decimal128(Some(*value), 38, 6)),
            )
            .unwrap();
            assert!(matches!(
                exact_rows_sum(input, 1),
                Err(DataFusionError::Execution(_))
            ));
        }
    }

    #[test]
    fn exact_rows_unsigned_current_frame_preserves_maximum_and_nulls() {
        use arrow::array::UInt64Array;
        let input: ArrayRef = Arc::new(UInt64Array::from(vec![
            Some(u64::MAX),
            Some(u64::MAX),
            None,
            Some(0),
        ]));
        let result = exact_rows_sum(input, 0).unwrap();
        assert_eq!(result.data_type(), &DataType::UInt64);
        for (row, target) in [Some(u64::MAX), Some(u64::MAX), None, Some(0)]
            .iter()
            .enumerate()
        {
            assert_eq!(
                ScalarValue::try_from_array(&result, row).unwrap(),
                ScalarValue::UInt64(*target)
            );
        }
    }

    #[test]
    fn exact_rows_decimal_scale_widening_retains_fraction_and_typed_nulls() {
        let input = ScalarValue::iter_to_array(
            [None, Some(12345), Some(-5), None, None]
                .iter()
                .map(|value| ScalarValue::Decimal128(*value, 12, 2)),
        )
        .unwrap();
        let result = exact_rows_sum_as(input, 1, DataType::Decimal(38, 6)).unwrap();
        assert_eq!(result.data_type(), &DataType::Decimal(38, 6));
        for (row, target) in [None, Some(123450000), Some(123400000), Some(-50000), None]
            .iter()
            .enumerate()
        {
            assert_eq!(
                ScalarValue::try_from_array(&result, row).unwrap(),
                ScalarValue::Decimal128(*target, 38, 6)
            );
        }
        let empty = ScalarValue::Decimal128(None, 12, 2).to_array_of_size(0);
        let result = exact_rows_sum_as(empty, 1, DataType::Decimal(38, 6)).unwrap();
        assert_eq!(result.len(), 0);
        assert_eq!(result.data_type(), &DataType::Decimal(38, 6));
    }

    #[test]
    fn wide_rows_frame_uses_complete_input_without_quadratic_rescans() {
        let size = 60001;
        let input = batch(
            vec![1; size],
            (0..size as i64).collect(),
            vec![Some(1); size],
        );
        let rows = window(
            WindowFrameUnits::Rows,
            WindowFrameBound::Preceding(Some(30000)),
            WindowFrameBound::Following(Some(30000)),
        );
        let actual = values(&rows.evaluate(&input).unwrap());
        assert_eq!(actual.len(), size);
        assert_eq!(actual[0], Some(30001));
        assert_eq!(actual[30000], Some(60001));
        assert_eq!(actual[size - 1], Some(30001));
    }
}

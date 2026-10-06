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

use arrow::array::{Array, DecimalArray, Int64Array};
use arrow::datatypes::DataType;
use datafusion::prelude::SessionContext;

#[tokio::test]
async fn wide_integer_literals_preserve_exact_decimal_values() {
    for literal in [
        "9223372036854775808",
        "9223372036854775809",
        "-9223372036854775809",
        "99999999999999999999999999999999999999",
        "-99999999999999999999999999999999999999",
    ] {
        let ctx = SessionContext::new();
        let rows = ctx
            .sql(&format!("SELECT {literal} AS value"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let precision = literal.trim_start_matches('-').len();
        assert_eq!(
            rows[0].schema().field(0).data_type(),
            &DataType::Decimal(precision, 0)
        );
        let values = rows[0]
            .column(0)
            .as_any()
            .downcast_ref::<DecimalArray>()
            .unwrap();
        assert_eq!(
            values.value(0),
            literal.parse::<i128>().unwrap(),
            "{literal}"
        );
    }
}

#[tokio::test]
async fn int64_cast_boundaries_and_null_remain_valid() {
    for scale in [0, 9] {
        let ctx = SessionContext::new();
        let rows = ctx.sql(&format!(
            "SELECT CAST(CAST(9223372036854775807 AS DECIMAL(38,{scale})) AS BIGINT) AS upper_bound, \
             CAST(CAST(-9223372036854775808 AS DECIMAL(38,{scale})) AS BIGINT) AS lower_bound, \
             CAST(CAST(NULL AS DECIMAL(38,{scale})) AS BIGINT) AS missing"
        )).await.unwrap().collect().await.unwrap();
        for index in 0..3 {
            assert_eq!(rows[0].schema().field(index).data_type(), &DataType::Int64);
        }
        let upper = rows[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let lower = rows[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(upper.value(0), i64::MAX);
        assert_eq!(lower.value(0), i64::MIN);
        assert!(rows[0].column(2).is_null(0));
    }
}

#[tokio::test]
async fn int64_cast_overflow_rejects_both_signs_without_rounding() {
    for scale in [0, 9] {
        for literal in ["9223372036854775808", "-9223372036854775809"] {
            let ctx = SessionContext::new();
            let sql = format!(
                "SELECT CAST(CAST({literal} AS DECIMAL(38,{scale})) AS BIGINT) AS value"
            );
            let failure = match ctx.sql(&sql).await {
                Err(error) => error,
                Ok(frame) => frame.collect().await.expect_err(&sql),
            };
            let message = failure.to_string().to_lowercase();
            assert!(
                message.contains("out of range") || message.contains("overflow"),
                "{sql}: {message}"
            );
        }
    }
}

#[tokio::test]
async fn integer_literals_beyond_decimal_precision_are_rejected() {
    for literal in [
        "999999999999999999999999999999999999999",
        "-999999999999999999999999999999999999999",
    ] {
        let ctx = SessionContext::new();
        let result = ctx.sql(&format!("SELECT {literal}")).await;
        let failure = match result {
            Err(error) => error,
            Ok(_) => panic!("Out-of-domain integer {literal} was accepted"),
        };
        assert!(failure
            .to_string()
            .contains("exceeds the native decimal precision"));
    }
}

#[tokio::test]
async fn ordinary_integer_and_fractional_literal_types_are_preserved() {
    let ctx = SessionContext::new();
    let rows = ctx
        .sql("SELECT 42 AS integer_value, -42 AS negative_value, 1.5 AS fractional_value")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let schema = rows[0].schema();
    assert_eq!(schema.field(0).data_type(), &DataType::Int64);
    assert_eq!(schema.field(1).data_type(), &DataType::Int64);
    assert_eq!(schema.field(2).data_type(), &DataType::Float64);
}

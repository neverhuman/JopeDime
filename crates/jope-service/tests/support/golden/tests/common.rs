use super::*;
use jope_service::ranker::landmarks::{
    compare_sampled_to_full, summarize_latency, LandmarkOutput, BUILDING_BLOCK_WIDTH,
    INTERACTION_WIDTH, RANKER_COLUMN_COUNT, RANKER_COLUMN_WIDTH, RAW_COLUMN_WIDTH, REPORT_WIDTH,
    TABLE_WIDTH, VECTOR_WIDTH,
};
use safetensors::tensor::{serialize, Dtype, TensorView};
use std::{process::Command, time::Duration};
use tempfile::tempdir;

const TEST_PROBE_ENV: &str = "JOPE_GOLDEN_TEST_PROBE";

fn fixture_array<const N: usize>(fixture: &GoldenFixture, name: &str) -> [f32; N] {
    fixture.values_f32[name]
        .clone()
        .try_into()
        .unwrap_or_else(|_| panic!("fixture tensor {name} must contain {N} values"))
}

fn fixture_bytes() -> Vec<u8> {
    let values = [1.25_f32, -2.0, 3.5, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let extra_values = [0.5_f32]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    serialize(
        [
            (
                "input",
                TensorView::new(Dtype::F32, vec![2, 2], &values).unwrap(),
            ),
            (
                "extra",
                TensorView::new(Dtype::F32, vec![1], &extra_values).unwrap(),
            ),
        ],
        None,
    )
    .unwrap()
}

fn landmark_fixture_bytes() -> Vec<u8> {
    let input_x: Vec<f32> = (0..8).map(|value| value as f32).collect();
    let input_y = [-0.5_f32, 0.5, -1.0, 1.0];
    let input_row_ids = [10_i64, 11, 20, 21];
    let report: Vec<f32> = (0..REPORT_WIDTH).map(|value| value as f32).collect();
    let table: Vec<f32> = (0..TABLE_WIDTH)
        .map(|value| 1_000.0 + value as f32)
        .collect();
    let interactions: Vec<f32> = (0..INTERACTION_WIDTH)
        .map(|value| 2_000.0 + value as f32)
        .collect();
    let building_blocks: Vec<f32> = (0..BUILDING_BLOCK_WIDTH)
        .map(|value| 3_000.0 + value as f32)
        .collect();
    let col_full: Vec<f32> = (0..2 * RAW_COLUMN_WIDTH)
        .map(|value| 4_000.0 + value as f32)
        .collect();
    let expected_vector: Vec<f32> = report
        .iter()
        .chain(&table)
        .chain(&interactions)
        .chain(&building_blocks)
        .copied()
        .collect();
    let mut expected_columns = vec![0.0; RANKER_COLUMN_COUNT * RANKER_COLUMN_WIDTH];
    for row in 0..2 {
        for column in 0..RANKER_COLUMN_WIDTH {
            expected_columns[row * RANKER_COLUMN_WIDTH + column] =
                col_full[row * RAW_COLUMN_WIDTH + column];
        }
    }
    let sampled_rows = [2.0_f32];
    let full_rows = [4.0_f32];
    let latency_ms = [1.0_f32, 3.0, 2.0];

    let as_bytes = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let input_x = as_bytes(&input_x);
    let input_y = as_bytes(&input_y);
    let input_row_ids = input_row_ids
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let report = as_bytes(&report);
    let table = as_bytes(&table);
    let interactions = as_bytes(&interactions);
    let building_blocks = as_bytes(&building_blocks);
    let col_full = as_bytes(&col_full);
    let v256 = as_bytes(&expected_vector);
    let col = as_bytes(&expected_columns);
    let sampled_rows = as_bytes(&sampled_rows);
    let full_rows = as_bytes(&full_rows);
    let latency_ms = as_bytes(&latency_ms);

    serialize(
        [
            (
                "input.X",
                TensorView::new(Dtype::F32, vec![4, 2], &input_x).unwrap(),
            ),
            (
                "input.y",
                TensorView::new(Dtype::F32, vec![4], &input_y).unwrap(),
            ),
            (
                "input.row_ids",
                TensorView::new(Dtype::I64, vec![4], &input_row_ids).unwrap(),
            ),
            (
                "report",
                TensorView::new(Dtype::F32, vec![REPORT_WIDTH], &report).unwrap(),
            ),
            (
                "table",
                TensorView::new(Dtype::F32, vec![TABLE_WIDTH], &table).unwrap(),
            ),
            (
                "interactions",
                TensorView::new(Dtype::F32, vec![INTERACTION_WIDTH], &interactions).unwrap(),
            ),
            (
                "building_blocks",
                TensorView::new(Dtype::F32, vec![BUILDING_BLOCK_WIDTH], &building_blocks).unwrap(),
            ),
            (
                "col_full",
                TensorView::new(Dtype::F32, vec![2, RAW_COLUMN_WIDTH], &col_full).unwrap(),
            ),
            (
                "v256",
                TensorView::new(Dtype::F32, vec![1, VECTOR_WIDTH], &v256).unwrap(),
            ),
            (
                "col",
                TensorView::new(
                    Dtype::F32,
                    vec![1, RANKER_COLUMN_COUNT, RANKER_COLUMN_WIDTH],
                    &col,
                )
                .unwrap(),
            ),
            (
                "sampled_rows",
                TensorView::new(Dtype::F32, vec![1], &sampled_rows).unwrap(),
            ),
            (
                "full_rows",
                TensorView::new(Dtype::F32, vec![1], &full_rows).unwrap(),
            ),
            (
                "latency_ms",
                TensorView::new(Dtype::F32, vec![3], &latency_ms).unwrap(),
            ),
        ],
        None,
    )
    .unwrap()
}

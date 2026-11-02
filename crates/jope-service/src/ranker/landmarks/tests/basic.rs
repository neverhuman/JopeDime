use super::*;

#[test]
#[ignore = "requires an externally staged, hash-pinned sampled/full recipe capture"]
fn report_external_sampled_vs_full_capture() {
    use std::{env, fs};

    let capture_path = env::var("JOPE_RNK05_CAPTURE_JSON").unwrap();
    let report_path = env::var("JOPE_RNK05_REPORT_JSON").unwrap();
    let capture: serde_json::Value = serde_json::from_slice(&fs::read(capture_path).unwrap())
        .expect("decode sampled/full recipe capture");

    let decode_output = |name: &str| {
        let output = &capture["outputs"][name];
        let vector: Vec<f32> = serde_json::from_value(output["vector256"].clone()).unwrap();
        assert_eq!(vector.len(), VECTOR_WIDTH, "{name}: vector width");
        let columns: Vec<[f32; RAW_COLUMN_WIDTH]> =
            serde_json::from_value(output["col_full"].clone()).unwrap();
        assert!(
            !columns.is_empty(),
            "{name}: raw column output must be present"
        );

        let report: [f32; REPORT_WIDTH] = vector[..REPORT_WIDTH].try_into().unwrap();
        let table: [f32; TABLE_WIDTH] = vector[REPORT_WIDTH..REPORT_WIDTH + TABLE_WIDTH]
            .try_into()
            .unwrap();
        let interactions: [f32; INTERACTION_WIDTH] = vector
            [REPORT_WIDTH + TABLE_WIDTH..REPORT_WIDTH + TABLE_WIDTH + INTERACTION_WIDTH]
            .try_into()
            .unwrap();
        let building_blocks: [f32; BUILDING_BLOCK_WIDTH] = vector
            [REPORT_WIDTH + TABLE_WIDTH + INTERACTION_WIDTH..]
            .try_into()
            .unwrap();
        let rows = output["rows"].as_u64().unwrap() as usize;

        (
            LandmarkOutput::from_descriptor_parts(
                &report,
                &table,
                &interactions,
                &building_blocks,
                &columns,
            ),
            rows,
        )
    };

    let (sampled, sampled_rows) = decode_output("sampled");
    let (full, full_rows) = decode_output("full");
    let sampled_recipe_elapsed_ms = capture["outputs"]["sampled"]["elapsed_ns"]
        .as_u64()
        .unwrap() as f64
        / 1_000_000.0;
    let full_recipe_elapsed_ms =
        capture["outputs"]["full"]["elapsed_ns"].as_u64().unwrap() as f64 / 1_000_000.0;
    assert!(sampled_rows <= 1_024);
    assert!(full_rows > sampled_rows);
    let summary = compare_sampled_to_full(&sampled, &full, sampled_rows, full_rows).unwrap();
    let report = serde_json::json!({
        "schema": "rnk05.sampled-full-agreement/v1",
        "dataset_id": capture["dataset_id"],
        "training_sha256": capture["training_sha256"],
        "sample_row_indices_sha256": capture["sample_row_indices_sha256"],
        "sample_method": capture["sample_method"],
        "sample_seed": capture["sample_seed"],
        "sampled_recipe_elapsed_ms": sampled_recipe_elapsed_ms,
        "full_recipe_elapsed_ms": full_recipe_elapsed_ms,
        "sampled_rows": summary.sampled_rows,
        "full_rows": summary.full_rows,
        "values_compared": summary.values_compared,
        "exact_matches": summary.exact_matches,
        "max_abs_error": summary.max_abs_error,
        "mean_abs_error": summary.mean_abs_error,
    });
    fs::write(report_path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
}

#[test]
fn vector_blocks_are_concatenated_in_training_order() {
    let report = std::array::from_fn(|i| i as f32);
    let table = std::array::from_fn(|i| (REPORT_WIDTH + i) as f32);
    let interactions = std::array::from_fn(|i| (REPORT_WIDTH + TABLE_WIDTH + i) as f32);
    let building_blocks =
        std::array::from_fn(|i| (REPORT_WIDTH + TABLE_WIDTH + INTERACTION_WIDTH + i) as f32);

    let vector = assemble_vector(&report, &table, &interactions, &building_blocks);

    assert_eq!(vector.len(), VECTOR_WIDTH);
    assert_eq!(vector[0], 0.0);
    assert_eq!(vector[REPORT_WIDTH - 1], (REPORT_WIDTH - 1) as f32);
    assert_eq!(vector[REPORT_WIDTH], REPORT_WIDTH as f32);
    assert_eq!(
        vector[REPORT_WIDTH + TABLE_WIDTH],
        (REPORT_WIDTH + TABLE_WIDTH) as f32
    );
    assert_eq!(vector[VECTOR_WIDTH - 1], (VECTOR_WIDTH - 1) as f32);
}

#[test]
fn columns_keep_first_17_values_and_zero_pad_or_truncate_to_64() {
    let raw: Vec<[f32; RAW_COLUMN_WIDTH]> = (0..RANKER_COLUMN_COUNT + 1)
        .map(|row| std::array::from_fn(|column| (row * 100 + column) as f32))
        .collect();

    let columns = pad_ranker_columns(&raw);

    assert_eq!(columns.len(), RANKER_COLUMN_COUNT);
    assert_eq!(columns[0].len(), RANKER_COLUMN_WIDTH);
    assert_eq!(columns[0][0], 0.0);
    assert_eq!(columns[0][16], 16.0);
    assert_eq!(columns[63][0], 6300.0);
    assert_eq!(columns[63][16], 6316.0);

    let short = pad_ranker_columns(&raw[..1]);
    assert_eq!(short[0][16], 16.0);
    assert_eq!(short[1], [0.0; RANKER_COLUMN_WIDTH]);
    assert_eq!(short[63], [0.0; RANKER_COLUMN_WIDTH]);
}

#[test]
fn non_finite_descriptor_values_are_zeroed() {
    let mut report = [1.0; REPORT_WIDTH];
    report[0] = f32::NAN;
    let mut table = [2.0; TABLE_WIDTH];
    table[TABLE_WIDTH - 1] = f32::INFINITY;
    let interactions = [3.0; INTERACTION_WIDTH];
    let building_blocks = [4.0; BUILDING_BLOCK_WIDTH];

    let vector = assemble_vector(&report, &table, &interactions, &building_blocks);
    assert_eq!(vector[0], 0.0);
    assert_eq!(vector[REPORT_WIDTH + TABLE_WIDTH - 1], 0.0);

    let mut raw = [[5.0; RAW_COLUMN_WIDTH]; 1];
    raw[0][16] = f32::NEG_INFINITY;
    let columns = pad_ranker_columns(&raw);
    assert_eq!(columns[0][16], 0.0);
    assert_eq!(columns[1], [0.0; RANKER_COLUMN_WIDTH]);
}

fn output_with_report_value(value: f32) -> LandmarkOutput {
    let mut report = [0.0; REPORT_WIDTH];
    report[0] = value;
    LandmarkOutput::from_descriptor_parts(
        &report,
        &[0.0; TABLE_WIDTH],
        &[0.0; INTERACTION_WIDTH],
        &[0.0; BUILDING_BLOCK_WIDTH],
        &[],
    )
}

#[test]
fn table_agreement_reports_full_shape_and_absolute_error_summary() {
    let sampled = output_with_report_value(2.0);
    let full = output_with_report_value(3.0);

    let summary = compare_sampled_to_full(&sampled, &full, 12, 100).unwrap();

    assert_eq!(
        summary,
        AgreementSummary {
            sampled_rows: 12,
            full_rows: 100,
            values_compared: VECTOR_WIDTH + RANKER_COLUMN_COUNT * RANKER_COLUMN_WIDTH,
            exact_matches: VECTOR_WIDTH + RANKER_COLUMN_COUNT * RANKER_COLUMN_WIDTH - 1,
            max_abs_error: 1.0,
            mean_abs_error: 1.0 / (VECTOR_WIDTH + RANKER_COLUMN_COUNT * RANKER_COLUMN_WIDTH) as f64,
        }
    );
}

#[test]
fn table_agreement_rejects_invalid_fit_row_counts() {
    let output = output_with_report_value(0.0);

    assert_eq!(
        compare_sampled_to_full(&output, &output, 0, 1),
        Err(AgreementError::EmptyTable)
    );
    assert_eq!(
        compare_sampled_to_full(&output, &output, 1_025, 2_000),
        Err(AgreementError::SampleOverLimit)
    );
    assert_eq!(
        compare_sampled_to_full(&output, &output, 4, 3),
        Err(AgreementError::SampleExceedsFullTable)
    );
}

#[test]
fn latency_summary_reports_median_and_range_in_milliseconds() {
    use std::time::Duration;

    assert_eq!(summarize_latency(&[]), None);
    assert_eq!(
        summarize_latency(&[
            Duration::from_millis(3),
            Duration::from_millis(1),
            Duration::from_millis(2),
        ]),
        Some(super::LatencySummary {
            samples: 3,
            min_ms: 1.0,
            median_ms: 2.0,
            max_ms: 3.0,
        })
    );
}

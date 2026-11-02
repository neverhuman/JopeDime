//! CPU implementation of the high-resolution landmark recipe.
//!
//! The FIT transform and descriptor extraction port the pinned
//! `FusedLandmarkWorkspace.fit_transform_and_extract_bundle_high` path into
//! Rust, including its 1,024-row FIT cap. The public builder returns the
//! 256-value vector and raw 24-column descriptors; ranker-facing columns are
//! projected into the existing 64-by-17 shape. Numeric feature scaling is
//! regression-only, matching the source recipe's numeric input contract.
#[path = "landmarks/mod.rs"]
mod recipe;
pub use recipe::{fit_transform_and_extract_bundle_high, RecipeError};

pub const VECTOR_WIDTH: usize = 256;
pub const REPORT_WIDTH: usize = 116;
pub const TABLE_WIDTH: usize = 24;
pub const INTERACTION_WIDTH: usize = 48;
pub const BUILDING_BLOCK_WIDTH: usize = 68;
pub const RAW_COLUMN_WIDTH: usize = 24;
pub const RANKER_COLUMN_WIDTH: usize = 17;
pub const RANKER_COLUMN_COUNT: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct LandmarkOutput {
    vector: [f32; VECTOR_WIDTH],
    columns: [[f32; RANKER_COLUMN_WIDTH]; RANKER_COLUMN_COUNT],
    raw_columns: Vec<[f32; RAW_COLUMN_WIDTH]>,
}

impl LandmarkOutput {
    pub fn from_descriptor_parts(
        report: &[f32; REPORT_WIDTH],
        table: &[f32; TABLE_WIDTH],
        interactions: &[f32; INTERACTION_WIDTH],
        building_blocks: &[f32; BUILDING_BLOCK_WIDTH],
        raw_columns: &[[f32; RAW_COLUMN_WIDTH]],
    ) -> Self {
        Self {
            vector: assemble_vector(report, table, interactions, building_blocks),
            columns: pad_ranker_columns(raw_columns),
            raw_columns: raw_columns.to_vec(),
        }
    }

    pub fn vector(&self) -> &[f32; VECTOR_WIDTH] {
        &self.vector
    }

    pub fn columns(&self) -> &[[f32; RANKER_COLUMN_WIDTH]; RANKER_COLUMN_COUNT] {
        &self.columns
    }

    pub fn raw_columns(&self) -> &[[f32; RAW_COLUMN_WIDTH]] {
        &self.raw_columns
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AgreementSummary {
    pub sampled_rows: usize,
    pub full_rows: usize,
    pub values_compared: usize,
    pub exact_matches: usize,
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgreementError {
    EmptyTable,
    SampleOverLimit,
    SampleExceedsFullTable,
}

impl std::fmt::Display for AgreementError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTable => formatter.write_str("landmark tables must contain FIT rows"),
            Self::SampleOverLimit => {
                formatter.write_str("landmark FIT sample exceeds the 1,024-row cap")
            }
            Self::SampleExceedsFullTable => {
                formatter.write_str("landmark FIT sample exceeds the full-table row count")
            }
        }
    }
}

impl std::error::Error for AgreementError {}

/// Summarize sampled-versus-full outputs without imposing a pass threshold.
/// T099 requires reporting agreement; the qualification owner sets the
/// acceptance interpretation when RES-09 fixtures and approved inputs land.
pub fn compare_sampled_to_full(
    sampled: &LandmarkOutput,
    full: &LandmarkOutput,
    sampled_rows: usize,
    full_rows: usize,
) -> Result<AgreementSummary, AgreementError> {
    if sampled_rows == 0 || full_rows == 0 {
        return Err(AgreementError::EmptyTable);
    }
    if sampled_rows > 1_024 {
        return Err(AgreementError::SampleOverLimit);
    }
    if sampled_rows > full_rows {
        return Err(AgreementError::SampleExceedsFullTable);
    }

    let mut values_compared = 0;
    let mut exact_matches = 0;
    let mut max_abs_error = 0.0_f64;
    let mut total_abs_error = 0.0_f64;

    for (sampled_value, full_value) in sampled
        .vector
        .iter()
        .chain(sampled.columns.iter().flatten())
        .zip(full.vector.iter().chain(full.columns.iter().flatten()))
    {
        let abs_error = f64::from((*sampled_value - *full_value).abs());
        values_compared += 1;
        exact_matches += usize::from(abs_error == 0.0);
        max_abs_error = max_abs_error.max(abs_error);
        total_abs_error += abs_error;
    }

    Ok(AgreementSummary {
        sampled_rows,
        full_rows,
        values_compared,
        exact_matches,
        max_abs_error,
        mean_abs_error: total_abs_error / values_compared as f64,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatencySummary {
    pub samples: usize,
    pub min_ms: f64,
    pub median_ms: f64,
    pub max_ms: f64,
}

/// Summarize repeated `landmarks.ms` measurements for a pre-stage report.
pub fn summarize_latency(samples: &[std::time::Duration]) -> Option<LatencySummary> {
    if samples.is_empty() {
        return None;
    }

    let mut millis = samples
        .iter()
        .map(std::time::Duration::as_secs_f64)
        .map(|seconds| seconds * 1_000.0)
        .collect::<Vec<_>>();
    millis.sort_by(f64::total_cmp);

    let middle = millis.len() / 2;
    let median_ms = if millis.len() % 2 == 0 {
        (millis[middle - 1] + millis[middle]) / 2.0
    } else {
        millis[middle]
    };

    Some(LatencySummary {
        samples: millis.len(),
        min_ms: millis[0],
        median_ms,
        max_ms: millis[millis.len() - 1],
    })
}

/// Join the four high-level descriptor blocks in the training order.
///
/// The source concatenates `vector116`, `table_vec24`, `inter_vec48`, and
/// `building_block_vec68`, then replaces NaN and infinities with zero.
pub fn assemble_vector(
    report: &[f32; REPORT_WIDTH],
    table: &[f32; TABLE_WIDTH],
    interactions: &[f32; INTERACTION_WIDTH],
    building_blocks: &[f32; BUILDING_BLOCK_WIDTH],
) -> [f32; VECTOR_WIDTH] {
    let mut result = [0.0; VECTOR_WIDTH];

    for (offset, value) in report
        .iter()
        .chain(table)
        .chain(interactions)
        .chain(building_blocks)
        .enumerate()
    {
        result[offset] = finite_or_zero(*value);
    }

    result
}

/// Adapt raw 24-wide per-column landmarks to the ranker's `[64, 17]` input.
///
/// The training consumer copies the first 17 values for the first 64 columns,
/// truncates excess columns, zero-pads missing columns, and sanitizes
/// non-finite values to zero.
pub fn pad_ranker_columns(
    raw_columns: &[[f32; RAW_COLUMN_WIDTH]],
) -> [[f32; RANKER_COLUMN_WIDTH]; RANKER_COLUMN_COUNT] {
    let mut result = [[0.0; RANKER_COLUMN_WIDTH]; RANKER_COLUMN_COUNT];

    for (target, source) in result.iter_mut().zip(raw_columns.iter()) {
        for (target_value, source_value) in target.iter_mut().zip(source.iter()) {
            *target_value = finite_or_zero(*source_value);
        }
    }

    result
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        assemble_vector, build_high_full_table, compare_sampled_to_full,
        fit_transform_and_extract_bundle_high, pad_ranker_columns, summarize_latency,
        AgreementError, AgreementSummary, LandmarkOutput, BUILDING_BLOCK_WIDTH, INTERACTION_WIDTH,
        RANKER_COLUMN_COUNT, RANKER_COLUMN_WIDTH, RAW_COLUMN_WIDTH, REPORT_WIDTH, TABLE_WIDTH,
        VECTOR_WIDTH,
    };

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
                mean_abs_error: 1.0
                    / (VECTOR_WIDTH + RANKER_COLUMN_COUNT * RANKER_COLUMN_WIDTH) as f64,
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

    #[test]
    #[ignore = "requires externally staged, hash-pinned RES-07 CPU fixtures and bundle"]
    fn matches_res07_real_landmark_output_assembly() {
        use super::super::adapter::real_cpu_parity::Fixtures;

        // PROVISIONAL, WATCHER-SET; owner may override. This reuses the frozen
        // RES-07/RNK-12 CPU parity arithmetic, not an owner-approved RNK-05 rule.
        const ATOL: f64 = 1e-5;
        const RTOL: f64 = 1e-4;

        fn assert_matches_golden(context: &str, name: &str, actual: &[f32], expected: &[f32]) {
            assert_eq!(actual.len(), expected.len(), "{context}: {name} length");
            for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                let actual = f64::from(actual);
                let expected = f64::from(expected);
                assert!(
                    actual.is_finite() && expected.is_finite(),
                    "{context}: {name}[{index}] must be finite"
                );
                let tolerance = ATOL + RTOL * expected.abs();
                let error = (actual - expected).abs();
                assert!(
                    error <= tolerance,
                    "{context}: {name}[{index}] actual={actual}, expected={expected}, error={error}, tolerance={tolerance}"
                );
            }
        }

        let fixtures = Fixtures::open();
        assert_eq!(fixtures.contexts.len(), 50);

        for context in &fixtures.contexts {
            let name = format!("landmarks/{context}.safetensors");
            let fixture = fixtures.load(&name);
            let schema = &fixtures.index["files"][&name]["tensors"];
            let input_shape = schema["input.X"]["shape"].as_array().unwrap();
            let fit_rows = input_shape[0].as_u64().unwrap() as usize;
            let feature_count = input_shape[1].as_u64().unwrap() as usize;

            assert!((1..=1_024).contains(&fit_rows), "{context}: FIT row cap");
            assert_eq!(
                schema["input.row_ids"]["shape"][0].as_u64().unwrap() as usize,
                fit_rows,
                "{context}: source row IDs must cover the FIT table"
            );
            assert_eq!(
                schema["input.y"]["shape"][0].as_u64().unwrap() as usize,
                fit_rows,
                "{context}: target rows must match the FIT table"
            );
            assert_eq!(
                schema["col_full"]["shape"][0].as_u64().unwrap() as usize,
                feature_count,
                "{context}: raw column output count must match feature count"
            );

            let vector = fixture.f32("v256");
            assert_eq!(vector.len(), VECTOR_WIDTH, "{context}: v256 width");
            let report: [f32; REPORT_WIDTH] = vector[..REPORT_WIDTH].try_into().unwrap();
            let table: [f32; TABLE_WIDTH] = vector[REPORT_WIDTH..REPORT_WIDTH + TABLE_WIDTH]
                .try_into()
                .unwrap();
            let interaction_start = REPORT_WIDTH + TABLE_WIDTH;
            let interactions: [f32; INTERACTION_WIDTH] = vector
                [interaction_start..interaction_start + INTERACTION_WIDTH]
                .try_into()
                .unwrap();
            let building_block_start = interaction_start + INTERACTION_WIDTH;
            let building_blocks: [f32; BUILDING_BLOCK_WIDTH] =
                vector[building_block_start..].try_into().unwrap();

            let raw_column_values = fixture.f32("col_full");
            assert_eq!(
                raw_column_values.len() % RAW_COLUMN_WIDTH,
                0,
                "{context}: raw column width"
            );
            let raw_columns = raw_column_values
                .chunks_exact(RAW_COLUMN_WIDTH)
                .map(|row| row.try_into().unwrap())
                .collect::<Vec<[f32; RAW_COLUMN_WIDTH]>>();

            let actual = LandmarkOutput::from_descriptor_parts(
                &report,
                &table,
                &interactions,
                &building_blocks,
                &raw_columns,
            );
            assert_matches_golden(context, "v256", actual.vector().as_slice(), &vector);

            let expected_columns = fixture.f32("col");
            let actual_columns = actual
                .columns()
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            assert_matches_golden(context, "col", &actual_columns, &expected_columns);
        }
    }

    #[test]
    #[ignore = "requires hash-pinned RES-07 inputs and captured full-recipe outputs"]
    fn matches_res07_full_recipe_outputs_and_measures_rust_latency() {
        use std::time::Instant;

        const ATOL: f64 = 1e-5;
        const RTOL: f64 = 1e-4;
        fn check(context: &str, name: &str, got: &[f32], expected: &[f32]) {
            assert_eq!(got.len(), expected.len(), "{context}: {name} shape");
            for (i, (&a, &g)) in got.iter().zip(expected).enumerate() {
                assert!(
                    a.is_finite() && g.is_finite(),
                    "{context}: {name}[{i}] nonfinite"
                );
                let error = (f64::from(a) - f64::from(g)).abs();
                let tolerance = ATOL + RTOL * f64::from(g).abs();
                assert!(error <= tolerance, "{context}: {name}[{i}] actual={a} golden={g} error={error} tolerance={tolerance}");
            }
        }

        let fixtures = super::super::adapter::real_cpu_parity::Fixtures::open();
        assert_eq!(fixtures.contexts.len(), 50);
        let mut elapsed = Vec::with_capacity(fixtures.contexts.len());
        let mut v_compared = 0;
        let mut raw_compared = 0;
        let mut ranker_compared = 0;
        let mut max_error = 0.0_f64;
        for context in &fixtures.contexts {
            let name = format!("landmarks/{context}.safetensors");
            let fixture = fixtures.load(&name);
            let schema = &fixtures.index["files"][&name]["tensors"];
            let shape = schema["input.X"]["shape"].as_array().unwrap();
            let rows = shape[0].as_u64().unwrap() as usize;
            let features = shape[1].as_u64().unwrap() as usize;
            assert!((1..=1_024).contains(&rows), "{context}: FIT row cap");
            let x = fixture.f32("input.X");
            let y = fixture.f32("input.y");
            let start = Instant::now();
            let got = fit_transform_and_extract_bundle_high(&x, rows, features, &y)
                .unwrap_or_else(|error| panic!("{context}: full Rust recipe failed: {error}"));
            elapsed.push(start.elapsed());

            let golden_vector = fixture.f32("v256");
            let raw_golden = fixture.f32("col_full");
            let raw_actual: Vec<f32> = got.raw_columns().iter().flatten().copied().collect();
            let ranker_golden = fixture.f32("col");
            let ranker_actual: Vec<f32> = got.columns().iter().flatten().copied().collect();
            check(context, "v256", got.vector(), &golden_vector);
            check(context, "col_full", &raw_actual, &raw_golden);
            check(context, "col", &ranker_actual, &ranker_golden);
            v_compared += golden_vector.len();
            raw_compared += raw_golden.len();
            ranker_compared += ranker_golden.len();
            for (a, g) in got
                .vector()
                .iter()
                .zip(&golden_vector)
                .chain(raw_actual.iter().zip(&raw_golden))
                .chain(ranker_actual.iter().zip(&ranker_golden))
            {
                max_error = max_error.max((f64::from(*a) - f64::from(*g)).abs());
            }
        }
        let latency = summarize_latency(&elapsed).unwrap();
        eprintln!("RNK-05 Rust full-recipe parity: contexts={} v256={} col_full={} col={} misses=0 max_abs_error={max_error:.9} landmarks.ms samples={} min={:.6} median={:.6} max={:.6}", fixtures.contexts.len(), v_compared, raw_compared, ranker_compared, latency.samples, latency.min_ms, latency.median_ms, latency.max_ms);
    }

    #[test]
    #[ignore = "requires hash-pinned RES-07 sampled/full training capture"]
    fn reports_res07_sampled_full_rust_agreement() {
        use sha2::{Digest, Sha256};
        use std::{env, fs, time::Instant};

        const ATOL: f64 = 1e-5;
        const RTOL: f64 = 1e-4;
        let train_path = env::var("JOPE_RNK05_TRAIN_CSV").unwrap();
        let indices_path = env::var("JOPE_RNK05_SAMPLE_INDICES").unwrap();
        let capture_path = env::var("JOPE_RNK05_CAPTURE_JSON").unwrap();
        let report_path = env::var("JOPE_RNK05_REPORT_JSON").unwrap();
        let train_bytes = fs::read(train_path).unwrap();
        let indices_bytes = fs::read(indices_path).unwrap();
        let capture: serde_json::Value =
            serde_json::from_slice(&fs::read(capture_path).unwrap()).unwrap();
        let train_sha = format!("{:x}", Sha256::digest(&train_bytes));
        let indices_sha = format!("{:x}", Sha256::digest(&indices_bytes));
        assert_eq!(train_sha, capture["train_file_sha256"]);
        assert_eq!(indices_sha, capture["sample_row_indices_sha256"]);
        assert_eq!(capture["schema"], "rnk05.sampled-full-recipe-capture/v1");

        let rows: Vec<Vec<f32>> = String::from_utf8(train_bytes)
            .unwrap()
            .lines()
            .map(|line| {
                line.split(',')
                    .map(|value| value.parse::<f32>().unwrap())
                    .collect()
            })
            .collect();
        let full_rows = rows.len();
        let features = rows[0].len() - 1;
        assert_eq!(
            full_rows,
            capture["training_rows"].as_u64().unwrap() as usize
        );
        assert_eq!(
            features + 1,
            capture["training_columns_including_target"]
                .as_u64()
                .unwrap() as usize
        );
        assert_eq!(indices_bytes.len() % 4, 0);
        let indices: Vec<usize> = indices_bytes
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()) as usize)
            .collect();
        assert_eq!(
            indices.len(),
            capture["sample_rows"].as_u64().unwrap() as usize
        );
        assert!(indices.iter().all(|&index| index < full_rows));
        let flatten_x = |selected: &[usize]| {
            selected
                .iter()
                .flat_map(|&i| rows[i][..features].iter().copied())
                .collect::<Vec<_>>()
        };
        let flatten_y = |selected: &[usize]| {
            selected
                .iter()
                .map(|&i| rows[i][features])
                .collect::<Vec<_>>()
        };
        let all_indices: Vec<usize> = (0..full_rows).collect();
        let sampled_x = flatten_x(&indices);
        let sampled_y = flatten_y(&indices);
        let full_x = flatten_x(&all_indices);
        let full_y = flatten_y(&all_indices);

        let started = Instant::now();
        let sampled =
            fit_transform_and_extract_bundle_high(&sampled_x, indices.len(), features, &sampled_y)
                .unwrap();
        let sampled_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let started = Instant::now();
        let full = build_high_full_table(&full_x, full_rows, features, &full_y).unwrap();
        let full_ms = started.elapsed().as_secs_f64() * 1_000.0;
        assert!(indices.len() <= 1_024 && full_rows > indices.len());

        let check_golden = |label: &str, got: &[f32], expected: &[f32]| {
            assert_eq!(got.len(), expected.len(), "{label} shape");
            for (index, (&actual, &golden)) in got.iter().zip(expected).enumerate() {
                assert!(
                    actual.is_finite() && golden.is_finite(),
                    "{label}[{index}] nonfinite"
                );
                let error = (f64::from(actual) - f64::from(golden)).abs();
                assert!(
                    error <= ATOL + RTOL * f64::from(golden).abs(),
                    "{label}[{index}] actual={actual} golden={golden} error={error}"
                );
            }
        };
        for (name, output) in [("sampled", &sampled), ("full", &full)] {
            let expected_vector: Vec<f32> =
                serde_json::from_value(capture["outputs"][name]["vector256"].clone()).unwrap();
            let expected_columns: Vec<Vec<f32>> =
                serde_json::from_value(capture["outputs"][name]["col_full"].clone()).unwrap();
            let expected_flat: Vec<f32> = expected_columns.into_iter().flatten().collect();
            let actual_columns: Vec<f32> = output.raw_columns().iter().flatten().copied().collect();
            check_golden(&format!("{name}.v256"), output.vector(), &expected_vector);
            check_golden(&format!("{name}.col_full"), &actual_columns, &expected_flat);
        }
        let agreement = compare_sampled_to_full(&sampled, &full, indices.len(), full_rows).unwrap();
        let report = serde_json::json!({
            "schema": "rnk05.sampled-full-rust-agreement/v1",
            "dataset_id": capture["dataset_id"],
            "dataset_canonical_sha256": capture["dataset_canonical_sha256"],
            "training_sha256": train_sha,
            "sample_row_indices_sha256": indices_sha,
            "sample_method": capture["sample_method"],
            "sample_seed": capture["sample_seed"],
            "sampled_rows": agreement.sampled_rows,
            "full_rows": agreement.full_rows,
            "values_compared": agreement.values_compared,
            "exact_matches": agreement.exact_matches,
            "max_abs_error": agreement.max_abs_error,
            "mean_abs_error": agreement.mean_abs_error,
            "sampled_landmarks_ms": sampled_ms,
            "full_landmarks_ms": full_ms,
            "golden_tolerance": "finite; abs(actual-golden) <= 1e-5 + 1e-4*abs(golden)",
            "golden_misses": 0
        });
        fs::write(report_path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
}

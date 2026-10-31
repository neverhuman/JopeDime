//! Pre-stage assembly for the high-resolution landmark recipe.
//!
//! This module ports the source-backed final assembly from the pinned
//! `FusedLandmarkWorkspace` consumer. Feature extraction itself depends on
//! native HPC kernels and RES-09 golden fixtures; this file does not claim
//! full recipe parity.

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
        }
    }

    pub fn vector(&self) -> &[f32; VECTOR_WIDTH] {
        &self.vector
    }

    pub fn columns(&self) -> &[[f32; RANKER_COLUMN_WIDTH]; RANKER_COLUMN_COUNT] {
        &self.columns
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
        assemble_vector, compare_sampled_to_full, pad_ranker_columns, summarize_latency,
        AgreementError, AgreementSummary, LandmarkOutput, BUILDING_BLOCK_WIDTH, INTERACTION_WIDTH,
        RANKER_COLUMN_COUNT, RANKER_COLUMN_WIDTH, RAW_COLUMN_WIDTH, REPORT_WIDTH, TABLE_WIDTH,
        VECTOR_WIDTH,
    };

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
            assert_eq!(actual.vector().as_slice(), vector, "{context}: v256 order");

            let expected_columns = fixture.f32("col");
            let actual_columns = actual
                .columns()
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            assert_eq!(actual_columns, expected_columns, "{context}: col adapter");
        }
    }

    #[test]
    #[ignore = "needs RES-09 full-high/v1 golden fixtures before recipe parity qualification"]
    fn matches_res09_full_high_landmark_goldens() {
        // RES-09 has not landed its named input/output fixture tensors yet.
        // Do not invent full-recipe expected values here.
        panic!("load RES-09 fixtures and assert full-recipe parity before unignoring");
    }
}

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
    let mut offset = 0;

    for value in report
        .iter()
        .chain(table)
        .chain(interactions)
        .chain(building_blocks)
    {
        result[offset] = finite_or_zero(*value);
        offset += 1;
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
        assemble_vector, pad_ranker_columns, BUILDING_BLOCK_WIDTH, INTERACTION_WIDTH,
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

    #[test]
    #[ignore = "needs RES-09 full-high/v1 golden fixtures before recipe parity qualification"]
    fn matches_res09_full_high_landmark_goldens() {
        // RES-09 has not landed its named input/output fixture tensors yet.
        // Do not invent full-recipe expected values here.
        panic!("load RES-09 fixtures and assert full-recipe parity before unignoring");
    }
}

use super::recipe::build_high_full_table;
use super::{
    assemble_vector, compare_sampled_to_full, fit_transform_and_extract_bundle_high,
    pad_ranker_columns, summarize_latency, AgreementError, AgreementSummary, LandmarkOutput,
    LatencySummary, BUILDING_BLOCK_WIDTH, INTERACTION_WIDTH, RANKER_COLUMN_COUNT,
    RANKER_COLUMN_WIDTH, RAW_COLUMN_WIDTH, REPORT_WIDTH, TABLE_WIDTH, VECTOR_WIDTH,
};

#[path = "tests/basic.rs"]
mod basic;
#[path = "tests/full.rs"]
mod full;
#[path = "tests/golden.rs"]
mod golden;

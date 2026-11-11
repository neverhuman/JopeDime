//! Re-export of the pure-Rust CPU landmark conditioning contract from `jope-prep`.
//!
//! Moving the numeric-regression `high` landmark recipe into `jope-prep` lets
//! native CLI, WASM clients, and the engine share identical deterministic FIT
//! conditioning without linking engine dependencies into thin clients.

pub use jope_prep::landmarks::{
    assemble_vector, compare_sampled_to_full, fit_transform_and_extract_bundle_high,
    pad_ranker_columns, summarize_latency, AgreementError, AgreementSummary, LandmarkOutput,
    LatencySummary, RecipeError, BUILDING_BLOCK_WIDTH, INTERACTION_WIDTH, RANKER_COLUMN_COUNT,
    RANKER_COLUMN_WIDTH, RAW_COLUMN_WIDTH, REPORT_WIDTH, TABLE_WIDTH, VECTOR_WIDTH,
};

#[cfg(test)]
mod recipe {
    pub(super) use jope_prep::landmarks::build_high_full_table;
}

#[cfg(test)]
#[path = "landmarks/tests.rs"]
mod tests;

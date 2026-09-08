//! Split-robust generalized GP campaign.
//!
//! This is an experimental, additive runner.  It deliberately does not share
//! mutable state, identities, or production pointers with Search V3 or the
//! legacy GP bridge.

mod catalog;
mod contracts;
mod evaluator;
mod evolution;
mod pilot;
mod splits;
mod store;

pub use catalog::{
    build_tranche_plan, eligible_real_datasets, freeze_catalog, select_max_min_pilot,
};
pub use contracts::*;
pub use evaluator::{
    CorpusEvalAdapter, DatasetRowV1, DatasetTableV1, EvaluatorOutcomeV1, SealedTestTableV1,
};
pub use evolution::{
    build_generation_zero, build_global_gene_ranking, build_next_generation,
    build_provisional_archive, confidence_score, rank_population, select_cv7_winner,
};
pub use pilot::{compare_pilot, deterministic_bootstrap_lower_bound};
pub use splits::{build_cv7_plan, build_rotating_split_plan};
pub use store::{ArtifactStoreV1, PredictionCacheV1};

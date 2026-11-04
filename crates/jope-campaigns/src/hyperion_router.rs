//! Train-only authority for the actual Hyperion two-stream GP router.
//!
//! This module is deliberately additive. `hyperion_gate` and the V1 budget
//! campaign remain replay authorities for their historical receipts; neither
//! is reinterpreted after the old twenty-dataset cohort was observed.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::search_v3::{canonical_sha256, validate_sha256};

// Public V2 contracts live in a task-neutral module.  Re-export them here so
// existing Hyperion callers can adopt the per-family API without changing
// their module import path.
pub use crate::router_contracts::{
    build_validation_folds_v2, mase_v2, predictor_columns_without_time,
    validate_family_ensemble_v2, FamilyRouteDecisionV2, FeatureViewV2, MemberInputPlanV2,
    ModelArtifactV4, PredictionBatchV2, PredictorRouteV2, RunManifestV6, SeedRouteMetricV2,
    StrategyScreenReceiptV3, ValidationFoldV2, ValidationSchemeV2,
};

pub const EXPLORATORY_CAMPAIGN_ID: &str =
    "hyperion-two-stream-current20-train-adaptive-exploratory-v2";
pub const CONFIRMATORY_CAMPAIGN_ID: &str = "hyperion-two-stream-fresh20-confirmatory-v1";
pub const ROUTER_SEEDS: [u64; 2] = [303, 404];
pub const OUTER_FOLDS: usize = 7;
pub const INNER_FOLDS: usize = 2;
pub const SEARCH_EVALUATIONS_PER_STREAM: u32 = 2_000;
pub const FIXED_BANK_SIZE: usize = 12;
pub const DEVELOPMENT_DATASET_COUNT: usize = 22;
pub const ROUTER_DATASET_COUNT: usize = 20;
pub const MINIMUM_FIT_ROWS: usize = 32;
pub const MINIMUM_HOLD_ROWS: usize = 2;
pub const MINIMUM_HOLD_GROUPS: usize = 2;
pub const MINIMUM_ATOMIC_GROUPS: usize = 14;
pub const GROUPED_BOOTSTRAP_REPETITIONS: usize = 4_096;
pub const FIXED_CANDIDATE_TIMEOUT_SECONDS: u32 = 120;

pub const CURRENT_EXPLORATORY_DATASET_IDS: [&str; ROUTER_DATASET_COUNT] = [
    "019de5d76a718163",
    "047e62b52a3f4ede",
    "13b5df533e59a626",
    "17a4542ee4fb9e44",
    "1cb97b1d7314d43d",
    "30ef3eaa6d2971a9",
    "3305a3ad973fb227",
    "3baa7e37fc1f7861",
    "3f781a51af45387e",
    "542e75bc0cfdd1bf",
    "693890aae9dc860b",
    "79e600bc5fcd5da7",
    "ab0239b9ef0ba28e",
    "b253fbfae8460dc1",
    "cf5e74b4e9594944",
    "d2daf53412ab18dc",
    "dee1fc8e9afd81c9",
    "e1c1db5260d0894e",
    "ee56a1015e344c11",
    "fe3cc716a05d036f",
];

fn raw_sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn require_finite(name: &str, values: &[f64]) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} contains a nonfinite value"));
    }
    Ok(())
}

fn matrix_shape(name: &str, rows: &[Vec<f64>]) -> Result<(usize, usize), String> {
    if rows.is_empty() || rows[0].is_empty() {
        return Err(format!("{name} must be a nonempty matrix"));
    }
    let width = rows[0].len();
    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != width || row.iter().any(|value| !value.is_finite()) {
            return Err(format!("{name} row {row_index} is ragged or nonfinite"));
        }
    }
    Ok((rows.len(), width))
}

mod activation;
mod contracts;
mod finalizer;
mod policy;
mod strategy;

pub use activation::*;
pub use contracts::*;
pub use finalizer::*;
pub use policy::*;
pub use strategy::*;

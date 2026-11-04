//! Additive four-island, over-decomposed Golden42 executor.
//!
//! The older 90-core campaigns are intentionally not parameterized through
//! this module. The aggressive campaign has its own schemas, resource gate, routing rule, and
//! checkpoint chain while sharing only the complete Rust candidate evaluator.

use super::*;
use jope_dime_regressor::paths;

const SCHEMA: &str = "JopeGolden42AggressiveManifest.v5";
const PREFLIGHT_SCHEMA: &str = "JopeGolden42AggressivePreflight.v1";
const CHECKPOINT_SCHEMA: &str = "JopeGolden42AggressiveCheckpoint.v1";
const CONFIRMATION_SCHEMA: &str = "JopeGolden42AggressiveConfirmation.v1";
const RESULT_SCHEMA: &str = "JopeGolden42AggressiveDatasetResult.v1";
const SUMMARY_SCHEMA: &str = "JopeGolden42AggressiveSummary.v1";
const MANIFEST_FILE: &str = "aggressive-manifest.v1.json";
const ADMISSION_FILE: &str = "backend-admission.v1.json";
const PREFLIGHT_FILE: &str = "preflight.v1.json";
const STAGING_FILE: &str = "scq-staging-receipt.v1.json";
const SUMMARY_FILE: &str = "aggressive-summary.v1.json";
const GOLDEN46_RUN_PREFIX: &str = "golden46-new3-20260723-v4";
const GOLDEN46_CORRECTED_PARENT_RUN: &str = "golden46-new3-20260723-v2";
const GOLDEN46_TRANSFER_SUFFIX: &str = "-target-family-informed";
jope_paths::once_str!(
    fn golden46_panel_manifest_str => paths::quant_root().join("campaigns/golden46-full42-20260723-v1/manifests/panel46.json")
);
jope_paths::once_str!(
    fn golden46_leader_manifest_str => paths::quant_root().join("campaigns/golden46-full42-20260723-v1/manifests/leaders-new3.json")
);
const GOLDEN46_CAMPAIGN_ID: &str = "golden46-full42-20260723-v1";
const GOLDEN46_PANEL_DATASETS: [&str; 4] =
    ["ZTS_small", "a3bd0a400ec4d7f9", "6054656956182296", "ZTS"];
const GOLDEN46_DATASETS: [&str; 3] = ["ZTS_small", "a3bd0a400ec4d7f9", "6054656956182296"];
const GENERATION_ZERO_SEEDS_FILE: &str = "generation-zero-seeds.v1.json";
const EVOLUTION_VIEW_MANIFEST_FILE: &str = "evolution-view-manifest.v1.json";
const ISLANDS: usize = 4;
const ISLAND_POPULATION: usize = 9;
const TOTAL_POPULATION: usize = ISLANDS * ISLAND_POPULATION;
const ELITES_PER_ISLAND: usize = 2;
const PARENT_POOL_PER_ISLAND: usize = 3;
const MIN_VALID_PER_ISLAND: usize = ELITES_PER_ISLAND;
const UNBOUNDED_GENERATIONS: usize = usize::MAX;
const MIN_AVAILABLE_RAM_BYTES: u64 = 48 * 1024 * 1024 * 1024;
const MIN_FREE_DISK_BYTES: u64 = 100 * 1024 * 1024 * 1024;
// All three nodes have 8 GiB swap. Linux may retain cold pages there long
// after a competing build exits even when tens of GiB of RAM are available.
// Real pressure is rejected by the available-RAM and competing-process gates;
// stale swap occupancy alone must not deadlock campaign admission.
const MAX_SWAP_USED_BYTES: u64 = 9 * 1024 * 1024 * 1024;

mod model;
pub(crate) use model::*;
mod preflight;
pub(crate) use preflight::*;
mod population;
pub(crate) use population::*;
mod checkpoint;
pub(crate) use checkpoint::*;
mod dataset;
pub(crate) use dataset::*;
mod views;
pub(crate) use views::*;
mod validation;
pub(crate) use validation::*;
mod runner;
pub(crate) use runner::{run_aggressive, run_golden46, validate_aggressive, validate_golden46};

#[cfg(test)]
mod tests;

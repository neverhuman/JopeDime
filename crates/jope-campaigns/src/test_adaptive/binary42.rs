//! Frozen Binary42 held-out-test-adaptive campaigns.
//!
//! This lane is intentionally benchmark-invalid: held-out test ROC-AUC is the
//! evolutionary fitness. Every persisted record carries both
//! `test_adaptive: true` and `benchmark_valid: false`.

use super::*;
use crate::commands::binary_cv::{
    binary_genome_complexity, evaluate_test_adaptive_binary, evaluate_test_adaptive_binary_cpp17,
    test_adaptive_next_population, test_adaptive_seed_population, BinaryAdaptiveRanked,
    BinaryAncestry, BinaryMember, SEED_BINARY_SHA256,
};
use crate::data::read_binary_csv;
use feat_math::metrics::auc;
use jope_dime_regressor::paths;
use serde_json::json;
use std::collections::VecDeque;
use std::os::unix::fs::PermissionsExt;
use std::sync::{Arc, Mutex};

const COHORT_BYTES: &[u8] = include_bytes!("../../manifests/binary42-test-adaptive.v1.json");
const BINARY_SEEDS: &[u8] = include_bytes!("../../seeds/generation-zero-binary.txt");
jope_paths::once_str!(
    fn scq_binary_str => paths::scq_binary()
);
jope_paths::once_str!(
    fn all_scq_binary_str => paths::super_cluster_scq_binary()
);
jope_paths::once_str!(
    fn linux_remote_root_str => paths::legacy_tree("jope-test-adaptive")
);
const PROJECT: &str = "jope";
const ILPD: &str = "962a35a53b0803a8";
const NODES: [&str; 4] = ["xbabe1", "xbabe2", "xbabe3", "xbabe4"];
const PHYSICAL_HOSTNAMES: [&str; 4] = ["xbabe1", "xbabe2", "xbabe3", "xbabe4"];
const POPULATION: usize = 60;
const SHARD_SIZE: usize = 15;
const ALL_SHARD_SIZES: [usize; 4] = [19, 19, 19, 3];
const ALL_ACTIVE_DATASETS: usize = 6;
const ALL_MAX_GENERATIONS: usize = 10;
const CANDIDATE_STACK_BYTES: usize = 32 * 1024 * 1024;
const CPP17_BACKEND: &str = "fitted-cpp17-dynamic-v2";
const CPP17_STANDARD: &str = "C++17";
const CPP17_FLAGS: [&str; 5] = [
    "-std=c++17",
    "-O3",
    "-march=znver2",
    "-ffp-contract=off",
    "-fno-fast-math",
];
const ELITES: usize = 7;
const PARENT_POOL: usize = 30;
const TOURNAMENT: usize = 3;
const LOGLOSS_REPLAY_MAX_ULPS: u64 = 16;
const LOGLOSS_REPLAY_MAX_ABS: f64 = 1e-12;
const RNG_SEED: u64 = 42 ^ 0x6269_6e34_325f_696c;
const CAMPAIGN_SCHEMA: &str = "JopeBinary42TestAdaptiveCampaign.v1";
const BASELINE_SCHEMA: &str = "JopeBinary42TestAdaptiveBaseline.v1";
const SEED_SCHEMA: &str = "JopeBinary42TestAdaptiveSeedManifest.v1";
const STAGING_SCHEMA: &str = "JopeBinary42TestAdaptiveStaging.v1";
const SHARD_SCHEMA: &str = "JopeBinary42TestAdaptiveShard.v1";
const WORKER_SCHEMA: &str = "JopeBinary42TestAdaptiveWorkerReceipt.v1";
const CANDIDATE_SCHEMA: &str = "JopeBinary42TestAdaptiveCandidateReceipt.v1";
const GENERATION_SCHEMA: &str = "JopeBinary42TestAdaptiveGeneration.v1";
const CHECKPOINT_SCHEMA: &str = "JopeBinary42TestAdaptiveCheckpoint.v1";
const CACHE_SCHEMA: &str = "JopeBinary42TestAdaptiveCacheEntry.v1";
const GENOME_SCHEMA: &str = "JopeBinary42TestAdaptiveGenome.v1";
const CONFIRMATION_SCHEMA: &str = "JopeBinary42TestAdaptiveConfirmation.v1";
const WINNER_SCHEMA: &str = "JopeBinary42TestAdaptiveWinner.v1";
const COMPLETE_SCHEMA: &str = "JopeBinary42TestAdaptiveComplete.v1";
const EXHAUSTED_SCHEMA: &str = "JopeBinary42TestAdaptiveExhausted.v1";
const ALL_CAMPAIGN_SCHEMA: &str = "JopeBinary42AllCampaign.v1";
const ALL_STAGING_SCHEMA: &str = "JopeBinary42AllStaging.v1";
const ALL_SUMMARY_SCHEMA: &str = "JopeBinary42AllSummary.v1";

mod schema;
use schema::*;
mod member_models;
use member_models::*;
mod worker_models;
use worker_models::*;
mod cohort_models;
use cohort_models::*;
mod index_models;
use index_models::*;

mod campaign_setup;
use campaign_setup::*;
mod staging;
mod worker;
use staging::*;
mod execution;
use execution::*;
mod generation;
use generation::*;
mod indexes;
use indexes::*;
mod confirmation;
use confirmation::*;
mod publication;
use publication::*;
mod campaign;
use campaign::*;
mod validation;

#[cfg(test)]
mod tests;

pub(crate) fn batch_worker(manifest_path: &Path, output_dir: &Path) -> Result<()> {
    worker::batch_worker_impl(manifest_path, output_dir)
}

pub(crate) fn run(run_id: &str, dataset: &str, resume: bool) -> Result<()> {
    campaign::run_impl(run_id, dataset, resume)
}

pub(crate) fn run_all(run_id: &str, resume: bool) -> Result<()> {
    campaign::run_all_impl(run_id, resume)
}

pub(crate) fn validate_all(run_id: &str) -> Result<()> {
    validation::validate_all_impl(run_id)
}

pub(crate) fn validate(run_id: &str, dataset: &str) -> Result<()> {
    validation::validate_impl(run_id, dataset)
}

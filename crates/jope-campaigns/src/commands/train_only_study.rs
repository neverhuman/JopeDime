//! Checkpoint-assisted, ten-generation Golden42 regression study.
//!
//! The module is deliberately isolated from the published campaigns.  Phase
//! one never constructs a test path or reads baseline metrics; phase two can
//! only start from independently validated, hash-linked selection receipts.

use crate::data::{read_regression_csv, RegressionData};
use anyhow::{bail, Context, Result};
use feat_math::dsl::ParamValue;
use feat_math::evolve::{crossover_canonical_pipelines, mutate_canonical_pipeline_v2};
use feat_math::model::fit_with_stage_cache;
use feat_math::stage_cache::{StageCache, StageCacheCtx, DEFAULT_BYTE_CAP};
use feat_math::{fit, GenomeIr};
use jope_dime_regressor::paths;
use rayon::prelude::*;
use rayon::ThreadPool;
use rusqlite::{Connection, OpenFlags};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

pub(crate) const STUDY_SEED: u64 = 20_260_729;
pub(crate) const GENERATIONS: usize = 10;
pub(crate) const POPULATION: usize = 60;
pub(crate) const DL_QUOTA: usize = 30;
pub(crate) const GLOBAL_QUOTA: usize = 30;
pub(crate) const ELITES: usize = 3;
pub(crate) const PARENT_POOL: usize = 30;
pub(crate) const TOURNAMENT: usize = 3;
pub(crate) const MAX_TRANSFORMS: usize = 12;
pub(crate) const KERNEL_VERSION: &str = "legacy-v2.checkpoint-assisted-study.v2";
pub(crate) const STUDY_DESIGN: &str = "checkpoint_assisted_transfer";
const CONTEXT_FLOATS: usize = 1536 + 512;
const CONTEXT_BYTES: usize = CONTEXT_FLOATS * 4;
const DEFAULT_STUDY_ROOT: &str = "runs/train-only-studies";
jope_paths::once_str!(
    fn default_sotarena_root_str => paths::sotarena_reporter_root()
);
jope_paths::once_str!(
    fn default_checkpoint_pointer_str => paths::flywheel_root().join("training/accepted-by-task.json")
);
jope_paths::once_str!(
    fn default_checkpoint_provenance_str => paths::flywheel_training_root().join("round1/rotation-inputs.v1.json")
);
jope_paths::once_str!(
    fn default_catalog_pointer_str => paths::catalog_pointer()
);
jope_paths::once_str!(
    fn default_observation_db_str => paths::corpus_db()
);
const DEFAULT_PROPOSAL_GENERATOR: &str = "scripts/test_exact_dl_generate.py";
const MIN_RANKABLE: usize = 30;
const EXPECTED_ELIGIBLE: usize = 37;
const FINAL_CV_TOP_K: usize = 10;
const FINAL_CV_FOLDS: usize = 5;

const FROZEN_DATASETS: &[&str] = &[
    "ee56a1015e344c11",
    "019de5d76a718163",
    "5f91a00bd3b66742",
    "f8536dfeef471c2f",
    "7af85a04de0c7018",
    "aebf5d861a8365b4",
    "55b144de19659467",
    "55f06d147bcc2a32",
    "bd49e740a79bcb6c",
    "b5beff4a53fa12c0",
];

const GOLDEN42: &[&str] = &[
    "019de5d76a718163",
    "047e62b52a3f4ede",
    "13b5df533e59a626",
    "17a4542ee4fb9e44",
    "1a803b27c2400dc4",
    "1cb97b1d7314d43d",
    "21fbfe8b53dbcd60",
    "24713c7f7bfa7807",
    "30ef3eaa6d2971a9",
    "3305a3ad973fb227",
    "3baa7e37fc1f7861",
    "3f781a51af45387e",
    "43b4cc3ab0ebbd97",
    "471f11cc2d38853a",
    "4f68d277d770477f",
    "542e75bc0cfdd1bf",
    "55b144de19659467",
    "55f06d147bcc2a32",
    "5f91a00bd3b66742",
    "693890aae9dc860b",
    "79e600bc5fcd5da7",
    "7af85a04de0c7018",
    "860e5558f732f638",
    "ab0239b9ef0ba28e",
    "aebf5d861a8365b4",
    "b1f909fcfceeef4b",
    "b253fbfae8460dc1",
    "b5beff4a53fa12c0",
    "b78d323566930bd7",
    "bd49e740a79bcb6c",
    "c32fa09b48fb2f27",
    "c7d815714ca52bce",
    "c8f90fc9cc5364be",
    "cf5e74b4e9594944",
    "d2daf53412ab18dc",
    "dee1fc8e9afd81c9",
    "e1c1db5260d0894e",
    "e5a3a7b3668a1cc2",
    "ee56a1015e344c11",
    "f0e7eb3d790df37d",
    "f8536dfeef471c2f",
    "fe3cc716a05d036f",
];

#[derive(Clone, Debug)]
pub(crate) struct PrepareArgs {
    pub run_id: String,
    pub study_root: PathBuf,
    pub checkpoint_pointer: PathBuf,
    pub checkpoint_provenance: PathBuf,
    pub catalog_pointer: PathBuf,
    pub observation_db: PathBuf,
    pub dl_proposals: Option<PathBuf>,
    pub proposal_generator: PathBuf,
    pub python: PathBuf,
    pub proposal_device: String,
    pub sotarena_root: PathBuf,
    pub workers: usize,
    pub smoke: bool,
    pub proposal_manifest_only: bool,
}

impl PrepareArgs {
    pub(crate) fn production(run_id: String, study_root: Option<PathBuf>) -> Self {
        Self {
            run_id,
            study_root: study_root.unwrap_or_else(|| PathBuf::from(DEFAULT_STUDY_ROOT)),
            checkpoint_pointer: PathBuf::from(default_checkpoint_pointer_str()),
            checkpoint_provenance: PathBuf::from(default_checkpoint_provenance_str()),
            catalog_pointer: PathBuf::from(default_catalog_pointer_str()),
            observation_db: PathBuf::from(default_observation_db_str()),
            dl_proposals: None,
            proposal_generator: PathBuf::from(DEFAULT_PROPOSAL_GENERATOR),
            python: PathBuf::from("python3"),
            proposal_device: "cuda".to_string(),
            sotarena_root: PathBuf::from(default_sotarena_root_str()),
            workers: 8,
            smoke: false,
            proposal_manifest_only: false,
        }
    }
}

mod results;
mod study;
use results::*;
use study::*;

mod eligibility;
mod evaluation;
mod fitness;
mod prepare;
mod provenance;
mod report;
mod run;
mod scoring;
mod splits;
mod validation;
pub(super) use eligibility::{
    build_eligibility, build_transfer_rankings, copy_context_slices, generation_zero,
    load_dl_proposals, proposal_score, EligibleDatasets,
};
pub(super) use evaluation::{
    evaluate_member_split, offspring_seed, refill_population, score_population, slice_data,
    tournament,
};
pub(crate) use fitness::paired_robust_fitness;
pub(super) use fitness::{
    canonical_json_bytes, canonicalize_json, expression_column_indices, genome_compatible,
    genome_complexity, immutable_write, metrics, rank_scored, read_receipt, receipt_digest,
    registry_source_family, selection_key, semantic_identity, sha256_bytes, sha256_file,
    sotarena_metadata_path, study_dir, target_identity, validate_sha256, write_receipt,
};
pub(crate) use prepare::prepare;
pub(super) use prepare::{
    cache_key, validate_execution_inputs, validate_manifest_contract, validate_prepared_inputs,
};
pub(super) use provenance::{
    absolute_path, checkpoint_exclusions, completed_baseline_report, corpus_dataset_ids,
    extract_json_string_field, load_catalog, resolve_checkpoint_provenance, resolve_pointer_path,
    sha256_f64, sotarena_report_hashes, validate_train_data,
};
pub(crate) use report::report;
pub(super) use report::{
    bootstrap_interval, cohort_elo, evolution_svg, median, paired_sign_test, panel_points,
    panel_points_scaled, report_dataset,
};
pub(crate) use run::run;
pub(super) use run::{
    champion_validation_path, checkpoint_path, dataset_validation_path, run_dataset, selection_path,
};
pub(crate) use scoring::score;
pub(super) use scoring::{
    baseline_metric, load_or_create_baselines, load_or_create_final_cv_reranks,
    optional_baseline_metric, prior_aspfm_metric, score_frozen_champion, stronger_tabfm,
};
pub(super) use splits::{build_final_cv_splits, build_splits, rank_final_cv};
pub(crate) use validation::validate;
pub(super) use validation::{
    read_dataset_validation_shard, require_validated_score_gate, validate_dataset_run,
    validate_dataset_shard, validate_generation_zero, validate_observation_link,
};

#[cfg(test)]
mod tests;

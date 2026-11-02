//! Isolated train-only red-team search and selection study.
//!
//! Selection code in this module is deliberately unable to derive a baseline
//! or test path. Those paths exist only in `score`, after both arm receipts,
//! every per-dataset selector receipt, and the global validation receipt have
//! been independently replayed.

use crate::data::{read_regression_csv, RegressionData};
use anyhow::{bail, Context, Result};
use feat_math::evolve::{
    crossover_canonical_pipelines, mutate_canonical_pipeline_v2,
    mutate_canonical_pipeline_with_arm, CanonicalMutationArm,
};
use feat_math::{fit, GenomeIr};
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

#[cfg(test)]
const DEFAULT_RUN_ID: &str = "golden42-train-only-redteam-20260730-v1";
const SMART_V3_CPU_RUN_ID: &str = "golden42-smart-v3-cpu-20260730-v1";
const CONFIG_SCHEMA: &str = "TrainOnlyRedteamConfig.v1";
const DEFAULT_PARENT_RUN: &str = "golden42-transfer-evolution-20260730-v1";
const PARENT_MANIFEST_SCHEMA: &str = "TrainOnlyStudyManifest.v2";
const DATASET_COUNT: usize = 10;
const FOLDS: usize = 5;
const SMART_POPULATION: usize = 90;
const ISLAND_SIZE: usize = 30;
const ISLANDS: usize = 3;
const SMART_MIN_GENERATIONS: usize = 10;
const SMART_MAX_GENERATIONS: usize = 20;
const SMART_MAX_TRANSFORMS: usize = 10;
const G20_GENERATIONS: std::ops::Range<usize> = 10..20;
const DISCLOSURE: &str = "These ten test sets have already been exposed. Results are development-only, are not confirmatory, and cannot support a new blind-generalization claim.";

const DATASETS: [&str; DATASET_COUNT] = [
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

const OWNERSHIP: [(&str, &str); DATASET_COUNT] = [
    ("5f91a00bd3b66742", "xbabe1"),
    ("55b144de19659467", "xbabe2"),
    ("019de5d76a718163", "xbabe2"),
    ("b5beff4a53fa12c0", "xbabe2"),
    ("f8536dfeef471c2f", "xbabe2"),
    ("bd49e740a79bcb6c", "xbabe3"),
    ("7af85a04de0c7018", "xbabe3"),
    ("55f06d147bcc2a32", "xbabe4"),
    ("aebf5d861a8365b4", "xbabe4"),
    ("ee56a1015e344c11", "xbabe4"),
];

const SMART_V3_CPU_OWNERSHIP: [(&str, &str); DATASET_COUNT] = [
    ("55f06d147bcc2a32", "xbabe1"),
    ("bd49e740a79bcb6c", "xbabe2"),
    ("55b144de19659467", "xbabe2"),
    ("aebf5d861a8365b4", "xbabe2"),
    ("019de5d76a718163", "xbabe2"),
    ("b5beff4a53fa12c0", "xbabe3"),
    ("5f91a00bd3b66742", "xbabe3"),
    ("ee56a1015e344c11", "xbabe3"),
    ("7af85a04de0c7018", "xbabe4"),
    ("f8536dfeef471c2f", "xbabe4"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrainOnlyRedteamArm {
    G20Extension,
    SmartV3,
}

pub(crate) type Arm = TrainOnlyRedteamArm;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrainOnlyRedteamSelectionMode {
    Combined,
    SmartOnly,
}

pub(crate) type SelectionMode = TrainOnlyRedteamSelectionMode;

#[derive(Clone, Debug)]
pub(crate) struct PrepareArgs {
    pub run_id: String,
    pub study_root: PathBuf,
    pub sotarena_root: PathBuf,
    pub parent_run: PathBuf,
    pub proposals: Option<PathBuf>,
    pub workers: usize,
    pub smoke: bool,
}

impl PrepareArgs {
    pub(crate) fn production(
        run_id: String,
        study_root: PathBuf,
        sotarena_root: PathBuf,
        parent_run: Option<PathBuf>,
        proposals: Option<PathBuf>,
        workers: usize,
        smoke: bool,
    ) -> Self {
        let parent_run = parent_run
            .unwrap_or_else(|| study_root.join(DEFAULT_PARENT_RUN).join("manifest.v2.json"));
        Self {
            run_id,
            study_root,
            sotarena_root,
            parent_run,
            proposals,
            workers,
            smoke,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct TrainOnlyRedteamConfig {
    schema: String,
    run_id: String,
    development_only: bool,
    confirmatory_valid: bool,
    test_reuse_disclosure: String,
    dataset_ids: Vec<String>,
    proposal: ProposalConfig,
    fold_banks: FoldBankConfig,
    g20_extension: G20Config,
    smart_v3: SmartConfig,
    selector: SelectorConfig,
    scoring: ScoringConfig,
    authoritative_baselines: BaselineIdentities,
    scq: ScqConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct ProposalConfig {
    exact_decoder_seeds: Vec<u64>,
    exact_per_seed: usize,
    jitter_decoder_seeds: Vec<u64>,
    perturbation_seeds: Vec<u64>,
    jitter_per_perturbation: usize,
    embedding_dimensions: usize,
    target_summary_dimensions: usize,
    jitter_scale: f64,
    scaling_statistic: String,
    exact_context_rescore: bool,
    target_summary_unchanged: bool,
    train_csv_unchanged: bool,
    transfer_candidates: usize,
    target_specific_genomes_forbidden: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct FoldBankConfig {
    folds: usize,
    search_seeds: Vec<u64>,
    selector_seeds: Vec<u64>,
    audit_seeds: Vec<u64>,
    target_stratified: bool,
    paired_within_stage: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct G20Config {
    parent_run_id: String,
    generations: Vec<usize>,
    population: usize,
    validation_policy: String,
    fitness: String,
    operators: String,
    seed_stream: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct SmartConfig {
    progressive_screen: ProgressiveScreenConfig,
    population: usize,
    islands: usize,
    island_size: usize,
    island_roots: Vec<String>,
    elites_per_island: usize,
    parent_pool_per_island: usize,
    tournament_size: usize,
    mutation_probability: f64,
    crossover_probability: f64,
    mutation_mix: BTreeMap<String, f64>,
    max_transforms: usize,
    ordering: String,
    archive_per_island_generation: usize,
    minimum_generations: usize,
    maximum_generations: usize,
    stagnation_generations: usize,
    minimum_relative_improvement: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct ProgressiveScreenConfig {
    first_repeat_all_candidates: bool,
    exact_advance: usize,
    jittered_advance: usize,
    transfer_advance: usize,
    second_repeat_backfill: bool,
    initial_dl: usize,
    initial_transfer: usize,
    initial_mixed: usize,
    max_per_terminal_head_family: usize,
    canonical_unique: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct SelectorConfig {
    smart_archive_per_island: usize,
    control_candidates: usize,
    one_standard_error_rule: bool,
    ensemble_leaders: usize,
    ensemble_head_family_cap: usize,
    residual_correlation_cap: f64,
    ensemble_max_models: usize,
    crossfit_minimum_relative_improvement: f64,
    audit_minimum_relative_improvement: f64,
    convex_nonnegative_weights: bool,
    full_data_refit: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct ScoringConfig {
    tie_relative_tolerance: f64,
    bradley_terry_iterations: usize,
    bradley_terry_regularization: f64,
    bradley_terry_tolerance: f64,
    elo_center: f64,
    success_wins_required: usize,
    oracle_determines_success: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct BaselineIdentity {
    method_id: String,
    label: String,
    role: String,
    candidate_evaluations: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct BaselineIdentities {
    prior_train_only: BaselineIdentity,
    tabpfn_v3: BaselineIdentity,
    tabfm: BaselineIdentity,
    oracle_aspfm: BaselineIdentity,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct ScqConfig {
    project: String,
    effective_priority: f64,
    gpu_proposal_nodes: Vec<String>,
    cpu_preparation_node: String,
    evaluation_ownership: BTreeMap<String, String>,
    evaluation_cpus: usize,
    evaluation_ram_gb: usize,
    non_preemptible: bool,
    max_runtime_hours: usize,
    require_all_nodes: bool,
    node_marker_dependencies: bool,
    controller_read_only_except_final_aggregation: bool,
}

impl TrainOnlyRedteamConfig {
    fn sealed(run_id: &str, parent_run_id: &str, smoke: bool) -> Self {
        let smart_v3_cpu = run_id == SMART_V3_CPU_RUN_ID;
        let mut mutation_mix = BTreeMap::new();
        mutation_mix.insert("parameter_jitter".to_string(), 0.30);
        mutation_mix.insert("insertion".to_string(), 0.10);
        mutation_mix.insert("deletion".to_string(), 0.20);
        mutation_mix.insert("adjacent_swap".to_string(), 0.10);
        mutation_mix.insert("head_replacement".to_string(), 0.15);
        mutation_mix.insert("simplification".to_string(), 0.15);
        Self {
            schema: CONFIG_SCHEMA.to_string(),
            run_id: run_id.to_string(),
            development_only: true,
            confirmatory_valid: false,
            test_reuse_disclosure: DISCLOSURE.to_string(),
            dataset_ids: DATASETS.iter().map(|value| (*value).to_string()).collect(),
            proposal: ProposalConfig {
                exact_decoder_seeds: vec![202_607_300_101, 202_607_300_211, 202_607_300_307, 202_607_300_401],
                exact_per_seed: 64,
                jitter_decoder_seeds: vec![202_607_301_103, 202_607_301_211, 202_607_301_307],
                perturbation_seeds: vec![202_607_302_101, 202_607_302_203, 202_607_302_307],
                jitter_per_perturbation: 64,
                embedding_dimensions: 1536,
                target_summary_dimensions: 512,
                jitter_scale: 0.05,
                scaling_statistic: "per-coordinate population SD over the frozen checkpoint training-context catalog".to_string(),
                exact_context_rescore: true,
                target_summary_unchanged: true,
                train_csv_unchanged: true,
                transfer_candidates: 256,
                target_specific_genomes_forbidden: true,
            },
            fold_banks: FoldBankConfig {
                folds: FOLDS,
                search_seeds: vec![202_607_303_101, 202_607_303_211],
                selector_seeds: vec![202_607_304_103, 202_607_304_223],
                audit_seeds: vec![202_607_305_107],
                target_stratified: true,
                paired_within_stage: true,
            },
            g20_extension: G20Config {
                parent_run_id: parent_run_id.to_string(),
                generations: G20_GENERATIONS.collect(),
                population: 60,
                validation_policy: "sealed v2 deterministic target-stratified 80/20 generation split".to_string(),
                fitness: "sealed v2 paired clipped-R2 mean minus 0.25 population-SD".to_string(),
                operators: "feat-math canonical-v2".to_string(),
                seed_stream: "sealed parent run-id train-only-offspring-v1 stream".to_string(),
            },
            smart_v3: SmartConfig {
                progressive_screen: ProgressiveScreenConfig {
                    first_repeat_all_candidates: true,
                    exact_advance: 80,
                    jittered_advance: 60,
                    transfer_advance: 100,
                    second_repeat_backfill: true,
                    initial_dl: 30,
                    initial_transfer: 30,
                    initial_mixed: 30,
                    max_per_terminal_head_family: 10,
                    canonical_unique: true,
                },
                population: SMART_POPULATION,
                islands: ISLANDS,
                island_size: ISLAND_SIZE,
                island_roots: vec!["dl".to_string(), "transfer".to_string(), "mixed".to_string()],
                elites_per_island: 2,
                parent_pool_per_island: 15,
                tournament_size: 3,
                mutation_probability: 0.75,
                crossover_probability: 0.25,
                mutation_mix,
                max_transforms: SMART_MAX_TRANSFORMS,
                ordering: "NSGA-II robust CV loss and transform complexity within island".to_string(),
                archive_per_island_generation: 3,
                minimum_generations: if smoke { 2 } else { SMART_MIN_GENERATIONS },
                maximum_generations: if smoke { 2 } else { SMART_MAX_GENERATIONS },
                stagnation_generations: 5,
                minimum_relative_improvement: 0.001,
            },
            selector: SelectorConfig {
                smart_archive_per_island: 20,
                control_candidates: 10,
                one_standard_error_rule: true,
                ensemble_leaders: 12,
                ensemble_head_family_cap: 4,
                residual_correlation_cap: 0.98,
                ensemble_max_models: 4,
                crossfit_minimum_relative_improvement: 0.001,
                audit_minimum_relative_improvement: 0.0025,
                convex_nonnegative_weights: true,
                full_data_refit: true,
            },
            scoring: ScoringConfig {
                tie_relative_tolerance: 1e-12,
                bradley_terry_iterations: 250,
                bradley_terry_regularization: 1e-4,
                bradley_terry_tolerance: 1e-8,
                elo_center: 1500.0,
                success_wins_required: 6,
                oracle_determines_success: false,
            },
            authoritative_baselines: BaselineIdentities {
                prior_train_only: BaselineIdentity {
                    method_id: "jope-dime-regressor".to_string(),
                    label: "prior train-only Jope Dime Regressor".to_string(),
                    role: "fair".to_string(),
                    candidate_evaluations: None,
                },
                tabpfn_v3: BaselineIdentity {
                    method_id: "tabpfn_v3_regressor".to_string(),
                    label: "TabPFN v3".to_string(),
                    role: "fair".to_string(),
                    candidate_evaluations: None,
                },
                tabfm: BaselineIdentity {
                    method_id: "stronger_per_dataset_tabfm".to_string(),
                    label: "stronger per-dataset TabFM".to_string(),
                    role: "fair".to_string(),
                    candidate_evaluations: None,
                },
                oracle_aspfm: BaselineIdentity {
                    method_id: "jope-dime".to_string(),
                    label: "best-test ASPFM envelope over 63,108 Golden42 candidate evaluations".to_string(),
                    role: "oracle_reference_only".to_string(),
                    candidate_evaluations: Some(63_108),
                },
            },
            scq: ScqConfig {
                project: "sotarena-b42".to_string(),
                effective_priority: 10.0,
                gpu_proposal_nodes: if smart_v3_cpu {
                    Vec::new()
                } else {
                    vec![
                        "xbabe1".to_string(),
                        "xbabe2".to_string(),
                        "xbabe3".to_string(),
                    ]
                },
                cpu_preparation_node: "xbabe4".to_string(),
                evaluation_ownership: (if smart_v3_cpu {
                    SMART_V3_CPU_OWNERSHIP
                } else {
                    OWNERSHIP
                })
                    .into_iter()
                    .map(|(dataset, node)| (dataset.to_string(), node.to_string()))
                    .collect(),
                evaluation_cpus: if smart_v3_cpu { 4 } else { 8 },
                evaluation_ram_gb: if smart_v3_cpu { 16 } else { 40 },
                non_preemptible: true,
                max_runtime_hours: 24,
                require_all_nodes: true,
                node_marker_dependencies: true,
                controller_read_only_except_final_aggregation: true,
            },
        }
    }

    fn validate(&self) -> Result<()> {
        let smart_v3_cpu = self.run_id == SMART_V3_CPU_RUN_ID;
        let expected_ownership = (if smart_v3_cpu {
            SMART_V3_CPU_OWNERSHIP
        } else {
            OWNERSHIP
        })
        .into_iter()
        .map(|(dataset, node)| (dataset.to_string(), node.to_string()))
        .collect::<BTreeMap<_, _>>();
        if self.schema != CONFIG_SCHEMA
            || !self.development_only
            || self.confirmatory_valid
            || self.test_reuse_disclosure != DISCLOSURE
            || self.dataset_ids
                != DATASETS
                    .iter()
                    .map(|dataset| (*dataset).to_string())
                    .collect::<Vec<_>>()
            || self.proposal.exact_decoder_seeds
                != [
                    202_607_300_101,
                    202_607_300_211,
                    202_607_300_307,
                    202_607_300_401,
                ]
            || self.proposal.exact_decoder_seeds.len() != 4
            || self.proposal.jitter_decoder_seeds
                != [202_607_301_103, 202_607_301_211, 202_607_301_307]
            || self.proposal.perturbation_seeds
                != [202_607_302_101, 202_607_302_203, 202_607_302_307]
            || self.proposal.jitter_decoder_seeds.len() != 3
            || self.proposal.perturbation_seeds.len() != 3
            || self.proposal.exact_per_seed != 64
            || self.proposal.jitter_per_perturbation != 64
            || self.proposal.embedding_dimensions != 1536
            || self.proposal.target_summary_dimensions != 512
            || self.proposal.jitter_scale != 0.05
            || !self.proposal.exact_context_rescore
            || !self.proposal.target_summary_unchanged
            || !self.proposal.train_csv_unchanged
            || self.proposal.transfer_candidates != 256
            || !self.proposal.target_specific_genomes_forbidden
            || self.fold_banks.folds != 5
            || self.fold_banks.search_seeds != [202_607_303_101, 202_607_303_211]
            || self.fold_banks.selector_seeds != [202_607_304_103, 202_607_304_223]
            || self.fold_banks.audit_seeds != [202_607_305_107]
            || self.fold_banks.search_seeds.len() != 2
            || self.fold_banks.selector_seeds.len() != 2
            || self.fold_banks.audit_seeds.len() != 1
            || !self.fold_banks.target_stratified
            || !self.fold_banks.paired_within_stage
            || self.g20_extension.parent_run_id != DEFAULT_PARENT_RUN
            || self.g20_extension.generations != G20_GENERATIONS.collect::<Vec<_>>()
            || self.g20_extension.population != 60
            || self.smart_v3.population != 90
            || self.smart_v3.islands != 3
            || self.smart_v3.island_size != 30
            || self.smart_v3.island_roots != ["dl", "transfer", "mixed"]
            || self.smart_v3.elites_per_island != 2
            || self.smart_v3.parent_pool_per_island != 15
            || self.smart_v3.tournament_size != 3
            || self.smart_v3.mutation_probability != 0.75
            || self.smart_v3.crossover_probability != 0.25
            || self.smart_v3.max_transforms != 10
            || self.smart_v3.archive_per_island_generation != 3
            || !matches!(self.smart_v3.minimum_generations, 2 | 10)
            || !matches!(self.smart_v3.maximum_generations, 2 | 20)
            || self.smart_v3.minimum_generations != self.smart_v3.maximum_generations.min(10)
            || self.smart_v3.stagnation_generations != 5
            || self.smart_v3.minimum_relative_improvement != 0.001
            || !self.smart_v3.progressive_screen.first_repeat_all_candidates
            || self.smart_v3.progressive_screen.exact_advance != 80
            || self.smart_v3.progressive_screen.jittered_advance != 60
            || self.smart_v3.progressive_screen.transfer_advance != 100
            || !self.smart_v3.progressive_screen.second_repeat_backfill
            || self.smart_v3.progressive_screen.initial_dl != 30
            || self.smart_v3.progressive_screen.initial_transfer != 30
            || self.smart_v3.progressive_screen.initial_mixed != 30
            || self
                .smart_v3
                .progressive_screen
                .max_per_terminal_head_family
                != 10
            || !self.smart_v3.progressive_screen.canonical_unique
            || self.selector.smart_archive_per_island != 20
            || self.selector.control_candidates != 10
            || !self.selector.one_standard_error_rule
            || self.selector.ensemble_leaders != 12
            || self.selector.ensemble_head_family_cap != 4
            || self.selector.residual_correlation_cap != 0.98
            || self.selector.ensemble_max_models != 4
            || self.selector.crossfit_minimum_relative_improvement != 0.001
            || self.selector.audit_minimum_relative_improvement != 0.0025
            || !self.selector.convex_nonnegative_weights
            || !self.selector.full_data_refit
            || self.scoring.tie_relative_tolerance != 1e-12
            || self.scoring.bradley_terry_iterations != 250
            || self.scoring.bradley_terry_regularization != 1e-4
            || self.scoring.bradley_terry_tolerance != 1e-8
            || self.scoring.elo_center != 1500.0
            || self.scoring.success_wins_required != 6
            || self.scoring.oracle_determines_success
            || self.authoritative_baselines.prior_train_only.method_id != "jope-dime-regressor"
            || self.authoritative_baselines.prior_train_only.role != "fair"
            || self.authoritative_baselines.tabpfn_v3.method_id != "tabpfn_v3_regressor"
            || self.authoritative_baselines.tabpfn_v3.role != "fair"
            || self.authoritative_baselines.tabfm.method_id != "stronger_per_dataset_tabfm"
            || self.authoritative_baselines.tabfm.role != "fair"
            || self.authoritative_baselines.oracle_aspfm.method_id != "jope-dime"
            || self.authoritative_baselines.oracle_aspfm.role != "oracle_reference_only"
            || self
                .authoritative_baselines
                .oracle_aspfm
                .candidate_evaluations
                != Some(63_108)
            || self.scq.project != "sotarena-b42"
            || self.scq.effective_priority != 10.0
            || self.scq.gpu_proposal_nodes
                != if smart_v3_cpu {
                    Vec::<String>::new()
                } else {
                    vec![
                        "xbabe1".to_string(),
                        "xbabe2".to_string(),
                        "xbabe3".to_string(),
                    ]
                }
            || self.scq.cpu_preparation_node != "xbabe4"
            || self.scq.evaluation_ownership != expected_ownership
            || self.scq.evaluation_cpus != if smart_v3_cpu { 4 } else { 8 }
            || self.scq.evaluation_ram_gb != if smart_v3_cpu { 16 } else { 40 }
            || !self.scq.non_preemptible
            || self.scq.max_runtime_hours != 24
            || !self.scq.require_all_nodes
            || !self.scq.node_marker_dependencies
            || !self.scq.controller_read_only_except_final_aggregation
        {
            bail!("TrainOnlyRedteamConfig.v1 violates the sealed contract");
        }
        let seeds = self
            .fold_banks
            .search_seeds
            .iter()
            .chain(&self.fold_banks.selector_seeds)
            .chain(&self.fold_banks.audit_seeds)
            .copied()
            .collect::<HashSet<_>>();
        if seeds.len() != 5 {
            bail!("fold-bank seeds are not independent");
        }
        let mix_sum = self.smart_v3.mutation_mix.values().sum::<f64>();
        if (mix_sum - 1.0).abs() > 1e-15
            || self.smart_v3.mutation_mix.get("parameter_jitter") != Some(&0.30)
            || self.smart_v3.mutation_mix.get("insertion") != Some(&0.10)
            || self.smart_v3.mutation_mix.get("deletion") != Some(&0.20)
            || self.smart_v3.mutation_mix.get("adjacent_swap") != Some(&0.10)
            || self.smart_v3.mutation_mix.get("head_replacement") != Some(&0.15)
            || self.smart_v3.mutation_mix.get("simplification") != Some(&0.15)
        {
            bail!("Smart-v3 mutation mix drift");
        }
        for dataset in &self.dataset_ids {
            if !DATASETS.contains(&dataset.as_str())
                || !self.scq.evaluation_ownership.contains_key(dataset)
            {
                bail!("dataset identity or SCQ ownership drift: {dataset}");
            }
        }
        Ok(())
    }
}

mod commands;
#[path = "train_only_redteam_execution.rs"]
mod execution;
mod preparation;
mod receipts;
#[path = "train_only_redteam_results.rs"]
mod results;
mod scoring;
#[cfg(test)]
mod tests;

pub(crate) use commands::{report, run, score, score_smart_v3, validate};
pub(super) use commands::{report_run, run_arm, score_run, validate_run};
pub(crate) use preparation::prepare;
pub(super) use preparation::{
    build_fold_bank, build_repeat, compute_scaling_statistic, parse_proposal_bundle,
    proposal_manifest_from_parent, validate_fold_bank,
};
pub(crate) use receipts::ScqExecutionProvenance;
pub(super) use receipts::{
    absolute, canonical_json_bytes, canonicalize_json, immutable_write, read_receipt,
    receipt_digest, run_root, scq_env, scq_execution_provenance, sha256_bytes, sha256_file,
    value_string, value_usize, write_receipt, DatasetInput, FoldBankReceipt, FoldRepeat,
    ParentLink, ProposalBundleLink, ProposalRequest, ScalingStatistic, SelectionFirewall,
    TrainOnlyRedteamManifest,
};
pub(super) use scoring::*;

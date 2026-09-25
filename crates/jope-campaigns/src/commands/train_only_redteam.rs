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
pub(crate) enum Arm {
    G20Extension,
    SmartV3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SelectionMode {
    Combined,
    SmartOnly,
}

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

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ParentLink {
    run_id: String,
    manifest_path: PathBuf,
    manifest_sha256: String,
    validation_path: PathBuf,
    validation_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProposalBundleLink {
    path: PathBuf,
    sha256: String,
    exact_rows_per_dataset: usize,
    jittered_rows_per_dataset: usize,
    scaling_statistic_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DatasetInput {
    dataset_hash: String,
    train_path: PathBuf,
    train_sha256: String,
    rows: usize,
    features: usize,
    context_sha256: String,
    transfer_ranking_path: PathBuf,
    transfer_ranking_sha256: String,
    parent_generation_paths: Vec<PathBuf>,
    parent_generation_sha256: Vec<String>,
    fold_bank_path: PathBuf,
    fold_bank_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainOnlyRedteamManifest {
    schema: String,
    run_id: String,
    config: TrainOnlyRedteamConfig,
    config_sha256: String,
    development_only: bool,
    confirmatory_valid: bool,
    test_reuse_disclosure: String,
    parent: ParentLink,
    proposal_bundle: ProposalBundleLink,
    scaling_statistic_path: PathBuf,
    scaling_statistic_sha256: String,
    datasets: Vec<DatasetInput>,
    firewall: SelectionFirewall,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SelectionFirewall {
    selection_workers_receive_training_only: bool,
    contains_test_paths: bool,
    contains_baseline_paths: bool,
    test_paths_constructed: bool,
    baseline_paths_constructed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ScqExecutionProvenance {
    pub expected_node: String,
    pub scheduler_reported_node: String,
    pub scq_job_id: String,
    pub scq_attempt: usize,
    pub release_binary_sha256: String,
    pub source_bundle_sha256: String,
    pub manifest_sha256: String,
    pub proposal_bundle_sha256: String,
    pub fold_bank_sha256: String,
    pub parent_input_sha256: BTreeMap<String, String>,
    pub output_package_sha256: String,
    pub node_marker_probe: String,
    pub workers: usize,
    pub cpus_reserved: usize,
    pub ram_gb: usize,
    pub non_preemptible: bool,
    pub max_runtime_hours: usize,
    pub imported_without_overwrite: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ScalingStatistic {
    schema: String,
    source_catalog_sha256: String,
    contexts: usize,
    dimensions: usize,
    population_sd: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FoldBankReceipt {
    schema: String,
    run_id: String,
    dataset_hash: String,
    train_sha256: String,
    rows: usize,
    folds: usize,
    search: Vec<FoldRepeat>,
    selector: Vec<FoldRepeat>,
    audit: Vec<FoldRepeat>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FoldRepeat {
    purpose: String,
    repeat: usize,
    seed: u64,
    validation_indices: Vec<Vec<usize>>,
    repeat_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProposalRequest {
    schema: String,
    run_id: String,
    parent_manifest_sha256: String,
    checkpoint_path: PathBuf,
    checkpoint_sha256: String,
    catalog_pointer_path: PathBuf,
    proposal_dataset_manifest_path: PathBuf,
    scaling_statistic_path: PathBuf,
    scaling_statistic_sha256: String,
    exact_decoder_seeds: Vec<u64>,
    jitter_decoder_seeds: Vec<u64>,
    perturbation_seeds: Vec<u64>,
    proposals_per_invocation: usize,
    expected_combined_path: PathBuf,
    development_only: bool,
    confirmatory_valid: bool,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hex::encode(hash.finalize()))
}

fn canonicalize_json(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut entries = object.into_iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            let mut sorted = Map::new();
            for (key, value) in entries {
                sorted.insert(key, canonicalize_json(value));
            }
            Value::Object(sorted)
        }
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize_json).collect()),
        other => other,
    }
}

fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&canonicalize_json(value.clone()))?)
}

fn receipt_digest<T: Serialize>(receipt: &T) -> Result<String> {
    Ok(sha256_bytes(&canonical_json_bytes(&serde_json::to_value(
        receipt,
    )?)?))
}

fn scq_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn scq_execution_provenance(
    manifest: &TrainOnlyRedteamManifest,
    dataset: Option<&DatasetInput>,
    workers: usize,
) -> Option<ScqExecutionProvenance> {
    let job_id = scq_env("REDTEAM_SCQ_JOB_ID")?;
    let parse = |name: &str, default: usize| {
        scq_env(name)
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    };
    let mut parent_input_sha256 = BTreeMap::from([
        (
            "parent_manifest".to_string(),
            manifest.parent.manifest_sha256.clone(),
        ),
        (
            "parent_validation".to_string(),
            manifest.parent.validation_sha256.clone(),
        ),
    ]);
    let fold_bank_sha256 = dataset
        .map(|dataset| {
            for (generation, digest) in dataset.parent_generation_sha256.iter().enumerate() {
                parent_input_sha256
                    .insert(format!("parent_generation_{generation:03}"), digest.clone());
            }
            scq_env("REDTEAM_FOLD_BANK_SHA256").unwrap_or_else(|| dataset.fold_bank_sha256.clone())
        })
        .unwrap_or_default();
    if let Some(encoded) = scq_env("REDTEAM_PARENT_INPUT_SHA256") {
        if let Ok(captured) = serde_json::from_str::<BTreeMap<String, String>>(&encoded) {
            parent_input_sha256 = captured;
        }
    }
    Some(ScqExecutionProvenance {
        expected_node: scq_env("REDTEAM_EXPECTED_NODE").unwrap_or_default(),
        scheduler_reported_node: scq_env("REDTEAM_SCHEDULER_NODE")
            .unwrap_or_else(|| "unknown".to_string()),
        scq_job_id: job_id,
        scq_attempt: parse("REDTEAM_SCQ_ATTEMPT", 0),
        release_binary_sha256: scq_env("REDTEAM_RELEASE_BINARY_SHA256").unwrap_or_default(),
        source_bundle_sha256: scq_env("REDTEAM_SOURCE_BUNDLE_SHA256").unwrap_or_default(),
        manifest_sha256: scq_env("REDTEAM_MANIFEST_SHA256")
            .unwrap_or_else(|| receipt_digest(manifest).unwrap_or_default()),
        proposal_bundle_sha256: scq_env("REDTEAM_PROPOSAL_BUNDLE_SHA256")
            .unwrap_or_else(|| manifest.proposal_bundle.sha256.clone()),
        fold_bank_sha256,
        parent_input_sha256,
        output_package_sha256: scq_env("REDTEAM_OUTPUT_PACKAGE_SHA256")
            .unwrap_or_else(|| "not-captured".to_string()),
        node_marker_probe: scq_env("REDTEAM_NODE_MARKER_PROBE").unwrap_or_default(),
        workers: parse("REDTEAM_WORKERS", workers),
        cpus_reserved: parse("REDTEAM_CPUS_RESERVED", 8),
        ram_gb: parse("REDTEAM_RAM_GB", 40),
        non_preemptible: scq_env("REDTEAM_NON_PREEMPTIBLE").is_none_or(|value| value == "true"),
        max_runtime_hours: parse("REDTEAM_MAX_RUNTIME_HOURS", 24),
        imported_without_overwrite: scq_env("REDTEAM_IMPORTED_WITHOUT_OVERWRITE")
            .is_some_and(|value| value == "true"),
    })
}

fn immutable_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("immutable artifact has no parent")?;
    fs::create_dir_all(parent)?;
    if path.exists() {
        if fs::read(path)? == bytes {
            return Ok(());
        }
        bail!(
            "immutable artifact already exists with different bytes: {}",
            path.display()
        );
    }
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("artifact"),
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn write_receipt<T: Serialize>(path: &Path, receipt: &T) -> Result<String> {
    let mut value = serde_json::to_value(receipt)?;
    let digest = sha256_bytes(&canonical_json_bytes(&value)?);
    value
        .as_object_mut()
        .context("receipt must be a JSON object")?
        .insert("receipt_sha256".to_string(), Value::String(digest.clone()));
    let mut bytes = canonical_json_bytes(&value)?;
    bytes.push(b'\n');
    immutable_write(path, &bytes)?;
    Ok(digest)
}

fn read_receipt<T: DeserializeOwned>(path: &Path) -> Result<(T, String)> {
    let mut value: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )?;
    let claimed = value
        .as_object_mut()
        .and_then(|object| object.remove("receipt_sha256"))
        .and_then(|value| value.as_str().map(str::to_string))
        .context("receipt lacks receipt_sha256")?;
    let actual = sha256_bytes(&canonical_json_bytes(&value)?);
    if claimed != actual {
        bail!("receipt digest mismatch: {}", path.display());
    }
    Ok((serde_json::from_value(value)?, actual))
}

fn absolute(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    if let Some(root) = std::env::var_os("REDTEAM_PATH_ROOT").map(PathBuf::from) {
        if let Ok(relative) = absolute.strip_prefix(root) {
            return Ok(Path::new("/").join(relative));
        }
    }
    Ok(absolute)
}

fn run_root(study_root: &Path, run_id: &str) -> Result<PathBuf> {
    if run_id.is_empty()
        || !run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        bail!("run id contains unsupported characters");
    }
    Ok(study_root.join(run_id))
}

fn value_string(value: &Value, pointer: &str) -> Result<String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .map(str::to_string)
        .with_context(|| format!("missing string at {pointer}"))
}

fn value_usize(value: &Value, pointer: &str) -> Result<usize> {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .with_context(|| format!("missing integer at {pointer}"))
}

fn build_repeat(purpose: &str, repeat: usize, seed: u64, targets: &[f64]) -> Result<FoldRepeat> {
    if targets.len() < FOLDS || targets.iter().any(|value| !value.is_finite()) {
        bail!("fold bank needs at least five finite targets");
    }
    let mut sorted = (0..targets.len()).collect::<Vec<_>>();
    sorted.sort_by(|left, right| {
        targets[*left]
            .total_cmp(&targets[*right])
            .then_with(|| left.cmp(right))
    });
    let mut strata = vec![Vec::new(); 10];
    let row_count = sorted.len();
    for (rank, row) in sorted.into_iter().enumerate() {
        strata[(rank * 10 / row_count).min(9)].push(row);
    }
    let mut validation_indices = vec![Vec::new(); FOLDS];
    for (stratum, rows) in strata.iter_mut().enumerate() {
        rows.sort_by_key(|row| {
            let mut hash = Sha256::new();
            hash.update(b"train-only-redteam-fold-order-v1\0");
            hash.update(seed.to_le_bytes());
            hash.update((*row as u64).to_le_bytes());
            hash.finalize().to_vec()
        });
        for (position, row) in rows.iter().enumerate() {
            validation_indices[(position + stratum) % FOLDS].push(*row);
        }
    }
    let mut coverage = vec![0_u8; targets.len()];
    for fold in &mut validation_indices {
        fold.sort_unstable();
        if fold.is_empty() {
            bail!("fold bank contains an empty fold");
        }
        for row in fold {
            coverage[*row] = coverage[*row].saturating_add(1);
        }
    }
    if coverage.iter().any(|count| *count != 1) {
        bail!("every row must appear exactly once as validation per repeat");
    }
    let repeat_sha256 = receipt_digest(&json!({
        "purpose": purpose,
        "repeat": repeat,
        "seed": seed,
        "validation_indices": validation_indices,
    }))?;
    Ok(FoldRepeat {
        purpose: purpose.to_string(),
        repeat,
        seed,
        validation_indices,
        repeat_sha256,
    })
}

fn build_fold_bank(
    run_id: &str,
    dataset_hash: &str,
    train_sha256: &str,
    targets: &[f64],
    config: &TrainOnlyRedteamConfig,
) -> Result<FoldBankReceipt> {
    let make = |purpose: &str, seeds: &[u64]| {
        seeds
            .iter()
            .enumerate()
            .map(|(repeat, seed)| build_repeat(purpose, repeat, *seed, targets))
            .collect::<Result<Vec<_>>>()
    };
    let bank = FoldBankReceipt {
        schema: "TrainOnlyRedteamFoldBank.v1".to_string(),
        run_id: run_id.to_string(),
        dataset_hash: dataset_hash.to_string(),
        train_sha256: train_sha256.to_string(),
        rows: targets.len(),
        folds: FOLDS,
        search: make("search", &config.fold_banks.search_seeds)?,
        selector: make("selector", &config.fold_banks.selector_seeds)?,
        audit: make("audit", &config.fold_banks.audit_seeds)?,
    };
    validate_fold_bank(&bank, config)?;
    Ok(bank)
}

fn validate_fold_bank(bank: &FoldBankReceipt, config: &TrainOnlyRedteamConfig) -> Result<()> {
    if bank.schema != "TrainOnlyRedteamFoldBank.v1"
        || bank.folds != FOLDS
        || bank.search.len() != 2
        || bank.selector.len() != 2
        || bank.audit.len() != 1
    {
        bail!("fold-bank shape drift");
    }
    let expected = [
        ("search", config.fold_banks.search_seeds.as_slice()),
        ("selector", config.fold_banks.selector_seeds.as_slice()),
        ("audit", config.fold_banks.audit_seeds.as_slice()),
    ];
    for (purpose, seeds) in expected {
        let repeats = match purpose {
            "search" => &bank.search,
            "selector" => &bank.selector,
            _ => &bank.audit,
        };
        for (repeat, expected_seed) in repeats.iter().zip(seeds) {
            if repeat.purpose != purpose || repeat.seed != *expected_seed {
                bail!("fold-bank purpose or seed drift");
            }
            let rebuilt = receipt_digest(&json!({
                "purpose": repeat.purpose,
                "repeat": repeat.repeat,
                "seed": repeat.seed,
                "validation_indices": repeat.validation_indices,
            }))?;
            if rebuilt != repeat.repeat_sha256 {
                bail!("fold-bank repeat digest drift");
            }
            let mut coverage = vec![0_u8; bank.rows];
            for fold in &repeat.validation_indices {
                for row in fold {
                    if *row >= bank.rows {
                        bail!("fold-bank row is out of bounds");
                    }
                    coverage[*row] = coverage[*row].saturating_add(1);
                }
            }
            if coverage.iter().any(|count| *count != 1) {
                bail!("fold-bank coverage drift");
            }
        }
    }
    Ok(())
}

fn compute_scaling_statistic(parent: &Value) -> Result<ScalingStatistic> {
    let pointer_path = PathBuf::from(value_string(parent, "/catalog_pointer_path")?);
    let pointer: Value = serde_json::from_slice(&fs::read(&pointer_path)?)?;
    let catalog_path = PathBuf::from(value_string(&pointer, "/catalog")?);
    let data_path = PathBuf::from(value_string(&pointer, "/data")?);
    let catalog_sha256 = value_string(&pointer, "/catalog_sha256")?;
    if sha256_file(&catalog_path)? != catalog_sha256
        || sha256_file(&data_path)? != value_string(&pointer, "/data_sha256")?
    {
        bail!("parent context catalog pointer drift");
    }
    let catalog: Value = serde_json::from_slice(&fs::read(catalog_path)?)?;
    let entries = catalog
        .get("entries")
        .and_then(Value::as_array)
        .context("context catalog lacks entries")?;
    let dimensions = value_usize(&catalog, "/dataset_dimensions")?;
    let target_dimensions = value_usize(&catalog, "/target_dimensions")?;
    if dimensions != 1536 || target_dimensions != 512 {
        bail!("context catalog dimensions drift");
    }
    let stride_floats = dimensions + target_dimensions;
    let blob = fs::read(data_path)?;
    let regression = entries
        .iter()
        .filter(|entry| {
            entry.get("task").and_then(Value::as_str) == Some("regression")
                && entry.get("split").and_then(Value::as_str) == Some("train")
        })
        .collect::<Vec<_>>();
    if regression.len() < 2 {
        bail!("scaling statistic needs at least two regression contexts");
    }
    let mut sums = vec![0.0; dimensions];
    let mut squared_sums = vec![0.0; dimensions];
    for entry in &regression {
        let index = value_usize(entry, "/catalog_index")?;
        let offset = index
            .checked_mul(stride_floats * 4)
            .context("context offset overflow")?;
        if offset + stride_floats * 4 > blob.len() {
            bail!("context slice outside frozen blob");
        }
        for coordinate in 0..dimensions {
            let start = offset + coordinate * 4;
            let value = f32::from_le_bytes(blob[start..start + 4].try_into().unwrap()) as f64;
            if !value.is_finite() {
                bail!("non-finite training-context coordinate");
            }
            sums[coordinate] += value;
            squared_sums[coordinate] += value * value;
        }
    }
    let count = regression.len() as f64;
    let population_sd = sums
        .iter()
        .zip(squared_sums)
        .map(|(sum, squared_sum)| {
            let variance = (squared_sum / count - (sum / count).powi(2)).max(0.0);
            variance.sqrt()
        })
        .collect();
    Ok(ScalingStatistic {
        schema: "TrainOnlyEmbeddingScalingStatistic.v1".to_string(),
        source_catalog_sha256: catalog_sha256,
        contexts: regression.len(),
        dimensions,
        population_sd,
    })
}

fn proposal_manifest_from_parent(parent: &Value, path: &Path, run_id: &str) -> Result<String> {
    let entries = parent
        .get("datasets")
        .and_then(Value::as_array)
        .context("parent manifest lacks datasets")?;
    let catalog_sha256 = value_string(parent, "/source_catalog_sha256")?;
    let receipt = json!({
        "schema": "TrainOnlyProposalDatasetManifest.v1",
        "run_id": run_id,
        "catalog_sha256": catalog_sha256,
        "entries": entries.iter().map(|entry| json!({
            "task": "regression",
            "dataset_hash": entry.get("dataset_hash").cloned().unwrap_or(Value::Null),
            "context_sha256": entry.pointer("/encoding/context_sha256").cloned().unwrap_or(Value::Null),
        })).collect::<Vec<_>>(),
    });
    write_receipt(path, &receipt)
}

fn parse_proposal_bundle(
    path: &Path,
    config: &TrainOnlyRedteamConfig,
    scaling_sha256: &str,
    parent: &Value,
) -> Result<ProposalBundleLink> {
    let mut expected_contexts = HashMap::new();
    for entry in parent
        .get("datasets")
        .and_then(Value::as_array)
        .context("parent manifest lacks datasets")?
    {
        let dataset = value_string(entry, "/dataset_hash")?;
        if !config.dataset_ids.contains(&dataset) {
            continue;
        }
        let context_sha256 = value_string(entry, "/encoding/context_sha256")?;
        let blob_path = PathBuf::from(value_string(entry, "/encoding/blob_path")?);
        let offset = entry
            .pointer("/encoding/offset_bytes")
            .and_then(Value::as_u64)
            .context("parent context lacks offset_bytes")? as usize;
        let length = entry
            .pointer("/encoding/length_bytes")
            .and_then(Value::as_u64)
            .context("parent context lacks length_bytes")? as usize;
        if length != (1536 + 512) * std::mem::size_of::<f32>() {
            bail!("parent context has an invalid byte length");
        }
        let blob = fs::read(&blob_path)?;
        let context = blob
            .get(offset..offset + length)
            .context("parent context is outside its frozen blob")?;
        if sha256_bytes(context) != context_sha256 {
            bail!("parent context bytes drift");
        }
        let target_summary_sha256 = sha256_bytes(&context[1536 * std::mem::size_of::<f32>()..]);
        expected_contexts.insert(dataset, (context_sha256, target_summary_sha256));
    }
    if expected_contexts.len() != config.dataset_ids.len() {
        bail!("parent context provenance does not cover the sealed cohort");
    }
    let file = File::open(path)?;
    let mut exact: HashMap<String, HashMap<u64, usize>> = HashMap::new();
    let mut jittered: HashMap<String, HashMap<u64, usize>> = HashMap::new();
    for (_row_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)?;
        let dataset = value_string(&row, "/dataset_hash")?;
        if !config.dataset_ids.contains(&dataset) {
            bail!("proposal bundle contains an unfrozen dataset");
        }
        let (expected_context_sha256, expected_target_summary_sha256) = expected_contexts
            .get(&dataset)
            .context("proposal dataset lacks frozen context provenance")?;
        if row.get("task").and_then(Value::as_str) != Some("regression")
            || row.get("target_summary_unchanged").and_then(Value::as_bool) != Some(true)
            || row.get("context_sha256") != row.get("exact_context_sha256")
            || row.get("exact_context_sha256").and_then(Value::as_str)
                != Some(expected_context_sha256.as_str())
            || row.get("target_summary_sha256").and_then(Value::as_str)
                != Some(expected_target_summary_sha256.as_str())
        {
            bail!("proposal row violates exact-context/target-summary provenance");
        }
        let origin = value_string(&row, "/proposal_origin")?;
        let decoder_seed = row
            .get("invocation_seed")
            .and_then(Value::as_u64)
            .context("proposal lacks invocation_seed")?;
        match origin.as_str() {
            "exact_context" => {
                if !config.proposal.exact_decoder_seeds.contains(&decoder_seed)
                    || row
                        .get("perturbation_seed")
                        .is_some_and(|value| !value.is_null())
                    || row.get("embedding_jitter_scale").and_then(Value::as_f64) != Some(0.0)
                    || row.get("proposal_context_sha256").and_then(Value::as_str)
                        != Some(expected_context_sha256.as_str())
                {
                    bail!("exact proposal seed or perturbation metadata drift");
                }
                *exact
                    .entry(dataset)
                    .or_default()
                    .entry(decoder_seed)
                    .or_default() += 1;
            }
            "jittered_context" => {
                let perturbation_seed = row
                    .get("perturbation_seed")
                    .and_then(Value::as_u64)
                    .context("jittered proposal lacks perturbation seed")?;
                let perturbation_index = config
                    .proposal
                    .perturbation_seeds
                    .iter()
                    .position(|seed| *seed == perturbation_seed);
                if perturbation_index.is_none()
                    || config.proposal.jitter_decoder_seeds[perturbation_index.unwrap()]
                        != decoder_seed
                    || row.get("embedding_jitter_scale").and_then(Value::as_f64) != Some(0.05)
                    || row
                        .get("embedding_scaling_statistic_sha256")
                        .and_then(Value::as_str)
                        != Some(scaling_sha256)
                    || row.get("proposal_context_sha256").and_then(Value::as_str)
                        == Some(expected_context_sha256.as_str())
                {
                    bail!("jittered proposal provenance drift");
                }
                *jittered
                    .entry(dataset)
                    .or_default()
                    .entry(perturbation_seed)
                    .or_default() += 1;
            }
            _ => bail!("unsupported proposal origin"),
        }
    }
    for dataset in &config.dataset_ids {
        let exact_seeds = exact
            .get(dataset)
            .with_context(|| format!("no exact proposals for {dataset}"))?;
        for seed in &config.proposal.exact_decoder_seeds {
            if exact_seeds.get(seed) != Some(&config.proposal.exact_per_seed) {
                bail!("exact proposal quota drift for {dataset} seed {seed}");
            }
        }
        let jitter_seeds = jittered
            .get(dataset)
            .with_context(|| format!("no jittered proposals for {dataset}"))?;
        for seed in &config.proposal.perturbation_seeds {
            if jitter_seeds.get(seed) != Some(&config.proposal.jitter_per_perturbation) {
                bail!("jittered proposal quota drift for {dataset} seed {seed}");
            }
        }
    }
    Ok(ProposalBundleLink {
        path: absolute(path)?,
        sha256: sha256_file(path)?,
        exact_rows_per_dataset: config.proposal.exact_decoder_seeds.len()
            * config.proposal.exact_per_seed,
        jittered_rows_per_dataset: config.proposal.perturbation_seeds.len()
            * config.proposal.jitter_per_perturbation,
        scaling_statistic_sha256: scaling_sha256.to_string(),
    })
}

pub(crate) fn prepare(args: PrepareArgs) -> Result<PathBuf> {
    if args.workers == 0 {
        bail!("--workers must be at least one");
    }
    let root = run_root(&args.study_root, &args.run_id)?;
    fs::create_dir_all(root.join("inputs"))?;
    fs::create_dir_all(root.join("datasets"))?;
    let manifest_path = root.join("manifest.redteam.v1.json");
    if manifest_path.exists() {
        let (manifest, _): (TrainOnlyRedteamManifest, _) = read_receipt(&manifest_path)?;
        manifest.config.validate()?;
        return Ok(manifest_path);
    }
    let (parent, parent_digest): (Value, _) = read_receipt(&args.parent_run)?;
    if parent.get("schema").and_then(Value::as_str) != Some(PARENT_MANIFEST_SCHEMA)
        || parent.get("campaign_scope").and_then(Value::as_str) != Some("full")
        || parent
            .get("selected_dataset_ids")
            .and_then(Value::as_array)
            .map(Vec::len)
            != Some(DATASET_COUNT)
    {
        bail!("parent v2 manifest is not the sealed full ten-dataset study");
    }
    let parent_run_id = value_string(&parent, "/run_id")?;
    let config = TrainOnlyRedteamConfig::sealed(&args.run_id, &parent_run_id, args.smoke);
    config.validate()?;
    let config_path = root.join("config.v1.json");
    let config_digest = write_receipt(&config_path, &config)?;

    let validation_path = args
        .parent_run
        .parent()
        .context("parent manifest has no run directory")?
        .join("validation-success.v2.json");
    let (parent_validation, parent_validation_digest): (Value, _) = read_receipt(&validation_path)?;
    if parent_validation.get("schema").and_then(Value::as_str)
        != Some("TrainOnlyValidationReceipt.v2")
        || parent_validation
            .get("benchmark_valid")
            .and_then(Value::as_bool)
            != Some(true)
        || parent_validation
            .get("validated_dataset_count")
            .and_then(Value::as_u64)
            != Some(DATASET_COUNT as u64)
    {
        bail!("parent validation receipt is not complete");
    }

    let scaling = compute_scaling_statistic(&parent)?;
    let scaling_path = root.join("inputs/embedding-sd.v1.json");
    let scaling_digest = write_receipt(&scaling_path, &scaling)?;
    let proposal_dataset_manifest_path = root.join("inputs/proposal-datasets.v1.json");
    proposal_manifest_from_parent(&parent, &proposal_dataset_manifest_path, &args.run_id)?;

    if args.proposals.is_none() {
        let checkpoint_path = PathBuf::from(value_string(&parent, "/checkpoint/checkpoint_path")?);
        let checkpoint_sha256 = value_string(&parent, "/checkpoint/checkpoint_sha256")?;
        if sha256_file(&checkpoint_path)? != checkpoint_sha256 {
            bail!("parent checkpoint bytes drift");
        }
        let request = ProposalRequest {
            schema: "TrainOnlyRedteamProposalRequest.v1".to_string(),
            run_id: args.run_id,
            parent_manifest_sha256: parent_digest,
            checkpoint_path: absolute(&checkpoint_path)?,
            checkpoint_sha256,
            catalog_pointer_path: PathBuf::from(value_string(&parent, "/catalog_pointer_path")?),
            proposal_dataset_manifest_path: absolute(&proposal_dataset_manifest_path)?,
            scaling_statistic_path: absolute(&scaling_path)?,
            scaling_statistic_sha256: scaling_digest,
            exact_decoder_seeds: config.proposal.exact_decoder_seeds,
            jitter_decoder_seeds: config.proposal.jitter_decoder_seeds,
            perturbation_seeds: config.proposal.perturbation_seeds,
            proposals_per_invocation: 64,
            expected_combined_path: absolute(&root.join("inputs/proposals.v1.jsonl"))?,
            development_only: true,
            confirmatory_valid: false,
        };
        let path = root.join("proposal-request.v1.json");
        write_receipt(&path, &request)?;
        return Ok(path);
    }
    let proposals = parse_proposal_bundle(
        args.proposals.as_deref().unwrap(),
        &config,
        &scaling_digest,
        &parent,
    )?;

    let parent_datasets = parent
        .get("datasets")
        .and_then(Value::as_array)
        .context("parent manifest lacks dataset entries")?;
    let mut by_dataset = parent_datasets
        .iter()
        .map(|entry| Ok((value_string(entry, "/dataset_hash")?, entry)))
        .collect::<Result<HashMap<_, _>>>()?;
    let parent_root = args
        .parent_run
        .parent()
        .context("parent manifest has no run directory")?;
    let mut datasets = Vec::with_capacity(config.dataset_ids.len());
    let training_registry = absolute(&args.sotarena_root.join("data/tabular-regression-v1"))?;
    for dataset_hash in &config.dataset_ids {
        let entry = by_dataset
            .remove(dataset_hash)
            .with_context(|| format!("parent lacks {dataset_hash}"))?;
        let train_path = PathBuf::from(value_string(entry, "/train_path")?);
        let expected_train_path = training_registry.join(dataset_hash).join("train.csv");
        if absolute(&train_path)? != expected_train_path {
            bail!("parent training path is outside the authoritative registry");
        }
        let train_sha256 = value_string(entry, "/train_sha256")?;
        if sha256_file(&train_path)? != train_sha256 {
            bail!("parent training bytes drift for {dataset_hash}");
        }
        let data = read_regression_csv(&train_path)?;
        let bank = build_fold_bank(&args.run_id, dataset_hash, &train_sha256, &data.y, &config)?;
        let dataset_root = root.join("datasets").join(dataset_hash);
        fs::create_dir_all(&dataset_root)?;
        let fold_bank_path = dataset_root.join("fold-banks.v1.json");
        let fold_bank_digest = write_receipt(&fold_bank_path, &bank)?;
        let transfer_ranking_path = PathBuf::from(value_string(entry, "/transfer_ranking_path")?);
        let transfer_ranking_sha256 = value_string(entry, "/transfer_ranking_sha256")?;
        let mut parent_generation_paths = Vec::with_capacity(10);
        let mut parent_generation_sha256 = Vec::with_capacity(10);
        for generation in 0..10 {
            let path = parent_root
                .join("datasets")
                .join(dataset_hash)
                .join("generations")
                .join(format!("generation-{generation:03}.v2.json"));
            let (_, digest): (Value, _) = read_receipt(&path)?;
            parent_generation_paths.push(absolute(&path)?);
            parent_generation_sha256.push(digest);
        }
        datasets.push(DatasetInput {
            dataset_hash: dataset_hash.clone(),
            train_path: absolute(&train_path)?,
            train_sha256,
            rows: data.y.len(),
            features: data.features,
            context_sha256: value_string(entry, "/encoding/context_sha256")?,
            transfer_ranking_path: absolute(&transfer_ranking_path)?,
            transfer_ranking_sha256,
            parent_generation_paths,
            parent_generation_sha256,
            fold_bank_path: absolute(&fold_bank_path)?,
            fold_bank_sha256: fold_bank_digest,
        });
    }
    let manifest = TrainOnlyRedteamManifest {
        schema: "TrainOnlyRedteamManifest.v1".to_string(),
        run_id: args.run_id,
        config,
        config_sha256: config_digest,
        development_only: true,
        confirmatory_valid: false,
        test_reuse_disclosure: DISCLOSURE.to_string(),
        parent: ParentLink {
            run_id: parent_run_id,
            manifest_path: absolute(&args.parent_run)?,
            manifest_sha256: parent_digest,
            validation_path: absolute(&validation_path)?,
            validation_sha256: parent_validation_digest,
        },
        proposal_bundle: proposals,
        scaling_statistic_path: absolute(&scaling_path)?,
        scaling_statistic_sha256: scaling_digest,
        datasets,
        firewall: SelectionFirewall {
            selection_workers_receive_training_only: true,
            contains_test_paths: false,
            contains_baseline_paths: false,
            test_paths_constructed: false,
            baseline_paths_constructed: false,
        },
    };
    let serialized = String::from_utf8(canonical_json_bytes(&serde_json::to_value(&manifest)?)?)?
        .to_ascii_lowercase();
    for forbidden in [
        "/test.csv",
        "\"test_sha256\"",
        "\"baseline_path\"",
        "\"metrics/rmse\"",
    ] {
        if serialized.contains(forbidden) {
            bail!("selection manifest contains forbidden material: {forbidden}");
        }
    }
    write_receipt(&manifest_path, &manifest)?;
    Ok(manifest_path)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CvScore {
    repeat_nrmse: Vec<f64>,
    fold_nrmse: Vec<Vec<f64>>,
    robust_loss: f64,
    worst_fold_nrmse: f64,
}

fn robust_cv_score(
    repeat_squared_error: &[f64],
    repeat_rows: &[usize],
    fold_nrmse: Vec<Vec<f64>>,
    target_sd: f64,
) -> Result<CvScore> {
    if repeat_squared_error.is_empty()
        || repeat_squared_error.len() != repeat_rows.len()
        || !target_sd.is_finite()
        || target_sd <= 0.0
    {
        bail!("robust CV score inputs are invalid");
    }
    let repeat_nrmse = repeat_squared_error
        .iter()
        .zip(repeat_rows)
        .map(|(squared_error, rows)| {
            if !squared_error.is_finite() || *squared_error < 0.0 || *rows == 0 {
                bail!("robust CV repeat is invalid");
            }
            Ok((squared_error / *rows as f64).sqrt() / target_sd)
        })
        .collect::<Result<Vec<_>>>()?;
    let mean = repeat_nrmse.iter().sum::<f64>() / repeat_nrmse.len() as f64;
    let population_sd = (repeat_nrmse
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / repeat_nrmse.len() as f64)
        .sqrt();
    let worst_fold_nrmse = fold_nrmse
        .iter()
        .flatten()
        .copied()
        .max_by(f64::total_cmp)
        .context("robust CV needs fold metrics")?;
    Ok(CvScore {
        repeat_nrmse,
        fold_nrmse,
        robust_loss: mean + 0.25 * population_sd,
        worst_fold_nrmse,
    })
}

fn target_sd(targets: &[f64]) -> Result<f64> {
    if targets.is_empty() || targets.iter().any(|value| !value.is_finite()) {
        bail!("target SD needs finite values");
    }
    let mean = targets.iter().sum::<f64>() / targets.len() as f64;
    let sd = (targets
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / targets.len() as f64)
        .sqrt();
    if sd <= 0.0 {
        bail!("target SD is zero");
    }
    Ok(sd)
}

#[derive(Clone, Debug)]
struct OneSeCandidate {
    semantic_sha256: String,
    complexity: usize,
    robust_loss: f64,
    repeat_nrmse: Vec<f64>,
    worst_fold_nrmse: f64,
}

fn one_standard_error_select(candidates: &[OneSeCandidate]) -> Result<usize> {
    let (best_index, best) = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            candidate.robust_loss.is_finite()
                && !candidate.repeat_nrmse.is_empty()
                && candidate.repeat_nrmse.iter().all(|value| value.is_finite())
        })
        .min_by(|(_, left), (_, right)| {
            left.robust_loss
                .total_cmp(&right.robust_loss)
                .then_with(|| left.worst_fold_nrmse.total_cmp(&right.worst_fold_nrmse))
                .then_with(|| left.complexity.cmp(&right.complexity))
                .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
        })
        .context("one-SE selection has no valid candidates")?;
    let mean = best.repeat_nrmse.iter().sum::<f64>() / best.repeat_nrmse.len() as f64;
    let sample_sd = if best.repeat_nrmse.len() > 1 {
        (best
            .repeat_nrmse
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (best.repeat_nrmse.len() - 1) as f64)
            .sqrt()
    } else {
        0.0
    };
    let threshold = best.robust_loss + sample_sd / (best.repeat_nrmse.len() as f64).sqrt();
    Ok(candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            candidate.robust_loss.is_finite() && candidate.robust_loss <= threshold
        })
        .min_by(|(_, left), (_, right)| {
            left.complexity
                .cmp(&right.complexity)
                .then_with(|| left.robust_loss.total_cmp(&right.robust_loss))
                .then_with(|| left.worst_fold_nrmse.total_cmp(&right.worst_fold_nrmse))
                .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
        })
        .map(|(index, _)| index)
        .unwrap_or(best_index))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct EloRating {
    method_id: String,
    rating: f64,
}

fn bradley_terry_elo(
    scores: &BTreeMap<String, BTreeMap<String, f64>>,
    config: &ScoringConfig,
) -> Result<Vec<EloRating>> {
    let methods = scores
        .values()
        .flat_map(|dataset| dataset.keys().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if methods.is_empty() {
        return Ok(Vec::new());
    }
    let method_index = methods
        .iter()
        .enumerate()
        .map(|(index, method)| (method.clone(), index))
        .collect::<HashMap<_, _>>();
    let mut matches = Vec::new();
    for dataset in scores.values() {
        if dataset.len() != methods.len() {
            bail!("Elo datasets must contain the same field");
        }
        let entries = dataset.iter().collect::<Vec<_>>();
        for left in 0..entries.len() {
            for right in (left + 1)..entries.len() {
                let (method_a, score_a) = entries[left];
                let (method_b, score_b) = entries[right];
                if !score_a.is_finite() || !score_b.is_finite() {
                    bail!("Elo score is non-finite");
                }
                let scale = score_a.abs().max(score_b.abs()).max(1.0);
                let outcome = if (score_a - score_b).abs() <= config.tie_relative_tolerance * scale
                {
                    0.5
                } else if score_a < score_b {
                    1.0
                } else {
                    0.0
                };
                matches.push((method_index[method_a], method_index[method_b], outcome));
            }
        }
    }
    let mut ratings = vec![config.elo_center; methods.len()];
    let elo_scale = 10_f64.ln() / 400.0;
    for _ in 0..config.bradley_terry_iterations {
        let mut gradient = vec![0.0; methods.len()];
        let mut information = vec![0.0; methods.len()];
        for &(left, right, outcome) in &matches {
            let delta = (elo_scale * (ratings[left] - ratings[right])).clamp(-700.0, 700.0);
            let probability = 1.0 / (1.0 + (-delta).exp());
            let residual = outcome - probability;
            let curvature = probability * (1.0 - probability);
            gradient[left] += elo_scale * residual;
            gradient[right] -= elo_scale * residual;
            let info = elo_scale * elo_scale * curvature;
            information[left] += info;
            information[right] += info;
        }
        let mut largest_update = 0.0_f64;
        for index in 0..ratings.len() {
            let regularized_gradient = gradient[index]
                - config.bradley_terry_regularization * (ratings[index] - config.elo_center);
            let update = (regularized_gradient
                / (information[index] + config.bradley_terry_regularization))
                .clamp(-40.0, 40.0);
            largest_update = largest_update.max(update.abs());
            ratings[index] += update;
        }
        if largest_update < config.bradley_terry_tolerance {
            break;
        }
    }
    let mean = ratings.iter().sum::<f64>() / ratings.len() as f64;
    for rating in &mut ratings {
        *rating += config.elo_center - mean;
    }
    let mut result = methods
        .into_iter()
        .zip(ratings)
        .map(|(method_id, rating)| EloRating { method_id, rating })
        .collect::<Vec<_>>();
    result.sort_by(|left, right| {
        right
            .rating
            .total_cmp(&left.rating)
            .then_with(|| left.method_id.cmp(&right.method_id))
    });
    Ok(result)
}

// Execution, validation, scoring, and reporting are below the contract and
// pure-selection helpers so their only route to external test material is
// visually auditable.
#[path = "train_only_redteam_execution.rs"]
mod execution;
#[path = "train_only_redteam_results.rs"]
mod results;

pub(crate) fn run(
    manifest_path: &Path,
    arm: Arm,
    selection_mode: SelectionMode,
    resume: bool,
    workers: usize,
    selected_dataset: Option<&str>,
) -> Result<PathBuf> {
    run_arm(
        manifest_path,
        arm,
        selection_mode,
        resume,
        workers,
        selected_dataset,
    )
}

pub(crate) fn validate(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
    selected_dataset: Option<&str>,
) -> Result<PathBuf> {
    validate_run(study_root, run_id, sotarena_root, workers, selected_dataset)
}

pub(crate) fn score(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    score_run(study_root, run_id, sotarena_root, workers)
}

pub(crate) fn score_smart_v3(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    results::score_smart_v3_run(study_root, run_id, sotarena_root, workers)
}

pub(crate) fn report(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    report_run(study_root, run_id, sotarena_root, workers)
}

fn run_arm(
    manifest_path: &Path,
    arm: Arm,
    selection_mode: SelectionMode,
    resume: bool,
    workers: usize,
    selected_dataset: Option<&str>,
) -> Result<PathBuf> {
    execution::run_arm(
        manifest_path,
        arm,
        selection_mode,
        resume,
        workers,
        selected_dataset,
    )
}

fn validate_run(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
    selected_dataset: Option<&str>,
) -> Result<PathBuf> {
    results::validate_run(study_root, run_id, sotarena_root, workers, selected_dataset)
}

fn score_run(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    results::score_run(study_root, run_id, sotarena_root, workers)
}

fn report_run(
    study_root: &Path,
    run_id: &str,
    sotarena_root: &Path,
    workers: usize,
) -> Result<PathBuf> {
    results::report_run(study_root, run_id, sotarena_root, workers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sealed_config_covers_requested_contract() {
        let config = TrainOnlyRedteamConfig::sealed(DEFAULT_RUN_ID, DEFAULT_PARENT_RUN, false);
        config.validate().unwrap();
        assert!(config.development_only);
        assert!(!config.confirmatory_valid);
        assert_eq!(config.dataset_ids.len(), 10);
        assert_eq!(config.smart_v3.mutation_mix.values().sum::<f64>(), 1.0);
        assert_eq!(
            config.authoritative_baselines.prior_train_only.method_id,
            "jope-dime-regressor"
        );
        assert_eq!(
            config.authoritative_baselines.oracle_aspfm.method_id,
            "jope-dime"
        );
        let smoke = TrainOnlyRedteamConfig::sealed(
            "golden42-train-only-redteam-20260730-v1-smoke",
            DEFAULT_PARENT_RUN,
            true,
        );
        smoke.validate().unwrap();
        assert_eq!(smoke.dataset_ids.len(), 10);
        assert_eq!(smoke.smart_v3.minimum_generations, 2);
        assert_eq!(smoke.smart_v3.maximum_generations, 2);
        let smart_cpu =
            TrainOnlyRedteamConfig::sealed(SMART_V3_CPU_RUN_ID, DEFAULT_PARENT_RUN, false);
        smart_cpu.validate().unwrap();
        assert_eq!(smart_cpu.scq.evaluation_cpus, 4);
        assert_eq!(smart_cpu.scq.evaluation_ram_gb, 16);
        assert!(smart_cpu.scq.gpu_proposal_nodes.is_empty());
        assert_eq!(
            smart_cpu.scq.evaluation_ownership["55f06d147bcc2a32"],
            "xbabe1"
        );
        assert_eq!(
            smart_cpu.scq.evaluation_ownership["ee56a1015e344c11"],
            "xbabe3"
        );
    }

    #[test]
    fn fold_repeats_cover_every_row_once_and_are_separated() {
        let targets = (0..103)
            .map(|row| ((row * 37) % 101) as f64 / 101.0)
            .collect::<Vec<_>>();
        let config = TrainOnlyRedteamConfig::sealed(DEFAULT_RUN_ID, DEFAULT_PARENT_RUN, false);
        let bank = build_fold_bank(
            DEFAULT_RUN_ID,
            DATASETS[0],
            &"a".repeat(64),
            &targets,
            &config,
        )
        .unwrap();
        validate_fold_bank(&bank, &config).unwrap();
        let hashes = bank
            .search
            .iter()
            .chain(&bank.selector)
            .chain(&bank.audit)
            .map(|repeat| &repeat.repeat_sha256)
            .collect::<HashSet<_>>();
        assert_eq!(hashes.len(), 5);
    }

    #[test]
    fn robust_loss_uses_pooled_repeat_rmse() {
        let score = robust_cv_score(
            &[25.0, 36.0],
            &[100, 100],
            vec![vec![0.5; 5], vec![0.6; 5]],
            1.0,
        )
        .unwrap();
        assert_eq!(score.repeat_nrmse, vec![0.5, 0.6]);
        assert!((score.robust_loss - 0.5625).abs() < 1e-15);
        assert_eq!(score.worst_fold_nrmse, 0.6);
    }

    #[test]
    fn one_se_prefers_simpler_eligible_model() {
        let candidates = vec![
            OneSeCandidate {
                semantic_sha256: "b".repeat(64),
                complexity: 8,
                robust_loss: 0.100,
                repeat_nrmse: vec![0.09, 0.11],
                worst_fold_nrmse: 0.12,
            },
            OneSeCandidate {
                semantic_sha256: "a".repeat(64),
                complexity: 3,
                robust_loss: 0.105,
                repeat_nrmse: vec![0.10, 0.11],
                worst_fold_nrmse: 0.12,
            },
        ];
        assert_eq!(one_standard_error_select(&candidates).unwrap(), 1);
    }

    #[test]
    fn bradley_terry_uses_official_relative_tie_rule_and_center() {
        let config = TrainOnlyRedteamConfig::sealed(DEFAULT_RUN_ID, DEFAULT_PARENT_RUN, false);
        let scores = BTreeMap::from([
            (
                "d1".to_string(),
                BTreeMap::from([
                    ("a".to_string(), 1.0),
                    ("b".to_string(), 1.0 + 5e-13),
                    ("c".to_string(), 2.0),
                ]),
            ),
            (
                "d2".to_string(),
                BTreeMap::from([
                    ("a".to_string(), 1.0),
                    ("b".to_string(), 0.9),
                    ("c".to_string(), 2.0),
                ]),
            ),
        ]);
        let ratings = bradley_terry_elo(&scores, &config.scoring).unwrap();
        let mean = ratings.iter().map(|row| row.rating).sum::<f64>() / ratings.len() as f64;
        assert!((mean - 1500.0).abs() < 1e-10);
        assert_eq!(ratings[0].method_id, "b");
        assert_eq!(ratings[2].method_id, "c");
    }

    #[test]
    fn bradley_terry_matches_canonical_arena_reference() {
        let config = TrainOnlyRedteamConfig::sealed(DEFAULT_RUN_ID, DEFAULT_PARENT_RUN, false);
        let scores = BTreeMap::from([
            (
                "d1".to_string(),
                BTreeMap::from([
                    ("a".to_string(), 1.0),
                    ("b".to_string(), 1.0 + 5e-13),
                    ("c".to_string(), 2.0),
                    ("d".to_string(), 0.8),
                ]),
            ),
            (
                "d2".to_string(),
                BTreeMap::from([
                    ("a".to_string(), 1.2),
                    ("b".to_string(), 0.9),
                    ("c".to_string(), 1.4),
                    ("d".to_string(), 1.1),
                ]),
            ),
            (
                "d3".to_string(),
                BTreeMap::from([
                    ("a".to_string(), 0.5),
                    ("b".to_string(), 0.6),
                    ("c".to_string(), 0.7),
                    ("d".to_string(), 0.4),
                ]),
            ),
        ]);
        let rust = bradley_terry_elo(&scores, &config.scoring)
            .unwrap()
            .into_iter()
            .map(|rating| (rating.method_id, rating.rating))
            .collect::<BTreeMap<_, _>>();
        // The maintained port of sotarena/elo.py is qualified separately by
        // jope-arena/tests/elo_regression.rs against retained ladder fixtures.
        // This is cross-implementation Rust parity, not a Python execution.
        let reference =
            jope_arena::elo::fit_elo(&jope_arena::elo::matches_from_scores(&scores, 1e-12));
        assert_eq!(
            rust.keys().collect::<Vec<_>>(),
            reference.keys().collect::<Vec<_>>()
        );
        for (method, rating) in rust {
            assert!(
                (rating - reference[&method]).abs() < 1e-9,
                "{method}: campaign={rating}, arena reference={}",
                reference[&method]
            );
        }
    }

    #[test]
    fn immutable_receipts_fail_closed_after_tampering() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("receipt.json");
        write_receipt(
            &path,
            &json!({
                "schema": "TrainOnlyRedteamTamperFixture.v1",
                "development_only": true,
                "confirmatory_valid": false,
            }),
        )
        .unwrap();
        let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value["development_only"] = Value::Bool(false);
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(read_receipt::<Value>(&path).is_err());
        assert!(write_receipt(
            &path,
            &json!({
                "schema": "TrainOnlyRedteamTamperFixture.v1",
                "development_only": true,
                "confirmatory_valid": false,
            }),
        )
        .is_err());
    }
}

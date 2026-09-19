use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::commands::exact_cv::{self, ValidatedFreeze, ORIGINAL_GENERATIONS};
use crate::core::require_core_plan;
use crate::data::{read_binary_csv, read_regression_csv};
use jope_dime_regressor::paths;

mod aggressive;
mod args;
mod commands;
mod core;
mod data;
mod evaluation_v2;
mod followup;
mod genome_boost;
mod golden42_v3;
mod ratchet;
mod runtime;
mod score_corpus;
mod task;
mod test_adaptive;

jope_paths::once_str!(
    fn registry_str => paths::jope_registry()
);
const PANEL_SHA256: &str = "b86c8ea5f5ab1051e8aafb4368f8dabdcfd8c740c63eaa1d086ddb72fae0ba75";
const PANEL_ANALYSIS_SHA256: &str =
    "0117abb32d547b134706e210376f2e6f437b706b2a12edb6d9302d74d4749282";
const SEED_SHA256: &str = "5f097a0017ddc7e92ad5ea8dfadaf783b548e82287d92626dc9d2fa9429b2105";
const BASE_SEED: u64 = 20260616;
// --- binary-classification lane (v2-binary; fixture-backed until real datasets land) ---
const PANEL_BINARY_SHA256: &str =
    "be3e8ff9654e31bbb1015a1e783bb9ba58f86295706dbdbaf137509e0b69d575";
const PANEL_ANALYSIS_BINARY_SHA256: &str =
    "34f476c21e055d58033eb9278247d5ddfdf9c7c65e4660f4e7372f8345b26da6";
const MODEL_BINARY: &str = "jope-dime-binary";
const BINARY_REPORT_FILE: &str = "jope-dime-binary-report.v1.json";
const BINARY_PREDICTION_FILE: &str = "jope-dime-binary-predictions.csv";
const BINARY_RESULTS_FILE: &str = "jope-dime-binary-panel-results.v1.json";
/// Lane tag ("jpbin_v1" as bytes) XORed into the seed derivation so binary
/// campaigns never share offspring streams with the regression lane.
const BINARY_LANE_TAG: u64 = 0x6a70_6269_6e5f_7631;
const MODEL: &str = "jope-dime";
const REPORT_FILE: &str = "jope-dime-regressor.json";
const PREDICTION_FILE: &str = "jope-dime-regressor.csv";
const BASELINES: &[&str] = &[
    "TabFM ensemble",
    "TabPFNv3",
    "Fable5",
    "Grok4p5",
    "OpenAI GPT-5.6-sol",
    "Kimik3",
];
const DATASETS: &[&str] = &[
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

#[derive(Parser)]
#[command(name = "jope-dime-regressor")]
struct Cli {
    #[command(subcommand)]
    command: Command,
    #[arg(long, global = true, env = "JOPE_REGISTRY", default_value = registry_str())]
    registry: PathBuf,
    #[arg(long, global = true, default_value = "grok4p5-42")]
    panel_id: String,
    #[arg(long, global = true)]
    dataset: Option<String>,
    /// Kernel contract for newly launched work. Existing manifests are
    /// permanently legacy-v2 and cannot be resumed under advanced-v3.
    #[arg(long, global = true, value_enum, default_value = "legacy-v2")]
    kernel_version: KernelVersion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum KernelVersion {
    LegacyV2,
    AdvancedV3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum TrainOnlyRedteamArm {
    G20Extension,
    SmartV3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum TrainOnlyRedteamSelectionMode {
    Combined,
    SmartOnly,
}

#[derive(Subcommand)]
enum Command {
    VerifyInputs,
    SelectPanel,
    ValidatePanel,
    ScorePanel,
    /// Freeze the checkpoint-assisted Golden42 transfer-evolution cohort.
    #[command(name = "prepare-train-only-study")]
    PrepareTrainOnlyStudy {
        #[arg(long, default_value = "golden42-transfer-evolution-20260730-v1")]
        run_id: String,
        #[arg(long, default_value = "runs/train-only-studies")]
        study_root: PathBuf,
        #[arg(
            long,
            default_value_os_t = paths::flywheel_root().join("training/accepted-by-task.json")
        )]
        checkpoint_pointer: PathBuf,
        #[arg(
            long,
            default_value_os_t = paths::flywheel_training_root().join("round1/rotation-inputs.v1.json")
        )]
        checkpoint_provenance: PathBuf,
        #[arg(
            long,
            default_value_os_t = paths::catalog_pointer()
        )]
        catalog_pointer: PathBuf,
        #[arg(
            long,
            default_value_os_t = paths::corpus_db()
        )]
        observation_db: PathBuf,
        #[arg(long)]
        dl_proposals: Option<PathBuf>,
        #[arg(long, default_value = "scripts/test_exact_dl_generate.py")]
        proposal_generator: PathBuf,
        #[arg(long, default_value = "python3")]
        python: PathBuf,
        #[arg(long, default_value = "cuda")]
        proposal_device: String,
        #[arg(
            long,
            env = "SOTARENA_REPORTER_ROOT",
            default_value_os_t = paths::sotarena_reporter_root()
        )]
        sotarena_root: PathBuf,
        #[arg(long, default_value_t = 8)]
        workers: usize,
        /// Run the fixed ee56a1015e344c11 two-generation smoke benchmark.
        #[arg(long, hide = true)]
        smoke: bool,
        /// Freeze only the exact proposal cohort for a portable SCQ proposal job.
        #[arg(long, hide = true)]
        proposal_manifest_only: bool,
    },
    /// Execute generations 0 through 9 from a sanitized train-only manifest.
    #[command(name = "run-train-only-study")]
    RunTrainOnlyStudy {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        resume: bool,
        #[arg(long, default_value_t = 8)]
        workers: usize,
    },
    /// Independently replay the train-only ledgers without opening test data.
    #[command(name = "validate-train-only-study")]
    ValidateTrainOnlyStudy {
        #[arg(long, default_value = "golden42-transfer-evolution-20260730-v1")]
        run_id: String,
        #[arg(long, default_value = "runs/train-only-studies")]
        study_root: PathBuf,
        #[arg(
            long,
            env = "SOTARENA_REPORTER_ROOT",
            default_value_os_t = paths::sotarena_reporter_root()
        )]
        sotarena_root: PathBuf,
        #[arg(long, default_value_t = 8)]
        workers: usize,
    },
    /// Open test data and baselines only after the validation gate succeeds.
    #[command(name = "score-train-only-study")]
    ScoreTrainOnlyStudy {
        #[arg(long, default_value = "golden42-transfer-evolution-20260730-v1")]
        run_id: String,
        #[arg(long, default_value = "runs/train-only-studies")]
        study_root: PathBuf,
        #[arg(
            long,
            env = "SOTARENA_REPORTER_ROOT",
            default_value_os_t = paths::sotarena_reporter_root()
        )]
        sotarena_root: PathBuf,
        #[arg(long, default_value_t = 8)]
        workers: usize,
    },
    /// Render immutable ledgers into per-dataset and cohort study reports.
    #[command(name = "report-train-only-study")]
    ReportTrainOnlyStudy {
        #[arg(long, default_value = "golden42-transfer-evolution-20260730-v1")]
        run_id: String,
        #[arg(long, default_value = "runs/train-only-studies")]
        study_root: PathBuf,
        #[arg(
            long,
            env = "SOTARENA_REPORTER_ROOT",
            default_value_os_t = paths::sotarena_reporter_root()
        )]
        sotarena_root: PathBuf,
        #[arg(long, default_value_t = 8)]
        workers: usize,
    },
    /// Freeze the isolated development-only train-only red-team study.
    #[command(name = "prepare-train-only-redteam")]
    PrepareTrainOnlyRedteam {
        #[arg(long, default_value = "golden42-train-only-redteam-20260730-v1")]
        run_id: String,
        #[arg(long, default_value = "runs/train-only-studies")]
        study_root: PathBuf,
        #[arg(
            long,
            env = "SOTARENA_REPORTER_ROOT",
            default_value_os_t = paths::sotarena_reporter_root()
        )]
        sotarena_root: PathBuf,
        #[arg(long)]
        parent_run: Option<PathBuf>,
        /// Combined, provenance-complete exact+jittered proposal JSONL.
        #[arg(long)]
        proposals: Option<PathBuf>,
        #[arg(long, default_value_t = 8)]
        workers: usize,
        /// Build ten-dataset, two-generation smoke contracts for all four owners.
        #[arg(long, hide = true)]
        smoke: bool,
    },
    /// Run one frozen red-team arm without baseline or test access.
    #[command(name = "run-train-only-redteam")]
    RunTrainOnlyRedteam {
        #[arg(long, default_value = "golden42-train-only-redteam-20260730-v1")]
        run_id: String,
        #[arg(long, default_value = "runs/train-only-studies")]
        study_root: PathBuf,
        #[arg(
            long,
            env = "SOTARENA_REPORTER_ROOT",
            default_value_os_t = paths::sotarena_reporter_root()
        )]
        sotarena_root: PathBuf,
        #[arg(long, value_enum)]
        arm: Option<TrainOnlyRedteamArm>,
        #[arg(long, value_enum, default_value = "combined")]
        selection_mode: TrainOnlyRedteamSelectionMode,
        #[arg(long)]
        resume: bool,
        #[arg(long, default_value_t = 8)]
        workers: usize,
    },
    /// Replay both arms and every frozen selector without baseline/test access.
    #[command(name = "validate-train-only-redteam")]
    ValidateTrainOnlyRedteam {
        #[arg(long, default_value = "golden42-train-only-redteam-20260730-v1")]
        run_id: String,
        #[arg(long, default_value = "runs/train-only-studies")]
        study_root: PathBuf,
        #[arg(
            long,
            env = "SOTARENA_REPORTER_ROOT",
            default_value_os_t = paths::sotarena_reporter_root()
        )]
        sotarena_root: PathBuf,
        #[arg(long, default_value_t = 8)]
        workers: usize,
    },
    /// Score frozen variants only after the global validation gate succeeds.
    #[command(name = "score-train-only-redteam")]
    ScoreTrainOnlyRedteam {
        #[arg(long, default_value = "golden42-train-only-redteam-20260730-v1")]
        run_id: String,
        #[arg(long, default_value = "runs/train-only-studies")]
        study_root: PathBuf,
        #[arg(
            long,
            env = "SOTARENA_REPORTER_ROOT",
            default_value_os_t = paths::sotarena_reporter_root()
        )]
        sotarena_root: PathBuf,
        #[arg(long, default_value_t = 8)]
        workers: usize,
    },
    /// Score ten frozen Smart-v3 selectors, opening test data only here.
    #[command(name = "score-smart-v3-redteam")]
    ScoreSmartV3Redteam {
        #[arg(long, default_value = "golden42-smart-v3-cpu-20260730-v1")]
        run_id: String,
        #[arg(long, default_value = "runs/train-only-studies")]
        study_root: PathBuf,
        #[arg(
            long,
            env = "SOTARENA_REPORTER_ROOT",
            default_value_os_t = paths::sotarena_reporter_root()
        )]
        sotarena_root: PathBuf,
        #[arg(long, default_value_t = 2)]
        workers: usize,
    },
    /// Render fair and oracle-reference ladders with development-only disclosure.
    #[command(name = "report-train-only-redteam")]
    ReportTrainOnlyRedteam {
        #[arg(long, default_value = "golden42-train-only-redteam-20260730-v1")]
        run_id: String,
        #[arg(long, default_value = "runs/train-only-studies")]
        study_root: PathBuf,
        #[arg(
            long,
            env = "SOTARENA_REPORTER_ROOT",
            default_value_os_t = paths::sotarena_reporter_root()
        )]
        sotarena_root: PathBuf,
        #[arg(long, default_value_t = 8)]
        workers: usize,
    },
    /// Prove one explicitly manifested train-only regression fold under V2 evidence keys.
    #[command(name = "verify-evaluation-v2", hide = true)]
    VerifyEvaluationV2 {
        #[arg(long)]
        manifest: PathBuf,
    },
    /// Verify binary panel training inputs (training-only firewall).
    #[command(name = "verify-inputs-binary")]
    VerifyInputsBinary,
    /// Run or resume the binary-classification selection pass.
    #[command(name = "select-panel-binary")]
    SelectPanelBinary,
    /// Validate all frozen binary selections without opening test data.
    #[command(name = "validate-panel-binary")]
    ValidatePanelBinary,
    /// Score frozen binary selections on test data (AUC strictly-greater wins).
    #[command(name = "score-panel-binary")]
    ScorePanelBinary,
    /// Run or resume the benchmark-invalid Binary42 ILPD held-out-test-adaptive campaign.
    #[command(name = "run-binary-test-adaptive")]
    RunBinaryTestAdaptive {
        #[arg(long, default_value = "binary42-test-adaptive-ilpd-v1")]
        run_id: String,
        #[arg(long)]
        resume: bool,
    },
    /// Independently validate and rebuild indexes for a Binary42 test-adaptive campaign.
    #[command(name = "validate-binary-test-adaptive")]
    ValidateBinaryTestAdaptive {
        #[arg(long, default_value = "binary42-test-adaptive-ilpd-v1")]
        run_id: String,
    },
    /// Run or resume all 42 benchmark-invalid Binary42 held-out-test-adaptive campaigns.
    #[command(name = "run-binary-test-adaptive-all")]
    RunBinaryTestAdaptiveAll {
        #[arg(long, default_value = "binary42-test-adaptive-all-v1")]
        run_id: String,
        #[arg(long)]
        resume: bool,
    },
    /// Validate all 42 Binary42 campaign artifacts and rebuild cohort indexes.
    #[command(name = "validate-binary-test-adaptive-all")]
    ValidateBinaryTestAdaptiveAll {
        #[arg(long, default_value = "binary42-test-adaptive-all-v1")]
        run_id: String,
    },
    /// Freeze the all-272 multiclass corpus, comparator locks, and generation-zero population.
    #[command(name = "plan-multiclass-test-adaptive")]
    PlanMulticlassTestAdaptive {
        #[arg(long, default_value = "multiclass272-test-adaptive-union43-gen20-v2")]
        run_id: String,
        #[arg(long, default_value_os_t = paths::test_adaptive_root())]
        campaign_root: PathBuf,
        #[arg(
            long,
            default_value_os_t = paths::quant_root().join("releases/mc-union-runnable272-v5/manifest.json")
        )]
        corpus_manifest: PathBuf,
        #[arg(
            long,
            default_value_os_t = paths::test_adaptive_root().join("comparators/post-coverage-v2")
        )]
        comparator_root: PathBuf,
        #[arg(long, default_value = "seeds/generation-zero.txt")]
        seed_catalog: PathBuf,
        #[arg(long, default_value_t = 60)]
        population: usize,
        #[arg(long, default_value_t = 20)]
        generations: usize,
    },
    /// Harvest, evolve, and resume all 272 held-out-test-adaptive multiclass campaigns.
    #[command(name = "run-multiclass-test-adaptive-all")]
    RunMulticlassTestAdaptiveAll {
        #[arg(long, default_value = "multiclass272-test-adaptive-union43-gen20-v2")]
        run_id: String,
        #[arg(long, default_value_os_t = paths::test_adaptive_root())]
        campaign_root: PathBuf,
        #[arg(long)]
        resume: bool,
    },
    /// Validate terminal ledgers, final incumbents, wave ordering, and three-node replay.
    #[command(name = "validate-multiclass-test-adaptive-all")]
    ValidateMulticlassTestAdaptiveAll {
        #[arg(long, default_value = "multiclass272-test-adaptive-union43-gen20-v2")]
        run_id: String,
        #[arg(long, default_value_os_t = paths::test_adaptive_root())]
        campaign_root: PathBuf,
    },
    /// Run or resume all 42 benchmark-invalid TimeSeries42 held-out-test-adaptive shards.
    #[command(name = "run-timeseries-test-adaptive-all")]
    RunTimeseriesTestAdaptiveAll {
        #[arg(long, default_value = "timeseries42-test-adaptive-forecast3-gen20-v2")]
        run_id: String,
        #[arg(long)]
        resume: bool,
    },
    /// Validate all TimeSeries42 campaign artifacts and rebuild final indexes.
    #[command(name = "validate-timeseries-test-adaptive-all")]
    ValidateTimeseriesTestAdaptiveAll {
        #[arg(long, default_value = "timeseries42-test-adaptive-forecast3-gen20-v2")]
        run_id: String,
    },
    /// Continue a finished TimeSeries42 base campaign through generation 29.
    #[command(name = "run-timeseries-test-adaptive-gen30")]
    RunTimeseriesTestAdaptiveGen30 {
        #[arg(long, default_value = "timeseries42-test-adaptive-all-v1")]
        base_run_id: String,
        #[arg(long, default_value = "timeseries42-test-adaptive-gen30-v1")]
        run_id: String,
        #[arg(long)]
        resume: bool,
    },
    /// Validate a TimeSeries42 gen30 continuation against its immutable base run.
    #[command(name = "validate-timeseries-test-adaptive-gen30")]
    ValidateTimeseriesTestAdaptiveGen30 {
        #[arg(long, default_value = "timeseries42-test-adaptive-all-v1")]
        base_run_id: String,
        #[arg(long, default_value = "timeseries42-test-adaptive-gen30-v1")]
        run_id: String,
    },
    /// Continue TimeSeries42 gen30 non-winners through a longer SCQ win-all search.
    #[command(name = "run-timeseries-test-adaptive-winall")]
    RunTimeseriesTestAdaptiveWinall {
        #[arg(long, default_value = "timeseries42-test-adaptive-gen30-v1")]
        source_run_id: String,
        #[arg(long, default_value = "timeseries42-test-adaptive-winall-v1")]
        run_id: String,
        #[arg(long)]
        max_generations: Option<usize>,
        #[arg(long)]
        lineage_salt: Option<String>,
        #[arg(long)]
        resume: bool,
    },
    /// Execute one immutable Binary42 candidate shard. Intended only for SCQ jobs.
    #[command(name = "binary-test-adaptive-batch-worker", hide = true)]
    BinaryTestAdaptiveBatchWorker {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    /// Execute one immutable TimeSeries42 corpus-eval shard. Intended only for SCQ jobs.
    #[command(name = "timeseries-test-adaptive-batch-worker", hide = true)]
    TimeseriesTestAdaptiveBatchWorker {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(
            long,
            default_value_os_t = paths::corpus_eval_binary()
        )]
        corpus_eval: PathBuf,
        #[arg(long, default_value_t = 300)]
        timeout_seconds: u64,
    },
    RunPanel,
    Summarize,
    /// Run or resume the exploratory 30-generation follow-up for the 35 original losses.
    #[command(name = "run-losses-3x")]
    RunLosses3x,
    /// Validate all 35 follow-up freezes without opening test data or baseline metrics.
    #[command(name = "validate-losses-3x")]
    ValidateLosses3x,
    /// Run or resume the isolated, intentionally test-adaptive b78 campaign.
    #[command(name = "run-test-adaptive")]
    RunTestAdaptive {
        #[arg(long, default_value = "test-adaptive-b78-v1")]
        run_id: String,
        #[arg(long)]
        resume: bool,
    },
    /// Run or resume the sequential test-adaptive campaign across all 42 Golden42 datasets.
    #[command(name = "run-test-adaptive-panel")]
    RunTestAdaptivePanel {
        #[arg(long, default_value = "test-adaptive-golden42-v1")]
        run_id: String,
        #[arg(long)]
        resume: bool,
    },
    /// Continue every base no-win through ten additional full generations (10-19).
    #[command(name = "run-test-adaptive-extension")]
    RunTestAdaptiveExtension {
        #[arg(long, default_value = "test-adaptive-golden42-v1")]
        run_id: String,
        #[arg(long)]
        resume: bool,
    },
    /// Continue the seven generation-19 no-wins on four isolated 30-member islands.
    #[command(name = "run-test-adaptive-island-extension")]
    RunTestAdaptiveIslandExtension {
        #[arg(long, default_value = "test-adaptive-golden42-v1")]
        run_id: String,
        #[arg(long)]
        resume: bool,
    },
    /// Continue generation-29 Golden42 no-wins with widening complexity until all 42 lead.
    #[command(name = "run-golden42-champions")]
    RunGolden42Champions {
        #[arg(long, default_value = "test-adaptive-golden42-v1")]
        run_id: String,
        #[arg(long)]
        resume: bool,
    },
    /// Run the same four-island SCQ campaign on TabArena Lite regression until all leaders fall.
    #[command(name = "run-tabarena-lite-islands")]
    RunTabarenaLiteIslands {
        #[arg(long, default_value = "tabarena-lite-regression-islands-v1")]
        run_id: String,
        #[arg(long)]
        resume: bool,
    },
    /// Recover a missing final result from an immutable leader-beating TabArena checkpoint.
    #[command(name = "recover-tabarena-lite-result", hide = true)]
    RecoverTabarenaLiteResult {
        #[arg(long)]
        run_id: String,
        #[arg(long)]
        dataset: String,
    },
    /// Evaluate one test-adaptive candidate. Intended only for SCQ jobs.
    #[command(name = "test-adaptive-worker", hide = true)]
    TestAdaptiveWorker {
        #[arg(long)]
        request: PathBuf,
    },
    /// Evaluate an immutable batch of test-adaptive candidates. Intended only for SCQ jobs.
    #[command(name = "test-adaptive-island-batch-worker", hide = true)]
    TestAdaptiveIslandBatchWorker {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    /// Independently audit a completed or in-progress test-adaptive run.
    #[command(name = "validate-test-adaptive")]
    ValidateTestAdaptive {
        #[arg(long)]
        run_id: String,
    },
    /// Independently validate a 42-dataset test-adaptive panel campaign.
    #[command(name = "validate-test-adaptive-panel")]
    ValidateTestAdaptivePanel {
        #[arg(long)]
        run_id: String,
    },
    /// Independently validate the immutable generations 10-19 no-win extension.
    #[command(name = "validate-test-adaptive-extension")]
    ValidateTestAdaptiveExtension {
        #[arg(long)]
        run_id: String,
    },
    /// Independently validate the immutable four-island generations 20-29 extension.
    #[command(name = "validate-test-adaptive-island-extension")]
    ValidateTestAdaptiveIslandExtension {
        #[arg(long, default_value = "test-adaptive-golden42-v1")]
        run_id: String,
    },
    /// Independently validate the generation-30+ all-champion Golden42 continuation.
    #[command(name = "validate-golden42-champions")]
    ValidateGolden42Champions {
        #[arg(long, default_value = "test-adaptive-golden42-v1")]
        run_id: String,
    },
    /// Independently validate a completed TabArena Lite four-island campaign.
    #[command(name = "validate-tabarena-lite-islands")]
    ValidateTabarenaLiteIslands {
        #[arg(long, default_value = "tabarena-lite-regression-islands-v1")]
        run_id: String,
    },
    /// Benchmark the additive Rust/C++17/Python/JavaScript inference fixtures.
    #[command(name = "bench-languages-v3")]
    BenchLanguagesV3 {
        #[arg(long)]
        output: PathBuf,
    },
    /// Record the correctness-first fitted backend admission decision.
    #[command(name = "admit-fitted-backend-v1")]
    AdmitFittedBackendV1 {
        #[arg(long)]
        output: PathBuf,
    },
    #[command(name = "admit-fitted-cpp-v3")]
    AdmitFittedCppV3 {
        #[arg(long)]
        output: PathBuf,
    },
    #[command(name = "run-golden42-v3-max")]
    RunGolden42V3Max {
        #[arg(long)]
        run_id: String,
        #[arg(long, default_value = "24713c7f7bfa7807")]
        dataset: String,
        #[arg(long, default_value = "xbabe2")]
        node: String,
        #[arg(long, default_value_t = 80)]
        workers: usize,
        #[arg(long, default_value_t = 80)]
        population: usize,
        #[arg(long)]
        resume: bool,
    },
    #[command(name = "validate-golden42-v3-max")]
    ValidateGolden42V3Max {
        #[arg(long)]
        run_id: String,
    },
    #[command(name = "golden42-v3-persistent-worker", hide = true)]
    Golden42V3PersistentWorker {
        #[arg(long)]
        bootstrap: PathBuf,
    },
    #[command(name = "golden42-v3-confirm", hide = true)]
    Golden42V3Confirm {
        #[arg(long)]
        request: PathBuf,
    },
    /// Write the immutable contract for the exploratory 80-worker study.
    #[command(name = "plan-golden42-aggressive")]
    PlanGolden42Aggressive {
        #[arg(long)]
        output: PathBuf,
    },
    /// Run or resume the additive 240-candidate, three-node SCQ80 study.
    #[command(name = "run-golden42-aggressive")]
    RunGolden42Aggressive {
        #[arg(long)]
        run_id: String,
        /// Run Golden42 datasets selected by one or more comma-separated exact hashes or unique prefixes.
        #[arg(long)]
        dataset: Option<String>,
        /// Skip the first N datasets in canonical Golden42 order.
        #[arg(long, default_value_t = 0)]
        start_index: usize,
        #[arg(long)]
        resume: bool,
    },
    /// Validate all complete barriers and confirmations in an SCQ80 study.
    #[command(name = "validate-golden42-aggressive")]
    ValidateGolden42Aggressive {
        #[arg(long)]
        run_id: String,
    },
    /// Run or resume the Golden46 four-island evolution on the three required datasets.
    #[command(name = "run-golden46")]
    RunGolden46 {
        #[arg(long, default_value = "golden46-new3-20260723-v4")]
        run_id: String,
        /// Exact ZTS/ZTS_small id, or an exact/unique hexadecimal dataset prefix.
        #[arg(long)]
        dataset: Option<String>,
        /// Legacy full-ZTS switch; rejected because full ZTS is out of scope.
        #[arg(long)]
        transfer_arm: bool,
        #[arg(long)]
        resume: bool,
    },
    /// Validate a primary or target-family-informed Golden46 evolution ledger.
    #[command(name = "validate-golden46")]
    ValidateGolden46 {
        #[arg(long, default_value = "golden46-new3-20260723-v4")]
        run_id: String,
        #[arg(long)]
        transfer_arm: bool,
    },
    /// Execute one 80-candidate SCQ80 node batch.
    #[command(name = "golden42-aggressive-batch-worker", hide = true)]
    Golden42AggressiveBatchWorker {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    /// Freeze the watched analysis archive into a deduplicated score-training corpus.
    #[command(name = "materialize-score-corpus")]
    MaterializeScoreCorpus {
        #[arg(long)]
        archive_root: PathBuf,
        #[arg(long)]
        output_root: PathBuf,
        #[arg(long)]
        cohort: Option<String>,
    },
    /// Validate a continuous-learning authority artifact without mutating state.
    #[command(name = "validate-ratchet-artifact")]
    ValidateRatchetArtifact {
        #[arg(long)]
        input: PathBuf,
    },
    /// Apply Rust's fixed 32/16/8/8 retention to a Python v2 seed response.
    #[command(name = "ratchet-retain-model-seeds")]
    RatchetRetainModelSeeds {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        response: PathBuf,
        /// JSON array of semantic SHA-256 identities already reserved by the archive.
        #[arg(long)]
        excluded_semantics: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Transactionally advance one authoritative DatasetLearningState.v2 pointer.
    #[command(name = "ratchet-commit-dataset-state")]
    RatchetCommitDatasetState {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        state: PathBuf,
        #[arg(long)]
        committed_at_unix_seconds: u64,
    },
    /// Publish a non-serving training head after one valid 100-step micro-batch.
    #[command(name = "ratchet-publish-training-head")]
    RatchetPublishTrainingHead {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        micro_batch: PathBuf,
        #[arg(long)]
        replay: PathBuf,
    },
    /// Atomically publish a serving wave after all dependency and evidence gates pass.
    #[command(name = "ratchet-publish-serving-wave")]
    RatchetPublishServingWave {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        rollout_switches: PathBuf,
        #[arg(long)]
        frozen_splits: PathBuf,
    },
    /// Admit one attempt against the latest serving wave and exact dataset state.
    #[command(name = "ratchet-admit-attempt")]
    RatchetAdmitAttempt {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        generation_zero: PathBuf,
        #[arg(long)]
        admission: PathBuf,
    },
    /// Emit bit-parity-gated fitted-C++17 genome prediction feature blocks
    /// (test-adaptive, benchmark-invalid) for the genome-boost campaign.
    #[command(name = "genome-boost-features")]
    GenomeBoostFeatures {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    /// Re-hash emitted genome-boost feature blocks and re-check their receipts.
    #[command(name = "validate-genome-boost-features")]
    ValidateGenomeBoostFeatures {
        #[arg(long)]
        output_dir: PathBuf,
    },
}

#[derive(Debug)]
struct ScoredDataset {
    dataset_hash: String,
    predictions: Vec<f64>,
    prediction_bytes: Vec<u8>,
    prediction_sha256: String,
    report_bytes: Vec<u8>,
    report_sha256: String,
    scores: BTreeMap<String, f64>,
    outcome: Value,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.kernel_version == KernelVersion::AdvancedV3 && command_starts_campaign(&cli.command) {
        bail!(
            "advanced-v3 campaign launch is admission-gated; use a fresh run only after the v3 benchmark gate has been recorded (existing manifests remain legacy-v2)"
        );
    }
    match cli.command {
        Command::VerifyInputs => verify_inputs(&cli.registry),
        Command::SelectPanel => select_panel(&cli.registry, &cli.panel_id, cli.dataset.as_deref()),
        Command::ValidatePanel => validate_panel(&cli.registry, &cli.panel_id),
        Command::ScorePanel => score_panel(&cli.registry, &cli.panel_id),
        Command::PrepareTrainOnlyStudy {
            run_id,
            study_root,
            checkpoint_pointer,
            checkpoint_provenance,
            catalog_pointer,
            observation_db,
            dl_proposals,
            proposal_generator,
            python,
            proposal_device,
            sotarena_root,
            workers,
            smoke,
            proposal_manifest_only,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("prepare-train-only-study is pinned to the legacy-v2 evaluator");
            }
            let mut args =
                commands::train_only_study::PrepareArgs::production(run_id, Some(study_root));
            args.checkpoint_pointer = checkpoint_pointer;
            args.checkpoint_provenance = checkpoint_provenance;
            args.catalog_pointer = catalog_pointer;
            args.observation_db = observation_db;
            args.dl_proposals = dl_proposals;
            args.proposal_generator = proposal_generator;
            args.python = python;
            args.proposal_device = proposal_device;
            args.sotarena_root = sotarena_root;
            args.workers = workers;
            args.smoke = smoke;
            args.proposal_manifest_only = proposal_manifest_only;
            let path = commands::train_only_study::prepare(&cli.registry, args)?;
            println!("{}", path.display());
            Ok(())
        }
        Command::RunTrainOnlyStudy {
            manifest,
            resume,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("run-train-only-study is pinned to the legacy-v2 evaluator");
            }
            let path = commands::train_only_study::run(
                &manifest,
                resume,
                workers,
                cli.dataset.as_deref(),
            )?;
            println!("{}", path.display());
            Ok(())
        }
        Command::ValidateTrainOnlyStudy {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("validate-train-only-study is pinned to the legacy-v2 evaluator");
            }
            let path = commands::train_only_study::validate(
                &study_root,
                &run_id,
                &sotarena_root,
                workers,
                cli.dataset.as_deref(),
            )?;
            println!("{}", path.display());
            Ok(())
        }
        Command::ScoreTrainOnlyStudy {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("score-train-only-study is pinned to the legacy-v2 evaluator");
            }
            let path =
                commands::train_only_study::score(&study_root, &run_id, &sotarena_root, workers)?;
            println!("{}", path.display());
            Ok(())
        }
        Command::ReportTrainOnlyStudy {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("report-train-only-study is pinned to the legacy-v2 evaluator");
            }
            let path =
                commands::train_only_study::report(&study_root, &run_id, &sotarena_root, workers)?;
            println!("{}", path.display());
            Ok(())
        }
        Command::PrepareTrainOnlyRedteam {
            run_id,
            study_root,
            sotarena_root,
            parent_run,
            proposals,
            workers,
            smoke,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("prepare-train-only-redteam is pinned to the legacy-v2 evaluator");
            }
            let args = commands::train_only_redteam::PrepareArgs::production(
                run_id,
                study_root,
                sotarena_root,
                parent_run,
                proposals,
                workers,
                smoke,
            );
            let path = commands::train_only_redteam::prepare(args)?;
            println!("{}", path.display());
            Ok(())
        }
        Command::RunTrainOnlyRedteam {
            run_id,
            study_root,
            sotarena_root: _,
            arm,
            selection_mode,
            resume,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("run-train-only-redteam is pinned to the legacy-v2 evaluator");
            }
            let selection_mode = match selection_mode {
                TrainOnlyRedteamSelectionMode::Combined => {
                    commands::train_only_redteam::SelectionMode::Combined
                }
                TrainOnlyRedteamSelectionMode::SmartOnly => {
                    commands::train_only_redteam::SelectionMode::SmartOnly
                }
            };
            let arm = match (selection_mode, arm) {
                (
                    commands::train_only_redteam::SelectionMode::Combined,
                    Some(TrainOnlyRedteamArm::G20Extension),
                ) => commands::train_only_redteam::Arm::G20Extension,
                (
                    commands::train_only_redteam::SelectionMode::Combined,
                    Some(TrainOnlyRedteamArm::SmartV3),
                )
                | (
                    commands::train_only_redteam::SelectionMode::SmartOnly,
                    None | Some(TrainOnlyRedteamArm::SmartV3),
                ) => commands::train_only_redteam::Arm::SmartV3,
                (commands::train_only_redteam::SelectionMode::Combined, None) => {
                    bail!("combined selection mode requires --arm")
                }
                (
                    commands::train_only_redteam::SelectionMode::SmartOnly,
                    Some(TrainOnlyRedteamArm::G20Extension),
                ) => bail!("smart-only selection mode runs only the Smart-v3 arm"),
            };
            let manifest = study_root.join(run_id).join("manifest.redteam.v1.json");
            let path = commands::train_only_redteam::run(
                &manifest,
                arm,
                selection_mode,
                resume,
                workers,
                cli.dataset.as_deref(),
            )?;
            println!("{}", path.display());
            Ok(())
        }
        Command::ValidateTrainOnlyRedteam {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("validate-train-only-redteam is pinned to the legacy-v2 evaluator");
            }
            let path = commands::train_only_redteam::validate(
                &study_root,
                &run_id,
                &sotarena_root,
                workers,
                cli.dataset.as_deref(),
            )?;
            println!("{}", path.display());
            Ok(())
        }
        Command::ScoreTrainOnlyRedteam {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("score-train-only-redteam is pinned to the legacy-v2 evaluator");
            }
            let path =
                commands::train_only_redteam::score(&study_root, &run_id, &sotarena_root, workers)?;
            println!("{}", path.display());
            Ok(())
        }
        Command::ScoreSmartV3Redteam {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("score-smart-v3-redteam is pinned to the legacy-v2 evaluator");
            }
            let path = commands::train_only_redteam::score_smart_v3(
                &study_root,
                &run_id,
                &sotarena_root,
                workers,
            )?;
            println!("{}", path.display());
            Ok(())
        }
        Command::ReportTrainOnlyRedteam {
            run_id,
            study_root,
            sotarena_root,
            workers,
        } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("report-train-only-redteam is pinned to the legacy-v2 evaluator");
            }
            let path = commands::train_only_redteam::report(
                &study_root,
                &run_id,
                &sotarena_root,
                workers,
            )?;
            println!("{}", path.display());
            Ok(())
        }
        Command::VerifyEvaluationV2 { manifest } => {
            if cli.kernel_version != KernelVersion::LegacyV2 {
                bail!("verify-evaluation-v2 is pinned to the legacy-v2 evaluator");
            }
            commands::exact_cv_v2::verify_manifest(&manifest, &cli.registry, cli.dataset.as_deref())
        }
        Command::VerifyInputsBinary => verify_inputs_binary(&cli.registry),
        Command::SelectPanelBinary => {
            select_panel_binary(&cli.registry, &cli.panel_id, cli.dataset.as_deref())
        }
        Command::ValidatePanelBinary => validate_panel_binary(&cli.registry, &cli.panel_id),
        Command::ScorePanelBinary => score_panel_binary(&cli.registry, &cli.panel_id),
        Command::RunBinaryTestAdaptive { run_id, resume } => {
            let dataset = cli
                .dataset
                .as_deref()
                .context("run-binary-test-adaptive requires --dataset")?;
            test_adaptive::run_binary42(&run_id, dataset, resume)
        }
        Command::ValidateBinaryTestAdaptive { run_id } => {
            let dataset = cli
                .dataset
                .as_deref()
                .context("validate-binary-test-adaptive requires --dataset")?;
            test_adaptive::validate_binary42(&run_id, dataset)
        }
        Command::RunBinaryTestAdaptiveAll { run_id, resume } => {
            if cli.dataset.is_some() {
                bail!("run-binary-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::run_binary42_all(&run_id, resume)
        }
        Command::ValidateBinaryTestAdaptiveAll { run_id } => {
            if cli.dataset.is_some() {
                bail!("validate-binary-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::validate_binary42_all(&run_id)
        }
        Command::PlanMulticlassTestAdaptive {
            run_id,
            campaign_root,
            corpus_manifest,
            comparator_root,
            seed_catalog,
            population,
            generations,
        } => {
            if cli.dataset.is_some() {
                bail!("plan-multiclass-test-adaptive does not accept --dataset");
            }
            if population != 60 || generations != 20 {
                bail!(
                    "multiclass all-dataset campaign is frozen at population=60 and generations=20"
                );
            }
            test_adaptive::plan_multiclass60(
                &run_id,
                &campaign_root,
                &corpus_manifest,
                &comparator_root,
                &seed_catalog,
            )
        }
        Command::RunMulticlassTestAdaptiveAll {
            run_id,
            campaign_root,
            resume,
        } => {
            if cli.dataset.is_some() {
                bail!("run-multiclass-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::run_multiclass60_all(&run_id, &campaign_root, resume)
        }
        Command::ValidateMulticlassTestAdaptiveAll {
            run_id,
            campaign_root,
        } => {
            if cli.dataset.is_some() {
                bail!("validate-multiclass-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::validate_multiclass60_all(&run_id, &campaign_root)
        }
        Command::RunTimeseriesTestAdaptiveAll { run_id, resume } => {
            if cli.dataset.is_some() {
                bail!("run-timeseries-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::run_timeseries42_all(&run_id, resume)
        }
        Command::ValidateTimeseriesTestAdaptiveAll { run_id } => {
            if cli.dataset.is_some() {
                bail!("validate-timeseries-test-adaptive-all does not accept --dataset");
            }
            test_adaptive::validate_timeseries42_all(&run_id)
        }
        Command::RunTimeseriesTestAdaptiveGen30 {
            base_run_id,
            run_id,
            resume,
        } => {
            if cli.dataset.is_some() {
                bail!("run-timeseries-test-adaptive-gen30 does not accept --dataset");
            }
            test_adaptive::run_timeseries42_gen30(&base_run_id, &run_id, resume)
        }
        Command::ValidateTimeseriesTestAdaptiveGen30 {
            base_run_id,
            run_id,
        } => {
            if cli.dataset.is_some() {
                bail!("validate-timeseries-test-adaptive-gen30 does not accept --dataset");
            }
            test_adaptive::validate_timeseries42_gen30(&base_run_id, &run_id)
        }
        Command::RunTimeseriesTestAdaptiveWinall {
            source_run_id,
            run_id,
            max_generations,
            lineage_salt,
            resume,
        } => {
            if cli.dataset.is_some() {
                bail!("run-timeseries-test-adaptive-winall does not accept --dataset");
            }
            test_adaptive::run_timeseries42_winall(
                &source_run_id,
                &run_id,
                resume,
                max_generations,
                lineage_salt.as_deref(),
            )
        }
        Command::BinaryTestAdaptiveBatchWorker {
            manifest,
            output_dir,
        } => test_adaptive::binary42_batch_worker(&manifest, &output_dir),
        Command::TimeseriesTestAdaptiveBatchWorker {
            manifest,
            output_dir,
            corpus_eval,
            timeout_seconds,
        } => test_adaptive::timeseries42_batch_worker(
            &manifest,
            &output_dir,
            &corpus_eval,
            timeout_seconds,
        ),
        Command::RunPanel => {
            select_panel(&cli.registry, &cli.panel_id, cli.dataset.as_deref())?;
            score_panel(&cli.registry, &cli.panel_id)
        }
        Command::Summarize => summarize(&cli.registry, &cli.panel_id),
        Command::RunLosses3x => followup::run(&cli.registry),
        Command::ValidateLosses3x => followup::validate(&cli.registry),
        Command::RunTestAdaptive { run_id, resume } => {
            test_adaptive::run(&cli.registry, &run_id, resume)
        }
        Command::RunTestAdaptivePanel { run_id, resume } => {
            test_adaptive::run_panel(&cli.registry, &run_id, resume)
        }
        Command::RunTestAdaptiveExtension { run_id, resume } => {
            test_adaptive::run_extension(&cli.registry, &run_id, resume)
        }
        Command::RunTestAdaptiveIslandExtension { run_id, resume } => {
            test_adaptive::run_island_extension(&cli.registry, &run_id, resume)
        }
        Command::RunGolden42Champions { run_id, resume } => {
            test_adaptive::run_champion_continuation(&cli.registry, &run_id, resume)
        }
        Command::RunTabarenaLiteIslands { run_id, resume } => {
            test_adaptive::run_tabarena_lite_islands(&run_id, resume, cli.dataset.as_deref())
        }
        Command::RecoverTabarenaLiteResult { run_id, dataset } => {
            test_adaptive::recover_tabarena_lite_result(&run_id, &dataset)
        }
        Command::TestAdaptiveWorker { request } => test_adaptive::worker(&request),
        Command::TestAdaptiveIslandBatchWorker {
            manifest,
            output_dir,
        } => test_adaptive::island_batch_worker(&manifest, &output_dir),
        Command::ValidateTestAdaptive { run_id } => test_adaptive::validate(&cli.registry, &run_id),
        Command::ValidateTestAdaptivePanel { run_id } => {
            test_adaptive::validate_panel(&cli.registry, &run_id)
        }
        Command::ValidateTestAdaptiveExtension { run_id } => {
            test_adaptive::validate_extension(&cli.registry, &run_id)
        }
        Command::ValidateTestAdaptiveIslandExtension { run_id } => {
            test_adaptive::validate_island_extension(&cli.registry, &run_id)
        }
        Command::ValidateGolden42Champions { run_id } => {
            test_adaptive::validate_champion_continuation(&cli.registry, &run_id)
        }
        Command::ValidateTabarenaLiteIslands { run_id } => {
            test_adaptive::validate_tabarena_lite_islands(&run_id)
        }
        Command::BenchLanguagesV3 { output } => aggressive::bench_languages(&output),
        Command::AdmitFittedBackendV1 { output } => {
            aggressive::write_fitted_backend_admission(&output)
        }
        Command::AdmitFittedCppV3 { output } => golden42_v3::admit(&output),
        Command::RunGolden42V3Max {
            run_id,
            dataset,
            node,
            workers,
            population,
            resume,
        } => {
            if cli.kernel_version != KernelVersion::AdvancedV3 {
                bail!("run-golden42-v3-max requires --kernel-version advanced-v3");
            }
            golden42_v3::run(
                &cli.registry,
                &run_id,
                &dataset,
                &node,
                workers,
                population,
                resume,
            )
        }
        Command::ValidateGolden42V3Max { run_id } => golden42_v3::validate(&run_id),
        Command::Golden42V3PersistentWorker { bootstrap } => {
            golden42_v3::persistent_worker(&bootstrap)
        }
        Command::Golden42V3Confirm { request } => golden42_v3::confirm(&request),
        Command::PlanGolden42Aggressive { output } => aggressive::preflight_contract(&output),
        Command::RunGolden42Aggressive {
            run_id,
            dataset,
            start_index,
            resume,
        } => test_adaptive::run_aggressive(
            &cli.registry,
            &run_id,
            dataset.as_deref(),
            start_index,
            resume,
        ),
        Command::ValidateGolden42Aggressive { run_id } => {
            test_adaptive::validate_aggressive(&cli.registry, &run_id)
        }
        Command::RunGolden46 {
            run_id,
            dataset,
            transfer_arm,
            resume,
        } => test_adaptive::run_golden46(
            &cli.registry,
            &run_id,
            dataset.as_deref(),
            transfer_arm,
            resume,
        ),
        Command::ValidateGolden46 {
            run_id,
            transfer_arm,
        } => test_adaptive::validate_golden46(&cli.registry, &run_id, transfer_arm),
        Command::Golden42AggressiveBatchWorker {
            manifest,
            output_dir,
        } => test_adaptive::aggressive_batch_worker(&manifest, &output_dir),
        Command::MaterializeScoreCorpus {
            archive_root,
            output_root,
            cohort,
        } => score_corpus::materialize(&archive_root, &output_root, cohort.as_deref()),
        Command::ValidateRatchetArtifact { input } => ratchet::validate_artifact_file(&input),
        Command::RatchetRetainModelSeeds {
            request,
            response,
            excluded_semantics,
            output,
        } => ratchet::retain_model_seeds_from_files(
            &request,
            &response,
            &excluded_semantics,
            &output,
        ),
        Command::RatchetCommitDatasetState {
            root,
            state,
            committed_at_unix_seconds,
        } => ratchet::commit_dataset_state_from_file(&root, &state, committed_at_unix_seconds),
        Command::RatchetPublishTrainingHead {
            root,
            manifest,
            micro_batch,
            replay,
        } => ratchet::publish_training_head_from_files(&root, &manifest, &micro_batch, &replay),
        Command::RatchetPublishServingWave {
            root,
            manifest,
            rollout_switches,
            frozen_splits,
        } => ratchet::publish_serving_wave_from_files(
            &root,
            &manifest,
            &rollout_switches,
            &frozen_splits,
        ),
        Command::RatchetAdmitAttempt {
            root,
            generation_zero,
            admission,
        } => ratchet::admit_attempt_from_files(&root, &generation_zero, &admission),
        Command::GenomeBoostFeatures {
            manifest,
            output_dir,
        } => genome_boost::run_features(&manifest, &output_dir),
        Command::ValidateGenomeBoostFeatures { output_dir } => {
            genome_boost::validate_features(&output_dir)
        }
    }
}

fn command_starts_campaign(command: &Command) -> bool {
    matches!(
        command,
        Command::SelectPanel
            | Command::RunPanel
            | Command::PrepareTrainOnlyStudy { .. }
            | Command::RunTrainOnlyStudy { .. }
            | Command::PrepareTrainOnlyRedteam { .. }
            | Command::RunTrainOnlyRedteam { .. }
            | Command::RunLosses3x
            | Command::RunTimeseriesTestAdaptiveAll { .. }
            | Command::RunTimeseriesTestAdaptiveGen30 { .. }
            | Command::TimeseriesTestAdaptiveBatchWorker { .. }
            | Command::RunTestAdaptive { .. }
            | Command::RunTestAdaptivePanel { .. }
            | Command::RunTestAdaptiveExtension { .. }
            | Command::RunTestAdaptiveIslandExtension { .. }
            | Command::RunGolden42Champions { .. }
            | Command::RunTabarenaLiteIslands { .. }
            | Command::TestAdaptiveWorker { .. }
            | Command::TestAdaptiveIslandBatchWorker { .. }
            | Command::RunGolden42Aggressive { .. }
            | Command::RunGolden46 { .. }
            | Command::Golden42AggressiveBatchWorker { .. }
    )
}

fn manifest_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("manifests")
}

fn run_root(panel_id: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("runs")
        .join(panel_id)
}

fn seed_bytes() -> &'static [u8] {
    include_bytes!("../seeds/generation-zero.txt")
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn hash_file(path: &Path) -> Result<String> {
    Ok(sha256_bytes(&fs::read(path)?))
}

fn canonical_json(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn dataset_seed(hash: &str) -> Result<u64> {
    Ok(BASE_SEED ^ u64::from_str_radix(hash, 16).context("dataset hash is not hexadecimal")?)
}

/// Binary-lane seed derivation: BASE_SEED ^ lane tag ^ dataset hash, so the
/// binary offspring streams never coincide with the regression lane's.
fn binary_dataset_seed(hash: &str) -> Result<u64> {
    Ok(BASE_SEED
        ^ BINARY_LANE_TAG
        ^ u64::from_str_radix(hash, 16).context("dataset hash is not hexadecimal")?)
}

fn binary_seed_bytes() -> &'static [u8] {
    include_bytes!("../seeds/generation-zero-binary.txt")
}

/// Pinned binary panel: manifest sha + seed sha verified before any campaign.
fn binary_panel() -> Result<Vec<String>> {
    if sha256_bytes(binary_seed_bytes()) != commands::binary_cv::SEED_BINARY_SHA256 {
        bail!("binary seed population hash mismatch");
    }
    let path = manifest_dir().join("panel-binary.txt");
    if hash_file(&path)? != PANEL_BINARY_SHA256 {
        bail!("panel-binary.txt SHA-256 mismatch");
    }
    let text = fs::read_to_string(&path)?;
    let hashes = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect::<Vec<_>>();
    if hashes.is_empty() {
        bail!("panel-binary.txt lists no datasets");
    }
    Ok(hashes)
}

/// Baseline AUC table from the pinned binary panel analysis.
fn binary_baselines() -> Result<BTreeMap<String, BTreeMap<String, f64>>> {
    let path = manifest_dir().join("panel-analysis-binary.v1.json");
    if hash_file(&path)? != PANEL_ANALYSIS_BINARY_SHA256 {
        bail!("panel-analysis-binary.v1.json SHA-256 mismatch");
    }
    let value: Value = serde_json::from_slice(&fs::read(&path)?)?;
    let mut table = BTreeMap::new();
    for entry in value["datasets"].as_array().context("datasets array")? {
        let hash = entry["dataset_hash"]
            .as_str()
            .context("dataset_hash")?
            .to_string();
        let mut aucs = BTreeMap::new();
        for (model, score) in entry["auc"].as_object().context("auc table")? {
            aucs.insert(model.clone(), score.as_f64().context("auc value")?);
        }
        table.insert(hash, aucs);
    }
    Ok(table)
}

fn verify_inputs_binary(registry: &Path) -> Result<()> {
    for hash in binary_panel()? {
        // Training-only firewall: test.csv is never opened here.
        let train_path = registry.join(&hash).join("train.csv");
        let train = read_binary_csv(&train_path)?;
        let positives = train.y.iter().filter(|v| **v == 1.0).count();
        println!(
            "{hash} train_rows={} features={} positives={} negatives={} train_sha256={}",
            train.y.len(),
            train.features,
            positives,
            train.y.len() - positives,
            hash_file(&train_path)?
        );
    }
    println!("verified binary selection inputs without opening test data or baseline metrics");
    Ok(())
}

fn select_panel_binary(registry: &Path, panel_id: &str, only: Option<&str>) -> Result<()> {
    let panel = binary_panel()?;
    if only.is_some_and(|wanted| !panel.iter().any(|hash| hash == wanted)) {
        bail!("requested dataset is absent from the pinned binary panel");
    }
    let root = run_root(panel_id);
    fs::create_dir_all(&root)?;
    let started = Instant::now();
    for (index, hash) in panel.iter().enumerate() {
        if only.is_some_and(|wanted| wanted != hash.as_str()) {
            continue;
        }
        let train_path = registry.join(hash).join("train.csv");
        let train = read_binary_csv(&train_path)?;
        let train_sha256 = hash_file(&train_path)?;
        let artifact = root.join(hash);
        if artifact.join("SELECTION_COMPLETE").exists() {
            commands::binary_cv::validate_freeze_binary(
                &artifact,
                hash,
                &train_sha256,
                train.features,
                ORIGINAL_GENERATIONS,
            )
            .with_context(|| format!("existing binary freeze is invalid for {hash}"))?;
            continue;
        }
        if artifact.exists() && fs::read_dir(&artifact)?.next().transpose()?.is_some() {
            bail!(
                "incomplete nonempty binary campaign directory requires operator recovery: {}",
                artifact.display()
            );
        }
        eprintln!(
            "[jope-dime] selecting binary {hash} ({}/{})",
            index + 1,
            panel.len()
        );
        commands::binary_cv::select_dataset_binary(
            &train,
            hash,
            &train_sha256,
            binary_dataset_seed(hash)?,
            binary_seed_bytes(),
            &artifact,
            ORIGINAL_GENERATIONS,
        )?;
        commands::binary_cv::validate_freeze_binary(
            &artifact,
            hash,
            &train_sha256,
            train.features,
            ORIGINAL_GENERATIONS,
        )?;
    }
    println!(
        "binary selection pass complete in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn validate_panel_binary(registry: &Path, panel_id: &str) -> Result<()> {
    let root = run_root(panel_id);
    for hash in binary_panel()? {
        // Training-only firewall: no test path or baseline metric is named.
        let train_path = registry.join(&hash).join("train.csv");
        let train = read_binary_csv(&train_path)?;
        let train_sha256 = hash_file(&train_path)?;
        commands::binary_cv::validate_freeze_binary(
            &root.join(&hash),
            &hash,
            &train_sha256,
            train.features,
            ORIGINAL_GENERATIONS,
        )
        .with_context(|| format!("binary freeze invalid for {hash}"))?;
        println!("{hash} binary freeze valid");
    }
    println!("validated binary panel without opening test data or baseline metrics");
    Ok(())
}

/// Round-robin Elo over per-dataset AUC tables: strictly greater AUC wins,
/// exact ties draw; K=16, initial 1500, datasets in pinned order and pairs in
/// lexicographic model order (fully deterministic).
fn binary_elo(per_dataset: &[BTreeMap<String, f64>]) -> BTreeMap<String, f64> {
    let mut ratings = BTreeMap::<String, f64>::new();
    for table in per_dataset {
        for model in table.keys() {
            ratings.entry(model.clone()).or_insert(1500.0);
        }
    }
    for table in per_dataset {
        let models = table.keys().cloned().collect::<Vec<_>>();
        for i in 0..models.len() {
            for j in (i + 1)..models.len() {
                let (a, b) = (&models[i], &models[j]);
                let (score_a, score_b) = (table[a], table[b]);
                let outcome_a = if score_a > score_b {
                    1.0
                } else if score_b > score_a {
                    0.0
                } else {
                    0.5
                };
                let rating_a = ratings[a];
                let rating_b = ratings[b];
                let expected_a = 1.0 / (1.0 + 10f64.powf((rating_b - rating_a) / 400.0));
                *ratings.get_mut(a).unwrap() += 16.0 * (outcome_a - expected_a);
                *ratings.get_mut(b).unwrap() += 16.0 * ((1.0 - outcome_a) - (1.0 - expected_a));
            }
        }
    }
    ratings
}

fn score_panel_binary(registry: &Path, panel_id: &str) -> Result<()> {
    let panel = binary_panel()?;
    let baselines = binary_baselines()?;
    let root = run_root(panel_id);
    let mut per_dataset = Vec::new();
    let mut outcomes = Vec::new();
    for hash in &panel {
        let train_path = registry.join(hash).join("train.csv");
        let test_path = registry.join(hash).join("test.csv");
        let train = read_binary_csv(&train_path)?;
        let test = read_binary_csv(&test_path)?;
        let train_sha256 = hash_file(&train_path)?;
        let score = commands::binary_cv::score_dataset_binary(
            &root.join(hash),
            hash,
            &train,
            &train_sha256,
            &test,
            ORIGINAL_GENERATIONS,
        )?;
        let dataset_baselines = baselines
            .get(hash)
            .with_context(|| format!("no baseline AUCs for {hash}"))?;
        let mut table = dataset_baselines.clone();
        table.insert(MODEL_BINARY.to_string(), score.auc);
        let best_baseline = dataset_baselines
            .values()
            .fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        let strict_win = score.auc > best_baseline;
        let mut report = serde_json::to_value(&score)?;
        let object = report.as_object_mut().context("report object")?;
        object.remove("predictions");
        object.insert(
            "schema".to_string(),
            json!("JopeDimeBinaryDatasetReport.v1"),
        );
        object.insert("model".to_string(), json!(MODEL_BINARY));
        object.insert(
            "baseline_auc".to_string(),
            serde_json::to_value(dataset_baselines)?,
        );
        object.insert("strict_win".to_string(), json!(strict_win));
        let report_bytes = canonical_json(&report)?;
        let report_path = root.join(hash).join(BINARY_REPORT_FILE);
        if !report_path.exists() {
            fs::write(&report_path, &report_bytes)?;
        }
        let mut predictions_text = String::new();
        for value in &score.predictions {
            predictions_text.push_str(&format!("{value:?}\n"));
        }
        let prediction_path = root.join(hash).join(BINARY_PREDICTION_FILE);
        if !prediction_path.exists() {
            fs::write(&prediction_path, predictions_text)?;
        }
        println!(
            "{hash} auc={:.6} best_baseline={best_baseline:.6} strict_win={strict_win}",
            score.auc
        );
        outcomes.push(json!({
            "dataset_hash": hash,
            "auc": score.auc,
            "best_baseline_auc": best_baseline,
            "strict_win": strict_win,
        }));
        per_dataset.push(table);
    }
    let ratings = binary_elo(&per_dataset);
    let wins = outcomes
        .iter()
        .filter(|o| o["strict_win"].as_bool() == Some(true))
        .count();
    let results = json!({
        "schema": "JopeDimeBinaryPanelResults.v1",
        "panel_id": panel_id,
        "model": MODEL_BINARY,
        "authority": "test AUC, strictly greater wins; exact ties draw",
        "strict_wins": wins,
        "strict_losses": outcomes.len() - wins,
        "outcomes": outcomes,
        "elo": ratings,
    });
    let results_path = root.join(BINARY_RESULTS_FILE);
    fs::write(&results_path, canonical_json(&results)?)?;
    println!(
        "binary panel scored: {wins}/{} strict wins; results at {}",
        panel.len(),
        results_path.display()
    );
    Ok(())
}

fn validate_static_inputs() -> Result<()> {
    if sha256_bytes(seed_bytes()) != SEED_SHA256 {
        bail!("seed population hash mismatch");
    }
    let panel_path = manifest_dir().join("panel.txt");
    if hash_file(&panel_path)? != PANEL_SHA256 {
        bail!("panel text SHA-256 mismatch");
    }
    let panel = fs::read_to_string(panel_path)?;
    let listed = panel.lines().collect::<Vec<_>>();
    if listed != DATASETS {
        bail!("compiled dataset order differs from the pinned panel");
    }
    Ok(())
}

fn verify_inputs(registry: &Path) -> Result<()> {
    validate_static_inputs()?;
    for hash in DATASETS {
        let train_path = registry.join(hash).join("train.csv");
        let train = read_regression_csv(&train_path)?;
        if train.y.len() < 7 {
            bail!("{hash} has too few train rows for balanced CV7");
        }
        let digest = hash_file(&train_path)?;
        println!(
            "{hash} train_rows={} features={} train_sha256={digest}",
            train.y.len(),
            train.features
        );
    }
    println!("verified 42 selection inputs without opening test data or baseline metrics");
    Ok(())
}

fn select_panel(registry: &Path, panel_id: &str, only: Option<&str>) -> Result<()> {
    validate_static_inputs()?;
    if only.is_some_and(|wanted| !DATASETS.contains(&wanted)) {
        bail!("requested dataset is absent from the pinned panel");
    }
    let root = run_root(panel_id);
    fs::create_dir_all(&root)?;
    let mut core_plan = None;
    let started = Instant::now();
    for (index, hash) in DATASETS.iter().enumerate() {
        if only.is_some_and(|wanted| wanted != *hash) {
            continue;
        }
        let train_path = registry.join(hash).join("train.csv");
        let train = read_regression_csv(&train_path)?;
        let train_sha256 = hash_file(&train_path)?;
        let seed = dataset_seed(hash)?;
        let artifact = root.join(hash);
        if artifact.join("SELECTION_COMPLETE").exists()
            || artifact.join("selection-freeze.v1.json").exists()
        {
            exact_cv::validate_freeze(
                &artifact,
                hash,
                &train_sha256,
                train.y.len(),
                seed,
                ORIGINAL_GENERATIONS,
            )
            .with_context(|| format!("existing immutable freeze is invalid for {hash}"))?;
            continue;
        }
        if artifact.exists() && fs::read_dir(&artifact)?.next().transpose()?.is_some() {
            bail!(
                "incomplete nonempty campaign directory requires operator recovery: {}",
                artifact.display()
            );
        }
        let cores = match &core_plan {
            Some(cores) => cores,
            None => core_plan.insert(require_core_plan(60)?),
        };
        eprintln!(
            "[jope-dime] selecting {hash} ({}/{})",
            index + 1,
            DATASETS.len()
        );
        exact_cv::select_dataset(
            &train,
            hash,
            &train_sha256,
            seed,
            seed_bytes(),
            &artifact,
            cores.clone(),
            ORIGINAL_GENERATIONS,
        )?;
        exact_cv::validate_freeze(
            &artifact,
            hash,
            &train_sha256,
            train.y.len(),
            seed,
            ORIGINAL_GENERATIONS,
        )?;
    }
    println!(
        "selection pass complete in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn load_validated_panel(
    registry: &Path,
    panel_id: &str,
) -> Result<BTreeMap<String, ValidatedFreeze>> {
    validate_static_inputs()?;
    let root = run_root(panel_id);
    let mut frozen = BTreeMap::new();
    let mut semantics = BTreeSet::new();
    let mut slots = 0usize;
    let mut finalists = 0usize;
    for hash in DATASETS {
        // Training-only firewall: no test path or baseline manifest is named here.
        let train_path = registry.join(hash).join("train.csv");
        let train = read_regression_csv(&train_path)?;
        let train_sha256 = hash_file(&train_path)?;
        let validated = exact_cv::validate_freeze(
            &root.join(hash),
            hash,
            &train_sha256,
            train.y.len(),
            dataset_seed(hash)?,
            ORIGINAL_GENERATIONS,
        )
        .with_context(|| format!("selection freeze validation failed for {hash}"))?;
        if !semantics.insert(validated.semantic_sha256.clone()) {
            bail!("selected semantic genome is reused across datasets");
        }
        slots += validated.evolutionary_slots;
        finalists += validated.finalist_receipts;
        frozen.insert((*hash).to_string(), validated);
    }
    if frozen.len() != 42 || slots != 25_200 || finalists != 420 || semantics.len() != 42 {
        bail!(
            "panel accounting failed: freezes={} slots={slots} finalists={finalists} distinct_genomes={}",
            frozen.len(),
            semantics.len()
        );
    }
    Ok(frozen)
}

fn validate_panel(registry: &Path, panel_id: &str) -> Result<()> {
    let frozen = load_validated_panel(registry, panel_id)?;
    println!(
        "validated {} independent freezes, 25200 evolutionary slots, 420 complete CV7 finalist receipts",
        frozen.len()
    );
    Ok(())
}

fn load_baselines() -> Result<BTreeMap<String, BTreeMap<String, f64>>> {
    let path = manifest_dir().join("panel-analysis.v1.json");
    if hash_file(&path)? != PANEL_ANALYSIS_SHA256 {
        bail!("baseline panel SHA-256 mismatch");
    }
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    let rows = value["datasets"]
        .as_array()
        .context("baseline panel datasets are missing")?;
    let mut result = BTreeMap::new();
    for row in rows {
        let hash = row["dataset_hash"]
            .as_str()
            .context("baseline dataset hash is missing")?;
        if !DATASETS.contains(&hash) {
            continue;
        }
        let rmse = row["rmse"]
            .as_object()
            .context("baseline RMSE map is missing")?;
        let mut scores = BTreeMap::new();
        for model in BASELINES {
            let score = rmse
                .get(*model)
                .and_then(Value::as_f64)
                .with_context(|| format!("missing {model} RMSE for {hash}"))?;
            if !score.is_finite() || score < 0.0 {
                bail!("invalid {model} RMSE for {hash}");
            }
            scores.insert((*model).to_string(), score);
        }
        if result.insert(hash.to_string(), scores).is_some() {
            bail!("duplicate baseline row for {hash}");
        }
    }
    if result.len() != DATASETS.len() {
        bail!("baseline panel does not contain the exact pinned 42 datasets");
    }
    Ok(result)
}

fn prediction_bytes(predictions: &[f64]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for prediction in predictions {
        if !prediction.is_finite() {
            bail!("cannot publish a non-finite prediction");
        }
        writeln!(bytes, "{prediction:.17}")?;
    }
    Ok(bytes)
}

fn relation(left: f64, right: f64) -> &'static str {
    match left.total_cmp(&right) {
        Ordering::Less => "win",
        Ordering::Equal => "tie",
        Ordering::Greater => "loss",
    }
}

fn baseline_comparisons(
    jope_rmse: f64,
    baselines: &BTreeMap<String, f64>,
) -> Result<(Value, f64, Vec<String>, String, String)> {
    let best = baselines.values().copied().fold(f64::INFINITY, f64::min);
    let best_models = baselines
        .iter()
        .filter_map(|(model, score)| (*score == best).then_some(model.clone()))
        .collect::<Vec<_>>();
    let mut comparisons = Map::new();
    for model in BASELINES {
        let baseline_rmse = baselines[*model];
        comparisons.insert(
            (*model).to_string(),
            json!({
                "rmse": baseline_rmse,
                "baseline_minus_jope_rmse": baseline_rmse - jope_rmse,
                "jope_outcome": relation(jope_rmse, baseline_rmse),
            }),
        );
    }
    Ok((
        Value::Object(comparisons),
        best,
        best_models,
        relation(jope_rmse, best).to_string(),
        relation(jope_rmse, baselines["TabFM ensemble"]).to_string(),
    ))
}

fn score_dataset(
    registry: &Path,
    hash: &str,
    freeze: &ValidatedFreeze,
    baselines: &BTreeMap<String, f64>,
) -> Result<ScoredDataset> {
    // This function is called only after all 42 training-only freezes validate.
    let directory = registry.join(hash);
    let train_path = directory.join("train.csv");
    let test_path = directory.join("test.csv");
    let train = read_regression_csv(&train_path)?;
    let test = read_regression_csv(&test_path)?;
    if train.features != test.features {
        bail!("train/test feature mismatch for {hash}");
    }
    let timed = exact_cv::fit_predict_timed(&freeze.canonical_genome, &train, &test)?;
    let (rmse, r2, mae) = exact_cv::metrics(&test.y, &timed.values)?;
    let prediction_bytes = prediction_bytes(&timed.values)?;
    let prediction_sha256 = sha256_bytes(&prediction_bytes);
    let (comparisons, best_rmse, best_models, overall_outcome, tabfm_outcome) =
        baseline_comparisons(rmse, baselines)?;
    let test_sha256 = hash_file(&test_path)?;
    let report = json!({
        "schema": "JopeDimeDatasetReport.v1",
        "algorithm": {
            "name": MODEL,
            "dataset_specific": true,
            "canonical_genome": freeze.canonical_genome,
            "semantic_sha256": freeze.semantic_sha256,
            "complexity": freeze.complexity,
        },
        "dataset_hash": hash,
        "status": "completed",
        "train_rows": train.y.len(),
        "test_rows": test.y.len(),
        "n_features": train.features,
        "metrics": {"rmse": rmse, "r2": r2, "mae": mae},
        "timings": {
            "fit_seconds": timed.fit_seconds,
            "predict_seconds": timed.predict_seconds,
            "total_seconds": timed.fit_seconds + timed.predict_seconds,
        },
        "prediction": {
            "rows": timed.values.len(),
            "sha256": prediction_sha256,
        },
        "selection_proxy": {
            "authority_label": "CV7 training-only selection proxy; not final test authority",
            "generation": freeze.generation,
            "slot": freeze.slot,
            "ranking_authority": freeze.ranking_authority,
            "quick_cv5_fold0": {
                "pooled_oof_r2": freeze.quick_pooled_oof_r2,
                "pooled_oof_rmse": freeze.quick_pooled_oof_rmse,
                "robust_fitness": freeze.quick_robust_fitness,
                "fold_manifest_sha256": freeze.cv5_fold_manifest_sha256,
            },
            "full_cv7": {
                "fold_r2": freeze.cv7_fold_r2,
                "pooled_oof_r2": freeze.cv7_pooled_oof_r2,
                "pooled_oof_rmse": freeze.cv7_pooled_oof_rmse,
                "robust_fitness": freeze.cv7_robust_fitness,
                "fit_seconds": freeze.cv7_fit_seconds,
                "predict_seconds": freeze.cv7_predict_seconds,
                "fold_manifest_sha256": freeze.cv7_fold_manifest_sha256,
            },
        },
        "comparisons": {
            "baselines": comparisons,
            "best_baseline": {"models": best_models, "rmse": best_rmse},
            "strict_overall": {
                "rule": "win iff jope-dime test RMSE is strictly lower than the best baseline test RMSE",
                "outcome": overall_outcome,
                "margin_best_baseline_minus_jope_rmse": best_rmse - rmse,
            },
            "tabfm_head_to_head": {
                "outcome": tabfm_outcome,
                "tabfm_rmse": baselines["TabFM ensemble"],
                "tabfm_minus_jope_rmse": baselines["TabFM ensemble"] - rmse,
            },
        },
        "provenance": {
            "selection_sha256": freeze.selection_sha256,
            "ledger_sha256": freeze.ledger_sha256,
            "train_sha256": freeze.train_sha256,
            "test_sha256": test_sha256,
            "panel_text_sha256": PANEL_SHA256,
            "baseline_panel_sha256": PANEL_ANALYSIS_SHA256,
            "evolutionary_slots": freeze.evolutionary_slots,
            "cv7_finalist_receipts": freeze.finalist_receipts,
        },
    });
    let report_bytes = canonical_json(&report)?;
    let report_sha256 = sha256_bytes(&report_bytes);
    let mut scores = baselines.clone();
    scores.insert(MODEL.to_string(), rmse);
    let outcome = json!({
        "dataset_hash": hash,
        "genome_semantic_sha256": freeze.semantic_sha256,
        "test_metrics": {"rmse": rmse, "r2": r2, "mae": mae},
        "timings": {
            "fit_seconds": timed.fit_seconds,
            "predict_seconds": timed.predict_seconds,
        },
        "prediction_sha256": prediction_sha256,
        "report_sha256": report_sha256,
        "best_baseline_rmse": best_rmse,
        "best_baseline_models": best_models,
        "strict_overall_outcome": overall_outcome,
        "tabfm_head_to_head_outcome": tabfm_outcome,
        "rmse_by_algorithm": scores,
    });
    Ok(ScoredDataset {
        dataset_hash: hash.to_string(),
        predictions: timed.values,
        prediction_bytes,
        prediction_sha256,
        report_bytes,
        report_sha256,
        scores,
        outcome,
    })
}

fn pairwise_results(scored: &[ScoredDataset]) -> Value {
    let players = players();
    let mut rows = Vec::new();
    for left_index in 0..players.len() {
        for right_index in left_index + 1..players.len() {
            let left = players[left_index];
            let right = players[right_index];
            let mut left_wins = 0usize;
            let mut right_wins = 0usize;
            let mut ties = 0usize;
            for dataset in scored {
                match dataset.scores[left].total_cmp(&dataset.scores[right]) {
                    Ordering::Less => left_wins += 1,
                    Ordering::Equal => ties += 1,
                    Ordering::Greater => right_wins += 1,
                }
            }
            rows.push(json!({
                "left": left,
                "right": right,
                "left_wins": left_wins,
                "right_wins": right_wins,
                "ties": ties,
            }));
        }
    }
    Value::Array(rows)
}

fn players() -> Vec<&'static str> {
    let mut values = BASELINES.to_vec();
    values.push(MODEL);
    values
}

fn elo_standings(scored: &[ScoredDataset]) -> Value {
    const K: f64 = 16.0;
    let players = players();
    let mut ratings = players
        .iter()
        .map(|player| ((*player).to_string(), 1500.0))
        .collect::<BTreeMap<_, _>>();
    for dataset in scored {
        let snapshot = ratings.clone();
        let mut deltas = players
            .iter()
            .map(|player| ((*player).to_string(), 0.0))
            .collect::<BTreeMap<_, _>>();
        for left_index in 0..players.len() {
            for right_index in left_index + 1..players.len() {
                let left = players[left_index];
                let right = players[right_index];
                let left_rating = snapshot[left];
                let right_rating = snapshot[right];
                let expected_left = 1.0 / (1.0 + 10_f64.powf((right_rating - left_rating) / 400.0));
                let actual_left = match dataset.scores[left].total_cmp(&dataset.scores[right]) {
                    Ordering::Less => 1.0,
                    Ordering::Equal => 0.5,
                    Ordering::Greater => 0.0,
                };
                let delta = K * (actual_left - expected_left);
                *deltas.get_mut(left).unwrap() += delta;
                *deltas.get_mut(right).unwrap() -= delta;
            }
        }
        for player in &players {
            *ratings.get_mut(*player).unwrap() += deltas[*player];
        }
    }
    let mut ordered = ratings.into_iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    Value::Array(
        ordered
            .into_iter()
            .enumerate()
            .map(|(index, (algorithm, rating))| {
                json!({"rank": index + 1, "algorithm": algorithm, "rating": rating})
            })
            .collect(),
    )
}

fn publish_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("publication path has no parent")?;
    fs::create_dir_all(parent)?;
    if path.exists() {
        bail!("refusing to overwrite {}", path.display());
    }
    let staged = parent.join(format!(
        ".jope-dime-{}-{}.stage",
        std::process::id(),
        &sha256_bytes(bytes)[..16]
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&staged)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    drop(file);
    let mut permissions = fs::metadata(&staged)?.permissions();
    permissions.set_readonly(true);
    fs::set_permissions(&staged, permissions)?;
    fs::hard_link(&staged, path).with_context(|| format!("publish {}", path.display()))?;
    fs::remove_file(staged)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn ensure_publication_targets_absent(registry: &Path, panel_id: &str) -> Result<()> {
    let root = run_root(panel_id);
    let ledger = root.join("jope-dime-panel-results.v1.json");
    if ledger.exists() {
        bail!("refusing to overwrite {}", ledger.display());
    }
    for hash in DATASETS {
        for path in [
            root.join(hash).join("phase2/predictions.csv"),
            registry
                .join(hash)
                .join("predictions")
                .join(PREDICTION_FILE),
            registry.join(hash).join("reports").join(REPORT_FILE),
        ] {
            if path.exists() {
                bail!("refusing to overwrite {}", path.display());
            }
        }
    }
    Ok(())
}

fn score_panel(registry: &Path, panel_id: &str) -> Result<()> {
    // The baseline manifest and every test.csv remain unopened until this
    // complete panel-wide training-only validation returns successfully.
    let frozen = load_validated_panel(registry, panel_id)?;
    let baselines = load_baselines()?;
    ensure_publication_targets_absent(registry, panel_id)?;

    let mut scored = Vec::with_capacity(DATASETS.len());
    for hash in DATASETS {
        eprintln!("[jope-dime] final scoring {hash}");
        scored.push(score_dataset(
            registry,
            hash,
            &frozen[*hash],
            &baselines[*hash],
        )?);
    }
    if scored.len() != 42 || scored.iter().any(|row| row.predictions.is_empty()) {
        bail!("final scoring did not produce exactly 42 complete outcomes");
    }

    let root = run_root(panel_id);
    for row in &scored {
        let local_prediction = root.join(&row.dataset_hash).join("phase2/predictions.csv");
        let public_prediction = registry
            .join(&row.dataset_hash)
            .join("predictions")
            .join(PREDICTION_FILE);
        let public_report = registry
            .join(&row.dataset_hash)
            .join("reports")
            .join(REPORT_FILE);
        publish_new(&local_prediction, &row.prediction_bytes)?;
        publish_new(&public_prediction, &row.prediction_bytes)?;
        publish_new(&public_report, &row.report_bytes)?;
        if hash_file(&local_prediction)? != row.prediction_sha256
            || hash_file(&public_prediction)? != row.prediction_sha256
            || hash_file(&public_report)? != row.report_sha256
        {
            bail!("published artifact hash mismatch for {}", row.dataset_hash);
        }
    }

    let wins = scored
        .iter()
        .filter(|row| row.outcome["strict_overall_outcome"] == "win")
        .count();
    let losses = scored
        .iter()
        .filter(|row| row.outcome["strict_overall_outcome"] == "loss")
        .count();
    let ties = 42 - wins - losses;
    let tabfm_wins = scored
        .iter()
        .filter(|row| row.outcome["tabfm_head_to_head_outcome"] == "win")
        .count();
    let tabfm_losses = scored
        .iter()
        .filter(|row| row.outcome["tabfm_head_to_head_outcome"] == "loss")
        .count();
    let tabfm_ties = 42 - tabfm_wins - tabfm_losses;
    let tabfm_status = match tabfm_wins.cmp(&tabfm_losses) {
        Ordering::Greater => "winning",
        Ordering::Equal => "tied",
        Ordering::Less => "losing",
    };
    let outcomes = scored
        .iter()
        .map(|row| row.outcome.clone())
        .collect::<Vec<_>>();
    let ledger = json!({
        "schema": "JopeDimePanelResults.v1",
        "panel_id": panel_id,
        "panel_text_sha256": PANEL_SHA256,
        "baseline_panel_sha256": PANEL_ANALYSIS_SHA256,
        "dataset_count": 42,
        "selection_accounting": {
            "independent_freezes": 42,
            "distinct_selected_semantic_genomes": 42,
            "evolutionary_slots": 25200,
            "full_cv7_finalist_receipts": 420,
        },
        "strict_overall": {
            "rule": "win iff jope-dime test RMSE is strictly lower than the best baseline test RMSE",
            "wins": wins,
            "losses": losses,
            "ties": ties,
        },
        "tabfm_head_to_head": {
            "wins": tabfm_wins,
            "losses": tabfm_losses,
            "ties": tabfm_ties,
            "status": tabfm_status,
        },
        "pairwise_results": pairwise_results(&scored),
        "elo": {
            "initial_rating": 1500.0,
            "k_factor": 16.0,
            "method": "Each dataset is a simultaneous 7-algorithm round robin; lower RMSE wins and exact equality ties; panel order is pinned.",
            "standings": elo_standings(&scored),
        },
        "outcomes": outcomes,
    });
    let ledger_path = root.join("jope-dime-panel-results.v1.json");
    publish_new(&ledger_path, &canonical_json(&ledger)?)?;
    println!(
        "published 42 reports and predictions: overall {wins}W/{losses}L/{ties}T; TabFM {tabfm_wins}W/{tabfm_losses}L/{tabfm_ties}T"
    );
    Ok(())
}

fn summarize(registry: &Path, panel_id: &str) -> Result<()> {
    let path = run_root(panel_id).join("jope-dime-panel-results.v1.json");
    let value: Value = serde_json::from_slice(&fs::read(&path)?)?;
    if value["schema"] == "JopeDimePanelResults.v2" {
        if panel_id != "grok4p5-42" {
            bail!("the v2 follow-up ledger is only valid at the original panel path");
        }
        return followup::summarize(registry);
    }
    if value["schema"] != "JopeDimePanelResults.v1"
        || value["dataset_count"].as_u64() != Some(42)
        || value["outcomes"].as_array().map(Vec::len) != Some(42)
        || value["pairwise_results"].as_array().map(Vec::len) != Some(21)
        || value["elo"]["standings"].as_array().map(Vec::len) != Some(7)
        || value["selection_accounting"]["evolutionary_slots"].as_u64() != Some(25_200)
        || value["selection_accounting"]["full_cv7_finalist_receipts"].as_u64() != Some(420)
    {
        bail!("panel ledger is incomplete or invalid");
    }
    for outcome in value["outcomes"].as_array().unwrap() {
        let hash = outcome["dataset_hash"]
            .as_str()
            .context("outcome dataset hash is missing")?;
        if !DATASETS.contains(&hash) {
            bail!("panel ledger contains an unknown dataset");
        }
        let prediction_sha256 = outcome["prediction_sha256"]
            .as_str()
            .context("outcome prediction hash is missing")?;
        let report_sha256 = outcome["report_sha256"]
            .as_str()
            .context("outcome report hash is missing")?;
        let local_prediction = run_root(panel_id).join(hash).join("phase2/predictions.csv");
        let public_prediction = registry
            .join(hash)
            .join("predictions")
            .join(PREDICTION_FILE);
        let public_report = registry.join(hash).join("reports").join(REPORT_FILE);
        if hash_file(&local_prediction)? != prediction_sha256
            || hash_file(&public_prediction)? != prediction_sha256
            || hash_file(&public_report)? != report_sha256
        {
            bail!("published artifact hash mismatch for {hash}");
        }
        let report: Value = serde_json::from_slice(&fs::read(public_report)?)?;
        if report["schema"] != "JopeDimeDatasetReport.v1"
            || report["dataset_hash"].as_str() != Some(hash)
            || report["status"] != "completed"
            || report["prediction"]["sha256"].as_str() != Some(prediction_sha256)
        {
            bail!("published report is invalid for {hash}");
        }
    }
    println!(
        "42 outcomes; strict overall {}W/{}L/{}T; TabFM {}W/{}L/{}T ({})",
        value["strict_overall"]["wins"],
        value["strict_overall"]["losses"],
        value["strict_overall"]["ties"],
        value["tabfm_head_to_head"]["wins"],
        value["tabfm_head_to_head"]["losses"],
        value["tabfm_head_to_head"]["ties"],
        value["tabfm_head_to_head"]["status"]
            .as_str()
            .unwrap_or("unknown"),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn binary_elo_is_deterministic_and_rewards_strictly_greater_auc() {
        let mut dataset = BTreeMap::new();
        dataset.insert("A".to_string(), 0.9);
        dataset.insert("B".to_string(), 0.7);
        let ratings = binary_elo(&[dataset.clone()]);
        // One game: A expected 0.5, wins -> +8; B symmetric -> -8.
        assert_eq!(ratings["A"], 1508.0);
        assert_eq!(ratings["B"], 1492.0);
        // Exact tie moves nothing at equal ratings.
        let mut tie = BTreeMap::new();
        tie.insert("A".to_string(), 0.8);
        tie.insert("B".to_string(), 0.8);
        let tie_ratings = binary_elo(&[tie]);
        assert_eq!(tie_ratings["A"], 1500.0);
        assert_eq!(tie_ratings["B"], 1500.0);
        // Determinism across repeated evaluation.
        assert_eq!(binary_elo(&[dataset.clone()]), binary_elo(&[dataset]));
    }

    #[test]
    fn binary_static_inputs_are_pinned() {
        assert_eq!(
            sha256_bytes(binary_seed_bytes()),
            commands::binary_cv::SEED_BINARY_SHA256
        );
        let panel = binary_panel().unwrap();
        assert_eq!(panel, vec!["b1a0f1c500000001", "b1a0f1c500000002"]);
        let baselines = binary_baselines().unwrap();
        assert_eq!(baselines.len(), 2);
        assert!(baselines["b1a0f1c500000001"]["BaselineBoosted"] > 0.8);
    }

    #[test]
    fn strict_relation_has_no_tolerance() {
        assert_eq!(relation(0.9, 1.0), "win");
        assert_eq!(relation(1.0, 1.0), "tie");
        assert_eq!(relation(1.0 + f64::EPSILON, 1.0), "loss");
    }

    #[test]
    fn baseline_comparison_covers_all_six_models() {
        let baselines = BASELINES
            .iter()
            .enumerate()
            .map(|(index, model)| ((*model).to_string(), 1.0 + index as f64))
            .collect::<BTreeMap<_, _>>();
        let (comparisons, best, models, overall, tabfm) =
            baseline_comparisons(0.5, &baselines).unwrap();
        assert_eq!(comparisons.as_object().unwrap().len(), 6);
        assert_eq!(best, 1.0);
        assert_eq!(models, vec!["TabFM ensemble"]);
        assert_eq!(overall, "win");
        assert_eq!(tabfm, "win");
    }

    #[test]
    fn immutable_publication_refuses_overwrite() {
        let sequence = TEMP_SEQUENCE.fetch_add(1, AtomicOrdering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "jope-dime-publication-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("report.json");
        publish_new(&path, b"first\n").unwrap();
        assert!(publish_new(&path, b"second\n").is_err());
        assert_eq!(fs::read(&path).unwrap(), b"first\n");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn timeseries_gen30_cli_parses_base_and_run_ids() {
        let cli = Cli::try_parse_from([
            "jope-dime-regressor",
            "run-timeseries-test-adaptive-gen30",
            "--base-run-id",
            "base",
            "--run-id",
            "next",
            "--resume",
        ])
        .unwrap();
        match cli.command {
            Command::RunTimeseriesTestAdaptiveGen30 {
                base_run_id,
                run_id,
                resume,
            } => {
                assert_eq!(base_run_id, "base");
                assert_eq!(run_id, "next");
                assert!(resume);
            }
            _ => panic!("wrong command variant"),
        }
    }

    #[test]
    fn timeseries_winall_cli_parses_source_run_id_and_generation_limit() {
        let cli = Cli::try_parse_from([
            "jope-dime-regressor",
            "run-timeseries-test-adaptive-winall",
            "--source-run-id",
            "gen30",
            "--run-id",
            "winall",
            "--max-generations",
            "200",
            "--lineage-salt",
            "salt-a",
            "--resume",
        ])
        .unwrap();
        match cli.command {
            Command::RunTimeseriesTestAdaptiveWinall {
                source_run_id,
                run_id,
                max_generations,
                lineage_salt,
                resume,
            } => {
                assert_eq!(source_run_id, "gen30");
                assert_eq!(run_id, "winall");
                assert_eq!(max_generations, Some(200));
                assert_eq!(lineage_salt.as_deref(), Some("salt-a"));
                assert!(resume);
            }
            _ => panic!("wrong command variant"),
        }
    }
}

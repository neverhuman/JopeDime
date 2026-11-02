use super::*;

#[derive(Parser)]
#[command(name = "jope-dime-regressor")]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
    #[arg(long, global = true, env = "JOPE_REGISTRY", default_value = registry_str())]
    pub(crate) registry: PathBuf,
    #[arg(long, global = true, default_value = "grok4p5-42")]
    pub(crate) panel_id: String,
    #[arg(long, global = true)]
    pub(crate) dataset: Option<String>,
    /// Kernel contract for newly launched work. Existing manifests are
    /// permanently legacy-v2 and cannot be resumed under advanced-v3.
    #[arg(long, global = true, value_enum, default_value = "legacy-v2")]
    pub(crate) kernel_version: KernelVersion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum KernelVersion {
    LegacyV2,
    AdvancedV3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum CliTrainOnlyRedteamArm {
    G20Extension,
    SmartV3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum CliTrainOnlyRedteamSelectionMode {
    Combined,
    SmartOnly,
}

#[derive(Subcommand)]
pub(crate) enum Command {
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
        arm: Option<CliTrainOnlyRedteamArm>,
        #[arg(long, value_enum, default_value = "combined")]
        selection_mode: CliTrainOnlyRedteamSelectionMode,
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

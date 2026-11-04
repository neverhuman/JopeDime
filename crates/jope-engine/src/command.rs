use clap::{Parser, Subcommand};
use std::path::PathBuf;
#[derive(Parser)]
#[command(
    version,
    about = "Real-data migration and isolated Rust policy/GP training"
)]
pub(crate) struct Cli {
    #[arg(long, default_value = ".")]
    pub(crate) root: PathBuf,
    #[command(subcommand)]
    pub(crate) command: EngineCmd,
}
#[derive(Subcommand)]
pub(crate) enum EngineCmd {
    Doctor,
    #[cfg(feature = "neural")]
    GpuWorker {
        #[arg(long)]
        pool: PathBuf,
        #[arg(long)]
        rank: usize,
    },
    #[cfg(feature = "neural")]
    ProposeRanked {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long, default_value_t = 0)]
        rank: usize,
        #[arg(long, default_value_t = 1)]
        world: usize,
    },
    #[cfg(feature = "neural")]
    AssembleRanked {
        #[arg(long)]
        manifest: PathBuf,
    },
    CompareSeeds {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long, default_value_t = 16)]
        workers: usize,
        #[arg(long, default_value_t = 8)]
        prefix: usize,
        #[arg(long, default_value_t = 3600)]
        seconds: u64,
        #[arg(long)]
        downstream: bool,
    },
    #[cfg(feature = "neural")]
    ApplyBinaryContexts {
        #[arg(long)]
        jobs: PathBuf,
        #[arg(long)]
        hyperion: PathBuf,
        #[arg(long)]
        foundation: PathBuf,
        #[arg(long)]
        target: PathBuf,
        #[arg(long)]
        assembly: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        aliases: PathBuf,
    },
    #[cfg(feature = "neural")]
    Propose {
        #[arg(long)]
        generation_manifest: Option<PathBuf>,
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 2)]
        cuts: usize,
        #[arg(long, default_value_t = 0)]
        cut_start: usize,
        #[arg(long)]
        shuffled: bool,
    },
    #[cfg(feature = "neural")]
    #[command(hide = true)]
    ProposeRank {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        rank: usize,
        #[arg(long)]
        world: usize,
        #[arg(long)]
        cuts: usize,
        #[arg(long, default_value_t = 0)]
        cut_start: usize,
        #[arg(long)]
        shuffled: bool,
    },
    #[cfg(feature = "neural")]
    Train {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 8)]
        world: usize,
        #[arg(long, default_value_t = 256)]
        microbatch: usize,
        #[arg(long, default_value_t = 300)]
        steps: usize,
        #[arg(long, default_value_t = 100)]
        checkpoint_steps: usize,
        #[arg(long)]
        resume: bool,
        #[arg(long)]
        initialize: Option<PathBuf>,
        #[arg(long, hide = true)]
        fail_rank: Option<usize>,
        #[arg(long)]
        replay_policy: Option<PathBuf>,
    },
    #[cfg(feature = "neural")]
    #[command(hide = true)]
    TrainRank {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        launch: PathBuf,
        #[arg(long)]
        rank: usize,
        #[arg(long)]
        world: usize,
        #[arg(long)]
        microbatch: usize,
        #[arg(long)]
        steps: usize,
        #[arg(long)]
        checkpoint_steps: usize,
        #[arg(long)]
        resume: bool,
        #[arg(long)]
        initialize: Option<PathBuf>,
        #[arg(long)]
        replay_policy: Option<PathBuf>,
        /// Explicit lineage-mode declaration, validated by the trainer.
        #[arg(long)]
        continuation_mode: Option<String>,
    },
    #[cfg(feature = "neural")]
    #[command(hide = true)]
    QualifyBatchRank {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        launch: PathBuf,
        #[arg(long)]
        rank: usize,
        #[arg(long)]
        microbatch: usize,
        #[arg(long)]
        global_batch: usize,
        #[arg(long)]
        steps: usize,
        #[arg(long)]
        initialize: PathBuf,
    },
    #[cfg(feature = "neural")]
    MonitorCheckpoint {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        panel: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    BuildCorpus {
        /// Publish immutable group shards through this private append-only store.
        #[arg(long, conflicts_with = "scan_cache")]
        teaching_store: Option<PathBuf>,
        /// Production packing requires verified original execution receipts.
        #[arg(long, default_value = "strict_midrank_v4", value_parser = ["reference_v1", "midrank_v2", "strict_midrank_v3", "strict_midrank_v4", "frozen_z_v1"])]
        label_policy: String,
        #[arg(long, requires = "teaching_store")]
        admission_manifest: Option<PathBuf>,
        /// Reuse hash-verified ledger contributions under this private runs directory.
        #[arg(long)]
        scan_cache: Option<PathBuf>,
        /// Publish a live packing heartbeat at this path.
        #[arg(long)]
        progress: Option<PathBuf>,
        #[arg(long)]
        contexts: PathBuf,
        #[arg(long)]
        vocabulary: PathBuf,
        #[arg(long, required_unless_present = "observation_manifest")]
        observations: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        observation_manifest: Option<PathBuf>,
        #[arg(long)]
        qualification: bool,
    },
    ExportTeachingCorpus {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Write a self-contained corpus copy restricted to training and validation contexts.
    ProjectCorpus {
        #[arg(long)]
        source_root: Option<PathBuf>,
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    ComparisonBaseline {
        #[arg(long)]
        jobs: PathBuf,
        #[arg(long)]
        vocabulary: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Enqueue {
        #[arg(long)]
        jobs: PathBuf,
        #[arg(long)]
        contexts: PathBuf,
        /// Frozen campaign index; jobs outside it are refused. Omit for research GP.
        #[arg(long)]
        campaign_contexts: Option<PathBuf>,
    },
    /// Write campaign vs union vs GP-cut coverage without admitting jobs.
    #[command(hide = true)]
    CampaignCoverage {
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        union: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        gp_root: Vec<PathBuf>,
    },
    #[cfg(feature = "neural")]
    AssembleContexts {
        #[arg(long)]
        jobs: PathBuf,
        #[arg(long)]
        hyperion: PathBuf,
        #[arg(long)]
        foundation: PathBuf,
        #[arg(long)]
        target: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        registry: Option<PathBuf>,
        #[arg(long)]
        qualification: bool,
    },
    #[cfg(feature = "neural")]
    ApplyContexts {
        #[arg(long)]
        jobs: PathBuf,
        #[arg(long)]
        hyperion: PathBuf,
        #[arg(long)]
        foundation: PathBuf,
        #[arg(long)]
        target: PathBuf,
        #[arg(long)]
        assembly: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        qualification: bool,
    },
    #[cfg(feature = "neural")]
    UnionContexts {
        #[arg(long, required = true)]
        indexes: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    #[cfg(feature = "neural")]
    FoundationService {
        #[arg(long)]
        jobs: PathBuf,
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 2)]
        batch: usize,
    },
    #[cfg(feature = "neural")]
    #[command(hide = true)]
    FoundationRank {
        #[arg(long)]
        jobs: PathBuf,
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        rank: usize,
        #[arg(long)]
        world: usize,
        #[arg(long, default_value_t = 2)]
        batch: usize,
    },
    #[cfg(feature = "neural")]
    TargetService {
        #[arg(long)]
        jobs: PathBuf,
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 128)]
        batch: usize,
    },
    #[cfg(feature = "neural")]
    #[command(hide = true)]
    TargetRank {
        #[arg(long)]
        jobs: PathBuf,
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        rank: usize,
        #[arg(long)]
        world: usize,
        #[arg(long, default_value_t = 128)]
        batch: usize,
    },
    ExpandCuts {
        #[arg(long)]
        prepared: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 20)]
        rounds: usize,
    },
    IndexCuts {
        #[arg(long)]
        prepared: PathBuf,
        #[arg(long)]
        backing_manifest: PathBuf,
        #[arg(long)]
        metadata: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    #[command(hide = true)]
    EvaluateView {
        #[arg(long)]
        prepared: PathBuf,
        #[arg(long)]
        cut: usize,
        #[arg(long)]
        genome: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    #[command(hide = true)]
    VerifyFitCompletion {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        completion_sha256: String,
        #[arg(long)]
        authority_instance: String,
    },
    #[command(hide = true)]
    RegistryFit {
        #[arg(long)]
        fd: i32,
        #[arg(long)]
        output: PathBuf,
    },
    /// Initialize new held native-fit storage; existing authorities are never overwritten.
    RegistryInitializeNative {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        scopes: PathBuf,
    },
    /// Initialize an empty independent witness store from reviewed public policy.
    RegistryWitnessInitialize {
        #[arg(long)]
        witness_root: PathBuf,
        #[arg(long)]
        policy: PathBuf,
    },
    /// Observe this loaded CPU executable and numerical runtime without authorizing work.
    RegistryInspectRuntime,
    /// Own all physical regression fits through one durable native authority.
    RegistryAuthority {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        socket: PathBuf,
        #[arg(long, default_value_t = 104)]
        workers: usize,
        #[arg(long)]
        event_witness_config: PathBuf,
        #[arg(long)]
        event_witness_config_sha256: String,
        #[arg(long)]
        funding_config: Option<PathBuf>,
        #[arg(long)]
        native_fit_policy: Option<PathBuf>,
    },
    /// Transport a typed funding request through the protected native client.
    FundingRequest {
        #[arg(long, default_value = "/etc/jope-dime/funding/client.json")]
        config: PathBuf,
    },
    /// Score a retained committed model with separately authorized query access.
    QueryEvaluate {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        socket: PathBuf,
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Retrieve an original query result without extending its execution deadline.
    QueryRetrieve {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        socket: PathBuf,
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        deadline_monotonic_ns: u64,
    },
    /// Verify the original public witness ledger without accessing either signing key.
    RegistryWitnessInspect {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        policy_sha256: String,
    },
    /// Retain one authenticated independent witness request from standard input.
    RegistryWitnessRequest {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        key: PathBuf,
    },
    /// Append one independently provisioned exact native grant to the original witness ledger.
    RegistryWitnessAddNativeGrant {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        key: PathBuf,
    },
    /// Append an independently reviewed epoch transition to the original witness ledger.
    RegistryWitnessEpochTransition {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        key: PathBuf,
    },
    /// Retrieve an original witness acknowledgment without recording another event.
    RegistryWitnessGetAck {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        key: PathBuf,
    },
    /// Capture a reviewed inventory without publishing coverage or enabling fits.
    HistoryStage {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        inventory: PathBuf,
    },
    HistoryImportBatch {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        batch: PathBuf,
    },
    /// Explicitly prepare the derived global historical lookup index; no admission.
    HistoryPrepareLookupIndex {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
    },
    HistoryCommitCoverage {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        stage_sha256: String,
    },
    /// Inspect signed preparation identities and the original authority head without issuing authorization.
    NativeScopeInspect {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        overlay: PathBuf,
        #[arg(long)]
        overlay_sha256: String,
        #[arg(long)]
        run: String,
    },
    NativeScopeRegister {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        registration: PathBuf,
        #[arg(long)]
        registration_sha256: String,
    },
    /// Commit complete original namespace coverage from independently reviewed aggregate evidence.
    HistoryCommitAggregateCoverage {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        aggregate: PathBuf,
        #[arg(long)]
        aggregate_sha256: String,
    },
    RecoveryInspect {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
    },
    RecoveryReplay {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
    },
    RecoveryReconcile {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
    },
    RecoveryCommitEpoch {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        report_sha256: String,
    },
    NamespaceActivate {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        scope: String,
        #[arg(long)]
        capability: String,
    },
    NamespaceDeactivate {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        scope: String,
    },
    #[command(hide = true)]
    RegistryServiceFit,
    #[command(hide = true)]
    RegistryServiceQuery,
    #[command(hide = true)]
    EvaluateMemory {
        #[arg(long)]
        fd: i32,
        #[arg(long)]
        genome: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, requires = "cut")]
        prepared: Option<PathBuf>,
        #[arg(long, requires = "prepared")]
        cut: Option<usize>,
    },
    HyperionService {
        #[arg(long)]
        jobs: PathBuf,
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    #[command(hide = true)]
    HyperionRank {
        #[arg(long)]
        jobs: PathBuf,
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        rank: usize,
        #[arg(long)]
        world: usize,
    },
    Fit {
        #[arg(long)]
        train: PathBuf,
        #[arg(long, default_value = "(pipe X (ridge))")]
        genome: String,
        #[arg(long)]
        output: PathBuf,
    },
    FitView {
        #[arg(long)]
        prepared: PathBuf,
        #[arg(long)]
        cut: usize,
        #[arg(long)]
        genome: String,
        #[arg(long)]
        output: PathBuf,
    },
    /// Isolated export worker: fits exactly the sealed projected input
    /// inherited as `--fd`; opens no prepared path (W1 follow-up 2c).
    #[command(hide = true)]
    FitMemory {
        #[arg(long)]
        fd: i32,
        #[arg(long)]
        training_sha256: String,
        #[arg(long)]
        preprocessing_sha256: String,
        #[arg(long)]
        genome: String,
        #[arg(long)]
        output: PathBuf,
    },
    Predict {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        features: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Score {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        test: PathBuf,
        #[arg(long)]
        receipt: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Outer (sealed final-test) score release authority: init, grant, score,
    /// release and events. Sealed labels reach only a granted outer scorer; the
    /// score is withheld until released, once.
    OuterRelease {
        #[command(subcommand)]
        command: jope_parallel::gp::outer_release_cli::OuterReleaseCommand,
    },
    Bootstrap {
        #[arg(long)]
        prepared: PathBuf,
        #[arg(long)]
        cut: usize,
        #[arg(long)]
        output: PathBuf,
    },
    Search {
        #[arg(long)]
        prepared: PathBuf,
        #[arg(long)]
        cut: usize,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 1000)]
        evaluations: usize,
        #[arg(long, default_value_t = 600)]
        seconds: u64,
        #[arg(long)]
        qualification: bool,
        #[arg(long)]
        seeds: Option<PathBuf>,
        #[arg(long)]
        vocabulary: Option<PathBuf>,
        #[arg(long, default_value_t = jope_parallel::manifest::SEED)]
        rng_seed: u64,
    },
    GpService {
        /// Remain available with a fresh waiting heartbeat when the queue drains.
        #[arg(long)]
        wait_for_work: bool,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long, default_value_t = 26)]
        workers: usize,
        #[arg(long, default_value_t = 4)]
        threads: usize,
    },
    #[command(hide = true)]
    EvaluateWorker {
        #[arg(long)]
        train: PathBuf,
        #[arg(long)]
        validation: PathBuf,
        #[arg(long)]
        genome: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Snapshot {
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        label: String,
        #[arg(long, required = true)]
        include: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    FreezeFamilies {
        #[arg(long)]
        audit: PathBuf,
        #[arg(long)]
        prepared: Vec<PathBuf>,
        #[arg(long)]
        review: PathBuf,
        #[arg(long)]
        decisions: PathBuf,
        #[arg(long)]
        parent: Option<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    Audit {
        #[arg(long, required = true)]
        source: Vec<PathBuf>,
        #[arg(long)]
        decisions: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Prepare {
        #[arg(long)]
        audit: PathBuf,
        #[arg(long)]
        dataset: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        archive_legacy: bool,
        #[arg(long)]
        qualification: bool,
        #[arg(long)]
        backing_manifest: PathBuf,
        #[arg(long)]
        family_registry: Option<PathBuf>,
        #[arg(long)]
        entity_review: Option<PathBuf>,
        /// The dataset tree declared authoritative for a frozen family
        /// partition when no frozen family registry is given. Never taken
        /// from the environment.
        #[arg(long)]
        authoritative_inventory_root: Option<PathBuf>,
    },
    Stage {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        destination: PathBuf,
    },
    Transfer {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        stage: PathBuf,
        #[arg(long)]
        host: String,
        #[arg(long)]
        identity: PathBuf,
        #[arg(long)]
        destination: String,
    },
    Verify {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        destination: PathBuf,
    },
    #[cfg(feature = "neural")]
    PolicyParity {
        #[arg(long)]
        directory: PathBuf,
    },
    /// Export a verified checkpoint's policy weights to safetensors on the CPU, with a receipt.
    #[cfg(feature = "neural")]
    ExportPolicy {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        receipt: PathBuf,
        /// state.json sha256 the operator preregistered for this checkpoint.
        #[arg(long)]
        expect_state_sha256: String,
        /// Optional parent checkpoint for the per-tensor relative change diagnostic.
        #[arg(long)]
        parent: Option<PathBuf>,
    },
    #[cfg(feature = "neural")]
    LossOptimizerParity {
        #[arg(long)]
        directory: PathBuf,
    },
    #[cfg(feature = "neural")]
    Encode {
        #[arg(long)]
        prepared: PathBuf,
        #[arg(long)]
        cut: usize,
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 0)]
        gpu: usize,
    },
    #[cfg(feature = "neural")]
    DistributedProbe {
        #[arg(long)]
        world: usize,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 120)]
        timeout: u64,
        #[arg(long, hide = true)]
        fail_rank: Option<usize>,
    },
    #[cfg(feature = "neural")]
    TrainingProbe {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        vocabulary: PathBuf,
        #[arg(long, default_value_t = 64)]
        microbatch: usize,
        #[arg(long, default_value_t = 2048)]
        global_batch: usize,
        #[arg(long, default_value_t = 30)]
        steps: usize,
        #[arg(long, default_value_t = 600)]
        timeout: u64,
    },
    #[cfg(feature = "neural")]
    #[command(hide = true)]
    TrainingProbeRank {
        #[arg(long)]
        directory: PathBuf,
        #[arg(long)]
        vocabulary: PathBuf,
        #[arg(long)]
        rank: usize,
        #[arg(long)]
        world: usize,
        #[arg(long)]
        microbatch: usize,
        #[arg(long, default_value_t = 2048)]
        global_batch: usize,
        #[arg(long)]
        steps: usize,
    },
    #[cfg(feature = "neural")]
    NcclProbe {
        #[arg(long)]
        directory: PathBuf,
        #[arg(long)]
        rank: usize,
        #[arg(long)]
        world: usize,
    },
    Status,
    Tui {
        /// Select an immutable experiment manifest or a bound dashboard manifest.
        #[arg(long)]
        experiment_manifest: Option<PathBuf>,
    },
    /// Read the full saved loss history without starting training or writing files.
    LearningHistory,
    Monitor,
    /// Check prerequisites; cannot launch an unqualified pilot.
    PilotGate,
    #[cfg(feature = "neural")]
    Campaign {
        #[arg(long)]
        spec: PathBuf,
    },
    #[cfg(feature = "neural")]
    Cycle {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 300)]
        steps: usize,
        #[arg(long, default_value_t = 2)]
        cuts: usize,
        #[arg(long, default_value_t = 0)]
        cut_start: usize,
        #[arg(long)]
        contexts: Option<PathBuf>,
        #[arg(long)]
        initialize: Option<PathBuf>,
        #[arg(long)]
        observations: Vec<PathBuf>,
        #[arg(long)]
        qualification: bool,
    },
}

use anyhow::Result;
use clap::{Parser, Subcommand};
use jope_parallel::{audit, operations, prepare, snapshot, transport};
use std::path::PathBuf;
#[derive(Parser)]
#[command(
    version,
    about = "Real-data migration and isolated Rust policy/GP training"
)]
struct Cli {
    #[arg(long, default_value = ".")]
    root: PathBuf,
    #[command(subcommand)]
    command: Cmd,
}
#[derive(Subcommand)]
enum Cmd {
    Doctor,
    #[cfg(feature = "neural")]
    GpuWorker {
        #[arg(long)] pool: PathBuf,
        #[arg(long)] rank: usize,
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
        #[arg(long)] jobs: PathBuf,
        #[arg(long)] hyperion: PathBuf,
        #[arg(long)] foundation: PathBuf,
        #[arg(long)] target: PathBuf,
        #[arg(long)] assembly: PathBuf,
        #[arg(long)] output: PathBuf,
        #[arg(long)] aliases: PathBuf,
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
        #[arg(long)] corpus: PathBuf,
        #[arg(long)] checkpoint: PathBuf,
        #[arg(long)] panel: PathBuf,
        #[arg(long)] output: PathBuf,
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
        #[arg(long)] corpus: PathBuf,
        #[arg(long)] output: PathBuf,
    },
    ComparisonBaseline {
        #[arg(long)] jobs: PathBuf,
        #[arg(long)] vocabulary: PathBuf,
        #[arg(long)] output: PathBuf,
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
        #[arg(long)] authority: PathBuf,
        #[arg(long)] witness: PathBuf,
        #[arg(long)] completion_sha256: String,
        #[arg(long)] authority_instance: String,
    },
    #[command(hide = true)]
    RegistryFit {
        #[arg(long)] fd: i32,
        #[arg(long)] output: PathBuf,
    },
    /// Own all physical regression fits through one durable native authority.
    RegistryAuthority {
        #[arg(long)] authority: PathBuf,
        #[arg(long)] witness: PathBuf,
        #[arg(long)] socket: PathBuf,
        #[arg(long, default_value_t = 104)] workers: usize,
        #[arg(long)] event_witness_config: PathBuf,
        #[arg(long)] event_witness_config_sha256: String,
        #[arg(long)] funding_config: Option<PathBuf>,
    },
    /// Transport a typed funding request through the protected native client.
    FundingRequest {
        #[arg(long, default_value = "/etc/jope-dime/funding/client.json")]
        config: PathBuf,
    },
    /// Score a retained committed model with separately authorized query access.
    QueryEvaluate {
        #[arg(long)] authority: PathBuf,
        #[arg(long)] witness: PathBuf,
        #[arg(long)] socket: PathBuf,
        #[arg(long)] request: PathBuf,
        #[arg(long)] output: PathBuf,
    },
    /// Retrieve an original query result without extending its execution deadline.
    QueryRetrieve {
        #[arg(long)] authority: PathBuf,
        #[arg(long)] witness: PathBuf,
        #[arg(long)] socket: PathBuf,
        #[arg(long)] request: PathBuf,
        #[arg(long)] output: PathBuf,
        #[arg(long)] deadline_monotonic_ns: u64,
    },
    /// Retain one authenticated independent witness request from standard input.
    RegistryWitnessRequest {
        #[arg(long)] root: PathBuf,
        #[arg(long)] policy: PathBuf,
        #[arg(long)] key: PathBuf,
    },
    /// Retrieve an original witness acknowledgment without recording another event.
    RegistryWitnessGetAck {
        #[arg(long)] root: PathBuf,
        #[arg(long)] policy: PathBuf,
        #[arg(long)] key: PathBuf,
    },
    /// Capture a reviewed inventory without publishing coverage or enabling fits.
    HistoryStage {
        #[arg(long)] authority: PathBuf,
        #[arg(long)] witness: PathBuf,
        #[arg(long)] inventory: PathBuf,
    },
    HistoryImportBatch {
        #[arg(long)] authority: PathBuf,
        #[arg(long)] witness: PathBuf,
        #[arg(long)] batch: PathBuf,
    },
    HistoryCommitCoverage {
        #[arg(long)] authority: PathBuf,
        #[arg(long)] witness: PathBuf,
        #[arg(long)] stage_sha256: String,
    },
    RecoveryInspect {
        #[arg(long)] authority: PathBuf,
        #[arg(long)] witness: PathBuf,
    },
    RecoveryReplay {
        #[arg(long)] authority: PathBuf,
        #[arg(long)] witness: PathBuf,
    },
    RecoveryReconcile {
        #[arg(long)] authority: PathBuf,
        #[arg(long)] witness: PathBuf,
    },
    RecoveryCommitEpoch {
        #[arg(long)] authority: PathBuf,
        #[arg(long)] witness: PathBuf,
        #[arg(long)] report_sha256: String,
    },
    NamespaceActivate {
        #[arg(long)] authority: PathBuf,
        #[arg(long)] witness: PathBuf,
        #[arg(long)] scope: String,
        #[arg(long)] capability: String,
    },
    NamespaceDeactivate {
        #[arg(long)] authority: PathBuf,
        #[arg(long)] witness: PathBuf,
        #[arg(long)] scope: String,
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
        #[arg(long, requires = "cut")] prepared: Option<PathBuf>,
        #[arg(long, requires = "prepared")] cut: Option<usize>,
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
        #[arg(long)] prepared:PathBuf,
        #[arg(long)] cut:usize,
        #[arg(long)] genome:String,
        #[arg(long)] output:PathBuf,
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
fn write_query_result(path: &std::path::Path, result: &impl serde::Serialize) -> Result<()> {
    use std::io::Write;
    let bytes = serde_json::to_vec_pretty(result)?;
    let mut output = std::fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&bytes)?;
    output.sync_all()?;
    std::fs::File::open(path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(std::path::Path::new(".")))?.sync_all()?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Cmd::VerifyFitCompletion { authority, witness, completion_sha256, authority_instance } => {
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            let completion = registry.verify_committed_completion(&completion_sha256)?;
            anyhow::ensure!(completion.authority_instance == authority_instance, "unapproved fit authority instance");
            println!("{}", serde_json::to_string(&completion)?);
        }
        Cmd::RegistryFit { fd, output } => jope_parallel::fit_boundary::execute(fd, &output)?,
        Cmd::RegistryAuthority { authority, witness, socket, workers, event_witness_config, event_witness_config_sha256, funding_config } =>
            jope_parallel::fit_boundary::serve(authority, witness, socket, workers, &event_witness_config, &event_witness_config_sha256, funding_config.as_deref())?,
        Cmd::FundingRequest { config } => {
            use std::io::Read;
            let mut bytes = Vec::new();
            std::io::stdin().take(8 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
            anyhow::ensure!(bytes.len() <= 8 * 1024 * 1024, "funding stdin exceeds bound");
            let request: jope_parallel::eval_registry::service::FundingRequest = serde_json::from_slice(&bytes)?;
            let reply = jope_parallel::eval_registry::service::funding_request(&config, &request)?;
            println!("{}", serde_json::to_string(&reply)?);
        }
        Cmd::QueryEvaluate { authority, witness, socket, request, output } => {
            anyhow::ensure!(!output.exists(), "query output reference already exists");
            let inputs = jope_parallel::manifest::read_json(&request)?;
            let result = jope_parallel::eval_registry::service::query_evaluate(&socket, &authority, &witness, &inputs)?;
            write_query_result(&output, &result)?;
            println!("{}", serde_json::to_string(&result)?);
        }
        Cmd::QueryRetrieve { authority, witness, socket, request, output, deadline_monotonic_ns } => {
            anyhow::ensure!(!output.exists(), "query output reference already exists");
            let inputs = jope_parallel::manifest::read_json(&request)?;
            let result = jope_parallel::eval_registry::service::query_retrieve(&socket, &authority, &witness, &inputs, deadline_monotonic_ns)?;
            write_query_result(&output, &result)?;
            println!("{}", serde_json::to_string(&result)?);
        }
        Cmd::RegistryWitnessRequest { root, policy, key } =>
            jope_parallel::eval_registry::witness::serve_stdio(&root, &policy, &key)?,
        Cmd::RegistryWitnessGetAck { root, policy, key } =>
            jope_parallel::eval_registry::witness::serve_stdio_get_ack(&root, &policy, &key)?,
        Cmd::HistoryStage { authority, witness, inventory } => {
            let inventory = jope_parallel::manifest::read_json(&inventory)?;
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            println!("{}", registry.history_stage(&inventory)?);
        }
        Cmd::HistoryImportBatch { authority, witness, batch } => {
            let batch = jope_parallel::manifest::read_json(&batch)?;
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            println!("{}", registry.history_import_batch(&batch)?);
        }
        Cmd::HistoryCommitCoverage { authority, witness, stage_sha256 } => {
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            println!("{}", registry.history_commit_coverage(&stage_sha256)?);
        }
        Cmd::RecoveryInspect { authority, witness } => {
            let report = jope_parallel::eval_registry::Authority::recovery_inspect(&authority, &witness)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::RecoveryReplay { authority, witness } => {
            let report = jope_parallel::eval_registry::Authority::recovery_replay(&authority, &witness)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::RecoveryReconcile { authority, witness } => {
            println!("{}", jope_parallel::eval_registry::Authority::recovery_reconcile(&authority, &witness)?);
        }
        Cmd::RecoveryCommitEpoch { authority, witness, report_sha256 } => {
            println!("{}", jope_parallel::eval_registry::Authority::recovery_commit_epoch(&authority, &witness, &report_sha256)?);
        }
        Cmd::NamespaceActivate { authority, witness, scope, capability } => {
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            println!("{}", registry.namespace_activate(&scope, &capability)?);
        }
        Cmd::NamespaceDeactivate { authority, witness, scope } => {
            jope_parallel::eval_registry::Authority::open(&authority, &witness)?.namespace_deactivate(&scope)?;
        }
        Cmd::RegistryServiceFit => jope_parallel::fit_boundary::execute_service()?,
        Cmd::RegistryServiceQuery => jope_parallel::eval_registry::service::execute_query_worker()?,
        Cmd::CompareSeeds { manifest, workers, prefix, seconds, downstream } => {
            use std::os::unix::process::CommandExt;
            let mut command = std::process::Command::new("python3");
            command.arg(cli.root.join("tools/seed_experiment.py")).arg("--root").arg(&cli.root)
                .arg("compare").arg("--manifest").arg(manifest)
                .arg("--workers").arg(workers.to_string()).arg("--prefix").arg(prefix.to_string())
                .arg("--seconds").arg(seconds.to_string());
            if downstream { command.arg("--downstream"); }
            return Err(command.exec().into());
        },
        #[cfg(feature = "neural")]
        Cmd::ApplyBinaryContexts { jobs, hyperion, foundation, target, assembly, output, aliases } =>
            jope_parallel::contexts::apply_binary_qualification(&cli.root, &jobs, &hyperion, &foundation, &target, &assembly, &output, &aliases)?,
        Cmd::Enqueue { jobs, contexts, campaign_contexts } => {
            println!("{} new GP jobs admitted with verified saved contexts",
                jope_parallel::queue::enqueue_bound(&cli.root, &jobs, &contexts, campaign_contexts.as_deref())?);
        },
        Cmd::CampaignCoverage { campaign, union, output, gp_root } => {
            let roots: Vec<_> = gp_root.iter().map(|p| cli.root.join(p)).collect();
            let refs: Vec<_> = roots.iter().map(|p| p.as_path()).collect();
            println!("{}", serde_json::to_string_pretty(&jope_parallel::queue::write_coverage(
                &cli.root, &campaign, &union, &refs, &output)?)?);
        },
        #[cfg(feature = "neural")]
        Cmd::ApplyContexts { jobs, hyperion, foundation, target, assembly, output, qualification } =>
            jope_parallel::contexts::apply(&cli.root, &jobs, &hyperion, &foundation, &target, &assembly, &output, qualification)?,
        #[cfg(feature = "neural")]
        Cmd::UnionContexts { indexes, output } =>
            jope_parallel::contexts::union(&cli.root, &indexes, &output)?,
        #[cfg(feature = "neural")]
        Cmd::AssembleContexts { jobs, hyperion, foundation, target, output, registry, qualification } => {
            if let Some(registry) = registry {
                jope_parallel::contexts::assemble_registered(&cli.root, &jobs, &hyperion, &foundation, &target, &registry, &output, qualification)?;
            } else {
                jope_parallel::contexts::assemble(&cli.root, &jobs, &hyperion, &foundation, &target, &output, qualification)?;
            }
        },

        #[cfg(feature = "neural")]
        Cmd::FoundationService {
            jobs,
            spec,
            output,
            batch,
        } => jope_parallel::components::service(
            &cli.root,
            &jobs,
            &spec,
            &output,
            batch,
            "foundation",
        )?,
        #[cfg(feature = "neural")]
        Cmd::FoundationRank {
            jobs,
            spec,
            output,
            rank,
            world,
            batch,
        } => jope_parallel::components::foundation_rank(
            &cli.root, &jobs, &spec, &output, rank, world, batch,
        )?,

        #[cfg(feature = "neural")]
        Cmd::TargetService {
            jobs,
            spec,
            output,
            batch,
        } => jope_parallel::components::service(&cli.root, &jobs, &spec, &output, batch, "target")?,
        #[cfg(feature = "neural")]
        Cmd::TargetRank {
            jobs,
            spec,
            output,
            rank,
            world,
            batch,
        } => jope_parallel::components::target_rank(
            &cli.root, &jobs, &spec, &output, rank, world, batch,
        )?,

        Cmd::HyperionService { jobs, spec, output } => {
            jope_parallel::hyperion::service(&cli.root, &jobs, &spec, &output)?
        }
        Cmd::HyperionRank {
            jobs,
            spec,
            output,
            rank,
            world,
        } => jope_parallel::hyperion::rank(&cli.root, &jobs, &spec, &output, rank, world)?,
        Cmd::ExpandCuts {
            prepared,
            output,
            rounds,
        } => {
            prepare::expand_cuts(&prepared, &output, rounds)?;
        }
        Cmd::IndexCuts {
            prepared,
            backing_manifest,
            metadata,
            output,
        } => prepare::index_existing(&prepared, &backing_manifest, &metadata, &output)?,
        Cmd::EvaluateView {
            prepared,
            cut,
            genome,
            output,
        } => jope_parallel::gp::evaluate_view(&prepared, cut, &genome, &output)?,
        Cmd::EvaluateMemory { fd, genome, output, prepared, cut } => {
            jope_parallel::gp::evaluate_prepared_memory(fd, &genome, &output, prepared.as_deref().zip(cut))?
        }
        #[cfg(feature = "neural")]
        Cmd::Train {
            corpus,
            output,
            world,
            microbatch,
            steps,
            checkpoint_steps,
            resume,
            initialize,
            fail_rank,
            replay_policy,
        } => jope_parallel::controller::train_with_policy(
            &cli.root,
            &corpus,
            &output,
            world,
            microbatch,
            steps,
            checkpoint_steps,
            resume,
            fail_rank,
            initialize.as_deref(),
            replay_policy.as_deref(),
        )?,
        #[cfg(feature = "neural")]
        Cmd::TrainRank {
            corpus,
            output,
            launch,
            rank,
            world,
            microbatch,
            steps,
            checkpoint_steps,
            resume,
            initialize,
            replay_policy,
        } => jope_parallel::training::rank_with_policy(
            &cli.root,
            &corpus,
            &output,
            &launch,
            rank,
            world,
            microbatch,
            steps,
            checkpoint_steps,
            resume,
            initialize.as_deref(),
            replay_policy.as_deref(),
        )?,
        #[cfg(feature = "neural")]
        Cmd::QualifyBatchRank { corpus, output, launch, rank, microbatch, global_batch, steps, initialize } =>
            jope_parallel::training::qualify_batch_rank(&cli.root, &corpus, &output, &launch,
                rank, microbatch, global_batch, steps, &initialize)?,
        #[cfg(feature = "neural")]
        Cmd::ProposeRanked { manifest, rank, world } =>
            jope_parallel::ranked_proposals::run(&cli.root, &manifest, rank, world)?,
        #[cfg(feature = "neural")]
        Cmd::AssembleRanked { manifest } =>
            jope_parallel::ranked_proposals::assemble(&cli.root, &manifest)?,
        #[cfg(feature = "neural")]
        Cmd::Propose {
            generation_manifest,
            corpus,
            checkpoint,
            output,
            cuts,
            cut_start,
            shuffled,
        } => jope_parallel::controller::propose_with_generation(
            &cli.root,
            &corpus,
            &checkpoint,
            &output,
            cuts,
            cut_start,
            shuffled,
            generation_manifest.as_deref(),
        )?,
        #[cfg(feature = "neural")]
        Cmd::ProposeRank {
            corpus,
            checkpoint,
            output,
            rank,
            world,
            cuts,
            cut_start,
            shuffled,
        } => jope_parallel::proposals::rank(
            &cli.root,
            &corpus,
            &checkpoint,
            &output,
            rank,
            world,
            cuts,
            cut_start,
            shuffled,
        )?,
        #[cfg(feature = "neural")]
        Cmd::MonitorCheckpoint { corpus, checkpoint, panel, output } =>
            jope_parallel::monitoring::evaluate(&cli.root, &corpus, &checkpoint, &panel, &output)?,
        Cmd::ComparisonBaseline { jobs, vocabulary, output } => {
            jope_parallel::tournament::baseline(&cli.root, &jope_parallel::manifest::read_json(&jobs)?,
                &jope_parallel::manifest::read_json(&vocabulary)?, &output)?;
        },
        Cmd::BuildCorpus { contexts, vocabulary, mut observations, observation_manifest, output, qualification, scan_cache, progress, teaching_store, label_policy, admission_manifest } => {
            if let Some(path) = observation_manifest {
                observations.extend(jope_parallel::manifest::read_json::<Vec<PathBuf>>(&path)?);
            }
            let heartbeat = progress.as_deref().map(|p| jope_parallel::telemetry::StageHeartbeat::start(p,
                serde_json::json!({"schema":"parallel.corpus_progress.v1","phase":"packing","output":output,"complete":false}))).transpose()?;
            if let Some(store) = teaching_store {
                if let Some(admission) = admission_manifest {
                    anyhow::ensure!(label_policy == "strict_midrank_v4", "admission manifest requires strict policy");
                    jope_parallel::teaching_store::publish_admitted(&cli.root,&contexts,&vocabulary,&observations,
                        &output,&store,qualification,&admission,|operation,done,total|
                            heartbeat.as_ref().map_or(Ok(()),|h|h.advance(operation,done,total)))?;
                } else {
                jope_parallel::teaching_store::publish(&cli.root, &contexts, &vocabulary, &observations,
                    &output, &store, qualification, &label_policy, |operation, done, total|
                        heartbeat.as_ref().map_or(Ok(()), |h| h.advance(operation, done, total)))?;
                }
            } else {
                anyhow::ensure!(label_policy == "reference_v1", "label-policy changes require a separate teaching store");
                jope_parallel::corpus_publication::build(&cli.root, &contexts, &vocabulary, &observations,
                    &output, qualification, scan_cache.as_deref(), heartbeat.as_ref())?;
            }
            if let Some(h) = heartbeat {
                h.advance("verified corpus published", 1, Some(1))?;
                h.finish()?;
                let path = progress.as_deref().expect("heartbeat has a progress path");
                let mut receipt: serde_json::Value = jope_parallel::manifest::read_json(path)?;
                receipt["complete"] = true.into();
                receipt["phase"] = "completed".into();
                receipt["timestamp"] = jope_parallel::telemetry::now().into();
                jope_parallel::manifest::write_json(path, &receipt)?;
            }
        },
        Cmd::ExportTeachingCorpus { corpus, output } => jope_parallel::teaching_store::export(&cli.root, &corpus, &output)?,
        Cmd::Doctor => println!("{}", serde_json::to_string_pretty(&operations::doctor())?),
        #[cfg(feature = "neural")]
        Cmd::GpuWorker { pool, rank } => jope_parallel::gpu_worker::serve(&cli.root,&pool,rank)?,
        Cmd::Fit {
            train,
            genome,
            output,
        } => jope_parallel::gp::fit(&train, &genome, &output)?,
        Cmd::FitView {prepared,cut,genome,output} => jope_parallel::gp::fit_view(&prepared,cut,&genome,&output)?,
        Cmd::Predict {
            model,
            features,
            output,
        } => jope_parallel::gp::predict(&model, &features, &output)?,
        Cmd::Score {
            model,
            test,
            receipt,
            output,
        } => jope_parallel::gp::score(&model, &test, &receipt, &output)?,
        Cmd::Bootstrap {
            prepared,
            cut,
            output,
        } => jope_parallel::gp::search_seeded_with_rng_in_root(&cli.root, &prepared, cut, &output, 80, 600, false, None, None, jope_parallel::manifest::SEED)?,
        Cmd::Search {
            prepared,
            cut,
            output,
            evaluations,
            seconds,
            qualification,
            seeds,
            vocabulary,
            rng_seed,
        } => jope_parallel::gp::search_seeded_with_rng_in_root(
            &cli.root,
            &prepared,
            cut,
            &output,
            evaluations,
            seconds,
            qualification,
            seeds.as_deref(),
            vocabulary.as_deref(),
            rng_seed,
        )?,
        Cmd::GpService {
            wait_for_work,
            manifest,
            workers,
            threads,
        } => jope_parallel::service::run_with_wait(&cli.root, &manifest, workers, threads, wait_for_work)?,
        Cmd::EvaluateWorker {
            train,
            validation,
            genome,
            output,
        } => jope_parallel::gp::evaluate(&train, &validation, &genome, &output)?,
        Cmd::Snapshot {
            source,
            label,
            include,
            output,
        } => {
            let m = snapshot::create(&source, &label, &include, &output)?;
            println!("snapshotted {} source files", m.files.len());
        }
        Cmd::FreezeFamilies { audit, prepared, review, decisions, parent, output } => {
            let registry = jope_parallel::families::freeze(&audit, &prepared, &review, &decisions, parent.as_deref(), &output)?;
            println!("{}", serde_json::to_string_pretty(&registry.quota_report())?);
        }
        Cmd::Audit {
            source,
            decisions,
            output,
        } => {
            let r = audit::run(&source, &decisions, &output)?;
            println!(
                "{}",
                serde_json::json!({"datasets":r.datasets.len(),"artifacts":r.artifact_counts,"complete":r.complete,"missing_roots":r.missing_roots})
            );
        }
        Cmd::Prepare {
            audit,
            dataset,
            output,
            archive_legacy,
            qualification,
            backing_manifest,
            family_registry,
            entity_review,
        } => {
            let p = prepare::run_with_admission(&audit, &dataset, &output, archive_legacy, qualification, &backing_manifest, family_registry.as_deref(), entity_review.as_deref())?;
            println!(
                "{}",
                serde_json::json!({"dataset":p.dataset,"rows":p.rows_total,"cuts":p.cuts.len(),"clean_evaluation":p.clean_evaluation,"family_partition_frozen":p.family_partition_frozen,"manifest":output.join("prepared.json")})
            );
        }
        Cmd::Stage {
            manifest,
            destination,
        } => transport::stage(&manifest, &destination)?,
        Cmd::Transfer {
            manifest,
            stage,
            host,
            identity,
            destination,
        } => transport::transfer(&manifest, &stage, &host, &identity, &destination)?,
        Cmd::Verify {
            manifest,
            destination,
        } => println!(
            "verified {} files",
            transport::verify(&manifest, &destination)?
        ),
        #[cfg(feature = "neural")]
        Cmd::PolicyParity { directory } => jope_parallel::neural::parity(&directory)?,
        #[cfg(feature = "neural")]
        Cmd::LossOptimizerParity { directory } => {
            jope_parallel::neural::loss_optimizer_parity(&directory)?
        }
        #[cfg(feature = "neural")]
        Cmd::Encode {
            prepared,
            cut,
            spec,
            output,
            gpu,
        } => {
            let e =
                jope_parallel::encoding::encode(&cli.root, &prepared, cut, &spec, &output, gpu)?;
            println!("{}", serde_json::to_string(&e)?);
        }
        #[cfg(feature = "neural")]
        Cmd::DistributedProbe {
            world,
            output,
            timeout,
            fail_rank,
        } => jope_parallel::controller::probe(world, &output, timeout, fail_rank)?,
        #[cfg(feature = "neural")]
        Cmd::TrainingProbe {
            output,
            vocabulary,
            microbatch,
            global_batch,
            steps,
            timeout,
        } => jope_parallel::controller::training_probe(&output, &vocabulary, microbatch, global_batch, steps, timeout)?,
        #[cfg(feature = "neural")]
        Cmd::TrainingProbeRank {
            directory,
            vocabulary,
            rank,
            world,
            microbatch,
            global_batch,
            steps,
        } => jope_parallel::neural::training_probe(rank, world, &directory, &vocabulary, microbatch, global_batch, steps)?,
        #[cfg(feature = "neural")]
        Cmd::NcclProbe {
            directory,
            rank,
            world,
        } => jope_parallel::neural::nccl_probe(rank, world, &directory)?,
        Cmd::Status => println!(
            "{}",
            serde_json::to_string_pretty(&operations::status(&cli.root)?)?
        ),
        Cmd::Tui { experiment_manifest } => jope_parallel::dashboard::run_with_manifest(&cli.root, experiment_manifest.as_deref())?,
        Cmd::LearningHistory => println!("{}", serde_json::to_string_pretty(&jope_parallel::learning_history::Cache::default().refresh(&cli.root).report())?),
        Cmd::Monitor => jope_parallel::telemetry::monitor(&cli.root)?,
        Cmd::PilotGate => operations::pilot_gate(&cli.root)?,
        #[cfg(feature = "neural")]
        Cmd::Campaign { spec } => jope_parallel::campaign::run(&cli.root, &spec)?,
        #[cfg(feature = "neural")]
        Cmd::Cycle {
            corpus,
            output,
            steps,
            cuts,
            cut_start,
            contexts,
            initialize,
            observations,
            qualification,
        } => jope_parallel::pipeline::run(&cli.root, &corpus, &output, steps, cuts, cut_start, contexts.as_deref(), initialize.as_deref(), &observations, qualification)?,
    }
    Ok(())
}

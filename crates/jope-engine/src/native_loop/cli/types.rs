use super::*;

#[derive(Parser)]
pub(crate) struct Args {
    #[command(flatten)]
    pub(crate) legacy: LegacyFlags,
    #[command(subcommand)]
    pub(crate) command: Command,
}
/// Fixed-loop arguments: all required by bind/bind-parent-decision/tick/serve, and
/// refused by the bounded-run operations.
#[derive(ClapArgs)]
pub(crate) struct LegacyFlags {
    #[arg(long)]
    root: Option<PathBuf>,
    #[arg(long)]
    state: Option<PathBuf>,
    #[arg(long)]
    graph_state: Option<PathBuf>,
    #[arg(long)]
    worker_state: Option<PathBuf>,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    config_sha256: Option<String>,
    #[arg(long)]
    socket: Option<PathBuf>,
    #[arg(long)]
    tenant: Option<String>,
    #[arg(long)]
    project: Option<String>,
}
pub(crate) struct LegacyArgs {
    pub(crate) root: PathBuf,
    pub(crate) state: PathBuf,
    pub(crate) graph_state: PathBuf,
    pub(crate) worker_state: PathBuf,
    pub(crate) config: PathBuf,
    pub(crate) config_sha256: String,
    pub(crate) socket: PathBuf,
    pub(crate) tenant: String,
    pub(crate) project: String,
}
impl LegacyFlags {
    pub(crate) fn any(&self) -> bool {
        self.root.is_some()
            || self.state.is_some()
            || self.graph_state.is_some()
            || self.worker_state.is_some()
            || self.config.is_some()
            || self.config_sha256.is_some()
            || self.socket.is_some()
            || self.tenant.is_some()
            || self.project.is_some()
    }
    pub(crate) fn complete(self) -> Option<LegacyArgs> {
        Some(LegacyArgs {
            root: self.root?,
            state: self.state?,
            graph_state: self.graph_state?,
            worker_state: self.worker_state?,
            config: self.config?,
            config_sha256: self.config_sha256?,
            socket: self.socket?,
            tenant: self.tenant?,
            project: self.project?,
        })
    }
}
#[derive(Subcommand)]
pub(crate) enum Command {
    Bind {
        binding: PathBuf,
    },
    BindParentDecision {
        decision: PathBuf,
        sha256: String,
    },
    Tick,
    Serve,
    /// Freeze a reviewed `jope.native_bounded_run.v1` contract; refuses invalid,
    /// underfunded and legacy manifests before creating anything.
    Plan {
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long)]
        contract: PathBuf,
        /// Required exactly when the contract names a predecessor attempt.
        #[arg(long)]
        predecessor_run_dir: Option<PathBuf>,
    },
    /// Check a pinned plan against a host census without admission or dispatch.
    Preflight {
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long)]
        plan_sha256: String,
        #[arg(long)]
        census: PathBuf,
    },
    /// Read-only status of a planned bounded run.
    Status {
        #[arg(long)]
        run_dir: PathBuf,
    },
    /// Read-only verification of the plan, journal and committed results.
    Verify {
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long)]
        plan_sha256: Option<String>,
    },
    /// Idempotently request cancellation of the reviewed plan.
    Cancel {
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long)]
        plan_sha256: String,
        #[arg(long)]
        reason: String,
    },
    /// Host census collection and merge (W7-T01); observes, never allocates or reserves.
    Census {
        #[command(subcommand)]
        op: CensusCommand,
    },
    /// Author an unsigned bounded-run contract from a declarative request and a merged census
    /// (W7-D6). Host identities come only from the census and closure digests only from files.
    Contract {
        #[command(subcommand)]
        op: ContractCommand,
    },
    /// Sign or verify a contract's participant grants (W7-D6). The grant-authority key is read
    /// only from an owner-only (0600) PKCS#8 file, never from arguments or the environment,
    /// and is never printed.
    Grant {
        #[command(subcommand)]
        op: GrantCommand,
    },
    /// A bounded-run rank worker (W7-T07 CPU phase), run by SmartCluster as the
    /// `jope-bounded-rank` trusted service. Exit 0: signed result published (or stopped);
    /// 75: control lost; any other failure: refused before admission, no work started.
    Participant {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        plan_sha256: String,
        #[arg(long)]
        launch_record: PathBuf,
        #[arg(long)]
        outbox: PathBuf,
        /// The coordinator's rendezvous address, `host:port`.
        #[arg(long)]
        coordinator: String,
        #[arg(long)]
        participant_key: PathBuf,
        #[arg(long)]
        installation: PathBuf,
    },
    /// Drive a planned bounded run to a terminal outcome through this host's SmartCluster
    /// supervisor (`--supervision`) or each rank's host agent (`--agents`), or reconcile it
    /// after a coordinator restart.
    Coordinate {
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long)]
        plan_sha256: String,
        /// The merged census, preflighted and journaled before dispatch.
        #[arg(long)]
        census: PathBuf,
        #[arg(long)]
        listen: std::net::SocketAddr,
        #[arg(long)]
        certificate: PathBuf,
        #[arg(long)]
        certificate_key: PathBuf,
        #[arg(long)]
        coordinator_key: PathBuf,
        /// Below the plan's heartbeat timeout.
        #[arg(long)]
        heartbeat_ms: u64,
        /// Supervise every rank through this host's SmartCluster (`HostSupervision` JSON).
        #[arg(long, conflicts_with = "agents", required_unless_present = "agents")]
        supervision: Option<PathBuf>,
        /// Supervise each rank through its host's agent: JSON `{"<rank>": "host:port"}`.
        #[arg(long)]
        agents: Option<PathBuf>,
    },
    /// Serve this host's SmartCluster supervisor to the planned coordinator over the pinned
    /// transport (W7-T07). Runs until stopped.
    Agent {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        plan_sha256: String,
        /// The planned host this agent serves.
        #[arg(long)]
        host: String,
        /// Serve only these planned ranks of the host (default: all of them).
        #[arg(long = "rank")]
        ranks: Vec<u32>,
        #[arg(long)]
        listen: std::net::SocketAddr,
        #[arg(long)]
        certificate: PathBuf,
        #[arg(long)]
        certificate_key: PathBuf,
        /// This host's SmartCluster supervision (`HostSupervision` JSON).
        #[arg(long)]
        supervision: PathBuf,
    },
}
#[derive(Subcommand)]
pub(crate) enum CensusCommand {
    /// Observe this host only and write one `jope.native_host_census.v1` document.
    /// `CUDA_VISIBLE_DEVICES=""` records a CPU-only census; unset queries `nvidia-smi`.
    Collect {
        /// New file (never replaced) in an existing directory.
        #[arg(long)]
        out: PathBuf,
        /// This host's address; must be assigned to a local interface.
        #[arg(long)]
        address: String,
        /// This host's transport certificate (DER, or PEM with one certificate).
        #[arg(long)]
        certificate: PathBuf,
        /// The installer-declared source commit of the pinned executable.
        #[arg(long)]
        source_commit: String,
        /// The installed rank worker executable.
        #[arg(long)]
        executable: PathBuf,
        #[arg(long = "runtime-library", required = true)]
        runtime_libraries: Vec<PathBuf>,
        #[arg(long)]
        compiler: PathBuf,
        #[arg(long = "cuda-library")]
        cuda_libraries: Vec<PathBuf>,
        #[arg(long = "nccl-library")]
        nccl_libraries: Vec<PathBuf>,
        #[arg(long = "installed-policy", required = true)]
        installed_policies: Vec<PathBuf>,
    },
    /// Combine per-host documents, each pinned by the SHA-256 its collector printed.
    Merge {
        /// New file (never replaced) in an existing directory.
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        max_age_ms: u64,
        #[arg(long)]
        clock_skew_ms: u64,
        /// `PATH=SHA256`, one per host.
        #[arg(long = "document", required = true)]
        documents: Vec<String>,
    },
}
#[derive(Subcommand)]
pub(crate) enum ContractCommand {
    /// Write the unsigned contract (new file, never replaced) and print its grant review.
    Author {
        /// `jope.native_bounded_contract_request.v1` JSON.
        #[arg(long)]
        request: PathBuf,
        /// The merged host census.
        #[arg(long)]
        census: PathBuf,
        /// The directory the request's closure paths are relative to.
        #[arg(long)]
        input_root: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}
#[derive(Subcommand)]
pub(crate) enum GrantCommand {
    /// Sign every grant of an unsigned contract and write the signed contract (new file).
    Sign {
        #[arg(long)]
        contract: PathBuf,
        /// The grant-authority key: PKCS#8 Ed25519, owner-only, not a symlink.
        #[arg(long)]
        key: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Check a signed contract's grants exactly as `plan` will, now.
    Verify {
        #[arg(long)]
        contract: PathBuf,
    },
}

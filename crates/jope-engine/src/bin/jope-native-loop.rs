//! Durable eligibility controller; native Graph/SmartCluster retain dispatch authority.
//!
//! `bind`, `bind-parent-decision`, `tick` and `serve` keep the fixed `jope.native_loop.v1`
//! contract and require the loop arguments. `plan`, `status`, `verify` and `cancel` operate
//! on the versioned `jope.native_bounded_run.v1` contract (W7-T02) and take none of them:
//! status and verify are read-only and never start numerical work. `census collect` observes
//! only the local host and `census merge` combines pinned per-host documents (W7-T01); neither
//! allocates or reserves anything. `coordinate` drives a planned run through SmartCluster
//! (directly, or through each host's `agent` over the pinned transport) and `participant` is
//! the rank worker SmartCluster runs (W7-T05/T07); none of them uses SSH.
use anyhow::{bail, ensure, Context, Result};
use clap::{Args as ClapArgs, Parser, Subcommand};
use jain_smartcluster::{Client, ProjectId, TenantId};
use jope_parallel::{
    native_graph::{Artifact, Graph},
    native_loop::{bounded, Binding, Loop},
};
use std::{fs, io::Read, os::unix::fs::OpenOptionsExt, path::PathBuf, time::Duration};
#[derive(Parser)]
struct Args {
    #[command(flatten)]
    legacy: LegacyFlags,
    #[command(subcommand)]
    command: Command,
}
/// Fixed-loop arguments: all required by bind/bind-parent-decision/tick/serve, and
/// refused by the bounded-run operations.
#[derive(ClapArgs)]
struct LegacyFlags {
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
struct LegacyArgs {
    root: PathBuf,
    state: PathBuf,
    graph_state: PathBuf,
    worker_state: PathBuf,
    config: PathBuf,
    config_sha256: String,
    socket: PathBuf,
    tenant: String,
    project: String,
}
impl LegacyFlags {
    fn any(&self) -> bool {
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
    fn complete(self) -> Option<LegacyArgs> {
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
enum Command {
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
enum CensusCommand {
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
fn read_census_document(path: &PathBuf) -> Result<Vec<u8>> {
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .with_context(|| format!("open census document {}", path.display()))?;
    ensure!(f.metadata()?.is_file(), "regular census document required");
    let limit = bounded::census::MAX_CENSUS_BYTES as u64;
    let mut b = Vec::new();
    f.take(limit + 1).read_to_end(&mut b)?;
    ensure!(
        b.len() as u64 <= limit,
        "census document exceeds {limit} bytes"
    );
    Ok(b)
}
fn host_supervision(path: &PathBuf) -> Result<bounded::supervisor::HostSupervision> {
    serde_json::from_slice(&read_census_document(path)?).context("supervision")
}
fn census_command(op: CensusCommand) -> Result<serde_json::Value> {
    use bounded::collect;
    match op {
        CensusCommand::Collect {
            out,
            address,
            certificate,
            source_commit,
            executable,
            runtime_libraries,
            compiler,
            cuda_libraries,
            nccl_libraries,
            installed_policies,
        } => {
            let spec = collect::CollectSpec {
                address,
                certificate,
                source_commit,
                executable,
                runtime_libraries,
                compiler,
                cuda_libraries,
                nccl_libraries,
                installed_policies,
                cuda_visible_devices: std::env::var_os("CUDA_VISIBLE_DEVICES"),
            };
            let now = jope_parallel::resident_protocol::now_ms()?;
            let (census, report) = collect::collect(&spec, &collect::SystemCommands, now)?;
            let bytes = collect::write_census(&out, &census)?;
            ensure!(
                jope_parallel::manifest::hash(&bytes) == report.census_sha256,
                "written census differs from the collected census"
            );
            Ok(serde_json::to_value(report)?)
        }
        CensusCommand::Merge {
            out,
            max_age_ms,
            clock_skew_ms,
            documents,
        } => {
            let mut pinned = Vec::new();
            for d in documents {
                let (path, sha256) = d.rsplit_once('=').context("--document takes PATH=SHA256")?;
                let path = PathBuf::from(path);
                pinned.push(collect::PinnedDocument {
                    label: path.display().to_string(),
                    bytes: read_census_document(&path)?,
                    sha256: sha256.to_string(),
                });
            }
            let now = jope_parallel::resident_protocol::now_ms()?;
            let (census, report) = collect::merge(
                &pinned,
                collect::MergePolicy {
                    max_age_ms,
                    clock_skew_ms,
                },
                now,
            )?;
            let bytes = collect::write_census(&out, &census)?;
            ensure!(
                jope_parallel::manifest::hash(&bytes) == report.census_sha256,
                "written census differs from the merged census"
            );
            Ok(serde_json::to_value(report)?)
        }
    }
}
fn read_contract(path: &PathBuf) -> Result<Vec<u8>> {
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .with_context(|| format!("open contract {}", path.display()))?;
    ensure!(f.metadata()?.is_file(), "regular contract file required");
    let limit = bounded::contract::MAX_CONTRACT_BYTES as u64;
    let mut b = Vec::new();
    f.take(limit + 1).read_to_end(&mut b)?;
    ensure!(b.len() as u64 <= limit, "contract size");
    Ok(b)
}
fn bounded_command(command: Command) -> Result<()> {
    let now = jope_parallel::resident_protocol::now_ms()?;
    let out = match command {
        Command::Plan {
            run_dir,
            contract,
            predecessor_run_dir,
        } => serde_json::to_value(bounded::controller::plan(
            &run_dir,
            &read_contract(&contract)?,
            now,
            predecessor_run_dir.as_deref(),
        )?)?,
        Command::Preflight {
            run_dir,
            plan_sha256,
            census,
        } => serde_json::to_value(bounded::controller::preflight_dry_run(
            &run_dir,
            &plan_sha256,
            &census,
            now,
        )?)?,
        Command::Status { run_dir } => {
            serde_json::to_value(bounded::controller::status(&run_dir)?)?
        }
        Command::Verify {
            run_dir,
            plan_sha256,
        } => serde_json::to_value(bounded::controller::verify(
            &run_dir,
            plan_sha256.as_deref(),
        )?)?,
        Command::Cancel {
            run_dir,
            plan_sha256,
            reason,
        } => serde_json::to_value(bounded::controller::cancel(
            &run_dir,
            &plan_sha256,
            &reason,
            now,
        )?)?,
        Command::Census { op } => census_command(op)?,
        Command::Coordinate {
            run_dir,
            plan_sha256,
            census,
            listen,
            certificate,
            certificate_key,
            coordinator_key,
            heartbeat_ms,
            supervision,
            agents,
        } => {
            let args = bounded::driver::CoordinateArgs {
                run_dir,
                plan_sha256,
                census,
                listen,
                certificate,
                certificate_key,
                coordinator_key,
                heartbeat: Duration::from_millis(heartbeat_ms),
            };
            let stop = std::sync::atomic::AtomicBool::new(false);
            let report = match (supervision, agents) {
                (Some(supervision), None) => bounded::driver::coordinate(
                    &args,
                    bounded::supervisor::SmartClusterSupervisor::new(host_supervision(
                        &supervision,
                    )?)?,
                    &stop,
                )?,
                (None, Some(agents)) => {
                    let agents: std::collections::BTreeMap<u32, std::net::SocketAddr> =
                        serde_json::from_slice(&read_census_document(&agents)?)
                            .context("agents")?;
                    let (contract, _) = bounded::launch::load_plan(
                        &args.run_dir.join("plan.json"),
                        &args.plan_sha256,
                    )?;
                    let credential = bounded::transport::HostCredential::load(
                        &args.certificate,
                        &args.certificate_key,
                    )?;
                    bounded::driver::coordinate(
                        &args,
                        bounded::agent::RemoteSupervisor::new(
                            &contract,
                            &args.plan_sha256,
                            credential,
                            agents,
                        )?,
                        &stop,
                    )?
                }
                _ => bail!("coordinate takes exactly one of --supervision and --agents"),
            };
            serde_json::to_value(report)?
        }
        Command::Agent {
            plan,
            plan_sha256,
            host,
            ranks,
            listen,
            certificate,
            certificate_key,
            supervision,
        } => {
            let supervision = host_supervision(&supervision)?;
            ensure!(
                supervision.host == host,
                "the supervision describes host `{}`, not `{host}`",
                supervision.host
            );
            let (_, bytes) = bounded::launch::load_plan(&plan, &plan_sha256)?;
            let served: std::collections::BTreeSet<u32> = ranks.iter().copied().collect();
            ensure!(
                served.len() == ranks.len(),
                "a rank is named more than once"
            );
            let mut agent = bounded::agent::Agent::new(
                &bytes,
                &plan_sha256,
                &host,
                (!served.is_empty()).then_some(served),
                bounded::supervisor::SmartClusterSupervisor::new(supervision)?,
            )?;
            let credential =
                bounded::transport::HostCredential::load(&certificate, &certificate_key)?;
            let acceptor =
                bounded::agent::agent_acceptor(agent.contract(), listen, &credential, &host)?;
            // One readiness line, so an operator (or a fixture) learns the bound address.
            println!(
                "{}",
                serde_json::json!({"agent": "listening", "host": host, "address": acceptor.local_addr()?})
            );
            std::io::Write::flush(&mut std::io::stdout())?;
            agent.serve(&acceptor, &std::sync::atomic::AtomicBool::new(false))?;
            serde_json::json!({"agent": "stopped", "host": host})
        }
        _ => bail!("not a bounded-run operation"),
    };
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
fn main() -> Result<()> {
    let args = Args::parse();
    if matches!(
        args.command,
        Command::Plan { .. }
            | Command::Preflight { .. }
            | Command::Status { .. }
            | Command::Verify { .. }
            | Command::Cancel { .. }
            | Command::Census { .. }
            | Command::Participant { .. }
            | Command::Coordinate { .. }
            | Command::Agent { .. }
    ) {
        ensure!(
            !args.legacy.any(),
            "bounded-run operations take no fixed-loop arguments; the frozen plan is the only input"
        );
        if let Command::Participant {
            plan,
            plan_sha256,
            launch_record,
            outbox,
            coordinator,
            participant_key,
            installation,
        } = args.command
        {
            let args = bounded::participant::ParticipantArgs {
                plan,
                plan_sha256,
                launch_record,
                outbox,
                coordinator,
                participant_key,
                installation,
            };
            match bounded::participant::run(&args, &bounded::collect::SystemCommands) {
                Ok(code) => std::process::exit(code),
                Err(e) => {
                    // The rank's supervisor keeps its outbox; record why nothing started.
                    bounded::participant::record_refusal(&args.outbox, &e);
                    return Err(e);
                }
            }
        }
        return bounded_command(args.command);
    }
    let a = args.legacy.complete().context(
        "bind/bind-parent-decision/tick/serve require --root, --state, --graph-state, --worker-state, --config, --config-sha256, --socket, --tenant and --project",
    )?;
    let controller = Loop::open(
        &a.root,
        &a.state,
        &a.graph_state,
        &Artifact {
            path: a.config,
            sha256: a.config_sha256,
        },
    )?;
    let graph = Graph::open(
        &a.root,
        &a.graph_state,
        &a.worker_state,
        TenantId::new(a.tenant)?,
        ProjectId::new(a.project)?,
    )?;
    let client = Client::new(a.socket);
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(async {
            match args.command {
                Command::Bind { binding } => {
                    let f = fs::OpenOptions::new()
                        .read(true)
                        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                        .open(binding)?;
                    ensure!(f.metadata()?.is_file(), "regular binding required");
                    let mut b = Vec::new();
                    f.take(1024 * 1024 + 1).read_to_end(&mut b)?;
                    ensure!(b.len() <= 1024 * 1024, "binding size");
                    controller.bind(&serde_json::from_slice::<Binding>(&b)?)?;
                }
                Command::BindParentDecision { decision, sha256 } => controller
                    .bind_parent_decision(&Artifact {
                        path: decision,
                        sha256,
                    })?,
                Command::Tick => println!(
                    "{}",
                    serde_json::to_string(&controller.tick(&graph, &client).await?)?
                ),
                Command::Serve => loop {
                    let progress = controller.tick(&graph, &client).await?;
                    println!("{}", serde_json::to_string(&progress)?);
                    if progress.complete {
                        break;
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                },
                _ => bail!("not a fixed-loop operation"),
            };
            Ok(())
        })
}

//! Durable eligibility controller; native Graph/SmartCluster retain dispatch authority.
//!
//! `bind`, `bind-parent-decision`, `tick` and `serve` keep the fixed `jope.native_loop.v1`
//! contract and require the loop arguments. `plan`, `status`, `verify` and `cancel` operate
//! on the versioned `jope.native_bounded_run.v1` contract (W7-T02) and take none of them:
//! status and verify are read-only and never start numerical work.
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
            | Command::Status { .. }
            | Command::Verify { .. }
            | Command::Cancel { .. }
    ) {
        ensure!(
            !args.legacy.any(),
            "bounded-run operations take no fixed-loop arguments; the frozen plan is the only input"
        );
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

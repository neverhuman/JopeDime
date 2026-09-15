//! Durable eligibility controller; native Graph/SmartCluster retain dispatch authority.
use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use jain_smartcluster::{Client, ProjectId, TenantId};
use jope_parallel::{
    native_graph::{Artifact, Graph},
    native_loop::{Binding, Loop},
};
use std::{fs, io::Read, os::unix::fs::OpenOptionsExt, path::PathBuf, time::Duration};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    root: PathBuf,
    #[arg(long)]
    state: PathBuf,
    #[arg(long)]
    graph_state: PathBuf,
    #[arg(long)]
    worker_state: PathBuf,
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    config_sha256: String,
    #[arg(long)]
    socket: PathBuf,
    #[arg(long)]
    tenant: String,
    #[arg(long)]
    project: String,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Bind { binding: PathBuf },
    BindParentDecision { decision: PathBuf, sha256: String },
    Tick,
    Serve,
}
fn main() -> Result<()> {
    let a = Args::parse();
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
            match a.command {
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
            };
            Ok(())
        })
}

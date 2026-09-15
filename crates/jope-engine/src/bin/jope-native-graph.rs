use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use jain_smartcluster::{Client, ProjectId, TenantId};
use jope_parallel::native_graph::{Admission, Graph};
use std::{fs, io::Read, os::unix::fs::OpenOptionsExt, path::PathBuf, time::Duration};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    root: PathBuf,
    #[arg(long)]
    state: PathBuf,
    #[arg(long)]
    worker_state: PathBuf,
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
    /// Persist a frozen scientific task; SmartCluster decides execution.
    Admit { manifest: PathBuf },
    /// Reconcile one bounded batch of admissions and durable results.
    Tick,
    /// Reconcile continuously; retains the exclusive coordinator lock.
    Serve,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let graph = Graph::open(
        &args.root,
        &args.state,
        &args.worker_state,
        TenantId::new(args.tenant)?,
        ProjectId::new(args.project)?,
    )?;
    let client = Client::new(args.socket);
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(async {
            match args.command {
                Command::Admit { manifest } => {
                    let file = fs::OpenOptions::new()
                        .read(true)
                        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                        .open(manifest)?;
                    ensure!(
                        file.metadata()?.is_file(),
                        "regular admission file required"
                    );
                    let mut bytes = Vec::new();
                    file.take(1024 * 1024 + 1).read_to_end(&mut bytes)?;
                    ensure!(bytes.len() <= 1024 * 1024, "admission exceeds 1 MiB");
                    let admission: Admission = serde_json::from_slice(&bytes)?;
                    println!("{}", graph.admit(&client, &admission).await?);
                }
                Command::Tick => {
                    println!("{}", serde_json::to_string(&graph.tick(&client).await?)?)
                }
                Command::Serve => loop {
                    let tick = graph.tick(&client).await?;
                    println!("{}", serde_json::to_string(&tick)?);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                },
            }
            Ok(())
        })
}

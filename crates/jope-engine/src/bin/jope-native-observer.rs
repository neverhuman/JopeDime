use anyhow::Result;
use clap::Parser;
use jain_smartcluster::{ProjectId, TenantId};
use jope_parallel::native_observer::{self, Config};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

/// Read actual native scheduler and kernel state into an atomic read-only TUI snapshot.
#[derive(Parser)]
struct Args {
    #[arg(long)]
    socket: PathBuf,
    #[arg(long)]
    cgroup: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    tenant: String,
    #[arg(long)]
    project: String,
    #[arg(long)]
    scope: String,
    /// Publish one observation and exit; otherwise refresh every ten seconds.
    #[arg(long)]
    once: bool,
    /// Read the fixed root-protected scientific catalog in a separate verifier thread.
    #[arg(long)]
    scientific: bool,
    /// For one-shot inspection only, wait this long for initial scientific evidence.
    #[arg(long, default_value_t = 0)]
    science_wait_seconds: u64,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let config = Config {
        socket: args.socket,
        cgroup: args.cgroup,
        output: args.output,
        tenant: TenantId::new(args.tenant)?,
        project: ProjectId::new(args.project)?,
        scope: args.scope,
    };
    anyhow::ensure!(
        args.science_wait_seconds <= 600
            && (args.science_wait_seconds == 0 || (args.once && args.scientific)),
        "science wait requires one-shot scientific mode and at most600seconds"
    );
    let science = if args.scientific {
        Some(native_observer::science::Reader::start(&config)?)
    } else {
        None
    };
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(async {
            if args.science_wait_seconds > 0 {
                tokio::time::sleep(Duration::from_secs(args.science_wait_seconds)).await;
            }
            loop {
                let start = Instant::now();
                let mut snapshot = native_observer::observe(&config).await?;
                if let Some(science) = &science {
                    science.enrich(&mut snapshot);
                }
                native_observer::publish(&config, &snapshot)?;
                if args.once {
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_secs(10).saturating_sub(start.elapsed())).await;
            }
        })
}

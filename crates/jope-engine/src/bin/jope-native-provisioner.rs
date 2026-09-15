//! Root-owned exact-task provisioning; never schedules numerical work.
use anyhow::{ensure, Result};
use clap::Parser;
use std::{fs, io::Read, os::unix::fs::OpenOptionsExt, path::PathBuf};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(
        long,
        required_unless_present = "serve",
        requires = "input_sha256",
        conflicts_with = "serve"
    )]
    input: Option<PathBuf>,
    #[arg(long, requires = "input", conflicts_with = "serve")]
    input_sha256: Option<String>,
    #[arg(long)]
    serve: bool,
    #[arg(long, default_value_t = 2000)]
    poll_ms: u64,
    #[arg(long, default_value_t = 86400)]
    max_runtime_seconds: u64,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    if args.serve {
        return runtime.block_on(jope_parallel::native_provisioner::serve(
            &args.config,
            args.poll_ms,
            args.max_runtime_seconds,
        ));
    }
    let input_path = args
        .input
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("input required"))?;
    let input_sha = args
        .input_sha256
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("input digest required"))?;
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(input_path)?;
    ensure!(
        file.metadata()?.is_file() && file.metadata()?.len() <= 1024 * 1024,
        "bounded regular provisioner input required"
    );
    let mut bytes = Vec::new();
    file.take(1024 * 1024 + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 1024 * 1024 && jope_parallel::manifest::hash(&bytes) == input_sha,
        "provisioner input changed"
    );
    let input = serde_json::from_slice(&bytes)?;
    let result = runtime.block_on(jope_parallel::native_provisioner::provision(
        &args.config,
        &input,
    ))?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

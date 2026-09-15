//! Read-only eligibility evidence; this executable never signs or dispatches work.
use anyhow::{ensure, Result};
use clap::Parser;
use jope_parallel::native_loop::{GrantProofInputs, Loop};
use std::{fs, io::Read, os::unix::fs::OpenOptionsExt, path::PathBuf};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    input_sha256: String,
    #[arg(long)]
    socket: PathBuf,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&args.input)?;
    ensure!(
        file.metadata()?.is_file() && file.metadata()?.len() <= 1024 * 1024,
        "bounded regular proof input required"
    );
    let mut bytes = Vec::new();
    file.take(1024 * 1024 + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 1024 * 1024 && jope_parallel::manifest::hash(&bytes) == args.input_sha256,
        "proof input changed"
    );
    let input: GrantProofInputs = serde_json::from_slice(&bytes)?;
    let client = jain_smartcluster::Client::new(args.socket);
    let proof = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(Loop::prove_native_grant_request(&input, &client))?;
    println!("{}", serde_json::to_string(&proof)?);
    Ok(())
}

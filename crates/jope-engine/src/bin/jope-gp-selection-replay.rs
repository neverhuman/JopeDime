//! Independent GP selector and stop replay over one completed search directory.
//! Read-only: it never fits, writes or publishes, and exits nonzero when the
//! ledger or its publication disagrees with `parallel.gp_selection_contract.v1`.
use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// The search's declared artifact root (its `search-request.json` binds it).
    #[arg(long)]
    root: PathBuf,
    /// The completed search output directory.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let a = Args::parse();
    let report = jope_parallel::gp::selection_replay::verify_search(&a.root, &a.output)?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

//! Fixed CPU packer for independently verified continuation targets.
use anyhow::{ensure, Result};
use clap::Parser;
use jope_parallel::{continuation_corpus, native_graph::Artifact};
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    root: PathBuf,
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    manifest_sha256: String,
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        !cfg!(feature = "neural"),
        "continuation packer requires qualified CPU binary"
    );
    continuation_corpus::build(
        &a.root,
        &Artifact {
            path: a.manifest,
            sha256: a.manifest_sha256,
        },
        &a.output,
    )
}

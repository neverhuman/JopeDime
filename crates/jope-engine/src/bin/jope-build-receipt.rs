//! Writes a `jope.product_evaluator_build.v1` receipt for a built binary.
//! Read-only with respect to the binary and source tree; `git` and `rustc`
//! run only as inspecting subprocesses.
use anyhow::{Context, Result};
use clap::Parser;
use jope_parallel::build_receipt::{build_receipt, check_bounds, ReceiptInput};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// The built binary to describe.
    #[arg(long)]
    binary: PathBuf,
    /// The Cargo source directory the binary was built from.
    #[arg(long)]
    source_dir: PathBuf,
    /// Where to write the receipt (must not already exist).
    #[arg(long)]
    output: PathBuf,
    /// Evaluator lineage name, e.g. `kernel-v1-8aa0affbf…`.
    #[arg(long)]
    lineage: Option<String>,
    /// File holding the builder's captured `rustc -vV` (default: run local rustc).
    #[arg(long)]
    rustc_vv: Option<PathBuf>,
    /// The exact build command, recorded verbatim.
    #[arg(long)]
    command: Option<String>,
    /// The builder container image, recorded verbatim.
    #[arg(long)]
    builder_image: Option<String>,
    /// Fail (after writing the receipt) if the binary requires a GLIBC
    /// symbol version newer than this, e.g. `2.36`.
    #[arg(long)]
    max_glibc: Option<String>,
    /// Fail (after writing the receipt) if any NEEDED entry is libcuda,
    /// libtorch or libc10.
    #[arg(long)]
    forbid_gpu_libraries: bool,
}

fn main() -> Result<()> {
    let a = Args::parse();
    let rustc_vv = a
        .rustc_vv
        .as_ref()
        .map(|p| std::fs::read_to_string(p).with_context(|| format!("read {}", p.display())))
        .transpose()?;
    let receipt = build_receipt(&ReceiptInput {
        binary: &a.binary,
        source_dir: &a.source_dir,
        lineage: a.lineage.as_deref(),
        rustc_vv,
        command: a.command.as_deref(),
        builder_image: a.builder_image.as_deref(),
    })?;
    let mut text = serde_json::to_string_pretty(&receipt)?;
    text.push('\n');
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&a.output)
        .with_context(|| format!("create {}", a.output.display()))?;
    std::io::Write::write_all(&mut file, text.as_bytes())?;
    print!("{text}");
    check_bounds(&receipt, a.max_glibc.as_deref(), a.forbid_gpu_libraries)
}

//! Writes a `jope.product_evaluator_build.v1` receipt for a built binary.
//! Read-only with respect to the binary and source tree; `git` and `rustc`
//! run only as inspecting subprocesses.
use anyhow::{Context, Result};
use clap::Parser;
use jope_parallel::build_receipt::{apply_bounds, build_receipt, ReceiptInput};
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
    /// File holding the builder's `RUSTUP_TOOLCHAIN` value (empty when unset),
    /// which overrides the source's rust-toolchain.toml. Only read together
    /// with --rustc-vv; otherwise the local environment's value is recorded.
    #[arg(long, requires = "rustc_vv")]
    rustup_toolchain: Option<PathBuf>,
    /// The exact build command, recorded verbatim.
    #[arg(long)]
    command: Option<String>,
    /// The builder container image, recorded verbatim.
    #[arg(long)]
    builder_image: Option<String>,
    /// Cargo profile the binary was built with.
    #[arg(long, default_value = "release")]
    profile: String,
    /// A Cargo feature enabled for the build (repeatable).
    #[arg(long = "feature")]
    features: Vec<String>,
    /// Fail if the binary requires a GLIBC symbol version newer than this,
    /// e.g. `2.36`. The limit and verdict are recorded under `bounds`.
    #[arg(long)]
    max_glibc: Option<String>,
    /// Fail if any NEEDED entry is a CUDA/NVIDIA/LibTorch library, or the
    /// binary has no dynamic section to check. Recorded under `bounds`.
    #[arg(long)]
    forbid_gpu_libraries: bool,
}

fn read(path: &Option<PathBuf>) -> Result<Option<String>> {
    path.as_ref()
        .map(|p| std::fs::read_to_string(p).with_context(|| format!("read {}", p.display())))
        .transpose()
}

fn main() -> Result<()> {
    let a = Args::parse();
    let rustc_vv = read(&a.rustc_vv)?;
    let rustup_toolchain = read(&a.rustup_toolchain)?;
    let mut receipt = build_receipt(&ReceiptInput {
        binary: &a.binary,
        source_dir: &a.source_dir,
        lineage: a.lineage.as_deref(),
        rustc_vv,
        rustup_toolchain: rustup_toolchain.as_deref().map(str::trim),
        command: a.command.as_deref(),
        builder_image: a.builder_image.as_deref(),
        profile: &a.profile,
        features: &a.features,
    })?;
    // The verdict is computed first and written into the receipt, so a
    // rejected build leaves a receipt that says `bounds.passed: false`.
    let verdict = apply_bounds(&mut receipt, a.max_glibc.as_deref(), a.forbid_gpu_libraries);
    let mut text = serde_json::to_string_pretty(&receipt)?;
    text.push('\n');
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&a.output)
        .with_context(|| format!("create {}", a.output.display()))?;
    std::io::Write::write_all(&mut file, text.as_bytes())?;
    print!("{text}");
    verdict
}

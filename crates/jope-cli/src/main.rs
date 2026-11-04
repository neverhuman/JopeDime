//! `jope`: the one command line. Fixture, kernel, champion and router verbs live in this
//! crate; `jope models …`, `jope datasets …`, `jope pipeline …` and `jope motifs …` delegate
//! to the library crates' own clap commands (no process is spawned).

use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use jope_cli::{
    emit_genome_with_train, fit_dataset, generate_fixtures, load_router, local_champions,
    local_train_step, refuse_s3, resolve_root, route_dataset, search_dataset, verify_inputs,
    write_json, Verification, DEFAULT_GENOME,
};
use std::{path::PathBuf, process::ExitCode};

mod heap_meter;

#[cfg(target_os = "linux")]
mod train;

#[derive(Parser)]
#[command(
    name = "jope",
    version,
    about = "JOPE-Dime command line for diagnostics, fixtures, benchmarking, optimization, training, encoding and fitting, models, datasets, and the product pipeline"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

include!("main/commands.rs");

include!("main/dispatch.rs");

include!("main/entry.rs");

//! `jope-datasets`: ingest, fetch, verify and link content-addressed datasets.

use clap::Parser;
use std::process::ExitCode;

fn main() -> ExitCode {
    jope_datasets::cli::exec(jope_datasets::cli::Cli::parse())
}

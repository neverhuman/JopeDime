//! `jope-models`: fetch, verify, publish and locate model bundles.

use clap::Parser;
use std::process::ExitCode;

fn main() -> ExitCode {
    jope_models::cli::exec(jope_models::cli::Cli::parse())
}

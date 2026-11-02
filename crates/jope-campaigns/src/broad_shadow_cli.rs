//! Rust entry point for the retained Python broad-shadow command.
//!
//! `jope.cli` remains the sole product importer, so this slice is PORT-KEEP.
//! Forwarding the original arguments through the installed `jope-py` entry
//! point preserves the existing argparse behavior while Rust callers migrate.

use anyhow::{Context, Result};
use std::ffi::{OsStr, OsString};
use std::process::{Command, ExitStatus};

/// Run `jope broad-shadow` through the existing Python implementation.
///
/// The Python module remains byte-identical until its product caller is moved.
pub fn run<I, S>(args: I) -> Result<ExitStatus>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let forwarded = forwarded_args(args);
    Command::new("jope-py")
        .args(forwarded)
        .status()
        .context("failed to launch jope-py broad-shadow")
}

fn forwarded_args<I, S>(args: I) -> Vec<OsString>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    std::iter::once(OsString::from("broad-shadow"))
        .chain(args.into_iter().map(|arg| arg.as_ref().to_os_string()))
        .collect()
}

#[cfg(test)]
mod tests;

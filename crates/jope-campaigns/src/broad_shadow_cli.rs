//! Rust entry point for the retained broad-shadow command.
//!
//! `jope.cli` remains the sole product importer, so this slice is PORT-KEEP.
//! Catalog and probe-slate validation use their Rust ports. Other operations
//! forward original arguments through `jope-py` until their Rust backends land.

use anyhow::{Context, Result};
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::process::Command;

/// Run `jope broad-shadow`, using available Rust validators and Python fallbacks.
///
/// The Python module remains byte-identical until its product caller is moved.
pub fn run<I, S>(args: I) -> Result<i32>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let args: Vec<OsString> = args
        .into_iter()
        .map(|arg| arg.as_ref().to_os_string())
        .collect();
    if run_native_validation(&args) {
        return Ok(0);
    }
    let status = Command::new("jope-py")
        .args(forwarded_args(args))
        .status()
        .context("failed to launch jope-py broad-shadow")?;
    Ok(status.code().unwrap_or(1))
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ValidationKind {
    Catalog,
    ProbeSlate,
}

#[derive(Debug, Eq, PartialEq)]
struct ValidationRequest {
    kind: ValidationKind,
    path: PathBuf,
    production: bool,
}

fn parse_native_validation(args: &[OsString]) -> Option<ValidationRequest> {
    if args.first()?.to_str()? != "validate" {
        return None;
    }
    let kind = match args.get(1)?.to_str()? {
        "catalog" => ValidationKind::Catalog,
        "slate" => ValidationKind::ProbeSlate,
        _ => return None,
    };
    let mut path = None;
    let mut production = false;
    for arg in &args[2..] {
        if arg == OsStr::new("--production") {
            production = true;
        } else if arg.to_string_lossy().starts_with('-') || path.is_some() {
            return None;
        } else {
            path = Some(PathBuf::from(arg));
        }
    }
    Some(ValidationRequest {
        kind,
        path: path?,
        production,
    })
}

fn run_native_validation(args: &[OsString]) -> bool {
    let Some(request) = parse_native_validation(args) else {
        return false;
    };
    let Ok(bytes) = std::fs::read(request.path) else {
        return false;
    };
    let Ok(payload) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return false;
    };
    let valid = match request.kind {
        ValidationKind::Catalog => {
            crate::broad_shadow_catalog::validate_broad_catalog(&payload, request.production)
                .is_ok()
        }
        ValidationKind::ProbeSlate => {
            crate::broad_shadow::contracts_slate::validate_probe_slate(&payload, request.production)
                .is_ok()
        }
    };
    if valid {
        println!("valid");
    }
    valid
}

#[cfg(test)]
mod tests;

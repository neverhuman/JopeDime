//! Input files, each read once and pinned by sha256.
//!
//! A declared input that cannot be read, whose `.sha256` sidecar disagrees, or whose sha256 does
//! not start with the expected prefix aborts the run: no plan is written from inputs that cannot
//! be identified.

use crate::canon::sha256_hex;
use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
/// `InputRecord` is a record for the input record contract in this module: Input files, each read once and pinned by sha256.
pub struct InputRecord {
    /// Data or authority role assigned to this value.
    pub role: String,
    /// Filesystem path associated with this input or output.
    pub path: String,
    /// Encoded byte count associated with this object or operation.
    pub bytes: u64,
    /// Sha-256 recorded for this input record.
    pub sha256: String,
    /// `verified <sidecar path>`, `absent`, plus `expected-prefix <p> ok` when one was given.
    pub check: String,
}

/// `Loaded` is a record for the loaded contract in this module: Input files, each read once and pinned by sha256.
pub struct Loaded {
    /// Record recorded for this loaded.
    pub record: InputRecord,
    /// Encoded byte count associated with this object or operation.
    pub bytes: Vec<u8>,
}

/// `ROLE=PATH[@SHA256PREFIX]`, as given on the command line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputSpec {
    /// Data or authority role assigned to this value.
    pub role: String,
    /// Filesystem path associated with this input or output.
    pub path: PathBuf,
    /// Optional expect prefix value; `None` means it is not available.
    pub expect_prefix: Option<String>,
}

impl InputSpec {
    /// Parses the supplied representation, returning an error when the input is invalid.
    pub fn parse(s: &str) -> Result<InputSpec> {
        let (role, rest) = s
            .split_once('=')
            .with_context(|| format!("{s:?} is not ROLE=PATH"))?;
        ensure!(
            !role.is_empty()
                && role
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'),
            "role {role:?} must be [a-z0-9-]+"
        );
        let (path, expect_prefix) = match rest.rsplit_once('@') {
            Some((p, sha)) if !sha.is_empty() && sha.bytes().all(|b| b.is_ascii_hexdigit()) => {
                ensure!(
                    sha.len() >= 8,
                    "expected sha256 prefix {sha:?} is shorter than 8 hex digits"
                );
                (p, Some(sha.to_ascii_lowercase()))
            }
            _ => (rest, None),
        };
        ensure!(!path.is_empty(), "{s:?} has an empty path");
        Ok(InputSpec {
            role: role.to_string(),
            path: PathBuf::from(path),
            expect_prefix,
        })
    }
}

fn sidecars(path: &Path) -> Vec<PathBuf> {
    let mut out = vec![PathBuf::from(format!("{}.sha256", path.display()))];
    if path.extension().is_some() {
        out.push(path.with_extension("sha256"));
    }
    out
}

/// Reads `path`, hashes it, and checks the sidecar and the expected prefix.
pub fn load(role: &str, path: &Path, expect_prefix: Option<&str>) -> Result<Loaded> {
    let bytes =
        fs::read(path).with_context(|| format!("input {role}: cannot read {}", path.display()))?;
    let sha = sha256_hex(&bytes);
    let mut check = String::from("absent");
    for side in sidecars(path) {
        if !side.is_file() {
            continue;
        }
        let text = fs::read_to_string(&side)
            .with_context(|| format!("cannot read sidecar {}", side.display()))?;
        let first = text
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        if first != sha {
            bail!(
                "input {role}: {} sha256 {sha} disagrees with sidecar {} ({first})",
                path.display(),
                side.display()
            );
        }
        check = format!("verified {}", side.display());
        break;
    }
    if let Some(p) = expect_prefix {
        if !sha.starts_with(p) {
            bail!(
                "input {role}: {} sha256 {sha} does not start with the expected {p}",
                path.display()
            );
        }
        check.push_str(&format!("; expected-prefix {p} ok"));
    }
    Ok(Loaded {
        record: InputRecord {
            role: role.to_string(),
            path: path.display().to_string(),
            bytes: bytes.len() as u64,
            sha256: sha,
            check,
        },
        bytes,
    })
}

/// Reads spec from the configured source.
pub fn load_spec(spec: &InputSpec) -> Result<Loaded> {
    load(&spec.role, &spec.path, spec.expect_prefix.as_deref())
}

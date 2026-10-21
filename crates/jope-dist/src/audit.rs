//! The deployment dependency fence (SPECIFICATION.md §14.1): the *selected* package, target and
//! feature set — not the research workspace — must not reach a forbidden crate, and every crate
//! that builds native (non-Rust) code must be an approved exception.
//!
//! `cargo tree` answers the question per selection. `cargo metadata`'s resolve cannot: it unifies
//! features across the whole workspace (`crates/jope-pipeline/tests/dependency_fence.rs`).
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

pub const POLICY_SCHEMA: &str = "jope.appliance-dependency-policy/v1";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub schema: String,
    pub package: String,
    pub target: String,
    /// Feature sets audited, each a list of the package's features.
    pub feature_sets: Vec<Vec<String>>,
    /// Package names (or `prefix*`) that must not appear in the normal or build closure.
    pub deny: Vec<String>,
    /// Packages known to compile or link non-Rust code, refused unless listed in `native_allowed`.
    pub native: Vec<String>,
    /// Approved native exceptions, `name = "reason"`.
    #[serde(default)]
    pub native_allowed: BTreeMap<String, String>,
}

impl Policy {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let policy: Policy =
            toml::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        ensure!(
            policy.schema == POLICY_SCHEMA,
            "{} is not {POLICY_SCHEMA}",
            path.display()
        );
        Ok(policy)
    }
}

fn matches(pattern: &str, name: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => name.starts_with(prefix),
        None => pattern == name,
    }
}

/// Package names `cargo tree` resolves for one selection (normal and build edges).
pub fn closure(
    manifest: &Path,
    package: &str,
    target: &str,
    features: &[String],
) -> Result<BTreeSet<String>> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let mut command = Command::new(cargo);
    command.args(["tree", "--locked", "--offline", "--manifest-path"]);
    command.arg(manifest);
    command.args([
        "-p",
        package,
        "--target",
        target,
        "--edges",
        "normal,build",
        "--prefix",
        "none",
        "--format",
        "{p}",
        "--no-default-features",
    ]);
    if !features.is_empty() {
        command.args(["--features", &features.join(",")]);
    }
    let output = command.output().context("run cargo tree")?;
    ensure!(
        output.status.success(),
        "cargo tree failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_owned)
        .collect())
}

/// Every violation of the policy for every feature set, as readable lines; empty means clean.
pub fn audit(manifest: &Path, policy: &Policy) -> Result<Vec<String>> {
    let mut violations = Vec::new();
    for features in &policy.feature_sets {
        let names = closure(manifest, &policy.package, &policy.target, features)?;
        ensure!(
            names.contains(&policy.package),
            "cargo tree did not resolve {}",
            policy.package
        );
        let label = if features.is_empty() {
            "no features".to_owned()
        } else {
            features.join(",")
        };
        for name in &names {
            if let Some(pattern) = policy.deny.iter().find(|p| matches(p, name)) {
                violations.push(format!("[{label}] {name} is denied ({pattern})"));
            }
            if policy.native.iter().any(|p| matches(p, name))
                && !policy.native_allowed.contains_key(name)
            {
                violations.push(format!(
                    "[{label}] {name} builds native code and is not an approved exception"
                ));
            }
        }
    }
    Ok(violations)
}

//! The third-party notice inventory embedded in every package (`jopedime licenses`,
//! SPECIFICATION.md §3.1): each crate in the appliance's normal closure for the selected target
//! and features, with its declared licence and the licence and notice files it ships. The MIT,
//! BSD and Apache licences require those texts to travel with a binary distribution.
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

pub const SCHEMA: &str = "jope.appliance-notices/v1";

#[derive(Serialize)]
pub struct Notices {
    pub schema: &'static str,
    pub package: String,
    pub target: String,
    pub features: Vec<String>,
    pub crates: Vec<CrateNotice>,
    /// Licence and notice texts by sha256, shared between crates that ship identical files.
    pub texts: BTreeMap<String, String>,
}

#[derive(Serialize)]
pub struct CrateNotice {
    pub name: String,
    pub version: String,
    pub license: String,
    pub repository: String,
    /// `file name -> sha256` into `texts`.
    pub files: BTreeMap<String, String>,
}

fn cargo() -> Command {
    Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned()))
}

fn is_notice_file(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    [
        "LICENSE",
        "LICENCE",
        "COPYING",
        "NOTICE",
        "COPYRIGHT",
        "UNLICENSE",
    ]
    .iter()
    .any(|prefix| upper.starts_with(prefix))
}

/// Build the inventory from `cargo tree` (the exact selection) and `cargo metadata` (where each
/// crate's files are).
pub fn collect(
    manifest: &Path,
    package: &str,
    target: &str,
    features: &[String],
) -> Result<Notices> {
    let selected = crate::audit::closure(manifest, package, target, features)?;
    // `{p}` gives "name vX.Y.Z (source)"; keep name and version.
    let mut command = cargo();
    command.args(["tree", "--locked", "--offline", "--manifest-path"]);
    command.arg(manifest);
    command.args([
        "-p",
        package,
        "--target",
        target,
        "--edges",
        "normal",
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
    let mut wanted = BTreeSet::new();
    for line in String::from_utf8(output.stdout)?.lines() {
        let mut parts = line.split_whitespace();
        if let (Some(name), Some(version)) = (parts.next(), parts.next()) {
            if selected.contains(name) {
                wanted.insert((name.to_owned(), version.trim_start_matches('v').to_owned()));
            }
        }
    }
    let mut command = cargo();
    command.args([
        "metadata",
        // Inventory all optional packages, then retain only the exact selected closure above.
        "--all-features",
        "--locked",
        "--offline",
        "--format-version",
        "1",
        "--manifest-path",
    ]);
    command.arg(manifest);
    let output = command.output().context("run cargo metadata")?;
    ensure!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let mut crates = Vec::new();
    let mut texts = BTreeMap::new();
    for p in metadata["packages"]
        .as_array()
        .context("metadata packages")?
    {
        let name = p["name"].as_str().unwrap_or_default().to_owned();
        let version = p["version"].as_str().unwrap_or_default().to_owned();
        if !wanted.contains(&(name.clone(), version.clone())) {
            continue;
        }
        let dir = Path::new(p["manifest_path"].as_str().unwrap_or_default())
            .parent()
            .context("manifest has a directory")?
            .to_owned();
        let mut files = BTreeMap::new();
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .with_context(|| format!("read {}", dir.display()))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| is_notice_file(n))
            .collect();
        entries.sort();
        for file in entries {
            let text = std::fs::read_to_string(dir.join(&file)).unwrap_or_default();
            if text.is_empty() {
                continue;
            }
            let sha = jope_assets::sha256_hex(text.as_bytes());
            texts.entry(sha.clone()).or_insert(text);
            files.insert(file, sha);
        }
        crates.push(CrateNotice {
            name,
            version,
            license: p["license"].as_str().unwrap_or("NOASSERTION").to_owned(),
            repository: p["repository"].as_str().unwrap_or_default().to_owned(),
            files,
        });
    }
    crates.sort_by(|a, b| (&a.name, &a.version).cmp(&(&b.name, &b.version)));
    ensure!(
        crates.len() == wanted.len(),
        "cargo metadata lacks {} of the {} selected crates",
        wanted.len() - crates.len(),
        wanted.len()
    );
    Ok(Notices {
        schema: SCHEMA,
        package: package.to_owned(),
        target: target.to_owned(),
        features: features.to_vec(),
        crates,
        texts,
    })
}

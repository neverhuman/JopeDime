//! `jope-datasets verify --repo <dir>`: the CI check for the committed test datasets.

use crate::catalog::{CatalogRow, Collection};
use crate::hash::{digest_file, looks_headerless, sha256_hex};
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const MAX_DATASETS: usize = 10;
pub const MAX_DATASET_BYTES: u64 = 600 * 1024;
pub const MAX_TOTAL_BYTES: u64 = 1536 * 1024;

#[derive(Clone, Debug, Default)]
pub struct RepoReport {
    pub datasets: usize,
    pub total_bytes: u64,
    pub collection: Option<String>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

fn dir_bytes(dir: &Path) -> Result<u64> {
    let mut total = 0;
    for e in std::fs::read_dir(dir)? {
        let e = e?;
        if e.file_type()?.is_file() {
            total += e.metadata()?.len();
        }
    }
    Ok(total)
}

/// Locate the collection document: `<repo>/manifest.json`, else `<repo>/../manifest.json`.
pub fn find_collection(repo: &Path) -> Option<PathBuf> {
    [
        repo.join("manifest.json"),
        repo.join("..").join("manifest.json"),
    ]
    .into_iter()
    .find(|p| p.is_file())
}

/// Optional SOTArena registry (`registry/datasets.json`) cross-check: `id -> (train sha, test sha)`.
fn load_registry(path: &Path) -> Result<BTreeMap<String, (String, String)>> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let rows = v
        .get("datasets")
        .and_then(|d| d.as_array())
        .cloned()
        .unwrap_or_default();
    let mut out = BTreeMap::new();
    for r in rows {
        let id = r
            .get("id")
            .and_then(|s| s.as_str())
            .unwrap_or_default()
            .to_string();
        let sha = |k: &str| {
            r.pointer(&format!("/files/{k}/sha256"))
                .and_then(|s| s.as_str())
                .unwrap_or_default()
                .to_string()
        };
        out.insert(id, (sha("train"), sha("test")));
    }
    Ok(out)
}

#[allow(clippy::too_many_lines)]
pub fn verify_repo(repo: &Path, registry: Option<&Path>) -> Result<RepoReport> {
    let mut rep = RepoReport::default();
    let collection = match find_collection(repo) {
        Some(p) => match std::fs::read(&p)
            .map_err(|e| vec![e.to_string()])
            .and_then(|b| {
                serde_json::from_slice::<serde_json::Value>(&b).map_err(|e| vec![e.to_string()])
            })
            .and_then(|v| Collection::validate_value(&v))
        {
            Ok(c) => {
                rep.collection = Some(c.name.clone());
                Some(c)
            }
            Err(e) => {
                rep.errors
                    .push(format!("{}: {}", p.display(), e.join("; ")));
                None
            }
        },
        None => {
            rep.errors.push(format!(
                "no manifest.json collection next to or above {}",
                repo.display()
            ));
            None
        }
    };
    let registry = match registry {
        Some(p) => {
            Some(load_registry(p).with_context(|| format!("load registry {}", p.display()))?)
        }
        None => None,
    };
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(repo)
        .with_context(|| format!("read_dir {}", repo.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    let mut rows: Vec<CatalogRow> = Vec::new();
    for dir in &dirs {
        let slug = dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let meta_path = dir.join("metadata.json");
        if !meta_path.is_file() {
            rep.errors.push(format!("{slug}: metadata.json is missing"));
            continue;
        }
        rep.datasets += 1;
        let bytes = dir_bytes(dir)?;
        rep.total_bytes += bytes;
        if bytes > MAX_DATASET_BYTES {
            rep.errors.push(format!(
                "{slug}: {bytes} bytes exceeds the per-dataset cap {MAX_DATASET_BYTES}"
            ));
        }
        let row = match std::fs::read(&meta_path)
            .map_err(|e| vec![e.to_string()])
            .and_then(|b| {
                serde_json::from_slice::<serde_json::Value>(&b).map_err(|e| vec![e.to_string()])
            })
            .and_then(|v| CatalogRow::validate_value(&v))
        {
            Ok(r) => r,
            Err(e) => {
                for m in e {
                    rep.errors.push(format!("{slug}/metadata.json: {m}"));
                }
                continue;
            }
        };
        for (file, f) in row.files.entries() {
            let path = dir.join(file);
            if !path.is_file() {
                if file == "meta.json" {
                    rep.warnings.push(format!(
                        "{slug}: files.meta is set but meta.json is not committed"
                    ));
                } else {
                    rep.errors.push(format!("{slug}/{file}: missing"));
                }
                continue;
            }
            let d = digest_file(&path)?;
            if d.sha256 != f.sha256 {
                rep.errors.push(format!(
                    "{slug}/{file}: sha256 {} != metadata {}",
                    d.sha256, f.sha256
                ));
            }
            if d.bytes != f.bytes {
                rep.errors.push(format!(
                    "{slug}/{file}: {} bytes != metadata {}",
                    d.bytes, f.bytes
                ));
            }
            if file.ends_with(".csv") {
                if f.rows != Some(d.rows) {
                    rep.errors.push(format!(
                        "{slug}/{file}: {} rows != metadata {:?}",
                        d.rows, f.rows
                    ));
                }
                if f.columns != Some(d.columns) {
                    rep.errors.push(format!(
                        "{slug}/{file}: {} columns != metadata {:?}",
                        d.columns, f.columns
                    ));
                }
                if !looks_headerless(&d.first_line) {
                    rep.errors.push(format!(
                        "{slug}/{file}: first line looks like a header: {:?}",
                        d.first_line
                    ));
                }
                if d.columns != row.shape.features + 1 {
                    rep.errors.push(format!(
                        "{slug}/{file}: {} columns != features {} + 1",
                        d.columns, row.shape.features
                    ));
                }
            }
        }
        let license_path = dir.join("LICENSE");
        match std::fs::read(&license_path) {
            Ok(text) => {
                if let Some(want) = &row.provenance.license.text_sha256 {
                    let got = sha256_hex(&text);
                    if &got != want {
                        rep.errors.push(format!(
                            "{slug}/LICENSE: sha256 {got} != license.text_sha256 {want}"
                        ));
                    }
                } else {
                    rep.warnings
                        .push(format!("{slug}: license.text_sha256 is not recorded"));
                }
            }
            Err(_) => rep.errors.push(format!("{slug}/LICENSE: missing")),
        }
        if let Some(reg) = &registry {
            match reg.get(&row.id) {
                Some((train, test)) => {
                    if train != &row.files.train.sha256 || test != &row.files.test.sha256 {
                        rep.errors
                            .push(format!("{slug}: sha256 differs from the SOTArena registry"));
                    }
                }
                None => rep
                    .warnings
                    .push(format!("{slug}: id {} is not in the registry", row.id)),
            }
        }
        if let Some(c) = &collection {
            if !c.members.contains(&row.id) {
                rep.errors.push(format!(
                    "{slug}: id {} is not a member of collection {}",
                    row.id, c.name
                ));
            }
            if !row.collections.contains(&c.name) {
                rep.warnings
                    .push(format!("{slug}: row.collections does not list {}", c.name));
            }
        }
        rows.push(row);
    }
    if rep.datasets > MAX_DATASETS {
        rep.errors.push(format!(
            "{} datasets exceeds the cap {MAX_DATASETS}",
            rep.datasets
        ));
    }
    if rep.total_bytes > MAX_TOTAL_BYTES {
        rep.errors.push(format!(
            "{} total bytes exceeds the cap {MAX_TOTAL_BYTES}",
            rep.total_bytes
        ));
    }
    if let Some(c) = &collection {
        for m in &c.members {
            if !rows.iter().any(|r| &r.id == m) {
                rep.errors.push(format!(
                    "collection {}: member {m} has no directory under {}",
                    c.name,
                    repo.display()
                ));
            }
        }
        let refs: Vec<&CatalogRow> = rows.iter().collect();
        let (errors, warnings) = c.license_findings(&refs);
        rep.errors.extend(errors);
        rep.warnings.extend(warnings);
    }
    Ok(rep)
}

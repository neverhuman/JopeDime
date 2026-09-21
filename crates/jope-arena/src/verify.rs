//! `verify-results <dir>`: every file a results directory's `evidence-manifest.json` lists must
//! be present with the recorded byte count and SHA-256, and every results file present must be
//! listed (or be the manifest itself / the derived `leaderboard.csv`).
//!
//! The manifest (`sotarena.golden45-evidence-manifest.v1`) records paths relative to the
//! benchmark root under `files[]` and `published_artifacts[]`; both lists are checked. Paths
//! outside the results directory (registry, restricted data bundles) are checked when present
//! under the root the directory belongs to and reported as `absent` otherwise — a results
//! snapshot published on its own does not carry them.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::io::{read_json, sha256_file};

/// One checked file.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct FileCheck {
    /// Path as recorded.
    pub path: String,
    /// `ok`, `missing`, `absent` (outside the snapshot), `sha256-mismatch`, `bytes-mismatch`.
    pub status: String,
    /// Recorded SHA-256.
    pub expected_sha256: String,
    /// Actual SHA-256 when the file exists.
    pub actual_sha256: Option<String>,
    /// Recorded byte count.
    pub expected_bytes: u64,
    /// Actual byte count when the file exists.
    pub actual_bytes: Option<u64>,
}

/// The outcome of one verification.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Verification {
    /// `jope.arena-verify/v1`.
    pub schema: String,
    /// The results directory.
    pub results_dir: String,
    /// The manifest's schema id.
    pub manifest_schema: String,
    /// Every listed file.
    pub files: Vec<FileCheck>,
    /// Files inside the directory the manifest does not list.
    pub unlisted: Vec<String>,
    /// Counts by status.
    pub ok: usize,
    /// Missing inside the snapshot.
    pub missing: usize,
    /// Outside the snapshot and not present.
    pub absent: usize,
    /// Hash or size mismatches.
    pub mismatched: usize,
}

impl Verification {
    /// True when every listed file inside the snapshot is present and matches. Unlisted files
    /// are reported; [`Verification::passed_strict`] also refuses them.
    pub fn passed(&self) -> bool {
        self.missing == 0 && self.mismatched == 0
    }

    /// [`Verification::passed`] and nothing unlisted.
    pub fn passed_strict(&self) -> bool {
        self.passed() && self.unlisted.is_empty()
    }
}

#[derive(Deserialize)]
struct ManifestFile {
    path: String,
    sha256: String,
    bytes: u64,
}

/// Files that live in a results directory without being evidence.
const DERIVED: [&str; 2] = ["evidence-manifest.json", "leaderboard.csv"];

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .collect::<std::io::Result<_>>()?;
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}

/// Locate the benchmark root a results directory belongs to: the manifest's paths are relative
/// to it (`results/<track>/…`), so it is the ancestor of `dir` under which the most listed paths
/// exist (nearest ancestor on ties); `None` when no ancestor resolves any of them.
fn benchmark_root(dir: &Path, listed: &[ManifestFile]) -> Option<PathBuf> {
    let canonical = dir.canonicalize().ok()?;
    let mut best: Option<(usize, PathBuf)> = None;
    for ancestor in canonical.ancestors() {
        let hits = listed
            .iter()
            .filter(|file| ancestor.join(&file.path).is_file())
            .count();
        if hits > 0 && best.as_ref().is_none_or(|(most, _)| hits > *most) {
            best = Some((hits, ancestor.to_path_buf()));
        }
    }
    best.map(|(_, root)| root)
}

/// Verify `dir` against its `evidence-manifest.json`.
pub fn verify_results(dir: &Path) -> Result<Verification> {
    ensure!(dir.is_dir(), "{} is not a directory", dir.display());
    let manifest_path = dir.join("evidence-manifest.json");
    ensure!(
        manifest_path.is_file(),
        "{} has no evidence-manifest.json",
        dir.display()
    );
    let manifest: Value = read_json(&manifest_path)?;
    let manifest_schema = manifest["schema"].as_str().unwrap_or_default().to_owned();
    let mut listed: Vec<ManifestFile> = Vec::new();
    for key in ["files", "published_artifacts"] {
        if let Some(items) = manifest[key].as_array() {
            for item in items {
                listed.push(
                    serde_json::from_value(item.clone())
                        .with_context(|| format!("evidence-manifest.json {key} entry"))?,
                );
            }
        }
    }
    ensure!(
        !listed.is_empty(),
        "{} lists no files",
        manifest_path.display()
    );
    let root = benchmark_root(dir, &listed);
    let canonical_dir = dir.canonicalize()?;
    let results_prefix = root
        .as_ref()
        .and_then(|root| canonical_dir.strip_prefix(root).ok())
        .map(|p| p.to_string_lossy().into_owned());

    let mut files = Vec::with_capacity(listed.len());
    let mut covered: BTreeSet<PathBuf> = BTreeSet::new();
    let (mut ok, mut missing, mut absent, mut mismatched) = (0, 0, 0, 0);
    for entry in listed {
        let inside = results_prefix
            .as_deref()
            .is_some_and(|prefix| Path::new(&entry.path).starts_with(prefix));
        let actual_path = match (&root, inside) {
            (Some(root), _) => root.join(&entry.path),
            (None, _) => dir.join(Path::new(&entry.path).file_name().unwrap_or_default()),
        };
        let mut check = FileCheck {
            path: entry.path.clone(),
            status: String::new(),
            expected_sha256: entry.sha256.clone(),
            actual_sha256: None,
            expected_bytes: entry.bytes,
            actual_bytes: None,
        };
        if actual_path.is_file() {
            let (sha, bytes) = sha256_file(&actual_path)?;
            check.actual_sha256 = Some(sha.clone());
            check.actual_bytes = Some(bytes);
            check.status = if sha != entry.sha256 {
                mismatched += 1;
                "sha256-mismatch".to_owned()
            } else if bytes != entry.bytes {
                mismatched += 1;
                "bytes-mismatch".to_owned()
            } else {
                ok += 1;
                "ok".to_owned()
            };
            if let Ok(canonical) = actual_path.canonicalize() {
                covered.insert(canonical);
            }
        } else if inside || root.is_none() {
            missing += 1;
            check.status = "missing".to_owned();
        } else {
            absent += 1;
            check.status = "absent".to_owned();
        }
        files.push(check);
    }

    let mut present = Vec::new();
    walk(dir, &mut present)?;
    let mut unlisted = Vec::new();
    for path in present {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if path.parent() == Some(dir) && DERIVED.contains(&name) {
            continue;
        }
        let canonical = path.canonicalize()?;
        if !covered.contains(&canonical) {
            unlisted.push(
                path.strip_prefix(dir)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
    Ok(Verification {
        schema: "jope.arena-verify/v1".to_owned(),
        results_dir: dir.display().to_string(),
        manifest_schema,
        files,
        unlisted,
        ok,
        missing,
        absent,
        mismatched,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::sha256_hex;

    fn write(root: &Path, rel: &str, bytes: &[u8]) {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn manifest_hashes_are_checked_and_extras_reported() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write(root, "results/tr/scores.csv", b"a,b\n");
        write(root, "results/tr/receipts/x.json", b"{}");
        write(root, "registry/datasets.json", b"[]");
        let manifest = serde_json::json!({
            "schema": "sotarena.golden45-evidence-manifest.v1",
            "files": [
                {"path": "results/tr/receipts/x.json", "sha256": sha256_hex(b"{}"), "bytes": 2},
                {"path": "results/tr/gone.json", "sha256": sha256_hex(b""), "bytes": 0}
            ],
            "published_artifacts": [
                {"path": "results/tr/scores.csv", "sha256": sha256_hex(b"a,b\n"), "bytes": 4},
                {"path": "registry/datasets.json", "sha256": sha256_hex(b"[]"), "bytes": 2},
                {"path": "data/tr/restricted.tar", "sha256": sha256_hex(b"?"), "bytes": 1}
            ]
        });
        write(
            root,
            "results/tr/evidence-manifest.json",
            manifest.to_string().as_bytes(),
        );
        write(root, "results/tr/leaderboard.csv", b"derived");
        write(root, "results/tr/stray.txt", b"stray");
        let report = verify_results(&root.join("results/tr")).unwrap();
        assert_eq!(report.ok, 3);
        assert_eq!(report.missing, 1);
        assert_eq!(report.absent, 1);
        assert_eq!(report.mismatched, 0);
        assert_eq!(report.unlisted, vec!["stray.txt".to_owned()]);
        assert!(!report.passed());
        assert!(!report.passed_strict());

        write(root, "results/tr/gone.json", b"");
        let report = verify_results(&root.join("results/tr")).unwrap();
        assert!(report.passed(), "{report:?}");
        assert!(!report.passed_strict());
        std::fs::remove_file(root.join("results/tr/stray.txt")).unwrap();
        let report = verify_results(&root.join("results/tr")).unwrap();
        assert!(report.passed_strict(), "{report:?}");

        write(root, "results/tr/scores.csv", b"a,c\n");
        let report = verify_results(&root.join("results/tr")).unwrap();
        assert_eq!(report.mismatched, 1);
        assert_eq!(
            report
                .files
                .iter()
                .find(|f| f.path.ends_with("scores.csv"))
                .unwrap()
                .status,
            "sha256-mismatch"
        );
    }

    #[test]
    fn missing_manifest_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(verify_results(dir.path())
            .unwrap_err()
            .to_string()
            .contains("no evidence-manifest.json"));
    }
}

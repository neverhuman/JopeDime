//! Read-only, bounded observations. A live observer never grants training or promotion authority.
use anyhow::{bail, Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
    time::UNIX_EPOCH,
};

/// Schema identifier emitted by the read-only observatory snapshot.
pub const SCHEMA: &str = "jope.observatory/v1";
const MAX_JSON: u64 = 16 * 1024 * 1024;
/// Maximum number of campaign directories included in one observation.
pub const MAX_RUNS: usize = 32;

/// Operator-supplied paths and network binding for the observatory server.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Repository root used for catalogs and training-wave records.
    pub repo_root: PathBuf,
    /// Campaign roots to inspect, in configured order.
    #[serde(default)]
    pub campaign_roots: Vec<PathBuf>,
    /// Hash-pinned GP catalog inputs.
    #[serde(default)]
    pub gp_catalogs: Vec<GpInput>,
    /// Optional current storage observation file.
    pub storage_snapshot: Option<PathBuf>,
    /// Optional code identity observation file.
    pub code_identity: Option<PathBuf>,
    /// Optional current dataset observation file.
    pub dataset_snapshot: Option<PathBuf>,
    /// Optional fleet snapshot files.
    #[serde(default)]
    pub fleet_snapshots: Vec<PathBuf>,
    /// Static dashboard asset directory.
    pub static_dir: PathBuf,
    /// Socket address for the loopback dashboard; defaults to `127.0.0.1:18894`.
    #[serde(default = "listen_default")]
    pub listen: String,
}
fn listen_default() -> String {
    "127.0.0.1:18894".into()
}
/// One operator-selected GP catalog with an expected content digest and public labels.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpInput {
    /// Path to the catalog JSON file.
    pub path: PathBuf,
    /// Expected lowercase SHA-256 digest of the file bytes.
    pub sha256: String,
    /// Label shown with observed catalog rows.
    pub label: String,
    /// Declared scope of this catalog input.
    pub scope: String,
}

/// Bounded observation of one evidence source; missing or invalid values remain unknown.
#[derive(Serialize, Clone)]
pub struct Observation {
    /// `observed`, `unavailable`, or a more specific qualification state.
    pub status: String,
    /// Display path or source identifier.
    pub source: String,
    /// Source modification time in Unix seconds, when available.
    pub observed_at: Option<f64>,
    /// SHA-256 of the observed bytes, when the source was accepted.
    pub sha256: Option<String>,
    /// Non-sensitive refusal reason, when the source could not be read.
    pub error: Option<String>,
    /// Parsed source value, or JSON null when unavailable.
    pub value: Value,
}

/// Current UTC wall-clock time as seconds since the Unix epoch.
pub fn now() -> f64 {
    Utc::now().timestamp_millis() as f64 / 1000.
}
/// Return the lowercase SHA-256 digest of `bytes`.
pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Reject linked components, non-files and oversized input, including growth after stat.
/// Config paths are operator-controlled; no request may supply a filesystem path.
pub fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut cursor = PathBuf::new();
    for c in path.components() {
        if matches!(c, Component::ParentDir) {
            bail!("parent path refused");
        }
        cursor.push(c);
        let meta = fs::symlink_metadata(&cursor).context("source unavailable")?;
        if meta.file_type().is_symlink() {
            bail!("linked source refused");
        }
    }
    let meta = fs::metadata(path)?;
    if !meta.is_file() || meta.len() > limit {
        bail!("source exceeds bounds or is not a file");
    }
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        bail!("source grew beyond bounds");
    }
    Ok(bytes)
}
/// Read and parse one bounded JSON evidence file without converting failures to zero values.
pub fn observe(path: &Path) -> Observation {
    let result = read_bounded(path, MAX_JSON).and_then(|b| {
        let value = serde_json::from_slice(&b).context("invalid JSON")?;
        Ok((value, digest(&b)))
    });
    let at = fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|t| t.as_secs_f64());
    match result {
        Ok((value, sha)) => Observation {
            status: "observed".into(),
            source: path.display().to_string(),
            observed_at: at,
            sha256: Some(sha),
            error: None,
            value,
        },
        Err(_) => Observation {
            status: "unavailable".into(),
            source: path.display().to_string(),
            observed_at: None,
            sha256: None,
            error: Some(
                "Missing, linked, malformed or oversized evidence; value is unknown".into(),
            ),
            value: Value::Null,
        },
    }
}
fn selected(v: &Value, keys: &[&str]) -> Value {
    Value::Object(
        keys.iter()
            .filter_map(|k| v.get(*k).map(|v| ((*k).to_owned(), v.clone())))
            .collect(),
    )
}
fn safe_id(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 180
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
}
fn finite(v: &Value) -> Option<f64> {
    v.as_f64().filter(|n| n.is_finite())
}

/// Extract up to 10,000 finite fixed-panel, non-dropout train/validation points.
pub fn curve(progress: &Value) -> Vec<Value> {
    let mut rows = vec![];
    for r in progress["loss_history"]
        .as_array()
        .into_iter()
        .flatten()
        .take(10_000)
    {
        // Keep panel identity: never silently join curves from different evaluations.
        if r["fixed_panel"] != true || r["dropout"] != false {
            continue;
        }
        let (Some(step), Some(train), Some(validation)) = (
            r["batch"].as_u64(),
            finite(&r["train_loss"]),
            finite(&r["validation_loss"]),
        ) else {
            continue;
        };
        rows.push(json!({"step":step,"train":train,"validation":validation,"at":r["at"],"objective":r["objective"],"panel_sha256":digest(serde_json::to_string(&r["panel_ids"]).unwrap_or_default().as_bytes()),"terms": {"training":r["metrics"]["training"]["terms"],"validation":r["metrics"]["validation"]["terms"]}}));
    }
    rows
}
fn directory_names(root: &Path, max: usize) -> Vec<String> {
    let Ok(entries) = fs::read_dir(root) else {
        return vec![];
    };
    entries
        .take(max)
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir() && !t.is_symlink()))
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|s| safe_id(s))
        .collect()
}

#[path = "observatory/campaign.rs"]
mod campaign;
pub use campaign::campaign;
#[path = "observatory/catalog.rs"]
mod catalog;
pub use catalog::{gp_catalog, storage, waves};
#[path = "observatory/snapshot.rs"]
mod snapshot;
pub use snapshot::snapshot;

#[cfg(test)]
mod tests {
    fn fixture_dir() -> tempfile::TempDir {
        let exe = std::env::current_exe().unwrap();
        let root = exe
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("observatory-test-fixtures");
        std::fs::create_dir_all(&root).unwrap();
        tempfile::tempdir_in(root).unwrap()
    }
    use super::*;
    #[test]
    fn invalid_loss_and_unfixed_panels_are_excluded() {
        let p = json!({"loss_history":[{"batch":0,"fixed_panel":true,"dropout":false,"train_loss":3.0,"validation_loss":4.0,"panel_ids":{"validation":["a"]}},{"batch":1,"fixed_panel":false,"dropout":false,"train_loss":1.,"validation_loss":1.},{"batch":2,"fixed_panel":true,"dropout":true,"train_loss":1.,"validation_loss":1.}]});
        assert_eq!(curve(&p).len(), 1);
        assert_eq!(curve(&p)[0]["validation"], 4.0);
    }
    #[test]
    fn config_mismatch_cannot_report_optimizer_progress() {
        let t = fixture_dir();
        fs::create_dir(t.path().join("training")).unwrap();
        fs::write(t.path().join("config.json"), b"{}").unwrap();
        fs::write(t.path().join("training/progress.json"),br#"{"schema":"jope.flywheel-neural-progress/v1","config_sha256":"wrong","optimizer_steps":100,"state":"running"}"#).unwrap();
        fs::create_dir_all(t.path().join("training/checkpoints/000000")).unwrap();
        fs::write(
            t.path().join("training/checkpoints/000000/manifest.json"),
            b"{}",
        )
        .unwrap();
        let v = campaign(t.path());
        assert_eq!(v["checkpoints"][0]["identity_matches"], false);
        assert_eq!(v["progress"]["status"], "unqualified");
        assert!(v["progress"]["value"].is_null());
    }
    #[test]
    fn missing_source_is_unknown_and_size_limits_are_enforced() {
        let t = fixture_dir();
        assert!(observe(&t.path().join("missing")).value.is_null());
        let p = t.path().join("too-big");
        fs::write(&p, b"123456").unwrap();
        assert!(read_bounded(&p, 5).is_err());
    }
    #[cfg(unix)]
    #[test]
    fn symlink_and_parent_directory_reads_are_refused() {
        let t = fixture_dir();
        let p = t.path().join("data");
        fs::write(&p, b"{}").unwrap();
        std::os::unix::fs::symlink(&p, t.path().join("link")).unwrap();
        assert!(read_bounded(&t.path().join("link"), 50).is_err());
        assert!(read_bounded(&t.path().join("a/../data"), 50).is_err());
    }
    #[test]
    fn gp_deduplicates_ids_and_requires_source_hash() {
        let t = fixture_dir();
        let p = t.path().join("gp.json");
        let id = "a".repeat(64);
        let bytes=serde_json::to_vec(&json!({"programs":[{"program_sha256":id,"dsl":"(pipe X (ridge_gcv))"},{"program_sha256":id,"dsl":"(pipe X (ridge_gcv))"}]})).unwrap();
        fs::write(&p, &bytes).unwrap();
        let mut input = GpInput {
            path: p,
            sha256: digest(&bytes),
            label: "research".into(),
            scope: "test snapshot".into(),
        };
        let v = gp_catalog(&[input.clone()]);
        assert_eq!(v["unique_programs"], 1);
        assert_eq!(v["programs"][0]["context_occurrences"], 2);
        assert_eq!(v["families"]["ridge_gcv"], 1);
        input.sha256 = "wrong".into();
        let v = gp_catalog(&[input]);
        assert!(v["unique_programs"].is_null());
        assert_eq!(v["status"], "unavailable");
    }
}

/// Complete GP source emission, validation and durable delivery.
pub mod gp_source;

/// Read-only retained search commentary transports.
pub mod narration;

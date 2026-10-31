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

/// Build a bounded, identity-checked observation of one campaign directory.
pub fn campaign(root: &Path) -> Value {
    let cfg = observe(&root.join("config.json"));
    let mut evidence = observe(&root.join("training/progress.json"));
    let loaded = observe(&root.join("training/loaded-receipt.json"));
    let p = evidence.value.clone();
    let identity_matches = cfg
        .sha256
        .as_deref()
        .is_some_and(|s| p["config_sha256"].as_str() == Some(s));
    let schema_matches = p["schema"] == "jope.flywheel-neural-progress/v1";
    let recognized = schema_matches && identity_matches;
    let stale = now() - evidence.observed_at.unwrap_or(0.) > 60.;
    let terminal = p["state"] == "completed" || p["state"] == "failed";
    if !recognized {
        evidence.status = "unqualified".into();
    } else if stale && !terminal {
        evidence.status = "stale".into();
    }
    let mut checkpoint_names = directory_names(&root.join("training/checkpoints"), 20_000);
    checkpoint_names.retain(|s| s.len() == 6 && s.bytes().all(|b| b.is_ascii_digit()));
    checkpoint_names.sort();
    let count = checkpoint_names.len();
    let checkpoints: Vec<_> = checkpoint_names.iter().rev().take(24).map(|name| {
        let manifest = observe(&root.join("training/checkpoints").join(name).join("manifest.json"));
        json!({"id":name,"evidence": {"source":manifest.source,"sha256":manifest.sha256,"status":manifest.status},"identity_matches":cfg.sha256.as_deref().is_some_and(|sha|manifest.value["config_sha256"].as_str()==Some(sha)),"manifest":selected(&manifest.value,&["schema","step","state_bytes","state_sha256","parameter_sha256","parameters","model_config","optimizer","learning_rate","weight_decay","config_sha256"]),"verification":"manifest observed; weights not rehashed by the dashboard"})
    }).collect();
    let mut cfg_public = selected(
        &cfg.value,
        &[
            "host",
            "schema",
            "unit",
            "duration_seconds",
            "closure_reserve_seconds",
            "expected_contexts",
            "expected_unique_teacher_programs",
            "research_optimizer_admission",
            "runtime_versions",
            "interpretation",
        ],
    );
    cfg_public["neural"] = selected(
        &cfg.value["neural"],
        &[
            "run_id",
            "max_updates",
            "max_epochs",
            "batch_size",
            "learning_rate",
            "seed",
            "device",
            "initialization",
            "layers",
            "heads",
            "d_model",
            "d_ff",
            "dropout",
            "max_seq",
            "input_dimension",
            "objective",
        ],
    );
    cfg_public["snapshot"] = selected(&cfg.value["snapshot"], &["sha256", "bytes"]);
    // These counters remain null on invalid/mismatched evidence, never fabricated zero.
    evidence.value = if recognized {
        selected(
            &p,
            &[
                "run_id",
                "state",
                "at",
                "optimizer_steps",
                "completed_epochs",
                "total_train_presentations",
                "unique_train_data_runs_seen",
                "unique_validation_data_runs_scored",
                "model_generated_programs",
                "novel_programs_executed",
                "published_results",
                "comparison_games",
                "elo",
                "winrate_vs_random",
                "winrate_vs_winner",
                "performance",
                "outbox",
                "proposal_catalog",
                "latest_full_validation",
                "scope",
            ],
        )
    } else {
        Value::Null
    };
    if evidence.value.is_object() {
        evidence.value["latest_full_validation"] = if p["latest_full_validation"].is_object() {
            selected(
                &p["latest_full_validation"],
                &[
                    "schema",
                    "epoch",
                    "context_count",
                    "loss",
                    "dropout",
                    "new_program_rewards_used",
                    "denominators",
                    "numerators",
                    "support",
                    "terms",
                ],
            )
        } else {
            Value::Null
        };
    }
    let source_map = observe(&root.join("source-map.json"));
    let mut terminal_receipt = observe(&root.join("status.json"));
    terminal_receipt.value = selected(
        &terminal_receipt.value,
        &["schema", "state", "at", "config_sha256", "optimizer_steps"],
    );
    json!({"id":root.file_name().unwrap_or_default().to_string_lossy(),"config":cfg_public,"config_sha256":cfg.sha256,"progress":evidence,"identity_matches":identity_matches,"curve":if recognized {curve(&p)} else {vec![]},"counts":selected(&loaded.value["counts"], &["partitions","loaded_kumo_contexts","current_eligible_views","pending_contexts","refused_rows","native_detail_records_declared"]),"membership_sha256":loaded.value["membership_sha256"],"checkpoints":checkpoints,"checkpoint_count_observed":count,"checkpoint_list_truncated":count>24,"source_identity":selected(&source_map.value,&["source_commit","source_main_sha","commit","verified_files"]),"terminal_receipt":terminal_receipt})
}

/// Observe configured storage evidence, falling back to the dated catalog when needed.
pub fn storage(c: &Config) -> Value {
    let catalog = observe(&c.repo_root.join("catalogs/s3.json"));
    if let Some(path) = &c.storage_snapshot {
        let live = observe(path);
        if live.value["schema"] == "jope.storage-observation/v1"
            && live.value["buckets"].as_array().is_some_and(|rows| {
                rows.iter().all(|b| {
                    b["bucket"].is_string() && b["region"].is_string() && b["status"].is_string()
                })
            })
            && (live.value["errors"].is_null()
                || live.value["errors"]
                    .as_array()
                    .is_some_and(|rows| rows.iter().all(Value::is_string)))
        {
            return json!({"evidence":live,"catalog_fallback":false});
        }
    }
    // Historical catalogs remain explicitly dated; object metadata is not a dataset census.
    let buckets: Vec<_> = catalog.value["buckets"].as_array().into_iter().flatten().map(|b| json!({"bucket":b["bucket"],"region":b["region"],"role":b["role"],"replica_of":b["replica_of"],"bytes":b["bytes"],"objects":b["objects"],"observed_at":b["inventory"]["captured_utc"],"status":"historical","scope":"current object versions in dated metadata inventory; replicas counted physically"})).collect();
    let bytes: u64 = buckets.iter().filter_map(|b| b["bytes"].as_u64()).sum();
    let objects: u64 = buckets.iter().filter_map(|b| b["objects"].as_u64()).sum();
    json!({"evidence":{"status":if buckets.is_empty(){"unavailable"}else{"historical"},"source":catalog.source,"sha256":catalog.sha256,"observed_at":catalog.value["inventoried"],"value":{"schema":"jope.storage-observation/v1","buckets":buckets,"observed_bytes":if buckets.iter().any(|b|b["bytes"].is_u64()) {json!(bytes)}else{Value::Null},"observed_objects":if buckets.iter().any(|b|b["objects"].is_u64()){json!(objects)}else{Value::Null},"complete":false,"scope":"last catalog snapshot; not a current fleet total","errors":[]}},"catalog_fallback":true})
}

/// Summarize up to eight hash-pinned GP catalog inputs without inferring HWM admission.
pub fn gp_catalog(inputs: &[GpInput]) -> Value {
    let mut programs: BTreeMap<String, Value> = BTreeMap::new();
    let mut families: BTreeMap<String, usize> = BTreeMap::new();
    let mut sources = vec![];
    let mut all_valid = !inputs.is_empty();
    for input in inputs.iter().take(8) {
        let obs = observe(&input.path);
        let hash_matches = obs.sha256.as_deref() == Some(&input.sha256);
        let rows = obs.value["programs"].as_array();
        all_valid &= hash_matches && rows.is_some_and(|r| r.len() <= 30_000);
        sources.push(json!({"label":input.label,"scope":input.scope,"source":obs.source,"sha256":obs.sha256,"expected_sha256":input.sha256,"hash_matches":hash_matches,"declared_rows":rows.map(Vec::len),"scan_complete":hash_matches&&rows.is_some_and(|r|r.len()<=30_000),"rows_scanned":rows.map(|r|r.len().min(30_000)),"observed_at":obs.observed_at}));
        if !hash_matches {
            continue;
        }
        for row in rows.into_iter().flatten().take(30_000) {
            let (Some(id), Some(dsl)) = (row["program_sha256"].as_str(), row["dsl"].as_str())
            else {
                continue;
            };
            if id.len() != 64 || !id.bytes().all(|b| b.is_ascii_hexdigit()) || dsl.len() > 32_768 {
                continue;
            }
            let operators: BTreeSet<String> = dsl
                .split('(')
                .skip(1)
                .filter_map(|s| s.split_whitespace().next())
                .filter(|s| !s.is_empty() && *s != "pipe")
                .map(|s| s.trim_end_matches(')').to_owned())
                .collect();
            let family = operators
                .iter()
                .find(|s| {
                    s.starts_with("ridge")
                        || s.starts_with("knn")
                        || s.starts_with("gpr")
                        || s.starts_with("logistic")
                })
                .cloned()
                .unwrap_or_else(|| "other".into());
            if let Some(existing) = programs.get_mut(id) {
                existing["context_occurrences"] =
                    json!(existing["context_occurrences"].as_u64().unwrap_or(1) + 1);
                continue;
            }
            *families.entry(family.clone()).or_default() += 1;
            programs.insert(id.into(),json!({"id":id,"dsl":dsl,"operators":operators,"family":family,"context_occurrences":1,"source":input.label,"scope":input.scope,"score":null,"score_note":"No evaluator-bound score in this library record"}));
        }
    }
    json!({"status":if all_valid{"observed"}else if programs.is_empty(){"unavailable"}else{"partial"},"unique_programs":if programs.is_empty()&&!all_valid{Value::Null}else{json!(programs.len())},"scope":"unique declared program identities in the configured, hash-pinned library snapshots; not a census of every fleet library; no HWM admission inferred","global_complete":false,"families":families,"sources":sources,"programs":programs.into_values().collect::<Vec<_>>()})
}

/// Read compact, recorded assertions for training waves under `repo`.
pub fn waves(repo: &Path) -> Value {
    let artifacts = observe(&repo.join("catalogs/training-artifacts.json"));
    let root = repo.join("docs/research/waves");
    let mut names = directory_names(&root, 256);
    names.sort();
    let mut result = vec![];
    for name in names {
        let dir = root.join(&name);
        let card = observe(&dir.join("card.json"));
        if card.value.is_null() {
            continue;
        }
        let manifest = observe(&dir.join("wave.manifest.json"));
        let metrics = observe(&dir.join("metrics.json"));
        let assertions: Vec<_> = metrics.value["assertions"]
            .as_array()
            .into_iter()
            .flatten()
            .take(1000)
            .map(|v| {
                selected(
                    v,
                    &[
                        "id",
                        "value",
                        "unit",
                        "numerator",
                        "denominator",
                        "status",
                        "evidence_sha256",
                        "source_field",
                    ],
                )
            })
            .collect();
        let archive_roots: Vec<_> = manifest.value["evidence_roots"]
            .as_array()
            .into_iter()
            .flatten()
            .take(64)
            .map(|r| {
                let mut value = selected(
                    r,
                    &[
                        "bytes",
                        "codec",
                        "member_sha256",
                        "physical_version",
                        "rows",
                        "sha256",
                        "version",
                    ],
                );
                let matching = artifacts.value["artifacts"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .find(|a| a["sha256"].is_string() && a["sha256"] == r["sha256"]);
                value["s3_uri"] = matching
                    .and_then(|a| a.get("s3_uri"))
                    .cloned()
                    .unwrap_or(Value::Null);
                value
            })
            .collect();
        result.push(json!({"id":name,"card":selected(&card.value,&["wave_id","version","stage","capture_status","archive_status","disposition","interpretation"]),"assertions":assertions,"manifest_sha256":manifest.sha256,"metrics_sha256":metrics.sha256,"source_commit":manifest.value["source_commit"],"training_identity_sha256":manifest.value["training_identity_sha256"],"archive_manifest_sha256":manifest.value["archive_manifest_sha256"],"archive_roots":archive_roots,"publication":"files observed in configured repository snapshot; Git publication requires commit receipt","archive":"producer claim; dashboard does not independently restore bulk payloads"}));
    }
    json!({"waves":result,"scope":"compact recorded assertions, not raw result rows or verified scientific wins"})
}

/// Assemble the configured read-only observatory snapshot.
pub fn snapshot(c: &Config) -> Value {
    let s3 = storage(c);
    let catalog = observe(&c.repo_root.join("catalogs/s3.json"));
    let mut datasets: Vec<_>=catalog.value["buckets"].as_array().into_iter().flatten().filter(|b|b["catalog_v1"].is_object()).map(|b|json!({"name":b["bucket"],"count":b["catalog_v1"]["datasets"],"license_status":b["catalog_v1"]["license_status"],"origins":b["catalog_v1"]["origins"],"sha256":b["catalog_v1"]["sha256"],"observed_at":catalog.value["inventoried"],"status":"historical","scope":"dataset IDs in versioned catalog; source data is not read by dashboard"})).collect();
    if let Some(path) = &c.dataset_snapshot {
        let current = observe(path);
        if current.value["schema"] == "jope.dataset-observation/v1" {
            if let Some(rows) = current.value["datasets"].as_array() {
                datasets = rows
                    .iter()
                    .take(32)
                    .map(|r| {
                        selected(
                            r,
                            &[
                                "name",
                                "count",
                                "license_status",
                                "origins",
                                "sha256",
                                "version_id",
                                "catalog_modified_at",
                                "observed_at",
                                "status",
                                "scope",
                            ],
                        )
                    })
                    .collect();
            }
        }
    }
    let fleet: Vec<_> = c
        .fleet_snapshots
        .iter()
        .take(4)
        .map(|p| {
            let mut o = observe(p);
            let valid = o.value["schema"] == "jope.fleet-node-observation/v1";
            o.value = if valid {
                selected(
                    &o.value,
                    &[
                        "schema",
                        "host",
                        "observed_at",
                        "status",
                        "cpu_count",
                        "load",
                        "memory_total_bytes",
                        "memory_available_bytes",
                        "disk_available_bytes",
                        "gpus",
                        "workload_scope",
                    ],
                )
            } else {
                Value::Null
            };
            if !valid {
                o.status = "unqualified".into();
            }
            json!(o)
        })
        .collect();
    let mut runs: Vec<_> = c
        .campaign_roots
        .iter()
        .take(MAX_RUNS)
        .map(|r| campaign(r))
        .collect();
    let mut identity = c.code_identity.as_ref().map(|p| observe(p));
    if let Some(o) = &mut identity {
        if o.value["schema"] == "jope.code-identity-observation/v1" {
            for r in &mut runs {
                if let Some(s) = o.value["training_runs"].get(r["id"].as_str().unwrap_or_default())
                {
                    r["source_identity"] = selected(
                        s,
                        &["source_commit", "verified_files", "receipt_sha256", "scope"],
                    );
                }
            }
            o.value = selected(
                &o.value,
                &[
                    "schema",
                    "observed_at",
                    "dashboard_commit",
                    "dashboard_dirty",
                    "training_runs",
                    "scope",
                ],
            );
        } else {
            o.status = "unqualified".into();
            o.value = Value::Null;
        }
    }
    json!({"schema":SCHEMA,"generated_at":now(),"service":{"version":env!("CARGO_PKG_VERSION"),"mode":"read-only","refresh_seconds":5,"production_authority":false},"code_identity":identity,"storage":s3,"datasets":datasets,"runs":runs,"gp":gp_catalog(&c.gp_catalogs),"research":waves(&c.repo_root),"fleet":fleet,"limits":{"max_runs":MAX_RUNS,"max_json_bytes":MAX_JSON,"bulk_payloads_read":false,"credentials_read":false,"note":"Read-only observations never admit work, promote a model, or establish verified self-play gains."}})
}

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

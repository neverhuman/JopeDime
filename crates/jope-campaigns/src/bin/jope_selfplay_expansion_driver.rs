//! jope_selfplay_expansion_driver — recurring, guardrailed dataset-unlock cycle.
//!
//! Operator mandate: unlock the majority of the locked registry into the live
//! self-play catalog, continuously, without ever slowing deep learning. Each
//! timer cycle (1 h) this driver: refreshes the dataset-state index, keeps a
//! preemptible binary encode chunk queued on xbabe3, and — when the guardrails
//! and the coverage valve allow — installs/curates/materializes/activates one
//! ≤2,048-dataset tranche and folds it into the live catalog via merge-live.
//! Pacing comes from the valve (previous tranche must reach ≥0.3 corpus
//! coverage), not the clock. All mutations go through the audited
//! jope_expanded_catalog tool; Python is invoked only for torch encoders.

use anyhow::{anyhow, bail, Context, Result};
use jope_dime_regressor::paths;
use jope_dime_regressor::recovery_contracts::V1EncodingQuarantineV1;
use jope_dime_regressor::selfplay_expansion::{corpus_dir_name, TASKS};
use jope_dime_regressor::selfplay_runtime::{atomic_json, sha256_bytes, sha256_file};
use rusqlite::Connection;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

jope_paths::once_str!(
    fn root_str => paths::flywheel_root()
);
jope_paths::once_str!(
    fn registry_str => paths::quant_root()
);
jope_paths::once_str!(
    fn encodings_str => paths::flywheel_encodings_root()
);
jope_paths::once_str!(
    fn foundation_str => paths::flywheel_encodings_root().join("foundation-v1/datasets")
);
jope_paths::once_str!(
    fn fetch_binary_str => paths::flywheel_encodings_root().join("_fetch/binary-expansion")
);
jope_paths::once_str!(
    fn scq_str => paths::scq_binary()
);
jope_paths::once_str!(
    /// Root of the foundation-expert tree, which the encoder imports as
    /// `python.elo_finetune.model_io`.
    ///
    /// This is on the job PYTHONPATH because `fusion.experts.TabPFNv3Reg::__init__`
    /// hard-codes `sys.path.insert(0, "<home>/fable_god")` -- a root that was
    /// MOVED here, so the import raises ModuleNotFoundError. `encode_external`
    /// catches that and logs "cuda expert load failed; falling back to cpu", and
    /// the CPU fallback calls the same loader and fails identically: the encoder is
    /// dead on both devices while the message blames the device. Bundled
    /// `runner.sh` does `export PYTHONPATH="${JOPE_RUNTIME_PYTHONPATH:?...}"`,
    /// REPLACING the variable rather than extending it, so the root has to be
    /// carried in the spec or no encode job on any node can resolve the expert.
    fn foundation_expert_root_str => paths::foundation_expert_root()
);
const SIDECARS: [&str; 3] = ["hyperion3train128dim.vs", "train3.vs", "train_target.vs"];
const TRANCHE_CAP: usize = 512;
const ENCODE_CHUNK: usize = 256;
const BINARY_INSTALL_MIN: usize = 1;
const COVERAGE_VALVE: f64 = 0.3;
/// How many times a binary dataset may be listed on an encode chunk before it
/// is parked. Encode chunks strand their listings whenever the job dies without
/// producing sidecars (job purged from scq, node preemption, the hyperion CPU
/// fallback); without a budget the retry path would re-list datasets that can
/// never encode (e.g. absent train.csv on the encode node) forever.
///
/// Deliberately generous. Most stranding is TRANSIENT — hyperion_vs falls back
/// to CPU only when GPU memory is near-exhausted (verified: it still picks
/// cuda:0 with an 11 GB co-tenant), so a chunk that fails now succeeds later.
/// Parking a dataset too eagerly re-creates the silent binary-lane death this
/// budget exists to prevent, while re-listing a genuinely broken dataset only
/// wastes one slot in a 2,000-dataset chunk. Bias hard toward retrying.
const MAX_ENCODE_ATTEMPTS: i64 = 8;

fn reject_v1_quarantine_window(encodings_root: &Path) -> Result<()> {
    let quarantine_root = encodings_root.join("incident-quarantine");
    if !quarantine_root.exists() {
        return Ok(());
    }
    let index_path = quarantine_root.join("index.v1.json");
    let index: V1EncodingQuarantineV1 = serde_json::from_reader(std::io::BufReader::new(
        std::fs::File::open(&index_path).with_context(|| {
            format!(
                "V1 incident quarantine exists without readable frozen index {}",
                index_path.display()
            )
        })?,
    ))?;
    index.validate()?;
    bail!("legacy expansion driver is disabled throughout the Encoder V2 quarantine window")
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn sh(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    if !output.status.success() {
        bail!(
            "{program} {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
                .lines()
                .last()
                .unwrap_or("")
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn temporarily_lacks_launch_capacity(message: &str) -> bool {
    message.contains("dependency-unsatisfied")
        && (message.contains("no eligible launch-ready node currently")
            || message.contains("eligible_node launch-ready-capacity")
            // The capability smoke pins encoding to xbabe3. If only another
            // node is launch-ready, its expected test(1) is a placement
            // backoff rather than a missing encoder dependency.
            || (message.contains("smoke xbabe3 prime encode environment")
                && message.contains("exit 1")))
}

/// A capacity-only dependency-check failure must not strand breadth encoding.
/// Retry submission in case capacity opened between check and submit; permanent
/// dependency failures still fail closed.
#[derive(Clone, Debug, Serialize)]
struct EncodeChunkSubmission {
    job_id: String,
    disposition: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    backoff_reason: Option<String>,
}

fn pending_breadth_admission(root: &Path) -> Result<Option<EncodeChunkSubmission>> {
    let path = root.join("ops/gpu-lane-admission/requests/breadth_encoder.json");
    if !path.is_file() {
        return Ok(None);
    }
    let value = read_json(&path)?;
    let job_id = value
        .get("immutable_job_identity")
        .and_then(Value::as_str)
        .context("breadth admission request lacks immutable job identity")?;
    Ok(Some(EncodeChunkSubmission {
        job_id: job_id.to_owned(),
        disposition: "gpu_admission_pending",
        backoff_reason: value
            .get("last_capacity_backoff")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| Some("waiting for xbabe3 to expose four core credits".to_owned())),
    }))
}

fn register_gpu_admission(spec: &Path, job_id: &str) -> Result<()> {
    let controller = bundled_binary("jope_gpu_lane_admission")?;
    let output = Command::new(controller)
        .args([
            "--register-role",
            "breadth-encoder",
            "--spec",
            spec.to_string_lossy().as_ref(),
            "--job-id",
            job_id,
        ])
        .output()
        .context("register breadth encoder GPU admission request")?;
    if !output.status.success() {
        bail!(
            "GPU admission request failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

fn scq_check_submit(spec: &Path, job_id: &str) -> Result<EncodeChunkSubmission> {
    let spec_text = spec.to_string_lossy();
    let check = Command::new(scq_str())
        .args(["check", spec_text.as_ref()])
        .output()
        .with_context(|| {
            format!(
                "run {scq_str} check {}",
                spec.display(),
                scq_str = scq_str()
            )
        })?;
    let check_message = String::from_utf8_lossy(if check.stderr.is_empty() {
        &check.stdout
    } else {
        &check.stderr
    });
    if !check.status.success() && temporarily_lacks_launch_capacity(&check_message) {
        let reason = format!("capacity backoff: {}", check_message.trim());
        register_gpu_admission(spec, job_id)?;
        return Ok(EncodeChunkSubmission {
            job_id: job_id.to_owned(),
            disposition: "gpu_admission_requested",
            backoff_reason: Some(reason),
        });
    }
    if !check.status.success() {
        bail!(
            "SCQ check rejected encoding chunk: {}",
            check_message.trim()
        );
    }

    let submit = Command::new(scq_str())
        .args(["submit", spec_text.as_ref()])
        .output()
        .with_context(|| {
            format!(
                "run {scq_str} submit {}",
                spec.display(),
                scq_str = scq_str()
            )
        })?;
    let submit_message = String::from_utf8_lossy(if submit.stderr.is_empty() {
        &submit.stdout
    } else {
        &submit.stderr
    });
    if submit.status.success()
        || submit_message
            .to_ascii_lowercase()
            .contains("already exists")
    {
        return Ok(EncodeChunkSubmission {
            job_id: job_id.to_owned(),
            disposition: "submitted",
            backoff_reason: None,
        });
    }
    if temporarily_lacks_launch_capacity(&submit_message) {
        let reason = format!("capacity backoff: {}", submit_message.trim());
        register_gpu_admission(spec, job_id)?;
        return Ok(EncodeChunkSubmission {
            job_id: job_id.to_owned(),
            disposition: "gpu_admission_requested",
            backoff_reason: Some(reason),
        });
    }
    bail!(
        "SCQ submit rejected encoding chunk: {}",
        submit_message.trim()
    )
}

fn runtime_bundle_root() -> Result<PathBuf> {
    let executable = std::fs::canonicalize(std::env::current_exe()?)?;
    let bin = executable
        .parent()
        .context("runtime executable has no parent")?;
    let runtime = bin.parent().context("runtime bin has no parent")?;
    let bundle = runtime
        .parent()
        .context("runtime directory has no parent")?;
    let manifest = read_json(&bundle.join("bundle.json"))?;
    if manifest.get("schema").and_then(Value::as_str) != Some("RuntimeBundle.v3") {
        bail!(
            "expansion driver is not executing from a verified RuntimeBundle.v3: {}",
            executable.display()
        );
    }
    Ok(bundle.to_path_buf())
}

fn bundled_binary(name: &str) -> Result<PathBuf> {
    let path = runtime_bundle_root()?.join("runtime/bin").join(name);
    if !path.is_file() {
        bail!("runtime bundle lacks required binary {}", path.display());
    }
    Ok(path)
}

fn bundled_encoding_asset(name: &str) -> Result<PathBuf> {
    let path = runtime_bundle_root()?.join("pytorch/encoding").join(name);
    if !path.is_file() {
        bail!("runtime bundle lacks encoding asset {}", path.display());
    }
    Ok(path)
}

fn publish_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.is_file() {
        if std::fs::read(path)? != bytes {
            bail!("immutable content drift: {}", path.display());
        }
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    use std::io::Write;
    file.write_all(bytes)?;
    file.sync_data()?;
    Ok(())
}

fn normalized_runtime_bundle_paths(body: &str) -> String {
    const MARKER: &str = "/runtime-bundles/";
    let mut parts = body.split(MARKER);
    let mut normalized = parts.next().unwrap_or_default().to_owned();
    for part in parts {
        normalized.push_str(MARKER);
        let component_end = part.find('/').unwrap_or(part.len());
        let component = &part[..component_end];
        if component == "current"
            || (component.len() == 64 && component.bytes().all(|byte| byte.is_ascii_hexdigit()))
        {
            normalized.push_str("<bundle>");
        } else {
            normalized.push_str(component);
        }
        normalized.push_str(&part[component_end..]);
    }
    normalized
}

fn publish_equivalent_bundle_spec(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.is_file() {
        let existing = std::fs::read_to_string(path)?;
        let proposed = std::str::from_utf8(bytes).context("rendered SCQ spec is not UTF-8")?;
        if normalized_runtime_bundle_paths(&existing) != normalized_runtime_bundle_paths(proposed) {
            bail!("immutable content drift: {}", path.display());
        }
        return Ok(());
    }
    publish_immutable(path, bytes)
}

fn append_spec_input(
    lines: &mut Vec<String>,
    name: &str,
    source: &Path,
    target: &str,
) -> Result<()> {
    let source = source
        .to_str()
        .context("encoding input path is not UTF-8")?;
    if [name, source, target]
        .iter()
        .any(|value| value.contains(['"', '\\', '\n', '\r']))
    {
        bail!("encoding spec input contains unsafe TOML text");
    }
    lines.extend([
        "[[input]]".to_owned(),
        format!("name = \"{name}\""),
        format!("source = \"{source}\""),
        format!("target = \"{target}\""),
        format!("sha256 = \"{}\"", sha256_file(Path::new(source))?),
        String::new(),
    ]);
    Ok(())
}

fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| path.display().to_string())?,
    )?)
}

fn file_age_secs(path: &Path) -> Option<u64> {
    let modified = path.metadata().ok()?.modified().ok()?;
    SystemTime::now()
        .duration_since(modified)
        .ok()
        .map(|d| d.as_secs())
}

// ---------------------------------------------------------------------------
// Guardrails
// ---------------------------------------------------------------------------

struct Guardrails {
    alarms_path: PathBuf,
    keeper_path: PathBuf,
    waves_dir: PathBuf,
    disk_paths: Vec<PathBuf>,
}

fn newest_wave_age(waves_dir: &Path) -> Option<u64> {
    std::fs::read_dir(waves_dir)
        .ok()?
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("strict-wave-"))
        .filter_map(|e| e.metadata().ok().and_then(|m| m.modified().ok()))
        .max()
        .and_then(|newest| SystemTime::now().duration_since(newest).ok())
        .map(|d| d.as_secs())
}

fn disk_use_pct(path: &Path) -> Option<f64> {
    // statvfs via `df` keeps this dependency-free.
    let out = Command::new("df")
        .arg("--output=pcent")
        .arg(path)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .nth(1)?
        .trim()
        .trim_end_matches('%')
        .parse::<f64>()
        .ok()
}

fn disk_pause_decision(was_paused: bool, maximum_used_pct: Option<f64>) -> bool {
    match maximum_used_pct {
        Some(used) if was_paused => used >= 80.0,
        Some(used) => used >= 85.0,
        None => was_paused,
    }
}

fn disk_hysteresis_block(root: &Path, paths: &[PathBuf]) -> Result<Option<String>> {
    let state_path = root.join("ops/expansion-disk-hysteresis.v1.json");
    let was_paused = state_path
        .is_file()
        .then(|| read_json(&state_path))
        .transpose()?
        .and_then(|value| value.get("paused").and_then(Value::as_bool))
        .unwrap_or(false);
    let measurements = paths
        .iter()
        .filter_map(|path| disk_use_pct(path).map(|used| (path, used)))
        .collect::<Vec<_>>();
    let maximum = measurements
        .iter()
        .map(|(_, used)| *used)
        .max_by(f64::total_cmp);
    let paused = disk_pause_decision(was_paused, maximum);
    atomic_json(
        &state_path,
        &json!({
            "schema": "ExpansionDiskHysteresis.v1",
            "paused": paused,
            "pause_threshold_percent": 85,
            "resume_below_percent": 80,
            "maximum_used_percent": maximum,
            "measurements": measurements
                .iter()
                .map(|(path, used)| json!({"path": path, "used_percent": used}))
                .collect::<Vec<_>>(),
            "updated_at_ms": now_ms(),
        }),
    )?;
    Ok(paused.then(|| match maximum {
        Some(used) => format!("disk expansion pause active at {used:.0}% (resume only below 80%)"),
        None => "disk expansion pause remains active; usage is unavailable".to_owned(),
    }))
}

/// Returns Some(reason) when the cycle must not run tranche work.
fn guardrail_block(g: &Guardrails, coverage: Option<f64>) -> Option<String> {
    if let Ok(alarms) = read_json(&g.alarms_path) {
        for alarm in alarms
            .get("alarms")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
        {
            if alarm.get("severity").and_then(Value::as_str) != Some("critical") {
                continue;
            }
            let name = alarm.get("alarm").and_then(Value::as_str).unwrap_or("?");
            // Tranche work materializes caches and rewrites the catalog on the
            // controller only. A remote worker's disk cannot corrupt or stall
            // that, so blocking on it just starves the catalog (xbabe1 at 90%
            // halted expansion for hours). Local disk is checked below.
            if name.starts_with("disk_high:") && !name.ends_with(":xbabe2") {
                continue;
            }
            // Alarms that mean "no work is running" must NOT halt expansion.
            // Activating datasets is what CREATES work, so blocking on these
            // turns any wave-lane defect into total cluster paralysis: waves
            // stall -> wave_stalled fires -> expansion halts -> no new datasets
            // -> the first-touch coverage lane drains to an empty backlog ->
            // every slot goes idle. That exact loop idled all 1,052 slots on
            // 2026-07-28. Block only on conditions expansion itself would make
            // worse; local disk is still checked below.
            if matches!(
                name,
                "wave_stalled" | "no_active_trainer" | "scq_jobs_held_starving"
            ) || name.starts_with("status_stale:ops/task-learning-controller")
                || name.starts_with("status_stale:ops/trainer-watchdog")
            {
                continue;
            }
            return Some(format!("critical alarm: {name}"));
        }
    }
    match file_age_secs(&g.keeper_path) {
        Some(age) if age > 900 => return Some(format!("keeper status stale {age}s")),
        None => return Some("keeper status missing".into()),
        _ => {}
    }
    // Wave age is deliberately NOT a block, for the same reason as wave_stalled
    // above: a stalled wave lane is precisely when the coverage lane needs new
    // datasets to keep the cluster busy. "No waves found" still blocks, because
    // that indicates an unbuilt/corrupt root rather than a stalled producer.
    if newest_wave_age(&g.waves_dir).is_none() {
        return Some("no waves found".into());
    }
    for path in &g.disk_paths {
        if let Some(pct) = disk_use_pct(path) {
            if pct >= 85.0 {
                return Some(format!("disk {}% at {}", pct, path.display()));
            }
        }
    }
    if let Some(cov) = coverage {
        if cov < COVERAGE_VALVE {
            return Some(format!(
                "coverage valve: previous tranche at {cov:.2} < {COVERAGE_VALVE}"
            ));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tranche bookkeeping
// ---------------------------------------------------------------------------

fn activations(root: &Path) -> Result<Vec<Value>> {
    let path = root.join("catalog/v2/activations.jsonl");
    let mut rows = Vec::new();
    if let Ok(text) = std::fs::read_to_string(path) {
        for line in text.lines() {
            if !line.trim().is_empty() {
                rows.push(serde_json::from_str(line)?);
            }
        }
    }
    Ok(rows)
}

fn next_tranche_id(existing: &[Value], prefix: char) -> String {
    let max = existing
        .iter()
        .filter_map(|row| row.get("tranche_id").and_then(Value::as_str))
        .filter_map(|id| id.strip_prefix(&format!("expanded-{prefix}")))
        .filter_map(|n| n.parse::<u32>().ok())
        .max()
        .unwrap_or(0);
    format!("expanded-{prefix}{:04}", max + 1)
}

/// Newest activation FOR ONE TASK. The valve must pace each task against its
/// own last tranche: keying it off the globally-last activation coupled the two
/// lanes, so a fresh regression tranche (coverage ~0) shut the binary lane down
/// for hours even when the last binary tranche was fully absorbed — the model
/// has two task branches and both must keep widening.
fn latest_tranche_for(existing: &[Value], prefix: char) -> Option<(String, PathBuf)> {
    let want = format!("expanded-{prefix}");
    existing
        .iter()
        .rev()
        .find(|row| {
            row.get("tranche_id")
                .and_then(Value::as_str)
                .is_some_and(|id| id.starts_with(&want))
        })
        .map(|row| {
            (
                row.get("tranche_id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned(),
                PathBuf::from(row.get("catalog").and_then(Value::as_str).unwrap_or("")),
            )
        })
}

// ---------------------------------------------------------------------------
// Dataset-state index
// ---------------------------------------------------------------------------

fn open_state_db(root: &Path) -> Result<Connection> {
    let conn = Connection::open(root.join("ops/dataset-state.sqlite"))?;
    conn.busy_timeout(Duration::from_secs(60))?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS datasets(
           task TEXT NOT NULL, dataset_hash TEXT NOT NULL,
           has_train_test_meta INTEGER NOT NULL DEFAULT 0,
           sidecars_complete INTEGER NOT NULL DEFAULT 0,
           foundation_npy INTEGER NOT NULL DEFAULT 0,
           cache_materialized INTEGER NOT NULL DEFAULT 0,
           tranche_id TEXT, catalog_active INTEGER NOT NULL DEFAULT 0,
           sealed INTEGER NOT NULL DEFAULT 0,
           encode_job_state TEXT,
           obs_count INTEGER NOT NULL DEFAULT 0,
           best_score REAL, last_observed_ms INTEGER,
           updated_at_ms INTEGER NOT NULL,
           PRIMARY KEY(task, dataset_hash)) WITHOUT ROWID;
         CREATE INDEX IF NOT EXISTS ds_stage ON datasets(task, sidecars_complete, catalog_active);
         CREATE INDEX IF NOT EXISTS ds_active_obs ON datasets(task, catalog_active, obs_count);
         CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);",
    )?;
    // Added after the first deployment: existing prod DBs predate the column,
    // and ALTER TABLE ADD COLUMN has no IF NOT EXISTS, so a duplicate-column
    // error here is the expected steady state and must not fail the cycle.
    let _ = conn.execute(
        "ALTER TABLE datasets ADD COLUMN encode_attempts INTEGER NOT NULL DEFAULT 0",
        [],
    );
    Ok(conn)
}

fn refresh_filesystem_state(conn: &Connection, root: &Path) -> Result<(u64, u64)> {
    let ptr = read_json(&root.join("catalog/current.json"))?;
    let cat = read_json(Path::new(
        ptr.get("catalog").and_then(Value::as_str).unwrap_or(""),
    ))?;
    let mut active: HashSet<(String, String)> = HashSet::new();
    let mut binding = Vec::new();
    for e in cat
        .get("entries")
        .and_then(Value::as_array)
        .unwrap_or(&binding)
    {
        active.insert((
            e.get("task")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
            e.get("dataset_hash")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned(),
        ));
    }
    binding.clear();
    let sealed =
        read_json(&root.join("catalog/private/sealed-expanded-v1.json")).unwrap_or(json!({}));
    let mut sealed_ids: HashSet<String> = HashSet::new();
    for e in sealed
        .get("entries")
        .and_then(Value::as_array)
        .unwrap_or(&binding)
    {
        if let Some(id) = e
            .get("dataset_id")
            .or_else(|| e.get("dataset_hash"))
            .and_then(Value::as_str)
        {
            sealed_ids.insert(id.to_owned());
        }
    }
    let now = now_ms();
    let mut scanned = 0_u64;
    let tx = conn.unchecked_transaction()?;
    for task in TASKS {
        let reg = Path::new(registry_str()).join(corpus_dir_name(task));
        let cache_root = Path::new(encodings_str()).join(format!("{task}-expanded"));
        // The legacy corpus roots are mandatory; a new task's root may not
        // exist on every host yet and simply contributes no rows.
        if !reg.is_dir() && !matches!(task, "regression" | "binary") {
            continue;
        }
        for entry in std::fs::read_dir(&reg)?.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let bare = name.strip_prefix("synth_").unwrap_or(&name);
            if name.starts_with('.') || bare.len() != 16 {
                continue;
            }
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            scanned += 1;
            let has_data = ["train.csv", "test.csv", "meta.json"]
                .iter()
                .all(|f| dir.join(f).is_file());
            let sidecars = SIDECARS
                .iter()
                .all(|s| dir.join(s).metadata().map(|m| m.len() > 0).unwrap_or(false));
            let npy = Path::new(foundation_str())
                .join(format!("{name}.npy"))
                .is_file();
            let cached = cache_root
                .join(&name)
                .join("cache-manifest.v1.json")
                .is_file();
            let is_active = active.contains(&(task.to_owned(), name.clone()));
            let is_sealed = sealed_ids.contains(&name);
            tx.execute(
                "INSERT INTO datasets(task,dataset_hash,has_train_test_meta,sidecars_complete,
                   foundation_npy,cache_materialized,catalog_active,sealed,updated_at_ms)
                 VALUES(?,?,?,?,?,?,?,?,?)
                 ON CONFLICT(task,dataset_hash) DO UPDATE SET
                   has_train_test_meta=excluded.has_train_test_meta,
                   sidecars_complete=excluded.sidecars_complete,
                   foundation_npy=excluded.foundation_npy,
                   cache_materialized=excluded.cache_materialized,
                   catalog_active=excluded.catalog_active,
                   sealed=excluded.sealed,
                   updated_at_ms=excluded.updated_at_ms",
                rusqlite::params![
                    task,
                    name,
                    has_data as i64,
                    sidecars as i64,
                    npy as i64,
                    cached as i64,
                    is_active as i64,
                    is_sealed as i64,
                    now as i64
                ],
            )?;
        }
    }
    tx.commit()?;
    Ok((scanned, now))
}

fn refresh_observation_state(conn: &Connection, root: &Path) -> Result<u64> {
    let last: i64 = conn
        .query_row(
            "SELECT value FROM meta WHERE key='obs_watermark_ms'",
            [],
            |r| r.get::<_, String>(0),
        )
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let corpus = Connection::open_with_flags(
        root.join("corpus/strict-test-exact.sqlite"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )?;
    corpus.busy_timeout(Duration::from_secs(60))?;
    // Count the WHOLE history of every dataset touched since the watermark, not
    // just the rows inside the window. Ingest re-evaluates with REPLACE, which
    // bumps created_at_ms, so a windowed count re-counts rows that were already
    // counted in an earlier cycle — accumulating those deltas inflated obs_count
    // to 5.94M against 3.09M real rows.
    let mut statement = corpus.prepare(
        "WITH changed(task,dataset_hash) AS (
           SELECT DISTINCT task,dataset_hash
             FROM observations INDEXED BY observations_created_v1
            WHERE created_at_ms > ?
         )
         SELECT observations.task, observations.dataset_hash, COUNT(*),
                MAX(observations.score), MAX(observations.created_at_ms)
           FROM changed
           JOIN observations
             ON observations.task=changed.task
            AND observations.dataset_hash=changed.dataset_hash
          GROUP BY observations.task, observations.dataset_hash",
    )?;
    let rows: Vec<(String, String, i64, Option<f64>, i64)> = statement
        .query_map([last], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
        })?
        .collect::<std::result::Result<_, _>>()?;
    drop(statement);
    drop(corpus);
    let mut watermark = last;
    let count = rows.len() as u64;
    let tx = conn.unchecked_transaction()?;
    for (task, dataset, obs, best, newest) in rows {
        watermark = watermark.max(newest);
        tx.execute(
            "INSERT INTO datasets(task,dataset_hash,obs_count,best_score,last_observed_ms,updated_at_ms)
             VALUES(?,?,?,?,?,?)
             ON CONFLICT(task,dataset_hash) DO UPDATE SET
               obs_count=excluded.obs_count,
               best_score=MAX(COALESCE(datasets.best_score,-1e308),COALESCE(excluded.best_score,-1e308)),
               last_observed_ms=excluded.last_observed_ms,
               updated_at_ms=excluded.updated_at_ms",
            rusqlite::params![task, dataset, obs, best, newest, now_ms() as i64],
        )?;
    }
    tx.execute(
        "INSERT INTO meta VALUES('obs_watermark_ms',?)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        [watermark.to_string()],
    )?;
    tx.commit()?;
    Ok(count)
}

fn stage_summary(conn: &Connection) -> Result<Value> {
    let mut summary = BTreeMap::new();
    for task in TASKS {
        let row = conn.query_row(
            "SELECT COUNT(*),
                    SUM(has_train_test_meta),
                    SUM(sidecars_complete),
                    SUM(CASE WHEN sidecars_complete=1 AND foundation_npy=1 THEN 1 ELSE 0 END),
                    SUM(cache_materialized),
                    SUM(catalog_active),
                    SUM(CASE WHEN obs_count>0 THEN 1 ELSE 0 END),
                    SUM(sealed)
             FROM datasets WHERE task=?",
            [task],
            |r| {
                Ok(json!({
                    "registry": r.get::<_, i64>(0)?,
                    "with_data": r.get::<_, Option<i64>>(1)?.unwrap_or(0),
                    "sidecars_complete": r.get::<_, Option<i64>>(2)?.unwrap_or(0),
                    "fully_encoded": r.get::<_, Option<i64>>(3)?.unwrap_or(0),
                    "cache_materialized": r.get::<_, Option<i64>>(4)?.unwrap_or(0),
                    "catalog_active": r.get::<_, Option<i64>>(5)?.unwrap_or(0),
                    "observed": r.get::<_, Option<i64>>(6)?.unwrap_or(0),
                    "sealed": r.get::<_, Option<i64>>(7)?.unwrap_or(0),
                }))
            },
        )?;
        summary.insert(task.to_owned(), row);
    }
    Ok(json!(summary))
}

fn emit_encoding_exclusion_receipts(conn: &Connection, root: &Path) -> Result<usize> {
    let directory = root.join("ops/encoding-exclusions-v1");
    std::fs::create_dir_all(&directory)?;
    let mut statement = conn.prepare(
        "SELECT task,dataset_hash,sealed,has_train_test_meta FROM datasets
          WHERE sealed=1 OR has_train_test_meta=0 ORDER BY task,dataset_hash",
    )?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut created = 0;
    for (task, dataset_hash, sealed, has_data) in rows {
        // Keep this maintenance pass bounded so a large legacy catalog cannot
        // delay encoding admission.  Subsequent two-minute cycles resume
        // monotonically because receipts are immutable and skipped above.
        if created >= 512 {
            break;
        }
        let path = directory.join(format!("{task}-{dataset_hash}.json"));
        if path.is_file() {
            continue;
        }
        let reason = if sealed != 0 {
            "sealed_audit_dataset"
        } else if has_data == 0 {
            "malformed_or_missing_train_test_meta"
        } else {
            "ineligible"
        };
        let receipt = json!({
            "schema": "EncodingEligibilityReceipt.v1",
            "task": task,
            "dataset_hash": dataset_hash,
            "eligible": false,
            "disposition": "excluded",
            "reason": reason,
            "recorded_at_ms": now_ms(),
        });
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        use std::io::Write;
        // Serialize the complete receipt before writing to this newly created file.
        // An I/O failure can still leave partial contents.
        let mut line = serde_json::to_vec(&receipt)?;
        line.push(b'\n');
        file.write_all(&line)?;
        file.sync_data()?;
        created += 1;
    }
    Ok(created)
}

fn tranche_coverage(conn: &Connection, tranche_catalog: &Path) -> Result<f64> {
    let tranche = read_json(tranche_catalog)?;
    let binding = Vec::new();
    let entries = tranche
        .get("entries")
        .and_then(Value::as_array)
        .unwrap_or(&binding);
    if entries.is_empty() {
        return Ok(1.0);
    }
    let mut observed = 0_usize;
    let mut statement =
        conn.prepare("SELECT obs_count FROM datasets WHERE task=? AND dataset_hash=?")?;
    for e in entries {
        let task = e
            .get("task")
            .and_then(Value::as_str)
            .or_else(|| e.pointer("/dataset/task").and_then(Value::as_str))
            .unwrap_or("");
        let hash = e
            .get("dataset_hash")
            .and_then(Value::as_str)
            .or_else(|| e.pointer("/dataset/dataset_hash").and_then(Value::as_str))
            .unwrap_or("");
        let count: i64 = statement.query_row([task, hash], |r| r.get(0)).unwrap_or(0);
        if count > 0 {
            observed += 1;
        }
    }
    Ok(observed as f64 / entries.len() as f64)
}

// ---------------------------------------------------------------------------
// Work steps
// ---------------------------------------------------------------------------

/// Ids of encode jobs scq still considers alive. `None` means scq could not be
/// read — callers must then assume a job MIGHT be live (never submit a
/// duplicate, never reclaim a marker) rather than guess.
fn live_encode_jobs() -> Option<HashSet<String>> {
    let out = sh(scq_str(), &["--json", "ls"]).ok()?;
    let value = serde_json::from_str::<Value>(&out).ok()?;
    let binding = Vec::new();
    Some(
        value
            .pointer("/Jobs/jobs")
            .and_then(Value::as_array)
            .unwrap_or(&binding)
            .iter()
            .filter(|job| {
                matches!(
                    job.get("state").and_then(Value::as_str),
                    Some("queued" | "starting" | "dispatching" | "running" | "preempting" | "held")
                )
            })
            .filter_map(|job| job.get("id").and_then(Value::as_str))
            .filter(|id| {
                id.starts_with("jope-selfplay-breadth-encode-") || id.starts_with("jope-binexp-")
            })
            .map(str::to_owned)
            .collect(),
    )
}

fn live_encode_job(live: &Option<HashSet<String>>) -> bool {
    // Unknown scq state is treated as "a job is live": refusing to submit is
    // recoverable next cycle, double-submitting the same chunk is not.
    live.as_ref().map(|ids| !ids.is_empty()).unwrap_or(true)
}

/// Return `listed:*` markers to NULL so `submit_encode_chunk` can re-list them.
///
/// A dataset is stranded when its chunk job is gone from scq but the encode
/// never landed (sidecars_complete still 0). That is the steady state after any
/// encode failure, and nothing in the driver ever cleared a marker: when the
/// b2c05..b2c16 series died on the hyperion CPU-fallback timeout and then aged
/// out of scq, all 22,747 remaining binary datasets kept their markers, the
/// candidate pool became permanently empty, and the binary lane went silent
/// while regression kept flowing. Clearing markers here is what makes "new
/// datasets always being added" hold for binary without a human.
fn reclaim_stale_encode_markers(
    conn: &Connection,
    live: &Option<HashSet<String>>,
    log: &mut Vec<Value>,
) -> Result<()> {
    let Some(live_ids) = live else { return Ok(()) };
    // Outputs already sitting in the fetch dir encoded fine and are merely
    // waiting for install_batch to flip sidecars_complete. They must not burn
    // retry budget in that window (submit_encode_chunk skips fetched ids
    // anyway, so re-listing them would be a no-op with a cost).
    let fetched: HashSet<String> = std::fs::read_dir(Path::new(fetch_binary_str()).join("targets"))
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    let mut statement = conn.prepare(
        "SELECT dataset_hash, encode_job_state, encode_attempts FROM datasets
          WHERE task='binary' AND encode_job_state LIKE 'listed:%'
            AND sidecars_complete=0 AND catalog_active=0",
    )?;
    let rows: Vec<(String, String, i64)> = statement
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .filter_map(|r| r.ok())
        .collect();
    drop(statement);
    let mut reclaimed = 0_usize;
    let mut exhausted = 0_usize;
    let tx = conn.unchecked_transaction()?;
    for (hash, marker, attempts) in rows {
        // Marker bodies have varied across driver generations ("b2c4",
        // "binexp-b2c05"); match by containment against live job ids instead of
        // reconstructing an exact id, so a live chunk is never reclaimed.
        let body = marker.trim_start_matches("listed:");
        if live_ids.iter().any(|id| id.contains(body)) || fetched.contains(&hash) {
            continue;
        }
        if attempts + 1 >= MAX_ENCODE_ATTEMPTS {
            tx.execute(
                "UPDATE datasets SET encode_job_state='exhausted', encode_attempts=encode_attempts+1
                  WHERE task='binary' AND dataset_hash=?",
                [&hash],
            )?;
            exhausted += 1;
        } else {
            tx.execute(
                "UPDATE datasets SET encode_job_state=NULL, encode_attempts=encode_attempts+1
                  WHERE task='binary' AND dataset_hash=?",
                [&hash],
            )?;
            reclaimed += 1;
        }
    }
    tx.commit()?;
    if reclaimed > 0 || exhausted > 0 {
        log.push(json!({
            "step": "encode_markers_reclaimed",
            "reclaimed": reclaimed,
            "exhausted": exhausted,
        }));
    }
    Ok(())
}

fn submit_encode_chunk(conn: &Connection) -> Result<Option<EncodeChunkSubmission>> {
    // Un-encoded = binary datasets with data but incomplete sidecars, not active,
    // not already present in the fetch dir.
    let fetched: HashSet<String> = std::fs::read_dir(Path::new(fetch_binary_str()).join("targets"))
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    let mut statement = conn.prepare(
        "SELECT dataset_hash, encode_attempts FROM datasets
          WHERE task='binary' AND has_train_test_meta=1 AND sidecars_complete=0
            AND catalog_active=0 AND sealed=0 AND encode_job_state IS NULL ORDER BY dataset_hash",
    )?;
    let pending: Vec<(String, i64)> = statement
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?
        .filter_map(|r| r.ok())
        .filter(|(id, _)| !fetched.contains(id))
        .take(ENCODE_CHUNK)
        .collect();
    if pending.is_empty() {
        return Ok(None);
    }
    let ids = pending.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>();
    let attempt = pending
        .iter()
        .map(|(_, attempts)| attempts + 1)
        .max()
        .unwrap_or(1);
    // Rows carry content pins; sha256 via coreutils in batches to stay dependency-free.
    let registry = Path::new(registry_str()).join("binary");
    let mut digests: BTreeMap<String, String> = BTreeMap::new();
    for chunk in ids.chunks(400) {
        let mut args: Vec<String> = Vec::with_capacity(chunk.len() * 2);
        for id in chunk {
            args.push(
                registry
                    .join(id)
                    .join("train.csv")
                    .to_string_lossy()
                    .into_owned(),
            );
            args.push(
                registry
                    .join(id)
                    .join("test.csv")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();
        for line in sh("sha256sum", &refs)?.lines() {
            if let Some((digest, path)) = line.split_once("  ") {
                digests.insert(path.trim().to_owned(), digest.trim().to_owned());
            }
        }
    }
    let rows: Vec<Value> = ids
        .iter()
        .map(|id| {
            let train = registry.join(id).join("train.csv");
            let test = registry.join(id).join("test.csv");
            json!({
                "id": id,
                "train_sha256": digests.get(&train.to_string_lossy().into_owned()),
                "test_sha256": digests.get(&test.to_string_lossy().into_owned()),
            })
        })
        .collect();
    let mut out = json!({
        "schema": "EncodingChunk.v2",
        "group": "binary-expansion",
        "src_root": format!("{registry_str}/binary", registry_str = registry_str()),
        "out_subdir": "binary-expansion",
        "count": ids.len(),
        "ids": rows,
        "lite": false,
    });
    let runner = bundled_encoding_asset("runner.sh")?;
    let driver = bundled_encoding_asset("encode_batch.py")?;
    let encode_external = bundled_encoding_asset("encode_external.py")?;
    let encoder_identity = sha256_bytes(
        format!(
            "runtime-bundle-v3-encoder|{}|{}|{}",
            sha256_file(&runner)?,
            sha256_file(&driver)?,
            sha256_file(&encode_external)?,
        )
        .as_bytes(),
    );
    let chunk_sha256 = sha256_bytes(&serde_json::to_vec(&json!({
        "schema": "EncodingChunk.v2",
        "encoder_identity_sha256": encoder_identity,
        "datasets": rows,
    }))?);
    let job_id = format!(
        "jope-selfplay-breadth-encode-{}-a{attempt}",
        &chunk_sha256[..16]
    );
    out["schema"] = json!("EncodingChunk.v2");
    out["chunk_sha256"] = json!(chunk_sha256);
    out["encoder_identity_sha256"] = json!(encoder_identity);
    out["attempt_number"] = json!(attempt);
    out["job_identity"] = json!(job_id);
    out["project"] = json!("jope-selfplay-breadth-encode");
    out["required_node"] = json!("xbabe3");
    out["preemptible"] = json!(true);
    out["reserved_cpu_cores"] = json!(4);
    let listing = format!(
        "{}/_batch_manifests/breadth-{}-a{attempt}.v2.json",
        encodings_str(),
        &chunk_sha256[..16]
    );
    let mut listing_bytes = serde_json::to_vec(&out)?;
    listing_bytes.push(b'\n');
    publish_immutable(Path::new(&listing), &listing_bytes)?;
    let spec_directory = Path::new(root_str()).join("specs/breadth-encoding");
    std::fs::create_dir_all(&spec_directory)?;
    let spec = spec_directory.join(format!("{job_id}.toml"));
    let pythonpath = runtime_bundle_root()?.join("pytorch");
    // The bundled runtime FIRST so a bundled module always wins, then the
    // foundation-expert tree. See `foundation_expert_root_str`.
    let job_pythonpath = format!(
        "{}:{foundation_expert_root_str}",
        pythonpath.to_string_lossy(),
        foundation_expert_root_str = foundation_expert_root_str()
    );
    let mut lines = vec![
        "[job]".to_owned(),
        format!("id = \"{job_id}\""),
        "project = \"jope-selfplay-breadth-encode\"".to_owned(),
        "tier = \"normal\"".to_owned(),
        "command = \"/bin/sh\"".to_owned(),
        "args = [\"{input:runner}\", \"{input:driver}\", \"{input:listing}\", \"{output:receipt}\", \"{input:encode_external}\"]".to_owned(),
        String::new(),
        "[resource]".to_owned(),
        "gpus_per_task = 1".to_owned(),
        "cpus_reserved = 4".to_owned(),
        "omp_threads = 4".to_owned(),
        "ram_mb = 16384".to_owned(),
        "gang_size = 1".to_owned(),
        "placement = \"single_node\"".to_owned(),
        String::new(),
        "[[dependencies.smoke]]".to_owned(),
        "name = \"xbabe3 prime encode environment\"".to_owned(),
        "command = \"/usr/bin/test\"".to_owned(),
        format!(
            "args = [\"-f\", \"{}\"]",
            paths::legacy_tree("jope-scratch/binary-expansion/prime_xbabe3.toml").display()
        ),
        "timeout_secs = 5".to_owned(),
        String::new(),
        "[[dependencies.smoke]]".to_owned(),
        "name = \"xbabe3 RuntimeBundle Python package\"".to_owned(),
        "command = \"/usr/bin/env\"".to_owned(),
        format!(
            "args = [{}, \"/usr/bin/python3\", \"-c\", \"import jope; import python.elo_finetune.model_io\"]",
            serde_json::to_string(&format!("PYTHONPATH={job_pythonpath}"))?
        ),
        "timeout_secs = 30".to_owned(),
        String::new(),
        "[env]".to_owned(),
        "OMP_NUM_THREADS = \"4\"".to_owned(),
        format!("JOPE_RUNTIME_PYTHONPATH = \"{job_pythonpath}\""),
        String::new(),
        "[preempt]".to_owned(),
        "preemptible = true".to_owned(),
        "checkpoint = \"none\"".to_owned(),
        "resume = \"rerun\"".to_owned(),
        String::new(),
        "[limits]".to_owned(),
        "max_runtime_hours = 12".to_owned(),
        "max_restarts = 2".to_owned(),
        String::new(),
    ];
    append_spec_input(&mut lines, "runner", &runner, "runner.sh")?;
    append_spec_input(&mut lines, "driver", &driver, "encode_batch.py")?;
    append_spec_input(
        &mut lines,
        "encode_external",
        &encode_external,
        "encode_external.py",
    )?;
    append_spec_input(&mut lines, "listing", Path::new(&listing), "listing.json")?;
    lines.extend([
        "[[output]]".to_owned(),
        "name = \"receipt\"".to_owned(),
        "source = \"receipt\"".to_owned(),
        "directory = true".to_owned(),
        String::new(),
    ]);
    let spec_bytes = lines.join("\n").into_bytes();
    publish_equivalent_bundle_spec(&spec, &spec_bytes)?;
    let submission = scq_check_submit(&spec, &job_id)?;
    // Each id is listed at most once; ids whose source is absent on the encode
    // node PREP-FAIL there and must not be re-listed forever. Successful ones
    // flip sidecars_complete via the fetched outputs regardless of this marker.
    let tx = conn.unchecked_transaction()?;
    for id in &ids {
        tx.execute(
            "UPDATE datasets SET encode_job_state=? WHERE task='binary' AND dataset_hash=?",
            rusqlite::params![format!("listed:{job_id}"), id],
        )?;
    }
    tx.commit()?;
    Ok(Some(submission))
}

/// Exhaustion must be a BACKOFF, never a death sentence. Datasets get parked
/// after MAX_ENCODE_ATTEMPTS, but placement failures can burn the whole budget
/// without a dataset ever reaching a node that has the encode environment
/// (scq picks the node and the spec's hostname dependency does not constrain
/// placement — b2c17 attempt 1 landed on xbabe1, which has no prime config).
/// Once nothing submittable is left, un-park everything and start over: broken
/// datasets are only ever retried after the healthy backlog is exhausted, so
/// this cannot starve real work, and the binary lane can never permanently die.
fn unpark_exhausted_if_starved(conn: &Connection, log: &mut Vec<Value>) -> Result<()> {
    let submittable: i64 = conn.query_row(
        "SELECT COUNT(*) FROM datasets WHERE task='binary' AND has_train_test_meta=1
           AND sidecars_complete=0 AND catalog_active=0 AND encode_job_state IS NULL",
        [],
        |r| r.get(0),
    )?;
    if submittable > 0 {
        return Ok(());
    }
    let unparked = conn.execute(
        "UPDATE datasets SET encode_job_state=NULL, encode_attempts=0
          WHERE task='binary' AND encode_job_state='exhausted'",
        [],
    )?;
    if unparked > 0 {
        log.push(json!({"step": "encode_exhausted_unparked", "datasets": unparked}));
    }
    Ok(())
}

fn fetch_root(task: &str) -> PathBuf {
    if task == "binary" {
        PathBuf::from(fetch_binary_str())
    } else {
        paths::flywheel_encodings_root().join("_fetch/regression-expansion")
    }
}

fn hash_directory(path: &Path) -> Result<String> {
    let mut files = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(directory) = stack.pop() {
        for entry in std::fs::read_dir(directory)?.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file() {
                files.push(path);
            }
        }
    }
    files.sort();
    let mut identity = Vec::new();
    for file in files {
        identity.extend_from_slice(
            file.strip_prefix(path)
                .unwrap_or(&file)
                .to_string_lossy()
                .as_bytes(),
        );
        identity.push(0);
        identity.extend_from_slice(sha256_file(&file)?.as_bytes());
        identity.push(b'\n');
    }
    Ok(sha256_bytes(&identity))
}

fn encoding_chunk_listing(job_id: &str) -> Result<Option<Value>> {
    let directory = paths::flywheel_encodings_root().join("_batch_manifests");
    for entry in std::fs::read_dir(directory)?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Ok(value) = read_json(&path) else {
            continue;
        };
        if value.get("schema").and_then(Value::as_str) == Some("EncodingChunk.v2")
            && value.get("job_identity").and_then(Value::as_str) == Some(job_id)
        {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn publish_encoding_receipt(receipt: &Value) -> Result<()> {
    let dataset_content = receipt
        .pointer("/dataset/content_sha256")
        .and_then(Value::as_str)
        .context("encoding receipt lacks dataset content identity")?;
    let attempt = receipt
        .get("attempt_number")
        .and_then(Value::as_u64)
        .context("encoding receipt lacks attempt number")?;
    let job = receipt
        .get("job_identity")
        .and_then(Value::as_str)
        .context("encoding receipt lacks job identity")?;
    let directory = Path::new(root_str()).join("ops/encoding-adoption-v2");
    std::fs::create_dir_all(&directory)?;
    let path = directory.join(format!(
        "{}-a{}-{}.json",
        dataset_content,
        attempt,
        &sha256_bytes(job.as_bytes())[..16]
    ));
    let mut bytes = serde_json::to_vec(receipt)?;
    bytes.push(b'\n');
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(mut file) => {
            use std::io::Write;
            file.write_all(&bytes)?;
            file.sync_data()?;
            let mut events = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(Path::new(root_str()).join("ops/encoding-adoption-v2.jsonl"))?;
            events.write_all(&bytes)?;
            events.sync_data()?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            if std::fs::read(&path)? != bytes {
                bail!(
                    "immutable encoding adoption receipt drift: {}",
                    path.display()
                );
            }
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn emit_chunk_adoption_receipts(
    job_id: &str,
    job_state: &str,
    task: &str,
    root: &Path,
) -> Result<usize> {
    let Some(listing) = encoding_chunk_listing(job_id)? else {
        return Ok(0);
    };
    let encoder = listing
        .get("encoder_identity_sha256")
        .and_then(Value::as_str)
        .context("EncodingChunk.v2 lacks encoder identity")?;
    let attempt = listing
        .get("attempt_number")
        .and_then(Value::as_u64)
        .context("EncodingChunk.v2 lacks attempt number")?;
    let rows = listing
        .get("ids")
        .and_then(Value::as_array)
        .context("EncodingChunk.v2 lacks dataset identities")?;
    let registry = Path::new(registry_str()).join(task);
    let mut adopted = 0;
    for row in rows {
        let dataset_id = row
            .get("id")
            .and_then(Value::as_str)
            .context("EncodingChunk.v2 dataset lacks id")?;
        let train_sha256 = row
            .get("train_sha256")
            .and_then(Value::as_str)
            .context("EncodingChunk.v2 dataset lacks train hash")?;
        let test_sha256 = row
            .get("test_sha256")
            .and_then(Value::as_str)
            .context("EncodingChunk.v2 dataset lacks test hash")?;
        let train = registry.join(dataset_id).join("train.csv");
        let test = registry.join(dataset_id).join("test.csv");
        if sha256_file(&train)? != train_sha256 || sha256_file(&test)? != test_sha256 {
            bail!("{job_id}: registry source content changed for {dataset_id}");
        }
        let dataset_content_sha256 =
            sha256_bytes(format!("{train_sha256}:{test_sha256}").as_bytes());
        let target = root.join("targets").join(dataset_id);
        let foundation = root.join("foundation").join(format!("{dataset_id}.npy"));
        let disposition = if target.is_dir() {
            adopted += 1;
            json!({
                "kind": "succeeded",
                "encoded_train_sha256": hash_directory(&target)?,
                "encoded_test_sha256": if foundation.is_file() {
                    sha256_file(&foundation)?
                } else {
                    hash_directory(&target)?
                },
            })
        } else {
            json!({
                "kind": "failed",
                "error_class": format!("{job_state}:missing_dataset_output"),
            })
        };
        publish_encoding_receipt(&json!({
            "schema": "EncodingAdoptionReceipt.v2",
            "dataset": {
                "dataset_id": dataset_id,
                "content_sha256": dataset_content_sha256,
            },
            "attempt_number": attempt,
            "job_identity": job_id,
            "encoder_identity_sha256": encoder,
            "disposition": disposition,
            "recorded_at_ms": now_ms(),
        }))?;
    }
    Ok(adopted)
}

/// Pull outputs of completed encode chunks into the per-task fetch dirs.
/// Encoding and fetching are deliberately OUTSIDE the guardrail/valve — the
/// valve paces catalog admission only, never encode throughput.
fn reconcile_encode_fetches(log: &mut Vec<Value>) -> Result<()> {
    let Ok(out) = sh(scq_str(), &["--json", "ls"]) else {
        return Ok(());
    };
    let Ok(value) = serde_json::from_str::<Value>(&out) else {
        return Ok(());
    };
    let binding = Vec::new();
    for job in value
        .pointer("/Jobs/jobs")
        .and_then(Value::as_array)
        .unwrap_or(&binding)
    {
        let id = job.get("id").and_then(Value::as_str).unwrap_or("");
        let job_state = job
            .get("state")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        if !(id.starts_with("jope-selfplay-breadth-encode-")
            || id.starts_with("jope-binexp-")
            || id.starts_with("jope-regexp-"))
        {
            continue;
        }
        if !matches!(job_state, "completed" | "failed" | "cancelled") {
            continue;
        }
        let task =
            if id.starts_with("jope-binexp-") || id.starts_with("jope-selfplay-breadth-encode-") {
                "binary"
            } else {
                "regression"
            };
        let root = fetch_root(task);
        let marker = root.join(format!(".fetched-{id}"));
        if marker.exists() {
            continue;
        }
        std::fs::create_dir_all(root.join("targets"))?;
        std::fs::create_dir_all(root.join("foundation"))?;
        let tmp = root.join(format!(".fetch-tmp-{id}"));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp)?;
        if sh(
            scq_str(),
            &["fetch", id, "--output", &tmp.to_string_lossy()],
        )
        .is_err()
        {
            let _ = std::fs::remove_dir_all(&tmp);
            continue;
        }
        let mut moved = 0_usize;
        let mut stack = vec![tmp.clone()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir)?.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().into_owned();
                if path.is_dir() && name == "targets" {
                    for ds in std::fs::read_dir(&path)?.flatten() {
                        let to = root.join("targets").join(ds.file_name());
                        if !to.exists() && std::fs::rename(ds.path(), &to).is_ok() {
                            moved += 1;
                        }
                    }
                } else if path.is_dir() && name == "foundation" {
                    for f in std::fs::read_dir(&path)?.flatten() {
                        let to = root.join("foundation").join(f.file_name());
                        if !to.exists() {
                            let _ = std::fs::rename(f.path(), &to);
                        }
                    }
                } else if path.is_dir() {
                    stack.push(path);
                }
            }
        }
        let _ = std::fs::remove_dir_all(&tmp);
        let adopted = emit_chunk_adoption_receipts(id, job_state, task, &root)?;
        std::fs::write(&marker, b"ok")?;
        log.push(json!({
            "step": "encode_fetched",
            "job": id,
            "terminal_state": job_state,
            "datasets_moved": moved,
            "datasets_adopted": adopted,
        }));
    }
    Ok(())
}

fn installable(conn: &Connection, task: &str) -> Result<Vec<String>> {
    let fetched: HashSet<String> = std::fs::read_dir(fetch_root(task).join("targets"))
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    let mut statement = conn.prepare(
        "SELECT dataset_hash FROM datasets
          WHERE task=? AND catalog_active=0 AND sealed=0 AND sidecars_complete=0
          ORDER BY dataset_hash",
    )?;
    let ids: Vec<String> = statement
        .query_map([task], |r| r.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .filter(|id| fetched.contains(id))
        .collect();
    Ok(ids)
}

fn curate_tranche(
    root: &Path,
    conn: &Connection,
    task: &str,
    tranche_file: &Path,
) -> Result<usize> {
    let inv = read_json(&root.join("catalog/v2/inventory-adaptive-v1.json"))?;
    let binding = Vec::new();
    let entries = inv
        .get("entries")
        .and_then(Value::as_array)
        .unwrap_or(&binding);
    // Regression foundation .npy files are produced by foundation_backfill.py
    // AFTER curation (run_tranche), and prune_tranche.py drops any entry whose
    // .npy is still missing before activation. Requiring foundation_npy=1 here
    // therefore starved curation to zero the moment the pre-encoded pool was
    // fully activated (the silent 0-tranche stall of 2026-07-27/28: 6,412
    // fully-encoded = 6,412 active, 17,557 sidecar-complete datasets waiting).
    // Binary has no backfill lane — its .npy arrives with the encode fetch —
    // so the gate stays for binary.
    let sql = if task == "regression" {
        "SELECT 1 FROM datasets WHERE task=? AND dataset_hash=? AND sidecars_complete=1
            AND catalog_active=0 AND sealed=0"
    } else {
        "SELECT 1 FROM datasets WHERE task=? AND dataset_hash=? AND sidecars_complete=1
            AND foundation_npy=1 AND catalog_active=0 AND sealed=0"
    };
    let mut statement = conn.prepare(sql)?;
    let mut selected = Vec::new();
    for e in entries {
        if e.get("task").and_then(Value::as_str) != Some(task) {
            continue;
        }
        let hash = e.get("dataset_hash").and_then(Value::as_str).unwrap_or("");
        if statement.query_row([task, hash], |_| Ok(())).is_ok() {
            selected.push(e.clone());
            if selected.len() >= TRANCHE_CAP {
                break;
            }
        }
    }
    std::fs::write(tranche_file, serde_json::to_vec(&selected)?)?;
    Ok(selected.len())
}

fn prune_tranche(tranche_file: &Path, task: &str) -> Result<Value> {
    let entries = read_json(tranche_file)?;
    let entries = entries
        .as_array()
        .context("tranche inventory must be a JSON array")?;
    let mut kept = Vec::with_capacity(entries.len());
    let mut dropped = Vec::new();
    for entry in entries {
        let dataset = entry
            .get("dataset_id")
            .and_then(Value::as_str)
            .context("tranche entry lacks dataset_id")?;
        let cache = Path::new(encodings_str())
            .join(format!("{task}-expanded"))
            .join(dataset);
        let mut reason = None;
        let foundation = Path::new(foundation_str()).join(format!("{dataset}.npy"));
        if !foundation.is_file() || foundation.metadata()?.len() == 0 {
            reason = Some("missing_or_empty_foundation_npy".to_owned());
        } else {
            for sidecar in SIDECARS {
                let path = cache.join(sidecar);
                if !path.is_file() || path.metadata()?.len() == 0 {
                    reason = Some(format!("empty_or_missing:{sidecar}"));
                    break;
                }
            }
        }
        if let Some(reason) = reason {
            dropped.push(json!({"dataset_id": dataset, "reason": reason}));
        } else {
            kept.push(entry.clone());
        }
    }
    let temporary = tranche_file.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&temporary, serde_json::to_vec(&kept)?)?;
    std::fs::rename(temporary, tranche_file)?;
    Ok(json!({"kept": kept.len(), "dropped": dropped}))
}

fn run_tranche(
    root: &Path,
    conn: &Connection,
    task: &str,
    log: &mut Vec<Value>,
) -> Result<Option<String>> {
    let acts = activations(root)?;
    let prefix = if task == "regression" { 'r' } else { 'b' };
    let tranche_id = next_tranche_id(&acts, prefix);
    let tranche_file = root.join(format!(
        "catalog/v2/tranche-{}.inventory.json",
        &tranche_id["expanded-".len()..]
    ));
    let selected = curate_tranche(root, conn, task, &tranche_file)?;
    if selected == 0 {
        // Log the empty selection: four clean-guardrail ticks stalled silently
        // (log: []) before this receipt line existed.
        log.push(json!({"step": "tranche_empty", "task": task, "tranche": tranche_id}));
        return Ok(None);
    }
    log.push(json!({"step": "curated", "tranche": tranche_id, "selected": selected}));
    if task == "regression" {
        // Foundation backfill (torch — the sanctioned Python boundary).
        let foundation_backfill = bundled_encoding_asset("foundation_backfill.py")?;
        sh(
            "nice",
            &[
                "-n",
                "15",
                "python3",
                &foundation_backfill.to_string_lossy(),
                &tranche_file.to_string_lossy(),
            ],
        )?;
    }
    let tool = bundled_binary("jope_expanded_catalog")?;
    let tool = tool.to_string_lossy();
    sh(
        &tool,
        &[
            "materialize",
            "--inventory",
            &tranche_file.to_string_lossy(),
            "--task",
            task,
        ],
    )?;
    let pruned = prune_tranche(&tranche_file, task)?;
    log.push(json!({"step": "pruned", "detail": pruned}));
    // Curation no longer pre-requires foundation_npy for regression, so a
    // wholesale backfill failure can prune the tranche to zero — never hand an
    // empty inventory to activate.
    let remaining = read_json(&tranche_file)?
        .as_array()
        .map(Vec::len)
        .unwrap_or(0);
    if remaining == 0 {
        log.push(json!({"step": "tranche_all_pruned", "task": task, "tranche": tranche_id}));
        return Ok(None);
    }
    sh(
        &tool,
        &[
            "activate",
            "--inventory",
            &tranche_file.to_string_lossy(),
            "--tranche-id",
            &tranche_id,
            "--limit",
            "512",
        ],
    )?;
    let merged = sh(&tool, &["merge-live"])?;
    log.push(json!({"step": "merged", "detail": merged.trim()}));
    Ok(Some(tranche_id))
}

fn install_batch(root: &Path, task: &str, ids: &[String], log: &mut Vec<Value>) -> Result<()> {
    let fetch = fetch_root(task);
    let fetched_foundation = fetch.join("foundation");
    let mut verified = 0_usize;
    for id in ids {
        let from = fetched_foundation.join(format!("{id}.npy"));
        if !from.is_file() {
            continue;
        }
        let published = Path::new(foundation_str()).join(format!("{id}.npy"));
        if !published.is_file() {
            bail!(
                "foundation V1 is immutable and has no object for {id}; keep expansion disabled until Encoder V2 publishes a namespaced object"
            );
        }
        if sha256_file(&from)? != sha256_file(&published)? {
            bail!("foundation V1 identity collision for {id}");
        }
        verified += 1;
    }
    let receipts = root.join("ops/sidecar-install-driver.v1.jsonl");
    let tool = bundled_binary("jope_expanded_catalog")?;
    sh(
        &tool.to_string_lossy(),
        &[
            "install-sidecars",
            "--from",
            &fetch.join("targets").to_string_lossy(),
            "--task",
            task,
            "--receipts",
            &receipts.to_string_lossy(),
        ],
    )?;
    log.push(json!({"step": "install", "task": task, "sidecar_batch": ids.len(), "foundation_v1_verified": verified}));
    Ok(())
}

// ---------------------------------------------------------------------------
// Cycle
// ---------------------------------------------------------------------------

fn run_cycle(root: &Path) -> Result<Value> {
    reject_v1_quarantine_window(Path::new(encodings_str()))?;
    let conn = open_state_db(root)?;
    let (scanned, _) = refresh_filesystem_state(&conn, root)?;
    let exclusion_receipts_created = emit_encoding_exclusion_receipts(&conn, root)?;
    let obs_rows = refresh_observation_state(&conn, root).unwrap_or(0);
    let acts = activations(root)?;
    let coverage_for = |prefix: char| -> Result<Option<f64>> {
        Ok(match latest_tranche_for(&acts, prefix) {
            Some((_, catalog)) if catalog.is_file() => Some(tranche_coverage(&conn, &catalog)?),
            _ => None,
        })
    };
    let coverage_regression = coverage_for('r')?;
    let coverage_binary = coverage_for('b')?;
    let mut log: Vec<Value> = Vec::new();
    let disk_paths = vec![paths::jope_state(), PathBuf::from(encodings_str())];
    let disk_block = disk_hysteresis_block(root, &disk_paths)?;
    // Encoding is valve-free by design: fetch completed chunk outputs and keep
    // the encode series topped up BEFORE any guardrail/valve consideration.
    //
    // The whole encode lane is NON-FATAL to the cycle. It used to propagate
    // with `?`, which meant one bad `scq` call took the regression lane down
    // with it: run_cycle returned Err before ever reaching the guardrail /
    // tranche / activation block below, so an unrelated binary-side hiccup
    // silently cost that hour's dataset activation. That is now impossible.
    // It also has to be non-fatal because the encode spec declares a real
    // dependency: scq accepts a dependency-declared job ONLY while a
    // satisfying node has free capacity, so `scq check` exiting 1 is a routine
    // "xbabe3 is busy, try next hour" state, not an error worth losing a
    // tranche over.
    let mut submitted_chunk = None;
    let encode_lane = if let Some(reason) = &disk_block {
        log.push(json!({"step": "encode_lane_disk_paused", "reason": reason}));
        Ok(())
    } else {
        (|| -> Result<()> {
            reconcile_encode_fetches(&mut log)?;
            let live = live_encode_jobs();
            // Return stranded listings before deciding whether to submit, so a dead
            // chunk is re-listed on the very same cycle that noticed it died.
            reclaim_stale_encode_markers(&conn, &live, &mut log)?;
            unpark_exhausted_if_starved(&conn, &mut log)?;
            // Rule (d): top-up submission — with the full series queued upfront this
            // only fires if the queue ever drains (e.g. after failures).
            if !live_encode_job(&live) {
                submitted_chunk = match pending_breadth_admission(root)? {
                    Some(pending) => Some(pending),
                    None => submit_encode_chunk(&conn)?,
                };
                if let Some(submission) = &submitted_chunk {
                    log.push(json!({
                        "step": if submission.disposition == "submitted" {
                            "encode_chunk_submitted"
                        } else {
                            "encode_chunk_capacity_backoff"
                        },
                        "job": submission.job_id,
                        "disposition": submission.disposition,
                        "reason": submission.backoff_reason,
                    }));
                }
            }
            Ok(())
        })()
    };
    if let Err(err) = encode_lane {
        // Recorded, never raised: the receipt is the only place this is
        // visible, so it must say plainly what failed.
        log.push(json!({
            "step": "encode_lane_deferred",
            "detail": format!("{err:#}"),
        }));
    }
    let guards = Guardrails {
        alarms_path: root.join("ops/health-alarms.v1.json"),
        keeper_path: root.join("ops/queue-keeper-v3.json"),
        waves_dir: root.join("waves"),
        disk_paths,
    };
    // Shared guards (alarms/keeper/waves/disk) gate everything; the valve is
    // evaluated per task against that task's own last tranche.
    let blocked = disk_block.or_else(|| guardrail_block(&guards, None));
    let valve_shut = |cov: Option<f64>| -> Option<String> {
        cov.filter(|c| *c < COVERAGE_VALVE)
            .map(|c| format!("coverage valve: previous tranche at {c:.2} < {COVERAGE_VALVE}"))
    };
    let mut tranche = None;
    if let Some(reason) = &blocked {
        log.push(json!({"step": "tranche_skipped", "reason": reason}));
    } else {
        // Rule (b): binary install+tranche when enough fetched encodes await.
        // Binary is preferred but must never STARVE regression: with per-task
        // valves a shut binary valve is a routine state, and falling through
        // keeps the regression lane widening meanwhile. Still at most one
        // tranche per cycle.
        let binary_ready = installable(&conn, "binary")?;
        let binary_valve = valve_shut(coverage_binary);
        if binary_ready.len() >= BINARY_INSTALL_MIN && binary_valve.is_none() {
            install_batch(root, "binary", &binary_ready, &mut log)?;
            refresh_filesystem_state(&conn, root)?;
            tranche = run_tranche(root, &conn, "binary", &mut log)?;
        } else {
            if let (true, Some(reason)) = (binary_ready.len() >= BINARY_INSTALL_MIN, &binary_valve)
            {
                log.push(json!({"step": "tranche_skipped", "task": "binary", "reason": reason}));
            }
            // Rule (c): regression — install any fetched full-encode outputs
            // first (they join the sidecar-ready pool), then tranche.
            match valve_shut(coverage_regression) {
                Some(reason) => log.push(
                    json!({"step": "tranche_skipped", "task": "regression", "reason": reason}),
                ),
                None => {
                    let regression_ready = installable(&conn, "regression")?;
                    if !regression_ready.is_empty() {
                        install_batch(root, "regression", &regression_ready, &mut log)?;
                        refresh_filesystem_state(&conn, root)?;
                    }
                    tranche = run_tranche(root, &conn, "regression", &mut log)?;
                }
            }
        }
    }
    let summary = stage_summary(&conn)?;
    let report = json!({
        "schema": "JopeSelfplayExpansionDriver.v1",
        "generated_at_ms": now_ms(),
        "registry_scanned": scanned,
        "encoding_exclusion_receipts_created": exclusion_receipts_created,
        "obs_rows_ingested": obs_rows,
        // Kept for receipt continuity with earlier cycles; regression is the
        // lane that paces catalog growth today.
        "previous_tranche_coverage": coverage_regression,
        "coverage_regression": coverage_regression,
        "coverage_binary": coverage_binary,
        "guardrail_block": blocked,
        "tranche_completed": tranche,
        "encode_chunk_submitted": &submitted_chunk,
        "encode_chunk_backoff_reason": submitted_chunk
            .as_ref()
            .and_then(|submission| submission.backoff_reason.clone()),
        "stage_summary": summary,
        "log": log,
    });
    let target = root.join("ops/expansion-driver-v1.json");
    let tmp = root.join(format!(
        "ops/.expansion-driver-v1.json.{}",
        std::process::id()
    ));
    std::fs::write(&tmp, serde_json::to_vec_pretty(&report)?)?;
    std::fs::rename(&tmp, &target)?;
    let mut line = serde_json::to_vec(&report)?;
    line.push(b'\n');
    use std::io::Write;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join("ops/expansion-events.v1.jsonl"))?
        .write_all(&line)?;
    Ok(report)
}

fn query(root: &Path, dataset: Option<&str>, summary: bool) -> Result<()> {
    let conn = open_state_db(root)?;
    if let Some(id) = dataset {
        let mut statement = conn.prepare("SELECT * FROM datasets WHERE dataset_hash=?")?;
        let names: Vec<String> = statement
            .column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let rows = statement.query_map([id], |r| {
            let mut object = serde_json::Map::new();
            for (index, name) in names.iter().enumerate() {
                let value: Value = match r.get_ref(index)? {
                    rusqlite::types::ValueRef::Null => Value::Null,
                    rusqlite::types::ValueRef::Integer(v) => json!(v),
                    rusqlite::types::ValueRef::Real(v) => json!(v),
                    rusqlite::types::ValueRef::Text(v) => json!(String::from_utf8_lossy(v)),
                    rusqlite::types::ValueRef::Blob(_) => json!("<blob>"),
                };
                object.insert(name.clone(), value);
            }
            Ok(Value::Object(object))
        })?;
        for row in rows {
            println!("{}", serde_json::to_string(&row?)?);
        }
    }
    if summary {
        println!("{}", serde_json::to_string_pretty(&stage_summary(&conn)?)?);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(root_str());
    match args.get(1).map(String::as_str) {
        Some("query") => {
            let dataset = args
                .iter()
                .position(|a| a == "--dataset")
                .and_then(|i| args.get(i + 1));
            let summary = args.iter().any(|a| a == "--summary");
            query(
                &root,
                dataset.map(String::as_str),
                summary || dataset.is_none(),
            )
        }
        Some("run") | None => {
            let report = run_cycle(&root)?;
            println!("{report}");
            Ok(())
        }
        Some(other) => Err(anyhow!("unknown subcommand {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incident_quarantine_disables_legacy_driver_before_cycle_work() {
        let dir = tempfile::tempdir().unwrap();
        reject_v1_quarantine_window(dir.path()).unwrap();
        std::fs::create_dir(dir.path().join("incident-quarantine")).unwrap();
        let error = reject_v1_quarantine_window(dir.path()).unwrap_err();
        assert!(error.to_string().contains("without readable frozen index"));
    }

    /// Every encode job must be able to import the foundation expert, and the
    /// smoke probe must test the SAME path the job runs with.
    ///
    /// Without the expert root on PYTHONPATH the encoder dies on both devices
    /// while logging a device fallback, so a job burns ~854 seconds and reports
    /// the wrong cause. Bundled runner.sh REPLACES PYTHONPATH with this one
    /// value, so if it is dropped here nothing downstream can recover it.
    #[test]
    fn encode_job_pythonpath_carries_the_foundation_expert_root() {
        let bundle_pytorch =
            jope_paths::to_string(&paths::runtime_bundles_root().join("abc/pytorch"));
        let bundle_pytorch = bundle_pytorch.as_str();
        let job_pythonpath = format!(
            "{bundle_pytorch}:{foundation_expert_root_str}",
            foundation_expert_root_str = foundation_expert_root_str()
        );

        assert!(
            job_pythonpath.contains(foundation_expert_root_str()),
            "the expert root must survive into the job environment"
        );
        // Bundled runtime first: a bundled module must win over the loose tree.
        assert!(
            job_pythonpath.starts_with(bundle_pytorch),
            "the content-addressed bundle must take precedence: {job_pythonpath}"
        );
        // Colon-joined, because runner.sh exports the value verbatim.
        assert_eq!(job_pythonpath.split(':').count(), 2, "{job_pythonpath}");
        // The root that vanished must never come back: the expert tree is the
        // one shipped inside the checkout, never a loose per-host directory.
        assert!(
            Path::new(foundation_expert_root_str()).starts_with(paths::jope_home()),
            "the pre-move root is gone; referencing it is how this broke"
        );
    }

    #[test]
    fn tranche_ids_increment_per_prefix() {
        let acts = vec![
            json!({"tranche_id": "expanded-r0001"}),
            json!({"tranche_id": "expanded-b0001"}),
            json!({"tranche_id": "expanded-r0002"}),
        ];
        assert_eq!(next_tranche_id(&acts, 'r'), "expanded-r0003");
        assert_eq!(next_tranche_id(&acts, 'b'), "expanded-b0002");
        assert_eq!(next_tranche_id(&[], 'r'), "expanded-r0001");
    }

    #[test]
    fn encoding_capacity_pressure_is_a_normal_backoff() {
        assert!(temporarily_lacks_launch_capacity(
            "error[dependency-unsatisfied]: dependencies cannot be verified on any eligible \
             node; eligible_node launch-ready-capacity: no eligible launch-ready node currently"
        ));
        assert!(!temporarily_lacks_launch_capacity(
            "error[dependency-unsatisfied]: xbabe3 prime capability is missing"
        ));
        assert!(temporarily_lacks_launch_capacity(
            "error[dependency-unsatisfied]: xbabe1: smoke xbabe3 prime encode environment: \
             smoke test exited with exit 1"
        ));
        assert!(!temporarily_lacks_launch_capacity(
            "error[dependency-unsatisfied]: unrelated smoke test exited with exit 1"
        ));
        assert!(!temporarily_lacks_launch_capacity(
            "no eligible launch-ready node currently"
        ));
    }

    #[test]
    fn equivalent_runtime_bundles_preserve_immutable_chunk_specs() {
        let first = format!(
            "source = \"/runtime-bundles/{}/pytorch/encoding/runner.sh\"\nsha256 = \"same\"\n",
            "a".repeat(64)
        );
        let second = format!(
            "source = \"/runtime-bundles/{}/pytorch/encoding/runner.sh\"\nsha256 = \"same\"\n",
            "b".repeat(64)
        );
        assert_eq!(
            normalized_runtime_bundle_paths(&first),
            normalized_runtime_bundle_paths(&second)
        );
        assert_ne!(
            normalized_runtime_bundle_paths(&first),
            normalized_runtime_bundle_paths(&second.replace("\"same\"", "\"drift\""))
        );
    }

    #[test]
    fn guardrails_block_on_each_condition() {
        let dir = std::env::temp_dir().join(format!("expdrv-test-{}", std::process::id()));
        let waves = dir.join("waves");
        std::fs::create_dir_all(waves.join("strict-wave-000001")).unwrap();
        let alarms = dir.join("alarms.json");
        let keeper = dir.join("keeper.json");
        std::fs::write(&alarms, r#"{"alarms":[]}"#).unwrap();
        std::fs::write(&keeper, "{}").unwrap();
        let guards = Guardrails {
            alarms_path: alarms.clone(),
            keeper_path: keeper.clone(),
            waves_dir: waves.clone(),
            disk_paths: vec![],
        };
        assert!(guardrail_block(&guards, Some(0.9)).is_none());
        assert!(guardrail_block(&guards, Some(0.1))
            .unwrap()
            .contains("coverage valve"));
        std::fs::write(
            &alarms,
            r#"{"alarms":[{"alarm":"x","severity":"critical"}]}"#,
        )
        .unwrap();
        assert!(guardrail_block(&guards, Some(0.9))
            .unwrap()
            .contains("critical alarm"));
        std::fs::write(
            &alarms,
            r#"{"alarms":[
                {"alarm":"status_stale:ops/task-learning-controller.v2.json","severity":"critical"},
                {"alarm":"status_stale:ops/trainer-watchdog-v1.json","severity":"critical"},
                {"alarm":"no_active_trainer","severity":"critical"},
                {"alarm":"scq_jobs_held_starving","severity":"critical"}
            ]}"#,
        )
        .unwrap();
        assert!(guardrail_block(&guards, Some(0.9)).is_none());
        std::fs::write(&alarms, r#"{"alarms":[]}"#).unwrap();
        let missing = Guardrails {
            alarms_path: alarms,
            keeper_path: dir.join("absent.json"),
            waves_dir: waves,
            disk_paths: vec![],
        };
        assert!(missing.keeper_path.metadata().is_err());
        assert!(guardrail_block(&missing, None).unwrap().contains("keeper"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn disk_hysteresis_stops_at_85_and_resumes_only_below_80() {
        assert!(!disk_pause_decision(false, Some(84.9)));
        assert!(disk_pause_decision(false, Some(85.0)));
        assert!(disk_pause_decision(true, Some(80.0)));
        assert!(!disk_pause_decision(true, Some(79.9)));
        assert!(disk_pause_decision(true, None));
    }

    #[test]
    fn valve_reads_the_last_tranche_of_its_own_task() {
        let acts = vec![
            json!({"tranche_id": "expanded-b0001", "catalog": "/b1.json"}),
            json!({"tranche_id": "expanded-r0002", "catalog": "/r2.json"}),
            json!({"tranche_id": "expanded-r0003", "catalog": "/r3.json"}),
        ];
        // Binary must still see b0001 even though r0003 is globally last.
        assert_eq!(latest_tranche_for(&acts, 'b').unwrap().0, "expanded-b0001");
        assert_eq!(latest_tranche_for(&acts, 'r').unwrap().0, "expanded-r0003");
        assert!(latest_tranche_for(&[], 'b').is_none());
    }

    #[test]
    fn reclaim_returns_dead_listings_and_parks_exhausted_ones() {
        let dir = std::env::temp_dir().join(format!("expdrv-reclaim-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("ops")).unwrap();
        let conn = open_state_db(&dir).unwrap();
        // dead: job gone from scq, no sidecars -> reclaimed
        // live: job still queued -> untouched
        // last: one attempt from the budget -> parked as exhausted
        // done: encode landed -> untouched even though the job is gone
        for (hash, marker, attempts, sidecars) in [
            ("dead", "listed:b2c05", 0, 0),
            ("live", "listed:b2c99", 0, 0),
            ("last", "listed:b2c06", MAX_ENCODE_ATTEMPTS - 1, 0),
            ("done", "listed:b2c07", 0, 1),
        ] {
            conn.execute(
                "INSERT INTO datasets(task,dataset_hash,sidecars_complete,encode_job_state,
                   encode_attempts,updated_at_ms) VALUES('binary',?,?,?,?,1)",
                rusqlite::params![hash, sidecars, marker, attempts],
            )
            .unwrap();
        }
        let live: Option<HashSet<String>> =
            Some(["jope-binexp-b2c99".to_owned()].into_iter().collect());
        let mut log = Vec::new();
        reclaim_stale_encode_markers(&conn, &live, &mut log).unwrap();
        let state = |h: &str| -> Option<String> {
            conn.query_row(
                "SELECT encode_job_state FROM datasets WHERE dataset_hash=?",
                [h],
                |r| r.get::<_, Option<String>>(0),
            )
            .unwrap()
        };
        assert_eq!(state("dead"), None, "dead listing must be re-listable");
        assert_eq!(
            state("live").as_deref(),
            Some("listed:b2c99"),
            "live chunk untouched"
        );
        assert_eq!(
            state("last").as_deref(),
            Some("exhausted"),
            "budget must park"
        );
        assert_eq!(
            state("done").as_deref(),
            Some("listed:b2c07"),
            "encoded rows untouched"
        );
        assert_eq!(log[0]["reclaimed"], json!(1));
        assert_eq!(log[0]["exhausted"], json!(1));
        // Unknown scq state must never reclaim.
        let mut log2 = Vec::new();
        reclaim_stale_encode_markers(&conn, &None, &mut log2).unwrap();
        assert!(log2.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn exhaustion_is_a_backoff_not_a_death_sentence() {
        let dir = std::env::temp_dir().join(format!("expdrv-unpark-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("ops")).unwrap();
        let conn = open_state_db(&dir).unwrap();
        conn.execute(
            "INSERT INTO datasets(task,dataset_hash,has_train_test_meta,sidecars_complete,
               catalog_active,encode_job_state,encode_attempts,updated_at_ms)
             VALUES('binary','parked',1,0,0,'exhausted',8,1)",
            [],
        )
        .unwrap();
        // While a submittable dataset still exists, parked rows stay parked.
        conn.execute(
            "INSERT INTO datasets(task,dataset_hash,has_train_test_meta,sidecars_complete,
               catalog_active,encode_job_state,encode_attempts,updated_at_ms)
             VALUES('binary','fresh',1,0,0,NULL,0,1)",
            [],
        )
        .unwrap();
        let mut log = Vec::new();
        unpark_exhausted_if_starved(&conn, &mut log).unwrap();
        let state = |h: &str| -> Option<String> {
            conn.query_row(
                "SELECT encode_job_state FROM datasets WHERE dataset_hash=?",
                [h],
                |r| r.get::<_, Option<String>>(0),
            )
            .unwrap()
        };
        assert_eq!(state("parked").as_deref(), Some("exhausted"));
        assert!(log.is_empty());
        // Once the healthy backlog drains, parked rows come back.
        conn.execute("DELETE FROM datasets WHERE dataset_hash='fresh'", [])
            .unwrap();
        unpark_exhausted_if_starved(&conn, &mut log).unwrap();
        assert_eq!(
            state("parked"),
            None,
            "exhausted rows must be revived when starved"
        );
        assert_eq!(log[0]["datasets"], json!(1));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn state_db_upsert_and_summary() {
        let dir = std::env::temp_dir().join(format!("expdrv-db-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("ops")).unwrap();
        let conn = open_state_db(&dir).unwrap();
        conn.execute(
            "INSERT INTO datasets(task,dataset_hash,has_train_test_meta,sidecars_complete,
               foundation_npy,catalog_active,obs_count,updated_at_ms)
             VALUES('regression','abc',1,1,1,1,5,1)",
            [],
        )
        .unwrap();
        let summary = stage_summary(&conn).unwrap();
        assert_eq!(summary["regression"]["catalog_active"], json!(1));
        assert_eq!(summary["regression"]["observed"], json!(1));
        assert_eq!(summary["binary"]["registry"], json!(0));
        std::fs::remove_dir_all(&dir).ok();
    }
}

//! jope_selfplay_corpus_freezer — advance the training corpus watermark on its
//! own schedule, independent of the promotion tournament.
//!
//! # Why this binary exists
//!
//! `freeze_corpus` is `PHASES[0]` of the promotion tournament
//! (`test_exact_controller.py:32-45`, `jope_selfplay_promotion.rs:51-64`). A new
//! tournament — and therefore a new freeze — is only ever created when the
//! PREVIOUS tournament has reached a terminal phase
//! (`test_exact_controller.py:247`: `if tournament is None:`). So one phase that
//! cannot complete freezes the corpus forever.
//!
//! That is not hypothetical. At the time this was written the live tournament
//! had been parked at `phase_index 9` (`seed_next_strict_wave`) for two days
//! with 848 consecutive failures, and `training/current-corpus.json` was still
//! pinned to watermark 1,805,437 while the observations table held 4,690,000+
//! rows. The trainer saw 2,857 datasets; 18,567 datasets had evidence. The 61%
//! of evidence outside the watermark is not a random 61% either: the frozen CTE
//! orders by `created_at_ms`, so the invisible part is precisely the NEWEST
//! evidence — exactly the datasets the expansion driver has been unlocking.
//!
//! This binary owns the pointer advance and nothing else. It does not touch the
//! tournament, the scheduler, the trainer, or any database.
//!
//! # What one cycle does
//!
//! 1. Read `training/current-corpus.json` (frozen watermark) and
//!    `ops/last-ingest.json` (`exact_rows_total`, the label watermark).
//! 2. If the gap is below `--label-step`, publish status and stop.
//! 3. Otherwise run `scripts/test_exact_train.py build-corpus`, which measures
//!    `COUNT(*)` itself and freezes there.
//! 4. Verify the resulting manifest and the republished pointer, then append a
//!    receipt.
//!
//! # Three invariants that are enforced here rather than assumed
//!
//! * `jope_selfplay_trainer_watchdog.rs:1192` hard-fails its cycle when the
//!   corpus pointer exceeds `exact_rows_total`. MEASURED: `exact_rows_total`
//!   LAGS `COUNT(*)` by ~3.0-3.8k rows and only steps at ingest-batch
//!   boundaries, while ingestion adds ~8.4 rows/s. A freeze takes far longer
//!   than the ~7.5 minutes the label pointer needs to overtake, so freezing at
//!   `COUNT(*)` is correct — but the freezer re-checks afterwards and raises
//!   `label_pointer_behind_corpus` if it has not caught up, because that is the
//!   one way this binary can stall the trainer.
//! * Clamping to `exact_rows_total` instead would be worse than slow: it makes
//!   `watermark >= total` false, so `build_corpus` cannot take the
//!   `covers_every_row` fast path (`test_exact_train.py:434`) and every freeze
//!   pays a 4.7M-row five-column external sort that provably cannot change the
//!   result.
//! * Only ONE freeze may run at a time. Two concurrent builds at the same
//!   watermark race on `os.replace(temporary, snapshot)`; two at different
//!   watermarks each read the whole 21 GB table. An exclusive `flock` on
//!   `ops/corpus-freezer.lock` (non-blocking) makes both impossible for this
//!   binary, and `--exit-on-tournament-freeze` keeps it off the tournament's
//!   toes as well.
//!
//! Never exits non-zero for a failed freeze: the error goes in the status
//! document, where `jope_ops_health_check` and `jope_selfplay_alarms` can see
//! it. An advance mechanism that takes its own unit down is a second thing to
//! monitor.

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::selfplay_runtime::sha256_file;
use serde_json::{json, Map, Value};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const DEFAULT_ROOT: &str = "/home/ubuntu/jope-flywheel/test_exact_gen0";
const STATUS: &str = "ops/corpus-freezer-v1.json";
const EVENTS: &str = "ops/corpus-freezer-events.v1.jsonl";
const LOCK: &str = "ops/corpus-freezer.lock";
const POINTER: &str = "training/current-corpus.json";
const LABEL_POINTER: &str = "ops/last-ingest.json";
const SCHEMA: &str = "JopeSelfplayCorpusFreezer.v1";
const MANIFEST_SCHEMA: &str = "JopeTestOnlyTrainingCorpus.v1";

/// Full consolidation remains above one hour at the current corpus size. A
/// 50,000-label step would make this unit permanently busy, so production
/// stays at 250,000 until measured build time is below one hour.
const DEFAULT_LABEL_STEP: u64 = 250_000;

/// A freeze at the current size writes tens of GB of JSONL and reads the whole
/// observations table. Six hours is the ceiling, not the expectation.
const BUILD_TIMEOUT_SECS: u64 = 6 * 3600;

#[derive(Parser)]
#[command(name = "jope-selfplay-corpus-freezer")]
#[command(about = "Advance training/current-corpus.json independently of the promotion tournament")]
struct Args {
    #[arg(long, default_value = DEFAULT_ROOT)]
    root: PathBuf,
    /// Compatibility override accepted only when it resolves to the exact
    /// script pinned by this executable's runtime bundle.
    #[arg(long)]
    trainer_script: Option<PathBuf>,
    /// Freeze once the label pointer is this far ahead of the frozen watermark.
    #[arg(long, default_value_t = DEFAULT_LABEL_STEP)]
    label_step: u64,
    #[arg(long, default_value = "full_consolidation")]
    corpus_mode: String,
    /// Seconds between cycles. Ignored with --once.
    #[arg(long, default_value_t = 600)]
    interval_seconds: u64,
    /// Complete corpus snapshots to keep, newest first, after a freeze.
    /// 0 disables retention entirely.
    #[arg(long, default_value_t = 3)]
    keep_snapshots: usize,
    /// Never delete a snapshot younger than this, whatever the keep count says.
    #[arg(long, default_value_t = 24)]
    keep_snapshots_min_age_hours: u64,
    /// Run one cycle and exit.
    #[arg(long)]
    once: bool,
    /// Report what a cycle WOULD do and change nothing. Takes no lock.
    #[arg(long)]
    audit: bool,
    /// Run ONLY the snapshot retention pass and exit. Deletion is the one
    /// destructive thing this binary does, so it has to be inspectable on the
    /// live root before it is trusted with a schedule; combine with --dry-run.
    #[arg(long)]
    prune_only: bool,
    /// Retention reports what it would delete and deletes nothing.
    #[arg(long)]
    dry_run: bool,
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis())
        .unwrap_or(0)
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

/// `training/current-corpus.json` may legitimately not exist yet on a fresh
/// root, which is a watermark of zero rather than an error.
fn frozen_watermark(root: &Path) -> Result<u64> {
    let path = root.join(POINTER);
    if !path.is_file() {
        return Ok(0);
    }
    Ok(read_json(&path)?
        .get("watermark")
        .and_then(Value::as_u64)
        .unwrap_or(0))
}

/// The same key `jope_selfplay_trainer_watchdog.rs:766-775` reads, and it must
/// stay the same key: the watchdog compares the corpus pointer against THIS
/// number, so a freezer reading a different one would be enforcing a different
/// invariant from the one that can stop the trainer.
fn label_watermark(root: &Path) -> Result<u64> {
    let path = root.join(LABEL_POINTER);
    let value = read_json(&path)?;
    value
        .get("exact_rows_total")
        .or_else(|| value.get("watermark"))
        .and_then(Value::as_u64)
        .with_context(|| format!("{} lacks exact_rows_total", path.display()))
}

fn atomic_json(path: &Path, value: &Value) -> Result<()> {
    let parent = path.parent().context("status path has no parent")?;
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .context("no name")?;
    let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .mode(0o644)
        .open(&temporary)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn append_event(root: &Path, event: &Value) -> Result<()> {
    let path = root.join(EVENTS);
    fs::create_dir_all(path.parent().context("events path has no parent")?)?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o644)
        .open(&path)?;
    file.write_all(serde_json::to_string(event)?.as_bytes())?;
    file.write_all(b"\n")?;
    file.sync_data()?;
    Ok(())
}

/// Non-blocking on purpose. A blocking lock would queue freezes behind each
/// other and a systemd restart would then find itself waiting on a six-hour
/// build with nothing in the status document to say so.
fn lock_exclusive(root: &Path) -> Result<File> {
    let path = root.join(LOCK);
    fs::create_dir_all(path.parent().context("lock path has no parent")?)?;
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .mode(0o644)
        .open(&path)?;
    let status = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if status != 0 {
        bail!(
            "another corpus freeze holds {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        );
    }
    Ok(file)
}

struct Decision {
    frozen: u64,
    labels: u64,
    gap: u64,
    freeze: bool,
    reason: String,
}

fn decide(root: &Path, label_step: u64) -> Result<Decision> {
    let frozen = frozen_watermark(root)?;
    let labels = label_watermark(root)?;
    let gap = labels.saturating_sub(frozen);
    let freeze = gap >= label_step;
    let reason = if freeze {
        format!("{gap} unfrozen labels >= step {label_step}")
    } else {
        format!("{gap} unfrozen labels < step {label_step}")
    };
    Ok(Decision {
        frozen,
        labels,
        gap,
        freeze,
        reason,
    })
}

#[derive(Clone, Debug)]
struct BundleRuntime {
    bundle_root: PathBuf,
    bundle_sha256: String,
    trainer_script: PathBuf,
    trainer_script_sha256: String,
}

fn resolve_bundle_runtime_from(
    executable: &Path,
    trainer_override: Option<&Path>,
) -> Result<BundleRuntime> {
    let executable = fs::canonicalize(executable)?;
    let bundle_root = executable
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .context("corpus freezer is not running from a runtime bundle")?
        .to_path_buf();
    let bundle_manifest = bundle_root.join("bundle.json");
    let bundle = read_json(&bundle_manifest)?;
    ensure!(
        bundle.get("schema").and_then(Value::as_str) == Some("RuntimeBundle.v3"),
        "corpus freezer bundle manifest schema drift"
    );
    let bundle_sha256 = bundle
        .get("bundle_sha256")
        .and_then(Value::as_str)
        .context("runtime bundle lacks bundle_sha256")?
        .to_owned();
    ensure!(
        bundle_sha256.len() == 64 && bundle_sha256.chars().all(|value| value.is_ascii_hexdigit()),
        "runtime bundle digest is invalid"
    );
    let trainer_script = bundle_root.join("pytorch/training/test_exact_train.py");
    let trainer_script_sha256 = sha256_file(&trainer_script)?;
    ensure!(
        bundle
            .get("files")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .any(|row| {
                row.get("target").and_then(Value::as_str)
                    == Some("pytorch/training/test_exact_train.py")
                    && row.get("sha256").and_then(Value::as_str)
                        == Some(trainer_script_sha256.as_str())
            }),
        "bundled trainer script is absent from or disagrees with bundle.json"
    );
    if let Some(requested) = trainer_override {
        ensure!(
            fs::canonicalize(requested)? == fs::canonicalize(&trainer_script)?,
            "corpus freezer refuses an unbundled trainer script override"
        );
    }
    Ok(BundleRuntime {
        bundle_root,
        bundle_sha256,
        trainer_script,
        trainer_script_sha256,
    })
}

fn resolve_bundle_runtime(args: &Args) -> Result<BundleRuntime> {
    resolve_bundle_runtime_from(&std::env::current_exe()?, args.trainer_script.as_deref())
}

fn stderr_tail(stderr: &str) -> String {
    stderr
        .chars()
        .rev()
        .take(2_000)
        .collect::<String>()
        .chars()
        .rev()
        .collect()
}

/// Spawn the bundled builder, publish its PID before it can become invisible,
/// and refresh the same status document throughout the build.
fn run_build(args: &Args, report: &mut Map<String, Value>) -> Result<(Value, String, Duration)> {
    let runtime = resolve_bundle_runtime(args)?;
    let started = SystemTime::now();
    let started_at_ms = now_ms();
    let logs = args.root.join("ops/corpus-freezer-builds");
    fs::create_dir_all(&logs)?;
    let log_key = format!("{started_at_ms}-{}", &runtime.trainer_script_sha256[..12]);
    let stdout_path = logs.join(format!("{log_key}.stdout"));
    let stderr_path = logs.join(format!("{log_key}.stderr"));
    let stdout_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&stdout_path)?;
    let stderr_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&stderr_path)?;
    let mut child = Command::new("/usr/bin/python3")
        .arg(&runtime.trainer_script)
        .arg("build-corpus")
        .arg("--corpus-mode")
        .arg(&args.corpus_mode)
        .current_dir(
            runtime
                .trainer_script
                .parent()
                .context("bundled trainer script has no parent")?,
        )
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .spawn()
        .context("spawn bundled test_exact_train.py build-corpus")?;
    report.insert("lifecycle".into(), json!("building"));
    report.insert("action".into(), json!("build_corpus"));
    report.insert("build_stage".into(), json!("build-corpus"));
    report.insert("build_started_at_ms".into(), json!(started_at_ms));
    report.insert("build_heartbeat_at_ms".into(), json!(started_at_ms));
    report.insert("child_pid".into(), json!(child.id()));
    report.insert("target_watermark".into(), report["label_watermark"].clone());
    report.insert("script_path".into(), json!(runtime.trainer_script));
    report.insert("script_sha256".into(), json!(runtime.trainer_script_sha256));
    report.insert("bundle_path".into(), json!(runtime.bundle_root));
    report.insert("bundle_sha256".into(), json!(runtime.bundle_sha256));
    report.insert("stdout_path".into(), json!(stdout_path));
    report.insert("stderr_path".into(), json!(stderr_path));
    atomic_json(&args.root.join(STATUS), &Value::Object(report.clone()))?;

    let mut last_heartbeat = SystemTime::now();
    let exit_status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        let elapsed = started.elapsed().unwrap_or_default();
        if elapsed.as_secs() >= BUILD_TIMEOUT_SECS {
            child.kill().context("terminate timed-out corpus builder")?;
            let status = child.wait()?;
            let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
            report.insert("lifecycle".into(), json!("error"));
            report.insert("action".into(), json!("build_timed_out"));
            report.insert("timed_out".into(), json!(true));
            report.insert("timeout_seconds".into(), json!(BUILD_TIMEOUT_SECS));
            report.insert("exit_status".into(), json!(status.to_string()));
            report.insert("stderr_tail".into(), json!(stderr_tail(&stderr)));
            report.insert("build_heartbeat_at_ms".into(), json!(now_ms()));
            atomic_json(&args.root.join(STATUS), &Value::Object(report.clone()))?;
            bail!("build-corpus timed out after {BUILD_TIMEOUT_SECS}s");
        }
        if last_heartbeat.elapsed().unwrap_or_default() >= Duration::from_secs(30) {
            report.insert("build_heartbeat_at_ms".into(), json!(now_ms()));
            report.insert("build_elapsed_seconds".into(), json!(elapsed.as_secs()));
            atomic_json(&args.root.join(STATUS), &Value::Object(report.clone()))?;
            last_heartbeat = SystemTime::now();
        }
        sleep(Duration::from_secs(1));
    };
    let elapsed = started.elapsed().unwrap_or_default();
    let stdout = fs::read(&stdout_path)?;
    let stderr = fs::read_to_string(&stderr_path)?.trim().to_owned();
    if !exit_status.success() {
        report.insert("exit_status".into(), json!(exit_status.to_string()));
        report.insert("stderr_tail".into(), json!(stderr_tail(&stderr)));
        bail!(
            "build-corpus exited {}: {}",
            exit_status,
            stderr_tail(&stderr)
        );
    }
    let manifest: Value =
        serde_json::from_slice(&stdout).context("build-corpus stdout is not a manifest")?;
    Ok((manifest, stderr, elapsed))
}

/// Everything the trainer watchdog will check, checked here first, so a bad
/// freeze is a freezer alarm rather than a silent trainer stop.
fn verify_published(root: &Path, manifest: &Value, before: u64) -> Result<u64> {
    ensure!(
        manifest.get("schema").and_then(Value::as_str) == Some(MANIFEST_SCHEMA),
        "manifest schema is not {MANIFEST_SCHEMA}"
    );
    let watermark = manifest
        .get("source_label_watermark")
        .and_then(Value::as_u64)
        .context("manifest lacks source_label_watermark")?;
    ensure!(
        manifest
            .get("source_family_disjoint")
            .and_then(Value::as_bool)
            == Some(true),
        "corpus is not family-disjoint"
    );
    let published = frozen_watermark(root)?;
    ensure!(
        published >= before,
        "corpus pointer regressed from {before} to {published}"
    );
    ensure!(
        published == watermark,
        "pointer watermark {published} does not match the manifest's {watermark}"
    );
    Ok(watermark)
}

/// Coverage evidence, straight out of the manifest the freeze just wrote.
fn coverage(manifest: &Value) -> Value {
    let sum = |key: &str| -> u64 {
        manifest
            .get(key)
            .and_then(Value::as_object)
            .map(|table| {
                table
                    .values()
                    .filter_map(|v| v.get("datasets").and_then(Value::as_u64))
                    .sum()
            })
            .unwrap_or(0)
    };
    let rows = |key: &str| -> u64 {
        manifest
            .get(key)
            .and_then(Value::as_object)
            .map(|table| {
                table
                    .values()
                    .filter_map(|v| v.get("rows").and_then(Value::as_u64))
                    .sum()
            })
            .unwrap_or(0)
    };
    json!({
        "datasets_total": sum("quota_stats"),
        "pool_rows": rows("quota_stats"),
        "epoch_rows": rows("epoch_draw_stats"),
        "selected_label_count": manifest.get("selected_label_count"),
        "candidate_label_count": manifest.get("candidate_label_count"),
        "watermark_covers_every_row": manifest.get("watermark_covers_every_row"),
        "freeze_guard": manifest.get("freeze_guard"),
        "split_median_divergence": manifest.get("split_median_divergence"),
        "corpus_mode": manifest.get("corpus_mode"),
        "sampler": manifest.pointer("/sampler/id"),
    })
}

/// MEASURED on the live root: one snapshot is 9.6 GB at watermark 1,805,437
/// (`corpus.train.jsonl` alone is 9.9 GB / 257,971 rows = 38,374 bytes per row,
/// because the 2,048-float dataset context block is re-serialised on EVERY row
/// of a dataset), and `training/` already held 46 GB across six snapshots with
/// nothing anywhere pruning them. At the full watermark a snapshot is ~35 GB.
/// Freezing on a 250,000-label step is then ~105 GB/day of new snapshot against
/// 1.4 TB free — so retention is not an optimisation, it is what makes the
/// freezer safe to leave running.
///
/// Deleting is safe against a live trainer: scq stages corpus inputs as
/// HARDLINKS (verified — the live `corpus.train.jsonl` carries a link count of
/// 15), so removing the snapshot's own link cannot break a staged run and frees
/// nothing until the last link goes.
///
/// Four independent protections, any one of which keeps a snapshot:
///   1. it is the directory `training/current-corpus.json` names;
///   2. its watermark appears in the live tournament state (the promotion
///      tournament pins ITS watermark and will re-read that snapshot);
///   3. it is among the newest `keep` by mtime;
///   4. it is younger than `min_age_hours`.
/// Anything without a `corpus-manifest.json` is left alone as well: that is
/// either a live `.tmp-<pid>` build or a quarantined partial, and neither is
/// this function's business.
fn prune_snapshots(args: &Args) -> Value {
    let root = &args.root;
    if args.keep_snapshots == 0 {
        return json!({"pruned": 0, "reason": "retention disabled"});
    }
    let training = root.join("training");
    let pointer_dir = fs::read(root.join(POINTER))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|value| {
            value
                .get("manifest")
                .and_then(Value::as_str)
                .map(|text| PathBuf::from(text))
        })
        .and_then(|path| path.parent().map(Path::to_path_buf));

    // Every watermark the live tournament state mentions, as a string, matched
    // against the zero-padded watermark in the directory name.
    let mut pinned: Vec<String> = Vec::new();
    if let Ok(bytes) = fs::read(root.join("ops/flywheel-state.json")) {
        if let Ok(state) = serde_json::from_slice::<Value>(&bytes) {
            for key in ["/tournament/watermark", "/last_frozen_watermark"] {
                if let Some(value) = state.pointer(key).and_then(Value::as_u64) {
                    pinned.push(format!("snapshot-{value:09}"));
                }
            }
        }
    }
    // A recoverable trainer intent has not necessarily been staged by SCQ yet:
    // its immutable spec may still name the source snapshot. Protect both the
    // crash-safe controller state and its last status document so retention
    // cannot turn a restart into an unrecoverable missing-input failure.
    for relative in [
        "training/task-learning-controller.v2.json",
        "ops/task-learning-controller.v2.json",
    ] {
        if let Ok(bytes) = fs::read(root.join(relative)) {
            if let Ok(state) = serde_json::from_slice::<Value>(&bytes) {
                for key in [
                    "/intent/trainer/corpus_watermark",
                    "/active_intent/trainer/corpus_watermark",
                ] {
                    if let Some(value) = state.pointer(key).and_then(Value::as_u64) {
                        pinned.push(format!("snapshot-{value:09}"));
                    }
                }
            }
        }
    }

    // EVERY complete snapshot, newest first — protections are applied to the
    // tail, not to the ranking. Excluding protected directories from the
    // ranking instead would mean `keep = 3` retained 3 snapshots PLUS the
    // pointer's PLUS every tournament-pinned one, which at ~35 GB each is how a
    // retention pass quietly fails to retain anything.
    let mut snapshots: Vec<(PathBuf, SystemTime, u64)> = Vec::new();
    let Ok(entries) = fs::read_dir(&training) else {
        return json!({"pruned": 0, "reason": "training directory is unreadable"});
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.starts_with("snapshot-") || !path.is_dir() {
            continue;
        }
        if !path.join("corpus-manifest.json").is_file() {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        let bytes = directory_bytes(&path);
        snapshots.push((path, modified, bytes));
    }
    snapshots.sort_by(|a, b| b.1.cmp(&a.1));
    let keep_after = Duration::from_secs(args.keep_snapshots_min_age_hours * 3600);
    let mut removed = Vec::new();
    let mut freed: u64 = 0;
    for (path, modified, bytes) in snapshots.into_iter().skip(args.keep_snapshots) {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_owned();
        if Some(&path) == pointer_dir.as_ref() {
            continue;
        }
        if pinned
            .iter()
            .any(|prefix| name.starts_with(prefix.as_str()))
        {
            continue;
        }
        let age = SystemTime::now()
            .duration_since(modified)
            .unwrap_or_default();
        if age < keep_after {
            continue;
        }
        if args.dry_run {
            freed = freed.saturating_add(bytes);
            removed.push(json!({
                "path": path.to_string_lossy(),
                "bytes": bytes,
                "age_hours": age.as_secs() / 3600,
                "dry_run": true,
            }));
            continue;
        }
        match fs::remove_dir_all(&path) {
            Ok(()) => {
                freed = freed.saturating_add(bytes);
                removed.push(json!({
                    "path": path.to_string_lossy(),
                    "bytes": bytes,
                    "age_hours": age.as_secs() / 3600,
                }));
            }
            Err(error) => removed.push(json!({
                "path": path.to_string_lossy(),
                "error": error.to_string(),
            })),
        }
    }
    json!({
        "pruned": removed.len(),
        "freed_bytes": freed,
        "keep": args.keep_snapshots,
        "min_age_hours": args.keep_snapshots_min_age_hours,
        "removed": removed,
    })
}

/// Shallow: corpus snapshots are flat directories of a few large files.
fn directory_bytes(path: &Path) -> u64 {
    fs::read_dir(path)
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|entry| entry.metadata().ok())
                .filter(|metadata| metadata.is_file())
                .map(|metadata| metadata.len())
                .sum()
        })
        .unwrap_or(0)
}

fn cycle(args: &Args) -> Value {
    let root = &args.root;
    let decision = match decide(root, args.label_step) {
        Ok(value) => value,
        Err(error) => {
            return json!({
                "schema": SCHEMA,
                "generated_at_ms": now_ms() as u64,
                "lifecycle": "error",
                "action": "decide_failed",
                "error": error.to_string(),
            })
        }
    };

    let base = json!({
        "schema": SCHEMA,
        "generated_at_ms": now_ms() as u64,
        "frozen_watermark": decision.frozen,
        "label_watermark": decision.labels,
        "pending_unfrozen_labels": decision.gap,
        "label_step": args.label_step,
        "corpus_mode": args.corpus_mode,
        "reason": decision.reason,
    });
    let mut report = base.as_object().cloned().unwrap_or_default();

    if args.audit {
        report.insert("lifecycle".into(), json!("audit"));
        report.insert(
            "action".into(),
            json!(if decision.freeze {
                "would_freeze"
            } else {
                "would_wait"
            }),
        );
        return Value::Object(report);
    }
    if !decision.freeze {
        report.insert("lifecycle".into(), json!("ok"));
        report.insert("action".into(), json!("collecting_labels"));
        report.insert(
            "next_freeze_at_labels".into(),
            json!(decision.frozen.saturating_add(args.label_step)),
        );
        return Value::Object(report);
    }

    let _guard = match lock_exclusive(root) {
        Ok(file) => file,
        Err(error) => {
            report.insert("lifecycle".into(), json!("ok"));
            report.insert("action".into(), json!("freeze_already_running"));
            report.insert("detail".into(), json!(error.to_string()));
            return Value::Object(report);
        }
    };

    let _ = append_event(
        root,
        &json!({
            "schema": "JopeSelfplayCorpusFreezerEvent.v1",
            "kind": "freeze_started",
            "at_ms": now_ms() as u64,
            "frozen_watermark": decision.frozen,
            "label_watermark": decision.labels,
            "corpus_mode": args.corpus_mode,
        }),
    );

    match run_build(args, &mut report) {
        Ok((manifest, stderr, elapsed)) => {
            match verify_published(root, &manifest, decision.frozen) {
                Ok(watermark) => {
                    let labels_after = label_watermark(root).unwrap_or(0);
                    report.insert("lifecycle".into(), json!("ok"));
                    report.insert("action".into(), json!("froze"));
                    report.insert("new_watermark".into(), json!(watermark));
                    report.insert(
                        "advanced_by".into(),
                        json!(watermark.saturating_sub(decision.frozen)),
                    );
                    report.insert("build_seconds".into(), json!(elapsed.as_secs()));
                    report.insert("coverage".into(), coverage(&manifest));
                    report.insert("label_watermark_after".into(), json!(labels_after));
                    // The one way this binary can stall the trainer: see the
                    // module docs and jope_selfplay_trainer_watchdog.rs:1192.
                    if labels_after < watermark {
                        report.insert("alarm".into(), json!("label_pointer_behind_corpus"));
                        report.insert(
                            "detail".into(),
                            json!(format!(
                                "exact_rows_total {labels_after} < frozen watermark \
                                 {watermark}; the trainer watchdog will fail its cycle \
                                 until ingestion catches up"
                            )),
                        );
                    }
                    let _ = append_event(
                        root,
                        &json!({
                            "schema": "JopeSelfplayCorpusFreezerEvent.v1",
                            "kind": "freeze_succeeded",
                            "at_ms": now_ms() as u64,
                            "previous_watermark": decision.frozen,
                            "watermark": watermark,
                            "build_seconds": elapsed.as_secs(),
                            "coverage": coverage(&manifest),
                            "label_watermark_after": labels_after,
                        }),
                    );
                    if !stderr.is_empty() {
                        report.insert("build_stderr".into(), json!(stderr));
                    }
                    // Only ever after a VERIFIED freeze: retention that ran on a
                    // failed cycle could delete the corpus the trainer is still
                    // reading and leave nothing to replace it.
                    report.insert("retention".into(), prune_snapshots(args));
                    report.insert("packed_retention".into(), prune_packed(args));
                }
                Err(error) => {
                    report.insert("lifecycle".into(), json!("error"));
                    report.insert("action".into(), json!("verify_failed"));
                    report.insert("error".into(), json!(error.to_string()));
                    let _ = append_event(
                        root,
                        &json!({
                            "schema": "JopeSelfplayCorpusFreezerEvent.v1",
                            "kind": "verify_failed",
                            "at_ms": now_ms() as u64,
                            "error": error.to_string(),
                        }),
                    );
                }
            }
        }
        Err(error) => {
            report.insert("lifecycle".into(), json!("error"));
            report.insert("action".into(), json!("build_failed"));
            report.insert("error".into(), json!(error.to_string()));
            let _ = append_event(
                root,
                &json!({
                    "schema": "JopeSelfplayCorpusFreezerEvent.v1",
                    "kind": "freeze_failed",
                    "at_ms": now_ms() as u64,
                    "error": error.to_string(),
                }),
            );
        }
    }
    Value::Object(report)
}

/// `jope_ops_health_check::judge` (src/bin/jope_ops_health_check.rs:129-142)
/// only calls a live `ops/*.json` unhealthy when its schema ends in `Error.v1`
/// or it carries `degraded: true`. This document deliberately keeps ONE schema
/// in both states — a consumer with a strict struct must not break because a
/// freeze failed — so `degraded` is how it tells the truth to the contract
/// checker. A freezer that cannot freeze is the starvation this binary exists
/// to end, and it should fail its unit rather than look healthy.
fn mark_degraded(report: &mut Value) {
    let degraded = report.get("lifecycle").and_then(Value::as_str) == Some("error")
        || report.get("alarm").is_some();
    if let Some(object) = report.as_object_mut() {
        object.insert("degraded".into(), json!(degraded));
    }
}

/// Packed-corpus retention, same shape as `prune_snapshots` for the same
/// reason: `training/packed/<sha>/` dirs are derived artifacts (each is the
/// packed form of a snapshot, ~1.5-2 GB) and nothing pruned them — by
/// 2026-08-01 there were 38 dirs / 73 GB on a root at 86%. Protections:
///   1. the sha named by `training/current-corpus.json` `packed_manifest`;
///   2. any sha the live tournament state mentions (flywheel-state.json and
///      the task-learning controller docs are scanned as text — a serialized
///      pointer is a pin wherever it appears, exactly like snapshot pins);
///   3. the newest `keep` by mtime;
///   4. younger than `min_age_hours`.
/// Dot-prefixed `.…tmp-<pid>` dirs are crashed pack builds (the pack path
/// renames on success); they are removed once older than `min_age_hours`.
fn prune_packed(args: &Args) -> Value {
    let root = &args.root;
    if args.keep_snapshots == 0 {
        return json!({"pruned": 0, "reason": "retention disabled"});
    }
    let packed_root = root.join("training/packed");
    let mut pinned: Vec<String> = Vec::new();
    if let Ok(bytes) = fs::read(root.join(POINTER)) {
        if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
            if let Some(path) = value.get("packed_manifest").and_then(Value::as_str) {
                if let Some(sha) = path
                    .split("packed/")
                    .nth(1)
                    .and_then(|rest| rest.split('/').next())
                {
                    pinned.push(sha.to_owned());
                }
            }
        }
    }
    for relative in [
        "ops/flywheel-state.json",
        "training/task-learning-controller.v2.json",
        "ops/task-learning-controller.v2.json",
    ] {
        if let Ok(text) = fs::read_to_string(root.join(relative)) {
            for chunk in text.split("packed/").skip(1) {
                let sha: String = chunk
                    .chars()
                    .take_while(|c| c.is_ascii_hexdigit())
                    .collect();
                if sha.len() >= 8 {
                    pinned.push(sha);
                }
            }
        }
    }

    let Ok(entries) = fs::read_dir(&packed_root) else {
        return json!({"pruned": 0, "reason": "packed directory is unreadable"});
    };
    let keep_after = Duration::from_secs(args.keep_snapshots_min_age_hours * 3600);
    let mut dirs: Vec<(PathBuf, SystemTime, u64, bool)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !path.is_dir() {
            continue;
        }
        let orphan_tmp = name.starts_with('.') && name.contains(".tmp-");
        if !orphan_tmp && !name.chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        let bytes = directory_bytes(&path);
        dirs.push((path, modified, bytes, orphan_tmp));
    }
    dirs.sort_by(|a, b| b.1.cmp(&a.1));
    let mut removed = Vec::new();
    let mut freed: u64 = 0;
    let mut rank = 0usize;
    for (path, modified, bytes, orphan_tmp) in dirs {
        if !orphan_tmp {
            rank += 1;
            if rank <= args.keep_snapshots {
                continue;
            }
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_owned();
        if !orphan_tmp && pinned.iter().any(|sha| name.starts_with(sha.as_str())) {
            continue;
        }
        let age = SystemTime::now()
            .duration_since(modified)
            .unwrap_or_default();
        if age < keep_after {
            continue;
        }
        if args.dry_run {
            freed = freed.saturating_add(bytes);
            removed.push(json!({
                "path": path.to_string_lossy(),
                "bytes": bytes,
                "age_hours": age.as_secs() / 3600,
                "orphan_tmp": orphan_tmp,
                "dry_run": true,
            }));
            continue;
        }
        match fs::remove_dir_all(&path) {
            Ok(()) => {
                freed = freed.saturating_add(bytes);
                removed.push(json!({
                    "path": path.to_string_lossy(),
                    "bytes": bytes,
                    "age_hours": age.as_secs() / 3600,
                    "orphan_tmp": orphan_tmp,
                }));
            }
            Err(error) => removed.push(json!({
                "path": path.to_string_lossy(),
                "error": error.to_string(),
            })),
        }
    }
    json!({
        "pruned": removed.len(),
        "freed_bytes": freed,
        "keep": args.keep_snapshots,
        "min_age_hours": args.keep_snapshots_min_age_hours,
        "removed": removed,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.prune_only {
        println!("{}", prune_snapshots(&args));
        println!("{}", prune_packed(&args));
        return Ok(());
    }
    loop {
        let mut report = cycle(&args);
        mark_degraded(&mut report);
        if !args.audit {
            let _ = atomic_json(&args.root.join(STATUS), &report);
        }
        println!("{report}");
        if args.once || args.audit {
            return Ok(());
        }
        sleep(Duration::from_secs(args.interval_seconds.max(30)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn root_with(pointer: Option<u64>, labels: u64) -> TempDir {
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join("ops")).unwrap();
        fs::create_dir_all(dir.path().join("training")).unwrap();
        if let Some(value) = pointer {
            fs::write(
                dir.path().join(POINTER),
                serde_json::to_vec(&json!({"watermark": value})).unwrap(),
            )
            .unwrap();
        }
        fs::write(
            dir.path().join(LABEL_POINTER),
            serde_json::to_vec(&json!({"exact_rows_total": labels})).unwrap(),
        )
        .unwrap();
        dir
    }

    /// The live severance, reproduced: pointer stuck at 1,805,437 while the
    /// label pointer says 4,687,525. The freezer must want to freeze, and the
    /// gap it reports must be the real one.
    #[test]
    fn severed_loop_triggers_a_freeze() {
        let dir = root_with(Some(1_805_437), 4_687_525);
        let decision = decide(dir.path(), DEFAULT_LABEL_STEP).unwrap();
        assert!(decision.freeze);
        assert_eq!(decision.gap, 2_882_088);
        assert_eq!(decision.frozen, 1_805_437);
    }

    #[test]
    fn below_the_step_waits() {
        let dir = root_with(Some(4_600_000), 4_610_000);
        let decision = decide(dir.path(), DEFAULT_LABEL_STEP).unwrap();
        assert!(!decision.freeze);
        assert_eq!(decision.gap, 10_000);
    }

    /// A fresh root has no pointer at all; that is watermark zero, not a crash.
    #[test]
    fn missing_pointer_is_watermark_zero() {
        let dir = root_with(None, 260_000);
        let decision = decide(dir.path(), DEFAULT_LABEL_STEP).unwrap();
        assert_eq!(decision.frozen, 0);
        assert!(decision.freeze);
    }

    /// `exact_rows_total` is the key the trainer watchdog reads
    /// (jope_selfplay_trainer_watchdog.rs:768). Reading a different one would
    /// enforce a different invariant from the one that can stop the trainer.
    #[test]
    fn label_watermark_reads_exact_rows_total() {
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join("ops")).unwrap();
        fs::write(
            dir.path().join(LABEL_POINTER),
            serde_json::to_vec(&json!({"exact_rows_total": 4_691_347, "rows": 2310})).unwrap(),
        )
        .unwrap();
        assert_eq!(label_watermark(dir.path()).unwrap(), 4_691_347);
    }

    /// A second freezer must not start a second 21 GB read.
    #[test]
    fn the_lock_is_exclusive_and_non_blocking() {
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join("ops")).unwrap();
        let first = lock_exclusive(dir.path()).unwrap();
        assert!(lock_exclusive(dir.path()).is_err());
        drop(first);
        assert!(lock_exclusive(dir.path()).is_ok());
    }

    /// The pointer is monotone in `publish_corpus_pointer`, so a manifest whose
    /// watermark is behind the live pointer means the pointer was NOT
    /// republished — verification must catch that rather than report a freeze.
    #[test]
    fn verify_rejects_a_pointer_that_did_not_move() {
        let dir = root_with(Some(1_805_437), 4_687_525);
        let manifest = json!({
            "schema": MANIFEST_SCHEMA,
            "source_label_watermark": 4_690_000u64,
            "source_family_disjoint": true,
        });
        let error = verify_published(dir.path(), &manifest, 1_805_437).unwrap_err();
        assert!(
            error.to_string().contains("does not match the manifest"),
            "{error}"
        );
    }

    #[test]
    fn verify_accepts_a_real_advance() {
        let dir = root_with(Some(4_690_000), 4_695_000);
        let manifest = json!({
            "schema": MANIFEST_SCHEMA,
            "source_label_watermark": 4_690_000u64,
            "source_family_disjoint": true,
        });
        assert_eq!(
            verify_published(dir.path(), &manifest, 1_805_437).unwrap(),
            4_690_000
        );
    }

    /// A corpus that is not family-disjoint is refused by
    /// jope_selfplay_trainer_watchdog.rs:1215, so refuse it here first.
    #[test]
    fn verify_rejects_a_non_disjoint_corpus() {
        let dir = root_with(Some(4_690_000), 4_695_000);
        let manifest = json!({
            "schema": MANIFEST_SCHEMA,
            "source_label_watermark": 4_690_000u64,
            "source_family_disjoint": false,
        });
        assert!(verify_published(dir.path(), &manifest, 1_805_437).is_err());
    }

    /// Coverage is summed off the real manifest shape, per-split.
    // -----------------------------------------------------------------------
    // Snapshot retention
    // -----------------------------------------------------------------------

    fn retention_args(root: &Path, keep: usize, min_age_hours: u64) -> Args {
        Args {
            root: root.to_path_buf(),
            trainer_script: None,
            label_step: DEFAULT_LABEL_STEP,
            corpus_mode: "full_consolidation".into(),
            interval_seconds: 600,
            keep_snapshots: keep,
            keep_snapshots_min_age_hours: min_age_hours,
            once: true,
            audit: false,
            prune_only: false,
            dry_run: false,
        }
    }

    #[test]
    fn freezer_resolves_only_the_bundle_pinned_trainer_script() {
        let directory = TempDir::new().unwrap();
        let bundle = directory.path().join("bundle");
        let executable = bundle.join("runtime/bin/jope_selfplay_corpus_freezer");
        let trainer = bundle.join("pytorch/training/test_exact_train.py");
        fs::create_dir_all(executable.parent().unwrap()).unwrap();
        fs::create_dir_all(trainer.parent().unwrap()).unwrap();
        fs::write(&executable, b"binary").unwrap();
        fs::write(&trainer, b"print('ok')\n").unwrap();
        atomic_json(
            &bundle.join("bundle.json"),
            &json!({
                "schema": "RuntimeBundle.v3",
                "bundle_sha256": "a".repeat(64),
                "files": [{
                    "target": "pytorch/training/test_exact_train.py",
                    "sha256": sha256_file(&trainer).unwrap(),
                }],
            }),
        )
        .unwrap();
        let runtime = resolve_bundle_runtime_from(&executable, None).unwrap();
        assert_eq!(runtime.trainer_script, trainer);

        let mutable = directory.path().join("mutable/test_exact_train.py");
        fs::create_dir_all(mutable.parent().unwrap()).unwrap();
        fs::write(&mutable, b"print('mutable')\n").unwrap();
        assert!(resolve_bundle_runtime_from(&executable, Some(&mutable)).is_err());
        assert!(resolve_bundle_runtime_from(&executable, Some(&trainer)).is_ok());
    }

    /// Complete snapshot directories, oldest first, each aged past any plausible
    /// min-age window.
    fn snapshot_dirs(root: &Path, names: &[&str]) -> Vec<PathBuf> {
        let mut made = Vec::new();
        for (index, name) in names.iter().enumerate() {
            let path = root.join("training").join(name);
            fs::create_dir_all(&path).unwrap();
            fs::write(path.join("corpus-manifest.json"), b"{}").unwrap();
            fs::write(path.join("corpus.train.jsonl"), vec![b'x'; 1024]).unwrap();
            // mtimes ascending with index, all far in the past.
            let when = SystemTime::now() - Duration::from_secs(3600 * (400 - index as u64));
            let times = fs::FileTimes::new().set_modified(when);
            File::options()
                .write(true)
                .open(&path)
                .or_else(|_| File::open(&path))
                .and_then(|handle| handle.set_times(times))
                .ok();
            made.push(path);
        }
        made
    }

    fn write_pointer(root: &Path, snapshot: &Path) {
        fs::create_dir_all(root.join("training")).unwrap();
        fs::write(
            root.join(POINTER),
            serde_json::to_vec(&json!({
                "watermark": 4_691_674u64,
                "manifest": snapshot.join("corpus-manifest.json").to_string_lossy(),
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn retention_keeps_the_pointers_snapshot_however_old_it_is() {
        let dir = TempDir::new().unwrap();
        let made = snapshot_dirs(
            dir.path(),
            &[
                "snapshot-000090066",
                "snapshot-000255045",
                "snapshot-000673032",
                "snapshot-001805437-full_consolidation",
                "snapshot-004691674-full_consolidation-eq32v32c128",
            ],
        );
        // The OLDEST directory is the live pointer: retention must not touch it
        // even though it is far outside the keep window.
        write_pointer(dir.path(), &made[0]);
        let report = prune_snapshots(&retention_args(dir.path(), 2, 0));
        assert!(made[0].is_dir(), "the pointer's snapshot was deleted");
        // 5 snapshots, newest 2 kept by rank, the oldest survives as the
        // pointer's — so exactly the two in between go.
        assert_eq!(report["pruned"], json!(2));
        assert!(!made[1].is_dir());
        assert!(!made[2].is_dir());
        assert!(made[3].is_dir());
        assert!(made[4].is_dir());
    }

    /// The ranking counts protected directories, so `keep` is a real ceiling.
    /// Ranking only the unprotected ones would retain `keep` PLUS the pointer
    /// PLUS every pinned watermark — at ~35 GB each, a retention pass that
    /// retains everything.
    #[test]
    fn protected_snapshots_still_consume_the_keep_budget() {
        let dir = TempDir::new().unwrap();
        let made = snapshot_dirs(
            dir.path(),
            &[
                "snapshot-000090066",
                "snapshot-000255045",
                "snapshot-001805437-full_consolidation",
                "snapshot-004691674-full_consolidation",
            ],
        );
        write_pointer(dir.path(), &made[3]);
        let report = prune_snapshots(&retention_args(dir.path(), 2, 0));
        // Newest two (indices 3 and 2) are inside the keep window; the two
        // oldest are deleted even though one of them was the second-newest
        // unprotected directory.
        assert_eq!(report["pruned"], json!(2));
        assert!(made[3].is_dir() && made[2].is_dir());
        assert!(!made[1].is_dir() && !made[0].is_dir());
    }

    /// The promotion tournament pins ITS watermark and re-reads that snapshot,
    /// so a parked tournament's corpus must survive retention.
    #[test]
    fn retention_keeps_a_snapshot_the_live_tournament_still_pins() {
        let dir = TempDir::new().unwrap();
        let made = snapshot_dirs(
            dir.path(),
            &[
                "snapshot-001805437-full_consolidation",
                "snapshot-002000000-full_consolidation",
                "snapshot-003000000-full_consolidation",
                "snapshot-004691674-full_consolidation",
            ],
        );
        write_pointer(dir.path(), &made[3]);
        fs::create_dir_all(dir.path().join("ops")).unwrap();
        fs::write(
            dir.path().join("ops/flywheel-state.json"),
            serde_json::to_vec(&json!({
                "tournament": {"watermark": 1_805_437u64, "phase_index": 9},
                "last_frozen_watermark": 1_805_437u64,
            }))
            .unwrap(),
        )
        .unwrap();
        prune_snapshots(&retention_args(dir.path(), 1, 0));
        assert!(
            made[0].is_dir(),
            "the parked tournament's corpus was deleted"
        );
        assert!(!made[1].is_dir());
    }

    #[test]
    fn retention_keeps_a_snapshot_a_recoverable_trainer_intent_pins() {
        let dir = TempDir::new().unwrap();
        let made = snapshot_dirs(
            dir.path(),
            &[
                "snapshot-001805437-full_consolidation",
                "snapshot-003000000-full_consolidation",
                "snapshot-004691674-full_consolidation",
            ],
        );
        write_pointer(dir.path(), &made[2]);
        fs::write(
            dir.path().join("training/task-learning-controller.v2.json"),
            serde_json::to_vec(&json!({
                "intent": {"trainer": {"corpus_watermark": 1_805_437u64}},
            }))
            .unwrap(),
        )
        .unwrap();
        prune_snapshots(&retention_args(dir.path(), 1, 0));
        assert!(
            made[0].is_dir(),
            "the recoverable trainer intent's corpus was deleted"
        );
        assert!(!made[1].is_dir());
    }

    #[test]
    fn retention_never_deletes_a_young_snapshot() {
        let dir = TempDir::new().unwrap();
        let made = snapshot_dirs(dir.path(), &["snapshot-000090066", "snapshot-000255045"]);
        write_pointer(dir.path(), &made[1]);
        // Both are ~400 hours old in the fixture; a 1000-hour floor keeps them.
        let report = prune_snapshots(&retention_args(dir.path(), 0_usize.max(0), 1000));
        assert_eq!(report["pruned"], json!(0));
        assert!(made[0].is_dir());
    }

    /// A `.tmp-<pid>` build in flight and a quarantined partial both lack a
    /// manifest. Neither is retention's business.
    #[test]
    fn retention_ignores_incomplete_directories() {
        let dir = TempDir::new().unwrap();
        let made = snapshot_dirs(dir.path(), &["snapshot-000090066"]);
        write_pointer(dir.path(), &made[0]);
        let partial = dir.path().join("training/snapshot-004691674-partial");
        fs::create_dir_all(&partial).unwrap();
        fs::write(partial.join("corpus.train.jsonl"), b"x").unwrap();
        let report = prune_snapshots(&retention_args(dir.path(), 0, 0));
        assert_eq!(report["pruned"], json!(0));
        assert!(partial.is_dir());
    }

    /// --dry-run must report the identical decision and touch nothing.
    #[test]
    fn a_dry_run_reports_the_same_deletions_without_making_them() {
        let dir = TempDir::new().unwrap();
        let made = snapshot_dirs(
            dir.path(),
            &[
                "snapshot-000090066",
                "snapshot-000255045",
                "snapshot-004691674-full_consolidation",
            ],
        );
        write_pointer(dir.path(), &made[2]);
        let mut args = retention_args(dir.path(), 1, 0);
        args.dry_run = true;
        let dry = prune_snapshots(&args);
        assert_eq!(dry["pruned"], json!(2));
        assert!(made[0].is_dir() && made[1].is_dir());

        args.dry_run = false;
        let wet = prune_snapshots(&args);
        assert_eq!(wet["pruned"], dry["pruned"]);
        assert_eq!(wet["freed_bytes"], dry["freed_bytes"]);
        assert!(!made[0].is_dir() && !made[1].is_dir());
        assert!(made[2].is_dir());
    }

    #[test]
    fn retention_can_be_disabled_outright() {
        let dir = TempDir::new().unwrap();
        let made = snapshot_dirs(dir.path(), &["snapshot-000090066", "snapshot-000255045"]);
        write_pointer(dir.path(), &made[1]);
        let mut args = retention_args(dir.path(), 0, 0);
        args.keep_snapshots = 0;
        let report = prune_snapshots(&args);
        assert_eq!(report["pruned"], json!(0));
        assert!(made[0].is_dir());
    }

    /// `jope_ops_health_check` reads `degraded`, not `lifecycle`.
    #[test]
    fn an_error_cycle_is_marked_degraded_for_the_contract_checker() {
        let mut ok = json!({"lifecycle": "ok", "action": "collecting_labels"});
        mark_degraded(&mut ok);
        assert_eq!(ok["degraded"], json!(false));

        let mut failed = json!({"lifecycle": "error", "action": "build_failed"});
        mark_degraded(&mut failed);
        assert_eq!(failed["degraded"], json!(true));

        // A successful freeze that left the label pointer behind is still a
        // trainer stop waiting to happen.
        let mut behind = json!({
            "lifecycle": "ok",
            "action": "froze",
            "alarm": "label_pointer_behind_corpus",
        });
        mark_degraded(&mut behind);
        assert_eq!(behind["degraded"], json!(true));
    }

    #[test]
    fn coverage_sums_the_live_manifest_shape() {
        let manifest = json!({
            "quota_stats": {
                "regression/train": {"datasets": 2072, "rows": 205935},
                "regression/validation": {"datasets": 243, "rows": 7776},
                "binary/train": {"datasets": 499, "rows": 52036},
                "binary/validation": {"datasets": 43, "rows": 1376},
            },
            "epoch_draw_stats": {
                "regression/train": {"datasets": 2072, "rows": 66304},
                "regression/validation": {"datasets": 243, "rows": 7776},
                "binary/train": {"datasets": 499, "rows": 15968},
                "binary/validation": {"datasets": 43, "rows": 1376},
            },
        });
        let value = coverage(&manifest);
        assert_eq!(value["datasets_total"], json!(2857));
        assert_eq!(value["pool_rows"], json!(267123));
        assert_eq!(value["epoch_rows"], json!(91424));
    }
}

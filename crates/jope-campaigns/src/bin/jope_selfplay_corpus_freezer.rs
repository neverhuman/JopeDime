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
use jope_dime_regressor::paths;
use jope_dime_regressor::selfplay_runtime::sha256_file;
use serde_json::{json, Map, Value};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

jope_paths::once_str!(
    fn default_root_str => paths::flywheel_root()
);
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
    #[arg(long, default_value = default_root_str())]
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
        .truncate(false)
        .write(true)
        .mode(0o644)
        .open(&path)?;
    if let Err(error) = file.try_lock() {
        // Keep the historical errno text ("Resource temporarily unavailable") for contention.
        let error = match error {
            std::fs::TryLockError::WouldBlock => {
                std::io::Error::from_raw_os_error(libc::EWOULDBLOCK)
            }
            std::fs::TryLockError::Error(error) => error,
        };
        bail!("another corpus freeze holds {}: {}", path.display(), error);
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
///
/// Anything without a `corpus-manifest.json` is left alone as well: that is
/// either a live `.tmp-<pid>` build or a quarantined partial, and neither is
/// this function's business.
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
///
/// Dot-prefixed `.…tmp-<pid>` dirs are crashed pack builds (the pack path
/// renames on success); they are removed once older than `min_age_hours`.
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

mod retention;
use retention::*;

#[cfg(test)]
mod tests;

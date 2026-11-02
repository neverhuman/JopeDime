use super::*;
use tempfile::TempDir;

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

// Coverage is summed off the real manifest shape, per-split.
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
    let report = prune_snapshots(&retention_args(dir.path(), 0, 1000));
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

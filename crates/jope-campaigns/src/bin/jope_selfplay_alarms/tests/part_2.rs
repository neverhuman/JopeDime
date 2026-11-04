#[test]
fn read_tail_drops_the_partial_leading_line() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("lines.jsonl");
    std::fs::write(&path, "aaaa\nbbbb\ncccc\n").unwrap();
    assert_eq!(read_tail(&path, 1024).unwrap(), "aaaa\nbbbb\ncccc\n");
    // Cap lands mid-line: the truncated head must be discarded, not parsed.
    assert_eq!(read_tail(&path, 8).unwrap(), "cccc\n");
}

#[test]
fn final_conditional_advantage_uses_the_last_row_that_has_the_key() {
    let temp = TempDir::new().unwrap();
    let run_dir = run_dir_with_metrics(
        &temp,
        "run-last-key",
        &[
            "{\"epoch\":1,\"val_conditional_advantage\":0.05}",
            "{\"epoch\":2,\"val_conditional_advantage\":-0.06}",
            // Later rows without the key (non-validation epochs) are skipped.
            "{\"epoch\":3,\"train_loss\":0.1}",
            "not json at all",
        ],
    );
    let (advantage, age) = final_conditional_advantage(&run_dir).unwrap();
    assert!((advantage - -0.06).abs() < 1e-12);
    assert!(age < 60);
}

#[test]
fn missing_metrics_cannot_be_judged() {
    let temp = TempDir::new().unwrap();
    let run_dir = temp.path().join("no-such-run");
    assert!(final_conditional_advantage(&run_dir).is_none());
    let mut alarms = Alarms { rows: Vec::new() };
    check_completed_conditionality("jope-selfplay-train-x", "completed", &run_dir, &mut alarms);
    assert!(alarms.rows.is_empty());
}

#[test]
fn warm_start_guard_deficiency_becomes_a_critical_alarm() {
    let temp = TempDir::new().unwrap();
    let ops = temp.path().join("ops");
    std::fs::create_dir_all(&ops).unwrap();
    let guard = ops.join("trainer-warm-start-guard.v1.json");
    let mut alarms = Alarms { rows: Vec::new() };

    // Absent guard file (pre-deploy) is silent.
    check_warm_start_guard(temp.path(), &mut alarms);
    assert!(alarms.rows.is_empty());

    // Healthy verdict is silent.
    std::fs::write(
        &guard,
        json!({"schema": "JopeTrainerWarmStartGuard.v1", "deficient": false}).to_string(),
    )
    .unwrap();
    check_warm_start_guard(temp.path(), &mut alarms);
    assert!(alarms.rows.is_empty());

    std::fs::write(
        &guard,
        json!({
            "schema": "JopeTrainerWarmStartGuard.v1",
            "deficient": true,
            "observed": "1 candidate(s), none full",
            "threshold": ">=152 tensors",
        })
        .to_string(),
    )
    .unwrap();
    check_warm_start_guard(temp.path(), &mut alarms);
    assert_eq!(alarms.rows.len(), 1);
    assert_eq!(alarms.rows[0]["alarm"], "trainer_warm_start_deficient");
    assert_eq!(alarms.rows[0]["severity"], "critical");
    assert_eq!(alarms.rows[0]["observed"], "1 candidate(s), none full");
}

#[test]
fn completed_non_conditional_run_raises_a_critical_alarm() {
    let temp = TempDir::new().unwrap();
    let run_dir = run_dir_with_metrics(
        &temp,
        "run-negative",
        &["{\"epoch\":9,\"val_conditional_advantage\":-0.06}"],
    );
    let mut alarms = Alarms { rows: Vec::new() };
    check_completed_conditionality(
        "jope-selfplay-train-a-g3",
        "completed",
        &run_dir,
        &mut alarms,
    );
    assert_eq!(
        alarm_names(&alarms),
        vec!["policy_not_conditional:jope-selfplay-train-a-g3".to_owned()]
    );
    let row = &alarms.rows[0];
    assert_eq!(row["severity"], "critical");
    assert!(row["observed"]
        .as_str()
        .unwrap()
        .contains("policy is not conditional"));
    assert!(row["observed"].as_str().unwrap().contains("-0.0600"));

    // Exactly zero advantage is still not conditional.
    let zero = run_dir_with_metrics(&temp, "run-zero", &["{\"val_conditional_advantage\":0.0}"]);
    let mut alarms = Alarms { rows: Vec::new() };
    check_completed_conditionality("jte-zero", "completed", &zero, &mut alarms);
    assert_eq!(alarms.rows.len(), 1);
}

#[test]
fn conditional_runs_and_live_runs_stay_quiet() {
    let temp = TempDir::new().unwrap();
    let healthy = run_dir_with_metrics(
        &temp,
        "run-healthy",
        &["{\"val_conditional_advantage\":0.031}"],
    );
    let mut alarms = Alarms { rows: Vec::new() };
    check_completed_conditionality("jte-healthy", "completed", &healthy, &mut alarms);
    assert!(alarms.rows.is_empty());
    // A run that is still training is judged by the stall check, not this one.
    let running =
        run_dir_with_metrics(&temp, "run-live", &["{\"val_conditional_advantage\":-0.2}"]);
    check_completed_conditionality("jte-running", "running", &running, &mut alarms);
    assert!(alarms.rows.is_empty());
}

/// Lay out `<runs>/<run_id>/attempts/0/outputs/training/watermark-.../train_metrics.jsonl`.
fn trainer_metrics(runs: &Path, run_id: &str, body: &str) -> PathBuf {
    let outputs = runs
        .join(run_id)
        .join("attempts/0/outputs/training/watermark-001805437");
    std::fs::create_dir_all(&outputs).unwrap();
    let path = outputs.join(th::METRICS_FILE);
    std::fs::write(&path, body).unwrap();
    path
}

fn archive_now(root: &Path, runs: &Path, run_id: &str) {
    let archive = th::archive_root(root);
    for found in th::scan_run_dir(runs, run_id) {
        th::ingest_source(
            &archive,
            &found.key,
            &th::FileSource(found.path.clone()),
            &found.path.to_string_lossy(),
            "xbabe2",
        )
        .unwrap();
    }
    th::rebuild_index(root, &archive).unwrap();
}

#[test]
fn a_fully_archived_run_is_silent() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("gen0");
    let runs = temp.path().join("runs");
    let run_id = "jope-selfplay-train-001805437-a-g9";
    trainer_metrics(&runs, run_id, "{\"epoch\":1,\"val_loss\":3.4}\n");
    archive_now(&root, &runs, run_id);

    let mut alarms = Alarms { rows: Vec::new() };
    check_training_history(&root, &runs, &mut alarms);
    assert!(alarms.rows.is_empty(), "{:?}", alarm_names(&alarms));
}

#[test]
fn a_missing_archive_is_critical() {
    let temp = TempDir::new().unwrap();
    let mut alarms = Alarms { rows: Vec::new() };
    check_training_history(temp.path(), &temp.path().join("runs"), &mut alarms);
    assert_eq!(alarm_names(&alarms), vec!["training_history_missing"]);
    assert_eq!(alarms.rows[0]["severity"], "critical");
}

#[test]
fn an_idle_run_with_unarchived_rows_is_critical() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("gen0");
    let runs = temp.path().join("runs");
    let run_id = "jope-selfplay-train-001805437-c";
    let path = trainer_metrics(&runs, run_id, "{\"epoch\":1,\"val_loss\":9.6}\n");
    archive_now(&root, &runs, run_id);
    // Rows appear that capture never picked up, and the trainer has since
    // gone quiet: this run is one GC pass from losing them.
    std::fs::write(
        &path,
        "{\"epoch\":1,\"val_loss\":9.6}\n{\"epoch\":2,\"val_loss\":9.7}\n",
    )
    .unwrap();
    let old = SystemTime::now() - std::time::Duration::from_secs(HISTORY_SOURCE_FRESH_SECS + 60);
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(old))
        .unwrap();

    let mut alarms = Alarms { rows: Vec::new() };
    check_training_history(&root, &runs, &mut alarms);
    let mine = alarms_for(&alarms, run_id);
    assert_eq!(mine.len(), 1, "{mine:?}");
    assert_eq!(
        mine[0]["alarm"],
        format!("training_history_unarchived:{run_id}")
    );
    assert_eq!(mine[0]["severity"], "critical");
    // The gap spans a newline, so it is complete rows, not a truncated tail.
    assert!(mine[0]["observed"]
        .as_str()
        .unwrap()
        .contains("complete_rows_missing"));
}

/// The truncated-tail policy, from the alarm's side: a trailing incomplete
/// row that has been recorded must NOT read as data loss, because a
/// permanent critical here blocks the expansion driver and buries the real
/// `training_history_unarchived` alarms under noise.
#[test]
fn a_recorded_truncated_tail_is_informational_not_critical() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("gen0");
    let runs = temp.path().join("runs");
    let run_id = "jope-selfplay-train-001805437-b-g6";
    let path = trainer_metrics(
        &runs,
        run_id,
        "{\"epoch\":1,\"val_loss\":9.6}\n{\"epoch\":2,\"val_loss\":9.5}\n{\"epoch\":3,\"val_",
    );
    archive_now(&root, &runs, run_id);
    let old = SystemTime::now() - std::time::Duration::from_secs(th::TAIL_QUIESCENT_SECS + 60);
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(old))
        .unwrap();

    // Before the hole is recorded: visible, but a warning, not critical.
    let mut alarms = Alarms { rows: Vec::new() };
    check_training_history(&root, &runs, &mut alarms);
    let mine = alarms_for(&alarms, run_id);
    assert_eq!(mine.len(), 1, "{mine:?}");
    assert_eq!(
        mine[0]["alarm"],
        format!("training_history_truncated_tail_pending:{run_id}")
    );
    assert_eq!(mine[0]["severity"], "warning");

    // Once capture records it, it drops to informational and stays there.
    let archive = th::archive_root(&root);
    let found = th::scan_run_dir(&runs, run_id).remove(0);
    let held = th::archived_bytes(&archive, &found.key);
    let th::Residual::PartialRow(fragment) =
        th::classify_residual(&th::FileSource(found.path.clone()), held, found.len).unwrap()
    else {
        panic!("expected a partial row");
    };
    th::record_truncation(
        &archive,
        &found.key,
        "xbabe2",
        &found.path.to_string_lossy(),
        &fragment,
        found.len,
        0,
    )
    .unwrap();

    let mut alarms = Alarms { rows: Vec::new() };
    check_training_history(&root, &runs, &mut alarms);
    let mine = alarms_for(&alarms, run_id);
    assert_eq!(mine.len(), 1, "{mine:?}");
    assert_eq!(
        mine[0]["alarm"],
        format!("training_history_truncated_tail:{run_id}")
    );
    assert_eq!(mine[0]["severity"], "info");
    assert!(
        !alarm_names(&alarms).contains(&format!("training_history_unarchived:{run_id}")),
        "a recorded truncated tail must not also fire the critical"
    );
}

#[test]
fn an_unreachable_capture_node_is_a_warning() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("gen0");
    let runs = temp.path().join("runs");
    let run_id = "jope-selfplay-train-001805437-a-g8";
    trainer_metrics(&runs, run_id, "{\"epoch\":1,\"val_loss\":3.4}\n");
    archive_now(&root, &runs, run_id);
    let archive = th::archive_root(&root);
    let mut nodes = std::collections::BTreeMap::new();
    nodes.insert(
        "xbabe3".to_string(),
        th::NodeStatus {
            node: "xbabe3".into(),
            last_attempt_ms: th::now_ms(),
            last_success_ms: 0,
            consecutive_failures: 4,
            last_error: "ssh exit Some(255)".into(),
            segments_seen: 0,
            reachable: false,
        },
    );
    th::save_nodes(&archive, &nodes).unwrap();

    let mut alarms = Alarms { rows: Vec::new() };
    check_training_history(&root, &runs, &mut alarms);
    let row = alarms
        .rows
        .iter()
        .find(|row| row["alarm"] == "training_history_node_unreachable:xbabe3")
        .expect("node alarm");
    assert_eq!(row["severity"], "warning");
    assert!(row["observed"].as_str().unwrap().contains("never swept"));
}

#[test]
fn a_row_being_written_right_now_is_not_an_alarm() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join("gen0");
    let runs = temp.path().join("runs");
    let run_id = "jope-selfplay-train-001805437-a-g9";
    let path = trainer_metrics(&runs, run_id, "{\"epoch\":1,\"val_loss\":3.4}\n");
    archive_now(&root, &runs, run_id);
    // Partial row in flight: source is ahead, but the archive just advanced,
    // so this is normal steady state and must stay quiet.
    std::fs::write(
        &path,
        "{\"epoch\":1,\"val_loss\":3.4}\n{\"epoch\":2,\"val_l",
    )
    .unwrap();

    let mut alarms = Alarms { rows: Vec::new() };
    check_training_history(&root, &runs, &mut alarms);
    assert!(alarms.rows.is_empty(), "{:?}", alarm_names(&alarms));
}

// -----------------------------------------------------------------------
// Proposal deficiency / head collapse
//
// Every fixture below is a real reading off the live root on 2026-07-28,
// not an invented shape.
// -----------------------------------------------------------------------

/// Publishes a proposal set and returns the root. `summary` is written both
/// where the pointer names it and beside the (empty) archived jsonl, so the
/// fallback path can be exercised by deleting one of them.
fn proposal_root(temp: &TempDir, summary: Value) -> PathBuf {
    let root = temp.path().to_path_buf();
    std::fs::create_dir_all(root.join("proposals/by-checkpoint")).unwrap();
    let archive = root.join("proposals/by-checkpoint/set.jsonl");
    std::fs::write(&archive, "").unwrap();
    let named = root.join("proposals/named.summary.json");
    std::fs::write(&named, serde_json::to_vec(&summary).unwrap()).unwrap();
    std::fs::write(
        root.join("proposals/by-checkpoint/set.summary.json"),
        serde_json::to_vec(&summary).unwrap(),
    )
    .unwrap();
    std::fs::write(
        root.join("proposals/current.json"),
        serde_json::to_vec(&json!({
            "checkpoint_sha256": "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592",
            "proposals": archive.to_string_lossy(),
            "summary": named.to_string_lossy(),
        }))
        .unwrap(),
    )
    .unwrap();
    root
}

fn proposal_scan(root: &Path) -> Alarms {
    let mut alarms = Alarms { rows: Vec::new() };
    check_proposal_deficiency(root, &mut alarms);
    alarms
}

/// Micro-cohort set 3a89b11ac7da, 2026-07-28T23:11Z, checkpoint fbcd5459:
/// 128 contexts, binary 19/64 short with a worst context of 20/64 and 583
/// borrowed rows; regression clean. Head mix measured off the emitted file.
fn live_micro_cohort_summary() -> Value {
    json!({
        "schema": "JopeTestOnlyDlProposalSet.v1",
        "model_checkpoint_sha256": "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592",
        "proposals_per_dataset": 64,
        "deficiency_by_task": {
            "binary": {
                "contexts": 64, "deficient_contexts": 19,
                "deficient_fraction": 19.0 / 64.0, "min_native_unique": 20,
                "mean_native_unique": 3513.0 / 64.0, "backfilled_rows": 583,
                "backfilled_fraction": 583.0 / 4096.0, "unfilled_contexts": 0
            },
            "regression": {
                "contexts": 64, "deficient_contexts": 0,
                "deficient_fraction": 0.0, "min_native_unique": 64,
                "mean_native_unique": 64.0, "backfilled_rows": 0,
                "backfilled_fraction": 0.0, "unfilled_contexts": 0
            }
        },
        "native_head_counts_by_task": {
            "binary": {
                "logit_ridge": 3084, "tab_boost": 275, "extra_trees": 100,
                "gam_boost": 53, "logistic": 1
            },
            "regression": {
                "ridge": 913, "gpr": 900, "histgbm2": 641, "elasticnet": 626,
                "ensemble_ridge_gbm": 325, "ensemble": 218, "histgbm": 200,
                "gbm": 160, "ksmooth": 113
            }
        },
        "proposal_validity_by_task": {"binary": 0.9904186320754716, "regression": 0.9997577519379846}
    })
}

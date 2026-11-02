use super::*;

#[test]
fn journal_idempotency_and_pending() {
    let temp = TempDir::new().unwrap();
    let journal = ActionJournal::open(temp.path()).unwrap();
    let request = json!({"probe": 1});
    // Pending first.
    let outcome = journal.perform("k1", "probe", &request, |_| Err(pending("not yet")));
    assert!(outcome.is_err());
    let status: String = journal
        .db
        .query_row(
            "SELECT status FROM actions WHERE idempotency_key='k1'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(status, "pending");
    // Then succeed.
    let result = journal
        .perform("k1", "probe", &request, |_| Ok(json!({"ok": true})))
        .unwrap();
    assert_eq!(result, json!({"ok": true}));
    // Idempotent short circuit: callback must not run again.
    let result = journal
        .perform("k1", "probe", &request, |_| {
            panic!("must not re-run succeeded action")
        })
        .unwrap();
    assert_eq!(result, json!({"ok": true}));
    let events = fs::read_to_string(temp.path().join("ops/controller-events.jsonl")).unwrap();
    assert_eq!(events.lines().count(), 4); // started,pending,started,succeeded
}

fn fixture_snapshot(labels: i64) -> Value {
    json!({
        "valid_labels": labels,
        "accepted_ancestor": "f".repeat(64),
        "catalog_sha256": "c".repeat(64),
        "catalog_entries": [
            {"task": "regression", "dataset_hash": "d1"},
            {"task": "binary", "dataset_hash": "d2"},
        ],
        "catalog_counts": {"regression": 1, "binary": 1},
        "evaluator_sha256": "e".repeat(64),
    })
}

#[test]
fn cycle_collects_labels_below_threshold() {
    let temp = TempDir::new().unwrap();
    let controller = Controller::open(temp.path()).unwrap();
    let result = controller.cycle(&fixture_snapshot(10)).unwrap();
    assert_eq!(result["phase"], json!("collecting_labels"));
    assert_eq!(result["next_watermark"], json!(LABEL_THRESHOLD));
}

#[test]
fn cycle_progresses_with_file_artifacts_and_gate_fail_skip() {
    let temp = TempDir::new().unwrap();
    let controller = Controller::open(temp.path()).unwrap();
    let snapshot = fixture_snapshot(60_000);
    // Freezes a tournament, then pends on the freeze_corpus file artifact.
    let result = controller.cycle(&snapshot).unwrap();
    assert_eq!(result["phase"], json!("freeze_corpus"));
    assert_eq!(result["pending"], json!(true));
    // Provide file artifacts for every phase up to the failing gate.
    let work = watermark_dir(temp.path(), 60_000);
    fs::create_dir_all(&work).unwrap();
    for phase in &PHASES[..7] {
        fs::write(work.join(format!("{phase}.json")), "{\"ok\": true}").unwrap();
    }
    fs::write(
        work.join("run_checkpoint_gate.json"),
        "{\"pass\": false, \"promotion_tier\": \"rejected\"}",
    )
    .unwrap();
    for _ in 0..8 {
        let outcome = controller.cycle(&snapshot).unwrap();
        assert_eq!(outcome.get("completed"), Some(&json!(true)));
    }
    // After the failed gate the pointer-advance phase must be skipped.
    let state = read_json(&temp.path().join("ops/flywheel-state.json")).unwrap();
    assert!(state["tournament"].is_null());
    assert_eq!(state["last_terminal"], json!("rejected"));
    let completed = read_json(Path::new(
        state["last_tournament"]
            .as_str()
            .expect("completed tournament path"),
    ))
    .unwrap();
    assert_eq!(
        completed.pointer("/artifacts/advance_accepted_pointer/advanced"),
        Some(&json!(false))
    );
}

/// WS-6: the per-task verdict must reach the WRITTEN gate artifact (not just
/// the phase return value), and only the branch that passed on its own
/// evidence may advance `training/accepted-by-task.json`. The global
/// verdict, and the two global lineage pointers, must be untouched.
#[test]
fn run_checkpoint_gate_records_a_per_task_verdict_and_advances_only_the_passing_branch() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    let checkpoint = root.join("candidate.model.pt");
    fs::write(&checkpoint, b"ws6-candidate").unwrap();
    let checkpoint_sha = sha256_file(&checkpoint).unwrap();
    // Binary +0.05 (clears its own +0.002 margin), regression -0.05 (its own
    // reason). Macro stays flat, so the GLOBAL gate must still reject.
    let mut evidence = synthetic_evidence(0.0);
    evidence["candidate"]["binary_pairwise_accuracy"] = json!(0.55);
    evidence["candidate"]["regression_pairwise_accuracy"] = json!(0.45);
    evidence["candidate"]["checkpoint_sha256"] = json!(checkpoint_sha);
    evidence["candidate"]["checkpoint_path"] = json!(checkpoint.to_string_lossy());
    evidence["corpus_watermark"] = json!(60_000);
    let request =
        json!({"watermark": 60_000, "artifacts": {"build_acceptance_evidence": evidence}});

    let out = handle_run_checkpoint_gate(&request, root).unwrap();
    assert_eq!(out["pass"], json!(false), "global gate must still reject");
    assert_eq!(out["promotion_tier"], json!("rejected"));
    assert_eq!(out["tasks_passing"], json!(["binary"]));
    assert_eq!(out["task_verdicts"]["binary"]["pass"], json!(true));
    assert_eq!(out["task_verdicts"]["binary"]["advisory"], json!(false));
    assert_eq!(out["task_verdicts"]["regression"]["pass"], json!(false));
    // Unanchored new tasks: recorded as advisory, failing on their own
    // (absent) evidence, and never gating.
    for task in ["timeseries", "multiclass"] {
        assert_eq!(out["task_verdicts"][task]["advisory"], json!(true));
        assert_eq!(out["task_verdicts"][task]["pass"], json!(false));
        assert_eq!(out["task_verdicts"][task]["pairwise_lift"], Value::Null);
    }

    // The artifact on disk, not just the return value.
    let artifact = read_json(&watermark_dir(root, 60_000).join("checkpoint-gate.json")).unwrap();
    assert_eq!(
        artifact["task_verdict_schema"],
        json!("JopeSelfplayCheckpointGateTaskVerdict.v1")
    );
    assert_eq!(artifact["tasks_passing"], json!(["binary"]));
    assert_eq!(artifact["pass"], json!(false));
    assert_eq!(
        artifact["task_verdicts"]["regression"]["blocking_reasons_owned_by_this_task"],
        json!([
            "regression_pairwise_lift_below_point_two",
            "regression_regressed_more_than_half_point"
        ])
    );

    let pointers = read_json(&task_accepted_pointer(root)).unwrap();
    // The written schema must match what the live readers require, and what the
    // fixture below already writes. This assertion previously expected
    // "AcceptedByTask.v2" — the shadow namespace's constant for a different file —
    // which is exactly the drift that would have broken the trainer and the causal
    // ledger on the first promotion cycle.
    assert_eq!(pointers["schema"], json!("JopeSelfplayAcceptedByTask.v1"));
    assert_eq!(
        pointers["tasks"]["binary"]["checkpoint_sha256"],
        json!(checkpoint_sha)
    );
    assert_eq!(
        pointers["tasks"]["binary"]["source"],
        json!("per_task_gate")
    );
    assert_eq!(pointers["tasks"]["binary"]["generation"], json!(1));
    assert_eq!(
        pointers["tasks"]["binary"]["parent_checkpoint_sha256"],
        json!("b".repeat(64))
    );
    assert!(pointers["tasks"]["binary"]["evidence_sha256"]
        .as_str()
        .is_some_and(|value| value.len() == 64));
    assert!(pointers["tasks"]["binary"]["rollback"].is_object());
    // The failing branch is bootstrapped to the global pointer, never advanced.
    assert_eq!(
        pointers["tasks"]["regression"]["source"],
        json!("global_accepted_bootstrap")
    );
    assert_ne!(
        pointers["tasks"]["regression"]["checkpoint_sha256"],
        json!(checkpoint_sha)
    );
    // Advisory-until-anchor: the new tasks are NOT bootstrapped — presence
    // in this file is the anchor, so only a passing verdict may create it.
    assert!(pointers["tasks"].get("timeseries").is_none());
    assert!(pointers["tasks"].get("multiclass").is_none());
    // ONE global lineage pointer, and this phase does not write it: the
    // freshness epoch is defined by these two files alone.
    assert!(!root.join("training/accepted.json").exists());
    assert!(!root.join("training/learning.json").exists());
}

/// U3 invariant: with no anchor, a new task's verdict — even a regressing
/// one computed from real evidence — must not alter any global decision;
/// once anchored, the same evidence gates exactly like a legacy branch.
#[test]
fn new_task_verdicts_are_advisory_until_anchored() {
    let mut evidence = synthetic_evidence(0.05);
    evidence["baseline"]["timeseries_pairwise_accuracy"] = json!(0.6);
    evidence["candidate"]["timeseries_pairwise_accuracy"] = json!(0.5);

    let advisory = evaluate_gate(&evidence, None).unwrap();
    assert_eq!(advisory["pass"], json!(true));
    assert_eq!(advisory["promotion_tier"], json!("milestone"));
    assert!(!gate_reasons(&advisory, "milestone_reasons")
        .iter()
        .any(|reason| reason.starts_with("timeseries_")));
    let verdict = &advisory["task_verdicts"]["timeseries"];
    assert_eq!(verdict["advisory"], json!(true));
    assert_eq!(verdict["pass"], json!(false));
    assert!(verdict["reasons"]
        .as_array()
        .unwrap()
        .contains(&json!("timeseries_regressed_more_than_half_point")));
    assert_eq!(
        advisory["task_pairwise_deltas"]["timeseries"],
        json!(0.5 - 0.6)
    );

    // An anchor entry in accepted-by-task.json flips the branch to gating.
    let anchored = json!({
        "schema": "JopeSelfplayAcceptedByTask.v1",
        "tasks": {"timeseries": {"checkpoint_sha256": "c".repeat(64)}},
    });
    assert!(task_verdict_is_gating("timeseries", Some(&anchored)));
    assert!(!task_verdict_is_gating("multiclass", Some(&anchored)));
    let gated = evaluate_gate(&evidence, Some(&anchored)).unwrap();
    assert_eq!(gated["pass"], json!(false));
    assert_eq!(gated["promotion_tier"], json!("rejected"));
    assert!(gate_reasons(&gated, "reasons")
        .contains(&"timeseries_regressed_more_than_half_point".to_owned()));
    assert_eq!(
        gated["task_verdicts"]["timeseries"]["advisory"],
        json!(false)
    );
    // Multiclass stays advisory and, with no evidence at all, records nulls.
    assert_eq!(
        gated["task_verdicts"]["multiclass"]["advisory"],
        json!(true)
    );
    assert_eq!(gated["task_pairwise_deltas"]["multiclass"], Value::Null);
}

#[test]
fn promote_refuses_ancestor_drift() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    let checkpoint = root.join("candidate.model.pt");
    fs::write(&checkpoint, b"weights").unwrap();
    let checkpoint_sha = sha256_file(&checkpoint).unwrap();
    let mut evidence = synthetic_evidence(0.005);
    evidence["candidate"]["checkpoint_sha256"] = json!(checkpoint_sha);
    evidence["candidate"]["checkpoint_path"] = json!(checkpoint.to_string_lossy());
    let evidence_path = root.join("evidence.json");
    atomic_json(&evidence_path, &evidence).unwrap();
    let pointer = root.join("learning.json");
    // Existing pointer whose checkpoint differs from the evidence ancestor.
    atomic_json(
        &pointer,
        &json!({"checkpoint_sha256": "9".repeat(64), "checkpoint": "x"}),
    )
    .unwrap();
    let error = promote(
        &evidence_path,
        &root.join("gate.json"),
        &pointer,
        &root.join("immutable"),
        true,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("accepted ancestor changed before pointer promotion"));
}

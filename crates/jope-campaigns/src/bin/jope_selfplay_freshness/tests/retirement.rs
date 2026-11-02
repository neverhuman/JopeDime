use super::*;

#[test]
fn automatic_promotion_turnover_holds_then_retires_only_stale_unstarted_work() {
    let mut fixture = retirement_fixture("queued", "jope-selfplay-eval");
    let status = fixture.supervisor.cycle(false, true).unwrap();
    assert_eq!(status["automatically_held"], 1);
    assert_eq!(
        status["automatic_retirement"]["result"]["running_interrupted"],
        0
    );
    assert_eq!(status["automatic_retirement"]["result"]["retired"], 1);
    let scheduler = fixture.scheduler.lock().unwrap();
    assert_eq!(scheduler.held, ["job-stale"]);
    assert_eq!(scheduler.cancelled, ["job-stale"]);
    assert_eq!(scheduler.jobs[0].state, "cancelled");
    drop(scheduler);
    let pair = sqlite_rows(&fixture.registry, "SELECT state,manifest_sha256 FROM pairs")
        .unwrap()
        .remove(0);
    assert_eq!(pair["state"], "available");
    assert!(pair["manifest_sha256"].is_null());
    let attempt = sqlite_rows(
        &fixture.registry,
        "SELECT state,finalized_at_ms FROM attempts",
    )
    .unwrap()
    .remove(0);
    assert_eq!(attempt["state"], "superseded_unstarted");
    assert!(attempt["finalized_at_ms"].as_i64().unwrap() > 0);
    let claim = sqlite_rows(&fixture.backlog, "SELECT state FROM claims")
        .unwrap()
        .remove(0);
    assert_eq!(claim["state"], "superseded_unstarted");
}

#[test]
fn automatic_retirement_is_idempotent_for_epoch_and_preview() {
    let mut fixture = retirement_fixture("held", "jope-selfplay-eval");
    let inventory = fixture.supervisor.inventory().unwrap();
    let preview = fixture
        .supervisor
        .preview_stale_retirement_from_inventory(&inventory)
        .unwrap();
    let first = fixture.supervisor.auto_retire_stale(&preview).unwrap();
    let second = fixture.supervisor.auto_retire_stale(&preview).unwrap();
    assert_eq!(first, second);
    assert_eq!(fixture.supervisor.state.mutations.len(), 1);
    assert_eq!(fixture.scheduler.lock().unwrap().cancelled.len(), 1);
    let preview_sha256 = preview["preview_receipt_sha256"].as_str().unwrap();
    let key =
        auto_retirement_idempotency_key(&fixture.supervisor.state.active_epoch, preview_sha256)
            .unwrap();
    assert_eq!(first["idempotency_key"], key);
    assert_ne!(
        key,
        auto_retirement_idempotency_key(&format!("checkpoint-{}", "c".repeat(64)), preview_sha256)
            .unwrap()
    );
}

#[test]
fn automatic_retirement_replays_persisted_keys_after_prefix_migration() {
    let mut fixture = retirement_fixture("held", "jope-selfplay-eval");
    let inventory = fixture.supervisor.inventory().unwrap();
    let preview = fixture
        .supervisor
        .preview_stale_retirement_from_inventory(&inventory)
        .unwrap();
    let preview_sha256 = preview["preview_receipt_sha256"].as_str().unwrap();
    let key =
        auto_retirement_idempotency_key(&fixture.supervisor.state.active_epoch, preview_sha256)
            .unwrap();
    let digest = key.rsplit_once('-').unwrap().1;
    let prior_result = json!({"retired": 1, "replayed": true});
    fixture
        .supervisor
        .finish_mutation(
            "apply_stale_retirement",
            preview_sha256,
            &format!("auto-stale-retirement-{digest}"),
            prior_result.clone(),
        )
        .unwrap();

    let result = fixture.supervisor.auto_retire_stale(&preview).unwrap();
    assert_eq!(result["idempotency_key"], key);
    assert_eq!(result["result"], prior_result);
    assert_eq!(fixture.supervisor.state.mutations.len(), 1);
    assert!(fixture.scheduler.lock().unwrap().cancelled.is_empty());
}

#[test]
fn automatic_retirement_loses_dispatch_race_without_interrupting_job() {
    let mut fixture = retirement_fixture("held", "jope-selfplay-eval");
    {
        let mut scheduler = fixture.scheduler.lock().unwrap();
        scheduler.race_on_jobs_call = Some(4);
        scheduler.race_state = Some("running".to_owned());
    }
    let status = fixture.supervisor.cycle(false, true).unwrap();
    assert_eq!(status["automatic_retirement"]["result"]["retired"], 0);
    assert_eq!(
        status["automatic_retirement"]["result"]["running_interrupted"],
        0
    );
    assert_eq!(fixture.scheduler.lock().unwrap().cancelled.len(), 0);
    let pair = sqlite_rows(&fixture.registry, "SELECT state FROM pairs")
        .unwrap()
        .remove(0);
    assert_eq!(pair["state"], "submitted");
    let claim = sqlite_rows(&fixture.backlog, "SELECT state FROM claims")
        .unwrap()
        .remove(0);
    assert_eq!(claim["state"], "submitted");
}

#[test]
fn automatic_retirement_never_touches_gpu_training_project() {
    let mut fixture = retirement_fixture("queued", "jope-selfplay-train");
    let status = fixture.supervisor.cycle(false, true).unwrap();
    assert_eq!(status["automatically_held"], 0);
    assert_eq!(status["automatic_retirement"]["result"]["retired"], 0);
    let scheduler = fixture.scheduler.lock().unwrap();
    assert!(scheduler.held.is_empty());
    assert!(scheduler.cancelled.is_empty());
    assert_eq!(scheduler.jobs[0].state, "queued");
}

#[test]
fn audit_mode_overrides_automatic_retirement_flag() {
    let mut fixture = retirement_fixture("queued", "jope-selfplay-eval");
    let status = fixture.supervisor.cycle(true, true).unwrap();
    assert_eq!(status["audit_only"], true);
    assert_eq!(status["auto_retire_stale"], false);
    assert!(status["automatic_retirement"].is_null());
    let scheduler = fixture.scheduler.lock().unwrap();
    assert!(scheduler.held.is_empty());
    assert!(scheduler.cancelled.is_empty());
    assert_eq!(scheduler.jobs[0].state, "queued");
}

#[test]
fn micro_wave_request_is_receipt_gated_and_idempotent() {
    let temporary = tempdir().unwrap();
    let root = temporary.path().join("root");
    let mut supervisor =
        Supervisor::open(root.clone(), PathBuf::from("/not-used-in-this-test")).unwrap();
    supervisor.state.active_epoch = format!("checkpoint-{}", "b".repeat(64));
    supervisor.state.active_checkpoint = "b".repeat(64);
    let preview = supervisor
        .write_preview("stale_retirement", json!({"actions": []}))
        .unwrap();
    let digest = preview["preview_receipt_sha256"].as_str().unwrap();
    let first = supervisor
        .request_micro_wave(digest, "stable-key", "test")
        .unwrap();
    let second = supervisor
        .request_micro_wave(digest, "stable-key", "changed-reason-is-ignored")
        .unwrap();
    assert_eq!(first, second);
    assert!(Path::new(first["request"].as_str().unwrap()).is_file());
    assert!(supervisor
        .request_micro_wave(digest, "other-key", "test")
        .is_ok());
    assert!(supervisor
        .request_micro_wave(&"f".repeat(64), "unknown-preview", "test")
        .is_err());
}

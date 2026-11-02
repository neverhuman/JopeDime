use super::*;
use std::sync::{Arc, Mutex};
use tempfile::tempdir;

#[derive(Default)]
struct MockSchedulerState {
    jobs: Vec<SchedulerJob>,
    jobs_calls: usize,
    held: Vec<String>,
    cancelled: Vec<String>,
    race_on_jobs_call: Option<usize>,
    race_state: Option<String>,
}

#[derive(Clone)]
struct MockScheduler {
    state: Arc<Mutex<MockSchedulerState>>,
}

impl Scheduler for MockScheduler {
    fn jobs(&self) -> Result<Vec<SchedulerJob>> {
        let mut state = self.state.lock().unwrap();
        state.jobs_calls += 1;
        if state.race_on_jobs_call == Some(state.jobs_calls) {
            let raced = state
                .race_state
                .clone()
                .unwrap_or_else(|| "running".to_owned());
            for job in &mut state.jobs {
                job.state = raced.clone();
            }
        }
        Ok(state.jobs.clone())
    }

    fn hold(&self, job_id: &str) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        let job = state
            .jobs
            .iter_mut()
            .find(|job| job.id == job_id)
            .context("mock hold job is absent")?;
        if matches!(job.state.as_str(), "queued" | "starting") {
            job.state = "held".to_owned();
        }
        state.held.push(job_id.to_owned());
        Ok(())
    }

    fn cancel(&self, job_id: &str) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        let job = state
            .jobs
            .iter_mut()
            .find(|job| job.id == job_id)
            .context("mock cancel job is absent")?;
        if job.state != "held" {
            bail!("mock refuses to cancel a non-held job");
        }
        job.state = "cancelled".to_owned();
        state.cancelled.push(job_id.to_owned());
        Ok(())
    }
}

struct RetirementFixture {
    _temporary: tempfile::TempDir,
    supervisor: Supervisor,
    scheduler: Arc<Mutex<MockSchedulerState>>,
    registry: PathBuf,
    backlog: PathBuf,
}

fn retirement_fixture(scheduler_state: &str, project: &str) -> RetirementFixture {
    let temporary = tempdir().unwrap();
    let root = temporary.path().join("root");
    fs::create_dir_all(root.join("ops")).unwrap();
    fs::create_dir_all(root.join("manifests")).unwrap();
    let old_checkpoint = "a".repeat(64);
    let manifest = root.join("manifests/stale.json");
    atomic_json(
        &manifest,
        &json!({
            "schema": "JopeFlywheelCorpusEvalShard.v1",
            "checkpoint_epoch": format!("checkpoint-{old_checkpoint}"),
            "candidates": []
        }),
    )
    .unwrap();
    let manifest_sha256 = sha256_file(&manifest).unwrap();
    let backlog = root.join("ops/backlog.sqlite");
    let connection = Connection::open(&backlog).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE claims(
                   manifest_sha256 TEXT NOT NULL,
                   manifest TEXT NOT NULL,
                   job_id TEXT PRIMARY KEY,
                   state TEXT NOT NULL,
                   reservation_owner TEXT,
                   updated_at_ms INTEGER NOT NULL
                 );",
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO claims VALUES(?1,?2,'job-stale','submitted','owner',1)",
            params![manifest_sha256, manifest.to_string_lossy()],
        )
        .unwrap();
    drop(connection);

    let registry = root.join("ops/pair-registry.sqlite");
    let connection = Connection::open(&registry).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE pairs(
                   task TEXT,dataset_hash TEXT,semantic_sha256 TEXT,protocol_sha256 TEXT,
                   canonical_genome TEXT,state TEXT,reservation_owner TEXT,manifest_path TEXT,
                   manifest_sha256 TEXT,job_id TEXT,receipt_sha256 TEXT,failure_class TEXT,
                   infrastructure_retries INTEGER,created_at_ms INTEGER,updated_at_ms INTEGER,
                   PRIMARY KEY(task,dataset_hash,semantic_sha256,protocol_sha256)
                 );
                 CREATE TABLE attempts(
                   id INTEGER PRIMARY KEY,task TEXT,dataset_hash TEXT,semantic_sha256 TEXT,
                   protocol_sha256 TEXT,attempt_no INTEGER,state TEXT,reservation_owner TEXT,
                   manifest_path TEXT,manifest_sha256 TEXT,job_id TEXT,receipt_sha256 TEXT,
                   failure_class TEXT,detail_json TEXT,created_at_ms INTEGER,finalized_at_ms INTEGER
                 );
                 CREATE TABLE events(
                   id INTEGER PRIMARY KEY,kind TEXT,task TEXT,dataset_hash TEXT,
                   semantic_sha256 TEXT,protocol_sha256 TEXT,detail_json TEXT,created_at_ms INTEGER
                 );",
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO pairs VALUES(
                   'regression','dataset','semantic','protocol','genome','submitted','owner',
                   ?1,?2,'job-stale',NULL,NULL,0,1,1
                 )",
            params![manifest.to_string_lossy(), manifest_sha256],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO attempts VALUES(
                   NULL,'regression','dataset','semantic','protocol',1,'submitted','owner',
                   ?1,?2,'job-stale',NULL,NULL,'{}',1,NULL
                 )",
            params![manifest.to_string_lossy(), manifest_sha256],
        )
        .unwrap();
    drop(connection);

    let scheduler = Arc::new(Mutex::new(MockSchedulerState {
        jobs: vec![SchedulerJob {
            id: "job-stale".to_owned(),
            state: scheduler_state.to_owned(),
            project: Some(project.to_owned()),
            ..SchedulerJob::default()
        }],
        ..MockSchedulerState::default()
    }));
    let mut supervisor = Supervisor::open(root, PathBuf::from("/not-used-in-this-test")).unwrap();
    supervisor.scheduler = Box::new(MockScheduler {
        state: scheduler.clone(),
    });
    let active = "b".repeat(64);
    supervisor.state.active_checkpoint = active.clone();
    supervisor.state.active_epoch = format!("checkpoint-{active}");
    RetirementFixture {
        _temporary: temporary,
        supervisor,
        scheduler,
        registry,
        backlog,
    }
}

#[test]
fn checkpoint_epoch_requires_lowercase_sha256() {
    let digest = "a".repeat(64);
    assert_eq!(
        epoch_for_checkpoint(&digest).unwrap(),
        format!("checkpoint-{digest}")
    );
    assert!(epoch_for_checkpoint("not-a-hash").is_err());
    assert!(epoch_for_checkpoint(&"A".repeat(64)).is_err());
}

#[test]
fn scope_excludes_protected_lanes() {
    let base = json!({
        "schema": "JopeFlywheelCorpusEvalShard.v1",
        "candidates": []
    });
    assert!(freshness_scoped_manifest(&base));
    for lane in [
        "checkpoint_gate",
        "sealed_audit",
        "evaluator_canary",
        "milestone-sealed-zero-shot",
    ] {
        let mut protected = base.clone();
        protected["evaluation_lane"] = json!(lane);
        assert!(!freshness_scoped_manifest(&protected));
    }
}

#[test]
fn manifest_epoch_falls_back_only_to_authority_or_legacy() {
    let digest = "b".repeat(64);
    assert_eq!(
        manifest_epoch(&json!({
            "schema": "JopeFlywheelCorpusEvalShard.v1",
            "candidates": [],
            "label_authority": {"model_checkpoint_sha256": digest}
        })),
        format!("checkpoint-{digest}")
    );
    assert_eq!(
        manifest_epoch(&json!({
            "schema": "JopeFlywheelCorpusEvalShard.v1",
            "candidates": []
        })),
        "legacy-jope-testonly"
    );
}

#[test]
fn sql_quoting_never_changes_statement_boundaries() {
    assert_eq!(
        sql_quote("job'; DROP TABLE claims;--"),
        "'job''; DROP TABLE claims;--'"
    );
}

#[test]
fn minimal_scheduler_decode_does_not_retain_large_job_payloads() {
    let command = "candidate-command-".repeat(64);
    let output = "receipt-output-".repeat(32);
    let rows = (0..12_000)
        .map(|index| {
            json!({
                "id": format!("job-{index:05}"),
                "state": "queued",
                "project": "jope-selfplay-eval",
                "node": null,
                "queue_ts": "2026-07-27T00:00:00Z",
                "command": command,
                "outputs": [output, output],
                "message": output,
                "run_dir": format!("/runs/job-{index:05}")
            })
        })
        .collect::<Vec<_>>();
    let encoded = serde_json::to_vec(&json!({"Jobs": {"jobs": rows}})).unwrap();
    let started = Instant::now();
    let decoded = parse_scheduler_jobs(&encoded).unwrap();
    let elapsed = started.elapsed();
    let retained_string_bytes = decoded
        .iter()
        .map(|job| {
            job.id.len()
                + job.state.len()
                + job.project.as_deref().map_or(0, str::len)
                + job.queue_ts.as_deref().map_or(0, str::len)
                + job.run_dir.as_deref().map_or(0, str::len)
        })
        .sum::<usize>();
    eprintln!(
        "minimal-scq-decode jobs={} input_bytes={} retained_string_bytes={} elapsed_ms={}",
        decoded.len(),
        encoded.len(),
        retained_string_bytes,
        elapsed.as_millis()
    );
    assert_eq!(decoded.len(), 12_000);
    assert!(retained_string_bytes * 20 < encoded.len());
}

#[test]
fn identical_preview_is_reused_and_expired_preview_is_pruned() {
    let temporary = tempdir().unwrap();
    let root = temporary.path().join("root");
    let mut supervisor = Supervisor::open(root, PathBuf::from("/not-used-in-this-test")).unwrap();
    let payload = json!({"actions": [{"job_id": "same"}]});
    let first = supervisor
        .write_preview("stale_retirement", payload.clone())
        .unwrap();
    let second = supervisor
        .write_preview("stale_retirement", payload)
        .unwrap();
    assert_eq!(
        first["preview_receipt_sha256"],
        second["preview_receipt_sha256"]
    );
    assert_eq!(supervisor.state.previews.len(), 1);

    supervisor.state.previews.insert(
        "f".repeat(64),
        Preview {
            schema: "JopeFreshnessPreview.v1".to_owned(),
            kind: "stale_retirement".to_owned(),
            created_at_ms: now_ms() - PREVIEW_MAX_AGE_MS - 1,
            payload: json!({"actions": [{"job_id": "expired"}]}),
        },
    );
    supervisor.prune_state();
    assert_eq!(supervisor.state.previews.len(), 1);
}

#[test]
fn manifest_cache_reuses_immutable_audit_and_detects_metadata_change() {
    let temporary = tempdir().unwrap();
    let root = temporary.path().join("root");
    let manifest = temporary.path().join("manifest.json");
    let initial = serde_json::to_vec(&json!({
        "schema": "JopeFlywheelCorpusEvalShard.v1",
        "checkpoint_epoch": format!("checkpoint-{}", "a".repeat(64)),
        "candidates": []
    }))
    .unwrap();
    fs::write(&manifest, initial).unwrap();
    let expected = sha256_file(&manifest).unwrap();
    let supervisor = Supervisor::open(root, PathBuf::from("/not-used-in-this-test")).unwrap();
    let first = supervisor
        .audit_manifest("job", &manifest, &expected, false)
        .unwrap();
    let second = supervisor
        .audit_manifest("job", &manifest, &expected, false)
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(supervisor.manifest_cache.borrow().len(), 1);

    fs::write(&manifest, b"{\"changed\":true,\"longer\":true}").unwrap();
    assert!(supervisor
        .audit_manifest("job", &manifest, &expected, false)
        .is_err());
}

#[test]
fn retirement_preserves_attempt_and_releases_pair() {
    let temporary = tempdir().unwrap();
    let root = temporary.path().join("root");
    fs::create_dir_all(root.join("ops")).unwrap();
    let registry = root.join("ops/pair-registry.sqlite");
    sqlite_exec(
        &registry,
        r#"
CREATE TABLE pairs(
  task TEXT,dataset_hash TEXT,semantic_sha256 TEXT,protocol_sha256 TEXT,
  canonical_genome TEXT,state TEXT,reservation_owner TEXT,manifest_path TEXT,
  manifest_sha256 TEXT,job_id TEXT,receipt_sha256 TEXT,failure_class TEXT,
  infrastructure_retries INTEGER,created_at_ms INTEGER,updated_at_ms INTEGER,
  PRIMARY KEY(task,dataset_hash,semantic_sha256,protocol_sha256)
);
CREATE TABLE attempts(
  id INTEGER PRIMARY KEY,task TEXT,dataset_hash TEXT,semantic_sha256 TEXT,
  protocol_sha256 TEXT,attempt_no INTEGER,state TEXT,reservation_owner TEXT,
  manifest_path TEXT,manifest_sha256 TEXT,job_id TEXT,receipt_sha256 TEXT,
  failure_class TEXT,detail_json TEXT,created_at_ms INTEGER,finalized_at_ms INTEGER
);
CREATE TABLE events(
  id INTEGER PRIMARY KEY,kind TEXT,task TEXT,dataset_hash TEXT,
  semantic_sha256 TEXT,protocol_sha256 TEXT,detail_json TEXT,created_at_ms INTEGER
);
INSERT INTO pairs VALUES(
  'regression','dataset','semantic','protocol','genome','submitted','owner',
  '/manifest','manifest','job',NULL,NULL,0,1,1
);
INSERT INTO attempts VALUES(
  NULL,'regression','dataset','semantic','protocol',1,'submitted','owner',
  '/manifest','manifest','job',NULL,NULL,'{}',1,NULL
);
"#,
    )
    .unwrap();
    let supervisor = Supervisor::open(root, PathBuf::from("/not-used-in-this-test")).unwrap();
    assert_eq!(
        supervisor
            .supersede_pairs("manifest", "job", "submitted", &"a".repeat(64))
            .unwrap(),
        1
    );
    let pair = sqlite_rows(&registry, "SELECT state,manifest_sha256 FROM pairs")
        .unwrap()
        .remove(0);
    assert_eq!(pair["state"], "available");
    assert!(pair["manifest_sha256"].is_null());
    let attempt = sqlite_rows(
        &registry,
        "SELECT state,receipt_sha256,finalized_at_ms FROM attempts",
    )
    .unwrap()
    .remove(0);
    assert_eq!(attempt["state"], "superseded_unstarted");
    assert_eq!(attempt["receipt_sha256"], "a".repeat(64));
    assert!(attempt["finalized_at_ms"].as_i64().unwrap() > 0);
}

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

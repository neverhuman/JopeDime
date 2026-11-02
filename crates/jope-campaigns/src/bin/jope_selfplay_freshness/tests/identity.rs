use super::*;

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

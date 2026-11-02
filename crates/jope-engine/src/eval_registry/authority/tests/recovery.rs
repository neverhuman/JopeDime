use super::*;

#[test]
fn deactivation_rechecks_authorization_and_start_consumption() {
    let fixture = Fixture::new(true);
    let (mut authority, fit, claim) = fixture.claim();
    authority.namespace_deactivate("synthetic").unwrap();
    assert!(authority
        .authorize(&fit, &claim, &ProcessIdentity::current().unwrap())
        .unwrap_err()
        .to_string()
        .contains("inactive"));
    authority.activate_synthetic_for_test().unwrap();
    let start = authority
        .authorize(&fit, &claim, &ProcessIdentity::current().unwrap())
        .unwrap();
    authority.namespace_deactivate("synthetic").unwrap();
    assert!(authority.consume_start(&fit, &start).is_err());
    assert_eq!(
        authority
            .db
            .query_row(
                "SELECT count(*) FROM entries WHERE state='running'",
                [],
                |r| r.get::<_, u64>(0)
            )
            .unwrap(),
        0
    );
}

#[test]
fn witnessed_journal_replay_preserves_unknown_start_and_epoch_does_not_activate() {
    let fixture = Fixture::new(true);
    {
        fixture
            .open()
            .db
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
    }
    let old_database = fs::read(fixture.root.join("registry.sqlite")).unwrap();
    let start = {
        let (mut authority, fit, claim) = fixture.claim();
        let start = authority
            .authorize_with_deadline(
                &fit,
                &claim,
                &ProcessIdentity::current().unwrap(),
                Duration::from_secs(120),
            )
            .unwrap();
        authority
            .db
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
        start
    };
    fs::write(fixture.root.join("registry.sqlite"), old_database).unwrap();
    assert!(Authority::open(&fixture.root, &fixture.witness).is_err());
    let inspection = Authority::recovery_inspect(&fixture.root, &fixture.witness).unwrap();
    assert!(
        inspection["journal"]["seq"].as_u64().unwrap()
            > inspection["database"]["seq"].as_u64().unwrap()
    );
    let replay = Authority::recovery_replay(&fixture.root, &fixture.witness).unwrap();
    assert_eq!(replay["unresolved_original_work"][0]["state"], "authorized");
    assert!(
        Authority::open(&fixture.root, &fixture.witness).is_err(),
        "replay cannot remove recovery hold"
    );
    let report = Authority::recovery_reconcile(&fixture.root, &fixture.witness).unwrap();
    Authority::recovery_commit_epoch(&fixture.root, &fixture.witness, &report).unwrap();
    let mut authority = fixture.open();
    assert_eq!(
        state(authority.lookup(&fit(), &request()).unwrap()),
        "authorized"
    );
    assert!(
        authority.consume_start(&fit(), &start).is_err(),
        "prior activation must not survive new epoch"
    );
    assert!(state(
        authority
            .claim(
                &fit_program("(pipe X (standard_scale) (ridge))"),
                &request()
            )
            .unwrap()
    )
    .contains("inactive"));
    authority.activate_synthetic_for_test().unwrap();
    assert!(
        authority
            .consume_start(&fit(), &start)
            .err()
            .unwrap()
            .to_string()
            .contains("prior recovery epoch"),
        "reactivation cannot resurrect a prior-epoch token"
    );
    assert!(
        authority.require_reconciled_service_start().is_err(),
        "original unknown work remains unavailable"
    );
}

#[test]
fn recovery_replay_refuses_missing_or_unacknowledged_journal_tails() {
    for missing_tail in [false, true] {
        let fixture = Fixture::new(true);
        let old_journal = fs::read(fixture.root.join("dispatch.jsonl")).unwrap();
        let old_witness = fs::read(&fixture.witness).unwrap();
        {
            let (mut authority, fit, claim) = fixture.claim();
            authority
                .authorize(&fit, &claim, &ProcessIdentity::current().unwrap())
                .unwrap();
        }
        if missing_tail {
            fs::write(fixture.root.join("dispatch.jsonl"), old_journal).unwrap();
        } else {
            fs::remove_file(&fixture.witness).unwrap();
            fs::write(&fixture.witness, old_witness).unwrap();
        }
        assert!(Authority::recovery_replay(&fixture.root, &fixture.witness).is_err());
        assert!(Authority::recovery_reconcile(&fixture.root, &fixture.witness).is_err());
        assert!(Authority::open(&fixture.root, &fixture.witness).is_err());
    }
}

#[test]
fn recovery_epoch_requires_current_retained_reconciliation_evidence() {
    let fixture = Fixture::new(true);
    let mut authority = fixture.open();
    let forged = authority
        .store_blob(
            &encoded(
                &json!({"verified":true,"schema":"parallel.authority_recovery_inspection.v1"}),
            )
            .unwrap(),
        )
        .unwrap();
    let report = Authority::recovery_reconcile(&fixture.root, &fixture.witness).unwrap();
    assert!(
        authority.claim(&fit(), &request()).is_err(),
        "already-open client must respect durable recovery hold"
    );
    assert!(Authority::recovery_commit_epoch(&fixture.root, &fixture.witness, &forged).is_err());
    Authority::recovery_commit_epoch(&fixture.root, &fixture.witness, &report).unwrap();
    assert!(Authority::recovery_commit_epoch(&fixture.root, &fixture.witness, &report).is_err());
    let mut authority = fixture.open();
    assert!(state(authority.claim(&fit(), &request()).unwrap()).contains("inactive"));
}

#[test]
fn recovery_inspection_exposes_index_corruption_and_missing_original_model() {
    let fixture = Fixture::new(true);
    let (mut authority, fit, _, observed) = run_child(&fixture, "success");
    let completion = authority
        .complete_observed(
            &fit,
            observed,
            ExecutionOutcome::Success,
            crate::eval_registry::CompletionArtifacts {
                output: b"original synthetic output",
                model: Some(b"original synthetic model"),
                predictions: None,
                metrics: None,
            },
        )
        .unwrap();
    let model = authority
        .completion(&completion)
        .unwrap()
        .model_sha256
        .unwrap();
    fs::remove_file(fixture.root.join("objects").join(model)).unwrap();
    let inspection = Authority::recovery_inspect(&fixture.root, &fixture.witness).unwrap();
    assert!(!inspection["retained_artifact_errors"]
        .as_array()
        .unwrap()
        .is_empty());
    assert!(Authority::recovery_reconcile(&fixture.root, &fixture.witness).is_err());
    drop(authority);
    let db = Connection::open(fixture.root.join("registry.sqlite")).unwrap();
    db.execute("DELETE FROM numerical_guards", []).unwrap();
    drop(db);
    let inspection = Authority::recovery_inspect(&fixture.root, &fixture.witness).unwrap();
    assert!(inspection["indexed_state_error"]
        .as_str()
        .unwrap()
        .contains("journal"));
}

#[test]
fn pending_history_suspends_prior_coverage_and_old_claims_cannot_cross_epochs() {
    let fixture = Fixture::new(true);
    let (mut authority, fit, claim) = fixture.claim();
    let inventory = staged_history_fixture(&fixture);
    let stage = authority.history_stage(&inventory).unwrap();
    assert!(state(authority.claim(&fit, &request()).unwrap()).contains("import incomplete"));
    assert!(authority
        .authorize(&fit, &claim, &ProcessIdentity::current().unwrap())
        .is_err());
    authority
        .history_import_batch(&HistoryBatch {
            stage_sha256: stage.clone(),
            start: 0,
            end: 3,
        })
        .unwrap();
    authority.history_commit_coverage(&stage).unwrap();
    assert!(authority
        .authorize(&fit, &claim, &ProcessIdentity::current().unwrap())
        .is_err());
    drop(authority);
    let report = Authority::recovery_reconcile(&fixture.root, &fixture.witness).unwrap();
    Authority::recovery_commit_epoch(&fixture.root, &fixture.witness, &report).unwrap();
    let mut authority = fixture.open();
    authority.activate_synthetic_for_test().unwrap();
    assert!(authority
        .authorize(&fit, &claim, &ProcessIdentity::current().unwrap())
        .unwrap_err()
        .to_string()
        .contains("prior recovery epoch"));
}

#[test]
fn replay_preserves_same_key_database_only_unknown_evidence() {
    let fixture = Fixture::new(true);
    let (authority, fit, _) = fixture.claim();
    let original = "database-only unknown authorization token";
    authority
        .db
        .execute(
            "UPDATE entries SET state='authorized',start_json=? WHERE key=?",
            params![original, fit.key],
        )
        .unwrap();
    drop(authority);
    let before = Connection::open(fixture.root.join("registry.sqlite")).unwrap();
    let row = Authority::state_row(&before, "entries", &fit.key).unwrap();
    drop(before);
    let report = Authority::recovery_inspect(&fixture.root, &fixture.witness).unwrap();
    assert!(report["database_checkpoint_state_error"]
        .as_str()
        .unwrap()
        .contains("conflicts"));
    assert!(Authority::recovery_replay(&fixture.root, &fixture.witness).is_err());
    let after = Connection::open(fixture.root.join("registry.sqlite")).unwrap();
    assert_eq!(
        Authority::state_row(&after, "entries", &fit.key).unwrap(),
        row
    );
    assert!(
        !fixture.root.join("recovery.pending").exists(),
        "conflict refusal must precede mutation"
    );
}

#[test]
fn recommitting_old_coverage_cannot_resurrect_prior_activation() {
    let fixture = Fixture::new(false);
    let first = staged_history_fixture(&fixture);
    let mut authority = fixture.open();
    let first_sha = authority.import_history(&first).unwrap();
    authority.activate_synthetic_for_test().unwrap();
    assert_eq!(
        state(authority.lookup(&fit(), &request()).unwrap()),
        "unseen: explicit claim required"
    );
    let mut second = first.clone();
    second.occurrences[0].reason = "second reviewed historical disposition".into();
    let second_sha = authority.import_history(&second).unwrap();
    assert_ne!(first_sha, second_sha);
    assert!(state(authority.lookup(&fit(), &request()).unwrap()).contains("inactive"));
    authority.history_commit_coverage(&first_sha).unwrap();
    drop(authority);
    let mut authority = fixture.open();
    assert!(state(authority.claim(&fit(), &request()).unwrap()).contains("inactive"));
    assert_eq!(
        authority
            .db
            .query_row(
                "SELECT state FROM activations WHERE scope='synthetic'",
                [],
                |r| r.get::<_, String>(0)
            )
            .unwrap(),
        "inactive"
    );
}

#[test]
fn native_v6_initialization_preserves_legacy_stores_and_legacy_admission_hold() {
    let directory = tempfile::tempdir().unwrap();
    let scopes = vec![ScopePolicy {
        scope: "synthetic".into(),
        training_sha256: vec![fit().identity.training_sha256.clone()],
        numerical_contract_sha256: vec![numerical_contract_sha256(&contract()).unwrap()],
        fit_seeds: vec![42],
    }];
    let legacy = directory.path().join("legacy");
    let legacy_witness = directory.path().join("legacy-witness");
    Authority::initialize(&legacy, &legacy_witness, scopes.clone()).unwrap();
    let before = fs::read(legacy.join("authority.json")).unwrap();
    assert!(Authority::initialize_native(&legacy, &legacy_witness, scopes.clone()).is_err());
    assert_eq!(before, fs::read(legacy.join("authority.json")).unwrap());
    let legacy_store = Authority::open(&legacy, &legacy_witness).unwrap();
    assert_eq!(legacy_store.manifest.schema, SCHEMA);
    assert!(!legacy_store
        .state_tables()
        .any(|table| table == "native_slots"));
    drop(legacy_store);
    let native = directory.path().join("native");
    let witness = directory.path().join("native-witness");
    Authority::initialize_native(&native, &witness, scopes).unwrap();
    let mut store = Authority::open(&native, &witness).unwrap();
    assert_eq!(store.manifest.schema, NATIVE_SCHEMA);
    assert!(store.state_tables().any(|table| table == "native_slots"));
    assert!(matches!(
        store.claim(&fit(), &request()).unwrap(),
        ClaimOutcome::Blocked { .. }
    ));
    assert_eq!(
        store
            .db
            .query_row("SELECT COUNT(*) FROM entries", [], |r| r.get::<_, u64>(0))
            .unwrap(),
        0
    );
    drop(store);
    assert_eq!(
        Authority::open(&native, &witness).unwrap().manifest.schema,
        NATIVE_SCHEMA
    );
}

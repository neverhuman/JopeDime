use super::*;

#[test]
fn missing_storage_and_uncovered_history_fail_closed() {
    let f = Fixture::new(false);
    let mut a = f.open();
    assert!(state(a.claim(&fit(), &request()).unwrap()).contains("coverage"));
    let sequence = a.checkpoint.seq;
    assert!(state(a.claim(&fit(), &request()).unwrap()).contains("coverage"));
    assert_eq!(
        sequence, a.checkpoint.seq,
        "blocked lookups must not journal"
    );
    fs::remove_file(f.root.join("registry.sqlite")).unwrap();
    assert!(a.claim(&fit(), &request()).is_err());
    assert!(Authority::open(&f.root, &f.witness).is_err());
    assert!(!f.root.join("registry.sqlite").exists());
}

#[test]
fn identities_preserve_math_and_separate_query_provenance() {
    let f = fit();
    let g = VerifiedFit::from_verified_runtime(
        " (pipe X (ridge)) ",
        crate::eval_registry::FitInputs {
            x: vec![vec![1.], vec![2.]],
            y: vec![2., 3.],
            preprocessing: b"preprocess".to_vec(),
            feature_view: b"raw".to_vec(),
        },
        42,
        f.identity.executable_sha256.clone(),
        f.identity.runtime_sha256.clone(),
        &contract(),
    )
    .unwrap();
    assert_eq!(f.key, g.key);
    let reverse = ordered_training_sha256(&[vec![2.], vec![1.]], &[3., 2.]).unwrap();
    assert_ne!(f.identity.training_sha256, reverse);
    assert!(ordered_training_sha256(&[vec![f64::NAN]], &[1.]).is_err());
    assert!(serde_json::from_value::<NumericalContract>(json!({"protocol":null})).is_err());
    assert_ne!(
        prediction_evaluation_key(b"model", b"original", b"query1", b"target", b"rmse").unwrap(),
        prediction_evaluation_key(b"model", b"original", b"query2", b"target", b"rmse").unwrap()
    );
    let fixture = Fixture::new(true);
    let (mut a, _, _) = fixture.claim();
    let mut p = request();
    p.task_id = "generation-and-seed-changed".into();
    p.details = json!({"assembly":"new","timeout":900,"admission":"new"});
    assert_eq!(state(a.claim(&g, &p).unwrap()), "claimed");
    let h1 = fit_program("(pipe X (standard_scale) (pca k=2) (ridge))");
    let h2 = fit_program("(pipe X (pca k=2) (standard_scale) (ridge))");
    assert_ne!(h1.key, h2.key);
}
#[test]
fn partial_exclusion_matches_richer_request_and_authorization_race() {
    let fixture = Fixture::new(true);
    let (mut a, f, c) = fixture.claim();
    a.import_exclusion(&HistoricalExclusion {
        scope: "synthetic".into(),
        canonical_program: Some(f.identity.canonical_program.clone()),
        training_sha256: None,
        evidence_sha256: hash(b"old occurrence"),
        reason: "historical fit may have run".into(),
    })
    .unwrap();
    assert!(a
        .authorize(&f, &c, &ProcessIdentity::current().unwrap())
        .is_err());
    assert!(state(a.claim(&f, &request()).unwrap()).contains("historical"));
}

#[test]
fn exclusion_between_authorization_and_consumption_blocks_start() {
    let fixture = Fixture::new(true);
    let (mut a, f, c) = fixture.claim();
    let token = a
        .authorize(&f, &c, &ProcessIdentity::current().unwrap())
        .unwrap();
    a.import_exclusion(&HistoricalExclusion {
        scope: "synthetic".into(),
        canonical_program: None,
        training_sha256: Some(f.identity.training_sha256.clone()),
        evidence_sha256: hash(b"unresolved program"),
        reason: "unresolved scope".into(),
    })
    .unwrap();
    assert!(a.consume_start(&f, &token).is_err());
}

#[test]
fn concurrent_104_requests_authorize_exactly_one_synthetic_start() {
    let fixture = Fixture::new(true);
    let barrier = Arc::new(Barrier::new(104));
    let threads: Vec<_> = (0..104)
        .map(|_| {
            let root = fixture.root.clone();
            let witness = fixture.witness.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let mut authority = Authority::open(&root, &witness).unwrap();
                let fit = fit();
                barrier.wait();
                match authority.claim(&fit, &request()).unwrap() {
                    ClaimOutcome::Claimed(claim) => {
                        // This tests single-start arbitration, with the original
                        // 30-second logical budget, independently of SDK hash cost.
                        let _clock = SyntheticClock::frozen();
                        let start = authority
                            .authorize(&fit, &claim, &ProcessIdentity::current().unwrap())
                            .unwrap();
                        let custody = authority.consume_start(&fit, &start).unwrap();
                        custody.verify().unwrap();
                        // This is a synthetic start count; no numerical evaluator runs.
                        1
                    }
                    _ => 0,
                }
            })
        })
        .collect();
    assert_eq!(
        threads.into_iter().map(|t| t.join().unwrap()).sum::<u32>(),
        1
    );
    let a = fixture.open();
    assert_eq!(
        a.checkpoint.seq, 5,
        "coverage, explicit synthetic activation, claim, authorize, consume only"
    );
    assert_eq!(
        state(fixture.open().lookup(&fit(), &request()).unwrap()),
        "running"
    );
}

#[test]
fn synthetic_clock_preserves_original_start_deadline_and_thread_isolation() {
    let fixture = Fixture::new(true);
    let (mut authority, fit, claim) = fixture.claim();
    let clock = SyntheticClock::frozen();
    assert!(
        std::thread::spawn(|| SYNTHETIC_MONOTONIC_NS.with(|clock| clock.get().is_none()))
            .join()
            .unwrap()
    );
    let start = authority
        .authorize(&fit, &claim, &ProcessIdentity::current().unwrap())
        .unwrap();
    assert_eq!(
        start.deadline_monotonic_ns - start.authorized_monotonic_ns,
        30_000_000_000
    );
    clock.advance_to(start.deadline_monotonic_ns);
    let error = authority.consume_start(&fit, &start).err().unwrap();
    assert_eq!(error.to_string(), "original start deadline expired");
    assert_eq!(authority.checkpoint.seq, 4, "no expired start consumed");
    assert_eq!(
        state(authority.claim(&fit, &request()).unwrap()),
        "authorized"
    );
    drop(clock);
    assert!(SYNTHETIC_MONOTONIC_NS.with(|clock| clock.get().is_none()));
}

#[test]
fn sequential_foreign_and_inherited_tokens_are_fenced() {
    let fixture = Fixture::new(true);
    let (mut a, f, c) = fixture.claim();
    // Isolate token identity and single consumption from SDK hash cost, using
    // the same original logical deadline as the concurrent arbitration fixture.
    let _clock = SyntheticClock::frozen();
    let token = a
        .authorize(&f, &c, &ProcessIdentity::current().unwrap())
        .unwrap();
    assert!(a
        .authorize(&f, &c, &ProcessIdentity::current().unwrap())
        .is_err());
    let mut foreign = token.clone();
    foreign.executor.birth_ticks += 1;
    assert!(a.consume_start(&f, &foreign).is_err());
    let custody = a.consume_start(&f, &token).unwrap();
    custody.verify().unwrap();
    drop(custody);
    assert!(a.consume_start(&f, &token).is_err());
    assert_eq!(state(a.claim(&f, &request()).unwrap()), "running");
}

#[test]
fn sqlite_restore_and_missing_or_torn_journal_never_erase_dispatch() {
    let fixture = Fixture::new(true);
    {
        fixture
            .open()
            .db
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
    }
    let old = fs::read(fixture.root.join("registry.sqlite")).unwrap();
    {
        let (mut a, f, c) = fixture.claim();
        a.authorize(&f, &c, &ProcessIdentity::current().unwrap())
            .unwrap();
        a.db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
    }
    fs::write(fixture.root.join("registry.sqlite"), old).unwrap();
    assert!(Authority::open(&fixture.root, &fixture.witness).is_err());
    let fixture = Fixture::new(true);
    let mut a = fixture.open();
    OpenOptions::new()
        .append(true)
        .open(fixture.root.join("dispatch.jsonl"))
        .unwrap()
        .write_all(b"{torn")
        .unwrap();
    assert!(a.claim(&fit(), &request()).is_err());
    assert!(Authority::open(&fixture.root, &fixture.witness).is_err());
    let fixture = Fixture::new(true);
    let mut a = fixture.open();
    fs::remove_file(fixture.root.join("dispatch.jsonl")).unwrap();
    assert!(a.claim(&fit(), &request()).is_err());
}

#[test]
fn old_database_and_old_journal_still_fail_against_independent_witness() {
    let fixture = Fixture::new(true);
    {
        fixture
            .open()
            .db
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
    }
    let old = fs::read(fixture.root.join("registry.sqlite")).unwrap();
    let journal = fs::read(fixture.root.join("dispatch.jsonl")).unwrap();
    {
        let (mut a, f, c) = fixture.claim();
        a.authorize(&f, &c, &ProcessIdentity::current().unwrap())
            .unwrap();
        a.db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
    }
    fs::write(fixture.root.join("registry.sqlite"), old).unwrap();
    fs::write(fixture.root.join("dispatch.jsonl"), journal).unwrap();
    assert!(Authority::open(&fixture.root, &fixture.witness).is_err());
}

#[test]
fn disk_failure_after_journal_blocks_authority_on_original_and_restart() {
    for failpoint in ["after_journal", "after_witness"] {
        let fixture = Fixture::new(true);
        let mut a = fixture.open();
        a.failpoint = Some(failpoint);
        assert!(a.claim(&fit(), &request()).is_err());
        a.failpoint = None;
        assert!(a.claim(&fit(), &request()).is_err());
        assert!(Authority::open(&fixture.root, &fixture.witness).is_err());
    }
}

#[test]
fn denied_scope_and_unbound_empty_history_do_not_mutate() {
    let fixture = Fixture::new(false);
    let mut a = fixture.open();
    let mut p = request();
    p.scope = "sealed_test".into();
    assert!(a.claim(&fit(), &p).is_err());
    let inventory = HistoryInventory {
        scope: "synthetic".into(),
        source_roots: vec![],
        files: BTreeMap::new(),
        occurrences: vec![],
        legacy_fence_receipt: PathBuf::new(),
        legacy_fence_sha256: hash(b"fake"),
    };
    assert!(a.import_history(&inventory).is_err());
    assert_eq!(a.checkpoint.seq, 0);
}

#[test]
fn history_import_requires_every_source_and_matches_partial_programs() {
    let fixture = Fixture::new(false);
    let mut a = fixture.open();
    let history = fixture._temp.path().join("history");
    fs::create_dir(&history).unwrap();
    let source = history.join("attempt.json");
    fs::write(&source, b"unresolved original attempt").unwrap();
    let fence = fixture._temp.path().join("legacy-fence.json");
    fs::write(&fence, b"synthetic stopped dispatcher").unwrap();
    let sha = hash(&fs::read(&source).unwrap());
    let mut inventory = HistoryInventory {
        scope: "synthetic".into(),
        source_roots: vec![history.clone()],
        files: BTreeMap::from([(source, sha.clone())]),
        occurrences: vec![HistoricalExclusion {
            scope: "synthetic".into(),
            canonical_program: None,
            training_sha256: None,
            evidence_sha256: sha,
            reason: "unresolved source program".into(),
        }],
        legacy_fence_receipt: fence,
        legacy_fence_sha256: hash(b"synthetic stopped dispatcher"),
    };
    fs::write(history.join("unaccounted"), b"missing source").unwrap();
    assert!(a.import_history(&inventory).is_err());
    let missing = history.join("unaccounted");
    let sha = hash(b"missing source");
    inventory.files.insert(missing, sha.clone());
    inventory.occurrences.push(HistoricalExclusion {
        scope: "synthetic".into(),
        canonical_program: None,
        training_sha256: None,
        evidence_sha256: sha,
        reason: "unresolved parse failure".into(),
    });
    a.import_history(&inventory).unwrap();
    assert!(state(a.claim(&fit(), &request()).unwrap()).contains("unresolved"));
}

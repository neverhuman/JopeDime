#[test]
fn outbox_limits_and_cadence_are_explicit_adopted_configuration() {
    let adopted = Config::adopted();
    assert_eq!(
        (
            adopted.limits.pause_entries,
            adopted.limits.pause_bytes,
            adopted.limits.max_entries,
            adopted.limits.max_bytes
        ),
        (7_500, 768 << 20, 10_000, 1 << 30)
    );
    assert_eq!(
        (
            adopted.cadence.flush_interval_ms,
            adopted.cadence.flush_bytes,
            adopted.cadence.checkpoint_updates,
            adopted.cadence.checkpoint_interval_ms
        ),
        (60_000, 1 << 20, 100, 300_000)
    );
    let report = adopted.report();
    assert!(report.adopted_defaults);
    assert_eq!(
        (report.approval, APPROVAL),
        (
            "adopted-by-owner-delegation-2026-09-26",
            "adopted-by-owner-delegation-2026-09-26"
        )
    );

    // An explicit file names every value; other values are reported as the operator's, and
    // nothing in the file can mark them adopted.
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("config.json");
    let explicit = json!({"schema": CONFIG_SCHEMA,
        "limits": {"pause_entries": 2, "pause_bytes": 1 << 20, "max_entries": 3, "max_bytes": 1 << 20},
        "cadence": {"flush_interval_ms": 1000, "flush_bytes": 4096, "checkpoint_updates": 10,
            "checkpoint_interval_ms": 5000}});
    fs::write(&file, explicit.to_string()).unwrap();
    let config = Config::load(&file).unwrap();
    let report = config.report();
    assert!(!report.adopted_defaults);
    assert_eq!(report.approval, OVERRIDDEN);
    for (label, mutate) in [
        (
            "an approval field",
            (|v: &mut serde_json::Value| v["approval"] = "approved".into())
                as fn(&mut serde_json::Value),
        ),
        ("a missing cadence", |v| {
            v.as_object_mut().unwrap().remove("cadence");
        }),
        ("a missing limit", |v| {
            v["limits"].as_object_mut().unwrap().remove("max_bytes");
        }),
        ("a zero flush interval", |v| {
            v["cadence"]["flush_interval_ms"] = 0.into()
        }),
        ("a pause above the maximum", |v| {
            v["limits"]["pause_entries"] = 4.into()
        }),
        ("another schema", |v| {
            v["schema"] = "jope.progress.config/v0".into()
        }),
    ] {
        let mut v = explicit.clone();
        mutate(&mut v);
        fs::write(&file, v.to_string()).unwrap();
        assert!(Config::load(&file).is_err(), "{label} was accepted");
    }

    // The configured limits govern the product outbox: at the pause, new work is refused with
    // nothing written, while the outcome of work in flight is still admitted.
    let engine = Engine::new();
    let h = host();
    let clock = TestClock::at(1_000);
    let mut d = Driver::init(&h.state, &h.remote, config, manifest(), PUBLISHER, 1_000).unwrap();
    let update = update_unit();
    d.schedule(&update).unwrap();
    assert_eq!(d.outbox().pending_entries(), 2);
    let before = snapshot(h.state.root());
    let err = d
        .schedule(
            &spec("corpus-publication", "teaching-a")
                .work_id(RUN)
                .unwrap(),
        )
        .unwrap_err();
    assert!(
        matches!(err.downcast_ref::<Refused>(), Some(Refused::Paused { .. })),
        "{err:#}"
    );
    assert_eq!(snapshot(h.state.root()), before);
    let result = b"result".to_vec();
    fs::write(engine.root().join("runs/train-a/r.json"), &result).unwrap();
    let err = d
        .result(
            &update,
            1,
            engine.root(),
            Path::new("runs/train-a/r.json"),
            &sha256_hex(&result),
        )
        .unwrap_err();
    // Two more entries would pass the maximum of three: refused whole, nothing written.
    assert!(
        matches!(err.downcast_ref::<Refused>(), Some(Refused::Full { .. })),
        "{err:#}"
    );
    assert_eq!(snapshot(h.state.root()), before);
    // A flush drains the outbox, and the refused work is then admitted.
    clock.set(1_000 + 1_000);
    let t = d.tick(&clock).unwrap();
    assert!(t.flushed, "{t:?}");
    d.result(
        &update,
        1,
        engine.root(),
        Path::new("runs/train-a/r.json"),
        &sha256_hex(&result),
    )
    .unwrap();
}

#[test]
fn hwm_decisions_link_through_this_hosts_ledger_and_flush_at_once() {
    let engine = Engine::new();
    let h = host();
    let clock = TestClock::at(1_000);
    let update = update_unit();
    let mut d = Driver::init(
        &h.state,
        &h.remote,
        Config::adopted(),
        manifest(),
        PUBLISHER,
        clock.now_ms(),
    )
    .unwrap();
    d.schedule(&update).unwrap();
    let result = b"fitted candidate".to_vec();
    fs::write(engine.root().join("runs/train-a/candidate.json"), &result).unwrap();
    d.result(
        &update,
        1,
        engine.root(),
        Path::new("runs/train-a/candidate.json"),
        &sha256_hex(&result),
    )
    .unwrap();
    let (record, blob) = super::fixture::hwm_record("cut-a", 0.5);
    let report = hwm::ingest(
        &h.state,
        None,
        vec![Candidate {
            record,
            blob,
            chain_from_source: false,
        }],
        json!({"test": "w8-product"}),
    )
    .unwrap();
    let batch = report.batch.unwrap();

    // Refusals write nothing.
    let before = durable(&h);
    for (label, attempt, path, seq) in [
        (
            "a batch this ledger does not hold",
            1,
            "2026/09/01K62ZZZZZZZZZZZZZZZZZZZZZ.jsonl",
            0,
        ),
        (
            "an event the batch does not hold",
            1,
            batch.path.as_str(),
            5,
        ),
        ("an attempt that never completed", 2, batch.path.as_str(), 0),
    ] {
        assert!(
            d.decision(&update, attempt, path, seq, &clock).is_err(),
            "{label} was linked"
        );
        assert_eq!(durable(&h), before, "{label}");
    }

    // The initialization flushes at once and the committed decision verifies against the ledger.
    let s = d.decision(&update, 1, &batch.path, 0, &clock).unwrap();
    assert!(matches!(
        &s.due,
        Some(FlushDue::Urgent { requests }) if requests == &[Urgent::HwmChange]
    ));
    assert_eq!(advanced(&s), 0);
    let state = reusable(&h.remote, RUN).unwrap().unwrap();
    let decisions = state.graph.all::<HwmDecision>();
    assert_eq!(decisions.len(), 1);
    decisions[0]
        .value()
        .verify_against(&crate::ledger::Ledger::open(h.state.root(), "hwm"))
        .unwrap();
    assert_eq!(
        decisions[0].value().transition,
        crate::hwm::Transition::Initialized
    );
}

#[test]
fn a_revoked_engine_publication_is_never_appended() {
    let engine = Engine::new();
    let h = host();
    let clock = TestClock::at(1_000);
    let mut d = Driver::init(
        &h.state,
        &h.remote,
        Config::adopted(),
        manifest(),
        PUBLISHER,
        clock.now_ms(),
    )
    .unwrap();
    let records = fs::read(engine.corpus().join("records.bin")).unwrap();
    let repo = tempfile::tempdir().unwrap();
    crate::revocation::testing::revoke(
        &h.state,
        repo.path(),
        &[("records", &records, &[])],
        &["records"],
    );
    let before = durable(&h);
    let error = format!("{:#}", d.corpus(&engine.corpus(), &clock).unwrap_err());
    assert!(error.contains("RevokedEvidence"), "{error}");
    assert_eq!(durable(&h), before, "a revoked publication was appended");
    assert_eq!(d.staged(), 0);
}

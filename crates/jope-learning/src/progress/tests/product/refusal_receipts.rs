#[test]
fn engine_events_that_do_not_verify_append_nothing_and_move_nothing() {
    type Case = (
        &'static str,
        fn(&Engine, &mut Driver<'_>, &TestClock) -> anyhow::Result<Step>,
    );
    let cases: Vec<Case> = vec![
        ("a corpus still in staging", |e, d, c| {
            let to = e.root().join("corpora/.teaching-b.staging");
            write_corpus(&to, true, "b");
            d.corpus(&to, c)
        }),
        ("a retired corpus build", |e, d, c| {
            let to = e.root().join("corpora/teaching-b.retired-1-2-0");
            write_corpus(&to, true, "b");
            d.corpus(&to, c)
        }),
        ("a symlinked corpus directory", |e, d, c| {
            let to = e.root().join("corpora/teaching-link");
            std::os::unix::fs::symlink(e.corpus(), &to).unwrap();
            d.corpus(&to, c)
        }),
        ("a diagnostic corpus", |e, d, c| {
            let to = e.root().join("corpora/teaching-b");
            write_corpus(&to, false, "b");
            d.corpus(&to, c)
        }),
        (
            "a corpus whose records changed after publication",
            |e, d, c| {
                let to = e.root().join("corpora/teaching-b");
                write_corpus(&to, true, "b");
                fs::write(to.join("records.bin"), b"bbbbbbbbbbbX").unwrap();
                d.corpus(&to, c)
            },
        ),
        ("a corpus with truncated records", |e, d, c| {
            let to = e.root().join("corpora/teaching-b");
            write_corpus(&to, true, "b");
            // Re-bind index and receipt to four records the payload does not hold.
            let mut index: serde_json::Value =
                serde_json::from_slice(&fs::read(to.join("index.json")).unwrap()).unwrap();
            index["records"] = 4.into();
            let index = pretty(&index);
            fs::write(to.join("index.json"), &index).unwrap();
            let mut prov: serde_json::Value =
                serde_json::from_slice(&fs::read(to.join("provenance.json")).unwrap()).unwrap();
            prov["index_sha256"] = sha256_hex(&index).into();
            fs::write(to.join("provenance.json"), pretty(&prov)).unwrap();
            d.corpus(&to, c)
        }),
        (
            "a corpus whose publication request names other contexts",
            |e, d, c| {
                let to = e.root().join("corpora/teaching-b");
                write_corpus(&to, true, "b");
                let mut p: serde_json::Value =
                    serde_json::from_slice(&fs::read(to.join("publication.json")).unwrap())
                        .unwrap();
                p["contexts_sha256"] = sha256_hex(b"other contexts").into();
                fs::write(to.join("publication.json"), pretty(&p)).unwrap();
                d.corpus(&to, c)
            },
        ),
        ("latest.json naming another run's checkpoint", |e, d, c| {
            e.publish(200);
            let latest = e.root().join("runs/train-a/latest.json");
            let mut v: serde_json::Value =
                serde_json::from_slice(&fs::read(&latest).unwrap()).unwrap();
            v["checkpoint"] = "runs/other/checkpoints/step-200".into();
            fs::write(&latest, pretty(&v)).unwrap();
            d.checkpoint(e.root(), &e.run(), &e.corpus(), 0, c)
        }),
        (
            "a checkpoint object changed after publication",
            |e, d, c| {
                e.publish(200);
                fs::write(
                    e.root().join("runs/train-a/checkpoints/step-200/model.ot"),
                    b"x",
                )
                .unwrap();
                d.checkpoint(e.root(), &e.run(), &e.corpus(), 0, c)
            },
        ),
        (
            "a checkpoint manifest latest.json did not publish",
            |e, d, c| {
                e.publish(200);
                let state = e
                    .root()
                    .join("runs/train-a/checkpoints/step-200/state.json");
                let mut v: serde_json::Value =
                    serde_json::from_slice(&fs::read(&state).unwrap()).unwrap();
                v["global_batch"] = 4096.into();
                fs::write(&state, pretty(&v)).unwrap();
                d.checkpoint(e.root(), &e.run(), &e.corpus(), 0, c)
            },
        ),
        ("a checkpoint trained on another corpus", |e, d, c| {
            let dir = e.write_checkpoint(200, &sha256_hex(b"another corpus index"));
            let state = fs::read(dir.join("state.json")).unwrap();
            fs::write(
                e.root().join("runs/train-a/latest.json"),
                pretty(&json!({"checkpoint": "runs/train-a/checkpoints/step-200",
                    "state_sha256": sha256_hex(&state), "step": 200})),
            )
            .unwrap();
            d.checkpoint(e.root(), &e.run(), &e.corpus(), 0, c)
        }),
        ("a symlinked checkpoint directory", |e, d, c| {
            let real = e.write_checkpoint(200, &e.ids.index);
            let moved = e.root().join("elsewhere-step-200");
            fs::rename(&real, &moved).unwrap();
            std::os::unix::fs::symlink(&moved, &real).unwrap();
            let state = fs::read(moved.join("state.json")).unwrap();
            fs::write(
                e.root().join("runs/train-a/latest.json"),
                pretty(&json!({"checkpoint": "runs/train-a/checkpoints/step-200",
                    "state_sha256": sha256_hex(&state), "step": 200})),
            )
            .unwrap();
            d.checkpoint(e.root(), &e.run(), &e.corpus(), 0, c)
        }),
        ("an incomplete recovery receipt", |e, d, _| {
            let path = e.root().join("receipt.json");
            fs::write(
                &path,
                pretty(
                    &json!({"schema": engine::RECOVERY_SCHEMA, "completed": false,
                    "published_step": 100, "moves": [{"name": "step-130", "moved": true}]}),
                ),
            )
            .unwrap();
            d.recovery(&update_unit(), 1, &path)
        }),
        ("a recovery for work the run never planned", |e, d, _| {
            let path = e.root().join("receipt.json");
            fs::write(
                &path,
                pretty(
                    &json!({"schema": engine::RECOVERY_SCHEMA, "completed": true,
                    "published_step": 100, "moves": [{"name": "step-130", "moved": true}]}),
                ),
            )
            .unwrap();
            d.recovery(&"0".repeat(64), 1, &path)
        }),
        (
            "a result whose bytes are not the published digest",
            |e, d, _| {
                fs::write(e.root().join("runs/train-a/result.json"), b"result").unwrap();
                d.result(
                    &update_unit(),
                    1,
                    e.root(),
                    Path::new("runs/train-a/result.json"),
                    &sha256_hex(b"another result"),
                )
            },
        ),
        ("a result path that leaves the engine root", |e, d, _| {
            fs::write(e.root().join("outside.json"), b"result").unwrap();
            d.result(
                &update_unit(),
                1,
                e.root(),
                Path::new("runs/../outside.json"),
                &sha256_hex(b"result"),
            )
        }),
        ("a symlinked result", |e, d, _| {
            fs::write(e.root().join("outside.json"), b"result").unwrap();
            std::os::unix::fs::symlink(
                e.root().join("outside.json"),
                e.root().join("runs/train-a/result.json"),
            )
            .unwrap();
            d.result(
                &update_unit(),
                1,
                e.root(),
                Path::new("runs/train-a/result.json"),
                &sha256_hex(b"result"),
            )
        }),
    ];
    for (label, run_case) in cases {
        let engine = Engine::new();
        let h = host();
        let clock = TestClock::at(5_000_000);
        let mut d = committed_update_zero(&h, &engine, &clock);
        let before = durable(&h);
        let head = d.head().cloned();
        let result = run_case(&engine, &mut d, &clock);
        assert!(result.is_err(), "{label} was accepted: {:?}", result.ok());
        assert_eq!(durable(&h), before, "{label}: something durable changed");
        assert_eq!(d.head().cloned(), head, "{label}");
        assert_eq!(d.staged(), 0, "{label}");
        drop(d);
    }

    // An unpublished checkpoint directory is never mapped, however new: only latest.json names
    // progress.
    let engine = Engine::new();
    let h = host();
    let clock = TestClock::at(5_000_000);
    let mut d = committed_update_zero(&h, &engine, &clock);
    engine.write_checkpoint(300, &engine.ids.index);
    let before = durable(&h);
    let s = d
        .checkpoint(engine.root(), &engine.run(), &engine.corpus(), 0, &clock)
        .unwrap();
    assert_eq!(s.checkpoint, Some(CheckpointStep::Nothing));
    assert_eq!(durable(&h), before);

    // Positive control: the next published checkpoint maps as update 1 and advances.
    engine.publish(200);
    let s = d
        .checkpoint(engine.root(), &engine.run(), &engine.corpus(), 0, &clock)
        .unwrap();
    assert_eq!(mapped(&s).1, 1);
    assert_eq!(advanced(&s), 2);
    // A checkpoint that does not advance past the last update is refused by the mapper itself.
    let corpus = engine::corpus(RUN, &engine.corpus(), None, 1 << 30).unwrap();
    assert!(engine::checkpoint(
        engine.root(),
        &engine.run(),
        &corpus,
        d.previous(),
        0,
        1 << 30
    )
    .is_err());
}

#[test]
fn a_failed_flush_keeps_progress_owed_and_claims_nothing() {
    let engine = Engine::new();
    let h = host();
    let clock = TestClock::at(1_000);
    let faulty = Faulty::new(&h.remote).fault_put(2, Fault::Disconnect);
    let mut d = Driver::init(
        &h.state,
        &faulty,
        Config::adopted(),
        manifest(),
        PUBLISHER,
        clock.now_ms(),
    )
    .unwrap();
    d.schedule(&update_unit()).unwrap();
    let s = d.corpus(&engine.corpus(), &clock).unwrap();
    match &s.publication {
        Some(Publication::Blocked { reason, .. }) => {
            assert!(reason.contains("delivery"), "{reason}")
        }
        other => panic!("expected a blocked publication, got {other:?}"),
    }
    assert!(!s.flushed && s.staged == 2 && d.head().is_none());
    assert!(h.remote.get(&head_key(RUN).unwrap()).unwrap().is_none());
    assert!(d.outbox().pending_entries() > 0);
    // The urgency is kept: the next tick retries and commits without any new event.
    let t = d.tick(&clock).unwrap();
    assert!(matches!(
        &t.due,
        Some(FlushDue::Urgent { requests }) if requests.contains(&Urgent::CorpusFreeze)
    ));
    assert_eq!(advanced(&t), 0);
    assert!(t.flushed && d.outbox().pending_entries() == 0);
    assert_eq!(t.tick.unwrap().durable_through, d.outbox().records());

    // A restart with appended but uncommitted progress commits it at once.
    let s = d.schedule(
        &spec("corpus-publication", "teaching-a")
            .work_id(RUN)
            .unwrap(),
    );
    s.unwrap();
    drop(d);
    let mut d = Driver::open(
        &h.state,
        &h.remote,
        Config::adopted(),
        RUN,
        PUBLISHER,
        clock.now_ms(),
    )
    .unwrap();
    assert_eq!(d.staged(), 1);
    assert_eq!(advanced(&d.tick(&clock).unwrap()), 1);
}

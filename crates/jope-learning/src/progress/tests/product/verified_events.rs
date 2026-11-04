#[test]
fn engine_events_reach_the_committed_graph_through_the_product_driver() {
    let engine = Engine::new();
    // The engine's own verifier accepts the fixture as a published corpus.
    jope_parallel::corpus_publication::verify(
        &engine.corpus(),
        &engine.ids.contexts,
        &engine.ids.vocabulary,
    )
    .unwrap();
    let h = host();
    let clock = TestClock::at(1_000_000);
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
    assert!(d.head().is_none());

    // Attempt 1 of the update is scheduled. Nothing is due yet: the flush interval is 60 s.
    let s = d.schedule(&update).unwrap();
    assert_eq!((s.appended.len(), s.staged), (1, 1));
    let t = d.tick(&clock).unwrap();
    assert!(t.due.is_none() && t.publication.is_none(), "{t:?}");

    // The corpus freeze flushes at once: the corpus and the scheduled event commit at height 0.
    let s = d.corpus(&engine.corpus(), &clock).unwrap();
    assert_eq!(
        s.appended.len(),
        7,
        "six payload files and the corpus manifest"
    );
    assert!(matches!(
        &s.due,
        Some(FlushDue::Urgent { requests }) if requests == &[Urgent::CorpusFreeze]
    ));
    assert_eq!(advanced(&s), 0);
    assert!(s.flushed && s.staged == 0);

    // Engine checkpoints at steps 40 and 80 are counted, not published as W8 updates.
    let root = engine.root().to_path_buf();
    let run = engine.run();
    let corpus = engine.corpus();
    for (step, at) in [(40, 1_010_000), (80, 1_020_000)] {
        engine.publish(step);
        clock.set(at);
        let s = d.checkpoint(&root, &run, &corpus, 0, &clock).unwrap();
        assert_eq!(
            s.checkpoint,
            Some(CheckpointStep::Deferred {
                step,
                uncheckpointed: step
            })
        );
        assert!(s.appended.is_empty() && s.publication.is_none());
    }
    // Step 100: the update trigger fires, and update 0 covers the 100 steps.
    engine.publish(100);
    clock.set(1_030_000);
    let s = d.checkpoint(&root, &run, &corpus, 0, &clock).unwrap();
    assert_eq!(
        mapped(&s),
        (
            100,
            0,
            100 * GLOBAL_BATCH,
            CheckpointDue::Updates {
                uncheckpointed: 100
            }
        )
    );
    assert_eq!(advanced(&s), 1);
    assert!(s.flushed);
    let checkpointed_at = clock.now_ms();

    // The ranks die after step 100: an unpublished checkpoint and a staging directory are left.
    // The engine's own recovery quarantines both and writes its receipt.
    engine.write_checkpoint(130, &engine.ids.index);
    let staging = root.join("runs/train-a/checkpoints/.step-140-4242");
    fs::create_dir_all(&staging).unwrap();
    fs::write(staging.join("model.ot"), b"partial").unwrap();
    jope_parallel::checkpoint_recovery::reconcile(&root, &root.join(&run)).unwrap();
    let receipt = fs::read_dir(root.join("runs/train-a/checkpoint-recovery"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path()
        .join("receipt.json");
    let s = d.recovery(&update, 1, &receipt).unwrap();
    assert_eq!(s.appended.len(), 1);
    let s = d.retry(&update, 2).unwrap();
    assert_eq!((s.appended.len(), s.staged), (1, 2));
    // The interval flush commits the failure and its retry: not at 59.999 s, at 60 s.
    clock.set(checkpointed_at + 59_999);
    assert!(d.tick(&clock).unwrap().publication.is_none());
    clock.set(checkpointed_at + 60_000);
    let t = d.tick(&clock).unwrap();
    assert!(matches!(t.due, Some(FlushDue::Interval { .. })), "{t:?}");
    assert_eq!(advanced(&t), 2);

    // Step 150, five minutes after update 0 was committed: the interval trigger fires.
    engine.publish(150);
    clock.set(checkpointed_at + 300_000);
    let s = d.checkpoint(&root, &run, &corpus, 0, &clock).unwrap();
    assert_eq!(
        mapped(&s),
        (
            150,
            1,
            50 * GLOBAL_BATCH,
            CheckpointDue::Interval { waited_ms: 300_000 }
        )
    );
    assert_eq!(advanced(&s), 3);

    // The retried attempt's result is a no-replace engine publication.
    let result = b"{\"schema\":\"parallel.training_complete.v1\",\"steps\":180}".to_vec();
    fs::write(root.join("runs/train-a/complete-attempt-2.json"), &result).unwrap();
    let s = d
        .result(
            &update,
            2,
            &root,
            Path::new("runs/train-a/complete-attempt-2.json"),
            &sha256_hex(&result),
        )
        .unwrap();
    assert_eq!(s.appended.len(), 2, "the result blob and its completion");

    // Controlled shutdown at step 180: the remaining 30 steps become update 2, and everything
    // pending, the completion included, commits.
    engine.publish(180);
    clock.set(checkpointed_at + 310_000);
    let s = d.shutdown(&root, &run, &corpus, 0, &clock).unwrap();
    assert_eq!(
        mapped(&s),
        (
            180,
            2,
            30 * GLOBAL_BATCH,
            CheckpointDue::Shutdown { uncheckpointed: 30 }
        )
    );
    assert_eq!(advanced(&s), 4);
    assert!(s.flushed && s.staged == 0);
    assert_eq!(d.outbox().pending_entries(), 0);

    // The committed graph: three consecutive updates on one corpus, and the census.
    let state = reusable(&h.remote, RUN).unwrap().unwrap();
    let mut checkpoints: Vec<_> = state.graph.all::<Checkpoint>();
    checkpoints.sort_by_key(|k| k.value().update_index);
    assert_eq!(
        checkpoints
            .iter()
            .map(|k| k.value().update_index)
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    for pair in checkpoints.windows(2) {
        assert_eq!(pair[1].value().parent.as_deref(), Some(pair[0].digest()));
    }
    let mut rows: Vec<(u64, u64)> = state
        .graph
        .all::<Exposure>()
        .iter()
        .map(|e| (e.value().update_index, e.value().consumed_rows))
        .collect();
    rows.sort();
    assert_eq!(rows, [(0, 800), (1, 400), (2, 240)]);
    let (summary, census) = summarize(&state).unwrap();
    assert_eq!(
        (summary.corpora, summary.checkpoints, summary.consumed_rows),
        (1, 3, 180 * GLOBAL_BATCH)
    );
    assert_eq!((census.counts.completed, census.counts.unscheduled), (1, 1));
    assert_eq!(census.counts.retries, 1);
    assert_eq!(census.counts.failures.get("worker-lost"), Some(&1));
    let unit = census.units.iter().find(|u| u.work_id == update).unwrap();
    assert!(matches!(
        unit.disposition,
        Disposition::Completed { attempt: 2, .. }
    ));
    // The quarantined step-130 checkpoint never became progress.
    let quarantined = fs::read(receipt.parent().unwrap().join("step-130/state.json")).unwrap();
    assert!(h
        .remote
        .get(&blob_key(&sha256_hex(&quarantined)))
        .unwrap()
        .is_none());

    // Restart: a reopened driver re-derives its state, and the same engine state maps to
    // nothing new.
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
    assert_eq!(d.previous().unwrap().step, 180);
    assert_eq!(d.previous().unwrap().checkpoint.value().update_index, 2);
    assert_eq!((d.staged(), d.head().unwrap().height), (0, 4));
    let s = d.checkpoint(&root, &run, &corpus, 0, &clock).unwrap();
    assert_eq!(s.checkpoint, Some(CheckpointStep::Nothing));
    let s = d.corpus(&engine.corpus(), &clock).unwrap();
    assert!(s.appended.is_empty() && s.existing.len() == 7 && s.publication.is_none());
}

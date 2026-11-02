// ------------------------------------------------- W7-T02 status / verify / cancel

/// Drive an in-process rendezvous to a persisted admission.
fn admit(
    c: &mut Coordinator<FakeSupervisor>,
    r: &Run,
    now: u64,
) -> (controller::PersistedAdmission, Vec<Hello>) {
    let (admission, hellos) = group_admission(c, r, now);
    (c.record_admission(admission, now).unwrap(), hellos)
}
/// Drive an in-process rendezvous to a signed full-group admission, not yet journaled.
fn group_admission(
    c: &mut Coordinator<FakeSupervisor>,
    r: &Run,
    now: u64,
) -> (rendezvous::GroupAdmission, Vec<Hello>) {
    let mut rv = c
        .open_rendezvous(
            &r.keys.coordinator,
            "fixture-coordinator-boot",
            &[1; 16],
            now,
        )
        .unwrap();
    let closure =
        rendezvous::observed_closure_digest(&r.root, &r.contract.artifact_closure).unwrap();
    let mut hellos = Vec::new();
    for rank in 0..3 {
        let body = hello_body(&r.contract, &r.sha, rv.challenge(), rank, now, &closure);
        let hello = rendezvous::sign_hello(&r.keys.ranks[rank as usize], body).unwrap();
        assert_eq!(
            rv.offer(&serde_json::to_vec(&hello).unwrap(), now).unwrap(),
            rank
        );
        hellos.push(hello);
    }
    (rv.admit(&r.keys.coordinator, now).unwrap().unwrap(), hellos)
}
fn deliver(
    c: &mut Coordinator<FakeSupervisor>,
    r: &Run,
    persisted: &controller::PersistedAdmission,
    hellos: &[Hello],
    now: u64,
) {
    let mut connected = Vec::new();
    let mut peers = Vec::new();
    for rank in 0..3 {
        let (a, b) = UnixStream::pair().unwrap();
        connected.push(Connected {
            global_rank: rank,
            stream: Box::new(DeadlineSocket::new(a)),
        });
        peers.push(b);
    }
    c.deliver_admission(persisted, &mut connected, now).unwrap();
    for (rank, mut peer) in peers.into_iter().enumerate() {
        let frame =
            rendezvous::read_frame(&mut peer, r.contract.rendezvous.max_message_bytes).unwrap();
        let member =
            rendezvous::verify_admission(&r.contract, &r.sha, &hellos[rank], &frame, now).unwrap();
        assert_eq!(member.global_rank(), rank as u32);
        assert_eq!(member.communicator_bootstrap(), [0xab; 128]);
    }
}
fn states(dir: &Path) -> Vec<RankState> {
    controller::status(dir)
        .unwrap()
        .ranks
        .iter()
        .map(|r| r.state)
        .collect()
}
fn live_states(c: &Coordinator<FakeSupervisor>) -> Vec<RankState> {
    c.status().unwrap().ranks.iter().map(|r| r.state).collect()
}

#[test]
fn status_distinguishes_all_seven_states() {
    use RankState::*;
    let now = now_ms();
    // scheduled, unknown, admitted, running, committed
    let r = planned(now);
    assert_eq!(states(&r.dir), vec![Scheduled; 3]);
    let fake = FakeSupervisor::default();
    let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), now).unwrap();
    c.preflight(&serde_json::to_vec(&census(&r.contract, now)).unwrap(), now)
        .unwrap();
    c.record_dispatch_intent(0, now).unwrap();
    assert_eq!(live_states(&c), vec![Unknown, Scheduled, Scheduled]);
    assert!(controller::status(&r.dir).unwrap().coordinator_live);
    c.dispatch_all(now).unwrap();
    assert_eq!(live_states(&c), vec![Scheduled; 3]);
    let (persisted, hellos) = admit(&mut c, &r, now);
    assert_eq!(live_states(&c), vec![Admitted; 3]);
    deliver(&mut c, &r, &persisted, &hellos, now);
    assert_eq!(live_states(&c), vec![Running; 3]);
    assert_eq!(states(&r.dir), vec![Running; 3]);
    for rank in 0..3u32 {
        let id = rendezvous::dispatch_id(&r.sha, 1, rank).unwrap();
        let retained = result(
            &r.contract,
            &r.sha,
            persisted.admission_sha256(),
            rank,
            &r.keys.ranks[rank as usize],
        );
        fake.set(
            &id,
            Liveness::Exited {
                code: Some(0),
                signal: None,
            },
            Some(retained),
        );
    }
    let done = c.poll(now).unwrap();
    assert_eq!(done.outcome, Some(Outcome::Committed));
    assert_eq!(done.committed_updates_total, 60_003);
    drop(c);
    assert_eq!(states(&r.dir), vec![Committed; 3]);
    let v = controller::verify(&r.dir, Some(&r.sha)).unwrap();
    assert_eq!(v.results_verified, 3);
    assert!(!v.numerical_work_started);

    // unknown: launched work with no live coordinator holding the run
    let r2 = planned(now);
    let fake2 = FakeSupervisor::default();
    let mut c2 = Coordinator::open(&r2.dir, &r2.sha, fake2.clone(), now).unwrap();
    c2.preflight(
        &serde_json::to_vec(&census(&r2.contract, now)).unwrap(),
        now,
    )
    .unwrap();
    c2.dispatch_all(now).unwrap();
    let (persisted2, hellos2) = admit(&mut c2, &r2, now);
    deliver(&mut c2, &r2, &persisted2, &hellos2, now);
    drop(c2);
    let s = controller::status(&r2.dir).unwrap();
    assert!(!s.coordinator_live);
    assert_eq!(
        s.ranks.iter().map(|x| x.state).collect::<Vec<_>>(),
        vec![Unknown; 3]
    );

    // failed + drained: one rank exits nonzero; the whole group drains
    let mut c2 = Coordinator::open(&r2.dir, &r2.sha, fake2.clone(), now).unwrap();
    assert!(c2.needs_reconcile());
    assert!(
        c2.poll(now).is_err(),
        "no supervision step before reconcile"
    );
    drop(c2);
    let r3 = planned(now);
    let fake3 = FakeSupervisor::default();
    let mut c3 = Coordinator::open(&r3.dir, &r3.sha, fake3.clone(), now).unwrap();
    c3.preflight(
        &serde_json::to_vec(&census(&r3.contract, now)).unwrap(),
        now,
    )
    .unwrap();
    c3.dispatch_all(now).unwrap();
    let (persisted3, hellos3) = admit(&mut c3, &r3, now);
    deliver(&mut c3, &r3, &persisted3, &hellos3, now);
    let id1 = rendezvous::dispatch_id(&r3.sha, 1, 1).unwrap();
    fake3.set(
        &id1,
        Liveness::Exited {
            code: Some(1),
            signal: None,
        },
        None,
    );
    let s = c3.poll(now).unwrap();
    assert_eq!(s.outcome, Some(Outcome::Drained));
    assert_eq!(s.committed_updates_total, 60_000);
    assert_eq!(
        s.ranks.iter().map(|x| x.state).collect::<Vec<_>>(),
        vec![Drained, Failed, Drained]
    );
    assert_eq!(fake3.running(), 0);
    controller::verify(&r3.dir, Some(&r3.sha)).unwrap();
}

#[test]
fn status_and_verify_are_read_only_and_start_nothing() {
    let now = now_ms();
    let r = planned(now);
    let fake = FakeSupervisor::default();
    let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), now).unwrap();
    c.preflight(&serde_json::to_vec(&census(&r.contract, now)).unwrap(), now)
        .unwrap();
    c.dispatch_all(now).unwrap();
    let (persisted, hellos) = admit(&mut c, &r, now);
    deliver(&mut c, &r, &persisted, &hellos, now);
    let launches = fake.0.lock().unwrap().launches;
    drop(c);
    let before = tree(&r.root);
    for _ in 0..2 {
        controller::status(&r.dir).unwrap();
        let v = controller::verify(&r.dir, Some(&r.sha)).unwrap();
        assert!(!v.numerical_work_started);
    }
    let exe = env!("CARGO_BIN_EXE_jope-native-loop");
    for args in [
        vec!["status", "--run-dir", r.dir.to_str().unwrap()],
        vec![
            "verify",
            "--run-dir",
            r.dir.to_str().unwrap(),
            "--plan-sha256",
            &r.sha,
        ],
    ] {
        let out = Command::new(exe).args(&args).output().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    assert_eq!(
        tree(&r.root),
        before,
        "status/verify changed the run directory"
    );
    let s = fake.0.lock().unwrap();
    assert_eq!((s.launches, s.terminations), (launches, 0));
    drop(s);
    let absent = r.root.join("never-planned");
    assert!(controller::status(&absent).is_err());
    assert!(controller::verify(&absent, None).is_err());
    assert!(!absent.exists());
    assert!(controller::verify(&r.dir, Some(&"0".repeat(64))).is_err());
}

#[test]
fn verify_detects_tampering_with_plan_journal_or_results() {
    let now = now_ms();
    let r = planned(now);
    let fake = FakeSupervisor::default();
    let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), now).unwrap();
    c.preflight(&serde_json::to_vec(&census(&r.contract, now)).unwrap(), now)
        .unwrap();
    c.dispatch_all(now).unwrap();
    let (persisted, hellos) = admit(&mut c, &r, now);
    deliver(&mut c, &r, &persisted, &hellos, now);
    for rank in 0..3u32 {
        let id = rendezvous::dispatch_id(&r.sha, 1, rank).unwrap();
        let retained = result(
            &r.contract,
            &r.sha,
            persisted.admission_sha256(),
            rank,
            &r.keys.ranks[rank as usize],
        );
        fake.set(
            &id,
            Liveness::Exited {
                code: Some(0),
                signal: None,
            },
            Some(retained),
        );
    }
    assert_eq!(c.poll(now).unwrap().outcome, Some(Outcome::Committed));
    drop(c);
    controller::verify(&r.dir, Some(&r.sha)).unwrap();
    let result_path = r.dir.join("results/rank-1/result.json");
    let original = fs::read(&result_path).unwrap();
    fs::write(&result_path, b"{\"fixture_rank\":9}").unwrap();
    assert!(controller::verify(&r.dir, None).is_err());
    fs::write(&result_path, &original).unwrap();
    controller::verify(&r.dir, None).unwrap();
    let record = r.dir.join("journal/00000004.json");
    let original = fs::read(&record).unwrap();
    let mut edited = original.clone();
    let at = edited.iter().position(|b| *b == b'1').unwrap();
    edited[at] = b'2';
    fs::write(&record, &edited).unwrap();
    assert!(controller::verify(&r.dir, None).is_err());
    assert!(controller::status(&r.dir).is_err());
    fs::write(&record, &original).unwrap();
    let plan = r.dir.join("plan.json");
    let frozen = fs::read(&plan).unwrap();
    let mut other = r.contract.clone();
    other.run = "w7-fixture-other".into();
    resign(&mut other, &r.keys);
    fs::write(&plan, bytes(&other)).unwrap();
    assert!(
        controller::verify(&r.dir, None).is_err(),
        "journal bound to another plan"
    );
    fs::write(&plan, &frozen).unwrap();
    controller::verify(&r.dir, Some(&r.sha)).unwrap();
}

#[test]
fn cancel_is_idempotent_and_bound_to_the_plan() {
    let now = now_ms();
    // Before dispatch and with no live coordinator: terminal immediately.
    let r = planned(now);
    let before = tree(&r.dir);
    assert!(controller::cancel(&r.dir, &"f".repeat(64), "wrong plan", now).is_err());
    assert_eq!(tree(&r.dir), before, "a refused cancel wrote state");
    let first = controller::cancel(&r.dir, &r.sha, "operator stop", now).unwrap();
    assert!(!first.already_requested && first.journaled);
    assert_eq!(first.outcome, Some(Outcome::Cancelled));
    let after = tree(&r.dir);
    let second = controller::cancel(&r.dir, &r.sha, "a different reason", now + 5).unwrap();
    assert!(second.already_requested && !second.journaled);
    assert_eq!(second.request, first.request);
    assert_eq!(tree(&r.dir), after, "repeat cancel changed state");
    let s = controller::status(&r.dir).unwrap();
    assert!(s
        .ranks
        .iter()
        .all(|x| x.disposition == Some(Disposition::NeverStarted)));
    controller::verify(&r.dir, Some(&r.sha)).unwrap();

    // With a live coordinator: delivered, then the whole group is drained by poll.
    let r = planned(now);
    let fake = FakeSupervisor::default();
    let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), now).unwrap();
    c.preflight(&serde_json::to_vec(&census(&r.contract, now)).unwrap(), now)
        .unwrap();
    c.dispatch_all(now).unwrap();
    let journal = controller::status(&r.dir).unwrap().journal_records;
    let live = controller::cancel(&r.dir, &r.sha, "operator stop", now).unwrap();
    assert!(live.delivered_to_live_coordinator && !live.journaled);
    assert_eq!(controller::status(&r.dir).unwrap().journal_records, journal);
    assert!(controller::status(&r.dir).unwrap().cancel_requested);
    assert!(
        c.record_dispatch_intent(0, now).is_err(),
        "no dispatch once cancelled"
    );
    let s = c.poll(now).unwrap();
    assert_eq!(s.outcome, Some(Outcome::Cancelled));
    assert_eq!(fake.running(), 0);
    assert!(s.ranks.iter().all(|x| x.state == RankState::Drained));
    drop(c);
    let again = controller::cancel(&r.dir, &r.sha, "operator stop", now).unwrap();
    assert!(again.already_requested && !again.journaled);

    // Dispatched work and no live coordinator: the drain waits for reconciliation.
    let r = planned(now);
    let fake = FakeSupervisor::default();
    let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), now).unwrap();
    c.preflight(&serde_json::to_vec(&census(&r.contract, now)).unwrap(), now)
        .unwrap();
    c.dispatch_all(now).unwrap();
    drop(c);
    let pending = controller::cancel(&r.dir, &r.sha, "operator stop", now).unwrap();
    assert!(pending.drain_pending_reconcile && pending.outcome.is_none());
    let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), now).unwrap();
    assert!(c.needs_reconcile());
    assert!(c.record_dispatch_intent(0, now).is_err());
    let s = c.reconcile(now).unwrap();
    assert_eq!(s.outcome, Some(Outcome::Cancelled));
    assert_eq!(fake.running(), 0);
}

#[test]
fn execution_binds_to_the_planned_artifact() {
    let now = now_ms();
    let r = planned(now);
    let other = sha(b"some other plan");
    assert!(Coordinator::open(&r.dir, &other, FakeSupervisor::default(), now).is_err());
    let c = Coordinator::open(&r.dir, &r.sha, FakeSupervisor::default(), now).unwrap();
    assert!(
        Coordinator::open(&r.dir, &r.sha, FakeSupervisor::default(), now).is_err(),
        "one writer per run"
    );
    drop(c);
    // Dispatch before an admitted preflight is refused and starts nothing.
    let fake = FakeSupervisor::default();
    let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), now).unwrap();
    assert!(c.record_dispatch_intent(0, now).is_err());
    let mut stale = census(&r.contract, now);
    stale.hosts[0].observed_unix_ms = now - 120_000;
    assert!(c
        .preflight(&serde_json::to_vec(&stale).unwrap(), now)
        .is_err());
    assert!(c.record_dispatch_intent(0, now).is_err());
    assert!(c
        .open_rendezvous(&r.keys.coordinator, "boot", &[0; 16], now)
        .is_err());
    assert_eq!(fake.0.lock().unwrap().launches, 0);
    let s = c.status().unwrap();
    assert!(s.ranks.iter().all(|x| x.state == RankState::Scheduled));
    // A fresh preflight admits; a stale receipt no longer permits dispatch later.
    c.preflight(&serde_json::to_vec(&census(&r.contract, now)).unwrap(), now)
        .unwrap();
    assert!(c.record_dispatch_intent(0, now + 61_000).is_err());
    c.record_dispatch_intent(0, now).unwrap();
    assert_eq!(fake.0.lock().unwrap().launches, 0, "intent is not launch");
}

// ------------------------------------- review of #111: one group window everywhere

/// A plan whose rank-2 grant closes the group window at `close`, before the other grants
/// and the original deadline. The receipt freshness limit is at its maximum, so a
/// preflight at `close - 600_000` (the wall ceiling) is still fresh at `close`.
fn planned_early_close(now: u64) -> (Run, u64) {
    let (t, root) = tempdir();
    write_closure(&root);
    let k = keys(3);
    let mut c = contract(now, 3, &k);
    let close = now + 605_000;
    c.telemetry.max_age_ms = 600_000;
    c.participants[2].grant.body.expires_unix_ms = close;
    resign(&mut c, &k);
    assert!(c.participants[0].grant.body.expires_unix_ms > close);
    assert!(c.original_deadline_unix_ms > close);
    let dir = root.join("run");
    let report = controller::plan(&dir, &bytes(&c), now, None).unwrap();
    (
        Run {
            _t: t,
            root,
            dir,
            sha: report.plan_sha256,
            keys: k,
            contract: c,
        },
        close,
    )
}
fn preflighted_at(r: &Run, at: u64) -> (Coordinator<FakeSupervisor>, FakeSupervisor) {
    let fake = FakeSupervisor::default();
    let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), at).unwrap();
    c.preflight(&serde_json::to_vec(&census(&r.contract, at)).unwrap(), at)
        .unwrap();
    (c, fake)
}

#[test]
fn dispatch_stops_when_any_participant_grant_expires() {
    let now = now_ms();
    let (r, close) = planned_early_close(now);
    let (mut c, fake) = preflighted_at(&r, close - 600_000);
    // Rank 0's own grant and the deadline are still open, but rank 2's grant is not:
    // the synchronous group can no longer run, so nothing more is dispatched.
    assert!(
        c.record_dispatch_intent(0, close).is_err(),
        "dispatch accepted after a participant grant expired"
    );
    assert!(c.record_dispatch_intent(1, close + 1).is_err());
    assert_eq!(fake.0.lock().unwrap().launches, 0);
    // Positive control: one millisecond earlier the same dispatch is accepted.
    c.record_dispatch_intent(0, close - 1).unwrap();
    c.launch(0, close - 1).unwrap();
    assert!(c.record_dispatch_intent(1, close).is_err());
    assert!(c.launch(1, close).is_err());
    assert_eq!(fake.0.lock().unwrap().launches, 1);
    drop(c);
    controller::verify(&r.dir, Some(&r.sha)).unwrap();
}

#[test]
fn admission_and_delivery_stop_when_any_participant_grant_expires() {
    let now = now_ms();
    let (r, close) = planned_early_close(now);
    let p = close - 600_000;
    let (mut c, fake) = preflighted_at(&r, p);
    c.dispatch_all(p).unwrap();
    let (admission, _hellos) = group_admission(&mut c, &r, p);
    assert!(
        c.record_admission(admission.clone(), close).is_err(),
        "admission journaled after a participant grant expired"
    );
    assert_eq!(c.status().unwrap().ranks[0].state, RankState::Scheduled);
    // Positive control: journaled one millisecond earlier.
    let persisted = c.record_admission(admission, close - 1).unwrap();
    // Delivery is the moment ranks may start work: after the window closed it is
    // refused and the whole group is drained instead.
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
    assert!(
        c.deliver_admission(&persisted, &mut connected, close)
            .is_err(),
        "admission delivered after a participant grant expired"
    );
    for mut peer in peers {
        peer.set_read_timeout(Some(std::time::Duration::from_millis(50)))
            .unwrap();
        assert!(rendezvous::read_frame(&mut peer, 64 * 1024).is_err());
    }
    let s = c.status().unwrap();
    assert_eq!(
        (s.outcome, s.committed_updates_total),
        (Some(Outcome::Drained), 60_000)
    );
    assert_eq!(fake.running(), 0);
    drop(c);
    let journal: String = fs::read_dir(r.dir.join("journal"))
        .unwrap()
        .map(|e| fs::read_to_string(e.unwrap().path()).unwrap())
        .collect();
    assert!(journal.contains("\"cause\":\"grant_expired\""));
    assert!(!journal.contains("\"event\":\"admission_delivered\""));
    controller::verify(&r.dir, Some(&r.sha)).unwrap();
}

/// Re-chain the journal after `edit`, exactly as a producer would have written it.
fn rewrite_journal(dir: &Path, edit: impl Fn(&mut controller::JournalRecord)) {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir.join("journal"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    paths.sort();
    let mut previous: Option<String> = None;
    for path in paths {
        let mut record: controller::JournalRecord =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        edit(&mut record);
        record.previous_sha256 = previous.clone();
        let bytes = serde_json::to_vec(&record).unwrap();
        fs::write(&path, &bytes).unwrap();
        previous = Some(sha(&bytes));
    }
}
fn event_kind(r: &controller::JournalRecord) -> String {
    serde_json::to_value(&r.event).unwrap()["event"]
        .as_str()
        .unwrap()
        .to_string()
}

#[test]
fn verify_refuses_acknowledgements_or_commits_journaled_after_the_window_closed() {
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
    let expiry = r
        .contract
        .participants
        .iter()
        .map(|p| p.grant.body.expires_unix_ms)
        .min()
        .unwrap();
    let journal = r.dir.join("journal");
    let snapshot: Vec<(PathBuf, Vec<u8>)> = fs::read_dir(&journal)
        .unwrap()
        .map(|e| {
            let p = e.unwrap().path();
            let b = fs::read(&p).unwrap();
            (p, b)
        })
        .collect();
    let restore = || {
        for (p, b) in &snapshot {
            fs::write(p, b).unwrap();
        }
    };
    for (kind, gated) in [
        ("result_acknowledged", true),
        ("group_committed", true),
        ("dispatch_intent", true),
        ("group_admitted", true),
        ("admission_delivered", true),
        ("disposition", false),
        ("terminal", false),
    ] {
        rewrite_journal(&r.dir, |record| {
            if event_kind(record) == kind {
                record.recorded_unix_ms = expiry;
            }
        });
        let verdict = controller::verify(&r.dir, Some(&r.sha));
        assert_eq!(
            verdict.is_err(),
            gated,
            "{kind} journaled at grant expiry: {verdict:?}"
        );
        restore();
    }
    controller::verify(&r.dir, Some(&r.sha)).unwrap();
}

#[test]
fn the_plan_review_names_work_kinds_plainly() {
    let c = fixed_plan();
    let review = c.review("x");
    let kinds: Vec<&str> = review["work_census_wall_ms_by_kind"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        kinds,
        [
            "artifact_movement",
            "cleanup",
            "failure_allowance",
            "numerical",
            "verification"
        ]
    );
    // Wall time per kind is summed over every rank.
    assert_eq!(
        review["work_census_wall_ms_by_kind"]["numerical"],
        3 * 400_000
    );
}

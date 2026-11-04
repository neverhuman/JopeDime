use super::*;

// ------------------------------------------------------------------------ scenarios

#[test]
fn positive_control_commits_the_whole_group_exactly_once() {
    let h = Harness::new(["complete"; 3]);
    let mut c = h.open();
    let mut connected = h.start(&mut c);
    let mut next = 0;
    let s = supervise(&mut c, &mut connected, &mut next);
    assert_eq!(s.outcome, Some(Outcome::Committed));
    assert_eq!(s.committed_updates_total, 60_003);
    assert!(s.ranks.iter().all(|r| r.state == RankState::Committed));
    let journal = s.journal_records;
    drop(c);
    h.assert_no_orphans();
    // Reopening a terminal run neither restarts nor recommits anything.
    let mut again = h.open();
    assert!(!again.needs_reconcile());
    let s = again.reconcile(now_ms()).unwrap();
    assert_eq!(
        (s.journal_records, s.committed_updates_total),
        (journal, 60_003)
    );
    drop(again);
    let v = controller::verify(&h.dir, Some(&h.sha)).unwrap();
    assert_eq!(v.results_verified, 3);
    assert!(!v.numerical_work_started);
}

#[test]
fn killed_rank_drains_the_whole_group_with_zero_commits() {
    let h = Harness::new(["hold"; 3]);
    let mut c = h.open();
    let mut connected = h.start(&mut c);
    h.wait_admitted();
    let mut next = 0;
    for _ in 0..3 {
        next += 1;
        heartbeat(&mut connected, next);
        assert!(c.poll(now_ms()).unwrap().outcome.is_none());
    }
    let victim = h.pids()[1];
    unsafe { libc::kill(victim, libc::SIGKILL) };
    let s = supervise(&mut c, &mut connected, &mut next);
    assert_eq!(s.outcome, Some(Outcome::Drained));
    assert_eq!(s.committed_updates_total, 60_000);
    assert_eq!(
        h.dispositions(),
        vec![
            Some(Disposition::Drained),
            Some(Disposition::Failed),
            Some(Disposition::Drained)
        ]
    );
    assert!(!h.dir.join("results").exists(), "no result was committed");
    drop(c);
    h.assert_no_orphans();
    controller::verify(&h.dir, Some(&h.sha)).unwrap();
}

#[test]
fn severed_control_stops_every_rank_and_drains_the_group() {
    let h = Harness::new(["hold"; 3]);
    let mut c = h.open();
    let connected = h.start(&mut c);
    h.wait_admitted();
    let pids = h.pids();
    // Sever control: stop heartbeats while holding the channels open. Each rank stops
    // itself within the frozen heartbeat timeout; none is killed by the fixture.
    wait(|| pids.iter().all(|p| exited(*p)));
    let s = c.poll(now_ms()).unwrap();
    assert_eq!(s.outcome, Some(Outcome::Drained));
    assert!(s
        .ranks
        .iter()
        .all(|r| r.disposition == Some(Disposition::Drained)));
    assert!(
        journal_contains(&h.dir, "\"cause\":\"control_lost\""),
        "drain cause must be the lost control channel"
    );
    drop(connected);
    drop(c);
    h.assert_no_orphans();
    controller::verify(&h.dir, Some(&h.sha)).unwrap();
}

#[test]
fn coordinator_restart_reconciles_and_drains_with_original_deadline_and_counters() {
    let h = Harness::new(["hold"; 3]);
    let mut c = h.open();
    let mut connected = h.start(&mut c);
    h.wait_admitted();
    heartbeat(&mut connected, 1);
    drop(c); // coordinator crash: no drain ran, ranks keep their channels
    let s = controller::status(&h.dir).unwrap();
    assert!(!s.coordinator_live);
    assert!(s.ranks.iter().all(|r| r.state == RankState::Unknown));
    heartbeat(&mut connected, 2);
    let mut c = h.open();
    assert!(c.needs_reconcile());
    assert!(c.poll(now_ms()).is_err());
    assert!(c.record_dispatch_intent(0, now_ms()).is_err());
    let s = c.reconcile(now_ms()).unwrap();
    assert_eq!(s.outcome, Some(Outcome::Drained));
    assert_eq!(
        s.original_deadline_unix_ms,
        h.contract.original_deadline_unix_ms
    );
    assert_eq!(
        (s.committed_updates_at_start, s.committed_updates_total),
        (60_000, 60_000)
    );
    assert!(s
        .ranks
        .iter()
        .all(|r| r.disposition == Some(Disposition::Drained)));
    drop(c);
    drop(connected);
    h.assert_no_orphans();

    // A successor attempt keeps the run, deadline, topology and committed counters.
    let outcome = controller::attempt_outcome(&h.dir, &h.sha).unwrap();
    assert!(outcome.terminal);
    let mut next = h.contract.clone();
    next.attempt = 2;
    next.predecessor = Some(Predecessor {
        plan_sha256: h.sha.clone(),
        attempt: 1,
        journal_tip_sha256: outcome.journal_tip_sha256.clone(),
        committed_updates: outcome.committed_updates_total,
    });
    resign(&mut next, &h.keys);
    let successor = h.root.join("attempt-2");
    let now = now_ms();
    assert!(controller::plan(&successor, &bytes(&next), now, None).is_err());
    assert!(!successor.exists());
    for (name, edit) in [
        (
            "renewed deadline",
            Box::new(|c: &mut BoundedRunContract| c.original_deadline_unix_ms += 60_000)
                as Box<dyn Fn(&mut BoundedRunContract)>,
        ),
        (
            "topology change",
            Box::new(|c: &mut BoundedRunContract| {
                c.participants[2].device.as_mut().unwrap().uuid = "GPU-fixture-9".into()
            }),
        ),
        (
            "counter reset",
            Box::new(|c: &mut BoundedRunContract| {
                c.counters.committed_updates_at_start = 0;
                c.predecessor.as_mut().unwrap().committed_updates = 0;
            }),
        ),
    ] {
        let mut bad = next.clone();
        edit(&mut bad);
        resign(&mut bad, &h.keys);
        let refusal = BoundedRunContract::parse(&bytes(&bad))
            .and_then(|b| b.validate_successor(&h.contract, &outcome));
        assert!(
            matches!(refusal, Err(PlanRefusal::Invalid(_))),
            "{name}: {refusal:?}"
        );
        assert!(
            controller::plan(&successor, &bytes(&bad), now, Some(&h.dir)).is_err(),
            "{name}"
        );
        assert!(!successor.exists(), "{name}");
    }
    let planned = controller::plan(&successor, &bytes(&next), now, Some(&h.dir)).unwrap();
    let s = controller::status(&successor).unwrap();
    assert_eq!(
        s.original_deadline_unix_ms,
        h.contract.original_deadline_unix_ms
    );
    assert_eq!(s.committed_updates_at_start, 60_000);
    assert_eq!(s.plan_sha256, planned.plan_sha256);
}

#[test]
fn restart_commits_a_complete_unacknowledged_group_exactly_once() {
    let h = Harness::new(["complete"; 3]);
    let mut c = h.open();
    let mut connected = h.start(&mut c);
    h.wait_admitted();
    for id in 1..=3 {
        heartbeat(&mut connected, id);
    }
    let pids = h.pids();
    wait(|| pids.iter().all(|p| exited(*p)));
    drop(c); // crashed before acknowledging any completion
    let pending = controller::status(&h.dir).unwrap();
    assert_eq!(pending.committed_updates_total, 60_000);
    let mut c = h.open();
    let s = c.reconcile(now_ms()).unwrap();
    assert_eq!(s.outcome, Some(Outcome::Committed));
    assert_eq!(s.committed_updates_total, 60_003);
    let journal = s.journal_records;
    drop(c);
    let mut c = h.open();
    let s = c.reconcile(now_ms()).unwrap();
    assert_eq!(
        (s.journal_records, s.committed_updates_total),
        (journal, 60_003)
    );
    drop(c);
    drop(connected);
    let commits = (1..=journal)
        .filter(|i| {
            fs::read_to_string(h.dir.join(format!("journal/{i:08}.json")))
                .unwrap()
                .contains("\"event\":\"group_committed\"")
        })
        .count();
    assert_eq!(commits, 1);
    h.assert_no_orphans();
    assert_eq!(
        controller::verify(&h.dir, Some(&h.sha))
            .unwrap()
            .results_verified,
        3
    );
}

#[test]
fn restart_distinguishes_unacknowledged_completion_from_running_work() {
    let h = Harness::new(["complete", "complete", "hold"]);
    let mut c = h.open();
    let mut connected = h.start(&mut c);
    h.wait_admitted();
    for id in 1..=3 {
        heartbeat(&mut connected, id);
    }
    let pids = h.pids();
    wait(|| exited(pids[0]) && exited(pids[1]));
    drop(c);
    heartbeat(&mut connected, 4);
    let mut c = h.open();
    let s = c.reconcile(now_ms()).unwrap();
    assert_eq!(s.outcome, Some(Outcome::Drained));
    assert_eq!(s.committed_updates_total, 60_000);
    assert_eq!(
        h.dispositions(),
        vec![
            Some(Disposition::CompletedUnacknowledged),
            Some(Disposition::CompletedUnacknowledged),
            Some(Disposition::Drained)
        ]
    );
    assert!(
        !h.dir.join("results").exists(),
        "a partial group commits nothing"
    );
    drop(c);
    drop(connected);
    h.assert_no_orphans();
    controller::verify(&h.dir, Some(&h.sha)).unwrap();
}

#[test]
fn restart_before_admission_distinguishes_never_started_from_unrecorded_launch() {
    let h = Harness::new(["hold"; 3]);
    let mut c = h.open();
    let now = now_ms();
    c.preflight(&serde_json::to_vec(&census(&h.contract, now)).unwrap(), now)
        .unwrap();
    let listener = UnixListener::bind(&h.socket).unwrap();
    // rank 0: journaled launch; rank 1: spawned but the launch record was never
    // written (crash window); rank 2: dispatch intent only.
    c.record_dispatch_intent(0, now).unwrap();
    c.launch(0, now).unwrap();
    let id1 = c.record_dispatch_intent(1, now).unwrap();
    let contract = c.contract().clone();
    let sha = c.plan_sha256().to_string();
    c.supervisor()
        .launch(&RankLaunch {
            dispatch_intent_unix_ms: now_ms(),
            contract: &contract,
            plan_sha256: &sha,
            global_rank: 1,
            dispatch_id: &id1,
        })
        .unwrap();
    c.record_dispatch_intent(2, now).unwrap();
    drop(c);
    let s = controller::status(&h.dir).unwrap();
    assert_eq!(
        s.ranks.iter().map(|r| r.state).collect::<Vec<_>>(),
        vec![RankState::Unknown; 3]
    );
    let mut c = h.open();
    let s = c.reconcile(now_ms()).unwrap();
    drop(listener);
    assert_eq!(s.outcome, Some(Outcome::Drained));
    let d = h.dispositions();
    assert_eq!(
        d,
        vec![
            Some(Disposition::Drained),
            Some(Disposition::Drained),
            Some(Disposition::NeverStarted)
        ]
    );
    assert!(
        s.ranks[1].detail.contains("launch was never journaled"),
        "{}",
        s.ranks[1].detail
    );
    drop(c);
    h.assert_no_orphans();
    controller::verify(&h.dir, Some(&h.sha)).unwrap();
}

#[test]
fn cancellation_and_grant_expiry_drain_the_whole_group() {
    let h = Harness::new(["hold"; 3]);
    let mut c = h.open();
    let mut connected = h.start(&mut c);
    h.wait_admitted();
    heartbeat(&mut connected, 1);
    let r = controller::cancel(&h.dir, &h.sha, "fixture cancellation", now_ms()).unwrap();
    assert!(r.delivered_to_live_coordinator);
    let mut next = 1;
    let s = supervise(&mut c, &mut connected, &mut next);
    assert_eq!(s.outcome, Some(Outcome::Cancelled));
    assert!(s
        .ranks
        .iter()
        .all(|r| r.disposition == Some(Disposition::Drained)));
    drop(c);
    drop(connected);
    h.assert_no_orphans();

    let h = Harness::new(["hold"; 3]);
    let mut c = h.open();
    let mut connected = h.start(&mut c);
    h.wait_admitted();
    heartbeat(&mut connected, 1);
    let expiry = h.contract.participants[0].grant.body.expires_unix_ms;
    let s = c.poll(expiry).unwrap();
    assert_eq!(s.outcome, Some(Outcome::Drained));
    assert_eq!(s.committed_updates_total, 60_000);
    assert!(journal_contains(&h.dir, "\"cause\":\"grant_expired\""));
    drop(c);
    drop(connected);
    h.assert_no_orphans();
    controller::verify(&h.dir, Some(&h.sha)).unwrap();
}

// ------------------------------------- review of #111: grant expiry during recovery

/// Regression (independent review of #111, P1). `poll` drains the group once any
/// participant grant has expired, but `reconcile` adopted and committed retained results
/// after a restart without that gate: poll gave Drained/60 000 at expiry while reopen +
/// reconcile gave Committed/60 003. A restart must not widen the grant window.
#[test]
fn restart_at_grant_expiry_drains_retained_results_instead_of_committing() {
    let h = Harness::new(["complete"; 3]);
    let connected = h.completed_then_crashed();
    let expiry = h.first_expiry();
    assert!(expiry < h.contract.original_deadline_unix_ms);
    let mut c = h.open_at(expiry);
    assert!(c.needs_reconcile());
    let s = c.reconcile(expiry).unwrap();
    assert_eq!(
        (s.outcome, s.committed_updates_total),
        (Some(Outcome::Drained), 60_000),
        "recovery bypassed grant expiry"
    );
    assert!(s
        .ranks
        .iter()
        .all(|r| r.disposition == Some(Disposition::CompletedUnacknowledged)));
    assert!(journal_contains(&h.dir, "\"cause\":\"grant_expired\""));
    assert_eq!(journal_count(&h.dir, "\"event\":\"group_committed\""), 0);
    assert_eq!(
        journal_count(&h.dir, "\"event\":\"result_acknowledged\""),
        0
    );
    assert!(!h.dir.join("results").exists(), "no result was adopted");
    // Reopening the terminal attempt later changes nothing.
    let journal = s.journal_records;
    drop(c);
    let mut again = h.open_at(expiry + 1);
    assert!(!again.needs_reconcile());
    let s = again.reconcile(expiry + 1).unwrap();
    assert_eq!(
        (s.journal_records, s.committed_updates_total),
        (journal, 60_000)
    );
    drop(again);
    drop(connected);
    h.assert_no_orphans();
    controller::verify(&h.dir, Some(&h.sha)).unwrap();
}

/// Positive control: the same crash reconciled one millisecond before the first grant
/// expiry commits the complete group exactly once.
#[test]
fn restart_just_before_execution_window_closes_commits_exactly_once() {
    let h = Harness::new(["complete"; 3]);
    let connected = h.completed_then_crashed();
    let inspected = h.open();
    let wall_end = inspected
        .journal()
        .iter()
        .filter_map(|r| match r.event {
            controller::Event::DispatchIntent { global_rank, .. } => Some(
                r.recorded_unix_ms
                    + h.contract
                        .participant(global_rank)
                        .unwrap()
                        .ceilings
                        .wall_ms,
            ),
            _ => None,
        })
        .min()
        .unwrap();
    drop(inspected);
    let before = h.first_expiry().min(wall_end) - 1;
    let mut c = h.open_at(before);
    let s = c.reconcile(before).unwrap();
    assert_eq!(
        (s.outcome, s.committed_updates_total),
        (Some(Outcome::Committed), 60_003)
    );
    let journal = s.journal_records;
    drop(c);
    let mut again = h.open_at(h.first_expiry() + 1);
    let s = again.reconcile(h.first_expiry() + 1).unwrap();
    assert_eq!(
        (s.outcome, s.journal_records, s.committed_updates_total),
        (Some(Outcome::Committed), journal, 60_003)
    );
    drop(again);
    drop(connected);
    assert_eq!(journal_count(&h.dir, "\"event\":\"group_committed\""), 1);
    h.assert_no_orphans();
    assert_eq!(
        controller::verify(&h.dir, Some(&h.sha))
            .unwrap()
            .results_verified,
        3
    );
}

/// Positive control: a commit already journaled before the crash is finished after the
/// grant window has closed, never drained, and never journaled a second time.
#[test]
fn an_already_journaled_commit_is_finished_not_drained_after_expiry() {
    let h = Harness::new(["complete"; 3]);
    let mut c = h.open();
    let mut connected = h.start(&mut c);
    let mut next = 0;
    let s = supervise(&mut c, &mut connected, &mut next);
    assert_eq!(s.outcome, Some(Outcome::Committed));
    drop(c);
    // Crash immediately after the group_committed record was published.
    let removed = truncate_journal_after(&h.dir, "\"event\":\"group_committed\"");
    assert_eq!(removed, 4, "three dispositions and the terminal record");
    let s = controller::status(&h.dir).unwrap();
    assert_eq!((s.outcome, s.committed_updates_total), (None, 60_003));
    let later = h.first_expiry() + 1;
    let mut c = h.open_at(later);
    assert!(c.needs_reconcile());
    let s = c.reconcile(later).unwrap();
    assert_eq!(
        (s.outcome, s.committed_updates_total),
        (Some(Outcome::Committed), 60_003)
    );
    assert!(s
        .ranks
        .iter()
        .all(|r| r.disposition == Some(Disposition::Committed)));
    assert_eq!(journal_count(&h.dir, "\"event\":\"group_committed\""), 1);
    assert_eq!(journal_count(&h.dir, "\"event\":\"drain_started\""), 0);
    drop(c);
    drop(connected);
    h.assert_no_orphans();
    assert_eq!(
        controller::verify(&h.dir, Some(&h.sha))
            .unwrap()
            .results_verified,
        3
    );
}

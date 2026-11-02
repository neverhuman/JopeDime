// ------------------------------------------- W7-T06 the plan selects the training partition

/// A plan at a fixed time, so its canonical bytes are reproducible.
fn fixed_plan() -> BoundedRunContract {
    contract(1_790_000_000_000, 3, &keys(3))
}
/// SHA-256 of `fixed_plan()`'s canonical bytes, recorded before a plan could select a
/// training partition (`w7b-F-pre`). A plan that selects none keeps exactly these bytes.
const FIXED_PLAN_SHA256: &str = "f73eaa7e1a0cbc3b01e6d50039fd194cc89ebd2011daed613931e9d0869cc976";

#[test]
fn a_plan_that_selects_no_training_keeps_its_exact_bytes() {
    let c = fixed_plan();
    let canonical = c.canonical_bytes().unwrap();
    assert_eq!(sha(&canonical), FIXED_PLAN_SHA256);
    assert_eq!(canonical, bytes(&c));
    assert!(!String::from_utf8(canonical).unwrap().contains("training"));
}

/// `c` selecting training: the partition and batch shape, and (W7-D2) the corpus and parent
/// checkpoint it trains from, both pinned in the input closure.
fn select(c: &BoundedRunContract, partition: &str, microbatch: u32, global_batch: u32) -> Vec<u8> {
    let mut v = serde_json::to_value(c).unwrap();
    let closure = v["artifact_closure"].as_array_mut().unwrap();
    for (path, bytes) in [
        ("corpus/index.json", &b"fixture corpus index"[..]),
        ("parent/state.json", &br#"{"step":60000}"#[..]),
    ] {
        closure.push(serde_json::json!({"path": path, "sha256": sha(bytes), "bytes": bytes.len()}));
    }
    closure.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()));
    v["training"] = serde_json::json!({
        "execution_partition": partition,
        "microbatch": microbatch,
        "global_batch": global_batch,
        "corpus": "corpus",
        "checkpoint_steps": 1,
        "initialize": "parent",
    });
    serde_json::to_vec(&v).unwrap()
}

#[test]
fn a_plan_can_select_the_training_partition() {
    let now = now_ms();
    let c = contract(now, 3, &keys(3));
    let selected = select(&c, "parallel.training_partition.v2", 256, 2048);
    let parsed = BoundedRunContract::parse(&selected);
    assert!(parsed.is_ok(), "{:?}", parsed.err());
    let parsed = parsed.unwrap();
    // The record is derived for exactly this plan's world: three ranks, lossless shards.
    let record = parsed.execution_partition().unwrap().unwrap();
    assert_eq!(
        record,
        jope_parallel::task_contract::training_partition_v2(3, 256, 2048).unwrap()
    );
    assert_eq!(
        jope_parallel::task_contract::partition_rows_per_rank(&record).unwrap(),
        vec![688, 680, 680]
    );
    // The selection is part of the frozen bytes and of the reviewed summary.
    let canonical = parsed.canonical_bytes().unwrap();
    assert!(String::from_utf8(canonical.clone()).unwrap().ends_with(
        r#","training":{"execution_partition":"parallel.training_partition.v2","microbatch":256,"global_batch":2048,"corpus":"corpus","checkpoint_steps":1,"initialize":"parent"}}"#
    ));
    let (_t, root) = tempdir();
    let dir = root.join("run");
    let report = controller::plan(&dir, &selected, now, None).unwrap();
    assert_eq!(report.plan_sha256, sha(&canonical));
    assert_eq!(fs::read(dir.join("plan.json")).unwrap(), canonical);
    assert_eq!(
        parsed.review(&report.plan_sha256)["training"]["execution_partition"],
        record
    );
    // A plan that selects no training derives no record and reviews without one.
    assert_eq!(c.execution_partition().unwrap(), None);
    assert!(c.review("x").get("training").is_none());
    // Partition v1 keeps its own rule: a two-rank plan may select it.
    let two = contract(now, 2, &keys(2));
    let v1 = BoundedRunContract::parse(&select(&two, "parallel.training_partition.v1", 256, 2048))
        .unwrap();
    assert_eq!(
        v1.execution_partition().unwrap().unwrap(),
        jope_parallel::task_contract::training_partition(2, 256, 2048).unwrap()
    );
}

#[test]
fn a_training_selection_the_trainer_cannot_run_is_refused() {
    let now = now_ms();
    let three = contract(now, 3, &keys(3));
    for (partition, micro, global, needle) in [
        // v1 cannot divide one global batch over three ranks: that is what v2 is for.
        (
            "parallel.training_partition.v1",
            256,
            2048,
            "select partition v2",
        ),
        (
            "parallel.training_partition.v3",
            256,
            2048,
            "unknown training execution partition",
        ),
        (
            "parallel.training_partition.v2",
            100,
            2048,
            "training batch shape",
        ),
        (
            "parallel.training_partition.v2",
            512,
            2048,
            "training batch shape",
        ),
        (
            "parallel.training_partition.v2",
            256,
            3000,
            "training batch shape",
        ),
        (
            "parallel.training_partition.v2",
            256,
            0,
            "training batch shape",
        ),
    ] {
        let e = BoundedRunContract::parse(&select(&three, partition, micro, global)).unwrap_err();
        assert!(
            matches!(&e, PlanRefusal::Invalid(m) if m.contains(needle)),
            "{partition} {micro} {global}: {e}"
        );
    }
    // Partition v2 covers one to eight ranks.
    let nine = contract(now, 9, &keys(9));
    let e = BoundedRunContract::parse(&select(&nine, "parallel.training_partition.v2", 256, 2048))
        .unwrap_err();
    assert!(e.to_string().contains("one to eight ranks"), "{e}");
    // An unknown field inside the selection is refused.
    let mut v = serde_json::to_value(&three).unwrap();
    v["training"] = serde_json::json!({
        "execution_partition": "parallel.training_partition.v2",
        "microbatch": 256, "global_batch": 2048, "world": 3,
    });
    assert!(BoundedRunContract::parse(&serde_json::to_vec(&v).unwrap()).is_err());
    // An explicit null selects nothing: Plan freezes exactly the bytes of the plan without it.
    let mut v = serde_json::to_value(&three).unwrap();
    v["training"] = serde_json::Value::Null;
    let null = serde_json::to_vec(&v).unwrap();
    let (_t, root) = tempdir();
    let report = controller::plan(&root.join("run"), &null, now, None).unwrap();
    assert_eq!(report.plan_sha256, sha(&bytes(&three)));
    // A later attempt cannot change the selected partition: that is a topology transition.
    let selected =
        BoundedRunContract::parse(&select(&three, "parallel.training_partition.v2", 256, 2048))
            .unwrap();
    let outcome = jope_parallel::native_loop::bounded::contract::AttemptOutcome {
        plan_sha256: sha(&selected.canonical_bytes().unwrap()),
        journal_tip_sha256: sha(b"journal tip"),
        terminal: true,
        committed_updates_total: 60_000,
    };
    let mut next = selected.clone();
    next.attempt = 2;
    next.predecessor = Some(jope_parallel::native_loop::bounded::contract::Predecessor {
        plan_sha256: outcome.plan_sha256.clone(),
        attempt: 1,
        journal_tip_sha256: outcome.journal_tip_sha256.clone(),
        committed_updates: 60_000,
    });
    next.validate_successor(&selected, &outcome).unwrap();
    let changes: [fn(&mut BoundedRunContract); 2] = [
        |t| t.training.as_mut().unwrap().microbatch = 128,
        |t| t.training = None,
    ];
    for change in changes {
        let mut changed = next.clone();
        change(&mut changed);
        let e = changed.validate_successor(&selected, &outcome).unwrap_err();
        assert!(e.to_string().contains("explicit transition"), "{e}");
    }
}

#[test]
fn blocking_poll_and_recovery_refresh_time_before_result_authority() {
    use std::cell::Cell;
    // Clock advances after probe, after result retrieval, or immediately before
    // group commit. Complete valid results must not turn the old timestamp into authority.
    for (recovery, expire_on_call) in [
        (false, 2),
        (false, 3),
        (false, 5),
        (false, 14),
        (false, 15),
        (true, 2),
        (true, 4),
        (true, 9),
        (true, 10),
    ] {
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
            fake.set(
                &id,
                Liveness::Exited {
                    code: Some(0),
                    signal: None,
                },
                Some(result(
                    &r.contract,
                    &r.sha,
                    persisted.admission_sha256(),
                    rank,
                    &r.keys.ranks[rank as usize],
                )),
            );
        }
        if recovery {
            drop(c);
            c = Coordinator::open(&r.dir, &r.sha, fake.clone(), now).unwrap();
        }
        let calls = Cell::new(0);
        let clock = || {
            calls.set(calls.get() + 1);
            if calls.get() >= expire_on_call {
                r.contract.original_deadline_unix_ms
            } else {
                now
            }
        };
        let done = if recovery {
            c.reconcile_with_clock(clock)
        } else {
            c.poll_with_clock(clock)
        }
        .unwrap();
        assert_eq!(done.outcome, Some(Outcome::Drained));
        assert_eq!(
            done.committed_updates_total,
            r.contract.counters.committed_updates_at_start
        );
        assert!(!c
            .journal()
            .iter()
            .any(|r| matches!(r.event, controller::Event::GroupCommitted { .. })));
    }
}

#[test]
fn wall_budget_is_measured_from_persisted_dispatch_not_coordinator_restart() {
    let now = now_ms();
    let r = planned(now);
    let fake = FakeSupervisor::default();
    let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), now).unwrap();
    c.preflight(&serde_json::to_vec(&census(&r.contract, now)).unwrap(), now)
        .unwrap();
    c.dispatch_all(now).unwrap();
    let end = now + r.contract.participants[0].ceilings.wall_ms;
    assert!(end < r.contract.original_deadline_unix_ms);
    drop(c);
    let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), end).unwrap();
    assert_eq!(c.reconcile(end).unwrap().outcome, Some(Outcome::Drained));
    assert_eq!(fake.running(), 0);
    assert!(c.journal().iter().any(|r| matches!(
        r.event,
        controller::Event::DrainStarted {
            cause: controller::DrainCause::WallBudgetElapsed { .. }
        }
    )));
    controller::verify(&r.dir, Some(&r.sha)).unwrap();
}

#[test]
fn daemon_deadline_is_the_earliest_wall_grant_or_original_limit() {
    use jope_parallel::native_loop::bounded::supervisor::rank_job_deadline;
    let now = now_ms();
    let k = keys(3);
    let mut c = contract(now, 3, &k);
    assert_eq!(
        rank_job_deadline(&c, 0, now).unwrap(),
        now + c.participants[0].ceilings.wall_ms
    );
    c.participants[0].grant.body.expires_unix_ms = now + 100;
    assert_eq!(rank_job_deadline(&c, 0, now).unwrap(), now + 100);
    c.original_deadline_unix_ms = now + 50;
    assert_eq!(rank_job_deadline(&c, 0, now).unwrap(), now + 50);
    assert!(rank_job_deadline(&c, 0, now + 50).is_err());
    assert!(rank_job_deadline(&c, 0, u64::MAX).is_err());
}

#[test]
fn blocking_admission_delivery_cannot_renew_the_execution_window() {
    use std::sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    };
    struct DelayedWrite {
        clock: Arc<AtomicU64>,
        expire_at: Option<u64>,
        writes: Arc<AtomicUsize>,
    }
    impl std::io::Read for DelayedWrite {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Ok(0)
        }
    }
    impl std::io::Write for DelayedWrite {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.writes.fetch_add(1, Ordering::SeqCst);
            if let Some(end) = self.expire_at {
                self.clock.store(end, Ordering::SeqCst);
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl rendezvous::ControlStream for DelayedWrite {
        fn set_timeouts(
            &self,
            _: Option<std::time::Duration>,
            _: Option<std::time::Duration>,
        ) -> std::io::Result<()> {
            Ok(())
        }
        // Its writes never block; the injected clock is what this fixture advances.
        fn set_deadline(&self, _: Option<std::time::Instant>) -> std::io::Result<()> {
            Ok(())
        }
    }
    // Include expiry on the final stream: no later iteration can catch it.
    for expire_rank in [Some(0), Some(2), None] {
        let now = now_ms();
        let r = planned(now);
        let fake = FakeSupervisor::default();
        let mut c = Coordinator::open(&r.dir, &r.sha, fake.clone(), now).unwrap();
        c.preflight(&serde_json::to_vec(&census(&r.contract, now)).unwrap(), now)
            .unwrap();
        c.dispatch_all(now).unwrap();
        let (persisted, _) = admit(&mut c, &r, now);
        let clock = Arc::new(AtomicU64::new(now));
        let writes: Vec<_> = (0..3).map(|_| Arc::new(AtomicUsize::new(0))).collect();
        let mut connected: Vec<_> = (0..3)
            .map(|rank| Connected {
                global_rank: rank,
                stream: Box::new(DelayedWrite {
                    clock: clock.clone(),
                    expire_at: (expire_rank == Some(rank))
                        .then_some(now + r.contract.participants[0].ceilings.wall_ms),
                    writes: writes[rank as usize].clone(),
                }),
            })
            .collect();
        let result = c.deliver_admission_with_clock(&persisted, &mut connected, || {
            clock.load(Ordering::SeqCst)
        });
        assert_eq!(result.is_ok(), expire_rank.is_none());
        let delivered = c
            .journal()
            .iter()
            .any(|record| matches!(record.event, controller::Event::AdmissionDelivered { .. }));
        assert_eq!(delivered, expire_rank.is_none());
        if let Some(rank) = expire_rank {
            assert_eq!(fake.running(), 0);
            for count in &writes[rank as usize + 1..] {
                assert_eq!(count.load(Ordering::SeqCst), 0);
            }
            assert_eq!(c.status().unwrap().outcome, Some(Outcome::Drained));
        }
        controller::verify(&r.dir, Some(&r.sha)).unwrap();
    }
}

use super::*;

#[test]
fn missing_census_cannot_reserve_and_historical_start_remains_unavailable() {
    let (tmp, jobs, _) = fixture();
    let root = tmp.path();
    let mut ledger = Ledger::open(root).unwrap();
    assert!(ledger
        .reserve(root, &jobs, 100)
        .unwrap_err()
        .to_string()
        .contains("census unavailable"));
    assert_eq!(ledger.status().unwrap()["attempted"], 0);
    assert_eq!(ledger.status().unwrap()["unavailable_strict_counts"], 5);
    fs::create_dir_all(root.join(&jobs[0].output)).unwrap();
    fs::write(
        root.join(&jobs[0].output).join("search-request.json"),
        b"original ambiguous request",
    )
    .unwrap();
    assert!(ledger.reserve(root, &jobs, 100).is_err());
    assert_eq!(ledger.status().unwrap()["unknown"], 1);
    assert!(ledger.reserve(root, &jobs, 100).is_err());
    assert_eq!(
        ledger.status().unwrap()["unknown"],
        1,
        "ambiguous start counted twice"
    );
}

#[test]
fn concurrent_clients_reserve_distinct_requests_and_restart_preserves_unknown() {
    let (tmp, jobs, rows) = fixture();
    let root = tmp.path();
    let ledger = Ledger::open(root).unwrap();
    seed_fixture_counts(&ledger, &rows);
    drop(ledger);
    let barrier = Arc::new(Barrier::new(2));
    let selected = std::thread::scope(|scope| {
        let handles = (0..2)
            .map(|_| {
                let barrier = barrier.clone();
                let jobs = &jobs;
                scope.spawn(move || {
                    let mut ledger = Ledger::open(root).unwrap();
                    barrier.wait();
                    ledger.reserve(root, jobs, 100).unwrap().unwrap()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_ne!(selected[0], selected[1]);
    let mut ledger = Ledger::open(root).unwrap();
    ledger
        .finish(&jobs[selected[0]], "unknown", "rank disappeared", 101)
        .unwrap();
    drop(ledger);
    let mut ledger = Ledger::open(root).unwrap();
    let mut alias = jobs[selected[0]].clone();
    alias.output = "runs/renamed-output".into();
    assert!(ledger
        .reserve(root, std::slice::from_ref(&alias), 200)
        .is_err());
    alias.rng_seed = Some(73);
    assert_eq!(
        ledger.reserve(root, &[alias], 200).unwrap(),
        None,
        "unknown context became eligible for a distinct request"
    );
    assert!(ledger
        .finish(&jobs[selected[0]], "completed", "unchecked retry", 201)
        .is_err());
    ledger.reused(&jobs[selected[0]], 202, true).unwrap();
    assert_eq!(
        ledger.status().unwrap()["unknown"],
        1,
        "historical unknown erased"
    );
    assert_eq!(ledger.status().unwrap()["reused"], 1);
    assert_eq!(ledger.status().unwrap()["unreconciled_reservations"], 1);
}

#[test]
fn failed_spawn_has_diagnosis_cooldown_and_changed_job_cannot_reuse_reservation() {
    let (tmp, jobs, rows) = fixture();
    let root = tmp.path();
    let mut ledger = Ledger::open(root).unwrap();
    seed_fixture_counts(&ledger, &rows);
    let job = &jobs[ledger.reserve(root, &jobs, 100).unwrap().unwrap()];
    assert!(ledger.finish(job, "failed_before_start", "", 100).is_err());
    ledger
        .finish(job, "failed_before_start", "exec rejected", 101)
        .unwrap();
    assert_eq!(
        ledger
            .reserve(root, std::slice::from_ref(job), 130)
            .unwrap(),
        None
    );
    let mut changed = job.clone();
    changed.rng_seed = Some(3);
    assert!(ledger.reserve(root, &[changed], 131).is_err());
    assert_eq!(
        ledger
            .reserve(root, std::slice::from_ref(job), 131)
            .unwrap(),
        Some(0)
    );
    assert_eq!(ledger.status().unwrap()["failed_before_start"], 1);
}

#[test]
fn actual_reservations_apply_inflight_weight_instead_of_excluding_busy_cuts() {
    let (tmp, jobs, rows) = fixture();
    let root = tmp.path();
    let mut ledger = Ledger::open(root).unwrap();
    seed_fixture_counts(&ledger, &rows);
    assert_eq!(ledger.reserve(root, &jobs[..1], 100).unwrap(), Some(0));
    // Advance the deterministic selection stream through real completed
    // scheduling requests, retaining exactly one in-flight request on cut 0.
    let dummy = distinct_job(&jobs[4], 1);
    ledger
        .reserve(root, std::slice::from_ref(&dummy), 100)
        .unwrap()
        .unwrap();
    ledger
        .finish(&dummy, "completed", "synthetic scheduling completion", 100)
        .unwrap();
    let busy = distinct_job(&jobs[0], 2);
    let free = distinct_job(&jobs[1], 3);
    {
        let tx = ledger.db.transaction().unwrap();
        let counts = [
            context_counts(&tx, &Context::from_row(&rows[0]).key().unwrap()).unwrap(),
            context_counts(&tx, &Context::from_row(&rows[1]).key().unwrap()).unwrap(),
        ];
        assert_eq!([counts[0].in_flight, counts[1].in_flight], [1, 0]);
        let p = probabilities(&counts).unwrap();
        assert!((p[0] - 0.23).abs() < 1e-15 && (p[1] - 0.77).abs() < 1e-15);
    }
    // Sequence 2 draw is between .23 and .5: omitting the in-flight penalty
    // would choose cut 0. The actual reservation must choose cut 1.
    assert!(draw(2) > 0.23 && draw(2) < 0.5);
    assert_eq!(
        ledger
            .reserve(root, &[busy.clone(), free.clone()], 100)
            .unwrap(),
        Some(1)
    );
    ledger
        .finish(&free, "completed", "synthetic scheduling completion", 100)
        .unwrap();
    for number in 4..7 {
        let dummy = distinct_job(&jobs[4], number);
        ledger
            .reserve(root, std::slice::from_ref(&dummy), 100)
            .unwrap()
            .unwrap();
        ledger
            .finish(&dummy, "completed", "synthetic scheduling completion", 100)
            .unwrap();
    }
    // Sequence 6 draw is below .23. A blanket busy-cut exclusion would choose
    // cut 1; correct weighted admission can reserve a distinct request on cut 0.
    assert!(draw(6) < 0.23);
    assert_eq!(
        ledger
            .reserve(root, &[busy, distinct_job(&jobs[1], 7)], 100)
            .unwrap(),
        Some(0)
    );
    assert_eq!(ledger.status().unwrap()["unreconciled_reservations"], 2);
}

#[test]
fn concurrent_same_request_has_one_acceptance_but_distinct_requests_can_share_cut() {
    for distinct in [false, true] {
        let (tmp, jobs, rows) = fixture();
        let root = tmp.path();
        let ledger = Ledger::open(root).unwrap();
        seed_fixture_counts(&ledger, &rows);
        drop(ledger);
        let barrier = Arc::new(Barrier::new(2));
        let requests = [
            jobs[0].clone(),
            if distinct {
                distinct_job(&jobs[0], 50)
            } else {
                jobs[0].clone()
            },
        ];
        let selected = std::thread::scope(|scope| {
            let handles = requests
                .iter()
                .map(|job| {
                    let barrier = barrier.clone();
                    scope.spawn(move || {
                        let mut ledger = Ledger::open(root).unwrap();
                        barrier.wait();
                        ledger
                            .reserve(root, std::slice::from_ref(job), 100)
                            .unwrap()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(
            selected.iter().filter(|i| i.is_some()).count(),
            if distinct { 2 } else { 1 }
        );
        let mut ledger = Ledger::open(root).unwrap();
        let mut renamed = jobs[0].clone();
        renamed.output = "runs/alias".into();
        assert!(
            ledger.reserve(root, &[renamed], 200).is_err(),
            "output alias minted a duplicate request"
        );
    }
}

#[test]
fn cross_population_plans_remain_eligible_and_transfer_to_dispatch_without_double_counting() {
    let (tmp, jobs, rows) = fixture();
    let root = tmp.path();
    let mut ledger = Ledger::open(root).unwrap();
    seed_fixture_counts(&ledger, &rows);
    let corpus = root.join("corpus");
    fs::create_dir(&corpus).unwrap();
    fs::write(corpus.join("index.json"), b"{}").unwrap();
    let owner = root.join("runs/generation-0");
    let first = ledger
        .freeze_population(root, &corpus, &owner, &rows[..1], 1, 100)
        .unwrap();
    let second = ledger
        .freeze_population(
            root,
            &corpus,
            &root.join("runs/generation-1"),
            &rows[..1],
            1,
            101,
        )
        .unwrap();
    assert_eq!(first["contexts"], second["contexts"]);
    assert_eq!(
        ledger
            .db
            .query_row("SELECT count(*) FROM coverage_plans", [], |r| r
                .get::<_, u64>(0))
            .unwrap(),
        2
    );
    // Re-reading a population does not mint another plan.
    assert_eq!(
        ledger
            .freeze_population(root, &corpus, &owner, &rows[..1], 1, 200)
            .unwrap(),
        first
    );
    ledger.reserve(root, &jobs[..1], 200).unwrap().unwrap();
    let tx = ledger.db.transaction().unwrap();
    let counts = context_counts(&tx, &Context::from_row(&rows[0]).key().unwrap()).unwrap();
    assert_eq!(
        counts.in_flight, 2,
        "one live request plus other population's plan expected"
    );
    assert_eq!(counts.attempted, 1);
    assert_eq!(
        counts.selection_exposure, 2,
        "active request counted twice for balancing"
    );
    assert_eq!(
        tx.query_row("SELECT count(*) FROM coverage_plans", [], |r| r
            .get::<_, u64>(0))
            .unwrap(),
        1
    );
}

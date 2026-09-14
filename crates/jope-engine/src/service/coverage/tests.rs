use super::*;
use std::sync::{Arc, Barrier};

fn context(family: &str, dataset: &str, cut: usize) -> Context {
    Context {
        family: family.into(),
        dataset: dataset.into(),
        partition: "meta_train".into(),
        prepared_sha256: "a".repeat(64),
        cut,
    }
}
fn known(attempted: u64, commits: u64) -> Counts {
    Counts {
        unique_strict_commits: Some(commits),
        attempted,
        selection_exposure: attempted,
        ..Counts::default()
    }
}

#[test]
fn frozen_probability_has_uniform_floor_and_explicit_missing_denominator() {
    let values = probabilities(&[known(0, 0), known(0, 1), known(0, 9)]).unwrap();
    let sum = 1.0 + 0.25 + 0.01;
    for (actual, weight) in values.iter().zip([1.0, 0.25, 0.01]) {
        assert!((*actual - (0.1 / 3.0 + 0.9 * weight / sum)).abs() < 1e-15);
    }
    assert!((values.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    assert!(probabilities(&[Counts::default()]).is_err());
    let mut flight = known(0, 0);
    flight.in_flight = 1;
    assert_eq!(
        probabilities(&[flight, known(0, 1)]).unwrap(),
        vec![0.5, 0.5]
    );
}

#[test]
fn balances_family_then_dataset_and_sweeps_all_existing_cuts() {
    let mut rows = vec![
        (context("a", "many", 0), known(10, 100)),
        (context("b", "single", 4), known(1, 0)),
        (context("a", "few", 4), known(0, 0)),
    ];
    assert_eq!(select(&rows, 0, 100).unwrap(), Some(1));
    rows[1].1.attempted = 11;
    rows[1].1.selection_exposure = 11;
    assert_eq!(select(&rows, 0, 100).unwrap(), Some(2));
    let mut rows = (0..5)
        .map(|i| (context("family", "dataset", i), known(0, 0)))
        .collect::<Vec<_>>();
    let mut selected = Vec::new();
    for sequence in [0, 10, 20, 30, 40] {
        let i = select(&rows, sequence, 100).unwrap().unwrap();
        selected.push(rows[i].0.cut);
        rows[i].1.attempted += 1;
        rows[i].1.selection_exposure += 1;
    }
    assert_eq!(selected, vec![0, 1, 2, 3, 4]);
    let selected = select(&rows, 19, 100).unwrap().map(|i| rows[i].0.clone());
    rows.reverse();
    assert_eq!(
        select(&rows, 19, 100).unwrap().map(|i| rows[i].0.clone()),
        selected
    );
}

fn fixture() -> (tempfile::TempDir, Vec<GpJob>, Vec<ContextRow>) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    fs::create_dir_all(root.join("prepared")).unwrap();
    let prepared = serde_json::json!({"schema":"parallel.prepared.v1","dataset":"dataset","family":"family",
        "meta_partition":"meta_train","cuts":[{"index":0},{"index":1},{"index":2},{"index":3},{"index":4}]});
    write_json(&root.join("prepared/prepared.json"), &prepared).unwrap();
    let digest = file_hash(&root.join("prepared/prepared.json")).unwrap();
    let jobs = (0..5)
        .map(|cut| GpJob {
            prepared: "prepared".into(),
            prepared_sha256: digest.clone(),
            cut,
            output: format!("runs/generation-0/dataset/cut-{cut}").into(),
            ..GpJob::default()
        })
        .collect::<Vec<_>>();
    let rows = jobs
        .iter()
        .map(|j| ContextRow {
            dataset: "dataset".into(),
            family: "family".into(),
            cut: j.cut,
            partition: "meta_train".into(),
            path: format!("context-{}", j.cut).into(),
            sha256: "b".repeat(64),
            prepared: j.prepared.clone(),
            prepared_sha256: digest.clone(),
        })
        .collect();
    (tmp, jobs, rows)
}

// Synthetic metadata only. Production cannot turn an absent recovery census
// into these verified-zero fixtures through an executable configuration flag.
fn seed_fixture_counts(ledger: &Ledger, rows: &[ContextRow]) {
    for row in rows {
        let context = Context::from_row(row);
        ledger
            .db
            .execute(
                "INSERT INTO coverage_contexts(identity,metadata,strict_commits) VALUES(?1,?2,0)",
                params![
                    context.key().unwrap(),
                    serde_json::to_string(&context).unwrap()
                ],
            )
            .unwrap();
    }
}

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
fn generation_population_is_frozen_training_only_and_distinct_within_population() {
    let (tmp, _, mut rows) = fixture();
    let root = tmp.path();
    let mut ledger = Ledger::open(root).unwrap();
    seed_fixture_counts(&ledger, &rows);
    let mut sealed = rows[0].clone();
    sealed.partition = "meta_test".into();
    rows.push(sealed);
    let corpus = root.join("corpus");
    fs::create_dir(&corpus).unwrap();
    fs::write(corpus.join("index.json"), b"{}").unwrap();
    let owner = root.join("runs/generation-a");
    let one = ledger
        .freeze_population(root, &corpus, &owner, &rows, 2, 100)
        .unwrap();
    assert_eq!(one["selected_contexts"], 2);
    let again = ledger
        .freeze_population(root, &corpus, &owner, &rows, 2, 999)
        .unwrap();
    assert_eq!(one, again);
    let two = ledger
        .freeze_population(
            root,
            &corpus,
            &root.join("runs/generation-b"),
            &rows,
            2,
            101,
        )
        .unwrap();
    for population in [&one, &two] {
        let rows = population["contexts"].as_array().unwrap();
        let unique = rows
            .iter()
            .map(|row| row["context_index"].as_u64().unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(unique.len(), rows.len());
    }
    let path = root.join("selection.json");
    write_json(&path, &one).unwrap();
    assert_eq!(selected_indices(&path, &corpus, &rows).unwrap().len(), 2);
    let mut legacy = one.clone();
    legacy["request"]
        .as_object_mut()
        .unwrap()
        .remove("implementation");
    write_json(&path, &legacy).unwrap();
    assert!(
        selected_indices(&path, &corpus, &rows).is_err(),
        "old selection implementation was silently accepted"
    );
    write_json(&path, &one).unwrap();
    let i = one["contexts"][0]["context_index"].as_u64().unwrap() as usize;
    rows[i].prepared_sha256 = "c".repeat(64);
    assert!(selected_indices(&path, &corpus, &rows).is_err());
    assert!(ledger
        .freeze_population(root, &corpus, &owner, &rows, 3, 200)
        .is_err());
}

fn distinct_job(base: &GpJob, number: u64) -> GpJob {
    GpJob {
        output: format!("runs/distinct-{number}").into(),
        rng_seed: Some(number),
        ..base.clone()
    }
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

#[test]
fn earlier_or_unknown_ledger_versions_are_rejected_without_mutating_original_bytes() {
    for version in [0, 1, 99] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        fs::create_dir(root.join("runs")).unwrap();
        let path = root.join("runs/gp-coverage.sqlite3");
        let db = Connection::open(&path).unwrap();
        db.execute_batch(
            "CREATE TABLE coverage_plans(context TEXT PRIMARY KEY,owner TEXT,timestamp INTEGER);
            INSERT INTO coverage_plans VALUES('context','original owner',100);",
        )
        .unwrap();
        db.pragma_update(None, "user_version", version).unwrap();
        drop(db);
        let before = file_hash(&path).unwrap();
        assert!(Ledger::open(root)
            .err()
            .unwrap()
            .to_string()
            .contains("explicit reconciliation"));
        assert_eq!(file_hash(&path).unwrap(), before);
    }
    let temp = tempfile::tempdir().unwrap();
    let ledger = Ledger::open(temp.path()).unwrap();
    assert_eq!(
        ledger
            .db
            .query_row("PRAGMA user_version", [], |r| r.get::<_, u32>(0))
            .unwrap(),
        LEDGER_VERSION
    );
    assert_eq!(ledger.status().unwrap()["implementation"], IMPLEMENTATION);
}

#[test]
fn queued_plans_balance_families_and_datasets_without_inflating_dispatch_attempts() {
    for families in [["a", "b"], ["same", "same"]] {
        let (tmp, _, mut rows) = fixture();
        let root = tmp.path();
        rows.truncate(2);
        for (i, row) in rows.iter_mut().enumerate() {
            row.family = families[i].into();
            row.dataset = format!("dataset-{i}");
        }
        let mut ledger = Ledger::open(root).unwrap();
        seed_fixture_counts(&ledger, &rows);
        let corpus = root.join("corpus");
        fs::create_dir(&corpus).unwrap();
        fs::write(corpus.join("index.json"), b"{}").unwrap();
        ledger
            .freeze_population(
                root,
                &corpus,
                &root.join("runs/earlier"),
                &rows[..1],
                1,
                100,
            )
            .unwrap();
        let next = ledger
            .freeze_population(root, &corpus, &root.join("runs/later"), &rows, 1, 101)
            .unwrap();
        assert_eq!(
            next["contexts"][0]["identity"]["dataset"], "dataset-1",
            "prior queued plan must contribute to family/dataset selection exposure"
        );
        assert_eq!(
            ledger.status().unwrap()["attempted"],
            0,
            "generation planning must not be reported as attempted dispatch"
        );
    }
}

use super::*;

#[test]
fn missing_gate_or_aborted_executor_retains_unknown_and_never_retries() {
    for bad_gate in [false, true] {
        let home = tempfile::tempdir().unwrap();
        let p = control_plan(home.path(), 40);
        let run = p.run_id.clone();
        let (c, _) = context(&p, 20, true, bad_gate);
        let store = Arc::new(Store::open(home.path().join("store")).unwrap());
        if bad_gate {
            assert!(Discovery::open_research(store.clone(), p.clone(), c.clone()).is_err());
            assert!(store.get(&RecordKey::Run(run)).unwrap().is_none());
            continue;
        }
        let mut d = Discovery::open_research(store.clone(), p.clone(), c.clone()).unwrap();
        let mut e = Evaluator::new(store.clone());
        if !bad_gate {
            e.panic_at = Some(2);
        }
        assert!(d
            .advance(&SPACE, &HeuristicAdmission, &e, 1, usize::MAX)
            .is_err());
        assert!(d.poisoned);
        assert_eq!(stored(&d)["capture"]["charged"], 4);
        assert!(d.projection().unwrap().as_array().unwrap().is_empty());
        drop(d);
        let mut d = Discovery::open_research(store.clone(), p, c).unwrap();
        let calls = e.calls.lock().unwrap().len();
        assert!(d
            .advance(&SPACE, &HeuristicAdmission, &e, 1, usize::MAX)
            .is_err());
        assert_eq!(calls, e.calls.lock().unwrap().len());
        assert!(d.checkpoint.capture.as_ref().unwrap().tainted);
        assert!(materialize_controls(&store, &run, &home.path().join("records")).is_err());
    }
}

#[test]
fn empty_oversized_seed_and_unobserved_admission_refuse_before_dispatch() {
    for seeds in [0, 17] {
        let home = tempfile::tempdir().unwrap();
        let p = control_plan(home.path(), 40);
        let store = Arc::new(Store::open(home.path().join("store")).unwrap());
        let (c, _) = context(&p, 20, true, false);
        let mut d = Discovery::open_research(store.clone(), p, c).unwrap();
        let e = Evaluator::new(store);
        assert!(d
            .advance(
                &Space {
                    seeds,
                    duplicate: false
                },
                &HeuristicAdmission,
                &e,
                1,
                usize::MAX
            )
            .is_err());
        assert!(e.calls.lock().unwrap().is_empty());
        assert_eq!(stored(&d)["capture"]["charged"], 0);
    }
    struct Unobserved;
    impl Admission for Unobserved {
        fn admit(&self, _: Vec<String>, _: bool, _: &mut ChaCha12Rng) -> Vec<String> {
            panic!("must refuse, not invoke an unobserved policy")
        }
    }
    let home = tempfile::tempdir().unwrap();
    let p = control_plan(home.path(), 40);
    let store = Arc::new(Store::open(home.path().join("store")).unwrap());
    let (c, _) = context(&p, 20, true, false);
    let mut d = Discovery::open_research(store.clone(), p, c).unwrap();
    let e = Evaluator::new(store);
    d.advance(&SPACE, &HeuristicAdmission, &e, 1, usize::MAX)
        .unwrap();
    let calls = e.calls.lock().unwrap().len();
    assert!(d.advance(&SPACE, &Unobserved, &e, 1, usize::MAX).is_err());
    assert_eq!(e.calls.lock().unwrap().len(), calls);
}

#[test]
fn capture_roots_survive_gc_and_normal_open_or_binding_drift_refuse() {
    let home = tempfile::tempdir().unwrap();
    let p = control_plan(home.path(), 40);
    let store = Arc::new(Store::open(home.path().join("store")).unwrap());
    let (c, _) = context(&p, 20, true, false);
    let mut d = Discovery::open_research(store.clone(), p.clone(), c).unwrap();
    let e = Evaluator::new(store.clone());
    run_to_end(&mut d, &e, 8);
    let roots = d.checkpoint.capture.as_ref().unwrap().roots();
    let orphan = store.publish_blob(&b"unreferenced-control"[..]).unwrap();
    let gc = store.gc(true).unwrap();
    assert_eq!(gc.blobs, vec![orphan]);
    assert!(gc.blobs.iter().all(|r| !roots.contains(r)));
    d.pause().unwrap();
    drop(d);
    assert!(Discovery::open(store.clone(), p.clone()).is_err());
    let (mut changed, _) = context(&p, 20, true, false);
    Arc::get_mut(&mut changed).unwrap().binding.tokens[0][0] = 0.5;
    assert!(Discovery::open_research(store, p, changed).is_err());
}

#[test]
fn round_transaction_failure_never_installs_outbox_or_public_results() {
    let home = tempfile::tempdir().unwrap();
    let p = control_plan(home.path(), 40);
    let store = Arc::new(Store::open(home.path().join("store")).unwrap());
    let (c, _) = context(&p, 20, true, false);
    let mut d = Discovery::open_research(store.clone(), p.clone(), c.clone()).unwrap();
    let e = Evaluator::new(store.clone());
    store
        .transaction(|tx| {
            tx.put(
                &RecordKey::Evaluation {
                    run: p.run_id.clone(),
                    number: 0,
                },
                &Record::metadata(json!({"conflict-control":true}), vec![]),
            )
        })
        .unwrap();
    assert!(d
        .advance(&SPACE, &HeuristicAdmission, &e, 1, usize::MAX)
        .is_err());
    assert!(d.poisoned);
    let cp = stored(&d);
    assert!(cp["elites"]["objective"].is_null() || cp["champion"].is_null());
    assert!(cp["capture"]["rounds"]["0"]["committed"].is_null());
    assert_eq!(cp["next_round"], 0);
    assert!(d.projection().unwrap().as_array().unwrap().is_empty());
    drop(d);
    let resumed = Discovery::open_research(store, p, c).unwrap();
    assert!(!resumed.checkpoint.capture.as_ref().unwrap().unknown());
    assert!(records(&resumed).is_empty());
}

#[test]
fn corrupt_or_missing_capture_cas_never_materializes() {
    let home = tempfile::tempdir().unwrap();
    let p = control_plan(home.path(), 40);
    let run = p.run_id.clone();
    let (c, _) = context(&p, 20, true, false);
    let store_home = home.path().join("store");
    let store = Arc::new(Store::open(&store_home).unwrap());
    let mut d = Discovery::open_research(store.clone(), p.clone(), c.clone()).unwrap();
    let e = Evaluator::new(store.clone());
    run_to_end(&mut d, &e, 1);
    let reference = d.checkpoint.capture.as_ref().unwrap().rounds[&1]
        .slate
        .clone();
    let name = hex_blob(&reference);
    let path = store_home
        .join("cas/sha256")
        .join(&name[..2])
        .join(&name[2..4])
        .join(&name[4..]);
    let saved = fs::read(&path).unwrap();
    fs::write(&path, b"deliberately corrupted own control fixture").unwrap();
    let output = home.path().join("records.jsonl");
    assert!(materialize_controls(&store, &run, &output).is_err());
    assert!(!output.exists());
    assert!(Discovery::open_research(store.clone(), p.clone(), c.clone()).is_err());
    fs::write(&path, &saved).unwrap();
    let backup = home.path().join("retained-cas-control");
    fs::rename(&path, &backup).unwrap();
    assert!(materialize_controls(&store, &run, &output).is_err());
    assert!(!output.exists());
    fs::rename(backup, path).unwrap();
    assert_eq!(materialize_controls(&store, &run, &output).unwrap(), 3);
}

#[test]
fn seed_endpoints_and_capture_input_bounds_are_explicit() {
    for seeds in [1, 16] {
        let home = tempfile::tempdir().unwrap();
        let p = control_plan(home.path(), 48);
        let (c, _) = context(&p, 16, true, false);
        let store = Arc::new(Store::open(home.path().join("store")).unwrap());
        let mut d = Discovery::open_research(store.clone(), p, c).unwrap();
        let e = Evaluator::new(store);
        d.advance(
            &Space {
                seeds,
                duplicate: false,
            },
            &HeuristicAdmission,
            &e,
            1,
            usize::MAX,
        )
        .unwrap();
        assert_eq!(records(&d)[0].proposals.len(), seeds);
        assert_eq!(records(&d)[0].outcomes.len(), seeds);
    }
    for case in 0..7 {
        let home = tempfile::tempdir().unwrap();
        let p = control_plan(home.path(), 40);
        let (mut c, _) = context(&p, 20, true, false);
        let b = &mut Arc::get_mut(&mut c).unwrap().binding;
        match case {
            0 => b.config.max_attempts = 41,
            1 => b.config.max_rounds = 0,
            2 => b.config.max_bytes = 0,
            3 => b.config.seed_protocol_sha256.clear(),
            4 => b.identity.task = RecordTask::Binary,
            5 => b.matrix_sha256 = digest(b"changed"),
            _ => b.seed += 1,
        }
        let store = Arc::new(Store::open(home.path().join("store")).unwrap());
        assert!(Discovery::open_research(store.clone(), p.clone(), c).is_err());
        assert!(store.get(&RecordKey::Run(p.run_id)).unwrap().is_none());
    }
}

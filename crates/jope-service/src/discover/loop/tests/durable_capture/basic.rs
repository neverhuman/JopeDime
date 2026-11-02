use super::*;

#[test]
fn labels_are_private_and_one_eight_workers_preserve_normal_gp() {
    let home = tempfile::tempdir().unwrap();
    let p = control_plan(home.path(), 48);
    let mut baseline = None;
    for workers in [1, 8] {
        let store = Arc::new(Store::open(home.path().join(format!("capture-{workers}"))).unwrap());
        let (c, provider) = context(&p, 40, true, false);
        let mut d = Discovery::open_research(store.clone(), p.clone(), c).unwrap();
        let e = Evaluator::new(store);
        for _ in 0..2 {
            d.advance(&SPACE, &HeuristicAdmission, &e, workers, usize::MAX)
                .unwrap();
        }
        assert_eq!(provider.observations.load(Ordering::SeqCst), 3);
        assert_eq!(d.admitted(), 12);
        assert_eq!(d.checkpoint.capture.as_ref().unwrap().charged, 20);
        assert_eq!(
            d.attempt, 16,
            "extra slots must not consume admitted attempt counter"
        );
        let captured = records(&d);
        assert_eq!(captured.len(), 3);
        assert!(captured[1].outcomes.iter().any(|o| matches!(
            o.measurement,
            Rung0Measurement::Success {
                objective: -10000.0,
                ..
            }
        )));
        assert!(captured.iter().all(|r| r.policy.slate_costs.is_none()));
        assert!(captured[1].revealed.iter().all(|r| r.surrogate == 100.0));
        let label_keys: Vec<_> = captured
            .iter()
            .flat_map(|r| {
                r.proposals
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !r.policy.admitted.contains(i))
                    .map(|(_, p)| p.program_sha256.clone())
            })
            .collect();
        assert!(label_keys.iter().all(|k| !d.checkpoint.memo.contains(k)));
        let public = d.projection().unwrap();
        let public_text = serde_json::to_string(&public).unwrap();
        for private in [
            "feat412",
            "proj16",
            "conditioning_sha256",
            "slate_costs",
            "synthetic_control",
            "-10000",
        ] {
            assert!(!public_text.contains(private));
        }
        let state = serde_json::to_value((
            &d.checkpoint.programs,
            &d.checkpoint.elites,
            &d.checkpoint.champion,
        ))
        .unwrap();
        if let Some((old_state, old_public)) = &baseline {
            assert_eq!(&state, old_state);
            assert_eq!(&public, old_public);
        } else {
            baseline = Some((state, public));
        }
    }
    let store = Arc::new(Store::open(home.path().join("normal")).unwrap());
    let mut normal = Discovery::open(store.clone(), p).unwrap();
    let e = Evaluator::new(store);
    for _ in 0..2 {
        normal
            .advance(&SPACE, &HeuristicAdmission, &e, 1, usize::MAX)
            .unwrap();
    }
    assert_eq!(
        serde_json::to_value((
            &normal.checkpoint.programs,
            &normal.checkpoint.elites,
            &normal.checkpoint.champion
        ))
        .unwrap(),
        baseline.as_ref().unwrap().0
    );
    assert_eq!(normal.projection().unwrap(), baseline.unwrap().1);
    assert_eq!(stored(&normal)["version"], 2);
    assert!(stored(&normal).get("capture").is_none());
    assert!(stored(&normal)["pending"]
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r.get("capture").is_none()));
}

#[test]
fn full_slates_keep_source_indices_and_original_precision() {
    let home = tempfile::tempdir().unwrap();
    let p = control_plan(home.path(), 40);
    let store = Arc::new(Store::open(home.path().join("store")).unwrap());
    let (c, _) = context(&p, 40, true, false);
    let mut d = Discovery::open_research(store.clone(), p, c).unwrap();
    let e = Evaluator::new(store);
    let space = Space {
        seeds: 4,
        duplicate: true,
    };
    d.advance(&space, &HeuristicAdmission, &e, 1, usize::MAX)
        .unwrap();
    d.checkpoint.stalled = 100; // Explicit software control for the exploring policy branch.
    d.commit_control(d.checkpoint.clone(), RunState::Running)
        .unwrap();
    d.advance(&space, &HeuristicAdmission, &e, 8, usize::MAX)
        .unwrap();
    let r = records(&d);
    assert_eq!(r[1].proposals.len(), 16);
    assert!(r[1].policy.explore);
    assert_eq!(&r[1].source_indices[..3], &[0, 2, 4]);
    assert_eq!(r[1].proposals[0].proj16[0], 1.000_000_000_000_01);
    assert_eq!(r[1].revealed.len(), 4);
    assert_eq!(r[1].outcomes.len(), 16);
    let mut rng = round_rng(d.checkpoint.plan.seed, 1);
    let raw = space.propose(&[], 16, &mut rng).unwrap();
    let mut seen = (0..4)
        .map(|i| digest(format!("seed-{i}").as_bytes()))
        .collect::<BTreeSet<_>>();
    let eligible = raw
        .iter()
        .map(|dsl| space.canonical(dsl).unwrap())
        .filter(|dsl| seen.insert(digest(dsl.as_bytes())))
        .collect();
    let expected = HeuristicAdmission
        .admit(eligible, true, &mut rng)
        .into_iter()
        .map(|dsl| digest(dsl.as_bytes()))
        .collect::<Vec<_>>();
    assert_eq!(
        r[1].policy
            .admitted
            .iter()
            .map(|i| r[1].proposals[*i].program_sha256.clone())
            .collect::<Vec<_>>(),
        expected
    );
}

#[test]
fn materialization_reads_committed_state_and_python_counts_both_modes() {
    for all in [false, true] {
        let home = tempfile::tempdir().unwrap();
        let p = control_plan(home.path(), 40);
        let run = p.run_id.clone();
        let store = Arc::new(Store::open(home.path().join("store")).unwrap());
        let (c, _) = context(&p, if all { 20 } else { 12 }, all, false);
        let mut d = Discovery::open_research(store.clone(), p, c).unwrap();
        let mut e = Evaluator::new(store.clone());
        e.fail_extra = true;
        run_to_end(&mut d, &e, 8);
        let output = home.path().join("records.jsonl");
        assert!(
            materialize_committed(&store, &run, &output).is_err(),
            "controls cannot claim Engine evidence"
        );
        assert!(!output.exists());
        assert_eq!(materialize_controls(&store, &run, &output).unwrap(), 3);
        let bytes = fs::read(&output).unwrap();
        assert!(materialize_controls(&store, &run, &output).is_err());
        assert_eq!(fs::read(&output).unwrap(), bytes);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&output).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/src/jope/north_star/record_reader.py");
        let mut cmd = Command::new("python3");
        cmd.arg(script).arg(&output).arg("--allow-synthetic");
        if all {
            cmd.arg("--require-evaluate-all");
        }
        let result = cmd.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let counts: Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(
            counts,
            json!({"rounds":3,"proposals":20,"outcomes":if all{20}else{12},"failures":if all{8}else{0},"incomplete_rounds":0})
        );
        assert!(
            fs::read_dir(home.path())
                .unwrap()
                .filter_map(Result::ok)
                .filter(|e| e
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".jope-research-"))
                .count()
                >= 2,
            "staging evidence must remain"
        );
    }
}

#[test]
fn budget_truncation_is_retained_but_not_exported() {
    let home = tempfile::tempdir().unwrap();
    let p = control_plan(home.path(), 40);
    let run = p.run_id.clone();
    let store = Arc::new(Store::open(home.path().join("store")).unwrap());
    let (c, _) = context(&p, 7, true, false);
    let mut d = Discovery::open_research(store.clone(), p, c).unwrap();
    let e = Evaluator::new(store.clone());
    run_to_end(&mut d, &e, 1);
    assert_eq!(d.checkpoint.capture.as_ref().unwrap().charged, 7);
    let r = records(&d);
    assert_eq!(r.len(), 2);
    assert_eq!(r[1].outcomes.len(), 3);
    assert!(!r[1].capture_complete);
    assert!(materialize_controls(&store, &run, &home.path().join("records")).is_err());
}

#[test]
fn pause_reopen_reuses_known_wave_without_proposal_or_admission_replay() {
    struct NoSpace;
    impl SearchSpace for NoSpace {
        fn canonical(&self, _: &str) -> Result<String> {
            panic!("recanonicalized")
        }
        fn seeds(&self) -> Result<Vec<String>> {
            panic!("reseeded")
        }
        fn propose(&self, _: &[String], _: usize, _: &mut ChaCha12Rng) -> Result<Vec<String>> {
            panic!("reproposed")
        }
    }
    let home = tempfile::tempdir().unwrap();
    let p = control_plan(home.path(), 40);
    let (c, provider) = context(&p, 20, true, false);
    let store = Arc::new(Store::open(home.path().join("store")).unwrap());
    let mut d = Discovery::open_research(store.clone(), p.clone(), c.clone()).unwrap();
    let e = Evaluator::new(store.clone());
    d.advance(&SPACE, &HeuristicAdmission, &e, 8, usize::MAX)
        .unwrap();
    d.advance(&SPACE, &HeuristicAdmission, &e, 8, 1).unwrap();
    assert_eq!(d.checkpoint.pending.len(), 1);
    assert_eq!(d.checkpoint.capture.as_ref().unwrap().charged, 20);
    d.pause().unwrap();
    drop(d);
    let before = e.calls.lock().unwrap().len();
    let mut d = Discovery::open_research(store, p, c).unwrap();
    d.advance(&NoSpace, &HeuristicAdmission, &e, 1, usize::MAX)
        .unwrap();
    assert_eq!(provider.observations.load(Ordering::SeqCst), 3);
    assert_eq!(
        e.calls.lock().unwrap().len() - before,
        0,
        "known pending outcomes must not execute again"
    );
    assert_eq!(records(&d).len(), 3);
    assert!(d.completed().is_some());
}

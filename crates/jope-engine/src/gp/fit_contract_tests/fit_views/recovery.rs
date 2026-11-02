use super::*;

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn experiment_crashes_at_durable_boundaries_recover_without_repeating_committed_fits() {
    for boundary in [
        "candidate_intent",
        "candidate_metric",
        "observation_journal",
        "observation_progress",
        "observations_snapshot",
        "search_summary",
    ] {
        let (temp, prepared) = view_fixture();
        let root = temp.path();
        let mut authority = Fixture::activated(root, vec![cut_training_hash(&prepared)]);
        authority.start();
        let experiment = root.join("experiment");
        fs::create_dir_all(&experiment).unwrap();
        write_json(
            &experiment.join("manifest.json"),
            &json!({"schema":"parallel.seed_experiment.v2"}),
        )
        .unwrap();
        // Keep numerical work bounded while testing the recovery protocol.
        // Three distinct recipes still require three original fit keys; deadline
        // refusal and bootstrap generation have separate contracts.
        let seeds = root.join("recovery-seeds.json");
        seed_population(
            &prepared,
            &seeds,
            &[
                PROGRAM,
                "(pipe X (standardize) (ridge))",
                "(pipe X (robust_scale) (ridge))",
            ],
            |_| None,
        );
        let output = experiment.join("search");
        let command = || {
            let mut c = authority.command();
            c.args(["search", "--root"])
                .arg(root)
                .args([
                    "--cut",
                    "0",
                    "--evaluations",
                    "3",
                    "--seconds",
                    "180",
                    "--qualification",
                ])
                .arg("--prepared")
                .arg(&prepared)
                .arg("--output")
                .arg(&output)
                .arg("--seeds")
                .arg(&seeds)
                .env("JOPE_EXPERIMENT_DIR", &experiment);
            c
        };
        let failed = command()
            .env("JOPE_EXPERIMENT_CRASH_AFTER", boundary)
            .output()
            .unwrap();
        assert!(
            !failed.status.success(),
            "failpoint was not reached: {boundary}"
        );
        let before = if output.join("observations.jsonl").exists() {
            read_journal(&output.join("observations.jsonl")).unwrap()
        } else {
            vec![]
        };
        let metric: Option<Value> = if boundary == "candidate_metric" {
            Some(read_json(&output.join("candidate-0.json")).unwrap())
        } else {
            None
        };
        if !output.join("search.json").exists() {
            let resumed = command().output().unwrap();
            assert!(
                resumed.status.success(),
                "{boundary}: {}",
                String::from_utf8_lossy(&resumed.stderr)
            );
        }
        let after: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
        assert_eq!(after.len(), 3, "{boundary}");
        // The independent replay reaches the published winner and stop from the
        // recovered ledger, including recovered metrics and unobserved intents.
        let replayed = selection_replay::verify_search(root, &output)
            .unwrap_or_else(|error| panic!("{boundary}: {error:#}"));
        assert_eq!(
            replayed.replay.stop,
            selection_replay::StopReason::EvaluationBudget,
            "{boundary}"
        );
        if boundary == "candidate_intent" {
            assert!(matches!(
                replayed.replay.rows[0].eligibility,
                selection_replay::Eligibility::Failed { .. }
            ));
        }
        assert_eq!(
            after[..before.len()],
            before,
            "committed observations changed at {boundary}"
        );
        if let Some(metric) = metric {
            assert_eq!(after[0]["metrics"], metric);
            assert_eq!(after[0]["recovered_completed_artifact"], true);
        }
        let counters = after
            .iter()
            .map(|r| r["proposal_attempts"].as_u64().unwrap())
            .collect::<Vec<_>>();
        assert!(
            counters.windows(2).all(|w| w[0] < w[1]),
            "RNG counter did not advance: {boundary}"
        );
        // Authority-side proof of "without repeating committed fits": every
        // original fit key started exactly once and nothing was replayed.
        let stats = authority.stop_service();
        let (entries, _) = authority.entries();
        let fitted = after.iter().filter(|r| !r["fit_outcome"].is_null()).count();
        assert_eq!(stats["worker_spawns"], entries, "{boundary}: {stats}");
        assert_eq!(stats["reused"], 0, "{boundary}: {stats}");
        assert_eq!(fitted as i64, entries, "{boundary}");
        if boundary == "candidate_intent" {
            // An intent without a completion is never dispatched again.
            assert_eq!(
                after[0]["failure"],
                "unknown_original_execution_without_completion"
            );
            assert!(after[0]["fit_outcome"].is_null());
            assert_eq!(entries, 2);
        }
    }
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn one_evaluation_budget_runs_the_incumbent_before_neural_candidates() {
    let (temp, prepared) = view_fixture();
    let vocabulary = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json");
    let vocab: crate::grammar::Vocab = read_json(&vocabulary).unwrap();
    let champion = PROGRAM.to_string();
    let mut seen = BTreeSet::from([champion.clone()]);
    let mut programs = Vec::new();
    for seed in 0..10000 {
        let raw = feat_math::evolve::random_canonical_pipeline(seed, 6);
        if let Ok(program) = vocab.project_proposal(&raw) {
            if seen.insert(program.clone()) {
                programs.push(program);
            }
        }
        if programs.len() == 79 {
            break;
        }
    }
    assert_eq!(programs.len(), 79);
    programs.insert(64, champion.clone());
    let p: Prepared = read_json(&prepared.join("prepared.json")).unwrap();
    let seeds = temp.path().join("seeds.json");
    write_json(
        &seeds,
        &SeedPopulation {
            schema: "parallel.seed_population.v1".into(),
            prepared_sha256: file_hash(&prepared.join("prepared.json")).unwrap(),
            cut: 0,
            training_sha256: p.cuts[0].training_sha256.clone(),
            context_sha256: "0".repeat(64),
            checkpoint_sha256: "1".repeat(64),
            candidates: programs
                .into_iter()
                .enumerate()
                .map(|(i, genome)| SeedCandidate {
                    genome,
                    origin: if i == 64 {
                        "incumbent_cut_teacher"
                    } else {
                        "neural_policy"
                    }
                    .into(),
                    policy_logprob: None,
                    proposal_record_sha256: None,
                })
                .collect(),
        },
    )
    .unwrap();
    let output = temp.path().join("teacher-first-search");
    let mut authority = Fixture::activated(temp.path(), vec![cut_training_hash(&prepared)]);
    authority.start();
    let result = authority
        .command()
        .args(["search", "--root"])
        .arg(temp.path())
        .arg("--prepared")
        .arg(&prepared)
        .args(["--cut", "0", "--output"])
        .arg(&output)
        .args([
            "--evaluations",
            "1",
            "--seconds",
            "90",
            "--qualification",
            "--seeds",
        ])
        .arg(&seeds)
        .arg("--vocabulary")
        .arg(&vocabulary)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let observations: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    assert_eq!(observations.len(), 1);
    assert_eq!(observations[0]["origin"], "incumbent_cut_teacher");
    assert_eq!(observations[0]["initial_seed_slot"], 64);
    assert_eq!(observations[0]["genome"], champion);
    assert!(observations[0]["failure"].is_null());
    authority.assert_one_original_fit();
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn deactivated_synthetic_namespace_refuses_a_configured_fit_before_any_start() {
    let tmp = tempfile::tempdir().unwrap();
    let (train, training) = raw_training(tmp.path());
    let mut authority = Fixture::activated(tmp.path(), vec![training]);
    authority.open().namespace_deactivate(SCOPE).unwrap();
    assert_eq!(authority.activation().unwrap().0, "inactive");
    authority.start();
    assert_fit_refused(&authority, authority.command(), &train, REFUSED);
    let stats = authority.stop_service();
    assert_eq!(stats["worker_spawns"], 0);
    assert_eq!(stats["committed"], 0);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn coverage_reimport_deactivates_the_synthetic_namespace_before_any_start() {
    let tmp = tempfile::tempdir().unwrap();
    let (train, training) = raw_training(tmp.path());
    let mut authority = Fixture::activated(tmp.path(), vec![training]);
    let prior = authority.coverage();
    authority.import_history(&["synthetic-tombstone.json", "synthetic-tombstone-2.json"]);
    assert_ne!(authority.coverage(), prior, "coverage was not replaced");
    assert_eq!(authority.activation().unwrap().0, "inactive");
    authority.start();
    assert_fit_refused(&authority, authority.command(), &train, REFUSED);
    let stats = authority.stop_service();
    assert_eq!(stats["worker_spawns"], 0);
    assert_eq!(stats["committed"], 0);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn out_of_scope_training_and_foreign_scope_are_refused_before_any_start() {
    let tmp = tempfile::tempdir().unwrap();
    let (train, training) = raw_training(tmp.path());
    let other = "a".repeat(64);
    assert_ne!(training, other);
    let mut authority = Fixture::activated(tmp.path(), vec![other]);
    authority.start();
    assert_fit_refused(
        &authority,
        authority.command(),
        &train,
        "training data outside authorized scope",
    );
    assert_fit_refused(
        &authority,
        authority.command_in_scope("synthetic-foreign"),
        &train,
        "scope denied",
    );
    // The namespace itself stayed active: the scope checks refused alone.
    assert_eq!(authority.activation().unwrap().0, "active");
    let stats = authority.stop_service();
    assert_eq!(stats["worker_spawns"], 0);
    assert_eq!(stats["committed"], 0);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn recovery_epoch_change_invalidates_the_synthetic_activation_before_any_start() {
    let tmp = tempfile::tempdir().unwrap();
    let (train, training) = raw_training(tmp.path());
    let mut authority = Fixture::activated(tmp.path(), vec![training]);
    let (_, activated_epoch) = authority.activation().unwrap();
    assert_eq!(activated_epoch, authority.current_epoch());
    let report = Authority::recovery_reconcile(&authority.authority, &authority.witness).unwrap();
    Authority::recovery_commit_epoch(&authority.authority, &authority.witness, &report).unwrap();
    // The row is untouched; only its binding to the prior epoch is now stale.
    assert_eq!(
        authority.activation().unwrap(),
        ("active".to_string(), activated_epoch.clone())
    );
    assert_ne!(authority.current_epoch(), activated_epoch);
    authority.start();
    assert_fit_refused(&authority, authority.command(), &train, REFUSED);
    let stats = authority.stop_service();
    assert_eq!(stats["worker_spawns"], 0);
    assert_eq!(stats["committed"], 0);
}

use super::*;

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn fitted_predictor_roundtrip_matches_native_gp_and_restores_target_units() {
    let tmp = tempfile::tempdir().unwrap();
    let (train, training) = raw_training(tmp.path());
    let model = tmp.path().join("model.json");
    let mut authority = Fixture::activated(tmp.path(), vec![training]);
    authority.start();
    success(
        authority
            .command()
            .args(["fit", "--train"])
            .arg(&train)
            .args(["--genome", PROGRAM, "--output"])
            .arg(&model)
            .output()
            .unwrap(),
    );
    authority.assert_one_original_fit();
    let features = tmp.path().join("features.csv");
    let x = vec![vec![2.0, 4.0], vec![10.0, 100.0]];
    fs::write(&features, csv_bytes(&x).unwrap()).unwrap();
    let out = tmp.path().join("predictions.csv");
    predict(&model, &features, &out).unwrap();
    let predicted = fs::read_to_string(out)
        .unwrap()
        .lines()
        .map(|s| s.parse::<f64>().unwrap())
        .collect::<Vec<_>>();
    assert!(predicted.iter().all(|p| *p > 90.0 && *p < 190.0));
    let p: Predictor = read_json(&model).unwrap();
    let decoded: Model = serde_json::from_slice(&serde_json::to_vec(&p.model).unwrap()).unwrap();
    let tx = x
        .iter()
        .map(|r| {
            p.preprocessing
                .features(&r.iter().map(f64::to_string).collect::<Vec<_>>())
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(p.model.predict(&tx), decoded.predict(&tx));
    let expected = p
        .model
        .predict(&tx)
        .unwrap()
        .iter()
        .map(|v| v * p.preprocessing.target_scale + p.preprocessing.target_mean)
        .collect::<Vec<_>>();
    assert_eq!(predicted, expected);
    let completion_ref = p.fit_completion.as_ref().unwrap();
    let completion: Value = read_json(Path::new(completion_ref["path"].as_str().unwrap())).unwrap();
    // Library half of `verify-fit-completion`: the committed original is
    // returned byte-for-byte and is bound to this authority instance only.
    let instance = authority.open().instance().to_owned();
    let verified = authority
        .open()
        .verify_committed_completion(completion_ref["sha256"].as_str().unwrap())
        .unwrap();
    assert_eq!(serde_json::to_value(&verified).unwrap(), completion);
    assert_eq!(verified.authority_instance, instance);
    assert_ne!(verified.authority_instance, "f".repeat(64));

    // Evicting a retained model leaves the original exclusion intact. Exporting
    // under a different destination cannot authorize replacement fitting.
    fs::remove_file(
        authority
            .authority
            .join("objects")
            .join(completion["model_sha256"].as_str().unwrap()),
    )
    .unwrap();
    let replacement = tmp.path().join("replacement-model.json");
    let unavailable = authority
        .command()
        .args(["fit", "--train"])
        .arg(&train)
        .args(["--genome", PROGRAM, "--output"])
        .arg(&replacement)
        .output()
        .unwrap();
    assert!(!unavailable.status.success());
    assert!(!replacement.exists());
    authority.assert_one_original_fit();
    let stats = authority.stop_service();
    assert_eq!(stats["worker_spawns"], 1);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn sealed_memory_view_matches_file_evaluation_and_rejects_writes() {
    use std::io::Write;
    use std::os::fd::AsRawFd;
    let tmp = tempfile::tempdir().unwrap();
    let rows = (0..30)
        .map(|i| vec![i as f64, (i * i) as f64, 3.0 + i as f64 * 0.8])
        .collect::<Vec<_>>();
    let mut memory = memory_view(&rows[..24], &rows[24..]).unwrap();
    assert!(memory.write_all(b"corrupt").is_err());
    let train = tmp.path().join("train.csv");
    let heldout = tmp.path().join("heldout.csv");
    let genome = tmp.path().join("p.dsl");
    fs::write(&train, csv_bytes(&rows[..24]).unwrap()).unwrap();
    fs::write(&heldout, csv_bytes(&rows[24..]).unwrap()).unwrap();
    fs::write(&genome, PROGRAM).unwrap();
    let a = tmp.path().join("a.json");
    let b = tmp.path().join("b.json");
    let mut authority = Fixture::activated(tmp.path(), vec![training_hash(&rows[..24])]);
    authority.start();
    success(
        authority
            .command()
            .args(["evaluate-worker", "--train"])
            .arg(&train)
            .arg("--validation")
            .arg(&heldout)
            .arg("--genome")
            .arg(&genome)
            .arg("--output")
            .arg(&a)
            .output()
            .unwrap(),
    );
    success(
        authority
            .command()
            .args(["evaluate-memory", "--fd"])
            .arg(memory.as_raw_fd().to_string())
            .arg("--genome")
            .arg(&genome)
            .arg("--output")
            .arg(&b)
            .output()
            .unwrap(),
    );
    let a: Value = read_json(&a).unwrap();
    let b: Value = read_json(&b).unwrap();
    assert_eq!(a["semantic_sha256"], b["semantic_sha256"]);
    assert_eq!(a["rmse"], b["rmse"]);
    assert_eq!(a["physical_fit_starts"], 1);
    assert_eq!(b["physical_fit_starts"], 0);
    assert_eq!(b["evaluation_origin"], "reuse");
    authority.assert_one_original_fit();
    let stats = authority.stop_service();
    assert_eq!(stats["worker_spawns"], 1);
    assert_eq!(stats["committed"], 1);
    assert_eq!(stats["reused"], 1);
    assert_eq!(stats["authority_opens"], 1);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn fitted_view_predictor_uses_the_exact_cut_and_original_target_units() {
    let (temp, prepared) = view_fixture();
    let output = temp.path().join("predictor.json");
    let mut authority = Fixture::activated(temp.path(), vec![cut_training_hash(&prepared)]);
    authority.start();
    success(
        authority
            .command()
            .args(["fit-view", "--root"])
            .arg(temp.path())
            .arg("--prepared")
            .arg(&prepared)
            .args(["--cut", "0", "--genome", PROGRAM, "--output"])
            .arg(&output)
            .output()
            .unwrap(),
    );
    let predictor: Predictor = read_json(&output).unwrap();
    let p: Prepared = read_json(&prepared.join("prepared.json")).unwrap();
    assert_eq!(predictor.training_sha256, p.cuts[0].training_sha256);
    let rows = load_view(&prepared, 0, true).unwrap();
    let x = rows.iter().map(|r| r[..1].to_vec()).collect::<Vec<_>>();
    let y = rows.iter().map(|r| r[1]).collect::<Vec<_>>();
    let expected = feat_math::fit(&GenomeIr::parse(PROGRAM).unwrap(), &x, &y).unwrap();
    assert_eq!(predictor.model.predict(&x), expected.predict(&x));
    let features = temp.path().join("features.csv");
    fs::write(&features, "3\n7\n").unwrap();
    let predictions = temp.path().join("predictions.csv");
    predict(&output, &features, &predictions).unwrap();
    let predictions = fs::read_to_string(predictions).unwrap();
    assert!(predictions
        .lines()
        .all(|r| r.parse::<f64>().unwrap() > 100.0));
    assert!(!prepared.join("cut-0/train.csv").exists());
    authority.assert_one_original_fit();
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn isolated_fitted_view_export_binds_success_and_preserves_worker_failures() {
    let (temp, prepared) = view_fixture();
    let executable = std::env::current_exe().unwrap();
    let mut authority = Fixture::activated(temp.path(), vec![cut_training_hash(&prepared)]);
    authority.start();
    let isolated = |genome: &str, output: &Path| {
        authority
            .command()
            .args(["fit-view-isolated", "--executable"])
            .arg(&executable)
            .arg("--root")
            .arg(temp.path())
            .arg("--prepared")
            .arg(&prepared)
            .args(["--cut", "0", "--genome", genome, "--output"])
            .arg(output)
            .output()
            .unwrap()
    };
    let output = temp.path().join("isolated.json");
    success(isolated(PROGRAM, &output));
    let receipt: Value = read_json(&output.with_extension("fit.json")).unwrap();
    assert_eq!(receipt["passed"], true);
    assert_eq!(receipt["predictor_sha256"], file_hash(&output).unwrap());
    assert_eq!(receipt["threads"], 1);
    assert_eq!(receipt["memory_limit_gib"], 8);
    let failed = temp.path().join("failed.json");
    assert!(!isolated("(pipe X (unknown))", &failed).status.success());
    let receipt: Value = read_json(&failed.with_extension("fit.json")).unwrap();
    assert_eq!(receipt["passed"], false);
    assert!(!receipt["failure"].as_str().unwrap().is_empty());
    assert!(!failed.exists());
    assert!(!fs::read_to_string(failed.with_extension("fit.log"))
        .unwrap()
        .is_empty());
    // The unknown operator is a deterministic numerical failure: it reaches the
    // authorized worker once and is retained as that original terminal outcome.
    assert_eq!(authority.entry_states(), ["completed", "terminal_failure"]);
}

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

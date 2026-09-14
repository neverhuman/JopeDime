use jope_parallel::{manifest::*, prepare::*};
use serde_json::json;
use std::{fs, path::PathBuf, process::Command};
fn fixture() -> (tempfile::TempDir, PathBuf) {
    let t = tempfile::tempdir().unwrap();
    let root = t.path();
    fs::create_dir(root.join("manifests")).unwrap();
    fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::create_dir_all(root.join("data/originals")).unwrap();
    let raw = (0..12)
        .map(|i| vec![i.to_string(), (100 + i * 3).to_string()])
        .collect::<Vec<_>>();
    let mut source = "x,target\n".to_string();
    for row in &raw {
        source.push_str(&format!("{},{}\n", row[0], row[1]));
    }
    fs::write(root.join("data/originals/raw.csv"), source).unwrap();
    let pp = Preprocessing::fit(&raw[..8], "regression").unwrap();
    let prepared = root.join("data/prepared/example");
    let dir = prepared.join("cut-0");
    fs::create_dir_all(&dir).unwrap();
    write_json(&dir.join("preprocessing.json"), &pp).unwrap();
    let tr = csv_bytes(&pp.transform(&raw[..8]).unwrap()).unwrap();
    let va = csv_bytes(&pp.transform(&raw[8..10]).unwrap()).unwrap();
    let p = json!({"schema":"parallel.prepared.v1","dataset":"example","family":"example","task":"regression","meta_partition":"meta_train","seed":SEED,"family_partition_frozen":false,"clean_evaluation":true,"clean_status":"raw_reconstructed","original_train_sha256":"0".repeat(64),"original_test_sha256":"1".repeat(64),"selected_train_indices":(0..10).collect::<Vec<_>>(),"selected_test_indices":[0,1],"target_last":true,"rows_total":12,"cuts":[{"index":0,"train_indices":(0..8).collect::<Vec<_>>(),"validation_indices":[8,9],"training_sha256":hash(&tr),"validation_sha256":hash(&va),"preprocessing_sha256":file_hash(&dir.join("preprocessing.json")).unwrap()}],"storage":{"source":"data/originals/raw.csv","source_sha256":file_hash(&root.join("data/originals/raw.csv")).unwrap(),"feature_columns":["x"],"target_column":"target","original_training_row_map":(0..10).collect::<Vec<_>>()}});
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    (t, prepared)
}

#[test]
fn one_evaluation_budget_runs_the_incumbent_before_neural_candidates() {
    let (temp, prepared) = fixture();
    let vocabulary = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json");
    let vocab: jope_parallel::grammar::Vocab = read_json(&vocabulary).unwrap();
    let champion = "(pipe X (ridge))".to_string();
    let mut seen = std::collections::BTreeSet::from([champion.clone()]);
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
        &jope_parallel::gp::SeedPopulation {
            schema: "parallel.seed_population.v1".into(),
            prepared_sha256: file_hash(&prepared.join("prepared.json")).unwrap(),
            cut: 0,
            training_sha256: p.cuts[0].training_sha256.clone(),
            context_sha256: "0".repeat(64),
            checkpoint_sha256: "1".repeat(64),
            candidates: programs
                .into_iter()
                .enumerate()
                .map(|(i, genome)| jope_parallel::gp::SeedCandidate {
                    genome,
                    origin: if i == 64 {
                        "incumbent_cut_teacher"
                    } else {
                        "neural_policy"
                    }
                    .into(),
                    policy_logprob: None,
                })
                .collect(),
        },
    )
    .unwrap();
    let output = temp.path().join("teacher-first-search");
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
        .env("JOPE_CPU_EVALUATOR", env!("CARGO_BIN_EXE_jope-parallel"))
        .arg("--root")
        .arg(temp.path())
        .args(["search", "--prepared"])
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
    let observations: Vec<serde_json::Value> =
        read_json(&output.join("observations.json")).unwrap();
    assert_eq!(observations.len(), 1);
    assert_eq!(observations[0]["origin"], "incumbent_cut_teacher");
    assert_eq!(observations[0]["initial_seed_slot"], 64);
    assert_eq!(observations[0]["genome"], champion);
    assert!(observations[0]["failure"].is_null());
}
#[test]
fn pointer_views_use_no_csv_copies_and_heldout_targets_do_not_enter_training() {
    let (temp, p) = fixture();
    let before = load_view(&p, 0, true).unwrap();
    assert!(!p.join("cut-0/train.csv").exists());
    assert!(!p.join("cut-0/validation.csv").exists());
    let raw = temp.path().join("data/originals/raw.csv");
    let contents = fs::read_to_string(&raw)
        .unwrap()
        .replace("11,133", "11,9999999");
    fs::write(&raw, contents).unwrap();
    assert!(load_view(&p, 0, true)
        .unwrap_err()
        .to_string()
        .contains("backing table changed"));
    // A new source revision changes held-out labels only. The exact training
    // byte hash and all cut preprocessing must still reproduce unchanged.
    let mut manifest: Prepared = read_json(&p.join("prepared.json")).unwrap();
    manifest.storage.as_mut().unwrap().source_sha256 = file_hash(&raw).unwrap();
    write_json(&p.join("prepared.json"), &manifest).unwrap();
    assert_eq!(load_view(&p, 0, true).unwrap(), before);
}

#[test]
fn qualification_preparation_uses_audited_test_identity_without_opening_test_csv() {
    let (temp, _) = fixture();
    let root = temp.path();
    let source = root.join("data/originals/raw.csv");
    let legacy = root.join("data/originals/regression/example");
    fs::create_dir_all(&legacy).unwrap();
    let original = (0..10)
        .map(|i| vec![i as f64 / 11.0, i as f64 / 11.0])
        .collect::<Vec<_>>();
    fs::write(legacy.join("train.csv"), csv_bytes(&original).unwrap()).unwrap();
    let metadata = json!({"source_path":source,"target_generation":"original_numeric_column","feature_columns":["x"],"target_column":"target","feature_encodings":{"x":{"kind":"numeric","impute_value":0.0}},"feature_scaling":{"min":[0.0],"max":[11.0]},"target_scaling":{"min":100.0,"max":133.0}});
    write_json(&legacy.join("meta.json"), &metadata).unwrap();
    let test_stats = json!({"rows":2,"columns":2,"target_index":1,"header":false,"missing_features":0,"invalid_targets":0,"duplicate_rows":0,"conflicting_features":0,"target_values":[]});
    let mut audit = json!({"schema":"parallel.audit.v1","seed":SEED,"roots":[],"datasets":[{"id":"example","task":"regression","source_dir":legacy,"source_key":"example","family":"example","parents":[],"admission":"approved","reason":"test fixture","evidence":[],"metadata_sha256":file_hash(&legacy.join("meta.json")).unwrap(),"train":null,"test":test_stats,"split_feature_overlap":1,"issues":[],"meta_partition":"meta_train"}],"artifact_counts":{},"missing_roots":[],"complete":false});
    let mut acquisition_stub = audit["datasets"][0].clone();
    acquisition_stub["source_dir"] = json!(root.join("acquisition/example-stub"));
    acquisition_stub["metadata_sha256"] = json!("f".repeat(64));
    audit["datasets"]
        .as_array_mut()
        .unwrap()
        .insert(0, acquisition_stub);
    let audit_path = root.join("manifests/audit.json");
    write_json(&audit_path, &audit).unwrap();
    write_json(&root.join("manifests/transfer.json"), &json!({"schema":"parallel.transfer.v1","files":[{"source":legacy.join("test.csv"),"destination":"data/originals/legacy/test.csv","bytes":32,"sha256":"1".repeat(64),"dataset":"example","role":"original_test"}]})).unwrap();
    let backing = root.join("manifests/backing.json");
    write_json(&backing, &json!({"schema":"parallel.transfer.v1","files":[{"source":source,"destination":"data/originals/raw.csv","bytes":fs::metadata(&source).unwrap().len(),"sha256":file_hash(&source).unwrap(),"dataset":"example","role":"backing_source"}]})).unwrap();
    let strict = root.join("data/prepared/strict");
    assert!(run(&audit_path, "example", &strict, false, false, &backing).is_err());
    assert!(!strict.exists());
    let qualified = root.join("data/prepared/qualified");
    let p = run(&audit_path, "example", &qualified, false, true, &backing).unwrap();
    assert!(p.qualification_only && !p.clean_evaluation);
    assert_eq!(p.original_test_sha256, "1".repeat(64));
    assert_eq!(p.cuts.len(), 5);
    assert!(!legacy.join("test.csv").exists());
    for cut in &p.cuts {
        load_view(&qualified, cut.index, true).unwrap();
        load_view(&qualified, cut.index, false).unwrap();
        assert!(!qualified
            .join(format!("cut-{}/train.csv", cut.index))
            .exists());
        assert!(!qualified
            .join(format!("cut-{}/validation.csv", cut.index))
            .exists());
    }
    let receipt: serde_json::Value = read_json(&qualified.join("reconstruction.json")).unwrap();
    assert_eq!(receipt["original_test_targets_opened"], false);
}

#[test]
fn expanding_pointer_cuts_preserves_entity_groups_with_distinct_features() {
    let (temp, prepared) = fixture();
    let raw_path = temp.path().join("data/originals/raw.csv");
    let mut raw = "subject,x,target\n".to_string();
    for i in 0..12 {
        raw.push_str(&format!("{},{},{}\n", i / 2, i, 100 + i * 3));
    }
    fs::write(&raw_path, raw).unwrap();
    let mut p: Prepared = read_json(&prepared.join("prepared.json")).unwrap();
    let storage = p.storage.as_mut().unwrap();
    storage.source_sha256 = file_hash(&raw_path).unwrap();
    storage.feature_columns = vec!["subject".into(), "x".into()];
    p.entity_columns = vec!["subject".into()];
    let rows = load_raw_view(
        &prepared,
        p.storage.as_ref().unwrap(),
        &p.selected_train_indices,
    )
    .unwrap();
    let entities = rows.iter().map(|r| r[0].clone()).collect::<Vec<_>>();
    p.cuts.clear();
    for (index, (train, heldout)) in folds(&rows, "regression", 1, Some(&entities))
        .unwrap()
        .into_iter()
        .enumerate()
    {
        let training = train.iter().map(|&i| rows[i].clone()).collect::<Vec<_>>();
        let validation = heldout.iter().map(|&i| rows[i].clone()).collect::<Vec<_>>();
        let pp = Preprocessing::fit(&training, "regression").unwrap();
        let dir = prepared.join(format!("cut-{index}"));
        fs::create_dir_all(&dir).unwrap();
        write_json(&dir.join("preprocessing.json"), &pp).unwrap();
        p.cuts.push(Cut {
            index,
            train_indices: train,
            validation_indices: heldout,
            training_sha256: hash(&csv_bytes(&pp.transform(&training).unwrap()).unwrap()),
            validation_sha256: hash(&csv_bytes(&pp.transform(&validation).unwrap()).unwrap()),
            preprocessing_sha256: file_hash(&dir.join("preprocessing.json")).unwrap(),
        });
    }
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    let expanded = temp.path().join("data/prepared/entity-expanded");
    let result = expand_cuts(&prepared, &expanded, 3).unwrap();
    assert_eq!(result.entity_columns, p.entity_columns);
    assert_eq!(result.family, p.family);
    for cut in &result.cuts {
        let train_entities = cut
            .train_indices
            .iter()
            .map(|i| i / 2)
            .collect::<std::collections::BTreeSet<_>>();
        assert!(cut
            .validation_indices
            .iter()
            .all(|i| !train_entities.contains(&(i / 2))));
        load_view(&expanded, cut.index, true).unwrap();
        load_view(&expanded, cut.index, false).unwrap();
    }
}

#[test]
fn fitted_view_predictor_uses_the_exact_cut_and_original_target_units() {
    let (temp, prepared) = fixture();
    let output = temp.path().join("predictor.json");
    jope_parallel::gp::fit_view(&prepared, 0, "(pipe X (ridge))", &output).unwrap();
    let predictor: jope_parallel::gp::Predictor = read_json(&output).unwrap();
    let p: Prepared = read_json(&prepared.join("prepared.json")).unwrap();
    assert_eq!(predictor.training_sha256, p.cuts[0].training_sha256);
    let rows = load_view(&prepared, 0, true).unwrap();
    let x = rows.iter().map(|r| r[..1].to_vec()).collect::<Vec<_>>();
    let y = rows.iter().map(|r| r[1]).collect::<Vec<_>>();
    let expected = feat_math::fit(
        &feat_math::GenomeIr::parse("(pipe X (ridge))").unwrap(),
        &x,
        &y,
    )
    .unwrap();
    assert_eq!(predictor.model.predict(&x), expected.predict(&x));
    let features = temp.path().join("features.csv");
    fs::write(&features, "3\n7\n").unwrap();
    let predictions = temp.path().join("predictions.csv");
    jope_parallel::gp::predict(&output, &features, &predictions).unwrap();
    let predictions = fs::read_to_string(predictions).unwrap();
    assert!(predictions
        .lines()
        .all(|r| r.parse::<f64>().unwrap() > 100.0));
    assert!(!prepared.join("cut-0/train.csv").exists());
}

#[test]
fn isolated_fitted_view_export_binds_success_and_preserves_worker_failures() {
    let (temp, prepared) = fixture();
    let executable = PathBuf::from(env!("CARGO_BIN_EXE_jope-parallel"));
    let output = temp.path().join("isolated.json");
    jope_parallel::gp::fit_view_isolated(&executable, &prepared, 0, "(pipe X (ridge))", &output)
        .unwrap();
    let receipt: serde_json::Value = read_json(&output.with_extension("fit.json")).unwrap();
    assert_eq!(receipt["passed"], true);
    assert_eq!(receipt["predictor_sha256"], file_hash(&output).unwrap());
    assert_eq!(receipt["threads"], 1);
    assert_eq!(receipt["memory_limit_gib"], 8);
    let failed = temp.path().join("failed.json");
    assert!(jope_parallel::gp::fit_view_isolated(
        &executable,
        &prepared,
        0,
        "(pipe X (unknown))",
        &failed
    )
    .is_err());
    let receipt: serde_json::Value = read_json(&failed.with_extension("fit.json")).unwrap();
    assert_eq!(receipt["passed"], false);
    assert!(!receipt["failure"].as_str().unwrap().is_empty());
    assert!(!failed.exists());
    assert!(!fs::read_to_string(failed.with_extension("fit.log"))
        .unwrap()
        .is_empty());
}
#[test]
fn gp_resumes_its_ledger_without_repeating_completed_evaluations() {
    let (temp, p) = fixture();
    let out = temp.path().join("search");
    let run = |count: &str| {
        Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
            .env("JOPE_CPU_EVALUATOR", env!("CARGO_BIN_EXE_jope-parallel"))
            .arg("--root")
            .arg(temp.path())
            .arg("search")
            .arg("--prepared")
            .arg(&p)
            .args([
                "--cut",
                "0",
                "--evaluations",
                count,
                "--seconds",
                "90",
                "--qualification",
                "--output",
            ])
            .arg(&out)
            .output()
            .unwrap()
    };
    let first = run("2");
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let original: Vec<serde_json::Value> = read_json(&out.join("observations.json")).unwrap();
    assert_eq!(original.len(), 2);
    fs::remove_file(out.join("search.json")).unwrap();
    let second = run("2");
    assert!(
        second.status.success(),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );
    let resumed: Vec<serde_json::Value> = read_json(&out.join("observations.json")).unwrap();
    assert_eq!(resumed, original);
    assert!(
        original[1]["completed_search_seconds"].as_f64().unwrap()
            >= original[0]["completed_search_seconds"].as_f64().unwrap()
    );
    fs::remove_file(out.join("search.json")).unwrap();
    let changed = run("3");
    assert!(!changed.status.success());
    assert!(String::from_utf8_lossy(&changed.stderr)
        .contains("accepted search artifact root or request changed"));
    assert_eq!(
        read_json::<Vec<serde_json::Value>>(&out.join("observations.json")).unwrap(),
        original
    );
}

fn clock_search(
    prepared: &std::path::Path,
    output: &std::path::Path,
    evaluations: &str,
) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
        .env("JOPE_CPU_EVALUATOR", env!("CARGO_BIN_EXE_jope-parallel"))
        .arg("--root")
        .arg(workspace_root(prepared).unwrap())
        .args([
            "search",
            "--cut",
            "0",
            "--seconds",
            "90",
            "--qualification",
            "--evaluations",
            evaluations,
        ])
        .arg("--prepared")
        .arg(prepared)
        .arg("--output")
        .arg(output)
        .env("RAYON_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .output()
        .unwrap()
}

#[test]
fn gp_resume_restores_time_from_a_journal_ahead_of_progress_without_new_budget() {
    let (temp, prepared) = fixture();
    let output = temp.path().join("search");
    assert!(clock_search(&prepared, &output, "2").status.success());
    let mut observations: Vec<serde_json::Value> =
        read_json(&output.join("observations.json")).unwrap();
    // Simulate a crash after the journal fsync but before progress was updated.
    observations[0]["completed_search_seconds"] = json!(89.4);
    observations[1]["completed_search_seconds"] = json!(89.5);
    let journal = observations
        .iter()
        .map(|o| format!("{}\n", o))
        .collect::<String>();
    fs::write(output.join("observations.jsonl"), journal).unwrap();
    let mut progress: serde_json::Value = read_json(&output.join("progress.json")).unwrap();
    progress["wall_seconds"] = json!(0.0);
    write_json(&output.join("progress.json"), &progress).unwrap();
    fs::remove_file(output.join("search.json")).unwrap();
    let resumed = clock_search(&prepared, &output, "2");
    assert!(
        resumed.status.success(),
        "{}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    let saved: Vec<serde_json::Value> = read_json(&output.join("observations.json")).unwrap();
    assert_eq!(saved, observations);
    let summary: serde_json::Value = read_json(&output.join("search.json")).unwrap();
    assert_eq!(summary["completed_evaluations"], 2);
    assert_eq!(summary["search_clock"]["resume_offset_seconds"], 119.5);
    assert!(summary["wall_seconds"].as_f64().unwrap() >= 119.5);
    let progress: serde_json::Value = read_json(&output.join("progress.json")).unwrap();
    assert!(progress["wall_seconds"].as_f64().unwrap() >= 119.5);
}

#[test]
fn gp_legacy_resume_preserves_rows_and_declares_inexact_wall_prefix() {
    let (temp, prepared) = fixture();
    let output = temp.path().join("search");
    assert!(clock_search(&prepared, &output, "2").status.success());
    let mut original: Vec<serde_json::Value> =
        read_json(&output.join("observations.json")).unwrap();
    for row in &mut original {
        for key in [
            "completed_search_seconds",
            "search_clock",
            "search_clock_accuracy",
        ] {
            row.as_object_mut().unwrap().remove(key);
        }
    }
    write_json(&output.join("observations.json"), &original).unwrap();
    fs::remove_file(output.join("observations.jsonl")).unwrap();
    fs::remove_file(output.join("search.json")).unwrap();
    let resumed = clock_search(&prepared, &output, "2");
    assert!(
        resumed.status.success(),
        "{}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    let saved: Vec<serde_json::Value> = read_json(&output.join("observations.json")).unwrap();
    assert_eq!(saved, original);
    let summary: serde_json::Value = read_json(&output.join("search.json")).unwrap();
    assert_eq!(
        summary["search_clock"]["legacy_observations_without_clock"],
        2
    );
    assert_eq!(
        summary["search_clock"]["exact_wall_prefix_available"],
        false
    );
}

#[test]
fn gp_resume_rejects_a_nonmonotonic_durable_search_clock() {
    let (temp, prepared) = fixture();
    let output = temp.path().join("search");
    assert!(clock_search(&prepared, &output, "2").status.success());
    let mut observations: Vec<serde_json::Value> =
        read_json(&output.join("observations.json")).unwrap();
    observations[0]["completed_search_seconds"] = json!(20.0);
    observations[1]["completed_search_seconds"] = json!(10.0);
    write_json(&output.join("observations.json"), &observations).unwrap();
    fs::write(
        output.join("observations.jsonl"),
        observations
            .iter()
            .map(|row| format!("{row}\n"))
            .collect::<String>(),
    )
    .unwrap();
    fs::remove_file(output.join("search.json")).unwrap();
    let resumed = clock_search(&prepared, &output, "2");
    assert!(!resumed.status.success());
    assert!(String::from_utf8_lossy(&resumed.stderr).contains("not monotonic"));
    assert!(!output.join("search.json").exists());
}

#[test]
fn gp_service_restores_all_durable_progress_before_dispatch_and_new_priority_admission() {
    let (temp, prepared) = fixture();
    let root = temp.path();
    fs::create_dir_all(root.join("runs")).unwrap();
    let job = |name: &str| json!({"prepared":"data/prepared/example","prepared_sha256":file_hash(&prepared.join("prepared.json")).unwrap(),"cut":0,"output":format!("runs/{name}")});
    let manifest = |jobs: Vec<serde_json::Value>| json!({"schema":"parallel.gp_jobs.v1","qualification":true,"jobs":jobs});
    for (name, evaluations) in [
        ("old-complete-a", 10),
        ("old-complete-b", 20),
        ("new-complete", 7),
    ] {
        write_json(
            &root.join(format!("runs/{name}/search.json")),
            &json!({"schema":"parallel.search.v1","completed_evaluations":evaluations}),
        )
        .unwrap();
    }
    let base = root.join("manifests/jobs.json");
    write_json(
        &base,
        &manifest(vec![
            job("old-pending-a"),
            job("old-complete-a"),
            job("old-pending-b"),
            job("old-complete-b"),
        ]),
    )
    .unwrap();
    let priority = root.join("runs/gp-priority-jobs.json");
    write_json(
        &priority,
        &manifest(vec![job("priority-pending"), job("old-complete-b")]),
    )
    .unwrap();
    struct Running(std::process::Child);
    impl Drop for Running {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    // No hardware heartbeat deliberately keeps admission paused. Durable
    // progress must be correct even before a single worker can be dispatched.
    let mut child = Running(
        Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
            .env("JOPE_CPU_EVALUATOR", env!("CARGO_BIN_EXE_jope-parallel"))
            .arg("--root")
            .arg(root)
            .args(["gp-service", "--workers", "1", "--threads", "1"])
            .arg("--manifest")
            .arg(&base)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let wait = |expected_completed: u64| -> serde_json::Value {
        let started = std::time::Instant::now();
        loop {
            let state: serde_json::Value =
                read_json(&root.join("runs/gp-service.json")).unwrap_or_default();
            if state["completed_views"].as_u64() == Some(expected_completed) {
                return state;
            }
            assert!(
                started.elapsed().as_secs() < 10,
                "service did not restore progress: {state}"
            );
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    };
    let first = wait(2);
    assert_eq!(first["completed_evaluations"], 30);
    assert_eq!(first["queued"], 3);
    assert_eq!(first["active"], 0);
    assert_eq!(first["evaluations_per_second"], 0.0);
    write_json(
        &priority,
        &manifest(vec![
            job("priority-pending"),
            job("new-complete"),
            job("new-pending"),
            job("old-complete-a"),
        ]),
    )
    .unwrap();
    let second = wait(3);
    assert_eq!(second["completed_evaluations"], 37);
    assert_eq!(second["restored_completed_views"], 3);
    assert_eq!(second["queued"], 4);
    assert_eq!(second["evaluations_per_second"], 0.0);
    assert!(child.0.try_wait().unwrap().is_none());
}

#[test]
fn qualification_only_preparation_never_enters_strict_gp_search() {
    let (temp, prepared) = fixture();
    let mut p: Prepared = read_json(&prepared.join("prepared.json")).unwrap();
    p.clean_evaluation = false;
    p.qualification_only = true;
    p.family_partition_frozen = true;
    p.clean_status = "raw_training_reconstructed_qualification_only".into();
    p.evaluation_limitations =
        vec!["original_test_feature_overlap:1; final evaluation remains ineligible".into()];
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    let run = |qualification: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_jope-parallel"));
        command.env("JOPE_CPU_EVALUATOR", env!("CARGO_BIN_EXE_jope-parallel"));
        command
            .arg("--root")
            .arg(temp.path())
            .args([
                "search",
                "--cut",
                "0",
                "--evaluations",
                "1",
                "--seconds",
                "90",
            ])
            .arg("--prepared")
            .arg(&prepared)
            .arg("--output")
            .arg(
                temp.path()
                    .join(if qualification { "qualified" } else { "strict" }),
            );
        if qualification {
            command.arg("--qualification");
        }
        command.output().unwrap()
    };
    assert!(!run(false).status.success());
    let accepted = run(true);
    assert!(
        accepted.status.success(),
        "{}",
        String::from_utf8_lossy(&accepted.stderr)
    );
    let observations: Vec<serde_json::Value> =
        read_json(&temp.path().join("qualified/observations.json")).unwrap();
    assert_eq!(observations[0]["qualification"], true);
    assert_eq!(observations[0]["independent_evaluation"], false);
    let stored: Prepared = read_json(&prepared.join("prepared.json")).unwrap();
    assert!(!stored.clean_evaluation && stored.qualification_only);
}

#[test]
fn projected_gp_candidates_are_exactly_representable_unique_and_durable() {
    let (temp, p) = fixture();
    let output = temp.path().join("projected-search");
    let vocabulary = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json");
    let result = Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
        .env("JOPE_CPU_EVALUATOR", env!("CARGO_BIN_EXE_jope-parallel"))
        .arg("--root")
        .arg(temp.path())
        .args([
            "search",
            "--cut",
            "0",
            "--evaluations",
            "100",
            "--seconds",
            "90",
            "--qualification",
        ])
        .arg("--prepared")
        .arg(&p)
        .arg("--output")
        .arg(&output)
        .arg("--vocabulary")
        .arg(&vocabulary)
        .env("RAYON_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let observations: Vec<serde_json::Value> =
        read_json(&output.join("observations.json")).unwrap();
    assert_eq!(observations.len(), 100);
    let vocab: jope_parallel::grammar::Vocab = read_json(&vocabulary).unwrap();
    let mut programs = std::collections::BTreeSet::new();
    for (index, observation) in observations.iter().enumerate() {
        let genome = observation["genome"].as_str().unwrap();
        vocab.encode(genome).unwrap();
        assert!(programs.insert(feat_math::GenomeIr::parse(genome).unwrap().canonical()));
        assert_eq!(observation["evaluation_index"], index);
    }
    let progress: serde_json::Value = read_json(&output.join("progress.json")).unwrap();
    assert_eq!(
        progress["vocabulary_sha256"],
        file_hash(&vocabulary).unwrap()
    );
    assert_eq!(progress["evaluations"], 100);
    assert!(!output.join("observations.jsonl").exists());
    assert!(fs::read_dir(output).unwrap().all(|entry| !entry
        .unwrap()
        .file_name()
        .to_string_lossy()
        .starts_with("candidate-")));
}

#[test]
fn repeated_folds_are_reproducible_distinct_and_pointer_only() {
    let (temp, base) = fixture();
    let mut p: Prepared = read_json(&base.join("prepared.json")).unwrap();
    let raw = load_raw_view(
        &base,
        p.storage.as_ref().unwrap(),
        &p.selected_train_indices,
    )
    .unwrap();
    p.cuts.clear();
    for (index, (tr, va)) in folds(&raw, "regression", 1, None)
        .unwrap()
        .into_iter()
        .enumerate()
    {
        let training = tr.iter().map(|&i| raw[i].clone()).collect::<Vec<_>>();
        let heldout = va.iter().map(|&i| raw[i].clone()).collect::<Vec<_>>();
        let pp = Preprocessing::fit(&training, "regression").unwrap();
        let dir = base.join(format!("cut-{index}"));
        fs::create_dir_all(&dir).unwrap();
        write_json(&dir.join("preprocessing.json"), &pp).unwrap();
        p.cuts.push(Cut {
            index,
            train_indices: tr,
            validation_indices: va,
            training_sha256: hash(&csv_bytes(&pp.transform(&training).unwrap()).unwrap()),
            validation_sha256: hash(&csv_bytes(&pp.transform(&heldout).unwrap()).unwrap()),
            preprocessing_sha256: file_hash(&dir.join("preprocessing.json")).unwrap(),
        });
    }
    write_json(&base.join("prepared.json"), &p).unwrap();
    let first = temp.path().join("data/prepared/expanded-a");
    let second = temp.path().join("data/prepared/expanded-b");
    let expanded = expand_cuts(&base, &first, 5).unwrap();
    expand_cuts(&base, &second, 5).unwrap();
    assert_eq!(
        fs::read(first.join("prepared.json")).unwrap(),
        fs::read(second.join("prepared.json")).unwrap()
    );
    assert!(expanded.cuts.len() > 5);
    let mut memberships = std::collections::BTreeSet::new();
    for cut in &expanded.cuts {
        assert!(memberships.insert(cut.train_indices.clone()));
        assert!(cut
            .train_indices
            .iter()
            .all(|i| !cut.validation_indices.contains(i)));
        assert!(!first.join(format!("cut-{}/train.csv", cut.index)).exists());
        load_view(&first, cut.index, true).unwrap();
        load_view(&first, cut.index, false).unwrap();
    }
    assert_eq!(expanded.rows_total, p.rows_total);
    assert_eq!(expanded.family, p.family);
}

#[test]
fn gp_journal_preserves_a_torn_tail_and_requires_reconciliation() {
    let t = tempfile::tempdir().unwrap();
    let p = t.path().join("observations.jsonl");
    fs::write(
        &p,
        b"{\"evaluation_index\":0}\n{\"evaluation_index\":1}\n{\"evaluation_",
    )
    .unwrap();
    let original = fs::read(&p).unwrap();
    assert!(jope_parallel::gp::read_journal(&p)
        .unwrap_err()
        .to_string()
        .contains("reconciliation"));
    assert_eq!(fs::read(&p).unwrap(), original);
    fs::write(&p, b"{\"evaluation_index\":0}\n{\"evaluation_index\":1}\n").unwrap();
    let records = jope_parallel::gp::read_journal(&p).unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[1]["evaluation_index"], 1);
    assert!(fs::read(&p).unwrap().ends_with(b"\n"));
    fs::write(&p, b"{broken}\n").unwrap();
    assert!(jope_parallel::gp::read_journal(&p).is_err());
    assert_eq!(fs::read(&p).unwrap(), b"{broken}\n");
}

#[test]
fn experiment_crashes_at_durable_boundaries_recover_without_repeating_committed_fits() {
    for boundary in [
        "candidate_intent",
        "candidate_metric",
        "observation_journal",
        "observation_progress",
        "observations_snapshot",
        "search_summary",
    ] {
        let (temp, prepared) = fixture();
        let experiment = temp.path().join("experiment");
        fs::create_dir_all(&experiment).unwrap();
        write_json(
            &experiment.join("manifest.json"),
            &json!({"schema":"parallel.seed_experiment.v2"}),
        )
        .unwrap();
        let output = experiment.join("search");
        let command = || {
            let mut c = Command::new(env!("CARGO_BIN_EXE_jope-parallel"));
            c.env("JOPE_CPU_EVALUATOR", env!("CARGO_BIN_EXE_jope-parallel"));
            c.arg("--root")
                .arg(temp.path())
                .args([
                    "search",
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
                .env("JOPE_EXPERIMENT_DIR", &experiment)
                .env("OMP_NUM_THREADS", "1")
                .env("RAYON_NUM_THREADS", "1");
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
            jope_parallel::gp::read_journal(&output.join("observations.jsonl")).unwrap()
        } else {
            vec![]
        };
        let metric: Option<serde_json::Value> = if boundary == "candidate_metric" {
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
        let after: Vec<serde_json::Value> = read_json(&output.join("observations.json")).unwrap();
        assert_eq!(after.len(), 3, "{boundary}");
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
    }
}

#[test]
fn matched_search_seed_is_recorded_and_cannot_change_on_resume() {
    let (temp, prepared) = fixture();
    let output = temp.path().join("seeded");
    let run = |seed: &str| {
        Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
            .env("JOPE_CPU_EVALUATOR", env!("CARGO_BIN_EXE_jope-parallel"))
            .arg("--root")
            .arg(temp.path())
            .args([
                "search",
                "--cut",
                "0",
                "--evaluations",
                "2",
                "--seconds",
                "180",
                "--qualification",
                "--rng-seed",
                seed,
            ])
            .arg("--prepared")
            .arg(&prepared)
            .arg("--output")
            .arg(&output)
            .output()
            .unwrap()
    };
    let initial = run("20260908");
    assert!(
        initial.status.success(),
        "{}",
        String::from_utf8_lossy(&initial.stderr)
    );
    let search: serde_json::Value = read_json(&output.join("search.json")).unwrap();
    assert_eq!(search["rng_seed"], 20260908);
    let rows: Vec<serde_json::Value> = read_json(&output.join("observations.json")).unwrap();
    assert!(rows.iter().all(|row| row["rng_seed"] == 20260908));
    fs::remove_file(output.join("search.json")).unwrap();
    let changed = run("20260909");
    assert!(!changed.status.success());
    assert!(String::from_utf8_lossy(&changed.stderr).contains("RNG seed mismatch"));
}

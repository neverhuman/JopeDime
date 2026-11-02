use super::*;

/// Shipped build. Teacher-first ordering is a search property and holds even
/// though every fit is refused without a verified activation; the fitted
/// outcome of the same budget is proven by
/// `gp::fit_contract_tests::one_evaluation_budget_runs_the_incumbent_before_neural_candidates`.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn fenced_one_evaluation_budget_orders_the_incumbent_first_and_records_its_refusal() {
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
                    proposal_record_sha256: None,
                })
                .collect(),
        },
    )
    .unwrap();
    let output = temp.path().join("teacher-first-search");
    let mut authority = authority::Fixture::new(
        temp.path(),
        vec![authority::training_hash(
            &load_view(&prepared, 0, true).unwrap(),
        )],
    );
    let result = authority
        .command()
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
    assert!(
        !observations[0]["failure"].is_null(),
        "shipped search fitted without a verified activation"
    );
    assert!(observations[0]["metrics"].is_null());
    let diagnostic = observations[0]["worker_diagnostic"].as_str().unwrap();
    assert!(
        diagnostic.contains(authority::UNVERIFIED_ACTIVATION),
        "{diagnostic}"
    );
    let summary: serde_json::Value = read_json(&output.join("search.json")).unwrap();
    assert_eq!(summary["successful_measurements"], 0);
    authority.assert_no_fit_starts();
    let stats = authority.shutdown();
    assert_eq!(stats["worker_spawns"], 0);
    assert_eq!(stats["committed"], 0);
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

/// Shipped build with no fit service binding: the boundary refuses before any
/// output, both in-process and inside the isolated export worker, whose failure
/// is still receipted. The positive fitted-view and isolated-export contracts
/// are `gp::fit_contract_tests::{fitted_view_predictor_uses_the_exact_cut_and_original_target_units,
/// isolated_fitted_view_export_binds_success_and_preserves_worker_failures}`.
#[test]
fn unbound_fitted_view_export_is_fenced_before_output_and_receipted() {
    assert!(
        std::env::var_os("JOPE_FIT_SERVICE_SOCKET").is_none(),
        "this contract requires an environment without a fit service binding"
    );
    let (temp, prepared) = fixture();
    // The pointer view is projected under its declared root (W1 route R2).
    let output = temp.path().join("predictor.json");
    let error =
        jope_parallel::gp::fit_view_in_root(temp.path(), &prepared, 0, authority::PROGRAM, &output)
            .unwrap_err()
            .to_string();
    assert!(error.contains(FENCED), "{error}");
    assert!(!output.exists());
    let executable = PathBuf::from(env!("CARGO_BIN_EXE_jope-parallel"));
    let isolated = temp.path().join("isolated.json");
    assert!(jope_parallel::gp::fit_view_isolated_in_root(
        &executable,
        temp.path(),
        &prepared,
        0,
        authority::PROGRAM,
        &isolated
    )
    .is_err());
    let receipt: serde_json::Value = read_json(&isolated.with_extension("fit.json")).unwrap();
    assert_eq!(receipt["passed"], false);
    assert_eq!(
        receipt["executable_sha256"],
        file_hash(&executable).unwrap()
    );
    assert!(receipt["predictor_sha256"].is_null());
    assert!(!isolated.exists());
    let log = fs::read_to_string(isolated.with_extension("fit.log")).unwrap();
    assert!(log.contains(FENCED), "{log}");
}

/// Shipped build: a fully configured, in-scope authority still refuses the cut
/// fit at namespace activation; nothing is claimed, started or published.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn configured_authority_refuses_fit_view_without_verified_activation() {
    let (temp, prepared) = fixture();
    let output = temp.path().join("predictor.json");
    let mut authority = authority::Fixture::new(
        temp.path(),
        vec![authority::training_hash(
            &load_view(&prepared, 0, true).unwrap(),
        )],
    );
    let result = authority
        .command()
        .arg("--root")
        .arg(temp.path())
        .args(["fit-view", "--prepared"])
        .arg(&prepared)
        .args(["--cut", "0", "--genome", authority::PROGRAM, "--output"])
        .arg(&output)
        .output()
        .unwrap();
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(
        !result.status.success(),
        "shipped fit-view ran without activation"
    );
    assert!(error.contains(authority::UNVERIFIED_ACTIVATION), "{error}");
    assert!(!output.exists(), "refused fit-view published a predictor");
    authority.assert_no_fit_starts();
    let stats = authority.shutdown();
    assert_eq!(stats["worker_spawns"], 0);
    assert_eq!(stats["committed"], 0);
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

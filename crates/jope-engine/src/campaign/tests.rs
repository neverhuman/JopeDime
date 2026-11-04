use super::*;
#[test]
fn restart_does_not_extend_deadline_or_admission() {
    let s = Schedule {
        spec_sha256: "a".into(),
        contexts_sha256: "b".into(),
        initial_corpus_sha256: "c".into(),
        vocabulary_sha256: "d".into(),
        executable_sha256: "e".into(),
        gp_executable_sha256: "g".into(),
        predictor_executable_sha256: "p".into(),
        started_at: 100,
        stop_admission_at: 200,
        deadline: 300,
    };
    let restored: Schedule = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
    assert!(restored.may_start(199));
    assert!(!restored.may_start(200));
    assert!(!restored.may_start(1000));
    assert_eq!(s, restored);
    let mut restart = s.clone();
    restart.started_at = 900;
    restart.stop_admission_at = 1000;
    restart.deadline = 1100;
    assert!(s.same_inputs(&restart));
    for field in 0..7 {
        let mut changed = restart.clone();
        match field {
            0 => changed.spec_sha256.push('x'),
            1 => changed.contexts_sha256.push('x'),
            2 => changed.initial_corpus_sha256.push('x'),
            3 => changed.vocabulary_sha256.push('x'),
            4 => changed.executable_sha256.push('x'),
            5 => changed.gp_executable_sha256.push('x'),
            _ => changed.predictor_executable_sha256.push('x'),
        }
        assert!(!s.same_inputs(&changed));
    }
}
#[cfg(feature = "neural")]
#[test]
fn expired_campaign_exits_successfully_without_work_or_restart_loop() {
    struct Environment(Option<std::ffi::OsString>);
    impl Drop for Environment {
        fn drop(&mut self) {
            match &self.0 {
                Some(value) => std::env::set_var("JOPE_GP_EXECUTABLE", value),
                None => std::env::remove_var("JOPE_GP_EXECUTABLE"),
            }
        }
    }
    let _environment = Environment(std::env::var_os("JOPE_GP_EXECUTABLE"));
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let initial = root.join("corpus");
    let output = root.join("runs/campaign");
    fs::create_dir_all(&initial).unwrap();
    fs::create_dir_all(&output).unwrap();
    let evaluator = root.join("evaluator");
    fs::write(&evaluator, b"pinned CPU evaluator fixture").unwrap();
    std::env::set_var("JOPE_GP_EXECUTABLE", &evaluator);
    write_json(
        &root.join("runs/gp-service.json"),
        &json!({"executable":evaluator,"executable_sha256":file_hash(&evaluator).unwrap()}),
    )
    .unwrap();
    let contexts = root.join("contexts.json");
    fs::write(&contexts, b"{}\n").unwrap();
    fs::write(initial.join("vocabulary.json"), b"{}\n").unwrap();
    fs::write(
        initial.join("records.bin"),
        vec![0u8; 2 * corpus::RECORD_BYTES],
    )
    .unwrap();
    fs::write(initial.join("labels.jsonl"), b"{}\n{}\n").unwrap();
    let context = |family: &str, partition: &str| corpus::ContextRow {
        dataset: family.into(),
        family: family.into(),
        cut: 0,
        partition: partition.into(),
        path: "unused".into(),
        sha256: "unused".into(),
        prepared: "unused".into(),
        prepared_sha256: "unused".into(),
    };
    let index = corpus::Index {
        schema: "parallel.teaching_corpus.v1".into(),
        qualification: true,
        vocabulary_sha256: file_hash(&initial.join("vocabulary.json")).unwrap(),
        context_index_sha256: file_hash(&contexts).unwrap(),
        records_sha256: file_hash(&initial.join("records.bin")).unwrap(),
        record_bytes: corpus::RECORD_BYTES,
        records: 2,
        legal_sets: vec![vec![0]],
        contexts: vec![
            context("train", "meta_train"),
            context("validation", "meta_validation"),
        ],
        labels_sha256: None,
        groups: vec![
            corpus::Group {
                context: 0,
                family: "train".into(),
                partition: "meta_train".into(),
                records: vec![0],
                max_length: 1,
            },
            corpus::Group {
                context: 1,
                family: "validation".into(),
                partition: "meta_validation".into(),
                records: vec![1],
                max_length: 1,
            },
        ],
    };
    write_json(&initial.join("index.json"), &index).unwrap();
    write_json(&initial.join("provenance.json"),&json!({"sources":[],"index_sha256":file_hash(&initial.join("index.json")).unwrap(),"labels_sha256":file_hash(&initial.join("labels.jsonl")).unwrap()})).unwrap();
    let spec = Spec {
        schema: "parallel.campaign.v1".into(),
        contexts: "contexts.json".into(),
        initial_corpus: "corpus".into(),
        output: "runs/campaign".into(),
        observations: vec![],
        hours: 3,
        steps: 1,
        cuts: 1,
        cut_strategy: CutStrategy::Rotation,
        qualification: true,
        required_training_families: 1,
        required_validation_families: 1,
    };
    let spec_path = root.join("spec.json");
    write_json(&spec_path, &spec).unwrap();
    let started = now() - 20000;
    let schedule = Schedule {
        spec_sha256: file_hash(&spec_path).unwrap(),
        contexts_sha256: file_hash(&contexts).unwrap(),
        initial_corpus_sha256: file_hash(&initial.join("index.json")).unwrap(),
        vocabulary_sha256: index.vocabulary_sha256.clone(),
        executable_sha256: file_hash(&std::env::current_exe().unwrap()).unwrap(),
        gp_executable_sha256: file_hash(&evaluator).unwrap(),
        predictor_executable_sha256: file_hash(&evaluator).unwrap(),
        started_at: started,
        stop_admission_at: started + 3600,
        deadline: started + 10800,
    };
    write_json(&output.join("schedule.json"), &schedule).unwrap();
    // The shipped entry must refuse even this expired, nonnumerical request.
    assert!(run(root, &spec_path)
        .unwrap_err()
        .to_string()
        .contains("original native GPU grant verifier unavailable"));
    assert!(!output.join("complete.json").exists());
    run_admitted(root, &spec_path).unwrap();
    let completed: Value = read_json(&output.join("complete.json")).unwrap();
    assert_eq!(completed["status"], "complete_expired");
    assert_eq!(completed["dl_gp_dl_complete"], false);
    assert!(!output.join("cycle-0000").exists());
    let receipt = fs::read(output.join("complete.json")).unwrap();
    run_admitted(root, &spec_path).unwrap();
    assert_eq!(fs::read(output.join("complete.json")).unwrap(), receipt);
    check_deadline().unwrap();
}

#[test]
fn ranges_ignore_test_only_cuts_and_keep_sparse_grouped_indices() {
    let row = |cut, partition: &str| corpus::ContextRow {
        dataset: "d".into(),
        family: "f".into(),
        cut,
        partition: partition.into(),
        path: "p".into(),
        sha256: "s".into(),
        prepared: "p".into(),
        prepared_sha256: "s".into(),
    };
    let index = corpus::Index {
        schema: "s".into(),
        qualification: true,
        vocabulary_sha256: "s".into(),
        context_index_sha256: "s".into(),
        records_sha256: "s".into(),
        record_bytes: 0,
        records: 0,
        legal_sets: vec![],
        contexts: vec![
            row(0, "meta_train"),
            row(1, "meta_train"),
            row(7, "meta_validation"),
            row(8, "meta_test"),
        ],
        labels_sha256: None,
        groups: vec![],
    };
    assert_eq!(cut_starts(&index, 2), vec![0, 6]);
    let starts = cut_starts(&index, 2);
    assert_eq!(CutStrategy::Rotation.start(&starts, 1), 6);
    assert_eq!(CutStrategy::FixedAnchors.start(&starts, 1), 0);
    assert_eq!(CutStrategy::FixedAnchors.start(&starts, 999), 0);
}

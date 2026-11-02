use super::*;

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
    // Bound the census by 100 observations, not by host-dependent worker startup time.
    // Keep the shipped 600-second pilot ceiling as a runaway guard; completion below
    // still requires the evaluation budget, never the wall-clock stop reason.
    // Completed observations include refusals; they are not successful fitted measurements.
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
            "600",
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
    let summary: serde_json::Value = read_json(&output.join("search.json")).unwrap();
    assert_eq!(summary["completed_evaluations"], 100);
    assert_eq!(summary["stop_reason"], "evaluation_budget");
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
    // The journal and each candidate's attempt, program and completion are retained
    // recovery/admission evidence after publication; artifact eviction never permits refits.
    assert_eq!(
        jope_parallel::gp::read_journal(&output.join("observations.jsonl")).unwrap(),
        observations
    );
    for index in 0..observations.len() {
        for suffix in ["attempt.json", "dsl", "completion.json"] {
            assert!(
                output.join(format!("candidate-{index}.{suffix}")).exists(),
                "candidate-{index}.{suffix} was not retained"
            );
        }
    }
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

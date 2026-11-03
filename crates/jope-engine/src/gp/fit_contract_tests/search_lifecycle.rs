use super::*;

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn a_blocked_fitter_and_its_helpers_are_killed_at_the_original_deadline() {
    let (temp, prepared) = view_fixture();
    let root = temp.path();
    let census = root.join("census");
    let _guard = CensusGuard::new(census.clone());
    let mut authority = Fixture::activated(root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let output = root.join("blocked");
    let started = Instant::now();
    success(
        timed_search_client(&authority, root, &prepared, &output, 2, 4)
            .env(FAULT, "fork_block")
            .env(CENSUS, &census)
            .output()
            .unwrap(),
    );
    assert!(started.elapsed() < Duration::from_secs(30));
    assert_eq!(recorded(&census), 1, "the fault fixture did not run");
    // No-orphan census: the helper dies with the killed evaluator.
    assert_eq!(
        settled_left_behind(&census),
        Vec::<i32>::new(),
        "a helper outlived its killed evaluator"
    );
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    assert_eq!(rows.len(), 1, "work was dispatched after the deadline");
    assert_eq!(rows[0]["failure"], "candidate_or_view_timeout");
    let dispatch = completion(root, &rows[0]);
    assert_eq!(dispatch["deadline_exceeded"], true);
    assert_eq!(dispatch["process_group"]["isolated"], true);
    assert_eq!(dispatch["process_group"]["members_after_drain"], 0);
    let search: Value = read_json(&output.join("search.json")).unwrap();
    assert_eq!(search["stop_reason"], "search_deadline");
    let replayed = selection_replay::verify_search(root, &output).unwrap();
    assert_eq!(
        replayed.replay.stop,
        selection_replay::StopReason::SearchDeadline
    );
    assert!(replayed.replay.winner.is_none() && search["best_program"].is_null());
    // A publication claiming another terminal disposition is refused.
    let mut forged = search.clone();
    forged["stop_reason"] = json!("evaluation_budget");
    write_json(&output.join("search.json"), &forged).unwrap();
    let error = format!(
        "{:#}",
        selection_replay::verify_search(root, &output).unwrap_err()
    );
    assert!(error.contains("stop reason"), "{error}");
    // The blocked evaluator never reached the authority: nothing started.
    authority.stop_service();
    assert_eq!(authority.entries().0, 0);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn helpers_left_by_a_finished_evaluator_are_drained_before_the_next_dispatch() {
    let (temp, prepared) = view_fixture();
    let root = temp.path();
    let census = root.join("census");
    let _guard = CensusGuard::new(census.clone());
    let mut authority = Fixture::activated(root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let output = root.join("leaky");
    success(
        search_client(&authority, root, &prepared, &output, 2)
            .env(FAULT, "fork_exit")
            .env(CENSUS, &census)
            .output()
            .unwrap(),
    );
    assert_eq!(recorded(&census), 2, "the fault fixture did not run");
    assert_eq!(
        settled_left_behind(&census),
        Vec::<i32>::new(),
        "a helper outlived its finished evaluator"
    );
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    assert_eq!(rows.len(), 2);
    for row in &rows {
        assert!(row["failure"].is_null(), "{row}");
        let dispatch = completion(root, row);
        // Each candidate's helper was found and killed before the next one ran.
        assert_eq!(dispatch["process_group"]["members_at_exit"], 1);
        assert_eq!(dispatch["process_group"]["members_after_drain"], 0);
    }
    selection_replay::verify_search(root, &output).unwrap();
    authority.stop_service();
}

/// W5-T05: processes an evaluator starts do not outlive a search controller that is itself
/// killed (SIGKILL, so it can run no cleanup). Reproduction: the evaluator's SIGKILL
/// parent-death signal stopped only the evaluator; its helper was reparented and kept running.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn helpers_do_not_survive_a_killed_search_controller() {
    let (temp, prepared) = view_fixture();
    let root = temp.path();
    let census = root.join("census");
    let _guard = CensusGuard::new(census.clone());
    let mut authority = Fixture::activated(root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let output = root.join("killed-controller");
    let mut controller = timed_search_client(&authority, root, &prepared, &output, 2, 120)
        .env(FAULT, "fork_block")
        .env(CENSUS, &census)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let started = Instant::now();
    while live_left_behind(&census).is_empty() {
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "the fault fixture did not run"
        );
        assert!(
            controller.try_wait().unwrap().is_none(),
            "the controller ended before its evaluator started a helper"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    controller.kill().unwrap();
    controller.wait().unwrap();
    assert_eq!(
        settled_left_behind(&census),
        Vec::<i32>::new(),
        "a helper outlived its killed search controller"
    );
    authority.stop_service();
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn a_second_searcher_cannot_dispatch_into_a_live_search() {
    let (temp, prepared) = view_fixture();
    let root = temp.path();
    let mut authority = Fixture::activated(root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let output = root.join("contended");
    let first = timed_search_client(&authority, root, &prepared, &output, 1, 6)
        .env(FAULT, "block")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Wait until the first searcher has reserved and dispatched candidate 0.
    let intent = output.join("candidate-0.attempt.json");
    let deadline = Instant::now() + Duration::from_secs(60);
    while !intent.exists() {
        assert!(Instant::now() < deadline, "first searcher never dispatched");
        std::thread::sleep(Duration::from_millis(20));
    }
    // Identical request, second process, while candidate 0 is still in flight.
    let second = timed_search_client(&authority, root, &prepared, &output, 1, 6)
        .output()
        .unwrap();
    let first = first.wait_with_output().unwrap();
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(
        !second.status.success(),
        "a concurrent searcher dispatched into a live search"
    );
    assert!(
        String::from_utf8_lossy(&second.stderr).contains("held by another live searcher"),
        "{}",
        String::from_utf8_lossy(&second.stderr)
    );
    // One reservation, one terminal outcome, one publication.
    let journal = read_journal(&output.join("observations.jsonl")).unwrap();
    assert_eq!(journal.len(), 1);
    assert_eq!(journal[0]["failure"], "candidate_or_view_timeout");
    selection_replay::verify_search(root, &output).unwrap();
    // A completed search stays closed to any later searcher.
    let late = timed_search_client(&authority, root, &prepared, &output, 1, 6)
        .output()
        .unwrap();
    assert!(!late.status.success());
    assert!(String::from_utf8_lossy(&late.stderr).contains("already complete"));
    authority.stop_service();
    assert_eq!(authority.entries().0, 0);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn an_evaluator_allocation_failure_is_a_contained_failed_row() {
    let (temp, prepared) = view_fixture();
    let root = temp.path();
    let mut authority = Fixture::activated(root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let output = root.join("oom");
    success(
        search_client(&authority, root, &prepared, &output, 2)
            .env(FAULT, "oom")
            .output()
            .unwrap(),
    );
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    assert_eq!(rows.len(), 2);
    for row in &rows {
        let failure = row["failure"].as_str().unwrap();
        assert!(failure.starts_with("worker_exit:"), "{failure}");
        assert!(row["metrics"].is_null());
        assert_eq!(completion(root, row)["outcome"], "infrastructure_failure");
    }
    // No measured candidate: a typed no-result, never a substitute winner.
    let search: Value = read_json(&output.join("search.json")).unwrap();
    assert!(search["best_program"].is_null());
    let replayed = selection_replay::verify_search(root, &output).unwrap();
    assert!(replayed.replay.winner.is_none());
    assert_eq!(
        replayed.replay.stop,
        selection_replay::StopReason::EvaluationBudget
    );
    authority.stop_service();
    assert_eq!(authority.entries().0, 0);
}

/// A completed three-candidate search whose seed population leads with two
/// fittable programs and one deterministic fit failure.
pub(crate) fn handoff_search(
    authority: &Fixture,
    root: &Path,
    prepared: &Path,
) -> (PathBuf, PathBuf) {
    let seeds = root.join("handoff-seeds.json");
    seed_population(
        prepared,
        &seeds,
        &["(pipe X (standardize) (ridge))", UNFITTABLE, PROGRAM],
        |_| None,
    );
    let output = root.join("handoff-search");
    success(
        search_client(authority, root, prepared, &output, 3)
            .arg("--seeds")
            .arg(&seeds)
            .output()
            .unwrap(),
    );
    (output, seeds)
}

/// The cycle pipeline's handoff, run as a configured client.
pub(crate) fn export_selected(
    authority: &Fixture,
    root: &Path,
    prepared: &Path,
    output: &Path,
    seeds: &Path,
    predictor: &Path,
) -> Command {
    let mut command = authority.command();
    command
        .args(["export-selected", "--root"])
        .arg(root)
        .arg("--output")
        .arg(output)
        .arg("--prepared")
        .arg(prepared)
        .args(["--cut", "0", "--seeds"])
        .arg(seeds)
        .arg("--executable")
        .arg(std::env::current_exe().unwrap())
        .arg("--predictor")
        .arg(predictor);
    command
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn the_predictor_handoff_exports_only_the_sealed_replayed_selection() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (output, seeds) = handoff_search(&authority, &root, &prepared);
    let winner = selection_replay::verify_search(&root, &output)
        .unwrap()
        .replay
        .winner
        .unwrap();
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    let other = rows
        .iter()
        .find(|row| row["failure"].is_null() && row["genome"] != winner.genome.as_str())
        .expect("a second measured program")["genome"]
        .clone();
    // A summary naming another evaluated program is not the selection.
    let search_path = output.join("search.json");
    let original = fs::read(&search_path).unwrap();
    let mut forged: Value = serde_json::from_slice(&original).unwrap();
    forged["best_program"] = other.clone();
    write_json(&search_path, &forged).unwrap();
    let tampered = root.join("tampered.json");
    let refused = export_selected(&authority, &root, &prepared, &output, &seeds, &tampered)
        .output()
        .unwrap();
    assert!(
        !refused.status.success(),
        "the handoff exported the summary's program {other}"
    );
    assert!(!tampered.exists(), "a predictor was published for {other}");
    fs::write(&search_path, &original).unwrap();
    // The honest path exports exactly the replayed winner on its training rows.
    let predictor = root.join("selected.json");
    success(
        export_selected(&authority, &root, &prepared, &output, &seeds, &predictor)
            .output()
            .unwrap(),
    );
    let exported: Predictor = read_json(&predictor).unwrap();
    assert_eq!(
        exported.canonical,
        GenomeIr::parse(&winner.genome).unwrap().canonical()
    );
    let seal: Value = read_json(&predictor.with_extension("seal.json")).unwrap();
    assert_eq!(seal["status"], "selected");
    assert_eq!(seal["selected"]["executable_program"], winner.genome);
    let scope = &seal["final_fit_scope"];
    assert_eq!(scope["kind"], handoff::FINAL_FIT_SCOPE);
    assert_eq!(scope["held_out_labels_in_fit"], false);
    assert_eq!(exported.training_sha256, scope["training_sha256"]);
    assert_ne!(
        scope["training_sha256"],
        scope["held_out_validation_sha256"]
    );
    let handoff: Value = read_json(&predictor.with_extension("handoff.json")).unwrap();
    assert_eq!(handoff["status"], "exported");
    assert_eq!(
        handoff["final_refit"]["transformation"],
        "selected_cut_training_refit_reuses_fold_fit"
    );
    assert_eq!(
        handoff["final_refit"]["model_sha256"],
        seal["selected"]["fold_model"]["sha256"]
    );
    // A resumed cycle re-verifies and reuses the recorded handoff.
    success(
        export_selected(&authority, &root, &prepared, &output, &seeds, &predictor)
            .output()
            .unwrap(),
    );
    authority.stop_service();
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn a_sealed_selection_refuses_swaps_and_keeps_the_selection_when_export_fails() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (output, seeds) = handoff_search(&authority, &root, &prepared);
    let inputs = handoff::Inputs {
        root: &root,
        output: &output,
        seeds: Some(&seeds),
        vocabulary: None,
    };
    let seal_path = root.join("selection.seal.json");
    let seal = handoff::write_seal(&inputs, &seal_path).unwrap();
    assert_eq!(seal["status"], "selected");
    handoff::verify_seal(&inputs, &seal_path).unwrap();
    // Each swap is refused by identity, whatever the predictions would be:
    // another candidate's fitted state or recipe, another scope, another status.
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    let position = seal["selected"]["position"].as_u64().unwrap() as usize;
    let other = rows
        .iter()
        .enumerate()
        .find(|(i, row)| *i != position && row["failure"].is_null())
        .expect("a second measured candidate")
        .1;
    let other_completion = completion(&root, other);
    for (pointer, value) in [
        ("/selected/fold_model", other_completion["model"].clone()),
        (
            "/selected/fold_fit_completion",
            other_completion["fit_completion"].clone(),
        ),
        ("/selected/executable_program", other["genome"].clone()),
        ("/final_fit_scope/training_sha256", json!("0".repeat(64))),
        ("/final_fit_scope/held_out_labels_in_fit", json!(true)),
        ("/status", json!("no_measured_candidate")),
    ] {
        let mut swapped = seal.clone();
        *swapped.pointer_mut(pointer).unwrap() = value;
        write_json(&seal_path, &swapped).unwrap();
        let error = format!(
            "{:#}",
            handoff::verify_seal(&inputs, &seal_path).unwrap_err()
        );
        assert!(
            error.contains("differs from the replayed selection"),
            "{pointer}: {error}"
        );
    }
    write_json(&seal_path, &seal).unwrap();
    // A ledger row missing its lineage event cannot be sealed at all.
    let files = ["observations.jsonl", "observations.json", "search.json"];
    let original: Vec<Vec<u8>> = files
        .iter()
        .map(|name| fs::read(output.join(name)).unwrap())
        .collect();
    let mut missing = rows.clone();
    missing[position].as_object_mut().unwrap().remove("lineage");
    republish_ledger(&output, &missing);
    let error = format!("{:#}", handoff::seal(&inputs).unwrap_err());
    assert!(error.contains("lineage"), "{error}");
    for (name, bytes) in files.iter().zip(&original) {
        fs::write(output.join(name), bytes).unwrap();
    }
    handoff::verify_seal(&inputs, &seal_path).unwrap();
    // A failed export keeps the sealed selection and substitutes nothing.
    let predictor = root.join("failed.json");
    let failed = export_selected(&authority, &root, &prepared, &output, &seeds, &predictor)
        .env(FAULT, "fail_export")
        .output()
        .unwrap();
    assert!(!failed.status.success());
    let recorded: Value = read_json(&predictor.with_extension("handoff.json")).unwrap();
    assert_eq!(recorded["status"], "export_failed");
    assert!(
        recorded["failure"]
            .as_str()
            .unwrap()
            .contains("worker exited"),
        "{recorded}"
    );
    assert!(recorded["predictor"].is_null() && !predictor.exists());
    assert_eq!(
        recorded["selected_fitted_state"],
        seal["selected"]["fold_model"]
    );
    assert_eq!(
        recorded["program_sha256"],
        seal["selected"]["program_sha256"]
    );
    // Only the sealed recipe was attempted.
    let receipt: Value = read_json(&predictor.with_extension("fit.json")).unwrap();
    assert_eq!(receipt["passed"], false);
    assert_eq!(
        receipt["program_sha256"],
        hash(
            seal["selected"]["executable_program"]
                .as_str()
                .unwrap()
                .as_bytes()
        )
    );
    // The recorded failure is terminal: a retry re-verifies and reports it.
    assert!(
        !export_selected(&authority, &root, &prepared, &output, &seeds, &predictor)
            .output()
            .unwrap()
            .status
            .success()
    );
    authority.stop_service();
}

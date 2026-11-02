use super::*;

/// W5-T01: the recorded-input verifier needs no caller paths, refuses a frozen input that no
/// longer carries the request's digest (including another slot's proposal record) or is
/// absent, and reads nothing the request does not pin.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn recorded_search_inputs_verify_without_caller_paths_and_refuse_substitution() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (output, _, _) = frozen_input_search(&root, &prepared, &authority, "recorded");
    let verified = lineage::verify_search_recorded(&root, &output).unwrap();
    assert_eq!(verified.lineage.chains[0].source, "seed_population_slot");
    let frozen = output.join(SEARCH_INPUTS).join(SEARCH_SEEDS);
    let original = fs::read(&frozen).unwrap();
    let refused = |needle: &str| {
        let error = format!(
            "{:#}",
            lineage::verify_search_recorded(&root, &output).unwrap_err()
        );
        assert!(error.contains(needle), "{needle}: {error}");
    };
    let mut population: Value = serde_json::from_slice(&original).unwrap();
    population["candidates"][0]["proposal_record_sha256"] = json!(hash(b"another record"));
    write_json(&frozen, &population).unwrap();
    refused("seed population differs from the frozen request");
    fs::remove_file(&frozen).unwrap();
    refused("is absent");
    fs::write(&frozen, &original).unwrap();
    lineage::verify_search_recorded(&root, &output).unwrap();
    // An unseeded, unadapted search pins neither input and freezes none.
    let generic = root.join("generic");
    success(
        search_client(&authority, &root, &prepared, &generic, 1)
            .output()
            .unwrap(),
    );
    assert!(!generic.join(SEARCH_INPUTS).exists());
    lineage::verify_search_recorded(&root, &generic).unwrap();
    authority.stop_service();
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn reused_fits_are_rescored_on_their_own_rows_and_never_inherit_scores() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    // Same training rows and preprocessing; only the scored membership differs.
    let first = view_fixture_in(&root, "example", &[8, 9]);
    let second = view_fixture_in(&root, "example-b", &[10, 11]);
    let a: Prepared = read_json(&first.join("prepared.json")).unwrap();
    let b: Prepared = read_json(&second.join("prepared.json")).unwrap();
    assert_eq!(a.cuts[0].training_sha256, b.cuts[0].training_sha256);
    assert_eq!(
        a.cuts[0].preprocessing_sha256,
        b.cuts[0].preprocessing_sha256
    );
    assert_ne!(a.cuts[0].validation_sha256, b.cuts[0].validation_sha256);
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&first)]);
    authority.start();
    let run = |prepared: &Path, name: &str| {
        let output = root.join(name);
        success(
            search_client(&authority, &root, prepared, &output, 3)
                .output()
                .unwrap(),
        );
        let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
        (output, rows)
    };
    let (_, cold) = run(&first, "cold");
    let fitted = authority.entries().0;
    let (warm_output, warm) = run(&second, "warm");
    // Identical fitting work was reused: no new physical fit anywhere.
    assert_eq!(authority.entries().0, fitted);
    let search: Value = read_json(&warm_output.join("search.json")).unwrap();
    let measured = warm.iter().filter(|row| row["failure"].is_null()).count();
    assert!(measured > 0);
    assert_eq!(search["execution_accounting"]["physical_fit_starts"], 0);
    assert_eq!(
        search["execution_accounting"]["reused_completions"],
        measured
    );
    let validation = load_view(&second, 0, false).unwrap();
    let x: Vec<Vec<f64>> = validation
        .iter()
        .map(|row| row[..row.len() - 1].to_vec())
        .collect();
    let mut fit_keys = BTreeSet::new();
    for (old, new) in cold.iter().zip(&warm) {
        assert_eq!(old["genome"], new["genome"]);
        if !new["failure"].is_null() {
            continue;
        }
        let (was, now) = (&old["metrics"], &new["metrics"]);
        assert_eq!(now["evaluation_origin"], "reuse");
        assert_eq!(now["physical_fit_starts"], 0);
        // Same fitted subject (fit key and model) ...
        assert_eq!(
            was["score_identity"]["fit_key"],
            now["score_identity"]["fit_key"]
        );
        assert_eq!(
            was["score_identity"]["model_sha256"],
            now["score_identity"]["model_sha256"]
        );
        fit_keys.insert(now["score_identity"]["fit_key"].to_string());
        // ... but a distinct score identity measured on its own rows.
        assert_eq!(
            now["score_identity"]["scoring_rows_sha256"],
            b.cuts[0].validation_sha256
        );
        assert_ne!(was["score_identity_sha256"], now["score_identity_sha256"]);
        // The stored fitted state, applied to the new rows, reproduces the new
        // predictions and RMSE: the score was measured, not carried over.
        let completion = crate::label_admission::artifact(&root, &new["clean_completion"]).unwrap();
        let model: feat_math::Model = serde_json::from_slice(
            &crate::label_admission::artifact_bytes(&root, &completion["model"]).unwrap(),
        )
        .unwrap();
        let predictions = model.predict(&x).unwrap();
        assert_eq!(
            hash(&serde_json::to_vec(&predictions).unwrap()),
            now["semantic_sha256"],
            "the reused fit was not rescored on the new rows"
        );
        let squared = validation
            .iter()
            .zip(&predictions)
            .map(|(row, p)| (row[row.len() - 1] - p).powi(2))
            .sum::<f64>();
        let rmse = (squared / predictions.len() as f64).sqrt();
        let recorded = now["rmse"].as_f64().unwrap();
        assert!(
            (rmse - recorded).abs() <= 1e-12 * rmse.abs().max(1.0),
            "recorded RMSE {recorded} is not the new rows' RMSE {rmse}"
        );
        assert_ne!(was["rmse"], now["rmse"], "a score carried over unchanged");
    }
    // Different recipes never share a fit key.
    assert_eq!(fit_keys.len(), measured);
    // Both searches replay; a score carried over from the other membership,
    // even with its own consistent identity, is refused.
    let cold_output = root.join("cold");
    selection_replay::verify_search(&root, &cold_output).unwrap();
    selection_replay::verify_search(&root, &warm_output).unwrap();
    let position = warm
        .iter()
        .position(|row| row["failure"].is_null())
        .unwrap();
    let mut stale = warm.clone();
    stale[position]["metrics"] = cold[position]["metrics"].clone();
    republish_ledger(&warm_output, &stale);
    let error = format!(
        "{:#}",
        selection_replay::verify_search(&root, &warm_output).unwrap_err()
    );
    assert!(
        error.contains("another fold") || error.contains("outside the frozen fold"),
        "{error}"
    );
    authority.stop_service();
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn feasibility_estimates_are_recorded_beside_observed_outcomes_under_one_rule() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    // A covered recipe, one the fitter provably refuses (a misplaced target
    // stage), and one with an uncovered head.
    let refused = "(pipe X (standardize) (target) (ridge))";
    let uncovered = "(pipe X (standardize) (gpr m=50 steps=5))";
    let seeds = root.join("seeds.json");
    seed_population(&prepared, &seeds, &[PROGRAM, refused, uncovered], |_| None);
    let output = root.join("feasibility");
    success(
        search_client(&authority, &root, &prepared, &output, 3)
            .arg("--seeds")
            .arg(&seeds)
            .output()
            .unwrap(),
    );
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    let verdicts: Vec<&str> = rows
        .iter()
        .map(|row| row["feasibility"]["verdict"].as_str().unwrap())
        .collect();
    assert_eq!(
        verdicts,
        ["proven_feasible", "proven_outside_limits", "unknown"]
    );
    // Estimate beside observation: the proven-feasible recipe fitted, and the
    // proven refusal was observed as a deterministic fit failure. Recording the
    // estimate did not change dispatch: all three reached the fitter.
    assert!(rows[0]["failure"].is_null());
    assert!(!rows[1]["failure"].is_null());
    assert_eq!(rows[1]["fit_outcome"], "deterministic_failure");
    assert_eq!(authority.entries().0, 3);
    // One rule for every arm: each recorded estimate is exactly the classifier
    // applied to the program and the cut's declared shape (8 rows, 1 feature).
    let shape = feat_math::feasibility::Shape { rows: 8, cols: 1 };
    for (i, row) in rows.iter().enumerate() {
        let attempt: Value =
            read_json(&output.join(format!("candidate-{i}.attempt.json"))).unwrap();
        let expected = serde_json::to_value(feat_math::feasibility::classify(
            &GenomeIr::parse(row["genome"].as_str().unwrap()).unwrap(),
            shape,
            &feat_math::feasibility::Limits::default(),
        ))
        .unwrap();
        assert_eq!(attempt["feasibility"], expected, "candidate {i}");
    }
    // Selection replay and lineage remain valid with the diagnostic present.
    lineage::verify_search(&root, &output, Some(seeds.as_path()), None).unwrap();
    authority.stop_service();
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn a_resume_under_other_selection_rules_is_refused_not_restarted() {
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
    let output = experiment.join("search");
    let command = || {
        let mut command = search_client(&authority, root, &prepared, &output, 2);
        command.env("JOPE_EXPERIMENT_DIR", &experiment);
        command
    };
    let crashed = command()
        .env("JOPE_EXPERIMENT_CRASH_AFTER", "observation_progress")
        .output()
        .unwrap();
    assert!(!crashed.status.success());
    // The objective, evidence, tie and stop rules were frozen with the request.
    let request_path = output.join("search-request.json");
    let original = fs::read(&request_path).unwrap();
    let request: Value = serde_json::from_slice(&original).unwrap();
    assert_eq!(request["selection_contract"], selection_replay::CONTRACT);
    assert_eq!(
        request["proposal_cap_per_slot"],
        selection_replay::PROPOSAL_CAP_PER_SLOT
    );
    let journal = fs::read(output.join("observations.jsonl")).unwrap();
    for (key, value) in [
        (
            "selection_contract",
            json!("parallel.gp_selection_contract.v0"),
        ),
        ("proposal_cap_per_slot", json!(1)),
    ] {
        let mut changed = request.clone();
        changed[key] = value;
        write_json(&request_path, &changed).unwrap();
        let resumed = command().output().unwrap();
        assert!(
            !resumed.status.success(),
            "{key}: resumed under other rules"
        );
        assert!(
            String::from_utf8_lossy(&resumed.stderr).contains("request changed"),
            "{key}: {}",
            String::from_utf8_lossy(&resumed.stderr)
        );
        // Refused before any new work: the ledger and the fit registry are unchanged.
        assert_eq!(
            fs::read(output.join("observations.jsonl")).unwrap(),
            journal
        );
        assert!(!output.join("search.json").exists());
        assert_eq!(authority.entries().0, 1);
    }
    // The original rules resume and complete; the replay accepts the result.
    fs::write(&request_path, &original).unwrap();
    success(command().output().unwrap());
    selection_replay::verify_search(root, &output).unwrap();
    authority.stop_service();
}

/// Live (non-zombie) helper processes among those a faulting evaluator
/// recorded. Only the fixture's own `sleep 300` helpers are counted, so a
/// reused pid is never mistaken for one.
pub(super) fn live_left_behind(census: &Path) -> Vec<i32> {
    let Ok(entries) = fs::read_dir(census) else {
        return Vec::new();
    };
    let mut live = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(pid) = name
            .strip_suffix(".pid")
            .and_then(|pid| pid.parse::<i32>().ok())
        else {
            continue;
        };
        let Ok(stat) = fs::read_to_string(format!("/proc/{pid}/stat")) else {
            continue;
        };
        let state = stat
            .rsplit_once(')')
            .and_then(|(_, rest)| rest.split_whitespace().next());
        let helper = fs::read(format!("/proc/{pid}/cmdline"))
            .is_ok_and(|cmdline| cmdline.starts_with(b"sleep\x00300\0"));
        if helper && state != Some("Z") {
            live.push(pid);
        }
    }
    live.sort_unstable();
    live
}

/// A killed orphan is reaped by init asynchronously: poll briefly.
pub(super) fn settled_left_behind(census: &Path) -> Vec<i32> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let live = live_left_behind(census);
        if live.is_empty() || Instant::now() >= deadline {
            return live;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// A failing test must not leak the helpers it provoked.
pub(super) struct CensusGuard(PathBuf);

impl Drop for CensusGuard {
    fn drop(&mut self) {
        for pid in live_left_behind(&self.0) {
            // SAFETY: kill takes only integers and has no memory-safety preconditions; `pid` is
            // a live `sleep 300` helper this test provoked, just found by the census.
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
        }
    }
}

pub(super) fn recorded(census: &Path) -> usize {
    fs::read_dir(census).map_or(0, |entries| entries.count())
}

pub(super) fn completion(root: &Path, row: &Value) -> Value {
    crate::label_admission::artifact(root, &row["clean_completion"]).unwrap()
}

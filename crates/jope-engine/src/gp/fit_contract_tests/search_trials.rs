use super::*;

/// A deterministic numerical failure that parses but has no fitted operator.
pub(super) const UNFITTABLE: &str = "(pipe X (unknown))";

/// An 80-slot seed population for the view fixture's cut 0. The first slots are
/// `leading`; every slot carries the given critic rank weight (never a score).
pub(super) fn seed_population(
    prepared: &Path,
    path: &Path,
    leading: &[&str],
    weight: impl Fn(usize) -> Option<f64>,
) {
    let mut programs: Vec<String> = leading.iter().map(|p| p.to_string()).collect();
    let mut seen: BTreeSet<String> = programs.iter().cloned().collect();
    let mut seed = 0;
    while programs.len() < 80 {
        let program = feat_math::evolve::random_canonical_pipeline(seed, 6);
        seed += 1;
        if seen.insert(program.clone()) {
            programs.push(program);
        }
    }
    let p: Prepared = read_json(&prepared.join("prepared.json")).unwrap();
    write_json(
        path,
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
                    origin: "neural_policy".into(),
                    policy_logprob: weight(i),
                    proposal_record_sha256: None,
                })
                .collect(),
        },
    )
    .unwrap();
}

/// A configured CLI search of cut 0 under `root`.
pub(super) fn search_client(
    authority: &Fixture,
    root: &Path,
    prepared: &Path,
    output: &Path,
    evaluations: usize,
) -> Command {
    timed_search_client(authority, root, prepared, output, evaluations, 180)
}

/// A configured CLI search of cut 0 with an explicit search deadline.
pub(super) fn timed_search_client(
    authority: &Fixture,
    root: &Path,
    prepared: &Path,
    output: &Path,
    evaluations: usize,
    seconds: u64,
) -> Command {
    let mut command = authority.command();
    command
        .args(["search", "--root"])
        .arg(root)
        .arg("--prepared")
        .arg(prepared)
        .arg("--output")
        .arg(output)
        .args(["--cut", "0", "--qualification"])
        .args(["--seconds", &seconds.to_string()])
        .args(["--evaluations", &evaluations.to_string()]);
    command
}

/// Rewrite a completed search's journal, snapshot and snapshot digest together,
/// so that only the deliberately tampered rows differ from the publication.
pub(super) fn republish_ledger(output: &Path, rows: &[Value]) {
    let mut journal = Vec::new();
    for row in rows {
        journal.extend(serde_json::to_vec(row).unwrap());
        journal.push(b'\n');
    }
    fs::write(output.join("observations.jsonl"), journal).unwrap();
    write_json(&output.join("observations.json"), &rows.to_vec()).unwrap();
    let mut search: Value = read_json(&output.join("search.json")).unwrap();
    search["observations_sha256"] = json!(file_hash(&output.join("observations.json")).unwrap());
    write_json(&output.join("search.json"), &search).unwrap();
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn independent_replay_reproduces_real_search_winners_and_refuses_tampered_ledgers() {
    use selection_replay::{verify_search, Eligibility, StopReason};
    let (temp, prepared) = view_fixture();
    let root = temp.path();
    let mut authority = Fixture::activated(root, vec![cut_training_hash(&prepared)]);
    authority.start();
    // The critic ranks the unfittable program first and the linear head last;
    // the second search reverses every rank weight. Neither may matter.
    let leading = ["(pipe X (standardize) (ridge))", UNFITTABLE, PROGRAM];
    let forward = root.join("seeds-forward.json");
    let reversed = root.join("seeds-reversed.json");
    seed_population(&prepared, &forward, &leading, |i| Some(-(i as f64)));
    seed_population(&prepared, &reversed, &leading, |i| Some(i as f64 - 80.0));
    let mut outputs = Vec::new();
    for (name, seeds) in [("search-forward", &forward), ("search-reversed", &reversed)] {
        let output = root.join(name);
        success(
            search_client(&authority, root, &prepared, &output, 3)
                .arg("--seeds")
                .arg(seeds)
                .output()
                .unwrap(),
        );
        outputs.push(output);
    }
    let output = &outputs[0];
    let before = crate::smartcluster_worker::hash_directory(output).unwrap();
    let replayed = verify_search(root, output).unwrap();
    // Replay is read-only: no authority-bearing or published byte changed.
    assert_eq!(
        crate::smartcluster_worker::hash_directory(output).unwrap(),
        before
    );
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    let search: Value = read_json(&output.join("search.json")).unwrap();
    let replay = &replayed.replay;
    assert_eq!(replay.stop, StopReason::EvaluationBudget);
    assert_eq!(rows[1]["genome"], UNFITTABLE);
    assert!(
        matches!(&replay.rows[1].eligibility, Eligibility::Failed { failure } if !failure.is_empty())
    );
    assert_eq!((replay.measured, replay.failed), (2, 1));
    // A third, trivially independent computation over the published rows.
    let scale = rows[0]["target_scale"].as_f64().unwrap();
    let measured: Vec<(usize, f64)> = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row["failure"].is_null())
        .map(|(i, row)| (i, row["metrics"]["rmse"].as_f64().unwrap() * scale))
        .collect();
    let best = measured.iter().fold(
        measured[0],
        |best, next| if next.1 < best.1 { *next } else { best },
    );
    let winner = replay.winner.clone().unwrap();
    assert_eq!(winner.position, best.0 as i64);
    assert_eq!(winner.objective.to_bits(), best.1.to_bits());
    assert_eq!(search["best_program"], rows[best.0]["genome"]);
    // Identical predictions are one ranked entry represented by the earliest row.
    if rows[0]["metrics"]["semantic_sha256"] == rows[2]["metrics"]["semantic_sha256"] {
        assert_eq!(replay.ranking.len(), 1);
        assert_eq!(winner.position, 0);
        assert!(matches!(
            replay.rows[2].eligibility,
            Eligibility::SemanticDuplicate { of_position: 0, .. }
        ));
    } else {
        assert_eq!(replay.ranking.len(), 2);
    }
    // Reversing every predicted rank weight changes no measurement or winner.
    let other = verify_search(root, &outputs[1]).unwrap().replay;
    assert_eq!(other.winner, replay.winner);
    assert_eq!(other.ranking, replay.ranking);
    assert_eq!(other.rows, replay.rows);

    // Tamper suite: each change refuses; the original bytes are then restored.
    let files = ["observations.jsonl", "observations.json", "search.json"];
    let original: Vec<Vec<u8>> = files
        .iter()
        .map(|name| fs::read(output.join(name)).unwrap())
        .collect();
    let restore = || {
        for (name, bytes) in files.iter().zip(&original) {
            fs::write(output.join(name), bytes).unwrap();
        }
    };
    let refused = |expected: &str| {
        let error = format!("{:#}", verify_search(root, output).unwrap_err());
        assert!(error.contains(expected), "expected {expected:?}: {error}");
        restore();
        assert!(verify_search(root, output).is_ok());
    };
    let mut swapped = rows.clone();
    swapped.swap(0, 2);
    republish_ledger(output, &swapped);
    refused("ledger order");
    let mut forged = rows.clone();
    forged[best.0]["metrics"]["rmse"] = json!(0.0);
    republish_ledger(output, &forged);
    refused("original evidence refused");
    let mut revived = rows.clone();
    revived[1]["failure"] = Value::Null;
    revived[1]["metrics"] = rows[best.0]["metrics"].clone();
    republish_ledger(output, &revived);
    refused("original evidence refused");
    let mut published = search.clone();
    published["best_program"] = rows[1]["genome"].clone();
    write_json(&output.join("search.json"), &published).unwrap();
    refused("published winner differs");
    let mut published = search.clone();
    published["best_validation_rmse"] = json!(best.1 / 2.0);
    write_json(&output.join("search.json"), &published).unwrap();
    refused("published winner differs");
    let mut torn = original[0].clone();
    torn.extend_from_slice(b"{\"schema\"");
    fs::write(output.join("observations.jsonl"), torn).unwrap();
    refused("torn tail");
    // A late or stray result outside the ledger is reported, never ranked.
    fs::write(
        output.join("candidate-3.json"),
        serde_json::to_vec(&json!({"rmse":0.0,"semantic_sha256":"9".repeat(64)})).unwrap(),
    )
    .unwrap();
    fs::write(output.join("candidate-3.attempt.json"), b"{}").unwrap();
    let late = verify_search(root, output).unwrap();
    assert_eq!(late.replay.winner, replay.winner);
    assert_eq!(
        late.unledgered_artifacts,
        [
            PathBuf::from("candidate-3.attempt.json"),
            PathBuf::from("candidate-3.json")
        ]
    );
    authority.stop_service();
    // Three original fit entries; the reversed search started no new fit.
    assert_eq!(authority.entries().0, 3);
    let reused: Vec<Value> = read_json(&outputs[1].join("observations.json")).unwrap();
    for row in reused.iter().filter(|row| row["failure"].is_null()) {
        assert_eq!(row["evaluation_origin"], "reuse");
        assert_eq!(row["metrics"]["physical_fit_starts"], 0);
    }
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn resumed_search_refuses_journaled_measurements_its_original_completion_does_not_support(
) {
    for forged in [json!(0.0), json!(-1.0)] {
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
        let journal = output.join("observations.jsonl");
        let mut rows = read_journal(&journal).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0]["failure"].is_null());
        assert_ne!(rows[0]["metrics"]["rmse"], forged);
        rows[0]["metrics"]["rmse"] = forged.clone();
        let mut bytes = serde_json::to_vec(&rows[0]).unwrap();
        bytes.push(b'\n');
        fs::write(&journal, bytes).unwrap();
        let resumed = command().output().unwrap();
        let published: Option<Value> = read_json(&output.join("search.json")).ok();
        assert!(
            !resumed.status.success(),
            "resume adopted the journaled RMSE {forged} as measured evidence: {:?}",
            published.map(|s| (s["best_program"].clone(), s["best_validation_rmse"].clone()))
        );
        assert!(!output.join("search.json").exists());
        // The refusal precedes any new dispatch: only the original fit exists.
        authority.stop_service();
        assert_eq!(authority.entries().0, 1);
    }
}

/// The featmath grammar vocabulary shipped with this crate.
pub(super) fn vocabulary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json")
}

/// Independent enumeration of the generic bootstrap: `(original generator
/// draw, its single adaptation)` per slot, starting with the fixed reference.
pub(super) fn generic_bootstrap(
    vocab: &crate::grammar::Vocab,
    slots: usize,
) -> Vec<(String, String, u64)> {
    let mut seen = BTreeSet::from([PROGRAM.to_string()]);
    let mut expected = vec![(PROGRAM.to_string(), PROGRAM.to_string(), 0)];
    let mut seed = SEED;
    while expected.len() < slots {
        let raw = feat_math::evolve::random_canonical_pipeline(seed, 6);
        if let Ok(adapted) = vocab.project_proposal(&raw) {
            if seen.insert(adapted.clone()) {
                expected.push((raw, adapted, seed));
            }
        }
        seed += 1;
    }
    expected
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn generic_bootstrap_candidates_retain_their_original_proposal_and_one_adaptation() {
    let vocabulary = vocabulary_path();
    let vocab: crate::grammar::Vocab = read_json(&vocabulary).unwrap();
    let expected = generic_bootstrap(&vocab, 12);
    let adapted = expected
        .iter()
        .position(|(raw, adapted, _)| raw != adapted)
        .expect("the fixture must exercise an adapted bootstrap slot");
    let (temp, prepared) = view_fixture();
    let root = temp.path();
    let mut authority = Fixture::activated(root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let output = root.join("generic");
    success(
        search_client(&authority, root, &prepared, &output, adapted + 1)
            .arg("--vocabulary")
            .arg(&vocabulary)
            .output()
            .unwrap(),
    );
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    assert_eq!(rows.len(), adapted + 1);
    for (i, row) in rows.iter().enumerate() {
        let (raw, executable, _) = &expected[i];
        assert_eq!(row["genome"], *executable, "slot {i}: executable recipe");
        let attempt: Value =
            read_json(&output.join(format!("candidate-{i}.attempt.json"))).unwrap();
        assert_eq!(
            attempt["raw_genome"], *raw,
            "slot {i}: the original generator proposal was replaced by its adaptation"
        );
        assert_eq!(
            row["proposal_before_projection"],
            if raw != executable {
                json!(raw)
            } else {
                Value::Null
            },
            "slot {i}: ledger original proposal"
        );
    }
    // The envelope re-derives each original from its generator draw, exactly one
    // adaptation, and the executable; the selected subject links to its fold fit.
    let verified = lineage::verify_search(root, &output, None, Some(vocabulary.as_path())).unwrap();
    assert!(verified.lineage.adapted >= 1);
    for (i, chain) in verified.lineage.chains.iter().enumerate() {
        let (raw, executable, seed) = &expected[i];
        assert_eq!(chain.original_sha256, hash(raw.as_bytes()));
        assert_eq!(chain.executable_sha256, hash(executable.as_bytes()));
        assert_eq!(chain.adapted, raw != executable);
        if i == 0 {
            assert_eq!(chain.source, "fixed_reference");
        } else {
            assert_eq!(chain.source, "generic_bootstrap");
            assert_eq!(rows[i]["lineage"]["source"]["generator_seed"], *seed);
        }
    }
    // A wrong or missing vocabulary is not the frozen request's adaptation.
    assert!(lineage::verify_search(root, &output, None, None).is_err());
    let selected = verified.selected.as_ref().expect("a measured winner");
    let winner = verified.selection.replay.winner.clone().unwrap().genome;
    // Final refit of the selected recipe on the same cut training rows: an
    // explicit transformation that reuses the selected fold fit's state.
    let fit_view = |genome: &str, predictor: &Path| {
        success(
            authority
                .command()
                .args(["fit-view", "--root"])
                .arg(root)
                .arg("--prepared")
                .arg(&prepared)
                .args(["--cut", "0", "--genome", genome, "--output"])
                .arg(predictor)
                .output()
                .unwrap(),
        );
    };
    let predictor = root.join("final-predictor.json");
    fit_view(&winner, &predictor);
    let refit = lineage::verify_final_refit(root, &verified, &predictor).unwrap();
    assert_eq!(refit.model_sha256, selected.fold_model["sha256"]);
    // Another recipe, or the selected recipe's name over another fitted state,
    // is not the selected subject's refit, whatever its predictions.
    let other = root.join("other-predictor.json");
    fit_view("(pipe X (standardize) (ridge))", &other);
    let error = format!(
        "{:#}",
        lineage::verify_final_refit(root, &verified, &other).unwrap_err()
    );
    assert!(
        error.contains("not the selected executable recipe"),
        "{error}"
    );
    let mut swapped: Predictor = read_json(&other).unwrap();
    swapped.canonical = GenomeIr::parse(&winner).unwrap().canonical();
    let swapped_path = root.join("swapped-predictor.json");
    write_json(&swapped_path, &swapped).unwrap();
    let error = format!(
        "{:#}",
        lineage::verify_final_refit(root, &verified, &swapped_path).unwrap_err()
    );
    assert!(error.contains("not the selected fold fit"), "{error}");
    authority.stop_service();
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn variation_children_record_parents_and_cannot_alter_parent_fitted_state() {
    let vocabulary = vocabulary_path();
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    // Slots 1..3 adapt onto slot 0's program, so their first proposals are
    // refused as duplicates and each slot is filled by an explicit variation.
    let seeds = root.join("seeds.json");
    seed_population(
        &prepared,
        &seeds,
        &[
            PROGRAM,
            "(pipe X  (ridge))",
            "(pipe X   (ridge))",
            "(pipe X    (ridge))",
        ],
        |_| None,
    );
    let output = root.join("variations");
    success(
        search_client(&authority, &root, &prepared, &output, 4)
            .arg("--seeds")
            .arg(&seeds)
            .arg("--vocabulary")
            .arg(&vocabulary)
            .output()
            .unwrap(),
    );
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    let verified = lineage::verify_search(
        &root,
        &output,
        Some(seeds.as_path()),
        Some(vocabulary.as_path()),
    )
    .unwrap();
    assert_eq!(verified.lineage.chains[0].source, "seed_population_slot");
    assert!(verified.lineage.variations >= 3, "{:?}", verified.lineage);
    // Frozen inputs are authenticated: another seed population is refused.
    let other = root.join("other-seeds.json");
    seed_population(&prepared, &other, &[PROGRAM], |_| Some(-1.0));
    assert!(lineage::verify_search(
        &root,
        &output,
        Some(other.as_path()),
        Some(vocabulary.as_path())
    )
    .is_err());
    // Every parent's original fitted state still reproduces its recorded
    // predictions after all of its children were fitted: children are new
    // subjects, never in-place edits of their parents.
    let validation = load_view(&prepared, 0, false).unwrap();
    let x: Vec<Vec<f64>> = validation
        .iter()
        .map(|row| row[..row.len() - 1].to_vec())
        .collect();
    let parents: BTreeSet<i64> = verified
        .lineage
        .chains
        .iter()
        .flat_map(|chain| chain.parents.iter().copied())
        .collect();
    assert!(
        !parents.is_empty(),
        "no variation recorded a measured parent"
    );
    for parent in parents {
        let row = &rows[parent as usize];
        let completion = crate::label_admission::artifact(&root, &row["clean_completion"]).unwrap();
        let model: feat_math::Model = serde_json::from_slice(
            &crate::label_admission::artifact_bytes(&root, &completion["model"]).unwrap(),
        )
        .unwrap();
        let predictions = model.predict(&x).unwrap();
        assert_eq!(
            hash(&serde_json::to_vec(&predictions).unwrap()),
            row["metrics"]["semantic_sha256"],
            "parent {parent} fitted state changed"
        );
    }
    // Tampering with the pre-dispatch attempt breaks its completion binding.
    let attempt_path = output.join("candidate-1.attempt.json");
    let original = fs::read(&attempt_path).unwrap();
    let mut attempt: Value = serde_json::from_slice(&original).unwrap();
    attempt["lineage"]["source"]["operator"] = json!("fresh_random");
    write_json(&attempt_path, &attempt).unwrap();
    assert!(lineage::verify_search(
        &root,
        &output,
        Some(seeds.as_path()),
        Some(vocabulary.as_path())
    )
    .is_err());
    fs::write(&attempt_path, original).unwrap();
    assert!(lineage::verify_search(
        &root,
        &output,
        Some(seeds.as_path()),
        Some(vocabulary.as_path())
    )
    .is_ok());
    authority.stop_service();
}

/// A seeded, vocabulary-adapted search whose seed slots name proposal records (W4-T07), with
/// the caller's input files removed afterwards. Returns the output and the input bytes.
pub(super) fn frozen_input_search(
    root: &Path,
    prepared: &Path,
    authority: &Fixture,
    name: &str,
) -> (PathBuf, Vec<u8>, Vec<u8>) {
    let seeds = root.join(format!("{name}-seeds.json"));
    seed_population(
        prepared,
        &seeds,
        &[PROGRAM, "(pipe X (standardize) (ridge))"],
        |_| None,
    );
    let mut population: Value = read_json(&seeds).unwrap();
    for (slot, candidate) in population["candidates"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        candidate["proposal_record_sha256"] =
            json!(hash(format!("proposal record {slot}").as_bytes()));
    }
    write_json(&seeds, &population).unwrap();
    let vocabulary = root.join(format!("{name}-vocabulary.json"));
    fs::copy(vocabulary_path(), &vocabulary).unwrap();
    let output = root.join(name);
    success(
        search_client(authority, root, prepared, &output, 2)
            .arg("--seeds")
            .arg(&seeds)
            .arg("--vocabulary")
            .arg(&vocabulary)
            .output()
            .unwrap(),
    );
    let bytes = (fs::read(&seeds).unwrap(), fs::read(&vocabulary).unwrap());
    // The caller's copies are gone: only the search output remains.
    fs::remove_file(&seeds).unwrap();
    fs::remove_file(&vocabulary).unwrap();
    (output, bytes.0, bytes.1)
}

/// W5-T01: a seeded search freezes the exact seed population and vocabulary bytes its request
/// pins, so its lineage (including each seed slot's proposal-record join) verifies from the
/// output alone. Reproduction: the output held only the digests, and the lineage verifier
/// needed the caller's original paths.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(super) fn a_seeded_search_freezes_the_inputs_its_lineage_is_verified_against() {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (output, seeds, vocabulary) = frozen_input_search(&root, &prepared, &authority, "frozen");
    let inputs = output.join("search-inputs");
    let (frozen_seeds, frozen_vocabulary) =
        (inputs.join("seeds.json"), inputs.join("vocabulary.json"));
    assert_eq!(
        (
            fs::read(&frozen_seeds).ok(),
            fs::read(&frozen_vocabulary).ok()
        ),
        (Some(seeds), Some(vocabulary)),
        "the search did not freeze the inputs its request pins"
    );
    let verified = lineage::verify_search(
        &root,
        &output,
        Some(frozen_seeds.as_path()),
        Some(frozen_vocabulary.as_path()),
    )
    .unwrap();
    assert_eq!(verified.lineage.chains[0].source, "seed_population_slot");
    // The W4-T07 join reached the candidate ledger.
    let rows: Vec<Value> = read_json(&output.join("observations.json")).unwrap();
    assert_eq!(
        rows[0]["lineage"]["source"]["proposal_record_sha256"],
        json!(hash(b"proposal record 0"))
    );
    authority.stop_service();
}

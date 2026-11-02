use super::*;

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

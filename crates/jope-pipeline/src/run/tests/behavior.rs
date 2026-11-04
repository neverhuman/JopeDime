use super::*;

#[test]
fn narration_covers_selection_and_emission_before_run_terminal() {
    let temp = tempfile::tempdir().unwrap();
    let recording = jope_narrator::RecordingTap::default();
    let mut options = small(&temp.path().join("run"), 1);
    options.cuts = 1;
    options.clock = Clock::Frozen;
    let receipt = run_with_narration(&options, Some(recording.tap())).unwrap();
    let facts = recording.facts();
    let selected = facts.iter().position(|fact| matches!(fact,
        jope_narrator::LineageFact::Selected { program_sha256, rule }
            if program_sha256 == &receipt.search.selected_program_sha256 && rule == &receipt.search.selection
    )).expect("missing actual selection");
    let emitted = facts.iter().position(|fact| matches!(fact,
        jope_narrator::LineageFact::Emitted { program_sha256, languages }
            if program_sha256 == &receipt.search.selected_program_sha256 && languages == &["rust".to_owned()]
    )).expect("missing completed emission");
    assert!(selected < emitted);
    assert!(
        matches!(facts.last(), Some(jope_narrator::LineageFact::RunStopped { status }) if status == "completed")
    );
    assert_eq!(
        facts
            .iter()
            .filter(|fact| matches!(fact, jope_narrator::LineageFact::RunStopped { .. }))
            .count(),
        1
    );
}

/// What a run decided, without the wall clock: genome, generation, origin, status, scores.
#[test]
fn a_seed_enters_generation_zero_after_the_bootstrap_as_an_incumbent() {
    let temp = tempfile::tempdir().unwrap();
    let seed = seeds::canonical_pipeline("(pipe X (standardize) (ridge_gcv))").unwrap();
    let file = seeds_file(temp.path(), &[&seed]);
    let output = temp.path().join("run");
    let receipt = run(&RunOptions {
        seed_genomes: Some(file.clone()),
        ..small(&output, 14)
    })
    .unwrap();
    assert_eq!(receipt.schema, receipt::RECEIPT_SCHEMA_ID);
    let seeding = receipt.search.seeding.as_ref().unwrap();
    assert_eq!(seeding.offered, 1);
    assert_eq!(seeding.seeds.len(), 1);
    assert_eq!(seeding.seeds[0].genome, seed);
    assert_eq!(seeding.seeds[0].program_sha256, sha256_hex(seed.as_bytes()));
    assert_eq!(seeding.seeds[0].source, seeds::FROM_SEEDS_FILE);
    assert_eq!(
        seeding.seeds_file_sha256.as_deref(),
        Some(crate::sha256_file(&file).unwrap().as_str())
    );
    assert!(seeding.parent_receipt_sha256.is_none());

    let verified = crate::evidence::verify_run(&output.join("receipt.json")).unwrap();
    let observations = &verified.cuts[0].1;
    assert_eq!(observations[0].origin, search::BOOTSTRAP);
    assert_eq!(observations[1].genome, seed);
    assert_eq!(observations[1].origin, search::INCUMBENT);
    assert_eq!(observations[1].generation, 0);
    // The seed spent budget like any candidate: the evaluation bound still holds.
    assert!(receipt.search.cuts[0].completed_evaluations <= 14);
}

/// A seeded run whose seconds bound has already passed still re-measures every seed in
/// generation 0 on every cut, so it writes a receipt its own evidence verification accepts
/// (the v3 seeding record and `verify_run` agree), rather than one only a comparison
/// re-measurement covers, or none.
#[test]
fn a_seeded_run_past_its_seconds_bound_writes_a_receipt_that_verifies() {
    let temp = tempfile::tempdir().unwrap();
    let seeds: Vec<String> = [
        "(pipe X (standardize) (ridge_gcv))",
        "(pipe X (rank) (ridge))",
    ]
    .iter()
    .map(|g| seeds::canonical_pipeline(g).unwrap())
    .collect();
    let file = seeds_file(temp.path(), &[&seeds[0], &seeds[1]]);
    let output = temp.path().join("run");
    let receipt = run(&RunOptions {
        seed_genomes: Some(file),
        cuts: 2,
        clock: Clock::Expired,
        ..small(&output, 40)
    })
    .unwrap();
    assert!(receipt.search.cuts.iter().all(|c| c
        .accounting
        .as_ref()
        .is_some_and(|a| a.deadline_overrun && a.exploration.stopped_by == "seconds")));
    let verified = crate::evidence::verify_champion_run(&output.join("receipt.json")).unwrap();
    for (_, observations) in &verified.cuts {
        let incumbents: Vec<&str> = observations
            .iter()
            .filter(|o| o.origin == search::INCUMBENT && o.generation == 0)
            .map(|o| o.genome.as_str())
            .collect();
        assert_eq!(incumbents, vec![seeds[0].as_str(), seeds[1].as_str()]);
    }
}

#[test]
fn the_same_seed_and_rng_replay_identically() {
    let temp = tempfile::tempdir().unwrap();
    let seed = seeds::canonical_pipeline("(pipe X (robust_scale) (ridge))").unwrap();
    let file = seeds_file(temp.path(), &[&seed]);
    let mut replays = Vec::new();
    for name in ["a", "b"] {
        let output = temp.path().join(name);
        run(&RunOptions {
            seed_genomes: Some(file.clone()),
            ..small(&output, 16)
        })
        .unwrap();
        replays.push(crate::evidence::verify_run(&output.join("receipt.json")).unwrap());
    }
    assert_eq!(decided(&replays[0]), decided(&replays[1]));
    assert_eq!(
        replays[0].receipt.search.selected_genome,
        replays[1].receipt.search.selected_genome
    );
    assert_eq!(
        replays[0].receipt.search.seeding,
        replays[1].receipt.search.seeding
    );
}

/// Seeding from a parent re-measures the parent's champion on this run's cuts, and the
/// selection over those measurements can only match or beat it.
#[test]
fn a_champion_is_never_worse_than_its_seed_on_the_same_cuts() {
    let temp = tempfile::tempdir().unwrap();
    let parent_dir = temp.path().join("parent");
    let parent = run(&RunOptions {
        cuts: 2,
        ..small(&parent_dir, 24)
    })
    .unwrap();
    let parent_best = parent.search.best.clone().unwrap();
    let child_dir = temp.path().join("child");
    let child = run(&RunOptions {
        cuts: 2,
        rng_seed: 8,
        incumbent: Some(parent_dir.join("receipt.json")),
        ..small(&child_dir, 24)
    })
    .unwrap();
    let seeding = child.search.seeding.clone().unwrap();
    assert_eq!(
        seeding.parent_receipt_sha256.as_deref(),
        Some(
            crate::sha256_file(&parent_dir.join("receipt.json"))
                .unwrap()
                .as_str()
        )
    );
    let child_best = child.search.best.clone().unwrap();
    assert!(
        child_best.validation_rmse <= parent_best.validation_rmse,
        "child {} ({}) is worse than its seed {} ({})",
        child_best.genome,
        child_best.validation_rmse,
        parent_best.genome,
        parent_best.validation_rmse
    );
    // Same cuts, same fit: the re-measurement reproduces the parent's own numbers.
    assert_eq!(child.prepared.sha256, parent.prepared.sha256);
    let verified = crate::evidence::verify_run(&child_dir.join("receipt.json")).unwrap();
    let ranking = verified.ranking().unwrap();
    if parent_best.genome != search::BOOTSTRAP_GENOME {
        let again = ranking
            .iter()
            .find(|b| b.genome == parent_best.genome)
            .expect("the seed was measured on every cut");
        assert_eq!(
            again.validation_rmse.to_bits(),
            parent_best.validation_rmse.to_bits()
        );
    }
}

#[test]
fn a_tampered_parent_receipt_is_refused_before_anything_is_written() {
    let temp = tempfile::tempdir().unwrap();
    let parent_dir = temp.path().join("parent");
    run(&small(&parent_dir, 12)).unwrap();
    let receipt_path = parent_dir.join("receipt.json");
    let pristine = std::fs::read(&receipt_path).unwrap();

    // A better score than the evidence shows.
    let mut value: serde_json::Value = serde_json::from_slice(&pristine).unwrap();
    let rmse = value["search"]["best"]["validation_rmse"].as_f64().unwrap();
    value["search"]["best"]["validation_rmse"] = (rmse / 2.0).into();
    std::fs::write(&receipt_path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    let child = temp.path().join("child-score");
    let error = format!(
        "{:#}",
        run(&RunOptions {
            incumbent: Some(receipt_path.clone()),
            ..small(&child, 12)
        })
        .unwrap_err()
    );
    assert!(error.contains("--incumbent"), "{error}");
    assert!(!child.exists(), "a refused run wrote {}", child.display());

    // An observation edited under an untouched receipt.
    std::fs::write(&receipt_path, &pristine).unwrap();
    let observations = parent_dir.join("search/cut-0/observations.json");
    let mut rows: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&observations).unwrap()).unwrap();
    rows[0]["rmse"] = 0.0.into();
    std::fs::write(&observations, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
    let child = temp.path().join("child-observation");
    let error = format!(
        "{:#}",
        run(&RunOptions {
            incumbent: Some(receipt_path.clone()),
            ..small(&child, 12)
        })
        .unwrap_err()
    );
    assert!(error.contains("hashes to"), "{error}");
    assert!(!child.exists());
}

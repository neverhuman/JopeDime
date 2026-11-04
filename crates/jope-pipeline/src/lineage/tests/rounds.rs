use super::*;

#[test]
fn rounds_chain_on_identical_cuts_never_get_worse_and_verify() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("lineage");
    let lineage = evolve_into(&out, 3, true, template(24)).unwrap();
    assert_eq!(lineage.rounds.len(), 3);
    assert_eq!(lineage.rounds[0].kind, UNSEEDED);
    assert_eq!(lineage.rounds[0].seed_origin, FROM_NONE);
    for pair in lineage.rounds.windows(2) {
        assert!(pair[1].validation_rmse <= pair[0].validation_rmse);
        assert_eq!(
            pair[1].parent_receipt_sha256.as_deref(),
            Some(pair[0].receipt_sha256.as_str())
        );
        assert_eq!(pair[1].seed_origin, FROM_PREVIOUS_ROUND);
        assert_ne!(pair[1].rng_seed, pair[0].rng_seed);
    }
    let control = lineage.control.as_ref().unwrap();
    assert_eq!(control.kind, CONTROL);
    assert!(control.seeds.is_empty() && control.parent_receipt_sha256.is_none());
    assert_eq!(control.rng_seed, lineage.rounds[2].rng_seed);
    // Round 2 starts from round 1's champion and ranked runners-up, nothing else.
    let first = evidence::verify_run(&out.join("round-001/receipt.json")).unwrap();
    let expected: Vec<String> = seeds_after(&first)
        .unwrap()
        .into_iter()
        .map(|c| c.genome)
        .filter(|g| g != search::BOOTSTRAP_GENOME)
        .collect();
    assert_eq!(
        lineage.rounds[1]
            .seeds
            .iter()
            .map(|s| s.genome.clone())
            .collect::<Vec<_>>(),
        expected
    );
    // Per-generation best-so-far never rises, and every round measured bred candidates.
    for round in &lineage.rounds {
        for cut in &round.cuts {
            let values: Vec<f64> = cut.best_so_far.iter().flatten().copied().collect();
            assert!(values.windows(2).all(|w| w[1] <= w[0]), "{values:?}");
        }
        assert!(round.generation_ge1_measured > 0);
    }
    assert_eq!(read_lineage(&out.join(LINEAGE_FILE)).unwrap(), lineage);
    let report = verify(&out.join(LINEAGE_FILE), true).unwrap();
    assert_eq!(report.rounds, 3);
    assert!(report.bred_candidates_in_every_round);
    // What the rounds found is reported per round, and agrees with the RMSEs.
    assert_eq!(
        report.champion_found_in_round,
        lineage
            .rounds
            .iter()
            .map(Round::champion_found_in_round)
            .collect::<Vec<_>>()
    );
    assert_eq!(report.improved_on_previous_round.len(), 2);
    for (index, improved) in report.improved_on_previous_round.iter().enumerate() {
        let (before, after) = (&lineage.rounds[index], &lineage.rounds[index + 1]);
        assert_eq!(*improved, after.validation_rmse < before.validation_rmse);
        // Never worse, so a round that did not improve tied its parent exactly.
        if !*improved {
            assert_eq!(after.validation_rmse, before.validation_rmse);
        }
    }
    assert_eq!(
        report.control_validation_rmse,
        Some(control.validation_rmse)
    );
    // The compute the control comparison rests on: the rounds' total against one round's.
    assert_eq!(
        report.rounds_completed_evaluations,
        lineage
            .rounds
            .iter()
            .map(|r| r.completed_evaluations)
            .sum::<u64>()
    );
    assert_eq!(
        report.control_completed_evaluations,
        Some(control.completed_evaluations)
    );
    assert!(report.rounds_completed_evaluations > control.completed_evaluations);
}

#[test]
fn a_tampered_lineage_round_receipt_or_seeds_file_fails_verification() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("lineage");
    evolve_into(&out, 2, false, template(16)).unwrap();
    let path = out.join(LINEAGE_FILE);
    verify(&path, false).unwrap();
    let expect_refused = |needle: &str| {
        let error = format!("{:#}", verify(&path, false).unwrap_err());
        assert!(error.contains(needle), "wanted {needle:?}: {error}");
    };

    let pristine = std::fs::read(&path).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&pristine).unwrap();
    let rmse = value["rounds"][1]["validation_rmse"].as_f64().unwrap();
    value["rounds"][1]["validation_rmse"] = (rmse / 2.0).into();
    std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    expect_refused("does not match what its receipt and evidence show");
    std::fs::write(&path, &pristine).unwrap();

    // The island model is lineage-wide: a lineage that claims another one than its rounds
    // were searched with is refused, not reported.
    for (field, claimed) in [("islands", 3u64), ("migration_interval", 1)] {
        let mut value: serde_json::Value = serde_json::from_slice(&pristine).unwrap();
        assert_ne!(value[field].as_u64(), Some(claimed));
        value[field] = claimed.into();
        std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        expect_refused("another island model than the lineage's");
        std::fs::write(&path, &pristine).unwrap();
    }

    let seeds = out.join("round-002/seeds.json");
    let original = std::fs::read(&seeds).unwrap();
    std::fs::write(&seeds, [original.as_slice(), b" "].concat()).unwrap();
    expect_refused("seeds.json is not the file");
    std::fs::write(&seeds, &original).unwrap();

    let observations = out.join("round-001/search/cut-0/observations.json");
    let original = std::fs::read(&observations).unwrap();
    std::fs::write(&observations, [original.as_slice(), b" "].concat()).unwrap();
    expect_refused("round 1");
    std::fs::write(&observations, &original).unwrap();

    verify(&path, false).unwrap();
}

/// `--threads 1` and `--threads 8` write the same lineage: every round directory and
/// `lineage.json`, byte for byte, with the wall clock frozen (so the evaluation budget, never
/// the seconds bound, ends every phase; a binding seconds bound is timing-dependent).
#[test]
fn one_and_eight_threads_write_byte_identical_lineages() {
    let temp = tempfile::tempdir().unwrap();
    let write = |threads: usize, name: &str| {
        let out = temp.path().join(name);
        evolve_into(
            &out,
            2,
            true,
            RunOptions {
                threads,
                clock: Clock::Frozen,
                ..template(20)
            },
        )
        .unwrap();
        tree(&out)
    };
    let workers = crate::parallel::affinity_cpus()
        .unwrap()
        .min(std::thread::available_parallelism().map_or(1, std::num::NonZero::get))
        .min(8);
    if workers < 2 {
        eprintln!("skipped: this host admits one worker");
        return;
    }
    let one = write(1, "one");
    let many = write(workers, "many");
    assert_eq!(
        one.keys().collect::<Vec<_>>(),
        many.keys().collect::<Vec<_>>()
    );
    for (path, bytes) in &one {
        assert!(
            many[path] == *bytes,
            "{path} differs between 1 and {workers} threads"
        );
    }
    assert!(one.contains_key(LINEAGE_FILE) && one.contains_key("control/receipt.json"));
}

/// A stop request during round 2 (SIGTERM in the CLI) leaves round 2 without a receipt and
/// the lineage with round 1 only, which still verifies.
#[test]
fn a_stop_request_during_a_round_keeps_only_completed_rounds() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("lineage");
    let flag = Arc::new(AtomicBool::new(false));
    let watcher = {
        let flag = flag.clone();
        let prepared = out.join("round-002/prepared/prepared.json");
        std::thread::spawn(move || {
            while !prepared.exists() {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            flag.store(true, Ordering::SeqCst);
        })
    };
    let error = evolve_into(
        &out,
        3,
        true,
        RunOptions {
            cancel: Some(flag),
            ..template(30)
        },
    )
    .unwrap_err();
    watcher.join().unwrap();
    assert!(crate::parallel::is_cancelled(&error), "{error:#}");
    let lineage = read_lineage(&out.join(LINEAGE_FILE)).unwrap();
    assert_eq!(lineage.rounds.len(), 1);
    assert!(lineage.control.is_none());
    assert!(out.join("round-001/receipt.json").exists());
    assert!(!out.join("round-002/receipt.json").exists());
    assert!(!out.join("round-003").exists() && !out.join(CONTROL_DIRECTORY).exists());
    verify(&out.join(LINEAGE_FILE), false).unwrap();
}

/// A round whose champion is one of its own seeds found nothing better, however many bred
/// candidates it measured; the report must say so rather than call it a discovery.
#[test]
fn a_round_whose_champion_is_its_seed_is_not_reported_as_a_discovery() {
    let temp = tempfile::tempdir().unwrap();
    // A strong pipeline for this dataset, against 3 bred candidates in generation 1.
    let genome =
        seeds::canonical_pipeline("(pipe X (rank append=true) (gpr m=400 steps=60))").unwrap();
    let file = temp.path().join("strong.json");
    seeds::write(
        &file,
        &seeds::SeedsFile::new(
            None,
            vec![seeds::SeedGenome {
                program_sha256: sha256_hex(genome.as_bytes()),
                genome: genome.clone(),
                bank_entry_sha256: None,
                receipt_sha256: None,
            }],
        ),
    )
    .unwrap();
    let out = temp.path().join("lineage");
    let lineage = evolve(&EvolveOptions {
        run: template(16),
        output: out.clone(),
        rounds: 1,
        control: false,
        seeds: Some(file),
    })
    .unwrap();
    let round = &lineage.rounds[0];
    assert_eq!(
        round.selected_genome, genome,
        "the seed was beaten after all"
    );
    assert!(round.generation_ge1_measured > 0);
    assert!(!round.champion_found_in_round());
    assert_eq!(round.champion_origin(), "carried from its seeds");
    // The bred-candidates check still holds (the search ran past generation 0), and the
    // report says, separately, that nothing it bred won.
    let report = serde_json::to_value(verify(&out.join(LINEAGE_FILE), true).unwrap()).unwrap();
    assert_eq!(report["bred_candidates_in_every_round"], true, "{report}");
    assert_eq!(
        report["champion_found_in_round"],
        serde_json::json!([false]),
        "{report}"
    );
    assert_eq!(report["improved_on_previous_round"], serde_json::json!([]));
    assert!(report.get("discovery_in_every_round").is_none(), "{report}");
}

#[test]
fn round_one_can_start_from_a_seeds_file() {
    let temp = tempfile::tempdir().unwrap();
    let genome = seeds::canonical_pipeline("(pipe X (standardize) (ridge_gcv))").unwrap();
    let file = temp.path().join("bank-export.json");
    let sha = seeds::write(
        &file,
        &seeds::SeedsFile::new(
            None,
            vec![seeds::SeedGenome {
                program_sha256: sha256_hex(genome.as_bytes()),
                genome: genome.clone(),
                bank_entry_sha256: None,
                receipt_sha256: None,
            }],
        ),
    )
    .unwrap();
    let out = temp.path().join("lineage");
    let lineage = evolve(&EvolveOptions {
        run: template(16),
        output: out.clone(),
        rounds: 1,
        control: false,
        seeds: Some(file),
    })
    .unwrap();
    assert_eq!(lineage.seeds_file_sha256.as_deref(), Some(sha.as_str()));
    let round = &lineage.rounds[0];
    assert_eq!(
        (round.kind.as_str(), round.seed_origin.as_str()),
        (SEEDED, FROM_SEEDS_FILE)
    );
    assert_eq!(round.seeds[0].genome, genome);
    verify(&out.join(LINEAGE_FILE), false).unwrap();
}

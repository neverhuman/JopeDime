use super::*;

#[test]
fn classification_is_refused_before_writing_a_regression_run() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("binary");
    std::fs::create_dir(&input).unwrap();
    // An admitted binary catalog and genuine 0/1 labels must never reach
    // this regression-only search, even though both parse as numeric CSV.
    let csv: String = (0..60).map(|i| format!("{},{}\n", i, i % 2)).collect();
    let mut row: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../datasets/test/pmlb-229-pwlinear/metadata.json"
    ))
    .unwrap();
    row["task"] = "binary".into();
    row["target"]["type"] = "binary".into();
    row["shape"] = serde_json::json!({
        "rows":120,"features":1,"categorical_features":0,"columns":2
    });
    row["split"]["train_rows"] = 60.into();
    row["split"]["test_rows"] = 60.into();
    for name in ["train", "test"] {
        row["files"][name] = serde_json::json!({
            "sha256": crate::sha256_hex(csv.as_bytes()), "bytes": csv.len(),
            "rows": 60, "columns": 2, "has_header": false
        });
        std::fs::write(input.join(format!("{name}.csv")), &csv).unwrap();
    }
    std::fs::write(input.join("metadata.json"), row.to_string()).unwrap();
    let output = temp.path().join("run");
    let options = RunOptions {
        budget: Budget {
            evaluations: 1,
            seconds: 1,
        },
        seed: 1,
        rng_seed: 1,
        languages: vec![Language::Rust],
        policy: Policy::Skip,
        ..RunOptions::new(input.display().to_string(), output.clone())
    };
    let error = format!("{:#}", run(&options).unwrap_err());
    assert!(error.contains("task binary is unsupported"), "{error}");
    assert!(!output.exists(), "refusal left a misleading run directory");
}

/// The dataset is unresolvable, so this also proves the policy is checked
/// before any other work — including anything that writes to `--output`.
#[test]
fn require_fails_closed_before_the_output_directory_is_touched() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("run");
    let options = RunOptions {
        seed: 1,
        rng_seed: 1,
        languages: vec![Language::Rust],
        policy: Policy::Require,
        ..RunOptions::new("0000000000000000", output.clone())
    };
    let error = format!("{:#}", run(&options).unwrap_err());
    assert!(error.contains("--policy require"), "{error}");
    assert!(error.contains("--policy skip"), "{error}");
    assert!(!output.exists());
}

fn committed_dataset() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../datasets/test/pmlb-229-pwlinear"
    )
    .to_owned()
}

/// A copy of the committed fixture whose training target is constant: it resolves, is a
/// regression task and parses, but no cut of it can be scored.
pub(crate) fn constant_target_dataset(root: &std::path::Path) -> String {
    let source = std::path::PathBuf::from(committed_dataset());
    let target = root.join("constant-target");
    std::fs::create_dir_all(&target).unwrap();
    std::fs::copy(source.join("test.csv"), target.join("test.csv")).unwrap();
    let train: String = std::fs::read_to_string(source.join("train.csv"))
        .unwrap()
        .lines()
        .map(|line| format!("{},0.5\n", line.rsplit_once(',').unwrap().0))
        .collect();
    std::fs::write(target.join("train.csv"), &train).unwrap();
    let mut row: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source.join("metadata.json")).unwrap()).unwrap();
    row["files"]["train"]["sha256"] = sha256_hex(train.as_bytes()).into();
    row["files"]["train"]["bytes"] = train.len().into();
    std::fs::write(target.join("metadata.json"), row.to_string()).unwrap();
    target.display().to_string()
}

fn small(output: &std::path::Path, evaluations: usize) -> RunOptions {
    RunOptions {
        budget: Budget {
            evaluations,
            seconds: 300,
        },
        rng_seed: 7,
        languages: vec![Language::Rust],
        policy: Policy::Skip,
        ..RunOptions::new(committed_dataset(), output)
    }
}

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
type Decided = Vec<(
    usize,
    String,
    usize,
    &'static str,
    &'static str,
    Option<f64>,
    Option<f64>,
)>;

fn decided(run: &crate::evidence::VerifiedRun) -> Decided {
    run.cuts
        .iter()
        .flat_map(|(cut, observations)| {
            observations.iter().map(move |o| {
                (
                    *cut,
                    o.genome.clone(),
                    o.generation,
                    o.origin,
                    o.status,
                    o.rmse,
                    o.hold_r2,
                )
            })
        })
        .collect()
}

fn seeds_file(dir: &std::path::Path, genomes: &[&str]) -> std::path::PathBuf {
    let path = dir.join("seeds.json");
    let file = seeds::SeedsFile::new(
        None,
        genomes
            .iter()
            .map(|g| seeds::SeedGenome {
                genome: (*g).to_owned(),
                program_sha256: sha256_hex(g.as_bytes()),
                bank_entry_sha256: None,
                receipt_sha256: None,
            })
            .collect(),
    );
    seeds::write(&path, &file).unwrap();
    path
}

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

/// Every file under `root`, by relative path.
fn tree(root: &std::path::Path) -> std::collections::BTreeMap<String, Vec<u8>> {
    fn walk(
        root: &std::path::Path,
        dir: &std::path::Path,
        out: &mut std::collections::BTreeMap<String, Vec<u8>>,
    ) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                out.insert(
                    path.strip_prefix(root).unwrap().display().to_string(),
                    std::fs::read(&path).unwrap(),
                );
            }
        }
    }
    let mut out = std::collections::BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// The worker count this host admits, up to eight.
fn workers_up_to_eight() -> usize {
    let available = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    crate::parallel::affinity_cpus()
        .unwrap()
        .min(available)
        .min(8)
}

/// `--threads 1` and `--threads 8` write the same run: the receipt, the prepared cuts, every
/// cut's search and observations, and the emitted sources, byte for byte. The wall clock is
/// frozen for the comparison because it is the one thing two runs never share. A frozen clock
/// never reaches the seconds bound, so this proves the case where the evaluation budget ends
/// every phase, which the test asserts; a binding seconds bound makes the admitted candidates
/// timing-dependent (`a_phase_the_seconds_bound_ended_is_recorded_as_stopped_by_seconds`).
#[test]
fn one_thread_and_eight_threads_write_byte_identical_runs() {
    let workers = workers_up_to_eight();
    if workers < 2 {
        eprintln!("skipped: this host admits one worker, so no parallel run can be compared");
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let write = |threads: usize, name: &str| {
        let output = temp.path().join(name);
        let receipt = run(&RunOptions {
            threads,
            clock: Clock::Frozen,
            cuts: 2,
            ..small(&output, 30)
        })
        .unwrap();
        (receipt, tree(&output))
    };
    let (one, one_tree) = write(1, "one");
    let (many, many_tree) = write(workers, "many");
    assert_eq!(
        one_tree.keys().collect::<Vec<_>>(),
        many_tree.keys().collect::<Vec<_>>()
    );
    for (path, bytes) in &one_tree {
        assert!(
            many_tree[path] == *bytes,
            "{path} differs between 1 and {workers} threads"
        );
    }
    assert_eq!(one, many);
    assert!(one_tree.contains_key("receipt.json"));
    assert!(one.search.cuts.iter().all(|c| c
        .accounting
        .as_ref()
        .unwrap()
        .comparison
        .completed_evaluations
        > 0));
    // The regime this comparison covers: no phase was ended by the seconds bound.
    assert!(one.search.cuts.iter().all(|c| {
        let accounting = c.accounting.as_ref().unwrap();
        c.stopped_by != search::STOPPED_BY_SECONDS
            && accounting.exploration.stopped_by != search::STOPPED_BY_SECONDS
            && accounting.comparison.stopped_by != search::STOPPED_BY_SECONDS
    }));
}

#[test]
fn workers_above_the_cpu_ceiling_or_the_affinity_mask_are_refused_first() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("run");
    let cpus = crate::parallel::affinity_cpus().unwrap();
    let error = format!(
        "{:#}",
        run(&RunOptions {
            threads: cpus + 1,
            ..small(&output, 4)
        })
        .unwrap_err()
    );
    assert!(error.contains("sched_getaffinity"), "{error}");
    let error = format!(
        "{:#}",
        run(&RunOptions {
            cpu_ceiling_millicores: Some(999),
            ..small(&output, 4)
        })
        .unwrap_err()
    );
    assert!(error.contains("--cpu-ceiling 999"), "{error}");
    assert!(!output.exists(), "a refused run wrote {}", output.display());
}

#[test]
fn an_island_run_records_its_islands_and_migrations_and_reverifies() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("islands");
    let receipt = run(&RunOptions {
        islands: 2,
        migration_interval: 1,
        cuts: 2,
        ..small(&output, 60)
    })
    .unwrap();
    let islands = receipt.search.islands.clone().unwrap();
    assert_eq!((islands.count, islands.migration_interval), (2, 1));
    assert!(!islands.migrations.is_empty());
    let verified = crate::evidence::verify_run(&output.join("receipt.json")).unwrap();
    assert!(verified.cuts.iter().all(|(_, observations)| observations
        .iter()
        .all(|o| o.origin == search::REMEASURE || o.island.is_some())));

    // A migration edited in the receipt no longer matches what the cuts published.
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("receipt.json")).unwrap()).unwrap();
    let admitted = value["search"]["islands"]["migrations"][0]["admitted"]
        .as_bool()
        .unwrap();
    value["search"]["islands"]["migrations"][0]["admitted"] = (!admitted).into();
    std::fs::write(
        output.join("receipt.json"),
        serde_json::to_vec_pretty(&value).unwrap(),
    )
    .unwrap();
    let error = format!(
        "{:#}",
        crate::evidence::verify_run(&output.join("receipt.json")).unwrap_err()
    );
    assert!(error.contains("recorded migrations"), "{error}");
}

#[test]
fn an_objective_run_ranks_programs_by_it_records_it_and_reverifies() {
    let temp = tempfile::tempdir().unwrap();
    let objective = jope_runtime::fitness::Fitness::parse("pinball:quantile=0.9").unwrap();
    let output = temp.path().join("objective");
    let receipt = run(&RunOptions {
        cuts: 2,
        islands: 2,
        migration_interval: 1,
        objective: Some(objective.clone()),
        ..small(&output, 40)
    })
    .unwrap();
    // The objective receipt, not v3: v3 is frozen and cannot carry what ranked the programs.
    assert_eq!(receipt.schema, receipt::OBJECTIVE_SCHEMA_ID);
    let recorded = receipt.search.objective.clone().unwrap();
    assert_eq!(recorded.name, "pinball");
    assert_eq!(recorded.fitness_sha256, objective.identity().unwrap());
    assert_eq!(
        receipt.search.selection,
        select::MIN_MEAN_VALIDATION_OBJECTIVE
    );
    let best = receipt.search.best.clone().unwrap();
    let loss = best.validation_objective.unwrap();
    assert!(loss.is_finite() && loss >= 0.0);
    let verified = crate::evidence::verify_run(&output.join("receipt.json")).unwrap();
    // Every measured program carries the objective, and the champion has the lowest mean
    // objective loss among programs measured on every cut, not necessarily the lowest RMSE.
    assert!(verified
        .cuts
        .iter()
        .flat_map(|(_, o)| o)
        .filter(|o| o.measured())
        .all(|o| o.objective_loss.is_some()));
    let ranking = verified.ranking().unwrap();
    assert_eq!(ranking[0].genome, receipt.search.selected_genome);
    assert!(ranking
        .windows(2)
        .all(|w| w[0].validation_objective <= w[1].validation_objective));

    // Relabelled v3, the frozen schema refuses the objective it carries.
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("receipt.json")).unwrap()).unwrap();
    let mut relabelled = value.clone();
    relabelled["schema"] = receipt::RECEIPT_SCHEMA_ID.into();
    assert!(receipt::validate_value(&relabelled).is_err());
    // Removing the objective from the receipt no longer re-derives its champion's evidence.
    value["search"].as_object_mut().unwrap().remove("objective");
    std::fs::write(
        output.join("receipt.json"),
        serde_json::to_vec_pretty(&value).unwrap(),
    )
    .unwrap();
    assert!(crate::evidence::verify_run(&output.join("receipt.json")).is_err());

    // The search cannot score order-dependent objectives or objectives that read columns.
    for refused in [
        "sortino",
        "mase",
        r#"{"metric":"weighted","weights":{"op":"column","name":"w"},"fitness":{"metric":"mae"}}"#,
    ] {
        let options = RunOptions {
            objective: Some(jope_runtime::fitness::Fitness::parse(refused).unwrap()),
            ..small(&temp.path().join("refused"), 8)
        };
        assert!(run(&options).is_err(), "{refused}");
        assert!(!temp.path().join("refused").exists());
    }
}

#[test]
fn islands_times_threads_above_the_ceiling_is_refused() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("run");
    let error = format!(
        "{:#}",
        run(&RunOptions {
            islands: 2,
            cpu_ceiling_millicores: Some(1000),
            ..small(&output, 4)
        })
        .unwrap_err()
    );
    assert!(error.contains("--islands 2 x --threads 1"), "{error}");
    assert!(error.contains("--cpu-ceiling 1000"), "{error}");
    assert!(!output.exists());
}

/// A stop request mid-search (SIGTERM sets this flag in the CLI) ends the run between
/// evaluations with `Cancelled`: no receipt, and the interrupted cut is not published.
#[test]
fn a_stop_request_mid_search_writes_no_receipt() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("run");
    let flag = Arc::new(AtomicBool::new(false));
    let watcher = {
        let flag = flag.clone();
        let prepared = output.join("prepared/prepared.json");
        std::thread::spawn(move || {
            while !prepared.exists() {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            flag.store(true, Ordering::SeqCst);
        })
    };
    let error = run(&RunOptions {
        cancel: Some(flag),
        ..small(&output, 400)
    })
    .unwrap_err();
    watcher.join().unwrap();
    assert!(crate::parallel::is_cancelled(&error), "{error:#}");
    assert!(!output.join("receipt.json").exists());
    assert!(!output.join("search/cut-0/search.json").exists());
    assert!(!output.join("emit").exists());
    // Positive control: the same run without a stop request completes.
    let control = temp.path().join("control");
    run(&RunOptions {
        cancel: Some(Arc::new(AtomicBool::new(false))),
        ..small(&control, 4)
    })
    .unwrap();
    assert!(control.join("receipt.json").exists());
}

#[test]
fn a_given_source_commit_is_the_receipts_git_sha_and_a_malformed_one_is_refused() {
    let temp = tempfile::tempdir().unwrap();
    let commit = "a".repeat(40);
    let output = temp.path().join("run");
    let receipt = run(&RunOptions {
        source_commit: Some(commit.clone()),
        ..small(&output, 2)
    })
    .unwrap();
    assert_eq!(receipt.git_sha.as_deref(), Some(commit.as_str()));
    for bad in ["HEAD", "ABCDEF", &"a".repeat(41), &"g".repeat(40)] {
        let output = temp.path().join("bad");
        let error = format!(
            "{:#}",
            run(&RunOptions {
                source_commit: Some(bad.to_owned()),
                ..small(&output, 2)
            })
            .unwrap_err()
        );
        assert!(error.contains("--source-commit"), "{error}");
        assert!(!output.exists());
    }
}

#[test]
fn a_budget_that_cannot_remeasure_the_seeds_is_refused() {
    let temp = tempfile::tempdir().unwrap();
    let file = seeds_file(temp.path(), &["(pipe X (standardize) (ridge))"]);
    let output = temp.path().join("run");
    let error = format!(
        "{:#}",
        run(&RunOptions {
            seed_genomes: Some(file),
            ..small(&output, 1)
        })
        .unwrap_err()
    );
    assert!(error.contains("cannot re-measure"), "{error}");
    assert!(!output.exists());
}

/// A cut count the partition refuses (0, or more than its folds) and a dataset it cannot cut
/// are refused, by `check` and by `run`, before `--output` is created, so the corrected run
/// goes into the same directory. `run` used to create `--output` first.
#[test]
fn a_refused_cut_count_or_partition_leaves_no_output_and_the_corrected_run_succeeds() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("run");
    for cuts in [0, prepare::FOLDS + 1] {
        let options = RunOptions {
            cuts,
            ..small(&output, 4)
        };
        let expected = format!("--cuts must be 1..={}, got {cuts}", prepare::FOLDS);
        for error in [check(&options).unwrap_err(), run(&options).unwrap_err()] {
            let error = format!("{error:#}");
            assert!(error.contains(&expected), "{error}");
        }
        assert!(!output.exists(), "--cuts {cuts} left {}", output.display());
    }
    let constant = RunOptions {
        dataset: constant_target_dataset(temp.path()),
        cuts: 3,
        ..small(&output, 4)
    };
    for error in [check(&constant).unwrap_err(), run(&constant).unwrap_err()] {
        let error = format!("{error:#}");
        assert!(error.contains("held-out target is constant"), "{error}");
    }
    assert!(
        !output.exists(),
        "a partition refusal left {}",
        output.display()
    );
    // Positive control: the corrected command, into the same --output.
    let receipt = run(&RunOptions {
        cuts: 3,
        ..small(&output, 4)
    })
    .unwrap();
    assert_eq!(receipt.prepared.cuts, vec![0, 1, 2]);
    crate::evidence::verify_run(&output.join("receipt.json")).unwrap();
}

/// An entry the run writes is refused even as a dangling symbolic link. `exists()` follows
/// links, so a dangling `emit` let the run write `prepared/` and `search/` and then fail
/// ("File exists"), after which the corrected retry was refused as "already holds a pipeline
/// run (prepared)"; a dangling `receipt.json` had the receipt written through it, outside
/// `--output`. An unrelated file is left alone and the run goes ahead.
#[test]
fn an_entry_the_run_writes_is_refused_even_as_a_dangling_link() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("run");
    std::fs::create_dir(&output).unwrap();
    std::fs::write(output.join("notes.txt"), b"kept\n").unwrap();
    let outside = temp.path().join("outside");
    std::fs::create_dir(&outside).unwrap();
    let names = |dir: &std::path::Path| {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    };
    for entry in ["receipt.json", "prepared", "search", "emit"] {
        std::os::unix::fs::symlink(outside.join(entry), output.join(entry)).unwrap();
        let error = format!("{:#}", run(&small(&output, 4)).unwrap_err());
        assert!(
            error.contains(&format!("already holds a pipeline run ({entry})")),
            "{error}"
        );
        let mut left = vec![entry, "notes.txt"];
        left.sort_unstable();
        assert_eq!(names(&output), left, "{entry}: it wrote into --output");
        assert!(names(&outside).is_empty(), "{entry}: it wrote outside");
        std::fs::remove_file(output.join(entry)).unwrap();
    }
    // Positive control: the same command into the same --output, the unrelated file kept.
    run(&small(&output, 4)).unwrap();
    crate::evidence::verify_run(&output.join("receipt.json")).unwrap();
    assert_eq!(std::fs::read(output.join("notes.txt")).unwrap(), b"kept\n");
    assert!(names(&outside).is_empty());
}

/// The byte identity across thread counts is a property of runs the evaluation budget ends:
/// the frozen clock never reaches a seconds bound. When the seconds bound ends a phase, how
/// many candidates started before it depends on how fast the fits ran, and so on the worker
/// count; the receipt discloses that case as the phase's stop reason, `seconds`. The
/// test-only expired clock reads past every deadline, so this is deterministic.
#[test]
fn a_phase_the_seconds_bound_ended_is_recorded_as_stopped_by_seconds() {
    let temp = tempfile::tempdir().unwrap();
    let exploration_stops = |clock: Clock, name: &str| {
        run(&RunOptions {
            clock,
            cuts: 2,
            ..small(&temp.path().join(name), 30)
        })
        .unwrap()
        .search
        .cuts
        .iter()
        .map(|c| {
            c.accounting
                .as_ref()
                .unwrap()
                .exploration
                .stopped_by
                .clone()
        })
        .collect::<Vec<_>>()
    };
    assert_eq!(
        exploration_stops(Clock::Expired, "expired"),
        vec![search::STOPPED_BY_SECONDS; 2]
    );
    let frozen = exploration_stops(Clock::Frozen, "frozen");
    assert!(
        frozen.iter().all(|s| s != search::STOPPED_BY_SECONDS),
        "{frozen:?}"
    );
}

#[test]
fn policies_parse() {
    assert_eq!(Policy::parse("auto"), Some(Policy::Auto));
    assert_eq!(Policy::parse(" SKIP "), Some(Policy::Skip));
    assert_eq!(Policy::parse("require"), Some(Policy::Require));
    assert_eq!(Policy::parse("maybe"), None);
    for policy in [Policy::Auto, Policy::Skip, Policy::Require] {
        assert_eq!(Policy::parse(policy.as_str()), Some(policy));
    }
}

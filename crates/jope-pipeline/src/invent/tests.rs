use super::*;

fn write_csv(dir: &Path, rows: usize) -> PathBuf {
    let path = dir.join("train.csv");
    let mut text = String::from("a,y,b\n");
    for i in 0..rows {
        let a = i as f64 * 0.37 % 5.0;
        let b = (i as f64 * 1.3).sin();
        let y = 2.0 * a - 3.0 * b + 0.01 * (i % 7) as f64;
        text.push_str(&format!("{a},{y},{b}\n"));
    }
    std::fs::write(&path, text).unwrap();
    path
}

#[test]
fn csv_moves_the_named_target_last_and_keeps_feature_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_csv(dir.path(), 4);
    let (features, rows, digest) = load_csv(&path, "y").unwrap();
    assert_eq!(features, ["a", "b"]);
    assert_eq!(rows.len(), 4);
    let (a, b) = (rows[1][0], rows[1][1]);
    assert!((rows[1][2] - (2.0 * a - 3.0 * b + 0.01)).abs() < 1e-12);
    assert_eq!(digest.len(), 64);
}

#[test]
fn csv_refuses_missing_target_text_and_non_finite_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_csv(dir.path(), 4);
    let error = load_csv(&path, "z").unwrap_err().to_string();
    assert!(error.contains("--target \"z\""), "{error}");
    for (body, needle) in [
        ("a,y\n1,NA\n", "not a number"),
        ("a,y\n1,inf\n", "not finite"),
        ("a,y\n1\n", "field"),
    ] {
        let bad = dir.path().join("bad.csv");
        std::fs::write(&bad, body).unwrap();
        let error = format!("{:#}", load_csv(&bad, "y").unwrap_err());
        assert!(error.contains(needle), "{body:?}: {error}");
    }
}

#[test]
fn work_dir_must_be_new_or_empty_and_is_checked_before_anything_is_written() {
    let dir = tempfile::tempdir().unwrap();
    ensure_fresh_work_dir(&dir.path().join("new")).unwrap();
    let empty = dir.path().join("empty");
    std::fs::create_dir(&empty).unwrap();
    ensure_fresh_work_dir(&empty).unwrap();

    let used = dir.path().join("used");
    std::fs::create_dir_all(used.join("prepared")).unwrap();
    std::fs::write(used.join("prepared").join("prepared.json"), "{}").unwrap();
    let error = ensure_fresh_work_dir(&used).unwrap_err().to_string();
    assert!(error.contains("not empty"), "{error}");

    let file = dir.path().join("file");
    std::fs::write(&file, "x").unwrap();
    let error = ensure_fresh_work_dir(&file).unwrap_err().to_string();
    assert!(error.contains("not a directory"), "{error}");

    // `invent` refuses the used directory and leaves what was there untouched.
    let train = write_csv(dir.path(), 60);
    let options = InventOptions::new(train, "y", &used);
    let error = format!("{:#}", invent(&options).unwrap_err());
    assert!(error.contains("not empty"), "{error}");
    assert_eq!(
        std::fs::read_to_string(used.join("prepared").join("prepared.json")).unwrap(),
        "{}"
    );
    assert_eq!(std::fs::read_dir(&used).unwrap().count(), 1);
}

#[test]
fn only_jain_executable_pipelines_are_admitted() {
    let executable = Executable::load().unwrap();
    assert_eq!(executable.engine, "featmath");
    assert_eq!(executable.rev.len(), 40);
    assert!(executable.admits("(pipe X (standardize) (ridge))"));
    // A search head jain-math 9fa594e does not implement.
    assert!(!executable.admits("(pipe X (standardize) (ridge_gcv))"));
    // Quarantined in jain-math.
    assert!(!executable.admits("(pipe X (rff_orthogonal) (ridge))"));
    assert!(!executable.admits("not a genome"));
}

#[test]
fn events_refuses_unresolved_parent_work_alias() {
    let dir = tempfile::tempdir().unwrap();
    let train = write_csv(dir.path(), 4);
    let work = dir.path().join("missing/../work");
    let events = dir.path().join("work/prepared/cuts.ndjson");
    let result = check_events(&events, &train, &work);
    assert!(
        result.is_err(),
        "unresolved-parent --work-dir alias was accepted"
    );
    assert!(!dir.path().join("work").exists());
    assert!(!dir.path().join("missing").exists());
}

#[test]
fn events_refuses_input_and_output_aliases() {
    let dir = tempfile::tempdir().unwrap();
    let train = write_csv(dir.path(), 4);
    let work = dir.path().join("work");
    let check = |events: &Path| check_events(events, &train, &work);
    // The input, by another spelling too.
    let msg = check(&train).unwrap_err().to_string();
    assert!(msg.contains("--train"), "{msg}");
    assert!(check(&dir.path().join("work/../train.csv")).is_err());
    // The work directory itself (not yet created) and anything under it.
    assert!(check(&work).unwrap_err().to_string().contains("--work-dir"));
    assert!(check(&work.join("progress.ndjson")).is_err());
    // An existing directory.
    let msg = check(dir.path()).unwrap_err().to_string();
    assert!(msg.contains("directory"), "{msg}");
    // A new file beside them is fine.
    check(&dir.path().join("progress.ndjson")).unwrap();
}

#[cfg(unix)]
#[test]
fn events_refuses_hard_link_input_and_nonregular_targets() {
    let dir = tempfile::tempdir().unwrap();
    let train = write_csv(dir.path(), 4);
    let before = std::fs::read(&train).unwrap();
    let work = dir.path().join("work");
    let alias = dir.path().join("input-alias.ndjson");
    std::fs::hard_link(&train, &alias).unwrap();
    let refused = check_events(&alias, &train, &work);
    assert!(refused.is_err(), "hard-linked --train was accepted");
    assert_eq!(std::fs::read(&train).unwrap(), before);
    assert!(!work.exists());
    // A Unix socket is an existing nonregular filesystem target. Never open it: the
    // preflight itself must refuse it, just as it must refuse a FIFO/device.
    let socket = dir.path().join("events.socket");
    let _listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    assert!(check_events(&socket, &train, &work).is_err());
    assert_eq!(std::fs::read(&train).unwrap(), before);
    assert!(!work.exists());
}

#[cfg(target_os = "linux")]
#[test]
fn events_open_preserves_input_and_truncates_only_valid_regular_output() {
    let dir = tempfile::tempdir().unwrap();
    let train = write_csv(dir.path(), 4);
    let before = std::fs::read(&train).unwrap();
    let work = dir.path().join("work");
    let alias = dir.path().join("alias");
    std::fs::hard_link(&train, &alias).unwrap();
    assert!(open_events(&alias, &train, &work).is_err());
    assert_eq!(std::fs::read(&train).unwrap(), before);
    let output = dir.path().join("events.ndjson");
    std::fs::write(&output, b"old journal").unwrap();
    let file = open_events(&output, &train, &work).unwrap();
    assert_eq!(file.metadata().unwrap().len(), 0);
    assert_eq!(std::fs::read(&train).unwrap(), before);
    assert!(!work.exists());
}

#[test]
fn preflight_refuses_bad_counts_before_reading_anything() {
    let mut options = InventOptions::new("/nonexistent.csv", "y", "/nonexistent");
    options.top_k = 0;
    assert!(preflight(&options)
        .unwrap_err()
        .to_string()
        .contains("--top-k"));
    options.top_k = 8;
    options.cuts = prepare::FOLDS + 1;
    assert!(preflight(&options).is_err());
    options.cuts = 3;
    options.source_commit = Some("abc".into());
    assert!(preflight(&options)
        .unwrap_err()
        .to_string()
        .contains("--source-commit"));
}

#[test]
fn invent_ranks_executable_genomes_and_is_deterministic_under_an_evaluation_budget() {
    let dir = tempfile::tempdir().unwrap();
    let train = write_csv(dir.path(), 120);
    let run = |work: &str| {
        let mut options = InventOptions::new(&train, "y", dir.path().join(work));
        options.budget = search::Budget {
            evaluations: 24,
            seconds: 600,
        };
        options.cuts = 2;
        options.top_k = 4;
        invent(&options).unwrap()
    };
    let first = run("one");
    assert!(first.ok, "{first:?}");
    assert_eq!(first.schema, SCHEMA);
    assert_eq!(first.rows, 120);
    assert_eq!(first.feature_names, ["a", "b"]);
    assert!(!first.candidates.is_empty() && first.candidates.len() <= 4);
    let executable = Executable::load().unwrap();
    for (index, candidate) in first.candidates.iter().enumerate() {
        assert_eq!(candidate.rank, index + 1);
        assert_ne!(candidate.origin, search::REMEASURE, "{}", candidate.genome);
        assert!(executable.admits(&candidate.genome), "{}", candidate.genome);
        assert_eq!(candidate.cuts_measured, 2);
        assert!(candidate.mean_hold_rmse.is_finite());
    }
    assert!(first
        .candidates
        .windows(2)
        .all(|w| w[0].mean_hold_rmse <= w[1].mean_hold_rmse));
    assert!(first.ranked - first.excluded_not_executable >= first.candidates.len());
    let second = run("two");
    let genomes = |c: &Candidates| {
        c.candidates
            .iter()
            .map(|c| c.genome.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(genomes(&first), genomes(&second));
    // A used work directory is refused.
    let mut again = InventOptions::new(&train, "y", dir.path().join("one"));
    again.budget = search::Budget {
        evaluations: 24,
        seconds: 600,
    };
    assert!(invent(&again)
        .unwrap_err()
        .to_string()
        .contains("--work-dir"));
}

#[test]
fn a_finish_request_ranks_what_was_measured_and_says_it_finished_early() {
    let dir = tempfile::tempdir().unwrap();
    let train = write_csv(dir.path(), 120);
    let budget = search::Budget {
        evaluations: 24,
        seconds: 600,
    };
    // Asked to finish before the first generation: only generation 0's exempt head is fitted
    // on each cut, the comparison still measures it on every cut, and it is ranked as usual.
    let mut options = InventOptions::new(&train, "y", dir.path().join("finished"));
    options.budget = budget;
    options.cuts = 2;
    options.top_k = 4;
    options.finish = Some(Arc::new(AtomicBool::new(true)));
    let finished = invent(&options).unwrap();
    assert!(finished.ok, "{finished:?}");
    assert!(finished.finished_early);
    assert!(!finished.candidates.is_empty());
    assert!(finished.candidates.iter().all(|c| c.cuts_measured == 2));
    // The cut record's `stopped_by` is the comparison's outcome, as on any run; what shows
    // the finish is how little each cut tried.
    for cut in &finished.searches {
        assert!(
            cut.completed_evaluations < budget.evaluations as u64,
            "{cut:?}"
        );
    }
    let document = serde_json::to_value(&finished).unwrap();
    assert_eq!(document["finished_early"], true);

    // Nobody asked: the flag stays down and the document has no such key.
    let mut options = InventOptions::new(&train, "y", dir.path().join("full"));
    options.budget = budget;
    options.cuts = 2;
    options.top_k = 4;
    options.finish = Some(Arc::new(AtomicBool::new(false)));
    let full = invent(&options).unwrap();
    assert!(!full.finished_early);
    assert!(serde_json::to_value(&full)
        .unwrap()
        .get("finished_early")
        .is_none());
    let tried = |c: &Candidates| {
        c.searches
            .iter()
            .map(|s| s.completed_evaluations)
            .sum::<u64>()
    };
    assert!(tried(&finished) < tried(&full));
}

#[test]
fn invent_ranks_candidates_by_an_objective_and_records_it() {
    let dir = tempfile::tempdir().unwrap();
    let train = write_csv(dir.path(), 120);
    let objective = jope_runtime::fitness::Fitness::parse(
        r#"{"metric":"asymmetric_absolute","under_cost":4,"over_cost":1}"#,
    )
    .unwrap();
    let mut options = InventOptions::new(&train, "y", dir.path().join("objective"));
    options.budget = search::Budget {
        evaluations: 24,
        seconds: 600,
    };
    options.cuts = 2;
    options.top_k = 4;
    options.objective = Some(objective.clone());
    let ranked = invent(&options).unwrap();
    assert_eq!(
        ranked.objective.as_ref().unwrap().fitness_sha256,
        objective.identity().unwrap()
    );
    assert!(ranked
        .candidates
        .iter()
        .all(|c| c.mean_hold_objective.is_some_and(f64::is_finite)));
    assert!(ranked
        .candidates
        .windows(2)
        .all(|w| w[0].mean_hold_objective <= w[1].mean_hold_objective));
    let mut refused = InventOptions::new(&train, "y", dir.path().join("refused"));
    refused.objective = Some(jope_runtime::fitness::Fitness::parse("sharpe").unwrap());
    assert!(invent(&refused).is_err());
    assert!(!dir.path().join("refused").exists());
}

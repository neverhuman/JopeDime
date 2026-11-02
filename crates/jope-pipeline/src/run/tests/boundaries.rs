use super::*;

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

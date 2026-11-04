use super::*;

#[test]
fn unsupported_cuts_concurrency_seeds_and_existing_output_fail_before_writing() {
    let temp = tempfile::tempdir().unwrap();
    for mode in ["cuts", "threads", "islands", "incumbent"] {
        let output = temp.path().join(mode);
        let mut opt = options(&output, 4);
        match mode {
            "cuts" => opt.run.cuts = 2,
            "threads" => opt.run.threads = 2,
            "islands" => opt.run.islands = 2,
            _ => opt.run.incumbent = Some("missing".into()),
        }
        assert!(run::run(&opt).is_err());
        assert!(!output.exists());
    }
    let output = temp.path().join("seed");
    let mut opt = options(&output, 4);
    let file = temp.path().join("invalid-seed.json");
    let seeds = crate::seeds::SeedsFile::new(
        None,
        vec![seed_entry("(pipe X (derive expr=\"exp(c0)\") (ridge))")],
    );
    std::fs::write(&file, serde_json::to_vec(&seeds).unwrap()).unwrap();
    opt.run.seed_genomes = Some(file);
    assert!(run::run(&opt).is_err());
    assert!(!output.exists());
    let output = temp.path().join("existing");
    std::fs::create_dir(&output).unwrap();
    std::os::unix::fs::symlink(temp.path().join("outside"), output.join("emit")).unwrap();
    assert!(run::run(&options(&output, 4)).is_err());
    assert_eq!(std::fs::read_dir(output).unwrap().count(), 1);
}

#[test]
fn cancelled_run_writes_nothing_and_the_same_uncancelled_command_succeeds() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("run");
    let mut opt = options(&output, 4);
    let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    opt.run.cancel = Some(cancelled.clone());
    assert!(crate::parallel::is_cancelled(&run::run(&opt).unwrap_err()));
    assert!(!output.exists());
    cancelled.store(false, std::sync::atomic::Ordering::SeqCst);
    run::run(&opt).unwrap();
    publication::verify_run(&output.join("receipt.json")).unwrap();
}

#[test]
fn profile_runs_replay_receipt_and_all_sidecars_with_a_frozen_clock() {
    let temp = tempfile::tempdir().unwrap();
    let a = options(&temp.path().join("a"), 16);
    let b = options(&temp.path().join("b"), 16);
    assert_eq!(run::run(&a).unwrap(), run::run(&b).unwrap());
    for relative in [
        "receipt.json",
        "prepared/prepared.json",
        "search/cut-0/search.json",
        "search/cut-0/observations.json",
        "search/cut-0/refinements.json",
        "emit/model.rs",
        "emit/manifest.json",
    ] {
        assert_eq!(
            std::fs::read(a.run.output.join(relative)).unwrap(),
            std::fs::read(b.run.output.join(relative)).unwrap(),
            "{relative}"
        );
    }
}

#[test]
fn lm_runs_replay_receipt_and_all_sidecars_with_a_frozen_clock() {
    let temp = tempfile::tempdir().unwrap();
    let a = lm_seeded(&temp.path().join("a"), 40);
    let b = lm_seeded(&temp.path().join("b"), 40);
    let receipt = run::run(&a).unwrap();
    assert_eq!(receipt, run::run(&b).unwrap());
    for relative in SIDECARS {
        assert_eq!(
            std::fs::read(a.run.output.join(relative)).unwrap(),
            std::fs::read(b.run.output.join(relative)).unwrap(),
            "{relative}"
        );
    }
    let engine = receipt.search.engine_selection.as_ref().unwrap();
    assert_eq!(
        engine.refine.as_ref().unwrap().id,
        crate::kernel2::DAMPED_LS
    );
    assert!(receipt.search.objective_trials > 0);
    assert_eq!(
        publication::verify_run(&a.run.output.join("receipt.json")).unwrap(),
        receipt
    );
    let refinements: Value = serde_json::from_slice(
        &std::fs::read(a.run.output.join("search/cut-0/refinements.json")).unwrap(),
    )
    .unwrap();
    assert!(refinements
        .as_array()
        .unwrap()
        .iter()
        .any(|r| r["refinement"]["trace"]["stages"]
            .as_array()
            .is_some_and(|s| s.iter().any(|s| s["kind"] == "surrogate"))));
}

#[test]
fn explicit_coordinate_method_keeps_receipt_bytes_and_lm_flags_are_checked() {
    let temp = tempfile::tempdir().unwrap();
    let source = "a".repeat(40);
    let dataset = options(&temp.path().join("unused"), 4).run.dataset;
    let base = [
        "--search-profile",
        "kernel2-baseline",
        "--refine-top-k",
        "1",
        "--refine-trials",
        "4",
        "--budget",
        "evaluations=6,seconds=300",
        "--policy",
        "skip",
        "--emit",
        "rust",
        "--rng-seed",
        "7",
        "--source-commit",
        source.as_str(),
    ];
    let make = |extra: &[&str], output: PathBuf| {
        let mut argv = base.to_vec();
        argv.extend_from_slice(extra);
        let (mut search, profile) = args(&argv);
        search.dataset = dataset.clone();
        let mut opt = profile.options(&search, output, false).unwrap().unwrap();
        opt.run.clock = Clock::Frozen;
        opt
    };
    let implicit = make(&[], temp.path().join("implicit"));
    let explicit = make(
        &["--refine-method", "coordinate"],
        temp.path().join("explicit"),
    );
    let receipt = run::run(&implicit).unwrap();
    assert_eq!(receipt, run::run(&explicit).unwrap());
    for relative in SIDECARS {
        assert_eq!(
            std::fs::read(implicit.run.output.join(relative)).unwrap(),
            std::fs::read(explicit.run.output.join(relative)).unwrap(),
            "{relative}"
        );
    }
    assert!(serde_json::to_value(&receipt).unwrap()["search"]
        .get("engine_selection")
        .is_none());
    assert!(!String::from_utf8(
        std::fs::read(implicit.run.output.join("search/cut-0/refinements.json")).unwrap()
    )
    .unwrap()
    .contains("\"trace\""));
    make(&["--refine-method", "lm"], temp.path().join("lm"));
    for (flags, refusal) in [
        (
            vec!["--refine-method", "lm"],
            "legacy search refuses kernel2 fault/refinement options",
        ),
        (
            vec![
                "--search-profile",
                "legacy",
                "--refine-method",
                "coordinate",
            ],
            "legacy search refuses kernel2 fault/refinement options",
        ),
        (
            vec![
                "--search-profile",
                "kernel2-baseline",
                "--refine-top-k",
                "1",
                "--refine-trials",
                "4",
                "--refine-method",
                "fast",
            ],
            "--refine-method must be coordinate or lm",
        ),
        (
            vec![
                "--search-profile",
                "kernel2-baseline",
                "--refine-top-k",
                "1",
                "--refine-trials",
                "2",
                "--refine-method",
                "lm",
            ],
            "--refine-trials of at least 3",
        ),
        (
            vec![
                "--search-profile",
                "kernel2-baseline",
                "--refine-top-k",
                "0",
                "--refine-trials",
                "4",
                "--refine-method",
                "lm",
            ],
            "--refine-top-k of at least 1",
        ),
    ] {
        let output = temp.path().join("refused");
        let (mut search, profile) = args(&flags);
        search.dataset = dataset.clone();
        let error = profile.options(&search, output.clone(), false).unwrap_err();
        assert!(
            format!("{error:#}").contains(refusal),
            "{flags:?}: {error:#}"
        );
        assert!(!output.exists());
    }
}

#[test]
fn rehashed_lm_traces_and_selection_records_are_refused() {
    let temp = tempfile::tempdir().unwrap();
    let opt = lm_seeded(&temp.path().join("run"), 40);
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
    let path = output.join("receipt.json");
    let originals: Vec<(PathBuf, Vec<u8>)> = [
        "receipt.json",
        "search/cut-0/search.json",
        "search/cut-0/refinements.json",
    ]
    .iter()
    .map(|r| (output.join(r), std::fs::read(output.join(r)).unwrap()))
    .collect();
    let restore = || {
        for (file, bytes) in &originals {
            std::fs::write(file, bytes).unwrap();
        }
    };
    let engine = receipt.search.engine_selection.clone();
    let refinements: Value = serde_json::from_slice(&originals[2].1).unwrap();
    // Rewriting the untouched evidence through the same path still verifies.
    rewrite_lm_evidence(output, &receipt, &refinements, engine.clone());
    publication::verify_run(&path).unwrap();
    restore();
    let traced = refinements
        .as_array()
        .unwrap()
        .iter()
        .position(|r| r["refinement"]["trace"].is_object())
        .expect("a refinement with a trace");
    let stage = &refinements[traced]["refinement"]["trace"]["stages"][0];
    let mut edits: Vec<(&str, Value)> = Vec::new();
    let mut dropped = refinements.clone();
    dropped[traced]["refinement"]
        .as_object_mut()
        .unwrap()
        .remove("trace");
    edits.push(("dropped trace", dropped));
    let mut reordered = refinements.clone();
    reordered[traced]["refinement"]["trace"]["method"] = "feat-math.other.v1".into();
    edits.push(("unknown method", reordered));
    if stage["kind"] == "surrogate" {
        for field in [
            "trials",
            "jacobians",
            "initial",
            "rescore",
            "rejected_trials",
        ] {
            let mut edited = refinements.clone();
            let value = &mut edited[traced]["refinement"]["trace"]["stages"][0]["surrogate"][field];
            *value = (value.as_u64().unwrap() + 1).into();
            edits.push((field, edited));
        }
        if stage["pipeline"].is_object() {
            let mut flipped = refinements.clone();
            let accepted =
                &mut flipped[traced]["refinement"]["trace"]["stages"][0]["pipeline"]["accepted"];
            *accepted = (!accepted.as_bool().unwrap()).into();
            edits.push(("pipeline decision", flipped));
        }
    } else {
        let mut edited = refinements.clone();
        let value = &mut edited[traced]["refinement"]["trace"]["stages"][0]["evaluations"];
        *value = (value.as_u64().unwrap() + 1).into();
        edits.push(("coordinate evaluations", edited));
    }
    for (label, edited) in &edits {
        let path = rewrite_lm_evidence(output, &receipt, edited, engine.clone());
        assert!(publication::verify_run(&path).is_err(), "{label}");
        restore();
    }
    // A trace without the selection that produces it.
    let path = rewrite_lm_evidence(output, &receipt, &refinements, None);
    assert!(publication::verify_run(&path).is_err(), "unselected trace");
    restore();
    // A selection the receipt cannot name.
    let mut value = serde_json::to_value(&receipt).unwrap();
    value["search"]["engine_selection"]["refine"]["id"] = "feat-math.other.v1".into();
    assert!(receipt::validate_value(&value).is_err());
    let mut value = serde_json::to_value(&receipt).unwrap();
    value["search"]["engine_selection"]["refine"]["jacobian"] = "one".into();
    receipt::validate_value(&value).unwrap();
    // With more than one literal site the two charges differ, so the other charge is refused.
    let multi_site = refinements.as_array().unwrap().iter().any(|r| {
        r["refinement"]["trace"]["stages"]
            .as_array()
            .is_some_and(|s| s.iter().any(|s| s["surrogate"]["sites"].as_u64() > Some(1)))
    });
    if multi_site {
        let mut record = engine.clone().unwrap();
        record.refine.as_mut().unwrap().jacobian = "one".into();
        let path = rewrite_lm_evidence(output, &receipt, &refinements, Some(record));
        assert!(publication::verify_run(&path).is_err(), "jacobian charge");
        restore();
    }
    publication::verify_run(&path).unwrap();
    // A coordinate run's refinements cannot claim the selection after the fact.
    let plain = seeded(&temp.path().join("plain"));
    let plain_receipt = run::run(&plain).unwrap();
    let plain_refinements: Value = serde_json::from_slice(
        &std::fs::read(plain.run.output.join("search/cut-0/refinements.json")).unwrap(),
    )
    .unwrap();
    let path = rewrite_lm_evidence(
        &plain.run.output,
        &plain_receipt,
        &plain_refinements,
        engine,
    );
    assert!(publication::verify_run(&path).is_err(), "claimed selection");
}

#[test]
/// `--refine-trials 4 --refine-method lm` on a single-site seed: no stage reaches its surrogate
/// minimum (one site plus five), so the run is the coordinate run, measurements and refinements
/// bit for bit, and every trace stage names the small allowance.
fn lm_under_the_surrogate_minimum_is_the_coordinate_run_with_its_reason() {
    let temp = tempfile::tempdir().unwrap();
    let seed = "(pipe X (derive expr=\"sin(0.5*c0)\") (ridge))";
    let coordinate = flag_run(&temp.path().join("coordinate"), seed, 40, 4, "coordinate");
    let lm = flag_run(&temp.path().join("lm"), seed, 40, 4, "lm");
    let (plain, traced) = (run::run(&coordinate).unwrap(), run::run(&lm).unwrap());
    for (opt, receipt) in [(&coordinate, &plain), (&lm, &traced)] {
        assert_eq!(
            publication::verify_run(&opt.run.output.join("receipt.json")).unwrap(),
            *receipt
        );
    }
    let read =
        |opt: &run::Options, relative: &str| std::fs::read(opt.run.output.join(relative)).unwrap();
    assert_eq!(
        read(&coordinate, "search/cut-0/observations.json"),
        read(&lm, "search/cut-0/observations.json")
    );
    let expected: Value =
        serde_json::from_slice(&read(&coordinate, "search/cut-0/refinements.json")).unwrap();
    let mut actual: Value =
        serde_json::from_slice(&read(&lm, "search/cut-0/refinements.json")).unwrap();
    assert!(!expected.as_array().unwrap().is_empty());
    let mut stages = 0;
    for record in actual.as_array_mut().unwrap() {
        let charged = record["refinement"]["evaluations"].as_u64().unwrap();
        let trace = record["refinement"]
            .as_object_mut()
            .unwrap()
            .remove("trace");
        assert_eq!(trace.is_some(), charged > 0);
        for stage in trace.iter().flat_map(|t| t["stages"].as_array().unwrap()) {
            assert_eq!(stage["kind"], "coordinate", "{stage}");
            assert_eq!(
                stage["reason"],
                crate::kernel2::refinement::SMALL_ALLOWANCE,
                "{stage}"
            );
            stages += 1;
        }
    }
    assert!(stages > 0);
    assert_eq!(actual, expected);
    // The summaries differ only by the selection and the refinements digest.
    let summary = |opt: &run::Options| {
        let mut value: Value =
            serde_json::from_slice(&read(opt, "search/cut-0/search.json")).unwrap();
        let fields = value.as_object_mut().unwrap();
        fields.remove("refinements_sha256");
        let selection = fields.remove("engine_selection");
        (value, selection.is_some())
    };
    assert_eq!(summary(&coordinate), (summary(&lm).0, false));
    assert!(summary(&lm).1);
    assert_eq!(
        (
            plain.search.objective_trials,
            plain.search.rejected_trials,
            plain.search.completed_evaluations
        ),
        (
            traced.search.objective_trials,
            traced.search.rejected_trials,
            traced.search.completed_evaluations
        )
    );
}

use super::*;

/// The run's CLI flags, parsed as `jope pipeline run` parses them.

pub(super) fn flags(extra: &[&str]) -> (crate::cli::SearchArgs, crate::kernel2::cli::ProfileArgs) {
    let mut argv = vec![
        "jope-pipeline",
        "run",
        "--dataset",
        "unresolved",
        "--output",
        "unused",
    ];
    argv.extend_from_slice(extra);
    let Cmd::Run {
        search, profile, ..
    } = Cli::try_parse_from(argv).unwrap().cmd
    else {
        panic!("run expected");
    };
    (search, profile)
}

/// Rewrites a run's move records (when given) and its selection record, rehashing search.json
/// and the receipt, so that only the move checks can refuse the change.
pub(super) fn rewrite(
    output: &Path,
    receipt: &Receipt,
    moves: Option<&[u8]>,
    engine: Option<EngineRecord>,
) -> PathBuf {
    if let Some(bytes) = moves {
        std::fs::write(output.join(PATH), bytes).unwrap();
    }
    let mut edited = receipt.clone();
    edited.search.engine_selection = engine;
    let summary = serde_json::to_vec_pretty(&edited.search).unwrap();
    std::fs::write(output.join("search/cut-0/search.json"), &summary).unwrap();
    edited.search_sha256 = crate::sha256_hex(&summary);
    let path = output.join("receipt.json");
    std::fs::write(&path, serde_json::to_vec(&edited).unwrap()).unwrap();
    path
}

pub(super) fn digested(engine: &EngineRecord, bytes: &[u8]) -> EngineRecord {
    let mut engine = engine.clone();
    engine.moves.as_mut().unwrap().sha256 = Some(crate::sha256_hex(bytes));
    engine
}

/// One named edit of the records at `measured` (an offered, measured line) and `refused` (a
/// refused line); `other` is another measured program's sha256.
pub(super) fn tamper(moves: &mut Vec<MoveRecord>, measured: usize, refused: usize, label: &str, other: &str) {
    fn fit(m: &mut MoveRecord) -> &mut FitRecord {
        m.linearization.as_mut().unwrap().fit.as_mut().unwrap()
    }
    match label {
        "record removed" => {
            moves.remove(measured);
            return;
        }
        "refused record removed" => {
            moves.remove(refused);
            return;
        }
        "refused offered" => {
            moves[refused].linearization.as_mut().unwrap().refusal = None;
            return;
        }
        "refused outcome" => {
            moves[refused].outcome = UNCHANGED.into();
            return;
        }
        "refused reason" => {
            moves[refused].linearization.as_mut().unwrap().refusal =
                Some(LINEARIZE_UNSAMPLED.into());
            return;
        }
        _ => {}
    }
    let m = &mut moves[measured];
    match label {
        "child program" => m.program = Some(ONE_TIMES.into()),
        "stage expression" => m.linearization.as_mut().unwrap().expression = Some("c0".into()),
        "path" => m.linearization.as_mut().unwrap().path = vec![0],
        "stage" => m.linearization.as_mut().unwrap().stage += 1,
        "subtree" => m.linearization.as_mut().unwrap().subtree = "(c0*3.0)".into(),
        "nodes" => m.linearization.as_mut().unwrap().nodes += 1,
        "degree" => m.linearization.as_mut().unwrap().degree = 2,
        "line nodes" => m.linearization.as_mut().unwrap().line_nodes = Some(3),
        "bound" => {
            let record = m.linearization.as_mut().unwrap();
            record.bound = record.bound.map(|b| b * 2.0);
        }
        "lowered error over the bound" => {
            let record = m.linearization.as_mut().unwrap();
            record.lowered_error = record.bound.map(|b| b * 2.0);
        }
        "refusal added" => {
            m.linearization.as_mut().unwrap().refusal = Some(LINEARIZE_MISSES_BOUND.into())
        }
        "fit removed" => m.linearization.as_mut().unwrap().fit = None,
        "fit stop opened" => fit(m).stop = "stalled".into(),
        "fit stop unknown" => fit(m).stop = "bisection".into(),
        "fit slope" => {
            let c = fit(m).coefficients.as_mut().unwrap();
            c[1] *= 1.5;
        }
        "fit intercept" => {
            let c = fit(m).coefficients.as_mut().unwrap();
            c[0] += 0.25;
        }
        "fit max error" => fit(m).max_error *= 2.0,
        "fit inverted bracket" => {
            let f = fit(m);
            f.levelled = f.max_error * 2.0;
        }
        "fit steps over the cap" => fit(m).steps = 33,
        "linearization removed" => m.linearization = None,
        "rewrites added" => m.rewrites = moves_rewrite(),
        "outcome" => {
            m.outcome = DUPLICATE.into();
            m.measurement = None;
        }
        "measured line claimed unfunded" => {
            m.outcome = NO_ALLOWANCE.into();
            m.measurement = None;
        }
        "measurement index" => m.measurement = Some(0),
        "round" => m.round += 1,
        "parent" => m.parent_sha256 = other.into(),
        "method id" => m.id = "feat-math.minimax-linearize.v2".into(),
        "rewrite id" => m.id = REWRITE_SATURATE.into(),
        _ => unreachable!("{label}"),
    }
}

/// A stage rewrite record, to attach where none belongs.
pub(super) fn moves_rewrite() -> Vec<StageRewrite> {
    vec![StageRewrite {
        stage: 0,
        stop: "saturated".into(),
        iterations: 1,
        enodes: 1,
        applications: 0,
        peak: 1,
        expression: None,
        refusal: None,
    }]
}

#[test]
fn verify_refuses_every_tampered_linearization_record_digest_or_selection() {
    let temp = tempfile::tempdir().unwrap();
    let opt = linearized_run(temp.path());
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
    let path = output.join("receipt.json");
    let originals: Vec<(PathBuf, Vec<u8>)> = ["receipt.json", "search/cut-0/search.json", PATH]
        .iter()
        .map(|r| (output.join(r), std::fs::read(output.join(r)).unwrap()))
        .collect();
    let restore = || {
        for (file, bytes) in &originals {
            std::fs::write(file, bytes).unwrap();
        }
    };
    let engine = receipt.search.engine_selection.clone().unwrap();
    let moves = published_moves(&opt);
    // The rewrite itself keeps a valid run.
    rewrite(
        output,
        &receipt,
        Some(originals[2].1.as_slice()),
        Some(engine.clone()),
    );
    publication::verify_run(&path).unwrap();
    restore();

    let measured = moves
        .iter()
        .position(|m| m.id == MINIMAX_LINEARIZE && m.outcome == MEASURED)
        .unwrap();
    let refused = moves
        .iter()
        .position(|m| {
            m.id == MINIMAX_LINEARIZE
                && m.outcome == REFUSED
                && line(m).fit.is_some()
                && line(m).refusal.is_some()
        })
        .unwrap();
    // The bootstrap program: measured, with no derive stage to linearize.
    let other = crate::sha256_hex(crate::search::BOOTSTRAP_GENOME.as_bytes());
    assert!(published_measurements(&opt)
        .iter()
        .any(|m| m.observation.program_sha256 == other
            && m.observation.status == crate::search::MEASURED));
    let mut errors = Vec::new();
    for label in [
        "child program",
        "stage expression",
        "path",
        "stage",
        "subtree",
        "nodes",
        "degree",
        "line nodes",
        "bound",
        "lowered error over the bound",
        "refusal added",
        "fit removed",
        "fit stop opened",
        "fit stop unknown",
        "fit slope",
        "fit intercept",
        "fit max error",
        "fit inverted bracket",
        "fit steps over the cap",
        "linearization removed",
        "rewrites added",
        "outcome",
        "measured line claimed unfunded",
        "measurement index",
        "round",
        "parent",
        "record removed",
        "method id",
        "rewrite id",
        "refused offered",
        "refused outcome",
        "refused reason",
        "refused record removed",
    ] {
        let mut edited = moves.clone();
        tamper(&mut edited, measured, refused, label, &other);
        assert_ne!(edited, moves, "{label}: the edit was not applied");
        let bytes = serde_json::to_vec_pretty(&edited).unwrap();
        let path = rewrite(
            output,
            &receipt,
            Some(bytes.as_slice()),
            Some(digested(&engine, &bytes)),
        );
        let error = publication::verify_run(&path).map(|_| ()).expect_err(label);
        errors.push(format!("{label}: {error:#}"));
        restore();
    }
    // Record edits, the file unchanged.
    let mut records: Vec<(&str, Option<EngineRecord>)> = Vec::new();
    let mut record = engine.clone();
    record.moves.as_mut().unwrap().sha256 = Some("0".repeat(64));
    records.push(("digest", Some(record)));
    let mut record = engine.clone();
    record.moves.as_mut().unwrap().ids = vec!["feat-math.minimax-linearize.v2".into()];
    records.push(("record id", Some(record)));
    let mut record = engine.clone();
    record.moves.as_mut().unwrap().ids = vec![REWRITE_SATURATE.into()];
    records.push(("the other pass's id", Some(record)));
    let mut record = engine.clone();
    record
        .moves
        .as_mut()
        .unwrap()
        .ids
        .push(MINIMAX_LINEARIZE.into());
    records.push(("repeated id", Some(record)));
    let mut record = engine.clone();
    record.moves.as_mut().unwrap().ids = vec![MINIMAX_LINEARIZE.into(), REWRITE_SATURATE.into()];
    records.push(("both ids, reversed", Some(record)));
    records.push(("record dropped, file kept", None));
    for (label, record) in records {
        assert_ne!(record.as_ref(), Some(&engine), "{label}");
        let path = rewrite(output, &receipt, None, record);
        let error = publication::verify_run(&path).map(|_| ()).expect_err(label);
        errors.push(format!("{label}: {error:#}"));
        restore();
    }
    publication::verify_run(&path).unwrap();
    // The refusals pass the shipped-text lint, with the run's random temporary path left out.
    let root = output.display().to_string();
    for error in errors {
        lint(&error.replace(&root, "<run>"));
    }
}

#[test]
fn the_selection_record_round_trips_and_the_flag_is_checked_and_lint_clean() {
    let record = linearize().record().unwrap();
    assert_eq!(
        serde_json::to_value(&record).unwrap(),
        serde_json::json!({"moves": {"ids": [MINIMAX_LINEARIZE]}})
    );
    assert_eq!(record.selection().unwrap(), linearize());
    let both = EngineSelection {
        simplify: true,
        linearize: true,
        ..EngineSelection::default()
    };
    assert_eq!(
        serde_json::to_value(both.record().unwrap()).unwrap(),
        serde_json::json!({"moves": {"ids": [REWRITE_SATURATE, MINIMAX_LINEARIZE]}})
    );
    assert_eq!(both.record().unwrap().selection().unwrap(), both);
    let all = EngineSelection {
        refine: RefineMethod::Lm(LmOptions::default()),
        archive: Some(ArchivePolicy::ArchiveOnly),
        simplify: true,
        linearize: true,
    };
    assert_eq!(all.record().unwrap().selection().unwrap(), all);
    for bad in [
        serde_json::json!({"moves": {"ids": [MINIMAX_LINEARIZE, REWRITE_SATURATE]}}),
        serde_json::json!({"moves": {"ids": [MINIMAX_LINEARIZE, MINIMAX_LINEARIZE]}}),
        serde_json::json!({"moves": {"ids": ["feat-math.minimax-linearize.v2"]}}),
    ] {
        let parsed: EngineRecord = serde_json::from_value(bad.clone()).unwrap();
        assert!(parsed.selection().is_err(), "{bad}");
    }
    assert!(serde_json::from_value::<MovesRecord>(
        serde_json::json!({"ids": [MINIMAX_LINEARIZE], "degree": 2})
    )
    .is_err());
    // The pass reads `--refine-top-k` parents, so it needs at least one.
    let none = Refinement {
        top_k: 0,
        trials_per_program: 4,
    };
    assert!(linearize().validate(none).is_err());
    assert!(linearize().validate(four_parents()).is_ok());
    let e = evaluator().with_engine(linearize()).unwrap();
    assert_eq!(e.engine(), &linearize());
    assert_eq!(e.engine_record(), linearize().record());
    // The CLI: kernel2 only, and refused without a parent.
    let temp = tempfile::tempdir().unwrap();
    let source = "a".repeat(40);
    let parse = |top_k: &str| {
        let (mut search, profile) = flags(&[
            "--search-profile",
            "kernel2-baseline",
            "--refine-top-k",
            top_k,
            "--refine-trials",
            "4",
            "--linearize-subtrees",
            "--source-commit",
            source.as_str(),
        ]);
        assert!(profile.linearize_subtrees && !profile.rewrite_simplify);
        search.dataset = dataset();
        (search, profile)
    };
    let (search, profile) = parse("1");
    assert!(profile
        .options(&search, temp.path().join("one"), false)
        .unwrap()
        .is_some());
    let (search, profile) = parse("0");
    let error = profile
        .options(&search, temp.path().join("zero"), false)
        .unwrap_err();
    assert!(format!("{error:#}").contains("--refine-top-k"), "{error:#}");
    lint(&format!("{error:#}"));
    let (mut legacy_search, legacy) = flags(&["--linearize-subtrees"]);
    legacy_search.dataset = dataset();
    let output = temp.path().join("refused");
    let error = legacy
        .options(&legacy_search, output.clone(), false)
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("legacy search refuses kernel2"),
        "{error:#}"
    );
    assert!(!output.exists());
    // Shipped text: the id, the path, the outcomes, the origin, every refusal, the flag's help
    // and the internal note. None calls the move a compression.
    for text in [
        MINIMAX_LINEARIZE,
        PATH,
        REFUSED,
        DUPLICATE,
        NOT_STARTED,
        NO_ALLOWANCE,
        MEASURED,
        crate::kernel2::LINEARIZE,
        PREFIX_FAULTS,
        LINEARIZE_UNSAMPLED,
        LINEARIZE_FIT_REFUSED,
        LINEARIZE_OPEN_BRACKET,
        LINEARIZE_NON_FINITE,
        LINEARIZE_NOT_SMALLER,
        LINEARIZE_OVER_BUDGET,
        LINEARIZE_OVER_LIMITS,
        LINEARIZE_OUT_OF_RANGE,
        LINEARIZE_MISSES_BOUND,
        LINEARIZE_PROFILE_REFUSED,
    ] {
        lint(text);
        assert!(!text.to_lowercase().contains("compress"), "{text}");
        assert!(!text.to_lowercase().contains("remez"), "{text}");
    }
    let note =
        include_str!("../../../../../../../docs/internal/kernel2/b19-minimax-linearization.md");
    lint(note);
    assert!(note.contains("bounded minimax linearization"));
    use clap::CommandFactory;
    let mut command = Cli::command();
    command.build();
    let run = command.find_subcommand_mut("run").unwrap();
    let flag = run
        .get_arguments()
        .find(|a| a.get_id() == "linearize_subtrees")
        .unwrap();
    let help = flag.get_help().unwrap().to_string();
    assert!(
        help.contains("bounded minimax linearization") && help.contains("moves.json"),
        "{help}"
    );
    assert!(!help.to_lowercase().contains("compress") && !help.to_lowercase().contains("remez"));
    lint(&help);
    if let Some(long) = flag.get_long_help() {
        lint(&long.to_string());
    }
}

// Fix round 01: recorded fits past the evaluator's range or hidden behind a refusal, whole parent
// groups moved, and the limits of what verification can see without the training rows.

/// The run's CLI flags, parsed as `jope pipeline run` parses them.
use super::*;

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

/// One named edit of the measured record at `first`; `other` is another measured program's
/// sha256.
pub(super) fn tamper(moves: &mut Vec<MoveRecord>, first: usize, label: &str, other: &str) {
    if label == "record removed" {
        moves.remove(first);
        return;
    }
    let m = &mut moves[first];
    match label {
        "child program" => m.program = Some(SINE.into()),
        "stage expression" => m.rewrites[0].expression = Some("c1".into()),
        "stop reason" => m.rewrites[0].stop = "node_limit".into(),
        "iterations" => m.rewrites[0].iterations += 1,
        "enodes" => m.rewrites[0].enodes += 1,
        "applications" => m.rewrites[0].applications += 1,
        "peak" => m.rewrites[0].peak += 1,
        "refusal" => m.rewrites[0].refusal = Some("refused".into()),
        "outcome" => {
            m.outcome = DUPLICATE.into();
            m.measurement = None;
        }
        "measurement index" => m.measurement = Some(0),
        "round" => m.round += 1,
        "parent" => m.parent_sha256 = other.into(),
        "method id" => m.id = "feat-math.rewrite-saturate.v2".into(),
        _ => unreachable!("{label}"),
    }
}

#[test]
fn verify_refuses_every_tampered_move_record_digest_or_selection() {
    let temp = tempfile::tempdir().unwrap();
    let opt = run_options(temp.path(), 40, four_parents(), simplify());
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

    let first = moves.iter().position(|m| m.outcome == MEASURED).unwrap();
    // The bootstrap program: measured, with no expression stage to rewrite.
    let other = crate::sha256_hex(crate::search::BOOTSTRAP_GENOME.as_bytes());
    assert!(published_measurements(&opt)
        .iter()
        .any(|m| m.observation.program_sha256 == other
            && m.observation.status == crate::search::MEASURED));
    let mut errors = Vec::new();
    for label in [
        "child program",
        "stage expression",
        "stop reason",
        "iterations",
        "enodes",
        "applications",
        "peak",
        "refusal",
        "outcome",
        "measurement index",
        "round",
        "parent",
        "record removed",
        "method id",
    ] {
        let mut edited = moves.clone();
        tamper(&mut edited, first, label, &other);
        assert_ne!(edited, moves, "{label}: the edit was not applied");
        let bytes = serde_json::to_vec_pretty(&edited).unwrap();
        let path = rewrite(
            output,
            &receipt,
            Some(bytes.as_slice()),
            Some(digested(&engine, &bytes)),
        );
        let error = publication::verify_run(&path).unwrap_err();
        errors.push(format!("{label}: {error:#}"));
        restore();
    }
    // Record edits, the file unchanged.
    let mut records: Vec<(&str, Option<EngineRecord>)> = Vec::new();
    let mut record = engine.clone();
    record.moves.as_mut().unwrap().sha256 = Some("0".repeat(64));
    records.push(("digest", Some(record)));
    let mut record = engine.clone();
    record.moves.as_mut().unwrap().sha256 = None;
    records.push(("digest dropped", Some(record)));
    let mut record = engine.clone();
    record.moves.as_mut().unwrap().sha256 = Some("XYZ".into());
    records.push(("malformed digest", Some(record)));
    let mut record = engine.clone();
    record.moves.as_mut().unwrap().ids = vec!["feat-math.rewrite-saturate.v2".into()];
    records.push(("record id", Some(record)));
    let mut record = engine.clone();
    record.moves.as_mut().unwrap().ids.clear();
    records.push(("no ids", Some(record)));
    let mut record = engine.clone();
    record
        .moves
        .as_mut()
        .unwrap()
        .ids
        .push(REWRITE_SATURATE.into());
    records.push(("repeated id", Some(record)));
    records.push(("record dropped, file kept", None));
    for (label, record) in records {
        assert_ne!(record.as_ref(), Some(&engine), "{label}");
        let path = rewrite(output, &receipt, None, record);
        let error = publication::verify_run(&path).unwrap_err();
        errors.push(format!("{label}: {error:#}"));
        restore();
    }
    // The file edited under its unchanged digest, then removed.
    let mut bytes = originals[2].1.clone();
    bytes.push(b'\n');
    std::fs::write(output.join(PATH), &bytes).unwrap();
    errors.push(format!("{:#}", publication::verify_run(&path).unwrap_err()));
    restore();
    std::fs::remove_file(output.join(PATH)).unwrap();
    errors.push(format!("{:#}", publication::verify_run(&path).unwrap_err()));
    restore();
    publication::verify_run(&path).unwrap();
    // The refusals pass the shipped-text lint, with the run's random temporary path left out.
    let root = output.display().to_string();
    for error in errors {
        lint(&error.replace(&root, "<run>"));
    }
}

#[test]
fn the_selection_record_round_trips_and_the_flag_is_checked_and_lint_clean() {
    let record = simplify().record().unwrap();
    assert_eq!(
        serde_json::to_value(&record).unwrap(),
        serde_json::json!({"moves": {"ids": [REWRITE_SATURATE]}})
    );
    assert_eq!(record.selection().unwrap(), simplify());
    let mut published = record.clone();
    published.moves.as_mut().unwrap().sha256 = Some("a".repeat(64));
    assert_eq!(published.selection().unwrap(), simplify());
    let all = EngineSelection {
        refine: RefineMethod::Lm(LmOptions::default()),
        archive: Some(ArchivePolicy::ArchiveOnly),
        simplify: true,
        linearize: false,
    };
    let record = all.record().unwrap();
    assert!(record.damped_ls().is_some() && record.archive.is_some() && record.moves.is_some());
    assert_eq!(record.selection().unwrap(), all);
    for bad in [
        serde_json::json!({"moves": {"ids": ["other"]}}),
        serde_json::json!({"moves": {"ids": []}}),
        serde_json::json!({"moves": {"ids": [REWRITE_SATURATE, REWRITE_SATURATE]}}),
    ] {
        let parsed: EngineRecord = serde_json::from_value(bad.clone()).unwrap();
        assert!(parsed.selection().is_err(), "{bad}");
    }
    assert!(serde_json::from_value::<MovesRecord>(
        serde_json::json!({"ids": [REWRITE_SATURATE], "rounds": 3})
    )
    .is_err());
    // The pass reads `--refine-top-k` parents, so it needs at least one.
    let none = Refinement {
        top_k: 0,
        trials_per_program: 4,
    };
    assert!(simplify().validate(none).is_err());
    assert!(simplify().validate(four_parents()).is_ok());
    let e = evaluator().with_engine(simplify()).unwrap();
    assert_eq!(e.engine(), &simplify());
    assert_eq!(e.engine_record(), simplify().record());
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
            "--rewrite-simplify",
            "--source-commit",
            source.as_str(),
        ]);
        assert!(profile.rewrite_simplify);
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
    let (mut legacy_search, legacy) = flags(&["--rewrite-simplify"]);
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
    // Shipped text: the id, the path, the outcomes, the flag's help and the internal note.
    for text in [
        REWRITE_SATURATE,
        PATH,
        UNCHANGED,
        REFUSED,
        DUPLICATE,
        NOT_STARTED,
        MEASURED,
        crate::kernel2::REWRITE,
    ] {
        lint(text);
    }
    lint(include_str!(
        "../../../../../../../docs/internal/kernel2/b20-rewrite-simplify.md"
    ));
    use clap::CommandFactory;
    let mut command = Cli::command();
    command.build();
    let run = command.find_subcommand_mut("run").unwrap();
    let flag = run
        .get_arguments()
        .find(|a| a.get_id() == "rewrite_simplify")
        .unwrap();
    let help = flag.get_help().unwrap().to_string();
    assert!(help.contains("moves.json"), "{help}");
    lint(&help);
    if let Some(long) = flag.get_long_help() {
        lint(&long.to_string());
    }
}

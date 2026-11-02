use super::*;

/// The run's CLI flags, parsed as `jope pipeline run` parses them.

fn flags(extra: &[&str]) -> (crate::cli::SearchArgs, ProfileArgs) {
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

#[test]
fn the_flag_selects_the_archive_on_kernel2_only_and_its_text_passes_the_lint() {
    let temp = tempfile::tempdir().unwrap();
    let source = "a".repeat(40);
    let kernel2 = [
        "--search-profile",
        "kernel2-baseline",
        "--refine-top-k",
        "1",
        "--refine-trials",
        "4",
        "--budget",
        "evaluations=16,seconds=300",
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
        let mut argv = kernel2.to_vec();
        argv.extend_from_slice(extra);
        let (mut search, profile) = flags(&argv);
        search.dataset = dataset();
        let mut opt = profile.options(&search, output, false).unwrap().unwrap();
        opt.run.clock = Clock::Frozen;
        opt
    };
    let plain = make(&[], temp.path().join("plain"));
    let flagged = make(&["--pareto-archive"], temp.path().join("flagged"));
    let (a, b) = (run::run(&plain).unwrap(), run::run(&flagged).unwrap());
    assert!(a.search.engine_selection.is_none());
    assert!(!plain.run.output.join(archive::PATH).exists());
    let record = b.search.engine_selection.as_ref().unwrap();
    assert!(record.refine.is_none() && record.archive.is_some());
    assert_eq!(
        publication::verify_run(&flagged.run.output.join("receipt.json")).unwrap(),
        b
    );
    assert_eq!(a.search.best, b.search.best);
    // With the method too.
    make(
        &["--pareto-archive", "--refine-method", "lm"],
        temp.path().join("lm"),
    );
    // The combination runs: both records published, and the run verifies.
    let combined = make(
        &["--pareto-archive", "--refine-method", "lm"],
        temp.path().join("lm-run"),
    );
    let receipt = run::run(&combined).unwrap();
    let both = receipt.search.engine_selection.as_ref().unwrap();
    assert!(both.damped_ls().is_some());
    assert_eq!(
        both.archive.as_ref().unwrap().sha256.as_deref(),
        Some(crate::sha256_hex(&read(&combined, archive::PATH)).as_str())
    );
    assert!(receipt.search.objective_trials > 0);
    assert_eq!(
        publication::verify_run(&combined.run.output.join("receipt.json")).unwrap(),
        receipt
    );
    check_front(
        &published(&combined),
        &derived(&combined, FaultPolicy::RejectHard),
    );
    // The legacy profile refuses the flag before anything is written.
    let output = temp.path().join("refused");
    let (mut search, profile) = flags(&["--pareto-archive"]);
    search.dataset = dataset();
    let error = profile.options(&search, output.clone(), false).unwrap_err();
    assert!(
        format!("{error:#}").contains("legacy search refuses kernel2"),
        "{error:#}"
    );
    assert!(!output.exists());
    // Shipped text: ids, the cost version, the record, the archive and the flag's help.
    let lint = |text: &str| {
        let hits = feat_math::transpile::lint_shipped_text(text);
        assert!(hits.is_empty(), "{text}: {hits:?}");
    };
    lint(FRONT_LOSS_COST);
    lint(pareto::PIPELINE_COST_VERSION);
    lint(archive::PATH);
    lint(&serde_json::to_string(record).unwrap());
    lint(&String::from_utf8(read(&flagged, archive::PATH)).unwrap());
    // The internal note that documents the flag, the ids and the record.
    lint(include_str!(
        "../../../../docs/internal/kernel2/b18-pareto-archive.md"
    ));
    use clap::CommandFactory;
    let mut command = Cli::command();
    command.build();
    let run = command.find_subcommand_mut("run").unwrap();
    let flag = run
        .get_arguments()
        .find(|a| a.get_id() == "pareto_archive")
        .unwrap();
    let help = flag.get_help().unwrap().to_string();
    assert!(help.contains("archive.json"), "{help}");
    lint(&help);
    if let Some(long) = flag.get_long_help() {
        lint(&long.to_string());
    }
    let record_check = receipt::validate_value(&{
        let mut value = serde_json::to_value(&b).unwrap();
        value["search"]["engine_selection"]["archive"]["sha256"] = "XYZ".into();
        value
    })
    .unwrap_err();
    lint(&format!("{record_check:#}"));
}

// Fix round 01: the selection off the publish path, the key's spellings, and the legacy refusal
// through the CLI's own dispatcher.

#[test]
fn run_cut_records_the_archive_selection_and_its_search_is_unchanged() {
    use super::search::{self, CutSearch};
    let cut = data();
    let make = |archive: bool| {
        let mut options = crate::search::SearchOptions::new(
            Budget {
                evaluations: 24,
                seconds: 120,
            },
            7,
        );
        options.clock = Clock::Frozen;
        let options = search::Options::new(
            options,
            "kernel2-baseline",
            FaultPolicy::RejectHard,
            Refinement {
                top_k: 1,
                trials_per_program: 4,
            },
        )
        .unwrap();
        if archive {
            options
                .with_engine(selection(RefineMethod::Coordinate, true))
                .unwrap()
        } else {
            options
        }
    };
    let plain = search::run_cut(&cut, &make(false)).unwrap();
    let selected = search::run_cut(&cut, &make(true)).unwrap();
    // The selection is recorded as selected, with no digest: nothing was built or written.
    assert!(plain.engine.is_none());
    assert_eq!(
        selected.engine,
        selection(RefineMethod::Coordinate, true).record()
    );
    assert!(selected
        .engine
        .as_ref()
        .and_then(|e| e.archive.as_ref())
        .is_some_and(|a| a.sha256.is_none()));
    // Every other field of the search output is the same, bit for bit.
    let without_engine = |c: &CutSearch| {
        let mut value = serde_json::to_value(c).unwrap();
        value.as_object_mut().unwrap().remove("engine");
        value
    };
    assert_eq!(without_engine(&plain), without_engine(&selected));
    assert_eq!(plain.observations, selected.observations);
    assert_eq!(plain.best, selected.best);
    // A caller that runs cuts itself builds the front from the observations, as publication does.
    assert_eq!(
        archive::build(&selected.observations).unwrap(),
        archive::build(&plain.observations).unwrap()
    );
}

#[test]
fn the_key_is_the_codec_bytes_so_expression_spellings_are_distinct_keys() {
    // The codec keeps an expression's text as written.
    let tight = "(pipe X (derive expr=\"1*c0\") (ridge))";
    let spaced = "(pipe X (derive expr=\"1 * c0\") (ridge))";
    assert_ne!(key(tight), key(spaced));
    // It keeps a number parameter as its value.
    assert_eq!(
        key("(pipe X (ridge lambda_frac=0.000000001))"),
        key("(pipe X (ridge lambda_frac=1e-9))")
    );
}

#[test]
fn the_legacy_dispatcher_refuses_each_kernel2_flag_before_writing() {
    // `ProfileArgs::options` writes nothing either way, so the flag tests' `!output.exists()`
    // after it cannot fail. The CLI's own dispatcher writes a run when it accepts one.
    for flag in [
        "--pareto-archive",
        "--rewrite-simplify",
        "--linearize-subtrees",
    ] {
        legacy_dispatch_refuses_before_writing(flag);
    }
}

/// The legacy profile refuses `flag` in the CLI's own dispatcher (`crate::cli::run`) before it
/// writes anything, while the same command without the flag runs there and writes its run, so
/// the check that nothing was written can fail.
fn legacy_dispatch_refuses_before_writing(flag: &str) {
    let temp = tempfile::tempdir().unwrap();
    let dataset = dataset();
    let source = "a".repeat(40);
    let argv = |output: &Path, extra: &[&str]| -> Vec<String> {
        let output = output.display().to_string();
        let mut argv: Vec<String> = [
            "jope-pipeline",
            "run",
            "--dataset",
            dataset.as_str(),
            "--output",
            output.as_str(),
            "--budget",
            "evaluations=8,seconds=300",
            "--policy",
            "skip",
            "--emit",
            "rust",
            "--rng-seed",
            "7",
            "--source-commit",
            source.as_str(),
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
        argv.extend(extra.iter().map(|s| (*s).to_owned()));
        argv
    };
    let refused = temp.path().join("refused");
    let error = crate::cli::run(Cli::try_parse_from(argv(&refused, &[flag])).unwrap()).unwrap_err();
    assert!(
        format!("{error:#}").contains("legacy search refuses kernel2"),
        "{flag}: {error:#}"
    );
    assert!(!refused.exists(), "{flag}");
    let ran = temp.path().join("ran");
    crate::cli::run(Cli::try_parse_from(argv(&ran, &[])).unwrap()).unwrap();
    assert!(ran.join("receipt.json").exists());
}

/// `verify_run` rebuilds the archive from published observations, which are untrusted, so each
/// archived program is parsed within the trace check's bounds (B17 review 02's class): the
/// pipeline form only, and each expression's nesting bytes counted before it is parsed. Unary plus
/// and redundant parentheses never reach the parsed tree, so a source at the limit costs what its
/// plain form costs; one byte more is refused unparsed. A closed-form program is refused by the
/// pre-parse check's own message, so the order is observed (B17-R03-PIPELINE-GUARD-COVERAGE).
#[test]
fn archived_programs_are_bounded_before_they_are_parsed() {
    use super::refinement::MAX_PARENT_EXPRESSION_NESTING;
    let e = evaluator();
    let plain = measured(&e, SEED_SINE, &data());
    assert!(archive::ranked(&plain));
    let body = "sin(0.5*c0)";
    let with = |source: &str| {
        let mut o = plain.clone();
        o.observation.genome = SEED_SINE.replace(body, source);
        o
    };
    let cost_of = |o: &Kernel2Observation| {
        let archive = archive::build(std::slice::from_ref(o)).unwrap();
        let members = archive.members();
        assert_eq!(members.len(), 1);
        members[0].complexity
    };
    assert_eq!(cost_of(&plain), 6);
    // `(` and `*`.
    let room = MAX_PARENT_EXPRESSION_NESTING - 2;
    let plus = |n: usize| format!("{}{body}", "+".repeat(n));
    let grouped = |n: usize| format!("{}{body}{}", "(".repeat(n), ")".repeat(n));
    let product = |n: usize| format!("{body}{}", "*1".repeat(n));
    for source in [plus(room), grouped(room)] {
        assert_eq!(cost_of(&with(&source)), 6, "{source}");
    }
    for source in [plus(room + 1), grouped(room + 1), product(room + 1)] {
        let error = archive::build(&[with(&source)]).unwrap_err();
        assert!(
            format!("{error:#}").contains("nests too deeply"),
            "{source}: {error:#}"
        );
    }
    let mut closed_form = plain.clone();
    closed_form.observation.genome = format!("(sym raw \"{}\")", plus(room + 1));
    let error = archive::build(&[closed_form]).unwrap_err();
    assert!(
        format!("{error:#}").contains("not in the pipeline form"),
        "{error:#}"
    );
}

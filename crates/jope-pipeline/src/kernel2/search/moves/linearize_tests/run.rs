use super::*;

fn dataset() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../datasets/test/pmlb-229-pwlinear")
        .display()
        .to_string()
}

/// The `cli_tests` run (pmlb-229-pwlinear, rng seed 7, a frozen clock, Rust only, `seeds` as the
/// seeds file beside the output) in `dir/run`. The default selection is left unapplied, as the
/// CLI does.
fn run_options(
    dir: &Path,
    seeds: &[String],
    evaluations: usize,
    refinement: Refinement,
    engine: EngineSelection,
) -> run::Options {
    run_options_on(dataset(), dir, seeds, evaluations, refinement, engine)
}

/// [`run_options`] on another dataset.
fn run_options_on(
    dataset: String,
    dir: &Path,
    seeds: &[String],
    evaluations: usize,
    refinement: Refinement,
    engine: EngineSelection,
) -> run::Options {
    std::fs::create_dir_all(dir).unwrap();
    let mut base = RunOptions::new(dataset, dir.join("run"));
    base.budget = Budget {
        evaluations,
        seconds: 300,
    };
    base.policy = Policy::Skip;
    base.languages = vec![Language::Rust];
    base.rng_seed = 7;
    base.clock = Clock::Frozen;
    base.source_commit = Some("a".repeat(40));
    let path = dir.join("seeds.json");
    let file = crate::seeds::SeedsFile::new(
        None,
        seeds
            .iter()
            .map(|genome| crate::seeds::SeedGenome {
                genome: genome.clone(),
                program_sha256: crate::sha256_hex(genome.as_bytes()),
                bank_entry_sha256: None,
                receipt_sha256: None,
            })
            .collect(),
    );
    std::fs::write(&path, serde_json::to_vec(&file).unwrap()).unwrap();
    base.seed_genomes = Some(path);
    let options = run::Options::new(
        base,
        "kernel2-baseline",
        FaultPolicy::RejectHard,
        refinement,
    )
    .unwrap();
    if engine == EngineSelection::default() {
        options
    } else {
        options.with_engine(engine).unwrap()
    }
}

fn four_parents() -> Refinement {
    Refinement {
        top_k: 4,
        trials_per_program: 4,
    }
}

fn read(opt: &run::Options, relative: &str) -> Vec<u8> {
    std::fs::read(opt.run.output.join(relative)).unwrap()
}

fn published_moves(opt: &run::Options) -> Vec<MoveRecord> {
    serde_json::from_slice(&read(opt, PATH)).unwrap()
}

fn published_measurements(opt: &run::Options) -> Vec<Measurement> {
    serde_json::from_slice(&read(opt, "search/cut-0/observations.json")).unwrap()
}

fn lint(text: &str) {
    let hits = feat_math::transpile::lint_shipped_text(text);
    assert!(hits.is_empty(), "{text}: {hits:?}");
}

/// The linearization run: both seeds, four parents per round, 40 evaluations.
fn linearized_run(dir: &Path) -> run::Options {
    run_options(
        dir,
        &[ONE_TIMES.into(), program(SINE)],
        40,
        four_parents(),
        linearize(),
    )
}

#[test]
fn linearization_runs_replay_every_file_with_a_frozen_clock_and_verify() {
    let temp = tempfile::tempdir().unwrap();
    let lm = RefineMethod::Lm(LmOptions::default());
    // Every run reads four parents per round, as in the rewrite pass's replay since #208
    // (52d0f6432) fixed publication's refinement child index. The damped least-squares run gets a
    // larger allowance, since each refinement spends up to 12 trials, to leave the passes
    // measurements.
    for (label, engine, refinement, evaluations, ids) in [
        (
            "linearize",
            linearize(),
            four_parents(),
            40,
            vec![MINIMAX_LINEARIZE],
        ),
        (
            "rewrite-linearize",
            EngineSelection {
                simplify: true,
                linearize: true,
                ..EngineSelection::default()
            },
            four_parents(),
            40,
            vec![REWRITE_SATURATE, MINIMAX_LINEARIZE],
        ),
        (
            "all",
            EngineSelection {
                refine: lm,
                archive: Some(ArchivePolicy::ArchiveOnly),
                simplify: true,
                linearize: true,
            },
            Refinement {
                top_k: 4,
                trials_per_program: 12,
            },
            120,
            vec![REWRITE_SATURATE, MINIMAX_LINEARIZE],
        ),
    ] {
        let make = |name: &str| {
            run_options(
                &temp.path().join(format!("{label}-{name}")),
                &[ONE_TIMES.into(), program(SINE)],
                evaluations,
                refinement,
                engine.clone(),
            )
        };
        let (a, b) = (make("a"), make("b"));
        let receipt = run::run(&a).unwrap();
        assert_eq!(receipt, run::run(&b).unwrap(), "{label}");
        for relative in SIDECARS.into_iter().chain([PATH]) {
            assert_eq!(read(&a, relative), read(&b, relative), "{label} {relative}");
        }
        if engine.archive.is_some() {
            assert_eq!(read(&a, archive::PATH), read(&b, archive::PATH), "{label}");
        }
        assert_eq!(
            publication::verify_run(&a.run.output.join("receipt.json")).unwrap(),
            receipt,
            "{label}"
        );
        // The pass measured lines, and each measured line is accounted for by one record.
        let moves = published_moves(&a);
        let measurements = published_measurements(&a);
        let linearized = |m: &&MoveRecord| m.id == MINIMAX_LINEARIZE && m.outcome == MEASURED;
        assert!(
            moves.iter().any(|m| linearized(&m)),
            "{label}: {:#?}",
            moves
                .iter()
                .map(|m| (
                    m.id.as_str(),
                    m.round,
                    m.outcome.as_str(),
                    &m.reason,
                    &m.linearization
                ))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            moves.iter().filter(linearized).count(),
            measurements
                .iter()
                .filter(|m| m.observation.origin == crate::kernel2::LINEARIZE)
                .count()
        );
        for m in moves.iter().filter(|m| m.id == MINIMAX_LINEARIZE) {
            let record = line(m);
            assert_eq!(record.degree, 1);
            if record.expression.is_some() {
                assert!(record.lowered_error.unwrap() <= record.bound.unwrap());
            }
        }
        let s = &receipt.search;
        assert_eq!(
            s.completed_evaluations,
            s.candidate_measurements + s.objective_trials
        );
        let record = s.engine_selection.as_ref().unwrap().moves.as_ref().unwrap();
        assert_eq!(record.ids, ids, "{label}");
        // With the rewrite pass, no stage of a measured parent failed to rewrite; without it, no
        // count.
        let rewrites = ids.contains(&REWRITE_SATURATE);
        assert_eq!(record.not_run, rewrites.then_some(0), "{label}");
        assert_eq!(
            record.sha256.as_deref(),
            Some(crate::sha256_hex(&read(&a, PATH)).as_str())
        );
        lint(&String::from_utf8(read(&a, PATH)).unwrap());
        lint(&serde_json::to_string(record).unwrap());
    }
}

/// The seeded `cli_tests` run's search sidecars (prepared, search, observations, refinements),
/// pinned by `default_runs_keep_the_pre_wiring_search_sidecars` at the commit before the first
/// engine method was wired, and repeated here for this flag.
const SEEDED_PINS: [&str; 4] = [
    "8b8dccfe93aff8fe789ebc37dd125145a28b9d9d702b42dea2fa737575008059",
    "b2444c5c34dee80c91aa52d19a4a34aeaa358bf68dd3919723f52463fb2c5420",
    "ad1e6ab3e218f237120668e39321daf2a60cc43b5a503011f1a05ecbbb907767",
    "a6e3af0c1a0b0e85bfe64e3d8bd5c869e6a58b890403ba3438efba144e12a4d7",
];

#[test]
fn with_the_linearization_flag_off_nothing_is_selected_called_or_written() {
    let temp = tempfile::tempdir().unwrap();
    let (_, profile) = flags(&[
        "--search-profile",
        "kernel2-baseline",
        "--refine-top-k",
        "1",
        "--refine-trials",
        "4",
    ]);
    assert!(!profile.linearize_subtrees);
    assert!(!EngineSelection::default().linearize);
    assert_eq!(EngineSelection::default().record(), None);
    // The seeded `cli_tests` run, without the flag, keeps its pinned bytes and writes nothing new.
    let seeded = run_options(
        &temp.path().join("seeded"),
        &[ONE_TIMES.into()],
        28,
        Refinement {
            top_k: 1,
            trials_per_program: 4,
        },
        EngineSelection::default(),
    );
    let receipt = run::run(&seeded).unwrap();
    for (relative, pin) in [
        "prepared/prepared.json",
        "search/cut-0/search.json",
        "search/cut-0/observations.json",
        "search/cut-0/refinements.json",
    ]
    .into_iter()
    .zip(SEEDED_PINS)
    {
        assert_eq!(
            crate::sha256_hex(&read(&seeded, relative)),
            pin,
            "{relative}"
        );
    }
    assert!(receipt.search.engine_selection.is_none());
    // The linearization run's seeds, without the flag.
    let plain = run_options(
        &temp.path().join("plain"),
        &[ONE_TIMES.into(), program(SINE)],
        40,
        four_parents(),
        EngineSelection::default(),
    );
    run::run(&plain).unwrap();
    for opt in [&seeded, &plain] {
        assert!(!opt.run.output.join(PATH).exists());
        for relative in ["receipt.json", "search/cut-0/search.json"] {
            let text = String::from_utf8(read(opt, relative)).unwrap();
            assert!(!text.contains("moves") && !text.contains("linearize"));
        }
        assert!(published_measurements(opt)
            .iter()
            .all(|m| m.observation.origin != crate::kernel2::LINEARIZE));
        publication::verify_run(&opt.run.output.join("receipt.json")).unwrap();
    }
    // With the flag on, every measurement before the first linearized program is the same as
    // off: the pass consumes no random stream, and nothing runs before it.
    let on = linearized_run(&temp.path().join("on"));
    run::run(&on).unwrap();
    let (off, with) = (published_measurements(&plain), published_measurements(&on));
    let first = with
        .iter()
        .position(|m| m.observation.origin == crate::kernel2::LINEARIZE)
        .unwrap();
    assert!(first > 0);
    assert_eq!(off[..first], with[..first]);
    assert_eq!(
        read(&plain, "prepared/prepared.json"),
        read(&on, "prepared/prepared.json")
    );
}

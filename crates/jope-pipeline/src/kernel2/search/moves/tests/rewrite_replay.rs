use super::cli_contract::flags;
use super::rewrite_scores::{SEED, SIDECARS};
use super::*;

pub(super) fn dataset() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../datasets/test/pmlb-229-pwlinear")
        .display()
        .to_string()
}

/// The `cli_tests` run (pmlb-229-pwlinear, rng seed 7, a frozen clock, Rust only, [`SEED`] as the
/// seeds file beside the output) in `dir/run`. The default selection is left unapplied, as the
/// CLI does.
pub(super) fn run_options(
    dir: &Path,
    evaluations: usize,
    refinement: Refinement,
    engine: EngineSelection,
) -> run::Options {
    std::fs::create_dir_all(dir).unwrap();
    let mut base = RunOptions::new(dataset(), dir.join("run"));
    base.budget = Budget {
        evaluations,
        seconds: 300,
    };
    base.policy = Policy::Skip;
    base.languages = vec![Language::Rust];
    base.rng_seed = 7;
    base.clock = Clock::Frozen;
    base.source_commit = Some("a".repeat(40));
    let seeds = dir.join("seeds.json");
    let file = crate::seeds::SeedsFile::new(
        None,
        vec![crate::seeds::SeedGenome {
            genome: SEED.into(),
            program_sha256: crate::sha256_hex(SEED.as_bytes()),
            bank_entry_sha256: None,
            receipt_sha256: None,
        }],
    );
    std::fs::write(&seeds, serde_json::to_vec(&file).unwrap()).unwrap();
    base.seed_genomes = Some(seeds);
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

pub(super) fn four_parents() -> Refinement {
    Refinement {
        top_k: 4,
        trials_per_program: 4,
    }
}

pub(super) fn read(opt: &run::Options, relative: &str) -> Vec<u8> {
    std::fs::read(opt.run.output.join(relative)).unwrap()
}

pub(super) fn published_moves(opt: &run::Options) -> Vec<MoveRecord> {
    serde_json::from_slice(&read(opt, PATH)).unwrap()
}

pub(super) fn published_measurements(opt: &run::Options) -> Vec<Measurement> {
    serde_json::from_slice(&read(opt, "search/cut-0/observations.json")).unwrap()
}

pub(super) fn lint(text: &str) {
    let hits = feat_math::transpile::lint_shipped_text(text);
    assert!(hits.is_empty(), "{text}: {hits:?}");
}

/// Every measured move's child scores bit for bit as its parent, in the published records.
pub(super) fn check_exact(opt: &run::Options) -> usize {
    let measurements = published_measurements(opt);
    let moves = published_moves(opt);
    let bits = |m: &Measurement| {
        (
            m.observation.train_r2.map(f64::to_bits),
            m.observation.hold_r2.map(f64::to_bits),
            m.observation.rmse.map(f64::to_bits),
            m.selection_score.map(f64::to_bits),
            m.faults.clone(),
        )
    };
    let mut count = 0;
    for m in moves.iter().filter(|m| m.outcome == MEASURED) {
        let child = &measurements[m.measurement.unwrap()];
        let parent = measurements
            .iter()
            .find(|p| {
                p.observation.program_sha256 == m.parent_sha256
                    && p.observation.status == crate::search::MEASURED
            })
            .unwrap();
        assert_eq!(child.observation.status, crate::search::MEASURED);
        assert_eq!(bits(child), bits(parent), "{}", child.observation.genome);
        count += 1;
    }
    assert_eq!(
        count,
        measurements
            .iter()
            .filter(|m| m.observation.origin == crate::kernel2::REWRITE)
            .count()
    );
    count
}

#[test]
fn rewrite_runs_replay_every_file_with_a_frozen_clock_and_verify() {
    let temp = tempfile::tempdir().unwrap();
    let lm = RefineMethod::Lm(LmOptions::default());
    // Both runs refine four parents per round. Before #208 (52d0f6432) the damped least-squares run
    // took one: with four parents, two refined to one program in one round on this fixture, and
    // `verify_run` refused that run, the pass on or off, because publication indexed each
    // refinement child by the first equal observation. The damped least-squares run gets a larger
    // allowance, since each refinement spends up to 12 trials, to leave the pass measurements.
    for (label, engine, refinement, evaluations) in [
        ("coordinate", simplify(), four_parents(), 40),
        (
            "lm-archive",
            EngineSelection {
                refine: lm,
                archive: Some(ArchivePolicy::ArchiveOnly),
                simplify: true,
                linearize: false,
            },
            Refinement {
                top_k: 4,
                trials_per_program: 12,
            },
            120,
        ),
    ] {
        let make = |name: &str| {
            run_options(
                &temp.path().join(format!("{label}-{name}")),
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
        // The pass measured children, each scoring exactly as its parent, and the shared
        // identity holds.
        assert!(
            check_exact(&a) > 0,
            "{label}: no child was measured: {:?}",
            published_moves(&a)
                .iter()
                .map(|m| (m.round, m.outcome.as_str()))
                .collect::<Vec<_>>()
        );
        let s = &receipt.search;
        assert_eq!(
            s.completed_evaluations,
            s.candidate_measurements + s.objective_trials
        );
        let record = s.engine_selection.as_ref().unwrap().moves.as_ref().unwrap();
        assert_eq!(record.ids, [REWRITE_SATURATE]);
        // No stage of a measured parent failed to rewrite.
        assert_eq!(record.not_run, Some(0), "{label}");
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
pub(super) const SEEDED_PINS: [&str; 4] = [
    "8b8dccfe93aff8fe789ebc37dd125145a28b9d9d702b42dea2fa737575008059",
    "b2444c5c34dee80c91aa52d19a4a34aeaa358bf68dd3919723f52463fb2c5420",
    "ad1e6ab3e218f237120668e39321daf2a60cc43b5a503011f1a05ecbbb907767",
    "a6e3af0c1a0b0e85bfe64e3d8bd5c869e6a58b890403ba3438efba144e12a4d7",
];

#[test]
fn with_the_flag_off_nothing_is_selected_called_or_written() {
    let temp = tempfile::tempdir().unwrap();
    let (_, profile) = flags(&[
        "--search-profile",
        "kernel2-baseline",
        "--refine-top-k",
        "1",
        "--refine-trials",
        "4",
    ]);
    assert!(!profile.rewrite_simplify);
    assert!(!EngineSelection::default().simplify);
    assert_eq!(EngineSelection::default().record(), None);
    // The seeded `cli_tests` run, without the flag, keeps its pinned bytes and writes nothing new.
    let seeded = run_options(
        &temp.path().join("seeded"),
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
    let plain = run_options(
        &temp.path().join("plain"),
        40,
        four_parents(),
        EngineSelection::default(),
    );
    run::run(&plain).unwrap();
    for opt in [&seeded, &plain] {
        assert!(!opt.run.output.join(PATH).exists());
        for relative in ["receipt.json", "search/cut-0/search.json"] {
            assert!(!String::from_utf8(read(opt, relative))
                .unwrap()
                .contains("moves"));
        }
        assert!(published_measurements(opt)
            .iter()
            .all(|m| m.observation.origin != crate::kernel2::REWRITE));
        publication::verify_run(&opt.run.output.join("receipt.json")).unwrap();
    }
    // With the flag on, every measurement before the first rewrite child is the same as off:
    // the pass consumes no random stream, and nothing runs before it.
    let on = run_options(&temp.path().join("on"), 40, four_parents(), simplify());
    run::run(&on).unwrap();
    let (off, with) = (published_measurements(&plain), published_measurements(&on));
    let first = with
        .iter()
        .position(|m| m.observation.origin == crate::kernel2::REWRITE)
        .unwrap();
    assert!(first > 0);
    assert_eq!(off[..first], with[..first]);
    assert_eq!(
        read(&plain, "prepared/prepared.json"),
        read(&on, "prepared/prepared.json")
    );
}

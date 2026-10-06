//! The exact rewrite pass at the end of each search round (laneD:B20-rewrite-simplify).
use super::super::{Refinement, State};
use super::*;
use crate::{
    cli::{Cli, Cmd},
    clock::Clock,
    emit::Language,
    kernel2::{
        archive, publication,
        receipt::Receipt,
        run,
        tests::{data, evaluator},
        ArchivePolicy, EngineSelection, Kernel2Observation, MovesRecord, RefineMethod,
    },
    run::{Policy, RunOptions},
    search::{Budget, SearchOptions},
};
use clap::Parser;
use feat_math::{evolve::FaultPolicy, kernel::expression_search::LmOptions};
use std::path::PathBuf;

/// `1*c0`, which the pass reduces to `c0`.
const ONE_TIMES: &str = "(pipe X (derive expr=\"1*c0\") (ridge))";
/// `c0*1.0`, the same program spelled the other way round.
const TIMES_ONE: &str = "(pipe X (derive expr=\"c0*1.0\") (ridge))";
/// A division by one and a foldable product, before a scaling stage.
const FOLDED: &str = "(pipe X (derive expr=\"(c0/1.0)+(2.0*3.0)\") (standardize) (ridge))";
/// An elementwise carrier.
const MAPPED: &str = "(pipe X (map expr=\"abs(abs(x))\") (ridge))";
/// `c0*(-1.0)` reduces to `-c0`, a negation of a variable the profile refuses.
const NEGATED: &str = "(pipe X (derive expr=\"c0*(-1.0)\") (ridge))";
/// A refused stage beside an admitted one.
const MIXED: &str = "(pipe X (derive expr=\"c0*(-1.0)\") (derive expr=\"1*c0\") (ridge))";
/// Already as small as its class.
const SINE: &str = "(pipe X (derive expr=\"sin(0.5*c0)\") (ridge))";

fn penalize() -> FaultPolicy {
    FaultPolicy::Penalize {
        lambda_domain: 0.05,
        lambda_saturation: 0.0,
        tie_break: true,
    }
}

fn simplify() -> EngineSelection {
    EngineSelection {
        simplify: true,
        ..EngineSelection::default()
    }
}

/// Search options with the pass selected, `top_k` parents per round and no refinement trials.
fn options(top_k: usize, evaluations: usize, policy: FaultPolicy) -> Options {
    let mut search = SearchOptions::new(
        Budget {
            evaluations,
            seconds: 120,
        },
        7,
    );
    search.clock = Clock::Frozen;
    Options::new(
        search,
        "kernel2-baseline",
        policy,
        Refinement {
            top_k,
            trials_per_program: 0,
        },
    )
    .unwrap()
    .with_engine(simplify())
    .unwrap()
}

/// A search state holding `genomes`, each measured on `cut` and charged.
fn parents(cut: &Cut, options: &Options, genomes: &[&str]) -> State {
    let mut state = State::new(options.search.budget.evaluations);
    for genome in genomes {
        let o = options
            .evaluator
            .measure(
                genome,
                cut,
                0,
                "random",
                options.search.clock,
                &mut state.budget,
            )
            .unwrap();
        assert!(o.observation.measured(), "{genome}: {:?}", o.observation);
        state.seen.insert(o.observation.genome.clone());
        state.observations.push(o);
    }
    state
}

/// `parent` with stage `index`'s expression replaced by `expression`, as the pass prints it.
fn with_expression(parent: &str, index: usize, expression: &str) -> String {
    let GenomeIr::Pipeline(mut pipe) = GenomeIr::parse(parent).unwrap() else {
        panic!("{parent} is not a pipeline");
    };
    let source = MathExpr::parse(expression).unwrap().to_source().unwrap();
    pipe.stages[index]
        .params
        .insert("expr".into(), ParamValue::Str(source));
    GenomeIr::Pipeline(pipe).canonical()
}

/// Everything a measurement scores, as bits.
type Scores = (
    Option<u64>,
    Option<u64>,
    Option<u64>,
    Option<u64>,
    u64,
    u64,
    u64,
);

fn scores(o: &Kernel2Observation) -> Scores {
    (
        o.observation.train_r2.map(f64::to_bits),
        o.observation.hold_r2.map(f64::to_bits),
        o.observation.rmse.map(f64::to_bits),
        o.selection_score.map(f64::to_bits),
        o.faults.evaluated_cells,
        o.faults.domain_cells,
        o.faults.saturation_cells,
    )
}

fn parent_of<'a>(state: &'a State, record: &MoveRecord) -> &'a Kernel2Observation {
    state
        .observations
        .iter()
        .find(|o| o.observation.program_sha256 == record.parent_sha256)
        .unwrap()
}

fn all_genomes_distinct(state: &State) -> bool {
    let genomes: BTreeSet<&str> = state
        .observations
        .iter()
        .map(|o| o.observation.genome.as_str())
        .collect();
    genomes.len() == state.observations.len()
}

#[test]
fn rewritten_children_score_bit_for_bit_as_their_parents_and_cost_one_each() {
    let cut = data();
    for policy in [FaultPolicy::RejectHard, penalize()] {
        let opt = options(4, 64, policy);
        let mut state = parents(&cut, &opt, &[ONE_TIMES, TIMES_ONE, FOLDED, MAPPED]);
        let before = state.budget.used();
        simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
        assert_eq!(state.moves.len(), 4, "{policy:?}");
        let measured: Vec<&MoveRecord> = state
            .moves
            .iter()
            .filter(|m| m.outcome == MEASURED)
            .collect();
        // `1*c0` and `c0*1.0` reduce to one program: the first ranked is measured, the other is a
        // duplicate and spends nothing.
        assert_eq!(measured.len(), 3, "{:?}", state.moves);
        let duplicate: Vec<&MoveRecord> = state
            .moves
            .iter()
            .filter(|m| m.outcome == DUPLICATE)
            .collect();
        assert_eq!(duplicate.len(), 1);
        assert!(duplicate[0].measurement.is_none());
        // One evaluation per measured child; the shared identity holds; no program twice.
        assert_eq!(state.budget.used() - before, measured.len());
        assert_eq!(state.budget.used(), state.observations.len());
        assert!(all_genomes_distinct(&state));
        let expected: BTreeSet<String> = [
            with_expression(ONE_TIMES, 0, "c0"),
            with_expression(FOLDED, 0, "c0+6.0"),
            with_expression(MAPPED, 0, "abs(x)"),
        ]
        .into_iter()
        .collect();
        let programs: BTreeSet<String> = measured
            .iter()
            .map(|m| m.program.clone().unwrap())
            .collect();
        assert_eq!(programs, expected);
        assert_eq!(
            duplicate[0].program,
            Some(with_expression(ONE_TIMES, 0, "c0"))
        );
        for m in &measured {
            let child = &state.observations[m.measurement.unwrap()];
            let parent = parent_of(&state, m);
            assert_eq!(child.observation.origin, crate::kernel2::REWRITE);
            assert_eq!(child.observation.generation, state.generation);
            assert_eq!(Some(&child.observation.genome), m.program.as_ref());
            assert!(child.observation.measured(), "{:?}", child.observation);
            assert_eq!(
                scores(child),
                scores(parent),
                "{} vs {}",
                child.observation.genome,
                parent.observation.genome
            );
            assert_eq!(m.rewrites.len(), 1);
            let rewrite = &m.rewrites[0];
            assert!(rewrite.expression.is_some() && rewrite.refusal.is_none());
            assert_eq!(rewrite.stop, "saturated");
            assert!(rewrite.iterations > 0 && rewrite.applications > 0);
            assert!(rewrite.enodes > 0 && rewrite.peak >= rewrite.enodes);
        }
    }
}

#[test]
fn refused_outputs_are_discarded_with_their_reason_and_admitted_stages_kept() {
    let cut = data();
    let opt = options(4, 64, FaultPolicy::RejectHard);
    let mut state = parents(&cut, &opt, &[NEGATED, MIXED, SINE]);
    let before = state.budget.used();
    simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
    let record = |genome: &str| {
        let sha = crate::sha256_hex(genome.as_bytes());
        state
            .moves
            .iter()
            .find(|m| m.parent_sha256 == sha)
            .unwrap()
            .clone()
    };
    // The whole proposal refused: nothing assembled, nothing charged, the reason kept.
    let negated = record(NEGATED);
    assert_eq!(negated.outcome, REFUSED);
    assert!(negated.program.is_none() && negated.measurement.is_none());
    assert!(negated.reason.is_none());
    let refusal = negated.rewrites[0].refusal.as_deref().unwrap();
    assert!(refusal.contains("negative constant"), "{refusal}");
    assert!(negated.rewrites[0].expression.is_none());
    // One stage refused, the other admitted: the refused stage keeps its expression.
    let mixed = record(MIXED);
    assert_eq!(mixed.outcome, MEASURED);
    assert!(mixed.rewrites[0].refusal.is_some() && mixed.rewrites[0].expression.is_none());
    assert!(mixed.rewrites[1].refusal.is_none() && mixed.rewrites[1].expression.is_some());
    assert_eq!(mixed.program, Some(with_expression(MIXED, 1, "c0")));
    let child = &state.observations[mixed.measurement.unwrap()];
    assert_eq!(scores(child), scores(parent_of(&state, &mixed)));
    // Nothing smaller: unchanged, with the saturation counters still recorded.
    let sine = record(SINE);
    assert_eq!(sine.outcome, UNCHANGED);
    assert!(sine.program.is_none() && sine.measurement.is_none());
    assert!(sine.rewrites[0].expression.is_none() && sine.rewrites[0].refusal.is_none());
    assert!(sine.rewrites[0].iterations > 0);
    assert_eq!(state.budget.used() - before, 1);
    assert_eq!(state.budget.used(), state.observations.len());
    // The refusal is the profile's own: the same program with `-c0` is refused by admission.
    let refused = with_expression(NEGATED, 0, "-c0");
    assert!(admit_child(&refused, opt.evaluator.profile(), 1).is_err());
}

#[test]
fn the_allowance_and_the_start_guard_bound_the_pass() {
    let cut = data();
    let genomes = [ONE_TIMES, FOLDED, MAPPED];
    // One evaluation left: one child, then the pass stops before the next proposal.
    let opt = options(4, genomes.len() + 1, FaultPolicy::RejectHard);
    let mut state = parents(&cut, &opt, &genomes);
    simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
    assert_eq!(state.moves.len(), 1);
    assert_eq!(state.moves[0].outcome, MEASURED);
    assert_eq!(state.budget.remaining(), 0);
    assert_eq!(state.budget.used(), state.observations.len());
    // None left: no proposal at all.
    let opt = options(4, genomes.len(), FaultPolicy::RejectHard);
    let mut state = parents(&cut, &opt, &genomes);
    simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
    assert!(state.moves.is_empty());
    // The guard refuses the first proposal: nothing is recorded or spent.
    let opt = options(4, 64, FaultPolicy::RejectHard);
    let mut state = parents(&cut, &opt, &genomes);
    let used = state.budget.used();
    simplify_round(&mut state, &cut, &opt, &mut || false).unwrap();
    assert!(state.moves.is_empty());
    assert_eq!(state.budget.used(), used);
    // The guard admits the proposal and refuses its measurement: recorded as not started, and
    // the pass ends.
    let mut state = parents(&cut, &opt, &genomes);
    let mut calls = 0;
    simplify_round(&mut state, &cut, &opt, &mut || {
        calls += 1;
        calls == 1
    })
    .unwrap();
    assert_eq!(state.moves.len(), 1);
    assert_eq!(state.moves[0].outcome, NOT_STARTED);
    assert!(state.moves[0].program.is_some() && state.moves[0].measurement.is_none());
    assert_eq!(state.budget.used(), used);
    // Only `top_k` parents are read.
    let opt = options(1, 64, FaultPolicy::RejectHard);
    let mut state = parents(&cut, &opt, &genomes);
    simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
    assert_eq!(state.moves.len(), 1);
}

#[test]
fn proposals_are_identical_with_a_poisoned_holdout() {
    let cut = data();
    let opt = options(4, 64, FaultPolicy::RejectHard);
    let mut poisoned = cut.clone();
    poisoned.hold_x = vec![vec![-1e100], vec![1e100]];
    poisoned.hold_y = vec![f64::NAN, f64::INFINITY];
    // The parents are fixed: measured on the clean rows.
    let fixed = parents(&cut, &opt, &[ONE_TIMES, FOLDED, NEGATED, MAPPED]);
    let run = |c: &Cut| {
        let mut state = State::new(opt.search.budget.evaluations);
        state.budget = fixed.budget.clone();
        state.observations = fixed.observations.clone();
        state.seen = fixed.seen.clone();
        simplify_round(&mut state, c, &opt, &mut || true).unwrap();
        state
    };
    let (clean, dirty) = (run(&cut), run(&poisoned));
    assert_eq!(clean.moves, dirty.moves);
    assert_eq!(clean.budget.used(), dirty.budget.used());
    // The poison reached the children's measurements, not the proposals.
    let children = |s: &State| -> Vec<bool> {
        s.moves
            .iter()
            .filter_map(|m| m.measurement)
            .map(|i| s.observations[i].observation.measured())
            .collect()
    };
    assert_eq!(children(&clean), vec![true; 3]);
    assert_eq!(children(&dirty), vec![false; 3]);
    // A proposal reads the parent's text and the input width only.
    for genome in [ONE_TIMES, FOLDED, NEGATED, MAPPED] {
        let a = propose(&opt.evaluator, genome, 1).unwrap();
        let b = propose(&opt.evaluator, genome, 1).unwrap();
        assert_eq!(
            (a.rewrites, a.program, a.reason, a.admitted),
            (b.rewrites, b.program, b.reason, b.admitted)
        );
    }
}

// Runs through the published route.

/// The `cli_tests` seed program, which the pass reduces to `c0`.
const SEED: &str = ONE_TIMES;

/// The files every run publishes.
const SIDECARS: [&str; 7] = [
    "receipt.json",
    "prepared/prepared.json",
    "search/cut-0/search.json",
    "search/cut-0/observations.json",
    "search/cut-0/refinements.json",
    "emit/model.rs",
    "emit/manifest.json",
];

fn dataset() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../datasets/test/pmlb-229-pwlinear")
        .display()
        .to_string()
}

/// The `cli_tests` run (pmlb-229-pwlinear, rng seed 7, a frozen clock, Rust only, [`SEED`] as the
/// seeds file beside the output) in `dir/run`. The default selection is left unapplied, as the
/// CLI does.
fn run_options(
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

/// Every measured move's child scores bit for bit as its parent, in the published records.
fn check_exact(opt: &run::Options) -> usize {
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
const SEEDED_PINS: [&str; 4] = [
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

/// The run's CLI flags, parsed as `jope pipeline run` parses them.
fn flags(extra: &[&str]) -> (crate::cli::SearchArgs, crate::kernel2::cli::ProfileArgs) {
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
fn rewrite(
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

fn digested(engine: &EngineRecord, bytes: &[u8]) -> EngineRecord {
    let mut engine = engine.clone();
    engine.moves.as_mut().unwrap().sha256 = Some(crate::sha256_hex(bytes));
    engine
}

/// One named edit of the measured record at `first`; `other` is another measured program's
/// sha256.
fn tamper(moves: &mut Vec<MoveRecord>, first: usize, label: &str, other: &str) {
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
        "../../../../../../docs/internal/kernel2/b20-rewrite-simplify.md"
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

// Fix round 01: parents that raise faults, inputs the rewrite cannot read, the pool refusal
// count, and the set and order of the published records.

/// `(pipe X (<op> expr="<source>") (ridge))`, as the search prints it.
fn carrier(op: &str, source: &str) -> String {
    let expression = MathExpr::parse(source).unwrap().to_source().unwrap();
    GenomeIr::parse(&format!("(pipe X ({op} expr=\"{expression}\") (ridge))"))
        .unwrap()
        .canonical()
}

/// Three columns with a `0.0` and a `-0.0` in `c0`, values of `c0` under -1, negative values of
/// `c1`, and a row whose columns are equal (a zero row standard deviation).
fn faulting_cut() -> Cut {
    let row = |i: usize| -> Vec<f64> {
        let t = i as f64;
        match i {
            3 => vec![-0.0, -0.0, 0.5],
            7 => vec![1.0, 1.0, 1.0],
            _ => vec![(t - 12.0) / 4.0, (i % 5) as f64 - 2.0, (0.7 * t).sin()],
        }
    };
    let target = |r: &Vec<f64>| r[0] + 0.5 * r[1] - r[2];
    let train_x: Vec<Vec<f64>> = (0..24).map(row).collect();
    let hold_x: Vec<Vec<f64>> = (24..32).map(row).collect();
    Cut {
        index: 2,
        train_y: train_x.iter().map(target).collect(),
        hold_y: hold_x.iter().map(target).collect(),
        hold_indices: (0..hold_x.len()).collect(),
        train_x,
        hold_x,
    }
}

#[test]
fn rewritten_children_keep_fault_counts_and_signed_zeros_bit_for_bit() {
    use feat_math::math_expr::FaultClass;
    let cut = faulting_cut();
    assert!(cut
        .train_x
        .iter()
        .any(|r| r[0] == 0.0 && r[0].is_sign_negative()));
    // Each parent, the expression its rewrite gives, and the fault class its own expression
    // raises on a training row (none checked for the elementwise carrier, whose cells the
    // row-summary evaluator reads).
    let cases: [(String, &str, Option<FaultClass>); 5] = [
        (
            carrier("derive", "(1.0*c1)/(c0*1.0)"),
            "c1/c0",
            Some(FaultClass::DivNearZero),
        ),
        (
            carrier("derive", "log1p(c0/1.0)"),
            "log1p(c0)",
            Some(FaultClass::Log1pDomain),
        ),
        (
            carrier("derive", "sqrt(c1*1.0)"),
            "sqrt(c1)",
            Some(FaultClass::SqrtProjected),
        ),
        // `c0 + (-0.0)` keeps the sign of every zero, so the division sees `-0.0` as before.
        (
            carrier("derive", "1.0/(c0+(-0.0))"),
            "1.0/c0",
            Some(FaultClass::DivNearZero),
        ),
        // A `map` stage that reads row summaries runs on the compiled evaluator.
        (
            carrier("map", "(x-row_mean)/(row_std*1.0)"),
            "(x-row_mean)/row_std",
            None,
        ),
    ];
    for policy in [FaultPolicy::RejectHard, penalize()] {
        let opt = options(1, 64, policy);
        for (parent, rewritten, class) in &cases {
            let mut state = parents(&cut, &opt, &[parent.as_str()]);
            let original = state.observations[0].clone();
            assert!(
                original.faults.domain_cells > 0,
                "{parent}: {:?}",
                original.faults
            );
            if let Some(class) = class {
                let GenomeIr::Pipeline(pipe) = GenomeIr::parse(parent).unwrap() else {
                    panic!("{parent} is not a pipeline");
                };
                let Some(ParamValue::Str(source)) = pipe.stages[0].params.get("expr") else {
                    panic!("{parent} has no expression");
                };
                let expression = MathExpr::parse(source).unwrap();
                assert!(
                    cut.train_x
                        .iter()
                        .any(|row| expression.eval_row_traced(row).1.contains(class.set())),
                    "{parent} raises no {}",
                    class.name()
                );
            }
            simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
            assert_eq!(state.moves.len(), 1, "{parent}");
            let m = &state.moves[0];
            assert_eq!(m.outcome, MEASURED, "{parent}: {m:?}");
            assert_eq!(m.program, Some(with_expression(parent, 0, rewritten)));
            let child = &state.observations[m.measurement.unwrap()];
            assert_eq!(child.observation.origin, crate::kernel2::REWRITE);
            // Status, R², RMSE, selection score and every fault count, bit for bit.
            assert_eq!(scored(child), scored(&original), "{policy:?} {parent}");
            assert_eq!(scores(child), scores(&original), "{policy:?} {parent}");
        }
    }
}

/// `c0/(-1.0)`, which the pass reduces to `-c0`, a negation of a variable the profile refuses.
const DIVIDED: &str = "(pipe X (derive expr=\"c0/(-1.0)\") (ridge))";
/// `x*(-1.0)` in an elementwise carrier, which the pass reduces to `-x`, refused the same way.
const MAPPED_NEGATED: &str = "(pipe X (map expr=\"x*(-1.0)\") (ridge))";

#[test]
fn the_pool_refusal_count_of_fixed_parents_is_pinned() {
    let cut = data();
    let opt = options(4, 64, FaultPolicy::RejectHard);
    let mut state = parents(&cut, &opt, &[NEGATED, MIXED, DIVIDED, MAPPED_NEGATED]);
    simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
    assert_eq!(state.moves.len(), 4);
    // One smaller expression dropped per parent: the negation of a variable each reduces to.
    assert_eq!(refused_by_pool(&state.moves), 4);
    let dropped: Vec<&StageRewrite> = state
        .moves
        .iter()
        .flat_map(|m| &m.rewrites)
        .filter(|r| r.refused_by_pool())
        .collect();
    assert_eq!(dropped.len(), 4);
    for r in dropped {
        let refusal = r.refusal.as_deref().unwrap();
        assert!(refusal.contains("negative constant"), "{refusal}");
        assert!(r.expression.is_none() && r.stop != NOT_RUN && r.applications > 0);
    }
    // The count is what publication writes and `verify_run` counts again.
    let ids = [REWRITE_SATURATE.to_owned()];
    assert_eq!(pool_count(&ids, &state.moves), Some(4));
    assert_eq!(
        pool_count(&[MINIMAX_LINEARIZE.to_owned()], &state.moves),
        None
    );
    // A stage the rewrite cannot read is refused, not counted.
    let unreadable = propose(
        &opt.evaluator,
        "(pipe X (derive expr=\"exp(c0)\") (ridge))",
        1,
    )
    .unwrap();
    assert!(!unreadable.rewrites[0].refused_by_pool());
}

/// `exp(c0)`: a function outside the kernel2 pool.
const OUTSIDE_POOL: &str = "(pipe X (derive expr=\"exp(c0)\") (ridge))";
/// 16 nodes at depth 4: over the 15-node cap.
const OVER_CAPS: &str =
    "(pipe X (derive expr=\"clip((c0*1.0),(c0*1.0),(c0*1.0))+(c0*(c0*1.0))\") (ridge))";

#[test]
fn a_parent_the_rewrite_cannot_read_is_refused_and_the_pass_goes_on() {
    let cut = data();
    let opt = options(4, 64, FaultPolicy::RejectHard);
    for (text, why) in [(OUTSIDE_POOL, "function"), (OVER_CAPS, "node")] {
        // `simplify_v2` refuses the expression itself: the stage is refused, not the run.
        let proposal = propose(&opt.evaluator, text, 1).unwrap();
        assert!(!proposal.admitted && proposal.program.is_none() && proposal.reason.is_none());
        assert_eq!(proposal.outcome(), REFUSED);
        let r = &proposal.rewrites[0];
        assert_eq!(r.stop, NOT_RUN);
        assert_eq!(
            (r.iterations, r.enodes, r.applications, r.peak),
            (0, 0, 0, 0)
        );
        assert!(r.expression.is_none() && !r.refused_by_pool());
        let refusal = r.refusal.as_deref().unwrap();
        assert!(
            refusal.starts_with(REWRITE_FAILED) && refusal.len() > REWRITE_FAILED.len() + 2,
            "{refusal}"
        );
        // The profile's own reason: the pool's functions, or the node cap.
        assert!(refusal.contains(why), "{refusal}");
        lint(refusal);
        // The same proposal again: what `check` re-derives.
        assert_eq!(
            propose(&opt.evaluator, text, 1).unwrap().rewrites,
            proposal.rewrites
        );
        // In a round, ranked first: refused, and the pass reads the next parent.
        let mut state = parents(&cut, &opt, &[ONE_TIMES]);
        let mut unreadable = state.observations[0].clone();
        unreadable.observation.genome = text.into();
        unreadable.observation.program_sha256 = crate::sha256_hex(text.as_bytes());
        unreadable.selection_score = unreadable.selection_score.map(|s| s + 1.0);
        state.observations.push(unreadable);
        let used = state.budget.used();
        simplify_round(&mut state, &cut, &opt, &mut || true).unwrap();
        assert_eq!(state.moves.len(), 2, "{text}");
        assert_eq!(
            state.moves[0].parent_sha256,
            crate::sha256_hex(text.as_bytes())
        );
        assert_eq!(state.moves[0].outcome, REFUSED);
        assert_eq!(state.moves[0].rewrites, proposal.rewrites);
        assert_eq!(state.moves[1].outcome, MEASURED);
        assert_eq!(state.budget.used(), used + 1);
        assert_eq!(refused_by_pool(&state.moves), 0);
        // The receipt's not-run count sees it.
        assert_eq!(not_run(&state.moves), 1);
        assert_eq!(
            not_run_count(&[REWRITE_SATURATE.to_owned()], &state.moves),
            Some(1)
        );
    }
}

/// Rewrites a run's observations, rehashing search.json and the receipt, so that only the checks
/// that read the observations can refuse the change.
fn rewrite_measurements(output: &Path, receipt: &Receipt, measurements: &[Measurement]) -> PathBuf {
    let bytes = serde_json::to_vec_pretty(measurements).unwrap();
    std::fs::write(output.join("search/cut-0/observations.json"), &bytes).unwrap();
    let mut edited = receipt.clone();
    edited.search.observations_sha256 = crate::sha256_hex(&bytes);
    let summary = serde_json::to_vec_pretty(&edited.search).unwrap();
    std::fs::write(output.join("search/cut-0/search.json"), &summary).unwrap();
    edited.search_sha256 = crate::sha256_hex(&summary);
    let path = output.join("receipt.json");
    std::fs::write(&path, serde_json::to_vec(&edited).unwrap()).unwrap();
    path
}

/// `verify_run`'s refusal of `path`, which must name one of `expected`.
fn refusal_of(path: &Path, label: &str, expected: &[&str]) -> String {
    let error = format!(
        "{:#}",
        publication::verify_run(path).map(|_| ()).expect_err(label)
    );
    assert!(
        expected.iter().any(|e| error.contains(e)),
        "{label}: {error}"
    );
    format!("{label}: {error}")
}

/// A named edit of one published measurement.
type MeasurementEdit = (&'static str, fn(&mut Measurement));

/// The refusals of the set and order check: a record off the ranked parents, a pass short of
/// them with records after it, or a pass short of them before the run's last round.
const OFF_THE_PARENTS: [&str; 3] = [
    "move records do not follow the ranked parents",
    "move records continue after their pass ended",
    "a move pass stopped short of its ranked parents before the run's last round",
];

#[test]
fn verify_refuses_records_off_the_ranked_parents_and_inexact_children() {
    let temp = tempfile::tempdir().unwrap();
    let opt = run_options(temp.path(), 40, four_parents(), simplify());
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
    let originals: Vec<(PathBuf, Vec<u8>)> = [
        "receipt.json",
        "search/cut-0/search.json",
        "search/cut-0/observations.json",
        PATH,
    ]
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
    let measurements = published_measurements(&opt);
    // The published count re-derives from the records.
    assert_eq!(
        engine.moves.as_ref().unwrap().refused_by_pool,
        Some(refused_by_pool(&moves))
    );
    let mut errors = Vec::new();
    // Records deleted: an unchanged, a refused or a duplicate one, whichever the run has.
    let mut deleted = 0;
    for outcome in [UNCHANGED, REFUSED, DUPLICATE] {
        let Some(at) = moves.iter().position(|m| m.outcome == outcome) else {
            continue;
        };
        let mut edited = moves.clone();
        edited.remove(at);
        let bytes = serde_json::to_vec_pretty(&edited).unwrap();
        let path = rewrite(
            output,
            &receipt,
            Some(bytes.as_slice()),
            Some(digested(&engine, &bytes)),
        );
        let label = format!("{outcome} record removed");
        errors.push(refusal_of(&path, &label, &OFF_THE_PARENTS));
        restore();
        deleted += 1;
    }
    assert!(deleted > 0, "the run has no record that measures nothing");
    // The last record of an earlier round's pass removed: that pass now stops short of its
    // parents, which only the run's last round may do.
    let last = (0..moves.len())
        .filter(|&i| moves[i].outcome != MEASURED)
        .find(|&i| {
            moves[i].round + 1 < receipt.search.generations
                && moves.get(i + 1).is_none_or(|n| n.round != moves[i].round)
        })
        .expect("an earlier round whose last record measures nothing");
    let mut edited = moves.clone();
    edited.remove(last);
    let bytes = serde_json::to_vec_pretty(&edited).unwrap();
    let path = rewrite(
        output,
        &receipt,
        Some(bytes.as_slice()),
        Some(digested(&engine, &bytes)),
    );
    errors.push(refusal_of(
        &path,
        "a round's last record removed",
        &OFF_THE_PARENTS[1..],
    ));
    restore();
    // A record for a program the pass did not read: the bootstrap, which has no expression stage,
    // so its record re-derives (nothing to rewrite) and only the parent list refuses it.
    let bootstrap = crate::search::BOOTSTRAP_GENOME;
    let proposal = propose(&evaluator(), bootstrap, 1).unwrap();
    assert!(proposal.rewrites.is_empty() && proposal.program.is_none());
    let mut edited = moves.clone();
    edited.insert(
        0,
        MoveRecord {
            id: REWRITE_SATURATE.into(),
            round: 0,
            parent_sha256: crate::sha256_hex(bootstrap.as_bytes()),
            program: None,
            outcome: UNCHANGED.into(),
            reason: None,
            measurement: None,
            rewrites: Vec::new(),
            linearization: None,
        },
    );
    let bytes = serde_json::to_vec_pretty(&edited).unwrap();
    let path = rewrite(
        output,
        &receipt,
        Some(bytes.as_slice()),
        Some(digested(&engine, &bytes)),
    );
    errors.push(refusal_of(
        &path,
        "record for an unread parent",
        &OFF_THE_PARENTS[..1],
    ));
    restore();
    // Two records of one round swapped.
    let pair = moves
        .windows(2)
        .position(|w| w[0].round == w[1].round && w[0].parent_sha256 != w[1].parent_sha256)
        .expect("a round with two records");
    let mut edited = moves.clone();
    edited.swap(pair, pair + 1);
    let bytes = serde_json::to_vec_pretty(&edited).unwrap();
    let path = rewrite(
        output,
        &receipt,
        Some(bytes.as_slice()),
        Some(digested(&engine, &bytes)),
    );
    errors.push(refusal_of(&path, "records swapped", &OFF_THE_PARENTS[..1]));
    restore();
    // The pool refusal count changed or dropped, the records unchanged.
    for (label, count) in [
        ("pool refusal count", Some(refused_by_pool(&moves) + 1)),
        ("pool refusal count dropped", None),
    ] {
        let mut record = engine.clone();
        record.moves.as_mut().unwrap().refused_by_pool = count;
        let path = rewrite(output, &receipt, None, Some(record));
        errors.push(refusal_of(
            &path,
            label,
            &["the pool refusal count does not re-derive from the move records"],
        ));
        restore();
    }
    // A rewritten child's scores or fault counts edited: it no longer scores as its parent. The
    // child is not the selected program, and under RejectHard neither edit moves its rank.
    let child = moves
        .iter()
        .filter(|m| m.outcome == MEASURED)
        .filter_map(|m| m.measurement)
        .find(|&i| measurements[i].observation.genome != receipt.search.best.observation.genome)
        .expect("a rewritten child that was not selected");
    let edits: [MeasurementEdit; 3] = [
        ("child training R2", |m| {
            m.observation.train_r2 = m.observation.train_r2.map(|v| v - 1e-9)
        }),
        ("child RMSE", |m| {
            m.observation.rmse = m.observation.rmse.map(|v| v * (1.0 + 1e-9) + 1e-12)
        }),
        ("child evaluated cells", |m| {
            m.faults.evaluated_cells += 1;
            let mut totals = feat_math::kernel::admission::FaultTotals::default();
            totals.evaluated_cells = m.faults.evaluated_cells;
            totals.domain_cells = m.faults.domain_cells;
            totals.saturation_cells = m.faults.saturation_cells;
            m.domain_fault_rate = totals.domain_rate();
            m.saturation_fault_rate = totals.saturation_rate();
        }),
    ];
    for (label, edit) in edits {
        let mut edited = measurements.clone();
        edit(&mut edited[child]);
        assert_ne!(edited, measurements, "{label}: the edit was not applied");
        let path = rewrite_measurements(output, &receipt, &edited);
        errors.push(refusal_of(
            &path,
            label,
            &["a rewritten program does not score bit for bit as its parent"],
        ));
        restore();
    }
    publication::verify_run(&path).unwrap();
    let root = output.display().to_string();
    for error in errors {
        lint(&error.replace(&root, "<run>"));
    }
}

#[test]
fn an_unstarted_record_ends_its_pass() {
    let temp = tempfile::tempdir().unwrap();
    // Every seconds bound has passed: generation 0 measures the bootstrap and the two seeds, and
    // the pass stops before its first parent, in a run that stops on its seconds bound.
    let mut opt = run_options(temp.path(), 40, four_parents(), simplify());
    let seeds = temp.path().join("two-seeds.json");
    let file = crate::seeds::SeedsFile::new(
        None,
        [ONE_TIMES, FOLDED]
            .iter()
            .map(|genome| crate::seeds::SeedGenome {
                genome: (*genome).into(),
                program_sha256: crate::sha256_hex(genome.as_bytes()),
                bank_entry_sha256: None,
                receipt_sha256: None,
            })
            .collect(),
    );
    std::fs::write(&seeds, serde_json::to_vec(&file).unwrap()).unwrap();
    opt.run.seed_genomes = Some(seeds);
    opt.run.clock = Clock::Expired;
    let receipt = run::run(&opt).unwrap();
    assert_eq!(receipt.search.stopped_by, crate::search::STOPPED_BY_SECONDS);
    assert_eq!(receipt.search.generations, 1);
    assert!(published_moves(&opt).is_empty());
    let output = &opt.run.output;
    let path = output.join("receipt.json");
    publication::verify_run(&path).unwrap();
    // The two parents round 0's pass would read, in rank order.
    let observations: Vec<Kernel2Observation> = published_measurements(&opt)
        .iter()
        .map(|m| m.derive(FaultPolicy::RejectHard).unwrap())
        .collect();
    let width = (receipt.dataset.columns - 1) as usize;
    let ranked = owned(ranked_parents(&evaluator(), &observations, REWRITE_SATURATE, 4).unwrap());
    assert_eq!(ranked.len(), 2);
    let unstarted = |(genome, sha256): &(String, String)| {
        let proposal = propose(&evaluator(), genome, width).unwrap();
        assert!(proposal.admitted, "{genome}");
        MoveRecord {
            id: REWRITE_SATURATE.into(),
            round: 0,
            parent_sha256: sha256.clone(),
            program: proposal.program,
            outcome: NOT_STARTED.into(),
            reason: None,
            measurement: None,
            rewrites: proposal.rewrites,
            linearization: None,
        }
    };
    let engine = receipt.search.engine_selection.clone().unwrap();
    // One unstarted record for the first parent: what a run whose start guard refused that
    // measurement publishes.
    let one = vec![unstarted(&ranked[0])];
    let bytes = serde_json::to_vec_pretty(&one).unwrap();
    let path = rewrite(
        output,
        &receipt,
        Some(bytes.as_slice()),
        Some(digested(&engine, &bytes)),
    );
    publication::verify_run(&path).unwrap();
    // A record after it: the pass had ended.
    let two = vec![unstarted(&ranked[0]), unstarted(&ranked[1])];
    let bytes = serde_json::to_vec_pretty(&two).unwrap();
    let path = rewrite(
        output,
        &receipt,
        Some(bytes.as_slice()),
        Some(digested(&engine, &bytes)),
    );
    let error = refusal_of(
        &path,
        "a record after an unstarted one",
        &["move records continue after their pass ended"],
    );
    lint(&error.replace(&output.display().to_string(), "<run>"));
    // The second parent alone: not the first ranked parent.
    let skipped = vec![unstarted(&ranked[1])];
    let bytes = serde_json::to_vec_pretty(&skipped).unwrap();
    let path = rewrite(
        output,
        &receipt,
        Some(bytes.as_slice()),
        Some(digested(&engine, &bytes)),
    );
    refusal_of(&path, "the first parent skipped", &OFF_THE_PARENTS[..1]);
}

// Fix round 02: the last round of a run stopped on its evaluations bound, the last-round rule on
// its own, and the not-run count.

/// `engine` with the digest and both counts of `moves` (serialized as `bytes`), as publication
/// writes them, so that only the record checks can refuse the records.
fn recounted(engine: &EngineRecord, moves: &[MoveRecord], bytes: &[u8]) -> EngineRecord {
    let mut engine = digested(engine, bytes);
    let record = engine.moves.as_mut().unwrap();
    record.refused_by_pool = Some(refused_by_pool(moves));
    record.not_run = Some(not_run(moves));
    engine
}

/// Publishes `moves` in place of the run's records, under a recounted selection record.
fn publish_recounted(
    output: &Path,
    receipt: &Receipt,
    engine: &EngineRecord,
    moves: &[MoveRecord],
) -> PathBuf {
    let bytes = serde_json::to_vec_pretty(moves).unwrap();
    rewrite(
        output,
        receipt,
        Some(bytes.as_slice()),
        Some(recounted(engine, moves, &bytes)),
    )
}

/// The files a records edit rewrites, read back so they can be restored.
fn originals(output: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    ["receipt.json", "search/cut-0/search.json", PATH]
        .iter()
        .map(|r| (output.join(r), std::fs::read(output.join(r)).unwrap()))
        .collect()
}

fn restore(originals: &[(PathBuf, Vec<u8>)]) {
    for (file, bytes) in originals {
        std::fs::write(file, bytes).unwrap();
    }
}

#[test]
fn under_an_evaluations_stop_the_last_round_stops_where_the_allowance_ran_out() {
    let temp = tempfile::tempdir().unwrap();
    // 24 evaluations: one round, whose rewrite pass measures one child with the last evaluation
    // and stops short of its other three ranked parents.
    let opt = run_options(temp.path(), 24, four_parents(), simplify());
    let receipt = run::run(&opt).unwrap();
    let s = &receipt.search;
    assert_eq!(s.stopped_by, crate::search::STOPPED_BY_EVALUATIONS);
    assert_eq!(s.generations, 1);
    let output = &opt.run.output;
    let saved = originals(output);
    let engine = s.engine_selection.clone().unwrap();
    let moves = published_moves(&opt);
    let measurements = published_measurements(&opt);
    assert_eq!(moves.len(), 1);
    assert_eq!(moves[0].outcome, MEASURED);
    assert_eq!(moves[0].measurement, Some(measurements.len() - 1));
    publication::verify_run(&output.join("receipt.json")).unwrap();
    // The round's ranked parents, as the pass read them.
    let observations: Vec<Kernel2Observation> = measurements
        .iter()
        .map(|m| m.derive(FaultPolicy::RejectHard).unwrap())
        .collect();
    let start = observations.len() - 1;
    let parents =
        owned(ranked_parents(&evaluator(), &observations[..start], REWRITE_SATURATE, 4).unwrap());
    assert_eq!(parents.len(), 4);
    assert_eq!(moves[0].parent_sha256, parents[0].1);
    let mut errors = Vec::new();
    // A record for the next ranked parent appended after the measurement that spent the
    // allowance. Its program was already measured, so as a duplicate it re-derives on its own,
    // and only the allowance refuses it.
    let width = (receipt.dataset.columns - 1) as usize;
    let proposal = propose(&evaluator(), &parents[1].0, width).unwrap();
    assert!(proposal.admitted);
    let program = proposal.program.clone().unwrap();
    assert!(measurements.iter().any(|m| m.observation.genome == program));
    let mut appended = moves.clone();
    appended.push(MoveRecord {
        id: REWRITE_SATURATE.into(),
        round: 0,
        parent_sha256: parents[1].1.clone(),
        program: proposal.program,
        outcome: DUPLICATE.into(),
        reason: proposal.reason,
        measurement: None,
        rewrites: proposal.rewrites,
        linearization: None,
    });
    let path = publish_recounted(output, &receipt, &engine, &appended);
    errors.push(refusal_of(
        &path,
        "a record after the allowance was spent",
        &["a move record starts after the allowance was spent"],
    ));
    restore(&saved);
    // The round's measured record removed: the pass now stops short with an evaluation of
    // allowance left, the one its child spent.
    let path = publish_recounted(output, &receipt, &engine, &[]);
    errors.push(refusal_of(
        &path,
        "the last round's measured record removed",
        &["a move pass stopped short of its ranked parents with allowance left"],
    ));
    restore(&saved);
    // The not-run count changed or dropped, the records unchanged.
    for (label, count) in [("not-run count", Some(1)), ("not-run count dropped", None)] {
        let mut record = engine.clone();
        assert_eq!(record.moves.as_ref().unwrap().not_run, Some(0));
        record.moves.as_mut().unwrap().not_run = count;
        let path = rewrite(output, &receipt, None, Some(record));
        errors.push(refusal_of(
            &path,
            label,
            &["the not-run count does not re-derive from the move records"],
        ));
        restore(&saved);
    }
    publication::verify_run(&output.join("receipt.json")).unwrap();
    let root = output.display().to_string();
    for error in errors {
        lint(&error.replace(&root, "<run>"));
    }
}

#[test]
fn only_the_last_round_may_stop_short() {
    let temp = tempfile::tempdir().unwrap();
    // 36 evaluations: two rounds. Round 0's pass reads four parents (measured, duplicate,
    // unchanged, unchanged); round 1's reads none, the allowance spent before it.
    let opt = run_options(temp.path(), 36, four_parents(), simplify());
    let receipt = run::run(&opt).unwrap();
    let s = &receipt.search;
    assert_eq!(s.stopped_by, crate::search::STOPPED_BY_EVALUATIONS);
    assert_eq!(s.generations, 2);
    let output = &opt.run.output;
    let saved = originals(output);
    let engine = s.engine_selection.clone().unwrap();
    let moves = published_moves(&opt);
    let outcomes: Vec<&str> = moves.iter().map(|m| m.outcome.as_str()).collect();
    assert_eq!(outcomes, [MEASURED, DUPLICATE, UNCHANGED, UNCHANGED]);
    assert!(moves.iter().all(|m| m.round == 0));
    // Round 0's records after its measured one removed. No measurement is left unaccounted, no
    // record follows, and the counts are recounted, so only the rule that a pass may stop short
    // only in the run's last round refuses it.
    let kept = &moves[..1];
    let path = publish_recounted(output, &receipt, &engine, kept);
    let error = refusal_of(
        &path,
        "an earlier round's tail removed",
        &["a move pass stopped short of its ranked parents before the run's last round"],
    );
    restore(&saved);
    publication::verify_run(&output.join("receipt.json")).unwrap();
    lint(&error.replace(&output.display().to_string(), "<run>"));
}

// Fix round 03: the owner's D9b ruling on what the pool refusal count counts.

#[test]
fn a_parent_ranked_top_for_n_rounds_adds_n_pool_refusals() {
    let temp = tempfile::tempdir().unwrap();
    // [`NEGATED`] as the only seed, one parent per round and no refinement trials: the seed stays
    // the top ranked parent with an expression stage in every round whose pass runs. Its only
    // rewrite output, the negation of `c0`, is refused by the profile, so no child is measured
    // and nothing outranks it. 48 evaluations: six rounds, the last with its allowance spent
    // before its pass.
    let mut opt = run_options(
        temp.path(),
        48,
        Refinement {
            top_k: 1,
            trials_per_program: 0,
        },
        simplify(),
    );
    let seeds = temp.path().join("negated-seed.json");
    let file = crate::seeds::SeedsFile::new(
        None,
        vec![crate::seeds::SeedGenome {
            genome: NEGATED.into(),
            program_sha256: crate::sha256_hex(NEGATED.as_bytes()),
            bank_entry_sha256: None,
            receipt_sha256: None,
        }],
    );
    std::fs::write(&seeds, serde_json::to_vec(&file).unwrap()).unwrap();
    opt.run.seed_genomes = Some(seeds);
    let receipt = run::run(&opt).unwrap();
    let s = &receipt.search;
    let moves = published_moves(&opt);
    let n = moves.len();
    assert!(n >= 3, "{n} rounds read the parent");
    assert_eq!(s.stopped_by, crate::search::STOPPED_BY_EVALUATIONS);
    assert_eq!(s.generations, n + 1);
    // One record per round, each for the same parent, each the same refusal by the pool.
    let parent = crate::sha256_hex(NEGATED.as_bytes());
    for (round, m) in moves.iter().enumerate() {
        assert_eq!(
            (m.round, m.parent_sha256.as_str(), m.outcome.as_str()),
            (round, parent.as_str(), REFUSED)
        );
        assert!(m.program.is_none() && m.measurement.is_none());
        assert_eq!(m.rewrites.len(), 1);
        assert!(m.rewrites[0].refused_by_pool(), "{:?}", m.rewrites[0]);
        assert_eq!(m.rewrites, moves[0].rewrites);
    }
    // The published count is N: one per round the parent is read, not one per distinct
    // expression.
    let record = s.engine_selection.as_ref().unwrap().moves.as_ref().unwrap();
    assert_eq!(record.refused_by_pool, Some(n));
    assert_eq!(refused_by_pool(&moves), n);
    assert_eq!(pool_count(&record.ids, &moves), Some(n));
    // `verify_run` accepts the run, re-deriving N from the records (and, since it re-derives the
    // set and order of records against the ranked measurements, that the parent was the top
    // ranked one in each of those rounds).
    let output = &opt.run.output;
    assert_eq!(
        publication::verify_run(&output.join("receipt.json")).unwrap(),
        receipt
    );
    // N - 1 and N + 1, search.json and the receipt rehashed: refused.
    let saved = originals(output);
    let engine = s.engine_selection.clone().unwrap();
    let mut errors = Vec::new();
    for count in [n - 1, n + 1] {
        let mut edited = engine.clone();
        edited.moves.as_mut().unwrap().refused_by_pool = Some(count);
        let path = rewrite(output, &receipt, None, Some(edited));
        errors.push(refusal_of(
            &path,
            &format!("a pool refusal count of {count} for {n} rounds"),
            &["the pool refusal count does not re-derive from the move records"],
        ));
        restore(&saved);
    }
    publication::verify_run(&output.join("receipt.json")).unwrap();
    let root = output.display().to_string();
    for error in errors {
        lint(&error.replace(&root, "<run>"));
    }
}

// Fix round 04: a record past the run's last round, which only the final check that every
// record was read refuses (mutants-01, M-E2).

#[test]
fn verify_refuses_a_move_record_past_the_runs_last_round() {
    let temp = tempfile::tempdir().unwrap();
    let opt = run_options(temp.path(), 40, four_parents(), simplify());
    let receipt = run::run(&opt).unwrap();
    let s = &receipt.search;
    let output = &opt.run.output;
    let saved = originals(output);
    let engine = s.engine_selection.clone().unwrap();
    let moves = published_moves(&opt);
    publication::verify_run(&output.join("receipt.json")).unwrap();
    // A copy of a record that measures nothing, for the round after the run's last, appended
    // with the counts recounted. The check reads records round by round up to the run's last
    // round, so it never compares this one with a parent, a proposal or a measurement; only the
    // final check that every record was read refuses it.
    let mut extra = moves
        .iter()
        .find(|m| m.outcome != MEASURED)
        .expect("a record that measures nothing")
        .clone();
    extra.round = s.generations;
    let mut appended = moves.clone();
    appended.push(extra);
    let path = publish_recounted(output, &receipt, &engine, &appended);
    let error = publication::verify_run(&path)
        .map(|_| ())
        .expect_err("a record past the run's last round");
    assert_eq!(
        error.root_cause().to_string(),
        "move records do not follow the ranked parents",
        "{error:#}"
    );
    lint(&format!("{error:#}").replace(&output.display().to_string(), "<run>"));
    restore(&saved);
    publication::verify_run(&output.join("receipt.json")).unwrap();
}

/// [`check`] reads the pass's parents back from evidence, which is untrusted, so
/// `has_expression_stage` and `propose` parse them within `parse_untrusted_pipeline`'s bounds (the
/// class of B17's review 02): the pipeline form only, and each expression's nesting bytes counted
/// before it is parsed. Unary plus and redundant parentheses never reach the parsed tree, so a
/// parent at the limit is read and rewritten as its plain form is; one byte more is not readable,
/// and `propose` refuses it unparsed.
#[test]
fn rewrite_parents_are_bounded_before_they_are_parsed() {
    use crate::kernel2::refinement::MAX_PARENT_EXPRESSION_NESTING;
    let opt = options(4, 64, FaultPolicy::RejectHard);
    let body = "1*c0";
    // `*`.
    let room = MAX_PARENT_EXPRESSION_NESTING - 1;
    let parent = |source: String| ONE_TIMES.replace(body, &source);
    let plus = |n: usize| parent(format!("{}{body}", "+".repeat(n)));
    let grouped = |n: usize| parent(format!("{}{body}{}", "(".repeat(n), ")".repeat(n)));
    let plain = propose(&opt.evaluator, ONE_TIMES, 1).unwrap();
    assert!(plain.admitted);
    for program in [plus(room), grouped(room)] {
        assert!(has_expression_stage(&program), "{program}");
        let padded = propose(&opt.evaluator, &program, 1).unwrap();
        assert_eq!(
            (&padded.program, padded.admitted),
            (&plain.program, plain.admitted),
            "{program}"
        );
    }
    for program in [plus(room + 1), grouped(room + 1)] {
        assert!(!has_expression_stage(&program), "{program}");
        let error = propose(&opt.evaluator, &program, 1).unwrap_err();
        assert!(
            format!("{error:#}").contains("nests too deeply"),
            "{program}: {error:#}"
        );
    }
    let closed_form = format!("(sym raw \"{}c0\")", "+".repeat(room + 1));
    assert!(!has_expression_stage(&closed_form));
    let error = propose(&opt.evaluator, &closed_form, 1).unwrap_err();
    assert!(format!("{error:#}").contains("not a pipeline"), "{error:#}");
}

/// Independent review 01 of the wiring PR (B20-R01-AST-FIRST): `MathExpr::simplify_v2`, the
/// public entry, checks a decoded tree's structure before it copies or normalizes it. A
/// `MathExpr` decoded from JSON has had no structure check, and the copy and `Search::finish`'s
/// normalization both recurse.
#[test]
fn simplify_v2_checks_a_decoded_tree_before_copying_it() {
    let opt = options(4, 64, FaultPolicy::RejectHard);
    let profile = opt.evaluator.profile();
    let pool = profile.expr.as_ref().unwrap();
    let scope = || ExpressionScope {
        columns: pool.column_upper_exclusive,
        elementwise: false,
        max_nodes: pool.max_nodes,
        max_depth: pool.max_depth,
    };
    // #202 review 04's shallow 65-node tree: one call with 64 literal arguments.
    let leaves = vec![serde_json::json!({ "Const": 1.0 }); 64];
    let wide: MathExpr =
        serde_json::from_value(serde_json::json!({ "root": { "Call": ["Abs", leaves] } })).unwrap();
    let error = wide.simplify_v2(profile, scope()).unwrap_err();
    assert!(
        format!("{error:#}").contains("before it is copied"),
        "{error:#}"
    );
    // An admitted tree passes the check and simplifies as before.
    assert!(MathExpr::parse("1*c0")
        .unwrap()
        .simplify_v2(profile, scope())
        .is_ok());
}

/// Independent review 01 of the wiring PR (B20-R01-ROUND-BOUND): `check` replays every round the
/// receipt declares, so the round count is bounded by the search's stop rule first, at most
/// `BARREN_ROUND_LIMIT` rounds per completed evaluation plus that many. One more is refused, with
/// `search.json` and the receipt rehashed so that only the bound can see it.
#[test]
fn verify_bounds_the_round_count_before_replaying_rounds() {
    let temp = tempfile::tempdir().unwrap();
    let opt = run_options(temp.path(), 40, four_parents(), simplify());
    let receipt = run::run(&opt).unwrap();
    let s = &receipt.search;
    let bound = BARREN_ROUND_LIMIT * (s.completed_evaluations + 1);
    assert!(s.generations <= bound, "{} > {bound}", s.generations);
    let output = &opt.run.output;
    let mut edited = receipt.clone();
    edited.search.generations = bound + 1;
    let summary = serde_json::to_vec_pretty(&edited.search).unwrap();
    std::fs::write(output.join("search/cut-0/search.json"), &summary).unwrap();
    edited.search_sha256 = crate::sha256_hex(&summary);
    let path = output.join("receipt.json");
    std::fs::write(&path, serde_json::to_vec(&edited).unwrap()).unwrap();
    let error = publication::verify_run(&path).unwrap_err();
    assert!(
        format!("{error:#}").contains("round count exceeds"),
        "{error:#}"
    );
}

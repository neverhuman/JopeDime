//! The bounded minimax linearization pass at the end of each search round
//! (laneD:B19-minimax-linearize).
use super::super::{Refinement, State};
use super::*;
use crate::{
    cli::{Cli, Cmd},
    clock::Clock,
    emit::Language,
    kernel2::{
        archive, publication, receipt::Receipt, run, tests::evaluator, ArchivePolicy,
        EngineSelection, Kernel2Observation, MovesRecord, RefineMethod,
    },
    run::{Policy, RunOptions},
    search::{Budget, SearchOptions},
};
use clap::Parser;
use feat_math::{
    evolve::FaultPolicy,
    kernel::expression_search::{
        LmOptions, LINEARIZE_FIT_REFUSED, LINEARIZE_MISSES_BOUND, LINEARIZE_NON_FINITE,
        LINEARIZE_NOT_SMALLER, LINEARIZE_OPEN_BRACKET, LINEARIZE_OUT_OF_RANGE,
        LINEARIZE_OVER_BUDGET, LINEARIZE_OVER_LIMITS, LINEARIZE_PROFILE_REFUSED,
        LINEARIZE_UNSAMPLED,
    },
    Rows,
};
use std::path::PathBuf;

/// The cubic `0.5 x³`, 7 nodes; its one considered subtree is the root.
const CUBE: &str = "(c0*c0)*(c0*0.5)";
/// The same values spelled the other way round.
const CUBE_SWAPPED: &str = "(c0*0.5)*(c0*c0)";
/// `x³ + x² + 0.3 x`, 13 nodes: the root, the 5-node cube (no smaller as a line) and the 7-node
/// quadratic part, whose line `0.3 x + 0.5` rises, so it fits the depth limit beside the cube.
const CUBE_PLUS: &str = "((c0*c0)*c0)+((c0*c0)+(c0*0.3))";
/// Exactly linear plus `1e-6 sin(c0)`: on [1e6, 1e6 + 1] its line loses about 1e-10 to rounding,
/// over the minimax bound.
const OFF_CENTRE: &str = "((c0*2.0)-2000001.0)+(sin(c0)*0.000001)";
/// `x³ - 3x`, 9 nodes, which turns back up past `x = 1`.
const TURN: &str = "((c0*c0)*c0)-(c0*3.0)";

fn linearize() -> EngineSelection {
    EngineSelection {
        linearize: true,
        ..EngineSelection::default()
    }
}

/// `(pipe X (derive expr="<source>") (ridge))`, as the search prints it.
fn program(source: &str) -> String {
    let expression = MathExpr::parse(source).unwrap().to_source().unwrap();
    GenomeIr::parse(&format!("(pipe X (derive expr=\"{expression}\") (ridge))"))
        .unwrap()
        .canonical()
}

/// A one-column cut: `points` training rows evenly over `[lo, hi]`, the held-out `hold` rows,
/// and `y` as the target of both.
fn cut(lo: f64, hi: f64, points: usize, hold: &[f64], y: fn(f64) -> f64) -> Cut {
    let train_x: Rows = (0..points)
        .map(|i| vec![lo + (hi - lo) * i as f64 / (points - 1) as f64])
        .collect();
    let hold_x: Rows = hold.iter().map(|&v| vec![v]).collect();
    Cut {
        index: 2,
        train_y: train_x.iter().map(|r| y(r[0])).collect(),
        hold_y: hold_x.iter().map(|r| y(r[0])).collect(),
        hold_indices: (0..hold.len()).collect(),
        train_x,
        hold_x,
    }
}

fn cube(v: f64) -> f64 {
    v * v * v
}

/// Training and held-out rows over [-0.9, 1.1], the target `x³`. The range is not symmetric, so
/// no line of an odd subtree has an intercept of exactly zero (a 3-node line).
fn cubic_cut() -> Cut {
    let hold: Vec<f64> = (0..41).map(|i| -0.9 + f64::from(i) / 20.0).collect();
    cut(-0.9, 1.1, 201, &hold, cube)
}

/// Search options with the given selection, `top_k` parents per round and no refinement trials.
fn options(top_k: usize, evaluations: usize, selection: EngineSelection) -> Options {
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
        FaultPolicy::RejectHard,
        Refinement {
            top_k,
            trials_per_program: 0,
        },
    )
    .unwrap()
    .with_engine(selection)
    .unwrap()
}

/// A search state holding `genomes`, each measured on `cut` and charged.
fn parents(cut: &Cut, options: &Options, genomes: &[String]) -> State {
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

fn records_of<'a>(state: &'a State, parent: &str) -> Vec<&'a MoveRecord> {
    let sha = crate::sha256_hex(parent.as_bytes());
    state
        .moves
        .iter()
        .filter(|m| m.parent_sha256 == sha)
        .collect()
}

fn line(m: &MoveRecord) -> &LinearizationRecord {
    m.linearization.as_ref().unwrap()
}

fn child<'a>(state: &'a State, m: &MoveRecord) -> &'a Kernel2Observation {
    &state.observations[m.measurement.unwrap()]
}

fn score(o: &Kernel2Observation) -> f64 {
    o.selection_score.unwrap()
}

/// Where `o` ranks among the state's measurements, as the search ranks them.
fn rank_of(state: &State, options: &Options, o: &Kernel2Observation) -> usize {
    options
        .evaluator
        .rank(&state.observations)
        .iter()
        .position(|r| r.observation.genome == o.observation.genome)
        .unwrap()
}

#[test]
fn a_line_that_misses_the_minimax_bound_is_refused_with_its_reason_and_nothing_is_measured() {
    let hold = [1e6 + 0.25, 1e6 + 0.5, 1e6 + 0.75];
    let cut = cut(1e6, 1e6 + 1.0, 201, &hold, |v| 2.0 * v - 2_000_001.0);
    let opt = options(4, 64, linearize());
    let parent = program(OFF_CENTRE);
    let mut state = parents(&cut, &opt, std::slice::from_ref(&parent));
    let (used, measured) = (state.budget.used(), state.observations.len());
    linearize_round(&mut state, &cut, &opt, &mut || true).unwrap();
    // The root, then its two children, each considered and refused.
    assert_eq!(state.moves.len(), 3, "{:?}", state.moves);
    let root = &state.moves[0];
    let record = line(root);
    assert_eq!((record.stage, record.path.as_slice()), (0, &[][..]));
    assert_eq!(record.degree, 1);
    let fit = record.fit.as_ref().unwrap();
    assert!(
        ["exact_fit", "converged", "noise_limited"].contains(&fit.stop.as_str()),
        "{fit:?}"
    );
    let (lowered, bound) = (record.lowered_error.unwrap(), record.bound.unwrap());
    assert!(lowered > bound, "{lowered} <= {bound}");
    assert_eq!(
        bound.to_bits(),
        (fit.max_error * (1.0 + 1e-6) + 64.0 * f64::EPSILON * fit.max_abs).to_bits()
    );
    assert_eq!(record.refusal.as_deref(), Some(LINEARIZE_MISSES_BOUND));
    assert_eq!(root.outcome, REFUSED);
    assert!(root.program.is_none() && root.reason.is_none() && root.measurement.is_none());
    assert!(record.expression.is_none());
    for m in &state.moves[1..] {
        assert_eq!(m.outcome, REFUSED);
        assert_eq!(line(m).refusal.as_deref(), Some(LINEARIZE_NOT_SMALLER));
    }
    // Nothing measured, nothing charged.
    assert_eq!(
        (state.budget.used(), state.observations.len()),
        (used, measured)
    );
    // The same records re-derive from the parent and the recorded fits.
    let recorded: Vec<LinearizationRecord> = state.moves.iter().map(|m| line(m).clone()).collect();
    let again = linearize::rederive(&opt.evaluator, &parent, 1, &recorded).unwrap();
    assert!(again
        .iter()
        .zip(&state.moves)
        .all(|(p, m)| Some(&p.linearization) == m.linearization.as_ref() && !p.admitted));
}

#[test]
fn near_linear_subtrees_are_linearized_measured_and_ranked_by_their_native_score() {
    let cut = cubic_cut();
    let opt = options(4, 64, linearize());
    let (a, b) = (program(CUBE), program(CUBE_PLUS));
    let mut state = parents(&cut, &opt, &[a.clone(), b.clone()]);
    let before = state.budget.used();
    linearize_round(&mut state, &cut, &opt, &mut || true).unwrap();
    let (from_a, from_b) = (records_of(&state, &a), records_of(&state, &b));
    assert_eq!(from_a.len(), 1);
    assert_eq!(
        from_b
            .iter()
            .map(|m| line(m).path.clone())
            .collect::<Vec<_>>(),
        vec![vec![], vec![0], vec![1]]
    );
    assert_eq!(from_b[1].outcome, REFUSED);
    assert_eq!(
        line(from_b[1]).refusal.as_deref(),
        Some(LINEARIZE_NOT_SMALLER)
    );
    let offered: Vec<&MoveRecord> = [from_a[0], from_b[0], from_b[2]].into();
    for m in &offered {
        let record = line(m);
        assert_eq!(m.outcome, MEASURED, "{m:?}");
        assert!(record.lowered_error.unwrap() <= record.bound.unwrap());
        assert!(record.refusal.is_none() && m.reason.is_none());
        let o = child(&state, m);
        assert_eq!(o.observation.origin, crate::kernel2::LINEARIZE);
        assert_eq!(Some(&o.observation.genome), m.program.as_ref());
        assert!(o.observation.measured(), "{:?}", o.observation);
        // The child's score is its own native measurement.
        let native = opt.evaluator.evaluate(
            &o.observation.genome,
            &cut,
            state.generation,
            crate::kernel2::LINEARIZE,
            Clock::Frozen,
        );
        assert_eq!(&native, o);
        // The program is the parent with the stage's expression replaced, and that expression is
        // the line in place of the subtree.
        let GenomeIr::Pipeline(pipe) = GenomeIr::parse(m.program.as_ref().unwrap()).unwrap() else {
            panic!("pipeline expected")
        };
        let Some(ParamValue::Str(source)) = pipe.stages[0].params.get("expr") else {
            panic!("expression expected")
        };
        assert_eq!(Some(source), record.expression.as_ref());
    }
    // One evaluation per measured child; the shared identity holds.
    assert_eq!(state.budget.used() - before, 3);
    assert_eq!(state.budget.used(), state.observations.len());
    // The cube's line is far closer to its subtree than the quadratic part's line, yet the
    // program keeping the cube (x³ plus a line) ranks above the program that is a line: the
    // measured score decides, never the line's error.
    let (line_a, line_b) = (line(from_a[0]), line(from_b[2]));
    assert!(line_a.lowered_error.unwrap() < line_b.lowered_error.unwrap());
    let (child_a, child_b) = (child(&state, from_a[0]), child(&state, from_b[2]));
    assert!(score(child_b) > score(child_a), "{child_a:?} {child_b:?}");
    assert!(rank_of(&state, &opt, child_b) < rank_of(&state, &opt, child_a));
    // Linearizing the quadratic part also beats its own parent; the exact cube stays first.
    let parent_b = &state.observations[1];
    assert!(score(child_b) > score(parent_b));
    assert_eq!(rank_of(&state, &opt, &state.observations[0]), 0);
}

#[test]
fn a_line_offered_on_the_training_range_can_be_worse_off_it_and_the_native_score_decides() {
    // `x³ - 3x`: on the training range its line falls steeply; on the held-out rows, in [1.2, 2],
    // the curve turns back up through the training targets' own range while the line keeps
    // falling.
    let hold = [1.2, 1.4, 1.6, 1.8, 2.0];
    let cut = cut(-1.0, 1.05, 201, &hold, |v| v * v * v - 3.0 * v);
    let opt = options(4, 64, linearize());
    let parent = program(TURN);
    let mut state = parents(&cut, &opt, std::slice::from_ref(&parent));
    linearize_round(&mut state, &cut, &opt, &mut || true).unwrap();
    let m = state
        .moves
        .iter()
        .find(|m| line(m).path.is_empty())
        .unwrap();
    let record = line(m);
    // Offered: within the minimax bound on the training range.
    assert_eq!(m.outcome, MEASURED);
    let lowered = record.lowered_error.unwrap();
    assert!(lowered <= record.bound.unwrap());
    // Off that range the line drifts from the subtree far past its error on it.
    let subtree = MathExpr::parse(&record.subtree).unwrap();
    let admitted = MathExpr::parse(record.expression.as_ref().unwrap()).unwrap();
    let (s2, _) = subtree.eval_row_traced(&[2.0]);
    let (l2, _) = admitted.eval_row_traced(&[2.0]);
    assert!((s2 - l2).abs() > 2.0 * lowered, "{s2} vs {l2}, {lowered}");
    // The parent fits the held-out rows; the child does not, and ranks under it.
    let parent_o = &state.observations[0];
    let child_o = child(&state, m);
    assert!(child_o.observation.measured());
    let (parent_hold, child_hold) = (
        parent_o.observation.hold_r2.unwrap(),
        child_o.observation.hold_r2.unwrap(),
    );
    let both = format!("{parent_o:?}\n{child_o:?}");
    assert!(parent_hold > 0.9, "{both}");
    assert!(child_hold < parent_hold - 0.5, "{both}");
    assert!(score(child_o) < score(parent_o), "{both}");
    assert!(rank_of(&state, &opt, parent_o) < rank_of(&state, &opt, child_o));
}

#[test]
fn proposals_are_identical_with_a_poisoned_holdout() {
    let clean = cubic_cut();
    let opt = options(4, 64, linearize());
    let mut poisoned = clean.clone();
    poisoned.hold_x = vec![vec![-1e100], vec![1e100]];
    poisoned.hold_y = vec![f64::NAN, f64::INFINITY];
    poisoned.hold_indices = vec![0, 1];
    // The parents are fixed: measured on the clean rows.
    let genomes = [program(CUBE), program(CUBE_PLUS)];
    let fixed = parents(&clean, &opt, &genomes);
    let run = |c: &Cut| {
        let mut state = State::new(opt.search.budget.evaluations);
        state.budget = fixed.budget.clone();
        state.observations = fixed.observations.clone();
        state.seen = fixed.seen.clone();
        linearize_round(&mut state, c, &opt, &mut || true).unwrap();
        state
    };
    let (a, b) = (run(&clean), run(&poisoned));
    assert_eq!(a.moves, b.moves);
    assert_eq!(a.budget.used(), b.budget.used());
    // The poison reached the children's measurements, not the proposals.
    let children = |s: &State| -> Vec<bool> {
        s.moves
            .iter()
            .filter_map(|m| m.measurement)
            .map(|i| s.observations[i].observation.measured())
            .collect()
    };
    assert_eq!(children(&a), vec![true; 3]);
    assert_eq!(children(&b), vec![false; 3]);
    // A proposal reads the parent's text and the training rows and labels only.
    for genome in &genomes {
        let x = linearize::propose(&opt.evaluator, genome, &clean, 1).unwrap();
        let y = linearize::propose(&opt.evaluator, genome, &poisoned, 1).unwrap();
        assert_eq!(x.len(), y.len());
        for (x, y) in x.iter().zip(&y) {
            assert_eq!(
                (&x.linearization, &x.program, &x.reason, x.admitted),
                (&y.linearization, &y.program, &y.reason, y.admitted)
            );
        }
    }
}

#[test]
fn the_allowance_the_start_guard_top_k_and_deduplication_bound_the_pass() {
    let cut = cubic_cut();
    let genomes = [program(CUBE), program(CUBE_SWAPPED), program(CUBE_PLUS)];
    // The swapped cube has the same values, so the same line and the same child: measured once.
    let opt = options(4, 64, linearize());
    let mut state = parents(&cut, &opt, &genomes);
    let before = state.budget.used();
    linearize_round(&mut state, &cut, &opt, &mut || true).unwrap();
    let outcomes: Vec<&str> = state.moves.iter().map(|m| m.outcome.as_str()).collect();
    assert_eq!(
        outcomes.iter().filter(|o| **o == DUPLICATE).count(),
        1,
        "{outcomes:?}"
    );
    let duplicate = state.moves.iter().find(|m| m.outcome == DUPLICATE).unwrap();
    assert!(duplicate.measurement.is_none());
    assert!(state
        .moves
        .iter()
        .any(|m| m.outcome == MEASURED && m.program == duplicate.program));
    let measured = outcomes.iter().filter(|o| **o == MEASURED).count();
    assert_eq!(measured, 3);
    assert_eq!(state.budget.used() - before, measured);
    assert_eq!(state.budget.used(), state.observations.len());
    let distinct: BTreeSet<&str> = state
        .observations
        .iter()
        .map(|o| o.observation.genome.as_str())
        .collect();
    assert_eq!(distinct.len(), state.observations.len());
    // One evaluation left for a parent with two offers: the first is measured, the second
    // records the spent allowance, and the pass ends.
    let plus = [program(CUBE_PLUS)];
    let opt = options(4, plus.len() + 1, linearize());
    let mut state = parents(&cut, &opt, &plus);
    linearize_round(&mut state, &cut, &opt, &mut || true).unwrap();
    let outcomes: Vec<&str> = state.moves.iter().map(|m| m.outcome.as_str()).collect();
    assert_eq!(outcomes, [MEASURED, REFUSED, NO_ALLOWANCE]);
    assert!(state.moves[2].program.is_some() && state.moves[2].measurement.is_none());
    assert_eq!(state.budget.remaining(), 0);
    assert_eq!(state.budget.used(), state.observations.len());
    // None left: no proposal at all.
    let opt = options(4, plus.len(), linearize());
    let mut state = parents(&cut, &opt, &plus);
    linearize_round(&mut state, &cut, &opt, &mut || true).unwrap();
    assert!(state.moves.is_empty());
    // The guard refuses the first parent: nothing is recorded or spent.
    let opt = options(4, 64, linearize());
    let mut state = parents(&cut, &opt, &plus);
    let used = state.budget.used();
    linearize_round(&mut state, &cut, &opt, &mut || false).unwrap();
    assert!(state.moves.is_empty());
    assert_eq!(state.budget.used(), used);
    // The guard admits the parent and refuses its first measurement: every offer of that parent
    // is recorded unstarted, and the pass ends.
    let mut state = parents(&cut, &opt, &plus);
    let mut calls = 0;
    linearize_round(&mut state, &cut, &opt, &mut || {
        calls += 1;
        calls == 1
    })
    .unwrap();
    let outcomes: Vec<&str> = state.moves.iter().map(|m| m.outcome.as_str()).collect();
    assert_eq!(outcomes, [NOT_STARTED, REFUSED, NOT_STARTED]);
    assert_eq!(calls, 2);
    assert_eq!(state.budget.used(), used);
    // Only `top_k` parents are read: the exact cube ranks first.
    let opt = options(1, 64, linearize());
    let mut state = parents(&cut, &opt, &[program(CUBE_PLUS), program(CUBE)]);
    linearize_round(&mut state, &cut, &opt, &mut || true).unwrap();
    let read: BTreeSet<&str> = state
        .moves
        .iter()
        .map(|m| m.parent_sha256.as_str())
        .collect();
    assert_eq!(
        read,
        BTreeSet::from([crate::sha256_hex(program(CUBE).as_bytes()).as_str()])
    );
}

// Runs through the published route.

/// The `cli_tests` seed program, which the rewrite pass reduces to `c0`.
const ONE_TIMES: &str = "(pipe X (derive expr=\"1*c0\") (ridge))";
/// A seed whose 6-node expression is a curve the linearization pass can replace by its line. It
/// reads `c1`, which takes three values on this dataset; `c0` takes two, too few for a line's
/// minimax fit.
const SINE: &str = "(sin((0.5*c1))*3.0)";

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

/// One named edit of the records at `measured` (an offered, measured line) and `refused` (a
/// refused line); `other` is another measured program's sha256.
fn tamper(moves: &mut Vec<MoveRecord>, measured: usize, refused: usize, label: &str, other: &str) {
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
fn moves_rewrite() -> Vec<StageRewrite> {
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
    let note = include_str!("../../../../../../docs/internal/kernel2/b19-minimax-linearization.md");
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

/// `verify_run`'s refusal of `path`, which must name `expected`.
fn refused_with(path: &Path, label: &str, expected: &str) -> String {
    let error = format!(
        "{:#}",
        publication::verify_run(path).map(|_| ()).expect_err(label)
    );
    assert!(error.contains(expected), "{label}: {error}");
    format!("{label}: {error}")
}

/// The bound a fit record's numbers give, as `LinearFit::bound` computes it.
fn bound_of(fit: &FitRecord) -> f64 {
    fit.max_error * (1.0 + 1e-6) + 64.0 * f64::EPSILON * fit.max_abs
}

/// The records `edit` leaves, published under a rehashed digest.
fn published_edit(
    output: &Path,
    receipt: &Receipt,
    engine: &EngineRecord,
    moves: &[MoveRecord],
    edit: impl FnOnce(&mut Vec<MoveRecord>),
) -> PathBuf {
    let mut edited = moves.to_vec();
    edit(&mut edited);
    assert_ne!(edited, moves, "the edit was not applied");
    let bytes = serde_json::to_vec_pretty(&edited).unwrap();
    rewrite(
        output,
        receipt,
        Some(bytes.as_slice()),
        Some(digested(engine, &bytes)),
    )
}

/// The cubic `0.5 x³` over `c1`, which takes three values on this dataset: a second seed with a
/// subtree the pass considers, so round 0's pass reads two parents.
const CUBE_C1: &str = "(c1*c1)*(c1*0.5)";

#[test]
fn verify_refuses_inflated_and_hidden_fits_and_moved_parent_groups() {
    let temp = tempfile::tempdir().unwrap();
    let opt = run_options(
        temp.path(),
        &[ONE_TIMES.into(), program(SINE), program(CUBE_C1)],
        40,
        four_parents(),
        linearize(),
    );
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
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
    let mut errors = Vec::new();
    fn fit_of(m: &mut MoveRecord) -> &mut LinearizationRecord {
        m.linearization.as_mut().unwrap()
    }

    // A fit's largest |f| past the evaluator's range, the bound recomputed to match: the line,
    // the program and the outcome are unchanged, so only the range check refuses it.
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        let record = fit_of(&mut m[measured]);
        let fit = record.fit.as_mut().unwrap();
        fit.max_abs = 2e12;
        record.bound = Some(bound_of(fit));
    });
    errors.push(refused_with(
        &path,
        "max|f| past the range",
        "a recorded fit disagrees with its stop",
    ));
    restore();
    // An exact-fit claim, whose closing test `max_error <= 4 eps max|f|` an inflated max|f| meets.
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        let record = fit_of(&mut m[measured]);
        let fit = record.fit.as_mut().unwrap();
        fit.stop = "exact_fit".into();
        fit.max_abs = (fit.max_error / (4.0 * f64::EPSILON) * 1.5).max(2e12);
        record.bound = Some(bound_of(fit));
    });
    errors.push(refused_with(
        &path,
        "exact fit with an inflated max|f|",
        "a recorded fit disagrees with its stop",
    ));
    restore();
    // A measured line's error made negative (B19 wiring review 01,
    // B19-R01-NEGATIVE-LOWERED-ERROR): within any bound, yet no largest absolute residual. JSON
    // carries no NaN or infinity; negative zero is the one other sign it can carry.
    for (label, error) in [
        ("a negative line error", -1.0),
        ("a negative zero line error", -0.0),
    ] {
        let path = published_edit(output, &receipt, &engine, &moves, |m| {
            fit_of(&mut m[measured]).lowered_error = Some(error);
        });
        errors.push(refused_with(
            &path,
            label,
            "a recorded line error is not a finite nonnegative residual",
        ));
        restore();
    }

    // A refused line's fit hidden behind the faulting-input refusal, while the stage's other
    // subtrees keep their own records.
    let group = |at: usize| -> Vec<usize> {
        let m = &moves[at];
        (0..moves.len())
            .filter(|&i| {
                moves[i].id == m.id
                    && moves[i].round == m.round
                    && moves[i].parent_sha256 == m.parent_sha256
                    && line(&moves[i]).stage == line(m).stage
            })
            .collect()
    };
    assert!(group(refused).len() > 1, "the stage has one subtree");
    fn hide(m: &mut MoveRecord, refusal: &str) {
        let record = fit_of(m);
        record.fit = None;
        record.bound = None;
        record.line_nodes = None;
        record.lowered_error = None;
        record.expression = None;
        record.refusal = Some(refusal.into());
    }
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        hide(&mut m[refused], PREFIX_FAULTS)
    });
    errors.push(refused_with(
        &path,
        "a fit hidden behind a faulting input",
        "a faulting training input refuses only some of its stage's subtrees",
    ));
    restore();

    // Whole parent groups: removed, swapped with the next, or repeated.
    let starts: Vec<usize> = (0..moves.len())
        .filter(|&i| {
            moves[i].id == MINIMAX_LINEARIZE
                && (i == 0
                    || moves[i - 1].parent_sha256 != moves[i].parent_sha256
                    || moves[i - 1].round != moves[i].round
                    || moves[i - 1].id != moves[i].id)
        })
        .collect();
    let next_in_round = starts
        .windows(2)
        .find(|w| moves[w[0]].round == moves[w[1]].round)
        .map(|w| (w[0], w[1]))
        .expect("a round in which the pass read two parents");
    let (first, second) = next_in_round;
    let end = (second..moves.len())
        .find(|&i| {
            moves[i].parent_sha256 != moves[second].parent_sha256
                || moves[i].round != moves[second].round
                || moves[i].id != moves[second].id
        })
        .unwrap_or(moves.len());
    let off = "move records do not follow the ranked parents";
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        m.drain(first..second);
    });
    errors.push(refused_with(&path, "a parent's group removed", off));
    restore();
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        let later: Vec<MoveRecord> = m.drain(second..end).collect();
        for (k, record) in later.into_iter().enumerate() {
            m.insert(first + k, record);
        }
    });
    errors.push(refused_with(&path, "two parents' groups swapped", off));
    restore();
    // Repeated after the next parent's group (next to itself it would read as one group, which
    // the subtree count refuses).
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        let copy: Vec<MoveRecord> = m[first..second].to_vec();
        for (k, record) in copy.into_iter().enumerate() {
            m.insert(end + k, record);
        }
    });
    errors.push(refused_with(&path, "a parent's group repeated", off));
    restore();

    // The documented limits: without the training rows, a closed fit scaled consistently (its
    // errors and max|f| ten times over, the bound recomputed) and a refused line's fit hidden as
    // an unsampled subtree both re-derive the same decisions and are accepted.
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        let record = fit_of(&mut m[measured]);
        let fit = record.fit.as_mut().unwrap();
        fit.max_error *= 10.0;
        fit.levelled *= 10.0;
        fit.max_abs *= 10.0;
        record.bound = Some(bound_of(fit));
    });
    publication::verify_run(&path).unwrap();
    restore();
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        hide(&mut m[refused], LINEARIZE_UNSAMPLED)
    });
    publication::verify_run(&path).unwrap();
    restore();
    // A third (review 01): an offered line's error edited to zero, under its fit's bracket, keeps
    // the decision and is accepted; only a negative or nonfinite one is refused.
    let path = published_edit(output, &receipt, &engine, &moves, |m| {
        let record = fit_of(&mut m[measured]);
        assert!(record.fit.as_ref().unwrap().levelled > 0.0);
        record.lowered_error = Some(0.0);
    });
    publication::verify_run(&path).unwrap();
    restore();
    publication::verify_run(&output.join("receipt.json")).unwrap();
    let root = output.display().to_string();
    for error in errors {
        lint(&error.replace(&root, "<run>"));
    }
}

/// `check` reads the linearization pass's parents back from evidence, which is untrusted, so
/// `offers` and `assemble` (behind `propose` and `rederive`) parse them within
/// `parse_untrusted_pipeline`'s bounds (B17 review 02's class): the pipeline form only, and each
/// expression's nesting bytes counted before it is parsed. Unary plus never reaches the parsed
/// tree, so a parent at the limit offers what its plain form offers; one byte more offers nothing,
/// and `propose` refuses it unparsed.
#[test]
fn linearization_parents_are_bounded_before_they_are_parsed() {
    use crate::kernel2::refinement::MAX_PARENT_EXPRESSION_NESTING;
    let cut = cubic_cut();
    let opt = options(4, 64, linearize());
    let width = cut.train_x[0].len();
    // Built directly: `program` would print the padding away.
    let parent = |source: &str| format!("(pipe X (derive expr=\"{source}\") (ridge))");
    let held = CUBE
        .bytes()
        .filter(|b| matches!(b, b'(' | b'+' | b'-' | b'*' | b'/' | b'^'))
        .count();
    let room = MAX_PARENT_EXPRESSION_NESTING - held;
    let plus = |n: usize| parent(&format!("{}{CUBE}", "+".repeat(n)));
    assert!(linearize::offers(&opt.evaluator, &parent(CUBE)).unwrap());
    assert!(linearize::offers(&opt.evaluator, &plus(room)).unwrap());
    assert!(!linearize::offers(&opt.evaluator, &plus(room + 1)).unwrap());
    let error = linearize::propose(&opt.evaluator, &plus(room + 1), &cut, width).unwrap_err();
    assert!(
        format!("{error:#}").contains("nests too deeply"),
        "{error:#}"
    );
    let closed_form = format!("(sym raw \"{}c0\")", "+".repeat(room + 1));
    assert!(!linearize::offers(&opt.evaluator, &closed_form).unwrap());
    let error = linearize::propose(&opt.evaluator, &closed_form, &cut, width).unwrap_err();
    assert!(format!("{error:#}").contains("not a pipeline"), "{error:#}");
}

// Fix round r01 (B19 wiring review 01): repeated refused subtrees through publication.

/// A one-feature dataset in `dir` whose feature cycles through `-1e-310`, `0` and `1e-310` (75
/// training and 15 test rows, the target cycling through 0 to 6), with the pmlb-229-pwlinear
/// fixture's metadata rewritten for it. Every line fitted on these subnormal values has a slope
/// past the finite range. The review's fixture.
fn subnormal_dataset(dir: &Path) -> String {
    let csv = |rows: std::ops::Range<usize>| -> String {
        rows.map(|i| format!("{},{}\n", ["-1e-310", "0", "1e-310"][i % 3], i % 7))
            .collect()
    };
    let (train, test) = (csv(0..75), csv(75..90));
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("train.csv"), &train).unwrap();
    std::fs::write(dir.join("test.csv"), &test).unwrap();
    let scaffold = std::fs::read(Path::new(&dataset()).join("metadata.json")).unwrap();
    let mut meta: serde_json::Value = serde_json::from_slice(&scaffold).unwrap();
    let hash = crate::sha256_hex(format!("{train}{test}").as_bytes());
    meta["id"] = hash[..16].into();
    meta["dataset_hash"] = hash[..16].into();
    meta["canonical_sha256"] = hash.into();
    meta["display_name"] = "subnormal linearization fixture".into();
    meta["shape"] = serde_json::json!({
        "rows": 90, "features": 1, "categorical_features": 0, "columns": 2
    });
    for (name, bytes, rows) in [("train", &train, 75), ("test", &test, 15)] {
        meta["files"][name] = serde_json::json!({
            "sha256": crate::sha256_hex(bytes.as_bytes()),
            "bytes": bytes.len(),
            "rows": rows,
            "columns": 2,
            "has_header": false,
        });
    }
    meta["split"] = serde_json::json!({
        "method": "deterministic test fixture", "seed": 0, "train_rows": 75, "test_rows": 15
    });
    meta["provenance"]["dataset_id"] = "subnormal-linearization".into();
    meta["provenance"]["citation"] =
        "synthetic test fixture; metadata scaffold from the PMLB fixture, not PMLB data".into();
    meta["collections"] = serde_json::json!([]);
    meta["ingest"] = serde_json::Value::Null;
    std::fs::write(
        dir.join("metadata.json"),
        serde_json::to_vec_pretty(&meta).unwrap(),
    )
    .unwrap();
    dir.display().to_string()
}

/// A fit refused for a coefficient past the finite range is published without its coefficients
/// and read back as NaN (B19 wiring review 01, B19-R01-NAN-REFUSAL-ROUNDTRIP). The two equal
/// children of `(sin(c0)+c0)*(sin(c0)+c0)` on the subnormal dataset are each refused so, and the
/// untouched run verifies: a repeated subtree's records compare bit for bit. A finite pair given
/// to one of them is still refused.
#[test]
fn repeated_subtrees_refused_for_their_coefficients_publish_and_verify() {
    let temp = tempfile::tempdir().unwrap();
    let data = subnormal_dataset(&temp.path().join("dataset"));
    let opt = run_options_on(
        data,
        temp.path(),
        &[program("(sin(c0)+c0)*(sin(c0)+c0)")],
        40,
        four_parents(),
        linearize(),
    );
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
    let path = output.join("receipt.json");
    assert_eq!(publication::verify_run(&path).unwrap(), receipt);
    let moves = published_moves(&opt);
    // The children at paths [0] and [1] of one parent's stage, each refused for its coefficients.
    let twin = |at: &[usize]| {
        moves
            .iter()
            .position(|m| m.id == MINIMAX_LINEARIZE && line(m).path == at)
            .unwrap()
    };
    let (left, right) = (twin(&[0]), twin(&[1]));
    assert_eq!(right, left + 1);
    for m in [&moves[left], &moves[right]] {
        let record = line(m);
        assert_eq!(record.subtree, "(sin(c0)+c0)");
        assert_eq!(record.refusal.as_deref(), Some(LINEARIZE_NON_FINITE));
        assert_eq!(m.outcome, REFUSED);
        assert!(record.fit.as_ref().unwrap().coefficients.is_none());
    }
    // A finite pair for one of the two: the twins record other samples.
    let engine = receipt.search.engine_selection.clone().unwrap();
    let edited = published_edit(output, &receipt, &engine, &moves, |m| {
        let record = m[right].linearization.as_mut().unwrap();
        record.fit.as_mut().unwrap().coefficients = Some([0.0, 1.0]);
    });
    let error = refused_with(
        &edited,
        "a finite pair for one twin",
        "a repeated subtree records other samples",
    );
    lint(&error.replace(&output.display().to_string(), "<run>"));
}

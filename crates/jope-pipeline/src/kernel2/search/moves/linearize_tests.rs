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

mod bounds;
mod edge_cases;
mod run;
mod selection;
mod verification;

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

mod cli_contract;
mod record_validation;
mod rewrite_faults;
mod rewrite_replay;
mod rewrite_scores;
mod stop_bounds;

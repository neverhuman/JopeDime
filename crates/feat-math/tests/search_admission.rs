//! The search's fault policies (kernel2 WP-1.5), end to end through the public entry points.
//!
//! - `Legacy` reproduces the frozen seeded searches of `compat_freeze` (`search_runs.v1.json`)
//!   line for line, and so does every non-legacy policy on those searches: none of their
//!   candidates has a `map` or `derive` stage, so none has a fault, and every score, rank and
//!   random draw is the legacy one.
//! - The non-legacy policies reject a candidate whose scope check fails and keep it out of the
//!   ranking; `Penalize` ranks a fault-free program above a faulting one of equal loss, by the
//!   penalty or, at zero weights, by the tie-break; a program without a fault scores bit for bit
//!   as under `Legacy`; and the penalty is the documented formula on the rates the public
//!   telemetry gives.
//! - They score on the CPU route: a GPU backend is never called, and exact reuse stays off.
//! - What they decide does not depend on the thread count.
//!
//! The informational run on the test datasets, whose rates and invalid-candidate counts the WP-1.5
//! receipt records, is ignored by default.

use feat_math::evolve::{
    evolve, evolve_lite_race, evolve_lite_race_with_exact_initial_population,
    evolve_lite_race_with_gpu_policy, evolve_lite_race_with_gpu_policy_and_exact_surrogate_dedup,
    rung0_trace_identity, EvolveConfig, FaultPolicy, FitnessMode, GenStats, GpuRidgeScoreBatch,
    GpuRidgeScorePolicy, GpuRidgeScoreProblem, Observer, RaceOutcome,
    Rung0ExactSurrogateDedupPolicy, Rung0FoldRoute, Scored, RACE_RUNG_FOLDS,
    RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
};
use feat_math::kernel::admission::{FaultTotals, Rejection};
use feat_math::kernel::scope::validate_scope;
use feat_math::kernel::telemetry::{apply_op_traced, fit_traced, predict_traced};
use feat_math::math_expr::{FaultClass, MathExpr};
use feat_math::ops::FittedOp;
use feat_math::transpile::Emitted;
use feat_math::{GenomeIr, Pcg32};
use sha2::{Digest, Sha256};
use std::sync::Mutex;

type Rows = Vec<Vec<f64>>;

/// The penalty the plan names for kernel2 profiles: domain weight 0.05, no saturation weight,
/// ties broken by faulted cells.
const PENALIZE: FaultPolicy = FaultPolicy::Penalize {
    lambda_domain: 0.05,
    lambda_saturation: 0.0,
    tie_break: true,
};

#[test]
fn one_cell_with_two_classes_of_a_severity_is_penalized_once() {
    let source = "log(col(0))/col(1)";
    let op = FittedOp::DeriveExpr {
        source: source.to_string(),
        expr: MathExpr::parse(source).unwrap(),
        append: false,
    };
    let (_, stage) = apply_op_traced(&op, &vec![vec![0.0, 0.0], vec![2.0, 1.0]]);
    assert_eq!(stage.evaluated_cells, 2);
    assert_eq!(stage.faulted(FaultClass::LogDomain), 1);
    assert_eq!(stage.faulted(FaultClass::DivNearZero), 1);
    assert!(!stage.constant_output);
    let mut totals = FaultTotals::default();
    totals.add_stage(&stage);
    assert_eq!(totals.domain_cells, 1);
    assert_eq!(totals.domain_rate().to_bits(), 0.5_f64.to_bits());
    assert_eq!(totals.penalize(1.0, 0.05, 0.0), Some(0.975));

    let source = "exp(col(0))";
    let op = FittedOp::DeriveExpr {
        source: source.to_string(),
        expr: MathExpr::parse(source).unwrap(),
        append: false,
    };
    let (_, stage) = apply_op_traced(&op, &vec![vec![31.0], vec![0.0]]);
    assert_eq!(stage.faulted(FaultClass::ExpArgClamped), 1);
    assert_eq!(stage.faulted(FaultClass::MagnitudeClamped), 1);
    assert!(!stage.constant_output);
    let mut totals = FaultTotals::default();
    totals.add_stage(&stage);
    assert_eq!(totals.saturation_cells, 1);
    assert_eq!(totals.saturation_rate().to_bits(), 0.5_f64.to_bits());
    assert_eq!(totals.penalize(1.0, 0.0, 0.02), Some(0.99));
}

fn penalize(lambda_domain: f64, lambda_saturation: f64, tie_break: bool) -> FaultPolicy {
    FaultPolicy::Penalize {
        lambda_domain,
        lambda_saturation,
        tie_break,
    }
}

/// Reads column 12 of a four-column input: its scope check fails at stage 0.
const OUT_OF_SCOPE: &str = "(pipe X (derive expr=\"col(12)\") (ridge))";
/// In scope as written, `rff_cos n=128` giving 128 columns; out of scope as the search fits it,
/// with `n` capped at 64.
const NARROWED: &str = "(pipe X (rff_cos n=128) (derive expr=\"col(100)\") (ridge))";
/// On negative data every row takes `log`'s domain fallback, so the derived column is constant.
const LOG_NEGATIVE: &str = "(pipe X (derive expr=\"log(col(0))\") (ridge))";
/// Fault-free on any finite data.
const LOG1P_ABS: &str = "(pipe X (derive expr=\"log1p(abs(col(0)))\") (ridge))";
/// On standardized `toy_data` some rows take `log`'s domain fallback and some clamp `exp`: both
/// rates are between 0 and 1. The expression reads the output of a fitted stage, so a fold's
/// training rows and its held-out rows fault at different rates: the held-out rows are
/// standardized with the training rows' statistics.
const PARTIAL_FAULTS: &str =
    "(pipe X (standardize) (derive expr=\"log(col(0)) + exp(col(1)*20)\") (ridge))";
/// Expression programs without a fault on `toy_data`, and one without an expression.
const FAULT_FREE: [&str; 3] = [
    "(pipe X (derive expr=\"col(0)*col(1) + sin(col(2))\") (ridge))",
    "(pipe X (map expr=\"tanh(x) + 0.5*x\" append=true) (histgbm rounds=20))",
    "(pipe X (standardize) (ridge))",
];

fn canonical(source: &str) -> String {
    GenomeIr::parse(source)
        .unwrap_or_else(|error| panic!("{source}: {error}"))
        .canonical()
}

/// `toy_data` of `compat_freeze.rs`, itself copied from `tests/golden.rs`: the data of the
/// frozen searches.
fn toy_data(n: usize, cols: usize, seed: u64) -> (Rows, Vec<f64>) {
    let mut rng = Pcg32::new(seed, 17);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..n {
        let row: Vec<f64> = (0..cols).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
        let target = 1.7 * row[0] - 0.9 * row[1]
            + 0.5 * (3.0 * row[2]).sin()
            + 0.3 * row[3] * row[4]
            + 0.1 * rng.gaussian();
        x.push(row);
        y.push(target);
    }
    (x, y)
}

/// Four columns in [-2, 2), for the programs that read past them.
fn four_columns(n: usize, seed: u64) -> (Rows, Vec<f64>) {
    let mut rng = Pcg32::new(seed, 23);
    let x: Rows = (0..n)
        .map(|_| (0..4).map(|_| rng.next_f64() * 4.0 - 2.0).collect())
        .collect();
    let y = x
        .iter()
        .map(|r| 1.3 * r[0] - 0.7 * r[1] + 0.4 * r[2] * r[3] + 0.05 * rng.gaussian())
        .collect();
    (x, y)
}

/// Column 0 in (-2, -1], so `log(col(0))` takes its domain fallback on every row, and a constant
/// target, so R² is 0 for every program and fold: `LOG_NEGATIVE` and `LOG1P_ABS` tie on loss
/// exactly.
fn equal_loss_data(n: usize) -> (Rows, Vec<f64>) {
    let mut rng = Pcg32::new(29, 5);
    let x: Rows = (0..n)
        .map(|_| {
            let mut row: Vec<f64> = (0..4).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
            row[0] = -1.0 - rng.next_f64();
            row
        })
        .collect();
    (x, vec![1.0; n])
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

// The record formats of `compat_freeze.rs` (`ranked_line`, and the `raced` and `summary` lines of
// `lite_race_run`), so that a run compares with `search_runs.v1.json` line for line.

fn hex_bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn bits_list(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| hex_bits(*value))
        .collect::<Vec<_>>()
        .join(" ")
}

fn ranked_line(scored: &Scored) -> String {
    format!(
        "{} | cv_mean={} | folds={}",
        scored.dsl,
        hex_bits(scored.cv_mean),
        bits_list(&scored.fold_scores)
    )
}

fn or_dash<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "-".to_string(), |value| value.to_string())
}

fn index_list_digest(values: &[usize]) -> String {
    let text = values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    let digest: String = Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("{} {}", values.len(), &digest[..16])
}

fn raced_lines(outcome: &RaceOutcome) -> Vec<String> {
    outcome
        .candidates
        .iter()
        .map(|candidate| {
            let rungs: Vec<String> = candidate
                .rung_fold_scores
                .iter()
                .map(|scores| scores.as_deref().map_or_else(|| "-".to_string(), bits_list))
                .collect();
            let routes = candidate.r0_fold_routes.as_ref().map_or_else(
                || "-".to_string(),
                |routes| {
                    routes
                        .iter()
                        .map(|receipt| format!("{:?}", receipt.route))
                        .collect::<Vec<_>>()
                        .join(",")
                },
            );
            let reuse = candidate
                .r0_exact_surrogate_dedup
                .as_ref()
                .map_or_else(|| "-".to_string(), |receipt| format!("{receipt:?}"));
            format!(
                "{} | r0={} | r1={} | r2={} | seq={} generation={} slot={} | routes={} | reuse={} | seen={} first={} last={}",
                candidate.dsl,
                rungs[0],
                rungs[1],
                rungs[2],
                or_dash(candidate.r0_seq),
                or_dash(candidate.r0_generation),
                or_dash(candidate.r0_generation_slot),
                routes,
                reuse,
                candidate.recurrence_count,
                candidate.first_generation,
                candidate.last_generation
            )
        })
        .collect()
}

fn summary_line(outcome: &RaceOutcome) -> String {
    format!(
        "rung_evals {} {} {} | target_hit {} | rung0_rows {} | rung0_assignments {}",
        outcome.rung_evals[0],
        outcome.rung_evals[1],
        outcome.rung_evals[2],
        outcome.target_hit,
        index_list_digest(&outcome.rung0_rows),
        index_list_digest(&outcome.rung0_assignments)
    )
}

/// One seeded search as `compat_freeze` records it: ranked lines, and for the lite race the raced
/// lines and the summary.
#[derive(Debug, PartialEq)]
struct Run {
    ranked: Vec<String>,
    raced: Vec<String>,
    summary: String,
}

fn race_run(outcome: &RaceOutcome) -> Run {
    Run {
        ranked: outcome.ranked.iter().map(ranked_line).collect(),
        raced: raced_lines(outcome),
        summary: summary_line(outcome),
    }
}

/// The frozen searches of `search_runs.v1.json`, in its order.
fn frozen_runs() -> Vec<Run> {
    let path = format!(
        "{}/tests/fixtures/compat_freeze/search_runs.v1.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{path}: {error}"));
    let json: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|error| panic!("{path}: {error}"));
    let strings = |value: &serde_json::Value| -> Vec<String> {
        value.as_array().map_or_else(Vec::new, |values| {
            values
                .iter()
                .map(|value| value.as_str().expect("a string line").to_string())
                .collect()
        })
    };
    json["runs"]
        .as_array()
        .expect("runs")
        .iter()
        .map(|run| Run {
            ranked: strings(&run["ranked"]),
            raced: strings(&run["raced"]),
            summary: run["summary"].as_str().unwrap_or_default().to_string(),
        })
        .collect()
}

/// The frozen searches under `policy`: the surrogate `evolve()`, then the lite race with
/// `rank_robust` off and on, with `compat_freeze`'s inputs. No candidate there has an expression
/// stage, so the policy must report no rejection and no rate.
fn seeded_runs(policy: FaultPolicy) -> Vec<Run> {
    let (x, y) = toy_data(96, 5, 42);
    let metric = |truth: &[f64], predicted: &[f64]| feat_math::r2(truth, predicted);
    let config = EvolveConfig {
        pop: 16,
        gens: 3,
        folds: 3,
        seed: 12345,
        max_transforms: 3,
        fault_policy: policy,
        ..EvolveConfig::default()
    };
    let log = Log::default();
    let ranked = evolve(&x, &y, &metric, &config, &log);
    log.assert_nothing_reported();
    let mut runs = vec![Run {
        ranked: ranked.iter().map(ranked_line).collect(),
        raced: Vec::new(),
        summary: String::new(),
    }];
    let (x, y) = toy_data(160, 5, 42);
    for rank_robust in [false, true] {
        let config = EvolveConfig {
            pop: 16,
            gens: 3,
            folds: 3,
            seed: 41,
            max_transforms: 3,
            time_budget_secs: 0.0,
            fitness: FitnessMode::LiteRace,
            rank_robust,
            fault_policy: policy,
            ..EvolveConfig::default()
        };
        let log = Log::default();
        let outcome = evolve_lite_race(&x, &y, &feat_math::r2, &config, &log);
        log.assert_nothing_reported();
        runs.push(race_run(&outcome));
    }
    runs
}

/// A generation's fault counters, floats as bits, and how many candidates it ranked.
#[derive(Clone, Debug, PartialEq)]
struct Counters {
    gen: usize,
    hard_rejected_share: u64,
    hard_rejected: u64,
    pop_domain_rate: u64,
    elite_domain_rate: u64,
    best_cv: u64,
    ranked: usize,
}

/// Records what a search reports: every rejection with its reason, and each generation's
/// counters.
#[derive(Default)]
struct Log {
    rejections: Mutex<Vec<(String, String)>>,
    generations: Mutex<Vec<Counters>>,
}

impl Observer for Log {
    fn on_generation(&self, stats: &GenStats) {
        self.generations.lock().unwrap().push(Counters {
            gen: stats.gen,
            hard_rejected_share: stats.hard_rejected_share.to_bits(),
            hard_rejected: stats.hard_rejected,
            pop_domain_rate: stats.pop_domain_rate.to_bits(),
            elite_domain_rate: stats.elite_domain_rate.to_bits(),
            best_cv: stats.best_cv.to_bits(),
            ranked: stats.pop_fitness.len(),
        });
    }
    fn on_improvement(&self, _: &Scored, _: &Emitted, _: &GenStats) {}
    fn on_reject(&self, dsl: &str, reason: &Rejection) {
        self.rejections
            .lock()
            .unwrap()
            .push((dsl.to_string(), reason.to_string()));
    }
}

impl Log {
    fn rejections(&self) -> Vec<(String, String)> {
        self.rejections.lock().unwrap().clone()
    }

    fn generations(&self) -> Vec<Counters> {
        self.generations.lock().unwrap().clone()
    }

    fn assert_nothing_reported(&self) {
        assert_eq!(self.rejections(), Vec::new());
        let generations = self.generations();
        assert!(!generations.is_empty());
        for counters in generations {
            assert_eq!(
                (
                    counters.hard_rejected_share,
                    counters.hard_rejected,
                    counters.pop_domain_rate,
                    counters.elite_domain_rate
                ),
                (0.0_f64.to_bits(), 0, 0.0_f64.to_bits(), 0.0_f64.to_bits()),
                "generation {}",
                counters.gen
            );
        }
    }
}

fn scores_of(outcome: &RaceOutcome, key: &str) -> [Option<Vec<f64>>; 3] {
    outcome
        .candidates
        .iter()
        .find(|candidate| candidate.dsl == key)
        .unwrap_or_else(|| panic!("{key} was not raced"))
        .rung_fold_scores
        .clone()
}

/// One program's rung-0 evaluation, replayed through the public entry points: its fold scores,
/// and the fault totals of its training passes alone and of every pass.
struct Replay {
    scores: Vec<f64>,
    training: FaultTotals,
    all: FaultTotals,
}

/// Replays rung 0 of `outcome` for `dsl`: its surrogate program (`rung0_trace_identity`), fitted
/// with `fit_traced` and predicted with `predict_traced` on the published rung-0 rows and folds.
/// `None` when a fold cannot be fitted or predicted.
fn replay_rung0(outcome: &RaceOutcome, x: &Rows, y: &[f64], dsl: &str) -> Option<Replay> {
    let (surrogate, _) = rung0_trace_identity(dsl).ok()?;
    let program = GenomeIr::parse(&surrogate).ok()?;
    let mut replay = Replay {
        scores: Vec::new(),
        training: FaultTotals::default(),
        all: FaultTotals::default(),
    };
    for fold in 0..RACE_RUNG_FOLDS[0] {
        let (mut xtr, mut ytr, mut xte, mut yte) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (&row, &assigned) in outcome.rung0_rows.iter().zip(&outcome.rung0_assignments) {
            if assigned == fold {
                xte.push(x[row].clone());
                yte.push(y[row]);
            } else {
                xtr.push(x[row].clone());
                ytr.push(y[row]);
            }
        }
        let (model, training) = fit_traced(&program, &xtr, &ytr).ok()?;
        replay.training.add_pass(&training);
        replay.all.add_pass(&training);
        let (predictions, held_out) = predict_traced(&model, &xte).ok()?;
        replay.all.add_pass(&held_out);
        replay.scores.push(feat_math::r2(&yte, &predictions));
    }
    Some(replay)
}

fn race_config(pop: usize, gens: usize, seed: u64, policy: FaultPolicy) -> EvolveConfig {
    EvolveConfig {
        pop,
        gens,
        seed,
        max_transforms: 3,
        fitness: FitnessMode::LiteRace,
        fault_policy: policy,
        ..EvolveConfig::default()
    }
}

#[path = "search_admission/datasets.rs"]
mod datasets;
#[path = "search_admission/policy.rs"]
mod policy;
#[path = "search_admission/routing.rs"]
mod routing;

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
//! The informational run on the test datasets, whose rates the WP-1.5 receipt records, is
//! ignored by default.

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
use feat_math::kernel::telemetry::{fit_traced, predict_traced};
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

#[test]
fn legacy_reproduces_the_frozen_search_runs() {
    assert_eq!(seeded_runs(FaultPolicy::Legacy), frozen_runs());
}

#[test]
fn reject_hard_reproduces_the_frozen_search_runs() {
    assert_eq!(seeded_runs(FaultPolicy::RejectHard), frozen_runs());
}

#[test]
fn penalize_reproduces_the_frozen_search_runs() {
    assert_eq!(seeded_runs(PENALIZE), frozen_runs());
}

#[test]
fn other_penalty_settings_reproduce_the_frozen_search_runs() {
    let frozen = frozen_runs();
    for policy in [penalize(0.0, 0.0, true), penalize(0.05, 0.01, false)] {
        assert_eq!(seeded_runs(policy), frozen, "{policy:?}");
    }
}

#[test]
fn an_out_of_scope_candidate_is_rejected_and_never_ranked() {
    let (x, y) = four_columns(120, 3);
    let metric = |truth: &[f64], predicted: &[f64]| feat_math::r2(truth, predicted);
    let key = canonical(OUT_OF_SCOPE);
    let reason = "scope: stage 0: COLUMN_OOB: column index 12 out of bounds for width 4";

    // The surrogate search: generation 0 holds the champion, the seed and six starters.
    let config = |gens: usize, policy: FaultPolicy| EvolveConfig {
        pop: 8,
        gens,
        folds: 3,
        seed: 7,
        max_transforms: 3,
        seed_genomes: vec![OUT_OF_SCOPE.to_string()],
        fault_policy: policy,
        ..EvolveConfig::default()
    };
    let legacy = evolve(
        &x,
        &y,
        &metric,
        &config(1, FaultPolicy::Legacy),
        &Log::default(),
    );
    assert!(
        legacy.iter().any(|scored| scored.dsl == key),
        "Legacy scores the fallback zeros"
    );
    for policy in [FaultPolicy::RejectHard, PENALIZE] {
        let log = Log::default();
        let ranked = evolve(&x, &y, &metric, &config(3, policy), &log);
        assert!(!ranked.is_empty());
        assert!(ranked.iter().all(|scored| scored.dsl != key));
        assert_eq!(log.rejections(), vec![(key.clone(), reason.to_string())]);
        let generations = log.generations();
        assert_eq!(generations.len(), 3);
        assert!(generations
            .iter()
            .all(|counters| counters.hard_rejected == 1));
        assert_eq!(
            generations[0].hard_rejected_share,
            (1.0_f64 / 8.0).to_bits()
        );
    }

    // The lite race, from an exact generation 0.
    let initial = [
        OUT_OF_SCOPE,
        "(pipe X (standardize) (ridge))",
        "(pipe X (rank) (ridge))",
        "(pipe X (derive expr=\"col(3)\") (ridge))",
    ]
    .map(str::to_string);
    let race = |policy: FaultPolicy, log: &Log| {
        evolve_lite_race_with_exact_initial_population(
            &x,
            &y,
            &metric,
            &race_config(4, 1, 11, policy),
            log,
            &initial,
        )
        .unwrap()
    };
    let legacy = race(FaultPolicy::Legacy, &Log::default());
    assert!(scores_of(&legacy, &key)[0].is_some());
    assert_eq!(legacy.ranked.len(), 4);
    for policy in [FaultPolicy::RejectHard, PENALIZE] {
        let log = Log::default();
        let outcome = race(policy, &log);
        assert_eq!(scores_of(&outcome, &key), [None, None, None]);
        assert_eq!(outcome.ranked.len(), 3);
        assert!(outcome.ranked.iter().all(|scored| scored.dsl != key));
        assert_eq!(log.rejections(), vec![(key.clone(), reason.to_string())]);
        assert_eq!(log.generations()[0].hard_rejected, 1);
        assert_eq!(
            log.generations()[0].hard_rejected_share,
            (1.0_f64 / 4.0).to_bits()
        );
    }
}

/// The scope check sees the program as the search fits it, not as it was generated: the search
/// caps `rff_cos n=128` at 64 columns, so `col(100)` is in scope for the generated program and out
/// of scope for the fitted one. Both searches reject the candidate with the fitted program's
/// reason, which names width 64. A check of the generated program would have passed it, and the
/// fit would have met `col(100)` as a Hard fault instead.
#[test]
fn the_scope_check_sees_the_program_as_the_search_fits_it() {
    let (x, y) = four_columns(120, 3);
    let metric = |truth: &[f64], predicted: &[f64]| feat_math::r2(truth, predicted);
    let key = canonical(NARROWED);
    let reason = "stage 1: COLUMN_OOB: column index 100 out of bounds for width 64";
    assert!(validate_scope(&GenomeIr::parse(NARROWED).unwrap(), 4).is_ok());
    // The rung-0 program, as the search fits it: out of scope, and a traced fit of it meets the
    // Hard fault that the check spares the search.
    let (surrogate, _) = rung0_trace_identity(NARROWED).unwrap();
    let fitted = GenomeIr::parse(&surrogate).unwrap();
    assert_eq!(validate_scope(&fitted, 4).unwrap_err().to_string(), reason);
    let (_, training) = fit_traced(&fitted, &x, &y).unwrap();
    let mut totals = FaultTotals::default();
    totals.add_pass(&training);
    assert_eq!(
        Rejection::HardFault(totals.hard_classes).to_string(),
        "hard fault: {COLUMN_OOB}"
    );

    let rejection = vec![(key.clone(), format!("scope: {reason}"))];
    for policy in [FaultPolicy::RejectHard, PENALIZE] {
        // The surrogate search: generation 0 holds the champion, the program and six starters.
        let log = Log::default();
        let config = EvolveConfig {
            pop: 8,
            gens: 1,
            folds: 3,
            seed: 7,
            max_transforms: 3,
            seed_genomes: vec![NARROWED.to_string()],
            fault_policy: policy,
            ..EvolveConfig::default()
        };
        let ranked = evolve(&x, &y, &metric, &config, &log);
        assert!(ranked.iter().all(|scored| scored.dsl != key));
        assert_eq!(log.rejections(), rejection, "{policy:?}");

        // The lite race, from an exact generation 0.
        let log = Log::default();
        let initial = [NARROWED, "(pipe X (standardize) (ridge))"].map(str::to_string);
        let outcome = evolve_lite_race_with_exact_initial_population(
            &x,
            &y,
            &metric,
            &race_config(2, 1, 11, policy),
            &log,
            &initial,
        )
        .unwrap();
        assert_eq!(scores_of(&outcome, &key), [None, None, None]);
        assert!(outcome.ranked.iter().all(|scored| scored.dsl != key));
        assert_eq!(log.rejections(), rejection, "{policy:?}");
    }
}

#[test]
fn penalize_ranks_the_fault_free_program_first_on_equal_loss() {
    let (x, y) = equal_loss_data(160);
    let metric = |truth: &[f64], predicted: &[f64]| feat_math::r2(truth, predicted);
    let (log_key, log1p_key) = (canonical(LOG_NEGATIVE), canonical(LOG1P_ABS));
    let initial = [LOG_NEGATIVE.to_string(), LOG1P_ABS.to_string()];
    let race = |policy: FaultPolicy| {
        evolve_lite_race_with_exact_initial_population(
            &x,
            &y,
            &metric,
            &race_config(2, 1, 17, policy),
            &Log::default(),
            &initial,
        )
        .unwrap()
    };
    let order = |policy: FaultPolicy| -> Vec<String> {
        race(policy)
            .ranked
            .into_iter()
            .map(|scored| scored.dsl)
            .collect()
    };
    let legacy_order = vec![log_key.clone(), log1p_key.clone()];
    let fault_free_first = vec![log1p_key.clone(), log_key.clone()];
    // Legacy keeps the tie in generation order.
    assert_eq!(order(FaultPolicy::Legacy), legacy_order);
    // The penalty alone decides: the faulting program loses 0.05 times its domain rate of 1.
    assert_eq!(order(PENALIZE), fault_free_first);
    assert_eq!(order(penalize(0.05, 0.0, false)), fault_free_first);
    // At zero weights the scores stay tied exactly, and only the tie-break reorders them.
    assert_eq!(order(penalize(0.0, 0.0, true)), fault_free_first);
    assert_eq!(order(penalize(0.0, 0.0, false)), legacy_order);

    // The scores themselves: every rung-0 fold of the faulting program is 0 - 0.05 * 1.
    let outcome = race(PENALIZE);
    let folds = RACE_RUNG_FOLDS[0];
    let penalized: f64 = 0.0 - 0.05 * 1.0;
    assert_eq!(
        scores_of(&outcome, &log_key)[0].as_deref().map(bits),
        Some(vec![penalized.to_bits(); folds])
    );
    assert_eq!(
        scores_of(&outcome, &log1p_key)[0].as_deref().map(bits),
        Some(vec![0.0_f64.to_bits(); folds])
    );

    // The surrogate search: its champion scores 0 here as well.
    let positions = |policy: FaultPolicy| {
        let config = EvolveConfig {
            pop: 3,
            gens: 1,
            folds: 3,
            seed: 17,
            seed_genomes: initial.to_vec(),
            fault_policy: policy,
            ..EvolveConfig::default()
        };
        let ranked = evolve(&x, &y, &metric, &config, &Log::default());
        let position = |key: &str| {
            ranked
                .iter()
                .position(|scored| scored.dsl == key)
                .unwrap_or_else(|| panic!("{key} is not ranked"))
        };
        (position(&log_key), position(&log1p_key))
    };
    let (log_at, log1p_at) = positions(FaultPolicy::Legacy);
    assert!(log_at < log1p_at);
    for policy in [PENALIZE, penalize(0.0, 0.0, true)] {
        let (log_at, log1p_at) = positions(policy);
        assert!(log1p_at < log_at, "{policy:?}");
    }
    let (log_at, log1p_at) = positions(penalize(0.0, 0.0, false));
    assert!(log_at < log1p_at);
}

/// The domain-rate counters' values in runs whose ranking is known. On `equal_loss_data` every
/// program scores 0 (the R² of a constant target), `LOG_NEGATIVE`'s domain rate is 1 at every
/// rung, and no other program has an expression fault: the mean over the ranked candidates is one
/// over their number, and the mean over the elites counts `LOG_NEGATIVE` only when it is an elite.
#[test]
fn the_domain_rate_counters_follow_the_ranking_and_the_elites() {
    let (x, y) = equal_loss_data(160);
    let metric = |truth: &[f64], predicted: &[f64]| feat_math::r2(truth, predicted);
    let (log_key, log1p_key) = (canonical(LOG_NEGATIVE), canonical(LOG1P_ABS));
    let initial = [LOG_NEGATIVE.to_string(), LOG1P_ABS.to_string()];
    let rates = |counters: &Counters| (counters.pop_domain_rate, counters.elite_domain_rate);

    // The race: two candidates and one elite, the first ranked.
    let race = |policy: FaultPolicy, log: &Log| {
        evolve_lite_race_with_exact_initial_population(
            &x,
            &y,
            &metric,
            &race_config(2, 1, 17, policy),
            log,
            &initial,
        )
        .unwrap()
    };
    let log = Log::default();
    race(FaultPolicy::Legacy, &log);
    log.assert_nothing_reported();
    for (policy, first, elite_rate) in [
        (FaultPolicy::RejectHard, &log_key, 1.0),
        (penalize(0.0, 0.0, false), &log_key, 1.0),
        (PENALIZE, &log1p_key, 0.0),
        (penalize(0.0, 0.0, true), &log1p_key, 0.0),
    ] {
        let log = Log::default();
        let outcome = race(policy, &log);
        let order: Vec<&str> = outcome
            .ranked
            .iter()
            .map(|scored| scored.dsl.as_str())
            .collect();
        assert_eq!(order.len(), 2, "{policy:?}");
        assert_eq!(order[0], first.as_str(), "{policy:?}");
        let generations = log.generations();
        let last = generations.last().expect("a ranked generation");
        assert_eq!(
            rates(last),
            (0.5_f64.to_bits(), f64::to_bits(elite_rate)),
            "{policy:?}"
        );
        assert_eq!(
            (last.hard_rejected, last.hard_rejected_share),
            (0, 0.0_f64.to_bits())
        );
    }

    // The surrogate search: sixteen candidates and two elites. Generation 0 is the champion, the
    // two programs, the eight starters and random programs, and the ranking keeps that order among
    // equal scores, so the elites are the champion and `LOG_NEGATIVE` unless the penalty or the
    // tie-break moves `LOG_NEGATIVE` down.
    let surrogate = |policy: FaultPolicy| {
        let config = EvolveConfig {
            pop: 16,
            gens: 1,
            folds: 3,
            seed: 17,
            seed_genomes: initial.to_vec(),
            fault_policy: policy,
            ..EvolveConfig::default()
        };
        let log = Log::default();
        let ranked = evolve(&x, &y, &metric, &config, &log);
        (ranked, log.generations())
    };
    for (policy, elite_rate) in [
        (FaultPolicy::RejectHard, 0.5),
        (PENALIZE, 0.0),
        (penalize(0.0, 0.0, true), 0.0),
    ] {
        let (ranked, generations) = surrogate(policy);
        let log_at = ranked
            .iter()
            .position(|scored| scored.dsl == log_key)
            .expect("LOG_NEGATIVE is ranked");
        assert_eq!(log_at < 2, elite_rate > 0.0, "{policy:?}");
        assert_eq!(generations.len(), 1);
        assert_eq!(generations[0].ranked, ranked.len());
        assert_eq!(
            rates(&generations[0]),
            (
                (1.0 / ranked.len() as f64).to_bits(),
                f64::to_bits(elite_rate)
            ),
            "{policy:?}"
        );
    }
}

#[test]
fn fault_free_programs_score_as_under_legacy() {
    let (x, y) = toy_data(160, 5, 11);
    let initial = FAULT_FREE.map(str::to_string);
    // Generation 0 only: later generations may hold offspring that fail the scope check.
    let race = |policy: FaultPolicy| {
        let log = Log::default();
        let outcome = evolve_lite_race_with_exact_initial_population(
            &x,
            &y,
            &feat_math::r2,
            &race_config(3, 1, 19, policy),
            &log,
            &initial,
        )
        .unwrap();
        log.assert_nothing_reported();
        race_run(&outcome)
    };
    let surrogate = |policy: FaultPolicy| {
        // Generation 0 is the champion and the three programs.
        let config = EvolveConfig {
            pop: 4,
            gens: 1,
            folds: 3,
            seed: 19,
            seed_genomes: initial.to_vec(),
            fault_policy: policy,
            ..EvolveConfig::default()
        };
        let log = Log::default();
        let ranked = evolve(&x, &y, &feat_math::r2, &config, &log);
        log.assert_nothing_reported();
        ranked.iter().map(ranked_line).collect::<Vec<_>>()
    };
    let (legacy_race, legacy_surrogate) =
        (race(FaultPolicy::Legacy), surrogate(FaultPolicy::Legacy));
    assert_eq!(legacy_race.ranked.len(), 3);
    for policy in [
        FaultPolicy::RejectHard,
        PENALIZE,
        penalize(0.05, 0.01, true),
    ] {
        assert_eq!(race(policy), legacy_race, "{policy:?}");
        assert_eq!(surrogate(policy), legacy_surrogate, "{policy:?}");
    }
}

#[test]
fn the_penalty_is_the_documented_formula_on_the_published_rates() {
    let (x, y) = toy_data(160, 5, 13);
    let key = canonical(PARTIAL_FAULTS);
    let initial = [PARTIAL_FAULTS.to_string()];
    let race = |policy: FaultPolicy| {
        evolve_lite_race_with_exact_initial_population(
            &x,
            &y,
            &feat_math::r2,
            &race_config(1, 1, 23, policy),
            &Log::default(),
            &initial,
        )
        .unwrap()
    };
    let legacy = race(FaultPolicy::Legacy);
    let penalized = race(penalize(0.05, 0.02, false));

    // Replay rung 0 through the public telemetry: its program, rows and folds.
    let replay = replay_rung0(&legacy, &x, &y, &key).unwrap();
    let totals = replay.all;
    assert!(!totals.is_hard());
    assert!(
        totals.domain_cells > 0 && totals.saturation_cells > 0,
        "{totals:?}"
    );
    assert!(totals.domain_rate() < 1.0 && totals.saturation_rate() < 1.0);
    // The held-out passes count: without them the rates would differ.
    assert_ne!(
        (
            replay.training.domain_rate().to_bits(),
            replay.training.saturation_rate().to_bits()
        ),
        (
            totals.domain_rate().to_bits(),
            totals.saturation_rate().to_bits()
        ),
        "the training passes alone give the same rates"
    );

    let legacy_folds = scores_of(&legacy, &key)[0].clone().unwrap();
    assert_eq!(
        bits(&legacy_folds),
        bits(&replay.scores),
        "the replay is the rung-0 evaluation"
    );
    let by_hand: Vec<f64> = legacy_folds
        .iter()
        .map(|score| score - 0.05 * totals.domain_rate() - 0.02 * totals.saturation_rate())
        .collect();
    let formula: Vec<f64> = legacy_folds
        .iter()
        .map(|&score| totals.penalize(score, 0.05, 0.02).unwrap())
        .collect();
    assert_eq!(bits(&formula), bits(&by_hand));
    assert_eq!(
        scores_of(&penalized, &key)[0].as_deref().map(bits),
        Some(bits(&by_hand))
    );
}

#[test]
fn non_legacy_policies_never_call_a_gpu_backend() {
    let (x, y) = toy_data(160, 5, 42);
    let backend = |_: Vec<GpuRidgeScoreProblem>| -> Result<GpuRidgeScoreBatch, String> {
        panic!("a non-legacy fault policy called the GPU backend")
    };
    for policy in [FaultPolicy::RejectHard, PENALIZE] {
        let config = race_config(16, 2, 41, policy);
        let cpu = evolve_lite_race(&x, &y, &feat_math::r2, &config, &Log::default());
        let routed = evolve_lite_race_with_gpu_policy(
            &x,
            &y,
            &feat_math::r2,
            &config,
            &Log::default(),
            Some(&backend),
            GpuRidgeScorePolicy::new(1, 1),
        );
        assert_eq!(race_run(&routed), race_run(&cpu));
        assert_eq!(routed.gpu_screen_timings.gpu_folds, 0);
        assert!(routed
            .candidates
            .iter()
            .filter_map(|candidate| candidate.r0_fold_routes.as_ref())
            .flatten()
            .all(|receipt| receipt.route == Rung0FoldRoute::CpuExact));
    }
}

#[test]
fn exact_reuse_stays_off_under_a_non_legacy_policy() {
    let (x, y) = toy_data(160, 5, 42);
    let reuse = Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
        RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
        &"ab".repeat(32),
    )
    .unwrap();
    let requested = |policy: FaultPolicy| {
        evolve_lite_race_with_gpu_policy_and_exact_surrogate_dedup(
            &x,
            &y,
            &feat_math::r2,
            &race_config(16, 2, 41, policy),
            &Log::default(),
            None,
            None,
            Some(reuse),
        )
    };
    assert!(
        requested(FaultPolicy::Legacy)
            .rung0_exact_surrogate_dedup
            .active,
        "under Legacy the request turns reuse on"
    );
    for policy in [FaultPolicy::RejectHard, PENALIZE] {
        let outcome = requested(policy);
        let receipt = &outcome.rung0_exact_surrogate_dedup;
        assert!(receipt.requested && !receipt.active);
        assert!(receipt
            .activation_blocker
            .as_deref()
            .is_some_and(|blocker| blocker.contains("non-legacy fault policy")));
        assert_eq!(
            (
                receipt.logical_candidates,
                receipt.physical_candidate_evaluations,
                receipt.reused_candidates
            ),
            (0, 0, 0)
        );
        let plain = evolve_lite_race(
            &x,
            &y,
            &feat_math::r2,
            &race_config(16, 2, 41, policy),
            &Log::default(),
        );
        assert_eq!(race_run(&outcome), race_run(&plain));
    }
}

#[test]
fn penalty_weights_must_be_finite_and_non_negative() {
    let (x, y) = toy_data(32, 5, 1);
    for (lambda_domain, lambda_saturation) in [
        (f64::NAN, 0.0),
        (0.0, f64::NAN),
        (-0.5, 0.0),
        (0.0, -1e-9),
        (f64::INFINITY, 0.0),
        (0.0, f64::NEG_INFINITY),
    ] {
        let policy = penalize(lambda_domain, lambda_saturation, true);
        let error = policy.validate().unwrap_err();
        assert!(error.contains("must be finite and non-negative"), "{error}");
        let refused = evolve_lite_race_with_exact_initial_population(
            &x,
            &y,
            &feat_math::r2,
            &race_config(1, 1, 3, policy),
            &Log::default(),
            &["(pipe X (ridge))".to_string()],
        )
        .unwrap_err();
        assert_eq!(refused, error);
    }
    for (lambda_domain, lambda_saturation) in [(0.0, 0.0), (0.05, 0.0), (1e3, 2.5)] {
        assert_eq!(
            penalize(lambda_domain, lambda_saturation, false).validate(),
            Ok(())
        );
    }
    assert_eq!(FaultPolicy::Legacy.validate(), Ok(()));
    assert_eq!(FaultPolicy::RejectHard.validate(), Ok(()));
    assert_eq!(FaultPolicy::default(), FaultPolicy::Legacy);
    assert_eq!(EvolveConfig::default().fault_policy, FaultPolicy::Legacy);
}

#[test]
#[should_panic(expected = "lambda_domain must be finite and non-negative")]
fn the_surrogate_search_refuses_an_invalid_penalty() {
    let (x, y) = toy_data(32, 5, 1);
    let config = EvolveConfig {
        pop: 2,
        gens: 1,
        fault_policy: penalize(f64::NAN, 0.0, false),
        ..EvolveConfig::default()
    };
    evolve(&x, &y, &feat_math::r2, &config, &Log::default());
}

#[test]
#[should_panic(expected = "lambda_saturation must be finite and non-negative")]
fn the_race_refuses_an_invalid_penalty() {
    let (x, y) = toy_data(32, 5, 1);
    let config = race_config(2, 1, 3, penalize(0.05, -1.0, true));
    evolve_lite_race(&x, &y, &feat_math::r2, &config, &Log::default());
}

/// Everything the policy decides is a function of the data and the seed: the same searches in a
/// one-thread and a four-thread pool give the same candidates, scores, rejections and counters.
#[test]
fn outcomes_do_not_depend_on_the_thread_count() {
    let (x, y) = toy_data(160, 5, 31);
    let seeds: Vec<String> = [
        PARTIAL_FAULTS,
        "(pipe X (derive expr=\"col(12)\") (ridge))",
        "(pipe X (derive expr=\"sqrt(col(2)) + 1/col(3)\" append=true) (ridge))",
        FAULT_FREE[0],
        FAULT_FREE[1],
    ]
    .map(str::to_string)
    .to_vec();
    let policy = penalize(0.05, 0.01, true);
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let race_log = Log::default();
            let race = evolve_lite_race(
                &x,
                &y,
                &feat_math::r2,
                &EvolveConfig {
                    seed_genomes: seeds.clone(),
                    ..race_config(12, 3, 5, policy)
                },
                &race_log,
            );
            let surrogate_log = Log::default();
            let surrogate = evolve(
                &x,
                &y,
                &feat_math::r2,
                &EvolveConfig {
                    pop: 12,
                    gens: 3,
                    folds: 3,
                    seed: 5,
                    max_transforms: 3,
                    seed_genomes: seeds.clone(),
                    fault_policy: policy,
                    ..EvolveConfig::default()
                },
                &surrogate_log,
            );
            (
                race_run(&race),
                race_log.rejections(),
                race_log.generations(),
                surrogate.iter().map(ranked_line).collect::<Vec<_>>(),
                surrogate_log.rejections(),
                surrogate_log.generations(),
            )
        })
    };
    let one = run(1);
    assert!(
        !one.1.is_empty() && !one.4.is_empty(),
        "the searches must reject something"
    );
    assert!(
        one.2
            .iter()
            .any(|counters| counters.pop_domain_rate != 0.0_f64.to_bits()),
        "the searches must rank a faulting candidate"
    );
    assert_eq!(run(4), one);
}

/// The generation-0 seeds of the informational run: programs with expression stages.
const EXPRESSION_SEEDS: [&str; 15] = [
    // The expression programs of `compat_freeze`'s NAMED_PROGRAMS: they read columns 0 to 4.
    "(pipe X (map expr=\"sigmoid(x) + 0.1*col(0)\" append=true) (derive expr=\"row_mean + row_std - c1/(c0 + 3)\" append=true) (ridge))",
    "(pipe X (map expr=\"x^2 - min(x, c2) + tanh(c3)\") (elasticnet alpha=0.005 l1_ratio=0.7))",
    "(pipe X (map expr=\"step(col(0), 0.0) + 0.5*step(row_mean, c1)\" append=true) (ridge))",
    "(pipe X (map expr=\"exp(clip(col(0),-2,2)) + log(abs(col(1)) + 1.0) + asinh(col(2)) + atan(col(3))\") (ridge))",
    "(pipe X (derive append=true expr=\"erf(col(0)) + sign(col(1)) * floor(col(2)) + round(col(3))\") (ridge))",
    "(pipe X (map expr=\"abs(x) + sqrt(abs(c0)) - log1p(abs(x))\" append=true) (derive expr=\"sin(c0) + cos(c1) * tanh(c2)\" append=true) (ridge))",
    "(pipe X (map expr=\"sigmoid(x) - min(x, c1) + max(x, c2)\") (derive expr=\"clip(c0, -1, 1) + step(c1, 0.5)\" append=true) (ridge))",
    "(pipe X (map expr=\"exp(clip(x, -3, 3)) + log(abs(x) + 1)\" append=true) (derive expr=\"asinh(c0) + atan(c1) - erf(c2)\" append=true) (ridge))",
    "(pipe X (map expr=\"sign(x) * floor(x) + round(x)\") (derive expr=\"row_sum - row_l1 + row_l2\" append=true) (ridge))",
    "(pipe X (map expr=\"x - row_mean + row_std\" append=true) (derive expr=\"row_min * row_max + c0 / (c1 + 3)\" append=true) (ridge))",
    "(pipe X (map expr=\"x^2 - (-x) * 0.5 + x / (c0 + 2)\") (derive expr=\"c3^3 - c4*c4*c4 + c2^0.5\" append=true) (elasticnet alpha=0.005 l1_ratio=0.7))",
    // `x` outside a map: an illegal scope everywhere.
    "(pipe X (derive expr=\"x\" append=true) (ridge))",
    // Guarded operations the scaled data reaches: log of zero, division by zero, sqrt of a
    // negative difference.
    "(pipe X (derive expr=\"log(col(0))\" append=true) (ridge))",
    "(pipe X (derive expr=\"1/col(1)\" append=true) (ridge))",
    "(pipe X (derive expr=\"sqrt(col(0) - col(1))\" append=true) (ridge))",
];

/// Every regression dataset of `datasets/test`, by name, with the rows of its `train.csv` (never
/// `test.csv`), checked against the sha256 its metadata records.
fn test_datasets() -> Vec<(String, Rows, Vec<f64>)> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/test");
    let mut dirs: Vec<_> = std::fs::read_dir(&root)
        .unwrap_or_else(|error| panic!("{}: {error}", root.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.join("metadata.json").is_file() && path.join("train.csv").is_file())
        .collect();
    dirs.sort();
    dirs.into_iter()
        .filter_map(|dir| {
            let metadata: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(dir.join("metadata.json")).unwrap())
                    .unwrap();
            if metadata["task"] != "regression" {
                return None;
            }
            let bytes = std::fs::read(dir.join("train.csv")).unwrap();
            let digest: String = Sha256::digest(&bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            assert_eq!(
                metadata["files"]["train"]["sha256"].as_str(),
                Some(digest.as_str()),
                "{}: train.csv does not match its metadata",
                dir.display()
            );
            let (mut x, mut y) = (Vec::new(), Vec::new());
            for line in std::str::from_utf8(&bytes).unwrap().lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let mut row: Vec<f64> = line
                    .split(',')
                    .map(|cell| cell.trim().parse::<f64>().unwrap())
                    .collect();
                y.push(row.pop().unwrap());
                x.push(row);
            }
            let name = dir.file_name().unwrap().to_string_lossy().into_owned();
            Some((name, x, y))
        })
        .collect()
}

/// The kind of a rejection reason: the scope class, or the Hard classes.
fn reason_kind(reason: &str) -> String {
    for kind in ["COLUMN_OOB", "ILLEGAL_SCOPE"] {
        if reason.starts_with("scope:") && reason.contains(kind) {
            return format!("scope {kind}");
        }
    }
    if reason.starts_with("scope:") {
        return "scope, unchecked".to_string();
    }
    reason.to_string()
}

/// The best program of a run, measured after it: how many expression stages it has, whether its
/// rung-0 program passes the scope check, and its rung-0 rates over every traced pass (with
/// `hard` when a pass raised a Hard class), replayed through the public entry points. This is how
/// the run under `Legacy`, which measures nothing, is compared with the others.
fn best_program_faults(outcome: &RaceOutcome, x: &Rows, y: &[f64]) -> String {
    let Some(best) = outcome.ranked.first() else {
        return "-".to_string();
    };
    let GenomeIr::Pipeline(pipeline) = GenomeIr::parse(&best.dsl).unwrap() else {
        unreachable!("the race ranks pipelines only")
    };
    let expressions = pipeline
        .stages
        .iter()
        .filter(|stage| stage.op == "map" || stage.op == "derive")
        .count();
    let (surrogate, _) = rung0_trace_identity(&best.dsl).unwrap();
    let scope = match validate_scope(&GenomeIr::parse(&surrogate).unwrap(), x[0].len()) {
        Ok(()) => "in scope".to_string(),
        Err(error) => format!("scope: {error}"),
    };
    let rates = replay_rung0(outcome, x, y, &best.dsl).map_or_else(
        || "not fitted".to_string(),
        |replay| {
            format!(
                "{:.4}/{:.4}{}",
                replay.all.domain_rate(),
                replay.all.saturation_rate(),
                if replay.all.is_hard() { " hard" } else { "" }
            )
        },
    );
    format!("{expressions} expression stages, {scope}, rates {rates}")
}

/// Informational only, with no quality gate (plan WP-1.5 acceptance): the rates of `Legacy`,
/// `RejectHard` and `Penalize{0.05, 0, tie-break}` in one seeded lite race per regression dataset
/// of `datasets/test`, seeded with expression programs, since the default vocabulary never
/// generates an expression stage and would measure nothing. Prints one line per run.
#[test]
#[ignore = "informational: fault-policy rates on datasets/test for the WP-1.5 receipt; run with --run-ignored only"]
fn informational_fault_policy_rates_on_the_test_datasets() {
    let datasets = test_datasets();
    assert_eq!(datasets.len(), 10);
    let policies = [
        ("legacy", FaultPolicy::Legacy),
        ("reject-hard", FaultPolicy::RejectHard),
        ("penalize-0.05-0-tie", PENALIZE),
    ];
    println!(
        "dataset | policy | features | raced | rejected | rejected offspring | reasons | pop domain rate | elite domain rate | best cv | best program measured at rung 0 | best program | seconds"
    );
    for (name, x, y) in &datasets {
        for (label, policy) in policies {
            let config = EvolveConfig {
                pop: 24,
                gens: 4,
                folds: 3,
                seed: 0x0DA7_A5E7,
                max_transforms: 3,
                time_budget_secs: 0.0,
                fitness: FitnessMode::LiteRace,
                rank_robust: true,
                seed_genomes: EXPRESSION_SEEDS.map(str::to_string).to_vec(),
                fault_policy: policy,
                ..EvolveConfig::default()
            };
            let log = Log::default();
            let started = std::time::Instant::now();
            let outcome = evolve_lite_race(x, y, &feat_math::r2, &config, &log);
            let seconds = started.elapsed().as_secs_f64();
            let rejections = log.rejections();
            let rejected: std::collections::BTreeSet<&str> =
                rejections.iter().map(|(dsl, _)| dsl.as_str()).collect();
            let offspring: Vec<&str> = outcome
                .candidates
                .iter()
                .filter(|candidate| candidate.r0_generation.is_some_and(|gen| gen > 0))
                .map(|candidate| candidate.dsl.as_str())
                .collect();
            let rejected_offspring = offspring
                .iter()
                .filter(|dsl| rejected.contains(**dsl))
                .count();
            let mut reasons = std::collections::BTreeMap::<String, usize>::new();
            for (_, reason) in &rejections {
                *reasons.entry(reason_kind(reason)).or_default() += 1;
            }
            let last = log.generations().last().cloned();
            let rate = |bits: Option<u64>| bits.map_or(f64::NAN, f64::from_bits);
            let best = outcome.ranked.first();
            println!(
                "{name} | {label} | {} | {} | {} | {}/{} | {:?} | {:.4} | {:.4} | {} | {} | {} | {seconds:.1}",
                x[0].len(),
                outcome.rung_evals[0],
                rejected.len(),
                rejected_offspring,
                offspring.len(),
                reasons,
                rate(last.as_ref().map(|counters| counters.pop_domain_rate)),
                rate(last.as_ref().map(|counters| counters.elite_domain_rate)),
                best.map_or_else(|| "-".to_string(), |scored| format!("{:.4}", scored.cv_mean)),
                best_program_faults(&outcome, x, y),
                best.map_or("-", |scored| scored.dsl.as_str()),
            );
        }
    }
}

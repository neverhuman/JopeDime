//! Evolutionary search over pipeline genomes.
//!
//! A compact generational GP over the `(pipe X ...)` DSL: a population of stage-lists is evolved by
//! param-jitter / stage-insert / stage-drop / stage-swap / head-swap mutations plus splice
//! crossover, seeded with the vendored champion genome and a few curated starters. Fitness is an
//! honest k-fold CV (the interpreter refits the whole pipeline inside every fold), scored by a
//! caller-provided metric (R² for regression, AUC for binary via continuous scores). Every genome
//! is EXECUTED by the Rust interpreter (`model::fit/predict`); evaluation is population-parallel
//! via rayon, with a cross-generation memo so duplicates are never re-evaluated.

use crate::credit::TokenLedger;
use crate::dsl::{DslExpr, ParamValue, Stage};
use crate::heads::RidgeModel;
use crate::kernel::admission::{outranks, prefer_fewer_faults, FaultTotals, Rejection};
use crate::kernel::lexicase::{CaseErrors, LexicaseConfig, Selection, Selector};
use crate::kernel::scope::validate_scope;
use crate::kernel::telemetry::{fit_traced, predict_traced, PipelineTelemetry};
use crate::ops::{apply_op, apply_op_train, fit_op, FittedOp};
use crate::rng::Pcg32;
use crate::target_transform::{fit_target, is_target_op, FittedTarget};
use crate::transpile::{emit, EmitOptions, Emitted};
use crate::{model, sanitize, GenomeIr, Rows, CHAMPION_GENOME};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

/// Heads eligible during search (krr/stack excluded: too slow for the inner loop; logistic/gbm are
/// aliases of ridge/histgbm). Weighted toward histgbm — the strongest head family. The v2.1 grammar
/// adds the composite-kernel Gaussian process (`gpr`) and the Nadaraya-Watson `ksmooth` head; both
/// are scored through the cheap ridge surrogate (see `search_surrogate`) but carried into emission
/// on improvement, so an invented champion can ship a real kernel-algebra GP or kernel-smoother.
const HEADS: [&str; 9] = [
    "histgbm",
    "histgbm",
    "ridge",
    "ridge_gcv",
    "elasticnet",
    "histgbm2",
    "histgbm2",
    "gpr",
    "ksmooth",
];
/// All head op names (must match spec::HEAD_OPS) — used to recognize/replace the head stage.
pub const HEAD_OPS: [&str; 14] = [
    "ridge",
    "logistic",
    "ridge_gcv",
    "elasticnet",
    "histgbm",
    "gbm",
    "ensemble",
    "ensemble_ridge_gbm",
    "histgbm2",
    "krr",
    "gpr",
    "ksmooth",
    "hurdle2",
    "stack",
];
/// Transforms eligible during search. Row-order-dependent ops (lag/roll) are excluded — they break
/// iid CV on shuffled tabular data. map/derive are excluded from *generation* (their expr
/// sub-language needs its own generator) but parse/execute fine if present in seeds. The matrix-heavy
/// kernels (nystrom / svd_project / landmark_dist) are likewise excluded from GENERATION — their
/// per-fold cost is O(rows²)-ish and dominates the inner loop — but stay parseable in seeds.
const TRANSFORMS: [&str; 18] = [
    "standardize",
    "robust_scale",
    "rank",
    "blom",
    "quantile_bins",
    "bin_onehot",
    "hinge",
    "winsor_clip",
    "cheb",
    "poly_interact",
    "rff_cos",
    "rff_sin",
    "rand_proj",
    "oblique_hinge",
    "row_stats",
    "pair_prod",
    "target_bin_encode",
    "sigmoid",
];

/// Expansion transforms reachable ONLY when `EvolveConfig::expansion_vocab`
/// is on. Mirrors `full_rust::EXPANSION_TRANSFORMS`; the legacy `TRANSFORMS`
/// array must never grow — its length is the modulus of every recorded
/// random-generation stream.
pub const EXPANSION_TRANSFORMS: [&str; 5] =
    ["symlog", "select_corr", "impute", "pair_ratio", "spline"];

/// Curated starter genomes (beyond the champion) covering distinct pipeline families, including the
/// v2.1 grammar: a composite-kernel GP, the Nadaraya-Watson smoother, and deep bagged boosting.
const STARTERS: [&str; 8] = [
    "(pipe X (robust_scale) (histgbm depth=4 lr=0.1 rounds=120))",
    "(pipe X (standardize) (ridge))",
    "(pipe X (winsor_clip q=0.05) (cheb degree=4) (robust_scale) (histgbm depth=3 rounds=150))",
    "(pipe X (rank) (poly_interact append=true) (histgbm2))",
    "(pipe X (standardize) (rff_cos n=64 gamma=0.5) (elasticnet alpha=0.001))",
    "(pipe X (winsor_clip q=0.02 append=true) (gpr kernel=rbf+linear m=300 steps=40))",
    "(pipe X (standardize) (ksmooth k=32 gamma=1.0))",
    "(pipe X (robust_scale) (histgbm2 leaves=63 rounds=500 lr=0.05 row_subsample=0.85 col_subsample=0.8 min_leaf=5 val_pct=20))",
];

/// Opt-in generation-zero priors learned from the difficult-ten campaign.
/// They expose broad high-regularization and deliberately underfit plateaus
/// without changing the legacy RNG stream or default starter population.
const HARD_TEN_STARTERS: [&str; 8] = [
    "(pipe X (standardize) (ridge lambda_frac=0.05))",
    "(pipe X (standardize) (ridge lambda_frac=0.2))",
    "(pipe X (standardize) (ridge lambda_frac=0.5))",
    "(pipe X (robust_scale) (ridge lambda_frac=0.2))",
    "(pipe X (cheb degree=3 append=true) (standardize) (ridge lambda_frac=0.2))",
    "(pipe X (standardize) (histgbm2 leaves=15 rounds=500 lr=0.03 row_subsample=0.85 col_subsample=0.8 min_leaf=12 val_pct=20))",
    "(pipe X (standardize) (histgbm2 leaves=31 rounds=500 lr=0.03 row_subsample=0.85 col_subsample=0.8 min_leaf=24 val_pct=20))",
    "(pipe X (standardize) (histgbm2 leaves=31 rounds=500 lr=0.03 row_subsample=0.85 col_subsample=0.8 min_leaf=48 val_pct=20))",
];

/// Opt-in generation-zero starters exercising the expansion vocabulary
/// (impute / symlog / select_corr / pair_ratio / spline + the zscore target).
/// Stored in canonical form (schema params materialized, keys sorted) so
/// round-trips are exact; injected only when `EvolveConfig::expansion_vocab`
/// is on, after the legacy starters, leaving the default pool untouched.
pub const EXPANSION_STARTERS: [&str; 8] = [
    "(pipe X (impute add_indicator=true strategy=median value=0.0) (symlog c=1.0) (select_corr by=corr k=64) (histgbm2 bags=3 col_subsample=1.0 leaves=31 lr=0.05 min_leaf=3 rounds=400 row_subsample=0.7 val_pct=20))",
    "(pipe X (target kind=zscore) (winsor_clip q=0.02) (pair_ratio append=true topk=8) (gpr kernel=rbf m=300 steps=40))",
    "(pipe X (spline append=true knots=5) (ridge_gcv))",
    "(pipe X (impute strategy=median value=0.0) (standardize) (spline knots=4) (elasticnet alpha=0.001 l1_ratio=0.2 sweeps=60))",
    "(pipe X (symlog c=1.0) (rank append=true) (pair_ratio topk=12) (histgbm bins=64 depth=4 lambda=1.0 lr=0.1 max_features=256 min_leaf=10 rounds=120))",
    "(pipe X (select_corr by=var k=128) (robust_scale) (rff_cos append=true gamma=1.0 n=48 seed=7) (ridge lambda_frac=0.05))",
    "(pipe X (target kind=zscore) (impute add_indicator=true strategy=mean value=0.0) (standardize) (histgbm2 bags=3 col_subsample=1.0 leaves=31 lr=0.03 min_leaf=3 rounds=600 row_subsample=0.7 val_pct=20))",
    "(pipe X (spline knots=6) (svd_project append=true iters=24 k=8 seed=7) (elasticnet alpha=0.01 l1_ratio=0.5 sweeps=60))",
];

/// How candidate genomes are scored during search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FitnessMode {
    /// v1: every candidate's head is swapped for a fixed-λ ridge surrogate. Millisecond-fast but
    /// HEAD-BLIND — candidates differing only in their head score identically.
    #[default]
    Surrogate,
    /// v2: 3-rung successive-halving race of the TRUE objective — the ridge surrogate survives
    /// only as a rung-0 junk filter; promoted candidates fit their OWN head at lite budget on
    /// nested row subsets (see `evolve_lite_race`).
    LiteRace,
}

/// How the search treats the faults of a candidate's `map` and `derive` stages (kernel2 WP-1.5).
///
/// [`FaultPolicy::Legacy`], the default, is the search as it has always run: no scope check, no
/// fault telemetry, a guarded fallback value scored like any other value, and exactly the random
/// numbers it has always drawn. The other policies change how candidates are scored, ranked and
/// selected. Parent selection keeps its draws, but a rejection or a penalty changes the ranking,
/// and so which parents are drawn and the programs generated from them; and when the elites'
/// rung-2 re-check rejects every candidate still ranked, the race reseeds with new random
/// programs, which it never does under `Legacy`. A candidate that passes the scope check and
/// raises no fault is scored as under `Legacy`, and the code emitted for a winner never reads the
/// policies' telemetry. Under either of them:
///
/// - The program that is actually fitted (the ridge surrogate in the surrogate search and at rung
///   0, the lite-budget program at rungs 1 and 2) is checked with
///   [`validate_scope`](crate::kernel::scope::validate_scope) at the input width before anything
///   is fitted. A failure rejects the candidate.
/// - Each fold is fitted with [`fit_traced`](crate::kernel::telemetry::fit_traced) and predicted
///   with [`predict_traced`](crate::kernel::telemetry::predict_traced), which return `fit`'s model
///   and `predict`'s predictions with the stage fault counts beside them. A Hard fault in any
///   stage of either pass of any fold rejects the candidate.
/// - A rejected candidate's score is `None`. In the lite race a rejection at any rung also removes
///   the candidate from the ranking, so it is never an elite or a parent. Rejections are counted in
///   [`GenStats::hard_rejected`] and reported with their reason through [`Observer::on_reject`].
/// - The lite race scores every rung on the CPU. A GPU rung-0 backend is never called, because
///   the prepared GPU path does not fit through the traced entry points, and exact rung-0 reuse
///   stays off, because a reused score carries no fault counts of its own; its receipt names the
///   reason.
///
/// The totals, rates, penalty and tie-break are defined in [`crate::kernel::admission`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum FaultPolicy {
    /// Today's search, unchanged.
    #[default]
    Legacy,
    /// Reject every candidate whose scope check fails or that raises a Hard fault.
    RejectHard,
    /// `RejectHard`, and every fold score of a candidate with a Domain or Saturation fault becomes
    /// `s - lambda_domain * domain_rate - lambda_saturation * saturation_rate`
    /// ([`FaultTotals::penalize`]), so every score derived from the folds carries the penalty. A
    /// candidate without a fault keeps its scores bit for bit. With `tie_break`, among candidates
    /// whose scores are within [`TIE_TOLERANCE`](crate::kernel::admission::TIE_TOLERANCE) of each
    /// other, the one with fewer faulted cells ranks first and wins parent selection. The cells
    /// are a count, not a rate, over the evaluation the score comes from (in the race, its highest
    /// scored rung), so an evaluation with more cells, such as rung 2's, counts more of them at the
    /// same rate. Both lambdas must be finite and non-negative ([`FaultPolicy::validate`]).
    Penalize {
        lambda_domain: f64,
        lambda_saturation: f64,
        tie_break: bool,
    },
}

impl FaultPolicy {
    /// The reason the policy cannot be run, if it cannot: a `Penalize` lambda that is negative,
    /// infinite or NaN. The search panics with this reason when it is given such a policy, except
    /// [`evolve_lite_race_with_exact_initial_population`], which returns it.
    pub fn validate(&self) -> Result<(), String> {
        if let Self::Penalize {
            lambda_domain,
            lambda_saturation,
            ..
        } = *self
        {
            for (name, lambda) in [
                ("lambda_domain", lambda_domain),
                ("lambda_saturation", lambda_saturation),
            ] {
                if !(lambda.is_finite() && lambda >= 0.0) {
                    return Err(format!(
                        "fault policy: {name} must be finite and non-negative, got {lambda}"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Whether near-tied candidates are ordered by their faulted cells.
    fn breaks_ties(self) -> bool {
        matches!(
            self,
            Self::Penalize {
                tie_break: true,
                ..
            }
        )
    }
}

// B10 lexicase: begin
/// How the next generation's parents are drawn (B10).
///
/// [`ParentSelection::Tournament3`], the default, is the selection the search has always run, and
/// nothing below is computed or drawn under it. [`ParentSelection::EpsilonLexicase`] is opt-in and
/// needs [`FaultPolicy::Legacy`] ([`ParentSelection::validate`]):
///
/// - A candidate's cases are its signed out-of-fold residuals `y[r] - prediction` on fixed
///   training rows: in the surrogate search the first [`LEXICASE_CASE_ROWS`] rows of the seeded
///   permutation the race also uses, ascending; in the race the rung-0 rows, taken at the rung the
///   candidate is ranked by. Each row is predicted only by models fitted on the other folds, so a
///   row's own label never reaches its prediction, and nothing outside the rows the search was
///   given enters. The fold scores are computed exactly as without lexicase.
/// - Every distinct ranked candidate of the generation is in the pool, and each parent draw is one
///   [`Selector`] event over it. The selection draws come from their own stream per generation,
///   `Pcg32::new(seed, LEXICASE_STREAM + generation)`; the search stream draws only variation.
///   [`Observer::on_selection`] receives each generation's trace.
/// - The race scores every rung on the CPU: the GPU rung-0 backend is never called and exact
///   rung-0 reuse stays off, because neither returns per-row predictions; the reuse receipt names
///   the reason.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ParentSelection {
    /// Tournament of three over the best half of the ranking, as the search has always drawn.
    #[default]
    Tournament3,
    /// Epsilon-lexicase over out-of-fold residual cases ([`crate::kernel::lexicase`]).
    EpsilonLexicase(LexicaseConfig),
}

/// The stream base of the per-generation lexicase selection streams.
pub const LEXICASE_STREAM: u64 = 0x4C45_5843_4153_4531;
/// The most training rows whose residuals are a surrogate-search candidate's lexicase cases.
pub const LEXICASE_CASE_ROWS: usize = 256;

impl ParentSelection {
    /// The reason the selection cannot run under `policy`, if it cannot: lexicase needs
    /// [`FaultPolicy::Legacy`] and a valid [`LexicaseConfig`]. The search panics with this reason,
    /// except [`evolve_lite_race_with_exact_initial_population`], which returns it.
    pub fn validate(&self, policy: FaultPolicy) -> Result<(), String> {
        match self {
            Self::Tournament3 => Ok(()),
            Self::EpsilonLexicase(_) if !matches!(policy, FaultPolicy::Legacy) => {
                Err("parent selection: epsilon-lexicase needs the default fault policy".to_string())
            }
            Self::EpsilonLexicase(config) => config
                .validate()
                .map_err(|error| format!("parent selection: {error}")),
        }
    }

    fn lexicase(self) -> Option<LexicaseConfig> {
        match self {
            Self::Tournament3 => None,
            Self::EpsilonLexicase(config) => Some(config),
        }
    }
}

/// Why exact rung-0 reuse is off under lexicase selection, as its receipt records it.
const LEXICASE_REUSE_BLOCKER: &str =
    "exact rung-0 reuse is off under lexicase selection: a reused score has no per-row residuals";
// B10 lexicase: end

// B11 ridge scoring: begin
/// How the surrogate search scores its candidates (B11).
///
/// [`RidgeScoring::Refit`], the default, fits every new candidate's surrogate on every fold, as
/// the search always has. [`RidgeScoring::Certified`] returns the same fold scores (and lexicase
/// cases) bit for bit with less work:
///
/// - a candidate whose ridge surrogate (the program the search actually fits) was already scored
///   in the run reuses that score, because the surrogate fixes everything the score depends on;
/// - transform prefixes that surrogates share on a fold are fitted once, through
///   [`model::fit_with_stage_cache`], whose hits return the output of the identical computation.
///
/// Both key on the exact bits of every parameter, never on canonical text, which prints a float
/// within 1e-12 of an integer as that integer: surrogates that print alike but differ in any bit
/// are scored apart, as `Refit` scores them. The search's own genome memo, which merges genomes
/// whose whole canonical text matches, is unchanged and works the same in both modes.
/// The ridge fit is the certified `RidgeModel::fit` in both modes. Certified scoring applies to
/// the surrogate search and needs [`FaultPolicy::Legacy`] ([`RidgeScoring::validate`]).
/// [`Observer::on_scoring`] receives its counters after each generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RidgeScoring {
    #[default]
    Refit,
    Certified,
}

/// The byte budget of the certified scoring's stage cache. The bytes the cache counts (payload, not
/// allocation overhead) never pass it: an entry that would pass it is refused and recomputed on a
/// later miss, with identical bits.
pub const CERTIFIED_STAGE_CACHE_BYTES: usize = 1 << 30;

impl RidgeScoring {
    /// The reason the scoring cannot run in this search, if it cannot. The search panics with it,
    /// except [`evolve_lite_race_with_exact_initial_population`], which returns it.
    pub fn validate(&self, fitness: FitnessMode, policy: FaultPolicy) -> Result<(), String> {
        match self {
            Self::Refit => Ok(()),
            Self::Certified if fitness != FitnessMode::Surrogate => {
                Err("ridge scoring: certified scoring applies to the surrogate search".to_string())
            }
            Self::Certified if !matches!(policy, FaultPolicy::Legacy) => {
                Err("ridge scoring: certified scoring needs the default fault policy".to_string())
            }
            Self::Certified => Ok(()),
        }
    }
}
// B11 ridge scoring: end

#[cfg(test)]
mod gpu_screen_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const PARITY_ABS_TOL: f64 = 5e-3;
    const TEST_EVALUATOR_RELEASE_SHA256: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    type Rung0TestRun = (
        Vec<Option<Vec<f64>>>,
        Rung0ExactSurrogateDedupReceipt,
        Vec<Option<Rung0ExactSurrogateCandidateReceipt>>,
        Vec<Vec<Rung0FoldRouteReceipt>>,
        GpuScreenPhaseTimings,
    );

    struct NullObs;
    impl Observer for NullObs {
        fn on_generation(&self, _stats: &GenStats) {}
        fn on_improvement(&self, _best: &Scored, _emitted: &Emitted, _stats: &GenStats) {}
    }

    struct XorShift(u64);
    impl XorShift {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn synth(n: usize, seed: u64) -> (Rows, Vec<f64>) {
        let mut rng = XorShift(seed | 1);
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let row: Vec<f64> = (0..6).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
            let target = 0.8 * row[0] - 1.3 * row[1]
                + 0.5 * row[2] * row[3]
                + (row[4] * 1.7).sin()
                + 0.05 * (rng.next_f64() - 0.5);
            x.push(row);
            y.push(target);
        }
        (x, y)
    }

    fn candidates(count: usize, seed: u64) -> Vec<(String, DslExpr)> {
        let mut rng = Pcg32::new(seed, 0x5EED);
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for source in STARTERS {
            if let Ok(GenomeIr::Pipeline(genome)) = GenomeIr::parse(source) {
                let key = genome.to_string();
                if seen.insert(key.clone()) {
                    out.push((key, genome));
                }
            }
        }
        while out.len() < count {
            let genome = rand_genome(&mut rng, 4);
            let key = genome.to_string();
            if seen.insert(key.clone()) {
                out.push((key, genome));
            }
        }
        out
    }

    fn run_rung0(
        x: &Rows,
        y: &[f64],
        candidates: &[(String, DslExpr)],
        gpu: Option<&GpuRidgeScoreBatchFn<'_>>,
    ) -> Vec<Option<Vec<f64>>> {
        let policy = gpu.map(|_| GpuRidgeScorePolicy::new(1, 1).unwrap());
        run_rung0_with_policy(x, y, candidates, gpu, policy)
    }

    fn run_rung0_with_policy(
        x: &Rows,
        y: &[f64],
        candidates: &[(String, DslExpr)],
        gpu: Option<&GpuRidgeScoreBatchFn<'_>>,
        policy: Option<GpuRidgeScorePolicy>,
    ) -> Vec<Option<Vec<f64>>> {
        run_rung0_with_options(x, y, candidates, gpu, policy, None).0
    }

    fn run_rung0_with_options(
        x: &Rows,
        y: &[f64],
        candidates: &[(String, DslExpr)],
        gpu: Option<&GpuRidgeScoreBatchFn<'_>>,
        policy: Option<GpuRidgeScorePolicy>,
        dedup: Option<Rung0ExactSurrogateDedupPolicy>,
    ) -> Rung0TestRun {
        run_rung0_with_options_and_observer(x, y, candidates, gpu, policy, dedup, &NullObs)
    }

    fn run_rung0_with_options_and_observer(
        x: &Rows,
        y: &[f64],
        candidates: &[(String, DslExpr)],
        gpu: Option<&GpuRidgeScoreBatchFn<'_>>,
        policy: Option<GpuRidgeScorePolicy>,
        dedup: Option<Rung0ExactSurrogateDedupPolicy>,
        obs: &dyn Observer,
    ) -> Rung0TestRun {
        let n = x.len().min(y.len());
        let perm = permute_indices(n, 7);
        let rows0 = perm[..n.min(RACE_RUNG_ROWS[0])].to_vec();
        let assign0 = fold_assign(rows0.len(), RACE_RUNG_FOLDS[0], "invent_r0");
        let dedup_scope = dedup.map(|policy| {
            rung0_fold_scope_sha256(
                x,
                y,
                &rows0,
                &assign0,
                RACE_RUNG_FOLDS[0],
                &policy.evaluator_release_sha256,
            )
            .unwrap()
        });
        let metric = |actual: &[f64], predicted: &[f64]| crate::metrics::r2(actual, predicted);
        let ctx = RungCtx {
            x,
            y,
            metric: &metric,
            rows: [rows0.clone(), rows0.clone(), rows0],
            assigns: [assign0.clone(), assign0.clone(), assign0],
            caps: [f64::INFINITY; 3],
            t0: Instant::now(),
            overall_budget: 0.0,
            gpu_screen: gpu,
            gpu_screen_policy: policy,
            rung0_exact_surrogate_scope_sha256: dedup_scope,
            capture_cases: false,
        };
        let mut state = RaceState {
            memo: HashMap::new(),
            occurrences: HashMap::new(),
            spent: [0.0; 3],
            rung_evals: [0; 3],
            evals: 0,
            ledger: TokenLedger::new(),
            last_eval_tick: Instant::now(),
            gpu_screen_timings: GpuScreenPhaseTimings::default(),
            rung0_exact_surrogate_cache: HashMap::new(),
            rung0_exact_surrogate_dedup: Rung0ExactSurrogateDedupReceipt {
                requested: dedup.is_some(),
                active: dedup_scope.is_some(),
                authority_receipt_sha256: dedup
                    .map(|_| RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256.to_string()),
                evaluator_release_sha256: dedup
                    .map(Rung0ExactSurrogateDedupPolicy::evaluator_release_sha256_hex),
                cache_scope: dedup.map(|_| "run_local".to_string()),
                wall_clock_completion_may_accelerate: dedup.is_some(),
                completed_candidate_score_bits_exact: dedup.is_some(),
                ..Rung0ExactSurrogateDedupReceipt::default()
            },
        };
        eval_rung(&ctx, &mut state, 0, 0, candidates, obs);
        let scores = candidates
            .iter()
            .map(|(key, _)| {
                state
                    .memo
                    .get(key)
                    .and_then(|entry| entry.rungs[0].clone())
                    .flatten()
            })
            .collect();
        let receipts = candidates
            .iter()
            .map(|(key, _)| {
                state
                    .memo
                    .get(key)
                    .and_then(|entry| entry.r0_exact_surrogate_dedup.clone())
            })
            .collect();
        let routes = candidates
            .iter()
            .map(|(key, _)| {
                state
                    .memo
                    .get(key)
                    .and_then(|entry| entry.r0_fold_routes.clone())
                    .unwrap()
            })
            .collect();
        (
            scores,
            state.rung0_exact_surrogate_dedup,
            receipts,
            routes,
            state.gpu_screen_timings,
        )
    }

    fn assert_same_shape(cpu: &[Option<Vec<f64>>], gpu: &[Option<Vec<f64>>]) {
        assert_eq!(cpu.len(), gpu.len());
        for (index, (cpu_scores, gpu_scores)) in cpu.iter().zip(gpu).enumerate() {
            assert_eq!(
                cpu_scores.is_some(),
                gpu_scores.is_some(),
                "candidate {index}: CPU/GPU scored-ness diverged"
            );
            if let (Some(cpu_scores), Some(gpu_scores)) = (cpu_scores, gpu_scores) {
                assert_eq!(
                    cpu_scores.len(),
                    gpu_scores.len(),
                    "candidate {index}: fold count diverged"
                );
            }
        }
    }

    fn exact_backend(problems: Vec<GpuRidgeScoreProblem>) -> Result<GpuRidgeScoreBatch, String> {
        let fold_bytes = vec![GpuRidgeScoreFoldBytes::default(); problems.len()];
        let scores = problems
            .into_iter()
            .map(|problem| {
                let mut rows: Rows = vec![vec![0.0; problem.p]; problem.train_n];
                // Column-major to row-major transpose. The index arithmetic
                // `column * train_n + row` is the operation being performed, not an
                // accidental way of walking a slice.
                #[allow(clippy::needless_range_loop)]
                for column in 0..problem.p {
                    for row in 0..problem.train_n {
                        rows[row][column] =
                            problem.train_x_cols[column * problem.train_n + row] as f64;
                    }
                }
                let target = problem
                    .train_y
                    .iter()
                    .map(|value| *value as f64)
                    .collect::<Vec<_>>();
                let model = RidgeModel::fit(&rows, &target, problem.lambda_frac);
                let validation = problem
                    .valid_x_rows
                    .chunks_exact(problem.p)
                    .map(|row| row.iter().map(|value| f64::from(*value)).collect())
                    .collect::<Rows>();
                crate::metrics::r2(&problem.valid_y, &model.predict(&validation))
            })
            .collect();
        Ok(GpuRidgeScoreBatch {
            scores,
            timings: GpuRidgeScoreTimings::default(),
            fold_bytes,
        })
    }

    #[test]
    fn gpu_screen_wave_seam_matches_cpu_path_with_exact_backend() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let (x, y) = synth(160, 0xFEED_CAFE);
        let candidates = candidates(192, 42);
        let calls = AtomicUsize::new(0);
        let backend = |problems: Vec<GpuRidgeScoreProblem>| {
            calls.fetch_add(1, Ordering::Relaxed);
            exact_backend(problems)
        };
        let cpu = run_rung0(&x, &y, &candidates, None);
        let gpu = run_rung0(&x, &y, &candidates, Some(&backend));
        assert!(calls.load(Ordering::Relaxed) > 0);
        assert_same_shape(&cpu, &gpu);
        let scored = cpu.iter().filter(|scores| scores.is_some()).count();
        assert!(scored * 2 >= candidates.len());
        for (index, (cpu_scores, gpu_scores)) in cpu.iter().zip(&gpu).enumerate() {
            if let (Some(cpu_scores), Some(gpu_scores)) = (cpu_scores, gpu_scores) {
                for (fold, (cpu_score, gpu_score)) in cpu_scores.iter().zip(gpu_scores).enumerate()
                {
                    let delta = (cpu_score - gpu_score).abs();
                    assert!(
                        delta <= PARITY_ABS_TOL,
                        "candidate {index} ({}) fold {fold}: CPU={cpu_score} backend={gpu_score} delta={delta}",
                        candidates[index].0
                    );
                }
            }
        }
    }

    #[test]
    fn gpu_screen_backend_error_falls_back_to_exact_cpu_path() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let (x, y) = synth(160, 0xBEEF_F00D);
        let candidates = candidates(96, 7);
        let calls = AtomicUsize::new(0);
        let backend = |_problems: Vec<GpuRidgeScoreProblem>| {
            calls.fetch_add(1, Ordering::Relaxed);
            Err("injected CUDA failure".to_string())
        };
        let cpu = run_rung0(&x, &y, &candidates, None);
        let fallback = run_rung0(&x, &y, &candidates, Some(&backend));
        assert!(calls.load(Ordering::Relaxed) > 0);
        assert_same_shape(&cpu, &fallback);
        for (index, (cpu_scores, fallback_scores)) in cpu.iter().zip(&fallback).enumerate() {
            if let (Some(cpu_scores), Some(fallback_scores)) = (cpu_scores, fallback_scores) {
                for (fold, (cpu_score, fallback_score)) in
                    cpu_scores.iter().zip(fallback_scores).enumerate()
                {
                    assert_eq!(
                        cpu_score.to_bits(),
                        fallback_score.to_bits(),
                        "candidate {index} fold {fold}: fallback changed CPU result"
                    );
                }
            }
        }
    }

    fn one_surrogate_candidates(count: usize) -> Vec<(String, DslExpr)> {
        (0..count)
            .map(|index| {
                let source = format!(
                    "(pipe X (standardize) (ridge lambda_frac={}))",
                    0.001 + index as f64 * 0.000_001
                );
                let GenomeIr::Pipeline(genome) = GenomeIr::parse(&source).unwrap() else {
                    unreachable!()
                };
                (genome.to_string(), genome)
            })
            .collect()
    }

    fn score_bits(scores: &[Option<Vec<f64>>]) -> Vec<Option<Vec<u64>>> {
        scores
            .iter()
            .map(|scores| {
                scores
                    .as_ref()
                    .map(|scores| scores.iter().map(|score| score.to_bits()).collect())
            })
            .collect()
    }

    #[test]
    fn exact_surrogate_dedup_is_bit_exact_ordered_and_cross_chunk_deterministic() {
        let (x, y) = synth(160, 0xDEC0_DED0);
        let candidates = one_surrogate_candidates(RACE_CHUNK * 2 + 1);
        let policy = Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
            RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
            TEST_EVALUATOR_RELEASE_SHA256,
        )
        .unwrap();

        let baseline = run_rung0_with_options(&x, &y, &candidates, None, None, None);
        let first = run_rung0_with_options(&x, &y, &candidates, None, None, Some(policy));
        let second = run_rung0_with_options(&x, &y, &candidates, None, None, Some(policy));

        assert_eq!(score_bits(&baseline.0), score_bits(&first.0));
        assert_eq!(score_bits(&first.0), score_bits(&second.0));
        assert_eq!(first.1, second.1);
        assert_eq!(first.2, second.2);
        assert!(!baseline.1.requested);
        assert_eq!(baseline.1.physical_candidate_evaluations, 0);
        assert!(first.1.requested && first.1.active);
        assert_eq!(
            first.1.evaluator_release_sha256.as_deref(),
            Some(TEST_EVALUATOR_RELEASE_SHA256)
        );
        assert_eq!(first.1.cache_scope.as_deref(), Some("run_local"));
        assert_eq!(first.1.logical_candidates, candidates.len() as u64);
        assert_eq!(first.1.physical_candidate_evaluations, 1);
        assert_eq!(first.1.unique_evaluation_signatures, 1);
        assert_eq!(first.1.reused_candidates, candidates.len() as u64 - 1);
        assert_eq!(first.1.rejected_chunks, 0);
        let physical_cpu_folds = first
            .3
            .iter()
            .flatten()
            .filter(|route| route.route == Rung0FoldRoute::CpuExact)
            .count();
        let reused_folds = first
            .3
            .iter()
            .flatten()
            .filter(|route| route.route == Rung0FoldRoute::ExactSurrogateReuse)
            .count();
        assert_eq!(physical_cpu_folds, RACE_RUNG_FOLDS[0]);
        assert_eq!(reused_folds, (candidates.len() - 1) * RACE_RUNG_FOLDS[0]);
        assert_eq!(first.4.cpu_folds as usize, physical_cpu_folds);
        assert_eq!(first.4.gpu_folds, 0);
        assert_eq!(
            first.1.physical_candidate_evaluations * RACE_RUNG_FOLDS[0] as u64,
            first.4.cpu_folds + first.4.gpu_folds
        );

        let representative = first.2[0].as_ref().unwrap();
        assert!(!representative.reused);
        assert_eq!(representative.representative_r0_seq, 0);
        for receipt in first.2.iter().skip(1) {
            let receipt = receipt.as_ref().unwrap();
            assert!(receipt.reused);
            assert_eq!(receipt.representative_r0_seq, 0);
            assert_eq!(
                receipt.evaluation_signature_sha256,
                representative.evaluation_signature_sha256
            );
        }

        let other_release = Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
            RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )
        .unwrap();
        let other = run_rung0_with_options(&x, &y, &candidates, None, None, Some(other_release));
        assert_eq!(score_bits(&first.0), score_bits(&other.0));
        assert_ne!(
            first.2[0].as_ref().unwrap().evaluation_signature_sha256,
            other.2[0].as_ref().unwrap().evaluation_signature_sha256
        );
    }

    #[test]
    fn exact_surrogate_dedup_rejects_signature_collisions_and_partial_scatter() {
        let candidates = ["(pipe X (standardize) (ridge))", "(pipe X (rank) (ridge))"]
            .into_iter()
            .map(|source| {
                let GenomeIr::Pipeline(genome) = GenomeIr::parse(source).unwrap() else {
                    unreachable!()
                };
                (genome.to_string(), genome)
            })
            .collect::<Vec<_>>();
        let scope = [3_u8; 32];
        let cache = HashMap::new();
        let collision =
            build_rung0_exact_surrogate_plan_with(&candidates, &scope, &cache, |_, _| [9_u8; 32])
                .unwrap_err();
        assert_eq!(collision, Rung0ExactSurrogateReject::SignatureCollision);

        let duplicate_candidates = one_surrogate_candidates(2);
        let mut plan =
            build_rung0_exact_surrogate_plan(&duplicate_candidates, &scope, &cache).unwrap();
        plan.member_sources.pop();
        let partial = scatter_rung0_exact_surrogates(
            &plan,
            &RungChunkEvaluation {
                scores: vec![Some(vec![0.25, 0.5])],
                timings: GpuScreenPhaseTimings::default(),
                routes: vec![vec![
                    Rung0FoldRouteReceipt::cpu(Rung0FoldRoute::CpuExact);
                    RACE_RUNG_FOLDS[0]
                ]],
            },
            &cache,
            0,
            RACE_RUNG_FOLDS[0],
        )
        .err()
        .unwrap();
        assert_eq!(partial, Rung0ExactSurrogateReject::PartialMapping);
    }

    #[test]
    fn exact_surrogate_dedup_rejects_failed_representative_and_unknown_authority() {
        assert!(Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
            &"0".repeat(64),
            TEST_EVALUATOR_RELEASE_SHA256,
        )
        .is_none());
        assert!(Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
            RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
            "",
        )
        .is_none());
        assert!(Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
            RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
            &"0".repeat(64),
        )
        .is_none());
        let candidates = one_surrogate_candidates(2);
        let scope = [4_u8; 32];
        let cache = HashMap::new();
        let plan = build_rung0_exact_surrogate_plan(&candidates, &scope, &cache).unwrap();
        let rejected = scatter_rung0_exact_surrogates(
            &plan,
            &RungChunkEvaluation {
                scores: vec![None],
                timings: GpuScreenPhaseTimings::default(),
                routes: vec![vec![
                    Rung0FoldRouteReceipt::cpu(Rung0FoldRoute::CpuExact);
                    RACE_RUNG_FOLDS[0]
                ]],
            },
            &cache,
            0,
            RACE_RUNG_FOLDS[0],
        )
        .err()
        .unwrap();
        assert_eq!(rejected, Rung0ExactSurrogateReject::RepresentativeFailure);
    }

    #[test]
    fn exact_surrogate_dedup_reduces_gpu_work_without_changing_score_bits() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let (x, y) = synth(160, 0xFA57_CADA);
        let candidates = one_surrogate_candidates(RACE_CHUNK * 2 + 1);
        let baseline_folds = AtomicUsize::new(0);
        let baseline_backend = |problems: Vec<GpuRidgeScoreProblem>| {
            baseline_folds.fetch_add(problems.len(), Ordering::Relaxed);
            exact_backend(problems)
        };
        let dedup_folds = AtomicUsize::new(0);
        let dedup_backend = |problems: Vec<GpuRidgeScoreProblem>| {
            dedup_folds.fetch_add(problems.len(), Ordering::Relaxed);
            exact_backend(problems)
        };
        let gpu_policy = GpuRidgeScorePolicy::new(1, 1);
        let dedup_policy = Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
            RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
            TEST_EVALUATOR_RELEASE_SHA256,
        )
        .unwrap();

        let baseline = run_rung0_with_options(
            &x,
            &y,
            &candidates,
            Some(&baseline_backend),
            gpu_policy,
            None,
        );
        let dedup = run_rung0_with_options(
            &x,
            &y,
            &candidates,
            Some(&dedup_backend),
            gpu_policy,
            Some(dedup_policy),
        );

        assert_eq!(score_bits(&baseline.0), score_bits(&dedup.0));
        assert_eq!(
            baseline_folds.load(Ordering::Relaxed),
            candidates.len() * RACE_RUNG_FOLDS[0]
        );
        assert_eq!(dedup_folds.load(Ordering::Relaxed), RACE_RUNG_FOLDS[0]);
        assert_eq!(dedup.1.physical_candidate_evaluations, 1);
        assert_eq!(dedup.1.reused_candidates, candidates.len() as u64 - 1);
        assert_eq!(dedup.4.gpu_folds, RACE_RUNG_FOLDS[0] as u64);
        assert_eq!(dedup.4.cpu_folds, 0);
        assert_eq!(
            dedup
                .3
                .iter()
                .flatten()
                .filter(|route| route.route == Rung0FoldRoute::Gpu)
                .count(),
            RACE_RUNG_FOLDS[0]
        );
        assert_eq!(
            dedup
                .3
                .iter()
                .flatten()
                .filter(|route| route.route == Rung0FoldRoute::ExactSurrogateReuse)
                .count(),
            (candidates.len() - 1) * RACE_RUNG_FOLDS[0]
        );
    }

    #[test]
    fn exact_surrogate_dedup_runs_bounded_and_preserves_common_completed_bits() {
        let (x, y) = synth(96, 0xB0_0DED);
        let seeds = one_surrogate_candidates(64);
        let config = EvolveConfig {
            pop: 64,
            gens: 1,
            time_budget_secs: 30.0,
            seed_genomes: seeds.iter().map(|(dsl, _)| dsl.clone()).collect(),
            fitness: FitnessMode::LiteRace,
            ..EvolveConfig::default()
        };
        let policy = Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
            RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
            TEST_EVALUATOR_RELEASE_SHA256,
        )
        .unwrap();
        let baseline = evolve_lite_race_with_gpu_policy(
            &x,
            &y,
            &crate::metrics::r2,
            &config,
            &NullObs,
            None,
            None,
        );
        let accelerated = evolve_lite_race_with_gpu_policy_and_exact_surrogate_dedup(
            &x,
            &y,
            &crate::metrics::r2,
            &config,
            &NullObs,
            None,
            None,
            Some(policy),
        );

        let completed_bits = |outcome: &RaceOutcome| {
            outcome
                .candidates
                .iter()
                .filter_map(|candidate| {
                    candidate.rung_fold_scores[0].as_ref().map(|scores| {
                        (
                            candidate.dsl.clone(),
                            scores
                                .iter()
                                .map(|score| score.to_bits())
                                .collect::<Vec<_>>(),
                        )
                    })
                })
                .collect::<BTreeMap<_, _>>()
        };
        let baseline_bits = completed_bits(&baseline);
        let accelerated_bits = completed_bits(&accelerated);
        let mut common = 0;
        for (dsl, bits) in &baseline_bits {
            if let Some(accelerated) = accelerated_bits.get(dsl) {
                assert_eq!(bits, accelerated, "bounded common candidate {dsl}");
                common += 1;
            }
        }

        assert!(common > 0);
        assert!(accelerated_bits.len() >= baseline_bits.len());
        assert!(accelerated.rung0_exact_surrogate_dedup.requested);
        assert!(accelerated.rung0_exact_surrogate_dedup.active);
        assert!(
            accelerated
                .rung0_exact_surrogate_dedup
                .wall_clock_completion_may_accelerate
        );
        assert!(
            accelerated
                .rung0_exact_surrogate_dedup
                .completed_candidate_score_bits_exact
        );
        assert!(accelerated.rung0_exact_surrogate_dedup.reused_candidates >= 1);
        assert!(
            accelerated
                .rung0_exact_surrogate_dedup
                .physical_candidate_evaluations
                < accelerated.rung0_exact_surrogate_dedup.logical_candidates
        );
        assert!(accelerated
            .rung0_exact_surrogate_dedup
            .activation_blocker
            .is_none());
    }

    struct CancelAfterRepresentative {
        checks: AtomicUsize,
    }

    impl Observer for CancelAfterRepresentative {
        fn on_generation(&self, _stats: &GenStats) {}
        fn on_improvement(&self, _best: &Scored, _emitted: &Emitted, _stats: &GenStats) {}
        fn is_cancelled(&self) -> bool {
            self.checks.fetch_add(1, Ordering::Relaxed) + 1 >= 6
        }
    }

    #[test]
    fn exact_surrogate_dedup_never_scatters_or_falls_back_after_cancellation() {
        let (x, y) = synth(160, 0x0CA1_1CE1);
        let candidates = one_surrogate_candidates(2);
        let policy = Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
            RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
            TEST_EVALUATOR_RELEASE_SHA256,
        )
        .unwrap();
        let observer = CancelAfterRepresentative {
            checks: AtomicUsize::new(0),
        };
        let result = run_rung0_with_options_and_observer(
            &x,
            &y,
            &candidates,
            None,
            None,
            Some(policy),
            &observer,
        );

        assert!(result.0.iter().all(Option::is_none));
        assert_eq!(result.1.physical_candidate_evaluations, 1);
        assert_eq!(result.1.reused_candidates, 0);
        assert_eq!(result.1.cancellation_rejections, 1);
        assert_eq!(result.1.rejected_chunks, 1);
        assert!(result.2.iter().all(Option::is_none));
    }

    #[test]
    fn rung0_trace_enrichment_replays_target_candidate_bit_exactly() {
        let (x, y) = synth(160, 0xA11C_E55E);
        let source = "(pipe X (target kind=rank_gauss) (standardize) (histgbm depth=4))";
        let GenomeIr::Pipeline(genome) = GenomeIr::parse(source).unwrap() else {
            unreachable!()
        };
        let candidates = vec![(genome.to_string(), genome)];
        let observed = run_rung0(&x, &y, &candidates, None)
            .pop()
            .flatten()
            .unwrap();
        let rows = permute_indices(x.len(), 7)[..x.len().min(RACE_RUNG_ROWS[0])].to_vec();
        let assignments = fold_assign(rows.len(), RACE_RUNG_FOLDS[0], "invent_r0");
        let trace =
            enrich_rung0_trace_candidate(&candidates[0].0, &x, &y, &rows, &assignments).unwrap();

        assert!(trace.surrogate_dsl.ends_with("(ridge))"));
        assert_eq!(trace.transform_prefixes.len(), 2);
        assert_eq!(trace.folds.len(), RACE_RUNG_FOLDS[0]);
        for (actual, replay) in observed.iter().zip(&trace.folds) {
            assert_eq!(actual.to_bits(), replay.cpu_replay_score.to_bits());
            assert!(replay.prepared_width > 0);
            assert!(replay.host_f64_bytes > replay.host_f32_bytes);
        }
    }

    #[test]
    fn rung0_trace_sequence_and_cpu_routes_are_dense() {
        let (x, y) = synth(96, 0x51A7_E001);
        let cfg = EvolveConfig {
            pop: 8,
            gens: 1,
            seed: 0x51A7_E001,
            fitness: FitnessMode::LiteRace,
            ..EvolveConfig::default()
        };
        let outcome = evolve_lite_race_with_gpu_policy(
            &x,
            &y,
            &crate::metrics::r2,
            &cfg,
            &NullObs,
            None,
            None,
        );
        let mut traced = outcome
            .candidates
            .iter()
            .filter_map(|candidate| candidate.r0_seq.map(|seq| (seq, candidate)))
            .collect::<Vec<_>>();
        traced.sort_by_key(|(seq, _)| *seq);
        assert_eq!(traced.len() as u64, outcome.rung_evals[0]);
        for (expected, (seq, candidate)) in traced.iter().enumerate() {
            assert_eq!(*seq, expected as u64);
            assert_eq!(candidate.r0_generation, Some(0));
            assert_eq!(candidate.r0_generation_slot, Some(expected));
            assert!(candidate
                .r0_fold_routes
                .as_ref()
                .unwrap()
                .iter()
                .all(|receipt| {
                    receipt.route == Rung0FoldRoute::CpuExact
                        && receipt.physical_h2d_bytes == 0
                        && receipt.physical_d2h_bytes == 0
                }));
        }
        assert_eq!(outcome.rung0_rows.len(), outcome.rung0_assignments.len());
        assert_ne!(
            outcome.rung0_rows,
            (0..outcome.rung0_rows.len()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn gpu_screen_requires_explicit_policy_and_routes_only_eligible_widths() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;

        let (x, y) = synth(160, 0xABCD_1234);
        let candidates = [
            "(pipe X (standardize) (ridge))",
            "(pipe X (poly_interact append=true) (ridge))",
            "(pipe X (target kind=rank_gauss) (standardize) (ridge))",
        ]
        .into_iter()
        .map(|source| {
            let GenomeIr::Pipeline(genome) = GenomeIr::parse(source).unwrap() else {
                unreachable!()
            };
            (genome.to_string(), genome)
        })
        .collect::<Vec<_>>();
        let calls = AtomicUsize::new(0);
        let seen_widths = Mutex::new(Vec::new());
        let backend = |problems: Vec<GpuRidgeScoreProblem>| {
            calls.fetch_add(1, Ordering::Relaxed);
            seen_widths
                .lock()
                .unwrap()
                .extend(problems.iter().map(|problem| problem.p));
            exact_backend(problems)
        };

        let cpu = run_rung0(&x, &y, &candidates, None);
        let fail_closed = run_rung0_with_policy(&x, &y, &candidates, Some(&backend), None);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        for (cpu_scores, fail_closed_scores) in cpu.iter().zip(&fail_closed) {
            assert_eq!(
                cpu_scores.as_ref().map(|scores| scores
                    .iter()
                    .map(|score| score.to_bits())
                    .collect::<Vec<_>>()),
                fail_closed_scores.as_ref().map(|scores| scores
                    .iter()
                    .map(|score| score.to_bits())
                    .collect::<Vec<_>>())
            );
        }

        let below_batch = run_rung0_with_policy(
            &x,
            &y,
            &candidates,
            Some(&backend),
            GpuRidgeScorePolicy::new(7, 3),
        );
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        for (cpu_scores, below_batch_scores) in cpu.iter().zip(&below_batch) {
            assert_eq!(
                cpu_scores.as_ref().map(|scores| scores
                    .iter()
                    .map(|score| score.to_bits())
                    .collect::<Vec<_>>()),
                below_batch_scores.as_ref().map(|scores| scores
                    .iter()
                    .map(|score| score.to_bits())
                    .collect::<Vec<_>>())
            );
        }

        let policy = GpuRidgeScorePolicy::new(7, 2).unwrap();
        let hybrid = run_rung0_with_policy(&x, &y, &candidates, Some(&backend), Some(policy));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        let widths = seen_widths.lock().unwrap();
        assert_eq!(widths.len(), RACE_RUNG_FOLDS[0]);
        assert!(widths.iter().all(|width| *width >= policy.min_p));
        drop(widths);
        assert_same_shape(&cpu, &hybrid);
        for (cpu_scores, hybrid_scores) in cpu.iter().zip(&hybrid) {
            if let (Some(cpu_scores), Some(hybrid_scores)) = (cpu_scores, hybrid_scores) {
                for (cpu_score, hybrid_score) in cpu_scores.iter().zip(hybrid_scores) {
                    assert!((cpu_score - hybrid_score).abs() <= PARITY_ABS_TOL);
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct EvolveConfig {
    pub pop: usize,
    pub gens: usize,
    /// CV folds during search (kept small for speed; finalists get proper CV downstream).
    /// Used by the v1 surrogate path; the lite-race uses its own per-rung fold counts.
    pub folds: usize,
    pub seed: u64,
    /// Max transform stages (excluding the head).
    pub max_transforms: usize,
    /// Hard wall-clock cap in seconds (0 = none).
    pub time_budget_secs: f64,
    /// Canonical external pipeline genomes inserted after the champion in gen0.
    pub seed_genomes: Vec<String>,
    /// Fitness mode: v1 ridge surrogate (default) or the v2 multi-rung lite-race.
    pub fitness: FitnessMode,
    /// Optional known-good score for this dataset. LiteRace only: once any rung-2 score clears
    /// `target_score + RACE_TARGET_MARGIN` the generation loop exits early. Ignored by v1.
    pub target_score: Option<f64>,
    /// Rank and promote by the FOLD-VARIANCE-PENALIZED score instead of the plain fold mean.
    ///
    /// Search has always ranked by `mean(fold_scores)` while the gate, the league, and the
    /// leaderboard all score `max(oof,-1) - 0.25*min(population_std(folds),1)`. Those are
    /// different objectives, and the penalty term is not small: measured across ten hard
    /// datasets it spans 0.001-0.037, which is larger than the entire margin on six of them.
    /// A genome with a strong mean and unstable folds therefore wins the search and then loses
    /// the gate, and every generation spent finding it is wasted.
    ///
    /// Off by default so the existing lane is bit-identical; enable per-arm to A/B it.
    /// LiteRace only — the v1 surrogate is head-blind, so its variance carries no signal.
    pub rank_robust: bool,
    /// Add the difficult-ten high-regularization and underfit starters to
    /// generation zero. Off by default to preserve the legacy population.
    pub hard_ten_priors: bool,
    /// Draw generated transforms from TRANSFORMS ++ EXPANSION_TRANSFORMS and
    /// add EXPANSION_STARTERS to generation zero. Off by default: the legacy
    /// vocabulary array's length is the modulus of every recorded stream.
    pub expansion_vocab: bool,
    /// How the search treats the faults of `map` and `derive` stages. `Legacy` by default, which
    /// runs the search unchanged; see [`FaultPolicy`].
    pub fault_policy: FaultPolicy,
    /// How the next generation's parents are drawn. `Tournament3` by default, which draws exactly
    /// as the search always has; see [`ParentSelection`].
    pub selection: ParentSelection,
    /// How the surrogate search scores candidates. `Refit` by default; `Certified` gives the
    /// same scores with less work; see [`RidgeScoring`].
    pub ridge_scoring: RidgeScoring,
}

impl Default for EvolveConfig {
    fn default() -> Self {
        EvolveConfig {
            pop: 64,
            gens: 60,
            folds: 3,
            seed: 0x1EA7_BEEF,
            max_transforms: 6,
            time_budget_secs: 0.0,
            seed_genomes: Vec::new(),
            fitness: FitnessMode::Surrogate,
            target_score: None,
            rank_robust: false,
            hard_ten_priors: false,
            expansion_vocab: false,
            fault_policy: FaultPolicy::Legacy,
            selection: ParentSelection::Tournament3,
            ridge_scoring: RidgeScoring::Refit,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Scored {
    /// Canonical DSL text of the genome.
    pub dsl: String,
    /// Mean CV score (caller's metric, higher = better).
    pub cv_mean: f64,
    /// Per-fold CV scores.
    pub fold_scores: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct GenStats {
    pub gen: usize,
    pub gens: usize,
    pub evals: u64,
    pub evals_per_sec: f64,
    pub tokens_total: f64,
    pub tokens_per_sec: f64,
    pub best_cv: f64,
    /// Current population fitness (sorted desc, non-finite filtered), for live visuals.
    pub pop_fitness: Vec<f64>,
    /// The share of this generation's distinct candidates that the fault policy has rejected
    /// ([`FaultPolicy`]). 0.0 under `Legacy`, which rejects nothing.
    pub hard_rejected_share: f64,
    /// Distinct candidates the fault policy has rejected since the run started. 0 under `Legacy`.
    pub hard_rejected: u64,
    /// The mean domain rate of the ranked candidates, each taken from the evaluation its score
    /// comes from. 0.0 under `Legacy`, which measures no rate.
    pub pop_domain_rate: f64,
    /// The same mean over the elites, the best ranked candidates that pass unchanged into the next
    /// generation. 0.0 under `Legacy`.
    pub elite_domain_rate: f64,
}

/// Streaming observer: per-generation stats, per-improvement code drops, cancellation.
pub trait Observer: Sync {
    fn on_generation(&self, stats: &GenStats);
    fn on_improvement(&self, best: &Scored, emitted: &Emitted, stats: &GenStats);
    /// Fired for a THROTTLED sample of individual candidate evaluations (not just new
    /// bests), so the UI can show the search churning through real algorithms with their
    /// scores as fast as it can render. `code` is Some only for the slower code-sample
    /// cadence (the emitted Python of that candidate); None for the fast score-only ticks.
    /// Default no-op so non-streaming observers are unaffected.
    fn on_eval(&self, _dsl: &str, _score: f64, _code: Option<&str>) {}
    fn is_cancelled(&self) -> bool {
        false
    }
    /// Fired for each evaluation a non-legacy [`FaultPolicy`] rejects, with the candidate's
    /// canonical text and the reason, in the order the candidates were evaluated. A rejected
    /// candidate is not evaluated again in later generations or at later rungs; only the
    /// surrogate search evaluates a program twice, when it appears twice in one generation, as it
    /// always has. Never fired under `Legacy`. Default no-op.
    fn on_reject(&self, _dsl: &str, _reason: &Rejection) {}
    /// Fired once per generation whose parents [`ParentSelection::EpsilonLexicase`] draws, after
    /// its last parent. Never fired under `Tournament3`. Default no-op.
    fn on_selection(&self, _receipt: &SelectionReceipt) {}
    /// Fired after each generation scored by [`RidgeScoring::Certified`], with the run's counters
    /// so far. Never fired under `Refit`. Default no-op.
    fn on_scoring(&self, _receipt: &ScoringReceipt) {}
}

/// The certified scoring's counters so far, as [`Observer::on_scoring`] reports them (B11).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScoringReceipt {
    pub generation: usize,
    /// New candidates scored since the run started.
    pub candidates: u64,
    /// Of those, candidates whose ridge surrogate was already scored, so no fold was fitted.
    pub surrogate_reuses: u64,
    /// Transform stages served by the stage cache, and fitted, over every fold fit. These are
    /// instrumentation and depend on thread scheduling (two threads building the same prefix both
    /// miss); the counts above and every score do not.
    pub stage_hits: u64,
    pub stage_misses: u64,
}

/// One generation's lexicase parent selection, as [`Observer::on_selection`] reports it.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectionReceipt {
    pub generation: usize,
    /// The distinct ranked candidates in the pool; parents index this ranking.
    pub pool: usize,
    /// The selection and its trace, or why the generation fell back to tournament-3: the pool's
    /// cases could not form one matrix.
    pub selection: Result<Selection, String>,
}

const EVAL_CODE_SAMPLE_MS: u128 = 83;
const EVAL_SCORE_SAMPLE_MS: u128 = 60;

// ---------------------------------------------------------------------------
// Random genome construction + mutation
// ---------------------------------------------------------------------------

fn pick<'a>(rng: &mut Pcg32, items: &'a [&'a str]) -> &'a str {
    items[(rng.next_u32() as usize) % items.len()]
}

fn rand_params(rng: &mut Pcg32, op: &str) -> BTreeMap<String, ParamValue> {
    let mut p = BTreeMap::new();
    let int = |rng: &mut Pcg32, lo: i64, hi: i64| -> ParamValue {
        ParamValue::Int(lo + (rng.next_u32() as i64) % (hi - lo + 1))
    };
    // Discrete lotteries over the exact value grids the engine's own search uses.
    let pick_i = |rng: &mut Pcg32, choices: &[i64]| -> ParamValue {
        ParamValue::Int(choices[(rng.next_u32() as usize) % choices.len()])
    };
    let pick_f = |rng: &mut Pcg32, choices: &[f64]| -> ParamValue {
        ParamValue::Float(choices[(rng.next_u32() as usize) % choices.len()])
    };
    match op {
        "quantile_bins" => {
            p.insert("bins".into(), int(rng, 4, 32));
        }
        "bin_onehot" | "target_bin_encode" => {
            p.insert("bins".into(), int(rng, 4, 16));
        }
        "hinge" => {
            p.insert("knots".into(), int(rng, 2, 8));
        }
        "winsor_clip" => {
            p.insert("q".into(), ParamValue::Float(rng.range_f64(0.01, 0.2)));
        }
        "cheb" => {
            p.insert("degree".into(), int(rng, 2, 8));
        }
        "rff_cos" | "rff_sin" | "oblique_hinge" => {
            p.insert("n".into(), int(rng, 16, 96));
            p.insert("gamma".into(), ParamValue::Float(rng.range_f64(0.1, 2.0)));
            p.insert("seed".into(), int(rng, 1, 9999));
        }
        "rand_proj" => {
            p.insert("k".into(), int(rng, 8, 64));
            p.insert("seed".into(), int(rng, 1, 9999));
        }
        "landmark_dist" | "nystrom" => {
            p.insert("k".into(), int(rng, 8, 48));
            p.insert("gamma".into(), ParamValue::Float(rng.range_f64(0.1, 2.0)));
            p.insert("seed".into(), int(rng, 1, 9999));
        }
        "pair_prod" => {
            p.insert("topk".into(), int(rng, 4, 16));
        }
        "svd_project" => {
            p.insert("k".into(), int(rng, 4, 32));
            p.insert("seed".into(), int(rng, 1, 9999));
        }
        // Expansion vocabulary: drawn only when `expansion_vocab` is on, so
        // these arms are inert for every legacy stream.
        "impute" => {
            let strategies = ["median", "median", "mean"];
            let strategy = strategies[(rng.next_u32() as usize) % strategies.len()];
            p.insert("strategy".into(), ParamValue::Str(strategy.to_string()));
            if rng.next_f64() < 0.5 {
                p.insert("add_indicator".into(), ParamValue::Bool(true));
            }
        }
        "symlog" => {
            p.insert("c".into(), pick_f(rng, &[0.1, 1.0, 10.0]));
        }
        "select_corr" => {
            p.insert("k".into(), pick_i(rng, &[16, 32, 64, 128]));
            let by = ["corr", "corr", "var"];
            let by = by[(rng.next_u32() as usize) % by.len()];
            p.insert("by".into(), ParamValue::Str(by.to_string()));
        }
        "pair_ratio" => {
            p.insert("topk".into(), int(rng, 4, 16));
        }
        "spline" => {
            p.insert("knots".into(), int(rng, 3, 8));
        }
        // heads
        "histgbm" | "gbm" | "ensemble_ridge_gbm" => {
            p.insert("rounds".into(), int(rng, 40, 300));
            p.insert("depth".into(), int(rng, 2, 6));
            p.insert("lr".into(), ParamValue::Float(rng.range_f64(0.03, 0.25)));
        }
        // Deep bagged boosting: leaf-wise trees with row/column subsampling, a
        // min-leaf guard, and a held-out early-stopping split (v2.1 knobs).
        "histgbm2" => {
            p.insert("leaves".into(), pick_i(rng, &[15, 31, 63, 127, 255]));
            p.insert("lr".into(), pick_f(rng, &[0.01, 0.03, 0.05, 0.08]));
            p.insert("bags".into(), pick_i(rng, &[3, 5]));
            p.insert("rounds".into(), pick_i(rng, &[500, 1000]));
            p.insert("row_subsample".into(), pick_f(rng, &[0.6, 0.7, 0.85, 1.0]));
            p.insert("col_subsample".into(), pick_f(rng, &[0.6, 0.8, 1.0]));
            p.insert("min_leaf".into(), pick_i(rng, &[1, 3, 5, 10, 20]));
            p.insert("val_pct".into(), pick_i(rng, &[15, 20]));
            if rng.next_f64() < 0.5 {
                p.insert("val_shuffle".into(), ParamValue::Bool(true));
            }
        }
        // Composite-kernel Gaussian process: m inducing points, a few marginal-
        // likelihood steps, and a kernel algebra over {rbf, rq, periodic,
        // linear} joined by sum (+) and product (*). Plain ARD ("rbf") is the
        // modal draw and carries no explicit kernel= param.
        "gpr" => {
            p.insert("m".into(), pick_i(rng, &[300, 400, 500]));
            p.insert("steps".into(), pick_i(rng, &[30, 40, 60]));
            let kernels = [
                "rbf",
                "rbf",
                "linear",
                "rq",
                "periodic",
                "rbf+linear",
                "rq+linear",
                "rbf*periodic",
                "periodic+linear",
            ];
            let kernel = kernels[(rng.next_u32() as usize) % kernels.len()];
            if kernel != "rbf" {
                p.insert("kernel".into(), ParamValue::Str(kernel.to_string()));
            }
        }
        // Nadaraya-Watson kernel smoother: k nearest anchors, RBF bandwidth gamma.
        "ksmooth" => {
            p.insert("k".into(), pick_i(rng, &[8, 16, 32, 64, 128]));
            p.insert("gamma".into(), pick_f(rng, &[0.5, 1.0, 2.0, 4.0]));
        }
        "elasticnet" => {
            p.insert("alpha".into(), ParamValue::Float(rng.range_f64(1e-4, 0.05)));
            p.insert(
                "l1_ratio".into(),
                ParamValue::Float(rng.range_f64(0.1, 0.9)),
            );
        }
        _ => {}
    }
    // transforms may append (keep original cols) or replace
    if !HEAD_OPS.contains(&op) && rng.next_f64() < 0.5 {
        p.insert("append".into(), ParamValue::Bool(true));
    }
    p
}

pub(crate) fn rand_stage_from(rng: &mut Pcg32, vocab: &[&str]) -> Stage {
    let op = pick(rng, vocab);
    Stage {
        op: op.to_string(),
        params: rand_params(rng, op),
    }
}

fn rand_stage(rng: &mut Pcg32) -> Stage {
    rand_stage_from(rng, &TRANSFORMS)
}

pub(crate) fn rand_head(rng: &mut Pcg32) -> Stage {
    let op = pick(rng, &HEADS);
    Stage {
        op: op.to_string(),
        params: rand_params(rng, op),
    }
}

fn rand_genome_from(rng: &mut Pcg32, max_transforms: usize, vocab: &[&str]) -> DslExpr {
    let n = 1 + (rng.next_u32() as usize) % max_transforms.clamp(1, 6);
    let mut stages: Vec<Stage> = (0..n).map(|_| rand_stage_from(rng, vocab)).collect();
    stages.push(rand_head(rng));
    DslExpr { stages }
}

fn rand_genome(rng: &mut Pcg32, max_transforms: usize) -> DslExpr {
    rand_genome_from(rng, max_transforms, &TRANSFORMS)
}

/// The transform vocabulary a search run draws from: the frozen legacy array,
/// plus the expansion ops when the config opts in. With the flag off the
/// contents (and so every draw) are identical to the legacy array.
fn generation_vocab(cfg: &EvolveConfig) -> Vec<&'static str> {
    let mut vocab: Vec<&'static str> = TRANSFORMS.to_vec();
    if cfg.expansion_vocab {
        vocab.extend_from_slice(&EXPANSION_TRANSFORMS);
    }
    vocab
}

/// Split a genome into (transforms, head). The head is always present in generated genomes.
fn split(g: &DslExpr) -> (Vec<Stage>, Option<Stage>) {
    match g.stages.last() {
        Some(last) if HEAD_OPS.contains(&last.op.as_str()) => {
            (g.stages[..g.stages.len() - 1].to_vec(), Some(last.clone()))
        }
        _ => (g.stages.clone(), None),
    }
}

fn jitter_params(rng: &mut Pcg32, stage: &mut Stage) {
    let keys: Vec<String> = stage.params.keys().cloned().collect();
    if keys.is_empty() {
        // maybe flip append on
        stage.params = rand_params(rng, &stage.op);
        return;
    }
    let key = keys[(rng.next_u32() as usize) % keys.len()].clone();
    if let Some(v) = stage.params.get(&key) {
        let jittered = match v {
            ParamValue::Int(i) => {
                let delta = ((rng.next_u32() % 5) as i64) - 2;
                ParamValue::Int((*i + delta).max(1))
            }
            ParamValue::Float(f) => ParamValue::Float((f * rng.range_f64(0.6, 1.5)).max(1e-6)),
            ParamValue::Bool(b) => ParamValue::Bool(!b),
            ParamValue::Str(s) => ParamValue::Str(s.clone()),
        };
        stage.params.insert(key, jittered);
    }
}

/// v2 param jitter: log-scale for positive floats (`v * exp(u)`, u uniform in
/// [-ln 2.5, +ln 2.5] — symmetric multiplicative spread, right for
/// lr/lambda/gamma-style params where v1's ×[0.6, 1.5] is narrow and
/// upward-biased), logit-space for `l1_ratio` (stays inside (0, 1)), and the
/// original int/bool/str behavior. Exactly the same RNG-draw count per param
/// as v1, so the two are clean drop-in twins; the frozen `jitter_params`
/// stays byte-identical for every recorded v1 stream. Schema clamps re-apply
/// at plan/fit time exactly as they do for v1 jitter.
pub(crate) fn jitter_params_v2(rng: &mut Pcg32, stage: &mut Stage) {
    let keys: Vec<String> = stage.params.keys().cloned().collect();
    if keys.is_empty() {
        // maybe flip append on (mirrors v1)
        stage.params = rand_params(rng, &stage.op);
        return;
    }
    let key = keys[(rng.next_u32() as usize) % keys.len()].clone();
    if let Some(v) = stage.params.get(&key) {
        let span = 2.5f64.ln();
        let jittered = match v {
            ParamValue::Int(i) => {
                let delta = ((rng.next_u32() % 5) as i64) - 2;
                ParamValue::Int((*i + delta).max(1))
            }
            ParamValue::Float(f) if key == "l1_ratio" => {
                let p = f.clamp(0.01, 0.99);
                let logit = (p / (1.0 - p)).ln() + rng.range_f64(-span, span);
                ParamValue::Float(1.0 / (1.0 + (-logit).exp()))
            }
            ParamValue::Float(f) => {
                ParamValue::Float((f * rng.range_f64(-span, span).exp()).max(1e-6))
            }
            ParamValue::Bool(b) => ParamValue::Bool(!b),
            ParamValue::Str(s) => ParamValue::Str(s.clone()),
        };
        stage.params.insert(key, jittered);
    }
}

/// First index i where stages i and i+1 are canonical-equal (same op, same
/// params — Stage derives PartialEq over its canonical fields).
fn adjacent_duplicate(tf: &[Stage]) -> Option<usize> {
    (0..tf.len().saturating_sub(1)).find(|&i| tf[i] == tf[i + 1])
}

/// v2 mutation over 7 equally-drawn outcomes (arm = draw % 7): {0, 1} param
/// jitter via `jitter_params_v2` (2/7 base mass, exactly v1's inner
/// stage-vs-head split), {2} insert drawn from `vocab` (legacy ++ expansion
/// when the caller opts in), {3} drop, {4} swap-adjacent, {5} head replace,
/// {6} simplify — delete the LATER stage of the first adjacent
/// canonical-duplicate pair (dissolves the observed 4x-standardize
/// pathology); with no such pair the draw falls through deterministically to
/// param jitter, so no draw is ever a no-op beyond v1's own degenerate arms.
/// The frozen `mutate` stays byte-identical for every recorded v1 stream.
fn mutate_v2(rng: &mut Pcg32, g: &DslExpr, max_transforms: usize, vocab: &[&str]) -> DslExpr {
    let (mut tf, head) = split(g);
    let mut head = head.unwrap_or_else(|| rand_head(rng));
    let mut arm = rng.next_u32() % 7;
    if arm == 6 {
        match adjacent_duplicate(&tf) {
            Some(i) => {
                tf.remove(i + 1);
                arm = 7; // handled; the match below is a no-op
            }
            None => arm = 0, // deterministic fall-through to param jitter
        }
    }
    match arm {
        0 | 1 => {
            if !tf.is_empty() && rng.next_f64() < 0.7 {
                let i = (rng.next_u32() as usize) % tf.len();
                jitter_params_v2(rng, &mut tf[i]);
            } else {
                jitter_params_v2(rng, &mut head);
            }
        }
        2 => {
            if tf.len() < max_transforms {
                let i = (rng.next_u32() as usize) % (tf.len() + 1);
                tf.insert(i, rand_stage_from(rng, vocab));
            }
        }
        3 => {
            if !tf.is_empty() {
                let i = (rng.next_u32() as usize) % tf.len();
                tf.remove(i);
            }
        }
        4 => {
            if tf.len() >= 2 {
                let i = (rng.next_u32() as usize) % (tf.len() - 1);
                tf.swap(i, i + 1);
            }
        }
        5 => {
            head = rand_head(rng);
        }
        _ => {}
    }
    let mut stages = tf;
    stages.push(head);
    DslExpr { stages }
}

fn mutate(rng: &mut Pcg32, g: &DslExpr, max_transforms: usize) -> DslExpr {
    let (mut tf, head) = split(g);
    let mut head = head.unwrap_or_else(|| rand_head(rng));
    match rng.next_u32() % 6 {
        // param jitter on a random stage (transform or head)
        0 | 1 => {
            if !tf.is_empty() && rng.next_f64() < 0.7 {
                let i = (rng.next_u32() as usize) % tf.len();
                jitter_params(rng, &mut tf[i]);
            } else {
                jitter_params(rng, &mut head);
            }
        }
        // insert a random transform
        2 => {
            if tf.len() < max_transforms {
                let i = (rng.next_u32() as usize) % (tf.len() + 1);
                tf.insert(i, rand_stage(rng));
            }
        }
        // drop a transform
        3 => {
            if !tf.is_empty() {
                let i = (rng.next_u32() as usize) % tf.len();
                tf.remove(i);
            }
        }
        // swap two adjacent transforms
        4 => {
            if tf.len() >= 2 {
                let i = (rng.next_u32() as usize) % (tf.len() - 1);
                tf.swap(i, i + 1);
            }
        }
        // replace the head
        _ => {
            head = rand_head(rng);
        }
    }
    let mut stages = tf;
    stages.push(head);
    DslExpr { stages }
}

fn crossover(rng: &mut Pcg32, a: &DslExpr, b: &DslExpr, max_transforms: usize) -> DslExpr {
    let (ta, ha) = split(a);
    let (tb, hb) = split(b);
    let cut_a = (rng.next_u32() as usize) % (ta.len() + 1);
    let cut_b = (rng.next_u32() as usize) % (tb.len() + 1);
    let mut tf: Vec<Stage> = ta[..cut_a].to_vec();
    tf.extend_from_slice(&tb[cut_b..]);
    tf.truncate(max_transforms);
    let head = if rng.next_f64() < 0.5 { ha } else { hb };
    let mut stages = tf;
    stages.push(head.unwrap_or_else(|| rand_head(rng)));
    DslExpr { stages }
}

/// Build one deterministic grammar-random pipeline using the evolution engine's RNG stream.
///
/// The returned source is canonical pipeline DSL. This is an additive generation-0 seam; it does
/// not alter or share RNG state with any evolution entry point.
pub fn random_canonical_pipeline(seed: u64, max_transforms: usize) -> String {
    let mut rng = Pcg32::new(seed, 0x5EED);
    GenomeIr::Pipeline(rand_genome(&mut rng, max_transforms)).canonical()
}

/// Deterministically mutate one valid pipeline and return canonical pipeline DSL.
///
/// A fresh RNG stream is derived solely from `seed`, so callers can record and replay each
/// generation-0 offspring independently.
pub fn mutate_canonical_pipeline(
    source: &str,
    seed: u64,
    max_transforms: usize,
) -> Result<String, String> {
    let parent = parse_canonical_pipeline(source)?;
    let mut rng = Pcg32::new(seed, 0x5EED);
    Ok(GenomeIr::Pipeline(mutate(&mut rng, &parent, max_transforms)).canonical())
}

/// Deterministically cross two valid pipelines and return canonical pipeline DSL.
///
/// A fresh RNG stream is derived solely from `seed`, so parent order, seed, and
/// `max_transforms` completely determine the child.
pub fn crossover_canonical_pipelines(
    left: &str,
    right: &str,
    seed: u64,
    max_transforms: usize,
) -> Result<String, String> {
    let left = parse_canonical_pipeline(left)?;
    let right = parse_canonical_pipeline(right)?;
    let mut rng = Pcg32::new(seed, 0x5EED);
    Ok(GenomeIr::Pipeline(crossover(&mut rng, &left, &right, max_transforms)).canonical())
}

/// v2 twin of `mutate_canonical_pipeline`: expansion-aware insert vocabulary,
/// log-scale param jitter, and a simplify arm that dissolves adjacent
/// duplicate stages. Pure function of (source, seed, max_transforms,
/// expansion_vocab) on the same Pcg32 seeding pattern; the v1 entry point and
/// every recorded stream stay byte-identical.
pub fn mutate_canonical_pipeline_v2(
    source: &str,
    seed: u64,
    max_transforms: usize,
    expansion_vocab: bool,
) -> Result<String, String> {
    let parent = parse_canonical_pipeline(source)?;
    let mut vocab: Vec<&'static str> = TRANSFORMS.to_vec();
    if expansion_vocab {
        vocab.extend_from_slice(&EXPANSION_TRANSFORMS);
    }
    let mut rng = Pcg32::new(seed, 0x5EED);
    Ok(GenomeIr::Pipeline(mutate_v2(&mut rng, &parent, max_transforms, &vocab)).canonical())
}

/// Explicit mutation arms for train-only search protocols that seal an
/// operator distribution independently from the legacy v2 sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanonicalMutationArm {
    ParameterJitter,
    Insert,
    Delete,
    AdjacentSwap,
    HeadReplacement,
    Simplify,
}

/// Apply exactly one requested v2-compatible mutation arm.
///
/// This is additive: the frozen v1/v2 entry points and their RNG streams are
/// unchanged. A structurally inapplicable arm may return the canonical parent;
/// callers that require semantic uniqueness must reject and retry it.
pub fn mutate_canonical_pipeline_with_arm(
    source: &str,
    seed: u64,
    max_transforms: usize,
    expansion_vocab: bool,
    arm: CanonicalMutationArm,
) -> Result<String, String> {
    let parent = parse_canonical_pipeline(source)?;
    let mut rng = Pcg32::new(seed, 0x5EED);
    let (mut transforms, head) = split(&parent);
    let mut head = head.unwrap_or_else(|| rand_head(&mut rng));
    let mut vocab: Vec<&'static str> = TRANSFORMS.to_vec();
    if expansion_vocab {
        vocab.extend_from_slice(&EXPANSION_TRANSFORMS);
    }
    match arm {
        CanonicalMutationArm::ParameterJitter => {
            if !transforms.is_empty() && rng.next_f64() < 0.7 {
                let index = (rng.next_u32() as usize) % transforms.len();
                jitter_params_v2(&mut rng, &mut transforms[index]);
            } else {
                jitter_params_v2(&mut rng, &mut head);
            }
        }
        CanonicalMutationArm::Insert => {
            if transforms.len() < max_transforms {
                let index = (rng.next_u32() as usize) % (transforms.len() + 1);
                transforms.insert(index, rand_stage_from(&mut rng, &vocab));
            }
        }
        CanonicalMutationArm::Delete => {
            if !transforms.is_empty() {
                let index = (rng.next_u32() as usize) % transforms.len();
                transforms.remove(index);
            }
        }
        CanonicalMutationArm::AdjacentSwap => {
            if transforms.len() >= 2 {
                let index = (rng.next_u32() as usize) % (transforms.len() - 1);
                transforms.swap(index, index + 1);
            }
        }
        CanonicalMutationArm::HeadReplacement => head = rand_head(&mut rng),
        CanonicalMutationArm::Simplify => {
            if let Some(index) = adjacent_duplicate(&transforms) {
                transforms.remove(index + 1);
            } else if !transforms.is_empty() {
                let index = (rng.next_u32() as usize) % transforms.len();
                transforms.remove(index);
            }
        }
    }
    transforms.push(head);
    Ok(GenomeIr::Pipeline(DslExpr { stages: transforms }).canonical())
}

fn parse_canonical_pipeline(source: &str) -> Result<DslExpr, String> {
    match GenomeIr::parse(source).map_err(|error| error.to_string())? {
        GenomeIr::Pipeline(genome) => Ok(genome),
        _ => Err("expected a pipeline genome".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Fitness
// ---------------------------------------------------------------------------

/// Deterministic row->fold assignment (FNV-1a over (key, row), matching folds.rs behavior).
fn fold_assign(n: usize, k: usize, key: &str) -> Vec<usize> {
    (0..n)
        .map(|row| crate::folds::fold_of(key, row, k))
        .collect()
}

/// k-fold CV of one genome: full refit inside every fold, scored by `metric` (higher = better).
/// Returns None for genomes that fail to fit or produce non-finite scores.
/// Widest transform knobs are clamped during search so the transformed frame stays cheap to build.
const SEARCH_MAX_DEGREE: i64 = 4;
const SEARCH_MAX_N: i64 = 64;

/// The heart of the engine's speed: during search we DO NOT train each genome's real (expensive)
/// head. We replace it with a cheap closed-form RIDGE surrogate and measure how linearly-predictable
/// the genome's TRANSFORMS have made the target. That is the whole bet — the transforms' job is to
/// linearize structure (cheb/poly/rff/quantile all do exactly this), so ridge is a principled and
/// millisecond-fast fitness. Only the top-K leaders are later gated to their real head + proper CV.
/// Also clamps the widest transform knobs so building the frame stays cheap.
fn search_surrogate(g: &DslExpr) -> DslExpr {
    let mut c = g.clone();
    for st in &mut c.stages {
        for (key, cap) in [("degree", SEARCH_MAX_DEGREE), ("n", SEARCH_MAX_N)] {
            if let Some(v) = st.params.get_mut(key) {
                if v.as_u64().map(|r| r as i64 > cap).unwrap_or(false) {
                    *v = ParamValue::Int(cap);
                }
            }
        }
    }
    match c.stages.last_mut() {
        Some(last) if HEAD_OPS.contains(&last.op.as_str()) => {
            last.op = "ridge".to_string();
            last.params.clear();
        }
        // No head present (all-transform genome) — append a ridge so it is scorable.
        _ => c.stages.push(Stage {
            op: "ridge".to_string(),
            params: BTreeMap::new(),
        }),
    }
    c
}

/// Lite head budgets for the v2 race: `(head op, param overrides)` applied by `lite_budget`.
/// One table so the knobs are tunable in one place. Heads absent from the table
/// (ridge / ridge_gcv / logistic / ksmooth / stack / ensemble / ensemble_ridge_gbm) are already
/// cheap at race row counts and pass through unchanged.
pub const LITE_HEAD_BUDGETS: [(&str, &[(&str, i64)]); 6] = [
    ("histgbm2", &[("rounds", 40), ("bags", 1)]),
    ("histgbm", &[("rounds", 25)]),
    ("gbm", &[("rounds", 25)]),
    ("gpr", &[("steps", 5), ("m", 96)]),
    ("krr", &[("m", 96)]),
    ("elasticnet", &[("sweeps", 15)]),
];

/// Lite-budget variant of a genome for rungs 1-2 of the race: KEEPS the genome's real head — the
/// whole point, since the ridge surrogate is head-blind — but caps the head's compute budget via
/// `LITE_HEAD_BUDGETS` (tuning knobs like leaves/lr/kernel are preserved) and applies the same
/// transform clamps as `search_surrogate`. A lab study showed this low-fidelity probe of the true
/// objective scores Spearman 0.731 vs honest CV (surrogate: 0.373) at a fraction of full-fit cost.
fn lite_budget(g: &DslExpr) -> DslExpr {
    let mut c = g.clone();
    for st in &mut c.stages {
        for (key, cap) in [("degree", SEARCH_MAX_DEGREE), ("n", SEARCH_MAX_N)] {
            if let Some(v) = st.params.get_mut(key) {
                if v.as_u64().map(|r| r as i64 > cap).unwrap_or(false) {
                    *v = ParamValue::Int(cap);
                }
            }
        }
    }
    match c.stages.last_mut() {
        Some(last) if HEAD_OPS.contains(&last.op.as_str()) => {
            if let Some((_, overrides)) = LITE_HEAD_BUDGETS.iter().find(|(op, _)| *op == last.op) {
                for (key, val) in overrides.iter() {
                    last.params
                        .insert((*key).to_string(), ParamValue::Int(*val));
                }
            }
        }
        // No head present (all-transform genome) — append a ridge so it is scorable.
        _ => c.stages.push(Stage {
            op: "ridge".to_string(),
            params: BTreeMap::new(),
        }),
    }
    c
}

fn score_genome(
    g: &DslExpr,
    x: &Rows,
    y: &[f64],
    assign: &[usize],
    k: usize,
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
) -> Option<Vec<f64>> {
    let genome = GenomeIr::Pipeline(search_surrogate(g));
    let mut fold_scores = Vec::with_capacity(k);
    for fold in 0..k {
        let (mut xtr, mut ytr, mut xte, mut yte) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..y.len() {
            if assign[i] == fold {
                xte.push(x[i].clone());
                yte.push(y[i]);
            } else {
                xtr.push(x[i].clone());
                ytr.push(y[i]);
            }
        }
        if xte.is_empty() || xtr.is_empty() {
            return None;
        }
        let m = model::fit(&genome, &xtr, &ytr).ok()?;
        let pred = m.predict(&xte).ok()?;
        let s = metric(&yte, &pred);
        if !s.is_finite() {
            return None;
        }
        fold_scores.push(s);
    }
    Some(fold_scores)
}

// B10 lexicase: begin
/// [`score_genome`], also returning every row's out-of-fold prediction, by row (`NaN` for a row
/// that is in no fold). The program, folds, fits, predictions and metric calls are
/// `score_genome`'s, in the same order, so the fold scores are the same bits.
fn score_genome_predictions(
    g: &DslExpr,
    x: &Rows,
    y: &[f64],
    assign: &[usize],
    k: usize,
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
) -> ScoredCases {
    let genome = GenomeIr::Pipeline(search_surrogate(g));
    let mut fold_scores = Vec::with_capacity(k);
    let mut predictions = vec![f64::NAN; y.len()];
    for fold in 0..k {
        let (mut xtr, mut ytr, mut xte, mut yte) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut held_out = Vec::new();
        for i in 0..y.len() {
            if assign[i] == fold {
                xte.push(x[i].clone());
                yte.push(y[i]);
                held_out.push(i);
            } else {
                xtr.push(x[i].clone());
                ytr.push(y[i]);
            }
        }
        if xte.is_empty() || xtr.is_empty() {
            return None;
        }
        let m = model::fit(&genome, &xtr, &ytr).ok()?;
        let pred = m.predict(&xte).ok()?;
        let s = metric(&yte, &pred);
        if !s.is_finite() {
            return None;
        }
        fold_scores.push(s);
        for (&row, &value) in held_out.iter().zip(&pred) {
            predictions[row] = value;
        }
    }
    Some((fold_scores, predictions))
}

/// Fold scores with a per-row vector beside them (out-of-fold predictions or residual cases), or
/// `None` where the legacy scorer gives `None`.
type ScoredCases = Option<(Vec<f64>, Vec<f64>)>;

/// [`score_genome`], also returning the signed out-of-fold residual `y[r] - prediction` of each
/// row of `case_rows`, in that order.
fn score_genome_cases(
    g: &DslExpr,
    x: &Rows,
    y: &[f64],
    assign: &[usize],
    k: usize,
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    case_rows: &[usize],
) -> ScoredCases {
    let (fold_scores, predictions) = score_genome_predictions(g, x, y, assign, k, metric)?;
    let residuals = case_rows
        .iter()
        .map(|&row| y[row] - predictions[row])
        .collect();
    Some((fold_scores, residuals))
}

/// The surrogate search's legacy scoring of `unseen` under lexicase selection:
/// [`score_genome_cases`] in parallel, keeping each scored candidate's cases in `cases`.
#[allow(clippy::too_many_arguments)]
fn score_unseen_with_cases(
    unseen: Vec<(String, DslExpr)>,
    x: &Rows,
    y: &[f64],
    assign: &[usize],
    k: usize,
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    case_rows: &[usize],
    cases: &mut HashMap<String, Vec<f64>>,
) -> Vec<(String, DslExpr, Option<Vec<f64>>)> {
    let scored: Vec<(String, DslExpr, ScoredCases)> = unseen
        .into_par_iter()
        .map(|(key, g)| {
            let scored = score_genome_cases(&g, x, y, assign, k, metric, case_rows);
            (key, g, scored)
        })
        .collect();
    scored
        .into_iter()
        .map(|(key, g, scored)| {
            let scores = scored.map(|(scores, residuals)| {
                cases.insert(key.clone(), residuals);
                scores
            });
            (key, g, scores)
        })
        .collect()
}
// B10 lexicase: end

// B11 ridge scoring: begin
/// The certified scorer of one surrogate search: the surrogate memo and the fold-scoped stage
/// cache ([`RidgeScoring::Certified`]). Every score it returns is [`score_genome`]'s, bit for bit.
/// Its keys pin the surrogate and the stage chain, not the data: they are sound because one scorer
/// serves one search run, whose rows, labels, folds and case rows are fixed.
struct CertifiedScoring {
    cache: crate::stage_cache::StageCache,
    /// A surrogate's exact key ([`exact_surrogate_key`]) to its fold scores and, when lexicase
    /// cases are kept, its residuals on the run's case rows (fixed for the run).
    memo: HashMap<Vec<u8>, ScoredCases>,
    candidates: u64,
    reuses: u64,
}

impl CertifiedScoring {
    fn new() -> Self {
        Self {
            cache: crate::stage_cache::StageCache::new(CERTIFIED_STAGE_CACHE_BYTES),
            memo: HashMap::new(),
            candidates: 0,
            reuses: 0,
        }
    }

    /// Scores `unseen` as the legacy scorer does, keeping each candidate's lexicase cases on
    /// `case_rows` when given. Each distinct surrogate not yet in the memo is scored once, in
    /// parallel.
    #[allow(clippy::too_many_arguments)]
    fn score_unseen(
        &mut self,
        unseen: Vec<(String, DslExpr)>,
        x: &Rows,
        y: &[f64],
        assign: &[usize],
        k: usize,
        metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
        case_rows: Option<&[usize]>,
        cases: &mut HashMap<String, Vec<f64>>,
    ) -> Vec<(String, DslExpr, Option<Vec<f64>>)> {
        let surrogates: Vec<DslExpr> = unseen.iter().map(|(_, g)| search_surrogate(g)).collect();
        let exact: Vec<Vec<u8>> = surrogates.iter().map(exact_surrogate_key).collect();
        let mut queued = std::collections::HashSet::new();
        let todo: Vec<(Vec<u8>, DslExpr)> = surrogates
            .into_iter()
            .zip(&exact)
            .filter(|(_, id)| !self.memo.contains_key(*id) && queued.insert((*id).clone()))
            .map(|(surrogate, id)| (id.clone(), surrogate))
            .collect();
        let cache = &self.cache;
        let scored: Vec<(Vec<u8>, ScoredCases)> = todo
            .into_par_iter()
            .map(|(id, surrogate)| {
                let genome = GenomeIr::Pipeline(surrogate);
                let scored =
                    score_surrogate_cached(&genome, x, y, assign, k, metric, cache, case_rows);
                (id, scored)
            })
            .collect();
        let computed = scored.len() as u64;
        self.memo.extend(scored);
        self.candidates += unseen.len() as u64;
        self.reuses += unseen.len() as u64 - computed;
        unseen
            .into_iter()
            .zip(exact)
            .map(|((key, g), id)| {
                let entry = &self.memo[&id];
                if let (Some(_), Some((_, residuals))) = (case_rows, entry) {
                    cases.insert(key.clone(), residuals.clone());
                }
                let scores = entry.as_ref().map(|(scores, _)| scores.clone());
                (key, g, scores)
            })
            .collect()
    }

    fn receipt(&self, generation: usize) -> ScoringReceipt {
        let (stage_hits, stage_misses, _) = self.cache.stats();
        ScoringReceipt {
            generation,
            candidates: self.candidates,
            surrogate_reuses: self.reuses,
            stage_hits,
            stage_misses,
        }
    }
}

/// The exact key of a surrogate: its stage count, then each stage's exact encoding (op and every
/// parameter, floats as their IEEE-754 bits). Unlike canonical text, it separates surrogates that
/// print alike but differ in any parameter bit.
fn exact_surrogate_key(surrogate: &DslExpr) -> Vec<u8> {
    let mut key = (surrogate.stages.len() as u64).to_le_bytes().to_vec();
    for stage in &surrogate.stages {
        crate::stage_cache::write_exact_stage(stage, &mut key);
    }
    key
}

/// [`score_genome_predictions`] for an already built surrogate, each fold fitted through the
/// stage cache: the same folds, the same fits (bit-identical, [`model::fit_with_stage_cache`]),
/// predictions and metric calls. With `case_rows`, also the residuals `y - prediction` on those
/// rows, in that order, as [`score_genome_cases`] computes them; otherwise an empty vector.
#[allow(clippy::too_many_arguments)]
fn score_surrogate_cached(
    surrogate: &GenomeIr,
    x: &Rows,
    y: &[f64],
    assign: &[usize],
    k: usize,
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    cache: &crate::stage_cache::StageCache,
    case_rows: Option<&[usize]>,
) -> ScoredCases {
    let keep = case_rows.is_some();
    let mut fold_scores = Vec::with_capacity(k);
    let mut predictions = if keep {
        vec![f64::NAN; y.len()]
    } else {
        Vec::new()
    };
    for fold in 0..k {
        let (mut xtr, mut ytr, mut xte, mut yte) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut held_out = Vec::new();
        for i in 0..y.len() {
            if assign[i] == fold {
                xte.push(x[i].clone());
                yte.push(y[i]);
                held_out.push(i);
            } else {
                xtr.push(x[i].clone());
                ytr.push(y[i]);
            }
        }
        if xte.is_empty() || xtr.is_empty() {
            return None;
        }
        let ctx = crate::stage_cache::StageCacheCtx {
            context_tag: "evolve-surrogate".to_string(),
            folds: k,
            fold,
        };
        let m = model::fit_with_stage_cache(surrogate, &xtr, &ytr, cache, &ctx).ok()?;
        let pred = m.predict(&xte).ok()?;
        let s = metric(&yte, &pred);
        if !s.is_finite() {
            return None;
        }
        fold_scores.push(s);
        if keep {
            for (&row, &value) in held_out.iter().zip(&pred) {
                predictions[row] = value;
            }
        }
    }
    let residuals = case_rows.map_or_else(Vec::new, |rows| {
        rows.iter().map(|&row| y[row] - predictions[row]).collect()
    });
    Some((fold_scores, residuals))
}
// B11 ridge scoring: end

// ---------------------------------------------------------------------------
// Fault policy (kernel2 WP-1.5): the scorers and bookkeeping of the non-legacy policies. Under
// `FaultPolicy::Legacy` none of this runs.
// ---------------------------------------------------------------------------

/// One evaluation under a non-legacy [`FaultPolicy`].
enum Admitted {
    /// Scored: the policy's fold scores, and the fault totals of every pass.
    Scored(Vec<f64>, FaultTotals),
    /// Not scored, for a reason the legacy scorers share: an empty fold, a fit or prediction
    /// error, a score that is not finite (a penalized one included), or the budget or a
    /// cancellation stopping the evaluation.
    Failed,
    /// Rejected by the policy.
    Rejected(Rejection),
}

/// The fit a non-legacy policy runs: [`fit_traced`], except in tests.
type TracedFit<'fit> = dyn Fn(&GenomeIr, &Rows, &[f64]) -> anyhow::Result<(model::Model, PipelineTelemetry)>
    + Sync
    + 'fit;

/// The scope check of a non-legacy policy: the program as it is fitted, at the input width. A set
/// without rows is left to the folds, which fail on it as the legacy scorers do.
fn scope_rejection(genome: &GenomeIr, x: &Rows) -> Option<Rejection> {
    let width = x.first()?.len();
    validate_scope(genome, width).err().map(Rejection::Scope)
}

/// One fold under a non-legacy policy: fit and predict with the traced entry points and add both
/// passes to `faults`. `stop` runs between the fit and the prediction, where [`score_rung`] checks
/// its budget. `Err` ends the evaluation: a Hard fault rejects it, anything else fails it.
fn traced_fold(
    genome: &GenomeIr,
    (xtr, ytr): (&Rows, &[f64]),
    (xte, yte): (&Rows, &[f64]),
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    fit: &TracedFit<'_>,
    stop: &dyn Fn() -> bool,
    faults: &mut FaultTotals,
) -> Result<f64, Admitted> {
    let hard = |faults: &FaultTotals| Admitted::Rejected(Rejection::HardFault(faults.hard_classes));
    let (fitted, training) = fit(genome, xtr, ytr).map_err(|_| Admitted::Failed)?;
    faults.add_pass(&training);
    if faults.is_hard() {
        return Err(hard(faults));
    }
    if stop() {
        return Err(Admitted::Failed);
    }
    let (predictions, held_out) = predict_traced(&fitted, xte).map_err(|_| Admitted::Failed)?;
    faults.add_pass(&held_out);
    if faults.is_hard() {
        return Err(hard(faults));
    }
    let score = metric(yte, &predictions);
    if score.is_finite() {
        Ok(score)
    } else {
        Err(Admitted::Failed)
    }
}

/// The end of an evaluation whose every fold was scored without a Hard fault: the fold scores as
/// `policy` keeps them.
fn admit(policy: FaultPolicy, fold_scores: Vec<f64>, faults: FaultTotals) -> Admitted {
    let scores = match policy {
        FaultPolicy::Penalize {
            lambda_domain,
            lambda_saturation,
            ..
        } => {
            let penalized = fold_scores
                .iter()
                .map(|&score| faults.penalize(score, lambda_domain, lambda_saturation))
                .collect::<Option<Vec<f64>>>();
            match penalized {
                Some(scores) => scores,
                None => return Admitted::Failed,
            }
        }
        FaultPolicy::Legacy | FaultPolicy::RejectHard => fold_scores,
    };
    Admitted::Scored(scores, faults)
}

/// [`score_genome`] under a non-legacy [`FaultPolicy`]: the same program, folds and metric, with
/// the scope check before anything is fitted and every fold run by [`traced_fold`].
#[allow(clippy::too_many_arguments)]
fn score_genome_admitted(
    g: &DslExpr,
    x: &Rows,
    y: &[f64],
    assign: &[usize],
    k: usize,
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    policy: FaultPolicy,
    fit: &TracedFit<'_>,
) -> Admitted {
    let genome = GenomeIr::Pipeline(search_surrogate(g));
    if let Some(rejection) = scope_rejection(&genome, x) {
        return Admitted::Rejected(rejection);
    }
    let mut faults = FaultTotals::default();
    let mut fold_scores = Vec::with_capacity(k);
    for fold in 0..k {
        let (mut xtr, mut ytr, mut xte, mut yte) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..y.len() {
            if assign[i] == fold {
                xte.push(x[i].clone());
                yte.push(y[i]);
            } else {
                xtr.push(x[i].clone());
                ytr.push(y[i]);
            }
        }
        if xte.is_empty() || xtr.is_empty() {
            return Admitted::Failed;
        }
        match traced_fold(
            &genome,
            (&xtr, &ytr),
            (&xte, &yte),
            metric,
            fit,
            &|| false,
            &mut faults,
        ) {
            Ok(score) => fold_scores.push(score),
            Err(end) => return end,
        }
    }
    admit(policy, fold_scores, faults)
}

/// What a non-legacy [`FaultPolicy`] knows about the candidates of one run: the fault totals of
/// every scored evaluation, per canonical program and rung (the surrogate search uses rung 0), and
/// the programs it rejected. It is only looked up by program, never iterated, so the order of its
/// hash maps cannot reach a result.
struct Admission {
    policy: FaultPolicy,
    faults: HashMap<String, [Option<FaultTotals>; 3]>,
    rejected: std::collections::HashSet<String>,
}

/// The fault counters of one generation's [`GenStats`]; all zero under `Legacy`.
#[derive(Default)]
struct FaultCounters {
    hard_rejected_share: f64,
    hard_rejected: u64,
    pop_domain_rate: f64,
    elite_domain_rate: f64,
}

impl Admission {
    /// The admission state of a run under `policy`, or `None` under `Legacy`, which runs the
    /// search unchanged. Panics when the policy cannot be run ([`FaultPolicy::validate`]).
    fn start(policy: FaultPolicy) -> Option<Self> {
        if matches!(policy, FaultPolicy::Legacy) {
            return None;
        }
        if let Err(reason) = policy.validate() {
            panic!("{reason}");
        }
        Some(Self {
            policy,
            faults: HashMap::new(),
            rejected: std::collections::HashSet::new(),
        })
    }

    /// Records one evaluation of `key` at `rung` and returns the scores the search keeps: the
    /// policy's scores, or `None` when the evaluation failed or was rejected. A rejection is
    /// reported to `obs`.
    fn record(
        &mut self,
        key: &str,
        rung: usize,
        admitted: Admitted,
        obs: &dyn Observer,
    ) -> Option<Vec<f64>> {
        match admitted {
            Admitted::Scored(scores, totals) => {
                self.faults.entry(key.to_string()).or_default()[rung] = Some(totals);
                Some(scores)
            }
            Admitted::Failed => None,
            Admitted::Rejected(reason) => {
                obs.on_reject(key, &reason);
                self.rejected.insert(key.to_string());
                None
            }
        }
    }

    /// The fault totals of `key`'s scored evaluation at `rung`; zero when there is none.
    fn totals(&self, key: &str, rung: usize) -> FaultTotals {
        self.faults
            .get(key)
            .and_then(|rungs| rungs[rung])
            .unwrap_or_default()
    }

    /// The faulted cells of each ranked candidate, in order, when the policy breaks ties.
    /// `rung_of` names the rung each candidate's score comes from.
    fn tie_break_cells(
        &self,
        ranked: &[Scored],
        rung_of: impl Fn(&str) -> usize,
    ) -> Option<Vec<u64>> {
        self.policy.breaks_ties().then(|| {
            ranked
                .iter()
                .map(|s| self.totals(&s.dsl, rung_of(&s.dsl)).faulted_cells())
                .collect()
        })
    }

    /// One generation's counters: `population` is the generation's candidates, `ranked` their
    /// ranking and `elites` how many ranked candidates pass into the next generation. `rung_of`
    /// names the rung each ranked candidate's score comes from.
    fn counters(
        &self,
        population: &[DslExpr],
        ranked: &[Scored],
        elites: usize,
        rung_of: impl Fn(&str) -> usize,
    ) -> FaultCounters {
        let mut seen = std::collections::HashSet::new();
        let (mut distinct, mut rejected) = (0_u64, 0_u64);
        for genome in population {
            let key = genome.to_string();
            let was_rejected = self.rejected.contains(&key);
            if seen.insert(key) {
                distinct += 1;
                rejected += u64::from(was_rejected);
            }
        }
        let mean_domain_rate = |candidates: &[Scored]| {
            if candidates.is_empty() {
                return 0.0;
            }
            candidates
                .iter()
                .map(|s| self.totals(&s.dsl, rung_of(&s.dsl)).domain_rate())
                .sum::<f64>()
                / candidates.len() as f64
        };
        FaultCounters {
            hard_rejected_share: if distinct == 0 {
                0.0
            } else {
                rejected as f64 / distinct as f64
            },
            hard_rejected: self.rejected.len() as u64,
            pop_domain_rate: mean_domain_rate(ranked),
            elite_domain_rate: mean_domain_rate(&ranked[..elites.min(ranked.len())]),
        }
    }
}

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

/// Evolve pipeline genomes on (x, y). Returns the deduped population best-first (callers take the
/// top-N finalists for proper CV downstream). Dispatches on `cfg.fitness`: the v1 surrogate path
/// is untouched, bit-identical code; the v2 lite-race is a separate loop (callers that need the
/// per-rung score dump should call `evolve_lite_race` directly).
pub fn evolve(
    x: &Rows,
    y: &[f64],
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    cfg: &EvolveConfig,
    obs: &dyn Observer,
) -> Vec<Scored> {
    match cfg.fitness {
        FitnessMode::Surrogate => evolve_surrogate(x, y, metric, cfg, obs),
        FitnessMode::LiteRace => evolve_lite_race(x, y, metric, cfg, obs).ranked,
    }
}

/// v1 fitness loop: ridge-surrogate scoring of every candidate (see `search_surrogate`).
fn evolve_surrogate(
    x: &Rows,
    y: &[f64],
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    cfg: &EvolveConfig,
    obs: &dyn Observer,
) -> Vec<Scored> {
    // `None` under `FaultPolicy::Legacy`: every step below then runs the legacy code.
    let mut admission = Admission::start(cfg.fault_policy);
    let t0 = Instant::now();
    let mut rng = Pcg32::new(cfg.seed, 0x5EED);
    let assign = fold_assign(y.len(), cfg.folds.max(2), "invent");
    let k = cfg.folds.max(2);
    let gen_vocab = generation_vocab(cfg);
    // B10 lexicase: begin
    if let Err(reason) = cfg.selection.validate(cfg.fault_policy) {
        panic!("{reason}");
    }
    let lexicase = cfg.selection.lexicase();
    let case_rows = match lexicase {
        Some(_) => lexicase_case_rows(y.len(), cfg.seed),
        None => Vec::new(),
    };
    let mut cases: HashMap<String, Vec<f64>> = HashMap::new();
    // B10 lexicase: end
    // B11 ridge scoring: begin
    if let Err(reason) = cfg.ridge_scoring.validate(cfg.fitness, cfg.fault_policy) {
        panic!("{reason}");
    }
    let mut certified = (cfg.ridge_scoring == RidgeScoring::Certified).then(CertifiedScoring::new);
    // B11 ridge scoring: end

    // Seeds: champion + external seeds + starters + random fill.
    let target_pop = cfg.pop.max(1);
    let mut population: Vec<DslExpr> = Vec::with_capacity(target_pop);
    let mut seed_seen = std::collections::HashSet::new();
    for src in std::iter::once(CHAMPION_GENOME)
        .chain(cfg.seed_genomes.iter().map(String::as_str))
        .chain(
            cfg.hard_ten_priors
                .then_some(HARD_TEN_STARTERS.as_slice())
                .into_iter()
                .flatten()
                .copied(),
        )
        .chain(STARTERS.iter().copied())
        .chain(
            cfg.expansion_vocab
                .then_some(EXPANSION_STARTERS.as_slice())
                .into_iter()
                .flatten()
                .copied(),
        )
    {
        if population.len() >= target_pop {
            break;
        }
        if let Ok(GenomeIr::Pipeline(d)) = GenomeIr::parse(src) {
            let canonical = d.to_string();
            if seed_seen.insert(canonical) {
                population.push(d);
            }
        }
    }
    while population.len() < target_pop {
        population.push(rand_genome_from(&mut rng, cfg.max_transforms, &gen_vocab));
    }

    let mut memo: HashMap<String, Option<Vec<f64>>> = HashMap::new();
    let mut ledger = TokenLedger::new();
    let mut evals: u64 = 0;
    let mut best: Option<Scored> = None;
    let mut ranked: Vec<Scored> = Vec::new();
    // Wall-clock throttles for the live streams (decoupled from generations). A fast
    // score-only tick (~16/sec) drives the ticker + score scatter. The code panel is fed
    // ONLY by the per-generation leader (`ranked[0]`), throttled and emitted just when the
    // leader changes — arbitrary candidates are never transpiled. Both are best-effort.
    let mut last_eval_tick = Instant::now();
    let mut last_eval_code = Instant::now();
    let mut last_leader: Option<String> = None;

    for gen in 0..cfg.gens.max(1) {
        if obs.is_cancelled() {
            break;
        }
        if cfg.time_budget_secs > 0.0 && t0.elapsed().as_secs_f64() > cfg.time_budget_secs {
            break;
        }

        // Evaluate all unseen genomes in parallel (the Rust interpreter executes each candidate).
        let unseen: Vec<(String, DslExpr)> = population
            .iter()
            .map(|g| {
                (
                    DslExpr {
                        stages: g.stages.clone(),
                    }
                    .to_string(),
                    g.clone(),
                )
            })
            .filter(|(key, _)| !memo.contains_key(key))
            .collect();
        let fresh: Vec<(String, DslExpr, Option<Vec<f64>>)> = match admission.as_mut() {
            // B11 ridge scoring: the same scores (and cases) from the memo and the stage cache.
            None if certified.is_some() => {
                let Some(scoring) = certified.as_mut() else {
                    unreachable!("certified scoring is set")
                };
                scoring.cache.begin_generation(gen as u64);
                let rows = lexicase.map(|_| case_rows.as_slice());
                let fresh =
                    scoring.score_unseen(unseen, x, y, &assign, k, metric, rows, &mut cases);
                scoring.cache.sweep();
                obs.on_scoring(&scoring.receipt(gen));
                fresh
            }
            // B10 lexicase: the same scores, keeping each candidate's residual cases.
            None if lexicase.is_some() => {
                score_unseen_with_cases(unseen, x, y, &assign, k, metric, &case_rows, &mut cases)
            }
            None => unseen
                .into_par_iter()
                .map(|(key, g)| {
                    let scores = score_genome(&g, x, y, &assign, k, metric);
                    (key, g, scores)
                })
                .collect(),
            Some(admission) => {
                let policy = admission.policy;
                let admitted: Vec<(String, DslExpr, Admitted)> = unseen
                    .into_par_iter()
                    .map(|(key, g)| {
                        let admitted = score_genome_admitted(
                            &g,
                            x,
                            y,
                            &assign,
                            k,
                            metric,
                            policy,
                            &fit_traced,
                        );
                        (key, g, admitted)
                    })
                    .collect();
                admitted
                    .into_iter()
                    .map(|(key, g, admitted)| {
                        let scores = admission.record(&key, 0, admitted, obs);
                        (key, g, scores)
                    })
                    .collect()
            }
        };
        for (key, g, scores) in &fresh {
            evals += 1;
            // Per-generation token accounting uses the STATIC estimator (X6, no
            // emission): every evaluated genome is a shippable dual-language
            // algorithm, so we credit its exact predicted Python+Rust token
            // count without paying to emit it. tokens/sec is therefore live even
            // when no improvement fires. Genomes the emitter cannot support
            // (never generated here) simply credit nothing.
            let est = crate::tokens::try_estimate_tokens(
                &GenomeIr::Pipeline(g.clone()),
                &EmitOptions::default(),
            )
            .map(|e| (e.py + e.rs) as f64)
            .unwrap_or(0.0);
            ledger.add(est);
            memo.insert(key.clone(), scores.clone());

            // Live candidate stream (throttled), SCORE ONLY. Arbitrary candidates are never
            // transpiled — only the per-generation leader is (see the leader-code emit after
            // ranking). This keeps the ticker + score scatter live without spraying code.
            if let Some(fold) = scores {
                if !fold.is_empty() {
                    let cv = fold.iter().sum::<f64>() / fold.len() as f64;
                    if cv.is_finite() {
                        let now = Instant::now();
                        if now.duration_since(last_eval_tick).as_millis() >= EVAL_SCORE_SAMPLE_MS {
                            last_eval_tick = now;
                            obs.on_eval(key, cv, None);
                        }
                    }
                }
            }
        }

        // Rank the current population.
        ranked = {
            let mut seen = std::collections::HashSet::new();
            let mut v: Vec<Scored> = population
                .iter()
                .filter_map(|g| {
                    let key = DslExpr {
                        stages: g.stages.clone(),
                    }
                    .to_string();
                    if !seen.insert(key.clone()) {
                        return None;
                    }
                    let fold_scores = memo.get(&key)?.clone()?;
                    let cv_mean = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
                    Some(Scored {
                        dsl: key,
                        cv_mean,
                        fold_scores,
                    })
                })
                .collect();
            v.sort_by(|a, b| {
                b.cv_mean
                    .partial_cmp(&a.cv_mean)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            v
        };
        if let Some(admission) = &admission {
            if admission.policy.breaks_ties() {
                prefer_fewer_faults(
                    &mut ranked,
                    |s| s.cv_mean,
                    |s| admission.totals(&s.dsl, 0).faulted_cells(),
                );
            }
        }
        if ranked.is_empty() {
            // population collapsed to invalid genomes — reseed randoms and continue
            population = (0..cfg.pop)
                .map(|_| rand_genome_from(&mut rng, cfg.max_transforms, &gen_vocab))
                .collect();
            continue;
        }

        // Stream ONLY the generation leader's code (ephemeral): whenever ranked[0] — the best
        // of this generation — changes, transpile just that one genome and push it to the code
        // panel, throttled so ultra-fast generations can't spray. No other candidate is ever
        // converted to code, and this rides the ephemeral eval channel so it is never persisted.
        {
            // Emit a NEW leader's code immediately (the first emittable one always goes); only
            // rate-limit subsequent leader changes so ultra-fast generations can't stream faster
            // than the panel can read. State advances only on a successful emit, so a leader the
            // transpiler can't render is retried, not silently swallowed.
            let leader_changed = last_leader.as_deref() != Some(ranked[0].dsl.as_str());
            let now = Instant::now();
            let first = last_leader.is_none();
            if leader_changed
                && (first || now.duration_since(last_eval_code).as_millis() >= EVAL_CODE_SAMPLE_MS)
            {
                if let Some(py) = GenomeIr::parse(&ranked[0].dsl)
                    .ok()
                    .and_then(|gm| emit(&gm, &EmitOptions::default()).ok())
                    .map(|e| e.python)
                {
                    last_eval_code = now;
                    last_leader = Some(ranked[0].dsl.clone());
                    obs.on_eval(&ranked[0].dsl, ranked[0].cv_mean, Some(&py));
                }
            }
        }

        let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
        let counters = admission
            .as_ref()
            .map_or_else(FaultCounters::default, |admission| {
                admission.counters(&population, &ranked, (cfg.pop / 8).clamp(1, 8), |_| 0)
            });
        let mut stats = GenStats {
            gen,
            gens: cfg.gens,
            evals,
            evals_per_sec: evals as f64 / elapsed,
            tokens_total: ledger.total(),
            tokens_per_sec: ledger.rate(),
            best_cv: ranked[0].cv_mean,
            pop_fitness: ranked.iter().map(|s| s.cv_mean).collect(),
            hard_rejected_share: counters.hard_rejected_share,
            hard_rejected: counters.hard_rejected,
            pop_domain_rate: counters.pop_domain_rate,
            elite_domain_rate: counters.elite_domain_rate,
        };

        // New best? Emit its FULL dual-language source and stream it.
        let improved = best
            .as_ref()
            .map(|b| ranked[0].cv_mean > b.cv_mean + 1e-9)
            .unwrap_or(true);
        if improved {
            best = Some(ranked[0].clone());
            if let Ok(genome) = GenomeIr::parse(&ranked[0].dsl) {
                if let Ok(emitted) = emit(&genome, &EmitOptions::default()) {
                    // real token credit for the artifact we actually produced
                    ledger.add(
                        (emitted.py_tokens.code_tokens
                            + emitted.py_tokens.doc_tokens
                            + emitted.rs_tokens.code_tokens
                            + emitted.rs_tokens.doc_tokens) as f64,
                    );
                    stats.tokens_total = ledger.total();
                    stats.tokens_per_sec = ledger.rate();
                    obs.on_improvement(&ranked[0], &emitted, &stats);
                }
            }
        }
        obs.on_generation(&stats);

        // Next generation: elites survive, the rest are tournament offspring.
        let elite = (cfg.pop / 8).clamp(1, 8);
        let mut next: Vec<DslExpr> = ranked
            .iter()
            .take(elite)
            .filter_map(|s| match GenomeIr::parse(&s.dsl) {
                Ok(GenomeIr::Pipeline(d)) => Some(d),
                _ => None,
            })
            .collect();
        let parents = ranked.len().min(cfg.pop / 2).max(1);
        let tie_cells = admission
            .as_ref()
            .and_then(|admission| admission.tie_break_cells(&ranked, |_| 0));
        // B10 lexicase: begin
        let matrix =
            lexicase.map(|_| lexicase_matrix(&ranked, |dsl| cases.get(dsl).map(Vec::as_slice)));
        let mut draws = start_lexicase(lexicase, matrix.as_ref(), cfg.seed, gen);
        // B10 lexicase: end
        while next.len() < cfg.pop {
            let pa = pick_parent(&mut rng, &ranked, parents, tie_cells.as_deref(), &mut draws);
            let child = if rng.next_f64() < 0.25 && ranked.len() >= 2 {
                let pb = pick_parent(&mut rng, &ranked, parents, tie_cells.as_deref(), &mut draws);
                let (ga, gb) = (parse_pipe(&ranked[pa].dsl), parse_pipe(&ranked[pb].dsl));
                match (ga, gb) {
                    (Some(a), Some(b)) => crossover(&mut rng, &a, &b, cfg.max_transforms),
                    _ => rand_genome_from(&mut rng, cfg.max_transforms, &gen_vocab),
                }
            } else {
                match parse_pipe(&ranked[pa].dsl) {
                    Some(a) => mutate(&mut rng, &a, cfg.max_transforms),
                    None => rand_genome_from(&mut rng, cfg.max_transforms, &gen_vocab),
                }
            };
            next.push(child);
        }
        report_selection(obs, gen, ranked.len(), draws);
        population = next;
    }

    ranked
}

// ---------------------------------------------------------------------------
// Fitness v2: multi-rung lite-race (successive halving of the TRUE objective)
// ---------------------------------------------------------------------------

/// Per-rung row budgets: rungs 0-1 race on the first 128 rows of a seeded permutation of the
/// search sample, rung 2 on the first 256 — nested prefixes, IDENTICAL for every candidate.
pub const RACE_RUNG_ROWS: [usize; 3] = [128, 128, 256];
/// Per-rung CV fold counts.
pub const RACE_RUNG_FOLDS: [usize; 3] = [2, 2, 3];
/// Cumulative wall-clock share of `time_budget_secs` each rung may consume across the whole run.
pub const RACE_RUNG_TIME_FRACS: [f64; 3] = [0.40, 0.35, 0.25];
/// Fraction of a rung's survivors promoted to the next rung.
pub const RACE_PROMOTE_FRAC: f64 = 0.5;
/// Opportunistic stop: exit the generation loop once a rung-2 score clears target + this margin.
pub const RACE_TARGET_MARGIN: f64 = 0.02;
/// Fold-variance penalty used when `EvolveConfig::rank_robust` is set. These MUST match the
/// downstream gate (`jope-ratchet contracts::robust_score`) or search optimizes a third
/// objective that is neither the old mean nor the thing it is actually judged on.
pub const RACE_ROBUST_STD_WEIGHT: f64 = 0.25;
pub const RACE_ROBUST_STD_CAP: f64 = 1.0;
/// Candidates evaluated between wall-clock checks (mid-generation budget enforcement).
const RACE_CHUNK: usize = 32;

// ---------------------------------------------------------------------------
// Optional rung-0 GPU screening seam (Lane B)
// ---------------------------------------------------------------------------
//
// feat-math never links a GPU crate: the batched-ridge backend arrives as a
// caller-provided closure (prime-jain supplies battle_gpu::gpu_ridge_score_batch).
// None preserves the existing CPU-only behavior.

/// Fresh candidates per rung-0 GPU wave. Each candidate contributes
/// `RACE_RUNG_FOLDS[0]` ridge problems, so a full wave batches about 2,048
/// solves while retaining frequent budget and cancellation checks.
const GPU_SCREEN_CHUNK: usize = 1024;

/// Widest transformed feature matrix routed to the GPU backend. Mirrors
/// `battle_gpu::MAX_RIDGE_P` and `crate::MAX_WIDTH`.
const GPU_SCREEN_MAX_P: usize = crate::MAX_WIDTH;

/// One complete rung-0 ridge fold. Training features are column-major for the
/// fit kernels; validation features are row-major for device prediction.
#[derive(Clone, Debug)]
pub struct GpuRidgeScoreProblem {
    pub train_x_cols: Vec<f32>,
    pub train_y: Vec<f32>,
    pub train_n: usize,
    pub valid_x_rows: Vec<f32>,
    pub valid_y: Vec<f64>,
    pub valid_n: usize,
    pub p: usize,
    pub lambda_frac: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuRidgeScoreTimings {
    pub pack_seconds: f64,
    pub h2d_seconds: f64,
    pub conversion_transfer_seconds: f64,
    pub solve_seconds: f64,
    pub predict_score_seconds: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuRidgeScoreFoldBytes {
    pub logical_h2d_bytes: u64,
    pub physical_h2d_bytes: u64,
    pub logical_d2h_bytes: u64,
    pub physical_d2h_bytes: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuRidgeScoreBatch {
    pub scores: Vec<f64>,
    pub timings: GpuRidgeScoreTimings,
    pub fold_bytes: Vec<GpuRidgeScoreFoldBytes>,
}

/// Order-preserving score backend. It consumes prepared fold buffers so the
/// adapter can move them into the GPU tier without cloning. An error falls
/// the whole wave back to the exact CPU path.
pub type GpuRidgeScoreBatchFn<'backend> =
    dyn Fn(Vec<GpuRidgeScoreProblem>) -> Result<GpuRidgeScoreBatch, String> + Sync + 'backend;

/// Explicit, benchmark-derived eligibility for the fused rung-0 score path.
/// No policy means fail closed to the exact CPU implementation. Callers must
/// not infer a crossover from the older RidgeFit backend: the fused path has
/// its own transfer, prediction, and scoring cost profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuRidgeScorePolicy {
    pub min_p: usize,
    pub min_batch_folds: usize,
}

/// Frozen-replay authority required to opt into exact rung-0 surrogate reuse.
/// Existing entry points never enable the optimization implicitly.
pub const RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256: &str =
    "36807088599bf8c4f0fa30daac57008ea9e0b1ef0aa89894bebddfaeaf7cae97";

/// Explicit capability for exact rung-0 surrogate reuse. Construction fails
/// closed unless the caller presents the independently validated trace receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rung0ExactSurrogateDedupPolicy {
    evaluator_release_sha256: [u8; 32],
}

impl Rung0ExactSurrogateDedupPolicy {
    pub fn from_authority_receipt(
        receipt_sha256: &str,
        evaluator_release_sha256: &str,
    ) -> Option<Self> {
        if receipt_sha256 != RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256
            || evaluator_release_sha256.len() != 64
        {
            return None;
        }
        let mut release = [0_u8; 32];
        for (index, byte) in release.iter_mut().enumerate() {
            let offset = index * 2;
            *byte = u8::from_str_radix(&evaluator_release_sha256[offset..offset + 2], 16).ok()?;
        }
        (release.iter().any(|byte| *byte != 0)).then_some(Self {
            evaluator_release_sha256: release,
        })
    }

    fn evaluator_release_sha256_hex(self) -> String {
        sha256_hex(&self.evaluator_release_sha256)
    }
}

/// Run-level accounting for exact rung-0 reuse. Logical candidate counters are
/// kept separate from physical surrogate evaluations so search telemetry cannot
/// accidentally relabel avoided work as evaluated work.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct Rung0ExactSurrogateDedupReceipt {
    pub requested: bool,
    pub active: bool,
    pub authority_receipt_sha256: Option<String>,
    pub evaluator_release_sha256: Option<String>,
    /// Always `run_local` when requested. The private cache cannot be persisted
    /// or shared across evaluator processes under this contract.
    pub cache_scope: Option<String>,
    /// Opted-in bounded runs may finish more logical candidates in the same
    /// wall clock because avoided physical work is the intended acceleration.
    pub wall_clock_completion_may_accelerate: bool,
    /// Every scattered completed candidate retains the representative's exact
    /// finite fold-score bits; proxy/reduced-precision scores are never cached.
    pub completed_candidate_score_bits_exact: bool,
    pub activation_blocker: Option<String>,
    pub logical_candidates: u64,
    pub physical_candidate_evaluations: u64,
    pub unique_evaluation_signatures: u64,
    pub reused_candidates: u64,
    pub rejected_chunks: u64,
    pub signature_collisions: u64,
    pub partial_mapping_rejections: u64,
    pub representative_failure_rejections: u64,
    pub cancellation_rejections: u64,
    pub admission_lost_rejections: u64,
}

/// Per-candidate reconstruction receipt. `representative_r0_seq` points to the
/// logical candidate whose score bits were physically evaluated.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Rung0ExactSurrogateCandidateReceipt {
    pub evaluation_signature_sha256: String,
    pub representative_r0_seq: u64,
    pub reused: bool,
}

impl GpuRidgeScorePolicy {
    pub fn new(min_p: usize, min_batch_folds: usize) -> Option<Self> {
        (min_p > 0 && min_p <= GPU_SCREEN_MAX_P && min_batch_folds > 0).then_some(Self {
            min_p,
            min_batch_folds,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuScreenPhaseTimings {
    pub prep_seconds: f64,
    pub pack_seconds: f64,
    pub h2d_seconds: f64,
    pub conversion_transfer_seconds: f64,
    pub solve_seconds: f64,
    pub predict_score_seconds: f64,
    pub cpu_score_seconds: f64,
    pub post_seconds: f64,
    pub gpu_folds: u64,
    pub cpu_folds: u64,
}

impl GpuScreenPhaseTimings {
    fn add(&mut self, other: Self) {
        self.prep_seconds += other.prep_seconds;
        self.pack_seconds += other.pack_seconds;
        self.h2d_seconds += other.h2d_seconds;
        self.conversion_transfer_seconds += other.conversion_transfer_seconds;
        self.solve_seconds += other.solve_seconds;
        self.predict_score_seconds += other.predict_score_seconds;
        self.cpu_score_seconds += other.cpu_score_seconds;
        self.post_seconds += other.post_seconds;
        self.gpu_folds += other.gpu_folds;
        self.cpu_folds += other.cpu_folds;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rung0FoldRoute {
    CpuExact,
    CpuPrepared,
    Gpu,
    GpuFallback,
    /// Score bits copied from the run-local exact-surrogate representative.
    /// No CPU fold, GPU fold, or transfer occurred for this logical fold.
    ExactSurrogateReuse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rung0FoldRouteReceipt {
    pub route: Rung0FoldRoute,
    pub logical_h2d_bytes: u64,
    pub physical_h2d_bytes: u64,
    pub logical_d2h_bytes: u64,
    pub physical_d2h_bytes: u64,
}

impl Rung0FoldRouteReceipt {
    fn cpu(route: Rung0FoldRoute) -> Self {
        Self {
            route,
            logical_h2d_bytes: 0,
            physical_h2d_bytes: 0,
            logical_d2h_bytes: 0,
            physical_d2h_bytes: 0,
        }
    }
}

/// Per-genome race state. `rungs[r]` is `Some(outcome)` once the genome was evaluated at rung r;
/// the inner Option is None when it failed there. A score from a lower rung is NEVER served as a
/// higher-rung score: promotion re-scores the genome at the higher rung into its own slot.
#[derive(Clone, Debug, Default)]
struct RaceEntry {
    rungs: [Option<Option<Vec<f64>>>; 3],
    r0_seq: Option<u64>,
    r0_generation: Option<usize>,
    r0_generation_slot: Option<usize>,
    r0_fold_routes: Option<Vec<Rung0FoldRouteReceipt>>,
    r0_exact_surrogate_dedup: Option<Rung0ExactSurrogateCandidateReceipt>,
    /// B10 lexicase: the signed out-of-fold residuals at each scored rung, on the rung-0 rows in
    /// their position order. Only lexicase selection fills them.
    cases: [Option<Vec<f64>>; 3],
}

impl RaceEntry {
    fn mean(&self, rung: usize) -> Option<f64> {
        match &self.rungs[rung] {
            Some(Some(fs)) if !fs.is_empty() => {
                let m = fs.iter().sum::<f64>() / fs.len() as f64;
                m.is_finite().then_some(m)
            }
            _ => None,
        }
    }

    /// Fold-variance-penalized rung score, matching the shape of the downstream gate:
    /// `mean - RACE_ROBUST_STD_WEIGHT * min(population_std(folds), RACE_ROBUST_STD_CAP)`.
    ///
    /// POPULATION std (divide by n), not sample std — the gate uses population std and a
    /// mismatch here would silently shift every comparison. A single-fold rung has zero
    /// variance by definition and so is unpenalized.
    fn robust(&self, rung: usize) -> Option<f64> {
        match &self.rungs[rung] {
            Some(Some(fs)) if !fs.is_empty() => {
                let n = fs.len() as f64;
                let m = fs.iter().sum::<f64>() / n;
                if !m.is_finite() {
                    return None;
                }
                let var = fs.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n;
                let sd = var.sqrt();
                if !sd.is_finite() {
                    return None;
                }
                let penalized = m - RACE_ROBUST_STD_WEIGHT * sd.min(RACE_ROBUST_STD_CAP);
                penalized.is_finite().then_some(penalized)
            }
            _ => None,
        }
    }

    /// Rung score used for ranking and promotion: penalized when `robust` is set.
    fn rank_score(&self, rung: usize, robust: bool) -> Option<f64> {
        if robust {
            self.robust(rung)
        } else {
            self.mean(rung)
        }
    }

    /// Highest rung with a valid score: `(rung, mean, fold_scores)`.
    fn best(&self) -> Option<(usize, f64, &Vec<f64>)> {
        let mut out = None;
        for r in 0..3 {
            if let Some(Some(fs)) = &self.rungs[r] {
                if !fs.is_empty() {
                    let m = fs.iter().sum::<f64>() / fs.len() as f64;
                    if m.is_finite() {
                        out = Some((r, m, fs));
                    }
                }
            }
        }
        out
    }
}

/// Public per-candidate race record (memo dump) for calibration / report plumbing.
#[derive(Clone, Debug, PartialEq)]
pub struct RaceScores {
    /// Canonical DSL text of the genome.
    pub dsl: String,
    /// Per-rung CV fold scores; None = not evaluated at that rung (or failed there).
    pub rung_fold_scores: [Option<Vec<f64>>; 3],
    pub r0_seq: Option<u64>,
    pub r0_generation: Option<usize>,
    pub r0_generation_slot: Option<usize>,
    pub r0_fold_routes: Option<Vec<Rung0FoldRouteReceipt>>,
    pub r0_exact_surrogate_dedup: Option<Rung0ExactSurrogateCandidateReceipt>,
    /// Number of population slots occupied by this canonical genome across all
    /// generations. Repeated semantics are consolidated, never duplicated.
    pub recurrence_count: u64,
    pub first_generation: usize,
    pub last_generation: usize,
}

impl RaceScores {
    pub fn rung_mean(&self, rung: usize) -> Option<f64> {
        let fs = self.rung_fold_scores.get(rung)?.as_ref()?;
        if fs.is_empty() {
            return None;
        }
        let m = fs.iter().sum::<f64>() / fs.len() as f64;
        m.is_finite().then_some(m)
    }
}

#[derive(Clone, Debug)]
pub struct RaceOutcome {
    /// Final population ranked by race fitness (highest-available-rung mean, lower rungs as ties).
    pub ranked: Vec<Scored>,
    /// Every genome ever raced with all its rung scores (sorted by dsl for determinism).
    pub candidates: Vec<RaceScores>,
    /// True when a rung-2 score cleared `target_score + RACE_TARGET_MARGIN` and the loop exited.
    pub target_hit: bool,
    /// Genome evaluations performed per rung (failed evals included).
    pub rung_evals: [u64; 3],
    /// Cumulative wall seconds spent evaluating each rung.
    pub rung_secs: [f64; 3],
    /// Successful GPU-screen phase totals. CPU fallback time remains in rung_secs.
    pub gpu_screen_timings: GpuScreenPhaseTimings,
    /// Exact-surrogate reuse accounting. Default entry points leave this OFF.
    pub rung0_exact_surrogate_dedup: Rung0ExactSurrogateDedupReceipt,
    /// Exact permuted rung-0 source-row order and its aligned fold assignment.
    pub rung0_rows: Vec<usize>,
    pub rung0_assignments: Vec<usize>,
}

/// Post-search replay metadata for one prepared rung-0 fold. This is produced
/// only after evolution has stopped, so trace serialization and enrichment can
/// never add I/O or hashing to the timed candidate path.
#[derive(Clone, Debug, PartialEq)]
pub struct Rung0PreparedTraceFold {
    pub fold_id: usize,
    pub prepared_width: usize,
    pub train_n: usize,
    pub valid_n: usize,
    pub host_f64_bytes: u64,
    pub host_f32_bytes: u64,
    /// Canonical full-model f64 CPU route used by CPU-exact/fallback folds.
    pub cpu_replay_score: f64,
    /// Optimized pre-transformed f64 route used by CPU-prepared folds.
    pub cpu_prepared_replay_score: f64,
}

/// Canonical surrogate identity plus deterministic CPU replay receipts for an
/// already-observed rung-0 candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct Rung0TraceEnrichment {
    pub surrogate_dsl: String,
    pub transform_prefixes: Vec<String>,
    pub folds: Vec<Rung0PreparedTraceFold>,
}

/// Return the exact canonical ridge surrogate and every ordered transform
/// prefix used to create it. The final ridge head is intentionally excluded
/// from `transform_prefixes`.
pub fn rung0_trace_identity(dsl: &str) -> Result<(String, Vec<String>), String> {
    let genome = match GenomeIr::parse(dsl).map_err(|error| error.to_string())? {
        GenomeIr::Pipeline(genome) => genome,
        _ => return Err("rung-0 trace requires a pipeline genome".to_string()),
    };
    let surrogate = search_surrogate(&genome);
    let prefixes = surrogate
        .stages
        .iter()
        .take(surrogate.stages.len().saturating_sub(1))
        .enumerate()
        .map(|(index, _)| DslExpr {
            stages: surrogate.stages[..=index].to_vec(),
        })
        .map(|prefix| prefix.to_string())
        .collect();
    Ok((surrogate.to_string(), prefixes))
}

/// Rebuild the exact rung-0 prepared folds after search and replay them through
/// the f64 CPU authority. Callers retain the originally observed score bits;
/// these replay scores are an independent reconstructibility receipt.
pub fn enrich_rung0_trace_candidate(
    dsl: &str,
    x: &Rows,
    y: &[f64],
    rows: &[usize],
    assignments: &[usize],
) -> Result<Rung0TraceEnrichment, String> {
    if rows.len() != assignments.len() {
        return Err("rung-0 rows/assignments length mismatch".to_string());
    }
    let genome = match GenomeIr::parse(dsl).map_err(|error| error.to_string())? {
        GenomeIr::Pipeline(genome) => genome,
        _ => return Err("rung-0 trace requires a pipeline genome".to_string()),
    };
    let (surrogate_dsl, transform_prefixes) = rung0_trace_identity(dsl)?;
    let surrogate_genome = GenomeIr::Pipeline(search_surrogate(&genome));
    let prepared = prep_rung0_candidate(
        &genome,
        x,
        y,
        rows,
        assignments,
        RACE_RUNG_FOLDS[0],
        &|| false,
    )
    .ok_or_else(|| "rung-0 candidate could not be deterministically prepared".to_string())?;
    let mut cpu_exact_scores = Vec::with_capacity(RACE_RUNG_FOLDS[0]);
    for fold_id in 0..RACE_RUNG_FOLDS[0] {
        let (mut train_x, mut train_y, mut valid_x, mut valid_y) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (position, &row) in rows.iter().enumerate() {
            if assignments[position] == fold_id {
                valid_x.push(x[row].clone());
                valid_y.push(y[row]);
            } else {
                train_x.push(x[row].clone());
                train_y.push(y[row]);
            }
        }
        let fitted = model::fit(&surrogate_genome, &train_x, &train_y)
            .map_err(|error| format!("rung-0 fold {fold_id} canonical CPU fit: {error}"))?;
        let predicted = fitted
            .predict(&valid_x)
            .map_err(|error| format!("rung-0 fold {fold_id} canonical CPU predict: {error}"))?;
        let score = crate::metrics::r2(&valid_y, &predicted);
        if !score.is_finite() {
            return Err(format!(
                "rung-0 fold {fold_id} canonical CPU replay produced no finite score"
            ));
        }
        cpu_exact_scores.push(score);
    }
    let folds = prepared
        .iter()
        .zip(cpu_exact_scores)
        .enumerate()
        .map(|(fold_id, (fold, cpu_replay_score))| {
            let train_n = fold.train_x.len();
            let valid_n = fold.valid_x.len();
            let host_f64_values = train_n
                .saturating_mul(fold.p)
                .saturating_add(train_n)
                .saturating_add(valid_n.saturating_mul(fold.p))
                .saturating_add(valid_n);
            let host_f32_values = train_n
                .saturating_mul(fold.p)
                .saturating_add(train_n)
                .saturating_add(valid_n.saturating_mul(fold.p));
            let cpu_prepared_replay_score = score_prepared_rung0_fold(fold, &crate::metrics::r2)
                .ok_or_else(|| format!("rung-0 fold {fold_id} CPU replay produced no score"))?;
            Ok(Rung0PreparedTraceFold {
                fold_id,
                prepared_width: fold.p,
                train_n,
                valid_n,
                host_f64_bytes: u64::try_from(host_f64_values)
                    .unwrap_or(u64::MAX)
                    .saturating_mul(8),
                host_f32_bytes: u64::try_from(host_f32_values)
                    .unwrap_or(u64::MAX)
                    .saturating_mul(4),
                cpu_replay_score,
                cpu_prepared_replay_score,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Rung0TraceEnrichment {
        surrogate_dsl,
        transform_prefixes,
        folds,
    })
}

/// Deterministic Fisher-Yates permutation of 0..n (same xorshift as the CLI's row sampler).
fn permute_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s = seed | 1;
    for i in (1..idx.len()).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        idx.swap(i, (s as usize) % (i + 1));
    }
    idx
}

/// Score one genome at one rung on the rung's fixed row subset. Rung 0 fits the ridge surrogate
/// (junk filter); rungs 1-2 fit the genome's OWN head at lite budget — a low-fidelity probe of
/// the true objective.
#[allow(clippy::too_many_arguments)]
fn score_rung(
    g: &DslExpr,
    rung: usize,
    x: &Rows,
    y: &[f64],
    rows: &[usize],
    assign: &[usize],
    k: usize,
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    budget_started: Instant,
    overall_budget: f64,
    obs: &dyn Observer,
) -> Option<Vec<f64>> {
    let prepared = if rung == 0 {
        search_surrogate(g)
    } else {
        lite_budget(g)
    };
    let genome = GenomeIr::Pipeline(prepared);
    let mut fold_scores = Vec::with_capacity(k);
    for fold in 0..k {
        if elapsed_budget_exhausted(budget_started.elapsed().as_secs_f64(), overall_budget)
            || obs.is_cancelled()
        {
            return None;
        }
        let (mut xtr, mut ytr, mut xte, mut yte) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (pos, &r) in rows.iter().enumerate() {
            if assign[pos] == fold {
                xte.push(x[r].clone());
                yte.push(y[r]);
            } else {
                xtr.push(x[r].clone());
                ytr.push(y[r]);
            }
        }
        if xte.is_empty() || xtr.is_empty() {
            return None;
        }
        let m = model::fit(&genome, &xtr, &ytr).ok()?;
        if elapsed_budget_exhausted(budget_started.elapsed().as_secs_f64(), overall_budget)
            || obs.is_cancelled()
        {
            return None;
        }
        let pred = m.predict(&xte).ok()?;
        let s = metric(&yte, &pred);
        if !s.is_finite() {
            return None;
        }
        fold_scores.push(s);
    }
    Some(fold_scores)
}

// B10 lexicase: begin
/// [`score_rung`], also returning the signed out-of-fold residual `y[rows[p]] - prediction` of
/// each of the first `cases` positions. The program, folds, budget checks, fits, predictions and
/// metric calls are `score_rung`'s, in the same order, so the fold scores are the same bits.
#[allow(clippy::too_many_arguments)]
fn score_rung_cases(
    g: &DslExpr,
    rung: usize,
    x: &Rows,
    y: &[f64],
    rows: &[usize],
    assign: &[usize],
    k: usize,
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    budget_started: Instant,
    overall_budget: f64,
    obs: &dyn Observer,
    cases: usize,
) -> ScoredCases {
    let prepared = if rung == 0 {
        search_surrogate(g)
    } else {
        lite_budget(g)
    };
    let genome = GenomeIr::Pipeline(prepared);
    let mut fold_scores = Vec::with_capacity(k);
    let mut predictions = vec![f64::NAN; rows.len()];
    for fold in 0..k {
        if elapsed_budget_exhausted(budget_started.elapsed().as_secs_f64(), overall_budget)
            || obs.is_cancelled()
        {
            return None;
        }
        let (mut xtr, mut ytr, mut xte, mut yte) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut held_out = Vec::new();
        for (pos, &r) in rows.iter().enumerate() {
            if assign[pos] == fold {
                xte.push(x[r].clone());
                yte.push(y[r]);
                held_out.push(pos);
            } else {
                xtr.push(x[r].clone());
                ytr.push(y[r]);
            }
        }
        if xte.is_empty() || xtr.is_empty() {
            return None;
        }
        let m = model::fit(&genome, &xtr, &ytr).ok()?;
        if elapsed_budget_exhausted(budget_started.elapsed().as_secs_f64(), overall_budget)
            || obs.is_cancelled()
        {
            return None;
        }
        let pred = m.predict(&xte).ok()?;
        let s = metric(&yte, &pred);
        if !s.is_finite() {
            return None;
        }
        fold_scores.push(s);
        for (&pos, &value) in held_out.iter().zip(&pred) {
            predictions[pos] = value;
        }
    }
    let residuals = rows
        .iter()
        .zip(&predictions)
        .take(cases)
        .map(|(&r, &prediction)| y[r] - prediction)
        .collect();
    Some((fold_scores, residuals))
}
// B10 lexicase: end

/// [`score_rung`] under a non-legacy [`FaultPolicy`]: the same program, rows, folds, metric and
/// budget checks, with the scope check before anything is fitted and every fold run by
/// [`traced_fold`].
#[allow(clippy::too_many_arguments)]
fn score_rung_admitted(
    g: &DslExpr,
    rung: usize,
    x: &Rows,
    y: &[f64],
    rows: &[usize],
    assign: &[usize],
    k: usize,
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    budget_started: Instant,
    overall_budget: f64,
    obs: &dyn Observer,
    policy: FaultPolicy,
    fit: &TracedFit<'_>,
) -> Admitted {
    let prepared = if rung == 0 {
        search_surrogate(g)
    } else {
        lite_budget(g)
    };
    let genome = GenomeIr::Pipeline(prepared);
    if let Some(rejection) = scope_rejection(&genome, x) {
        return Admitted::Rejected(rejection);
    }
    let stop = || {
        elapsed_budget_exhausted(budget_started.elapsed().as_secs_f64(), overall_budget)
            || obs.is_cancelled()
    };
    let mut faults = FaultTotals::default();
    let mut fold_scores = Vec::with_capacity(k);
    for fold in 0..k {
        if stop() {
            return Admitted::Failed;
        }
        let (mut xtr, mut ytr, mut xte, mut yte) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (pos, &r) in rows.iter().enumerate() {
            if assign[pos] == fold {
                xte.push(x[r].clone());
                yte.push(y[r]);
            } else {
                xtr.push(x[r].clone());
                ytr.push(y[r]);
            }
        }
        if xte.is_empty() || xtr.is_empty() {
            return Admitted::Failed;
        }
        match traced_fold(
            &genome,
            (&xtr, &ytr),
            (&xte, &yte),
            metric,
            fit,
            &stop,
            &mut faults,
        ) {
            Ok(score) => fold_scores.push(score),
            Err(end) => return end,
        }
    }
    admit(policy, fold_scores, faults)
}

fn elapsed_budget_exhausted(elapsed_secs: f64, budget_secs: f64) -> bool {
    budget_secs > 0.0 && elapsed_secs >= budget_secs
}

/// Immutable per-run race context (row subsets, fold assignments, budgets).
struct RungCtx<'data, 'backend> {
    x: &'data Rows,
    y: &'data [f64],
    metric: &'data (dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    rows: [Vec<usize>; 3],
    assigns: [Vec<usize>; 3],
    caps: [f64; 3],
    t0: Instant,
    overall_budget: f64,
    /// Optional rung-0 batch backend. Rungs 1-2 remain CPU-exact.
    gpu_screen: Option<&'backend GpuRidgeScoreBatchFn<'backend>>,
    /// Explicit fused-score crossover. None keeps rung 0 entirely on CPU.
    gpu_screen_policy: Option<GpuRidgeScorePolicy>,
    /// Content identity of every fold-relevant rung-0 input. None keeps reuse off.
    rung0_exact_surrogate_scope_sha256: Option<[u8; 32]>,
    /// B10 lexicase: score every rung on the CPU and keep each candidate's residual cases.
    capture_cases: bool,
}

#[derive(Clone, Debug)]
struct CachedRung0ExactSurrogate {
    surrogate_dsl: String,
    scores: Vec<f64>,
    representative_r0_seq: u64,
}

/// Mutable race bookkeeping threaded through the rung evaluators.
struct RaceState {
    memo: HashMap<String, RaceEntry>,
    occurrences: HashMap<String, Occurrence>,
    spent: [f64; 3],
    rung_evals: [u64; 3],
    evals: u64,
    ledger: TokenLedger,
    last_eval_tick: Instant,
    gpu_screen_timings: GpuScreenPhaseTimings,
    rung0_exact_surrogate_cache: HashMap<[u8; 32], CachedRung0ExactSurrogate>,
    rung0_exact_surrogate_dedup: Rung0ExactSurrogateDedupReceipt,
}

#[derive(Clone, Copy, Debug)]
struct Occurrence {
    count: u64,
    first_generation: usize,
    last_generation: usize,
}

fn rung0_fold_scope_sha256(
    x: &Rows,
    y: &[f64],
    rows: &[usize],
    assignments: &[usize],
    folds: usize,
    evaluator_release_sha256: &[u8; 32],
) -> Result<[u8; 32], String> {
    if folds == 0 || rows.len() != assignments.len() {
        return Err("rung-0 exact-surrogate fold scope is incomplete".into());
    }
    let mut hasher = Sha256::new();
    hasher.update(b"FeatMathRung0FoldScope.v1\0");
    hasher.update(evaluator_release_sha256);
    hasher.update(
        u64::try_from(folds)
            .map_err(|_| "fold count overflow")?
            .to_le_bytes(),
    );
    hasher.update(
        u64::try_from(rows.len())
            .map_err(|_| "row count overflow")?
            .to_le_bytes(),
    );
    for (&row, &assignment) in rows.iter().zip(assignments) {
        if assignment >= folds {
            return Err("rung-0 exact-surrogate fold assignment is out of range".into());
        }
        let features = x
            .get(row)
            .ok_or_else(|| "rung-0 exact-surrogate feature row is missing".to_string())?;
        let target = y
            .get(row)
            .ok_or_else(|| "rung-0 exact-surrogate target row is missing".to_string())?;
        hasher.update(
            u64::try_from(row)
                .map_err(|_| "row index overflow")?
                .to_le_bytes(),
        );
        hasher.update(
            u64::try_from(assignment)
                .map_err(|_| "fold assignment overflow")?
                .to_le_bytes(),
        );
        hasher.update(
            u64::try_from(features.len())
                .map_err(|_| "feature width overflow")?
                .to_le_bytes(),
        );
        for value in features {
            hasher.update(value.to_bits().to_le_bytes());
        }
        hasher.update(target.to_bits().to_le_bytes());
    }
    Ok(hasher.finalize().into())
}

fn rung0_evaluation_signature(scope: &[u8; 32], surrogate_dsl: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"FeatMathRung0EvaluationSignature.v1\0");
    hasher.update(scope);
    hasher.update((surrogate_dsl.len() as u64).to_le_bytes());
    hasher.update(surrogate_dsl.as_bytes());
    hasher.finalize().into()
}

fn sha256_hex(value: &[u8; 32]) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rung0DedupMemberSource {
    Cached,
    Representative(usize),
}

#[derive(Debug)]
struct Rung0ExactSurrogatePlan {
    representatives: Vec<usize>,
    member_sources: Vec<Rung0DedupMemberSource>,
    signatures: Vec<[u8; 32]>,
    surrogate_dsls: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rung0ExactSurrogateReject {
    SignatureCollision,
    PartialMapping,
    RepresentativeFailure,
}

fn build_rung0_exact_surrogate_plan(
    chunk: &[(String, DslExpr)],
    scope: &[u8; 32],
    cache: &HashMap<[u8; 32], CachedRung0ExactSurrogate>,
) -> Result<Rung0ExactSurrogatePlan, Rung0ExactSurrogateReject> {
    build_rung0_exact_surrogate_plan_with(chunk, scope, cache, rung0_evaluation_signature)
}

fn build_rung0_exact_surrogate_plan_with(
    chunk: &[(String, DslExpr)],
    scope: &[u8; 32],
    cache: &HashMap<[u8; 32], CachedRung0ExactSurrogate>,
    signature_fn: impl Fn(&[u8; 32], &str) -> [u8; 32],
) -> Result<Rung0ExactSurrogatePlan, Rung0ExactSurrogateReject> {
    let mut representatives = Vec::new();
    let mut member_sources = Vec::with_capacity(chunk.len());
    let mut signatures = Vec::with_capacity(chunk.len());
    let mut surrogate_dsls = Vec::with_capacity(chunk.len());
    let mut local: HashMap<[u8; 32], (String, usize)> = HashMap::new();

    for (candidate_index, (_, genome)) in chunk.iter().enumerate() {
        let surrogate_dsl = search_surrogate(genome).to_string();
        let signature = signature_fn(scope, &surrogate_dsl);
        let source = if let Some(cached) = cache.get(&signature) {
            if cached.surrogate_dsl != surrogate_dsl {
                return Err(Rung0ExactSurrogateReject::SignatureCollision);
            }
            Rung0DedupMemberSource::Cached
        } else if let Some((existing_dsl, representative_slot)) = local.get(&signature) {
            if existing_dsl != &surrogate_dsl {
                return Err(Rung0ExactSurrogateReject::SignatureCollision);
            }
            Rung0DedupMemberSource::Representative(*representative_slot)
        } else {
            let representative_slot = representatives.len();
            representatives.push(candidate_index);
            local.insert(signature, (surrogate_dsl.clone(), representative_slot));
            Rung0DedupMemberSource::Representative(representative_slot)
        };
        member_sources.push(source);
        signatures.push(signature);
        surrogate_dsls.push(surrogate_dsl);
    }

    Ok(Rung0ExactSurrogatePlan {
        representatives,
        member_sources,
        signatures,
        surrogate_dsls,
    })
}

struct RungChunkEvaluation {
    scores: Vec<Option<Vec<f64>>>,
    timings: GpuScreenPhaseTimings,
    routes: Vec<Vec<Rung0FoldRouteReceipt>>,
}

fn evaluate_rung_chunk(
    ctx: &RungCtx<'_, '_>,
    rung: usize,
    chunk: &[(String, DslExpr)],
    gpu: Option<(&GpuRidgeScoreBatchFn<'_>, GpuRidgeScorePolicy)>,
    overall_budget: f64,
    obs: &dyn Observer,
) -> RungChunkEvaluation {
    let rows = &ctx.rows[rung];
    let assign = &ctx.assigns[rung];
    let k = RACE_RUNG_FOLDS[rung];
    let gpu_attempt = gpu.map(|(backend, policy)| {
        eval_rung0_gpu_wave(ctx, chunk, backend, policy, overall_budget, obs)
    });
    match gpu_attempt {
        Some(GpuWaveAttempt {
            scored: Some(scored),
            timings,
            routes,
        }) => RungChunkEvaluation {
            scores: scored.into_iter().map(|(_, _, scores)| scores).collect(),
            timings,
            routes,
        },
        Some(GpuWaveAttempt {
            scored: None,
            mut timings,
            routes,
        }) => {
            let cpu_started = Instant::now();
            let scores = chunk
                .par_iter()
                .map(|(_, genome)| {
                    score_rung(
                        genome,
                        rung,
                        ctx.x,
                        ctx.y,
                        rows,
                        assign,
                        k,
                        ctx.metric,
                        ctx.t0,
                        overall_budget,
                        obs,
                    )
                })
                .collect();
            timings.cpu_score_seconds += cpu_started.elapsed().as_secs_f64();
            timings.gpu_folds = 0;
            timings.cpu_folds = (chunk.len() * k) as u64;
            RungChunkEvaluation {
                scores,
                timings,
                routes,
            }
        }
        None => {
            let cpu_started = Instant::now();
            let scores = chunk
                .par_iter()
                .map(|(_, genome)| {
                    score_rung(
                        genome,
                        rung,
                        ctx.x,
                        ctx.y,
                        rows,
                        assign,
                        k,
                        ctx.metric,
                        ctx.t0,
                        overall_budget,
                        obs,
                    )
                })
                .collect();
            let mut timings = GpuScreenPhaseTimings::default();
            let routes = if rung == 0 {
                timings.cpu_score_seconds = cpu_started.elapsed().as_secs_f64();
                timings.cpu_folds = (chunk.len() * k) as u64;
                vec![vec![Rung0FoldRouteReceipt::cpu(Rung0FoldRoute::CpuExact); k]; chunk.len()]
            } else {
                Vec::new()
            };
            RungChunkEvaluation {
                scores,
                timings,
                routes,
            }
        }
    }
}

struct Rung0ExactSurrogateScatter {
    evaluation: RungChunkEvaluation,
    candidate_receipts: Vec<Option<Rung0ExactSurrogateCandidateReceipt>>,
    cache_entries: Vec<([u8; 32], CachedRung0ExactSurrogate)>,
    reused_candidates: u64,
}

fn unscored_rung0_chunk(
    candidates: usize,
    folds: usize,
    timings: GpuScreenPhaseTimings,
) -> RungChunkEvaluation {
    RungChunkEvaluation {
        scores: vec![None; candidates],
        timings,
        routes: vec![vec![Rung0FoldRouteReceipt::cpu(Rung0FoldRoute::CpuExact); folds]; candidates],
    }
}

fn rung_time_admission_live(
    ctx: &RungCtx<'_, '_>,
    st: &RaceState,
    rung: usize,
    chunk_started: Instant,
) -> bool {
    st.spent[rung] + chunk_started.elapsed().as_secs_f64() <= ctx.caps[rung]
        && !elapsed_budget_exhausted(ctx.t0.elapsed().as_secs_f64(), ctx.overall_budget)
}

fn scatter_rung0_exact_surrogates(
    plan: &Rung0ExactSurrogatePlan,
    representative_evaluation: &RungChunkEvaluation,
    cache: &HashMap<[u8; 32], CachedRung0ExactSurrogate>,
    logical_r0_seq_base: u64,
    folds: usize,
) -> Result<Rung0ExactSurrogateScatter, Rung0ExactSurrogateReject> {
    let members = plan.member_sources.len();
    if plan.signatures.len() != members
        || plan.surrogate_dsls.len() != members
        || representative_evaluation.scores.len() != plan.representatives.len()
        || representative_evaluation.routes.len() != plan.representatives.len()
    {
        return Err(Rung0ExactSurrogateReject::PartialMapping);
    }
    let mut representative_seen = vec![false; plan.representatives.len()];
    for (slot, &member_index) in plan.representatives.iter().enumerate() {
        if member_index >= members
            || representative_seen[slot]
            || plan.member_sources[member_index] != Rung0DedupMemberSource::Representative(slot)
        {
            return Err(Rung0ExactSurrogateReject::PartialMapping);
        }
        representative_seen[slot] = true;
        let scores = representative_evaluation.scores[slot]
            .as_ref()
            .ok_or(Rung0ExactSurrogateReject::RepresentativeFailure)?;
        if scores.len() != folds || scores.iter().any(|score| !score.is_finite()) {
            return Err(Rung0ExactSurrogateReject::RepresentativeFailure);
        }
        if representative_evaluation.routes[slot].len() != folds {
            return Err(Rung0ExactSurrogateReject::PartialMapping);
        }
    }

    let mut cache_entries = Vec::with_capacity(plan.representatives.len());
    for (slot, &member_index) in plan.representatives.iter().enumerate() {
        let representative_r0_seq = logical_r0_seq_base
            .checked_add(member_index as u64)
            .ok_or(Rung0ExactSurrogateReject::PartialMapping)?;
        cache_entries.push((
            plan.signatures[member_index],
            CachedRung0ExactSurrogate {
                surrogate_dsl: plan.surrogate_dsls[member_index].clone(),
                scores: representative_evaluation.scores[slot]
                    .as_ref()
                    .expect("validated representative scores")
                    .clone(),
                representative_r0_seq,
            },
        ));
    }

    let mut scores = Vec::with_capacity(members);
    let mut routes = Vec::with_capacity(members);
    let mut candidate_receipts = Vec::with_capacity(members);
    let mut reused_candidates = 0_u64;
    for member_index in 0..members {
        let signature = plan.signatures[member_index];
        let (member_scores, representative_r0_seq, reused, member_routes) = match plan
            .member_sources[member_index]
        {
            Rung0DedupMemberSource::Cached => {
                let cached = cache
                    .get(&signature)
                    .ok_or(Rung0ExactSurrogateReject::PartialMapping)?;
                if cached.surrogate_dsl != plan.surrogate_dsls[member_index]
                    || cached.scores.len() != folds
                    || cached.scores.iter().any(|score| !score.is_finite())
                {
                    return Err(Rung0ExactSurrogateReject::PartialMapping);
                }
                (
                    cached.scores.clone(),
                    cached.representative_r0_seq,
                    true,
                    vec![Rung0FoldRouteReceipt::cpu(Rung0FoldRoute::ExactSurrogateReuse); folds],
                )
            }
            Rung0DedupMemberSource::Representative(slot) => {
                let &representative_index = plan
                    .representatives
                    .get(slot)
                    .ok_or(Rung0ExactSurrogateReject::PartialMapping)?;
                let representative_r0_seq = logical_r0_seq_base
                    .checked_add(representative_index as u64)
                    .ok_or(Rung0ExactSurrogateReject::PartialMapping)?;
                let reused = member_index != representative_index;
                let member_routes = if reused {
                    vec![Rung0FoldRouteReceipt::cpu(Rung0FoldRoute::ExactSurrogateReuse); folds]
                } else {
                    representative_evaluation.routes[slot].clone()
                };
                (
                    representative_evaluation.scores[slot]
                        .as_ref()
                        .ok_or(Rung0ExactSurrogateReject::RepresentativeFailure)?
                        .clone(),
                    representative_r0_seq,
                    reused,
                    member_routes,
                )
            }
        };
        reused_candidates += u64::from(reused);
        scores.push(Some(member_scores));
        routes.push(member_routes);
        candidate_receipts.push(Some(Rung0ExactSurrogateCandidateReceipt {
            evaluation_signature_sha256: sha256_hex(&signature),
            representative_r0_seq,
            reused,
        }));
    }

    Ok(Rung0ExactSurrogateScatter {
        evaluation: RungChunkEvaluation {
            scores,
            timings: representative_evaluation.timings,
            routes,
        },
        candidate_receipts,
        cache_entries,
        reused_candidates,
    })
}

/// Evaluate `todo` at `rung` in wall-clock-capped chunks. If the rung's cumulative budget (or the
/// overall budget) is exhausted mid-batch, the remaining candidates simply stay at their lower
/// rung — no hang, no partial scores.
#[allow(clippy::unnecessary_unwrap)]
fn eval_rung(
    ctx: &RungCtx<'_, '_>,
    st: &mut RaceState,
    rung: usize,
    generation: usize,
    todo: &[(String, DslExpr)],
    obs: &dyn Observer,
) {
    let k = RACE_RUNG_FOLDS[rung];
    let gpu = if rung == 0 {
        ctx.gpu_screen.zip(ctx.gpu_screen_policy)
    } else {
        None
    };
    let chunk_len = if gpu.is_some() {
        GPU_SCREEN_CHUNK
    } else {
        RACE_CHUNK
    };
    for (chunk_index, chunk) in todo.chunks(chunk_len).enumerate() {
        // Always admit one bounded rung-0 chunk. The run-level clock starts
        // before vocabulary/population construction, so scheduler starvation
        // can otherwise consume a positive budget before any candidate is
        // ranked. Cancellation still wins, and every later chunk remains
        // subject to the configured wall-clock and per-rung caps.
        let bootstrap_chunk = rung == 0 && st.rung_evals[0] == 0 && chunk_index == 0;
        if !bootstrap_chunk && st.spent[rung] > ctx.caps[rung] {
            break;
        }
        if !bootstrap_chunk
            && elapsed_budget_exhausted(ctx.t0.elapsed().as_secs_f64(), ctx.overall_budget)
        {
            break;
        }
        if obs.is_cancelled() {
            break;
        }
        let c0 = Instant::now();
        let chunk_budget = if bootstrap_chunk {
            0.0
        } else {
            ctx.overall_budget
        };
        let logical_r0_seq_base = st.rung_evals[0];
        let mut candidate_receipts = vec![None; chunk.len()];
        let mut evaluation = if rung == 0
            && st.rung0_exact_surrogate_dedup.active
            && ctx.rung0_exact_surrogate_scope_sha256.is_some()
        {
            st.rung0_exact_surrogate_dedup.logical_candidates += chunk.len() as u64;
            let scope = ctx
                .rung0_exact_surrogate_scope_sha256
                .as_ref()
                .expect("active exact-surrogate scope");
            match build_rung0_exact_surrogate_plan(chunk, scope, &st.rung0_exact_surrogate_cache) {
                Ok(plan) => {
                    let representatives = plan
                        .representatives
                        .iter()
                        .map(|&index| chunk[index].clone())
                        .collect::<Vec<_>>();
                    let representative_count = representatives.len() as u64;
                    let representative_evaluation =
                        evaluate_rung_chunk(ctx, rung, &representatives, gpu, chunk_budget, obs);
                    st.rung0_exact_surrogate_dedup
                        .physical_candidate_evaluations += representative_count;
                    let cancelled = obs.is_cancelled();
                    if cancelled
                        || (!bootstrap_chunk && !rung_time_admission_live(ctx, st, rung, c0))
                    {
                        st.rung0_exact_surrogate_dedup.rejected_chunks += 1;
                        if cancelled {
                            st.rung0_exact_surrogate_dedup.cancellation_rejections += 1;
                        } else {
                            st.rung0_exact_surrogate_dedup.admission_lost_rejections += 1;
                        }
                        unscored_rung0_chunk(chunk.len(), k, representative_evaluation.timings)
                    } else {
                        match scatter_rung0_exact_surrogates(
                            &plan,
                            &representative_evaluation,
                            &st.rung0_exact_surrogate_cache,
                            logical_r0_seq_base,
                            k,
                        ) {
                            Ok(scatter) => {
                                for (signature, cached) in scatter.cache_entries {
                                    st.rung0_exact_surrogate_cache.insert(signature, cached);
                                }
                                st.rung0_exact_surrogate_dedup.unique_evaluation_signatures +=
                                    representative_count;
                                st.rung0_exact_surrogate_dedup.reused_candidates +=
                                    scatter.reused_candidates;
                                candidate_receipts = scatter.candidate_receipts;
                                scatter.evaluation
                            }
                            Err(rejection) => {
                                st.rung0_exact_surrogate_dedup.rejected_chunks += 1;
                                match rejection {
                                    Rung0ExactSurrogateReject::SignatureCollision => {
                                        st.rung0_exact_surrogate_dedup.signature_collisions += 1;
                                        st.rung0_exact_surrogate_dedup.active = false;
                                        st.rung0_exact_surrogate_dedup.activation_blocker = Some(
                                            "runtime evaluation-signature collision rejected"
                                                .into(),
                                        );
                                        st.rung0_exact_surrogate_cache.clear();
                                    }
                                    Rung0ExactSurrogateReject::PartialMapping => {
                                        st.rung0_exact_surrogate_dedup
                                            .partial_mapping_rejections += 1;
                                        st.rung0_exact_surrogate_dedup.active = false;
                                        st.rung0_exact_surrogate_dedup.activation_blocker = Some(
                                            "runtime exact-surrogate scatter mapping rejected"
                                                .into(),
                                        );
                                        st.rung0_exact_surrogate_cache.clear();
                                    }
                                    Rung0ExactSurrogateReject::RepresentativeFailure => {
                                        st.rung0_exact_surrogate_dedup
                                            .representative_failure_rejections += 1;
                                    }
                                }
                                let cancelled = obs.is_cancelled();
                                if cancelled
                                    || (!bootstrap_chunk
                                        && !rung_time_admission_live(ctx, st, rung, c0))
                                {
                                    if cancelled {
                                        st.rung0_exact_surrogate_dedup.cancellation_rejections += 1;
                                    } else {
                                        st.rung0_exact_surrogate_dedup.admission_lost_rejections +=
                                            1;
                                    }
                                    unscored_rung0_chunk(
                                        chunk.len(),
                                        k,
                                        representative_evaluation.timings,
                                    )
                                } else {
                                    let mut fallback = evaluate_rung_chunk(
                                        ctx,
                                        rung,
                                        chunk,
                                        gpu,
                                        chunk_budget,
                                        obs,
                                    );
                                    st.rung0_exact_surrogate_dedup
                                        .physical_candidate_evaluations += chunk.len() as u64;
                                    fallback.timings.add(representative_evaluation.timings);
                                    fallback
                                }
                            }
                        }
                    }
                }
                Err(rejection) => {
                    st.rung0_exact_surrogate_dedup.rejected_chunks += 1;
                    match rejection {
                        Rung0ExactSurrogateReject::SignatureCollision => {
                            st.rung0_exact_surrogate_dedup.signature_collisions += 1;
                            st.rung0_exact_surrogate_dedup.active = false;
                            st.rung0_exact_surrogate_dedup.activation_blocker =
                                Some("runtime evaluation-signature collision rejected".into());
                            st.rung0_exact_surrogate_cache.clear();
                        }
                        Rung0ExactSurrogateReject::PartialMapping => {
                            st.rung0_exact_surrogate_dedup.partial_mapping_rejections += 1;
                        }
                        Rung0ExactSurrogateReject::RepresentativeFailure => {
                            st.rung0_exact_surrogate_dedup
                                .representative_failure_rejections += 1;
                        }
                    }
                    let cancelled = obs.is_cancelled();
                    if cancelled
                        || (!bootstrap_chunk && !rung_time_admission_live(ctx, st, rung, c0))
                    {
                        if cancelled {
                            st.rung0_exact_surrogate_dedup.cancellation_rejections += 1;
                        } else {
                            st.rung0_exact_surrogate_dedup.admission_lost_rejections += 1;
                        }
                        unscored_rung0_chunk(chunk.len(), k, GpuScreenPhaseTimings::default())
                    } else {
                        st.rung0_exact_surrogate_dedup
                            .physical_candidate_evaluations += chunk.len() as u64;
                        evaluate_rung_chunk(ctx, rung, chunk, gpu, chunk_budget, obs)
                    }
                }
            }
        } else {
            evaluate_rung_chunk(ctx, rung, chunk, gpu, chunk_budget, obs)
        };
        st.gpu_screen_timings.add(evaluation.timings);
        st.spent[rung] += c0.elapsed().as_secs_f64();
        for (fresh_index, ((key, genome), scores)) in
            chunk.iter().zip(evaluation.scores.drain(..)).enumerate()
        {
            let r0_seq = (rung == 0).then_some(st.rung_evals[0]);
            st.evals += 1;
            st.rung_evals[rung] += 1;
            // Token credit once per genome (rung 0 sees every candidate exactly once).
            if rung == 0 {
                let est = crate::tokens::try_estimate_tokens(
                    &GenomeIr::Pipeline(genome.clone()),
                    &EmitOptions::default(),
                )
                .map(|e| (e.py + e.rs) as f64)
                .unwrap_or(0.0);
                st.ledger.add(est);
            }
            // Live candidate stream (throttled), SCORE ONLY — same contract as v1.
            if let Some(fs) = &scores {
                if !fs.is_empty() {
                    let cv = fs.iter().sum::<f64>() / fs.len() as f64;
                    if cv.is_finite() {
                        let now = Instant::now();
                        if now.duration_since(st.last_eval_tick).as_millis() >= EVAL_SCORE_SAMPLE_MS
                        {
                            st.last_eval_tick = now;
                            obs.on_eval(key, cv, None);
                        }
                    }
                }
            }
            let entry = st.memo.entry(key.clone()).or_default();
            if rung == 0 {
                entry.r0_seq = r0_seq;
                entry.r0_generation = Some(generation);
                entry.r0_generation_slot = Some(chunk_index * chunk_len + fresh_index);
                entry.r0_fold_routes = Some(
                    evaluation
                        .routes
                        .get(fresh_index)
                        .cloned()
                        .unwrap_or_else(|| {
                            vec![Rung0FoldRouteReceipt::cpu(Rung0FoldRoute::CpuExact); k]
                        }),
                );
                entry.r0_exact_surrogate_dedup = candidate_receipts[fresh_index].clone();
            }
            entry.rungs[rung] = Some(scores);
        }
    }
}

/// Why exact rung-0 reuse is off under a non-legacy [`FaultPolicy`], as its receipt records it.
const FAULT_POLICY_REUSE_BLOCKER: &str =
    "exact rung-0 reuse is off under a non-legacy fault policy: a reused score has no fault counts";

/// [`eval_rung`] under a non-legacy [`FaultPolicy`]: the same chunks, budget gates and
/// bookkeeping, with every candidate scored on the CPU by [`score_rung_admitted`], as `eval_rung`
/// scores them when no GPU backend and no exact reuse are configured. The GPU rung-0 wave is
/// never run, because its prepared path fits without the traced entry points, and exact rung-0
/// reuse is never used, because a reused score carries no fault counts of its own.
fn eval_rung_admitted(
    ctx: &RungCtx<'_, '_>,
    st: &mut RaceState,
    admission: &mut Admission,
    rung: usize,
    generation: usize,
    todo: &[(String, DslExpr)],
    obs: &dyn Observer,
) {
    let k = RACE_RUNG_FOLDS[rung];
    let rows = &ctx.rows[rung];
    let assign = &ctx.assigns[rung];
    let policy = admission.policy;
    for (chunk_index, chunk) in todo.chunks(RACE_CHUNK).enumerate() {
        // The chunk gates of `eval_rung`, unchanged.
        let bootstrap_chunk = rung == 0 && st.rung_evals[0] == 0 && chunk_index == 0;
        if !bootstrap_chunk && st.spent[rung] > ctx.caps[rung] {
            break;
        }
        if !bootstrap_chunk
            && elapsed_budget_exhausted(ctx.t0.elapsed().as_secs_f64(), ctx.overall_budget)
        {
            break;
        }
        if obs.is_cancelled() {
            break;
        }
        let c0 = Instant::now();
        let chunk_budget = if bootstrap_chunk {
            0.0
        } else {
            ctx.overall_budget
        };
        let cpu_started = Instant::now();
        let admitted: Vec<Admitted> = chunk
            .par_iter()
            .map(|(_, genome)| {
                score_rung_admitted(
                    genome,
                    rung,
                    ctx.x,
                    ctx.y,
                    rows,
                    assign,
                    k,
                    ctx.metric,
                    ctx.t0,
                    chunk_budget,
                    obs,
                    policy,
                    &fit_traced,
                )
            })
            .collect();
        if rung == 0 {
            st.gpu_screen_timings.add(GpuScreenPhaseTimings {
                cpu_score_seconds: cpu_started.elapsed().as_secs_f64(),
                cpu_folds: (chunk.len() * k) as u64,
                ..GpuScreenPhaseTimings::default()
            });
        }
        st.spent[rung] += c0.elapsed().as_secs_f64();
        for (fresh_index, ((key, genome), admitted)) in chunk.iter().zip(admitted).enumerate() {
            let scores = admission.record(key, rung, admitted, obs);
            // The bookkeeping of `eval_rung`, unchanged.
            let r0_seq = (rung == 0).then_some(st.rung_evals[0]);
            st.evals += 1;
            st.rung_evals[rung] += 1;
            if rung == 0 {
                let est = crate::tokens::try_estimate_tokens(
                    &GenomeIr::Pipeline(genome.clone()),
                    &EmitOptions::default(),
                )
                .map(|e| (e.py + e.rs) as f64)
                .unwrap_or(0.0);
                st.ledger.add(est);
            }
            if let Some(fs) = &scores {
                if !fs.is_empty() {
                    let cv = fs.iter().sum::<f64>() / fs.len() as f64;
                    if cv.is_finite() {
                        let now = Instant::now();
                        if now.duration_since(st.last_eval_tick).as_millis() >= EVAL_SCORE_SAMPLE_MS
                        {
                            st.last_eval_tick = now;
                            obs.on_eval(key, cv, None);
                        }
                    }
                }
            }
            let entry = st.memo.entry(key.clone()).or_default();
            if rung == 0 {
                entry.r0_seq = r0_seq;
                entry.r0_generation = Some(generation);
                entry.r0_generation_slot = Some(chunk_index * RACE_CHUNK + fresh_index);
                entry.r0_fold_routes =
                    Some(vec![
                        Rung0FoldRouteReceipt::cpu(Rung0FoldRoute::CpuExact);
                        k
                    ]);
                entry.r0_exact_surrogate_dedup = None;
            }
            entry.rungs[rung] = Some(scores);
        }
    }
}

// B10 lexicase: begin
/// [`eval_rung`] under lexicase selection: the same chunks, budget gates and bookkeeping, with
/// every candidate scored on the CPU by [`score_rung_cases`], as `eval_rung` scores them when no
/// GPU backend and no exact reuse are configured, and its residual cases kept at the rung. The
/// GPU rung-0 wave and exact rung-0 reuse are never used, because neither returns per-row
/// predictions.
fn eval_rung_cases(
    ctx: &RungCtx<'_, '_>,
    st: &mut RaceState,
    rung: usize,
    generation: usize,
    todo: &[(String, DslExpr)],
    obs: &dyn Observer,
) {
    let k = RACE_RUNG_FOLDS[rung];
    let rows = &ctx.rows[rung];
    let assign = &ctx.assigns[rung];
    let cases = ctx.rows[0].len();
    for (chunk_index, chunk) in todo.chunks(RACE_CHUNK).enumerate() {
        // The chunk gates of `eval_rung`, unchanged.
        let bootstrap_chunk = rung == 0 && st.rung_evals[0] == 0 && chunk_index == 0;
        if !bootstrap_chunk && st.spent[rung] > ctx.caps[rung] {
            break;
        }
        if !bootstrap_chunk
            && elapsed_budget_exhausted(ctx.t0.elapsed().as_secs_f64(), ctx.overall_budget)
        {
            break;
        }
        if obs.is_cancelled() {
            break;
        }
        let c0 = Instant::now();
        let chunk_budget = if bootstrap_chunk {
            0.0
        } else {
            ctx.overall_budget
        };
        let cpu_started = Instant::now();
        let scored: Vec<ScoredCases> = chunk
            .par_iter()
            .map(|(_, genome)| {
                score_rung_cases(
                    genome,
                    rung,
                    ctx.x,
                    ctx.y,
                    rows,
                    assign,
                    k,
                    ctx.metric,
                    ctx.t0,
                    chunk_budget,
                    obs,
                    cases,
                )
            })
            .collect();
        if rung == 0 {
            st.gpu_screen_timings.add(GpuScreenPhaseTimings {
                cpu_score_seconds: cpu_started.elapsed().as_secs_f64(),
                cpu_folds: (chunk.len() * k) as u64,
                ..GpuScreenPhaseTimings::default()
            });
        }
        st.spent[rung] += c0.elapsed().as_secs_f64();
        for (fresh_index, ((key, genome), scored)) in chunk.iter().zip(scored).enumerate() {
            let (scores, residuals) = scored.unzip();
            // The bookkeeping of `eval_rung`, unchanged.
            let r0_seq = (rung == 0).then_some(st.rung_evals[0]);
            st.evals += 1;
            st.rung_evals[rung] += 1;
            if rung == 0 {
                let est = crate::tokens::try_estimate_tokens(
                    &GenomeIr::Pipeline(genome.clone()),
                    &EmitOptions::default(),
                )
                .map(|e| (e.py + e.rs) as f64)
                .unwrap_or(0.0);
                st.ledger.add(est);
            }
            if let Some(fs) = &scores {
                if !fs.is_empty() {
                    let cv = fs.iter().sum::<f64>() / fs.len() as f64;
                    if cv.is_finite() {
                        let now = Instant::now();
                        if now.duration_since(st.last_eval_tick).as_millis() >= EVAL_SCORE_SAMPLE_MS
                        {
                            st.last_eval_tick = now;
                            obs.on_eval(key, cv, None);
                        }
                    }
                }
            }
            let entry = st.memo.entry(key.clone()).or_default();
            if rung == 0 {
                entry.r0_seq = r0_seq;
                entry.r0_generation = Some(generation);
                entry.r0_generation_slot = Some(chunk_index * RACE_CHUNK + fresh_index);
                entry.r0_fold_routes =
                    Some(vec![
                        Rung0FoldRouteReceipt::cpu(Rung0FoldRoute::CpuExact);
                        k
                    ]);
                entry.r0_exact_surrogate_dedup = None;
            }
            entry.rungs[rung] = Some(scores);
            entry.cases[rung] = residuals;
        }
    }
}
// B10 lexicase: end

/// [`eval_rung`], or [`eval_rung_admitted`] when a non-legacy policy runs (`admission` is
/// `Some`).
fn eval_rung_policy(
    ctx: &RungCtx<'_, '_>,
    st: &mut RaceState,
    admission: &mut Option<Admission>,
    rung: usize,
    generation: usize,
    todo: &[(String, DslExpr)],
    obs: &dyn Observer,
) {
    match admission {
        // B10 lexicase: the legacy scores on the CPU, keeping each candidate's residual cases.
        None if ctx.capture_cases => eval_rung_cases(ctx, st, rung, generation, todo, obs),
        None => eval_rung(ctx, st, rung, generation, todo, obs),
        Some(admission) => eval_rung_admitted(ctx, st, admission, rung, generation, todo, obs),
    }
}

/// One CPU-prepared rung-0 fold. Keeping the f64 matrices until the routing
/// decision lets ineligible shapes use the exact CPU ridge implementation and
/// lets a failed GPU wave retry without reconstructing the transforms.
struct PreparedRung0Fold {
    train_x: Rows,
    train_y: Vec<f64>,
    valid_x: Rows,
    valid_y: Vec<f64>,
    p: usize,
    lambda_frac: f64,
    target: Option<FittedTarget>,
}

impl PreparedRung0Fold {
    fn to_gpu_problem(&self) -> GpuRidgeScoreProblem {
        debug_assert!(self.target.is_none());
        let train_n = self.train_x.len();
        let valid_n = self.valid_x.len();
        let mut train_x_cols = Vec::with_capacity(train_n * self.p);
        for column in 0..self.p {
            for row in &self.train_x {
                train_x_cols.push(row[column] as f32);
            }
        }
        let valid_x_rows = self
            .valid_x
            .iter()
            .flat_map(|row| row.iter().map(|value| *value as f32))
            .collect();
        GpuRidgeScoreProblem {
            train_x_cols,
            train_y: self.train_y.iter().map(|value| *value as f32).collect(),
            train_n,
            valid_x_rows,
            valid_y: self.valid_y.clone(),
            valid_n,
            p: self.p,
            lambda_frac: self.lambda_frac,
        }
    }
}

fn score_prepared_rung0_fold(
    fold: &PreparedRung0Fold,
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
) -> Option<f64> {
    let model = RidgeModel::fit(&fold.train_x, &fold.train_y, fold.lambda_frac);
    let mut predictions = model.predict(&fold.valid_x);
    if let Some(target) = &fold.target {
        for prediction in &mut predictions {
            *prediction = target.inverse(*prediction).clamp(-60.0, 60.0);
        }
    }
    let score = metric(&fold.valid_y, &predictions);
    score.is_finite().then_some(score)
}

/// Prepare one candidate for a fused rung-0 wave. Feature transformations use
/// the same f64 CPU implementation as `model::fit`; the policy is applied only
/// after every candidate has an exact CPU-ready representation.
#[allow(clippy::too_many_arguments)]
fn prep_rung0_gpu_candidate(
    g: &DslExpr,
    x: &Rows,
    y: &[f64],
    rows: &[usize],
    assign: &[usize],
    k: usize,
    budget_started: Instant,
    overall_budget: f64,
    obs: &dyn Observer,
) -> Option<Vec<PreparedRung0Fold>> {
    prep_rung0_candidate(g, x, y, rows, assign, k, &|| {
        elapsed_budget_exhausted(budget_started.elapsed().as_secs_f64(), overall_budget)
            || obs.is_cancelled()
    })
}

#[allow(clippy::too_many_arguments)]
fn prep_rung0_candidate(
    g: &DslExpr,
    x: &Rows,
    y: &[f64],
    rows: &[usize],
    assign: &[usize],
    k: usize,
    should_stop: &(dyn Fn() -> bool + Sync),
) -> Option<Vec<PreparedRung0Fold>> {
    let surrogate = search_surrogate(g);
    let stages = &surrogate.stages;
    let (target_stage, rest) = match stages.first() {
        Some(first) if is_target_op(&first.op) => (Some(first), &stages[1..]),
        _ => (None, &stages[..]),
    };
    if rest.iter().any(|stage| is_target_op(&stage.op)) {
        return None;
    }
    let (head, transform_stages) = rest.split_last()?;
    if head.op != "ridge" {
        return None;
    }
    let lambda_frac = head.get_f64("lambda_frac", 0.02);
    let mut problems = Vec::with_capacity(k);
    for fold in 0..k {
        if should_stop() {
            return None;
        }
        let (mut xtr, mut ytr, mut xte, mut yte) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (pos, &row) in rows.iter().enumerate() {
            if assign[pos] == fold {
                xte.push(x[row].clone());
                yte.push(y[row]);
            } else {
                xtr.push(x[row].clone());
                ytr.push(y[row]);
            }
        }
        if xte.is_empty() || xtr.is_empty() {
            return None;
        }

        let target = match target_stage {
            Some(stage) => Some(fit_target(stage, &ytr).ok()?),
            None => None,
        };
        let ytr_eff = match &target {
            Some(fitted) => fitted.forward_all(&ytr),
            None => ytr,
        };

        sanitize(&mut xtr);
        sanitize(&mut xte);
        let mut cur = xtr;
        let mut fitted: Vec<FittedOp> = Vec::with_capacity(transform_stages.len());
        for stage in transform_stages {
            let op = fit_op(stage, &cur, &ytr_eff).ok()?;
            // TRAIN pass: must use the train variant so leave-one-out-style
            // ops see their training encoding here, exactly as model::fit does.
            cur = apply_op_train(&op, &cur).ok()?;
            fitted.push(op);
        }
        let mut te = xte;
        for op in &fitted {
            te = apply_op(op, &te).ok()?;
        }

        let n = cur.len();
        let p = cur.first().map(Vec::len).unwrap_or(0);
        if n == 0 || p == 0 || p > GPU_SCREEN_MAX_P {
            return None;
        }
        if te.len() != yte.len() || te.iter().any(|row| row.len() != p) {
            return None;
        }
        problems.push(PreparedRung0Fold {
            train_x: cur,
            train_y: ytr_eff,
            valid_x: te,
            valid_y: yte,
            p,
            lambda_frac,
            target,
        });
    }
    Some(problems)
}

/// Evaluate one rung-0 wave through an order-preserving batch backend. A
/// backend error or length mismatch returns `None`, causing the caller to
/// retry the entire wave on the existing exact CPU path. Candidates that
/// cannot be prepared for the backend are CPU-scored inside a successful
/// wave.
/// One scored candidate from a wave: its genome text, its parsed expression, and the
/// per-fold scores, which are `None` when that candidate fell back to the CPU path.
type ScoredCandidate<'chunk> = (&'chunk String, &'chunk DslExpr, Option<Vec<f64>>);

struct GpuWaveAttempt<'chunk> {
    scored: Option<Vec<ScoredCandidate<'chunk>>>,
    timings: GpuScreenPhaseTimings,
    routes: Vec<Vec<Rung0FoldRouteReceipt>>,
}

fn eval_rung0_gpu_wave<'chunk>(
    ctx: &RungCtx<'_, '_>,
    chunk: &'chunk [(String, DslExpr)],
    backend: &GpuRidgeScoreBatchFn<'_>,
    policy: GpuRidgeScorePolicy,
    overall_budget: f64,
    obs: &dyn Observer,
) -> GpuWaveAttempt<'chunk> {
    let rows = &ctx.rows[0];
    let assign = &ctx.assigns[0];
    let k = RACE_RUNG_FOLDS[0];
    let prep_started = Instant::now();
    let preps: Vec<Option<Vec<PreparedRung0Fold>>> = chunk
        .par_iter()
        .map(|(_, genome)| {
            prep_rung0_gpu_candidate(
                genome,
                ctx.x,
                ctx.y,
                rows,
                assign,
                k,
                ctx.t0,
                overall_budget,
                obs,
            )
        })
        .collect();
    let mut timings = GpuScreenPhaseTimings {
        prep_seconds: prep_started.elapsed().as_secs_f64(),
        ..GpuScreenPhaseTimings::default()
    };

    let conversion_started = Instant::now();
    let eligible_folds = preps
        .iter()
        .flatten()
        .flatten()
        .filter(|fold| fold.target.is_none() && fold.p >= policy.min_p)
        .count();
    let gpu_enabled = eligible_folds >= policy.min_batch_folds;
    let gpu_selected = preps
        .iter()
        .map(|prep| {
            prep.as_ref()
                .map(|folds| {
                    folds
                        .iter()
                        .map(|fold| gpu_enabled && fold.target.is_none() && fold.p >= policy.min_p)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec![false; k])
        })
        .collect::<Vec<_>>();
    let mut routes = preps
        .iter()
        .enumerate()
        .map(|(candidate_index, prep)| match prep {
            Some(folds) => folds
                .iter()
                .enumerate()
                .map(|(fold_index, _)| {
                    if gpu_selected[candidate_index][fold_index] {
                        Rung0FoldRouteReceipt::cpu(Rung0FoldRoute::Gpu)
                    } else {
                        Rung0FoldRouteReceipt::cpu(Rung0FoldRoute::CpuPrepared)
                    }
                })
                .collect(),
            None => vec![Rung0FoldRouteReceipt::cpu(Rung0FoldRoute::CpuExact); k],
        })
        .collect::<Vec<_>>();
    let mut gpu_slots = Vec::new();
    let mut problems = Vec::new();
    for (candidate_index, prep) in preps.iter().enumerate() {
        let Some(folds) = prep else { continue };
        for (fold_index, fold) in folds.iter().enumerate() {
            if gpu_selected[candidate_index][fold_index] {
                gpu_slots.push((candidate_index, fold_index));
                problems.push(fold.to_gpu_problem());
            }
        }
    }
    let problem_count = problems.len();
    timings.conversion_transfer_seconds += conversion_started.elapsed().as_secs_f64();

    enum CpuCandidateScores {
        Exact(Option<Vec<f64>>),
        Prepared(Vec<Option<f64>>),
    }

    let (batch_result, (cpu_scores, cpu_seconds, cpu_folds)) = rayon::join(
        || {
            if problems.is_empty() {
                Ok(GpuRidgeScoreBatch::default())
            } else {
                backend(problems)
            }
        },
        || {
            let cpu_started = Instant::now();
            let cpu_scores = chunk
                .par_iter()
                .zip(&preps)
                .enumerate()
                .map(|(candidate_index, ((_, genome), prep))| match prep {
                    Some(folds) => CpuCandidateScores::Prepared(
                        folds
                            .iter()
                            .enumerate()
                            .map(|(fold_index, fold)| {
                                (!gpu_selected[candidate_index][fold_index])
                                    .then(|| score_prepared_rung0_fold(fold, ctx.metric))
                                    .flatten()
                            })
                            .collect(),
                    ),
                    None => CpuCandidateScores::Exact(score_rung(
                        genome,
                        0,
                        ctx.x,
                        ctx.y,
                        rows,
                        assign,
                        k,
                        ctx.metric,
                        ctx.t0,
                        overall_budget,
                        obs,
                    )),
                })
                .collect::<Vec<_>>();
            let cpu_folds = preps
                .iter()
                .enumerate()
                .map(|(candidate_index, prep)| match prep {
                    Some(folds) => folds
                        .iter()
                        .enumerate()
                        .filter(|(fold_index, _)| !gpu_selected[candidate_index][*fold_index])
                        .count(),
                    None => k,
                })
                .sum::<usize>();
            (cpu_scores, cpu_started.elapsed().as_secs_f64(), cpu_folds)
        },
    );
    timings.cpu_score_seconds += cpu_seconds;
    timings.cpu_folds += cpu_folds as u64;
    let batch = match batch_result {
        Ok(batch)
            if batch.scores.len() == problem_count && batch.fold_bytes.len() == problem_count =>
        {
            batch
        }
        _ => {
            for candidate_routes in &mut routes {
                for route in candidate_routes {
                    route.route = if route.route == Rung0FoldRoute::Gpu {
                        Rung0FoldRoute::GpuFallback
                    } else {
                        Rung0FoldRoute::CpuExact
                    };
                }
            }
            return GpuWaveAttempt {
                scored: None,
                timings,
                routes,
            };
        }
    };
    timings.pack_seconds += batch.timings.pack_seconds;
    timings.h2d_seconds += batch.timings.h2d_seconds;
    timings.conversion_transfer_seconds += batch.timings.conversion_transfer_seconds;
    timings.solve_seconds += batch.timings.solve_seconds;
    timings.predict_score_seconds += batch.timings.predict_score_seconds;
    timings.gpu_folds += problem_count as u64;

    let post_started = Instant::now();
    let mut gpu_scores = vec![vec![None; k]; chunk.len()];
    for ((candidate_index, fold_index), (score, bytes)) in gpu_slots
        .into_iter()
        .zip(batch.scores.into_iter().zip(batch.fold_bytes))
    {
        gpu_scores[candidate_index][fold_index] = score.is_finite().then_some(score);
        routes[candidate_index][fold_index] = Rung0FoldRouteReceipt {
            route: Rung0FoldRoute::Gpu,
            logical_h2d_bytes: bytes.logical_h2d_bytes,
            physical_h2d_bytes: bytes.physical_h2d_bytes,
            logical_d2h_bytes: bytes.logical_d2h_bytes,
            physical_d2h_bytes: bytes.physical_d2h_bytes,
        };
    }
    let mut out = Vec::with_capacity(chunk.len());
    for (candidate_index, ((key, genome), cpu_scores)) in chunk.iter().zip(cpu_scores).enumerate() {
        let scores = match cpu_scores {
            CpuCandidateScores::Exact(scores) => scores,
            CpuCandidateScores::Prepared(cpu_folds) => {
                let fold_scores = cpu_folds
                    .into_iter()
                    .zip(&gpu_scores[candidate_index])
                    .map(|(cpu_score, gpu_score)| cpu_score.or(*gpu_score))
                    .collect::<Option<Vec<_>>>();
                fold_scores.filter(|scores| scores.len() == k)
            }
        };
        out.push((key, genome, scores));
    }
    timings.post_seconds += post_started.elapsed().as_secs_f64();
    GpuWaveAttempt {
        scored: Some(out),
        timings,
        routes,
    }
}

/// Successive-halving promotion: among `pool` (in scoring order), keep the top
/// `RACE_PROMOTE_FRAC` by `from`-rung mean that have not been scored at `to` yet.
fn promote(
    st: &RaceState,
    pool: &[(String, DslExpr)],
    from: usize,
    to: usize,
    rank_robust: bool,
) -> Vec<(String, DslExpr)> {
    // Rung 0 is a head-blind ridge-surrogate junk filter over 2 folds; its fold variance is
    // noise about a proxy, not evidence about the genome, so it is never penalized. Penalize
    // from rung 1 onward, where the candidate's own head is actually being fit.
    let robust = rank_robust && from >= 1;
    let mut scored: Vec<(f64, &(String, DslExpr))> = pool
        .iter()
        .filter_map(|kg| {
            st.memo
                .get(&kg.0)
                .and_then(|e| e.rank_score(from, robust))
                .map(|m| (m, kg))
        })
        .collect();
    // Stable sort: ties keep pool order, so promotion is deterministic.
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let keep = ((scored.len() as f64) * RACE_PROMOTE_FRAC).ceil() as usize;
    scored
        .into_iter()
        .take(keep)
        .filter(|(_, (key, _))| {
            st.memo
                .get(key)
                .map(|e| e.rungs[to].is_none())
                .unwrap_or(true)
        })
        .map(|(_, kg)| kg.clone())
        .collect()
}

/// A ranked candidate with its per-rung means (NEG_INFINITY placeholder = unscored rung).
struct RankRec {
    scored: Scored,
    means: [f64; 3],
}

/// Rank the population by highest-available-rung mean score; ties fall back to lower-rung scores
/// (a rung-2-validated candidate therefore outranks an equal-scored lower-rung one).
fn rank_race(
    population: &[DslExpr],
    memo: &HashMap<String, RaceEntry>,
    rank_robust: bool,
) -> Vec<RankRec> {
    let mut seen = std::collections::HashSet::new();
    let mut v: Vec<RankRec> = population
        .iter()
        .filter_map(|g| {
            let key = DslExpr {
                stages: g.stages.clone(),
            }
            .to_string();
            if !seen.insert(key.clone()) {
                return None;
            }
            let entry = memo.get(&key)?;
            let (best_rung, cv_mean, fold_scores) = entry.best()?;
            // `Scored.cv_mean` is the ranking key AND the value reported downstream. When the
            // robust key is on, rank on the penalized score at the highest scored rung so the
            // ordering matches the gate. Rung 0 stays unpenalized (head-blind surrogate).
            let rank_key = entry
                .rank_score(best_rung, rank_robust && best_rung >= 1)
                .unwrap_or(cv_mean);
            let mut means = [f64::NEG_INFINITY; 3];
            for (r, slot) in means.iter_mut().enumerate() {
                if let Some(m) = entry.rank_score(r, rank_robust && r >= 1) {
                    *slot = m;
                }
            }
            Some(RankRec {
                scored: Scored {
                    dsl: key,
                    cv_mean: rank_key,
                    fold_scores: fold_scores.clone(),
                },
                means,
            })
        })
        .collect();
    v.sort_by(|a, b| {
        b.scored
            .cv_mean
            .partial_cmp(&a.scored.cv_mean)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.means[2]
                    .partial_cmp(&a.means[2])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                b.means[1]
                    .partial_cmp(&a.means[1])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                b.means[0]
                    .partial_cmp(&a.means[0])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    v
}

/// The rung a raced program's ranking score comes from: its highest rung with a valid score.
fn best_rung(memo: &HashMap<String, RaceEntry>, key: &str) -> usize {
    memo.get(key)
        .and_then(RaceEntry::best)
        .map_or(0, |(rung, _, _)| rung)
}

/// [`rank_race`] under a non-legacy [`FaultPolicy`]: the programs the policy rejected at any rung
/// are left out, and a tie-breaking `Penalize` puts the fewer faulted cells first among near-tied
/// scores, each program's cells taken at the rung its score comes from. Otherwise the order is
/// `rank_race`'s: among near-ties with equal counts, its rung-mean fallbacks still decide.
fn rank_race_admitted(
    population: &[DslExpr],
    memo: &HashMap<String, RaceEntry>,
    rank_robust: bool,
    admission: &Admission,
) -> Vec<RankRec> {
    let mut recs = rank_race(population, memo, rank_robust);
    recs.retain(|rec| !admission.rejected.contains(&rec.scored.dsl));
    if admission.policy.breaks_ties() {
        prefer_fewer_faults(
            &mut recs,
            |rec| rec.scored.cv_mean,
            |rec| {
                admission
                    .totals(&rec.scored.dsl, best_rung(memo, &rec.scored.dsl))
                    .faulted_cells()
            },
        );
    }
    recs
}

/// [`rank_race`], or [`rank_race_admitted`] when a non-legacy policy runs.
fn rank_race_policy(
    population: &[DslExpr],
    memo: &HashMap<String, RaceEntry>,
    rank_robust: bool,
    admission: &Option<Admission>,
) -> Vec<RankRec> {
    match admission {
        None => rank_race(population, memo, rank_robust),
        Some(admission) => rank_race_admitted(population, memo, rank_robust, admission),
    }
}

/// Fitness v2: 3-rung successive-halving race of the true objective.
///
/// The v1 surrogate is head-blind (measured: 98.7% of runs end with all finalists proxy-tied), so
/// selection regret concentrates exactly where the search must discriminate. The race keeps the
/// surrogate as a cheap junk filter and spends the saved budget probing each promoted genome's
/// OWN head at tiny budget:
///   rung 0: ridge surrogate, 128 rows / 2 folds — every new candidate
///   rung 1: lite own-head,  128 rows / 2 folds — top 50% of rung 0 (this generation's new)
///   rung 2: lite own-head,  256 rows / 3 folds — top 50% of rung 1 (~25% of new)
/// Elites are re-validated at rung 2 before selection. Per-rung cumulative wall caps
/// (`RACE_RUNG_TIME_FRACS` of `time_budget_secs`) keep any one rung from starving the others;
/// when a cap is hit mid-generation the unscored candidates simply stay at their lower rung.
pub fn evolve_lite_race(
    x: &Rows,
    y: &[f64],
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    cfg: &EvolveConfig,
    obs: &dyn Observer,
) -> RaceOutcome {
    evolve_lite_race_with_gpu(x, y, metric, cfg, obs, None)
}

/// Run the lite race from exactly the caller-provided generation 0.
///
/// `initial_population` must contain exactly `cfg.pop` valid, canonically unique pipeline genomes.
/// Unlike the legacy entry points, this path adds no vendored champion, configured seed, curated
/// starter, or grammar-random genome to generation 0. Later generations retain the established
/// elite/tournament/mutation/crossover behavior.
pub fn evolve_lite_race_with_exact_initial_population(
    x: &Rows,
    y: &[f64],
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    cfg: &EvolveConfig,
    obs: &dyn Observer,
    initial_population: &[String],
) -> Result<RaceOutcome, String> {
    cfg.fault_policy.validate()?;
    // B10 lexicase: the selection's refusal is returned here, as the fault policy's is.
    cfg.selection.validate(cfg.fault_policy)?;
    // B11 ridge scoring: certified scoring applies to the surrogate search only.
    cfg.ridge_scoring
        .validate(FitnessMode::LiteRace, cfg.fault_policy)?;
    if cfg.pop == 0 {
        return Err("exact initial population requires cfg.pop greater than zero".to_string());
    }
    if initial_population.len() != cfg.pop {
        return Err(format!(
            "exact initial population has {} genomes; cfg.pop requires {}",
            initial_population.len(),
            cfg.pop
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(cfg.pop);
    let mut parsed = Vec::with_capacity(cfg.pop);
    for (index, source) in initial_population.iter().enumerate() {
        let genome = parse_canonical_pipeline(source)
            .map_err(|error| format!("initial pool entry {index} is invalid: {error}"))?;
        let canonical = genome.to_string();
        if !seen.insert(canonical) {
            return Err(format!(
                "initial pool entry {index} duplicates a canonical pipeline"
            ));
        }
        parsed.push(genome);
    }
    Ok(evolve_lite_race_impl(
        x,
        y,
        metric,
        cfg,
        obs,
        None,
        None,
        None,
        Some(parsed),
    ))
}

/// Run the lite race with an optional rung-0 GPU batch backend. The backend
/// lifetime is explicit so callers may borrow task-local telemetry state.
/// Backend failures fall each affected wave back to the exact CPU path; all
/// later rungs and honest finalist evaluation remain CPU-exact.
pub fn evolve_lite_race_with_gpu<'backend>(
    x: &Rows,
    y: &[f64],
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    cfg: &EvolveConfig,
    obs: &dyn Observer,
    gpu_screen: Option<&'backend GpuRidgeScoreBatchFn<'backend>>,
) -> RaceOutcome {
    evolve_lite_race_with_gpu_policy(x, y, metric, cfg, obs, gpu_screen, None)
}

/// Run the lite race with an explicitly benchmarked fused-score eligibility
/// policy. This is the only entry point that may route rung-0 folds to the
/// backend; the compatibility wrapper above intentionally fails closed.
pub fn evolve_lite_race_with_gpu_policy<'backend>(
    x: &Rows,
    y: &[f64],
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    cfg: &EvolveConfig,
    obs: &dyn Observer,
    gpu_screen: Option<&'backend GpuRidgeScoreBatchFn<'backend>>,
    gpu_screen_policy: Option<GpuRidgeScorePolicy>,
) -> RaceOutcome {
    evolve_lite_race_with_gpu_policy_and_exact_surrogate_dedup(
        x,
        y,
        metric,
        cfg,
        obs,
        gpu_screen,
        gpu_screen_policy,
        None,
    )
}

/// Explicit opt-in seam for exact rung-0 surrogate reuse. The compatibility
/// wrappers above always pass `None`, so release/runtime wiring remains OFF.
///
/// Bounded runs intentionally may complete more logical candidates in the same
/// wall clock. Admission still occurs only at the existing chunk boundary, and
/// every completed logical candidate receives bit-exact representative folds.
#[allow(clippy::too_many_arguments, clippy::unnecessary_unwrap)]
pub fn evolve_lite_race_with_gpu_policy_and_exact_surrogate_dedup<'backend>(
    x: &Rows,
    y: &[f64],
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    cfg: &EvolveConfig,
    obs: &dyn Observer,
    gpu_screen: Option<&'backend GpuRidgeScoreBatchFn<'backend>>,
    gpu_screen_policy: Option<GpuRidgeScorePolicy>,
    exact_surrogate_dedup: Option<Rung0ExactSurrogateDedupPolicy>,
) -> RaceOutcome {
    evolve_lite_race_impl(
        x,
        y,
        metric,
        cfg,
        obs,
        gpu_screen,
        gpu_screen_policy,
        exact_surrogate_dedup,
        None,
    )
}

#[allow(clippy::too_many_arguments, clippy::unnecessary_unwrap)]
fn evolve_lite_race_impl<'backend>(
    x: &Rows,
    y: &[f64],
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    cfg: &EvolveConfig,
    obs: &dyn Observer,
    gpu_screen: Option<&'backend GpuRidgeScoreBatchFn<'backend>>,
    gpu_screen_policy: Option<GpuRidgeScorePolicy>,
    exact_surrogate_dedup: Option<Rung0ExactSurrogateDedupPolicy>,
    exact_initial_population: Option<Vec<DslExpr>>,
) -> RaceOutcome {
    // `None` under `FaultPolicy::Legacy`: every step below then runs the legacy code.
    let mut admission = Admission::start(cfg.fault_policy);
    // B10 lexicase: begin
    if let Err(reason) = cfg.selection.validate(cfg.fault_policy) {
        panic!("{reason}");
    }
    let lexicase = cfg.selection.lexicase();
    // B10 lexicase: end
    // B11 ridge scoring: certified scoring applies to the surrogate search only.
    if let Err(reason) = cfg
        .ridge_scoring
        .validate(FitnessMode::LiteRace, cfg.fault_policy)
    {
        panic!("{reason}");
    }
    let t0 = Instant::now();
    let mut rng = Pcg32::new(cfg.seed, 0x5EED);
    let n = x.len().min(y.len());
    let perm = permute_indices(n, cfg.seed);
    let rows: [Vec<usize>; 3] = [
        perm[..n.min(RACE_RUNG_ROWS[0])].to_vec(),
        perm[..n.min(RACE_RUNG_ROWS[1])].to_vec(),
        perm[..n.min(RACE_RUNG_ROWS[2])].to_vec(),
    ];
    let assigns: [Vec<usize>; 3] = [
        fold_assign(rows[0].len(), RACE_RUNG_FOLDS[0], "invent_r0"),
        fold_assign(rows[1].len(), RACE_RUNG_FOLDS[1], "invent_r1"),
        fold_assign(rows[2].len(), RACE_RUNG_FOLDS[2], "invent_r2"),
    ];
    let caps: [f64; 3] = if cfg.time_budget_secs > 0.0 {
        [
            cfg.time_budget_secs * RACE_RUNG_TIME_FRACS[0],
            cfg.time_budget_secs * RACE_RUNG_TIME_FRACS[1],
            cfg.time_budget_secs * RACE_RUNG_TIME_FRACS[2],
        ]
    } else {
        [f64::INFINITY; 3]
    };
    let (rung0_exact_surrogate_scope_sha256, activation_blocker) =
        if exact_surrogate_dedup.is_none() {
            (None, None)
        } else {
            let policy = exact_surrogate_dedup.expect("checked exact-surrogate policy");
            match rung0_fold_scope_sha256(
                x,
                y,
                &rows[0],
                &assigns[0],
                RACE_RUNG_FOLDS[0],
                &policy.evaluator_release_sha256,
            ) {
                Ok(scope) => (Some(scope), None),
                Err(error) => (None, Some(error)),
            }
        };
    let ctx = RungCtx {
        x,
        y,
        metric,
        rows,
        assigns,
        caps,
        t0,
        overall_budget: cfg.time_budget_secs,
        gpu_screen,
        gpu_screen_policy,
        rung0_exact_surrogate_scope_sha256,
        capture_cases: lexicase.is_some(),
    };

    let gen_vocab = generation_vocab(cfg);
    let mut population = if let Some(population) = exact_initial_population {
        population
    } else {
        // Seeds: champion + external seeds + starters + random fill (same recipe as v1).
        let target_pop = cfg.pop.max(1);
        let mut population: Vec<DslExpr> = Vec::with_capacity(target_pop);
        let mut seed_seen = std::collections::HashSet::new();
        for src in std::iter::once(CHAMPION_GENOME)
            .chain(cfg.seed_genomes.iter().map(String::as_str))
            .chain(
                cfg.hard_ten_priors
                    .then_some(HARD_TEN_STARTERS.as_slice())
                    .into_iter()
                    .flatten()
                    .copied(),
            )
            .chain(STARTERS.iter().copied())
            .chain(
                cfg.expansion_vocab
                    .then_some(EXPANSION_STARTERS.as_slice())
                    .into_iter()
                    .flatten()
                    .copied(),
            )
        {
            if population.len() >= target_pop {
                break;
            }
            if let Ok(GenomeIr::Pipeline(d)) = GenomeIr::parse(src) {
                let canonical = d.to_string();
                if seed_seen.insert(canonical) {
                    population.push(d);
                }
            }
        }
        while population.len() < target_pop {
            population.push(rand_genome_from(&mut rng, cfg.max_transforms, &gen_vocab));
        }
        population
    };

    let mut st = RaceState {
        memo: HashMap::new(),
        occurrences: HashMap::new(),
        spent: [0.0; 3],
        rung_evals: [0; 3],
        evals: 0,
        ledger: TokenLedger::new(),
        last_eval_tick: Instant::now(),
        gpu_screen_timings: GpuScreenPhaseTimings::default(),
        rung0_exact_surrogate_cache: HashMap::new(),
        rung0_exact_surrogate_dedup: Rung0ExactSurrogateDedupReceipt {
            requested: exact_surrogate_dedup.is_some(),
            active: rung0_exact_surrogate_scope_sha256.is_some(),
            authority_receipt_sha256: exact_surrogate_dedup
                .map(|_| RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256.to_string()),
            evaluator_release_sha256: exact_surrogate_dedup
                .map(Rung0ExactSurrogateDedupPolicy::evaluator_release_sha256_hex),
            cache_scope: exact_surrogate_dedup.map(|_| "run_local".to_string()),
            wall_clock_completion_may_accelerate: exact_surrogate_dedup.is_some(),
            completed_candidate_score_bits_exact: exact_surrogate_dedup.is_some(),
            activation_blocker,
            ..Rung0ExactSurrogateDedupReceipt::default()
        },
    };
    if admission.is_some() && st.rung0_exact_surrogate_dedup.active {
        st.rung0_exact_surrogate_dedup.active = false;
        st.rung0_exact_surrogate_dedup.activation_blocker =
            Some(FAULT_POLICY_REUSE_BLOCKER.to_string());
    }
    // B10 lexicase: begin
    if ctx.capture_cases && st.rung0_exact_surrogate_dedup.active {
        st.rung0_exact_surrogate_dedup.active = false;
        st.rung0_exact_surrogate_dedup.activation_blocker =
            Some(LEXICASE_REUSE_BLOCKER.to_string());
    }
    // B10 lexicase: end
    let mut best: Option<Scored> = None;
    let mut ranked: Vec<Scored> = Vec::new();
    let mut last_eval_code = Instant::now();
    let mut last_leader: Option<String> = None;
    let mut target_hit = false;

    'gens: for gen in 0..cfg.gens.max(1) {
        if obs.is_cancelled() {
            break;
        }
        if st.rung_evals[0] > 0
            && cfg.time_budget_secs > 0.0
            && t0.elapsed().as_secs_f64() > cfg.time_budget_secs
        {
            break;
        }

        for genome in &population {
            let key = genome.to_string();
            st.occurrences
                .entry(key)
                .and_modify(|occurrence| {
                    occurrence.count = occurrence.count.saturating_add(1);
                    occurrence.last_generation = gen;
                })
                .or_insert(Occurrence {
                    count: 1,
                    first_generation: gen,
                    last_generation: gen,
                });
        }

        // Rung 0 (junk filter): every genome not yet raced.
        let new_candidates: Vec<(String, DslExpr)> = {
            let mut seen = std::collections::HashSet::new();
            population
                .iter()
                .filter_map(|g| {
                    let key = DslExpr {
                        stages: g.stages.clone(),
                    }
                    .to_string();
                    if !seen.insert(key.clone()) || st.memo.contains_key(&key) {
                        return None;
                    }
                    Some((key, g.clone()))
                })
                .collect()
        };
        eval_rung_policy(&ctx, &mut st, &mut admission, 0, gen, &new_candidates, obs);

        // Rung 1: top half of this generation's new candidates by rung-0 score.
        let promoted1 = promote(&st, &new_candidates, 0, 1, cfg.rank_robust);
        eval_rung_policy(&ctx, &mut st, &mut admission, 1, gen, &promoted1, obs);

        // Rung 2: top half of the rung-1 survivors (~25% of new).
        let promoted2 = promote(&st, &promoted1, 1, 2, cfg.rank_robust);
        eval_rung_policy(&ctx, &mut st, &mut admission, 2, gen, &promoted2, obs);

        // Rank by highest-available-rung mean (lower rungs break ties).
        let mut recs = rank_race_policy(&population, &st.memo, cfg.rank_robust, &admission);
        if recs.is_empty() {
            // population collapsed to invalid genomes — reseed randoms and continue
            population = (0..cfg.pop)
                .map(|_| rand_genome_from(&mut rng, cfg.max_transforms, &gen_vocab))
                .collect();
            continue;
        }

        // Elites must hold a rung-2 (highest-fidelity) score: re-validate until the elite band
        // is rung-2-attempted or the rung-2 budget runs out (no progress => stop).
        let elite = (cfg.pop / 8).clamp(1, 8);
        loop {
            let elite_todo: Vec<(String, DslExpr)> = recs
                .iter()
                .take(elite)
                .filter(|r| {
                    st.memo
                        .get(&r.scored.dsl)
                        .map(|e| e.rungs[2].is_none())
                        .unwrap_or(false)
                })
                .filter_map(|r| parse_pipe(&r.scored.dsl).map(|g| (r.scored.dsl.clone(), g)))
                .collect();
            if elite_todo.is_empty() {
                break;
            }
            let before = st.rung_evals[2];
            eval_rung_policy(&ctx, &mut st, &mut admission, 2, gen, &elite_todo, obs);
            if st.rung_evals[2] == before {
                break; // budget exhausted — elites keep their lower-rung scores
            }
            recs = rank_race_policy(&population, &st.memo, cfg.rank_robust, &admission);
        }
        if recs.is_empty() {
            // Only a non-legacy fault policy empties the ranking here, by rejecting every ranked
            // candidate at its rung-2 re-check. Reseed randoms and continue, as above.
            population = (0..cfg.pop)
                .map(|_| rand_genome_from(&mut rng, cfg.max_transforms, &gen_vocab))
                .collect();
            continue;
        }
        ranked = recs.iter().map(|r| r.scored.clone()).collect();

        // Stream ONLY the generation leader's code (ephemeral) — identical contract to v1.
        {
            let leader_changed = last_leader.as_deref() != Some(ranked[0].dsl.as_str());
            let now = Instant::now();
            let first = last_leader.is_none();
            if leader_changed
                && (first || now.duration_since(last_eval_code).as_millis() >= EVAL_CODE_SAMPLE_MS)
            {
                if let Some(py) = GenomeIr::parse(&ranked[0].dsl)
                    .ok()
                    .and_then(|gm| emit(&gm, &EmitOptions::default()).ok())
                    .map(|e| e.python)
                {
                    last_eval_code = now;
                    last_leader = Some(ranked[0].dsl.clone());
                    obs.on_eval(&ranked[0].dsl, ranked[0].cv_mean, Some(&py));
                }
            }
        }

        let elapsed = t0.elapsed().as_secs_f64().max(1e-9);
        let counters = admission
            .as_ref()
            .map_or_else(FaultCounters::default, |admission| {
                admission.counters(&population, &ranked, elite, |key| best_rung(&st.memo, key))
            });
        let mut stats = GenStats {
            gen,
            gens: cfg.gens,
            evals: st.evals,
            evals_per_sec: st.evals as f64 / elapsed,
            tokens_total: st.ledger.total(),
            tokens_per_sec: st.ledger.rate(),
            best_cv: ranked[0].cv_mean,
            pop_fitness: ranked.iter().map(|s| s.cv_mean).collect(),
            hard_rejected_share: counters.hard_rejected_share,
            hard_rejected: counters.hard_rejected,
            pop_domain_rate: counters.pop_domain_rate,
            elite_domain_rate: counters.elite_domain_rate,
        };

        // New best? Emit its FULL dual-language source and stream it.
        let improved = best
            .as_ref()
            .map(|b| ranked[0].cv_mean > b.cv_mean + 1e-9)
            .unwrap_or(true);
        if improved {
            best = Some(ranked[0].clone());
            if let Ok(genome) = GenomeIr::parse(&ranked[0].dsl) {
                if let Ok(emitted) = emit(&genome, &EmitOptions::default()) {
                    st.ledger.add(
                        (emitted.py_tokens.code_tokens
                            + emitted.py_tokens.doc_tokens
                            + emitted.rs_tokens.code_tokens
                            + emitted.rs_tokens.doc_tokens) as f64,
                    );
                    stats.tokens_total = st.ledger.total();
                    stats.tokens_per_sec = st.ledger.rate();
                    obs.on_improvement(&ranked[0], &emitted, &stats);
                }
            }
        }
        obs.on_generation(&stats);

        // Opportunistic target exit: a rung-2-VALIDATED score clears the target + margin.
        if let Some(target) = cfg.target_score {
            if recs
                .iter()
                .any(|r| r.means[2] >= target + RACE_TARGET_MARGIN)
            {
                target_hit = true;
                break 'gens;
            }
        }

        // Next generation: elites survive, the rest are tournament offspring (same as v1).
        let mut next: Vec<DslExpr> = ranked
            .iter()
            .take(elite)
            .filter_map(|s| match GenomeIr::parse(&s.dsl) {
                Ok(GenomeIr::Pipeline(d)) => Some(d),
                _ => None,
            })
            .collect();
        let parents = ranked.len().min(cfg.pop / 2).max(1);
        let tie_cells = admission.as_ref().and_then(|admission| {
            admission.tie_break_cells(&ranked, |key| best_rung(&st.memo, key))
        });
        // B10 lexicase: begin
        let matrix = lexicase.map(|_| {
            lexicase_matrix(&ranked, |dsl| {
                let entry = st.memo.get(dsl)?;
                entry.cases[best_rung(&st.memo, dsl)].as_deref()
            })
        });
        let mut draws = start_lexicase(lexicase, matrix.as_ref(), cfg.seed, gen);
        // B10 lexicase: end
        while next.len() < cfg.pop {
            let pa = pick_parent(&mut rng, &ranked, parents, tie_cells.as_deref(), &mut draws);
            let child = if rng.next_f64() < 0.25 && ranked.len() >= 2 {
                let pb = pick_parent(&mut rng, &ranked, parents, tie_cells.as_deref(), &mut draws);
                let (ga, gb) = (parse_pipe(&ranked[pa].dsl), parse_pipe(&ranked[pb].dsl));
                match (ga, gb) {
                    (Some(a), Some(b)) => crossover(&mut rng, &a, &b, cfg.max_transforms),
                    _ => rand_genome_from(&mut rng, cfg.max_transforms, &gen_vocab),
                }
            } else {
                match parse_pipe(&ranked[pa].dsl) {
                    Some(a) => mutate(&mut rng, &a, cfg.max_transforms),
                    None => rand_genome_from(&mut rng, cfg.max_transforms, &gen_vocab),
                }
            };
            next.push(child);
        }
        report_selection(obs, gen, ranked.len(), draws);
        population = next;
    }

    let occurrences = st.occurrences;
    let mut candidates: Vec<RaceScores> = st
        .memo
        .into_iter()
        .map(|(dsl, e)| {
            let occurrence = occurrences.get(&dsl).copied().unwrap_or(Occurrence {
                count: 1,
                first_generation: e.r0_generation.unwrap_or(0),
                last_generation: e.r0_generation.unwrap_or(0),
            });
            RaceScores {
                dsl,
                rung_fold_scores: e.rungs.map(Option::flatten),
                r0_seq: e.r0_seq,
                r0_generation: e.r0_generation,
                r0_generation_slot: e.r0_generation_slot,
                r0_fold_routes: e.r0_fold_routes,
                r0_exact_surrogate_dedup: e.r0_exact_surrogate_dedup,
                recurrence_count: occurrence.count,
                first_generation: occurrence.first_generation,
                last_generation: occurrence.last_generation,
            }
        })
        .collect();
    candidates.sort_by(|a, b| a.dsl.cmp(&b.dsl));

    RaceOutcome {
        ranked,
        candidates,
        target_hit,
        rung_evals: st.rung_evals,
        rung_secs: st.spent,
        gpu_screen_timings: st.gpu_screen_timings,
        rung0_exact_surrogate_dedup: st.rung0_exact_surrogate_dedup,
        rung0_rows: ctx.rows[0].clone(),
        rung0_assignments: ctx.assigns[0].clone(),
    }
}

fn parse_pipe(dsl: &str) -> Option<DslExpr> {
    match GenomeIr::parse(dsl) {
        Ok(GenomeIr::Pipeline(d)) => Some(d),
        _ => None,
    }
}

/// Tournament-of-3 over the top `parents` ranked entries; returns an index into `ranked`.
fn tournament(rng: &mut Pcg32, ranked: &[Scored], parents: usize) -> usize {
    let cap = parents.min(ranked.len()).max(1);
    let mut bi = (rng.next_u32() as usize) % cap;
    for _ in 0..2 {
        let c = (rng.next_u32() as usize) % cap;
        if ranked[c].cv_mean > ranked[bi].cv_mean {
            bi = c;
        }
    }
    bi
}

/// [`tournament`] under a tie-breaking `Penalize`: the same three draws, but a contender within
/// [`TIE_TOLERANCE`](crate::kernel::admission::TIE_TOLERANCE) of the current winner and with a
/// different count of faulted cells wins exactly when its count is lower ([`outranks`]).
/// `faulted_cells[i]` belongs to `ranked[i]`.
fn tournament_admitted(
    rng: &mut Pcg32,
    ranked: &[Scored],
    parents: usize,
    faulted_cells: &[u64],
) -> usize {
    let cap = parents.min(ranked.len()).max(1);
    let mut bi = (rng.next_u32() as usize) % cap;
    for _ in 0..2 {
        let c = (rng.next_u32() as usize) % cap;
        if outranks(
            ranked[c].cv_mean,
            faulted_cells[c],
            ranked[bi].cv_mean,
            faulted_cells[bi],
        ) {
            bi = c;
        }
    }
    bi
}

/// [`tournament`], or [`tournament_admitted`] when a tie-breaking policy gives the faulted cells
/// of `ranked`.
fn select_parent(
    rng: &mut Pcg32,
    ranked: &[Scored],
    parents: usize,
    faulted_cells: Option<&[u64]>,
) -> usize {
    match faulted_cells {
        None => tournament(rng, ranked, parents),
        Some(faulted_cells) => tournament_admitted(rng, ranked, parents, faulted_cells),
    }
}

// B10 lexicase: begin
/// The surrogate search's lexicase case rows: the first [`LEXICASE_CASE_ROWS`] rows of
/// `permute_indices(n, seed)`, ascending. It draws nothing from the search stream.
fn lexicase_case_rows(n: usize, seed: u64) -> Vec<usize> {
    let mut rows = permute_indices(n, seed);
    rows.truncate(LEXICASE_CASE_ROWS);
    rows.sort_unstable();
    rows
}

/// The case matrix of `ranked`, in ranking order, from each candidate's residual cases.
fn lexicase_matrix<'cases>(
    ranked: &[Scored],
    cases_of: impl Fn(&str) -> Option<&'cases [f64]>,
) -> Result<CaseErrors, String> {
    let mut residuals = Vec::new();
    let mut width = None;
    for scored in ranked {
        let cases =
            cases_of(&scored.dsl).ok_or_else(|| format!("no residual cases for {}", scored.dsl))?;
        if *width.get_or_insert(cases.len()) != cases.len() {
            return Err("the ranked candidates have different case counts".to_string());
        }
        residuals.extend_from_slice(cases);
    }
    CaseErrors::from_residuals(ranked.len(), width.unwrap_or(0), &residuals)
        .map_err(|error| format!("lexicase case matrix: {error}"))
}

/// One generation's parent draws: `None` under tournament-3; under lexicase the selector and the
/// generation's stream, or why the generation falls back to tournament-3.
type LexicaseDraws<'matrix> = Option<Result<(Selector<'matrix>, Pcg32), String>>;

fn start_lexicase<'matrix>(
    config: Option<LexicaseConfig>,
    matrix: Option<&'matrix Result<CaseErrors, String>>,
    seed: u64,
    generation: usize,
) -> LexicaseDraws<'matrix> {
    let config = config?;
    let matrix = match matrix? {
        Ok(matrix) => matrix,
        Err(reason) => return Some(Err(reason.clone())),
    };
    let mut stream = Pcg32::new(seed, LEXICASE_STREAM.wrapping_add(generation as u64));
    Some(
        Selector::new(matrix, config, &mut stream)
            .map(|selector| (selector, stream))
            .map_err(|error| format!("lexicase selection: {error}")),
    )
}

/// One lexicase event on the generation's stream; otherwise, under tournament-3 or a generation
/// that fell back, [`select_parent`] with the identical arguments and stream.
fn pick_parent(
    rng: &mut Pcg32,
    ranked: &[Scored],
    parents: usize,
    faulted_cells: Option<&[u64]>,
    lexicase: &mut LexicaseDraws<'_>,
) -> usize {
    match lexicase {
        Some(Ok((selector, stream))) => selector.draw_parent(stream),
        _ => select_parent(rng, ranked, parents, faulted_cells),
    }
}

/// Reports a lexicase generation's selection; nothing under tournament-3.
fn report_selection(obs: &dyn Observer, generation: usize, pool: usize, lexicase: LexicaseDraws) {
    if let Some(outcome) = lexicase {
        obs.on_selection(&SelectionReceipt {
            generation,
            pool,
            selection: outcome.map(|(selector, _)| selector.finish()),
        });
    }
}
// B10 lexicase: end

#[cfg(test)]
mod robust_rank_tests {
    use super::*;

    fn entry(rung: usize, folds: Vec<f64>) -> RaceEntry {
        let mut e = RaceEntry::default();
        e.rungs[rung] = Some(Some(folds));
        e
    }

    /// The penalty must equal the downstream gate's arithmetic exactly. The gate is
    /// `max(oof,-1) - 0.25*min(population_std(folds),1)`; here the rung mean stands in for oof.
    /// POPULATION std (÷n), not sample std (÷n-1) — this is the detail most likely to be got
    /// wrong, and getting it wrong shifts every comparison in the search.
    #[test]
    fn robust_matches_the_gate_formula() {
        // folds [0.4, 0.6]: mean 0.5, population std 0.1 -> 0.5 - 0.25*0.1 = 0.475
        let e = entry(1, vec![0.4, 0.6]);
        assert_eq!(e.mean(1), Some(0.5));
        let got = e.robust(1).unwrap();
        assert!((got - 0.475).abs() < 1e-12, "got {got}");

        // Sample std would be 0.1414..., giving 0.4646 — assert we did NOT do that.
        assert!(
            (got - 0.4646).abs() > 1e-3,
            "used sample std instead of population std"
        );
    }

    #[test]
    fn zero_variance_is_unpenalized_and_cap_applies() {
        let flat = entry(1, vec![0.3, 0.3, 0.3]);
        assert_eq!(flat.robust(1), flat.mean(1));

        // population std here far exceeds 1.0, so the penalty clamps at 0.25.
        let wild = entry(1, vec![-5.0, 5.0]);
        let m = wild.mean(1).unwrap();
        let r = wild.robust(1).unwrap();
        assert!(
            (m - r - 0.25).abs() < 1e-12,
            "cap not applied: mean {m} robust {r}"
        );
    }

    /// A high-mean/unstable genome must lose to a slightly-lower-mean/stable one under the
    /// robust key, and win under the mean key. This is the entire point of the change: the
    /// gate already prefers the stable one, so search should too.
    #[test]
    fn robust_key_reverses_an_unstable_winner() {
        let unstable = entry(2, vec![0.30, 0.02]); // mean 0.16, pop sd 0.14 -> 0.125
        let stable = entry(2, vec![0.15, 0.15]); // mean 0.15, pop sd 0.00 -> 0.150

        assert!(
            unstable.mean(2).unwrap() > stable.mean(2).unwrap(),
            "mean key prefers unstable"
        );
        assert!(
            stable.robust(2).unwrap() > unstable.robust(2).unwrap(),
            "robust key must prefer the stable genome"
        );
    }

    #[test]
    fn rank_score_selects_the_key_and_defaults_off() {
        let e = entry(1, vec![0.4, 0.6]);
        assert_eq!(e.rank_score(1, false), e.mean(1));
        assert_eq!(e.rank_score(1, true), e.robust(1));
        // Default config must leave the existing lane untouched.
        assert!(!EvolveConfig::default().rank_robust);
    }

    #[test]
    fn unscored_rungs_stay_none_under_both_keys() {
        let e = entry(1, vec![0.5]);
        assert_eq!(e.rank_score(0, true), None);
        assert_eq!(e.rank_score(2, true), None);
        // a single fold has no variance, so robust == mean
        assert_eq!(e.rank_score(1, true), e.mean(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountObs {
        gens: AtomicUsize,
        improvements: AtomicUsize,
    }
    impl Observer for CountObs {
        fn on_generation(&self, _s: &GenStats) {
            self.gens.fetch_add(1, Ordering::Relaxed);
        }
        fn on_improvement(&self, best: &Scored, emitted: &Emitted, _s: &GenStats) {
            assert!(emitted.python.contains("def "), "python artifact expected");
            assert!(emitted.rust.contains("fn main"), "rust artifact expected");
            assert!(best.cv_mean.is_finite());
            self.improvements.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records every `on_eval` — how many carried code vs were score-only, and the code payloads.
    struct CodeObs {
        code_evals: AtomicUsize,
        score_evals: AtomicUsize,
        code_pythons: std::sync::Mutex<Vec<String>>,
    }
    impl Observer for CodeObs {
        fn on_generation(&self, _s: &GenStats) {}
        fn on_improvement(&self, _best: &Scored, _emitted: &Emitted, _s: &GenStats) {}
        fn on_eval(&self, _dsl: &str, _score: f64, code: Option<&str>) {
            match code {
                Some(c) => {
                    self.code_evals.fetch_add(1, Ordering::Relaxed);
                    self.code_pythons.lock().unwrap().push(c.to_string());
                }
                None => {
                    self.score_evals.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    /// y = 3*x0 - 2*x1 + noise: evolution must find a strongly-fitting pipeline fast.
    #[test]
    fn evolve_improves_on_synthetic_regression() {
        let mut rng = Pcg32::new(11, 4);
        let n = 240;
        let x: Rows = (0..n)
            .map(|_| (0..4).map(|_| rng.gaussian()).collect::<Vec<f64>>())
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|r| 3.0 * r[0] - 2.0 * r[1] + 0.05 * rng.gaussian())
            .collect();
        let metric = |yt: &[f64], yp: &[f64]| crate::metrics::r2(yt, yp);
        let cfg = EvolveConfig {
            pop: 16,
            gens: 4,
            folds: 3,
            seed: 7,
            max_transforms: 3,
            time_budget_secs: 30.0,
            seed_genomes: Vec::new(),
            fitness: FitnessMode::Surrogate,
            target_score: None,
            rank_robust: false,
            hard_ten_priors: false,
            expansion_vocab: false,
            fault_policy: FaultPolicy::Legacy,
            selection: ParentSelection::Tournament3,
            ridge_scoring: RidgeScoring::Refit,
        };
        let obs = CountObs {
            gens: AtomicUsize::new(0),
            improvements: AtomicUsize::new(0),
        };
        let ranked = evolve(&x, &y, &metric, &cfg, &obs);
        assert!(!ranked.is_empty());
        assert!(
            ranked[0].cv_mean > 0.8,
            "near-linear target should fit well, got {}",
            ranked[0].cv_mean
        );
        assert!(obs.gens.load(Ordering::Relaxed) >= 1);
        assert!(obs.improvements.load(Ordering::Relaxed) >= 1);
        assert_eq!(ranked[0].fold_scores.len(), 3);
    }

    /// Fixed-fixture pin of the v1 (ridge-surrogate) search path: same seed + data must always
    /// produce the same champion and the same fold scores, bit for bit. Captured from the
    /// pre-fitness-v2 build; guards the v1 code path against accidental drift.
    #[test]
    fn v1_regression_bit_identical_champion() {
        let mut rng = Pcg32::new(42, 1);
        let n = 200;
        let x: Rows = (0..n)
            .map(|_| (0..5).map(|_| rng.gaussian()).collect::<Vec<f64>>())
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|r| r[0] * r[0] - 1.5 * r[1] + 0.5 * (r[2] * 2.0).sin() + 0.02 * rng.gaussian())
            .collect();
        let metric = |yt: &[f64], yp: &[f64]| crate::metrics::r2(yt, yp);
        let cfg = EvolveConfig {
            pop: 16,
            gens: 3,
            folds: 3,
            seed: 12345,
            max_transforms: 3,
            time_budget_secs: 0.0,
            seed_genomes: Vec::new(),
            fitness: FitnessMode::Surrogate,
            target_score: None,
            rank_robust: false,
            hard_ten_priors: false,
            expansion_vocab: false,
            fault_policy: FaultPolicy::Legacy,
            selection: ParentSelection::Tournament3,
            ridge_scoring: RidgeScoring::Refit,
        };
        let obs = CountObs {
            gens: AtomicUsize::new(0),
            improvements: AtomicUsize::new(0),
        };
        let ranked = evolve(&x, &y, &metric, &cfg, &obs);
        assert!(!ranked.is_empty());
        // Expected values captured from the pre-fitness-v2 build (bit-exact).
        assert_eq!(
            ranked[0].dsl,
            "(pipe X (rank append=true) (poly_interact append=true) (histgbm2 bags=3 \
             col_subsample=0.6 leaves=255 lr=0.08 min_leaf=10 rounds=500 row_subsample=0.6 \
             val_pct=20 val_shuffle=true))"
        );
        assert_eq!(ranked[0].cv_mean.to_bits(), 4607139655722794152);
        assert_eq!(
            ranked[0]
                .fold_scores
                .iter()
                .map(|s| s.to_bits())
                .collect::<Vec<u64>>(),
            vec![
                4607151003950554968,
                4607135252342903357,
                4607132710874924134
            ]
        );
    }

    /// The live code panel is fed ONLY by per-generation leaders: `on_eval` carries code at most
    /// once per generation (never an arbitrary-candidate spray), and every code payload is a real
    /// transpiled algorithm. Score-only ticks still fire so the ticker/scatter stay live.
    #[test]
    fn code_stream_is_leader_only() {
        let mut rng = Pcg32::new(5, 9);
        let n = 200;
        let x: Rows = (0..n)
            .map(|_| (0..4).map(|_| rng.gaussian()).collect::<Vec<f64>>())
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|r| 2.0 * r[0] - r[2] + 0.05 * rng.gaussian())
            .collect();
        let metric = |yt: &[f64], yp: &[f64]| crate::metrics::r2(yt, yp);
        let cfg = EvolveConfig {
            pop: 16,
            gens: 5,
            folds: 3,
            seed: 3,
            max_transforms: 3,
            time_budget_secs: 30.0,
            seed_genomes: Vec::new(),
            fitness: FitnessMode::Surrogate,
            target_score: None,
            rank_robust: false,
            hard_ten_priors: false,
            expansion_vocab: false,
            fault_policy: FaultPolicy::Legacy,
            selection: ParentSelection::Tournament3,
            ridge_scoring: RidgeScoring::Refit,
        };
        let obs = CodeObs {
            code_evals: AtomicUsize::new(0),
            score_evals: AtomicUsize::new(0),
            code_pythons: std::sync::Mutex::new(Vec::new()),
        };
        let ranked = evolve(&x, &y, &metric, &cfg, &obs);
        assert!(!ranked.is_empty());
        let code_evals = obs.code_evals.load(Ordering::Relaxed);
        // At least one leader streamed code, but NEVER more than one per generation (no spray).
        assert!(code_evals >= 1, "expected at least one leader code emit");
        assert!(
            code_evals <= cfg.gens,
            "code must be leader-only (<= 1/gen), got {code_evals} over {} gens",
            cfg.gens
        );
        // Every streamed leader is a real, transpiled algorithm.
        for py in obs.code_pythons.lock().unwrap().iter() {
            assert!(
                py.contains("def "),
                "leader code must be valid python, got: {py:?}"
            );
        }
    }

    #[test]
    fn public_offspring_wrappers_are_canonical_and_seed_deterministic() {
        let parent_a = "(pipe X (standardize) (rff_cos n=32 seed=7) (ridge))";
        let parent_b = "(pipe X (winsor_clip q=0.05) (cheb degree=3) (elasticnet alpha=0.01))";

        let random = random_canonical_pipeline(17, 3);
        assert_eq!(random, random_canonical_pipeline(17, 3));

        let mutated = mutate_canonical_pipeline(parent_a, 29, 3).unwrap();
        assert_eq!(mutated, mutate_canonical_pipeline(parent_a, 29, 3).unwrap());

        let crossed = crossover_canonical_pipelines(parent_a, parent_b, 41, 3).unwrap();
        assert_eq!(
            crossed,
            crossover_canonical_pipelines(parent_a, parent_b, 41, 3).unwrap()
        );

        for source in [random, mutated, crossed] {
            let genome = GenomeIr::parse(&source).unwrap();
            assert!(matches!(genome, GenomeIr::Pipeline(_)));
            assert_eq!(source, genome.canonical());
        }
        assert!(mutate_canonical_pipeline("(sym affine \"col(0)\")", 1, 3).is_err());
        assert!(crossover_canonical_pipelines(parent_a, "not-a-genome", 1, 3).is_err());
    }

    // ------------------------------ fitness v2 ------------------------------

    fn race_fixture(seed: u64, n: usize) -> (Rows, Vec<f64>) {
        let mut rng = Pcg32::new(seed, 17);
        let x: Rows = (0..n)
            .map(|_| (0..4).map(|_| rng.gaussian()).collect::<Vec<f64>>())
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|r| 2.5 * r[0] - 1.2 * r[1] + 0.4 * r[2] * r[2] + 0.05 * rng.gaussian())
            .collect();
        (x, y)
    }

    fn race_cfg(seed: u64) -> EvolveConfig {
        EvolveConfig {
            pop: 16,
            gens: 3,
            folds: 3,
            seed,
            max_transforms: 3,
            time_budget_secs: 0.0,
            seed_genomes: Vec::new(),
            fitness: FitnessMode::LiteRace,
            target_score: None,
            rank_robust: false,
            hard_ten_priors: false,
            expansion_vocab: false,
            fault_policy: FaultPolicy::Legacy,
            selection: ParentSelection::Tournament3,
            ridge_scoring: RidgeScoring::Refit,
        }
    }

    fn quiet_obs() -> CountObs {
        CountObs {
            gens: AtomicUsize::new(0),
            improvements: AtomicUsize::new(0),
        }
    }

    #[test]
    fn difficult_ten_priors_are_opt_in_generation_zero_candidates() {
        let (x, y) = race_fixture(73, 160);
        let metric = |yt: &[f64], yp: &[f64]| crate::metrics::r2(yt, yp);
        let mut cfg = race_cfg(991);
        cfg.pop = 24;
        cfg.gens = 1;
        cfg.hard_ten_priors = true;
        let outcome = evolve_lite_race(&x, &y, &metric, &cfg, &quiet_obs());
        let raced = outcome
            .candidates
            .iter()
            .map(|candidate| candidate.dsl.as_str())
            .collect::<std::collections::HashSet<_>>();
        for source in HARD_TEN_STARTERS {
            let canonical = GenomeIr::parse(source).unwrap().canonical();
            assert!(
                raced.contains(canonical.as_str()),
                "missing difficult-ten prior {canonical}"
            );
        }
        assert!(!EvolveConfig::default().hard_ten_priors);
    }

    fn exact_population(count: usize) -> Vec<String> {
        (0..count)
            .map(|index| {
                format!(
                    "(pipe X (standardize) (rff_cos gamma=0.5 n={} seed={}) (ridge))",
                    16 + index,
                    100 + index
                )
            })
            .collect()
    }

    #[test]
    fn exact_initial_population_is_generation_zero_without_hidden_seeds() {
        let (x, y) = race_fixture(91, 160);
        let metric = |yt: &[f64], yp: &[f64]| crate::metrics::r2(yt, yp);
        let initial = exact_population(8);
        let mut cfg = race_cfg(53);
        cfg.pop = initial.len();
        cfg.gens = 1;
        cfg.seed_genomes = vec![CHAMPION_GENOME.to_string(), STARTERS[0].to_string()];

        let out = evolve_lite_race_with_exact_initial_population(
            &x,
            &y,
            &metric,
            &cfg,
            &quiet_obs(),
            &initial,
        )
        .unwrap();
        let expected = initial
            .iter()
            .map(|source| GenomeIr::parse(source).unwrap().canonical())
            .collect::<std::collections::BTreeSet<_>>();
        let observed = out
            .candidates
            .iter()
            .map(|candidate| candidate.dsl.clone())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(observed, expected);
        assert_eq!(out.rung_evals[0], initial.len() as u64);
        assert!(out.candidates.iter().all(|candidate| {
            candidate.recurrence_count == 1
                && candidate.first_generation == 0
                && candidate.last_generation == 0
        }));
        assert!(!observed.contains(&GenomeIr::parse(CHAMPION_GENOME).unwrap().canonical()));
        assert!(!observed.contains(&GenomeIr::parse(STARTERS[0]).unwrap().canonical()));
    }

    #[test]
    fn exact_initial_population_rejects_count_invalid_kind_and_canonical_duplicates() {
        let metric = |_yt: &[f64], _yp: &[f64]| 0.0;
        let mut cfg = race_cfg(1);
        cfg.pop = 2;

        let wrong_count = exact_population(1);
        assert!(evolve_lite_race_with_exact_initial_population(
            &Vec::new(),
            &[],
            &metric,
            &cfg,
            &quiet_obs(),
            &wrong_count,
        )
        .unwrap_err()
        .contains("cfg.pop"));

        let invalid_kind = vec![
            "(pipe X (ridge))".to_string(),
            "(sym affine \"col(0)\")".to_string(),
        ];
        assert!(evolve_lite_race_with_exact_initial_population(
            &Vec::new(),
            &[],
            &metric,
            &cfg,
            &quiet_obs(),
            &invalid_kind,
        )
        .unwrap_err()
        .contains("pipeline"));

        let duplicates = vec![
            "(pipe X (standardize) (ridge))".to_string(),
            "(pipe X   (standardize)   (ridge))".to_string(),
        ];
        assert!(evolve_lite_race_with_exact_initial_population(
            &Vec::new(),
            &[],
            &metric,
            &cfg,
            &quiet_obs(),
            &duplicates,
        )
        .unwrap_err()
        .contains("duplicates"));
    }

    /// lite_budget keeps the real head op and every non-budget param, caps only the budget knobs
    /// from LITE_HEAD_BUDGETS, and applies the surrogate's transform clamps.
    #[test]
    fn lite_budget_caps_budget_knobs_and_preserves_everything_else() {
        let g = parse_pipe(
            "(pipe X (cheb degree=8) (rff_cos gamma=0.5 n=96 seed=7) (histgbm2 bags=5 \
             leaves=63 lr=0.05 rounds=1000 val_pct=15))",
        )
        .unwrap();
        let lite = lite_budget(&g);
        // transform clamps mirror search_surrogate's
        assert_eq!(
            lite.stages[0].params.get("degree"),
            Some(&ParamValue::Int(4))
        );
        assert_eq!(lite.stages[1].params.get("n"), Some(&ParamValue::Int(64)));
        assert_eq!(
            lite.stages[1].params.get("gamma"),
            Some(&ParamValue::Float(0.5))
        );
        // the REAL head survives with budget knobs capped and tuning knobs untouched
        let head = lite.stages.last().unwrap();
        assert_eq!(head.op, "histgbm2");
        assert_eq!(head.params.get("rounds"), Some(&ParamValue::Int(40)));
        assert_eq!(head.params.get("bags"), Some(&ParamValue::Int(1)));
        assert_eq!(head.params.get("leaves"), Some(&ParamValue::Int(63)));
        assert_eq!(head.params.get("lr"), Some(&ParamValue::Float(0.05)));
        assert_eq!(head.params.get("val_pct"), Some(&ParamValue::Int(15)));

        let g = parse_pipe("(pipe X (gpr kernel=rbf+linear m=400 steps=40))").unwrap();
        let lite = lite_budget(&g);
        let head = lite.stages.last().unwrap();
        assert_eq!(head.op, "gpr");
        assert_eq!(head.params.get("steps"), Some(&ParamValue::Int(5)));
        assert_eq!(head.params.get("m"), Some(&ParamValue::Int(96)));
        assert_eq!(
            head.params.get("kernel"),
            Some(&ParamValue::Str("rbf+linear".into()))
        );

        // heads outside the budget table pass through unchanged
        let g = parse_pipe("(pipe X (standardize) (ridge_gcv))").unwrap();
        assert_eq!(lite_budget(&g), g);
        let g = parse_pipe("(pipe X (ksmooth gamma=1.0 k=32))").unwrap();
        assert_eq!(lite_budget(&g), g);
    }

    /// Memo/promotion invariants: a higher-rung score implies the genome passed rung 0; ranked
    /// fitness always equals the mean at the genome's HIGHEST populated rung (a low-rung score is
    /// never served as a high-rung one); the champion is rung-2 validated.
    #[test]
    fn race_promotes_through_rungs_and_ranks_by_highest_rung() {
        let (x, y) = race_fixture(31, 300);
        let metric = |yt: &[f64], yp: &[f64]| crate::metrics::r2(yt, yp);
        let obs = quiet_obs();
        let out = evolve_lite_race(&x, &y, &metric, &race_cfg(9), &obs);
        assert!(!out.ranked.is_empty());
        assert!(
            out.rung_evals[0] > 0 && out.rung_evals[1] > 0 && out.rung_evals[2] > 0,
            "all rungs must race, got {:?}",
            out.rung_evals
        );
        // successive halving really halves (rung1 <= rung0, rung2 evals bounded by rung1 + elites)
        assert!(out.rung_evals[1] <= out.rung_evals[0]);
        let by_dsl: HashMap<&str, &RaceScores> =
            out.candidates.iter().map(|c| (c.dsl.as_str(), c)).collect();
        for c in &out.candidates {
            for r in [1usize, 2] {
                if c.rung_fold_scores[r].is_some() {
                    assert!(
                        c.rung_fold_scores[0].is_some(),
                        "rung {r} score without a rung-0 pass: {}",
                        c.dsl
                    );
                }
            }
            for ((r, fs), folds) in c.rung_fold_scores.iter().enumerate().zip(RACE_RUNG_FOLDS) {
                if let Some(fs) = fs {
                    assert_eq!(fs.len(), folds, "fold count at rung {r}");
                }
            }
        }
        for s in &out.ranked {
            let c = by_dsl[s.dsl.as_str()];
            let best = (0..3).rev().find_map(|r| c.rung_mean(r)).unwrap();
            assert!(
                (s.cv_mean - best).abs() < 1e-12,
                "ranked score must be the highest-rung mean for {}",
                s.dsl
            );
        }
        // the champion went through the full race (elite rung-2 validation)
        let champ = by_dsl[out.ranked[0].dsl.as_str()];
        assert!(
            champ.rung_fold_scores[2].is_some(),
            "champion not rung-2 validated"
        );
    }

    /// Same seed, same data, no wall budget => identical outcome (ranking AND full memo dump).
    #[test]
    fn race_is_deterministic_for_a_fixed_seed() {
        let (x, y) = race_fixture(77, 300);
        let metric = |yt: &[f64], yp: &[f64]| crate::metrics::r2(yt, yp);
        let a = evolve_lite_race(&x, &y, &metric, &race_cfg(41), &quiet_obs());
        let b = evolve_lite_race(&x, &y, &metric, &race_cfg(41), &quiet_obs());
        assert_eq!(a.ranked.len(), b.ranked.len());
        for (sa, sb) in a.ranked.iter().zip(&b.ranked) {
            assert_eq!(sa.dsl, sb.dsl);
            assert_eq!(sa.cv_mean.to_bits(), sb.cv_mean.to_bits());
            assert_eq!(sa.fold_scores, sb.fold_scores);
        }
        assert_eq!(a.candidates, b.candidates);
        assert_eq!(a.rung_evals, b.rung_evals);
    }

    /// Even a budget already exhausted by setup admits exactly one bounded
    /// rung-0 chunk, then terminates promptly. This is deterministic under
    /// concurrent test load and prevents a positive-budget race from returning
    /// no ranking merely because its worker was descheduled before admission.
    #[test]
    fn race_respects_tiny_time_budget() {
        let (x, y) = race_fixture(5, 400);
        let metric = |yt: &[f64], yp: &[f64]| crate::metrics::r2(yt, yp);
        let mut cfg = race_cfg(13);
        cfg.pop = 64;
        cfg.gens = 10_000;
        cfg.time_budget_secs = f64::MIN_POSITIVE;
        let t = Instant::now();
        let out = evolve_lite_race(&x, &y, &metric, &cfg, &quiet_obs());
        assert!(
            t.elapsed().as_secs_f64() < 30.0,
            "race must respect the wall budget"
        );
        assert!(!out.ranked.is_empty());
        assert_eq!(out.rung_evals[0], RACE_CHUNK as u64);
    }

    #[test]
    fn production_wall_budget_boundary_is_testable_without_waiting() {
        assert!(!elapsed_budget_exhausted(299.999, 300.0));
        assert!(elapsed_budget_exhausted(300.0, 300.0));
        assert!(elapsed_budget_exhausted(301.0, 300.0));
        assert!(!elapsed_budget_exhausted(30_000.0, 0.0));
    }

    /// With a beatable target, the loop exits at the first generation whose rung-2 scores clear
    /// target + margin, and reports the hit.
    #[test]
    fn race_exits_early_when_target_beaten_at_rung2() {
        let (x, y) = race_fixture(19, 300);
        let metric = |yt: &[f64], yp: &[f64]| crate::metrics::r2(yt, yp);
        let mut cfg = race_cfg(23);
        cfg.gens = 50;
        cfg.target_score = Some(-0.5); // any sane rung-2 score clears this
        let obs = quiet_obs();
        let out = evolve_lite_race(&x, &y, &metric, &cfg, &obs);
        assert!(out.target_hit);
        assert_eq!(
            obs.gens.load(Ordering::Relaxed),
            1,
            "expected an exit within the first generation"
        );
        // without a target the same run goes the distance
        let mut cfg2 = race_cfg(23);
        cfg2.gens = 3;
        let out2 = evolve_lite_race(&x, &y, &metric, &cfg2, &quiet_obs());
        assert!(!out2.target_hit);
    }

    /// The expansion vocabulary is strictly opt-in: the flag defaults off and
    /// the off-path draws from a vocab slice whose contents equal the frozen
    /// legacy array, so every recorded stream is untouched (the pinned-stream
    /// suites — champion parity, the exact-cv two-generation fixture, and the
    /// deterministic race tests above — run against the off-path and prove
    /// byte-identity end to end).
    #[test]
    fn expansion_off_streams_are_byte_identical() {
        assert!(!EvolveConfig::default().expansion_vocab);
        let cfg = EvolveConfig::default();
        let legacy: Vec<&str> = TRANSFORMS.to_vec();
        assert_eq!(generation_vocab(&cfg), legacy);
        // Off-path draws are literally the legacy draws.
        let mut a = Pcg32::new(97, 3);
        let mut b = Pcg32::new(97, 3);
        for _ in 0..50 {
            let lhs = GenomeIr::Pipeline(rand_genome(&mut a, 6)).canonical();
            let rhs = GenomeIr::Pipeline(rand_genome_from(&mut b, 6, &generation_vocab(&cfg)))
                .canonical();
            assert_eq!(lhs, rhs);
        }
    }

    /// Every expansion starter is canonical (round-trips exactly), supported
    /// by the emitters, and fits + predicts finite values on toy data — the
    /// 100%-mappable requirement for seeds.
    #[test]
    fn expansion_starters_are_canonical_supported_and_fit() {
        let mut rng = Pcg32::new(31, 9);
        let n = 24;
        let x: Rows = (0..n)
            .map(|_| (0..5).map(|_| rng.gaussian()).collect::<Vec<f64>>())
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|r| 1.5 * r[0] - 0.7 * r[1] + 0.05 * rng.gaussian())
            .collect();
        for src in EXPANSION_STARTERS {
            let canonical = crate::full_rust::canonicalize(src)
                .unwrap_or_else(|e| panic!("starter rejected by canonicalize: {src}: {e}"));
            assert_eq!(
                canonical.canonical, src,
                "starter is not stored in canonical form"
            );
            let genome = GenomeIr::parse(src).unwrap();
            crate::transpile::supports(&genome)
                .unwrap_or_else(|m| panic!("supports() rejected {src}: {m:?}"));
            let predictions = crate::model::fit_predict(&genome, &x, &y, &x)
                .unwrap_or_else(|e| panic!("starter failed to fit: {src}: {e}"));
            assert_eq!(predictions.len(), n);
            assert!(
                predictions.iter().all(|p| p.is_finite()),
                "non-finite prediction from {src}"
            );
        }
    }

    /// With the flag on, generation reaches every expansion op, and every
    /// generated genome canonicalizes + emits cleanly (100% mappable).
    #[test]
    fn expansion_vocab_generates_new_ops() {
        let cfg = EvolveConfig {
            expansion_vocab: true,
            ..EvolveConfig::default()
        };
        let vocab = generation_vocab(&cfg);
        assert_eq!(vocab.len(), TRANSFORMS.len() + EXPANSION_TRANSFORMS.len());
        let mut rng = Pcg32::new(0xE0_57A6, 12);
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for _ in 0..200 {
            let genome = rand_genome_from(&mut rng, 6, &vocab);
            for stage in &genome.stages {
                seen.insert(stage.op.clone());
            }
            let source = GenomeIr::Pipeline(genome).canonical();
            let canonical = crate::full_rust::canonicalize(&source)
                .unwrap_or_else(|e| panic!("generated genome not canonicalizable: {source}: {e}"));
            let parsed = GenomeIr::parse(&canonical.canonical).unwrap();
            crate::transpile::supports(&parsed)
                .unwrap_or_else(|m| panic!("supports() rejected generated {source}: {m:?}"));
        }
        for op in EXPANSION_TRANSFORMS {
            assert!(seen.contains(op), "expansion op '{op}' never generated");
        }
    }

    /// Same (source, seed, flags) -> identical v2 mutation output.
    #[test]
    fn v2_mutation_is_deterministic() {
        let sources = [
            "(pipe X (standardize) (ridge))",
            "(pipe X (rank append=true) (winsor_clip q=0.1) (histgbm depth=4 lr=0.1 rounds=80))",
            "(pipe X (symlog c=1.0) (select_corr by=corr k=32) (elasticnet alpha=0.001 l1_ratio=0.5 sweeps=60))",
            "(pipe X (standardize) (standardize) (pair_prod topk=8) (ridge_gcv))",
        ];
        let mut cases = 0;
        for source in sources {
            let canonical = crate::full_rust::canonicalize(source).unwrap().canonical;
            for seed in 0..13u64 {
                for expansion in [false, true] {
                    let a = mutate_canonical_pipeline_v2(&canonical, seed, 6, expansion).unwrap();
                    let b = mutate_canonical_pipeline_v2(&canonical, seed, 6, expansion).unwrap();
                    assert_eq!(
                        a, b,
                        "nondeterministic v2 mutation: {canonical} seed={seed}"
                    );
                    cases += 1;
                }
            }
        }
        assert!(cases >= 100, "grid too small: {cases}");
    }

    #[test]
    fn explicit_redteam_mutation_arms_are_deterministic_and_bounded() {
        let source =
            crate::full_rust::canonicalize("(pipe X (standardize) (standardize) (rank) (ridge))")
                .unwrap()
                .canonical;
        let arms = [
            CanonicalMutationArm::ParameterJitter,
            CanonicalMutationArm::Insert,
            CanonicalMutationArm::Delete,
            CanonicalMutationArm::AdjacentSwap,
            CanonicalMutationArm::HeadReplacement,
            CanonicalMutationArm::Simplify,
        ];
        for (index, arm) in arms.into_iter().enumerate() {
            let seed = 202_607_300 + index as u64;
            let first = mutate_canonical_pipeline_with_arm(&source, seed, 10, true, arm).unwrap();
            let replay = mutate_canonical_pipeline_with_arm(&source, seed, 10, true, arm).unwrap();
            assert_eq!(first, replay);
            let parsed = GenomeIr::parse(&first).unwrap();
            assert_eq!(parsed.canonical(), first);
            let GenomeIr::Pipeline(pipeline) = parsed else {
                panic!("mutation arm returned a non-pipeline genome");
            };
            assert!(pipeline.stages.len() - 1 <= 10);
        }
    }

    /// The simplify arm deletes the later stage of an adjacent duplicate
    /// pair, and the result stays canonical + supported.
    #[test]
    fn v2_simplify_dissolves_adjacent_duplicates() {
        let source = crate::full_rust::canonicalize(
            "(pipe X (standardize) (standardize) (standardize) (standardize) (ridge))",
        )
        .unwrap()
        .canonical;
        let parent_stages = GenomeIr::parse(&source).unwrap();
        let parent_len = match &parent_stages {
            GenomeIr::Pipeline(p) => p.stages.len(),
            _ => unreachable!(),
        };
        let mut simplified = 0;
        for seed in 0..200u64 {
            let child = mutate_canonical_pipeline_v2(&source, seed, 6, false).unwrap();
            let parsed = GenomeIr::parse(&child).unwrap();
            let child_len = match &parsed {
                GenomeIr::Pipeline(p) => p.stages.len(),
                _ => unreachable!(),
            };
            if child_len < parent_len && child.matches("(standardize)").count() == 3 {
                // One duplicate dissolved (drop arm can also shorten, so
                // require exactly one standardize removed and nothing else).
                simplified += 1;
                assert_eq!(
                    GenomeIr::parse(&child).unwrap().canonical(),
                    child,
                    "simplify output not canonical"
                );
                crate::transpile::supports(&parsed)
                    .unwrap_or_else(|m| panic!("supports() rejected simplified {child}: {m:?}"));
            }
        }
        assert!(
            simplified >= 10,
            "simplify arm never fired across 200 seeds: {simplified}"
        );
    }

    /// Every v2 mutation output parses, canonicalizes to itself, passes
    /// supports(), and (sampled) fits — 100% mappable.
    #[test]
    fn v2_outputs_are_always_canonical_and_supported() {
        let mut rng = Pcg32::new(0x2BAD_5EED, 4);
        let n = 24;
        let x: Rows = (0..n)
            .map(|_| (0..5).map(|_| rng.gaussian()).collect::<Vec<f64>>())
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|r| 1.5 * r[0] - 0.7 * r[1] + 0.05 * rng.gaussian())
            .collect();
        let parents = [
            "(pipe X (standardize) (ridge))",
            "(pipe X (impute add_indicator=true strategy=median) (symlog c=1.0) (histgbm depth=3 lr=0.1 rounds=40))",
            "(pipe X (spline knots=4) (pair_ratio topk=6) (elasticnet alpha=0.01 l1_ratio=0.3 sweeps=40))",
            "(pipe X (rank) (rank) (quantile_bins bins=8) (ridge_gcv))",
            "(pipe X (winsor_clip q=0.05) (rff_cos gamma=1.0 n=16 seed=7) (ridge lambda_frac=0.05))",
        ];
        let mut checked = 0;
        let mut fitted = 0;
        for parent in parents {
            let canonical = crate::full_rust::canonicalize(parent).unwrap().canonical;
            for seed in 0..50u64 {
                let child = mutate_canonical_pipeline_v2(&canonical, seed, 6, true).unwrap();
                let parsed = GenomeIr::parse(&child)
                    .unwrap_or_else(|e| panic!("unparseable v2 child {child}: {e}"));
                assert_eq!(parsed.canonical(), child, "v2 child not canonical: {child}");
                crate::transpile::supports(&parsed)
                    .unwrap_or_else(|m| panic!("supports() rejected v2 child {child}: {m:?}"));
                checked += 1;
                if checked % 5 == 0 {
                    let predictions = crate::model::fit_predict(&parsed, &x, &y, &x)
                        .unwrap_or_else(|e| panic!("v2 child failed to fit {child}: {e}"));
                    assert!(
                        predictions.iter().all(|p| p.is_finite()),
                        "non-finite prediction from v2 child {child}"
                    );
                    fitted += 1;
                }
            }
        }
        assert_eq!(checked, 250);
        assert_eq!(fitted, 50);
    }

    /// Log-scale jitter stays positive, respects the 1e-6 floor, and spreads
    /// multiplicatively ~exp(±ln 2.5).
    #[test]
    fn v2_log_jitter_respects_clamps_and_is_logscale() {
        let mut rng = Pcg32::new(77, 21);
        let mut ratios: Vec<f64> = Vec::new();
        for _ in 0..200 {
            let mut stage = Stage::new("histgbm");
            stage
                .params
                .insert("lr".to_string(), ParamValue::Float(0.05));
            jitter_params_v2(&mut rng, &mut stage);
            let lr = stage.get_f64("lr", f64::NAN);
            assert!(lr.is_finite() && lr > 0.0, "jittered lr not positive: {lr}");
            ratios.push(lr / 0.05);
        }
        let max = ratios.iter().cloned().fold(f64::MIN, f64::max);
        let min = ratios.iter().cloned().fold(f64::MAX, f64::min);
        assert!(max <= 2.5 + 1e-9, "spread above exp(+ln 2.5): {max}");
        assert!(min >= 1.0 / 2.5 - 1e-9, "spread below exp(-ln 2.5): {min}");
        assert!(
            max / min > 4.0,
            "spread not log-scale wide enough: {max}/{min}"
        );
        // l1_ratio jitters in logit space and never leaves (0, 1).
        let mut rng = Pcg32::new(78, 22);
        for _ in 0..200 {
            let mut stage = Stage::new("elasticnet");
            stage
                .params
                .insert("l1_ratio".to_string(), ParamValue::Float(0.5));
            jitter_params_v2(&mut rng, &mut stage);
            let l1 = stage.get_f64("l1_ratio", f64::NAN);
            assert!(
                l1 > 0.0 && l1 < 1.0,
                "l1_ratio left the unit interval: {l1}"
            );
        }
    }

    /// The frozen v1 operator is byte-untouched: pinned (seed -> output sha)
    /// pairs captured before the v2 twins landed.
    #[test]
    fn v1_operators_are_byte_untouched() {
        let grid = [
            (
                "(pipe X (standardize) (ridge))",
                0u64,
                "beeee67c8c273bb5fc87168752e6cc2df381904f498d42f49b30b8df10037e02",
            ),
            (
                "(pipe X (standardize) (ridge))",
                1,
                "beeee67c8c273bb5fc87168752e6cc2df381904f498d42f49b30b8df10037e02",
            ),
            (
                "(pipe X (standardize) (ridge))",
                2,
                "9fc5a1009ce23caacd87cfa7187fd22c97c6888fcd5ed2319a16554a3e28e01a",
            ),
            (
                "(pipe X (rank append=true) (winsor_clip q=0.1) (histgbm depth=4 lr=0.1 rounds=80))",
                3,
                "8caf69b6494afdb45e1f4077f95617060f3e381b95efc1ce84ce2a6ff6ca45b5",
            ),
            (
                "(pipe X (rank append=true) (winsor_clip q=0.1) (histgbm depth=4 lr=0.1 rounds=80))",
                4,
                "8caf69b6494afdb45e1f4077f95617060f3e381b95efc1ce84ce2a6ff6ca45b5",
            ),
            (
                "(pipe X (rank append=true) (winsor_clip q=0.1) (histgbm depth=4 lr=0.1 rounds=80))",
                5,
                "79a576498aaf6364a2e13d7bd026fe37fd4441b29a8e1a807a730600d9f252b3",
            ),
        ];
        for (source, seed, expected) in grid {
            let out = mutate_canonical_pipeline(source, seed, 6).unwrap();
            let mut hasher = Sha256::new();
            hasher.update(out.as_bytes());
            let sha: String = hasher
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(
                sha, expected,
                "frozen v1 mutation drifted for {source:?} seed={seed}: {out}"
            );
        }
    }

    /// Generated expansion-op params land inside their spec clamps.
    #[test]
    fn expansion_params_are_within_clamps() {
        let cfg = EvolveConfig {
            expansion_vocab: true,
            ..EvolveConfig::default()
        };
        let vocab = generation_vocab(&cfg);
        let mut rng = Pcg32::new(0xC1AB_44F5, 8);
        for _ in 0..400 {
            let stage = rand_stage_from(&mut rng, &vocab);
            match stage.op.as_str() {
                "symlog" => {
                    let c = stage.get_f64("c", 1.0);
                    assert!((1e-9..=1e9).contains(&c), "symlog c out of clamp: {c}");
                }
                "select_corr" => {
                    let k = stage.get_usize("k", 32);
                    assert!((1..=512).contains(&k), "select_corr k out of clamp: {k}");
                }
                "pair_ratio" => {
                    let topk = stage.get_usize("topk", 8);
                    assert!(
                        (1..=64).contains(&topk),
                        "pair_ratio topk out of clamp: {topk}"
                    );
                }
                "spline" => {
                    let knots = stage.get_usize("knots", 5);
                    assert!(
                        (3..=12).contains(&knots),
                        "spline knots out of clamp: {knots}"
                    );
                }
                _ => {}
            }
        }
    }

    // ------------------------------ fault policy ------------------------------

    fn pipe(source: &str) -> DslExpr {
        parse_pipe(source).unwrap_or_else(|| panic!("not a pipeline: {source}"))
    }

    fn non_legacy_policies() -> [FaultPolicy; 2] {
        [
            FaultPolicy::RejectHard,
            FaultPolicy::Penalize {
                lambda_domain: 0.05,
                lambda_saturation: 0.01,
                tie_break: true,
            },
        ]
    }

    /// A fit that fails the test when it is called.
    fn fit_never(
        _: &GenomeIr,
        _: &Rows,
        _: &[f64],
    ) -> anyhow::Result<(model::Model, PipelineTelemetry)> {
        panic!("a program the scope check rejected was fitted")
    }

    /// Records every rejection a fault policy reports.
    #[derive(Default)]
    struct RejectLog {
        rejections: std::sync::Mutex<Vec<(String, String)>>,
    }
    impl Observer for RejectLog {
        fn on_generation(&self, _s: &GenStats) {}
        fn on_improvement(&self, _best: &Scored, _emitted: &Emitted, _s: &GenStats) {}
        fn on_reject(&self, dsl: &str, reason: &Rejection) {
            self.rejections
                .lock()
                .unwrap()
                .push((dsl.to_string(), reason.to_string()));
        }
    }

    /// The scope check comes before anything is fitted: the fit handed to the scorers fails the
    /// test when it is called. It checks the program as the search fits it, which the search's
    /// caps can narrow. A program in scope is fitted once per fold through that same parameter,
    /// so the check stands on the path of every fit.
    #[test]
    fn scope_rejection_comes_before_any_fit() {
        let (x, y) = race_fixture(7, 96);
        let metric = |yt: &[f64], yp: &[f64]| crate::metrics::r2(yt, yp);
        let out_of_scope = pipe("(pipe X (derive expr=\"col(12)\") (ridge))");
        let assign = fold_assign(y.len(), 3, "invent");
        let rows: Vec<usize> = (0..y.len()).collect();
        let rung_assign = fold_assign(rows.len(), 2, "invent_r0");
        let scope_error = |admitted: Admitted| match admitted {
            Admitted::Rejected(Rejection::Scope(error)) => error,
            _ => panic!("expected a scope rejection"),
        };
        for policy in non_legacy_policies() {
            let error = scope_error(score_genome_admitted(
                &out_of_scope,
                &x,
                &y,
                &assign,
                3,
                &metric,
                policy,
                &fit_never,
            ));
            assert_eq!(
                error.to_string(),
                "stage 0: COLUMN_OOB: column index 12 out of bounds for width 4"
            );
            for rung in 0..3 {
                let error = scope_error(score_rung_admitted(
                    &out_of_scope,
                    rung,
                    &x,
                    &y,
                    &rows,
                    &rung_assign,
                    2,
                    &metric,
                    Instant::now(),
                    0.0,
                    &quiet_obs(),
                    policy,
                    &fit_never,
                ));
                assert_eq!(error.class, Some(crate::math_expr::FaultClass::ColumnOob));
            }
        }
        // Legacy fits the same program and scores the fallback zeros of `col(12)`.
        assert!(score_genome(&out_of_scope, &x, &y, &assign, 3, &metric).is_some());

        // The check sees the program as it is fitted, not as it was generated. The search caps
        // `rff_cos n=128` at 64 columns before fitting, so `col(100)`, in scope for the generated
        // program, is out of scope for the fitted one at every rung, and the fit that panics is
        // not called. A check of the generated program would pass it and fit it.
        let narrowed = pipe("(pipe X (rff_cos n=128) (derive expr=\"col(100)\") (ridge))");
        assert!(validate_scope(&GenomeIr::Pipeline(narrowed.clone()), 4).is_ok());
        let narrowed_reason = "stage 1: COLUMN_OOB: column index 100 out of bounds for width 64";
        for policy in non_legacy_policies() {
            let error = scope_error(score_genome_admitted(
                &narrowed, &x, &y, &assign, 3, &metric, policy, &fit_never,
            ));
            assert_eq!(error.to_string(), narrowed_reason);
            for rung in 0..3 {
                let error = scope_error(score_rung_admitted(
                    &narrowed,
                    rung,
                    &x,
                    &y,
                    &rows,
                    &rung_assign,
                    2,
                    &metric,
                    Instant::now(),
                    0.0,
                    &quiet_obs(),
                    policy,
                    &fit_never,
                ));
                assert_eq!(error.to_string(), narrowed_reason);
            }
        }

        let fits = AtomicUsize::new(0);
        let counting = |program: &GenomeIr, x: &Rows, y: &[f64]| {
            fits.fetch_add(1, Ordering::Relaxed);
            fit_traced(program, x, y)
        };
        let in_scope = pipe("(pipe X (derive expr=\"col(3)\") (ridge))");
        for policy in non_legacy_policies() {
            fits.store(0, Ordering::Relaxed);
            assert!(matches!(
                score_genome_admitted(&in_scope, &x, &y, &assign, 3, &metric, policy, &counting),
                Admitted::Scored(..)
            ));
            assert_eq!(fits.load(Ordering::Relaxed), 3);
            fits.store(0, Ordering::Relaxed);
            assert!(matches!(
                score_rung_admitted(
                    &in_scope,
                    1,
                    &x,
                    &y,
                    &rows,
                    &rung_assign,
                    2,
                    &metric,
                    Instant::now(),
                    0.0,
                    &quiet_obs(),
                    policy,
                    &counting,
                ),
                Admitted::Scored(..)
            ));
            assert_eq!(fits.load(Ordering::Relaxed), 2);
        }
    }

    /// Data on which a `nan_bounds_at_rung_2` program meets a Hard fault at rung 2 only. Rungs 0
    /// and 1 race on the first `RACE_RUNG_ROWS[0]` rows of the race's seeded permutation, where
    /// column 1 is 0; it is 1 on the rows that rung 2 adds.
    fn rung_2_fault_data(n: usize, seed: u64) -> (Rows, Vec<f64>) {
        let mut rng = Pcg32::new(3, 7);
        let mut x: Rows = (0..n)
            .map(|_| (0..4).map(|_| rng.gaussian()).collect::<Vec<f64>>())
            .collect();
        for (position, &row) in permute_indices(n, seed).iter().enumerate() {
            x[row][1] = if position < RACE_RUNG_ROWS[0] {
                0.0
            } else {
                1.0
            };
        }
        let y = x.iter().map(|r| 2.0 * r[0] - r[2] + 0.5 * r[3]).collect();
        (x, y)
    }

    /// Clips `column` between bounds that are 0 where column 1 is 0, and NaN (inf * 0) where it
    /// is 1: a Hard fault, `INVALID_CLIP_BOUNDS`, on those rows only.
    fn nan_bounds_at_rung_2(column: usize) -> String {
        format!(
            "(pipe X (derive expr=\"clip(col({column}), col(1)*1e300*1e300*0, \
             col(1)*1e300*1e300*0)\") (ridge))"
        )
    }

    /// A Hard fault that only a later rung meets removes the program from the ranking, although
    /// its earlier rungs scored it. Legacy ranks the same program by its rung-2 score, fallback
    /// zeros included.
    #[test]
    fn a_hard_fault_at_a_later_rung_removes_the_program_from_the_ranking() {
        let mut cfg = race_cfg(5);
        cfg.pop = 1;
        cfg.gens = 1;
        let (x, y) = rung_2_fault_data(256, cfg.seed);
        let metric = |yt: &[f64], yp: &[f64]| crate::metrics::r2(yt, yp);
        let program = nan_bounds_at_rung_2(0);
        let initial = [program.clone()];
        let legacy = evolve_lite_race_with_exact_initial_population(
            &x,
            &y,
            &metric,
            &cfg,
            &quiet_obs(),
            &initial,
        )
        .unwrap();
        cfg.fault_policy = FaultPolicy::RejectHard;
        let log = RejectLog::default();
        let admitted =
            evolve_lite_race_with_exact_initial_population(&x, &y, &metric, &cfg, &log, &initial)
                .unwrap();

        let key = GenomeIr::parse(&program).unwrap().canonical();
        assert_eq!(
            legacy
                .ranked
                .iter()
                .map(|s| s.dsl.as_str())
                .collect::<Vec<_>>(),
            vec![key.as_str()]
        );
        assert!(legacy.candidates[0]
            .rung_fold_scores
            .iter()
            .all(Option::is_some));
        assert!(admitted.ranked.is_empty());
        let [r0, r1, r2] = &admitted.candidates[0].rung_fold_scores;
        assert_eq!(r0, &legacy.candidates[0].rung_fold_scores[0]);
        assert_eq!(r1, &legacy.candidates[0].rung_fold_scores[1]);
        assert!(r2.is_none());
        assert_eq!(
            log.rejections.into_inner().unwrap(),
            vec![(key, "hard fault: {INVALID_CLIP_BOUNDS}".to_string())]
        );
    }

    /// When the elites' rung-2 re-check rejects every candidate still ranked, the generation ends
    /// in the reseed that an empty ranking has always taken, and the next generation ranks the
    /// reseeded candidates. Of two such programs, one is promoted to rung 2 and rejected there,
    /// and the other is rejected when the re-check reaches it as the only ranked candidate.
    #[test]
    fn a_ranking_emptied_by_rung_2_rejections_reseeds() {
        let mut cfg = race_cfg(5);
        cfg.pop = 2;
        cfg.gens = 2;
        cfg.fault_policy = FaultPolicy::RejectHard;
        let (x, y) = rung_2_fault_data(256, cfg.seed);
        let metric = |yt: &[f64], yp: &[f64]| crate::metrics::r2(yt, yp);
        let initial = [nan_bounds_at_rung_2(0), nan_bounds_at_rung_2(2)];
        let log = RejectLog::default();
        let outcome =
            evolve_lite_race_with_exact_initial_population(&x, &y, &metric, &cfg, &log, &initial)
                .unwrap();
        let rejections = log.rejections.into_inner().unwrap();
        assert_eq!(rejections.len(), 2);
        for source in &initial {
            let key = GenomeIr::parse(source).unwrap().canonical();
            assert!(rejections.contains(&(key.clone(), "hard fault: {INVALID_CLIP_BOUNDS}".into())));
            let candidate = outcome.candidates.iter().find(|c| c.dsl == key).unwrap();
            assert!(candidate.rung_fold_scores[0].is_some());
            assert!(candidate.rung_fold_scores[2].is_none());
            assert!(outcome.ranked.iter().all(|scored| scored.dsl != key));
        }
        assert!(
            !outcome.ranked.is_empty(),
            "the reseeded generation is ranked"
        );
        assert!(outcome
            .candidates
            .iter()
            .any(|candidate| candidate.r0_generation == Some(1)));
    }

    /// Records each generation's domain-rate counters, as bits.
    #[derive(Default)]
    struct RateLog {
        generations: std::sync::Mutex<Vec<(u64, u64)>>,
    }
    impl Observer for RateLog {
        fn on_generation(&self, s: &GenStats) {
            self.generations
                .lock()
                .unwrap()
                .push((s.pop_domain_rate.to_bits(), s.elite_domain_rate.to_bits()));
        }
        fn on_improvement(&self, _best: &Scored, _emitted: &Emitted, _s: &GenStats) {}
    }

    /// The domain-rate counters take each ranked candidate's rate from the evaluation its score
    /// comes from, in the race its highest scored rung. On `rung_2_fault_data` the program's `log`
    /// argument is at least 0.5 on the rows of rungs 0 and 1, where column 1 is 0, and negative on
    /// about half of the rows that rung 2 adds, where it is 1: its rung-0 rate is 0, and both
    /// counters are its rung-2 rate.
    #[test]
    fn the_domain_rate_counters_read_the_rung_of_the_ranking_score() {
        let mut cfg = race_cfg(5);
        cfg.pop = 1;
        cfg.gens = 1;
        let (x, y) = rung_2_fault_data(256, cfg.seed);
        let metric = |yt: &[f64], yp: &[f64]| crate::metrics::r2(yt, yp);
        let source = "(pipe X (derive expr=\"log(col(0)*col(0) + 0.5 - col(1))\") (ridge))";
        let program = pipe(source);
        let perm = permute_indices(x.len(), cfg.seed);
        for policy in non_legacy_policies() {
            // One rung's evaluation, with the rows, folds and scorer the race gives it.
            let totals = |rung: usize| {
                let rows = perm[..x.len().min(RACE_RUNG_ROWS[rung])].to_vec();
                let key = ["invent_r0", "invent_r1", "invent_r2"][rung];
                let assign = fold_assign(rows.len(), RACE_RUNG_FOLDS[rung], key);
                match score_rung_admitted(
                    &program,
                    rung,
                    &x,
                    &y,
                    &rows,
                    &assign,
                    RACE_RUNG_FOLDS[rung],
                    &metric,
                    Instant::now(),
                    0.0,
                    &quiet_obs(),
                    policy,
                    &fit_traced,
                ) {
                    Admitted::Scored(_, totals) => totals,
                    _ => panic!("rung {rung} did not score"),
                }
            };
            let (rung_0, rung_2) = (totals(0), totals(2));
            assert_eq!(rung_0.domain_cells, 0);
            assert!(
                rung_2.domain_cells > 0 && rung_2.domain_rate() < 1.0,
                "{rung_2:?}"
            );

            cfg.fault_policy = policy;
            let log = RateLog::default();
            let outcome = evolve_lite_race_with_exact_initial_population(
                &x,
                &y,
                &metric,
                &cfg,
                &log,
                &[source.to_string()],
            )
            .unwrap();
            assert!(outcome.candidates[0]
                .rung_fold_scores
                .iter()
                .all(Option::is_some));
            assert_eq!(outcome.ranked.len(), 1);
            let rate = rung_2.domain_rate().to_bits();
            assert_eq!(log.generations.into_inner().unwrap(), vec![(rate, rate)]);
        }
    }

    /// The tie-breaking selection draws exactly the numbers the legacy selection draws. It picks
    /// a different parent only inside a near-tie, and then the one with fewer faulted cells.
    #[test]
    fn tie_breaking_selection_draws_the_legacy_numbers() {
        let ranked: Vec<Scored> = [0.9, 0.5, 0.5, 0.5 - 5e-13, 0.1]
            .iter()
            .enumerate()
            .map(|(i, &cv)| Scored {
                dsl: format!("candidate {i}"),
                cv_mean: cv,
                fold_scores: vec![cv],
            })
            .collect();
        let equal = vec![0_u64; ranked.len()];
        let cells = vec![0_u64, 4, 1, 0, 0];
        let mut differed = 0;
        for seed in 0..300 {
            let (mut a, mut b) = (Pcg32::new(seed, 1), Pcg32::new(seed, 1));
            assert_eq!(
                tournament(&mut a, &ranked, 5),
                tournament_admitted(&mut b, &ranked, 5, &equal)
            );
            assert_eq!(a.next_u32(), b.next_u32(), "the draws must not change");

            let (mut a, mut b) = (Pcg32::new(seed, 1), Pcg32::new(seed, 1));
            let legacy = tournament(&mut a, &ranked, 5);
            let admitted = tournament_admitted(&mut b, &ranked, 5, &cells);
            assert_eq!(a.next_u32(), b.next_u32(), "the draws must not change");
            if legacy != admitted {
                differed += 1;
                assert!((1..=3).contains(&legacy) && (1..=3).contains(&admitted));
                assert!(cells[admitted] < cells[legacy]);
            }
        }
        assert!(differed > 0, "the near-tie never decided a selection");
    }
}

// B10 lexicase: begin
#[cfg(test)]
mod lexicase_selection_tests {
    use super::*;
    use crate::kernel::lexicase::{CaseSampling, EpsilonRule};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    const MAD: ParentSelection = ParentSelection::EpsilonLexicase(LexicaseConfig {
        epsilon: EpsilonRule::MedianAbsoluteDeviation,
        sampling: CaseSampling::All,
    });

    #[derive(Default)]
    struct Receipts {
        selections: Mutex<Vec<SelectionReceipt>>,
        first_generation: Mutex<Option<Vec<u64>>>,
    }

    impl Observer for Receipts {
        fn on_generation(&self, stats: &GenStats) {
            let mut first = self.first_generation.lock().unwrap();
            if first.is_none() {
                *first = Some(stats.pop_fitness.iter().map(|v| v.to_bits()).collect());
            }
        }
        fn on_improvement(&self, _best: &Scored, _emitted: &Emitted, _stats: &GenStats) {}
        fn on_selection(&self, receipt: &SelectionReceipt) {
            self.selections.lock().unwrap().push(receipt.clone());
        }
    }

    /// Two regimes: rows with `x2 > 0` follow `3 x0`, the others `-2 x1`, plus noise.
    fn regimes(seed: u64, n: usize) -> (Rows, Vec<f64>) {
        let mut rng = Pcg32::new(seed, 9);
        let x: Rows = (0..n)
            .map(|_| (0..3).map(|_| rng.gaussian()).collect())
            .collect();
        let y = x
            .iter()
            .map(|r| {
                let signal = if r[2] > 0.0 { 3.0 * r[0] } else { -2.0 * r[1] };
                signal + 0.1 * rng.gaussian()
            })
            .collect();
        (x, y)
    }

    fn config(fitness: FitnessMode, selection: ParentSelection) -> EvolveConfig {
        EvolveConfig {
            pop: 12,
            gens: 3,
            folds: 3,
            seed: 0xB10,
            max_transforms: 3,
            fitness,
            selection,
            ..EvolveConfig::default()
        }
    }

    fn r2(truth: &[f64], predicted: &[f64]) -> f64 {
        crate::metrics::r2(truth, predicted)
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    fn dump(ranked: &[Scored]) -> Vec<(String, u64, Vec<u64>)> {
        ranked
            .iter()
            .map(|s| (s.dsl.clone(), s.cv_mean.to_bits(), bits(&s.fold_scores)))
            .collect()
    }

    fn ranked(scores: &[f64]) -> Vec<Scored> {
        scores
            .iter()
            .enumerate()
            .map(|(i, &cv)| Scored {
                dsl: format!("c{i}"),
                cv_mean: cv,
                fold_scores: vec![cv],
            })
            .collect()
    }

    #[test]
    fn tournament_three_is_the_default_and_reports_nothing() {
        assert_eq!(
            EvolveConfig::default().selection,
            ParentSelection::Tournament3
        );
        assert_eq!(ParentSelection::default(), ParentSelection::Tournament3);
        let (x, y) = regimes(1, 120);
        for fitness in [FitnessMode::Surrogate, FitnessMode::LiteRace] {
            let obs = Receipts::default();
            let ranked = evolve(
                &x,
                &y,
                &r2,
                &config(fitness, ParentSelection::Tournament3),
                &obs,
            );
            assert!(!ranked.is_empty());
            assert!(obs.selections.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn without_lexicase_pick_parent_draws_the_legacy_numbers() {
        let ranked = ranked(&[0.9, 0.5, 0.5, 0.2, 0.1]);
        let cells = vec![0_u64, 4, 1, 0, 0];
        for seed in 0..300 {
            for faulted in [None, Some(cells.as_slice())] {
                for mut draws in [None, Some(Err("fallback".to_string()))] {
                    let (mut a, mut b) = (Pcg32::new(seed, 1), Pcg32::new(seed, 1));
                    assert_eq!(
                        pick_parent(&mut a, &ranked, 3, faulted, &mut draws),
                        select_parent(&mut b, &ranked, 3, faulted)
                    );
                    assert_eq!(a.next_u32(), b.next_u32(), "the draws must not change");
                }
            }
        }
    }

    #[test]
    fn lexicase_draws_come_from_their_own_stream() {
        let ranked = ranked(&[0.9, 0.5, 0.2]);
        let matrix = lexicase_matrix(&ranked, |dsl| match dsl {
            "c0" => Some(&[0.0, 4.0, 4.0][..]),
            "c1" => Some(&[4.0, 0.0, 4.0][..]),
            _ => Some(&[4.0, 4.0, 0.0][..]),
        });
        let mut draws = start_lexicase(Some(LexicaseConfig::default()), Some(&matrix), 5, 2);
        let mut search = Pcg32::new(5, 1);
        let mut picked: Vec<usize> = (0..60)
            .map(|_| pick_parent(&mut search, &ranked, 1, None, &mut draws))
            .collect();
        assert_eq!(search.next_u32(), Pcg32::new(5, 1).next_u32());
        // Each specialist wins its own case, so all three are drawn, not only the best mean.
        picked.sort_unstable();
        picked.dedup();
        assert_eq!(picked, [0, 1, 2]);
        let obs = Receipts::default();
        report_selection(&obs, 2, ranked.len(), draws);
        let receipt = obs.selections.into_inner().unwrap().remove(0);
        assert_eq!((receipt.generation, receipt.pool), (2, 3));
        let selection = receipt.selection.unwrap();
        assert_eq!(selection.parents.len(), 60);
        // The generation's stream is `LEXICASE_STREAM + generation` under the run seed.
        let mut replay = Pcg32::new(5, LEXICASE_STREAM.wrapping_add(2));
        let errors = matrix.as_ref().unwrap();
        let again =
            crate::kernel::lexicase::select(errors, LexicaseConfig::default(), 60, &mut replay)
                .unwrap();
        assert_eq!(selection, again);
    }

    #[test]
    fn case_scores_keep_the_legacy_fold_bits_and_never_see_their_own_labels() {
        let (x, y) = regimes(2, 90);
        let assign = fold_assign(y.len(), 3, "invent");
        let genome =
            parse_pipe("(pipe X (standardize) (cheb degree=3 append=true) (ridge))").unwrap();
        let legacy = score_genome(&genome, &x, &y, &assign, 3, &r2).unwrap();
        let (scores, predictions) =
            score_genome_predictions(&genome, &x, &y, &assign, 3, &r2).unwrap();
        assert_eq!(bits(&legacy), bits(&scores));
        let rows = lexicase_case_rows(y.len(), 0xB10);
        assert_eq!(rows.len(), 90);
        assert!(rows.windows(2).all(|pair| pair[0] < pair[1]));
        let (case_scores, residuals) =
            score_genome_cases(&genome, &x, &y, &assign, 3, &r2, &rows).unwrap();
        assert_eq!(bits(&legacy), bits(&case_scores));
        for (&row, residual) in rows.iter().zip(&residuals) {
            assert_eq!(residual.to_bits(), (y[row] - predictions[row]).to_bits());
        }
        // A fold's held-out predictions come only from models fitted on the other folds.
        for fold in 0..3 {
            let poisoned: Vec<f64> = y
                .iter()
                .zip(&assign)
                .map(|(&v, &f)| if f == fold { v + 1e3 } else { v })
                .collect();
            let (_, after) =
                score_genome_predictions(&genome, &x, &poisoned, &assign, 3, &r2).unwrap();
            for row in (0..y.len()).filter(|&row| assign[row] == fold) {
                assert_eq!(
                    after[row].to_bits(),
                    predictions[row].to_bits(),
                    "row {row}"
                );
            }
        }
        // The case rows are a sorted prefix of the race's permutation, capped.
        assert_eq!(lexicase_case_rows(1000, 3).len(), LEXICASE_CASE_ROWS);
        let mut prefix = permute_indices(1000, 3)[..LEXICASE_CASE_ROWS].to_vec();
        prefix.sort_unstable();
        assert_eq!(lexicase_case_rows(1000, 3), prefix);
    }

    #[test]
    fn race_case_scores_keep_the_legacy_fold_bits() {
        let (x, y) = regimes(3, 300);
        let perm = permute_indices(y.len(), 7);
        let genome = parse_pipe("(pipe X (robust_scale) (histgbm depth=3 rounds=20))").unwrap();
        let obs = Receipts::default();
        for rung in 0..3 {
            let rows = &perm[..y.len().min(RACE_RUNG_ROWS[rung])];
            let k = RACE_RUNG_FOLDS[rung];
            let assign = fold_assign(rows.len(), k, "invent_r");
            let started = Instant::now();
            let legacy = score_rung(
                &genome, rung, &x, &y, rows, &assign, k, &r2, started, 0.0, &obs,
            )
            .unwrap();
            let (scores, residuals) = score_rung_cases(
                &genome, rung, &x, &y, rows, &assign, k, &r2, started, 0.0, &obs, 128,
            )
            .unwrap();
            assert_eq!(bits(&legacy), bits(&scores), "rung {rung}");
            assert_eq!(residuals.len(), 128);
            assert!(residuals.iter().all(|r| r.is_finite()));
        }
    }

    #[test]
    fn lexicase_surrogate_search_replays_and_reports_each_generation() {
        let (x, y) = regimes(4, 160);
        let run = |selection| {
            let obs = Receipts::default();
            let ranked = evolve(
                &x,
                &y,
                &r2,
                &config(FitnessMode::Surrogate, selection),
                &obs,
            );
            (ranked, obs)
        };
        let (a, obs_a) = run(MAD);
        let (b, obs_b) = run(MAD);
        let (_, tournament) = run(ParentSelection::Tournament3);
        assert_eq!(dump(&a), dump(&b));
        // Generation 0 is scored before any parent is drawn: its ranking is tournament-3's.
        assert_eq!(
            *obs_a.first_generation.lock().unwrap(),
            *tournament.first_generation.lock().unwrap()
        );
        let receipts = obs_a.selections.into_inner().unwrap();
        assert_eq!(receipts, obs_b.selections.into_inner().unwrap());
        assert_eq!(receipts.len(), 3);
        for (generation, receipt) in receipts.iter().enumerate() {
            assert_eq!(receipt.generation, generation);
            let selection = receipt.selection.as_ref().unwrap();
            // One event per mutated child, two per crossover child: 11 children of 12.
            assert!((11..=22).contains(&selection.parents.len()));
            assert_eq!(selection.trace.events.len(), selection.parents.len());
            assert!(selection.parents.iter().all(|&p| p < receipt.pool));
            assert_eq!(selection.trace.cases.len(), 160);
        }
    }

    #[test]
    fn lexicase_race_scores_on_the_cpu_and_keeps_reuse_off() {
        let (x, y) = regimes(5, 300);
        let called = AtomicUsize::new(0);
        let backend: &GpuRidgeScoreBatchFn = &|_| {
            called.fetch_add(1, Ordering::Relaxed);
            Err("unavailable".to_string())
        };
        let dedup = Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
            RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
            &"a".repeat(64),
        )
        .unwrap();
        let race = |obs: &Receipts| {
            evolve_lite_race_with_gpu_policy_and_exact_surrogate_dedup(
                &x,
                &y,
                &r2,
                &config(FitnessMode::LiteRace, MAD),
                obs,
                Some(backend),
                GpuRidgeScorePolicy::new(1, 1),
                Some(dedup),
            )
        };
        let obs = Receipts::default();
        let outcome = race(&obs);
        assert_eq!(called.load(Ordering::Relaxed), 0, "the GPU backend ran");
        assert!(!outcome.rung0_exact_surrogate_dedup.active);
        assert_eq!(
            outcome
                .rung0_exact_surrogate_dedup
                .activation_blocker
                .as_deref(),
            Some(LEXICASE_REUSE_BLOCKER)
        );
        assert!(outcome
            .candidates
            .iter()
            .filter_map(|c| c.r0_fold_routes.as_ref())
            .flatten()
            .all(|route| *route == Rung0FoldRouteReceipt::cpu(Rung0FoldRoute::CpuExact)));
        let receipts = obs.selections.into_inner().unwrap();
        assert!(!receipts.is_empty());
        for receipt in &receipts {
            let selection = receipt.selection.as_ref().unwrap();
            assert_eq!(selection.trace.cases.len(), RACE_RUNG_ROWS[0]);
            assert!(selection.parents.iter().all(|&p| p < receipt.pool));
        }
        let again = race(&Receipts::default());
        assert_eq!(outcome.candidates, again.candidates);
        assert_eq!(dump(&outcome.ranked), dump(&again.ranked));
        assert_eq!(outcome.rung_evals, again.rung_evals);
    }

    #[test]
    fn lexicase_needs_the_legacy_fault_policy_and_a_valid_rate() {
        let strict = FaultPolicy::RejectHard;
        assert!(MAD
            .validate(strict)
            .unwrap_err()
            .contains("default fault policy"));
        assert!(ParentSelection::Tournament3.validate(strict).is_ok());
        assert!(MAD.validate(FaultPolicy::Legacy).is_ok());
        let rate = ParentSelection::EpsilonLexicase(LexicaseConfig {
            epsilon: EpsilonRule::Zero,
            sampling: CaseSampling::DownSampled { per_mille: 1001 },
        });
        assert!(rate.validate(FaultPolicy::Legacy).is_err());
        let (x, y) = regimes(6, 60);
        let mut cfg = config(FitnessMode::LiteRace, MAD);
        cfg.fault_policy = strict;
        let refused = evolve_lite_race_with_exact_initial_population(
            &x,
            &y,
            &r2,
            &cfg,
            &Receipts::default(),
            &[],
        );
        assert!(refused.unwrap_err().contains("default fault policy"));
    }

    #[test]
    #[should_panic(expected = "epsilon-lexicase needs the default fault policy")]
    fn the_surrogate_search_refuses_lexicase_under_a_strict_fault_policy() {
        let (x, y) = regimes(7, 60);
        let mut cfg = config(FitnessMode::Surrogate, MAD);
        cfg.fault_policy = FaultPolicy::RejectHard;
        evolve(&x, &y, &r2, &cfg, &Receipts::default());
    }

    #[test]
    fn a_pool_without_one_case_matrix_falls_back_to_tournament_three() {
        let ranked = ranked(&[0.9, 0.5, 0.5, 0.2, 0.1]);
        let missing = lexicase_matrix(&ranked, |dsl| (dsl != "c2").then_some(&[1.0, 2.0][..]));
        assert!(missing.unwrap_err().contains("no residual cases"));
        let uneven = lexicase_matrix(&ranked, |dsl| {
            Some(if dsl == "c2" {
                &[1.0][..]
            } else {
                &[1.0, 2.0][..]
            })
        });
        assert!(uneven.unwrap_err().contains("different case counts"));
        let matrix = Err("no cases".to_string());
        let mut draws = start_lexicase(Some(LexicaseConfig::default()), Some(&matrix), 1, 0);
        assert!(start_lexicase(None, Some(&matrix), 1, 0).is_none());
        for seed in 0..50 {
            let (mut a, mut b) = (Pcg32::new(seed, 1), Pcg32::new(seed, 1));
            assert_eq!(
                pick_parent(&mut a, &ranked, 3, None, &mut draws),
                select_parent(&mut b, &ranked, 3, None)
            );
            assert_eq!(a.next_u32(), b.next_u32());
        }
        let obs = Receipts::default();
        report_selection(&obs, 4, ranked.len(), draws);
        let receipts = obs.selections.into_inner().unwrap();
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].selection, Err("no cases".to_string()));
    }

    /// One synthetic regression task: rows of five gaussian features, labelled by `label`.
    fn task(name: &str, seed: u64, n: usize) -> (Rows, Vec<f64>) {
        let mut rng = Pcg32::new(seed, 0x0B10);
        let x: Rows = (0..n)
            .map(|_| (0..5).map(|_| rng.gaussian()).collect())
            .collect();
        let y = x
            .iter()
            .map(|r| {
                let noise = rng.gaussian();
                match name {
                    "regimes" => (if r[2] > 0.0 { 3.0 * r[0] } else { -2.0 * r[1] }) + 0.1 * noise,
                    "linear" => 3.0 * r[0] - 2.0 * r[1] + 0.2 * noise,
                    "interaction" => r[0] * r[1] + 0.5 * r[2] + 0.1 * noise,
                    "periodic" => (2.0 * r[0]).sin() + 0.5 * r[1] + 0.1 * noise,
                    "outliers" => {
                        let shift = if noise > 1.8 { 8.0 } else { 0.0 };
                        r[0] + 0.5 * r[1] + shift + 0.1 * noise
                    }
                    _ => r[0] + r[1].abs() * 0.5 * noise,
                }
            })
            .collect();
        (x, y)
    }

    /// Six tasks x five seeds at an equal generation budget: the same population size, generation
    /// count, folds, data and search seed for both arms. Each winner's surrogate is refit on the
    /// task's training rows and scored on held-out rows the search never saw. About six minutes
    /// in a debug build, so it is evidence, not a PR-tier test: `cargo test -p feat-math --release
    /// --lib six_tasks_five_seeds -- --ignored --nocapture`. Floating-point results do not depend
    /// on the build profile.
    #[test]
    #[ignore]
    fn six_tasks_five_seeds_equal_budget_selection_ablation() {
        const TASKS: [&str; 6] = [
            "regimes",
            "linear",
            "interaction",
            "periodic",
            "outliers",
            "heteroscedastic",
        ];
        const SEEDS: [u64; 5] = [11, 29, 47, 71, 101];
        let heldout = |dsl: &str, x: &Rows, y: &[f64], xt: &Rows, yt: &[f64]| {
            let genome = GenomeIr::Pipeline(search_surrogate(&parse_pipe(dsl).unwrap()));
            let model = model::fit(&genome, x, y).unwrap();
            r2(yt, &model.predict(xt).unwrap())
        };
        let median = |values: &mut Vec<f64>| {
            values.sort_by(f64::total_cmp);
            let n = values.len();
            (values[(n - 1) / 2] + values[n / 2]) / 2.0
        };
        let (mut base, mut lexi, mut diversity) = (Vec::new(), Vec::new(), Vec::new());
        let (mut base_distinct, mut lexi_distinct) = (0usize, 0usize);
        let mut paired = [0usize; 3]; // lexicase better, equal, worse
        let mut rows = Vec::new();
        for name in TASKS {
            for seed in SEEDS {
                let (x, y) = task(name, seed, 280);
                let (train, test) = (x[..160].to_vec(), x[160..].to_vec());
                let (ytrain, ytest) = (y[..160].to_vec(), y[160..].to_vec());
                let score = |selection| {
                    let cfg = EvolveConfig {
                        pop: 16,
                        gens: 5,
                        seed,
                        ..config(FitnessMode::Surrogate, selection)
                    };
                    let obs = Receipts::default();
                    let ranked = evolve(&train, &ytrain, &r2, &cfg, &obs);
                    let fraction = obs
                        .selections
                        .into_inner()
                        .unwrap()
                        .iter()
                        .filter_map(|r| r.selection.as_ref().ok())
                        .map(|s| crate::kernel::lexicase::distinct_parent_fraction(&s.parents))
                        .sum::<f64>();
                    let held = heldout(&ranked[0].dsl, &train, &ytrain, &test, &ytest);
                    (held, ranked.len(), fraction / cfg.gens as f64)
                };
                let (tournament, tournament_distinct, _) = score(ParentSelection::Tournament3);
                let (lexicase, lexicase_distinct, fraction) = score(MAD);
                assert!(tournament.is_finite() && lexicase.is_finite());
                base.push(tournament);
                lexi.push(lexicase);
                diversity.push(fraction);
                base_distinct += tournament_distinct;
                lexi_distinct += lexicase_distinct;
                paired[match lexicase.total_cmp(&tournament) {
                    std::cmp::Ordering::Greater => 0,
                    std::cmp::Ordering::Equal => 1,
                    std::cmp::Ordering::Less => 2,
                }] += 1;
                rows.push(format!(
                    "{{\"task\":\"{name}\",\"seed\":{seed},\"tournament_r2\":{tournament},\
                     \"lexicase_r2\":{lexicase},\"tournament_ranked\":{tournament_distinct},\
                     \"lexicase_ranked\":{lexicase_distinct},\
                     \"lexicase_distinct_parent_fraction\":{fraction}}}"
                ));
            }
        }
        let (base_median, lexi_median) = (median(&mut base), median(&mut lexi));
        println!("B10_ABLATION [{}]", rows.join(","));
        println!(
            "B10_ABLATION_SUMMARY {{\"runs\":{},\"tournament_median_r2\":{base_median},\
             \"lexicase_median_r2\":{lexi_median},\"mean_distinct_parent_fraction\":{},\
             \"tournament_mean_final_distinct\":{},\"lexicase_mean_final_distinct\":{},\
             \"paired_lexicase_better_equal_worse\":{paired:?}}}",
            rows.len(),
            diversity.iter().sum::<f64>() / diversity.len() as f64,
            base_distinct as f64 / rows.len() as f64,
            lexi_distinct as f64 / rows.len() as f64,
        );
        assert_eq!(rows.len(), 30);
        // The card's acceptance at this equal budget: median held-out R² no worse, and a more
        // diverse surviving population (distinct ranked genomes at the end of the search).
        assert!(lexi_median >= base_median, "{lexi_median} < {base_median}");
        assert!(lexi_distinct > base_distinct);
    }
}
// B10 lexicase: end

// B11 ridge scoring: begin
#[cfg(test)]
mod ridge_scoring_tests {
    use super::*;
    use crate::kernel::lexicase::{CaseSampling, EpsilonRule};
    use std::sync::Mutex;

    #[derive(Default)]
    struct Log {
        scoring: Mutex<Vec<ScoringReceipt>>,
        selection: Mutex<Vec<SelectionReceipt>>,
    }

    impl Observer for Log {
        fn on_generation(&self, _stats: &GenStats) {}
        fn on_improvement(&self, _best: &Scored, _emitted: &Emitted, _stats: &GenStats) {}
        fn on_selection(&self, receipt: &SelectionReceipt) {
            self.selection.lock().unwrap().push(receipt.clone());
        }
        fn on_scoring(&self, receipt: &ScoringReceipt) {
            self.scoring.lock().unwrap().push(receipt.clone());
        }
    }

    fn fixture(seed: u64, n: usize) -> (Rows, Vec<f64>) {
        let mut rng = Pcg32::new(seed, 21);
        let x: Rows = (0..n)
            .map(|_| (0..4).map(|_| rng.gaussian()).collect())
            .collect();
        let y = x
            .iter()
            .map(|r| 2.0 * r[0] - r[1] * r[1] + (2.0 * r[2]).sin() + 0.1 * rng.gaussian())
            .collect();
        (x, y)
    }

    fn r2(truth: &[f64], predicted: &[f64]) -> f64 {
        crate::metrics::r2(truth, predicted)
    }

    fn dump(ranked: &[Scored]) -> Vec<(String, u64, Vec<u64>)> {
        ranked
            .iter()
            .map(|s| {
                let folds = s.fold_scores.iter().map(|v| v.to_bits()).collect();
                (s.dsl.clone(), s.cv_mean.to_bits(), folds)
            })
            .collect()
    }

    fn run(scoring: RidgeScoring, selection: ParentSelection) -> (Vec<Scored>, Log) {
        let (x, y) = fixture(3, 180);
        let cfg = EvolveConfig {
            pop: 16,
            gens: 4,
            seed: 0xB11,
            max_transforms: 4,
            selection,
            ridge_scoring: scoring,
            ..EvolveConfig::default()
        };
        let log = Log::default();
        let ranked = evolve(&x, &y, &r2, &cfg, &log);
        (ranked, log)
    }

    #[test]
    fn certified_scoring_is_the_legacy_search_bit_for_bit() {
        let lexicase = ParentSelection::EpsilonLexicase(LexicaseConfig {
            epsilon: EpsilonRule::MedianAbsoluteDeviation,
            sampling: CaseSampling::All,
        });
        for selection in [ParentSelection::Tournament3, lexicase] {
            let (legacy, legacy_log) = run(RidgeScoring::Refit, selection);
            let (certified, certified_log) = run(RidgeScoring::Certified, selection);
            assert_eq!(dump(&legacy), dump(&certified), "{selection:?}");
            // Lexicase cases from the memo are the legacy residuals, so selection is unchanged.
            assert_eq!(
                *legacy_log.selection.lock().unwrap(),
                *certified_log.selection.lock().unwrap()
            );
            assert!(legacy_log.scoring.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn certified_scoring_reports_reuse_and_stage_cache_counters() {
        let (_, log) = run(RidgeScoring::Certified, ParentSelection::Tournament3);
        let receipts = log.scoring.into_inner().unwrap();
        assert_eq!(receipts.len(), 4);
        for (generation, pair) in receipts.windows(2).enumerate() {
            assert_eq!(pair[0].generation, generation);
            assert!(pair[1].candidates >= pair[0].candidates);
            assert!(pair[1].surrogate_reuses >= pair[0].surrogate_reuses);
        }
        let last = receipts.last().unwrap();
        assert!(last.surrogate_reuses <= last.candidates);
        // Head-only variation and shared prefixes are common in a short search.
        assert!(last.surrogate_reuses > 0, "{last:?}");
        assert!(last.stage_hits > 0, "{last:?}");
    }

    #[test]
    fn a_different_genome_with_the_same_surrogate_reuses_its_scores_and_cases() {
        let (x, y) = fixture(5, 120);
        let k = 3;
        let assign: Vec<usize> = (0..y.len()).map(|i| i % k).collect();
        let parse = |text: &str| match GenomeIr::parse(text).unwrap() {
            GenomeIr::Pipeline(genome) => genome,
            other => panic!("{other:?}"),
        };
        // Different heads, so different genome keys, but one ridge surrogate.
        let a = parse("(pipe X (standardize) (poly_interact append=true) (gbm))");
        let b = parse("(pipe X (standardize) (poly_interact append=true) (elasticnet))");
        assert_ne!(a.to_string(), b.to_string());
        assert_eq!(
            search_surrogate(&a).to_string(),
            search_surrogate(&b).to_string()
        );
        let rows = lexicase_case_rows(y.len(), 7);
        let mut scoring = CertifiedScoring::new();
        let mut cases = HashMap::new();
        let first = scoring.score_unseen(
            vec![(a.to_string(), a.clone())],
            &x,
            &y,
            &assign,
            k,
            &r2,
            Some(&rows),
            &mut cases,
        );
        let second = scoring.score_unseen(
            vec![(b.to_string(), b.clone())],
            &x,
            &y,
            &assign,
            k,
            &r2,
            Some(&rows),
            &mut cases,
        );
        assert_eq!(
            (scoring.candidates, scoring.reuses, scoring.memo.len()),
            (2, 1, 1)
        );
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let (legacy_scores, legacy_cases) =
            score_genome_cases(&b, &x, &y, &assign, k, &r2, &rows).unwrap();
        assert_eq!(bits(second[0].2.as_ref().unwrap()), bits(&legacy_scores));
        assert_eq!(bits(first[0].2.as_ref().unwrap()), bits(&legacy_scores));
        assert_eq!(bits(&cases[&b.to_string()]), bits(&legacy_cases));
        assert_eq!(bits(&cases[&a.to_string()]), bits(&legacy_cases));
    }

    #[test]
    fn surrogates_that_print_alike_get_distinct_entries_and_their_own_scores() {
        let (x, y) = fixture(5, 120);
        let k = 3;
        let assign: Vec<usize> = (0..y.len()).map(|i| i % k).collect();
        let parse = |text: &str| match GenomeIr::parse(text).unwrap() {
            GenomeIr::Pipeline(genome) => genome,
            other => panic!("{other:?}"),
        };
        // Different heads, so the search's genome memo keeps them apart in both modes. Their
        // surrogates differ only in gamma, 1.0 against 1.0 + 5e-13, which prints as 1.0: the same
        // canonical text for two different computations.
        let a = parse("(pipe X (standardize) (rff_cos gamma=1.0 n=8 seed=3 append=true) (gbm))");
        let mut b =
            parse("(pipe X (standardize) (rff_cos gamma=1.0 n=8 seed=3 append=true) (elasticnet))");
        b.stages[1]
            .params
            .insert("gamma".to_string(), ParamValue::Float(1.0 + 5e-13));
        assert_ne!(1.0f64.to_bits(), (1.0f64 + 5e-13).to_bits());
        assert_ne!(a.to_string(), b.to_string());
        let (sa, sb) = (search_surrogate(&a), search_surrogate(&b));
        assert_eq!(sa.to_string(), sb.to_string());
        assert_ne!(exact_surrogate_key(&sa), exact_surrogate_key(&sb));
        let rows = lexicase_case_rows(y.len(), 7);
        let mut scoring = CertifiedScoring::new();
        let mut cases = HashMap::new();
        let mut score = |g: &DslExpr| {
            let scored = scoring.score_unseen(
                vec![(g.to_string(), g.clone())],
                &x,
                &y,
                &assign,
                k,
                &r2,
                Some(&rows),
                &mut cases,
            );
            let (_, misses, _) = scoring.cache.stats();
            (scored[0].2.clone().unwrap(), misses)
        };
        let (first, misses_a) = score(&a);
        let (second, misses_b) = score(&b);
        // Two memo entries and no reuse; the shared standardize stage hits on every fold, while
        // b's rff_cos block is computed afresh on every fold instead of reusing a's.
        assert_eq!(
            (scoring.candidates, scoring.reuses, scoring.memo.len()),
            (2, 0, 2)
        );
        assert!(misses_b >= misses_a + k as u64, "{misses_a} -> {misses_b}");
        // Each candidate gets exactly the score and cases Refit computes for it.
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for (genome, scores) in [(&a, &first), (&b, &second)] {
            let (refit_scores, refit_cases) =
                score_genome_cases(genome, &x, &y, &assign, k, &r2, &rows).unwrap();
            assert_eq!(bits(scores), bits(&refit_scores));
            assert_eq!(bits(&cases[&genome.to_string()]), bits(&refit_cases));
        }
    }

    #[test]
    fn print_alike_surrogates_that_run_different_programs_are_scored_apart_in_one_batch() {
        let (x, y) = fixture(5, 120);
        let k = 3;
        let assign: Vec<usize> = (0..y.len()).map(|i| i % k).collect();
        let parse = |text: &str| match GenomeIr::parse(text).unwrap() {
            GenomeIr::Pipeline(genome) => genome,
            other => panic!("{other:?}"),
        };
        // `k` = 2.0 and 2.0 - 5e-13 both print as 2.0, but `lag` truncates `k` to an integer, so
        // the second surrogate runs lag 1, not lag 2: a different program with a different Refit
        // score. A text key would hand it the first one's score.
        let a = parse("(pipe X (lag k=2.0) (gbm))");
        let mut b = parse("(pipe X (lag k=2.0) (elasticnet))");
        b.stages[0]
            .params
            .insert("k".to_string(), ParamValue::Float(2.0 - 5e-13));
        assert_eq!(
            search_surrogate(&a).to_string(),
            search_surrogate(&b).to_string()
        );
        let rows = lexicase_case_rows(y.len(), 7);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let refit = |g: &DslExpr| score_genome_cases(g, &x, &y, &assign, k, &r2, &rows).unwrap();
        let ((refit_a, _), (refit_b, _)) = (refit(&a), refit(&b));
        assert_ne!(
            bits(&refit_a),
            bits(&refit_b),
            "lag 2 and lag 1 must score apart"
        );
        let mut scoring = CertifiedScoring::new();
        let mut cases = HashMap::new();
        let scored = scoring.score_unseen(
            vec![(a.to_string(), a.clone()), (b.to_string(), b.clone())],
            &x,
            &y,
            &assign,
            k,
            &r2,
            Some(&rows),
            &mut cases,
        );
        assert_eq!(
            (scoring.candidates, scoring.reuses, scoring.memo.len()),
            (2, 0, 2)
        );
        assert_eq!(bits(scored[0].2.as_ref().unwrap()), bits(&refit_a));
        assert_eq!(bits(scored[1].2.as_ref().unwrap()), bits(&refit_b));
    }

    #[test]
    fn certified_search_scores_print_alike_seeds_as_refit_does() {
        // The independent review's probe: both lag stages print as (lag k=2.0), but lag reads k by
        // truncation, so the first seed lags by one column and the second by two. Their whole
        // surrogates differ, so only the stage cache can confuse them; it keys on exact bits.
        let seeds = [
            "(pipe X (lag k=1.9999999999995) (standardize) (ridge))",
            "(pipe X (lag k=2.0) (sigmoid) (ridge))",
        ];
        let (x, y) = fixture(3, 180);
        let run = |scoring: RidgeScoring, threads: usize| {
            let cfg = EvolveConfig {
                pop: 3,
                gens: 1,
                seed: 0xB11,
                max_transforms: 3,
                seed_genomes: seeds.iter().map(|seed| seed.to_string()).collect(),
                ridge_scoring: scoring,
                ..EvolveConfig::default()
            };
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            dump(&pool.install(|| evolve(&x, &y, &r2, &cfg, &Log::default())))
        };
        let refit = run(RidgeScoring::Refit, 1);
        // Both seeds are scored and ranked, each under its canonical text.
        for seed in seeds {
            let canonical = GenomeIr::parse(seed).unwrap().canonical();
            assert!(
                refit.iter().any(|(dsl, _, _)| *dsl == canonical),
                "{canonical}"
            );
        }
        assert_eq!(run(RidgeScoring::Certified, 1), refit);
        for _ in 0..3 {
            assert_eq!(run(RidgeScoring::Certified, 4), refit);
        }
    }

    #[test]
    fn certified_scoring_is_the_same_on_one_thread_and_on_four() {
        let pool = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        };
        let certified = || run(RidgeScoring::Certified, ParentSelection::Tournament3);
        let (one, one_log) = pool(1).install(certified);
        let (four, four_log) = pool(4).install(certified);
        let (legacy, _) =
            pool(4).install(|| run(RidgeScoring::Refit, ParentSelection::Tournament3));
        assert_eq!(dump(&one), dump(&four));
        assert_eq!(dump(&four), dump(&legacy));
        // The memo counters follow the candidate stream alone. The stage-cache counters are
        // instrumentation: two threads building the same prefix both count a miss.
        let memo = |log: Log| {
            log.scoring
                .into_inner()
                .unwrap()
                .iter()
                .map(|r| (r.generation, r.candidates, r.surrogate_reuses))
                .collect::<Vec<_>>()
        };
        assert_eq!(memo(one_log), memo(four_log));
    }

    #[test]
    fn certified_scoring_refuses_the_race_and_strict_policies() {
        let certified = RidgeScoring::Certified;
        assert!(certified
            .validate(FitnessMode::Surrogate, FaultPolicy::Legacy)
            .is_ok());
        assert!(certified
            .validate(FitnessMode::LiteRace, FaultPolicy::Legacy)
            .unwrap_err()
            .contains("surrogate search"));
        assert!(certified
            .validate(FitnessMode::Surrogate, FaultPolicy::RejectHard)
            .unwrap_err()
            .contains("default fault policy"));
        assert!(RidgeScoring::Refit
            .validate(FitnessMode::LiteRace, FaultPolicy::RejectHard)
            .is_ok());
        let (x, y) = fixture(4, 60);
        let cfg = EvolveConfig {
            pop: 4,
            fitness: FitnessMode::LiteRace,
            ridge_scoring: certified,
            ..EvolveConfig::default()
        };
        let refused =
            evolve_lite_race_with_exact_initial_population(&x, &y, &r2, &cfg, &Log::default(), &[]);
        assert!(refused.unwrap_err().contains("surrogate search"));
    }

    #[test]
    #[should_panic(expected = "certified scoring needs the default fault policy")]
    fn the_surrogate_search_panics_on_certified_scoring_under_a_strict_policy() {
        let (x, y) = fixture(4, 60);
        let cfg = EvolveConfig {
            pop: 4,
            gens: 1,
            fault_policy: FaultPolicy::RejectHard,
            ridge_scoring: RidgeScoring::Certified,
            ..EvolveConfig::default()
        };
        evolve(&x, &y, &r2, &cfg, &Log::default());
    }

    #[test]
    #[should_panic(expected = "certified scoring applies to the surrogate search")]
    fn the_race_panics_on_certified_scoring() {
        let (x, y) = fixture(4, 60);
        let cfg = EvolveConfig {
            pop: 4,
            gens: 1,
            fitness: FitnessMode::LiteRace,
            ridge_scoring: RidgeScoring::Certified,
            ..EvolveConfig::default()
        };
        evolve(&x, &y, &r2, &cfg, &Log::default());
    }
}
// B11 ridge scoring: end

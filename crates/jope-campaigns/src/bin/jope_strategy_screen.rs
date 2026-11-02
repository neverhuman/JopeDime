//! Strategy screen: race five DL-seeded GP search protocols on one dataset.
//!
//! One process = one (strategy, dataset, master_seed) triple, run entirely
//! in-memory on a <=1000-row train.csv. The `run` subcommand NEVER reads
//! test.csv -- the job it runs in is staged without one, and final scoring is
//! a separate `score` invocation executed locally after the winner gene's
//! identity is frozen in result.json. All search fitness comes from resampled
//! splits and CV inside train.csv.
//!
//! The fitness aggregator (mean - 0.25*sd, worst-split tie-break) and the
//! per-generation distinct split-seed discipline are copied from
//! src/commands/train_only_study.rs (paired_robust_fitness :1080, rank_scored
//! :1098, split seeding :1150) -- copied rather than imported because those
//! live in the main binary crate, and this screen must not edit shared files
//! while another agent holds working-tree state across them.

use std::collections::HashSet;
use std::fs;
use std::process::ExitCode;

use feat_math::full_rust::{
    canonicalize, crossover_pipelines, mutate_pipeline, mutate_pipeline_expanded,
};
use feat_math::{model, GenomeIr, Pcg32, Rows};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SCHEMA_CONFIG_V1: &str = "JopeStrategyScreenConfig.v1";
const SCHEMA_CONFIG_V2: &str = "JopeStrategyScreenConfig.v2";
const SCHEMA_RESULT_V1: &str = "JopeStrategyScreenResult.v1";
const SCHEMA_RESULT_V2: &str = "JopeStrategyScreenResult.v2";
const CROSSOVER_PROBABILITY: f64 = 0.25;
const TOURNAMENT: usize = 3;
const BREED_RETRIES: usize = 100;
const STRICT_MIN_FIT_ROWS: usize = 32;
const STRICT_MIN_HOLD_ROWS: usize = 2;

#[derive(Deserialize)]
struct Budget {
    population: usize,
    generations: usize,
    #[serde(default)]
    splits_per_gen: usize,
    elites: usize,
    #[serde(default)]
    cv_fitness: usize,
    #[serde(default)]
    dl_inject_per_gen: usize,
    #[serde(default)]
    halving: Vec<(usize, usize)>,
}

#[derive(Deserialize)]
struct Config {
    schema: String,
    strategy: String,
    dataset_hash: String,
    master_seed: u64,
    dl_genes: Vec<String>,
    #[serde(default)]
    dl_draw_bank: Vec<String>,
    #[serde(default)]
    global_prior: Vec<String>,
    #[serde(default)]
    neighbor_prior: Vec<String>,
    budget: Budget,
    final_cv_folds: usize,
    final_top_k: usize,
    #[serde(default)]
    perturb_fitness: bool,
    /// "legacy" keeps the 31-transform vocabulary every prior round searched;
    /// "expanded" additionally reaches EXPANSION_TRANSFORMS, the zscore target
    /// and the wider expression set. Defaulting to legacy keeps recorded runs
    /// replayable bit-for-bit.
    #[serde(default = "default_vocab")]
    vocab: String,
    /// Seed generation 0 with BASELINE_SEEDS. Off by default so recorded runs
    /// replay bit-for-bit.
    #[serde(default)]
    baseline_seeds: bool,
    /// Score the finale by pooled out-of-fold R^2 instead of the mean of
    /// per-fold R^2. Off by default so recorded runs replay bit-for-bit.
    #[serde(default)]
    pooled_oof: bool,
    /// "cv7" (default, historical) picks the argmax of the finale CV score.
    /// "rcv_1se" picks by repeated CV under the one-standard-error rule, which
    /// steps back from the argmax that the winner's curse inflates.
    #[serde(default = "default_winner_rule")]
    winner_rule: String,
    /// Exclude folds that never produced a number (timeout/spawn/crash) from a
    /// gene's CV mean instead of scoring them as the clip floor. Off by default
    /// so recorded runs replay bit-for-bit. Implied by pooled_oof, which cannot
    /// pool predictions it never received.
    #[serde(default)]
    drop_failed_folds: bool,
    /// Width of the repeated-CV acceptance band in standard errors. 1.0 is the
    /// textbook one-standard-error rule but admitted most of the pool on our
    /// fold sizes; 0.5 is the measured-sane default for this harness.
    #[serde(default = "default_se_band")]
    se_band: f64,
}

impl Config {
    fn strict_pooled_search(&self) -> bool {
        self.schema == SCHEMA_CONFIG_V2
    }

    fn validate(&self) -> Result<(), String> {
        // Applies to every schema. These are bare string/float reads at their use
        // sites, so a typo ("rcv1se", "Expanded") silently selected the HISTORICAL
        // behaviour and still emitted a normal-looking artifact -- for a selection
        // A/B that means the treatment arm records as the control and nothing
        // anywhere says so. Reject unknown values instead.
        if !matches!(self.vocab.as_str(), "legacy" | "expanded") {
            return Err(format!(
                "unknown vocab {:?}: expected \"legacy\" or \"expanded\"",
                self.vocab
            ));
        }
        if !matches!(self.winner_rule.as_str(), "cv7" | "rcv_1se") {
            return Err(format!(
                "unknown winner_rule {:?}: expected \"cv7\" or \"rcv_1se\"",
                self.winner_rule
            ));
        }
        if !self.se_band.is_finite() || self.se_band < 0.0 {
            return Err(format!(
                "se_band must be finite and non-negative, got {}",
                self.se_band
            ));
        }
        match self.schema.as_str() {
            SCHEMA_CONFIG_V1 => Ok(()),
            SCHEMA_CONFIG_V2 => {
                if self.budget.cv_fitness != 2 {
                    return Err(format!("{SCHEMA_CONFIG_V2} requires budget.cv_fitness=2"));
                }
                if !self.pooled_oof {
                    return Err(format!("{SCHEMA_CONFIG_V2} requires pooled_oof=true"));
                }
                if self.drop_failed_folds {
                    return Err(format!(
                        "{SCHEMA_CONFIG_V2} forbids drop_failed_folds; every fold must complete"
                    ));
                }
                if self.winner_rule != "rcv_1se" {
                    return Err(format!("{SCHEMA_CONFIG_V2} requires winner_rule=rcv_1se"));
                }
                if (self.se_band - 0.5).abs() > f64::EPSILON {
                    return Err(format!("{SCHEMA_CONFIG_V2} requires se_band=0.5"));
                }
                if self.perturb_fitness {
                    return Err(format!(
                        "{SCHEMA_CONFIG_V2} forbids perturb_fitness in pooled CV search"
                    ));
                }
                if self.budget.population == 0 || self.budget.generations == 0 {
                    return Err(format!(
                        "{SCHEMA_CONFIG_V2} requires positive population and generations"
                    ));
                }
                if self.final_cv_folds < 2 {
                    return Err(format!("{SCHEMA_CONFIG_V2} requires final_cv_folds >= 2"));
                }
                Ok(())
            }
            other => Err(format!("unsupported config schema {other}")),
        }
    }
}

fn default_se_band() -> f64 {
    1.0
}

fn default_winner_rule() -> String {
    "cv7".to_string()
}

fn default_vocab() -> String {
    "legacy".to_string()
}

/// Fixed textbook models, fittable train-only, chosen a priori rather than by
/// any test signal. See the D1 ceiling probe for why these must be in the pool.
const BASELINE_SEEDS: [&str; 6] = [
    "(pipe X (ridge lambda_frac=0.0))",
    "(pipe X (ridge_gcv))",
    "(pipe X (elasticnet))",
    "(pipe X (histgbm2))",
    "(pipe X (stack))",
    "(pipe X (portfolio preset=core))",
];

#[derive(Clone, Serialize)]
struct Member {
    canonical: String,
    semantic: String,
    complexity: usize,
    origin: String,
    #[serde(skip)]
    scores: Vec<f64>,
    #[serde(skip)]
    fitness_complete: bool,
}

#[derive(Serialize)]
struct TopGene {
    canonical: String,
    semantic: String,
    origin: String,
    complexity: usize,
    robust_fitness: f64,
    worst_split: f64,
    splits_seen: usize,
    search_fitness_complete: bool,
    cv7_mean_r2: Option<f64>,
    rcv_mean_r2: Option<f64>,
    rcv_se: Option<f64>,
    boot_mean_r2: Option<f64>,
    boot_sd_r2: Option<f64>,
    /// Folds that produced no number at all (timeout/spawn/crash), as opposed to
    /// folds the gene genuinely scored badly on. Non-zero means this gene's CV
    /// estimate rests on fewer folds than its rivals'.
    cv_failed_folds: usize,
    /// Repeated-CV partitions lacking at least one required fold prediction.
    rcv_failed_repeats: usize,
}

fn seed_from(tag: &str) -> u64 {
    let digest = Sha256::digest(tag.as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("sha256 has 8 bytes"))
}

/// Headerless numeric CSV, target in the final column (the registry contract:
/// meta.json `headers_removed_from_csv: true`, `target_is_final_column: true`).
mod evaluation;
mod search;
#[cfg(test)]
mod tests;

#[allow(unused_imports)]
use evaluation::*;
#[allow(unused_imports)]
use search::*;
enum EvalFailure {
    Spawn,
    Timeout,
    Crashed,
    Unparseable,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct EvalFailureCounts {
    spawn: usize,
    timeout: usize,
    crashed: usize,
    unparseable: usize,
    invalid_oof: usize,
}

impl EvalFailureCounts {
    fn record(&mut self, failure: EvalFailure) {
        match failure {
            EvalFailure::Spawn => self.spawn += 1,
            EvalFailure::Timeout => self.timeout += 1,
            EvalFailure::Crashed => self.crashed += 1,
            EvalFailure::Unparseable => self.unparseable += 1,
        }
    }

    fn merge(&mut self, other: Self) {
        self.spawn += other.spawn;
        self.timeout += other.timeout;
        self.crashed += other.crashed;
        self.unparseable += other.unparseable;
        self.invalid_oof += other.invalid_oof;
    }

    fn total(self) -> usize {
        self.spawn + self.timeout + self.crashed + self.unparseable + self.invalid_oof
    }
}

struct GenerationStat {
    generation: usize,
    evaluations: usize,
    best_fitness: f64,
    invalid_members: usize,
}

#[allow(clippy::too_many_arguments)]
fn final_cv(
    config: &Config,
    train_path: &str,
    _x: &Rows,
    y: &[f64],
    top: &mut [TopGene],
) -> Result<usize, String> {
    let fold_count = if config.strict_pooled_search() {
        config.final_cv_folds
    } else {
        usable_folds(y.len(), config.final_cv_folds)
    };
    let folds = kfold(
        y,
        fold_count,
        seed_from(&format!(
            "screen:{}:{}:finalcv",
            config.dataset_hash, config.master_seed
        )),
    );
    if config.strict_pooled_search() {
        validate_strict_fold_plan(y.len(), &folds)?;
    }
    let mut evals = 0usize;
    let fold_train: Vec<Vec<usize>> = (0..folds.len())
        .map(|f| {
            folds
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != f)
                .flat_map(|(_, ids)| ids.iter().copied())
                .collect()
        })
        .collect();
    let pairs: Vec<(usize, usize)> = (0..top.len())
        .flat_map(|g| (0..folds.len()).map(move |f| (g, f)))
        .collect();
    // Per-fold R^2 is not merely unstable on a one-row fold, it is UNDEFINED:
    // ss_tot is 0, r2() returns None, and the caller scores the clip floor. With
    // leave-one-out that happens on EVERY fold, so every gene ties at -1.0 and
    // the winner falls through to the lexicographically smallest hash. Pooling
    // is therefore not optional below two rows per fold -- it is the only
    // defined estimator, so force it regardless of the flag.
    let degenerate_fold = folds.iter().any(|f| f.len() < 2);
    if config.pooled_oof || degenerate_fold {
        // Pooled out-of-fold R^2: one ratio computed over every row once, rather
        // than an average of per-fold R^2 values that each carry their own
        // ss_tot denominator. feat_math documents the pooled form as the primary
        // CV estimate; the per-fold mean is unstable on small folds and rewards
        // a gene that drew one lucky slice.
        type FoldPrediction = ((usize, usize), Result<Vec<f64>, EvalFailure>);
        let fold_preds: Vec<FoldPrediction> = pairs
            .par_iter()
            .map(|&(g, f)| {
                (
                    (g, f),
                    eval_predictions(train_path, &top[g].canonical, &fold_train[f], &folds[f]),
                )
            })
            .collect();
        for (g, gene) in top.iter_mut().enumerate() {
            let mut oof_true: Vec<f64> = Vec::new();
            let mut oof_pred: Vec<f64> = Vec::new();
            let mut prediction_views: Vec<Option<&[f64]>> = Vec::with_capacity(folds.len());
            let mut failures = 0usize;
            for (f, fold) in folds.iter().enumerate() {
                let entry = fold_preds
                    .iter()
                    .find(|((gg, ff), _)| *gg == g && *ff == f)
                    .map(|(_, r)| r);
                match entry {
                    Some(Ok(pred)) if pred.len() == fold.len() => {
                        prediction_views.push(Some(pred.as_slice()));
                        for (k, &row) in fold.iter().enumerate() {
                            oof_true.push(y[row]);
                            oof_pred.push(pred[k]);
                        }
                    }
                    _ => {
                        prediction_views.push(None);
                        failures += 1;
                    }
                }
            }
            gene.cv_failed_folds = failures;
            gene.cv7_mean_r2 = if config.strict_pooled_search() {
                complete_pooled_oof_r2(y, &folds, &prediction_views)
            } else if oof_true.is_empty() {
                None
            } else {
                r2(&oof_true, &oof_pred).map(|v| v.max(-1.0))
            };
            evals += folds.len();
        }
        return Ok(evals);
    }

    let fold_scores: Vec<((usize, usize), Result<f64, EvalFailure>)> = pairs
        .par_iter()
        .map(|&(g, f)| {
            (
                (g, f),
                eval_checked(train_path, &top[g].canonical, &fold_train[f], &folds[f]),
            )
        })
        .collect();
    let cv_scores: Vec<(Option<f64>, usize)> = (0..top.len())
        .map(|g| {
            let mine: Vec<&Result<f64, EvalFailure>> = fold_scores
                .iter()
                .filter(|((gg, _), _)| *gg == g)
                .map(|(_, v)| v)
                .collect();
            let failures = mine.iter().filter(|r| r.is_err()).count();
            // Legacy behaviour scores an infrastructure failure as the clip
            // floor, which is indistinguishable from a genuinely bad model.
            // Dropping those folds instead is a real change, so it stays behind
            // a flag: with it off, recorded runs replay bit-for-bit.
            let s: Vec<f64> = if config.drop_failed_folds {
                mine.iter()
                    .filter_map(|r| r.as_ref().ok().copied())
                    .collect()
            } else {
                mine.iter().map(|r| *r.as_ref().unwrap_or(&-1.0)).collect()
            };
            let mean = if s.is_empty() {
                None
            } else {
                Some(s.iter().sum::<f64>() / s.len() as f64)
            };
            (mean, failures)
        })
        .collect();
    for (gene, (cv, failures)) in top.iter_mut().zip(cv_scores) {
        gene.cv7_mean_r2 = cv;
        gene.cv_failed_folds = failures;
        evals += folds.len();
    }
    Ok(evals)
}

fn run(config_path: &str, train_path: &str, output_path: &str) -> Result<(), String> {
    let config: Config = serde_json::from_str(
        &fs::read_to_string(config_path).map_err(|e| format!("read {config_path}: {e}"))?,
    )
    .map_err(|e| format!("parse config: {e}"))?;
    config.validate()?;
    let (x, y) = load_csv(train_path)?;
    let started = std::time::Instant::now();
    let (population, stats, search_evals) = run_strategy(&config, train_path, &x, &y)?;

    // Seeding baselines into generation 0 is not enough: the in-generation
    // fitness culls every one of them before the finale (measured — all 8
    // finalists came back origin=evolved on a seeded run). Union them into the
    // finale pool so CV always gets to compare against the floor.
    let mut finalists: Vec<Member> = population
        .iter()
        .take(config.final_top_k)
        .cloned()
        .collect();
    if config.baseline_seeds {
        let chosen: HashSet<String> = finalists.iter().map(|m| m.semantic.clone()).collect();
        // Rebuilt from the constant, not filtered out of `population`: evolution
        // has already dropped them by this point, which is the whole problem.
        for g in BASELINE_SEEDS {
            if let Some(m) = make_member(g, "baseline") {
                if !chosen.contains(&m.semantic) {
                    finalists.push(m);
                }
            }
        }
    }

    let mut top: Vec<TopGene> = finalists
        .iter()
        .map(|m| TopGene {
            canonical: m.canonical.clone(),
            semantic: m.semantic.clone(),
            origin: m.origin.clone(),
            complexity: m.complexity,
            robust_fitness: robust_fitness(&m.scores),
            worst_split: worst_split(&m.scores),
            splits_seen: m.scores.len(),
            search_fitness_complete: m.fitness_complete,
            cv7_mean_r2: None,
            rcv_mean_r2: None,
            rcv_se: None,
            boot_mean_r2: None,
            boot_sd_r2: None,
            cv_failed_folds: 0,
            rcv_failed_repeats: 0,
        })
        .collect();
    let cv_evals = final_cv(&config, train_path, &x, &y, &mut top)?;
    if config.strict_pooled_search() && top.iter().all(|gene| gene.cv7_mean_r2.is_none()) {
        return Err("strict pooled final CV produced no complete finalist".to_string());
    }

    // ---- T4: repeated stratified CV (5 x k folds), 1-SE + parsimony ----
    let repeats = 5usize;
    let mut rcv_evals = 0usize;
    let mut oof_matrix: Vec<Vec<(usize, f64)>> = vec![Vec::new(); top.len()];
    for rep in 0..repeats {
        let fold_count = if config.strict_pooled_search() {
            config.final_cv_folds
        } else {
            usable_folds(y.len(), config.final_cv_folds)
        };
        let folds = kfold(
            &y,
            fold_count,
            seed_from(&format!(
                "screen:{}:{}:rcv:{rep}",
                config.dataset_hash, config.master_seed
            )),
        );
        if config.strict_pooled_search() {
            validate_strict_fold_plan(y.len(), &folds)?;
        }
        let fold_train: Vec<Vec<usize>> = (0..folds.len())
            .map(|f| {
                folds
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != f)
                    .flat_map(|(_, ids)| ids.iter().copied())
                    .collect()
            })
            .collect();
        let pairs: Vec<(usize, usize)> = (0..top.len())
            .flat_map(|g| (0..folds.len()).map(move |f| (g, f)))
            .collect();
        rcv_evals += pairs.len();
        // One pooled OOF R^2 per repeat, not a mean of per-fold R^2. Per-fold
        // R^2 is undefined under leave-one-out -- a single held-out row has
        // ss_tot = 0, so every fold returns None and the whole estimator
        // collapses to the clip floor with zero variance, which then hands
        // selection to the complexity/hash fallback. Pooling the repeat's
        // predictions keeps the statistic defined at every n we support.
        type FoldPrediction = ((usize, usize), Result<Vec<f64>, EvalFailure>);
        let outs: Vec<FoldPrediction> = pairs
            .par_iter()
            .map(|&(g, f)| {
                (
                    (g, f),
                    eval_predictions(train_path, &top[g].canonical, &fold_train[f], &folds[f]),
                )
            })
            .collect();
        for g in 0..top.len() {
            let mut truth: Vec<f64> = Vec::new();
            let mut pred: Vec<f64> = Vec::new();
            let mut prediction_views: Vec<Option<&[f64]>> = Vec::with_capacity(folds.len());
            for (f, fold) in folds.iter().enumerate() {
                if let Some((_, Ok(p))) = outs.iter().find(|((gg, ff), _)| *gg == g && *ff == f) {
                    if p.len() == fold.len() {
                        prediction_views.push(Some(p.as_slice()));
                        for (k, &row) in fold.iter().enumerate() {
                            truth.push(y[row]);
                            pred.push(p[k]);
                        }
                    } else {
                        prediction_views.push(None);
                    }
                } else {
                    prediction_views.push(None);
                }
            }
            let score = if config.strict_pooled_search() {
                complete_pooled_oof_r2(&y, &folds, &prediction_views)
            } else {
                r2(&truth, &pred).map(|value| value.max(-1.0))
            };
            if let Some(v) = score {
                oof_matrix[g].push((rep, v.max(-1.0)));
            } else {
                top[g].rcv_failed_repeats += 1;
            }
        }
    }
    for (g, gene) in top.iter_mut().enumerate() {
        let vals: Vec<f64> = oof_matrix[g].iter().map(|(_, v)| *v).collect();
        let complete = !config.strict_pooled_search()
            || (gene.rcv_failed_repeats == 0 && vals.len() == repeats);
        if complete && !vals.is_empty() {
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let sd =
                (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt();
            gene.rcv_mean_r2 = Some(mean);
            gene.rcv_se = Some(sd / (vals.len() as f64).sqrt());
        }
    }
    let t4_choice = {
        let best = top
            .iter()
            .filter(|g| !config.strict_pooled_search() || g.cv_failed_folds == 0)
            .filter_map(|g| g.rcv_mean_r2.map(|m| (m, g.rcv_se.unwrap_or(0.0))))
            .fold(
                (f64::NEG_INFINITY, 0.0),
                |a, b| if b.0 > a.0 { b } else { a },
            );
        // Band width is configurable because 1.0 SE admitted 10-14 of 16
        // candidates on measured runs, and a band containing most of the pool
        // is not a selection rule -- combined with the complexity tie-break it
        // degenerates to "pick whichever injected baseline is simplest".
        let threshold = best.0 - config.se_band * best.1;
        top.iter()
            .filter(|g| !config.strict_pooled_search() || g.cv_failed_folds == 0)
            .filter(|g| g.rcv_mean_r2.unwrap_or(f64::NEG_INFINITY) >= threshold)
            // Ties at equal complexity break by SCORE, then by hash. Breaking
            // straight to the hash made this an OLS bias, not a 1-SE rule: all
            // six baselines are complexity 1 and (pipe X (ridge lambda_frac=0.0))
            // hashes smallest, so plain OLS won every tie it entered -- on one
            // measured run it was the worst of five tied candidates.
            .min_by(|a, b| {
                a.complexity
                    .cmp(&b.complexity)
                    .then(
                        b.rcv_mean_r2
                            .unwrap_or(f64::NEG_INFINITY)
                            .total_cmp(&a.rcv_mean_r2.unwrap_or(f64::NEG_INFINITY)),
                    )
                    .then(a.semantic.cmp(&b.semantic))
            })
            .map(|g| g.semantic.clone())
    };
    if config.strict_pooled_search() && config.winner_rule == "rcv_1se" && t4_choice.is_none() {
        return Err("strict repeated CV produced no complete finalist".to_string());
    }

    // ---- T3: bootstrap-refit stability over the top pool ----
    let bags = 20usize;
    let n = y.len();
    let bag_sets: Vec<(Vec<usize>, Vec<usize>)> = (0..bags)
        .map(|b| {
            let seed = seed_from(&format!(
                "screen:{}:{}:boot:{b}",
                config.dataset_hash, config.master_seed
            ));
            let mut rng = Pcg32::new(seed, 0x424f_4f54);
            let mut inb = Vec::with_capacity(n);
            let mut seen = vec![false; n];
            for _ in 0..n {
                let i = (rng.next_u32() as usize) % n;
                inb.push(i);
                seen[i] = true;
            }
            let oob: Vec<usize> = (0..n).filter(|&i| !seen[i]).collect();
            (inb, oob)
        })
        .collect();
    let boot_pairs: Vec<(usize, usize)> = (0..top.len())
        .flat_map(|g| (0..bags).map(move |b| (g, b)))
        .collect();
    let bootstrap_evals = boot_pairs.len();
    let boot_outs: Vec<((usize, usize), f64)> = boot_pairs
        .par_iter()
        .map(|&(g, b)| {
            (
                (g, b),
                eval_with_timeout(
                    train_path,
                    &top[g].canonical,
                    &bag_sets[b].0,
                    &bag_sets[b].1,
                ),
            )
        })
        .collect();
    let mut boot_scores: Vec<Vec<f64>> = vec![Vec::new(); top.len()];
    for ((g, _), v) in boot_outs {
        boot_scores[g].push(v);
    }
    for (g, gene) in top.iter_mut().enumerate() {
        let vals = &boot_scores[g];
        if !vals.is_empty() {
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let sd =
                (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt();
            gene.boot_mean_r2 = Some(mean);
            gene.boot_sd_r2 = Some(sd);
        }
    }
    let t3_choice = top
        .iter()
        .filter(|g| g.boot_mean_r2.is_some())
        .max_by(|a, b| {
            let fa = a.boot_mean_r2.unwrap() - 0.5 * a.boot_sd_r2.unwrap_or(0.0);
            let fb = b.boot_mean_r2.unwrap() - 0.5 * b.boot_sd_r2.unwrap_or(0.0);
            fa.total_cmp(&fb).then(b.semantic.cmp(&a.semantic))
        })
        .map(|g| g.semantic.clone());

    // ---- T2: ensembles from a common OOF partition (first rcv repeat folds) ----
    let ensemble_fold_count = if config.strict_pooled_search() {
        config.final_cv_folds
    } else {
        usable_folds(y.len(), config.final_cv_folds)
    };
    let ens_folds = kfold(
        &y,
        ensemble_fold_count,
        seed_from(&format!(
            "screen:{}:{}:rcv:0",
            config.dataset_hash, config.master_seed
        )),
    );
    if config.strict_pooled_search() {
        validate_strict_fold_plan(y.len(), &ens_folds)?;
    }
    let ensemble_evals = top.len() * ens_folds.len();
    let ens_train: Vec<Vec<usize>> = (0..ens_folds.len())
        .map(|f| {
            ens_folds
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != f)
                .flat_map(|(_, ids)| ids.iter().copied())
                .collect()
        })
        .collect();
    // per-gene OOF prediction vector via child predict? cheap approximation:
    // per-gene per-fold R2 already captured; greedy selection needs prediction
    // vectors -- run one more pass collecting predictions in-process (no
    // timeout risk dominates: reuse eval budget bound by top_k * folds).
    let mut oof_preds: Vec<Vec<f64>> = vec![vec![f64::NAN; n]; top.len()];
    for (g, gene) in top.iter().enumerate() {
        if let Ok(genome) = GenomeIr::parse(&gene.canonical) {
            for f in 0..ens_folds.len() {
                let (tx, ty) = subset(&x, &y, &ens_train[f]);
                let (vx, _) = subset(&x, &y, &ens_folds[f]);
                if let Ok(pred) = model::fit_predict(&genome, &tx, &ty, &vx) {
                    for (j, &row) in ens_folds[f].iter().enumerate() {
                        if let Some(v) = pred.get(j) {
                            oof_preds[g][row] = *v;
                        }
                    }
                }
            }
        }
    }
    let oof_rmse = |mix: &[f64]| -> f64 {
        let mut se = 0.0;
        let mut c = 0usize;
        for i in 0..n {
            if mix[i].is_finite() {
                se += (mix[i] - y[i]).powi(2);
                c += 1;
            }
        }
        if c == 0 {
            f64::INFINITY
        } else {
            (se / c as f64).sqrt()
        }
    };
    // uniform top-5
    let ens_uniform: Vec<String> = top.iter().take(5).map(|g| g.semantic.clone()).collect();
    // greedy Caruana on OOF predictions, with replacement, up to 8 picks
    let mut picks: Vec<usize> = Vec::new();
    let mut mix = vec![f64::NAN; n];
    for _ in 0..8 {
        let mut best: Option<(usize, f64)> = None;
        for (g, predictions) in oof_preds[..top.len()].iter().enumerate() {
            if predictions.iter().all(|v| !v.is_finite()) {
                continue;
            }
            let k = picks.len() as f64;
            let cand: Vec<f64> = (0..n)
                .map(|i| {
                    let a = if mix[i].is_finite() { mix[i] * k } else { 0.0 };
                    let b = if predictions[i].is_finite() {
                        predictions[i]
                    } else {
                        return f64::NAN;
                    };
                    (a + b) / (k + 1.0)
                })
                .collect();
            let r = oof_rmse(&cand);
            if best.is_none() || r < best.unwrap().1 {
                best = Some((g, r));
            }
        }
        match best {
            Some((g, _)) => {
                let k = picks.len() as f64;
                for i in 0..n {
                    let b = oof_preds[g][i];
                    if b.is_finite() {
                        let a = if mix[i].is_finite() { mix[i] * k } else { 0.0 };
                        mix[i] = (a + b) / (k + 1.0);
                    }
                }
                picks.push(g);
            }
            None => break,
        }
    }
    let ens_greedy: Vec<String> = picks.iter().map(|&g| top[g].semantic.clone()).collect();
    // winner: best CV mean, complexity/semantic tie-break via stable sort order
    let mut order: Vec<usize> = (0..top.len()).collect();
    order.sort_by(|&a, &b| {
        top[b]
            .cv7_mean_r2
            .unwrap_or(-2.0)
            .total_cmp(&top[a].cv7_mean_r2.unwrap_or(-2.0))
            .then_with(|| top[a].semantic.cmp(&top[b].semantic))
    });
    let mut winner = order.first().copied().ok_or("empty final population")?;
    // Captured BEFORE any winner_rule reassignment. Emitting top[winner] after
    // the rule ran made control and treatment the same value, so the A/B this
    // field exists to record was silently never recorded.
    let control_cv7_semantic = top[winner].semantic.clone();
    if config.winner_rule == "rcv_1se" {
        // The argmax of a CV score computed on the same rows the search already
        // optimised against is a winner's-curse estimator. Step back to the
        // simplest gene statistically indistinguishable from the best under
        // repeated CV. t4_choice is exactly that rule, already computed.
        if let Some(pick) = t4_choice
            .as_ref()
            .and_then(|sem| top.iter().position(|g| &g.semantic == sem))
        {
            winner = pick;
        }
    }

    let result_schema = if config.strict_pooled_search() {
        SCHEMA_RESULT_V2
    } else {
        SCHEMA_RESULT_V1
    };
    let total_model_fits = search_evals + cv_evals + rcv_evals + bootstrap_evals + ensemble_evals;
    let result = serde_json::json!({
        "schema": result_schema,
        "strategy": config.strategy,
        "dataset_hash": config.dataset_hash,
        "master_seed": config.master_seed,
        "search_fitness": if config.strict_pooled_search() {
            "complete_pooled_oof_r2"
        } else {
            "legacy_history"
        },
        "strict_fold_contract": if config.strict_pooled_search() {
            Some(serde_json::json!({
                "minimum_fit_rows": STRICT_MIN_FIT_ROWS,
                "minimum_hold_rows": STRICT_MIN_HOLD_ROWS,
                "required_search_folds": config.budget.cv_fitness,
                "required_final_folds": config.final_cv_folds,
                "complete_predictions_required": true,
                "fail_on_incomplete_candidate": true,
                "candidate_timeout_seconds": eval_timeout_seconds(),
            }))
        } else {
            None
        },
        "winner_genome": top[winner].canonical,
        "winner_semantic": top[winner].semantic,
        "winner_origin": top[winner].origin,
        "winner_cv_mean_r2": top[winner].cv7_mean_r2,
        "finales": {
            "control_cv7": control_cv7_semantic,
            "t3_bootstrap": t3_choice,
            "t4_rcv_1se": t4_choice,
            "t2_uniform5": ens_uniform,
            "t2_greedy": ens_greedy,
        },
        "search_evaluations": search_evals,
        "final_cv_evaluations": cv_evals,
        "model_fit_accounting": {
            "search": search_evals,
            "final_cv": cv_evals,
            "repeated_cv": rcv_evals,
            "bootstrap": bootstrap_evals,
            "ensemble_oof": ensemble_evals,
            "total": total_model_fits,
        },
        "runtime_seconds": started.elapsed().as_secs_f64(),
        "top": top,
        "generations": stats.iter().map(|s| serde_json::json!({
            "generation": s.generation,
            "evaluations": s.evaluations,
            "best_fitness": s.best_fitness,
            "invalid_members": s.invalid_members,
        })).collect::<Vec<_>>(),
        "test_csv_read": false,
    });
    fs::write(output_path, serde_json::to_string_pretty(&result).unwrap())
        .map_err(|e| format!("write {output_path}: {e}"))?;
    Ok(())
}

fn score(genome_file: &str, train_path: &str, test_path: &str) -> Result<(), String> {
    let source = fs::read_to_string(genome_file).map_err(|e| e.to_string())?;
    let canonical = canonicalize(source.trim()).map_err(|e| format!("canonicalize: {e}"))?;
    let genome = GenomeIr::parse(&canonical.canonical).map_err(|e| format!("parse: {e}"))?;
    let (tx, ty) = load_csv(train_path)?;
    let (vx, vy) = load_csv(test_path)?;
    let pred = model::fit_predict(&genome, &tx, &ty, &vx).map_err(|e| format!("fit: {e}"))?;
    let out = serde_json::json!({
        "rmse": rmse(&vy, &pred),
        "r2": r2(&vy, &pred),
        "semantic": canonical.semantic_sha256,
    });
    println!("{out}");
    Ok(())
}

fn score_ensemble(members_file: &str, train_path: &str, test_path: &str) -> Result<(), String> {
    let members: Vec<String> =
        serde_json::from_str(&fs::read_to_string(members_file).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if members.is_empty() {
        return Err("empty ensemble".into());
    }
    let (tx, ty) = load_csv(train_path)?;
    let (vx, vy) = load_csv(test_path)?;
    let mut acc = vec![0.0f64; vy.len()];
    let mut used = 0usize;
    for m in &members {
        let canonical = canonicalize(m.trim()).map_err(|e| format!("canonicalize: {e}"))?;
        let genome = GenomeIr::parse(&canonical.canonical).map_err(|e| format!("parse: {e}"))?;
        match model::fit_predict(&genome, &tx, &ty, &vx) {
            Ok(p) if p.len() == vy.len() && p.iter().all(|v| v.is_finite()) => {
                for (a, b) in acc.iter_mut().zip(&p) {
                    *a += b;
                }
                used += 1;
            }
            _ => {}
        }
    }
    if used == 0 {
        return Err("no ensemble member fit".into());
    }
    let pred: Vec<f64> = acc.iter().map(|v| v / used as f64).collect();
    let out = serde_json::json!({
        "rmse": rmse(&vy, &pred),
        "r2": r2(&vy, &pred),
        "members_used": used,
    });
    println!("{out}");
    Ok(())
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let outcome = match args.get(1).map(String::as_str) {
        Some("run") => match (
            arg(&args, "--config"),
            arg(&args, "--train"),
            arg(&args, "--output"),
        ) {
            (Some(c), Some(t), Some(o)) => run(&c, &t, &o),
            _ => Err("usage: run --config <json> --train <csv> --output <json>".into()),
        },
        Some("eval-one") => match arg(&args, "--train") {
            Some(t) => eval_one(&t),
            None => {
                Err("usage: eval-one --train <csv>   (stdin: {genome,train_ids,valid_ids})".into())
            }
        },
        Some("score-ensemble") => match (
            arg(&args, "--members-file"),
            arg(&args, "--train"),
            arg(&args, "--test"),
        ) {
            (Some(m), Some(t), Some(e)) => score_ensemble(&m, &t, &e),
            _ => {
                Err("usage: score-ensemble --members-file <json> --train <csv> --test <csv>".into())
            }
        },
        Some("score") => match (
            arg(&args, "--genome-file"),
            arg(&args, "--train"),
            arg(&args, "--test"),
        ) {
            (Some(g), Some(t), Some(e)) => score(&g, &t, &e),
            _ => Err("usage: score --genome-file <txt> --train <csv> --test <csv>".into()),
        },
        _ => Err("subcommands: run | score".into()),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("jope_strategy_screen: {message}");
            ExitCode::FAILURE
        }
    }
}

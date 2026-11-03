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
#[path = "jope_strategy_screen/evaluation.rs"]
mod evaluation;
#[path = "jope_strategy_screen/search.rs"]
mod search;
#[cfg(test)]
#[path = "jope_strategy_screen/tests.rs"]
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

#[path = "jope_strategy_screen/final_cv.rs"]
mod final_cv;
#[path = "jope_strategy_screen/screen.rs"]
mod screen;

use final_cv::final_cv;
use screen::run;

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

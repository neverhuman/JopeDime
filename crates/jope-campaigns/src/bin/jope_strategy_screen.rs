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
fn load_csv(path: &str) -> Result<(Rows, Vec<f64>), String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut x: Rows = Vec::new();
    let mut y = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut row: Vec<f64> = Vec::new();
        for cell in line.split(',') {
            let v: f64 = cell
                .trim()
                .parse()
                .map_err(|e| format!("{path}:{}: {e}", line_no + 1))?;
            row.push(v);
        }
        if row.len() < 2 {
            return Err(format!("{path}:{}: fewer than 2 columns", line_no + 1));
        }
        y.push(row.pop().expect("row has columns"));
        x.push(row);
    }
    if x.is_empty() {
        return Err(format!("{path}: no rows"));
    }
    let width = x[0].len();
    if x.iter().any(|r| r.len() != width) {
        return Err(format!("{path}: ragged rows"));
    }
    Ok((x, y))
}

fn r2(actual: &[f64], predicted: &[f64]) -> Option<f64> {
    if actual.len() != predicted.len() || actual.is_empty() {
        return None;
    }
    if predicted.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mean = actual.iter().sum::<f64>() / actual.len() as f64;
    let ss_tot: f64 = actual.iter().map(|a| (a - mean).powi(2)).sum();
    let ss_res: f64 = actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p).powi(2))
        .sum();
    if ss_tot <= 0.0 {
        return None;
    }
    Some(1.0 - ss_res / ss_tot)
}

fn rmse(actual: &[f64], predicted: &[f64]) -> Option<f64> {
    if actual.len() != predicted.len() || actual.is_empty() {
        return None;
    }
    if predicted.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mse: f64 = actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>()
        / actual.len() as f64;
    Some(mse.sqrt())
}

/// Copied from train_only_study.rs::paired_robust_fitness (mean - 0.25*sd,
/// scores clipped to >= -1 so one catastrophic split cannot dominate).
fn robust_fitness(scores: &[f64]) -> f64 {
    if scores.is_empty() {
        return -1.0;
    }
    let clipped: Vec<f64> = scores.iter().map(|v| v.max(-1.0)).collect();
    let mean = clipped.iter().sum::<f64>() / clipped.len() as f64;
    let sd =
        (clipped.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / clipped.len() as f64).sqrt();
    mean - 0.25 * sd.min(1.0)
}

fn worst_split(scores: &[f64]) -> f64 {
    scores
        .iter()
        .map(|v| v.max(-1.0))
        .fold(f64::INFINITY, f64::min)
}

/// rank_scored ordering from train_only_study.rs: fitness desc, worst-split
/// desc, complexity asc, semantic asc (fully deterministic).
fn rank(members: &mut [Member]) {
    members.sort_by(|a, b| {
        b.fitness_complete
            .cmp(&a.fitness_complete)
            .then_with(|| robust_fitness(&b.scores).total_cmp(&robust_fitness(&a.scores)))
            .then_with(|| worst_split(&b.scores).total_cmp(&worst_split(&a.scores)))
            .then_with(|| a.complexity.cmp(&b.complexity))
            .then_with(|| a.semantic.cmp(&b.semantic))
    });
}

/// Deterministic stratified index split: rows sorted by target, sliced into
/// deciles, each decile shuffled with its own stream, the leading `hold_frac`
/// of each shuffled decile becoming the validation fold. Fresh `seed` per
/// generation gives a distinct split every time (train_only_study discipline).
fn split_8020(y: &[f64], seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| y[a].total_cmp(&y[b]).then(a.cmp(&b)));
    let deciles = 10usize.min(y.len().max(1));
    let chunk = order.len().div_ceil(deciles);
    let mut train = Vec::new();
    let mut valid = Vec::new();
    for (d, ids) in order.chunks(chunk).enumerate() {
        let mut ids: Vec<usize> = ids.to_vec();
        let mut rng = Pcg32::new(seed, 0x5350_4c49_5430 ^ d as u64);
        for i in (1..ids.len()).rev() {
            let j = (rng.next_u32() as usize) % (i + 1);
            ids.swap(i, j);
        }
        let hold = (ids.len() as f64 * 0.2).round().max(1.0) as usize;
        for (k, id) in ids.into_iter().enumerate() {
            if k < hold {
                valid.push(id);
            } else {
                train.push(id);
            }
        }
    }
    if train.is_empty() || valid.is_empty() {
        let mid = y.len() * 4 / 5;
        return ((0..mid).collect(), (mid..y.len()).collect());
    }
    (train, valid)
}

/// Legacy V1 fold-count adaptation. Strict V2 never changes the declared fold
/// count: an infeasible plan is a structural failure rather than a reason to
/// substitute leave-one-out CV.
fn usable_folds(n: usize, configured_k: usize) -> usize {
    if n < 2 {
        return 0;
    }
    let k = configured_k.max(2).min(n);
    if n < 2 * k {
        n
    } else {
        k
    }
}

/// Deterministic stratified k-fold: per-decile shuffle, then deal round-robin.
fn kfold(y: &[f64], k: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| y[a].total_cmp(&y[b]).then(a.cmp(&b)));
    let deciles = 10usize.min(y.len().max(1));
    let chunk = order.len().div_ceil(deciles);
    let mut folds = vec![Vec::new(); k];
    let mut card = 0usize;
    for (d, ids) in order.chunks(chunk).enumerate() {
        let mut ids: Vec<usize> = ids.to_vec();
        let mut rng = Pcg32::new(seed, 0x4b46_4f4c_44 ^ d as u64);
        for i in (1..ids.len()).rev() {
            let j = (rng.next_u32() as usize) % (i + 1);
            ids.swap(i, j);
        }
        for id in ids {
            folds[card % k].push(id);
            card += 1;
        }
    }
    folds
}

fn validate_strict_fold_plan(row_count: usize, folds: &[Vec<usize>]) -> Result<(), String> {
    if folds.len() < 2 {
        return Err("strict pooled CV requires at least two folds".to_string());
    }
    let mut coverage = vec![0u8; row_count];
    for (fold_index, fold) in folds.iter().enumerate() {
        if fold.len() < STRICT_MIN_HOLD_ROWS {
            return Err(format!(
                "fold {fold_index} has {} hold rows; minimum is {STRICT_MIN_HOLD_ROWS}",
                fold.len()
            ));
        }
        let fit_rows = row_count.saturating_sub(fold.len());
        if fit_rows < STRICT_MIN_FIT_ROWS {
            return Err(format!(
                "fold {fold_index} has {fit_rows} fit rows; minimum is {STRICT_MIN_FIT_ROWS}"
            ));
        }
        for &row in fold {
            let Some(count) = coverage.get_mut(row) else {
                return Err(format!("fold {fold_index} contains out-of-range row {row}"));
            };
            *count = count.saturating_add(1);
        }
    }
    if let Some((row, count)) = coverage.iter().enumerate().find(|(_, count)| **count != 1) {
        return Err(format!(
            "row {row} appears in {count} hold folds; expected exactly one"
        ));
    }
    Ok(())
}

/// Complete pooled OOF R^2 in original row order. For a fixed truth vector,
/// ranking by this value is exactly equivalent to ranking by pooled OOF RMSE.
/// Any absent, malformed, duplicated, or nonfinite fold fails closed.
fn complete_pooled_oof_r2(
    y: &[f64],
    folds: &[Vec<usize>],
    fold_predictions: &[Option<&[f64]>],
) -> Option<f64> {
    if fold_predictions.len() != folds.len() || y.is_empty() {
        return None;
    }
    let mut predictions = vec![f64::NAN; y.len()];
    let mut covered = vec![false; y.len()];
    for (fold, maybe_predictions) in folds.iter().zip(fold_predictions) {
        let values = maybe_predictions.as_ref()?;
        if values.len() != fold.len() {
            return None;
        }
        for (&row, &prediction) in fold.iter().zip(values.iter()) {
            if row >= y.len() || covered[row] || !prediction.is_finite() {
                return None;
            }
            predictions[row] = prediction;
            covered[row] = true;
        }
    }
    if covered.iter().any(|covered| !covered) {
        return None;
    }
    r2(y, &predictions).map(|value| value.max(-1.0))
}

fn replace_generation_outcomes(population: &mut [Member], outcomes: Vec<Option<f64>>) -> usize {
    assert_eq!(population.len(), outcomes.len());
    let mut invalid = 0usize;
    for (member, outcome) in population.iter_mut().zip(outcomes) {
        member.scores.clear();
        member.fitness_complete = outcome.is_some();
        if !member.fitness_complete {
            invalid += 1;
        }
        member.scores.push(outcome.unwrap_or(-1.0));
    }
    invalid
}

fn expected_search_evaluations(schedule: &[(usize, usize)], folds: usize) -> usize {
    schedule.iter().map(|(width, _)| width * folds).sum()
}

/// T5: deterministic per-generation perturbation of the TRAIN-side rows only.
/// Even generations: Gaussian column noise at 5% of column SD. Odd: 10% row
/// dropout (rows zero-weighted by exclusion from train_ids). Validation rows
/// are always untouched. Returns the CSV path children should load and the
/// possibly-thinned train ids.
fn perturbed_generation_csv(
    x: &Rows,
    y: &[f64],
    train_ids: &[usize],
    generation: usize,
    tag: &str,
) -> Result<(String, Vec<usize>), String> {
    let seed = seed_from(&format!("screen:perturb:{tag}:{generation}"));
    let mut rng = Pcg32::new(seed, 0x5045_5254);
    let cols = x[0].len();
    let mut out = x.clone();
    let mut kept: Vec<usize> = train_ids.to_vec();
    if generation % 2 == 0 {
        let mut sd = vec![0.0f64; cols];
        for c in 0..cols {
            let vals: Vec<f64> = train_ids.iter().map(|&i| x[i][c]).collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            sd[c] =
                (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt();
        }
        for &i in train_ids {
            for c in 0..cols {
                out[i][c] += 0.05 * sd[c] * rng.gaussian();
            }
        }
    } else {
        kept = train_ids
            .iter()
            .copied()
            .filter(|_| rng.next_f64() >= 0.10)
            .collect();
        if kept.len() < train_ids.len() / 2 {
            kept = train_ids.to_vec();
        }
    }
    let path = format!(".perturbed-gen{generation}.csv");
    let mut text = String::new();
    for (i, row) in out.iter().enumerate() {
        for v in row {
            text.push_str(&format!("{v},"));
        }
        text.push_str(&format!(
            "{}
",
            y[i]
        ));
    }
    fs::write(&path, text).map_err(|e| e.to_string())?;
    Ok((path, kept))
}

fn subset(x: &Rows, y: &[f64], ids: &[usize]) -> (Rows, Vec<f64>) {
    (
        ids.iter().map(|&i| x[i].clone()).collect(),
        ids.iter().map(|&i| y[i]).collect(),
    )
}

/// Fit on the train subset, score R^2 on the validation subset. Any failure
/// (parse, fit, nonfinite, degenerate target) scores the clip floor -1.0 so a
/// broken gene ranks last instead of aborting the run.
fn eval_on(member: &Member, x: &Rows, y: &[f64], train_ids: &[usize], valid_ids: &[usize]) -> f64 {
    let Ok(genome) = GenomeIr::parse(&member.canonical) else {
        return -1.0;
    };
    let (tx, ty) = subset(x, y, train_ids);
    let (vx, vy) = subset(x, y, valid_ids);
    match model::fit_predict(&genome, &tx, &ty, &vx) {
        Ok(pred) => r2(&vy, &pred).map_or(-1.0, |v| v.max(-1.0)),
        Err(_) => -1.0,
    }
}

/// Fit on the train subset and return the raw validation predictions, so the
/// parent can pool them across folds. Pooled out-of-fold R^2 is a single ratio
/// over every row; the per-fold mean this replaces normalises each fold by its
/// own ss_tot, which on small folds rewards a gene that drew a lucky slice.
fn predict_on(
    member: &Member,
    x: &Rows,
    y: &[f64],
    train_ids: &[usize],
    valid_ids: &[usize],
) -> Option<Vec<f64>> {
    let genome = GenomeIr::parse(&member.canonical).ok()?;
    let (tx, ty) = subset(x, y, train_ids);
    let (vx, _) = subset(x, y, valid_ids);
    model::fit_predict(&genome, &tx, &ty, &vx).ok()
}

/// Child mode: read {genome, train_ids, valid_ids} from stdin, print r2.
/// With "want":"pred" the child prints a JSON array of validation predictions
/// instead, which is what the pooled-OOF finale consumes.
fn eval_one(train_path: &str) -> Result<(), String> {
    use std::io::Read;
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .map_err(|e| e.to_string())?;
    let req: serde_json::Value = serde_json::from_str(&buf).map_err(|e| e.to_string())?;
    let genome = req["genome"].as_str().ok_or("genome missing")?;
    let ids = |k: &str| -> Vec<usize> {
        req[k]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|u| u as usize))
                    .collect()
            })
            .unwrap_or_default()
    };
    let (x, y) = load_csv(train_path)?;
    let member = Member {
        canonical: genome.to_string(),
        semantic: String::new(),
        complexity: 0,
        origin: String::new(),
        scores: Vec::new(),
        fitness_complete: false,
    };
    if req["want"].as_str() == Some("pred") {
        let pred = predict_on(&member, &x, &y, &ids("train_ids"), &ids("valid_ids"))
            .ok_or("fit_predict failed")?;
        println!(
            "{}",
            serde_json::to_string(&pred).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    let score = eval_on(&member, &x, &y, &ids("train_ids"), &ids("valid_ids"));
    println!("{score}");
    Ok(())
}

/// Why a child evaluation produced no usable number. A model that fits and
/// scores badly is NOT a failure -- it earns the -1.0 clip floor honestly.
/// These variants are the cases where we learned nothing about the gene, and
/// conflating them with a bad score systematically punishes whatever is
/// slowest (wide feature sets, heavy heads) for infrastructure reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvalFailure {
    Spawn,
    Timeout,
    Crashed,
    Unparseable,
}

/// Parent mode: run one evaluation in a killable child with a hard deadline
/// (campaign parity: golden42 enforced a 30 s per-candidate timeout at the
/// worker level). Returns the child's raw stdout.
fn run_eval_child(
    train_path: &str,
    genome: &str,
    train_ids: &[usize],
    valid_ids: &[usize],
    want_pred: bool,
) -> Result<String, EvalFailure> {
    use std::io::Write;
    let timeout = std::env::var("JSS_EVAL_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30);
    let exe = std::env::current_exe().map_err(|_| EvalFailure::Spawn)?;
    let mut child = std::process::Command::new(exe)
        .arg("eval-one")
        .arg("--train")
        .arg(train_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|_| EvalFailure::Spawn)?;
    let mut payload = serde_json::json!({
        "genome": genome, "train_ids": train_ids, "valid_ids": valid_ids,
    });
    if want_pred {
        payload["want"] = serde_json::Value::String("pred".to_string());
    }
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(payload.to_string().as_bytes());
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return Err(EvalFailure::Crashed);
                }
                let mut out = String::new();
                use std::io::Read;
                if let Some(mut stdout) = child.stdout.take() {
                    let _ = stdout.read_to_string(&mut out);
                }
                return Ok(out);
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(EvalFailure::Timeout);
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(EvalFailure::Crashed);
            }
        }
    }
}

/// Score, distinguishing "did not run" (None) from "ran and was bad" (Some).
fn eval_checked(
    train_path: &str,
    genome: &str,
    train_ids: &[usize],
    valid_ids: &[usize],
) -> Result<f64, EvalFailure> {
    let out = run_eval_child(train_path, genome, train_ids, valid_ids, false)?;
    out.trim()
        .parse::<f64>()
        .map(|v| v.max(-1.0))
        .map_err(|_| EvalFailure::Unparseable)
}

/// Validation predictions for one fold, for the pooled-OOF finale.
fn eval_predictions(
    train_path: &str,
    genome: &str,
    train_ids: &[usize],
    valid_ids: &[usize],
) -> Result<Vec<f64>, EvalFailure> {
    let out = run_eval_child(train_path, genome, train_ids, valid_ids, true)?;
    serde_json::from_str::<Vec<f64>>(out.trim()).map_err(|_| EvalFailure::Unparseable)
}

/// Legacy scoring surface: every failure becomes the clip floor. Retained so
/// the search loop's recorded behaviour is unchanged.
fn eval_with_timeout(
    train_path: &str,
    genome: &str,
    train_ids: &[usize],
    valid_ids: &[usize],
) -> f64 {
    eval_checked(train_path, genome, train_ids, valid_ids).unwrap_or(-1.0)
}

fn make_member(source: &str, origin: &str) -> Option<Member> {
    let canonical = canonicalize(source).ok()?;
    Some(Member {
        canonical: canonical.canonical,
        semantic: canonical.semantic_sha256,
        complexity: canonical.complexity,
        origin: origin.to_string(),
        scores: Vec::new(),
        fitness_complete: false,
    })
}

fn tournament_pick(rng: &mut Pcg32, pool: usize) -> usize {
    let mut best = usize::MAX;
    for _ in 0..TOURNAMENT {
        let c = (rng.next_u32() as usize) % pool;
        best = best.min(c);
    }
    best
}

/// Breed one child from the ranked pool (golden42 operators: tournament-3
/// parents, 0.25 crossover, else mutate; 100 deterministic retries).
fn breed_child(
    ranked: &[Member],
    pool: usize,
    features: usize,
    tag: &str,
    seen: &HashSet<String>,
    expanded_vocab: bool,
) -> Option<Member> {
    for retry in 0..BREED_RETRIES {
        let seed = seed_from(&format!("screen:{tag}:{retry}"));
        let mut rng = Pcg32::new(seed, 0x5343_5245_454e);
        let first = tournament_pick(&mut rng, pool);
        let evolved = if rng.next_f64() < CROSSOVER_PROBABILITY {
            let second = tournament_pick(&mut rng, pool);
            crossover_pipelines(
                &ranked[first].canonical,
                &ranked[second].canonical,
                rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
            )
        } else {
            let mutate = if expanded_vocab {
                mutate_pipeline_expanded
            } else {
                mutate_pipeline
            };
            mutate(
                &ranked[first].canonical,
                rng.next_u32() as u64 | ((rng.next_u32() as u64) << 32),
                features,
            )
        };
        if let Ok(g) = evolved {
            if !seen.contains(&g.semantic_sha256) {
                return Some(Member {
                    canonical: g.canonical,
                    semantic: g.semantic_sha256,
                    complexity: g.complexity,
                    origin: "evolved".to_string(),
                    scores: Vec::new(),
                    fitness_complete: false,
                });
            }
        }
    }
    None
}

fn fill_population(
    base: Vec<Member>,
    target: usize,
    features: usize,
    tag: &str,
    expanded_vocab: bool,
) -> Vec<Member> {
    let mut seen: HashSet<String> = base.iter().map(|m| m.semantic.clone()).collect();
    let mut population = base;
    let mut slot = 0usize;
    while population.len() < target && slot < target * 40 {
        if let Some(child) = breed_child(
            &population,
            population.len(),
            features,
            &format!("{tag}:fill:{slot}"),
            &seen,
            expanded_vocab,
        ) {
            seen.insert(child.semantic.clone());
            population.push(child);
        }
        slot += 1;
    }
    population
}

struct GenerationStat {
    generation: usize,
    evaluations: usize,
    best_fitness: f64,
    invalid_members: usize,
}

#[allow(clippy::too_many_arguments)]
fn run_strategy(
    config: &Config,
    train_path: &str,
    x: &Rows,
    y: &[f64],
) -> Result<(Vec<Member>, Vec<GenerationStat>, usize), String> {
    let features = x[0].len();
    let ds = &config.dataset_hash;
    let ms = config.master_seed;
    let mut evaluations = 0usize;
    let mut stats = Vec::new();

    // ---- generation 0 composition ----
    let mut gen0: Vec<Member> = Vec::new();
    let mut seen = HashSet::new();
    let push_unique = |pool: &mut Vec<Member>, seen: &mut HashSet<String>, m: Member| {
        if seen.insert(m.semantic.clone()) {
            pool.push(m);
        }
    };
    for g in config.dl_genes.iter().take(10) {
        if let Some(m) = make_member(g, "dl_seed") {
            push_unique(&mut gen0, &mut seen, m);
        }
    }
    let prior: Vec<(&str, &Vec<String>)> = match config.strategy.as_str() {
        "s2" => vec![("neighbor_prior", &config.neighbor_prior)],
        "s5" => vec![
            ("global_prior", &config.global_prior),
            ("neighbor_prior", &config.neighbor_prior),
        ],
        _ => vec![("global_prior", &config.global_prior)],
    };
    for (origin, list) in prior {
        for g in list
            .iter()
            .take(if config.strategy == "s5" { 32 } else { 10 })
        {
            if let Some(m) = make_member(g, origin) {
                push_unique(&mut gen0, &mut seen, m);
            }
        }
    }
    // A fixed, train-only baseline set. D1 measured that a plain fit of one of
    // these beats a full 1,000-eval search on 7 of 22 datasets (e5a3a7b3 by 37%,
    // b78d3235 by 12%), and that none of them ever entered a pool: DL proposals
    // and priors are all elaborate pipelines, and mutation only elaborates
    // further, so the search could never fall back on a textbook model. Seeding
    // them costs |BASELINE_SEEDS| evaluations and floors the result.
    if config.baseline_seeds {
        for g in BASELINE_SEEDS {
            if let Some(m) = make_member(g, "baseline") {
                push_unique(&mut gen0, &mut seen, m);
            }
        }
    }

    let schedule: Vec<(usize, usize)> = if config.strategy == "s5" {
        config.budget.halving.clone()
    } else {
        let splits = if config.budget.cv_fitness > 0 {
            0
        } else {
            config.budget.splits_per_gen
        };
        (0..config.budget.generations)
            .map(|_| (config.budget.population, splits))
            .collect()
    };
    if config.strict_pooled_search() && schedule.is_empty() {
        return Err("strict pooled search requires a non-empty schedule".to_string());
    }
    let expanded_vocab = config.vocab == "expanded";
    let gen0_target = schedule.first().map_or(config.budget.population, |s| s.0);
    if config.strict_pooled_search() && gen0.is_empty() {
        return Err("strict pooled search requires at least one valid seed genome".to_string());
    }
    if config.strict_pooled_search() && gen0.len() > gen0_target {
        return Err(format!(
            "strict pooled search has {} unique seeds for {gen0_target} generation-0 slots",
            gen0.len()
        ));
    }
    let mut population = fill_population(
        gen0,
        gen0_target,
        features,
        &format!("{ds}:{ms}:g0"),
        expanded_vocab,
    );

    // ---- DL draw bank for s3 (continual injection) ----
    let mut bank: Vec<Member> = config
        .dl_draw_bank
        .iter()
        .filter_map(|g| make_member(g, "dl_injected"))
        .filter(|m| !seen.contains(&m.semantic))
        .collect();

    // ---- generational loop ----
    let mut split_hashes: HashSet<String> = HashSet::new();
    for (generation, &(width, splits_per_gen)) in schedule.iter().enumerate() {
        // shrink (successive halving) or plain carry
        if population.len() > width {
            rank(&mut population);
            population.truncate(width);
        }
        if config.strict_pooled_search() && population.len() != width {
            return Err(format!(
                "generation {generation} contains {} members; exact budget requires {width}",
                population.len()
            ));
        }
        let mut invalid_members = 0usize;
        // V2: every member receives exactly one complete pooled OOF score from
        // the current generation's common two-fold bank. No member carries a
        // longer fitness history merely because it survived another generation.
        if config.strict_pooled_search() {
            let folds = kfold(
                y,
                config.budget.cv_fitness,
                seed_from(&format!("screen:{ds}:{ms}:cvgen:{generation}")),
            );
            validate_strict_fold_plan(y.len(), &folds)?;
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
            let outcomes: Vec<Option<f64>> = population
                .par_iter()
                .map(|member| {
                    let predictions: Vec<Result<Vec<f64>, EvalFailure>> = (0..folds.len())
                        .map(|f| {
                            eval_predictions(
                                train_path,
                                &member.canonical,
                                &fold_train[f],
                                &folds[f],
                            )
                        })
                        .collect();
                    let prediction_views: Vec<Option<&[f64]>> = predictions
                        .iter()
                        .map(|result| result.as_ref().ok().map(Vec::as_slice))
                        .collect();
                    complete_pooled_oof_r2(y, &folds, &prediction_views)
                })
                .collect();
            invalid_members = replace_generation_outcomes(&mut population, outcomes);
            if invalid_members == population.len() {
                return Err(format!(
                    "generation {generation} produced no complete member fitness"
                ));
            }
            evaluations += population.len() * folds.len();
        // V1 replay: fresh split(s) or CV folds append to member history.
        } else if config.budget.cv_fitness > 0 {
            let folds = kfold(
                y,
                usable_folds(y.len(), config.budget.cv_fitness),
                seed_from(&format!("screen:{ds}:{ms}:cvgen:{generation}")),
            );
            for f in 0..folds.len() {
                let valid_ids = &folds[f];
                let train_ids: Vec<usize> = folds
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != f)
                    .flat_map(|(_, ids)| ids.iter().copied())
                    .collect();
                let scores: Vec<f64> = population
                    .par_iter()
                    .map(|m| eval_with_timeout(train_path, &m.canonical, &train_ids, valid_ids))
                    .collect();
                for (m, s) in population.iter_mut().zip(scores) {
                    m.scores.push(s);
                    evaluations += 1;
                }
            }
        } else {
            for s in 0..splits_per_gen.max(1) {
                let seed = seed_from(&format!("screen:{ds}:{ms}:split:{generation}:{s}"));
                let (train_ids, valid_ids) = split_8020(y, seed);
                let (eval_path, train_ids) = if config.perturb_fitness {
                    match perturbed_generation_csv(
                        x,
                        y,
                        &train_ids,
                        generation,
                        &format!("{ds}:{ms}"),
                    ) {
                        Ok(v) => v,
                        Err(_) => (train_path.to_string(), train_ids),
                    }
                } else {
                    (train_path.to_string(), train_ids)
                };
                // train_only_study discipline: refuse silently-identical splits
                let mut h = Sha256::new();
                for i in &valid_ids {
                    h.update(i.to_le_bytes());
                }
                let hh = format!("{:x}", h.finalize());
                if !split_hashes.insert(hh) {
                    continue;
                }
                let scores: Vec<f64> = population
                    .par_iter()
                    .map(|m| eval_with_timeout(&eval_path, &m.canonical, &train_ids, &valid_ids))
                    .collect();
                for (m, sc) in population.iter_mut().zip(scores) {
                    m.scores.push(sc);
                    evaluations += 1;
                }
                if eval_path.starts_with(".perturbed-") {
                    let _ = fs::remove_file(&eval_path);
                }
            }
        }
        rank(&mut population);
        stats.push(GenerationStat {
            generation,
            evaluations,
            best_fitness: population
                .first()
                .map_or(-1.0, |m| robust_fitness(&m.scores)),
            invalid_members,
        });
        // breed the next generation (skip after the final evaluation round)
        if generation + 1 == schedule.len() {
            break;
        }
        let next_width = schedule[generation + 1].0;
        let elites = config.budget.elites.min(population.len());
        let mut next: Vec<Member> = population.iter().take(elites).cloned().collect();
        let mut next_seen: HashSet<String> = next.iter().map(|m| m.semantic.clone()).collect();
        // s3: fresh DL draws take offspring slots each generation
        let mut injected = 0usize;
        while injected < config.budget.dl_inject_per_gen && !bank.is_empty() {
            let m = bank.remove(0);
            if next_seen.insert(m.semantic.clone()) {
                next.push(m);
                injected += 1;
            }
        }
        let mut slot = 0usize;
        while next.len() < next_width && slot < next_width * 40 {
            if let Some(child) = breed_child(
                &population,
                population.len().min(width),
                features,
                &format!("{ds}:{ms}:g{}:{slot}", generation + 1),
                &next_seen,
                expanded_vocab,
            ) {
                next_seen.insert(child.semantic.clone());
                next.push(child);
            }
            slot += 1;
        }
        if config.strict_pooled_search() && next.len() != next_width {
            return Err(format!(
                "generation {} could produce only {} of {next_width} required members",
                generation + 1,
                next.len()
            ));
        }
        population = next;
    }
    rank(&mut population);
    if config.strict_pooled_search() {
        let expected = expected_search_evaluations(&schedule, config.budget.cv_fitness);
        if evaluations != expected {
            return Err(format!(
                "search evaluation count {evaluations} does not match exact budget {expected}"
            ));
        }
    }
    Ok((population, stats, evaluations))
}

fn final_cv(
    config: &Config,
    train_path: &str,
    x: &Rows,
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
        let fold_preds: Vec<((usize, usize), Result<Vec<f64>, EvalFailure>)> = pairs
            .par_iter()
            .map(|&(g, f)| {
                (
                    (g, f),
                    eval_predictions(train_path, &top[g].canonical, &fold_train[f], &folds[f]),
                )
            })
            .collect();
        for g in 0..top.len() {
            let mut oof_true: Vec<f64> = Vec::new();
            let mut oof_pred: Vec<f64> = Vec::new();
            let mut prediction_views: Vec<Option<&[f64]>> = Vec::with_capacity(folds.len());
            let mut failures = 0usize;
            for f in 0..folds.len() {
                let entry = fold_preds
                    .iter()
                    .find(|((gg, ff), _)| *gg == g && *ff == f)
                    .map(|(_, r)| r);
                match entry {
                    Some(Ok(pred)) if pred.len() == folds[f].len() => {
                        prediction_views.push(Some(pred.as_slice()));
                        for (k, &row) in folds[f].iter().enumerate() {
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
            top[g].cv_failed_folds = failures;
            top[g].cv7_mean_r2 = if config.strict_pooled_search() {
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
    #[allow(unreachable_code)]
    if false {
        for gene in top.iter_mut() {
            let Ok(genome) = GenomeIr::parse(&gene.canonical) else {
                gene.cv7_mean_r2 = None;
                continue;
            };
            let mut scores = Vec::new();
            for f in 0..folds.len() {
                let valid_ids = &folds[f];
                let train_ids: Vec<usize> = folds
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != f)
                    .flat_map(|(_, ids)| ids.iter().copied())
                    .collect();
                let (tx, ty) = subset(x, y, &train_ids);
                let (vx, vy) = subset(x, y, valid_ids);
                evals += 1;
                match model::fit_predict(&genome, &tx, &ty, &vx) {
                    Ok(pred) => scores.push(r2(&vy, &pred).map_or(-1.0, |v| v.max(-1.0))),
                    Err(_) => scores.push(-1.0),
                }
            }
            gene.cv7_mean_r2 = Some(scores.iter().sum::<f64>() / scores.len() as f64);
        }
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
        let outs: Vec<((usize, usize), Result<Vec<f64>, EvalFailure>)> = pairs
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
            for f in 0..folds.len() {
                if let Some((_, Ok(p))) = outs.iter().find(|((gg, ff), _)| *gg == g && *ff == f) {
                    if p.len() == folds[f].len() {
                        prediction_views.push(Some(p.as_slice()));
                        for (k, &row) in folds[f].iter().enumerate() {
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
        for g in 0..top.len() {
            if oof_preds[g].iter().all(|v| !v.is_finite()) {
                continue;
            }
            let k = picks.len() as f64;
            let cand: Vec<f64> = (0..n)
                .map(|i| {
                    let a = if mix[i].is_finite() { mix[i] * k } else { 0.0 };
                    let b = if oof_preds[g][i].is_finite() {
                        oof_preds[g][i]
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

#[cfg(test)]
mod tests {
    use super::*;

    fn member(name: &str, scores: Vec<f64>) -> Member {
        Member {
            canonical: name.to_string(),
            semantic: name.to_string(),
            complexity: 1,
            origin: "fixture".to_string(),
            scores,
            fitness_complete: false,
        }
    }

    fn strict_config() -> Config {
        Config {
            schema: SCHEMA_CONFIG_V2.to_string(),
            strategy: "fixture".to_string(),
            dataset_hash: "development-fixture".to_string(),
            master_seed: 101,
            dl_genes: vec![BASELINE_SEEDS[0].to_string()],
            dl_draw_bank: Vec::new(),
            global_prior: Vec::new(),
            neighbor_prior: Vec::new(),
            budget: Budget {
                population: 40,
                generations: 25,
                splits_per_gen: 0,
                elites: 3,
                cv_fitness: 2,
                dl_inject_per_gen: 0,
                halving: Vec::new(),
            },
            final_cv_folds: 7,
            final_top_k: 10,
            perturb_fitness: false,
            vocab: "expanded".to_string(),
            baseline_seeds: true,
            pooled_oof: true,
            winner_rule: "rcv_1se".to_string(),
            drop_failed_folds: false,
            se_band: 0.5,
        }
    }

    #[test]
    fn v2_config_binds_strict_selection_semantics() {
        let config = strict_config();
        config.validate().unwrap();

        let mut incomplete = strict_config();
        incomplete.drop_failed_folds = true;
        assert!(incomplete.validate().is_err());

        let mut winner_curse = strict_config();
        winner_curse.winner_rule = "cv7".to_string();
        assert!(winner_curse.validate().is_err());
    }

    #[test]
    fn v1_config_retains_replay_compatibility() {
        let mut config = strict_config();
        config.schema = SCHEMA_CONFIG_V1.to_string();
        config.budget.cv_fitness = 0;
        config.pooled_oof = false;
        config.drop_failed_folds = true;
        config.winner_rule = "cv7".to_string();
        config.perturb_fitness = true;
        config.validate().unwrap();
    }

    #[test]
    fn strict_fold_plan_rejects_nine_row_cv7() {
        let y: Vec<f64> = (0..9).map(|value| value as f64).collect();
        let folds = kfold(&y, 7, 17);
        assert!(validate_strict_fold_plan(y.len(), &folds).is_err());
    }

    #[test]
    fn strict_fold_plan_rejects_one_row_holds() {
        let y: Vec<f64> = (0..40).map(|value| value as f64).collect();
        let folds = kfold(&y, 39, 18);
        let error = validate_strict_fold_plan(y.len(), &folds).unwrap_err();
        assert!(error.contains("hold rows"), "{error}");
    }

    #[test]
    fn strict_fold_plan_rejects_too_few_fit_rows() {
        let y: Vec<f64> = (0..60).map(|value| value as f64).collect();
        let folds = kfold(&y, 2, 19);
        let error = validate_strict_fold_plan(y.len(), &folds).unwrap_err();
        assert!(error.contains("fit rows"), "{error}");
    }

    #[test]
    fn strict_fold_plan_accepts_complete_feasible_plan() {
        let y: Vec<f64> = (0..80).map(|value| value as f64).collect();
        let folds = kfold(&y, 2, 23);
        validate_strict_fold_plan(y.len(), &folds).unwrap();
    }

    #[test]
    fn complete_pooled_score_restores_original_row_order() {
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let folds = vec![vec![0, 2], vec![1, 3]];
        let first = vec![0.1, 2.1];
        let second = vec![0.9, 3.2];
        let views = vec![Some(first.as_slice()), Some(second.as_slice())];
        let actual = complete_pooled_oof_r2(&y, &folds, &views).unwrap();
        let expected = r2(&y, &[0.1, 0.9, 2.1, 3.2]).unwrap();
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn complete_pooled_score_fails_closed_on_missing_fold() {
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let folds = vec![vec![0, 2], vec![1, 3]];
        let first = vec![0.1, 2.1];
        let views = vec![Some(first.as_slice()), None];
        assert!(complete_pooled_oof_r2(&y, &folds, &views).is_none());
    }

    #[test]
    fn complete_pooled_score_fails_closed_on_duplicate_row() {
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let folds = vec![vec![0, 2], vec![0, 3]];
        let first = vec![0.1, 2.1];
        let second = vec![0.2, 3.2];
        let views = vec![Some(first.as_slice()), Some(second.as_slice())];
        assert!(complete_pooled_oof_r2(&y, &folds, &views).is_none());
    }

    #[test]
    fn strict_generation_scores_replace_member_history() {
        let mut population = vec![
            member("old-elite", vec![0.1, 0.2, 0.3]),
            member("new-child", Vec::new()),
        ];
        let invalid = replace_generation_outcomes(&mut population, vec![Some(0.4), Some(0.5)]);
        assert_eq!(invalid, 0);
        assert_eq!(population[0].scores, vec![0.4]);
        assert_eq!(population[1].scores, vec![0.5]);
        assert!(population.iter().all(|member| member.fitness_complete));
    }

    #[test]
    fn incomplete_fitness_cannot_win_a_clip_floor_tie() {
        let mut valid = member("valid", vec![-1.0]);
        valid.fitness_complete = true;
        let invalid = member("invalid-but-simpler", vec![-1.0]);
        let mut population = vec![invalid, valid];
        rank(&mut population);
        assert_eq!(population[0].semantic, "valid");
    }

    #[test]
    fn forty_by_twenty_five_by_two_is_exactly_two_thousand_fits() {
        let schedule = vec![(40, 0); 25];
        assert_eq!(expected_search_evaluations(&schedule, 2), 2_000);
    }
}

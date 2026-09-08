//! Fast, deterministic self-play policy primitives.
//!
//! This module intentionally contains no service, database, or process I/O.
//! Callers own persistence and orchestration; these functions own the policy
//! formulas and fail closed on malformed numeric inputs.

use serde_json::json;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const SELFPLAY_SELECTION_BIAS: &str = "test-exact-reused";
pub const RAMP_MID_SLOTS: i64 = 30;
pub const RAMP_FINAL_SLOTS: i64 = 40;
pub const RAMP_MID_DURATION_MS: i64 = 10 * 60_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyError(pub String);

impl Display for PolicyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for PolicyError {}

pub type PolicyResult<T> = Result<T, PolicyError>;

fn error(message: impl Into<String>) -> PolicyError {
    PolicyError(message.into())
}

fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(sha256_bytes(bytes))
}

fn validate_finite(value: f64, label: &str) -> PolicyResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(error(format!("{label} must be finite")))
    }
}

/// The checkpoint-fresh queue policy and observed p95.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueueTargets {
    pub target: i64,
    pub hard_cap: i64,
    pub p95_jobs_per_minute: f64,
}

/// Calculate the nearest-rank p95 and the work-conserving queue horizon.
pub fn dynamic_queue_targets(
    jobs_per_minute: &[f64],
    effective_cluster_slots: i64,
) -> PolicyResult<QueueTargets> {
    if effective_cluster_slots < 0 {
        return Err(error("effective cluster slots cannot be negative"));
    }
    let mut values = Vec::with_capacity(jobs_per_minute.len());
    for &value in jobs_per_minute {
        values.push(validate_finite(value, "jobs per minute")?.max(0.0));
    }
    values.sort_by(f64::total_cmp);
    let p95 = if values.is_empty() {
        0.0
    } else {
        let rank = ((0.95 * values.len() as f64).ceil() as usize).max(1);
        values[rank - 1]
    };
    let slot_horizon = ((i128::from(effective_cluster_slots) * 5 + 3) / 4).min(4_096) as i64;
    Ok(QueueTargets {
        target: slot_horizon.max(2_048),
        hard_cap: 4_096,
        p95_jobs_per_minute: p95,
    })
}

pub fn queue_refill_needed(queued_jobs: i64, target: i64, hard_cap: i64) -> PolicyResult<i64> {
    if target < 0 || hard_cap < target {
        return Err(error(
            "queue hard cap must be at least the nonnegative queue target",
        ));
    }
    if queued_jobs >= target {
        return Ok(0);
    }
    Ok((target - queued_jobs).min(hard_cap - queued_jobs))
}

/// Assign complete source families to stable train/validation splits.
pub fn family_disjoint_split(
    dataset_families: &BTreeMap<String, String>,
    validation_fraction: f64,
    seed: u64,
) -> PolicyResult<BTreeMap<String, String>> {
    if !(0.0..1.0).contains(&validation_fraction) || validation_fraction == 0.0 {
        return Err(error("validation_fraction must be between zero and one"));
    }
    let mut families = BTreeMap::<String, Vec<String>>::new();
    for (dataset, family) in dataset_families {
        if dataset.is_empty() || family.is_empty() {
            return Err(error("dataset and family must be nonempty"));
        }
        families
            .entry(family.clone())
            .or_default()
            .push(dataset.clone());
    }
    if dataset_families.len() < 2 || families.is_empty() {
        return Err(error(
            "family topology cannot produce nonempty disjoint splits",
        ));
    }
    let mut ordered = families.keys().cloned().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        family_order_digest(seed, left)
            .cmp(&family_order_digest(seed, right))
            .then_with(|| left.cmp(right))
    });
    let rounded = (dataset_families.len() as f64 * validation_fraction).round_ties_even() as usize;
    let target = (dataset_families.len() - 1).min(rounded.max(1));
    let mut validation = BTreeSet::new();
    let mut remaining = target;
    for family in ordered {
        let mut members = families[&family].clone();
        members.sort();
        if remaining == 0 {
            break;
        }
        if members.len() <= remaining || validation.is_empty() {
            remaining = remaining.saturating_sub(members.len());
            validation.extend(members);
        }
    }
    if validation.is_empty() || validation.len() == dataset_families.len() {
        return Err(error(
            "family topology cannot produce nonempty disjoint splits",
        ));
    }
    Ok(dataset_families
        .keys()
        .map(|dataset| {
            let split = if validation.contains(dataset) {
                "validation"
            } else {
                "train"
            };
            (dataset.clone(), split.to_owned())
        })
        .collect())
}

fn family_order_digest(seed: u64, family: &str) -> [u8; 32] {
    sha256_bytes(format!("{seed}|{family}").as_bytes())
}

#[derive(Clone, Debug, PartialEq)]
pub struct NumericDataset {
    pub features: Vec<Vec<f64>>,
    pub targets: Vec<f64>,
}

/// Parse a numeric CSV whose final column is the target.
pub fn parse_last_target_csv(csv: &str) -> PolicyResult<NumericDataset> {
    let mut features = Vec::new();
    let mut targets = Vec::new();
    for (index, line) in csv.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_fields(line)
            .map_err(|message| error(format!("CSV row {}: {message}", index + 1)))?;
        let mut values = Vec::with_capacity(fields.len());
        for field in fields {
            let value = field
                .trim()
                .parse::<f64>()
                .map_err(|_| error(format!("CSV row {} contains a nonnumeric value", index + 1)))?;
            if !value.is_finite() {
                return Err(error(format!(
                    "CSV row {} contains a non-finite value",
                    index + 1
                )));
            }
            values.push(value);
        }
        if values.len() < 2 {
            return Err(error(format!(
                "CSV row {} must contain a feature and target",
                index + 1
            )));
        }
        let target = values.pop().expect("length checked");
        features.push(values);
        targets.push(target);
    }
    if features.is_empty() {
        return Err(error("dataset is empty"));
    }
    Ok(NumericDataset { features, targets })
}

fn parse_csv_fields(line: &str) -> Result<Vec<String>, &'static str> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = line.trim_end_matches('\r').chars().peekable();
    while let Some(character) = chars.next() {
        match character {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(std::mem::take(&mut field));
            }
            _ => field.push(character),
        }
    }
    if quoted {
        return Err("unterminated quoted field");
    }
    fields.push(field);
    Ok(fields)
}

pub trait AdaptiveEstimator {
    fn fit(&mut self, features: &[Vec<f64>], targets: &[f64]) -> PolicyResult<()>;
    fn predict(&self, features: &[Vec<f64>]) -> PolicyResult<Vec<f64>>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveFitness {
    pub task: String,
    pub metric: &'static str,
    pub score: f64,
    pub secondary_metric: &'static str,
    pub secondary_score: f64,
    pub fit_split: &'static str,
    pub fitness_split: &'static str,
    pub cross_validation_fitness: bool,
    pub test_adaptive: bool,
    pub benchmark_valid: bool,
    pub selection_bias: &'static str,
    pub valid: bool,
}

/// Fit exactly once on train.csv and calculate fitness only on test.csv.
pub fn evaluate_adaptive<E: AdaptiveEstimator>(
    task: &str,
    train_csv: &str,
    test_csv: &str,
    estimator: &mut E,
) -> PolicyResult<AdaptiveFitness> {
    let train = parse_last_target_csv(train_csv)?;
    let test = parse_last_target_csv(test_csv)?;
    if train.features[0].len() != test.features[0].len() {
        return Err(error("train/test feature widths differ"));
    }
    estimator.fit(&train.features, &train.targets)?;
    let predictions = estimator.predict(&test.features)?;
    if predictions.len() != test.targets.len() || predictions.iter().any(|value| !value.is_finite())
    {
        return Err(error("adaptive predictions are invalid"));
    }
    let (metric, score, secondary_metric, secondary_score) = match task {
        "regression" => {
            let score = r2_score(&test.targets, &predictions)?;
            let mean_squared_error = test
                .targets
                .iter()
                .zip(&predictions)
                .map(|(target, prediction)| (target - prediction).powi(2))
                .sum::<f64>()
                / test.targets.len() as f64;
            ("r2", score, "rmse", mean_squared_error.sqrt())
        }
        "binary" => {
            if train
                .targets
                .iter()
                .chain(&test.targets)
                .any(|target| *target != 0.0 && *target != 1.0)
            {
                return Err(error("binary adaptive targets must be zero/one"));
            }
            (
                "auc",
                auc_score(&test.targets, &predictions)?,
                "log_loss",
                log_loss(&test.targets, &predictions)?,
            )
        }
        _ => return Err(error(format!("unsupported adaptive task {task}"))),
    };
    Ok(AdaptiveFitness {
        task: task.to_owned(),
        metric,
        score,
        secondary_metric,
        secondary_score,
        fit_split: "train.csv",
        fitness_split: "test.csv",
        cross_validation_fitness: false,
        test_adaptive: true,
        benchmark_valid: false,
        selection_bias: SELFPLAY_SELECTION_BIAS,
        valid: score.is_finite() && secondary_score.is_finite(),
    })
}

pub fn r2_score(targets: &[f64], predictions: &[f64]) -> PolicyResult<f64> {
    validate_score_vectors(targets, predictions)?;
    let mean = targets.iter().sum::<f64>() / targets.len() as f64;
    let denominator = targets
        .iter()
        .map(|target| (target - mean).powi(2))
        .sum::<f64>();
    if denominator == 0.0 {
        return Ok(f64::NAN);
    }
    Ok(1.0
        - targets
            .iter()
            .zip(predictions)
            .map(|(target, prediction)| (target - prediction).powi(2))
            .sum::<f64>()
            / denominator)
}

pub fn auc_score(targets: &[f64], predictions: &[f64]) -> PolicyResult<f64> {
    validate_score_vectors(targets, predictions)?;
    if targets
        .iter()
        .any(|target| *target != 0.0 && *target != 1.0)
    {
        return Err(error("AUC targets must be zero/one"));
    }
    let positives = targets.iter().filter(|target| **target == 1.0).count();
    let negatives = targets.len() - positives;
    if positives == 0 || negatives == 0 {
        return Ok(f64::NAN);
    }
    let mut ordered = predictions
        .iter()
        .copied()
        .zip(targets.iter().copied())
        .collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut rank_sum = 0.0;
    let mut offset = 0;
    while offset < ordered.len() {
        let mut end = offset + 1;
        while end < ordered.len() && ordered[end].0 == ordered[offset].0 {
            end += 1;
        }
        let average_rank = (offset + 1 + end) as f64 / 2.0;
        let tied_positives = ordered[offset..end]
            .iter()
            .filter(|row| row.1 == 1.0)
            .count();
        rank_sum += average_rank * tied_positives as f64;
        offset = end;
    }
    Ok((rank_sum - positives as f64 * (positives + 1) as f64 / 2.0)
        / (positives * negatives) as f64)
}

pub fn log_loss(targets: &[f64], predictions: &[f64]) -> PolicyResult<f64> {
    validate_score_vectors(targets, predictions)?;
    if targets
        .iter()
        .any(|target| *target != 0.0 && *target != 1.0)
    {
        return Err(error("log-loss targets must be zero/one"));
    }
    let epsilon = 1e-15;
    let total = targets
        .iter()
        .zip(predictions)
        .map(|(target, prediction)| {
            let probability = prediction.clamp(epsilon, 1.0 - epsilon);
            -(target * probability.ln() + (1.0 - target) * (1.0 - probability).ln())
        })
        .sum::<f64>();
    Ok(total / targets.len() as f64)
}

fn validate_score_vectors(targets: &[f64], predictions: &[f64]) -> PolicyResult<()> {
    if targets.is_empty() || targets.len() != predictions.len() {
        return Err(error(
            "targets and predictions must have the same nonzero length",
        ));
    }
    if targets
        .iter()
        .chain(predictions)
        .any(|value| !value.is_finite())
    {
        return Err(error("targets and predictions must be finite"));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct RankingObservation {
    pub task: String,
    pub dataset_hash: String,
    pub semantic_sha256: String,
    pub score: Option<f64>,
    pub secondary_score: Option<f64>,
    pub valid: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RankedObservation {
    pub observation: RankingObservation,
    pub within_dataset_rank: Option<usize>,
    pub within_dataset_percentile: Option<f64>,
    pub regret_to_incumbent: Option<f64>,
    pub incumbent_available: bool,
}

pub fn attach_ranking_labels(
    rows: &[RankingObservation],
    incumbents: Option<&BTreeMap<(String, String), f64>>,
) -> Vec<RankedObservation> {
    let mut grouped = BTreeMap::<(String, String), Vec<(usize, &RankingObservation)>>::new();
    for (index, row) in rows.iter().enumerate() {
        grouped
            .entry((row.task.clone(), row.dataset_hash.clone()))
            .or_default()
            .push((index, row));
    }
    let mut output = Vec::with_capacity(rows.len());
    for (identity, values) in grouped {
        let incumbent = incumbents.and_then(|values| values.get(&identity)).copied();
        let mut valid = values
            .iter()
            .copied()
            .filter(|(_, row)| row.valid && row.score.is_some_and(f64::is_finite))
            .collect::<Vec<_>>();
        valid.sort_by(|(_, left), (_, right)| {
            right
                .score
                .expect("filtered")
                .total_cmp(&left.score.expect("filtered"))
                .then_with(|| {
                    if identity.0 == "binary" {
                        left.secondary_score
                            .unwrap_or(0.0)
                            .total_cmp(&right.secondary_score.unwrap_or(0.0))
                    } else {
                        Ordering::Equal
                    }
                })
                .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
        });
        let count = valid.len();
        let valid_indexes = valid
            .iter()
            .map(|(index, _)| *index)
            .collect::<BTreeSet<_>>();
        for (offset, (_, row)) in valid.into_iter().enumerate() {
            let rank = offset + 1;
            let percentile = if count == 1 {
                1.0
            } else {
                (count - rank) as f64 / (count - 1) as f64
            };
            output.push(RankedObservation {
                observation: row.clone(),
                within_dataset_rank: Some(rank),
                within_dataset_percentile: Some(percentile),
                regret_to_incumbent: incumbent.map(|value| value - row.score.expect("filtered")),
                incumbent_available: incumbent.is_some(),
            });
        }
        for (index, row) in values {
            if !valid_indexes.contains(&index) {
                output.push(RankedObservation {
                    observation: row.clone(),
                    within_dataset_rank: None,
                    within_dataset_percentile: None,
                    regret_to_incumbent: None,
                    incumbent_available: incumbent.is_some(),
                });
            }
        }
    }
    output
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn compiled_cache_key(
    task: &str,
    train_sha256: &str,
    semantic_sha256: &str,
    evaluator_protocol_sha256: &str,
    compiler_bundle_sha256: &str,
    target_triple: &str,
) -> PolicyResult<String> {
    for (label, value) in [
        ("dataset_train_sha256", train_sha256),
        ("semantic_sha256", semantic_sha256),
        ("evaluator_protocol_sha256", evaluator_protocol_sha256),
        ("compiler_bundle_sha256", compiler_bundle_sha256),
    ] {
        if !valid_sha256(value) {
            return Err(error(format!("invalid compile-cache identity: {label}")));
        }
    }
    if !matches!(task, "regression" | "binary" | "timeseries" | "multiclass")
        || target_triple.is_empty()
    {
        return Err(error("invalid compile-cache task or target"));
    }
    let material = json!({
        "task": task,
        "dataset_train_sha256": train_sha256,
        "semantic_sha256": semantic_sha256,
        "evaluator_protocol_sha256": evaluator_protocol_sha256,
        "compiler_bundle_sha256": compiler_bundle_sha256,
        "target_triple": target_triple,
    });
    let encoded = serde_json::to_vec(&material)
        .map_err(|cause| error(format!("encode cache key: {cause}")))?;
    Ok(sha256_hex(&encoded))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PanelDataset {
    pub dataset_hash: String,
    pub source_family: String,
}

pub fn family_diverse_panel(
    datasets: &[PanelDataset],
    size: usize,
    seed: u64,
) -> PolicyResult<Vec<PanelDataset>> {
    let mut by_family = BTreeMap::<String, Vec<PanelDataset>>::new();
    for row in datasets {
        if row.dataset_hash.is_empty() || row.source_family.is_empty() {
            return Err(error("panel dataset identities must be nonempty"));
        }
        by_family
            .entry(row.source_family.clone())
            .or_default()
            .push(row.clone());
    }
    let mut families = by_family.keys().cloned().collect::<Vec<_>>();
    families.sort_by(|left, right| {
        family_order_digest(seed, left)
            .cmp(&family_order_digest(seed, right))
            .then_with(|| left.cmp(right))
    });
    let mut selected = Vec::with_capacity(size);
    for family in &families {
        let mut values = by_family[family].clone();
        values.sort_by(|left, right| left.dataset_hash.cmp(&right.dataset_hash));
        selected.push(values[0].clone());
        if selected.len() == size {
            return Ok(selected);
        }
    }
    for family in &families {
        let mut values = by_family[family].clone();
        values.sort_by(|left, right| left.dataset_hash.cmp(&right.dataset_hash));
        for row in values.into_iter().skip(1) {
            if selected.len() == size {
                return Ok(selected);
            }
            selected.push(row);
        }
    }
    if selected.len() != size {
        return Err(error(format!("replay panel requires {size} datasets")));
    }
    Ok(selected)
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompileCandidate {
    pub task: String,
    pub dataset_hash: String,
    pub semantic_sha256: String,
    pub score: f64,
    pub valid: bool,
}

pub fn compiled_finalists(
    rows: &[CompileCandidate],
    incumbents: &BTreeMap<(String, String), f64>,
) -> Vec<CompileCandidate> {
    let mut grouped = BTreeMap::<(String, String), Vec<CompileCandidate>>::new();
    for row in rows {
        if row.valid && row.score.is_finite() {
            grouped
                .entry((row.task.clone(), row.dataset_hash.clone()))
                .or_default()
                .push(row.clone());
        }
    }
    let mut result = Vec::new();
    for (identity, mut ranked) in grouped {
        ranked.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
        });
        let incumbent = incumbents.get(&identity).copied();
        result.extend(
            ranked
                .into_iter()
                .enumerate()
                .filter(|(index, row)| {
                    *index < 2 || incumbent.is_some_and(|value| row.score > value)
                })
                .take(2)
                .map(|(_, row)| row),
        );
    }
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParityResult {
    pub prediction_bit_parity: bool,
    pub score_bit_parity: bool,
    pub qualified: bool,
}

pub fn exact_parity(
    interpreted: &[f64],
    compiled: &[f64],
    interpreted_score: f64,
    compiled_score: f64,
) -> ParityResult {
    let prediction_bit_parity = interpreted.len() == compiled.len()
        && interpreted
            .iter()
            .zip(compiled)
            .all(|(left, right)| left.to_bits() == right.to_bits());
    let score_bit_parity = interpreted_score.to_bits() == compiled_score.to_bits();
    ParityResult {
        prediction_bit_parity,
        score_bit_parity,
        qualified: prediction_bit_parity && score_bit_parity,
    }
}

pub fn node_slot_target(effective_cores: i64, multiplier: f64) -> PolicyResult<i64> {
    if effective_cores < 0 || !multiplier.is_finite() || multiplier <= 0.0 {
        return Err(error("invalid effective cores or CPU-job slot multiplier"));
    }
    let target = effective_cores as f64 * multiplier;
    if !target.is_finite() || target > i64::MAX as f64 {
        return Err(error("CPU-job slot target overflows"));
    }
    Ok(target.floor() as i64)
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodeCapacity {
    pub node: String,
    pub physical_cores: i64,
    pub effective_cores: i64,
    pub cpu_job_slot_multiplier: f64,
    pub slot_target: i64,
    pub running_slots: i64,
}

pub fn node_capacity(
    node: &str,
    physical_cores: i64,
    effective_cores: i64,
    running_slots: i64,
    multiplier: f64,
) -> PolicyResult<NodeCapacity> {
    if physical_cores < effective_cores
        || effective_cores < 0
        || running_slots < 0
        || node.is_empty()
    {
        return Err(error("invalid node capacity counters"));
    }
    Ok(NodeCapacity {
        node: node.to_owned(),
        physical_cores,
        effective_cores,
        cpu_job_slot_multiplier: multiplier,
        slot_target: node_slot_target(effective_cores, multiplier)?,
        running_slots,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PressureSample {
    pub cpu_psi: f64,
    pub memory_psi: f64,
    pub io_psi: f64,
    pub available_memory_fraction: Option<f64>,
    pub failure_rate: f64,
    pub p95_runtime_seconds: f64,
    pub throughput_per_slot: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PressureBaseline {
    pub failure_rate: f64,
    pub p95_runtime_seconds: f64,
    pub throughput_per_slot: f64,
}

pub fn unsafe_sample(
    sample: &PressureSample,
    baseline: &PressureBaseline,
) -> PolicyResult<Vec<&'static str>> {
    for (name, value) in [
        ("cpu_psi", sample.cpu_psi),
        ("memory_psi", sample.memory_psi),
        ("io_psi", sample.io_psi),
        ("failure_rate", sample.failure_rate),
        ("p95_runtime_seconds", sample.p95_runtime_seconds),
        ("throughput_per_slot", sample.throughput_per_slot),
        ("baseline failure rate", baseline.failure_rate),
        ("baseline p95 runtime seconds", baseline.p95_runtime_seconds),
        ("baseline throughput per slot", baseline.throughput_per_slot),
    ] {
        validate_finite(value, name)?;
    }
    if sample
        .available_memory_fraction
        .is_some_and(|value| !value.is_finite())
    {
        return Err(error("available memory fraction must be finite"));
    }
    let mut reasons = Vec::new();
    if sample.cpu_psi >= 0.25 {
        reasons.push("cpu_psi");
    }
    if sample.memory_psi >= 0.05 {
        reasons.push("memory_psi");
    }
    if sample.io_psi >= 0.10 {
        reasons.push("io_psi");
    }
    if sample.available_memory_fraction.unwrap_or(1.0) < 0.15 {
        reasons.push("available_memory");
    }
    if sample.failure_rate - baseline.failure_rate > 0.0025 {
        reasons.push("failure_rate_regression");
    }
    let runtime_regressed = baseline.p95_runtime_seconds > 0.0
        && sample.p95_runtime_seconds > 1.5 * baseline.p95_runtime_seconds;
    let throughput_regressed = baseline.throughput_per_slot > 0.0
        && sample.throughput_per_slot <= 0.8 * baseline.throughput_per_slot;
    if runtime_regressed && throughput_regressed {
        reasons.push("runtime_and_throughput_regression");
    }
    Ok(reasons)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RampStage {
    Baseline,
    Ramp30,
    Target40,
    RolledBack,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RampState {
    pub stage: RampStage,
    pub effective_cores: i64,
    pub multiplier: f64,
    pub slot_target: i64,
    pub stage_started_at_ms: Option<i64>,
    pub rollback_at_ms: Option<i64>,
    pub rollback_reasons: Vec<String>,
    pub baseline: PressureBaseline,
}

impl RampState {
    pub fn baseline(effective_cores: i64, baseline: PressureBaseline) -> PolicyResult<Self> {
        Ok(Self {
            stage: RampStage::Baseline,
            effective_cores,
            multiplier: 1.0,
            slot_target: node_slot_target(effective_cores, 1.0)?,
            stage_started_at_ms: None,
            rollback_at_ms: None,
            rollback_reasons: Vec::new(),
            baseline,
        })
    }
}

pub fn ramp_decision(
    state: &RampState,
    samples: &[PressureSample],
    now_at_ms: i64,
) -> PolicyResult<RampState> {
    let recent = samples.iter().rev().take(3).collect::<Vec<_>>();
    let unsafe_values = recent
        .iter()
        .map(|sample| unsafe_sample(sample, &state.baseline))
        .collect::<PolicyResult<Vec<_>>>()?;
    if unsafe_values.len() == 3 && unsafe_values.iter().all(|reasons| !reasons.is_empty()) {
        let mut reasons = unsafe_values
            .into_iter()
            .flatten()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        reasons.sort();
        reasons.dedup();
        let mut result = state.clone();
        result.stage = RampStage::RolledBack;
        result.multiplier = 1.0;
        result.slot_target = state.effective_cores;
        result.rollback_at_ms = Some(now_at_ms);
        result.rollback_reasons = reasons;
        return Ok(result);
    }
    let mut result = state.clone();
    match state.stage {
        RampStage::Baseline => {
            result.stage = RampStage::Ramp30;
            result.multiplier = 1.5;
            result.slot_target = RAMP_MID_SLOTS;
            result.stage_started_at_ms = Some(now_at_ms);
        }
        RampStage::Ramp30 => {
            let stage_started = state.stage_started_at_ms.unwrap_or(now_at_ms);
            if now_at_ms - stage_started >= RAMP_MID_DURATION_MS {
                result.stage = RampStage::Target40;
                result.multiplier = 2.0;
                result.slot_target = RAMP_FINAL_SLOTS;
                result.stage_started_at_ms = Some(now_at_ms);
            }
        }
        RampStage::Target40 | RampStage::RolledBack => {}
    }
    Ok(result)
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BurnSample {
    pub cluster_cpu_fraction: f64,
    pub pressure: PressureSample,
    pub baseline: PressureBaseline,
}

pub fn xbabe2_burn_eligible(samples: &[BurnSample], fresh_queue_depth: i64) -> PolicyResult<bool> {
    if samples.len() < 15 || fresh_queue_depth <= 0 {
        return Ok(false);
    }
    let recent = &samples[samples.len() - 15..];
    if recent
        .iter()
        .any(|sample| !sample.cluster_cpu_fraction.is_finite())
    {
        return Err(error("cluster CPU fraction must be finite"));
    }
    if recent
        .iter()
        .map(|sample| sample.cluster_cpu_fraction)
        .sum::<f64>()
        / 15.0
        >= 0.75
    {
        return Ok(false);
    }
    for sample in recent {
        if !unsafe_sample(&sample.pressure, &sample.baseline)?.is_empty() {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_policy_is_fixed_while_p95_remains_observable() {
        assert_eq!(
            dynamic_queue_targets(&[], 364).unwrap(),
            QueueTargets {
                target: 2_048,
                hard_cap: 4_096,
                p95_jobs_per_minute: 0.0,
            }
        );
        assert_eq!(
            dynamic_queue_targets(&[1_000.0], 1).unwrap(),
            QueueTargets {
                target: 2_048,
                hard_cap: 4_096,
                p95_jobs_per_minute: 1_000.0,
            }
        );
        assert_eq!(dynamic_queue_targets(&[], 2_000).unwrap().target, 2_500);
        assert_eq!(dynamic_queue_targets(&[], 4_000).unwrap().target, 4_096);
        let values = (1..=20).map(f64::from).collect::<Vec<_>>();
        assert_eq!(
            dynamic_queue_targets(&values, 0)
                .unwrap()
                .p95_jobs_per_minute,
            19.0
        );
        assert!(dynamic_queue_targets(&[f64::NAN], 1).is_err());
        assert!(dynamic_queue_targets(&[], -1).is_err());
    }

    #[test]
    fn refill_stops_at_target_and_rejects_inverted_cap() {
        assert_eq!(queue_refill_needed(200, 256, 512).unwrap(), 56);
        assert_eq!(queue_refill_needed(256, 256, 512).unwrap(), 0);
        assert!(queue_refill_needed(0, 513, 512).is_err());
    }

    #[test]
    fn family_split_is_deterministic_and_never_splits_a_family() {
        let families = BTreeMap::from([
            ("a".to_owned(), "shared".to_owned()),
            ("b".to_owned(), "shared".to_owned()),
            ("c".to_owned(), "other-c".to_owned()),
            ("d".to_owned(), "other-d".to_owned()),
        ]);
        let first = family_disjoint_split(&families, 0.25, 20260726).unwrap();
        let second = family_disjoint_split(&families, 0.25, 20260726).unwrap();
        assert_eq!(first, second);
        assert_eq!(first["a"], first["b"]);
        assert!(first.values().any(|split| split == "train"));
        assert!(first.values().any(|split| split == "validation"));
    }

    #[test]
    fn family_split_rejects_impossible_topology() {
        let one = BTreeMap::from([("a".to_owned(), "only".to_owned())]);
        assert!(family_disjoint_split(&one, 0.1, 1).is_err());
        let same = BTreeMap::from([
            ("a".to_owned(), "only".to_owned()),
            ("b".to_owned(), "only".to_owned()),
        ]);
        assert!(family_disjoint_split(&same, 0.1, 1).is_err());
    }

    #[test]
    fn csv_parser_accepts_quotes_and_last_target() {
        let dataset = parse_last_target_csv("\"1\",2,0\r\n3,4,1\n").unwrap();
        assert_eq!(dataset.features, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(dataset.targets, vec![0.0, 1.0]);
        assert!(parse_last_target_csv("x,1\n").is_err());
        assert!(parse_last_target_csv("1,NaN\n").is_err());
    }

    struct MeanEstimator {
        fitted: bool,
        mean: f64,
    }

    impl AdaptiveEstimator for MeanEstimator {
        fn fit(&mut self, _: &[Vec<f64>], targets: &[f64]) -> PolicyResult<()> {
            self.fitted = true;
            self.mean = targets.iter().sum::<f64>() / targets.len() as f64;
            Ok(())
        }

        fn predict(&self, features: &[Vec<f64>]) -> PolicyResult<Vec<f64>> {
            assert!(self.fitted);
            Ok(vec![self.mean; features.len()])
        }
    }

    #[test]
    fn adaptive_evaluation_fits_train_and_scores_test_without_cv() {
        let mut estimator = MeanEstimator {
            fitted: false,
            mean: 0.0,
        };
        let fitness =
            evaluate_adaptive("regression", "0,1\n1,3\n", "2,2\n3,4\n", &mut estimator).unwrap();
        assert!(estimator.fitted);
        assert_eq!(fitness.fit_split, "train.csv");
        assert_eq!(fitness.fitness_split, "test.csv");
        assert_eq!(fitness.metric, "r2");
        assert_eq!(fitness.secondary_metric, "rmse");
        assert!(!fitness.cross_validation_fitness);
        assert!(fitness.test_adaptive);
        assert!(!fitness.benchmark_valid);
    }

    #[test]
    fn binary_metrics_handle_ties_and_clamp_log_loss() {
        assert_eq!(
            auc_score(&[0.0, 1.0, 0.0, 1.0], &[0.1, 0.9, 0.5, 0.5]).unwrap(),
            0.875
        );
        assert!(log_loss(&[0.0, 1.0], &[-10.0, 10.0]).unwrap().is_finite());
        assert!(auc_score(&[1.0, 1.0], &[0.1, 0.2]).unwrap().is_nan());
    }

    #[test]
    fn ranking_uses_score_binary_log_loss_and_semantic_order() {
        let rows = vec![
            RankingObservation {
                task: "binary".into(),
                dataset_hash: "d".into(),
                semantic_sha256: "b".into(),
                score: Some(0.8),
                secondary_score: Some(0.4),
                valid: true,
            },
            RankingObservation {
                task: "binary".into(),
                dataset_hash: "d".into(),
                semantic_sha256: "a".into(),
                score: Some(0.8),
                secondary_score: Some(0.3),
                valid: true,
            },
            RankingObservation {
                task: "binary".into(),
                dataset_hash: "d".into(),
                semantic_sha256: "invalid".into(),
                score: None,
                secondary_score: None,
                valid: false,
            },
        ];
        let incumbents = BTreeMap::from([(("binary".into(), "d".into()), 0.95)]);
        let ranked = attach_ranking_labels(&rows, Some(&incumbents));
        assert_eq!(ranked[0].observation.semantic_sha256, "a");
        assert_eq!(ranked[0].within_dataset_rank, Some(1));
        assert_eq!(ranked[0].within_dataset_percentile, Some(1.0));
        assert!((ranked[0].regret_to_incumbent.unwrap() - 0.15).abs() < f64::EPSILON);
        assert_eq!(ranked[2].within_dataset_rank, None);
    }

    #[test]
    fn compile_cache_key_is_canonical_and_dataset_specific() {
        let key = compiled_cache_key(
            "regression",
            &"1".repeat(64),
            &"2".repeat(64),
            &"3".repeat(64),
            &"4".repeat(64),
            "x86_64-unknown-linux-gnu",
        )
        .unwrap();
        let changed = compiled_cache_key(
            "regression",
            &"5".repeat(64),
            &"2".repeat(64),
            &"3".repeat(64),
            &"4".repeat(64),
            "x86_64-unknown-linux-gnu",
        )
        .unwrap();
        assert_eq!(key.len(), 64);
        assert_ne!(key, changed);
        assert!(compiled_cache_key(
            "regression",
            "BAD",
            &"2".repeat(64),
            &"3".repeat(64),
            &"4".repeat(64),
            "target"
        )
        .is_err());
    }

    #[test]
    fn family_panel_takes_one_per_family_before_remainders() {
        let rows = (0..32)
            .flat_map(|family| {
                [0, 1].map(move |member| PanelDataset {
                    dataset_hash: format!("d-{family:02}-{member}"),
                    source_family: format!("f-{family:02}"),
                })
            })
            .collect::<Vec<_>>();
        let panel = family_diverse_panel(&rows, 32, 20260726).unwrap();
        assert_eq!(
            panel
                .iter()
                .map(|row| &row.source_family)
                .collect::<BTreeSet<_>>()
                .len(),
            32
        );
        assert!(family_diverse_panel(&rows[..3], 32, 1).is_err());
    }

    #[test]
    fn compile_finalists_are_bounded_at_two_per_dataset() {
        let rows = (0..4)
            .map(|index| CompileCandidate {
                task: "regression".into(),
                dataset_hash: "d".into(),
                semantic_sha256: format!("{index}"),
                score: index as f64 / 10.0,
                valid: true,
            })
            .collect::<Vec<_>>();
        let incumbents = BTreeMap::from([(("regression".into(), "d".into()), 0.15)]);
        let finalists = compiled_finalists(&rows, &incumbents);
        assert_eq!(finalists.len(), 2);
        assert_eq!(finalists[0].score, 0.3);
        assert_eq!(finalists[1].score, 0.2);
    }

    #[test]
    fn parity_is_bit_exact_including_signed_zero() {
        assert!(exact_parity(&[1.0], &[1.0], 0.5, 0.5).qualified);
        let mismatch = exact_parity(&[-0.0], &[0.0], 0.5, 0.5);
        assert!(!mismatch.prediction_bit_parity);
        assert!(!mismatch.qualified);
    }

    #[test]
    fn slot_targets_floor_multiplier_and_capacity_is_explicit() {
        assert_eq!(node_slot_target(20, 2.0).unwrap(), 40);
        assert_eq!(node_slot_target(3, 1.2).unwrap(), 3);
        let capacity = node_capacity("xbabe4", 24, 20, 13, 2.0).unwrap();
        assert_eq!(capacity.slot_target, 40);
        assert_eq!(capacity.physical_cores, 24);
        assert!(node_capacity("xbabe4", 19, 20, 0, 2.0).is_err());
    }

    #[test]
    fn pressure_thresholds_are_inclusive_except_failure_delta() {
        let baseline = PressureBaseline {
            failure_rate: 0.1,
            p95_runtime_seconds: 10.0,
            throughput_per_slot: 5.0,
        };
        let sample = PressureSample {
            cpu_psi: 0.25,
            memory_psi: 0.05,
            io_psi: 0.10,
            available_memory_fraction: Some(0.149),
            failure_rate: 0.1026,
            p95_runtime_seconds: 15.1,
            throughput_per_slot: 4.0,
        };
        assert_eq!(
            unsafe_sample(&sample, &baseline).unwrap(),
            vec![
                "cpu_psi",
                "memory_psi",
                "io_psi",
                "available_memory",
                "failure_rate_regression",
                "runtime_and_throughput_regression",
            ]
        );
    }

    #[test]
    fn ramp_advances_after_ten_minutes_and_rolls_back_after_three_samples() {
        let baseline = PressureBaseline {
            failure_rate: 0.01,
            p95_runtime_seconds: 10.0,
            throughput_per_slot: 5.0,
        };
        let state = RampState::baseline(20, baseline).unwrap();
        let state = ramp_decision(&state, &[], 1).unwrap();
        assert_eq!(state.stage, RampStage::Ramp30);
        assert_eq!(state.slot_target, 30);
        let state = ramp_decision(&state, &[], 600_001).unwrap();
        assert_eq!(state.stage, RampStage::Target40);
        assert_eq!(state.slot_target, 40);
        let unsafe_samples = vec![
            PressureSample {
                cpu_psi: 0.25,
                ..PressureSample::default()
            };
            3
        ];
        let state = ramp_decision(&state, &unsafe_samples, 700_000).unwrap();
        assert_eq!(state.stage, RampStage::RolledBack);
        assert_eq!(state.slot_target, 20);
        assert_eq!(state.rollback_reasons, vec!["cpu_psi"]);
    }

    #[test]
    fn xbabe2_gate_requires_fifteen_safe_underutilized_samples_and_fresh_work() {
        let samples = vec![
            BurnSample {
                cluster_cpu_fraction: 0.74,
                ..BurnSample::default()
            };
            15
        ];
        assert!(xbabe2_burn_eligible(&samples, 1).unwrap());
        assert!(!xbabe2_burn_eligible(&samples[..14], 1).unwrap());
        assert!(!xbabe2_burn_eligible(&samples, 0).unwrap());
        let busy = vec![
            BurnSample {
                cluster_cpu_fraction: 0.75,
                ..BurnSample::default()
            };
            15
        ];
        assert!(!xbabe2_burn_eligible(&busy, 1).unwrap());
    }
}

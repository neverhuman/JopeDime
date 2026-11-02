use super::*;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskPromotionMetricsV3 {
    pub contexts: usize,
    pub confidence: ClusteredConfidenceBoundsV3,
    pub maximum_slice_regression: f64,
    pub validity: f64,
    pub replay_verification: f64,
    pub failure_rate_increase: f64,
    pub latency_ratio: f64,
    pub strict_wins: usize,
    pub equal_budget_wins: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskPromotionEvaluationV3 {
    pub schema: String,
    pub task: Task,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    pub candidate_checkpoint_sha256: Sha256Digest,
    pub panel_consumption_sha256: Sha256Digest,
    pub run_index: usize,
    pub metrics: TaskPromotionMetricsV3,
    pub accepted: bool,
    #[serde(default)]
    pub rejection_reasons: Vec<String>,
    pub evaluated_at_ms: u128,
}

impl TaskPromotionEvaluationV3 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == TASK_PROMOTION_EVALUATION_V3_SCHEMA,
            "task promotion evaluation schema is not v3"
        );
        ensure!(self.run_index < 3, "promotion run index must be 0..3");
        ensure!(
            (PROMOTION_MIN_CONTEXTS..=PROMOTION_MAX_CONTEXTS).contains(&self.metrics.contexts)
                && (self.metrics.contexts - PROMOTION_MIN_CONTEXTS)
                    .is_multiple_of(PROMOTION_CONTEXT_STEP),
            "promotion evaluation context count is outside the sequential panel schedule"
        );
        ensure!(
            self.metrics.confidence.resamples == PROMOTION_BOOTSTRAP_RESAMPLES,
            "promotion evaluation must use exactly 10000 clustered resamples"
        );
        ensure!(
            [
                self.metrics.confidence.candidate_vs_accepted_lower,
                self.metrics.confidence.candidate_vs_gp_lower,
                self.metrics.confidence.conditional_advantage_lower,
                self.metrics.maximum_slice_regression,
                self.metrics.validity,
                self.metrics.replay_verification,
                self.metrics.failure_rate_increase,
                self.metrics.latency_ratio,
            ]
            .into_iter()
            .all(f64::is_finite),
            "promotion evaluation contains a non-finite metric"
        );
        ensure!(
            self.accepted == self.rejection_reasons.is_empty(),
            "promotion evaluation decision/reason mismatch"
        );
        Ok(())
    }
}

fn maximum_slice_regression(observations: &[PromotionObservationV3]) -> f64 {
    let mut slices: BTreeMap<(&str, &str), (f64, usize)> = BTreeMap::new();
    for row in observations {
        for (dimension, value) in &row.slices {
            let summary = slices
                .entry((dimension.as_str(), value.as_str()))
                .or_default();
            summary.0 += row.candidate_score - row.accepted_score;
            summary.1 += 1;
        }
    }
    slices
        .values()
        .map(|(sum, count)| -(sum / *count as f64))
        .fold(0.0, f64::max)
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_task_promotion_v3(
    task: Task,
    accepted_parent_checkpoint_sha256: Sha256Digest,
    candidate_checkpoint_sha256: Sha256Digest,
    panel_consumption_sha256: Sha256Digest,
    run_index: usize,
    observations: &[PromotionObservationV3],
    bootstrap_seed: u64,
    evaluated_at_ms: u128,
) -> Result<TaskPromotionEvaluationV3> {
    ensure!(run_index < 3, "promotion run index must be 0..3");
    ensure!(
        (PROMOTION_MIN_CONTEXTS..=PROMOTION_MAX_CONTEXTS).contains(&observations.len())
            && (observations.len() - PROMOTION_MIN_CONTEXTS).is_multiple_of(PROMOTION_CONTEXT_STEP),
        "promotion run must evaluate 128..512 contexts in steps of 64"
    );
    let contexts = observations
        .iter()
        .map(|row| &row.context_sha256)
        .collect::<BTreeSet<_>>();
    ensure!(
        contexts.len() == observations.len(),
        "promotion observations repeat a context"
    );
    let confidence = clustered_confidence_bounds_v3(observations, bootstrap_seed)?;
    let count = observations.len() as f64;
    let validity = observations.iter().filter(|row| row.valid).count() as f64 / count;
    let replay = observations
        .iter()
        .filter(|row| row.replay_verified)
        .count() as f64
        / count;
    let candidate_failures = observations
        .iter()
        .filter(|row| row.candidate_failed)
        .count() as f64
        / count;
    let accepted_failures = observations
        .iter()
        .filter(|row| row.accepted_failed)
        .count() as f64
        / count;
    let candidate_latency = observations
        .iter()
        .map(|row| row.candidate_latency_ms as u128)
        .sum::<u128>() as f64
        / count;
    let accepted_latency = observations
        .iter()
        .map(|row| row.accepted_latency_ms as u128)
        .sum::<u128>() as f64
        / count;
    let latency_ratio = if accepted_latency == 0.0 {
        f64::INFINITY
    } else {
        candidate_latency / accepted_latency
    };
    let metrics = TaskPromotionMetricsV3 {
        contexts: observations.len(),
        confidence,
        maximum_slice_regression: maximum_slice_regression(observations),
        validity,
        replay_verification: replay,
        failure_rate_increase: candidate_failures - accepted_failures,
        latency_ratio,
        strict_wins: observations.iter().filter(|row| row.strict_win).count(),
        equal_budget_wins: observations
            .iter()
            .filter(|row| row.equal_budget_win)
            .count(),
    };
    let mut rejection_reasons = Vec::new();
    for (condition, reason) in [
        (
            metrics.confidence.candidate_vs_accepted_lower >= 0.002,
            "candidate-versus-accepted lower bound < 0.002",
        ),
        (
            metrics.confidence.candidate_vs_gp_lower > 0.0,
            "candidate-versus-GP lower bound <= 0",
        ),
        (
            metrics.confidence.conditional_advantage_lower >= 0.005,
            "conditional-advantage lower bound < 0.005",
        ),
        (
            metrics.maximum_slice_regression <= 0.005,
            "a predefined slice regresses by more than 0.005",
        ),
        (metrics.validity >= 0.99, "validity < 0.99"),
        (
            (metrics.replay_verification - 1.0).abs() < f64::EPSILON,
            "replay verification is not 100%",
        ),
        (
            metrics.failure_rate_increase <= 0.0025,
            "failure-rate increase > 0.0025",
        ),
        (metrics.latency_ratio <= 1.25, "latency ratio > 1.25"),
        (metrics.strict_wins > 0, "no strict win"),
        (metrics.equal_budget_wins > 0, "no equal-budget win"),
    ] {
        if !condition {
            rejection_reasons.push(reason.to_owned());
        }
    }
    let evaluation = TaskPromotionEvaluationV3 {
        schema: TASK_PROMOTION_EVALUATION_V3_SCHEMA.to_owned(),
        task,
        accepted_parent_checkpoint_sha256,
        candidate_checkpoint_sha256,
        panel_consumption_sha256,
        run_index,
        metrics,
        accepted: rejection_reasons.is_empty(),
        rejection_reasons,
        evaluated_at_ms,
    };
    evaluation.validate()?;
    Ok(evaluation)
}

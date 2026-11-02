use super::{
    Sha256Digest, Task, CANARY_INTENT_SCHEMA, CANARY_PROPOSALS_PER_LANE,
    TASK_CANARY_ACCEPTANCE_V2_SCHEMA,
};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CanaryMode {
    Shadow,
    Production,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CanaryContextClass {
    Frontier,
    Hard,
    Anchor,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CanaryContextV2 {
    pub context_identity: String,
    pub class: CanaryContextClass,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CanaryIntentV2 {
    pub schema: String,
    pub mode: CanaryMode,
    pub task: Task,
    pub candidate_checkpoint_sha256: Sha256Digest,
    #[serde(
        rename = "accepted_parent_checkpoint_sha256",
        alias = "accepted_checkpoint_sha256"
    )]
    pub accepted_checkpoint_sha256: Sha256Digest,
    pub run_index: usize,
    pub seed: u64,
    pub contexts: Vec<CanaryContextV2>,
    pub candidate_proposals_per_context: usize,
    pub accepted_baseline_proposals_per_context: usize,
    pub zero_dl_gp_proposals_per_context: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at_ms: Option<u128>,
}

impl CanaryIntentV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CANARY_INTENT_SCHEMA,
            "canary intent schema is not v2"
        );
        ensure!(
            self.contexts.len() == 16
                && self
                    .contexts
                    .iter()
                    .filter(|row| row.class == CanaryContextClass::Frontier)
                    .count()
                    == 12
                && self
                    .contexts
                    .iter()
                    .filter(|row| row.class == CanaryContextClass::Hard)
                    .count()
                    == 2
                && self
                    .contexts
                    .iter()
                    .filter(|row| row.class == CanaryContextClass::Anchor)
                    .count()
                    == 2,
            "canary panel must be 12 frontier, 2 hard, and 2 anchor contexts"
        );
        ensure!(
            self.contexts
                .iter()
                .map(|row| row.context_identity.as_str())
                .collect::<BTreeSet<_>>()
                .len()
                == self.contexts.len(),
            "canary panel repeats a context"
        );
        ensure!(
            self.candidate_proposals_per_context == CANARY_PROPOSALS_PER_LANE
                && self.accepted_baseline_proposals_per_context == CANARY_PROPOSALS_PER_LANE
                && self.zero_dl_gp_proposals_per_context == CANARY_PROPOSALS_PER_LANE,
            "canary must evaluate eight proposals in every lane"
        );
        ensure!(
            self.mode == CanaryMode::Shadow || self.run_index < 3,
            "production canary run index must be 0..3"
        );
        Ok(())
    }
}

pub fn deterministic_canary_contexts(
    candidate: &Sha256Digest,
    run_index: usize,
    frontier: &[String],
    hard: &[String],
    anchor: &[String],
) -> Result<Vec<CanaryContextV2>> {
    fn select(
        candidate: &Sha256Digest,
        run_index: usize,
        pool: &[String],
        count: usize,
        class: CanaryContextClass,
    ) -> Result<Vec<CanaryContextV2>> {
        let mut ranked = pool
            .iter()
            .map(|identity| {
                let digest = Sha256::digest(format!("{candidate}\0{identity}").as_bytes());
                (digest.to_vec(), identity)
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(right.1)));
        let offset = run_index.saturating_mul(count);
        ensure!(
            ranked.len() >= offset.saturating_add(count),
            "canary pool cannot provide a disjoint run panel"
        );
        Ok(ranked[offset..offset + count]
            .iter()
            .map(|(_, identity)| CanaryContextV2 {
                context_identity: (*identity).clone(),
                class,
            })
            .collect())
    }
    let mut panel = select(
        candidate,
        run_index,
        frontier,
        12,
        CanaryContextClass::Frontier,
    )?;
    panel.extend(select(
        candidate,
        run_index,
        hard,
        2,
        CanaryContextClass::Hard,
    )?);
    panel.extend(select(
        candidate,
        run_index,
        anchor,
        2,
        CanaryContextClass::Anchor,
    )?);
    Ok(panel)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCanaryMetricsV2 {
    pub task_pairwise_lift: f64,
    pub maximum_task_regression: f64,
    pub validity: f64,
    pub strict_wins: usize,
    pub equal_budget_win: bool,
    pub failure_rate_increase: f64,
    pub latency_ratio: f64,
    pub replay_verification_complete: bool,
    pub conditional_advantage_lift: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCanaryAcceptanceV2 {
    pub schema: String,
    pub task: Task,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    pub candidate_checkpoint_sha256: Sha256Digest,
    pub run_index: usize,
    pub metrics: TaskCanaryMetricsV2,
    pub accepted: bool,
    #[serde(default)]
    pub rejection_reasons: Vec<String>,
}

impl TaskCanaryAcceptanceV2 {
    pub fn evaluate(mut self) -> Result<Self> {
        self.rejection_reasons.clear();
        let metric = &self.metrics;
        for (condition, reason) in [
            (metric.task_pairwise_lift >= 0.002, "pairwise lift < 0.002"),
            (
                metric.maximum_task_regression <= 0.005,
                "task regression > 0.005",
            ),
            (metric.validity >= 0.99, "validity < 0.99"),
            (metric.strict_wins >= 1, "no strict win"),
            (metric.equal_budget_win, "no equal-budget win"),
            (
                metric.failure_rate_increase <= 0.0025,
                "failure-rate increase > 0.0025",
            ),
            (metric.latency_ratio <= 1.25, "latency ratio > 1.25"),
            (
                metric.replay_verification_complete,
                "replay verification incomplete",
            ),
            (
                metric.conditional_advantage_lift >= 0.005,
                "conditional lift < 0.005",
            ),
        ] {
            if !condition {
                self.rejection_reasons.push(reason.to_owned());
            }
        }
        self.accepted = self.rejection_reasons.is_empty();
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == TASK_CANARY_ACCEPTANCE_V2_SCHEMA,
            "task canary acceptance schema is not v2"
        );
        ensure!(self.run_index < 3, "task canary run index must be 0..3");
        ensure!(
            [
                self.metrics.task_pairwise_lift,
                self.metrics.maximum_task_regression,
                self.metrics.validity,
                self.metrics.failure_rate_increase,
                self.metrics.latency_ratio,
                self.metrics.conditional_advantage_lift,
            ]
            .into_iter()
            .all(f64::is_finite),
            "task canary metrics must be finite"
        );
        ensure!(
            self.accepted == self.rejection_reasons.is_empty(),
            "task canary decision/reason mismatch"
        );
        Ok(())
    }
}

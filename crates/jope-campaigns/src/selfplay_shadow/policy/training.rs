use super::*;

pub fn next_direct_budget(current: usize, evidence: EfficiencyEvidence) -> Result<usize> {
    let position = DIRECT_BUDGETS
        .iter()
        .position(|value| *value == current)
        .ok_or_else(|| anyhow::anyhow!("unsupported direct budget"))?;
    ensure!(
        [
            evidence.lower_90_efficiency_ratio,
            evidence.upper_90_efficiency_ratio,
            evidence.validity,
        ]
        .into_iter()
        .all(f64::is_finite),
        "efficiency evidence must be finite"
    );
    if evidence.failure_guard_triggered || evidence.validity < 0.99 {
        return Ok(DIRECT_BUDGETS[position.saturating_sub(1)]);
    }
    if evidence.complete_waves < 3 || evidence.direct_evaluations < 1_024 {
        return Ok(current);
    }
    if evidence.lower_90_efficiency_ratio > 1.0 {
        return Ok(DIRECT_BUDGETS[(position + 1).min(DIRECT_BUDGETS.len() - 1)]);
    }
    if evidence.upper_90_efficiency_ratio < 1.0 {
        return Ok(DIRECT_BUDGETS[position.saturating_sub(1)]);
    }
    Ok(current)
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SelectorReplayTask {
    pub reward_ranked_strict_wins_per_evaluation: f64,
    pub first_unique_strict_wins_per_evaluation: f64,
    pub emitted_validity: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SelectorReplay {
    pub binary: SelectorReplayTask,
    pub regression: SelectorReplayTask,
}

impl SelectorReplay {
    pub fn passes(&self) -> bool {
        [self.binary, self.regression].into_iter().all(|task| {
            task.emitted_validity >= 0.99
                && task.reward_ranked_strict_wins_per_evaluation
                    > task.first_unique_strict_wins_per_evaluation
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeadTrainingPolicy {
    pub train_reward_head: bool,
    pub train_stability_head: bool,
    pub train_decoder_imitation: bool,
}

pub fn training_policy(
    authoritative: bool,
    valid: bool,
    high_performing: bool,
) -> HeadTrainingPolicy {
    HeadTrainingPolicy {
        train_reward_head: authoritative,
        train_stability_head: authoritative,
        train_decoder_imitation: authoritative && valid && high_performing,
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetCheckpointMetrics {
    pub pairwise_accuracy: f64,
    pub genome_only_pairwise_accuracy: f64,
    pub conditional_advantage: f64,
    pub strict_win_calibration: f64,
    pub top_budget_regret: f64,
    pub failure_accuracy: f64,
    pub validity: f64,
    pub uniqueness: f64,
    pub collapse_guard_triggered: bool,
}

impl TargetCheckpointMetrics {
    pub fn selection_score(&self) -> Option<f64> {
        let finite = [
            self.pairwise_accuracy,
            self.genome_only_pairwise_accuracy,
            self.conditional_advantage,
            self.strict_win_calibration,
            self.top_budget_regret,
            self.failure_accuracy,
            self.validity,
            self.uniqueness,
        ]
        .into_iter()
        .all(f64::is_finite);
        (finite && self.conditional_advantage > 0.0 && !self.collapse_guard_triggered)
            .then_some(self.pairwise_accuracy + 2.0 * self.conditional_advantage)
    }
}

/// Enforce the 50% target-task / 50% auxiliary mix of a training batch.
///
/// `auxiliary_rows` is the POOLED count of every non-target task: with the
/// four-task vocabulary the auxiliary half may mix regression, binary,
/// timeseries, and multiclass rows, and this validator is deliberately
/// indifferent to that composition (per-task quota policy is D5 work on the
/// trainer side). Semantics are unchanged from the two-task era, where the
/// auxiliary half was simply the one other task.
pub fn validate_target_aux_mix(target_rows: usize, auxiliary_rows: usize) -> Result<()> {
    ensure!(
        target_rows == auxiliary_rows && target_rows > 0,
        "training batches must be 50% target-task and 50% pooled auxiliary-task"
    );
    Ok(())
}

pub fn ranking_weight(row_task: Task, target_task: Task) -> f64 {
    if row_task == target_task {
        2.0
    } else {
        1.0
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotProgress {
    pub new_labels: usize,
    pub new_datasets: usize,
    pub new_families: usize,
    pub elapsed_ms: u64,
}

pub fn snapshot_due(progress: SnapshotProgress) -> bool {
    progress.new_labels >= 50_000
        || progress.new_datasets >= 32
        || progress.new_families >= 8
        || (progress.elapsed_ms >= 2 * 60 * 60 * 1_000 && progress.new_labels >= 10_000)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusRowRef {
    pub row_id: String,
    pub dataset_id: String,
    pub observed_at_ms: u64,
}

/// Deterministic depth-first reservoir order: every dataset contributes its
/// first row before any dataset contributes its second row.
pub fn breadth_first_reservoir(rows: &[CorpusRowRef]) -> Vec<String> {
    let mut by_dataset = BTreeMap::<String, Vec<&CorpusRowRef>>::new();
    for row in rows {
        by_dataset
            .entry(row.dataset_id.clone())
            .or_default()
            .push(row);
    }
    for dataset_rows in by_dataset.values_mut() {
        dataset_rows.sort_by(|left, right| {
            right
                .observed_at_ms
                .cmp(&left.observed_at_ms)
                .then_with(|| left.row_id.cmp(&right.row_id))
        });
    }
    let maximum_depth = by_dataset.values().map(Vec::len).max().unwrap_or(0);
    let mut output = Vec::with_capacity(rows.len());
    for depth in 0..maximum_depth {
        for dataset_rows in by_dataset.values() {
            if let Some(row) = dataset_rows.get(depth) {
                output.push(row.row_id.clone());
            }
        }
    }
    output
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TournamentSpec {
    pub task: Task,
    pub candidate_checkpoint_sha256: Sha256Digest,
    pub parent_checkpoint_sha256: Sha256Digest,
    pub cohort_sha256: Sha256Digest,
    pub corpus_sha256: Sha256Digest,
    pub evaluator_sha256: Sha256Digest,
    pub protocol_sha256: Sha256Digest,
    pub proposal_sha256: Sha256Digest,
    pub context_sha256: Sha256Digest,
    pub seeds: Vec<u64>,
    pub budget_per_dataset: usize,
    pub family_disjoint: bool,
    pub absent_from_candidate_corpus: bool,
    pub absent_from_sealed_audit: bool,
}

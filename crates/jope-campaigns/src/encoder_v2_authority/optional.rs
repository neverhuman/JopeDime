use super::*;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OptionalTaskRow {
    pub task: String,
    pub state: String,
    pub distinct_dataset_identities: u64,
    pub minimum_required: u64,
    pub minimum_rolling_origins_per_dataset: Option<u32>,
    pub valid_v2_cache_bindings: u64,
    pub valid_context_transform_bindings: u64,
    pub evaluator_retained: bool,
    pub evaluator_sha256: String,
    pub task_valid_head_count: u64,
    pub robust_bank_sha256: String,
    pub fresh_unbiased_shadow_evidence: bool,
    pub causal_reduction_verified: Option<bool>,
    pub no_future_leakage: Option<bool>,
    pub primary_metric: String,
    pub native_error_metric: String,
    pub training_allowed: bool,
    pub blockers: Vec<String>,
}

impl OptionalTaskRow {
    pub fn computed_training_allowed(&self) -> Result<bool> {
        ensure!(
            matches!(
                self.state.as_str(),
                "absent" | "collecting" | "replay_pending" | "shadow_ready" | "blocked"
            ),
            "optional-task state is invalid"
        );
        require_sha256(&self.evaluator_sha256, "optional evaluator SHA-256")?;
        require_sha256(&self.robust_bank_sha256, "optional robust-bank SHA-256")?;
        let common = self.state == "shadow_ready"
            && self.distinct_dataset_identities >= self.minimum_required
            && self.valid_v2_cache_bindings >= self.distinct_dataset_identities
            && self.valid_context_transform_bindings >= self.distinct_dataset_identities
            && self.evaluator_retained
            && self.task_valid_head_count > 0
            && self.fresh_unbiased_shadow_evidence
            && self.blockers.is_empty();
        match self.task.as_str() {
            "multiclass" => Ok(common
                && self.minimum_required == 500
                && self.minimum_rolling_origins_per_dataset.is_none()
                && self.causal_reduction_verified.is_none()
                && self.no_future_leakage.is_none()
                && self.primary_metric == "macro_f1"
                && self.native_error_metric == "one_minus_macro_f1"),
            "timeseries" => Ok(common
                && self.distinct_dataset_identities == 42
                && self.minimum_required == 42
                && self
                    .minimum_rolling_origins_per_dataset
                    .is_some_and(|origins| origins >= 3)
                && self.causal_reduction_verified == Some(true)
                && self.no_future_leakage == Some(true)
                && self.primary_metric == "negative_mase"
                && self.native_error_metric == "mase"),
            _ => Err(anyhow::anyhow!("readiness row is not an optional task")),
        }
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.training_allowed == self.computed_training_allowed()?,
            "optional-task training claim does not match its evidence"
        );
        if self.training_allowed {
            ensure!(
                self.blockers.is_empty(),
                "trainable optional task has blockers"
            );
        } else {
            ensure!(
                !self.blockers.is_empty(),
                "unready optional task lacks typed blockers"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OptionalTaskReadinessV2 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub input_digests: Vec<String>,
    pub tasks: Vec<OptionalTaskRow>,
}

impl OptionalTaskReadinessV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == OPTIONAL_READINESS_SCHEMA,
            "invalid optional readiness schema"
        );
        validate_status(
            self.generated_at_unix_ms,
            self.freshness_budget_seconds,
            &self.input_digests,
        )?;
        ensure!(
            self.tasks.len() == 2,
            "optional readiness must have exactly two rows"
        );
        ensure!(
            self.tasks[0].task == "multiclass" && self.tasks[1].task == "timeseries",
            "optional readiness rows are missing or out of order"
        );
        for row in &self.tasks {
            row.validate()?;
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OptionalTaskEvidenceRow {
    pub task: String,
    pub cache_coverage_sha256: Option<String>,
    pub context_identity_sha256: Option<String>,
    pub evaluator_retention_sha256: Option<String>,
    pub comparator_lock_sha256: Option<String>,
    pub unbiased_shadow_sha256: Option<String>,
    pub typed_absence_reason: Option<String>,
}

impl OptionalTaskEvidenceRow {
    fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.task.as_str(), "multiclass" | "timeseries"),
            "invalid task"
        );
        let present = [
            &self.cache_coverage_sha256,
            &self.context_identity_sha256,
            &self.evaluator_retention_sha256,
            &self.comparator_lock_sha256,
            &self.unbiased_shadow_sha256,
        ];
        for digest in present.iter().filter_map(|value| value.as_deref()) {
            require_sha256(digest, "optional-task evidence digest")?;
        }
        if present.iter().all(|value| value.is_some()) {
            ensure!(
                self.typed_absence_reason.is_none(),
                "complete evidence claims absence"
            );
        } else {
            ensure!(
                self.typed_absence_reason
                    .as_deref()
                    .is_some_and(|reason| !reason.is_empty()),
                "incomplete optional evidence lacks typed absence"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OptionalTaskEvidenceV1 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub input_digests: Vec<String>,
    pub readiness_sha256: String,
    pub tasks: Vec<OptionalTaskEvidenceRow>,
}

impl OptionalTaskEvidenceV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == OPTIONAL_EVIDENCE_SCHEMA,
            "invalid optional evidence schema"
        );
        validate_status(
            self.generated_at_unix_ms,
            self.freshness_budget_seconds,
            &self.input_digests,
        )?;
        require_sha256(&self.readiness_sha256, "optional readiness SHA-256")?;
        ensure!(
            self.tasks.len() == 2
                && self.tasks[0].task == "multiclass"
                && self.tasks[1].task == "timeseries",
            "optional evidence rows are incomplete or out of order"
        );
        for row in &self.tasks {
            row.validate()?;
        }
        Ok(())
    }
}

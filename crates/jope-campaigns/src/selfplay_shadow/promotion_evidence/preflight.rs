use super::*;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidencePreflightCohortV1 {
    pub sequence: usize,
    pub task: Task,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    pub contexts: usize,
    pub requested_slots: usize,
    pub valid_slots: usize,
    pub completed: bool,
}

impl EvidencePreflightCohortV1 {
    pub fn validity(&self) -> f64 {
        if self.requested_slots == 0 {
            0.0
        } else {
            self.valid_slots as f64 / self.requested_slots as f64
        }
    }

    pub fn validate_for(
        &self,
        task: Task,
        accepted_parent: &Sha256Digest,
        sequence: usize,
    ) -> Result<()> {
        ensure!(
            self.sequence == sequence,
            "preflight cohorts are not consecutive"
        );
        ensure!(self.task == task, "preflight task drift");
        ensure!(
            &self.accepted_parent_checkpoint_sha256 == accepted_parent,
            "preflight accepted-parent drift"
        );
        ensure!(
            self.contexts == PREFLIGHT_CONTEXTS_PER_COHORT
                && self.requested_slots
                    == PREFLIGHT_CONTEXTS_PER_COHORT * EVIDENCE_SLOTS_PER_CONTEXT,
            "preflight cohort must request eight slots for 64 contexts"
        );
        ensure!(
            self.completed
                && self.valid_slots <= self.requested_slots
                && self.validity() >= PREFLIGHT_MIN_VALIDITY,
            "preflight cohort does not meet 99.5% requested-slot validity"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceEpochContextV1 {
    pub context_identity: String,
    pub context_sha256: Sha256Digest,
    pub requested_wave: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalEvidenceEpochV1 {
    pub schema: String,
    pub task: Task,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    pub start_wave: u64,
    pub runtime_policy_sha256: Sha256Digest,
    pub proposer_policy_sha256: Sha256Digest,
    pub evaluator_policy_sha256: Sha256Digest,
    pub catalog_eligibility_sha256: Sha256Digest,
    pub requested_contexts: Vec<EvidenceEpochContextV1>,
    pub slots_per_context: usize,
    pub preflight_cohorts: Vec<EvidencePreflightCohortV1>,
    pub opened_at_ms: u128,
}

impl CausalEvidenceEpochV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CAUSAL_EVIDENCE_EPOCH_SCHEMA,
            "causal evidence epoch schema is not v1"
        );
        ensure!(
            self.start_wave >= 191,
            "evidence epoch starts before causal waves"
        );
        ensure!(
            self.slots_per_context == EVIDENCE_SLOTS_PER_CONTEXT,
            "evidence epoch must request eight slots per context"
        );
        ensure!(
            !self.requested_contexts.is_empty(),
            "evidence epoch has no requested contexts"
        );
        let identities = self
            .requested_contexts
            .iter()
            .map(|context| context.context_identity.as_str())
            .collect::<BTreeSet<_>>();
        let digests = self
            .requested_contexts
            .iter()
            .map(|context| &context.context_sha256)
            .collect::<BTreeSet<_>>();
        ensure!(
            identities.len() == self.requested_contexts.len()
                && digests.len() == self.requested_contexts.len()
                && self.requested_contexts.iter().all(|context| {
                    !context.context_identity.trim().is_empty()
                        && context.requested_wave >= self.start_wave
                }),
            "evidence epoch contexts are empty, duplicated, or pre-epoch"
        );
        ensure!(
            self.preflight_cohorts.len() == PREFLIGHT_COHORTS_REQUIRED,
            "evidence epoch requires ten preflight cohorts"
        );
        for (sequence, cohort) in self.preflight_cohorts.iter().enumerate() {
            cohort.validate_for(self.task, &self.accepted_parent_checkpoint_sha256, sequence)?;
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<Sha256Digest> {
        self.validate()?;
        Ok(Sha256Digest::of_bytes(&serde_json::to_vec(self)?))
    }

    pub fn requested_slots(&self) -> usize {
        self.requested_contexts.len() * self.slots_per_context
    }
}

pub fn ensure_evidence_epoch_immutable(
    existing: &CausalEvidenceEpochV1,
    proposed: &CausalEvidenceEpochV1,
) -> Result<()> {
    ensure!(
        existing.identity()? == proposed.identity()?,
        "an opened evidence epoch cannot be reset or mutated"
    );
    Ok(())
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionEpochEvidenceV1 {
    pub epoch_sha256: Sha256Digest,
    pub evidence: TaskCausalEvidenceV2,
    pub policy_pins_match: bool,
    pub logged_slots: usize,
    pub permanent_missing_slots: usize,
    pub direct_count: usize,
    pub comparator_count: usize,
    pub production_gate_open: bool,
    #[serde(default)]
    pub gate_reasons: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalEvidenceLedgerV3 {
    pub schema: String,
    pub generated_at_ms: u128,
    pub authoritative: bool,
    pub all_history: CausalEvidenceLedgerV2,
    pub promotion_epoch: BTreeMap<Task, PromotionEpochEvidenceV1>,
}

impl CausalEvidenceLedgerV3 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CAUSAL_EVIDENCE_LEDGER_V3_SCHEMA,
            "causal evidence ledger schema is not v3"
        );
        self.all_history.validate()?;
        for (task, row) in &self.promotion_epoch {
            ensure!(
                row.direct_count == row.comparator_count
                    && row.direct_count == row.evidence.complete_pairs,
                "{task} promotion epoch lane counts differ"
            );
            if row.production_gate_open {
                ensure!(
                    self.authoritative
                        && row.policy_pins_match
                        && row.logged_slots == row.evidence.requested_slots
                        && row.permanent_missing_slots == 0
                        && row.evidence.complete_pairs >= MIN_PARENT_DIRECT_OUTCOMES
                        && row.evidence.proposal_validity >= PROMOTION_MIN_VALIDITY
                        && row.evidence.zero_cross_context_backfill
                        && row.evidence.completed_causal_waves
                        && row.evidence.valid_transitive_lineage,
                    "{task} v3 gate opened without clean epoch evidence"
                );
            }
        }
        Ok(())
    }
}

pub fn build_causal_evidence_ledger_v3(
    all_history: CausalEvidenceLedgerV2,
    epochs: &BTreeMap<Task, CausalEvidenceEpochV1>,
    receipts: &[CausalEvaluationReceiptV2],
    policy_pins_match: &BTreeMap<Task, bool>,
    logged_slots: &BTreeMap<Task, usize>,
    permanent_missing_slots: &BTreeMap<Task, usize>,
    generated_at_ms: u128,
) -> Result<CausalEvidenceLedgerV3> {
    all_history.validate()?;
    let mut promotion_epoch = BTreeMap::new();
    for (task, epoch) in epochs {
        epoch.validate()?;
        ensure!(epoch.task == *task, "evidence epoch task-key drift");
        let contexts = epoch
            .requested_contexts
            .iter()
            .map(|context| context.context_identity.as_str())
            .collect::<BTreeSet<_>>();
        let epoch_receipts = receipts
            .iter()
            .filter(|receipt| {
                receipt.task == *task
                    && receipt.wave >= epoch.start_wave
                    && receipt.accepted_parent_checkpoint_sha256
                        == epoch.accepted_parent_checkpoint_sha256
                    && contexts.contains(receipt.context_identity.as_str())
                    && receipt.pair_index < epoch.slots_per_context
            })
            .cloned()
            .collect::<Vec<_>>();
        let requested = BTreeMap::from([(*task, epoch.requested_slots())]);
        let parents = BTreeMap::from([(*task, epoch.accepted_parent_checkpoint_sha256.clone())]);
        let evidence =
            build_causal_evidence_ledger_v2(&parents, &requested, epoch_receipts, generated_at_ms)?
                .tasks
                .get(task)
                .cloned()
                .context("v3 epoch builder lost task evidence")?;
        let pins_match = policy_pins_match.get(task).copied().unwrap_or(false);
        let slots_logged = logged_slots.get(task).copied().unwrap_or_default();
        let permanent_missing = permanent_missing_slots
            .get(task)
            .copied()
            .unwrap_or_default();
        let mut gate_reasons = evidence.gate_reasons.clone();
        if !pins_match {
            gate_reasons.push("epoch policy pins do not match wave receipts".to_owned());
        }
        if slots_logged != epoch.requested_slots() {
            gate_reasons.push(format!(
                "slot log {slots_logged}/{}",
                epoch.requested_slots()
            ));
        }
        if permanent_missing > 0 {
            gate_reasons.push(format!("permanent missing epoch slots {permanent_missing}"));
        }
        let gate_open = gate_reasons.is_empty();
        promotion_epoch.insert(
            *task,
            PromotionEpochEvidenceV1 {
                epoch_sha256: epoch.identity()?,
                direct_count: evidence.complete_pairs,
                comparator_count: evidence.complete_pairs,
                evidence,
                policy_pins_match: pins_match,
                logged_slots: slots_logged,
                permanent_missing_slots: permanent_missing,
                production_gate_open: gate_open,
                gate_reasons,
            },
        );
    }
    let ledger = CausalEvidenceLedgerV3 {
        schema: CAUSAL_EVIDENCE_LEDGER_V3_SCHEMA.to_owned(),
        generated_at_ms,
        authoritative: true,
        all_history,
        promotion_epoch,
    };
    ledger.validate()?;
    Ok(ledger)
}

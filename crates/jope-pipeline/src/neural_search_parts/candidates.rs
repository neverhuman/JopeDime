impl IncumbentSeed {
    fn validate(&self) -> Result<()> {
        ensure!(
            jope_policy::is_sha256_hex(&self.frozen_recipe_sha256)
                && SynthesizedRecipe::new(self.original_proposal.clone())?.executable_sha256
                    == self.program_sha256,
            "incumbent identity changed"
        );
        Ok(())
    }
    /// Verify one retained arm winner for this exact pair selection. The winner may
    /// come from an earlier round (another experiment) on the same dataset, split,
    /// rows, evaluator and evaluation seed; `neural` names the arm that produced it.
    pub fn from_frozen_recipe(
        bytes: &[u8],
        pinned_sha256: &str,
        selection: &SelectionPlan,
        neural: bool,
    ) -> Result<Self> {
        ensure!(
            sha256_hex(bytes) == pinned_sha256,
            "incumbent bytes differ from pinned digest"
        );
        let frozen: FrozenRecipe = serde_json::from_slice(bytes)?;
        let candidate = &frozen.candidate;
        candidate.recipe.validate()?;
        ensure!(
            candidate.recipe == SynthesizedRecipe::new(candidate.genome.clone())?
                && candidate.program_sha256 == candidate.recipe.executable_sha256,
            "incumbent recipe derivation changed"
        );
        let feedback = &frozen.selection_feedback;
        ensure!(
            feedback.status == EvaluationStatus::Completed
                && feedback.candidate_sha256 == candidate.id_sha256
                && feedback.loss.is_some_and(f64::is_finite),
            "incumbent lacks a completed train-side selection"
        );
        let mut expected =
            selection.refit_request(&candidate.program_sha256, &candidate.id_sha256)?;
        expected
            .experiment_sha256
            .clone_from(&frozen.refit_request.experiment_sha256);
        expected
            .checkpoint_sha256
            .clone_from(&frozen.refit_request.checkpoint_sha256);
        ensure!(
            expected.identity()? == frozen.refit_request.identity()?,
            "incumbent belongs to another dataset, split, rows or evaluator"
        );
        ensure!(
            frozen.refit_request.checkpoint_sha256.is_some() == neural,
            "incumbent was not produced by this arm"
        );
        let seed = Self {
            frozen_recipe_sha256: pinned_sha256.into(),
            original_proposal: candidate.genome.clone(),
            program_sha256: candidate.program_sha256.clone(),
        };
        seed.validate()?;
        Ok(seed)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub id_sha256: String,
    pub genome: String,
    pub recipe: SynthesizedRecipe,
    /// Canonical executable recipe, including explicitly recorded preprocessing.
    pub program_sha256: String,
    pub generation: usize,
    /// Original proposal slot for generation zero; no invented slot for offspring.
    pub initial_slot: Option<usize>,
    pub origin: Origin,
    pub parents: Vec<String>,
    pub rank_log_weight: Option<f64>,
    /// W4-T03/T07: the proposal-ledger record of the draw a policy slot came from (its original
    /// tokens, terminal outcome, scores and parser result), as its seed slot names it. Absent
    /// for generic, incumbent and offspring candidates and in older ledgers; not part of the
    /// candidate identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposal_record_sha256: Option<String>,
}
impl Candidate {
    /// The record identity may only join a generation-zero policy slot of the neural arm.
    fn check_proposal_record(&self, neural_policy_slot: bool) -> Result<()> {
        ensure!(
            self.proposal_record_sha256
                .as_deref()
                .is_none_or(|r| neural_policy_slot && jope_policy::is_sha256_hex(r)),
            "proposal record outside a generation-zero policy slot"
        );
        Ok(())
    }
    pub fn validate_initial(&self, arm: &ArmBudget) -> Result<()> {
        ensure!(
            self.generation == 0 && self.parents.is_empty(),
            "not an initial candidate"
        );
        let slot = self.initial_slot.context("initial candidate slot absent")?;
        ensure!(
            slot < INITIAL_SIZE,
            "initial candidate slot outside population"
        );
        let mut expected = Self::new(
            arm,
            self.genome.clone(),
            0,
            slot,
            self.origin.clone(),
            vec![],
            self.rank_log_weight,
        )?;
        self.check_proposal_record(matches!(
            self.origin,
            Origin::NeuralPolicy | Origin::NeuralRejectionFallback | Origin::RandomExploration
        ))?;
        expected
            .proposal_record_sha256
            .clone_from(&self.proposal_record_sha256);
        ensure!(
            serde_json::to_value(self)? == serde_json::to_value(expected)?,
            "initial candidate identity changed"
        );
        Ok(())
    }
    fn new(
        arm: &ArmBudget,
        genome: String,
        generation: usize,
        slot: usize,
        origin: Origin,
        parents: Vec<String>,
        rank_log_weight: Option<f64>,
    ) -> Result<Self> {
        let recipe = SynthesizedRecipe::new(genome.clone())?;
        ensure!(
            rank_log_weight.is_none_or(f64::is_finite),
            "nonfinite policy rank weight"
        );
        let program_sha256 = recipe.executable_sha256.clone();
        let id_sha256 = sha256_hex(&serde_json::to_vec(&(
            &arm.id_sha256,
            generation,
            slot,
            &program_sha256,
            &origin,
            &parents,
        ))?);
        Ok(Self {
            id_sha256,
            genome,
            recipe,
            program_sha256,
            generation,
            initial_slot: (generation == 0).then_some(slot),
            origin,
            parents,
            rank_log_weight,
            proposal_record_sha256: None,
        })
    }
}

/// Scoring stays in the controller; the worker receives train labels only.
/// `physical_fit_starts` must come from the worker's fit receipt, not exit code,
/// scheduling, or a policy-origin label. Unknown counters remain None.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Feedback {
    pub candidate_sha256: String,
    pub status: EvaluationStatus,
    pub loss: Option<f64>,
    pub physical_fit_starts: Option<u64>,
    pub worker_receipt_sha256: Option<String>,
    /// Ordered fold identities: exact evidence for any admitted/unknown attempt,
    /// normalized identity only for jobs proven never to have released a gate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_evidence_sha256: Option<String>,
    pub fitted_artifact_sha256: Option<String>,
    pub reason: Option<String>,
    pub oof: Option<OofEvidence>,
    /// At least one required operation was not admitted. Retry is permitted only
    /// inside the original arm clock. Expired final receipts retain this historical
    /// observation; the flag does not authorize a later launch.
    pub retryable_inventory: bool,
}
impl Feedback {
    fn enforce_selection(&mut self, selection: &SelectionPlan, candidate: &Candidate) {
        self.enforce_measurement();
        if self.status == EvaluationStatus::Completed
            && !self.oof.as_ref().is_some_and(|o| {
                o.validate(selection, &candidate.program_sha256, &candidate.id_sha256)
                    .is_ok()
                    && o.total_fit_starts() == self.physical_fit_starts
                    && o.fitted_bundle_sha256().ok() == self.fitted_artifact_sha256
            })
        {
            self.status = EvaluationStatus::Failed;
            self.loss = None;
            self.reason = Some("selection lacks exact complete train-side OOF measurements".into());
        }
    }
    fn enforce_measurement(&mut self) {
        if self.status == EvaluationStatus::Completed
            && !(self.loss.is_some_and(f64::is_finite)
                && self.physical_fit_starts.is_some_and(|n| n > 0)
                && self
                    .worker_receipt_sha256
                    .as_ref()
                    .is_some_and(|s| jope_policy::is_sha256_hex(s))
                && self
                    .fitted_artifact_sha256
                    .as_ref()
                    .is_some_and(|s| jope_policy::is_sha256_hex(s)))
        {
            self.status = EvaluationStatus::Failed;
            self.reason = Some("completed fit lacks finite loss, physical-fit counter, fitted artifact or worker receipt".into());
        }
        if self.status != EvaluationStatus::Completed {
            self.loss = None;
        }
    }
    #[cfg(test)]
    fn exhausted(candidate: &Candidate) -> Self {
        Self {
            candidate_sha256: candidate.id_sha256.clone(),
            status: EvaluationStatus::BudgetExhausted,
            loss: None,
            physical_fit_starts: Some(0),
            worker_receipt_sha256: None,
            replay_evidence_sha256: None,
            fitted_artifact_sha256: None,
            reason: Some("original arm deadline reached before fit admission".into()),
            oof: None,
            retryable_inventory: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Observation {
    pub candidate: Candidate,
    pub feedback: Feedback,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchReceipt {
    pub schema: String,
    pub comparison_sha256: String,
    pub selection_sha256: String,
    pub arm: ArmBudget,
    pub deadline_unix_ms: u64,
    pub proposal: Option<ProposalEvidence>,
    pub observations: Vec<Observation>,
    pub stop_reason: String,
}

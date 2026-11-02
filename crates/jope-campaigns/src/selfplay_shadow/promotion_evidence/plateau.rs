use super::*;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptedGenerationV2 {
    pub task: Task,
    pub parent_checkpoint_sha256: Sha256Digest,
    pub child_checkpoint_sha256: Sha256Digest,
    pub independent_external_passes: Vec<Sha256Digest>,
    pub accepted_at_ms: u128,
}

impl AcceptedGenerationV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.parent_checkpoint_sha256 != self.child_checkpoint_sha256,
            "accepted generation child equals its parent"
        );
        ensure!(
            self.independent_external_passes.len() == 3
                && self
                    .independent_external_passes
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
                    == 3,
            "accepted generation requires three independent external passes"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlateauDiagnosisV1 {
    pub task: Task,
    pub unchanged_parent_checkpoint_sha256: Sha256Digest,
    pub rejected_candidates: Vec<Sha256Digest>,
    pub proposal_validity: f64,
    pub gp_disadvantage: f64,
    pub calibration_summary: String,
    pub data_slice_summary: String,
    pub latency_summary: String,
    pub replay_summary: String,
    pub diagnosed_at_ms: u128,
}

impl PlateauDiagnosisV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.rejected_candidates.len() >= 3
                && self
                    .rejected_candidates
                    .iter()
                    .collect::<BTreeSet<_>>()
                    .len()
                    == self.rejected_candidates.len(),
            "plateau diagnosis requires three distinct rejected shadows"
        );
        ensure!(
            self.proposal_validity.is_finite() && self.gp_disadvantage.is_finite(),
            "plateau diagnosis contains a non-finite metric"
        );
        ensure!(
            [
                self.calibration_summary.as_str(),
                self.data_slice_summary.as_str(),
                self.latency_summary.as_str(),
                self.replay_summary.as_str(),
            ]
            .into_iter()
            .all(|value| !value.trim().is_empty()),
            "plateau diagnosis is incomplete"
        );
        Ok(())
    }
}

pub const PLATEAU_DIAGNOSIS_V2_SCHEMA: &str = "PlateauDiagnosis.v2";
pub const RECIPE_CHANGE_DIRECTIVE_V1_SCHEMA: &str = "RecipeChangeDirective.v1";
pub const RECIPE_CHANGE_DIRECTIVE_V2_SCHEMA: &str = "RecipeChangeDirective.v2";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlateauFailureClassV2 {
    ValidityLoss,
    GpDisadvantage,
    FamilyWeakness,
    ConditionalCollapse,
    LatencyRegression,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeKnobV2 {
    HardNegativeValiditySampling,
    ExactCausalPairGpComparatorMixture,
    SliceBalancedSampling,
    AblationRoutingLoss,
    ProposalModelComplexityBudget,
}

impl PlateauFailureClassV2 {
    fn required_knob(self) -> RecipeKnobV2 {
        match self {
            Self::ValidityLoss => RecipeKnobV2::HardNegativeValiditySampling,
            Self::GpDisadvantage => RecipeKnobV2::ExactCausalPairGpComparatorMixture,
            Self::FamilyWeakness => RecipeKnobV2::SliceBalancedSampling,
            Self::ConditionalCollapse => RecipeKnobV2::AblationRoutingLoss,
            Self::LatencyRegression => RecipeKnobV2::ProposalModelComplexityBudget,
        }
    }
}

/// One rejected shadow counted toward a recipe-scoped plateau, with the
/// recipe identity its trainer intent resolves to (None = legacy receipt
/// without a resolvable recipe; counted conservatively unless an operator
/// RecipeChangeDirective bridges it).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlateauCountedCandidateV1 {
    pub candidate_checkpoint_sha256: Sha256Digest,
    pub recipe_identity_sha256: Option<String>,
}

/// Recipe-identity-scoped plateau diagnosis. The v1 pause keyed only on
/// (>=3 rejected shadows, unchanged parent) — a hard deadlock, because no
/// new shadow can train while paused, so the parent can never change. v2
/// counts a rejected shadow only when its training recipe identity matches
/// the recipe the lane would train with RIGHT NOW: a genuine recipe change
/// re-schedules training exactly as the v1 status string always promised
/// ("identical shadow training is paused"). v1 documents are preserved and
/// still written; this receipt is additive.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlateauDiagnosisV2 {
    pub schema: String,
    pub task: Task,
    pub unchanged_parent_checkpoint_sha256: Sha256Digest,
    /// Identity of the recipe the paused lane would use now; None when it
    /// could not be computed (scoping then stays conservative: everything
    /// counts, exactly the v1 behavior).
    pub current_recipe_identity_sha256: Option<String>,
    pub counted_candidates: Vec<PlateauCountedCandidateV1>,
    pub recipe_change_directive_present: bool,
    pub proposal_validity: f64,
    pub gp_disadvantage: f64,
    pub calibration_summary: String,
    pub data_slice_summary: String,
    pub latency_summary: String,
    pub replay_summary: String,
    #[serde(default)]
    pub failure_classes: BTreeSet<PlateauFailureClassV2>,
    pub diagnosed_at_ms: u128,
}

/// A recipe change tied to the exact immutable plateau diagnosis it answers.
/// Every diagnosed failure class must be addressed by its auditable recipe
/// knob, and the next recipe identity must differ from the exhausted one.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeChangeDirectiveV2 {
    pub schema: String,
    pub task: Task,
    pub unchanged_parent_checkpoint_sha256: Sha256Digest,
    pub answers_diagnosis_sha256: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_recipe_identity_sha256: Option<Sha256Digest>,
    pub next_recipe_identity_sha256: Sha256Digest,
    pub failure_classes: BTreeSet<PlateauFailureClassV2>,
    pub changed_knobs: BTreeSet<RecipeKnobV2>,
    pub created_at_ms: u128,
}

impl RecipeChangeDirectiveV2 {
    pub fn validate(&self, diagnosis: &PlateauDiagnosisV2) -> Result<()> {
        ensure!(
            self.schema == RECIPE_CHANGE_DIRECTIVE_V2_SCHEMA,
            "recipe change directive v2 schema mismatch"
        );
        ensure!(
            self.task == diagnosis.task
                && self.unchanged_parent_checkpoint_sha256
                    == diagnosis.unchanged_parent_checkpoint_sha256,
            "recipe change directive v2 diagnosis scope mismatch"
        );
        let mut diagnosed = diagnosis.failure_classes.clone();
        if diagnosed.is_empty() {
            if diagnosis.proposal_validity < 0.99 {
                diagnosed.insert(PlateauFailureClassV2::ValidityLoss);
            }
            if diagnosis.gp_disadvantage > 0.0 {
                diagnosed.insert(PlateauFailureClassV2::GpDisadvantage);
            }
            if diagnosis
                .latency_summary
                .split_whitespace()
                .last()
                .and_then(|value| value.parse::<f64>().ok())
                .is_some_and(|ratio| ratio > 1.25)
            {
                diagnosed.insert(PlateauFailureClassV2::LatencyRegression);
            }
        }
        ensure!(
            !self.failure_classes.is_empty() && self.failure_classes == diagnosed,
            "recipe change directive v2 must answer the diagnosed failure classes exactly"
        );
        ensure!(
            self.failure_classes
                .iter()
                .all(|class| self.changed_knobs.contains(&class.required_knob())),
            "recipe change directive v2 does not change the knob required by its diagnosis"
        );
        let diagnosed_recipe = diagnosis
            .current_recipe_identity_sha256
            .as_ref()
            .context("recipe change directive v2 requires a recipe-scoped diagnosis")
            .and_then(|identity| Sha256Digest::parse(identity.clone()))?;
        ensure!(
            self.previous_recipe_identity_sha256.as_ref() == Some(&diagnosed_recipe),
            "recipe change directive v2 previous recipe does not match its diagnosis"
        );
        ensure!(
            diagnosed_recipe != self.next_recipe_identity_sha256,
            "recipe change directive v2 does not change recipe identity"
        );
        Ok(())
    }
}

impl PlateauDiagnosisV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == PLATEAU_DIAGNOSIS_V2_SCHEMA,
            "plateau diagnosis v2 schema mismatch"
        );
        ensure!(
            self.counted_candidates.len() >= 3
                && self
                    .counted_candidates
                    .iter()
                    .map(|candidate| &candidate.candidate_checkpoint_sha256)
                    .collect::<BTreeSet<_>>()
                    .len()
                    == self.counted_candidates.len(),
            "plateau diagnosis v2 requires three distinct counted candidates"
        );
        ensure!(
            self.proposal_validity.is_finite() && self.gp_disadvantage.is_finite(),
            "plateau diagnosis v2 contains a non-finite metric"
        );
        if let Some(identity) = &self.current_recipe_identity_sha256 {
            Sha256Digest::parse(identity.clone())
                .context("plateau diagnosis v2 has an invalid recipe identity")?;
        }
        Ok(())
    }
}

/// Operator receipt attesting that the training recipe HAS changed in a way
/// receipts cannot prove (the six legacy rejects predate recipe stamping in
/// TrainerIntent.v3). Its only effect: rejected shadows whose recipe
/// identity is UNRESOLVABLE stop counting toward the named parent's
/// plateau. Rejects that resolve to the currently-scheduled recipe identity
/// still pause — ground truth beats attestation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeChangeDirectiveV1 {
    pub schema: String,
    pub task: Task,
    pub unchanged_parent_checkpoint_sha256: Sha256Digest,
    /// Path or digest of the plateau diagnosis this directive answers.
    pub answers_diagnosis: String,
    /// The concrete, enumerated recipe changes the operator is attesting to.
    pub enumerated_changes: Vec<String>,
    pub created_at_ms: u128,
}

impl RecipeChangeDirectiveV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == RECIPE_CHANGE_DIRECTIVE_V1_SCHEMA,
            "recipe change directive schema mismatch"
        );
        ensure!(
            !self.answers_diagnosis.trim().is_empty(),
            "recipe change directive must name the diagnosis it answers"
        );
        ensure!(
            !self.enumerated_changes.is_empty()
                && self
                    .enumerated_changes
                    .iter()
                    .all(|change| !change.trim().is_empty()),
            "recipe change directive requires enumerated concrete changes"
        );
        Ok(())
    }
}

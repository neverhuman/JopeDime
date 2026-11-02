use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelfplayShadowOrigin {
    DlDirect,
    ShadowDlDirect,
    GpComparator,
    GpExploration,
    CachedReference,
    Requalification,
}

/// Backward-compatible name for a selfplay shadow candidate's origin.
pub type Origin = SelfplayShadowOrigin;

impl SelfplayShadowOrigin {
    pub fn is_gp(self) -> bool {
        matches!(self, Self::GpComparator | Self::GpExploration)
    }

    pub fn is_new_action(self) -> bool {
        !matches!(self, Self::CachedReference)
    }
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
#[serde(rename_all = "snake_case")]
pub enum DlAncestryKind {
    #[default]
    None,
    AcceptedDl,
    ShadowDl,
    Mixed,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Parentage {
    #[serde(default)]
    pub parent_semantic_sha256: Vec<Sha256Digest>,
    #[serde(default)]
    pub has_dl_parent_or_ancestor: bool,
    #[serde(default)]
    pub dl_ancestry_kind: DlAncestryKind,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionSignals {
    pub conditional_reward: f64,
    pub stability: f64,
    pub incumbent_lift_probability: f64,
    pub diversity: f64,
    /// Recorded for calibration, but intentionally receives zero ranking
    /// weight until a family-held calibration proves monotonic utility.
    pub raw_uncertainty: f64,
}

impl SelectionSignals {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            [
                self.conditional_reward,
                self.stability,
                self.incumbent_lift_probability,
                self.diversity,
                self.raw_uncertainty,
            ]
            .into_iter()
            .all(f64::is_finite),
            "selection signals must be finite"
        );
        ensure!(
            (0.0..=1.0).contains(&self.incumbent_lift_probability),
            "incumbent lift probability must be in [0, 1]"
        );
        Ok(())
    }

    pub fn action_score(&self) -> f64 {
        // raw_uncertainty is deliberately absent.
        self.conditional_reward
            + 0.5 * self.stability
            + 2.0 * self.incumbent_lift_probability
            + 0.25 * self.diversity
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetContext {
    pub task: Task,
    pub dataset_id: String,
    pub dataset_content_sha256: Sha256Digest,
    pub source_family: String,
    pub context_sha256: Sha256Digest,
}

impl DatasetContext {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.dataset_id.trim().is_empty(), "dataset id is empty");
        ensure!(
            !self.source_family.trim().is_empty(),
            "source family is empty"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyCandidate {
    pub task: Task,
    pub dataset_id: String,
    pub dataset_content_sha256: Sha256Digest,
    pub context_sha256: Sha256Digest,
    pub canonical_program: String,
    pub semantic_sha256: Sha256Digest,
    pub source_family: String,
    pub origin: Origin,
    pub parentage: Parentage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub producer_checkpoint_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behavior_checkpoint_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder_log_probability: Option<f64>,
    pub proposal_rank: u32,
    pub seed: u64,
    pub signals: SelectionSignals,
}

impl PolicyCandidate {
    pub fn validate_for(
        &self,
        context: &DatasetContext,
        accepted_parent: &Sha256Digest,
    ) -> Result<()> {
        context.validate()?;
        self.signals.validate()?;
        ensure!(self.task == context.task, "candidate task/context mismatch");
        ensure!(
            self.dataset_id == context.dataset_id
                && self.dataset_content_sha256 == context.dataset_content_sha256
                && self.context_sha256 == context.context_sha256,
            "cross-context candidate backfill is forbidden"
        );
        ensure!(
            !self.canonical_program.trim().is_empty(),
            "canonical program is empty"
        );
        ensure!(
            Sha256Digest::of_bytes(self.canonical_program.as_bytes()) == self.semantic_sha256,
            "semantic digest does not match authoritative canonical bytes"
        );
        ensure!(
            !self.source_family.trim().is_empty(),
            "candidate source family is empty"
        );
        if let Some(value) = self.decoder_log_probability {
            ensure!(value.is_finite(), "decoder log probability must be finite");
        }
        match self.origin {
            Origin::DlDirect => {
                ensure!(
                    self.producer_checkpoint_sha256.as_ref() == Some(accepted_parent),
                    "dl_direct must be produced by the task's accepted parent"
                );
                ensure!(
                    self.behavior_checkpoint_sha256.as_ref() == Some(accepted_parent),
                    "dl_direct behavior checkpoint must equal its accepted parent"
                );
                ensure!(
                    self.parentage.parent_semantic_sha256.is_empty()
                        && !self.parentage.has_dl_parent_or_ancestor
                        && self.parentage.dl_ancestry_kind == DlAncestryKind::None,
                    "dl_direct must be a native checkpoint action, not an offspring"
                );
            }
            Origin::ShadowDlDirect => {
                ensure!(
                    self.producer_checkpoint_sha256.is_some()
                        && self.behavior_checkpoint_sha256 == self.producer_checkpoint_sha256,
                    "shadow_dl_direct requires one explicit shadow producer/behavior checkpoint"
                );
                ensure!(
                    self.parentage.parent_semantic_sha256.is_empty()
                        && !self.parentage.has_dl_parent_or_ancestor
                        && self.parentage.dl_ancestry_kind == DlAncestryKind::None,
                    "shadow_dl_direct must be a native shadow checkpoint action"
                );
                ensure!(
                    self.producer_checkpoint_sha256.as_ref() != Some(accepted_parent),
                    "shadow_dl_direct cannot masquerade as accepted-DL evidence"
                );
            }
            Origin::GpComparator | Origin::GpExploration => {
                ensure!(
                    self.producer_checkpoint_sha256.is_none(),
                    "GP candidates cannot carry a producer checkpoint"
                );
                ensure!(
                    !self.parentage.has_dl_parent_or_ancestor
                        && self.parentage.dl_ancestry_kind == DlAncestryKind::None,
                    "GP candidates cannot have a DL parent or ancestor"
                );
            }
            Origin::CachedReference => {
                ensure!(
                    self.producer_checkpoint_sha256.is_none(),
                    "cached references are not checkpoint actions"
                );
            }
            Origin::Requalification => {
                ensure!(
                    self.producer_checkpoint_sha256.is_none(),
                    "protocol requalification is not a direct policy action"
                );
            }
        }
        Ok(())
    }
}

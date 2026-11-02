use super::*;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProtocolIdentity {
    pub objective_sha256: String,
    pub recipe_sha256: String,
    pub architecture_sha256: String,
    pub vocabulary_sha256: String,
    pub support_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PolicyAnchor {
    pub checkpoint_sha256: String,
    pub weights_sha256: String,
    pub optimizer_step: u64,
    /// Historical unknowns stay unknown; the descendant cannot certify them.
    pub objective_sha256: Option<String>,
    pub committed_updates: Option<u64>,
    pub architecture_sha256: String,
    pub vocabulary_sha256: String,
    pub support_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OptimizerEpoch {
    pub schema: String,
    pub origin_checkpoint_sha256: String,
    /// Identifies an actual reset attempt; neither an optimizer nor a witness counter.
    pub reset_nonce_sha256: String,
    pub recipe_sha256: String,
}
impl OptimizerEpoch {
    pub fn sha256(&self) -> Result<String> {
        ensure!(self.schema == EPOCH_SCHEMA, "unsupported optimizer epoch");
        for d in [
            &self.origin_checkpoint_sha256,
            &self.reset_nonce_sha256,
            &self.recipe_sha256,
        ] {
            digest(d)?;
        }
        object_id(EPOCH_SCHEMA, &canonical_bytes(self)?)
    }
}

/// Identity allocation only. I12 owns executable intent freezing and execution.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TrainingIntent {
    pub schema: String,
    pub parent_checkpoint_sha256: String,
    pub protocol: ProtocolIdentity,
    pub corpus_sha256: String,
    pub optimizer_epoch_sha256: String,
    pub initial_step: u64,
    pub target_step: u64,
    pub execution_partition_sha256: String,
    pub sampler_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TrainingEligibility {
    pub schema: String,
    pub purpose: String,
    pub intent_sha256: String,
    pub corpus_sha256: String,
    pub role_exposure_closure_sha256: String,
    /// Only the training role is accepted by this projection. No evaluation feedback.
    pub source_roles: BTreeSet<String>,
}

/// A durable applied-update receipt projection, never inferred from a step label.
/// I13 must bind its commit identity to the complete atomic checkpoint closure.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CommittedUpdateCredit {
    pub schema: String,
    pub commit_sha256: String,
    pub parent_checkpoint_sha256: String,
    pub checkpoint_sha256: String,
    pub optimizer_epoch_sha256: String,
    pub intent_sha256: String,
    pub eligibility_sha256: String,
    pub rank: u64,
    pub world: u64,
    pub initial_update: u64,
    pub target_update: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LineageEdge {
    pub schema: String,
    pub previous_edge_sha256: Option<String>,
    pub parent_checkpoint_sha256: String,
    pub checkpoint_sha256: String,
    pub weights_sha256: String,
    pub optimizer_sha256: String,
    pub mode: ContinuationMode,
    pub epoch: OptimizerEpoch,
    pub intent: TrainingIntent,
    pub committed_credit_sha256: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PolicyLineage {
    pub schema: String,
    pub anchor: PolicyAnchor,
    pub protocol: ProtocolIdentity,
    /// Ordered content addresses, starting at the immutable anchor.
    pub edges: Vec<String>,
}

/// Small embedded closure, suitable for a reviewed policy. No path or network reads.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LineageBundle {
    pub lineage: PolicyLineage,
    /// Address -> exact canonical JSON bytes, UTF-8 encoded.
    pub evidence: BTreeMap<String, String>,
}

/// Additive predecessor adapter. The original completion artifact keeps its bytes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineageSuccessorEvidence {
    pub schema: String,
    pub completion: crate::native_graph::Artifact,
    pub bundle: LineageBundle,
}
pub const SUCCESSOR_EVIDENCE_SCHEMA: &str = "jope.native-lineage-successor/v1";

/// Coordinator-owned OPTIONAL signed WitnessPolicy field. Absent means legacy rules.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DescendantLineagePolicy {
    pub schema: String,
    pub run: String,
    pub scientific_policy_sha256: String,
    pub previous_completion_sha256: String,
    pub bundle: LineageBundle,
}

/// Created only by the bounded verifier. Counters describe this epoch and committed
/// descendant exposure separately; a legacy anchor's exposure can remain unknown.
#[derive(Clone, Debug)]
pub struct VerifiedLineage {
    pub(crate) lineage: PolicyLineage,
    pub(crate) edges: Vec<LineageEdge>,
    pub(crate) committed_updates: u64,
    pub(crate) cumulative_updates: Option<u64>,
}
impl VerifiedLineage {
    pub fn lineage(&self) -> &PolicyLineage {
        &self.lineage
    }
    pub fn edges(&self) -> &[LineageEdge] {
        &self.edges
    }
    pub fn tip(&self) -> &LineageEdge {
        self.edges.last().expect("verified nonempty lineage")
    }
    pub fn committed_updates(&self) -> u64 {
        self.committed_updates
    }
    pub fn cumulative_updates(&self) -> Option<u64> {
        self.cumulative_updates
    }
    pub fn local_step(&self) -> u64 {
        self.tip().intent.target_step
    }
}

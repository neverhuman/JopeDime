//! Additive ancestry evidence. Verification establishes byte-bound fixture/receipt
//! continuity; the existing signed run, witness and execution grants remain mandatory.
use crate::{manifest::hash, task_contract::ContinuationMode};
use anyhow::{ensure, Context, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const LINEAGE_SCHEMA: &str = "jope.policy-lineage/v1";
pub const INTENT_SCHEMA: &str = "jope.training-intent/v1";
pub const EPOCH_SCHEMA: &str = "jope.optimizer-epoch/v1";
pub const EDGE_SCHEMA: &str = "jope.policy-lineage-edge/v1";
pub const CREDIT_SCHEMA: &str = "jope.committed-update-credit/v1";
/// Draft I04 adapter: the integrator must bind this projection to accepted role evidence.
pub const ELIGIBILITY_SCHEMA: &str = "jope.lineage-training-eligibility/v1";
pub const DESCENDANT_POLICY_SCHEMA: &str = "jope.descendant-lineage-policy/v1";
/// Independent of the witness authorization-epoch limit.
pub const MAX_LINEAGE_EDGES: usize = 64;
pub const MAX_LINEAGE_BYTES: usize = 1024 * 1024;
pub const MAX_EVIDENCE_BYTES: usize = 8 * 1024 * 1024;
const MAX_RANKS: u64 = 64;
const SAFE_INTEGER: u64 = 9_007_199_254_740_991;

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
    lineage: PolicyLineage,
    edges: Vec<LineageEdge>,
    committed_updates: u64,
    cumulative_updates: Option<u64>,
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

fn digest(d: &str) -> Result<()> {
    ensure!(
        d.len() == 64
            && d.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "invalid lineage digest"
    );
    Ok(())
}
fn protocol(p: &ProtocolIdentity) -> Result<()> {
    for d in [
        &p.objective_sha256,
        &p.recipe_sha256,
        &p.architecture_sha256,
        &p.vocabulary_sha256,
        &p.support_sha256,
    ] {
        digest(d)?;
    }
    Ok(())
}

/// Cross-check a retained manifest at an actual native/ancestry entrypoint.
/// Missing epoch/recipe bindings stay held until I13 supplies committed manifests.
pub fn check_checkpoint(state: &serde_json::Value, edge: &LineageEdge) -> Result<()> {
    ensure!(
        state["schema"] == "parallel.checkpoint.v1"
            && state["step"].as_u64() == Some(edge.intent.target_step)
            && state["model_sha256"] == edge.weights_sha256
            && state["optimizer_sha256"] == edge.optimizer_sha256
            && state["optimizer_epoch_sha256"] == edge.intent.optimizer_epoch_sha256
            && state["optimizer_recipe_sha256"] == edge.epoch.recipe_sha256
            && state["objective_sha256"] == edge.intent.protocol.objective_sha256
            && state["vocabulary_sha256"] == edge.intent.protocol.vocabulary_sha256
            && state["corpus_sha256"] == edge.intent.corpus_sha256,
        "retained checkpoint differs from committed lineage/epoch closure"
    );
    Ok(())
}
fn integer(n: u64) -> Result<()> {
    ensure!(
        n <= SAFE_INTEGER,
        "lineage integer overflow or outside canonical safe range"
    );
    Ok(())
}

/// Canonical sorted-key JSON for the restricted wire DTOs (strings/integers, no floats).
pub fn canonical_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    fn check(v: &mut serde_json::Value) -> Result<()> {
        match v {
            serde_json::Value::Object(m) => {
                m.sort_keys();
                for v in m.values_mut() {
                    check(v)?;
                }
            }
            serde_json::Value::Array(a) => {
                for v in a {
                    check(v)?;
                }
            }
            serde_json::Value::Number(n) => integer(
                n.as_u64()
                    .context("lineage identity needs nonnegative integers")?,
            )?,
            _ => (),
        }
        Ok(())
    }
    let mut v = serde_json::to_value(value)?;
    check(&mut v)?;
    Ok(serde_json::to_vec(&v)?)
}

/// I02's adopted formula, restricted here to typed lineage payloads. It is independent
/// of historical checkpoint raw-byte SHA256 identities.
pub fn object_id(schema: &str, bytes: &[u8]) -> Result<String> {
    let (family, version) = schema
        .strip_prefix("jope.")
        .and_then(|s| s.rsplit_once("/v"))
        .context("invalid canonical schema ID")?;
    ensure!(
        (8..=128).contains(&schema.len())
            && !family.is_empty()
            && family
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"._-".contains(&b))
            && !version.is_empty()
            && !version.starts_with('0')
            && version.bytes().all(|b| b.is_ascii_digit()),
        "invalid canonical schema ID"
    );
    ensure!(bytes.len() >= 2, "canonical payload byte bound");
    ensure!(
        bytes.len() <= MAX_LINEAGE_BYTES,
        "lineage object byte bound"
    );
    let mut framed = Vec::with_capacity(schema.len() + 9 + bytes.len());
    framed.extend_from_slice(schema.as_bytes());
    framed.push(0);
    framed.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
    framed.extend_from_slice(bytes);
    Ok(hash(&framed))
}

fn load<T: DeserializeOwned + Serialize>(
    schema: &str,
    id: &str,
    source: &impl Fn(&str) -> Result<Vec<u8>>,
    used: &mut usize,
) -> Result<T> {
    digest(id)?;
    let bytes = source(id).with_context(|| format!("missing lineage evidence {id}"))?;
    *used = used
        .checked_add(bytes.len())
        .context("lineage evidence overflow")?;
    ensure!(*used <= MAX_EVIDENCE_BYTES, "lineage evidence byte bound");
    ensure!(object_id(schema, &bytes)? == id, "altered lineage evidence");
    let value: T = serde_json::from_slice(&bytes)?;
    ensure!(
        canonical_bytes(&value)? == bytes,
        "noncanonical lineage evidence"
    );
    Ok(value)
}

impl LineageBundle {
    pub fn verify(&self) -> Result<VerifiedLineage> {
        ensure!(
            self.evidence.len() <= MAX_LINEAGE_EDGES * (MAX_RANKS as usize + 2),
            "lineage object count bound"
        );
        let size = self.evidence.values().try_fold(0usize, |n, s| {
            n.checked_add(s.len()).context("evidence size overflow")
        })?;
        ensure!(size <= MAX_EVIDENCE_BYTES, "lineage evidence byte bound");
        verify(&self.lineage, &|id| {
            Ok(self
                .evidence
                .get(id)
                .context("lineage evidence absent")?
                .as_bytes()
                .to_vec())
        })
    }
}

pub fn verify(
    lineage: &PolicyLineage,
    source: &impl Fn(&str) -> Result<Vec<u8>>,
) -> Result<VerifiedLineage> {
    ensure!(
        lineage.schema == LINEAGE_SCHEMA,
        "unsupported policy lineage"
    );
    ensure!(
        !lineage.edges.is_empty() && lineage.edges.len() <= MAX_LINEAGE_EDGES,
        "lineage edge/depth bound"
    );
    ensure!(
        canonical_bytes(lineage)?.len() <= MAX_LINEAGE_BYTES,
        "lineage byte bound"
    );
    protocol(&lineage.protocol)?;
    let a = &lineage.anchor;
    for d in [
        &a.checkpoint_sha256,
        &a.weights_sha256,
        &a.architecture_sha256,
        &a.vocabulary_sha256,
        &a.support_sha256,
    ] {
        digest(d)?;
    }
    integer(a.optimizer_step)?;
    if let Some(n) = a.committed_updates {
        integer(n)?;
    }
    if let Some(d) = &a.objective_sha256 {
        digest(d)?;
        ensure!(
            *d == lineage.protocol.objective_sha256,
            "anchor objective mismatch"
        );
    }
    ensure!(
        a.committed_updates.is_none(),
        "anchor exposure cannot be relabeled as complete without an adopted anchor receipt protocol"
    );
    ensure!(
        a.architecture_sha256 == lineage.protocol.architecture_sha256
            && a.vocabulary_sha256 == lineage.protocol.vocabulary_sha256
            && a.support_sha256 == lineage.protocol.support_sha256,
        "unsupported architecture/vocabulary/support migration"
    );
    let mut seen_edges = BTreeSet::new();
    let mut seen_checkpoints = BTreeSet::from([a.checkpoint_sha256.clone()]);
    let mut seen_epochs = BTreeSet::new();
    let mut seen_commits = BTreeSet::new();
    let mut seen_credit = BTreeSet::new();
    let mut edges: Vec<LineageEdge> = Vec::new();
    let mut used = 0usize;
    let mut updates = 0u64;
    for (index, id) in lineage.edges.iter().enumerate() {
        ensure!(seen_edges.insert(id), "duplicate/cyclic lineage edge");
        let edge: LineageEdge = load(EDGE_SCHEMA, id, source, &mut used)?;
        let prior = edges.last();
        let parent = prior.map_or(&a.checkpoint_sha256, |p| &p.checkpoint_sha256);
        ensure!(
            edge.schema == EDGE_SCHEMA
                && edge.parent_checkpoint_sha256 == *parent
                && edge.previous_edge_sha256.as_ref()
                    == index.checked_sub(1).map(|i| &lineage.edges[i]),
            "missing or forked lineage edge"
        );
        for d in [
            &edge.checkpoint_sha256,
            &edge.weights_sha256,
            &edge.optimizer_sha256,
        ] {
            digest(d)?;
        }
        ensure!(
            seen_checkpoints.insert(edge.checkpoint_sha256.clone()),
            "lineage checkpoint cycle"
        );
        let epoch = edge.epoch.sha256()?;
        let intent = &edge.intent;
        ensure!(
            intent.schema == INTENT_SCHEMA
                && intent.parent_checkpoint_sha256 == *parent
                && intent.protocol == lineage.protocol
                && intent.optimizer_epoch_sha256 == epoch
                && edge.epoch.recipe_sha256 == lineage.protocol.recipe_sha256,
            "stale intent or objective/recipe mismatch"
        );
        for d in [
            &intent.corpus_sha256,
            &intent.execution_partition_sha256,
            &intent.sampler_sha256,
        ] {
            digest(d)?;
        }
        integer(intent.initial_step)?;
        integer(intent.target_step)?;
        let delta = intent
            .target_step
            .checked_sub(intent.initial_step)
            .context("lineage update underflow")?;
        ensure!(delta > 0, "empty committed update delta");
        match edge.mode {
            ContinuationMode::WeightWarmStart => ensure!(
                intent.initial_step == 0
                    && edge.epoch.origin_checkpoint_sha256 == *parent
                    && seen_epochs.insert(epoch.clone()),
                "reset must start a fresh optimizer epoch at zero"
            ),
            ContinuationMode::NewCorpusTransition => {
                let p = prior.context("unknown anchor optimizer/objective cannot be carried")?;
                ensure!(
                    p.epoch == edge.epoch && intent.initial_step == p.intent.target_step,
                    "optimizer carry changed epoch or local step"
                );
            }
            _ => anyhow::bail!("cold initialization/exact resume is not a new ancestry edge"),
        }
        let intent_id = object_id(INTENT_SCHEMA, &canonical_bytes(intent)?)?;
        ensure!(
            !edge.committed_credit_sha256.is_empty()
                && edge.committed_credit_sha256.len() <= MAX_RANKS as usize,
            "missing or oversized committed credit"
        );
        let mut ranks = BTreeSet::new();
        let mut world = None;
        let mut commit = None;
        for credit_id in &edge.committed_credit_sha256 {
            ensure!(
                seen_credit.insert(credit_id.clone()),
                "reused committed update credit"
            );
            let c: CommittedUpdateCredit = load(CREDIT_SCHEMA, credit_id, source, &mut used)?;
            digest(&c.commit_sha256)?;
            ensure!(
                c.schema == CREDIT_SCHEMA
                    && c.parent_checkpoint_sha256 == *parent
                    && c.checkpoint_sha256 == edge.checkpoint_sha256
                    && c.optimizer_epoch_sha256 == epoch
                    && c.intent_sha256 == intent_id
                    && c.initial_update == intent.initial_step
                    && c.target_update == intent.target_step
                    && c.world > 0
                    && c.world <= MAX_RANKS
                    && c.rank < c.world,
                "committed credit differs from edge/intent/update identity"
            );
            ensure!(
                ranks.insert(c.rank)
                    && world.is_none_or(|w| w == c.world)
                    && commit.as_ref().is_none_or(|d| d == &c.commit_sha256)
                    && seen_commits.insert((epoch.clone(), c.commit_sha256.clone(), c.rank)),
                "forked or reused epoch/commit/rank credit"
            );
            world = Some(c.world);
            commit = Some(c.commit_sha256);
            let eligible: TrainingEligibility =
                load(ELIGIBILITY_SCHEMA, &c.eligibility_sha256, source, &mut used)?;
            ensure!(
                eligible.schema == ELIGIBILITY_SCHEMA
                    && eligible.purpose == "training"
                    && eligible.intent_sha256 == intent_id
                    && eligible.corpus_sha256 == intent.corpus_sha256
                    && eligible.source_roles == BTreeSet::from(["train".into()]),
                "ineligible training role/exposure closure"
            );
            digest(&eligible.role_exposure_closure_sha256)?;
        }
        ensure!(
            ranks.len() as u64 == world.context("missing rank credit")?,
            "incomplete rank credit"
        );
        updates = updates
            .checked_add(delta)
            .context("committed exposure overflow")?;
        integer(updates)?;
        edges.push(edge);
    }
    let cumulative = a
        .committed_updates
        .map(|n| {
            n.checked_add(updates)
                .context("cumulative exposure overflow")
        })
        .transpose()?;
    if let Some(n) = cumulative {
        integer(n)?;
    }
    Ok(VerifiedLineage {
        lineage: lineage.clone(),
        edges,
        committed_updates: updates,
        cumulative_updates: cumulative,
    })
}

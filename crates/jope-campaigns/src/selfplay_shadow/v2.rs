use super::{
    DlAncestryKind, Sha256Digest, Task, TrainerMode, CANARY_INTENT_SCHEMA,
    CANDIDATE_PROVENANCE_SCHEMA, CATALOG_BINDING_SCHEMA, CAUSAL_EVALUATION_RECEIPT_SCHEMA,
    CAUSAL_EVIDENCE_LEDGER_V2_SCHEMA, CORPUS_FREEZE_STATE_SCHEMA, FLYWHEEL_LIVENESS_SCHEMA,
    MIN_PARENT_DIRECT_OUTCOMES, PACKED_CORPUS_SCHEMA, TASK_CANARY_ACCEPTANCE_V2_SCHEMA,
    TRAINER_INTENT_V3_SCHEMA,
};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

pub const REQUESTED_CAUSAL_PROPOSALS_PER_CONTEXT: usize = 8;
pub const CANARY_PROPOSALS_PER_LANE: usize = 8;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogBindingV1 {
    pub schema: String,
    pub catalog_path: PathBuf,
    pub catalog_sha256: Sha256Digest,
    pub data_path: PathBuf,
    pub data_sha256: Sha256Digest,
    pub pointer_snapshot_path: PathBuf,
    pub pointer_snapshot_sha256: Sha256Digest,
    pub created_at_ms: u128,
}

impl CatalogBindingV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CATALOG_BINDING_SCHEMA,
            "catalog binding schema is not v1"
        );
        ensure!(
            self.catalog_path.is_absolute()
                && self.data_path.is_absolute()
                && self.pointer_snapshot_path.is_absolute(),
            "catalog binding paths must be absolute"
        );
        Ok(())
    }

    pub fn identity(&self) -> Result<Sha256Digest> {
        self.validate()?;
        Ok(Sha256Digest::of_bytes(&serde_json::to_vec(self)?))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusFreezeStage {
    RowsFrozen,
    Packed,
    Published,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusFreezeStateV2 {
    pub schema: String,
    pub watermark: u64,
    pub catalog_binding_path: PathBuf,
    pub catalog_binding_sha256: Sha256Digest,
    pub selected_rows_path: PathBuf,
    pub selected_rows_sha256: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed_corpus_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed_corpus_sha256: Option<Sha256Digest>,
    pub stage: CorpusFreezeStage,
    pub revision: u64,
}

impl CorpusFreezeStateV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CORPUS_FREEZE_STATE_SCHEMA,
            "corpus freeze state schema is not v2"
        );
        ensure!(self.watermark > 0, "corpus freeze watermark is zero");
        ensure!(
            self.stage == CorpusFreezeStage::RowsFrozen
                || (self.packed_corpus_path.is_some() && self.packed_corpus_sha256.is_some()),
            "packed or published freeze lacks its packed corpus identity"
        );
        Ok(())
    }

    pub fn advance(
        &mut self,
        next: CorpusFreezeStage,
        packed: Option<(PathBuf, Sha256Digest)>,
    ) -> Result<()> {
        let legal = matches!(
            (self.stage, next),
            (CorpusFreezeStage::RowsFrozen, CorpusFreezeStage::Packed)
                | (CorpusFreezeStage::Packed, CorpusFreezeStage::Published)
        );
        ensure!(legal, "corpus freeze stages must advance exactly once");
        if let Some((path, digest)) = packed {
            self.packed_corpus_path = Some(path);
            self.packed_corpus_sha256 = Some(digest);
        }
        self.stage = next;
        self.revision = self.revision.saturating_add(1);
        self.validate()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PackedCorpusShardV2 {
    pub first_receipt: u64,
    pub last_receipt: u64,
    pub rows: usize,
    pub path: PathBuf,
    pub sha256: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reused_from_corpus_sha256: Option<Sha256Digest>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PackedCorpusV2 {
    pub schema: String,
    pub watermark: u64,
    pub catalog_binding_sha256: Sha256Digest,
    pub selected_row_index_path: PathBuf,
    pub selected_row_index_sha256: Sha256Digest,
    pub task_columns: BTreeMap<Task, PathBuf>,
    pub shards: Vec<PackedCorpusShardV2>,
}

impl PackedCorpusV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == PACKED_CORPUS_SCHEMA,
            "packed corpus schema is not v2"
        );
        ensure!(
            self.task_columns.contains_key(&Task::Binary)
                && self.task_columns.contains_key(&Task::Regression),
            "packed corpus lacks a memory-mapped task column"
        );
        ensure!(!self.shards.is_empty(), "packed corpus has no label shards");
        let mut previous = 0;
        for shard in &self.shards {
            ensure!(
                shard.rows > 0
                    && shard.first_receipt <= shard.last_receipt
                    && shard.first_receipt > previous,
                "packed corpus shards overlap or are unordered"
            );
            previous = shard.last_receipt;
        }
        ensure!(
            previous <= self.watermark,
            "packed shard exceeds the frozen watermark"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateProvenanceV2 {
    pub schema: String,
    pub candidate_sha256: Sha256Digest,
    #[serde(default)]
    pub parents: Vec<Sha256Digest>,
    pub ancestry: DlAncestryKind,
}

impl CandidateProvenanceV2 {
    pub fn inherited(
        candidate_sha256: Sha256Digest,
        parents: Vec<Sha256Digest>,
        parent_ancestry: impl IntoIterator<Item = DlAncestryKind>,
    ) -> Self {
        let kinds = parent_ancestry.into_iter().collect::<BTreeSet<_>>();
        let ancestry = if kinds.is_empty() || kinds == BTreeSet::from([DlAncestryKind::None]) {
            DlAncestryKind::None
        } else if kinds == BTreeSet::from([DlAncestryKind::AcceptedDl]) {
            DlAncestryKind::AcceptedDl
        } else if kinds == BTreeSet::from([DlAncestryKind::ShadowDl]) {
            DlAncestryKind::ShadowDl
        } else {
            DlAncestryKind::Mixed
        };
        Self {
            schema: CANDIDATE_PROVENANCE_SCHEMA.to_owned(),
            candidate_sha256,
            parents,
            ancestry,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CANDIDATE_PROVENANCE_SCHEMA,
            "candidate provenance schema is not v2"
        );
        let unique = self.parents.iter().collect::<BTreeSet<_>>();
        ensure!(
            unique.len() == self.parents.len(),
            "candidate provenance repeats a parent"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalLane {
    NativeDlDirect,
    MatchedGpComparator,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalEvaluationReceiptV2 {
    pub schema: String,
    pub receipt_sha256: Sha256Digest,
    pub manifest_sha256: Sha256Digest,
    pub candidate_sha256: Sha256Digest,
    pub task: Task,
    pub context_identity: String,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    pub wave: u64,
    pub lane: CausalLane,
    pub pair_index: usize,
    pub pair_id: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    pub success: bool,
    pub proposal_valid: bool,
    pub runtime_ms: u64,
    pub ancestry: DlAncestryKind,
    pub cross_context_backfill: bool,
    pub completed_causal_wave: bool,
    pub valid_lineage: bool,
    #[serde(default)]
    pub canary: bool,
}

pub fn causal_pair_id(
    parent: &Sha256Digest,
    wave: u64,
    task: Task,
    context_identity: &str,
    pair_index: usize,
) -> Sha256Digest {
    Sha256Digest::of_bytes(
        format!(
            "CausalPair.v1\0{}\0{wave}\0{}\0{context_identity}\0{pair_index}",
            parent,
            task.as_str()
        )
        .as_bytes(),
    )
}

impl CausalEvaluationReceiptV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CAUSAL_EVALUATION_RECEIPT_SCHEMA,
            "causal evaluation receipt schema is not v2"
        );
        ensure!(
            self.wave >= 191,
            "wave 189 and pre-causal waves are not receipt-eligible"
        );
        ensure!(
            !self.context_identity.trim().is_empty(),
            "causal receipt context is empty"
        );
        ensure!(
            self.score.map(f64::is_finite).unwrap_or(true),
            "causal receipt score is not finite"
        );
        ensure!(
            self.pair_id
                == causal_pair_id(
                    &self.accepted_parent_checkpoint_sha256,
                    self.wave,
                    self.task,
                    &self.context_identity,
                    self.pair_index,
                ),
            "causal receipt pair identity drift"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairRepairRequestV1 {
    pub pair_id: Sha256Digest,
    pub task: Task,
    pub missing_lane: CausalLane,
    pub original_manifest_sha256: Sha256Digest,
    pub permanent_failure: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalEvidenceExclusionsV2 {
    pub canary: usize,
    pub duplicate_receipt: usize,
    pub failed_or_invalid: usize,
    pub wrong_parent: usize,
    pub invalid_lineage: usize,
    pub cross_context_backfill: usize,
    pub comparator_dl_ancestry: usize,
    pub incomplete_wave: usize,
    pub partial_pair: usize,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCausalEvidenceV2 {
    pub accepted_parent_checkpoint_sha256: Option<Sha256Digest>,
    pub requested_slots: usize,
    pub valid_unique_proposals: usize,
    pub proposal_validity: f64,
    pub complete_pairs: usize,
    pub completed_native_direct: usize,
    pub completed_matched_gp: usize,
    pub zero_cross_context_backfill: bool,
    pub completed_causal_waves: bool,
    pub valid_transitive_lineage: bool,
    pub production_gate_open: bool,
    #[serde(default)]
    pub gate_reasons: Vec<String>,
    pub exclusions: CausalEvidenceExclusionsV2,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalEvidenceLedgerV2 {
    pub schema: String,
    pub generated_at_ms: u128,
    pub authoritative: bool,
    pub tasks: BTreeMap<Task, TaskCausalEvidenceV2>,
    pub complete_pair_ids: Vec<Sha256Digest>,
    pub pair_repairs: Vec<PairRepairRequestV1>,
}

impl CausalEvidenceLedgerV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CAUSAL_EVIDENCE_LEDGER_V2_SCHEMA,
            "causal evidence ledger schema is not v2"
        );
        for (task, row) in &self.tasks {
            ensure!(
                row.completed_native_direct == row.complete_pairs
                    && row.completed_matched_gp == row.complete_pairs,
                "{task} receipt-level lane counts differ"
            );
            ensure!(
                row.proposal_validity.is_finite() && (0.0..=1.0).contains(&row.proposal_validity),
                "{task} requested-slot validity is outside [0,1]"
            );
            if row.production_gate_open {
                ensure!(
                    self.authoritative
                        && row.complete_pairs >= MIN_PARENT_DIRECT_OUTCOMES
                        && row.proposal_validity >= 0.99
                        && row.zero_cross_context_backfill
                        && row.completed_causal_waves
                        && row.valid_transitive_lineage,
                    "{task} production gate opened without strict receipt evidence"
                );
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct PairMembers {
    direct: BTreeMap<Sha256Digest, Vec<CausalEvaluationReceiptV2>>,
    comparator: BTreeMap<Sha256Digest, Vec<CausalEvaluationReceiptV2>>,
    manifest: Option<Sha256Digest>,
    task: Option<Task>,
}

fn lane_receipts(
    members: &PairMembers,
    lane: CausalLane,
) -> &BTreeMap<Sha256Digest, Vec<CausalEvaluationReceiptV2>> {
    match lane {
        CausalLane::NativeDlDirect => &members.direct,
        CausalLane::MatchedGpComparator => &members.comparator,
    }
}

fn receipt_is_eligible(receipt: &CausalEvaluationReceiptV2) -> bool {
    receipt.success
        && receipt.proposal_valid
        && !receipt.cross_context_backfill
        && receipt.completed_causal_wave
        && receipt.valid_lineage
        && !matches!(
            (receipt.lane, receipt.ancestry),
            (
                CausalLane::NativeDlDirect,
                DlAncestryKind::ShadowDl | DlAncestryKind::Mixed
            ) | (
                CausalLane::MatchedGpComparator,
                DlAncestryKind::AcceptedDl | DlAncestryKind::ShadowDl | DlAncestryKind::Mixed
            )
        )
}

fn eligible_lane_receipt(
    members: &PairMembers,
    lane: CausalLane,
) -> Option<&CausalEvaluationReceiptV2> {
    let candidates = lane_receipts(members, lane);
    (candidates.len() == 1)
        .then(|| candidates.values().next())
        .flatten()
        .and_then(|receipts| receipts.iter().find(|receipt| receipt_is_eligible(receipt)))
}

fn representative_lane_receipt(
    members: &PairMembers,
    lane: CausalLane,
) -> Option<&CausalEvaluationReceiptV2> {
    lane_receipts(members, lane)
        .values()
        .next()
        .and_then(|receipts| receipts.first())
}

fn lane_failure_is_permanent(members: &PairMembers, lane: CausalLane) -> bool {
    let candidates = lane_receipts(members, lane);
    if candidates.len() > 1 {
        return true;
    }
    candidates.values().next().is_some_and(|receipts| {
        receipts.iter().all(|receipt| {
            !receipt.proposal_valid
                || receipt.cross_context_backfill
                || !receipt.completed_causal_wave
                || !receipt.valid_lineage
                || matches!(
                    (receipt.lane, receipt.ancestry),
                    (
                        CausalLane::NativeDlDirect,
                        DlAncestryKind::ShadowDl | DlAncestryKind::Mixed
                    ) | (
                        CausalLane::MatchedGpComparator,
                        DlAncestryKind::AcceptedDl
                            | DlAncestryKind::ShadowDl
                            | DlAncestryKind::Mixed
                    )
                )
        })
    })
}

pub fn build_causal_evidence_ledger_v2(
    accepted_parents: &BTreeMap<Task, Sha256Digest>,
    requested_slots: &BTreeMap<Task, usize>,
    receipts: Vec<CausalEvaluationReceiptV2>,
    generated_at_ms: u128,
) -> Result<CausalEvidenceLedgerV2> {
    let mut tasks = [Task::Binary, Task::Regression]
        .into_iter()
        .map(|task| {
            (
                task,
                TaskCausalEvidenceV2 {
                    accepted_parent_checkpoint_sha256: accepted_parents.get(&task).cloned(),
                    requested_slots: requested_slots.get(&task).copied().unwrap_or_default(),
                    zero_cross_context_backfill: true,
                    completed_causal_waves: requested_slots.get(&task).copied().unwrap_or_default()
                        > 0,
                    valid_transitive_lineage: true,
                    ..TaskCausalEvidenceV2::default()
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut seen_receipts = BTreeSet::new();
    let mut valid_proposals = BTreeSet::new();
    let mut pairs: BTreeMap<Sha256Digest, PairMembers> = BTreeMap::new();

    for receipt in receipts {
        receipt.validate()?;
        let row = tasks
            .get_mut(&receipt.task)
            .expect("task rows are complete");
        if receipt.canary {
            row.exclusions.canary += 1;
            continue;
        }
        let receipt_key = (
            receipt.receipt_sha256.clone(),
            receipt.manifest_sha256.clone(),
            receipt.candidate_sha256.clone(),
        );
        if !seen_receipts.insert(receipt_key) {
            row.exclusions.duplicate_receipt += 1;
            continue;
        }
        if accepted_parents.get(&receipt.task) != Some(&receipt.accepted_parent_checkpoint_sha256) {
            row.exclusions.wrong_parent += 1;
            continue;
        }
        if receipt.lane == CausalLane::NativeDlDirect && receipt.proposal_valid {
            valid_proposals.insert((
                receipt.task,
                receipt.context_identity.clone(),
                receipt.candidate_sha256.clone(),
            ));
        }
        if !receipt.success || !receipt.proposal_valid {
            row.exclusions.failed_or_invalid += 1;
        }
        if receipt.cross_context_backfill {
            row.zero_cross_context_backfill = false;
            row.exclusions.cross_context_backfill += 1;
        }
        if !receipt.completed_causal_wave {
            row.completed_causal_waves = false;
            row.exclusions.incomplete_wave += 1;
        }
        if !receipt.valid_lineage
            || matches!(
                (receipt.lane, receipt.ancestry),
                (
                    CausalLane::NativeDlDirect,
                    DlAncestryKind::ShadowDl | DlAncestryKind::Mixed
                )
            )
        {
            row.valid_transitive_lineage = false;
            row.exclusions.invalid_lineage += 1;
        }
        if receipt.lane == CausalLane::MatchedGpComparator
            && receipt.ancestry != DlAncestryKind::None
        {
            row.exclusions.comparator_dl_ancestry += 1;
        }
        let pair = pairs.entry(receipt.pair_id.clone()).or_default();
        pair.manifest.get_or_insert(receipt.manifest_sha256.clone());
        pair.task.get_or_insert(receipt.task);
        match receipt.lane {
            CausalLane::NativeDlDirect => pair
                .direct
                .entry(receipt.candidate_sha256.clone())
                .or_default()
                .push(receipt),
            CausalLane::MatchedGpComparator => pair
                .comparator
                .entry(receipt.candidate_sha256.clone())
                .or_default()
                .push(receipt),
        }
    }

    for (task, row) in &mut tasks {
        row.valid_unique_proposals = valid_proposals
            .iter()
            .filter(|(candidate_task, _, _)| candidate_task == task)
            .count();
        row.proposal_validity = if row.requested_slots == 0 {
            0.0
        } else {
            row.valid_unique_proposals.min(row.requested_slots) as f64 / row.requested_slots as f64
        };
    }

    let mut complete_pair_ids = Vec::new();
    let mut pair_repairs = Vec::new();
    for (pair_id, members) in pairs {
        let task = members.task.expect("a pair is created from a receipt");
        let row = tasks.get_mut(&task).expect("task rows are complete");
        let direct = eligible_lane_receipt(&members, CausalLane::NativeDlDirect);
        let comparator = eligible_lane_receipt(&members, CausalLane::MatchedGpComparator);
        if direct.is_some() && comparator.is_some() {
            row.complete_pairs += 1;
            row.completed_native_direct += 1;
            row.completed_matched_gp += 1;
            complete_pair_ids.push(pair_id);
            continue;
        }
        row.exclusions.partial_pair += 1;
        let missing_lane = if direct.is_some() {
            CausalLane::MatchedGpComparator
        } else {
            CausalLane::NativeDlDirect
        };
        let permanent_failure = lane_failure_is_permanent(&members, missing_lane);
        let manifest = representative_lane_receipt(&members, missing_lane)
            .or_else(|| {
                representative_lane_receipt(
                    &members,
                    match missing_lane {
                        CausalLane::NativeDlDirect => CausalLane::MatchedGpComparator,
                        CausalLane::MatchedGpComparator => CausalLane::NativeDlDirect,
                    },
                )
            })
            .map(|receipt| receipt.manifest_sha256.clone())
            .or(members.manifest)
            .expect("a pair has a manifest");
        pair_repairs.push(PairRepairRequestV1 {
            pair_id,
            task,
            missing_lane,
            original_manifest_sha256: manifest,
            permanent_failure,
        });
    }
    complete_pair_ids.sort();
    pair_repairs.sort_by(|left, right| left.pair_id.cmp(&right.pair_id));
    for row in tasks.values_mut() {
        if row.complete_pairs < MIN_PARENT_DIRECT_OUTCOMES {
            row.gate_reasons.push(format!(
                "complete receipt pairs {}/{}",
                row.complete_pairs, MIN_PARENT_DIRECT_OUTCOMES
            ));
        }
        if row.proposal_validity < 0.99 {
            row.gate_reasons.push(format!(
                "requested-slot validity {:.4} < 0.99",
                row.proposal_validity
            ));
        }
        if !row.zero_cross_context_backfill {
            row.gate_reasons
                .push("cross-context backfill is nonzero".to_owned());
        }
        if !row.completed_causal_waves {
            row.gate_reasons
                .push("causal waves are incomplete".to_owned());
        }
        if !row.valid_transitive_lineage {
            row.gate_reasons
                .push("transitive lineage is invalid".to_owned());
        }
        row.production_gate_open = row.gate_reasons.is_empty();
    }
    let ledger = CausalEvidenceLedgerV2 {
        schema: CAUSAL_EVIDENCE_LEDGER_V2_SCHEMA.to_owned(),
        generated_at_ms,
        authoritative: true,
        tasks,
        complete_pair_ids,
        pair_repairs,
    };
    ledger.validate()?;
    Ok(ledger)
}

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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingRecipeV3 {
    pub broad_replay_fraction: f64,
    pub exact_causal_pair_fraction: f64,
    pub hard_novel_oversampling: bool,
    pub fixed_anchor_holdout: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerIntentV3 {
    pub schema: String,
    pub task: Task,
    pub mode: TrainerMode,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_parent_checkpoint_sha256: Option<Sha256Digest>,
    pub warm_start_checkpoint_sha256: Sha256Digest,
    pub corpus_identity_sha256: Sha256Digest,
    pub corpus_watermark: u64,
    pub runtime_bundle_path: PathBuf,
    pub runtime_bundle_sha256: Sha256Digest,
    pub catalog_binding_path: PathBuf,
    pub catalog_binding_sha256: Sha256Digest,
    pub recipe: TrainingRecipeV3,
    pub seed: u64,
    pub immutable_job_identity: String,
    pub immutable_spec_sha256: Sha256Digest,
    pub created_at_ms: u128,
}

impl TrainerIntentV3 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == TRAINER_INTENT_V3_SCHEMA,
            "trainer intent schema is not v3"
        );
        ensure!(
            !self.immutable_job_identity.trim().is_empty(),
            "trainer job identity is empty"
        );
        ensure!(
            self.recipe.broad_replay_fraction.is_finite()
                && self.recipe.exact_causal_pair_fraction.is_finite()
                && (self.recipe.broad_replay_fraction + self.recipe.exact_causal_pair_fraction
                    - 1.0)
                    .abs()
                    < 1e-9,
            "trainer recipe fractions must sum to one"
        );
        if self.mode == TrainerMode::Production {
            ensure!(
                self.shadow_parent_checkpoint_sha256.is_none()
                    && self.warm_start_checkpoint_sha256 == self.accepted_parent_checkpoint_sha256,
                "production training must warm-start from the accepted task pointer"
            );
        }
        if self.recipe.exact_causal_pair_fraction > 0.0 {
            ensure!(
                self.mode == TrainerMode::Shadow
                    && (self.recipe.broad_replay_fraction - 0.75).abs() < 1e-9
                    && (self.recipe.exact_causal_pair_fraction - 0.25).abs() < 1e-9
                    && self.recipe.hard_novel_oversampling
                    && self.recipe.fixed_anchor_holdout,
                "unproven causal-pair recipes are shadow-only 75/25"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FlywheelLaneLivenessV1 {
    pub eligible: bool,
    pub active: bool,
    pub last_progress_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alarm: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FlywheelLivenessV1 {
    pub schema: String,
    pub generated_at_ms: u128,
    pub queue_depth: usize,
    pub ready_reserve: usize,
    pub dynamic_target: usize,
    pub ingestion_lag_labels: u64,
    pub corpus_lag_labels: u64,
    pub proposer: FlywheelLaneLivenessV1,
    pub trainer: FlywheelLaneLivenessV1,
    pub encoder: FlywheelLaneLivenessV1,
    pub disk_used_percent: u8,
    #[serde(default)]
    pub alarms: Vec<String>,
}

impl FlywheelLivenessV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == FLYWHEEL_LIVENESS_SCHEMA,
            "flywheel liveness schema is not v1"
        );
        ensure!(
            (2_048..=4_096).contains(&self.dynamic_target),
            "dynamic supply target is outside 2048..4096"
        );
        ensure!(
            self.disk_used_percent <= 100,
            "disk usage percentage exceeds 100"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> Sha256Digest {
        Sha256Digest::parse(byte.to_string().repeat(64)).unwrap()
    }

    fn receipt(lane: CausalLane, pair_index: usize) -> CausalEvaluationReceiptV2 {
        let parent = digest('a');
        CausalEvaluationReceiptV2 {
            schema: CAUSAL_EVALUATION_RECEIPT_SCHEMA.to_owned(),
            receipt_sha256: Sha256Digest::of_bytes(
                format!("receipt-{lane:?}-{pair_index}").as_bytes(),
            ),
            manifest_sha256: digest('d'),
            candidate_sha256: Sha256Digest::of_bytes(
                format!("candidate-{lane:?}-{pair_index}").as_bytes(),
            ),
            task: Task::Binary,
            context_identity: "context".to_owned(),
            accepted_parent_checkpoint_sha256: parent.clone(),
            wave: 191,
            lane,
            pair_index,
            pair_id: causal_pair_id(&parent, 191, Task::Binary, "context", pair_index),
            score: Some(1.0),
            success: true,
            proposal_valid: true,
            runtime_ms: 10,
            ancestry: if lane == CausalLane::NativeDlDirect {
                DlAncestryKind::AcceptedDl
            } else {
                DlAncestryKind::None
            },
            cross_context_backfill: false,
            completed_causal_wave: true,
            valid_lineage: true,
            canary: false,
        }
    }

    #[test]
    fn exact_pair_matching_equalizes_lane_counts_and_repairs_partials() {
        let parents = BTreeMap::from([(Task::Binary, digest('a'))]);
        let requested = BTreeMap::from([(Task::Binary, 16)]);
        let ledger = build_causal_evidence_ledger_v2(
            &parents,
            &requested,
            vec![
                receipt(CausalLane::NativeDlDirect, 0),
                receipt(CausalLane::MatchedGpComparator, 0),
                receipt(CausalLane::NativeDlDirect, 1),
            ],
            1,
        )
        .unwrap();
        let binary = &ledger.tasks[&Task::Binary];
        assert_eq!(binary.complete_pairs, 1);
        assert_eq!(binary.completed_native_direct, binary.completed_matched_gp);
        assert_eq!(ledger.pair_repairs.len(), 1);
        assert_eq!(
            ledger.pair_repairs[0].missing_lane,
            CausalLane::MatchedGpComparator
        );
        assert!(!binary.production_gate_open);
    }

    #[test]
    fn requested_slot_validity_counts_missing_slots() {
        let parents = BTreeMap::from([(Task::Binary, digest('a'))]);
        let requested = BTreeMap::from([(Task::Binary, 8)]);
        let ledger = build_causal_evidence_ledger_v2(
            &parents,
            &requested,
            vec![receipt(CausalLane::NativeDlDirect, 0)],
            1,
        )
        .unwrap();
        assert_eq!(ledger.tasks[&Task::Binary].proposal_validity, 0.125);
    }

    #[test]
    fn successful_exact_repair_supersedes_a_failed_receipt() {
        let parents = BTreeMap::from([(Task::Binary, digest('a'))]);
        let requested = BTreeMap::from([(Task::Binary, 8)]);
        let direct = receipt(CausalLane::NativeDlDirect, 0);
        let mut failed = receipt(CausalLane::MatchedGpComparator, 0);
        failed.success = false;
        failed.score = None;
        let mut repaired = failed.clone();
        repaired.receipt_sha256 = Sha256Digest::of_bytes(b"repaired-receipt");
        repaired.success = true;
        repaired.score = Some(1.0);
        let ledger = build_causal_evidence_ledger_v2(
            &parents,
            &requested,
            vec![direct, failed, repaired],
            1,
        )
        .unwrap();
        assert_eq!(ledger.tasks[&Task::Binary].complete_pairs, 1);
        assert!(ledger.pair_repairs.is_empty());
    }

    #[test]
    fn ambiguous_lane_candidates_are_permanently_excluded() {
        let parents = BTreeMap::from([(Task::Binary, digest('a'))]);
        let requested = BTreeMap::from([(Task::Binary, 8)]);
        let direct = receipt(CausalLane::NativeDlDirect, 0);
        let comparator = receipt(CausalLane::MatchedGpComparator, 0);
        let mut ambiguous = comparator.clone();
        ambiguous.receipt_sha256 = Sha256Digest::of_bytes(b"ambiguous-receipt");
        ambiguous.candidate_sha256 = Sha256Digest::of_bytes(b"different-candidate");
        let ledger = build_causal_evidence_ledger_v2(
            &parents,
            &requested,
            vec![direct, comparator, ambiguous],
            1,
        )
        .unwrap();
        assert_eq!(ledger.tasks[&Task::Binary].complete_pairs, 0);
        assert_eq!(ledger.pair_repairs.len(), 1);
        assert!(ledger.pair_repairs[0].permanent_failure);
        assert_eq!(
            ledger.pair_repairs[0].missing_lane,
            CausalLane::MatchedGpComparator
        );
    }

    #[test]
    fn invalid_proposal_is_not_retried_as_an_evaluator_failure() {
        let parents = BTreeMap::from([(Task::Binary, digest('a'))]);
        let requested = BTreeMap::from([(Task::Binary, 8)]);
        let mut direct = receipt(CausalLane::NativeDlDirect, 0);
        direct.proposal_valid = false;
        let comparator = receipt(CausalLane::MatchedGpComparator, 0);
        let ledger =
            build_causal_evidence_ledger_v2(&parents, &requested, vec![direct, comparator], 1)
                .unwrap();
        assert_eq!(ledger.pair_repairs.len(), 1);
        assert!(ledger.pair_repairs[0].permanent_failure);
    }

    #[test]
    fn transitive_ancestry_becomes_mixed() {
        let provenance = CandidateProvenanceV2::inherited(
            digest('a'),
            vec![digest('b'), digest('c')],
            [DlAncestryKind::AcceptedDl, DlAncestryKind::None],
        );
        assert_eq!(provenance.ancestry, DlAncestryKind::Mixed);
    }

    #[test]
    fn production_canary_panels_are_disjoint() {
        let frontier = (0..36).map(|index| format!("f{index}")).collect::<Vec<_>>();
        let hard = (0..6).map(|index| format!("h{index}")).collect::<Vec<_>>();
        let anchor = (0..6).map(|index| format!("a{index}")).collect::<Vec<_>>();
        let panels = (0..3)
            .map(|run| {
                deterministic_canary_contexts(&digest('a'), run, &frontier, &hard, &anchor)
                    .unwrap()
                    .into_iter()
                    .map(|row| row.context_identity)
                    .collect::<BTreeSet<_>>()
            })
            .collect::<Vec<_>>();
        assert!(panels[0].is_disjoint(&panels[1]));
        assert!(panels[0].is_disjoint(&panels[2]));
        assert!(panels[1].is_disjoint(&panels[2]));
    }
}

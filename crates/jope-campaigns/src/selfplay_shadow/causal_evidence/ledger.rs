use super::*;

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
    let mut tasks = Task::ALL
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

use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    NativeDlDirect,
    MatchedGpComparator,
    ShadowDlDirect,
    Failed,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalEvidenceKey {
    pub task: Task,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    pub wave: u64,
    pub manifest_sha256: Sha256Digest,
    pub candidate_semantic_sha256: Sha256Digest,
    pub receipt_sha256: Sha256Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalEvidenceObservation {
    pub key: CausalEvidenceKey,
    pub context_identity: String,
    pub kind: EvidenceKind,
    pub successfully_ingested: bool,
    pub valid_proposal: bool,
    pub zero_cross_context_backfill: bool,
    pub completed_causal_parent_wave: bool,
    pub valid_original_lineage: bool,
    pub dl_ancestry_kind: DlAncestryKind,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceExclusions {
    pub wave_189: usize,
    pub shadow: usize,
    pub failed: usize,
    pub duplicate: usize,
    pub wrong_parent: usize,
    pub dl_ancestry: usize,
    pub cross_context_backfill: usize,
    pub incomplete_parent_wave: usize,
    pub invalid_original_lineage: usize,
    pub unmatched_partial_manifest: usize,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCausalEvidence {
    pub accepted_parent_checkpoint_sha256: Option<Sha256Digest>,
    pub completed_native_direct: usize,
    pub completed_matched_gp: usize,
    pub valid_native_direct: usize,
    pub native_direct_attempts: usize,
    pub proposal_validity: f64,
    pub zero_cross_context_backfill: bool,
    pub completed_causal_parent_wave: bool,
    pub valid_original_lineage: bool,
    pub eligible_waves: BTreeSet<u64>,
    pub exclusions: EvidenceExclusions,
    pub production_gate_open: bool,
    #[serde(default)]
    pub gate_reasons: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalEvidenceLedgerV1 {
    pub schema: String,
    pub generated_at_ms: u128,
    pub tasks: BTreeMap<Task, TaskCausalEvidence>,
    pub records: Vec<CausalEvidenceObservation>,
}

impl CausalEvidenceLedgerV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CAUSAL_EVIDENCE_LEDGER_SCHEMA,
            "causal evidence ledger schema is not v1"
        );
        for (task, row) in &self.tasks {
            ensure!(
                row.completed_matched_gp <= row.completed_native_direct,
                "{task} matched-GP count exceeds direct evidence"
            );
            ensure!(
                row.proposal_validity.is_finite() && (0.0..=1.0).contains(&row.proposal_validity),
                "{task} proposal validity is outside [0,1]"
            );
            if row.production_gate_open {
                ensure!(
                    row.completed_native_direct >= MIN_PARENT_DIRECT_OUTCOMES
                        && row.completed_matched_gp >= MIN_PARENT_DIRECT_OUTCOMES
                        && row.completed_matched_gp == row.completed_native_direct
                        && row.proposal_validity >= 0.99
                        && row.zero_cross_context_backfill
                        && row.completed_causal_parent_wave
                        && row.valid_original_lineage,
                    "{task} production gate opened without strict causal evidence"
                );
            }
        }
        Ok(())
    }
}

pub fn build_causal_evidence_ledger(
    accepted_parents: &BTreeMap<Task, Sha256Digest>,
    observations: Vec<CausalEvidenceObservation>,
    generated_at_ms: u128,
) -> Result<CausalEvidenceLedgerV1> {
    let mut tasks = [Task::Binary, Task::Regression]
        .into_iter()
        .map(|task| {
            (
                task,
                TaskCausalEvidence {
                    accepted_parent_checkpoint_sha256: accepted_parents.get(&task).cloned(),
                    zero_cross_context_backfill: true,
                    valid_original_lineage: true,
                    ..TaskCausalEvidence::default()
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut unique = BTreeSet::new();
    let mut eligible_direct = Vec::new();
    let mut comparators: BTreeMap<(Task, u64, String), Vec<CausalEvidenceObservation>> =
        BTreeMap::new();

    for observation in observations {
        let row = tasks
            .get_mut(&observation.key.task)
            .expect("both task rows are initialized");
        if observation.kind == EvidenceKind::NativeDlDirect {
            row.native_direct_attempts += 1;
            if observation.valid_proposal {
                row.valid_native_direct += 1;
            }
        }
        if observation.key.wave == 189 {
            row.exclusions.wave_189 += 1;
            continue;
        }
        if observation.kind == EvidenceKind::ShadowDlDirect {
            row.exclusions.shadow += 1;
            continue;
        }
        if observation.kind == EvidenceKind::Failed || !observation.successfully_ingested {
            row.exclusions.failed += 1;
            continue;
        }
        if !unique.insert(observation.key.clone()) {
            row.exclusions.duplicate += 1;
            continue;
        }
        if accepted_parents.get(&observation.key.task)
            != Some(&observation.key.accepted_parent_checkpoint_sha256)
        {
            row.exclusions.wrong_parent += 1;
            continue;
        }
        if observation.dl_ancestry_kind != DlAncestryKind::None {
            row.exclusions.dl_ancestry += 1;
            continue;
        }
        if !observation.zero_cross_context_backfill {
            row.zero_cross_context_backfill = false;
            row.exclusions.cross_context_backfill += 1;
            continue;
        }
        if !observation.completed_causal_parent_wave {
            row.exclusions.incomplete_parent_wave += 1;
            continue;
        }
        if !observation.valid_original_lineage {
            row.valid_original_lineage = false;
            row.exclusions.invalid_original_lineage += 1;
            continue;
        }
        row.completed_causal_parent_wave = true;
        row.eligible_waves.insert(observation.key.wave);
        match observation.kind {
            EvidenceKind::NativeDlDirect => eligible_direct.push(observation),
            EvidenceKind::MatchedGpComparator => comparators
                .entry((
                    observation.key.task,
                    observation.key.wave,
                    observation.context_identity.clone(),
                ))
                .or_default()
                .push(observation),
            EvidenceKind::ShadowDlDirect | EvidenceKind::Failed => unreachable!(),
        }
    }

    for values in comparators.values_mut() {
        values.sort_by(|left, right| left.key.cmp(&right.key));
    }
    eligible_direct.sort_by(|left, right| left.key.cmp(&right.key));
    let mut retained = Vec::new();
    for direct in eligible_direct {
        let row = tasks
            .get_mut(&direct.key.task)
            .expect("both task rows are initialized");
        row.completed_native_direct += 1;
        let group = (
            direct.key.task,
            direct.key.wave,
            direct.context_identity.clone(),
        );
        if let Some(pool) = comparators.get_mut(&group) {
            if let Some(comparator) = pool.pop() {
                row.completed_matched_gp += 1;
                retained.push(comparator);
            } else {
                row.exclusions.unmatched_partial_manifest += 1;
            }
        } else {
            row.exclusions.unmatched_partial_manifest += 1;
        }
        retained.push(direct);
    }
    for row in tasks.values_mut() {
        row.proposal_validity = if row.native_direct_attempts == 0 {
            0.0
        } else {
            row.valid_native_direct as f64 / row.native_direct_attempts as f64
        };
        if row.completed_native_direct < MIN_PARENT_DIRECT_OUTCOMES {
            row.gate_reasons.push(format!(
                "native direct evidence {}/{}",
                row.completed_native_direct, MIN_PARENT_DIRECT_OUTCOMES
            ));
        }
        if row.completed_matched_gp < MIN_PARENT_DIRECT_OUTCOMES {
            row.gate_reasons.push(format!(
                "matched GP evidence {}/{}",
                row.completed_matched_gp, MIN_PARENT_DIRECT_OUTCOMES
            ));
        }
        if row.completed_matched_gp != row.completed_native_direct {
            row.gate_reasons.push(format!(
                "matched GP evidence must equal native direct evidence ({}/{})",
                row.completed_matched_gp, row.completed_native_direct
            ));
        }
        if row.proposal_validity < 0.99 {
            row.gate_reasons.push(format!(
                "proposal validity {:.4} < 0.99",
                row.proposal_validity
            ));
        }
        if !row.zero_cross_context_backfill {
            row.gate_reasons
                .push("cross-context backfill is nonzero".to_owned());
        }
        if !row.completed_causal_parent_wave {
            row.gate_reasons
                .push("no completed causal parent wave".to_owned());
        }
        if !row.valid_original_lineage {
            row.gate_reasons
                .push("original accepted lineage is invalid".to_owned());
        }
        row.production_gate_open = row.gate_reasons.is_empty();
    }
    let ledger = CausalEvidenceLedgerV1 {
        schema: CAUSAL_EVIDENCE_LEDGER_SCHEMA.to_owned(),
        generated_at_ms,
        tasks,
        records: retained,
    };
    ledger.validate()?;
    Ok(ledger)
}

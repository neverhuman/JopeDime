use super::*;

pub(super) fn export_value(verified: &VerifiedProposals, task: Task, dataset: &str) -> Value {
    json!({
        "schema": EXPORT_SCHEMA,
        "task": task.as_str(),
        "dataset_hash": dataset,
        "model_checkpoint_sha256": verified.checkpoint_sha256,
        "checkpoint_artifact": verified.checkpoint_artifact,
        "checkpoint_artifact_sha256": verified.checkpoint_sha256,
        "checkpoint_pointer_sha256": verified.checkpoint_pointer_sha256,
        "proposal_file": verified.proposal_file,
        "proposal_set_sha256": verified.proposal_file_sha256,
        "proposal_pointer_sha256": verified.proposal_pointer_sha256,
        "population_budget_delta": 0,
        "generation_zero_consumption": "replace_existing_seed_slots",
        "proposals": verified.proposals,
    })
}

fn proposal_member(proposal: &GpProposal, checkpoint: &str, operation: &str) -> GpPopulationMember {
    GpPopulationMember {
        canonical_genome: proposal.canonical_genome.clone(),
        semantic_sha256: proposal.semantic_sha256.clone(),
        // Proposal-model priority is not GP fitness on this dataset. The GP must
        // evaluate every injected seed before selection can use its fitness.
        score: None,
        dl_origin: true,
        injected_checkpoint_sha256: Some(checkpoint.to_owned()),
        ancestry: Some(GpAncestry {
            operation: operation.to_owned(),
            parents: vec![proposal.semantic_sha256.clone()],
        }),
    }
}

pub(super) fn merge_generation_zero(
    existing: &[GpPopulationMember],
    proposals: &[GpProposal],
    checkpoint: &str,
    robust: &[RobustSeed],
    robust_slots: usize,
    task: Task,
    dataset: &str,
) -> Result<Value> {
    ensure!(
        proposals.len() == GP_SEED_SLOTS,
        "generation zero requires exactly 64 DL proposals"
    );
    ensure!(
        existing.len() > GP_SEED_SLOTS,
        "population budget must leave at least one robust GP seed after 64 DL proposals"
    );
    ensure!(robust_slots > 0, "robust slot request must be positive");
    let required_robust = robust_slots.min(existing.len() - GP_SEED_SLOTS);
    let mut used = HashSet::new();
    let mut members = Vec::with_capacity(existing.len());
    for proposal in proposals {
        ensure!(
            used.insert(proposal.semantic_sha256.clone()),
            "DL proposal semantics repeat"
        );
        members.push(proposal_member(
            proposal,
            checkpoint,
            "generation_zero_current_checkpoint_dl_seed",
        ));
    }
    let mut selected_robust = Vec::with_capacity(required_robust);
    for row in robust {
        if !used.insert(row.semantic_sha256.clone()) {
            continue;
        }
        members.push(GpPopulationMember {
            canonical_genome: row.canonical_genome.clone(),
            semantic_sha256: row.semantic_sha256.clone(),
            // Cross-dataset evidence selects this seed but is not reusable as
            // fitness for the requested dataset.
            score: None,
            dl_origin: false,
            injected_checkpoint_sha256: None,
            ancestry: Some(GpAncestry {
                operation: "generation_zero_repeated_cross_dataset_gp_winner".to_owned(),
                parents: Vec::new(),
            }),
        });
        selected_robust.push(json!({
            "semantic_sha256": row.semantic_sha256,
            "dataset_count": row.dataset_count,
            "mean_within_dataset_percentile": row.mean_within_dataset_percentile,
            "worst_within_dataset_percentile": row.worst_within_dataset_percentile,
            "strict_win_count": row.strict_win_count,
        }));
        if selected_robust.len() == required_robust {
            break;
        }
    }
    ensure!(
        selected_robust.len() == required_robust,
        "robust seed bank has only {} usable distinct repeated winners; {required_robust} required",
        selected_robust.len()
    );
    for member in existing {
        if members.len() == existing.len() {
            break;
        }
        if used.insert(member.semantic_sha256.clone()) {
            members.push(member.clone());
        }
    }
    ensure!(
        members.len() == existing.len(),
        "deduplication left generation zero below its existing population budget"
    );
    Ok(json!({
        "schema": GENERATION_ZERO_SCHEMA,
        "task": task.as_str(),
        "dataset_hash": dataset,
        "checkpoint_sha256": checkpoint,
        "population_budget": existing.len(),
        "population_budget_delta": 0,
        "composition": {
            "current_checkpoint_dl": GP_SEED_SLOTS,
            "repeated_cross_dataset_gp_winners": required_robust,
            "retained_existing": existing.len() - GP_SEED_SLOTS - required_robust,
        },
        "robust_selection": selected_robust,
        "population": members,
    }))
}

pub(super) fn injection_value(
    existing: &[GpPopulationMember],
    proposals: &[GpProposal],
    checkpoint: &str,
    previous_checkpoint: &str,
    elite_count: usize,
    task: Task,
    dataset: &str,
) -> Result<Value> {
    validate_sha256(previous_checkpoint, "previous checkpoint SHA-256")?;
    ensure!(
        checkpoint != previous_checkpoint,
        "checkpoint did not advance; injection is not permitted"
    );
    ensure!(
        existing.len().saturating_sub(elite_count) >= GP_INJECTION_SLOTS,
        "population has fewer than eight non-elites"
    );
    let elites_before = canonical_json_bytes(&serde_json::to_value(&existing[..elite_count])?)?;
    let (mut members, receipt) =
        inject_promoted_proposals(existing, proposals, checkpoint, elite_count)?;
    ensure!(
        receipt.injected == GP_INJECTION_SLOTS,
        "checkpoint advancement has only {} new proposals; exactly eight required",
        receipt.injected
    );
    ensure!(
        elites_before == canonical_json_bytes(&serde_json::to_value(&members[..elite_count])?)?,
        "checkpoint injection changed an elite"
    );
    let proposal_by_semantic = proposals
        .iter()
        .map(|row| (row.semantic_sha256.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    let mut proposal_priority = Vec::with_capacity(GP_INJECTION_SLOTS);
    for replacement in &receipt.replacements {
        let proposal = proposal_by_semantic
            .get(replacement.injected_semantic_sha256.as_str())
            .context("injection receipt references an unknown proposal")?;
        proposal_priority.push(json!({
            "slot": replacement.slot,
            "semantic_sha256": proposal.semantic_sha256,
            "proposal_score": proposal.proposal_score,
            "observed_score": proposal.score,
        }));
        // The score ranks proposals for replacement but is not fitness on the
        // GP's current dataset. Force evaluation before selection.
        members[replacement.slot].score = None;
    }
    Ok(json!({
        "schema": INJECTION_SCHEMA,
        "task": task.as_str(),
        "dataset_hash": dataset,
        "previous_checkpoint_sha256": previous_checkpoint,
        "checkpoint_sha256": checkpoint,
        "population_budget": existing.len(),
        "population_budget_delta": 0,
        "population": members,
        "injection": receipt,
        "injected_proposal_priority": proposal_priority,
        "injected_fitness_state": "unevaluated",
    }))
}

use super::validation::{
    ensure_prereg_round_provenance, ensure_valid_round_structure, evaluate_round_kl_guard,
    find_unique_parity, find_unique_test_score, validate_and_bind_prereg,
    validate_plan_and_identity,
};
use super::*;

mod evaluate;
use evaluate::ContextEvaluation;

/// Evaluate multi-round self-play outputs against test-split scores, parity records, parent
/// lineage clusters, and pre-registered diagnostic checks without an external pre-registration packet.
pub fn qualify_rounds(inputs: RoundQualInputs<'_>) -> Result<RoundQualReport> {
    qualify_rounds_with_prereg(inputs, None)
}

/// Evaluate multi-round self-play outputs against test-split scores, parity records, parent
/// lineage clusters, and an optional pre-registration packet (`jope.selfplay-round1-prereg/v1`).
pub fn qualify_rounds_with_prereg(
    inputs: RoundQualInputs<'_>,
    prereg_bytes: Option<&[u8]>,
) -> Result<RoundQualReport> {
    let RoundQualInputs {
        plan,
        identity,
        ratchet_outputs,
        test_scores,
        parity,
        parent_of_context,
    } = inputs;

    let identity_sha = validate_plan_and_identity(plan, identity)?;
    ensure!(
        !ratchet_outputs.is_empty(),
        "ratchet_outputs must not be empty"
    );
    ensure!(
        !parent_of_context.is_empty(),
        "parent_of_context must not be empty"
    );
    for (ctx, parent) in parent_of_context {
        ensure!(
            !ctx.trim().is_empty() && !parent.trim().is_empty(),
            "parent_of_context contains empty context or parent cluster id"
        );
    }

    let verified_prereg = match prereg_bytes {
        Some(bytes) => Some(validate_and_bind_prereg(
            bytes,
            plan,
            identity,
            parent_of_context,
        )?),
        None => None,
    };

    let expected_context_set: BTreeSet<String> = if plan.expected_contexts.is_empty() {
        parent_of_context.keys().cloned().collect()
    } else {
        let mut s = BTreeSet::new();
        for ctx in &plan.expected_contexts {
            let trimmed = ctx.trim();
            ensure!(
                !trimmed.is_empty(),
                "plan.expected_contexts contains an empty context id"
            );
            ensure!(
                parent_of_context.contains_key(trimmed),
                "plan.expected_contexts context {trimmed} is missing from parent_of_context"
            );
            ensure!(
                s.insert(trimmed.to_owned()),
                "duplicate context {trimmed} in plan.expected_contexts"
            );
        }
        s
    };

    // Validate all QualTestScore records upfront, enforce 1-to-1 dataset id <-> dataset_hash
    // fingerprint coherence, and reject duplicates or ambiguous aliases.
    let mut seen_dataset_genome = BTreeSet::new();
    let mut seen_hash_prog = BTreeSet::new();
    let mut dataset_by_id: BTreeMap<String, ScoredDataset> = BTreeMap::new();
    let mut dataset_by_hash: BTreeMap<String, ScoredDataset> = BTreeMap::new();

    for ts in test_scores {
        ensure!(
            ts.schema == TEST_SCORE_SCHEMA_ID,
            "unsupported test score schema {:?}; expected {TEST_SCORE_SCHEMA_ID}",
            ts.schema
        );
        ensure!(
            ts.split == "test",
            "test score record for dataset {} has split {:?}; expected \"test\"",
            ts.dataset.id,
            ts.split
        );
        ensure!(
            !ts.dataset.id.trim().is_empty()
                && !ts.dataset.dataset_hash.trim().is_empty()
                && !ts.dataset.train_sha256.trim().is_empty()
                && !ts.dataset.test_sha256.trim().is_empty()
                && ts.dataset.train_rows > 0
                && ts.dataset.test_rows > 0,
            "test score record for dataset {:?} has empty dataset identifiers or zero rows",
            ts.dataset.id
        );
        ensure!(
            !ts.genome.trim().is_empty() && is_hex64(&ts.program_sha256),
            "test score record for dataset {} has empty genome or invalid program_sha256",
            ts.dataset.id
        );
        let expected_prog_sha = sha256_hex(ts.genome.as_bytes());
        ensure!(
            ts.program_sha256 == expected_prog_sha,
            "test score record for dataset {} genome {} has program_sha256={} != sha256(genome)={expected_prog_sha}",
            ts.dataset.id,
            ts.genome,
            ts.program_sha256
        );
        ensure!(
            ts.test_rmse.is_finite()
                && ts.test_rmse > 0.0
                && ts.validation_rmse.is_finite()
                && ts.test_r2.is_finite()
                && ts.seconds.is_finite(),
            "test score record for dataset {} genome {} has non-finite or non-positive metrics",
            ts.dataset.id,
            ts.genome
        );

        if let Some(prev) = dataset_by_id.get(&ts.dataset.id) {
            ensure!(
                prev == &ts.dataset,
                "inconsistent ScoredDataset fingerprint for dataset.id {}",
                ts.dataset.id
            );
        } else {
            dataset_by_id.insert(ts.dataset.id.clone(), ts.dataset.clone());
        }
        if let Some(prev) = dataset_by_hash.get(&ts.dataset.dataset_hash) {
            ensure!(
                prev == &ts.dataset,
                "inconsistent ScoredDataset fingerprint for dataset.dataset_hash {}",
                ts.dataset.dataset_hash
            );
        } else {
            dataset_by_hash.insert(ts.dataset.dataset_hash.clone(), ts.dataset.clone());
        }
        // Prevent cross-dataset alias collision where dataset A's id equals dataset B's dataset_hash.
        if let Some(other) = dataset_by_hash.get(&ts.dataset.id) {
            ensure!(
                other == &ts.dataset,
                "ambiguous dataset identifier collision: dataset.id {} matches another dataset's dataset_hash",
                ts.dataset.id
            );
        }
        if let Some(other) = dataset_by_id.get(&ts.dataset.dataset_hash) {
            ensure!(
                other == &ts.dataset,
                "ambiguous dataset identifier collision: dataset.dataset_hash {} matches another dataset's id",
                ts.dataset.dataset_hash
            );
        }

        ensure!(
            seen_dataset_genome.insert((ts.dataset.id.clone(), ts.genome.clone())),
            "duplicate (dataset, genome) test score record: ({}, {})",
            ts.dataset.id,
            ts.genome
        );
        ensure!(
            seen_hash_prog.insert((ts.dataset.dataset_hash.clone(), ts.program_sha256.clone())),
            "duplicate (dataset_hash, program_sha256) test score record: ({}, {})",
            ts.dataset.dataset_hash,
            ts.program_sha256
        );
    }

    // Validate all ParityRecord entries for non-empty identifiers.
    let ContextEvaluation {
        expected_context_set,
        used_test_score_indices,
        used_parity_indices,
        n_insufficient_rounds_contexts,
        n_missing_test_scores,
        n_registry_mismatch_violations,
        n_earlier_looks_violations,
        n_feeds_selection_violations,
        n_missing_parity,
        n_failed_parity,
        n_unequal_attempt_rounds,
        n_unequal_resource_contexts,
        n_unmeasured_kl_contexts,
        n_kl_frozen_contexts,
        any_ratchet_marked_non_qualified,
        total_policy_optimizer_updates,
        context_rows,
        cluster_log_ratios,
        cluster_log_reductions,
    } = evaluate::evaluate_contexts(
        plan,
        identity,
        ratchet_outputs,
        test_scores,
        parity,
        parent_of_context,
        expected_context_set,
        &verified_prereg,
        &dataset_by_id,
        &dataset_by_hash,
    )?;

    let n_missing_expected_contexts = expected_context_set.len();
    let n_unreferenced_test_scores = test_scores
        .len()
        .saturating_sub(used_test_score_indices.len());
    let n_unreferenced_parity_records = parity.len().saturating_sub(used_parity_indices.len());

    let n_complete_paired_contexts = context_rows
        .iter()
        .filter(|r| r.included_in_paired_interval)
        .count();
    let n_observed_parent_clusters = cluster_log_reductions.len();

    let (paired_log_ratio, paired_log_reduction) =
        if n_complete_paired_contexts >= 2 && n_observed_parent_clusters >= 2 {
            let ratio_groups: Vec<Vec<f64>> = cluster_log_ratios.into_values().collect();
            let reduction_groups: Vec<Vec<f64>> = cluster_log_reductions.into_values().collect();
            let ratio_ci = cluster_bootstrap_ci(
                &ratio_groups,
                plan.bootstrap_resamples as usize,
                plan.bootstrap_seed,
                plan.alpha,
            )?;
            let reduction_ci = cluster_bootstrap_ci(
                &reduction_groups,
                plan.bootstrap_resamples as usize,
                plan.bootstrap_seed,
                plan.alpha,
            )?;
            (Some(ratio_ci), Some(reduction_ci))
        } else {
            (None, None)
        };

    let verdict = paired_log_reduction
        .as_ref()
        .map(|pi| pi.verdict)
        .unwrap_or(Verdict::Inconclusive);
    let margin_cleared = paired_log_reduction
        .as_ref()
        .is_some_and(|pi| pi.cluster_bootstrap.low > plan.practical_margin_log_ratio);

    let sealed_family_boundary_adopted = !GENERALIZATION_CLAIM.starts_with("none:");

    let mut disqualification_reasons = Vec::new();
    if plan.source_grade != SourceGrade::Qualified || any_ratchet_marked_non_qualified {
        disqualification_reasons.push("source_grade_not_qualified".to_owned());
    }
    if n_missing_expected_contexts > 0 {
        disqualification_reasons.push("missing_expected_contexts".to_owned());
    }
    if n_insufficient_rounds_contexts > 0 {
        disqualification_reasons.push("insufficient_rounds".to_owned());
    }
    if n_missing_test_scores > 0 {
        disqualification_reasons.push("missing_test_scores".to_owned());
    }
    if n_registry_mismatch_violations > 0 {
        disqualification_reasons.push("test_score_registry_mismatch".to_owned());
    }
    if n_earlier_looks_violations > 0 {
        disqualification_reasons.push("test_split_earlier_looks_sequence_violation".to_owned());
    }
    if n_feeds_selection_violations > 0 {
        disqualification_reasons.push("test_score_feeds_selection".to_owned());
    }
    if n_missing_parity > 0 {
        disqualification_reasons.push("missing_parity_record".to_owned());
    }
    if n_failed_parity > 0 {
        disqualification_reasons.push("parity_failure".to_owned());
    }
    if n_unreferenced_test_scores > 0 {
        disqualification_reasons.push("unreferenced_test_score_records".to_owned());
    }
    if n_unreferenced_parity_records > 0 {
        disqualification_reasons.push("unreferenced_parity_records".to_owned());
    }
    if n_unequal_attempt_rounds > 0 {
        disqualification_reasons.push("unequal_attempts_in_round".to_owned());
    }
    if n_unequal_resource_contexts > 0 {
        disqualification_reasons.push("unequal_resource_comparison".to_owned());
    }
    if n_unmeasured_kl_contexts > 0 {
        disqualification_reasons.push("kl_guard_unmeasured".to_owned());
    }
    if n_kl_frozen_contexts > 0 && !plan.allow_kl_frozen_rounds {
        disqualification_reasons.push("kl_guard_frozen_round".to_owned());
    }
    if n_observed_parent_clusters < plan.min_parent_clusters {
        disqualification_reasons.push("insufficient_parent_clusters".to_owned());
    }
    if paired_log_reduction.is_none() {
        disqualification_reasons.push("paired_interval_not_computed".to_owned());
    } else if !margin_cleared {
        disqualification_reasons.push("margin_not_cleared".to_owned());
    }

    let caller_asserted_diagnostic_checks_passed = disqualification_reasons.is_empty();
    let provenance_binding_status = match verified_prereg {
        None => {
            disqualification_reasons.push("provenance_and_prereg_binding_unimplemented".to_owned());
            PROVENANCE_BINDING_STATUS.to_owned()
        }
        Some((_, pinned)) => {
            if !pinned {
                disqualification_reasons.push("prereg_unpinned_or_no_execution_grant".to_owned());
            }
            disqualification_reasons.push(PREREG_CALLER_ASSERTED_REASON.to_owned());
            disqualification_reasons
                .push("qualified_execution_mode_disabled_offline_verifier_only".to_owned());
            if pinned {
                PREREG_PINNED_BINDING_STATUS.to_owned()
            } else {
                PREREG_UNPINNED_BINDING_STATUS.to_owned()
            }
        }
    };
    if !sealed_family_boundary_adopted {
        disqualification_reasons.push("no_adopted_sealed_family_boundary".to_owned());
    }

    Ok(RoundQualReport {
        schema: ROUND_QUAL_SCHEMA.to_owned(),
        experiment_identity_sha256: identity_sha,
        prereg_sha256: plan.prereg_sha256.clone(),
        provenance_binding_status,
        multiplicity_adjustment: MULTIPLICITY_ADJUSTMENT_STATUS.to_owned(),
        source_grade: plan.source_grade,
        pinned_test_score_registry: plan.pinned_test_score_registry.clone(),
        endpoint: plan.endpoint,
        alpha: plan.alpha,
        min_parent_clusters: plan.min_parent_clusters,
        practical_margin_log_ratio: plan.practical_margin_log_ratio,
        allow_kl_frozen_rounds: plan.allow_kl_frozen_rounds,
        max_conditional_base_kl: plan.max_conditional_base_kl,
        max_new_support_mass: plan.max_new_support_mass,
        n_ratchet_contexts: ratchet_outputs.len(),
        n_missing_expected_contexts,
        n_complete_paired_contexts,
        n_observed_parent_clusters,
        n_insufficient_rounds_contexts,
        n_missing_test_scores,
        n_registry_mismatch_violations,
        n_earlier_looks_violations,
        n_feeds_selection_violations,
        n_missing_parity,
        n_failed_parity,
        n_unreferenced_test_scores,
        n_unreferenced_parity_records,
        n_unequal_attempt_rounds,
        n_unequal_resource_contexts,
        n_unmeasured_kl_contexts,
        n_kl_frozen_contexts,
        policy_optimizer_updates: total_policy_optimizer_updates,
        paired_log_ratio,
        paired_log_reduction,
        verdict,
        margin_cleared,
        sealed_family_boundary_adopted,
        caller_asserted_diagnostic_checks_passed,
        qualified_claim: false,
        promotion: false,
        l6_claim: false,
        generalization_claim: GENERALIZATION_CLAIM.to_owned(),
        disqualification_reasons,
        contexts: context_rows,
    })
}

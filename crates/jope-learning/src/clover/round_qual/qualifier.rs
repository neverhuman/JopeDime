use super::validation::{
    ensure_prereg_round_provenance, ensure_valid_round_structure, evaluate_round_kl_guard,
    find_unique_parity, find_unique_test_score, validate_and_bind_prereg,
    validate_plan_and_identity,
};
use super::*;

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

    let mut expected_context_set: BTreeSet<String> = if plan.expected_contexts.is_empty() {
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
    let mut seen_parity = BTreeSet::new();
    for pr in parity {
        ensure!(
            !pr.dataset_id.trim().is_empty() && !pr.genome.trim().is_empty(),
            "parity record has empty dataset_id or genome"
        );
        ensure!(
            seen_parity.insert((pr.dataset_id.clone(), pr.genome.clone())),
            "duplicate parity record for ({}, {})",
            pr.dataset_id,
            pr.genome
        );
    }

    let mut seen_contexts = BTreeSet::new();
    let mut used_test_score_indices = BTreeSet::new();
    let mut used_parity_indices = BTreeSet::new();

    let mut n_insufficient_rounds_contexts = 0usize;
    let mut n_missing_test_scores = 0usize;
    let mut n_registry_mismatch_violations = 0usize;
    let mut n_earlier_looks_violations = 0usize;
    let mut n_feeds_selection_violations = 0usize;
    let mut n_missing_parity = 0usize;
    let mut n_failed_parity = 0usize;
    let mut n_unequal_attempt_rounds = 0usize;
    let mut n_unequal_resource_contexts = 0usize;
    let mut n_unmeasured_kl_contexts = 0usize;
    let mut n_kl_frozen_contexts = 0usize;
    let mut any_ratchet_marked_non_qualified = false;
    let mut total_policy_optimizer_updates = 0u64;

    let mut context_rows = Vec::with_capacity(ratchet_outputs.len());
    let mut cluster_log_ratios: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut cluster_log_reductions: BTreeMap<String, Vec<f64>> = BTreeMap::new();

    for raw in ratchet_outputs {
        let schema = raw
            .get("schema")
            .and_then(Value::as_str)
            .unwrap_or_default();
        ensure!(
            schema == RATCHET_SCHEMA,
            "unsupported ratchet output schema {schema:?}; expected {RATCHET_SCHEMA}"
        );
        let context_id = raw
            .get("context_sha256")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .context("ratchet output missing non-empty context_sha256")?
            .to_owned();
        ensure!(
            seen_contexts.insert(context_id.clone()),
            "duplicate ratchet output for context {context_id}"
        );

        let roster_sha = raw
            .get("roster_sha256")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .with_context(|| format!("context {context_id} missing non-empty roster_sha256"))?;
        ensure!(
            roster_sha == identity.data_roster_sha256,
            "context {context_id} roster_sha256 ({roster_sha}) != identity.data_roster_sha256 ({})",
            identity.data_roster_sha256
        );

        let parent_id = parent_of_context
            .get(&context_id)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .with_context(|| format!("context {context_id} has no parent mapping"))?
            .to_owned();
        if !plan.expected_contexts.is_empty() {
            ensure!(
                expected_context_set.contains(&context_id),
                "context {context_id} is not in plan.expected_contexts"
            );
        }
        expected_context_set.remove(&context_id);

        let qual_tag = raw
            .get("qualification")
            .and_then(Value::as_str)
            .unwrap_or("scratch_only");
        if qual_tag != "qualified" {
            any_ratchet_marked_non_qualified = true;
        }

        let updates = raw
            .get("policy_optimizer_updates")
            .and_then(Value::as_u64)
            .with_context(|| {
                format!("context {context_id} missing integer policy_optimizer_updates")
            })?;
        total_policy_optimizer_updates = total_policy_optimizer_updates.saturating_add(updates);

        let equal_resource_comparison = raw
            .get("equal_resource_comparison")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !equal_resource_comparison {
            n_unequal_resource_contexts += 1;
        }

        let rounds = raw
            .get("rounds")
            .and_then(Value::as_array)
            .with_context(|| format!("context {context_id} missing rounds array"))?;

        for (idx, round) in rounds.iter().enumerate() {
            ensure_valid_round_structure(&context_id, idx, round)?;
        }
        if let Some((ref prereg, _)) = verified_prereg {
            ensure_prereg_round_provenance(&context_id, raw, rounds, prereg)?;
        }

        let mut exclusion_reasons = Vec::new();
        if rounds.len() < 2 {
            n_insufficient_rounds_contexts += 1;
            exclusion_reasons.push("insufficient_rounds".to_owned());
        }

        let mut equal_attempts_all_rounds = true;
        let mut any_round_unmeasured_kl = false;
        let mut any_round_kl_frozen = false;
        for round in rounds {
            if round
                .get("equal_attempts_in_round")
                .and_then(Value::as_bool)
                != Some(true)
            {
                equal_attempts_all_rounds = false;
                n_unequal_attempt_rounds += 1;
            }
            let (kl_valid, kl_frozen) = evaluate_round_kl_guard(
                round,
                plan.max_conditional_base_kl,
                plan.max_new_support_mass,
            );
            if !kl_valid {
                any_round_unmeasured_kl = true;
            }
            if kl_frozen {
                any_round_kl_frozen = true;
            }
        }
        if raw.get("kl_guard_evaluated").and_then(Value::as_bool) != Some(true) {
            any_round_unmeasured_kl = true;
        }
        match raw
            .get("any_round_frozen_by_kl_guard")
            .and_then(Value::as_bool)
        {
            Some(top_frozen) => {
                if top_frozen != any_round_kl_frozen {
                    any_round_unmeasured_kl = true;
                }
                if top_frozen {
                    any_round_kl_frozen = true;
                }
            }
            None => {
                any_round_unmeasured_kl = true;
            }
        }
        if any_round_unmeasured_kl {
            n_unmeasured_kl_contexts += 1;
        }
        if any_round_kl_frozen {
            n_kl_frozen_contexts += 1;
        }

        let candidate_genome = rounds
            .last()
            .and_then(|r| r.get("ratchet_champion_genome"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);

        let reference_genome = match plan.endpoint {
            Endpoint::FinalVsRound0Incumbent => rounds
                .first()
                .and_then(|r| r.get("ratchet_champion_genome"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned),
            Endpoint::RatchetVsOrdinaryFinalRound => {
                let Some(last_round) = rounds.last() else {
                    bail!(
                        "context {context_id} has no rounds for Endpoint::RatchetVsOrdinaryFinalRound"
                    );
                };
                let ord = last_round
                    .get("ordinary_champion_genome")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !is_placeholder_sentinel(s))
                    .map(str::to_owned)
                    .with_context(|| {
                        format!(
                            "context {context_id} final round missing non-empty string 'ordinary_champion_genome' required by Endpoint::RatchetVsOrdinaryFinalRound"
                        )
                    })?;
                Some(ord)
            }
        };

        let cand_ts = match candidate_genome.as_deref() {
            Some(g) => {
                find_unique_test_score(test_scores, &context_id, g, &mut used_test_score_indices)?
            }
            None => None,
        };
        let ref_ts = match reference_genome.as_deref() {
            Some(g) => {
                find_unique_test_score(test_scores, &context_id, g, &mut used_test_score_indices)?
            }
            None => None,
        };

        if cand_ts.is_none() || ref_ts.is_none() {
            n_missing_test_scores += 1;
            exclusion_reasons.push("missing_test_score".to_owned());
        }

        if let (Some(c), Some(r)) = (cand_ts, ref_ts) {
            ensure!(
                c.dataset == r.dataset,
                "context {context_id} candidate and reference test scores have mismatched ScoredDataset fingerprints"
            );
        }

        let mut registry_matches_plan = true;
        let mut earlier_looks_clean = true;
        let mut feeds_selection_clean = true;

        for ts in [ref_ts, cand_ts].into_iter().flatten() {
            if ts.registry != plan.pinned_test_score_registry {
                registry_matches_plan = false;
            }
            if ts.feeds_selection {
                feeds_selection_clean = false;
            }
        }

        match (ref_ts, cand_ts) {
            (Some(r), Some(c)) => match (&r.earlier_looks, &c.earlier_looks) {
                (Some(ref_looks), Some(cand_looks)) => {
                    if r.program_sha256 == c.program_sha256 {
                        if !ref_looks.is_empty() || !cand_looks.is_empty() {
                            earlier_looks_clean = false;
                        }
                    } else {
                        let expected_cand_looks = vec![r.program_sha256.clone()];
                        if !registry_matches_plan
                            || !ref_looks.is_empty()
                            || cand_looks != &expected_cand_looks
                        {
                            earlier_looks_clean = false;
                        }
                    }
                }
                _ => {
                    earlier_looks_clean = false;
                }
            },
            (Some(only), None) | (None, Some(only)) => {
                if only.earlier_looks.as_ref().is_none_or(|v| !v.is_empty()) {
                    earlier_looks_clean = false;
                }
            }
            (None, None) => {}
        }

        if !registry_matches_plan {
            n_registry_mismatch_violations += 1;
            exclusion_reasons.push("test_score_registry_mismatch".to_owned());
        }
        if !earlier_looks_clean {
            n_earlier_looks_violations += 1;
            exclusion_reasons.push("earlier_looks_sequence_violation".to_owned());
        }
        if !feeds_selection_clean {
            n_feeds_selection_violations += 1;
            exclusion_reasons.push("test_score_feeds_selection".to_owned());
        }

        let cand_parity = match candidate_genome.as_deref() {
            Some(g) => find_unique_parity(parity, &context_id, g, &mut used_parity_indices)?,
            None => None,
        };
        let ref_parity = match reference_genome.as_deref() {
            Some(g) => find_unique_parity(parity, &context_id, g, &mut used_parity_indices)?,
            None => None,
        };

        let candidate_parity_passed = cand_parity.map(|p| p.passed);
        let reference_parity_passed = ref_parity.map(|p| p.passed);

        if cand_parity.is_none() || ref_parity.is_none() {
            n_missing_parity += 1;
            exclusion_reasons.push("missing_parity_record".to_owned());
        }
        if candidate_parity_passed == Some(false) || reference_parity_passed == Some(false) {
            n_failed_parity += 1;
            exclusion_reasons.push("parity_failure".to_owned());
        }

        let (paired_log_ratio, paired_log_reduction, tie_within_f10_tolerance) =
            match (cand_ts, ref_ts) {
                (Some(c), Some(r)) => {
                    let raw_ratio = c.test_rmse.ln() - r.test_rmse.ln();
                    ensure!(
                        raw_ratio.is_finite(),
                        "context {context_id} produced non-finite log RMSE ratio"
                    );
                    let tie = raw_ratio.abs() <= F10_TIE_TOLERANCE;
                    let ratio = if tie { 0.0 } else { raw_ratio };
                    let reduction = if tie { 0.0 } else { -raw_ratio };
                    (Some(ratio), Some(reduction), Some(tie))
                }
                _ => (None, None, None),
            };

        let included_in_paired_interval = rounds.len() >= 2
            && paired_log_ratio.is_some()
            && paired_log_reduction.is_some()
            && candidate_parity_passed == Some(true)
            && reference_parity_passed == Some(true);

        if included_in_paired_interval {
            if let (Some(log_ratio), Some(log_reduction)) = (paired_log_ratio, paired_log_reduction)
            {
                cluster_log_ratios
                    .entry(parent_id.clone())
                    .or_default()
                    .push(log_ratio);
                cluster_log_reductions
                    .entry(parent_id.clone())
                    .or_default()
                    .push(log_reduction);
            }
        }

        context_rows.push(ContextQualRow {
            context_id,
            parent_cluster_id: parent_id,
            rounds_observed: rounds.len(),
            candidate_genome,
            reference_genome,
            candidate_test_rmse: cand_ts.map(|t| t.test_rmse),
            reference_test_rmse: ref_ts.map(|t| t.test_rmse),
            paired_log_ratio,
            paired_log_reduction,
            tie_within_f10_tolerance,
            candidate_parity_passed,
            reference_parity_passed,
            registry_matches_plan,
            earlier_looks_clean,
            feeds_selection_clean,
            equal_attempts_all_rounds,
            equal_resource_comparison,
            kl_guard_evaluated: !any_round_unmeasured_kl,
            any_round_kl_frozen,
            included_in_paired_interval,
            exclusion_reasons,
        });
    }

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

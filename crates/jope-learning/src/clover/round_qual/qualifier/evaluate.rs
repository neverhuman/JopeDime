use super::*;

pub(crate) struct ContextEvaluation {
    pub(crate) expected_context_set: BTreeSet<String>,
    pub(crate) used_test_score_indices: BTreeSet<usize>,
    pub(crate) used_parity_indices: BTreeSet<usize>,
    pub(crate) n_insufficient_rounds_contexts: usize,
    pub(crate) n_missing_test_scores: usize,
    pub(crate) n_registry_mismatch_violations: usize,
    pub(crate) n_earlier_looks_violations: usize,
    pub(crate) n_feeds_selection_violations: usize,
    pub(crate) n_missing_parity: usize,
    pub(crate) n_failed_parity: usize,
    pub(crate) n_unequal_attempt_rounds: usize,
    pub(crate) n_unequal_resource_contexts: usize,
    pub(crate) n_unmeasured_kl_contexts: usize,
    pub(crate) n_kl_frozen_contexts: usize,
    pub(crate) any_ratchet_marked_non_qualified: bool,
    pub(crate) total_policy_optimizer_updates: u64,
    pub(crate) context_rows: Vec<ContextQualRow>,
    pub(crate) cluster_log_ratios: BTreeMap<String, Vec<f64>>,
    pub(crate) cluster_log_reductions: BTreeMap<String, Vec<f64>>,
}

pub(super) struct ContextEvaluationInput<'a> {
    pub(super) plan: &'a RoundQualPlan,
    pub(super) identity: &'a ExperimentIdentity,
    pub(super) ratchet_outputs: &'a [Value],
    pub(super) test_scores: &'a [QualTestScore],
    pub(super) parity: &'a [ParityRecord],
    pub(super) parent_of_context: &'a BTreeMap<String, String>,
    pub(super) expected_context_set: BTreeSet<String>,
    pub(super) verified_prereg: &'a Option<(SelfPlayPrereg, bool)>,
}

pub(crate) fn evaluate_contexts(input: ContextEvaluationInput<'_>) -> Result<ContextEvaluation> {
    let ContextEvaluationInput {
        plan,
        identity,
        ratchet_outputs,
        test_scores,
        parity,
        parent_of_context,
        mut expected_context_set,
        verified_prereg,
    } = input;
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
        if let Some((prereg, _)) = verified_prereg.as_ref() {
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

    Ok(ContextEvaluation {
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
    })
}

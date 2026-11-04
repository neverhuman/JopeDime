use super::*;

pub(crate) fn ensure_prereg_round_provenance(
    context_id: &str,
    raw_ratchet: &Value,
    rounds: &[Value],
    prereg: &SelfPlayPrereg,
) -> Result<()> {
    let observed_seed_cap = raw_ratchet
        .get("seed_cap_b")
        .and_then(Value::as_u64)
        .with_context(|| {
            format!("context {context_id} missing integer 'seed_cap_b' required by prereg binding")
        })?;
    ensure!(
        observed_seed_cap == prereg.arms_and_rounds.seed_cap_b as u64,
        "context {context_id} seed_cap_b ({observed_seed_cap}) != prereg arms_and_rounds.seed_cap_b ({})",
        prereg.arms_and_rounds.seed_cap_b
    );
    let observed_max_base_kl = raw_ratchet
        .get("max_conditional_base_kl")
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .with_context(|| {
            format!(
                "context {context_id} missing finite 'max_conditional_base_kl' required by prereg binding"
            )
        })?;
    ensure!(
        (observed_max_base_kl - prereg.arms_and_rounds.kl_guard.max_conditional_base_kl).abs()
            <= 1e-12,
        "context {context_id} max_conditional_base_kl ({observed_max_base_kl}) != prereg max_conditional_base_kl ({})",
        prereg.arms_and_rounds.kl_guard.max_conditional_base_kl
    );
    let observed_max_new_mass = raw_ratchet
        .get("max_new_support_mass")
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .with_context(|| {
            format!(
                "context {context_id} missing finite 'max_new_support_mass' required by prereg binding"
            )
        })?;
    ensure!(
        (observed_max_new_mass - prereg.arms_and_rounds.kl_guard.max_new_support_mass).abs()
            <= 1e-12,
        "context {context_id} max_new_support_mass ({observed_max_new_mass}) != prereg max_new_support_mass ({})",
        prereg.arms_and_rounds.kl_guard.max_new_support_mass
    );
    let policy_val = raw_ratchet
        .get("measured_tilt_policy")
        .filter(|v| !v.is_null())
        .with_context(|| {
            format!(
                "context {context_id} missing 'measured_tilt_policy' object required by prereg binding"
            )
        })?;
    let observed_policy: MeasuredTiltPolicyConfig = serde_json::from_value(policy_val.clone())
        .with_context(|| format!("context {context_id} invalid 'measured_tilt_policy'"))?;
    observed_policy
        .validate()
        .with_context(|| format!("context {context_id} invalid 'measured_tilt_policy'"))?;
    let expected_policy = &prereg.arms_and_rounds.measured_tilt_policy;
    ensure!(
        (observed_policy.temperature_tau - expected_policy.temperature_tau).abs() <= 1e-12
            && (observed_policy.advantage_clip - expected_policy.advantage_clip).abs() <= 1e-12
            && (observed_policy.ref_new_support_mass - expected_policy.ref_new_support_mass).abs()
                <= 1e-12
            && (observed_policy.policy_new_support_cap - expected_policy.policy_new_support_cap)
                .abs()
                <= 1e-12
            && (observed_policy.step_size_eta - expected_policy.step_size_eta).abs() <= 1e-12,
        "context {context_id} measured_tilt_policy ({observed_policy:?}) != prereg measured_tilt_policy ({expected_policy:?})"
    );
    ensure!(
        rounds.len() == prereg.arms_and_rounds.rounds,
        "context {context_id} rounds.len() ({}) != prereg arms_and_rounds.rounds ({})",
        rounds.len(),
        prereg.arms_and_rounds.rounds
    );
    for (round_idx, round) in rounds.iter().enumerate() {
        let kl_source = round.get("kl_source").and_then(Value::as_str);
        ensure!(
            kl_source == Some("measured_tilt_leaf_policy"),
            "context {context_id} round {round_idx} kl_source {:?} != \"measured_tilt_leaf_policy\" required by prereg binding",
            kl_source
        );
        for genome_field in ["ratchet_champion_genome", "ordinary_champion_genome"] {
            let valid_str = round
                .get(genome_field)
                .and_then(Value::as_str)
                .is_some_and(|s| !is_placeholder_sentinel(s));
            ensure!(
                valid_str,
                "context {context_id} round {round_idx} missing non-empty string '{genome_field}' required by prereg binding"
            );
        }
        for sha_field in [
            "ordinary_search_sha256",
            "ordinary_observations_sha256",
            "ratchet_search_sha256",
            "ratchet_observations_sha256",
        ] {
            let valid_sha = round
                .get(sha_field)
                .and_then(Value::as_str)
                .is_some_and(is_hex64);
            ensure!(
                valid_sha,
                "context {context_id} round {round_idx} field '{sha_field}' must be a 64-hex lowercase digest"
            );
        }
    }
    Ok(())
}

pub(crate) fn find_unique_test_score<'a>(
    scores: &'a [QualTestScore],
    context_id: &str,
    genome: &str,
    used_indices: &mut BTreeSet<usize>,
) -> Result<Option<&'a QualTestScore>> {
    let mut matches = Vec::new();
    for (idx, ts) in scores.iter().enumerate() {
        if (ts.dataset.id == context_id || ts.dataset.dataset_hash == context_id)
            && ts.genome == genome
        {
            matches.push((idx, ts));
        }
    }
    ensure!(
        matches.len() <= 1,
        "ambiguous test score matches ({}) for context {context_id} genome {genome}",
        matches.len()
    );
    if let Some((idx, ts)) = matches.into_iter().next() {
        used_indices.insert(idx);
        Ok(Some(ts))
    } else {
        Ok(None)
    }
}

pub(crate) fn find_unique_parity<'a>(
    parity: &'a [ParityRecord],
    context_id: &str,
    genome: &str,
    used_indices: &mut BTreeSet<usize>,
) -> Result<Option<&'a ParityRecord>> {
    let mut matches = Vec::new();
    for (idx, pr) in parity.iter().enumerate() {
        if pr.dataset_id == context_id && pr.genome == genome {
            matches.push((idx, pr));
        }
    }
    ensure!(
        matches.len() <= 1,
        "ambiguous parity record matches ({}) for context {context_id} genome {genome}",
        matches.len()
    );
    if let Some((idx, pr)) = matches.into_iter().next() {
        used_indices.insert(idx);
        Ok(Some(pr))
    } else {
        Ok(None)
    }
}

pub(crate) fn ensure_valid_round_structure(
    context_id: &str,
    round_idx: usize,
    round: &Value,
) -> Result<()> {
    let observed_round = round
        .get("round")
        .and_then(Value::as_u64)
        .with_context(|| {
            format!("context {context_id} round[{round_idx}] missing integer 'round'")
        })?;
    ensure!(
        observed_round == round_idx as u64,
        "context {context_id} round[{round_idx}] has out-of-order round={observed_round}; expected {round_idx}"
    );
    let cum_prior = round
        .get("cumulative_prior_rounds")
        .and_then(Value::as_u64)
        .with_context(|| {
            format!(
                "context {context_id} round[{round_idx}] missing integer 'cumulative_prior_rounds'"
            )
        })?;
    ensure!(
        cum_prior == round_idx as u64,
        "context {context_id} round[{round_idx}] has cumulative_prior_rounds={cum_prior}; expected {round_idx}"
    );

    for field in [
        "ordinary_best_rmse",
        "ordinary_best_q",
        "ratchet_best_rmse",
        "ratchet_best_q",
        "surplus_q_vs_ordinary",
    ] {
        let Some(val) = round.get(field) else {
            bail!("context {context_id} round {round_idx} missing numeric field {field}");
        };
        let Some(num) = val.as_f64() else {
            bail!("context {context_id} round {round_idx} field {field} is not a finite number");
        };
        ensure!(
            num.is_finite(),
            "context {context_id} round {round_idx} field {field} is non-finite"
        );
    }
    if let Some(ord_genome) = round.get("ordinary_champion_genome") {
        ensure!(
            ord_genome
                .as_str()
                .is_some_and(|s| !is_placeholder_sentinel(s)),
            "context {context_id} round {round_idx} field ordinary_champion_genome must be a non-empty string"
        );
    }
    Ok(())
}

/// Validate a single round's KL guard fields fail-closed.
/// Returns `(kl_measured_and_valid, frozen_by_kl_guard)`.
pub(crate) fn evaluate_round_kl_guard(
    round: &Value,
    max_conditional_base_kl: f64,
    max_new_support_mass: f64,
) -> (bool, bool) {
    let Some(kl_source) = round.get("kl_source").and_then(Value::as_str) else {
        return (false, false);
    };
    if !matches!(
        kl_source,
        "measured_tilt_leaf_policy" | "caller_evaluated_leaf_log_probs"
    ) {
        let frozen = round
            .get("seed_update_frozen_by_kl_guard")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        return (false, frozen);
    }
    if !round.get("kl_malformed_reason").is_none_or(Value::is_null) {
        return (false, true);
    }
    let Some(decomposed_val) = round.get("decomposed_kl") else {
        return (false, false);
    };
    let Ok(decomposed) = serde_json::from_value::<DecomposedSupportKl>(decomposed_val.clone())
    else {
        return (false, false);
    };
    let numbers_valid = [
        decomposed.conditional_base_kl,
        decomposed.conditional_new_kl,
        decomposed.base_support_mass,
        decomposed.new_support_mass,
        decomposed.ref_new_support_mass,
        decomposed.bernoulli_allocation_kl,
        decomposed.total_kl,
    ]
    .into_iter()
    .all(|x| x.is_finite() && x >= 0.0);
    let masses_valid = [
        decomposed.base_support_mass,
        decomposed.new_support_mass,
        decomposed.ref_new_support_mass,
    ]
    .into_iter()
    .all(|m| m <= 1.0 + 1e-12)
        && (decomposed.base_support_mass + decomposed.new_support_mass - 1.0).abs() <= 1e-9;
    let expected_total_kl = decomposed.bernoulli_allocation_kl
        + decomposed.base_support_mass * decomposed.conditional_base_kl
        + decomposed.new_support_mass * decomposed.conditional_new_kl;
    let chain_rule_valid =
        expected_total_kl.is_finite() && (decomposed.total_kl - expected_total_kl).abs() <= 1e-9;
    if !numbers_valid || !masses_valid || !chain_rule_valid {
        return (false, false);
    }
    let expected_admissible = decomposed.conditional_base_kl <= max_conditional_base_kl + 1e-12
        && decomposed.new_support_mass <= max_new_support_mass + 1e-12;
    if decomposed.admissible != expected_admissible {
        return (false, false);
    }
    let Some(frozen) = round
        .get("seed_update_frozen_by_kl_guard")
        .and_then(Value::as_bool)
    else {
        return (false, false);
    };
    if frozen != !expected_admissible {
        return (false, frozen);
    }
    (true, frozen)
}

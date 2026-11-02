use super::*;

/// Parse a context-to-parent-cluster map from JSON (`{"ctx": "parent"}`) or 2-column TSV
/// (`ctx<TAB>parent`), rejecting duplicate context keys in both formats and rejecting extra TSV
/// columns.

pub fn parse_parent_map(bytes: &[u8]) -> Result<BTreeMap<String, String>> {
    let text = std::str::from_utf8(bytes).context("parent_map: invalid UTF-8")?;
    let trimmed = text.trim();
    ensure!(!trimmed.is_empty(), "parent_map must not be empty");
    if trimmed.starts_with('{') {
        struct NoDupMapVisitor;
        impl<'de> Visitor<'de> for NoDupMapVisitor {
            type Value = BTreeMap<String, String>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(
                    "a JSON object mapping context_id to parent_cluster_id without duplicate keys",
                )
            }

            fn visit_map<M>(self, mut access: M) -> std::result::Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut map = BTreeMap::new();
                while let Some((k, v)) = access.next_entry::<String, String>()? {
                    let key = k.trim().to_owned();
                    let val = v.trim().to_owned();
                    if key.is_empty() || val.is_empty() {
                        return Err(de::Error::custom(
                            "parent_map JSON contains empty context_id or parent_cluster_id",
                        ));
                    }
                    if map.insert(key.clone(), val).is_some() {
                        return Err(de::Error::custom(format!(
                            "duplicate context key in parent_map JSON: {key}"
                        )));
                    }
                }
                Ok(map)
            }
        }
        let mut de = serde_json::Deserializer::from_str(trimmed);
        let map = serde::Deserializer::deserialize_map(&mut de, NoDupMapVisitor)
            .context("parent_map: JSON object")?;
        de.end()
            .context("parent_map: trailing characters after JSON object")?;
        ensure!(!map.is_empty(), "parent_map must not be empty");
        return Ok(map);
    }

    let mut map = BTreeMap::new();
    for (i, raw_line) in trimmed.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        ensure!(
            cols.len() == 2,
            "parent_map line {}: expected exactly 2 tab-separated columns (context_id<TAB>parent_id), got {}",
            i + 1,
            cols.len()
        );
        let ctx = cols[0].trim().to_owned();
        let parent = cols[1].trim().to_owned();
        ensure!(
            !ctx.is_empty() && !parent.is_empty(),
            "parent_map line {}: empty context_id or parent_id",
            i + 1
        );
        ensure!(
            map.insert(ctx.clone(), parent).is_none(),
            "parent_map line {}: duplicate context key {ctx}",
            i + 1
        );
    }
    ensure!(!map.is_empty(), "parent_map must not be empty");
    Ok(map)
}

pub(super) fn validate_plan_and_identity(
    plan: &RoundQualPlan,
    identity: &ExperimentIdentity,
) -> Result<String> {
    let computed_id = identity
        .sha256()
        .context("invalid ExperimentIdentity for round qualification")?;
    ensure!(
        is_hex64(&plan.experiment_identity_sha256),
        "plan.experiment_identity_sha256 must be a 64-hex lowercase digest"
    );
    ensure!(
        computed_id == plan.experiment_identity_sha256,
        "experiment identity sha256 mismatch: identity={computed_id}, plan={}",
        plan.experiment_identity_sha256
    );
    ensure!(
        is_hex64(&plan.prereg_sha256),
        "plan.prereg_sha256 must be a 64-hex lowercase digest"
    );
    ensure!(
        !plan.pinned_test_score_registry.trim().is_empty(),
        "plan.pinned_test_score_registry must be non-empty"
    );
    ensure!(
        plan.min_parent_clusters >= 2,
        "plan.min_parent_clusters must be at least 2 for parent-cluster bootstrap inference"
    );
    ensure!(
        plan.practical_margin_log_ratio.is_finite() && plan.practical_margin_log_ratio >= 0.0,
        "plan.practical_margin_log_ratio must be finite and >= 0"
    );
    ensure!(
        plan.bootstrap_resamples >= MIN_RESAMPLES,
        "plan.bootstrap_resamples ({}) must be >= {MIN_RESAMPLES}",
        plan.bootstrap_resamples
    );
    ensure!(
        plan.alpha.is_finite() && plan.alpha > 0.0 && plan.alpha <= MAX_SINGLE_ENDPOINT_ALPHA,
        "plan.alpha ({}) must be finite and in (0, {MAX_SINGLE_ENDPOINT_ALPHA}] while joint Holm-Bonferroni adjustment is unimplemented",
        plan.alpha
    );
    ensure!(
        plan.max_conditional_base_kl.is_finite() && plan.max_conditional_base_kl >= 0.0,
        "plan.max_conditional_base_kl ({}) must be finite and >= 0",
        plan.max_conditional_base_kl
    );
    ensure!(
        plan.max_new_support_mass.is_finite() && (0.0..=1.0).contains(&plan.max_new_support_mass),
        "plan.max_new_support_mass ({}) must be finite and in [0, 1]",
        plan.max_new_support_mass
    );
    Ok(computed_id)
}

pub(super) fn validate_and_bind_prereg(
    prereg_bytes: &[u8],
    plan: &RoundQualPlan,
    identity: &ExperimentIdentity,
    parent_of_context: &BTreeMap<String, String>,
) -> Result<(SelfPlayPrereg, bool)> {
    let raw_sha = sha256_hex(prereg_bytes);
    let prereg_val = parse_strict_json_value(prereg_bytes, "prereg")?;
    let canon_sha = canonical_sha256(&prereg_val).context("prereg: canonical sha256")?;
    ensure!(
        plan.prereg_sha256 == raw_sha || plan.prereg_sha256 == canon_sha,
        "prereg_sha256 mismatch: plan.prereg_sha256={} != sha256(prereg_bytes)={raw_sha} (canonical={canon_sha})",
        plan.prereg_sha256
    );

    let prereg: SelfPlayPrereg =
        serde_json::from_value(prereg_val).context("prereg: schema validation")?;
    ensure!(
        prereg.schema == PREREG_SCHEMA,
        "unsupported prereg schema {:?}; expected {PREREG_SCHEMA}",
        prereg.schema
    );
    ensure!(
        !is_placeholder_sentinel(&prereg.study_id) && !is_placeholder_sentinel(&prereg.status),
        "prereg study_id and status must be non-empty and not placeholder sentinels"
    );
    ensure!(
        prereg.qualification_report_schema == ROUND_QUAL_SCHEMA,
        "prereg qualification_report_schema {:?} != {ROUND_QUAL_SCHEMA}",
        prereg.qualification_report_schema
    );
    ensure!(
        prereg.source_commit_at_draft.len() == 40
            && prereg
                .source_commit_at_draft
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "prereg source_commit_at_draft must be a 40-character lowercase hex commit sha"
    );

    // 1. Endpoints and multiplicity invariants
    ensure!(
        prereg.endpoints.primary == Endpoint::FinalVsRound0Incumbent
            && prereg.endpoints.secondary == Endpoint::RatchetVsOrdinaryFinalRound,
        "prereg endpoints must bind primary=final_vs_round0_incumbent and secondary=ratchet_vs_ordinary_final_round"
    );
    ensure!(
        prereg.endpoints.secondary_status == PREREG_SECONDARY_STATUS,
        "prereg secondary_status {:?} != {PREREG_SECONDARY_STATUS}",
        prereg.endpoints.secondary_status
    );
    ensure!(
        prereg.endpoints.cli_single_endpoint_alpha_cap.is_finite()
            && (prereg.endpoints.cli_single_endpoint_alpha_cap - MAX_SINGLE_ENDPOINT_ALPHA).abs()
                <= 1e-15,
        "prereg cli_single_endpoint_alpha_cap ({}) != {MAX_SINGLE_ENDPOINT_ALPHA}",
        prereg.endpoints.cli_single_endpoint_alpha_cap
    );
    ensure!(
        prereg.endpoints.alpha_primary_strict.is_finite()
            && (prereg.endpoints.alpha_primary_strict - MAX_SINGLE_ENDPOINT_ALPHA).abs() <= 1e-15,
        "prereg alpha_primary_strict ({}) != {MAX_SINGLE_ENDPOINT_ALPHA}",
        prereg.endpoints.alpha_primary_strict
    );
    ensure!(
        prereg.endpoints.f10_tie_tolerance_log_ratio.is_finite()
            && (prereg.endpoints.f10_tie_tolerance_log_ratio - F10_TIE_TOLERANCE).abs() <= 1e-18,
        "prereg f10_tie_tolerance_log_ratio ({}) != {F10_TIE_TOLERANCE}",
        prereg.endpoints.f10_tie_tolerance_log_ratio
    );
    ensure!(
        plan.alpha <= prereg.endpoints.cli_single_endpoint_alpha_cap + 1e-15,
        "plan.alpha ({}) exceeds prereg cli_single_endpoint_alpha_cap ({})",
        plan.alpha,
        prereg.endpoints.cli_single_endpoint_alpha_cap
    );

    // 2. Roster and parent-cluster invariants
    ensure!(
        prereg.roster_and_families.unit_of_inference == "parent_dataset_cluster",
        "prereg unit_of_inference must be \"parent_dataset_cluster\""
    );
    ensure!(
        prereg.roster_and_families.roster_role == "confirmation_family_held_out",
        "prereg roster_role must be \"confirmation_family_held_out\""
    );
    if let Some(roster_sha) = prereg.roster_and_families.sealed_roster_sha256.as_pinned() {
        ensure!(
            is_hex64(roster_sha) && roster_sha == &identity.data_roster_sha256,
            "prereg sealed_roster_sha256 ({roster_sha}) != identity.data_roster_sha256 ({})",
            identity.data_roster_sha256
        );
    }
    if let Some(parent_sha) = prereg
        .roster_and_families
        .parent_cluster_map_sha256
        .as_pinned()
    {
        let expected_parent_sha = canonical_sha256(parent_of_context)?;
        ensure!(
            is_hex64(parent_sha) && parent_sha == &expected_parent_sha,
            "prereg parent_cluster_map_sha256 ({parent_sha}) != canonical_sha256(parent_of_context) ({expected_parent_sha})"
        );
    }
    if let Some(ruling) = prereg
        .roster_and_families
        .g23_sealed_boundary_ruling
        .as_pinned()
    {
        ensure!(
            !is_placeholder_sentinel(ruling),
            "prereg g23_sealed_boundary_ruling must be non-empty and not a placeholder when pinned"
        );
    }

    // 3. Sample size, margin, and bootstrap invariants
    ensure!(
        prereg.sample_size_and_margin.bootstrap_resamples == plan.bootstrap_resamples,
        "prereg bootstrap_resamples ({}) != plan.bootstrap_resamples ({})",
        prereg.sample_size_and_margin.bootstrap_resamples,
        plan.bootstrap_resamples
    );
    ensure!(
        prereg.sample_size_and_margin.bootstrap_seed == plan.bootstrap_seed,
        "prereg bootstrap_seed ({}) != plan.bootstrap_seed ({})",
        prereg.sample_size_and_margin.bootstrap_seed,
        plan.bootstrap_seed
    );
    ensure!(
        prereg.sample_size_and_margin.bootstrap_method == "parent_cluster_percentile_splitmix64",
        "prereg bootstrap_method must be \"parent_cluster_percentile_splitmix64\""
    );
    if let Some(&min_clusters) = prereg
        .sample_size_and_margin
        .min_parent_clusters
        .as_pinned()
    {
        ensure!(
            min_clusters >= 2 && min_clusters == plan.min_parent_clusters,
            "prereg min_parent_clusters ({min_clusters}) != plan.min_parent_clusters ({})",
            plan.min_parent_clusters
        );
    }
    if let Some(&margin) = prereg
        .sample_size_and_margin
        .practical_margin_log_ratio
        .as_pinned()
    {
        ensure!(
            margin.is_finite()
                && margin >= 0.0
                && (margin - plan.practical_margin_log_ratio).abs() <= 1e-12,
            "prereg practical_margin_log_ratio ({margin}) != plan.practical_margin_log_ratio ({})",
            plan.practical_margin_log_ratio
        );
    }
    if let Some(c158) = prereg
        .sample_size_and_margin
        .c158_g_judge_binding_receipt
        .as_pinned()
    {
        ensure!(
            !is_placeholder_sentinel(c158),
            "prereg c158_g_judge_binding_receipt must be non-empty and not a placeholder when pinned"
        );
    }

    // 4. Arms, rounds, measured-tilt policy, and KL guard invariants
    ensure!(
        prereg.arms_and_rounds.rounds >= 2,
        "prereg arms_and_rounds.rounds ({}) must be >= 2",
        prereg.arms_and_rounds.rounds
    );
    ensure!(
        prereg.arms_and_rounds.seed_cap_b >= 1,
        "prereg arms_and_rounds.seed_cap_b ({}) must be >= 1",
        prereg.arms_and_rounds.seed_cap_b
    );
    ensure!(
        prereg.arms_and_rounds.equal_attempts_per_round_required
            && prereg.arms_and_rounds.equal_resource_comparison_required,
        "prereg equal_attempts_per_round_required and equal_resource_comparison_required must be true"
    );
    prereg
        .arms_and_rounds
        .measured_tilt_policy
        .validate()
        .context("prereg measured_tilt_policy invalid")?;
    ensure!(
        (prereg.arms_and_rounds.kl_guard.max_conditional_base_kl - plan.max_conditional_base_kl)
            .abs()
            <= 1e-12,
        "prereg max_conditional_base_kl ({}) != plan.max_conditional_base_kl ({})",
        prereg.arms_and_rounds.kl_guard.max_conditional_base_kl,
        plan.max_conditional_base_kl
    );
    ensure!(
        (prereg.arms_and_rounds.kl_guard.max_new_support_mass - plan.max_new_support_mass).abs()
            <= 1e-12,
        "prereg max_new_support_mass ({}) != plan.max_new_support_mass ({})",
        prereg.arms_and_rounds.kl_guard.max_new_support_mass,
        plan.max_new_support_mass
    );
    ensure!(
        prereg.arms_and_rounds.kl_guard.allow_kl_frozen_rounds == plan.allow_kl_frozen_rounds,
        "prereg allow_kl_frozen_rounds ({}) != plan.allow_kl_frozen_rounds ({})",
        prereg.arms_and_rounds.kl_guard.allow_kl_frozen_rounds,
        plan.allow_kl_frozen_rounds
    );
    ensure!(
        prereg.arms_and_rounds.kl_guard.allowlisted_measured_sources
            == vec!["measured_tilt_leaf_policy".to_owned()],
        "prereg allowlisted_measured_sources must match [\"measured_tilt_leaf_policy\"]"
    );

    // 5. Held-out test-score and parity rules
    ensure!(
        prereg.held_out_and_parity_rules.test_score_schema == TEST_SCORE_SCHEMA_ID,
        "prereg test_score_schema {:?} != {TEST_SCORE_SCHEMA_ID}",
        prereg.held_out_and_parity_rules.test_score_schema
    );
    if let Some(reg) = prereg
        .held_out_and_parity_rules
        .pinned_test_score_registry
        .as_pinned()
    {
        ensure!(
            !is_placeholder_sentinel(reg) && reg == &plan.pinned_test_score_registry,
            "prereg pinned_test_score_registry ({reg}) != plan.pinned_test_score_registry ({})",
            plan.pinned_test_score_registry
        );
    }
    let looks = &prereg
        .held_out_and_parity_rules
        .sequential_same_registry_looks;
    ensure!(
        looks.reference_earlier_looks.is_empty()
            && looks.candidate_earlier_looks_when_distinct
                == vec!["<reference.program_sha256>".to_owned()]
            && looks.reject_omitted_earlier_looks_field
            && looks.reject_registry_reset_both_empty_when_distinct,
        "prereg sequential_same_registry_looks contract does not match B1 verifier rules"
    );
    ensure!(
        prereg
            .held_out_and_parity_rules
            .require_feeds_selection_false
            && prereg
                .held_out_and_parity_rules
                .r5_parity_failure_disposition
                == "charge_as_program_failure_never_drop"
            && prereg
                .held_out_and_parity_rules
                .reject_unreferenced_score_or_parity_records,
        "prereg held_out_and_parity_rules contract does not match R5/M5 verifier rules"
    );

    // 6. Experiment identity & execution grant admission gate
    if let Some(id_sha) = prereg.experiment_identity_sha256.as_pinned() {
        ensure!(
            is_hex64(id_sha) && id_sha == &plan.experiment_identity_sha256,
            "prereg experiment_identity_sha256 ({id_sha}) != plan.experiment_identity_sha256 ({})",
            plan.experiment_identity_sha256
        );
    }
    if let Some(loc) = prereg.execution_grant.owner_admission_locator.as_pinned() {
        ensure!(
            !is_placeholder_sentinel(loc),
            "prereg owner_admission_locator must be non-empty and not a placeholder when pinned"
        );
    }
    if let Some(hosts) = prereg.execution_grant.hosts.as_pinned() {
        ensure!(
            !hosts.is_empty() && hosts.iter().all(|h| !is_placeholder_sentinel(h)),
            "prereg execution_grant.hosts must be non-empty and contain no placeholders when pinned"
        );
    }
    if let Some(&cpu_cap) = prereg.execution_grant.cpu_ceiling_percent.as_pinned() {
        ensure!(
            (1..=100).contains(&cpu_cap),
            "prereg execution_grant.cpu_ceiling_percent ({cpu_cap}) must be in 1..=100"
        );
    }
    ensure!(
        (1..=100).contains(&prereg.execution_grant.memory_floor_free_percent),
        "prereg execution_grant.memory_floor_free_percent ({}) must be in 1..=100",
        prereg.execution_grant.memory_floor_free_percent
    );
    if let Some(deadline) = prereg.execution_grant.deadline_utc.as_pinned() {
        ensure!(
            !is_placeholder_sentinel(deadline),
            "prereg execution_grant.deadline_utc must be non-empty and not a placeholder when pinned"
        );
    }

    ensure!(
        !is_placeholder_sentinel(&prereg.study_id),
        "prereg study_id must be non-empty and not a placeholder"
    );
    let has_unpinned = prereg.has_unpinned_fields();
    if prereg.execution_grant.authorized {
        ensure!(
            !has_unpinned,
            "prereg execution_grant.authorized is true while one or more required prereg fields remain UNPINNED"
        );
        let status_upper = prereg.status.to_ascii_uppercase();
        ensure!(
            !is_placeholder_sentinel(&prereg.status)
                && !status_upper.contains("UNPINNED")
                && !status_upper.contains("NO_EXECUTION_GRANT"),
            "prereg execution_grant.authorized is true while status {:?} indicates unpinned or no execution grant",
            prereg.status
        );
        ensure!(
            prereg.provenance_binding_status == PREREG_PINNED_BINDING_STATUS,
            "prereg provenance_binding_status {:?} != {PREREG_PINNED_BINDING_STATUS} when authorized",
            prereg.provenance_binding_status
        );
    } else {
        ensure!(
            prereg.status == "PRE_REGISTERED_UNPINNED_NO_EXECUTION_GRANT",
            "prereg status {:?} must be PRE_REGISTERED_UNPINNED_NO_EXECUTION_GRANT when execution_grant.authorized is false",
            prereg.status
        );
        ensure!(
            prereg.provenance_binding_status == PREREG_UNPINNED_BINDING_STATUS,
            "prereg provenance_binding_status {:?} != {PREREG_UNPINNED_BINDING_STATUS} when execution_grant.authorized is false",
            prereg.provenance_binding_status
        );
    }
    let fully_pinned_and_authorized = !has_unpinned && prereg.execution_grant.authorized;
    Ok((prereg, fully_pinned_and_authorized))
}

pub(super) fn ensure_prereg_round_provenance(
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

pub(super) fn find_unique_test_score<'a>(
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

pub(super) fn find_unique_parity<'a>(
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

pub(super) fn ensure_valid_round_structure(
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
pub(super) fn evaluate_round_kl_guard(
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

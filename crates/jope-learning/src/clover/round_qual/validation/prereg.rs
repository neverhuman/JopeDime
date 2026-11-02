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

pub(crate) fn validate_plan_and_identity(
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

pub(crate) fn validate_and_bind_prereg(
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

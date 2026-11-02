use super::*;

#[test]
fn rejects_invalid_prereg_mutations() {
    let fixture = prereg_fixture();
    let PreregFixture {
        identity,
        plan,
        parsed_committed,
        parent_map,
        ratchets,
        scores,
        parity,
        pinned_bytes,
        pinned_plan,
        ..
    } = fixture;
    let assert_prereg_rejected = |label: &str, mutated: &SelfPlayPrereg| {
        let bytes = serde_json::to_vec(mutated).unwrap();
        let mut p = pinned_plan.clone();
        p.prereg_sha256 = sha256_hex(&bytes);
        let res = qualify_rounds_with_prereg(
            RoundQualInputs {
                plan: &p,
                identity: &identity,
                ratchet_outputs: &ratchets,
                test_scores: &scores,
                parity: &parity,
                parent_of_context: &parent_map,
            },
            Some(&bytes),
        );
        assert!(
            res.is_err(),
            "expected prereg mutation `{label}` to be rejected, got Ok"
        );
    };

    type PreregMutator = (&'static str, Box<dyn Fn(&mut SelfPlayPrereg)>);
    let prereg_mutations: Vec<PreregMutator> = vec![
        ("schema", Box::new(|p| p.schema = "wrong".into())),
        (
            "qualification_report_schema",
            Box::new(|p| p.qualification_report_schema = "wrong".into()),
        ),
        (
            "source_commit_at_draft_len",
            Box::new(|p| p.source_commit_at_draft = "not-40-hex".into()),
        ),
        (
            "endpoints.primary",
            Box::new(|p| p.endpoints.primary = Endpoint::RatchetVsOrdinaryFinalRound),
        ),
        (
            "endpoints.secondary",
            Box::new(|p| p.endpoints.secondary = Endpoint::FinalVsRound0Incumbent),
        ),
        (
            "endpoints.secondary_status",
            Box::new(|p| p.endpoints.secondary_status = "WRONG_SECONDARY_STATUS".into()),
        ),
        (
            "endpoints.alpha_primary_strict",
            Box::new(|p| p.endpoints.alpha_primary_strict = 0.05),
        ),
        (
            "endpoints.cli_single_endpoint_alpha_cap",
            Box::new(|p| p.endpoints.cli_single_endpoint_alpha_cap = 0.01),
        ),
        (
            "endpoints.f10_tie_tolerance_log_ratio",
            Box::new(|p| p.endpoints.f10_tie_tolerance_log_ratio = 1e-6),
        ),
        (
            "roster.unit_of_inference",
            Box::new(|p| p.roster_and_families.unit_of_inference = "dataset".into()),
        ),
        (
            "roster.roster_role",
            Box::new(|p| p.roster_and_families.roster_role = "test".into()),
        ),
        (
            "roster.sealed_roster_sha256",
            Box::new(|p| {
                p.roster_and_families.sealed_roster_sha256 = UnpinnedOr::Pinned("0".repeat(64));
            }),
        ),
        (
            "roster.parent_cluster_map_sha256",
            Box::new(|p| {
                p.roster_and_families.parent_cluster_map_sha256 =
                    UnpinnedOr::Pinned("0".repeat(64));
            }),
        ),
        (
            "roster.g23_empty",
            Box::new(|p| {
                p.roster_and_families.g23_sealed_boundary_ruling = UnpinnedOr::Pinned("   ".into());
            }),
        ),
        (
            "roster.g23_sentinel_unpinned_ws",
            Box::new(|p| {
                p.roster_and_families.g23_sealed_boundary_ruling =
                    UnpinnedOr::Pinned(" unpinned ".into());
            }),
        ),
        (
            "roster.g23_sentinel_tbd",
            Box::new(|p| {
                p.roster_and_families.g23_sealed_boundary_ruling = UnpinnedOr::Pinned("TBD".into());
            }),
        ),
        (
            "sample.bootstrap_resamples",
            Box::new(|p| p.sample_size_and_margin.bootstrap_resamples = 999),
        ),
        (
            "sample.bootstrap_seed",
            Box::new(|p| p.sample_size_and_margin.bootstrap_seed = 42),
        ),
        (
            "sample.bootstrap_method",
            Box::new(|p| p.sample_size_and_margin.bootstrap_method = "percentile".into()),
        ),
        (
            "sample.min_parent_clusters",
            Box::new(|p| p.sample_size_and_margin.min_parent_clusters = UnpinnedOr::Pinned(99)),
        ),
        (
            "sample.practical_margin_log_ratio",
            Box::new(|p| {
                p.sample_size_and_margin.practical_margin_log_ratio = UnpinnedOr::Pinned(0.99);
            }),
        ),
        (
            "sample.c158_sentinel_todo",
            Box::new(|p| {
                p.sample_size_and_margin.c158_g_judge_binding_receipt =
                    UnpinnedOr::Pinned(" todo ".into());
            }),
        ),
        (
            "schedule.rounds",
            Box::new(|p| p.arms_and_rounds.rounds = 2),
        ),
        (
            "schedule.seed_cap_b",
            Box::new(|p| p.arms_and_rounds.seed_cap_b = 4),
        ),
        (
            "schedule.equal_attempts",
            Box::new(|p| p.arms_and_rounds.equal_attempts_per_round_required = false),
        ),
        (
            "schedule.equal_resource",
            Box::new(|p| p.arms_and_rounds.equal_resource_comparison_required = false),
        ),
        (
            "schedule.tilt.temperature_tau",
            Box::new(|p| p.arms_and_rounds.measured_tilt_policy.temperature_tau = 0.99),
        ),
        (
            "schedule.tilt.advantage_clip",
            Box::new(|p| p.arms_and_rounds.measured_tilt_policy.advantage_clip = 9.0),
        ),
        (
            "schedule.tilt.ref_new_support_mass",
            Box::new(|p| p.arms_and_rounds.measured_tilt_policy.ref_new_support_mass = 0.9),
        ),
        (
            "schedule.tilt.policy_new_support_cap",
            Box::new(|p| {
                p.arms_and_rounds
                    .measured_tilt_policy
                    .policy_new_support_cap = 0.9
            }),
        ),
        (
            "schedule.tilt.step_size_eta",
            Box::new(|p| p.arms_and_rounds.measured_tilt_policy.step_size_eta = 0.9),
        ),
        (
            "kl.max_conditional_base_kl",
            Box::new(|p| p.arms_and_rounds.kl_guard.max_conditional_base_kl = 0.50),
        ),
        (
            "kl.max_new_support_mass",
            Box::new(|p| p.arms_and_rounds.kl_guard.max_new_support_mass = 0.50),
        ),
        (
            "kl.allow_kl_frozen_rounds",
            Box::new(|p| p.arms_and_rounds.kl_guard.allow_kl_frozen_rounds = true),
        ),
        (
            "kl.allowlisted_both_sources_rejected_in_prereg",
            Box::new(|p| {
                p.arms_and_rounds.kl_guard.allowlisted_measured_sources = vec![
                    "measured_tilt_leaf_policy".into(),
                    "caller_evaluated_leaf_log_probs".into(),
                ];
            }),
        ),
        (
            "kl.allowlisted_caller_source_only_rejected_in_prereg",
            Box::new(|p| {
                p.arms_and_rounds.kl_guard.allowlisted_measured_sources =
                    vec!["caller_evaluated_leaf_log_probs".into()];
            }),
        ),
        (
            "held_out.test_score_schema",
            Box::new(|p| p.held_out_and_parity_rules.test_score_schema = "wrong".into()),
        ),
        (
            "held_out.pinned_test_score_registry",
            Box::new(|p| {
                p.held_out_and_parity_rules.pinned_test_score_registry =
                    UnpinnedOr::Pinned("wrong".into());
            }),
        ),
        (
            "held_out.sequential_same_registry_looks.reject_omitted",
            Box::new(|p| {
                p.held_out_and_parity_rules
                    .sequential_same_registry_looks
                    .reject_omitted_earlier_looks_field = false;
            }),
        ),
        (
            "held_out.require_feeds_selection_false",
            Box::new(|p| p.held_out_and_parity_rules.require_feeds_selection_false = false),
        ),
        (
            "held_out.r5_parity_failure_disposition",
            Box::new(|p| {
                p.held_out_and_parity_rules.r5_parity_failure_disposition = "ignore".into()
            }),
        ),
        (
            "held_out.reject_unreferenced_score_or_parity_records",
            Box::new(|p| {
                p.held_out_and_parity_rules
                    .reject_unreferenced_score_or_parity_records = false;
            }),
        ),
        (
            "experiment_identity_sha256",
            Box::new(|p| p.experiment_identity_sha256 = UnpinnedOr::Pinned("0".repeat(64))),
        ),
        (
            "grant.authorized_true_with_unpinned_field",
            Box::new(|p| p.roster_and_families.sealed_roster_sha256 = UnpinnedOr::Unpinned),
        ),
        (
            "grant.owner_admission_locator_sentinel_pending",
            Box::new(|p| {
                p.execution_grant.owner_admission_locator = UnpinnedOr::Pinned(" PENDING ".into());
            }),
        ),
        (
            "grant.hosts_empty",
            Box::new(|p| p.execution_grant.hosts = UnpinnedOr::Pinned(vec![])),
        ),
        (
            "grant.hosts_sentinel_unpinned",
            Box::new(|p| p.execution_grant.hosts = UnpinnedOr::Pinned(vec!["UNPINNED".into()])),
        ),
        (
            "grant.hosts_sentinel_tbd",
            Box::new(|p| p.execution_grant.hosts = UnpinnedOr::Pinned(vec![" tbd ".into()])),
        ),
        (
            "grant.cpu_ceiling_0",
            Box::new(|p| p.execution_grant.cpu_ceiling_percent = UnpinnedOr::Pinned(0)),
        ),
        (
            "grant.cpu_ceiling_101",
            Box::new(|p| p.execution_grant.cpu_ceiling_percent = UnpinnedOr::Pinned(101)),
        ),
        (
            "grant.memory_floor_0",
            Box::new(|p| p.execution_grant.memory_floor_free_percent = 0),
        ),
        (
            "grant.memory_floor_101",
            Box::new(|p| p.execution_grant.memory_floor_free_percent = 101),
        ),
        (
            "grant.deadline_sentinel_unpinned",
            Box::new(|p| {
                p.execution_grant.deadline_utc = UnpinnedOr::Pinned("unpinned".into());
            }),
        ),
        (
            "pinned.status_contains_unpinned",
            Box::new(|p| p.status = "PRE_REGISTERED_UNPINNED_CONTRADICTION".into()),
        ),
        (
            "pinned.provenance_binding_status_wrong",
            Box::new(|p| p.provenance_binding_status = PREREG_UNPINNED_BINDING_STATUS.into()),
        ),
    ];

    for (label, mutate) in &prereg_mutations {
        let mut candidate = pinned_prereg.clone();
        mutate(&mut candidate);
        assert_prereg_rejected(label, &candidate);
    }

    // Also test unpinned consistency violations (authorized=false with wrong provenance_binding_status or status):
    let mut bad_unpinned_binding = parsed_committed.clone();
    bad_unpinned_binding.provenance_binding_status = PREREG_PINNED_BINDING_STATUS.to_owned();
    assert_prereg_rejected("unpinned.wrong_binding_status", &bad_unpinned_binding);

    let mut bad_unpinned_status = parsed_committed.clone();
    bad_unpinned_status.status = "PRE_REGISTERED_PINNED_NO_UNPINNED_TOKEN".to_owned();
    assert_prereg_rejected("unpinned.status_missing_unpinned", &bad_unpinned_status);

    // Digest mismatch against plan.prereg_sha256:
    assert!(qualify_rounds_with_prereg(
        RoundQualInputs {
            plan: &plan,
            identity: &identity,
            ratchet_outputs: &ratchets,
            test_scores: &scores,
            parity: &parity,
            parent_of_context: &parent_map,
        },
        Some(&pinned_bytes),
    )
    .is_err());
}

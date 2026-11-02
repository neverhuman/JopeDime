use super::*;

pub(crate) struct PreregFixture {
    pub(crate) identity: ExperimentIdentity,
    pub(crate) plan: RoundQualPlan,
    pub(crate) prereg_path: std::path::PathBuf,
    pub(crate) committed_prereg_bytes: Vec<u8>,
    pub(crate) parsed_committed: SelfPlayPrereg,
    pub(crate) parent_map: BTreeMap<String, String>,
    pub(crate) ratchets: Vec<Value>,
    pub(crate) scores: Vec<QualTestScore>,
    pub(crate) parity: Vec<ParityRecord>,
    pub(crate) pinned_prereg: SelfPlayPrereg,
    pub(crate) pinned_bytes: Vec<u8>,
    pub(crate) pinned_plan: RoundQualPlan,
}

pub(crate) fn prereg_fixture() -> PreregFixture {
    let identity = fixture_identity();
    let mut plan = fixture_plan(&identity);
    plan.bootstrap_resamples = 2000;
    plan.bootstrap_seed = 20_261_016;
    plan.endpoint = Endpoint::RatchetVsOrdinaryFinalRound;

    let prereg_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/internal/clover/selfplay-round1-prereg.v1.json");
    let committed_prereg_bytes =
        std::fs::read(&prereg_path).expect("read committed selfplay-round1-prereg.v1.json");
    let parsed_committed: SelfPlayPrereg =
        serde_json::from_slice(&committed_prereg_bytes).expect("parse SelfPlayPrereg");
    assert_eq!(parsed_committed.schema, PREREG_SCHEMA);
    assert!(parsed_committed.has_unpinned_fields());
    assert!(!parsed_committed.execution_grant.authorized);

    plan.prereg_sha256 = sha256_hex(&committed_prereg_bytes);

    let mut parent_map = BTreeMap::new();
    let mut ratchets = Vec::new();
    let mut scores = Vec::new();
    let mut parity = Vec::new();

    for (ctx, parent, ord_rmse, cand_rmse) in [
        ("ctx1", "parent_a", 1.00, 0.72),
        ("ctx2", "parent_b", 1.10, 0.78),
        ("ctx3", "parent_c", 0.90, 0.64),
        ("ctx4", "parent_c", 1.20, 0.84),
    ] {
        parent_map.insert(ctx.to_owned(), parent.to_owned());
        let g_r0 = format!("g_r0_{ctx}");
        let g_ord_final = format!("g_ord_final_{ctx}");
        let g_rat_final = format!("g_rat_final_{ctx}");

        let ref_score = make_score(ctx, &g_ord_final, ord_rmse, Vec::new());
        let cand_score = make_score(
            ctx,
            &g_rat_final,
            cand_rmse,
            vec![ref_score.program_sha256.clone()],
        );
        scores.push(ref_score);
        scores.push(cand_score);
        for g in [&g_ord_final, &g_rat_final] {
            parity.push(ParityRecord {
                dataset_id: ctx.to_owned(),
                genome: (*g).clone(),
                passed: true,
                failure_reason: None,
            });
        }

        let rounds: Vec<Value> = (0..4)
        .map(|r_idx| {
            let is_last = r_idx == 3;
            let rat_g = if is_last { &g_rat_final } else { &g_r0 };
            let ord_g = if is_last { &g_ord_final } else { &g_r0 };
            json!({
                "round": r_idx,
                "cumulative_prior_rounds": r_idx,
                "ordinary_best_rmse": 1.00 - 0.01 * (r_idx as f64),
                "ordinary_best_q": 0.50 + 0.005 * (r_idx as f64),
                "ordinary_champion_genome": ord_g,
                "ordinary_search_sha256": sha256_hex(format!("ord_s:{ctx}:{r_idx}").as_bytes()),
                "ordinary_observations_sha256": sha256_hex(format!("ord_o:{ctx}:{r_idx}").as_bytes()),
                "ratchet_best_rmse": 1.00 - 0.08 * (r_idx as f64),
                "ratchet_best_q": 0.50 + 0.03 * (r_idx as f64),
                "ratchet_champion_genome": rat_g,
                "ratchet_search_sha256": sha256_hex(format!("rat_s:{ctx}:{r_idx}").as_bytes()),
                "ratchet_observations_sha256": sha256_hex(format!("rat_o:{ctx}:{r_idx}").as_bytes()),
                "surplus_q_vs_ordinary": 0.025 * (r_idx as f64),
                "equal_attempts_in_round": true,
                "kl_source": "measured_tilt_leaf_policy",
                "kl_malformed_reason": Value::Null,
                "decomposed_kl": fixture_decomposed_kl(true),
                "seed_update_frozen_by_kl_guard": false
            })
        })
        .collect();

        ratchets.push(json!({
            "schema": RATCHET_SCHEMA,
            "qualification": "qualified",
            "roster_sha256": identity.data_roster_sha256,
            "context_sha256": ctx,
            "seed_cap_b": 8,
            "max_conditional_base_kl": 0.20,
            "max_new_support_mass": 0.15,
            "measured_tilt_policy": parsed_committed.arms_and_rounds.measured_tilt_policy,
            "equal_resource_comparison": true,
            "kl_guard_evaluated": true,
            "any_round_frozen_by_kl_guard": false,
            "policy_optimizer_updates": 0,
            "rounds": rounds
        }));
    }

    assert_eq!(
        parsed_committed.endpoints.secondary_status,
        PREREG_SECONDARY_STATUS
    );
    assert_eq!(
        parsed_committed.provenance_binding_status,
        PREREG_UNPINNED_BINDING_STATUS
    );
    assert_eq!(
        parsed_committed
            .arms_and_rounds
            .kl_guard
            .allowlisted_measured_sources,
        vec!["measured_tilt_leaf_policy".to_owned()]
    );

    let mut pinned_prereg = parsed_committed.clone();
    pinned_prereg.status = "PRE_REGISTERED_PINNED_SYNTHETIC_FIXTURE".to_owned();
    pinned_prereg.provenance_binding_status = PREREG_PINNED_BINDING_STATUS.to_owned();
    pinned_prereg.experiment_identity_sha256 =
        UnpinnedOr::Pinned(plan.experiment_identity_sha256.clone());
    pinned_prereg.roster_and_families.sealed_roster_sha256 =
        UnpinnedOr::Pinned(identity.data_roster_sha256.clone());
    pinned_prereg.roster_and_families.parent_cluster_map_sha256 =
        UnpinnedOr::Pinned(canonical_sha256(&parent_map).unwrap());
    pinned_prereg.roster_and_families.g23_sealed_boundary_ruling =
        UnpinnedOr::Pinned("g23-synthetic-ruling-fixture".to_owned());
    pinned_prereg.sample_size_and_margin.min_parent_clusters =
        UnpinnedOr::Pinned(plan.min_parent_clusters);
    pinned_prereg
        .sample_size_and_margin
        .practical_margin_log_ratio = UnpinnedOr::Pinned(plan.practical_margin_log_ratio);
    pinned_prereg
        .sample_size_and_margin
        .c158_g_judge_binding_receipt = UnpinnedOr::Pinned("c158-synthetic-receipt".to_owned());
    pinned_prereg
        .held_out_and_parity_rules
        .pinned_test_score_registry = UnpinnedOr::Pinned(plan.pinned_test_score_registry.clone());
    pinned_prereg.execution_grant.authorized = true;
    pinned_prereg.execution_grant.owner_admission_locator =
        UnpinnedOr::Pinned("pending-review-locator-2026".to_owned());
    pinned_prereg.execution_grant.hosts = UnpinnedOr::Pinned(vec!["xbabe2".to_owned()]);
    pinned_prereg.execution_grant.cpu_ceiling_percent = UnpinnedOr::Pinned(50);
    pinned_prereg.execution_grant.deadline_utc =
        UnpinnedOr::Pinned("2026-10-18T00:00:00Z".to_owned());
    assert!(!pinned_prereg.has_unpinned_fields());

    let pinned_bytes = serde_json::to_vec_pretty(&pinned_prereg).unwrap();
    let mut pinned_plan = plan.clone();
    pinned_plan.prereg_sha256 = sha256_hex(&pinned_bytes);

    PreregFixture {
        identity,
        plan,
        prereg_path,
        committed_prereg_bytes,
        parsed_committed,
        parent_map,
        ratchets,
        scores,
        parity,
        pinned_prereg,
        pinned_bytes,
        pinned_plan,
    }
}

pub(crate) struct CliInputs {
    pub(crate) dir: tempfile::TempDir,
    pub(crate) plan_file: std::path::PathBuf,
    pub(crate) id_file: std::path::PathBuf,
    pub(crate) ratchets_file: std::path::PathBuf,
    pub(crate) scores_file: std::path::PathBuf,
    pub(crate) parity_file: std::path::PathBuf,
    pub(crate) parents_file: std::path::PathBuf,
    pub(crate) out_file: std::path::PathBuf,
}

pub(crate) fn write_cli_inputs(f: &PreregFixture) -> CliInputs {
    let dir = tempfile::tempdir().expect("tempdir");
    let plan_file = dir.path().join("plan.json");
    let id_file = dir.path().join("id.json");
    let ratchets_file = dir.path().join("ratchets.json");
    let scores_file = dir.path().join("scores.json");
    let parity_file = dir.path().join("parity.json");
    let parents_file = dir.path().join("parents.json");
    let out_file = dir.path().join("report_prereg.json");
    std::fs::write(&plan_file, serde_json::to_vec(&f.plan).unwrap()).unwrap();
    std::fs::write(&id_file, serde_json::to_vec(&f.identity).unwrap()).unwrap();
    std::fs::write(&ratchets_file, serde_json::to_vec(&f.ratchets).unwrap()).unwrap();
    std::fs::write(&scores_file, serde_json::to_vec(&f.scores).unwrap()).unwrap();
    std::fs::write(&parity_file, serde_json::to_vec(&f.parity).unwrap()).unwrap();
    std::fs::write(&parents_file, serde_json::to_vec(&f.parent_map).unwrap()).unwrap();
    CliInputs {
        dir,
        plan_file,
        id_file,
        ratchets_file,
        scores_file,
        parity_file,
        parents_file,
        out_file,
    }
}

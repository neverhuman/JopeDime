use super::*;

#[test]
fn closed_loop_surrogate_funnel_and_measured_tilt_selfplay_ratchet() {
    use crate::pools::pool::{Member, Pool, Status, POOL_SCHEMA};
    use crate::pools::selectors::{token_bag_features, ContextualSurrogateQHat, Features};

    let cut = Cut {
        index: 0,
        train_x: (0..24)
            .map(|i| {
                let u = (i as f64 - 12.0) * 0.25;
                vec![u, u * u + 0.5 * u.sin(), (i % 5) as f64 - 2.0]
            })
            .collect(),
        train_y: (0..24)
            .map(|i| {
                let u = (i as f64 - 12.0) * 0.25;
                2.0 + 1.5 * u + 0.8 * (u * u) - 0.3 * ((i % 5) as f64 - 2.0).abs()
            })
            .collect(),
        hold_x: (24..32)
            .map(|i| {
                let u = (i as f64 - 20.0) * 0.25;
                vec![u, u * u + 0.5 * u.sin(), (i % 5) as f64 - 2.0]
            })
            .collect(),
        hold_y: (24..32)
            .map(|i| {
                let u = (i as f64 - 20.0) * 0.25;
                2.0 + 1.5 * u + 0.8 * (u * u) - 0.3 * ((i % 5) as f64 - 2.0).abs()
            })
            .collect(),
        hold_indices: (24..32).collect(),
    };
    let context = DiscoveryContext {
        context_sha256: pin("nonlinear-ratchet-context"),
        parent_family: "synthetic-linear-parent".into(),
        role: "train".into(),
        dataset_sha256: pin("nonlinear-ratchet-data"),
        cut: 0,
        training_sha256: rows_digest(&cut.train_x, &cut.train_y).unwrap(),
        inner_holdout_sha256: rows_digest(&cut.hold_x, &cut.hold_y).unwrap(),
        preprocessing_sha256: pin("fixture-preprocessing"),
        encoder_input_sha256: jope_pipeline::sha256_hex(&vec![0; 2051 * 8]),
    };
    let (_, _, base_policy, mut retrieval) = fixture();
    retrieval.seeds[0].source_context_sha256 = context.context_sha256.clone();
    let roster = TrainingRoster {
        schema: "clover.discovery-roster/v1".into(),
        role_manifest_sha256: jope_pipeline::sha256_hex(&roles()),
        contexts: vec![context.clone()],
    };
    let target_ctx = &roster.contexts[0];

    // 1. Fit ContextualSurrogateQHat on 4 training pools and use select_seeds_with_surrogate_funnel
    // to pre-screen 4 distinct canonical genomes (K=4 -> B=2) into a validated PolicySeeds batch.
    let pop = jope_parallel::gp::metadata_initial_population(20260907).unwrap();
    let mut candidate_genomes = vec![
        canonical_genome("(pipe X (standardize append=false) (ridge))").unwrap(),
        canonical_genome("(pipe X (row_stats append=true) (ridge))").unwrap(),
    ];
    for g in &pop {
        let c = canonical_genome(g).unwrap();
        if c != search::BOOTSTRAP_GENOME && !candidate_genomes.contains(&c) {
            candidate_genomes.push(c);
            if candidate_genomes.len() == 4 {
                break;
            }
        }
    }
    assert_eq!(candidate_genomes.len(), 4);
    let cand_shas: Vec<String> = candidate_genomes
        .iter()
        .map(|g| jope_pipeline::sha256_hex(g.as_bytes()))
        .collect();

    let mk_train_pool = |idx: usize, z_pos: bool| -> Pool {
        let cid = pin(&format!("surrogate-train-ctx-{idx}"));
        let members = cand_shas
            .iter()
            .enumerate()
            .map(|(m_idx, sha)| {
                let is_even = m_idx % 2 == 0;
                let q = if z_pos == is_even { 0.85 } else { 0.25 };
                Member {
                    program_sha256: sha.clone(),
                    status: Status::Complete,
                    q: Some(q),
                    tokens: Some(vec![
                        1,
                        if is_even { 10 } else { 20 },
                        (3 + m_idx) as u32,
                        2,
                    ]),
                }
            })
            .collect();
        Pool {
            schema: POOL_SCHEMA.into(),
            context_id: cid,
            parent_id: format!("parent-{idx}"),
            role: "train".into(),
            protocol_sha256: pin("surrogate-protocol"),
            incumbent_q: Some(0.20),
            members,
        }
    };
    let train_pools = vec![
        mk_train_pool(0, true),
        mk_train_pool(1, false),
        mk_train_pool(2, true),
        mk_train_pool(3, false),
    ];
    let prog_feats = token_bag_features(&train_pools).unwrap();
    let mut ctx_feats = Features::context_id_descriptors(&train_pools, 1).unwrap();
    ctx_feats.rows.insert(
        target_ctx.context_sha256.clone(),
        Features::context_id_vector(&target_ctx.context_sha256, 1).unwrap(),
    );
    let train_refs: Vec<&Pool> = train_pools.iter().collect();
    let qhat =
        ContextualSurrogateQHat::fit(&train_refs, &ctx_feats, &prog_feats, 2, 2.0, 1e-2).unwrap();

    let (screened_policy, funnel) = select_seeds_with_surrogate_funnel(SurrogateSeedRequest {
        context: target_ctx,
        snapshot: &base_policy.snapshot,
        candidate_genomes: &candidate_genomes,
        selector: &qhat,
        proposal_pool_k: 4,
        eval_budget_b: 2,
    })
    .unwrap();
    assert_eq!(funnel.proposal_pool_k, 4);
    assert_eq!(funnel.eval_budget_b, 2);
    assert_eq!(screened_policy.proposals.len(), 2);
    assert_eq!(
        screened_policy.sampling_protocol,
        SAMPLING_SURROGATE_FUNNEL_TOPB_V1
    );
    assert_eq!(screened_policy.temperature, 0.0);
    assert!(screened_policy
        .proposals
        .iter()
        .all(|p| p.draw_seed == 0 && p.behavior_logprob == 0.0));
    let prov = screened_policy.surrogate_funnel.as_ref().unwrap();
    assert_eq!(prov.selector_name, "contextual_surrogate_qhat");
    assert_eq!(prov.allocation, funnel);
    assert_eq!(
        screened_policy.batch_protocol_sha256,
        canonical_sha256(prov).unwrap()
    );
    // Tampering with deterministic funnel invariants is rejected by validate.
    let mut bad_funnel_policy = screened_policy.clone();
    bad_funnel_policy.proposals[0].draw_seed = 1;
    assert!(bad_funnel_policy.validate(target_ctx).is_err());
    let mut bad_trie_policy = base_policy.clone();
    bad_trie_policy.surrogate_funnel = Some(prov.clone());
    assert!(bad_trie_policy.validate(target_ctx).is_err());

    // 2. Run 3-round closed-loop self-play ratchet with MeasuredTiltPolicyConfig.
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("closed_loop_ratchet");
    let round_budgets = vec![
        ScratchBudget {
            evaluations_per_arm: 16,
            seconds_per_arm: 30,
            rng_seed: 20261001,
        },
        ScratchBudget {
            evaluations_per_arm: 16,
            seconds_per_arm: 30,
            rng_seed: 20261002,
        },
        ScratchBudget {
            evaluations_per_arm: 16,
            seconds_per_arm: 30,
            rng_seed: 20261003,
        },
    ];
    let tilt_cfg = MeasuredTiltPolicyConfig {
        temperature_tau: 0.05,
        advantage_clip: 2.0,
        ref_new_support_mass: 0.02,
        policy_new_support_cap: 0.12,
        step_size_eta: 0.25,
    };
    let role_bytes = roles();
    let req = SelfPlayRatchetRequest {
        roster: &roster,
        role_manifest: &role_bytes,
        context_sha256: &target_ctx.context_sha256,
        cut: &cut,
        initial_policy: &screened_policy,
        initial_retrieval: &retrieval,
        round_budgets: &round_budgets,
        seed_cap_b: 4,
        max_conditional_base_kl: 0.05,
        max_new_support_mass: 0.15,
        round_leaf_log_prob_fn: None,
        measured_tilt_policy: Some(tilt_cfg),
    };
    let receipt = run_selfplay_ratchet(req, &output).unwrap();
    assert_eq!(receipt["kl_guard_evaluated"], true);
    assert_eq!(receipt["all_rounds_kl_admissible"], true);
    assert_eq!(receipt["any_round_frozen_by_kl_guard"], false);
    assert_eq!(receipt["monotonic_non_decreasing_best_q"], true);
    assert_eq!(receipt["matched_or_beat_ordinary_in_final_round"], true);
    assert_eq!(receipt["policy_optimizer_updates"], 0);
    assert_eq!(receipt["measured_tilt_rounds_admissible"], 3);
    assert_eq!(receipt["max_conditional_base_kl"], 0.05);
    assert_eq!(receipt["max_new_support_mass"], 0.15);
    assert_eq!(
        serde_json::from_value::<MeasuredTiltPolicyConfig>(receipt["measured_tilt_policy"].clone())
            .unwrap(),
        tilt_cfg
    );
    for r in receipt["rounds"].as_array().unwrap() {
        assert_eq!(r["kl_source"], "measured_tilt_leaf_policy");
        assert!(r["kl_malformed_reason"].is_null());
        let ce = r["policy_archive_ce_loss"].as_f64().unwrap();
        assert!(ce.is_finite() && ce > 0.0);
        assert_eq!(r["decomposed_kl"]["admissible"], true);
        assert_eq!(r["seed_update_frozen_by_kl_guard"], false);
        assert_eq!(r["next_round_seed_selector"], "measured_tilt_leaf_policy");
    }

    // P3-3 compatibility: a legacy round receipt JSON that omits `ordinary_champion_genome`
    // still decodes into `SelfPlayRoundReceipt` (with empty `ordinary_champion_genome`) and
    // round-trips without emitting the omitted field.
    let mut legacy_round_val = receipt["rounds"][0].clone();
    legacy_round_val
        .as_object_mut()
        .unwrap()
        .remove("ordinary_champion_genome");
    let decoded_legacy: SelfPlayRoundReceipt =
        serde_json::from_value(legacy_round_val.clone()).expect("decode legacy round receipt");
    assert!(decoded_legacy.ordinary_champion_genome.is_empty());
    let reencoded_legacy = serde_json::to_value(&decoded_legacy).unwrap();
    assert_eq!(reencoded_legacy, legacy_round_val);
}

use super::*;

#[test]
fn exp_next_02_producer_selfplay_ratchet_to_round_qual_bridge() {
    use jope_learning::clover::expansion::{
        canonical_genome, rows_digest, run_selfplay_ratchet, DiscoveryContext,
        MeasuredTiltPolicyConfig, PolicyProposal, PolicySeeds, PolicySnapshot, RetrievalSeed,
        RetrievalSeeds, ScratchBudget, SelfPlayRatchetRequest, SelfPlayRoundReceipt,
        TrainingRoster,
    };
    use jope_pipeline::prepare::Cut;

    let role_bytes = serde_json::to_vec(&json!({
        "schema": "clover.discovery-family-roles/v1",
        "families": {
            "parent_family_a": "train",
            "parent_family_b": "train"
        }
    }))
    .unwrap();
    let role_sha = sha256_hex(&role_bytes);

    let make_cut = |offset: f64| Cut {
        index: 0,
        train_x: (0..20)
            .map(|i| {
                let u = (i as f64 - 10.0) * 0.25 + offset;
                vec![u, u * u + 0.3 * u.sin()]
            })
            .collect(),
        train_y: (0..20)
            .map(|i| {
                let u = (i as f64 - 10.0) * 0.25 + offset;
                1.5 + 1.2 * u + 0.6 * u * u
            })
            .collect(),
        hold_x: (20..26)
            .map(|i| {
                let u = (i as f64 - 16.0) * 0.25 + offset;
                vec![u, u * u + 0.3 * u.sin()]
            })
            .collect(),
        hold_y: (20..26)
            .map(|i| {
                let u = (i as f64 - 16.0) * 0.25 + offset;
                1.5 + 1.2 * u + 0.6 * u * u
            })
            .collect(),
        hold_indices: (20..26).collect(),
    };

    let cut_a = make_cut(0.0);
    let cut_b = make_cut(0.5);
    let ctx_a = DiscoveryContext {
        context_sha256: sha256_hex(b"bridge-ctx-a"),
        parent_family: "parent_family_a".into(),
        role: "train".into(),
        dataset_sha256: sha256_hex(b"bridge-data-a"),
        cut: 0,
        training_sha256: rows_digest(&cut_a.train_x, &cut_a.train_y).unwrap(),
        inner_holdout_sha256: rows_digest(&cut_a.hold_x, &cut_a.hold_y).unwrap(),
        preprocessing_sha256: sha256_hex(b"bridge-prep"),
        encoder_input_sha256: sha256_hex(&vec![0u8; 2051 * 8]),
    };
    let ctx_b = DiscoveryContext {
        context_sha256: sha256_hex(b"bridge-ctx-b"),
        parent_family: "parent_family_b".into(),
        role: "train".into(),
        dataset_sha256: sha256_hex(b"bridge-data-b"),
        cut: 0,
        training_sha256: rows_digest(&cut_b.train_x, &cut_b.train_y).unwrap(),
        inner_holdout_sha256: rows_digest(&cut_b.hold_x, &cut_b.hold_y).unwrap(),
        preprocessing_sha256: sha256_hex(b"bridge-prep"),
        encoder_input_sha256: sha256_hex(&vec![0u8; 2051 * 8]),
    };
    let roster = TrainingRoster {
        schema: "clover.discovery-roster/v1".into(),
        role_manifest_sha256: role_sha,
        contexts: vec![ctx_a.clone(), ctx_b.clone()],
    };
    let roster_digest = roster.digest().unwrap();

    let seed_g1 = canonical_genome("(pipe X (standardize append=false) (ridge))").unwrap();
    let seed_g2 = canonical_genome("(pipe X (row_stats append=true) (ridge))").unwrap();
    let budgets = vec![
        ScratchBudget {
            evaluations_per_arm: 12,
            seconds_per_arm: 30,
            rng_seed: 20261011,
        },
        ScratchBudget {
            evaluations_per_arm: 12,
            seconds_per_arm: 30,
            rng_seed: 20261012,
        },
    ];
    let tilt_cfg = MeasuredTiltPolicyConfig {
        temperature_tau: 0.25,
        advantage_clip: 2.0,
        ref_new_support_mass: 0.05,
        policy_new_support_cap: 0.12,
        step_size_eta: 0.20,
    };

    let temp = tempfile::tempdir().unwrap();
    let mut produced_ratchets = Vec::new();
    let mut scores = Vec::new();
    let mut parity = Vec::new();
    let mut parent_map = BTreeMap::new();

    for (idx, (ctx, cut)) in [(&ctx_a, &cut_a), (&ctx_b, &cut_b)].into_iter().enumerate() {
        parent_map.insert(ctx.context_sha256.clone(), ctx.parent_family.clone());
        let policy = PolicySeeds {
            schema: "clover.policy-seeds/v1".into(),
            snapshot: PolicySnapshot {
                model_class: "PolicyLM".into(),
                context_width: 2051,
                parameter_sha256: sha256_hex(b"bridge-params"),
                checkpoint_sha256: sha256_hex(b"bridge-ckpt"),
                code_sha256: sha256_hex(b"bridge-code"),
                tokenizer_sha256: sha256_hex(b"bridge-tok"),
                support_sha256: sha256_hex(b"bridge-sup"),
                normalizer_sha256: sha256_hex(b"bridge-norm"),
            },
            context_sha256: ctx.context_sha256.clone(),
            encoder_input_sha256: ctx.encoder_input_sha256.clone(),
            sampling_protocol: "trie_children_temperature_v1".into(),
            batch_protocol_sha256: sha256_hex(b"bridge-batch"),
            temperature: 1.0,
            proposals: vec![PolicyProposal {
                program_sha256: sha256_hex(seed_g1.as_bytes()),
                genome: seed_g1.clone(),
                draw_seed: 1,
                behavior_logprob: -2.0,
            }],
            surrogate_funnel: None,
        };
        let retrieval = RetrievalSeeds {
            seeds: vec![RetrievalSeed {
                genome: seed_g2.clone(),
                source_context_sha256: ctx.context_sha256.clone(),
                measurement_receipt_sha256: sha256_hex(b"bridge-ret-rec"),
            }],
        };
        let out_dir = temp.path().join(format!("ratchet_{idx}"));
        let receipt = run_selfplay_ratchet(
            SelfPlayRatchetRequest {
                roster: &roster,
                role_manifest: &role_bytes,
                context_sha256: &ctx.context_sha256,
                cut,
                initial_policy: &policy,
                initial_retrieval: &retrieval,
                round_budgets: &budgets,
                seed_cap_b: 4,
                max_conditional_base_kl: 0.20,
                max_new_support_mass: 0.15,
                round_leaf_log_prob_fn: None,
                measured_tilt_policy: Some(tilt_cfg),
            },
            &out_dir,
        )
        .unwrap();

        assert_eq!(receipt["max_conditional_base_kl"], json!(0.20));
        assert_eq!(receipt["max_new_support_mass"], json!(0.15));
        assert_eq!(
            receipt["measured_tilt_policy"],
            serde_json::to_value(tilt_cfg).unwrap()
        );

        let round_recs: Vec<SelfPlayRoundReceipt> =
            serde_json::from_value(receipt["rounds"].clone()).unwrap();
        assert_eq!(round_recs.len(), 2);
        let final_round = round_recs.last().unwrap();
        assert!(!final_round.ordinary_champion_genome.is_empty());
        assert!(!final_round.ratchet_champion_genome.is_empty());

        let ref_g = &final_round.ordinary_champion_genome;
        let cand_g = &final_round.ratchet_champion_genome;
        if ref_g == cand_g {
            scores.push(make_score(&ctx.context_sha256, ref_g, 0.80, Vec::new()));
            parity.push(ParityRecord {
                dataset_id: ctx.context_sha256.clone(),
                genome: ref_g.clone(),
                passed: true,
                failure_reason: None,
            });
        } else {
            let ref_s = make_score(&ctx.context_sha256, ref_g, 0.95, Vec::new());
            let cand_s = make_score(
                &ctx.context_sha256,
                cand_g,
                0.70,
                vec![ref_s.program_sha256.clone()],
            );
            scores.push(ref_s);
            scores.push(cand_s);
            for g in [ref_g, cand_g] {
                parity.push(ParityRecord {
                    dataset_id: ctx.context_sha256.clone(),
                    genome: g.clone(),
                    passed: true,
                    failure_reason: None,
                });
            }
        }
        produced_ratchets.push(receipt);
    }

    let mut identity = fixture_identity();
    identity.data_roster_sha256 = roster_digest;
    let mut plan = fixture_plan(&identity);
    plan.min_parent_clusters = 2;
    plan.endpoint = Endpoint::RatchetVsOrdinaryFinalRound;

    let report = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &produced_ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("qualify real run_selfplay_ratchet outputs on RatchetVsOrdinaryFinalRound");

    assert_eq!(report.n_ratchet_contexts, 2);
    assert_eq!(report.n_complete_paired_contexts, 2);
    assert_eq!(report.n_observed_parent_clusters, 2);
    assert_eq!(report.n_unmeasured_kl_contexts, 0);
    assert!(!report.qualified_claim);
    assert!(!report.promotion);
    assert!(!report.l6_claim);
    assert!(report
        .disqualification_reasons
        .contains(&"source_grade_not_qualified".to_owned()));
    assert!(report
        .disqualification_reasons
        .contains(&"unequal_resource_comparison".to_owned()));

    // Also verify end-to-end producer -> qualify_rounds_with_prereg binding using the real
    // `run_selfplay_ratchet` receipts (including their emitted `measured_tilt_policy`,
    // `max_conditional_base_kl`, and `max_new_support_mass` contract fields).
    let prereg_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/internal/clover/selfplay-round1-prereg.v1.json");
    let mut bridge_prereg: SelfPlayPrereg =
        serde_json::from_slice(&std::fs::read(&prereg_path).unwrap()).unwrap();
    bridge_prereg.arms_and_rounds.rounds = 2;
    bridge_prereg.arms_and_rounds.seed_cap_b = 4;
    bridge_prereg.sample_size_and_margin.bootstrap_resamples = plan.bootstrap_resamples;
    bridge_prereg.sample_size_and_margin.bootstrap_seed = plan.bootstrap_seed;
    let bridge_prereg_bytes = serde_json::to_vec(&bridge_prereg).unwrap();
    plan.prereg_sha256 = sha256_hex(&bridge_prereg_bytes);

    let report_with_prereg = qualify_rounds_with_prereg(
        RoundQualInputs {
            plan: &plan,
            identity: &identity,
            ratchet_outputs: &produced_ratchets,
            test_scores: &scores,
            parity: &parity,
            parent_of_context: &parent_map,
        },
        Some(&bridge_prereg_bytes),
    )
    .expect("qualify real run_selfplay_ratchet outputs with --prereg binding");
    assert_eq!(
        report_with_prereg.provenance_binding_status,
        PREREG_UNPINNED_BINDING_STATUS
    );
    assert!(!report_with_prereg.qualified_claim);
    assert!(!report_with_prereg.promotion);
    assert!(!report_with_prereg.l6_claim);
    assert!(report_with_prereg
        .disqualification_reasons
        .contains(&PREREG_CALLER_ASSERTED_REASON.to_owned()));
    assert!(report_with_prereg
        .disqualification_reasons
        .contains(&"qualified_execution_mode_disabled_offline_verifier_only".to_owned()));
}

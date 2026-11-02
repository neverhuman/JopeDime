use super::*;

pub fn verify_spec_b_g12_off_bin_and_unsupported() {
    let bound = bound_vocab();
    let parent_prog = "(pipe X (ridge))";
    let off_bin_descendant_prog = "(pipe X (standardize) (ridge))";
    let parent_tokens = bound.representable_tokens(parent_prog).unwrap();
    let parent_sha = sha256_hex(parent_prog.as_bytes());
    let descendant_sha = sha256_hex(off_bin_descendant_prog.as_bytes());
    assert_ne!(parent_sha, descendant_sha);

    // Construct an 80-candidate ProposalLedger (64 ranked pool slots + 16 exploration slots)
    // plus 3 unsupported/shadow draws (1 ParserRejected neural, 1 Incomplete neural, 1 ParserRejected exploration)
    let scores: Vec<f64> = (0..64).map(|i| 10.0 - (i as f64) * 0.05).collect();
    let rank_weights = jope_policy::seeds::logprob_from_scores(&scores).unwrap();

    let mut neural_draws = Vec::with_capacity(66);
    let mut exploration_draws = Vec::with_capacity(17);
    let mut candidates = Vec::with_capacity(80);

    for slot in 0..64usize {
        let prog = format!("(pipe X (select {slot}) (ridge))");
        let prog_sha = sha256_hex(prog.as_bytes());
        let mut toks = parent_tokens.clone();
        toks.insert(2, 10 + (slot % 300));
        let toks_sha = sha256_hex(&serde_json::to_vec(&toks).unwrap());
        let canon_sha = sha256_hex(format!("canonical:{slot}").as_bytes());

        neural_draws.push(NeuralDraw {
            round: 0,
            row: slot,
            outcome: DrawOutcome::Completed,
            tokens: toks.clone(),
            grammar_phase: "done".into(),
            program_sha256: Some(prog_sha.clone()),
        });

        candidates.push(SlotRecord {
            slot,
            origin: ORIGIN_NEURAL.into(),
            source: SlotSource::NeuralDraw {
                round: 0,
                row: slot,
            },
            tokens: toks,
            program: prog,
            program_sha256: prog_sha.clone(),
            canonical_recipe_sha256: canon_sha,
            adaptations: vec![
                AdaptationEdge {
                    kind: EDGE_DECODE.into(),
                    input_sha256: toks_sha.clone(),
                    output_sha256: prog_sha.clone(),
                    changed: false,
                },
                AdaptationEdge {
                    kind: EDGE_SCORE_ENCODING.into(),
                    input_sha256: prog_sha.clone(),
                    output_sha256: toks_sha,
                    changed: false,
                },
            ],
            prediction: Some(SlotPrediction {
                rank: slot,
                predicted_quality: scores[slot],
                reward_heads: [scores[slot]; 4],
                predicted_runtime_seconds: 0.05,
                rank_log_weight: rank_weights[slot],
                scored_program_sha256: prog_sha,
                scored_token_count: 6,
            }),
        });
    }

    // Add 2 unsupported/refused neural draws in the same 66-row round
    neural_draws.push(NeuralDraw {
        round: 0,
        row: 64,
        outcome: DrawOutcome::ParserRejected,
        tokens: parent_tokens.clone(),
        grammar_phase: "done".into(),
        program_sha256: None,
    });
    neural_draws.push(NeuralDraw {
        round: 0,
        row: 65,
        outcome: DrawOutcome::Incomplete,
        tokens: vec![1; 128],
        grammar_phase: "open".into(),
        program_sha256: None,
    });

    let exp_seed_base = 5000u64;
    // First exploration draw is ParserRejected (unsupported shadow draw)
    exploration_draws.push(UniformDraw {
        seed: exp_seed_base,
        outcome: UniformOutcome::ParserRejected,
        tokens: parent_tokens.clone(),
        program_sha256: None,
    });

    for idx in 0..16usize {
        let slot = 64 + idx;
        let seed = exp_seed_base + 1 + (idx as u64);
        let prog = format!("(pipe X (select {}) (lasso))", 100 + idx);
        let prog_sha = sha256_hex(prog.as_bytes());
        let mut toks = parent_tokens.clone();
        toks.insert(2, 200 + idx);
        let toks_sha = sha256_hex(&serde_json::to_vec(&toks).unwrap());
        let canon_sha = sha256_hex(format!("canonical:exp:{idx}").as_bytes());

        exploration_draws.push(UniformDraw {
            seed,
            outcome: UniformOutcome::Admitted,
            tokens: toks.clone(),
            program_sha256: Some(prog_sha.clone()),
        });

        candidates.push(SlotRecord {
            slot,
            origin: jope_policy::seeds::ORIGIN_RANDOM.into(),
            source: SlotSource::ExplorationDraw { seed },
            tokens: toks,
            program: prog,
            program_sha256: prog_sha.clone(),
            canonical_recipe_sha256: canon_sha,
            adaptations: vec![AdaptationEdge {
                kind: EDGE_DECODE.into(),
                input_sha256: toks_sha,
                output_sha256: prog_sha,
                changed: false,
            }],
            prediction: None,
        });
    }

    let ledger = ProposalLedger {
        schema: LEDGER_SCHEMA.into(),
        task: "regression".into(),
        checkpoint_sha256: digest(0xaa),
        context_sha256: digest(0xbb),
        vocabulary_sha256: bound.sha256().to_string(),
        sampling: SamplingRecord {
            sampling_seed: 1,
            fallback_seed: 2,
            exploration_seed: exp_seed_base,
            rounds_scheduled: 1,
            rounds_run: 1,
            batch: 66,
            temperature: 1.0,
            token_budget: 128,
        },
        neural_draws,
        fallback_draws: vec![],
        exploration_draws,
        candidates,
        denominators: Denominators {
            scheduled_neural_draws: 66,
            completed: 64,
            surplus: 0,
            duplicate: 0,
            parser_rejected: 1,
            incomplete: 1,
            complete_sequences: 65,
            valid_programs: 64,
            unique_programs: 64,
            fallback_draws: 0,
            fallback_admitted: 0,
            exploration_draws: 17,
            exploration_admitted: 16,
            scored: 64,
            rank_survivors: 64,
            handed_off: 80,
            unique_canonical_handed_off: 80,
            shadow: 3,
        },
    };

    ledger
        .reconcile()
        .expect("valid 80-candidate ledger with counted unsupported/shadow draws must reconcile");

    let shadows = ledger.shadow();
    assert_eq!(
        shadows.len(),
        3,
        "all 3 non-handed-off draws must be tracked as shadow records"
    );
    for s in &shadows {
        assert!(
            !s.fitting_authority,
            "shadow/unsupported record must never carry fitting_authority"
        );
    }

    // SPEC_B G12: Mutate candidate 0 so an off-bin descendant attempts to inherit parent's projected score
    let mut inherited = ledger.clone();
    inherited.candidates[0]
        .prediction
        .as_mut()
        .unwrap()
        .scored_program_sha256 = descendant_sha;
    assert!(
        inherited.reconcile().is_err(),
        "SPEC_B G12: inheriting projected score from a different program must be refused"
    );

    // Also verify ContextAbiV2 residual closure refuses model_state / feature_selection inheritance across off-bin edits
    let mut bad_ctx = make_context(16, &[0, 4, 8], vec![1, 3, 2], None, 100);
    bad_ctx.residual.model_state_sha256 = digest(0xfe);
    assert!(
        bad_ctx.validate().is_err(),
        "SPEC_B G12: residual closure with stale parent model_state_sha256 must fail closed"
    );
}

pub fn verify_spec_b_g13_action_laws_normalize_and_no_minted_propensity() {
    let ctx = make_context(
        16,
        &[1, 5, 9, 12],
        vec![1, 7, 11, 2],
        Some("unix_seconds".into()),
        400,
    );
    let p1 = ctx.column_pointer(1).unwrap();
    let p5 = ctx.column_pointer(5).unwrap();
    let p9 = ctx.column_pointer(9).unwrap();

    let k_eos = EditKey::eos();
    let k1 = EditKey {
        kind: EditKind::ColumnTransform,
        site: Some(EditSite::Column(p1)),
        operator: Some(EditOperator::Replace),
        fidelity: Some(Fidelity::Exact),
        stop: false,
    };
    let k5 = EditKey {
        kind: EditKind::ColumnTransform,
        site: Some(EditSite::Column(p5)),
        operator: Some(EditOperator::Remove),
        fidelity: Some(Fidelity::Exact),
        stop: false,
    };
    let k9_time = EditKey {
        kind: EditKind::ColumnTransform,
        site: Some(EditSite::Column(p9)),
        operator: Some(EditOperator::TimeLag),
        fidelity: Some(Fidelity::Approximate),
        stop: false,
    };

    let discrete_policy = EditPolicy::new(
        ctx.clone(),
        vec![
            PolicyComponent {
                weight: 0.15,
                key: k_eos.clone(),
                parameter: None,
            },
            PolicyComponent {
                weight: 0.35,
                key: k1.clone(),
                parameter: None,
            },
            PolicyComponent {
                weight: 0.20,
                key: k5.clone(),
                parameter: None,
            },
            PolicyComponent {
                weight: 0.30,
                key: k9_time.clone(),
                parameter: None,
            },
        ],
    )
    .unwrap();

    // Test normalization across multiple legal sub-masks
    let masks = [
        ActionMask::new([k_eos.clone(), k1.clone(), k5.clone(), k9_time.clone()]).unwrap(),
        ActionMask::new([k_eos.clone(), k1.clone()]).unwrap(),
        ActionMask::new([k5.clone(), k9_time.clone()]).unwrap(),
        ActionMask::new([k9_time.clone()]).unwrap(),
    ];

    for mask in &masks {
        let mut mass_sum = 0.0f64;
        for key in &mask.allowed {
            let m = discrete_policy
                .score(
                    mask,
                    &EditAction {
                        key: key.clone(),
                        parameter: None,
                    },
                )
                .expect("score legal discrete action");
            let ProbabilityMeasure::Mass(p) = m else {
                panic!("expected discrete Mass");
            };
            assert!(p > 0.0 && p <= 1.0);
            mass_sum += p;
        }
        assert!(
            (mass_sum - 1.0).abs() <= 1e-12,
            "SPEC_B G13: enumerated discrete action law must normalize to 1.0, got {mass_sum}"
        );
    }

    // Verify unnormalized weights are rejected at construction
    assert!(
        EditPolicy::new(
            ctx,
            vec![PolicyComponent {
                weight: 0.75,
                key: k_eos,
                parameter: None,
            }],
        )
        .is_err(),
        "unnormalized mixture weights must be rejected"
    );

    // Verify Unknown selection propensity cannot mint unbiasedness authority
    let unknown = BehaviorLaw::unknown_selection("off-policy handoff without propensity");
    assert!(!unknown.permits_unbiased_claim());
    assert!(unknown.is_unknown());
    assert!(unknown.validate().is_err());
    assert!(matches!(unknown.value(), PropensityValue::Unknown { .. }));
}

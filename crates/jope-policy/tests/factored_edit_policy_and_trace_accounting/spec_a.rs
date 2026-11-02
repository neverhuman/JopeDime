use super::*;

pub fn verify_spec_a_g7_10k_attempts() {
    let base_ctx = make_context(32, &[0, 3, 7, 15, 21, 31], vec![1, 3, 19, 42, 2], None, 250);
    let ptr_3 = base_ctx.column_pointer(3).unwrap();
    let ptr_15 = base_ctx.column_pointer(15).unwrap();
    let ptr_31 = base_ctx.column_pointer(31).unwrap();

    let key_eos = EditKey::eos();
    let key_col3 = EditKey {
        kind: EditKind::ColumnTransform,
        site: Some(EditSite::Column(ptr_3.clone())),
        operator: Some(EditOperator::Replace),
        fidelity: Some(Fidelity::Exact),
        stop: false,
    };
    let key_col15_param = EditKey {
        kind: EditKind::ColumnTransform,
        site: Some(EditSite::Column(ptr_15.clone())),
        operator: Some(EditOperator::Insert),
        fidelity: Some(Fidelity::Approximate),
        stop: false,
    };
    let key_node = EditKey {
        kind: EditKind::ProgramTransform,
        site: Some(EditSite::Node(2)),
        operator: Some(EditOperator::Versioned(3)),
        fidelity: Some(Fidelity::Exact),
        stop: false,
    };
    let key_col31 = EditKey {
        kind: EditKind::ColumnTransform,
        site: Some(EditSite::Column(ptr_31.clone())),
        operator: Some(EditOperator::Remove),
        fidelity: Some(Fidelity::Exact),
        stop: false,
    };

    let policy = EditPolicy::new(
        base_ctx.clone(),
        vec![
            PolicyComponent {
                weight: 0.15,
                key: key_eos.clone(),
                parameter: None,
            },
            PolicyComponent {
                weight: 0.25,
                key: key_col3.clone(),
                parameter: None,
            },
            PolicyComponent {
                weight: 0.30,
                key: key_col15_param.clone(),
                parameter: Some(NormalDensity {
                    mean: 0.25,
                    standard_deviation: 0.75,
                }),
            },
            PolicyComponent {
                weight: 0.20,
                key: key_node.clone(),
                parameter: None,
            },
            PolicyComponent {
                weight: 0.10,
                key: key_col31.clone(),
                parameter: None,
            },
        ],
    )
    .expect("valid factored EditPolicy");

    let full_mask = policy.action_mask().unwrap();
    let restricted_mask =
        ActionMask::new([key_eos.clone(), key_col3.clone(), key_col15_param.clone()]).unwrap();

    let mut accepted_children = 0usize;
    let mut explicitly_refused = 0usize;
    let mut truncated_accepted = 0usize;
    let mut misbound_accepted = 0usize;

    // Run 10,500 free-running attempts combining valid draws and adversarial boundary probes
    let total_attempts = 10_500usize;
    for attempt in 0..total_attempts {
        let mask = if attempt % 3 == 0 {
            &restricted_mask
        } else {
            &full_mask
        };

        // Every 25th attempt injects an explicit invalid/truncation/misbinding probe that MUST be refused
        if attempt % 25 == 24 {
            let probe_kind = (attempt / 25) % 4;
            let refused = match probe_kind {
                0 => {
                    // Over-length program (> MAX_PROGRAM_TOKENS) must not silently truncate
                    let mut overlong = base_ctx.clone();
                    overlong.program_tokens = vec![1; MAX_PROGRAM_TOKENS + 5];
                    overlong.validate().is_err()
                }
                1 => {
                    // Unselected column index (column 2 is in schema but not in selected_column_keys)
                    let bad_ptr = ColumnPointer {
                        schema_sha256: base_ctx.schema_sha256.clone(),
                        index: 2,
                        key: "col_002".into(),
                    };
                    bad_ptr.validate_against(&base_ctx).is_err()
                }
                2 => {
                    // Misbound schema_sha256 pointer
                    let bad_ptr = ColumnPointer {
                        schema_sha256: digest(0x99),
                        index: 3,
                        key: "col_003".into(),
                    };
                    let bad_key = EditKey {
                        kind: EditKind::ColumnTransform,
                        site: Some(EditSite::Column(bad_ptr)),
                        operator: Some(EditOperator::Replace),
                        fidelity: Some(Fidelity::Exact),
                        stop: false,
                    };
                    bad_key.validate(&base_ctx).is_err()
                }
                _ => {
                    // Undeclared TimeLag edit on context without ordered_time_semantics
                    let bad_time = EditKey {
                        kind: EditKind::ProgramTransform,
                        site: Some(EditSite::Program),
                        operator: Some(EditOperator::TimeLag),
                        fidelity: Some(Fidelity::Exact),
                        stop: false,
                    };
                    bad_time.validate(&base_ctx).is_err()
                }
            };
            assert!(
                refused,
                "adversarial probe {probe_kind} at attempt {attempt} must fail closed"
            );
            explicitly_refused += 1;
            continue;
        }

        // Deterministic quasi-random draws in (0, 1) and finite normal_draw
        let u = ((attempt as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            >> 11) as f64
            / ((1u64 << 53) as f64);
        let z = (((attempt as i64 % 41) - 20) as f64) * 0.125;

        let action = policy
            .sample(mask, u, z)
            .expect("valid masked sample must succeed");

        // Check zero silent truncation or misbinding on accepted child
        if action.key.validate(&policy.context).is_err()
            || !mask.allowed.contains(&action.key)
            || policy.context.program_tokens.len() > MAX_PROGRAM_TOKENS
        {
            truncated_accepted += 1;
        }
        if let Some(EditSite::Column(ref ptr)) = action.key.site {
            if ptr.validate_against(&policy.context).is_err() {
                misbound_accepted += 1;
            }
        }
        let measure = policy
            .score(mask, &action)
            .expect("accepted child must have positive score under mask");
        match measure {
            ProbabilityMeasure::Mass(m) => assert!(m.is_finite() && m > 0.0 && m <= 1.0),
            ProbabilityMeasure::Density(d) => assert!(d.is_finite() && d > 0.0),
        }

        // Deterministic replay check
        let replayed = policy.sample(mask, u, z).unwrap();
        assert_eq!(
            action, replayed,
            "sample replay mismatch at attempt {attempt}"
        );
        accepted_children += 1;
    }

    assert!(
        accepted_children >= 10_000,
        "expected >= 10,000 accepted children, got {accepted_children}"
    );
    assert!(explicitly_refused >= 400);
    assert_eq!(
        truncated_accepted, 0,
        "SPEC_A G7: zero silently truncated accepted children required"
    );
    assert_eq!(
        misbound_accepted, 0,
        "SPEC_A G7: zero misbound accepted children required"
    );
}

pub fn verify_spec_a_g7_legacy_mode_parity() {
    let bound = bound_vocab();
    let vocab = bound.vocab();
    assert_eq!(vocab.tokens.len(), 401);

    let law = BehaviorLaw::native(bound.sha256(), 0.85).expect("valid native BehaviorLaw");
    let programs = [
        "(pipe X (ridge))",
        "(pipe X (blom) (rank) (ridge))",
        "(pipe X (standardize) (sigmoid) (robust_scale) (stack))",
        "(pipe X (blom) (rank) (stack))",
    ];

    for program in programs {
        let tokens = bound
            .representable_tokens(program)
            .expect("representable legacy program");
        let decoded = vocab.decode(&tokens).expect("decode tokens");
        assert_eq!(
            decoded, program,
            "exact legacy decode roundtrip for {program}"
        );

        // Replay legal sets step-by-step and compare against deterministic replay & State::advance walk
        let replayed_legal_sets = vocab.replay(&tokens).expect("vocab.replay");
        let replayed_again = vocab.replay(&tokens).expect("vocab.replay second pass");
        assert_eq!(
            replayed_legal_sets, replayed_again,
            "legacy replay legal sets must be byte-identical across runs for {program}"
        );
        let mut state = State::default();
        assert_eq!(replayed_legal_sets.len(), tokens.len() - 1);

        let mut logits_rows = Vec::with_capacity(tokens.len() - 1);
        for (step, &next_tok) in tokens[1..].iter().enumerate() {
            let direct_legal = state.legal(vocab);
            let replay_legal = &replayed_legal_sets[step];
            // Sampling legal set (`State::default()`) is a strict subset of `vocab.replay` (`sampling: false`)
            // and both contain `next_tok` in sorted order
            assert!(
                direct_legal
                    .iter()
                    .all(|tok| replay_legal.binary_search(tok).is_ok()),
                "sampling legal set must be contained in full replay legal set at step {step} for {program}"
            );
            assert!(
                direct_legal.binary_search(&next_tok).is_ok()
                    && replay_legal.binary_search(&next_tok).is_ok(),
                "token {next_tok} must be in both legal sets at step {step}"
            );

            let mut row = vec![0.0f32; vocab.tokens.len()];
            for (k, &lt) in replay_legal.iter().enumerate() {
                row[lt] = ((step * 17 + k * 7) % 19) as f32 * 0.15 - 0.9;
            }
            let via_decode = decode::score_legal(&row, replay_legal, 0.85).unwrap();
            let via_law = law.score_legal(&row, replay_legal).unwrap();
            assert_eq!(via_decode.len(), via_law.len(), "score_legal length parity");
            for ((t1, p1), (t2, p2)) in via_decode.iter().zip(via_law.iter()) {
                assert_eq!(t1, t2);
                assert_eq!(
                    p1.to_bits(),
                    p2.to_bits(),
                    "exact bit equality between decode::score_legal and BehaviorLaw::score_legal"
                );
            }

            logits_rows.push(row);
            state
                .advance(vocab, next_tok)
                .expect("advance grammar state");
        }
        assert!(
            state.done(),
            "complete program must reach terminal EOS state"
        );

        let prop1 = law.score_program(&bound, &tokens, &logits_rows).unwrap();
        let prop2 = law.score_program(&bound, &tokens, &logits_rows).unwrap();
        assert_eq!(
            serde_json::to_vec(&prop1).unwrap(),
            serde_json::to_vec(&prop2).unwrap(),
            "byte-identical propensity trace for {program}"
        );
    }
}

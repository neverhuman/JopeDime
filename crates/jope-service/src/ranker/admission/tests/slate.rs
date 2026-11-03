use super::*;

#[test]
fn custom_objective_changes_elite_order_without_changing_ranker_inputs() {
    use crate::discover::elites::{Elite, ElitePool, ScoreOrder};
    use crate::objective::regression::RegressionObjective;

    let fixture = AdmissionSlateFixture::new();
    for explore in [false, true] {
        let baseline = fixture
            .prepare(RevealedRule::SurrogateFitness, explore)
            .unwrap();
        let expected = baseline.canonical_inputs();
        for direction in ["minimize", "maximize"] {
            let objective = RegressionObjective::parse(
                &serde_json::json!({
                    "metric": "custom", "name": "rel07_untrained_pred_mean",
                    "direction": direction, "expression": {"op": "pred"},
                    "aggregation": {"kind": "mean"}
                })
                .to_string(),
            )
            .unwrap();
            let mut elites = ElitePool::new(ScoreOrder::Minimize, ScoreOrder::Minimize);
            for (i, measured) in fixture.measured().iter().enumerate() {
                let score = objective.evaluate(&[0.0], &[i as f64]).unwrap();
                elites
                    .insert(Elite {
                        algorithm_id: measured.id.to_owned(),
                        surrogate: measured.surrogate_fitness,
                        objective: score.loss,
                    })
                    .unwrap();
            }
            assert_eq!(
                elites.by_objective()[0].algorithm_id,
                if direction == "minimize" {
                    "program-0"
                } else {
                    "program-5"
                }
            );
            let ordered: Vec<_> = elites
                .by_objective()
                .iter()
                .map(|e| {
                    let i = fixture
                        .keys
                        .iter()
                        .position(|k| k == &e.algorithm_id)
                        .unwrap();
                    MeasuredAlgorithm {
                        id: &fixture.keys[i],
                        surrogate_fitness: e.surrogate,
                        gate: i % 2 == 0,
                    }
                })
                .collect();
            let programs = fixture.programs();
            let pool: Vec<_> = ordered
                .iter()
                .map(|m| programs.iter().find(|p| p.key == m.id).copied().unwrap())
                .collect();
            let slate = ServingAdmissionSlate::prepare(
                &ordered,
                &pool,
                &programs[6..if explore { 22 } else { 14 }],
                &fixture.conditioning,
                RevealedRule::SurrogateFitness,
                explore,
            )
            .unwrap();
            let actual = slate.canonical_inputs();
            assert_eq!(slate.revealed_keys(), baseline.revealed_keys());
            assert_eq!(slate.proposal_keys(), baseline.proposal_keys());
            assert_eq!(
                (actual.batch, actual.candidates),
                (expected.batch, expected.candidates)
            );
            assert_eq!(actual.revealed_indices, expected.revealed_indices);
            // Check every floating-point input bit, including conditioning,
            // reveals, features and the frozen context/prior/scale channels.
            let channels = |input: &PredictCostsInputs<'_>| {
                [
                    input.prog_412,
                    input.proj_16,
                    input.tokens,
                    input.token_mask,
                    input.revealed_fitness,
                    input.revealed_gates,
                    input.ctx_feat,
                    input.base_prior_cost,
                    input.sos_prior,
                    &[input.scale],
                ]
                .into_iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
            };
            assert_eq!(channels(&actual), channels(&expected));
        }
    }
}

#[test]
fn assembly_aligns_rows_and_frozen_context_channels_for_both_shapes() {
    let fixture = AdmissionSlateFixture::new();
    for explore in [false, true] {
        let slate = fixture
            .prepare(RevealedRule::SurrogateFitness, explore)
            .unwrap();
        assert_eq!(slate.revealed_pool_indices(), [5, 4, 3, 2]);
        assert_eq!(
            slate.revealed_keys(),
            ["program-5", "program-4", "program-3", "program-2"]
        );
        let expected_proposals: Vec<_> = fixture.keys[6..if explore { 22 } else { 14 }]
            .iter()
            .map(String::as_str)
            .collect();
        assert_eq!(slate.proposal_keys(), expected_proposals);
        let inputs = slate.canonical_inputs();
        assert_eq!(
            (inputs.batch, inputs.candidates),
            (1, if explore { 20 } else { 12 })
        );
        // Frozen cpu_exact_fused.py:1084 / serving caller:412-413:
        // one conditioning row per context batch, independent of N.
        assert_eq!(inputs.tokens, fixture.conditioning.tokens.as_flattened());
        assert_eq!(inputs.token_mask, fixture.conditioning.mask);
        assert_eq!(inputs.revealed_indices, [0, 1, 2, 3]);
        assert_eq!(inputs.revealed_fitness, [0., 1., 2., 3.]);
        assert_eq!(inputs.revealed_gates, [0., 1., 0., 1.]);
        assert_eq!(inputs.scale.to_bits(), adapter::SCALE.to_bits());
        assert!(inputs
            .ctx_feat
            .iter()
            .chain(inputs.base_prior_cost)
            .chain(inputs.sos_prior)
            .all(|&v| v == 0.));
        for (row, source) in [5, 4, 3, 2]
            .into_iter()
            .chain(6..if explore { 22 } else { 14 })
            .enumerate()
        {
            assert_eq!(
                &inputs.prog_412[row * adapter::PROG_FEATURES..(row + 1) * adapter::PROG_FEATURES],
                &fixture.features[source]
            );
            assert_eq!(
                &inputs.proj_16[row * adapter::PROJ_FEATURES..(row + 1) * adapter::PROJ_FEATURES],
                &fixture.projections[source]
            );
        }
    }
}

#[test]
fn assembly_uses_explicit_reveal_rule_without_rewriting_measurements() {
    let fixture = AdmissionSlateFixture::new();
    let stored = fixture.prepare(RevealedRule::StorageOrder, false).unwrap();
    let sorted = fixture
        .prepare(RevealedRule::SurrogateFitness, false)
        .unwrap();
    assert_eq!(stored.revealed_pool_indices(), [0, 1, 2, 3]);
    assert_eq!(stored.canonical_inputs().revealed_fitness, [5., 4., 3., 2.]);
    assert_eq!(stored.canonical_inputs().revealed_gates, [1., 0., 1., 0.]);
    assert_eq!(stored.proposal_keys(), sorted.proposal_keys());
    assert_ne!(stored.revealed_keys(), sorted.revealed_keys());
}

#[test]
fn assembly_refuses_partial_or_wrong_exploration_slates() {
    let fixture = AdmissionSlateFixture::new();
    let programs = fixture.programs();
    for (count, explore) in [
        (0, false),
        (7, false),
        (9, false),
        (16, false),
        (8, true),
        (15, true),
    ] {
        assert_eq!(
            ServingAdmissionSlate::prepare(
                &fixture.measured(),
                &programs[..6],
                &programs[6..6 + count],
                &fixture.conditioning,
                RevealedRule::StorageOrder,
                explore
            )
            .unwrap_err(),
            AdmissionError::InvalidSlate
        );
    }
}

#[test]
fn assembly_refuses_unaligned_empty_duplicate_or_already_measured_keys() {
    let fixture = AdmissionSlateFixture::new();
    for case in 0..5 {
        let mut programs = fixture.programs();
        match case {
            0 => programs[0].key = "wrong-measurement",
            1 => programs[6].key = "",
            2 => programs[7].key = programs[6].key,
            3 => programs[6].key = programs[5].key,
            _ => {}
        }
        let pool_len = if case == 4 { 5 } else { 6 };
        assert_eq!(
            ServingAdmissionSlate::prepare(
                &fixture.measured(),
                &programs[..pool_len],
                &programs[6..14],
                &fixture.conditioning,
                RevealedRule::StorageOrder,
                false
            )
            .unwrap_err(),
            AdmissionError::InvalidProgram
        );
    }
}

#[test]
fn assembly_refuses_nonfinite_program_channels_and_preserves_finite_values() {
    for row in [0, 5, 6, 13] {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for projection in [false, true] {
                let mut fixture = AdmissionSlateFixture::new();
                if projection {
                    fixture.projections[row][15] = invalid;
                } else {
                    fixture.features[row][411] = invalid;
                }
                assert_eq!(
                    fixture
                        .prepare(RevealedRule::StorageOrder, false)
                        .unwrap_err(),
                    AdmissionError::InvalidProgram
                );
            }
        }
    }
    let mut fixture = AdmissionSlateFixture::new();
    fixture.features[0][0] = -0.0;
    fixture.projections[6][0] = f32::MAX;
    let slate = fixture.prepare(RevealedRule::StorageOrder, false).unwrap();
    assert_eq!(
        slate.canonical_inputs().prog_412[0].to_bits(),
        (-0.0_f32).to_bits()
    );
    assert_eq!(
        slate.canonical_inputs().proj_16[4 * adapter::PROJ_FEATURES],
        f32::MAX
    );
}

#[test]
fn assembly_refuses_nonfinite_tokens_and_nonbinary_masks() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1., 0.5, 2.] {
        let mut fixture = AdmissionSlateFixture::new();
        fixture.conditioning.mask[9] = invalid;
        assert_eq!(
            fixture
                .prepare(RevealedRule::StorageOrder, false)
                .unwrap_err(),
            AdmissionError::InvalidConditioning
        );
    }
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut fixture = AdmissionSlateFixture::new();
        fixture.conditioning.tokens[9][31] = invalid;
        assert_eq!(
            fixture
                .prepare(RevealedRule::StorageOrder, false)
                .unwrap_err(),
            AdmissionError::InvalidConditioning
        );
    }
}

#[test]
fn assembly_refuses_surrogate_overflow_in_adapter_fp32_conversion() {
    let fixture = AdmissionSlateFixture::new();
    let programs = fixture.programs();
    for overflow in [f64::MAX, -f64::MAX] {
        let mut measured = fixture.measured();
        measured[0].surrogate_fitness = overflow;
        assert_eq!(
            ServingAdmissionSlate::prepare(
                &measured,
                &programs[..6],
                &programs[6..14],
                &fixture.conditioning,
                RevealedRule::StorageOrder,
                false
            )
            .unwrap_err(),
            AdmissionError::InvalidRevealed
        );
    }
}

#[test]
fn assembled_selection_checks_adapter_compatibility_before_rng_use() {
    let fixture = AdmissionSlateFixture::new();
    for explore in [false, true] {
        let prepared = fixture
            .prepare(RevealedRule::StorageOrder, explore)
            .unwrap();
        let mut costs = vec![-1e100; REVEALED];
        costs.extend((0..prepared.proposal_keys().len()).map(|i| -(i as f64)));
        let mut actual = ChaCha12Rng::seed_from_u64(71);
        let mut control = ChaCha12Rng::seed_from_u64(71);
        let selection = prepared.select(&costs, &mut actual);
        if let Err(error) = prepared.inputs() {
            assert!(matches!(error.field, "tokens" | "token_mask"));
            assert!(error.expected > error.actual);
            assert_eq!(selection, Err(AdmissionError::InvalidAdapter));
            assert_eq!(actual.next_u64(), control.next_u64());
            continue;
        }
        let chosen = selection.unwrap();
        assert_eq!(chosen, select_costs(&costs, explore, &mut control).unwrap());
        assert_eq!(actual.next_u64(), control.next_u64());
        assert_eq!(
            prepared.proposal_keys()[chosen[0]],
            if explore { "program-21" } else { "program-13" }
        );
        let mut untouched = ChaCha12Rng::seed_from_u64(71);
        assert_eq!(
            prepared.select(&costs[..costs.len() - 1], &mut untouched),
            Err(AdmissionError::InvalidSlate)
        );
        assert_eq!(
            untouched.next_u64(),
            ChaCha12Rng::seed_from_u64(71).next_u64()
        );
    }
}

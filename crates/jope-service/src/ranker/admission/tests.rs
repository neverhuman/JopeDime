use super::*;
use jope_api::Severity;
use rand_chacha::rand_core::SeedableRng;

fn rng(seed: u64) -> ChaCha12Rng {
    ChaCha12Rng::seed_from_u64(seed)
}

fn slate(costs: &[f64]) -> Vec<f64> {
    [vec![-1e100; REVEALED], costs.to_vec()].concat()
}

#[test]
fn normal_selection_excludes_revealed_and_preserves_rng() {
    let costs = slate(&[8., 7., 6., 5., 4., 3., 2., 1.]);
    let mut actual = rng(7);
    assert_eq!(select_costs(&costs, false, &mut actual), Ok([7, 6, 5, 4]));
    assert_eq!(actual.next_u64(), rng(7).next_u64());
}

#[test]
fn ties_and_signed_zero_keep_proposal_order() {
    let costs = slate(&[1., -0., 0., -0., 0., 1., 1., 1.]);
    assert_eq!(select_costs(&costs, false, &mut rng(0)), Ok([1, 2, 3, 4]));
    assert_eq!(
        select_costs(&slate(&[0.; 8]), false, &mut rng(0)),
        Ok([0, 1, 2, 3])
    );
}

#[test]
fn exploration_has_two_best_and_two_distinct_random_choices() {
    let costs = slate(&(0..16).map(|v| -(v as f64)).collect::<Vec<_>>());
    let mut sampled = BTreeSet::<usize>::new();
    for seed in 0..100 {
        let mut actual = rng(seed);
        let chosen = select_costs(&costs, true, &mut actual).unwrap();
        assert_eq!(chosen[..2], [15, 14]);
        assert_eq!(chosen.into_iter().collect::<BTreeSet<_>>().len(), 4);
        assert!(chosen[2..].iter().all(|index| *index < 14));
        sampled.extend(chosen[2..].iter().copied());
        let mut replay = rng(seed);
        assert_eq!(select_costs(&costs, true, &mut replay).unwrap(), chosen);
        assert_eq!(actual.next_u64(), replay.next_u64());
    }
    assert_eq!(sampled, (0..14).collect());
}

#[test]
fn selection_refuses_shapes_and_all_nonfinite_positions_before_rng_use() {
    for len in [0, 4, 8, 11, 13, 19, 21] {
        let mut actual = rng(12);
        assert_eq!(
            select_costs(&vec![0.; len], false, &mut actual),
            Err(AdmissionError::InvalidSlate)
        );
        assert_eq!(actual.next_u64(), rng(12).next_u64());
    }
    for explore in [false, true] {
        let len = if explore { 20 } else { 12 };
        for index in 0..len {
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut costs = vec![0.; len];
                costs[index] = invalid;
                let mut actual = rng(12);
                assert_eq!(
                    select_costs(&costs, explore, &mut actual),
                    Err(AdmissionError::NonFiniteCost)
                );
                assert_eq!(actual.next_u64(), rng(12).next_u64());
            }
        }
    }
    assert_eq!(
        select_costs(&[0.; 12], true, &mut rng(0)),
        Err(AdmissionError::InvalidSlate)
    );
    assert_eq!(
        select_costs(&[0.; 20], false, &mut rng(0)),
        Err(AdmissionError::InvalidSlate)
    );
}

fn measurements() -> Vec<MeasuredAlgorithm<'static>> {
    ["a", "b", "c", "d", "e", "f"]
        .into_iter()
        .enumerate()
        .map(|(i, id)| MeasuredAlgorithm {
            id,
            surrogate_fitness: 5. - i as f64,
            gate: i % 2 == 0,
        })
        .collect()
}

#[test]
fn reveal_choice_is_explicit_and_keeps_original_measurements_paired() {
    let measured = measurements();
    assert_eq!(
        select_revealed(&measured, RevealedRule::StorageOrder),
        Ok([0, 1, 2, 3])
    );
    let sorted = select_revealed(&measured, RevealedRule::SurrogateFitness).unwrap();
    assert_eq!(sorted, [5, 4, 3, 2]);
    assert_eq!(sorted.map(|i| measured[i].gate), [false, true, false, true]);
    assert_eq!(
        sorted.map(|i| measured[i].surrogate_fitness),
        [0., 1., 2., 3.]
    );
    let mut tied = measured;
    tied.iter_mut().for_each(|v| v.surrogate_fitness = 0.);
    assert_eq!(
        select_revealed(&tied, RevealedRule::SurrogateFitness),
        Ok([0, 1, 2, 3])
    );
}

#[test]
fn reveal_refuses_short_large_duplicate_empty_and_nonfinite_pools() {
    let ids: Vec<_> = (0..ELITE_CAPACITY)
        .map(|i| format!("algorithm-{i}"))
        .collect();
    let full: Vec<_> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| MeasuredAlgorithm {
            id,
            surrogate_fitness: i as f64,
            gate: true,
        })
        .collect();
    assert_eq!(
        select_revealed(&full, RevealedRule::SurrogateFitness),
        Ok([0, 1, 2, 3])
    );
    for rule in [RevealedRule::StorageOrder, RevealedRule::SurrogateFitness] {
        assert_eq!(
            select_revealed(&[], rule),
            Err(AdmissionError::InvalidRevealed)
        );
        let mut measured = measurements();
        assert_eq!(
            select_revealed(&measured[..3], rule),
            Err(AdmissionError::InvalidRevealed)
        );
        measured[5].id = measured[0].id;
        assert_eq!(
            select_revealed(&measured, rule),
            Err(AdmissionError::InvalidRevealed)
        );
        measured[5].id = "";
        assert_eq!(
            select_revealed(&measured, rule),
            Err(AdmissionError::InvalidRevealed)
        );
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut measured = measurements();
            measured[5].surrogate_fitness = invalid;
            assert_eq!(
                select_revealed(&measured, rule),
                Err(AdmissionError::InvalidRevealed)
            );
        }
        assert_eq!(
            select_revealed(&vec![measurements()[0]; 49], rule),
            Err(AdmissionError::InvalidRevealed)
        );
    }
}

#[test]
fn unqualified_auto_is_heuristic_with_catalogued_warning_and_real_reason() {
    for reason in [
        GuidanceUnavailableReason::Integrity,
        GuidanceUnavailableReason::Parity,
        GuidanceUnavailableReason::Quality,
        GuidanceUnavailableReason::Capability,
        GuidanceUnavailableReason::RevealedDecision,
    ] {
        let (plan, warnings) =
            resolve_unqualified(Guidance::Auto, LandmarkMode::Full, reason).unwrap();
        assert_eq!(plan.value, Guidance::Heuristic);
        assert!(!plan.prior_only);
        assert_eq!(plan.reason, reason.message());
        assert_eq!(
            warnings.len(),
            if reason == GuidanceUnavailableReason::Integrity {
                2
            } else {
                1
            }
        );
        assert_eq!(warnings[0].code, "GUIDANCE_HEURISTIC");
        assert_eq!(warnings[0].severity, Severity::Info);
        assert!(!warnings[0].host);
        assert_eq!(warnings[0].message, plan.reason);
        if reason == GuidanceUnavailableReason::Integrity {
            assert_eq!(warnings[1].code, "MODEL_INTEGRITY_FAILED");
            assert_eq!(warnings[1].severity, Severity::Error);
        }
    }
}

#[test]
fn learned_is_refused_and_progressive_remains_unavailable_for_every_request() {
    assert_eq!(
        resolve_unqualified(
            Guidance::Learned,
            LandmarkMode::Full,
            GuidanceUnavailableReason::Integrity
        )
        .unwrap_err(),
        ErrorCode::ModelIntegrityFailed
    );
    for reason in [
        GuidanceUnavailableReason::Parity,
        GuidanceUnavailableReason::Quality,
        GuidanceUnavailableReason::Capability,
        GuidanceUnavailableReason::RevealedDecision,
    ] {
        assert_eq!(
            resolve_unqualified(Guidance::Learned, LandmarkMode::Full, reason).unwrap_err(),
            ErrorCode::CapabilityUnavailable
        );
    }
    for requested in [Guidance::Auto, Guidance::Learned, Guidance::Heuristic] {
        assert_eq!(
            resolve_unqualified(
                requested,
                LandmarkMode::Progressive,
                GuidanceUnavailableReason::Integrity
            )
            .unwrap_err(),
            ErrorCode::CapabilityUnavailable
        );
    }
    let (plan, warnings) = resolve_unqualified(
        Guidance::Heuristic,
        LandmarkMode::None,
        GuidanceUnavailableReason::Integrity,
    )
    .unwrap();
    assert_eq!(plan.value, Guidance::Heuristic);
    assert!(!plan.prior_only);
    assert!(warnings.is_empty());
}

#[test]
fn auto_without_landmarks_stays_heuristic_for_every_unqualified_reason() {
    for reason in [
        GuidanceUnavailableReason::Integrity,
        GuidanceUnavailableReason::Parity,
        GuidanceUnavailableReason::Quality,
        GuidanceUnavailableReason::Capability,
        GuidanceUnavailableReason::RevealedDecision,
    ] {
        let (plan, warnings) =
            resolve_unqualified(Guidance::Auto, LandmarkMode::None, reason).unwrap();
        assert_eq!(plan.value, Guidance::Heuristic);
        assert!(!plan.prior_only);
        assert_eq!(plan.reason, reason.message());
        assert_eq!(warnings[0].code, "GUIDANCE_HEURISTIC");
        assert_eq!(warnings[0].message, plan.reason);
        assert_eq!(
            warnings.len(),
            if reason == GuidanceUnavailableReason::Integrity {
                2
            } else {
                1
            }
        );
    }
}

#[test]
fn explicit_heuristic_never_claims_learned_qualification_or_emits_fallback_warning() {
    for reason in [
        GuidanceUnavailableReason::Integrity,
        GuidanceUnavailableReason::Parity,
        GuidanceUnavailableReason::Quality,
        GuidanceUnavailableReason::Capability,
        GuidanceUnavailableReason::RevealedDecision,
    ] {
        for landmarks in [LandmarkMode::None, LandmarkMode::Full] {
            let (plan, warnings) =
                resolve_unqualified(Guidance::Heuristic, landmarks, reason).unwrap();
            assert_eq!(plan.value, Guidance::Heuristic);
            assert!(!plan.prior_only);
            assert_eq!(plan.reason, "Heuristic guidance was requested.");
            assert!(warnings.is_empty());
        }
    }
}

struct AdmissionSlateFixture {
    keys: Vec<String>,
    features: Vec<[f32; adapter::PROG_FEATURES]>,
    projections: Vec<[f32; adapter::PROJ_FEATURES]>,
    conditioning: TokenProjection,
}

impl AdmissionSlateFixture {
    fn new() -> Self {
        Self {
            keys: (0..22).map(|i| format!("program-{i}")).collect(),
            features: (0..22)
                .map(|i| [i as f32; adapter::PROG_FEATURES])
                .collect(),
            projections: (0..22)
                .map(|i| [100.0 + i as f32; adapter::PROJ_FEATURES])
                .collect(),
            conditioning: TokenProjection {
                tokens: std::array::from_fn(|i| {
                    std::array::from_fn(|j| (i * 32 + j) as f32 / 1000.0)
                }),
                mask: [1., 1., 1., 0., 1., 1., 0., 1., 0., 1.],
            },
        }
    }

    fn measured(&self) -> Vec<MeasuredAlgorithm<'_>> {
        self.keys[..6]
            .iter()
            .enumerate()
            .map(|(i, key)| MeasuredAlgorithm {
                id: key,
                surrogate_fitness: 5.0 - i as f64,
                gate: i % 2 == 0,
            })
            .collect()
    }

    fn programs(&self) -> Vec<AdmissionProgram<'_>> {
        (0..22)
            .map(|i| AdmissionProgram {
                key: &self.keys[i],
                features412: &self.features[i],
                projection16: &self.projections[i],
            })
            .collect()
    }

    fn prepare(
        &self,
        rule: RevealedRule,
        explore: bool,
    ) -> Result<ServingAdmissionSlate<'_>, AdmissionError> {
        let programs = self.programs();
        ServingAdmissionSlate::prepare(
            &self.measured(),
            &programs[..6],
            &programs[6..if explore { 22 } else { 14 }],
            &self.conditioning,
            rule,
            explore,
        )
    }
}

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
        let costs = slate(
            &(0..prepared.proposal_keys().len())
                .map(|i| -(i as f64))
                .collect::<Vec<_>>(),
        );
        let mut actual = rng(71);
        let mut control = rng(71);
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
        let mut untouched = rng(71);
        assert_eq!(
            prepared.select(&costs[..costs.len() - 1], &mut untouched),
            Err(AdmissionError::InvalidSlate)
        );
        assert_eq!(untouched.next_u64(), rng(71).next_u64());
    }
}

// M5 CPU prep only: this drives the real frozen adapter on a serving-shaped
// 12-candidate slate, but uses synthetic program channels. It is not the
// required RES-07 serving-fixture qualification or disconnected-ranker gate.
#[test]
#[ignore = "requires externally staged, hash-pinned RES-06 CPU bundle"]
fn real_cpu_adapter_admission_is_sensitive_to_program_channels_and_revealed_fitness() {
    use sha2::{Digest, Sha256};
    use std::{env, fs, path::PathBuf};

    const JOPE_SHA256: &str = "ce39c75f37624e64dfbf89e6493adbf74811a5248f9355997161491e93a132ef";

    fn execute(
        adapter: &CpuAdapter,
        fixture: &AdmissionSlateFixture,
        fitness_override: Option<f64>,
    ) -> [usize; ADMITTED] {
        let mut measured = fixture.measured();
        if let Some(fitness) = fitness_override {
            measured[0].surrogate_fitness = fitness;
        }
        let programs = fixture.programs();
        let slate = ServingAdmissionSlate::prepare(
            &measured,
            &programs[..6],
            &programs[6..14],
            &fixture.conditioning,
            RevealedRule::StorageOrder,
            false,
        )
        .unwrap();
        let predicted = adapter.predict_costs(slate.inputs().unwrap()).unwrap();
        let costs: Vec<_> = predicted
            .total_cost
            .iter()
            .map(|&cost| f64::from(cost))
            .collect();
        slate.select(&costs, &mut rng(0x4d35)).unwrap()
    }

    let root = PathBuf::from(
        env::var_os("JOPE_PARITY_BUNDLE_DIR")
            .expect("JOPE_PARITY_BUNDLE_DIR required for the real CPU adapter prep test"),
    );
    let bytes = fs::read(root.join("jope.safetensors")).expect("read frozen CPU weights");
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        JOPE_SHA256,
        "frozen CPU adapter identity"
    );
    let adapter =
        CpuAdapter::from_bytes_on_device(&bytes, &candle_core_0_11_0::Device::Cpu).unwrap();

    let baseline_fixture = AdmissionSlateFixture::new();
    let baseline = execute(&adapter, &baseline_fixture, None);

    let mut program_channels_changed = false;
    'program_channels: for candidate in 6..14 {
        for value in [-10_000.0, -1_000.0, -100.0, 100.0, 1_000.0, 10_000.0] {
            let mut fixture = AdmissionSlateFixture::new();
            fixture.features[candidate].fill(value);
            fixture.projections[candidate].fill(value);
            if execute(&adapter, &fixture, None) != baseline {
                program_channels_changed = true;
                break 'program_channels;
            }
        }
    }
    assert!(
        program_channels_changed,
        "real CPU ranker admitted set did not change under deterministic program-channel perturbations"
    );

    let mut fitness_changed = false;
    for fitness in [-10_000.0, -1_000.0, -100.0, 100.0, 1_000.0, 10_000.0] {
        if execute(&adapter, &baseline_fixture, Some(fitness)) != baseline {
            fitness_changed = true;
            break;
        }
    }
    assert!(
        fitness_changed,
        "real CPU ranker admission did not change under deterministic revealed-fitness perturbations"
    );
}

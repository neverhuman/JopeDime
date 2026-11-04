use super::{
    hashed, output_oracle_matches, Fixtures, Provider, Stats, BUNDLE, JOPE_SHA, R882_JOPE_SHA,
};
use crate::discover::r#loop::{Plan, RoundAdmission, RoundContext, RoundMeasurement};
use crate::ranker::admission::{select_costs, CpuRoundAdmission, RevealedRule};
#[cfg(feature = "research")]
use crate::ranker::stage1b::TokenProjection;
use rand_chacha::{
    rand_core::{RngCore, SeedableRng},
    ChaCha12Rng,
};
use serde_json::{json, Value};
#[cfg(feature = "research")]
use sha2::{Digest, Sha256};
use std::{cell::Cell, env, fs, path::PathBuf};

pub(crate) fn round_admission_fixture(causal_measurements_only: bool) {
    let fixtures = Fixtures::open();
    let spec = &fixtures.index["files"]["programs.jsonl"];
    let bytes = hashed(
        &fixtures.root.join("programs.jsonl"),
        spec["sha256"].as_str().unwrap(),
    );
    let dsls: Vec<String> = std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .map(|line| {
            serde_json::from_str::<Value>(line).unwrap()["dsl"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .take(20)
        .collect();
    assert_eq!(dsls.len(), 20);
    let plan = Plan {
        run_id: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
        dataset_id: "ds_21fbfe8b53dbcd605d79486b".parse().unwrap(),
        problem_digest: "a".repeat(64),
        bundle_digest: if causal_measurements_only {
            super::super::super::cuda::release::R882
                .strip_prefix("sha256:")
                .unwrap()
                .into()
        } else {
            BUNDLE.strip_prefix("sha256:").unwrap().into()
        },
        binary_version: "RNK10-real-CPU-replay".into(),
        matrix_path: "FIT-only-not-opened-by-replay".into(),
        matrix_sha256: "b".repeat(64),
        target: "target".into(),
        features: vec![],
        objective: "rmse".into(),
        seed: 73,
        planned_workers: 8,
        effort: jope_api::Effort::Fast,
        evaluations: 24,
        wall_seconds: 60,
        evaluation_wall_ms: 15_000,
        evaluation_rss_bytes: 2 * 1024 * 1024 * 1024,
        groups: None,
        times: None,
        gap_rows: 0,
    };
    let mut stats = Stats::default();
    #[cfg(feature = "research")]
    let (mut recorded_slates, mut recording_refusals) = (0, 0);
    let (mut decisive, mut ties, mut mismatches) = (0, 0, 0);
    let (mut landmark_changes, mut fitness_changes, mut disconnected_changes) = (0, 0, 0);
    let (
        mut serving_slates,
        mut serving_landmark_changes,
        mut serving_fitness_changes,
        mut serving_disconnected_changes,
    ) = (0, 0, 0, 0);
    let mut max_kl = 0.0_f64;
    let mut checked_cost_values = 0;
    let mut invalid_cost_values = 0;
    let weights = if causal_measurements_only {
        None
    } else {
        Some(fixtures.weights())
    };
    let causal_bundle = if causal_measurements_only {
        let root = PathBuf::from(
            env::var_os("JOPE_RNK10_SIGNED_BUNDLE_DIR")
                .expect("M5 requires an externally staged signed r882 bundle"),
        );
        let verified = super::super::super::cuda::release::load_r882_release(&root).unwrap();
        assert_eq!(
            verified.digest().to_string(),
            super::super::super::cuda::release::R882
        );
        Some(verified)
    } else {
        None
    };
    let actual_bundle_digest = causal_bundle
        .as_ref()
        .map_or(BUNDLE, |_| super::super::super::cuda::release::R882);
    let actual_jope_sha = causal_bundle.as_ref().map_or(JOPE_SHA, |_| R882_JOPE_SHA);
    let compare_reference_outputs = output_oracle_matches(
        fixtures.index["bundle_digest"].as_str().unwrap(),
        JOPE_SHA,
        actual_bundle_digest,
        actual_jope_sha,
    );
    if causal_measurements_only {
        assert!(!compare_reference_outputs);
        let root = PathBuf::from(env::var_os("JOPE_RNK10_SIGNED_BUNDLE_DIR").unwrap());
        hashed(&root.join("jope.safetensors"), R882_JOPE_SHA);
    }
    for cid in &fixtures.contexts {
        for width in [12, 20] {
            let f = fixtures.load(&format!("slates/{cid}-{width}.safetensors"));
            assert_eq!(f.i64("input.revealed_indices"), vec![0, 1, 2, 3]);
            let fitness = f.f32("input.revealed_raw_f");
            let mode = Cell::new(0);
            let provider = Provider {
                dsls: &dsls,
                prog: f.f32("input.prog_412"),
                proj: f.f32("input.proj_16"),
                tokens: f.f32("input.mm_tokens_10x32"),
                mask: f.f32("input.mm_mask_10"),
                gates: f.f32("input.revealed_gate"),
                mode: &mode,
            };
            let adapter = match &causal_bundle {
                Some(bundle) => super::super::CpuAdapter::from_bundle(bundle).unwrap(),
                None => super::super::CpuAdapter::from_bytes(weights.as_ref().unwrap()).unwrap(),
            };
            let admission = CpuRoundAdmission::for_cpu_replay(
                adapter,
                plan.clone(),
                provider,
                RevealedRule::StorageOrder,
            );
            let ids: Vec<_> = (0..4).map(|i| format!("alg_{i:08}")).collect();
            let mut context = RoundContext {
                plan: &plan,
                round: 1,
                measured: (0..4)
                    .map(|i| RoundMeasurement {
                        algorithm_id: &ids[i],
                        dsl: &dsls[i],
                        surrogate_fitness: f64::from(fitness[i]),
                    })
                    .collect(),
            };
            let proposals = dsls[4..width].to_vec();
            let rng = ChaCha12Rng::from_seed([42; 32]);
            let mut actual_rng = rng.clone();
            let actual = admission
                .evaluate_round(&context, &proposals, width == 20, &mut actual_rng)
                .unwrap();
            if compare_reference_outputs {
                let expected = f.numbers("output.total_cost");
                stats.compare(&actual.costs.total_cost, &expected);
                let mut reference_rng = rng.clone();
                let indices = select_costs(&expected, width == 20, &mut reference_rng).unwrap();
                let reference: Vec<_> = indices.iter().map(|&i| proposals[i].clone()).collect();
                let mut order: Vec<_> = expected[4..].to_vec();
                order.sort_by(f64::total_cmp);
                let cutoff = if width == 12 { 4 } else { 2 };
                let (a, b) = (order[cutoff - 1], order[cutoff]);
                if b - a > 2.0 * (1e-5 + 1e-4 * a.abs().max(b.abs())) {
                    decisive += 1;
                    mismatches += usize::from(actual.admitted != reference);
                } else {
                    ties += 1;
                }
            } else {
                checked_cost_values += actual.costs.total_cost.len();
                invalid_cost_values += actual
                    .costs
                    .total_cost
                    .iter()
                    .filter(|value| !value.is_finite())
                    .count();
            }
            max_kl = max_kl.max(actual.costs.proposal_kl_to_anchor[0]);
            let actual_costs: Vec<_> = actual
                .costs
                .total_cost
                .iter()
                .map(|&value| f64::from(value))
                .collect();
            let mut selection_rng = rng.clone();
            let selected = select_costs(&actual_costs, width == 20, &mut selection_rng).unwrap();
            assert_eq!(actual.admitted_indices, selected);
            assert_eq!(
                actual_rng.next_u64(),
                selection_rng.next_u64(),
                "same round RNG consumption"
            );
            disconnected_changes += usize::from(actual.admitted != proposals[..4]);
            if width == 12 {
                serving_slates += 1;
                serving_disconnected_changes += usize::from(actual.admitted != proposals[..4]);
                for intervention in 5..=7 {
                    mode.set(intervention);
                    let landmark_only = admission
                        .evaluate_round(&context, &proposals, false, &mut rng.clone())
                        .unwrap();
                    if landmark_only.admitted != actual.admitted {
                        serving_landmark_changes += 1;
                        break;
                    }
                }
            }
            mode.set(1);
            let changed = admission
                .evaluate_round(&context, &proposals, width == 20, &mut rng.clone())
                .unwrap();
            landmark_changes += usize::from(changed.admitted != actual.admitted);
            mode.set(0);
            for (i, m) in context.measured.iter_mut().enumerate() {
                m.surrogate_fitness = f64::from(fitness[3 - i]);
            }
            let changed = admission
                .evaluate_round(&context, &proposals, width == 20, &mut rng.clone())
                .unwrap();
            fitness_changes += usize::from(changed.admitted != actual.admitted);
            if width == 12 {
                serving_fitness_changes += usize::from(changed.admitted != actual.admitted);
            }
            for (i, m) in context.measured.iter_mut().enumerate() {
                m.surrogate_fitness = f64::from(fitness[i]);
            }
            assert_eq!(
                admission
                    .admit_round(&context, proposals.clone(), width == 20, &mut rng.clone())
                    .unwrap(),
                actual.admitted
            );
            #[cfg(feature = "research")]
            {
                use crate::discover::r#loop::RecordedAdmissionContext;
                use crate::research::{
                    ObservedSlate, RecordedPolicyKind, RecordedProgram, RecordedReveal,
                };
                let prog = f.f32("input.prog_412");
                let proj = f.f32("input.proj_16");
                let gates = f.f32("input.revealed_gate");
                let tokens = f.f32("input.mm_tokens_10x32");
                let mask = f.f32("input.mm_mask_10");
                let program = |i: usize| RecordedProgram {
                    program_sha256: format!("{:x}", Sha256::digest(dsls[i].as_bytes())),
                    // Record metadata is a software control, not an Engine evidence grant.
                    operator_family: "fixture_observation".into(),
                    feat412: prog[i * 412..(i + 1) * 412].to_vec(),
                    proj16: proj[i * 16..(i + 1) * 16]
                        .iter()
                        .map(|&v| f64::from(v))
                        .collect(),
                };
                let observed = ObservedSlate {
                    programs: (4..width).map(program).collect(),
                    revealed: (0..4)
                        .map(|i| RecordedReveal {
                            program: program(i),
                            surrogate: f64::from(fitness[i]),
                            gate: gates[i] == 1.0,
                        })
                        .collect(),
                };
                let conditioning = TokenProjection {
                    tokens: std::array::from_fn(|i| {
                        tokens[i * 32..(i + 1) * 32].try_into().unwrap()
                    }),
                    mask: mask.as_slice().try_into().unwrap(),
                };
                let mut recorded_rng = rng.clone();
                let policy = admission
                    .admit_recorded_round(
                        &context,
                        proposals.clone(),
                        width == 20,
                        &mut recorded_rng,
                        &RecordedAdmissionContext {
                            slate: &observed,
                            conditioning: TokenProjection {
                                tokens: conditioning.tokens,
                                mask: conditioning.mask,
                            },
                        },
                    )
                    .unwrap();
                assert_eq!(policy.kind, RecordedPolicyKind::Learned);
                assert_eq!(policy.explore, width == 20);
                assert_eq!(policy.admitted, actual.admitted_indices);
                let actual_costs: Vec<_> = actual
                    .costs
                    .total_cost
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect();
                assert_eq!(policy.slate_costs.as_ref().unwrap(), &actual_costs);
                let mut expected_rng = rng.clone();
                select_costs(&actual_costs, width == 20, &mut expected_rng).unwrap();
                assert_eq!(recorded_rng.next_u64(), expected_rng.next_u64());
                recorded_slates += 1;
                for bad in 0..15 {
                    let mut changed = ObservedSlate {
                        programs: observed.programs.clone(),
                        revealed: observed.revealed.clone(),
                    };
                    let mut changed_conditioning = TokenProjection {
                        tokens: conditioning.tokens,
                        mask: conditioning.mask,
                    };
                    match bad {
                        0 => changed.programs[0].program_sha256 = "c".repeat(64),
                        1 => changed.programs[0].feat412[0] += 1.0,
                        2 => changed.programs[0].proj16[0] += 1.0,
                        3 => changed.revealed.swap(0, 1),
                        // One f64 ULP can survive as the same graph f32: raw recorded
                        // fitness still must equal the committed measurement exactly.
                        4 => {
                            changed.revealed[0].surrogate =
                                f64::from_bits(changed.revealed[0].surrogate.to_bits() ^ 1)
                        }
                        5 => changed.revealed[0].gate = !changed.revealed[0].gate,
                        6 => changed_conditioning.tokens[0][0] += 1.0,
                        7 => changed_conditioning.mask[0] = 1.0 - changed_conditioning.mask[0],
                        8 => {
                            changed.revealed.pop();
                        }
                        9 => {
                            changed.programs.pop();
                        }
                        10 => changed.revealed[0].program.feat412[0] += 1.0,
                        11 => changed.revealed[0].program.proj16[0] += 1.0,
                        12 => changed.revealed[0].program.program_sha256 = "d".repeat(64),
                        13 => changed.programs[0].proj16[0] = f64::INFINITY,
                        _ => changed_conditioning.mask[0] = f32::NAN,
                    }
                    let mut refused_rng = rng.clone();
                    assert!(
                        admission
                            .admit_recorded_round(
                                &context,
                                proposals.clone(),
                                width == 20,
                                &mut refused_rng,
                                &RecordedAdmissionContext {
                                    slate: &changed,
                                    conditioning: changed_conditioning
                                },
                            )
                            .is_err(),
                        "accepted different recorded graph inputs, case {bad}"
                    );
                    assert_eq!(refused_rng.next_u64(), rng.clone().next_u64());
                    recording_refusals += 1;
                }
            }
            // Alignment and identity failures must not advance the caller's RNG.
            for bad in [2, 3, 4] {
                mode.set(bad);
                let mut refused_rng = rng.clone();
                assert!(admission
                    .evaluate_round(&context, &proposals, width == 20, &mut refused_rng)
                    .is_err());
                assert_eq!(refused_rng.next_u64(), rng.clone().next_u64());
            }
            mode.set(0);
            for field in 0..6 {
                let mut wrong = plan.clone();
                match field {
                    0 => wrong.problem_digest = "c".repeat(64),
                    1 => wrong.bundle_digest = "d".repeat(64),
                    2 => wrong.matrix_sha256 = "e".repeat(64),
                    3 => wrong.binary_version.push('x'),
                    4 => wrong.objective = "mae".into(),
                    _ => wrong.seed += 1,
                }
                let wrong_context = RoundContext {
                    plan: &wrong,
                    round: context.round,
                    measured: context
                        .measured
                        .iter()
                        .map(|m| RoundMeasurement {
                            algorithm_id: m.algorithm_id,
                            dsl: m.dsl,
                            surrogate_fitness: m.surrogate_fitness,
                        })
                        .collect(),
                };
                let mut refused_rng = rng.clone();
                assert!(admission
                    .evaluate_round(&wrong_context, &proposals, width == 20, &mut refused_rng)
                    .is_err());
                assert_eq!(refused_rng.next_u64(), rng.clone().next_u64());
            }
        }
    }
    if causal_measurements_only {
        use std::io::Write;
        assert_eq!(serving_slates, 50, "M5 requires every serving 4+8 fixture");
        assert_eq!(
            checked_cost_values + stats.values,
            1600,
            "M5 requires both complete slate widths"
        );
        let passed = invalid_cost_values == 0
            && (!compare_reference_outputs || mismatches == 0)
            && serving_landmark_changes > 0
            && serving_fitness_changes > 0
            && serving_disconnected_changes > 0;
        let report = json!({
            "schema":"jope.m5-causal/v3","consumer":"M5-causal","device":"cpu",
            "passed":passed,
            "output_checks":{"finite_cost_values":checked_cost_values,"invalid_cost_values":invalid_cost_values},
            "reference_oracle":{"status":"not_comparable_model_weights","comparison_performed":false,
                "reference_bundle_digest":fixtures.index["bundle_digest"],"reference_jope_weights_sha256":JOPE_SHA,
                "actual_bundle_digest":actual_bundle_digest,"actual_jope_weights_sha256":actual_jope_sha},
            "details":{"contexts":50,"slates":100,
                "admitted_order_comparison": if compare_reference_outputs { "same_model_oracle" } else { "not_comparable_model_weights" },
                "admitted_order_mismatches": if compare_reference_outputs { Some(mismatches) } else { None },
                "quality_gate_granted":false,"Q1_ready":false,
                "serving_4_plus_8":{"slates":serving_slates,
                    "landmark_only_changed_admission":serving_landmark_changes,
                    "fitness_changed_admission":serving_fitness_changes,
                    "disconnected_heuristic_mismatches":serving_disconnected_changes,
                    "landmark_slots":[0,1,7,9],
                    "intervention_kind":"zero_mean_additive_landmark_delta",
                    "interventions":[-6.0,0.0,6.0],
                    "masks_and_dope_only_slots_preserved":true}}
        });
        let root = PathBuf::from(
            env::var_os("JOPE_PARITY_REPORT_DIR").expect("M5 report directory required"),
        );
        fs::create_dir_all(&root).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join("M5-causal.json"))
            .unwrap()
            .write_all(&serde_json::to_vec_pretty(&report).unwrap())
            .unwrap();
        println!("{report}");
        return;
    }
    assert!(decisive > 0, "no decisive reference admission comparisons");
    assert!(landmark_changes > 0, "disconnected landmark control");
    assert!(fitness_changes > 0, "disconnected measured-fitness control");
    assert!(
        disconnected_changes > 0,
        "heuristic disconnected control must fail"
    );
    #[cfg(feature = "research")]
    let recorded_summary = json!({"executed":true,"slates":recorded_slates,"input_mismatch_refusals":recording_refusals,"same_costs_indices_and_RNG":true});
    #[cfg(not(feature = "research"))]
    let recorded_summary = json!({"executed":false,"reason":"research feature required"});
    stats.failures += mismatches;
    stats.finish("RNK-10-composition", json!({
        "contexts":50,"slates":100,"recorded_admission":recorded_summary,"decisive_admission_slates":decisive,"tie_band_slates":ties,
        "admitted_order_mismatches":mismatches,"landmark_perturbation_changed_admission":landmark_changes,
        "fitness_perturbation_changed_admission":fitness_changes,"disconnected_heuristic_mismatches":disconnected_changes,
        "serving_4_plus_8":{"slates":serving_slates,"landmark_only_changed_admission":serving_landmark_changes,
            "fitness_changed_admission":serving_fitness_changes,"disconnected_heuristic_mismatches":serving_disconnected_changes,
            "landmark_slots":[0,1,7,9],"masks_and_dope_only_slots_preserved":true},
        "max_proposal_KL":max_kl,"quality_gate_granted":false,"Q1_ready":false,
        "reveal_rule":"explicit StorageOrder matching captured first four; not an OD13 grant",
        "identity_scope":"packet and weights hash-pinned; dummy Plan used only to exercise binding refusals"
    }));
}

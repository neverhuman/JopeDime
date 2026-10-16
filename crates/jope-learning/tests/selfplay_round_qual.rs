use std::collections::BTreeMap;
use std::process::Command;

use jope_learning::canon::sha256_hex;
use jope_learning::clover::round_qual::{
    qualify_rounds, Endpoint, ParityRecord, RoundQualInputs, RoundQualPlan, SourceGrade,
    RATCHET_SCHEMA, ROUND_QUAL_SCHEMA,
};
use jope_learning::clover::{Budget, ExperimentIdentity, InitialState, SourceIdentity, SCHEMA};
use jope_learning::pools::bootstrap::Verdict;
use jope_pipeline::test_score::{ScoredDataset, TestScore, TEST_SCORE_SCHEMA_ID};
use serde_json::{json, Value};

fn fixture_identity() -> ExperimentIdentity {
    let hex64 = |b: u8| std::iter::repeat_n(b as char, 64).collect::<String>();
    let mut guards = BTreeMap::new();
    guards.insert("quantile".to_owned(), hex64(b'1'));
    guards.insert("label_visibility".to_owned(), hex64(b'2'));
    guards.insert("hold_recovery".to_owned(), hex64(b'3'));
    ExperimentIdentity {
        schema: SCHEMA.to_owned(),
        source: SourceIdentity {
            commit: "525be65b38540eb64b84c9bf36f7da6812812f2f".to_owned(),
            closure_sha256: hex64(b'a'),
        },
        initial_state: InitialState::Checkpoint {
            parameters_sha256: hex64(b'b'),
            optimizer_sha256: hex64(b'c'),
            rng_state_sha256: hex64(b'd'),
        },
        evaluator_sha256: hex64(b'e'),
        parity_contract_sha256: hex64(b'f'),
        support_sha256: hex64(b'4'),
        objective_sha256: hex64(b'5'),
        data_roster_sha256: hex64(b'6'),
        seed: 20_261_016,
        budget: Budget {
            attempts: 64,
            wall_seconds: 120,
            cpu_seconds: 240,
            gpu_seconds: 0,
            memory_bytes: 1_073_741_824,
            daily_bytes: 1_000_000,
            daily_puts: 1_000,
        },
        guards,
    }
}

fn fixture_plan(identity: &ExperimentIdentity) -> RoundQualPlan {
    RoundQualPlan {
        experiment_identity_sha256: identity.sha256().expect("valid identity"),
        prereg_sha256: "8".repeat(64),
        source_grade: SourceGrade::Qualified,
        min_parent_clusters: 3,
        practical_margin_log_ratio: 0.05,
        bootstrap_resamples: 1000,
        bootstrap_seed: 20_261_016,
        alpha: 0.05,
        endpoint: Endpoint::FinalVsRound0Incumbent,
        allow_kl_frozen_rounds: false,
    }
}

fn fixture_ratchet(
    context_id: &str,
    round0_genome: &str,
    final_genome: &str,
    ordinary_final_genome: &str,
) -> Value {
    json!({
        "schema": RATCHET_SCHEMA,
        "qualification": "qualified",
        "context_sha256": context_id,
        "equal_resource_comparison": true,
        "kl_guard_evaluated": true,
        "any_round_frozen_by_kl_guard": false,
        "policy_optimizer_updates": 0,
        "rounds": [
            {
                "round": 0,
                "ordinary_best_rmse": 1.0,
                "ordinary_best_q": 0.5,
                "ratchet_best_rmse": 1.0,
                "ratchet_best_q": 0.5,
                "surplus_q_vs_ordinary": 0.0,
                "ratchet_champion_genome": round0_genome,
                "ordinary_champion_genome": round0_genome,
                "equal_attempts_in_round": true,
                "kl_source": "measured_tilt_policy",
                "seed_update_frozen_by_kl_guard": false
            },
            {
                "round": 1,
                "ordinary_best_rmse": 0.92,
                "ordinary_best_q": 0.52,
                "ratchet_best_rmse": 0.68,
                "ratchet_best_q": 0.60,
                "surplus_q_vs_ordinary": 0.08,
                "ratchet_champion_genome": final_genome,
                "ordinary_champion_genome": ordinary_final_genome,
                "equal_attempts_in_round": true,
                "kl_source": "measured_tilt_policy",
                "seed_update_frozen_by_kl_guard": false
            }
        ]
    })
}

fn fixture_test_score(context_id: &str, genome: &str, test_rmse: f64) -> TestScore {
    TestScore {
        schema: TEST_SCORE_SCHEMA_ID.to_owned(),
        tool_version: "0.3.0".to_owned(),
        created_at: "2026-10-16T17:30:00Z".to_owned(),
        split: "test".to_owned(),
        receipt_sha256: "a".repeat(64),
        dataset: ScoredDataset {
            id: context_id.to_owned(),
            dataset_hash: context_id.to_owned(),
            train_sha256: "b".repeat(64),
            test_sha256: "c".repeat(64),
            train_rows: 100,
            test_rows: 50,
        },
        genome: genome.to_owned(),
        program_sha256: sha256_hex(genome.as_bytes()),
        test_rmse,
        test_r2: 0.85,
        validation_rmse: test_rmse * 0.98,
        registry: "fixture-registry".to_owned(),
        earlier_looks: Vec::new(),
        feeds_selection: false,
        seconds: 0.1,
    }
}

type FixtureBundle = (
    ExperimentIdentity,
    RoundQualPlan,
    Vec<Value>,
    Vec<TestScore>,
    Vec<ParityRecord>,
    BTreeMap<String, String>,
);

fn fixture_bundle() -> FixtureBundle {
    let identity = fixture_identity();
    let plan = fixture_plan(&identity);
    let mut parent_map = BTreeMap::new();
    let mut ratchets = Vec::new();
    let mut scores = Vec::new();
    let mut parity = Vec::new();

    for (ctx, parent, ref_rmse, ord_rmse, cand_rmse) in [
        ("ctx1", "parent_a", 1.00, 0.95, 0.72),
        ("ctx2", "parent_b", 1.10, 1.02, 0.78),
        ("ctx3", "parent_c", 0.90, 0.86, 0.64),
        ("ctx4", "parent_c", 1.20, 1.12, 0.82),
    ] {
        parent_map.insert(ctx.to_owned(), parent.to_owned());
        let g0 = format!("g0_{ctx}");
        let gord = format!("gord_{ctx}");
        let g1 = format!("g1_{ctx}");
        ratchets.push(fixture_ratchet(ctx, &g0, &g1, &gord));
        for (g, rmse) in [(&g0, ref_rmse), (&gord, ord_rmse), (&g1, cand_rmse)] {
            scores.push(fixture_test_score(ctx, g, rmse));
            parity.push(ParityRecord {
                dataset_id: ctx.to_owned(),
                genome: g.clone(),
                passed: true,
                failure_reason: None,
            });
        }
    }
    (identity, plan, ratchets, scores, parity, parent_map)
}

#[test]
fn refuses_malformed_inputs_and_identity_mismatch() {
    let (identity, plan, ratchets, scores, parity, parent_map) = fixture_bundle();

    // 1. Identity SHA mismatch
    let mut bad_plan = plan.clone();
    bad_plan.experiment_identity_sha256 = "0".repeat(64);
    assert!(qualify_rounds(RoundQualInputs {
        plan: &bad_plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // 2. Resamples below MIN_RESAMPLES (1000)
    let mut low_resamples = plan.clone();
    low_resamples.bootstrap_resamples = 999;
    assert!(qualify_rounds(RoundQualInputs {
        plan: &low_resamples,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // 3. Wrong ratchet schema
    let mut bad_ratchets = ratchets.clone();
    bad_ratchets[0]["schema"] = json!("clover.selfplay-ratchet/v0");
    assert!(qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &bad_ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // 4. Wrong test score schema
    let mut bad_scores = scores.clone();
    bad_scores[0].schema = "jope.pipeline-test-score/v0".to_owned();
    assert!(qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &bad_scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // 5. Non-finite or non-positive test_rmse
    let mut nan_scores = scores.clone();
    nan_scores[0].test_rmse = f64::NAN;
    assert!(qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &nan_scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    let mut zero_scores = scores.clone();
    zero_scores[0].test_rmse = 0.0;
    assert!(qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &zero_scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    // 6. Context missing from parent_of_context
    let mut incomplete_parents = parent_map.clone();
    incomplete_parents.remove("ctx2");
    assert!(qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &incomplete_parents,
    })
    .is_err());

    // 7. Duplicate (dataset, genome) test score
    let mut dup_scores = scores.clone();
    dup_scores.push(scores[0].clone());
    assert!(qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &dup_scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());
}

#[test]
fn downgrades_scratch_and_reused_development_sources() {
    let (identity, mut plan, mut ratchets, scores, parity, parent_map) = fixture_bundle();
    plan.source_grade = SourceGrade::ReusedDevelopment;
    let rep = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");
    assert!(!rep.qualified_claim);
    assert!(!rep.rubric_satisfied_modulo_sealed_boundary);
    assert!(rep
        .disqualification_reasons
        .contains(&"source_grade_not_qualified".to_owned()));

    // Even when plan.source_grade is Qualified, a ratchet output marked "scratch_only" is flagged.
    plan.source_grade = SourceGrade::Qualified;
    ratchets[0]["qualification"] = json!("scratch_only");
    let rep2 = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");
    assert!(rep2
        .disqualification_reasons
        .contains(&"source_grade_not_qualified".to_owned()));
}

#[test]
fn downgrades_earlier_looks_and_feeds_selection_violations() {
    let (identity, plan, ratchets, mut scores, parity, parent_map) = fixture_bundle();
    scores[0].earlier_looks = vec!["prior_genome_sha".to_owned()];
    scores[2].feeds_selection = true;

    let rep = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");
    assert_eq!(rep.n_earlier_looks_violations, 1);
    assert_eq!(rep.n_feeds_selection_violations, 1);
    assert!(rep
        .disqualification_reasons
        .contains(&"test_split_earlier_looks_non_empty".to_owned()));
    assert!(rep
        .disqualification_reasons
        .contains(&"test_score_feeds_selection".to_owned()));
}

#[test]
fn downgrades_missing_or_failed_parity_without_dropping_r5_failures() {
    let (identity, plan, ratchets, scores, mut parity, parent_map) = fixture_bundle();
    parity[0].passed = false;
    parity[0].failure_reason = Some("cpp17 NaN divergence".to_owned());
    parity.pop();

    let rep = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");
    assert_eq!(rep.n_failed_parity, 1);
    assert_eq!(rep.n_missing_parity, 1);
    assert_eq!(rep.n_complete_paired_contexts, 2);
    assert!(rep
        .disqualification_reasons
        .contains(&"parity_failure".to_owned()));
    assert!(rep
        .disqualification_reasons
        .contains(&"missing_parity_record".to_owned()));
}

#[test]
fn downgrades_unequal_attempts_unequal_resources_and_insufficient_rounds() {
    let (identity, plan, mut ratchets, scores, parity, parent_map) = fixture_bundle();
    ratchets[0]["equal_resource_comparison"] = json!(false);
    ratchets[1]["rounds"][1]["equal_attempts_in_round"] = json!(false);
    ratchets[2]["rounds"]
        .as_array_mut()
        .expect("array")
        .truncate(1);

    let rep = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");
    assert_eq!(rep.n_unequal_resource_contexts, 1);
    assert_eq!(rep.n_unequal_attempt_rounds, 1);
    assert_eq!(rep.n_insufficient_rounds_contexts, 1);
    for expected in [
        "unequal_resource_comparison",
        "unequal_attempts_in_round",
        "insufficient_rounds",
    ] {
        assert!(
            rep.disqualification_reasons.contains(&expected.to_owned()),
            "missing {expected} in {:?}",
            rep.disqualification_reasons
        );
    }
}

#[test]
fn downgrades_unmeasured_kl_and_kl_frozen_rounds_unless_allowed() {
    let (identity, mut plan, mut ratchets, scores, parity, parent_map) = fixture_bundle();
    ratchets[0]["rounds"][1]["seed_update_frozen_by_kl_guard"] = json!(true);
    ratchets[1]["rounds"][1]["kl_source"] = json!("unmeasured_no_policy_log_probs");

    let rep = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");
    assert_eq!(rep.n_kl_frozen_contexts, 1);
    assert_eq!(rep.n_unmeasured_kl_contexts, 1);
    assert!(rep
        .disqualification_reasons
        .contains(&"kl_guard_frozen_round".to_owned()));
    assert!(rep
        .disqualification_reasons
        .contains(&"kl_guard_unmeasured".to_owned()));

    // Allowing KL-frozen rounds removes kl_guard_frozen_round while keeping kl_guard_unmeasured.
    plan.allow_kl_frozen_rounds = true;
    let rep2 = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");
    assert!(!rep2
        .disqualification_reasons
        .contains(&"kl_guard_frozen_round".to_owned()));
    assert!(rep2
        .disqualification_reasons
        .contains(&"kl_guard_unmeasured".to_owned()));
}

#[test]
fn downgrades_insufficient_parent_clusters_and_unmet_practical_margin() {
    let (identity, mut plan, ratchets, scores, parity, parent_map) = fixture_bundle();
    plan.min_parent_clusters = 10;
    plan.practical_margin_log_ratio = 0.80;

    let rep = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("report");
    assert!(!rep.margin_cleared);
    assert!(rep
        .disqualification_reasons
        .contains(&"insufficient_parent_clusters".to_owned()));
    assert!(rep
        .disqualification_reasons
        .contains(&"margin_not_cleared".to_owned()));
}

#[test]
fn ratchet_vs_ordinary_final_round_endpoint_and_cli_create_only() {
    let (identity, mut plan, ratchets, scores, parity, parent_map) = fixture_bundle();
    plan.endpoint = Endpoint::RatchetVsOrdinaryFinalRound;

    let direct = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("direct report");
    assert_eq!(direct.endpoint, Endpoint::RatchetVsOrdinaryFinalRound);
    assert_eq!(direct.verdict, Verdict::Greater);
    assert!(direct.margin_cleared);
    assert!(direct.rubric_satisfied_modulo_sealed_boundary);

    let dir = tempfile::tempdir().expect("tempdir");
    let plan_path = dir.path().join("plan.json");
    let id_path = dir.path().join("identity.json");
    let ratchets_path = dir.path().join("ratchets.json");
    let scores_path = dir.path().join("scores.json");
    let parity_path = dir.path().join("parity.json");
    let parents_path = dir.path().join("parents.json");
    let out_path = dir.path().join("report.json");

    std::fs::write(&plan_path, serde_json::to_vec(&plan).unwrap()).unwrap();
    std::fs::write(&id_path, serde_json::to_vec(&identity).unwrap()).unwrap();
    std::fs::write(&ratchets_path, serde_json::to_vec(&ratchets).unwrap()).unwrap();
    std::fs::write(&scores_path, serde_json::to_vec(&scores).unwrap()).unwrap();
    std::fs::write(&parity_path, serde_json::to_vec(&parity).unwrap()).unwrap();
    std::fs::write(&parents_path, serde_json::to_vec(&parent_map).unwrap()).unwrap();

    let bin = env!("CARGO_BIN_EXE_jope_pools");
    let first = Command::new(bin)
        .args([
            "qualify-rounds",
            "--plan",
            plan_path.to_str().unwrap(),
            "--identity",
            id_path.to_str().unwrap(),
            "--ratchets",
            ratchets_path.to_str().unwrap(),
            "--test-scores",
            scores_path.to_str().unwrap(),
            "--parity",
            parity_path.to_str().unwrap(),
            "--parent-map",
            parents_path.to_str().unwrap(),
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("run jope_pools qualify-rounds");
    assert!(
        first.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    let stdout = String::from_utf8_lossy(&first.stdout);
    let expected_sha = direct.canonical_sha256().unwrap();
    assert!(stdout.contains(&expected_sha), "stdout={stdout}");

    let written: Value = serde_json::from_slice(&std::fs::read(&out_path).unwrap()).unwrap();
    assert_eq!(written["schema"], ROUND_QUAL_SCHEMA);

    // Second invocation against the same --out path must fail closed (create-only).
    let second = Command::new(bin)
        .args([
            "qualify-rounds",
            "--plan",
            plan_path.to_str().unwrap(),
            "--identity",
            id_path.to_str().unwrap(),
            "--ratchets",
            ratchets_path.to_str().unwrap(),
            "--test-scores",
            scores_path.to_str().unwrap(),
            "--parity",
            parity_path.to_str().unwrap(),
            "--parent-map",
            parents_path.to_str().unwrap(),
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("re-run jope_pools qualify-rounds");
    assert!(!second.status.success());
}

//! Integration, security-boundary (`CT3`), adaptive planner, envelope seal, Spot-Hybrid 3-Gate,
//! and CLI smoke tests for `jope-carbon` (Carbon Phase C1b).

use std::process::Command;

use jope_carbon::{
    carbon_https_agent, evaluate_spot_hybrid_gates, plan_carbon_execution, CarbonEffortTier,
    CarbonError, CarbonExecutionMode, CarbonPlanInput, CarbonProductTier, CarbonRole,
    CarbonSealKey, EvalMemoryClass, MonthlyWindowMetrics, SealedCheckpointEnvelope,
    SpotAdvisorRecommendation, SpotAdvisorThresholds, CARBON_DOCS_URL,
};

#[test]
fn ct3_edge_feature_graph_excludes_engine_search_and_neural_crates() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let output = Command::new("cargo")
        .args([
            "tree",
            "--locked",
            "-p",
            "jope-carbon",
            "--no-default-features",
            "--features",
            "edge",
            "--prefix",
            "none",
        ])
        .current_dir(manifest_dir)
        .output()
        .expect("run cargo tree for carbon-edge");
    assert!(
        output.status.success(),
        "cargo tree failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let tree = String::from_utf8(output.stdout).expect("utf-8 cargo tree output");
    for forbidden in [
        "jope-service ",
        "jope-runtime ",
        "feat-math ",
        "candle-core ",
        "hyperion ",
        "starforge ",
    ] {
        assert!(
            !tree.contains(forbidden),
            "CT3 IP isolation violation: `{forbidden}` appeared in `jope-carbon --features edge` graph:\n{tree}"
        );
    }
    assert!(tree.contains("jope-prep "));
    assert!(tree.contains("jope-api "));
    assert!(tree.contains("aws-lc-rs "));
}

#[test]
fn tier_admission_enforces_free_pro_and_enterprise_no_limit() {
    // Free tier admits <= 5,000 rows, <= 64 cols, fast/medium.
    CarbonProductTier::Free
        .validate_admission(5_000, 64, CarbonEffortTier::Medium)
        .unwrap();
    assert_eq!(
        CarbonProductTier::Free.validate_admission(5_001, 16, CarbonEffortTier::Fast),
        Err(CarbonError::RowLimitExceeded {
            tier: "free",
            rows: 5_001,
            limit: 5_000,
        })
    );
    assert_eq!(
        CarbonProductTier::Free.validate_admission(1_000, 65, CarbonEffortTier::Fast),
        Err(CarbonError::FeatureLimitExceeded {
            tier: "free",
            features: 65,
            limit: 64,
        })
    );
    assert_eq!(
        CarbonProductTier::Free.validate_admission(1_000, 16, CarbonEffortTier::High),
        Err(CarbonError::EffortTierNotPermitted {
            product_tier: "free",
            effort: "high",
        })
    );

    // Pro tier admits <= 100,000 rows, <= 512 cols, fast..ultra.
    CarbonProductTier::Pro
        .validate_admission(100_000, 512, CarbonEffortTier::Ultra)
        .unwrap();
    assert!(matches!(
        CarbonProductTier::Pro.validate_admission(100_001, 32, CarbonEffortTier::High),
        Err(CarbonError::RowLimitExceeded { tier: "pro", .. })
    ));

    // Enterprise no_limit() admits arbitrary row and feature counts.
    CarbonProductTier::no_limit()
        .validate_admission(2_500_000, 2_048, CarbonEffortTier::Ultra)
        .unwrap();
}

#[test]
fn adaptive_planner_selects_solo_for_fast_and_hybrid_fanout_for_deep_waves() {
    // Fast run always chooses SoloSingleLambda with zero fan-out shards.
    let fast_plan = plan_carbon_execution(&CarbonPlanInput {
        product_tier: CarbonProductTier::Free,
        effort: CarbonEffortTier::Fast,
        total_rows: 1_200,
        fit_rows: 960,
        features: 16,
        est_ms_per_candidate_1vcpu: 12.0,
    })
    .unwrap();
    assert_eq!(fast_plan.mode, CarbonExecutionMode::SoloSingleLambda);
    assert_eq!(fast_plan.shards_per_wave, 0);
    assert_eq!(fast_plan.eval_class, EvalMemoryClass::Xs512Mb);

    // High effort on a 40,000-row Pro dataset chooses HybridFanOut across right-sized workers.
    let high_plan = plan_carbon_execution(&CarbonPlanInput {
        product_tier: CarbonProductTier::Pro,
        effort: CarbonEffortTier::High,
        total_rows: 40_000,
        fit_rows: 32_000,
        features: 48,
        est_ms_per_candidate_1vcpu: 95.0,
    })
    .unwrap();
    assert_eq!(high_plan.mode, CarbonExecutionMode::HybridFanOut);
    assert_eq!(high_plan.eval_class, EvalMemoryClass::M1769Mb);
    assert!(high_plan.shards_per_wave >= 8);
    assert!(high_plan.hybrid_wall_ms * 2.2 <= high_plan.solo_wall_ms);

    // Multi-gigabyte Enterprise working set exceeding XL (7,076 MB) returns typed refusal directing to `carbon serve`.
    let huge = plan_carbon_execution(&CarbonPlanInput {
        product_tier: CarbonProductTier::no_limit(),
        effort: CarbonEffortTier::Ultra,
        total_rows: 20_000_000,
        fit_rows: 16_000_000,
        features: 64,
        est_ms_per_candidate_1vcpu: 500.0,
    });
    assert!(matches!(
        huge,
        Err(CarbonError::WorkingSetExceedsLambdaClass { .. })
    ));
}

#[test]
fn seal_envelope_authenticates_roundtrip_and_refuses_tamper_or_wrong_context() {
    let key = CarbonSealKey::from_bytes([0x5A; 32]);
    let secret_payload = b"opaque-candidate-archive-wave-3";
    let env =
        SealedCheckpointEnvelope::seal(&key, "tenant_alpha", "run_01J", 3, secret_payload).unwrap();
    assert_eq!(env.wave_index(), 3);
    assert!(!env
        .wire_bytes()
        .windows(secret_payload.len())
        .any(|w| w == secret_payload));

    let (wave, opened) =
        SealedCheckpointEnvelope::open(&key, "tenant_alpha", "run_01J", env.wire_bytes()).unwrap();
    assert_eq!(wave, 3);
    assert_eq!(opened.as_slice(), secret_payload);

    // Cross-tenant or cross-run replay is cryptographically refused.
    assert!(matches!(
        SealedCheckpointEnvelope::open(&key, "tenant_beta", "run_01J", env.wire_bytes()),
        Err(CarbonError::SealAuthenticationFailed { .. })
    ));
    assert!(matches!(
        SealedCheckpointEnvelope::open(&key, "tenant_alpha", "run_other", env.wire_bytes()),
        Err(CarbonError::SealAuthenticationFailed { .. })
    ));

    // Single-bit ciphertext mutation is refused by AES-256-GCM tag verification.
    let mut tampered = env.wire_bytes().to_vec();
    let last = tampered.len() - 1;
    tampered[last] ^= 0x80;
    assert!(matches!(
        SealedCheckpointEnvelope::open(&key, "tenant_alpha", "run_01J", &tampered),
        Err(CarbonError::SealAuthenticationFailed { .. })
    ));
}

#[test]
fn spot_advisor_enforces_three_conjunctive_gates_and_two_month_window() {
    let thresholds = SpotAdvisorThresholds::default();

    // Spiky traffic (P95/mean = 8.0 > 3.5) stays pure Lambda even with high spend.
    let spiky = MonthlyWindowMetrics::for_region("us-east-2", 4_200.0, 20.0, 22.0, 176.0, 0.01);
    let spiky_verdict = evaluate_spot_hybrid_gates(&[spiky, spiky], &thresholds);
    assert!(spiky_verdict.latest_spend_gate_passed);
    assert!(spiky_verdict.latest_duty_cycle_gate_passed);
    assert!(!spiky_verdict.latest_burstiness_gate_passed);
    assert_eq!(
        spiky_verdict.recommendation,
        SpotAdvisorRecommendation::StayPureLambda
    );

    // Moderate spend + elevated cold-start rate triggers Step 1.5 Warm Standby before Spot.
    let cold_heavy = MonthlyWindowMetrics::for_region("eu-west-1", 600.0, 2.0, 4.0, 18.0, 0.09);
    let warm_verdict = evaluate_spot_hybrid_gates(&[cold_heavy], &thresholds);
    assert_eq!(
        warm_verdict.recommendation,
        SpotAdvisorRecommendation::EnableWarmStandbyLambda
    );

    // Sustained flat load passing S-1 ($3,400), S-2 (P20 = 18 vCPUs), and S-3 (48/24 = 2.0x)
    // for 1 month waits, then qualifies on month 2!
    let steady = MonthlyWindowMetrics::for_region("us-east-2", 3_400.0, 18.0, 24.0, 48.0, 0.01);
    assert_eq!(
        evaluate_spot_hybrid_gates(&[steady], &thresholds).recommendation,
        SpotAdvisorRecommendation::StayPureLambda
    );
    let qualified = evaluate_spot_hybrid_gates(&[steady, steady], &thresholds);
    assert_eq!(qualified.consecutive_qualifying_months, 2);
    assert_eq!(
        qualified.recommendation,
        SpotAdvisorRecommendation::EligibleForEnterpriseSpotOverflow
    );
}

#[test]
fn cli_smoke_tls_provider_and_guidance_cover_all_variants() {
    let _agent = carbon_https_agent();
    assert_eq!("edge".parse::<CarbonRole>().unwrap(), CarbonRole::Edge);
    assert_eq!(
        "orchestrate".parse::<CarbonRole>().unwrap(),
        CarbonRole::EngineOrchestrate
    );

    let edge_bin = env!("CARGO_BIN_EXE_carbon-edge");
    let edge_out = Command::new(edge_bin)
        .arg("--healthcheck")
        .output()
        .expect("run carbon-edge --healthcheck");
    assert!(edge_out.status.success());
    let edge_json: serde_json::Value = serde_json::from_slice(&edge_out.stdout).unwrap();
    assert_eq!(edge_json["status"], "ok");
    assert_eq!(edge_json["tls_provider"], "aws_lc_rs");

    let carbon_bin = env!("CARGO_BIN_EXE_carbon");
    let plan_out = Command::new(carbon_bin)
        .args([
            "plan",
            "--tier",
            "pro",
            "--effort",
            "high",
            "--total-rows",
            "20000",
            "--fit-rows",
            "16000",
            "--features",
            "32",
        ])
        .output()
        .expect("run carbon plan");
    assert!(plan_out.status.success());
    let plan_json: serde_json::Value = serde_json::from_slice(&plan_out.stdout).unwrap();
    assert_eq!(plan_json["mode"], "hybrid_fan_out");

    let all_errors = [
        CarbonError::RowLimitExceeded {
            tier: "free",
            rows: 6_000,
            limit: 5_000,
        },
        CarbonError::FeatureLimitExceeded {
            tier: "free",
            features: 80,
            limit: 64,
        },
        CarbonError::EffortTierNotPermitted {
            product_tier: "free",
            effort: "ultra",
        },
        CarbonError::InvalidPlanInput { reason: "bad" },
        CarbonError::WorkingSetExceedsLambdaClass {
            required_mb: 9_000,
            max_lambda_mb: 7_076,
        },
        CarbonError::SealAuthenticationFailed { reason: "tag" },
        CarbonError::InvalidRoleOrCommand {
            value: "bogus".into(),
        },
    ];
    for err in all_errors {
        let g = err.guidance();
        assert!(!g.purpose.is_empty());
        assert_eq!(g.reason, err.to_string());
        assert!(!g.common_fixes.is_empty());
        assert_eq!(g.docs_url, CARBON_DOCS_URL);
        assert!(!g.repair_hint.is_empty());
    }
}

//! Integration, security-boundary (`CT3`), adaptive planner, `CCKP\x02` domain-separated seal,
//! Spot-Hybrid 3-Gate, and CLI smoke tests for `jope-carbon`.

use std::process::Command;

use jope_carbon::{
    carbon_https_agent, evaluate_spot_hybrid_gates, plan_carbon_execution, CarbonEffortTier,
    CarbonError, CarbonExecutionMode, CarbonPlanInput, CarbonProductTier, CarbonRole,
    CarbonSealDomain, CarbonSealKey, EvalMemoryClass, MonthlyWindowMetrics,
    SealedCheckpointEnvelope, SpotAdvisorRecommendation, SpotAdvisorThresholds, CARBON_DOCS_URL,
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

    CarbonProductTier::Pro
        .validate_admission(100_000, 512, CarbonEffortTier::Ultra)
        .unwrap();
    assert!(matches!(
        CarbonProductTier::Pro.validate_admission(100_001, 32, CarbonEffortTier::High),
        Err(CarbonError::RowLimitExceeded { tier: "pro", .. })
    ));

    CarbonProductTier::no_limit()
        .validate_admission(2_500_000, 2_048, CarbonEffortTier::Ultra)
        .unwrap();
}

#[test]
fn adaptive_planner_selects_solo_for_fast_and_hybrid_fanout_for_deep_waves() {
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
fn seal_envelope_authenticates_domain_and_udek_and_refuses_tamper() {
    let key = CarbonSealKey::from_bytes([0x5A; 32]);
    let u_dek = [0x99u8; 32];
    let other_dek = [0xAAu8; 32];
    let secret_payload = b"opaque-candidate-archive-wave-3";
    let env = SealedCheckpointEnvelope::seal_with_domain(
        &key,
        CarbonSealDomain::CheckpointArchive,
        "tenant_alpha",
        "run_01J",
        3,
        Some(&u_dek),
        secret_payload,
    )
    .unwrap();
    assert_eq!(env.wave_index(), 3);
    assert!(!env
        .wire_bytes()
        .windows(secret_payload.len())
        .any(|w| w == secret_payload));

    let (wave, opened) = SealedCheckpointEnvelope::open_with_domain(
        &key,
        CarbonSealDomain::CheckpointArchive,
        "tenant_alpha",
        "run_01J",
        Some(&u_dek),
        env.wire_bytes(),
    )
    .unwrap();
    assert_eq!(wave, 3);
    assert_eq!(opened.as_slice(), secret_payload);

    // Domain swap (`CheckpointArchive` -> `ChampionModel`) MUST fail closed (`RT-06`).
    assert!(matches!(
        SealedCheckpointEnvelope::open_with_domain(
            &key,
            CarbonSealDomain::ChampionModel,
            "tenant_alpha",
            "run_01J",
            Some(&u_dek),
            env.wire_bytes()
        ),
        Err(CarbonError::SealAuthenticationFailed { .. })
    ));

    // Wrong per-user `U_DEK` or missing `U_DEK` MUST fail closed (`RT-06`).
    assert!(matches!(
        SealedCheckpointEnvelope::open_with_domain(
            &key,
            CarbonSealDomain::CheckpointArchive,
            "tenant_alpha",
            "run_01J",
            Some(&other_dek),
            env.wire_bytes()
        ),
        Err(CarbonError::SealAuthenticationFailed { .. })
    ));
    assert!(matches!(
        SealedCheckpointEnvelope::open_with_domain(
            &key,
            CarbonSealDomain::CheckpointArchive,
            "tenant_alpha",
            "run_01J",
            None,
            env.wire_bytes()
        ),
        Err(CarbonError::SealAuthenticationFailed { .. })
    ));

    // Cross-tenant or bit-mutation is cryptographically refused.
    assert!(matches!(
        SealedCheckpointEnvelope::open_with_domain(
            &key,
            CarbonSealDomain::CheckpointArchive,
            "tenant_beta",
            "run_01J",
            Some(&u_dek),
            env.wire_bytes()
        ),
        Err(CarbonError::SealAuthenticationFailed { .. })
    ));
    let mut tampered = env.wire_bytes().to_vec();
    let last = tampered.len() - 1;
    tampered[last] ^= 0x80;
    assert!(matches!(
        SealedCheckpointEnvelope::open_with_domain(
            &key,
            CarbonSealDomain::CheckpointArchive,
            "tenant_alpha",
            "run_01J",
            Some(&u_dek),
            &tampered
        ),
        Err(CarbonError::SealAuthenticationFailed { .. })
    ));
}

#[test]
fn spot_advisor_enforces_three_conjunctive_gates_and_two_month_window() {
    let thresholds = SpotAdvisorThresholds::default();

    let spiky = MonthlyWindowMetrics::for_region("us-east-2", 4_200.0, 20.0, 22.0, 176.0, 0.01);
    let spiky_verdict = evaluate_spot_hybrid_gates(&[spiky, spiky], &thresholds);
    assert!(spiky_verdict.latest_spend_gate_passed);
    assert!(spiky_verdict.latest_duty_cycle_gate_passed);
    assert!(!spiky_verdict.latest_burstiness_gate_passed);
    assert_eq!(
        spiky_verdict.recommendation,
        SpotAdvisorRecommendation::StayPureLambda
    );

    let cold_heavy = MonthlyWindowMetrics::for_region("eu-west-1", 600.0, 2.0, 4.0, 18.0, 0.09);
    let warm_verdict = evaluate_spot_hybrid_gates(&[cold_heavy], &thresholds);
    assert_eq!(
        warm_verdict.recommendation,
        SpotAdvisorRecommendation::EnableWarmStandbyLambda
    );

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
        CarbonError::UserEnvelopeFailed { reason: "cuen" },
        CarbonError::CapabilityAuthFailed { reason: "ctk1" },
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

#[test]
fn cloudformation_cell_template_and_deterministic_packager_invariants() {
    use jope_carbon::{
        build_deterministic_zip_bytes, expand_region_targets, synthesize_cell_template_for_release,
        verify_arm64_lambda_binary, CarbonZipEntry, CANONICAL_17_REGIONS,
    };

    // 1. Synthesis + zero-trust CloudFormation invariants
    let synth_str = synthesize_cell_template_for_release("da0ae4a7c0ffee").unwrap();
    let tpl: serde_json::Value = serde_json::from_str(&synth_str).unwrap();
    let resources = tpl["Resources"].as_object().unwrap();

    assert!(resources.contains_key("EdgeVersionda0ae4a7c0ffee"));
    assert!(resources.contains_key("EngineVersionda0ae4a7c0ffee"));
    assert!(resources.contains_key("EvalVersionda0ae4a7c0ffee"));
    assert!(!resources.contains_key("EdgeVersionCurrent"));

    // EdgeFunctionUrl MUST enforce AWS_IAM on the :live alias and never Principal: "*"
    assert_eq!(
        resources["EdgeFunctionUrl"]["Properties"]["AuthType"],
        "AWS_IAM"
    );
    assert_eq!(
        resources["EdgeFunctionUrl"]["Properties"]["Qualifier"],
        "live"
    );
    assert_ne!(
        resources["EdgeLiveUrlPermission"]["Properties"]["Principal"],
        "*"
    );
    assert_ne!(
        resources["EdgeLiveInvokePermission"]["Properties"]["Principal"],
        "*"
    );

    // Public API Gateway HTTP API v2.0 MUST target only immutable EdgeLiveAlias, enforce PayloadFormatVersion 2.0,
    // configure stage throttling, and scope Lambda permission strictly to EdgePublicHttpApi.
    assert_eq!(
        resources["EdgePublicHttpApi"]["Properties"]["ProtocolType"],
        "HTTP"
    );
    assert_eq!(
        resources["EdgePublicHttpIntegration"]["Properties"]["PayloadFormatVersion"],
        "2.0"
    );
    assert_eq!(
        resources["EdgePublicHttpIntegration"]["Properties"]["IntegrationUri"]["Ref"],
        "EdgeLiveAlias"
    );
    assert_eq!(
        resources["EdgePublicHttpStage"]["Properties"]["AutoDeploy"],
        true
    );
    assert!(
        resources["EdgePublicHttpStage"]["Properties"]["DefaultRouteSettings"]
            .get("ThrottlingRateLimit")
            .is_some()
    );
    assert_eq!(
        resources["EdgePublicApiGatewayInvokePermission"]["Properties"]["Principal"],
        "apigateway.amazonaws.com"
    );
    assert_eq!(
        resources["EdgePublicApiGatewayInvokePermission"]["Properties"]["FunctionName"]["Ref"],
        "EdgeLiveAlias"
    );
    assert!(tpl["Outputs"].get("PublicEdgeEndpointUrl").is_some());

    // Multi-region CloudFormation safety: every IAM RoleName and S3 BucketName MUST embed
    // `${AWS::Region}` so deploying `jopedime-carbon-prod` across multiple regions never collides!
    for (logical_id, res_val) in resources {
        if res_val["Type"] == "AWS::IAM::Role" {
            let role_sub = res_val["Properties"]["RoleName"]["Fn::Sub"]
                .as_str()
                .expect("IAM RoleName must use Fn::Sub");
            assert!(
                role_sub.contains("${AWS::Region}"),
                "IAM Role {logical_id} RoleName '{role_sub}' must include ${{AWS::Region}} for multi-region deploys"
            );
        }
        if res_val["Type"] == "AWS::S3::Bucket" {
            let bkt_sub = res_val["Properties"]["BucketName"]["Fn::Sub"]
                .as_str()
                .expect("S3 BucketName must use Fn::Sub");
            assert!(
                bkt_sub.contains("${AWS::Region}"),
                "S3 Bucket {logical_id} BucketName '{bkt_sub}' must include ${{AWS::Region}}"
            );
        }
    }
    assert!(resources.contains_key("EdgePublicHttp5xxAlarm"));
    assert!(resources.contains_key("EvalThrottleAlarm"));

    // Public clouds discovery manifest resolution + P-256 KEK validation (v1 + v2 multi-region)
    let kek = jope_carbon::CarbonEngineKek::generate().unwrap();
    let kek_west = jope_carbon::CarbonEngineKek::generate().unwrap();
    let mut clouds_map = std::collections::BTreeMap::new();
    clouds_map.insert(
        "dev".to_owned(),
        jope_carbon::CarbonCloudCellEntry {
            environment: "dev".to_owned(),
            region: "us-east-2".to_owned(),
            public_endpoint: "https://example-dev.execute-api.us-east-2.amazonaws.com".to_owned(),
            engine_kek_pubkey_b64: Some(kek.pubkey_b64()),
            release_id: Some("da0ae4a7c0ffee".to_owned()),
            regions: vec![
                jope_carbon::CarbonCloudRegionPin {
                    region: "us-east-2".to_owned(),
                    public_endpoint: "https://example-dev.execute-api.us-east-2.amazonaws.com"
                        .to_owned(),
                    engine_kek_pubkey_b64: Some(kek.pubkey_b64()),
                    release_id: Some("da0ae4a7c0ffee".to_owned()),
                },
                jope_carbon::CarbonCloudRegionPin {
                    region: "us-west-2".to_owned(),
                    public_endpoint: "https://example-dev.execute-api.us-west-2.amazonaws.com"
                        .to_owned(),
                    engine_kek_pubkey_b64: Some(kek_west.pubkey_b64()),
                    release_id: Some("da0ae4a7c0ffee".to_owned()),
                },
            ],
        },
    );
    let clouds_doc = jope_carbon::CarbonPublicCloudsManifest {
        schema: "jope-carbon-clouds-v2".to_owned(),
        generated_epoch_ms: 1794400000000,
        clouds: clouds_map,
    };
    let tmp_dir = tempfile::tempdir().unwrap();
    let manifest_file = tmp_dir.path().join("carbon-clouds.json");
    std::fs::write(&manifest_file, serde_json::to_vec(&clouds_doc).unwrap()).unwrap();
    let resolved_dev =
        jope_carbon::resolve_public_cloud_endpoint("dev", manifest_file.to_str()).unwrap();
    assert_eq!(resolved_dev.region, "us-east-2");
    assert_eq!(resolved_dev.candidate_endpoints().len(), 2);
    assert_eq!(resolved_dev.pinned_keks_by_endpoint().len(), 2);
    assert!(jope_carbon::resolve_public_cloud_endpoint("prod", manifest_file.to_str()).is_err());

    // Deterministic single-binary `.tar.gz` archive check
    let tgz_a =
        jope_carbon::build_deterministic_single_bin_targz("carbon", b"\x7fELFtest").unwrap();
    let tgz_b =
        jope_carbon::build_deterministic_single_bin_targz("carbon", b"\x7fELFtest").unwrap();
    assert_eq!(tgz_a, tgz_b);
    assert!(jope_carbon::build_deterministic_single_bin_targz("../bad", b"x").is_err());

    // R882ModelLayer MUST attach strictly to EngineFunction, never EdgeFunction or EvalFunction
    assert_eq!(
        resources["EngineFunction"]["Properties"]["Timeout"], 900,
        "EngineFunction Timeout must be 900s (15 min) for high/ultra search legs"
    );
    assert!(resources["EngineFunction"]["Properties"]
        .get("Layers")
        .is_some());
    assert!(resources["EdgeFunction"]["Properties"]
        .get("Layers")
        .is_none());
    assert!(resources["EvalFunction"]["Properties"]
        .get("Layers")
        .is_none());

    // Role-separated secrets: EdgeFunction must never receive seal_secret_hex or engine_kek_hex
    let edge_env =
        serde_json::to_string(&resources["EdgeFunction"]["Properties"]["Environment"]).unwrap();
    assert!(!edge_env.contains("seal_secret_hex"));
    assert!(!edge_env.contains("engine_kek_hex"));

    let eval_env =
        serde_json::to_string(&resources["EvalFunction"]["Properties"]["Environment"]).unwrap();
    assert!(!eval_env.contains("auth_secret_hex"));
    assert!(!eval_env.contains("engine_kek_hex"));

    // 2. Deterministic reproducible ZIP archive invariants + traversal rejection
    let zip_a = build_deterministic_zip_bytes(vec![
        CarbonZipEntry {
            rel_path: "b/weights.bin".to_owned(),
            bytes: vec![1, 2, 3, 4, 5],
            unix_mode: 0o100644,
        },
        CarbonZipEntry {
            rel_path: "bootstrap".to_owned(),
            bytes: b"arm64-bootstrap-stub".to_vec(),
            unix_mode: 0o100755,
        },
    ])
    .unwrap();
    let zip_b = build_deterministic_zip_bytes(vec![
        CarbonZipEntry {
            rel_path: "bootstrap".to_owned(),
            bytes: b"arm64-bootstrap-stub".to_vec(),
            unix_mode: 0o100755,
        },
        CarbonZipEntry {
            rel_path: "b/weights.bin".to_owned(),
            bytes: vec![1, 2, 3, 4, 5],
            unix_mode: 0o100644,
        },
    ])
    .unwrap();
    assert_eq!(
        zip_a, zip_b,
        "deterministic ZIP must be order-independent and bit-exact"
    );
    assert!(build_deterministic_zip_bytes(vec![CarbonZipEntry {
        rel_path: "../escape".to_owned(),
        bytes: vec![0],
        unix_mode: 0o100644,
    }])
    .is_err());

    // 3. ARM64 ELF64 + CT3 symbol fence verifier
    let mut fake_arm64_elf = vec![0u8; 64];
    fake_arm64_elf[0..4].copy_from_slice(b"\x7fELF");
    fake_arm64_elf[4] = 2; // ELFCLASS64
    fake_arm64_elf[5] = 1; // ELFDATA2LSB
    fake_arm64_elf[18..20].copy_from_slice(&0x00B7u16.to_le_bytes()); // EM_AARCH64
    verify_arm64_lambda_binary(&fake_arm64_elf, true).unwrap();

    let mut leaked_edge = fake_arm64_elf.clone();
    leaked_edge.extend_from_slice(b"models/bundles/r882");
    assert!(verify_arm64_lambda_binary(&leaked_edge, true).is_err());

    let mut fake_x86_elf = fake_arm64_elf;
    fake_x86_elf[18..20].copy_from_slice(&0x003Eu16.to_le_bytes()); // EM_X86_64
    assert!(verify_arm64_lambda_binary(&fake_x86_elf, false).is_err());

    // 4. Region expansion presets
    assert_eq!(CANONICAL_17_REGIONS.len(), 17);
    assert_eq!(
        expand_region_targets(&["all-17".to_owned()]).unwrap().len(),
        17
    );
    assert_eq!(
        expand_region_targets(&["canary".to_owned()]).unwrap(),
        vec!["us-east-2".to_owned()]
    );
}

#[test]
fn capability_claims_unverified_decode_credentials_store_and_installer_guards() {
    use jope_api::carbon::CarbonTierWire;
    use jope_carbon::{
        load_stored_cloud_token, remove_stored_cloud_token, save_stored_cloud_token,
        CarbonAuthVerifier, CarbonCapabilityClaims,
    };

    let verifier = CarbonAuthVerifier::from_secret([0x3C; 32]);
    let pro_tok = verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: "acme_pro_team".to_owned(),
            tier: CarbonTierWire::Pro,
            exp_epoch_secs: 4_102_444_800,
            admin: false,
        })
        .expect("mint pro token");
    let decoded = CarbonCapabilityClaims::decode_unverified(&pro_tok).expect("decode pro claims");
    assert_eq!(decoded.tenant_id, "acme_pro_team");
    assert_eq!(decoded.tier, CarbonTierWire::Pro);
    assert_eq!(decoded.exp_epoch_secs, 4_102_444_800);
    assert!(!decoded.admin);

    assert!(CarbonCapabilityClaims::decode_unverified("badprefix.abc.def").is_err());
    assert!(CarbonCapabilityClaims::decode_unverified("ctk1.not_b64!.00").is_err());
    let bad_hex = format!("{}.zzzz", pro_tok.rsplit_once('.').unwrap().0);
    assert!(CarbonCapabilityClaims::decode_unverified(&bad_hex).is_err());

    // Verify 0600 credentials store & redacted Debug formatting via isolated subprocess / env check
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg_dir = tmp.path().join("jope-cfg");

    // Verify relative HOME and relative --bin-dir are rejected by deploy/carbon/install.sh
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let installer = repo_root.join("deploy/carbon/install.sh");
    assert!(installer.is_file(), "missing {}", installer.display());

    let rel_home_out = Command::new("sh")
        .arg(&installer)
        .arg("--skip-ping")
        .env("HOME", "relative/home")
        .output()
        .expect("run install.sh with relative HOME");
    assert!(
        !rel_home_out.status.success(),
        "install.sh must reject relative HOME"
    );

    let rel_bin_out = Command::new("sh")
        .arg(&installer)
        .args(["--bin-dir", "rel/bin", "--config-dir"])
        .arg(&cfg_dir)
        .arg("--skip-ping")
        .output()
        .expect("run install.sh with relative --bin-dir");
    assert!(
        !rel_bin_out.status.success(),
        "install.sh must reject relative --bin-dir"
    );

    // Verify install.sh rejects forged/tampered SHA256SUMS.txt.sig before writing $BIN_DIR/carbon
    let fake_src = tmp.path().join("forged-release");
    let bin_dest = tmp.path().join("bin-out");
    std::fs::create_dir_all(&fake_src).unwrap();
    let arch_asset = if cfg!(target_arch = "aarch64") {
        "carbon-linux-aarch64-musl"
    } else {
        "carbon-linux-x86_64-musl"
    };
    std::fs::write(fake_src.join(arch_asset), b"#!/bin/sh\necho forged\n").unwrap();
    std::fs::write(
        fake_src.join("carbon-clouds.json"),
        br#"{"schema":"carbon.clouds/v1","generated_epoch_ms":1,"clouds":{}}"#,
    )
    .unwrap();
    std::fs::write(
        fake_src.join("SHA256SUMS.txt"),
        format!(
            "{}  {}\n{}  carbon-clouds.json\n",
            "0".repeat(64),
            arch_asset,
            "1".repeat(64)
        ),
    )
    .unwrap();
    std::fs::write(
        fake_src.join("SHA256SUMS.txt.sig"),
        "MEUCIQD00000000000000000\n",
    )
    .unwrap();

    let forged_out = Command::new("sh")
        .arg(&installer)
        .arg("--source-dir")
        .arg(&fake_src)
        .arg("--bin-dir")
        .arg(&bin_dest)
        .arg("--config-dir")
        .arg(&cfg_dir)
        .arg("--skip-ping")
        .output()
        .expect("run install.sh with forged signature");
    assert!(
        !forged_out.status.success(),
        "install.sh must fail closed on invalid ECDSA-P256 signature"
    );
    assert!(
        !bin_dest.join("carbon").exists(),
        "install.sh must never write $BIN_DIR/carbon when signature verification fails"
    );

    // Verify CARBON_CONFIG_DIR store lifecycle & 0600 permissions
    let prev_cfg = std::env::var_os("CARBON_CONFIG_DIR");
    std::env::set_var("CARBON_CONFIG_DIR", "relative/cfg");
    assert!(save_stored_cloud_token("prod", "https://prod.example", &pro_tok).is_err());

    std::env::set_var("CARBON_CONFIG_DIR", &cfg_dir);
    let saved = save_stored_cloud_token("prod", "https://prod.example", &pro_tok)
        .expect("save prod credential");
    assert_eq!(saved.tenant_id, "acme_pro_team");

    let dev_tok = verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: "acme_dev_team".to_owned(),
            tier: CarbonTierWire::Pro,
            exp_epoch_secs: 4_102_444_800,
            admin: false,
        })
        .expect("mint dev token");
    save_stored_cloud_token("dev", "https://dev.example", &dev_tok).expect("save dev credential");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let meta = std::fs::metadata(cfg_dir.join("carbon-credentials.json")).unwrap();
        assert_eq!(meta.permissions().mode() & 0o777, 0o600);
    }

    let loaded_prod = load_stored_cloud_token("prod")
        .expect("load prod")
        .expect("prod present");
    assert_eq!(loaded_prod.tenant_id, "acme_pro_team");
    let dbg_str = format!("{loaded_prod:?}");
    assert!(
        !dbg_str.contains("ctk1."),
        "Debug formatting of CarbonStoredCloudCredential must redact raw token: {dbg_str}"
    );

    assert_eq!(remove_stored_cloud_token(Some("dev")).unwrap(), 1);
    assert!(load_stored_cloud_token("dev").unwrap().is_none());
    assert!(load_stored_cloud_token("prod").unwrap().is_some());
    assert_eq!(remove_stored_cloud_token(None).unwrap(), 1);
    assert!(!cfg_dir.join("carbon-credentials.json").exists());

    match prev_cfg {
        Some(v) => std::env::set_var("CARBON_CONFIG_DIR", v),
        None => std::env::remove_var("CARBON_CONFIG_DIR"),
    }

    // Verify registered Free capability token minting (30-day TTL, label normalization & validation)
    let reg_labeled = verifier
        .mint_registered_free(
            Some("Alice_Research-01"),
            1_794_400_000,
            Some("kek_pub_demo".to_owned()),
        )
        .expect("mint labeled registered free token");
    assert_eq!(reg_labeled.tier, CarbonTierWire::Free);
    assert_eq!(reg_labeled.expires_epoch_secs.0, 1_794_400_000 + 2_592_000);
    assert!(
        reg_labeled.tenant_id.starts_with("free_alice_research_01_"),
        "unexpected labeled tenant_id: {}",
        reg_labeled.tenant_id
    );
    let bearer_hdr = format!("Bearer {}", reg_labeled.capability_token);
    let reg_claims = verifier
        .verify_bearer_header(Some(&bearer_hdr), 1_794_400_001)
        .expect("verify registered free token");
    assert_eq!(reg_claims.tier, CarbonTierWire::Free);
    assert!(!reg_claims.admin);

    let reg_unlabeled = verifier
        .mint_registered_free(None, 1_794_400_000, None)
        .expect("mint unlabeled registered free token");
    assert!(reg_unlabeled.tenant_id.starts_with("free_"));

    assert!(verifier
        .mint_registered_free(Some("bad space"), 1_794_400_000, None)
        .is_err());
    assert!(verifier
        .mint_registered_free(Some("../../etc"), 1_794_400_000, None)
        .is_err());
    assert!(verifier
        .mint_registered_free(Some(&"a".repeat(25)), 1_794_400_000, None)
        .is_err());

    // Verify planner per-dataset cost breakdown across all 4 efforts (`fast` < `medium` < `high` < `ultra`)
    let mut prev_usd = 0.0;
    for effort in [
        jope_carbon::CarbonEffortTier::Fast,
        jope_carbon::CarbonEffortTier::Medium,
        jope_carbon::CarbonEffortTier::High,
        jope_carbon::CarbonEffortTier::Ultra,
    ] {
        let decision = jope_carbon::plan_carbon_execution(&jope_carbon::CarbonPlanInput {
            product_tier: jope_carbon::CarbonProductTier::Pro,
            effort,
            total_rows: 1_600,
            fit_rows: 1_280,
            features: 11,
            est_ms_per_candidate_1vcpu: 28.0,
        })
        .expect("plan Pro execution");
        let usd = decision.full_dataset_cost_estimate.total_estimated_usd;
        assert!(
            usd > prev_usd,
            "expected monotonic per-dataset cost across effort {effort:?}: {usd} vs {prev_usd}"
        );
        assert!(decision
            .full_dataset_cost_estimate
            .runs_per_1_usd
            .is_some_and(|r| r > 0.0 && r.is_finite()));
        prev_usd = usd;
    }

    // Enterprise VPC self-hosted mode reports $0.00 AWS cloud fee and `runs_per_1_usd: None` (valid JSON)
    let vpc_decision = jope_carbon::plan_carbon_execution(&jope_carbon::CarbonPlanInput {
        product_tier: jope_carbon::CarbonProductTier::no_limit(),
        effort: jope_carbon::CarbonEffortTier::High,
        total_rows: 50_000,
        fit_rows: 40_000,
        features: 24,
        est_ms_per_candidate_1vcpu: 35.0,
    })
    .expect("plan Enterprise VPC execution");
    assert_eq!(
        vpc_decision.full_dataset_cost_estimate.total_estimated_usd,
        0.0
    );
    assert!(vpc_decision
        .full_dataset_cost_estimate
        .runs_per_1_usd
        .is_none());
    let serialized = serde_json::to_string(&vpc_decision).expect("serialize CarbonPlanDecision");
    assert!(serialized.contains("\"full_dataset_cost_estimate\""));
}

#[test]
fn edge_async_202_threshold_and_retry_idempotent_create_run() {
    use base64::Engine as _;
    use jope_api::{
        carbon::{
            CarbonCreateRunRequest, CarbonRegionCatalog, CarbonRegionDescriptor, CarbonTierWire,
        },
        decimal::DecimalU64,
        ids::{DatasetId, RunId},
        Effort,
    };
    use jope_carbon::{
        CarbonAuthVerifier, CarbonCasStore, CarbonEdgeConfig, CarbonEdgeRouter,
        CarbonEngineInvoker, CarbonEngineKek, CarbonEngineWireCommand, CarbonFsCas,
        CarbonGatewayRequest, CarbonPresigner, CarbonThinClient, CAS_BLOB_CUEN_SCOPE,
    };
    use jope_prep::cuen::CuenPayloadKind;
    use std::collections::BTreeMap;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct CountingAsyncEngineInvoker {
        sync_calls: AtomicUsize,
        async_calls: AtomicUsize,
    }

    impl CarbonEngineInvoker for CountingAsyncEngineInvoker {
        fn invoke_engine(&self, _command: &CarbonEngineWireCommand) -> Result<Vec<u8>, String> {
            self.sync_calls.fetch_add(1, Ordering::SeqCst);
            Err("unexpected synchronous invoke_engine call".to_owned())
        }

        fn invoke_engine_async(&self, _command: &CarbonEngineWireCommand) -> Result<(), String> {
            self.async_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    let tmp = tempfile::tempdir().expect("tempdir");
    let cas = Arc::new(CarbonFsCas::open(tmp.path().join("cas")).expect("open fs cas"));
    let invoker = Arc::new(CountingAsyncEngineInvoker {
        sync_calls: AtomicUsize::new(0),
        async_calls: AtomicUsize::new(0),
    });
    let kek = CarbonEngineKek::generate().expect("generate KEK");
    let kek_b64 = kek.pubkey_b64();
    let kek_pub = jope_carbon::decode_p256_pubkey_b64(&kek_b64).expect("decode KEK pubkey");
    let auth_verifier = CarbonAuthVerifier::from_secret([0x5Au8; 32]);
    let reg = auth_verifier
        .mint_registered_free(Some("ux_retry"), 1_794_500_000, Some(kek_b64.clone()))
        .expect("mint registered free");

    let router = CarbonEdgeRouter::new(
        CarbonEdgeConfig {
            region: "us-east-2".to_owned(),
            public_base_url: "https://edge-test.example".to_owned(),
            enterprise_self_hosted: false,
            catalog: CarbonRegionCatalog {
                catalog_version: 1,
                regions: vec![CarbonRegionDescriptor {
                    region: "us-east-2".to_owned(),
                    geography: "americas".to_owned(),
                    endpoint_url: "https://edge-test.example".to_owned(),
                    priority_rank: 1,
                }],
            },
            presigner: CarbonPresigner::from_secret([0x6Bu8; 32]),
            auth_verifier: Some(auth_verifier),
            engine_kek_pubkey_b64: Some(kek_b64),
        },
        cas.clone(),
        invoker.clone(),
    );

    // 64 rows x 40 features = 2,560 cells (>= 2,500 threshold, < 10,000 old threshold) on Effort::Fast
    let (hdrs, mat, _) = jope_carbon::generate_nonlinear_benchmark_rows(64, 40, 77);
    let prep = CarbonThinClient::prepare_numeric_dataset(hdrs, &mat, "y", 77, 0.20).unwrap();
    let u_dek = jope_carbon::generate_random_u_dek().unwrap();
    let fit_enc = jope_carbon::encrypt_cuen_payload(
        &u_dek,
        CuenPayloadKind::FitJmat,
        &reg.tenant_id,
        CAS_BLOB_CUEN_SCOPE,
        &prep.artifacts.fit_jmat,
    )
    .unwrap();
    let hold_enc = jope_carbon::encrypt_cuen_payload(
        &u_dek,
        CuenPayloadKind::HoldoutJmat,
        &reg.tenant_id,
        CAS_BLOB_CUEN_SCOPE,
        &prep.artifacts.holdout_jmat,
    )
    .unwrap();
    let fit_sha = jope_carbon::sha256_lower_hex(&fit_enc);
    let hold_sha = jope_carbon::sha256_lower_hex(&hold_enc);
    cas.put_blob_verified(&fit_sha, &fit_enc).unwrap();
    cas.put_blob_verified(&hold_sha, &hold_enc).unwrap();

    let run_id_str = "run_01JCCARB0NE2E0000000000044";
    let run_id: RunId = run_id_str.parse().unwrap();
    let ds_id: DatasetId = "ds_444444444444444444444444".parse().unwrap();
    let enc_lm = jope_carbon::encrypt_cuen_payload(
        &u_dek,
        CuenPayloadKind::LandmarkJlmb,
        &reg.tenant_id,
        run_id_str,
        &prep.landmark_bundle.to_wire_bytes(),
    )
    .unwrap();
    let cudk =
        jope_carbon::wrap_u_dek_to_pubkey(&u_dek, &kek_pub, &reg.tenant_id, run_id_str).unwrap();

    let create_req = CarbonCreateRunRequest {
        run_id,
        dataset_id: ds_id,
        tenant_id: reg.tenant_id.clone(),
        tier: CarbonTierWire::Free,
        upload_id: None,
        fit_jmat_sha256: fit_sha,
        holdout_jmat_sha256: hold_sha,
        fit_rows: prep.fit_rows,
        holdout_rows: prep.holdout_rows,
        feature_names: prep.feature_names,
        target_name: prep.target_name,
        objective: "rmse".to_owned(),
        effort: Effort::Fast,
        seed: DecimalU64(77),
        landmark_bundle_b64: base64::engine::general_purpose::STANDARD.encode(enc_lm),
        encrypted_cuen: true,
        wrapped_u_dek_b64: Some(base64::engine::general_purpose::STANDARD.encode(cudk)),
    };
    let create_bytes = serde_json::to_vec(&create_req).unwrap();

    let mut headers = BTreeMap::new();
    headers.insert(
        "x-carbon-capability-token".to_owned(),
        format!("Bearer {}", reg.capability_token),
    );

    let first_resp = router.handle(&CarbonGatewayRequest {
        method: "POST".to_owned(),
        path: "/v1/carbon/runs".to_owned(),
        query: BTreeMap::new(),
        headers: headers.clone(),
        body_b64: None,
        raw_body: create_bytes.clone(),
    });
    assert_eq!(
        first_resp.status, 202,
        "2,560-cell fast run must dispatch asynchronously (HTTP 202) to avoid API Gateway 30s timeout"
    );
    assert_eq!(invoker.sync_calls.load(Ordering::SeqCst), 0);
    assert_eq!(invoker.async_calls.load(Ordering::SeqCst), 1);

    let retry_same_resp = router.handle(&CarbonGatewayRequest {
        method: "POST".to_owned(),
        path: "/v1/carbon/runs".to_owned(),
        query: BTreeMap::new(),
        headers: headers.clone(),
        body_b64: None,
        raw_body: create_bytes,
    });
    assert_eq!(
        retry_same_resp.status, 202,
        "byte-identical client retry of POST /v1/carbon/runs must return HTTP 202 idempotently"
    );
    assert_eq!(
        invoker.async_calls.load(Ordering::SeqCst),
        1,
        "idempotent retry must NOT re-invoke engine_async"
    );

    let mut mutated_req = create_req;
    mutated_req.seed = DecimalU64(999);
    let mutated_resp = router.handle(&CarbonGatewayRequest {
        method: "POST".to_owned(),
        path: "/v1/carbon/runs".to_owned(),
        query: BTreeMap::new(),
        headers,
        body_b64: None,
        raw_body: serde_json::to_vec(&mutated_req).unwrap(),
    });
    assert_eq!(
        mutated_resp.status, 409,
        "modified duplicate run_id payload must fail closed with HTTP 409"
    );
    assert_eq!(invoker.async_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn spec_png_and_pdf_verifiers_accept_valid_streams_and_reject_tampering_or_secrets() {
    use jope_carbon::{verify_pdf_bytes, verify_png_bytes, MIN_SPEC_PDF_PAGES};

    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in bytes {
            crc ^= u32::from(b);
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320u32 & mask);
            }
        }
        !crc
    }

    fn push_png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], payload: &[u8]) {
        out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        let start = out.len();
        out.extend_from_slice(kind);
        out.extend_from_slice(payload);
        let c = crc32(&out[start..]);
        out.extend_from_slice(&c.to_be_bytes());
    }

    let mut valid_png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&2240u32.to_be_bytes());
    ihdr.extend_from_slice(&640u32.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    push_png_chunk(&mut valid_png, b"IHDR", &ihdr);
    push_png_chunk(
        &mut valid_png,
        b"IDAT",
        &[0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01],
    );
    push_png_chunk(&mut valid_png, b"IEND", &[]);

    assert_eq!(
        verify_png_bytes("valid.png", &valid_png).unwrap(),
        (2240, 640)
    );

    let mut bad_crc_png = valid_png.clone();
    let flip_idx = bad_crc_png.len() - 15;
    bad_crc_png[flip_idx] ^= 0xFF;
    assert!(verify_png_bytes("corrupt.png", &bad_crc_png).is_err());

    let mut small_png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut small_ihdr = Vec::new();
    small_ihdr.extend_from_slice(&800u32.to_be_bytes());
    small_ihdr.extend_from_slice(&400u32.to_be_bytes());
    small_ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    push_png_chunk(&mut small_png, b"IHDR", &small_ihdr);
    push_png_chunk(&mut small_png, b"IDAT", &[1, 2, 3]);
    push_png_chunk(&mut small_png, b"IEND", &[]);
    assert!(verify_png_bytes("too_narrow.png", &small_png).is_err());

    let mut valid_pdf = b"%PDF-1.7\n1 0 obj << /Type /Pages /Count 14 >> endobj\n".to_vec();
    for idx in 0..14 {
        valid_pdf
            .extend_from_slice(format!("{} 0 obj << /Type /Page >> endobj\n", idx + 2).as_bytes());
    }
    valid_pdf.extend_from_slice(&vec![b' '; 85_000]);
    valid_pdf.extend_from_slice(b"\n%%EOF\n");
    let pages = verify_pdf_bytes("spec.pdf", &valid_pdf).unwrap();
    assert!(pages >= MIN_SPEC_PDF_PAGES);
    assert_eq!(pages, 14);

    let mut leaked_pdf = valid_pdf.clone();
    let tail = leaked_pdf.split_off(leaked_pdf.len() - 10);
    leaked_pdf.extend_from_slice(b"ctk1.secretpayload.deadbeef\n");
    leaked_pdf.extend_from_slice(&tail);
    assert!(verify_pdf_bytes("leaked.pdf", &leaked_pdf).is_err());
}

#[test]
fn dual_key_rotation_rate_limiter_eviction_and_cloudformation_zero_secret_invariants() {
    use jope_api::carbon::CarbonTierWire;
    use jope_carbon::{
        synthesize_cell_template_for_release, unwrap_u_dek_with_kek, wrap_u_dek_to_pubkey,
        CarbonAuthVerifier, CarbonCapabilityClaims, CarbonEngineKek, CarbonGatewayRequest,
        CarbonPresigner, CarbonRateLimiter, CarbonSealDomain, CarbonSealKey,
        SealedCheckpointEnvelope,
    };
    use std::collections::BTreeMap;

    // 1. Dual-key `[active, previous]` rotation across Auth, Presigner, SealKey, and EngineKek
    let old_secret = [0x11u8; 32];
    let new_secret = [0x22u8; 32];
    let old_verifier = CarbonAuthVerifier::from_secret(old_secret);
    let tok_from_old = old_verifier
        .mint_token(&CarbonCapabilityClaims {
            tenant_id: "rot_tenant".to_owned(),
            tier: CarbonTierWire::Pro,
            exp_epoch_secs: 4_102_444_800,
            admin: false,
        })
        .unwrap();

    let dual_verifier =
        CarbonAuthVerifier::from_secret(new_secret).with_previous_secret(Some(old_secret));
    let verified = dual_verifier
        .verify_token(&tok_from_old, 1_794_500_000)
        .expect("dual verifier must accept token minted under previous secret");
    assert_eq!(verified.tenant_id, "rot_tenant");

    let pruned_verifier = CarbonAuthVerifier::from_secret(new_secret);
    assert!(
        pruned_verifier
            .verify_token(&tok_from_old, 1_794_500_000)
            .is_err(),
        "once previous secret is pruned, old tokens must be rejected"
    );

    let old_presigner = CarbonPresigner::from_secret(old_secret);
    let signed_url = old_presigner.sign_gateway_scoped_upload_url(
        "https://edge.example",
        Some("rot_tenant/upl_01J"),
        &"a".repeat(64),
        1_024,
        2_000_000_000,
    );
    let sig_hex = signed_url.rsplit_once("sig=").unwrap().1;
    let dual_presigner =
        CarbonPresigner::from_secret(new_secret).with_previous_secret(Some(old_secret));
    dual_presigner
        .verify_gateway_scoped_upload(
            Some("rot_tenant/upl_01J"),
            &"a".repeat(64),
            1_024,
            2_000_000_000,
            sig_hex,
            1_900_000_000,
        )
        .expect("dual presigner must accept URL signed under previous secret");
    assert!(CarbonPresigner::from_secret(new_secret)
        .verify_gateway_scoped_upload(
            Some("rot_tenant/upl_01J"),
            &"a".repeat(64),
            1_024,
            2_000_000_000,
            sig_hex,
            1_900_000_000,
        )
        .is_err());

    let old_seal = CarbonSealKey::from_bytes(old_secret);
    let u_dek = [0x77u8; 32];
    let ckp = SealedCheckpointEnvelope::seal_with_domain(
        &old_seal,
        CarbonSealDomain::ChampionModel,
        "rot_tenant",
        "run_01JROT",
        2,
        Some(&u_dek),
        b"champion-bytes",
    )
    .unwrap();
    let dual_seal = CarbonSealKey::from_bytes(new_secret).with_previous_bytes(Some(old_secret));
    let (wave, opened) = SealedCheckpointEnvelope::open_with_domain(
        &dual_seal,
        CarbonSealDomain::ChampionModel,
        "rot_tenant",
        "run_01JROT",
        Some(&u_dek),
        ckp.wire_bytes(),
    )
    .expect("dual seal key must open checkpoint sealed under previous secret");
    assert_eq!(wave, 2);
    assert_eq!(opened.as_slice(), b"champion-bytes");
    assert!(SealedCheckpointEnvelope::open_with_domain(
        &CarbonSealKey::from_bytes(new_secret),
        CarbonSealDomain::ChampionModel,
        "rot_tenant",
        "run_01JROT",
        Some(&u_dek),
        ckp.wire_bytes(),
    )
    .is_err());

    let old_kek_hex = "1111111111111111111111111111111111111111111111111111111111111111";
    let new_kek_hex = "2222222222222222222222222222222222222222222222222222222222222222";
    let old_kek = CarbonEngineKek::from_hex(old_kek_hex).unwrap();
    let cudk_old =
        wrap_u_dek_to_pubkey(&u_dek, old_kek.pubkey_bytes(), "rot_tenant", "run_01JROT").unwrap();
    let dual_kek = CarbonEngineKek::from_hex_with_previous(new_kek_hex, Some(old_kek_hex)).unwrap();
    let unwrapped = unwrap_u_dek_with_kek(&dual_kek, &cudk_old, "rot_tenant", "run_01JROT")
        .expect("dual KEK must unwrap CUDK encrypted to previous KEK pubkey");
    assert_eq!(*unwrapped, u_dek);
    let new_only_kek = CarbonEngineKek::from_hex(new_kek_hex).unwrap();
    assert!(unwrap_u_dek_with_kek(&new_only_kek, &cudk_old, "rot_tenant", "run_01JROT").is_err());

    // 2. Trusted caller IP prefers verified sourceIp over spoofed X-Forwarded-For
    let mut hdrs = BTreeMap::new();
    hdrs.insert("x-forwarded-for".to_owned(), "198.51.100.99".to_owned());
    hdrs.insert(
        "x-carbon-trusted-source-ip".to_owned(),
        "203.0.113.7".to_owned(),
    );
    let gw_req = CarbonGatewayRequest {
        method: "POST".to_owned(),
        path: "/v1/carbon/auth/anonymous".to_owned(),
        query: BTreeMap::new(),
        headers: hdrs,
        body_b64: None,
        raw_body: Vec::new(),
    };
    assert_eq!(gw_req.trusted_caller_ip("unknown"), "203.0.113.7");

    // 3. Rate limiter lowest-count eviction under 65,536-key flood (`GAP-SEC3`)
    let limiter = CarbonRateLimiter::new(60);
    assert!(limiter.check_and_record("heavy_abuser", 600, 5));
    assert!(limiter.check_and_record("heavy_abuser", 600, 5));
    assert!(limiter.check_and_record("heavy_abuser", 600, 5));
    assert!(limiter.check_and_record("heavy_abuser", 600, 5));
    assert!(limiter.check_and_record("heavy_abuser", 600, 5));
    assert!(!limiter.check_and_record("heavy_abuser", 600, 5));
    for idx in 0..65_536usize {
        let key = format!("flood_{idx}");
        assert!(limiter.check_and_record(&key, 600, 5));
    }
    assert!(
        limiter.check_and_record("legitimate_new_user", 600, 5),
        "new legitimate caller must NOT be locked out when 65,536 flood keys are present"
    );
    assert!(
        !limiter.check_and_record("heavy_abuser", 600, 5),
        "heavy_abuser with saturated counter must NOT be evicted ahead of single-hit flood keys"
    );

    // 4. CloudFormation zero-plaintext-secrets & read-only EvalRole IAM invariants
    let synth = synthesize_cell_template_for_release("112233445566").unwrap();
    assert!(
        !synth.contains("{{resolve:secretsmanager:"),
        "synthesized CloudFormation template must never inject dynamic plaintext secret strings into Lambda env vars"
    );
    let tpl: serde_json::Value = serde_json::from_str(&synth).unwrap();
    let edge_vars = &tpl["Resources"]["EdgeFunction"]["Properties"]["Environment"]["Variables"];
    assert!(edge_vars.get("CARBON_SECRETS_ARN").is_some());
    assert!(edge_vars.get("CARBON_SECRETS_VERSION_ID").is_some());
    let eval_policies =
        serde_json::to_string(&tpl["Resources"]["EvalRole"]["Properties"]["Policies"]).unwrap();
    assert!(eval_policies.contains("s3:GetObject"));
    assert!(
        !eval_policies.contains("s3:PutObject") && !eval_policies.contains("s3:DeleteObject"),
        "EvalRole must be strictly read-only on S3 CAS: {eval_policies}"
    );
}

#[test]
fn release_tag_parity_multiformat_installer_and_self_verify_tamper_guards() {
    use jope_carbon::{
        build_deterministic_single_bin_targz, sha256_lower_hex,
        verify_ecdsa_p256_spki_pem_signature, CARBON_GIT_SHA, CARBON_RELEASE_TAG, SPEC_RELEASE_TAG,
    };

    assert_eq!(CARBON_RELEASE_TAG, SPEC_RELEASE_TAG);
    assert!(!CARBON_GIT_SHA.is_empty());

    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let install_sh = std::fs::read_to_string(repo_root.join("deploy/carbon/install.sh")).unwrap();
    let install_ps1 = std::fs::read_to_string(repo_root.join("deploy/carbon/install.ps1")).unwrap();
    assert!(
        install_sh.contains(CARBON_RELEASE_TAG),
        "deploy/carbon/install.sh must pin {CARBON_RELEASE_TAG}"
    );
    assert!(
        install_ps1.contains(CARBON_RELEASE_TAG),
        "deploy/carbon/install.ps1 must pin {CARBON_RELEASE_TAG}"
    );

    // Feature-isolated `cli` and `eval` Cargo graphs (`CT3` client & worker boundaries)
    let cli_tree_out = Command::new("cargo")
        .args([
            "tree",
            "--locked",
            "-p",
            "jope-carbon",
            "--no-default-features",
            "--features",
            "cli",
            "--prefix",
            "none",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("cargo tree --features cli");
    assert!(cli_tree_out.status.success());
    let cli_tree = String::from_utf8(cli_tree_out.stdout).unwrap();
    for forbidden in [
        "axum ",
        "tokio ",
        "jope-service ",
        "jope-runtime ",
        "feat-math ",
        "candle-core ",
    ] {
        assert!(
            !cli_tree.contains(forbidden),
            "thin CLI feature graph leaked server/engine crate `{forbidden}`:\n{cli_tree}"
        );
    }

    let worker_bin = env!("CARGO_BIN_EXE_carbon-worker");
    let worker_hc = Command::new(worker_bin)
        .arg("--healthcheck")
        .output()
        .expect("carbon-worker --healthcheck");
    assert!(worker_hc.status.success());

    // Sign a real temporary release bundle with an ephemeral OpenSSL P-256 keypair, install `.tar.gz`
    // via `install.sh --pub-key`, then verify pure-Rust `verify_ecdsa_p256_spki_pem_signature` + `carbon self-verify`!
    let tmp = tempfile::tempdir().unwrap();
    let priv_pem = tmp.path().join("ephemeral_priv.pem");
    let pub_pem = tmp.path().join("ephemeral_pub.pem");
    assert!(Command::new("openssl")
        .args([
            "ecparam",
            "-name",
            "prime256v1",
            "-genkey",
            "-noout",
            "-out"
        ])
        .arg(&priv_pem)
        .status()
        .unwrap()
        .success());
    assert!(Command::new("openssl")
        .args(["ec", "-in"])
        .arg(&priv_pem)
        .arg("-pubout")
        .arg("-out")
        .arg(&pub_pem)
        .status()
        .unwrap()
        .success());

    let src_dir = tmp.path().join("rel-src");
    let bin_dir = tmp.path().join("bin-out");
    let cfg_dir = tmp.path().join("cfg-out");
    std::fs::create_dir_all(&src_dir).unwrap();

    let carbon_bin = std::path::PathBuf::from(env!("CARGO_BIN_EXE_carbon"));
    let carbon_bytes = std::fs::read(&carbon_bin).unwrap();
    let arch_base = if cfg!(target_arch = "aarch64") {
        "carbon-linux-aarch64-musl"
    } else {
        "carbon-linux-x86_64-musl"
    };
    let tgz_name = format!("{arch_base}.tar.gz");
    let tgz_bytes = build_deterministic_single_bin_targz("carbon", &carbon_bytes).unwrap();
    std::fs::write(src_dir.join(&tgz_name), &tgz_bytes).unwrap();

    let kek = jope_carbon::CarbonEngineKek::generate().unwrap();
    let clouds_bytes = serde_json::to_vec_pretty(&serde_json::json!({
        "schema": "jope-carbon-clouds-v2",
        "generated_epoch_ms": 1794500000000u64,
        "clouds": {
            "prod": {
                "environment": "prod",
                "region": "us-east-2",
                "public_endpoint": "https://prod.example",
                "engine_kek_pubkey_b64": kek.pubkey_b64(),
                "regions": [{
                    "region": "us-east-2",
                    "public_endpoint": "https://prod.example",
                    "engine_kek_pubkey_b64": kek.pubkey_b64()
                }]
            }
        }
    }))
    .unwrap();
    std::fs::write(src_dir.join("carbon-clouds.json"), &clouds_bytes).unwrap();

    let sums_content = format!(
        "{}  {}\n{}  {}\n{}  carbon-clouds.json\n",
        sha256_lower_hex(&carbon_bytes),
        arch_base,
        sha256_lower_hex(&tgz_bytes),
        tgz_name,
        sha256_lower_hex(&clouds_bytes),
    );
    let sums_path = src_dir.join("SHA256SUMS.txt");
    std::fs::write(&sums_path, sums_content.as_bytes()).unwrap();

    let sig_der_out = Command::new("openssl")
        .args(["dgst", "-sha256", "-sign"])
        .arg(&priv_pem)
        .arg(&sums_path)
        .output()
        .unwrap();
    assert!(sig_der_out.status.success());
    let pub_pem_text = std::fs::read_to_string(&pub_pem).unwrap();
    verify_ecdsa_p256_spki_pem_signature(
        &pub_pem_text,
        sums_content.as_bytes(),
        &sig_der_out.stdout,
    )
    .expect("pure-Rust aws-lc-rs ECDSA-P256 verifier must accept valid OpenSSL signature");
    let mut bad_sums = sums_content.as_bytes().to_vec();
    bad_sums[0] ^= 0x01;
    assert!(
        verify_ecdsa_p256_spki_pem_signature(&pub_pem_text, &bad_sums, &sig_der_out.stdout)
            .is_err(),
        "pure-Rust ECDSA-P256 verifier must reject tampered manifest"
    );

    use base64::Engine as _;
    let sig_b64 = format!(
        "{}\n",
        base64::engine::general_purpose::STANDARD.encode(&sig_der_out.stdout)
    );
    std::fs::write(src_dir.join("SHA256SUMS.txt.sig"), sig_b64).unwrap();

    let inst_out = Command::new("sh")
        .arg(repo_root.join("deploy/carbon/install.sh"))
        .arg("--source-dir")
        .arg(&src_dir)
        .arg("--bin-dir")
        .arg(&bin_dir)
        .arg("--config-dir")
        .arg(&cfg_dir)
        .arg("--allow-test-pubkey")
        .arg(&pub_pem)
        .arg("--skip-ping")
        .output()
        .expect("run install.sh with .tar.gz archive");
    assert!(
        inst_out.status.success(),
        "install.sh .tar.gz install failed: {}",
        String::from_utf8_lossy(&inst_out.stderr)
    );
    assert!(bin_dir.join("carbon").is_file());
    assert!(cfg_dir.join("SHA256SUMS.txt").is_file());
    assert!(cfg_dir.join("SHA256SUMS.txt.sig").is_file());

    // Tampered signature must cause `carbon self-verify` to fail closed!
    let bad_sig_file = tmp.path().join("bad.sig");
    std::fs::write(&bad_sig_file, "MEUCIQD00000000000000000\n").unwrap();
    let sv_bad = Command::new(bin_dir.join("carbon"))
        .arg("self-verify")
        .arg("--sums")
        .arg(cfg_dir.join("SHA256SUMS.txt"))
        .arg("--sig")
        .arg(&bad_sig_file)
        .env("CARBON_CONFIG_DIR", &cfg_dir)
        .output()
        .expect("run carbon self-verify with bad sig");
    assert!(
        !sv_bad.status.success(),
        "carbon self-verify must fail closed on invalid signature"
    );

    // Unknown subcommand must exit with code 2 (not silently run as CSV)!
    let unk = Command::new(bin_dir.join("carbon"))
        .arg("nonexistent-verb")
        .output()
        .expect("run unknown verb");
    assert_eq!(unk.status.code(), Some(2));

    // Verify `execute_carbon_spec_report` validates all 8 virgin-Docker dev+prod 4-tier runs and renders Table 6C + Table 6D
    let ci_qual_fixture = tmp.path().join("ci-qual-fixture.json");
    std::fs::write(
        &ci_qual_fixture,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": "jopedime.carbon.ci-quickstart-qualification.v1",
            "timestamp_epoch_ms": 1794500000000u64,
            "source_sha": "e10c2dd594b64a0d9717823bbc5374db1fa0189d",
            "binary_version": "1.0.0",
            "release_tag": "carbon-cli-v1.1.0",
            "emulator_regions": ["prod:us-east-2", "prod:eu-central-1", "dev:us-east-2"],
            "emulator_edge_invocations": 42,
            "emulator_engine_invocations": 12,
            "emulator_eval_invocations": 8,
            "residual_cas_blobs_after_shred": 0,
            "total_tests": 8,
            "passed_tests": 8,
            "all_passed": true,
            "total_wall_ms": 1234.5,
            "steps": [],
            "redteam_vectors": []
        }))
        .unwrap(),
    )
    .unwrap();
    let spec_out_dir = tmp.path().join("spec-out");
    let spec_html_path =
        jope_carbon::execute_carbon_spec_report(&jope_carbon::CarbonSpecReportOptions {
            evidence_dir: repo_root.join("docs/carbon/evidence"),
            out_dir: spec_out_dir.clone(),
            run_ci_qualification: false,
            ci_qualification_in: Some(ci_qual_fixture),
            source_sha: Some("e10c2dd594b64a0d9717823bbc5374db1fa0189d".to_owned()),
            repo_root: repo_root.to_path_buf(),
        })
        .expect("execute_carbon_spec_report must validate 8-run dev+prod virgin-Docker receipt");
    let spec_html = std::fs::read_to_string(&spec_html_path).unwrap();
    assert!(spec_html.contains("Table 6C-1"));
    assert!(spec_html.contains("Table 6C-2"));
    assert!(spec_html.contains("Table 6D"));
    assert!(spec_html.contains("Table 6E"));
    assert!(spec_html.contains("https://downloads.neverhuman.org/carbon/install.sh"));
    let docker_ev: serde_json::Value = serde_json::from_slice(
        &std::fs::read(repo_root.join("docs/carbon/evidence/docker-virgin-ubuntu-prod-e2e.json"))
            .unwrap(),
    )
    .unwrap();
    let dev_ultra_id = docker_ev["dev_runs"]["ultra"]["receipt"]["run_id"]
        .as_str()
        .expect("dev ultra run_id");
    let prod_ultra_id = docker_ev["prod_runs"]["ultra"]["receipt"]["run_id"]
        .as_str()
        .expect("prod ultra run_id");
    assert!(spec_html.contains(dev_ultra_id));
    assert!(spec_html.contains(prod_ultra_id));
}

#[test]
fn xz_and_zstd_stream_compress_large_incompressible_payload_without_pipe_deadlock() {
    use jope_carbon::{build_deterministic_single_bin_tar, try_external_stream_compress};

    // Deterministic 512 KiB high-entropy SplitMix64 payload wrapped in USTAR so compressed output
    // (> 512 KiB) is > 8x the 64 KiB OS pipe buffer and would deterministically deadlock if stdin
    // were written synchronously before draining stdout.
    let mut raw = vec![0u8; 512 * 1_024];
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for chunk in raw.chunks_exact_mut(8) {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        chunk.copy_from_slice(&z.to_le_bytes());
    }
    let tar_bytes = build_deterministic_single_bin_tar("carbon", &raw).expect("ustar archive");
    assert!(tar_bytes.len() > 512 * 1_024);

    let mut exercised_any = false;
    if Command::new("xz")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
    {
        exercised_any = true;
        let compressed = try_external_stream_compress("xz", &["-1", "-q", "-c"], &tar_bytes)
            .expect("xz stream compress >512 KiB payload without deadlock");
        assert!(
            compressed.len() > 512 * 1_024,
            "high-entropy xz stream must exceed 64 KiB pipe buffer, got {} B",
            compressed.len()
        );
        let roundtrip = try_external_stream_compress("xz", &["-d", "-q", "-c"], &compressed)
            .expect("xz stream decompress >512 KiB payload without deadlock");
        assert_eq!(roundtrip, tar_bytes);
    }

    if Command::new("zstd")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
    {
        exercised_any = true;
        let compressed = try_external_stream_compress("zstd", &["-1", "-q", "-c"], &tar_bytes)
            .expect("zstd stream compress >512 KiB payload without deadlock");
        assert!(
            compressed.len() > 512 * 1_024,
            "high-entropy zstd stream must exceed 64 KiB pipe buffer, got {} B",
            compressed.len()
        );
        let roundtrip = try_external_stream_compress("zstd", &["-d", "-q", "-c"], &compressed)
            .expect("zstd stream decompress >512 KiB payload without deadlock");
        assert_eq!(roundtrip, tar_bytes);
    }

    assert!(
        exercised_any,
        "expected at least one of `xz` or `zstd` to be installed in test environment"
    );
    assert!(
        try_external_stream_compress("nonexistent-compressor-binary-xyz", &["-c"], &tar_bytes)
            .is_none(),
        "missing compressor binary must return None cleanly"
    );
}

#[test]
fn spec_report_renders_full_product_coverage_matrix_and_rejects_failed_live_check() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let tmp = tempfile::tempdir().expect("tempdir");
    let ci_qual_fixture = tmp.path().join("ci-qual.json");
    std::fs::write(
        &ci_qual_fixture,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": "jopedime.carbon.ci-quickstart-qualification.v1",
            "timestamp_epoch_ms": 1794622000000u64,
            "source_sha": "92ef2d2c1bc4356f95482255e6cc093f7ea76e87",
            "binary_version": "1.0.0",
            "release_tag": "carbon-cli-v1.1.0",
            "emulator_regions": ["prod:us-east-2", "prod:eu-central-1", "dev:us-east-2"],
            "emulator_edge_invocations": 42,
            "emulator_engine_invocations": 12,
            "emulator_eval_invocations": 8,
            "residual_cas_blobs_after_shred": 0,
            "total_tests": 8,
            "passed_tests": 8,
            "all_passed": true,
            "total_wall_ms": 1234.5,
            "steps": [],
            "redteam_vectors": []
        }))
        .expect("serialize ci_qual"),
    )
    .expect("write ci_qual");

    assert_eq!(
        jope_carbon::cli::spec_report::REQUIRED_EVIDENCE_FILES.len(),
        14
    );
    let out_ok = tmp.path().join("spec-ok");
    let html_path =
        jope_carbon::execute_carbon_spec_report(&jope_carbon::CarbonSpecReportOptions {
            evidence_dir: repo_root.join("docs/carbon/evidence"),
            out_dir: out_ok.clone(),
            run_ci_qualification: false,
            ci_qualification_in: Some(ci_qual_fixture.clone()),
            source_sha: Some("92ef2d2c1bc4356f95482255e6cc093f7ea76e87".to_owned()),
            repo_root: repo_root.clone(),
        })
        .expect("spec-report must validate Section 11 coverage matrix and all 14 v1.1.0 receipts");
    let manifest_json: serde_json::Value = serde_json::from_slice(
        &std::fs::read(out_ok.join("spec-bundle-manifest.json")).expect("read manifest"),
    )
    .expect("parse manifest");
    assert_eq!(
        manifest_json["evidence_ledger"]
            .as_array()
            .expect("ledger array")
            .len(),
        14
    );
    let html = std::fs::read_to_string(&html_path).expect("read spec html");
    for expected in [
        "Table 2B",
        "Table 5B",
        "Table 6C-1",
        "Table 6C-2",
        "Table 6D",
        "Table 6E",
        "Table 8A",
        "Table 8C",
        "11. Full Product Test Coverage Matrix",
        "Table 11A",
        "Table 11B",
        "Table 11C",
        "Table 11D",
        "Table 11E",
        "tested-in-CI",
        "tested-live-prod",
        "NOT covered",
        "live-product-coverage-dev-prod.json",
        "macos-arm64-live-e2e.json",
        "aarch64-apple-darwin",
        "run_01M7CC4P6K4EPKSD1B3FDDWMS1",
        "rollback-dev-live.json",
        "origin_local_storage_dependency = false",
        "https://downloads.neverhuman.org/carbon/install.sh",
        "17.77",
        "$0.000299 / run",
        "284.43s",
        "317.98s",
        "852.03s",
        "21/32",
        "$0.055166",
        "13. Full 64-Hex SHA-256 Cryptographic Provenance Ledger",
    ] {
        assert!(html.contains(expected), "missing '{expected}' in spec HTML");
    }
    for stale in ["17.24", "$0.000353", "277.23s", "845.41s", "13/32"] {
        assert!(
            !html.contains(stale),
            "stale pre-v1.1.0 literal '{stale}' remained in spec HTML"
        );
    }

    // Mutating `macos-arm64-live-e2e.json` (`quickstart_shredded = false`) must fail closed.
    let bad_mac = tmp.path().join("bad-macos");
    std::fs::create_dir_all(&bad_mac).expect("mkdir bad-macos");
    for entry in std::fs::read_dir(repo_root.join("docs/carbon/evidence")).expect("readdir") {
        let entry = entry.expect("dir entry");
        if entry.file_type().expect("ft").is_file() {
            std::fs::copy(entry.path(), bad_mac.join(entry.file_name())).expect("copy ev");
        }
    }
    let mac_path = bad_mac.join("macos-arm64-live-e2e.json");
    let mut mac_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&mac_path).expect("read mac")).expect("parse mac");
    mac_json["quickstart_shredded"] = serde_json::Value::Bool(false);
    std::fs::write(
        &mac_path,
        serde_json::to_vec_pretty(&mac_json).expect("ser mac"),
    )
    .expect("write mac");
    assert!(
        jope_carbon::execute_carbon_spec_report(&jope_carbon::CarbonSpecReportOptions {
            evidence_dir: bad_mac,
            out_dir: tmp.path().join("spec-bad-mac"),
            run_ci_qualification: false,
            ci_qualification_in: Some(ci_qual_fixture.clone()),
            source_sha: Some("92ef2d2c1bc4356f95482255e6cc093f7ea76e87".to_owned()),
            repo_root: repo_root.clone(),
        })
        .is_err(),
        "unshredded macOS arm64 receipt must reject spec-report"
    );

    // Mutating a live prod check to `passed: false` must fail `execute_carbon_spec_report` closed.
    let bad_ev = tmp.path().join("bad-evidence");
    std::fs::create_dir_all(&bad_ev).expect("mkdir bad-evidence");
    for entry in std::fs::read_dir(repo_root.join("docs/carbon/evidence")).expect("readdir") {
        let entry = entry.expect("dir entry");
        if entry.file_type().expect("ft").is_file() {
            std::fs::copy(entry.path(), bad_ev.join(entry.file_name())).expect("copy ev");
        }
    }
    let cov_path = bad_ev.join("live-product-coverage-dev-prod.json");
    let mut cov_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&cov_path).expect("read cov")).expect("parse cov");
    cov_json["clouds"]["prod"]["checks"][0]["passed"] = serde_json::Value::Bool(false);
    std::fs::write(
        &cov_path,
        serde_json::to_vec_pretty(&cov_json).expect("ser"),
    )
    .expect("write");

    let err = jope_carbon::execute_carbon_spec_report(&jope_carbon::CarbonSpecReportOptions {
        evidence_dir: bad_ev,
        out_dir: tmp.path().join("spec-bad"),
        run_ci_qualification: false,
        ci_qualification_in: Some(ci_qual_fixture),
        source_sha: Some("92ef2d2c1bc4356f95482255e6cc093f7ea76e87".to_owned()),
        repo_root,
    })
    .expect_err("failed live prod coverage check must reject spec-report");
    assert!(
        err.contains("API-LIVE-01") || err.contains("failed"),
        "unexpected error: {err}"
    );
}

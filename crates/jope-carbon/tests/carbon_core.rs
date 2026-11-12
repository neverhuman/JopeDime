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

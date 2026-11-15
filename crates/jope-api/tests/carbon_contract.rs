use jope_api::{
    carbon::{
        CarbonAnonymousAuthResponse, CarbonCreateRunRequest, CarbonEarlyStopRequest,
        CarbonExecutionModeWire, CarbonExplainRequest, CarbonExplainResponse, CarbonExportLangWire,
        CarbonExportRequest, CarbonExportResponse, CarbonFeatureImportanceWire, CarbonPingResponse,
        CarbonPredictRequest, CarbonPredictResponse, CarbonPreparePredictRequest,
        CarbonPreparePredictResponse, CarbonPrepareUploadRequest, CarbonPrepareUploadResponse,
        CarbonPresignedTarget, CarbonPublicWaveProgress, CarbonRegionCatalog,
        CarbonRegionDescriptor, CarbonRegisterFreeRequest, CarbonRegisterFreeResponse,
        CarbonResumeRunRequest, CarbonRunStateWire, CarbonShredRunResponse, CarbonTierWire,
        CarbonUploadSlotWire, CarbonVersionResponse, CARBON_DEFAULT_REGIONS,
    },
    decimal::DecimalU64,
    ids::{DatasetId, RunId},
    Effort,
};

#[test]
fn carbon_wire_round_trip_and_strict_unknown_field_rejection() {
    let run_id: RunId = "run_01JCCARB0N0000000000000001"
        .parse()
        .expect("valid RunId");
    let dataset_id: DatasetId = "ds_111111111111111111111111"
        .parse()
        .expect("valid DatasetId");

    let anon = CarbonAnonymousAuthResponse {
        tenant_id: "anon_01JCCARB0N0000000000000099".to_owned(),
        capability_token: "ctk1.payload.sig".to_owned(),
        tier: CarbonTierWire::Free,
        expires_epoch_secs: DecimalU64(1_800_000_000u64),
        engine_kek_pubkey_b64: Some("BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=".to_owned()),
    };
    assert_eq!(
        serde_json::from_str::<CarbonAnonymousAuthResponse>(
            &serde_json::to_string(&anon).expect("encode anon")
        )
        .expect("decode anon"),
        anon
    );

    let reg_req = CarbonRegisterFreeRequest {
        label: Some("alice_laptop".to_owned()),
    };
    assert_eq!(
        serde_json::from_str::<CarbonRegisterFreeRequest>(
            &serde_json::to_string(&reg_req).expect("encode reg_req")
        )
        .expect("decode reg_req"),
        reg_req
    );
    let reg_resp = CarbonRegisterFreeResponse {
        tenant_id: "free_alice_laptop_01jccarb0n".to_owned(),
        capability_token: "ctk1.payload.sig".to_owned(),
        tier: CarbonTierWire::Free,
        expires_epoch_secs: DecimalU64(1_802_592_000u64),
        engine_kek_pubkey_b64: Some("BA==".to_owned()),
    };
    assert_eq!(
        serde_json::from_str::<CarbonRegisterFreeResponse>(
            &serde_json::to_string(&reg_resp).expect("encode reg_resp")
        )
        .expect("decode reg_resp"),
        reg_resp
    );

    let prepare = CarbonPrepareUploadRequest {
        tenant_id: "tenant-pro".to_owned(),
        tier: CarbonTierWire::Pro,
        fit_rows: 1_200,
        holdout_rows: 300,
        feature_count: 12,
        fit_jmat: CarbonUploadSlotWire {
            sha256: "a".repeat(64),
            bytes: DecimalU64(48_052u64),
        },
        holdout_jmat: CarbonUploadSlotWire {
            sha256: "b".repeat(64),
            bytes: DecimalU64(12_052u64),
        },
        encrypted_cuen: true,
    };
    let prepare_json = serde_json::to_string(&prepare).expect("serialize prepare");
    assert_eq!(
        serde_json::from_str::<CarbonPrepareUploadRequest>(&prepare_json).expect("decode prepare"),
        prepare
    );

    let prep_resp = CarbonPrepareUploadResponse {
        upload_id: "upl-001".to_owned(),
        fit_target: CarbonPresignedTarget {
            url: "https://us-east-2.carbon.jopedime.com/v1/carbon/cas/aaa".to_owned(),
            method: "PUT".to_owned(),
            sha256: "a".repeat(64),
            max_bytes: DecimalU64(48_052u64),
            expires_epoch_secs: DecimalU64(1_800_000_000u64),
        },
        holdout_target: CarbonPresignedTarget {
            url: "https://us-east-2.carbon.jopedime.com/v1/carbon/cas/bbb".to_owned(),
            method: "PUT".to_owned(),
            sha256: "b".repeat(64),
            max_bytes: DecimalU64(12_052u64),
            expires_epoch_secs: DecimalU64(1_800_000_000u64),
        },
        engine_kek_pubkey_b64: Some("BA==".to_owned()),
    };
    assert_eq!(
        serde_json::from_str::<CarbonPrepareUploadResponse>(
            &serde_json::to_string(&prep_resp).expect("encode")
        )
        .expect("decode"),
        prep_resp
    );

    let create = CarbonCreateRunRequest {
        run_id: run_id.clone(),
        dataset_id,
        tenant_id: "tenant-pro".to_owned(),
        tier: CarbonTierWire::Pro,
        fit_jmat_sha256: "a".repeat(64),
        holdout_jmat_sha256: "b".repeat(64),
        fit_rows: 1_200,
        holdout_rows: 300,
        feature_names: vec!["x0".to_owned(), "x1".to_owned()],
        target_name: "y".to_owned(),
        objective: "rmse".to_owned(),
        effort: Effort::Medium,
        seed: DecimalU64(42u64),
        landmark_bundle_b64: "Q1VFTgE=".to_owned(),
        encrypted_cuen: true,
        wrapped_u_dek_b64: Some("Q1VESwE=".to_owned()),
        upload_id: Some("upl_01jccarb0n0000000000000001".to_owned()),
        aux_column_names: Vec::new(),
        split_mode: None,
        calibration_mode: None,
    };
    assert_eq!(
        serde_json::from_str::<CarbonCreateRunRequest>(
            &serde_json::to_string(&create).expect("encode create")
        )
        .expect("decode create"),
        create
    );

    let progress = CarbonPublicWaveProgress {
        run_id: run_id.clone(),
        state: CarbonRunStateWire::Paused,
        execution_mode: CarbonExecutionModeWire::SoloSingleLambda,
        eval_memory_mb: 512,
        eval_shards: 0,
        eval_invocations: DecimalU64(0u64),
        eval_total_wall_ms: DecimalU64(0u64),
        wave_index: 3,
        waves_target: 8,
        candidates_evaluated: DecimalU64(96u64),
        best_cv_loss: Some(0.142),
        best_cv_metric: Some(0.142),
        outer_holdout_loss: Some(0.149),
        outer_holdout_metric: Some(0.149),
        neural_seed_applied: true,
        neural_guidance_mode: Some("multi_wave_learned".to_owned()),
        bundle_digest: Some(
            "sha256:057e50a9a741d30d541ed8af8339cec84a5157af621100cb70857052ef14dacc".to_owned(),
        ),
        tier_applied: Some(CarbonTierWire::Pro),
        encrypted_cuen: true,
        can_resume: true,
        sealed_checkpoint_sha256: Some("c".repeat(64)),
        sealed_champion_sha256: Some("d".repeat(64)),
        elapsed_ms: DecimalU64(1_450u64),
        region: "us-east-2".to_owned(),
        failure_message: None,
        eval_fallback_shards: None,
    };
    let progress_json = serde_json::to_string(&progress).expect("encode progress");
    for forbidden in [
        "\"dsl\"",
        "\"ast\"",
        "\"genome\"",
        "\"expression\"",
        "\"program\"",
        "\"operator\"",
    ] {
        assert!(
            !progress_json.contains(forbidden),
            "progress wire leaked forbidden key {forbidden}"
        );
    }
    let mut future_server: serde_json::Value =
        serde_json::from_str(&progress_json).expect("valid progress json");
    future_server["future_server_telemetry_field"] = serde_json::json!(42);
    let decoded_future: CarbonPublicWaveProgress = serde_json::from_value(future_server).expect(
        "server->client responses must tolerate additive server fields for forward compatibility",
    );
    assert_eq!(decoded_future, progress);
    let mut legacy_progress: serde_json::Value =
        serde_json::from_str(&progress_json).expect("valid progress json");
    legacy_progress
        .as_object_mut()
        .unwrap()
        .remove("eval_invocations");
    legacy_progress
        .as_object_mut()
        .unwrap()
        .remove("eval_total_wall_ms");
    let decoded_legacy: CarbonPublicWaveProgress = serde_json::from_value(legacy_progress)
        .expect("decode legacy progress without eval counters");
    assert_eq!(decoded_legacy.eval_invocations, DecimalU64(0));
    assert_eq!(decoded_legacy.eval_total_wall_ms, DecimalU64(0));

    let stop = CarbonEarlyStopRequest {
        tenant_id: "tenant-pro".to_owned(),
        reason: Some("user_see_it_now".to_owned()),
    };
    assert_eq!(
        serde_json::from_str::<CarbonEarlyStopRequest>(
            &serde_json::to_string(&stop).expect("encode stop")
        )
        .expect("decode stop"),
        stop
    );

    let resume = CarbonResumeRunRequest {
        tenant_id: "tenant-pro".to_owned(),
        effort: Some(Effort::High),
        additional_waves: Some(4),
        wrapped_u_dek_b64: Some("Q1VESwE=".to_owned()),
    };
    assert_eq!(
        serde_json::from_str::<CarbonResumeRunRequest>(
            &serde_json::to_string(&resume).expect("encode resume")
        )
        .expect("decode resume"),
        resume
    );

    let pred_req = CarbonPredictRequest {
        tenant_id: "tenant-pro".to_owned(),
        feature_names: vec![],
        rows: vec![],
        encrypted_rows_cuen_b64: Some("Q1VFTgEE".to_owned()),
        encrypted_rows_cas_sha256: None,
        predict_upload_id: None,
        prefer_cas_output: false,
        client_response_pubkey_b64: Some("BA==".to_owned()),
        wrapped_u_dek_b64: Some("Q1VESwE=".to_owned()),
    };
    assert_eq!(
        serde_json::from_str::<CarbonPredictRequest>(
            &serde_json::to_string(&pred_req).expect("encode pred_req")
        )
        .expect("decode pred_req"),
        pred_req
    );

    let pred_resp = CarbonPredictResponse {
        run_id: run_id.clone(),
        predictions: vec![],
        predictions_sha256: "e".repeat(64),
        row_count: 2,
        encrypted_predictions_cuen_b64: Some("Q1VFTgEF".to_owned()),
        encrypted_predictions_cas_url: None,
        encrypted_predictions_cas_sha256: None,
        response_cudk_b64: Some("Q1VESwE=".to_owned()),
    };
    assert_eq!(
        serde_json::from_str::<CarbonPredictResponse>(
            &serde_json::to_string(&pred_resp).expect("encode pred_resp")
        )
        .expect("decode pred_resp"),
        pred_resp
    );

    let shred = CarbonShredRunResponse {
        run_id,
        shredded: true,
        objects_deleted: 5,
    };
    assert_eq!(
        serde_json::from_str::<CarbonShredRunResponse>(
            &serde_json::to_string(&shred).expect("encode shred")
        )
        .expect("decode shred"),
        shred
    );

    let ping = CarbonPingResponse {
        region: "eu-west-1".to_owned(),
        healthy: true,
        server_epoch_ms: server_epoch_secs_or_ms(),
        enterprise_self_hosted: false,
        engine_kek_pubkey_b64: Some("BA==".to_owned()),
    };
    assert!(ping.healthy);
}

fn server_epoch_secs_or_ms() -> DecimalU64 {
    DecimalU64(1_762_000_000_000u64)
}

#[test]
fn carbon_region_catalog_covers_all_17_graviton_cells_and_generates_json_schema() {
    let catalog = CarbonRegionCatalog::default_17_region_catalog("carbon.jopedime.com");
    assert_eq!(catalog.catalog_version, 1);
    assert_eq!(catalog.regions.len(), CARBON_DEFAULT_REGIONS.len());
    assert_eq!(catalog.regions.len(), 17);
    assert_eq!(
        catalog.regions[0].endpoint_url,
        "https://us-east-2.carbon.jopedime.com"
    );

    let schema = schemars::schema_for!(CarbonPublicWaveProgress);
    let schema_str = serde_json::to_string(&schema).expect("schema json");
    assert!(schema_str.contains("outer_holdout_loss"));
    assert!(schema_str.contains("bundle_digest"));
    assert!(!schema_str.contains("\"dsl\""));
}

#[test]
fn default_create_run_request_is_byte_compatible_with_frozen_v1_1_0_strict_edge() {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct FrozenV110CreateRunRequest {
        run_id: RunId,
        dataset_id: DatasetId,
        tenant_id: String,
        tier: CarbonTierWire,
        fit_jmat_sha256: String,
        holdout_jmat_sha256: String,
        fit_rows: u32,
        holdout_rows: u32,
        feature_names: Vec<String>,
        target_name: String,
        objective: String,
        effort: Effort,
        seed: DecimalU64,
        landmark_bundle_b64: String,
        #[serde(default)]
        encrypted_cuen: bool,
        #[serde(default)]
        wrapped_u_dek_b64: Option<String>,
        #[serde(default)]
        upload_id: Option<String>,
    }

    let req = CarbonCreateRunRequest {
        run_id: "run_01JCCARB0N0000000000000001".parse().unwrap(),
        dataset_id: "ds_111111111111111111111111".parse().unwrap(),
        tenant_id: "free_alice_01".to_owned(),
        tier: CarbonTierWire::Free,
        fit_jmat_sha256: "a".repeat(64),
        holdout_jmat_sha256: "b".repeat(64),
        fit_rows: 100,
        holdout_rows: 25,
        feature_names: vec!["x0".to_owned()],
        target_name: "y".to_owned(),
        objective: "rmse".to_owned(),
        effort: Effort::Fast,
        seed: DecimalU64(42),
        landmark_bundle_b64: "Q1VFTgE=".to_owned(),
        encrypted_cuen: true,
        wrapped_u_dek_b64: Some("Q1VESwE=".to_owned()),
        upload_id: Some("upl_01".to_owned()),
        aux_column_names: Vec::new(),
        split_mode: None,
        calibration_mode: None,
    };
    let wire = serde_json::to_string(&req).expect("serialize default request");
    assert!(!wire.contains("aux_column_names"));
    assert!(!wire.contains("split_mode"));
    assert!(!wire.contains("calibration_mode"));
    let frozen: FrozenV110CreateRunRequest = serde_json::from_str(&wire)
        .expect("default CarbonCreateRunRequest must deserialize on frozen v1.1.0 strict edge");
    assert_eq!(frozen.run_id, req.run_id);
    assert_eq!(frozen.dataset_id, req.dataset_id);
    assert_eq!(frozen.tenant_id, req.tenant_id);
    assert_eq!(frozen.tier, req.tier);
    assert_eq!(frozen.fit_jmat_sha256, req.fit_jmat_sha256);
    assert_eq!(frozen.holdout_jmat_sha256, req.holdout_jmat_sha256);
    assert_eq!(frozen.fit_rows, req.fit_rows);
    assert_eq!(frozen.holdout_rows, req.holdout_rows);
    assert_eq!(frozen.feature_names, req.feature_names);
    assert_eq!(frozen.target_name, req.target_name);
    assert_eq!(frozen.objective, req.objective);
    assert_eq!(frozen.effort, req.effort);
    assert_eq!(frozen.seed, req.seed);
    assert_eq!(frozen.landmark_bundle_b64, req.landmark_bundle_b64);
    assert_eq!(frozen.encrypted_cuen, req.encrypted_cuen);
    assert_eq!(frozen.wrapped_u_dek_b64, req.wrapped_u_dek_b64);
    assert_eq!(frozen.upload_id, req.upload_id);

    let explain_req = CarbonExplainRequest {
        tenant_id: "free_alice_01".to_owned(),
        top_k: Some(10),
        client_response_pubkey_b64: Some("BA==".to_owned()),
        wrapped_u_dek_b64: Some("Q1VESwE=".to_owned()),
    };
    assert_eq!(
        serde_json::from_str::<CarbonExplainRequest>(&serde_json::to_string(&explain_req).unwrap())
            .unwrap(),
        explain_req
    );
    let explain_resp = CarbonExplainResponse {
        run_id: req.run_id.clone(),
        method: "holdout_permutation_loss_delta".to_owned(),
        baseline_holdout_loss: 0.125,
        feature_importances: vec![CarbonFeatureImportanceWire {
            feature: "x0".to_owned(),
            importance: 1.0,
            delta_loss: 0.85,
        }],
        encrypted_explain_cuen_b64: None,
        response_cudk_b64: None,
    };
    assert_eq!(
        serde_json::from_str::<CarbonExplainResponse>(
            &serde_json::to_string(&explain_resp).unwrap()
        )
        .unwrap(),
        explain_resp
    );

    let export_req = CarbonExportRequest {
        tenant_id: "free_alice_01".to_owned(),
        lang: CarbonExportLangWire::RustRefittable,
        client_response_pubkey_b64: Some("BA==".to_owned()),
        wrapped_u_dek_b64: Some("Q1VESwE=".to_owned()),
    };
    assert_eq!(
        serde_json::from_str::<CarbonExportRequest>(&serde_json::to_string(&export_req).unwrap())
            .unwrap(),
        export_req
    );
    let export_resp = CarbonExportResponse {
        run_id: req.run_id,
        lang: CarbonExportLangWire::Python,
        source_code: None,
        source_sha256: "f".repeat(64),
        encrypted_export_cuen_b64: Some("Q1VFTgEH".to_owned()),
        response_cudk_b64: Some("Q1VESwE=".to_owned()),
    };
    assert_eq!(
        serde_json::from_str::<CarbonExportResponse>(&serde_json::to_string(&export_resp).unwrap())
            .unwrap(),
        export_resp
    );
}

#[test]
fn carbon_v12_prepare_predict_version_export_langs_and_unknown_field_refusal() {
    let prep_pred_req = CarbonPreparePredictRequest {
        tenant_id: "free_alice_01".to_owned(),
        predict_cuen: CarbonUploadSlotWire {
            sha256: "9".repeat(64),
            bytes: DecimalU64(131_072),
        },
    };
    assert_eq!(
        serde_json::from_str::<CarbonPreparePredictRequest>(
            &serde_json::to_string(&prep_pred_req).expect("encode CarbonPreparePredictRequest")
        )
        .expect("decode CarbonPreparePredictRequest"),
        prep_pred_req
    );

    let prep_pred_resp = CarbonPreparePredictResponse {
        upload_id: "upl_pred_01jccarb0n".to_owned(),
        input_target: CarbonPresignedTarget {
            url: "https://us-east-2.carbon.jopedime.com/v1/carbon/cas/999".to_owned(),
            method: "PUT".to_owned(),
            sha256: "9".repeat(64),
            max_bytes: DecimalU64(131_072),
            expires_epoch_secs: DecimalU64(1_800_000_900),
        },
    };
    assert_eq!(
        serde_json::from_str::<CarbonPreparePredictResponse>(
            &serde_json::to_string(&prep_pred_resp).expect("encode CarbonPreparePredictResponse")
        )
        .expect("decode CarbonPreparePredictResponse"),
        prep_pred_resp
    );

    let ver = CarbonVersionResponse {
        version: "1.2.0".to_owned(),
        release_tag: "carbon-cli-v1.2.0".to_owned(),
        git_sha: "7988e2ecd3067647526762774e9bdfc522745990".to_owned(),
        build_target: "aarch64-unknown-linux-gnu".to_owned(),
        region: "us-east-2".to_owned(),
        enterprise_self_hosted: false,
        release_id: Some("rel_01jccarb0n".to_owned()),
        r882_bundle_digest: Some("sha256:057e50a9".to_owned()),
        engine_kek_pubkey_b64: Some("BA==".to_owned()),
        server_epoch_ms: DecimalU64(1_762_000_000_000),
    };
    assert_eq!(
        serde_json::from_str::<CarbonVersionResponse>(
            &serde_json::to_string(&ver).expect("encode CarbonVersionResponse")
        )
        .expect("decode CarbonVersionResponse"),
        ver
    );

    let ping = CarbonPingResponse {
        region: "us-east-2".to_owned(),
        healthy: true,
        server_epoch_ms: DecimalU64(1_762_000_000_000),
        enterprise_self_hosted: true,
        engine_kek_pubkey_b64: None,
    };
    assert_eq!(
        serde_json::from_str::<CarbonPingResponse>(
            &serde_json::to_string(&ping).expect("encode CarbonPingResponse")
        )
        .expect("decode CarbonPingResponse"),
        ping
    );

    for (variant, wire_name) in [
        (CarbonExportLangWire::Python, "python"),
        (CarbonExportLangWire::Rust, "rust"),
        (CarbonExportLangWire::RustRefittable, "rust-refittable"),
        (CarbonExportLangWire::Js, "js"),
        (CarbonExportLangWire::C, "c"),
        (CarbonExportLangWire::Cpp, "cpp"),
    ] {
        assert_eq!(variant.as_str(), wire_name);
        let enc = serde_json::to_string(&variant).expect("encode lang");
        assert_eq!(enc, format!("\"{wire_name}\""));
        assert_eq!(
            serde_json::from_str::<CarbonExportLangWire>(&enc).expect("decode lang"),
            variant
        );
    }

    assert!(
        serde_json::from_str::<CarbonRegisterFreeRequest>(r#"{"label":"ok","extra":1}"#).is_err()
    );
    assert!(serde_json::from_str::<CarbonUploadSlotWire>(&format!(
        r#"{{"sha256":"{}","bytes":10,"extra":1}}"#,
        "a".repeat(64)
    ))
    .is_err());
    assert!(serde_json::from_str::<CarbonPrepareUploadRequest>(
        &format!(
            r#"{{"tenant_id":"t","tier":"free","fit_rows":10,"holdout_rows":5,"feature_count":1,"fit_jmat":{{"sha256":"{0}","bytes":1}},"holdout_jmat":{{"sha256":"{0}","bytes":1}},"extra":true}}"#,
            "a".repeat(64)
        )
    )
    .is_err());
    assert!(serde_json::from_str::<CarbonCreateRunRequest>(
        &format!(
            r#"{{"run_id":"run_01JCCARB0N0000000000000001","dataset_id":"ds_111111111111111111111111","tenant_id":"t","tier":"free","fit_jmat_sha256":"{0}","holdout_jmat_sha256":"{0}","fit_rows":10,"holdout_rows":5,"feature_names":["x"],"target_name":"y","objective":"rmse","effort":"fast","seed":1,"landmark_bundle_b64":"AA==","extra":1}}"#,
            "a".repeat(64)
        )
    )
    .is_err());
    assert!(
        serde_json::from_str::<CarbonEarlyStopRequest>(r#"{"tenant_id":"t","extra":1}"#).is_err()
    );
    assert!(
        serde_json::from_str::<CarbonResumeRunRequest>(r#"{"tenant_id":"t","extra":1}"#).is_err()
    );
    assert!(
        serde_json::from_str::<CarbonPreparePredictRequest>(&format!(
            r#"{{"tenant_id":"t","predict_cuen":{{"sha256":"{}","bytes":1}},"extra":1}}"#,
            "a".repeat(64)
        ))
        .is_err()
    );
    assert!(
        serde_json::from_str::<CarbonPredictRequest>(r#"{"tenant_id":"t","extra":1}"#).is_err()
    );
    assert!(
        serde_json::from_str::<CarbonExplainRequest>(r#"{"tenant_id":"t","extra":1}"#).is_err()
    );
    assert!(serde_json::from_str::<CarbonExportRequest>(
        r#"{"tenant_id":"t","lang":"python","extra":1}"#
    )
    .is_err());

    let schema_str = serde_json::to_string(&schemars::schema_for!(CarbonVersionResponse))
        .expect("CarbonVersionResponse JsonSchema");
    assert!(schema_str.contains("r882_bundle_digest"));

    macro_rules! assert_carbon_json_schema {
        ($($ty:ty),+ $(,)?) => {{
            $(
                let rendered = serde_json::to_string(&schemars::schema_for!($ty))
                    .unwrap_or_else(|e| panic!("JsonSchema failed for {}: {e}", stringify!($ty)));
                assert!(
                    rendered.contains("\"$schema\""),
                    "missing $schema root in {}",
                    stringify!($ty)
                );
            )+
        }};
    }
    assert_carbon_json_schema!(
        CarbonTierWire,
        CarbonRunStateWire,
        CarbonExecutionModeWire,
        CarbonExportLangWire,
        CarbonAnonymousAuthResponse,
        CarbonRegisterFreeRequest,
        CarbonRegisterFreeResponse,
        CarbonUploadSlotWire,
        CarbonPrepareUploadRequest,
        CarbonPresignedTarget,
        CarbonPrepareUploadResponse,
        CarbonCreateRunRequest,
        CarbonEarlyStopRequest,
        CarbonResumeRunRequest,
        CarbonPublicWaveProgress,
        CarbonPreparePredictRequest,
        CarbonPreparePredictResponse,
        CarbonPredictRequest,
        CarbonPredictResponse,
        CarbonFeatureImportanceWire,
        CarbonExplainRequest,
        CarbonExplainResponse,
        CarbonExportRequest,
        CarbonExportResponse,
        CarbonShredRunResponse,
        CarbonPingResponse,
        CarbonVersionResponse,
        CarbonRegionDescriptor,
        CarbonRegionCatalog,
    );
}

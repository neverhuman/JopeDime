use jope_api::{
    carbon::{
        CarbonAnonymousAuthResponse, CarbonCreateRunRequest, CarbonEarlyStopRequest,
        CarbonExecutionModeWire, CarbonPingResponse, CarbonPredictRequest, CarbonPredictResponse,
        CarbonPrepareUploadRequest, CarbonPrepareUploadResponse, CarbonPresignedTarget,
        CarbonPublicWaveProgress, CarbonRegionCatalog, CarbonRegisterFreeRequest,
        CarbonRegisterFreeResponse, CarbonResumeRunRequest, CarbonRunStateWire,
        CarbonShredRunResponse, CarbonTierWire, CarbonUploadSlotWire, CARBON_DEFAULT_REGIONS,
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
    let mut poisoned: serde_json::Value =
        serde_json::from_str(&progress_json).expect("valid progress json");
    poisoned["dsl"] = serde_json::Value::String("x0 + x1".to_owned());
    assert!(
        serde_json::from_value::<CarbonPublicWaveProgress>(poisoned).is_err(),
        "CarbonPublicWaveProgress must reject unknown candidate-structure fields"
    );
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

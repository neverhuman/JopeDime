use jope_api::{
    carbon::{
        CarbonCreateRunRequest, CarbonEarlyStopRequest, CarbonExecutionModeWire,
        CarbonPingResponse, CarbonPredictRequest, CarbonPredictResponse,
        CarbonPrepareUploadRequest, CarbonPrepareUploadResponse, CarbonPresignedTarget,
        CarbonPublicWaveProgress, CarbonRegionCatalog, CarbonResumeRunRequest, CarbonRunStateWire,
        CarbonTierWire, CarbonUploadSlotWire, CARBON_DEFAULT_REGIONS,
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

    let prepare = CarbonPrepareUploadRequest {
        tenant_id: "tenant-pro".to_owned(),
        tier: CarbonTierWire::Pro,
        fit_rows: 1_200,
        holdout_rows: 300,
        feature_count: 12,
        fit_jmat: CarbonUploadSlotWire {
            sha256: "a".repeat(64),
            bytes: DecimalU64(48_000u64),
        },
        holdout_jmat: CarbonUploadSlotWire {
            sha256: "b".repeat(64),
            bytes: DecimalU64(12_000u64),
        },
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
            expires_epoch_secs: DecimalU64(1_800_000_000u64),
        },
        holdout_target: CarbonPresignedTarget {
            url: "https://us-east-2.carbon.jopedime.com/v1/carbon/cas/bbb".to_owned(),
            method: "PUT".to_owned(),
            sha256: "b".repeat(64),
            expires_epoch_secs: DecimalU64(1_800_000_000u64),
        },
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
        landmark_bundle_b64: "SkxNQg==".to_owned(),
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
        eval_shards: 1,
        wave_index: 3,
        waves_target: 8,
        candidates_evaluated: DecimalU64(96u64),
        best_cv_loss: Some(0.142),
        best_cv_metric: Some(0.142),
        outer_holdout_loss: Some(0.149),
        outer_holdout_metric: Some(0.149),
        neural_seed_applied: true,
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
        feature_names: vec!["x0".to_owned(), "x1".to_owned()],
        rows: vec![vec![Some(1.5), None], vec![Some(-2.0), Some(3.25)]],
    };
    assert_eq!(
        serde_json::from_str::<CarbonPredictRequest>(
            &serde_json::to_string(&pred_req).expect("encode pred_req")
        )
        .expect("decode pred_req"),
        pred_req
    );

    let pred_resp = CarbonPredictResponse {
        run_id,
        predictions: vec![4.25, 1.75],
        predictions_sha256: "e".repeat(64),
        row_count: 2,
    };
    assert_eq!(
        serde_json::from_str::<CarbonPredictResponse>(
            &serde_json::to_string(&pred_resp).expect("encode pred_resp")
        )
        .expect("decode pred_resp"),
        pred_resp
    );

    let ping = CarbonPingResponse {
        region: "eu-west-1".to_owned(),
        healthy: true,
        server_epoch_ms: server_epoch_secs_or_ms(),
        enterprise_self_hosted: false,
    };
    assert!(ping.healthy);
}

fn server_epoch_secs_or_ms() -> DecimalU64 {
    DecimalU64(1_762_000_000_000u64)
}

#[test]
fn carbon_region_catalog_covers_all_17_graviton_cells_and_generates_json_schema() {
    let catalog = CarbonRegionCatalog::default_17_region_catalog("carbon.jopedime.com");
    assert_eq!(CARBON_DEFAULT_REGIONS.len(), 17);
    assert_eq!(catalog.regions.len(), 17);
    assert_eq!(catalog.regions[0].region, "us-east-2");
    assert_eq!(
        catalog.regions[0].endpoint_url,
        "https://us-east-2.carbon.jopedime.com"
    );
    assert_eq!(catalog.regions[16].region, "me-central-1");

    let schema = schemars::schema_for!(CarbonPublicWaveProgress);
    let schema_text = serde_json::to_string(&schema).expect("serialize schema");
    assert!(schema_text.contains("outer_holdout_loss"));
    assert!(!schema_text.contains("\"dsl\""));
}

use jope_campaigns::router_contracts_v2::*;

fn hash(byte: u8) -> String {
    (0..32).map(|_| format!("{byte:02x}")).collect()
}

fn metrics() -> Vec<SeedRouteMetricV2> {
    [303, 404]
        .into_iter()
        .map(|seed| SeedRouteMetricV2 {
            seed,
            raw_loss: 1.0,
            routed_loss: 0.98,
            relative_improvement: 0.02,
            fold_wins: 7,
            valid: true,
        })
        .collect()
}

#[test]
fn per_family_member_routes_can_mix_raw_and_cell() {
    let decisions = vec![
        FamilyRouteDecisionV2 {
            schema: FamilyRouteDecisionV2::SCHEMA.into(),
            family_id: "catboost".into(),
            selected_route: PredictorRouteV2::cell_only(),
            matched_raw_fallback: PredictorRouteV2::raw(),
            per_seed_metrics: metrics(),
            blend_coefficient: None,
            latency_ms: 1.0,
            valid: true,
            receipt_sha256: hash(b'a'),
        },
        FamilyRouteDecisionV2 {
            schema: FamilyRouteDecisionV2::SCHEMA.into(),
            family_id: "lightgbm".into(),
            selected_route: PredictorRouteV2::raw(),
            matched_raw_fallback: PredictorRouteV2::raw(),
            per_seed_metrics: metrics(),
            blend_coefficient: None,
            latency_ms: 1.0,
            valid: true,
            receipt_sha256: hash(b'b'),
        },
    ];
    let members = vec![
        MemberInputPlanV2 {
            schema: MemberInputPlanV2::SCHEMA.into(),
            member_id: "catboost-0".into(),
            family_id: "catboost".into(),
            route: PredictorRouteV2::cell_only(),
            raw_width: 4,
            predictor_sha256: hash(b'c'),
        },
        MemberInputPlanV2 {
            schema: MemberInputPlanV2::SCHEMA.into(),
            member_id: "lightgbm-0".into(),
            family_id: "lightgbm".into(),
            route: PredictorRouteV2::raw(),
            raw_width: 4,
            predictor_sha256: hash(b'd'),
        },
    ];
    validate_family_ensemble_v2(&decisions, &members).unwrap();
    assert_eq!(members[0].input_width().unwrap(), 4);
}

#[test]
fn vector_batches_and_legacy_mappings_are_safe() {
    let batch = PredictionBatchV2::new(
        2,
        2,
        vec![0.8, 0.2, 0.1, 0.9],
        vec!["no".into(), "yes".into()],
    )
    .unwrap();
    assert_eq!(batch.row(1).unwrap(), &[0.1, 0.9]);
    assert!(batch.class_order_sha256().unwrap().is_some());
    assert!(PredictionBatchV2::new(2, 2, vec![0.1], vec![]).is_err());
    let artifact = ModelArtifactV4::from_legacy_v3(
        "multiclass",
        3,
        vec!["a".into(), "b".into(), "c".into()],
        hash(b'e'),
    )
    .unwrap();
    assert!(artifact.selected_route.is_raw());
    let manifest =
        RunManifestV6::from_legacy_v5("multiclass", 3, artifact.class_label_names, hash(b'f'))
            .unwrap();
    assert!(manifest.selected_route.is_raw());
}

#[test]
fn blend_is_clipped_and_zero_denominators_fall_back_to_raw() {
    let fit = fit_raw_biased_blend_v2(&[0.0, 1.0], &[0.0, 0.0], &[0.0, 1.0]).unwrap();
    assert_eq!(fit.alpha, 0.5);
    assert_eq!(
        apply_raw_biased_blend_v2(&[0.0, 0.0], &[0.0, 1.0], fit).unwrap(),
        vec![0.0, 0.5]
    );
    let unstable = fit_raw_biased_blend_v2(&[1.0, 1.0], &[0.0, 0.0], &[0.0, 0.0]).unwrap();
    assert_eq!(unstable.alpha, 0.0);
    assert!(!unstable.stable);
}

#[test]
fn time_series_folds_keep_timestamps_atomic_and_are_expanding() {
    let times = (0..80).map(|row| (row / 2) as i64).collect::<Vec<_>>();
    let groups = (0..80)
        .map(|row| format!("atomic-{row}"))
        .collect::<Vec<_>>();
    let scheme = ValidationSchemeV2::TimeSeries {
        time_column: "time".into(),
        seasonal_period: 1,
        purge_groups: 0,
        embargo_groups: 0,
    };
    let folds = build_validation_folds_v2(Some(&times), &groups, &scheme).unwrap();
    assert_eq!(folds.len(), 7);
    for fold in &folds {
        assert!(fold.fit_rows.len() >= 32);
        assert!(fold.hold_rows.len() >= 2);
        assert!(fold
            .fit_rows
            .iter()
            .all(|row| !fold.hold_rows.contains(row)));
        assert!(
            fold.fit_rows.iter().map(|row| times[*row]).max().unwrap()
                < fold.hold_rows.iter().map(|row| times[*row]).min().unwrap()
        );
    }
    assert_eq!(
        predictor_columns_without_time(&["time".into(), "x".into()], "time").unwrap(),
        vec!["x".to_string()]
    );
    assert!(
        (mase_v2(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0], &[4.0, 7.0], 1).unwrap() - 1.0).abs() < 1e-12
    );
}

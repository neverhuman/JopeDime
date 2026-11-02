#[test]
fn budget_ladder_is_nested_resumable_and_rule_frozen() {
    let bundle = ladder(true);
    bundle.validate().unwrap();
    let replay = bundle.clone();
    verify_prefix_equality(&bundle, &replay, 1_000).unwrap();

    let mut bad_prefix = replay;
    bad_prefix.checkpoints[3]
        .prefix_prediction_roots
        .insert(1_000, digest("changed-prefix"));
    assert!(bad_prefix.validate().is_err());

    let stopped = ladder(false);
    stopped.validate().unwrap();
    let mut illicit_ten_k = stopped;
    illicit_ten_k
        .checkpoints
        .push(bundle.checkpoints[3].clone());
    assert!(illicit_ten_k.validate().is_err());
}

#[test]
fn component_predictors_bind_inference_and_license_tracks() {
    let research = FrozenPredictorV1::ComponentResidualComposite {
        base: tabfm(),
        residual: FrozenPredictorComponentV1::ExternalModel {
            family: ExternalModelFamilyV1::GpControl,
            artifact_sha256: digest("gp-control"),
            artifact_bytes: 1,
            inference_protocol_sha256: digest("gp-control-inference"),
            license: LicenseIdentityV1 {
                spdx_expression: "Apache-2.0".into(),
                ..license()
            },
            complexity: 3,
        },
        eta: 0.5,
        cross_fit_protocol_sha256: digest("cross-fit"),
        license_track: LicenseTrackV1::ResearchOnly,
    };
    research.validate().unwrap();
    assert_eq!(
        research
            .reconstruct(&[vec![1.0, 2.0], vec![2.0, -2.0]])
            .unwrap(),
        vec![2.0, 1.0]
    );
    let decoded: FrozenPredictorV1 =
        serde_json::from_slice(&serde_json::to_vec(&research).unwrap()).unwrap();
    assert_eq!(decoded.semantic_sha256(), research.semantic_sha256());

    let mut foreign = research;
    if let FrozenPredictorV1::ComponentResidualComposite { license_track, .. } = &mut foreign {
        *license_track = LicenseTrackV1::ProductionCompatible;
    }
    assert!(foreign.validate().unwrap_err().contains("research-only"));

    let roots = TrackArtifactNamespacesV1 {
        production_compatible_root: "/artifacts/production".into(),
        research_only_root: "/artifacts/research".into(),
    };
    roots
        .validate_artifact(
            LicenseTrackV1::ProductionCompatible,
            "/artifacts/production/result.json",
        )
        .unwrap();
    assert!(roots
        .validate_artifact(
            LicenseTrackV1::ProductionCompatible,
            "/artifacts/research/result.json",
        )
        .is_err());
}

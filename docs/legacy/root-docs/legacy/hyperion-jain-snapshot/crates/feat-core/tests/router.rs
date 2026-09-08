use feat_core::router::{
    certify_margin, family_smooth_lifts, route_algorithm, route_dataset, starlight_anchor_index,
    tiny_elo, DatasetSketch, RouteFamily, RouterArtifacts, RouterConfig,
};
use std::path::PathBuf;

fn fixture() -> RouterArtifacts {
    RouterArtifacts {
        algorithms: vec![
            "catboost".into(),
            "xgboost_hist".into(),
            "starlight_core_classifier".into(),
        ],
        train_features: vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ],
        train_utilities: vec![
            vec![0.70, 0.68, 0.71],
            vec![0.72, 0.73, 0.70],
            vec![0.69, 0.68, 0.75],
            vec![0.71, 0.70, 0.74],
        ],
        train_source_families: vec!["a".into(), "a".into(), "b".into(), "b".into()],
        family_names: vec!["catboost".into(), "starlight_core".into(), "xgboost".into()],
        family_ids: vec![0, 2, 1],
        standardize: true,
    }
}

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(name);
    p
}

#[test]
fn family_smoothing_moves_lifts_toward_family_mean() {
    let mut lifts = vec![1.0, 0.0, 0.2];
    family_smooth_lifts(&mut lifts, &[0, 0, 1], 0.5);
    assert!((lifts[0] - 0.75).abs() < 1e-9);
    assert!((lifts[1] - 0.25).abs() < 1e-9);
    assert!((lifts[2] - 0.2).abs() < 1e-9);
}

#[test]
fn starlight_anchor_and_route_mapping_are_internal() {
    let art = fixture();
    let prior = art.prior();
    let anchor = starlight_anchor_index(&art.algorithms, &prior);
    assert_eq!(art.algorithms[anchor], "starlight_core_classifier");
    let mapped = route_algorithm("starlight_core_classifier_real", Some("catboost"));
    assert_eq!(mapped.selected, "starlight_core");
    assert_eq!(mapped.family, RouteFamily::StarlightCore);
    let unsupported = route_algorithm("sklearn_svc", Some("catboost"));
    assert_eq!(unsupported.selected, "catboost");
    assert_eq!(unsupported.family, RouteFamily::Unsupported);
    assert!(unsupported.audit.iter().any(|entry| {
        entry == "unsupported router algorithm 'sklearn_svc' mapped to default route 'catboost'"
    }));
    assert_eq!(route_algorithm("lightgbm_gbdt", None).selected, "lightgbm");
}

#[test]
fn v301_routes_and_certifies_anchor() {
    let art = fixture();
    let sketch = DatasetSketch {
        features: vec![0.0, 1.0],
        source_family: "b".into(),
    };
    let cfg = RouterConfig {
        enabled: true,
        z: 0.0,
        ..RouterConfig::default()
    };
    let decision = route_dataset(&art, &sketch, &cfg).unwrap();
    assert_eq!(decision.selected_label, "CVBEST_t0.002_s0.12_z0.0");
    assert_eq!(decision.route.selected, "starlight_core");
    assert!(certify_margin(&decision.scores, decision.anchor_index, 0.0));
}

#[test]
fn router_artifact_loads_shape_and_tiny_elo_scores() {
    let art = fixture();
    let p = tmp("jain_router_artifact.json");
    std::fs::write(&p, serde_json::to_string(&art).unwrap()).unwrap();
    let loaded = RouterArtifacts::load(&p).unwrap();
    assert_eq!(loaded.algorithms.len(), 3);
    assert_eq!(loaded.train_features.len(), 4);
    assert!(tiny_elo(&[0, 1, 2], &[0, 2, 2]) > 0.0);
}

#[test]
#[ignore]
fn external_best_npz_parity_placeholder() {
    let p = std::path::Path::new(
        "/home/ubuntu/remote_super/quant/models/tools/artifacts/binary_router_cache/best.npz",
    );
    if p.exists() {
        // The production router intentionally does not depend on an NPZ reader or Python. This ignored
        // test is the hook for full-environment parity fixtures when that compact JSON export exists.
        assert!(p.is_file());
    }
}

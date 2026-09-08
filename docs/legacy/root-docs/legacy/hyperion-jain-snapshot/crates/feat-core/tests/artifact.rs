use feat_core::artifact::{
    materialize_selected, materialize_selected_for_domain, read_manifest, transform_scoring_csv,
    transform_scoring_csv_text, write_manifest, LoadedArtifact, ModelManifest, SavedFuser,
    SavedMember,
};
use feat_core::backend::{Backend, TrainSpec};
use feat_core::buildmatrix::FeatureInfo;
use feat_core::canary::{fit_specs, generate_column, CanaryDomain};
use feat_core::data::{load_train_hold, Dataset, Matrix};
use feat_core::dsl::{Bin, Expr};
use feat_core::encode::CatEncoding;
use feat_core::ensemble::FuserMode;
use feat_core::hyperion::{
    hyperion_output_names, HyperionMode, HyperionModelIdentity, HyperionQuantileBounds,
    HyperionState, HyperionSweepSettings, HYPERION_VERSION,
};
use feat_core::starforge_integration::SavedStarforgeMember;
use std::io::Write;
use std::path::PathBuf;

fn write_tmp(name: &str, content: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(name);
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    p
}

fn unique_dir(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("jain-{name}-{}-{nanos}", std::process::id()))
}

#[test]
fn manifest_accepts_native_and_starforge_members() {
    let manifest_json = r#"{
      "version": 1,
      "target": "y",
      "targets": ["y"],
      "task": {
        "kind": "Binary",
        "n_targets": 1,
        "n_classes": 2,
        "class_values": [0.0, 1.0],
        "prevalence": 0.5
      },
      "base_names": ["x"],
      "encoders": {
        "version": 2,
        "policy": "Hybrid",
        "low_cardinality_max": 32,
        "default_hash_buckets": 64,
        "width_cap": 20000,
        "missing": {
          "numeric_fill_scaled": 0.0,
          "categorical_missing_token": "__MISSING__",
          "categorical_other_token": "__OTHER__"
        },
        "columns": [{
          "input_name": "x",
          "kind": "Numeric",
          "encoding": {"Numeric": {"scaler": {"min": 0.0, "max": 1.0, "fill": 0.0, "constant": false}}},
          "output_names": ["x"]
        }],
        "target_scaler": {"min": 0.0, "max": 1.0, "scale": 1.0},
        "issues": []
      },
      "selected_cols": [0],
      "features": [{
        "source": "base",
        "formula": "x",
        "expr": {"Col": 0},
        "fill": 0.0
      }],
      "fuser": {
        "name": "mixed",
        "rank_outputs": false,
        "members": [
          {"name": "catboost", "backend": "catboost", "weight": 0.6, "model_file": "models/cat.cbm"},
          {
            "name": "starforge_top500",
            "backend": "starforge",
            "weight": 0.4,
            "model_file": "starforge/context.json",
            "starforge": {
              "selected_cols": [0],
              "weights_path": "/opt/jain/starforge/starlight_core.safetensors",
              "context_file": "starforge/context.json",
              "dtype": "bf16",
              "context_rows": 32,
              "n_features": 1
            }
          }
        ]
      }
    }"#;
    let manifest: ModelManifest = serde_json::from_str(manifest_json).unwrap();
    assert_eq!(manifest.fuser.mode, FuserMode::WeightedBlend);
    assert_eq!(manifest.fuser.members.len(), 2);
    assert!(manifest.fuser.members[0].starforge.is_none());
    assert_eq!(
        manifest.fuser.members[1]
            .starforge
            .as_ref()
            .unwrap()
            .context_rows,
        32
    );

    let encoded = serde_json::to_string(&manifest).unwrap();
    let round: ModelManifest = serde_json::from_str(&encoded).unwrap();
    assert_eq!(
        round.fuser.members[1].starforge.as_ref().unwrap().dtype,
        "bf16"
    );

    let native = SavedMember {
        name: "native".into(),
        backend: "catboost".into(),
        weight: 1.0,
        model_file: "models/native.cbm".into(),
        starforge: None,
    };
    assert!(serde_json::to_value(native)
        .unwrap()
        .get("starforge")
        .is_none());
    let _sf = SavedStarforgeMember {
        selected_cols: vec![0],
        weights_path: "weights.safetensors".into(),
        context_file: "starforge/context.json".into(),
        dtype: "f32".into(),
        context_rows: 4,
        n_features: 1,
        task: feat_core::starforge_integration::StarforgeContextTask::Classifier,
        regression_engine: None,
        chimera_weights_path: None,
        tabicl_regressor_weights_path: None,
    };
}

#[test]
fn residual_fuser_metadata_round_trips() {
    let fuser_json = r#"{
      "name": "catboost_residual_starforge",
      "mode": "residual",
      "rank_outputs": false,
      "members": [],
      "residual": {
        "base_member": "catboost",
        "residual_member": "starforge_residual_regressor",
        "alpha": 0.35,
        "clamp_min": 0.000001,
        "clamp_max": 0.999999
      }
    }"#;
    let fuser: SavedFuser = serde_json::from_str(fuser_json).unwrap();
    assert_eq!(fuser.mode, FuserMode::Residual);
    assert_eq!(fuser.residual.as_ref().unwrap().base_member, "catboost");
    let encoded = serde_json::to_string(&fuser).unwrap();
    let round: SavedFuser = serde_json::from_str(&encoded).unwrap();
    assert_eq!(round.mode, FuserMode::Residual);
    assert!((round.residual.unwrap().alpha - 0.35).abs() < 1e-12);
}

#[test]
fn artifact_manifest_and_preprocessing_replay_round_trip() {
    let tr = write_tmp(
        "jain_art_tr.csv",
        "num,region,y\n1,EAST,0\n2,WEST,1\n3,EAST,0\n4,NORTH,1\n",
    );
    let ho = write_tmp("jain_art_ho.csv", "num,region,y\n5,EAST,0\n6,WEST,1\n");
    let pair = load_train_hold(&tr, &ho, &["y"], "y", CatEncoding::Hybrid).unwrap();

    let features: Vec<FeatureInfo> = pair
        .train
        .names
        .iter()
        .enumerate()
        .map(|(j, name)| FeatureInfo {
            source: "base".into(),
            formula: name.clone(),
            expr: Expr::Col(j as u16),
            fill: 0.0,
            scale_min: 0.0,
            scale_span: 1.0,
            canary: None,
            canary_refs: Vec::new(),
        })
        .collect();
    let manifest = ModelManifest::new(
        "y".into(),
        vec!["y".into()],
        pair.task.clone(),
        pair.train.names.clone(),
        pair.encoders.clone(),
        (0..pair.train.names.len() as u32).collect(),
        features,
        SavedFuser {
            name: "test".into(),
            mode: FuserMode::WeightedBlend,
            rank_outputs: false,
            members: vec![],
            residual: None,
        },
    );
    let mut dir = std::env::temp_dir();
    dir.push("jain_artifact_round_trip");
    std::fs::remove_dir_all(&dir).ok();
    write_manifest(&dir, &manifest).unwrap();
    let loaded = read_manifest(&dir).unwrap();
    assert_eq!(loaded.version, feat_core::artifact::ARTIFACT_VERSION);

    let score = write_tmp("jain_art_score.csv", "num,region\n5,EAST\n,UNKNOWN\n");
    let base = transform_scoring_csv(&score, &loaded).unwrap();
    let base_text = transform_scoring_csv_text("num,region\n5,EAST\n,UNKNOWN\n", &loaded).unwrap();
    assert_eq!(base_text.data.len(), base.data.len());
    for (left, right) in base_text.data.iter().zip(base.data.iter()) {
        assert!(left == right || (left.is_nan() && right.is_nan()));
    }
    let selected = materialize_selected(&base, &loaded.features, &loaded.selected_cols).unwrap();
    assert_eq!(selected.n, 2);
    assert_eq!(selected.c, loaded.base_names.len());
    assert!(selected
        .data
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
    assert_eq!(selected.col(0), &[1.0, 0.0]);
    let other = loaded
        .base_names
        .iter()
        .position(|n| n == "region::__OTHER__")
        .unwrap();
    assert_eq!(selected.col(other), &[0.0, 1.0]);
}

#[test]
fn math_artifact_model_round_trips_with_prediction_parity() {
    let tr = write_tmp(
        "jain_art_math_tr.csv",
        "x0,x1,y\n0.0,0.0,0\n0.2,0.1,0\n0.8,0.7,1\n1.0,0.9,1\n",
    );
    let ho = write_tmp("jain_art_math_ho.csv", "x0,x1,y\n0.1,0.0,0\n0.9,0.8,1\n");
    let pair = load_train_hold(&tr, &ho, &["y"], "y", CatEncoding::Hybrid).unwrap();
    let features: Vec<FeatureInfo> = pair
        .train
        .names
        .iter()
        .enumerate()
        .map(|(j, name)| FeatureInfo {
            source: "base".into(),
            formula: name.clone(),
            expr: Expr::Col(j as u16),
            fill: 0.0,
            scale_min: 0.0,
            scale_span: 1.0,
            canary: None,
            canary_refs: Vec::new(),
        })
        .collect();
    let selected_cols: Vec<u32> = (0..features.len() as u32).collect();
    let native_train =
        feat_core::artifact::materialize_selected_train(&pair.train.x, &features, &selected_cols)
            .unwrap();
    let native_hold = materialize_selected(&pair.hold.x, &features, &selected_cols).unwrap();
    let math = feat_core::learners::math::MathBackend::from_dsl("(pipe X (ridge))").unwrap();
    let model = math.train_single(
        native_train.data.as_ref().as_slice(),
        native_train.n,
        native_train.c,
        &pair.train.y,
        &TrainSpec::default(),
    );
    let expected = feat_core::task::model_predict(
        &*model,
        native_hold.data.as_ref().as_slice(),
        native_hold.n,
        native_hold.c,
        pair.task.kind,
        pair.task.n_classes,
    );

    let dir = unique_dir("artifact-math-roundtrip");
    let rel = "models/00_math_ridge.json";
    model.save_native(&dir.join(rel)).unwrap();
    let manifest = ModelManifest::new(
        "y".into(),
        vec!["y".into()],
        pair.task.clone(),
        pair.train.names.clone(),
        pair.encoders.clone(),
        selected_cols,
        features,
        SavedFuser {
            name: "math_ridge".into(),
            mode: FuserMode::WeightedBlend,
            rank_outputs: false,
            members: vec![SavedMember {
                name: "math_ridge".into(),
                backend: "math".into(),
                weight: 1.0,
                model_file: rel.into(),
                starforge: None,
            }],
            residual: None,
        },
    );
    write_manifest(&dir, &manifest).unwrap();

    let loaded = LoadedArtifact::load(&dir).unwrap();
    assert_eq!(loaded.manifest.fuser.members[0].backend, "math");
    let actual = loaded.predict_base_matrix(pair.hold.x.clone()).unwrap();
    assert_eq!(actual, expected);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn canary_features_materialize_and_manifest_round_trip() {
    let base = Matrix::new(vec![0.0, 1.0, 0.0, 1.0, 0.2, 0.4, 0.6, 0.8], 4, 2);
    let ds = Dataset {
        x: base.clone(),
        y: vec![0.0, 1.0, 0.0, 1.0],
        y_multi: None,
        names: vec!["x0".into(), "x1".into()],
    };
    let spec = fit_specs(&ds, 1234).remove(0);
    let canary_col = spec.search_width.max(spec.source_index + 1) + spec.canary_index;
    assert_eq!(canary_col, 2);

    let direct = FeatureInfo {
        source: "canary".into(),
        formula: spec.name.clone(),
        expr: Expr::Col(canary_col as u16),
        fill: 0.0,
        scale_min: 0.0,
        scale_span: 1.0,
        canary: Some(spec.clone()),
        canary_refs: Vec::new(),
    };
    let gp_ref = FeatureInfo {
        source: "gp".into(),
        formula: format!("({} + 0)", spec.name),
        expr: Expr::Bin(
            Bin::Add,
            Box::new(Expr::Col(canary_col as u16)),
            Box::new(Expr::Const(0.0)),
        ),
        fill: 0.0,
        scale_min: 0.0,
        scale_span: 1.0,
        canary: None,
        canary_refs: vec![spec.clone()],
    };
    let features = vec![direct, gp_ref];

    let train =
        materialize_selected_for_domain(&base, &features, &[0, 1], CanaryDomain::Train).unwrap();
    assert_eq!(train.n, 4);
    assert_eq!(train.c, 2);
    let expected_train = generate_column(&spec, base.n, CanaryDomain::Train)
        .into_iter()
        .map(|v| {
            if v.is_finite() {
                v.clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(train.col(0), expected_train.as_slice());
    assert_eq!(train.col(1), expected_train.as_slice());

    let predict = materialize_selected(&base, &features, &[1]).unwrap();
    let expected_predict = generate_column(&spec, base.n, CanaryDomain::Predict)
        .into_iter()
        .map(|v| {
            if v.is_finite() {
                v.clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(predict.col(0), expected_predict.as_slice());

    let tr = write_tmp("jain_art_canary_tr.csv", "x0,x1,y\n0,0.2,0\n1,0.4,1\n");
    let ho = write_tmp("jain_art_canary_ho.csv", "x0,x1,y\n0,0.6,0\n1,0.8,1\n");
    let pair = load_train_hold(&tr, &ho, &["y"], "y", CatEncoding::Hybrid).unwrap();
    let manifest = ModelManifest::new(
        "y".into(),
        vec!["y".into()],
        pair.task.clone(),
        pair.train.names.clone(),
        pair.encoders.clone(),
        vec![0, 1],
        features,
        SavedFuser {
            name: "test".into(),
            mode: FuserMode::WeightedBlend,
            rank_outputs: false,
            members: vec![],
            residual: None,
        },
    );
    let dir = unique_dir("artifact-canary-manifest");
    std::fs::remove_dir_all(&dir).ok();
    write_manifest(&dir, &manifest).unwrap();
    let loaded = read_manifest(&dir).unwrap();
    assert_eq!(loaded.features[0].canary.as_ref(), Some(&spec));
    assert_eq!(loaded.features[1].canary_refs, vec![spec]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn scoring_fails_when_artifact_requires_unavailable_hyperion() {
    let tr = write_tmp("jain_art_hyp_tr.csv", "x,y\n1,0\n2,1\n3,0\n4,1\n");
    let ho = write_tmp("jain_art_hyp_ho.csv", "x,y\n5,0\n6,1\n");
    let pair = load_train_hold(&tr, &ho, &["y"], "y", CatEncoding::Hybrid).unwrap();
    let mut base_names = pair.train.names.clone();
    base_names.extend(hyperion_output_names(128));
    let features = vec![FeatureInfo {
        source: "base".into(),
        formula: "hyperion::v000".into(),
        expr: Expr::Col(pair.train.names.len() as u16),
        fill: 0.0,
        scale_min: 0.0,
        scale_span: 1.0,
        canary: None,
        canary_refs: Vec::new(),
    }];
    let state = HyperionState {
        version: HYPERION_VERSION.into(),
        requested_mode: HyperionMode::On,
        output_names: hyperion_output_names(128),
        input_names: pair.encoders.output_names(),
        input_indices: (0..pair.encoders.output_width()).collect(),
        quantile_bounds: vec![HyperionQuantileBounds {
            low: 0.0,
            high: 1.0,
        }],
        sweep: HyperionSweepSettings {
            version: HYPERION_VERSION.into(),
            step: 2000,
            dim: 512,
            heads: 8,
            layers: 12,
            emb_dim: 512,
            min_patch: 2,
            max_patch: 24,
            n_bands: 1,
            q_low: 1.0,
            q_high: 99.0,
            quant_scale: 3.0,
            pad_u8: 128,
            ws_list: vec![4],
            seed: 1,
        },
        output_dim: 128,
        model: HyperionModelIdentity {
            weights_path: "/definitely/missing/hyperion.safetensors".into(),
            manifest_path: "/definitely/missing/manifest.json".into(),
            weights_bytes: 0,
            weights_hash: "missing".into(),
            manifest_hash: "missing".into(),
            tensor_count: 0,
        },
        engine: "legacy".into(),
        sweeps: 1,
        scales: Vec::new(),
        batch: 0,
    };
    let manifest = ModelManifest::new(
        "y".into(),
        vec!["y".into()],
        pair.task.clone(),
        base_names,
        pair.encoders.clone(),
        vec![0],
        features,
        SavedFuser {
            name: "test".into(),
            mode: FuserMode::WeightedBlend,
            rank_outputs: false,
            members: vec![],
            residual: None,
        },
    )
    .with_hyperion(Some(state));
    let score = write_tmp("jain_art_hyp_score.csv", "x\n5\n");
    let err = match transform_scoring_csv(&score, &manifest) {
        Ok(_) => panic!("Hyperion-required scoring unexpectedly succeeded"),
        Err(err) => err,
    };
    assert!(
        format!("{err:#}").contains("Hyperion"),
        "error should name Hyperion: {err:#}"
    );
}

#[test]
fn artifact_load_reports_failure_modes() {
    let missing = unique_dir("artifact-missing-manifest");
    std::fs::remove_dir_all(&missing).ok();
    std::fs::create_dir_all(&missing).unwrap();
    let err = match LoadedArtifact::load(&missing) {
        Ok(_) => panic!("missing manifest unexpectedly loaded"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("manifest.json"));

    let tr = write_tmp(
        "jain_art_fail_tr.csv",
        "num,region,y\n1,EAST,0\n2,WEST,1\n3,EAST,0\n4,NORTH,1\n",
    );
    let ho = write_tmp("jain_art_fail_ho.csv", "num,region,y\n5,EAST,0\n6,WEST,1\n");
    let pair = load_train_hold(&tr, &ho, &["y"], "y", CatEncoding::Hybrid).unwrap();
    let features: Vec<FeatureInfo> = pair
        .train
        .names
        .iter()
        .enumerate()
        .map(|(j, name)| FeatureInfo {
            source: "base".into(),
            formula: name.clone(),
            expr: Expr::Col(j as u16),
            fill: 0.0,
            scale_min: 0.0,
            scale_span: 1.0,
            canary: None,
            canary_refs: Vec::new(),
        })
        .collect();
    let base_manifest = ModelManifest::new(
        "y".into(),
        vec!["y".into()],
        pair.task.clone(),
        pair.train.names.clone(),
        pair.encoders.clone(),
        (0..pair.train.names.len() as u32).collect(),
        features,
        SavedFuser {
            name: "test".into(),
            mode: FuserMode::WeightedBlend,
            rank_outputs: false,
            members: vec![],
            residual: None,
        },
    );

    let empty_dir = unique_dir("artifact-empty-fuser");
    std::fs::remove_dir_all(&empty_dir).ok();
    write_manifest(&empty_dir, &base_manifest).unwrap();
    let err = match LoadedArtifact::load(&empty_dir) {
        Ok(_) => panic!("empty fuser unexpectedly loaded"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("no members"));

    let mut unsupported = base_manifest;
    unsupported.fuser.members.push(SavedMember {
        name: "unsupported".into(),
        backend: "unknown_backend".into(),
        weight: 1.0,
        model_file: "models/missing.bin".into(),
        starforge: None,
    });
    let unsupported_dir = unique_dir("artifact-unsupported-backend");
    std::fs::remove_dir_all(&unsupported_dir).ok();
    write_manifest(&unsupported_dir, &unsupported).unwrap();
    let err = match LoadedArtifact::load(&unsupported_dir) {
        Ok(_) => panic!("unsupported backend unexpectedly loaded"),
        Err(err) => format!("{err:#}"),
    };
    assert!(err.contains("unsupported persisted backend"));
}

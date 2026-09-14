use feat_core::artifact::{materialize_selected, ModelManifest, SavedFuser};
use feat_core::buildmatrix::FeatureInfo;
use feat_core::data::{load_train_hold, load_train_split, Matrix};
use feat_core::dsl::Expr;
use feat_core::encode::CatEncoding;
use feat_core::ensemble::FuserMode;
use feat_core::ingest::read_header;
use feat_core::metrics::task_score;
use feat_core::router::{route_algorithm, RouteFamily};
use feat_core::starforge_integration::{
    read_context, write_context, StarforgeConfig, StarforgeDevice, StarforgeDiagnostics,
    StarforgeMode,
};
use feat_core::task::TaskKind;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

fn tmp_file(name: &str, body: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    path.push(format!("jain-core-api-{}-{id}-{name}", std::process::id()));
    std::fs::write(&path, body).unwrap();
    path
}

#[test]
fn data_encoding_artifact_metrics_and_diagnostics_stay_rust_only() {
    let train = tmp_file(
        "train.csv",
        "num,num,city,target\n\
         1,10,EAST,0\n\
         2,,WEST,1\n\
         3,30,EAST,0\n\
         4,40,NORTH,1\n",
    );
    let hold = tmp_file(
        "hold.csv",
        "num,num,city,target\n\
         5,50,EAST,0\n\
         6,,UNKNOWN,1\n",
    );

    let header = read_header(&train).unwrap();
    assert_eq!(header, ["num", "num__dup1", "city", "target"]);

    let pair = load_train_hold(&train, &hold, &["target"], "target", CatEncoding::Hybrid)
        .expect("load typed train/hold");
    assert_eq!(pair.task.kind, TaskKind::Binary);
    assert!(pair
        .cat_cols
        .iter()
        .any(|&j| pair.train.names[j].starts_with("city")));
    assert!(pair
        .train
        .x
        .data
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));

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
    let manifest = ModelManifest::new(
        "target".into(),
        vec!["target".into()],
        pair.task.clone(),
        pair.train.names.clone(),
        pair.encoders.clone(),
        selected_cols.clone(),
        features.clone(),
        SavedFuser {
            name: "empty_contract".into(),
            mode: FuserMode::WeightedBlend,
            rank_outputs: false,
            members: Vec::new(),
            residual: None,
        },
    );
    let selected = materialize_selected(&pair.hold.x, &manifest.features, &selected_cols).unwrap();
    assert_eq!(selected.n, pair.hold.n());
    assert_eq!(selected.c, pair.train.p());

    let perfect = task_score(TaskKind::Binary, &[0.0, 1.0], &[0.1, 0.9], 2);
    assert!((perfect - 1.0).abs() < 1e-9);

    let starlight = route_algorithm("starlight_core_classifier", None);
    assert_eq!(starlight.family, RouteFamily::StarlightCore);
    assert_eq!(starlight.selected, "starlight_core");

    let diag = StarforgeDiagnostics::for_config(&StarforgeConfig::disabled());
    assert_eq!(diag.compiled, cfg!(feature = "starforge-cpu"));
    assert_eq!(diag.cuda_compiled, cfg!(feature = "starforge-cuda"));
    let cpu_mode_diag = StarforgeDiagnostics::for_config(&StarforgeConfig {
        mode: StarforgeMode::On,
        device: StarforgeDevice::Auto,
        ..StarforgeConfig::disabled()
    });
    if !cpu_mode_diag.cuda_available {
        assert_eq!(
            cpu_mode_diag.warning.as_deref(),
            Some(
                "CUDA is unavailable; Starforge will use CPU execution mode unless GPU is required"
            )
        );
    }
}

#[test]
fn regression_target_scaler_round_trips_values() {
    let train = tmp_file(
        "reg-train.csv",
        "x,group,target\n1,A,10.5\n2,B,20.25\n3,A,30.75\n4,B,40.5\n",
    );
    let hold = tmp_file("reg-hold.csv", "x,group,target\n5,A,50.5\n6,C,60.25\n");
    let pair = load_train_hold(&train, &hold, &["target"], "target", CatEncoding::Hybrid).unwrap();

    assert_eq!(pair.task.kind, TaskKind::Regression);
    assert_eq!(pair.encoders.target_scaler.min, 10.5);
    assert_eq!(pair.encoders.target_scaler.max, 40.5);
    assert!((pair.encoders.target_scaler.inverse(pair.train.y[0]) - 10.5).abs() < 1e-6);
    assert!((pair.encoders.target_scaler.inverse(pair.train.y[3]) - 40.5).abs() < 1e-6);
}

#[test]
fn starforge_context_helpers_round_trip_without_inference() {
    let x = Matrix::new(
        vec![
            1.0, 2.0, 3.0, 4.0, // col 0
            10.0, 20.0, 30.0, 40.0, // col 1
        ],
        4,
        2,
    );
    let y = vec![0.0, 1.0, 1.0, 0.0];
    let path = tmp_file("starforge-context.json", "");
    let ctx = write_context(&path, &x, &y, &[1, 0], 3, 7, TaskKind::Binary).unwrap();
    assert_eq!(ctx.n_features, 2);
    assert!(ctx.n_rows <= 3);
    assert_eq!(ctx.x.len(), ctx.n_rows * ctx.n_features);
    assert_eq!(ctx.y.len(), ctx.n_rows);

    let loaded = read_context(&path).unwrap();
    assert_eq!(loaded.selected_cols, vec![1, 0]);
    assert_eq!(loaded.x, ctx.x);
    assert_eq!(loaded.y, ctx.y);

    std::fs::write(
        &path,
        r#"{"selected_cols":[0],"n_rows":2,"n_features":2,"x":[1.0],"y":[0,1]}"#,
    )
    .unwrap();
    let err = read_context(&path).unwrap_err().to_string();
    assert!(err.contains("invalid Starforge context shape"));
}

#[test]
fn internal_holdout_split_handles_tiny_and_degenerate_targets() {
    let binary = tmp_file("split-binary.csv", "x,target\n1,0\n2,0\n");
    let pair =
        load_train_split(&binary, &["target"], "target", CatEncoding::Hybrid, 0.8, 11).unwrap();
    assert_eq!(pair.train.n() + pair.hold.n(), 2);
    assert!(pair.train.n() > 0);
    assert!(pair.hold.n() > 0);

    let regression = tmp_file("split-regression.csv", "x,target\n1,1.25\n2,2.50\n3,3.75\n");
    let pair = load_train_split(
        &regression,
        &["target"],
        "target",
        CatEncoding::Hybrid,
        0.99,
        12,
    )
    .unwrap();
    assert_eq!(pair.task.kind, TaskKind::Regression);
    assert_eq!(pair.train.n() + pair.hold.n(), 3);
    assert!(pair.train.n() > 0);
    assert!(pair.hold.n() > 0);
}

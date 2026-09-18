use anyhow::{Context, Result};
use candle_core::{DType, Device};
use serde::Deserialize;
use std::path::PathBuf;

use starforge::model::{ChimeraClassifier, ChimeraRegressor, StarforgeModel};

#[derive(Deserialize)]
struct Fixture {
    ctx_x: Vec<Vec<f32>>,
    ctx_y: Vec<u8>,
    qry_x: Vec<Vec<f32>>,
    predict_single: Vec<f64>,
    predict_ensemble_auc: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct RegressionFixture {
    ctx_x: Vec<Vec<f32>>,
    ctx_y: Vec<f32>,
    qry_x: Vec<Vec<f32>>,
    predict_regression_single: Vec<f64>,
}

#[derive(Deserialize)]
struct ChimeraRegressionFixture {
    ctx_x: Vec<Vec<f32>>,
    ctx_y: Vec<f32>,
    qry_x: Vec<Vec<f32>>,
    predict_chimera: Vec<f64>,
}

#[derive(Deserialize)]
struct ChimeraClassificationFixture {
    ctx_x: Vec<Vec<f32>>,
    ctx_y: Vec<u8>,
    qry_x: Vec<Vec<f32>>,
    predict_chimera: Vec<f64>,
}

#[test]
fn predict_single_matches_golden_fixture_when_artifacts_are_provided() -> Result<()> {
    let Some(weights) = std::env::var_os("STARFORGE_WEIGHTS") else {
        eprintln!("skipping golden parity: STARFORGE_WEIGHTS is not set");
        return Ok(());
    };
    let Some(golden) = std::env::var_os("STARFORGE_GOLDEN_JSON") else {
        eprintln!("skipping golden parity: STARFORGE_GOLDEN_JSON is not set");
        return Ok(());
    };

    let weights = resolve_repo_path(PathBuf::from(weights));
    let golden = resolve_repo_path(PathBuf::from(golden));

    let fixture_text = std::fs::read_to_string(&golden)
        .with_context(|| format!("reading {}", golden.display()))?;
    let fixture: Fixture = serde_json::from_str(&fixture_text)?;
    let (ctx_x, n_features) = flatten_rows(&fixture.ctx_x)?;
    let (qry_x, query_features) = flatten_rows(&fixture.qry_x)?;
    assert_eq!(n_features, query_features);
    assert_eq!(fixture.ctx_y.len() * n_features, ctx_x.len());
    assert_eq!(fixture.predict_single.len() * n_features, qry_x.len());

    let model = StarforgeModel::load_dtype(&weights, Device::Cpu, DType::F32)?;
    let actual = model.predict_single(&ctx_x, &fixture.ctx_y, &qry_x, n_features)?;
    assert_eq!(actual.len(), fixture.predict_single.len());

    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    for (got, want) in actual.iter().zip(&fixture.predict_single) {
        let err = (got - want).abs();
        max_abs = max_abs.max(err);
        sum_abs += err;
    }
    let mean_abs = sum_abs / actual.len().max(1) as f64;
    assert!(
        max_abs <= 5e-4 && mean_abs <= 1e-4,
        "golden parity mismatch: max_abs={max_abs:.8}, mean_abs={mean_abs:.8}"
    );
    if let Some(expected) = fixture.predict_ensemble_auc {
        let actual = model.predict_ensemble_auc(&ctx_x, &fixture.ctx_y, &qry_x, n_features)?;
        assert_eq!(actual.len(), expected.len());
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(got, want)| (got - want).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_abs <= 1e-3,
            "golden ensemble parity mismatch: max_abs={max_abs:.8}"
        );
    }
    Ok(())
}

#[test]
fn predict_regression_single_matches_golden_fixture_when_artifacts_are_provided() -> Result<()> {
    let Some(weights) = std::env::var_os("STARFORGE_REGRESSOR_WEIGHTS") else {
        eprintln!("skipping regression golden parity: STARFORGE_REGRESSOR_WEIGHTS is not set");
        return Ok(());
    };
    let Some(golden) = std::env::var_os("STARFORGE_REGRESSOR_GOLDEN_JSON") else {
        eprintln!("skipping regression golden parity: STARFORGE_REGRESSOR_GOLDEN_JSON is not set");
        return Ok(());
    };

    let weights = resolve_repo_path(PathBuf::from(weights));
    let golden = resolve_repo_path(PathBuf::from(golden));

    let fixture_text = std::fs::read_to_string(&golden)
        .with_context(|| format!("reading {}", golden.display()))?;
    let fixture: RegressionFixture = serde_json::from_str(&fixture_text)?;
    let (ctx_x, n_features) = flatten_rows(&fixture.ctx_x)?;
    let (qry_x, query_features) = flatten_rows(&fixture.qry_x)?;
    assert_eq!(n_features, query_features);
    assert_eq!(fixture.ctx_y.len() * n_features, ctx_x.len());
    assert_eq!(
        fixture.predict_regression_single.len() * n_features,
        qry_x.len()
    );

    let model = StarforgeModel::load_dtype(&weights, Device::Cpu, DType::F32)?;
    let actual = model.predict_regression_single(&ctx_x, &fixture.ctx_y, &qry_x, n_features)?;
    assert_eq!(actual.len(), fixture.predict_regression_single.len());

    assert_close(&actual, &fixture.predict_regression_single)?;
    Ok(())
}

#[test]
fn predict_chimera_regression_matches_golden_fixture_when_artifacts_are_provided() -> Result<()> {
    let Some(starlight) = std::env::var_os("STARFORGE_REGRESSOR_WEIGHTS") else {
        eprintln!("skipping Chimera golden parity: STARFORGE_REGRESSOR_WEIGHTS is not set");
        return Ok(());
    };
    let Some(tabicl) = std::env::var_os("TABICL_REGRESSOR_WEIGHTS") else {
        eprintln!("skipping Chimera golden parity: TABICL_REGRESSOR_WEIGHTS is not set");
        return Ok(());
    };
    let Some(chimera) = std::env::var_os("STARFORGE_CHIMERA_WEIGHTS") else {
        eprintln!("skipping Chimera golden parity: STARFORGE_CHIMERA_WEIGHTS is not set");
        return Ok(());
    };
    let Some(golden) = std::env::var_os("STARFORGE_CHIMERA_GOLDEN_JSON") else {
        eprintln!("skipping Chimera golden parity: STARFORGE_CHIMERA_GOLDEN_JSON is not set");
        return Ok(());
    };

    let starlight = resolve_repo_path(PathBuf::from(starlight));
    let tabicl = resolve_repo_path(PathBuf::from(tabicl));
    let chimera = resolve_repo_path(PathBuf::from(chimera));
    let golden = resolve_repo_path(PathBuf::from(golden));

    let fixture_text = std::fs::read_to_string(&golden)
        .with_context(|| format!("reading {}", golden.display()))?;
    let fixture: ChimeraRegressionFixture = serde_json::from_str(&fixture_text)?;
    let (ctx_x, n_features) = flatten_rows(&fixture.ctx_x)?;
    let (qry_x, query_features) = flatten_rows(&fixture.qry_x)?;
    assert_eq!(n_features, query_features);
    assert_eq!(fixture.ctx_y.len() * n_features, ctx_x.len());
    assert_eq!(fixture.predict_chimera.len() * n_features, qry_x.len());

    let smoke_device =
        std::env::var("STARFORGE_CHIMERA_SMOKE_DEVICE").unwrap_or_else(|_| "cpu".into());
    let (device, dtype) = match smoke_device.as_str() {
        "cuda" | "cuda:0" => (Device::new_cuda(0)?, DType::BF16),
        "cpu" => (Device::Cpu, DType::F32),
        other => anyhow::bail!("unsupported STARFORGE_CHIMERA_SMOKE_DEVICE={other}"),
    };
    let model = ChimeraRegressor::load_dtype(&starlight, &tabicl, &chimera, device, dtype)?;
    let actual = model.predict_regression_single(&ctx_x, &fixture.ctx_y, &qry_x, n_features)?;
    assert_eq!(actual.len(), fixture.predict_chimera.len());

    assert_close_tol(&actual, &fixture.predict_chimera, 1e-3, 2e-4)?;
    Ok(())
}

#[test]
fn predict_chimera_regression_real_weights_smoke_when_requested() -> Result<()> {
    if std::env::var("STARFORGE_CHIMERA_SMOKE").ok().as_deref() != Some("1") {
        eprintln!("skipping Chimera real-weight smoke: STARFORGE_CHIMERA_SMOKE=1 is not set");
        return Ok(());
    }
    let starlight = resolve_repo_path(PathBuf::from(
        std::env::var("STARFORGE_REGRESSOR_WEIGHTS")
            .unwrap_or_else(|_| "artifacts/starforge/starlight_core_regressor.safetensors".into()),
    ));
    let tabicl = resolve_repo_path(PathBuf::from(
        std::env::var("TABICL_REGRESSOR_WEIGHTS").unwrap_or_else(|_| {
            "artifacts/foundation/tabicl-regressor-v2-20260212.safetensors".into()
        }),
    ));
    let chimera = resolve_repo_path(PathBuf::from(
        std::env::var("STARFORGE_CHIMERA_WEIGHTS")
            .unwrap_or_else(|_| "artifacts/starforge/chimera_stackfusionreg.safetensors".into()),
    ));
    let smoke_device =
        std::env::var("STARFORGE_CHIMERA_SMOKE_DEVICE").unwrap_or_else(|_| "cpu".into());
    let (device, dtype) = match smoke_device.as_str() {
        "cuda" | "cuda:0" => (Device::new_cuda(0)?, DType::BF16),
        "cpu" => (Device::Cpu, DType::F32),
        other => anyhow::bail!("unsupported STARFORGE_CHIMERA_SMOKE_DEVICE={other}"),
    };
    let model = ChimeraRegressor::load_dtype(&starlight, &tabicl, &chimera, device, dtype)?;
    let ctx_x = vec![
        0.0, 0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.7, 0.8, 0.6, 0.7, 0.8,
        0.9, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 1.0, 0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1, 0.0, 0.1, 0.9,
        0.8, 0.7, 0.6, 0.5,
    ];
    let ctx_y = vec![0.05, 0.18, 0.31, 0.44, 0.57, 0.63, 0.21, 0.48];
    let qx = vec![0.15, 0.25, 0.35, 0.45, 0.55, 0.55, 0.45, 0.35, 0.25, 0.15];
    let pred = model.predict_regression_single(&ctx_x, &ctx_y, &qx, 5)?;
    anyhow::ensure!(pred.len() == 2, "expected two Chimera predictions");
    anyhow::ensure!(
        pred.iter().all(|v| v.is_finite()),
        "Chimera smoke produced non-finite predictions: {pred:?}"
    );
    emit_gpu_qualification_marker(
        "regression",
        &smoke_device,
        pred.len(),
        pred.iter().all(|value| value.is_finite()),
    )?;
    eprintln!("Chimera smoke device={smoke_device} predictions: {pred:?}");
    Ok(())
}

#[test]
fn predict_chimera_classifier_matches_golden_fixture_when_artifacts_are_provided() -> Result<()> {
    let Some(starlight) = std::env::var_os("STARFORGE_CHIMERA_CLASSIFIER_EXPERT_A_WEIGHTS") else {
        eprintln!(
            "skipping Chimera classifier golden parity: STARFORGE_CHIMERA_CLASSIFIER_EXPERT_A_WEIGHTS is not set"
        );
        return Ok(());
    };
    let Some(tabicl) = std::env::var_os("TABICL_CLASSIFIER_WEIGHTS") else {
        eprintln!(
            "skipping Chimera classifier golden parity: TABICL_CLASSIFIER_WEIGHTS is not set"
        );
        return Ok(());
    };
    let Some(chimera) = std::env::var_os("STARFORGE_CHIMERA_CLASSIFIER_TRUNK_WEIGHTS") else {
        eprintln!(
            "skipping Chimera classifier golden parity: STARFORGE_CHIMERA_CLASSIFIER_TRUNK_WEIGHTS is not set"
        );
        return Ok(());
    };
    let Some(golden) = std::env::var_os("STARFORGE_CHIMERA_CLASSIFIER_GOLDEN_JSON") else {
        eprintln!(
            "skipping Chimera classifier golden parity: STARFORGE_CHIMERA_CLASSIFIER_GOLDEN_JSON is not set"
        );
        return Ok(());
    };

    let starlight = resolve_repo_path(PathBuf::from(starlight));
    let tabicl = resolve_repo_path(PathBuf::from(tabicl));
    let chimera = resolve_repo_path(PathBuf::from(chimera));
    let golden = resolve_repo_path(PathBuf::from(golden));

    let fixture_text = std::fs::read_to_string(&golden)
        .with_context(|| format!("reading {}", golden.display()))?;
    let fixture: ChimeraClassificationFixture = serde_json::from_str(&fixture_text)?;
    let (ctx_x, n_features) = flatten_rows(&fixture.ctx_x)?;
    let (qry_x, query_features) = flatten_rows(&fixture.qry_x)?;
    assert_eq!(n_features, query_features);
    assert_eq!(fixture.ctx_y.len() * n_features, ctx_x.len());
    assert_eq!(fixture.predict_chimera.len() * n_features, qry_x.len());

    let smoke_device =
        std::env::var("STARFORGE_CHIMERA_CLASSIFIER_SMOKE_DEVICE").unwrap_or_else(|_| "cpu".into());
    let (device, dtype) = match smoke_device.as_str() {
        "cuda" | "cuda:0" => (Device::new_cuda(0)?, DType::BF16),
        "cpu" => (Device::Cpu, DType::F32),
        other => anyhow::bail!("unsupported STARFORGE_CHIMERA_CLASSIFIER_SMOKE_DEVICE={other}"),
    };
    let model = ChimeraClassifier::load_dtype(&starlight, &tabicl, &chimera, device, dtype)?;
    let actual = model.predict_classification_single(&ctx_x, &fixture.ctx_y, &qry_x, n_features)?;
    assert_eq!(actual.len(), fixture.predict_chimera.len());

    assert_close_tol(&actual, &fixture.predict_chimera, 2e-3, 5e-4)?;
    Ok(())
}

#[test]
fn predict_chimera_classifier_real_weights_smoke_when_requested() -> Result<()> {
    if std::env::var("STARFORGE_CHIMERA_CLASSIFIER_SMOKE")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!(
            "skipping Chimera classifier real-weight smoke: STARFORGE_CHIMERA_CLASSIFIER_SMOKE=1 is not set"
        );
        return Ok(());
    }
    let starlight = resolve_repo_path(PathBuf::from(
        std::env::var("STARFORGE_CHIMERA_CLASSIFIER_EXPERT_A_WEIGHTS").unwrap_or_else(|_| {
            "artifacts/starforge/chimera_classification_expert_a.safetensors".into()
        }),
    ));
    let tabicl = resolve_repo_path(PathBuf::from(
        std::env::var("TABICL_CLASSIFIER_WEIGHTS").unwrap_or_else(|_| {
            "artifacts/foundation/tabicl-classifier-v2-chimera-shuffled.safetensors".into()
        }),
    ));
    let chimera = resolve_repo_path(PathBuf::from(
        std::env::var("STARFORGE_CHIMERA_CLASSIFIER_TRUNK_WEIGHTS").unwrap_or_else(|_| {
            "artifacts/starforge/chimera_classification_attnfusion.safetensors".into()
        }),
    ));
    let smoke_device =
        std::env::var("STARFORGE_CHIMERA_CLASSIFIER_SMOKE_DEVICE").unwrap_or_else(|_| "cpu".into());
    let (device, dtype) = match smoke_device.as_str() {
        "cuda" | "cuda:0" => (Device::new_cuda(0)?, DType::BF16),
        "cpu" => (Device::Cpu, DType::F32),
        other => anyhow::bail!("unsupported STARFORGE_CHIMERA_CLASSIFIER_SMOKE_DEVICE={other}"),
    };
    let model = ChimeraClassifier::load_dtype(&starlight, &tabicl, &chimera, device, dtype)?;
    let ctx_x = vec![
        0.0, 0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.7, 0.8, 0.6, 0.7, 0.8,
        0.9, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 1.0, 0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1, 0.0, 0.1, 0.9,
        0.8, 0.7, 0.6, 0.5,
    ];
    let ctx_y = vec![0, 0, 0, 1, 1, 1, 0, 1];
    let qx = vec![0.15, 0.25, 0.35, 0.45, 0.55, 0.55, 0.45, 0.35, 0.25, 0.15];
    let pred = model.predict_classification_single(&ctx_x, &ctx_y, &qx, 5)?;
    anyhow::ensure!(
        pred.len() == 2,
        "expected two Chimera classifier predictions"
    );
    anyhow::ensure!(
        pred.iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
        "Chimera classifier smoke produced invalid probabilities: {pred:?}"
    );
    emit_gpu_qualification_marker(
        "classifier",
        &smoke_device,
        pred.len(),
        pred.iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
    )?;
    eprintln!("Chimera classifier smoke device={smoke_device} predictions: {pred:?}");
    Ok(())
}

fn resolve_repo_path(path: PathBuf) -> PathBuf {
    if path.is_absolute() || path.exists() {
        return path;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path)
}

fn flatten_rows(rows: &[Vec<f32>]) -> Result<(Vec<f32>, usize)> {
    let n_features = rows.first().map_or(0, Vec::len);
    let mut flat = Vec::with_capacity(rows.len() * n_features);
    for row in rows {
        if row.len() != n_features {
            anyhow::bail!("ragged fixture matrix");
        }
        flat.extend_from_slice(row);
    }
    Ok((flat, n_features))
}

fn emit_gpu_qualification_marker(
    model: &str,
    smoke_device: &str,
    predictions: usize,
    output_valid: bool,
) -> Result<()> {
    if std::env::var("STARFORGE_GPU_QUALIFICATION").ok().as_deref() != Some("1") {
        return Ok(());
    }
    anyhow::ensure!(
        matches!(model, "regression" | "classifier"),
        "unsupported GPU qualification model {model:?}"
    );
    anyhow::ensure!(
        smoke_device == "cuda:0",
        "GPU qualification requires explicit CUDA device 0, got {smoke_device:?}"
    );
    anyhow::ensure!(predictions > 0, "GPU qualification produced no predictions");
    anyhow::ensure!(output_valid, "GPU qualification produced invalid output");
    eprintln!(
        "STARFORGE_GPU_QUALIFICATION model={model} backend=cuda device=0 predictions={predictions} output_valid=true"
    );
    Ok(())
}

fn assert_close(actual: &[f64], expected: &[f64]) -> Result<()> {
    assert_close_tol(actual, expected, 5e-4, 1e-4)
}

fn assert_close_tol(actual: &[f64], expected: &[f64], max_tol: f64, mean_tol: f64) -> Result<()> {
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    for (got, want) in actual.iter().zip(expected) {
        let err = (got - want).abs();
        max_abs = max_abs.max(err);
        sum_abs += err;
    }
    let mean_abs = sum_abs / actual.len().max(1) as f64;
    anyhow::ensure!(
        max_abs <= max_tol && mean_abs <= mean_tol,
        "golden parity mismatch: max_abs={max_abs:.8}, mean_abs={mean_abs:.8}"
    );
    Ok(())
}

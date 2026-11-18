//! CLI execution (`carbon run`) and Free + Pro `r882` zero-knowledge live Lambda benchmark harness (`carbon bench`).

use crate::error::CarbonErrCtx as _;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use base64::Engine as _;
use jope_api::{
    carbon::{
        CarbonExecutionModeWire, CarbonPublicWaveProgress, CarbonRunStateWire, CarbonTierWire,
    },
    ids::{DatasetId, RunId},
    Effort,
};
use jope_comparators::{rmse as comp_rmse, FeatureMatrix, RidgeModel, RidgeParams};
use jope_prep::{compute_binary_holdout_metrics, ObjectiveSpec};
use serde::{Deserialize, Serialize};

use crate::{
    cells::engine_invoker::now_epoch_ms,
    cli::{
        bench_suite::{compute_quick_run_comparators, CarbonComparatorScore},
        client::{CarbonClientPreparedDataset, CarbonProbedEndpoint, CarbonThinClient},
        model_store::{
            compute_residual_quantiles, persist_model_handle_with_options,
            CarbonFeatureImportanceEntry, CarbonResidualQuantiles, CarbonSavedModelHandle,
            CARBON_MODEL_HANDLE_SCHEMA,
        },
        tabular::{
            load_delimited_table, load_delimited_table_with_options,
            prepare_encoded_training_set_with_options, CarbonFeatureTransform,
            CarbonTabularEncoder, CarbonTrainingPrepOptions,
        },
    },
    modes::planner::{
        compute_live_run_cost, estimate_ms_per_candidate_1vcpu, plan_carbon_execution,
        CarbonExecutionMode, CarbonIngressCostMode, CarbonLiveCostInputs, CarbonPlanInput,
        CarbonRunCostBreakdown,
    },
    observability::{log_warn, sigint_received, stderr_is_tty},
    role::{CarbonEffortTier, CarbonProductTier},
    storage::cas::sha256_lower_hex,
};

/// Resolve `--objective` / `--fitness` CLI input, supporting `@path/to/objective.json` file loading
/// as well as inline JSON specs and preset names (`rmse`, `mae`, `huber`, `asymmetric_mse`, etc.).
pub fn resolve_objective_argument(raw: &str) -> Result<(String, ObjectiveSpec), String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        let spec = ObjectiveSpec::parse("rmse").map_err(crate::error::err_to_string)?;
        return Ok(("rmse".to_owned(), spec));
    }
    let loaded = if let Some(path_part) = trimmed.strip_prefix('@') {
        let p = path_part.trim();
        if p.is_empty() {
            return Err("empty file path after '@' in --objective/--fitness".to_owned());
        }
        std::fs::read_to_string(p).map_err(|e| format!("read objective spec file '{p}': {e}"))?
    } else {
        trimmed.to_owned()
    };
    let spec = ObjectiveSpec::parse(&loaded).err_ctx("invalid objective spec")?;
    let wire_text = spec.to_wire_objective();
    if wire_text.len() > 65_536 {
        return Err(format!(
            "serialized objective spec exceeds 64 KiB wire limit ({} bytes)",
            wire_text.len()
        ));
    }
    Ok((wire_text, spec))
}

/// Canonical schema identifier for `carbon fit --format json` and `carbon run --format json` (`v1`).
pub const CARBON_FIT_RECEIPT_SCHEMA: &str = "jope.carbon.fit_receipt.v1";
/// Monotonic integer schema version for [`CarbonCliRunReceipt`].
pub const CARBON_FIT_RECEIPT_SCHEMA_VERSION: u32 = 1;

fn default_fit_receipt_schema() -> String {
    CARBON_FIT_RECEIPT_SCHEMA.to_owned()
}

fn default_fit_receipt_schema_version() -> u32 {
    CARBON_FIT_RECEIPT_SCHEMA_VERSION
}

fn default_split_mode_random() -> String {
    "random".to_owned()
}

/// Compute a deterministic domain-separated SHA-256 hex digest over the ordered holdout rows
/// (`0`-based original row index, IEEE-754 feature vector, and target scalar) shared by Carbon's
/// `holdout.jmat` evaluation and same-split `comparator_baselines` (`baseline-protocol/v1`).
pub fn compute_canonical_holdout_rows_sha256(
    holdout_indices: &[usize],
    holdout_x: &[Vec<f64>],
    holdout_y: &[f64],
) -> String {
    use sha2::{Digest as _, Sha256};

    const CANONICAL_NAN_BITS: u64 = 0x7ff8_0000_0000_0000;
    let canon_f64 = |v: f64| -> [u8; 8] {
        let bits = if v.is_nan() {
            CANONICAL_NAN_BITS
        } else if v == 0.0 {
            0u64
        } else {
            v.to_bits()
        };
        bits.to_le_bytes()
    };

    let n = holdout_indices
        .len()
        .min(holdout_x.len())
        .min(holdout_y.len());
    let feat_count = holdout_x.first().map_or(0usize, Vec::len);
    let mut h = Sha256::new();
    h.update(b"jope.carbon.holdout_rows.v1\0");
    h.update((n as u64).to_le_bytes());
    h.update((feat_count as u64).to_le_bytes());
    for i in 0..n {
        h.update((holdout_indices[i] as u64).to_le_bytes());
        for &xv in &holdout_x[i] {
            h.update(canon_f64(xv));
        }
        h.update(canon_f64(holdout_y[i]));
    }
    hex::encode(h.finalize())
}

struct ShredOnDropGuard {
    client: CarbonThinClient,
    tenant_id: String,
    run_id: String,
    upload_id: Option<String>,
    fit_sha256: Option<String>,
    holdout_sha256: Option<String>,
    active: bool,
}

impl ShredOnDropGuard {
    fn new(
        client: CarbonThinClient,
        tenant_id: String,
        run_id: String,
        upload_id: Option<String>,
        fit_sha256: Option<String>,
        holdout_sha256: Option<String>,
        active: bool,
    ) -> Self {
        Self {
            client,
            tenant_id,
            run_id,
            upload_id,
            fit_sha256,
            holdout_sha256,
            active,
        }
    }

    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for ShredOnDropGuard {
    fn drop(&mut self) {
        if self.active && !self.run_id.is_empty() {
            let _ = self.client.shred_run_with_upload(
                &self.tenant_id,
                &self.run_id,
                self.upload_id.as_deref(),
                self.fit_sha256.as_deref(),
                self.holdout_sha256.as_deref(),
            );
        }
    }
}

type SpinnerProgressTuple = (Option<CarbonRunStateWire>, u32, u32, u64, Option<f64>);

struct TtySearchSpinner {
    stop: Arc<std::sync::atomic::AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl TtySearchSpinner {
    fn start(enabled: bool, shared: Arc<Mutex<SpinnerProgressTuple>>) -> Self {
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        if !enabled {
            return Self { stop, handle: None };
        }
        let stop_bg = Arc::clone(&stop);
        let t0 = Instant::now();
        let handle = thread::spawn(move || {
            const FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
            let mut tick = 0usize;
            while !stop_bg.load(std::sync::atomic::Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(250));
                if stop_bg.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                if let Ok(guard) = shared.lock() {
                    let (st, w_idx, w_tgt, cands, cv) = *guard;
                    let frame = FRAMES[tick % FRAMES.len()];
                    tick = tick.wrapping_add(1);
                    let st_lbl = st
                        .map(|s| format!("{s:?}"))
                        .unwrap_or_else(|| "Starting".to_owned());
                    let cv_lbl = cv
                        .map(|v| format!("{v:.4}"))
                        .unwrap_or_else(|| "-".to_owned());
                    eprint!(
                        "\r\x1b[2K[carbon] {frame} Wave {w_idx}/{w_tgt} ({st_lbl}) | candidates={cands} | best_cv={cv_lbl} | {:.1}s",
                        t0.elapsed().as_secs_f64()
                    );
                    let _ = std::io::Write::flush(&mut std::io::stderr());
                }
            }
            eprint!("\r\x1b[2K");
            let _ = std::io::Write::flush(&mut std::io::stderr());
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }

    fn finish(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for TtySearchSpinner {
    fn drop(&mut self) {
        self.finish();
    }
}

/// `CarbonCliRunUxConfig` is a struct supplying optional pre-probed KEK-pinned client reuse,
/// explicit `--target` detection, universal CSV options, model-handle persistence, and TTY telemetry.
#[derive(Clone, Default)]
pub struct CarbonCliRunUxConfig {
    pub pinned_keks: BTreeMap<String, String>,
    pub pre_probed: Option<(CarbonThinClient, CarbonProbedEndpoint)>,
    pub target_was_explicit: bool,
    pub show_stderr_progress: bool,
    pub delimiter_override: Option<char>,
    pub no_header: bool,
    pub persist_handle: bool,
    pub save_model_path: Option<std::path::PathBuf>,
    pub cloud_name: Option<String>,
    pub split_mode: Option<String>,
    pub calibration_mode: Option<String>,
    pub extra_na_values: Vec<String>,
    pub inf_as_nan: bool,
    pub embed_token: bool,
    pub dry_run: bool,
    pub allow_unencrypted_enterprise: bool,
}

/// `CarbonCliRunOptions` is a struct configuring a single `carbon run` CLI invocation.
#[derive(Clone, Debug)]
pub struct CarbonCliRunOptions {
    pub endpoints: Vec<String>,
    pub csv_path: Option<std::path::PathBuf>,
    pub target_column: String,
    pub synthetic_rows: usize,
    pub synthetic_features: usize,
    pub tenant_id: String,
    pub tier: CarbonTierWire,
    pub effort: Effort,
    pub objective: String,
    pub seed: u64,
    pub holdout_fraction: f64,
    pub stop_after_ms: Option<u64>,
    pub resume_after_stop: bool,
    pub predict_rows: usize,
    pub auth_token: Option<String>,
    pub shred_after_run: bool,
}

/// `CarbonCliRunReceipt` is a struct recording timings, execution mode, zero-knowledge encryption,
/// `r882` neural fusion guidance telemetry, scores, and sample predictions for `carbon run`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonCliRunReceipt {
    #[serde(default = "default_fit_receipt_schema")]
    pub schema: String,
    #[serde(default = "default_fit_receipt_schema_version")]
    pub schema_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release_id: Option<String>,
    pub selected_endpoint: String,
    pub selected_region: String,
    pub ping_rtt_ms: f64,
    pub run_id: String,
    pub dataset_id: String,
    pub tenant_id: String,
    pub tier_applied: Option<CarbonTierWire>,
    pub encrypted_cuen: bool,
    pub neural_seed_applied: bool,
    pub neural_guidance_mode: Option<String>,
    pub bundle_digest: Option<String>,
    pub shredded_after_run: bool,
    pub total_rows: usize,
    pub fit_rows: u32,
    pub holdout_rows: u32,
    pub feature_count: usize,
    #[serde(default = "default_split_mode_random")]
    pub split_mode: String,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub holdout_rows_sha256: String,
    #[serde(default)]
    pub holdout_jmat_sha256: String,
    pub jlmb_wire_bytes: usize,
    pub fit_jmat_bytes: usize,
    pub holdout_jmat_bytes: usize,
    pub prep_ms: f64,
    pub upload_ms: f64,
    pub search_ms: f64,
    pub predict_ms: f64,
    pub total_wall_ms: f64,
    pub final_state: CarbonRunStateWire,
    pub paused_mid_run: bool,
    pub execution_mode: CarbonExecutionModeWire,
    pub eval_memory_mb: u32,
    pub eval_shards: u32,
    #[serde(default)]
    pub eval_invocations: u64,
    #[serde(default)]
    pub eval_total_wall_ms: u64,
    #[serde(default)]
    pub poll_requests: u32,
    pub waves_completed: u32,
    pub waves_target: u32,
    pub candidates_evaluated: u64,
    pub best_cv_score: Option<f64>,
    pub outer_holdout_score: Option<f64>,
    #[serde(default)]
    pub holdout_r2: Option<f64>,
    #[serde(default)]
    pub mean_baseline_rmse: Option<f64>,
    #[serde(default)]
    pub linear_baseline_rmse: Option<f64>,
    #[serde(default)]
    pub linear_baseline_r2: Option<f64>,
    #[serde(default)]
    pub rmse_reduction_vs_mean_pct: Option<f64>,
    #[serde(default)]
    pub rmse_reduction_vs_linear_pct: Option<f64>,
    #[serde(default)]
    pub e2e_predict_holdout_rmse: Option<f64>,
    #[serde(default)]
    pub e2e_predict_holdout_r2: Option<f64>,
    #[serde(default)]
    pub cost_estimate: Option<CarbonRunCostBreakdown>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_quantiles: Option<CarbonResidualQuantiles>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub comparator_baselines: Vec<CarbonComparatorScore>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub feature_importances: Vec<CarbonFeatureImportanceEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saved_model_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective_direction: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_target: Option<CarbonBinaryHoldoutSummary>,
    #[serde(default)]
    pub sample_targets: Vec<f64>,
    pub sample_predictions: Vec<f64>,
}

/// `CarbonBinaryHoldoutSummary` reports learned class labels and probability metrics when a
/// 2-class categorical/boolean target was conditioned into `{0.0, 1.0}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CarbonBinaryHoldoutSummary {
    pub negative_label: String,
    pub positive_label: String,
    pub positive_rate_fit: f64,
    pub positive_rate_holdout: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holdout_brier: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holdout_logloss: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holdout_roc_auc: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holdout_accuracy: Option<f64>,
}

/// `CarbonBenchScenarioSummary` is a struct summarizing one scenario inside `carbon bench`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonBenchScenarioSummary {
    pub scenario_id: String,
    pub description: String,
    pub passed: bool,
    pub execution_mode: String,
    pub wall_ms: f64,
    pub prep_ms: f64,
    pub upload_ms: f64,
    pub search_ms: f64,
    pub predict_ms: f64,
    pub candidates_evaluated: u64,
    pub candidates_per_sec: f64,
    pub eval_shards: u32,
    #[serde(default)]
    pub eval_invocations: u64,
    #[serde(default)]
    pub estimated_cost_usd: f64,
    pub best_cv_score: Option<f64>,
    pub outer_holdout_score: Option<f64>,
    #[serde(default)]
    pub holdout_r2: Option<f64>,
    #[serde(default)]
    pub mean_baseline_rmse: Option<f64>,
    #[serde(default)]
    pub linear_baseline_rmse: Option<f64>,
    #[serde(default)]
    pub rmse_reduction_vs_linear_pct: Option<f64>,
    pub notes: String,
}

/// `CarbonBenchmarkSuiteReport` is a struct capturing the complete output of `carbon bench`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonBenchmarkSuiteReport {
    pub timestamp_epoch_ms: u64,
    pub target_endpoint: String,
    pub probed_region: String,
    pub enterprise_self_hosted: bool,
    pub zero_knowledge_kek_enabled: bool,
    pub cold_ping_ms: f64,
    pub warm_ping_p50_ms: f64,
    pub warm_ping_min_ms: f64,
    pub all_passed: bool,
    #[serde(default)]
    pub total_suite_estimated_usd: f64,
    pub scenarios: Vec<CarbonBenchScenarioSummary>,
}

/// Generate a deterministic nonlinear regression matrix (`rows x (features + 1)`) where `y` is final.
pub fn generate_nonlinear_benchmark_rows(
    rows: usize,
    features: usize,
    seed: u64,
) -> (Vec<String>, Vec<Vec<f64>>, String) {
    let n_feat = features.max(2);
    let mut headers = Vec::with_capacity(n_feat + 1);
    for j in 0..n_feat {
        headers.push(format!("x{j}"));
    }
    let target_name = "y".to_owned();
    headers.push(target_name.clone());

    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next_unit = || -> f64 {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let v = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)) * 6.0 - 3.0
    };

    let mut matrix = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut row = Vec::with_capacity(n_feat + 1);
        for _ in 0..n_feat {
            row.push(next_unit());
        }
        let x0 = row[0];
        let x1 = row[1];
        let x2 = if n_feat > 2 { row[2] } else { 0.5 * x0 };
        let noise = next_unit() * 0.02;
        let y = 2.4 * x0 - 1.35 * x1 + 0.55 * x0 * x1 + 0.8 * x2.sin() + 0.75 + noise;
        row.push(y);
        matrix.push(row);
    }
    (headers, matrix, target_name)
}

/// Parse a numeric CSV file with a header row into `(headers, numeric_rows)`.
/// Delegates RFC-4180 delimiter/quote/UTF-8 BOM handling to [`load_delimited_table`].
pub fn load_numeric_csv(path: &Path) -> Result<(Vec<String>, Vec<Vec<f64>>), String> {
    let raw_table = load_delimited_table(path, Some(','), true)?;
    if raw_table.headers.len() < 2 {
        return Err(format!(
            "CSV {} must contain at least 1 feature column and 1 target column",
            path.display()
        ));
    }
    let mut rows = Vec::with_capacity(raw_table.rows.len());
    for (idx, row) in raw_table.rows.iter().enumerate() {
        if row.len() != raw_table.headers.len() {
            return Err(format!(
                "CSV {} row {} has {} columns, expected {}",
                path.display(),
                idx + 2,
                row.len(),
                raw_table.headers.len()
            ));
        }
        let mut parsed_row = Vec::with_capacity(raw_table.headers.len());
        for (col_idx, cell) in row.iter().enumerate() {
            let trimmed = cell.as_deref().map(str::trim).unwrap_or("");
            let val: f64 = trimmed
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(|| {
                    format!(
                        "CSV {} row {} column '{}' ('{trimmed}') is not a finite number",
                        path.display(),
                        idx + 2,
                        raw_table.headers[col_idx]
                    )
                })?;
            parsed_row.push(val);
        }
        rows.push(parsed_row);
    }
    Ok((raw_table.headers, rows))
}

fn mint_run_and_dataset_ids() -> Result<(RunId, DatasetId), String> {
    let u = ulid::Ulid::new().to_string();
    let run_str = format!("run_{u}");
    let run_id: RunId = run_str
        .parse()
        .map_err(|_| format!("construct RunId '{run_str}'"))?;
    let ds_hex = &sha256_lower_hex(u.as_bytes())[..24];
    let ds_str = format!("ds_{ds_hex}");
    let ds_id: DatasetId = ds_str
        .parse()
        .map_err(|_| format!("construct DatasetId '{ds_str}'"))?;
    Ok((run_id, ds_id))
}

#[derive(Clone, Debug)]
struct PreparedDatasetWithSplitRows {
    prepared: CarbonClientPreparedDataset,
    encoder: CarbonTabularEncoder,
    total_rows: usize,
    fit_x: Vec<Vec<f64>>,
    fit_y: Vec<f64>,
    holdout_x: Vec<Vec<f64>>,
    holdout_y: Vec<f64>,
    holdout_aux_cols: BTreeMap<String, Vec<f64>>,
    predict_batch_x: Vec<Vec<f64>>,
    predict_batch_y: Vec<f64>,
    eval_holdout_count: usize,
    sample_count: usize,
}

#[derive(Clone, Copy, Debug)]
struct HoldoutBaselineStats {
    sse_mean: f64,
    mean_baseline_rmse: f64,
    linear_baseline_rmse: Option<f64>,
    linear_baseline_r2: Option<f64>,
}

fn flatten_split_feature_matrices(
    fit_x: &[Vec<f64>],
    holdout_x: &[Vec<f64>],
) -> Option<(Vec<f64>, Vec<f64>, usize)> {
    let p = fit_x.first().map(Vec::len).unwrap_or(0);
    if p == 0 {
        return None;
    }
    let mut fit_flat = Vec::with_capacity(fit_x.len() * p);
    for r in fit_x {
        if r.len() != p {
            return None;
        }
        fit_flat.extend_from_slice(r);
    }
    let mut ho_flat = Vec::with_capacity(holdout_x.len() * p);
    for r in holdout_x {
        if r.len() != p {
            return None;
        }
        ho_flat.extend_from_slice(r);
    }
    Some((fit_flat, ho_flat, p))
}

fn compute_holdout_baselines(
    fit_x: &[Vec<f64>],
    fit_y: &[f64],
    holdout_x: &[Vec<f64>],
    holdout_y: &[f64],
    precomputed_ridge: Option<(f64, Option<f64>)>,
) -> Option<HoldoutBaselineStats> {
    if fit_y.is_empty() || holdout_y.is_empty() {
        return None;
    }
    let n_fit = fit_y.len();
    let n_hold = holdout_y.len();
    let fit_mean = fit_y.iter().copied().sum::<f64>() / (n_fit as f64);
    let sse_mean: f64 = holdout_y
        .iter()
        .map(|&y| {
            let d = y - fit_mean;
            d * d
        })
        .sum();
    let mean_baseline_rmse = (sse_mean / (n_hold as f64)).sqrt();

    let p = fit_x.first().map(Vec::len).unwrap_or(0);
    let (linear_baseline_rmse, linear_baseline_r2) = if let Some((r_rmse, r_r2)) = precomputed_ridge
    {
        (Some(r_rmse), r_r2)
    } else if p > 0 && p <= 256 {
        if let Some((flat_fit, flat_hold, p)) = flatten_split_feature_matrices(fit_x, holdout_x) {
            match (
                FeatureMatrix::new(&flat_fit, n_fit, p),
                FeatureMatrix::new(&flat_hold, n_hold, p),
            ) {
                (Ok(train_mat), Ok(hold_mat)) => {
                    if let Ok(ridge) = RidgeModel::fit(
                        train_mat,
                        fit_y,
                        &RidgeParams {
                            alpha: 1e-6 * (n_fit.max(1) as f64),
                        },
                    ) {
                        if let Ok(preds) = ridge.predict(hold_mat) {
                            if let Ok(lin_rmse) = comp_rmse(holdout_y, &preds) {
                                let sse_lin = lin_rmse * lin_rmse * (n_hold as f64);
                                let lin_r2 = if sse_mean > 1e-18 {
                                    Some(1.0 - (sse_lin / sse_mean))
                                } else {
                                    None
                                };
                                (Some(lin_rmse), lin_r2)
                            } else {
                                (None, None)
                            }
                        } else {
                            (None, None)
                        }
                    } else {
                        (None, None)
                    }
                }
                _ => (None, None),
            }
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    Some(HoldoutBaselineStats {
        sse_mean,
        mean_baseline_rmse,
        linear_baseline_rmse,
        linear_baseline_r2,
    })
}

fn compute_holdout_feature_drivers(
    feature_names: &[String],
    fit_x: &[Vec<f64>],
    fit_y: &[f64],
    holdout_x: &[Vec<f64>],
    holdout_y: &[f64],
    seed: u64,
) -> Vec<CarbonFeatureImportanceEntry> {
    let p = feature_names.len();
    if p == 0
        || p > 48
        || fit_x.len() < 12
        || fit_x.len() > 2_500
        || holdout_x.len() < 8
        || fit_x.len() != fit_y.len()
        || holdout_x.len() != holdout_y.len()
    {
        return Vec::new();
    }
    let Some((fit_flat, ho_flat, p_chk)) = flatten_split_feature_matrices(fit_x, holdout_x) else {
        return Vec::new();
    };
    if p_chk != p {
        return Vec::new();
    }
    let Ok(x_tr) = jope_comparators::FeatureMatrix::new(&fit_flat, fit_x.len(), p) else {
        return Vec::new();
    };
    let Ok(x_ho) = jope_comparators::FeatureMatrix::new(&ho_flat, holdout_x.len(), p) else {
        return Vec::new();
    };
    let deadline = jope_comparators::Deadline::after(Duration::from_millis(250));
    let Ok(rf) = jope_comparators::fit_comparator(
        jope_comparators::ComparatorId::RandomForest,
        jope_comparators::ComparatorTier::Medium,
        x_tr,
        fit_y,
        seed ^ 0xA5A5_5A5A_1234_5678,
        &deadline,
    ) else {
        return Vec::new();
    };
    let Ok(base_preds) = rf.predict(x_ho) else {
        return Vec::new();
    };
    let Ok(base_rmse) = jope_comparators::rmse(holdout_y, &base_preds) else {
        return Vec::new();
    };

    let n_ho = holdout_x.len();
    let mut deltas = Vec::with_capacity(p);
    for (col, feat_name) in feature_names.iter().enumerate().take(p) {
        let mut perm_flat = ho_flat.clone();
        // Deterministic Fisher-Yates column permutation across holdout rows.
        let mut st = seed.wrapping_add((col as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            ^ 0xBF58_476D_1CE4_E5B9;
        for i in (1..n_ho).rev() {
            st ^= st >> 12;
            st ^= st << 25;
            st ^= st >> 27;
            let j = (st.wrapping_mul(0x2545_F491_4F6C_DD1D) as usize) % (i + 1);
            perm_flat.swap(i * p + col, j * p + col);
        }
        if let Ok(x_perm) = jope_comparators::FeatureMatrix::new(&perm_flat, n_ho, p) {
            if let Ok(p_preds) = rf.predict(x_perm) {
                if let Ok(p_rmse) = jope_comparators::rmse(holdout_y, &p_preds) {
                    deltas.push((feat_name.clone(), (p_rmse - base_rmse).max(0.0)));
                }
            }
        }
    }
    let total_delta: f64 = deltas.iter().map(|(_, d)| *d).sum();
    let mut entries: Vec<CarbonFeatureImportanceEntry> = deltas
        .into_iter()
        .map(|(feature, rmse_delta)| CarbonFeatureImportanceEntry {
            feature,
            rmse_delta,
            relative_pct: if total_delta > 1e-12 {
                100.0 * rmse_delta / total_delta
            } else {
                0.0
            },
        })
        .collect();
    entries.sort_by(|a, b| {
        b.rmse_delta
            .partial_cmp(&a.rmse_delta)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    entries.truncate(12);
    entries
}

#[allow(clippy::too_many_arguments)]
fn compute_objective_run_comparators(
    spec: &ObjectiveSpec,
    is_plain_rmse: bool,
    fit_x: &[Vec<f64>],
    fit_y: &[f64],
    holdout_x: &[Vec<f64>],
    holdout_y: &[f64],
    holdout_aux: &BTreeMap<String, Vec<f64>>,
    seed: u64,
) -> Vec<CarbonComparatorScore> {
    if is_plain_rmse {
        return compute_quick_run_comparators(fit_x, fit_y, holdout_x, holdout_y, seed);
    }
    let p = fit_x.first().map(Vec::len).unwrap_or(0);
    if p == 0
        || fit_x.len() < 8
        || holdout_x.is_empty()
        || fit_x.len() != fit_y.len()
        || holdout_x.len() != holdout_y.len()
    {
        return Vec::new();
    }
    let Some((fit_flat, ho_flat, p_chk)) = flatten_split_feature_matrices(fit_x, holdout_x) else {
        return Vec::new();
    };
    if p_chk != p {
        return Vec::new();
    }
    let (Ok(x_tr), Ok(x_ho)) = (
        jope_comparators::FeatureMatrix::new(&fit_flat, fit_x.len(), p),
        jope_comparators::FeatureMatrix::new(&ho_flat, holdout_x.len(), p),
    ) else {
        return Vec::new();
    };
    let fit_mean = fit_y.iter().copied().sum::<f64>() / (fit_y.len() as f64);
    let sst: f64 = holdout_y
        .iter()
        .map(|&y| {
            let d = y - fit_mean;
            d * d
        })
        .sum();
    let mut out = Vec::new();
    let t_ridge = Instant::now();
    if let Ok(ridge) = RidgeModel::fit(x_tr, fit_y, &RidgeParams { alpha: 1.0 }) {
        if let Ok(preds) = ridge.predict(x_ho) {
            if let Ok(obj_score) = spec.evaluate(holdout_y, &preds, holdout_aux) {
                let r2 = comp_rmse(holdout_y, &preds).ok().and_then(|rm| {
                    if sst > 1e-18 {
                        Some(1.0 - (rm * rm * (holdout_y.len() as f64) / sst))
                    } else {
                        None
                    }
                });
                out.push(CarbonComparatorScore {
                    key: "ridge_l2".to_owned(),
                    label: "Standardized Ridge (alpha=1.0, Rust)".to_owned(),
                    reference: "Closed-form Cholesky Ridge".to_owned(),
                    known_gaps: "fixed alpha=1.0".to_owned(),
                    rmse: obj_score.value,
                    mae: 0.0,
                    r2,
                    fit_ms: t_ridge.elapsed().as_secs_f64() * 1_000.0,
                });
            }
        }
    }
    for cid in [
        jope_comparators::ComparatorId::RandomForest,
        jope_comparators::ComparatorId::LightgbmStyle,
        jope_comparators::ComparatorId::AutogluonStyle,
    ] {
        let t0 = Instant::now();
        let desc = cid.descriptor();
        let deadline = jope_comparators::Deadline::after(Duration::from_millis(300));
        if let Ok(model) = jope_comparators::fit_comparator(
            cid,
            jope_comparators::ComparatorTier::Medium,
            x_tr,
            fit_y,
            seed,
            &deadline,
        ) {
            if let Ok(preds) = model.predict(x_ho) {
                if let Ok(obj_score) = spec.evaluate(holdout_y, &preds, holdout_aux) {
                    let r2 = comp_rmse(holdout_y, &preds).ok().and_then(|rm| {
                        if sst > 1e-18 {
                            Some(1.0 - (rm * rm * (holdout_y.len() as f64) / sst))
                        } else {
                            None
                        }
                    });
                    out.push(CarbonComparatorScore {
                        key: desc.key.to_owned(),
                        label: desc.label.to_owned(),
                        reference: desc.reference.to_owned(),
                        known_gaps: desc.known_gaps.to_owned(),
                        rmse: obj_score.value,
                        mae: 0.0,
                        r2,
                        fit_ms: t0.elapsed().as_secs_f64() * 1_000.0,
                    });
                }
            }
        }
    }
    out
}

fn validate_target_domain_for_objective(
    spec: &ObjectiveSpec,
    target_col: &str,
    fit_y: &[f64],
    holdout_y: &[f64],
) -> Result<(), String> {
    let name_lc = spec.name().to_ascii_lowercase();
    let base_metric = name_lc.split(':').next().unwrap_or(name_lc.as_str());
    match base_metric {
        "rmsle" | "msle" | "poisson" | "tweedie" => {
            if let Some(bad_y) = fit_y
                .iter()
                .chain(holdout_y.iter())
                .copied()
                .find(|&y| y < 0.0)
            {
                return Err(format!(
                    "target column '{target_col}' contains negative value {bad_y}, which is outside the valid domain (y >= 0) for objective '{}'",
                    spec.name()
                ));
            }
        }
        "gamma" => {
            if let Some(bad_y) = fit_y
                .iter()
                .chain(holdout_y.iter())
                .copied()
                .find(|&y| y <= 0.0)
            {
                return Err(format!(
                    "target column '{target_col}' contains non-positive value {bad_y}, which is outside the valid domain (y > 0) for objective 'gamma'"
                ));
            }
        }
        "mape" => {
            let all_zero = fit_y
                .iter()
                .chain(holdout_y.iter())
                .all(|&y| y.abs() <= 1e-15);
            if all_zero {
                return Err(format!(
                    "target column '{target_col}' is identically zero on all rows, making objective 'mape' undefined"
                ));
            }
        }
        _ => {}
    }
    if spec.direction() == jope_prep::ObjectiveDirection::Maximize {
        for (split_lbl, ys) in [("FIT", fit_y), ("holdout", holdout_y)] {
            if ys.len() >= 2 {
                let first = ys[0];
                if ys.iter().all(|&v| (v - first).abs() <= 1e-15) {
                    return Err(format!(
                        "target column '{target_col}' is constant ({first}) on {split_lbl} split, making maximization objective '{}' undefined",
                        spec.name()
                    ));
                }
            }
        }
    }
    Ok(())
}

fn prepare_from_options(
    opts: &CarbonCliRunOptions,
    ux: &CarbonCliRunUxConfig,
    objective_spec: &ObjectiveSpec,
) -> Result<PreparedDatasetWithSplitRows, String> {
    if let Some(csv_path) = &opts.csv_path {
        let raw_table = load_delimited_table_with_options(
            csv_path,
            ux.delimiter_override,
            !ux.no_header,
            &ux.extra_na_values,
            ux.inf_as_nan,
        )?;
        let requested_target = if ux.target_was_explicit || ux.no_header {
            Some(opts.target_column.as_str())
        } else {
            None
        };
        let max_encoded_features = match opts.tier {
            CarbonTierWire::Free => 64,
            CarbonTierWire::Pro => 512,
            CarbonTierWire::Enterprise => 2_048,
        };
        let encoded = prepare_encoded_training_set_with_options(
            &raw_table,
            &CarbonTrainingPrepOptions {
                requested_target,
                seed: opts.seed,
                holdout_fraction: opts.holdout_fraction,
                max_encoded_features,
                objective_spec: Some(objective_spec),
                split_mode: ux.split_mode.as_deref(),
                calibration_mode: ux.calibration_mode.as_deref(),
                reject_constant_target: true,
            },
        )?;
        for warn_msg in &encoded.hygiene_warnings {
            log_warn(warn_msg);
        }
        if ux.show_stderr_progress {
            if !ux.target_was_explicit && encoded.encoder.target_column != opts.target_column {
                eprintln!(
                    "[carbon] Auto-selected target column '{}' (--target was not specified)",
                    encoded.encoder.target_column
                );
            }
            if let Some(bm) = &encoded.encoder.target_binary_map {
                eprintln!(
                    "[carbon] Conditioned binary target '{}' ({:?} -> 0, {:?} -> 1)",
                    encoded.encoder.target_column, bm.negative_label, bm.positive_label
                );
            }
            if !encoded.encoder.dropped_columns.is_empty() {
                let summary: Vec<String> = encoded
                    .encoder
                    .dropped_columns
                    .iter()
                    .map(|(c, reason)| format!("{c} ({reason})"))
                    .collect();
                eprintln!(
                    "[carbon] Auto-dropped non-predictive column(s): {}",
                    summary.join(", ")
                );
            }
            if encoded.dropped_target_rows > 0 {
                eprintln!(
                    "[carbon] Skipped {} row(s) with missing/non-numeric target '{}'",
                    encoded.dropped_target_rows, encoded.encoder.target_column
                );
            }
        }
        validate_target_domain_for_objective(
            objective_spec,
            &encoded.encoder.target_column,
            &encoded.fit_y,
            &encoded.holdout_y,
        )?;
        let total_rows = encoded.total_valid_rows;
        let requested_samples = opts.predict_rows.max(1).min(total_rows);
        let eval_holdout_count = encoded.holdout_x.len().min(256);
        let mut predict_batch_x: Vec<Vec<f64>> = encoded
            .holdout_x
            .iter()
            .take(eval_holdout_count)
            .cloned()
            .collect();
        let mut predict_batch_y: Vec<f64> = encoded
            .holdout_y
            .iter()
            .take(eval_holdout_count)
            .copied()
            .collect();
        if predict_batch_x.len() < requested_samples {
            for (fx, &fy) in encoded.fit_x.iter().zip(encoded.fit_y.iter()) {
                if predict_batch_x.len() >= requested_samples {
                    break;
                }
                predict_batch_x.push(fx.clone());
                predict_batch_y.push(fy);
            }
        }
        return Ok(PreparedDatasetWithSplitRows {
            prepared: encoded.prepared,
            encoder: encoded.encoder,
            total_rows,
            fit_x: encoded.fit_x,
            fit_y: encoded.fit_y,
            holdout_x: encoded.holdout_x,
            holdout_y: encoded.holdout_y,
            holdout_aux_cols: encoded.holdout_aux_cols,
            predict_batch_x,
            predict_batch_y,
            eval_holdout_count,
            sample_count: requested_samples,
        });
    }

    let (headers, rows, target_col) =
        generate_nonlinear_benchmark_rows(opts.synthetic_rows, opts.synthetic_features, opts.seed);
    let target_idx = headers
        .iter()
        .position(|h| h == &target_col)
        .ok_or_else(|| format!("target column '{target_col}' not found in synthetic headers"))?;
    let total_rows = rows.len();
    let extract_features = |r: &[f64]| -> Vec<f64> {
        r.iter()
            .enumerate()
            .filter_map(|(i, &v)| if i == target_idx { None } else { Some(v) })
            .collect()
    };
    let raw_feature_columns: Vec<String> = headers
        .iter()
        .enumerate()
        .filter_map(|(i, h)| {
            if i == target_idx {
                None
            } else {
                Some(h.clone())
            }
        })
        .collect();
    let transforms: Vec<CarbonFeatureTransform> = raw_feature_columns
        .iter()
        .map(|col| CarbonFeatureTransform::Numeric {
            source_column: col.clone(),
            median: 0.0,
            emit_missing_indicator: false,
        })
        .collect();
    let encoder = CarbonTabularEncoder {
        raw_feature_columns: raw_feature_columns.clone(),
        dropped_columns: Vec::new(),
        transforms,
        encoded_feature_names: raw_feature_columns,
        target_column: target_col.clone(),
        delimiter: ',',
        had_header: true,
        extra_na_tokens: Vec::new(),
        inf_as_nan: false,
        target_binary_map: None,
    };
    let prepared = CarbonThinClient::prepare_numeric_dataset_with_options(
        headers,
        &rows,
        &target_col,
        &[],
        opts.seed,
        opts.holdout_fraction,
        ux.split_mode.as_deref(),
        ux.calibration_mode.as_deref(),
    )?;

    let mut fit_x = Vec::with_capacity(prepared.split_plan.fit_rows().len());
    let mut fit_y = Vec::with_capacity(prepared.split_plan.fit_rows().len());
    for &idx in prepared.split_plan.fit_rows() {
        fit_x.push(extract_features(&rows[idx]));
        fit_y.push(rows[idx][target_idx]);
    }

    let mut holdout_x = Vec::with_capacity(prepared.split_plan.holdout_rows().len());
    let mut holdout_y = Vec::with_capacity(prepared.split_plan.holdout_rows().len());
    for &idx in prepared.split_plan.holdout_rows() {
        holdout_x.push(extract_features(&rows[idx]));
        holdout_y.push(rows[idx][target_idx]);
    }
    validate_target_domain_for_objective(objective_spec, &target_col, &fit_y, &holdout_y)?;

    let requested_samples = opts.predict_rows.max(1).min(total_rows);
    let eval_holdout_count = holdout_x.len().min(256);
    let mut predict_batch_x: Vec<Vec<f64>> =
        holdout_x.iter().take(eval_holdout_count).cloned().collect();
    let mut predict_batch_y: Vec<f64> =
        holdout_y.iter().take(eval_holdout_count).copied().collect();
    if predict_batch_x.len() < requested_samples {
        for r in &rows {
            if predict_batch_x.len() >= requested_samples {
                break;
            }
            predict_batch_x.push(extract_features(r));
            predict_batch_y.push(r[target_idx]);
        }
    }

    Ok(PreparedDatasetWithSplitRows {
        prepared,
        encoder,
        total_rows,
        fit_x,
        fit_y,
        holdout_x,
        holdout_y,
        holdout_aux_cols: BTreeMap::new(),
        predict_batch_x,
        predict_batch_y,
        eval_holdout_count,
        sample_count: requested_samples,
    })
}

/// Format a scannable human terminal scorecard for `carbon run` / `carbon fit` (`--format pretty` and TTY `stderr`).
pub fn render_pretty_run_scorecard(receipt: &CarbonCliRunReceipt) -> String {
    let mut out = String::with_capacity(1_024);
    let mode_desc = match receipt.execution_mode {
        CarbonExecutionModeWire::HybridFanOut => format!(
            "HybridFanOut ({} shards x {} MB)",
            receipt.eval_shards, receipt.eval_memory_mb
        ),
        CarbonExecutionModeWire::SoloSingleLambda => {
            format!("SoloSingleLambda ({} MB)", receipt.eval_memory_mb)
        }
    };
    let r2_str = receipt
        .holdout_r2
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "n/a".to_owned());
    let rmse_str = receipt
        .outer_holdout_score
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "n/a".to_owned());
    let cost_str = receipt
        .cost_estimate
        .as_ref()
        .map(|c| {
            let runs_per_dollar = if c.total_estimated_usd > 1e-9 {
                format!(" (~{:.0} runs/$1)", 1.0 / c.total_estimated_usd)
            } else {
                String::new()
            };
            format!("${:.6} USD{}", c.total_estimated_usd, runs_per_dollar)
        })
        .unwrap_or_else(|| "n/a".to_owned());

    out.push_str("┌─ Carbon Zero-Knowledge Run Summary & Scorecard ───────────────────────────┐\n");
    out.push_str(&format!(
        "  Run & Region  : {} | {} ({:.1} ms RTT)\n",
        receipt.run_id, receipt.selected_region, receipt.ping_rtt_ms
    ));
    out.push_str(&format!(
        "  Topology      : {} | {}/{} waves | {} candidates in {:.2}s ({:.2}s total)\n",
        mode_desc,
        receipt.waves_completed,
        receipt.waves_target,
        receipt.candidates_evaluated,
        receipt.search_ms / 1_000.0,
        receipt.total_wall_ms / 1_000.0
    ));
    out.push_str(&format!(
        "  Dataset       : {} rows ({} fit / {} holdout) x {} features\n",
        receipt.total_rows, receipt.fit_rows, receipt.holdout_rows, receipt.feature_count
    ));
    let obj_lbl = receipt
        .objective_name
        .as_deref()
        .filter(|s| !s.eq_ignore_ascii_case("rmse"))
        .unwrap_or("RMSE");
    out.push_str(&format!(
        "  Holdout Score : R^2 = {r2_str} | {obj_lbl} = {rmse_str}"
    ));
    if let Some(q) = &receipt.residual_quantiles {
        out.push_str(&format!(
            " | MAE = {:.4} (p50={:.4}, p90={:.4}, p99={:.4})",
            q.mae, q.abs_err_p50, q.abs_err_p90, q.abs_err_p99
        ));
    }
    out.push('\n');

    if !receipt.comparator_baselines.is_empty() {
        let higher_is_better = receipt.objective_direction.as_deref() == Some("maximize");
        let dir_phrase = if higher_is_better {
            "higher is better"
        } else {
            "lower is better"
        };
        out.push_str(&format!(
            "  Same-Split Leaderboard (Holdout {obj_lbl} — {dir_phrase}):\n"
        ));
        if let Some(c_rmse) = receipt.outer_holdout_score {
            out.push_str(&format!(
                "    * {:<22} {:<4} = {:<10.4} R^2 = {}\n",
                "Carbon (r882 GP)", obj_lbl, c_rmse, r2_str
            ));
            for comp in &receipt.comparator_baselines {
                let lift = if comp.rmse.abs() > 1e-12 {
                    let delta = if higher_is_better {
                        c_rmse - comp.rmse
                    } else {
                        comp.rmse - c_rmse
                    };
                    format!("{:+.1}% vs Carbon", 100.0 * delta / comp.rmse.abs())
                } else {
                    String::new()
                };
                let cr2 = comp
                    .r2
                    .map(|v| format!("{v:.6}"))
                    .unwrap_or_else(|| "n/a".to_owned());
                out.push_str(&format!(
                    "      {:<22} {:<4} = {:<10.4} R^2 = {:<9} ({lift})\n",
                    comp.label, obj_lbl, comp.rmse, cr2
                ));
            }
        }
    }

    if !receipt.feature_importances.is_empty() {
        let top: Vec<String> = receipt
            .feature_importances
            .iter()
            .take(4)
            .filter(|f| f.relative_pct >= 1.0)
            .map(|f| format!("{} ({:.0}%)", f.feature, f.relative_pct))
            .collect();
        if !top.is_empty() {
            out.push_str(&format!("  Top Drivers   : {}\n", top.join(", ")));
        }
    }

    out.push_str(&format!(
        "  Zero-Knowledge: CUEN\\x01 = {} | Shredded = {} | Est. Cost = {cost_str}\n",
        receipt.encrypted_cuen, receipt.shredded_after_run
    ));
    if let Some(saved) = &receipt.saved_model_path {
        out.push_str(&format!("  Saved Handle  : {saved} (0600)\n"));
        out.push_str(&format!(
            "  Next (Predict): carbon predict <NEW_DATA.csv> --model \"{saved}\" --out predictions.csv\n"
        ));
    }
    out.push_str("└───────────────────────────────────────────────────────────────────────────┘");
    out
}

fn emit_stderr_run_summary(receipt: &CarbonCliRunReceipt) {
    for line in render_pretty_run_scorecard(receipt).lines() {
        eprintln!("[carbon] {line}");
    }
}

/// Execute a complete `carbon run` lifecycle against the fastest reachable endpoint.
pub fn execute_carbon_cli_run(opts: &CarbonCliRunOptions) -> Result<CarbonCliRunReceipt, String> {
    execute_carbon_cli_run_with_ux(opts, &CarbonCliRunUxConfig::default())
}

/// Execute a complete `carbon run` lifecycle with optional pre-probed KEK-pinned client reuse,
/// `--target` ergonomics, and interactive `stderr` wave/summary telemetry.
pub fn execute_carbon_cli_run_with_ux(
    opts: &CarbonCliRunOptions,
    ux: &CarbonCliRunUxConfig,
) -> Result<CarbonCliRunReceipt, String> {
    let t_total = Instant::now();
    let (objective_wire, objective_spec) = resolve_objective_argument(&opts.objective)?;
    let is_plain_rmse = objective_wire == "rmse";
    let is_maximize = objective_spec.direction() == jope_prep::ObjectiveDirection::Maximize;
    let objective_direction_label = if is_plain_rmse {
        None
    } else {
        Some(objective_spec.direction().as_str().to_owned())
    };
    let resolve_natural_metric = |metric: Option<f64>, loss: Option<f64>| -> Option<f64> {
        metric.or_else(|| loss.map(|l| if is_maximize { -l } else { l }))
    };
    let objective_display_name = if is_plain_rmse {
        None
    } else {
        let spec_name = objective_spec.name();
        if !spec_name.is_empty() {
            Some(spec_name)
        } else if !objective_wire.starts_with('{') {
            Some(objective_wire.clone())
        } else {
            Some("custom_loss".to_owned())
        }
    };

    let t_prep = Instant::now();
    let PreparedDatasetWithSplitRows {
        prepared,
        encoder,
        total_rows,
        fit_x,
        fit_y,
        holdout_x,
        holdout_y,
        holdout_aux_cols,
        predict_batch_x,
        predict_batch_y,
        eval_holdout_count,
        sample_count,
    } = prepare_from_options(opts, ux, &objective_spec)?;
    let comparator_baselines = if fit_x.len() <= 2_500 {
        compute_objective_run_comparators(
            &objective_spec,
            is_plain_rmse,
            &fit_x,
            &fit_y,
            &holdout_x,
            &holdout_y,
            &holdout_aux_cols,
            opts.seed,
        )
    } else {
        Vec::new()
    };
    let precomputed_ridge = if is_plain_rmse {
        comparator_baselines
            .iter()
            .find(|c| c.key == "ridge_l2")
            .map(|c| (c.rmse, c.r2))
    } else {
        None
    };
    let baseline_stats =
        compute_holdout_baselines(&fit_x, &fit_y, &holdout_x, &holdout_y, precomputed_ridge);
    let feature_importances = compute_holdout_feature_drivers(
        &prepared.feature_names,
        &fit_x,
        &fit_y,
        &holdout_x,
        &holdout_y,
        opts.seed,
    );
    let prep_ms = t_prep.elapsed().as_secs_f64() * 1_000.0;
    let jlmb_bytes = prepared.landmark_bundle.to_wire_bytes().len();
    let split_mode_applied = prepared
        .split_mode
        .clone()
        .unwrap_or_else(|| "random".to_owned());
    let holdout_rows_sha256 = compute_canonical_holdout_rows_sha256(
        prepared.split_plan.holdout_rows(),
        &holdout_x,
        &holdout_y,
    );
    let holdout_jmat_sha256 = hex::encode(prepared.artifacts.holdout_sha256);

    if ux.dry_run {
        let ptier = match opts.tier {
            CarbonTierWire::Free => CarbonProductTier::Free,
            CarbonTierWire::Pro => CarbonProductTier::Pro,
            CarbonTierWire::Enterprise => CarbonProductTier::Enterprise,
        };
        let eff = match opts.effort {
            Effort::Fast => CarbonEffortTier::Fast,
            Effort::Medium => CarbonEffortTier::Medium,
            Effort::High => CarbonEffortTier::High,
            Effort::Ultra => CarbonEffortTier::Ultra,
        };
        ptier
            .validate_admission(total_rows, prepared.feature_names.len(), eff)
            .map_err(crate::error::err_to_string)?;
        let est_ms = estimate_ms_per_candidate_1vcpu(
            prepared.fit_rows as usize,
            prepared.feature_names.len(),
        );
        let plan = plan_carbon_execution(&CarbonPlanInput {
            product_tier: ptier,
            effort: eff,
            total_rows,
            fit_rows: prepared.fit_rows as usize,
            features: prepared.feature_names.len(),
            est_ms_per_candidate_1vcpu: est_ms,
        })
        .map_err(crate::error::err_to_string)?;
        let mode_wire = match plan.mode {
            CarbonExecutionMode::SoloSingleLambda => CarbonExecutionModeWire::SoloSingleLambda,
            CarbonExecutionMode::HybridFanOut => CarbonExecutionModeWire::HybridFanOut,
        };
        let dry_binary_target = encoder.target_binary_map.as_ref().map(|bm| {
            let pr_fit = if fit_y.is_empty() {
                0.0
            } else {
                fit_y.iter().copied().sum::<f64>() / (fit_y.len() as f64)
            };
            let pr_hold = if holdout_y.is_empty() {
                0.0
            } else {
                holdout_y.iter().copied().sum::<f64>() / (holdout_y.len() as f64)
            };
            CarbonBinaryHoldoutSummary {
                negative_label: bm.negative_label.clone(),
                positive_label: bm.positive_label.clone(),
                positive_rate_fit: pr_fit,
                positive_rate_holdout: pr_hold,
                holdout_brier: None,
                holdout_logloss: None,
                holdout_roc_auc: None,
                holdout_accuracy: None,
            }
        });
        return Ok(CarbonCliRunReceipt {
            schema: CARBON_FIT_RECEIPT_SCHEMA.to_owned(),
            schema_version: CARBON_FIT_RECEIPT_SCHEMA_VERSION,
            release_id: None,
            selected_endpoint: "dry-run://local-preflight".to_owned(),
            selected_region: "dry-run".to_owned(),
            ping_rtt_ms: 0.0,
            run_id: "run_dry_run_preflight".to_owned(),
            dataset_id: "ds_000000000000000000000000".to_owned(),
            tenant_id: opts.tenant_id.clone(),
            tier_applied: Some(opts.tier),
            encrypted_cuen: true,
            neural_seed_applied: false,
            neural_guidance_mode: Some("dry_run_validated".to_owned()),
            bundle_digest: None,
            shredded_after_run: opts.shred_after_run,
            total_rows,
            fit_rows: prepared.fit_rows,
            holdout_rows: prepared.holdout_rows,
            feature_count: prepared.feature_names.len(),
            split_mode: split_mode_applied,
            seed: opts.seed,
            holdout_rows_sha256,
            holdout_jmat_sha256,
            jlmb_wire_bytes: jlmb_bytes,
            fit_jmat_bytes: prepared.artifacts.fit_jmat.len(),
            holdout_jmat_bytes: prepared.artifacts.holdout_jmat.len(),
            prep_ms,
            upload_ms: 0.0,
            search_ms: 0.0,
            predict_ms: 0.0,
            total_wall_ms: t_total.elapsed().as_secs_f64() * 1_000.0,
            final_state: CarbonRunStateWire::Completed,
            paused_mid_run: false,
            execution_mode: mode_wire,
            eval_memory_mb: u32::try_from(plan.eval_class.memory_mb()).unwrap_or(1_769),
            eval_shards: plan.shards_per_wave,
            eval_invocations: 0,
            eval_total_wall_ms: 0,
            poll_requests: 0,
            waves_completed: 0,
            waves_target: plan.planned_waves,
            candidates_evaluated: 0,
            best_cv_score: None,
            outer_holdout_score: None,
            holdout_r2: None,
            mean_baseline_rmse: baseline_stats.map(|b| b.mean_baseline_rmse),
            linear_baseline_rmse: baseline_stats.and_then(|b| b.linear_baseline_rmse),
            linear_baseline_r2: baseline_stats.and_then(|b| b.linear_baseline_r2),
            rmse_reduction_vs_mean_pct: None,
            rmse_reduction_vs_linear_pct: None,
            e2e_predict_holdout_rmse: None,
            e2e_predict_holdout_r2: None,
            cost_estimate: Some(plan.full_dataset_cost_estimate),
            residual_quantiles: None,
            comparator_baselines,
            feature_importances,
            saved_model_path: None,
            objective_name: objective_display_name,
            objective_direction: objective_direction_label,
            binary_target: dry_binary_target,
            sample_targets: Vec::new(),
            sample_predictions: Vec::new(),
        });
    }

    let (mut client, best_probe) = if let Some((pre_client, pre_probe)) = &ux.pre_probed {
        pre_client.reset_telemetry_counters();
        (pre_client.clone(), pre_probe.clone())
    } else {
        let (c, probed) =
            CarbonThinClient::connect_fastest_pinned(opts.endpoints.clone(), &ux.pinned_keks)?;
        let first = probed
            .into_iter()
            .next()
            .ok_or_else(|| "no Carbon endpoints probed".to_owned())?;
        (c, first)
    };
    if ux.allow_unencrypted_enterprise {
        client = client.with_allow_unencrypted_enterprise(true);
    }
    if ux.show_stderr_progress {
        eprintln!(
            "[carbon] Connected to {} ({:.1} ms RTT, zero-knowledge KEK={})",
            best_probe.region,
            best_probe.rtt_ms,
            best_probe.engine_kek_pubkey_b64.is_some()
        );
    }

    let mut effective_tenant = opts.tenant_id.clone();
    let mut minted_ephemeral_token = false;
    if let Some(tok) = &opts.auth_token {
        client = client.with_capability_token(tok.clone());
    } else if !client.enterprise_self_hosted()
        && opts.tier == CarbonTierWire::Free
        && std::env::var("CARBON_AUTH_TOKEN").is_err()
        && std::env::var("CARBON_AUTH_SECRET_HEX").is_err()
    {
        let anon = client.authenticate_anonymous()?;
        effective_tenant = anon.tenant_id;
        minted_ephemeral_token = true;
    }

    if ux.show_stderr_progress {
        eprintln!(
            "[carbon] Prepared {} rows ({} fit / {} holdout, {} features, target='{}') in {:.1} ms",
            total_rows,
            prepared.fit_rows,
            prepared.holdout_rows,
            prepared.feature_names.len(),
            prepared.target_name,
            prep_ms
        );
    }

    let (run_id, ds_id) = mint_run_and_dataset_ids()?;
    let run_id_str = run_id.to_string();

    let t_upload = Instant::now();
    let prep_upload = client.upload_prepared(&effective_tenant, opts.tier, &prepared)?;
    let upload_ms = t_upload.elapsed().as_secs_f64() * 1_000.0;
    if ux.show_stderr_progress {
        let total_kib = ((prep_upload.fit_target.max_bytes.0
            + prep_upload.holdout_target.max_bytes.0) as f64)
            / 1024.0;
        eprintln!(
            "[carbon] Encrypted & uploaded JMAT blobs ({total_kib:.1} KiB) in {upload_ms:.1} ms"
        );
    }

    // Arm `ShredOnDropGuard` unconditionally right after upload so any failed `start_run`,
    // `resume_run`, `/predict`, or handle persistence automatically shreds the staged CAS blobs
    // and run state instead of leaving ciphertext until S3 lifecycle expiry.
    let mut shred_guard = ShredOnDropGuard::new(
        client.clone(),
        effective_tenant.clone(),
        run_id_str.clone(),
        Some(prep_upload.upload_id.clone()),
        Some(prep_upload.fit_target.sha256.clone()),
        Some(prep_upload.holdout_target.sha256.clone()),
        true,
    );

    let is_tty_progress = ux.show_stderr_progress && stderr_is_tty();
    let spinner_state = Arc::new(Mutex::new((
        None::<CarbonRunStateWire>,
        0u32,
        0u32,
        0u64,
        None::<f64>,
    )));
    if ux.show_stderr_progress {
        let seen = Arc::new(Mutex::new((None::<CarbonRunStateWire>, u32::MAX)));
        let spinner_ref = Arc::clone(&spinner_state);
        let sigint_client = client.clone();
        let sigint_tenant = effective_tenant.clone();
        let sigint_run = run_id_str.clone();
        client = client.with_progress_observer(move |p: &CarbonPublicWaveProgress| {
            if sigint_received() {
                let _ = sigint_client.signal_early_stop(
                    &sigint_tenant,
                    &sigint_run,
                    Some("cli_sigint".to_owned()),
                );
            }
            let cv_disp = p
                .best_cv_metric
                .or_else(|| p.best_cv_loss.map(|l| if is_maximize { -l } else { l }));
            if let Ok(mut sp) = spinner_ref.lock() {
                *sp = (
                    Some(p.state),
                    p.wave_index,
                    p.waves_target,
                    p.candidates_evaluated.0,
                    cv_disp,
                );
            }
            if is_tty_progress {
                return;
            }
            let Ok(mut guard) = seen.lock() else {
                return;
            };
            if guard.0 == Some(p.state) && guard.1 == p.wave_index {
                return;
            }
            *guard = (Some(p.state), p.wave_index);
            let cv_txt = cv_disp
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "-".to_owned());
            eprintln!(
                "[carbon] Wave {}/{} ({:?}) | candidates={} | best_cv={} | elapsed={:.1}s",
                p.wave_index,
                p.waves_target,
                p.state,
                p.candidates_evaluated.0,
                cv_txt,
                (p.elapsed_ms.0 as f64) / 1_000.0
            );
        });
    }

    let mut tty_spinner = TtySearchSpinner::start(is_tty_progress, spinner_state);

    let t_search = Instant::now();
    let mut paused_mid_run = false;
    let mut engine_search_invocations = 1u32;
    let mut progress = if let Some(delay_ms) = opts.stop_after_ms {
        let client_arc = Arc::new(client.clone());
        let stopper_client = Arc::clone(&client_arc);
        let stop_tenant = effective_tenant.clone();
        let stop_run = run_id_str.clone();
        let stopper = thread::spawn(move || {
            thread::sleep(Duration::from_millis(delay_ms));
            let _ = stopper_client.signal_early_stop(
                &stop_tenant,
                &stop_run,
                Some("cli_stop_after_ms".to_owned()),
            );
        });
        let p = client_arc.start_run(
            run_id.clone(),
            ds_id.clone(),
            &effective_tenant,
            opts.tier,
            opts.effort,
            &objective_wire,
            opts.seed,
            &prepared,
        )?;
        let _ = stopper.join();
        p
    } else {
        client.start_run(
            run_id.clone(),
            ds_id.clone(),
            &effective_tenant,
            opts.tier,
            opts.effort,
            &objective_wire,
            opts.seed,
            &prepared,
        )?
    };

    let persist_interrupted_handle = |prog: &CarbonPublicWaveProgress| -> Option<String> {
        if opts.shred_after_run || !(ux.persist_handle || ux.save_model_path.is_some()) {
            return None;
        }
        let sel_reg = if prog.region.is_empty() {
            best_probe.region.clone()
        } else {
            prog.region.clone()
        };
        let u_dek_b64 = client
            .export_run_udek(&effective_tenant, &run_id_str)
            .map(|dek| base64::engine::general_purpose::STANDARD.encode(dek));
        let handle = CarbonSavedModelHandle {
            schema: CARBON_MODEL_HANDLE_SCHEMA.to_owned(),
            created_epoch_ms: now_epoch_ms(),
            run_id: run_id_str.clone(),
            dataset_id: ds_id.to_string(),
            tenant_id: effective_tenant.clone(),
            tier: opts.tier,
            cloud: ux.cloud_name.clone(),
            endpoint: client.active_endpoint().to_owned(),
            region: sel_reg,
            engine_kek_pubkey_b64: client.active_kek_pubkey_b64(),
            encrypted_cuen: prog.encrypted_cuen,
            u_dek_b64,
            capability_token: client.active_capability_token(),
            encoder: encoder.clone(),
            fit_rows: prepared.fit_rows,
            holdout_rows: prepared.holdout_rows,
            holdout_rmse: resolve_natural_metric(
                prog.outer_holdout_metric,
                prog.outer_holdout_loss,
            ),
            holdout_mae: None,
            holdout_r2: None,
            objective_name: objective_display_name.clone(),
            objective_direction: objective_direction_label.clone(),
            feature_importances: feature_importances.clone(),
        };
        persist_model_handle_with_options(
            &handle,
            ux.save_model_path.as_deref(),
            ux.embed_token || minted_ephemeral_token,
        )
        .ok()
        .map(|p| p.display().to_string())
    };

    if sigint_received() {
        tty_spinner.finish();
        if opts.shred_after_run {
            let _ = client.shred_run_with_upload(
                &effective_tenant,
                &run_id_str,
                Some(&prep_upload.upload_id),
                Some(&prep_upload.fit_target.sha256),
                Some(&prep_upload.holdout_target.sha256),
            );
            shred_guard.disarm();
            return Err("interrupted by SIGINT (Ctrl+C); zero-knowledge run shredded".to_owned());
        }
        if let Some(saved_p) = persist_interrupted_handle(&progress) {
            shred_guard.disarm();
            return Err(format!(
                "interrupted by SIGINT (Ctrl+C); paused run '{}' saved to {saved_p} (resume with `carbon resume --run {run_id_str}` or shred with `carbon shred --run {run_id_str}`)",
                run_id_str
            ));
        }
        return Err(format!(
            "interrupted by SIGINT (Ctrl+C); run '{}' stopped early",
            run_id_str
        ));
    }

    if opts.stop_after_ms.is_some() {
        if progress.state == CarbonRunStateWire::Paused {
            paused_mid_run = true;
            if opts.resume_after_stop {
                engine_search_invocations += 1;
                progress = client.resume_run(
                    &effective_tenant,
                    &run_id_str,
                    Some(opts.effort),
                    Some(2),
                )?;
            }
        }
    } else {
        while progress.state == CarbonRunStateWire::Paused
            && progress.can_resume
            && progress.wave_index < progress.waves_target
            && engine_search_invocations < 4
            && !sigint_received()
        {
            paused_mid_run = true;
            engine_search_invocations += 1;
            let remaining = progress
                .waves_target
                .saturating_sub(progress.wave_index)
                .max(1);
            progress = client.resume_run(
                &effective_tenant,
                &run_id_str,
                Some(opts.effort),
                Some(remaining),
            )?;
        }
    }
    tty_spinner.finish();
    let search_ms = t_search.elapsed().as_secs_f64() * 1_000.0;

    if sigint_received() {
        if opts.shred_after_run {
            let _ = client.shred_run_with_upload(
                &effective_tenant,
                &run_id_str,
                Some(&prep_upload.upload_id),
                Some(&prep_upload.fit_target.sha256),
                Some(&prep_upload.holdout_target.sha256),
            );
            shred_guard.disarm();
            return Err("interrupted by SIGINT (Ctrl+C); zero-knowledge run shredded".to_owned());
        }
        if let Some(saved_p) = persist_interrupted_handle(&progress) {
            shred_guard.disarm();
            return Err(format!(
                "interrupted by SIGINT (Ctrl+C); paused run '{}' saved to {saved_p} (resume with `carbon resume --run {run_id_str}` or shred with `carbon shred --run {run_id_str}`)",
                run_id_str
            ));
        }
        return Err(format!(
            "interrupted by SIGINT (Ctrl+C); run '{}' stopped early",
            run_id_str
        ));
    }

    let predict_inputs: Vec<Vec<Option<f64>>> = predict_batch_x
        .into_iter()
        .map(|r| r.into_iter().map(Some).collect())
        .collect();
    let t_pred = Instant::now();
    let pred_resp = client.predict(
        &effective_tenant,
        &run_id_str,
        prepared.feature_names.clone(),
        predict_inputs,
    )?;
    let predict_ms = t_pred.elapsed().as_secs_f64() * 1_000.0;

    let mut shredded_after_run = false;
    if opts.shred_after_run {
        let shred = client.shred_run_with_upload(
            &effective_tenant,
            &run_id_str,
            Some(&prep_upload.upload_id),
            Some(&prep_upload.fit_target.sha256),
            Some(&prep_upload.holdout_target.sha256),
        )?;
        shredded_after_run = shred.shredded;
        shred_guard.disarm();
    }
    let total_wall_ms = t_total.elapsed().as_secs_f64() * 1_000.0;

    let n_holdout_f = (prepared.holdout_rows as f64).max(1.0);
    let (e2e_predict_holdout_rmse, e2e_predict_holdout_r2) =
        if eval_holdout_count > 0 && pred_resp.predictions.len() >= eval_holdout_count {
            let fit_mean = if !fit_y.is_empty() {
                fit_y.iter().copied().sum::<f64>() / (fit_y.len() as f64)
            } else {
                0.0
            };
            let mut sse_pred = 0.0;
            let mut sse_sub_mean = 0.0;
            for (&yhat, &ytrue) in pred_resp
                .predictions
                .iter()
                .take(eval_holdout_count)
                .zip(predict_batch_y.iter().take(eval_holdout_count))
            {
                let e = ytrue - yhat;
                sse_pred += e * e;
                let dm = ytrue - fit_mean;
                sse_sub_mean += dm * dm;
            }
            let rmse_p = (sse_pred / (eval_holdout_count as f64)).sqrt();
            let r2_p = if sse_sub_mean > 1e-18 {
                Some(1.0 - (sse_pred / sse_sub_mean))
            } else {
                None
            };
            (Some(rmse_p), r2_p)
        } else {
            (None, None)
        };
    let holdout_r2 = match (progress.outer_holdout_loss, baseline_stats) {
        (Some(ho_rmse), Some(b)) if b.sse_mean > 1e-18 && is_plain_rmse => {
            let sse_model = ho_rmse * ho_rmse * n_holdout_f;
            Some(1.0 - (sse_model / b.sse_mean))
        }
        _ => e2e_predict_holdout_r2,
    };
    let mean_baseline_rmse = baseline_stats.map(|b| b.mean_baseline_rmse);
    let linear_baseline_rmse = baseline_stats.and_then(|b| b.linear_baseline_rmse);
    let linear_baseline_r2 = baseline_stats.and_then(|b| b.linear_baseline_r2);
    let rmse_reduction_vs_mean_pct = match (progress.outer_holdout_loss, mean_baseline_rmse) {
        (Some(ho), Some(mb)) if mb > 1e-12 && is_plain_rmse => Some(100.0 * (1.0 - ho / mb)),
        _ => None,
    };
    let rmse_reduction_vs_linear_pct = match (progress.outer_holdout_loss, linear_baseline_rmse) {
        (Some(ho), Some(lb)) if lb > 1e-12 && is_plain_rmse => Some(100.0 * (1.0 - ho / lb)),
        _ => None,
    };
    let residual_quantiles =
        if eval_holdout_count > 0 && pred_resp.predictions.len() >= eval_holdout_count {
            let fit_mean = if !fit_y.is_empty() {
                Some(fit_y.iter().copied().sum::<f64>() / (fit_y.len() as f64))
            } else {
                None
            };
            compute_residual_quantiles(
                &predict_batch_y[..eval_holdout_count],
                &pred_resp.predictions[..eval_holdout_count],
                fit_mean,
            )
        } else {
            None
        };

    let binary_target = encoder.target_binary_map.as_ref().map(|bm| {
        let pr_fit = if fit_y.is_empty() {
            0.0
        } else {
            fit_y.iter().copied().sum::<f64>() / (fit_y.len() as f64)
        };
        let pr_hold = if holdout_y.is_empty() {
            0.0
        } else {
            holdout_y.iter().copied().sum::<f64>() / (holdout_y.len() as f64)
        };
        let bin_metrics =
            if eval_holdout_count > 0 && pred_resp.predictions.len() >= eval_holdout_count {
                compute_binary_holdout_metrics(
                    &predict_batch_y[..eval_holdout_count],
                    &pred_resp.predictions[..eval_holdout_count],
                )
            } else {
                None
            };
        CarbonBinaryHoldoutSummary {
            negative_label: bm.negative_label.clone(),
            positive_label: bm.positive_label.clone(),
            positive_rate_fit: pr_fit,
            positive_rate_holdout: pr_hold,
            holdout_brier: bin_metrics.map(|m| m.brier),
            holdout_logloss: bin_metrics.map(|m| m.logloss),
            holdout_roc_auc: bin_metrics.and_then(|m| m.roc_auc),
            holdout_accuracy: bin_metrics.map(|m| m.accuracy),
        }
    });

    let selected_region = if progress.region.is_empty() {
        best_probe.region.clone()
    } else {
        progress.region.clone()
    };

    let saved_model_path =
        if !shredded_after_run && (ux.persist_handle || ux.save_model_path.is_some()) {
            let u_dek_b64 = client
                .export_run_udek(&effective_tenant, &run_id_str)
                .map(|dek| base64::engine::general_purpose::STANDARD.encode(dek));
            let handle = CarbonSavedModelHandle {
                schema: CARBON_MODEL_HANDLE_SCHEMA.to_owned(),
                created_epoch_ms: now_epoch_ms(),
                run_id: run_id_str.clone(),
                dataset_id: ds_id.to_string(),
                tenant_id: effective_tenant.clone(),
                tier: opts.tier,
                cloud: ux.cloud_name.clone(),
                endpoint: client.active_endpoint().to_owned(),
                region: selected_region.clone(),
                engine_kek_pubkey_b64: client.active_kek_pubkey_b64(),
                encrypted_cuen: progress.encrypted_cuen,
                u_dek_b64,
                capability_token: client.active_capability_token(),
                encoder,
                fit_rows: prepared.fit_rows,
                holdout_rows: prepared.holdout_rows,
                holdout_rmse: resolve_natural_metric(
                    progress.outer_holdout_metric,
                    progress.outer_holdout_loss,
                ),
                holdout_mae: residual_quantiles.map(|q| q.mae),
                holdout_r2,
                objective_name: objective_display_name.clone(),
                objective_direction: objective_direction_label.clone(),
                feature_importances: feature_importances.clone(),
            };
            let persisted = persist_model_handle_with_options(
                &handle,
                ux.save_model_path.as_deref(),
                ux.embed_token || minted_ephemeral_token,
            )?;
            Some(persisted.display().to_string())
        } else {
            None
        };
    shred_guard.disarm();

    let sample_targets: Vec<f64> = predict_batch_y.into_iter().take(sample_count).collect();
    let sample_predictions: Vec<f64> = pred_resp
        .predictions
        .into_iter()
        .take(sample_count)
        .collect();

    let (poll_requests, http_request_bytes, http_response_bytes) = client.telemetry_counters();
    let ingress_mode = if client.enterprise_self_hosted() {
        CarbonIngressCostMode::SelfHostedVpc
    } else if client.active_endpoint().contains(".lambda-url.") {
        CarbonIngressCostMode::LambdaFunctionUrl
    } else {
        CarbonIngressCostMode::ApiGatewayHttpV2
    };
    let uploaded_cas_bytes =
        prepared.artifacts.fit_jmat.len() + prepared.artifacts.holdout_jmat.len();
    let cost_breakdown = compute_live_run_cost(&CarbonLiveCostInputs {
        ingress_mode,
        minted_ephemeral_token,
        engine_search_invocations,
        poll_requests,
        signaled_early_stop: opts.stop_after_ms.is_some(),
        shredded_after_run,
        waves_completed: progress.wave_index,
        eval_shards: progress.eval_shards,
        eval_memory_mb: progress.eval_memory_mb,
        eval_invocations: progress.eval_invocations.0,
        eval_total_wall_ms: progress.eval_total_wall_ms.0,
        search_wall_ms: search_ms,
        predict_wall_ms: predict_ms,
        uploaded_cas_bytes,
        http_request_bytes,
        http_response_bytes,
    });

    let receipt = CarbonCliRunReceipt {
        schema: CARBON_FIT_RECEIPT_SCHEMA.to_owned(),
        schema_version: CARBON_FIT_RECEIPT_SCHEMA_VERSION,
        release_id: client.active_release_id(),
        selected_endpoint: client.active_endpoint().to_owned(),
        selected_region,
        ping_rtt_ms: best_probe.rtt_ms,
        run_id: run_id_str,
        dataset_id: ds_id.to_string(),
        tenant_id: effective_tenant,
        tier_applied: progress.tier_applied,
        encrypted_cuen: progress.encrypted_cuen,
        neural_seed_applied: progress.neural_seed_applied,
        neural_guidance_mode: progress.neural_guidance_mode,
        bundle_digest: progress.bundle_digest,
        shredded_after_run,
        total_rows,
        fit_rows: prepared.fit_rows,
        holdout_rows: prepared.holdout_rows,
        feature_count: prepared.feature_names.len(),
        split_mode: split_mode_applied,
        seed: opts.seed,
        holdout_rows_sha256,
        holdout_jmat_sha256,
        jlmb_wire_bytes: jlmb_bytes,
        fit_jmat_bytes: prepared.artifacts.fit_jmat.len(),
        holdout_jmat_bytes: prepared.artifacts.holdout_jmat.len(),
        prep_ms,
        upload_ms,
        search_ms,
        predict_ms,
        total_wall_ms,
        final_state: progress.state,
        paused_mid_run,
        execution_mode: progress.execution_mode,
        eval_memory_mb: progress.eval_memory_mb,
        eval_shards: progress.eval_shards,
        eval_invocations: progress.eval_invocations.0,
        eval_total_wall_ms: progress.eval_total_wall_ms.0,
        poll_requests,
        waves_completed: progress.wave_index,
        waves_target: progress.waves_target,
        candidates_evaluated: progress.candidates_evaluated.0,
        best_cv_score: resolve_natural_metric(progress.best_cv_metric, progress.best_cv_loss),
        outer_holdout_score: resolve_natural_metric(
            progress.outer_holdout_metric,
            progress.outer_holdout_loss,
        ),
        holdout_r2,
        mean_baseline_rmse,
        linear_baseline_rmse,
        linear_baseline_r2,
        rmse_reduction_vs_mean_pct,
        rmse_reduction_vs_linear_pct,
        e2e_predict_holdout_rmse,
        e2e_predict_holdout_r2,
        cost_estimate: Some(cost_breakdown),
        residual_quantiles,
        comparator_baselines,
        feature_importances,
        saved_model_path,
        objective_name: objective_display_name,
        objective_direction: objective_direction_label,
        binary_target,
        sample_targets,
        sample_predictions,
    };
    if ux.show_stderr_progress {
        emit_stderr_run_summary(&receipt);
    }
    Ok(receipt)
}

/// Execute the 5-scenario live Carbon Free + Pro `r882` zero-knowledge benchmark suite (`carbon bench`).
pub fn execute_carbon_cli_benchmark(
    endpoints: Vec<String>,
    tier: CarbonTierWire,
    tenant_id: &str,
) -> Result<CarbonBenchmarkSuiteReport, String> {
    execute_carbon_cli_benchmark_with_token(endpoints, tier, tenant_id, None)
}

/// Execute the 5-scenario live Carbon Free + Pro `r882` zero-knowledge benchmark suite with an
/// optional in-memory `ctk1.*` capability token minted from AWS Secrets Manager.
pub fn execute_carbon_cli_benchmark_with_token(
    endpoints: Vec<String>,
    tier: CarbonTierWire,
    tenant_id: &str,
    pro_auth_token: Option<String>,
) -> Result<CarbonBenchmarkSuiteReport, String> {
    let (base_client, probed) = CarbonThinClient::connect_fastest(endpoints)?;
    let first_probe = probed
        .first()
        .ok_or_else(|| "no reachable Carbon endpoints".to_owned())?;
    let cold_ping_ms = first_probe.rtt_ms;

    let mut warm_pings = Vec::with_capacity(5);
    let mut enterprise_self_hosted = false;
    for _ in 0..5 {
        let t0 = Instant::now();
        let parsed = base_client.ping_active()?;
        let elapsed = t0.elapsed().as_secs_f64() * 1_000.0;
        enterprise_self_hosted = parsed.enterprise_self_hosted;
        warm_pings.push(elapsed);
    }
    warm_pings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let warm_ping_p50_ms = warm_pings[warm_pings.len() / 2];
    let warm_ping_min_ms = warm_pings[0];

    let client = match &pro_auth_token {
        Some(tok) => base_client.clone().with_capability_token(tok.clone()),
        None => base_client.clone(),
    };

    let mut scenarios = Vec::new();
    let active_ep = client.active_endpoint().to_owned();

    // Scenario 1: Free anonymous capability auth + CUEN\x01 + Wave-0 r882 neural seeding (or Enterprise Fast).
    if !enterprise_self_hosted {
        let mut anon_client = base_client.clone();
        let anon_auth = anon_client.authenticate_anonymous()?;
        let free_fast = execute_carbon_cli_run(&CarbonCliRunOptions {
            endpoints: vec![active_ep.clone()],
            csv_path: None,
            target_column: "y".to_owned(),
            synthetic_rows: 200,
            synthetic_features: 5,
            tenant_id: anon_auth.tenant_id,
            tier: CarbonTierWire::Free,
            effort: Effort::Fast,
            objective: "rmse".to_owned(),
            seed: 101,
            holdout_fraction: 0.20,
            stop_after_ms: None,
            resume_after_stop: false,
            predict_rows: 4,
            auth_token: Some(anon_auth.capability_token),
            shred_after_run: true,
        })?;
        scenarios.push(receipt_to_scenario(
            "free_anon_r882_wave0_cuen",
            "Free tier anonymous auth + CUEN v1 + r882 Wave-0 seed + E2E predict + shred (200x5)",
            &free_fast,
            free_fast.final_state == CarbonRunStateWire::Completed && free_fast.shredded_after_run,
        ));
    } else {
        let ent_fast = execute_carbon_cli_run(&CarbonCliRunOptions {
            endpoints: vec![active_ep.clone()],
            csv_path: None,
            target_column: "y".to_owned(),
            synthetic_rows: 200,
            synthetic_features: 5,
            tenant_id: tenant_id.to_owned(),
            tier,
            effort: Effort::Fast,
            objective: "rmse".to_owned(),
            seed: 101,
            holdout_fraction: 0.20,
            stop_after_ms: None,
            resume_after_stop: false,
            predict_rows: 4,
            auth_token: None,
            shred_after_run: false,
        })?;
        scenarios.push(receipt_to_scenario(
            "solo_fast_cold",
            "Enterprise Fast effort (SoloSingleLambda, 200x5, initial)",
            &ent_fast,
            ent_fast.final_state == CarbonRunStateWire::Completed,
        ));
    }

    // Scenario 2: Pro (or Enterprise) SoloSingleLambda Fast warm container reuse (`multi_wave_learned`).
    let pro_fast = execute_carbon_cli_run(&CarbonCliRunOptions {
        endpoints: vec![active_ep.clone()],
        csv_path: None,
        target_column: "y".to_owned(),
        synthetic_rows: 200,
        synthetic_features: 5,
        tenant_id: tenant_id.to_owned(),
        tier,
        effort: Effort::Fast,
        objective: "rmse".to_owned(),
        seed: 202,
        holdout_fraction: 0.20,
        stop_after_ms: None,
        resume_after_stop: false,
        predict_rows: 4,
        auth_token: pro_auth_token.clone(),
        shred_after_run: true,
    })?;
    scenarios.push(receipt_to_scenario(
        "pro_solo_fast_warm",
        "Pro Fast effort (SoloSingleLambda, warm container, r882 multi_wave_learned, 200x5)",
        &pro_fast,
        pro_fast.final_state == CarbonRunStateWire::Completed,
    ));

    // Scenario 3: Pro Medium async `202 Accepted` multi-wave (`4` waves, `multi_wave_learned`).
    let s3_rows = if cfg!(debug_assertions) { 64 } else { 250 };
    let s3_effort = if cfg!(debug_assertions) {
        Effort::Fast
    } else {
        Effort::Medium
    };
    let medium_run = execute_carbon_cli_run(&CarbonCliRunOptions {
        endpoints: vec![active_ep.clone()],
        csv_path: None,
        target_column: "y".to_owned(),
        synthetic_rows: s3_rows,
        synthetic_features: 6,
        tenant_id: tenant_id.to_owned(),
        tier,
        effort: s3_effort,
        objective: "rmse".to_owned(),
        seed: 303,
        holdout_fraction: 0.20,
        stop_after_ms: None,
        resume_after_stop: false,
        predict_rows: 4,
        auth_token: pro_auth_token.clone(),
        shred_after_run: true,
    })?;
    scenarios.push(receipt_to_scenario(
        "pro_medium_async_multiwave",
        "Pro Medium async 202 (4 GP waves + r882 multi_wave_learned, 250x6)",
        &medium_run,
        medium_run.final_state == CarbonRunStateWire::Completed,
    ));

    // Scenario 4: Pro High HybridFanOut (8 eval shards w/ 60s FitJmat subkey grants) + early-stop ->
    // E2E encrypted `/predict` -> cross-container `/resume` -> cryptographic `DELETE` shred!
    if tier != CarbonTierWire::Free {
        let s4_rows = if cfg!(debug_assertions) { 64 } else { 300 };
        let s4_effort = if cfg!(debug_assertions) {
            Effort::Medium
        } else {
            Effort::High
        };
        let s4_resume_waves = if cfg!(debug_assertions) { 1 } else { 2 };
        client.reset_telemetry_counters();
        let t_s4 = Instant::now();
        let PreparedDatasetWithSplitRows {
            prepared: prep_hi,
            fit_x: fit_x_hi,
            fit_y: fit_y_hi,
            holdout_x: holdout_x_hi,
            holdout_y: holdout_y_hi,
            predict_batch_x,
            sample_count: sample_n_hi,
            ..
        } = prepare_from_options(
            &CarbonCliRunOptions {
                endpoints: vec![active_ep.clone()],
                csv_path: None,
                target_column: "y".to_owned(),
                synthetic_rows: s4_rows,
                synthetic_features: 6,
                tenant_id: tenant_id.to_owned(),
                tier,
                effort: s4_effort,
                objective: "rmse".to_owned(),
                seed: 404,
                holdout_fraction: 0.20,
                stop_after_ms: None,
                resume_after_stop: false,
                predict_rows: 3,
                auth_token: None,
                shred_after_run: true,
            },
            &CarbonCliRunUxConfig {
                target_was_explicit: true,
                ..CarbonCliRunUxConfig::default()
            },
            &ObjectiveSpec::parse("rmse").map_err(crate::error::err_to_string)?,
        )?;
        let sample_hi: Vec<Vec<f64>> = predict_batch_x.into_iter().take(sample_n_hi).collect();
        let s4_baselines =
            compute_holdout_baselines(&fit_x_hi, &fit_y_hi, &holdout_x_hi, &holdout_y_hi, None);
        let t_up4 = Instant::now();
        client.upload_prepared(tenant_id, tier, &prep_hi)?;
        let up4_ms = t_up4.elapsed().as_secs_f64() * 1_000.0;
        let (run_hi, ds_hi) = mint_run_and_dataset_ids()?;
        let run_hi_str = run_hi.to_string();

        let client_arc = Arc::new(client.clone());
        let stopper_c = Arc::clone(&client_arc);
        let stop_tenant = tenant_id.to_owned();
        let stop_run_id = run_hi_str.clone();
        let stopper = thread::spawn(move || {
            for _ in 0..120 {
                thread::sleep(Duration::from_millis(200));
                if let Ok(st) = stopper_c.get_run_status(&stop_tenant, &stop_run_id) {
                    if st.state == CarbonRunStateWire::Running && st.wave_index >= 1 {
                        let _ = stopper_c.signal_early_stop(
                            &stop_tenant,
                            &stop_run_id,
                            Some("bench_see_it_now".to_owned()),
                        );
                        return true;
                    }
                    if st.state == CarbonRunStateWire::Completed {
                        return false;
                    }
                }
            }
            let _ = stopper_c.signal_early_stop(
                &stop_tenant,
                &stop_run_id,
                Some("bench_fallback_stop".to_owned()),
            );
            true
        });

        let t_search4 = Instant::now();
        let first_prog = client_arc.start_run(
            run_hi, ds_hi, tenant_id, tier, s4_effort, "rmse", 404, &prep_hi,
        )?;
        let signaled_stop_s4 = stopper.join().unwrap_or(true);

        let pred_in_hi: Vec<Vec<Option<f64>>> = sample_hi
            .into_iter()
            .map(|r| r.into_iter().map(Some).collect())
            .collect();
        let t_pred4 = Instant::now();
        let paused_pred = client.predict(
            tenant_id,
            &run_hi_str,
            prep_hi.feature_names.clone(),
            pred_in_hi,
        )?;
        let pred4_ms = t_pred4.elapsed().as_secs_f64() * 1_000.0;

        let was_paused = first_prog.state == CarbonRunStateWire::Paused;
        let final_prog = if was_paused {
            client.resume_run(
                tenant_id,
                &run_hi_str,
                Some(s4_effort),
                Some(s4_resume_waves),
            )?
        } else {
            first_prog.clone()
        };
        let search4_ms = t_search4.elapsed().as_secs_f64() * 1_000.0;
        let shred4 = client.shred_run(tenant_id, &run_hi_str)?;
        let s4_wall = t_s4.elapsed().as_secs_f64() * 1_000.0;
        let s4_cands = final_prog.candidates_evaluated.0;
        let s4_cps = if search4_ms > 0.0 {
            (s4_cands as f64) / (search4_ms / 1_000.0)
        } else {
            0.0
        };
        let (s4_polls, s4_req_bytes, s4_resp_bytes) = client.telemetry_counters();
        let s4_ingress = if enterprise_self_hosted {
            CarbonIngressCostMode::SelfHostedVpc
        } else if active_ep.contains(".lambda-url.") {
            CarbonIngressCostMode::LambdaFunctionUrl
        } else {
            CarbonIngressCostMode::ApiGatewayHttpV2
        };
        let s4_cost = compute_live_run_cost(&CarbonLiveCostInputs {
            ingress_mode: s4_ingress,
            minted_ephemeral_token: false,
            engine_search_invocations: if was_paused { 2 } else { 1 },
            poll_requests: s4_polls,
            signaled_early_stop: signaled_stop_s4,
            shredded_after_run: shred4.shredded,
            waves_completed: final_prog.wave_index,
            eval_shards: final_prog.eval_shards,
            eval_memory_mb: final_prog.eval_memory_mb,
            eval_invocations: final_prog.eval_invocations.0,
            eval_total_wall_ms: final_prog.eval_total_wall_ms.0,
            search_wall_ms: search4_ms,
            predict_wall_ms: pred4_ms,
            uploaded_cas_bytes: prep_hi.artifacts.fit_jmat.len()
                + prep_hi.artifacts.holdout_jmat.len(),
            http_request_bytes: s4_req_bytes,
            http_response_bytes: s4_resp_bytes,
        });
        let s4_r2 = match (final_prog.outer_holdout_loss, s4_baselines) {
            (Some(ho), Some(b)) if b.sse_mean > 1e-18 => {
                let sse_m = ho * ho * (prep_hi.holdout_rows as f64).max(1.0);
                Some(1.0 - (sse_m / b.sse_mean))
            }
            _ => None,
        };
        let s4_lin_rmse = s4_baselines.and_then(|b| b.linear_baseline_rmse);
        let s4_lift_lin = match (final_prog.outer_holdout_loss, s4_lin_rmse) {
            (Some(ho), Some(lb)) if lb > 1e-12 => Some(100.0 * (1.0 - ho / lb)),
            _ => None,
        };
        scenarios.push(CarbonBenchScenarioSummary {
            scenario_id: "pro_hybrid_fanout_stop_predict_resume_shred".to_owned(),
            description:
                "Pro High (HybridFanOut 8 shards, async 202 early-stop -> encrypted predict -> resume -> shred)"
                    .to_owned(),
            passed: paused_pred.row_count == 3
                && paused_pred.predictions.len() == 3
                && shred4.shredded
                && matches!(
                    final_prog.state,
                    CarbonRunStateWire::Completed | CarbonRunStateWire::Paused
                ),
            execution_mode: format!("{:?}", final_prog.execution_mode),
            wall_ms: s4_wall,
            prep_ms: 0.0,
            upload_ms: up4_ms,
            search_ms: search4_ms,
            predict_ms: pred4_ms,
            candidates_evaluated: s4_cands,
            candidates_per_sec: s4_cps,
            eval_shards: final_prog.eval_shards,
            eval_invocations: final_prog.eval_invocations.0,
            estimated_cost_usd: s4_cost.total_estimated_usd,
            best_cv_score: final_prog.best_cv_loss,
            outer_holdout_score: final_prog.outer_holdout_loss,
            holdout_r2: s4_r2,
            mean_baseline_rmse: s4_baselines.map(|b| b.mean_baseline_rmse),
            linear_baseline_rmse: s4_lin_rmse,
            rmse_reduction_vs_linear_pct: s4_lift_lin,
            notes: format!(
                "paused={was_paused}, wave_pause={}, wave_end={}, guidance={}, cuen={}, shredded_objs={}",
                first_prog.wave_index,
                final_prog.wave_index,
                final_prog.neural_guidance_mode.as_deref().unwrap_or("-"),
                final_prog.encrypted_cuen,
                shred4.objects_deleted,
            ),
        });

        // Scenario 4b: Pro Ultra deep multi-wave (`HybridFanOut`, `32` waves + `r882 multi_wave_learned`).
        let ultra_rows = if cfg!(debug_assertions) { 64 } else { 180 };
        let ultra_effort = if cfg!(debug_assertions) {
            Effort::Fast
        } else {
            Effort::Ultra
        };
        let ultra_run = execute_carbon_cli_run(&CarbonCliRunOptions {
            endpoints: vec![active_ep.clone()],
            csv_path: None,
            target_column: "y".to_owned(),
            synthetic_rows: ultra_rows,
            synthetic_features: 5,
            tenant_id: tenant_id.to_owned(),
            tier,
            effort: ultra_effort,
            objective: "rmse".to_owned(),
            seed: 454,
            holdout_fraction: 0.20,
            stop_after_ms: None,
            resume_after_stop: false,
            predict_rows: 4,
            auth_token: pro_auth_token.clone(),
            shred_after_run: true,
        })?;
        scenarios.push(receipt_to_scenario(
            "pro_ultra_hybrid_fanout",
            "Pro Ultra async 202 (32 GP waves + HybridFanOut + r882 multi_wave_learned, 180x5)",
            &ultra_run,
            ultra_run.final_state == CarbonRunStateWire::Completed,
        ));
    }

    // Scenario 5: Free-tier >5,000 row admission refusal (HTTP 403) using genuine Free token.
    if !enterprise_self_hosted {
        let t_s5 = Instant::now();
        let mut free_guard_client = base_client.clone();
        let anon5 = free_guard_client.authenticate_anonymous()?;
        let (headers5, rows5, target5) = generate_nonlinear_benchmark_rows(5_200, 4, 505);
        let prep5 =
            CarbonThinClient::prepare_numeric_dataset(headers5, &rows5, &target5, 505, 0.20)?;
        let free_res =
            free_guard_client.upload_prepared(&anon5.tenant_id, CarbonTierWire::Free, &prep5);
        let rejected_403 = match &free_res {
            Err(msg) => msg.contains("403"),
            Ok(_) => false,
        };
        let guard_edge_usd = if active_ep.contains(".lambda-url.") {
            2.0 * (0.018 * (256.0 / 1_024.0) * 0.000_013_333_4 + 0.000_000_2)
        } else {
            2.0 * (0.018 * (256.0 / 1_024.0) * 0.000_013_333_4 + 0.000_000_2 + 0.000_001)
        };
        scenarios.push(CarbonBenchScenarioSummary {
            scenario_id: "free_tier_row_cap_403".to_owned(),
            description: "Free tier >5,000 row admission refusal (HTTP 403)".to_owned(),
            passed: rejected_403,
            execution_mode: "EdgeAdmissionGuard".to_owned(),
            wall_ms: t_s5.elapsed().as_secs_f64() * 1_000.0,
            prep_ms: 0.0,
            upload_ms: 0.0,
            search_ms: 0.0,
            predict_ms: 0.0,
            candidates_evaluated: 0,
            candidates_per_sec: 0.0,
            eval_shards: 0,
            eval_invocations: 0,
            estimated_cost_usd: guard_edge_usd,
            best_cv_score: None,
            outer_holdout_score: None,
            holdout_r2: None,
            mean_baseline_rmse: None,
            linear_baseline_rmse: None,
            rmse_reduction_vs_linear_pct: None,
            notes: "5,200 rows rejected at edge before S3 presign".to_owned(),
        });
    }

    let all_passed = scenarios.iter().all(|s| s.passed);
    let total_suite_estimated_usd: f64 = scenarios.iter().map(|s| s.estimated_cost_usd).sum();
    Ok(CarbonBenchmarkSuiteReport {
        timestamp_epoch_ms: now_epoch_ms(),
        target_endpoint: client.active_endpoint().to_owned(),
        probed_region: first_probe.region.clone(),
        enterprise_self_hosted,
        zero_knowledge_kek_enabled: client.has_zero_knowledge_kek(),
        cold_ping_ms,
        warm_ping_p50_ms,
        warm_ping_min_ms,
        all_passed,
        total_suite_estimated_usd,
        scenarios,
    })
}

fn receipt_to_scenario(
    id: &str,
    description: &str,
    r: &CarbonCliRunReceipt,
    passed: bool,
) -> CarbonBenchScenarioSummary {
    let cps = if r.search_ms > 0.0 {
        (r.candidates_evaluated as f64) / (r.search_ms / 1_000.0)
    } else {
        0.0
    };
    let short_digest = r
        .bundle_digest
        .as_deref()
        .map(|d| if d.len() >= 19 { &d[..19] } else { d })
        .unwrap_or("none");
    let est_usd = r
        .cost_estimate
        .as_ref()
        .map(|c| c.total_estimated_usd)
        .unwrap_or(0.0);
    CarbonBenchScenarioSummary {
        scenario_id: id.to_owned(),
        description: description.to_owned(),
        passed,
        execution_mode: format!("{:?}", r.execution_mode),
        wall_ms: r.total_wall_ms,
        prep_ms: r.prep_ms,
        upload_ms: r.upload_ms,
        search_ms: r.search_ms,
        predict_ms: r.predict_ms,
        candidates_evaluated: r.candidates_evaluated,
        candidates_per_sec: cps,
        eval_shards: r.eval_shards,
        eval_invocations: r.eval_invocations,
        estimated_cost_usd: est_usd,
        best_cv_score: r.best_cv_score,
        outer_holdout_score: r.outer_holdout_score,
        holdout_r2: r.holdout_r2,
        mean_baseline_rmse: r.mean_baseline_rmse,
        linear_baseline_rmse: r.linear_baseline_rmse,
        rmse_reduction_vs_linear_pct: r.rmse_reduction_vs_linear_pct,
        notes: format!(
            "waves={}/{}, eval_calls={}, guidance={}, model={}, cuen={}",
            r.waves_completed,
            r.waves_target,
            r.eval_invocations,
            r.neural_guidance_mode.as_deref().unwrap_or("-"),
            short_digest,
            r.encrypted_cuen,
        ),
    }
}

/// Format a `CarbonBenchmarkSuiteReport` as a GitHub-flavored Markdown benchmark receipt.
pub fn render_benchmark_markdown(report: &CarbonBenchmarkSuiteReport) -> String {
    let mut md = String::new();
    md.push_str("# Carbon Serverless Benchmark Receipt\n\n");
    md.push_str(&format!(
        "- **Endpoint**: `{}`\n- **Region**: `{}` (`enterprise_self_hosted={}`, `zero_knowledge_kek={}`)\n- **Timestamp (epoch ms)**: `{}`\n- **Ping RTT**: initial `{:.2} ms`, warm p50 `{:.2} ms`, warm min `{:.2} ms`\n- **Total Suite Est. Cost**: `${:.6}`\n- **Suite Verdict**: **{}**\n\n",
        report.target_endpoint,
        report.probed_region,
        report.enterprise_self_hosted,
        report.zero_knowledge_kek_enabled,
        report.timestamp_epoch_ms,
        report.cold_ping_ms,
        report.warm_ping_p50_ms,
        report.warm_ping_min_ms,
        report.total_suite_estimated_usd,
        if report.all_passed { "PASS" } else { "FAIL" }
    ));
    md.push_str("| Scenario | Mode | Shards | Eval Calls | Wall (ms) | Search (ms) | Predict (ms) | Cands | CV RMSE | Holdout RMSE | Holdout R² | Linear RMSE | Lift vs OLS | Est. Cost ($) | Status | Notes |\n");
    md.push_str("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n");
    for s in &report.scenarios {
        let cv = s
            .best_cv_score
            .map(|v| format!("{v:.5}"))
            .unwrap_or_else(|| "-".to_owned());
        let ho = s
            .outer_holdout_score
            .map(|v| format!("{v:.5}"))
            .unwrap_or_else(|| "-".to_owned());
        let r2 = s
            .holdout_r2
            .map(|v| format!("{v:.5}"))
            .unwrap_or_else(|| "-".to_owned());
        let lin = s
            .linear_baseline_rmse
            .map(|v| format!("{v:.5}"))
            .unwrap_or_else(|| "-".to_owned());
        let lift = s
            .rmse_reduction_vs_linear_pct
            .map(|v| format!("{v:+.1}%"))
            .unwrap_or_else(|| "-".to_owned());
        md.push_str(&format!(
            "| `{}` | `{}` | {} | {} | {:.1} | {:.1} | {:.1} | {} | {} | {} | {} | {} | {} | ${:.6} | {} | {} |\n",
            s.scenario_id,
            s.execution_mode,
            s.eval_shards,
            s.eval_invocations,
            s.wall_ms,
            s.search_ms,
            s.predict_ms,
            s.candidates_evaluated,
            cv,
            ho,
            r2,
            lin,
            lift,
            s.estimated_cost_usd,
            if s.passed { "PASS" } else { "FAIL" },
            s.notes
        ));
    }
    md
}

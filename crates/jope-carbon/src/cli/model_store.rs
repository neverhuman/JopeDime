//! Persistent `0600` zero-knowledge model handles (`model.carbon.json` & `$CARBON_CONFIG_DIR/runs/`),
//! two-step `carbon fit` -> `carbon predict` scoring on unlabeled CSVs, `carbon runs` / `status` /
//! `explain` / `export` / `shred` lifecycle management, and ECDSA-P256 verified `carbon upgrade`.

use crate::error::CarbonErrCtx as _;
use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use base64::Engine as _;
use jope_api::carbon::{CarbonPublicWaveProgress, CarbonShredRunResponse, CarbonTierWire};
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

use crate::{
    cli::{
        client::{
            list_stored_cloud_tokens, load_stored_cloud_token, resolve_clouds_redirect_target,
            resolve_public_cloud_endpoint, validate_endpoint_transport,
            verified_absolute_jopedime_config_dir, CarbonPublicCloudsManifest, CarbonThinClient,
        },
        tabular::{
            load_delimited_table_with_options, parse_flexible_numeric_f64, CarbonTabularEncoder,
        },
    },
    observability::{telemetry_minimized, write_stdout_line},
    role::CarbonProductTier,
    security::user_envelope::{decode_p256_pubkey_b64, verify_ecdsa_p256_spki_pem_signature},
    storage::cas::sha256_lower_hex,
    CARBON_BUILD_TARGET, CARBON_PKG_VERSION, CARBON_RELEASE_TAG,
};

/// Schema identifier for persisted zero-knowledge Carbon model handles.
pub const CARBON_MODEL_HANDLE_SCHEMA: &str = "jope-carbon-model-handle-v1";

/// `CarbonFeatureImportanceEntry` is a struct recording holdout permutation importance for one feature.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CarbonFeatureImportanceEntry {
    pub feature: String,
    pub rmse_delta: f64,
    pub relative_pct: f64,
}

/// `CarbonResidualQuantiles` is a struct recording holdout error distribution metrics (`MAE`, `p50`, `p90`, `p99`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CarbonResidualQuantiles {
    pub mae: f64,
    pub rmse: f64,
    pub r2: Option<f64>,
    pub abs_err_p50: f64,
    pub abs_err_p90: f64,
    pub abs_err_p99: f64,
    pub max_abs_err: f64,
}

/// `CarbonSavedModelHandle` is a struct persisted in `0600` local storage so a user can train once
/// (`carbon fit`) and score arbitrary unlabeled CSVs (`carbon predict`) in zero-knowledge mode.
#[derive(Clone, Serialize, Deserialize)]
pub struct CarbonSavedModelHandle {
    pub schema: String,
    pub created_epoch_ms: u64,
    pub run_id: String,
    pub dataset_id: String,
    pub tenant_id: String,
    pub tier: CarbonTierWire,
    pub cloud: Option<String>,
    pub endpoint: String,
    pub region: String,
    pub engine_kek_pubkey_b64: Option<String>,
    pub encrypted_cuen: bool,
    pub u_dek_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability_token: Option<String>,
    pub encoder: CarbonTabularEncoder,
    pub fit_rows: u32,
    pub holdout_rows: u32,
    pub holdout_rmse: Option<f64>,
    pub holdout_mae: Option<f64>,
    pub holdout_r2: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective_direction: Option<String>,
    #[serde(default)]
    pub feature_importances: Vec<CarbonFeatureImportanceEntry>,
}

impl std::fmt::Debug for CarbonSavedModelHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CarbonSavedModelHandle")
            .field("run_id", &self.run_id)
            .field("tenant_id", &self.tenant_id)
            .field("region", &self.region)
            .field("u_dek_b64", &self.u_dek_b64.as_ref().map(|_| "<redacted>"))
            .field(
                "capability_token",
                &self.capability_token.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

/// `CarbonPredictCsvReceipt` is a struct summarizing a `carbon predict` batch scoring run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonPredictCsvReceipt {
    pub run_id: String,
    pub tenant_id: String,
    pub endpoint: String,
    pub region: String,
    pub encrypted_cuen: bool,
    pub input_csv: String,
    pub output_csv: Option<String>,
    pub rows_scored: usize,
    pub batches: usize,
    pub wall_ms: f64,
    pub predictions_sha256: String,
    pub evaluation_if_labeled: Option<CarbonResidualQuantiles>,
    pub sample_predictions: Vec<f64>,
}

/// `CarbonRunRegistryEntry` is a struct summarizing a locally stored unshredded zero-knowledge run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonRunRegistryEntry {
    pub run_id: String,
    pub created_epoch_ms: u64,
    pub tenant_id: String,
    pub tier: CarbonTierWire,
    pub region: String,
    pub endpoint: String,
    pub target_column: String,
    pub feature_count: usize,
    pub fit_rows: u32,
    pub holdout_rmse: Option<f64>,
    pub holdout_r2: Option<f64>,
    pub handle_path: String,
}

fn runs_registry_dir() -> Result<PathBuf, String> {
    let base = verified_absolute_jopedime_config_dir()?;
    Ok(base.join("runs"))
}

fn write_secret_json_0600(path: &Path, bytes: &[u8]) -> Result<(), String> {
    static SECRET_TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("create directory {}: {e}", parent.display()))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if parent.ends_with("jopedime") || parent.ends_with("runs") {
                    let _ = fs::set_permissions(parent, fs::Permissions::from_mode(0o700));
                }
            }
        }
    }
    let seq = SECRET_TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp_path = path.with_extension(format!("tmp.{}.{}", std::process::id(), seq));
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(&tmp_path)
            .map_err(|e| format!("open {} (0600): {e}", tmp_path.display()))?;
        if let Err(e) = f.write_all(bytes).and_then(|()| f.sync_all()) {
            let _ = fs::remove_file(&tmp_path);
            return Err(format!("write {}: {e}", tmp_path.display()));
        }
    }
    #[cfg(not(unix))]
    {
        let mut f = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp_path)
            .map_err(|e| format!("open {}: {e}", tmp_path.display()))?;
        if let Err(e) = f.write_all(bytes).and_then(|()| f.sync_all()) {
            let _ = fs::remove_file(&tmp_path);
            return Err(format!("write {}: {e}", tmp_path.display()));
        }
        #[cfg(windows)]
        {
            let _ = std::process::Command::new("icacls")
                .arg(&tmp_path)
                .args(["/inheritance:r", "/grant:r", "*S-1-3-4:(F)"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
        }
    }
    if let Err(e) = fs::rename(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(format!("atomic rename -> {}: {e}", path.display()));
    }
    Ok(())
}

pub(crate) fn shred_local_file(path: &Path) -> Result<bool, String> {
    if !path.is_file() {
        return Ok(false);
    }
    if let Ok(meta) = fs::metadata(path) {
        let len = (meta.len() as usize).clamp(64, 16 * 1_048_576);
        if let Ok(mut f) = fs::OpenOptions::new().write(true).open(path) {
            let zeros = vec![0u8; len];
            let _ = f.write_all(&zeros);
            let _ = f.sync_all();
        }
    }
    fs::remove_file(path).map_err(|e| format!("remove {}: {e}", path.display()))?;
    Ok(true)
}

/// Save a `CarbonSavedModelHandle` into `$CARBON_CONFIG_DIR/runs/<run_id>.json` (`0600`) and
/// optionally also to an explicit user `--save <PATH>` (`0600`, stripping `capability_token` by default).
pub fn persist_model_handle(
    handle: &CarbonSavedModelHandle,
    explicit_save_path: Option<&Path>,
) -> Result<PathBuf, String> {
    persist_model_handle_with_options(handle, explicit_save_path, false)
}

/// Save a `CarbonSavedModelHandle` with explicit control over whether `capability_token` is
/// embedded inside a user-specified `--save <PATH>` file (`--embed-token`).
pub fn persist_model_handle_with_options(
    handle: &CarbonSavedModelHandle,
    explicit_save_path: Option<&Path>,
    embed_token: bool,
) -> Result<PathBuf, String> {
    let reg_payload =
        Zeroizing::new(serde_json::to_vec_pretty(handle).err_ctx("serialize model handle")?);
    let reg_dir = runs_registry_dir()?;
    fs::create_dir_all(&reg_dir)
        .map_err(|e| format!("create runs registry {}: {e}", reg_dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&reg_dir, fs::Permissions::from_mode(0o700));
    }
    let reg_path = reg_dir.join(format!("{}.json", handle.run_id));
    write_secret_json_0600(&reg_path, &reg_payload)?;

    if let Some(user_path) = explicit_save_path {
        if embed_token {
            write_secret_json_0600(user_path, &reg_payload)?;
        } else {
            let mut stripped = handle.clone();
            stripped.capability_token = None;
            let user_payload = Zeroizing::new(
                serde_json::to_vec_pretty(&stripped).err_ctx("serialize stripped model handle")?,
            );
            write_secret_json_0600(user_path, &user_payload)?;
        }
        return Ok(user_path.to_path_buf());
    }
    Ok(reg_path)
}

/// Resolve a `CarbonSavedModelHandle` from `--model <PATH>`, `--run <RUN_ID>`, or `latest`.
pub fn load_model_handle_by_spec(
    model_path: Option<&Path>,
    run_id: Option<&str>,
) -> Result<(CarbonSavedModelHandle, PathBuf), String> {
    let target_path = if let Some(p) = model_path {
        p.to_path_buf()
    } else if let Some(rid) = run_id {
        let trimmed = rid.trim();
        if trimmed.eq_ignore_ascii_case("latest") {
            let entries = list_saved_model_handles()?;
            let latest = entries.first().ok_or_else(|| {
                "no saved runs found locally; train a persistent model first with `carbon fit <CSV> --target <COL>`"
                    .to_owned()
            })?;
            PathBuf::from(&latest.handle_path)
        } else {
            runs_registry_dir()?.join(format!("{trimmed}.json"))
        }
    } else {
        let entries = list_saved_model_handles()?;
        let latest = entries.first().ok_or_else(|| {
            "pass --model <FILE.carbon.json> or --run <RUN_ID>, or train first with `carbon fit <CSV>`"
                .to_owned()
        })?;
        PathBuf::from(&latest.handle_path)
    };

    let raw = Zeroizing::new(
        fs::read(&target_path)
            .map_err(|e| format!("read model handle {}: {e}", target_path.display()))?,
    );
    let handle: CarbonSavedModelHandle = serde_json::from_slice(&raw)
        .map_err(|e| format!("parse model handle {}: {e}", target_path.display()))?;
    if handle.schema != CARBON_MODEL_HANDLE_SCHEMA {
        return Err(format!(
            "unsupported model handle schema '{}' in {}",
            handle.schema,
            target_path.display()
        ));
    }
    Ok((handle, target_path))
}

/// List all locally saved zero-knowledge run handles sorted newest-first (`carbon runs`).
pub fn list_saved_model_handles() -> Result<Vec<CarbonRunRegistryEntry>, String> {
    let reg_dir = runs_registry_dir()?;
    if !reg_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let rd =
        fs::read_dir(&reg_dir).map_err(|e| format!("read runs dir {}: {e}", reg_dir.display()))?;
    for entry_res in rd {
        let Ok(entry) = entry_res else {
            continue;
        };
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        };
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        let z_bytes = Zeroizing::new(bytes);
        let Ok(h) = serde_json::from_slice::<CarbonSavedModelHandle>(&z_bytes) else {
            continue;
        };
        if h.schema != CARBON_MODEL_HANDLE_SCHEMA {
            continue;
        }
        out.push(CarbonRunRegistryEntry {
            run_id: h.run_id,
            created_epoch_ms: h.created_epoch_ms,
            tenant_id: h.tenant_id,
            tier: h.tier,
            region: h.region,
            endpoint: h.endpoint,
            target_column: h.encoder.target_column,
            feature_count: h.encoder.encoded_feature_names.len(),
            fit_rows: h.fit_rows,
            holdout_rmse: h.holdout_rmse,
            holdout_r2: h.holdout_r2,
            handle_path: path.display().to_string(),
        });
    }
    out.sort_by_key(|b| std::cmp::Reverse(b.created_epoch_ms));
    Ok(out)
}

fn resolve_fallback_token_for_handle(handle: &CarbonSavedModelHandle) -> Option<String> {
    if let Ok(env_tok) = std::env::var("CARBON_AUTH_TOKEN") {
        let trimmed = env_tok.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_owned());
        }
    }
    if let Ok(reg_dir) = runs_registry_dir() {
        let reg_path = reg_dir.join(format!("{}.json", handle.run_id));
        if let Ok(raw) = fs::read(&reg_path) {
            let z_raw = Zeroizing::new(raw);
            if let Ok(reg_handle) = serde_json::from_slice::<CarbonSavedModelHandle>(&z_raw) {
                if reg_handle.tenant_id == handle.tenant_id {
                    if let Some(tok) = reg_handle.capability_token.filter(|s| !s.trim().is_empty())
                    {
                        return Some(tok);
                    }
                }
            }
        }
    }
    let norm_ep = handle.endpoint.trim().trim_end_matches('/');
    if let Some(cloud_name) = handle.cloud.as_deref() {
        if let Ok(Some(cred)) = load_stored_cloud_token(cloud_name) {
            let cred_ep = cred.endpoint.trim().trim_end_matches('/');
            let endpoint_belongs_to_cloud = cred_ep == norm_ep
                || resolve_public_cloud_endpoint(cloud_name, None)
                    .map(|entry| entry.candidate_endpoints().iter().any(|ep| ep == norm_ep))
                    .unwrap_or(false);
            if cred.tenant_id == handle.tenant_id
                && endpoint_belongs_to_cloud
                && !cred.capability_token.trim().is_empty()
            {
                return Some(cred.capability_token);
            }
        }
    }
    if let Ok(all_creds) = list_stored_cloud_tokens() {
        for cred in all_creds.values() {
            let cred_ep = cred.endpoint.trim().trim_end_matches('/');
            if cred_ep == norm_ep && !cred.capability_token.trim().is_empty() {
                return Some(cred.capability_token.clone());
            }
        }
    }
    None
}

/// Build a pinned zero-knowledge `CarbonThinClient` initialized with a handle's `U_DEK` and token.
pub fn build_client_for_handle(
    handle: &CarbonSavedModelHandle,
    override_auth_token: Option<String>,
) -> Result<CarbonThinClient, String> {
    let mut client = if let Some(kek) = handle
        .engine_kek_pubkey_b64
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        CarbonThinClient::from_pinned_handle(handle.endpoint.clone(), kek)?
    } else {
        CarbonThinClient::connect_fastest_pinned(vec![handle.endpoint.clone()], &BTreeMap::new())?.0
    };
    let resolved_token = override_auth_token
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            handle
                .capability_token
                .clone()
                .filter(|s| !s.trim().is_empty())
        })
        .or_else(|| resolve_fallback_token_for_handle(handle));
    if let Some(tok) = resolved_token {
        client = client.with_capability_token(tok);
    }
    if let Some(dek_b64) = &handle.u_dek_b64 {
        let raw = Zeroizing::new(
            base64::engine::general_purpose::STANDARD
                .decode(dek_b64.trim())
                .err_ctx("invalid u_dek_b64 in model handle")?,
        );
        if raw.len() != 32 {
            return Err(format!(
                "invalid u_dek byte length {} in model handle (expected 32)",
                raw.len()
            ));
        }
        let mut dek = [0u8; 32];
        dek.copy_from_slice(&raw);
        client.import_run_udek(&handle.tenant_id, &handle.run_id, dek);
    }
    Ok(client)
}

/// Compute `MAE`, `RMSE`, `R^2`, and `p50`/`p90`/`p99` absolute error quantiles.
pub fn compute_residual_quantiles(
    y_true: &[f64],
    y_pred: &[f64],
    reference_mean: Option<f64>,
) -> Option<CarbonResidualQuantiles> {
    if y_true.is_empty() || y_true.len() != y_pred.len() {
        return None;
    }
    let n = y_true.len() as f64;
    let mut abs_errs = Vec::with_capacity(y_true.len());
    let mut sse = 0.0;
    let mut sae = 0.0;
    for (&yt, &yp) in y_true.iter().zip(y_pred.iter()) {
        if !yt.is_finite() || !yp.is_finite() {
            return None;
        }
        let ae = (yt - yp).abs();
        abs_errs.push(ae);
        sae += ae;
        sse += ae * ae;
    }
    abs_errs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean_ref = reference_mean.unwrap_or_else(|| y_true.iter().sum::<f64>() / n);
    let sst: f64 = y_true
        .iter()
        .map(|&y| {
            let d = y - mean_ref;
            d * d
        })
        .sum();
    let r2 = if sst > 1e-18 {
        Some(1.0 - (sse / sst))
    } else {
        None
    };
    let q_idx = |q: f64| -> usize {
        (((abs_errs.len() - 1) as f64) * q)
            .round()
            .clamp(0.0, (abs_errs.len() - 1) as f64) as usize
    };
    Some(CarbonResidualQuantiles {
        mae: sae / n,
        rmse: (sse / n).sqrt(),
        r2,
        abs_err_p50: abs_errs[q_idx(0.50)],
        abs_err_p90: abs_errs[q_idx(0.90)],
        abs_err_p99: abs_errs[q_idx(0.99)],
        max_abs_err: abs_errs.last().copied().unwrap_or(0.0),
    })
}

/// Score an unlabeled (or labeled) CSV file against a saved zero-knowledge Carbon model handle,
/// transparently chunking batches and writing `--out predictions.csv` (or `-` for stdout).
#[allow(clippy::too_many_arguments)]
pub fn execute_predict_csv_with_handle(
    handle: &CarbonSavedModelHandle,
    csv_path: &Path,
    output_csv: Option<&Path>,
    delimiter_override: Option<char>,
    no_header: bool,
    id_column: Option<&str>,
    include_inputs: bool,
    override_auth_token: Option<String>,
) -> Result<CarbonPredictCsvReceipt, String> {
    let t0 = Instant::now();
    let has_header = if no_header {
        false
    } else {
        handle.encoder.had_header
    };
    let table = load_delimited_table_with_options(
        csv_path,
        delimiter_override,
        has_header,
        &handle.encoder.extra_na_tokens,
        handle.encoder.inf_as_nan,
    )?;
    let encoded_rows = handle.encoder.encode_prediction_table(&table)?;
    if encoded_rows.is_empty() {
        return Err(format!(
            "CSV {} has zero rows to predict",
            csv_path.display()
        ));
    }

    let client = build_client_for_handle(handle, override_auth_token)?;
    let ptier = match handle.tier {
        CarbonTierWire::Free => CarbonProductTier::Free,
        CarbonTierWire::Pro => CarbonProductTier::Pro,
        CarbonTierWire::Enterprise => CarbonProductTier::Enterprise,
    };
    let chunk_cap = ptier
        .max_predict_rows_per_call()
        .unwrap_or(50_000)
        .min(5_000);
    let mut all_predictions = Vec::with_capacity(encoded_rows.len());
    let mut batches = 0usize;

    for chunk in encoded_rows.chunks(chunk_cap) {
        let resp = client.predict(
            &handle.tenant_id,
            &handle.run_id,
            handle.encoder.encoded_feature_names.clone(),
            chunk.to_vec(),
        )?;
        if resp.predictions.len() != chunk.len() {
            return Err(format!(
                "predict batch returned {} rows, expected {}",
                resp.predictions.len(),
                chunk.len()
            ));
        }
        all_predictions.extend_from_slice(&resp.predictions);
        batches += 1;
    }

    // If the input CSV also includes the target column with finite values (or mapped binary labels),
    // compute out-of-sample evaluation metrics!
    let eval_if_labeled = table
        .headers
        .iter()
        .position(|h| h == &handle.encoder.target_column)
        .and_then(|t_idx| {
            let mut y_true = Vec::with_capacity(table.rows.len());
            for r in &table.rows {
                let raw_cell = r[t_idx].as_deref()?;
                let val = match &handle.encoder.target_binary_map {
                    Some(bm) => bm.encode_label(raw_cell)?,
                    None => parse_flexible_numeric_f64(raw_cell)?,
                };
                y_true.push(val);
            }
            compute_residual_quantiles(&y_true, &all_predictions, None)
        });

    if let Some(out_path) = output_csv {
        let to_stdout = out_path == Path::new("-");
        let id_idx = match id_column {
            Some(col) => Some(table.headers.iter().position(|h| h == col).ok_or_else(|| {
                format!(
                    "--id-column '{col}' not found in [{}]",
                    table.headers.join(", ")
                )
            })?),
            None => None,
        };
        let push_csv_cell = |buf: &mut String, val: &str| {
            if val.contains(',') || val.contains('"') || val.contains('\n') || val.contains('\r') {
                buf.push('"');
                buf.push_str(&val.replace('"', "\"\""));
                buf.push('"');
            } else {
                buf.push_str(val);
            }
        };
        let write_pred_tail = |buf: &mut String, pred: f64| {
            if let Some(bm) = &handle.encoder.target_binary_map {
                let (prob, lbl) = bm.decode_prediction(pred, 0.5);
                buf.push_str(&format!("{pred:.8},{prob:.8},"));
                push_csv_cell(buf, lbl);
                buf.push('\n');
            } else {
                buf.push_str(&format!("{pred:.8}\n"));
            }
        };
        let pred_header = if handle.encoder.target_binary_map.is_some() {
            "prediction,predicted_probability,predicted_label\n"
        } else {
            "prediction\n"
        };
        let mut csv_buf = String::with_capacity(all_predictions.len() * 32 + 128);
        if include_inputs {
            for h in &table.headers {
                push_csv_cell(&mut csv_buf, h);
                csv_buf.push(',');
            }
            csv_buf.push_str(pred_header);
            for (row, &pred) in table.rows.iter().zip(all_predictions.iter()) {
                for cell in row {
                    if let Some(s) = cell {
                        push_csv_cell(&mut csv_buf, s);
                    }
                    csv_buf.push(',');
                }
                write_pred_tail(&mut csv_buf, pred);
            }
        } else if let Some(idx) = id_idx {
            push_csv_cell(&mut csv_buf, &table.headers[idx]);
            csv_buf.push(',');
            csv_buf.push_str(pred_header);
            for (row, &pred) in table.rows.iter().zip(all_predictions.iter()) {
                let id_val = row[idx].as_deref().unwrap_or("");
                push_csv_cell(&mut csv_buf, id_val);
                csv_buf.push(',');
                write_pred_tail(&mut csv_buf, pred);
            }
        } else {
            csv_buf.push_str(pred_header);
            for &pred in &all_predictions {
                write_pred_tail(&mut csv_buf, pred);
            }
        }

        if to_stdout {
            if let Err(false) = write_stdout_line(csv_buf.trim_end_matches('\n')) {
                return Err("write predictions CSV to stdout failed".to_owned());
            }
        } else {
            if let Some(parent) = out_path.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)
                        .map_err(|e| format!("create {}: {e}", parent.display()))?;
                }
            }
            fs::write(out_path, csv_buf.as_bytes())
                .map_err(|e| format!("write {}: {e}", out_path.display()))?;
        }
    }

    let mut le_bytes = Vec::with_capacity(all_predictions.len() * 8);
    for &p in &all_predictions {
        le_bytes.extend_from_slice(&p.to_le_bytes());
    }
    let predictions_sha256 = sha256_lower_hex(&le_bytes);

    Ok(CarbonPredictCsvReceipt {
        run_id: handle.run_id.clone(),
        tenant_id: handle.tenant_id.clone(),
        endpoint: handle.endpoint.clone(),
        region: handle.region.clone(),
        encrypted_cuen: handle.encrypted_cuen,
        input_csv: csv_path.display().to_string(),
        output_csv: output_csv.map(|p| p.display().to_string()),
        rows_scored: all_predictions.len(),
        batches,
        wall_ms: t0.elapsed().as_secs_f64() * 1_000.0,
        predictions_sha256,
        evaluation_if_labeled: eval_if_labeled,
        sample_predictions: all_predictions.into_iter().take(10).collect(),
    })
}

/// Query remote status for a saved model handle (`carbon status`).
pub fn query_handle_status(
    handle: &CarbonSavedModelHandle,
    override_auth_token: Option<String>,
) -> Result<CarbonPublicWaveProgress, String> {
    let client = build_client_for_handle(handle, override_auth_token)?;
    client.get_run_status(&handle.tenant_id, &handle.run_id)
}

/// Cryptographically shred remote S3/CAS objects AND zero-overwrite local key files (`carbon shred`).
pub fn shred_handle_and_local_keys(
    handle: &CarbonSavedModelHandle,
    explicit_handle_path: Option<&Path>,
    override_auth_token: Option<String>,
) -> Result<(CarbonShredRunResponse, bool), String> {
    let client = build_client_for_handle(handle, override_auth_token)?;
    let remote = client.shred_run(&handle.tenant_id, &handle.run_id)?;
    let mut local_deleted = false;
    if let Ok(reg_dir) = runs_registry_dir() {
        let reg_file = reg_dir.join(format!("{}.json", handle.run_id));
        if shred_local_file(&reg_file)? {
            local_deleted = true;
        }
    }
    if let Some(p) = explicit_handle_path {
        if shred_local_file(p)? {
            local_deleted = true;
        }
    }
    Ok((remote, local_deleted))
}

/// `CarbonUpgradeReceipt` is a struct returned by `carbon upgrade [--check]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonUpgradeReceipt {
    pub current_version: String,
    pub current_release_tag: String,
    pub target_release_tag: String,
    pub asset_name: String,
    pub expected_sha256: String,
    pub signature_verified: bool,
    pub check_only: bool,
    pub up_to_date: bool,
    pub updated_binary_path: Option<String>,
}

fn current_release_asset_candidates() -> &'static [&'static str] {
    match CARBON_BUILD_TARGET {
        t if t.contains("x86_64") && t.contains("linux") => &["carbon-linux-x86_64-musl"],
        t if t.contains("aarch64") && t.contains("linux") => &["carbon-linux-aarch64-musl"],
        t if (t.starts_with("arm") || t.contains("armv7")) && t.contains("linux") => {
            &["carbon-linux-armv7-musleabihf"]
        }
        t if t.contains("aarch64") && t.contains("apple-darwin") => {
            &["carbon-darwin-arm64", "carbon-darwin-universal"]
        }
        t if t.contains("x86_64") && t.contains("apple-darwin") => {
            &["carbon-darwin-x86_64", "carbon-darwin-universal"]
        }
        t if t.contains("x86_64") && t.contains("windows") => &["carbon-windows-x86_64-msvc.exe"],
        t if t.contains("aarch64") && t.contains("windows") => &["carbon-windows-aarch64-msvc.exe"],
        _ => &[],
    }
}

fn extract_carbon_from_tar_reader<R: std::io::Read>(
    mut reader: R,
    max_bytes: u64,
) -> Result<Vec<u8>, String> {
    let mut consumed: u64 = 0;
    let mut hdr = [0u8; 512];
    loop {
        let mut read_hdr = 0usize;
        while read_hdr < 512 {
            let n = reader
                .read(&mut hdr[read_hdr..])
                .err_ctx("read upgrade tar header")?;
            if n == 0 {
                if read_hdr == 0 {
                    return Err(
                        "upgrade tar archive did not contain a 'carbon' regular executable entry"
                            .to_owned(),
                    );
                }
                return Err("truncated 512-byte header in upgrade tar".to_owned());
            }
            read_hdr += n;
        }
        consumed = consumed.saturating_add(512);
        if consumed > max_bytes {
            return Err("upgrade tar archive exceeded 512 MiB safety bound".to_owned());
        }
        if hdr.iter().all(|&b| b == 0) {
            break;
        }
        let raw_name = &hdr[0..100];
        let name_end = raw_name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(raw_name.len());
        let entry_name = std::str::from_utf8(&raw_name[..name_end])
            .map_err(|_| "non-UTF-8 entry name in upgrade tar".to_owned())?;
        let size_field = std::str::from_utf8(&hdr[124..136])
            .map_err(|_| "invalid tar octal size header".to_owned())?
            .trim_matches(|c: char| c == '\0' || c.is_ascii_whitespace());
        let size = if size_field.is_empty() {
            0usize
        } else {
            usize::from_str_radix(size_field, 8)
                .map_err(|e| format!("parse tar octal size '{size_field}': {e}"))?
        };
        let size_u64 = u64::try_from(size).map_err(|_| "tar entry size overflow".to_owned())?;
        if consumed.saturating_add(size_u64) > max_bytes {
            return Err("upgrade tar archive exceeded 512 MiB safety bound".to_owned());
        }
        let typeflag = hdr[156];
        let base_name = entry_name.rsplit('/').next().unwrap_or(entry_name);
        if (typeflag == 0 || typeflag == b'0')
            && (base_name == "carbon" || base_name == "carbon.exe")
        {
            let mut payload = vec![0u8; size];
            reader
                .read_exact(&mut payload)
                .err_ctx("truncated carbon executable in upgrade tar")?;
            return Ok(payload);
        }
        let padded = size_u64
            .checked_add(511)
            .map(|n| n & !511)
            .ok_or_else(|| "tar block alignment overflow".to_owned())?;
        consumed = consumed.saturating_add(padded);
        if consumed > max_bytes {
            return Err("upgrade tar archive exceeded 512 MiB safety bound".to_owned());
        }
        let skipped = std::io::copy(
            &mut std::io::Read::take(&mut reader, padded),
            &mut std::io::sink(),
        )
        .err_ctx("skip non-matching tar entry")?;
        if skipped != padded {
            return Err("truncated tar entry in upgrade archive".to_owned());
        }
    }
    Err("upgrade tar archive did not contain a 'carbon' regular executable entry".to_owned())
}

const MAX_DECOMPRESSED_UPGRADE_BYTES: u64 = 512 * 1024 * 1024;

fn extract_carbon_binary_from_targz(targz_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let decoder = flate2::read::GzDecoder::new(targz_bytes);
    extract_carbon_from_tar_reader(decoder, MAX_DECOMPRESSED_UPGRADE_BYTES)
}

fn extract_carbon_binary_from_tarzst(tarzst_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let decoder = ruzstd::decoding::StreamingDecoder::new(tarzst_bytes)
        .err_ctx("init zstd decoder for upgrade .tar.zst")?;
    extract_carbon_from_tar_reader(decoder, MAX_DECOMPRESSED_UPGRADE_BYTES)
}

fn extract_carbon_binary_from_tarxz(tarxz_bytes: &[u8]) -> Result<Vec<u8>, String> {
    use std::process::{Command, Stdio};
    let mut child = Command::new("xz")
        .arg("-dc")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .err_ctx("spawn xz -dc for upgrade .tar.xz")?;
    let payload = tarxz_bytes.to_vec();
    let writer_handle = child.stdin.take().map(|mut stdin| {
        std::thread::spawn(move || {
            let _ = std::io::Write::write_all(&mut stdin, &payload);
        })
    });
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| "xz -dc missing stdout pipe".to_owned())?;
    let tar_res = crate::storage::cas::read_bounded(
        &mut stdout,
        MAX_DECOMPRESSED_UPGRADE_BYTES,
        "xz -dc upgrade stream",
    );
    let tar_bytes = match tar_res {
        Ok(b) => b,
        Err(err) => {
            let _ = child.kill();
            if let Some(h) = writer_handle {
                let _ = h.join();
            }
            let _ = child.wait();
            return Err(err);
        }
    };
    if let Some(h) = writer_handle {
        let _ = h.join();
    }
    let status = child.wait().err_ctx("wait xz -dc for upgrade .tar.xz")?;
    if !status.success() {
        return Err(format!("xz -dc exited with {status}"));
    }
    extract_carbon_from_tar_reader(
        std::io::Cursor::new(tar_bytes),
        MAX_DECOMPRESSED_UPGRADE_BYTES,
    )
}

/// Remove any stale `*.exe.old.*` replacement backups left alongside `exe_path` from prior Windows upgrades.
pub fn sweep_stale_windows_upgrade_backups(exe_path: &Path) {
    let Some(parent) = exe_path.parent() else {
        return;
    };
    let Ok(entries) = fs::read_dir(parent) else {
        return;
    };
    for entry in entries.flatten() {
        let fname = entry.file_name();
        let name = fname.to_string_lossy();
        if name.contains(".exe.old.") {
            let _ = fs::remove_file(entry.path());
        }
    }
}

fn discover_latest_carbon_cli_tag() -> Option<String> {
    let api_url = std::env::var("CARBON_GITHUB_RELEASES_API_URL")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| {
            "https://api.github.com/repos/neverhuman/JopeDime/releases?per_page=20".to_owned()
        });
    validate_endpoint_transport(&api_url).ok()?;
    let mut resp = crate::tls::carbon_probe_https_agent()
        .get(&api_url)
        .config()
        .http_status_as_error(false)
        .build()
        .header("Accept", "application/vnd.github+json")
        .header("User-Agent", &format!("jope-carbon/{CARBON_PKG_VERSION}"))
        .call()
        .ok()?;
    if resp.status().as_u16() != 200 {
        return None;
    }
    let body = crate::storage::cas::read_bounded(
        resp.body_mut().as_reader(),
        2 * 1024 * 1024,
        "GitHub releases list",
    )
    .ok()?;
    let items: Vec<serde_json::Value> = serde_json::from_slice(&body).ok()?;
    for item in items {
        let draft = item.get("draft").and_then(|v| v.as_bool()).unwrap_or(false);
        let prerelease = item
            .get("prerelease")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let Some(tag) = item.get("tag_name").and_then(|v| v.as_str()) else {
            continue;
        };
        if !draft && !prerelease && tag.starts_with("carbon-cli-v") {
            return Some(tag.to_owned());
        }
    }
    None
}

fn extract_carbon_exe_from_zip(zip_bytes: &[u8]) -> Result<Vec<u8>, String> {
    // Locate End of Central Directory (`PK\x05\x06`) from the tail of `zip_bytes`.
    if zip_bytes.len() < 22 {
        return Err("upgrade .zip archive is truncated (< 22 bytes)".to_owned());
    }
    let search_start = zip_bytes.len().saturating_sub(65_557);
    let mut eocd_pos = None;
    for pos in (search_start..=zip_bytes.len() - 22).rev() {
        if &zip_bytes[pos..pos + 4] == b"PK\x05\x06" {
            eocd_pos = Some(pos);
            break;
        }
    }
    let eocd =
        eocd_pos.ok_or_else(|| "missing End-of-Central-Directory in upgrade .zip".to_owned())?;
    let total_entries = u16::from_le_bytes([zip_bytes[eocd + 10], zip_bytes[eocd + 11]]) as usize;
    let cd_offset = u32::from_le_bytes([
        zip_bytes[eocd + 16],
        zip_bytes[eocd + 17],
        zip_bytes[eocd + 18],
        zip_bytes[eocd + 19],
    ]) as usize;
    let mut cursor = cd_offset;
    for _ in 0..total_entries {
        if cursor + 46 > zip_bytes.len() || &zip_bytes[cursor..cursor + 4] != b"PK\x01\x02" {
            return Err("corrupt Central Directory header in upgrade .zip".to_owned());
        }
        let method = u16::from_le_bytes([zip_bytes[cursor + 10], zip_bytes[cursor + 11]]);
        let comp_size = u32::from_le_bytes([
            zip_bytes[cursor + 20],
            zip_bytes[cursor + 21],
            zip_bytes[cursor + 22],
            zip_bytes[cursor + 23],
        ]) as usize;
        let uncomp_size = u32::from_le_bytes([
            zip_bytes[cursor + 24],
            zip_bytes[cursor + 25],
            zip_bytes[cursor + 26],
            zip_bytes[cursor + 27],
        ]) as u64;
        let name_len =
            u16::from_le_bytes([zip_bytes[cursor + 28], zip_bytes[cursor + 29]]) as usize;
        let extra_len =
            u16::from_le_bytes([zip_bytes[cursor + 30], zip_bytes[cursor + 31]]) as usize;
        let comment_len =
            u16::from_le_bytes([zip_bytes[cursor + 32], zip_bytes[cursor + 33]]) as usize;
        let local_offset = u32::from_le_bytes([
            zip_bytes[cursor + 42],
            zip_bytes[cursor + 43],
            zip_bytes[cursor + 44],
            zip_bytes[cursor + 45],
        ]) as usize;
        let name_end = cursor + 46 + name_len;
        if name_end > zip_bytes.len() {
            return Err("truncated filename in upgrade .zip Central Directory".to_owned());
        }
        let name = std::str::from_utf8(&zip_bytes[cursor + 46..name_end]).unwrap_or("");
        let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
        if base.eq_ignore_ascii_case("carbon.exe") {
            if uncomp_size > MAX_DECOMPRESSED_UPGRADE_BYTES {
                return Err("upgrade .zip entry exceeded 512 MiB safety bound".to_owned());
            }
            if local_offset + 30 > zip_bytes.len()
                || &zip_bytes[local_offset..local_offset + 4] != b"PK\x03\x04"
            {
                return Err("corrupt Local File Header in upgrade .zip".to_owned());
            }
            let l_name_len =
                u16::from_le_bytes([zip_bytes[local_offset + 26], zip_bytes[local_offset + 27]])
                    as usize;
            let l_extra_len =
                u16::from_le_bytes([zip_bytes[local_offset + 28], zip_bytes[local_offset + 29]])
                    as usize;
            let data_start = local_offset + 30 + l_name_len + l_extra_len;
            let data_end = data_start
                .checked_add(comp_size)
                .filter(|&end| end <= zip_bytes.len())
                .ok_or_else(|| "truncated compressed data in upgrade .zip".to_owned())?;
            let slice = &zip_bytes[data_start..data_end];
            return match method {
                0 => Ok(slice.to_vec()),
                8 => {
                    let mut dec = flate2::read::DeflateDecoder::new(slice)
                        .take(MAX_DECOMPRESSED_UPGRADE_BYTES + 1);
                    let mut out = Vec::with_capacity(uncomp_size as usize);
                    dec.read_to_end(&mut out)
                        .err_ctx("inflate carbon.exe from .zip")?;
                    if (out.len() as u64) > MAX_DECOMPRESSED_UPGRADE_BYTES {
                        return Err("upgrade .zip exceeded 512 MiB safety bound".to_owned());
                    }
                    Ok(out)
                }
                other => Err(format!("unsupported ZIP compression method {other}")),
            };
        }
        cursor = name_end + extra_len + comment_len;
    }
    Err("upgrade .zip archive did not contain 'carbon.exe'".to_owned())
}

fn validate_clouds_manifest_bytes(bytes: &[u8]) -> Result<(), String> {
    let manifest: CarbonPublicCloudsManifest =
        serde_json::from_slice(bytes).err_ctx("invalid carbon-clouds.json schema")?;
    if manifest.clouds.is_empty() {
        return Err("carbon-clouds.json contains zero cloud entries".to_owned());
    }
    for (name, entry) in &manifest.clouds {
        if entry.public_endpoint.trim().is_empty() {
            return Err(format!(
                "carbon-clouds.json cloud '{name}' has empty public_endpoint"
            ));
        }
        if let Some(kek) = &entry.engine_kek_pubkey_b64 {
            let _ = decode_p256_pubkey_b64(kek)
                .map_err(|e| format!("invalid KEK pin for '{name}': {e}"))?;
        }
        for reg in &entry.regions {
            if let Some(kek) = &reg.engine_kek_pubkey_b64 {
                let _ = decode_p256_pubkey_b64(kek).map_err(|e| {
                    format!("invalid KEK pin for '{name}' region '{}': {e}", reg.region)
                })?;
            }
        }
    }
    Ok(())
}

/// Check or install a cryptographically signed Carbon CLI update (`SHA256SUMS.txt` + ECDSA-P256 `.sig`).
pub fn execute_self_upgrade(
    base_url_override: Option<&str>,
    target_tag: Option<&str>,
    signing_pub_pem: &str,
    check_only: bool,
) -> Result<CarbonUpgradeReceipt, String> {
    execute_self_upgrade_with_options(
        base_url_override,
        target_tag,
        signing_pub_pem,
        check_only,
        false,
    )
}

/// Check or install a cryptographically signed Carbon CLI update with explicit operator-replace guard control.
pub fn execute_self_upgrade_with_options(
    base_url_override: Option<&str>,
    target_tag: Option<&str>,
    signing_pub_pem: &str,
    check_only: bool,
    force_operator_replace: bool,
) -> Result<CarbonUpgradeReceipt, String> {
    if !check_only && cfg!(feature = "engine") && !force_operator_replace {
        return Err(
            "current binary includes full operator/engine features (`--features engine`); `carbon upgrade` installs the thin public CLI (`--features cli`). Pass `--force` to replace it intentionally or rebuild from source."
                .to_owned(),
        );
    }

    let candidates = current_release_asset_candidates();
    if candidates.is_empty() {
        return Err(format!(
            "automatic binary upgrade is not supported on target '{CARBON_BUILD_TARGET}'"
        ));
    }

    if let Ok(cur_exe) = std::env::current_exe() {
        sweep_stale_windows_upgrade_backups(&cur_exe);
    }

    let discovered_tag = if base_url_override.is_none() && target_tag.is_none() {
        discover_latest_carbon_cli_tag()
    } else {
        None
    };
    let effective_tag = target_tag.or(discovered_tag.as_deref());

    let (primary_base_url, fallback_base_url) = if let Some(b) = base_url_override {
        (b.trim_end_matches('/').to_owned(), None)
    } else if let Some(tag) = effective_tag {
        let gh = format!("https://github.com/neverhuman/JopeDime/releases/download/{tag}");
        let dl = format!("https://downloads.neverhuman.org/carbon/releases/{tag}");
        if telemetry_minimized() {
            (gh, Some(dl))
        } else {
            (dl, Some(gh))
        }
    } else {
        let gh = "https://github.com/neverhuman/JopeDime/releases/latest/download".to_owned();
        let dl = "https://downloads.neverhuman.org/carbon/releases/latest".to_owned();
        if telemetry_minimized() {
            (gh, Some(dl))
        } else {
            (dl, Some(gh))
        }
    };

    let fetch_single = |url: &str| -> Result<Vec<u8>, String> {
        if !url.starts_with("https://") && !url.starts_with("http://") {
            return fs::read(url).map_err(|e| format!("read local upgrade asset {url}: {e}"));
        }
        validate_endpoint_transport(url)?;
        let mut current = url.to_owned();
        for _hop in 0..=5 {
            validate_endpoint_transport(&current)?;
            let mut resp = crate::tls::carbon_https_agent()
                .get(&current)
                .config()
                .http_status_as_error(false)
                .build()
                .call()
                .map_err(|e| format!("GET {current} failed: {e}"))?;
            let status = resp.status().as_u16();
            if matches!(status, 301 | 302 | 303 | 307 | 308) {
                let loc = resp
                    .headers()
                    .get("location")
                    .and_then(|v| v.to_str().ok())
                    .ok_or_else(|| {
                        format!("GET {current} returned redirect HTTP {status} without Location")
                    })?;
                current = resolve_clouds_redirect_target(&current, loc)?;
                continue;
            }
            let buf = crate::storage::cas::read_bounded(
                resp.body_mut().as_reader(),
                64 * 1024 * 1024,
                &format!("GET {current}"),
            )?;
            if status >= 300 {
                return Err(format!(
                    "GET {current} returned HTTP {status}: {}",
                    String::from_utf8_lossy(&buf)
                ));
            }
            return Ok(buf);
        }
        Err(format!("GET {url} exceeded maximum 5 redirects"))
    };

    let fetch_asset = |asset_name: &str| -> Result<Vec<u8>, String> {
        let primary = format!("{primary_base_url}/{asset_name}");
        match fetch_single(&primary) {
            Ok(bytes) => Ok(bytes),
            Err(primary_err) => {
                if let Some(fb) = &fallback_base_url {
                    let secondary = format!("{fb}/{asset_name}");
                    fetch_single(&secondary).map_err(|fb_err| {
                        format!("{primary_err}; mirror fallback failed: {fb_err}")
                    })
                } else {
                    Err(primary_err)
                }
            }
        }
    };

    let sums_bytes = fetch_asset("SHA256SUMS.txt")?;
    let sig_bytes = fetch_asset("SHA256SUMS.txt.sig")?;
    let sig_b64 = String::from_utf8_lossy(&sig_bytes);
    let sig_der = base64::engine::general_purpose::STANDARD
        .decode(sig_b64.trim())
        .err_ctx("decode SHA256SUMS.txt.sig base64")?;
    verify_ecdsa_p256_spki_pem_signature(signing_pub_pem, &sums_bytes, &sig_der)
        .err_ctx("release signature verification failed")?;

    let sums_text = String::from_utf8_lossy(&sums_bytes);
    let mut verified_sums = BTreeMap::new();
    for line in sums_text.lines() {
        let mut parts = line.split_whitespace();
        if let (Some(sha), Some(name)) = (parts.next(), parts.next()) {
            let clean_name = name.trim_start_matches('*');
            verified_sums.insert(clean_name.to_owned(), sha.to_ascii_lowercase());
        }
    }

    let (asset, expected_sha256) = candidates
        .iter()
        .find_map(|&cand| verified_sums.get(cand).cloned().map(|sha| (cand, sha)))
        .ok_or_else(|| {
            format!(
                "none of candidate assets [{}] found in verified SHA256SUMS.txt",
                candidates.join(", ")
            )
        })?;

    let exe_path = std::env::current_exe().err_ctx("resolve current_exe")?;
    let current_sha = fs::read(&exe_path)
        .ok()
        .map(|b| sha256_lower_hex(&b))
        .unwrap_or_default();
    let up_to_date = current_sha == expected_sha256;
    let resolved_tag = effective_tag.unwrap_or("latest").to_owned();

    if check_only || up_to_date {
        return Ok(CarbonUpgradeReceipt {
            current_version: CARBON_PKG_VERSION.to_owned(),
            current_release_tag: CARBON_RELEASE_TAG.to_owned(),
            target_release_tag: resolved_tag,
            asset_name: asset.to_owned(),
            expected_sha256,
            signature_verified: true,
            check_only,
            up_to_date,
            updated_binary_path: None,
        });
    }

    let tarzst_asset = format!("{asset}.tar.zst");
    let tarxz_asset = format!("{asset}.tar.xz");
    let targz_asset = format!("{asset}.tar.gz");
    let zip_asset = format!("{}.zip", asset.trim_end_matches(".exe"));

    let bin_bytes = if let Some(expected_zst_sha) = verified_sums.get(&tarzst_asset) {
        match fetch_asset(&tarzst_asset) {
            Ok(zst_bytes) => {
                let actual = sha256_lower_hex(&zst_bytes);
                if &actual != expected_zst_sha {
                    return Err(format!(
                        "downloaded {tarzst_asset} SHA-256 mismatch: expected {expected_zst_sha}, got {actual}"
                    ));
                }
                extract_carbon_binary_from_tarzst(&zst_bytes)?
            }
            Err(_) => fetch_asset(asset)?,
        }
    } else if let Some(expected_xz_sha) = verified_sums.get(&tarxz_asset) {
        match fetch_asset(&tarxz_asset) {
            Ok(xz_bytes) => {
                let actual = sha256_lower_hex(&xz_bytes);
                if &actual != expected_xz_sha {
                    return Err(format!(
                        "downloaded {tarxz_asset} SHA-256 mismatch: expected {expected_xz_sha}, got {actual}"
                    ));
                }
                match extract_carbon_binary_from_tarxz(&xz_bytes) {
                    Ok(bin) => bin,
                    Err(_) => fetch_asset(asset)?,
                }
            }
            Err(_) => fetch_asset(asset)?,
        }
    } else if let Some(expected_tgz_sha) = verified_sums.get(&targz_asset) {
        match fetch_asset(&targz_asset) {
            Ok(tgz_bytes) => {
                let actual = sha256_lower_hex(&tgz_bytes);
                if &actual != expected_tgz_sha {
                    return Err(format!(
                        "downloaded {targz_asset} SHA-256 mismatch: expected {expected_tgz_sha}, got {actual}"
                    ));
                }
                extract_carbon_binary_from_targz(&tgz_bytes)?
            }
            Err(_) => fetch_asset(asset)?,
        }
    } else if asset.ends_with(".exe") && verified_sums.contains_key(&zip_asset) {
        let expected_zip_sha = &verified_sums[&zip_asset];
        match fetch_asset(&zip_asset) {
            Ok(zip_bytes) => {
                let actual = sha256_lower_hex(&zip_bytes);
                if &actual != expected_zip_sha {
                    return Err(format!(
                        "downloaded {zip_asset} SHA-256 mismatch: expected {expected_zip_sha}, got {actual}"
                    ));
                }
                extract_carbon_exe_from_zip(&zip_bytes)?
            }
            Err(_) => fetch_asset(asset)?,
        }
    } else {
        fetch_asset(asset)?
    };

    let actual_sha = sha256_lower_hex(&bin_bytes);
    if actual_sha != expected_sha256 {
        return Err(format!(
            "downloaded {asset} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}"
        ));
    }

    static UPGRADE_STAGE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let stage_seq = UPGRADE_STAGE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stage_path =
        exe_path.with_extension(format!("upgrade.{}.{}", std::process::id(), stage_seq));
    {
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            let mut sf = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o755)
                .open(&stage_path)
                .map_err(|e| format!("stage {}: {e}", stage_path.display()))?;
            if let Err(e) = sf.write_all(&bin_bytes).and_then(|()| sf.sync_all()) {
                let _ = fs::remove_file(&stage_path);
                return Err(format!("stage {}: {e}", stage_path.display()));
            }
        }
        #[cfg(not(unix))]
        {
            let mut sf = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&stage_path)
                .map_err(|e| format!("stage {}: {e}", stage_path.display()))?;
            if let Err(e) = sf.write_all(&bin_bytes).and_then(|()| sf.sync_all()) {
                let _ = fs::remove_file(&stage_path);
                return Err(format!("stage {}: {e}", stage_path.display()));
            }
        }
    }
    if let Err(rename_err) = fs::rename(&stage_path, &exe_path) {
        #[cfg(windows)]
        {
            let old_locked = exe_path.with_extension(format!("exe.old.{}", std::process::id()));
            let _ = fs::remove_file(&old_locked);
            fs::rename(&exe_path, &old_locked).map_err(|e| {
                format!(
                    "rename running {} -> {} ({rename_err}): {e}",
                    exe_path.display(),
                    old_locked.display()
                )
            })?;
            fs::rename(&stage_path, &exe_path)
                .map_err(|e| format!("atomic replace {}: {e}", exe_path.display()))?;
        }
        #[cfg(not(windows))]
        {
            let _ = fs::remove_file(&stage_path);
            return Err(format!(
                "atomic replace {}: {rename_err}",
                exe_path.display()
            ));
        }
    }

    // Refresh cached `SHA256SUMS.txt`, `.sig`, and cryptographically verified `carbon-clouds.json`.
    if let Ok(cfg_dir) = verified_absolute_jopedime_config_dir() {
        let _ = write_secret_json_0600(&cfg_dir.join("SHA256SUMS.txt"), &sums_bytes);
        let _ = write_secret_json_0600(&cfg_dir.join("SHA256SUMS.txt.sig"), &sig_bytes);
        if let Some(expected_clouds_sha) = verified_sums.get("carbon-clouds.json") {
            if let Ok(clouds_bytes) = fetch_asset("carbon-clouds.json") {
                if &sha256_lower_hex(&clouds_bytes) == expected_clouds_sha
                    && validate_clouds_manifest_bytes(&clouds_bytes).is_ok()
                {
                    let _ =
                        write_secret_json_0600(&cfg_dir.join("carbon-clouds.json"), &clouds_bytes);
                }
            }
        }
    }

    Ok(CarbonUpgradeReceipt {
        current_version: CARBON_PKG_VERSION.to_owned(),
        current_release_tag: CARBON_RELEASE_TAG.to_owned(),
        target_release_tag: resolved_tag,
        asset_name: asset.to_owned(),
        expected_sha256,
        signature_verified: true,
        check_only: false,
        up_to_date: false,
        updated_binary_path: Some(exe_path.display().to_string()),
    })
}

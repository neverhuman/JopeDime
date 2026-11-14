//! Persistent `0600` zero-knowledge model handles (`model.carbon.json` & `$CARBON_CONFIG_DIR/runs/`),
//! two-step `carbon fit` -> `carbon predict` scoring on unlabeled CSVs, `carbon runs` / `status` /
//! `shred` lifecycle management, and ECDSA-P256 verified `carbon upgrade`.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use base64::Engine as _;
use jope_api::carbon::{CarbonPublicWaveProgress, CarbonShredRunResponse, CarbonTierWire};
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

use crate::{
    cli::{
        client::{
            resolve_clouds_redirect_target, verified_absolute_jopedime_config_dir, CarbonThinClient,
        },
        tabular::{load_delimited_table, CarbonTabularEncoder},
    },
    role::CarbonProductTier,
    security::user_envelope::verify_ecdsa_p256_spki_pem_signature,
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
    #[serde(default)]
    pub capability_token: Option<String>,
    pub encoder: CarbonTabularEncoder,
    pub fit_rows: u32,
    pub holdout_rows: u32,
    pub holdout_rmse: Option<f64>,
    pub holdout_mae: Option<f64>,
    pub holdout_r2: Option<f64>,
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
    let tmp_path = path.with_extension(format!("tmp.{}", std::process::id()));
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .mode(0o600)
            .open(&tmp_path)
            .map_err(|e| format!("open {} (0600): {e}", tmp_path.display()))?;
        f.write_all(bytes)
            .map_err(|e| format!("write {}: {e}", tmp_path.display()))?;
        f.sync_all()
            .map_err(|e| format!("sync {}: {e}", tmp_path.display()))?;
    }
    #[cfg(not(unix))]
    {
        fs::write(&tmp_path, bytes).map_err(|e| format!("write {}: {e}", tmp_path.display()))?;
    }
    fs::rename(&tmp_path, path).map_err(|e| format!("atomic rename -> {}: {e}", path.display()))?;
    Ok(())
}

fn shred_local_file(path: &Path) -> Result<bool, String> {
    if !path.is_file() {
        return Ok(false);
    }
    if let Ok(meta) = fs::metadata(path) {
        let len = (meta.len() as usize).clamp(64, 1_048_576);
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
/// optionally also to an explicit user `--save <PATH>` (`0600`).
pub fn persist_model_handle(
    handle: &CarbonSavedModelHandle,
    explicit_save_path: Option<&Path>,
) -> Result<PathBuf, String> {
    let payload = Zeroizing::new(
        serde_json::to_vec_pretty(handle).map_err(|e| format!("serialize model handle: {e}"))?,
    );
    let reg_dir = runs_registry_dir()?;
    fs::create_dir_all(&reg_dir)
        .map_err(|e| format!("create runs registry {}: {e}", reg_dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&reg_dir, fs::Permissions::from_mode(0o700));
    }
    let reg_path = reg_dir.join(format!("{}.json", handle.run_id));
    write_secret_json_0600(&reg_path, &payload)?;

    if let Some(user_path) = explicit_save_path {
        write_secret_json_0600(user_path, &payload)?;
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
        }
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

fn build_client_for_handle(
    handle: &CarbonSavedModelHandle,
    override_auth_token: Option<String>,
) -> Result<CarbonThinClient, String> {
    let mut pins = BTreeMap::new();
    if let Some(kek) = &handle.engine_kek_pubkey_b64 {
        pins.insert(
            handle.endpoint.trim().trim_end_matches('/').to_owned(),
            kek.clone(),
        );
    }
    let (mut client, _) =
        CarbonThinClient::connect_fastest_pinned(vec![handle.endpoint.clone()], &pins)?;
    if let Some(tok) = override_auth_token.or_else(|| handle.capability_token.clone()) {
        client = client.with_capability_token(tok);
    }
    if let Some(dek_b64) = &handle.u_dek_b64 {
        let raw = Zeroizing::new(
            base64::engine::general_purpose::STANDARD
                .decode(dek_b64.trim())
                .map_err(|e| format!("invalid u_dek_b64 in model handle: {e}"))?,
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
/// transparently chunking batches and writing `--out predictions.csv`.
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
    let table = load_delimited_table(csv_path, delimiter_override, has_header)?;
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

    // If the input CSV also includes the target column with finite values, compute out-of-sample metrics!
    let eval_if_labeled = table
        .headers
        .iter()
        .position(|h| h == &handle.encoder.target_column)
        .and_then(|t_idx| {
            let mut y_true = Vec::with_capacity(table.rows.len());
            for r in &table.rows {
                let val = r[t_idx]
                    .as_deref()?
                    .trim()
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite())?;
                y_true.push(val);
            }
            compute_residual_quantiles(&y_true, &all_predictions, None)
        });

    if let Some(out_path) = output_csv {
        if let Some(parent) = out_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("create {}: {e}", parent.display()))?;
            }
        }
        let id_idx = match id_column {
            Some(col) => Some(table.headers.iter().position(|h| h == col).ok_or_else(|| {
                format!(
                    "--id-column '{col}' not found in [{}]",
                    table.headers.join(", ")
                )
            })?),
            None => None,
        };
        let mut csv_buf = String::with_capacity(all_predictions.len() * 24 + 128);
        if include_inputs {
            for h in &table.headers {
                csv_buf.push_str(h);
                csv_buf.push(',');
            }
            csv_buf.push_str("prediction\n");
            for (row, &pred) in table.rows.iter().zip(all_predictions.iter()) {
                for cell in row {
                    if let Some(s) = cell {
                        if s.contains(',') || s.contains('"') {
                            csv_buf.push('"');
                            csv_buf.push_str(&s.replace('"', "\"\""));
                            csv_buf.push('"');
                        } else {
                            csv_buf.push_str(s);
                        }
                    }
                    csv_buf.push(',');
                }
                csv_buf.push_str(&format!("{pred:.8}\n"));
            }
        } else if let Some(idx) = id_idx {
            csv_buf.push_str(&format!("{},prediction\n", table.headers[idx]));
            for (row, &pred) in table.rows.iter().zip(all_predictions.iter()) {
                let id_val = row[idx].as_deref().unwrap_or("");
                csv_buf.push_str(&format!("{id_val},{pred:.8}\n"));
            }
        } else {
            csv_buf.push_str("prediction\n");
            for &pred in &all_predictions {
                csv_buf.push_str(&format!("{pred:.8}\n"));
            }
        }
        fs::write(out_path, csv_buf.as_bytes())
            .map_err(|e| format!("write {}: {e}", out_path.display()))?;
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

fn current_release_asset_name() -> Option<&'static str> {
    match CARBON_BUILD_TARGET {
        t if t.contains("x86_64") && t.contains("linux") => Some("carbon-linux-x86_64-musl"),
        t if t.contains("aarch64") && t.contains("linux") => Some("carbon-linux-aarch64-musl"),
        t if t.contains("aarch64") && t.contains("apple-darwin") => Some("carbon-darwin-arm64"),
        t if t.contains("x86_64") && t.contains("apple-darwin") => Some("carbon-darwin-x86_64"),
        _ => None,
    }
}

/// Check or install a cryptographically signed Carbon CLI update (`SHA256SUMS.txt` + ECDSA-P256 `.sig`).
pub fn execute_self_upgrade(
    base_url_override: Option<&str>,
    target_tag: Option<&str>,
    signing_pub_pem: &str,
    check_only: bool,
) -> Result<CarbonUpgradeReceipt, String> {
    let asset = current_release_asset_name().ok_or_else(|| {
        format!("automatic binary upgrade is not supported on target '{CARBON_BUILD_TARGET}'")
    })?;
    let base_url = if let Some(b) = base_url_override {
        b.trim_end_matches('/').to_owned()
    } else if let Some(tag) = target_tag {
        format!("https://github.com/neverhuman/JopeDime/releases/download/{tag}")
    } else {
        "https://github.com/neverhuman/JopeDime/releases/latest/download".to_owned()
    };

    let fetch_bytes = |url: &str| -> Result<Vec<u8>, String> {
        if !url.starts_with("https://") && !url.starts_with("http://") {
            return fs::read(url).map_err(|e| format!("read local upgrade asset {url}: {e}"));
        }
        let mut current = url.to_owned();
        for _hop in 0..=5 {
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
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut resp.body_mut().as_reader(), &mut buf)
                .map_err(|e| format!("read {current}: {e}"))?;
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

    let sums_bytes = fetch_bytes(&format!("{base_url}/SHA256SUMS.txt"))?;
    let sig_bytes = fetch_bytes(&format!("{base_url}/SHA256SUMS.txt.sig"))?;
    let sig_b64 = String::from_utf8_lossy(&sig_bytes);
    let sig_der = base64::engine::general_purpose::STANDARD
        .decode(sig_b64.trim())
        .map_err(|e| format!("decode SHA256SUMS.txt.sig base64: {e}"))?;
    verify_ecdsa_p256_spki_pem_signature(signing_pub_pem, &sums_bytes, &sig_der)
        .map_err(|e| format!("release signature verification failed: {e}"))?;

    let sums_text = String::from_utf8_lossy(&sums_bytes);
    let mut expected_sha = None;
    for line in sums_text.lines() {
        let mut parts = line.split_whitespace();
        if let (Some(sha), Some(name)) = (parts.next(), parts.next()) {
            let clean_name = name.trim_start_matches('*');
            if clean_name == asset {
                expected_sha = Some(sha.to_ascii_lowercase());
                break;
            }
        }
    }
    let expected_sha256 = expected_sha
        .ok_or_else(|| format!("asset '{asset}' not found in verified SHA256SUMS.txt"))?;

    let exe_path = std::env::current_exe().map_err(|e| format!("resolve current_exe: {e}"))?;
    let current_sha = fs::read(&exe_path)
        .ok()
        .map(|b| sha256_lower_hex(&b))
        .unwrap_or_default();
    let up_to_date = current_sha == expected_sha256;
    let resolved_tag = target_tag.unwrap_or("latest").to_owned();

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

    let bin_bytes = fetch_bytes(&format!("{base_url}/{asset}"))?;
    let actual_sha = sha256_lower_hex(&bin_bytes);
    if actual_sha != expected_sha256 {
        return Err(format!(
            "downloaded {asset} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}"
        ));
    }

    let stage_path = exe_path.with_extension(format!("upgrade.{}", std::process::id()));
    fs::write(&stage_path, &bin_bytes)
        .map_err(|e| format!("stage {}: {e}", stage_path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&stage_path, fs::Permissions::from_mode(0o755))
            .map_err(|e| format!("chmod 0755 {}: {e}", stage_path.display()))?;
    }
    fs::rename(&stage_path, &exe_path)
        .map_err(|e| format!("atomic replace {}: {e}", exe_path.display()))?;

    // Best-effort refresh of cached `SHA256SUMS.txt`, `.sig`, and `carbon-clouds.json` in config dir.
    if let Ok(cfg_dir) = verified_absolute_jopedime_config_dir() {
        let _ = write_secret_json_0600(&cfg_dir.join("SHA256SUMS.txt"), &sums_bytes);
        let _ = write_secret_json_0600(&cfg_dir.join("SHA256SUMS.txt.sig"), &sig_bytes);
        if let Ok(clouds_bytes) = fetch_bytes(&format!("{base_url}/carbon-clouds.json")) {
            let _ = write_secret_json_0600(&cfg_dir.join("carbon-clouds.json"), &clouds_bytes);
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

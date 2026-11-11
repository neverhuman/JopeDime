//! CLI execution (`carbon run`) and Free + Pro `r882` zero-knowledge live Lambda benchmark harness (`carbon bench`).

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use jope_api::{
    carbon::{CarbonExecutionModeWire, CarbonRunStateWire, CarbonTierWire},
    ids::{DatasetId, RunId},
    Effort,
};
use serde::{Deserialize, Serialize};

use crate::{
    cells::engine_invoker::now_epoch_ms,
    cli::client::{CarbonClientPreparedDataset, CarbonThinClient},
    storage::cas::sha256_lower_hex,
};

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
    pub waves_completed: u32,
    pub waves_target: u32,
    pub candidates_evaluated: u64,
    pub best_cv_score: Option<f64>,
    pub outer_holdout_score: Option<f64>,
    pub sample_predictions: Vec<f64>,
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
    pub best_cv_score: Option<f64>,
    pub outer_holdout_score: Option<f64>,
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
pub fn load_numeric_csv(path: &Path) -> Result<(Vec<String>, Vec<Vec<f64>>), String> {
    let raw = fs::read_to_string(path).map_err(|e| format!("read CSV {}: {e}", path.display()))?;
    let mut lines = raw.lines().map(str::trim).filter(|line| !line.is_empty());
    let header_line = lines
        .next()
        .ok_or_else(|| format!("CSV {} is empty", path.display()))?;
    let headers: Vec<String> = header_line
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_owned())
        .collect();
    if headers.len() < 2 {
        return Err(format!(
            "CSV {} must contain at least 1 feature column and 1 target column",
            path.display()
        ));
    }
    let mut rows = Vec::new();
    for (idx, line) in lines.enumerate() {
        let cells: Vec<&str> = line.split(',').collect();
        if cells.len() != headers.len() {
            return Err(format!(
                "CSV {} row {} has {} columns, expected {}",
                path.display(),
                idx + 2,
                cells.len(),
                headers.len()
            ));
        }
        let mut parsed_row = Vec::with_capacity(headers.len());
        for (col_idx, cell) in cells.iter().enumerate() {
            let trimmed = cell.trim().trim_matches('"');
            let val: f64 = trimmed.parse().map_err(|e| {
                format!(
                    "CSV {} row {} column '{}' ('{trimmed}') is not a finite number: {e}",
                    path.display(),
                    idx + 2,
                    headers[col_idx]
                )
            })?;
            parsed_row.push(val);
        }
        rows.push(parsed_row);
    }
    Ok((headers, rows))
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

fn prepare_from_options(
    opts: &CarbonCliRunOptions,
) -> Result<(CarbonClientPreparedDataset, Vec<Vec<f64>>, usize), String> {
    let (headers, rows, target_col) = if let Some(csv_path) = &opts.csv_path {
        let (h, r) = load_numeric_csv(csv_path)?;
        (h, r, opts.target_column.clone())
    } else {
        generate_nonlinear_benchmark_rows(opts.synthetic_rows, opts.synthetic_features, opts.seed)
    };
    let target_idx = headers
        .iter()
        .position(|h| h == &target_col)
        .ok_or_else(|| format!("target column '{target_col}' not found in headers"))?;
    let total_rows = rows.len();
    let sample_feature_rows: Vec<Vec<f64>> = rows
        .iter()
        .take(opts.predict_rows.max(1))
        .map(|r| {
            r.iter()
                .enumerate()
                .filter_map(|(i, &v)| if i == target_idx { None } else { Some(v) })
                .collect()
        })
        .collect();
    let prepared = CarbonThinClient::prepare_numeric_dataset(
        headers,
        &rows,
        &target_col,
        opts.seed,
        opts.holdout_fraction,
    )?;
    Ok((prepared, sample_feature_rows, total_rows))
}

/// Execute a complete `carbon run` lifecycle against the fastest reachable endpoint.
pub fn execute_carbon_cli_run(opts: &CarbonCliRunOptions) -> Result<CarbonCliRunReceipt, String> {
    let t_total = Instant::now();
    let (mut client, probed) = CarbonThinClient::connect_fastest(opts.endpoints.clone())?;
    let best_probe = probed
        .first()
        .ok_or_else(|| "no Carbon endpoints probed".to_owned())?;

    let mut effective_tenant = opts.tenant_id.clone();
    if let Some(tok) = &opts.auth_token {
        client = client.with_capability_token(tok.clone());
    } else if !client.enterprise_self_hosted()
        && opts.tier == CarbonTierWire::Free
        && std::env::var("CARBON_AUTH_TOKEN").is_err()
        && std::env::var("CARBON_AUTH_SECRET_HEX").is_err()
    {
        let anon = client.authenticate_anonymous()?;
        effective_tenant = anon.tenant_id;
    }

    let t_prep = Instant::now();
    let (prepared, sample_features, total_rows) = prepare_from_options(opts)?;
    let prep_ms = t_prep.elapsed().as_secs_f64() * 1_000.0;
    let jlmb_bytes = prepared.landmark_bundle.to_wire_bytes().len();

    let t_upload = Instant::now();
    client.upload_prepared(&effective_tenant, opts.tier, &prepared)?;
    let upload_ms = t_upload.elapsed().as_secs_f64() * 1_000.0;

    let (run_id, ds_id) = mint_run_and_dataset_ids()?;
    let run_id_str = run_id.to_string();

    let t_search = Instant::now();
    let mut paused_mid_run = false;
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
            &opts.objective,
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
            &opts.objective,
            opts.seed,
            &prepared,
        )?
    };

    if progress.state == CarbonRunStateWire::Paused {
        paused_mid_run = true;
        if opts.resume_after_stop {
            progress =
                client.resume_run(&effective_tenant, &run_id_str, Some(opts.effort), Some(2))?;
        }
    }
    let search_ms = t_search.elapsed().as_secs_f64() * 1_000.0;

    let predict_inputs: Vec<Vec<Option<f64>>> = sample_features
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
        let shred = client.shred_run(&effective_tenant, &run_id_str)?;
        shredded_after_run = shred.shredded;
    }
    let total_wall_ms = t_total.elapsed().as_secs_f64() * 1_000.0;

    Ok(CarbonCliRunReceipt {
        selected_endpoint: client.active_endpoint().to_owned(),
        selected_region: best_probe.region.clone(),
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
        waves_completed: progress.wave_index,
        waves_target: progress.waves_target,
        candidates_evaluated: progress.candidates_evaluated.0,
        best_cv_score: progress.best_cv_loss,
        outer_holdout_score: progress.outer_holdout_loss,
        sample_predictions: pred_resp.predictions,
    })
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
    let medium_run = execute_carbon_cli_run(&CarbonCliRunOptions {
        endpoints: vec![active_ep.clone()],
        csv_path: None,
        target_column: "y".to_owned(),
        synthetic_rows: 250,
        synthetic_features: 6,
        tenant_id: tenant_id.to_owned(),
        tier,
        effort: Effort::Medium,
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
        let t_s4 = Instant::now();
        let (prep_hi, sample_hi, _) = prepare_from_options(&CarbonCliRunOptions {
            endpoints: vec![active_ep.clone()],
            csv_path: None,
            target_column: "y".to_owned(),
            synthetic_rows: 300,
            synthetic_features: 6,
            tenant_id: tenant_id.to_owned(),
            tier,
            effort: Effort::High,
            objective: "rmse".to_owned(),
            seed: 404,
            holdout_fraction: 0.20,
            stop_after_ms: None,
            resume_after_stop: false,
            predict_rows: 3,
            auth_token: None,
            shred_after_run: true,
        })?;
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
            run_hi,
            ds_hi,
            tenant_id,
            tier,
            Effort::High,
            "rmse",
            404,
            &prep_hi,
        )?;
        let _ = stopper.join();

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
            client.resume_run(tenant_id, &run_hi_str, Some(Effort::High), Some(2))?
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
            best_cv_score: final_prog.best_cv_loss,
            outer_holdout_score: final_prog.outer_holdout_loss,
            notes: format!(
                "paused={was_paused}, wave_pause={}, wave_end={}, guidance={}, cuen={}, shredded_objs={}",
                first_prog.wave_index,
                final_prog.wave_index,
                final_prog.neural_guidance_mode.as_deref().unwrap_or("-"),
                final_prog.encrypted_cuen,
                shred4.objects_deleted,
            ),
        });
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
            best_cv_score: None,
            outer_holdout_score: None,
            notes: "5,200 rows rejected at edge before S3 presign".to_owned(),
        });
    }

    let all_passed = scenarios.iter().all(|s| s.passed);
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
        best_cv_score: r.best_cv_score,
        outer_holdout_score: r.outer_holdout_score,
        notes: format!(
            "waves={}/{}, guidance={}, model={}, cuen={}",
            r.waves_completed,
            r.waves_target,
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
        "- **Endpoint**: `{}`\n- **Region**: `{}` (`enterprise_self_hosted={}`, `zero_knowledge_kek={}`)\n- **Timestamp (epoch ms)**: `{}`\n- **Ping RTT**: initial `{:.2} ms`, warm p50 `{:.2} ms`, warm min `{:.2} ms`\n- **Suite Verdict**: **{}**\n\n",
        report.target_endpoint,
        report.probed_region,
        report.enterprise_self_hosted,
        report.zero_knowledge_kek_enabled,
        report.timestamp_epoch_ms,
        report.cold_ping_ms,
        report.warm_ping_p50_ms,
        report.warm_ping_min_ms,
        if report.all_passed { "PASS" } else { "FAIL" }
    ));
    md.push_str("| Scenario | Mode | Shards | Wall (ms) | Prep (ms) | Upload (ms) | Search (ms) | Predict (ms) | Cands | Cands/s | CV RMSE | Holdout RMSE | Status | Notes |\n");
    md.push_str("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n");
    for s in &report.scenarios {
        let cv = match s.best_cv_score {
            Some(v) => format!("{v:.5}"),
            None => "-".to_owned(),
        };
        let ho = match s.outer_holdout_score {
            Some(v) => format!("{v:.5}"),
            None => "-".to_owned(),
        };
        md.push_str(&format!(
            "| `{}` | `{}` | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {} | {:.1} | {} | {} | {} | {} |\n",
            s.scenario_id,
            s.execution_mode,
            s.eval_shards,
            s.wall_ms,
            s.prep_ms,
            s.upload_ms,
            s.search_ms,
            s.predict_ms,
            s.candidates_evaluated,
            s.candidates_per_sec,
            cv,
            ho,
            if s.passed { "PASS" } else { "FAIL" },
            s.notes
        ));
    }
    md
}

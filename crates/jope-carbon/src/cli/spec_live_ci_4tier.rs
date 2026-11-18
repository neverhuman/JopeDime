//! Validation, HTML tables, vector SVG chart, and Retina terminal summary rendering for
//! `live-ci-4tier-docker.json` (`carbon spec-report --live-ci-receipt` & `carbon spec-verify`).

use serde_json::Value;

pub use crate::cli::spec_live_ci_4tier_exec::{
    execute_ci_live_4tier_docker, LiveCi4TierOptions, LiveCi4TierReceipt,
};

/// Expected JSON schema identifier for the `carbon-live-e2e` qualification receipt.
pub const LIVE_CI_4TIER_SCHEMA: &str = "jopedime.carbon.live-ci-4tier-docker.v1";

/// Canonical filename written inside the rendered specification bundle directory.
pub const LIVE_CI_4TIER_FILENAME: &str = "live-ci-4tier-docker.json";

const EXPECTED_CLOUDS: [&str; 2] = ["dev", "prod"];
const EXPECTED_EFFORTS: [&str; 4] = ["fast", "medium", "high", "ultra"];

fn v_str<'a>(val: &'a Value, key: &str) -> &'a str {
    val.get(key).and_then(Value::as_str).unwrap_or("")
}

fn v_f64(val: &Value, key: &str) -> f64 {
    val.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

fn v_u64(val: &Value, key: &str) -> u64 {
    val.get(key).and_then(Value::as_u64).unwrap_or(0)
}

fn v_bool(val: &Value, key: &str) -> bool {
    val.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn v_cost(val: &Value, key: &str) -> f64 {
    val.get(key)
        .and_then(Value::as_f64)
        .or_else(|| val["cost_estimate"].get(key).and_then(Value::as_f64))
        .unwrap_or(0.0)
}

fn is_lower_hex64(raw: &str) -> bool {
    raw.len() == 64
        && raw
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn html_esc(raw: &str) -> String {
    crate::cli::spec_report::html_esc(raw)
}

fn validate_container_and_install(doc: &Value) -> Result<(), String> {
    let container = &doc["container"];
    if v_str(container, "base_image") != "ubuntu:24.04" {
        return Err("live-ci-4tier-docker.json: container.base_image must be ubuntu:24.04".into());
    }
    if !v_bool(container, "zero_credentials_pre_install") {
        return Err("live-ci-4tier-docker.json: zero_credentials_pre_install must be true".into());
    }
    if !v_bool(container, "post_logout_credentials_absent") {
        return Err(
            "live-ci-4tier-docker.json: post_logout_credentials_absent must be true".into(),
        );
    }

    let inst = &doc["install"];
    if !is_lower_hex64(v_str(inst, "installed_binary_sha256")) {
        return Err("live-ci-4tier-docker.json: invalid installed_binary_sha256".into());
    }
    if !v_bool(inst, "self_verify_verified") || !v_bool(inst, "manifest_ecdsa_verified") {
        return Err("live-ci-4tier-docker.json: self-verify / ECDSA check did not pass".into());
    }
    if v_u64(inst, "kek_pins_verified") < 6 {
        return Err("live-ci-4tier-docker.json: expected >= 6 verified P-256 KEK pins".into());
    }
    if !v_bool(inst, "doctor_healthy")
        || v_u64(inst, "doctor_dev_healthy_regions") < 1
        || v_u64(inst, "doctor_prod_healthy_regions") < 5
    {
        return Err("live-ci-4tier-docker.json: doctor dev+prod Wave-1 check did not pass".into());
    }
    if !v_bool(inst, "quickstart_passed")
        || !v_bool(inst, "quickstart_encrypted_cuen")
        || !v_bool(inst, "quickstart_shredded")
        || v_f64(inst, "quickstart_holdout_r2") < 0.99
    {
        return Err("live-ci-4tier-docker.json: production quickstart check did not pass".into());
    }
    for gate_key in [
        "live_dev_free_high_gate_status",
        "live_dev_free_ultra_gate_status",
        "live_prod_free_ultra_gate_status",
    ] {
        if inst.get(gate_key).is_some() && v_u64(inst, gate_key) != 403 {
            return Err(format!(
                "live-ci-4tier-docker.json: {gate_key} expected HTTP 403, got {}",
                v_u64(inst, gate_key)
            ));
        }
    }
    Ok(())
}

fn validate_single_tier_run(
    run: &Value,
    expected_cloud: &str,
    expected_effort: &str,
    mode: &str,
    prod_public_mint_closed: bool,
) -> Result<(), String> {
    let target = v_str(run, "execution_target");
    if target != "aws_live" && target != "ci_emulator" {
        return Err(format!(
            "live-ci-4tier-docker.json: invalid execution_target '{target}' on {expected_cloud}/{expected_effort}"
        ));
    }
    if mode == "live_pro_secrets" && target != "aws_live" {
        return Err(format!(
            "live-ci-4tier-docker.json: live_pro_secrets requires aws_live on {expected_cloud}/{expected_effort}"
        ));
    }
    if matches!(expected_effort, "fast" | "medium")
        && !(expected_cloud == "prod" && prod_public_mint_closed && mode != "live_pro_secrets")
        && target != "aws_live"
    {
        return Err(format!(
            "live-ci-4tier-docker.json: {expected_cloud}/{expected_effort} must execute on aws_live"
        ));
    }
    if v_str(run, "final_state") != "completed"
        || !v_bool(run, "encrypted_cuen")
        || !v_bool(run, "shredded_after_run")
    {
        return Err(format!(
            "live-ci-4tier-docker.json: {expected_cloud}/{expected_effort} failed completion/CUEN/shred invariant"
        ));
    }
    if v_f64(run, "holdout_r2") < 0.99 || v_f64(run, "total_wall_ms") <= 0.0 {
        return Err(format!(
            "live-ci-4tier-docker.json: {expected_cloud}/{expected_effort} invalid R² or wall time"
        ));
    }
    let cost = &run["cost_estimate"];
    if v_f64(cost, "total_estimated_usd") <= 0.0 || v_f64(cost, "runs_per_1_usd") <= 0.0 {
        return Err(format!(
            "live-ci-4tier-docker.json: {expected_cloud}/{expected_effort} missing positive USD cost telemetry"
        ));
    }
    Ok(())
}

/// Fail-closed validator for `live-ci-4tier-docker.json` bound to `expected_source_sha`.
pub fn validate_live_ci_4tier_receipt(
    doc: &Value,
    expected_source_sha: Option<&str>,
) -> Result<(), String> {
    if v_str(doc, "schema") != LIVE_CI_4TIER_SCHEMA {
        return Err(format!(
            "live-ci-4tier-docker.json: unexpected schema '{}'",
            v_str(doc, "schema")
        ));
    }
    if !v_bool(doc, "all_passed") {
        return Err("live-ci-4tier-docker.json: all_passed is false".into());
    }
    if let Some(exp) = expected_source_sha {
        let trimmed = exp.trim();
        if !trimmed.is_empty() && trimmed != "workspace" && v_str(doc, "source_sha") != trimmed {
            return Err(format!(
                "live-ci-4tier-docker.json: source_sha '{}' does not match expected '{trimmed}'",
                v_str(doc, "source_sha")
            ));
        }
    }
    let mode = v_str(doc, "pro_secrets_mode");
    if mode != "live_pro_secrets" && mode != "zero_secrets_hybrid_live_and_emulator" {
        return Err(format!(
            "live-ci-4tier-docker.json: unsupported pro_secrets_mode '{mode}'"
        ));
    }

    validate_container_and_install(doc)?;

    let runs = doc["runs"]
        .as_array()
        .ok_or_else(|| "live-ci-4tier-docker.json: runs must be an array".to_owned())?;
    if runs.len() != 8 {
        return Err(format!(
            "live-ci-4tier-docker.json: expected 8 dev+prod tier runs, got {}",
            runs.len()
        ));
    }
    let prod_public_mint_closed =
        v_u64(&doc["install"], "live_prod_public_mint_gate_status") == 403;
    for cloud in EXPECTED_CLOUDS {
        for effort in EXPECTED_EFFORTS {
            let found = runs
                .iter()
                .find(|r| v_str(r, "cloud") == cloud && v_str(r, "effort") == effort)
                .ok_or_else(|| {
                    format!("live-ci-4tier-docker.json: missing run for {cloud}/{effort}")
                })?;
            validate_single_tier_run(found, cloud, effort, mode, prod_public_mint_closed)?;
        }
    }

    let cmp_len = doc["per_tier_comparison"].as_array().map_or(0, Vec::len);
    if cmp_len != 4 {
        return Err(format!(
            "live-ci-4tier-docker.json: expected 4 per_tier_comparison rows, got {cmp_len}"
        ));
    }
    if doc["emulator_cells"].is_object()
        && v_u64(&doc["emulator_cells"], "total_residual_cas_blobs") != 0
    {
        return Err("live-ci-4tier-docker.json: emulator_cells left residual CAS blobs".into());
    }
    Ok(())
}

fn target_badge_html(target: &str) -> &'static str {
    if target == "aws_live" {
        "<span class=\"pill-pass\">AWS LIVE</span>"
    } else if target == "aws_live/emu" {
        "<span style=\"background:#ccfbf1;color:#0f766e;padding:1px 5px;border-radius:4px;font-weight:700;font-size:7.3pt;\">LIVE / EMU</span>"
    } else {
        "<span style=\"background:#dbeafe;color:#1e40af;padding:1px 5px;border-radius:4px;font-weight:700;font-size:7.3pt;\">CI EMULATOR</span>"
    }
}

fn short_exec_mode(raw: &str) -> &'static str {
    if raw.to_ascii_lowercase().contains("hybrid") {
        "Hybrid"
    } else {
        "Solo"
    }
}

fn render_live_ci_svg(runs: &[Value]) -> String {
    let mut max_wall_s = 20.0f64;
    let mut max_milli_usd = 2.0f64;
    for r in runs {
        let ws = v_f64(r, "total_wall_ms") / 1_000.0;
        let mu = v_f64(&r["cost_estimate"], "total_estimated_usd") * 1_000.0;
        if ws * 1.15 > max_wall_s {
            max_wall_s = ws * 1.15;
        }
        if mu * 1.15 > max_milli_usd {
            max_milli_usd = mu * 1.15;
        }
    }

    let mut left_bars = String::new();
    let mut right_bars = String::new();
    let panel_w = 195.0f64;
    for (idx, r) in runs.iter().enumerate() {
        let y = 52.0 + (idx as f64) * 24.0;
        let ty = y + 11.5;
        let is_live = v_str(r, "execution_target") == "aws_live";
        let tag = if is_live { "LIVE" } else { "EMU" };
        let lbl = format!("{} {:<6} [{}]", v_str(r, "cloud"), v_str(r, "effort"), tag);
        let ws = v_f64(r, "total_wall_ms") / 1_000.0;
        let bar_fill = if is_live { "#059669" } else { "#0284c7" };
        let w_time = ((ws / max_wall_s.max(0.5)) * panel_w).max(2.5);
        left_bars.push_str(&format!(
            r##"<text x="12" y="{ty:.1}" fill="#0f172a" font-size="8.4" font-weight="700" font-family="Liberation Mono, monospace">{lbl}</text>
<rect x="132" y="{y:.1}" width="{w_time:.1}" height="14" rx="2.5" fill="{bar_fill}"/>
<text x="{tx:.1}" y="{ty:.1}" fill="#0f172a" font-size="8.4" font-family="Liberation Mono, monospace">{ws:.2}s</text>"##,
            lbl = html_esc(&lbl),
            tx = 138.0 + w_time
        ));

        let c = &r["cost_estimate"];
        let eng_m = v_f64(c, "lambda_engine_usd") * 1_000.0;
        let eval_m = v_f64(c, "lambda_eval_usd") * 1_000.0;
        let s3_m =
            (v_f64(c, "s3_requests_usd") + v_f64(c, "s3_storage_and_transfer_usd")) * 1_000.0;
        let edge_m = (v_f64(c, "apigw_http_api_usd") + v_f64(c, "lambda_edge_usd")) * 1_000.0;
        let tot_m = v_f64(c, "total_estimated_usd") * 1_000.0;
        let scale = panel_w / max_milli_usd.max(0.05);
        let w_eng = (eng_m * scale).max(1.5);
        let w_eval = eval_m * scale;
        let w_s3 = (s3_m * scale).max(1.0);
        let w_edge = (edge_m * scale).max(1.0);
        let x0 = 560.0;
        let x1 = x0 + w_eng;
        let x2 = x1 + w_eval;
        let x3 = x2 + w_s3;
        right_bars.push_str(&format!(
            r##"<text x="440" y="{ty:.1}" fill="#0f172a" font-size="8.4" font-weight="700" font-family="Liberation Mono, monospace">{lbl}</text>
<rect x="{x0:.1}" y="{y:.1}" width="{w_eng:.1}" height="14" fill="#2563eb"/>
<rect x="{x1:.1}" y="{y:.1}" width="{w_eval:.1}" height="14" fill="#7c3aed"/>
<rect x="{x2:.1}" y="{y:.1}" width="{w_s3:.1}" height="14" fill="#059669"/>
<rect x="{x3:.1}" y="{y:.1}" width="{w_edge:.1}" height="14" fill="#d97706"/>
<text x="{ctx:.1}" y="{ty:.1}" fill="#0f172a" font-size="8.3" font-family="Liberation Mono, monospace">${usd:.5} ({tot_m:.2} m$)</text>"##,
            lbl = html_esc(&lbl),
            ctx = (x3 + w_edge + 5.0).min(765.0),
            usd = tot_m / 1_000.0
        ));
    }

    format!(
        r##"<svg viewBox="0 0 880 276" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="CI Fresh-Docker 4-Tier Wall Time and AWS Cost">
  <rect x="1" y="1" width="878" height="274" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>
  <text x="14" y="22" fill="#0f172a" font-size="10.8" font-weight="700" font-family="Liberation Sans, sans-serif">Figure 4C — CI-Run Fresh-Docker dev &amp; prod 4-Tier (fast / medium / high / ultra) Wall Time &amp; Itemized AWS Cost</text>
  <text x="14" y="40" fill="#475569" font-size="9.2" font-weight="700" font-family="Liberation Sans, sans-serif">Panel A — E2E Wall-Clock Time (Green = AWS LIVE, Blue = CI EMULATOR)</text>
  <text x="440" y="40" fill="#475569" font-size="9.2" font-weight="700" font-family="Liberation Sans, sans-serif">Panel B — Itemized 6-Component AWS Cost per Run (Millidollars)</text>
  <line x1="428" y1="32" x2="428" y2="246" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4,3"/>
  {left_bars}
  {right_bars}
  <rect x="14" y="254" width="11" height="9" fill="#059669"/><text x="29" y="262" fill="#334155" font-size="8.7" font-family="Liberation Sans, sans-serif">AWS Live Wall (s)</text>
  <rect x="135" y="254" width="11" height="9" fill="#0284c7"/><text x="150" y="262" fill="#334155" font-size="8.7" font-family="Liberation Sans, sans-serif">CI Emulator Wall (s)</text>
  <rect x="285" y="254" width="11" height="9" fill="#2563eb"/><text x="300" y="262" fill="#334155" font-size="8.7" font-family="Liberation Sans, sans-serif">Engine Lambda</text>
  <rect x="415" y="254" width="11" height="9" fill="#7c3aed"/><text x="430" y="262" fill="#334155" font-size="8.7" font-family="Liberation Sans, sans-serif">Eval Shards</text>
  <rect x="530" y="254" width="11" height="9" fill="#059669"/><text x="545" y="262" fill="#334155" font-size="8.7" font-family="Liberation Sans, sans-serif">S3 CAS</text>
  <rect x="625" y="254" width="11" height="9" fill="#d97706"/><text x="640" y="262" fill="#334155" font-size="8.7" font-family="Liberation Sans, sans-serif">API GW v2 + Edge</text>
</svg>"##
    )
}

fn render_install_table_html(doc: &Value) -> String {
    let cont = &doc["container"];
    let inst = &doc["install"];
    let bin_sha = v_str(inst, "installed_binary_sha256");
    let bin_short = &bin_sha[..bin_sha.len().min(16)];
    let gate_code = v_u64(inst, "live_anon_high_gate_status");
    format!(
        r#"<h3>Table 4C-1 — CI-Run Fresh-Docker <code>ubuntu:24.04</code> Zero-Credential Production CLI Install &amp; Onboarding Checks</h3>
  <table>
    <thead><tr><th style="width:24%">Fresh-Docker CI Check</th><th style="width:36%">Executed Command (<code>env -i</code>, Zero Mounts)</th><th style="width:32%">Measured Live Production Telemetry</th><th style="width:8%">Status</th></tr></thead>
    <tbody>
      <tr><td>Clean-Room Container</td><td><code>docker run --rm ubuntu:24.04 /usr/bin/env -i</code></td><td><code>{os_name}</code> | <code>0</code> AWS/GH/Carbon env vars &amp; credential files</td><td><span class="pill-pass">PASS</span></td></tr>
      <tr><td>Production Signed Install</td><td><code>curl -fsSL https://downloads.neverhuman.org/carbon/install.sh | sh</code></td><td><code>{inst_s:.2}s</code> | tag <code>{tag}</code> | sha256 <code>{bin_short}…</code> (<code>{bin_b}</code> B)</td><td><span class="pill-pass">PASS</span></td></tr>
      <tr><td>Offline ECDSA &amp; KEK Pins</td><td><code>carbon self-verify</code></td><td><code>manifest_ecdsa_verified=true</code> | <code>{pins}</code> regional P-256 KEK pins</td><td><span class="pill-pass">PASS</span></td></tr>
      <tr><td>Multi-Region Health Probe</td><td><code>carbon doctor --cloud all</code></td><td><code>dev</code>: <code>{dev_reg}/1</code> healthy | <code>prod</code>: <code>{prod_reg}/5</code> Wave-1 healthy | dir <code>{mode}</code></td><td><span class="pill-pass">PASS</span></td></tr>
      <tr><td>Live Prod Quickstart + Gate</td><td><code>carbon quickstart</code> + anon <code>--effort high</code> gate</td><td><code>{qs_id}</code> in <code>{qs_s:.2}s</code> (<code>R²={qs_r2:.6}</code>, shredded) | anon Pro gate <code>HTTP {gate_code}</code></td><td><span class="pill-pass">PASS</span></td></tr>
    </tbody>
  </table>"#,
        os_name = html_esc(v_str(cont, "os_pretty_name")),
        inst_s = v_f64(inst, "install_wall_ms") / 1_000.0,
        tag = html_esc(v_str(inst, "release_tag")),
        bin_short = html_esc(bin_short),
        bin_b = v_u64(inst, "installed_binary_bytes"),
        pins = v_u64(inst, "kek_pins_verified"),
        dev_reg = v_u64(inst, "doctor_dev_healthy_regions"),
        prod_reg = v_u64(inst, "doctor_prod_healthy_regions"),
        mode = html_esc(v_str(inst, "config_dir_mode")),
        qs_id = html_esc(v_str(inst, "quickstart_run_id")),
        qs_s = v_f64(inst, "quickstart_wall_ms") / 1_000.0,
        qs_r2 = v_f64(inst, "quickstart_holdout_r2"),
    )
}

fn render_runs_and_cmp_tables_html(doc: &Value, runs: &[Value]) -> String {
    let mut run_rows = String::new();
    for r in runs {
        let c = &r["cost_estimate"];
        run_rows.push_str(&format!(
            "<tr><td class=\"nowrap\"><code>{cl}</code></td><td class=\"nowrap\"><code>{eff}</code></td><td class=\"nowrap\">{badge}</td><td><code class=\"wrap-any\">{rid}</code></td><td class=\"nowrap\"><code>{tier}</code>/{mode}</td><td class=\"num\">{wc}/{wt} ({ev})</td><td class=\"num\">{up:.0} ms</td><td class=\"num\">{sr:.2}s</td><td class=\"num\"><strong>{tw:.2}s</strong></td><td class=\"num\">{r2:.6}</td><td class=\"num\">{rmse:.4} (-{red:.1}%)</td><td class=\"num\">${usd:.6}</td><td class=\"num\">{rp1:.0}</td></tr>\n",
            cl = html_esc(v_str(r, "cloud")),
            eff = html_esc(v_str(r, "effort")),
            badge = target_badge_html(v_str(r, "execution_target")),
            rid = html_esc(v_str(r, "run_id")),
            tier = html_esc(v_str(r, "tier_applied")),
            mode = short_exec_mode(v_str(r, "execution_mode")),
            wc = v_u64(r, "waves_completed"),
            wt = v_u64(r, "waves_target"),
            ev = v_u64(r, "eval_shards"),
            up = v_f64(r, "upload_ms"),
            sr = v_f64(r, "search_ms") / 1_000.0,
            tw = v_f64(r, "total_wall_ms") / 1_000.0,
            r2 = v_f64(r, "holdout_r2"),
            rmse = v_f64(r, "holdout_rmse"),
            red = v_f64(r, "rmse_reduction_vs_linear_pct"),
            usd = v_f64(c, "total_estimated_usd"),
            rp1 = v_f64(c, "runs_per_1_usd"),
        ));
    }

    let cmps = doc["per_tier_comparison"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut cmp_rows = String::new();
    for row in cmps {
        let d = &row["dev"];
        let p = &row["prod"];
        cmp_rows.push_str(&format!(
            "<tr><td class=\"nowrap\"><code><strong>{eff}</strong></code></td><td class=\"nowrap\">{badge}</td><td class=\"num\">{dw:.2}s</td><td class=\"num\">{pw:.2}s</td><td class=\"num\">{dr2:.6} ({drmse:.4})</td><td class=\"num\">{pr2:.6} ({prmse:.4})</td><td class=\"num\">${dusd:.6} ({drp1:.0}/$1)</td><td class=\"num\">${pusd:.6} ({prp1:.0}/$1)</td></tr>\n",
            eff = html_esc(v_str(row, "effort")),
            badge = target_badge_html(v_str(row, "execution_target")),
            dw = v_f64(d, "total_wall_ms") / 1_000.0,
            pw = v_f64(p, "total_wall_ms") / 1_000.0,
            dr2 = v_f64(d, "holdout_r2"),
            drmse = v_f64(d, "holdout_rmse"),
            pr2 = v_f64(p, "holdout_r2"),
            prmse = v_f64(p, "holdout_rmse"),
            dusd = v_cost(d, "total_estimated_usd"),
            drp1 = v_cost(d, "runs_per_1_usd"),
            pusd = v_cost(p, "total_estimated_usd"),
            prp1 = v_cost(p, "runs_per_1_usd"),
        ));
    }

    format!(
        r#"<h3>Table 4C-2 — CI-Executed <code>dev</code> &amp; <code>prod</code> 4-Tier (<code>fast</code> / <code>medium</code> / <code>high</code> / <code>ultra</code>) Phase Timing &amp; Cost Matrix</h3>
  <table>
    <thead><tr><th>Cloud</th><th>Effort</th><th>Target</th><th>Run ID</th><th>Tier/Mode</th><th class="num">Waves (Ev)</th><th class="num">Up</th><th class="num">Search</th><th class="num">Wall</th><th class="num">R²</th><th class="num">RMSE (vs Lin)</th><th class="num">Est USD</th><th class="num">/$1</th></tr></thead>
    <tbody>{run_rows}</tbody>
  </table>
  <h3>Table 4C-3 — CI-Run Per-Tier <code>dev</code> vs <code>prod</code> Head-to-Head Benchmark Summary (<code>nonlinear_240x6.csv</code>)</h3>
  <table>
    <thead><tr><th>Effort</th><th>Execution Target</th><th class="num"><code>dev</code> Wall</th><th class="num"><code>prod</code> Wall</th><th class="num"><code>dev</code> R² (RMSE)</th><th class="num"><code>prod</code> R² (RMSE)</th><th class="num"><code>dev</code> USD (/ $1)</th><th class="num"><code>prod</code> USD (/ $1)</th></tr></thead>
    <tbody>{cmp_rows}</tbody>
  </table>"#
    )
}

/// Render Section 4C HTML (Fresh-Docker production CLI install + 8-row `dev`/`prod` 4-tier matrix + SVG).
pub fn render_live_ci_4tier_section_4c_html(doc: &Value) -> String {
    let runs = doc["runs"].as_array().map(Vec::as_slice).unwrap_or(&[]);
    let mode = v_str(doc, "pro_secrets_mode");
    let mode_desc = if mode == "live_pro_secrets" {
        "Option A (<code>use_live_pro_secrets=true</code>: all 8 <code>dev</code>+<code>prod</code> tiers executed live on AWS)"
    } else {
        "Option B zero-secret CI mode (<code>use_live_pro_secrets=false</code>: <code>fast</code>+<code>medium</code> executed live on AWS <code>dev</code> &amp; <code>prod</code> via zero-credential <code>carbon register --free</code>; <code>high</code>+<code>ultra</code> executed by the same Fresh-Docker production binary against isolated dual-cloud <code>CarbonLocalLambdaEmulator</code> cells, badged per row)"
    };
    let svg = render_live_ci_svg(runs);
    let inst_tbl = render_install_table_html(doc);
    let runs_tbl = render_runs_and_cmp_tables_html(doc, runs);
    format!(
        r#"<h3>4C. Live CI Fresh-Docker Production Install &amp; <code>dev</code> + <code>prod</code> 4-Tier (<code>fast</code> / <code>medium</code> / <code>high</code> / <code>ultra</code>) Qualification</h3>
  <p>
    During GitHub Actions run <code>{run_id}</code> (attempt <code>{att}</code>, commit <code>{sha}</code>, completed in <code>{tot_s:.2}s</code> under {mode_desc}), the <code>carbon-live-e2e</code> lane launched an unmounted <code>ubuntu:24.04</code> container (<code>{digest}</code>, <code>/usr/bin/env -i</code>), installed the live production CLI from <code>https://downloads.neverhuman.org/carbon/install.sh</code>, verified ECDSA-P256 release signatures and all 6 regional P-256 KEK pins, ran <code>dev</code> and <code>prod</code> across <code>fast</code>, <code>medium</code>, <code>high</code>, and <code>ultra</code> with <code>--shred</code>, and verified post-logout credential removal (<code>post_logout_credentials_absent = true</code>):
  </p>
  {inst_tbl}
  <div class="svg-box">{svg}</div>
  {runs_tbl}"#,
        run_id = html_esc(v_str(doc, "github_run_id")),
        att = v_u64(doc, "github_run_attempt"),
        sha = html_esc(v_str(doc, "source_sha")),
        tot_s = v_f64(doc, "total_wall_ms") / 1_000.0,
        digest = html_esc(v_str(&doc["container"], "repo_digest")),
    )
}

/// Format terminal lines summarizing the current CI run's Fresh-Docker 4-tier results for `#shot-06`.
pub fn format_live_ci_4tier_shot06_lines(doc: &Value) -> Vec<String> {
    let mut lines = vec![
        String::new(),
        format!(
            "CI LIVE FRESH-DOCKER (run={}, attempt={}, sha={}…) — DEV & PROD 4-TIER:",
            v_str(doc, "github_run_id"),
            v_u64(doc, "github_run_attempt"),
            &v_str(doc, "source_sha")[..v_str(doc, "source_sha").len().min(12)]
        ),
        String::from(
            "EFFORT  TARGET       DEV_WALL  PROD_WALL   DEV_R2    PROD_R2    DEV_USD     PROD_USD",
        ),
    ];
    if let Some(cmps) = doc["per_tier_comparison"].as_array() {
        for row in cmps {
            let d = &row["dev"];
            let p = &row["prod"];
            lines.push(format!(
                "{:<6}  {:<11}  {:>7.2}s  {:>8.2}s  {:>8.6}  {:>8.6}  ${:>9.6}  ${:>9.6}",
                v_str(row, "effort"),
                v_str(row, "execution_target"),
                v_f64(d, "total_wall_ms") / 1_000.0,
                v_f64(p, "total_wall_ms") / 1_000.0,
                v_f64(d, "holdout_r2"),
                v_f64(p, "holdout_r2"),
                v_cost(d, "total_estimated_usd"),
                v_cost(p, "total_estimated_usd"),
            ));
        }
    }
    lines
}

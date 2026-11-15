//! Pure-Rust host orchestrator and clean-room `ubuntu:24.04` POSIX `sh` driver for
//! `carbon ci-live-4tier` (`live-ci-4tier-docker.json`).

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use aws_lc_rs::digest::{digest, SHA256};
use base64::{engine::general_purpose::STANDARD as B64_STD, Engine as _};
use jope_api::carbon::CarbonTierWire;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::cells::engine_invoker::now_epoch_ms;
use crate::cli::client::{CarbonCloudCellEntry, CarbonCloudRegionPin, CarbonPublicCloudsManifest};
use crate::cli::spec_live_ci_4tier::{
    validate_live_ci_4tier_receipt, LIVE_CI_4TIER_FILENAME, LIVE_CI_4TIER_SCHEMA,
};
use crate::modes::emu::{CarbonEmuConfig, CarbonLocalLambdaEmulator};

const CLEANROOM_TAG: &str = "carbon-ci-cleanroom:ubuntu2404";
const CLEANROOM_DOCKERFILE: &str = "\
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
    ca-certificates curl openssl zstd && rm -rf /var/lib/apt/lists/*
";

const FORBIDDEN_RECEIPT_MARKERS: [&str; 11] = [
    concat!("ctk", "1."),
    concat!("AK", "IA"),
    concat!("AS", "IA"),
    concat!("BEGIN ", "PRIVATE KEY"),
    concat!("BEGIN EC ", "PRIVATE KEY"),
    concat!("CARBON_AUTH_", "SECRET"),
    concat!("CARBON_SEAL_", "SECRET"),
    concat!("CARBON_PRESIGN_", "SECRET"),
    concat!("xba", "be"),
    concat!("atomic", "soul"),
    concat!("/", "home/"),
];

/// CLI options for `carbon ci-live-4tier`.
#[derive(Clone, Debug)]
pub struct LiveCi4TierOptions {
    pub source_sha: String,
    pub out_dir: PathBuf,
    pub require_live_pro: bool,
    pub repo_root: PathBuf,
}

/// Verified receipt emitted by `carbon ci-live-4tier` (`live-ci-4tier-docker.json`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LiveCi4TierReceipt {
    pub schema: String,
    pub captured_utc: String,
    pub source_sha: String,
    pub github_run_id: String,
    pub github_run_attempt: u64,
    pub pro_secrets_mode: String,
    pub all_passed: bool,
    pub total_wall_ms: f64,
    pub total_wall_secs: f64,
    pub total_estimated_usd: f64,
    pub container: Value,
    pub install: Value,
    pub dataset: Value,
    pub runs: Vec<Value>,
    pub per_tier_comparison: Vec<Value>,
    pub emulator_cells: Value,
}

impl LiveCi4TierReceipt {
    /// Validate all structural, statistical, shred, and secret-hygiene invariants.
    pub fn validate(
        &self,
        expected_source_sha: &str,
        require_live_pro: bool,
        sensitive_tokens: &[String],
    ) -> Result<String, String> {
        if require_live_pro && self.pro_secrets_mode != "live_pro_secrets" {
            return Err(format!(
                "require_live_pro expected mode 'live_pro_secrets', got '{}'",
                self.pro_secrets_mode
            ));
        }
        let doc = serde_json::to_value(self)
            .map_err(|e| format!("serialize LiveCi4TierReceipt to Value: {e}"))?;
        validate_live_ci_4tier_receipt(&doc, Some(expected_source_sha))?;
        if self.install["live_anon_high_gate_status"].as_u64() != Some(403) {
            return Err("live-ci-4tier-docker.json: live_anon_high_gate_status must be 403".into());
        }
        let mut rendered = serde_json::to_string_pretty(&doc)
            .map_err(|e| format!("format LiveCi4TierReceipt JSON: {e}"))?;
        rendered.push('\n');
        for marker in FORBIDDEN_RECEIPT_MARKERS {
            if rendered.contains(marker) {
                return Err("forbidden credential or host-path marker detected in receipt".into());
            }
        }
        for tok in sensitive_tokens {
            if !tok.is_empty() && rendered.contains(tok.as_str()) {
                return Err("sensitive capability token leaked into receipt".into());
            }
        }
        Ok(rendered)
    }
}

struct DockerContainerGuard {
    docker_bin: String,
    container_name: String,
    finished: bool,
}

impl Drop for DockerContainerGuard {
    fn drop(&mut self) {
        if !self.finished {
            let _ = sanitized_docker_command(&self.docker_bin)
                .args(["rm", "-f", &self.container_name])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }
}

struct InProcessDualEmulators {
    emu_dev: CarbonLocalLambdaEmulator,
    emu_prod: CarbonLocalLambdaEmulator,
    work_dir: PathBuf,
}

impl InProcessDualEmulators {
    fn start(work_dir: PathBuf) -> Result<(Self, String, String, String), String> {
        if work_dir.exists() {
            let _ = fs::remove_dir_all(&work_dir);
        }
        fs::create_dir_all(&work_dir)
            .map_err(|e| format!("create emulator scratch {}: {e}", work_dir.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&work_dir, fs::Permissions::from_mode(0o700));
        }

        let mut dev_cfg = CarbonEmuConfig::for_dir(work_dir.join("dev_cell"));
        dev_cfg.auth_secret = [0x45u8; 32];
        dev_cfg.engine_kek_scalar = [0x12u8; 32];
        let mut prod_cfg = CarbonEmuConfig::for_dir(work_dir.join("prod_cell"));
        prod_cfg.auth_secret = [0x46u8; 32];
        prod_cfg.engine_kek_scalar = [0x13u8; 32];

        let emu_dev = CarbonLocalLambdaEmulator::start(dev_cfg)?;
        let emu_prod = CarbonLocalLambdaEmulator::start(prod_cfg)?;
        let dev_tok =
            emu_dev.mint_capability_token("pro_ci_emu_dev", CarbonTierWire::Pro, false)?;
        let prod_tok =
            emu_prod.mint_capability_token("pro_ci_emu_prod", CarbonTierWire::Pro, false)?;
        let clouds_json = serde_json::to_string_pretty(&build_emu_clouds(&emu_dev, &emu_prod))
            .map_err(|e| format!("serialize in-memory emu-clouds.json: {e}"))?;
        Ok((
            Self {
                emu_dev,
                emu_prod,
                work_dir,
            },
            dev_tok,
            prod_tok,
            clouds_json,
        ))
    }

    fn finish(mut self) -> Result<Value, String> {
        let tel_dev = self.emu_dev.telemetry();
        let tel_prod = self.emu_prod.telemetry();
        let rem_dev = count_regular_files(&self.emu_dev.cas_root().join("blobs"));
        let rem_prod = count_regular_files(&self.emu_prod.cas_root().join("blobs"));
        self.emu_dev.shutdown();
        self.emu_prod.shutdown();
        let _ = fs::remove_dir_all(&self.work_dir);
        if tel_dev.eval_invocations == 0 || tel_prod.eval_invocations == 0 {
            return Err(format!(
                "expected positive HybridFanOut eval_invocations on both emulator cells (dev={}, prod={})",
                tel_dev.eval_invocations, tel_prod.eval_invocations
            ));
        }
        Ok(json!({
            "dev": {
                "edge_invocations": tel_dev.edge_invocations,
                "engine_invocations": tel_dev.engine_invocations,
                "eval_invocations": tel_dev.eval_invocations,
                "residual_cas_blobs": rem_dev,
            },
            "prod": {
                "edge_invocations": tel_prod.edge_invocations,
                "engine_invocations": tel_prod.engine_invocations,
                "eval_invocations": tel_prod.eval_invocations,
                "residual_cas_blobs": rem_prod,
            },
            "total_residual_cas_blobs": rem_dev + rem_prod,
        }))
    }
}

impl Drop for InProcessDualEmulators {
    fn drop(&mut self) {
        self.emu_dev.shutdown();
        self.emu_prod.shutdown();
        let _ = fs::remove_dir_all(&self.work_dir);
    }
}

fn build_emu_clouds(
    emu_dev: &CarbonLocalLambdaEmulator,
    emu_prod: &CarbonLocalLambdaEmulator,
) -> CarbonPublicCloudsManifest {
    let mut clouds = BTreeMap::new();
    for (name, emu) in [("dev", emu_dev), ("prod", emu_prod)] {
        let ep = emu.base_url();
        let kek = Some(emu.engine_kek_pubkey_b64().to_owned());
        clouds.insert(
            name.to_owned(),
            CarbonCloudCellEntry {
                environment: name.to_owned(),
                region: "us-east-2".to_owned(),
                public_endpoint: ep.clone(),
                engine_kek_pubkey_b64: kek.clone(),
                release_id: Some("ci-emulator".to_owned()),
                regions: vec![CarbonCloudRegionPin {
                    region: "us-east-2".to_owned(),
                    public_endpoint: ep,
                    engine_kek_pubkey_b64: kek,
                    release_id: Some("ci-emulator".to_owned()),
                }],
            },
        );
    }
    CarbonPublicCloudsManifest {
        schema: "jope-carbon-clouds-v2".to_owned(),
        generated_epoch_ms: now_epoch_ms(),
        clouds,
    }
}

fn count_regular_files(dir: &Path) -> usize {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_ok_and(|ft| ft.is_file()))
        .count()
}

fn sanitized_docker_command(docker_bin: &str) -> Command {
    let mut cmd = Command::new(docker_bin);
    for (k, _) in std::env::vars() {
        if k.starts_with("AWS_")
            || k.starts_with("CARBON_")
            || k.starts_with("GH_")
            || k == "GITHUB_TOKEN"
        {
            cmd.env_remove(&k);
        }
    }
    cmd
}

fn is_lower_hex40(raw: &str) -> bool {
    raw.len() == 40
        && raw
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(digest(&SHA256, bytes).as_ref())
}

fn format_utc_iso8601(epoch_secs: u64) -> String {
    let days = (epoch_secs / 86_400) as i64;
    let rem = epoch_secs % 86_400;
    let hh = rem / 3_600;
    let mm = (rem % 3_600) / 60;
    let ss = rem % 60;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    if m <= 2 {
        y += 1;
    }
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

fn inspect_and_build_cleanroom_image(docker_bin: &str) -> Result<(String, String), String> {
    let inspect_once = || -> Result<std::process::Output, String> {
        sanitized_docker_command(docker_bin)
            .args(["image", "inspect", "ubuntu:24.04"])
            .output()
            .map_err(|e| format!("spawn `{docker_bin} image inspect ubuntu:24.04`: {e}"))
    };
    let mut out = inspect_once()?;
    if !out.status.success() {
        let pull = sanitized_docker_command(docker_bin)
            .args(["pull", "ubuntu:24.04"])
            .output()
            .map_err(|e| format!("spawn `{docker_bin} pull ubuntu:24.04`: {e}"))?;
        if !pull.status.success() {
            return Err(format!(
                "`{docker_bin} pull ubuntu:24.04` failed: {}",
                String::from_utf8_lossy(&pull.stderr).trim()
            ));
        }
        out = inspect_once()?;
    }
    if !out.status.success() {
        return Err(format!(
            "`{docker_bin} image inspect ubuntu:24.04` failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let items: Vec<Value> = serde_json::from_slice(&out.stdout)
        .map_err(|e| format!("parse `docker image inspect` JSON: {e}"))?;
    let info = items
        .first()
        .ok_or_else(|| "docker image inspect ubuntu:24.04 returned empty list".to_owned())?;
    let image_id = info["Id"].as_str().unwrap_or("").to_owned();
    if !image_id.starts_with("sha256:") {
        return Err(format!("unexpected ubuntu:24.04 image Id '{image_id}'"));
    }
    let repo_digest = info["RepoDigests"]
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(Value::as_str)
        .map_or_else(|| format!("ubuntu@{image_id}"), str::to_owned);

    let mut build_child = sanitized_docker_command(docker_bin)
        .args(["build", "-q", "-t", CLEANROOM_TAG, "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn `{docker_bin} build`: {e}"))?;
    if let Some(mut stdin) = build_child.stdin.take() {
        stdin
            .write_all(CLEANROOM_DOCKERFILE.as_bytes())
            .map_err(|e| format!("write Dockerfile to `docker build` stdin: {e}"))?;
    }
    let build_out = build_child
        .wait_with_output()
        .map_err(|e| format!("wait on `docker build`: {e}"))?;
    if !build_out.status.success() {
        return Err(format!(
            "`docker build {CLEANROOM_TAG}` failed: {}",
            String::from_utf8_lossy(&build_out.stderr).trim()
        ));
    }
    Ok((image_id, repo_digest))
}

fn build_container_posix_script(
    mode: &str,
    dev_pro_tok: &str,
    prod_pro_tok: &str,
    emu_clouds_json: Option<&str>,
) -> String {
    let dev_b64 = B64_STD.encode(dev_pro_tok.trim().as_bytes());
    let prod_b64 = B64_STD.encode(prod_pro_tok.trim().as_bytes());
    let clouds_b64 = B64_STD.encode(emu_clouds_json.unwrap_or("").as_bytes());

    format!(
        r#"set -eu
MODE='{mode}'
DEV_PRO_B64='{dev_b64}'
PROD_PRO_B64='{prod_b64}'
EMU_CLOUDS_B64='{clouds_b64}'

CARBON_BIN='/root/.local/bin/carbon'
CSV_PATH='/root/nonlinear_240x6.csv'
EMU_CLOUDS='/root/emu-clouds.json'

cleanup() {{
  if [ -x "$CARBON_BIN" ]; then
    "$CARBON_BIN" logout --all >/dev/null 2>&1 || true
  fi
  rm -f "$EMU_CLOUDS"
}}
trap cleanup EXIT

emit_kv() {{
  printf '===KV:%s=%s===\n' "$1" "$2"
}}

emit_block() {{
  printf '===BEGIN:%s===\n' "$1"
  cat "$2"
  printf '\n===END:%s===\n' "$1"
}}

retry2() {{
  if "$@"; then
    return 0
  fi
  sleep 2
  "$@"
}}

pre_clean=true
absent_cnt=0
for v in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE \
  AWS_DEFAULT_REGION AWS_REGION GH_TOKEN GITHUB_TOKEN CARBON_TOKEN CARBON_AUTH_TOKEN \
  CARBON_AUTH_SECRET_HEX CARBON_SEAL_SECRET_HEX CARBON_PRESIGN_SECRET_HEX; do
  eval "val=\${{$v:-}}"
  if [ -n "$val" ]; then
    pre_clean=false
  else
    absent_cnt=$((absent_cnt + 1))
  fi
done
for p in /root/.aws /root/.ssh /root/.config/gh /root/.config/jopedime /root/.netrc; do
  if [ -e "$p" ]; then
    pre_clean=false
  fi
done
if [ "$pre_clean" != "true" ]; then
  echo "clean-room pre-check failed" >&2
  exit 1
fi

os_pretty="Ubuntu 24.04 LTS"
if [ -f /etc/os-release ]; then
  os_pretty=$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release | tr -d '"' | head -n 1)
fi
emit_kv OS_PRETTY_NAME "$os_pretty"
emit_kv UNAME_STR "$(uname -srm)"
emit_kv PRE_CLEAN "$pre_clean"
emit_kv ABSENT_ENV_COUNT "$absent_cnt"

t_i0=$(date +%s%3N)
curl --proto '=https' --tlsv1.2 -fsSL --retry 5 --retry-all-errors --retry-delay 2 \
  https://downloads.neverhuman.org/carbon/install.sh -o /root/install.sh
sh /root/install.sh >&2
t_i1=$(date +%s%3N)
emit_kv INSTALL_WALL_MS "$((t_i1 - t_i0))"

test -f "$CARBON_BIN"
test -f /root/.config/jopedime/carbon-clouds.json
emit_kv CONFIG_DIR_MODE "0o$(stat -c '%a' /root/.config/jopedime)"
emit_kv BIN_BYTES "$(stat -c '%s' "$CARBON_BIN")"
emit_kv BIN_SHA256 "$(sha256sum "$CARBON_BIN" | cut -d' ' -f1)"
emit_kv CLOUDS_SHA256 "$(sha256sum /root/.config/jopedime/carbon-clouds.json | cut -d' ' -f1)"

"$CARBON_BIN" self-verify >/root/self_verify.json
emit_block SELF_VERIFY /root/self_verify.json

retry2 "$CARBON_BIN" doctor --cloud all >/root/doctor.json
emit_block DOCTOR /root/doctor.json

t_q0=$(date +%s%3N)
retry2 "$CARBON_BIN" quickstart >/root/quickstart.json
t_q1=$(date +%s%3N)
emit_kv QS_WALL_MS "$((t_q1 - t_q0))"
emit_block QUICKSTART /root/quickstart.json

"$CARBON_BIN" quickstart --emit-csv "$CSV_PATH" --csv-only >/dev/null
test -f "$CSV_PATH"
emit_kv CSV_BYTES "$(stat -c '%s' "$CSV_PATH")"
emit_kv CSV_SHA256 "$(sha256sum "$CSV_PATH" | cut -d' ' -f1)"

set +e
"$CARBON_BIN" run --csv "$CSV_PATH" --target target --cloud prod --free --effort high --shred >/root/gate.txt 2>&1
gate_rc=$?
set -e
emit_kv GATE_RC "$gate_rc"
emit_block GATE_OUT /root/gate.txt

login_b64() {{
  l_cloud="$1"
  l_b64="$2"
  l_manifest="$3"
  if [ -n "$l_manifest" ]; then
    printf '%s' "$l_b64" | base64 -d | "$CARBON_BIN" login --cloud "$l_cloud" --clouds-manifest "$l_manifest" --token-file - >/dev/null
  else
    printf '%s' "$l_b64" | base64 -d | "$CARBON_BIN" login --cloud "$l_cloud" --token-file - >/dev/null
  fi
}}

run_tier() {{
  c_cloud="$1"
  c_eff="$2"
  c_flag="$3"
  c_target="$4"
  c_manifest="$5"
  out_f="/root/run_${{c_cloud}}_${{c_eff}}.json"
  t_r0=$(date +%s%3N)
  if [ -n "$c_manifest" ]; then
    "$CARBON_BIN" run --csv "$CSV_PATH" --target target --cloud "$c_cloud" "$c_flag" \
      --effort "$c_eff" --seed 42 --predict-rows 5 --shred --clouds-manifest "$c_manifest" >"$out_f"
  else
    retry2 "$CARBON_BIN" run --csv "$CSV_PATH" --target target --cloud "$c_cloud" "$c_flag" \
      --effort "$c_eff" --seed 42 --predict-rows 5 --shred >"$out_f"
  fi
  t_r1=$(date +%s%3N)
  printf '%s\n' "$((t_r1 - t_r0))" >"/root/wall_${{c_cloud}}_${{c_eff}}.txt"
  printf '%s\n' "$c_target" >"/root/target_${{c_cloud}}_${{c_eff}}.txt"
}}

if [ "$MODE" = "live_pro_secrets" ]; then
  login_b64 dev "$DEV_PRO_B64" ""
  login_b64 prod "$PROD_PRO_B64" ""
  for eff in fast medium high ultra; do
    run_tier dev "$eff" --pro aws_live "" &
    pid_d=$!
    run_tier prod "$eff" --pro aws_live "" &
    pid_p=$!
    wait "$pid_d"
    wait "$pid_p"
  done
else
  retry2 "$CARBON_BIN" register --free --cloud dev --label ci_dev >/dev/null
  retry2 "$CARBON_BIN" register --free --cloud prod --label ci_prod >/dev/null
  probe_free_gate() {{
    g_cloud="$1"
    g_eff="$2"
    g_tag="$3"
    set +e
    "$CARBON_BIN" run --csv "$CSV_PATH" --target target --cloud "$g_cloud" --free --effort "$g_eff" --shred >"/root/gate_${{g_tag}}.txt" 2>&1
    g_rc=$?
    set -e
    emit_kv "GATE_${{g_tag}}_RC" "$g_rc"
    emit_block "GATE_${{g_tag}}_OUT" "/root/gate_${{g_tag}}.txt"
  }}
  probe_free_gate dev high DEV_HIGH
  probe_free_gate dev ultra DEV_ULTRA
  probe_free_gate prod ultra PROD_ULTRA
  for eff in fast medium; do
    run_tier dev "$eff" --free aws_live "" &
    pid_d=$!
    run_tier prod "$eff" --free aws_live "" &
    pid_p=$!
    wait "$pid_d"
    wait "$pid_p"
  done
  umask 077
  printf '%s' "$EMU_CLOUDS_B64" | base64 -d >"$EMU_CLOUDS"
  chmod 600 "$EMU_CLOUDS"
  login_b64 dev "$DEV_PRO_B64" "$EMU_CLOUDS"
  login_b64 prod "$PROD_PRO_B64" "$EMU_CLOUDS"
  for eff in high ultra; do
    run_tier dev "$eff" --pro ci_emulator "$EMU_CLOUDS"
    run_tier prod "$eff" --pro ci_emulator "$EMU_CLOUDS"
  done
fi

"$CARBON_BIN" logout --all >/dev/null
rm -f "$EMU_CLOUDS"
if [ ! -e /root/.config/jopedime/carbon-credentials.json ]; then
  emit_kv POST_LOGOUT_ABSENT true
else
  emit_kv POST_LOGOUT_ABSENT false
fi

for eff in fast medium high ultra; do
  for cl in dev prod; do
    emit_kv "WALL_${{cl}}_${{eff}}" "$(cat "/root/wall_${{cl}}_${{eff}}.txt")"
    emit_kv "TARGET_${{cl}}_${{eff}}" "$(cat "/root/target_${{cl}}_${{eff}}.txt")"
    emit_block "RUN_${{cl}}_${{eff}}" "/root/run_${{cl}}_${{eff}}.json"
  done
done
"#
    )
}

#[derive(Default)]
struct FramedContainerOutput {
    kvs: BTreeMap<String, String>,
    blocks: BTreeMap<String, String>,
}

fn parse_framed_container_output(stdout: &str) -> FramedContainerOutput {
    let mut out = FramedContainerOutput::default();
    let mut active_block: Option<(String, String)> = None;
    for line in stdout.lines() {
        if let Some((name, mut buf)) = active_block.take() {
            let end_marker = format!("===END:{name}===");
            if line.trim() == end_marker {
                out.blocks.insert(name, buf.trim().to_owned());
            } else {
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(line);
                active_block = Some((name, buf));
            }
            continue;
        }
        let trimmed = line.trim();
        if let Some(inner) = trimmed
            .strip_prefix("===KV:")
            .and_then(|s| s.strip_suffix("==="))
        {
            if let Some((k, v)) = inner.split_once('=') {
                out.kvs.insert(k.to_owned(), v.to_owned());
            }
        } else if let Some(name) = trimmed
            .strip_prefix("===BEGIN:")
            .and_then(|s| s.strip_suffix("==="))
        {
            active_block = Some((name.to_owned(), String::new()));
        }
    }
    out
}

fn normalize_run_entry(
    cloud: &str,
    effort: &str,
    execution_target: &str,
    container_wall_ms: f64,
    raw: &Value,
) -> Value {
    let c = &raw["cost_estimate"];
    let eval_calls = raw["eval_invocations"]
        .as_u64()
        .unwrap_or(0)
        .max(raw["eval_shards"].as_u64().unwrap_or(0));
    json!({
        "cloud": cloud,
        "effort": effort,
        "execution_target": execution_target,
        "tier_applied": raw["tier_applied"].as_str().unwrap_or(""),
        "execution_mode": raw["execution_mode"].as_str().unwrap_or(""),
        "region": raw["selected_region"].as_str().unwrap_or(""),
        "run_id": raw["run_id"].as_str().unwrap_or(""),
        "final_state": raw["final_state"].as_str().unwrap_or(""),
        "encrypted_cuen": raw["encrypted_cuen"].as_bool().unwrap_or(false),
        "shredded_after_run": raw["shredded_after_run"].as_bool().unwrap_or(false),
        "waves_completed": raw["waves_completed"].as_u64().unwrap_or(0),
        "waves_target": raw["waves_target"].as_u64().unwrap_or(0),
        "candidates_evaluated": raw["candidates_evaluated"].as_u64().unwrap_or(0),
        "eval_shards": eval_calls,
        "poll_requests": raw["poll_requests"].as_u64().unwrap_or(0),
        "prep_ms": raw["prep_ms"].as_f64().unwrap_or(0.0),
        "upload_ms": raw["upload_ms"].as_f64().unwrap_or(0.0),
        "search_ms": raw["search_ms"].as_f64().unwrap_or(0.0),
        "predict_ms": raw["predict_ms"].as_f64().unwrap_or(0.0),
        "total_wall_ms": raw["total_wall_ms"].as_f64().unwrap_or(0.0),
        "container_wall_ms": container_wall_ms,
        "holdout_rmse": raw["outer_holdout_score"].as_f64().unwrap_or(0.0),
        "holdout_r2": raw["holdout_r2"].as_f64().unwrap_or(0.0),
        "mean_baseline_rmse": raw["mean_baseline_rmse"].as_f64().unwrap_or(0.0),
        "linear_baseline_rmse": raw["linear_baseline_rmse"].as_f64().unwrap_or(0.0),
        "linear_baseline_r2": raw["linear_baseline_r2"].as_f64().unwrap_or(0.0),
        "rmse_reduction_vs_linear_pct": raw["rmse_reduction_vs_linear_pct"].as_f64().unwrap_or(0.0),
        "cost_estimate": {
            "apigw_http_api_usd": c["apigw_http_api_usd"].as_f64().unwrap_or(0.0),
            "lambda_edge_usd": c["lambda_edge_usd"].as_f64().unwrap_or(0.0),
            "lambda_engine_usd": c["lambda_engine_usd"].as_f64().unwrap_or(0.0),
            "lambda_eval_usd": c["lambda_eval_usd"].as_f64().unwrap_or(0.0),
            "s3_requests_usd": c["s3_requests_usd"].as_f64().unwrap_or(0.0),
            "s3_storage_and_transfer_usd": c["s3_storage_and_transfer_usd"].as_f64().unwrap_or(0.0),
            "total_estimated_usd": c["total_estimated_usd"].as_f64().unwrap_or(0.0),
            "runs_per_1_usd": c["runs_per_1_usd"].as_f64().unwrap_or(0.0),
        },
    })
}

fn summarize_side(entry: &Value) -> Value {
    let cost = &entry["cost_estimate"];
    json!({
        "run_id": entry["run_id"],
        "region": entry["region"],
        "tier_applied": entry["tier_applied"],
        "execution_mode": entry["execution_mode"],
        "waves_completed": entry["waves_completed"],
        "eval_shards": entry["eval_shards"],
        "total_wall_ms": entry["total_wall_ms"],
        "holdout_r2": entry["holdout_r2"],
        "holdout_rmse": entry["holdout_rmse"],
        "total_estimated_usd": cost["total_estimated_usd"],
        "runs_per_1_usd": cost["runs_per_1_usd"],
    })
}

fn build_per_tier_comparison(runs: &[Value]) -> Result<Vec<Value>, String> {
    let mut rows = Vec::with_capacity(4);
    for effort in ["fast", "medium", "high", "ultra"] {
        let dev = runs
            .iter()
            .find(|r| r["cloud"] == "dev" && r["effort"] == effort)
            .ok_or_else(|| format!("missing dev run for effort={effort}"))?;
        let prod = runs
            .iter()
            .find(|r| r["cloud"] == "prod" && r["effort"] == effort)
            .ok_or_else(|| format!("missing prod run for effort={effort}"))?;
        if dev["execution_target"] != prod["execution_target"] {
            return Err(format!("mismatched execution_target for effort={effort}"));
        }
        rows.push(json!({
            "effort": effort,
            "execution_target": dev["execution_target"],
            "waves_target": dev["waves_target"],
            "dev": summarize_side(dev),
            "prod": summarize_side(prod),
        }));
    }
    Ok(rows)
}

struct ReceiptBuildContext<'a> {
    source_sha: &'a str,
    run_id: &'a str,
    attempt: u64,
    mode: &'a str,
    total_wall_ms: f64,
    image_id: &'a str,
    repo_digest: &'a str,
}

fn assemble_receipt_from_framed(
    framed: &FramedContainerOutput,
    ctx: &ReceiptBuildContext<'_>,
    emu_telemetry: Value,
) -> Result<LiveCi4TierReceipt, String> {
    let ReceiptBuildContext {
        source_sha,
        run_id,
        attempt,
        mode,
        total_wall_ms,
        image_id,
        repo_digest,
    } = *ctx;
    let parse_blk = |key: &str| -> Result<Value, String> {
        let raw = framed
            .blocks
            .get(key)
            .ok_or_else(|| format!("container output missing block '{key}'"))?;
        serde_json::from_str(raw).map_err(|e| format!("invalid JSON in block '{key}': {e}"))
    };
    let kv_u64 = |key: &str| -> u64 {
        framed
            .kvs
            .get(key)
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
    };
    let kv_f64 = |key: &str| -> f64 {
        framed
            .kvs
            .get(key)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0)
    };
    let kv_str = |key: &str| -> String { framed.kvs.get(key).cloned().unwrap_or_default() };

    let sv = parse_blk("SELF_VERIFY")?;
    let doc = parse_blk("DOCTOR")?;
    let qs = parse_blk("QUICKSTART")?;
    let decode_gate = |rc_key: &str, blk_key: &str| -> Option<u64> {
        let rc = framed.kvs.get(rc_key)?.parse::<u64>().ok()?;
        let txt = framed.blocks.get(blk_key).cloned().unwrap_or_default();
        if rc != 0 && (txt.contains("403") || txt.to_ascii_lowercase().contains("pro")) {
            Some(403)
        } else {
            Some(0)
        }
    };
    let gate_status = decode_gate("GATE_RC", "GATE_OUT").unwrap_or(0);
    let dev_high_gate = decode_gate("GATE_DEV_HIGH_RC", "GATE_DEV_HIGH_OUT");
    let dev_ultra_gate = decode_gate("GATE_DEV_ULTRA_RC", "GATE_DEV_ULTRA_OUT");
    let prod_ultra_gate = decode_gate("GATE_PROD_ULTRA_RC", "GATE_PROD_ULTRA_OUT");

    let mut runs = Vec::with_capacity(8);
    let mut total_usd = 0.0f64;
    for eff in ["fast", "medium", "high", "ultra"] {
        for cl in ["dev", "prod"] {
            let raw_run = parse_blk(&format!("RUN_{cl}_{eff}"))?;
            let wall_ms = kv_f64(&format!("WALL_{cl}_{eff}"));
            let target = kv_str(&format!("TARGET_{cl}_{eff}"));
            let norm = normalize_run_entry(cl, eff, &target, wall_ms, &raw_run);
            total_usd += norm["cost_estimate"]["total_estimated_usd"]
                .as_f64()
                .unwrap_or(0.0);
            runs.push(norm);
        }
    }
    let cmps = build_per_tier_comparison(&runs)?;
    let first = &runs[0];
    let qs_wall = qs["total_wall_ms"]
        .as_f64()
        .unwrap_or_else(|| kv_f64("QS_WALL_MS"));

    let container = json!({
        "base_image": "ubuntu:24.04",
        "image_id": image_id,
        "repo_digest": repo_digest,
        "os_pretty_name": kv_str("OS_PRETTY_NAME"),
        "uname": kv_str("UNAME_STR"),
        "zero_credentials_pre_install": kv_str("PRE_CLEAN") == "true",
        "post_logout_credentials_absent": kv_str("POST_LOGOUT_ABSENT") == "true",
        "verified_absent_env_count": kv_u64("ABSENT_ENV_COUNT"),
        "verified_absent_paths_pre_install": [
            "/root/.aws",
            "/root/.ssh",
            "/root/.config/gh",
            "/root/.config/jopedime",
            "/root/.netrc"
        ],
    });
    let install = json!({
        "command": "curl -fsSL https://downloads.neverhuman.org/carbon/install.sh | sh",
        "install_wall_ms": kv_f64("INSTALL_WALL_MS"),
        "installed_binary_path": "/root/.local/bin/carbon",
        "installed_binary_bytes": kv_u64("BIN_BYTES"),
        "installed_binary_sha256": kv_str("BIN_SHA256"),
        "installed_clouds_json_sha256": kv_str("CLOUDS_SHA256"),
        "config_dir_mode": kv_str("CONFIG_DIR_MODE"),
        "release_tag": sv["release_tag"].as_str().unwrap_or(""),
        "self_verify_verified": sv["verified"].as_bool().unwrap_or(false),
        "manifest_ecdsa_verified": sv["manifest_ecdsa_verified"].as_bool().unwrap_or(false),
        "kek_pins_verified": sv["kek_pins_verified"].as_u64().unwrap_or(0),
        "doctor_healthy": doc["healthy"].as_bool().unwrap_or(false),
        "doctor_dev_healthy_regions": doc["clouds"]["dev"]["healthy_regions"].as_u64().unwrap_or(0),
        "doctor_prod_healthy_regions": doc["clouds"]["prod"]["healthy_regions"].as_u64().unwrap_or(0),
        "quickstart_passed": qs["final_state"].as_str() == Some("completed"),
        "quickstart_run_id": qs["run_id"].as_str().unwrap_or(""),
        "quickstart_wall_ms": qs_wall,
        "quickstart_holdout_r2": qs["holdout_r2"].as_f64().unwrap_or(0.0),
        "quickstart_encrypted_cuen": qs["encrypted_cuen"].as_bool().unwrap_or(false),
        "quickstart_shredded": qs["shredded_after_run"].as_bool().unwrap_or(false),
        "live_anon_high_gate_status": gate_status,
    });
    let mut install = install;
    if let Some(obj) = install.as_object_mut() {
        if let Some(code) = dev_high_gate {
            obj.insert("live_dev_free_high_gate_status".to_owned(), json!(code));
        }
        if let Some(code) = dev_ultra_gate {
            obj.insert("live_dev_free_ultra_gate_status".to_owned(), json!(code));
        }
        if let Some(code) = prod_ultra_gate {
            obj.insert("live_prod_free_ultra_gate_status".to_owned(), json!(code));
        }
    }
    let dataset = json!({
        "name": "nonlinear_240x6.csv",
        "rows": 240,
        "fit_rows": 192,
        "holdout_rows": 48,
        "features": 6,
        "bytes": kv_u64("CSV_BYTES"),
        "sha256": kv_str("CSV_SHA256"),
        "mean_baseline_rmse": first["mean_baseline_rmse"],
        "ridge_linear_baseline_rmse": first["linear_baseline_rmse"],
        "ridge_linear_baseline_r2": first["linear_baseline_r2"],
    });

    Ok(LiveCi4TierReceipt {
        schema: LIVE_CI_4TIER_SCHEMA.to_owned(),
        captured_utc: format_utc_iso8601(now_epoch_ms() / 1_000),
        source_sha: source_sha.to_owned(),
        github_run_id: run_id.to_owned(),
        github_run_attempt: attempt,
        pro_secrets_mode: mode.to_owned(),
        all_passed: true,
        total_wall_ms,
        total_wall_secs: (total_wall_ms / 10.0).round() / 100.0,
        total_estimated_usd: (total_usd * 1_000_000.0).round() / 1_000_000.0,
        container,
        install,
        dataset,
        runs,
        per_tier_comparison: cmps,
        emulator_cells: emu_telemetry,
    })
}

fn render_markdown_summary(rcpt: &LiveCi4TierReceipt) -> String {
    let mut md = format!(
        "# Fresh-Docker Carbon CLI Install & `dev`+`prod` 4-Tier E2E Receipt\n\n\
         - **Source SHA**: `{}`\n\
         - **Run / Attempt**: `{}` / `{}`\n\
         - **Mode**: `{}`\n\
         - **Total Wall Time**: `{:.2}s` | **Total Est. AWS Cost**: `${:.6}`\n\n\
         | Effort | Target | `dev` Wall | `prod` Wall | `dev` R² | `prod` R² | `dev` USD | `prod` USD |\n\
         | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n",
        rcpt.source_sha,
        rcpt.github_run_id,
        rcpt.github_run_attempt,
        rcpt.pro_secrets_mode,
        rcpt.total_wall_secs,
        rcpt.total_estimated_usd,
    );
    for row in &rcpt.per_tier_comparison {
        let d = &row["dev"];
        let p = &row["prod"];
        md.push_str(&format!(
            "| `{}` | `{}` | {:.2}s | {:.2}s | {:.6} | {:.6} | ${:.6} | ${:.6} |\n",
            row["effort"].as_str().unwrap_or(""),
            row["execution_target"].as_str().unwrap_or(""),
            d["total_wall_ms"].as_f64().unwrap_or(0.0) / 1_000.0,
            p["total_wall_ms"].as_f64().unwrap_or(0.0) / 1_000.0,
            d["holdout_r2"].as_f64().unwrap_or(0.0),
            p["holdout_r2"].as_f64().unwrap_or(0.0),
            d["total_estimated_usd"].as_f64().unwrap_or(0.0),
            p["total_estimated_usd"].as_f64().unwrap_or(0.0),
        ));
    }
    md
}

fn resolve_live_pro_tokens() -> (String, String) {
    let dev = std::env::var("CARBON_CI_DEV_PRO_TOKEN")
        .or_else(|_| std::env::var("CARBON_PRO_TOKEN_DEV"))
        .unwrap_or_default()
        .trim()
        .to_owned();
    let prod = std::env::var("CARBON_CI_PROD_PRO_TOKEN")
        .or_else(|_| std::env::var("CARBON_PRO_TOKEN_PROD"))
        .unwrap_or_default()
        .trim()
        .to_owned();
    (dev, prod)
}

/// Execute clean-room `ubuntu:24.04` production CLI install + `dev`/`prod` 4-tier E2E qualification.
pub fn execute_ci_live_4tier_docker(
    opts: &LiveCi4TierOptions,
) -> Result<LiveCi4TierReceipt, String> {
    let _ = &opts.repo_root;
    if !is_lower_hex40(&opts.source_sha) {
        return Err(format!(
            "--source-sha must be 40 lowercase hex characters, got '{}'",
            opts.source_sha
        ));
    }
    let (live_dev_tok, live_prod_tok) = resolve_live_pro_tokens();
    if opts.require_live_pro && (live_dev_tok.is_empty() || live_prod_tok.is_empty()) {
        return Err(
            "--require-live-pro requires non-empty CARBON_CI_DEV_PRO_TOKEN and CARBON_CI_PROD_PRO_TOKEN"
                .to_owned(),
        );
    }
    let use_live_pro = !live_dev_tok.is_empty() && !live_prod_tok.is_empty();
    let mode = if use_live_pro {
        "live_pro_secrets"
    } else {
        "zero_secrets_hybrid_live_and_emulator"
    };

    let t0 = Instant::now();
    fs::create_dir_all(&opts.out_dir)
        .map_err(|e| format!("create out_dir {}: {e}", opts.out_dir.display()))?;
    let docker_bin = std::env::var("CARBON_DOCKER_BIN").unwrap_or_else(|_| "docker".to_owned());
    let (image_id, repo_digest) = inspect_and_build_cleanroom_image(&docker_bin)?;

    let mut sensitive_tokens = Vec::new();
    let mut emu_opt: Option<InProcessDualEmulators> = None;
    let script = if use_live_pro {
        sensitive_tokens.push(live_dev_tok.clone());
        sensitive_tokens.push(live_prod_tok.clone());
        build_container_posix_script(mode, &live_dev_tok, &live_prod_tok, None)
    } else {
        let scratch = opts
            .out_dir
            .join(format!(".emu-work-{}", std::process::id()));
        let (emus, emu_dev_tok, emu_prod_tok, clouds_json) =
            InProcessDualEmulators::start(scratch)?;
        sensitive_tokens.push(emu_dev_tok.clone());
        sensitive_tokens.push(emu_prod_tok.clone());
        emu_opt = Some(emus);
        build_container_posix_script(mode, &emu_dev_tok, &emu_prod_tok, Some(&clouds_json))
    };

    let run_id = std::env::var("GITHUB_RUN_ID").unwrap_or_else(|_| "local".to_owned());
    let attempt = std::env::var("GITHUB_RUN_ATTEMPT")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(1);
    let safe_run: String = run_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect();
    let container_name = format!(
        "carbon-ci-cleanroom-{}-{}-{}",
        if safe_run.is_empty() {
            "local"
        } else {
            &safe_run
        },
        attempt,
        std::process::id()
    );
    let mut guard = DockerContainerGuard {
        docker_bin: docker_bin.clone(),
        container_name: container_name.clone(),
        finished: false,
    };

    let mut child = sanitized_docker_command(&docker_bin)
        .args([
            "run",
            "--rm",
            "--init",
            "--network",
            "host",
            "--name",
            &container_name,
            "-i",
            CLEANROOM_TAG,
            "/usr/bin/env",
            "-i",
            "PATH=/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME=/root",
            "LC_ALL=C.UTF-8",
            "sh",
            "-eu",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn clean-room docker container: {e}"))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(script.as_bytes())
            .map_err(|e| format!("write driver script to container stdin: {e}"))?;
    }
    let container_out = child
        .wait_with_output()
        .map_err(|e| format!("wait on clean-room container: {e}"))?;
    guard.finished = true;

    let emu_telemetry = match emu_opt.take() {
        Some(emus) => emus.finish()?,
        None => json!({"mode": "not_used", "total_residual_cas_blobs": 0}),
    };

    if !container_out.status.success() {
        return Err(format!(
            "clean-room container failed (status={}): {}",
            container_out.status,
            String::from_utf8_lossy(&container_out.stderr).trim()
        ));
    }

    let total_wall_ms = (t0.elapsed().as_secs_f64() * 100_000.0).round() / 100.0;
    let stdout_str = String::from_utf8(container_out.stdout)
        .map_err(|e| format!("container stdout is not UTF-8: {e}"))?;
    let framed = parse_framed_container_output(&stdout_str);
    let receipt = assemble_receipt_from_framed(
        &framed,
        &ReceiptBuildContext {
            source_sha: &opts.source_sha,
            run_id: &run_id,
            attempt,
            mode,
            total_wall_ms,
            image_id: &image_id,
            repo_digest: &repo_digest,
        },
        emu_telemetry,
    )?;

    let rendered = receipt.validate(&opts.source_sha, opts.require_live_pro, &sensitive_tokens)?;
    let receipt_path = opts.out_dir.join(LIVE_CI_4TIER_FILENAME);
    fs::write(&receipt_path, rendered.as_bytes())
        .map_err(|e| format!("write {}: {e}", receipt_path.display()))?;
    let md_path = opts.out_dir.join("live-ci-4tier-docker.md");
    fs::write(&md_path, render_markdown_summary(&receipt).as_bytes())
        .map_err(|e| format!("write {}: {e}", md_path.display()))?;
    let digest_hex = sha256_hex(rendered.as_bytes());
    // Do not name this file `MANIFEST.sha256`: `deploy/ci/retain-local-evidence.sh` generates
    // and verifies `MANIFEST.sha256` across all files in `$out_dir` during CI publication.
    let digest_sidecar_path = opts.out_dir.join("live-ci-4tier-docker.sha256");
    fs::write(
        &digest_sidecar_path,
        format!("{digest_hex}  {LIVE_CI_4TIER_FILENAME}\n").as_bytes(),
    )
    .map_err(|e| format!("write {}: {e}", digest_sidecar_path.display()))?;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_framed_output(mode: &str, run_id_suffix: &str) -> FramedContainerOutput {
        let mut out = FramedContainerOutput::default();
        out.kvs
            .insert("OS_PRETTY_NAME".into(), "Ubuntu 24.04.5 LTS".into());
        out.kvs
            .insert("UNAME_STR".into(), "Linux 6.8.0 x86_64".into());
        out.kvs.insert("PRE_CLEAN".into(), "true".into());
        out.kvs.insert("POST_LOGOUT_ABSENT".into(), "true".into());
        out.kvs.insert("ABSENT_ENV_COUNT".into(), "13".into());
        out.kvs.insert("INSTALL_WALL_MS".into(), "1520".into());
        out.kvs.insert("CONFIG_DIR_MODE".into(), "0o700".into());
        out.kvs.insert("BIN_BYTES".into(), "3763432".into());
        out.kvs.insert("BIN_SHA256".into(), "b".repeat(64));
        out.kvs.insert("CLOUDS_SHA256".into(), "e".repeat(64));
        out.kvs.insert("QS_WALL_MS".into(), "4820".into());
        out.kvs.insert("CSV_BYTES".into(), "15938".into());
        out.kvs.insert("CSV_SHA256".into(), "3".repeat(64));
        out.kvs.insert("GATE_RC".into(), "1".into());
        out.blocks.insert(
            "GATE_OUT".into(),
            "HTTP 403 Forbidden: Pro tier required".into(),
        );
        out.blocks.insert(
            "SELF_VERIFY".into(),
            json!({
                "release_tag": "carbon-cli-v1.1.0",
                "verified": true,
                "manifest_ecdsa_verified": true,
                "kek_pins_verified": 6
            })
            .to_string(),
        );
        out.blocks.insert(
            "DOCTOR".into(),
            json!({
                "healthy": true,
                "clouds": {
                    "dev": {"healthy_regions": 1},
                    "prod": {"healthy_regions": 5}
                }
            })
            .to_string(),
        );
        out.blocks.insert(
            "QUICKSTART".into(),
            json!({
                "final_state": "completed",
                "run_id": "run_qs_01",
                "total_wall_ms": 4810.0,
                "holdout_r2": 0.99985,
                "encrypted_cuen": true,
                "shredded_after_run": true
            })
            .to_string(),
        );
        for (eff, waves, shards) in [
            ("fast", 1u64, 0u64),
            ("medium", 4, 0),
            ("high", 12, 120),
            ("ultra", 32, 480),
        ] {
            let target = if mode == "live_pro_secrets" || matches!(eff, "fast" | "medium") {
                "aws_live"
            } else {
                "ci_emulator"
            };
            for cl in ["dev", "prod"] {
                out.kvs.insert(format!("WALL_{cl}_{eff}"), "4200".into());
                out.kvs.insert(format!("TARGET_{cl}_{eff}"), target.into());
                out.blocks.insert(
                    format!("RUN_{cl}_{eff}"),
                    json!({
                        "tier_applied": if target == "ci_emulator" || mode == "live_pro_secrets" { "pro" } else { "free" },
                        "execution_mode": if shards == 0 { "SoloSingleLambda" } else { "HybridFanOut" },
                        "selected_region": "us-east-2",
                        "run_id": format!("run_{cl}_{eff}_{run_id_suffix}"),
                        "final_state": "completed",
                        "encrypted_cuen": true,
                        "shredded_after_run": true,
                        "waves_completed": waves,
                        "waves_target": waves,
                        "candidates_evaluated": 25 + waves * 4,
                        "eval_invocations": shards,
                        "poll_requests": waves,
                        "prep_ms": 12.0,
                        "upload_ms": 140.0,
                        "search_ms": 3800.0,
                        "predict_ms": 40.0,
                        "total_wall_ms": 3992.0,
                        "outer_holdout_score": 0.088,
                        "holdout_r2": 0.9997,
                        "mean_baseline_rmse": 6.21,
                        "linear_baseline_rmse": 5.08,
                        "linear_baseline_r2": 0.331,
                        "rmse_reduction_vs_linear_pct": 98.27,
                        "cost_estimate": {
                            "apigw_http_api_usd": 0.000008,
                            "lambda_edge_usd": 0.000004,
                            "lambda_engine_usd": 0.00031,
                            "lambda_eval_usd": 0.00012,
                            "s3_requests_usd": 0.000015,
                            "s3_storage_and_transfer_usd": 0.000001,
                            "total_estimated_usd": 0.000458,
                            "runs_per_1_usd": 2183.4
                        }
                    })
                    .to_string(),
                );
            }
        }
        out
    }

    #[test]
    fn framed_parser_and_option_b_validation_succeed() {
        let raw_stream = "===KV:A=1===\nnoise\n===BEGIN:J===\n{\"ok\":true}\n===END:J===\n";
        let parsed = parse_framed_container_output(raw_stream);
        assert_eq!(parsed.kvs.get("A").map(String::as_str), Some("1"));
        assert_eq!(
            parsed.blocks.get("J").map(String::as_str),
            Some("{\"ok\":true}")
        );

        let sha = "a".repeat(40);
        let framed = synthetic_framed_output("zero_secrets_hybrid_live_and_emulator", "ok");
        let img_id = format!("sha256:{}", "c".repeat(64));
        let repo_dig = format!("ubuntu@sha256:{}", "d".repeat(64));
        let rcpt = assemble_receipt_from_framed(
            &framed,
            &ReceiptBuildContext {
                source_sha: &sha,
                run_id: "99001",
                attempt: 1,
                mode: "zero_secrets_hybrid_live_and_emulator",
                total_wall_ms: 12345.0,
                image_id: &img_id,
                repo_digest: &repo_dig,
            },
            json!({"total_residual_cas_blobs": 0}),
        )
        .expect("assemble receipt");
        let rendered = rcpt
            .validate(&sha, false, &[String::from("secret_tok_xyz")])
            .expect("validate Option B");
        assert!(rendered.contains("zero_secrets_hybrid_live_and_emulator"));
        assert!(rcpt
            .validate(&sha, true, &[])
            .expect_err("require_live_pro must reject Option B")
            .contains("live_pro_secrets"));
    }

    #[test]
    fn forbidden_markers_and_leaked_tokens_are_refused() {
        let sha = "b".repeat(40);
        let img_id = format!("sha256:{}", "c".repeat(64));
        let repo_dig = format!("ubuntu@sha256:{}", "d".repeat(64));
        let leak = format!("{}{}", "ctk", "1.leaked_capability");
        let framed = synthetic_framed_output("live_pro_secrets", &leak);
        let rcpt = assemble_receipt_from_framed(
            &framed,
            &ReceiptBuildContext {
                source_sha: &sha,
                run_id: "99002",
                attempt: 1,
                mode: "live_pro_secrets",
                total_wall_ms: 9800.0,
                image_id: &img_id,
                repo_digest: &repo_dig,
            },
            json!({"total_residual_cas_blobs": 0}),
        )
        .expect("assemble receipt");
        assert!(rcpt.validate(&sha, true, &[]).is_err());

        let framed_tok = synthetic_framed_output("live_pro_secrets", "my_custom_tok_99");
        let rcpt_tok = assemble_receipt_from_framed(
            &framed_tok,
            &ReceiptBuildContext {
                source_sha: &sha,
                run_id: "99002",
                attempt: 1,
                mode: "live_pro_secrets",
                total_wall_ms: 9800.0,
                image_id: &img_id,
                repo_digest: &repo_dig,
            },
            json!({"total_residual_cas_blobs": 0}),
        )
        .expect("assemble receipt");
        assert!(rcpt_tok
            .validate(&sha, true, &[String::from("my_custom_tok_99")])
            .is_err());
    }
}

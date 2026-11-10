//! Bound privileged probes without signals or duplicate blocked readers.
//! Native Rust replacement for `bounded_consumer_probe.py`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::json::Json;
use crate::json_obj;
use crate::sys::{atomic_write_str, current_uid, now_nanos, utc_iso, FlockGuard};

pub const UNIT: &str = "disk-steward-consumer-probe.service";

fn xdg_runtime_dir() -> String {
    std::env::var("XDG_RUNTIME_DIR")
        .unwrap_or_else(|_| format!("/run/user/{}", current_uid()))
}

fn held(reason: &str, details: Json) -> Json {
    json_obj! {
        "complete" => false,
        "errors" => vec![reason.to_string()],
        "consumers" => Json::Obj(BTreeMap::new()),
        "probe_supervisor" => details,
    }
}

fn unit_state(unit: &str) -> Result<BTreeMap<String, String>, String> {
    let out = Command::new("systemctl")
        .env("XDG_RUNTIME_DIR", xdg_runtime_dir())
        .args([
            "--user",
            "show",
            unit,
            "-p",
            "LoadState",
            "-p",
            "ActiveState",
            "-p",
            "MainPID",
            "-p",
            "InvocationID",
        ])
        .output()
        .map_err(|e| format!("cannot establish probe unit state: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut row = BTreeMap::new();
    for line in stdout.lines() {
        if let Some((k, v)) = line.split_once('=') {
            row.insert(k.to_string(), v.to_string());
        }
    }
    if row.is_empty() {
        let stderr: String = String::from_utf8_lossy(&out.stderr)
            .chars()
            .take(200)
            .collect();
        return Err(format!("cannot establish probe unit state: {stderr}"));
    }
    Ok(row)
}

pub fn request(
    roots: &[PathBuf],
    base: &Path,
    deadline_secs: u64,
    unit: &str,
    custom_command: Option<Vec<String>>,
) -> Json {
    let folder = base.join("state/consumer-probes");
    if let Err(e) = fs::create_dir_all(&folder) {
        return held(
            &format!("consumer probe supervisor incomplete: {e}"),
            Json::empty_obj(),
        );
    }
    let Ok(_lock) = FlockGuard::acquire_nonblocking(&folder.join("supervisor.lock")) else {
        return held(
            "another consumer-probe supervisor is active",
            Json::empty_obj(),
        );
    };

    let state = match unit_state(unit) {
        Ok(s) => s,
        Err(e) => return held(&format!("consumer probe supervisor incomplete: {e}"), Json::empty_obj()),
    };
    let active = state.get("ActiveState").map(|s| s.as_str()).unwrap_or("");
    let load = state.get("LoadState").map(|s| s.as_str()).unwrap_or("");
    if !matches!(active, "inactive" | "failed") && load != "not-found" {
        let state_json: BTreeMap<String, Json> = state
            .into_iter()
            .map(|(k, v)| (k, Json::Str(v)))
            .collect();
        return held(
            "existing owned consumer probe still active; no duplicate launched",
            json_obj! { "unit" => Json::Obj(state_json) },
        );
    }

    let stamp = now_nanos().to_string();
    let packet = folder.join(&stamp);
    if let Err(e) = fs::create_dir_all(&packet) {
        return held(&format!("mkdir packet failed: {e}"), Json::empty_obj());
    }
    let _ = fs::set_permissions(&packet, fs::Permissions::from_mode(0o700));

    let self_exe = std::env::current_exe()
        .unwrap_or_else(|_| base.join("bin/jope-disk-steward"));
    let default_cmd = vec![
        "sudo".to_string(),
        "-n".to_string(),
        self_exe.to_string_lossy().to_string(),
        "probe-consumers".to_string(),
    ];
    let cmd_vec = custom_command.unwrap_or(default_cmd);
    let root_strs: Vec<String> = roots
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    let result_path = packet.join("result.json");
    let spec_path = packet.join("request.json");

    let spec = json_obj! {
        "utc" => utc_iso(),
        "roots" => root_strs.clone(),
        "command" => cmd_vec,
        "result" => result_path.to_string_lossy().to_string(),
    };
    if let Err(e) = atomic_write_str(&spec_path, &format!("{}\n", spec.to_compact())) {
        return held(&format!("write request.json failed: {e}"), Json::empty_obj());
    }
    let _ = atomic_write_str(
        &folder.join("latest.json"),
        &format!(
            "{}\n",
            json_obj! {
                "utc" => utc_iso(),
                "unit" => unit,
                "request" => spec_path.to_string_lossy().to_string(),
                "status" => "launching",
            }
            .to_compact()
        ),
    );

    let launch = Command::new("systemd-run")
        .env("XDG_RUNTIME_DIR", xdg_runtime_dir())
        .args([
            "--user",
            "--no-block",
            "--collect",
            &format!("--unit={unit}"),
            "--property=Type=oneshot",
            "--property=Nice=19",
            "--property=IOSchedulingClass=idle",
            "--property=TimeoutStartSec=infinity",
            &self_exe.to_string_lossy(),
            "bounded-probe-worker",
            &spec_path.to_string_lossy(),
        ])
        .output();

    match launch {
        Ok(q) if !q.status.success() => {
            let stderr: String = String::from_utf8_lossy(&q.stderr)
                .chars()
                .take(300)
                .collect();
            return held(
                &format!("owned probe launch failed: {stderr}"),
                json_obj! { "request" => spec_path.to_string_lossy().to_string() },
            );
        }
        Err(e) => {
            return held(
                &format!("owned probe launch failed: {e}"),
                json_obj! { "request" => spec_path.to_string_lossy().to_string() },
            );
        }
        _ => {}
    }

    let deadline = Instant::now() + Duration::from_secs(deadline_secs);
    while Instant::now() < deadline {
        if result_path.exists() {
            let Ok(row) = Json::load_file(&result_path) else {
                return held("consumer probe supervisor incomplete: malformed result.json", Json::empty_obj());
            };
            let _ = atomic_write_str(
                &folder.join("latest.json"),
                &format!(
                    "{}\n",
                    json_obj! {
                        "utc" => utc_iso(),
                        "unit" => unit,
                        "request" => spec_path.to_string_lossy().to_string(),
                        "status" => "finished",
                        "result" => result_path.to_string_lossy().to_string(),
                    }
                    .to_compact()
                ),
            );
            let rc = row.get("returncode").and_then(|v| v.as_i64()).unwrap_or(1);
            let stderr = row.get_str("stderr").unwrap_or("");
            if rc != 0 || !stderr.is_empty() {
                let msg: String = stderr.chars().take(500).collect();
                return held(
                    &format!("privileged consumer probe failed: {msg}"),
                    json_obj! { "result" => result_path.to_string_lossy().to_string() },
                );
            }
            let stdout = row.get_str("stdout").unwrap_or("");
            let Ok(mut raw) = Json::parse(stdout) else {
                return held("consumer probe supervisor incomplete: invalid probe stdout JSON", Json::empty_obj());
            };
            let Some(complete) = raw.get_bool("complete") else {
                return held("consumer probe supervisor incomplete: missing complete flag", Json::empty_obj());
            };
            let Some(consumers_map) = raw.get("consumers").and_then(|v| v.as_obj()) else {
                return held("consumer probe supervisor incomplete: missing consumers object", Json::empty_obj());
            };
            if complete {
                let got_keys: BTreeSet<String> = consumers_map.keys().cloned().collect();
                let exp_keys: BTreeSet<String> = root_strs.into_iter().collect();
                if got_keys != exp_keys {
                    return held("consumer probe supervisor incomplete: probe omitted requested roots", Json::empty_obj());
                }
            }
            raw.insert(
                "probe_supervisor",
                json_obj! {
                    "result" => result_path.to_string_lossy().to_string(),
                    "unit" => unit,
                    "fresh_request" => spec_path.to_string_lossy().to_string(),
                },
            );
            return raw;
        }
        thread::sleep(Duration::from_millis(100));
    }

    let _ = atomic_write_str(
        &folder.join("latest.json"),
        &format!(
            "{}\n",
            json_obj! {
                "utc" => utc_iso(),
                "unit" => unit,
                "request" => spec_path.to_string_lossy().to_string(),
                "status" => "pending_after_deadline",
            }
            .to_compact()
        ),
    );
    held(
        "consumer probe deadline reached; owned worker retained, all candidates held",
        json_obj! {
            "request" => spec_path.to_string_lossy().to_string(),
            "unit" => unit,
        },
    )
}

pub fn run_worker(spec_path: &Path) -> i32 {
    let Ok(spec) = Json::load_file(spec_path) else {
        return 1;
    };
    let started = utc_iso();
    let cmd_arr: Vec<String> = spec
        .get("command")
        .and_then(|v| v.as_arr())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let roots = spec.get("roots").cloned().unwrap_or_else(Json::empty_arr);
    let result_file = PathBuf::from(spec.get_str("result").unwrap_or(""));
    if cmd_arr.is_empty() {
        return 1;
    }

    let input_payload = json_obj! { "roots" => roots }.to_compact();
    let mut command = Command::new(&cmd_arr[0]);
    command
        .args(&cmd_arr[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let result_json = match command.spawn() {
        Ok(mut child) => {
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(input_payload.as_bytes());
            }
            match child.wait_with_output() {
                Ok(out) => json_obj! {
                    "started" => started,
                    "finished" => utc_iso(),
                    "returncode" => out.status.code().unwrap_or(1),
                    "stdout" => String::from_utf8_lossy(&out.stdout).to_string(),
                    "stderr" => String::from_utf8_lossy(&out.stderr).to_string(),
                },
                Err(e) => json_obj! {
                    "started" => started,
                    "finished" => utc_iso(),
                    "returncode" => 127,
                    "stdout" => "",
                    "stderr" => e.to_string(),
                },
            }
        }
        Err(e) => json_obj! {
            "started" => started,
            "finished" => utc_iso(),
            "returncode" => 127,
            "stdout" => "",
            "stderr" => e.to_string(),
        },
    };
    let _ = atomic_write_str(&result_file, &format!("{}\n", result_json.to_compact()));
    0
}

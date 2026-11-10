//! Persistent 20-minute measurement pass across xbabe1, xbabe2, and xbabe3.
//! Native Rust replacement for `monitor_pass.py`.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;

use crate::json::Json;
use crate::json_obj;
use crate::sys::{
    atomic_write_str, base_dir, utc_iso, utc_stamp_compact, FlockGuard,
};

fn parse_du_snapshot(path: &Path) -> (BTreeMap<String, i64>, Option<Json>, Option<f64>) {
    let mut rows = BTreeMap::new();
    let mut df = None;
    let mut when = None;
    let Ok(text) = fs::read_to_string(path) else {
        return (rows, df, when);
    };
    for line in text.lines() {
        if let Ok(item) = Json::parse(line) {
            if let Some(t) = item.get("time").and_then(|v| v.as_f64()) {
                when = Some(t);
            }
        }
        if let Some((n_str, p_str)) = line.split_once('\t') {
            if let Ok(n) = n_str.trim().parse::<i64>() {
                rows.insert(p_str.to_string(), n);
            }
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() == 6 && fields[5] == "/" && fields[4].ends_with('%') {
            if let (Ok(used), Ok(avail), Ok(df_pct)) = (
                fields[2].parse::<u64>(),
                fields[3].parse::<u64>(),
                fields[4].trim_end_matches('%').parse::<u64>(),
            ) {
                let total = used.saturating_add(avail);
                let actual = if total > 0 {
                    100.0 * (used as f64) / (total as f64)
                } else {
                    100.0
                };
                df = Some(json_obj! {
                    "used_bytes" => used,
                    "available_bytes" => avail,
                    "df_percent" => df_pct,
                    "actual_percent" => actual,
                });
            }
        }
    }
    (rows, df, when)
}

fn measure_host(
    host: &str,
    out: &Path,
    source_script: &str,
    process_script: &str,
    previous: Option<String>,
) -> Json {
    let du_path = out.join(format!("{host}.du"));
    let err_path = out.join(format!("{host}.errors"));
    let run_remote_py = |script: &str, extra_arg: Option<&str>, stdout_p: &Path, stderr_p: &Path| -> i32 {
        let mut cmd = if host == "xbabe2" {
            let mut c = Command::new("nice");
            c.args(["-n", "19", "ionice", "-c3", "python3", "-"]);
            if let Some(arg) = extra_arg {
                c.arg(arg);
            }
            c
        } else {
            let mut c = Command::new("ssh");
            let remote_cmd = match extra_arg {
                Some(a) => format!("nice -n 19 ionice -c3 python3 - {a}"),
                None => "nice -n 19 ionice -c3 python3 -".to_string(),
            };
            c.args([
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=12",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                host,
                &remote_cmd,
            ]);
            c
        };
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        match cmd.spawn() {
            Ok(mut child) => {
                if let Some(mut stdin) = child.stdin.take() {
                    let _ = stdin.write_all(script.as_bytes());
                }
                match child.wait_with_output() {
                    Ok(o) => {
                        let _ = fs::write(stdout_p, &o.stdout);
                        let _ = fs::write(stderr_p, &o.stderr);
                        o.status.code().unwrap_or(1)
                    }
                    Err(_) => 1,
                }
            }
            Err(_) => 1,
        }
    };

    let rc = run_remote_py(source_script, None, &du_path, &err_path);
    let proc_file = out.join(format!("{host}.processes.json"));
    let proc_err = out.join(format!("{host}.processes.stderr"));
    let proc_rc = run_remote_py(process_script, Some("--save"), &proc_file, &proc_err);

    let mut probe_status = json_obj! {
        "returncode" => proc_rc,
        "snapshot" => proc_file.to_string_lossy().to_string(),
        "stderr_file" => proc_err.to_string_lossy().to_string(),
    };
    if let Ok(proc_data) = Json::load_file(&proc_file) {
        if let Some(obs) = proc_data.get("observed_utc") {
            probe_status.insert("observed_utc", obs.clone());
        }
        if let Some(lrc) = proc_data.get("lsof_returncode") {
            probe_status.insert("lsof_returncode", lrc.clone());
        }
    } else {
        probe_status.insert(
            "error",
            "Process snapshot absent or malformed; not evidence of no consumers.",
        );
    }

    let (rows, df, when) = parse_du_snapshot(&du_path);
    let mut changes = Vec::new();
    let mut new_obs = Vec::new();
    if let Some(ref prev_str) = previous {
        let prev_path = PathBuf::from(prev_str);
        if prev_path.exists() {
            let (before, _, prior_time) = parse_du_snapshot(&prev_path);
            if let (Some(w), Some(pt)) = (when, prior_time) {
                let seconds = w - pt;
                if seconds > 0.0 {
                    for (p, size) in &rows {
                        if let Some(old_size) = before.get(p) {
                            changes.push((size - old_size, p.clone(), seconds));
                        } else {
                            new_obs.push((*size, p.clone(), seconds));
                        }
                    }
                    changes.sort_by(|a, b| b.0.cmp(&a.0));
                    new_obs.sort_by(|a, b| b.0.cmp(&a.0));
                }
            }
        }
    }

    let growth_json: Vec<Json> = changes
        .into_iter()
        .map(|(delta, p, secs)| {
            json_obj! {
                "path" => p,
                "delta_bytes" => delta,
                "interval_seconds" => secs,
            }
        })
        .collect();
    let new_json: Vec<Json> = new_obs
        .into_iter()
        .map(|(sz, p, secs)| {
            json_obj! {
                "path" => p,
                "observed_allocated_bytes" => sz,
                "interval_seconds" => secs,
            }
        })
        .collect();

    let partial = fs::metadata(&err_path).map(|m| m.len() > 0).unwrap_or(false);
    let res = json_obj! {
        "host" => host,
        "returncode" => rc,
        "snapshot" => du_path.to_string_lossy().to_string(),
        "observed_epoch" => when.map(Json::Float).unwrap_or(Json::Null),
        "df" => df.unwrap_or(Json::Null),
        "scope" => "focused known writers; partial/error roots excluded from growth comparisons",
        "partial" => partial,
        "previous_snapshot" => previous.map(Json::Str).unwrap_or(Json::Null),
        "growth" => Json::Arr(growth_json),
        "new_observations" => Json::Arr(new_json),
        "new_observation_scope" => "Absent from previous snapshot; creation and growth are unproven. Entries can overlap and must not be summed as root growth.",
        "process_probe" => probe_status,
        "completed_utc" => utc_iso(),
    };
    let _ = atomic_write_str(
        &out.join(format!("{host}.json")),
        &format!("{}\n", res.to_pretty()),
    );
    res
}

pub fn run_monitor_pass() -> i32 {
    let base = base_dir();
    let state = base.join("state");
    let Ok(_lock) = FlockGuard::acquire_nonblocking(&state.join("monitor.lock")) else {
        return 0;
    };
    let stamp = utc_stamp_compact();
    let started_utc = utc_iso();
    let out = state.join("loop").join(&stamp);
    if fs::create_dir_all(&out).is_err() {
        return 1;
    }
    let last_file = state.join("loop-last.json");
    let mut last = Json::load_file(&last_file).unwrap_or_else(|_| Json::empty_obj());
    let source_script = fs::read_to_string(state.join("measure_focused.py")).unwrap_or_default();
    let process_script =
        fs::read_to_string(base.join("bin/process_snapshot.py")).unwrap_or_default();

    let mut handles = Vec::new();
    for host in ["xbabe1", "xbabe2", "xbabe3"] {
        let out_c = out.clone();
        let src_c = source_script.clone();
        let proc_c = process_script.clone();
        let prev_c = last.get_str(host).map(|s| s.to_string());
        handles.push(thread::spawn(move || {
            measure_host(host, &out_c, &src_c, &proc_c, prev_c)
        }));
    }

    let mut results = Vec::new();
    for h in handles {
        if let Ok(r) = h.join() {
            results.push(r);
        }
    }

    let mut summaries = Vec::new();
    for r in &results {
        let h = r.get_str("host").unwrap_or("unknown");
        let rc = r.get("returncode").and_then(|v| v.as_i64()).unwrap_or(1);
        if let Some(df) = r.get("df").and_then(|v| v.as_obj()) {
            if rc == 0 {
                if let Some(snap) = r.get_str("snapshot") {
                    last.insert(h, snap);
                }
            }
            let df_pct = df.get("df_percent").and_then(|v| v.as_u64()).unwrap_or(0);
            let act_pct = df
                .get("actual_percent")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let avail_gib = (df
                .get("available_bytes")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as f64)
                / (1024.0 * 1024.0 * 1024.0);
            summaries.push(format!(
                "{h} df {df_pct}%, actual {act_pct:.3}%, free {avail_gib:.1} GiB"
            ));
        } else {
            summaries.push(format!("{h} measurement failed"));
        }
    }

    let _ = atomic_write_str(&last_file, &format!("{}\n", last.to_pretty()));
    let summary_str = summaries.join("; ");
    let line = format!(
        "{} — Persistent 20-minute measurement pass {}: {}. Focused snapshots/growth: {}. Full inventory, safe-reclamation evaluation and root-cause work continue separately.\n",
        utc_iso(),
        stamp,
        summary_str,
        out.display()
    );
    if let Ok(mut f) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(base.join("STATUS.md"))
    {
        let _ = f.write_all(format!("\n{line}").as_bytes());
    }
    let _ = atomic_write_str(
        &out.join("pass.json"),
        &format!(
            "{}\n",
            json_obj! {
                "started_utc" => started_utc,
                "summary" => summary_str,
                "hosts" => Json::Arr(results),
            }
            .to_pretty()
        ),
    );
    print!("{line}");
    0
}

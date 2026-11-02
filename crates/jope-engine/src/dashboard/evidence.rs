use super::campaign::campaign;
use super::prelude::*;

pub fn campaign_with_manifest(root: &Path, selection: Option<&Path>) -> Result<Value> {
    let mut run = campaign(root);
    let root = root.canonicalize()?;
    if let Some(selection) = selection {
        let selected = root.join(selection).canonicalize()?;
        ensure!(
            selected.starts_with(&root),
            "experiment manifest outside workspace"
        );
        let manifest: Value = read_json(&selected)?;
        let mut sources = Vec::new();
        if manifest["schema"] == "parallel.experiment_dashboard.v1" {
            for key in ["seed", "binary", "matched"] {
                if let Some(path) = manifest[key]["path"].as_str() {
                    let path = root.join(path).canonicalize()?;
                    ensure!(
                        path.starts_with(&root)
                            && crate::manifest::file_hash(&path)? == manifest[key]["sha256"],
                        "dashboard manifest binding changed"
                    );
                    sources.push(path);
                }
            }
            ensure!(!sources.is_empty(), "empty experiment dashboard selection");
        } else {
            sources.push(selected.clone());
        }
        run["experiment_manifest"] = selected.to_string_lossy().into_owned().into();
        run["experiment_manifest_sha256"] = crate::manifest::file_hash(&selected)?.into();
        // Explicit selection must never silently fall back to an old experiment.
        run["seed_experiment"] = Value::Null;
        run["seed_quality"] = Value::Null;
        run["binary_qualification"] = Value::Null;
        for path in sources {
            let manifest: Value = read_json(&path)?;
            let directory = path
                .parent()
                .context("experiment manifest parent missing")?;
            let schema = manifest["schema"].as_str().unwrap_or("");
            if schema == "parallel.binary_full_replay.v2" {
                run["binary_qualification"] = state(&directory.join("progress.json"));
                run["binary_verification"] = state(&directory.join("verification.json"));
                run["binary_admitted_contexts"] = manifest["views"].clone();
                run["binary_verified_contexts"] = manifest["original_verified_contexts"].clone();
            } else if schema == "parallel.seed_experiment.v1"
                || schema == "parallel.seed_experiment.v2"
            {
                if manifest["workflow"] == "matched_downstream_v2" {
                    run["matched_quality"] = state(&directory.join("report.json"));
                    run["matched_progress"] = state(&directory.join("progress.json"));
                } else {
                    run["seed_experiment"] = state(&directory.join("progress.json"));
                    run["seed_quality"] = state(&directory.join("report.json"));
                }
                run["experiment_checkpoint"] = manifest["checkpoint"].clone();
            } else {
                anyhow::bail!("unsupported experiment dashboard manifest: {schema}");
            }
        }
    }
    run["production_checkpoint"] = run["selected_checkpoint"]["checkpoint"].clone();
    if run["production_checkpoint"].is_null() {
        run["production_checkpoint"] = run["auxiliary"]["best"]["checkpoint"].clone();
    }
    if run["experiment_checkpoint"].is_null() {
        run["experiment_checkpoint"] = run["seed_quality"]["selected_checkpoint"].clone();
    }
    run["experiment_worker_evidence"] = experiment_worker_evidence(&root);
    run["campaign_resource_evidence"] = campaign_resource_evidence(&root);
    Ok(run)
}

pub(crate) fn experimental_gpu_rank(command: &[u8]) -> bool {
    let args = command.split(|&byte| byte == 0).collect::<Vec<_>>();
    let number = |flag: &[u8]| {
        args.windows(2).find_map(|pair| {
            if pair[0] != flag {
                return None;
            }
            std::str::from_utf8(pair[1]).ok()?.parse::<usize>().ok()
        })
    };
    number(b"--rank").is_some()
        && (number(b"--world")
            .or_else(|| number(b"--world-size"))
            .is_some_and(|world| world > 0)
            || args.contains(&b"qualify-batch-rank".as_slice()))
}

#[cfg(test)]
mod worker_identity_tests;

pub(crate) fn experiment_worker_evidence(root: &Path) -> Value {
    use std::os::unix::fs::MetadataExt;
    let locks = std::fs::read_to_string("/proc/locks");
    let lock_owner = |name: &str| -> Option<u32> {
        let metadata = std::fs::metadata(root.join("runs").join(name)).ok()?;
        let inode = format!(
            "{:02x}:{:02x}:{}",
            libc::major(metadata.dev()),
            libc::minor(metadata.dev()),
            metadata.ino()
        );
        locks.as_ref().ok()?.lines().find_map(|line| {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            (fields.get(5) == Some(&inode.as_str()))
                .then(|| fields.get(4)?.parse().ok())
                .flatten()
        })
    };
    let mut workers = Vec::new();
    let mut inventory_available = true;
    match std::fs::read_dir("/proc") {
        Ok(entries) => {
            for entry in entries.flatten() {
                let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else {
                    continue;
                };
                let cgroup =
                    std::fs::read_to_string(entry.path().join("cgroup")).unwrap_or_default();
                if !cgroup.contains("jope-experiment-") {
                    continue;
                }
                let command = std::fs::read(entry.path().join("cmdline")).unwrap_or_default();
                if command.is_empty() {
                    continue;
                }
                let gpu = experimental_gpu_rank(&command);
                workers.push(serde_json::json!({"pid":pid,"gpu":gpu,"cgroup":cgroup}));
            }
        }
        Err(_) => inventory_available = false,
    }
    let gpu_owner = lock_owner("encoding-service.lock");
    let experimental_gpu_owner = gpu_owner.is_some_and(|pid| {
        std::fs::read_to_string(format!("/proc/{pid}/cgroup"))
            .unwrap_or_default()
            .contains("jope-experiment-")
    });
    serde_json::json!({"timestamp":now(),"inventory_available":inventory_available,"locks_available":locks.is_ok(),
        "cpu_lock_owner":lock_owner("experiment-spare-cpus.lock"),"gpu_lock_owner":gpu_owner,
        "experimental_gpu_owner":experimental_gpu_owner,"gpu_workers":workers.iter().filter(|p|p["gpu"]==true).count(),"workers":workers})
}

// Read current workers as well as counters: lifetime memory.high events alone
// cannot establish that a campaign is blocked now.
pub(crate) fn campaign_resource_evidence(root: &Path) -> Value {
    let directory = Path::new("/sys/fs/cgroup/system.slice/jope-parallel-campaign.service");
    let mut result = serde_json::json!({"timestamp":now(),"available":false});
    let Ok(pids) = std::fs::read_to_string(directory.join("cgroup.procs")) else {
        return result;
    };
    let mut workers = Vec::new();
    let mut matched = false;
    let mut available = true;
    for pid in pids
        .split_whitespace()
        .filter_map(|p| p.parse::<u32>().ok())
    {
        let Ok(command) = std::fs::read(format!("/proc/{pid}/cmdline")) else {
            available = false;
            continue;
        };
        let command = String::from_utf8_lossy(&command).replace('\0', " ");
        if !command.contains(root.to_string_lossy().as_ref()) {
            continue;
        }
        matched = true;
        if !command.split_whitespace().any(|v| v == "train-rank") {
            continue;
        }
        let wait = std::fs::read_to_string(format!("/proc/{pid}/wchan"));
        if wait.is_err() {
            available = false;
        }
        workers.push(serde_json::json!({"pid":pid,"wait":wait.unwrap_or_default().trim()}));
    }
    result["available"] = (available && matched).into();
    for (file, key) in [
        ("memory.current", "memory_current"),
        ("memory.high", "memory_high"),
        ("memory.max", "memory_max"),
    ] {
        result[key] = std::fs::read_to_string(directory.join(file))
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(Value::from)
            .unwrap_or(Value::Null);
    }
    result["training_workers"] = workers.len().into();
    result["throttled_workers"] = workers
        .iter()
        .filter(|v| {
            v["wait"]
                .as_str()
                .is_some_and(|s| s.contains("mem_cgroup_handle_over_high"))
        })
        .count()
        .into();
    result["workers"] = workers.into();
    result
}

pub(crate) fn memory_throttle_reason(run: &Value) -> Option<String> {
    let evidence = &run["campaign_resource_evidence"];
    if number(evidence, "throttled_workers") == 0 {
        return None;
    }
    if evidence["available"] != true
        || number(evidence, "timestamp") == 0
        || number(evidence, "timestamp") > now()
        || now().saturating_sub(number(evidence, "timestamp")) > 5
        || !evidence["memory_current"].is_u64()
        || number(evidence, "memory_high") == 0
    {
        return Some(
            "Training resource state uncertain: memory-throttle evidence is incomplete or stale."
                .into(),
        );
    }
    Some(format!("Host-memory throttle: {} / {} training workers blocked; {:.1} GiB used, {:.1} GiB soft limit. GPU VRAM is separate.",
        number(evidence,"throttled_workers"), number(evidence,"training_workers"),
        number(evidence,"memory_current") as f64 / 1024f64.powi(3),
        number(evidence,"memory_high") as f64 / 1024f64.powi(3)))
}

pub(crate) fn panel(title: &str) -> Block<'_> {
    Block::bordered()
        .title(title)
        .border_type(BorderType::Rounded)
        .title_style(Style::default().fg(BLUE).bold())
        .border_style(Style::default().fg(Color::Rgb(50, 74, 98)))
}
pub(crate) fn tail(path: &Path, bytes: u64) -> String {
    (|| -> std::io::Result<String> {
        let mut file = std::fs::File::open(path)?;
        let start = file.metadata()?.len().saturating_sub(bytes);
        file.seek(SeekFrom::Start(start))?;
        let mut text = String::new();
        file.read_to_string(&mut text)?;
        if start > 0 {
            text = text
                .split_once('\n')
                .map(|(_, v)| v.to_owned())
                .unwrap_or_default();
        }
        Ok(text)
    })()
    .unwrap_or_default()
}
pub(crate) fn metrics(path: &Path) -> Vec<Value> {
    metric_tail(path, 8 * 1024 * 1024)
}
pub(crate) fn metric_tail(path: &Path, bytes: u64) -> Vec<Value> {
    tail(path, bytes)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|v| v["rank"].as_u64().unwrap_or(0) == 0 && v["invalidated"] != true)
        .collect()
}
pub(crate) fn policy_receipt(v: &Value) -> bool {
    v["invalidated"] != true
        && v["corpus_sha256"].is_string()
        && !v["run"].as_str().is_some_and(|path| {
            Path::new(path)
                .components()
                .any(|part| part.as_os_str() == "parity")
        })
}
pub(crate) fn duration(seconds: u64) -> String {
    if seconds >= 3600 {
        format!("{}h {:02}m", seconds / 3600, (seconds % 3600) / 60)
    } else if seconds >= 60 {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}
pub(crate) fn state(path: &Path) -> Value {
    read_json(path).unwrap_or_default()
}
pub(crate) fn number(v: &Value, key: &str) -> u64 {
    v[key].as_u64().unwrap_or(0)
}
pub(crate) fn decimal(v: &Value, key: &str) -> String {
    v[key]
        .as_f64()
        .map(|n| format!("{n:.4}"))
        .unwrap_or_else(|| "—".into())
}
pub(crate) fn rmse(v: &Value, key: &str) -> String {
    v[key]
        .as_f64()
        .filter(|x| x.is_finite())
        .map(|x| {
            if x != 0. && x.abs() < 0.001 {
                format!("{x:.3e}")
            } else {
                format!("{x:.4}")
            }
        })
        .unwrap_or_else(|| "—".into())
}
pub(crate) fn phase(v: &Value) -> &str {
    v["phase"].as_str().unwrap_or("waiting for work")
}
pub(crate) fn age(v: &Value) -> String {
    age_with_limit(v, 5)
}
pub(crate) fn age_with_limit(v: &Value, stale_after: u64) -> String {
    let t = number(v, "timestamp");
    if t == 0 {
        "unavailable".into()
    } else {
        let elapsed = now().saturating_sub(t);
        if elapsed > stale_after {
            format!("stale · {elapsed}s ago")
        } else {
            "live".into()
        }
    }
}
pub(crate) fn completed(v: &Value) -> bool {
    v["invalidated"] != true
        && (phase(v).starts_with("completed") || v["complete"] == true || v["status"] == "complete")
}
pub(crate) fn status_age(v: &Value) -> String {
    status_age_with_limit(v, 5)
}
pub(crate) fn status_age_with_limit(v: &Value, stale_after: u64) -> String {
    if v["invalidated"] == true {
        "INVALIDATED · rerun required".into()
    } else if completed(v) {
        "completed · saved receipt".into()
    } else {
        age_with_limit(v, stale_after)
    }
}
pub(crate) fn count_pair(done: u64, total: u64) -> String {
    if total == 0 {
        "awaiting manifest".into()
    } else {
        format!("{done} / {total}")
    }
}
/// Units `live_operations` asks systemd about. `systemctl show` also answers for a unit
/// that is not installed (`ActiveState=inactive`, `SubState=dead`, exit status 0), so a
/// name with no installable unit reads as a stopped service, never as an absent one.
/// Each name must therefore have a tracked template under `deploy/systemd/engine/`.
/// `jope-parallel-console` was dropped when the Python removal P1 (ADR-0006) archived its
/// template, which ran the archived `tui-session.py`; no view read its row.
pub(crate) const OBSERVED_SERVICES: [&str; 2] = ["jope-parallel-campaign", "jope-parallel-gp"];
/// Read durable campaign receipts, without traversing candidate directories or opening data.
/// Called every five seconds; hardware and worker progress retain their one-second refresh.
pub(crate) fn live_operations(root: &Path) -> Value {
    let mut services = serde_json::Map::new();
    for name in OBSERVED_SERVICES {
        let output = std::process::Command::new("timeout")
            .args([
                "2",
                "systemctl",
                "show",
                name,
                "-p",
                "ActiveState",
                "-p",
                "SubState",
                "-p",
                "MainPID",
                "-p",
                "Result",
            ])
            .output();
        let mut row = serde_json::Map::new();
        if let Ok(output) = output {
            if output.status.success() {
                for line in String::from_utf8_lossy(&output.stdout).lines() {
                    if let Some((key, value)) = line.split_once('=') {
                        row.insert(key.into(), value.into());
                    }
                }
            }
        }
        services.insert(name.into(), row.into());
    }
    let owners = std::process::Command::new("timeout")
        .args([
            "2",
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        });
    let gpu_units = owners
        .iter()
        .flatten()
        .filter_map(|line| line.split(',').next())
        .filter_map(|pid| pid.trim().parse::<u32>().ok())
        .filter_map(|pid| std::fs::read_to_string(format!("/proc/{pid}/cgroup")).ok())
        .flat_map(|groups| {
            groups
                .split('/')
                .map(str::trim)
                .filter(|part| part.ends_with(".service"))
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .collect::<BTreeSet<_>>();
    let gp = crate::telemetry::observed_status(&state(&root.join("runs/gp-service.json")), now());
    let gp_units = gp["active_rows"]
        .as_array()
        .into_iter()
        .flatten()
        .take(1)
        .filter(|_| now().saturating_sub(number(&gp, "timestamp")) <= 15)
        .filter_map(|row| row["pid"].as_u64())
        .filter_map(|pid| std::fs::read_to_string(format!("/proc/{pid}/cgroup")).ok())
        .flat_map(|groups| {
            groups
                .split('/')
                .map(str::trim)
                .filter(|part| part.ends_with(".service"))
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .collect::<BTreeSet<_>>();
    serde_json::json!({"timestamp":now(),"services":services,"gpu_owners":owners,"gpu_units":gpu_units,"gp_units":gp_units})
}

#[cfg(test)]
mod observed_service_tests {
    use super::OBSERVED_SERVICES;
    use std::path::{Path, PathBuf};

    fn engine_unit_templates() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../deploy/systemd/engine")
    }

    /// The names in `names` with no `<name>.service.in` under `templates`.
    fn untracked<'a>(names: &[&'a str], templates: &Path) -> Vec<&'a str> {
        names
            .iter()
            .copied()
            .filter(|name| !templates.join(format!("{name}.service.in")).is_file())
            .collect()
    }

    #[test]
    fn every_observed_service_has_a_tracked_unit_template() {
        let templates = engine_unit_templates();
        assert!(
            templates.is_dir(),
            "no engine unit templates at {}",
            templates.display()
        );
        assert_eq!(
            untracked(&OBSERVED_SERVICES, &templates),
            Vec::<&str>::new(),
            "the dashboard observes services with no tracked unit template under {}",
            templates.display()
        );
        assert_eq!(
            untracked(&["jope-parallel-no-such-unit"], &templates),
            vec!["jope-parallel-no-such-unit"],
            "an observed name without a template must be reported"
        );
    }
}

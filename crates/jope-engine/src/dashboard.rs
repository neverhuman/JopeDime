#[path = "dashboard_gp.rs"]
mod library_gp;
use crate::{
    manifest::read_json,
    telemetry::{now, Hardware},
};
use anyhow::{ensure, Context, Result};
use ratatui::{prelude::*, widgets::*};
use serde_json::Value;
use std::{
    collections::{BTreeSet, VecDeque},
    io::{Read, Seek, SeekFrom},
    path::Path,
    time::{Duration, Instant},
};

const INK: Color = Color::Rgb(218, 229, 242);
const MUTED: Color = Color::Rgb(126, 149, 174);
const BLUE: Color = Color::Rgb(92, 205, 255);
const GREEN: Color = Color::Rgb(99, 224, 174);
const GOLD: Color = Color::Rgb(250, 197, 101);
const VIOLET: Color = Color::Rgb(181, 159, 255);
const RED: Color = Color::Rgb(255, 116, 135);

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

fn experimental_gpu_rank(command: &[u8]) -> bool {
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
mod worker_identity_tests {
    use super::experimental_gpu_rank;

    #[test]
    fn recognizes_batch_workers_without_a_world_flag() {
        assert!(experimental_gpu_rank(
            b"jope-parallel\0qualify-batch-rank\0--rank\x000\0"
        ));
        assert!(experimental_gpu_rank(
            b"jope-parallel\0train-rank\0--rank\x007\0--world\08\0"
        ));
        assert!(experimental_gpu_rank(
            b"python3\0encode.py\0--rank\x000\0--world-size\08\0"
        ));
        assert!(!experimental_gpu_rank(b"python3\0batch-controller.py\0"));
        assert!(!experimental_gpu_rank(
            b"jope-parallel\0qualify-batch-rank\0--rank\0invalid\0"
        ));
        assert!(!experimental_gpu_rank(
            b"python3\0/tmp/qualify-batch-rank\0--rank\x000\0"
        ));
        assert!(!experimental_gpu_rank(
            b"python3\0script.py\0--rank\x000\0--world\x000\0"
        ));
    }
}

fn experiment_worker_evidence(root: &Path) -> Value {
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
fn campaign_resource_evidence(root: &Path) -> Value {
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

fn memory_throttle_reason(run: &Value) -> Option<String> {
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

fn panel(title: &str) -> Block<'_> {
    Block::bordered()
        .title(title)
        .border_type(BorderType::Rounded)
        .title_style(Style::default().fg(BLUE).bold())
        .border_style(Style::default().fg(Color::Rgb(50, 74, 98)))
}
fn tail(path: &Path, bytes: u64) -> String {
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
fn metrics(path: &Path) -> Vec<Value> {
    metric_tail(path, 8 * 1024 * 1024)
}
fn metric_tail(path: &Path, bytes: u64) -> Vec<Value> {
    tail(path, bytes)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|v| v["rank"].as_u64().unwrap_or(0) == 0 && v["invalidated"] != true)
        .collect()
}
fn policy_receipt(v: &Value) -> bool {
    v["invalidated"] != true
        && v["corpus_sha256"].is_string()
        && !v["run"].as_str().is_some_and(|path| {
            Path::new(path)
                .components()
                .any(|part| part.as_os_str() == "parity")
        })
}
fn duration(seconds: u64) -> String {
    if seconds >= 3600 {
        format!("{}h {:02}m", seconds / 3600, (seconds % 3600) / 60)
    } else if seconds >= 60 {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}
fn state(path: &Path) -> Value {
    read_json(path).unwrap_or_default()
}
fn number(v: &Value, key: &str) -> u64 {
    v[key].as_u64().unwrap_or(0)
}
fn decimal(v: &Value, key: &str) -> String {
    v[key]
        .as_f64()
        .map(|n| format!("{n:.4}"))
        .unwrap_or_else(|| "—".into())
}
fn rmse(v: &Value, key: &str) -> String {
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
fn phase(v: &Value) -> &str {
    v["phase"].as_str().unwrap_or("waiting for work")
}
fn age(v: &Value) -> String {
    age_with_limit(v, 5)
}
fn age_with_limit(v: &Value, stale_after: u64) -> String {
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
fn completed(v: &Value) -> bool {
    v["invalidated"] != true
        && (phase(v).starts_with("completed") || v["complete"] == true || v["status"] == "complete")
}
fn status_age(v: &Value) -> String {
    status_age_with_limit(v, 5)
}
fn status_age_with_limit(v: &Value, stale_after: u64) -> String {
    if v["invalidated"] == true {
        "INVALIDATED · rerun required".into()
    } else if completed(v) {
        "completed · saved receipt".into()
    } else {
        age_with_limit(v, stale_after)
    }
}
fn count_pair(done: u64, total: u64) -> String {
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
const OBSERVED_SERVICES: [&str; 2] = ["jope-parallel-campaign", "jope-parallel-gp"];
/// Read durable campaign receipts, without traversing candidate directories or opening data.
/// Called every five seconds; hardware and worker progress retain their one-second refresh.
fn live_operations(root: &Path) -> Value {
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

pub fn campaign(root: &Path) -> Value {
    let mut run = state(&root.join("runs/status.json"));
    if !run.is_object() {
        run = serde_json::json!({});
    }
    let index_path = run["contexts_index"]
        .as_str()
        .unwrap_or("data/encodings/contexts-v1/index.json");
    let index = state(&root.join(index_path));
    if let Some(encodings) = index["encodings"].as_array() {
        let families: BTreeSet<_> = encodings
            .iter()
            .filter_map(|v| v["family"].as_str())
            .collect();
        run["contexts"] = serde_json::json!({"completed":encodings.len(),"total":encodings.len(),"families":families.len(),"dimensions":number(&index,"dimensions"),"qualification":index["qualification"],"source":index_path});
    }
    if let Some(manifests) = run["prepared_job_manifests"].as_array() {
        let mut views = BTreeSet::new();
        let mut valid = !manifests.is_empty();
        for name in manifests {
            let jobs = name
                .as_str()
                .map(|name| state(&root.join(name)))
                .unwrap_or_default();
            if let Some(rows) = jobs["jobs"].as_array() {
                for row in rows {
                    if let (Some(prepared), Some(cut)) =
                        (row["prepared"].as_str(), row["cut"].as_u64())
                    {
                        if let Some(dataset) = Path::new(prepared).file_name() {
                            views.insert((dataset.to_string_lossy().into_owned(), cut));
                        } else {
                            valid = false;
                        }
                    } else {
                        valid = false;
                    }
                }
            } else {
                valid = false;
            }
        }
        if valid {
            let datasets = views
                .iter()
                .map(|(dataset, _)| dataset)
                .collect::<BTreeSet<_>>()
                .len();
            run["preparation"] = serde_json::json!({"views":views.len(),"datasets":datasets});
            run["contexts"]["total"] = views.len().into();
        }
    }
    run["cycle"] = state(&root.join("runs/cycle-progress.json"));
    run["operations"] = live_operations(root);
    run["continuous"] = state(&root.join("runs/continuous-progress.json"));
    if let Some(path) = run["continuous"]["packing_progress"].as_str() {
        run["continuous_packing"] = state(&root.join(path));
    }
    run["frozen_monitoring"] = state(&root.join("runs/frozen-monitoring.json"));
    run["encoding_pipeline"] = state(&root.join("runs/encoding-pipeline.json"));
    run["campaign_service"] = state(&root.join("runs/campaign-progress.json"));
    run["bootstrap"] = state(&root.join("runs/cohort64-bootstrap-readiness.json"));
    run["corpus_progress"] = state(&root.join("runs/corpus-progress.json"));
    run["runtime_readiness"] = state(&root.join("runs/runtime-readiness.json"));
    // Nothing in this workspace writes the next three receipts. Their writers,
    // `watch_campaign_reports.py`, `opportunistic_proposals.py` and
    // `opportunistic_proposals_v2.py`, were archived by the Python removal P1 (ADR-0006).
    // A receipt left in the state root is history and keeps its saved status; without
    // one, no row is shown.
    run["report_progress"] = state(&root.join("runs/report-progress.json"));
    run["proposal_library"] = state(&root.join("runs/proposal-library.json"));
    run["pure_proposal_library"] = state(&root.join("runs/pure-proposal-library.json"));
    run["seed_experiment"] = state(&root.join("runs/seed-quality-experiment-v1/progress.json"));
    run["seed_quality"] = state(&root.join("runs/seed-quality-experiment-v1/report.json"));
    run["binary_qualification"] = state(&root.join("runs/binary-qualification-v1/progress.json"));
    run["proposal_library_admission"] = state(&root.join("runs/proposal-library-admission.json"));
    if let Some(path) = run["proposal_library_admission"]["receipt"].as_str() {
        let path = root.join(path);
        let receipt = state(&path);
        if receipt["outcome"] == "admitted" {
            run["proposal_library_admission"]["committed"] = true.into();
            run["proposal_library_admission"]["requested_jobs"] = receipt["requested_jobs"].clone();
            run["proposal_library_gp"] = library_gp::progress(root, &path, &receipt);
        }
    }
    run["auxiliary"] = state(&root.join("runs/auxiliary-training.json"));
    if !run["auxiliary"]["maximum_steps"].is_u64() {
        if let Some(output) = run["auxiliary"]["output"].as_str() {
            let spec = state(&root.join(output).join("spec.json"));
            if spec["options"]["max_steps"].is_u64() {
                run["auxiliary"]["maximum_steps"] = spec["options"]["max_steps"].clone();
            }
        }
    }
    run["teacher"] = state(&root.join("runs/teacher-progress.json"));
    run["policy"] = state(&root.join("runs/neural-service.json"));
    run["latest_neural_activity"] = run["policy"].clone();
    run["hyperion_recovery"] =
        state(&root.join("runs/parity/hyperion-reviewed4-recovery-parity.json"));
    let mut cycles = BTreeSet::new();
    for entry in std::fs::read_dir(root.join("runs"))
        .into_iter()
        .flatten()
        .flatten()
    {
        let path = entry.path();
        if path.join("cycle.json").is_file() {
            cycles.insert(path);
        }
    }
    if let Some(path) = run["campaign_service"]["current_cycle"].as_str() {
        let current = root.join(path);
        if let Some(parent) = current.parent() {
            for row in state(&parent.join("completed-cycles.json"))["cycles"]
                .as_array()
                .into_iter()
                .flatten()
            {
                if let Some(path) = row["cycle"].as_str() {
                    cycles.insert(root.join(path));
                }
            }
        }
        cycles.insert(current);
    }
    // Display chronology comes from durable receipt times, never directory spelling.
    let current_cycle = run["campaign_service"]["current_cycle"]
        .as_str()
        .map(|p| root.join(p));
    let mut cycles = cycles
        .into_iter()
        .map(|path| {
            let complete = state(&path.join("complete.json"));
            let started = state(&path.join("cycle.json"));
            let timestamp = [
                number(&complete, "completed_at"),
                number(&complete, "timestamp"),
                number(&started, "timestamp"),
                number(&started, "started_at"),
            ]
            .into_iter()
            .max()
            .unwrap_or(0);
            let current = current_cycle.as_ref() == Some(&path);
            (current, timestamp, path)
        })
        .collect::<Vec<_>>();
    cycles.sort();
    let mut reports = Vec::new();
    let mut checkpoints = Vec::new();
    for (_, _, cycle) in cycles.iter().rev().take(16).rev() {
        let report = state(&cycle.join("report-v1/report.json"));
        if report["dl_gp_dl_complete"] == true {
            reports.push(serde_json::json!({"cycle":cycle.file_name().unwrap_or_default().to_string_lossy(),"path":cycle.join("report-v1/report.json"),"records":report["teaching_records"],"predictors":report["fitted_predictors_passed"],"views":report["comparison_views"],"searches":report["comparison_searches"],"validation":report["partitions"]["meta_validation"],"generation":report["generation"],"budget":report["matched_evaluation_budget"],"clock_coverage":report["clock_coverage"]}));
        } else {
            let raw = state(&cycle.join("comparisons.json"));
            let complete = state(&cycle.join("complete.json"));
            if raw["comparisons"].is_array() {
                reports.push(serde_json::json!({"cycle":cycle.file_name().unwrap_or_default().to_string_lossy(),"path":cycle.join("comparisons.json"),"raw_final_scores_only":true,"comparisons":raw["comparisons"],"views":raw["comparisons"].as_array().map_or(0,Vec::len),"complete":complete["dl_gp_dl_complete"],"predictors":complete["predictor_exports_passed"]}));
            }
        }
        for block in ["train-1", "train-2"] {
            let path = cycle.join(block);
            let latest = state(&path.join("latest.json"));
            if latest["checkpoint"].is_string() && !path.join("invalidated.json").exists() {
                let samples = metric_tail(&path.join("metrics.jsonl"), 16 * 1024);
                let last = samples.last().cloned().unwrap_or_default();
                checkpoints.push(serde_json::json!({"name":format!("{} / {block}",cycle.file_name().unwrap_or_default().to_string_lossy()),"path":path,"checkpoint":latest["checkpoint"],"step":latest["step"],"complete":path.join("complete.json").is_file(),"metrics":last,"initialization":state(&path.join("initialization.json"))}));
            }
        }
    }
    if let Some(path) = run["policy"]["run"].as_str().map(|p| root.join(p)) {
        if !checkpoints
            .iter()
            .any(|v| v["path"].as_str() == path.to_str())
            && !path.join("invalidated.json").is_file()
        {
            let latest = state(&path.join("latest.json"));
            if latest["checkpoint"].is_string() && policy_receipt(&run["policy"]) {
                checkpoints.push(serde_json::json!({"name":path.file_name().unwrap_or_default().to_string_lossy(),"path":path,"checkpoint":latest["checkpoint"],"step":latest["step"],"complete":path.join("complete.json").is_file(),"metrics":run["policy"],"initialization":state(&path.join("initialization.json"))}));
            }
        }
    }
    // Standalone real-corpus training remains visible while a diagnostic worker
    // temporarily owns the global neural-service receipt.
    for entry in std::fs::read_dir(root.join("runs"))
        .into_iter()
        .flatten()
        .flatten()
    {
        let path = entry.path();
        if path.join("invalidated.json").is_file()
            || !path.join("latest.json").is_file()
            || checkpoints
                .iter()
                .any(|v| v["path"].as_str() == path.to_str())
        {
            continue;
        }
        let latest = state(&path.join("latest.json"));
        let samples = metric_tail(&path.join("metrics.jsonl"), 16 * 1024);
        if let Some(last) = samples.last().filter(|v| policy_receipt(v)) {
            if latest["checkpoint"].is_string() {
                checkpoints.push(serde_json::json!({"name":path.file_name().unwrap_or_default().to_string_lossy(),"path":path,"checkpoint":latest["checkpoint"],"step":latest["step"],"complete":path.join("complete.json").is_file(),"metrics":last,"initialization":state(&path.join("initialization.json"))}));
            }
        }
    }
    checkpoints.sort_by_key(|v| number(&v["metrics"], "timestamp"));
    run["reports"] = reports.into();
    run["checkpoints"] = checkpoints.into();
    if !policy_receipt(&run["policy"]) {
        if let Some(last) = run["checkpoints"]
            .as_array()
            .and_then(|v| v.last())
            .cloned()
        {
            if policy_receipt(&last["metrics"]) {
                run["policy"] = last["metrics"].clone();
                run["policy"]["checkpoint"] = last["checkpoint"].clone();
                if last["complete"] == true {
                    run["policy"]["status"] = "complete".into();
                }
            }
        }
    }
    if let Some(path) = run["policy"]["run"].as_str() {
        run["learning_curve"] = metrics(&root.join(path).join("metrics.jsonl")).into();
    }
    if let Some(path) = run["family_registry"].as_str() {
        if let Ok(registry) = read_json::<crate::families::Registry>(&root.join(path)) {
            run["family_readiness"] = registry.quota_report();
            run["dataset_names"] = registry
                .families
                .values()
                .flat_map(|family| {
                    family.datasets.iter().map(move |dataset| {
                        (
                            dataset.dataset.clone(),
                            Value::String(format!(
                                "{} / {}",
                                family.family.trim_start_matches("rdatasets-"),
                                dataset.dataset.chars().take(4).collect::<String>()
                            )),
                        )
                    })
                })
                .collect::<serde_json::Map<_, _>>()
                .into();
        }
    }
    run["gpu_wait"] = state(&root.join("runs/gpu-wait.json"));
    // Keep the campaign's completed policy visible while a newer encoding or
    // recovery-check receipt owns the global live neural panel.
    let active_cycle = std::fs::read_dir(root.join("runs"))
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path();
            let modified = path.join("cycle.json").metadata().ok()?.modified().ok()?;
            Some((modified, path))
        })
        .max_by_key(|(modified, _)| *modified);
    let active_path = run["campaign_service"]["current_cycle"]
        .as_str()
        .map(|p| root.join(p))
        .filter(|p| p.join("cycle.json").is_file())
        .or_else(|| active_cycle.map(|(_, p)| p));
    if let Some(path) = active_path {
        run["active_cycle"] = path.to_string_lossy().as_ref().into();
        let exports = state(&path.join("predictor-exports.json"));
        run["predictor_exports"] = exports.clone();
        let mut selected = state(&path.join("train-1/latest.json"));
        let best = state(&path.join("auxiliary/train/best.json"));
        if best["checkpoint"].is_string() {
            selected = best;
        }
        if path.join("train-2/complete.json").is_file() {
            selected = state(&path.join("train-2/latest.json"));
        }
        if let Some(checkpoint) = selected["checkpoint"].as_str() {
            let checkpoint_state = state(&root.join(checkpoint).join("state.json"));
            selected["corpus_sha256"] = checkpoint_state["corpus_sha256"].clone();
            selected["model_sha256"] = checkpoint_state["model_sha256"].clone();
            run["selected_checkpoint"] = selected;
        }
        let cycle_gp = &run["cycle"]["neural_seeded_gp"];
        if exports["completed"].is_number()
            && exports["total"].is_number()
            && number(&exports, "completed") < number(&exports, "total")
            && number(cycle_gp, "total") > 0
            && number(cycle_gp, "completed") == number(cycle_gp, "total")
        {
            run["cycle"]["phase"] = "Fit and save GP predictors".into();
            run["cycle"]["reason"] = format!(
                "{} / {} fitted predictor exports saved; each fit has a 30-second and 8 GiB limit.",
                number(&exports, "completed"),
                number(&exports, "total")
            )
            .into();
        }
        let live_train = state(&root.join("runs/neural-service.json"));
        if live_train["run"]
            .as_str()
            .is_some_and(|p| root.join(p) == path.join("train-2"))
        {
            run["cycle"]["retraining"]["step"] = live_train["step"].clone();
        }
        for block in ["train-1", "train-2"] {
            let receipt = state(&path.join(block).join("latest.json"));
            if path.join(block).join("complete.json").exists() && receipt["step"].is_number() {
                let steps = number(&receipt, "step");
                let policy = serde_json::json!({"phase":"completed policy training block","step":steps,"steps":steps,"world":8,"checkpoint":receipt["checkpoint"]});
                if block == "train-1" {
                    run["fresh_policy"] = policy;
                }
                run["checkpoint"] = receipt;
            }
        }
    }
    let latest = std::fs::read_dir(root.join("runs"))
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path().join("latest.json");
            let modified = path.metadata().ok()?.modified().ok()?;
            let receipt = state(&path);
            receipt["checkpoint"].as_str()?;
            Some((modified, receipt))
        })
        .max_by_key(|(modified, _)| *modified);
    if run["checkpoint"].is_null() {
        if let Some((_, receipt)) = latest {
            run["checkpoint"] = receipt;
        }
    }
    // The bootstrap progress receipt does not change as later corpora grow.
    // Read only the index header after a complete cycle binds its corpus hash.
    if let Some(active) = run["active_cycle"].as_str() {
        if let Some(parent) = Path::new(active).parent() {
            for entry in std::fs::read_dir(parent).into_iter().flatten().flatten() {
                let cycle = entry.path();
                let complete = state(&cycle.join("complete.json"));
                if run["selected_checkpoint"]["corpus_sha256"].is_string()
                    && complete["teaching_corpus_sha256"]
                        == run["selected_checkpoint"]["corpus_sha256"]
                {
                    let mut bytes = vec![0; 2048];
                    if let Ok(mut file) = std::fs::File::open(cycle.join("corpus-2/index.json")) {
                        if let Ok(n) = file.read(&mut bytes) {
                            let header = String::from_utf8_lossy(&bytes[..n]);
                            if let Some((_, suffix)) = header.split_once("\"records\":") {
                                let digits = suffix
                                    .trim_start()
                                    .chars()
                                    .take_while(char::is_ascii_digit)
                                    .collect::<String>();
                                if let Ok(records) = digits.parse::<u64>() {
                                    run["corpus_progress"]["records"] = records.into();
                                    run["corpus_progress"]["output"] =
                                        cycle.join("corpus-2").to_string_lossy().as_ref().into();
                                    run["corpus_progress"]["reason"] =
                                        "Current selected checkpoint corpus; saved index header"
                                            .into();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    run
}
fn goal_phase<'a>(run: &'a Value, neural: &'a Value, gp: &'a Value) -> &'a str {
    if run["continuous"].is_object() {
        if run["operations"]["services"]["jope-parallel-campaign"]["ActiveState"] != "active"
            || now().saturating_sub(number(&run["continuous"], "timestamp")) > 15
        {
            return "Continuous controller unavailable · inspect saved evidence";
        }
        if run["continuous"]["phase"] == "paused" {
            return "Continuous learning paused";
        }
        if completed(neural) {
            return "Continuous learning · scheduling fresh GP targets";
        }
    }
    let policy = if run["policy"].is_object() {
        &run["policy"]
    } else {
        neural
    };
    if training_hold(run, policy).is_some() {
        return "Automatic training held · inspect task state";
    }
    if number(neural, "timestamp") > 0
        && !completed(neural)
        && now().saturating_sub(number(neural, "timestamp")) <= 15
    {
        return phase(neural);
    }
    if !completed(&run["auxiliary"])
        && number(&run["auxiliary"], "timestamp") > 0
        && now().saturating_sub(number(&run["auxiliary"], "timestamp")) <= 15
    {
        return phase(&run["auxiliary"]);
    }
    let pure = &run["pure_proposal_library"];
    if phase(pure) == "running"
        && number(pure, "timestamp") > 0
        && now().saturating_sub(number(pure, "timestamp")) <= 15
    {
        return "Saving pure policy predictions on eight GPUs";
    }
    let library = &run["proposal_library"];
    if phase(library) == "running"
        && number(library, "timestamp") > 0
        && now().saturating_sub(number(library, "timestamp")) <= 15
    {
        return "Building saved proposal seed library on eight GPUs";
    }
    let pipeline = &run["encoding_pipeline"];
    if !completed(pipeline)
        && number(pipeline, "timestamp") > 0
        && now().saturating_sub(number(pipeline, "timestamp")) <= 15
    {
        return phase(pipeline);
    }
    if !completed(&run["cycle"]) {
        if let Some(value) = run["cycle"]["phase"].as_str() {
            return value;
        }
    }
    if number(gp, "active") > 0 && now().saturating_sub(number(gp, "timestamp")) <= 5 {
        return "GP teaching corpus growing";
    }
    if run["contexts"]["completed"].is_number() {
        return "Encoded corpus ready · next learning cycle";
    }
    phase(run)
}
fn gpu_reason(run: &Value, neural: &Value, hardware: &Hardware) -> String {
    if hardware.gpus.len() != 8 {
        return "GPU telemetry incomplete; inspect Hardware for missing devices.".into();
    }
    if let Some(reason) = memory_throttle_reason(run) {
        if number(&run["campaign_resource_evidence"], "throttled_workers") > 0
            && (hardware.timestamp == 0
                || hardware.timestamp > now()
                || now().saturating_sub(hardware.timestamp) > 5)
        {
            return "GPU activity unknown: current host-memory contention observed, but GPU telemetry is stale.".into();
        }
        return reason;
    }
    let evidence = &run["experiment_worker_evidence"];
    if evidence.is_object() {
        if evidence["inventory_available"] != true
            || evidence["locks_available"] != true
            || now().saturating_sub(number(evidence, "timestamp")) > 5
        {
            return "GPU ownership unknown: worker or scheduler evidence is unavailable or stale."
                .into();
        }
        if number(evidence, "gpu_workers") > 0 {
            if evidence["experimental_gpu_owner"] != true {
                return "GPU ownership uncertain: experimental workers and scheduler lease disagree.".into();
            }
            return format!("{} experimental GPU ranks observed with the scheduler lease; production handoff takes priority.",number(evidence,"gpu_workers"));
        }
        for key in ["seed_experiment", "binary_qualification"] {
            let receipt = &run[key];
            if number(receipt, "gpu_world") > 0
                && !completed(receipt)
                && now().saturating_sub(number(receipt, "timestamp")) <= 15
            {
                return "GPU activity uncertain: a recent experiment receipt has no matching live GPU ranks.".into();
            }
        }
    }
    let policy = if run["policy"].is_object() {
        &run["policy"]
    } else {
        neural
    };
    if let Some(reason) = training_hold(run, policy) {
        return format!("GPU training held: {reason}; existing tasks require reconciliation.");
    }
    if !completed(neural) && now().saturating_sub(number(neural, "timestamp")) <= 5 {
        return format!("GPU assignment: {}", phase(neural));
    }
    for key in ["binary_qualification", "seed_experiment"] {
        let work = &run[key];
        if work.is_object()
            && number(work, "timestamp") > 0
            && now().saturating_sub(number(work, "timestamp")) <= 15
        {
            return format!(
                "Qualification {}: {} · CPUs 0–7, 62–69 shared by encoding and comparison",
                phase(work),
                work["reason"]
                    .as_str()
                    .unwrap_or("bounded spare-resource window")
            );
        }
    }
    if number(&run["gpu_wait"], "timestamp") > 0
        && now().saturating_sub(number(&run["gpu_wait"], "timestamp")) <= 5
    {
        return format!(
            "Queued GPU work: {} · waiting for current block to finish",
            phase(&run["gpu_wait"])
        );
    }
    let pipeline = &run["encoding_pipeline"];
    if pipeline["phase"].is_string() && !completed(pipeline) {
        return format!(
            "Next encoding work: {} · {} · {}",
            phase(pipeline),
            pipeline["reason"].as_str().unwrap_or("see Data / Events"),
            age(pipeline)
        );
    }
    if completed(&run["auxiliary"])
        && run["cycle"]["phase"] == "GP optimizes cut targets for the next teaching corpus"
    {
        return format!(
            "Auxiliary learning finished; CPU GP still evaluating frozen proposals. {}",
            run["auxiliary"]["reason"]
                .as_str()
                .unwrap_or("Next mandatory GPU block follows GP and corpus packing.")
        );
    }
    if run["cycle"]["phase"]
        .as_str()
        .is_some_and(|p| p.starts_with("Pack ") || p.starts_with("Fit and save"))
    {
        let cycle = &run["cycle"];
        if number(cycle, "timestamp") == 0
            || number(cycle, "timestamp") > now()
            || now().saturating_sub(number(cycle, "timestamp")) > 5
        {
            return format!(
                "GPU wait reason unconfirmed: stale {} receipt.",
                phase(cycle)
            );
        }
        return format!(
            "GPUs waiting for {}: {}",
            phase(cycle),
            stage_work_detail(cycle)
        );
    }
    let pure = &run["pure_proposal_library"];
    if pure.is_object() {
        return format!("Pure policy predictions {}: {} · {} / {} pools saved · separate scope, no GP queue writes · {}",phase(pure),pure["reason"].as_str().unwrap_or("saved training-context predictions await GP admission"),decimal_count(pure,"saved_seed_pools"),decimal_count(pure,"planned_seed_pools"),status_age_with_limit(pure,15));
    }
    let library = &run["proposal_library"];
    if library.is_object() {
        return format!(
            "Proposal library {}: {} · {} / {} pools saved · {}",
            phase(library),
            library["reason"]
                .as_str()
                .unwrap_or("seed generation precedes GP teaching scores"),
            decimal_count(library, "saved_seed_pools"),
            decimal_count(library, "planned_seed_pools"),
            status_age_with_limit(library, 15)
        );
    }
    if run["contexts"]["completed"].is_number() {
        if completed(&run["auxiliary"]) {
            return format!("Auxiliary training complete: {}. GP searches retain their frozen proposal checkpoint.",run["auxiliary"]["reason"].as_str().unwrap_or("bounded learning block saved"));
        }
        return "Encoding queue finished; vectors are saved. GPUs await the next training/proposal block.".into();
    }
    "GPU work receipt unavailable; consult Neural and recent events.".into()
}

fn stage_work_detail(cycle: &Value) -> String {
    if cycle["heartbeat_role"] == "controller_liveness" {
        let work = &cycle["work"];
        format!(
            "controller live; {} {} · work age {}",
            work["operation"]
                .as_str()
                .unwrap_or("awaiting work receipt"),
            work["completed"]
                .as_u64()
                .map(|n| n.to_string())
                .unwrap_or_default(),
            duration(now().saturating_sub(number(cycle, "last_progress_at")))
        )
    } else {
        cycle["reason"]
            .as_str()
            .unwrap_or("CPU preparation precedes the next training block")
            .into()
    }
}
fn milestone_lines(run: &Value, gp: &Value, neural: &Value) -> Vec<Line<'static>> {
    let contexts = &run["contexts"];
    let policy =
        if run["policy"]["corpus_sha256"].is_string() && run["policy"]["invalidated"] != true {
            &run["policy"]
        } else if run["fresh_policy"].is_object() {
            &run["fresh_policy"]
        } else {
            neural
        };
    let gp_total = number(gp, "completed_views")
        + number(gp, "failed_views")
        + gp["active"]
            .as_u64()
            .or_else(|| gp["last_recorded_activity"]["active"].as_u64())
            .unwrap_or(0)
        + number(gp, "queued");
    let cycle = &run["cycle"];
    let mut lines = vec![
        label(
            "1  Encodings on disk",
            format!(
                "{} views · {:.1}% · {} encoded families · {} dimensions · {}",
                count_pair(number(contexts, "completed"), number(contexts, "total")),
                100.0 * number(contexts, "completed") as f64
                    / number(contexts, "total").max(1) as f64,
                number(contexts, "families"),
                number(contexts, "dimensions"),
                if contexts["qualification"] == false {
                    "qualified"
                } else {
                    "qualification cohort"
                }
            ),
        ),
        label(
            "2  GP teaching targets",
            format!(
                "{} search jobs · {} evaluations · {} failed",
                count_pair(number(gp, "completed_views"), gp_total),
                number(gp, "completed_evaluations") + number(gp, "running_evaluations"),
                number(gp, "failed_views")
            ),
        ),
        label(
            "3  Policy checkpoint",
            if policy["invalidated"] == true {
                "previous run invalidated · corrected training pending".into()
            } else if policy["encoding"] == true || !policy["step"].is_number() {
                "awaiting training receipt".into()
            } else {
                format!(
                    "{} steps · {} ranks · {}",
                    count_pair(number(policy, "step"), number(policy, "steps")),
                    number(policy, "world"),
                    status_age(policy)
                )
            },
        ),
        label(
            "4  Neural → GP → DL",
            if run["continuous"].is_object() {
                let (status, detail) = campaign_task_state(run, neural);
                format!("{status} · {detail}")
            } else if cycle["complete"] == true {
                "completed · comparisons and fitted predictors saved".into()
            } else if cycle["phase"].is_string() {
                format!(
                    "{} · proposals {} · GP {} · retrain {}",
                    phase(cycle),
                    count_pair(
                        number(&cycle["proposals"], "completed"),
                        number(&cycle["proposals"], "total")
                    ),
                    count_pair(
                        number(&cycle["neural_seeded_gp"], "completed"),
                        number(&cycle["neural_seeded_gp"], "total")
                    ),
                    count_pair(
                        number(&cycle["retraining"], "step"),
                        number(&cycle["retraining"], "steps")
                    )
                )
            } else {
                "pending neural proposals, GP reevaluation, then retraining".into()
            },
        ),
    ];
    if run["encoding_pipeline"]["phase"].is_string() {
        let pipeline = &run["encoding_pipeline"];
        lines.push(label(
            "Next encoding cohort",
            format!(
                "{} · {} contexts · {}",
                phase(pipeline),
                count_pair(number(pipeline, "completed"), number(pipeline, "total")),
                status_age(pipeline)
            ),
        ));
    }
    if run["continuous"].is_object() {
        let (status, detail) = campaign_task_state(run, neural);
        lines.push(label("Continuous learning", format!("{status} · {detail}")));
    }
    if run["campaign_service"]["phase"].is_string() {
        let campaign = &run["campaign_service"];
        lines.push(label(
            if run["continuous"].is_object() {
                "Historical campaign"
            } else {
                "Managed campaign"
            },
            format!(
                "{} cycles complete · {}",
                number(campaign, "completed_cycles"),
                phase(campaign)
            ),
        ));
    }
    if let Some(checkpoint) = run["selected_checkpoint"]["checkpoint"].as_str() {
        lines.push(label(
            "Selected checkpoint",
            checkpoint.trim_start_matches("runs/"),
        ));
    }
    lines.push(label(
        "24-hour pilot",
        if run["pilot_started"] == true {
            "running · inspect pilot receipt"
        } else {
            "not started · qualification work; full pilot gates remain"
        },
    ));
    lines
}
fn label<'a>(key: &str, value: impl Into<String>) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("{key:<23}"), Style::default().fg(MUTED)),
        Span::raw(value.into()),
    ])
}
struct TaskRow {
    name: String,
    status: String,
    progress: String,
    detail: String,
}
impl TaskRow {
    fn table_row(self) -> Row<'static> {
        let color = match self.status.as_str() {
            "DONE" => GREEN,
            "RUNNING" => BLUE,
            "FAILED" => RED,
            "STALE" | "HELD" => GOLD,
            _ => MUTED,
        };
        Row::new(vec![
            Cell::from(self.name).style(Style::default().bold()),
            Cell::from(self.status).style(Style::default().fg(color).bold()),
            Cell::from(self.progress),
            Cell::from(self.detail),
        ])
    }
}
fn task_row(name: &str, status: &str, progress: String, detail: String) -> TaskRow {
    TaskRow {
        name: name.into(),
        status: status.into(),
        progress,
        detail,
    }
}
fn training_hold(run: &Value, policy: &Value) -> Option<String> {
    for value in [&run["continuous"], policy] {
        let state = value["status"].as_str().unwrap_or("").to_ascii_lowercase();
        let phase = phase(value).to_ascii_lowercase();
        if matches!(state.as_str(), "held" | "paused" | "blocked")
            || phase == "held"
            || phase.starts_with("held ")
            || phase == "paused"
            || phase.contains("training held")
            || value["automatic_training"] == false
            || value["automatic_training_hold"] == true
            || value["automatic_training_hold"]["block_new_training"] == true
        {
            return Some(
                value["training_hold_reason"]
                    .as_str()
                    .or_else(|| value["reason"].as_str())
                    .unwrap_or("automatic training is held")
                    .into(),
            );
        }
    }
    None
}
fn receipt_task_state(v: &Value) -> &'static str {
    if v["invalidated"] == true || v["status"] == "failed" {
        "FAILED"
    } else if training_hold(&Value::Null, v).is_some() {
        "HELD"
    } else if completed(v) {
        "DONE"
    } else if number(v, "timestamp") == 0 {
        "QUEUED"
    } else if number(v, "timestamp") > now() || now().saturating_sub(number(v, "timestamp")) > 15 {
        "STALE"
    } else if phase(v).to_ascii_lowercase().contains("waiting") {
        "WAITING"
    } else {
        "RUNNING"
    }
}
fn policy_task_state(run: &Value, policy: &Value) -> (&'static str, String) {
    if let Some(reason) = training_hold(run, policy) {
        let active = run["continuous"]["training_task_reservation"].is_object();
        return (
            "HELD",
            format!(
                "{reason} · policy receipt {}{}",
                status_age(policy),
                if active {
                    " · existing funded task requires separate reconciliation"
                } else {
                    ""
                }
            ),
        );
    }
    (
        receipt_task_state(policy),
        format!(
            "{} trained families · NLL {}",
            number(policy, "training_families"),
            decimal(policy, "sequence_nll")
        ),
    )
}
// Campaign progress is written at phase transitions. It is not a heartbeat.
fn campaign_task_state(run: &Value, neural: &Value) -> (&'static str, String) {
    let campaign = &run["campaign_service"];
    let actual = &run["operations"]["services"]["jope-parallel-campaign"];
    let continuous = &run["continuous"];
    if continuous["phase"].is_string() {
        if actual["ActiveState"] != "active" {
            return (
                "STALE",
                format!(
                    "controller {} · saved {}",
                    actual["ActiveState"],
                    phase(continuous)
                ),
            );
        }
        if now().saturating_sub(number(continuous, "timestamp")) > 15 {
            return ("STALE", "continuous controller heartbeat stale".into());
        }
        return (
            if continuous["phase"] == "paused" {
                "PAUSED"
            } else {
                "RUNNING"
            },
            format!(
                "{} training blocks committed · next {}",
                number(continuous, "completed_training_blocks"),
                continuous["next_scheduled_work"]
            ),
        );
    }
    let deadline = phase(campaign).to_ascii_lowercase().contains("deadline")
        || campaign["status"] == "deadline";
    if deadline {
        if actual["ActiveState"] == "active" {
            return (
                "RECOVERY",
                format!("recorded continuation · {}", phase(&run["cycle"])),
            );
        }
        return (
            "DEADLINE",
            "qualification deadline reached; retraining completion requires a cycle receipt".into(),
        );
    }
    if actual["ActiveState"] == "failed" {
        return ("FAILED", format!("systemd result {}", actual["Result"]));
    }
    if completed(campaign) {
        return ("DONE", phase(campaign).into());
    }
    if campaign["invalidated"] == true || campaign["status"] == "failed" {
        return ("FAILED", phase(campaign).into());
    }
    if !campaign["phase"].is_string() {
        return (
            "QUEUED",
            "Awaiting corpus admission and launch receipt".into(),
        );
    }
    let relative = |path: &str| {
        path.split_once("/runs/")
            .map(|(_, tail)| tail)
            .unwrap_or(path.strip_prefix("runs/").unwrap_or(path))
            .to_owned()
    };
    let current = campaign["current_cycle"].as_str().map(relative);
    let fresh = |v: &Value| {
        number(v, "timestamp") > 0
            && now().saturating_sub(number(v, "timestamp")) <= 15
            && !completed(v)
    };
    let same_cycle = current.as_ref().is_some_and(|current| {
        run["active_cycle"].as_str().map(relative).as_ref() == Some(current)
    });
    let neural_in_cycle = current.as_ref().is_some_and(|current| {
        neural["run"]
            .as_str()
            .is_some_and(|path| Path::new(&relative(path)).starts_with(current))
    });
    let active = (same_cycle && fresh(&run["cycle"])) || (neural_in_cycle && fresh(neural));
    let timestamp = number(campaign, "timestamp");
    let phase_age = if timestamp > 0 {
        format!("phase age {}", duration(now().saturating_sub(timestamp)))
    } else {
        "phase age unavailable".into()
    };
    (
        if active { "RUNNING" } else { "PHASE" },
        format!(
            "{} · {}{}",
            phase(campaign),
            phase_age,
            if active {
                " · current-cycle activity live"
            } else {
                " · transition receipt; inspect worker activity"
            }
        ),
    )
}
fn proposal_library_status(v: &Value) -> &'static str {
    if v["complete"] == true || phase(v).ends_with("comparison complete") {
        return "SAVED";
    }
    match phase(v) {
        "failed" => "FAILED",
        "completed" => "SAVED",
        "stopped" => "STOPPED",
        "paused" => "PAUSED",
        _ if number(v, "timestamp") == 0 => "QUEUED",
        _ if now().saturating_sub(number(v, "timestamp")) > 15 => "STALE",
        p if p.to_ascii_lowercase().contains("waiting") => "WAITING",
        _ => "RUNNING",
    }
}
fn report_status(v: &Value) -> &'static str {
    if completed(v) {
        "DONE"
    } else if v["invalidated"] == true || phase(v) == "Report needs inspection" {
        "FAILED"
    } else if number(v, "timestamp") == 0 {
        "QUEUED"
    } else if now().saturating_sub(number(v, "timestamp")) > 45 {
        "STALE"
    } else if phase(v).starts_with("Waiting") {
        "WAITING"
    } else {
        "RUNNING"
    }
}
fn workflow_rows(run: &Value, gp: &Value, neural: &Value) -> Vec<TaskRow> {
    let contexts = &run["contexts"];
    let boot = &run["bootstrap"];
    let cycle = &run["cycle"];
    let policy = if run["policy"].is_object() {
        &run["policy"]
    } else {
        neural
    };
    let live = receipt_task_state;
    let (policy_status, policy_detail) = policy_task_state(run, policy);
    let (campaign_status, campaign_detail) = campaign_task_state(run, neural);
    let mut rows = vec![
        task_row(
            "Saved contexts",
            if number(contexts, "completed") > 0
                && number(contexts, "completed") == number(contexts, "total")
            {
                "DONE"
            } else {
                live(contexts)
            },
            count_pair(number(contexts, "completed"), number(contexts, "total")),
            "2048D · source pointers · frozen encoders".into(),
        ),
        task_row(
            "GP teacher search jobs",
            if gp["current_activity"] == "stale" {
                "STALE"
            } else if gp["current_activity"] == "terminal" {
                "STOPPED"
            } else if gp["phase"] == "waiting" {
                "WAITING"
            } else if gp["phase"] == "completed" {
                "EXHAUSTED"
            } else if gp["phase"] == "paused" {
                "PAUSED"
            } else if number(gp, "active") > 0 {
                live(gp)
            } else {
                "QUEUED"
            },
            format!(
                "{} done / {} active",
                number(gp, "completed_views"),
                decimal_count(gp, "active")
            ),
            format!(
                "{} queued · {} eval/s",
                number(gp, "queued"),
                decimal(gp, "evaluations_per_second")
            ),
        ),
        task_row(
            "64-family bootstrap",
            if boot["compact_receipt_ready"] == true {
                "DONE"
            } else {
                "QUEUED"
            },
            format!(
                "{} train · {} validation",
                number(
                    &boot["partitions"]["meta_train"],
                    "families_with_completed_view"
                ),
                number(
                    &boot["partitions"]["meta_validation"],
                    "families_with_completed_view"
                )
            ),
            "Finite reward-group admission remains separate".into(),
        ),
        task_row(
            "Teaching corpus",
            live(&run["corpus_progress"]),
            format!(
                "{} records · {} / {} families",
                number(&run["corpus_progress"], "records"),
                number(&run["corpus_progress"], "completed"),
                number(&run["corpus_progress"], "total")
            ),
            run["corpus_progress"]["reason"]
                .as_str()
                .unwrap_or("Awaiting finite reward-group admission receipt")
                .into(),
        ),
        if run["continuous"].is_object() {
            task_row(
                "Self-play runtime",
                campaign_status,
                "Continuous supervisor".into(),
                campaign_detail.clone(),
            )
        } else {
            task_row(
                "Self-play runtime",
                live(&run["runtime_readiness"]),
                run["runtime_readiness"]["phase"]
                    .as_str()
                    .unwrap_or("verification pending")
                    .into(),
                run["runtime_readiness"]["reason"]
                    .as_str()
                    .unwrap_or("A complete corpus does not imply the next campaign has launched")
                    .into(),
            )
        },
        task_row(
            "Policy optimization",
            policy_status,
            count_pair(number(policy, "step"), number(policy, "steps")),
            policy_detail,
        ),
        task_row(
            "Auxiliary learning",
            live(&run["auxiliary"]),
            format!(
                "{} / {} steps",
                run["auxiliary"]["completed_optimizer_steps"]
                    .as_u64()
                    .or_else(|| run["auxiliary"]["step"].as_u64())
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "—".into()),
                decimal_count(&run["auxiliary"], "maximum_steps")
            ),
            run["auxiliary"]["reason"]
                .as_str()
                .or_else(|| run["auxiliary"]["phase"].as_str())
                .unwrap_or("Bounded learning during GP wait; proposal checkpoint stays frozen")
                .into(),
        ),
        task_row(
            "Neural proposals",
            if number(&cycle["proposals"], "total") > 0
                && number(&cycle["proposals"], "completed") == number(&cycle["proposals"], "total")
            {
                "DONE"
            } else {
                "QUEUED"
            },
            count_pair(
                number(&cycle["proposals"], "completed"),
                number(&cycle["proposals"], "total"),
            ),
            "Real context + shuffled context comparison".into(),
        ),
        task_row(
            "DL → GP → DL",
            if cycle["complete"] == true {
                "DONE"
            } else {
                live(cycle)
            },
            format!(
                "{} completed cycles",
                run["reports"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|v| v["raw_final_scores_only"] != true || v["complete"] == true)
                    .count()
            ),
            phase(cycle).into(),
        ),
        task_row(
            "Managed campaign",
            campaign_status,
            format!(
                "{} cycles saved",
                number(&run["campaign_service"], "completed_cycles")
            ),
            campaign_detail,
        ),
    ];
    // The report watcher that wrote this receipt was archived (see `campaign`), so a missing
    // receipt is not a pending one: show the row only for a saved receipt.
    let report = &run["report_progress"];
    if report.is_object() {
        rows.push(task_row(
            "Matched-budget reports",
            report_status(report),
            if phase(report).starts_with("Waiting") && number(report, "total") == 0 {
                "0 completed cycles".into()
            } else {
                count_pair(number(report, "completed"), number(report, "total"))
            },
            report["reason"]
                .as_str()
                .or_else(|| report["phase"].as_str())
                .unwrap_or("Completed-cycle reporting receipt pending")
                .into(),
        ));
    }
    let mut operations_rows = Vec::new();
    if run["operations"].is_object() {
        let services = &run["operations"]["services"];
        let progress = if run["operations"]["gp_units"]
            .as_array()
            .is_some_and(|v| !v.is_empty())
        {
            format!(
                "GP {} · DL {}",
                decimal_count(gp, "active"),
                services["jope-parallel-campaign"]["ActiveState"]
                    .as_str()
                    .unwrap_or("unknown")
            )
        } else {
            format!(
                "Managed DL {} · GP {}",
                services["jope-parallel-campaign"]["ActiveState"]
                    .as_str()
                    .unwrap_or("unknown"),
                services["jope-parallel-gp"]["ActiveState"]
                    .as_str()
                    .unwrap_or("unknown")
            )
        };
        operations_rows.push(task_row(
            "Actual services",
            "OBSERVED",
            progress,
            format!(
                "GP units: {} · GPU units: {} · owners: {}",
                run["operations"]["gp_units"],
                run["operations"]["gpu_units"],
                run["operations"]["gpu_owners"]
            ),
        ));
    }
    let continuous = &run["continuous"];
    if continuous.is_object() {
        operations_rows.push(task_row(
            "GPU inactivity",
            continuous["inactivity"].as_str().unwrap_or("OBSERVED"),
            format!(
                "{} idle · {} GPU-s paid",
                duration(number(continuous, "gpu_idle_seconds")),
                number(continuous, "paid_gpu_idle_seconds")
            ),
            continuous["idle_explanation"]
                .as_str()
                .unwrap_or("No active packing explanation")
                .into(),
        ));
        operations_rows.push(task_row(
            "Last committed model",
            "SAVED",
            format!("{} blocks", number(continuous, "completed_training_blocks")),
            continuous["last_committed_checkpoint"]
                .as_str()
                .unwrap_or("unavailable")
                .into(),
        ));
        let pack = &run["continuous_packing"];
        operations_rows.push(task_row(
            "Snapshot packing",
            if !pack.is_object() {
                "WAITING"
            } else if pack["complete"] == true {
                "DONE"
            } else if now().saturating_sub(number(pack, "timestamp")) > 15 {
                "STALE"
            } else {
                "RUNNING"
            },
            format!("{}", pack["work"]["completed"]),
            format!("{}", pack["work"]["operation"]),
        ));
        let monitor = &run["frozen_monitoring"];
        if monitor.is_object() {
            operations_rows.push(task_row(
                "Frozen validation",
                "SAVED",
                format!(
                    "NLL {} · BT {}",
                    decimal(monitor, "sequence_nll"),
                    decimal(monitor, "ranking_bt")
                ),
                format!(
                    "{} examples · panel {} · adaptive validation; separate from training loss",
                    number(monitor, "sampled_examples"),
                    monitor["binding"]["panel_sha256"]
                        .as_str()
                        .unwrap_or("unknown")
                ),
            ));
        }
    }
    let library = &run["proposal_library"];
    if library.is_object() {
        let admission = &run["proposal_library_admission"];
        let observed = &run["proposal_library_gp"];
        let detail = if admission["committed"] == true {
            format!(
                "{} ranked · GP: {} admitted / {} observed evals · {} / {} outputs checked",
                decimal_count(library, "saved_ranked_programs"),
                decimal_count(admission, "requested_jobs"),
                decimal_count(observed, "observed_evaluations"),
                decimal_count(observed, "checked_jobs"),
                decimal_count(admission, "requested_jobs")
            )
        } else if admission.is_object() {
            format!(
                "{} ranked · GP admission {} · 0 admitted / 0 evaluations",
                decimal_count(library, "saved_ranked_programs"),
                phase(admission)
            )
        } else {
            format!(
                "{} ranked programs · GP: {} jobs admitted / {} evaluations · seeds only",
                decimal_count(library, "saved_ranked_programs"),
                decimal_count(library, "gp_jobs_admitted"),
                decimal_count(library, "gp_evaluations")
            )
        };
        rows.insert(
            8,
            task_row(
                "Proposal seed library",
                proposal_library_status(library),
                format!(
                    "{} / {} pools",
                    decimal_count(library, "saved_seed_pools"),
                    decimal_count(library, "planned_seed_pools")
                ),
                detail,
            ),
        );
    }
    let pure = &run["pure_proposal_library"];
    if pure.is_object() {
        let admission = if pure["gp_jobs_admitted"] == 0 {
            "unadmitted".into()
        } else {
            format!("{} admitted", decimal_count(pure, "gp_jobs_admitted"))
        };
        let detail = format!(
            "{} ranked · {} / {} GP evals · separate pure scope",
            decimal_count(pure, "saved_ranked_programs"),
            admission,
            decimal_count(pure, "gp_evaluations")
        );
        rows.insert(
            8 + usize::from(library.is_object()),
            task_row(
                "Pure policy predictions",
                proposal_library_status(pure),
                format!(
                    "{} / {} pools",
                    decimal_count(pure, "saved_seed_pools"),
                    decimal_count(pure, "planned_seed_pools")
                ),
                detail,
            ),
        );
    }
    if run["active_cycle"].is_string() {
        let exports = &run["predictor_exports"];
        rows.push(task_row(
            "Predictor export",
            if number(exports, "total") > 0
                && number(exports, "completed") == number(exports, "total")
            {
                "DONE"
            } else if phase(cycle).starts_with("Fit and save") {
                live(cycle)
            } else {
                "QUEUED"
            },
            count_pair(number(exports, "completed"), number(exports, "total")),
            if phase(cycle).starts_with("Fit and save") {
                stage_work_detail(cycle)
            } else {
                "Fitted predictors · 30-second / 8 GiB limits".into()
            },
        ));
        rows.push(task_row(
            "Corpus packing",
            if phase(cycle).starts_with("Pack ") {
                live(cycle)
            } else {
                "WAITING"
            },
            format!(
                "{} current records",
                decimal_count(&run["corpus_progress"], "records")
            ),
            if phase(cycle).starts_with("Pack ") {
                stage_work_detail(cycle)
            } else {
                "New scores packed before retraining; exact program labels".into()
            },
        ));
    }
    for (key, title, count) in [
        ("seed_experiment", "Seed quality experiment", "completed"),
        ("binary_qualification", "Binary qualification", "completed"),
    ] {
        let v = &run[key];
        if v.is_object() {
            let (progress, detail) =
                if key == "binary_qualification" && run["binary_admitted_contexts"].is_number() {
                    (
                        format!(
                            "{} admitted · {} replayed",
                            decimal_count(run, "binary_admitted_contexts"),
                            decimal_count(&run["binary_verification"], "verified_pairs")
                        ),
                        binary_replay_detail(run),
                    )
                } else {
                    (
                        count_pair(number(v, count), number(v, "total")),
                        format!(
                            "{} · {}",
                            phase(v),
                            v["reason"].as_str().unwrap_or("qualification results only")
                        ),
                    )
                };
            rows.push(task_row(
                title,
                proposal_library_status(v),
                progress,
                detail,
            ));
        }
    }
    if run["hyperion_recovery"]["passed"] == true {
        rows.push(task_row(
            "CUDA recovery",
            "DONE",
            format!(
                "{} comparisons",
                number(&run["hyperion_recovery"], "shared_input_comparisons")
            ),
            "Smaller batches passed · previous vectors preserved".into(),
        ));
    }
    operations_rows.extend(rows);
    operations_rows
}
fn binary_replay_detail(run: &Value) -> String {
    format!(
        "{} admitted · {} bindings verified · {} exact replay pairs · {} raw components recomputed",
        decimal_count(run, "binary_admitted_contexts"),
        decimal_count(run, "binary_verified_contexts"),
        decimal_count(&run["binary_verification"], "verified_pairs"),
        decimal_count(
            &run["binary_verification"],
            "independently_computed_raw_components"
        )
    )
}
fn comparison_lines(run: &Value) -> Vec<Line<'static>> {
    let mut lines = vec![Line::styled(
        "POLICY VALUE · MATCHED SEARCH BUDGETS",
        Style::default().fg(VIOLET).bold(),
    )];
    if run["experiment_manifest"].is_string() {
        lines.push(Line::raw(format!(
            "Experiment: {}",
            run["experiment_manifest"].as_str().unwrap_or("unavailable")
        )));
        lines.push(Line::raw(format!(
            "Production selected: {}",
            run["production_checkpoint"]
                .as_str()
                .unwrap_or("unavailable")
        )));
        lines.push(Line::raw(format!(
            "Experiment frozen: {}",
            run["experiment_checkpoint"]
                .as_str()
                .unwrap_or("unavailable")
        )));
    }
    let matched = &run["matched_quality"];
    if matched.is_object() {
        lines.push(Line::raw(format!(
            "Matched repeats: {} / {} uninterrupted blocks · {} interrupted preserved",
            number(matched, "completed_blocks"),
            number(matched, "required_blocks"),
            number(matched, "interrupted_blocks_preserved")
        )));
        for seed in [20260907, 20260908, 20260909] {
            let key = format!("{seed}:larger_pool_vs_pure_first64");
            lines.push(Line::raw(format!(
                "Seed {seed}: larger / pure_first64 {} · {} / {} families",
                decimal(&matched["comparisons"][&key], "family_balanced_ratio"),
                number(&matched["comparisons"][&key], "families_completed"),
                number(&matched["comparisons"][&key], "families_required")
            )));
        }
        lines.push(Line::raw(format!(
            "Evaluated fits: {} · selector remains undeployed",
            number(matched, "successful_fits_all_attempts")
        )));
        lines.push(Line::raw(format!(
            "Evaluated teachers: {} · selected-seed wins: {} / {}",
            number(matched, "evaluated_teachers_all_attempts"),
            number(
                &matched["attributed_improvements"]["pure_first64"],
                "neural_selection_wins"
            ),
            number(
                &matched["attributed_improvements"]["generic"],
                "neural_selection_wins"
            )
        )));
    }
    if run["binary_admitted_contexts"].is_number() {
        let verified = &run["binary_verification"];
        lines.push(Line::raw(format!("Binary: {} admitted · {} bindings verified · {} exact replay pairs · {} raw components recomputed",run["binary_admitted_contexts"],decimal_count(run,"binary_verified_contexts"),number(verified,"verified_pairs"),number(verified,"independently_computed_raw_components"))));
    }
    let experiment = &run["seed_quality"];
    if experiment.is_object() {
        let frozen_cycle = experiment["selected_checkpoint"]
            .as_str()
            .and_then(|path| path.split('/').find(|part| part.starts_with("cycle-")))
            .unwrap_or("checkpoint");
        lines[0] = Line::styled(
            format!("PURE SEED QUALITY · FROZEN {frozen_cycle} · NO TEACHERS"),
            Style::default().fg(VIOLET).bold(),
        );
        for (key, label) in [
            ("larger_pool_vs_pure_first64", "larger / pure_first64"),
            ("larger_pool_vs_generic", "larger / generic"),
        ] {
            let legacy = if key == "larger_pool_vs_pure_first64" {
                "larger_pool_vs_current"
            } else {
                key
            };
            let best = if experiment["comparisons"][key].is_object() {
                &experiment["comparisons"][key]["best8"]
            } else {
                &experiment["comparisons"][legacy]["best8"]
            };
            lines.push(Line::raw(format!(
                "Best8 {label}: {} · {}W / {}T / {}L · {} / {} families",
                decimal(best, "family_balanced_ratio"),
                number(best, "wins"),
                number(best, "ties"),
                number(best, "losses"),
                number(best, "families_completed"),
                number(best, "families_required")
            )));
        }
        let fit_key = if experiment["top8_physical_fits"].is_number() {
            "top8_physical_fits"
        } else {
            "physical_fits"
        };
        let fail_key = if experiment["top8_failures"].is_number() {
            "top8_failures"
        } else {
            "failures"
        };
        lines.push(Line::raw(format!(
            "Top8: {} fits · {} logical evals · {} failures",
            decimal_count(experiment, fit_key),
            decimal_count(experiment, "logical_evaluations"),
            decimal_count(experiment, fail_key)
        )));
        let downstream = &experiment["downstream"];
        if downstream.is_object() {
            lines.push(Line::raw(format!(
                "GP {} / {} · larger/pure_first64 {} · larger/generic {}",
                number(downstream, "completed_views"),
                number(downstream, "required_views"),
                decimal(
                    &downstream["comparisons"]["current"],
                    "family_balanced_ratio"
                ),
                decimal(
                    &downstream["comparisons"]["generic"],
                    "family_balanced_ratio"
                )
            )));
            lines.push(Line::raw(format!(
                "Selected-seed wins {} vs pure_first64 / {} vs generic; inherited/evolved excluded",
                number(
                    &downstream["direct_neural_selected_winner_improvements"],
                    "current"
                ),
                number(
                    &downstream["direct_neural_selected_winner_improvements"],
                    "generic"
                )
            )));
        }
        lines.push(Line::styled(
            experiment["conclusion"]
                .as_str()
                .unwrap_or("Comparison incomplete; existing production seeder retained")
                .to_owned(),
            Style::default().fg(if experiment["positive_meeting_result"] == true {
                GREEN
            } else {
                GOLD
            }),
        ));
        lines.push(Line::raw(
            "Qualification within previously used families; production seeder unchanged.",
        ));
        return lines;
    }
    for report in run["reports"]
        .as_array()
        .into_iter()
        .flatten()
        .rev()
        .take(2)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        let val = &report["validation"];
        let comparisons = &val["evaluation_prefix"];
        lines.push(Line::styled(
            report["cycle"].as_str().unwrap_or("Saved cycle").to_owned(),
            Style::default().fg(INK).bold(),
        ));
        if report["raw_final_scores_only"] == true {
            lines.push(Line::styled(
                "  RAW FINAL SCORES · unequal completed budgets",
                Style::default().fg(GOLD),
            ));
            let rows = report["comparisons"].as_array();
            if let Some(row) = rows
                .into_iter()
                .flatten()
                .find(|v| v["partition"] == "meta_validation")
                .or_else(|| rows.and_then(|v| v.first()))
            {
                lines.push(Line::raw(format!(
                    "  {} cut{} · NN {} / generic {} / shuffled {}",
                    row["family"].as_str().unwrap_or("family"),
                    number(row, "cut"),
                    rmse(&row["neural"], "best_validation_rmse"),
                    rmse(&row["generic_retained"], "best_validation_rmse"),
                    rmse(&row["shuffled_context"], "best_validation_rmse")
                )));
                lines.push(Line::raw(format!(
                    "  Eval budgets {}/{}/{} · matched-prefix report pending",
                    number(&row["neural"], "completed_evaluations"),
                    number(&row["generic_retained"], "completed_evaluations"),
                    number(&row["shuffled_context"], "completed_evaluations")
                )));
            }
            continue;
        }
        lines.push(Line::raw(format!(
            "  NN / generic {} · NN / shuffled {} · {} families",
            decimal(
                &comparisons["neural_over_generic"],
                "family_balanced_rmse_ratio"
            ),
            decimal(
                &comparisons["neural_over_shuffled"],
                "family_balanced_rmse_ratio"
            ),
            number(val, "source_families")
        )));
        lines.push(Line::raw(format!(
            "  {} searches · {} teacher records · {} fitted predictors",
            number(report, "searches"),
            number(report, "records"),
            number(report, "predictors")
        )));
        let wall = &val["wall_prefix_bounds"]["neural_over_generic"];
        if wall["rmse_ratio_lower"].is_number() {
            lines.push(Line::raw(format!(
                "  Wall-matched NN/generic bounds {}–{}",
                decimal(wall, "rmse_ratio_lower"),
                decimal(wall, "rmse_ratio_upper")
            )));
        }
    }
    if run["reports"].as_array().is_none_or(|v| v.is_empty()) {
        lines.push(Line::raw(
            "Awaiting a measured three-arm comparison report.",
        ));
    }
    lines.push(Line::styled(
        "Lower RMSE ratio is better. Policy advantage is not established.",
        Style::default().fg(GOLD),
    ));
    lines.push(Line::raw(
        "Adaptive held-out teaching scores; original final tests stay sealed.",
    ));
    lines
}
fn teacher_lines(run: &Value) -> Vec<Line<'static>> {
    let teacher = &run["teacher"];
    let boot = &run["bootstrap"];
    let mut lines = vec![
        Line::styled(
            "BEST GP + TRAINING ENCODING → POLICY → NEXT GP",
            Style::default().fg(GREEN).bold(),
        ),
        Line::raw("Improve the teacher on each cut, then learn its program choices."),
        Line::raw("A stronger teacher and a stronger policy are measured separately."),
        Line::raw(""),
    ];
    for (key, label) in [
        ("meta_train", "Training families"),
        ("meta_validation", "Validation families"),
        ("meta_test", "Test families · sealed"),
    ] {
        let v = &boot["partitions"][key];
        lines.push(self::label(
            label,
            format!(
                "{} / {} with completed GP views",
                number(v, "families_with_completed_view"),
                number(v, "selected_families")
            ),
        ));
    }
    lines.push(label(
        "Coverage receipt",
        if boot["compact_receipt_ready"] == true {
            "coverage ready · saved receipt".into()
        } else {
            status_age(boot)
        },
    ));
    lines.push(label(
        "Best-teacher coverage",
        teacher["cuts_with_teacher"]
            .as_u64()
            .map(|v| format!("{v} unique cuts"))
            .unwrap_or_else(|| "awaiting compact teacher receipt".into()),
    ));
    lines.push(label(
        "New best programs",
        teacher["new_bests"]
            .as_u64()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unavailable".into()),
    ));
    lines.push(label(
        "Teacher telemetry",
        status_age_with_limit(teacher, 15),
    ));
    lines
}
fn teacher_evolution_lines(run: &Value) -> Vec<Line<'static>> {
    let teacher = &run["teacher"];
    let mut lines = vec![
        Line::styled(
            "BEST FOUND · EXACT CUT / EVALUATOR IDENTITY",
            Style::default().fg(GREEN).bold(),
        ),
        label(
            "Teacher coverage",
            teacher["cuts_with_teacher"]
                .as_u64()
                .map(|n| {
                    format!(
                        "{n} unique cuts · {} identities",
                        number(teacher, "evaluator_cut_identities")
                    )
                })
                .unwrap_or_else(|| "awaiting durable champion receipt".into()),
        ),
        label(
            "New best / improving cuts",
            format!(
                "{} / {}",
                decimal_count(teacher, "new_bests"),
                decimal_count(teacher, "matched_cut_improvements")
            ),
        ),
        label(
            "Champion reuse",
            teacher["champion_reuse"]["status"]
                .as_str()
                .unwrap_or("unavailable"),
        ),
    ];
    for item in teacher["top_improvements"]
        .as_array()
        .into_iter()
        .flatten()
        .take(2)
    {
        let family = item["family"]
            .as_str()
            .or_else(|| item["dataset"].as_str())
            .unwrap_or("cut");
        lines.push(Line::raw(format!(
            "{} · cut {}",
            family,
            number(item, "cut")
        )));
        lines.push(Line::styled(
            format!(
                "  RMSE {} → {} · {} evaluations",
                rmse(item, "old_rmse"),
                rmse(item, "new_rmse"),
                number(item, "evaluations")
            ),
            Style::default().fg(GREEN),
        ));
        let points = item["trajectory"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v["rmse"].as_f64())
            .filter(|v| v.is_finite())
            .collect::<Vec<_>>();
        if points.len() > 1 {
            let low = points.iter().copied().fold(f64::INFINITY, f64::min);
            let high = points.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let glyphs = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
            let curve = points
                .iter()
                .map(|v| {
                    glyphs[(((v - low) / (high - low).max(1e-20) * 7.).round() as usize).min(7)]
                })
                .collect::<String>();
            lines.push(Line::styled(
                format!(
                    "  Best-found RMSE {curve} · {} ordered updates",
                    points.len()
                ),
                Style::default().fg(VIOLET),
            ));
        }
    }
    lines.push(Line::styled(
        "Best observed programs, without a global-optimum claim.",
        Style::default().fg(MUTED),
    ));
    lines
}
fn decimal_count(v: &Value, key: &str) -> String {
    v[key]
        .as_u64()
        .map(|n| n.to_string())
        .unwrap_or_else(|| "—".into())
}
fn learning_chart(f: &mut Frame, area: Rect, run: &Value) {
    let samples = run["learning_curve"].as_array();
    let train = samples
        .into_iter()
        .flatten()
        .filter_map(|v| Some((v["step"].as_f64()?, v["sequence_nll"].as_f64()?)))
        .filter(|(_, y)| y.is_finite())
        .collect::<Vec<_>>();
    let mut validation = samples
        .into_iter()
        .flatten()
        .filter_map(|v| {
            Some((
                v["validation"]["evaluated_step"].as_f64()?,
                v["validation"]["sequence_nll"].as_f64()?,
            ))
        })
        .filter(|(_, y)| y.is_finite())
        .collect::<Vec<_>>();
    validation.dedup_by(|a, b| a.0 == b.0);
    if train.len() < 2 {
        f.render_widget(
            Paragraph::new("Learning curve unavailable until two optimizer receipts are saved.")
                .wrap(Wrap { trim: true })
                .block(panel(" Sequence NLL · saved optimizer history ")),
            area,
        );
        return;
    }
    let xmax = train.last().unwrap().0.max(1.);
    let ymin = train
        .iter()
        .chain(&validation)
        .map(|(_, y)| *y)
        .fold(f64::INFINITY, f64::min);
    let ymax = train
        .iter()
        .chain(&validation)
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);
    let padding = ((ymax - ymin) * 0.1).max(0.02);
    let mut datasets = vec![Dataset::default()
        .name("train")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(BLUE))
        .data(&train)];
    if !validation.is_empty() {
        datasets.push(
            Dataset::default()
                .name("validation (bound step)")
                .marker(symbols::Marker::Dot)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(GOLD))
                .data(&validation),
        );
    }
    f.render_widget(
        Chart::new(datasets)
            .block(panel(" Sequence NLL · saved optimizer history "))
            .x_axis(
                Axis::default()
                    .title("optimizer step")
                    .bounds([0., xmax])
                    .labels(["0".to_string(), format!("{xmax:.0}")])
                    .style(Style::default().fg(MUTED)),
            )
            .y_axis(
                Axis::default()
                    .bounds([(ymin - padding).max(0.), ymax + padding])
                    .labels([format!("{ymin:.2}"), format!("{ymax:.2}")])
                    .style(Style::default().fg(MUTED)),
            ),
        area,
    );
}
fn neural_lines(v: &Value) -> Vec<Line<'static>> {
    if v["encoding"].as_bool() == Some(true) {
        return vec![
            Line::styled(
                phase(v).to_string(),
                Style::default().fg(Color::Cyan).bold(),
            ),
            Line::raw(""),
            label("GPU workers", "8"),
            label(
                "Cut encodings saved",
                format!("{} / {}", number(v, "completed"), number(v, "total")),
            ),
            label(
                "CPU allocation",
                if v["persistent"] == true {
                    "120 CPUs · 4 host CPUs reserved"
                } else {
                    "16 CPUs"
                },
            ),
            label("Storage", "pooled vectors + hash receipts"),
            label("Cut data", "source pointers; memory views"),
            label("Telemetry", status_age(v)),
            Line::raw("Frozen pretrained encoders. No final-test targets enter training context."),
        ];
    }
    vec![
        Line::styled(
            phase(v).to_string(),
            Style::default().fg(Color::Cyan).bold(),
        ),
        Line::raw(""),
        label("Telemetry", status_age(v)),
        label("Synchronized ranks", format!("{} / 8", number(v, "world"))),
        label(
            "Optimizer steps",
            format!("{} / {}", number(v, "step"), number(v, "steps")),
        ),
        label(
            "Global / micro batch",
            format!(
                "{} / {}",
                number(v, "global_batch"),
                number(v, "microbatch")
            ),
        ),
        label("Loss", decimal(v, "loss")),
        label(
            "NLL / pairwise rank",
            format!(
                "{} / {}",
                decimal(v, "sequence_nll"),
                decimal(v, "ranking_bt")
            ),
        ),
        label(
            "Step / reduction (s)",
            format!(
                "{} / {}",
                decimal(v, "step_seconds"),
                decimal(v, "communication_seconds")
            ),
        ),
        label(
            "Examples / second",
            if v["end_to_end_examples_per_second"].is_number() {
                format!(
                    "{} end-to-end",
                    decimal(v, "end_to_end_examples_per_second")
                )
            } else {
                format!("{} · step timing", decimal(v, "examples_per_second"))
            },
        ),
        label(
            "Corpus eligibility",
            if v["corpus_eligible"] == true {
                "qualified"
            } else {
                "qualification only"
            },
        ),
    ]
}
#[derive(Default)]
pub struct History {
    gpu: [VecDeque<u64>; 8],
    last: u64,
    worker_offset: usize,
    pub compact_offsets: [u16; 5],
    pub learning: std::sync::Arc<crate::learning_history::Snapshot>,
    pub learning_navigation: crate::learning_history::Navigation,
    pub learning_details: bool,
}
impl History {
    fn update(&mut self, h: &Hardware) {
        if h.timestamp == self.last {
            return;
        }
        self.last = h.timestamp;
        for g in &h.gpus {
            if let Some(history) = self.gpu.get_mut(g.index) {
                history.push_back(g.utilization as u64);
                if history.len() > 120 {
                    history.pop_front();
                }
            }
        }
    }
}
fn compact_layout(area: Rect) -> bool {
    area.width < 100 || area.height < 40
}

// Reflow whole records before scrolling so narrow terminals retain every field,
// including long paths. Full layouts retain their existing tables and charts.
fn compact_wrap(lines: Vec<Line<'static>>, width: u16) -> Vec<Line<'static>> {
    let mut output = Vec::new();
    for line in lines {
        let text = line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        for paragraph in text.split('\n') {
            let mut current = String::new();
            let mut used = 0;
            for word in paragraph.split_whitespace() {
                let word_width = Span::raw(word).width();
                if used > 0 && used + 1 + word_width > width as usize {
                    output.push(Line::raw(std::mem::take(&mut current)));
                    used = 0;
                }
                if used > 0 {
                    current.push(' ');
                    used += 1;
                }
                for ch in word.chars() {
                    let size = Span::raw(ch.to_string()).width();
                    if used + size > width as usize && !current.is_empty() {
                        output.push(Line::raw(std::mem::take(&mut current)));
                        used = 0;
                    }
                    current.push(ch);
                    used += size;
                }
            }
            output.push(Line::raw(current));
        }
    }
    output
}
/// Observed search, training and run evidence rendered together.
#[derive(Clone, Copy)]
pub struct DashboardData<'a> {
    pub gp: &'a Value,
    pub neural: &'a Value,
    pub run: &'a Value,
    pub events: &'a str,
    pub history: &'a History,
}

fn compact_lines(
    view: usize,
    h: &Hardware,
    data: DashboardData<'_>,
    width: u16,
) -> Vec<Line<'static>> {
    let DashboardData {
        gp,
        neural,
        run,
        events,
        history,
    } = data;

    let policy = if run["policy"].is_object() {
        &run["policy"]
    } else {
        neural
    };
    let (status, detail) = policy_task_state(run, policy);
    let mut lines = Vec::new();
    match view {
        0 => {
            lines.push(Line::raw(format!("Policy optimization: {status}")));
            lines.push(Line::raw(detail));
            for row in workflow_rows(run, gp, neural) {
                if row.name == "Policy optimization" {
                    continue;
                }
                lines.push(Line::raw(format!(
                    "{}: {} · {}",
                    row.name, row.status, row.progress
                )));
                lines.push(Line::raw(row.detail));
            }
            lines.extend(milestone_lines(run, gp, neural));
            lines.extend(teacher_lines(run));
            lines.extend(comparison_lines(run));
            lines.push(Line::raw(gpu_reason(run, neural, h)));
        }
        1 => {
            let fresh =
                h.timestamp > 0 && h.timestamp <= now() && now().saturating_sub(h.timestamp) <= 3;
            lines.push(Line::raw(if fresh {
                "Hardware: current observation"
            } else {
                "Hardware: STALE / UNAVAILABLE"
            }));
            lines.push(Line::raw(format!(
                "GPU devices observed: {} / 8",
                h.gpus.len()
            )));
            for g in &h.gpus {
                lines.push(Line::raw(format!(
                    "GPU {}: {:.1}% · {:.1}/{:.1} GiB · {:.0} W · {:.0} C",
                    g.index,
                    g.utilization,
                    g.memory_mb / 1024.,
                    g.memory_total_mb / 1024.,
                    g.watts,
                    g.temperature
                )));
            }
            lines.push(Line::raw(format!(
                "RAM {:.1}/{:.1} GiB; available {:.1}; disk free {:.1} GiB",
                h.ram_used_gb, h.ram_total_gb, h.available_gb, h.disk_free_gb
            )));
            lines.push(Line::raw(format!(
                "CPU total {:.1}% · load {}",
                h.cpu_total,
                h.load.trim()
            )));
            for (cpu, activity) in h.cpu.iter().enumerate() {
                lines.push(Line::raw(format!("CPU {cpu}: {activity:.1}%")));
            }
            for numa in &h.numa {
                lines.push(Line::raw(format!(
                    "NUMA {}: {:.1}% [{}]",
                    numa.node, numa.activity, numa.cpus
                )));
            }
        }
        2 => {
            lines.push(Line::raw(format!("Policy optimization: {status}")));
            lines.push(Line::raw(detail));
            lines.push(Line::raw("Training fit is not generalization."));
            lines.push(Line::raw("Validation is adaptive; compare the same panel."));
            lines.push(Line::raw(format!(
                "Saved checkpoint: {}",
                run["selected_checkpoint"]["checkpoint"]
                    .as_str()
                    .unwrap_or("unavailable")
            )));
            lines.push(Line::raw(format!(
                "Full saved history: {} runs · {} optimizer records",
                history.learning.runs.len(),
                history
                    .learning
                    .runs
                    .iter()
                    .map(|run| run.points.len())
                    .sum::<usize>()
            )));
            // Latest losses before the per-run list, so they stay in the first visible rows.
            let (summary, notice) = crate::learning_history::latest_summary(&history.learning);
            lines.extend(summary.into_iter().map(Line::raw));
            lines.push(Line::raw(notice));
            for saved in &history.learning.runs {
                lines.push(Line::raw(format!(
                    "{}: {} · {} records · {} attempts",
                    saved.path.display(),
                    saved.status,
                    saved.points.len(),
                    saved.attempts
                )));
                if let Some(point) = saved.points.last() {
                    lines.push(Line::raw(format!(
                        "Saved step {} · training NLL {:.5}",
                        point.step, point.train
                    )));
                }
            }
            lines.extend(history.learning.errors.iter().cloned().map(Line::raw));
            if history.learning_details {
                lines.extend(neural_lines(policy));
            }
            lines.push(Line::raw(
                "Enlarge for full-history charts; d toggles optimizer details.",
            ));
        }
        3 => {
            lines.push(Line::raw(format!(
                "GP search: {} · {}",
                phase(gp),
                status_age(gp)
            )));
            lines.push(Line::raw(format!(
                "{} active · {} queued · {} completed · {} failed",
                decimal_count(gp, "active"),
                number(gp, "queued"),
                number(gp, "completed_views"),
                number(gp, "failed_views")
            )));
            for worker in gp["active_rows"].as_array().into_iter().flatten() {
                lines.push(Line::raw(format!(
                    "Worker {} · {} · cut {} · {} evaluations · RMSE {} · {} s",
                    number(worker, "slot"),
                    worker["dataset"].as_str().unwrap_or("unknown"),
                    number(worker, "cut"),
                    number(worker, "evaluations"),
                    rmse(worker, "best_rmse"),
                    decimal(worker, "elapsed_seconds")
                )));
            }
            lines.extend(teacher_evolution_lines(run));
            lines.extend(comparison_lines(run));
        }
        _ => {
            lines.push(Line::raw(format!(
                "Campaign: {}",
                goal_phase(run, neural, gp)
            )));
            lines.push(Line::raw(format!(
                "{} saved contexts · {} families",
                number(&run["contexts"], "completed"),
                number(&run["contexts"], "families")
            )));
            lines.push(Line::raw(
                "Final tests stay sealed. Displayed validation is adaptive.",
            ));
            lines.push(Line::raw("Recent events (newest first):"));
            lines.extend(
                events
                    .lines()
                    .rev()
                    .take(30)
                    .map(|line| Line::raw(line.to_owned())),
            );
        }
    }
    lines.extend(
        h.errors
            .iter()
            .map(|error| Line::raw(format!("Hardware error: {error}"))),
    );
    compact_wrap(lines, width)
}
fn render_compact(f: &mut Frame, view: usize, h: &Hardware, data: DashboardData<'_>) {
    let DashboardData {
        gp,
        neural,
        run,
        events,
        history,
    } = data;

    let areas = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(1),
    ])
    .split(f.area());
    let title = [
        "Overview",
        "Hardware",
        "Deep learning",
        "GP search",
        "Data / Events",
    ][view.min(4)];
    f.render_widget(
        Paragraph::new(format!("JOPE / PARALLEL · {title}"))
            .style(Style::default().fg(BLUE).bold()),
        areas[0],
    );
    f.render_widget(
        Paragraph::new("1 Overview 2 HW 3 DL 4 GP 5 Events"),
        areas[1],
    );
    let lines = compact_lines(
        view,
        h,
        crate::dashboard::DashboardData {
            gp,
            neural,
            run,
            events,
            history,
        },
        areas[2].width,
    );
    let maximum = lines
        .len()
        .saturating_sub(areas[2].height as usize)
        .min(u16::MAX as usize) as u16;
    let offset = history.compact_offsets[view.min(4)].min(maximum);
    let count = lines.len();
    f.render_widget(Paragraph::new(lines).scroll((offset, 0)), areas[2]);
    f.render_widget(
        Paragraph::new(format!(
            "↑↓ scroll {}/{} · ←→ tabs · q detach",
            u32::from(offset) + 1,
            count
        )),
        areas[3],
    );
}

pub fn render(f: &mut Frame, view: usize, h: &Hardware, data: DashboardData<'_>) {
    let DashboardData {
        gp,
        neural,
        run,
        events,
        history,
    } = data;

    let observed_gp = crate::telemetry::observed_status(gp, now());
    let gp = &observed_gp;
    f.render_widget(
        Block::default().style(Style::default().bg(Color::Rgb(9, 17, 28)).fg(INK)),
        f.area(),
    );
    if f.area().width < 48 || f.area().height < 15 {
        f.render_widget(
            Paragraph::new(
                "JOPE / PARALLEL\nResize to at least 48 × 15.\nq detaches; services keep running.",
            )
            .block(panel(" Terminal size ")),
            f.area(),
        );
        return;
    }
    if compact_layout(f.area()) {
        render_compact(
            f,
            view,
            h,
            crate::dashboard::DashboardData {
                gp,
                neural,
                run,
                events,
                history,
            },
        );
        return;
    }
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Min(1),
        Constraint::Length(1),
    ])
    .split(f.area());
    let stale = h.timestamp == 0 || now().saturating_sub(h.timestamp) > 3;
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" JOPE / PARALLEL  ", Style::default().fg(BLUE).bold()),
            Span::raw(format!(
                "8 × A100  ·  {} CPUs  ·  {}",
                h.cpu.len(),
                goal_phase(run, neural, gp)
            )),
        ]))
        .block(panel(
            " LEARNING OPERATIONS · live receipts, saved evidence ",
        )),
        areas[0],
    );
    f.render_widget(
        Tabs::new([
            "1 Overview",
            "2 Hardware",
            "3 Deep learning",
            "4 GP search",
            "5 Data / Events",
        ])
        .select(view)
        .highlight_style(Style::default().fg(Color::Rgb(9, 17, 28)).bg(BLUE).bold())
        .style(Style::default().fg(MUTED))
        .block(panel(" Navigate ")),
        areas[1],
    );
    let body = areas[2];
    match view {
        0 => {
            let rows = Layout::vertical([
                Constraint::Length(3),
                Constraint::Length(
                    7 + u16::from(run["encoding_pipeline"]["phase"].is_string())
                        + u16::from(run["campaign_service"]["phase"].is_string())
                        + u16::from(run["selected_checkpoint"]["checkpoint"].is_string()),
                ),
                Constraint::Length((workflow_rows(run, gp, neural).len() + 3) as u16),
                Constraint::Min(1),
                Constraint::Length(4),
            ])
            .split(body);
            let cards = Layout::horizontal([Constraint::Ratio(1, 3); 3]).split(rows[0]);
            let gpu =
                h.gpus.iter().map(|g| g.utilization).sum::<f64>() / h.gpus.len().max(1) as f64;
            for (area, title, value, color, available) in [
                (
                    cards[0],
                    " GPU activity ",
                    gpu,
                    Color::Cyan,
                    h.gpus.len() == 8,
                ),
                (
                    cards[1],
                    " CPU activity ",
                    h.cpu_total,
                    Color::Green,
                    !h.cpu.is_empty(),
                ),
                (
                    cards[2],
                    " Memory used ",
                    100.0 * h.ram_used_gb / h.ram_total_gb.max(1.0),
                    Color::Magenta,
                    h.ram_total_gb > 0.,
                ),
            ] {
                f.render_widget(
                    Gauge::default()
                        .block(panel(title))
                        .ratio((value / 100.0).clamp(0.0, 1.0))
                        .label(if stale || !available {
                            "unavailable".into()
                        } else {
                            format!("{value:.1}%")
                        })
                        .gauge_style(Style::default().fg(if stale || !available {
                            Color::DarkGray
                        } else {
                            color
                        })),
                    area,
                );
            }
            f.render_widget(
                Paragraph::new(milestone_lines(run, gp, neural))
                    .block(panel(" Progress toward a complete learning cycle ")),
                rows[1],
            );
            f.render_widget(
                Table::new(
                    workflow_rows(run, gp, neural)
                        .into_iter()
                        .map(TaskRow::table_row),
                    [
                        Constraint::Length(24),
                        Constraint::Length(10),
                        Constraint::Length(29),
                        Constraint::Min(15),
                    ],
                )
                .header(
                    Row::new(["WORKFLOW", "STATE", "ACTUAL PROGRESS", "NEXT / EVIDENCE"])
                        .style(Style::default().fg(MUTED)),
                )
                .column_spacing(1)
                .block(panel(" Task board · all independent work ")),
                rows[2],
            );
            let lanes = Layout::horizontal([Constraint::Ratio(1, 2); 2]).split(rows[3]);
            f.render_widget(
                Paragraph::new(teacher_lines(run))
                    .wrap(Wrap { trim: true })
                    .block(panel(" Teacher strength and family coverage ")),
                lanes[0],
            );
            f.render_widget(
                Paragraph::new(comparison_lines(run))
                    .wrap(Wrap { trim: true })
                    .block(panel(" Policy learning · completed comparisons ")),
                lanes[1],
            );
            f.render_widget(Paragraph::new(format!("{}\nRAM available {:.0} GiB · disk free {:.0} GiB · 4 CPUs reserved for host services",gpu_reason(run,neural,h),h.available_gb,h.disk_free_gb)).wrap(Wrap{trim:true}).block(panel(" Resource assignment / next work ")),rows[4]);
        }
        1 => {
            let parts = Layout::vertical([
                Constraint::Length(11),
                Constraint::Min(1),
                Constraint::Length(5),
            ])
            .split(body);
            let rows = (0..8).map(|i| {
                if let Some(g) = h.gpus.iter().find(|g| g.index == i) {
                    Row::new(vec![
                        format!("A100 / {i}"),
                        format!("{:5.1}%", g.utilization),
                        format!(
                            "{:.1} / {:.0} GiB",
                            g.memory_mb / 1024.0,
                            g.memory_total_mb / 1024.0
                        ),
                        format!("{:.0} W", g.watts),
                        format!("{:.0} °C", g.temperature),
                        if stale { "STALE".into() } else { "live".into() },
                    ])
                    .style(Style::default().fg(if stale {
                        GOLD
                    } else if g.utilization >= 85. {
                        GREEN
                    } else if g.utilization >= 20. {
                        BLUE
                    } else {
                        MUTED
                    }))
                } else {
                    Row::new(vec![
                        format!("A100 / {i}"),
                        "—".into(),
                        "—".into(),
                        "—".into(),
                        "—".into(),
                        "unavailable".into(),
                    ])
                }
            });
            f.render_widget(
                Table::new(
                    rows,
                    [
                        Constraint::Length(12),
                        Constraint::Length(10),
                        Constraint::Length(20),
                        Constraint::Length(9),
                        Constraint::Length(9),
                        Constraint::Min(8),
                    ],
                )
                .header(
                    Row::new(["Device", "Activity", "VRAM", "Power", "Temp", "Telemetry"])
                        .style(Style::default().fg(Color::Cyan)),
                )
                .block(panel(" All eight GPUs ")),
                parts[0],
            );
            let bands =
                Layout::horizontal([Constraint::Percentage(36), Constraint::Percentage(64)])
                    .split(parts[1]);
            let plots = Layout::vertical([Constraint::Ratio(1, 8); 8]).split(bands[0]);
            for i in 0..8 {
                f.render_widget(
                    Sparkline::default()
                        .data(history.gpu[i].iter().copied().collect::<Vec<_>>())
                        .max(100)
                        .style(Style::default().fg(Color::Cyan))
                        .block(Block::default().title(format!("GPU {i} · 120s"))),
                    plots[i],
                );
            }
            let per_row = ((bands[1].width.saturating_sub(2)) as usize / 9).max(1);
            let lines = h
                .cpu
                .chunks(per_row)
                .enumerate()
                .map(|(r, chunk)| {
                    Line::from(
                        chunk
                            .iter()
                            .enumerate()
                            .map(|(c, &usage)| {
                                Span::styled(
                                    format!("{:03}:{:3.0}% ", r * per_row + c, usage),
                                    Style::default()
                                        .bg(if usage > 85.0 {
                                            Color::Rgb(24, 91, 80)
                                        } else if usage > 30.0 {
                                            Color::Rgb(51, 65, 78)
                                        } else {
                                            Color::Rgb(17, 30, 43)
                                        })
                                        .fg(if usage > 85. {
                                            GREEN
                                        } else if usage > 30. {
                                            GOLD
                                        } else {
                                            MUTED
                                        }),
                                )
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            f.render_widget(
                Paragraph::new(lines).block(panel(" Every logical CPU ")),
                bands[1],
            );
            let numa = if h.numa.is_empty() {
                "NUMA activity unavailable".into()
            } else {
                h.numa
                    .iter()
                    .map(|n| format!("NUMA {}: {:.1}% [{}]", n.node, n.activity, n.cpus))
                    .collect::<Vec<_>>()
                    .join(" · ")
            };
            f.render_widget(Paragraph::new(format!("RAM {:.1} / {:.1} GiB · available {:.1} GiB · disk free {:.0} GiB\n{}\nDisk read/write {} / {} MiB/s · load {}",h.ram_used_gb,h.ram_total_gb,h.available_gb,h.disk_free_gb,numa,h.disk_read_mib_s.map(|v|format!("{v:.1}")).unwrap_or_else(||"—".into()),h.disk_write_mib_s.map(|v|format!("{v:.1}")).unwrap_or_else(||"—".into()),h.load.trim())).block(panel(" Host resources ")),parts[2]);
        }
        2 if !history.learning_details => {
            crate::learning_history::render(
                f,
                body,
                &history.learning,
                &history.learning_navigation,
                run,
            );
        }
        2 => {
            let policy = if run["policy"].is_object() {
                &run["policy"]
            } else {
                neural
            };
            let parts = Layout::vertical([
                Constraint::Length(18),
                Constraint::Length(3),
                Constraint::Min(7),
                Constraint::Length(5),
            ])
            .split(body);
            let top = Layout::horizontal([Constraint::Percentage(49), Constraint::Percentage(51)])
                .split(parts[0]);
            let mut lines = neural_lines(policy);
            let (status, detail) = policy_task_state(run, policy);
            lines[0] = Line::raw(format!("Policy optimization: {status}"));
            lines.insert(1, Line::raw(detail));
            lines.push(label(
                "Trained / val families",
                format!(
                    "{} / {}",
                    number(policy, "training_families"),
                    number(&policy["validation"], "source_families")
                ),
            ));
            lines.push(label(
                "Initialization",
                policy["initialization"]["kind"]
                    .as_str()
                    .unwrap_or("unavailable"),
            ));
            lines.push(label(
                "Validation step",
                policy["validation"]["evaluated_step"]
                    .as_u64()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "unbound legacy step · no curve plotted".into()),
            ));
            lines.push(label(
                "Elapsed",
                policy["elapsed_seconds"]
                    .as_f64()
                    .map(|v| duration(v as u64))
                    .unwrap_or_else(|| "—".into()),
            ));
            f.render_widget(
                Paragraph::new(lines)
                    .wrap(Wrap { trim: true })
                    .block(panel(" Policy optimizer · BF16 / FP32 AdamW ")),
                top[0],
            );
            learning_chart(f, top[1], run);
            let done = number(policy, "step");
            let total = number(policy, "steps");
            f.render_widget(
                Gauge::default()
                    .ratio((done as f64 / total.max(1) as f64).clamp(0., 1.))
                    .label(format!("{done} / {total} steps · {}", status_age(policy)))
                    .gauge_style(Style::default().fg(VIOLET))
                    .block(panel(if completed(policy) {
                        " Saved policy block "
                    } else {
                        " Current training block "
                    })),
                parts[1],
            );
            let rows = run["checkpoints"]
                .as_array()
                .into_iter()
                .flatten()
                .rev()
                .map(|v| {
                    let m = &v["metrics"];
                    let origin = m["initialization"]["kind"].as_str().unwrap_or("—");
                    Row::new(vec![
                        v["name"].as_str().unwrap_or("checkpoint").to_owned(),
                        if v["complete"] == true {
                            "saved".into()
                        } else {
                            "active".into()
                        },
                        origin.into(),
                        number(v, "step").to_string(),
                        number(m, "training_families").to_string(),
                        decimal(m, "sequence_nll"),
                        decimal(&m["validation"], "sequence_nll"),
                        decimal(&m["validation"], "ranking_bt"),
                    ])
                });
            f.render_widget(
                Table::new(
                    rows,
                    [
                        Constraint::Min(29),
                        Constraint::Length(8),
                        Constraint::Length(12),
                        Constraint::Length(7),
                        Constraint::Length(9),
                        Constraint::Length(11),
                        Constraint::Length(11),
                        Constraint::Length(11),
                    ],
                )
                .header(
                    Row::new([
                        "MODEL / TRAINING BLOCK",
                        "STATE",
                        "LINEAGE",
                        "STEP",
                        "FAMILIES",
                        "TRAIN NLL",
                        "VAL NLL",
                        "VAL RANK",
                    ])
                    .style(Style::default().fg(VIOLET)),
                )
                .block(panel(" Saved checkpoint history · durable model lineage ")),
                parts[2],
            );
            f.render_widget(Paragraph::new(vec![label("Saved checkpoint",run["selected_checkpoint"]["checkpoint"].as_str().or_else(||policy["checkpoint"].as_str()).or_else(||run["checkpoint"]["checkpoint"].as_str()).unwrap_or("awaiting complete policy receipt")),
                Line::styled("Adaptive teaching rewards optimize each cut's held-out targets; final tests stay sealed.",Style::default().fg(GOLD)),
                Line::raw("Lower imitation loss is learning evidence; matched GP comparisons measure policy value.")]).wrap(Wrap{trim:true}).block(panel(" Learning meaning and durable state ")),parts[3]);
        }
        3 => {
            let parts = Layout::vertical([
                Constraint::Length(5),
                Constraint::Min(8),
                Constraint::Length(14),
            ])
            .split(body);
            f.render_widget(Paragraph::new(format!("{} activity · {} active jobs · {} queued\n{} completed · {} failed · {} committed evaluations\n{} eval/s · {}",gp["current_activity"].as_str().unwrap_or("unknown"),decimal_count(gp,"active"),number(gp,"queued"),number(gp,"completed_views"),number(gp,"failed_views"),number(gp,"completed_evaluations"),decimal(gp,"evaluations_per_second"),age(gp))).block(panel(" GP search jobs ")),parts[0]);
            let rows = gp["active_rows"]
                .as_array()
                .into_iter()
                .flatten()
                .skip(history.worker_offset)
                .map(|v| {
                    Row::new(vec![
                        number(v, "slot").to_string(),
                        v["dataset"]
                            .as_str()
                            .map(|id| run["dataset_names"][id].as_str().unwrap_or(id))
                            .unwrap_or("—")
                            .to_string(),
                        number(v, "cut").to_string(),
                        number(v, "evaluations").to_string(),
                        v["generation"]
                            .as_u64()
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "—".into()),
                        rmse(v, "best_rmse"),
                        decimal(v, "elapsed_seconds"),
                    ])
                });
            f.render_widget(
                Table::new(
                    rows,
                    [
                        Constraint::Length(7),
                        Constraint::Min(30),
                        Constraint::Length(7),
                        Constraint::Length(13),
                        Constraint::Length(9),
                        Constraint::Length(18),
                        Constraint::Min(12),
                    ],
                )
                .header(
                    Row::new([
                        "Worker",
                        "Dataset",
                        "Cut",
                        "Evaluations",
                        "Gen",
                        "Best RMSE orig",
                        "Seconds",
                    ])
                    .style(Style::default().fg(Color::Green)),
                )
                .block(panel(" Adaptive GP teacher searches · ↑ ↓ scroll workers ")),
                parts[1],
            );
            let evidence =
                Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(parts[2]);
            f.render_widget(
                Paragraph::new(teacher_evolution_lines(run))
                    .wrap(Wrap { trim: true })
                    .block(panel(" Best teacher progress · no optimum claim ")),
                evidence[0],
            );
            let mut lines = comparison_lines(run);
            if let Some(report) = run["reports"].as_array().and_then(|v| v.last()) {
                let g = &report["generation"]["neural"];
                lines.push(label(
                    "Neural proposal validity",
                    format!(
                        "{} / {} · {} unique · {} duplicates",
                        number(g, "valid"),
                        number(g, "attempted"),
                        number(g, "unique_neural"),
                        number(g, "duplicates")
                    ),
                ));
            }
            f.render_widget(
                Paragraph::new(lines)
                    .wrap(Wrap { trim: true })
                    .block(panel(" Matched evaluations · NN / generic / shuffled ")),
                evidence[1],
            );
        }
        _ => {
            let mut lines = vec![
                label("Campaign", goal_phase(run,neural,gp)),
                label(
                    "Pilot launched",
                    if run["pilot_started"] == true {
                        "yes"
                    } else {
                        "no"
                    },
                ),
                label("Context coverage",format!("{} saved views; {} source families",number(&run["contexts"],"completed"),number(&run["contexts"],"families"))),
                label("Prepared workload",format!("{} distinct cut problems from {} datasets",number(&run["preparation"],"views"),number(&run["preparation"],"datasets"))),
                label("Cut storage","source pointers + exact indices + preprocessing recipes"),
                label("Final tests","sealed; teaching held-out scores are adaptive"),
                label("Family preparation", format!("train {} / 48 · validation {} / 8 · test {} / 8 eligible families", number(&run["family_readiness"]["families_with_verified_preparation_candidates"],"meta_train"), number(&run["family_readiness"]["families_with_verified_preparation_candidates"],"meta_validation"), number(&run["family_readiness"]["families_with_verified_preparation_candidates"],"meta_test"))),
                label("Encoding pipeline", format!("{} · {}", phase(&run["encoding_pipeline"]), status_age(&run["encoding_pipeline"]))),
                label("Next dependency",run["encoding_pipeline"]["reason"].as_str().unwrap_or("No queued encoding work receipt")),
                label("Qualification","complete real-data inventory and final evaluation remain pilot gates"),
                Line::raw(run["cycle"]["reason"].as_str().unwrap_or("All displayed progress comes from saved manifests and live worker receipts.").to_string()),
                Line::raw(""),
            ];
            lines.extend(
                events
                    .lines()
                    .rev()
                    .take(30)
                    .map(|s| Line::raw(s.to_string())),
            );
            if run["binary_qualification"].is_object() {
                let binary = &run["binary_qualification"];
                let detail = if run["binary_admitted_contexts"].is_number() {
                    binary_replay_detail(run)
                } else {
                    format!(
                        "{} / {} saved · {}",
                        decimal_count(binary, "completed"),
                        decimal_count(binary, "total"),
                        phase(binary)
                    )
                };
                lines.insert(0, label("Binary contexts", detail));
                lines.insert(1,label("Binary input scope","training rows only · original source references · frozen regression projection qualification"));
                lines.insert(
                    2,
                    label(
                        "Binary limitation",
                        "Sonar acquisition grouping unresolved; duplicate groups preserved",
                    ),
                );
            }
            lines.extend(
                h.errors
                    .iter()
                    .map(|s| Line::styled(s.clone(), Style::default().fg(Color::Yellow))),
            );
            f.render_widget(
                Paragraph::new(lines)
                    .wrap(Wrap { trim: true })
                    .block(panel(" Provenance and recent events ")),
                body,
            );
        }
    }
    f.render_widget(
        Paragraph::new(if stale {
            " TELEMETRY STALE / UNAVAILABLE · q detaches; workers keep running"
        } else if h.gpus.len() != 8 || h.cpu.is_empty() {
            " PARTIAL TELEMETRY · inspect Hardware / Events · q detaches; workers keep running"
        } else {
            " LIVE · 1 Hz telemetry / 4 fps · ← → or 1–5 navigate · q detaches"
        })
        .style(Style::default().fg(if stale {
            Color::Yellow
        } else {
            Color::DarkGray
        })),
        areas[3],
    );
}
pub fn run(root: &Path) -> Result<()> {
    run_with_manifest(root, None)
}

pub fn run_with_manifest(root: &Path, experiment_manifest: Option<&Path>) -> Result<()> {
    use crossterm::{
        event::{self, Event, KeyCode},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    };
    enable_raw_mode()?;
    execute!(std::io::stdout(), EnterAlternateScreen)?;
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = disable_raw_mode();
            let _ = crossterm::execute!(std::io::stdout(), LeaveAlternateScreen);
        }
    }
    let _restore = Restore;
    let mut terminal = Terminal::new(CrosstermBackend::new(std::io::stdout()))?;
    let mut view = 0;
    let mut history = History::default();
    let learning = crate::learning_history::Watcher::start(root.to_path_buf());
    let mut run = campaign_with_manifest(root, experiment_manifest)?;
    let mut campaign_updated = Instant::now();
    loop {
        if let Some(snapshot) = learning.latest() {
            history.learning = snapshot;
        }
        let h: Hardware = read_json(&root.join("runs/hardware.json")).unwrap_or_default();
        history.update(&h);
        let gp =
            crate::telemetry::observed_status(&state(&root.join("runs/gp-service.json")), now());
        let mut neural = state(&root.join("runs/neural-service.json"));
        let encoding = state(&root.join("runs/encoding-service.json"));
        if number(&encoding, "timestamp") > number(&neural, "timestamp") {
            neural = encoding;
            neural["encoding"] = Value::Bool(true);
        }
        if campaign_updated.elapsed() >= Duration::from_secs(5) {
            run = campaign_with_manifest(root, experiment_manifest)?;
            campaign_updated = Instant::now();
        }
        let events = tail(&root.join("runs/events.log"), 64 * 1024);
        terminal.draw(|f| {
            render(
                f,
                view,
                &h,
                crate::dashboard::DashboardData {
                    gp: &gp,
                    neural: &neural,
                    run: &run,
                    events: &events,
                    history: &history,
                },
            )
        })?;
        if event::poll(Duration::from_millis(250))? {
            if let Event::Key(key) = event::read()? {
                let size = terminal.size()?;
                if compact_layout(Rect::new(0, 0, size.width, size.height)) {
                    let visible = size.height.saturating_sub(3) as usize;
                    let maximum = compact_lines(
                        view,
                        &h,
                        crate::dashboard::DashboardData {
                            gp: &gp,
                            neural: &neural,
                            run: &run,
                            events: &events,
                            history: &history,
                        },
                        size.width,
                    )
                    .len()
                    .saturating_sub(visible)
                    .min(u16::MAX as usize) as u16;
                    let current = history.compact_offsets[view].min(maximum);
                    let selected = match key.code {
                        KeyCode::Down => Some(current.saturating_add(1)),
                        KeyCode::Up => Some(current.saturating_sub(1)),
                        KeyCode::PageDown => Some(current.saturating_add(visible as u16)),
                        KeyCode::PageUp => Some(current.saturating_sub(visible as u16)),
                        KeyCode::Home => Some(0),
                        KeyCode::End => Some(maximum),
                        _ => None,
                    };
                    if let Some(selected) = selected {
                        history.compact_offsets[view] = selected.min(maximum);
                        continue;
                    }
                }
                match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char(c @ '1'..='5') => view = c as usize - '1' as usize,
                    KeyCode::Char('d') if view == 2 => {
                        history.learning_details = !history.learning_details
                    }
                    KeyCode::Char('a') if view == 2 => {
                        history.learning_details = false;
                        history.learning_navigation.selected = None;
                    }
                    KeyCode::Down if view == 2 => {
                        history.learning_navigation.move_by(&history.learning, 1)
                    }
                    KeyCode::Up if view == 2 => {
                        history.learning_navigation.move_by(&history.learning, -1)
                    }
                    KeyCode::PageDown if view == 2 => {
                        history.learning_navigation.move_by(&history.learning, 10)
                    }
                    KeyCode::PageUp if view == 2 => {
                        history.learning_navigation.move_by(&history.learning, -10)
                    }
                    KeyCode::Home if view == 2 => {
                        history.learning_navigation.edge(&history.learning, false)
                    }
                    KeyCode::End if view == 2 => {
                        history.learning_navigation.edge(&history.learning, true)
                    }
                    KeyCode::Down if view == 3 => {
                        history.worker_offset = (history.worker_offset + 1)
                            .min(number(&gp, "active").saturating_sub(1) as usize)
                    }
                    KeyCode::Up if view == 3 => {
                        history.worker_offset = history.worker_offset.saturating_sub(1)
                    }
                    KeyCode::PageDown if view == 3 => {
                        history.worker_offset = (history.worker_offset + 20)
                            .min(number(&gp, "active").saturating_sub(1) as usize)
                    }
                    KeyCode::PageUp if view == 3 => {
                        history.worker_offset = history.worker_offset.saturating_sub(20)
                    }
                    KeyCode::Right => view = (view + 1) % 5,
                    KeyCode::Left => view = (view + 4) % 5,
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

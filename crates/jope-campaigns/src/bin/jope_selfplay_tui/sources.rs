use super::*;

pub(super) fn collect_sources(args: &Args) -> RawSources {
    let started = Instant::now();
    let mut raw = RawSources {
        collected_at_ms: now_ms(),
        ..RawSources::default()
    };
    raw.scq = match load_scq_snapshot(&args.scq) {
        Ok(value) => {
            raw.source_states.push(SourceState {
                name: "scq".to_owned(),
                healthy: true,
                note: "snapshot".to_owned(),
            });
            Some(value)
        }
        Err(error) => {
            raw.source_states.push(SourceState {
                name: "scq".to_owned(),
                healthy: false,
                note: format!("{error:#}"),
            });
            None
        }
    };

    let status_path = if args.root.join("ops/status-v3.json").is_file() {
        args.root.join("ops/status-v3.json")
    } else {
        args.root.join("ops/status.json")
    };
    raw.status = read_json_source(&status_path, "status", true, &mut raw.source_states);
    raw.queue = read_json_source(
        &args.root.join("ops/queue-keeper-v3.json"),
        "queue-v3",
        true,
        &mut raw.source_states,
    );
    raw.freshness = read_json_source(
        &args.root.join("ops/freshness-status.json"),
        "freshness",
        true,
        &mut raw.source_states,
    );
    raw.supply = read_json_source(
        &args.root.join("ops/micro-wave-supply-v2.json"),
        "supply",
        true,
        &mut raw.source_states,
    );
    raw.supply_breaker = read_json_source(
        &args.root.join("ops/cohort-supply-circuit-breaker.v1.json"),
        "supply-breaker",
        true,
        &mut raw.source_states,
    );
    raw.trainer_controller = read_json_source(
        &args.root.join("ops/task-learning-controller.v2.json"),
        "trainer-controller",
        true,
        &mut raw.source_states,
    );
    raw.proposer_state = read_newest_json(
        &args.root.join("ops/cohort-supply-state-v2"),
        &mut raw.source_states,
    );
    raw.gpu_admission = read_json_source(
        &args.root.join("ops/gpu-lane-admission/status.v1.json"),
        "gpu-admission",
        true,
        &mut raw.source_states,
    );
    raw.utilization = read_json_source(
        &args.root.join("ops/utilization-supervisor-v1.json"),
        "utilization-supervisor",
        true,
        &mut raw.source_states,
    );
    raw.campaign = read_json_source(
        &args.root.join("ops/dlgp-budget-campaign-status.v1.json"),
        "budget-campaign",
        true,
        &mut raw.source_states,
    );
    raw.proposal_summary = read_json_source(
        &args.root.join("proposals/dl-proposals.summary.json"),
        "proposals",
        true,
        &mut raw.source_states,
    );
    let trainer_path = args.root.join("ops/trainer-status.json");
    let trainer_fallback = if trainer_path.is_file() {
        trainer_path
    } else {
        args.root.join("ops/dl-job.json")
    };
    raw.trainer_fallback =
        read_json_source(&trainer_fallback, "trainer", false, &mut raw.source_states);
    raw.flywheel = read_json_source(
        &args.root.join("ops/flywheel-state.json"),
        "flywheel",
        false,
        &mut raw.source_states,
    );
    raw.accepted_by_task = read_json_source(
        &args.root.join("training/accepted-by-task.json"),
        "accepted-by-task",
        false,
        &mut raw.source_states,
    );
    raw.four_task_telemetry = read_four_task_telemetry(
        &args.root.join("ops/four-task-telemetry.v1.json"),
        &mut raw.source_states,
    );
    raw.evidence_bytes = fs::metadata(args.root.join("indexes/gene-evidence.v2.jsonl"))
        .ok()
        .map(|metadata| metadata.len());
    for relative in [
        "ops/freshness-events.v1.jsonl",
        "ops/micro-wave-supply-events.v1.jsonl",
        "ops/controller-events.jsonl",
    ] {
        raw.local_events.extend(tail_json_lines(
            &args.root.join(relative),
            8,
            EVENT_TAIL_BYTES,
        ));
    }
    raw.collection_latency_ms = started.elapsed().as_millis();
    raw
}

pub(super) fn read_four_task_telemetry(
    path: &Path,
    states: &mut Vec<SourceState>,
) -> Option<JopeFourTaskTelemetryV1> {
    let result = fs::read(path)
        .with_context(|| format!("read {}", path.display()))
        .and_then(|bytes| {
            let telemetry: JopeFourTaskTelemetryV1 = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse {}", path.display()))?;
            telemetry.validate()?;
            Ok(telemetry)
        });
    match result {
        Ok(telemetry) => {
            let age_seconds = now_ms().saturating_sub(telemetry.generated_at_ms) / 1_000;
            states.push(SourceState {
                name: "four-task-telemetry".to_owned(),
                healthy: age_seconds <= 45,
                note: if age_seconds <= 45 {
                    format!("fresh {age_seconds}s")
                } else {
                    format!("stale {age_seconds}s; last-known only")
                },
            });
            Some(telemetry)
        }
        Err(error) => {
            states.push(SourceState {
                name: "four-task-telemetry".to_owned(),
                healthy: false,
                note: format!("{error:#}"),
            });
            None
        }
    }
}

pub(super) fn read_json_source(
    path: &Path,
    name: &str,
    required: bool,
    states: &mut Vec<SourceState>,
) -> Option<Value> {
    match fs::read(path) {
        Ok(bytes) => match serde_json::from_slice(&bytes) {
            Ok(value) => {
                states.push(SourceState {
                    name: name.to_owned(),
                    healthy: true,
                    note: path
                        .file_name()
                        .and_then(|value| value.to_str())
                        .unwrap_or("json")
                        .to_owned(),
                });
                Some(value)
            }
            Err(error) => {
                states.push(SourceState {
                    name: name.to_owned(),
                    healthy: false,
                    note: format!("invalid JSON: {error}"),
                });
                None
            }
        },
        Err(error) => {
            if required || error.kind() != io::ErrorKind::NotFound {
                states.push(SourceState {
                    name: name.to_owned(),
                    healthy: false,
                    note: if error.kind() == io::ErrorKind::NotFound {
                        "missing".to_owned()
                    } else {
                        error.to_string()
                    },
                });
            }
            None
        }
    }
}

pub(super) fn read_newest_json(directory: &Path, states: &mut Vec<SourceState>) -> Option<Value> {
    if !directory.is_dir() {
        return None;
    }
    let newest = fs::read_dir(directory)
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().and_then(|value| value.to_str()) == Some("json"))
        .max_by_key(|entry| {
            entry
                .metadata()
                .and_then(|metadata| metadata.modified())
                .ok()
        })?;
    read_json_source(&newest.path(), "proposer-intent", false, states)
}

pub(super) fn load_scq_snapshot(scq: &Path) -> Result<Value> {
    match run_scq_json(scq, &["snapshot"]) {
        Ok(value) if value.get("Snapshot").is_some() => Ok(value),
        primary => {
            let primary_error = primary
                .err()
                .map(|error| format!("{error:#}"))
                .unwrap_or_else(|| "snapshot response lacks Snapshot".to_owned());
            let nodes = run_scq_json(scq, &["nodes"]).with_context(|| {
                format!("SCQ snapshot failed ({primary_error}); nodes fallback")
            })?;
            let jobs = run_scq_json(scq, &["ls"])
                .with_context(|| format!("SCQ snapshot failed ({primary_error}); jobs fallback"))?;
            let priorities = run_scq_json(scq, &["priority", "ls"]).unwrap_or_else(|_| json!({}));
            Ok(json!({
                "Snapshot": {
                    "epoch": null,
                    "ts": null,
                    "nodes": nodes.pointer("/Nodes/nodes").cloned().unwrap_or_else(|| json!([])),
                    "gpus": nodes.pointer("/Nodes/gpus").cloned().unwrap_or_else(|| json!([])),
                    "jobs": jobs.pointer("/Jobs/jobs").cloned().unwrap_or_else(|| json!([])),
                    "families": priorities.pointer("/FamilyPriorities/families")
                        .cloned().unwrap_or_else(|| json!([])),
                    "events": [],
                    "alerts": [format!("snapshot fallback: {primary_error}")],
                    "counters": {},
                }
            }))
        }
    }
}

pub(super) fn run_scq_json(scq: &Path, arguments: &[&str]) -> Result<Value> {
    let output = Command::new(scq)
        .arg("--json")
        .args(arguments)
        .output()
        .with_context(|| format!("run {} --json {}", scq.display(), arguments.join(" ")))?;
    if !output.status.success() {
        bail!(
            "{}",
            String::from_utf8_lossy(if output.stderr.is_empty() {
                &output.stdout
            } else {
                &output.stderr
            })
            .trim()
        );
    }
    serde_json::from_slice(&output.stdout).context("parse SCQ JSON")
}

pub(super) fn tail_json_lines(path: &Path, limit: usize, maximum_bytes: u64) -> Vec<Value> {
    let Ok(mut file) = File::open(path) else {
        return Vec::new();
    };
    let Ok(length) = file.metadata().map(|metadata| metadata.len()) else {
        return Vec::new();
    };
    let start = length.saturating_sub(maximum_bytes);
    if file.seek(SeekFrom::Start(start)).is_err() {
        return Vec::new();
    }
    let mut bytes = Vec::with_capacity((length - start) as usize);
    if file.read_to_end(&mut bytes).is_err() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&bytes);
    let mut lines = text.lines().collect::<Vec<_>>();
    if start > 0 && !lines.is_empty() {
        lines.remove(0);
    }
    lines
        .into_iter()
        .rev()
        .filter_map(|line| serde_json::from_str(line).ok())
        .take(limit)
        .collect()
}

use super::*;

pub(super) struct Guardrails {
    pub(super) alarms_path: PathBuf,
    pub(super) keeper_path: PathBuf,
    pub(super) waves_dir: PathBuf,
    pub(super) disk_paths: Vec<PathBuf>,
}

pub(super) fn newest_wave_age(waves_dir: &Path) -> Option<u64> {
    std::fs::read_dir(waves_dir)
        .ok()?
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("strict-wave-"))
        .filter_map(|e| e.metadata().ok().and_then(|m| m.modified().ok()))
        .max()
        .and_then(|newest| SystemTime::now().duration_since(newest).ok())
        .map(|d| d.as_secs())
}

pub(super) fn disk_use_pct(path: &Path) -> Option<f64> {
    // statvfs via `df` keeps this dependency-free.
    let out = Command::new("df")
        .arg("--output=pcent")
        .arg(path)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .nth(1)?
        .trim()
        .trim_end_matches('%')
        .parse::<f64>()
        .ok()
}

pub(super) fn disk_pause_decision(was_paused: bool, maximum_used_pct: Option<f64>) -> bool {
    match maximum_used_pct {
        Some(used) if was_paused => used >= 80.0,
        Some(used) => used >= 85.0,
        None => was_paused,
    }
}

pub(super) fn disk_hysteresis_block(root: &Path, paths: &[PathBuf]) -> Result<Option<String>> {
    let state_path = root.join("ops/expansion-disk-hysteresis.v1.json");
    let was_paused = state_path
        .is_file()
        .then(|| read_json(&state_path))
        .transpose()?
        .and_then(|value| value.get("paused").and_then(Value::as_bool))
        .unwrap_or(false);
    let measurements = paths
        .iter()
        .filter_map(|path| disk_use_pct(path).map(|used| (path, used)))
        .collect::<Vec<_>>();
    let maximum = measurements
        .iter()
        .map(|(_, used)| *used)
        .max_by(f64::total_cmp);
    let paused = disk_pause_decision(was_paused, maximum);
    atomic_json(
        &state_path,
        &json!({
            "schema": "ExpansionDiskHysteresis.v1",
            "paused": paused,
            "pause_threshold_percent": 85,
            "resume_below_percent": 80,
            "maximum_used_percent": maximum,
            "measurements": measurements
                .iter()
                .map(|(path, used)| json!({"path": path, "used_percent": used}))
                .collect::<Vec<_>>(),
            "updated_at_ms": now_ms(),
        }),
    )?;
    Ok(paused.then(|| match maximum {
        Some(used) => format!("disk expansion pause active at {used:.0}% (resume only below 80%)"),
        None => "disk expansion pause remains active; usage is unavailable".to_owned(),
    }))
}

/// Returns Some(reason) when the cycle must not run tranche work.
pub(super) fn guardrail_block(g: &Guardrails, coverage: Option<f64>) -> Option<String> {
    if let Ok(alarms) = read_json(&g.alarms_path) {
        for alarm in alarms
            .get("alarms")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
        {
            if alarm.get("severity").and_then(Value::as_str) != Some("critical") {
                continue;
            }
            let name = alarm.get("alarm").and_then(Value::as_str).unwrap_or("?");
            // Tranche work materializes caches and rewrites the catalog on the
            // controller only. A remote worker's disk cannot corrupt or stall
            // that, so blocking on it just starves the catalog (xbabe1 at 90%
            // halted expansion for hours). Local disk is checked below.
            if name.starts_with("disk_high:") && !name.ends_with(":xbabe2") {
                continue;
            }
            // Alarms that mean "no work is running" must NOT halt expansion.
            // Activating datasets is what CREATES work, so blocking on these
            // turns any wave-lane defect into total cluster paralysis: waves
            // stall -> wave_stalled fires -> expansion halts -> no new datasets
            // -> the first-touch coverage lane drains to an empty backlog ->
            // every slot goes idle. That exact loop idled all 1,052 slots on
            // 2026-07-28. Block only on conditions expansion itself would make
            // worse; local disk is still checked below.
            if matches!(
                name,
                "wave_stalled" | "no_active_trainer" | "scq_jobs_held_starving"
            ) || name.starts_with("status_stale:ops/task-learning-controller")
                || name.starts_with("status_stale:ops/trainer-watchdog")
            {
                continue;
            }
            return Some(format!("critical alarm: {name}"));
        }
    }
    match file_age_secs(&g.keeper_path) {
        Some(age) if age > 900 => return Some(format!("keeper status stale {age}s")),
        None => return Some("keeper status missing".into()),
        _ => {}
    }
    // Wave age is deliberately NOT a block, for the same reason as wave_stalled
    // above: a stalled wave lane is precisely when the coverage lane needs new
    // datasets to keep the cluster busy. "No waves found" still blocks, because
    // that indicates an unbuilt/corrupt root rather than a stalled producer.
    if newest_wave_age(&g.waves_dir).is_none() {
        return Some("no waves found".into());
    }
    for path in &g.disk_paths {
        if let Some(pct) = disk_use_pct(path) {
            if pct >= 85.0 {
                return Some(format!("disk {}% at {}", pct, path.display()));
            }
        }
    }
    if let Some(cov) = coverage {
        if cov < COVERAGE_VALVE {
            return Some(format!(
                "coverage valve: previous tranche at {cov:.2} < {COVERAGE_VALVE}"
            ));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tranche bookkeeping
// ---------------------------------------------------------------------------

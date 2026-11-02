use super::*;

// ---------------------------------------------------------------------------
// arguments
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(about = "Detect, diagnose, repair and escalate an under-utilized cluster")]
pub(super) struct Args {
    #[arg(long, default_value = root_str())]
    pub(super) root: PathBuf,
    #[arg(long, default_value = scq_str())]
    pub(super) scq: PathBuf,

    /// Core-budget occupancy (%) below which the cluster counts as IDLE.
    /// Core budget, not `job_slot_target`: slot targets are multiplied up
    /// (1,052 slots over 344 real core-budget units) so slot occupancy reads
    /// ~30% on a cluster that is in fact core-saturated.
    #[arg(long, default_value_t = 25.0)]
    pub(super) idle_core_pct: f64,
    /// Queued+dispatching jobs below this is "under-supplied": the cluster is
    /// coasting on in-flight work with nothing behind it.
    #[arg(long, default_value_t = 2048)]
    pub(super) queue_floor_jobs: u64,
    /// Consecutive IDLE cycles before escalating to the owner.
    #[arg(long, default_value_t = 5)]
    pub(super) escalate_idle_cycles: u32,
    /// Consecutive DEGRADED cycles (busy now, producer broken) before escalating.
    #[arg(long, default_value_t = 15)]
    pub(super) escalate_degraded_cycles: u32,

    /// Hard ceiling on repair actions in any rolling hour. Never negotiable:
    /// a supervisor that can restart-storm is a second outage.
    #[arg(long, default_value_t = 6)]
    pub(super) max_actions_per_hour: usize,
    /// First retry delay for a given action; doubles per consecutive attempt.
    #[arg(long, default_value_t = 300)]
    pub(super) backoff_base_seconds: u64,
    #[arg(long, default_value_t = 3600)]
    pub(super) backoff_max_seconds: u64,
    /// Minimum gap between owner notifications while an escalation is open.
    #[arg(long, default_value_t = 1800)]
    pub(super) renotify_seconds: u64,

    /// Diagnose and record; never repair, never notify.
    #[arg(long)]
    pub(super) dry_run: bool,
    /// Force the IDLE verdict regardless of measurement. Rehearsal only.
    #[arg(long)]
    pub(super) simulate_idle: bool,
    /// Print an historical utilization report over the last N hours and exit.
    #[arg(long)]
    pub(super) report_hours: Option<f64>,

    #[arg(long)]
    pub(super) state_file: Option<PathBuf>,
    #[arg(long)]
    pub(super) status_file: Option<PathBuf>,
    #[arg(long)]
    pub(super) series_file: Option<PathBuf>,
    #[arg(long)]
    pub(super) events_file: Option<PathBuf>,
    #[arg(long)]
    pub(super) alarms_file: Option<PathBuf>,
    #[arg(long)]
    pub(super) banner_file: Option<PathBuf>,
}

impl Args {
    pub(super) fn state(&self) -> PathBuf {
        match &self.state_file {
            Some(path) => path.clone(),
            None => self.root.join("ops/utilization-supervisor-state.v1.json"),
        }
    }
    pub(super) fn status(&self) -> PathBuf {
        match &self.status_file {
            Some(path) => path.clone(),
            None => self.root.join("ops/utilization-supervisor-v1.json"),
        }
    }
    pub(super) fn series(&self) -> PathBuf {
        match &self.series_file {
            Some(path) => path.clone(),
            None => self.root.join("ops/utilization-series.v1.jsonl"),
        }
    }
    pub(super) fn events(&self) -> PathBuf {
        match &self.events_file {
            Some(path) => path.clone(),
            None => self.root.join("ops/utilization-supervisor-events.v1.jsonl"),
        }
    }
    pub(super) fn alarms(&self) -> PathBuf {
        match &self.alarms_file {
            Some(path) => path.clone(),
            None => self.root.join("ops/utilization-alarms.v1.json"),
        }
    }
    pub(super) fn banner(&self) -> PathBuf {
        match &self.banner_file {
            Some(path) => path.clone(),
            None => self.root.join("ops/UTILIZATION-ESCALATION.txt"),
        }
    }
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

pub(super) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Epoch millis -> `YYYY-MM-DDTHH:MM:SSZ` (Howard Hinnant civil-from-days).
pub(super) fn iso_utc(ms: u64) -> String {
    let secs = (ms / 1000) as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

pub(super) fn read_json(path: &Path) -> Option<Value> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

/// Atomic replace: a torn status file read by the TUI mid-write is a bug report
/// about the supervisor, which defeats the point of the supervisor.
pub(super) fn write_json_atomic(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let temporary = path.with_extension("tmp");
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    fs::write(&temporary, &bytes).with_context(|| format!("write {}", temporary.display()))?;
    fs::rename(&temporary, path).with_context(|| format!("rename into {}", path.display()))?;
    Ok(())
}

pub(super) fn append_jsonl(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{}", serde_json::to_string(value)?)?;
    Ok(())
}

pub(super) fn str_at<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}
pub(super) fn u64_at(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}
pub(super) fn f64_at(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

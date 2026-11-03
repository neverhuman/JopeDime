//! Engine telemetry service (spec §9.1, card TEL-04).
//!
//! Samples the engine and its workers at 1 Hz, computes the §9.1 metrics and publishes each
//! sample as an ephemeral SSE telemetry frame through
//! [`crate::events::EventLog::publish_telemetry`]. Sensors
//! and producer counters are injected, so a missing source is explicit: `value: null` with
//! `availability: "unavailable:<reason>"`, never zero. Telemetry only reads counters and OS
//! state; nothing here can reach the search, so it cannot change the champion (gate T3).

use crate::events::EventLog;
use jope_api::{
    decimal::DecimalU64, ids::RunId, MetricSample, MetricValue, TelemetryBody, TelemetryEvent,
};
use jope_telemetry::nvml_device::UnavailableReason;
use nvml_wrapper::{error::NvmlError, Nvml};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, MutexGuard,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// Registry names from `jope-telemetry` (`summary.rs`), the single source of truth.
pub use jope_telemetry::summary::{TOK_S_ALL_LANGUAGES, TOK_S_RUST, TOK_S_RUST_TIMED_ALONE};

/// Hardware sampling rate (§9.1).
pub const DEFAULT_INTERVAL: Duration = Duration::from_secs(1);

const NOT_REPORTED: &str = "not_reported";
const NO_PREVIOUS_SAMPLE: &str = "no_previous_sample";
const NO_TEXT_TOKEN_PATH: &str = "no_text_token_path";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProcessReading {
    pub cpu_seconds: f64,
    pub rss_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuReading {
    pub uuid: String,
    /// Device-wide utilization over NVML's sampling window, or the unavailable reason.
    pub util_pct: Result<f64, String>,
    pub mem_used_bytes: Result<u64, String>,
}

/// OS and device sources. A failure carries the reason published as `unavailable:<reason>`.
pub trait Sensors: Send {
    fn process(&mut self, pid: u32) -> Result<ProcessReading, String>;
    fn logical_cpus(&mut self) -> Result<f64, String>;
    /// CPUs this process may use: affinity, further limited by a cgroup quota.
    fn allowed_cpus(&mut self) -> Result<f64, String>;
    fn mem_available(&mut self) -> Result<u64, String>;
    fn disk_free(&mut self) -> Result<u64, String>;
    fn gpus(&mut self) -> Result<Vec<GpuReading>, String>;
}

/// Producer-reported rates and durations without an OS sensor (§9.1 rows owned by other
/// components). Exact byte counts use [`Counters::report_store_bytes`] instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Reported {
    RankerScoresPerS,
    RankerLatencyMs,
    DopeEncodeMs,
    LandmarksMs,
    IoReadGbS,
    PredictRowsPerS,
}

impl Reported {
    const ALL: [Self; 6] = [
        Self::RankerScoresPerS,
        Self::RankerLatencyMs,
        Self::DopeEncodeMs,
        Self::LandmarksMs,
        Self::IoReadGbS,
        Self::PredictRowsPerS,
    ];

    fn metric(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::RankerScoresPerS => ("ranker.scores_per_s", "candidates/s", "engine"),
            Self::RankerLatencyMs => ("ranker.latency_ms", "ms", "engine"),
            Self::DopeEncodeMs => ("dope.encode_ms", "ms", "engine"),
            Self::LandmarksMs => ("landmarks.ms", "ms", "engine"),
            Self::IoReadGbS => ("io.read_gb_s", "GB/s", "engine"),
            Self::PredictRowsPerS => ("predict.rows_per_s", "rows/s", "engine"),
        }
    }
}

/// Text-output token counts (TEL-07) and the time spent producing that output.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TokenCounts {
    pub rust_tokens: u64,
    pub rust_seconds: f64,
    pub rust_timed_alone: bool,
    pub all_tokens: u64,
    pub all_seconds: f64,
}

#[derive(Default)]
struct Marks {
    first_model: Option<Instant>,
    best: Option<Instant>,
}

/// Counters the engine updates; telemetry only reads them.
#[derive(Default)]
pub struct Counters {
    rung0_completed: AtomicU64,
    failed: Mutex<BTreeMap<String, u64>>,
    marks: Mutex<Marks>,
    tokens: Mutex<Option<TokenCounts>>,
    reported: Mutex<BTreeMap<Reported, (f64, Instant)>>,
    store_bytes: Mutex<Option<(u64, Instant)>>,
    workers: Mutex<Vec<u32>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl Counters {
    pub fn rung0_completed(&self) {
        self.rung0_completed.fetch_add(1, Ordering::Relaxed);
    }
    pub fn evaluation_failed(&self, reason: &str) {
        *lock(&self.failed).entry(reason.to_owned()).or_default() += 1;
    }
    /// A new champion was committed at `at`; the first one also sets `time.first_model_s`.
    pub fn champion_committed(&self, at: Instant) {
        let mut marks = lock(&self.marks);
        marks.first_model.get_or_insert(at);
        marks.best = Some(at);
    }
    /// Accumulate TEL-07 token counts and production time for newly published source.
    pub fn add_tokens(&self, counts: TokenCounts) {
        let mut tokens = lock(&self.tokens);
        let total = tokens.get_or_insert_with(|| TokenCounts {
            rust_timed_alone: true,
            ..TokenCounts::default()
        });
        total.rust_tokens += counts.rust_tokens;
        total.rust_seconds += counts.rust_seconds;
        total.rust_timed_alone &= counts.rust_timed_alone;
        total.all_tokens += counts.all_tokens;
        total.all_seconds += counts.all_seconds;
    }
    /// A non-finite or negative value is published as `unavailable`, never as a measurement.
    pub fn report(&self, metric: Reported, value: f64, at: Instant) {
        lock(&self.reported).insert(metric, (value, at));
    }
    /// Exact bytes owned by the store (§4.1 exact counts, no float conversion).
    pub fn report_store_bytes(&self, bytes: u64, at: Instant) {
        *lock(&self.store_bytes) = Some((bytes, at));
    }
    /// The live worker PIDs; telemetry samples these alongside the engine.
    pub fn set_workers(&self, pids: &[u32]) {
        *lock(&self.workers) = pids.to_vec();
    }
}

fn sample(
    name: &str,
    unit: &str,
    scope: &str,
    window: Duration,
    age: Duration,
    value: Result<MetricValue, String>,
) -> MetricSample {
    let millis = |d: Duration| u32::try_from(d.as_millis()).unwrap_or(u32::MAX);
    let (value, availability) = match value {
        Ok(value) => (Some(value), "measured".to_owned()),
        Err(reason) => (None, format!("unavailable:{reason}")),
    };
    MetricSample {
        name: name.to_owned(),
        value,
        unit: unit.to_owned(),
        scope: scope.to_owned(),
        window_ms: millis(window),
        age_ms: millis(age),
        availability,
    }
}

fn number(value: f64) -> Result<MetricValue, String> {
    if value.is_finite() {
        Ok(MetricValue::Number(value))
    } else {
        Err("non_finite".to_owned())
    }
}

fn bytes(value: u64) -> MetricValue {
    MetricValue::Decimal(DecimalU64(value))
}

/// Run-record summary written by the run owner at completion.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TelemetrySummary {
    pub samples: u64,
    pub cpu_cores_mean: Option<f64>,
    pub cpu_cores_peak: Option<f64>,
    pub mem_rss_peak_bytes: Option<u64>,
    pub evaluations_rung0: u64,
    pub evaluations_failed: BTreeMap<String, u64>,
    pub time_first_model_s: Option<f64>,
    pub time_best_s: Option<f64>,
    pub tokens_per_s_rust: Option<f64>,
    pub tokens_per_s_all_languages: Option<f64>,
    pub tokens_per_s_rust_timed_alone: Option<bool>,
    /// Last `unavailable:<reason>` per metric name.
    pub unavailable: BTreeMap<String, String>,
}

impl TelemetrySummary {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// Stateful §9.1 computation; one `tick` per sampling interval.
pub struct Telemetry {
    engine_pid: u32,
    started: Instant,
    last_at: Option<Instant>,
    last_cpu: BTreeMap<u32, f64>,
    last_rung0: u64,
    cores_sum: f64,
    cores_count: u64,
    summary: TelemetrySummary,
}

impl Telemetry {
    pub fn new(engine_pid: u32, started: Instant) -> Self {
        Self {
            engine_pid,
            started,
            last_at: None,
            last_cpu: BTreeMap::new(),
            last_rung0: 0,
            cores_sum: 0.0,
            cores_count: 0,
            summary: TelemetrySummary::default(),
        }
    }

    pub fn tick(
        &mut self,
        now: Instant,
        sensors: &mut dyn Sensors,
        counters: &Counters,
    ) -> Vec<MetricSample> {
        let window = self.last_at.map(|last| now.saturating_duration_since(last));
        let seconds = window.map(|w| w.as_secs_f64()).filter(|s| *s > 0.0);
        let zero = Duration::ZERO;
        let window_or_zero = window.unwrap_or(zero);
        let mut out = Vec::with_capacity(32);

        // CPU and RSS over the engine plus live workers, with per-PID deltas (F06, no clamping).
        let mut pids = vec![self.engine_pid];
        pids.extend(
            lock(&counters.workers)
                .iter()
                .copied()
                .filter(|p| *p != self.engine_pid),
        );
        let mut cpu_delta = 0.0;
        let mut rss = 0u64;
        let mut cpu_now = BTreeMap::new();
        // A failed reading of any listed process makes the totals incomplete: publish them as
        // unavailable rather than as a falsely complete measurement.
        let mut incomplete: Option<String> = None;
        for pid in &pids {
            match sensors.process(*pid) {
                Ok(reading) => {
                    if let Some(previous) = self.last_cpu.get(pid) {
                        cpu_delta += (reading.cpu_seconds - previous).max(0.0);
                    }
                    cpu_now.insert(*pid, reading.cpu_seconds);
                    rss = rss.saturating_add(reading.rss_bytes);
                }
                Err(reason) if incomplete.is_none() => {
                    let who = if *pid == self.engine_pid {
                        "engine"
                    } else {
                        "worker"
                    };
                    incomplete = Some(format!("{who}_{reason}"));
                }
                Err(_) => {}
            }
        }
        self.last_cpu = cpu_now;
        let cores = match (&incomplete, seconds) {
            (Some(reason), _) => Err(reason.clone()),
            (None, None) => Err(NO_PREVIOUS_SAMPLE.to_owned()),
            (None, Some(seconds)) => Ok(cpu_delta / seconds),
        };
        if let Ok(cores) = cores {
            self.cores_sum += cores;
            self.cores_count += 1;
            self.summary.cpu_cores_peak =
                Some(self.summary.cpu_cores_peak.map_or(cores, |p| p.max(cores)));
        }
        out.push(sample(
            "cpu.cores",
            "core-equivalents",
            "engine",
            window_or_zero,
            zero,
            cores.clone().and_then(number),
        ));
        let share = |total: Result<f64, String>| -> Result<MetricValue, String> {
            let cores = cores.clone()?;
            let total = total?;
            if total <= 0.0 {
                return Err("no_cpus".to_owned());
            }
            number(100.0 * cores / total)
        };
        out.push(sample(
            "cpu.host_pct",
            "%",
            "host",
            window_or_zero,
            zero,
            share(sensors.logical_cpus()),
        ));
        out.push(sample(
            "cpu.allowed_pct",
            "%",
            "host",
            window_or_zero,
            zero,
            share(sensors.allowed_cpus()),
        ));
        let rss = match &incomplete {
            None => {
                self.summary.mem_rss_peak_bytes =
                    Some(self.summary.mem_rss_peak_bytes.map_or(rss, |p| p.max(rss)));
                Ok(bytes(rss))
            }
            Some(reason) => Err(reason.clone()),
        };
        out.push(sample("mem.rss", "bytes", "engine", zero, zero, rss));
        out.push(sample(
            "mem.available",
            "bytes",
            "host",
            zero,
            zero,
            sensors.mem_available().map(bytes),
        ));

        // GPU, device-wide; no synchronization for telemetry (§9.1 Rates).
        match sensors.gpus() {
            Ok(devices) if !devices.is_empty() => {
                for device in devices {
                    let scope = format!("device:{}", device.uuid);
                    out.push(sample(
                        "gpu.util_pct",
                        "%",
                        &scope,
                        zero,
                        zero,
                        device.util_pct.and_then(number),
                    ));
                    out.push(sample(
                        "gpu.mem_used",
                        "bytes",
                        &scope,
                        zero,
                        zero,
                        device.mem_used_bytes.map(bytes),
                    ));
                }
            }
            result => {
                let reason = result.err().unwrap_or_else(|| "no_devices".to_owned());
                out.push(sample(
                    "gpu.util_pct",
                    "%",
                    "device:*",
                    zero,
                    zero,
                    Err(reason.clone()),
                ));
                out.push(sample(
                    "gpu.mem_used",
                    "bytes",
                    "device:*",
                    zero,
                    zero,
                    Err(reason),
                ));
            }
        }

        // Producer-reported rows.
        let reported = lock(&counters.reported).clone();
        for metric in Reported::ALL {
            let (name, unit, scope) = metric.metric();
            let value = match reported.get(&metric) {
                Some((value, at)) if *value < 0.0 => (
                    Err("negative_value".to_owned()),
                    now.saturating_duration_since(*at),
                ),
                Some((value, at)) => (number(*value), now.saturating_duration_since(*at)),
                None => (Err(NOT_REPORTED.to_owned()), zero),
            };
            out.push(sample(name, unit, scope, zero, value.1, value.0));
        }
        let store = *lock(&counters.store_bytes);
        out.push(match store {
            Some((value, at)) => sample(
                "store.bytes",
                "bytes",
                "store",
                zero,
                now.saturating_duration_since(at),
                Ok(bytes(value)),
            ),
            None => sample(
                "store.bytes",
                "bytes",
                "store",
                zero,
                zero,
                Err(NOT_REPORTED.to_owned()),
            ),
        });

        // Evaluation rate and failures.
        let rung0 = counters.rung0_completed.load(Ordering::Relaxed);
        let rate = match seconds {
            Some(seconds) => number(rung0.saturating_sub(self.last_rung0) as f64 / seconds),
            None => Err(NO_PREVIOUS_SAMPLE.to_owned()),
        };
        self.last_rung0 = rung0;
        self.summary.evaluations_rung0 = rung0;
        out.push(sample(
            "eval.per_s",
            "evals/s",
            "engine",
            window_or_zero,
            zero,
            rate,
        ));
        let failed = lock(&counters.failed).clone();
        let total: u64 = failed.values().sum();
        out.push(sample(
            "eval.failed",
            "count",
            "engine",
            zero,
            zero,
            Ok(bytes(total)),
        ));
        for (reason, count) in &failed {
            out.push(sample(
                &format!("eval.failed.{reason}"),
                "count",
                "engine",
                zero,
                zero,
                Ok(bytes(*count)),
            ));
        }
        self.summary.evaluations_failed = failed;

        // tok/s: Rust-only headline plus all languages, both labeled (§9.1, owner ruling).
        let tokens = *lock(&counters.tokens);
        let rate = |tokens: u64, seconds: f64| -> Result<MetricValue, String> {
            if seconds > 0.0 {
                number(tokens as f64 / seconds)
            } else {
                Err(NO_TEXT_TOKEN_PATH.to_owned())
            }
        };
        let (rust, all, alone) = match tokens {
            Some(t) => (
                rate(t.rust_tokens, t.rust_seconds),
                rate(t.all_tokens, t.all_seconds),
                if t.rust_seconds > 0.0 {
                    Ok(MetricValue::Boolean(t.rust_timed_alone))
                } else {
                    Err(NO_TEXT_TOKEN_PATH.to_owned())
                },
            ),
            None => (
                Err(NO_TEXT_TOKEN_PATH.to_owned()),
                Err(NO_TEXT_TOKEN_PATH.to_owned()),
                Err(NO_TEXT_TOKEN_PATH.to_owned()),
            ),
        };
        self.summary.tokens_per_s_rust = rust.as_ref().ok().and_then(as_f64);
        self.summary.tokens_per_s_all_languages = all.as_ref().ok().and_then(as_f64);
        self.summary.tokens_per_s_rust_timed_alone = match &alone {
            Ok(MetricValue::Boolean(b)) => Some(*b),
            _ => None,
        };
        out.push(sample(TOK_S_RUST, "tokens/s", "engine", zero, zero, rust));
        out.push(sample(
            TOK_S_ALL_LANGUAGES,
            "tokens/s",
            "engine",
            zero,
            zero,
            all,
        ));
        out.push(sample(
            TOK_S_RUST_TIMED_ALONE,
            "bool",
            "engine",
            zero,
            zero,
            alone,
        ));

        // Time to first and to current champion.
        let (first, best) = {
            let marks = lock(&counters.marks);
            (marks.first_model, marks.best)
        };
        let since = |mark: Option<Instant>| -> Result<f64, String> {
            let at = mark.ok_or_else(|| "no_model_yet".to_owned())?;
            // A mark before the canonical run start is inconsistent input, never a measured 0.
            at.checked_duration_since(self.started)
                .map(|d| d.as_secs_f64())
                .ok_or_else(|| "before_run_start".to_owned())
        };
        self.summary.time_first_model_s = since(first).ok();
        self.summary.time_best_s = since(best).ok();
        out.push(sample(
            "time.first_model_s",
            "s",
            "engine",
            zero,
            zero,
            since(first).and_then(number),
        ));
        out.push(sample(
            "time.best_s",
            "s",
            "engine",
            zero,
            zero,
            since(best).and_then(number),
        ));

        out.push(sample(
            "disk.free",
            "bytes",
            "store",
            zero,
            zero,
            sensors.disk_free().map(bytes),
        ));

        for metric in &out {
            if metric.value.is_none() {
                self.summary
                    .unavailable
                    .insert(metric.name.clone(), metric.availability.clone());
            } else {
                self.summary.unavailable.remove(&metric.name);
            }
        }
        self.summary.samples += 1;
        self.summary.cpu_cores_mean =
            (self.cores_count > 0).then(|| self.cores_sum / self.cores_count as f64);
        self.last_at = Some(now);
        out
    }

    pub fn summary(&self) -> TelemetrySummary {
        self.summary.clone()
    }
}

fn as_f64(value: &MetricValue) -> Option<f64> {
    match value {
        MetricValue::Number(n) => Some(*n),
        _ => None,
    }
}

/// The 1 Hz sampling thread. It publishes ephemeral frames and never touches the search.
pub struct TelemetryService {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<TelemetrySummary>>,
}

impl TelemetryService {
    /// `run_started` is the run's canonical start, so `time.*` metrics stay correct when
    /// telemetry attaches after the run began (for example after an engine restart).
    pub fn spawn(
        run: RunId,
        log: Arc<EventLog>,
        mut sensors: Box<dyn Sensors>,
        counters: Arc<Counters>,
        run_started: Instant,
        interval: Duration,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let thread = thread::Builder::new()
            .name("jope-telemetry".to_owned())
            .spawn(move || {
                let mut telemetry = Telemetry::new(std::process::id(), run_started);
                while !flag.load(Ordering::Acquire) {
                    let samples = telemetry.tick(Instant::now(), sensors.as_mut(), &counters);
                    log.publish_telemetry(TelemetryEvent {
                        run_id: run.clone(),
                        time: chrono::Utc::now()
                            .to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
                        event: TelemetryBody::Telemetry(samples),
                    });
                    thread::park_timeout(interval);
                }
                telemetry.summary()
            })
            .expect("spawn telemetry thread");
        Self {
            stop,
            thread: Some(thread),
        }
    }

    /// Stop sampling and return the summary for the run record.
    pub fn stop(mut self) -> TelemetrySummary {
        self.finish().unwrap_or_default()
    }

    fn finish(&mut self) -> Option<TelemetrySummary> {
        self.stop.store(true, Ordering::Release);
        let thread = self.thread.take()?;
        thread.thread().unpark();
        thread.join().ok()
    }
}

impl Drop for TelemetryService {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

/// NVML device queries needed for the §9.1 `gpu.*` rows. The boundary keeps tests off real
/// hardware; [`SystemGpu`] is the `nvml-wrapper` implementation.
mod gpu;
pub use gpu::{GpuQuery, NvmlSensor, SystemGpu, SystemSensors};

/// CPUs allowed by affinity, further limited by every cgroup v2 `cpu.max` quota from this
/// process's own cgroup (from `/proc/self/cgroup`, line `0::<path>`) up to the root.
/// `cpu_max(relative)` reads `<relative>/cpu.max` under the cgroup v2 mount; an absent file or
/// `max` quota imposes no limit. Without a v2 membership line, affinity alone applies.
/// Unresolved or relative membership is unavailable and never passed to the reader.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn cgroup_allowed_cpus(
    membership: &str,
    affinity: f64,
    cpu_max: impl Fn(&str) -> Option<String>,
) -> Result<f64, String> {
    let Some(path) = membership
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .map(str::trim)
    else {
        return Ok(affinity);
    };
    if !path.starts_with('/') || path.split('/').any(|part| matches!(part, "." | "..")) {
        return Err("cgroup_path_unresolved".to_owned());
    }
    let mut allowed = affinity;
    let mut current = path.trim_matches('/').to_owned();
    loop {
        if let Some(limit) = cpu_max(&current).and_then(|line| {
            let mut parts = line.split_whitespace();
            let quota = parts.next()?.parse::<f64>().ok()?;
            let period = parts.next()?.parse::<f64>().ok()?;
            (quota > 0.0 && period > 0.0).then_some(quota / period)
        }) {
            allowed = allowed.min(limit);
        }
        if current.is_empty() {
            return Ok(allowed);
        }
        current = current
            .rsplit_once('/')
            .map_or(String::new(), |(parent, _)| parent.to_owned());
    }
}

#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
mod platform {
    use nix::{
        sched::{sched_getaffinity, CpuSet},
        sys::statvfs::statvfs,
        unistd::Pid,
    };
    use std::{fs, path::Path};

    pub(super) fn logical_cpus() -> Result<f64, String> {
        // SAFETY: sysconf has no pointer arguments or memory-safety preconditions.
        let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
        if n > 0 {
            Ok(n as f64)
        } else {
            Err("sysconf_failed".to_owned())
        }
    }

    pub(super) fn allowed_cpus() -> Result<f64, String> {
        let set = sched_getaffinity(Pid::from_raw(0)).map_err(|_| "affinity_unavailable")?;
        let mut affinity = 0.0;
        for cpu in 0..CpuSet::count() {
            if set.is_set(cpu).map_err(|_| "affinity_unavailable")? {
                affinity += 1.0;
            }
        }
        let membership = fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
        super::cgroup_allowed_cpus(&membership, affinity, |path| {
            fs::read_to_string(Path::new("/sys/fs/cgroup").join(path).join("cpu.max")).ok()
        })
    }

    pub(super) fn mem_available() -> Result<u64, String> {
        let meminfo = fs::read_to_string("/proc/meminfo").map_err(|_| "meminfo_unavailable")?;
        meminfo
            .lines()
            .find_map(|line| line.strip_prefix("MemAvailable:"))
            .and_then(|rest| rest.split_whitespace().next()?.parse::<u64>().ok())
            .map(|kib| kib.saturating_mul(1024))
            .ok_or_else(|| "meminfo_malformed".to_owned())
    }

    pub(super) fn disk_free(path: &Path) -> Result<u64, String> {
        let stats = statvfs(path).map_err(|_| "statvfs_failed")?;
        Ok((stats.blocks_available() as u64).saturating_mul(stats.fragment_size() as u64))
    }
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod platform {
    use nix::sys::statvfs::statvfs;
    use std::path::Path;

    // libc exposes host_statistics64 and mach_host_self, but not the matching
    // mach_port_deallocate declaration. Keep this narrow FFI next to the Mach sensor.
    #[link(name = "System")]
    unsafe extern "C" {
        fn mach_host_self() -> libc::mach_port_t;
        fn mach_task_self() -> libc::mach_port_t;
        fn mach_port_deallocate(task: libc::task_t, name: libc::mach_port_t)
            -> libc::kern_return_t;
    }

    pub(super) fn logical_cpus() -> Result<f64, String> {
        std::thread::available_parallelism()
            .map(|n| n.get() as f64)
            .map_err(|_| "parallelism_unavailable".to_owned())
    }

    pub(super) fn allowed_cpus() -> Result<f64, String> {
        logical_cpus()
    }

    /// Conservative available memory from free and inactive VM pages.
    /// Failures stay unavailable; admission never substitutes total RAM for availability.
    pub(super) fn mem_available() -> Result<u64, String> {
        // SAFETY: obtains this process's send right; it is released below on every path.
        let host = unsafe { mach_host_self() };
        if host == 0 {
            return Err("mach_host_unavailable".to_owned());
        }
        // `host_statistics64` writes a counted array of integer_t words. Keep
        // that ABI buffer as a fully initialized typed array; never construct a
        // partially written `vm_statistics64` value.
        let mut stats = [0 as libc::integer_t; libc::HOST_VM_INFO64_COUNT as usize];
        let mut count = libc::HOST_VM_INFO64_COUNT;
        // SAFETY: `stats` is aligned writable storage for HOST_VM_INFO64_COUNT
        // integer_t words as required by the Mach API; `count` is writable.
        let status = unsafe {
            libc::host_statistics64(host, libc::HOST_VM_INFO64, stats.as_mut_ptr(), &mut count)
        };
        // SAFETY: release the send right obtained by mach_host_self, in our own task.
        let released = unsafe { mach_port_deallocate(mach_task_self(), host) };
        if status != libc::KERN_SUCCESS {
            return Err("host_statistics_unavailable".to_owned());
        }
        if released != libc::KERN_SUCCESS {
            return Err("host_port_release_failed".to_owned());
        }
        let required_count = (std::mem::offset_of!(libc::vm_statistics64, inactive_count)
            + std::mem::size_of::<libc::natural_t>())
        .div_ceil(std::mem::size_of::<libc::integer_t>())
            as libc::mach_msg_type_number_t;
        if count < required_count {
            return Err("host_statistics_truncated".to_owned());
        }
        let word_at = |field_offset: usize| -> Result<libc::natural_t, String> {
            if std::mem::size_of::<libc::natural_t>() != std::mem::size_of::<libc::integer_t>()
                || field_offset % std::mem::size_of::<libc::integer_t>() != 0
            {
                return Err("host_statistics_layout_unavailable".to_owned());
            }
            let index = field_offset / std::mem::size_of::<libc::integer_t>();
            let word = stats
                .get(index)
                .ok_or_else(|| "host_statistics_truncated".to_owned())?;
            Ok(libc::natural_t::from_ne_bytes(word.to_ne_bytes()))
        };
        let free_count = word_at(std::mem::offset_of!(libc::vm_statistics64, free_count))?;
        let inactive_count = word_at(std::mem::offset_of!(libc::vm_statistics64, inactive_count))?;
        // XNU mach/vm_statistics.h documents speculative_count as already included
        // in free_count. Adding it again would overstate admission headroom.
        let pages = u64::from(free_count)
            .checked_add(u64::from(inactive_count))
            .ok_or_else(|| "memory_pages_overflow".to_owned())?;
        // SAFETY: libSystem initializes this immutable VM page-size value before main.
        let page_size = unsafe { libc::vm_page_size } as u64;
        if page_size == 0 {
            return Err("page_size_unavailable".to_owned());
        }
        pages
            .checked_mul(page_size)
            .ok_or_else(|| "memory_bytes_overflow".to_owned())
    }

    pub(super) fn disk_free(path: &Path) -> Result<u64, String> {
        let stats = statvfs(path).map_err(|_| "statvfs_failed")?;
        Ok((stats.blocks_available() as u64).saturating_mul(stats.fragment_size() as u64))
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod platform {
    use std::path::Path;

    pub(super) fn logical_cpus() -> Result<f64, String> {
        std::thread::available_parallelism()
            .map(|n| n.get() as f64)
            .map_err(|_| "parallelism_unavailable".to_owned())
    }
    pub(super) fn allowed_cpus() -> Result<f64, String> {
        logical_cpus()
    }
    pub(super) fn mem_available() -> Result<u64, String> {
        Err("unsupported_platform".to_owned())
    }
    pub(super) fn disk_free(_: &Path) -> Result<u64, String> {
        Err("unsupported_platform".to_owned())
    }
}

#[cfg(test)]
mod tests;

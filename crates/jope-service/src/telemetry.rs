//! Engine telemetry service (spec §9.1, card TEL-04).
//!
//! Samples the engine and its workers at 1 Hz, computes the §9.1 metrics and publishes each
//! sample as an ephemeral SSE telemetry frame through [`EventLog::publish_telemetry`]. Sensors
//! and producer counters are injected, so a missing source is explicit: `value: null` with
//! `availability: "unavailable:<reason>"`, never zero. Telemetry only reads counters and OS
//! state; nothing here can reach the search, so it cannot change the champion (gate T3).

use crate::events::EventLog;
use jope_api::{
    decimal::DecimalU64, ids::RunId, MetricSample, MetricValue, TelemetryBody, TelemetryEvent,
};
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

/// Registry names shared with `jope-telemetry` (`summary.rs` `TOK_S_*`); kept identical.
pub const TOK_S_RUST: &str = "throughput.tokens_per_s.rust";
pub const TOK_S_ALL_LANGUAGES: &str = "throughput.tokens_per_s.all_languages";
pub const TOK_S_RUST_TIMED_ALONE: &str = "throughput.tokens_per_s.rust_timed_alone";

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
    pub util_pct: Option<f64>,
    pub mem_used_bytes: Option<u64>,
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

/// Producer-reported values without an OS sensor (§9.1 rows owned by other components).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Reported {
    RankerScoresPerS,
    RankerLatencyMs,
    DopeEncodeMs,
    LandmarksMs,
    IoReadGbS,
    PredictRowsPerS,
    StoreBytes,
}

impl Reported {
    const ALL: [Self; 7] = [
        Self::RankerScoresPerS,
        Self::RankerLatencyMs,
        Self::DopeEncodeMs,
        Self::LandmarksMs,
        Self::IoReadGbS,
        Self::PredictRowsPerS,
        Self::StoreBytes,
    ];

    fn metric(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::RankerScoresPerS => ("ranker.scores_per_s", "candidates/s", "engine"),
            Self::RankerLatencyMs => ("ranker.latency_ms", "ms", "engine"),
            Self::DopeEncodeMs => ("dope.encode_ms", "ms", "engine"),
            Self::LandmarksMs => ("landmarks.ms", "ms", "engine"),
            Self::IoReadGbS => ("io.read_gb_s", "GB/s", "engine"),
            Self::PredictRowsPerS => ("predict.rows_per_s", "rows/s", "engine"),
            Self::StoreBytes => ("store.bytes", "bytes", "store"),
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
    pub fn report(&self, metric: Reported, value: f64, at: Instant) {
        lock(&self.reported).insert(metric, (value, at));
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
        let engine = sensors.process(self.engine_pid);
        let mut cpu_delta = 0.0;
        let mut rss = 0u64;
        let mut cpu_now = BTreeMap::new();
        for pid in &pids {
            let reading = if *pid == self.engine_pid {
                engine.clone()
            } else {
                sensors.process(*pid)
            };
            if let Ok(reading) = reading {
                if let Some(previous) = self.last_cpu.get(pid) {
                    cpu_delta += (reading.cpu_seconds - previous).max(0.0);
                }
                cpu_now.insert(*pid, reading.cpu_seconds);
                rss = rss.saturating_add(reading.rss_bytes);
            }
        }
        self.last_cpu = cpu_now;
        let cores = match (&engine, seconds) {
            (Err(reason), _) => Err(format!("engine_{reason}")),
            (Ok(_), None) => Err(NO_PREVIOUS_SAMPLE.to_owned()),
            (Ok(_), Some(seconds)) => Ok(cpu_delta / seconds),
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
        let rss = match &engine {
            Ok(_) => {
                self.summary.mem_rss_peak_bytes =
                    Some(self.summary.mem_rss_peak_bytes.map_or(rss, |p| p.max(rss)));
                Ok(bytes(rss))
            }
            Err(reason) => Err(format!("engine_{reason}")),
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
                        device
                            .util_pct
                            .ok_or_else(|| NOT_REPORTED.to_owned())
                            .and_then(number),
                    ));
                    out.push(sample(
                        "gpu.mem_used",
                        "bytes",
                        &scope,
                        zero,
                        zero,
                        device
                            .mem_used_bytes
                            .map(bytes)
                            .ok_or_else(|| NOT_REPORTED.to_owned()),
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
                Some((value, at)) if metric == Reported::StoreBytes && *value >= 0.0 => {
                    (Ok(bytes(*value as u64)), now.saturating_duration_since(*at))
                }
                Some((value, at)) => (number(*value), now.saturating_duration_since(*at)),
                None => (Err(NOT_REPORTED.to_owned()), zero),
            };
            out.push(sample(name, unit, scope, zero, value.1, value.0));
        }

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
        let since = |mark: Option<Instant>| {
            mark.map(|at| at.saturating_duration_since(self.started).as_secs_f64())
                .ok_or_else(|| "no_model_yet".to_owned())
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
    pub fn spawn(
        run: RunId,
        log: Arc<EventLog>,
        mut sensors: Box<dyn Sensors>,
        counters: Arc<Counters>,
        interval: Duration,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let thread = thread::Builder::new()
            .name("jope-telemetry".to_owned())
            .spawn(move || {
                let mut telemetry = Telemetry::new(std::process::id(), Instant::now());
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

/// Host sensors: WRK-03 per-PID sampling plus OS memory, CPU and disk queries.
pub struct SystemSensors {
    store: PathBuf,
}

impl SystemSensors {
    /// `store` is the `$JOPEDIME_HOME` volume used for `disk.free`.
    pub fn new(store: impl Into<PathBuf>) -> Self {
        Self {
            store: store.into(),
        }
    }
}

impl Sensors for SystemSensors {
    fn process(&mut self, pid: u32) -> Result<ProcessReading, String> {
        crate::sampler::sample(pid)
            .map(|(cpu_seconds, rss_bytes)| ProcessReading {
                cpu_seconds,
                rss_bytes,
            })
            .map_err(|error| match error.kind() {
                std::io::ErrorKind::NotFound => "process_exited".to_owned(),
                _ => "sampler_error".to_owned(),
            })
    }
    fn logical_cpus(&mut self) -> Result<f64, String> {
        platform::logical_cpus()
    }
    fn allowed_cpus(&mut self) -> Result<f64, String> {
        platform::allowed_cpus()
    }
    fn mem_available(&mut self) -> Result<u64, String> {
        platform::mem_available()
    }
    fn disk_free(&mut self) -> Result<u64, String> {
        platform::disk_free(&self.store)
    }
    fn gpus(&mut self) -> Result<Vec<GpuReading>, String> {
        // NVML-backed sampling is TEL-04-F01 (pending the jope-service dependency patch).
        Err("nvml_not_linked".to_owned())
    }
}

#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
mod platform {
    use std::{ffi::CString, fs, os::unix::ffi::OsStrExt, path::Path};

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
        // SAFETY: an all-zero cpu_set_t is valid; sched_getaffinity writes at most its size.
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<libc::cpu_set_t>();
        // SAFETY: `set` is a valid, writable cpu_set_t of `size` bytes for pid 0 (this thread).
        if unsafe { libc::sched_getaffinity(0, size, &mut set) } != 0 {
            return Err("affinity_unavailable".to_owned());
        }
        // SAFETY: `set` was initialized by sched_getaffinity above.
        let affinity = unsafe { libc::CPU_COUNT(&set) } as f64;
        // cgroup v2 quota: "<quota> <period>" or "max <period>".
        let quota = fs::read_to_string("/sys/fs/cgroup/cpu.max")
            .ok()
            .and_then(|line| {
                let mut parts = line.split_whitespace();
                let quota = parts.next()?.parse::<f64>().ok()?;
                let period = parts.next()?.parse::<f64>().ok()?;
                (period > 0.0).then(|| quota / period)
            });
        Ok(quota.map_or(affinity, |q| q.min(affinity)))
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
        let path = CString::new(path.as_os_str().as_bytes()).map_err(|_| "invalid_store_path")?;
        // SAFETY: an all-zero statvfs is valid output storage for statvfs below.
        let mut stats: libc::statvfs = unsafe { std::mem::zeroed() };
        // SAFETY: `path` is NUL-terminated and `stats` is valid writable storage.
        if unsafe { libc::statvfs(path.as_ptr(), &mut stats) } != 0 {
            return Err("statvfs_failed".to_owned());
        }
        Ok((stats.f_bavail as u64).saturating_mul(stats.f_frsize as u64))
    }
}

#[cfg(not(target_os = "linux"))]
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
mod tests {
    use super::*;

    const ENGINE: u32 = 1;

    #[derive(Clone, Default)]
    struct Fake {
        process: BTreeMap<u32, ProcessReading>,
        logical: Option<f64>,
        allowed: Option<f64>,
        mem: Option<u64>,
        disk: Option<u64>,
        gpus: Option<Vec<GpuReading>>,
    }
    impl Sensors for Fake {
        fn process(&mut self, pid: u32) -> Result<ProcessReading, String> {
            self.process
                .get(&pid)
                .copied()
                .ok_or_else(|| "process_exited".to_owned())
        }
        fn logical_cpus(&mut self) -> Result<f64, String> {
            self.logical.ok_or_else(|| "no_logical".to_owned())
        }
        fn allowed_cpus(&mut self) -> Result<f64, String> {
            self.allowed.ok_or_else(|| "no_allowed".to_owned())
        }
        fn mem_available(&mut self) -> Result<u64, String> {
            self.mem.ok_or_else(|| "meminfo_unavailable".to_owned())
        }
        fn disk_free(&mut self) -> Result<u64, String> {
            self.disk.ok_or_else(|| "statvfs_failed".to_owned())
        }
        fn gpus(&mut self) -> Result<Vec<GpuReading>, String> {
            self.gpus
                .clone()
                .ok_or_else(|| "nvml_not_linked".to_owned())
        }
    }
    fn get<'a>(samples: &'a [MetricSample], name: &str) -> &'a MetricSample {
        samples
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("{name} missing"))
    }
    fn num(sample: &MetricSample) -> f64 {
        match sample.value {
            Some(MetricValue::Number(n)) => n,
            ref other => panic!("{} not a number: {other:?}", sample.name),
        }
    }
    fn reading(cpu: f64, rss: u64) -> ProcessReading {
        ProcessReading {
            cpu_seconds: cpu,
            rss_bytes: rss,
        }
    }

    #[test]
    fn cpu_math_follows_f06_without_clamping() {
        let t0 = Instant::now();
        let counters = Counters::default();
        for (cpu_per_s, cores, host_pct) in
            [(1.0, 1.00, 12.5), (4.0, 4.00, 50.0), (16.0, 16.0, 200.0)]
        {
            let mut fake = Fake {
                logical: Some(8.0),
                allowed: Some(4.0),
                ..Fake::default()
            };
            let mut telemetry = Telemetry::new(ENGINE, t0);
            fake.process.insert(ENGINE, reading(10.0, 1));
            let first = telemetry.tick(t0, &mut fake, &counters);
            assert_eq!(
                get(&first, "cpu.cores").availability,
                "unavailable:no_previous_sample"
            );
            fake.process
                .insert(ENGINE, reading(10.0 + 2.0 * cpu_per_s, 1));
            let second = telemetry.tick(t0 + Duration::from_secs(2), &mut fake, &counters);
            assert!((num(get(&second, "cpu.cores")) - cores).abs() < 1e-9);
            assert!((num(get(&second, "cpu.host_pct")) - host_pct).abs() < 1e-9);
            assert!((num(get(&second, "cpu.allowed_pct")) - 100.0 * cores / 4.0).abs() < 1e-9);
            assert_eq!(get(&second, "cpu.cores").window_ms, 2000);
        }
    }

    #[test]
    fn rss_sums_engine_and_live_workers_and_new_workers_baseline() {
        let t0 = Instant::now();
        let counters = Counters::default();
        let mut fake = Fake::default();
        let mut telemetry = Telemetry::new(ENGINE, t0);
        fake.process.insert(ENGINE, reading(0.0, 100));
        fake.process.insert(2, reading(0.0, 10));
        counters.set_workers(&[2]);
        telemetry.tick(t0, &mut fake, &counters);
        // Worker 2 exits; worker 3 appears with 50 s of history that must not count as a spike.
        fake.process.remove(&2);
        fake.process.insert(3, reading(50.0, 20));
        fake.process.insert(ENGINE, reading(1.0, 100));
        counters.set_workers(&[2, 3]);
        let s = telemetry.tick(t0 + Duration::from_secs(1), &mut fake, &counters);
        assert!((num(get(&s, "cpu.cores")) - 1.0).abs() < 1e-9);
        assert_eq!(get(&s, "mem.rss").value, Some(bytes(120)));
        fake.process.insert(3, reading(51.0, 20));
        fake.process.insert(ENGINE, reading(2.0, 100));
        let s = telemetry.tick(t0 + Duration::from_secs(2), &mut fake, &counters);
        assert!((num(get(&s, "cpu.cores")) - 2.0).abs() < 1e-9);
        let summary = telemetry.summary();
        assert_eq!(summary.mem_rss_peak_bytes, Some(120));
        assert_eq!(summary.cpu_cores_peak, Some(2.0));
        assert_eq!(summary.cpu_cores_mean, Some(1.5));
    }

    #[test]
    fn missing_sensors_produce_null_with_a_reason_never_zero() {
        let t0 = Instant::now();
        let counters = Counters::default();
        let mut fake = Fake::default();
        let mut telemetry = Telemetry::new(ENGINE, t0);
        let s = telemetry.tick(t0, &mut fake, &counters);
        for metric in &s {
            if metric.value.is_none() {
                assert!(
                    metric.availability.starts_with("unavailable:"),
                    "{metric:?}"
                );
                assert!(metric.availability.len() > "unavailable:".len());
            } else {
                assert_eq!(metric.availability, "measured");
            }
        }
        let reason = |name: &str| get(&s, name).availability.clone();
        assert_eq!(reason("cpu.cores"), "unavailable:engine_process_exited");
        assert_eq!(reason("mem.rss"), "unavailable:engine_process_exited");
        assert_eq!(reason("mem.available"), "unavailable:meminfo_unavailable");
        assert_eq!(reason("disk.free"), "unavailable:statvfs_failed");
        assert_eq!(reason("gpu.util_pct"), "unavailable:nvml_not_linked");
        assert_eq!(reason("ranker.scores_per_s"), "unavailable:not_reported");
        assert_eq!(reason("store.bytes"), "unavailable:not_reported");
        assert_eq!(reason(TOK_S_RUST), "unavailable:no_text_token_path");
        assert_eq!(
            reason(TOK_S_ALL_LANGUAGES),
            "unavailable:no_text_token_path"
        );
        assert_eq!(
            reason(TOK_S_RUST_TIMED_ALONE),
            "unavailable:no_text_token_path"
        );
        assert_eq!(reason("time.first_model_s"), "unavailable:no_model_yet");
        assert_eq!(reason("eval.per_s"), "unavailable:no_previous_sample");
        let summary = telemetry.summary();
        assert_eq!(
            summary.unavailable["gpu.mem_used"],
            "unavailable:nvml_not_linked"
        );
        assert_eq!(summary.tokens_per_s_rust, None);
    }

    #[test]
    fn tokens_per_second_uses_tel07_counts_and_registry_names() {
        let t0 = Instant::now();
        let counters = Counters::default();
        counters.add_tokens(TokenCounts {
            rust_tokens: 600,
            rust_seconds: 1.0,
            rust_timed_alone: true,
            all_tokens: 2000,
            all_seconds: 3.0,
        });
        counters.add_tokens(TokenCounts {
            rust_tokens: 400,
            rust_seconds: 1.0,
            rust_timed_alone: false,
            all_tokens: 1000,
            all_seconds: 1.0,
        });
        let mut fake = Fake::default();
        let s = Telemetry::new(ENGINE, t0).tick(t0, &mut fake, &counters);
        assert_eq!(num(get(&s, TOK_S_RUST)), 500.0);
        assert_eq!(num(get(&s, TOK_S_ALL_LANGUAGES)), 750.0);
        assert_eq!(
            get(&s, TOK_S_RUST_TIMED_ALONE).value,
            Some(MetricValue::Boolean(false))
        );
        assert_eq!(get(&s, TOK_S_RUST).unit, "tokens/s");
    }

    #[test]
    fn registry_names_match_jope_telemetry() {
        let summary = include_str!("../../jope-telemetry/src/summary.rs");
        for (constant, value) in [
            ("TOK_S_RUST", TOK_S_RUST),
            ("TOK_S_ALL_LANGUAGES", TOK_S_ALL_LANGUAGES),
            ("TOK_S_RUST_TIMED_ALONE", TOK_S_RUST_TIMED_ALONE),
        ] {
            assert!(
                summary.contains(&format!("pub const {constant}: &str = \"{value}\";")),
                "{constant} drifted from jope-telemetry"
            );
        }
    }

    #[test]
    fn counters_feed_rates_times_and_reported_rows() {
        let t0 = Instant::now();
        let counters = Counters::default();
        let mut fake = Fake {
            process: [(ENGINE, reading(0.0, 1))].into(),
            gpus: Some(vec![GpuReading {
                uuid: "GPU-abc".into(),
                util_pct: Some(37.5),
                mem_used_bytes: Some(1 << 30),
            }]),
            ..Fake::default()
        };
        let mut telemetry = Telemetry::new(ENGINE, t0);
        telemetry.tick(t0, &mut fake, &counters);
        for _ in 0..6 {
            counters.rung0_completed();
        }
        counters.evaluation_failed("timeout");
        counters.evaluation_failed("timeout");
        counters.evaluation_failed("memory_limit");
        counters.champion_committed(t0 + Duration::from_millis(1500));
        counters.champion_committed(t0 + Duration::from_millis(2500));
        counters.report(
            Reported::RankerScoresPerS,
            1234.0,
            t0 + Duration::from_millis(2000),
        );
        counters.report(
            Reported::StoreBytes,
            4096.0,
            t0 + Duration::from_millis(2000),
        );
        let s = telemetry.tick(t0 + Duration::from_secs(3), &mut fake, &counters);
        assert_eq!(num(get(&s, "eval.per_s")), 2.0);
        assert_eq!(get(&s, "eval.failed").value, Some(bytes(3)));
        assert_eq!(get(&s, "eval.failed.timeout").value, Some(bytes(2)));
        assert_eq!(num(get(&s, "time.first_model_s")), 1.5);
        assert_eq!(num(get(&s, "time.best_s")), 2.5);
        let ranker = get(&s, "ranker.scores_per_s");
        assert_eq!((num(ranker), ranker.age_ms), (1234.0, 1000));
        assert_eq!(get(&s, "store.bytes").value, Some(bytes(4096)));
        let gpu = get(&s, "gpu.util_pct");
        assert_eq!((gpu.scope.as_str(), num(gpu)), ("device:GPU-abc", 37.5));
        assert_eq!(get(&s, "gpu.mem_used").value, Some(bytes(1 << 30)));
        let json = telemetry.summary().to_json();
        assert_eq!(json["evaluations_rung0"], 6);
        assert_eq!(json["evaluations_failed"]["timeout"], 2);
    }

    #[test]
    fn service_publishes_ephemeral_frames_and_returns_a_summary() {
        let (_dir, log, snapshot) = crate::events::tests::fixture();
        let mut frames = log.subscribe_telemetry();
        let counters = Arc::new(Counters::default());
        let fake = Fake {
            process: [(std::process::id(), reading(0.0, 4096))].into(),
            ..Fake::default()
        };
        let service = TelemetryService::spawn(
            snapshot.run_id.clone(),
            log.clone(),
            Box::new(fake),
            counters,
            Duration::from_millis(5),
        );
        let mut received = 0;
        let deadline = Instant::now() + Duration::from_secs(10);
        while received < 3 && Instant::now() < deadline {
            match frames.try_recv() {
                Ok(frame) => {
                    assert_eq!(frame.run_id, snapshot.run_id);
                    let TelemetryBody::Telemetry(samples) = frame.event;
                    assert!(samples.iter().any(|s| s.name == "cpu.cores"));
                    received += 1;
                }
                Err(_) => thread::sleep(Duration::from_millis(2)),
            }
        }
        let summary = service.stop();
        assert!(received >= 3, "received {received} frames");
        assert!(summary.samples >= 3);
        assert_eq!(summary.mem_rss_peak_bytes, Some(4096));
    }

    /// Gate T3: telemetry on or off yields the identical champion and committed history.
    #[test]
    fn telemetry_on_or_off_leaves_the_search_identical() {
        use crate::run::{Control, Evaluator, FakeEvaluator, RunController};
        use jope_api::{Effort, RunPhase};
        use std::num::NonZeroU32;
        fn search() -> Vec<String> {
            let mut run = RunController::new(
                "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
                Effort::Medium,
                NonZeroU32::new(2).unwrap(),
            );
            for phase in [RunPhase::Encoding, RunPhase::Seeding, RunPhase::Searching] {
                run.advance_phase(phase).unwrap();
            }
            let fake = FakeEvaluator::new(Duration::ZERO);
            for round in 0..20u64 {
                let requests: Vec<_> = (0..3u32)
                    .map(|slot| {
                        run.dispatch(round, slot, 31 * round + u64::from(slot))
                            .unwrap()
                    })
                    .collect();
                for request in &requests {
                    run.receive(fake.evaluate(request)).unwrap();
                }
                run.round_committed().unwrap();
            }
            run.control(Control::Cancel, None).unwrap();
            run.committed().iter().map(|r| format!("{r:?}")).collect()
        }
        let off = search();
        let (_dir, log, snapshot) = crate::events::tests::fixture();
        let service = TelemetryService::spawn(
            snapshot.run_id,
            log,
            Box::new(SystemSensors::new(std::env::temp_dir())),
            Arc::new(Counters::default()),
            Duration::from_millis(1),
        );
        let on = search();
        let summary = service.stop();
        assert!(summary.samples > 0);
        assert_eq!(off, on);
    }

    /// Gate T3: sampling cost per tick is far below 2% of the 1 s interval.
    #[test]
    fn real_sensor_tick_cost_is_below_two_percent_of_the_interval() {
        let counters = Counters::default();
        counters.set_workers(&[std::process::id()]);
        let mut sensors = SystemSensors::new(std::env::temp_dir());
        let mut telemetry = Telemetry::new(std::process::id(), Instant::now());
        let ticks = 200;
        let start = Instant::now();
        for _ in 0..ticks {
            std::hint::black_box(telemetry.tick(Instant::now(), &mut sensors, &counters));
        }
        let per_tick = start.elapsed() / ticks;
        let budget = DEFAULT_INTERVAL / 50;
        println!("mean tick {per_tick:?}; budget {budget:?}");
        assert!(
            per_tick < budget,
            "tick {per_tick:?} exceeds 2% of {DEFAULT_INTERVAL:?}"
        );
    }

    /// Fixed-work wall-time benchmark for gate T3; run explicitly:
    /// `cargo test -p jope-service --lib telemetry::tests::fixed_work_overhead -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn fixed_work_overhead() {
        fn work() -> Duration {
            let start = Instant::now();
            let mut state = 0x9e3779b97f4a7c15u64;
            for i in 0..400_000_000u64 {
                state = (state ^ i).wrapping_mul(0xbf58476d1ce4e5b9).rotate_left(17);
            }
            std::hint::black_box(state);
            start.elapsed()
        }
        work();
        let off: Duration = (0..3).map(|_| work()).min().unwrap();
        let (_dir, log, snapshot) = crate::events::tests::fixture();
        let service = TelemetryService::spawn(
            snapshot.run_id,
            log,
            Box::new(SystemSensors::new(std::env::temp_dir())),
            Arc::new(Counters::default()),
            DEFAULT_INTERVAL,
        );
        let on: Duration = (0..3).map(|_| work()).min().unwrap();
        service.stop();
        let overhead = on.as_secs_f64() / off.as_secs_f64() - 1.0;
        println!(
            "fixed work off {off:?} on {on:?} overhead {:.3}%",
            overhead * 100.0
        );
    }
}

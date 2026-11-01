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
pub trait GpuQuery: Send {
    fn initialize(&mut self) -> Result<(), NvmlError>;
    fn reset(&mut self);
    fn device_uuids(&self) -> Result<Vec<String>, NvmlError>;
    fn utilization_pct(&self, uuid: &str) -> Result<u32, NvmlError>;
    fn memory_used_bytes(&self, uuid: &str) -> Result<u64, NvmlError>;
}

/// `nvml-wrapper` loads the NVML library at runtime, so a host without the NVIDIA driver
/// returns an error here rather than failing to start.
#[derive(Default)]
pub struct SystemGpu {
    nvml: Option<Nvml>,
}

impl SystemGpu {
    fn nvml(&self) -> Result<&Nvml, NvmlError> {
        self.nvml.as_ref().ok_or(NvmlError::Uninitialized)
    }
}

impl GpuQuery for SystemGpu {
    fn initialize(&mut self) -> Result<(), NvmlError> {
        self.nvml = None;
        self.nvml = Some(Nvml::init()?);
        Ok(())
    }
    fn reset(&mut self) {
        self.nvml = None;
    }
    fn device_uuids(&self) -> Result<Vec<String>, NvmlError> {
        let nvml = self.nvml()?;
        (0..nvml.device_count()?)
            .map(|index| nvml.device_by_index(index)?.uuid())
            .collect()
    }
    fn utilization_pct(&self, uuid: &str) -> Result<u32, NvmlError> {
        Ok(self.nvml()?.device_by_uuid(uuid)?.utilization_rates()?.gpu)
    }
    fn memory_used_bytes(&self, uuid: &str) -> Result<u64, NvmlError> {
        Ok(self.nvml()?.device_by_uuid(uuid)?.memory_info()?.used)
    }
}

/// Stable TEL-05 reason codes (`nvml_device::UnavailableReason`) for an NVML error.
fn nvml_reason(error: &NvmlError) -> UnavailableReason {
    match error {
        NvmlError::LibraryNotFound | NvmlError::LibloadingError(_) => {
            UnavailableReason::MissingLibrary
        }
        NvmlError::DriverNotLoaded => UnavailableReason::MissingDriver,
        NvmlError::NoPermission => UnavailableReason::PermissionDenied,
        NvmlError::GpuLost => UnavailableReason::DeviceLost,
        _ => UnavailableReason::NvmlError,
    }
}

const NVML_INITIAL_BACKOFF: Duration = Duration::from_millis(100);
const NVML_MAX_BACKOFF: Duration = Duration::from_secs(5);

/// Device-wide GPU readings with bounded exponential retry after NVML failures. It never
/// panics and never synchronizes the device (§9.1 Rates).
pub struct NvmlSensor<Q: GpuQuery> {
    query: Q,
    initialized: bool,
    backoff: Duration,
    retry_at: Option<Instant>,
    last_reason: Option<UnavailableReason>,
}

impl<Q: GpuQuery> NvmlSensor<Q> {
    pub fn new(query: Q) -> Self {
        Self {
            query,
            initialized: false,
            backoff: NVML_INITIAL_BACKOFF,
            retry_at: None,
            last_reason: None,
        }
    }

    fn fail(&mut self, error: &NvmlError, now: Instant) -> String {
        let reason = nvml_reason(error);
        self.query.reset();
        self.initialized = false;
        self.last_reason = Some(reason);
        self.retry_at = Some(now + self.backoff);
        self.backoff = (self.backoff * 2).min(NVML_MAX_BACKOFF);
        reason.as_str().to_owned()
    }

    pub fn readings(&mut self, now: Instant) -> Result<Vec<GpuReading>, String> {
        if self.retry_at.is_some_and(|at| now < at) {
            let reason = self.last_reason.unwrap_or(UnavailableReason::NvmlError);
            return Err(reason.as_str().to_owned());
        }
        self.retry_at = None;
        if !self.initialized {
            if let Err(error) = self.query.initialize() {
                return Err(self.fail(&error, now));
            }
            self.initialized = true;
        }
        let uuids = match self.query.device_uuids() {
            Ok(uuids) => uuids,
            Err(error) => return Err(self.fail(&error, now)),
        };
        self.backoff = NVML_INITIAL_BACKOFF;
        self.last_reason = None;
        Ok(uuids
            .into_iter()
            .map(|uuid| GpuReading {
                util_pct: self
                    .query
                    .utilization_pct(&uuid)
                    .map(f64::from)
                    .map_err(|e| nvml_reason(&e).as_str().to_owned()),
                mem_used_bytes: self
                    .query
                    .memory_used_bytes(&uuid)
                    .map_err(|e| nvml_reason(&e).as_str().to_owned()),
                uuid,
            })
            .collect())
    }
}

/// Host sensors: WRK-03 per-PID sampling plus OS memory, CPU and disk queries.
pub struct SystemSensors {
    store: PathBuf,
    gpu: Option<NvmlSensor<SystemGpu>>,
}

impl SystemSensors {
    /// `store` is the `$JOPEDIME_HOME` volume used for `disk.free`. GPU metrics come from
    /// NVML, loaded at runtime: no library, driver or device yields `unavailable:<reason>`.
    pub fn new(store: impl Into<PathBuf>) -> Self {
        Self {
            store: store.into(),
            gpu: Some(NvmlSensor::new(SystemGpu::default())),
        }
    }

    /// Host sensors without GPU sampling (`gpu.*` = `unavailable:gpu_sampling_disabled`).
    pub fn without_gpu(store: impl Into<PathBuf>) -> Self {
        Self {
            store: store.into(),
            gpu: None,
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
        match &mut self.gpu {
            Some(gpu) => gpu.readings(Instant::now()),
            None => Err("gpu_sampling_disabled".to_owned()),
        }
    }
}

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

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod platform {
    use std::{ffi::CString, os::unix::ffi::OsStrExt, path::Path};

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
        // SAFETY: this C statistics struct consists of integer fields and accepts zeroes.
        let mut stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
        let mut count = libc::HOST_VM_INFO64_COUNT;
        // SAFETY: stats has room for HOST_VM_INFO64_COUNT integer words; count is writable.
        let status = unsafe {
            libc::host_statistics64(
                host,
                libc::HOST_VM_INFO64,
                (&mut stats as *mut libc::vm_statistics64).cast(),
                &mut count,
            )
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
        // XNU mach/vm_statistics.h documents speculative_count as already included
        // in free_count. Adding it again would overstate admission headroom.
        let pages = u64::from(stats.free_count)
            .checked_add(u64::from(stats.inactive_count))
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
        let path = CString::new(path.as_os_str().as_bytes()).map_err(|_| "invalid_store_path")?;
        // SAFETY: an all-zero statvfs is valid output storage for statvfs below.
        let mut stats: libc::statvfs = unsafe { std::mem::zeroed() };
        // SAFETY: path is NUL-terminated; stats is valid writable output storage.
        if unsafe { libc::statvfs(path.as_ptr(), &mut stats) } != 0 {
            return Err("statvfs_failed".to_owned());
        }
        Ok((stats.f_bavail as u64).saturating_mul(stats.f_frsize as u64))
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
mod tests {
    use super::*;

    /// A fresh, empty store directory removed on drop.
    fn test_store() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix("jope-telemetry-")
            .tempdir()
            .unwrap()
    }

    const ENGINE: u32 = 1;

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_system_memory_sensor_reports_available_bytes() {
        let available = platform::mem_available().expect("native Mach VM memory sensor");
        assert!(available > 0, "available memory must be nonzero");
    }

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
                .ok_or_else(|| "fake_gpu_absent".to_owned())
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
        counters.set_workers(&[3]);
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
        assert_eq!(reason("gpu.util_pct"), "unavailable:fake_gpu_absent");
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
            "unavailable:fake_gpu_absent"
        );
        assert_eq!(summary.tokens_per_s_rust, None);
    }

    #[test]
    fn a_failed_live_worker_makes_totals_unavailable_not_falsely_complete() {
        let t0 = Instant::now();
        let counters = Counters::default();
        let mut fake = Fake::default();
        let mut telemetry = Telemetry::new(ENGINE, t0);
        fake.process.insert(ENGINE, reading(0.0, 100));
        fake.process.insert(2, reading(0.0, 10));
        counters.set_workers(&[2]);
        telemetry.tick(t0, &mut fake, &counters);
        // Worker 2 is still listed as live but its reading fails.
        fake.process.remove(&2);
        fake.process.insert(ENGINE, reading(1.0, 100));
        let s = telemetry.tick(t0 + Duration::from_secs(1), &mut fake, &counters);
        for name in ["cpu.cores", "cpu.host_pct", "mem.rss"] {
            let metric = get(&s, name);
            assert_eq!(metric.value, None, "{name}");
            assert_eq!(
                metric.availability, "unavailable:worker_process_exited",
                "{name}"
            );
        }
        assert_eq!(telemetry.summary().mem_rss_peak_bytes, Some(110));
    }

    #[test]
    fn store_bytes_are_exact_and_invalid_reports_are_unavailable() {
        let t0 = Instant::now();
        let counters = Counters::default();
        let exact = (1u64 << 53) + 1; // not representable as f64
        counters.report_store_bytes(exact, t0);
        counters.report(Reported::RankerLatencyMs, -3.0, t0);
        counters.report(Reported::IoReadGbS, f64::INFINITY, t0);
        counters.report(Reported::PredictRowsPerS, f64::NAN, t0);
        let mut fake = Fake::default();
        let s = Telemetry::new(ENGINE, t0).tick(t0, &mut fake, &counters);
        assert_eq!(get(&s, "store.bytes").value, Some(bytes(exact)));
        assert_eq!(
            get(&s, "ranker.latency_ms").availability,
            "unavailable:negative_value"
        );
        assert_eq!(
            get(&s, "io.read_gb_s").availability,
            "unavailable:non_finite"
        );
        assert_eq!(
            get(&s, "predict.rows_per_s").availability,
            "unavailable:non_finite"
        );
        for name in ["ranker.latency_ms", "io.read_gb_s", "predict.rows_per_s"] {
            assert_eq!(get(&s, name).value, None);
        }
    }

    #[test]
    fn allowed_cpus_follow_the_tightest_quota_on_the_processes_own_cgroup_path() {
        let files: BTreeMap<&str, &str> = [
            ("", "max 100000\n"),
            ("system.slice", "800000 100000\n"),
            ("system.slice/jope.service", "250000 100000\n"),
            ("system.slice/jope.service/worker", "max 100000\n"),
        ]
        .into();
        let read = |path: &str| files.get(path).map(|s| s.to_string());
        let membership = "0::/system.slice/jope.service/worker\n";
        assert_eq!(cgroup_allowed_cpus(membership, 64.0, read), Ok(2.5));
        // Affinity can be tighter than any quota.
        assert_eq!(cgroup_allowed_cpus(membership, 2.0, read), Ok(2.0));
        // A sibling cgroup does not inherit an unrelated quota.
        assert_eq!(
            cgroup_allowed_cpus("0::/user.slice\n", 64.0, read),
            Ok(64.0)
        );
        // cgroup v1 only (no 0:: line): affinity alone.
        assert_eq!(
            cgroup_allowed_cpus("4:cpu,cpuacct:/x\n", 12.0, read),
            Ok(12.0)
        );
        // Root membership reads the root file only.
        assert_eq!(
            cgroup_allowed_cpus("0::/\n", 16.0, |p| (p.is_empty())
                .then(|| "400000 100000".to_string())),
            Ok(4.0)
        );
    }

    #[test]
    fn invalid_cgroup_membership_never_reads_quota_outside_the_mount() {
        for path in [
            "/../outside",
            "/a/../../outside",
            "/a/../b",
            "/./a",
            "relative",
            "",
        ] {
            let reads = std::cell::Cell::new(0);
            let result = cgroup_allowed_cpus(&format!("0::{path}\n"), 8.0, |_| {
                reads.set(reads.get() + 1);
                Some("max 100000".to_owned())
            });
            assert_eq!(reads.get(), 0, "invalid cgroup membership was read: {path}");
            assert_eq!(result, Err("cgroup_path_unresolved".to_owned()));
        }
    }

    /// (uuid, utilization, memory used) for one fake device.
    type FakeDevice = (String, Result<u32, NvmlError>, Result<u64, NvmlError>);

    #[derive(Default)]
    struct FakeGpu {
        init_error: Option<NvmlError>,
        list_error: Option<NvmlError>,
        devices: Vec<FakeDevice>,
        inits: u32,
        resets: u32,
    }
    impl GpuQuery for FakeGpu {
        fn initialize(&mut self) -> Result<(), NvmlError> {
            self.inits += 1;
            match self.init_error.take() {
                Some(error) => Err(error),
                None => Ok(()),
            }
        }
        fn reset(&mut self) {
            self.resets += 1;
        }
        fn device_uuids(&self) -> Result<Vec<String>, NvmlError> {
            match &self.list_error {
                Some(_) => Err(NvmlError::GpuLost),
                None => Ok(self.devices.iter().map(|d| d.0.clone()).collect()),
            }
        }
        fn utilization_pct(&self, uuid: &str) -> Result<u32, NvmlError> {
            match &self.devices.iter().find(|d| d.0 == uuid).unwrap().1 {
                Ok(v) => Ok(*v),
                Err(_) => Err(NvmlError::NoPermission),
            }
        }
        fn memory_used_bytes(&self, uuid: &str) -> Result<u64, NvmlError> {
            match &self.devices.iter().find(|d| d.0 == uuid).unwrap().2 {
                Ok(v) => Ok(*v),
                Err(_) => Err(NvmlError::GpuLost),
            }
        }
    }

    #[test]
    fn missing_nvml_is_unavailable_with_bounded_retry_and_never_panics() {
        let t0 = Instant::now();
        let mut sensor = NvmlSensor::new(FakeGpu {
            init_error: Some(NvmlError::LibraryNotFound),
            ..FakeGpu::default()
        });
        assert_eq!(sensor.readings(t0), Err("missing_library".to_owned()));
        // Within the backoff window NVML is not re-initialized.
        assert_eq!(
            sensor.readings(t0 + Duration::from_millis(50)),
            Err("missing_library".to_owned())
        );
        assert_eq!(sensor.query.inits, 1);
        // After the backoff it retries; the driver is now present but has no devices.
        assert_eq!(sensor.readings(t0 + Duration::from_millis(150)), Ok(vec![]));
        assert_eq!(sensor.query.inits, 2);

        for (error, reason) in [
            (NvmlError::DriverNotLoaded, "missing_driver"),
            (NvmlError::NoPermission, "permission_denied"),
            (NvmlError::Uninitialized, "nvml_error"),
        ] {
            let mut sensor = NvmlSensor::new(FakeGpu {
                init_error: Some(error),
                ..FakeGpu::default()
            });
            assert_eq!(sensor.readings(t0), Err(reason.to_owned()));
        }
    }

    #[test]
    fn nvml_readings_are_device_scoped_and_per_device_failures_carry_a_reason() {
        let t0 = Instant::now();
        let mut sensor = NvmlSensor::new(FakeGpu {
            devices: vec![
                ("GPU-a".into(), Ok(37), Ok(1 << 30)),
                (
                    "GPU-b".into(),
                    Err(NvmlError::NoPermission),
                    Err(NvmlError::GpuLost),
                ),
            ],
            ..FakeGpu::default()
        });
        let readings = sensor.readings(t0).unwrap();
        assert_eq!(readings[0].util_pct, Ok(37.0));
        assert_eq!(readings[0].mem_used_bytes, Ok(1 << 30));
        assert_eq!(readings[1].util_pct, Err("permission_denied".to_owned()));
        assert_eq!(readings[1].mem_used_bytes, Err("device_lost".to_owned()));

        let mut fake = Fake {
            gpus: Some(readings),
            ..Fake::default()
        };
        let s = Telemetry::new(ENGINE, t0).tick(t0, &mut fake, &Counters::default());
        let device = |name: &str, scope: &str| {
            s.iter()
                .find(|m| m.name == name && m.scope == scope)
                .unwrap()
                .clone()
        };
        assert_eq!(num(&device("gpu.util_pct", "device:GPU-a")), 37.0);
        assert_eq!(
            device("gpu.util_pct", "device:GPU-b").availability,
            "unavailable:permission_denied"
        );
        assert_eq!(
            device("gpu.mem_used", "device:GPU-b").availability,
            "unavailable:device_lost"
        );
    }

    #[test]
    fn a_lost_device_resets_nvml_and_reinitializes_after_backoff() {
        let t0 = Instant::now();
        let mut sensor = NvmlSensor::new(FakeGpu {
            list_error: Some(NvmlError::GpuLost),
            devices: vec![("GPU-a".into(), Ok(5), Ok(7))],
            ..FakeGpu::default()
        });
        assert_eq!(sensor.readings(t0), Err("device_lost".to_owned()));
        assert_eq!(sensor.query.resets, 1);
        sensor.query.list_error = None;
        assert_eq!(
            sensor.readings(t0 + Duration::from_millis(10)),
            Err("device_lost".to_owned())
        );
        let readings = sensor.readings(t0 + Duration::from_millis(150)).unwrap();
        assert_eq!(sensor.query.inits, 2);
        assert_eq!(readings[0].util_pct, Ok(5.0));
    }

    #[test]
    fn gpu_less_system_sensors_report_a_reason() {
        let store = test_store();
        let mut sensors = SystemSensors::without_gpu(store.path());
        assert_eq!(sensors.gpus(), Err("gpu_sampling_disabled".to_owned()));
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
    fn registry_names_are_the_jope_telemetry_constants() {
        assert_eq!(TOK_S_RUST, "throughput.tokens_per_s.rust");
        assert_eq!(TOK_S_ALL_LANGUAGES, "throughput.tokens_per_s.all_languages");
        assert_eq!(
            TOK_S_RUST_TIMED_ALONE,
            "throughput.tokens_per_s.rust_timed_alone"
        );
    }

    #[test]
    fn counters_feed_rates_times_and_reported_rows() {
        let t0 = Instant::now();
        let counters = Counters::default();
        let mut fake = Fake {
            process: [(ENGINE, reading(0.0, 1))].into(),
            gpus: Some(vec![GpuReading {
                uuid: "GPU-abc".into(),
                util_pct: Ok(37.5),
                mem_used_bytes: Ok(1 << 30),
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
        counters.report_store_bytes(4096, t0 + Duration::from_millis(2000));
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
            Instant::now(),
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

    #[test]
    fn delayed_attachment_measures_times_from_the_canonical_run_start() {
        let (_dir, log, snapshot) = crate::events::tests::fixture();
        let mut frames = log.subscribe_telemetry();
        let counters = Arc::new(Counters::default());
        let run_started = Instant::now();
        // The run started, a champion was committed, and only then telemetry attached.
        let committed = run_started + Duration::from_millis(40);
        counters.champion_committed(committed);
        while Instant::now() < committed + Duration::from_millis(20) {
            thread::sleep(Duration::from_millis(5));
        }
        let fake = Fake {
            process: [(std::process::id(), reading(0.0, 1))].into(),
            ..Fake::default()
        };
        let service = TelemetryService::spawn(
            snapshot.run_id,
            log,
            Box::new(fake),
            counters,
            run_started,
            Duration::from_millis(5),
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        let first = loop {
            match frames.try_recv() {
                Ok(frame) => break frame,
                Err(_) if Instant::now() < deadline => thread::sleep(Duration::from_millis(2)),
                Err(error) => panic!("no telemetry frame: {error:?}"),
            }
        };
        let summary = service.stop();
        let TelemetryBody::Telemetry(samples) = first.event;
        let first_model = num(get(&samples, "time.first_model_s"));
        assert!(
            (first_model - 0.040).abs() < 1e-9,
            "first_model_s {first_model}"
        );
        assert_eq!(summary.time_first_model_s, Some(first_model));

        // A mark before the run start is refused rather than published as a measured 0.
        let before = Counters::default();
        before.champion_committed(run_started);
        let mut fake = Fake::default();
        let s = Telemetry::new(ENGINE, run_started + Duration::from_secs(1)).tick(
            run_started + Duration::from_secs(2),
            &mut fake,
            &before,
        );
        assert_eq!(
            get(&s, "time.first_model_s").availability,
            "unavailable:before_run_start"
        );
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
        let mut frames = log.subscribe_telemetry();
        let store = test_store();
        let service = TelemetryService::spawn(
            snapshot.run_id,
            log,
            Box::new(SystemSensors::without_gpu(store.path())),
            Arc::new(Counters::default()),
            Instant::now(),
            Duration::from_millis(1),
        );
        // The zero-duration fake search can finish before the sampler is scheduled.
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match frames.try_recv() {
                Ok(_) => break,
                Err(_) if Instant::now() < deadline => thread::sleep(Duration::from_millis(2)),
                Err(error) => panic!("telemetry did not start: {error:?}"),
            }
        }
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
        let store = test_store();
        let mut sensors = SystemSensors::without_gpu(store.path());
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
        let store = test_store();
        let service = TelemetryService::spawn(
            snapshot.run_id,
            log,
            Box::new(SystemSensors::without_gpu(store.path())),
            Arc::new(Counters::default()),
            Instant::now(),
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

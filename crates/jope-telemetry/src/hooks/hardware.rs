//! GPU readings for the DIME trainers, ported from `dime_score/train.py` (`_gpu_telemetry`,
//! `_HardwareMonitor`) and the `RatchetLiveTelemetry.v2` snapshot that `broad_train.py` wrote.
//!
//! NVML replaces the `nvidia-smi --query-gpu` subprocess. It reads the same eight values of GPU 0
//! (`nvidia-smi` and NVML both order devices by PCI bus id), `memory_info` is the v2 query that
//! `nvidia-smi` reports, and a reading stays all-or-nothing: one failed query leaves every value
//! missing, as one unparsable `nvidia-smi` field did.

use crate::hooks::json::J;
use anyhow::{anyhow, Result};
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use nvml_wrapper::Nvml;
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The snapshot keys, in the order the trainers merged them into their records.
pub const GPU_KEYS: [&str; 10] = [
    "gpu_utilization_pct",
    "gpu_memory_used_mib",
    "gpu_memory_total_mib",
    "gpu_temperature_c",
    "gpu_power_watts",
    "gpu_graphics_clock_mhz",
    "gpu_sm_clock_mhz",
    "gpu_memory_clock_mhz",
    "gpu_headroom_mib",
    "gpu_headroom_ok",
];

pub const RATCHET_SCHEMA: &str = "RatchetLiveTelemetry.v2";
const RATCHET_PERIOD: Duration = Duration::from_millis(100);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuReading {
    pub utilization_pct: f64,
    pub memory_used_mib: f64,
    pub memory_total_mib: f64,
    pub temperature_c: f64,
    pub power_watts: f64,
    pub graphics_clock_mhz: f64,
    pub sm_clock_mhz: f64,
    pub memory_clock_mhz: f64,
}

impl GpuReading {
    pub fn headroom_mib(&self) -> f64 {
        self.memory_total_mib - self.memory_used_mib
    }
}

/// The `GPU_KEYS` record; every value is `null` without a reading.
pub fn snapshot(reading: Option<&GpuReading>, minimum_headroom_mib: f64) -> J {
    let Some(r) = reading else {
        return J::obj(GPU_KEYS.iter().map(|k| (*k, J::Null)));
    };
    let values = [
        r.utilization_pct,
        r.memory_used_mib,
        r.memory_total_mib,
        r.temperature_c,
        r.power_watts,
        r.graphics_clock_mhz,
        r.sm_clock_mhz,
        r.memory_clock_mhz,
        r.headroom_mib(),
    ];
    let mut fields: Vec<(&str, J)> = GPU_KEYS
        .iter()
        .zip(values)
        .map(|(k, v)| (*k, J::Float(v)))
        .collect();
    fields.push((
        GPU_KEYS[9],
        J::Bool(r.headroom_mib() >= minimum_headroom_mib),
    ));
    J::obj(fields)
}

/// NVML on GPU 0. A host without NVML retries the load on each read, as each `nvidia-smi` call
/// did, so a driver that appears later is picked up.
pub struct GpuReader {
    nvml: Option<Nvml>,
}

impl Default for GpuReader {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuReader {
    pub fn new() -> Self {
        Self {
            nvml: Nvml::init().ok(),
        }
    }

    pub fn read(&mut self) -> Option<GpuReading> {
        if self.nvml.is_none() {
            self.nvml = Nvml::init().ok();
        }
        let device = self.nvml.as_ref()?.device_by_index(0).ok()?;
        let utilization = device.utilization_rates().ok()?;
        let memory = device.memory_info().ok()?;
        let temperature = device.temperature(TemperatureSensor::Gpu).ok()?;
        let power_mw = device.power_usage().ok()?;
        let clock = |c| device.clock_info(c).ok().map(f64::from);
        Some(GpuReading {
            utilization_pct: utilization.gpu.into(),
            memory_used_mib: mib(memory.used),
            memory_total_mib: mib(memory.total),
            temperature_c: temperature.into(),
            power_watts: f64::from(power_mw) / 1000.0,
            graphics_clock_mhz: clock(Clock::Graphics)?,
            sm_clock_mhz: clock(Clock::SM)?,
            memory_clock_mhz: clock(Clock::Memory)?,
        })
    }
}

/// Whole MiB rounded up, as `nvidia-smi --format=nounits` prints memory (17301.125 MiB of
/// v2 `used` prints as 17302).
fn mib(bytes: u64) -> f64 {
    bytes.div_ceil(1 << 20) as f64
}

pub type SharedReading = Arc<Mutex<Option<GpuReading>>>;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Background GPU reader: one reading at construction, then one every `1 / hz` seconds after
/// `start`, measured from the end of the previous reading.
pub struct HardwareMonitor {
    latest: SharedReading,
    period: Duration,
    reader: Option<GpuReader>,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl HardwareMonitor {
    pub fn new(hz: f64) -> Result<Self> {
        let period = if hz > 0.0 {
            Duration::try_from_secs_f64(1.0 / hz).ok()
        } else {
            None
        }
        .ok_or_else(|| anyhow!("hardware telemetry frequency must be positive"))?;
        let mut reader = GpuReader::new();
        let first = reader.read();
        Ok(Self {
            latest: Arc::new(Mutex::new(first)),
            period,
            reader: Some(reader),
            stop: Arc::new(AtomicBool::new(false)),
            join: None,
        })
    }

    pub fn start(&mut self) -> Result<()> {
        let mut reader = self
            .reader
            .take()
            .ok_or_else(|| anyhow!("threads can only be started once"))?;
        let latest = Arc::clone(&self.latest);
        let stop = Arc::clone(&self.stop);
        let period = self.period;
        let join = thread::Builder::new()
            .name("dime-hardware-telemetry".to_owned())
            .stack_size(512 * 1024)
            .spawn(move || {
                let mut next = Instant::now() + period;
                while !stop.load(Ordering::Relaxed) {
                    let now = Instant::now();
                    if now < next {
                        thread::park_timeout(next - now);
                        continue;
                    }
                    let reading = reader.read();
                    *lock(&latest) = reading;
                    next = Instant::now() + period;
                }
            })?;
        self.join = Some(join);
        Ok(())
    }

    pub fn reading(&self) -> Option<GpuReading> {
        *lock(&self.latest)
    }

    pub fn shared(&self) -> SharedReading {
        Arc::clone(&self.latest)
    }

    /// Stop and wait up to `max(2 s, period + 0.5 s)` for the reader, then leave it detached,
    /// the bounded join the Python monitor had.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let Some(join) = self.join.take() else {
            return;
        };
        join.thread().unpark();
        let deadline =
            Instant::now() + (self.period + Duration::from_millis(500)).max(Duration::from_secs(2));
        while !join.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        if join.is_finished() {
            let _ = join.join();
        }
    }
}

impl Drop for HardwareMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}

/// `live.v2.json` for the broad trainer: at most one atomic rewrite per 100 ms with the latest
/// GPU reading, in the schema `jope-campaigns`' ratchet view reads.
pub struct RatchetLive {
    path: PathBuf,
    owner: String,
    node: String,
    scq_job_id: Option<String>,
    reading: SharedReading,
    started: Instant,
    last: Option<Instant>,
}

impl RatchetLive {
    pub fn new(path: impl Into<PathBuf>, reading: SharedReading, owner: &str) -> Self {
        let host = std::env::var_os("HOSTNAME")
            .map_or_else(|| "local".to_owned(), |h| h.to_string_lossy().into_owned());
        Self {
            path: path.into(),
            owner: owner.to_owned(),
            node: host.split('.').next().unwrap_or_default().to_owned(),
            scq_job_id: std::env::var_os("SCQ_JOB_ID").map(|v| v.to_string_lossy().into_owned()),
            reading,
            started: Instant::now(),
            last: None,
        }
    }

    /// Rewrite the file unless the last write was under 100 ms ago; `true` when it wrote.
    /// `best_known` reports `time_to_best_seconds` (time since construction) instead of `null`.
    pub fn tick(&mut self, best_known: bool) -> std::io::Result<bool> {
        let now = Instant::now();
        if self.last.is_some_and(|last| now - last < RATCHET_PERIOD) {
            return Ok(false);
        }
        let reading = *lock(&self.reading);
        let since = best_known.then(|| self.started.elapsed().as_secs_f64());
        let document = self.document(reading.as_ref(), unix_ms(), since);
        let mut payload = serde_json::to_string_pretty(&document).map_err(std::io::Error::other)?;
        payload.push('\n');
        write_atomic(&self.path, payload.as_bytes())?;
        self.last = Some(now);
        Ok(true)
    }

    pub fn document(
        &self,
        reading: Option<&GpuReading>,
        now_ms: i64,
        time_to_best: Option<f64>,
    ) -> Value {
        let used = reading.map(|r| r.memory_used_mib);
        let total = reading.map(|r| r.memory_total_mib);
        json!({
            "schema": RATCHET_SCHEMA,
            "updated_at_ms": now_ms,
            "render_hz": 10,
            "hardware_hz": 1,
            "nodes": [{
                "node": self.node,
                "role": "trainer",
                "state": "running",
                "scq_job_id": self.scq_job_id,
                "scq_owner": self.owner,
                "gpu_utilization_percent": reading.map(|r| r.utilization_pct),
                "used_vram_mib": used.map(|u| u as i64),
                "free_vram_mib": total.zip(used).map(|(t, u)| (t - u) as i64),
                "total_vram_mib": total.map(|t| t as i64),
                "updated_at_ms": now_ms,
            }],
            "labels": {
                "labels_per_second": 0.0,
                "evaluator_backlog": 0,
                "failures_total": 0,
                "self_play_wins_total": 0,
                "time_to_best_seconds": time_to_best,
            },
            "evaluator_failures": [],
        })
    }
}

fn unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| (d.as_secs_f64() * 1000.0) as i64)
}

/// Write `.<name>.tmp.<pid>` beside `path`, fsync it, and rename it over `path`.
pub fn write_atomic(path: &Path, payload: &[u8]) -> std::io::Result<()> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let temporary = path.with_file_name(format!(".{name}.tmp.{}", std::process::id()));
    let mut file = File::create(&temporary)?;
    file.write_all(payload)?;
    file.flush()?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading() -> GpuReading {
        GpuReading {
            utilization_pct: 87.0,
            memory_used_mib: 20_000.0,
            memory_total_mib: 24_576.0,
            temperature_c: 61.0,
            power_watts: 301.25,
            graphics_clock_mhz: 2520.0,
            sm_clock_mhz: 2520.0,
            memory_clock_mhz: 10501.0,
        }
    }

    #[test]
    fn snapshot_keys_and_headroom() {
        let full = snapshot(Some(&reading()), 4096.0);
        let text = serde_json::to_string(&full).unwrap();
        assert_eq!(
            text,
            "{\"gpu_utilization_pct\":87.0,\"gpu_memory_used_mib\":20000.0,\
             \"gpu_memory_total_mib\":24576.0,\"gpu_temperature_c\":61.0,\
             \"gpu_power_watts\":301.25,\"gpu_graphics_clock_mhz\":2520.0,\
             \"gpu_sm_clock_mhz\":2520.0,\"gpu_memory_clock_mhz\":10501.0,\
             \"gpu_headroom_mib\":4576.0,\"gpu_headroom_ok\":true}"
        );
        assert_eq!(
            snapshot(Some(&reading()), 5000.0).get("gpu_headroom_ok"),
            Some(&J::Bool(false))
        );
        let empty = snapshot(None, 4096.0);
        for key in GPU_KEYS {
            assert_eq!(empty.get(key), Some(&J::Null), "{key}");
        }
        assert_eq!(mib(24_576 * 1024 * 1024), 24_576.0);
        assert_eq!(mib(17_301 * 1024 * 1024 + 131_072), 17_302.0);
    }

    #[test]
    fn monitor_rejects_bad_rates_and_stops() {
        assert!(HardwareMonitor::new(0.0).is_err());
        assert!(HardwareMonitor::new(f64::NAN).is_err());
        assert!(HardwareMonitor::new(1e-300).is_err());
        let mut monitor = HardwareMonitor::new(50.0).unwrap();
        monitor.start().unwrap();
        assert!(monitor.start().is_err());
        thread::sleep(Duration::from_millis(60));
        let started = Instant::now();
        monitor.stop();
        assert!(started.elapsed() < Duration::from_secs(2));
        monitor.stop();
        // Hosts without a GPU read nothing; hosts with one read GPU 0.
        if let Some(r) = monitor.reading() {
            assert!(r.memory_total_mib > 0.0 && r.headroom_mib() <= r.memory_total_mib);
        }
    }

    #[test]
    fn ratchet_document_and_rate_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("live.v2.json");
        let shared: SharedReading = Arc::new(Mutex::new(Some(reading())));
        let mut live = RatchetLive::new(&path, Arc::clone(&shared), "broad_shadow");
        live.node = "xbabe1".to_owned();
        live.scq_job_id = None;
        let doc = live.document(Some(&reading()), 1_700_000_000_123, Some(12.5));
        assert_eq!(
            serde_json::to_string_pretty(&doc).unwrap(),
            r#"{
  "evaluator_failures": [],
  "hardware_hz": 1,
  "labels": {
    "evaluator_backlog": 0,
    "failures_total": 0,
    "labels_per_second": 0.0,
    "self_play_wins_total": 0,
    "time_to_best_seconds": 12.5
  },
  "nodes": [
    {
      "free_vram_mib": 4576,
      "gpu_utilization_percent": 87.0,
      "node": "xbabe1",
      "role": "trainer",
      "scq_job_id": null,
      "scq_owner": "broad_shadow",
      "state": "running",
      "total_vram_mib": 24576,
      "updated_at_ms": 1700000000123,
      "used_vram_mib": 20000
    }
  ],
  "render_hz": 10,
  "schema": "RatchetLiveTelemetry.v2",
  "updated_at_ms": 1700000000123
}"#
        );
        let empty = live.document(None, 1, None);
        assert_eq!(empty["nodes"][0]["free_vram_mib"], Value::Null);
        assert_eq!(empty["labels"]["time_to_best_seconds"], Value::Null);

        assert!(live.tick(false).unwrap());
        assert!(!live.tick(true).unwrap(), "within 100 ms");
        let written: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(written["schema"], RATCHET_SCHEMA);
        assert_eq!(written["nodes"][0]["used_vram_mib"], 20000);
        assert!(
            fs::read_dir(dir.path()).unwrap().count() == 1,
            "no temporary file left"
        );
    }
}

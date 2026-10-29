//! One-hertz, background-only host and device probes.
//!
//! CPU values are read from Linux procfs. NVML values are device-wide context and are never
//! represented as process utilization. Unsupported probes emit a missing-value event once.

use crate::core::ProducerHandle;
use crate::train::{
    CPU_CORE_UTIL, CPU_NODE_UTIL, CPU_PROCESS_RSS_BYTES, CPU_PROCESS_UTIL, GPU_MEMORY_BYTES,
    GPU_UTIL,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

const REASON_NO_VISIBLE_GPU: u16 = 7;
const REASON_PROCFS_UNAVAILABLE: u16 = 8;
const REASON_NVML_UNAVAILABLE: u16 = 9;
const HOST_SAMPLE_MS: u64 = 1_000;

pub struct SystemCollector {
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl SystemCollector {
    pub fn start(mut producer: ProducerHandle, interval: Duration) -> std::io::Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let interval = interval.max(Duration::from_millis(HOST_SAMPLE_MS));
        let join = thread::Builder::new()
            .name("jope-telemetry-host".to_owned())
            .stack_size(512 * 1024)
            .spawn(move || sample_loop(&mut producer, thread_stop, interval))?;
        Ok(Self {
            stop,
            join: Some(join),
        })
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(join) = self.join.take() {
            join.thread().unpark();
            let _ = join.join();
        }
    }
}

impl Drop for SystemCollector {
    fn drop(&mut self) {
        self.stop();
    }
}

fn sample_loop(producer: &mut ProducerHandle, stop: Arc<AtomicBool>, interval: Duration) {
    let mut step = 0_u64;
    let mut previous_cpu = cpu_counters();
    let mut previous_cores = cpu_core_counters();
    let mut previous_process = process_ticks();
    let mut previous_at = std::time::Instant::now();
    let mut node_unavailable = false;
    let mut process_unavailable = false;
    let mut rss_unavailable = false;
    let mut cores_unavailable = false;
    let mut gpu_probe = GpuProbe::new();

    while !stop.load(Ordering::Relaxed) {
        if let Some(rss) = process_rss_bytes() {
            producer.emit_scalar(CPU_PROCESS_RSS_BYTES, 0, step, rss as f64, true);
        } else if !rss_unavailable {
            producer.emit_unavailable(
                CPU_PROCESS_RSS_BYTES,
                0,
                step,
                REASON_PROCFS_UNAVAILABLE,
                true,
            );
            rss_unavailable = true;
        }
        gpu_probe.sample(producer, step);

        thread::park_timeout(interval);
        if stop.load(Ordering::Relaxed) {
            break;
        }

        let now = std::time::Instant::now();
        let current_cpu = cpu_counters();
        let total_delta = current_cpu
            .zip(previous_cpu)
            .map(|((total, _), (prev_total, _))| total.saturating_sub(prev_total))
            .unwrap_or(0);
        let current_cores = cpu_core_counters();
        let node_cores = current_cores.as_ref().map_or(1, Vec::len) as f64;
        if let (Some((_, idle)), Some((_, prev_idle))) = (current_cpu, previous_cpu) {
            let idle_delta = idle.saturating_sub(prev_idle);
            if total_delta > 0 {
                let busy = total_delta.saturating_sub(idle_delta);
                producer.emit_scalar(
                    CPU_NODE_UTIL,
                    0,
                    step,
                    (100.0 * busy as f64 / total_delta as f64).clamp(0.0, 100.0),
                    true,
                );
            }
        } else if !node_unavailable {
            producer.emit_unavailable(CPU_NODE_UTIL, 0, step, REASON_PROCFS_UNAVAILABLE, true);
            node_unavailable = true;
        }
        previous_cpu = current_cpu;

        match (current_cores, previous_cores.as_ref()) {
            (Some(current), Some(previous)) => {
                for (index, ((total, idle), (old_total, old_idle))) in current
                    .iter()
                    .zip(previous)
                    .enumerate()
                    .take(u16::MAX as usize)
                {
                    let total_delta = total.saturating_sub(*old_total);
                    let idle_delta = idle.saturating_sub(*old_idle);
                    if total_delta > 0 {
                        let busy = total_delta.saturating_sub(idle_delta);
                        producer.emit_scalar(
                            CPU_CORE_UTIL,
                            index as u16,
                            step,
                            (100.0 * busy as f64 / total_delta as f64).clamp(0.0, 100.0),
                            true,
                        );
                    }
                }
                previous_cores = Some(current);
            }
            _ if !cores_unavailable => {
                producer.emit_unavailable(CPU_CORE_UTIL, 0, step, REASON_PROCFS_UNAVAILABLE, true);
                previous_cores = cpu_core_counters();
                cores_unavailable = true;
            }
            _ => {}
        }

        if let (Some(current), Some(previous)) = (process_ticks(), previous_process) {
            let tick_delta = current.saturating_sub(previous);
            let elapsed = now.duration_since(previous_at).as_secs_f64();
            if elapsed > 0.0 && total_delta > 0 {
                // Normalize process use to one whole node: 100% means all node CPUs busy.
                let process_pct = 100.0 * tick_delta as f64 / (total_delta as f64 / node_cores);
                producer.emit_scalar(
                    CPU_PROCESS_UTIL,
                    0,
                    step,
                    process_pct.clamp(0.0, 100.0),
                    true,
                );
            } else if !process_unavailable {
                producer.emit_unavailable(
                    CPU_PROCESS_UTIL,
                    0,
                    step,
                    REASON_PROCFS_UNAVAILABLE,
                    true,
                );
                process_unavailable = true;
            }
            previous_process = Some(current);
        } else if !process_unavailable {
            producer.emit_unavailable(CPU_PROCESS_UTIL, 0, step, REASON_PROCFS_UNAVAILABLE, true);
            process_unavailable = true;
        }
        previous_at = now;
        step = step.saturating_add(1);
    }
}

#[cfg(target_os = "linux")]
fn cpu_counters() -> Option<(u64, u64)> {
    let stat = std::fs::read_to_string("/proc/stat").ok()?;
    let mut fields = stat.lines().next()?.split_whitespace();
    if fields.next()? != "cpu" {
        return None;
    }
    let ticks: Vec<u64> = fields.map(str::parse).collect::<Result<_, _>>().ok()?;
    if ticks.len() < 4 {
        return None;
    }
    let total = ticks.iter().take(8).copied().sum();
    Some((
        total,
        ticks[3].saturating_add(ticks.get(4).copied().unwrap_or(0)),
    ))
}

#[cfg(target_os = "linux")]
fn cpu_core_counters() -> Option<Vec<(u64, u64)>> {
    let stat = std::fs::read_to_string("/proc/stat").ok()?;
    let values = stat
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let name = fields.next()?;
            let suffix = name.strip_prefix("cpu")?;
            if suffix.is_empty() || !suffix.chars().all(|ch| ch.is_ascii_digit()) {
                return None;
            }
            let ticks: Vec<u64> = fields.map(str::parse).collect::<Result<_, _>>().ok()?;
            (ticks.len() >= 4).then(|| {
                let total = ticks.iter().take(8).copied().sum();
                (
                    total,
                    ticks[3].saturating_add(ticks.get(4).copied().unwrap_or(0)),
                )
            })
        })
        .collect::<Vec<_>>();
    (!values.is_empty()).then_some(values)
}

#[cfg(not(target_os = "linux"))]
fn cpu_core_counters() -> Option<Vec<(u64, u64)>> {
    None
}

#[cfg(not(target_os = "linux"))]
fn cpu_counters() -> Option<(u64, u64)> {
    None
}

#[cfg(target_os = "linux")]
fn process_ticks() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let close = stat.rfind(')')?;
    let fields: Vec<&str> = stat[close + 1..].split_whitespace().collect();
    let user: u64 = fields.get(11)?.parse().ok()?;
    let system: u64 = fields.get(12)?.parse().ok()?;
    Some(user.saturating_add(system))
}

#[cfg(not(target_os = "linux"))]
fn process_ticks() -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
fn process_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb = status.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        (name == "VmRSS").then(|| value.split_whitespace().next()?.parse::<u64>().ok())?
    })?;
    Some(kb.saturating_mul(1024))
}

#[cfg(not(target_os = "linux"))]
fn process_rss_bytes() -> Option<u64> {
    None
}

struct GpuProbe {
    nvml: Option<nvml_wrapper::Nvml>,
    unavailable_reported: bool,
}

impl GpuProbe {
    fn new() -> Self {
        Self {
            nvml: nvml_wrapper::Nvml::init().ok(),
            unavailable_reported: false,
        }
    }

    fn sample(&mut self, producer: &mut ProducerHandle, step: u64) {
        let Some(nvml) = &self.nvml else {
            self.emit_unavailable(producer, step);
            return;
        };
        let Ok(count) = nvml.device_count() else {
            self.nvml = None;
            self.emit_unavailable(producer, step);
            return;
        };
        if count == 0 {
            if !self.unavailable_reported {
                producer.emit_unavailable(GPU_UTIL, 0, step, REASON_NO_VISIBLE_GPU, true);
                producer.emit_unavailable(GPU_MEMORY_BYTES, 0, step, REASON_NO_VISIBLE_GPU, true);
                self.unavailable_reported = true;
            }
            return;
        }
        for index in 0..count.min(u16::MAX as u32) {
            let group = index as u16;
            let Ok(device) = nvml.device_by_index(index) else {
                producer.emit_unavailable(GPU_UTIL, group, step, REASON_NVML_UNAVAILABLE, true);
                producer.emit_unavailable(
                    GPU_MEMORY_BYTES,
                    group,
                    step,
                    REASON_NVML_UNAVAILABLE,
                    true,
                );
                continue;
            };
            match device.utilization_rates() {
                Ok(util) => {
                    producer.emit_scalar(GPU_UTIL, group, step, util.gpu as f64, true);
                }
                Err(_) => {
                    producer.emit_unavailable(GPU_UTIL, group, step, REASON_NVML_UNAVAILABLE, true);
                }
            }
            match device.memory_info() {
                Ok(mem) => {
                    producer.emit_scalar(GPU_MEMORY_BYTES, group, step, mem.used as f64, true);
                }
                Err(_) => {
                    producer.emit_unavailable(
                        GPU_MEMORY_BYTES,
                        group,
                        step,
                        REASON_NVML_UNAVAILABLE,
                        true,
                    );
                }
            }
        }
    }

    fn emit_unavailable(&mut self, producer: &mut ProducerHandle, step: u64) {
        if self.unavailable_reported {
            return;
        }
        producer.emit_unavailable(GPU_UTIL, 0, step, REASON_NVML_UNAVAILABLE, true);
        producer.emit_unavailable(GPU_MEMORY_BYTES, 0, step, REASON_NVML_UNAVAILABLE, true);
        self.unavailable_reported = true;
    }
}

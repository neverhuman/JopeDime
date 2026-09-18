//! Cross-process FIFO queue for shared-GPU sections.
//!
//! The queue is intentionally std-only: ticket files make acquisition visible
//! across CLI, web, and CI processes; a heartbeat keeps active tickets fresh;
//! expired tickets are removed before each FIFO check.

#[cfg(any(
    feature = "gpu",
    feature = "gpu-dynamic-loading",
    feature = "gpu-dynamic-linking",
    test
))]
use super::control::GpuInterruption;
use super::control::{GpuCheckpoint, NoopCheckpoint};
use anyhow::{anyhow, Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

mod storage;
pub use storage::cleanup_expired_tickets;
use storage::{create_ticket, env_duration_millis, env_duration_secs, list_tickets, touch_ticket};

const DEFAULT_EXPIRED_AFTER_SECS: u64 = 120;
const DEFAULT_POLL_MS: u64 = 50;
const DEFAULT_HEARTBEAT_MS: u64 = 1_000;
const DEFAULT_MAX_WAIT_SECS: u64 = 30 * 60;

static TICKET_SEQ: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct GpuQueueConfig {
    pub dir: PathBuf,
    pub expired_after: Duration,
    pub poll_interval: Duration,
    pub heartbeat_interval: Duration,
    pub max_wait: Duration,
}

impl Default for GpuQueueConfig {
    fn default() -> Self {
        let dir = std::env::var_os("JAIN_GPU_QUEUE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("jain-gpu-queue"));
        Self {
            dir,
            expired_after: env_duration_secs(
                "JAIN_GPU_QUEUE_EXPIRED_SECS",
                DEFAULT_EXPIRED_AFTER_SECS,
            ),
            poll_interval: env_duration_millis("JAIN_GPU_QUEUE_POLL_MS", DEFAULT_POLL_MS),
            heartbeat_interval: env_duration_millis(
                "JAIN_GPU_QUEUE_HEARTBEAT_MS",
                DEFAULT_HEARTBEAT_MS,
            ),
            max_wait: env_duration_secs("JAIN_GPU_QUEUE_MAX_WAIT_SECS", DEFAULT_MAX_WAIT_SECS),
        }
    }
}

#[derive(Debug)]
pub struct GpuLease {
    ticket: PathBuf,
    waited: Duration,
    stop: Option<Sender<()>>,
    heartbeat: Option<JoinHandle<()>>,
}

impl GpuLease {
    pub fn ticket_path(&self) -> &Path {
        &self.ticket
    }

    pub fn waited(&self) -> Duration {
        self.waited
    }
}

impl Drop for GpuLease {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(heartbeat) = self.heartbeat.take() {
            let _ = heartbeat.join();
        }
        let _ = fs::remove_file(&self.ticket);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuCpuHandoffBadge {
    pub backend: String,
    pub cpu_handoff_reason: String,
}

pub fn gpu_cpu_handoff_badge(reason: impl Into<String>) -> GpuCpuHandoffBadge {
    GpuCpuHandoffBadge {
        backend: "cpu".into(),
        cpu_handoff_reason: reason.into(),
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GpuLeaseRequest {
    pub cpu_estimate: Option<Duration>,
    pub gpu_compute_estimate: Duration,
}

pub fn gpu_30x_bypass_reason(
    cpu_estimate: Duration,
    queue_wait_estimate: Duration,
    gpu_compute_estimate: Duration,
) -> Option<String> {
    if cpu_estimate.is_zero() {
        return None;
    }
    let cpu_secs = cpu_estimate.as_secs_f64();
    let gpu_secs = queue_wait_estimate.as_secs_f64() + gpu_compute_estimate.as_secs_f64();
    if gpu_secs > 30.0 * cpu_secs {
        Some(format!(
            "CPU path selected: estimated GPU wall time {:.3}s exceeds 30x CPU estimate {:.3}s",
            gpu_secs, cpu_secs
        ))
    } else {
        None
    }
}

pub fn acquire_default_gpu_lease() -> Result<GpuLease> {
    acquire_gpu_lease(&GpuQueueConfig::default())
}

pub fn acquire_default_gpu_lease_with_request(request: GpuLeaseRequest) -> Result<GpuLease> {
    acquire_gpu_lease_with_request(&GpuQueueConfig::default(), request)
}

pub fn acquire_default_gpu_lease_with_request_and_checkpoint(
    request: GpuLeaseRequest,
    checkpoint: &dyn GpuCheckpoint,
) -> Result<GpuLease> {
    acquire_gpu_lease_with_request_and_checkpoint(&GpuQueueConfig::default(), request, checkpoint)
}

pub fn acquire_gpu_lease(config: &GpuQueueConfig) -> Result<GpuLease> {
    acquire_gpu_lease_with_request(config, GpuLeaseRequest::default())
}

pub fn acquire_gpu_lease_with_request(
    config: &GpuQueueConfig,
    request: GpuLeaseRequest,
) -> Result<GpuLease> {
    acquire_gpu_lease_with_request_and_checkpoint(config, request, &NoopCheckpoint)
}

pub fn acquire_gpu_lease_with_request_and_checkpoint(
    config: &GpuQueueConfig,
    request: GpuLeaseRequest,
    checkpoint: &dyn GpuCheckpoint,
) -> Result<GpuLease> {
    checkpoint.checkpoint()?;
    fs::create_dir_all(&config.dir)
        .with_context(|| format!("create gpu queue dir {}", config.dir.display()))?;
    cleanup_expired_tickets(&config.dir, config.expired_after)?;

    let ticket = create_ticket(&config.dir)?;
    let (stop_tx, stop_rx) = mpsc::channel();
    let heartbeat_path = ticket.clone();
    let heartbeat_interval = config.heartbeat_interval;
    let heartbeat = thread::spawn(move || loop {
        match stop_rx.recv_timeout(heartbeat_interval) {
            Ok(_) => break,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let _ = touch_ticket(&heartbeat_path);
            }
        }
    });

    let started = Instant::now();
    let mut lease = GpuLease {
        ticket,
        waited: Duration::ZERO,
        stop: Some(stop_tx),
        heartbeat: Some(heartbeat),
    };
    loop {
        checkpoint.checkpoint()?;
        cleanup_expired_tickets(&config.dir, config.expired_after)?;
        let tickets = list_tickets(&config.dir)?;
        if tickets.first() == Some(&lease.ticket) {
            let waited = started.elapsed();
            if let Some(cpu_estimate) = request.cpu_estimate {
                if let Some(reason) =
                    gpu_30x_bypass_reason(cpu_estimate, waited, request.gpu_compute_estimate)
                {
                    return Err(anyhow!("{reason}; queue wait was {waited:?}"));
                }
            }
            lease.waited = waited;
            return Ok(lease);
        }
        if started.elapsed() > config.max_wait {
            return Err(anyhow!(
                "GPU queue wait timed out after {:?} (max {:?}); CPU path selected because FIFO contention did not clear",
                started.elapsed(),
                config.max_wait
            ));
        }
        thread::sleep(config.poll_interval);
    }
}

pub fn gpu_cpu_handoff_reason(context: &str, detail: impl std::fmt::Display) -> String {
    let detail = detail.to_string();
    let lower = detail.to_ascii_lowercase();
    let class = if lower.contains("30x") {
        "GPU bypassed by the 30x CPU wall-time guard"
    } else if lower.contains("queue wait") || lower.contains("fifo contention") {
        "GPU queue contention timed out"
    } else if lower.contains("out of memory")
        || lower.contains("outofmemory")
        || lower.contains("cuda_error_out_of_memory")
        || lower.contains("oom")
    {
        "CUDA device ran out of memory"
    } else if lower.contains("libcuda not present")
        || lower.contains("cuda device 0 is not available")
        || lower.contains("cuda gpu is not available")
        || lower.contains("without the `gpu` feature")
        || lower.contains("without the 'gpu' feature")
    {
        "CUDA GPU is not available on this host"
    } else if lower.contains("cross-check")
        || lower.contains("cuda init")
        || lower.contains("creating cuda context")
        || lower.contains("nvrtc")
        || lower.contains("driver error")
    {
        "CUDA initialization or startup cross-check failed"
    } else {
        "GPU execution failed"
    };
    format!("{class}; CPU handoff selected for {context}: {detail}")
}

pub fn gpu_cpu_handoff_error(context: &str, err: anyhow::Error) -> anyhow::Error {
    anyhow!(gpu_cpu_handoff_reason(context, format!("{err:#}")))
}

/// Cancellation and deadline errors are operation control, not GPU fallback.
/// Preserve their concrete type so Core can distinguish a stopped operation
/// from a CUDA failure while retaining the existing CPU-handoff taxonomy for
/// every actual queue/device error.
#[cfg(any(
    feature = "gpu",
    feature = "gpu-dynamic-loading",
    feature = "gpu-dynamic-linking",
    test
))]
pub(crate) fn gpu_cpu_handoff_unless_interrupted(
    context: &str,
    err: anyhow::Error,
) -> anyhow::Error {
    if err.downcast_ref::<GpuInterruption>().is_some() {
        err
    } else {
        gpu_cpu_handoff_error(context, err)
    }
}

#[cfg(test)]
#[path = "queue_tests.rs"]
mod tests;

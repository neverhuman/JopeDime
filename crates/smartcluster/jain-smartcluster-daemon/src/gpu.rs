use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::time::Duration;

use nvml_wrapper::bitmasks::event::EventTypes;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::{Device, Nvml};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

use crate::config::MpsConfig;

const VRAM_HEADROOM_PERCENT: u64 = 10;
const MEBIBYTE: u64 = 1024 * 1024;

/// Administrator-qualified driver bookkeeping, never a general free-memory allowance.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualifiedIdleBaseline {
    pub driver_version: String,
    pub idle_used_vram_bytes: u64,
    pub qualification_receipt_sha256: String,
}

impl QualifiedIdleBaseline {
    pub fn validate(&self, uuid: &str) -> Result<(), String> {
        if !uuid.starts_with("GPU-")
            || uuid.len() != 40
            || !uuid[4..].bytes().enumerate().all(|(index, byte)| {
                if [8, 13, 18, 23].contains(&index) {
                    byte == b'-'
                } else {
                    byte.is_ascii_hexdigit()
                }
            })
            || self.driver_version.is_empty()
            || self.driver_version.len() > 64
            || !self
                .driver_version
                .split('.')
                .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
            || self.idle_used_vram_bytes == 0
            || self.idle_used_vram_bytes > 512 * MEBIBYTE
            || self.qualification_receipt_sha256.len() != 64
            || !self
                .qualification_receipt_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("idle baseline requires a physical GPU UUID, exact driver, 1..=512MiB measured NVML v1 driver usage and a lowercase qualification receipt SHA256".to_owned());
        }
        Ok(())
    }
}

/// Both process inventories must succeed before any idle-baseline decision.
fn complete_process_inventory<T>(
    compute: Result<Vec<T>, NvmlError>,
    graphics: Result<Vec<T>, NvmlError>,
) -> Result<Vec<T>, NvmlError> {
    let mut processes = compute?;
    processes.extend(graphics?);
    Ok(processes)
}

pub(crate) fn qualified_idle_usage(
    device: &GpuDevice,
    baselines: &BTreeMap<String, QualifiedIdleBaseline>,
) -> bool {
    baselines.get(&device.uuid).is_some_and(|baseline| {
        baseline.validate(&device.uuid).is_ok()
            && baseline.driver_version == device.driver_version
            && device.processes.is_empty()
            && device.utilization_per_mille == Some(0)
            && baseline.idle_used_vram_bytes <= device.total_vram_bytes / 32
            && device.used_vram_bytes <= baseline.idle_used_vram_bytes
    })
}

#[cfg(test)]
mod idle_baseline_tests {
    use super::*;
    const UUID: &str = "GPU-44c63d19-492a-ab46-cc6a-56e12a2ef51b";
    fn device() -> GpuDevice {
        GpuDevice {
            id: "0".into(),
            physical_index: 0,
            uuid: UUID.into(),
            driver_version: "570.211.01".into(),
            parent_uuid: None,
            name: "fixture".into(),
            total_vram_bytes: 24 * 1024 * MEBIBYTE,
            used_vram_bytes: MEBIBYTE,
            allocatable_vram_bytes: 20 * 1024 * MEBIBYTE,
            utilization_per_mille: Some(0),
            temperature_millidegrees_celsius: None,
            power_milliwatts: None,
            power_limit_milliwatts: None,
            mig_instance: None,
            processes: vec![],
            healthy: true,
        }
    }
    fn baselines() -> BTreeMap<String, QualifiedIdleBaseline> {
        BTreeMap::from([(
            UUID.into(),
            QualifiedIdleBaseline {
                driver_version: "570.211.01".into(),
                idle_used_vram_bytes: MEBIBYTE,
                qualification_receipt_sha256: "a".repeat(64),
            },
        )])
    }
    #[test]
    fn idle_driver_usage_requires_exact_qualified_identity_and_bound() {
        let base = baselines();
        let mut d = device();
        assert!(!qualified_idle_usage(&d, &BTreeMap::new()));
        assert!(qualified_idle_usage(&d, &base));
        d.used_vram_bytes += 1;
        assert!(!qualified_idle_usage(&d, &base));
        d = device();
        d.driver_version = "580.1".into();
        assert!(!qualified_idle_usage(&d, &base));
        d = device();
        d.uuid = "GPU-00000000-0000-0000-0000-000000000000".into();
        assert!(!qualified_idle_usage(&d, &base));
        d = device();
        d.utilization_per_mille = None;
        assert!(!qualified_idle_usage(&d, &base));
        d.utilization_per_mille = Some(1);
        assert!(!qualified_idle_usage(&d, &base));
        d = device();
        d.processes.push(GpuProcess {
            pid: 123,
            used_vram_bytes: None,
            gpu_instance_id: None,
            compute_instance_id: None,
        });
        assert!(!qualified_idle_usage(&d, &base));
    }
    #[test]
    fn missing_compute_or_graphics_inventory_cannot_become_idle() {
        assert!(complete_process_inventory::<u32>(Err(NvmlError::Unknown), Ok(vec![])).is_err());
        assert!(complete_process_inventory::<u32>(Ok(vec![]), Err(NvmlError::Unknown)).is_err());
        assert_eq!(
            complete_process_inventory(Ok(vec![1]), Ok(vec![2])).unwrap(),
            vec![1, 2]
        );
    }
    #[test]
    fn idle_baseline_rejects_unbounded_or_unqualified_configuration() {
        let mut b = baselines().remove(UUID).unwrap();
        b.idle_used_vram_bytes = 513 * MEBIBYTE;
        assert!(b.validate(UUID).is_err());
        b.idle_used_vram_bytes = MEBIBYTE;
        b.qualification_receipt_sha256.clear();
        assert!(b.validate(UUID).is_err());
    }

    #[test]
    fn v1_driver_reservation_requires_a_device_bounded_measured_baseline() {
        let mut d = device();
        let mut base = baselines();
        d.used_vram_bytes = 466_354_176;
        base.get_mut(UUID).unwrap().idle_used_vram_bytes = d.used_vram_bytes;
        assert!(qualified_idle_usage(&d, &base));
        d.used_vram_bytes += 1;
        assert!(!qualified_idle_usage(&d, &base));
        d.used_vram_bytes -= 1;
        d.total_vram_bytes = 8 * 1024 * MEBIBYTE;
        assert!(!qualified_idle_usage(&d, &base));
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuProcess {
    pub pid: u32,
    pub used_vram_bytes: Option<u64>,
    pub gpu_instance_id: Option<u32>,
    pub compute_instance_id: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuDevice {
    pub id: String,
    pub physical_index: u32,
    pub uuid: String,
    #[serde(default)]
    pub driver_version: String,
    pub parent_uuid: Option<String>,
    pub name: String,
    pub total_vram_bytes: u64,
    pub used_vram_bytes: u64,
    pub allocatable_vram_bytes: u64,
    pub utilization_per_mille: Option<u16>,
    pub temperature_millidegrees_celsius: Option<u32>,
    pub power_milliwatts: Option<u64>,
    pub power_limit_milliwatts: Option<u64>,
    pub mig_instance: Option<String>,
    pub processes: Vec<GpuProcess>,
    pub healthy: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuInventory {
    pub nvml_available: bool,
    pub mig_inventory: bool,
    pub devices: Vec<GpuDevice>,
    pub unavailable_reason: Option<String>,
    pub mps: MpsCapability,
}

impl GpuInventory {
    pub fn discover(cuda_visible_devices: Option<&str>) -> Self {
        match discover_nvml(cuda_visible_devices) {
            Ok((devices, mig_inventory)) => Self {
                nvml_available: true,
                mig_inventory,
                devices,
                unavailable_reason: None,
                mps: MpsCapability::detect(),
            },
            Err(error) => Self {
                nvml_available: false,
                mig_inventory: false,
                devices: Vec::new(),
                unavailable_reason: Some(error.to_string()),
                mps: MpsCapability::detect(),
            },
        }
    }
}

fn discover_nvml(visible: Option<&str>) -> Result<(Vec<GpuDevice>, bool), GpuError> {
    let nvml = Nvml::init()?;
    let driver_version = nvml.sys_driver_version()?;
    let selectors = visible.map(parse_visible_devices);
    if selectors.as_ref().is_some_and(BTreeSet::is_empty) {
        return Ok((Vec::new(), false));
    }
    let mut devices = Vec::new();
    let mut mig_inventory = false;
    for index in 0..nvml.device_count()? {
        let device = nvml.device_by_index(index)?;
        let uuid = device.uuid()?;
        let parent_visible = is_visible(selectors.as_ref(), index, &uuid);
        if device.mig_mode().is_ok_and(|mode| mode.current != 0) {
            match device.mig_device_count() {
                Ok(count) => {
                    mig_inventory = true;
                    for mig_index in 0..count {
                        if let Ok(mig) = device.mig_device_by_index(mig_index) {
                            if let Ok(sample) = sample_device(
                                &mig,
                                index,
                                Some(mig_index),
                                Some(uuid.clone()),
                                &driver_version,
                            ) {
                                if parent_visible
                                    || selectors
                                        .as_ref()
                                        .is_none_or(|selectors| selectors.contains(&sample.uuid))
                                {
                                    devices.push(sample);
                                }
                            }
                        }
                    }
                }
                Err(_) => {
                    // MIG mode without readable instance handles is not advertised as isolatable.
                    mig_inventory = false;
                }
            }
        } else if parent_visible {
            devices.push(sample_device(&device, index, None, None, &driver_version)?);
        }
    }
    Ok((devices, mig_inventory))
}

fn sample_device(
    device: &Device<'_>,
    physical_index: u32,
    mig_index: Option<u32>,
    parent_uuid: Option<String>,
    driver_version: &str,
) -> Result<GpuDevice, GpuError> {
    let memory = device.memory_info()?;
    let uuid = device.uuid()?;
    let processes = complete_process_inventory(
        device.running_compute_processes(),
        device.running_graphics_processes(),
    )?
    .into_iter()
    .map(|process| GpuProcess {
        pid: process.pid,
        used_vram_bytes: match process.used_gpu_memory {
            UsedGpuMemory::Used(bytes) => Some(bytes),
            UsedGpuMemory::Unavailable => None,
        },
        gpu_instance_id: process.gpu_instance_id,
        compute_instance_id: process.compute_instance_id,
    })
    .collect();
    let utilization_per_mille = device
        .utilization_rates()
        .ok()
        .and_then(|utilization| u16::try_from(utilization.gpu.saturating_mul(10)).ok());
    let temperature = device
        .temperature(TemperatureSensor::Gpu)
        .ok()
        .map(|celsius| celsius.saturating_mul(1_000));
    let allocatable_vram_bytes = memory.total.saturating_mul(100 - VRAM_HEADROOM_PERCENT) / 100;
    Ok(GpuDevice {
        id: mig_index.map_or_else(|| physical_index.to_string(), |_| uuid.clone()),
        physical_index,
        uuid,
        driver_version: driver_version.to_owned(),
        parent_uuid,
        name: device.name()?,
        total_vram_bytes: memory.total,
        used_vram_bytes: memory.used,
        allocatable_vram_bytes,
        utilization_per_mille,
        temperature_millidegrees_celsius: temperature,
        power_milliwatts: device.power_usage().ok().map(u64::from),
        power_limit_milliwatts: device.power_management_limit().ok().map(u64::from),
        mig_instance: mig_index.map(|index| index.to_string()),
        processes,
        healthy: true,
    })
}

fn parse_visible_devices(value: &str) -> BTreeSet<String> {
    if value.split(',').any(|token| token.trim() == "-1") {
        return BTreeSet::new();
    }
    value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn is_visible(selectors: Option<&BTreeSet<String>>, index: u32, uuid: &str) -> bool {
    selectors
        .is_none_or(|selectors| selectors.contains(&index.to_string()) || selectors.contains(uuid))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MpsCapability {
    pub control_executable: Option<PathBuf>,
    pub pipe_directory: Option<PathBuf>,
    pub compute_limit: bool,
    pub vram_limit: bool,
    pub safe_client_termination: bool,
    pub shareable: bool,
    pub unavailable_reason: Option<String>,
}

impl MpsCapability {
    pub fn detect() -> Self {
        Self {
            control_executable: None,
            pipe_directory: None,
            compute_limit: false,
            vram_limit: false,
            safe_client_termination: false,
            shareable: false,
            unavailable_reason: Some(
                "MPS sharing requires an explicitly configured and successfully probed controller"
                    .to_owned(),
            ),
        }
    }

    pub fn unavailable(config: &MpsConfig, reason: impl Into<String>) -> Self {
        Self {
            control_executable: config.control_executable.clone(),
            pipe_directory: config.pipe_directory.clone(),
            compute_limit: false,
            vram_limit: false,
            safe_client_termination: false,
            shareable: false,
            unavailable_reason: Some(reason.into()),
        }
    }

    pub fn disable(&mut self, reason: impl Into<String>) {
        self.compute_limit = false;
        self.vram_limit = false;
        self.safe_client_termination = false;
        self.shareable = false;
        self.unavailable_reason = Some(reason.into());
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MpsClientLimits {
    pub compute_share_per_mille: u16,
    pub vram_bytes_by_device: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MpsClient {
    pub pid: u32,
    pub client_id: String,
    pub server_pid: u32,
    pub device: String,
    pub namespace_pid: u64,
    pub command: String,
}

/// Matches an MPS-reported client to host cgroup PIDs, including PID namespaces.
pub fn mps_client_matches_pids(
    client: &MpsClient,
    host_pids: &BTreeSet<u32>,
) -> Result<bool, GpuError> {
    for host_pid in host_pids {
        if *host_pid == client.pid {
            return Ok(true);
        }
        let namespace = match pid_namespace_inode(*host_pid) {
            Ok(namespace) => namespace,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(source) => {
                return Err(GpuError::MpsPidIdentity {
                    pid: *host_pid,
                    source,
                });
            }
        };
        if namespace == client.namespace_pid {
            let namespace_pids =
                namespace_pids(*host_pid).map_err(|source| GpuError::MpsPidIdentity {
                    pid: *host_pid,
                    source,
                })?;
            if namespace_pids.contains(&client.pid) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn pid_namespace_inode(pid: u32) -> io::Result<u64> {
    let target = std::fs::read_link(format!("/proc/{pid}/ns/pid"))?;
    let target = target.to_string_lossy();
    target
        .strip_prefix("pid:[")
        .and_then(|value| value.strip_suffix(']'))
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid PID namespace link"))
}

fn namespace_pids(pid: u32) -> io::Result<BTreeSet<u32>> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status"))?;
    let values = status
        .lines()
        .find_map(|line| line.strip_prefix("NSpid:"))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing NSpid status"))?;
    values
        .split_whitespace()
        .map(|value| {
            value.parse().map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "invalid NSpid status value")
            })
        })
        .collect()
}

/// A controller is only constructed after a bounded `ps` exchange proves that
/// the configured daemon exposes the Volta MPS client/server identity protocol.
#[derive(Clone, Debug)]
pub struct MpsController {
    control_executable: PathBuf,
    pipe_directory: PathBuf,
    command_timeout: Duration,
    max_control_output_bytes: usize,
}

impl MpsController {
    pub async fn probe(config: &MpsConfig) -> Result<Self, GpuError> {
        let (control_executable, pipe_directory) =
            match (&config.control_executable, &config.pipe_directory) {
                (Some(executable), Some(directory)) => (executable.clone(), directory.clone()),
                _ => return Err(GpuError::MpsUnavailable),
            };
        if !control_executable.is_absolute() || !pipe_directory.is_absolute() {
            return Err(GpuError::MpsPathNotAbsolute);
        }
        if !control_executable.is_file() {
            return Err(GpuError::MpsControlExecutable(control_executable));
        }
        if !pipe_directory.is_dir() {
            return Err(GpuError::MpsPipeDirectory(pipe_directory));
        }
        if config.command_timeout_ms == 0
            || config.max_control_output_bytes == 0
            || config.max_control_output_bytes > 16 * 1024 * 1024
        {
            return Err(GpuError::MpsControlBounds);
        }
        let controller = Self {
            control_executable,
            pipe_directory,
            command_timeout: config.command_timeout(),
            max_control_output_bytes: config.max_control_output_bytes,
        };

        // `ps` is a Volta MPS command and its SERVER column is the identity
        // required by safe `terminate_client`. The two per-client limits are
        // only advertised together with this successfully probed protocol.
        controller.clients().await?;
        Ok(controller)
    }

    pub fn capability(&self) -> MpsCapability {
        MpsCapability {
            control_executable: Some(self.control_executable.clone()),
            pipe_directory: Some(self.pipe_directory.clone()),
            compute_limit: true,
            vram_limit: true,
            safe_client_termination: true,
            shareable: true,
            unavailable_reason: None,
        }
    }

    pub fn validate_worker_access(&self, uid: u32, gid: u32) -> Result<(), GpuError> {
        let executable = std::fs::metadata(&self.control_executable).map_err(|source| {
            GpuError::MpsWorkerAccess {
                path: self.control_executable.clone(),
                reason: source.to_string(),
            }
        })?;
        if !mode_allows(&executable, uid, gid, 0o5) {
            return Err(GpuError::MpsWorkerAccess {
                path: self.control_executable.clone(),
                reason: format!("uid {uid} gid {gid} lacks read/execute access"),
            });
        }
        let directory = std::fs::metadata(&self.pipe_directory).map_err(|source| {
            GpuError::MpsWorkerAccess {
                path: self.pipe_directory.clone(),
                reason: source.to_string(),
            }
        })?;
        if !mode_allows(&directory, uid, gid, 0o7) {
            return Err(GpuError::MpsWorkerAccess {
                path: self.pipe_directory.clone(),
                reason: format!("uid {uid} gid {gid} lacks read/write/search access"),
            });
        }
        Ok(())
    }

    pub const fn command_timeout(&self) -> Duration {
        self.command_timeout
    }

    pub fn client_environment(
        &self,
        limits: &MpsClientLimits,
    ) -> Result<BTreeMap<String, String>, GpuError> {
        let compute_percent = compute_percent(limits.compute_share_per_mille)?;
        let vram_limits = format_vram_limits(&limits.vram_bytes_by_device)?;
        let pipe_directory = self
            .pipe_directory
            .to_str()
            .ok_or(GpuError::MpsPathNotUtf8)?;
        Ok([
            (
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE".to_owned(),
                compute_percent.to_string(),
            ),
            ("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT".to_owned(), vram_limits),
            (
                "CUDA_MPS_PIPE_DIRECTORY".to_owned(),
                pipe_directory.to_owned(),
            ),
        ]
        .into_iter()
        .collect())
    }

    pub async fn clients(&self) -> Result<Vec<MpsClient>, GpuError> {
        let output = self.run_control("ps\n").await?;
        parse_ps(&output)
    }

    pub async fn terminate_client(&self, server_pid: u32, client_pid: u32) -> Result<(), GpuError> {
        if server_pid == 0 || client_pid == 0 {
            return Err(GpuError::MpsClientIdentity);
        }
        let clients = self.clients().await?;
        if !clients
            .iter()
            .any(|client| client.server_pid == server_pid && client.pid == client_pid)
        {
            return Err(GpuError::MpsClientNotFound {
                server_pid,
                client_pid,
            });
        }
        let output = self
            .run_control(&format!("terminate_client {server_pid} {client_pid}\n"))
            .await?;
        parse_termination_response(&output)
    }

    async fn run_control(&self, input: &str) -> Result<String, GpuError> {
        let mut child = Command::new(&self.control_executable)
            .env_clear()
            .env("CUDA_MPS_PIPE_DIRECTORY", &self.pipe_directory)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|source| GpuError::MpsControlSpawn {
                executable: self.control_executable.clone(),
                source,
            })?;
        let mut stdin = child.stdin.take().ok_or(GpuError::MpsControlPipe)?;
        let stdout = child.stdout.take().ok_or(GpuError::MpsControlPipe)?;
        let stderr = child.stderr.take().ok_or(GpuError::MpsControlPipe)?;
        let bound = self.max_control_output_bytes;
        let operation = async {
            stdin.write_all(input.as_bytes()).await?;
            stdin.shutdown().await?;
            let (status, stdout, stderr) = tokio::join!(
                child.wait(),
                read_bounded(stdout, bound),
                read_bounded(stderr, bound)
            );
            Ok::<_, io::Error>((status?, stdout?, stderr?))
        };
        let (status, stdout, stderr) =
            match tokio::time::timeout(self.command_timeout, operation).await {
                Ok(result) => result.map_err(GpuError::MpsControlIo)?,
                Err(_) => {
                    let _ = child.kill().await;
                    let _ = child.wait().await;
                    return Err(GpuError::MpsControlTimeout(self.command_timeout));
                }
            };
        if stdout.len() > bound || stderr.len() > bound {
            return Err(GpuError::MpsControlOutputTooLarge(bound));
        }
        if !status.success() {
            return Err(GpuError::MpsControlExit(status.code()));
        }
        if !stderr.iter().all(u8::is_ascii_whitespace) {
            return Err(GpuError::MpsControlStderr);
        }
        String::from_utf8(stdout).map_err(|_| GpuError::MpsControlUtf8)
    }
}

fn mode_allows(metadata: &std::fs::Metadata, uid: u32, gid: u32, required: u32) -> bool {
    if uid == 0 {
        return true;
    }
    let granted = if metadata.uid() == uid {
        (metadata.mode() >> 6) & 0o7
    } else if metadata.gid() == gid {
        (metadata.mode() >> 3) & 0o7
    } else {
        metadata.mode() & 0o7
    };
    granted & required == required
}

async fn read_bounded(reader: impl AsyncRead + Unpin, max_bytes: usize) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    reader
        .take(
            u64::try_from(max_bytes)
                .unwrap_or(u64::MAX)
                .saturating_add(1),
        )
        .read_to_end(&mut output)
        .await?;
    Ok(output)
}

fn compute_percent(compute_share_per_mille: u16) -> Result<u16, GpuError> {
    if !(10..=1_000).contains(&compute_share_per_mille) {
        return Err(GpuError::ComputeShare(compute_share_per_mille));
    }
    // MPS accepts whole percentages. Rounding down keeps the actual limit at
    // or below the scheduler's per-mille budget.
    Ok(compute_share_per_mille / 10)
}

fn format_vram_limits(limits: &BTreeMap<String, u64>) -> Result<String, GpuError> {
    if limits.is_empty() {
        return Err(GpuError::MpsVramLimitsEmpty);
    }
    limits
        .iter()
        .map(|(device, bytes)| {
            if !valid_device_selector(device) {
                return Err(GpuError::MpsDeviceSelector(device.clone()));
            }
            let mebibytes = bytes / MEBIBYTE;
            if mebibytes == 0 {
                return Err(GpuError::MpsVramLimit(*bytes));
            }
            Ok(format!("{device}={mebibytes}M"))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|parts| parts.join(","))
}

fn valid_device_selector(device: &str) -> bool {
    let ordinal = !device.is_empty() && device.bytes().all(|byte| byte.is_ascii_digit());
    let uuid = device.strip_prefix("GPU-").is_some_and(|suffix| {
        !suffix.is_empty()
            && suffix.len() <= 124
            && suffix
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() || byte == b'-')
    });
    ordinal || uuid
}

fn parse_ps(output: &str) -> Result<Vec<MpsClient>, GpuError> {
    let mut lines = output.lines().filter(|line| !line.trim().is_empty());
    let header: Vec<_> = lines
        .next()
        .ok_or(GpuError::MpsPsHeader)?
        .split_whitespace()
        .collect();
    if header.len() < 6 || header[..6] != ["PID", "ID", "SERVER", "DEVICE", "NAMESPACE", "COMMAND"]
    {
        return Err(GpuError::MpsPsHeader);
    }
    lines
        .map(|line| {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.len() < 6 {
                return Err(GpuError::MpsPsRecord);
            }
            Ok(MpsClient {
                pid: fields[0].parse().map_err(|_| GpuError::MpsPsRecord)?,
                client_id: fields[1].to_owned(),
                server_pid: fields[2].parse().map_err(|_| GpuError::MpsPsRecord)?,
                device: fields[3].to_owned(),
                namespace_pid: fields[4].parse().map_err(|_| GpuError::MpsPsRecord)?,
                command: fields[5..].join(" "),
            })
        })
        .collect()
}

fn parse_termination_response(output: &str) -> Result<(), GpuError> {
    if output.trim() == "0" {
        Ok(())
    } else {
        Err(GpuError::UnsafeMpsTermination)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuEventKind {
    CriticalXid(Option<String>),
    DoubleBitEcc,
    DeviceLost,
    MigConfigurationChanged,
    MonitorFailed(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuEvent {
    pub device_uuid: Option<String>,
    pub kind: GpuEventKind,
}

pub struct GpuEventMonitor {
    stopped: Arc<AtomicBool>,
}

impl GpuEventMonitor {
    pub fn spawn(sender: tokio::sync::mpsc::Sender<GpuEvent>) -> Result<Self, GpuError> {
        let stopped = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&stopped);
        let (ready_sender, ready_receiver) = std_mpsc::sync_channel(1);
        tokio::task::spawn_blocking(move || monitor_events(sender, &stop, ready_sender));
        match ready_receiver.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => {}
            Ok(Err(message)) => return Err(GpuError::EventMonitorSetup(message)),
            Err(error) => return Err(GpuError::EventMonitorSetup(error.to_string())),
        }
        Ok(Self { stopped })
    }
}

impl Drop for GpuEventMonitor {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Release);
    }
}

fn monitor_events(
    sender: tokio::sync::mpsc::Sender<GpuEvent>,
    stopped: &AtomicBool,
    ready: std_mpsc::SyncSender<Result<(), String>>,
) {
    let nvml = match Nvml::init() {
        Ok(nvml) => nvml,
        Err(error) => {
            let _ = ready.send(Err(error.to_string()));
            return;
        }
    };
    let mut set = match nvml.create_event_set() {
        Ok(set) => set,
        Err(error) => {
            let _ = ready.send(Err(error.to_string()));
            return;
        }
    };
    let count = match nvml.device_count() {
        Ok(count) => count,
        Err(error) => {
            let _ = ready.send(Err(error.to_string()));
            return;
        }
    };
    let mut registered_any = false;
    for index in 0..count {
        let Ok(device) = nvml.device_by_index(index) else {
            let _ = ready.send(Err(format!("cannot open NVML device {index}")));
            return;
        };
        let Ok(supported) = device.supported_event_types() else {
            let _ = ready.send(Err(format!("cannot query NVML events for device {index}")));
            return;
        };
        let requested = supported
            & (EventTypes::CRITICAL_XID_ERROR
                | EventTypes::DOUBLE_BIT_ECC_ERROR
                | EventTypes::MIG_CONFIG_CHANGE);
        if requested.is_empty() {
            let _ = ready.send(Err(format!(
                "visible NVML device {index} exposes none of the required safety events"
            )));
            return;
        }
        let Ok(registered) = device.register_events(requested, set) else {
            let _ = ready.send(Err(format!(
                "cannot register NVML events for device {index}"
            )));
            return;
        };
        set = registered;
        registered_any = true;
    }
    if !registered_any {
        let _ = ready.send(Err(
            "no visible GPU supports required NVML safety events".to_owned()
        ));
        return;
    }
    if ready.send(Ok(())).is_err() {
        return;
    }
    while !stopped.load(Ordering::Acquire) {
        match set.wait(500) {
            Ok(event) => {
                let uuid = event.device.uuid().ok();
                let kind = if event.event_type.contains(EventTypes::CRITICAL_XID_ERROR) {
                    GpuEventKind::CriticalXid(event.event_data.map(|value| format!("{value:?}")))
                } else if event.event_type.contains(EventTypes::DOUBLE_BIT_ECC_ERROR) {
                    GpuEventKind::DoubleBitEcc
                } else if event.event_type.contains(EventTypes::MIG_CONFIG_CHANGE) {
                    GpuEventKind::MigConfigurationChanged
                } else {
                    continue;
                };
                if sender
                    .blocking_send(GpuEvent {
                        device_uuid: uuid,
                        kind,
                    })
                    .is_err()
                {
                    return;
                }
            }
            Err(NvmlError::Timeout) => {}
            Err(NvmlError::GpuLost) => {
                let _ = sender.blocking_send(GpuEvent {
                    device_uuid: None,
                    kind: GpuEventKind::DeviceLost,
                });
                return;
            }
            Err(error) => {
                let _ = sender.blocking_send(GpuEvent {
                    device_uuid: None,
                    kind: GpuEventKind::MonitorFailed(error.to_string()),
                });
                return;
            }
        }
    }
}

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("NVML operation failed: {0}")]
    Nvml(#[from] NvmlError),
    #[error("MPS is unavailable")]
    MpsUnavailable,
    #[error("MPS control executable and pipe directory must be absolute")]
    MpsPathNotAbsolute,
    #[error("MPS control executable is unavailable: {0}")]
    MpsControlExecutable(PathBuf),
    #[error("MPS pipe directory is unavailable: {0}")]
    MpsPipeDirectory(PathBuf),
    #[error("MPS control timeout or output bound is invalid")]
    MpsControlBounds,
    #[error("MPS pipe directory is not valid UTF-8")]
    MpsPathNotUtf8,
    #[error("failed to start MPS control executable {executable}: {source}")]
    MpsControlSpawn {
        executable: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to communicate with MPS control daemon: {0}")]
    MpsControlIo(#[source] io::Error),
    #[error("MPS control command timed out after {0:?}")]
    MpsControlTimeout(Duration),
    #[error("MPS control output exceeded {0} bytes")]
    MpsControlOutputTooLarge(usize),
    #[error("MPS control command exited unsuccessfully: {0:?}")]
    MpsControlExit(Option<i32>),
    #[error("MPS control command produced unexpected stderr")]
    MpsControlStderr,
    #[error("MPS control command did not expose all configured pipes")]
    MpsControlPipe,
    #[error("MPS control output was not UTF-8")]
    MpsControlUtf8,
    #[error("MPS ps output did not contain the required Volta protocol header")]
    MpsPsHeader,
    #[error("MPS ps output contained an invalid client record")]
    MpsPsRecord,
    #[error("invalid MPS compute share: {0}; at least one whole percent is required")]
    ComputeShare(u16),
    #[error("at least one MPS device VRAM limit is required")]
    MpsVramLimitsEmpty,
    #[error("invalid MPS device selector: {0}")]
    MpsDeviceSelector(String),
    #[error("MPS VRAM limit is below the one-MiB representable minimum: {0}")]
    MpsVramLimit(u64),
    #[error("MPS server and client PIDs must be positive")]
    MpsClientIdentity,
    #[error("MPS client {client_pid} is not connected to server {server_pid}")]
    MpsClientNotFound { server_pid: u32, client_pid: u32 },
    #[error("MPS client {0} remained connected after terminate_client")]
    MpsClientStillConnected(u32),
    #[error("MPS cgroup membership changed during the no-client safety proof")]
    MpsMembershipChanged,
    #[error("cannot read MPS cgroup membership: {0}")]
    MpsCgroupMembership(String),
    #[error("safe MPS terminate_client is unavailable; refusing to signal a shared client")]
    UnsafeMpsTermination,
    #[error("cannot identify MPS client membership for host pid {pid}: {source}")]
    MpsPidIdentity {
        pid: u32,
        #[source]
        source: io::Error,
    },
    #[error("NVML event monitor setup failed: {0}")]
    EventMonitorSetup(String),
    #[error("configured MPS worker identity cannot access {path}: {reason}")]
    MpsWorkerAccess { path: PathBuf, reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_devices_filters_empty_and_disabled_values() {
        assert!(parse_visible_devices("").is_empty());
        assert!(parse_visible_devices("-1").is_empty());
        assert!(parse_visible_devices("-1,0").is_empty());
        assert_eq!(
            parse_visible_devices("0, GPU-abc"),
            ["0".to_owned(), "GPU-abc".to_owned()].into_iter().collect()
        );
    }

    #[test]
    fn mig_uuid_visibility_is_matched_after_parent_enumeration() {
        let selectors = parse_visible_devices("MIG-01234567-89ab-cdef");
        assert!(!is_visible(Some(&selectors), 0, "GPU-parent"));
        assert!(selectors.contains("MIG-01234567-89ab-cdef"));
    }

    #[test]
    fn mps_fails_closed_without_safe_peer_termination() {
        let capability = MpsCapability::detect();
        assert!(!capability.shareable);
        assert!(!capability.compute_limit);
        assert!(!capability.vram_limit);
        assert!(!capability.safe_client_termination);
    }

    #[test]
    fn parses_bounded_mps_process_snapshots() {
        let clients = parse_ps(
            "PID ID SERVER DEVICE NAMESPACE COMMAND ATTRIBUTES\n\
             9741 0 6472 GPU-cb1213a3-d6a4 4026531836 ./nbody --steps 4\n\
             9743 1 6472 GPU-cb1213a3-d6a4 4026531836 ./matrixMul\n",
        )
        .expect("valid MPS ps response");
        assert_eq!(clients.len(), 2);
        assert_eq!(clients[0].pid, 9741);
        assert_eq!(clients[0].server_pid, 6472);
        assert_eq!(clients[0].command, "./nbody --steps 4");
        assert!(parse_ps("PID SERVER DEVICE\n").is_err());
        assert!(
            parse_ps("PID ID SERVER DEVICE NAMESPACE COMMAND\nnot-a-pid 0 6472 GPU-a 1 app\n")
                .is_err()
        );
    }

    #[test]
    fn termination_requires_the_exact_success_response() {
        assert!(parse_termination_response("0\n").is_ok());
        assert!(parse_termination_response("CUDA_SUCCESS\n").is_err());
        assert!(parse_termination_response("0\nwarning\n").is_err());
    }

    #[test]
    fn client_environment_is_budgeted_and_deterministic() {
        let controller = MpsController {
            control_executable: PathBuf::from("/usr/bin/nvidia-cuda-mps-control"),
            pipe_directory: PathBuf::from("/run/nvidia-mps"),
            command_timeout: Duration::from_secs(1),
            max_control_output_bytes: 1024,
        };
        let limits = MpsClientLimits {
            compute_share_per_mille: 337,
            vram_bytes_by_device: [
                ("GPU-abcdef01".to_owned(), 3 * 1024 * MEBIBYTE + 9),
                ("0".to_owned(), 1536 * MEBIBYTE + 1),
            ]
            .into_iter()
            .collect(),
        };
        let environment = controller
            .client_environment(&limits)
            .expect("representable limits");
        assert_eq!(
            environment
                .get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
                .map(String::as_str),
            Some("33")
        );
        assert_eq!(
            environment
                .get("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT")
                .map(String::as_str),
            Some("0=1536M,GPU-abcdef01=3072M")
        );
        assert_eq!(
            environment
                .get("CUDA_MPS_PIPE_DIRECTORY")
                .map(String::as_str),
            Some("/run/nvidia-mps")
        );
    }

    #[test]
    fn client_environment_rejects_unrepresentable_limits() {
        assert!(matches!(compute_percent(9), Err(GpuError::ComputeShare(9))));
        assert!(matches!(
            format_vram_limits(&[("0".to_owned(), MEBIBYTE - 1)].into_iter().collect()),
            Err(GpuError::MpsVramLimit(_))
        ));
        assert!(matches!(
            format_vram_limits(&[("0,1".to_owned(), MEBIBYTE)].into_iter().collect()),
            Err(GpuError::MpsDeviceSelector(_))
        ));
    }

    #[tokio::test]
    #[ignore = "requires an administrator-provisioned NVIDIA MPS control daemon"]
    async fn probes_real_mps_control_daemon_when_provisioned() {
        let Some(pipe_directory) = std::env::var_os("CUDA_MPS_PIPE_DIRECTORY") else {
            eprintln!("CUDA_MPS_PIPE_DIRECTORY is unset; skipping MPS host probe");
            return;
        };
        let control_executable = PathBuf::from("/usr/bin/nvidia-cuda-mps-control");
        if !control_executable.is_file() {
            eprintln!("nvidia-cuda-mps-control is unavailable; skipping MPS host probe");
            return;
        }
        let controller = MpsController::probe(&MpsConfig {
            control_executable: Some(control_executable),
            pipe_directory: Some(PathBuf::from(pipe_directory)),
            ..MpsConfig::default()
        })
        .await
        .expect("configured MPS daemon must pass the control protocol probe");
        let capability = controller.capability();
        assert!(capability.compute_limit);
        assert!(capability.vram_limit);
        assert!(capability.safe_client_termination);
        assert!(capability.shareable);
    }

    #[test]
    fn ten_percent_vram_headroom_is_reserved() {
        let total = 80_u64 * 1024 * 1024 * 1024;
        assert_eq!(total.saturating_mul(90) / 100, 72_u64 * 1024 * 1024 * 1024);
    }
}

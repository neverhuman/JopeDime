use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use jain_smartcluster_core::{
    DaemonMode, DaemonStatus, EffectiveCapacity as ProtocolCapacity, GpuDeviceStatus,
    PressureLevel as ProtocolPressureLevel, PressureStatus, ProtocolVersion, SchedulerCapabilities,
    Timestamp,
};

use crate::capacity::EffectiveCapacity;
use crate::cgroup::{CgroupManager, IsolationMode};
use crate::gpu::GpuInventory;
use crate::overload::{OverloadStage, PressureLevel};
use crate::psi::PsiResource;

#[derive(Clone, Debug)]
pub struct RuntimeHealth {
    inner: Arc<RwLock<HealthState>>,
}

#[derive(Clone, Debug)]
struct HealthState {
    started_at: Timestamp,
    paused: bool,
    pressure: PressureStatus,
    overload_stage: OverloadStage,
    queued_jobs: u64,
    running_jobs: u64,
    mode: DaemonMode,
    capabilities: SchedulerCapabilities,
    capacity: ProtocolCapacity,
    gpus: Vec<GpuDeviceStatus>,
}

impl RuntimeHealth {
    pub fn new(
        capacity: &EffectiveCapacity,
        cgroups: &CgroupManager,
        gpu: &GpuInventory,
        psi_registered: bool,
    ) -> Self {
        let delegated = cgroups.mode() == IsolationMode::Delegated;
        let state = HealthState {
            started_at: Timestamp::from_unix_millis(now_unix_ms()),
            paused: false,
            pressure: PressureStatus::default(),
            overload_stage: OverloadStage::Normal,
            queued_jobs: 0,
            running_jobs: 0,
            mode: if delegated {
                DaemonMode::Full
            } else {
                DaemonMode::SafeSingleAllocation
            },
            capabilities: SchedulerCapabilities {
                cgroup_v2: std::path::Path::new("/sys/fs/cgroup/cgroup.controllers").is_file(),
                cgroup_delegated: delegated,
                // V1 uses the stopped Rust helper fallback. Never claim clone3 support until
                // CLONE_INTO_CGROUP is actually selected for launches.
                clone_into_cgroup: false,
                // The stopped-child fallback does not yet select pidfds for lifecycle
                // operations, so capability evidence must remain conservative.
                pidfd: false,
                freezer: delegated && cgroups.root().join("cgroup.freeze").is_file(),
                psi: psi_registered,
                nvml: gpu.nvml_available,
                mps: gpu.mps.shareable,
                mps_unavailable_reason: gpu.mps.unavailable_reason.clone(),
                mig_inventory: gpu.mig_inventory,
                pty: cfg!(target_os = "linux"),
                scratch_quota: false,
            },
            capacity: ProtocolCapacity {
                cpu_millicores: capacity.cpu_millicores.try_into().unwrap_or(u32::MAX),
                memory_bytes: capacity.memory_max_bytes,
                swap_bytes: capacity.swap_max_bytes,
                scratch_bytes: capacity.scratch_available_bytes,
                logical_cpus: capacity.cpus.iter().collect(),
                numa_nodes: capacity.numa_nodes.len().try_into().unwrap_or(u16::MAX),
            },
            gpus: gpu
                .devices
                .iter()
                .map(|device| GpuDeviceStatus {
                    id: device.id.clone(),
                    uuid: device.uuid.clone(),
                    name: device.name.clone(),
                    total_vram_bytes: device.total_vram_bytes,
                    allocatable_vram_bytes: device.allocatable_vram_bytes,
                    healthy: device.healthy,
                    mig_instance: device.mig_instance.clone(),
                    temperature_millidegrees_celsius: device.temperature_millidegrees_celsius,
                })
                .collect(),
        };
        Self {
            inner: Arc::new(RwLock::new(state)),
        }
    }

    pub fn status(&self) -> DaemonStatus {
        let state = self
            .inner
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        DaemonStatus {
            daemon_version: env!("CARGO_PKG_VERSION").to_owned(),
            protocol_version: ProtocolVersion::CURRENT,
            mode: state.mode,
            started_at: state.started_at,
            paused: state.paused,
            capabilities: state.capabilities.clone(),
            effective_capacity: state.capacity.clone(),
            pressure: state.pressure.clone(),
            gpus: state.gpus.clone(),
            queued_jobs: state.queued_jobs,
            running_jobs: state.running_jobs,
        }
    }

    pub fn set_paused(&self, paused: bool) {
        self.write().paused = paused;
    }

    pub fn set_job_counts(&self, queued: u64, running: u64) {
        let mut state = self.write();
        state.queued_jobs = queued;
        state.running_jobs = running;
    }

    pub fn set_pressure(&self, resource: PsiResource, level: PressureLevel) {
        let state = &mut *self.write();
        let level = protocol_pressure(level);
        match resource {
            PsiResource::Cpu => state.pressure.cpu = level,
            PsiResource::Memory => state.pressure.memory = level,
            PsiResource::Io => state.pressure.io = level,
        }
    }

    pub fn set_overload_stage(&self, stage: OverloadStage) {
        let mut state = self.write();
        state.overload_stage = stage;
        state.pressure.admissions_clamped = stage != OverloadStage::Normal;
    }

    pub fn disable_mps(&self, reason: String) {
        let mut state = self.write();
        state.capabilities.mps = false;
        state.capabilities.mps_unavailable_reason = Some(reason);
    }

    pub fn update_gpu_inventory(&self, gpu: &GpuInventory) {
        let mut state = self.write();
        state.capabilities.nvml = gpu.nvml_available;
        state.capabilities.mig_inventory = gpu.mig_inventory;
        if state.capabilities.mps {
            state.capabilities.mps = gpu.mps.shareable;
            state.capabilities.mps_unavailable_reason = gpu.mps.unavailable_reason.clone();
        }
        state.gpus = gpu
            .devices
            .iter()
            .map(|device| GpuDeviceStatus {
                id: device.id.clone(),
                uuid: device.uuid.clone(),
                name: device.name.clone(),
                total_vram_bytes: device.total_vram_bytes,
                allocatable_vram_bytes: device.allocatable_vram_bytes,
                healthy: device.healthy,
                mig_instance: device.mig_instance.clone(),
                temperature_millidegrees_celsius: device.temperature_millidegrees_celsius,
            })
            .collect();
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, HealthState> {
        self.inner
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

const fn protocol_pressure(level: PressureLevel) -> ProtocolPressureLevel {
    match level {
        PressureLevel::Normal => ProtocolPressureLevel::Normal,
        PressureLevel::Warning => ProtocolPressureLevel::Warning,
        PressureLevel::Critical => ProtocolPressureLevel::Critical,
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capacity::CpuSet;

    #[test]
    fn status_reports_safe_mode_without_claiming_isolation() {
        let temp = tempfile::tempdir().expect("tempdir");
        let cgroups = CgroupManager::initialize(temp.path().join("missing"), true)
            .expect("embedded fallback");
        let capacity = EffectiveCapacity {
            cgroup_path: temp.path().to_path_buf(),
            cpus: CpuSet::parse("0-1").expect("cpus"),
            numa_nodes: Vec::new(),
            cpu_millicores: 2_000,
            memory_max_bytes: 1024,
            swap_max_bytes: None,
            scratch_available_bytes: 2048,
        };
        let gpu = GpuInventory {
            nvml_available: false,
            mig_inventory: false,
            devices: Vec::new(),
            unavailable_reason: Some("test".to_owned()),
            mps: crate::gpu::MpsCapability::detect(),
        };
        let status = RuntimeHealth::new(&capacity, &cgroups, &gpu, false).status();
        assert_eq!(status.mode, DaemonMode::SafeSingleAllocation);
        assert!(!status.capabilities.cgroup_delegated);
        assert!(!status.capabilities.mps);
    }
}

use std::collections::BTreeSet;

use jain_smartcluster_core::{GpuMode, GpuRequest};

use super::{
    GpuAllocation, GpuAllocationMode, GpuCompatibility, GpuDevice, GpuDeviceKind,
    GpuLearningSnapshot, GpuPolicyDecision, GpuPolicyMode, SchedulerError,
};

const SHARE_LEARNING_RUNS: u32 = 5;
const LEARNED_VRAM_MARGIN_BYTES: u64 = 512 * 1024 * 1024;

/// Pure fail-safe policy, exposed for model/property tests and admission logs.
pub fn gpu_policy(
    request: GpuRequest,
    compatibility: GpuCompatibility,
    learning: Option<GpuLearningSnapshot>,
    mig_available: bool,
) -> GpuPolicyDecision {
    let learning = learning.unwrap_or_default();
    let learned_vram = learning
        .max_observed_vram_bytes
        .saturating_mul(5)
        .saturating_div(4)
        .saturating_add(LEARNED_VRAM_MARGIN_BYTES);
    let reserved_vram_bytes = request.vram_bytes.max(learned_vram);
    if request.mode == GpuMode::Exclusive {
        return exclusive_policy(request.vram_bytes, "exclusive request");
    }
    if learning.exclusive_retry_pending {
        return exclusive_policy(request.vram_bytes, "exclusive retry after shared CUDA OOM");
    }
    if learning.sharing_disabled {
        return exclusive_policy(request.vram_bytes, "sharing disabled after shared CUDA OOM");
    }
    if !compatibility.trusted {
        return exclusive_policy(
            request.vram_bytes,
            "workload is not allowlisted for GPU sharing",
        );
    }
    if request.mode == GpuMode::MigPreferred && compatibility.mig && mig_available {
        return GpuPolicyDecision {
            mode: GpuPolicyMode::MigInstance,
            reserved_vram_bytes: request.vram_bytes,
            compute_share_per_mille: 1_000,
            reason: "existing MIG instance",
        };
    }
    if !compatibility.mps {
        return exclusive_policy(
            request.vram_bytes,
            "workload has no supported MPS isolation",
        );
    }
    if learning.clean_exclusive_runs < SHARE_LEARNING_RUNS {
        return exclusive_policy(
            request.vram_bytes,
            "fingerprint has fewer than five clean runs",
        );
    }
    GpuPolicyDecision {
        mode: GpuPolicyMode::SharedMps,
        reserved_vram_bytes,
        compute_share_per_mille: request.compute_share_per_mille,
        reason: "MPS budgets validated",
    }
}

fn exclusive_policy(vram_bytes: u64, reason: &'static str) -> GpuPolicyDecision {
    GpuPolicyDecision {
        mode: GpuPolicyMode::Exclusive,
        reserved_vram_bytes: vram_bytes,
        compute_share_per_mille: 1_000,
        reason,
    }
}

#[derive(Clone)]
struct DeviceState {
    device: GpuDevice,
    exclusive: bool,
    shared_vram: u64,
    shared_compute: u16,
}

#[derive(Clone)]
pub(super) struct DeviceStates(Vec<DeviceState>);

impl DeviceStates {
    pub(super) fn new(devices: &[GpuDevice]) -> Self {
        let mut devices: Vec<_> = devices
            .iter()
            .filter(|device| device.healthy)
            .cloned()
            .map(|device| DeviceState {
                device,
                exclusive: false,
                shared_vram: 0,
                shared_compute: 0,
            })
            .collect();
        devices.sort_by(|left, right| left.device.id.cmp(&right.device.id));
        Self(devices)
    }

    pub(super) fn add_running(
        &mut self,
        allocations: &[GpuAllocation],
    ) -> Result<(), SchedulerError> {
        for allocation in allocations {
            let state = self
                .0
                .iter_mut()
                .find(|state| state.device.id == allocation.device_id)
                .ok_or_else(|| SchedulerError::UnknownGpu(allocation.device_id.clone()))?;
            match allocation.mode {
                GpuAllocationMode::Exclusive | GpuAllocationMode::MigInstance => {
                    if state.exclusive || state.shared_compute > 0 {
                        return Err(SchedulerError::ConflictingGpuAllocation(
                            allocation.device_id.clone(),
                        ));
                    }
                    state.exclusive = true;
                }
                GpuAllocationMode::SharedMps => {
                    if state.exclusive {
                        return Err(SchedulerError::ConflictingGpuAllocation(
                            allocation.device_id.clone(),
                        ));
                    }
                    state.shared_compute = state
                        .shared_compute
                        .checked_add(allocation.compute_share_per_mille)
                        .ok_or_else(|| {
                            SchedulerError::OvercommittedGpu(allocation.device_id.clone())
                        })?;
                    state.shared_vram = state
                        .shared_vram
                        .saturating_add(allocation.reserved_vram_bytes);
                    if state.shared_compute > 1_000 || state.shared_vram > state.device.vram_bytes {
                        return Err(SchedulerError::OvercommittedGpu(
                            allocation.device_id.clone(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn allocate(
        &mut self,
        request: Option<GpuRequest>,
        compatibility: GpuCompatibility,
        learning: Option<GpuLearningSnapshot>,
        vram_headroom_per_mille: u16,
    ) -> Result<Vec<GpuAllocation>, String> {
        let Some(request) = request else {
            return Ok(Vec::new());
        };
        let mig_available = self.0.iter().any(|state| {
            state.device.kind == GpuDeviceKind::MigInstance
                && !state.exclusive
                && state.shared_compute == 0
        });
        let policy = gpu_policy(request, compatibility, learning, mig_available);
        let mut allocations = Vec::with_capacity(usize::from(request.count));
        let mut selected_devices = BTreeSet::new();
        for _ in 0..request.count {
            let selected = match policy.mode {
                GpuPolicyMode::Exclusive => self
                    .0
                    .iter_mut()
                    .filter(|state| {
                        !selected_devices.contains(&state.device.id)
                            && !state.exclusive
                            && state.shared_compute == 0
                            && (policy.reserved_vram_bytes == u64::MAX
                                || policy.reserved_vram_bytes <= state.device.vram_bytes)
                    })
                    .min_by_key(|state| match state.device.kind {
                        GpuDeviceKind::Physical => 0,
                        GpuDeviceKind::MigInstance => 1,
                    }),
                GpuPolicyMode::MigInstance => self.0.iter_mut().find(|state| {
                    !selected_devices.contains(&state.device.id)
                        && state.device.kind == GpuDeviceKind::MigInstance
                        && !state.exclusive
                        && state.shared_compute == 0
                        && (policy.reserved_vram_bytes == u64::MAX
                            || policy.reserved_vram_bytes <= state.device.vram_bytes)
                }),
                GpuPolicyMode::SharedMps => self.0.iter_mut().find(|state| {
                    let usable_vram = state
                        .device
                        .vram_bytes
                        .saturating_mul(u64::from(1_000 - vram_headroom_per_mille))
                        / 1_000;
                    !selected_devices.contains(&state.device.id)
                        && state.device.kind == GpuDeviceKind::Physical
                        && !state.exclusive
                        && u32::from(state.shared_compute)
                            + u32::from(policy.compute_share_per_mille)
                            <= 1_000
                        && state.shared_vram.saturating_add(policy.reserved_vram_bytes)
                            <= usable_vram
                }),
            }
            .ok_or_else(|| format!("no GPU satisfies {} allocation", policy.reason))?;
            selected_devices.insert(selected.device.id.clone());
            let reserved_vram_bytes = if policy.reserved_vram_bytes == u64::MAX
                && matches!(
                    policy.mode,
                    GpuPolicyMode::Exclusive | GpuPolicyMode::MigInstance
                ) {
                selected.device.vram_bytes
            } else {
                policy.reserved_vram_bytes
            };
            let mode = match policy.mode {
                GpuPolicyMode::Exclusive => {
                    selected.exclusive = true;
                    match selected.device.kind {
                        GpuDeviceKind::Physical => GpuAllocationMode::Exclusive,
                        GpuDeviceKind::MigInstance => GpuAllocationMode::MigInstance,
                    }
                }
                GpuPolicyMode::MigInstance => {
                    selected.exclusive = true;
                    GpuAllocationMode::MigInstance
                }
                GpuPolicyMode::SharedMps => {
                    selected.shared_compute = selected
                        .shared_compute
                        .saturating_add(policy.compute_share_per_mille);
                    selected.shared_vram = selected
                        .shared_vram
                        .saturating_add(policy.reserved_vram_bytes);
                    GpuAllocationMode::SharedMps
                }
            };
            allocations.push(GpuAllocation {
                device_id: selected.device.id.clone(),
                mode,
                reserved_vram_bytes,
                compute_share_per_mille: policy.compute_share_per_mille,
            });
        }
        Ok(allocations)
    }
}

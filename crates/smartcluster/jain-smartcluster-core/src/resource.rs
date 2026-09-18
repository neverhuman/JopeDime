use serde::{Deserialize, Serialize};

use crate::{Validate, ValidationError};

/// Elastic CPU request. The scheduler admits against `minimum_millicores`.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CpuRequest {
    pub minimum_millicores: u32,
    pub desired_millicores: u32,
    pub maximum_millicores: u32,
}

impl CpuRequest {
    pub const fn fixed(millicores: u32) -> Self {
        Self {
            minimum_millicores: millicores,
            desired_millicores: millicores,
            maximum_millicores: millicores,
        }
    }
}

impl Validate for CpuRequest {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.minimum_millicores > self.desired_millicores {
            return Err(ValidationError::new(
                "cpu.minimum_millicores",
                "must not exceed desired_millicores",
            ));
        }
        if self.desired_millicores > self.maximum_millicores {
            return Err(ValidationError::new(
                "cpu.desired_millicores",
                "must not exceed maximum_millicores",
            ));
        }
        Ok(())
    }
}

/// Recoverable and hard cgroup memory boundaries.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MemoryRequest {
    pub soft_bytes: u64,
    pub hard_bytes: u64,
}

impl MemoryRequest {
    pub const fn fixed(bytes: u64) -> Self {
        Self {
            soft_bytes: bytes,
            hard_bytes: bytes,
        }
    }
}

impl Validate for MemoryRequest {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.hard_bytes == 0 {
            return Err(ValidationError::new(
                "ram.hard_bytes",
                "must be greater than zero",
            ));
        }
        if self.soft_bytes > self.hard_bytes {
            return Err(ValidationError::new(
                "ram.soft_bytes",
                "must not exceed hard_bytes",
            ));
        }
        Ok(())
    }
}

/// Isolation preference for an optional GPU allocation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuMode {
    Exclusive,
    Shareable,
    MigPreferred,
}

/// A fail-safe GPU allocation request.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GpuRequest {
    pub count: u16,
    pub vram_bytes: u64,
    /// Requested active-thread share, from 1 through 1000 per mille.
    pub compute_share_per_mille: u16,
    pub mode: GpuMode,
}

impl Validate for GpuRequest {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.count == 0 {
            return Err(ValidationError::new(
                "gpu.count",
                "must be greater than zero",
            ));
        }
        if self.vram_bytes == 0 {
            return Err(ValidationError::new(
                "gpu.vram_bytes",
                "must be greater than zero",
            ));
        }
        if !(1..=1000).contains(&self.compute_share_per_mille) {
            return Err(ValidationError::new(
                "gpu.compute_share_per_mille",
                "must be between 1 and 1000",
            ));
        }
        if !matches!(self.mode, GpuMode::Exclusive) && self.compute_share_per_mille < 10 {
            return Err(ValidationError::new(
                "gpu.compute_share_per_mille",
                "shareable GPU allocations must request at least 10 (one percent)",
            ));
        }
        if matches!(self.mode, GpuMode::Exclusive) && self.compute_share_per_mille != 1000 {
            return Err(ValidationError::new(
                "gpu.compute_share_per_mille",
                "exclusive allocations must request 1000",
            ));
        }
        Ok(())
    }
}

/// CPU, memory, scratch, and optional GPU resources requested by a workload.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResourceRequest {
    pub cpu: CpuRequest,
    pub ram: MemoryRequest,
    pub scratch_bytes: u64,
    pub gpu: Option<GpuRequest>,
}

impl ResourceRequest {
    pub const fn cpu_and_ram(cpu_millicores: u32, ram_bytes: u64) -> Self {
        Self {
            cpu: CpuRequest::fixed(cpu_millicores),
            ram: MemoryRequest::fixed(ram_bytes),
            scratch_bytes: 0,
            gpu: None,
        }
    }
}

impl Validate for ResourceRequest {
    fn validate(&self) -> Result<(), ValidationError> {
        self.cpu
            .validate()
            .map_err(|error| error.prefixed("resources"))?;
        self.ram
            .validate()
            .map_err(|error| error.prefixed("resources"))?;
        if let Some(gpu) = self.gpu {
            gpu.validate()
                .map_err(|error| error.prefixed("resources"))?;
        }
        if self.cpu.maximum_millicores == 0 && self.gpu.is_none() {
            return Err(ValidationError::new(
                "resources",
                "must request CPU or GPU capacity",
            ));
        }
        if self.gpu.is_none() && (self.cpu.minimum_millicores == 0 || self.ram.soft_bytes == 0) {
            return Err(ValidationError::new(
                "resources",
                "CPU-only workloads require nonzero CPU minimum and RAM soft reservations",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn ordered_cpu_ranges_validate(minimum in 0_u32..100_000, desired_extra in 0_u32..100_000, maximum_extra in 0_u32..100_000) {
            let desired = minimum.saturating_add(desired_extra);
            let maximum = desired.saturating_add(maximum_extra);
            let request = CpuRequest {
                minimum_millicores: minimum,
                desired_millicores: desired,
                maximum_millicores: maximum,
            };
            prop_assert!(request.validate().is_ok());
        }
    }

    #[test]
    fn invalid_resource_ranges_are_rejected() {
        assert!(CpuRequest {
            minimum_millicores: 2000,
            desired_millicores: 1000,
            maximum_millicores: 4000,
        }
        .validate()
        .is_err());
        assert!(MemoryRequest {
            soft_bytes: 2,
            hard_bytes: 1,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn exclusive_gpu_requests_full_compute_share() {
        let request = GpuRequest {
            count: 1,
            vram_bytes: 1024,
            compute_share_per_mille: 500,
            mode: GpuMode::Exclusive,
        };
        assert!(request.validate().is_err());
    }

    #[test]
    fn shareable_gpu_request_must_be_representable_by_mps() {
        let request = GpuRequest {
            count: 1,
            vram_bytes: 1024,
            compute_share_per_mille: 9,
            mode: GpuMode::Shareable,
        };
        assert!(request.validate().is_err());
    }

    #[test]
    fn cpu_only_workload_cannot_bypass_drf_with_zero_minima() {
        let request = ResourceRequest {
            cpu: CpuRequest {
                minimum_millicores: 0,
                desired_millicores: 100,
                maximum_millicores: 100,
            },
            ram: MemoryRequest {
                soft_bytes: 0,
                hard_bytes: 1024,
            },
            scratch_bytes: 0,
            gpu: None,
        };
        assert!(request.validate().is_err());
    }
}

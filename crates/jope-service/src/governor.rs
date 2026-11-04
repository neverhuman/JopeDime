//! Resource policy at fixed planned R (spec §9.2). Sensors and engine effects are injected.
//!
//! Poll before admission and while a round runs; apply effects only after a round drains.
//! The engine adapter owns durable checkpointing and actual worker retirement. Opening the
//! resource gate never resumes a paused run or changes its plan, evaluator limits or RNG.

use crate::{telemetry::Sensors, warnings::catalog};
use jope_api::{
    decimal::DecimalU64,
    error::{ErrorCode, Problem},
    ResourceEstimates, Warning, WarningState,
};
use std::{io, num::NonZeroU32, num::NonZeroU64};

pub const DEFAULT_DISK_RESERVE: u64 = 1024 * 1024 * 1024;

/// Byte estimates supplied by registration/planning before the plan digest is computed.
#[derive(Clone, Copy, Debug, Default)]
pub struct Requirements {
    pub bundle_ram_bytes: u64,
    pub matrix_ram_bytes: u64,
    pub per_worker_ram_bytes: u64,
    pub dataset_copy_bytes: u64,
    pub matrix_disk_bytes: u64,
    pub artifact_bytes: u64,
}

/// Operations supplied by the engine owner, not a second worker supervisor or store.
pub trait Control {
    /// This is an additional admission gate; `true` must not issue run Continue.
    fn resource_admission(&mut self, allowed: bool) -> io::Result<()>;
    /// At a drained boundary, retire/restore exactly the difference in live workers.
    /// Preserve R, seeds, planned per-worker limits and fixed fit-thread counts.
    /// This absolute request must be idempotent when retried after an error.
    fn live_workers(&mut self, workers: NonZeroU32) -> io::Result<()>;
    /// Commit the drained work and durably pause with this reason before returning success.
    /// On failure keep admission closed; retries must be idempotent.
    fn checkpoint_and_pause(&mut self, reason: ErrorCode) -> io::Result<()>;
}

#[derive(Debug)]
pub struct Observation {
    pub memory_available: Result<u64, String>,
    pub disk_free: Result<u64, String>,
    pub admission_allowed: bool,
    pub warnings: Vec<Warning>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundEffect {
    None,
    Workers { previous: u32, current: u32 },
    Paused(ErrorCode),
}

pub struct Governor {
    total_ram: NonZeroU64,
    planned_workers: NonZeroU32,
    live_workers: NonZeroU32,
    planned_r: NonZeroU32,
    disk_reserve: u64,
    memory_pressure: bool,
    retire_pending: bool,
    worker_target: Option<NonZeroU32>,
    disk_low: bool,
    memory_available: Option<u64>,
    pause_pending: bool,
    admission_allowed: bool,
    admission_confirmed: bool,
    last_boundary: Option<u64>,
}

impl Governor {
    pub fn new(total_ram: NonZeroU64, planned_workers: NonZeroU32, disk_reserve: u64) -> Self {
        Self {
            total_ram,
            planned_workers,
            live_workers: planned_workers,
            planned_r: NonZeroU32::new((planned_workers.get() / 4).max(1)).unwrap(),
            disk_reserve,
            memory_pressure: false,
            retire_pending: false,
            worker_target: None,
            disk_low: false,
            memory_available: None,
            pause_pending: false,
            admission_allowed: false,
            admission_confirmed: false,
            last_boundary: None,
        }
    }

    pub fn planned_workers(&self) -> NonZeroU32 {
        self.planned_workers
    }
    /// Last acknowledged capacity; an in-doubt effect keeps admission closed until reconciled.
    pub fn live_workers(&self) -> NonZeroU32 {
        self.live_workers
    }
    pub fn planned_r(&self) -> NonZeroU32 {
        self.planned_r
    }

    /// Return canonical estimates for the owning plan builder; never edit a hashed plan.
    /// Exact equality leaves the reserve intact and is admissible. Unavailable evidence
    /// cannot establish that an operation is safe; it is never interpreted as zero bytes.
    pub fn preflight(
        &self,
        requirements: Requirements,
        sensors: &mut dyn Sensors,
    ) -> Result<ResourceEstimates, Box<Problem>> {
        let overflow = || {
            problem(
                ErrorCode::InvalidInput,
                "Resource estimate overflows u64 bytes.",
            )
        };
        let workers = requirements
            .per_worker_ram_bytes
            .checked_mul(u64::from(self.planned_workers.get()))
            .ok_or_else(overflow)?;
        let peak_ram = requirements
            .bundle_ram_bytes
            .checked_add(requirements.matrix_ram_bytes)
            .and_then(|sum| sum.checked_add(workers))
            .ok_or_else(overflow)?;
        let disk = requirements
            .dataset_copy_bytes
            .checked_add(requirements.matrix_disk_bytes)
            .and_then(|sum| sum.checked_add(requirements.artifact_bytes))
            .ok_or_else(overflow)?;
        let needed = disk.checked_add(self.disk_reserve).ok_or_else(overflow)?;
        let free = sensors.disk_free().map_err(|reason| {
            problem(
                ErrorCode::CapabilityUnavailable,
                format!("Store disk-free sensor unavailable: {reason}"),
            )
        })?;
        if free < needed {
            return Err(problem(
                ErrorCode::DiskLow,
                format!(
                    "Operation needs {disk} bytes plus {} reserve; store has {free} bytes free.",
                    self.disk_reserve
                ),
            ));
        }
        Ok(ResourceEstimates {
            peak_ram_bytes: DecimalU64(peak_ram),
            disk_bytes: DecimalU64(disk),
        })
    }

    /// Close admission immediately on low/unknown disk evidence. A confirmed low sample
    /// latches a pause even if space recovers before the next drained boundary.
    pub fn poll(
        &mut self,
        sensors: &mut dyn Sensors,
        control: &mut dyn Control,
    ) -> io::Result<Observation> {
        let memory = sensors.mem_available();
        let disk = sensors.disk_free();
        let low_disk = disk.as_ref().is_ok_and(|free| *free < self.disk_reserve);
        self.pause_pending |= low_disk && !self.disk_low;
        let allowed = disk.as_ref().is_ok_and(|free| *free >= self.disk_reserve)
            && !self.pause_pending
            && self.worker_target.is_none();
        // A failed adapter must not leave the policy claiming admission is safe.
        self.admission_allowed = false;
        self.admission_confirmed = false;
        control.resource_admission(allowed)?;
        self.admission_allowed = allowed;
        self.admission_confirmed = true;
        self.memory_available = memory.as_ref().ok().copied();
        let mut warnings = Vec::new();
        if let Ok(available) = memory.as_ref() {
            let pressure = if self.below_ten_percent(*available) {
                self.retire_pending = true;
                true
            } else if self.above_twenty_percent(*available) {
                self.retire_pending = false;
                false
            } else {
                self.memory_pressure
            };
            if pressure != self.memory_pressure {
                warnings.push(warning(
                    "MEMORY_PRESSURE",
                    pressure,
                    *available,
                    self.total_ram.get() as f64 * if pressure { 0.1 } else { 0.2 },
                ));
                self.memory_pressure = pressure;
            }
        }
        if let Ok(free) = disk.as_ref() {
            if low_disk != self.disk_low {
                warnings.push(warning(
                    "DISK_LOW",
                    low_disk,
                    *free,
                    self.disk_reserve as f64,
                ));
                self.disk_low = low_disk;
            }
        }
        Ok(Observation {
            memory_available: memory,
            disk_free: disk,
            admission_allowed: allowed,
            warnings,
        })
    }

    /// Apply at most one live-worker change per monotonically numbered drained boundary.
    /// A low sample remains pending through the band; confirmed >20% recovery supersedes it.
    /// Duplicate calls do nothing. Failed effects retain their absolute target and close
    /// admission until acknowledged, even if sensor readings change; R is immutable.
    pub fn round_boundary(
        &mut self,
        boundary: u64,
        control: &mut dyn Control,
    ) -> io::Result<RoundEffect> {
        if self.last_boundary == Some(boundary) {
            return Ok(RoundEffect::None);
        }
        if self.last_boundary.is_some_and(|last| boundary < last) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "round boundary moved backwards",
            ));
        }
        if !self.admission_confirmed {
            return Err(io::Error::other(
                "resource admission gate was not acknowledged",
            ));
        }
        let effect = if self.pause_pending {
            control.checkpoint_and_pause(ErrorCode::DiskLow)?;
            self.pause_pending = false;
            RoundEffect::Paused(ErrorCode::DiskLow)
        } else {
            let previous = self.live_workers.get();
            let current = match (self.worker_target, self.memory_available) {
                (Some(target), _) => target.get(),
                (None, _) if self.retire_pending => previous.saturating_sub(1).max(1),
                (None, Some(value)) if self.below_ten_percent(value) => {
                    previous.saturating_sub(1).max(1)
                }
                (None, Some(value))
                    if self.above_twenty_percent(value) && self.admission_allowed =>
                {
                    previous.saturating_add(1).min(self.planned_workers.get())
                }
                _ => previous,
            };
            if current != previous {
                let workers = NonZeroU32::new(current).unwrap();
                self.worker_target = Some(workers);
                if let Err(error) = control.live_workers(workers) {
                    self.admission_allowed = false;
                    self.admission_confirmed = false;
                    if let Err(close) = control.resource_admission(false) {
                        return Err(io::Error::other(format!(
                            "worker resize failed ({error}); admission close failed ({close})"
                        )));
                    }
                    self.admission_confirmed = true;
                    return Err(error);
                }
                self.live_workers = workers;
                self.worker_target = None;
                RoundEffect::Workers { previous, current }
            } else {
                RoundEffect::None
            }
        };
        if !matches!(effect, RoundEffect::Paused(_)) {
            self.retire_pending = false;
        }
        self.last_boundary = Some(boundary);
        Ok(effect)
    }

    fn below_ten_percent(&self, available: u64) -> bool {
        u128::from(available) * 10 < u128::from(self.total_ram.get())
    }
    fn above_twenty_percent(&self, available: u64) -> bool {
        u128::from(available) * 5 > u128::from(self.total_ram.get())
    }
}

fn warning(code: &str, raised: bool, value: u64, limit: f64) -> Warning {
    let state = if raised {
        WarningState::Raised
    } else {
        WarningState::Cleared
    };
    let entry = catalog(code).expect("governor warnings are in TEL-06 catalog");
    let (message, remedy) = if code == "DISK_LOW" {
        (
            if raised {
                "Store free space is below the reserve; checkpoint and pause."
            } else {
                "Store disk reserve recovered; an existing pause still requires Continue."
            },
            "Free space or run `jopedime gc`.",
        )
    } else {
        (
            if raised {
                "Host memory pressure; adjust live workers at the next round."
            } else {
                "Host memory recovered above 20%; workers may be restored."
            },
            "Lower effort or close other programs.",
        )
    };
    entry
        .event(state, Some(value as f64), Some(limit), message, remedy)
        .expect("u64 byte counts and fixed thresholds are finite")
}

fn problem(code: ErrorCode, detail: impl Into<String>) -> Box<Problem> {
    Box::new(Problem {
        r#type: "about:blank".into(),
        title: code.as_str().into(),
        status: code.http_status().unwrap_or(500),
        detail: detail.into(),
        instance: None,
        code,
        retriable: code.retriable().unwrap_or(false),
        remedy: if code == ErrorCode::DiskLow {
            "Free space or run `jopedime gc`, then retry."
        } else {
            "Check resource estimates and sensor availability before retrying."
        }
        .into(),
        field: None,
    })
}

#[cfg(test)]
mod tests;

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
        let allowed =
            disk.as_ref().is_ok_and(|free| *free >= self.disk_reserve) && !self.pause_pending;
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
    /// Duplicate calls do nothing. Failed effects can retry the same boundary; R is immutable.
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
            let current = match self.memory_available {
                _ if self.retire_pending => previous.saturating_sub(1).max(1),
                Some(value) if self.below_ten_percent(value) => previous.saturating_sub(1).max(1),
                Some(value) if self.above_twenty_percent(value) && self.admission_allowed => {
                    previous.saturating_add(1).min(self.planned_workers.get())
                }
                _ => previous,
            };
            if current != previous {
                let workers = NonZeroU32::new(current).unwrap();
                control.live_workers(workers)?;
                self.live_workers = workers;
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
mod tests {
    use super::*;
    use crate::telemetry::{GpuReading, ProcessReading};

    struct Readings {
        memory: Result<u64, String>,
        disk: Result<u64, String>,
    }
    impl Readings {
        fn new(memory: u64, disk: u64) -> Self {
            Self {
                memory: Ok(memory),
                disk: Ok(disk),
            }
        }
    }
    impl Sensors for Readings {
        fn mem_available(&mut self) -> Result<u64, String> {
            self.memory.clone()
        }
        fn disk_free(&mut self) -> Result<u64, String> {
            self.disk.clone()
        }
        fn process(&mut self, _: u32) -> Result<ProcessReading, String> {
            panic!("no extra process probe")
        }
        fn logical_cpus(&mut self) -> Result<f64, String> {
            panic!("no CPU probe")
        }
        fn allowed_cpus(&mut self) -> Result<f64, String> {
            panic!("no CPU probe")
        }
        fn gpus(&mut self) -> Result<Vec<GpuReading>, String> {
            panic!("governor never probes GPU")
        }
    }

    #[derive(Default)]
    struct Effects {
        calls: Vec<String>,
        workers: u32,
        admission: bool,
        fail_resize: bool,
        fail_checkpoint: bool,
        fail_admission: bool,
    }
    impl Control for Effects {
        fn resource_admission(&mut self, allowed: bool) -> io::Result<()> {
            if self.fail_admission {
                return Err(io::Error::other("admission gate unavailable"));
            }
            self.admission = allowed;
            self.calls.push(format!("admit:{allowed}"));
            Ok(())
        }
        fn live_workers(&mut self, workers: NonZeroU32) -> io::Result<()> {
            if self.fail_resize {
                return Err(io::Error::other("resize unavailable"));
            }
            self.workers = workers.get();
            self.calls.push(format!("workers:{}", workers.get()));
            Ok(())
        }
        fn checkpoint_and_pause(&mut self, reason: ErrorCode) -> io::Result<()> {
            assert!(!self.admission, "stop admission precedes checkpoint");
            assert_eq!(reason, ErrorCode::DiskLow);
            if self.fail_checkpoint {
                return Err(io::Error::other("checkpoint unavailable"));
            }
            self.calls
                .extend(["commit".into(), "pause:DISK_LOW".into()]);
            Ok(())
        }
    }
    fn governor(workers: u32) -> Governor {
        Governor::new(
            NonZeroU64::new(1000).unwrap(),
            NonZeroU32::new(workers).unwrap(),
            100,
        )
    }

    #[test]
    fn preflight_populates_canonical_estimates_and_preserves_exact_reserve() {
        let gov = governor(8);
        let needs = Requirements {
            bundle_ram_bytes: 10,
            matrix_ram_bytes: 20,
            per_worker_ram_bytes: 30,
            dataset_copy_bytes: 40,
            matrix_disk_bytes: 50,
            artifact_bytes: 60,
        };
        let mut sensors = Readings::new(500, 250);
        let estimates = gov.preflight(needs, &mut sensors).unwrap();
        assert_eq!(estimates.peak_ram_bytes.0, 270);
        assert_eq!(estimates.disk_bytes.0, 150);
        assert_eq!(gov.planned_workers().get(), 8);
        assert_eq!(gov.planned_r().get(), 2);
        sensors.disk = Ok(249);
        let error = gov.preflight(needs, &mut sensors).unwrap_err();
        assert_eq!((error.code, error.status), (ErrorCode::DiskLow, 507));
        assert!(error.remedy.contains("jopedime gc"));
        sensors.disk = Err("permission_denied".into());
        assert_eq!(
            gov.preflight(needs, &mut sensors).unwrap_err().code,
            ErrorCode::CapabilityUnavailable
        );
        sensors.disk = Ok(u64::MAX);
        for needs in [
            Requirements {
                per_worker_ram_bytes: u64::MAX,
                ..Default::default()
            },
            Requirements {
                bundle_ram_bytes: u64::MAX,
                matrix_ram_bytes: 1,
                ..Default::default()
            },
            Requirements {
                dataset_copy_bytes: u64::MAX,
                artifact_bytes: 1,
                ..Default::default()
            },
            Requirements {
                dataset_copy_bytes: u64::MAX,
                ..Default::default()
            },
        ] {
            assert_eq!(
                gov.preflight(needs, &mut sensors).unwrap_err().code,
                ErrorCode::InvalidInput
            );
        }
    }

    #[test]
    fn strict_thresholds_hold_the_band_and_change_only_one_worker_per_boundary() {
        let mut gov = governor(8);
        let mut effects = Effects::default();
        let mut sensors = Readings::new(100, 100);
        assert!(gov
            .poll(&mut sensors, &mut effects)
            .unwrap()
            .warnings
            .is_empty());
        assert_eq!(
            gov.round_boundary(0, &mut effects).unwrap(),
            RoundEffect::None
        );
        sensors.memory = Ok(99);
        let sample = gov.poll(&mut sensors, &mut effects).unwrap();
        assert_eq!(sample.warnings.len(), 1);
        assert_eq!(sample.warnings[0].code, "MEMORY_PRESSURE");
        assert!(sample.warnings[0].host);
        assert_eq!(sample.warnings[0].state, WarningState::Raised);
        assert_eq!(
            gov.live_workers().get(),
            8,
            "poll does not resize mid-round"
        );
        assert_eq!(
            gov.round_boundary(1, &mut effects).unwrap(),
            RoundEffect::Workers {
                previous: 8,
                current: 7
            }
        );
        assert_eq!(
            gov.round_boundary(1, &mut effects).unwrap(),
            RoundEffect::None
        );
        for (boundary, memory) in [(2, 100), (3, 150), (4, 200)] {
            sensors.memory = Ok(memory);
            assert!(gov
                .poll(&mut sensors, &mut effects)
                .unwrap()
                .warnings
                .is_empty());
            assert_eq!(
                gov.round_boundary(boundary, &mut effects).unwrap(),
                RoundEffect::None
            );
            assert_eq!(gov.live_workers().get(), 7);
        }
        sensors.memory = Ok(99);
        gov.poll(&mut sensors, &mut effects).unwrap();
        for boundary in 5..20 {
            gov.round_boundary(boundary, &mut effects).unwrap();
            assert_eq!(gov.planned_r().get(), 2);
        }
        assert_eq!(gov.live_workers().get(), 1);
        sensors.memory = Ok(201);
        let recovered = gov.poll(&mut sensors, &mut effects).unwrap();
        assert_eq!(recovered.warnings[0].state, WarningState::Cleared);
        assert_eq!(
            gov.round_boundary(20, &mut effects).unwrap(),
            RoundEffect::Workers {
                previous: 1,
                current: 2
            }
        );
        for boundary in 21..40 {
            gov.round_boundary(boundary, &mut effects).unwrap();
        }
        assert_eq!(gov.live_workers(), gov.planned_workers());
        assert_eq!(gov.planned_r().get(), 2);
        assert!(gov.round_boundary(38, &mut effects).is_err());
        assert_eq!(governor(1).planned_r().get(), 1);
    }

    #[test]
    fn byte_thresholds_remain_exact_above_f64_integer_precision() {
        let total = u64::MAX - 5; // exactly divisible by ten
        let mut gov = Governor::new(
            NonZeroU64::new(total).unwrap(),
            NonZeroU32::new(2).unwrap(),
            0,
        );
        let mut effects = Effects::default();
        let mut sensors = Readings::new(total / 10, 0);
        assert!(gov
            .poll(&mut sensors, &mut effects)
            .unwrap()
            .warnings
            .is_empty());
        gov.round_boundary(0, &mut effects).unwrap();
        assert_eq!(gov.live_workers().get(), 2);
        sensors.memory = Ok(total / 10 - 1);
        assert_eq!(
            gov.poll(&mut sensors, &mut effects).unwrap().warnings.len(),
            1
        );
        gov.round_boundary(1, &mut effects).unwrap();
        assert_eq!(gov.live_workers().get(), 1);
        sensors.memory = Ok(total / 5);
        assert!(gov
            .poll(&mut sensors, &mut effects)
            .unwrap()
            .warnings
            .is_empty());
        gov.round_boundary(2, &mut effects).unwrap();
        assert_eq!(gov.live_workers().get(), 1);
        sensors.memory = Ok(total / 5 + 1);
        assert_eq!(
            gov.poll(&mut sensors, &mut effects).unwrap().warnings[0].state,
            WarningState::Cleared
        );
        gov.round_boundary(3, &mut effects).unwrap();
        assert_eq!(gov.live_workers().get(), 2);
    }

    #[test]
    fn disk_low_stops_admission_then_checkpoints_and_pauses_once_without_auto_resume() {
        let mut gov = governor(8);
        let mut effects = Effects::default();
        let mut sensors = Readings::new(500, 99);
        let sample = gov.poll(&mut sensors, &mut effects).unwrap();
        assert!(!sample.admission_allowed);
        assert_eq!(sample.warnings[0].code, "DISK_LOW");
        assert!(sample.warnings[0].host);
        assert_eq!(effects.calls, ["admit:false"]);
        sensors.disk = Ok(100);
        let recovered = gov.poll(&mut sensors, &mut effects).unwrap();
        assert!(
            !recovered.admission_allowed,
            "confirmed pause survives early space recovery"
        );
        assert_eq!(recovered.warnings[0].state, WarningState::Cleared);
        assert_eq!(
            gov.round_boundary(0, &mut effects).unwrap(),
            RoundEffect::Paused(ErrorCode::DiskLow)
        );
        assert_eq!(
            effects.calls,
            ["admit:false", "admit:false", "commit", "pause:DISK_LOW"]
        );
        assert_eq!(
            gov.round_boundary(0, &mut effects).unwrap(),
            RoundEffect::None
        );
        assert!(
            gov.poll(&mut sensors, &mut effects)
                .unwrap()
                .admission_allowed
        );
        assert!(!effects.calls.iter().any(|call| call.contains("continue")));
        sensors.disk = Ok(99);
        gov.poll(&mut sensors, &mut effects).unwrap();
        gov.round_boundary(1, &mut effects).unwrap();
        gov.poll(&mut sensors, &mut effects).unwrap();
        assert_eq!(
            gov.round_boundary(2, &mut effects).unwrap(),
            RoundEffect::None
        );
        assert_eq!(gov.planned_r().get(), 2);
    }

    #[test]
    fn unavailable_sources_hold_memory_state_and_never_claim_disk_low_or_recovery() {
        let mut gov = governor(8);
        let mut effects = Effects::default();
        let mut sensors = Readings::new(99, 101);
        gov.poll(&mut sensors, &mut effects).unwrap();
        gov.round_boundary(0, &mut effects).unwrap();
        sensors.memory = Err("unavailable".into());
        sensors.disk = Err("permission_denied".into());
        let sample = gov.poll(&mut sensors, &mut effects).unwrap();
        assert!(!sample.admission_allowed);
        assert!(sample.warnings.is_empty());
        assert_eq!(sample.disk_free, Err("permission_denied".into()));
        assert_eq!(
            gov.round_boundary(1, &mut effects).unwrap(),
            RoundEffect::None
        );
        assert_eq!(gov.live_workers().get(), 7);
        sensors.memory = Ok(150);
        sensors.disk = Ok(101);
        assert!(gov
            .poll(&mut sensors, &mut effects)
            .unwrap()
            .warnings
            .is_empty());
        assert_eq!(
            gov.round_boundary(2, &mut effects).unwrap(),
            RoundEffect::None
        );
        sensors.memory = Ok(201);
        assert_eq!(
            gov.poll(&mut sensors, &mut effects).unwrap().warnings[0].state,
            WarningState::Cleared
        );
    }

    #[test]
    fn failed_actions_retry_without_advancing_worker_count_or_boundary() {
        let mut gov = governor(2);
        let mut effects = Effects {
            fail_resize: true,
            ..Default::default()
        };
        let mut sensors = Readings::new(99, 100);
        gov.poll(&mut sensors, &mut effects).unwrap();
        assert!(gov.round_boundary(0, &mut effects).is_err());
        assert_eq!(gov.live_workers().get(), 2);
        effects.fail_resize = false;
        gov.round_boundary(0, &mut effects).unwrap();
        assert_eq!(gov.live_workers().get(), 1);
        sensors.disk = Ok(99);
        gov.poll(&mut sensors, &mut effects).unwrap();
        effects.fail_checkpoint = true;
        assert!(gov.round_boundary(1, &mut effects).is_err());
        assert!(!effects.admission);
        effects.fail_checkpoint = false;
        assert_eq!(
            gov.round_boundary(1, &mut effects).unwrap(),
            RoundEffect::Paused(ErrorCode::DiskLow)
        );
        assert_eq!(
            effects
                .calls
                .iter()
                .filter(|call| *call == "commit")
                .count(),
            1
        );
    }

    #[test]
    fn unacknowledged_admission_cannot_checkpoint_or_resize() {
        let mut gov = governor(8);
        let mut effects = Effects {
            fail_admission: true,
            ..Default::default()
        };
        let mut sensors = Readings::new(99, 99);
        assert!(gov.round_boundary(0, &mut effects).is_err());
        assert!(gov.poll(&mut sensors, &mut effects).is_err());
        assert!(gov.round_boundary(0, &mut effects).is_err());
        assert!(effects.calls.is_empty());
        effects.fail_admission = false;
        gov.poll(&mut sensors, &mut effects).unwrap();
        assert_eq!(
            gov.round_boundary(0, &mut effects).unwrap(),
            RoundEffect::Paused(ErrorCode::DiskLow)
        );
    }

    /// Seeded controller fixture: actual live thread counts change, not native/model/OS state.
    fn replay(case: &str) -> (serde_json::Value, Vec<u32>) {
        use crate::run::{
            EvaluationOutcome, Evaluator, FakeEvaluator, ResultDisposition, RunController,
        };
        use jope_api::{Effort, RunPhase};
        use sha2::{Digest, Sha256};
        use std::time::Duration;
        let mut gov = governor(8);
        let mut effects = Effects::default();
        let mut run = RunController::new(
            "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
            Effort::Medium,
            gov.planned_r(),
        );
        for phase in [RunPhase::Encoding, RunPhase::Seeding, RunPhase::Searching] {
            run.advance_phase(phase).unwrap();
        }
        let mut sensors = Readings::new(500, 100);
        let mut boundary = 0;
        if case == "one" {
            sensors.memory = Ok(99);
            gov.poll(&mut sensors, &mut effects).unwrap();
            for _ in 0..7 {
                gov.round_boundary(boundary, &mut effects).unwrap();
                boundary += 1;
            }
        }
        let mut worker_counts = Vec::new();
        for first in (0..6_u64).step_by(2) {
            sensors.memory = Ok(match case {
                "one" => 150,
                "eight" => 500,
                _ if first < 4 => 99,
                _ => 201,
            });
            let observed = gov.poll(&mut sensors, &mut effects).unwrap();
            assert!(observed.warnings.iter().all(|warning| warning.host));
            assert!(observed.admission_allowed);
            gov.round_boundary(boundary, &mut effects).unwrap();
            boundary += 1;
            let workers = gov.live_workers().get();
            worker_counts.push(workers);
            assert_eq!(gov.planned_r().get(), 2);
            assert_eq!(run.snapshot().planned_rounds_in_flight, gov.planned_r());
            let mut requests = Vec::new();
            for round in first..first + 2 {
                for slot in 0..8_u32 {
                    let mut hash = Sha256::new();
                    hash.update(73_u64.to_le_bytes());
                    hash.update(round.to_le_bytes());
                    hash.update(slot.to_le_bytes());
                    let seed = u64::from_le_bytes(hash.finalize()[..8].try_into().unwrap());
                    requests.push(run.dispatch(round, slot, seed).unwrap());
                }
            }
            assert!(
                run.dispatch(first + 2, 0, 0).is_err(),
                "live worker count cannot change planned R"
            );
            if first == 2 && matches!(case, "pause" | "recovery") {
                let late = FakeEvaluator::new(Duration::ZERO).evaluate(&requests[0]);
                run.receive(late.clone()).unwrap();
                if case == "pause" {
                    run.control(crate::run::Control::Pause, None).unwrap();
                    run.expire_pause().unwrap();
                } else {
                    assert!(run.recover_interrupted().unwrap());
                }
                run.control(crate::run::Control::Continue, None).unwrap();
                requests = requests
                    .iter()
                    .map(|request| {
                        run.dispatch(request.key.round, request.key.slot, request.seed)
                            .unwrap()
                    })
                    .collect();
                assert_eq!(run.receive(late).unwrap(), ResultDisposition::DiscardedLate);
            }
            // Govern actual concurrent evaluation threads, and reverse result delivery.
            for batch in requests.chunks(workers as usize) {
                let results = std::thread::scope(|scope| {
                    let jobs: Vec<_> = batch
                        .iter()
                        .map(|request| {
                            scope.spawn(move || {
                                FakeEvaluator::new(Duration::from_millis(u64::from(
                                    request.key.slot % 3,
                                )))
                                .evaluate(request)
                            })
                        })
                        .collect();
                    jobs.into_iter()
                        .map(|job| job.join().unwrap())
                        .collect::<Vec<_>>()
                });
                for result in results.into_iter().rev() {
                    run.receive(result).unwrap();
                }
            }
            run.round_committed().unwrap();
            run.round_committed().unwrap();
        }
        let projection: Vec<_> = run.committed().iter().flat_map(|round| {
            round.evaluations.iter().map(|evaluation| {
                let EvaluationOutcome::Scored(value) = evaluation.outcome else { panic!("unexpected fixture failure") };
                serde_json::json!({"round":round.round,"slot":evaluation.key.slot,"effort":round.effort,"fixture_value_bits":format!("{:016x}",value.to_bits())})
            })
        }).collect();
        assert_eq!(projection.len(), 48);
        (serde_json::json!(projection), worker_counts)
    }

    #[test]
    fn s2_controller_replay_preserves_r_under_retirement_restore_pause_and_recovery() {
        let (baseline, workers) = replay("eight");
        assert_eq!(workers, [8, 8, 8]);
        let mut receipts = Vec::new();
        for case in ["one", "retirement", "pause", "recovery"] {
            let (projection, workers) = replay(case);
            assert_eq!(projection, baseline, "{case}");
            assert_eq!(
                workers,
                if case == "one" {
                    vec![1, 1, 1]
                } else {
                    vec![7, 6, 7]
                }
            );
            receipts.push(serde_json::json!({"case":case,"live_threads_by_window":workers,"planned_r":2,"committed_fixture_evaluations":48,"projection":projection}));
        }
        if let Some(path) = std::env::var_os("JOPE_GOVERNOR_PROOF_DIR") {
            let path = std::path::PathBuf::from(path);
            std::fs::create_dir_all(&path).unwrap();
            let proof = serde_json::json!({"kind":"S2 seeded controller fixture replay","native_model_or_os_sigkill_qualification":false,"baseline_threads":[8,8,8],"baseline_projection":baseline,"cases":receipts});
            std::fs::write(
                path.join("controller-s2.json"),
                serde_json::to_vec_pretty(&proof).unwrap(),
            )
            .unwrap();
        }
    }

    #[test]
    fn pressure_seen_mid_round_is_applied_once_even_if_next_sample_is_in_the_band() {
        let mut gov = governor(8);
        let mut effects = Effects::default();
        let mut sensors = Readings::new(99, 100);
        gov.poll(&mut sensors, &mut effects).unwrap();
        sensors.memory = Ok(150);
        gov.poll(&mut sensors, &mut effects).unwrap();
        assert_eq!(
            gov.round_boundary(0, &mut effects).unwrap(),
            RoundEffect::Workers {
                previous: 8,
                current: 7
            }
        );
        assert_eq!(
            gov.round_boundary(1, &mut effects).unwrap(),
            RoundEffect::None
        );
        assert_eq!(gov.live_workers().get(), 7);
    }
}

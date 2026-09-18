use std::time::{Duration, Instant};

pub use jain_smartcluster_core::PressureLevel;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum OverloadStage {
    Normal,
    AdmissionsClamped,
    BurstReduced,
    CpuJobsFrozen,
    CheckpointRequested,
    GpuJobsRequeued,
    HardKillAuthorized,
    Recovering,
}

impl OverloadStage {
    const fn next_warning(self) -> Self {
        match self {
            Self::Normal | Self::Recovering => Self::AdmissionsClamped,
            Self::AdmissionsClamped => Self::BurstReduced,
            Self::BurstReduced => Self::CpuJobsFrozen,
            Self::CpuJobsFrozen => Self::CheckpointRequested,
            Self::CheckpointRequested | Self::GpuJobsRequeued | Self::HardKillAuthorized => self,
        }
    }

    const fn next_critical(self) -> Self {
        match self {
            Self::Normal | Self::Recovering => Self::AdmissionsClamped,
            Self::AdmissionsClamped => Self::BurstReduced,
            Self::BurstReduced => Self::CpuJobsFrozen,
            Self::CpuJobsFrozen => Self::CheckpointRequested,
            Self::CheckpointRequested => Self::GpuJobsRequeued,
            Self::GpuJobsRequeued => Self::HardKillAuthorized,
            Self::HardKillAuthorized => Self::HardKillAuthorized,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OverloadDecision {
    pub previous: OverloadStage,
    pub current: OverloadStage,
    pub changed: bool,
}

#[derive(Debug)]
pub struct OverloadController {
    stage: OverloadStage,
    recovery_quiet: Duration,
    last_pressure_at: Option<Instant>,
}

impl OverloadController {
    pub fn new(recovery_quiet: Duration) -> Self {
        Self {
            stage: OverloadStage::Normal,
            recovery_quiet,
            last_pressure_at: None,
        }
    }

    pub const fn stage(&self) -> OverloadStage {
        self.stage
    }

    pub fn observe(&mut self, pressure: PressureLevel, now: Instant) -> OverloadDecision {
        let previous = self.stage;
        match pressure {
            PressureLevel::Warning => {
                self.last_pressure_at = Some(now);
                self.stage = self.stage.next_warning();
            }
            PressureLevel::Critical => {
                self.last_pressure_at = Some(now);
                self.stage = self.stage.next_critical();
            }
            PressureLevel::Normal if self.stage == OverloadStage::Normal => {
                self.last_pressure_at = None;
            }
            PressureLevel::Normal => {
                if self
                    .last_pressure_at
                    .is_some_and(|last| now.saturating_duration_since(last) >= self.recovery_quiet)
                {
                    self.stage = OverloadStage::Normal;
                    self.last_pressure_at = None;
                }
            }
        }
        OverloadDecision {
            previous,
            current: self.stage,
            changed: previous != self.stage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escalation_order_is_fixed() {
        let now = Instant::now();
        let mut controller = OverloadController::new(Duration::from_secs(2));
        let expected = [
            OverloadStage::AdmissionsClamped,
            OverloadStage::BurstReduced,
            OverloadStage::CpuJobsFrozen,
            OverloadStage::CheckpointRequested,
            OverloadStage::GpuJobsRequeued,
            OverloadStage::HardKillAuthorized,
        ];
        for stage in expected {
            assert_eq!(
                controller.observe(PressureLevel::Critical, now).current,
                stage
            );
        }
    }

    #[test]
    fn recovery_requires_a_quiet_window() {
        let now = Instant::now();
        let mut controller = OverloadController::new(Duration::from_secs(2));
        controller.observe(PressureLevel::Warning, now);
        controller.observe(PressureLevel::Warning, now);
        assert_eq!(
            controller
                .observe(PressureLevel::Normal, now + Duration::from_secs(1))
                .current,
            OverloadStage::BurstReduced
        );
        assert_eq!(
            controller
                .observe(PressureLevel::Normal, now + Duration::from_secs(3))
                .current,
            OverloadStage::Normal
        );
    }
}

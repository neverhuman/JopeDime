use std::{collections::BTreeMap, fmt};

use serde::{Deserialize, Serialize};

use crate::{
    FailureClass, FencedAttempt, FencedTransition, JobId, Timestamp, Validate, ValidationError,
};

pub const MAX_EVENT_CHUNK_BYTES: usize = 1024 * 1024;

/// Monotonic, job-independent durable event-log offset.
#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
#[serde(transparent)]
pub struct EventOffset(u64);

impl EventOffset {
    pub const ZERO: Self = Self(0);

    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }

    pub const fn next(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}

impl From<u64> for EventOffset {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl fmt::Display for EventOffset {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StructuredProgress {
    pub phase: String,
    pub completed: u64,
    pub total: Option<u64>,
    pub message: Option<String>,
    #[serde(default)]
    pub fields: BTreeMap<String, String>,
}

impl Validate for StructuredProgress {
    fn validate(&self) -> Result<(), ValidationError> {
        crate::validation::validate_text("progress.phase", &self.phase, 256, false)?;
        if let Some(total) = self.total {
            if self.completed > total {
                return Err(ValidationError::new(
                    "progress.completed",
                    "must not exceed total",
                ));
            }
        }
        if let Some(message) = &self.message {
            crate::validation::validate_text("progress.message", message, 16 * 1024, true)?;
        }
        if self.fields.len() > 64 {
            return Err(ValidationError::new(
                "progress.fields",
                "must contain at most 64 entries",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CheckpointEvent {
    pub checkpoint_id: String,
    pub phase: String,
    pub durable: bool,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl Validate for CheckpointEvent {
    fn validate(&self) -> Result<(), ValidationError> {
        crate::validation::validate_identifier(
            "checkpoint.checkpoint_id",
            &self.checkpoint_id,
            256,
        )?;
        crate::validation::validate_text("checkpoint.phase", &self.phase, 256, false)?;
        if self.metadata.len() > 64 {
            return Err(ValidationError::new(
                "checkpoint.metadata",
                "must contain at most 64 entries",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GpuMetrics {
    pub device: String,
    pub vram_bytes: u64,
    pub utilization_per_mille: u16,
    pub temperature_millidegrees_celsius: Option<u32>,
    pub power_milliwatts: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct MetricsSnapshot {
    pub cpu_usage_micros: u64,
    pub memory_current_bytes: u64,
    pub memory_peak_bytes: u64,
    pub io_read_bytes: u64,
    pub io_write_bytes: u64,
    #[serde(default)]
    pub gpus: Vec<GpuMetrics>,
}

impl Validate for MetricsSnapshot {
    fn validate(&self) -> Result<(), ValidationError> {
        for (index, gpu) in self.gpus.iter().enumerate() {
            if gpu.utilization_per_mille > 1000 {
                return Err(ValidationError::new(
                    format!("metrics.gpus[{index}].utilization_per_mille"),
                    "must not exceed 1000",
                ));
            }
            crate::validation::validate_text(
                &format!("metrics.gpus[{index}].device"),
                &gpu.device,
                256,
                false,
            )?;
        }
        Ok(())
    }
}

/// Marker emitted when the requested event history has already been compacted.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StreamGap {
    pub requested: EventOffset,
    pub available_from: EventOffset,
    pub latest: EventOffset,
    pub reason: String,
}

impl Validate for StreamGap {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.available_from <= self.requested {
            return Err(ValidationError::new(
                "stream_gap.available_from",
                "must be later than the requested offset",
            ));
        }
        if self.latest < self.available_from {
            return Err(ValidationError::new(
                "stream_gap.latest",
                "must not precede available_from",
            ));
        }
        crate::validation::validate_text("stream_gap.reason", &self.reason, 4096, false)
    }
}

/// Durable payloads. Output chunks are bounded so pipes never depend on subscribers.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobEvent {
    StateTransition(FencedTransition),
    Stdout(Vec<u8>),
    Stderr(Vec<u8>),
    Pty(Vec<u8>),
    Progress(StructuredProgress),
    Checkpoint(CheckpointEvent),
    Metrics(MetricsSnapshot),
    StreamGap(StreamGap),
    Failure {
        class: FailureClass,
        message: String,
    },
}

impl Validate for JobEvent {
    fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Stdout(data) | Self::Stderr(data) | Self::Pty(data) => {
                if data.len() > MAX_EVENT_CHUNK_BYTES {
                    Err(ValidationError::new(
                        "event.payload",
                        format!("must be at most {MAX_EVENT_CHUNK_BYTES} bytes"),
                    ))
                } else {
                    Ok(())
                }
            }
            Self::Progress(progress) => progress.validate(),
            Self::Checkpoint(checkpoint) => checkpoint.validate(),
            Self::Metrics(metrics) => metrics.validate(),
            Self::StateTransition(transition) => transition
                .from
                .validate_transition(transition.to)
                .map_err(|error| ValidationError::new("event.state_transition", error.to_string())),
            Self::StreamGap(gap) => gap.validate().map_err(|error| error.prefixed("event")),
            Self::Failure { message, .. } => {
                crate::validation::validate_text("event.failure.message", message, 4096, false)
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EventRecord {
    pub offset: EventOffset,
    pub job_id: JobId,
    pub attempt: Option<FencedAttempt>,
    pub at: Timestamp,
    pub event: JobEvent,
}

impl Validate for EventRecord {
    fn validate(&self) -> Result<(), ValidationError> {
        if let Some(attempt) = self.attempt {
            if attempt.job_id != self.job_id {
                return Err(ValidationError::new(
                    "event.attempt",
                    "belongs to a different job",
                ));
            }
        }
        if let JobEvent::StateTransition(transition) = &self.event {
            if transition.job_id != self.job_id {
                return Err(ValidationError::new(
                    "event.state_transition.job_id",
                    "does not match the event job",
                ));
            }
            if transition.attempt != self.attempt {
                return Err(ValidationError::new(
                    "event.state_transition.attempt",
                    "does not match the event attempt fence",
                ));
            }
        }
        self.event.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_offsets_advance_without_wrapping() {
        assert_eq!(EventOffset::new(41).next(), Some(EventOffset::new(42)));
        assert_eq!(EventOffset::new(u64::MAX).next(), None);
    }

    #[test]
    fn output_chunks_are_bounded() {
        let event = JobEvent::Stdout(vec![0; MAX_EVENT_CHUNK_BYTES + 1]);
        assert!(event.validate().is_err());
    }

    #[test]
    fn progress_never_exceeds_known_total() {
        let progress = StructuredProgress {
            phase: "train".into(),
            completed: 11,
            total: Some(10),
            message: None,
            fields: BTreeMap::new(),
        };
        assert!(progress.validate().is_err());
    }

    #[test]
    fn stream_gaps_identify_a_resumable_retained_range() {
        let gap = StreamGap {
            requested: EventOffset::new(5),
            available_from: EventOffset::new(17),
            latest: EventOffset::new(42),
            reason: "event segments were compacted".to_owned(),
        };
        assert!(gap.validate().is_ok());

        let invalid = StreamGap {
            available_from: EventOffset::new(5),
            ..gap
        };
        assert!(invalid.validate().is_err());
    }
}

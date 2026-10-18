use serde::{Deserialize, Serialize};

/// Stable event names emitted by the deterministic strategy kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Proposed,
    Duplicate,
    Reserved,
    ReservationRefused,
    Evaluated,
}

/// A causal event. Driver-specific wall-clock telemetry is intentionally not part of this trace.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    pub sequence: u64,
    pub kind: EventKind,
    pub candidate_id: String,
    pub candidate_sha256: String,
    pub origin: Option<String>,
    pub reservation_id: Option<String>,
    pub cost_units: Option<u64>,
    pub outcome: Option<String>,
    pub score_bits: Option<u64>,
}

/// Injectable sink used by product/bench adapters. It is called in causal sequence order.
pub trait Recorder {
    fn record(&mut self, event: Event);
}

/// In-memory, serializable trace for deterministic replay and conformance checks.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trace {
    pub event_version: String,
    pub identity_sha256: String,
    pub events: Vec<Event>,
}

impl Recorder for Trace {
    fn record(&mut self, event: Event) {
        self.events.push(event);
    }
}

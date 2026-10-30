//! Deterministic event replay for terminal and client tests.
//!
//! Each fixture line is decoded as the public durable event type. The first event is available
//! immediately; subsequent events carry the configured delay before delivery. A fake clock or
//! transport can consume that schedule without making replay depend on wall-clock timing.

use std::time::Duration;

use jope_api::PublicEvent;

/// An event together with the delay before a fake transport should deliver it.
#[derive(Clone, Debug, PartialEq)]
pub struct TimedEvent {
    pub delay_before: Duration,
    pub event: PublicEvent,
}

/// A deterministic, timed replay of newline-delimited [`PublicEvent`] JSON.
#[derive(Clone, Debug)]
pub struct FakeTransport {
    events: Vec<PublicEvent>,
    interval: Duration,
    next: usize,
}

impl FakeTransport {
    /// Decode every JSONL line as a public event and schedule events at a fixed cadence.
    ///
    /// The first event has no delay. Each later event has `interval` as its delay from its
    /// predecessor. The iterator returns this schedule instead of sleeping, so tests can replay it
    /// deterministically with either real or simulated time.
    pub fn from_jsonl(jsonl: &str, interval: Duration) -> Result<Self, serde_json::Error> {
        let events = jsonl
            .lines()
            .map(serde_json::from_str::<PublicEvent>)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            events,
            interval,
            next: 0,
        })
    }
}

impl Iterator for FakeTransport {
    type Item = TimedEvent;

    fn next(&mut self) -> Option<Self::Item> {
        let event = self.events.get(self.next)?.clone();
        let delay_before = if self.next == 0 {
            Duration::ZERO
        } else {
            self.interval
        };
        self.next += 1;

        Some(TimedEvent {
            delay_before,
            event,
        })
    }
}

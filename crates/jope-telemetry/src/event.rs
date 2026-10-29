//! Fixed-layout numeric event envelope used by every hot-path producer.

use std::mem::size_of;

pub const EVENT_SCHEMA_VERSION: u8 = 1;

pub mod flags {
    pub const VALUE_PRESENT: u8 = 1 << 0;
    pub const SAMPLED: u8 = 1 << 1;
    pub const NONFINITE: u8 = 1 << 2;
    pub const COUNTER_DELTA: u8 = 1 << 3;
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    Scalar = 1,
    Counter = 2,
    Span = 3,
    Health = 4,
}

/// One event is exactly one 64-byte cache-friendly numeric record.
///
/// String labels, source names, units and metric definitions live in the run manifest. The
/// explicit reserved bytes keep future v1-compatible additions from changing the ring stride.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Event {
    pub run_id: [u8; 16],
    pub sequence: u64,
    pub monotonic_ns: u64,
    pub step: u64,
    pub value: f64,
    pub schema_version: u8,
    pub kind: u8,
    pub flags: u8,
    pub reserved0: u8,
    pub source_id: u16,
    pub metric_id: u16,
    pub group_id: u16,
    pub availability_reason_id: u16,
    pub reserved1: [u8; 4],
}

const _: [(); 64] = [(); size_of::<Event>()];

impl Event {
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn scalar(
        run_id: [u8; 16],
        sequence: u64,
        monotonic_ns: u64,
        source_id: u16,
        metric_id: u16,
        group_id: u16,
        step: u64,
        value: f64,
        sampled: bool,
    ) -> Self {
        let mut event_flags = flags::VALUE_PRESENT;
        if sampled {
            event_flags |= flags::SAMPLED;
        }
        if !value.is_finite() {
            event_flags |= flags::NONFINITE;
        }
        Self {
            run_id,
            sequence,
            monotonic_ns,
            step,
            value,
            schema_version: EVENT_SCHEMA_VERSION,
            kind: EventKind::Scalar as u8,
            flags: event_flags,
            reserved0: 0,
            source_id,
            metric_id,
            group_id,
            availability_reason_id: 0,
            reserved1: [0; 4],
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn unavailable(
        run_id: [u8; 16],
        sequence: u64,
        monotonic_ns: u64,
        source_id: u16,
        metric_id: u16,
        group_id: u16,
        step: u64,
        reason_id: u16,
        sampled: bool,
    ) -> Self {
        let mut event = Self::scalar(
            run_id,
            sequence,
            monotonic_ns,
            source_id,
            metric_id,
            group_id,
            step,
            0.0,
            sampled,
        );
        event.flags &= !flags::VALUE_PRESENT;
        event.availability_reason_id = reason_id;
        event
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_layout_is_fixed_and_nonfinite_is_preserved() {
        assert_eq!(size_of::<Event>(), 64);
        let e = Event::scalar([7; 16], 2, 3, 4, 5, 6, 7, f64::NAN, false);
        assert_eq!(e.flags & flags::NONFINITE, flags::NONFINITE);
        assert_eq!(e.value.to_bits(), f64::NAN.to_bits());
        assert_eq!(e.sequence, 2);
        let missing = Event::unavailable([7; 16], 3, 4, 1, 5, 0, 7, 1, false);
        assert_eq!(missing.flags & flags::VALUE_PRESENT, 0);
        assert_eq!(missing.availability_reason_id, 1);
    }
}

//! Whole-request timing with explicit synchronization boundaries (W11-T03).
//!
//! A request has one enclosing wall time. It runs from a synchronized start to a synchronized
//! finish and includes everything the declared request requires: transfers, encoding and
//! decoding, fitting, emitted compilation and first prediction. Inside it are three separate
//! records:
//!
//! - **Exclusive stages** never overlap: `stage` borrows the timeline mutably, so stages
//!   cannot nest. Each stage is bracketed by synchronization, so device work a stage queued is
//!   charged to that stage and not to the next one.
//! - **Cumulative milestones** record the synchronized time since the start at which a point
//!   was reached.
//! - **Unattributed time** is the whole time minus every stage: gaps between stages.
//!
//! Standalone backend timings (a backend measured on its own) are a different record and are
//! never part of a request timing. The state a request ran in is recorded beside its times:
//! cold or warm model, cache and process, and whether compilation was reused.
use anyhow::{ensure, Result};
use serde::Serialize;
use std::time::{Duration, Instant};

/// A monotonic clock; [`Monotonic`] in production, a controlled clock in tests.
pub trait Clock {
    fn now(&self) -> Duration;
}

/// `std::time::Instant` since construction.
pub struct Monotonic(Instant);

impl Monotonic {
    pub fn start() -> Self {
        Self(Instant::now())
    }
}

impl Clock for Monotonic {
    fn now(&self) -> Duration {
        self.0.elapsed()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Warmth {
    Cold,
    Warm,
}

/// The state a request ran in; times are only comparable between equal states.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RunState {
    pub model: Warmth,
    pub cache: Warmth,
    pub process: Warmth,
    pub compilation_reused: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StageTime {
    pub name: String,
    /// Synchronized offset from the request start.
    pub start_seconds: f64,
    pub seconds: f64,
    pub failed: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Milestone {
    pub name: String,
    pub at_seconds: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RequestTiming {
    pub schema: &'static str,
    pub state: RunState,
    pub whole_seconds: f64,
    pub stages: Vec<StageTime>,
    pub milestones: Vec<Milestone>,
    pub unattributed_seconds: f64,
}

pub const SCHEMA: &str = "jope.request-timing/v1";

pub struct RequestTimeline<'a> {
    clock: &'a dyn Clock,
    sync: &'a dyn Fn() -> Result<()>,
    start: Duration,
    state: RunState,
    stages: Vec<StageTime>,
    milestones: Vec<Milestone>,
}

impl<'a> RequestTimeline<'a> {
    /// Synchronize, then start the enclosing clock.
    pub fn begin(
        state: RunState,
        clock: &'a dyn Clock,
        sync: &'a dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        sync()?;
        Ok(Self {
            clock,
            sync,
            start: clock.now(),
            state,
            stages: Vec::new(),
            milestones: Vec::new(),
        })
    }

    fn since_start(&self) -> f64 {
        (self.clock.now() - self.start).as_secs_f64()
    }

    /// Run `work` as one exclusive stage, synchronized before and after. A failed stage is
    /// still recorded, with its time, before its error is returned.
    pub fn stage<T>(&mut self, name: &str, work: impl FnOnce() -> Result<T>) -> Result<T> {
        ensure!(
            !self.stages.iter().any(|s| s.name == name),
            "stage {name} recorded twice"
        );
        (self.sync)()?;
        let begun = self.since_start();
        let value = work();
        (self.sync)()?;
        let ended = self.since_start();
        self.stages.push(StageTime {
            name: name.into(),
            start_seconds: begun,
            seconds: ended - begun,
            failed: value.is_err(),
        });
        value
    }

    /// Record the synchronized time since the start at which `name` was reached.
    pub fn milestone(&mut self, name: &str) -> Result<()> {
        ensure!(
            !self.milestones.iter().any(|m| m.name == name),
            "milestone {name} recorded twice"
        );
        (self.sync)()?;
        let at_seconds = self.since_start();
        self.milestones.push(Milestone {
            name: name.into(),
            at_seconds,
        });
        Ok(())
    }

    /// Synchronize and stop the enclosing clock.
    pub fn finish(self) -> Result<RequestTiming> {
        (self.sync)()?;
        let whole_seconds = self.since_start();
        let staged = self.stages.iter().map(|s| s.seconds).sum::<f64>();
        let unattributed_seconds = whole_seconds - staged;
        ensure!(
            unattributed_seconds >= -1e-9
                && self
                    .stages
                    .iter()
                    .all(|s| s.start_seconds + s.seconds <= whole_seconds + 1e-9),
            "stages exceed the enclosing request time"
        );
        Ok(RequestTiming {
            schema: SCHEMA,
            state: self.state,
            whole_seconds,
            stages: self.stages,
            milestones: self.milestones,
            unattributed_seconds: unattributed_seconds.max(0.0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// A host clock plus a device queue. Host work advances the clock at once; device work
    /// is only queued, and advances the clock when a synchronization drains it (like CUDA).
    #[derive(Default)]
    struct Device {
        clock: Cell<Duration>,
        queued: Cell<Duration>,
    }

    impl Device {
        fn host(&self, ms: u64) {
            self.clock.set(self.clock.get() + Duration::from_millis(ms));
        }
        fn launch(&self, ms: u64) {
            self.queued
                .set(self.queued.get() + Duration::from_millis(ms));
        }
        fn synchronize(&self) -> Result<()> {
            self.clock.set(self.clock.get() + self.queued.take());
            Ok(())
        }
    }

    impl Clock for Device {
        fn now(&self) -> Duration {
            self.clock.get()
        }
    }

    fn cold() -> RunState {
        RunState {
            model: Warmth::Cold,
            cache: Warmth::Cold,
            process: Warmth::Cold,
            compilation_reused: false,
        }
    }

    fn seconds(timing: &RequestTiming, name: &str) -> f64 {
        timing
            .stages
            .iter()
            .find(|s| s.name == name)
            .unwrap()
            .seconds
    }

    fn close(actual: f64, expected_ms: u64) {
        assert!(
            (actual - expected_ms as f64 / 1000.0).abs() < 1e-9,
            "{actual} vs {expected_ms} ms"
        );
    }

    /// The acceptance fixture: delays injected into transfer, fit, compilation and prediction
    /// (host and queued device work) appear in their own stage and in the whole request, never
    /// in another stage; the gap between stages is unattributed.
    #[test]
    fn injected_delays_land_in_their_enclosing_stage_and_the_whole_request() {
        let device = Device::default();
        let sync = || device.synchronize();
        let mut timeline = RequestTimeline::begin(cold(), &device, &sync).unwrap();
        timeline
            .stage("transfer", || {
                device.host(5);
                device.launch(30);
                Ok(())
            })
            .unwrap();
        timeline
            .stage("fit", || {
                device.host(60);
                Ok(())
            })
            .unwrap();
        timeline.milestone("fitted").unwrap();
        device.host(20); // between stages
        timeline
            .stage("compile", || {
                device.host(40);
                Ok(())
            })
            .unwrap();
        timeline
            .stage("first_prediction", || {
                device.launch(10);
                Ok(())
            })
            .unwrap();
        let timing = timeline.finish().unwrap();
        close(seconds(&timing, "transfer"), 35);
        close(seconds(&timing, "fit"), 60);
        close(seconds(&timing, "compile"), 40);
        close(seconds(&timing, "first_prediction"), 10);
        close(timing.unattributed_seconds, 20);
        close(timing.whole_seconds, 165);
        close(timing.milestones[0].at_seconds, 95);
        close(timing.stages[2].start_seconds, 115);
        assert_eq!(timing.schema, SCHEMA);
    }

    /// Positive control for the injection: without stage synchronization, device work queued
    /// by one stage is charged to whichever later synchronization drains it.
    #[test]
    fn unsynchronized_stages_misattribute_queued_device_work() {
        let device = Device::default();
        let no_sync = || Ok(());
        let mut timeline = RequestTimeline::begin(cold(), &device, &no_sync).unwrap();
        timeline
            .stage("transfer", || {
                device.host(5);
                device.launch(30);
                Ok(())
            })
            .unwrap();
        timeline
            .stage("fit", || {
                device.synchronize()?; // the fit's first blocking call drains the transfer
                device.host(60);
                Ok(())
            })
            .unwrap();
        let timing = timeline.finish().unwrap();
        close(seconds(&timing, "transfer"), 5);
        close(seconds(&timing, "fit"), 90);
    }

    #[test]
    fn failed_stages_are_recorded_and_names_are_unique() {
        let device = Device::default();
        let sync = || device.synchronize();
        let mut timeline = RequestTimeline::begin(cold(), &device, &sync).unwrap();
        let failed: Result<()> = timeline.stage("compile", || {
            device.host(7);
            anyhow::bail!("compiler refused")
        });
        assert!(failed.is_err());
        assert!(timeline.stage("compile", || Ok(())).is_err());
        timeline.milestone("refused").unwrap();
        assert!(timeline.milestone("refused").is_err());
        let timing = timeline.finish().unwrap();
        assert_eq!(timing.stages.len(), 1);
        assert!(timing.stages[0].failed);
        close(timing.stages[0].seconds, 7);
        close(timing.whole_seconds, 7);
    }

    #[test]
    fn warm_and_cold_states_are_part_of_the_record() {
        let device = Device::default();
        let sync = || device.synchronize();
        let warm = RunState {
            model: Warmth::Warm,
            cache: Warmth::Warm,
            process: Warmth::Warm,
            compilation_reused: true,
        };
        let timing = RequestTimeline::begin(warm.clone(), &device, &sync)
            .unwrap()
            .finish()
            .unwrap();
        assert_eq!(timing.state, warm);
        let json = serde_json::to_value(&timing).unwrap();
        assert_eq!(json["state"]["model"], "warm");
        assert_eq!(json["state"]["compilation_reused"], true);
        assert_ne!(timing.state, cold());
    }

    /// The production clock is monotonic and synchronization is called at every boundary.
    #[test]
    fn production_clock_brackets_every_boundary_with_synchronization() {
        let clock = Monotonic::start();
        let calls = Cell::new(0);
        let sync = || {
            calls.set(calls.get() + 1);
            Ok(())
        };
        let mut timeline = RequestTimeline::begin(cold(), &clock, &sync).unwrap();
        timeline.stage("encode", || Ok(())).unwrap();
        timeline.milestone("encoded").unwrap();
        let timing = timeline.finish().unwrap();
        assert_eq!(calls.get(), 5); // begin, stage before/after, milestone, finish
        assert!(timing.whole_seconds >= timing.stages[0].seconds);
    }
}

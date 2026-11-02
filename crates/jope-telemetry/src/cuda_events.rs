//! CUDA copy-event orchestration for one H2D source.
//!
//! Torch-specific event creation and stream access stay behind [`CudaEventBackend`]. The
//! recorder owns sampling, slots, dropped-sample counts and emitted telemetry values.

/// Operations supplied by the telemetry recorder for one H2D source.
pub trait H2dRecorder {
    fn h2d_enabled(&self, source: usize) -> bool;
    fn h2d_unavailable(&mut self, source: usize);
    fn h2d_will_sample(&self, source: usize, step: u64) -> bool;
    fn h2d_pending(&self, source: usize) -> Vec<usize>;
    fn h2d_complete(&mut self, source: usize, slot: usize, elapsed_ms: f64);
    fn h2d_begin(
        &mut self,
        source: usize,
        step: u64,
        size_bytes: u64,
        data_wait_ms: Option<f64>,
    ) -> i64;
    fn h2d_fail(&mut self, source: usize, slot: Option<usize>);
    fn h2d_recorded(&mut self, source: usize, slot: usize);
    fn h2d_abandon(&mut self, source: usize);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaEventError;

impl std::fmt::Display for CudaEventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CUDA event operation failed")
    }
}

impl std::error::Error for CudaEventError {}

/// CUDA event operations bound to the trainer's current stream.
pub trait CudaEventBackend {
    /// Create `slots` start/end event pairs for `device` and bind the trainer stream.
    fn initialize(&mut self, device: &str, slots: usize) -> Result<(), CudaEventError>;
    fn record_start(&mut self, slot: usize) -> Result<(), CudaEventError>;
    fn record_end(&mut self, slot: usize) -> Result<(), CudaEventError>;
    fn end_query(&mut self, slot: usize) -> Result<bool, CudaEventError>;
    fn elapsed_time_ms(&mut self, slot: usize) -> Result<f64, CudaEventError>;
    fn synchronize(&mut self) -> Result<(), CudaEventError>;
}

/// Records asynchronous host-to-device copy timings for one recorder source.
pub struct CudaH2d<'a, R, C> {
    recorder: &'a mut R,
    backend: &'a mut C,
    source: usize,
}

impl<'a, R: H2dRecorder, C: CudaEventBackend> CudaH2d<'a, R, C> {
    /// Initializes event pairs only when the recorder has enabled this source.
    pub fn new(
        backend: &'a mut C,
        device: &str,
        recorder: &'a mut R,
        source: usize,
        slots: usize,
    ) -> Self {
        if recorder.h2d_enabled(source) && backend.initialize(device, slots).is_err() {
            recorder.h2d_unavailable(source);
        }
        Self {
            recorder,
            backend,
            source,
        }
    }

    pub fn will_sample(&self, step: u64) -> bool {
        self.recorder.h2d_will_sample(self.source, step)
    }

    fn poll(&mut self) {
        for slot in self.recorder.h2d_pending(self.source) {
            let result = match self.backend.end_query(slot) {
                Ok(true) => self.backend.elapsed_time_ms(slot).map(Some),
                Ok(false) => Ok(None),
                Err(error) => Err(error),
            };
            match result {
                Ok(Some(elapsed_ms)) => {
                    self.recorder.h2d_complete(self.source, slot, elapsed_ms);
                }
                Ok(None) => {}
                Err(_) => self.recorder.h2d_fail(self.source, Some(slot)),
            }
        }
    }

    pub fn begin(&mut self, step: u64, size_bytes: u64, data_wait_ms: Option<f64>) -> i64 {
        if !self.recorder.h2d_enabled(self.source) {
            return -1;
        }
        self.poll();
        let slot = self
            .recorder
            .h2d_begin(self.source, step, size_bytes, data_wait_ms);
        if slot >= 0 && self.backend.record_start(slot as usize).is_err() {
            self.recorder.h2d_fail(self.source, None);
            return -1;
        }
        slot
    }

    pub fn end(&mut self, slot: i64) {
        if slot >= 0 {
            let slot = slot as usize;
            if self.backend.record_end(slot).is_ok() {
                self.recorder.h2d_recorded(self.source, slot);
            } else {
                self.recorder.h2d_fail(self.source, Some(slot));
            }
        }
    }

    /// At shutdown, optionally synchronize once so outstanding samples become readable.
    pub fn flush(&mut self, synchronize: bool) -> bool {
        let sync_attempted = synchronize && self.recorder.h2d_enabled(self.source);
        if sync_attempted && self.backend.synchronize().is_err() {
            self.recorder.h2d_abandon(self.source);
        }
        if self.recorder.h2d_enabled(self.source) {
            self.poll();
        }
        sync_attempted
    }
}

#[cfg(test)]
mod tests {
    use super::{CudaEventBackend, CudaEventError, CudaH2d, H2dRecorder};

    #[derive(Default)]
    struct Recorder {
        enabled: bool,
        unavailable: usize,
        begins: usize,
        completed: Vec<(usize, f64)>,
        failed: Vec<Option<usize>>,
        recorded: Vec<usize>,
        abandoned: usize,
        pending: Vec<usize>,
    }

    impl H2dRecorder for Recorder {
        fn h2d_enabled(&self, _: usize) -> bool {
            self.enabled
        }
        fn h2d_unavailable(&mut self, _: usize) {
            self.enabled = false;
            self.unavailable += 1;
        }
        fn h2d_will_sample(&self, _: usize, step: u64) -> bool {
            step.is_multiple_of(2)
        }
        fn h2d_pending(&self, _: usize) -> Vec<usize> {
            self.pending.clone()
        }
        fn h2d_complete(&mut self, _: usize, slot: usize, elapsed_ms: f64) {
            self.completed.push((slot, elapsed_ms));
            self.pending.retain(|pending| *pending != slot);
        }
        fn h2d_begin(&mut self, _: usize, _: u64, _: u64, _: Option<f64>) -> i64 {
            self.begins += 1;
            0
        }
        fn h2d_fail(&mut self, _: usize, slot: Option<usize>) {
            self.failed.push(slot);
        }
        fn h2d_recorded(&mut self, _: usize, slot: usize) {
            self.recorded.push(slot);
            self.pending.push(slot);
        }
        fn h2d_abandon(&mut self, _: usize) {
            self.abandoned += 1;
        }
    }

    #[derive(Default)]
    struct Backend {
        initialized: bool,
        initialize_fails: bool,
        query_ready: bool,
        elapsed_ms: f64,
        record_fails: bool,
        sync_fails: bool,
        syncs: usize,
    }

    impl CudaEventBackend for Backend {
        fn initialize(&mut self, _: &str, _: usize) -> Result<(), CudaEventError> {
            if self.initialize_fails {
                Err(CudaEventError)
            } else {
                self.initialized = true;
                Ok(())
            }
        }
        fn record_start(&mut self, _: usize) -> Result<(), CudaEventError> {
            if self.record_fails {
                Err(CudaEventError)
            } else {
                Ok(())
            }
        }
        fn record_end(&mut self, _: usize) -> Result<(), CudaEventError> {
            if self.record_fails {
                Err(CudaEventError)
            } else {
                Ok(())
            }
        }
        fn end_query(&mut self, _: usize) -> Result<bool, CudaEventError> {
            Ok(self.query_ready)
        }
        fn elapsed_time_ms(&mut self, _: usize) -> Result<f64, CudaEventError> {
            Ok(self.elapsed_ms)
        }
        fn synchronize(&mut self) -> Result<(), CudaEventError> {
            self.syncs += 1;
            if self.sync_fails {
                Err(CudaEventError)
            } else {
                Ok(())
            }
        }
    }

    // Mirrors the CPU-only smoke path: disabled recorders allocate no CUDA events and
    // begin/flush do not attempt synchronization.
    #[test]
    fn cpu_only_round_does_not_initialize_or_synchronize_cuda() {
        let mut recorder = Recorder::default();
        let mut backend = Backend::default();
        {
            let mut events = CudaH2d::new(&mut backend, "cpu", &mut recorder, 1, 4);
            assert!(!events.will_sample(1));
            assert!(events.will_sample(2));
            assert_eq!(events.begin(1, 100, Some(1.5)), -1);
            assert!(!events.flush(true));
        }
        assert!(!backend.initialized);
        assert_eq!(backend.syncs, 0);
    }

    #[test]
    fn records_and_completes_ready_copy_events() {
        let mut recorder = Recorder {
            enabled: true,
            ..Recorder::default()
        };
        let mut backend = Backend {
            query_ready: true,
            elapsed_ms: 2.5,
            ..Backend::default()
        };
        {
            let mut events = CudaH2d::new(&mut backend, "cuda:0", &mut recorder, 1, 4);
            assert_eq!(events.begin(2, 2048, Some(0.5)), 0);
            events.end(0);
            events.flush(false);
        }
        assert!(backend.initialized);
        assert_eq!(recorder.recorded, vec![0]);
        assert_eq!(recorder.completed, vec![(0, 2.5)]);
        assert!(recorder.failed.is_empty());
    }

    #[test]
    fn initialization_and_stream_failures_are_reported_to_recorder() {
        let mut recorder = Recorder {
            enabled: true,
            ..Recorder::default()
        };
        let mut backend = Backend {
            initialize_fails: true,
            ..Backend::default()
        };
        {
            let _events = CudaH2d::new(&mut backend, "cuda:0", &mut recorder, 1, 4);
        }
        assert!(!recorder.enabled);
        assert_eq!(recorder.unavailable, 1);

        let mut recorder = Recorder {
            enabled: true,
            ..Recorder::default()
        };
        let mut backend = Backend {
            record_fails: true,
            ..Backend::default()
        };
        {
            let mut events = CudaH2d::new(&mut backend, "cuda:0", &mut recorder, 1, 4);
            assert_eq!(events.begin(2, 10, None), -1);
        }
        assert_eq!(recorder.failed, vec![None]);
    }

    #[test]
    fn failed_shutdown_sync_abandons_pending_samples() {
        let mut recorder = Recorder {
            enabled: true,
            pending: vec![0],
            ..Recorder::default()
        };
        let mut backend = Backend {
            sync_fails: true,
            ..Backend::default()
        };
        {
            let mut events = CudaH2d::new(&mut backend, "cuda:0", &mut recorder, 1, 4);
            assert!(events.flush(true));
        }
        assert_eq!(recorder.abandoned, 1);
        assert_eq!(backend.syncs, 1);
    }
}

//! Live search progress for `jope pipeline invent` (`jope.invent.progress.v1`).
//!
//! Best-effort compact JSON lines on stderr, for a parent process to show search progress.
//! Only fixed schema/phase labels and numbers are included: no program source, genomes,
//! column names or raw cell values. The stdout schema/channel (`jope.candidates.v1`) is unchanged.
//!
//! Search threads enqueue without waiting for stderr. A process-wide worker owns a duplicate
//! stderr descriptor/handle, so a blocked pipe does not hold Rust's global stderr lock. The queue
//! holds at most 64 lines of at most 1024 bytes, plus the worker's current line. Full or disconnected
//! queues drop updates. The worker is never joined: even the final `done` update can be lost at
//! exit, and sequence gaps are allowed. Neither record atomicity nor delivery is guaranteed.
//!
//! Updates are attempted at generation/comparison boundaries: at most once per [`MIN_INTERVAL`]
//! unless a new best was observed, a cut ends, or the search finishes. Multiple improvements
//! within one exploration generation are coalesced. See `docs/pipeline.md`.

use crate::search::Observation;
use serde::Serialize;
use std::{
    io::Write,
    sync::{
        mpsc::{self, Receiver, SyncSender},
        Mutex, OnceLock,
    },
    time::{Duration, Instant},
};

pub const SCHEMA: &str = "jope.invent.progress.v1";

/// Rate limit for lines that report neither an improvement nor the end of a cut.
pub const MIN_INTERVAL: Duration = Duration::from_secs(1);

/// The search phase a line was emitted in.
pub const EXPLORATION: &str = "exploration";
pub const COMPARISON: &str = "comparison";
pub const DONE: &str = "done";

/// One `jope.invent.progress.v1` line. Field order is the serialized order.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct Line {
    pub schema: String,
    /// Per-run line counter, from 0 (the `sequence` of the planned `jope.run-event/v1`).
    pub sequence: u64,
    pub phase: String,
    /// 0-based position of the current cut among the searched cuts.
    pub cut: usize,
    pub cuts: usize,
    /// Generations completed so far, cumulative across cuts.
    pub generation: u64,
    /// Candidate evaluations so far (exploration and comparison), cumulative across cuts.
    pub evaluations: u64,
    /// Evaluation budget of the whole search: the per-cut budget times `cuts` (0 if none).
    pub eval_budget: u64,
    pub measured: u64,
    pub failed: u64,
    /// Lowest held-out RMSE of any measured evaluation so far, in target units.
    pub best_hold_rmse: Option<f64>,
    /// Held-out R^2 of that same evaluation.
    pub best_hold_r2: Option<f64>,
    /// Wall seconds since the search started.
    pub elapsed_s: f64,
    /// Active-seconds budget of the whole search: the per-cut bound times `cuts` (0 if none).
    pub seconds_budget: f64,
}

const QUEUE_CAPACITY: usize = 64;
const MAX_LINE_BYTES: usize = 1024;
static STDERR_QUEUE: OnceLock<Option<SyncSender<Vec<u8>>>> = OnceLock::new();

#[cfg(unix)]
fn duplicate_stderr() -> std::io::Result<std::fs::File> {
    use std::os::fd::AsFd;
    std::io::stderr()
        .as_fd()
        .try_clone_to_owned()
        .map(std::fs::File::from)
}

#[cfg(windows)]
fn duplicate_stderr() -> std::io::Result<std::fs::File> {
    use std::os::windows::io::AsHandle;
    std::io::stderr()
        .as_handle()
        .try_clone_to_owned()
        .map(std::fs::File::from)
}

#[cfg(not(any(unix, windows)))]
fn duplicate_stderr() -> std::io::Result<std::fs::File> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "stderr duplication unavailable",
    ))
}

fn drain(mut sink: impl Write, receiver: Receiver<Vec<u8>>) {
    for bytes in receiver {
        // A write_all may perform multiple writes; other diagnostics can interleave.
        // Stop permanently on returned I/O errors. A blocking write occupies only this worker.
        if sink.write_all(&bytes).and_then(|()| sink.flush()).is_err() {
            break;
        }
    }
}

fn stderr_queue() -> Option<SyncSender<Vec<u8>>> {
    STDERR_QUEUE
        .get_or_init(|| {
            let sink = duplicate_stderr().ok()?;
            let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
            std::thread::Builder::new()
                .name("jope-invent-progress".into())
                .spawn(move || drain(sink, receiver))
                .ok()?;
            Some(sender)
        })
        .clone()
}

struct State {
    sender: Option<SyncSender<Vec<u8>>>,
    started: Instant,
    last_emit: Option<Instant>,
    /// `(phase, cut, generation, evaluations)` of the last line, so a cut's end does not repeat
    /// the generation boundary line it coincides with.
    last_key: Option<(&'static str, usize, u64, u64)>,
    sequence: u64,
    phase: &'static str,
    cut: usize,
    cuts: usize,
    /// Generations completed in cuts before the current one.
    generations_before: u64,
    /// Generations completed in the current cut.
    generations_in_cut: u64,
    evaluations: u64,
    eval_budget: u64,
    measured: u64,
    failed: u64,
    best: Option<(f64, Option<f64>)>,
    improved: bool,
    seconds_budget: f64,
}

/// Collects a search's counters and writes rate-limited progress lines to its sink.
pub struct Progress {
    state: Mutex<State>,
}

impl std::fmt::Debug for Progress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Progress").finish_non_exhaustive()
    }
}

impl Progress {
    // Inject a bounded queue in tests; the production stderr reporter shares one process worker.
    fn with_sender(
        sender: Option<SyncSender<Vec<u8>>>,
        cuts: usize,
        budget: crate::search::Budget,
        started: Instant,
    ) -> Self {
        Progress {
            state: Mutex::new(State {
                sender,
                started,
                last_emit: None,
                last_key: None,
                sequence: 0,
                phase: EXPLORATION,
                cut: 0,
                cuts,
                generations_before: 0,
                generations_in_cut: 0,
                evaluations: 0,
                eval_budget: (budget.evaluations as u64).saturating_mul(cuts as u64),
                measured: 0,
                failed: 0,
                best: None,
                improved: false,
                seconds_budget: budget.seconds as f64 * cuts as f64,
            }),
        }
    }

    /// A best-effort stderr reporter, started now. Queue/worker failure disables delivery.
    pub fn stderr(cuts: usize, budget: crate::search::Budget) -> Self {
        Self::with_sender(stderr_queue(), cuts, budget, Instant::now())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The search moved to the cut at `position` (0-based) of its exploration phase.
    pub fn begin_cut(&self, position: usize) {
        let mut state = self.lock();
        state.generations_before += state.generations_in_cut;
        state.generations_in_cut = 0;
        state.cut = position;
        state.phase = EXPLORATION;
    }

    /// The cross-cut comparison phase started.
    pub fn begin_comparison(&self) {
        let mut state = self.lock();
        state.phase = COMPARISON;
    }

    /// Count one finished exploration evaluation on the current cut.
    pub fn observe_current(&self, observation: &Observation) {
        let mut state = self.lock();
        Self::count(&mut state, observation);
    }

    /// Count one finished evaluation, made on the cut at `position`.
    pub fn observe(&self, position: usize, observation: &Observation) {
        let mut state = self.lock();
        state.cut = position;
        Self::count(&mut state, observation);
    }

    fn count(state: &mut State, observation: &Observation) {
        state.evaluations += 1;
        if !observation.measured() {
            state.failed += 1;
            return;
        }
        state.measured += 1;
        if let Some(rmse) = observation.rmse.filter(|r| r.is_finite()) {
            if state.best.is_none_or(|(best, _)| rmse < best) {
                state.best = Some((rmse, observation.hold_r2));
                state.improved = true;
            }
        }
    }

    /// The current cut has completed `generations` generations; emit if due.
    pub fn generation_done(&self, generations: u64) {
        self.generation_done_at(generations, Instant::now());
    }

    fn generation_done_at(&self, generations: u64, now: Instant) {
        let mut state = self.lock();
        state.generations_in_cut = generations;
        Self::emit_if_due(&mut state, now);
    }

    /// A rate-limited (or improvement-driven) line between generation boundaries.
    pub fn tick(&self) {
        self.tick_at(Instant::now());
    }

    fn tick_at(&self, now: Instant) {
        Self::emit_if_due(&mut self.lock(), now);
    }

    /// The current cut's exploration ended after `generations` generations: emit its end state
    /// unless the last line already reported exactly that.
    pub fn cut_done(&self, generations: u64) {
        self.cut_done_at(generations, Instant::now());
    }

    fn cut_done_at(&self, generations: u64, now: Instant) {
        let mut state = self.lock();
        state.generations_in_cut = generations;
        if state.last_key != Some(Self::key(&state)) {
            Self::emit(&mut state, now);
        }
    }

    /// The search finished: attempt to enqueue its end state (phase `done`), without waiting.
    pub fn finish(&self) {
        let mut state = self.lock();
        state.phase = DONE;
        state.cut = state.cuts.saturating_sub(1);
        Self::emit(&mut state, Instant::now());
    }

    fn key(state: &State) -> (&'static str, usize, u64, u64) {
        (
            state.phase,
            state.cut,
            state.generations_before + state.generations_in_cut,
            state.evaluations,
        )
    }

    fn emit_if_due(state: &mut State, now: Instant) {
        let due = state.improved
            || state
                .last_emit
                .is_none_or(|last| now.saturating_duration_since(last) >= MIN_INTERVAL);
        if due {
            Self::emit(state, now);
        }
    }

    fn emit(state: &mut State, now: Instant) {
        let line = Line {
            schema: SCHEMA.to_owned(),
            sequence: state.sequence,
            phase: state.phase.to_owned(),
            cut: state.cut,
            cuts: state.cuts,
            generation: state.generations_before + state.generations_in_cut,
            evaluations: state.evaluations,
            eval_budget: state.eval_budget,
            measured: state.measured,
            failed: state.failed,
            best_hold_rmse: state.best.map(|(rmse, _)| rmse),
            best_hold_r2: state.best.and_then(|(_, r2)| r2.filter(|r| r.is_finite())),
            elapsed_s: now.saturating_duration_since(state.started).as_secs_f64(),
            seconds_budget: state.seconds_budget,
        };
        let Ok(mut bytes) = serde_json::to_vec(&line) else {
            return;
        };
        bytes.push(b'\n');
        if bytes.len() <= MAX_LINE_BYTES {
            if let Some(sender) = &state.sender {
                // Never wait for an unresponsive parent or a full/disconnected queue.
                let _ = sender.try_send(bytes);
            }
        }
        state.sequence += 1;
        state.last_key = Some(Self::key(state));
        state.last_emit = Some(now);
        state.improved = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, sync::Arc};

    struct Shared {
        receiver: Receiver<Vec<u8>>,
        history: RefCell<Vec<Line>>,
    }

    impl Shared {
        fn lines(&self) -> Vec<Line> {
            let mut history = self.history.borrow_mut();
            for bytes in self.receiver.try_iter() {
                assert!(bytes.len() <= MAX_LINE_BYTES);
                history.push(serde_json::from_slice(&bytes).unwrap());
            }
            history.clone()
        }
    }

    fn observation(rmse: Option<f64>) -> Observation {
        Observation {
            genome: "(pipe X (ridge 1.0))".into(),
            program_sha256: String::new(),
            cut: 0,
            generation: 0,
            origin: "random",
            train_r2: rmse.map(|_| 0.5),
            hold_r2: rmse.map(|r| 1.0 - r),
            rmse,
            seconds: 0.0,
            status: if rmse.is_some() {
                crate::search::MEASURED
            } else {
                "fit_failed"
            },
            failure: rmse.is_none().then(|| "boom".into()),
            island: None,
        }
    }

    fn reporter(t0: Instant) -> (Progress, Shared) {
        let (sender, receiver) = mpsc::sync_channel(QUEUE_CAPACITY);
        let sink = Shared {
            receiver,
            history: RefCell::new(Vec::new()),
        };
        let budget = crate::search::Budget {
            evaluations: 10,
            seconds: 30,
        };
        (Progress::with_sender(Some(sender), 2, budget, t0), sink)
    }

    #[test]
    fn lines_are_rate_limited_except_improvements_and_cut_ends() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let (progress, sink) = reporter(t0);
        progress.begin_cut(0);
        // The first boundary always reports.
        progress.observe(0, &observation(None));
        progress.generation_done_at(1, at(10));
        assert_eq!(sink.lines().len(), 1);
        // Within the second and no improvement: suppressed.
        progress.observe(0, &observation(None));
        progress.generation_done_at(2, at(400));
        progress.tick_at(at(900));
        assert_eq!(sink.lines().len(), 1);
        // An improvement reports immediately, inside the interval.
        progress.observe(0, &observation(Some(0.5)));
        progress.generation_done_at(3, at(950));
        assert_eq!(sink.lines().len(), 2);
        // A worse measurement inside the interval is suppressed; after it, it reports.
        progress.observe(0, &observation(Some(0.9)));
        progress.generation_done_at(4, at(1500));
        assert_eq!(sink.lines().len(), 2);
        progress.generation_done_at(5, at(1950));
        assert_eq!(sink.lines().len(), 3);
        // The end of a cut reports, unless the line just written already is its end state.
        progress.cut_done_at(5, at(1960));
        assert_eq!(sink.lines().len(), 3);
        progress.begin_cut(1);
        progress.observe(1, &observation(Some(0.7)));
        progress.generation_done_at(1, at(2000));
        assert_eq!(sink.lines().len(), 3);
        progress.cut_done_at(1, at(2010));
        assert_eq!(sink.lines().len(), 4);
        progress.finish();

        let lines = sink.lines();
        assert_eq!(lines.len(), 5);
        assert!(lines.iter().all(|l| l.schema == SCHEMA && l.cuts == 2));
        assert!(lines
            .windows(2)
            .all(|w| w[0].evaluations <= w[1].evaluations
                && w[0].generation <= w[1].generation
                && w[1].sequence == w[0].sequence + 1));
        assert_eq!(lines[0].best_hold_rmse, None);
        assert_eq!(lines[1].best_hold_rmse, Some(0.5));
        assert_eq!(lines[1].best_hold_r2, Some(0.5));
        let last = lines.last().unwrap();
        assert_eq!(last.phase, DONE);
        assert_eq!((last.cut, last.generation), (1, 6));
        assert_eq!((last.evaluations, last.measured, last.failed), (5, 3, 2));
        assert_eq!(last.best_hold_rmse, Some(0.5));
        assert_eq!((last.eval_budget, last.seconds_budget), (20, 60.0));
    }

    #[test]
    fn full_and_disconnected_queues_do_not_wait_or_lose_counter_state() {
        let budget = crate::search::Budget {
            evaluations: 10,
            seconds: 30,
        };
        let (sender, receiver) = mpsc::sync_channel(1);
        let progress = Progress::with_sender(Some(sender), 2, budget, Instant::now());
        progress.observe(0, &observation(Some(0.5)));
        progress.tick(); // fills queue
        progress.observe(0, &observation(Some(0.4)));
        progress.finish(); // full queue: drops done, without changing evaluation counters
        let first: Line = serde_json::from_slice(&receiver.try_recv().unwrap()).unwrap();
        assert_eq!((first.sequence, first.evaluations), (0, 1));
        progress.observe(1, &observation(None));
        progress.finish();
        let latest: Line = serde_json::from_slice(&receiver.try_recv().unwrap()).unwrap();
        assert_eq!(
            (
                latest.sequence,
                latest.evaluations,
                latest.measured,
                latest.failed
            ),
            (2, 3, 2, 1)
        );
        assert_eq!(latest.phase, DONE);
        assert_eq!(latest.best_hold_rmse, Some(0.4));
        drop(receiver);
        progress.finish(); // disconnected queue: also returns
        assert_eq!(progress.lock().sequence, 4);
        assert_eq!(progress.lock().evaluations, 3);
    }

    #[test]
    fn worker_drains_records_and_stops_on_write_error() {
        struct Sink(Arc<Mutex<Vec<u8>>>);
        impl Write for Sink {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                // Deliberately accept partial writes: drain must finish the record.
                let count = bytes.len().min(3);
                self.0.lock().unwrap().extend_from_slice(&bytes[..count]);
                Ok(count)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let (sender, receiver) = mpsc::sync_channel(2);
        sender.try_send(b"first\n".to_vec()).unwrap();
        sender.try_send(b"second\n".to_vec()).unwrap();
        drop(sender);
        drain(Sink(bytes.clone()), receiver);
        assert_eq!(*bytes.lock().unwrap(), b"first\nsecond\n");

        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                panic!("failed write must not flush")
            }
        }
        let (sender, receiver) = mpsc::sync_channel(1);
        sender.try_send(vec![0]).unwrap();
        drain(Broken, receiver);
        assert!(matches!(
            sender.try_send(vec![1]),
            Err(mpsc::TrySendError::Disconnected(_))
        ));
    }

    #[test]
    fn blocked_worker_cannot_hold_the_search_or_grow_the_queue() {
        struct Blocked {
            entered: mpsc::Sender<()>,
            release: Receiver<()>,
        }
        impl Write for Blocked {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.entered.send(()).unwrap();
                self.release.recv().unwrap();
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let (entered, wait_entered) = mpsc::channel();
        let (release, wait_release) = mpsc::channel();
        let (sender, receiver) = mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            drain(
                Blocked {
                    entered,
                    release: wait_release,
                },
                receiver,
            )
        });
        sender.try_send(b"blocked\n".to_vec()).unwrap();
        wait_entered.recv_timeout(Duration::from_secs(2)).unwrap();
        let (complete, wait_complete) = mpsc::channel();
        let search = std::thread::spawn(move || {
            let progress = Progress::with_sender(
                Some(sender),
                2,
                crate::search::Budget {
                    evaluations: 1000,
                    seconds: 30,
                },
                Instant::now(),
            );
            for _ in 0..1000 {
                progress.observe(0, &observation(None));
                progress.finish();
            }
            let summary = {
                let state = progress.lock();
                (state.evaluations, state.sequence)
            };
            complete.send(summary).unwrap();
        });
        // This synchronizes on actual completion while the writer remains blocked.
        let result = wait_complete.recv_timeout(Duration::from_secs(2));
        // Release first write and queued record even if the no-wait assertion fails.
        release.send(()).unwrap();
        release.send(()).unwrap();
        assert_eq!(result.unwrap(), (1000, 1000));
        search.join().unwrap();
        worker.join().unwrap();
    }

    #[test]
    fn nonfinite_scores_never_cross_the_progress_json_boundary() {
        let (progress, sink) = reporter(Instant::now());
        progress.observe(0, &observation(Some(f64::NAN)));
        progress.tick();
        let mut measured = observation(Some(0.5));
        measured.hold_r2 = Some(f64::INFINITY);
        progress.observe(0, &measured);
        progress.finish();
        let lines = sink.lines();
        assert_eq!(lines[0].best_hold_rmse, None);
        assert_eq!(lines[1].best_hold_rmse, Some(0.5));
        assert_eq!(lines[1].best_hold_r2, None);
        assert_eq!(lines[1].evaluations, lines[1].measured + lines[1].failed);
    }
}

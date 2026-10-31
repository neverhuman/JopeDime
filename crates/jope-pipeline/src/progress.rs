//! Live search progress for `jope pipeline invent` (`jope.invent.progress.v1`).
//!
//! Opt-in best-effort NDJSON to a checked regular file, never stderr. Only fixed labels and
//! numbers/null are included, without candidate source, column names or raw cell values.
//! One process-wide worker serves all journals from a bounded 64-record queue, at most 2048
//! bytes per record plus its current record. Search threads never wait for writes. Full or
//! disconnected queues drop updates; file errors can lose records. No worker is joined, so
//! sequence gaps and a missing final done record are possible, including at process exit.
//! Writes are not guaranteed atomic. `kind: "progress"` updates occur at generation/comparison
//! boundaries, once per second except observed improvements and cut/search ends. Improvements
//! within one generation are coalesced. `kind: "candidate"` records report one evaluation's
//! scores (numbers/null and a fixed status label) as evaluations finish, at most one per
//! [`CANDIDATE_INTERVAL`]; a newer evaluation replaces the waiting one and the waiting one is
//! attempted when the search ends. Emission overhead can affect a wall-limited search.
//! See docs/pipeline.md; stdout schema/channel stays jope.candidates.v1.

use crate::search::Observation;
use serde::Serialize;
#[cfg(test)]
use std::sync::mpsc::Receiver;
use std::{
    io::Write,
    sync::{
        mpsc::{self, SyncSender},
        Arc, Mutex, OnceLock,
    },
    time::{Duration, Instant},
};

pub const SCHEMA: &str = "jope.invent.progress.v1";

/// Rate limit for lines that report neither an improvement nor the end of a cut.
pub const MIN_INTERVAL: Duration = Duration::from_secs(1);
/// At most five `candidate` records per second.
pub const CANDIDATE_INTERVAL: Duration = Duration::from_millis(200);
/// Most values in a `population_r2` snapshot (fewer when the record would exceed its cap).
pub const POPULATION_SAMPLES: usize = 64;

/// Record kinds.
pub const PROGRESS: &str = "progress";
pub const CANDIDATE: &str = "candidate";

/// The search phase a line was emitted in.
pub const EXPLORATION: &str = "exploration";
pub const COMPARISON: &str = "comparison";
pub const DONE: &str = "done";

/// One `jope.invent.progress.v1` line. Field order is the serialized order.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct Line {
    pub schema: String,
    /// [`PROGRESS`].
    pub kind: String,
    /// Per-run record counter over every kind, from 0 (the `sequence` of the planned
    /// `jope.run-event/v1`).
    pub sequence: u64,
    pub phase: String,
    /// 0-based position of the current cut among the searched cuts.
    pub cut: usize,
    pub cuts: usize,
    /// Generations completed so far, cumulative across cuts.
    pub generation: u64,
    /// Generations the search will run, when the configuration fixes it. The search stops on
    /// its evaluation and seconds bounds, so this is `null`.
    pub generations: Option<u64>,
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
    /// Lexical source tokens (code plus documentation) of every evaluated candidate's emitted
    /// source in all four languages, cumulative; a repeated candidate is charged again.
    pub emitted_tokens: u64,
    pub emitted_tokens_by_language: ByLanguage,
    /// `emitted_tokens / elapsed_s`; 0 when no time has elapsed.
    pub emitted_tokens_per_second: f64,
    /// Held-out R^2 of the candidates the latest generation ended with (carried survivors and
    /// newly measured, every island): finite only, sorted descending, evenly downsampled to at
    /// most [`POPULATION_SAMPLES`] and further, halving, while the record would exceed
    /// `MAX_LINE_BYTES`. `null` before the first generation ends.
    pub population_r2: Option<Vec<f64>>,
}

/// One `kind: "candidate"` record: a single evaluation's scores.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct CandidateLine {
    pub schema: String,
    /// [`CANDIDATE`].
    pub kind: String,
    pub sequence: u64,
    pub elapsed_s: f64,
    /// Cumulative 0-based index of the generation the evaluation belongs to (in the comparison
    /// phase, the generations completed).
    pub generation: u64,
    pub cut: usize,
    /// Finite held-out R^2 and RMSE of a measured evaluation; `null` otherwise.
    pub hold_r2: Option<f64>,
    pub hold_rmse: Option<f64>,
    /// `measured` or `failed`.
    pub status: String,
}

/// An evaluation waiting for its `candidate` record.
#[derive(Clone, Copy)]
struct PendingCandidate {
    generation: u64,
    cut: usize,
    hold_r2: Option<f64>,
    hold_rmse: Option<f64>,
    measured: bool,
}

/// Finite values sorted descending, evenly downsampled (first and last kept) to `max`.
pub fn population_snapshot(values: impl IntoIterator<Item = f64>, max: usize) -> Vec<f64> {
    let mut finite: Vec<f64> = values.into_iter().filter(|v| v.is_finite()).collect();
    finite.sort_by(|a, b| b.total_cmp(a));
    if finite.len() <= max || max < 2 {
        finite.truncate(max);
        return finite;
    }
    let last = finite.len() - 1;
    (0..max).map(|i| finite[i * last / (max - 1)]).collect()
}

/// Per-language split of [`Line::emitted_tokens`], in source-output ledger order
/// ([`feat_math::source_stream::LANGUAGES`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, serde::Deserialize)]
pub struct ByLanguage {
    pub rust: u64,
    pub python: u64,
    pub cpp17: u64,
    pub javascript: u64,
}

const QUEUE_CAPACITY: usize = 64;
const MAX_LINE_BYTES: usize = 2048;
struct PendingLine {
    file: Arc<std::fs::File>,
    bytes: Vec<u8>,
}
static FILE_QUEUE: OnceLock<Option<SyncSender<PendingLine>>> = OnceLock::new();

enum Destination {
    #[cfg(test)]
    Test(SyncSender<Vec<u8>>),
    File {
        sender: SyncSender<PendingLine>,
        file: Arc<std::fs::File>,
    },
}

fn file_queue() -> Option<SyncSender<PendingLine>> {
    FILE_QUEUE
        .get_or_init(|| {
            let (sender, receiver) = mpsc::sync_channel::<PendingLine>(QUEUE_CAPACITY);
            std::thread::Builder::new()
                .name("jope-invent-events".into())
                .spawn(move || {
                    for pending in receiver {
                        let mut file = &*pending.file;
                        // A failed file does not disable other journals. Blocking affects only this
                        // single process worker; producers never wait or join it.
                        let _ = file.write_all(&pending.bytes).and_then(|()| file.flush());
                    }
                })
                .ok()?;
            Some(sender)
        })
        .clone()
}

#[cfg(test)]
fn drain(mut sink: impl Write, receiver: Receiver<Vec<u8>>) {
    for bytes in receiver {
        if sink.write_all(&bytes).and_then(|()| sink.flush()).is_err() {
            break;
        }
    }
}

struct State {
    destination: Option<Destination>,
    started: Instant,
    last_emit: Option<Instant>,
    /// `(phase, cut, generation, evaluations)` of the last line, so a cut's end does not repeat
    /// the generation boundary line it coincides with.
    last_key: Option<(&'static str, usize, u64, u64)>,
    sequence: u64,
    last_candidate: Option<Instant>,
    pending: Option<PendingCandidate>,
    population: Option<Vec<f64>>,
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
    tokens: [u64; 4],
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
    // Production files share one process worker; tests can inject bounded queues.
    fn with_destination(
        destination: Option<Destination>,
        cuts: usize,
        budget: crate::search::Budget,
        started: Instant,
    ) -> Self {
        Progress {
            state: Mutex::new(State {
                destination,
                started,
                last_emit: None,
                last_key: None,
                sequence: 0,
                last_candidate: None,
                pending: None,
                population: None,
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
                tokens: [0; 4],
            }),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_sender(
        sender: Option<SyncSender<Vec<u8>>>,
        cuts: usize,
        budget: crate::search::Budget,
        started: Instant,
    ) -> Self {
        Self::with_destination(sender.map(Destination::Test), cuts, budget, started)
    }

    /// Best-effort output to a prevalidated regular file. One bounded worker serves the process.
    pub fn file(file: std::fs::File, cuts: usize, budget: crate::search::Budget) -> Self {
        let destination = file_queue().map(|sender| Destination::File {
            sender,
            file: Arc::new(file),
        });
        Self::with_destination(destination, cuts, budget, Instant::now())
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
        self.observe_at(None, observation, Instant::now());
    }

    /// Count one finished evaluation, made on the cut at `position`.
    pub fn observe(&self, position: usize, observation: &Observation) {
        self.observe_at(Some(position), observation, Instant::now());
    }

    fn observe_at(&self, position: Option<usize>, observation: &Observation, now: Instant) {
        let tokens = feat_math::source_stream::emitted_tokens(&observation.genome);
        let mut state = self.lock();
        if let Some(position) = position {
            state.cut = position;
        }
        Self::count(&mut state, observation, tokens);
        let measured = observation.measured();
        let finite = |value: Option<f64>| value.filter(|v| measured && v.is_finite());
        state.pending = Some(PendingCandidate {
            generation: state.generations_before + state.generations_in_cut,
            cut: state.cut,
            hold_r2: finite(observation.hold_r2),
            hold_rmse: finite(observation.rmse),
            measured,
        });
        Self::flush_candidate_if_due(&mut state, now);
    }

    /// Attempt the waiting `candidate` record when the candidate interval allows it.
    fn flush_candidate_if_due(state: &mut State, now: Instant) {
        let due = state
            .last_candidate
            .is_none_or(|last| now.saturating_duration_since(last) >= CANDIDATE_INTERVAL);
        let Some(pending) = state.pending.filter(|_| due) else {
            return;
        };
        state.pending = None;
        let line = CandidateLine {
            schema: SCHEMA.to_owned(),
            kind: CANDIDATE.to_owned(),
            sequence: state.sequence,
            elapsed_s: now.saturating_duration_since(state.started).as_secs_f64(),
            generation: pending.generation,
            cut: pending.cut,
            hold_r2: pending.hold_r2,
            hold_rmse: pending.hold_rmse,
            status: if pending.measured {
                "measured"
            } else {
                "failed"
            }
            .to_owned(),
        };
        if let Ok(bytes) = serde_json::to_vec(&line) {
            Self::send(state, bytes);
        }
        state.sequence += 1;
        state.last_candidate = Some(now);
    }

    fn count(state: &mut State, observation: &Observation, tokens: [u64; 4]) {
        for (total, add) in state.tokens.iter_mut().zip(tokens) {
            *total += add;
        }
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

    /// The current cut has completed `generations` generations, ending with candidates whose
    /// held-out R^2 values are `population_r2`; emit if due.
    pub fn generation_done(&self, generations: u64, population_r2: Vec<f64>) {
        self.generation_done_at(generations, population_r2, Instant::now());
    }

    fn generation_done_at(&self, generations: u64, population_r2: Vec<f64>, now: Instant) {
        let mut state = self.lock();
        state.generations_in_cut = generations;
        state.population = Some(population_snapshot(population_r2, POPULATION_SAMPLES));
        Self::flush_candidate_if_due(&mut state, now);
        Self::emit_if_due(&mut state, now);
    }

    /// A rate-limited (or improvement-driven) line between generation boundaries.
    pub fn tick(&self) {
        self.tick_at(Instant::now());
    }

    fn tick_at(&self, now: Instant) {
        let mut state = self.lock();
        Self::flush_candidate_if_due(&mut state, now);
        Self::emit_if_due(&mut state, now);
    }

    /// The current cut's exploration ended after `generations` generations: emit its end state
    /// unless the last line already reported exactly that.
    pub fn cut_done(&self, generations: u64) {
        self.cut_done_at(generations, Instant::now());
    }

    fn cut_done_at(&self, generations: u64, now: Instant) {
        let mut state = self.lock();
        state.generations_in_cut = generations;
        Self::flush_candidate_if_due(&mut state, now);
        if state.last_key != Some(Self::key(&state)) {
            Self::emit(&mut state, now);
        }
    }

    /// The search finished: attempt to enqueue its end state (phase `done`), without waiting.
    pub fn finish(&self) {
        let now = Instant::now();
        let mut state = self.lock();
        // The waiting candidate is attempted even inside the candidate interval.
        state.last_candidate = None;
        Self::flush_candidate_if_due(&mut state, now);
        state.phase = DONE;
        state.cut = state.cuts.saturating_sub(1);
        Self::emit(&mut state, now);
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
        let elapsed_s = now.saturating_duration_since(state.started).as_secs_f64();
        let emitted_tokens: u64 = state.tokens.iter().sum();
        let mut line = Line {
            schema: SCHEMA.to_owned(),
            kind: PROGRESS.to_owned(),
            sequence: state.sequence,
            phase: state.phase.to_owned(),
            cut: state.cut,
            cuts: state.cuts,
            generation: state.generations_before + state.generations_in_cut,
            generations: None,
            evaluations: state.evaluations,
            eval_budget: state.eval_budget,
            measured: state.measured,
            failed: state.failed,
            best_hold_rmse: state.best.map(|(rmse, _)| rmse),
            best_hold_r2: state.best.and_then(|(_, r2)| r2.filter(|r| r.is_finite())),
            elapsed_s,
            seconds_budget: state.seconds_budget,
            emitted_tokens,
            emitted_tokens_by_language: ByLanguage {
                rust: state.tokens[0],
                python: state.tokens[1],
                cpp17: state.tokens[2],
                javascript: state.tokens[3],
            },
            emitted_tokens_per_second: if elapsed_s > 0.0 {
                emitted_tokens as f64 / elapsed_s
            } else {
                0.0
            },
            population_r2: state.population.clone(),
        };
        // The snapshot gives way to the record cap: halve it until the record fits.
        let bytes = loop {
            let Ok(bytes) = serde_json::to_vec(&line) else {
                break None;
            };
            match &mut line.population_r2 {
                Some(values) if bytes.len() + 1 > MAX_LINE_BYTES && !values.is_empty() => {
                    *values = population_snapshot(values.iter().copied(), values.len() / 2);
                }
                _ => break Some(bytes),
            }
        };
        if let Some(bytes) = bytes {
            Self::send(state, bytes);
        }
        state.sequence += 1;
        state.last_key = Some(Self::key(state));
        state.last_emit = Some(now);
        state.improved = false;
    }

    /// Attempt one record (newline added) without waiting; over the cap it is dropped.
    fn send(state: &State, mut bytes: Vec<u8>) {
        bytes.push(b'\n');
        if bytes.len() <= MAX_LINE_BYTES {
            match &state.destination {
                #[cfg(test)]
                Some(Destination::Test(sender)) => {
                    let _ = sender.try_send(bytes);
                }
                Some(Destination::File { sender, file }) => {
                    let _ = sender.try_send(PendingLine {
                        file: file.clone(),
                        bytes,
                    });
                }
                None => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, sync::Arc};

    struct Shared {
        receiver: Receiver<Vec<u8>>,
        history: RefCell<Vec<serde_json::Value>>,
    }

    impl Shared {
        fn records(&self) -> Vec<serde_json::Value> {
            let mut history = self.history.borrow_mut();
            for bytes in self.receiver.try_iter() {
                assert!(bytes.len() <= MAX_LINE_BYTES);
                history.push(serde_json::from_slice(&bytes).unwrap());
            }
            history.clone()
        }
        fn kind<T: serde::de::DeserializeOwned>(&self, kind: &str) -> Vec<T> {
            self.records()
                .into_iter()
                .filter(|r| r["kind"] == kind)
                .map(|r| serde_json::from_value(r).unwrap())
                .collect()
        }
        fn lines(&self) -> Vec<Line> {
            self.kind(PROGRESS)
        }
    }

    fn observation(rmse: Option<f64>) -> Observation {
        Observation {
            genome: crate::select::BOOTSTRAP_GENOME.into(),
            program_sha256: String::new(),
            cut: 0,
            generation: 0,
            origin: "random",
            train_r2: rmse.map(|_| 0.5),
            hold_r2: rmse.map(|r| 1.0 - r),
            rmse,
            objective_loss: None,
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
        progress.generation_done_at(1, vec![], at(10));
        assert_eq!(sink.lines().len(), 1);
        // Within the second and no improvement: suppressed.
        progress.observe(0, &observation(None));
        progress.generation_done_at(2, vec![], at(400));
        progress.tick_at(at(900));
        assert_eq!(sink.lines().len(), 1);
        // An improvement reports immediately, inside the interval.
        progress.observe(0, &observation(Some(0.5)));
        progress.generation_done_at(3, vec![0.5], at(950));
        assert_eq!(sink.lines().len(), 2);
        // A worse measurement inside the interval is suppressed; after it, it reports.
        progress.observe(0, &observation(Some(0.9)));
        progress.generation_done_at(4, vec![0.5, 0.1], at(1500));
        assert_eq!(sink.lines().len(), 2);
        progress.generation_done_at(5, vec![0.1, f64::NAN, 0.5], at(1950));
        assert_eq!(sink.lines().len(), 3);
        // The end of a cut reports, unless the line just written already is its end state.
        progress.cut_done_at(5, at(1960));
        assert_eq!(sink.lines().len(), 3);
        progress.begin_cut(1);
        progress.observe(1, &observation(Some(0.7)));
        progress.generation_done_at(1, vec![0.3], at(2000));
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
                && w[0].sequence < w[1].sequence));
        // One counter numbers every kind.
        let sequences: Vec<u64> = sink
            .records()
            .iter()
            .map(|r| r["sequence"].as_u64().unwrap())
            .collect();
        assert_eq!(sequences, (0..sequences.len() as u64).collect::<Vec<_>>());
        assert!(lines
            .iter()
            .all(|l| l.kind == PROGRESS && l.generations.is_none()));
        assert_eq!(lines[0].population_r2, Some(vec![]));
        assert_eq!(lines[2].population_r2, Some(vec![0.5, 0.1]));
        assert_eq!(lines.last().unwrap().population_r2, Some(vec![0.3]));
        assert_eq!(lines[0].best_hold_rmse, None);
        assert_eq!(lines[1].best_hold_rmse, Some(0.5));
        assert_eq!(lines[1].best_hold_r2, Some(0.5));
        let last = lines.last().unwrap();
        assert_eq!(last.phase, DONE);
        assert_eq!((last.cut, last.generation), (1, 6));
        assert_eq!((last.evaluations, last.measured, last.failed), (5, 3, 2));
        assert_eq!(last.best_hold_rmse, Some(0.5));
        assert_eq!((last.eval_budget, last.seconds_budget), (20, 60.0));
        // Every evaluation, failed or not, is charged its emitted source in all four languages.
        let one = feat_math::source_stream::emitted_tokens(crate::select::BOOTSTRAP_GENOME);
        assert!(one.iter().all(|&n| n > 0), "{one:?}");
        assert_eq!(last.emitted_tokens, 5 * one.iter().sum::<u64>());
        let by = &last.emitted_tokens_by_language;
        assert_eq!(
            by.rust + by.python + by.cpp17 + by.javascript,
            last.emitted_tokens
        );
        assert!(lines
            .windows(2)
            .all(|w| w[0].emitted_tokens <= w[1].emitted_tokens));
    }

    #[test]
    fn full_and_disconnected_queues_do_not_wait_or_lose_counter_state() {
        let budget = crate::search::Budget {
            evaluations: 10,
            seconds: 30,
        };
        let (sender, receiver) = mpsc::sync_channel(1);
        let progress = Progress::with_sender(Some(sender), 2, budget, Instant::now());
        // Counted directly: this test is about progress records, not candidate records.
        Progress::count(&mut progress.lock(), &observation(Some(0.5)), [0; 4]);
        progress.tick(); // fills queue
        Progress::count(&mut progress.lock(), &observation(Some(0.4)), [0; 4]);
        progress.finish(); // full queue: drops done, without changing evaluation counters
        let first: Line = serde_json::from_slice(&receiver.try_recv().unwrap()).unwrap();
        assert_eq!((first.sequence, first.evaluations), (0, 1));
        progress.lock().cut = 1;
        Progress::count(&mut progress.lock(), &observation(None), [0; 4]);
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
                Progress::count(&mut progress.lock(), &observation(None), [0; 4]);
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
    fn file_worker_writes_regular_output_with_actual_token_counts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.ndjson");
        let progress = Progress::file(
            std::fs::File::create(&path).unwrap(),
            1,
            crate::search::Budget {
                evaluations: 1,
                seconds: 30,
            },
        );
        progress.observe(0, &observation(Some(0.5)));
        progress.finish();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let text = std::fs::read_to_string(&path).unwrap();
            if let Some(line) = text
                .lines()
                .filter_map(|s| serde_json::from_str::<Line>(s).ok())
                .find(|line| line.phase == DONE)
            {
                let expected =
                    feat_math::source_stream::emitted_tokens(crate::select::BOOTSTRAP_GENOME);
                assert_eq!(line.emitted_tokens, expected.iter().sum::<u64>());
                assert!(line.emitted_tokens > 0);
                assert_eq!(line.evaluations, 1);
                break;
            }
            assert!(
                Instant::now() < deadline,
                "regular-file worker produced no done record"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
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

    #[test]
    fn candidates_are_rate_limited_newest_wins_and_the_last_is_attempted_at_the_end() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let (progress, sink) = reporter(t0);
        progress.begin_cut(0);
        // 50 evaluations 10 ms apart (100 a second); the last one measured.
        for i in 0..50 {
            let rmse = (i == 49).then_some(0.25);
            progress.observe_at(Some(0), &observation(rmse), at(i * 10));
        }
        progress.finish();
        let candidates: Vec<CandidateLine> = sink.kind(CANDIDATE);
        // 0, 200 and 400 ms, then the newest waiting one (490 ms) at the end.
        assert_eq!(candidates.len(), 4);
        assert!(candidates[..3]
            .windows(2)
            .all(|w| w[1].elapsed_s - w[0].elapsed_s >= 0.2 - 1e-9));
        assert!(candidates[..3]
            .iter()
            .all(|c| c.status == "failed" && c.hold_r2.is_none() && c.hold_rmse.is_none()));
        let last = candidates.last().unwrap();
        assert_eq!(
            (last.status.as_str(), last.hold_rmse),
            ("measured", Some(0.25))
        );
        assert_eq!(last.hold_r2, Some(0.75));
        // Fixed labels and numbers only.
        for record in sink.records() {
            let object = record.as_object().unwrap();
            assert!(object.iter().all(|(key, value)| !value.is_string()
                || matches!(key.as_str(), "schema" | "kind" | "phase" | "status")));
        }
    }

    #[test]
    fn worst_case_records_fit_the_cap() {
        let (progress, sink) = reporter(Instant::now());
        {
            let mut state = progress.lock();
            state.evaluations = u64::MAX;
            state.measured = u64::MAX;
            state.failed = u64::MAX;
            state.eval_budget = u64::MAX;
            state.generations_before = u64::MAX / 2;
            state.generations_in_cut = u64::MAX / 2;
            state.tokens = [u64::MAX / 4; 4];
            state.cut = usize::MAX;
            state.cuts = usize::MAX;
            state.seconds_budget = -f64::MAX;
            state.best = Some((-f64::MIN_POSITIVE, Some(-f64::MAX)));
            state.sequence = u64::MAX / 2;
            state.pending = Some(PendingCandidate {
                generation: u64::MAX,
                cut: usize::MAX,
                hold_r2: Some(-f64::MIN_POSITIVE),
                hold_rmse: Some(-f64::MAX),
                measured: false,
            });
            state.phase = COMPARISON;
        }
        // 1000 of the longest finite values: sorted, downsampled to 64, then halved to fit.
        let wide = (0..1000).map(|i| -f64::MAX / (1.0 + i as f64 * 1e-13));
        progress.generation_done(u64::MAX / 2, wide.collect());
        let records = sink.records();
        let line: Line = serde_json::from_value(
            records
                .iter()
                .find(|r| r["kind"] == PROGRESS)
                .unwrap()
                .clone(),
        )
        .unwrap();
        let candidate = records.iter().find(|r| r["kind"] == CANDIDATE).unwrap();
        assert!(serde_json::to_vec(candidate).unwrap().len() < 512);
        let population = line.population_r2.unwrap();
        assert!(!population.is_empty() && population.len() < POPULATION_SAMPLES);
        assert!(population.windows(2).all(|w| w[0] >= w[1]));
        // Typical R^2 values keep all 64.
        let (progress, sink) = reporter(Instant::now());
        progress.generation_done(1, (0..500).map(|i| 1.0 - i as f64 * 0.001234567).collect());
        assert_eq!(sink.lines()[0].population_r2.as_ref().unwrap().len(), 64);
    }

    #[test]
    fn population_is_finite_sorted_descending_and_at_most_64() {
        let values = (0..1000).map(|i| i as f64 / 1000.0).chain([
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ]);
        let snapshot = population_snapshot(values, POPULATION_SAMPLES);
        assert_eq!(snapshot.len(), 64);
        assert!(snapshot.iter().all(|v| v.is_finite()));
        assert!(snapshot.windows(2).all(|w| w[0] >= w[1]));
        assert_eq!((snapshot[0], snapshot[63]), (0.999, 0.0));
        assert_eq!(
            population_snapshot([0.2, f64::NAN, 0.9], 64),
            vec![0.9, 0.2]
        );
    }
}

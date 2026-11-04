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

mod records;
use records::PendingCandidate;
pub use records::{population_snapshot, ByLanguage, CandidateLine, Line};
#[cfg(test)]
mod tests;
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

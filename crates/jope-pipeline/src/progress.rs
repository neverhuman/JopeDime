//! Live search progress for `jope pipeline invent` (`jope.invent.progress.v1`).
//!
//! Opt-in (`--events PATH`): one compact JSON object per line in the events file, each written
//! with a single `write_all` and flushed, so a parent process can show progress while the search
//! runs. Lines carry numbers only: never candidate source, column names or data values. stdout
//! (`jope.candidates.v1`) and stderr are untouched. These records are interim, intended to become
//! a kind of the planned `jope.run-event/v1` envelope. See `docs/pipeline.md`.
//!
//! Cadence: at most one line per [`MIN_INTERVAL`] at generation (and comparison-evaluation)
//! boundaries, plus one whenever the best-so-far held-out RMSE improves, one at the end of every
//! cut's exploration and one when the whole search finishes, so the last line is the end state.

use crate::search::Observation;
use serde::Serialize;
use std::{
    io::Write,
    sync::Mutex,
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
    /// Lexical source tokens (code plus documentation) of every evaluated candidate's emitted
    /// source in all four languages, cumulative; a repeated candidate is charged again.
    pub emitted_tokens: u64,
    pub emitted_tokens_by_language: ByLanguage,
    /// `emitted_tokens / elapsed_s`; 0 when no time has elapsed.
    pub emitted_tokens_per_second: f64,
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

struct State {
    sink: Box<dyn Write + Send>,
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
    /// A reporter for `cuts` cuts with the given per-cut budget, writing to `sink`.
    pub fn new(
        sink: Box<dyn Write + Send>,
        cuts: usize,
        budget: crate::search::Budget,
        started: Instant,
    ) -> Self {
        Progress {
            state: Mutex::new(State {
                sink,
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
                tokens: [0; 4],
            }),
        }
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
        let tokens = feat_math::source_stream::emitted_tokens(&observation.genome);
        let mut state = self.lock();
        Self::count(&mut state, observation, tokens);
    }

    /// Count one finished evaluation, made on the cut at `position`.
    pub fn observe(&self, position: usize, observation: &Observation) {
        // Emitted outside the lock, so concurrently compared cuts do not wait on each other.
        let tokens = feat_math::source_stream::emitted_tokens(&observation.genome);
        let mut state = self.lock();
        state.cut = position;
        Self::count(&mut state, observation, tokens);
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

    /// The whole search finished: always emit the end state (phase `done`).
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
        let elapsed_s = now.saturating_duration_since(state.started).as_secs_f64();
        let emitted_tokens: u64 = state.tokens.iter().sum();
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
        };
        let Ok(mut bytes) = serde_json::to_vec(&line) else {
            return;
        };
        bytes.push(b'\n');
        // Progress is advisory: a failing events file never stops the search.
        let _ = state
            .sink
            .write_all(&bytes)
            .and_then(|()| state.sink.flush());
        state.sequence += 1;
        state.last_key = Some(Self::key(state));
        state.last_emit = Some(now);
        state.improved = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[derive(Clone, Default)]
    struct Shared(Arc<Mutex<Vec<u8>>>);

    impl Write for Shared {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Shared {
        fn lines(&self) -> Vec<Line> {
            String::from_utf8(self.0.lock().unwrap().clone())
                .unwrap()
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect()
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
        let sink = Shared::default();
        let budget = crate::search::Budget {
            evaluations: 10,
            seconds: 30,
        };
        (Progress::new(Box::new(sink.clone()), 2, budget, t0), sink)
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
}

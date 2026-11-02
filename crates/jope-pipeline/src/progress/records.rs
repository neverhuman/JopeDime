use super::*;

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
pub(super) struct PendingCandidate {
    pub(super) generation: u64,
    pub(super) cut: usize,
    pub(super) hold_r2: Option<f64>,
    pub(super) hold_rmse: Option<f64>,
    pub(super) measured: bool,
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

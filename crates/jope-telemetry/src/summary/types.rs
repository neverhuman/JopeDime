use serde::Serialize;
use std::collections::BTreeMap;

pub const SUMMARY_SCHEMA: &str = "jope.telemetry.train-gp-summary/v1";

/// A scalar from the hot path or an import, by spec metric name.
#[derive(Clone, Copy, Debug)]
pub struct ScalarIn<'a> {
    pub metric: &'a str,
    pub group: Option<&'a str>,
    pub step: Option<u64>,
    pub value: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Dist {
    pub n: u64,
    pub nonfinite: u64,
    pub first: Option<f64>,
    pub last: Option<f64>,
    pub min: Option<f64>,
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub max: Option<f64>,
}

impl Dist {
    pub(super) fn of(values: &[f64]) -> Self {
        let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        let first = finite.first().copied();
        let last = finite.last().copied();
        finite.sort_by(f64::total_cmp);
        let q = |p| crate::gp::quantile_sorted(&finite, p);
        Self {
            n: values.len() as u64,
            nonfinite: (values.len() - finite.len()) as u64,
            first,
            last,
            min: finite.first().copied(),
            p50: q(0.5),
            p95: q(0.95),
            max: finite.last().copied(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TrainingSummary {
    pub steps: u64,
    pub loss: Dist,
    pub grad_norm: Dist,
    pub clipped_frac: Option<f64>,
    pub step_wall_ms: Dist,
    /// Mean share of the host step wall time per host span (rows T11), largest first.
    pub host_span_share: Vec<(String, f64)>,
    pub unattributed_share: Option<f64>,
    pub val_loss: Dist,
    pub generalization_gap_last: Option<f64>,
}

/// tok/s as the two labeled metrics every report shows (Jepson ruling 2026-10-29 19:18 MDT):
/// Rust-only, the headline, and all languages. Legacy unlabeled values are kept apart, and so is
/// reasoning tok/s (ruling t518u), a run total rather than a per-step distribution.
#[derive(Clone, Debug, Default, Serialize)]
pub struct TokSummary {
    pub rust_only: Dist,
    pub all_languages: Dist,
    /// Steps whose Rust-only rate was timed on the Rust output alone, or on the shared span.
    pub rust_timed_alone_steps: u64,
    pub rust_shared_span_steps: u64,
    /// `throughput.tokens_per_s` from producers that predate the pair; not either metric.
    pub legacy_unlabeled: Option<Dist>,
    /// The run's total reasoning tokens over its wall-clock seconds, from the last emission, with
    /// the two counts it divides and the tokenizer that counted the tokens.
    pub reasoning_tok_per_s: Option<f64>,
    pub reasoning_tokens: Option<f64>,
    pub reasoning_wall_s: Option<f64>,
    pub reasoning_tokenizer: Option<String>,
}

pub const TOK_S_RUST: &str = "throughput.tokens_per_s.rust";
pub const TOK_S_ALL_LANGUAGES: &str = "throughput.tokens_per_s.all_languages";
pub const TOK_S_RUST_TIMED_ALONE: &str = "throughput.tokens_per_s.rust_timed_alone";
pub const TOK_S_LEGACY: &str = "throughput.tokens_per_s";
pub const TOK_S_REASONING: &str = "throughput.reasoning_tok_per_s";
pub const REASONING_TOKENS: &str = "throughput.reasoning_tokens";
pub const REASONING_WALL_S: &str = "throughput.reasoning_wall_s";
pub const TOK_S_REASONING_LABEL: &str = "reasoning tok/s";
/// The labels every report prints next to the two tok/s metrics, Rust-only first.
pub const TOK_S_LABELS: [(&str, &str); 2] = [
    (TOK_S_RUST, "Rust-only tok/s (headline)"),
    (TOK_S_ALL_LANGUAGES, "all-languages tok/s"),
];

/// How the Rust-only rate was timed, from its `rust_timed_alone` step counts.
pub fn rust_timing(alone: u64, shared: u64) -> String {
    match (alone, shared) {
        (0, 0) => "timing basis unrecorded".into(),
        (_, 0) => "timed on the Rust output alone".into(),
        (0, _) => "timed on the shared all-languages span".into(),
        (a, b) => format!("timed on the Rust output alone in {a} of {} steps", a + b),
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DopeSummary {
    pub guard_records: u64,
    pub compared: u64,
    pub all_equal: Option<bool>,
    pub violations: Vec<String>,
    pub last_composite_sha256: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct GpContextSummary {
    pub generations: u64,
    pub last_unique_genotype_frac: Option<f64>,
    pub last_fitness_profile_unique_frac: Option<f64>,
    pub last_op_entropy_norm: Option<f64>,
    pub best_fitness: Option<f64>,
    pub gens_since_improved: u32,
    pub failed: u64,
    pub evaluated: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct EvolutionSummary {
    pub rounds: Vec<u32>,
    /// Per target-delta key, values by round order.
    pub target_deltas: BTreeMap<String, Vec<f64>>,
    /// Keys whose last value is under 10% of the first in magnitude: the round-over-round gain
    /// is fading.
    pub diminishing: Vec<String>,
    pub hwm_rounds_inconsistent: Vec<u32>,
    pub hwm_stale_files: u64,
    pub retention_totals: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct AlarmSummary {
    pub by_code: BTreeMap<String, u64>,
    pub fail_closed: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Summary {
    pub schema: String,
    pub training: TrainingSummary,
    pub tok_s: TokSummary,
    pub dope: DopeSummary,
    pub gp: BTreeMap<String, GpContextSummary>,
    pub evolution: EvolutionSummary,
    pub alarms: AlarmSummary,
    /// The latest Elo snapshot, read against the method panel (row F2).
    pub elo: Option<EloSummary>,
    /// Metrics that were looked for and not found, with the recorded reason when there is one.
    pub missing: BTreeMap<String, String>,
    pub unparsed_records: u64,
}

/// Elo against methods for one task type, from one fit.
#[derive(Clone, Debug, Default, Serialize)]
pub struct EloTaskSummary {
    pub contenders: u32,
    /// The best-rated flywheel generation (`ROUND_k` contenders), its Elo and 1-based rank.
    pub flywheel: Option<(String, f64, u32)>,
    /// The best-rated method that is not a flywheel generation.
    pub best_method: Option<(String, f64)>,
    pub delta_vs_best_method: Option<f64>,
    pub top: Vec<(String, f64)>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct EloSummary {
    pub round: u32,
    pub fit_digest: String,
    pub snapshots_seen: u32,
    pub tasks: BTreeMap<TaskType, EloTaskSummary>,
    /// Why these ratings aren't qualified yet (missing roster, incomplete panel, ...).
    pub caveats: Vec<String>,
}

/// Contenders named like this are flywheel generations; every other contender is a method.
pub fn is_flywheel_contender(name: &str) -> bool {
    name.starts_with("ROUND_") || name.starts_with("flywheel")
}

pub(super) fn opt(v: Option<f64>) -> String {
    v.map_or_else(|| "-".to_owned(), |x| format!("{x:.4}"))
}

use serde_json::{json, Value};

/// Fixed composite coefficients. Lineage B runs no learned or uncertainty
/// weighting: `stability` and every `loss_log_vars` scalar are connected only
/// through a zero coefficient so that every parameter has a defined gradient
/// and the packed all-reduce layout is identical on every rank.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Coefficients {
    pub sequence_nll: f64,
    pub ranking_bt: f64,
    pub ranking_listwise: f64,
    pub runtime_huber: f64,
    pub contextfree_bt: f64,
    pub stability: f64,
    pub loss_log_vars: f64,
}

/// The accepted lineage-B coefficients. Changing any of them is a different
/// objective, not a tuning of this one.
pub const COEFFICIENTS: Coefficients = Coefficients {
    sequence_nll: 1.0,
    ranking_bt: 0.25,
    ranking_listwise: 0.10,
    runtime_huber: 0.10,
    contextfree_bt: 0.05,
    stability: 0.0,
    loss_log_vars: 0.0,
};

/// Identity of the runtime term, recorded in checkpoints since before W6.
pub const RUNTIME_OBJECTIVE_ID: &str = "huber_log1p_seconds_v2";
pub const OBJECTIVE_SCHEMA: &str = "jope.training_objective.v1";

/// The complete lineage-B objective as one canonical value: every
/// coefficient, mask, smoothing, weight, reduction, tie and empty-support
/// rule and inactive term, taken from the constants the arithmetic uses.
/// Two runs have the same objective exactly when these values are equal
/// (W6 G1); the partition dependence is bound separately by
/// `task_contract::training_partition`.
pub fn identity() -> Value {
    let c = COEFFICIENTS;
    json!({
        "schema": OBJECTIVE_SCHEMA,
        "coefficients": {
            "sequence_nll": c.sequence_nll,
            "ranking_bt": c.ranking_bt,
            "ranking_listwise": c.ranking_listwise,
            "runtime_huber": c.runtime_huber,
            "contextfree_bt": c.contextfree_bt,
            "stability": c.stability,
            "loss_log_vars": c.loss_log_vars,
        },
        "sequence": {
            "targets": "tokens_1_to_T_minus_1_bos_never_a_target",
            "legal_mask": "grammar_legal_set_neg_infinity",
            "label_smoothing": crate::neural::LABEL_SMOOTHING,
            "smoothing_support": "grammar_legal_set",
            "row_reduction": "mean_over_supervised_positions",
            "row_weight": {"floor": ROW_WEIGHT_FLOOR, "percentile_exponent": ROW_WEIGHT_EXPONENT},
            "batch_reduction": "row_weighted_mean",
            "row_without_supervision": "refused",
        },
        "ranking": {
            "backward": crate::neural::RANKING_BACKWARD_ID,
            "score": "mean_of_reward_heads",
            "reward_heads": REWARD_HEADS,
            "group": "context_and_comparison_of_8",
            "pairs": "all_unordered_within_group",
            "pair_weight": "absolute_value_difference",
            "boost_threshold": crate::neural::BOOST_THRESHOLD,
            "boost_factor": crate::neural::BOOST_FACTOR,
            "ties": "skipped",
            "pair_reduction": "weighted_mean_over_microbatch",
            "empty_support": "connected_zero",
            "listwise_temperature": crate::neural::LISTWISE_TEMPERATURE,
            "listwise_min_members": crate::neural::LISTWISE_MIN_MEMBERS,
            "listwise_reduction": "mean_over_eligible_groups",
        },
        "runtime": {
            "objective": RUNTIME_OBJECTIVE_ID,
            "huber_beta": RUNTIME_HUBER_BETA,
            "reduction": "mean_over_rows",
        },
        "ablation": {"included": true, "construction": "ranking_bt_on_context_free_head"},
        "inactive": ["stability", "loss_log_vars"],
        "reward_head_dropout": crate::neural::REWARD_HEAD_DROPOUT,
        "supervision": "validate_rows_v1",
        "nonfinite_objective": "refused",
        "normalization": "microbatch_weighted_means_rank_mean_v1",
    })
}

/// SHA-256 of the canonical (sorted-key) serialization of [`identity`].
pub fn identity_sha256() -> String {
    identity_digest(&identity())
}

/// Digest of a recorded objective value, for checking a manifest's own binding.
pub fn identity_digest(objective: &Value) -> String {
    crate::manifest::hash(&serde_json::to_vec(objective).expect("JSON value serializes"))
}

/// Telemetry and receipt names of the five active terms, in value order.
pub const TERM_NAMES: [&str; 5] = [
    "sequence_nll",
    "ranking_bt",
    "ranking_listwise",
    "runtime_huber",
    "contextfree_bt",
];

/// Parameter roots that receive an exact-zero gradient from [`composite`] and
/// therefore only AdamW decoupled decay. They must never be cited as learning.
/// Also without learning signal, but not listed here because they share a
/// tensor with active rows or are zero only in exact arithmetic: embedding rows
/// at or above the vocabulary size, embedding row 0 (PAD), and the
/// shift-invariant output biases `reward_heads.k.3.bias` and
/// `reward_head_ablation.3.bias`.
pub const INACTIVE_PARAMETER_PREFIXES: [&str; 2] = ["stability_head.", "loss_log_vars."];

/// Sequence row weight `ROW_WEIGHT_FLOOR + percentile^ROW_WEIGHT_EXPONENT`,
/// formed in f32 because the corpus stores the percentile as f32.
pub const ROW_WEIGHT_FLOOR: f32 = 0.05;
pub const ROW_WEIGHT_EXPONENT: i32 = 4;
/// SmoothL1 transition point on `log1p(seconds)`.
pub const RUNTIME_HUBER_BETA: f64 = 1.0;
/// Rows in one sampled comparison: the champion plus seven quantile comparators.
pub const COMPARISON_ROWS: usize = 8;
/// Reward heads averaged into one ranking score per row.
pub const REWARD_HEADS: i64 = 4;

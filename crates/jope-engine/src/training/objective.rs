//! Lineage-B composite training objective for one rank-local microbatch.
//!
//! The trainer (`training::rank_impl`) calls [`composite`] for every training
//! and fixed-panel validation microbatch, so the arithmetic here is the update
//! path itself. Its term-by-term specification, including masks, reductions and
//! empty-support rules, is `docs/internal/core-ip/work-orders/evidence/
//! W06-objective-specification.md`; `tests/objective_fixtures.rs` recomputes
//! every term and its gradients independently in f64.
//!
//! The objective is partition dependent: every reduction below is a
//! microbatch-local weighted mean, and the trainer averages microbatches and
//! ranks (`task_contract::training_partition`,
//! `microbatch_weighted_means_rank_mean_v1`). Partition v2
//! (`training::partition`, W7-T06) reuses [`terms`] and reduces each term by
//! its global numerator and denominator instead.
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use tch::{Kind, Tensor};

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

/// The per-record labels the objective consumes. `record` is the corpus record
/// identity the row was sampled as; it is what makes duplicate detection and
/// group membership checkable at the objective boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowLabels {
    pub record: usize,
    pub context: usize,
    /// `1 / (1 + rmse / median_group_rmse)`: higher is better, in (0, 1].
    pub value: f32,
    /// Rank target over the whole corpus group, in [0, 1].
    pub percentile: f32,
    /// `ln(1 + seconds)` of a completed measurement, >= 0.
    pub runtime: f32,
}

impl RowLabels {
    pub fn from_record(record: usize, r: &crate::corpus::Record) -> Self {
        Self {
            record,
            context: r.context,
            value: r.value,
            percentile: r.percentile,
            runtime: r.runtime,
        }
    }
}

/// Labels are supervision, not data to be reinterpreted. Every row must carry
/// a finite value in (0, 1], a finite percentile in [0, 1] and a finite
/// completed runtime label >= 0. Rows form complete comparisons of
/// [`COMPARISON_ROWS`] that share one context and never repeat a record, so no
/// cross-identity pair and no duplicated member can reach a ranking term.
///
/// `neural::ranking` itself still skips nonfinite values and ineligible lists
/// (its declared behavior for other callers); at the training boundary such a
/// row is malformed supervision and is refused before any loss is formed.
pub fn validate_rows(rows: &[RowLabels]) -> Result<()> {
    ensure!(
        !rows.is_empty() && rows.len() % COMPARISON_ROWS == 0,
        "objective rows must form complete comparisons of {COMPARISON_ROWS}"
    );
    for (i, r) in rows.iter().enumerate() {
        ensure!(
            r.value.is_finite() && r.value > 0.0 && r.value <= 1.0,
            "invalid value label at row {i}: {}",
            r.value
        );
        ensure!(
            r.percentile.is_finite() && (0.0..=1.0).contains(&r.percentile),
            "invalid percentile label at row {i}: {}",
            r.percentile
        );
        ensure!(
            r.runtime.is_finite() && r.runtime >= 0.0,
            "invalid runtime label at row {i}: {}",
            r.runtime
        );
    }
    for (k, comparison) in rows.chunks(COMPARISON_ROWS).enumerate() {
        ensure!(
            comparison
                .iter()
                .all(|r| r.context == comparison[0].context),
            "comparison {k} spans contexts"
        );
        let distinct = comparison
            .iter()
            .map(|r| r.record)
            .collect::<std::collections::BTreeSet<_>>();
        ensure!(
            distinct.len() == comparison.len(),
            "comparison {k} repeats a record"
        );
    }
    Ok(())
}

/// The supervision one microbatch actually delivers after masking, derived
/// from the same labels and record lengths [`composite`] consumes. Journaled
/// per microbatch so consumption is reported in supervised units, not rows.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Supervision {
    pub rows: usize,
    /// Padded decoder width `T` of the microbatch.
    pub width: usize,
    /// Supervised target positions: `Σ (length − 1)`; BOS is never a target.
    pub supervised_positions: usize,
    /// Sequence-term denominator `Σ (0.05 + p⁴)`, summed in f64 from f32 weights.
    pub row_weight_sum: f64,
    /// Bradley–Terry pairs with distinct values (shared by `ablation_bt`).
    pub ranking_pairs: usize,
    /// Pairs skipped as exact value ties.
    pub tied_pairs: usize,
    /// Listwise groups (every validated comparison is eligible).
    pub listwise_groups: usize,
}

pub fn supervision(rows: &[RowLabels], lengths: &[usize]) -> Result<Supervision> {
    validate_rows(rows)?;
    let sequence = crate::corpus::SEQUENCE;
    ensure!(
        lengths.len() == rows.len() && lengths.iter().all(|&l| (2..=sequence).contains(&l)),
        "record lengths must cover BOS, at least one target and at most {sequence} tokens"
    );
    let longest = *lengths.iter().max().expect("validated rows are non-empty");
    let (mut ranking_pairs, mut tied_pairs) = (0, 0);
    for comparison in rows.chunks(COMPARISON_ROWS) {
        for (a, left) in comparison.iter().enumerate() {
            for right in &comparison[a + 1..] {
                if left.value == right.value {
                    tied_pairs += 1;
                } else {
                    ranking_pairs += 1;
                }
            }
        }
    }
    Ok(Supervision {
        rows: rows.len(),
        width: (longest.div_ceil(8) * 8).min(sequence),
        supervised_positions: lengths.iter().map(|l| l - 1).sum(),
        row_weight_sum: rows
            .iter()
            .map(|r| f64::from(row_weight(r.percentile)))
            .sum(),
        ranking_pairs,
        tied_pairs,
        listwise_groups: rows.len() / COMPARISON_ROWS,
    })
}

/// Sampled comparisons checked against the frozen corpus index: every
/// comparison is drawn from exactly one corpus group of the expected
/// partition, starts with that group's champion (`records[0]`, the lowest
/// cut-normalized RMSE), and its rows carry that group's context.
pub struct SampledComparisons {
    record_group: Vec<u32>,
}

impl SampledComparisons {
    pub fn new(index: &crate::corpus::Index) -> Result<Self> {
        ensure!(
            index.groups.len() < u32::MAX as usize,
            "too many corpus groups"
        );
        let mut record_group = vec![u32::MAX; index.records];
        for (g, group) in index.groups.iter().enumerate() {
            for &record in &group.records {
                let slot = record_group.get_mut(record).ok_or_else(|| {
                    anyhow::anyhow!("corpus group {g} names record {record} outside the corpus")
                })?;
                ensure!(
                    *slot == u32::MAX,
                    "record {record} belongs to two corpus groups"
                );
                *slot = g as u32;
            }
        }
        Ok(Self { record_group })
    }

    fn group_of(&self, record: usize) -> Option<usize> {
        self.record_group
            .get(record)
            .filter(|&&g| g != u32::MAX)
            .map(|&g| g as usize)
    }

    /// Check sampled record identities before any record is read.
    pub fn check_ids(
        &self,
        index: &crate::corpus::Index,
        ids: &[usize],
        partition: &str,
    ) -> Result<()> {
        ensure!(
            !ids.is_empty() && ids.len() % COMPARISON_ROWS == 0,
            "sampled records must form complete comparisons of {COMPARISON_ROWS}"
        );
        for (k, comparison) in ids.chunks(COMPARISON_ROWS).enumerate() {
            let g = self.group_of(comparison[0]).ok_or_else(|| {
                anyhow::anyhow!("comparison {k} starts outside every corpus group")
            })?;
            let group = &index.groups[g];
            ensure!(
                group.partition == partition,
                "comparison {k} draws from partition {} instead of {partition}",
                group.partition
            );
            ensure!(
                group.records.first() == Some(&comparison[0]),
                "comparison {k} does not start with its group champion"
            );
            ensure!(
                comparison.iter().all(|&id| self.group_of(id) == Some(g)),
                "comparison {k} mixes corpus groups"
            );
            let distinct = comparison.iter().collect::<std::collections::BTreeSet<_>>();
            ensure!(
                distinct.len() == comparison.len(),
                "comparison {k} repeats a record"
            );
        }
        Ok(())
    }

    /// Check the labels read for those identities: valid supervision and the
    /// context of the corpus group each record was sampled from.
    pub fn check_rows(&self, index: &crate::corpus::Index, rows: &[RowLabels]) -> Result<()> {
        validate_rows(rows)?;
        for r in rows {
            let g = self.group_of(r.record).ok_or_else(|| {
                anyhow::anyhow!("record {} is outside every corpus group", r.record)
            })?;
            ensure!(
                r.context == index.groups[g].context,
                "record {} carries context {} instead of its group context {}",
                r.record,
                r.context,
                index.groups[g].context
            );
        }
        Ok(())
    }
}

/// Policy forward outputs for one microbatch of `B` rows and width `T`:
/// logits `[B, T, V]`, rewards `[B, 4]`, runtime (seconds scale) `[B]`,
/// stability `[B]` and context-free ablation reward `[B]`.
pub struct Outputs<'a> {
    pub logits: &'a Tensor,
    pub rewards: &'a Tensor,
    pub runtime: &'a Tensor,
    pub stability: &'a Tensor,
    pub ablation: &'a Tensor,
}

/// Sequence row weight exactly as the trainer forms it.
pub fn row_weight(percentile: f32) -> f32 {
    ROW_WEIGHT_FLOOR + percentile.powi(ROW_WEIGHT_EXPONENT)
}

/// Comparison group key: rows `8k..8k+8` of a microbatch form one group, so
/// two comparisons drawn from the same context never pair with each other.
pub fn group_keys(rows: &[RowLabels]) -> Vec<String> {
    rows.iter()
        .enumerate()
        .map(|(i, r)| format!("{}:{}", r.context, i / COMPARISON_ROWS))
        .collect()
}

/// The five active terms of one microbatch as microbatch-local reductions, in
/// [`TERM_NAMES`] order, and the inactive stability sum. Shared by
/// [`composite`] (partition v1) and `training::partition::composite` (v2),
/// so both partitions run the same term arithmetic.
pub struct Terms {
    pub active: [Tensor; 5],
    pub stability: Tensor,
}

/// Validate one microbatch and form its [`Terms`].
///
/// `tokens` and `pad` are `[B, T]`; `legal` is the `[B, T-1, V]` legal-token
/// mask for targets `tokens[:, 1..T]` (BOS is never a target).
pub fn terms(
    outputs: &Outputs<'_>,
    tokens: &Tensor,
    pad: &Tensor,
    legal: &Tensor,
    rows: &[RowLabels],
) -> Result<Terms> {
    validate_rows(rows)?;
    let (batch, length) = tokens.size2()?;
    ensure!(
        batch as usize == rows.len() && length > 1 && pad.size() == tokens.size(),
        "objective rows and token batch differ"
    );
    // Malformed outputs are explicit errors, never a silently different
    // reduction (for example a mean over fewer reward heads) or a panic.
    let (logit_rows, positions, vocabulary) = outputs.logits.size3()?;
    ensure!(
        logit_rows == batch && positions == length + 1,
        "decoder logits must cover the context slot and every token position"
    );
    ensure!(
        legal.size() == [batch, length - 1, vocabulary],
        "legal-token mask differs from the supervised positions"
    );
    ensure!(
        outputs.rewards.size() == [batch, REWARD_HEADS]
            && outputs.runtime.size() == [batch]
            && outputs.stability.size() == [batch]
            && outputs.ablation.size() == [batch],
        "policy head outputs differ from the objective rows"
    );
    let device = tokens.device();
    let weights = Tensor::from_slice(
        &rows
            .iter()
            .map(|r| row_weight(r.percentile))
            .collect::<Vec<_>>(),
    )
    .to_device(device);
    let nll = crate::neural::sequence_nll(
        &outputs.logits.narrow(1, 1, length - 1),
        &tokens.narrow(1, 1, length - 1),
        legal,
        &pad.narrow(1, 1, length - 1),
        &weights,
    )?;
    let reward = outputs.rewards.mean_dim(&[1i64][..], false, Kind::Float);
    let values = rows.iter().map(|r| r.value as f64).collect::<Vec<_>>();
    let percentiles = rows.iter().map(|r| r.percentile).collect::<Vec<_>>();
    let groups = group_keys(rows);
    let (bt, list) = crate::neural::ranking(&reward, &values, &percentiles, &groups)?;
    let abt = crate::neural::ranking_pairwise(outputs.ablation, &values, &percentiles, &groups)?;
    let runtime_label =
        Tensor::from_slice(&rows.iter().map(|r| r.runtime).collect::<Vec<_>>()).to_device(device);
    let rt = outputs.runtime.to_kind(Kind::Float).log1p().smooth_l1_loss(
        &runtime_label,
        tch::Reduction::Mean,
        RUNTIME_HUBER_BETA,
    );
    Ok(Terms {
        active: [nll, bt, list, rt, abt],
        stability: outputs.stability.sum(Kind::Float),
    })
}

/// The composite scalar objective of one microbatch and its five active raw
/// term values `[sequence_nll, ranking_bt, ranking_listwise, runtime_huber,
/// contextfree_bt]`, reduced microbatch-locally (partition v1).
///
/// `tokens` and `pad` are `[B, T]`; `legal` is the `[B, T-1, V]` legal-token
/// mask for targets `tokens[:, 1..T]` (BOS is never a target).
pub fn composite<'a>(
    outputs: &Outputs<'_>,
    loss_log_vars: impl IntoIterator<Item = &'a Tensor>,
    tokens: &Tensor,
    pad: &Tensor,
    legal: &Tensor,
    rows: &[RowLabels],
) -> Result<(Tensor, [f64; 5])> {
    let Terms {
        active: [nll, bt, list, rt, abt],
        stability,
    } = terms(outputs, tokens, pad, legal, rows)?;
    let c = COEFFICIENTS;
    let mut loss = &nll * c.sequence_nll
        + &bt * c.ranking_bt
        + &list * c.ranking_listwise
        + &rt * c.runtime_huber
        + &abt * c.contextfree_bt
        + stability * c.stability;
    // Inactive optional objectives stay connected for identical packed gradient layouts.
    for x in loss_log_vars {
        loss += x * c.loss_log_vars;
    }
    let values = [
        nll.double_value(&[]),
        bt.double_value(&[]),
        list.double_value(&[]),
        rt.double_value(&[]),
        abt.double_value(&[]),
    ];
    // An inactive head overflowing to inf times 0.0 makes the scalar NaN while
    // every gradient stays finite; a nonfinite objective is never logged as a
    // completed microbatch or backpropagated (W6 G12).
    let total = loss.double_value(&[]);
    ensure!(
        total.is_finite() && values.iter().all(|x| x.is_finite()),
        "nonfinite training objective: total {total}, terms {values:?}"
    );
    Ok((loss, values))
}

/// Coefficient-weighted term values for telemetry, from the rank-mean raw
/// values. Uses the same coefficients as [`composite`].
pub fn weighted_losses(values: &[f32]) -> Value {
    let c = COEFFICIENTS;
    json!({"sequence_nll":values[0] * c.sequence_nll as f32,
        "ranking_bt":values[1] * c.ranking_bt as f32,
        "ranking_listwise":values[2] * c.ranking_listwise as f32,
        "runtime_huber":values[3] * c.runtime_huber as f32,
        "contextfree_bt":values[4] * c.contextfree_bt as f32})
}

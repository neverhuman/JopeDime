//! Lossless partition of one global batch across any number of ranks (W7-T06).
//!
//! Partition v1 (`task_contract::training_partition`) averages
//! microbatch-local means over microbatches and ranks. It needs
//! `global_batch % (world × microbatch) == 0` and a world in {1, 2, 4, 8}, so
//! no three-rank partition of the 2048-row batch exists. With unequal shards
//! it would also weight comparisons unequally: each rank-local mean would
//! count as much as any other, whatever its size.
//!
//! Partition v2 is a new version, never a reinterpretation of v1.
//! - **Comparisons stay whole.** The global batch is `G = global_batch / 8`
//!   complete comparisons, drawn by the same serial sampler at the same global
//!   cursors whatever the rank count, so the data of a step does not depend on
//!   the partition.
//! - **Deterministic, lossless assignment.** Rank `r` takes a contiguous block
//!   of `⌊G/W⌋ + [r < G mod W]` comparisons, so nothing is dropped. Each
//!   rank's block is split into microbatches of at most `microbatch_rows / 8`
//!   comparisons; the last microbatch may be smaller.
//! - **Global reduction.** Every term is reduced by its global numerator over
//!   its global denominator: `L = Σ_T c_T · N_T / D_T`.
//!   - The denominators depend only on labels ([`denominators`]), so every
//!     rank computes them for the whole global batch from the deterministic
//!     sampler, with no extra communication.
//!   - A microbatch's objective is `Σ_T c_T · N_T,mb / D_T` ([`composite`]),
//!     formed from the v1 microbatch-local means scaled by
//!     `D_T,mb / D_T`. A rank sums its microbatches without dividing by their
//!     count, and gradients are summed across ranks. With a mean-reducing
//!     transport each rank scales its objective by `W` ([`Plan::local_scale`]).
//! - **Empty cases are explicit.**
//!   - A rank with no comparisons has no microbatch. It backpropagates
//!     [`empty_rank_objective`], an exact zero connected to every parameter,
//!     so it contributes a defined zero gradient and still joins every
//!     collective with the same packed layout.
//!   - A term whose global denominator is zero is globally empty: value 0 and
//!     gradient 0, recorded in [`Denominators::empty_terms`]. For validated
//!     supervision only the two Bradley–Terry terms can be empty, when every
//!     pair ties.
//! - **Rank-local state.** The dropout RNG is counter based
//!   (`consumption::RNG_ID`, reset before every microbatch from seed, local
//!   step, rank and microbatch), and the sampler is a pure function of the
//!   global cursor. [`State`] persists both for every rank at a committed step,
//!   with the global committed update, and exact resume re-derives and checks
//!   them.
//! - **Rank-count changes are explicit transitions.** The objective does not
//!   depend on the rank count, but the RNG streams and shard boundaries do, so
//!   a different rank count or microbatch is a changed execution partition:
//!   exact resume refuses it, an optimizer-carry transition must declare it
//!   (`continuation::check_parent_partition`), and a weight warm start
//!   carries nothing. Exact equality with the single-process reference is
//!   claimed only with dropout off.
use super::objective::{self, row_weight, Outputs, RowLabels, COEFFICIENTS, COMPARISON_ROWS};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tch::{Kind, Tensor};

pub const SCHEMA: &str = "parallel.training_partition.v2";
pub const NORMALIZATION: &str = "global_term_sums_v1";
pub const ASSIGNMENT: &str = "contiguous_blocks_remainder_to_lowest_ranks";
pub const STATE_SCHEMA: &str = "parallel.partition_state.v1";
/// Frozen numerical policy against the declared single-process reference:
/// the v1 [`objective::composite`] over the whole global batch as one
/// microbatch, in the same f32 arithmetic. A partitioned objective must be
/// within `LOSS_RTOL · max(1, |reference|)`; each output gradient within
/// `GRADIENT_RTOL · max|reference gradient|` (so an exactly zero reference
/// gradient must be reproduced exactly).
pub const LOSS_RTOL: f64 = 1e-5;
pub const GRADIENT_RTOL: f64 = 1e-5;
/// Relative slack when comparing a local f64 denominator sum with the global
/// one; only rounding of the same positive summands can exceed it.
const DENOMINATOR_SLACK: f64 = 1e-9;

fn digest(value: &Value) -> String {
    crate::manifest::hash(&serde_json::to_vec(value).expect("JSON value serializes"))
}

fn is_sha256(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The lineage-B objective under v2 normalization. It differs from
/// [`objective::identity`] only in the reductions, which are global rather
/// than microbatch-local, so moving a lineage between v1 and v2 is an
/// objective change (a weight warm start), never an optimizer carry.
pub fn objective_identity() -> Value {
    let mut identity = objective::identity();
    identity["normalization"] = json!(NORMALIZATION);
    identity["sequence"]["batch_reduction"] = json!("row_weighted_mean_over_global_batch");
    identity["ranking"]["pair_reduction"] = json!("weighted_mean_over_global_batch");
    identity["ranking"]["listwise_reduction"] = json!("mean_over_global_eligible_groups");
    identity["ranking"]["empty_support"] = json!("globally_empty_term_is_zero");
    identity["runtime"]["reduction"] = json!("mean_over_global_rows");
    identity
}

pub fn objective_sha256() -> String {
    objective::identity_digest(&objective_identity())
}

/// One rank's contiguous share of the global comparisons.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shard {
    pub rank: usize,
    pub first_group: usize,
    pub groups: usize,
    /// Comparisons per microbatch, in order; empty for an empty rank.
    pub microbatches: Vec<usize>,
}

impl Shard {
    /// First global comparison of microbatch `mb`.
    pub fn microbatch_first_group(&self, mb: usize) -> usize {
        self.first_group + self.microbatches[..mb].iter().sum::<usize>()
    }
}

/// Assign `total_groups` comparisons to `world` ranks: contiguous blocks,
/// remainder to the lowest ranks, microbatches of at most `microbatch_groups`.
pub fn assign(total_groups: usize, world: usize, microbatch_groups: usize) -> Result<Vec<Shard>> {
    ensure!(
        world > 0 && microbatch_groups > 0,
        "partition needs ranks and a positive microbatch"
    );
    let (base, remainder) = (total_groups / world, total_groups % world);
    let mut first = 0;
    Ok((0..world)
        .map(|rank| {
            let groups = base + usize::from(rank < remainder);
            let mut microbatches = Vec::new();
            let mut left = groups;
            while left > 0 {
                let n = left.min(microbatch_groups);
                microbatches.push(n);
                left -= n;
            }
            let shard = Shard {
                rank,
                first_group: first,
                groups,
                microbatches,
            };
            first += groups;
            shard
        })
        .collect())
}

/// How the trainer's all-reduce combines rank gradients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    Sum,
    /// `neural::Nccl::reduce`: a sum divided by the world size.
    Mean,
}

/// One rank-local microbatch of one optimizer step.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Slot {
    pub rank: usize,
    pub microbatch: usize,
    /// First comparison, counted within the global batch of the step.
    pub first_comparison: usize,
    pub comparisons: usize,
    /// Global sampler cursor (row serial) of the first comparison.
    pub serial: u64,
    pub rows: usize,
    /// `consumption::dropout_seed(seed, local_step, rank, microbatch)`.
    pub dropout_seed: i64,
}

/// A validated v2 partition of one global batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    world: usize,
    global_batch: usize,
    microbatch_rows: usize,
    shards: Vec<Shard>,
}

impl Plan {
    /// `microbatch_rows` is the largest rank-local microbatch.
    pub fn new(world: usize, global_batch: usize, microbatch_rows: usize) -> Result<Self> {
        ensure!(
            global_batch > 0 && global_batch % COMPARISON_ROWS == 0,
            "global batch must be whole comparisons of {COMPARISON_ROWS}"
        );
        ensure!(
            microbatch_rows > 0 && microbatch_rows % COMPARISON_ROWS == 0,
            "microbatch must be whole comparisons of {COMPARISON_ROWS}"
        );
        let shards = assign(
            global_batch / COMPARISON_ROWS,
            world,
            microbatch_rows / COMPARISON_ROWS,
        )?;
        Ok(Self {
            world,
            global_batch,
            microbatch_rows,
            shards,
        })
    }

    pub fn world(&self) -> usize {
        self.world
    }

    pub fn global_batch(&self) -> usize {
        self.global_batch
    }

    pub fn microbatch_rows(&self) -> usize {
        self.microbatch_rows
    }

    pub fn comparisons(&self) -> usize {
        self.global_batch / COMPARISON_ROWS
    }

    pub fn shards(&self) -> &[Shard] {
        &self.shards
    }

    /// The execution-partition record, bound into checkpoints exactly as v1's
    /// record is. `world` may be any positive count, including 3.
    pub fn record(&self) -> Value {
        json!({
            "schema": SCHEMA,
            "world": self.world,
            "global_batch": self.global_batch,
            "microbatch": self.microbatch_rows,
            "comparisons": self.comparisons(),
            "assignment": ASSIGNMENT,
            "shards": self.shards,
            "normalization": NORMALIZATION,
            "gradient_reduction": "sum_of_rank_gradients_of_global_normalized_local_objectives",
            "mean_transport_scale": "world",
            "empty_rank": "explicit_zero_gradient_joins_every_collective",
            "empty_term": "globally_zero_denominator_gives_zero_value_and_gradient",
            "rng": super::consumption::RNG_ID,
            "sampler": "global_comparison_serial",
            "autocast": "scoped_bf16_fp32_master_v2",
            "numerical_policy": {
                "reference": "single_process_v1_composite_over_the_whole_global_batch",
                "arithmetic": "f32_terms_f64_denominators",
                "loss_rtol": LOSS_RTOL,
                "gradient_rtol_of_max_abs": GRADIENT_RTOL,
                "dropout": "off_for_equivalence_claims",
            },
        })
    }

    /// Global sampler cursor of the first row of `local_step`.
    pub fn global_serial(&self, local_step: u64) -> Result<u64> {
        local_step
            .checked_mul(self.global_batch as u64)
            .context("sampler cursor overflow")
    }

    /// Every microbatch rank `rank` runs at `local_step`, in order. Empty for
    /// an empty rank.
    pub fn slots(&self, rank: usize, seed: u64, local_step: u64) -> Result<Vec<Slot>> {
        let shard = self
            .shards
            .get(rank)
            .context("rank outside the partition")?;
        let base = self.global_serial(local_step)?;
        shard
            .microbatches
            .iter()
            .enumerate()
            .map(|(mb, &comparisons)| {
                let first = shard.microbatch_first_group(mb);
                Ok(Slot {
                    rank,
                    microbatch: mb,
                    first_comparison: first,
                    comparisons,
                    serial: base
                        .checked_add((first * COMPARISON_ROWS) as u64)
                        .context("sampler cursor overflow")?,
                    rows: comparisons * COMPARISON_ROWS,
                    dropout_seed: super::consumption::dropout_seed(seed, local_step, rank, mb),
                })
            })
            .collect()
    }

    /// Factor on each rank's objective (and term contributions) so that the
    /// transport delivers the sum of rank gradients.
    pub fn local_scale(&self, transport: Transport) -> f64 {
        match transport {
            Transport::Sum => 1.0,
            Transport::Mean => self.world as f64,
        }
    }

    fn rank_state(&self, rank: usize, seed: u64, next_local_step: u64) -> Result<RankState> {
        let next_slots = self.slots(rank, seed, next_local_step)?;
        Ok(RankState {
            rank,
            rng: super::consumption::RNG_ID.into(),
            next_local_step,
            sampler_cursor: next_slots.first().map(|s| s.serial),
            next_slots,
        })
    }

    /// The persisted partition state after the committed `update`.
    pub fn state(&self, seed: u64, sampling_offset: u64, update: CommittedUpdate) -> Result<State> {
        let partition = self.record();
        let next_local_step = update
            .optimizer_step
            .checked_sub(sampling_offset)
            .context("committed step precedes the sampling origin")?;
        let state = State {
            schema: STATE_SCHEMA.into(),
            partition_sha256: digest(&partition),
            partition,
            seed,
            sampling_offset,
            committed_step: update.optimizer_step,
            ranks: (0..self.world)
                .map(|rank| self.rank_state(rank, seed, next_local_step))
                .collect::<Result<_>>()?,
            committed_update: update,
        };
        state.verify(self)?;
        Ok(state)
    }
}

/// Global denominators of every term, computed from labels alone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Denominators {
    /// `Σ (0.05 + p⁴)` over all rows (row weights formed in f32, as on device).
    pub sequence_weight: f64,
    /// `Σ |Δv|·boost` over all non-tied pairs; shared by the ablation term.
    pub pair_weight: f64,
    /// Listwise-eligible comparisons (every validated comparison).
    pub listwise_groups: usize,
    pub runtime_rows: usize,
}

impl Denominators {
    /// Denominators in [`objective::TERM_NAMES`] order.
    pub fn by_term(&self) -> [f64; 5] {
        [
            self.sequence_weight,
            self.pair_weight,
            self.listwise_groups as f64,
            self.runtime_rows as f64,
            self.pair_weight,
        ]
    }

    /// Terms with a zero global denominator: value 0, gradient 0.
    pub fn empty_terms(&self) -> Vec<String> {
        objective::TERM_NAMES
            .iter()
            .zip(self.by_term())
            .filter(|(_, d)| *d == 0.0)
            .map(|(name, _)| name.to_string())
            .collect()
    }

    /// `D_T,local / D_T` for every term. A zero global denominator gives 0 and
    /// requires a zero local one; a local denominator above the global one
    /// means the global denominators do not cover this microbatch.
    pub fn fractions_of(&self, global: &Denominators) -> Result<[f64; 5]> {
        let (local, global) = (self.by_term(), global.by_term());
        let mut fractions = [0.0; 5];
        for i in 0..5 {
            let (l, g) = (local[i], global[i]);
            ensure!(
                l.is_finite() && g.is_finite() && l >= 0.0 && g >= 0.0,
                "{} denominator is not a finite non-negative sum",
                objective::TERM_NAMES[i]
            );
            ensure!(
                l <= g * (1.0 + DENOMINATOR_SLACK),
                "{} local denominator {l} exceeds the global {g}",
                objective::TERM_NAMES[i]
            );
            fractions[i] = if g == 0.0 { 0.0 } else { (l / g).min(1.0) };
        }
        Ok(fractions)
    }
}

pub fn denominators(rows: &[RowLabels]) -> Result<Denominators> {
    objective::validate_rows(rows)?;
    let mut pair_weight = 0.0;
    for comparison in rows.chunks(COMPARISON_ROWS) {
        for (a, left) in comparison.iter().enumerate() {
            for right in &comparison[a + 1..] {
                if left.value == right.value {
                    continue;
                }
                // Exactly `neural::ranking_losses`: f64 values, boost on the
                // larger f32 percentile.
                let boost =
                    if left.percentile.max(right.percentile) >= crate::neural::BOOST_THRESHOLD {
                        crate::neural::BOOST_FACTOR
                    } else {
                        1.0
                    };
                pair_weight += (f64::from(left.value) - f64::from(right.value)).abs() * boost;
            }
        }
    }
    Ok(Denominators {
        sequence_weight: rows
            .iter()
            .map(|r| f64::from(row_weight(r.percentile)))
            .sum(),
        pair_weight,
        listwise_groups: rows.len() / COMPARISON_ROWS,
        runtime_rows: rows.len(),
    })
}

/// The v2 objective of one rank-local microbatch and its contribution to each
/// global raw term value. Summed over every microbatch of every rank, the
/// objectives give `L` and the contributions give the global term values.
///
/// The term arithmetic is [`objective::terms`], shared with v1; each
/// microbatch-local mean `N_T,mb / D_T,mb` is scaled by `D_T,mb / D_T`.
pub fn composite<'a>(
    outputs: &Outputs<'_>,
    loss_log_vars: impl IntoIterator<Item = &'a Tensor>,
    tokens: &Tensor,
    pad: &Tensor,
    legal: &Tensor,
    rows: &[RowLabels],
    global: &Denominators,
) -> Result<(Tensor, [f64; 5])> {
    let fractions = denominators(rows)?.fractions_of(global)?;
    let objective::Terms { active, stability } =
        objective::terms(outputs, tokens, pad, legal, rows)?;
    let c = COEFFICIENTS;
    let coefficients = [
        c.sequence_nll,
        c.ranking_bt,
        c.ranking_listwise,
        c.runtime_huber,
        c.contextfree_bt,
    ];
    let mut loss = stability * c.stability;
    let mut contributions = [0.0; 5];
    for (i, term) in active.iter().enumerate() {
        let value = term.double_value(&[]);
        ensure!(
            value.is_finite(),
            "nonfinite {} term",
            objective::TERM_NAMES[i]
        );
        loss += term * (coefficients[i] * fractions[i]);
        contributions[i] = value * fractions[i];
    }
    // Inactive optional objectives stay connected for identical packed gradient layouts.
    for x in loss_log_vars {
        loss += x * c.loss_log_vars;
    }
    let total = loss.double_value(&[]);
    ensure!(
        total.is_finite(),
        "nonfinite training objective: total {total}, contributions {contributions:?}"
    );
    Ok((loss, contributions))
}

/// Objective of a rank without comparisons this step: an exact zero
/// connected to every given parameter, so backward defines a zero gradient
/// for each of them.
pub fn empty_rank_objective<'a>(
    parameters: impl IntoIterator<Item = &'a Tensor>,
) -> Result<Tensor> {
    parameters
        .into_iter()
        .map(|p| p.sum(Kind::Float) * 0.0)
        .reduce(|a, b| a + b)
        .context("an empty-rank objective needs the trainable parameters")
}

/// Whether a partitioned objective is within the frozen policy.
pub fn loss_within_policy(actual: f64, reference: f64) -> bool {
    actual.is_finite()
        && reference.is_finite()
        && (actual - reference).abs() <= LOSS_RTOL * reference.abs().max(1.0)
}

/// Whether one output gradient is within the frozen policy.
pub fn gradient_within_policy(actual: &[f64], reference: &[f64]) -> bool {
    let scale = reference.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    actual.len() == reference.len()
        && actual
            .iter()
            .zip(reference)
            .all(|(a, r)| a.is_finite() && (a - r).abs() <= GRADIENT_RTOL * scale)
}

/// Digest of a packed f32 gradient's little-endian bytes.
pub fn gradient_sha256(gradient: &Tensor) -> Result<String> {
    let values = Vec::<f32>::try_from(
        gradient
            .to_device(tch::Device::Cpu)
            .to_kind(Kind::Float)
            .flatten(0, -1),
    )?;
    Ok(crate::manifest::hash(
        &values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    ))
}

/// The global update a checkpoint commits: every rank holds the same values
/// after reduction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommittedUpdate {
    /// Optimizer step after this update.
    pub optimizer_step: u64,
    /// Sampler-local step the update consumed.
    pub local_step: u64,
    pub comparisons: usize,
    pub denominators: Denominators,
    pub empty_terms: Vec<String>,
    /// Global raw term values, [`objective::TERM_NAMES`] order.
    pub term_values: [f64; 5],
    pub loss: f64,
    /// Norm of the reduced gradient before clipping.
    pub gradient_norm: f64,
    pub clipped: bool,
    /// Digest of the reduced gradient as applied (after clipping).
    pub gradient_sha256: String,
    pub parameter_sha256: String,
}

impl CommittedUpdate {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        plan: &Plan,
        optimizer_step: u64,
        local_step: u64,
        global: &Denominators,
        term_values: [f64; 5],
        applied_gradient: &Tensor,
        gradient_norm: f64,
        parameter_sha256: &str,
    ) -> Result<Self> {
        let c = COEFFICIENTS;
        let update = Self {
            optimizer_step,
            local_step,
            comparisons: plan.comparisons(),
            denominators: global.clone(),
            empty_terms: global.empty_terms(),
            term_values,
            loss: weighted(&c, &term_values),
            gradient_norm,
            clipped: gradient_norm > 1.0,
            gradient_sha256: gradient_sha256(applied_gradient)?,
            parameter_sha256: parameter_sha256.into(),
        };
        update.verify(plan)?;
        Ok(update)
    }

    pub fn verify(&self, plan: &Plan) -> Result<()> {
        let d = &self.denominators;
        ensure!(
            self.comparisons == plan.comparisons()
                && d.listwise_groups == plan.comparisons()
                && d.runtime_rows == plan.global_batch(),
            "committed update does not cover the whole global batch"
        );
        ensure!(
            d.sequence_weight.is_finite()
                && d.sequence_weight > 0.0
                && d.pair_weight.is_finite()
                && d.pair_weight >= 0.0,
            "committed update denominators are invalid"
        );
        ensure!(
            self.empty_terms == d.empty_terms(),
            "committed update empty terms differ from its denominators"
        );
        for (i, value) in self.term_values.iter().enumerate() {
            ensure!(
                value.is_finite(),
                "committed update term {} is nonfinite",
                objective::TERM_NAMES[i]
            );
            ensure!(
                d.by_term()[i] != 0.0 || *value == 0.0,
                "globally empty term {} has a nonzero value",
                objective::TERM_NAMES[i]
            );
        }
        let expected = weighted(&COEFFICIENTS, &self.term_values);
        ensure!(
            loss_within_policy(self.loss, expected),
            "committed update loss {} differs from its terms {expected}",
            self.loss
        );
        ensure!(
            self.gradient_norm.is_finite()
                && self.gradient_norm >= 0.0
                && self.clipped == (self.gradient_norm > 1.0),
            "committed update gradient norm or clipping is inconsistent"
        );
        ensure!(
            is_sha256(&self.gradient_sha256) && is_sha256(&self.parameter_sha256),
            "committed update digests are malformed"
        );
        Ok(())
    }
}

fn weighted(c: &objective::Coefficients, values: &[f64; 5]) -> f64 {
    values[0] * c.sequence_nll
        + values[1] * c.ranking_bt
        + values[2] * c.ranking_listwise
        + values[3] * c.runtime_huber
        + values[4] * c.contextfree_bt
}

/// A rank's RNG and sampler state at a committed step: the next local step
/// and every microbatch it will run, with its cursor and dropout seed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankState {
    pub rank: usize,
    pub rng: String,
    pub next_local_step: u64,
    /// Cursor of the rank's next microbatch; `None` for an empty rank.
    pub sampler_cursor: Option<u64>,
    pub next_slots: Vec<Slot>,
}

/// Persisted v2 partition state of one committed checkpoint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub schema: String,
    pub partition: Value,
    pub partition_sha256: String,
    pub seed: u64,
    pub sampling_offset: u64,
    pub committed_step: u64,
    pub ranks: Vec<RankState>,
    pub committed_update: CommittedUpdate,
}

impl State {
    /// Check the persisted state against the requested plan by re-deriving
    /// every rank's RNG and sampler state.
    pub fn verify(&self, plan: &Plan) -> Result<()> {
        ensure!(
            self.schema == STATE_SCHEMA,
            "unknown partition state schema"
        );
        ensure!(
            self.partition == plan.record(),
            "the persisted execution partition differs from the requested one; a rank-count \
             or microbatch change is an explicit transition, never an exact resume"
        );
        ensure!(
            self.partition_sha256 == digest(&self.partition),
            "partition state digest differs"
        );
        self.committed_update.verify(plan)?;
        ensure!(
            self.committed_update.optimizer_step == self.committed_step
                && self
                    .committed_update
                    .local_step
                    .checked_add(1)
                    .and_then(|s| s.checked_add(self.sampling_offset))
                    == Some(self.committed_step),
            "the committed update does not produce the committed step"
        );
        let next = self.committed_step - self.sampling_offset;
        ensure!(
            self.ranks.len() == plan.world(),
            "partition state does not cover every rank"
        );
        for (rank, persisted) in self.ranks.iter().enumerate() {
            ensure!(
                *persisted == plan.rank_state(rank, self.seed, next)?,
                "rank {rank} RNG or sampler state differs from its derivation"
            );
        }
        Ok(())
    }

    /// Exact resume: the verified next microbatches of every rank.
    pub fn resume(&self, plan: &Plan, seed: u64) -> Result<Vec<Vec<Slot>>> {
        self.verify(plan)?;
        ensure!(
            self.seed == seed,
            "exact resume requires the persisted sampling seed"
        );
        Ok(self.ranks.iter().map(|r| r.next_slots.clone()).collect())
    }
}

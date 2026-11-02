use super::*;

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

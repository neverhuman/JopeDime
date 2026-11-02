use super::*;

pub const SCHEMA: &str = "jope.training_consumption.v1";
/// Identity of the per-microbatch dropout seed derivation, see [`dropout_seed`].
pub const RNG_ID: &str = "mix_seed_local_step_rank_microbatch_v1";

/// The seed passed to `tch::manual_seed` before each microbatch.
pub fn dropout_seed(seed: u64, local_step: u64, rank: usize, microbatch: usize) -> i64 {
    super::mix(
        seed ^ local_step.wrapping_mul(65537) ^ (rank as u64).wrapping_mul(257) ^ microbatch as u64,
    ) as i64
}

/// Sampler cursor of one rank-local microbatch: the global example index its
/// first comparison is drawn at.
pub fn sampler_serial(
    local_step: u64,
    global_batch: usize,
    world: usize,
    rank: usize,
    microbatch: usize,
    micro: usize,
) -> Result<u64> {
    ensure!(
        world > 0 && rank < world && global_batch % world == 0,
        "invalid sampler partition"
    );
    local_step
        .checked_mul(global_batch as u64)
        .and_then(|x| x.checked_add((rank * (global_batch / world)) as u64))
        .and_then(|x| x.checked_add((microbatch * micro) as u64))
        .context("sampler cursor overflow")
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub schema: String,
    pub rank: usize,
    pub world: usize,
    pub microbatch: usize,
    pub global_batch: usize,
    pub accumulation: usize,
    pub seed: u64,
    pub sampler_id: String,
    pub rng: String,
    /// Optimizer step at which this lineage's sampler counter is zero.
    pub sampling_offset: u64,
    pub initial_optimizer_step: u64,
    pub target_optimizer_step: u64,
    pub corpus_sha256: String,
    pub replay_policy_sha256: Option<String>,
    /// Digest of the checkpoint state this attempt resumed from, if any.
    pub resume_state_sha256: Option<String>,
}

impl Header {
    /// `shard` is the rank's partition-v2 shard; `None` is partition v1,
    /// whose equal shards satisfy `accumulation × world × microbatch ==
    /// global_batch`. Under v2 the shard must be exactly the plan's shard for
    /// this rank, and `accumulation` its microbatch count (0 for an empty rank).
    pub(super) fn validate(&self, shard: Option<&Shard>) -> Result<()> {
        ensure!(
            self.schema == SCHEMA && self.rng == RNG_ID,
            "unknown consumption journal schema or RNG identity"
        );
        match shard {
            None => ensure!(
                self.world > 0
                    && self.rank < self.world
                    && self.microbatch > 0
                    && self.microbatch % super::objective::COMPARISON_ROWS == 0
                    && self.accumulation > 0
                    && self
                        .accumulation
                        .checked_mul(self.world)
                        .and_then(|x| x.checked_mul(self.microbatch))
                        == Some(self.global_batch),
                "consumption journal partition is inconsistent"
            ),
            Some(shard) => {
                ensure!(
                    self.world > 0 && self.rank < self.world,
                    "consumption journal partition is inconsistent"
                );
                let plan =
                    super::partition::Plan::new(self.world, self.global_batch, self.microbatch)?;
                ensure!(
                    plan.shards().get(self.rank) == Some(shard)
                        && self.accumulation == shard.microbatches.len(),
                    "consumption journal shard differs from the partition-v2 plan"
                );
            }
        }
        ensure!(
            self.sampling_offset <= self.initial_optimizer_step
                && self.initial_optimizer_step <= self.target_optimizer_step,
            "consumption journal step window is inconsistent"
        );
        ensure!(
            is_sha256(&self.corpus_sha256)
                && self.replay_policy_sha256.as_deref().is_none_or(is_sha256)
                && self.resume_state_sha256.as_deref().is_none_or(is_sha256),
            "consumption journal identity digest is malformed"
        );
        Ok(())
    }

    /// Sampler cursor of `microbatch` within optimizer `step`.
    pub fn serial(&self, step: u64, microbatch: usize) -> Result<u64> {
        let local = step
            .checked_sub(self.sampling_offset)
            .context("step precedes the sampling origin")?;
        sampler_serial(
            local,
            self.global_batch,
            self.world,
            self.rank,
            microbatch,
            self.microbatch,
        )
    }

    pub fn dropout_seed(&self, step: u64, microbatch: usize) -> Result<i64> {
        let local = step
            .checked_sub(self.sampling_offset)
            .context("step precedes the sampling origin")?;
        Ok(dropout_seed(self.seed, local, self.rank, microbatch))
    }

    /// Sampler cursor under either partition: v1's [`Header::serial`], or
    /// the v2 global cursor `local_step × global_batch + 8 × first
    /// comparison of the microbatch`. `microbatch` may equal the shard's
    /// microbatch count (the cursor after the shard).
    pub fn serial_in(&self, shard: Option<&Shard>, step: u64, microbatch: usize) -> Result<u64> {
        let Some(shard) = shard else {
            return self.serial(step, microbatch);
        };
        ensure!(
            microbatch <= shard.microbatches.len(),
            "microbatch {microbatch} outside the rank's shard"
        );
        let local = step
            .checked_sub(self.sampling_offset)
            .context("step precedes the sampling origin")?;
        local
            .checked_mul(self.global_batch as u64)
            .and_then(|x| {
                x.checked_add(
                    (shard.microbatch_first_group(microbatch) * super::objective::COMPARISON_ROWS)
                        as u64,
                )
            })
            .context("sampler cursor overflow")
    }

    /// Rows of `microbatch`: the fixed v1 microbatch, or the v2 shard's.
    pub fn rows_in(&self, shard: Option<&Shard>, microbatch: usize) -> Result<usize> {
        match shard {
            None => Ok(self.microbatch),
            Some(shard) => Ok(shard
                .microbatches
                .get(microbatch)
                .context("microbatch outside the rank's shard")?
                * super::objective::COMPARISON_ROWS),
        }
    }
}

/// A rank's partition-v2 shard, recorded beside the v1 header fields.
pub use super::super::partition::Shard;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingUpdateOutcome {
    Applied,
    Rejected,
}

/// Compatibility name for the optimizer-update outcome.
pub type Outcome = TrainingUpdateOutcome;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
pub enum TrainingConsumptionEvent {
    Attempt {
        header: Header,
        /// Partition v2 only. Absent, and never serialized, under v1, so
        /// v1 journals are byte-identical to those written before v2.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        shard: Option<Shard>,
    },
    Microbatch {
        step: u64,
        microbatch: usize,
        serial: u64,
        dropout_seed: i64,
        records: Vec<u64>,
        supervision: Supervision,
    },
    Update {
        step: u64,
        outcome: TrainingUpdateOutcome,
        microbatches: usize,
        gradient_norm: Option<f64>,
        clipped: Option<bool>,
        reason: Option<String>,
    },
    Commit {
        step: u64,
        state_sha256: String,
    },
}

/// Compatibility alias for the domain-specific `TrainingConsumptionEvent` type.
pub type Event = TrainingConsumptionEvent;

impl TrainingConsumptionEvent {
    pub(super) fn key(&self) -> String {
        match self {
            Self::Attempt { .. } => "attempt".into(),
            Self::Microbatch {
                step, microbatch, ..
            } => format!("microbatch:{step}:{microbatch}"),
            Self::Update { step, .. } => format!("update:{step}"),
            Self::Commit { step, .. } => format!("commit:{step}"),
        }
    }
}

/// Consumption measured in three units. Presentations count sampled rows,
/// including a record presented again; supervised positions count targets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Units {
    pub microbatches: u64,
    pub presentations: u64,
    pub supervised_positions: u64,
}

impl Units {
    pub(super) fn add(&mut self, other: &Self) -> Result<()> {
        let sum = |a: u64, b: u64| a.checked_add(b).context("consumption counter overflow");
        *self = Self {
            microbatches: sum(self.microbatches, other.microbatches)?,
            presentations: sum(self.presentations, other.presentations)?,
            supervised_positions: sum(self.supervised_positions, other.supervised_positions)?,
        };
        Ok(())
    }
}

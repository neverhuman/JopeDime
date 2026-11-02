use super::*;

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

    /// The execution-partition record
    /// (`task_contract::training_partition_v2`), bound into checkpoints
    /// exactly as v1's record is. `world` may be any positive count,
    /// including 3.
    pub fn record(&self) -> Value {
        crate::task_contract::training_partition_v2(
            self.world,
            self.microbatch_rows,
            self.global_batch,
        )
        .expect("a validated plan has a record")
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

    pub(super) fn rank_state(
        &self,
        rank: usize,
        seed: u64,
        next_local_step: u64,
    ) -> Result<RankState> {
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

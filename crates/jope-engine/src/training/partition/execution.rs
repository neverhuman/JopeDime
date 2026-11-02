use super::*;

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

/// Which partition an attempt runs. Every existing request runs v1; v2 is
/// selected only by a frozen request whose execution-partition record names
/// it, never by default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Version {
    V1,
    V2,
}

impl Version {
    pub fn of_record(record: &Value) -> Result<Self> {
        match record["schema"].as_str() {
            Some(PARTITION_V1_SCHEMA) => Ok(Self::V1),
            Some(PARTITION_V2_SCHEMA) => Ok(Self::V2),
            _ => anyhow::bail!("unknown execution partition schema"),
        }
    }
}

/// The validated execution partition of one trainer attempt.
#[derive(Clone, Debug, PartialEq)]
pub struct Execution {
    version: Version,
    world: usize,
    microbatch: usize,
    global_batch: usize,
    record: Value,
    plan: Option<Plan>,
}

/// The trainer's topology gate, formerly inline in `training::rank_impl`.
///
/// - v1 keeps its exact gate: world in {1, 2, 4, 8}, the standard microbatch
///   (or the batch-qualification one) and `global % (world × micro) == 0`.
/// - v2 accepts 1 to 8 ranks with no divisibility requirement, the standard
///   microbatch and global batch, and never batch qualification.
pub fn execution(
    version: Version,
    world: usize,
    micro: usize,
    global: usize,
    batch_qualification: bool,
) -> Result<Execution> {
    let micro_ok = if batch_qualification {
        [256, 512, 1024, 2048].contains(&micro)
    } else {
        [64, 128, 256].contains(&micro)
    };
    ensure!(
        micro_ok && [2048, 4096, 8192, 16384].contains(&global),
        "invalid training topology"
    );
    let (record, plan) = match version {
        Version::V1 => {
            ensure!(
                [1, 2, 4, 8].contains(&world) && global % (world * micro) == 0,
                "invalid training topology"
            );
            (
                crate::task_contract::training_partition(world, micro, global)?,
                None,
            )
        }
        Version::V2 => {
            ensure!(
                !batch_qualification,
                "batch qualification runs partition v1 only"
            );
            ensure!(
                (1..=8).contains(&world),
                "partition v2 supports one to eight ranks"
            );
            let plan = Plan::new(world, global, micro)?;
            (plan.record(), Some(plan))
        }
    };
    Ok(Execution {
        version,
        world,
        microbatch: micro,
        global_batch: global,
        record,
        plan,
    })
}

impl Execution {
    /// The execution a frozen request declares, checked against the requested
    /// topology.
    pub fn declared(
        record: &Value,
        world: usize,
        micro: usize,
        global: usize,
        batch_qualification: bool,
    ) -> Result<Self> {
        let execution = execution(
            Version::of_record(record)?,
            world,
            micro,
            global,
            batch_qualification,
        )?;
        ensure!(
            execution.record == *record,
            "declared execution partition differs from the requested topology"
        );
        Ok(execution)
    }

    pub fn version(&self) -> Version {
        self.version
    }

    pub fn world(&self) -> usize {
        self.world
    }

    pub fn microbatch(&self) -> usize {
        self.microbatch
    }

    pub fn global_batch(&self) -> usize {
        self.global_batch
    }

    /// The execution-partition record bound into checkpoints.
    pub fn record(&self) -> &Value {
        &self.record
    }

    /// The v2 plan; `None` under v1.
    pub fn plan(&self) -> Option<&Plan> {
        self.plan.as_ref()
    }

    /// Rows each rank consumes per optimizer step.
    pub fn rows_per_rank(&self) -> Vec<usize> {
        match &self.plan {
            None => vec![self.global_batch / self.world; self.world],
            Some(plan) => plan
                .shards()
                .iter()
                .map(|s| s.groups * COMPARISON_ROWS)
                .collect(),
        }
    }

    /// Microbatches rank `rank` runs per optimizer step (0 for an empty rank).
    pub fn accumulation(&self, rank: usize) -> Result<usize> {
        match &self.plan {
            None => Ok(self.global_batch / (self.world * self.microbatch)),
            Some(plan) => Ok(plan
                .shards()
                .get(rank)
                .context("rank outside the partition")?
                .microbatches
                .len()),
        }
    }

    /// This rank's v2 shard, recorded in its consumption journal.
    pub fn shard(&self, rank: usize) -> Result<Option<Shard>> {
        self.plan
            .as_ref()
            .map(|plan| {
                plan.shards()
                    .get(rank)
                    .cloned()
                    .context("rank outside the partition")
            })
            .transpose()
    }

    /// The objective the partition's reductions define.
    pub fn objective_identity(&self) -> Value {
        match self.version {
            Version::V1 => objective::identity(),
            Version::V2 => objective_identity(),
        }
    }

    pub fn objective_sha256(&self) -> String {
        objective::identity_digest(&self.objective_identity())
    }
}

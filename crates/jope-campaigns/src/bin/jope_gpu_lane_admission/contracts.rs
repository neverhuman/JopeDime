const SENTINEL_SCHEMA: &str = "GpuLanePauseSentinel.v1";
const ARBITRATION_STATUS_MAX_AGE_MS: u128 = 2 * 60 * 1_000;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RoleArg {
    CohortProposer,
    AlternatingTrainer,
    BreadthEncoder,
}

impl From<RoleArg> for GpuLaneRole {
    fn from(value: RoleArg) -> Self {
        match value {
            RoleArg::CohortProposer => Self::CohortProposer,
            RoleArg::AlternatingTrainer => Self::AlternatingTrainer,
            RoleArg::BreadthEncoder => Self::BreadthEncoder,
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "jope-gpu-lane-admission",
    about = "Persist pause/hold/capacity/submit/release transactions for xbabe1/2/3"
)]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long, default_value = default_scq_str())]
    scq: PathBuf,
    #[arg(long)]
    once: bool,
    #[arg(
        long,
        default_value_t = (GPU_ADMISSION_RETRY_MS / 1_000) as u64,
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    interval_seconds: u64,
    /// Register one immutable request and exit.
    #[arg(long, value_enum, requires_all = ["spec", "job_id"])]
    register_role: Option<RoleArg>,
    #[arg(long)]
    spec: Option<PathBuf>,
    #[arg(long)]
    job_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct SchedulerJob {
    id: String,
    state: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct NodeCapacity {
    node: String,
    up: bool,
    gpus_free: u64,
    free_core_budget: i64,
    #[serde(default)]
    gpu_companion_core_reserve: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GpuArbitrationEvidence {
    candidate_evaluation_work: u64,
    candidate_evaluation_work_target: u64,
    protected_complete_waves: u64,
}

impl GpuArbitrationEvidence {
    fn runway_low(self) -> bool {
        self.candidate_evaluation_work < self.candidate_evaluation_work_target
            || self.protected_complete_waves < 2
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PauseSentinel {
    schema: String,
    created_at_ms: u128,
    updated_at_ms: u128,
    roles: Vec<GpuLaneRole>,
    held_evaluator_ids: Vec<String>,
    released_evaluator_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hold_inflight_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    release_inflight_id: Option<String>,
}

impl PauseSentinel {
    #[cfg(test)]
    fn new(roles: Vec<GpuLaneRole>) -> Self {
        let timestamp = now_ms();
        Self {
            schema: SENTINEL_SCHEMA.to_owned(),
            created_at_ms: timestamp,
            updated_at_ms: timestamp,
            roles,
            held_evaluator_ids: Vec::new(),
            released_evaluator_ids: Vec::new(),
            hold_inflight_id: None,
            release_inflight_id: None,
        }
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SENTINEL_SCHEMA,
            "GPU pause sentinel schema is not v1"
        );
        let held = self.held_evaluator_ids.iter().collect::<BTreeSet<_>>();
        let released = self.released_evaluator_ids.iter().collect::<BTreeSet<_>>();
        ensure!(
            held.len() == self.held_evaluator_ids.len(),
            "pause sentinel has duplicate held evaluator IDs"
        );
        ensure!(
            released.is_subset(&held),
            "pause sentinel released an evaluator it did not hold"
        );
        Ok(())
    }

    fn unreleased(&self) -> Vec<String> {
        let released = self
            .released_evaluator_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        self.held_evaluator_ids
            .iter()
            .filter(|id| !released.contains(id.as_str()))
            .cloned()
            .collect()
    }
}

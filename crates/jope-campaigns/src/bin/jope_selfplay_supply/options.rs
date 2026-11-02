use super::*;

#[derive(Parser, Debug)]
#[command(
    name = "jope-selfplay-supply",
    about = "Maintain a bounded reserve of immutable current-checkpoint micro-waves"
)]
pub(super) struct Args {
    #[arg(long, default_value = default_root_str())]
    pub(super) root: PathBuf,
    /// Build waves. Without this flag the supervisor is read-only.
    #[arg(long)]
    pub(super) build: bool,
    #[arg(long)]
    pub(super) once: bool,
    #[arg(long, default_value_t = 5)]
    pub(super) interval_seconds: u64,
    #[arg(long, default_value_t = 2_048)]
    pub(super) queue_floor: usize,
    #[arg(long, default_value_t = 2_048)]
    pub(super) queued_target: usize,
    #[arg(long, default_value_t = 4_096)]
    pub(super) queued_hard_cap: usize,
    /// Maintain this many current-checkpoint manifests beyond scheduler demand.
    #[arg(long, default_value_t = 2_048)]
    pub(super) ready_reserve_min: usize,
    /// Minimum queued/starting/dispatching plus checkpoint-valid ready work.
    #[arg(long, default_value_t = MIN_USEFUL_WORK)]
    pub(super) useful_work_min: usize,
    /// Canonical runway floor measured in candidate evaluations, not jobs.
    #[arg(long, default_value_t = MIN_CANDIDATE_EVALUATION_WORK)]
    pub(super) candidate_work_min: usize,
    /// Protected ready reserve measured in candidate evaluations.
    #[arg(long, default_value_t = MIN_CANDIDATE_RESERVE)]
    pub(super) candidate_reserve_min: usize,
    /// Newest complete waves protected from scheduler admission.
    #[arg(long, default_value_t = MIN_COMPLETE_RESERVE_WAVES)]
    pub(super) min_complete_reserve_waves: usize,
    /// Maximum current-checkpoint waves with unclaimed manifests.
    #[arg(long, default_value_t = 8)]
    pub(super) max_overlap_waves: usize,
    /// Exactly one wave is built before waiting for keeper acknowledgment.
    #[arg(long, default_value_t = 1)]
    pub(super) max_builds_per_cycle: usize,
    /// Production default is 50. A matched 25-vs-50 experiment remains gated.
    #[arg(long, default_value_t = 50)]
    pub(super) max_candidates_per_manifest: usize,
    #[arg(long, default_value_t = 30)]
    pub(super) queue_status_max_age_seconds: u64,
    #[arg(long, default_value_t = 300)]
    pub(super) performance_max_age_seconds: u64,
    #[arg(long, default_value = default_scq_str())]
    pub(super) scq: PathBuf,
    #[arg(
        long,
        default_value_os_t = paths::release_binary("jope_selfplay_wave")
    )]
    pub(super) wave_builder: PathBuf,
    #[arg(
        long,
        default_value_os_t = paths::research_script("test_exact_context_proposals.py")
    )]
    pub(super) proposal_generator: PathBuf,
    #[arg(
        long,
        default_value_os_t = paths::research_script("test_exact_dl_generate.py")
    )]
    pub(super) proposal_legacy_generator: PathBuf,
    #[arg(
        long,
        default_value_os_t = paths::research_script("test_exact_common.py")
    )]
    pub(super) proposal_common: PathBuf,
    #[arg(
        long,
        default_value_os_t = paths::runtime_bundle_pytorch()
    )]
    pub(super) proposal_pythonpath: PathBuf,
    #[arg(long)]
    pub(super) catalog_pointer: Option<PathBuf>,
    #[arg(long)]
    pub(super) checkpoint_pointer: Option<PathBuf>,
    #[arg(long)]
    pub(super) proposals_pointer: Option<PathBuf>,
    #[arg(long)]
    pub(super) corpus: Option<PathBuf>,
    #[arg(long)]
    pub(super) sealed_cohort: Option<PathBuf>,
    #[arg(long)]
    pub(super) robust_seed_bank: Option<PathBuf>,
    #[arg(long)]
    pub(super) pair_registry: Option<PathBuf>,
    #[arg(long)]
    pub(super) queue_status: Option<PathBuf>,
}

impl Args {
    pub(super) fn catalog_pointer(&self) -> PathBuf {
        match &self.catalog_pointer {
            Some(path) => path.clone(),
            None => self.root.join("catalog/current.json"),
        }
    }

    pub(super) fn proposals_pointer(&self) -> PathBuf {
        match &self.proposals_pointer {
            Some(path) => path.clone(),
            None => self.root.join("proposals/current.json"),
        }
    }

    pub(super) fn corpus(&self) -> PathBuf {
        match &self.corpus {
            Some(path) => path.clone(),
            None => self.root.join("corpus/strict-test-exact.sqlite"),
        }
    }

    pub(super) fn sealed_cohort(&self) -> PathBuf {
        match &self.sealed_cohort {
            Some(path) => path.clone(),
            None => self.root.join("catalog/private/sealed-current-v1.json"),
        }
    }

    pub(super) fn robust_seed_bank(&self) -> PathBuf {
        match &self.robust_seed_bank {
            Some(path) => path.clone(),
            None => self.root.join("corpus/robust-gp-seed-bank.v1.json"),
        }
    }

    pub(super) fn pair_registry(&self) -> PathBuf {
        match &self.pair_registry {
            Some(path) => path.clone(),
            None => self.root.join("ops/pair-registry.sqlite"),
        }
    }

    pub(super) fn queue_status(&self) -> PathBuf {
        match &self.queue_status {
            Some(path) => path.clone(),
            None => self.root.join("ops/queue-keeper-v3.json"),
        }
    }
}

#[derive(Debug, Deserialize)]
pub(super) struct QueueStatus {
    pub(super) schema: String,
    pub(super) generated_at_ms: u128,
    pub(super) admission_allowed: bool,
    #[serde(default)]
    pub(super) degraded: bool,
    #[serde(default)]
    pub(super) error: Option<String>,
    #[serde(default)]
    pub(super) fail_closed_reason: Option<String>,
    pub(super) active_checkpoint_sha256: Option<String>,
    pub(super) checkpoint_epoch: Option<String>,
    pub(super) queued_target: i64,
    pub(super) queued_hard_cap: i64,
    pub(super) queued: usize,
    pub(super) starting: usize,
    pub(super) queued_starting: usize,
    pub(super) dispatching: usize,
    pub(super) admission_waves: Vec<KeeperAdmissionWave>,
    pub(super) protected_reserve_waves: Vec<KeeperAdmissionWave>,
    pub(super) complete_current_checkpoint_waves: usize,
    pub(super) partial_current_checkpoint_waves: usize,
    pub(super) complete_reserve_waves: usize,
    pub(super) reserve_manifests: usize,
    pub(super) admission_manifests: usize,
    pub(super) validated_ready: usize,
    pub(super) useful_work: usize,
    pub(super) candidate_evaluation_work: u64,
    pub(super) unclaimed_current_checkpoint_manifests: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct KeeperAdmissionWave {
    pub(super) wave: String,
    pub(super) summary_sha256: String,
    pub(super) manifest_count: usize,
    pub(super) unclaimed_manifests: usize,
    pub(super) candidate_evaluation_count: u64,
    pub(super) complete: bool,
}

#[derive(Debug)]
pub(super) struct CheckpointInputs {
    pub(super) checkpoint_sha256: String,
    pub(super) checkpoint_pointer: PathBuf,
    pub(super) checkpoint_artifact: PathBuf,
    pub(super) proposals_pointer: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct ReadyWave {
    pub(super) wave: String,
    pub(super) summary_sha256: String,
    pub(super) manifests: usize,
    pub(super) unclaimed_manifests: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SupplyDecision {
    pub(super) below_floor: bool,
    pub(super) overloaded: bool,
    pub(super) candidate_evaluation_work: usize,
    pub(super) candidate_evaluation_work_target: usize,
    pub(super) estimated_manifests_per_wave: usize,
    pub(super) target_feasible: bool,
    pub(super) builds_planned: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PerformanceMeasurements {
    pub(super) generated_at_ms: u128,
    pub(super) p95_jobs_per_minute: f64,
    pub(super) p95_candidate_evaluations_per_minute: f64,
    pub(super) proposer_to_ready_minutes: f64,
    pub(super) candidate_evaluation_work_target: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct BuiltWave {
    pub(super) wave: String,
    pub(super) cohort_sha256: String,
    pub(super) proposal_pointer_sha256: String,
    pub(super) summary_sha256: String,
    pub(super) manifests: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct CycleReport {
    pub(super) schema: &'static str,
    pub(super) generated_at_ms: u128,
    pub(super) mode: &'static str,
    pub(super) active_checkpoint_sha256: String,
    pub(super) checkpoint_epoch: String,
    pub(super) queue_floor: usize,
    pub(super) queued_target: usize,
    pub(super) queued_hard_cap: usize,
    pub(super) ready_reserve_min: usize,
    pub(super) useful_work_min: usize,
    pub(super) useful_work_target: usize,
    pub(super) useful_work_target_formula: &'static str,
    pub(super) p95_jobs_per_minute: Option<f64>,
    pub(super) p95_candidate_evaluations_per_minute: Option<f64>,
    pub(super) proposer_to_ready_minutes: Option<f64>,
    pub(super) performance_generated_at_ms: Option<u128>,
    pub(super) performance_fresh: bool,
    pub(super) keeper_queued_target: i64,
    pub(super) keeper_queued_hard_cap: i64,
    pub(super) queued_starting: usize,
    pub(super) queued: usize,
    pub(super) starting: usize,
    pub(super) dispatching: usize,
    pub(super) queue_depth: usize,
    pub(super) below_floor: bool,
    pub(super) overloaded: bool,
    pub(super) current_ready_waves: Vec<ReadyWave>,
    pub(super) ready_manifests: usize,
    pub(super) validated_ready: usize,
    pub(super) reserve_manifests: usize,
    pub(super) admission_manifests: usize,
    pub(super) complete_current_checkpoint_waves: usize,
    pub(super) partial_current_checkpoint_waves: usize,
    pub(super) complete_reserve_waves: usize,
    pub(super) minimum_complete_reserve_waves: usize,
    pub(super) protected_reserve_waves: Vec<KeeperAdmissionWave>,
    pub(super) useful_work: usize,
    pub(super) candidate_work_min: usize,
    pub(super) candidate_reserve_min: usize,
    pub(super) candidate_evaluation_work: usize,
    pub(super) candidate_evaluation_work_target: usize,
    pub(super) protected_candidate_evaluations: usize,
    pub(super) target_feasible: bool,
    pub(super) critical_infeasibility_reason: Option<String>,
    pub(super) projected_jobs_before_build: usize,
    pub(super) projected_jobs_after_build: usize,
    pub(super) max_overlap_waves: usize,
    pub(super) builds_planned: usize,
    pub(super) builds_completed: Vec<BuiltWave>,
    pub(super) awaiting_keeper_ack: Option<String>,
    pub(super) awaiting_keeper_ack_since_ms: Option<u128>,
    pub(super) keeper_status_generated_at_ms: u128,
    pub(super) admission_allowed: bool,
    pub(super) hold_reason: Option<String>,
}

pub(super) struct PendingAck {
    pub(super) wave: String,
    pub(super) since_ms: u128,
}

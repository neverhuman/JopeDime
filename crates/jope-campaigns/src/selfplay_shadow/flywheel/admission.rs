use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuLaneRole {
    CohortProposer,
    AlternatingTrainer,
    BreadthEncoder,
}

impl GpuLaneRole {
    pub fn target_node(self) -> &'static str {
        match self {
            Self::CohortProposer => "xbabe1",
            Self::AlternatingTrainer => "xbabe2",
            Self::BreadthEncoder => "xbabe3",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::CohortProposer => "cohort_proposer",
            Self::AlternatingTrainer => "alternating_trainer",
            Self::BreadthEncoder => "breadth_encoder",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GpuResourceReservation {
    pub gpus: usize,
    pub companion_cpu_cores: usize,
}

impl Default for GpuResourceReservation {
    fn default() -> Self {
        Self {
            gpus: 1,
            companion_cpu_cores: GPU_COMPANION_CORES,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuAdmissionStage {
    Requested,
    QueuePaused,
    HoldingEvaluators,
    WaitingForCapacity,
    Submitted,
    ReleasingEvaluators,
    Complete,
    DependencyFailed,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryDisposition {
    ContinueRequest,
    ResumeHolding,
    ResumeCapacityWait,
    ConfirmSubmissionThenRelease,
    ResumeExactRelease,
    NothingToRecover,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GpuLaneAdmissionV1 {
    pub schema: String,
    pub role: GpuLaneRole,
    pub target_node: String,
    pub resource_reservation: GpuResourceReservation,
    pub immutable_spec_path: PathBuf,
    pub immutable_spec_sha256: Sha256Digest,
    pub immutable_job_identity: String,
    #[serde(default)]
    pub held_evaluator_ids: Vec<String>,
    #[serde(default)]
    pub released_evaluator_ids: Vec<String>,
    pub stage: GpuAdmissionStage,
    pub requested_at_ms: u128,
    pub updated_at_ms: u128,
    pub eligible_idle_since_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_capacity_backoff: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dependency_failure: Option<String>,
    pub recovery_disposition: RecoveryDisposition,
}

impl GpuLaneAdmissionV1 {
    pub fn new(
        role: GpuLaneRole,
        immutable_spec_path: PathBuf,
        immutable_spec_sha256: Sha256Digest,
        immutable_job_identity: String,
        now_ms: u128,
    ) -> Result<Self> {
        ensure!(
            !immutable_job_identity.trim().is_empty(),
            "GPU job identity is empty"
        );
        let value = Self {
            schema: GPU_LANE_ADMISSION_SCHEMA.to_owned(),
            role,
            target_node: role.target_node().to_owned(),
            resource_reservation: GpuResourceReservation::default(),
            immutable_spec_path,
            immutable_spec_sha256,
            immutable_job_identity,
            held_evaluator_ids: Vec::new(),
            released_evaluator_ids: Vec::new(),
            stage: GpuAdmissionStage::Requested,
            requested_at_ms: now_ms,
            updated_at_ms: now_ms,
            eligible_idle_since_ms: now_ms,
            last_capacity_backoff: None,
            dependency_failure: None,
            recovery_disposition: RecoveryDisposition::ContinueRequest,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == GPU_LANE_ADMISSION_SCHEMA,
            "GPU lane admission schema is not v1"
        );
        ensure!(
            self.target_node == self.role.target_node(),
            "GPU role/target-node reservation drift"
        );
        ensure!(
            self.resource_reservation == GpuResourceReservation::default(),
            "GPU lane must reserve one GPU and four companion cores"
        );
        ensure!(
            !self.immutable_job_identity.trim().is_empty(),
            "GPU job identity is empty"
        );
        let held = self.held_evaluator_ids.iter().collect::<BTreeSet<_>>();
        let released = self.released_evaluator_ids.iter().collect::<BTreeSet<_>>();
        ensure!(
            held.len() == self.held_evaluator_ids.len(),
            "held evaluator identity is duplicated"
        );
        ensure!(
            released.len() == self.released_evaluator_ids.len(),
            "released evaluator identity is duplicated"
        );
        ensure!(
            released.is_subset(&held),
            "an evaluator was released without an exact recorded hold"
        );
        if self.stage == GpuAdmissionStage::Complete {
            ensure!(
                held == released,
                "completed admission has asymmetric hold/release identities"
            );
        }
        Ok(())
    }

    pub fn record_held(&mut self, evaluator_id: String, now_ms: u128) -> Result<bool> {
        ensure!(
            matches!(
                self.stage,
                GpuAdmissionStage::QueuePaused
                    | GpuAdmissionStage::HoldingEvaluators
                    | GpuAdmissionStage::WaitingForCapacity
            ),
            "evaluator hold is not legal in the current admission stage"
        );
        if self.held_evaluator_ids.contains(&evaluator_id) {
            return Ok(false);
        }
        self.stage = GpuAdmissionStage::HoldingEvaluators;
        self.held_evaluator_ids.push(evaluator_id);
        self.held_evaluator_ids.sort();
        self.updated_at_ms = now_ms;
        self.recovery_disposition = RecoveryDisposition::ResumeHolding;
        self.validate()?;
        Ok(true)
    }

    pub fn record_released(&mut self, evaluator_id: &str, now_ms: u128) -> Result<bool> {
        ensure!(
            self.held_evaluator_ids.iter().any(|id| id == evaluator_id),
            "release identity was never held"
        );
        if self
            .released_evaluator_ids
            .iter()
            .any(|id| id == evaluator_id)
        {
            return Ok(false);
        }
        self.stage = GpuAdmissionStage::ReleasingEvaluators;
        self.released_evaluator_ids.push(evaluator_id.to_owned());
        self.released_evaluator_ids.sort();
        self.updated_at_ms = now_ms;
        self.recovery_disposition = RecoveryDisposition::ResumeExactRelease;
        self.validate()?;
        Ok(true)
    }

    pub fn unreleased_evaluator_ids(&self) -> Vec<&str> {
        let released = self
            .released_evaluator_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        self.held_evaluator_ids
            .iter()
            .map(String::as_str)
            .filter(|id| !released.contains(id))
            .collect()
    }

    pub fn recover(&mut self) {
        self.recovery_disposition = match self.stage {
            GpuAdmissionStage::Requested => RecoveryDisposition::ContinueRequest,
            GpuAdmissionStage::QueuePaused | GpuAdmissionStage::HoldingEvaluators => {
                RecoveryDisposition::ResumeHolding
            }
            GpuAdmissionStage::WaitingForCapacity => RecoveryDisposition::ResumeCapacityWait,
            GpuAdmissionStage::Submitted => RecoveryDisposition::ConfirmSubmissionThenRelease,
            GpuAdmissionStage::ReleasingEvaluators => RecoveryDisposition::ResumeExactRelease,
            GpuAdmissionStage::Complete | GpuAdmissionStage::DependencyFailed => {
                RecoveryDisposition::NothingToRecover
            }
        };
    }

    pub fn idle_alarm_due(&self, now_ms: u128) -> bool {
        !matches!(
            self.stage,
            GpuAdmissionStage::Complete | GpuAdmissionStage::DependencyFailed
        ) && now_ms.saturating_sub(self.eligible_idle_since_ms) > GPU_IDLE_ALARM_MS
    }
}

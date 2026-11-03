use super::*;

#[derive(Clone, Copy, Debug, serde::Deserialize, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SupplyIntentStateV2 {
    Planned,
    Proposing,
    Built,
    Published,
    KeeperAcked,
    Superseded,
    FailedRetryable,
    FailedPermanent,
}

impl SupplyIntentStateV2 {
    pub fn can_transition_to(self, next: Self) -> bool {
        match self {
            Self::Planned => matches!(
                next,
                Self::Proposing | Self::Superseded | Self::FailedPermanent
            ),
            Self::Proposing => matches!(
                next,
                Self::Built | Self::FailedRetryable | Self::FailedPermanent | Self::Superseded
            ),
            Self::Built => matches!(
                next,
                Self::Published | Self::FailedRetryable | Self::FailedPermanent | Self::Superseded
            ),
            Self::Published => matches!(
                next,
                Self::KeeperAcked | Self::FailedRetryable | Self::FailedPermanent
            ),
            Self::FailedRetryable => matches!(
                next,
                Self::Proposing
                    | Self::Built
                    | Self::Published
                    | Self::FailedPermanent
                    | Self::Superseded
            ),
            Self::KeeperAcked | Self::Superseded | Self::FailedPermanent => false,
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize, Eq, PartialEq, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct SupplyIntentV2 {
    pub schema: String,
    pub checkpoint_sha256: String,
    pub cohort_sha256: String,
    pub catalog_sha256: String,
    pub replay_protocol_sha256: String,
    pub task_activation_sha256: String,
    pub proposal_identity_sha256: String,
    pub seed_namespace_sha256: String,
    pub expected_manifest_count: u32,
    pub admission_budget: u32,
    pub state: SupplyIntentStateV2,
}

impl SupplyIntentV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SUPPLY_INTENT_SCHEMA,
            "invalid supply-intent schema"
        );
        for (value, field) in [
            (&self.checkpoint_sha256, "checkpoint"),
            (&self.cohort_sha256, "cohort"),
            (&self.catalog_sha256, "catalog"),
            (&self.replay_protocol_sha256, "replay protocol"),
            (&self.task_activation_sha256, "task activation"),
            (&self.proposal_identity_sha256, "proposal identity"),
            (&self.seed_namespace_sha256, "seed namespace"),
        ] {
            sha256(value, field)?;
        }
        ensure!(
            matches!(self.expected_manifest_count, 128 | 192 | 256),
            "supply intent has an unsupported manifest count"
        );
        ensure!(
            self.admission_budget >= self.expected_manifest_count,
            "supply admission budget is below its manifested work"
        );
        Ok(())
    }

    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::json!({
            "domain": "jope-supply-intent-v2",
            "checkpoint_sha256": self.checkpoint_sha256,
            "cohort_sha256": self.cohort_sha256,
            "catalog_sha256": self.catalog_sha256,
            "replay_protocol_sha256": self.replay_protocol_sha256,
            "task_activation_sha256": self.task_activation_sha256,
            "proposal_identity_sha256": self.proposal_identity_sha256,
            "seed_namespace_sha256": self.seed_namespace_sha256,
            "expected_manifest_count": self.expected_manifest_count,
            "admission_budget": self.admission_budget,
        }))
    }

    pub fn transition(&mut self, next: SupplyIntentStateV2) -> Result<()> {
        ensure!(
            self.state.can_transition_to(next),
            "invalid supply-intent transition {:?} -> {:?}",
            self.state,
            next
        );
        self.state = next;
        Ok(())
    }
}

#[derive(Clone, Debug, serde::Deserialize, Eq, PartialEq, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineEdgeV1 {
    pub schema: String,
    pub edge_id: String,
    pub producer: String,
    pub consumer: String,
    pub accepted_schemas: Vec<String>,
    pub claim_ttl_seconds: u64,
    pub runnable_slo_seconds: u64,
    pub progress_slo_seconds: u64,
    pub terminal_result_schemas: Vec<String>,
}

impl PipelineEdgeV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == PIPELINE_EDGE_SCHEMA,
            "invalid pipeline-edge schema"
        );
        for (value, field) in [
            (&self.edge_id, "edge ID"),
            (&self.producer, "producer"),
            (&self.consumer, "consumer"),
        ] {
            ensure!(!value.trim().is_empty(), "pipeline {field} is empty");
        }
        ensure!(
            !self.accepted_schemas.is_empty() && sorted_unique(&self.accepted_schemas),
            "accepted schemas are empty, duplicated, or unsorted"
        );
        ensure!(
            !self.terminal_result_schemas.is_empty()
                && sorted_unique(&self.terminal_result_schemas),
            "terminal result schemas are empty, duplicated, or unsorted"
        );
        ensure!(
            self.claim_ttl_seconds > 0
                && self.runnable_slo_seconds > 0
                && self.progress_slo_seconds > 0,
            "pipeline edge has a zero SLO"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, serde::Deserialize, Eq, PartialEq, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkIntentV1 {
    pub schema: String,
    pub intent_sha256: String,
    pub edge_id: String,
    pub payload_schema: String,
    pub payload_sha256: String,
    pub created_at_unix_ms: u64,
}

impl WorkIntentV1 {
    pub fn validate_for(&self, edge: &PipelineEdgeV1) -> Result<()> {
        ensure!(
            self.schema == WORK_INTENT_SCHEMA,
            "invalid work-intent schema"
        );
        sha256(&self.intent_sha256, "work intent")?;
        sha256(&self.payload_sha256, "work payload")?;
        ensure!(
            self.created_at_unix_ms > 0,
            "work intent creation time is absent"
        );
        ensure!(
            self.edge_id == edge.edge_id && edge.accepted_schemas.contains(&self.payload_schema),
            "work intent is incompatible with its edge"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, serde::Deserialize, Eq, PartialEq, serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum PipelineStageStateV2 {
    Running,
    Idle,
    Waiting {
        dependency: String,
        deadline_unix_ms: u64,
    },
    Paused {
        policy: String,
        unblock_condition: String,
    },
    BlockedCapacity,
    NoCompatibleConsumer,
    ConsumerUnavailable,
    ClaimStalled,
    FailedRetryable,
    FailedPermanent,
}

#[derive(Clone, Debug, serde::Deserialize, Eq, PartialEq, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConsumerHeartbeatV1 {
    pub schema: String,
    pub edge_id: String,
    pub consumer: String,
    pub state: PipelineStageStateV2,
    pub process_heartbeat_unix_ms: u64,
    pub last_state_transition_unix_ms: u64,
    pub last_semantic_progress_unix_ms: u64,
    pub oldest_runnable_dependency_unix_ms: Option<u64>,
    /// Age of the oldest unresolved dependency at the process heartbeat.
    /// This is explicit so consumers never mistake a heartbeat for progress.
    pub oldest_unresolved_dependency_age_ms: Option<u64>,
    pub claimed_intent_sha256: Option<String>,
    pub claim_expires_unix_ms: Option<u64>,
}

impl ConsumerHeartbeatV1 {
    pub fn validate_at(&self, edge: &PipelineEdgeV1, now_unix_ms: u64) -> Result<()> {
        ensure!(
            self.schema == CONSUMER_HEARTBEAT_SCHEMA,
            "invalid consumer-heartbeat schema"
        );
        ensure!(
            self.edge_id == edge.edge_id && self.consumer == edge.consumer,
            "consumer heartbeat/edge mismatch"
        );
        ensure!(
            self.process_heartbeat_unix_ms <= now_unix_ms
                && self.last_state_transition_unix_ms <= now_unix_ms
                && self.last_semantic_progress_unix_ms <= now_unix_ms,
            "consumer heartbeat contains a future clock"
        );
        ensure!(
            self.last_state_transition_unix_ms <= self.process_heartbeat_unix_ms
                && self.last_semantic_progress_unix_ms <= self.process_heartbeat_unix_ms,
            "state transition or semantic progress is newer than process heartbeat"
        );
        match (
            self.oldest_runnable_dependency_unix_ms,
            self.oldest_unresolved_dependency_age_ms,
        ) {
            (Some(oldest), Some(age)) => ensure!(
                oldest <= self.process_heartbeat_unix_ms
                    && age == self.process_heartbeat_unix_ms.saturating_sub(oldest),
                "oldest unresolved dependency age is inconsistent"
            ),
            (None, None) => {}
            _ => bail!("oldest unresolved dependency timestamp and age are not paired"),
        }
        match (&self.claimed_intent_sha256, self.claim_expires_unix_ms) {
            (Some(intent), Some(expires)) => {
                sha256(intent, "claimed work intent")?;
                if matches!(self.state, PipelineStageStateV2::Running) {
                    ensure!(
                        expires >= now_unix_ms,
                        "running consumer has an expired claim"
                    );
                    ensure!(
                        now_unix_ms.saturating_sub(self.last_semantic_progress_unix_ms)
                            <= edge.progress_slo_seconds.saturating_mul(1_000),
                        "running consumer has stale semantic progress"
                    );
                }
            }
            (None, None) => ensure!(
                !matches!(
                    self.state,
                    PipelineStageStateV2::Running | PipelineStageStateV2::ClaimStalled
                ),
                "running or stalled consumer lacks a claim"
            ),
            _ => bail!("consumer claim identity and expiry are not paired"),
        }
        if let PipelineStageStateV2::Waiting {
            dependency,
            deadline_unix_ms,
        } = &self.state
        {
            ensure!(
                !dependency.trim().is_empty() && *deadline_unix_ms >= now_unix_ms,
                "waiting state lacks a live dependency deadline"
            );
        }
        if let PipelineStageStateV2::Paused {
            policy,
            unblock_condition,
        } = &self.state
        {
            ensure!(
                !policy.trim().is_empty() && !unblock_condition.trim().is_empty(),
                "paused state lacks policy or unblock condition"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, serde::Deserialize, Eq, PartialEq, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct JopePipelineStatusV2 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub valid_until_unix_ms: u64,
    pub protected_complete_wave_count: u64,
    pub manifest_count: u64,
    pub candidate_evaluation_count: u64,
    pub predicted_cpu_seconds: u64,
    pub edges: Vec<PipelineEdgeV1>,
    pub consumers: Vec<ConsumerHeartbeatV1>,
}

impl JopePipelineStatusV2 {
    pub fn validate_at(&self, now_unix_ms: u64) -> Result<()> {
        ensure!(
            self.schema == PIPELINE_STATUS_SCHEMA,
            "invalid pipeline-status schema"
        );
        ensure!(
            self.generated_at_unix_ms > 0 && self.generated_at_unix_ms <= now_unix_ms,
            "pipeline status timestamp is absent or future-dated"
        );
        ensure!(
            self.valid_until_unix_ms > self.generated_at_unix_ms
                && now_unix_ms <= self.valid_until_unix_ms,
            "pipeline status is stale or has an invalid freshness window"
        );
        ensure!(
            (self.manifest_count == 0
                && self.candidate_evaluation_count == 0
                && self.predicted_cpu_seconds == 0
                && self.protected_complete_wave_count == 0)
                || (self.manifest_count > 0
                    && self.candidate_evaluation_count >= self.manifest_count
                    && self.predicted_cpu_seconds > 0),
            "pipeline work accounting is inconsistent"
        );
        ensure!(!self.edges.is_empty(), "pipeline has no edges");
        let mut edge_ids = BTreeSet::new();
        for edge in &self.edges {
            edge.validate()?;
            ensure!(
                edge_ids.insert(edge.edge_id.as_str()),
                "pipeline repeats an edge ID"
            );
            let matches = self
                .consumers
                .iter()
                .filter(|heartbeat| heartbeat.edge_id == edge.edge_id)
                .collect::<Vec<_>>();
            ensure!(
                matches.len() == 1,
                "pipeline edge does not have exactly one consumer heartbeat"
            );
            ensure!(
                matches[0].process_heartbeat_unix_ms <= self.generated_at_unix_ms,
                "consumer heartbeat is newer than its pipeline status"
            );
            matches[0].validate_at(edge, now_unix_ms)?;
        }
        ensure!(
            self.consumers.len() == self.edges.len(),
            "pipeline has orphan consumer heartbeats"
        );
        Ok(())
    }

    /// Preflight a producer-to-consumer route before production work exists.
    pub fn preflight_route(
        &self,
        producer: &str,
        consumer: &str,
        now_unix_ms: u64,
    ) -> Result<Vec<String>> {
        self.validate_at(now_unix_ms)?;
        let mut queue = VecDeque::from([(producer.to_owned(), Vec::<String>::new())]);
        let mut visited = BTreeSet::new();
        while let Some((stage, path)) = queue.pop_front() {
            if stage == consumer {
                return Ok(path);
            }
            if !visited.insert(stage.clone()) {
                continue;
            }
            for edge in self.edges.iter().filter(|edge| edge.producer == stage) {
                let heartbeat = self
                    .consumers
                    .iter()
                    .find(|row| row.edge_id == edge.edge_id)
                    .expect("validated one heartbeat per edge");
                ensure!(
                    !matches!(
                        heartbeat.state,
                        PipelineStageStateV2::NoCompatibleConsumer
                            | PipelineStageStateV2::ConsumerUnavailable
                            | PipelineStageStateV2::FailedPermanent
                    ),
                    "pipeline route is unavailable at {}",
                    edge.edge_id
                );
                let mut next_path = path.clone();
                next_path.push(edge.edge_id.clone());
                queue.push_back((edge.consumer.clone(), next_path));
            }
        }
        bail!("no complete pipeline route from {producer} to {consumer}")
    }
}

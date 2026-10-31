use super::{
    CacheDisposition, EncodingAdoptionReceiptV2, EvaluationKeyV2, EvaluationResultV2, Origin,
    Sha256Digest, Task, ACCEPTED_BY_TASK_SCHEMA, MIN_PARENT_DIRECT_OUTCOMES, RUNTIME_BUNDLE_SCHEMA,
    SUPPLY_STATE_SCHEMA,
};
#[cfg(test)]
use super::{EncodingAdoptionDisposition, EncodingDatasetIdentity};
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CohortSupplyStage {
    FreezeCohort,
    SubmitCohortProposer,
    FetchValidateProposals,
    CacheFilterResample,
    BuildWave,
    KeeperAcknowledgment,
}

impl CohortSupplyStage {
    fn ordinal(self) -> u8 {
        match self {
            Self::FreezeCohort => 0,
            Self::SubmitCohortProposer => 1,
            Self::FetchValidateProposals => 2,
            Self::CacheFilterResample => 3,
            Self::BuildWave => 4,
            Self::KeeperAcknowledgment => 5,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CohortSupplyStateV1 {
    pub schema: String,
    pub task: Task,
    pub accepted_checkpoint_sha256: Sha256Digest,
    pub cohort_sha256: Sha256Digest,
    pub catalog_sha256: Sha256Digest,
    pub decoder_configuration_sha256: Sha256Digest,
    pub seed: u64,
    pub stage: CohortSupplyStage,
    pub revision: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposal_job_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proposal_pointer_sha256: Option<Sha256Digest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wave_id: Option<String>,
}

impl CohortSupplyStateV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        task: Task,
        accepted_checkpoint_sha256: Sha256Digest,
        cohort_sha256: Sha256Digest,
        catalog_sha256: Sha256Digest,
        decoder_configuration_sha256: Sha256Digest,
        seed: u64,
    ) -> Self {
        Self {
            schema: SUPPLY_STATE_SCHEMA.to_owned(),
            task,
            accepted_checkpoint_sha256,
            cohort_sha256,
            catalog_sha256,
            decoder_configuration_sha256,
            seed,
            stage: CohortSupplyStage::FreezeCohort,
            revision: 0,
            proposal_job_id: None,
            proposal_pointer_sha256: None,
            wave_id: None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SUPPLY_STATE_SCHEMA,
            "cohort supply state schema is not v1"
        );
        if self.stage >= CohortSupplyStage::SubmitCohortProposer {
            ensure!(
                self.proposal_job_id
                    .as_deref()
                    .is_some_and(|value| !value.trim().is_empty()),
                "submitted supply state is missing its proposer job identity"
            );
        }
        if self.stage >= CohortSupplyStage::FetchValidateProposals {
            ensure!(
                self.proposal_pointer_sha256.is_some(),
                "validated supply state is missing its proposal pointer"
            );
        }
        if self.stage >= CohortSupplyStage::BuildWave {
            ensure!(
                self.wave_id
                    .as_deref()
                    .is_some_and(|value| !value.trim().is_empty()),
                "built supply state is missing its wave identity"
            );
        }
        Ok(())
    }

    pub fn advance(
        &mut self,
        next: CohortSupplyStage,
        proposal_job_id: Option<String>,
        proposal_pointer_sha256: Option<Sha256Digest>,
        wave_id: Option<String>,
    ) -> Result<bool> {
        ensure!(
            next.ordinal() == self.stage.ordinal() || next.ordinal() == self.stage.ordinal() + 1,
            "cohort supply stage must advance exactly once"
        );
        set_once(&mut self.proposal_job_id, proposal_job_id, "proposal job")?;
        set_once(
            &mut self.proposal_pointer_sha256,
            proposal_pointer_sha256,
            "proposal pointer",
        )?;
        set_once(&mut self.wave_id, wave_id, "wave")?;
        if next == self.stage {
            self.validate()?;
            return Ok(false);
        }
        self.stage = next;
        self.revision += 1;
        self.validate()?;
        Ok(true)
    }
}

fn set_once<T: Eq>(slot: &mut Option<T>, value: Option<T>, label: &str) -> Result<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if let Some(existing) = slot {
        ensure!(existing == &value, "{label} identity cannot change");
    } else {
        *slot = Some(value);
    }
    Ok(())
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncodingAdoptionLedgerV2 {
    #[serde(default)]
    pub successful: BTreeMap<String, EncodingAdoptionReceiptV2>,
    #[serde(default)]
    pub latest_failures: BTreeMap<String, EncodingAdoptionReceiptV2>,
}

impl EncodingAdoptionLedgerV2 {
    fn key(receipt: &EncodingAdoptionReceiptV2) -> String {
        format!(
            "{}:{}",
            receipt.dataset.dataset_id, receipt.dataset.content_sha256
        )
    }

    pub fn validate(&self) -> Result<()> {
        for (key, receipt) in &self.successful {
            receipt.validate()?;
            ensure!(
                receipt.succeeded(),
                "successful adoption contains a failure"
            );
            ensure!(*key == Self::key(receipt), "successful adoption key drift");
        }
        for (key, receipt) in &self.latest_failures {
            receipt.validate()?;
            ensure!(!receipt.succeeded(), "failure ledger contains a success");
            ensure!(*key == Self::key(receipt), "encoding failure key drift");
            ensure!(
                !self.successful.contains_key(key),
                "successful dataset cannot retain a failure disposition"
            );
        }
        Ok(())
    }

    pub fn adopt(&mut self, receipt: EncodingAdoptionReceiptV2) -> Result<bool> {
        receipt.validate()?;
        let key = Self::key(&receipt);
        if let Some(success) = self.successful.get(&key) {
            ensure!(
                success == &receipt || !receipt.succeeded(),
                "a successful dataset adoption cannot be replaced"
            );
            return Ok(false);
        }
        if receipt.succeeded() {
            self.latest_failures.remove(&key);
            self.successful.insert(key, receipt);
            self.validate()?;
            return Ok(true);
        }
        if let Some(previous) = self.latest_failures.get(&key) {
            ensure!(
                receipt.attempt_number >= previous.attempt_number,
                "encoding failure attempts cannot move backward"
            );
            if receipt == *previous {
                return Ok(false);
            }
        }
        self.latest_failures.insert(key, receipt);
        self.validate()?;
        Ok(true)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptedFront {
    pub task: Task,
    pub generation: u64,
    pub checkpoint_sha256: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_checkpoint_sha256: Option<Sha256Digest>,
    pub promotion_evidence_sha256: Sha256Digest,
    pub promoted_at_ms: u64,
}

impl AcceptedFront {
    pub fn validate(&self, expected_task: Task) -> Result<()> {
        ensure!(self.task == expected_task, "accepted task branch mismatch");
        ensure!(self.promoted_at_ms > 0, "promotion timestamp is absent");
        if self.generation == 0 {
            ensure!(
                self.parent_checkpoint_sha256.is_none(),
                "generation zero cannot have an accepted parent"
            );
        } else {
            ensure!(
                self.parent_checkpoint_sha256.is_some(),
                "accepted child is missing its parent"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
/// The schema-v2 accepted-checkpoint pointers for binary and regression tasks.
///
/// New task families require a later schema; this type intentionally keeps only
/// the two fronts represented by the current wire format.
pub struct AcceptedByTaskV2 {
    /// Wire-format schema identifier.
    pub schema: String,
    /// Accepted checkpoint and promotion metadata for the binary task.
    pub binary: AcceptedFront,
    /// Accepted checkpoint and promotion metadata for the regression task.
    pub regression: AcceptedFront,
}

impl AcceptedByTaskV2 {
    /// Builds and validates a schema-v2 pointer from its two supported fronts.
    pub fn new(binary: AcceptedFront, regression: AcceptedFront) -> Result<Self> {
        let value = Self {
            schema: ACCEPTED_BY_TASK_SCHEMA.to_owned(),
            binary,
            regression,
        };
        value.validate()?;
        Ok(value)
    }

    /// Checks the schema identifier and each task-specific accepted front.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == ACCEPTED_BY_TASK_SCHEMA,
            "accepted pointer schema is not v2"
        );
        self.binary.validate(Task::Binary)?;
        self.regression.validate(Task::Regression)?;
        Ok(())
    }

    /// Returns the accepted front for a binary or regression task.
    ///
    /// # Panics
    /// Panics for timeseries and multiclass tasks because schema v2 has no
    /// corresponding accepted-front fields. Use a newer schema for those tasks.
    pub fn front(&self, task: Task) -> &AcceptedFront {
        match task {
            Task::Binary => &self.binary,
            Task::Regression => &self.regression,
            // AcceptedByTask.v2 is a two-front wire schema; new-task fronts
            // arrive with a future schema revision. No caller can reach this
            // today (no new-task anchors exist), and reaching it would mean a
            // routing bug, so fail loudly rather than mislabel a front.
            Task::Timeseries | Task::Multiclass => {
                panic!("AcceptedByTask.v2 has no {task} front")
            }
        }
    }

    /// Replaces one supported task front if its current checkpoint matches the
    /// expected parent, recording the child and its evidence.
    pub fn compare_and_swap(
        &mut self,
        task: Task,
        expected_parent: &Sha256Digest,
        child_checkpoint: Sha256Digest,
        evidence_sha256: Sha256Digest,
        promoted_at_ms: u64,
    ) -> Result<()> {
        self.validate()?;
        let front = match task {
            Task::Binary => &mut self.binary,
            Task::Regression => &mut self.regression,
            Task::Timeseries | Task::Multiclass => {
                bail!("AcceptedByTask.v2 has no {task} front to swap")
            }
        };
        ensure!(
            &front.checkpoint_sha256 == expected_parent,
            "accepted task pointer changed before compare-and-swap"
        );
        ensure!(
            child_checkpoint != *expected_parent,
            "accepted child equals its parent"
        );
        *front = AcceptedFront {
            task,
            generation: front.generation + 1,
            checkpoint_sha256: child_checkpoint,
            parent_checkpoint_sha256: Some(expected_parent.clone()),
            promotion_evidence_sha256: evidence_sha256,
            promoted_at_ms,
        };
        self.validate()
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LineageStage {
    AcceptedParent,
    PolicyActions,
    Evaluations,
    CorpusShards,
    Candidate,
    Tournament,
    AcceptedChild,
    NextActions,
}

impl LineageStage {
    fn ordinal(self) -> u8 {
        match self {
            Self::AcceptedParent => 0,
            Self::PolicyActions => 1,
            Self::Evaluations => 2,
            Self::CorpusShards => 3,
            Self::Candidate => 4,
            Self::Tournament => 5,
            Self::AcceptedChild => 6,
            Self::NextActions => 7,
        }
    }
}

impl FromStr for LineageStage {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "accepted_parent" => Ok(Self::AcceptedParent),
            "policy_actions" => Ok(Self::PolicyActions),
            "evaluations" => Ok(Self::Evaluations),
            "corpus_shards" => Ok(Self::CorpusShards),
            "candidate" => Ok(Self::Candidate),
            "tournament" => Ok(Self::Tournament),
            "accepted_child" => Ok(Self::AcceptedChild),
            "next_actions" => Ok(Self::NextActions),
            _ => bail!("unknown causal lineage stage"),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LineageEventV2 {
    pub schema: String,
    pub sequence: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_event_sha256: Option<Sha256Digest>,
    pub event_sha256: Sha256Digest,
    pub task: Task,
    pub generation: u64,
    pub parent_checkpoint_sha256: Sha256Digest,
    pub stage: LineageStage,
    pub artifact_sha256: Sha256Digest,
    pub occurred_at_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
struct LineageEventPayload<'a> {
    schema: &'static str,
    sequence: u64,
    previous_event_sha256: &'a Option<Sha256Digest>,
    task: Task,
    generation: u64,
    parent_checkpoint_sha256: &'a Sha256Digest,
    stage: LineageStage,
    artifact_sha256: &'a Sha256Digest,
    occurred_at_ms: u64,
}

impl LineageEventV2 {
    fn calculate_hash(&self) -> Sha256Digest {
        let payload = LineageEventPayload {
            schema: "CausalLineageEvent.v2",
            sequence: self.sequence,
            previous_event_sha256: &self.previous_event_sha256,
            task: self.task,
            generation: self.generation,
            parent_checkpoint_sha256: &self.parent_checkpoint_sha256,
            stage: self.stage,
            artifact_sha256: &self.artifact_sha256,
            occurred_at_ms: self.occurred_at_ms,
        };
        Sha256Digest::of_bytes(
            &serde_json::to_vec(&payload).expect("lineage event serialization cannot fail"),
        )
    }

    pub fn validate_hash(&self) -> Result<()> {
        ensure!(
            self.schema == "CausalLineageEvent.v2",
            "lineage event schema is not v2"
        );
        ensure!(
            self.event_sha256 == self.calculate_hash(),
            "lineage event hash mismatch"
        );
        ensure!(self.occurred_at_ms > 0, "lineage timestamp is absent");
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalLineage {
    #[serde(default)]
    pub events: Vec<LineageEventV2>,
}

impl CausalLineage {
    pub fn append(
        &mut self,
        task: Task,
        generation: u64,
        parent_checkpoint_sha256: Sha256Digest,
        stage: LineageStage,
        artifact_sha256: Sha256Digest,
        occurred_at_ms: u64,
    ) -> Result<Sha256Digest> {
        self.validate()?;
        let prior_for_generation = self
            .events
            .iter()
            .rev()
            .find(|event| event.task == task && event.generation == generation);
        match prior_for_generation {
            None => ensure!(
                stage == LineageStage::AcceptedParent,
                "lineage generation must begin with accepted_parent"
            ),
            Some(previous) => {
                ensure!(
                    previous.parent_checkpoint_sha256 == parent_checkpoint_sha256,
                    "lineage parent drift"
                );
                ensure!(
                    stage.ordinal() == previous.stage.ordinal() + 1,
                    "lineage stage is not the next causal transition"
                );
            }
        }
        let mut event = LineageEventV2 {
            schema: "CausalLineageEvent.v2".to_owned(),
            sequence: self.events.len() as u64,
            previous_event_sha256: self.events.last().map(|row| row.event_sha256.clone()),
            event_sha256: Sha256Digest::of_bytes(b"pending"),
            task,
            generation,
            parent_checkpoint_sha256,
            stage,
            artifact_sha256,
            occurred_at_ms,
        };
        event.event_sha256 = event.calculate_hash();
        let digest = event.event_sha256.clone();
        self.events.push(event);
        self.validate()?;
        Ok(digest)
    }

    pub fn validate(&self) -> Result<()> {
        let mut stages = BTreeMap::<(Task, u64), LineageStage>::new();
        for (index, event) in self.events.iter().enumerate() {
            event.validate_hash()?;
            ensure!(
                event.sequence as usize == index,
                "lineage sequence is not contiguous"
            );
            ensure!(
                event.previous_event_sha256
                    == index
                        .checked_sub(1)
                        .map(|previous| self.events[previous].event_sha256.clone()),
                "lineage hash chain is broken"
            );
            let key = (event.task, event.generation);
            if let Some(previous) = stages.insert(key, event.stage) {
                ensure!(
                    event.stage.ordinal() == previous.ordinal() + 1,
                    "lineage stage ordering is invalid"
                );
            } else {
                ensure!(
                    event.stage == LineageStage::AcceptedParent,
                    "lineage generation does not start at its accepted parent"
                );
            }
        }
        Ok(())
    }

    pub fn complete_generation(&self, task: Task, generation: u64) -> bool {
        self.events.iter().any(|event| {
            event.task == task
                && event.generation == generation
                && event.stage == LineageStage::NextActions
        })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CompletedParentWave {
    pub task: Task,
    pub parent_checkpoint_sha256: Sha256Digest,
    pub wave_sha256: Sha256Digest,
    pub completed_at_ms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParentEvidence {
    pub completed_parent_waves: usize,
    pub authoritative_dl_direct_outcomes: usize,
}

pub fn parent_evidence(
    front: &AcceptedFront,
    waves: &[CompletedParentWave],
    results: &[EvaluationResultV2],
) -> ParentEvidence {
    let completed_parent_waves = waves
        .iter()
        .filter(|wave| {
            wave.task == front.task
                && wave.parent_checkpoint_sha256 == front.checkpoint_sha256
                && wave.completed_at_ms > front.promoted_at_ms
        })
        .map(|wave| wave.wave_sha256.clone())
        .collect::<BTreeSet<_>>()
        .len();
    let authoritative_dl_direct_outcomes = results
        .iter()
        .filter(|result| {
            result.authoritative
                && result.evaluation_key.task == front.task
                && result.completed_at_ms > front.promoted_at_ms
                && result.metadata.origin == Origin::DlDirect
                && result.metadata.cache_disposition == CacheDisposition::Evaluated
                && result.metadata.producer_checkpoint_sha256.as_ref()
                    == Some(&front.checkpoint_sha256)
        })
        .map(|result| result.evaluation_key.id())
        .collect::<BTreeSet<_>>()
        .len();
    ParentEvidence {
        completed_parent_waves,
        authoritative_dl_direct_outcomes,
    }
}

pub fn training_is_eligible(evidence: ParentEvidence) -> bool {
    evidence.completed_parent_waves >= 1
        && evidence.authoritative_dl_direct_outcomes >= MIN_PARENT_DIRECT_OUTCOMES
}

pub fn choose_shared_training_task(
    accepted: &AcceptedByTaskV2,
    evidence: &BTreeMap<Task, ParentEvidence>,
) -> Option<Task> {
    [Task::Binary, Task::Regression]
        .into_iter()
        .filter(|task| {
            evidence
                .get(task)
                .is_some_and(|row| training_is_eligible(*row))
        })
        .min_by_key(|task| {
            let front = accepted.front(*task);
            let row = evidence[task];
            (
                front.generation,
                front.promoted_at_ms,
                Reverse(row.authoritative_dl_direct_outcomes),
            )
        })
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptState {
    Reserved,
    Submitted,
    Terminal,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QueueAttempt {
    pub attempt_number: u32,
    pub reservation_owner: String,
    pub state: AttemptState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator_run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_receipt_sha256: Option<Sha256Digest>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QueueReservationBook {
    #[serde(default)]
    pub pairs: Vec<QueuePairAttempts>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QueuePairAttempts {
    pub evaluation_key: EvaluationKeyV2,
    pub attempts: Vec<QueueAttempt>,
}

impl QueueReservationBook {
    pub fn validate(&self) -> Result<()> {
        let mut keys = BTreeSet::new();
        for entry in &self.pairs {
            entry.evaluation_key.validate()?;
            ensure!(
                keys.insert(entry.evaluation_key.clone()),
                "queue reservation book repeats an evaluation key"
            );
            let mut active = 0;
            for (index, attempt) in entry.attempts.iter().enumerate() {
                ensure!(
                    attempt.attempt_number as usize == index + 1,
                    "queue attempt numbers are not contiguous"
                );
                ensure!(
                    !attempt.reservation_owner.trim().is_empty(),
                    "queue attempt owner is empty"
                );
                match attempt.state {
                    AttemptState::Reserved => {
                        active += 1;
                        ensure!(
                            attempt.evaluator_run_id.is_none()
                                && attempt.final_receipt_sha256.is_none(),
                            "reserved attempt has execution or terminal identity"
                        );
                    }
                    AttemptState::Submitted => {
                        active += 1;
                        ensure!(
                            attempt.evaluator_run_id.is_some()
                                && attempt.final_receipt_sha256.is_none(),
                            "submitted attempt identity is incomplete"
                        );
                    }
                    AttemptState::Terminal => ensure!(
                        attempt.evaluator_run_id.is_some()
                            && attempt.final_receipt_sha256.is_some(),
                        "terminal attempt is not exactly-once finalized"
                    ),
                }
            }
            ensure!(active <= 1, "evaluation pair has duplicate active attempts");
        }
        Ok(())
    }

    pub fn reserve(&mut self, key: EvaluationKeyV2, owner: &str) -> Result<u32> {
        self.validate()?;
        ensure!(!owner.trim().is_empty(), "reservation owner is empty");
        key.validate()?;
        let position = match self
            .pairs
            .iter()
            .position(|entry| entry.evaluation_key == key)
        {
            Some(position) => position,
            None => {
                self.pairs.push(QueuePairAttempts {
                    evaluation_key: key,
                    attempts: Vec::new(),
                });
                self.pairs.len() - 1
            }
        };
        let attempts = &mut self.pairs[position].attempts;
        if let Some(active) = attempts
            .iter()
            .find(|attempt| attempt.state != AttemptState::Terminal)
        {
            ensure!(
                active.reservation_owner == owner,
                "evaluation pair is reserved by another owner"
            );
            return Ok(active.attempt_number);
        }
        ensure!(
            attempts.is_empty(),
            "terminal evaluations require explicit retry, not a fresh reservation"
        );
        attempts.push(QueueAttempt {
            attempt_number: 1,
            reservation_owner: owner.to_owned(),
            state: AttemptState::Reserved,
            evaluator_run_id: None,
            final_receipt_sha256: None,
        });
        Ok(1)
    }

    pub fn submit(
        &mut self,
        key: &EvaluationKeyV2,
        owner: &str,
        attempt_number: u32,
        evaluator_run_id: &str,
    ) -> Result<()> {
        self.validate()?;
        ensure!(
            !evaluator_run_id.trim().is_empty(),
            "evaluator run id is empty"
        );
        let attempt = self.attempt_mut(key, owner, attempt_number)?;
        match attempt.state {
            AttemptState::Reserved => {
                attempt.state = AttemptState::Submitted;
                attempt.evaluator_run_id = Some(evaluator_run_id.to_owned());
                Ok(())
            }
            AttemptState::Submitted => {
                ensure!(
                    attempt.evaluator_run_id.as_deref() == Some(evaluator_run_id),
                    "idempotent submission changed evaluator run identity"
                );
                Ok(())
            }
            AttemptState::Terminal => bail!("terminal attempt cannot be resubmitted"),
        }
    }

    pub fn finalize(
        &mut self,
        key: &EvaluationKeyV2,
        owner: &str,
        attempt_number: u32,
        receipt_sha256: Sha256Digest,
    ) -> Result<()> {
        self.validate()?;
        let attempt = self.attempt_mut(key, owner, attempt_number)?;
        match attempt.state {
            AttemptState::Submitted => {
                attempt.state = AttemptState::Terminal;
                attempt.final_receipt_sha256 = Some(receipt_sha256);
                Ok(())
            }
            AttemptState::Terminal => {
                ensure!(
                    attempt.final_receipt_sha256.as_ref() == Some(&receipt_sha256),
                    "attempt was already finalized with a different receipt"
                );
                Ok(())
            }
            AttemptState::Reserved => bail!("unsubmitted attempt cannot be finalized"),
        }
    }

    pub fn retry(&mut self, key: &EvaluationKeyV2, owner: &str) -> Result<u32> {
        self.validate()?;
        ensure!(!owner.trim().is_empty(), "reservation owner is empty");
        let attempts = self
            .pairs
            .iter_mut()
            .find(|entry| &entry.evaluation_key == key)
            .map(|entry| &mut entry.attempts)
            .ok_or_else(|| anyhow::anyhow!("evaluation pair has no prior attempt"))?;
        let previous = attempts
            .last()
            .ok_or_else(|| anyhow::anyhow!("evaluation pair has no prior attempt"))?;
        ensure!(
            previous.state == AttemptState::Terminal && previous.final_receipt_sha256.is_some(),
            "retry requires an exactly-once finalized prior attempt"
        );
        let attempt_number = previous
            .attempt_number
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("attempt number overflow"))?;
        attempts.push(QueueAttempt {
            attempt_number,
            reservation_owner: owner.to_owned(),
            state: AttemptState::Reserved,
            evaluator_run_id: None,
            final_receipt_sha256: None,
        });
        Ok(attempt_number)
    }

    pub fn active_attempts(&self) -> usize {
        self.pairs
            .iter()
            .flat_map(|entry| entry.attempts.iter())
            .filter(|attempt| attempt.state != AttemptState::Terminal)
            .count()
    }

    fn attempt_mut(
        &mut self,
        key: &EvaluationKeyV2,
        owner: &str,
        attempt_number: u32,
    ) -> Result<&mut QueueAttempt> {
        let attempt = self
            .pairs
            .iter_mut()
            .find(|entry| &entry.evaluation_key == key)
            .and_then(|entry| {
                entry
                    .attempts
                    .iter_mut()
                    .find(|attempt| attempt.attempt_number == attempt_number)
            })
            .ok_or_else(|| anyhow::anyhow!("queue attempt does not exist"))?;
        ensure!(
            attempt.reservation_owner == owner,
            "queue attempt owner mismatch"
        );
        Ok(attempt)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QueueStatusEnvelope {
    pub schema: String,
    pub generated_at_ms: u64,
    pub admission_allowed: bool,
    pub degraded: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fail_closed_reason: Option<String>,
    pub queued: usize,
    pub running: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SupplyAdmission {
    Admit,
    HealthyHold(String),
    Degraded(String),
}

pub fn interpret_queue_status(status: &QueueStatusEnvelope) -> Result<SupplyAdmission> {
    ensure!(
        status.schema == "JopeSelfplayQueueKeeper.v3",
        "queue status envelope schema is not v3"
    );
    ensure!(
        status.generated_at_ms > 0,
        "queue status timestamp is absent"
    );
    if status.admission_allowed {
        ensure!(
            !status.degraded && status.error.is_none(),
            "admission cannot be allowed while queue status is degraded"
        );
        return Ok(SupplyAdmission::Admit);
    }
    let reason = status
        .fail_closed_reason
        .clone()
        .or_else(|| status.error.clone())
        .unwrap_or_else(|| "queue admission is held".to_owned());
    if status.degraded {
        Ok(SupplyAdmission::Degraded(reason))
    } else {
        Ok(SupplyAdmission::HealthyHold(reason))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceIdentity {
    pub service: String,
    pub bundle_sha256: Sha256Digest,
    pub configuration_sha256: Sha256Digest,
    pub evaluator_sha256: Sha256Digest,
    pub started_at_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeBundleV3Identity {
    pub schema: String,
    pub bundle_sha256: Sha256Digest,
    pub configuration_sha256: Sha256Digest,
    pub evaluator_sha256: Sha256Digest,
    pub services: Vec<ServiceIdentity>,
}

impl RuntimeBundleV3Identity {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == RUNTIME_BUNDLE_SCHEMA,
            "runtime bundle schema is not v3"
        );
        ensure!(!self.services.is_empty(), "runtime bundle has no services");
        let mut names = BTreeSet::new();
        for service in &self.services {
            ensure!(
                !service.service.trim().is_empty() && names.insert(service.service.clone()),
                "runtime service names must be nonempty and unique"
            );
            ensure!(
                service.started_at_ms > 0,
                "service start identity is absent"
            );
            ensure!(
                service.bundle_sha256 == self.bundle_sha256
                    && service.configuration_sha256 == self.configuration_sha256
                    && service.evaluator_sha256 == self.evaluator_sha256,
                "runtime service identity drift"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenRunIdentity {
    pub checkpoint_sha256: Sha256Digest,
    pub corpus_sha256: Sha256Digest,
    pub catalog_sha256: Sha256Digest,
    pub evaluator_sha256: Sha256Digest,
    pub proposal_sha256: Sha256Digest,
    pub context_sha256: Sha256Digest,
}

impl FrozenRunIdentity {
    pub fn validate_observed(&self, observed: &Self) -> Result<()> {
        ensure!(
            self == observed,
            "checkpoint, corpus, catalog, evaluator, proposal, or context drift"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActionAccounting {
    pub direct_actions: usize,
    pub direct_evaluations: usize,
    pub direct_strict_wins: usize,
    pub matched_gp_evaluations: usize,
    pub matched_gp_strict_wins: usize,
    pub descendant_evaluations: usize,
    pub requested_d: usize,
    pub actual_d: usize,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheAccounting {
    pub exact_hits: usize,
    pub avoided_evaluations: usize,
    pub active_conflicts: usize,
    pub cross_protocol_conflicts: usize,
    pub requalifications: usize,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCoverage {
    pub registry: usize,
    pub encoded: usize,
    pub active: usize,
    pub observed: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowStatusV2 {
    pub schema: String,
    pub binary_front: AcceptedFront,
    pub regression_front: AcceptedFront,
    pub binary_actions: ActionAccounting,
    pub regression_actions: ActionAccounting,
    pub cache: CacheAccounting,
    pub binary_coverage: TaskCoverage,
    pub regression_coverage: TaskCoverage,
    pub frontier_contexts: usize,
    pub hard_contexts: usize,
    pub anchor_contexts: usize,
    pub proposal_validity: f64,
    pub proposal_uniqueness: f64,
    pub selector_calibrated: bool,
    pub zero_backfill: bool,
    pub corpus_fresh_labels: usize,
    pub parent_evidence_threshold: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_trainer_task: Option<Task>,
    pub proposer_active: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_hold_reason: Option<String>,
}

impl ShadowStatusV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "SelfPlayShadowStatus.v2",
            "status schema is not v2"
        );
        self.binary_front.validate(Task::Binary)?;
        self.regression_front.validate(Task::Regression)?;
        ensure!(
            self.frontier_contexts + self.hard_contexts + self.anchor_contexts
                == super::DATASETS_PER_COHORT,
            "status cohort accounting does not total 64"
        );
        ensure!(
            self.proposal_validity.is_finite()
                && self.proposal_uniqueness.is_finite()
                && (0.0..=1.0).contains(&self.proposal_validity)
                && (0.0..=1.0).contains(&self.proposal_uniqueness),
            "proposal metrics are invalid"
        );
        ensure!(self.zero_backfill, "zero-backfill assertion failed");
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutEvidence {
    pub consecutive_accepted_generations: usize,
    pub zero_backfill: bool,
    pub validity: f64,
    pub complete_lineage: bool,
    pub aggregate_dl_gp_lower_95: f64,
}

pub fn canary_ready(evidence: RolloutEvidence) -> bool {
    evidence.consecutive_accepted_generations >= 3
        && evidence.zero_backfill
        && evidence.validity >= 0.99
        && evidence.complete_lineage
        && evidence.aggregate_dl_gp_lower_95 > 1.0
}

pub fn canary_rung_ready(waves: usize, distinct_datasets: usize) -> bool {
    waves >= 3 && distinct_datasets >= 256
}

pub fn stable_canary_bucket(source_family: &str) -> u8 {
    let digest = Sha256::digest(format!("selfplay-canary-v2|{source_family}").as_bytes());
    digest[0] % 100
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> Sha256Digest {
        Sha256Digest::parse(character.to_string().repeat(64)).unwrap()
    }

    fn front(task: Task, checkpoint: char, generation: u64) -> AcceptedFront {
        AcceptedFront {
            task,
            generation,
            checkpoint_sha256: digest(checkpoint),
            parent_checkpoint_sha256: (generation > 0).then(|| digest('0')),
            promotion_evidence_sha256: digest('e'),
            promoted_at_ms: 10,
        }
    }

    fn key() -> EvaluationKeyV2 {
        EvaluationKeyV2::new(
            Task::Binary,
            digest('a'),
            digest('b'),
            digest('c'),
            digest('d'),
        )
    }

    #[test]
    fn supply_state_resumes_after_every_persisted_stage() {
        let mut state = CohortSupplyStateV1::new(
            Task::Binary,
            digest('a'),
            digest('b'),
            digest('c'),
            digest('d'),
            11,
        );
        let transitions = [
            (
                CohortSupplyStage::SubmitCohortProposer,
                Some("proposal-job".to_owned()),
                None,
                None,
            ),
            (
                CohortSupplyStage::FetchValidateProposals,
                Some("proposal-job".to_owned()),
                Some(digest('e')),
                None,
            ),
            (
                CohortSupplyStage::CacheFilterResample,
                None,
                Some(digest('e')),
                None,
            ),
            (
                CohortSupplyStage::BuildWave,
                None,
                None,
                Some("wave-1".to_owned()),
            ),
            (
                CohortSupplyStage::KeeperAcknowledgment,
                None,
                None,
                Some("wave-1".to_owned()),
            ),
        ];
        for (stage, job, pointer, wave) in transitions {
            assert!(state.advance(stage, job, pointer, wave).unwrap());
            state = serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
            state.validate().unwrap();
        }
        assert!(!state
            .advance(
                CohortSupplyStage::KeeperAcknowledgment,
                Some("proposal-job".into()),
                Some(digest('e')),
                Some("wave-1".into()),
            )
            .unwrap());
        assert!(state
            .advance(
                CohortSupplyStage::KeeperAcknowledgment,
                Some("different-job".into()),
                None,
                None,
            )
            .is_err());
    }

    fn encoding_receipt(
        dataset_id: &str,
        content: char,
        attempt_number: u32,
        succeeded: bool,
    ) -> EncodingAdoptionReceiptV2 {
        EncodingAdoptionReceiptV2 {
            schema: "EncodingAdoptionReceipt.v2".into(),
            dataset: EncodingDatasetIdentity {
                dataset_id: dataset_id.into(),
                content_sha256: digest(content),
            },
            attempt_number,
            job_identity: format!("encode-{dataset_id}-{attempt_number}"),
            encoder_identity_sha256: digest('9'),
            disposition: if succeeded {
                EncodingAdoptionDisposition::Succeeded {
                    encoded_train_sha256: digest('7'),
                    encoded_test_sha256: digest('8'),
                }
            } else {
                EncodingAdoptionDisposition::Failed {
                    error_class: "dataset-malformed".into(),
                }
            },
            recorded_at_ms: attempt_number as u64,
        }
    }

    #[test]
    fn encoding_adoption_is_monotonic_per_dataset_across_partial_chunks() {
        let mut ledger = EncodingAdoptionLedgerV2::default();
        assert!(ledger.adopt(encoding_receipt("a", 'a', 1, true)).unwrap());
        assert!(ledger.adopt(encoding_receipt("b", 'b', 1, false)).unwrap());
        let mut ledger: EncodingAdoptionLedgerV2 =
            serde_json::from_slice(&serde_json::to_vec(&ledger).unwrap()).unwrap();
        assert!(!ledger.adopt(encoding_receipt("a", 'a', 2, false)).unwrap());
        assert!(ledger.adopt(encoding_receipt("b", 'b', 2, true)).unwrap());
        assert_eq!(ledger.successful.len(), 2);
        assert!(ledger.latest_failures.is_empty());

        let mut changed = encoding_receipt("a", 'a', 3, true);
        changed.disposition = EncodingAdoptionDisposition::Succeeded {
            encoded_train_sha256: digest('5'),
            encoded_test_sha256: digest('6'),
        };
        assert!(ledger.adopt(changed).is_err());
    }

    #[test]
    fn task_compare_and_swap_does_not_move_the_other_front() {
        let mut accepted =
            AcceptedByTaskV2::new(front(Task::Binary, '1', 0), front(Task::Regression, '2', 0))
                .unwrap();
        let regression = accepted.regression.clone();
        accepted
            .compare_and_swap(Task::Binary, &digest('1'), digest('3'), digest('4'), 20)
            .unwrap();
        assert_eq!(accepted.binary.generation, 1);
        assert_eq!(accepted.regression, regression);
    }

    #[test]
    fn lineage_is_hash_chained_and_causally_ordered() {
        let mut lineage = CausalLineage::default();
        for (index, stage) in [
            LineageStage::AcceptedParent,
            LineageStage::PolicyActions,
            LineageStage::Evaluations,
            LineageStage::CorpusShards,
            LineageStage::Candidate,
            LineageStage::Tournament,
            LineageStage::AcceptedChild,
            LineageStage::NextActions,
        ]
        .into_iter()
        .enumerate()
        {
            lineage
                .append(
                    Task::Binary,
                    0,
                    digest('a'),
                    stage,
                    Sha256Digest::of_bytes(format!("artifact-{index}").as_bytes()),
                    index as u64 + 1,
                )
                .unwrap();
        }
        lineage.validate().unwrap();
        assert!(lineage.complete_generation(Task::Binary, 0));
        assert!(lineage
            .append(
                Task::Regression,
                0,
                digest('b'),
                LineageStage::Evaluations,
                digest('c'),
                10,
            )
            .is_err());
    }

    #[test]
    fn repeated_queue_submission_never_duplicates_evaluator_execution() {
        let mut queue = QueueReservationBook::default();
        assert_eq!(queue.reserve(key(), "wave-1").unwrap(), 1);
        assert_eq!(queue.reserve(key(), "wave-1").unwrap(), 1);
        queue.submit(&key(), "wave-1", 1, "eval-run-1").unwrap();
        queue.submit(&key(), "wave-1", 1, "eval-run-1").unwrap();
        assert!(queue.submit(&key(), "wave-1", 1, "eval-run-2").is_err());
        assert_eq!(queue.active_attempts(), 1);
        queue.finalize(&key(), "wave-1", 1, digest('e')).unwrap();
        queue.finalize(&key(), "wave-1", 1, digest('e')).unwrap();
        assert_eq!(queue.retry(&key(), "wave-1-retry").unwrap(), 2);
        assert_eq!(queue.active_attempts(), 1);
    }

    #[test]
    fn admission_false_is_a_healthy_supply_hold() {
        let status = QueueStatusEnvelope {
            schema: "JopeSelfplayQueueKeeper.v3".into(),
            generated_at_ms: 1,
            admission_allowed: false,
            degraded: false,
            error: None,
            fail_closed_reason: Some("reservation pressure".into()),
            queued: 0,
            running: 0,
        };
        assert_eq!(
            interpret_queue_status(&status).unwrap(),
            SupplyAdmission::HealthyHold("reservation pressure".into())
        );
    }

    #[test]
    fn bundle_drift_fails_closed() {
        let bundle = RuntimeBundleV3Identity {
            schema: RUNTIME_BUNDLE_SCHEMA.into(),
            bundle_sha256: digest('a'),
            configuration_sha256: digest('b'),
            evaluator_sha256: digest('c'),
            services: vec![ServiceIdentity {
                service: "queue".into(),
                bundle_sha256: digest('a'),
                configuration_sha256: digest('b'),
                evaluator_sha256: digest('d'),
                started_at_ms: 1,
            }],
        };
        assert!(bundle.validate().is_err());
    }

    #[test]
    fn shared_training_slot_chooses_the_oldest_eligible_task_front() {
        let accepted =
            AcceptedByTaskV2::new(front(Task::Binary, '1', 0), front(Task::Regression, '2', 1))
                .unwrap();
        let evidence = BTreeMap::from([
            (
                Task::Binary,
                ParentEvidence {
                    completed_parent_waves: 1,
                    authoritative_dl_direct_outcomes: 2_048,
                },
            ),
            (
                Task::Regression,
                ParentEvidence {
                    completed_parent_waves: 2,
                    authoritative_dl_direct_outcomes: 4_096,
                },
            ),
        ]);
        assert_eq!(
            choose_shared_training_task(&accepted, &evidence),
            Some(Task::Binary)
        );
    }
}

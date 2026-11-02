use super::*;

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

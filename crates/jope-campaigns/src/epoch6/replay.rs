use super::*;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateSlotArithmeticV2 {
    pub slots_per_dataset: u32,
    pub common_winner_slots: Vec<u32>,
    pub checkpoint_proposal_slots: Vec<u32>,
}

impl CandidateSlotArithmeticV2 {
    fn validate(&self) -> Result<()> {
        ensure!(self.slots_per_dataset > 0, "candidate slot count is zero");
        let validate = |slots: &[u32], name: &str| -> Result<()> {
            ensure!(
                sorted_unique(slots),
                "{name} slots are duplicated or unsorted"
            );
            ensure!(
                slots.iter().all(|slot| *slot < self.slots_per_dataset),
                "{name} slot is out of range"
            );
            Ok(())
        };
        validate(&self.common_winner_slots, "common winner")?;
        validate(&self.checkpoint_proposal_slots, "checkpoint proposal")?;
        let common = self
            .common_winner_slots
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        ensure!(
            self.checkpoint_proposal_slots
                .iter()
                .all(|slot| !common.contains(slot)),
            "candidate slot groups overlap"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorModulusV2 {
    pub operator: String,
    pub modulus: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DlSlotPolicyV2 {
    pub maximum_slots: u32,
    pub replacement_only: bool,
    pub candidate_evaluation_calls: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RngAlgorithmV2 {
    Sha256Counter,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayProtocolV2 {
    pub schema: String,
    pub task_order: Vec<Task>,
    pub task_lane_sizes: BTreeMap<Task, u32>,
    pub selector_versions: BTreeMap<Task, String>,
    pub frozen_lane_profiles: BTreeMap<Task, TaskLaneProfileV1>,
    pub candidate_slots: CandidateSlotArithmeticV2,
    pub gp_seed_slots: Vec<u32>,
    pub operator_vocabulary: Vec<OperatorModulusV2>,
    pub dl_slot_policy: DlSlotPolicyV2,
    pub proposal_sampling_policy: String,
    pub rng_algorithm: RngAlgorithmV2,
    pub evaluator_protocol_sha256: String,
    pub task_semantics_sha256: String,
    pub encoding_protocol_sha256: String,
}

impl ReplayProtocolV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == REPLAY_PROTOCOL_SCHEMA,
            "invalid replay protocol schema"
        );
        ensure!(
            self.task_order == Task::CANONICAL_ORDER,
            "replay task order is not canonical"
        );
        let expected = Task::CANONICAL_ORDER.into_iter().collect::<BTreeSet<_>>();
        for (actual, field) in [
            (
                self.task_lane_sizes
                    .keys()
                    .copied()
                    .collect::<BTreeSet<_>>(),
                "lane sizes",
            ),
            (
                self.selector_versions
                    .keys()
                    .copied()
                    .collect::<BTreeSet<_>>(),
                "selector versions",
            ),
            (
                self.frozen_lane_profiles
                    .keys()
                    .copied()
                    .collect::<BTreeSet<_>>(),
                "lane profiles",
            ),
        ] {
            ensure!(actual == expected, "replay {field} is not exhaustive");
        }
        ensure!(
            self.task_lane_sizes.values().all(|size| *size > 0),
            "replay lane size is zero"
        );
        ensure!(
            self.selector_versions
                .values()
                .all(|version| !version.trim().is_empty()),
            "selector version is empty"
        );
        for (task, profile) in &self.frozen_lane_profiles {
            profile.validate()?;
            ensure!(profile.task == *task, "lane profile map key mismatch");
        }
        self.candidate_slots.validate()?;
        ensure!(
            sorted_unique(&self.gp_seed_slots),
            "GP seed slots are duplicated or unsorted"
        );
        ensure!(
            self.gp_seed_slots
                .iter()
                .all(|slot| *slot < self.candidate_slots.slots_per_dataset),
            "GP seed slot is out of range"
        );
        ensure!(
            !self.operator_vocabulary.is_empty(),
            "operator vocabulary is empty"
        );
        let operators = self
            .operator_vocabulary
            .iter()
            .map(|row| row.operator.as_str())
            .collect::<Vec<_>>();
        ensure!(
            sorted_unique(&operators),
            "operator vocabulary is duplicated or unsorted"
        );
        ensure!(
            self.operator_vocabulary.iter().all(|row| row.modulus > 0),
            "operator modulus is zero"
        );
        ensure!(
            self.dl_slot_policy.maximum_slots <= self.candidate_slots.slots_per_dataset,
            "DL slot limit exceeds the candidate lane"
        );
        ensure!(
            self.dl_slot_policy.replacement_only
                && self.dl_slot_policy.candidate_evaluation_calls == 0,
            "DL policy permits additional scientific work"
        );
        ensure!(
            !self.proposal_sampling_policy.trim().is_empty(),
            "proposal sampling policy is empty"
        );
        sha256(&self.evaluator_protocol_sha256, "evaluator protocol")?;
        sha256(&self.encoding_protocol_sha256, "encoding protocol")?;
        verify_embedded_task_semantics()?;
        ensure!(
            self.task_semantics_sha256 == TASK_SEMANTICS_SHA256,
            "replay task-semantics digest drift"
        );
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn derive_item_seed(
        &self,
        wave: u64,
        task: Task,
        dataset_sha256: &str,
        slot_group: &str,
        slot: u32,
        operator: &str,
        local_counter: u64,
    ) -> Result<Sha256Digest> {
        let protocol_sha256 = self.digest()?;
        sha256(dataset_sha256, "dataset")?;
        ensure!(
            self.task_order.contains(&task),
            "seed task is absent from replay protocol"
        );
        ensure!(
            slot < self.candidate_slots.slots_per_dataset,
            "seed slot is out of range"
        );
        ensure!(
            self.operator_vocabulary
                .iter()
                .any(|row| row.operator == operator),
            "seed operator is absent from replay protocol"
        );
        for (field, name) in [(slot_group, "slot group"), (operator, "operator")] {
            ensure!(!field.trim().is_empty(), "seed {name} is empty");
        }
        let value = serde_json::json!({
            "domain": "jope-replay-protocol-v2-item-seed",
            "protocol_sha256": protocol_sha256,
            "wave": wave,
            "task": task,
            "dataset_sha256": dataset_sha256,
            "slot_group": slot_group,
            "slot": slot,
            "operator": operator,
            "local_counter": local_counter,
        });
        Sha256Digest::parse(canonical_json_sha256(&value)?)
    }
}

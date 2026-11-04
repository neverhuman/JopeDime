use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneMaturity {
    Bootstrap,
    Mature,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneRole {
    StableAnchor,
    LeastObserved,
    HardUncertain,
    GrammarImmigrant,
    GpSeed,
    DlProposal,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskLaneProfileV1 {
    pub schema: String,
    pub task: Task,
    pub maturity: LaneMaturity,
    pub roles: Vec<LaneRole>,
}

impl TaskLaneProfileV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == TASK_LANE_PROFILE_SCHEMA,
            "invalid task-lane profile schema"
        );
        ensure!(
            !self.roles.is_empty() && sorted_unique(&self.roles),
            "lane roles are empty, duplicated, or unsorted"
        );
        match self.maturity {
            LaneMaturity::Bootstrap => ensure!(
                self.roles.contains(&LaneRole::GrammarImmigrant),
                "bootstrap lane lacks grammar immigrants"
            ),
            LaneMaturity::Mature => ensure!(
                self.roles.contains(&LaneRole::StableAnchor)
                    && self.roles.contains(&LaneRole::LeastObserved),
                "mature lane lacks stable and least-observed roles"
            ),
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskActivationV1 {
    pub task: Task,
    pub activation_epoch: String,
    pub activation_wave: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LaneActivationSetV1 {
    pub schema: String,
    pub activations: Vec<TaskActivationV1>,
}

impl LaneActivationSetV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == LANE_ACTIVATION_SCHEMA,
            "invalid lane-activation schema"
        );
        let tasks = self
            .activations
            .iter()
            .map(|row| row.task)
            .collect::<Vec<_>>();
        ensure!(
            !tasks.is_empty() && canonical_task_order(&tasks),
            "activations are empty, duplicated, or noncanonical"
        );
        for activation in &self.activations {
            ensure!(
                !activation.activation_epoch.trim().is_empty(),
                "activation epoch is empty"
            );
        }
        Ok(())
    }

    pub fn is_active(&self, task: Task, epoch: &str, wave: u64) -> bool {
        self.activations.iter().any(|activation| {
            activation.task == task
                && activation.activation_epoch == epoch
                && wave >= activation.activation_wave
        })
    }
}

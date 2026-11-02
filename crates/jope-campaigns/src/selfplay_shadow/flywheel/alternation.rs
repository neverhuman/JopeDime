use super::*;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowAlternationState {
    #[serde(default)]
    pub next_task: Option<Task>,
    /// Slot cursor into [`Task::ROTATION_CYCLE`]. Absent in legacy state
    /// files, where the cursor is derived from `next_task`; `next_task`
    /// remains the override channel (terminal-failure retries write it), so
    /// whenever the two disagree, `next_task` wins.
    #[serde(default)]
    pub rotation_position: Option<usize>,
    #[serde(default)]
    pub accepted_parent_by_task: BTreeMap<Task, Sha256Digest>,
    #[serde(default)]
    pub last_shadow_by_task: BTreeMap<Task, Sha256Digest>,
}

/// A task may hold a rotation slot once it has an accepted anchor; the two
/// legacy tasks are unconditionally eligible (their anchors predate this
/// eligibility rule, and a missing legacy anchor must surface loudly in the
/// accepted-pointer readers, not silently stall the rotation).
pub fn rotation_eligible(task: Task, accepted: &BTreeMap<Task, Sha256Digest>) -> bool {
    matches!(task, Task::Regression | Task::Binary) || accepted.contains_key(&task)
}

impl ShadowAlternationState {
    /// First slot of `task` in the rotation cycle (used to re-anchor the
    /// cursor after a `next_task` override or when migrating legacy state).
    fn cycle_anchor(task: Task) -> usize {
        Task::ROTATION_CYCLE
            .iter()
            .position(|slot| *slot == task)
            .expect("every task appears in the rotation cycle")
    }

    pub fn next(&mut self, accepted: &BTreeMap<Task, Sha256Digest>) -> Task {
        for task in Task::ALL {
            if self.accepted_parent_by_task.get(&task) != accepted.get(&task) {
                self.last_shadow_by_task.remove(&task);
            }
        }
        self.accepted_parent_by_task = accepted.clone();
        let position = match (self.next_task, self.rotation_position) {
            (Some(task), Some(position)) => {
                let (predicted, _) =
                    Task::rotation_successor(position, |task| rotation_eligible(task, accepted));
                if predicted == task {
                    // The pending task is exactly what the cursor would pick:
                    // trust the cursor (its slot may sit deeper in the cycle
                    // than the task's first occurrence).
                    position
                } else {
                    // Override (terminal-failure retry) or eligibility drift:
                    // re-anchor on the explicitly requested task.
                    Self::cycle_anchor(task)
                }
            }
            // Legacy state predates the cursor: re-anchor on the pending task.
            (Some(task), None) => Self::cycle_anchor(task),
            (None, Some(position)) => position,
            // Fresh state starts the cycle at regression, as before.
            (None, None) => 0,
        };
        let (task, next_position) =
            Task::rotation_successor(position, |task| rotation_eligible(task, accepted));
        self.rotation_position = Some(next_position);
        let (pending, _) =
            Task::rotation_successor(next_position, |task| rotation_eligible(task, accepted));
        self.next_task = Some(pending);
        task
    }

    pub fn warm_start(&self, task: Task) -> Option<&Sha256Digest> {
        self.last_shadow_by_task
            .get(&task)
            .or_else(|| self.accepted_parent_by_task.get(&task))
    }
}

pub fn production_task(ledger: &CausalEvidenceLedgerV1) -> Option<Task> {
    Task::ALL.into_iter().find(|task| {
        ledger
            .tasks
            .get(task)
            .is_some_and(|row| row.production_gate_open)
    })
}

pub fn can_admit_dl_origin_seed(
    total_after_insert: usize,
    accepted_dl_origin_after_insert: usize,
    shadow_dl_origin_after_insert: usize,
) -> bool {
    accepted_dl_origin_after_insert.saturating_add(shadow_dl_origin_after_insert)
        <= total_after_insert / 3
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCanaryStreaks {
    #[serde(default)]
    pub streaks: BTreeMap<Task, usize>,
}

impl TaskCanaryStreaks {
    pub fn record(&mut self, task: Task, accepted: bool) -> bool {
        let streak = self.streaks.entry(task).or_default();
        *streak = if accepted {
            streak.saturating_add(1)
        } else {
            0
        };
        *streak >= REQUIRED_CONSECUTIVE_CANARY_ACCEPTANCES
    }

    pub fn reset_after_promotion(&mut self, task: Task) {
        self.streaks.insert(task, 0);
    }
}

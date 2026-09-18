use std::collections::{BTreeMap, BTreeSet};

use jain_smartcluster_core::JobState;

use crate::cgroup::{AdoptedCgroup, AttemptKey};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecoverableState {
    Starting,
    Running,
    Paused,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpectedAttempt {
    pub key: AttemptKey,
    pub state: RecoverableState,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetiredAttempt {
    pub key: AttemptKey,
    pub state: JobState,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecoveryAction {
    Adopt {
        expected: ExpectedAttempt,
        cgroup: AdoptedCgroup,
    },
    RetryWithoutFailureCharge {
        expected: ExpectedAttempt,
    },
    CleanupEmpty {
        cgroup: AdoptedCgroup,
    },
    ReapplyRetiredEffect {
        retired: RetiredAttempt,
        cgroup: AdoptedCgroup,
    },
    Quarantine {
        cgroup: AdoptedCgroup,
        reason: QuarantineReason,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuarantineReason {
    UnknownAttempt,
    SupersededFence,
}

pub fn reconcile_attempts(
    expected: Vec<ExpectedAttempt>,
    retired: Vec<RetiredAttempt>,
    discovered: Vec<AdoptedCgroup>,
) -> Vec<RecoveryAction> {
    let mut by_key: BTreeMap<(String, u32, String), AdoptedCgroup> = discovered
        .into_iter()
        .map(|cgroup| (key_tuple(&cgroup.key), cgroup))
        .collect();
    let known_identity: BTreeSet<(String, u32)> = expected
        .iter()
        .map(|attempt| (attempt.key.job_id.clone(), attempt.key.attempt))
        .chain(
            retired
                .iter()
                .map(|attempt| (attempt.key.job_id.clone(), attempt.key.attempt)),
        )
        .collect();
    let mut actions = Vec::new();

    for attempt in expected {
        match by_key.remove(&key_tuple(&attempt.key)) {
            Some(cgroup) if cgroup.populated || !cgroup.pids.is_empty() => {
                actions.push(RecoveryAction::Adopt {
                    expected: attempt,
                    cgroup,
                });
            }
            Some(cgroup) => {
                actions.push(RecoveryAction::RetryWithoutFailureCharge { expected: attempt });
                actions.push(RecoveryAction::CleanupEmpty { cgroup });
            }
            None => actions.push(RecoveryAction::RetryWithoutFailureCharge { expected: attempt }),
        }
    }

    for attempt in retired {
        if let Some(cgroup) = by_key.remove(&key_tuple(&attempt.key)) {
            if cgroup.populated || !cgroup.pids.is_empty() {
                actions.push(RecoveryAction::ReapplyRetiredEffect {
                    retired: attempt,
                    cgroup,
                });
            } else {
                actions.push(RecoveryAction::CleanupEmpty { cgroup });
            }
        }
    }

    actions.extend(by_key.into_values().map(|cgroup| {
        let identity = (cgroup.key.job_id.clone(), cgroup.key.attempt);
        let reason = if known_identity.contains(&identity) {
            QuarantineReason::SupersededFence
        } else {
            QuarantineReason::UnknownAttempt
        };
        RecoveryAction::Quarantine { cgroup, reason }
    }));
    actions
}

fn key_tuple(key: &AttemptKey) -> (String, u32, String) {
    (key.job_id.clone(), key.attempt, key.fence.clone())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn expected(fence: &str) -> ExpectedAttempt {
        ExpectedAttempt {
            key: AttemptKey {
                job_id: "job".to_owned(),
                attempt: 1,
                fence: fence.to_owned(),
            },
            state: RecoverableState::Running,
        }
    }

    fn cgroup(fence: &str, populated: bool) -> AdoptedCgroup {
        AdoptedCgroup {
            key: expected(fence).key,
            name: fence.to_owned(),
            path: PathBuf::from(fence),
            pids: populated.then_some(17).into_iter().collect(),
            populated,
            frozen: false,
        }
    }

    #[test]
    fn populated_subtree_without_leader_pid_never_requeues() {
        let mut live = cgroup("fresh", true);
        live.pids.clear();
        assert!(matches!(
            reconcile_attempts(vec![expected("fresh")], Vec::new(), vec![live.clone()]).as_slice(),
            [RecoveryAction::Adopt { .. }]
        ));
        let retired = RetiredAttempt {
            key: live.key.clone(),
            state: JobState::RetryWait,
        };
        assert!(matches!(
            reconcile_attempts(Vec::new(), vec![retired], vec![live]).as_slice(),
            [RecoveryAction::ReapplyRetiredEffect { .. }]
        ));
    }

    #[test]
    fn matching_live_fence_is_adopted() {
        assert!(matches!(
            &reconcile_attempts(
                vec![expected("fresh")],
                Vec::new(),
                vec![cgroup("fresh", true)]
            )[0],
            RecoveryAction::Adopt { .. }
        ));
    }

    #[test]
    fn superseded_fence_is_never_adopted() {
        let actions = reconcile_attempts(
            vec![expected("fresh")],
            Vec::new(),
            vec![cgroup("superseded", true)],
        );
        assert!(actions
            .iter()
            .any(|action| matches!(action, RecoveryAction::RetryWithoutFailureCharge { .. })));
        assert!(actions.iter().any(|action| matches!(
            action,
            RecoveryAction::Quarantine {
                reason: QuarantineReason::SupersededFence,
                ..
            }
        )));
    }

    #[test]
    fn cancelled_before_signal_reapplies_retired_effect() {
        let retired = RetiredAttempt {
            key: expected("cancelled").key,
            state: JobState::Cancelled,
        };
        let actions =
            reconcile_attempts(Vec::new(), vec![retired], vec![cgroup("cancelled", true)]);
        assert!(matches!(
            &actions[0],
            RecoveryAction::ReapplyRetiredEffect {
                retired: RetiredAttempt {
                    state: JobState::Cancelled,
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn retry_wait_before_kill_reapplies_retired_effect() {
        let retired = RetiredAttempt {
            key: expected("retry").key,
            state: JobState::RetryWait,
        };
        let actions = reconcile_attempts(Vec::new(), vec![retired], vec![cgroup("retry", true)]);
        assert!(matches!(
            &actions[0],
            RecoveryAction::ReapplyRetiredEffect {
                retired: RetiredAttempt {
                    state: JobState::RetryWait,
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn known_empty_starting_cgroup_is_cleaned_after_retry_is_committed() {
        let actions = reconcile_attempts(
            vec![expected("empty")],
            Vec::new(),
            vec![cgroup("empty", false)],
        );
        assert!(matches!(
            actions.as_slice(),
            [
                RecoveryAction::RetryWithoutFailureCharge { .. },
                RecoveryAction::CleanupEmpty { .. }
            ]
        ));
    }
}

use serde::Serialize;

use crate::ExecutionError;

/// Typed, agent-friendly exception/error surface for a failed node execution.
///
/// `purpose` names the control that failed, `reason` preserves the concrete
/// error, and the remaining fields route an operator to a bounded local rerun.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExecutionGuidance {
    pub purpose: &'static str,
    pub reason: String,
    pub common_fixes: Vec<&'static str>,
    pub docs_url: &'static str,
    pub repair_hint: &'static str,
}

impl ExecutionError {
    pub fn guidance(&self) -> ExecutionGuidance {
        let (purpose, common_fixes, repair_hint) = match self {
            Self::UnsupportedHost | Self::IsolationBroker(_) => (
                "prove the host isolation boundary before executable bytes run",
                vec![
                    "reissue_the_root_isolation_qualification",
                    "verify_the_exact_cgroup_and_device_policy",
                ],
                "rerun the JainNode isolation qualification and its root-broker readback",
            ),
            Self::InvalidWorkUnit | Self::StaleAuthority | Self::Assignment(_) => (
                "preserve signed work, lease, attempt, and fence authority",
                vec![
                    "resubmit_the_exact_signed_work_unit",
                    "refresh_the_controller_lease_and_fence",
                ],
                "rerun the fenced assignment and restart-recovery tests",
            ),
            Self::InvalidStaging | Self::DurabilityUnavailable | Self::Storage(_) => (
                "publish only digest-verified outputs with Shard quorum",
                vec![
                    "restore_two_distinct_writable_domains",
                    "restage_inputs_from_authenticated_artifact_ids",
                ],
                "rerun the Shard quorum, staging, and publication qualification",
            ),
            Self::NetworkPolicyUnavailable | Self::Network(_) => (
                "deny undeclared egress and isolate each run network",
                vec![
                    "remove_stale_exact_owned_networks",
                    "requalify_the_closed_egress_engine",
                ],
                "rerun the closed-egress Docker network tests",
            ),
            Self::Cancelled | Self::CancellationState => (
                "make cancellation durable and fence-exact",
                vec![
                    "repair_the_exact_cancellation_record",
                    "drain_the_affected_node_session",
                ],
                "rerun the durable cancellation and prepared-revocation tests",
            ),
            Self::DeadlineExceeded
            | Self::MemoryLimitExceeded
            | Self::NonZeroExit(_)
            | Self::LogBudgetExceeded => (
                "enforce the declared WorkUnit resource and output budgets",
                vec![
                    "inspect_the_signed_resource_limits",
                    "reduce_the_workload_or_raise_a_governed_budget",
                ],
                "rerun the bounded executor failure-path tests",
            ),
            Self::StreamProtocolViolation | Self::OutboundControlUnavailable => (
                "retain bounded ordered controller and workload streams",
                vec![
                    "replay_the_durable_outbox",
                    "reestablish_the_authenticated_outbound_session",
                ],
                "rerun the signed outbox confirmation and reconnect-replay tests",
            ),
            Self::ClockUnavailable => (
                "evaluate deadlines and signatures against a trustworthy clock",
                vec![
                    "restore_the_host_clock",
                    "drain_the_node_until_time_is_stable",
                ],
                "rerun the clock-bound admission tests",
            ),
            Self::Docker(_) => (
                "execute only through the bounded OCI engine adapter",
                vec![
                    "verify_the_governed_Docker_socket",
                    "inspect_the_exact_container_receipt",
                ],
                "rerun the Docker framing, isolation, and cleanup tests",
            ),
        };
        ExecutionGuidance {
            purpose,
            reason: self.to_string(),
            common_fixes,
            docs_url: "docs/testing.md",
            repair_hint,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_representative_failure_has_bounded_local_repair_context() {
        let errors = [
            ExecutionError::UnsupportedHost,
            ExecutionError::InvalidWorkUnit,
            ExecutionError::DurabilityUnavailable,
            ExecutionError::NetworkPolicyUnavailable,
            ExecutionError::CancellationState,
            ExecutionError::MemoryLimitExceeded,
            ExecutionError::StreamProtocolViolation,
            ExecutionError::ClockUnavailable,
        ];
        for error in errors {
            let guidance = error.guidance();
            assert!(!guidance.purpose.is_empty());
            assert_eq!(guidance.reason, error.to_string());
            assert!(!guidance.common_fixes.is_empty());
            assert!(guidance.docs_url.starts_with("docs/"));
            assert!(guidance.repair_hint.starts_with("rerun"));
            assert!(serde_json::to_vec(&guidance).unwrap().len() < 2_048);
        }
    }
}

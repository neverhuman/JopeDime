use super::*;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct ExecutionKey {
    pub(super) run_id: String,
    pub(super) attempt: u32,
    pub(super) fence: u64,
}

impl ExecutionKey {
    pub(super) fn from_work(work: &VerifiedWorkUnit) -> Self {
        Self {
            run_id: work.run_id.to_string(),
            attempt: work.attempt,
            fence: work.attempt_fence,
        }
    }
}

pub(super) struct PreparedExecution {
    pub(super) accepted: AcceptedAssignment,
    pub(super) acknowledgement: WorkerAcknowledgement,
}

pub(super) type ProductExecutor<D, S, N, M, I, K> =
    JainNodeExecutor<D, S, N, DurableCancellationRegistry, M, DurableControlOutbox, I, K>;

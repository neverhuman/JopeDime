use std::{
    collections::{btree_map::Entry, BTreeMap, VecDeque},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

use async_trait::async_trait;
use jain_smartcluster_fabric::{VerifiedWorkUnit, WorkerAcknowledgement};

use crate::{
    AcceptedAssignment, AdmittedInfer, ClockPort, ContainerIsolationPort, DockerEnginePort,
    DurableCancellationRegistry, DurableControlOutbox, ExecutionError, JainNodeExecutor,
    MetricsPort, NetworkPolicyPort, ShardPort,
};

mod execution_types;
pub(super) use execution_types::{ExecutionKey, PreparedExecution, ProductExecutor};

#[async_trait]
pub trait NodeExecutionPort: Send + Sync {
    async fn admit(&self, admitted: AdmittedInfer, now_ms: u64) -> Result<(), ExecutionError>;

    async fn prepare(&self, admitted: AdmittedInfer, now_ms: u64) -> Result<(), ExecutionError> {
        self.admit(admitted, now_ms).await
    }

    fn launch(&self, _run_id: &str, _attempt: u32, _fence: u64) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn discard_prepared(
        &self,
        _run_id: &str,
        _attempt: u32,
        _fence: u64,
    ) -> Result<(), ExecutionError> {
        Err(ExecutionError::StaleAuthority)
    }

    fn recover_committed(&self, _control_position: u64) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn committed_recovery_count(&self, _control_position: u64) -> Result<usize, ExecutionError> {
        Ok(0)
    }

    fn active_work(
        &self,
        run_id: &str,
        attempt: u32,
        fence: u64,
    ) -> Result<Option<VerifiedWorkUnit>, ExecutionError>;

    fn active_count(&self) -> Result<usize, ExecutionError>;

    async fn cancel(&self, run_id: &str, attempt: u32, fence: u64) -> Result<(), ExecutionError>;

    async fn revoke_all(&self) -> Result<(), ExecutionError>;

    fn ready_for_rotation(&self) -> Result<bool, ExecutionError>;

    fn drain_completions(&self) -> Result<usize, ExecutionError>;
}

pub struct ConcurrentExecutorRuntime<D, S, N, M, I, K>
where
    D: DockerEnginePort,
    S: ShardPort,
    N: NetworkPolicyPort,
    M: MetricsPort,
    I: ContainerIsolationPort,
    K: ClockPort,
{
    executor: Arc<ProductExecutor<D, S, N, M, I, K>>,
    cancellations: Arc<DurableCancellationRegistry>,
    outbox: Arc<DurableControlOutbox>,
    prepared: Arc<Mutex<BTreeMap<ExecutionKey, PreparedExecution>>>,
    active: Arc<Mutex<BTreeMap<ExecutionKey, VerifiedWorkUnit>>>,
    completions: Arc<Mutex<VecDeque<ExecutionKey>>>,
    fatal: Arc<AtomicBool>,
}

impl<D, S, N, M, I, K> ConcurrentExecutorRuntime<D, S, N, M, I, K>
where
    D: DockerEnginePort,
    S: ShardPort,
    N: NetworkPolicyPort,
    M: MetricsPort,
    I: ContainerIsolationPort,
    K: ClockPort,
{
    pub fn new(
        executor: Arc<ProductExecutor<D, S, N, M, I, K>>,
        cancellations: Arc<DurableCancellationRegistry>,
        outbox: Arc<DurableControlOutbox>,
    ) -> Self {
        Self {
            executor,
            cancellations,
            outbox,
            prepared: Arc::new(Mutex::new(BTreeMap::new())),
            active: Arc::new(Mutex::new(BTreeMap::new())),
            completions: Arc::new(Mutex::new(VecDeque::new())),
            fatal: Arc::new(AtomicBool::new(false)),
        }
    }

    fn fail_terminally(&self) {
        self.fatal.store(true, Ordering::SeqCst);
    }

    fn register_active(
        &self,
        key: ExecutionKey,
        work: VerifiedWorkUnit,
    ) -> Result<(), ExecutionError> {
        let mut active = self
            .active
            .lock()
            .map_err(|_| ExecutionError::StaleAuthority)?;
        match active.entry(key) {
            Entry::Vacant(slot) => {
                slot.insert(work);
                Ok(())
            }
            Entry::Occupied(_) => Err(ExecutionError::StaleAuthority),
        }
    }
}

impl<D, S, N, M, I, K> ConcurrentExecutorRuntime<D, S, N, M, I, K>
where
    D: DockerEnginePort + 'static,
    S: ShardPort + 'static,
    N: NetworkPolicyPort + 'static,
    M: MetricsPort + 'static,
    I: ContainerIsolationPort + 'static,
    K: ClockPort + 'static,
{
    pub fn recover(
        executor: Arc<ProductExecutor<D, S, N, M, I, K>>,
        cancellations: Arc<DurableCancellationRegistry>,
        outbox: Arc<DurableControlOutbox>,
        now_ms: u64,
    ) -> Result<Self, ExecutionError> {
        let runtime = Self::new(executor, cancellations, outbox);
        let recovery = runtime.executor.recover_assignments(now_ms)?;
        if !recovery.quarantined.is_empty() {
            runtime.fail_terminally();
            return Err(ExecutionError::StaleAuthority);
        }
        for accepted in recovery.replayable {
            let key = ExecutionKey::from_work(accepted.work());
            let acknowledgement = runtime.executor.recovered_acknowledgement(&accepted)?;
            let prepared = PreparedExecution {
                accepted,
                acknowledgement,
            };
            if runtime
                .prepared
                .lock()
                .map_err(|_| ExecutionError::StaleAuthority)?
                .insert(key, prepared)
                .is_some()
            {
                return Err(ExecutionError::StaleAuthority);
            }
        }
        Ok(runtime)
    }

    fn publish_and_spawn(
        &self,
        key: ExecutionKey,
        accepted: AcceptedAssignment,
        acknowledgement: jain_smartcluster_fabric::WorkerAcknowledgement,
    ) -> Result<(), ExecutionError> {
        let acknowledgement_pending = self
            .outbox
            .contains_acknowledgement(&acknowledgement)
            .map_err(|_| ExecutionError::OutboundControlUnavailable)?;
        if !acknowledgement_pending
            && self
                .outbox
                .enqueue_acknowledgement(acknowledgement)
                .is_err()
        {
            self.fail_terminally();
            return Err(ExecutionError::OutboundControlUnavailable);
        }

        let executor = Arc::clone(&self.executor);
        let outbox = Arc::clone(&self.outbox);
        let active = Arc::clone(&self.active);
        let completions = Arc::clone(&self.completions);
        let fatal = Arc::clone(&self.fatal);
        tokio::spawn(async move {
            let work = accepted.work();
            let run_id = work.run_id.to_string();
            let attempt = work.attempt;
            let fence = work.attempt_fence;
            let control_position = work.control_position;
            let result = executor.execute(&accepted).await;
            let at_ms = crate::clock::unix_time_ms();
            let terminal = match (result, at_ms) {
                (Ok(receipt), Ok(at_ms)) => outbox.enqueue_terminal_success(
                    run_id,
                    attempt,
                    fence,
                    control_position,
                    at_ms,
                    receipt,
                ),
                (Err(error), Ok(at_ms)) => {
                    if executor.terminalize_failure(&accepted).is_err() {
                        fatal.store(true, Ordering::SeqCst);
                        return;
                    }
                    outbox.enqueue_terminal_failure(
                        run_id,
                        attempt,
                        fence,
                        control_position,
                        at_ms,
                        &error,
                    )
                }
                (_, Err(())) => {
                    fatal.store(true, Ordering::SeqCst);
                    return;
                }
            };
            if terminal.is_err() {
                fatal.store(true, Ordering::SeqCst);
                return;
            }
            let removed = active
                .lock()
                .ok()
                .and_then(|mut active| active.remove(&key));
            if removed.is_none() {
                fatal.store(true, Ordering::SeqCst);
                return;
            }
            if let Ok(mut completed) = completions.lock() {
                completed.push_back(key);
            } else {
                fatal.store(true, Ordering::SeqCst);
            }
        });
        Ok(())
    }
}

#[async_trait]
impl<D, S, N, M, I, K> NodeExecutionPort for ConcurrentExecutorRuntime<D, S, N, M, I, K>
where
    D: DockerEnginePort + 'static,
    S: ShardPort + 'static,
    N: NetworkPolicyPort + 'static,
    M: MetricsPort + 'static,
    I: ContainerIsolationPort + 'static,
    K: ClockPort + 'static,
{
    async fn admit(&self, admitted: AdmittedInfer, now_ms: u64) -> Result<(), ExecutionError> {
        let run_id = admitted.work.run_id.to_string();
        let attempt = admitted.work.attempt;
        let fence = admitted.work.attempt_fence;
        self.prepare(admitted, now_ms).await?;
        self.launch(&run_id, attempt, fence)
    }

    async fn prepare(&self, admitted: AdmittedInfer, now_ms: u64) -> Result<(), ExecutionError> {
        if self.fatal.load(Ordering::SeqCst) {
            return Err(ExecutionError::OutboundControlUnavailable);
        }
        let key = ExecutionKey::from_work(&admitted.work);
        {
            let prepared = self
                .prepared
                .lock()
                .map_err(|_| ExecutionError::StaleAuthority)?;
            if let Some(existing) = prepared.get(&key) {
                return if existing.accepted.work() == &admitted.work
                    && existing.accepted.placement() == &admitted.placement
                    && existing.accepted.session_sequence() == admitted.sequence
                {
                    Ok(())
                } else {
                    Err(ExecutionError::StaleAuthority)
                };
            }
        }
        if self
            .active_work(&key.run_id, key.attempt, key.fence)?
            .is_some()
        {
            return Err(ExecutionError::StaleAuthority);
        }
        let (accepted, acknowledgement) = self.executor.accept_assignment(
            admitted.work,
            admitted.placement,
            admitted.sequence,
            now_ms,
        )?;
        let mut prepared = self
            .prepared
            .lock()
            .map_err(|_| ExecutionError::StaleAuthority)?;
        match prepared.entry(key) {
            Entry::Vacant(slot) => {
                slot.insert(PreparedExecution {
                    accepted,
                    acknowledgement,
                });
                Ok(())
            }
            Entry::Occupied(_) => Err(ExecutionError::StaleAuthority),
        }
    }

    fn launch(&self, run_id: &str, attempt: u32, fence: u64) -> Result<(), ExecutionError> {
        let key = ExecutionKey {
            run_id: run_id.to_owned(),
            attempt,
            fence,
        };
        let prepared = self
            .prepared
            .lock()
            .map_err(|_| ExecutionError::StaleAuthority)?
            .remove(&key)
            .ok_or(ExecutionError::StaleAuthority)?;
        if let Err(error) = self.register_active(key.clone(), prepared.accepted.work().clone()) {
            self.prepared
                .lock()
                .map_err(|_| ExecutionError::StaleAuthority)?
                .insert(key, prepared);
            return Err(error);
        }
        self.publish_and_spawn(key, prepared.accepted, prepared.acknowledgement)
    }

    fn discard_prepared(
        &self,
        run_id: &str,
        attempt: u32,
        fence: u64,
    ) -> Result<(), ExecutionError> {
        let key = ExecutionKey {
            run_id: run_id.to_owned(),
            attempt,
            fence,
        };
        let prepared = self
            .prepared
            .lock()
            .map_err(|_| ExecutionError::StaleAuthority)?
            .remove(&key)
            .ok_or(ExecutionError::StaleAuthority)?;
        self.executor.reject_prepared(&prepared.accepted)
    }

    fn recover_committed(&self, control_position: u64) -> Result<(), ExecutionError> {
        let keys = self
            .prepared
            .lock()
            .map_err(|_| ExecutionError::StaleAuthority)?
            .iter()
            .filter(|(_, prepared)| prepared.accepted.session_sequence() <= control_position)
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        for key in keys {
            self.launch(&key.run_id, key.attempt, key.fence)?;
        }
        Ok(())
    }

    fn committed_recovery_count(&self, control_position: u64) -> Result<usize, ExecutionError> {
        self.prepared
            .lock()
            .map_err(|_| ExecutionError::StaleAuthority)
            .map(|prepared| {
                prepared
                    .values()
                    .filter(|prepared| prepared.accepted.session_sequence() <= control_position)
                    .count()
            })
    }

    fn active_work(
        &self,
        run_id: &str,
        attempt: u32,
        fence: u64,
    ) -> Result<Option<VerifiedWorkUnit>, ExecutionError> {
        let key = ExecutionKey {
            run_id: run_id.to_owned(),
            attempt,
            fence,
        };
        self.active
            .lock()
            .map_err(|_| ExecutionError::StaleAuthority)
            .map(|active| active.get(&key).cloned())
    }

    fn active_count(&self) -> Result<usize, ExecutionError> {
        self.active
            .lock()
            .map_err(|_| ExecutionError::StaleAuthority)
            .map(|active| active.len())
    }

    async fn cancel(&self, run_id: &str, attempt: u32, fence: u64) -> Result<(), ExecutionError> {
        if self.active_work(run_id, attempt, fence)?.is_none() {
            return Err(ExecutionError::StaleAuthority);
        }
        self.cancellations.accept(run_id, attempt, fence).await
    }

    async fn revoke_all(&self) -> Result<(), ExecutionError> {
        let keys = self
            .active
            .lock()
            .map_err(|_| ExecutionError::StaleAuthority)?
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        for key in keys {
            self.cancellations
                .accept(&key.run_id, key.attempt, key.fence)
                .await?;
        }
        let prepared_keys = self
            .prepared
            .lock()
            .map_err(|_| ExecutionError::StaleAuthority)?
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        for key in prepared_keys {
            let prepared = self
                .prepared
                .lock()
                .map_err(|_| ExecutionError::StaleAuthority)?
                .remove(&key)
                .ok_or(ExecutionError::StaleAuthority)?;
            if let Err(error) = self.executor.reject_prepared(&prepared.accepted) {
                self.prepared
                    .lock()
                    .map_err(|_| ExecutionError::StaleAuthority)?
                    .insert(key, prepared);
                self.fail_terminally();
                return Err(error);
            }
        }
        Ok(())
    }

    fn ready_for_rotation(&self) -> Result<bool, ExecutionError> {
        let prepared_empty = self
            .prepared
            .lock()
            .map_err(|_| ExecutionError::StaleAuthority)
            .map(|prepared| prepared.is_empty())?;
        let active_empty = self
            .active
            .lock()
            .map_err(|_| ExecutionError::StaleAuthority)
            .map(|active| active.is_empty())?;
        let outbox_empty = self
            .outbox
            .pending_count()
            .map_err(|_| ExecutionError::OutboundControlUnavailable)?
            == 0;
        Ok(prepared_empty && active_empty && outbox_empty)
    }

    fn drain_completions(&self) -> Result<usize, ExecutionError> {
        if self.fatal.load(Ordering::SeqCst) {
            return Err(ExecutionError::OutboundControlUnavailable);
        }
        let mut completions = self
            .completions
            .lock()
            .map_err(|_| ExecutionError::StaleAuthority)?;
        let count = completions.len();
        completions.clear();
        Ok(count)
    }
}

#[cfg(test)]
#[path = "dispatch_inline_tests.rs"]
mod tests;

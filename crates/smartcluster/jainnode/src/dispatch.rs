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

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ExecutionKey {
    run_id: String,
    attempt: u32,
    fence: u64,
}

impl ExecutionKey {
    fn from_work(work: &VerifiedWorkUnit) -> Self {
        Self {
            run_id: work.run_id.to_string(),
            attempt: work.attempt,
            fence: work.attempt_fence,
        }
    }
}

struct PreparedExecution {
    accepted: AcceptedAssignment,
    acknowledgement: WorkerAcknowledgement,
}

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

type ProductExecutor<D, S, N, M, I, K> =
    JainNodeExecutor<D, S, N, DurableCancellationRegistry, M, DurableControlOutbox, I, K>;

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
mod tests {
    use std::{collections::BTreeMap, path::PathBuf, sync::atomic::AtomicUsize};

    use ed25519_dalek::SigningKey;
    use jain_smartcluster_fabric::{
        image_signed_bytes, projection_signed_bytes, ArtifactMount, DomainSignature,
        GpuRequirement, OutputDeclaration, Placement, ResourceLimits, SignatureAuthority,
        SignedWorkUnitProjection, WorkUnitProjection,
    };
    use ulid::Ulid;

    use super::*;
    use crate::{
        CommitFragmentWrite, ControlError, ControlFlow, ControlFrameHandler, InferRequest,
        NodeControlFrame, NodeProfile, NodeProtocolState, ProductionControlHandler, ProtocolError,
        SignedStorageControlRequest, StorageControlBody,
    };

    struct FixedClock;

    impl ClockPort for FixedClock {
        fn now_ms(&self) -> Result<u64, ExecutionError> {
            Ok(50_000)
        }
    }

    #[derive(Default)]
    struct FakeRuntime {
        active: Mutex<Option<VerifiedWorkUnit>>,
        admitted: AtomicUsize,
        cancelled: AtomicBool,
        revoked: AtomicBool,
        fail_cancel_once: AtomicBool,
        fail_revoke_once: AtomicBool,
        completions: AtomicUsize,
        recovery_count: AtomicUsize,
        recovered: AtomicBool,
    }

    impl FakeRuntime {
        fn with_active(work: VerifiedWorkUnit) -> Self {
            Self {
                active: Mutex::new(Some(work)),
                ..Self::default()
            }
        }

        fn complete_one(&self) {
            self.completions.store(1, Ordering::SeqCst);
        }
    }

    #[async_trait]
    impl NodeExecutionPort for FakeRuntime {
        async fn admit(&self, admitted: AdmittedInfer, _now_ms: u64) -> Result<(), ExecutionError> {
            let mut active = self
                .active
                .lock()
                .map_err(|_| ExecutionError::StaleAuthority)?;
            if active.is_some() {
                return Err(ExecutionError::StaleAuthority);
            }
            *active = Some(admitted.work);
            self.admitted.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn active_work(
            &self,
            run_id: &str,
            attempt: u32,
            fence: u64,
        ) -> Result<Option<VerifiedWorkUnit>, ExecutionError> {
            Ok(self
                .active
                .lock()
                .map_err(|_| ExecutionError::StaleAuthority)?
                .as_ref()
                .filter(|work| {
                    work.run_id.to_string() == run_id
                        && work.attempt == attempt
                        && work.attempt_fence == fence
                })
                .cloned())
        }

        fn active_count(&self) -> Result<usize, ExecutionError> {
            Ok(usize::from(
                self.active
                    .lock()
                    .map_err(|_| ExecutionError::StaleAuthority)?
                    .is_some(),
            ))
        }

        fn recover_committed(&self, _control_position: u64) -> Result<(), ExecutionError> {
            self.recovered.store(true, Ordering::SeqCst);
            Ok(())
        }

        fn committed_recovery_count(
            &self,
            _control_position: u64,
        ) -> Result<usize, ExecutionError> {
            Ok(self.recovery_count.load(Ordering::SeqCst))
        }

        async fn cancel(
            &self,
            _run_id: &str,
            _attempt: u32,
            _fence: u64,
        ) -> Result<(), ExecutionError> {
            if self.fail_cancel_once.swap(false, Ordering::SeqCst) {
                return Err(ExecutionError::CancellationState);
            }
            self.cancelled.store(true, Ordering::SeqCst);
            Ok(())
        }

        async fn revoke_all(&self) -> Result<(), ExecutionError> {
            if self.fail_revoke_once.swap(false, Ordering::SeqCst) {
                return Err(ExecutionError::CancellationState);
            }
            self.revoked.store(true, Ordering::SeqCst);
            Ok(())
        }

        fn ready_for_rotation(&self) -> Result<bool, ExecutionError> {
            Ok(self
                .active
                .lock()
                .map_err(|_| ExecutionError::StaleAuthority)?
                .is_none())
        }

        fn drain_completions(&self) -> Result<usize, ExecutionError> {
            let count = self.completions.swap(0, Ordering::SeqCst);
            if count != 0 {
                *self
                    .active
                    .lock()
                    .map_err(|_| ExecutionError::StaleAuthority)? = None;
            }
            Ok(count)
        }
    }

    fn hub_key() -> SigningKey {
        SigningKey::from_bytes(&[91; 32])
    }

    fn work() -> VerifiedWorkUnit {
        let key = hub_key();
        let image_digest = [2; 32];
        let image_hex = image_digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let projection = WorkUnitProjection {
            run_id: Ulid::from(1_u128),
            project_id: "project-1".to_owned(),
            workload_class: crate::WorkloadClass::Training,
            image_reference: format!("registry.invalid/worker@sha256:{image_hex}"),
            image_digest,
            argv: vec!["/jain-platform".to_owned(), "run".to_owned()],
            smoke_argv: vec!["/jain-platform".to_owned(), "smoke".to_owned()],
            smoke_timeout_ms: 1_000,
            smoke_only: false,
            working_directory: "work".to_owned(),
            caller_environment: BTreeMap::new(),
            inputs: vec![ArtifactMount {
                artifact_id: "input".to_owned(),
                artifact_digest: [4; 32],
                mount_path: "/inputs/data".to_owned(),
                size_bytes: 100,
                durable_receipt_digest: [5; 32],
            }],
            outputs: vec![OutputDeclaration {
                relative_path: "result.json".to_owned(),
                media_type: "application/json".to_owned(),
                maximum_bytes: 1_000,
                required: true,
            }],
            limits: ResourceLimits {
                cpu_millicores: 1_000,
                memory_bytes: 1_000,
                scratch_bytes: 1_000,
                pids: 32,
                wall_time_ms: 10_000,
                read_bytes: 1_000,
                write_bytes: 1_000,
                io_operations: 100,
            },
            gpu: Some(GpuRequirement {
                model: Some("RTX 3090".to_owned()),
                count: 1,
                minimum_vram_bytes: 500,
            }),
            open_outbound: false,
            network_bytes: 0,
            network_connections: 0,
            network_deadline_ms: 0,
            stdout_bytes: 1_000,
            stderr_bytes: 1_000,
            minimum_writable_domains: 2,
            attempt: 1,
            attempt_fence: 9,
            hub_epoch: 2,
            control_position: 3,
            replay_nonce: [6; 32],
        };
        let issuer = "hub-issuer".to_owned();
        VerifiedWorkUnit::verify(
            SignedWorkUnitProjection {
                image_signature: DomainSignature::sign(
                    SignatureAuthority::OciImage,
                    issuer.clone(),
                    "hub-key".to_owned(),
                    &key,
                    &image_signed_bytes(&projection.image_reference, projection.image_digest)
                        .unwrap(),
                ),
                work_unit_signature: DomainSignature::sign(
                    SignatureAuthority::WorkUnitIssuer,
                    issuer.clone(),
                    "hub-key".to_owned(),
                    &key,
                    &projection_signed_bytes(&projection).unwrap(),
                ),
                projection,
                issuer,
            },
            "hub-issuer",
            key.verifying_key().to_bytes(),
            "hub-issuer",
            key.verifying_key().to_bytes(),
        )
        .unwrap()
    }

    fn placement() -> Placement {
        let work = work();
        let key = hub_key();
        let mut placement = Placement {
            node_id: "node-1".to_owned(),
            session_id: "session-1".to_owned(),
            lease_id: "lease-1".to_owned(),
            lease_expires_at_ms: 100_000,
            work_unit_digest: work.work_unit_digest(),
            command_digest: work.command_digest(),
            attempt: work.attempt,
            attempt_fence: work.attempt_fence,
            hub_epoch: work.hub_epoch,
            control_position: work.control_position,
            capability_advertisement_digest: [8; 32],
            gpu_uuids: vec!["GPU-exact".to_owned()],
            cached_input_bytes: 100,
            isolation_evidence_digest: [7; 32],
            signature: DomainSignature {
                authority: SignatureAuthority::PlacementOffer,
                issuer: "hub-issuer".to_owned(),
                key_id: "hub-key".to_owned(),
                public_key: key.verifying_key().to_bytes(),
                signature: Vec::new(),
            },
        };
        placement.signature = DomainSignature::sign(
            SignatureAuthority::PlacementOffer,
            "hub-issuer".to_owned(),
            "hub-key".to_owned(),
            &key,
            &placement.signed_bytes().unwrap(),
        );
        placement
    }

    fn infer(sequence: u64) -> NodeControlFrame {
        let work = work();
        NodeControlFrame::Infer {
            request: Box::new(InferRequest {
                session_id: "session-1".to_owned(),
                sequence,
                deadline_ms: 90_000,
                work: work.signed_projection().clone(),
                placement: placement(),
            }),
        }
    }

    fn signed_cancel(sequence: u64) -> NodeControlFrame {
        let work = work();
        sign_control(NodeControlFrame::Cancel {
            session_id: "session-1".to_owned(),
            sequence,
            run_id: work.run_id.to_string(),
            attempt: work.attempt,
            attempt_fence: work.attempt_fence,
            signature: empty_hub_signature(),
        })
    }

    fn signed_revoke(sequence: u64) -> NodeControlFrame {
        sign_control(NodeControlFrame::Revoke {
            session_id: "session-1".to_owned(),
            sequence,
            reason: "owner revocation".to_owned(),
            signature: empty_hub_signature(),
        })
    }

    fn signed_rotate(sequence: u64) -> NodeControlFrame {
        sign_control(NodeControlFrame::Rotate {
            session_id: "session-1".to_owned(),
            sequence,
            next_session_id: "session-2".to_owned(),
            expires_at_ms: 90_000,
            signature: empty_hub_signature(),
        })
    }

    fn empty_hub_signature() -> DomainSignature {
        DomainSignature {
            authority: SignatureAuthority::NodeSession,
            issuer: "hub-issuer".to_owned(),
            key_id: "hub-key".to_owned(),
            public_key: hub_key().verifying_key().to_bytes(),
            signature: Vec::new(),
        }
    }

    fn sign_control(mut frame: NodeControlFrame) -> NodeControlFrame {
        let bytes = frame.signed_control_bytes().unwrap();
        let signature = DomainSignature::sign(
            SignatureAuthority::NodeSession,
            "hub-issuer".to_owned(),
            "hub-key".to_owned(),
            &hub_key(),
            &bytes,
        );
        match &mut frame {
            NodeControlFrame::Cancel {
                signature: target, ..
            }
            | NodeControlFrame::Rotate {
                signature: target, ..
            }
            | NodeControlFrame::Revoke {
                signature: target, ..
            } => *target = signature,
            _ => unreachable!(),
        }
        frame
    }

    fn protocol(root: PathBuf) -> NodeProtocolState {
        NodeProtocolState::open(
            root,
            "node-1".to_owned(),
            "session-1".to_owned(),
            NodeProfile::Gpu,
            crate::WorkerProvision::SmartclusterCompute,
            "hub-issuer".to_owned(),
            hub_key().verifying_key().to_bytes(),
            "hub-issuer".to_owned(),
            hub_key().verifying_key().to_bytes(),
            1,
        )
        .unwrap()
    }

    #[test]
    fn recovery_capacity_is_validated_before_any_assignment_launch() {
        let root = tempfile::tempdir().unwrap();
        let runtime = Arc::new(FakeRuntime::default());
        runtime.recovery_count.store(2, Ordering::SeqCst);
        assert!(matches!(
            ProductionControlHandler::new(
                protocol(root.path().join("protocol")),
                Arc::clone(&runtime),
                Arc::new(FixedClock),
            ),
            Err(ControlError::Protocol(ProtocolError::AdmissionDenied))
        ));
        assert!(!runtime.recovered.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn infer_cancel_retry_and_completion_accounting_are_ordered() {
        let root = tempfile::tempdir().unwrap();
        let runtime = Arc::new(FakeRuntime::default());
        let mut handler = ProductionControlHandler::new(
            protocol(root.path().join("protocol")),
            Arc::clone(&runtime),
            Arc::new(FixedClock),
        )
        .unwrap();
        assert_eq!(
            handler.handle(infer(1)).await.unwrap(),
            ControlFlow::Continue
        );
        assert_eq!(runtime.admitted.load(Ordering::SeqCst), 1);
        assert!(matches!(
            handler.handle(infer(2)).await,
            Err(ControlError::Protocol(ProtocolError::AdmissionDenied))
        ));

        let mut wrong_fence = signed_cancel(2);
        let NodeControlFrame::Cancel { attempt_fence, .. } = &mut wrong_fence else {
            unreachable!();
        };
        *attempt_fence += 1;
        assert!(matches!(
            handler.handle(wrong_fence).await,
            Err(ControlError::Protocol(ProtocolError::FenceRejected))
        ));
        let mut forged = signed_cancel(2);
        let NodeControlFrame::Cancel { signature, .. } = &mut forged else {
            unreachable!();
        };
        signature.signature[0] ^= 1;
        assert!(matches!(
            handler.handle(forged).await,
            Err(ControlError::Protocol(ProtocolError::SessionRevoked))
        ));
        assert!(!runtime.cancelled.load(Ordering::SeqCst));

        runtime.fail_cancel_once.store(true, Ordering::SeqCst);
        let cancel = signed_cancel(3);
        assert!(matches!(
            handler.handle(cancel.clone()).await,
            Err(ControlError::Execution(ExecutionError::CancellationState))
        ));
        assert_eq!(
            handler.handle(cancel.clone()).await.unwrap(),
            ControlFlow::Continue
        );
        assert!(runtime.cancelled.load(Ordering::SeqCst));
        assert!(matches!(
            handler.handle(cancel).await,
            Err(ControlError::Protocol(ProtocolError::ReplayedPosition))
        ));

        runtime.complete_one();
        assert_eq!(
            handler.handle(signed_rotate(4)).await.unwrap(),
            ControlFlow::Stop
        );
        assert_eq!(handler.protocol().session_id(), "session-2");
    }

    #[tokio::test]
    async fn authenticated_storage_control_stays_fail_closed_without_composition() {
        let root = tempfile::tempdir().unwrap();
        let runtime = Arc::new(FakeRuntime::default());
        let mut handler = ProductionControlHandler::new(
            protocol(root.path().join("protocol")),
            runtime,
            Arc::new(FixedClock),
        )
        .unwrap();
        let request = SignedStorageControlRequest::sign(
            "hub-issuer".to_owned(),
            "hub-key".to_owned(),
            &hub_key(),
            "session-1".to_owned(),
            1,
            60_000,
            StorageControlBody::CommitWrite(CommitFragmentWrite {
                transfer_id: "transfer-1".to_owned(),
            }),
        )
        .unwrap();
        let frame = NodeControlFrame::storage_control(request).unwrap();
        assert!(matches!(
            handler.handle(frame.clone()).await,
            Err(ControlError::StorageCompositionUnavailable)
        ));
        assert_eq!(handler.protocol().control_position(), 0);
        assert!(matches!(
            handler.handle(frame).await,
            Err(ControlError::StorageCompositionUnavailable)
        ));

        let mut forged_request = SignedStorageControlRequest::sign(
            "hub-issuer".to_owned(),
            "hub-key".to_owned(),
            &hub_key(),
            "session-1".to_owned(),
            2,
            60_000,
            StorageControlBody::CommitWrite(CommitFragmentWrite {
                transfer_id: "transfer-2".to_owned(),
            }),
        )
        .unwrap();
        forged_request.signature.signature[0] ^= 1;
        assert!(matches!(
            handler
                .handle(NodeControlFrame::StorageControl {
                    request: Box::new(forged_request),
                })
                .await,
            Err(ControlError::Protocol(ProtocolError::InvalidStorageControl))
        ));
    }

    #[tokio::test]
    async fn revoke_fences_runtime_before_durable_session_commit() {
        let root = tempfile::tempdir().unwrap();
        let state_root = root.path().join("protocol");
        let runtime = Arc::new(FakeRuntime::with_active(work()));
        runtime.fail_revoke_once.store(true, Ordering::SeqCst);
        let mut handler = ProductionControlHandler::new(
            protocol(state_root.clone()),
            Arc::clone(&runtime),
            Arc::new(FixedClock),
        )
        .unwrap();
        let revoke = signed_revoke(5);
        assert!(matches!(
            handler.handle(revoke.clone()).await,
            Err(ControlError::Execution(ExecutionError::CancellationState))
        ));
        assert!(!handler.protocol().revoked());
        assert_eq!(handler.handle(revoke).await.unwrap(), ControlFlow::Stop);
        assert!(runtime.revoked.load(Ordering::SeqCst));
        drop(handler);
        assert!(protocol(state_root).revoked());
    }
}

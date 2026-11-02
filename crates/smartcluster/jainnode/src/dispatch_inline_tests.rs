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

    #[path = "cases.rs"]
    mod cases;
}

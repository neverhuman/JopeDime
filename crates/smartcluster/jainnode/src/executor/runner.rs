use super::*;

impl<D, S, N, C, M, E, I, K> JainNodeExecutor<D, S, N, C, M, E, I, K>
where
    D: DockerEnginePort + 'static,
    S: ShardPort,
    N: NetworkPolicyPort,
    C: CancellationPort,
    M: MetricsPort,
    E: EventSink,
    I: ContainerIsolationPort,
    K: ClockPort,
{
    pub fn new(
        ports: ExecutorPorts<D, S, N, C, M, E, I, K>,
        isolation: IsolationEvidence,
        identity: NodeRuntimeIdentity,
        assignment_journal: AssignmentJournal,
    ) -> Result<Self, ExecutionError> {
        if !isolation.qualified() {
            return Err(ExecutionError::UnsupportedHost);
        }
        Ok(Self {
            docker: ports.docker,
            shard: ports.shard,
            network: ports.network,
            cancellation: ports.cancellation,
            metrics: ports.metrics,
            events: ports.events,
            isolation,
            identity,
            assignment_journal,
            isolation_enforcer: ports.isolation_enforcer,
            clock: ports.clock,
        })
    }

    pub fn accept_assignment(
        &self,
        work: VerifiedWorkUnit,
        placement: Placement,
        session_sequence: u64,
        now_ms: u64,
    ) -> Result<
        (
            AcceptedAssignment,
            jain_smartcluster_fabric::WorkerAcknowledgement,
        ),
        ExecutionError,
    > {
        if placement.isolation_evidence_digest != self.isolation.digest {
            return Err(ExecutionError::UnsupportedHost);
        }
        self.assignment_journal
            .accept_before_ack(&self.identity, work, placement, session_sequence, now_ms)
            .map_err(ExecutionError::Assignment)
    }

    pub fn recover_assignments(&self, now_ms: u64) -> Result<RecoveryReport, ExecutionError> {
        self.assignment_journal
            .recover(&self.identity, now_ms)
            .map_err(ExecutionError::Assignment)
    }

    pub(crate) fn recovered_acknowledgement(
        &self,
        accepted: &AcceptedAssignment,
    ) -> Result<jain_smartcluster_fabric::WorkerAcknowledgement, ExecutionError> {
        accepted
            .acknowledgement(&self.identity)
            .map_err(ExecutionError::Assignment)
    }

    pub(crate) fn reject_prepared(
        &self,
        accepted: &AcceptedAssignment,
    ) -> Result<(), ExecutionError> {
        self.assignment_journal
            .mark_rejected(accepted)
            .map_err(ExecutionError::Assignment)
    }

    pub async fn execute(
        &self,
        accepted: &AcceptedAssignment,
    ) -> Result<ExecutionReceipt, ExecutionError> {
        let work = accepted.work();
        let placement = accepted.placement();
        work.validate()
            .map_err(|_| ExecutionError::InvalidWorkUnit)?;
        if placement.isolation_evidence_digest != self.isolation.digest {
            return Err(ExecutionError::UnsupportedHost);
        }
        self.assignment_journal
            .mark_running(accepted)
            .map_err(ExecutionError::Assignment)?;
        let run_id = work.run_id.to_string();
        let mut emitter = Emitter::new(work, ExecutionPhase::Smoke);
        let inputs = self
            .shard
            .stage_inputs(&work.project_id, &run_id, &work.inputs)
            .await?;
        if inputs.len() != work.inputs.len()
            || inputs.iter().any(|input| !input.read_only)
            || inputs.iter().any(|input| {
                !input.host_path.is_absolute()
                    || !input.container_path.starts_with("/inputs/")
                    || input.container_path.contains("..")
            })
        {
            let _ = self.shard.cleanup_staging(&run_id).await;
            return Err(ExecutionError::InvalidStaging);
        }

        let smoke_result = self
            .execute_phase(
                work,
                placement,
                &inputs,
                ExecutionPhase::Smoke,
                &mut emitter,
            )
            .await;
        let (smoke_logs, _) = match smoke_result {
            Ok(result) => result,
            Err(error) => {
                let _ = self.shard.cleanup_staging(&run_id).await;
                return Err(error);
            }
        };
        if work.smoke_only {
            self.shard.cleanup_staging(&run_id).await?;
            self.assignment_journal
                .mark_terminal(accepted)
                .map_err(ExecutionError::Assignment)?;
            return Ok(ExecutionReceipt {
                outputs: Vec::new(),
                logs: smoke_logs,
                smoke_only: true,
            });
        }

        let run_result = self
            .execute_phase(work, placement, &inputs, ExecutionPhase::Run, &mut emitter)
            .await;
        let (run_logs, output_staging) = match run_result {
            Ok(result) => result,
            Err(error) => {
                let _ = self.shard.cleanup_staging(&run_id).await;
                return Err(error);
            }
        };
        let declarations = work
            .outputs
            .iter()
            .map(|output| ArtifactPublication {
                relative_path: output.relative_path.clone(),
                media_type: output.media_type.clone(),
                maximum_bytes: output.maximum_bytes,
                required: output.required,
            })
            .collect::<Vec<_>>();
        let outputs_result = self
            .shard
            .seal_outputs(
                &work.project_id,
                &run_id,
                &output_staging,
                &declarations,
                work.minimum_writable_domains,
            )
            .await;
        let outputs = match outputs_result {
            Ok(outputs) => outputs,
            Err(error) => {
                let _ = self.shard.cleanup_staging(&run_id).await;
                return Err(error);
            }
        };
        let published_paths = outputs
            .iter()
            .map(|output| output.relative_path.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        if declarations
            .iter()
            .filter(|output| output.required)
            .any(|output| !published_paths.contains(output.relative_path.as_str()))
            || outputs.iter().any(|artifact| {
                artifact.durable_receipt_digest == [0; 32]
                    || artifact.writable_domains < work.minimum_writable_domains
            })
        {
            let _ = self.shard.cleanup_staging(&run_id).await;
            return Err(ExecutionError::DurabilityUnavailable);
        }
        for artifact in &outputs {
            self.events
                .emit(NodeEvent::ArtifactPublished {
                    identity: emitter.next(0),
                    artifact: artifact.clone(),
                })
                .await?;
        }
        let mut logs = smoke_logs;
        logs.extend(run_logs);
        self.shard.cleanup_staging(&run_id).await?;
        self.assignment_journal
            .mark_terminal(accepted)
            .map_err(ExecutionError::Assignment)?;
        Ok(ExecutionReceipt {
            outputs,
            logs,
            smoke_only: false,
        })
    }

    pub(crate) fn terminalize_failure(
        &self,
        accepted: &AcceptedAssignment,
    ) -> Result<(), ExecutionError> {
        self.assignment_journal
            .mark_terminal(accepted)
            .map_err(ExecutionError::Assignment)
    }

    async fn execute_phase(
        &self,
        work: &VerifiedWorkUnit,
        placement: &Placement,
        inputs: &[PreparedInput],
        phase: ExecutionPhase,
        emitter: &mut Emitter,
    ) -> Result<(Vec<PublishedArtifact>, PathBuf), ExecutionError> {
        let run_id = work.run_id.to_string();
        let output_staging = self
            .shard
            .allocate_output_staging(&work.project_id, &run_id, phase)
            .await?;
        if !output_staging.is_absolute() {
            return Err(ExecutionError::InvalidStaging);
        }
        let network = self
            .network
            .allocate(
                &run_id,
                work.open_outbound,
                work.network_bytes,
                work.network_connections,
                work.network_deadline_ms,
            )
            .await?;
        if network.run_id != run_id
            || network.name.is_empty()
            || matches!(
                network.name.as_str(),
                "host" | "bridge" | "default" | "none"
            )
            || network.network_id.is_empty()
            || network.policy_digest == [0; 32]
        {
            let _ = self.network.revoke(&network).await;
            return Err(ExecutionError::NetworkPolicyUnavailable);
        }

        let argv = match phase {
            ExecutionPhase::Smoke => work.smoke_argv.clone(),
            ExecutionPhase::Run => work.argv.clone(),
        };
        let mut environment = work.caller_environment.clone();
        environment.extend(platform_environment(work, placement, phase));
        let spec = DockerRunSpec {
            name: format!(
                "jain-{}-{}-{}",
                work.run_id,
                work.attempt,
                phase_name(phase)
            ),
            image: work.image_reference.clone(),
            argv,
            working_directory: work.working_directory.clone(),
            environment,
            readonly_binds: inputs
                .iter()
                .map(|input| (input.host_path.clone(), input.container_path.clone()))
                .collect(),
            output_staging: output_staging.clone(),
            scratch_bytes: work.limits.scratch_bytes,
            cpu_millicores: work.limits.cpu_millicores,
            memory_bytes: work.limits.memory_bytes,
            pids: work.limits.pids,
            read_bytes: work.limits.read_bytes,
            write_bytes: work.limits.write_bytes,
            io_operations: work.limits.io_operations,
            gpu_uuids: placement.gpu_uuids.clone(),
            network_name: network.name.clone(),
            seccomp_profile: self.isolation.seccomp_profile.clone(),
            apparmor_profile: self.isolation.apparmor_profile.clone(),
        };
        emitter.phase = phase;
        let result = self
            .run_container(work, placement, phase, &spec, emitter)
            .await;
        let revoke_result = self.network.revoke(&network).await;
        match (result, revoke_result) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok((stdout, stderr)), Ok(())) => {
                let logs = self
                    .shard
                    .seal_log(
                        &work.project_id,
                        &run_id,
                        phase,
                        &stdout,
                        &stderr,
                        work.minimum_writable_domains,
                    )
                    .await?;
                if logs.iter().any(|artifact| {
                    artifact.durable_receipt_digest == [0; 32]
                        || artifact.writable_domains < work.minimum_writable_domains
                }) {
                    return Err(ExecutionError::DurabilityUnavailable);
                }
                Ok((logs, output_staging))
            }
        }
    }

    async fn run_container(
        &self,
        work: &VerifiedWorkUnit,
        placement: &Placement,
        phase: ExecutionPhase,
        spec: &DockerRunSpec,
        emitter: &mut Emitter,
    ) -> Result<(Vec<u8>, Vec<u8>), ExecutionError> {
        let container_id = self.docker.create(spec).await?;
        let outcome = async {
            let isolation_receipt = self
                .isolation_enforcer
                .enforce_stopped(
                    &container_id,
                    spec,
                    self.isolation.digest,
                    work.work_unit_digest(),
                    placement
                        .digest()
                        .map_err(|_| ExecutionError::InvalidWorkUnit)?,
                )
                .await?;
            let placement_digest = placement
                .digest()
                .map_err(|_| ExecutionError::InvalidWorkUnit)?;
            if isolation_receipt.container_id != container_id
                || isolation_receipt.qualification_digest != self.isolation.digest
                || isolation_receipt.work_unit_digest != work.work_unit_digest()
                || isolation_receipt.placement_digest != placement_digest
                || isolation_receipt.gpu_uuids != placement.gpu_uuids
                || !isolation_receipt.cgroup_path.is_absolute()
                || !isolation_receipt.stopped_before_executable_bytes
                || !isolation_receipt.cgroup_owned
                || !isolation_receipt.cgroup_membership_verified
                || !isolation_receipt.descendant_fencing
                || !isolation_receipt.device_filter_enforced
                || !isolation_receipt.io_limits_enforced
            {
                return Err(ExecutionError::UnsupportedHost);
            }
            self.identity
                .reverify_work(work)
                .map_err(ExecutionError::Assignment)?;
            self.identity
                .reverify_placement(work, placement, self.clock.now_ms()?)
                .map_err(|_| ExecutionError::StaleAuthority)?;
            if phase == ExecutionPhase::Smoke {
                self.events
                    .emit(NodeEvent::SmokeStarted(emitter.next(0)))
                    .await?;
            }
            let mut logs = self.docker.logs(&container_id).await?;
            self.cancellation
                .start_unless_cancelled(
                    &*self.docker,
                    &container_id,
                    &work.run_id.to_string(),
                    work.attempt,
                    work.attempt_fence,
                )
                .await?;
            let initial_metrics = self.metrics.sample(&container_id, &spec.gpu_uuids).await?;
            self.events
                .emit(NodeEvent::Metrics {
                    identity: emitter.next(0),
                    cpu_millis: initial_metrics.cpu_millis,
                    memory_bytes: initial_metrics.memory_bytes,
                    gpu_millis: initial_metrics.gpu_millis,
                })
                .await?;
            let timeout_ms = match phase {
                ExecutionPhase::Smoke => work.smoke_timeout_ms,
                ExecutionPhase::Run => work.limits.wall_time_ms,
            };
            let mut wait = Box::pin(self.docker.wait(&container_id));
            let cancellation_run_id = work.run_id.to_string();
            let mut cancel = Box::pin(self.cancellation.cancelled(
                &cancellation_run_id,
                work.attempt,
                work.attempt_fence,
            ));
            let deadline = tokio::time::sleep(Duration::from_millis(timeout_ms));
            tokio::pin!(deadline);
            let metrics_tick = tokio::time::interval(Duration::from_secs(1));
            tokio::pin!(metrics_tick);
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            let mut logs_complete = false;
            let mut status = None;
            loop {
                if logs_complete && status.is_some() {
                    break;
                }
                tokio::select! {
                    result = &mut wait, if status.is_none() => status = Some(result?),
                    () = &mut cancel => {
                        self.docker.kill(&container_id).await?;
                        return Err(ExecutionError::Cancelled);
                    }
                    () = &mut deadline => {
                        self.docker.kill(&container_id).await?;
                        return Err(ExecutionError::DeadlineExceeded);
                    }
                    event = logs.recv(), if !logs_complete => {
                        match event {
                            Some(Ok(DockerLogEvent::Chunk(chunk))) => {
                                record_and_emit_chunk(
                                    &*self.events,
                                    emitter,
                                    chunk,
                                    &mut stdout,
                                    &mut stderr,
                                    work.stdout_bytes,
                                    work.stderr_bytes,
                                ).await?;
                            }
                            Some(Ok(DockerLogEvent::Complete)) => logs_complete = true,
                            Some(Err(error)) => return Err(error.into()),
                            None => return Err(ExecutionError::StreamProtocolViolation),
                        }
                    }
                    _ = metrics_tick.tick() => {
                        let sample = self.metrics.sample(&container_id, &spec.gpu_uuids).await?;
                        self.events.emit(NodeEvent::Metrics {
                            identity: emitter.next(0),
                            cpu_millis: sample.cpu_millis,
                            memory_bytes: sample.memory_bytes,
                            gpu_millis: sample.gpu_millis,
                        }).await?;
                    }
                }
            }
            ensure_success(status.ok_or(ExecutionError::StreamProtocolViolation)?)?;
            if phase == ExecutionPhase::Smoke {
                self.events
                    .emit(NodeEvent::SmokePassed(emitter.next(0)))
                    .await?;
            }
            Ok((stdout, stderr))
        }
        .await;
        let remove = self.docker.remove(&container_id).await;
        match (outcome, remove) {
            (Err(error), _) => {
                if phase == ExecutionPhase::Smoke {
                    let _ = self
                        .events
                        .emit(NodeEvent::SmokeFailed {
                            identity: emitter.next(0),
                            reason: error.to_string(),
                        })
                        .await;
                }
                Err(error)
            }
            (Ok(_), Err(error)) => Err(error.into()),
            (Ok(output), Ok(())) => Ok(output),
        }
    }
}

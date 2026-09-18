use super::*;

use crate::gpu::MpsClientLimits;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MpsTerminationOutcome {
    Terminated(usize),
    NoConnectedClient,
    CgroupQuiesced,
}

pub(super) fn allocation_uses_mps(allocation: &Allocation) -> bool {
    allocation
        .gpu
        .iter()
        .any(|gpu| gpu.mode == GpuAllocationMode::SharedMps)
}

pub(super) fn job_may_use_mps(job: &JobRecord) -> bool {
    job.spec
        .resources
        .gpu
        .is_some_and(|request| request.mode == jain_smartcluster_core::GpuMode::Shareable)
}

impl DaemonActor {
    pub(super) async fn dispose_uninstalled_workload(
        &mut self,
        running: crate::executor::RunningWorkload,
    ) -> Result<(), ActorError> {
        // `Executor::launch` returns a SIGSTOP-fenced helper. Until `start` succeeds,
        // user code and CUDA cannot have run, so direct cleanup cannot affect MPS peers.
        running.force_kill().await;
        Ok(())
    }

    pub(super) fn mps_launch_environment(
        &self,
        allocation: &Allocation,
    ) -> Result<BTreeMap<String, String>, ActorError> {
        let shared: Vec<_> = allocation
            .gpu
            .iter()
            .filter(|gpu| gpu.mode == GpuAllocationMode::SharedMps)
            .collect();
        if shared.is_empty() {
            return Ok(BTreeMap::new());
        }
        if shared.len() != allocation.gpu.len() {
            return Err(ActorError::MpsSafety(
                "mixed shared and isolated GPU allocations are not launchable".to_owned(),
            ));
        }
        let controller = self.mps_controller.as_ref().ok_or_else(|| {
            ActorError::MpsSafety(
                "scheduler selected shared MPS without a probed controller".to_owned(),
            )
        })?;
        let compute_share_per_mille = shared
            .iter()
            .map(|gpu| gpu.compute_share_per_mille)
            .min()
            .ok_or_else(|| ActorError::MpsSafety("shared MPS allocation is empty".to_owned()))?;
        let vram_bytes_by_device = shared
            .into_iter()
            .map(|gpu| (gpu.device_id.clone(), gpu.reserved_vram_bytes))
            .collect();
        controller
            .client_environment(&MpsClientLimits {
                compute_share_per_mille,
                vram_bytes_by_device,
            })
            .map_err(|error| ActorError::MpsSafety(error.to_string()))
    }

    async fn prepare_mps_termination(
        &mut self,
        job_id: JobId,
        attempt: FencedAttempt,
    ) -> Result<(), ActorError> {
        let Some(active) = self.active.get(&job_id) else {
            return Ok(());
        };
        if active.attempt != attempt || !active.mps_managed {
            return Ok(());
        }
        let result = self.terminate_mps_clients(active.cgroup.clone()).await;
        if let Err(error) = result {
            let message = format!("MPS peer-safe termination failed: {error}");
            self.record_mps_safety_failure(job_id, attempt, &message)?;
            return Err(ActorError::MpsSafety(message));
        }
        tracing::debug!(job_id = %job_id, outcome = ?result.expect("checked above"), "MPS client termination fenced");
        Ok(())
    }

    pub(super) async fn terminate_mps_clients(
        &self,
        cgroup: Option<JobCgroup>,
    ) -> Result<MpsTerminationOutcome, ActorError> {
        let controller = self.mps_controller.clone().ok_or_else(|| {
            ActorError::MpsSafety("probed MPS controller is unavailable".to_owned())
        })?;
        let cgroup = cgroup.ok_or_else(|| {
            ActorError::MpsSafety(
                "shared MPS termination requires delegated cgroup membership".to_owned(),
            )
        })?;
        let total_timeout = controller.command_timeout().saturating_mul(8);
        let operation = async {
            let pids: BTreeSet<_> = self
                .cgroups
                .pids(&cgroup)
                .map_err(|error| crate::gpu::GpuError::MpsCgroupMembership(error.to_string()))?
                .into_iter()
                .collect();
            if pids.is_empty() {
                return Ok(MpsTerminationOutcome::CgroupQuiesced);
            }
            let clients = controller.clients().await?;
            let mut identities = BTreeSet::new();
            for client in &clients {
                if crate::gpu::mps_client_matches_pids(client, &pids)? {
                    identities.insert((client.server_pid, client.pid));
                }
            }
            if identities.is_empty() {
                // Require a second complete `ps` snapshot with stable cgroup membership.
                // This is the explicit proof that a launch which has not initialized CUDA
                // has no MPS client to terminate; a missing/ambiguous snapshot fails closed.
                let confirmation = controller.clients().await?;
                let confirmed_pids: BTreeSet<_> = self
                    .cgroups
                    .pids(&cgroup)
                    .map_err(|error| crate::gpu::GpuError::MpsCgroupMembership(error.to_string()))?
                    .into_iter()
                    .collect();
                if confirmed_pids != pids {
                    return Err(crate::gpu::GpuError::MpsMembershipChanged);
                }
                for client in &confirmation {
                    if crate::gpu::mps_client_matches_pids(client, &pids)? {
                        identities.insert((client.server_pid, client.pid));
                    }
                }
                if identities.is_empty() {
                    return Ok(MpsTerminationOutcome::NoConnectedClient);
                }
            }
            let terminated = identities.len();
            for (server_pid, client_pid) in &identities {
                controller
                    .terminate_client(*server_pid, *client_pid)
                    .await?;
            }
            let remaining = controller.clients().await?;
            for client in &remaining {
                if crate::gpu::mps_client_matches_pids(client, &pids)? {
                    return Err(crate::gpu::GpuError::MpsClientStillConnected(client.pid));
                }
            }
            Ok::<MpsTerminationOutcome, crate::gpu::GpuError>(MpsTerminationOutcome::Terminated(
                terminated,
            ))
        };
        match tokio::time::timeout(total_timeout, operation).await {
            Ok(Ok(outcome)) => Ok(outcome),
            Ok(Err(error)) => Err(ActorError::MpsSafety(error.to_string())),
            Err(_) => Err(ActorError::MpsSafety(format!(
                "MPS termination exceeded the configured total bound of {total_timeout:?}"
            ))),
        }
    }

    fn record_mps_safety_failure(
        &mut self,
        job_id: JobId,
        attempt: FencedAttempt,
        message: &str,
    ) -> Result<(), ActorError> {
        self.gpu.mps.disable(message.to_owned());
        self.health.disable_mps(message.to_owned());
        if let Some(active) = self.active.get_mut(&job_id) {
            active.pending_failure = Some((FailureClass::Infrastructure, message.to_owned()));
        }
        self.journal.append(
            job_id,
            Some(attempt),
            JobEvent::Failure {
                class: FailureClass::Infrastructure,
                message: message.to_owned(),
            },
            now_timestamp(),
        )?;
        Ok(())
    }

    pub(super) async fn signal_active_safely(
        &mut self,
        job_id: JobId,
        attempt: FencedAttempt,
        signal: Signal,
    ) -> Result<(), ActorError> {
        if self
            .active
            .get(&job_id)
            .is_some_and(|active| active.attempt == attempt && active.mps_managed)
            && !matches!(signal, Signal::SIGTERM | Signal::SIGKILL)
        {
            return Err(ActorError::MpsSafety(format!(
                "refusing unproven signal {signal:?} for a shared MPS client"
            )));
        }
        if matches!(signal, Signal::SIGTERM | Signal::SIGKILL) {
            self.prepare_mps_termination(job_id, attempt).await?;
        }
        let Some(active) = self.active.get(&job_id) else {
            return Ok(());
        };
        if active.attempt != attempt {
            return Ok(());
        }
        signal_active(&self.cgroups, active, signal)?;
        Ok(())
    }

    pub(super) async fn kill_active_safely(
        &mut self,
        job_id: JobId,
        attempt: FencedAttempt,
    ) -> Result<(), ActorError> {
        self.prepare_mps_termination(job_id, attempt).await?;
        let Some(active) = self.active.get(&job_id) else {
            return Ok(());
        };
        if active.attempt != attempt {
            return Ok(());
        }
        if let Some(cgroup) = &active.cgroup {
            self.cgroups.kill(cgroup)?;
        } else if (active.leader_exit.is_some() && process_group_exists(active.pid))
            || (active.leader_exit.is_none() && same_process(active.pid, active.pid_start_time))
        {
            killpg(Pid::from_raw(active.pid as i32), Signal::SIGKILL)?;
        }
        Ok(())
    }

    pub(super) fn queue_termination(
        &self,
        job_id: JobId,
        attempt: FencedAttempt,
        delay: Duration,
        grace: Duration,
    ) {
        self.queue_termination_attempt(job_id, attempt, delay, grace, 0);
    }

    fn queue_termination_attempt(
        &self,
        job_id: JobId,
        attempt: FencedAttempt,
        delay: Duration,
        grace: Duration,
        retry: u8,
    ) {
        let sender = self.sender.clone();
        tokio::spawn(async move {
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            let _ = sender
                .send(ActorCommand::Terminate {
                    job_id,
                    attempt,
                    grace,
                    retry,
                })
                .await;
        });
    }

    pub(super) fn retry_termination(
        &self,
        job_id: JobId,
        attempt: FencedAttempt,
        grace: Duration,
        retry: u8,
    ) {
        if !self
            .active
            .get(&job_id)
            .is_some_and(|active| active.attempt == attempt)
        {
            return;
        }
        let next = retry.saturating_add(1);
        self.queue_termination_attempt(job_id, attempt, effect_retry_delay(next), grace, next);
    }

    pub(super) fn retry_hard_kill(&self, job_id: JobId, attempt: FencedAttempt, retry: u8) {
        if !self
            .active
            .get(&job_id)
            .is_some_and(|active| active.attempt == attempt)
        {
            return;
        }
        let sender = self.sender.clone();
        let next = retry.saturating_add(1);
        tokio::spawn(async move {
            tokio::time::sleep(effect_retry_delay(next)).await;
            let _ = sender
                .send(ActorCommand::HardKill {
                    job_id,
                    attempt,
                    retry: next,
                })
                .await;
        });
    }

    pub(super) async fn handle_terminate(
        &mut self,
        job_id: JobId,
        attempt: FencedAttempt,
        grace: Duration,
    ) -> Result<(), ActorError> {
        self.signal_active_safely(job_id, attempt, Signal::SIGTERM)
            .await?;
        let sender = self.sender.clone();
        tokio::spawn(async move {
            tokio::time::sleep(grace).await;
            let _ = sender
                .send(ActorCommand::HardKill {
                    job_id,
                    attempt,
                    retry: 0,
                })
                .await;
        });
        Ok(())
    }

    pub(super) async fn handle_hard_kill(
        &mut self,
        job_id: JobId,
        attempt: FencedAttempt,
    ) -> Result<(), ActorError> {
        self.kill_active_safely(job_id, attempt).await
    }
}

pub(super) fn effect_retry_delay(retry: u8) -> Duration {
    Duration::from_millis(50_u64.saturating_mul(1_u64 << retry.min(5)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::GpuAllocation;

    #[test]
    fn shared_allocations_are_detected_without_counting_exclusive_devices() {
        let allocation = Allocation {
            resources: test_resources(),
            gpu: vec![GpuAllocation {
                device_id: "GPU-a".to_owned(),
                mode: GpuAllocationMode::SharedMps,
                reserved_vram_bytes: 1024 * 1024,
                compute_share_per_mille: 500,
            }],
            backfill: false,
        };
        assert!(allocation_uses_mps(&allocation));
    }

    fn test_resources() -> jain_smartcluster_core::ResourceRequest {
        jain_smartcluster_core::ResourceRequest {
            cpu: jain_smartcluster_core::CpuRequest::fixed(1_000),
            ram: jain_smartcluster_core::MemoryRequest::fixed(1024),
            scratch_bytes: 0,
            gpu: None,
        }
    }
}

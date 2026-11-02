#[derive(Default)]
struct DockerMock {
    specs: Mutex<Vec<DockerRunSpec>>,
    exits: Mutex<Vec<ExitStatus>>,
    killed: Mutex<usize>,
    removed: Mutex<usize>,
    started: Mutex<Vec<String>>,
    waiting: Mutex<Vec<String>>,
    poll_wait: Notify,
    wait_observed: Notify,
    block_wait: bool,
    block_run_wait: bool,
    log_error: bool,
    incomplete_log: bool,
}

#[async_trait]
impl DockerEnginePort for DockerMock {
    async fn create(&self, spec: &DockerRunSpec) -> Result<String, DockerError> {
        self.specs.lock().unwrap().push(spec.clone());
        Ok(format!("container-{}", self.specs.lock().unwrap().len()))
    }

    async fn start(&self, container_id: &str) -> Result<(), DockerError> {
        self.started.lock().unwrap().push(container_id.to_owned());
        Ok(())
    }

    async fn logs(
        &self,
        _container_id: &str,
    ) -> Result<mpsc::Receiver<Result<DockerLogEvent, DockerError>>, DockerError> {
        let (sender, receiver) = mpsc::channel(4);
        if self.log_error {
            sender
                .send(Err(DockerError::Transport(
                    "injected log failure".to_owned(),
                )))
                .await
                .unwrap();
            return Ok(receiver);
        }
        sender
            .send(Ok(DockerLogEvent::Chunk(DockerLogChunk {
                stderr: false,
                bytes: b"stdout\n".to_vec(),
            })))
            .await
            .unwrap();
        sender
            .send(Ok(DockerLogEvent::Chunk(DockerLogChunk {
                stderr: true,
                bytes: b"stderr\n".to_vec(),
            })))
            .await
            .unwrap();
        if !self.incomplete_log {
            sender.send(Ok(DockerLogEvent::Complete)).await.unwrap();
        }
        drop(sender);
        Ok(receiver)
    }

    async fn wait(&self, container_id: &str) -> Result<ExitStatus, DockerError> {
        if self.block_wait || (self.block_run_wait && container_id == "container-2") {
            loop {
                self.waiting.lock().unwrap().push(container_id.to_owned());
                self.wait_observed.notify_one();
                self.poll_wait.notified().await;
            }
        }
        self.waiting.lock().unwrap().push(container_id.to_owned());
        self.wait_observed.notify_one();
        Ok(self.exits.lock().unwrap().remove(0))
    }

    async fn kill(&self, _container_id: &str) -> Result<(), DockerError> {
        *self.killed.lock().unwrap() += 1;
        self.wait_observed.notify_one();
        Ok(())
    }

    async fn remove(&self, _container_id: &str) -> Result<(), DockerError> {
        *self.removed.lock().unwrap() += 1;
        Ok(())
    }
}

#[derive(Default)]
struct ShardMock {
    durable: bool,
    output_seals: Mutex<usize>,
    cleanups: Mutex<usize>,
}

#[async_trait]
impl ShardPort for ShardMock {
    async fn stage_inputs(
        &self,
        _project_id: &str,
        _run_id: &str,
        inputs: &[ArtifactMount],
    ) -> Result<Vec<PreparedInput>, ExecutionError> {
        Ok(inputs
            .iter()
            .map(|input| PreparedInput {
                artifact_id: input.artifact_id.clone(),
                host_path: PathBuf::from(format!("/var/lib/jainnode/inputs/{}", input.artifact_id)),
                container_path: input.mount_path.clone(),
                read_only: true,
            })
            .collect())
    }

    async fn allocate_output_staging(
        &self,
        _project_id: &str,
        run_id: &str,
        phase: ExecutionPhase,
    ) -> Result<PathBuf, ExecutionError> {
        Ok(PathBuf::from(format!(
            "/var/lib/jainnode/outputs/{run_id}/{}",
            phase_name(phase)
        )))
    }

    async fn seal_outputs(
        &self,
        _project_id: &str,
        _run_id: &str,
        _staging: &Path,
        outputs: &[ArtifactPublication],
        minimum_writable_domains: u16,
    ) -> Result<Vec<PublishedArtifact>, ExecutionError> {
        *self.output_seals.lock().unwrap() += 1;
        Ok(outputs
            .iter()
            .map(|output| PublishedArtifact {
                artifact_id: format!("output-{}", output.relative_path),
                relative_path: output.relative_path.clone(),
                digest: [20; 32],
                durable_receipt_digest: if self.durable { [21; 32] } else { [0; 32] },
                writable_domains: if self.durable {
                    minimum_writable_domains
                } else {
                    0
                },
            })
            .collect())
    }

    async fn seal_log(
        &self,
        _project_id: &str,
        _run_id: &str,
        phase: ExecutionPhase,
        _stdout: &[u8],
        _stderr: &[u8],
        minimum_writable_domains: u16,
    ) -> Result<Vec<PublishedArtifact>, ExecutionError> {
        Ok(vec![PublishedArtifact {
            artifact_id: format!("{}-log", phase_name(phase)),
            relative_path: format!("{}.log", phase_name(phase)),
            digest: [22; 32],
            durable_receipt_digest: if self.durable { [23; 32] } else { [0; 32] },
            writable_domains: if self.durable {
                minimum_writable_domains
            } else {
                0
            },
        }])
    }

    async fn cleanup_staging(&self, _run_id: &str) -> Result<(), ExecutionError> {
        *self.cleanups.lock().unwrap() += 1;
        Ok(())
    }
}

#[derive(Default)]
struct NetworkMock {
    allocations: Mutex<usize>,
    revocations: Mutex<usize>,
}

#[async_trait]
impl NetworkPolicyPort for NetworkMock {
    async fn allocate(
        &self,
        run_id: &str,
        _open_outbound: bool,
        _byte_limit: u64,
        _connection_limit: u32,
        _deadline_ms: u64,
    ) -> Result<NetworkLease, ExecutionError> {
        *self.allocations.lock().unwrap() += 1;
        Ok(NetworkLease {
            run_id: run_id.to_owned(),
            name: "jain-dedicated-network".to_owned(),
            network_id: "network-id-1".to_owned(),
            policy_digest: [24; 32],
        })
    }

    async fn revoke(&self, _lease: &NetworkLease) -> Result<(), ExecutionError> {
        *self.revocations.lock().unwrap() += 1;
        Ok(())
    }
}

struct NeverCancel;

#[async_trait]
impl CancellationPort for NeverCancel {
    async fn start_unless_cancelled(
        &self,
        docker: &dyn DockerEnginePort,
        container_id: &str,
        _run_id: &str,
        _attempt: u32,
        _fence: u64,
    ) -> Result<(), ExecutionError> {
        docker.start(container_id).await.map_err(Into::into)
    }

    async fn cancelled(&self, _run_id: &str, _attempt: u32, _fence: u64) {
        pending::<()>().await;
    }
}

struct ImmediateCancel;

#[async_trait]
impl CancellationPort for ImmediateCancel {
    async fn start_unless_cancelled(
        &self,
        docker: &dyn DockerEnginePort,
        container_id: &str,
        _run_id: &str,
        _attempt: u32,
        _fence: u64,
    ) -> Result<(), ExecutionError> {
        docker.start(container_id).await.map_err(Into::into)
    }

    async fn cancelled(&self, _run_id: &str, _attempt: u32, _fence: u64) {}
}

struct AlreadyCancelled;

#[async_trait]
impl CancellationPort for AlreadyCancelled {
    async fn start_unless_cancelled(
        &self,
        _docker: &dyn DockerEnginePort,
        _container_id: &str,
        _run_id: &str,
        _attempt: u32,
        _fence: u64,
    ) -> Result<(), ExecutionError> {
        Err(ExecutionError::Cancelled)
    }

    async fn cancelled(&self, _run_id: &str, _attempt: u32, _fence: u64) {}
}

struct MetricsMock;

#[async_trait]
impl MetricsPort for MetricsMock {
    async fn sample(
        &self,
        _container_id: &str,
        _gpu_uuids: &[String],
    ) -> Result<MetricsSample, ExecutionError> {
        Ok(MetricsSample {
            cpu_millis: 2,
            memory_bytes: 3,
            gpu_millis: 4,
        })
    }
}

struct FixedClock(u64);

impl ClockPort for FixedClock {
    fn now_ms(&self) -> Result<u64, ExecutionError> {
        Ok(self.0)
    }
}

struct IsolationMock;

#[async_trait]
impl ContainerIsolationPort for IsolationMock {
    async fn enforce_stopped(
        &self,
        container_id: &str,
        _spec: &DockerRunSpec,
        qualification_digest: Digest32,
        work_unit_digest: Digest32,
        placement_digest: Digest32,
    ) -> Result<ContainerIsolationReceipt, ExecutionError> {
        Ok(ContainerIsolationReceipt {
            container_id: container_id.to_owned(),
            qualification_digest,
            work_unit_digest,
            placement_digest,
            gpu_uuids: _spec.gpu_uuids.clone(),
            cgroup_path: PathBuf::from(format!("/sys/fs/cgroup/jain/{container_id}")),
            stopped_before_executable_bytes: true,
            cgroup_owned: true,
            cgroup_membership_verified: true,
            descendant_fencing: true,
            device_filter_enforced: true,
            io_limits_enforced: true,
        })
    }
}

struct OrderingIsolationMock {
    docker: Arc<DockerMock>,
}

#[async_trait]
impl ContainerIsolationPort for OrderingIsolationMock {
    async fn enforce_stopped(
        &self,
        container_id: &str,
        spec: &DockerRunSpec,
        qualification_digest: Digest32,
        work_unit_digest: Digest32,
        placement_digest: Digest32,
    ) -> Result<ContainerIsolationReceipt, ExecutionError> {
        assert!(!self
            .docker
            .started
            .lock()
            .unwrap()
            .iter()
            .any(|started| started == container_id));
        IsolationMock
            .enforce_stopped(
                container_id,
                spec,
                qualification_digest,
                work_unit_digest,
                placement_digest,
            )
            .await
    }
}

#[derive(Default)]
struct EventsMock(Mutex<Vec<NodeEvent>>);

#[async_trait]
impl EventSink for EventsMock {
    async fn emit(&self, event: NodeEvent) -> Result<(), ExecutionError> {
        self.0.lock().unwrap().push(event);
        Ok(())
    }
}

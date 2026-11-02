use super::*;

pub struct DaemonServices {
    pub store: Store,
    pub events: EventLog<jain_smartcluster_core::EventRecord>,
    pub authorizer: Authorizer,
    pub executor: Executor,
    pub resolver: WorkloadResolver,
    pub cgroups: CgroupManager,
    pub capacity: EffectiveCapacity,
    pub gpu: GpuInventory,
    pub mps_controller: Option<MpsController>,
    pub health: RuntimeHealth,
    pub execution: ExecutionConfig,
    pub recovery_quiet: Duration,
    pub interactive_headroom_per_mille: u16,
    pub interactive_gpu_headroom_per_mille: u16,
    pub max_running_jobs: usize,
    pub max_running_jobs_per_tenant: usize,
    pub max_queued_jobs: usize,
    pub max_queued_jobs_per_tenant: usize,
    pub tenant_weights: BTreeMap<jain_smartcluster_core::TenantId, u32>,
}

#[derive(Clone)]
pub struct DaemonHandle {
    sender: mpsc::Sender<ActorCommand>,
}

impl DaemonHandle {
    pub async fn start(services: DaemonServices) -> Result<Self, ActorStartError> {
        let (sender, receiver) = mpsc::channel(ACTOR_CAPACITY);
        let (startup, ready) = oneshot::channel();
        let journal = EventJournal::new(services.events, services.execution.subscriber_capacity);
        let schedule_pending = Arc::new(AtomicBool::new(false));
        let actor = DaemonActor {
            store: Arc::new(services.store),
            journal,
            authorizer: services.authorizer,
            executor: services.executor,
            resolver: services.resolver,
            cgroups: services.cgroups,
            capacity: services.capacity,
            gpu: services.gpu,
            mps_controller: services.mps_controller,
            gpu_quarantine: BTreeSet::new(),
            gpu_global_quarantine: false,
            health: services.health,
            execution: services.execution,
            overload: OverloadController::new(services.recovery_quiet),
            recovery_quiet: services.recovery_quiet,
            pressure_generations: [0; 3],
            pressure_levels: [PressureLevel::Normal; 3],
            scheduler: Scheduler::new(crate::scheduler::SchedulerConfig {
                interactive_headroom_per_mille: services.interactive_headroom_per_mille,
                interactive_gpu_headroom_per_mille: services.interactive_gpu_headroom_per_mille,
                // GPU inventory already removes the mandatory ten-percent
                // physical-VRAM safety headroom before it reaches the actor.
                shared_vram_headroom_per_mille: 0,
                max_admissions_per_pass: MAX_ADMISSIONS_PER_PASS,
                max_running_jobs: services.max_running_jobs,
                max_running_jobs_per_tenant: services.max_running_jobs_per_tenant,
                ..crate::scheduler::SchedulerConfig::default()
            })
            .expect("scheduler configuration is valid"),
            active: HashMap::new(),
            shutdown_replies: Vec::new(),
            shutdown_started: BTreeSet::new(),
            queued_jobs: 0,
            queued_by_tenant: BTreeMap::new(),
            max_queued_jobs: services.max_queued_jobs,
            max_queued_jobs_per_tenant: services.max_queued_jobs_per_tenant,
            tenant_weights: services.tenant_weights,
            streams: HashMap::new(),
            waiters: HashMap::new(),
            next_waiter_id: 0,
            receiver,
            sender: sender.clone(),
            schedule_pending,
        };
        tokio::spawn(actor.run(startup));
        ready
            .await
            .map_err(|_| ActorStartError("recovery task stopped before readiness".to_owned()))?
            .map_err(ActorStartError)?;
        Ok(Self { sender })
    }

    /// Close admission and wait for recursive workload drain before the process exits.
    pub async fn shutdown(&self) -> Result<(), ActorStartError> {
        let (reply, drained) = oneshot::channel();
        self.sender
            .send(ActorCommand::Shutdown(reply))
            .await
            .map_err(|_| ActorStartError("shutdown actor unavailable".to_owned()))?;
        drained
            .await
            .map_err(|_| ActorStartError("shutdown actor stopped before drain".to_owned()))
    }

    pub async fn pressure(&self, event: PsiEvent) {
        let _ = self.sender.send(ActorCommand::Pressure(event)).await;
    }

    pub async fn gpu_event(&self, event: GpuEvent) {
        let _ = self.sender.send(ActorCommand::Gpu(event)).await;
    }
}

#[async_trait]
impl RequestHandler for DaemonHandle {
    async fn request(
        &self,
        peer: &AuthenticatedPeer,
        request: Request,
        sink: ConnectionSink,
    ) -> Result<Response, ServerError> {
        let (reply, receiver) = oneshot::channel();
        self.sender
            .send(ActorCommand::Request {
                peer: peer.clone(),
                request,
                sink,
                reply,
            })
            .await
            .map_err(|_| unavailable())?;
        receiver.await.map_err(|_| unavailable())?
    }

    async fn stream_input(
        &self,
        peer: &AuthenticatedPeer,
        frame: StreamInputFrame,
    ) -> Result<(), ServerError> {
        let (reply, receiver) = oneshot::channel();
        self.sender
            .send(ActorCommand::StreamInput {
                peer: peer.clone(),
                frame,
                reply,
            })
            .await
            .map_err(|_| unavailable())?;
        receiver.await.map_err(|_| unavailable())?
    }

    async fn close_stream(&self, peer: &AuthenticatedPeer, stream_id: StreamId) {
        let _ = self
            .sender
            .send(ActorCommand::CloseStream {
                peer: peer.clone(),
                stream_id,
            })
            .await;
    }

    async fn stream_response_written(&self, peer: &AuthenticatedPeer, stream_id: StreamId) {
        let _ = self
            .sender
            .send(ActorCommand::ActivateStream {
                peer: peer.clone(),
                stream_id,
            })
            .await;
    }
}

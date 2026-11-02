use super::*;

impl Client {
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        let config = ClientConfig::new(socket_path);
        Self {
            codec: FrameCodec::default(),
            config: Arc::new(config),
        }
    }

    pub fn with_config(config: ClientConfig) -> Result<Self, DispatchError> {
        config.validate()?;
        let codec = FrameCodec::new(config.max_frame_bytes)?;
        Ok(Self {
            config: Arc::new(config),
            codec,
        })
    }

    pub fn config(&self) -> &ClientConfig {
        &self.config
    }

    pub fn socket_path(&self) -> &Path {
        &self.config.socket_path
    }

    pub async fn status(&self) -> Result<DaemonStatus, DispatchError> {
        match self
            .round_trip(
                Request::Status(jain_smartcluster_core::StatusRequest),
                Some(self.config.request_timeout),
                "status",
            )
            .await?
        {
            Response::Status(status) => Ok(status),
            response => Err(unexpected("status", &response)),
        }
    }

    pub async fn submit(&self, spec: JobSpec) -> Result<JobRecord, DispatchError> {
        self.submit_request(SubmitRequest {
            spec,
            effective_principal: None,
        })
        .await
    }

    pub async fn submit_as(
        &self,
        spec: JobSpec,
        effective_principal: Principal,
    ) -> Result<JobRecord, DispatchError> {
        self.submit_request(SubmitRequest {
            spec,
            effective_principal: Some(effective_principal),
        })
        .await
    }

    async fn submit_request(&self, request: SubmitRequest) -> Result<JobRecord, DispatchError> {
        match self
            .round_trip(
                Request::Submit(Box::new(request)),
                Some(self.config.request_timeout),
                "submit",
            )
            .await?
        {
            Response::Submitted(job) => Ok(job),
            response => Err(unexpected("submitted", &response)),
        }
    }

    pub async fn get(&self, job_id: JobId) -> Result<Option<JobRecord>, DispatchError> {
        match self
            .round_trip(
                Request::Get(GetRequest { job_id }),
                Some(self.config.request_timeout),
                "get",
            )
            .await?
        {
            Response::Job(job) => Ok(job),
            response => Err(unexpected("job", &response)),
        }
    }

    pub async fn list(&self, request: ListRequest) -> Result<JobPage, DispatchError> {
        match self
            .round_trip(
                Request::List(request),
                Some(self.config.request_timeout),
                "list",
            )
            .await?
        {
            Response::Jobs(page) => Ok(page),
            response => Err(unexpected("jobs", &response)),
        }
    }

    pub async fn wait(&self, job_id: JobId) -> Result<JobRecord, DispatchError> {
        match self
            .round_trip(
                Request::Wait(WaitRequest { job_id }),
                self.config.wait_timeout,
                "wait",
            )
            .await?
        {
            Response::Waited(job) => Ok(job),
            response => Err(unexpected("waited", &response)),
        }
    }

    pub async fn cancel(&self, request: CancelRequest) -> Result<JobRecord, DispatchError> {
        self.mutate(Request::Cancel(request), "cancel").await
    }

    pub async fn pause(&self, job_id: JobId) -> Result<JobRecord, DispatchError> {
        self.mutate(Request::Pause(PauseRequest { job_id }), "pause")
            .await
    }

    pub async fn resume(&self, job_id: JobId) -> Result<JobRecord, DispatchError> {
        self.mutate(Request::Resume(ResumeRequest { job_id }), "resume")
            .await
    }

    async fn mutate(
        &self,
        request: Request,
        operation: &'static str,
    ) -> Result<JobRecord, DispatchError> {
        match self
            .round_trip(request, Some(self.config.request_timeout), operation)
            .await?
        {
            Response::Mutated(job) => Ok(job),
            response => Err(unexpected("mutated", &response)),
        }
    }

    /// Only a qualified native fit authority leader may call this operation.
    pub async fn inspect_attempt(
        &self,
        attempt: jain_smartcluster_core::FencedAttempt,
    ) -> Result<Option<jain_smartcluster_core::AttemptInspection>, DispatchError> {
        match self
            .round_trip(
                Request::InspectAttempt(jain_smartcluster_core::InspectAttemptRequest { attempt }),
                Some(self.config.request_timeout),
                "inspect_attempt",
            )
            .await?
        {
            Response::AttemptInspected(value) => Ok(value),
            response => Err(unexpected("attempt_inspected", &response)),
        }
    }

    pub async fn attach_trusted_fit_child(
        &self,
        request: jain_smartcluster_core::AttachTrustedFitChildRequest,
    ) -> Result<jain_smartcluster_core::TrustedFitChildPlacement, DispatchError> {
        match self
            .round_trip(
                Request::AttachTrustedFitChild(Box::new(request)),
                Some(self.config.request_timeout),
                "attach_trusted_fit_child",
            )
            .await?
        {
            Response::TrustedFitChildPlaced(receipt) => Ok(receipt),
            response => Err(unexpected("trusted_fit_child_placed", &response)),
        }
    }

    pub async fn send_control(
        &self,
        request: SendControlRequest,
    ) -> Result<OperationAck, DispatchError> {
        match self
            .round_trip(
                Request::SendControl(request),
                Some(self.config.request_timeout),
                "send_control",
            )
            .await?
        {
            Response::ControlAccepted(ack) => Ok(ack),
            response => Err(unexpected("control_accepted", &response)),
        }
    }

    pub async fn subscribe(&self, request: SubscribeRequest) -> Result<EventStream, DispatchError> {
        let (socket, opened) = self
            .open_stream(Request::Subscribe(request), StreamKind::Subscription)
            .await?;
        let (reader, writer) = socket.into_split();
        Ok(EventStream::spawn(
            opened.stream_id,
            reader,
            writer,
            self.codec,
            self.config.stream_buffer,
        ))
    }

    pub async fn attach(&self, request: AttachRequest) -> Result<Attachment, DispatchError> {
        let (socket, opened) = self
            .open_stream(Request::Attach(request), StreamKind::Attachment)
            .await?;
        let (reader, writer) = socket.into_split();
        let events = EventStream::spawn(
            opened.stream_id,
            reader,
            writer,
            self.codec,
            self.config.stream_buffer,
        );
        Ok(Attachment::new(events, self.codec))
    }

    async fn open_stream(
        &self,
        request: Request,
        kind: StreamKind,
    ) -> Result<(UnixStream, StreamOpened), DispatchError> {
        let mut socket = self.connect().await?;
        let response = self
            .exchange_on(
                &mut socket,
                request,
                Some(self.config.request_timeout),
                kind.operation(),
            )
            .await?;
        let opened = match (kind, response) {
            (StreamKind::Subscription, Response::Subscribed(opened))
            | (StreamKind::Attachment, Response::Attached(opened)) => opened,
            (_, response) => return Err(unexpected(kind.expected_response(), &response)),
        };
        Ok((socket, opened))
    }

    async fn round_trip(
        &self,
        request: Request,
        request_timeout: Option<Duration>,
        operation: &'static str,
    ) -> Result<Response, DispatchError> {
        let mut socket = self.connect().await?;
        self.exchange_on(&mut socket, request, request_timeout, operation)
            .await
    }

    async fn exchange_on(
        &self,
        socket: &mut UnixStream,
        request: Request,
        request_timeout: Option<Duration>,
        operation: &'static str,
    ) -> Result<Response, DispatchError> {
        request.validate()?;
        let envelope = RequestEnvelope::new(request);
        let request_id = envelope.request_id;
        let exchange = async {
            self.codec
                .write_frame(socket, &ClientFrame::Request(Box::new(envelope)))
                .await?;
            let frame = self
                .codec
                .read_frame::<ServerFrame, _>(socket)
                .await?
                .ok_or(DispatchError::ConnectionClosed)?;
            let ServerFrame::Response(response) = frame else {
                return Err(DispatchError::UnexpectedServerFrame(server_frame_name(
                    &frame,
                )));
            };
            if !ProtocolVersion::CURRENT.is_compatible_with(response.version) {
                return Err(DispatchError::ProtocolMismatch {
                    client: ProtocolVersion::CURRENT,
                    server: response.version,
                });
            }
            if response.request_id != request_id {
                return Err(DispatchError::WrongRequestId {
                    expected: request_id,
                    actual: response.request_id,
                });
            }
            response.result.map_err(DispatchError::from)
        };
        with_timeout(request_timeout, operation, exchange).await
    }

    async fn connect(&self) -> Result<UnixStream, DispatchError> {
        let path = self.config.socket_path.clone();
        match timeout(self.config.connect_timeout, UnixStream::connect(&path)).await {
            Ok(Ok(socket)) => Ok(socket),
            Ok(Err(source)) => Err(DispatchError::Connect { path, source }),
            Err(_) => Err(DispatchError::Timeout {
                operation: "connect",
            }),
        }
    }

    /// Submit an allowlisted Jain workload through the schema-versioned API.
    /// The lower-level [`JobSpec`] methods remain the compatibility adapter;
    /// production callers should use this method or [`WorkloadDispatcher`].
    pub async fn dispatch_submit(
        &self,
        identity: RequestIdentity,
        workload: JainWorkload,
        idempotency: IdempotencyKey,
    ) -> Result<JobHandle, DispatchError> {
        identity.validate()?;
        let spec = typed_job_spec(workload, idempotency)?;
        let job = if identity.is_delegated() {
            self.submit_as(spec, identity.effective).await?
        } else {
            self.submit(spec).await?
        };
        Ok(JobHandle::from(&job))
    }

    pub async fn dispatch_attach(
        &self,
        job_id: JobId,
        after: EventOffset,
    ) -> Result<EventStream, DispatchError> {
        Ok(self
            .attach(AttachRequest {
                job_id,
                offset: after,
                read_only: true,
            })
            .await?
            .into_parts()
            .0)
    }

    pub async fn dispatch_cancel(
        &self,
        job_id: JobId,
        mode: CancelMode,
    ) -> Result<CancelReceipt, DispatchError> {
        let job = self.cancel(CancelRequest { job_id, mode }).await?;
        Ok(CancelReceipt::from(&job))
    }

    pub async fn dispatch_status(&self, job_id: JobId) -> Result<JobStatus, DispatchError> {
        let job = self.get(job_id).await?.ok_or_else(|| {
            DispatchError::Server(ServerError::new(ErrorCode::NotFound, "job does not exist"))
        })?;
        Ok(JobStatus::from(&job))
    }
}

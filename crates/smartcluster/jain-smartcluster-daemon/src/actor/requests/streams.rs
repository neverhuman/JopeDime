use super::*;

impl DaemonActor {
    pub(in super::super) async fn handle_stream_input(
        &mut self,
        peer: &AuthenticatedPeer,
        frame: StreamInputFrame,
    ) -> Result<(), ServerError> {
        let registration = self
            .streams
            .get(&frame.stream_id)
            .ok_or_else(|| ServerError::new(ErrorCode::NotFound, "stream does not exist"))?;
        if registration.peer != peer.identity {
            return Err(ServerError::new(
                ErrorCode::Forbidden,
                "stream belongs to another peer",
            ));
        }
        let job_id = registration.job_id.ok_or_else(|| {
            ServerError::new(
                ErrorCode::Conflict,
                "subscription stream does not accept input",
            )
        })?;
        if registration.read_only {
            return Err(ServerError::new(
                ErrorCode::Forbidden,
                "attach stream is read-only",
            ));
        }
        let active = self.active.get_mut(&job_id).ok_or_else(|| {
            ServerError::new(ErrorCode::Conflict, "job is not attached to a live process")
        })?;
        let io = active.io.as_mut().ok_or_else(|| {
            ServerError::new(
                ErrorCode::Conflict,
                "adopted job has no recoverable terminal or pipe handles",
            )
        })?;
        match frame.input {
            AttachInput::Stdin(data) => match io.input() {
                crate::executor::WorkloadInput::Pipe(writer)
                | crate::executor::WorkloadInput::Pty(writer) => {
                    tokio::time::timeout(self.execution.actor_io_timeout(), async {
                        writer.write_all(&data).await?;
                        writer.flush().await
                    })
                    .await
                    .map_err(|_| io_timeout_error())?
                    .map_err(io_error)?;
                }
            },
            AttachInput::Resize { rows, columns } => {
                io.resize(rows, columns).map_err(executor_error)?;
            }
            AttachInput::Control(control) => {
                let control = bounded_control(control, &self.execution)?;
                let attempt = active.attempt;
                if let Err(error) =
                    send_control(active, control, self.execution.actor_io_timeout()).await
                {
                    self.begin_control_recovery(attempt, FailureClass::Infrastructure)
                        .map_err(actor_server_error)?;
                    return Err(executor_error(error));
                }
            }
            AttachInput::CloseStdin => match io.input() {
                crate::executor::WorkloadInput::Pipe(writer)
                | crate::executor::WorkloadInput::Pty(writer) => {
                    tokio::time::timeout(self.execution.actor_io_timeout(), writer.shutdown())
                        .await
                        .map_err(|_| io_timeout_error())?
                        .map_err(io_error)?;
                }
            },
        }
        Ok(())
    }

    pub(in super::super) fn open_stream(
        &mut self,
        peer: &AuthenticatedPeer,
        offset: EventOffset,
        filter: EventFilter,
        job_id: Option<JobId>,
        read_only: bool,
        sink: ConnectionSink,
    ) -> Result<Response, ServerError> {
        let stream_id = StreamId::new();
        let (snapshot_end, subscriber) = self.journal.snapshot_and_subscribe();
        let (cancel, cancelled) = watch::channel(false);
        let (activate, activated) = oneshot::channel();
        self.streams.insert(
            stream_id,
            StreamRegistration {
                peer: peer.identity,
                job_id,
                read_only,
                cancel,
                activate: Some(activate),
            },
        );
        spawn_event_stream(EventStream {
            stream_id,
            requested: offset.get(),
            snapshot_end,
            filter,
            peer: peer.clone(),
            authorizer: self.authorizer.clone(),
            store: Arc::clone(&self.store),
            journal: self.journal.clone(),
            subscriber,
            sink,
            cancelled,
            activated,
            actor: self.sender.clone(),
        });
        Ok(if job_id.is_some() {
            Response::Attached(StreamOpened {
                stream_id,
                next_offset: offset,
            })
        } else {
            Response::Subscribed(StreamOpened {
                stream_id,
                next_offset: offset,
            })
        })
    }
}

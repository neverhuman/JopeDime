use super::*;

impl DaemonActor {
    pub(in super::super) async fn handle_request(
        &mut self,
        peer: &AuthenticatedPeer,
        request: Request,
        sink: ConnectionSink,
    ) -> Result<Response, ServerError> {
        match request {
            Request::InspectAttempt(request) => self.inspect_attempt(peer, request.attempt),
            Request::AttachTrustedFitChild(request) => {
                self.attach_trusted_fit_child(peer, *request)
            }
            Request::Status(_) => Ok(Response::Status(self.health.status())),
            Request::Submit(request) => self.submit(peer, *request),
            Request::Get(request) => {
                let job = self.store.get_job(request.job_id).map_err(store_error)?;
                match job {
                    Some(job) => {
                        self.authorizer
                            .authorize_job_access(peer, &job.submitted_by)
                            .map_err(auth_error)?;
                        Ok(Response::Job(Some(job)))
                    }
                    None => Ok(Response::Job(None)),
                }
            }
            Request::List(request) => self.list(peer, request),
            Request::Cancel(request) => self.cancel(peer, request.job_id, request.mode).await,
            Request::Pause(request) => self.pause(peer, request.job_id),
            Request::Resume(request) => self.resume(peer, request.job_id),
            Request::SendControl(request) => {
                let job = self.authorized_job(peer, request.job_id)?;
                let active = self.active.get_mut(&job.id).ok_or_else(|| {
                    ServerError::new(ErrorCode::Conflict, "job has no live control channel")
                })?;
                if active.control.is_none() {
                    return Err(ServerError::new(
                        ErrorCode::Conflict,
                        "adopted job has no recoverable control channel",
                    ));
                }
                let control = bounded_control(request.control, &self.execution)?;
                let attempt = active.attempt;
                if let Err(error) =
                    send_control(active, control, self.execution.actor_io_timeout()).await
                {
                    self.begin_control_recovery(attempt, FailureClass::Infrastructure)
                        .map_err(actor_server_error)?;
                    return Err(executor_error(error));
                }
                Ok(Response::ControlAccepted(OperationAck {
                    accepted_at: now_timestamp(),
                }))
            }
            Request::Subscribe(request) => {
                self.open_stream(peer, request.offset, request.filter, None, true, sink)
            }
            Request::Attach(request) => {
                let job = self.authorized_job(peer, request.job_id)?;
                self.open_stream(
                    peer,
                    request.offset,
                    EventFilter {
                        job_id: Some(job.id),
                        ..EventFilter::default()
                    },
                    Some(job.id),
                    request.read_only,
                    sink,
                )
            }
            Request::Wait(_) => unreachable!("wait requests are deferred in the actor loop"),
        }
    }
}

use std::collections::BTreeMap;

use jain_smartcluster_core::{
    AttachInput, CancelMode, CheckpointPolicy, ClientFrame, CpuRequest, DaemonMode,
    EffectiveCapacity, Environment, EventEnvelope, EventFilter, EventOffset, EventRecord,
    IdempotencyKey, IoMode, JainWorkload, JobEvent, JobPage, ListFilter, MemoryRequest,
    PreemptionPolicy, PressureStatus, PriorityClass, ProjectId, Request, ResourceRequest,
    ResponseEnvelope, RetryPolicy, SchedulerCapabilities, SendControlRequest, ServerFrame,
    StreamGap, StreamId, StreamOpened, TenantId, Timestamp, Workload,
};
use tempfile::TempDir;
use tokio::net::UnixListener;

use super::*;

fn spec() -> JobSpec {
    JobSpec {
        tenant: TenantId::new("tenant-a").unwrap(),
        project: ProjectId::new("project-a").unwrap(),
        idempotency_key: IdempotencyKey::new("request-a").unwrap(),
        priority: PriorityClass::Batch,
        workload: Workload::Jain {
            template: JainWorkload::Training,
            arguments: Vec::new(),
            environment: Environment::default(),
        },
        resources: ResourceRequest {
            cpu: CpuRequest::fixed(1000),
            ram: MemoryRequest::fixed(1024 * 1024),
            scratch_bytes: 0,
            gpu: None,
        },
        retry: RetryPolicy::default(),
        preemption: PreemptionPolicy {
            preemptible: true,
            checkpoint: CheckpointPolicy::ControlFd { deadline_ms: 1000 },
        },
        io: IoMode::Pipes,
        deadline: None,
        labels: BTreeMap::new(),
    }
}

fn job() -> JobRecord {
    JobRecord::queued(
        JobId::new(),
        spec(),
        Principal::unix(1000),
        Timestamp::from_unix_millis(1),
    )
}

fn status() -> DaemonStatus {
    DaemonStatus {
        daemon_version: "8.0.0".into(),
        protocol_version: ProtocolVersion::CURRENT,
        mode: DaemonMode::Full,
        started_at: 1.into(),
        paused: false,
        capabilities: SchedulerCapabilities::default(),
        effective_capacity: EffectiveCapacity::default(),
        pressure: PressureStatus::default(),
        gpus: Vec::new(),
        queued_jobs: 1,
        running_jobs: 0,
    }
}

fn listener() -> (TempDir, PathBuf, UnixListener) {
    let directory = tempfile::tempdir().unwrap();
    let socket_path = directory.path().join("scqd.sock");
    let listener = UnixListener::bind(&socket_path).unwrap();
    (directory, socket_path, listener)
}

#[tokio::test]
async fn every_unary_operation_maps_to_its_typed_response() {
    let (_directory, socket_path, listener) = listener();
    let expected_job = job();
    let server_job = expected_job.clone();
    let server = tokio::spawn(async move {
        let codec = FrameCodec::default();
        for _ in 0..9 {
            let (mut socket, _) = listener.accept().await.unwrap();
            let ClientFrame::Request(envelope) = codec
                .read_frame::<ClientFrame, _>(&mut socket)
                .await
                .unwrap()
                .unwrap()
            else {
                panic!("expected request frame");
            };
            let response = match envelope.request {
                Request::Status(_) => Response::Status(status()),
                Request::Submit(request) => {
                    assert_eq!(request.effective_principal, None);
                    Response::Submitted(server_job.clone())
                }
                Request::Get(_) => Response::Job(Some(server_job.clone())),
                Request::List(_) => Response::Jobs(JobPage {
                    jobs: vec![server_job.clone()],
                    next_cursor: Some("next".into()),
                }),
                Request::Wait(_) => Response::Waited(server_job.clone()),
                Request::Cancel(_) | Request::Pause(_) | Request::Resume(_) => {
                    Response::Mutated(server_job.clone())
                }
                Request::SendControl(_) => Response::ControlAccepted(OperationAck {
                    accepted_at: 2.into(),
                }),
                Request::Subscribe(_)
                | Request::Attach(_)
                | Request::AttachTrustedFitChild(_)
                | Request::InspectAttempt(_) => panic!("unexpected operation"),
            };
            codec
                .write_frame(
                    &mut socket,
                    &ServerFrame::Response(Box::new(ResponseEnvelope::ok(
                        envelope.request_id,
                        response,
                    ))),
                )
                .await
                .unwrap();
        }
    });

    let client = Client::new(socket_path);
    assert_eq!(client.status().await.unwrap().daemon_version, "8.0.0");
    assert_eq!(client.submit(spec()).await.unwrap(), expected_job);
    assert_eq!(
        client.get(expected_job.id).await.unwrap(),
        Some(expected_job.clone())
    );
    assert_eq!(
        client
            .list(ListRequest {
                filter: ListFilter::default(),
                cursor: None,
                limit: 10,
            })
            .await
            .unwrap()
            .jobs,
        vec![expected_job.clone()]
    );
    assert_eq!(client.wait(expected_job.id).await.unwrap(), expected_job);
    assert_eq!(
        client
            .cancel(CancelRequest {
                job_id: expected_job.id,
                mode: CancelMode::Graceful { grace_ms: 1000 },
            })
            .await
            .unwrap(),
        expected_job
    );
    assert_eq!(client.pause(expected_job.id).await.unwrap(), expected_job);
    assert_eq!(client.resume(expected_job.id).await.unwrap(), expected_job);
    assert_eq!(
        client
            .send_control(SendControlRequest {
                job_id: expected_job.id,
                control: jain_smartcluster_core::ControlMessage::Advance,
            })
            .await
            .unwrap()
            .accepted_at,
        Timestamp::from_unix_millis(2)
    );
    server.await.unwrap();
}

#[tokio::test]
async fn same_major_different_minor_response_is_rejected() {
    let (_directory, socket_path, listener) = listener();
    let server = tokio::spawn(async move {
        let codec = FrameCodec::default();
        let (mut socket, _) = listener.accept().await.unwrap();
        let ClientFrame::Request(envelope) = codec
            .read_frame::<ClientFrame, _>(&mut socket)
            .await
            .unwrap()
            .unwrap()
        else {
            panic!("expected request frame");
        };
        let mut response = ResponseEnvelope::ok(envelope.request_id, Response::Status(status()));
        response.version.minor += 1;
        codec
            .write_frame(&mut socket, &ServerFrame::response(response))
            .await
            .unwrap();
    });

    let error = Client::new(socket_path).status().await.unwrap_err();
    assert!(matches!(
        error,
        DispatchError::ProtocolMismatch { client, server }
            if client == ProtocolVersion::CURRENT
                && server.major == ProtocolVersion::CURRENT.major
                && server.minor == ProtocolVersion::CURRENT.minor + 1
    ));
    server.await.unwrap();
}

#[tokio::test]
async fn subscription_and_attachment_stream_events_and_input() {
    let (_directory, socket_path, listener) = listener();
    let job_id = JobId::new();
    let server = tokio::spawn(async move {
        let codec = FrameCodec::default();
        let stream_id = StreamId::new();
        let (mut socket, _) = listener.accept().await.unwrap();
        let ClientFrame::Request(envelope) = codec
            .read_frame::<ClientFrame, _>(&mut socket)
            .await
            .unwrap()
            .unwrap()
        else {
            panic!("expected subscribe request");
        };
        assert!(matches!(envelope.request, Request::Subscribe(_)));
        codec
            .write_frame(
                &mut socket,
                &ServerFrame::Response(Box::new(ResponseEnvelope::ok(
                    envelope.request_id,
                    Response::Subscribed(StreamOpened {
                        stream_id,
                        next_offset: EventOffset::new(7),
                    }),
                ))),
            )
            .await
            .unwrap();
        codec
            .write_frame(
                &mut socket,
                &ServerFrame::StreamGap {
                    stream_id,
                    gap: StreamGap {
                        requested: EventOffset::new(5),
                        available_from: EventOffset::new(7),
                        latest: EventOffset::new(8),
                        reason: "event segments were compacted".to_owned(),
                    },
                },
            )
            .await
            .unwrap();
        codec
            .write_frame(
                &mut socket,
                &ServerFrame::Event(EventEnvelope {
                    stream_id,
                    event: EventRecord {
                        offset: 7.into(),
                        job_id,
                        attempt: None,
                        at: 1.into(),
                        event: JobEvent::Stdout(b"hello".to_vec()),
                    },
                }),
            )
            .await
            .unwrap();
        codec
            .write_frame(
                &mut socket,
                &ServerFrame::StreamClosed {
                    stream_id,
                    error: None,
                },
            )
            .await
            .unwrap();

        let attach_stream_id = StreamId::new();
        let (mut socket, _) = listener.accept().await.unwrap();
        let ClientFrame::Request(envelope) = codec
            .read_frame::<ClientFrame, _>(&mut socket)
            .await
            .unwrap()
            .unwrap()
        else {
            panic!("expected attach request");
        };
        assert!(matches!(envelope.request, Request::Attach(_)));
        codec
            .write_frame(
                &mut socket,
                &ServerFrame::Response(Box::new(ResponseEnvelope::ok(
                    envelope.request_id,
                    Response::Attached(StreamOpened {
                        stream_id: attach_stream_id,
                        next_offset: EventOffset::ZERO,
                    }),
                ))),
            )
            .await
            .unwrap();
        let input = codec
            .read_frame::<ClientFrame, _>(&mut socket)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            input,
            ClientFrame::StreamInput(jain_smartcluster_core::StreamInputFrame {
                stream_id: attach_stream_id,
                input: AttachInput::Resize {
                    rows: 40,
                    columns: 120,
                },
            })
        );
        codec
            .write_frame(
                &mut socket,
                &ServerFrame::Event(EventEnvelope {
                    stream_id: attach_stream_id,
                    event: EventRecord {
                        offset: 8.into(),
                        job_id,
                        attempt: None,
                        at: 2.into(),
                        event: JobEvent::Pty(b"prompt".to_vec()),
                    },
                }),
            )
            .await
            .unwrap();
        codec
            .write_frame(
                &mut socket,
                &ServerFrame::StreamClosed {
                    stream_id: attach_stream_id,
                    error: None,
                },
            )
            .await
            .unwrap();
    });

    let client = Client::new(socket_path);
    let mut subscription = client
        .subscribe(SubscribeRequest {
            offset: EventOffset::new(5),
            filter: EventFilter::default(),
        })
        .await
        .unwrap();
    let gap = subscription.recv().await.unwrap().unwrap_err();
    assert!(matches!(
        gap,
        DispatchError::StreamGap {
            requested,
            available_from,
            latest,
            ..
        } if requested == EventOffset::new(5)
            && available_from == EventOffset::new(7)
            && latest == EventOffset::new(8)
    ));
    let event = subscription.recv().await.unwrap().unwrap();
    assert_eq!(event.event, JobEvent::Stdout(b"hello".to_vec()));
    assert!(subscription.recv().await.is_none());

    let mut attachment = client
        .attach(AttachRequest {
            job_id,
            offset: EventOffset::ZERO,
            read_only: false,
        })
        .await
        .unwrap();
    attachment.sender.resize(40, 120).await.unwrap();
    let event = attachment.events.recv().await.unwrap().unwrap();
    assert_eq!(event.event, JobEvent::Pty(b"prompt".to_vec()));
    assert!(attachment.events.recv().await.is_none());
    server.await.unwrap();
}

#[test]
fn client_implements_object_safe_dispatch_interface() {
    fn accepts_dispatcher(_dispatcher: &dyn WorkloadDispatcher) {}
    let client = Client::default();
    accepts_dispatcher(&client);
}

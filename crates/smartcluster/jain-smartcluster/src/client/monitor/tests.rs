use std::path::PathBuf;

use jain_smartcluster_core::{
    ClientFrame, ErrorCode, EventEnvelope, JobEvent, Request, Response, ResponseEnvelope,
    ServerError, ServerFrame, StreamGap, StreamId, StreamOpened,
};
use tokio::{
    io::AsyncWriteExt,
    net::{UnixListener, UnixStream},
    time::timeout,
};

use super::*;
use crate::FrameCodec;

fn listener() -> (tempfile::TempDir, PathBuf, UnixListener) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("monitor.sock");
    let listener = UnixListener::bind(&path).unwrap();
    (dir, path, listener)
}

async fn open(listener: &UnixListener, job: JobId, offset: u64) -> (UnixStream, StreamId) {
    let (mut socket, _) = timeout(Duration::from_secs(3), listener.accept())
        .await
        .unwrap()
        .unwrap();
    let codec = FrameCodec::default();
    let ClientFrame::Request(request) = codec
        .read_frame::<ClientFrame, _>(&mut socket)
        .await
        .unwrap()
        .unwrap()
    else {
        panic!("expected subscription");
    };
    assert!(matches!(&request.request, Request::Subscribe(subscription)
        if subscription.offset.get() == offset && subscription.filter.job_id == Some(job)));
    let id = StreamId::new();
    codec
        .write_frame(
            &mut socket,
            &ServerFrame::response(ResponseEnvelope::ok(
                request.request_id,
                Response::Subscribed(StreamOpened {
                    stream_id: id,
                    next_offset: offset.into(),
                }),
            )),
        )
        .await
        .unwrap();
    (socket, id)
}

async fn event(socket: &mut UnixStream, stream_id: StreamId, job_id: JobId, offset: u64) {
    FrameCodec::default()
        .write_frame(
            socket,
            &ServerFrame::Event(EventEnvelope {
                stream_id,
                event: EventRecord {
                    offset: offset.into(),
                    job_id,
                    attempt: None,
                    at: 1.into(),
                    event: JobEvent::Stdout(vec![1]),
                },
            }),
        )
        .await
        .unwrap();
}

async fn close(socket: &mut UnixStream, stream_id: StreamId) {
    FrameCodec::default()
        .write_frame(
            socket,
            &ServerFrame::StreamClosed {
                stream_id,
                error: None,
            },
        )
        .await
        .unwrap();
}

async fn next(monitor: &mut JobMonitor) -> Result<Option<EventRecord>, DispatchError> {
    timeout(Duration::from_secs(3), monitor.next())
        .await
        .expect("bounded monitor operation")
}

#[tokio::test]
async fn transport_loss_and_refused_reconnect_recover_inside_next() {
    let (_dir, path, first_listener) = listener();
    let rebound = path.clone();
    let job = JobId::new();
    let server = tokio::spawn(async move {
        let (mut socket, id) = open(&first_listener, job, 7).await;
        event(&mut socket, id, job, 7).await;
        drop(first_listener);
        drop(socket);
        tokio::time::sleep(Duration::from_millis(250)).await;
        std::fs::remove_file(&rebound).unwrap();
        let second_listener = UnixListener::bind(rebound).unwrap();
        let (mut socket, id) = open(&second_listener, job, 8).await;
        event(&mut socket, id, job, 8).await;
        close(&mut socket, id).await;
    });
    let mut monitor = JobMonitor::new(Client::new(path), job, 7.into());
    assert_eq!(next(&mut monitor).await.unwrap().unwrap().offset.get(), 7);
    assert_eq!(monitor.offset().get(), 8);
    assert_eq!(next(&mut monitor).await.unwrap().unwrap().offset.get(), 8);
    assert_eq!(monitor.offset().get(), 9);
    assert!(next(&mut monitor).await.unwrap().is_none());
    assert!(next(&mut monitor).await.unwrap().is_none());
    server.await.unwrap();
}

#[tokio::test]
async fn compacted_history_resumes_at_first_retained_event_without_repeating_gap() {
    let (_dir, path, listener) = listener();
    let job = JobId::new();
    let server = tokio::spawn(async move {
        let (mut socket, id) = open(&listener, job, 2).await;
        FrameCodec::default()
            .write_frame(
                &mut socket,
                &ServerFrame::StreamGap {
                    stream_id: id,
                    gap: StreamGap {
                        requested: 2.into(),
                        available_from: 7.into(),
                        latest: 8.into(),
                        reason: "compacted".into(),
                    },
                },
            )
            .await
            .unwrap();
        drop(socket);
        let (mut socket, id) = open(&listener, job, 7).await;
        event(&mut socket, id, job, 7).await;
        event(&mut socket, id, job, 8).await;
        close(&mut socket, id).await;
    });
    let mut monitor = JobMonitor::new(Client::new(path), job, 2.into());
    assert_eq!(next(&mut monitor).await.unwrap().unwrap().offset.get(), 7);
    assert_eq!(next(&mut monitor).await.unwrap().unwrap().offset.get(), 8);
    assert!(next(&mut monitor).await.unwrap().is_none());
    server.await.unwrap();
}

#[tokio::test]
async fn persisted_next_cursor_and_last_delivered_adapter_do_not_replay() {
    let (_dir, path, listener) = listener();
    let job = JobId::new();
    let server = tokio::spawn(async move {
        for offset in [7, 8, 9] {
            let (mut socket, id) = open(&listener, job, offset).await;
            event(&mut socket, id, job, offset).await;
            close(&mut socket, id).await;
        }
    });
    let client = Client::new(path);
    let mut first = JobMonitor::new(client.clone(), job, 7.into());
    assert_eq!(next(&mut first).await.unwrap().unwrap().offset.get(), 7);
    let persisted = first.offset();
    drop(first);
    let mut resumed = JobMonitor::new(client.clone(), job, persisted);
    let last = next(&mut resumed).await.unwrap().unwrap().offset;
    assert_eq!(last.get(), 8);
    drop(resumed);
    let mut legacy = JobMonitor::after(client, job, last);
    assert_eq!(next(&mut legacy).await.unwrap().unwrap().offset.get(), 9);
    server.await.unwrap();
}

#[tokio::test]
async fn cancelling_next_preserves_the_backoff_deadline() {
    let dir = tempfile::tempdir().unwrap();
    let mut monitor = JobMonitor::new(
        Client::new(dir.path().join("absent.sock")),
        JobId::new(),
        0.into(),
    );
    assert!(timeout(Duration::from_millis(10), monitor.next())
        .await
        .is_err());
    let deadline = monitor.retry_at.expect("connect failure schedules retry");
    let delay = monitor.retry_delay;
    assert!(timeout(Duration::from_millis(10), monitor.next())
        .await
        .is_err());
    assert_eq!(monitor.retry_at, Some(deadline));
    assert_eq!(monitor.retry_delay, delay);
}

#[tokio::test]
async fn malformed_frames_and_authorization_refusals_are_not_retried() {
    let (_dir, path, listener) = listener();
    let job = JobId::new();
    let server = tokio::spawn(async move {
        let (mut socket, _) = open(&listener, job, 0).await;
        socket.write_all(&0_u32.to_be_bytes()).await.unwrap();
        drop(socket);
        let (mut socket, id) = open(&listener, job, 0).await;
        FrameCodec::default()
            .write_frame(
                &mut socket,
                &ServerFrame::StreamClosed {
                    stream_id: id,
                    error: Some(ServerError::new(ErrorCode::Forbidden, "not your job")),
                },
            )
            .await
            .unwrap();
    });
    let client = Client::new(path);
    let mut malformed = JobMonitor::new(client.clone(), job, 0.into());
    assert!(matches!(
        next(&mut malformed).await,
        Err(DispatchError::Frame(FrameError::EmptyFrame))
    ));
    assert!(malformed.retry_at.is_none());
    let mut denied = JobMonitor::new(client, job, 0.into());
    assert!(
        matches!(next(&mut denied).await, Err(DispatchError::Server(error)) if error.code == ErrorCode::Forbidden)
    );
    assert!(denied.retry_at.is_none());
    server.await.unwrap();
}

#[tokio::test]
async fn nonadvancing_gap_is_returned_instead_of_looping() {
    let (_dir, path, listener) = listener();
    let job = JobId::new();
    let server = tokio::spawn(async move {
        let (mut socket, id) = open(&listener, job, 7).await;
        FrameCodec::default()
            .write_frame(
                &mut socket,
                &ServerFrame::StreamGap {
                    stream_id: id,
                    gap: StreamGap {
                        requested: 7.into(),
                        available_from: 7.into(),
                        latest: 8.into(),
                        reason: "invalid gap".into(),
                    },
                },
            )
            .await
            .unwrap();
    });
    let mut monitor = JobMonitor::new(Client::new(path), job, 7.into());
    assert!(matches!(
        next(&mut monitor).await,
        Err(DispatchError::StreamGap { .. })
    ));
    server.await.unwrap();
}

#[tokio::test]
async fn exhausted_last_delivered_cursor_does_not_reconnect() {
    let dir = tempfile::tempdir().unwrap();
    let mut monitor = JobMonitor::after(
        Client::new(dir.path().join("absent.sock")),
        JobId::new(),
        u64::MAX.into(),
    );
    assert!(next(&mut monitor).await.unwrap().is_none());
}

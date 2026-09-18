use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use futures_core::Stream;
use jain_smartcluster_core::{
    AttachInput, ClientFrame, ControlMessage, EventRecord, ServerFrame, StreamId, StreamInputFrame,
    Validate,
};
use tokio::{
    net::unix::{OwnedReadHalf, OwnedWriteHalf},
    sync::{mpsc, Mutex},
    task::JoinHandle,
};

use crate::{DispatchError, FrameCodec};

/// Reconnectable event stream. Resume by opening a new subscription at the last offset.
#[derive(Debug)]
pub struct EventStream {
    stream_id: StreamId,
    receiver: mpsc::Receiver<Result<EventRecord, DispatchError>>,
    writer: Arc<Mutex<OwnedWriteHalf>>,
    reader_task: JoinHandle<()>,
}

impl EventStream {
    pub(crate) fn spawn(
        stream_id: StreamId,
        reader: OwnedReadHalf,
        writer: OwnedWriteHalf,
        codec: FrameCodec,
        buffer: usize,
    ) -> Self {
        let (sender, receiver) = mpsc::channel(buffer);
        let reader_task = tokio::spawn(read_events(reader, codec, stream_id, sender));
        Self {
            stream_id,
            receiver,
            writer: Arc::new(Mutex::new(writer)),
            reader_task,
        }
    }

    pub const fn stream_id(&self) -> StreamId {
        self.stream_id
    }

    pub async fn recv(&mut self) -> Option<Result<EventRecord, DispatchError>> {
        self.receiver.recv().await
    }

    pub(crate) fn attach_sender(&self, codec: FrameCodec) -> AttachSender {
        AttachSender {
            stream_id: self.stream_id,
            writer: Arc::clone(&self.writer),
            codec,
        }
    }
}

impl Stream for EventStream {
    type Item = Result<EventRecord, DispatchError>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(context)
    }
}

impl Drop for EventStream {
    fn drop(&mut self) {
        self.reader_task.abort();
    }
}

async fn read_events(
    mut reader: OwnedReadHalf,
    codec: FrameCodec,
    expected_stream: StreamId,
    sender: mpsc::Sender<Result<EventRecord, DispatchError>>,
) {
    loop {
        let frame = match codec.read_frame::<ServerFrame, _>(&mut reader).await {
            Ok(Some(frame)) => frame,
            Ok(None) => {
                let _ = sender.send(Err(DispatchError::ConnectionClosed)).await;
                break;
            }
            Err(error) => {
                let _ = sender.send(Err(error.into())).await;
                break;
            }
        };
        match frame {
            ServerFrame::Event(envelope) if envelope.stream_id == expected_stream => {
                if sender.send(Ok(envelope.event)).await.is_err() {
                    break;
                }
            }
            ServerFrame::StreamGap { stream_id, gap } if stream_id == expected_stream => {
                if sender
                    .send(Err(DispatchError::StreamGap {
                        requested: gap.requested,
                        available_from: gap.available_from,
                        latest: gap.latest,
                        reason: gap.reason,
                    }))
                    .await
                    .is_err()
                {
                    break;
                }
            }
            ServerFrame::StreamClosed { stream_id, error } if stream_id == expected_stream => {
                if let Some(error) = error {
                    let _ = sender.send(Err(error.into())).await;
                }
                break;
            }
            ServerFrame::Event(envelope) => {
                let _ = sender
                    .send(Err(DispatchError::WrongStream {
                        expected: expected_stream,
                        actual: envelope.stream_id,
                    }))
                    .await;
                break;
            }
            ServerFrame::StreamClosed { stream_id, .. } => {
                let _ = sender
                    .send(Err(DispatchError::WrongStream {
                        expected: expected_stream,
                        actual: stream_id,
                    }))
                    .await;
                break;
            }
            ServerFrame::StreamGap { stream_id, .. } => {
                let _ = sender
                    .send(Err(DispatchError::WrongStream {
                        expected: expected_stream,
                        actual: stream_id,
                    }))
                    .await;
                break;
            }
            ServerFrame::Response(_) => {
                let _ = sender
                    .send(Err(DispatchError::UnexpectedServerFrame("response")))
                    .await;
                break;
            }
        }
    }
}

/// Concurrent input half of an interactive attachment.
#[derive(Clone, Debug)]
pub struct AttachSender {
    stream_id: StreamId,
    writer: Arc<Mutex<OwnedWriteHalf>>,
    codec: FrameCodec,
}

impl AttachSender {
    pub async fn send(&self, input: AttachInput) -> Result<(), DispatchError> {
        input.validate()?;
        let frame = ClientFrame::StreamInput(StreamInputFrame {
            stream_id: self.stream_id,
            input,
        });
        let mut writer = self.writer.lock().await;
        self.codec.write_frame(&mut *writer, &frame).await?;
        Ok(())
    }

    pub async fn stdin(&self, data: impl Into<Vec<u8>>) -> Result<(), DispatchError> {
        self.send(AttachInput::Stdin(data.into())).await
    }

    pub async fn resize(&self, rows: u16, columns: u16) -> Result<(), DispatchError> {
        self.send(AttachInput::Resize { rows, columns }).await
    }

    pub async fn control(&self, control: ControlMessage) -> Result<(), DispatchError> {
        self.send(AttachInput::Control(control)).await
    }

    pub async fn close_stdin(&self) -> Result<(), DispatchError> {
        self.send(AttachInput::CloseStdin).await
    }

    pub async fn close_stream(&self) -> Result<(), DispatchError> {
        let frame = ClientFrame::CloseStream {
            stream_id: self.stream_id,
        };
        let mut writer = self.writer.lock().await;
        self.codec.write_frame(&mut *writer, &frame).await?;
        Ok(())
    }
}

/// Bidirectional PTY/pipes attachment.
#[derive(Debug)]
pub struct Attachment {
    pub events: EventStream,
    pub sender: AttachSender,
}

impl Attachment {
    pub(crate) fn new(events: EventStream, codec: FrameCodec) -> Self {
        let sender = events.attach_sender(codec);
        Self { events, sender }
    }

    pub fn into_parts(self) -> (EventStream, AttachSender) {
        (self.events, self.sender)
    }
}

use std::io;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::net::UnixStream as StdUnixStream;
use std::time::Duration;

use nix::fcntl::{fcntl, FcntlArg, FdFlag};
use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::UnixStream;

pub const CONTROL_FD_ENV: &str = "SCQ_CONTROL_FD";

pub struct ControlPair {
    daemon: StdUnixStream,
    worker: StdUnixStream,
}

impl ControlPair {
    pub fn create() -> Result<Self, ControlError> {
        let (daemon, worker) = StdUnixStream::pair()?;
        let fd = worker.as_raw_fd();
        let flags = FdFlag::from_bits_truncate(fcntl(fd, FcntlArg::F_GETFD)?);
        fcntl(fd, FcntlArg::F_SETFD(flags & !FdFlag::FD_CLOEXEC))?;
        Ok(Self { daemon, worker })
    }

    #[cfg(test)]
    pub(crate) fn clone_worker_for_test(&self) -> std::io::Result<StdUnixStream> {
        self.worker.try_clone()
    }

    pub fn worker_fd(&self) -> RawFd {
        self.worker.as_raw_fd()
    }

    pub fn close_worker_and_open_daemon(
        self,
        max_frame: usize,
    ) -> Result<ControlChannel, ControlError> {
        drop(self.worker);
        self.daemon.set_nonblocking(true)?;
        Ok(ControlChannel {
            stream: UnixStream::from_std(self.daemon)?,
            max_frame,
        })
    }
}

pub struct ControlChannel {
    stream: UnixStream,
    max_frame: usize,
}

impl ControlChannel {
    pub async fn send<T: Serialize>(&mut self, frame: &T) -> Result<(), ControlError> {
        let payload = postcard::to_stdvec(frame).map_err(ControlError::Encode)?;
        if payload.len() > self.max_frame {
            return Err(ControlError::FrameTooLarge(payload.len()));
        }
        let length =
            u32::try_from(payload.len()).map_err(|_| ControlError::FrameTooLarge(payload.len()))?;
        self.stream.write_all(&length.to_be_bytes()).await?;
        self.stream.write_all(&payload).await?;
        Ok(())
    }

    pub async fn receive<T: DeserializeOwned>(&mut self) -> Result<T, ControlError> {
        let mut length = [0_u8; 4];
        self.stream.read_exact(&mut length).await?;
        let length = u32::from_be_bytes(length) as usize;
        if length > self.max_frame {
            return Err(ControlError::FrameTooLarge(length));
        }
        let mut payload = vec![0; length];
        self.stream.read_exact(&mut payload).await?;
        let (frame, remaining): (T, &[u8]) =
            postcard::take_from_bytes(&payload).map_err(ControlError::Decode)?;
        if !remaining.is_empty() {
            return Err(ControlError::TrailingBytes);
        }
        Ok(frame)
    }

    pub async fn receive_until<T: DeserializeOwned>(
        &mut self,
        timeout: Duration,
    ) -> Result<T, ControlError> {
        tokio::time::timeout(timeout, self.receive())
            .await
            .map_err(|_| ControlError::Timeout)?
    }

    pub fn split(self) -> (ControlSender, ControlReceiver) {
        let (reader, writer) = self.stream.into_split();
        (
            ControlSender {
                writer,
                max_frame: self.max_frame,
            },
            ControlReceiver {
                reader,
                max_frame: self.max_frame,
            },
        )
    }
}

pub struct ControlSender {
    writer: OwnedWriteHalf,
    max_frame: usize,
}

impl ControlSender {
    pub async fn send<T: Serialize>(&mut self, frame: &T) -> Result<(), ControlError> {
        let payload = postcard::to_stdvec(frame).map_err(ControlError::Encode)?;
        if payload.len() > self.max_frame {
            return Err(ControlError::FrameTooLarge(payload.len()));
        }
        let length =
            u32::try_from(payload.len()).map_err(|_| ControlError::FrameTooLarge(payload.len()))?;
        self.writer.write_all(&length.to_be_bytes()).await?;
        self.writer.write_all(&payload).await?;
        Ok(())
    }
}

pub struct ControlReceiver {
    reader: OwnedReadHalf,
    max_frame: usize,
}

impl ControlReceiver {
    pub async fn receive<T: DeserializeOwned>(&mut self) -> Result<T, ControlError> {
        let mut length = [0_u8; 4];
        self.reader.read_exact(&mut length).await?;
        let length = u32::from_be_bytes(length) as usize;
        if length > self.max_frame {
            return Err(ControlError::FrameTooLarge(length));
        }
        let mut payload = vec![0; length];
        self.reader.read_exact(&mut payload).await?;
        let (frame, remaining): (T, &[u8]) =
            postcard::take_from_bytes(&payload).map_err(ControlError::Decode)?;
        if !remaining.is_empty() {
            return Err(ControlError::TrailingBytes);
        }
        Ok(frame)
    }

    pub async fn receive_until<T: DeserializeOwned>(
        &mut self,
        timeout: Duration,
    ) -> Result<T, ControlError> {
        tokio::time::timeout(timeout, self.receive())
            .await
            .map_err(|_| ControlError::Timeout)?
    }
}

#[derive(Debug, Error)]
pub enum ControlError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("control frame encode failed: {0}")]
    Encode(postcard::Error),
    #[error("control frame decode failed: {0}")]
    Decode(postcard::Error),
    #[error("control frame is too large: {0} bytes")]
    FrameTooLarge(usize),
    #[error("control frame contains trailing bytes")]
    TrailingBytes,
    #[error("worker did not answer before the control deadline")]
    Timeout,
    #[error("control descriptor setup failed: {0}")]
    Fcntl(#[from] nix::Error),
}

#[cfg(test)]
mod tests {
    use jain_smartcluster_core::{
        AttemptNumber, ControlMessage, FenceToken, FencedAttempt, JobId, ProtocolVersion,
        WorkerControlFrame,
    };

    use super::*;

    #[test]
    fn worker_descriptor_survives_exec() {
        let pair = ControlPair::create().expect("pair");
        let flags = FdFlag::from_bits_truncate(
            fcntl(pair.worker_fd(), FcntlArg::F_GETFD).expect("get flags"),
        );
        assert!(!flags.contains(FdFlag::FD_CLOEXEC));
    }

    #[tokio::test]
    async fn frames_are_bounded_and_versioned() {
        let pair = ControlPair::create().expect("pair");
        let fd = pair.worker_fd();
        let worker = pair.worker.try_clone().expect("clone worker");
        let mut daemon = pair
            .close_worker_and_open_daemon(1024)
            .expect("daemon channel");
        worker.set_nonblocking(true).expect("nonblocking");
        let mut worker = ControlChannel {
            stream: UnixStream::from_std(worker).expect("worker channel"),
            max_frame: 1024,
        };
        assert!(fd >= 0);

        let frame = WorkerControlFrame {
            version: ProtocolVersion::CURRENT,
            attempt: FencedAttempt::new(
                JobId::new(),
                AttemptNumber::new(1).expect("attempt"),
                FenceToken::new(1).expect("fence"),
            ),
            sequence: 1,
            control: ControlMessage::Advance,
        };
        daemon.send(&frame).await.expect("send");
        let received: WorkerControlFrame = worker.receive().await.expect("receive");
        assert_eq!(received, frame);
    }
}

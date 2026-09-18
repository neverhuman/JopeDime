use std::io;

use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const DEFAULT_MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;
pub const ABSOLUTE_MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum FrameError {
    #[error("frame I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("frame length cannot be zero")]
    EmptyFrame,
    #[error("frame length {length} exceeds limit {limit}")]
    FrameTooLarge { length: usize, limit: usize },
    #[error("frame serialization failed: {0}")]
    Serialization(#[from] postcard::Error),
    #[error("frame contains {remaining} trailing bytes")]
    TrailingBytes { remaining: usize },
    #[error("invalid frame limit {0}")]
    InvalidLimit(usize),
}

/// Four-byte big-endian length prefix followed by a bounded Postcard payload.
#[derive(Clone, Copy, Debug)]
pub struct FrameCodec {
    max_frame_bytes: usize,
}

impl FrameCodec {
    pub fn new(max_frame_bytes: usize) -> Result<Self, FrameError> {
        if max_frame_bytes == 0 || max_frame_bytes > ABSOLUTE_MAX_FRAME_BYTES {
            return Err(FrameError::InvalidLimit(max_frame_bytes));
        }
        Ok(Self { max_frame_bytes })
    }

    pub const fn max_frame_bytes(self) -> usize {
        self.max_frame_bytes
    }

    pub fn encode<T: Serialize>(&self, value: &T) -> Result<Vec<u8>, FrameError> {
        let bytes = postcard::to_stdvec(value)?;
        self.check_length(bytes.len())?;
        Ok(bytes)
    }

    pub fn decode<T: DeserializeOwned>(&self, bytes: &[u8]) -> Result<T, FrameError> {
        self.check_length(bytes.len())?;
        if bytes.is_empty() {
            return Err(FrameError::EmptyFrame);
        }
        let (value, remaining) = postcard::take_from_bytes(bytes)?;
        if !remaining.is_empty() {
            return Err(FrameError::TrailingBytes {
                remaining: remaining.len(),
            });
        }
        Ok(value)
    }

    pub async fn write_frame<T, W>(&self, writer: &mut W, value: &T) -> Result<(), FrameError>
    where
        T: Serialize,
        W: AsyncWrite + Unpin,
    {
        let bytes = self.encode(value)?;
        let length = u32::try_from(bytes.len()).map_err(|_| FrameError::FrameTooLarge {
            length: bytes.len(),
            limit: self.max_frame_bytes,
        })?;
        writer.write_all(&length.to_be_bytes()).await?;
        writer.write_all(&bytes).await?;
        writer.flush().await?;
        Ok(())
    }

    /// Returns `Ok(None)` only for a clean EOF before a new frame starts.
    pub async fn read_frame<T, R>(&self, reader: &mut R) -> Result<Option<T>, FrameError>
    where
        T: DeserializeOwned,
        R: AsyncRead + Unpin,
    {
        let mut length_bytes = [0_u8; 4];
        let first = reader.read(&mut length_bytes[..1]).await?;
        if first == 0 {
            return Ok(None);
        }
        reader.read_exact(&mut length_bytes[1..]).await?;
        let encoded_length = u32::from_be_bytes(length_bytes);
        let length = usize::try_from(encoded_length).map_err(|_| FrameError::FrameTooLarge {
            length: usize::MAX,
            limit: self.max_frame_bytes,
        })?;
        self.check_length(length)?;
        if length == 0 {
            return Err(FrameError::EmptyFrame);
        }
        let mut bytes = vec![0_u8; length];
        reader.read_exact(&mut bytes).await?;
        self.decode(&bytes).map(Some)
    }

    fn check_length(&self, length: usize) -> Result<(), FrameError> {
        if length > self.max_frame_bytes {
            Err(FrameError::FrameTooLarge {
                length,
                limit: self.max_frame_bytes,
            })
        } else {
            Ok(())
        }
    }
}

impl Default for FrameCodec {
    fn default() -> Self {
        Self {
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
        }
    }
}

#[cfg(test)]
mod tests {
    use jain_smartcluster_core::{ClientFrame, GetRequest, JobId, Request, RequestEnvelope};
    use tokio::io::{duplex, AsyncWriteExt};

    use super::*;

    #[tokio::test]
    async fn frames_round_trip_over_fragmentable_io() {
        let codec = FrameCodec::new(4096).unwrap();
        let frame =
            ClientFrame::Request(Box::new(RequestEnvelope::new(Request::Get(GetRequest {
                job_id: JobId::new(),
            }))));
        let (mut left, mut right) = duplex(64);
        let sent = frame.clone();
        let writer = tokio::spawn(async move { codec.write_frame(&mut left, &sent).await });
        let decoded: ClientFrame = codec.read_frame(&mut right).await.unwrap().unwrap();
        writer.await.unwrap().unwrap();
        assert_eq!(decoded, frame);
    }

    #[tokio::test]
    async fn oversized_prefix_is_rejected_before_payload_allocation() {
        let codec = FrameCodec::new(16).unwrap();
        let (mut left, mut right) = duplex(16);
        left.write_all(&17_u32.to_be_bytes()).await.unwrap();
        let error = codec
            .read_frame::<ClientFrame, _>(&mut right)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            FrameError::FrameTooLarge {
                length: 17,
                limit: 16
            }
        ));
    }

    #[tokio::test]
    async fn clean_eof_is_distinct_from_truncated_frame() {
        let codec = FrameCodec::default();
        let (left, mut right) = duplex(16);
        drop(left);
        assert!(codec
            .read_frame::<ClientFrame, _>(&mut right)
            .await
            .unwrap()
            .is_none());

        let (mut left, mut right) = duplex(16);
        left.write_all(&[0, 0]).await.unwrap();
        drop(left);
        assert!(matches!(
            codec.read_frame::<ClientFrame, _>(&mut right).await,
            Err(FrameError::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof
        ));
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let codec = FrameCodec::default();
        let frame =
            ClientFrame::Request(Box::new(RequestEnvelope::new(Request::Get(GetRequest {
                job_id: JobId::new(),
            }))));
        let mut encoded = codec.encode(&frame).unwrap();
        encoded.push(0);
        assert!(matches!(
            codec.decode::<ClientFrame>(&encoded),
            Err(FrameError::TrailingBytes { remaining: 1 })
        ));
    }
}

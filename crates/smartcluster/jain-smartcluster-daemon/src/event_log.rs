//! Durable, bounded event segments.
//!
//! An append is acknowledged only after its record reaches `sync_data`. Readers
//! are deliberately decoupled from writers: they reopen immutable/append-only
//! segments and resume from a monotonic offset. A missing retained range is
//! represented as a [`LogItem::Gap`] instead of being silently skipped.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use crc32fast::Hasher;
use postcard::{from_bytes, to_stdvec};
use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;

const MAGIC: &[u8; 8] = b"SCQEVT01";
const FORMAT_VERSION: u16 = 1;
const HEADER_LEN: u64 = 24;
const RECORD_PREFIX_LEN: u64 = 16;
const SEGMENT_EXTENSION: &str = "seg";

/// Bounds for one event stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventLogOptions {
    /// Maximum encoded size of one event.
    pub max_event_bytes: usize,
    /// Maximum on-disk bytes in one segment, including framing.
    pub max_segment_bytes: u64,
    /// Maximum retained segment count. Rotation deletes the oldest segment.
    pub max_segments: usize,
}

impl Default for EventLogOptions {
    fn default() -> Self {
        Self {
            max_event_bytes: 1024 * 1024,
            max_segment_bytes: 8 * 1024 * 1024,
            max_segments: 16,
        }
    }
}

impl EventLogOptions {
    fn validate(self) -> Result<Self, EventLogError> {
        if self.max_segments == 0 {
            return Err(EventLogError::InvalidOptions(
                "max_segments must be greater than zero",
            ));
        }
        let largest_record = u64::try_from(self.max_event_bytes)
            .unwrap_or(u64::MAX)
            .saturating_add(RECORD_PREFIX_LEN)
            .saturating_add(HEADER_LEN);
        if self.max_event_bytes == 0 || self.max_segment_bytes < largest_record {
            return Err(EventLogError::InvalidOptions(
                "max_segment_bytes must fit the header and one maximum-size event",
            ));
        }
        Ok(self)
    }
}

/// A typed event read from durable storage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoggedEvent<T> {
    pub offset: u64,
    pub event: T,
}

/// A range no longer available to a subscriber.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamGap {
    pub first_missing: u64,
    pub next_available: u64,
}

/// One resumable stream item.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LogItem<T> {
    Event(LoggedEvent<T>),
    Gap(StreamGap),
}

/// A bounded read result. Continue at `resume_offset`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadBatch<T> {
    pub items: Vec<LogItem<T>>,
    pub resume_offset: u64,
    pub end_offset: u64,
}

/// Recovery details useful for health evidence and tests.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RecoveryReport {
    pub truncated_segments: usize,
    pub truncated_bytes: u64,
}

/// A validated record frame returned by [`decode_record_bytes`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodedRecordBytes<'a> {
    pub offset: u64,
    pub payload: &'a [u8],
    pub consumed: usize,
}

/// Bounded, side-effect-free record parser used by recovery and fuzzing.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum RecordDecodeError {
    #[error("record frame is incomplete")]
    Incomplete,
    #[error("record payload length {actual} exceeds limit {maximum}")]
    TooLarge { actual: usize, maximum: usize },
    #[error("record checksum does not match")]
    Checksum,
}

/// Parse one CRC-protected record without allocating or touching the filesystem.
/// `consumed` permits a fuzz target to iterate over concatenated frames.
pub fn decode_record_bytes(
    bytes: &[u8],
    max_event_bytes: usize,
) -> Result<DecodedRecordBytes<'_>, RecordDecodeError> {
    if bytes.len() < RECORD_PREFIX_LEN as usize {
        return Err(RecordDecodeError::Incomplete);
    }
    let payload_len = u32::from_le_bytes(bytes[..4].try_into().expect("fixed slice")) as usize;
    if payload_len > max_event_bytes {
        return Err(RecordDecodeError::TooLarge {
            actual: payload_len,
            maximum: max_event_bytes,
        });
    }
    let consumed = (RECORD_PREFIX_LEN as usize)
        .checked_add(payload_len)
        .ok_or(RecordDecodeError::TooLarge {
            actual: payload_len,
            maximum: max_event_bytes,
        })?;
    if bytes.len() < consumed {
        return Err(RecordDecodeError::Incomplete);
    }
    let offset = u64::from_le_bytes(bytes[4..12].try_into().expect("fixed slice"));
    let expected_checksum = u32::from_le_bytes(bytes[12..16].try_into().expect("fixed slice"));
    let payload = &bytes[RECORD_PREFIX_LEN as usize..consumed];
    if record_checksum(offset, payload) != expected_checksum {
        return Err(RecordDecodeError::Checksum);
    }
    Ok(DecodedRecordBytes {
        offset,
        payload,
        consumed,
    })
}

#[derive(Debug, Error)]
pub enum EventLogError {
    #[error("invalid event-log options: {0}")]
    InvalidOptions(&'static str),
    #[error("event encodes to {actual} bytes, above the {maximum}-byte limit")]
    EventTooLarge { actual: usize, maximum: usize },
    #[error("event offset space is exhausted")]
    OffsetExhausted,
    #[error("segment {path} starts at {actual}, before prior end offset {minimum}")]
    NonMonotonicSegment {
        path: PathBuf,
        actual: u64,
        minimum: u64,
    },
    #[error("event at offset {offset} failed to decode: {source}")]
    Decode {
        offset: u64,
        #[source]
        source: postcard::Error,
    },
    #[error("event-log I/O failed for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("event encoding failed: {0}")]
    Encode(#[from] postcard::Error),
}

impl EventLogError {
    fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

#[derive(Clone, Debug)]
struct Segment {
    path: PathBuf,
    start: u64,
    end: u64,
    bytes: u64,
}

/// Single-writer durable log for one event stream.
///
/// The type parameter is encoded with Postcard.
/// Opening the same directory with a different `T` is a schema error that will
/// surface when records are read; callers should version their event type.
pub struct EventLog<T> {
    directory: PathBuf,
    options: EventLogOptions,
    segments: Vec<Segment>,
    current: File,
    next_offset: u64,
    recovery: RecoveryReport,
    marker: PhantomData<fn() -> T>,
}

impl<T> fmt::Debug for EventLog<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EventLog")
            .field("directory", &self.directory)
            .field("options", &self.options)
            .field("next_offset", &self.next_offset)
            .field("recovery", &self.recovery)
            .finish_non_exhaustive()
    }
}

impl<T> EventLog<T>
where
    T: Serialize + DeserializeOwned,
{
    /// Open or create a stream directory and recover every existing segment.
    pub fn open(
        directory: impl AsRef<Path>,
        options: EventLogOptions,
    ) -> Result<Self, EventLogError> {
        let options = options.validate()?;
        let directory = directory.as_ref().to_path_buf();
        fs::create_dir_all(&directory).map_err(|error| EventLogError::io(&directory, error))?;

        let mut paths = segment_paths(&directory)?;
        let mut segments = Vec::with_capacity(paths.len().max(1));
        let mut report = RecoveryReport::default();
        let mut prior_end = 0;

        for path in paths.drain(..) {
            let filename_start = segment_start_from_path(&path).expect("filtered segment path");
            if filename_start < prior_end {
                return Err(EventLogError::NonMonotonicSegment {
                    path,
                    actual: filename_start,
                    minimum: prior_end,
                });
            }
            match recover_segment(&path, filename_start, options.max_event_bytes)? {
                RecoveredSegment::Usable {
                    end,
                    bytes,
                    truncated_bytes,
                } => {
                    if truncated_bytes > 0 {
                        report.truncated_segments += 1;
                        report.truncated_bytes =
                            report.truncated_bytes.saturating_add(truncated_bytes);
                    }
                    segments.push(Segment {
                        path,
                        start: filename_start,
                        end,
                        bytes,
                    });
                    prior_end = end;
                }
                RecoveredSegment::InvalidHeader { removed_bytes } => {
                    fs::remove_file(&path).map_err(|error| EventLogError::io(&path, error))?;
                    sync_directory(&directory)?;
                    report.truncated_segments += 1;
                    report.truncated_bytes = report.truncated_bytes.saturating_add(removed_bytes);
                    prior_end = prior_end.max(filename_start);
                }
            }
        }

        if segments.is_empty() {
            let path = create_segment(&directory, prior_end)?;
            segments.push(Segment {
                path,
                start: prior_end,
                end: prior_end,
                bytes: HEADER_LEN,
            });
        }

        while segments.len() > options.max_segments {
            remove_oldest_segment(&directory, &mut segments)?;
        }

        let next_offset = segments.last().map_or(0, |segment| segment.end);
        let current_path = segments
            .last()
            .expect("event log always has a segment")
            .path
            .clone();
        let current = OpenOptions::new()
            .read(true)
            .append(true)
            .open(&current_path)
            .map_err(|error| EventLogError::io(&current_path, error))?;

        Ok(Self {
            directory,
            options,
            segments,
            current,
            next_offset,
            recovery: report,
            marker: PhantomData,
        })
    }

    pub fn next_offset(&self) -> u64 {
        self.next_offset
    }

    pub fn first_available_offset(&self) -> u64 {
        self.segments
            .first()
            .map_or(self.next_offset, |segment| segment.start)
    }

    pub fn recovery_report(&self) -> RecoveryReport {
        self.recovery
    }

    /// Append and durably acknowledge one event.
    pub fn append(&mut self, event: &T) -> Result<u64, EventLogError> {
        if self.next_offset == u64::MAX {
            return Err(EventLogError::OffsetExhausted);
        }
        let payload = to_stdvec(event)?;
        if payload.len() > self.options.max_event_bytes {
            return Err(EventLogError::EventTooLarge {
                actual: payload.len(),
                maximum: self.options.max_event_bytes,
            });
        }
        let payload_len =
            u32::try_from(payload.len()).map_err(|_| EventLogError::EventTooLarge {
                actual: payload.len(),
                maximum: self.options.max_event_bytes,
            })?;
        let framed_len = RECORD_PREFIX_LEN + u64::from(payload_len);
        if self
            .segments
            .last()
            .is_some_and(|segment| segment.bytes > HEADER_LEN)
            && self.segments.last().is_some_and(|segment| {
                segment.bytes.saturating_add(framed_len) > self.options.max_segment_bytes
            })
        {
            self.rotate()?;
        }

        let offset = self.next_offset;
        let checksum = record_checksum(offset, &payload);
        let mut prefix = [0_u8; RECORD_PREFIX_LEN as usize];
        prefix[..4].copy_from_slice(&payload_len.to_le_bytes());
        prefix[4..12].copy_from_slice(&offset.to_le_bytes());
        prefix[12..16].copy_from_slice(&checksum.to_le_bytes());

        let current_path = self
            .segments
            .last()
            .expect("event log always has a segment")
            .path
            .clone();
        self.current
            .write_all(&prefix)
            .and_then(|()| self.current.write_all(&payload))
            .and_then(|()| self.current.sync_data())
            .map_err(|error| EventLogError::io(current_path, error))?;

        self.next_offset = self.next_offset.saturating_add(1);
        let segment = self
            .segments
            .last_mut()
            .expect("event log always has a segment");
        segment.end = self.next_offset;
        segment.bytes = segment.bytes.saturating_add(framed_len);
        Ok(offset)
    }

    /// Read at most `limit` events or gap markers beginning at `offset`.
    pub fn read_from(&self, offset: u64, limit: usize) -> Result<ReadBatch<T>, EventLogError> {
        let mut cursor = offset.min(self.next_offset);
        let mut items = Vec::with_capacity(limit.min(256));
        if limit == 0 {
            return Ok(ReadBatch {
                items,
                resume_offset: cursor,
                end_offset: self.next_offset,
            });
        }

        for segment in &self.segments {
            if items.len() == limit || segment.end <= cursor {
                continue;
            }
            if cursor < segment.start {
                items.push(LogItem::Gap(StreamGap {
                    first_missing: cursor,
                    next_available: segment.start,
                }));
                cursor = segment.start;
                if items.len() == limit {
                    break;
                }
            }
            read_segment(segment, &mut cursor, limit, &mut items)?;
        }

        Ok(ReadBatch {
            items,
            resume_offset: cursor,
            end_offset: self.next_offset,
        })
    }

    fn rotate(&mut self) -> Result<(), EventLogError> {
        self.current.sync_all().map_err(|error| {
            EventLogError::io(&self.segments.last().expect("segment").path, error)
        })?;
        let path = create_segment(&self.directory, self.next_offset)?;
        self.current = OpenOptions::new()
            .read(true)
            .append(true)
            .open(&path)
            .map_err(|error| EventLogError::io(&path, error))?;
        self.segments.push(Segment {
            path,
            start: self.next_offset,
            end: self.next_offset,
            bytes: HEADER_LEN,
        });
        while self.segments.len() > self.options.max_segments {
            remove_oldest_segment(&self.directory, &mut self.segments)?;
        }
        Ok(())
    }
}

mod storage;
use storage::*;

#[cfg(test)]
mod tests;

use std::{io, time::Duration};

use jain_smartcluster_core::{EventOffset, EventRecord, JobId};
use tokio::time::{sleep_until, Instant};

use super::{job_subscription, Client, DispatchError};
use crate::{EventStream, FrameError};

const INITIAL_RETRY: Duration = Duration::from_millis(100);
const MAX_RETRY: Duration = Duration::from_secs(5);

/// Durable job reader. Transient transport failures reconnect within `next()`;
/// protocol/authorization errors remain visible and graceful stream closure ends it.
pub struct JobMonitor {
    client: Client,
    job_id: JobId,
    next_offset: EventOffset,
    stream: Option<EventStream>,
    retry_at: Option<Instant>,
    retry_delay: Duration,
    finished: bool,
}

impl JobMonitor {
    /// Start at the first unseen offset (inclusive), matching the subscription wire contract.
    pub const fn new(client: Client, job_id: JobId, next_offset: EventOffset) -> Self {
        Self {
            client,
            job_id,
            next_offset,
            stream: None,
            retry_at: None,
            retry_delay: INITIAL_RETRY,
            finished: false,
        }
    }

    /// Resume a caller that persists the last delivered record rather than the next cursor.
    /// For a new job with no delivered record, use `new(..., EventOffset::ZERO)`.
    pub fn after(client: Client, job_id: JobId, last_delivered: EventOffset) -> Self {
        let mut monitor = Self::new(client, job_id, last_delivered);
        match last_delivered.next() {
            Some(next) => monitor.next_offset = next,
            None => monitor.finished = true,
        }
        monitor
    }

    /// First unseen offset, suitable for passing back to `new()` after restart.
    pub const fn offset(&self) -> EventOffset {
        self.next_offset
    }

    pub async fn next(&mut self) -> Result<Option<EventRecord>, DispatchError> {
        loop {
            if self.finished {
                return Ok(None);
            }
            // Keep the deadline in self: cancelling this future must not erase backoff.
            if let Some(deadline) = self.retry_at {
                sleep_until(deadline).await;
                self.retry_at = None;
            }
            if self.stream.is_none() {
                match self
                    .client
                    .subscribe(job_subscription(self.job_id, self.next_offset))
                    .await
                {
                    Ok(stream) => self.stream = Some(stream),
                    Err(error) if retryable(&error) => {
                        self.schedule_retry();
                        continue;
                    }
                    Err(error) => return Err(error),
                }
            }
            match self.stream.as_mut().expect("stream opened").recv().await {
                Some(Ok(record)) => {
                    if record.job_id != self.job_id || record.offset < self.next_offset {
                        self.stream = None;
                        return Err(DispatchError::UnexpectedServerFrame(
                            "invalid job event cursor",
                        ));
                    }
                    match record.offset.next() {
                        Some(next) => self.next_offset = next,
                        None => self.finished = true,
                    }
                    self.retry_delay = INITIAL_RETRY;
                    return Ok(Some(record));
                }
                Some(Err(error)) => {
                    self.stream = None;
                    if let DispatchError::StreamGap { available_from, .. } = &error {
                        // The wire offset is inclusive. Subtracting one replays the same gap.
                        if *available_from <= self.next_offset {
                            return Err(error);
                        }
                        self.next_offset = *available_from;
                        self.schedule_retry();
                    } else if retryable(&error) {
                        self.schedule_retry();
                    } else {
                        return Err(error);
                    }
                }
                None => {
                    // Raw EOF is ConnectionClosed above; None is an explicit clean close.
                    self.stream = None;
                    self.finished = true;
                    return Ok(None);
                }
            }
        }
    }

    fn schedule_retry(&mut self) {
        self.retry_at = Some(Instant::now() + self.retry_delay);
        self.retry_delay = (self.retry_delay * 2).min(MAX_RETRY);
    }
}

fn retryable(error: &DispatchError) -> bool {
    match error {
        DispatchError::ConnectionClosed | DispatchError::Timeout { .. } => true,
        DispatchError::Connect { source, .. } | DispatchError::Frame(FrameError::Io(source)) => {
            matches!(
                source.kind(),
                io::ErrorKind::NotFound
                    | io::ErrorKind::ConnectionRefused
                    | io::ErrorKind::ConnectionReset
                    | io::ErrorKind::ConnectionAborted
                    | io::ErrorKind::BrokenPipe
                    | io::ErrorKind::TimedOut
                    | io::ErrorKind::Interrupted
                    | io::ErrorKind::UnexpectedEof
            )
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests;

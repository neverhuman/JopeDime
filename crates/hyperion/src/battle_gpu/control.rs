//! Bounded cancellation/deadline checkpoints for queue admission and GPU chunks.

use anyhow::Result;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Instant;

/// Shared cancellation handle. Cancellation is monotonic and lock-free.
#[derive(Clone, Debug, Default)]
pub struct GpuCancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl GpuCancellationToken {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuInterruption {
    Cancelled,
    Deadline,
}

impl std::fmt::Display for GpuInterruption {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("GPU operation cancelled"),
            Self::Deadline => formatter.write_str("GPU operation deadline expired"),
        }
    }
}

impl std::error::Error for GpuInterruption {}

/// Operation-local bounds used at queue admission and between GPU chunks.
#[derive(Clone, Debug, Default)]
pub struct GpuOperationControl {
    pub deadline: Option<Instant>,
    pub cancellation: GpuCancellationToken,
}

impl GpuOperationControl {
    pub fn checkpoint(&self) -> Result<()> {
        if self.cancellation.is_cancelled() {
            return Err(GpuInterruption::Cancelled.into());
        }
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(GpuInterruption::Deadline.into());
        }
        Ok(())
    }
}

/// Dependency-inversion seam for Core/Math operation controls.
///
/// A closure such as `|| math_control.checkpoint()` implements this trait, so
/// Battle does not need to own or import Core's operation type.
pub trait GpuCheckpoint: Send + Sync {
    fn checkpoint(&self) -> Result<()>;
}

impl GpuCheckpoint for GpuOperationControl {
    fn checkpoint(&self) -> Result<()> {
        GpuOperationControl::checkpoint(self)
    }
}

impl<F> GpuCheckpoint for F
where
    F: Fn() -> Result<()> + Send + Sync,
{
    fn checkpoint(&self) -> Result<()> {
        self()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NoopCheckpoint;

impl GpuCheckpoint for NoopCheckpoint {
    fn checkpoint(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_and_deadline_are_typed() {
        let cancelled = GpuOperationControl::default();
        cancelled.cancellation.cancel();
        let error = cancelled.checkpoint().unwrap_err();
        assert_eq!(
            error.downcast_ref::<GpuInterruption>(),
            Some(&GpuInterruption::Cancelled)
        );

        let expired = GpuOperationControl {
            deadline: Some(Instant::now()),
            cancellation: GpuCancellationToken::default(),
        };
        let error = expired.checkpoint().unwrap_err();
        assert_eq!(
            error.downcast_ref::<GpuInterruption>(),
            Some(&GpuInterruption::Deadline)
        );

        let control = GpuOperationControl::default();
        let checkpoint: &dyn GpuCheckpoint = &control;
        checkpoint.checkpoint().unwrap();

        let closure = || Ok(());
        GpuCheckpoint::checkpoint(&closure).unwrap();
    }
}

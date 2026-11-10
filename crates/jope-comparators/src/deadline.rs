//! Cooperative wall-clock ceiling for comparator fits.
//!
//! Fits are deterministic work counts (trees, rounds, candidates). The deadline is only an outer
//! safety ceiling: it is checked between units of work, so it can stop a fit early with
//! [`ComparatorError::DeadlineExceeded`] but can never change the model a completed fit returns.

use crate::error::{ComparatorError, ComparatorResult};
use std::time::{Duration, Instant};

/// `Deadline` is a struct for the cooperative fit ceiling contract in this module: an optional
/// instant after which a running fit stops with [`ComparatorError::DeadlineExceeded`].
#[derive(Clone, Copy, Debug)]
pub struct Deadline {
    expires_at: Option<Instant>,
}

impl Deadline {
    /// A deadline that never expires.
    pub fn no_limit() -> Self {
        Self { expires_at: None }
    }

    /// A deadline `budget` from now. A budget too large to represent never expires.
    pub fn after(budget: Duration) -> Self {
        Self {
            expires_at: Instant::now().checked_add(budget),
        }
    }

    /// Whether the deadline has passed.
    pub fn expired(&self) -> bool {
        self.expires_at
            .is_some_and(|expires_at| Instant::now() >= expires_at)
    }

    /// Refuse to continue once the deadline has passed.
    pub fn check(&self) -> ComparatorResult<()> {
        if self.expired() {
            Err(ComparatorError::DeadlineExceeded)
        } else {
            Ok(())
        }
    }
}

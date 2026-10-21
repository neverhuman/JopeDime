//! Time and thread budgets (SPECIFICATION.md §4.3, §12.3).
//!
//! Enforcement in this build is cooperative and stated as such in every receipt: the deadline is
//! checked before each selection fold fit and before fitting a named method, never inside a fit.
//! Once selection has chosen a recipe, its final refit always runs and is charged to the run, so
//! `consumed_seconds` can exceed the limit. One Rayon pool of `threads` workers carries all
//! parallel numerical work. Same-binary worker processes with a parent-enforced wall deadline are
//! the later hard-deadline step.
use crate::Failure;
use anyhow::Result;
use serde::Serialize;
use std::time::{Duration, Instant};

/// How the deadline is enforced, as receipts name it.
pub const ENFORCEMENT: &str = "cooperative: checked before every selection fold fit and before \
     fitting a named method; the chosen recipe's final refit always runs and is charged";

#[derive(Clone, Debug)]
pub struct Budget {
    started: Instant,
    limit: Option<Duration>,
    pub threads: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct BudgetRecord {
    pub time_limit_seconds: Option<f64>,
    pub consumed_seconds: f64,
    pub threads: usize,
    pub enforcement: &'static str,
}

impl Budget {
    /// `threads == 0` means the available parallelism.
    pub fn new(limit: Option<Duration>, threads: usize) -> Self {
        let threads = if threads == 0 {
            std::thread::available_parallelism().map_or(1, usize::from)
        } else {
            threads
        };
        Self {
            started: Instant::now(),
            limit,
            threads,
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    pub fn remaining(&self) -> Option<Duration> {
        self.limit.map(|limit| limit.saturating_sub(self.elapsed()))
    }

    pub fn exhausted(&self) -> bool {
        self.remaining().is_some_and(|r| r.is_zero())
    }

    /// Fail with a budget failure when the deadline has passed.
    pub fn check(&self, before: &str) -> Result<()> {
        if self.exhausted() {
            return Err(Failure::budget(format!(
                "time limit of {:.1}s reached before {before}",
                self.limit.unwrap_or_default().as_secs_f64()
            )));
        }
        Ok(())
    }

    /// Run `work` on a dedicated pool of `threads` workers.
    pub fn install<T: Send>(&self, work: impl FnOnce() -> T + Send) -> Result<T> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.threads)
            .thread_name(|i| format!("jopedime-{i}"))
            .build()?;
        Ok(pool.install(work))
    }

    pub fn record(&self) -> BudgetRecord {
        BudgetRecord {
            time_limit_seconds: self.limit.map(|l| l.as_secs_f64()),
            consumed_seconds: self.elapsed().as_secs_f64(),
            threads: self.threads,
            enforcement: ENFORCEMENT,
        }
    }
}

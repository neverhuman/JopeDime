//! The wall clock a run measures itself with.
//!
//! Every duration a receipt, a `search.json` or an `observations.json` records is a real
//! measurement of this machine, so two runs never agree on those bytes. Tests need to show that
//! everything *else* a run writes is byte-identical — across thread counts in particular — and
//! the only honest way to do that is to take the clock out of the comparison, not to scrub its
//! readings afterwards. `Clock::Frozen` exists for that and only under `cfg(test)`: no command
//! line can reach it, and a frozen clock never reaches a seconds bound, so those comparisons
//! cover runs the evaluation budget ends. Under a binding seconds bound the real clock decides
//! which candidates start, and two real runs need not agree on them. `Clock::Expired` is its
//! opposite, also test-only: every seconds bound has already passed, which is what a slow fit on a
//! loaded host looks like to the admission checks.

use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Clock {
    #[default]
    Wall,
    /// Every elapsed time reads zero and every timestamp is the epoch.
    #[cfg(test)]
    Frozen,
    /// Every elapsed time reads [`EXPIRED`], past the longest seconds budget, and every timestamp
    /// is the epoch.
    #[cfg(test)]
    Expired,
}

/// What [`Clock::Expired`] reads: one second past [`crate::search::MAX_SECONDS`].
#[cfg(test)]
pub const EXPIRED: Duration = Duration::from_secs(crate::search::MAX_SECONDS + 1);

/// A started measurement.
#[derive(Clone, Copy, Debug)]
pub struct Stamp(Reading);

#[derive(Clone, Copy, Debug)]
enum Reading {
    Wall(Instant),
    #[cfg(test)]
    Fixed(Duration),
}

impl Clock {
    pub fn start(self) -> Stamp {
        match self {
            Clock::Wall => Stamp(Reading::Wall(Instant::now())),
            #[cfg(test)]
            Clock::Frozen => Stamp(Reading::Fixed(Duration::ZERO)),
            #[cfg(test)]
            Clock::Expired => Stamp(Reading::Fixed(EXPIRED)),
        }
    }

    /// RFC 3339 UTC, to the second.
    pub fn now_rfc3339(self) -> String {
        match self {
            Clock::Wall => jope_datasets::paths::now_rfc3339(),
            #[cfg(test)]
            Clock::Frozen | Clock::Expired => "1970-01-01T00:00:00Z".to_owned(),
        }
    }
}

impl Stamp {
    pub fn elapsed(&self) -> Duration {
        match self.0 {
            Reading::Wall(started) => started.elapsed(),
            #[cfg(test)]
            Reading::Fixed(reading) => reading,
        }
    }

    pub fn secs(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }
}

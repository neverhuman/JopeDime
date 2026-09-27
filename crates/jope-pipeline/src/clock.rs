//! The wall clock a run measures itself with.
//!
//! Every duration a receipt, a `search.json` or an `observations.json` records is a real
//! measurement of this machine, so two runs never agree on those bytes. Tests need to show that
//! everything *else* a run writes is byte-identical — across thread counts in particular — and
//! the only honest way to do that is to take the clock out of the comparison, not to scrub its
//! readings afterwards. [`Clock::Frozen`] exists for that and only under `cfg(test)`: no command
//! line can reach it, and a frozen clock never reaches a seconds bound.

use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Clock {
    #[default]
    Wall,
    /// Every elapsed time reads zero and every timestamp is the epoch.
    #[cfg(test)]
    Frozen,
}

/// A started measurement.
#[derive(Clone, Copy, Debug)]
pub struct Stamp(Option<Instant>);

impl Clock {
    pub fn start(self) -> Stamp {
        match self {
            Clock::Wall => Stamp(Some(Instant::now())),
            #[cfg(test)]
            Clock::Frozen => Stamp(None),
        }
    }

    /// RFC 3339 UTC, to the second.
    pub fn now_rfc3339(self) -> String {
        match self {
            Clock::Wall => jope_datasets::paths::now_rfc3339(),
            #[cfg(test)]
            Clock::Frozen => "1970-01-01T00:00:00Z".to_owned(),
        }
    }
}

impl Stamp {
    pub fn elapsed(&self) -> Duration {
        self.0.map_or(Duration::ZERO, |started| started.elapsed())
    }

    pub fn secs(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }
}

//! Safe wrappers for the raw OS queries this crate shares: deadline clocks and process identity.
//!
//! Deadlines cross process boundaries here as `CLOCK_MONOTONIC` (or, for the fit authority,
//! `CLOCK_BOOTTIME`) nanoseconds, so the opaque `std::time::Instant` cannot replace them. Eight
//! modules each carried their own `unsafe { libc::clock_gettime(..) }` plus seconds-to-nanos
//! arithmetic with slightly different overflow handling; they now share this one checked path.

use nix::time::{clock_gettime, ClockId};
use std::io;

fn nanos(clock: ClockId) -> io::Result<u64> {
    let time = clock_gettime(clock).map_err(io::Error::from)?;
    let seconds = u64::try_from(time.tv_sec())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "negative clock seconds"))?;
    let nanos = u64::try_from(time.tv_nsec())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "negative clock nanoseconds"))?;
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(nanos))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "clock nanoseconds overflow"))
}

/// `CLOCK_MONOTONIC` in nanoseconds: the same-host deadline clock shared with child processes.
pub fn monotonic_ns() -> io::Result<u64> {
    nanos(ClockId::CLOCK_MONOTONIC)
}

/// `CLOCK_BOOTTIME` in nanoseconds: monotonic, and keeps counting across suspend.
pub fn boottime_ns() -> io::Result<u64> {
    nanos(ClockId::CLOCK_BOOTTIME)
}

/// Effective user id of this process.
pub fn effective_uid() -> u32 {
    nix::unistd::geteuid().as_raw()
}

/// Process id of this process's parent (`getppid(2)`).
pub fn parent_pid() -> u32 {
    std::os::unix::process::parent_id()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clocks_are_nondecreasing_and_agree_with_libc() {
        let first = monotonic_ns().unwrap();
        let mut raw = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: clock_gettime writes one timespec through the valid out-pointer.
        assert_eq!(
            unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut raw) },
            0
        );
        let libc_ns = raw.tv_sec as u64 * 1_000_000_000 + raw.tv_nsec as u64;
        let second = monotonic_ns().unwrap();
        assert!(first <= libc_ns && libc_ns <= second);
        assert!(boottime_ns().unwrap() >= second.saturating_sub(1_000_000_000));
    }

    #[test]
    fn identity_matches_libc() {
        // SAFETY: geteuid and getppid have no preconditions.
        let (euid, ppid) = unsafe { (libc::geteuid(), libc::getppid()) };
        assert_eq!(effective_uid(), euid);
        assert_eq!(parent_pid(), ppid as u32);
    }
}

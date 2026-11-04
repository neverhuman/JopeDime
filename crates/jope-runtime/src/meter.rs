//! Process resource meters for resource-aware objectives: peak heap growth and process CPU time.
//!
//! **Heap.** A binary that wants heap readings installs a counting global allocator that reports
//! every allocation and release here ([`note_alloc`], [`note_dealloc`]); the `jope` binary does.
//! Counting costs one relaxed load per allocation while no window is open. A [`HeapWindow`]
//! reports the largest net growth of the *process* heap since it opened, so it is a per-candidate
//! reading only while one candidate runs at a time (resource-ranked searches evaluate serially).
//! Without a counting allocator [`heap_meter_installed`] is false and no reading is invented.
//!
//! **CPU.** [`process_cpu_seconds`] sums every thread's on-CPU nanoseconds from
//! `/proc/self/task/*/schedstat` (Linux), so work a fit hands to a thread pool is counted.
//!
//! Everything here is safe code: atomics only, no allocation on the counting path.
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU64, Ordering::Relaxed};

static OPEN: AtomicBool = AtomicBool::new(false);
static NET: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);
static SEEN: AtomicU64 = AtomicU64::new(0);

/// Called by a counting allocator for every successful allocation of `size` bytes.
#[inline]
pub fn note_alloc(size: usize) {
    if SEEN.load(Relaxed) == 0 {
        SEEN.store(1, Relaxed);
    }
    if OPEN.load(Relaxed) {
        let now = NET.fetch_add(size as isize, Relaxed) + size as isize;
        PEAK.fetch_max(now, Relaxed);
    }
}

/// Called by a counting allocator for every release of `size` bytes.
#[inline]
pub fn note_dealloc(size: usize) {
    if OPEN.load(Relaxed) {
        NET.fetch_sub(size as isize, Relaxed);
    }
}

/// Whether a counting allocator has reported at least one allocation in this process.
pub fn heap_meter_installed() -> bool {
    SEEN.load(Relaxed) != 0
}

/// An open heap measurement. Only one window is meaningful at a time; opening a second resets
/// the first.
#[derive(Debug)]
pub struct HeapWindow(());

impl HeapWindow {
    /// Open a window at the current heap level, or `None` without a counting allocator.
    pub fn open() -> Option<Self> {
        if !heap_meter_installed() {
            return None;
        }
        NET.store(0, Relaxed);
        PEAK.store(0, Relaxed);
        OPEN.store(true, Relaxed);
        Some(Self(()))
    }

    /// Close the window: the largest net heap growth, in bytes, while it was open.
    pub fn close(self) -> u64 {
        OPEN.store(false, Relaxed);
        PEAK.load(Relaxed).max(0) as u64
    }
}

impl Drop for HeapWindow {
    fn drop(&mut self) {
        OPEN.store(false, Relaxed);
    }
}

/// On-CPU seconds of every thread of this process, or `None` where `/proc` schedstat is absent.
pub fn process_cpu_seconds() -> Option<f64> {
    let mut nanos: u128 = 0;
    for entry in std::fs::read_dir("/proc/self/task").ok()? {
        let Ok(entry) = entry else { continue };
        // A thread can exit between listing and reading; it no longer runs, so skip it.
        let Ok(text) = std::fs::read_to_string(entry.path().join("schedstat")) else {
            continue;
        };
        nanos += text.split_whitespace().next()?.parse::<u128>().ok()?;
    }
    Some(nanos as f64 / 1e9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_reports_net_growth_and_ignores_releases_of_older_memory() {
        // No allocator reports in unit tests, so drive the counters directly.
        note_alloc(1);
        let window = HeapWindow::open().expect("a reported allocation installs the meter");
        note_dealloc(4096); // memory allocated before the window opened
        note_alloc(1000);
        note_alloc(500);
        note_dealloc(1500);
        note_alloc(200);
        assert_eq!(
            window.close(),
            0,
            "growth never exceeded the starting level"
        );
        let window = HeapWindow::open().unwrap();
        note_alloc(1000);
        note_alloc(500);
        note_dealloc(1500);
        note_alloc(200);
        assert_eq!(window.close(), 1500);
        note_alloc(10_000);
        let window = HeapWindow::open().unwrap();
        assert_eq!(window.close(), 0, "a closed window counts nothing");
    }

    #[test]
    fn process_cpu_time_is_monotone_where_proc_exists() {
        let Some(before) = process_cpu_seconds() else {
            return;
        };
        let mut x = 0u64;
        for i in 0..2_000_000u64 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(i);
        }
        std::hint::black_box(x);
        assert!(process_cpu_seconds().unwrap() >= before);
    }
}

//! Exclusive advisory locking shared by the campaign binaries.

use std::fs::{File, TryLockError};
use std::io;

/// Lock this open file description until all its descriptors are closed.
/// Nonblocking contention returns EWOULDBLOCK (`ErrorKind::WouldBlock`); other
/// failures retain their OS error. Neither kind of failure is retried here.
///
/// std's `File::lock`/`File::try_lock` are a single `flock(2)` call with
/// `LOCK_EX`/`LOCK_EX | LOCK_NB` on Linux, so no unsafe code is needed.
pub fn exclusive(file: &File, blocking: bool) -> io::Result<()> {
    if blocking {
        return file.lock();
    }
    file.try_lock().map_err(|error| match error {
        // Keep the raw errno so callers' messages stay "Resource temporarily unavailable".
        TryLockError::WouldBlock => io::Error::from_raw_os_error(libc::EWOULDBLOCK),
        TryLockError::Error(error) => error,
    })
}

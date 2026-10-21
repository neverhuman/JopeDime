//! Exclusive advisory locking shared by the campaign binaries.

use std::fs::{File, TryLockError};
use std::io;

/// Lock this open file description until all its descriptors are closed.
///
/// `File::lock`/`File::try_lock` are `flock(2)` with `LOCK_EX` (and `LOCK_NB`) on Linux, so the
/// lock interoperates with every other `flock` holder of the same file. Nonblocking contention
/// returns the raw EWOULDBLOCK error (`ErrorKind::WouldBlock`, same message as before); other
/// failures retain their OS error. Neither kind of failure is retried here.
pub fn exclusive(file: &File, blocking: bool) -> io::Result<()> {
    if blocking {
        return file.lock();
    }
    file.try_lock().map_err(|error| match error {
        TryLockError::WouldBlock => io::Error::from_raw_os_error(libc::EWOULDBLOCK),
        TryLockError::Error(error) => error,
    })
}

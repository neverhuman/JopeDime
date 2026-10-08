//! Exclusive advisory locking shared by the campaign binaries.

use std::io;
use std::os::fd::{AsRawFd, BorrowedFd};

/// Lock this open file description until all its descriptors are closed.
/// Nonblocking contention returns EWOULDBLOCK (`ErrorKind::WouldBlock`); other
/// failures retain their OS error. Neither kind of failure is retried here.
pub fn exclusive(fd: BorrowedFd<'_>, blocking: bool) -> io::Result<()> {
    let operation: libc::c_int = if blocking {
        libc::LOCK_EX
    } else {
        libc::LOCK_EX | libc::LOCK_NB
    };
    // SAFETY: BorrowedFd keeps the caller's live descriptor borrowed for this
    // call. Its raw value and these libc lock flags have the required c_int ABI.
    // flock takes no pointers, neither closes nor transfers the descriptor, and
    // returns synchronously; the owning File remains responsible for its lifetime.
    let status = unsafe { libc::flock(fd.as_raw_fd(), operation) };
    if status != 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

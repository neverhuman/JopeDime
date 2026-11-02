use super::*;

/// The TLS server name a client sends; identity is the pinned certificate, never this name.
pub(crate) const SERVER_NAME: &str = "jope-bounded-peer";
/// Bound on the courtesy `close_notify` of a dropped [`TlsStream`], whatever its write timeout.
pub const CLOSE_TIMEOUT: Duration = Duration::from_secs(1);

/// A socket whose kernel timeouts bound one read or one write at a time.
pub trait TimedSocket: Read + Write {
    fn apply_read_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()>;
    fn apply_write_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()>;
}
impl TimedSocket for TcpStream {
    fn apply_read_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()> {
        self.set_read_timeout(timeout)
    }
    fn apply_write_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()> {
        self.set_write_timeout(timeout)
    }
}
impl TimedSocket for UnixStream {
    fn apply_read_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()> {
        self.set_read_timeout(timeout)
    }
    fn apply_write_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()> {
        self.set_write_timeout(timeout)
    }
}
impl TimedSocket for &UnixStream {
    fn apply_read_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()> {
        self.set_read_timeout(timeout)
    }
    fn apply_write_timeout(&self, timeout: Option<Duration>) -> std::io::Result<()> {
        self.set_write_timeout(timeout)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Limits {
    read: Option<Duration>,
    write: Option<Duration>,
    deadline: Option<Instant>,
}
/// A socket whose every blocking read and write is bounded twice: by a per-operation timeout
/// (inactivity) and by an optional absolute deadline. Before each operation the kernel
/// timeout is set to the smaller of the two, so no single read or write can outlast the
/// deadline; at or after the deadline an operation is refused with `TimedOut` and the socket
/// is not touched. The deadline covers every operation a caller makes through this socket,
/// including each of the many a TLS connection can make inside one call.
pub struct DeadlineSocket<S> {
    socket: S,
    limits: Mutex<Limits>,
}
impl<S: TimedSocket> DeadlineSocket<S> {
    /// No per-operation timeout and no deadline until [`ControlStream::set_timeouts`] and
    /// [`ControlStream::set_deadline`] set them.
    pub fn new(socket: S) -> Self {
        Self {
            socket,
            limits: Mutex::new(Limits::default()),
        }
    }
    pub fn get_ref(&self) -> &S {
        &self.socket
    }
    fn limits(&self) -> MutexGuard<'_, Limits> {
        self.limits.lock().unwrap_or_else(PoisonError::into_inner)
    }
    /// Bring the deadline forward to `cap` if that is sooner; it is never moved later.
    fn cap_deadline(&self, cap: Instant) {
        let mut limits = self.limits();
        limits.deadline = Some(limits.deadline.map_or(cap, |d| d.min(cap)));
    }
    /// The kernel timeout for the next read (`write == false`) or write: its per-operation
    /// timeout, cut to what is left before the deadline.
    fn next_timeout(&self, write: bool) -> std::io::Result<Option<Duration>> {
        let limits = *self.limits();
        let per_operation = if write { limits.write } else { limits.read };
        let Some(deadline) = limits.deadline else {
            return Ok(per_operation);
        };
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "the I/O deadline has passed",
            ));
        }
        Ok(Some(per_operation.map_or(left, |t| t.min(left))))
    }
}
impl<S: TimedSocket> Read for DeadlineSocket<S> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let timeout = self.next_timeout(false)?;
        self.socket.apply_read_timeout(timeout)?;
        self.socket.read(buf)
    }
}
impl<S: TimedSocket> Write for DeadlineSocket<S> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let timeout = self.next_timeout(true)?;
        self.socket.apply_write_timeout(timeout)?;
        self.socket.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.socket.flush()
    }
}
impl<S: TimedSocket + Send> ControlStream for DeadlineSocket<S> {
    fn set_timeouts(&self, read: Option<Duration>, write: Option<Duration>) -> std::io::Result<()> {
        if [read, write].into_iter().flatten().any(|t| t.is_zero()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot set a 0 duration timeout",
            ));
        }
        let mut limits = self.limits();
        (limits.read, limits.write) = (read, write);
        Ok(())
    }
    fn set_deadline(&self, deadline: Option<Instant>) -> std::io::Result<()> {
        self.limits().deadline = deadline;
        Ok(())
    }
}
type Socket = DeadlineSocket<TcpStream>;

use super::*;

/// Frames carrying strictly consecutive delivery ids (1, 2, 3, ...) on one stream, for
/// request/response protocols. The frame limit covers the 8-byte id. A repeated, regressed
/// or skipped id refuses the channel for good.
pub struct Sequenced<S: Read + Write> {
    stream: S,
    limit: u32,
    sent: u64,
    received: u64,
    refused: bool,
}
impl<S: Read + Write> Sequenced<S> {
    pub fn new(stream: S, limit: u32) -> Self {
        Self {
            stream,
            limit,
            sent: 0,
            received: 0,
            refused: false,
        }
    }
    pub fn get_ref(&self) -> &S {
        &self.stream
    }
    /// Send `body` under the next delivery id and return that id.
    pub fn send(&mut self, body: &[u8]) -> Result<u64> {
        ensure!(!self.refused, "sequenced channel already refused");
        let id = self.sent + 1;
        let mut frame = id.to_be_bytes().to_vec();
        frame.extend_from_slice(body);
        write_frame(&mut self.stream, &frame, self.limit)?;
        self.sent = id;
        Ok(id)
    }
    /// Receive the next frame; only the next consecutive delivery id is accepted.
    pub fn receive(&mut self) -> Result<(u64, Vec<u8>)> {
        ensure!(!self.refused, "sequenced channel already refused");
        let frame = read_frame(&mut self.stream, self.limit);
        self.accept(frame)
    }
    fn accept(&mut self, frame: Result<Vec<u8>>) -> Result<(u64, Vec<u8>)> {
        let result = frame.and_then(|frame| {
            ensure!(frame.len() >= 8, "frame without a delivery id");
            let id = u64::from_be_bytes(frame[..8].try_into()?);
            ensure!(
                id == self.received + 1,
                "delivery id {id} is not the next id {}: replayed, duplicated or reordered",
                self.received + 1
            );
            Ok((id, frame[8..].to_vec()))
        });
        match result {
            Ok((id, body)) => {
                self.received = id;
                Ok((id, body))
            }
            Err(e) => {
                self.refused = true;
                Err(e)
            }
        }
    }
}
impl<S: ControlStream> Sequenced<S> {
    /// Run `op` with every blocking read and write on the stream ending by `deadline`, then
    /// remove the deadline.
    fn capped<T>(
        &mut self,
        deadline: Instant,
        op: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        self.stream.set_deadline(Some(deadline))?;
        let result = op(self);
        let cleared = self.stream.set_deadline(None);
        let value = result?;
        cleared?;
        Ok(value)
    }
    /// [`Sequenced::send`], ending by `deadline` however slowly the peer reads.
    pub fn send_by(&mut self, body: &[u8], deadline: Instant) -> Result<u64> {
        self.capped(deadline, |s| s.send(body))
    }
    /// [`Sequenced::receive`], ending by `deadline` however the peer paces its bytes.
    pub fn receive_by(&mut self, deadline: Instant) -> Result<(u64, Vec<u8>)> {
        self.capped(deadline, Self::receive)
    }
    /// Like [`Sequenced::receive`], for a server waiting on an idle peer: `Ok(None)` when no
    /// byte arrived within `wait` (nothing was consumed, the channel stays usable). A frame
    /// that has started must arrive whole within `within` of its first byte, however its
    /// bytes are paced, or the channel is refused.
    pub fn poll_receive(
        &mut self,
        wait: Duration,
        within: Duration,
    ) -> Result<Option<(u64, Vec<u8>)>> {
        ensure!(!self.refused, "sequenced channel already refused");
        let mut first = [0u8; 1];
        match self.capped(Instant::now() + wait, |s| Ok(s.stream.read(&mut first)))? {
            Ok(0) => {
                self.refused = true;
                bail!("peer closed the sequenced channel")
            }
            Ok(_) => {}
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                return Ok(None)
            }
            Err(e) => {
                self.refused = true;
                return Err(e.into());
            }
        }
        let limit = self.limit;
        let frame = self.capped(Instant::now() + within, |s| {
            let mut rest = [0u8; 3];
            s.stream.read_exact(&mut rest)?;
            let len = u32::from_be_bytes([first[0], rest[0], rest[1], rest[2]]);
            ensure!(
                len <= limit,
                "frame of {len} bytes exceeds message limit {limit}"
            );
            let mut body = vec![0u8; len as usize];
            s.stream.read_exact(&mut body)?;
            Ok(body)
        });
        self.accept(frame).map(Some)
    }
}

//! Streaming SSE decoding and explicit reconnect, without an async runtime.
use crate::transport::{header, Error, HeaderValue, Request, Result, Transport, MAX_JSON_BYTES};
use jope_api::{decimal::DecimalU64, ids::RunId, PublicEvent, TelemetryEvent};
use std::io::{BufRead, BufReader, Read};

/// A validated event from an SSE subscription.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamEvent {
    /// A durable event with a decimal sequence cursor.
    Durable(Box<PublicEvent>),
    /// A telemetry event, which does not advance the durable cursor.
    Telemetry(TelemetryEvent),
}

/// Incrementally decodes one SSE response and validates its event envelopes.
///
/// Replayed durable events at or before `after` are skipped. An incomplete or malformed frame
/// produces an error and terminates iteration; callers decide whether and when to reconnect.
pub struct SseReader<R: Read> {
    reader: BufReader<R>,
    run: RunId,
    cursor: Option<DecimalU64>,
    first_line: bool,
    skip_lf: bool,
    done: bool,
}
impl<R: Read> SseReader<R> {
    /// Start reading events for `run`, optionally resuming after a durable sequence.
    pub fn new(reader: R, run: RunId, after: Option<DecimalU64>) -> Self {
        Self {
            reader: BufReader::new(reader),
            run,
            cursor: after,
            first_line: true,
            skip_lf: false,
            done: false,
        }
    }
    /// Return the last accepted durable event sequence, if one has been read.
    pub fn last_event_id(&self) -> Option<DecimalU64> {
        self.cursor
    }
    fn line(&mut self, budget: usize) -> Result<Option<(Vec<u8>, usize)>> {
        let mut line = Vec::new();
        let mut consumed = 0;
        loop {
            let bytes = self.reader.fill_buf()?;
            if bytes.is_empty() {
                return Ok((!line.is_empty()).then_some((line, consumed)));
            }
            if self.skip_lf {
                self.skip_lf = false;
                if bytes[0] == b'\n' {
                    self.reader.consume(1);
                    consumed += 1;
                    if consumed > budget {
                        return Err(Error::Protocol("SSE frame exceeds 8 MiB"));
                    }
                    continue;
                }
            }
            let end = bytes.iter().position(|b| matches!(b, b'\r' | b'\n'));
            let count = end.map_or(bytes.len(), |n| n + 1);
            if consumed + count > budget {
                return Err(Error::Protocol("SSE frame exceeds 8 MiB"));
            }
            line.extend_from_slice(&bytes[..end.unwrap_or(count)]);
            let carriage_return = end.is_some_and(|i| bytes[i] == b'\r');
            self.reader.consume(count);
            consumed += count;
            if end.is_some() {
                self.skip_lf = carriage_return;
                return Ok(Some((line, consumed)));
            }
        }
    }
    fn read_event(&mut self) -> Result<Option<StreamEvent>> {
        loop {
            let mut data = Vec::new();
            let mut id = None;
            let mut size = 0;
            loop {
                let Some((mut line, n)) = self.line(MAX_JSON_BYTES - size)? else {
                    return Ok(None);
                }; // Incomplete frames are never acknowledged.
                size += n;
                if self.first_line {
                    self.first_line = false;
                    if line.starts_with(&[0xef, 0xbb, 0xbf]) {
                        line.drain(..3);
                    }
                }
                if line.is_empty() {
                    break;
                }
                if line.starts_with(b":") {
                    continue;
                }
                let line =
                    std::str::from_utf8(&line).map_err(|_| Error::Protocol("SSE is not UTF-8"))?;
                let (field, value) = line.split_once(':').unwrap_or((line, ""));
                let value = value.strip_prefix(' ').unwrap_or(value);
                match field {
                    "data" => {
                        data.extend_from_slice(value.as_bytes());
                        data.push(b'\n');
                    }
                    "id" => {
                        if value.contains('\0') {
                            return Err(Error::Protocol("invalid SSE id"));
                        }
                        id = Some(value.to_owned());
                    }
                    // Event labels and retry hints do not change the typed JSON
                    // envelope or impose an automatic reconnect policy.
                    _ => {}
                }
            }
            if data.is_empty() {
                continue;
            }
            data.pop();
            let json: serde_json::Value = serde_json::from_slice(&data)?;
            if json.get("kind").and_then(|v| v.as_str()) == Some("telemetry") {
                if id.is_some() || json.get("sequence").is_some() {
                    return Err(Error::Protocol("telemetry must not carry a durable id"));
                }
                let event: TelemetryEvent = serde_json::from_value(json)?;
                if event.run_id != self.run {
                    return Err(Error::Protocol("SSE run does not match subscription"));
                }
                return Ok(Some(StreamEvent::Telemetry(event)));
            }
            let id = id.ok_or(Error::Protocol("durable event is missing its SSE id"))?;
            let sequence: DecimalU64 = serde_json::from_value(serde_json::Value::String(id))?;
            let event: PublicEvent = serde_json::from_value(json)?;
            if event.run_id != self.run || event.sequence != sequence {
                return Err(Error::Protocol(
                    "SSE id/run does not match durable envelope",
                ));
            }
            if self.cursor.is_some_and(|old| sequence <= old) {
                continue;
            }
            self.cursor = Some(sequence);
            return Ok(Some(StreamEvent::Durable(Box::new(event))));
        }
    }
}
impl<R: Read> Iterator for SseReader<R> {
    type Item = Result<StreamEvent>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.read_event() {
            Ok(Some(event)) => Some(Ok(event)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

/// An explicit-reconnect event stream over a [`Transport`].
///
/// The stream does not reconnect automatically and does not retry requests. Call
/// [`reconnect`](Self::reconnect) after handling an iteration error or end of stream.
pub struct EventStream<'a, T: Transport + ?Sized> {
    transport: &'a T,
    run: RunId,
    reader: SseReader<Box<dyn Read + Send>>,
}
impl<'a, T: Transport + ?Sized> EventStream<'a, T> {
    /// Subscribe to events for `run`, resuming after `after` when provided.
    pub fn new(transport: &'a T, run: RunId, after: Option<DecimalU64>) -> Result<Self> {
        let reader = Self::connect(transport, &run, after)?;
        Ok(Self {
            transport,
            run,
            reader,
        })
    }
    fn connect(
        transport: &T,
        run: &RunId,
        after: Option<DecimalU64>,
    ) -> Result<SseReader<Box<dyn Read + Send>>> {
        let mut request = Request::new(11, 0, &[("id", &run.to_string())])?;
        request.headers.insert(
            header::ACCEPT,
            HeaderValue::from_static("text/event-stream"),
        );
        if let Some(after) = after {
            request.headers.insert(
                "last-event-id",
                HeaderValue::from_str(&after.0.to_string()).unwrap(),
            );
        }
        let response = transport.execute(request)?.checked("text/event-stream")?;
        Ok(SseReader::new(response.body, run.clone(), after))
    }
    /// Return the last accepted durable event sequence, if one has been read.
    pub fn last_event_id(&self) -> Option<DecimalU64> {
        self.reader.last_event_id()
    }
    /// Reconnect is explicit: no hidden retries, sleeps or mutation replay.
    pub fn reconnect(&mut self) -> Result<()> {
        let next = Self::connect(self.transport, &self.run, self.last_event_id())?;
        self.reader = next;
        Ok(())
    }
}
impl<T: Transport + ?Sized> Iterator for EventStream<'_, T> {
    type Item = Result<StreamEvent>;
    fn next(&mut self) -> Option<Self::Item> {
        self.reader.next()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub const RUN: &str = "run_01J00000000000000000000000";
    pub fn durable(sequence: u64) -> String {
        serde_json::json!({"run_id":RUN,"sequence":sequence.to_string(),"time":"2026-01-01T00:00:00Z",
            "kind":"run.state","body":{"state":"running","phase":"searching","pending":null,"reason":null}}).to_string()
    }
    pub fn telemetry() -> String {
        serde_json::json!({"run_id":RUN,"time":"2026-01-01T00:00:00Z","kind":"telemetry","body":[]})
            .to_string()
    }
    fn reader(text: String, after: Option<u64>) -> SseReader<std::io::Cursor<Vec<u8>>> {
        SseReader::new(
            std::io::Cursor::new(text.into_bytes()),
            RUN.parse().unwrap(),
            after.map(DecimalU64),
        )
    }
    #[test]
    fn durable_telemetry_and_replays_keep_an_exact_decimal_cursor() {
        let n = 9_007_199_254_740_993;
        let bytes = format!("\u{feff}: keepalive\r\nid: {n}\r\ndata: {}\r\n\r\ndata: {}\n\nid: {n}\ndata: {}\n\nid: {}\ndata: {}\n\n",
            durable(n),telemetry(),durable(n),n+1,durable(n+1));
        let mut r = reader(bytes, None);
        assert!(matches!(
            r.next().unwrap().unwrap(),
            StreamEvent::Durable(_)
        ));
        assert_eq!(r.last_event_id(), Some(DecimalU64(n)));
        assert!(matches!(
            r.next().unwrap().unwrap(),
            StreamEvent::Telemetry(_)
        ));
        assert_eq!(r.last_event_id(), Some(DecimalU64(n)));
        assert!(matches!(
            r.next().unwrap().unwrap(),
            StreamEvent::Durable(_)
        ));
        assert_eq!(r.last_event_id(), Some(DecimalU64(n + 1)));
        assert!(r.next().is_none());
    }
    #[test]
    fn incomplete_and_invalid_frames_never_advance_the_cursor() {
        for ending in ["\n", "\r\n", "\r"] {
            let mut partial = reader(
                format!("id: 2{ending}data: {}{ending}", durable(2)),
                Some(1),
            );
            assert!(partial.next().is_none());
            assert_eq!(partial.last_event_id(), Some(DecimalU64(1)));
        }
        for (id, payload) in [
            ("02", durable(2)),
            ("3", durable(2)),
            ("2", "bad json".into()),
            ("2", telemetry()),
            (
                "2",
                durable(2).replace(RUN, "run_01J00000000000000000000001"),
            ),
        ] {
            let mut r = reader(format!("id: {id}\ndata: {payload}\n\n"), Some(1));
            assert!(r.next().unwrap().is_err(), "{id} {payload}");
            assert_eq!(r.last_event_id(), Some(DecimalU64(1)));
            assert!(r.next().is_none());
        }
    }
    #[test]
    fn split_reads_multiline_data_and_ignored_fields_work() {
        struct OneByte(std::io::Cursor<Vec<u8>>);
        impl Read for OneByte {
            fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
                let n = b.len().min(1);
                self.0.read(&mut b[..n])
            }
        }
        let json = durable(1).replacen('{', "{\n", 1);
        let text = format!(
            "retry: 1000\nunknown: field\nid: 1\ndata: {}\n\n",
            json.replace('\n', "\ndata: ")
        );
        let mut r = SseReader::new(
            OneByte(std::io::Cursor::new(text.into_bytes())),
            RUN.parse().unwrap(),
            None,
        );
        assert!(r.next().unwrap().is_ok());
        assert!(r.next().is_none());
    }
    #[test]
    fn a_single_unbounded_line_is_refused() {
        let mut r = reader(format!("data: {}\n\n", "x".repeat(MAX_JSON_BYTES)), None);
        assert!(matches!(r.next(), Some(Err(Error::Protocol(_)))));
        assert_eq!(r.last_event_id(), None);
    }
    #[test]
    fn carriage_return_line_endings_dispatch_without_a_newline() {
        let mut r = reader(format!("id: 1\rdata: {}\r\r", durable(1)), None);
        assert!(matches!(r.next(), Some(Ok(StreamEvent::Durable(_)))));
        assert_eq!(r.last_event_id(), Some(DecimalU64(1)));
    }
}

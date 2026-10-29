//! Append-only JSONL source of truth for events and low-rate records.

use crate::event::{flags, Event};
use serde::ser::Serializer;
use serde::Serialize;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const EVENT_FILE: &str = "events.jsonl";
pub const WRITER_BUFFER_BYTES: usize = 64 * 1024;
pub const MAX_RECORD_BYTES: usize = 16 * 1024;
pub const RECORD_QUEUE_CAPACITY: usize = 64;
const RECOVERY_TAIL_CAP: u64 = MAX_RECORD_BYTES as u64;

#[derive(Serialize)]
struct EventLine<'a> {
    kind: &'static str,
    schema: &'static str,
    event: EventPayload<'a>,
}

#[derive(Serialize)]
struct EventPayload<'a> {
    #[serde(serialize_with = "serialize_run_id")]
    run_id: &'a [u8; 16],
    producer_sequence: u64,
    monotonic_ns: u64,
    step: u64,
    source_id: u16,
    metric_id: u16,
    group_id: u16,
    event_kind: u8,
    flags: u8,
    value: Option<f64>,
    nonfinite: Option<&'static str>,
    availability_reason_id: Option<u16>,
}

fn serialize_run_id<S>(run_id: &&[u8; 16], serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    struct Hex<'a>(&'a [u8; 16]);
    impl fmt::Display for Hex<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            for byte in self.0 {
                write!(f, "{byte:02x}")?;
            }
            Ok(())
        }
    }
    serializer.collect_str(&Hex(run_id))
}

pub struct JsonlWriter {
    path: PathBuf,
    writer: BufWriter<File>,
}

impl JsonlWriter {
    pub fn create(run_dir: &Path) -> io::Result<Self> {
        let path = run_dir.join(EVENT_FILE);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok(Self {
            path,
            writer: BufWriter::with_capacity(WRITER_BUFFER_BYTES, file),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn append_event(&mut self, event: &Event) -> io::Result<()> {
        let (value, nonfinite) = if event.flags & flags::VALUE_PRESENT == 0 {
            (None, None)
        } else if event.value.is_finite() {
            (Some(event.value), None)
        } else {
            let reason = if event.value.is_nan() {
                "nan"
            } else if event.value.is_sign_positive() {
                "+inf"
            } else {
                "-inf"
            };
            (None, Some(reason))
        };
        let line = EventLine {
            kind: "event",
            schema: "jope.telemetry.event/v1",
            event: EventPayload {
                run_id: &event.run_id,
                producer_sequence: event.sequence,
                monotonic_ns: event.monotonic_ns,
                step: event.step,
                source_id: event.source_id,
                metric_id: event.metric_id,
                group_id: event.group_id,
                event_kind: event.kind,
                flags: event.flags,
                value,
                nonfinite,
                availability_reason_id: (event.availability_reason_id != 0)
                    .then_some(event.availability_reason_id),
            },
        };
        serde_json::to_writer(&mut self.writer, &line).map_err(io::Error::other)?;
        self.writer.write_all(b"\n")
    }

    pub fn append_record_line(&mut self, line: &[u8]) -> io::Result<()> {
        self.writer.write_all(line)
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    pub fn into_inner(self) -> io::Result<File> {
        self.writer.into_inner().map_err(|e| e.into_error())
    }
}

/// Repair only a malformed final line from an interrupted writer. Complete malformed lines and
/// oversized tails are errors; earlier bytes are never rewritten. A valid final JSON value that
/// lacks its newline is completed by appending the delimiter.
pub fn recover_final_line(path: &Path) -> io::Result<u64> {
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)?;
    let len = file.metadata()?.len();
    if len == 0 {
        return Ok(0);
    }
    file.seek(SeekFrom::End(-1))?;
    let mut last = [0_u8; 1];
    file.read_exact(&mut last)?;
    if last[0] == b'\n' {
        return Ok(0);
    }
    let mut start = len;
    let mut byte = [0_u8; 1];
    while start > 0 && len - start < RECOVERY_TAIL_CAP {
        start -= 1;
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(&mut byte)?;
        if byte[0] == b'\n' {
            start += 1;
            break;
        }
    }
    if len - start > RECOVERY_TAIL_CAP {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "final JSONL line exceeds recovery cap",
        ));
    }
    file.seek(SeekFrom::Start(start))?;
    let mut tail = Vec::with_capacity((len - start) as usize);
    file.read_to_end(&mut tail)?;
    if serde_json::from_slice::<serde_json::Value>(&tail).is_ok() {
        file.seek(SeekFrom::End(0))?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        return Ok(0);
    }
    let truncated = len - start;
    file.set_len(start)?;
    file.sync_all()?;
    Ok(truncated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::Event;
    #[test]
    fn writes_finite_and_nonfinite_values_without_fabricating_zero() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = JsonlWriter::create(dir.path()).unwrap();
        writer
            .append_event(&Event::scalar([0xabu8; 16], 4, 5, 6, 7, 8, 9, 1.25, true))
            .unwrap();
        writer
            .append_event(&Event::scalar(
                [0xabu8; 16],
                5,
                6,
                6,
                7,
                8,
                10,
                f64::NAN,
                false,
            ))
            .unwrap();
        writer
            .append_event(&Event::unavailable(
                [0xabu8; 16],
                6,
                7,
                6,
                7,
                8,
                11,
                1,
                false,
            ))
            .unwrap();
        writer.flush().unwrap();
        drop(writer);
        let text = std::fs::read_to_string(dir.path().join(EVENT_FILE)).unwrap();
        assert!(text.contains("\"value\":1.25"));
        assert!(text.contains("\"value\":null,\"nonfinite\":\"nan\""));
        assert!(text.contains("\"availability_reason_id\":1"));
        assert!(text.contains("abababababababababababababababab"));
    }

    #[test]
    fn recovery_truncates_only_a_malformed_unterminated_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(EVENT_FILE);
        std::fs::write(&path, b"{\"kind\":\"good\"}\n{\"kind\":").unwrap();
        assert_eq!(recover_final_line(&path).unwrap(), 8);
        assert_eq!(std::fs::read(&path).unwrap(), b"{\"kind\":\"good\"}\n");
    }

    #[test]
    fn recovery_completes_valid_json_and_rejects_malformed_complete_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(EVENT_FILE);
        std::fs::write(&path, b"{\"kind\":\"complete\"}").unwrap();
        assert_eq!(recover_final_line(&path).unwrap(), 0);
        assert!(std::fs::read(&path).unwrap().ends_with(b"\n"));
        std::fs::write(&path, b"{bad}\n").unwrap();
        assert_eq!(recover_final_line(&path).unwrap(), 0);
        assert_eq!(std::fs::read(&path).unwrap(), b"{bad}\n");
    }
}

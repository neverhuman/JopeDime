//! Bounded low-rate record path for generation, epoch, alarm, import and round summaries.

use serde::Serialize;
use serde_json::Value;
use std::io::Write;
use std::sync::Mutex;

use crate::sink::jsonl::MAX_RECORD_BYTES;

const MAX_RECORD_KIND_BYTES: usize = 128;

#[derive(Serialize)]
struct RecordEnvelope<'a, T: ?Sized> {
    kind: &'a str,
    body: &'a T,
}

/// Serialize a value into a bounded JSONL envelope. The complete queued line, including newline,
/// is at most MAX_RECORD_BYTES.
pub fn encode_record_line(kind: &str, body: &Value) -> Result<Vec<u8>, String> {
    encode_serializable_record_line(kind, body)
}

/// Serialize a typed record directly to the bounded envelope without first materializing a
/// potentially large serde_json::Value.
pub fn encode_serializable_record_line<T: Serialize + ?Sized>(
    kind: &str,
    body: &T,
) -> Result<Vec<u8>, String> {
    validate_kind(kind)?;
    let mut writer = CappedVec::new(MAX_RECORD_BYTES - 1);
    serde_json::to_writer(&mut writer, &RecordEnvelope { kind, body })
        .map_err(|error| error.to_string())?;
    writer.bytes.push(b'\n');
    Ok(writer.bytes)
}

/// Validate and wrap caller-encoded JSON without constructing a JSON tree.
pub fn encode_record_json_line(kind: &str, body_json: &str) -> Result<Vec<u8>, String> {
    validate_kind(kind)?;
    if body_json.len() > MAX_RECORD_BYTES {
        return Err("telemetry record exceeds size cap".to_owned());
    }
    serde_json::from_str::<serde::de::IgnoredAny>(body_json)
        .map_err(|error| format!("record body is not valid JSON: {error}"))?;
    let mut writer = CappedVec::new(MAX_RECORD_BYTES - 1);
    writer
        .write_all(b"{\"kind\":")
        .map_err(|error| error.to_string())?;
    serde_json::to_writer(&mut writer, kind).map_err(|error| error.to_string())?;
    writer
        .write_all(b",\"body\":")
        .and_then(|_| writer.write_all(body_json.as_bytes()))
        .and_then(|_| writer.write_all(b"}"))
        .map_err(|error| error.to_string())?;
    writer.bytes.push(b'\n');
    Ok(writer.bytes)
}

fn validate_kind(kind: &str) -> Result<(), String> {
    if kind.is_empty() || kind.len() > MAX_RECORD_KIND_BYTES {
        return Err("record kind is empty or exceeds 128 UTF-8 bytes".to_owned());
    }
    Ok(())
}

struct CappedVec {
    bytes: Vec<u8>,
    max: usize,
}

impl CappedVec {
    fn new(max: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(max),
            max,
        }
    }
}

impl std::io::Write for CappedVec {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.bytes.len().saturating_add(bytes.len()) > self.max {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "telemetry record exceeds size cap",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub trait RecordSink: Send + Sync {
    /// Queue a pre-encoded bounded JSONL record without waiting.
    fn record_line(&self, kind: &str, line: Vec<u8>) -> bool;

    /// Called when a typed or JSON record failed bounded encoding.
    fn record_encoding_error(&self, _oversized: bool) {}

    fn record_value(&self, kind: &str, body: Value) -> bool {
        match encode_record_line(kind, &body) {
            Ok(line) => self.record_line(kind, line),
            Err(error) => {
                self.record_encoding_error(error.contains("exceeds size cap"));
                false
            }
        }
    }
}

/// Serialize and queue a typed low-rate record without an intermediate JSON value tree.
pub fn record<T: Serialize + ?Sized>(sink: &dyn RecordSink, kind: &str, rec: &T) -> bool {
    match encode_serializable_record_line(kind, rec) {
        Ok(line) => sink.record_line(kind, line),
        Err(error) => {
            sink.record_encoding_error(error.contains("exceeds size cap"));
            false
        }
    }
}

/// In-memory sink for tests and for jtel import.
#[derive(Default)]
pub struct VecSink {
    pub records: Mutex<Vec<(String, Value)>>,
}

#[derive(serde::Deserialize)]
struct Envelope {
    kind: String,
    body: Value,
}

impl RecordSink for VecSink {
    fn record_line(&self, _kind: &str, line: Vec<u8>) -> bool {
        let Ok(envelope) = serde_json::from_slice::<Envelope>(&line) else {
            return false;
        };
        self.records
            .lock()
            .map(|mut records| records.push((envelope.kind, envelope.body)))
            .is_ok()
    }
}

#[cfg(test)]
mod bounded_tests {
    use super::*;

    #[test]
    fn bounded_record_encoder_writes_one_jsonl_envelope() {
        let line =
            encode_record_line("jope.telemetry.test/v1", &serde_json::json!({"n": 3})).unwrap();
        assert_eq!(line.last(), Some(&b'\n'));
        let parsed: serde_json::Value = serde_json::from_slice(&line).unwrap();
        assert_eq!(parsed["kind"], "jope.telemetry.test/v1");
        assert_eq!(parsed["body"]["n"], 3);
        assert!(encode_record_line(&"k".repeat(129), &Value::Null).is_err());
    }

    #[test]
    fn typed_and_encoded_records_are_bounded() {
        let line = encode_serializable_record_line("typed/v1", &vec!["x"; 3]).unwrap();
        assert!(serde_json::from_slice::<Value>(&line).is_ok());
        assert!(encode_record_json_line("raw/v1", r#"{"ok":true}"#).is_ok());
        assert!(encode_record_json_line("raw/v1", "{broken").is_err());
        assert!(encode_record_json_line(
            "raw/v1",
            &format!("\"{}\"", "x".repeat(MAX_RECORD_BYTES))
        )
        .is_err());
    }

    #[test]
    fn oversized_record_is_refused_before_queueing() {
        let body = serde_json::json!({"payload": "x".repeat(MAX_RECORD_BYTES)});
        assert!(encode_record_line("test/v1", &body).is_err());
    }
}

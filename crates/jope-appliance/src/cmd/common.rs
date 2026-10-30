//! Shared batch-command helpers.
#![allow(dead_code)] // Command handlers consume these shared helpers in their own task scopes.

use serde::Serialize;
use serde_json::{Map, Value};
use std::fmt::Display;
use std::io::{self, Write};
use ulid::Ulid;

/// Use the supplied idempotency key or generate a CLI-scoped ULID key.
pub(crate) fn resolve_idempotency_key(key: Option<&str>) -> String {
    key.map(str::to_owned)
        .unwrap_or_else(|| format!("cli_{}", Ulid::new()))
}

/// Return the catalogued process exit code for an API error code.
///
/// `ErrorCode` implements [`Display`] as its stable wire name. Keeping this helper generic avoids
/// coupling the appliance command crate to the API crate just to map that wire vocabulary.
pub(crate) fn error_exit_code(error_code: impl Display) -> Option<u8> {
    match error_code.to_string().as_str() {
        "INVALID_INPUT"
        | "DATASET_INVALID"
        | "TASK_AMBIGUOUS"
        | "OBJECTIVE_INVALID"
        | "OBJECTIVE_REQUIRES_ORDER"
        | "LANGUAGE_INTERSECTION_EMPTY"
        | "CAPABILITY_UNAVAILABLE"
        | "NOT_FOUND" => Some(2),
        "IMMUTABLE_FIELD" | "REVISION_CONFLICT" | "IDEMPOTENCY_CONFLICT" | "PLAN_CHANGED" => {
            Some(8)
        }
        "MODEL_INTEGRITY_FAILED" => Some(3),
        "BACKEND_UNAVAILABLE" => Some(4),
        "NO_VALID_MODEL" => Some(5),
        "NUMERICAL_FAILURE" => Some(6),
        "IO_FAILURE" | "OUTPUT_EXISTS" | "DISK_LOW" => Some(7),
        "UNAUTHORIZED" | "ENGINE_UNREACHABLE" | "ENGINE_BUSY" => Some(9),
        "FITNESS_UNDEFINED" => None,
        _ => None,
    }
}

/// Add the resolved key to a JSON object, including a JSON receipt payload.
pub(crate) fn add_idempotency_key(output: &mut Map<String, Value>, key: &str) {
    output.insert("idempotency_key".into(), Value::String(key.into()));
}

/// Print the resolved key to stderr (or another caller-supplied diagnostic writer).
pub(crate) fn write_idempotency_key(writer: &mut impl Write, key: &str) -> io::Result<()> {
    writeln!(writer, "idempotency-key: {key}")
}

/// Serialize a receipt as pretty JSON followed by a newline.
pub(crate) fn write_receipt<T: Serialize>(writer: &mut impl Write, receipt: &T) -> io::Result<()> {
    let json = serde_json::to_vec_pretty(receipt)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    writer.write_all(&json)?;
    writer.write_all(b"\n")
}

#[cfg(test)]
mod tests {
    use super::{
        add_idempotency_key, error_exit_code, resolve_idempotency_key, write_idempotency_key,
        write_receipt,
    };
    use serde_json::{json, Map, Value};

    const ERROR_CODES: [(&str, Option<u8>); 23] = [
        ("INVALID_INPUT", Some(2)),
        ("DATASET_INVALID", Some(2)),
        ("TASK_AMBIGUOUS", Some(2)),
        ("OBJECTIVE_INVALID", Some(2)),
        ("OBJECTIVE_REQUIRES_ORDER", Some(2)),
        ("LANGUAGE_INTERSECTION_EMPTY", Some(2)),
        ("CAPABILITY_UNAVAILABLE", Some(2)),
        ("NOT_FOUND", Some(2)),
        ("IMMUTABLE_FIELD", Some(8)),
        ("REVISION_CONFLICT", Some(8)),
        ("IDEMPOTENCY_CONFLICT", Some(8)),
        ("PLAN_CHANGED", Some(8)),
        ("MODEL_INTEGRITY_FAILED", Some(3)),
        ("BACKEND_UNAVAILABLE", Some(4)),
        ("NO_VALID_MODEL", Some(5)),
        ("NUMERICAL_FAILURE", Some(6)),
        ("IO_FAILURE", Some(7)),
        ("OUTPUT_EXISTS", Some(7)),
        ("DISK_LOW", Some(7)),
        ("UNAUTHORIZED", Some(9)),
        ("ENGINE_UNREACHABLE", Some(9)),
        ("ENGINE_BUSY", Some(9)),
        ("FITNESS_UNDEFINED", None),
    ];

    #[test]
    fn every_error_code_uses_its_catalogued_exit_code() {
        assert_eq!(ERROR_CODES.len(), 23);
        for (error_code, expected) in ERROR_CODES {
            assert_eq!(error_exit_code(error_code), expected, "{error_code}");
        }
        assert_eq!(error_exit_code("UNKNOWN"), None);
    }

    #[test]
    fn absent_key_is_a_cli_ulid_and_provided_key_is_preserved() {
        let generated = resolve_idempotency_key(None);
        assert!(generated.starts_with("cli_"));
        assert_eq!(generated.len(), 30);
        assert!(generated.is_ascii());
        assert_eq!(resolve_idempotency_key(Some("caller-key")), "caller-key");
    }

    #[test]
    fn generated_key_is_added_to_json_and_stderr_output() {
        let generated = resolve_idempotency_key(None);
        let mut output = Map::new();
        output.insert("status".into(), Value::String("ok".into()));
        add_idempotency_key(&mut output, &generated);
        let json = serde_json::to_string(&output).unwrap();
        assert!(json.contains(&generated));

        let mut stderr = Vec::new();
        write_idempotency_key(&mut stderr, &generated).unwrap();
        let stderr = String::from_utf8(stderr).unwrap();
        assert!(stderr.contains(&generated));
    }

    #[test]
    fn receipt_helper_writes_json_with_a_trailing_newline() {
        let receipt = json!({"command":"fit","outcome":"succeeded"});
        let mut output = Vec::new();
        write_receipt(&mut output, &receipt).unwrap();
        assert!(output.ends_with(b"\n"));
        assert_eq!(serde_json::from_slice::<Value>(&output).unwrap(), receipt);
    }
}

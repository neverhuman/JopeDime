//! Shared batch-command helpers.
#![allow(dead_code)] // Command handlers consume these shared helpers in their own task scopes.

#[cfg(all(feature = "client", not(feature = "engine")))]
use serde::Deserialize;
use serde::Serialize;
use serde_json::{Map, Value};
use std::fmt::Display;
#[cfg(feature = "client")]
use std::io::Read;
use std::io::{self, Write};
#[cfg(feature = "client")]
use std::path::Path;
use ulid::Ulid;

#[cfg(feature = "client")]
/// Resolve the running local service from the owner-only daemon record.
pub(crate) fn resolve_client() -> anyhow::Result<jope_client::Http> {
    resolve_client_at(&jope_paths::jopedime_home())
}

#[cfg(feature = "client")]
fn resolve_client_at(home: &Path) -> anyhow::Result<jope_client::Http> {
    #[cfg(feature = "engine")]
    let (url, token) = {
        use jope_service::daemon_file::DaemonFileStatus;
        match jope_service::daemon_file::read_daemon_file(home) {
            Ok(DaemonFileStatus::Running(record) | DaemonFileStatus::Unknown(record)) => {
                (record.url, record.token)
            }
            Ok(DaemonFileStatus::Stale(_)) | Err(_) => {
                return Err(CommandError::unreachable("engine is not running").into());
            }
        }
    };
    #[cfg(not(feature = "engine"))]
    let (url, token) = {
        #[derive(Deserialize)]
        struct DaemonFile {
            url: String,
            token: String,
        }
        let bytes = std::fs::read(home.join("daemon.json"))
            .map_err(|_| CommandError::unreachable("engine is not running"))?;
        let record: DaemonFile = serde_json::from_slice(&bytes)
            .map_err(|_| CommandError::unreachable("engine is not running"))?;
        (record.url, record.token)
    };
    jope_client::Http::new(&url, &token)
        .map_err(|_| CommandError::unreachable("engine is not running").into())
}

#[cfg(feature = "client")]
#[derive(Debug)]
struct CommandError {
    code: u8,
    message: &'static str,
}

#[cfg(feature = "client")]
impl std::fmt::Display for CommandError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.message)
    }
}

#[cfg(feature = "client")]
impl std::error::Error for CommandError {}

#[cfg(feature = "client")]
impl CommandError {
    fn unreachable(message: &'static str) -> Self {
        Self { code: 9, message }
    }
}

/// Return an existing API process code for typed service errors or resolver failures.
pub(crate) fn report_exit_code(error: &anyhow::Error) -> Option<u8> {
    #[cfg(feature = "client")]
    {
        error.chain().find_map(|cause| {
            if let Some(command) = cause.downcast_ref::<CommandError>() {
                return Some(command.code);
            }
            if let Some(client) = cause.downcast_ref::<jope_client::Error>() {
                return match client {
                    jope_client::Error::Problem(problem) => error_exit_code(problem.code),
                    jope_client::Error::Http(_) => Some(9),
                    _ => None,
                };
            }
            None
        })
    }
    #[cfg(not(feature = "client"))]
    {
        let _ = error;
        None
    }
}

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

pub(crate) fn announce_generated_key(supplied: Option<&str>, key: &str) -> anyhow::Result<()> {
    if supplied.is_none() {
        write_idempotency_key(&mut io::stderr().lock(), key)
            .map_err(|error| jope_runtime::Failure::output(error.to_string()))?;
    }
    Ok(())
}

/// Serialize a receipt as pretty JSON followed by a newline.
pub(crate) fn write_receipt<T: Serialize>(writer: &mut impl Write, receipt: &T) -> io::Result<()> {
    let json = serde_json::to_vec_pretty(receipt)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    writer.write_all(&json)?;
    writer.write_all(b"\n")
}

pub(crate) fn print_json<T: Serialize>(value: &T) -> anyhow::Result<()> {
    write_receipt(&mut io::stdout().lock(), value)
        .map_err(|error| jope_runtime::Failure::output(error.to_string()))
}

/// Read an optional raw API JSON body from a named file or stdin (`-`).
#[cfg(feature = "client")]
pub(crate) fn read_json_body(path: Option<&str>) -> anyhow::Result<Option<Value>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let mut bytes = Vec::new();
    if path == "-" {
        io::stdin()
            .take((jope_client::transport::MAX_JSON_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|error| jope_runtime::Failure::input(error.to_string()))?;
    } else {
        std::fs::File::open(path)
            .map_err(|error| jope_runtime::Failure::input(error.to_string()))?
            .take((jope_client::transport::MAX_JSON_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|error| jope_runtime::Failure::input(error.to_string()))?;
    }
    if bytes.len() > jope_client::transport::MAX_JSON_BYTES {
        return Err(jope_runtime::Failure::input("request body exceeds 8 MiB"));
    }
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| jope_runtime::Failure::input(error.to_string()))?;
    if !value.is_object() {
        return Err(jope_runtime::Failure::input(
            "request body must be a JSON object",
        ));
    }
    Ok(Some(value))
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

    #[cfg(feature = "client")]
    #[test]
    fn daemon_record_resolves_a_client_and_missing_or_stale_is_unreachable() {
        let home = tempfile::tempdir().unwrap();
        #[cfg(feature = "engine")]
        jope_service::daemon_file::write_daemon_file(
            home.path(),
            "http://127.0.0.1:43123",
            "test",
            "1.0",
            "now",
        )
        .unwrap();
        #[cfg(not(feature = "engine"))]
        std::fs::write(
            home.path().join("daemon.json"),
            serde_json::to_vec(&json!({
                "url":"http://127.0.0.1:43123",
                "token":"ab".repeat(32),
                "pid":1,
                "version":"test",
                "api_version":"1.0",
                "started_at":"now"
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(super::resolve_client_at(home.path()).is_ok());

        let missing = tempfile::tempdir().unwrap();
        let error = super::resolve_client_at(missing.path()).unwrap_err();
        assert_eq!(super::report_exit_code(&error), Some(9));
        assert_eq!(error.to_string(), "engine is not running");
    }

    #[cfg(all(feature = "client", feature = "engine"))]
    #[test]
    fn stale_daemon_record_uses_existing_unreachable_code() {
        let home = tempfile::tempdir().unwrap();
        let record = jope_service::daemon_file::DaemonFile {
            url: "http://127.0.0.1:43123".into(),
            token: "ab".repeat(32),
            pid: u32::MAX,
            version: "test".into(),
            api_version: "1.0".into(),
            started_at: "now".into(),
        };
        std::fs::write(
            home.path().join("daemon.json"),
            serde_json::to_vec(&record).unwrap(),
        )
        .unwrap();
        let error = super::resolve_client_at(home.path()).unwrap_err();
        assert_eq!(super::report_exit_code(&error), Some(9));
    }

    #[cfg(feature = "client")]
    #[test]
    fn api_problem_uses_catalogued_conflict_and_not_found_codes() {
        use jope_api::error::{ErrorCode, Problem};

        for (code, status, expected) in [
            (ErrorCode::RevisionConflict, 409, 8),
            (ErrorCode::NotFound, 404, 2),
        ] {
            let problem = Problem {
                r#type: "about:blank".into(),
                title: "request failed".into(),
                status,
                detail: "golden detail".into(),
                instance: None,
                code,
                retriable: false,
                remedy: "none".into(),
                field: None,
            };
            let error: anyhow::Error = jope_client::Error::Problem(Box::new(problem)).into();
            assert_eq!(super::report_exit_code(&error), Some(expected));
        }
    }
}

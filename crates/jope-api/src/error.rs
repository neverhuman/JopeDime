//! Appendix D error catalog and the JopeDime v1 problem document (§4.1).
//!
//! HTTP adapters send [`Problem::CONTENT_TYPE`](crate::error::Problem::CONTENT_TYPE)
//! and use the same status in the response and document. Batch clients use
//! [`ErrorCode::exit_code`](crate::error::ErrorCode::exit_code).
//! This module contains data only; transport and occurrence-specific retry
//! decisions belong to the caller.

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};

/// Preserve Appendix D's conditions and uncertainty instead of treating them
/// as an unconditional retry instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetryPolicy {
    Never,
    Always,
    AfterRereading,
    AfterReplanning,
    AfterFreeingSpace,
    Maybe,
    NotApplicable,
}

impl RetryPolicy {
    /// The exact Appendix D cell, including its condition or missing value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Never => "no",
            Self::Always => "yes",
            Self::AfterRereading => "yes, after re-reading",
            Self::AfterReplanning => "yes, after re-planning",
            Self::AfterFreeingSpace => "after freeing space",
            Self::Maybe => "maybe",
            Self::NotApplicable => "—",
        }
    }

    /// Whether retry is permitted, **after** satisfying the policy's condition.
    /// `None` retains `maybe`/not-applicable, rather than inventing a false flag.
    pub const fn retriable(self) -> Option<bool> {
        match self {
            Self::Never => Some(false),
            Self::Always
            | Self::AfterRereading
            | Self::AfterReplanning
            | Self::AfterFreeingSpace => Some(true),
            Self::Maybe | Self::NotApplicable => None,
        }
    }
}

macro_rules! error_catalog {
    ($($variant:ident => ($wire:literal, $http:expr, $exit:expr, $retry:ident)),+ $(,)?) => {
        /// Every Appendix D code. There is no arbitrary-string/fallback variant.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, JsonSchema)]
        pub enum ErrorCode {
            $(#[serde(rename = $wire)] $variant),+
        }

        impl ErrorCode {
            /// Catalog order follows Appendix D, including the event-only row.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            pub const fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $wire),+ }
            }

            /// `NO_VALID_MODEL`'s 200 describes its receipt, not an error status.
            /// CLI-only and event-only rows have no HTTP status.
            pub const fn http_status(self) -> Option<u16> {
                match self { $(Self::$variant => $http),+ }
            }

            /// Event-only FITNESS_UNDEFINED has no process exit code.
            pub const fn exit_code(self) -> Option<u8> {
                match self { $(Self::$variant => $exit),+ }
            }

            pub const fn retry_policy(self) -> RetryPolicy {
                match self { $(Self::$variant => RetryPolicy::$retry),+ }
            }

            pub const fn retriable(self) -> Option<bool> {
                self.retry_policy().retriable()
            }
        }

        impl<'de> Deserialize<'de> for ErrorCode {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                // A code is a JSON string, never an externally tagged object.
                let code = String::deserialize(deserializer)?;
                match code.as_str() {
                    $($wire => Ok(Self::$variant)),+,
                    _ => Err(serde::de::Error::unknown_variant(&code, &[$($wire),+])),
                }
            }
        }
    };
}

error_catalog! {
    InvalidInput => ("INVALID_INPUT", Some(400), Some(2), Never),
    DatasetInvalid => ("DATASET_INVALID", Some(422), Some(2), Never),
    TaskAmbiguous => ("TASK_AMBIGUOUS", Some(422), Some(2), Never),
    ObjectiveInvalid => ("OBJECTIVE_INVALID", Some(422), Some(2), Never),
    ObjectiveRequiresOrder => ("OBJECTIVE_REQUIRES_ORDER", Some(422), Some(2), Never),
    LanguageIntersectionEmpty => ("LANGUAGE_INTERSECTION_EMPTY", Some(422), Some(2), Never),
    CapabilityUnavailable => ("CAPABILITY_UNAVAILABLE", Some(422), Some(2), Never),
    NotFound => ("NOT_FOUND", Some(404), Some(2), Never),
    ImmutableField => ("IMMUTABLE_FIELD", Some(409), Some(8), Never),
    RevisionConflict => ("REVISION_CONFLICT", Some(409), Some(8), AfterRereading),
    IdempotencyConflict => ("IDEMPOTENCY_CONFLICT", Some(409), Some(8), Never),
    PlanChanged => ("PLAN_CHANGED", Some(409), Some(8), AfterReplanning),
    ModelIntegrityFailed => ("MODEL_INTEGRITY_FAILED", Some(500), Some(3), Never),
    BackendUnavailable => ("BACKEND_UNAVAILABLE", Some(503), Some(4), Never),
    NoValidModel => ("NO_VALID_MODEL", Some(200), Some(5), Never),
    NumericalFailure => ("NUMERICAL_FAILURE", Some(500), Some(6), Never),
    IoFailure => ("IO_FAILURE", Some(500), Some(7), Maybe),
    OutputExists => ("OUTPUT_EXISTS", None, Some(7), Never),
    DiskLow => ("DISK_LOW", Some(507), Some(7), AfterFreeingSpace),
    Unauthorized => ("UNAUTHORIZED", Some(401), Some(9), Never),
    EngineUnreachable => ("ENGINE_UNREACHABLE", None, Some(9), Always),
    EngineBusy => ("ENGINE_BUSY", None, Some(9), Always),
    FitnessUndefined => ("FITNESS_UNDEFINED", None, None, NotApplicable),
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// JopeDime's [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457.html) profile.
///
/// Serialize with **`Content-Type: application/problem+json`**, not the success
/// response media type. Set the actual HTTP response status to [`Self::status`].
/// The spec requires `type`, `title`, `status`, `detail` and the three required
/// extensions; `instance` and `field` are optional and omitted when absent.
/// Unknown extension members are ignored when reading the document.
///
/// `retriable` describes this occurrence. For IO_FAILURE the producer decides it
/// from the actual failure; for conditional policies it must also provide the
/// required action through `remedy`. Catalog `maybe` is not silently made false.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Problem {
    pub r#type: String,
    pub title: String,
    pub status: u16,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<String>,
    pub code: ErrorCode,
    pub retriable: bool,
    pub remedy: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
}

impl Problem {
    pub const CONTENT_TYPE: &'static str = "application/problem+json";
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    type CatalogRow = (
        &'static str,
        Option<u16>,
        Option<u8>,
        &'static str,
        Option<bool>,
    );

    // Independent transcription of every Appendix D row, including qualifiers.
    const EXPECTED: &[CatalogRow] = &[
        ("INVALID_INPUT", Some(400), Some(2), "no", Some(false)),
        ("DATASET_INVALID", Some(422), Some(2), "no", Some(false)),
        ("TASK_AMBIGUOUS", Some(422), Some(2), "no", Some(false)),
        ("OBJECTIVE_INVALID", Some(422), Some(2), "no", Some(false)),
        (
            "OBJECTIVE_REQUIRES_ORDER",
            Some(422),
            Some(2),
            "no",
            Some(false),
        ),
        (
            "LANGUAGE_INTERSECTION_EMPTY",
            Some(422),
            Some(2),
            "no",
            Some(false),
        ),
        (
            "CAPABILITY_UNAVAILABLE",
            Some(422),
            Some(2),
            "no",
            Some(false),
        ),
        ("NOT_FOUND", Some(404), Some(2), "no", Some(false)),
        ("IMMUTABLE_FIELD", Some(409), Some(8), "no", Some(false)),
        (
            "REVISION_CONFLICT",
            Some(409),
            Some(8),
            "yes, after re-reading",
            Some(true),
        ),
        (
            "IDEMPOTENCY_CONFLICT",
            Some(409),
            Some(8),
            "no",
            Some(false),
        ),
        (
            "PLAN_CHANGED",
            Some(409),
            Some(8),
            "yes, after re-planning",
            Some(true),
        ),
        (
            "MODEL_INTEGRITY_FAILED",
            Some(500),
            Some(3),
            "no",
            Some(false),
        ),
        ("BACKEND_UNAVAILABLE", Some(503), Some(4), "no", Some(false)),
        ("NO_VALID_MODEL", Some(200), Some(5), "no", Some(false)),
        ("NUMERICAL_FAILURE", Some(500), Some(6), "no", Some(false)),
        ("IO_FAILURE", Some(500), Some(7), "maybe", None),
        ("OUTPUT_EXISTS", None, Some(7), "no", Some(false)),
        (
            "DISK_LOW",
            Some(507),
            Some(7),
            "after freeing space",
            Some(true),
        ),
        ("UNAUTHORIZED", Some(401), Some(9), "no", Some(false)),
        ("ENGINE_UNREACHABLE", None, Some(9), "yes", Some(true)),
        ("ENGINE_BUSY", None, Some(9), "yes", Some(true)),
        ("FITNESS_UNDEFINED", None, None, "—", None),
    ];

    #[test]
    fn every_appendix_d_row_matches_exactly_and_round_trips() {
        assert_eq!(EXPECTED.len(), 23);
        assert_eq!(ErrorCode::ALL.len(), EXPECTED.len());
        let mut seen = std::collections::HashSet::new();
        for (code, &(wire, http, exit, retry, flag)) in ErrorCode::ALL.iter().zip(EXPECTED) {
            assert!(seen.insert(code.as_str()));
            assert_eq!(code.as_str(), wire);
            assert_eq!(code.to_string(), wire);
            assert_eq!(code.http_status(), http);
            assert_eq!(code.exit_code(), exit);
            assert_eq!(code.retry_policy().as_str(), retry);
            assert_eq!(code.retriable(), flag);
            assert_eq!(serde_json::to_value(code).unwrap(), json!(wire));
            assert_eq!(
                serde_json::from_value::<ErrorCode>(json!(wire)).unwrap(),
                *code
            );
        }
    }

    #[test]
    fn unknown_codes_and_non_string_representations_are_rejected() {
        for value in [
            json!("UNKNOWN"),
            json!("invalid_input"),
            json!(" INVALID_INPUT"),
            json!("INVALID_INPUT "),
            json!(""),
            json!(400),
            json!(null),
            json!(true),
            json!({"INVALID_INPUT":null}),
            json!(["INVALID_INPUT"]),
        ] {
            assert!(serde_json::from_value::<ErrorCode>(value).is_err());
        }
    }

    fn problem() -> Problem {
        Problem {
            r#type: "urn:jopedime:problem:INVALID_INPUT".into(),
            title: "Invalid input".into(),
            status: 400,
            detail: "Target is missing".into(),
            instance: None,
            code: ErrorCode::InvalidInput,
            retriable: false,
            remedy: "Set --target to an existing column".into(),
            field: Some("target".into()),
        }
    }

    #[test]
    fn problem_serializes_standard_members_extensions_and_media_type() {
        assert_eq!(Problem::CONTENT_TYPE, "application/problem+json");
        let value = json!({"type":"urn:jopedime:problem:INVALID_INPUT", "title":"Invalid input",
            "status":400, "detail":"Target is missing", "code":"INVALID_INPUT",
            "retriable":false, "remedy":"Set --target to an existing column", "field":"target"});
        assert_eq!(serde_json::to_value(problem()).unwrap(), value);
        assert_eq!(serde_json::from_value::<Problem>(value).unwrap(), problem());
    }

    #[test]
    fn optional_instance_field_and_future_extensions_follow_the_rfc_profile() {
        let mut value = problem();
        value.instance = Some("/v1/requests/7".into());
        value.field = None;
        let mut encoded = serde_json::to_value(&value).unwrap();
        assert_eq!(encoded["instance"], "/v1/requests/7");
        assert!(encoded.get("field").is_none());
        encoded["future_extension"] = json!({"ignored":true});
        assert_eq!(serde_json::from_value::<Problem>(encoded).unwrap(), value);
    }

    #[test]
    fn problem_rejects_unknown_code_and_missing_required_extensions() {
        let value = serde_json::to_value(problem()).unwrap();
        let mut unknown = value.clone();
        unknown["code"] = json!("FUTURE_ERROR");
        assert!(serde_json::from_value::<Problem>(unknown).is_err());
        for name in [
            "type",
            "title",
            "status",
            "detail",
            "code",
            "retriable",
            "remedy",
        ] {
            let mut missing = value.clone();
            missing.as_object_mut().unwrap().remove(name);
            assert!(
                serde_json::from_value::<Problem>(missing).is_err(),
                "{name}"
            );
        }
    }

    #[test]
    fn error_schema_is_a_closed_string_catalog_and_problem_exposes_the_profile() {
        let schema = serde_json::to_value(schemars::schema_for!(ErrorCode)).unwrap();
        assert_eq!(schema["type"], "string");
        let expected: Vec<Value> = EXPECTED.iter().map(|row| json!(row.0)).collect();
        assert_eq!(schema["enum"], Value::Array(expected));
        let problem = serde_json::to_value(schemars::schema_for!(Problem)).unwrap();
        let required = problem["required"].as_array().unwrap();
        for name in [
            "type",
            "title",
            "status",
            "detail",
            "code",
            "retriable",
            "remedy",
        ] {
            assert!(required.contains(&json!(name)), "{name}");
        }
        assert!(!required.contains(&json!("field")));
        assert!(!required.contains(&json!("instance")));
    }
}

//! Stable request identities and stored JSON outcomes (spec §4.1, §5.3).
//! Callers supply an authenticated principal identifier, never the bearer secret.

use crate::store::{Result as StoreResult, StoreError};
use axum::{
    Json,
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use jope_api::{
    error::{ErrorCode, Problem},
    registry::OPERATIONS,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug)]
pub struct RequestIdentity {
    pub(crate) principal: String,
    pub(crate) key: String,
    pub(crate) operation_id: u8,
    pub(crate) digest: [u8; 32],
}

impl RequestIdentity {
    /// Bind the operation/method, actual path+query, precondition and JSON body.
    /// Uploaded content must be represented by its verified content digest in body.
    pub fn new(
        principal: impl Into<String>,
        key: impl Into<String>,
        operation_id: u8,
        target: &str,
        if_match: Option<&str>,
        body: &Value,
    ) -> Result<Self, IdempotencyError> {
        let principal = principal.into();
        let key = key.into();
        if principal.is_empty() || u32::try_from(principal.len()).is_err() {
            return Err(IdempotencyError::InvalidRequest(
                "invalid authenticated principal",
            ));
        }
        if !(1..=128).contains(&key.len()) || !key.bytes().all(|b| (32..=126).contains(&b)) {
            return Err(IdempotencyError::InvalidRequest(
                "Idempotency-Key must be 1–128 printable ASCII bytes",
            ));
        }
        let operation = OPERATIONS
            .iter()
            .find(|op| op.id == operation_id && op.requires_idempotency_key)
            .ok_or(IdempotencyError::InvalidRequest(
                "operation does not require an idempotency key",
            ))?;
        if target.is_empty() {
            return Err(IdempotencyError::InvalidRequest("request target is empty"));
        }
        let canonical = Value::Array(vec![
            Value::from(operation_id),
            Value::from(operation.method.as_str()),
            Value::from(target),
            if_match.map(Value::from).unwrap_or(Value::Null),
            canonical_json(body),
        ]);
        let bytes = serde_json::to_vec(&canonical).map_err(StoreError::from)?;
        // The HTTP body limit excludes identity metadata (URI/precondition/operation).
        let mut hash = Sha256::new();
        hash.update(b"jope.command-request/v1\0");
        hash.update(bytes);
        Ok(Self {
            principal,
            key,
            operation_id,
            digest: hash.finalize().into(),
        })
    }
}

fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut pairs: Vec<_> = map.iter().collect();
            pairs.sort_unstable_by(|a, b| a.0.cmp(b.0));
            Value::Object(
                pairs
                    .into_iter()
                    .map(|(k, v)| (k.clone(), canonical_json(v)))
                    .collect(),
            )
        }
        Value::Array(values) => Value::Array(values.iter().map(canonical_json).collect()),
        other => other.clone(),
    }
}

/// Durable response status/body. Transport version and JSON headers are fixed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandOutcome {
    status: u16,
    body: Value,
}
impl CommandOutcome {
    pub fn json(status: u16, body: Value) -> StoreResult<Self> {
        let outcome = Self { status, body };
        outcome.validate()?;
        Ok(outcome)
    }
    pub fn status(&self) -> u16 {
        self.status
    }
    pub fn body(&self) -> &Value {
        &self.body
    }
    pub(crate) fn validate(&self) -> StoreResult<()> {
        if !(200..=599).contains(&self.status)
            || (matches!(self.status, 204 | 205 | 304) && !self.body.is_null())
        {
            return Err(StoreError::Integrity(
                "invalid stored command response status",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CommandReceipt {
    pub operation_id: u8,
    pub outcome: CommandOutcome,
    /// Post-commit integrations must not schedule work again for a replay.
    pub replayed: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StoredCommand {
    pub request_digest: [u8; 32],
    pub operation_id: u8,
    pub outcome: CommandOutcome,
}

#[derive(Debug)]
pub enum IdempotencyError {
    InvalidRequest(&'static str),
    Conflict,
    Store(StoreError),
}
impl From<StoreError> for IdempotencyError {
    fn from(error: StoreError) -> Self {
        Self::Store(error)
    }
}
impl std::fmt::Display for IdempotencyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(reason) => write!(f, "invalid command request: {reason}"),
            Self::Conflict => write!(f, "IDEMPOTENCY_CONFLICT"),
            Self::Store(error) => write!(f, "command storage: {error}"),
        }
    }
}
impl std::error::Error for IdempotencyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(error) => Some(error),
            _ => None,
        }
    }
}
fn version(mut response: Response) -> Response {
    response
        .headers_mut()
        .insert("jope-api-version", HeaderValue::from_static("1.0"));
    response
}
impl IntoResponse for CommandOutcome {
    fn into_response(self) -> Response {
        if let Err(error) = self.validate() {
            return IdempotencyError::Store(error).into_response();
        }
        let status = StatusCode::from_u16(self.status).expect("validated status");
        if matches!(self.status, 204 | 205 | 304) {
            return version(status.into_response());
        }
        version((status, Json(self.body)).into_response())
    }
}
impl IntoResponse for CommandReceipt {
    fn into_response(self) -> Response {
        self.outcome.into_response()
    }
}
impl IntoResponse for IdempotencyError {
    fn into_response(self) -> Response {
        let (code, detail, remedy, retriable) = match self {
            Self::Conflict => (
                ErrorCode::IdempotencyConflict,
                "Idempotency-Key was already used for a different request.",
                "Use a new key for a different request.",
                false,
            ),
            Self::InvalidRequest(reason) => (
                ErrorCode::InvalidInput,
                reason,
                "Provide a valid authenticated principal, operation and Idempotency-Key.",
                false,
            ),
            Self::Store(_) => (
                ErrorCode::IoFailure,
                "The command's committed outcome could not be confirmed.",
                "Read back/reopen the store, then retry the same request with the same key.",
                true,
            ),
        };
        let status = StatusCode::from_u16(code.http_status().expect("HTTP catalog code"))
            .expect("valid catalog status");
        let document = Problem {
            r#type: "about:blank".into(),
            title: status.canonical_reason().unwrap_or("Request failed").into(),
            status: status.as_u16(),
            detail: detail.into(),
            instance: None,
            code,
            retriable,
            remedy: remedy.into(),
            field: None,
        };
        let mut response = (status, Json(document)).into_response();
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static(Problem::CONTENT_TYPE),
        );
        version(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_digest_binds_semantics_but_not_object_order() {
        let a: Value = serde_json::from_str(r#"{"z":[1,2],"a":{"y":2,"x":1}}"#).unwrap();
        let b: Value = serde_json::from_str(r#"{"a":{"x":1,"y":2},"z":[1,2]}"#).unwrap();
        let make = |operation, target, condition, body: &Value| {
            RequestIdentity::new("owner", "key", operation, target, condition, body).unwrap()
        };
        let first = make(6, "/v1/runs", Some("\"1\""), &a);
        assert_eq!(first.digest, make(6, "/v1/runs", Some("\"1\""), &b).digest);
        for changed in [
            make(8, "/v1/runs", Some("\"1\""), &a),
            make(6, "/v1/runs?dry_run=true", Some("\"1\""), &a),
            make(6, "/v1/runs", Some("\"2\""), &a),
            make(6, "/v1/runs", None, &a),
            make(6, "/v1/runs", Some("\"1\""), &json!({"z":[2,1]})),
        ] {
            assert_ne!(first.digest, changed.digest);
        }
    }

    #[test]
    fn maximum_json_body_is_not_charged_for_request_identity_metadata() {
        let body = Value::String("x".repeat(8 * 1024 * 1024 - 2));
        assert_eq!(serde_json::to_vec(&body).unwrap().len(), 8 * 1024 * 1024);
        assert!(RequestIdentity::new("owner", "key", 6, "/v1/runs", Some("\"1\""), &body).is_ok());
    }

    #[tokio::test]
    async fn bodyless_statuses_replay_empty_instead_of_serialized_json_null() {
        for status in [204, 205, 304] {
            assert!(CommandOutcome::json(status, json!({"unexpected":"body"})).is_err());
            let response = CommandOutcome::json(status, Value::Null)
                .unwrap()
                .into_response();
            assert_eq!(response.status().as_u16(), status);
            assert!(response.headers().get("content-type").is_none());
            assert_eq!(response.headers()["jope-api-version"], "1.0");
            assert!(
                axum::body::to_bytes(response.into_body(), 1024)
                    .await
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[test]
    fn invalid_keys_and_read_operations_are_refused() {
        for key in ["".to_owned(), "x".repeat(129), "a\nb".into(), "é".into()] {
            assert!(matches!(
                RequestIdentity::new("owner", key, 6, "/v1/runs", None, &json!({})),
                Err(IdempotencyError::InvalidRequest(_))
            ));
        }
        assert!(RequestIdentity::new("", "key", 6, "/v1/runs", None, &json!({})).is_err());
        assert!(RequestIdentity::new("owner", "key", 1, "/v1/system", None, &json!({})).is_err());
        assert!(CommandOutcome::json(199, json!({})).is_err());
        assert!(CommandOutcome::json(600, json!({})).is_err());
    }
}

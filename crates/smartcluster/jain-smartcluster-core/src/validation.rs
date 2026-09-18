use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A validation failure with a stable field path suitable for API responses.
#[derive(Clone, Debug, Deserialize, Eq, Error, PartialEq, Serialize)]
#[error("{field}: {message}")]
pub struct ValidationError {
    pub field: String,
    pub message: String,
}

impl ValidationError {
    pub fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }

    pub(crate) fn prefixed(self, prefix: &str) -> Self {
        Self {
            field: format!("{prefix}.{}", self.field),
            message: self.message,
        }
    }
}

/// Performs semantic validation after decoding an untrusted value.
pub trait Validate {
    fn validate(&self) -> Result<(), ValidationError>;
}

pub(crate) fn validate_identifier(
    field: &str,
    value: &str,
    maximum_bytes: usize,
) -> Result<(), ValidationError> {
    if value.is_empty() {
        return Err(ValidationError::new(field, "must not be empty"));
    }
    if value.len() > maximum_bytes {
        return Err(ValidationError::new(
            field,
            format!("must be at most {maximum_bytes} bytes"),
        ));
    }
    if !value.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'@' | b'/')
    }) {
        return Err(ValidationError::new(
            field,
            "contains characters outside [A-Za-z0-9._-:@/]",
        ));
    }
    Ok(())
}

pub(crate) fn validate_text(
    field: &str,
    value: &str,
    maximum_bytes: usize,
    allow_empty: bool,
) -> Result<(), ValidationError> {
    if !allow_empty && value.is_empty() {
        return Err(ValidationError::new(field, "must not be empty"));
    }
    if value.len() > maximum_bytes {
        return Err(ValidationError::new(
            field,
            format!("must be at most {maximum_bytes} bytes"),
        ));
    }
    if value.contains('\0') {
        return Err(ValidationError::new(field, "must not contain NUL"));
    }
    Ok(())
}

use super::*;

#[derive(Clone, Debug)]
pub(crate) enum ReadableGeneration0 {
    // Retain the typed payload used by the compatibility artifact validator.
    Legacy(
        #[allow(
            dead_code,
            reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
        )]
        Value,
    ),
    // Retain the typed payload used by the compatibility artifact validator.
    V3(
        #[allow(
            dead_code,
            reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
        )]
        Generation0ReceiptV3,
    ),
}

pub(crate) fn read_generation_zero(bytes: &[u8]) -> Result<ReadableGeneration0> {
    let value: Value = serde_json::from_slice(bytes)?;
    match value.get("schema").and_then(Value::as_str) {
        Some(GENERATION_ZERO_V3_SCHEMA) => {
            Ok(ReadableGeneration0::V3(serde_json::from_value(value)?))
        }
        Some(schema) if schema.starts_with("Generation0Receipt.v") => {
            Ok(ReadableGeneration0::Legacy(value))
        }
        _ => bail!("unknown generation-zero receipt schema"),
    }
}

#[derive(Clone, Debug)]
// Preserve the existing compatibility receipt payload and allocation layout.
#[allow(clippy::large_enum_variant)]
pub(crate) enum ReadableAdmission {
    // Retain the typed payload used by the compatibility artifact validator.
    Legacy(
        #[allow(
            dead_code,
            reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
        )]
        Value,
    ),
    // Retain the typed payload used by the compatibility artifact validator.
    V3(
        #[allow(
            dead_code,
            reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
        )]
        AttemptAdmissionManifestV3,
    ),
}

pub(crate) fn read_admission(bytes: &[u8]) -> Result<ReadableAdmission> {
    let value: Value = serde_json::from_slice(bytes)?;
    match value.get("schema").and_then(Value::as_str) {
        Some(ADMISSION_V3_SCHEMA) => Ok(ReadableAdmission::V3(serde_json::from_value(value)?)),
        Some(schema) if schema.starts_with("AttemptAdmissionManifest.v") => {
            Ok(ReadableAdmission::Legacy(value))
        }
        _ => bail!("unknown attempt-admission schema"),
    }
}

#[derive(Clone, Debug)]
// Preserve the existing compatibility receipt payload and allocation layout.
#[allow(clippy::large_enum_variant)]
pub(crate) enum ReadableTelemetry {
    // Retain the typed payload used by the compatibility artifact validator.
    Legacy(
        #[allow(
            dead_code,
            reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
        )]
        Value,
    ),
    // Retain the typed payload used by the compatibility artifact validator.
    V2(
        #[allow(
            dead_code,
            reason = "serde compatibility carrier: typed deserialization in validate_artifact_file enforces the schema; payload intentionally unread"
        )]
        RatchetLiveTelemetryV2,
    ),
}

pub(crate) fn read_telemetry(bytes: &[u8]) -> Result<ReadableTelemetry> {
    let value: Value = serde_json::from_slice(bytes)?;
    match value.get("schema").and_then(Value::as_str) {
        Some("RatchetLiveTelemetry.v2") => {
            Ok(ReadableTelemetry::V2(serde_json::from_value(value)?))
        }
        Some(schema) if schema.starts_with("RatchetLiveTelemetry.v") => {
            Ok(ReadableTelemetry::Legacy(value))
        }
        _ => bail!("unknown ratchet telemetry schema"),
    }
}

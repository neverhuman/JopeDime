//! Additive wire-contract codec, domain-separated identity projection, envelope adapters,
//! and outcome-authority boundary (`jope.canonical-json/v1`, task `I02`).
//!
//! Historical canonicalization and digest schemes (`jope_learning::canon`,
//! `jope_records::pyjson`, `jope.hwm/v1`, `jope.score/v1`, and separator-only
//! `jope.family-partition/v1\0`) remain untouched and read-only.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;

/// Canonical envelope schema identifier for the `I02` additive wire contract.
pub const CANONICAL_JSON_ENVELOPE_SCHEMA: &str = "jope.canonical-json/v1";

/// Normative domain-separation formula adopted for all new `jope.canonical-json/v1` envelopes
/// (`S5.md:1242`, resolving the `S1.md:1548` / `S6.md:1137` separator-only conflict).
pub const DOMAIN_SEPARATED_ID_FORMULA: &str =
    "sha256(schema_utf8||0x00||u64_be(len)||canonical_payload_bytes)";

/// Minimum exact integer value permitted in `jope.canonical-json/v1` (`-(2^53 - 1)`).
pub const SAFE_INT_MIN: i64 = -9_007_199_254_740_991;

/// Maximum exact integer value permitted in `jope.canonical-json/v1` (`+(2^53 - 1)`).
pub const SAFE_INT_MAX: i64 = 9_007_199_254_740_991;

/// Forbidden self-referential or forward-cycle keys inside a canonical payload object.
pub const FORBIDDEN_PAYLOAD_CYCLE_KEYS: &[&str] = &[
    "own_id",
    "signature",
    "later_git_commit",
    "later_assessment_id",
];

/// Known top-level keys permitted in an `EnvelopeFrame` JSON object.
pub const ALLOWED_ENVELOPE_KEYS: &[&str] = &[
    "display",
    "domain",
    "envelope_schema",
    "own_id",
    "payload",
    "schema_id",
    "signature",
];

/// Errors returned by the strict `jope.canonical-json/v1` lexer, canonicalizer,
/// envelope verifier, and outcome-authority boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireContractError {
    ByteOrderMark,
    InvalidUtf8,
    UnexpectedEof,
    TrailingBytes {
        offset: usize,
    },
    SyntaxError {
        offset: usize,
        detail: String,
    },
    DuplicateObjectKey {
        path: String,
        key: String,
    },
    UnescapedControlChar {
        offset: usize,
        byte: u8,
    },
    InvalidStringEscape {
        offset: usize,
        detail: String,
    },
    UnpairedSurrogate {
        offset: usize,
        code_unit: u16,
    },
    NegativeZeroInteger {
        offset: usize,
    },
    LeadingZeroNumber {
        offset: usize,
    },
    RawFloatInIdentityPayload {
        path: String,
    },
    IntegerOverflow {
        path: String,
        raw: String,
    },
    NonCanonicalByteEncoding {
        detail: String,
    },
    MissingRequiredField {
        path: String,
        field: &'static str,
    },
    UnknownField {
        path: String,
        field: String,
    },
    UnknownEnumVariant {
        path: String,
        value: String,
    },
    InvalidSchemaId {
        schema_id: String,
        reason: &'static str,
    },
    SchemaIdMismatch {
        envelope: String,
        payload: String,
    },
    ForbiddenCycleField {
        path: String,
        field: String,
    },
    InvalidHexEncoding {
        path: String,
        value: String,
        expected_len: usize,
    },
    NonFiniteFloatBits {
        path: String,
        hex_bits: String,
    },
    InvalidOutcomeAuthority {
        reason: String,
    },
    EnvelopeIdMismatch {
        expected: String,
        actual: String,
    },
    SignatureMismatch {
        reason: String,
    },
}

impl fmt::Display for WireContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ByteOrderMark => write!(f, "UTF-8 BOM (0xEF 0xBB 0xBF) is forbidden"),
            Self::InvalidUtf8 => write!(f, "input is not valid UTF-8"),
            Self::UnexpectedEof => write!(f, "unexpected end of JSON input"),
            Self::TrailingBytes { offset } => write!(f, "trailing bytes at offset {offset}"),
            Self::SyntaxError { offset, detail } => {
                write!(f, "JSON syntax error at offset {offset}: {detail}")
            }
            Self::DuplicateObjectKey { path, key } => {
                write!(f, "duplicate JSON object key {key:?} at {path}")
            }
            Self::UnescapedControlChar { offset, byte } => {
                write!(f, "unescaped control byte 0x{byte:02x} at offset {offset}")
            }
            Self::InvalidStringEscape { offset, detail } => {
                write!(f, "invalid string escape at offset {offset}: {detail}")
            }
            Self::UnpairedSurrogate { offset, code_unit } => {
                write!(
                    f,
                    "unpaired UTF-16 surrogate 0x{code_unit:04x} at offset {offset}"
                )
            }
            Self::NegativeZeroInteger { offset } => {
                write!(
                    f,
                    "negative zero integer (-0) is forbidden at offset {offset}"
                )
            }
            Self::LeadingZeroNumber { offset } => {
                write!(f, "leading zero in integer is forbidden at offset {offset}")
            }
            Self::RawFloatInIdentityPayload { path } => {
                write!(
                    f,
                    "raw floating-point number at {path} is forbidden in identity payload; use ExactF32/ExactF64 bit tags"
                )
            }
            Self::IntegerOverflow { path, raw } => {
                write!(
                    f,
                    "integer {raw:?} at {path} exceeds exact IEEE-754 safe range [{SAFE_INT_MIN}, {SAFE_INT_MAX}]"
                )
            }
            Self::NonCanonicalByteEncoding { detail } => {
                write!(f, "non-canonical jope.canonical-json/v1 bytes: {detail}")
            }
            Self::MissingRequiredField { path, field } => {
                write!(f, "missing required field {field:?} at {path}")
            }
            Self::UnknownField { path, field } => {
                write!(f, "unknown field {field:?} at {path}")
            }
            Self::UnknownEnumVariant { path, value } => {
                write!(f, "unknown enum variant {value:?} at {path}")
            }
            Self::InvalidSchemaId { schema_id, reason } => {
                write!(f, "invalid schema_id {schema_id:?}: {reason}")
            }
            Self::SchemaIdMismatch { envelope, payload } => {
                write!(
                    f,
                    "envelope schema_id {envelope:?} does not match payload.schema {payload:?}"
                )
            }
            Self::ForbiddenCycleField { path, field } => {
                write!(
                    f,
                    "forbidden cycle/self-reference field {field:?} at {path}"
                )
            }
            Self::InvalidHexEncoding {
                path,
                value,
                expected_len,
            } => {
                write!(
                    f,
                    "expected {expected_len} lowercase hex chars at {path}, got {value:?}"
                )
            }
            Self::NonFiniteFloatBits { path, hex_bits } => {
                write!(
                    f,
                    "non-finite IEEE-754 bit pattern {hex_bits:?} (NaN or +-Inf) is forbidden at {path}"
                )
            }
            Self::InvalidOutcomeAuthority { reason } => {
                write!(f, "invalid outcome authority: {reason}")
            }
            Self::EnvelopeIdMismatch { expected, actual } => {
                write!(
                    f,
                    "domain-separated own_id mismatch: expected {expected}, got {actual}"
                )
            }
            Self::SignatureMismatch { reason } => {
                write!(f, "signature envelope verification failed: {reason}")
            }
        }
    }
}

impl std::error::Error for WireContractError {}

/// Exact IEEE-754 binary32 (`f32`) bit pattern, preserving `+0.0` vs `-0.0` and subnormals
/// while rejecting `NaN` and `±Inf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExactF32 {
    bits: u32,
}

impl ExactF32 {
    /// Construct from raw IEEE-754 `u32` bits; fails closed on `NaN` or `±Inf`.
    pub fn from_bits(bits: u32) -> Result<Self, WireContractError> {
        if (bits & 0x7f80_0000) == 0x7f80_0000 {
            return Err(WireContractError::NonFiniteFloatBits {
                path: "/f32_bits".to_string(),
                hex_bits: format!("{bits:08x}"),
            });
        }
        Ok(Self { bits })
    }

    /// Construct from a finite `f32` value, preserving `-0.0` and subnormal bit patterns.
    pub fn from_f32(value: f32) -> Result<Self, WireContractError> {
        Self::from_bits(value.to_bits())
    }

    /// Parse from an 8-character lowercase hexadecimal string.
    pub fn from_hex(path: &str, hex_str: &str) -> Result<Self, WireContractError> {
        let bits = parse_fixed_lowercase_hex_u32(path, hex_str)?;
        if (bits & 0x7f80_0000) == 0x7f80_0000 {
            return Err(WireContractError::NonFiniteFloatBits {
                path: path.to_string(),
                hex_bits: hex_str.to_string(),
            });
        }
        Ok(Self { bits })
    }

    pub fn bits(self) -> u32 {
        self.bits
    }

    pub fn to_hex(self) -> String {
        format!("{:08x}", self.bits)
    }

    pub fn to_f32(self) -> f32 {
        f32::from_bits(self.bits)
    }

    pub fn is_negative_zero(self) -> bool {
        self.bits == 0x8000_0000
    }

    pub fn is_subnormal(self) -> bool {
        (self.bits & 0x7f80_0000) == 0 && (self.bits & 0x007f_ffff) != 0
    }
}

/// Exact IEEE-754 binary64 (`f64`) bit pattern, preserving `+0.0` vs `-0.0` and subnormals
/// while rejecting `NaN` and `±Inf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExactF64 {
    bits: u64,
}

impl ExactF64 {
    /// Construct from raw IEEE-754 `u64` bits; fails closed on `NaN` or `±Inf`.
    pub fn from_bits(bits: u64) -> Result<Self, WireContractError> {
        if (bits & 0x7ff0_0000_0000_0000) == 0x7ff0_0000_0000_0000 {
            return Err(WireContractError::NonFiniteFloatBits {
                path: "/f64_bits".to_string(),
                hex_bits: format!("{bits:016x}"),
            });
        }
        Ok(Self { bits })
    }

    /// Construct from a finite `f64` value, preserving `-0.0` and subnormal bit patterns.
    pub fn from_f64(value: f64) -> Result<Self, WireContractError> {
        Self::from_bits(value.to_bits())
    }

    /// Parse from a 16-character lowercase hexadecimal string.
    pub fn from_hex(path: &str, hex_str: &str) -> Result<Self, WireContractError> {
        let bits = parse_fixed_lowercase_hex_u64(path, hex_str)?;
        if (bits & 0x7ff0_0000_0000_0000) == 0x7ff0_0000_0000_0000 {
            return Err(WireContractError::NonFiniteFloatBits {
                path: path.to_string(),
                hex_bits: hex_str.to_string(),
            });
        }
        Ok(Self { bits })
    }

    pub fn bits(self) -> u64 {
        self.bits
    }

    pub fn to_hex(self) -> String {
        format!("{:016x}", self.bits)
    }

    pub fn to_f64(self) -> f64 {
        f64::from_bits(self.bits)
    }

    pub fn is_negative_zero(self) -> bool {
        self.bits == 0x8000_0000_0000_0000
    }

    pub fn is_subnormal(self) -> bool {
        (self.bits & 0x7ff0_0000_0000_0000) == 0 && (self.bits & 0x000f_ffff_ffff_ffff) != 0
    }
}

/// Fixed-width 16-hex-character `u64` seed/identity value for integers outside `±(2^53 - 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HexU64(pub u64);

impl HexU64 {
    pub fn from_hex(path: &str, hex_str: &str) -> Result<Self, WireContractError> {
        Ok(Self(parse_fixed_lowercase_hex_u64(path, hex_str)?))
    }

    pub fn to_hex(self) -> String {
        format!("{:016x}", self.0)
    }
}

/// Strict canonical JSON value tree used by `jope.canonical-json/v1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalValue {
    Null,
    Bool(bool),
    Int(i64),
    String(String),
    Array(Vec<CanonicalValue>),
    Object(BTreeMap<String, CanonicalValue>),
}

impl CanonicalValue {
    pub fn as_object(&self) -> Option<&BTreeMap<String, CanonicalValue>> {
        match self {
            Self::Object(map) => Some(map),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Serialize this value into `jope.canonical-json/v1` canonical UTF-8 bytes.
    pub fn to_canonical_string(&self) -> String {
        let mut out = String::new();
        self.write_canonical(&mut out);
        out
    }

    fn write_canonical(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(true) => out.push_str("true"),
            Self::Bool(false) => out.push_str("false"),
            Self::Int(n) => out.push_str(&n.to_string()),
            Self::String(s) => write_canonical_string(s, out),
            Self::Array(items) => {
                out.push('[');
                for (idx, item) in items.iter().enumerate() {
                    if idx > 0 {
                        out.push(',');
                    }
                    item.write_canonical(out);
                }
                out.push(']');
            }
            Self::Object(map) => {
                out.push('{');
                for (idx, (k, v)) in map.iter().enumerate() {
                    if idx > 0 {
                        out.push(',');
                    }
                    write_canonical_string(k, out);
                    out.push(':');
                    v.write_canonical(out);
                }
                out.push('}');
            }
        }
    }
}

/// Write a string using `jope.canonical-json/v1` escaping rules (`S5.md:1240`):
/// - `"` -> `\"`
/// - `\` -> `\\`
/// - `U+0000`..=`U+001F` -> lowercase six-character `\u00xx`
/// - `/` -> literal `/`
/// - all other Unicode scalar values -> literal UTF-8
pub fn write_canonical_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) <= 0x1f => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Canonical payload bytes validated under `jope.canonical-json/v1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalPayload {
    schema_id: String,
    canonical_bytes: Vec<u8>,
    value: CanonicalValue,
}

impl CanonicalPayload {
    /// Parse arbitrary UTF-8 JSON bytes, enforce all lexical/duplicate-key/numeric/authority/cycle
    /// rules, strip non-identity `display_f64` annotations from tagged float objects, and emit
    /// canonical `jope.canonical-json/v1` bytes.
    pub fn parse_and_canonicalize(raw: &[u8]) -> Result<Self, WireContractError> {
        let mut value = parse_strict_json_bytes(raw, true)?;
        validate_and_project_payload_root(&mut value)?;
        let schema_id = value
            .as_object()
            .and_then(|m| m.get("schema"))
            .and_then(CanonicalValue::as_str)
            .ok_or(WireContractError::MissingRequiredField {
                path: "/".to_string(),
                field: "schema",
            })?
            .to_string();
        let canonical_bytes = value.to_canonical_string().into_bytes();
        Ok(Self {
            schema_id,
            canonical_bytes,
            value,
        })
    }

    /// Verify that `bytes` are already in exact `jope.canonical-json/v1` canonical form
    /// (zero insignificant whitespace, keys already in UTF-8/Unicode-scalar order, canonical
    /// escaping, and no unprojected `display_f64` fields).
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, WireContractError> {
        let value = parse_strict_json_bytes(bytes, true)?;
        let mut projected = value.clone();
        validate_and_project_payload_root(&mut projected)?;
        if projected != value {
            return Err(WireContractError::NonCanonicalByteEncoding {
                detail: "non-identity display_f64 field present in canonical payload bytes"
                    .to_string(),
            });
        }
        let expected = projected.to_canonical_string().into_bytes();
        if expected != bytes {
            return Err(WireContractError::NonCanonicalByteEncoding {
                detail: "input bytes differ from canonical serialization (whitespace, key order, or escaping)".to_string(),
            });
        }
        let schema_id = projected
            .as_object()
            .and_then(|m| m.get("schema"))
            .and_then(CanonicalValue::as_str)
            .ok_or(WireContractError::MissingRequiredField {
                path: "/".to_string(),
                field: "schema",
            })?
            .to_string();
        Ok(Self {
            schema_id,
            canonical_bytes: expected,
            value: projected,
        })
    }

    pub fn schema_id(&self) -> &str {
        &self.schema_id
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.canonical_bytes).expect("canonical bytes are valid UTF-8")
    }

    pub fn value(&self) -> &CanonicalValue {
        &self.value
    }

    /// Compute the normative length-prefixed domain-separated ID for this canonical payload.
    pub fn domain_separated_id(&self) -> Result<DomainSeparatedId, WireContractError> {
        DomainSeparatedId::compute(&self.schema_id, &self.canonical_bytes)
    }
}

/// Normative domain-separated SHA-256 identity (`S5.md:1242`):
/// `SHA256(schema_utf8 || 0x00 || u64_be(payload_byte_length) || canonical_payload_bytes)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainSeparatedId {
    schema_id: String,
    payload_byte_len: u64,
    sha256_hex: String,
}

impl DomainSeparatedId {
    pub fn compute(
        schema_id: &str,
        canonical_payload_bytes: &[u8],
    ) -> Result<Self, WireContractError> {
        validate_schema_id(schema_id)?;
        let len_u64 = canonical_payload_bytes.len() as u64;
        if !(2..=(SAFE_INT_MAX as u64)).contains(&len_u64) {
            return Err(WireContractError::IntegerOverflow {
                path: "/own_id/payload_byte_len".to_string(),
                raw: len_u64.to_string(),
            });
        }
        let mut hasher = Sha256::new();
        hasher.update(schema_id.as_bytes());
        hasher.update([0x00]);
        hasher.update(len_u64.to_be_bytes());
        hasher.update(canonical_payload_bytes);
        let sha256_hex = hex_lower(&hasher.finalize());
        Ok(Self {
            schema_id: schema_id.to_string(),
            payload_byte_len: len_u64,
            sha256_hex,
        })
    }

    pub fn formula(&self) -> &'static str {
        DOMAIN_SEPARATED_ID_FORMULA
    }

    pub fn schema_id(&self) -> &str {
        &self.schema_id
    }

    pub fn payload_byte_len(&self) -> u64 {
        self.payload_byte_len
    }

    pub fn sha256_hex(&self) -> &str {
        &self.sha256_hex
    }

    pub fn to_canonical_value(&self) -> CanonicalValue {
        let mut map = BTreeMap::new();
        map.insert(
            "formula".to_string(),
            CanonicalValue::String(self.formula().to_string()),
        );
        map.insert(
            "payload_byte_len".to_string(),
            CanonicalValue::Int(self.payload_byte_len as i64),
        );
        map.insert(
            "schema_id".to_string(),
            CanonicalValue::String(self.schema_id.clone()),
        );
        map.insert(
            "sha256".to_string(),
            CanonicalValue::String(self.sha256_hex.clone()),
        );
        CanonicalValue::Object(map)
    }

    pub fn from_canonical_value(
        path: &str,
        val: &CanonicalValue,
    ) -> Result<Self, WireContractError> {
        let obj = val
            .as_object()
            .ok_or_else(|| WireContractError::SyntaxError {
                offset: 0,
                detail: format!("expected object at {path}"),
            })?;
        for k in obj.keys() {
            if !matches!(
                k.as_str(),
                "formula" | "schema_id" | "payload_byte_len" | "sha256"
            ) {
                return Err(WireContractError::UnknownField {
                    path: path.to_string(),
                    field: k.clone(),
                });
            }
        }
        let formula = required_str(obj, path, "formula")?;
        if formula != DOMAIN_SEPARATED_ID_FORMULA {
            return Err(WireContractError::UnknownEnumVariant {
                path: format!("{path}/formula"),
                value: formula.to_string(),
            });
        }
        let schema_id = required_str(obj, path, "schema_id")?.to_string();
        validate_schema_id(&schema_id)?;
        let len_i64 = required_i64(obj, path, "payload_byte_len")?;
        if len_i64 < 2 {
            return Err(WireContractError::IntegerOverflow {
                path: format!("{path}/payload_byte_len"),
                raw: len_i64.to_string(),
            });
        }
        let sha256_hex = required_str(obj, path, "sha256")?.to_string();
        validate_lowercase_hex(&format!("{path}/sha256"), &sha256_hex, 64)?;
        Ok(Self {
            schema_id,
            payload_byte_len: len_i64 as u64,
            sha256_hex,
        })
    }
}

/// Historical digest conventions retained read-only for legacy contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyHashScheme {
    /// `SHA256(canonical_json_bytes)` as used by `jope.hwm/v1` and `jope.score/v1`
    /// (`crates/jope-learning/src/canon.rs`).
    PlainCanonicalSha256,
    /// `SHA256(domain_utf8 || 0x00 || canonical_json_bytes)` without a u64 length prefix,
    /// as used by `jope.family-partition/v1` (`crates/jope-cli/src/train/corpus.rs:238`).
    SeparatorOnlyNullTerminated,
}

impl LegacyHashScheme {
    pub fn compute_digest(self, domain: &str, canonical_bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        match self {
            Self::PlainCanonicalSha256 => {
                hasher.update(canonical_bytes);
            }
            Self::SeparatorOnlyNullTerminated => {
                hasher.update(domain.as_bytes());
                hasher.update([0x00]);
                hasher.update(canonical_bytes);
            }
        }
        hex_lower(&hasher.finalize())
    }
}

/// Bridge record linking an immutable legacy digest (`PlainCanonicalSha256` or
/// `SeparatorOnlyNullTerminated`) to a new `DomainSeparatedId` without rehashing or
/// mutating the legacy artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyBridgeRecord {
    pub legacy_schema: String,
    pub legacy_scheme: LegacyHashScheme,
    pub legacy_sha256: String,
    pub new_id: DomainSeparatedId,
}

/// Adapter domain surface (`native`, `inference`, or `storage`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeDomain {
    Native,
    Inference,
    Storage,
}

impl EnvelopeDomain {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Inference => "inference",
            Self::Storage => "storage",
        }
    }

    pub fn parse(path: &str, raw: &str) -> Result<Self, WireContractError> {
        match raw {
            "native" => Ok(Self::Native),
            "inference" => Ok(Self::Inference),
            "storage" => Ok(Self::Storage),
            other => Err(WireContractError::UnknownEnumVariant {
                path: path.to_string(),
                value: other.to_string(),
            }),
        }
    }
}

/// Optional attestation/signature envelope kept strictly outside the payload identity projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureEnvelope {
    pub signer_role: String,
    pub signed_own_id_sha256: String,
    pub signature_hex: String,
}

impl SignatureEnvelope {
    pub fn new(
        signer_role: &str,
        signed_own_id_sha256: &str,
        signature_hex: &str,
    ) -> Result<Self, WireContractError> {
        if !matches!(
            signer_role,
            "native_worker" | "inference_worker" | "storage_committer" | "offline_verifier"
        ) {
            return Err(WireContractError::UnknownEnumVariant {
                path: "/signature/signer_role".to_string(),
                value: signer_role.to_string(),
            });
        }
        validate_lowercase_hex("/signature/signed_own_id_sha256", signed_own_id_sha256, 64)?;
        validate_lowercase_hex("/signature/signature_hex", signature_hex, 64)?;
        Ok(Self {
            signer_role: signer_role.to_string(),
            signed_own_id_sha256: signed_own_id_sha256.to_string(),
            signature_hex: signature_hex.to_string(),
        })
    }

    pub fn to_canonical_value(&self) -> CanonicalValue {
        let mut map = BTreeMap::new();
        map.insert(
            "signature_hex".to_string(),
            CanonicalValue::String(self.signature_hex.clone()),
        );
        map.insert(
            "signed_own_id_sha256".to_string(),
            CanonicalValue::String(self.signed_own_id_sha256.clone()),
        );
        map.insert(
            "signer_role".to_string(),
            CanonicalValue::String(self.signer_role.clone()),
        );
        CanonicalValue::Object(map)
    }

    pub fn from_canonical_value(
        path: &str,
        val: &CanonicalValue,
    ) -> Result<Self, WireContractError> {
        let obj = val
            .as_object()
            .ok_or_else(|| WireContractError::SyntaxError {
                offset: 0,
                detail: format!("expected object at {path}"),
            })?;
        for k in obj.keys() {
            if !matches!(
                k.as_str(),
                "signer_role" | "signed_own_id_sha256" | "signature_hex"
            ) {
                return Err(WireContractError::UnknownField {
                    path: path.to_string(),
                    field: k.clone(),
                });
            }
        }
        let signer_role = required_str(obj, path, "signer_role")?;
        let signed_own_id_sha256 = required_str(obj, path, "signed_own_id_sha256")?;
        let signature_hex = required_str(obj, path, "signature_hex")?;
        Self::new(signer_role, signed_own_id_sha256, signature_hex)
    }
}

/// Complete separated wire envelope (`payload` + `own_id` + optional `signature` + optional `display`).
///
/// `own_id` depends only on `(schema_id, payload.as_bytes())`. Mutating `domain`, `signature`,
/// or `display_note` never changes `payload.as_bytes()` or `own_id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvelopeFrame {
    domain: EnvelopeDomain,
    schema_id: String,
    own_id: DomainSeparatedId,
    payload: CanonicalPayload,
    signature: Option<SignatureEnvelope>,
    display_note: Option<String>,
}

impl EnvelopeFrame {
    pub fn new(
        domain: EnvelopeDomain,
        payload: CanonicalPayload,
        signature: Option<SignatureEnvelope>,
        display_note: Option<String>,
    ) -> Result<Self, WireContractError> {
        let own_id = payload.domain_separated_id()?;
        if let Some(sig) = &signature {
            if sig.signed_own_id_sha256 != own_id.sha256_hex {
                return Err(WireContractError::SignatureMismatch {
                    reason: format!(
                        "signature binds own_id {}, but payload own_id is {}",
                        sig.signed_own_id_sha256, own_id.sha256_hex
                    ),
                });
            }
        }
        Ok(Self {
            domain,
            schema_id: payload.schema_id().to_string(),
            own_id,
            payload,
            signature,
            display_note,
        })
    }

    pub fn domain(&self) -> EnvelopeDomain {
        self.domain
    }

    pub fn schema_id(&self) -> &str {
        &self.schema_id
    }

    pub fn own_id(&self) -> &DomainSeparatedId {
        &self.own_id
    }

    pub fn payload(&self) -> &CanonicalPayload {
        &self.payload
    }

    pub fn signature(&self) -> Option<&SignatureEnvelope> {
        self.signature.as_ref()
    }

    pub fn display_note(&self) -> Option<&str> {
        self.display_note.as_deref()
    }

    /// Emit the complete envelope in `jope.canonical-json/v1` canonical byte form.
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut root = BTreeMap::new();
        if let Some(note) = &self.display_note {
            let mut disp = BTreeMap::new();
            disp.insert("note".to_string(), CanonicalValue::String(note.clone()));
            root.insert("display".to_string(), CanonicalValue::Object(disp));
        }
        root.insert(
            "domain".to_string(),
            CanonicalValue::String(self.domain.as_str().to_string()),
        );
        root.insert(
            "envelope_schema".to_string(),
            CanonicalValue::String(CANONICAL_JSON_ENVELOPE_SCHEMA.to_string()),
        );
        root.insert("own_id".to_string(), self.own_id.to_canonical_value());
        root.insert("payload".to_string(), self.payload.value().clone());
        root.insert(
            "schema_id".to_string(),
            CanonicalValue::String(self.schema_id.clone()),
        );
        if let Some(sig) = &self.signature {
            root.insert("signature".to_string(), sig.to_canonical_value());
        }
        CanonicalValue::Object(root)
            .to_canonical_string()
            .into_bytes()
    }

    /// Parse and verify an `EnvelopeFrame` from UTF-8 JSON bytes, checking all lexical,
    /// schema, cycle, `own_id`, and signature invariants.
    pub fn parse_and_verify(raw: &[u8]) -> Result<Self, WireContractError> {
        let val = parse_strict_json_bytes(raw, true)?;
        let obj = val
            .as_object()
            .ok_or_else(|| WireContractError::SyntaxError {
                offset: 0,
                detail: "envelope root must be a JSON object".to_string(),
            })?;
        for k in obj.keys() {
            if !ALLOWED_ENVELOPE_KEYS.contains(&k.as_str()) {
                return Err(WireContractError::UnknownField {
                    path: "/".to_string(),
                    field: k.clone(),
                });
            }
        }
        let env_schema = required_str(obj, "/", "envelope_schema")?;
        if env_schema != CANONICAL_JSON_ENVELOPE_SCHEMA {
            return Err(WireContractError::UnknownEnumVariant {
                path: "/envelope_schema".to_string(),
                value: env_schema.to_string(),
            });
        }
        let domain = EnvelopeDomain::parse("/domain", required_str(obj, "/", "domain")?)?;
        let schema_id = required_str(obj, "/", "schema_id")?.to_string();
        validate_schema_id(&schema_id)?;

        let own_id_val = obj
            .get("own_id")
            .ok_or(WireContractError::MissingRequiredField {
                path: "/".to_string(),
                field: "own_id",
            })?;
        let claimed_own_id = DomainSeparatedId::from_canonical_value("/own_id", own_id_val)?;

        let payload_val = obj
            .get("payload")
            .ok_or(WireContractError::MissingRequiredField {
                path: "/".to_string(),
                field: "payload",
            })?;
        let payload_bytes = payload_val.to_canonical_string().into_bytes();
        let payload = CanonicalPayload::parse_and_canonicalize(&payload_bytes)?;

        if payload.schema_id() != schema_id {
            return Err(WireContractError::SchemaIdMismatch {
                envelope: schema_id,
                payload: payload.schema_id().to_string(),
            });
        }
        let computed_own_id = payload.domain_separated_id()?;
        if claimed_own_id != computed_own_id {
            return Err(WireContractError::EnvelopeIdMismatch {
                expected: computed_own_id.sha256_hex().to_string(),
                actual: claimed_own_id.sha256_hex().to_string(),
            });
        }

        let signature = match obj.get("signature") {
            Some(sig_val) => Some(SignatureEnvelope::from_canonical_value(
                "/signature",
                sig_val,
            )?),
            None => None,
        };
        let display_note = match obj.get("display") {
            Some(disp_val) => {
                let disp_obj =
                    disp_val
                        .as_object()
                        .ok_or_else(|| WireContractError::SyntaxError {
                            offset: 0,
                            detail: "expected object at /display".to_string(),
                        })?;
                for k in disp_obj.keys() {
                    if !matches!(k.as_str(), "note" | "display_f64") {
                        return Err(WireContractError::UnknownField {
                            path: "/display".to_string(),
                            field: k.clone(),
                        });
                    }
                }
                disp_obj
                    .get("note")
                    .and_then(CanonicalValue::as_str)
                    .map(str::to_string)
            }
            None => None,
        };

        Self::new(domain, payload, signature, display_note)
    }
}

/// Native training/evaluation envelope adapter (`domain = "native"`).
#[derive(Debug, Default, Clone, Copy)]
pub struct NativeEnvelopeAdapter;

impl NativeEnvelopeAdapter {
    pub fn seal(
        payload_json: &[u8],
        signature: Option<SignatureEnvelope>,
        display_note: Option<String>,
    ) -> Result<EnvelopeFrame, WireContractError> {
        let payload = CanonicalPayload::parse_and_canonicalize(payload_json)?;
        EnvelopeFrame::new(EnvelopeDomain::Native, payload, signature, display_note)
    }

    pub fn open(envelope_bytes: &[u8]) -> Result<EnvelopeFrame, WireContractError> {
        let frame = EnvelopeFrame::parse_and_verify(envelope_bytes)?;
        if frame.domain() != EnvelopeDomain::Native {
            return Err(WireContractError::UnknownEnumVariant {
                path: "/domain".to_string(),
                value: frame.domain().as_str().to_string(),
            });
        }
        Ok(frame)
    }
}

/// Inference-facing envelope adapter (`domain = "inference"`).
#[derive(Debug, Default, Clone, Copy)]
pub struct InferenceEnvelopeAdapter;

impl InferenceEnvelopeAdapter {
    pub fn seal(
        payload_json: &[u8],
        signature: Option<SignatureEnvelope>,
        display_note: Option<String>,
    ) -> Result<EnvelopeFrame, WireContractError> {
        let payload = CanonicalPayload::parse_and_canonicalize(payload_json)?;
        EnvelopeFrame::new(EnvelopeDomain::Inference, payload, signature, display_note)
    }

    pub fn open(envelope_bytes: &[u8]) -> Result<EnvelopeFrame, WireContractError> {
        let frame = EnvelopeFrame::parse_and_verify(envelope_bytes)?;
        if frame.domain() != EnvelopeDomain::Inference {
            return Err(WireContractError::UnknownEnumVariant {
                path: "/domain".to_string(),
                value: frame.domain().as_str().to_string(),
            });
        }
        Ok(frame)
    }
}

/// Storage-facing envelope adapter (`domain = "storage"`).
#[derive(Debug, Default, Clone, Copy)]
pub struct StorageEnvelopeAdapter;

impl StorageEnvelopeAdapter {
    pub fn seal(
        payload_json: &[u8],
        signature: Option<SignatureEnvelope>,
        display_note: Option<String>,
    ) -> Result<EnvelopeFrame, WireContractError> {
        let payload = CanonicalPayload::parse_and_canonicalize(payload_json)?;
        EnvelopeFrame::new(EnvelopeDomain::Storage, payload, signature, display_note)
    }

    pub fn open(envelope_bytes: &[u8]) -> Result<EnvelopeFrame, WireContractError> {
        let frame = EnvelopeFrame::parse_and_verify(envelope_bytes)?;
        if frame.domain() != EnvelopeDomain::Storage {
            return Err(WireContractError::UnknownEnumVariant {
                path: "/domain".to_string(),
                value: frame.domain().as_str().to_string(),
            });
        }
        Ok(frame)
    }
}

/// Enumerated evidence classes eligible to back a `VerifiedMeasuredOutcome`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasuredEvidenceKind {
    MeasuredCpuCv,
    MeasuredHoldoutEval,
    MeasuredBenchmarkRun,
}

impl MeasuredEvidenceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MeasuredCpuCv => "measured_cpu_cv",
            Self::MeasuredHoldoutEval => "measured_holdout_eval",
            Self::MeasuredBenchmarkRun => "measured_benchmark_run",
        }
    }

    pub fn parse(path: &str, s: &str) -> Result<Self, WireContractError> {
        match s {
            "measured_cpu_cv" => Ok(Self::MeasuredCpuCv),
            "measured_holdout_eval" => Ok(Self::MeasuredHoldoutEval),
            "measured_benchmark_run" => Ok(Self::MeasuredBenchmarkRun),
            other => Err(WireContractError::UnknownEnumVariant {
                path: path.to_string(),
                value: other.to_string(),
            }),
        }
    }
}

/// Typed reason for a `MissingOutcome` (never coerced to `0.0` or `null`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingReason {
    NotApplicable,
    Unscheduled,
    BudgetExhausted,
    FailedClosed,
    SealedHoldout,
}

impl MissingReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotApplicable => "not_applicable",
            Self::Unscheduled => "unscheduled",
            Self::BudgetExhausted => "budget_exhausted",
            Self::FailedClosed => "failed_closed",
            Self::SealedHoldout => "sealed_holdout",
        }
    }

    pub fn parse(path: &str, s: &str) -> Result<Self, WireContractError> {
        match s {
            "not_applicable" => Ok(Self::NotApplicable),
            "unscheduled" => Ok(Self::Unscheduled),
            "budget_exhausted" => Ok(Self::BudgetExhausted),
            "failed_closed" => Ok(Self::FailedClosed),
            "sealed_holdout" => Ok(Self::SealedHoldout),
            other => Err(WireContractError::UnknownEnumVariant {
                path: path.to_string(),
                value: other.to_string(),
            }),
        }
    }
}

/// Unverified measured claim parsed from JSON wire bytes.
///
/// Deserializing or parsing JSON wire bytes NEVER yields [`VerifiedMeasuredOutcome`] directly;
/// a `ParsedMeasuredClaim` must pass [`VerifiedMeasuredOutcome::verify_from_claim`] with an
/// independent [`MeasuredEvidenceProof`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedMeasuredClaim {
    pub metric_name: String,
    pub metric: ExactF64,
    pub evidence_receipt_sha256: String,
    pub evidence_kind: MeasuredEvidenceKind,
}

/// Proof artifact supplied by the evidence verifier to upgrade a [`ParsedMeasuredClaim`]
/// into a [`VerifiedMeasuredOutcome`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasuredEvidenceProof {
    pub receipt_sha256: String,
    pub evidence_kind: MeasuredEvidenceKind,
    pub independent_verifier_passed: bool,
}

/// Verified measured outcome authority token (`S8.md:423-437`).
///
/// Private fields prevent direct struct-literal construction and unchecked deserialization.
/// Neither [`PredictedOutcome`], [`SyntheticOutcome`], [`MissingOutcome`], nor unverified
/// [`ParsedMeasuredClaim`] can be converted into `VerifiedMeasuredOutcome` or passed to
/// [`accept_measured_claim`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedMeasuredOutcome {
    metric_name: String,
    metric: ExactF64,
    evidence_receipt_sha256: String,
    evidence_kind: MeasuredEvidenceKind,
    _sealed: (),
}

impl VerifiedMeasuredOutcome {
    /// Construct a `VerifiedMeasuredOutcome` only after verifying an independent evidence proof
    /// against a parsed measured claim.
    pub fn verify_from_claim(
        claim: &ParsedMeasuredClaim,
        proof: &MeasuredEvidenceProof,
    ) -> Result<Self, WireContractError> {
        if claim.metric_name.trim().is_empty() {
            return Err(WireContractError::InvalidOutcomeAuthority {
                reason: "metric_name must be non-empty".to_string(),
            });
        }
        validate_lowercase_hex(
            "/outcome/evidence_receipt_sha256",
            &claim.evidence_receipt_sha256,
            64,
        )?;
        if claim.evidence_receipt_sha256.bytes().all(|b| b == b'0') {
            return Err(WireContractError::InvalidOutcomeAuthority {
                reason: "all-zero evidence_receipt_sha256 is rejected".to_string(),
            });
        }
        if !proof.independent_verifier_passed {
            return Err(WireContractError::InvalidOutcomeAuthority {
                reason: "independent_verifier_passed is false".to_string(),
            });
        }
        if proof.receipt_sha256 != claim.evidence_receipt_sha256 {
            return Err(WireContractError::InvalidOutcomeAuthority {
                reason: format!(
                    "proof receipt_sha256 {} does not match claim {}",
                    proof.receipt_sha256, claim.evidence_receipt_sha256
                ),
            });
        }
        if proof.evidence_kind != claim.evidence_kind {
            return Err(WireContractError::InvalidOutcomeAuthority {
                reason: format!(
                    "proof evidence_kind {} does not match claim {}",
                    proof.evidence_kind.as_str(),
                    claim.evidence_kind.as_str()
                ),
            });
        }
        Ok(Self {
            metric_name: claim.metric_name.clone(),
            metric: claim.metric,
            evidence_receipt_sha256: claim.evidence_receipt_sha256.clone(),
            evidence_kind: claim.evidence_kind,
            _sealed: (),
        })
    }

    pub fn metric_name(&self) -> &str {
        &self.metric_name
    }

    pub fn metric(&self) -> ExactF64 {
        self.metric
    }

    pub fn evidence_receipt_sha256(&self) -> &str {
        &self.evidence_receipt_sha256
    }

    pub fn evidence_kind(&self) -> MeasuredEvidenceKind {
        self.evidence_kind
    }
}

/// Model-predicted surrogate outcome (`authority = "predicted"`).
/// Cannot enter measured-claim APIs ([`accept_measured_claim`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictedOutcome {
    pub metric_name: String,
    pub metric: ExactF64,
    pub model_bundle_sha256: String,
}

/// Synthetic fixture outcome (`authority = "synthetic"`).
/// Cannot enter measured-claim APIs ([`accept_measured_claim`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticOutcome {
    pub metric_name: String,
    pub metric: ExactF64,
    pub synthetic_fixture_id: String,
}

/// Missing outcome (`authority = "missing"`), carrying an explicit [`MissingReason`]
/// rather than coercing to `0.0` or `null`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingOutcome {
    pub metric_name: String,
    pub missing_reason: MissingReason,
}

/// Parsed wire representation of an outcome record before evidence verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedOutcomeWire {
    UnverifiedMeasured(ParsedMeasuredClaim),
    Predicted(PredictedOutcome),
    Synthetic(SyntheticOutcome),
    Missing(MissingOutcome),
}

/// Receipt emitted by [`accept_measured_claim`] when a [`VerifiedMeasuredOutcome`] is admitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasuredClaimReceipt {
    pub metric_name: String,
    pub metric_f64_bits: String,
    pub evidence_receipt_sha256: String,
    pub evidence_kind: MeasuredEvidenceKind,
}

/// Admit a verified measured outcome into the measured-claim boundary.
///
/// By type construction, only [`VerifiedMeasuredOutcome`] is accepted. Passing
/// [`PredictedOutcome`], [`SyntheticOutcome`], [`MissingOutcome`], or an unverified
/// [`ParsedMeasuredClaim`] is a compile-time type error.
///
/// # Examples
///
/// Verified measured evidence is accepted after proof verification:
///
/// ```
/// use jope_records::wire_contract::{
///     accept_measured_claim, ExactF64, MeasuredEvidenceKind, MeasuredEvidenceProof,
///     ParsedMeasuredClaim, VerifiedMeasuredOutcome,
/// };
///
/// let receipt_sha = "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90";
/// let claim = ParsedMeasuredClaim {
///     metric_name: "cv_r2".to_string(),
///     metric: ExactF64::from_f64(0.875).unwrap(),
///     evidence_receipt_sha256: receipt_sha.to_string(),
///     evidence_kind: MeasuredEvidenceKind::MeasuredCpuCv,
/// };
/// let proof = MeasuredEvidenceProof {
///     receipt_sha256: receipt_sha.to_string(),
///     evidence_kind: MeasuredEvidenceKind::MeasuredCpuCv,
///     independent_verifier_passed: true,
/// };
/// let verified = VerifiedMeasuredOutcome::verify_from_claim(&claim, &proof).unwrap();
/// let admitted = accept_measured_claim(&verified);
/// assert_eq!(admitted.metric_f64_bits, "3fec000000000000");
/// ```
///
/// A [`PredictedOutcome`] cannot enter `accept_measured_claim` (compile-time rejection):
///
/// ```compile_fail
/// use jope_records::wire_contract::{accept_measured_claim, ExactF64, PredictedOutcome};
///
/// let pred = PredictedOutcome {
///     metric_name: "cv_r2".to_string(),
///     metric: ExactF64::from_f64(0.875).unwrap(),
///     model_bundle_sha256: "11".repeat(32),
/// };
/// let _ = accept_measured_claim(&pred);
/// ```
///
/// A [`SyntheticOutcome`] cannot enter `accept_measured_claim` (compile-time rejection):
///
/// ```compile_fail
/// use jope_records::wire_contract::{accept_measured_claim, ExactF64, SyntheticOutcome};
///
/// let syn = SyntheticOutcome {
///     metric_name: "cv_r2".to_string(),
///     metric: ExactF64::from_f64(0.875).unwrap(),
///     synthetic_fixture_id: "fixture-01".to_string(),
/// };
/// let _ = accept_measured_claim(&syn);
/// ```
///
/// An unverified [`ParsedMeasuredClaim`] or direct struct-literal construction of
/// [`VerifiedMeasuredOutcome`] cannot enter `accept_measured_claim` (compile-time rejection):
///
/// ```compile_fail
/// use jope_records::wire_contract::{
///     accept_measured_claim, ExactF64, MeasuredEvidenceKind, ParsedMeasuredClaim,
/// };
///
/// let unverified = ParsedMeasuredClaim {
///     metric_name: "cv_r2".to_string(),
///     metric: ExactF64::from_f64(0.875).unwrap(),
///     evidence_receipt_sha256: "aa".repeat(32),
///     evidence_kind: MeasuredEvidenceKind::MeasuredCpuCv,
/// };
/// let _ = accept_measured_claim(&unverified);
/// ```
pub fn accept_measured_claim(outcome: &VerifiedMeasuredOutcome) -> MeasuredClaimReceipt {
    MeasuredClaimReceipt {
        metric_name: outcome.metric_name.clone(),
        metric_f64_bits: outcome.metric.to_hex(),
        evidence_receipt_sha256: outcome.evidence_receipt_sha256.clone(),
        evidence_kind: outcome.evidence_kind,
    }
}

/// Parse an outcome object from a `CanonicalValue` into [`ParsedOutcomeWire`].
pub fn parse_outcome_wire(
    path: &str,
    val: &CanonicalValue,
) -> Result<ParsedOutcomeWire, WireContractError> {
    let obj = val
        .as_object()
        .ok_or_else(|| WireContractError::SyntaxError {
            offset: 0,
            detail: format!("expected outcome object at {path}"),
        })?;
    for k in obj.keys() {
        if !matches!(
            k.as_str(),
            "authority"
                | "metric_name"
                | "metric"
                | "evidence_receipt_sha256"
                | "evidence_kind"
                | "model_bundle_sha256"
                | "synthetic_fixture_id"
                | "missing_reason"
        ) {
            return Err(WireContractError::UnknownField {
                path: path.to_string(),
                field: k.clone(),
            });
        }
    }
    let authority = required_str(obj, path, "authority")?;
    let metric_name = required_str(obj, path, "metric_name")?.to_string();
    if metric_name.trim().is_empty() {
        return Err(WireContractError::InvalidOutcomeAuthority {
            reason: format!("empty metric_name at {path}"),
        });
    }
    match authority {
        "verified_measured" => {
            for forbidden in [
                "model_bundle_sha256",
                "synthetic_fixture_id",
                "missing_reason",
            ] {
                if obj.contains_key(forbidden) {
                    return Err(WireContractError::InvalidOutcomeAuthority {
                        reason: format!(
                            "verified_measured outcome at {path} cannot contain {forbidden:?}"
                        ),
                    });
                }
            }
            let metric_val = obj
                .get("metric")
                .ok_or(WireContractError::MissingRequiredField {
                    path: path.to_string(),
                    field: "metric",
                })?;
            let metric = parse_exact_f64_object(&format!("{path}/metric"), metric_val)?;
            let evidence_receipt_sha256 =
                required_str(obj, path, "evidence_receipt_sha256")?.to_string();
            validate_lowercase_hex(
                &format!("{path}/evidence_receipt_sha256"),
                &evidence_receipt_sha256,
                64,
            )?;
            let evidence_kind = MeasuredEvidenceKind::parse(
                &format!("{path}/evidence_kind"),
                required_str(obj, path, "evidence_kind")?,
            )?;
            Ok(ParsedOutcomeWire::UnverifiedMeasured(ParsedMeasuredClaim {
                metric_name,
                metric,
                evidence_receipt_sha256,
                evidence_kind,
            }))
        }
        "predicted" => {
            for forbidden in [
                "evidence_receipt_sha256",
                "evidence_kind",
                "synthetic_fixture_id",
                "missing_reason",
            ] {
                if obj.contains_key(forbidden) {
                    return Err(WireContractError::InvalidOutcomeAuthority {
                        reason: format!("predicted outcome at {path} cannot contain {forbidden:?}"),
                    });
                }
            }
            let metric_val = obj
                .get("metric")
                .ok_or(WireContractError::MissingRequiredField {
                    path: path.to_string(),
                    field: "metric",
                })?;
            let metric = parse_exact_f64_object(&format!("{path}/metric"), metric_val)?;
            let model_bundle_sha256 = required_str(obj, path, "model_bundle_sha256")?.to_string();
            validate_lowercase_hex(
                &format!("{path}/model_bundle_sha256"),
                &model_bundle_sha256,
                64,
            )?;
            Ok(ParsedOutcomeWire::Predicted(PredictedOutcome {
                metric_name,
                metric,
                model_bundle_sha256,
            }))
        }
        "synthetic" => {
            for forbidden in [
                "evidence_receipt_sha256",
                "evidence_kind",
                "model_bundle_sha256",
                "missing_reason",
            ] {
                if obj.contains_key(forbidden) {
                    return Err(WireContractError::InvalidOutcomeAuthority {
                        reason: format!("synthetic outcome at {path} cannot contain {forbidden:?}"),
                    });
                }
            }
            let metric_val = obj
                .get("metric")
                .ok_or(WireContractError::MissingRequiredField {
                    path: path.to_string(),
                    field: "metric",
                })?;
            let metric = parse_exact_f64_object(&format!("{path}/metric"), metric_val)?;
            let synthetic_fixture_id = required_str(obj, path, "synthetic_fixture_id")?.to_string();
            if synthetic_fixture_id.trim().is_empty() {
                return Err(WireContractError::InvalidOutcomeAuthority {
                    reason: format!("empty synthetic_fixture_id at {path}"),
                });
            }
            Ok(ParsedOutcomeWire::Synthetic(SyntheticOutcome {
                metric_name,
                metric,
                synthetic_fixture_id,
            }))
        }
        "missing" => {
            for forbidden in [
                "metric",
                "evidence_receipt_sha256",
                "evidence_kind",
                "model_bundle_sha256",
                "synthetic_fixture_id",
            ] {
                if obj.contains_key(forbidden) {
                    return Err(WireContractError::InvalidOutcomeAuthority {
                        reason: format!("missing outcome at {path} cannot contain {forbidden:?}"),
                    });
                }
            }
            let missing_reason = MissingReason::parse(
                &format!("{path}/missing_reason"),
                required_str(obj, path, "missing_reason")?,
            )?;
            Ok(ParsedOutcomeWire::Missing(MissingOutcome {
                metric_name,
                missing_reason,
            }))
        }
        other => Err(WireContractError::UnknownEnumVariant {
            path: format!("{path}/authority"),
            value: other.to_string(),
        }),
    }
}

fn parse_exact_f64_object(path: &str, val: &CanonicalValue) -> Result<ExactF64, WireContractError> {
    let obj = val
        .as_object()
        .ok_or_else(|| WireContractError::SyntaxError {
            offset: 0,
            detail: format!("expected exact_f64 object at {path}"),
        })?;
    for k in obj.keys() {
        if k != "f64_bits" {
            return Err(WireContractError::UnknownField {
                path: path.to_string(),
                field: k.clone(),
            });
        }
    }
    let hex_bits = required_str(obj, path, "f64_bits")?;
    ExactF64::from_hex(&format!("{path}/f64_bits"), hex_bits)
}

fn parse_exact_f32_object(path: &str, val: &CanonicalValue) -> Result<ExactF32, WireContractError> {
    let obj = val
        .as_object()
        .ok_or_else(|| WireContractError::SyntaxError {
            offset: 0,
            detail: format!("expected exact_f32 object at {path}"),
        })?;
    for k in obj.keys() {
        if k != "f32_bits" {
            return Err(WireContractError::UnknownField {
                path: path.to_string(),
                field: k.clone(),
            });
        }
    }
    let hex_bits = required_str(obj, path, "f32_bits")?;
    ExactF32::from_hex(&format!("{path}/f32_bits"), hex_bits)
}

/// Validate a logical `schema_id` (`jope.<family>/v<N>`, no NUL byte, length 8..=128).
pub fn validate_schema_id(schema_id: &str) -> Result<(), WireContractError> {
    if schema_id.as_bytes().contains(&0) {
        return Err(WireContractError::InvalidSchemaId {
            schema_id: schema_id.to_string(),
            reason: "schema_id must not contain NUL (0x00) byte",
        });
    }
    if !(8..=128).contains(&schema_id.len()) {
        return Err(WireContractError::InvalidSchemaId {
            schema_id: schema_id.to_string(),
            reason: "schema_id length must be in 8..=128",
        });
    }
    let Some(rest) = schema_id.strip_prefix("jope.") else {
        return Err(WireContractError::InvalidSchemaId {
            schema_id: schema_id.to_string(),
            reason: "schema_id must start with 'jope.'",
        });
    };
    let Some((name, ver)) = rest.rsplit_once("/v") else {
        return Err(WireContractError::InvalidSchemaId {
            schema_id: schema_id.to_string(),
            reason: "schema_id must end with '/v<positive-integer>'",
        });
    };
    if name.is_empty()
        || !name.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-')
        })
    {
        return Err(WireContractError::InvalidSchemaId {
            schema_id: schema_id.to_string(),
            reason: "schema family name must contain only [a-z0-9._-]",
        });
    }
    if ver.is_empty() || ver.starts_with('0') || !ver.bytes().all(|b| b.is_ascii_digit()) {
        return Err(WireContractError::InvalidSchemaId {
            schema_id: schema_id.to_string(),
            reason: "schema version suffix must be a positive integer without leading zeros",
        });
    }
    Ok(())
}

fn validate_and_project_payload_root(root: &mut CanonicalValue) -> Result<(), WireContractError> {
    let Some(obj) = root.as_object() else {
        return Err(WireContractError::SyntaxError {
            offset: 0,
            detail: "canonical payload root must be a JSON object".to_string(),
        });
    };
    let schema = required_str(obj, "/", "schema")?;
    validate_schema_id(schema)?;
    validate_and_project_value("", root)
}

fn validate_and_project_value(
    path: &str,
    val: &mut CanonicalValue,
) -> Result<(), WireContractError> {
    let here = if path.is_empty() { "/" } else { path };
    match val {
        CanonicalValue::Null | CanonicalValue::Bool(_) | CanonicalValue::String(_) => Ok(()),
        CanonicalValue::Int(n) => {
            if !(SAFE_INT_MIN..=SAFE_INT_MAX).contains(n) {
                return Err(WireContractError::IntegerOverflow {
                    path: here.to_string(),
                    raw: n.to_string(),
                });
            }
            Ok(())
        }
        CanonicalValue::Array(items) => {
            for (idx, item) in items.iter_mut().enumerate() {
                let child_path = format!("{path}/{idx}");
                validate_and_project_value(&child_path, item)?;
            }
            Ok(())
        }
        CanonicalValue::Object(map) => {
            for &forbidden in FORBIDDEN_PAYLOAD_CYCLE_KEYS {
                if map.contains_key(forbidden) {
                    return Err(WireContractError::ForbiddenCycleField {
                        path: here.to_string(),
                        field: forbidden.to_string(),
                    });
                }
            }
            // Strip optional human-only display_f64 from tagged ExactF32 / ExactF64 objects
            // so human display numbers never enter the canonical identity projection.
            if map.contains_key("f64_bits") || map.contains_key("f32_bits") {
                map.remove("display_f64");
            }
            for (k, v) in map.iter_mut() {
                let child_path = format!("{path}/{k}");
                validate_and_project_value(&child_path, v)?;
            }
            if map.contains_key("f64_bits") {
                let _ = parse_exact_f64_object(here, &CanonicalValue::Object(map.clone()))?;
            }
            if map.contains_key("f32_bits") {
                let _ = parse_exact_f32_object(here, &CanonicalValue::Object(map.clone()))?;
            }
            if let Some(seed_val) = map.get("seed_u64_hex") {
                let s = seed_val
                    .as_str()
                    .ok_or_else(|| WireContractError::SyntaxError {
                        offset: 0,
                        detail: format!("seed_u64_hex at {here} must be a string"),
                    })?;
                let _ = HexU64::from_hex(&format!("{path}/seed_u64_hex"), s)?;
            }
            if let Some(outcome_val) = map.get("outcome") {
                let _ = parse_outcome_wire(&format!("{path}/outcome"), outcome_val)?;
            }
            Ok(())
        }
    }
}

fn required_str<'a>(
    obj: &'a BTreeMap<String, CanonicalValue>,
    path: &str,
    field: &'static str,
) -> Result<&'a str, WireContractError> {
    obj.get(field)
        .and_then(CanonicalValue::as_str)
        .ok_or_else(|| WireContractError::MissingRequiredField {
            path: path.to_string(),
            field,
        })
}

fn required_i64(
    obj: &BTreeMap<String, CanonicalValue>,
    path: &str,
    field: &'static str,
) -> Result<i64, WireContractError> {
    obj.get(field)
        .and_then(CanonicalValue::as_i64)
        .ok_or_else(|| WireContractError::MissingRequiredField {
            path: path.to_string(),
            field,
        })
}

fn validate_lowercase_hex(
    path: &str,
    value: &str,
    expected_len: usize,
) -> Result<(), WireContractError> {
    if value.len() != expected_len
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(WireContractError::InvalidHexEncoding {
            path: path.to_string(),
            value: value.to_string(),
            expected_len,
        });
    }
    Ok(())
}

fn parse_fixed_lowercase_hex_u32(path: &str, value: &str) -> Result<u32, WireContractError> {
    validate_lowercase_hex(path, value, 8)?;
    u32::from_str_radix(value, 16).map_err(|_| WireContractError::InvalidHexEncoding {
        path: path.to_string(),
        value: value.to_string(),
        expected_len: 8,
    })
}

fn parse_fixed_lowercase_hex_u64(path: &str, value: &str) -> Result<u64, WireContractError> {
    validate_lowercase_hex(path, value, 16)?;
    u64::from_str_radix(value, 16).map_err(|_| WireContractError::InvalidHexEncoding {
        path: path.to_string(),
        value: value.to_string(),
        expected_len: 16,
    })
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

struct StrictParser<'a> {
    bytes: &'a [u8],
    pos: usize,
    allow_display_f64: bool,
}

fn parse_strict_json_bytes(
    raw: &[u8],
    allow_display_f64: bool,
) -> Result<CanonicalValue, WireContractError> {
    if raw.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return Err(WireContractError::ByteOrderMark);
    }
    let _ = std::str::from_utf8(raw).map_err(|_| WireContractError::InvalidUtf8)?;
    let mut parser = StrictParser {
        bytes: raw,
        pos: 0,
        allow_display_f64,
    };
    parser.skip_ws();
    let val = parser.parse_value("")?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(WireContractError::TrailingBytes { offset: parser.pos });
    }
    Ok(val)
}

impl<'a> StrictParser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self, path: &str) -> Result<CanonicalValue, WireContractError> {
        self.skip_ws();
        let Some(b) = self.peek() else {
            return Err(WireContractError::UnexpectedEof);
        };
        match b {
            b'{' => self.parse_object(path),
            b'[' => self.parse_array(path),
            b'"' => Ok(CanonicalValue::String(self.parse_string()?)),
            b't' => {
                if self.bytes[self.pos..].starts_with(b"true") {
                    self.pos += 4;
                    Ok(CanonicalValue::Bool(true))
                } else {
                    Err(WireContractError::SyntaxError {
                        offset: self.pos,
                        detail: "invalid literal".to_string(),
                    })
                }
            }
            b'f' => {
                if self.bytes[self.pos..].starts_with(b"false") {
                    self.pos += 5;
                    Ok(CanonicalValue::Bool(false))
                } else {
                    Err(WireContractError::SyntaxError {
                        offset: self.pos,
                        detail: "invalid literal".to_string(),
                    })
                }
            }
            b'n' => {
                if self.bytes[self.pos..].starts_with(b"null") {
                    self.pos += 4;
                    Ok(CanonicalValue::Null)
                } else {
                    Err(WireContractError::SyntaxError {
                        offset: self.pos,
                        detail: "invalid literal".to_string(),
                    })
                }
            }
            b'-' | b'0'..=b'9' => self.parse_number(path),
            other => Err(WireContractError::SyntaxError {
                offset: self.pos,
                detail: format!("unexpected byte 0x{other:02x}"),
            }),
        }
    }

    fn parse_object(&mut self, path: &str) -> Result<CanonicalValue, WireContractError> {
        let here = if path.is_empty() { "/" } else { path };
        self.pos += 1; // '{'
        self.skip_ws();
        let mut map = BTreeMap::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(CanonicalValue::Object(map));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(WireContractError::SyntaxError {
                    offset: self.pos,
                    detail: "expected string key in object".to_string(),
                });
            }
            let key = self.parse_string()?;
            if map.contains_key(&key) {
                return Err(WireContractError::DuplicateObjectKey {
                    path: here.to_string(),
                    key,
                });
            }
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(WireContractError::SyntaxError {
                    offset: self.pos,
                    detail: "expected ':' after object key".to_string(),
                });
            }
            self.pos += 1;
            let child_path = format!("{path}/{key}");
            let value = if key == "display_f64" && self.allow_display_f64 {
                self.parse_display_f64(&child_path)?
            } else {
                self.parse_value(&child_path)?
            };
            map.insert(key, value);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                    self.skip_ws();
                    if self.peek() == Some(b'}') {
                        return Err(WireContractError::SyntaxError {
                            offset: self.pos,
                            detail: "trailing comma in object is forbidden".to_string(),
                        });
                    }
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                Some(other) => {
                    return Err(WireContractError::SyntaxError {
                        offset: self.pos,
                        detail: format!("expected ',' or '}}' in object, got 0x{other:02x}"),
                    });
                }
                None => return Err(WireContractError::UnexpectedEof),
            }
        }
        Ok(CanonicalValue::Object(map))
    }

    fn parse_array(&mut self, path: &str) -> Result<CanonicalValue, WireContractError> {
        self.pos += 1; // '['
        self.skip_ws();
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(CanonicalValue::Array(items));
        }
        let mut idx = 0usize;
        loop {
            let child_path = format!("{path}/{idx}");
            let val = self.parse_value(&child_path)?;
            items.push(val);
            idx += 1;
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                    self.skip_ws();
                    if self.peek() == Some(b']') {
                        return Err(WireContractError::SyntaxError {
                            offset: self.pos,
                            detail: "trailing comma in array is forbidden".to_string(),
                        });
                    }
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                Some(other) => {
                    return Err(WireContractError::SyntaxError {
                        offset: self.pos,
                        detail: format!("expected ',' or ']' in array, got 0x{other:02x}"),
                    });
                }
                None => return Err(WireContractError::UnexpectedEof),
            }
        }
        Ok(CanonicalValue::Array(items))
    }

    fn parse_string(&mut self) -> Result<String, WireContractError> {
        self.pos += 1; // '"'
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == b'"' {
                self.pos += 1;
                return Ok(out);
            }
            if b < 0x20 {
                return Err(WireContractError::UnescapedControlChar {
                    offset: self.pos,
                    byte: b,
                });
            }
            if b == b'\\' {
                let esc_offset = self.pos;
                self.pos += 1;
                let Some(esc) = self.peek() else {
                    return Err(WireContractError::UnexpectedEof);
                };
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let code_unit = self.read_hex_u16(esc_offset)?;
                        if (0xD800..=0xDBFF).contains(&code_unit) {
                            // High surrogate: must be immediately followed by \uDC00..=\uDFFF
                            if !self.bytes[self.pos..].starts_with(b"\\u") {
                                return Err(WireContractError::UnpairedSurrogate {
                                    offset: esc_offset,
                                    code_unit,
                                });
                            }
                            let low_offset = self.pos;
                            self.pos += 2; // skip "\u"
                            let low_unit = self.read_hex_u16(low_offset)?;
                            if !(0xDC00..=0xDFFF).contains(&low_unit) {
                                return Err(WireContractError::UnpairedSurrogate {
                                    offset: esc_offset,
                                    code_unit,
                                });
                            }
                            let scalar = 0x10000
                                + (((code_unit as u32 - 0xD800) << 10)
                                    | (low_unit as u32 - 0xDC00));
                            let ch = char::from_u32(scalar).ok_or(
                                WireContractError::UnpairedSurrogate {
                                    offset: esc_offset,
                                    code_unit,
                                },
                            )?;
                            out.push(ch);
                        } else if (0xDC00..=0xDFFF).contains(&code_unit) {
                            return Err(WireContractError::UnpairedSurrogate {
                                offset: esc_offset,
                                code_unit,
                            });
                        } else {
                            let ch = char::from_u32(code_unit as u32).ok_or(
                                WireContractError::UnpairedSurrogate {
                                    offset: esc_offset,
                                    code_unit,
                                },
                            )?;
                            out.push(ch);
                        }
                    }
                    other => {
                        return Err(WireContractError::InvalidStringEscape {
                            offset: esc_offset,
                            detail: format!("unknown escape '\\{}'", other as char),
                        });
                    }
                }
            } else {
                let s = std::str::from_utf8(&self.bytes[self.pos..])
                    .map_err(|_| WireContractError::InvalidUtf8)?;
                let ch = s.chars().next().ok_or(WireContractError::UnexpectedEof)?;
                out.push(ch);
                self.pos += ch.len_utf8();
            }
        }
        Err(WireContractError::UnexpectedEof)
    }

    fn read_hex_u16(&mut self, esc_offset: usize) -> Result<u16, WireContractError> {
        if self.pos + 4 > self.bytes.len() {
            return Err(WireContractError::InvalidStringEscape {
                offset: esc_offset,
                detail: "truncated \\uXXXX escape".to_string(),
            });
        }
        let slice = &self.bytes[self.pos..self.pos + 4];
        let hex_str = std::str::from_utf8(slice).map_err(|_| WireContractError::InvalidUtf8)?;
        if !hex_str.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(WireContractError::InvalidStringEscape {
                offset: esc_offset,
                detail: format!("non-hex \\u{hex_str} escape"),
            });
        }
        self.pos += 4;
        u16::from_str_radix(hex_str, 16).map_err(|_| WireContractError::InvalidStringEscape {
            offset: esc_offset,
            detail: format!("invalid \\u{hex_str} escape"),
        })
    }

    fn parse_number(&mut self, path: &str) -> Result<CanonicalValue, WireContractError> {
        let here = if path.is_empty() { "/" } else { path };
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        let digits_start = self.pos;
        while let Some(b'0'..=b'9') = self.peek() {
            self.pos += 1;
        }
        if self.pos == digits_start {
            return Err(WireContractError::SyntaxError {
                offset: start,
                detail: "expected digits in number".to_string(),
            });
        }
        // Check for fractional or exponent suffix
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            let frac_start = self.pos;
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
            if self.pos == frac_start {
                return Err(WireContractError::SyntaxError {
                    offset: start,
                    detail: "missing fractional digits after '.'".to_string(),
                });
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let exp_start = self.pos;
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
            if self.pos == exp_start {
                return Err(WireContractError::SyntaxError {
                    offset: start,
                    detail: "missing exponent digits".to_string(),
                });
            }
        }
        let raw_num = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| WireContractError::InvalidUtf8)?;
        if raw_num == "-0" {
            return Err(WireContractError::NegativeZeroInteger { offset: start });
        }
        let digits = &self.bytes[digits_start..self.pos];
        if !is_float && digits.len() > 1 && digits[0] == b'0' {
            return Err(WireContractError::LeadingZeroNumber { offset: start });
        }
        if is_float {
            return Err(WireContractError::RawFloatInIdentityPayload {
                path: here.to_string(),
            });
        }
        let parsed = raw_num
            .parse::<i64>()
            .map_err(|_| WireContractError::IntegerOverflow {
                path: here.to_string(),
                raw: raw_num.to_string(),
            })?;
        if !(SAFE_INT_MIN..=SAFE_INT_MAX).contains(&parsed) {
            return Err(WireContractError::IntegerOverflow {
                path: here.to_string(),
                raw: raw_num.to_string(),
            });
        }
        Ok(CanonicalValue::Int(parsed))
    }

    fn parse_display_f64(&mut self, path: &str) -> Result<CanonicalValue, WireContractError> {
        self.skip_ws();
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        let digits_start = self.pos;
        while let Some(b'0'..=b'9') = self.peek() {
            self.pos += 1;
        }
        if self.pos == digits_start {
            return Err(WireContractError::SyntaxError {
                offset: start,
                detail: format!("invalid display_f64 number at {path}"),
            });
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            let frac_start = self.pos;
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
            if self.pos == frac_start {
                return Err(WireContractError::SyntaxError {
                    offset: start,
                    detail: format!("invalid display_f64 fraction at {path}"),
                });
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let exp_start = self.pos;
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
            if self.pos == exp_start {
                return Err(WireContractError::SyntaxError {
                    offset: start,
                    detail: format!("invalid display_f64 exponent at {path}"),
                });
            }
        }
        let raw_num = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| WireContractError::InvalidUtf8)?;
        let val = raw_num
            .parse::<f64>()
            .map_err(|_| WireContractError::SyntaxError {
                offset: start,
                detail: format!("unparseable display_f64 {raw_num:?} at {path}"),
            })?;
        if !val.is_finite() {
            return Err(WireContractError::NonFiniteFloatBits {
                path: path.to_string(),
                hex_bits: raw_num.to_string(),
            });
        }
        Ok(CanonicalValue::String(raw_num.to_string()))
    }
}

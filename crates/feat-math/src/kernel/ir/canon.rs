//! I02 canonical JSON and length-prefixed identity projection, without a new Cargo edge.
use super::Result;
use crate::dsl::ParamValue;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use sha2::{Digest, Sha256};

pub const MAX_SAFE_INT: i64 = 9_007_199_254_740_991;
pub const MAX_BYTES: usize = 1_048_576;
pub const FORMULA: &str = "sha256(schema_utf8||0x00||u64_be(len)||canonical_payload_bytes)";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactF64 {
    pub f64_bits: String,
}
impl ExactF64 {
    pub fn new(value: f64) -> Result<Self> {
        if !value.is_finite() {
            return Err("nonfinite graph constant".into());
        }
        Ok(Self {
            f64_bits: format!("{:016x}", value.to_bits()),
        })
    }
    pub fn value(&self) -> Result<f64> {
        let bits = hex_u64(&self.f64_bits)?;
        let value = f64::from_bits(bits);
        if !value.is_finite() {
            return Err("nonfinite graph constant".into());
        }
        Ok(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Int(i64),
    Float(ExactF64),
    Bool(bool),
    Text(String),
}
impl Value {
    pub fn from_legacy(value: &ParamValue) -> Result<Self> {
        let result = match value {
            ParamValue::Int(v) => Self::Int(*v),
            ParamValue::Float(v) => Self::Float(ExactF64::new(*v)?),
            ParamValue::Bool(v) => Self::Bool(*v),
            ParamValue::Str(v) => Self::Text(v.clone()),
        };
        result.to_legacy()?;
        Ok(result)
    }
    pub fn to_legacy(&self) -> Result<ParamValue> {
        Ok(match self {
            Self::Int(v) if (-MAX_SAFE_INT..=MAX_SAFE_INT).contains(v) => ParamValue::Int(*v),
            Self::Int(_) => return Err("unsafe integer in graph parameter".into()),
            Self::Float(v) => ParamValue::Float(v.value()?),
            Self::Bool(v) => ParamValue::Bool(*v),
            Self::Text(v) if v.len() <= 4096 => ParamValue::Str(v.clone()),
            Self::Text(_) => return Err("graph parameter string bound".into()),
        })
    }
}

pub fn hex_u64(value: &str) -> Result<u64> {
    if value.len() != 16
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("expected 16 lowercase hex digits".into());
    }
    u64::from_str_radix(value, 16).map_err(|e| e.to_string())
}
pub fn digest_ref(value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        || value.bytes().all(|b| b == b'0')
    {
        return Err("expected nonzero 64-character lowercase digest".into());
    }
    Ok(())
}
pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn string(value: &str, out: &mut String) {
    use std::fmt::Write;
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 32 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// The serializer accepts only I02 safe integers and tagged finite floats.
/// Array/operand order and Unicode spelling are preserved exactly.
pub fn canonical_value(value: &Json) -> Result<Vec<u8>> {
    fn write(value: &Json, out: &mut String, depth: usize) -> Result<()> {
        if depth > 96 || out.len() > MAX_BYTES {
            return Err("canonical payload resource bound".into());
        }
        match value {
            Json::Null => out.push_str("null"),
            Json::Bool(v) => out.push_str(if *v { "true" } else { "false" }),
            Json::String(v) => string(v, out),
            Json::Number(v) => {
                let n = v
                    .as_i64()
                    .filter(|v| (-MAX_SAFE_INT..=MAX_SAFE_INT).contains(v))
                    .ok_or("raw float or unsafe integer in identity payload")?;
                out.push_str(&n.to_string());
            }
            Json::Array(values) => {
                out.push('[');
                for (i, v) in values.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write(v, out, depth + 1)?;
                }
                out.push(']');
            }
            Json::Object(values) => {
                if values.keys().any(|k| {
                    [
                        "own_id",
                        "signature",
                        "later_git_commit",
                        "later_assessment_id",
                    ]
                    .contains(&k.as_str())
                }) {
                    return Err("self-referential identity field".into());
                }
                if let Some((key, bits, width)) = values
                    .get("f64_bits")
                    .map(|v| ("f64_bits", v, 16))
                    .or_else(|| values.get("f32_bits").map(|v| ("f32_bits", v, 8)))
                {
                    if values.keys().any(|k| k != key && k != "display_f64") {
                        return Err("unknown tagged-float field".into());
                    }
                    let bits = bits.as_str().ok_or("float bits must be text")?;
                    if bits.len() != width
                        || !bits
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                    {
                        return Err("invalid exact float bits".into());
                    }
                    let number = u64::from_str_radix(bits, 16).map_err(|e| e.to_string())?;
                    let finite = if width == 16 {
                        f64::from_bits(number).is_finite()
                    } else {
                        f32::from_bits(number as u32).is_finite()
                    };
                    if !finite {
                        return Err("nonfinite exact float".into());
                    }
                    if values
                        .get("display_f64")
                        .is_some_and(|v| !v.as_f64().is_some_and(f64::is_finite))
                    {
                        return Err("invalid float display".into());
                    }
                    out.push('{');
                    string(key, out);
                    out.push(':');
                    string(bits, out);
                    out.push('}');
                } else {
                    out.push('{');
                    let mut entries: Vec<_> = values.iter().collect();
                    entries.sort_by(|a, b| a.0.cmp(b.0));
                    for (i, (k, v)) in entries.into_iter().enumerate() {
                        if i > 0 {
                            out.push(',');
                        }
                        string(k, out);
                        out.push(':');
                        write(v, out, depth + 1)?;
                    }
                    out.push('}');
                }
            }
        }
        if out.len() > MAX_BYTES {
            return Err("canonical payload byte bound".into());
        }
        Ok(())
    }
    let mut out = String::new();
    write(value, &mut out, 0)?;
    Ok(out.into_bytes())
}
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    canonical_value(&serde_json::to_value(value).map_err(|e| e.to_string())?)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramIdV2 {
    pub schema_id: String,
    pub sha256: String,
    pub formula: String,
}
impl ProgramIdV2 {
    pub fn compute(schema_id: &str, payload: &[u8]) -> Result<Self> {
        if schema_id.is_empty() || schema_id.contains('\0') || payload.len() > MAX_BYTES {
            return Err("invalid identity framing".into());
        }
        let mut hasher = Sha256::new();
        hasher.update(schema_id.as_bytes());
        hasher.update([0]);
        hasher.update((payload.len() as u64).to_be_bytes());
        hasher.update(payload);
        Ok(Self {
            schema_id: schema_id.into(),
            sha256: format!("{:x}", hasher.finalize()),
            formula: FORMULA.into(),
        })
    }
}

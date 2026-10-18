use super::ResearchCapsuleV2;
use sha2::{Digest, Sha256};
use std::fmt::Write;

pub const CAPSULE_BYTE_LIMIT: usize = 12 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapsuleBytes {
    pub canonical_json: Vec<u8>,
    pub sha256: String,
}
#[derive(Debug, PartialEq, Eq)]
pub struct RenderError(String);
impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for RenderError {}

pub fn render_capsule(c: &ResearchCapsuleV2) -> Result<CapsuleBytes, RenderError> {
    let mut s = String::new();
    write!(s,"{{\"schema_version\":2,\"round_id\":{},\"source_commit\":{},\"stage\":{},\"outcome\":{},\"assertion_supported\":{},\"denominator\":{{",q(&c.round_id),opt(&c.source_commit),q(c.stage.as_str()),q(&c.outcome),c.assertion_supported).map_err(err)?;
    for (i, (k, v)) in c.denominator.iter().enumerate() {
        if i > 0 {
            s.push(',')
        }
        write!(s, "{}:{}", q(k), v).map_err(err)?;
    }
    s.push_str("},\"facts\":");
    arr(&mut s, &c.facts)?;
    s.push_str(",\"inference\":");
    arr(&mut s, &c.inference)?;
    s.push_str(",\"limitations\":");
    arr(&mut s, &c.limitations)?;
    s.push_str(",\"evidence\":[");
    for (i, e) in c.evidence.iter().enumerate() {
        if i > 0 {
            s.push(',')
        }
        write!(s,"{{\"uri\":{},\"sha256\":{},\"version\":{},\"unit\":{},\"numerator\":{},\"denominator\":{},\"estimate\":{},\"ci_low\":{},\"ci_high\":{},\"status\":{},\"estimator\":{},\"protocol\":{}}}",q(&e.locator.uri),q(&e.locator.sha256),q(&e.locator.version),q(&e.unit),e.numerator,e.denominator,num(e.estimate),num(e.ci_low),num(e.ci_high),q(&e.status),q(&e.estimator),q(&e.protocol)).map_err(err)?;
    }
    s.push_str("]}");
    let bytes = s.into_bytes();
    if bytes.len() > CAPSULE_BYTE_LIMIT {
        return Err(RenderError(format!(
            "capsule is {} bytes; limit is {CAPSULE_BYTE_LIMIT}",
            bytes.len()
        )));
    }
    let hash = format!("{:x}", Sha256::digest(&bytes));
    Ok(CapsuleBytes {
        canonical_json: bytes,
        sha256: hash,
    })
}
fn arr(s: &mut String, a: &[String]) -> Result<(), RenderError> {
    s.push('[');
    for (i, v) in a.iter().enumerate() {
        if i > 0 {
            s.push(',')
        }
        s.push_str(&q(v));
    }
    s.push(']');
    Ok(())
}
fn q(v: &str) -> String {
    let mut s = String::from("\"");
    for c in v.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            c if c < ' ' => {
                write!(s, "\\u{:04x}", c as u32).ok();
            }
            c => s.push(c),
        }
    }
    s.push('"');
    s
}
fn opt(v: &Option<String>) -> String {
    v.as_deref().map(q).unwrap_or_else(|| "null".into())
}
fn num(v: Option<f64>) -> String {
    v.filter(|x| x.is_finite())
        .map(|x| format!("{x}"))
        .unwrap_or_else(|| "null".into())
}
fn err(e: std::fmt::Error) -> RenderError {
    RenderError(e.to_string())
}

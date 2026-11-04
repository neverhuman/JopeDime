//! Authenticated grown-family (component) assignment and its verifier.
//!
//! Native port of the retired `crates/jope-engine/policy/family_growth_policy.py`
//! (`assign_new_components` / `verify_assignment`, archived at `8a55c4fef`; see
//! `docs/archive/python-archive-manifest.json`). `families::freeze_growth` used to hash-check
//! that helper against the compiled release and execute it with `/usr/bin/python3`; the compiled
//! release now *is* the verifier, so no helper file in the workspace is consulted at all.
//!
//! The rules, their order and every refusal message are the Python ones, so a receipt accepted
//! or refused there is accepted or refused here for the same reason. Python's own semantics are
//! reproduced where the policy relies on them: truthiness, `==` across ints/floats/bools, `in`
//! on lists/strings/dicts, iteration of dicts/strings, set de-duplication and `str.strip()`.
//! Differential proof: `tests/families.rs` replays the outputs captured from the Python module
//! before it was deleted (`tests/fixtures/family-growth/`) on the existing fixtures and on a
//! fixed-seed random scenario set.
//!
//! Deliberate, fail-closed differences (inputs Python accepted that are refused here): JSON
//! Python's `json` module reads but RFC 8259 does not (`NaN`, `Infinity`, UTF-16/32, a BOM);
//! integers outside `i64::MIN..=u64::MAX`, up to 4300 digits (`serde_json` reads them as the
//! nearest double, so they are refused rather than rounded; beyond 4300 digits Python 3.12 refused
//! them too); a float literal beyond the range of a double (`1e400`, which Python read as an
//! infinity and `serde_json` refuses);
//! non-ASCII reviewer names (Python's `str.isalnum` admitted them; the reviewer key is a path
//! component); paths spelled with exactly two leading slashes, which Python's `pathlib`
//! keeps as a distinct root; and a policy or reviewer key whose resolved name is too long to
//! resolve but lies inside the root (Python built the name and read the file; the custody check
//! here cannot establish custody without resolving it, so it refuses as an escape). Custody that
//! cannot be established at all (an `lstat` or resolution error) is never taken as custody kept.
//!
//! Growth publishes membership only: the assignment never applies contexts, never opens a final
//! test and never makes anything execution eligible.

use crate::manifest::{hash, SEED};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

pub const ASSIGNMENT_SCHEMA: &str = "parallel.family_growth_assignment.v3";
pub const POLICY_SCHEMA: &str = "parallel.family_growth_policy.v3";
pub const POLICY_PATH: &str = "manifests/family-growth-policy.json";
pub const REVIEWER_KEYS: &str = "credentials/provenance-review";
/// Domain separator of the reviewer HMAC over a decision receipt's SHA-256.
pub const SIGNATURE_DOMAIN: &str = "parallel.family_growth.v3:";
const GRAPH_SCHEMA: &str = "parallel.component_graph.v1";
const REGISTRY_SCHEMA: &str = "parallel.family_registry.v1";
const DECISION_SCHEMA: &str = "parallel.provenance_decision.v3";

/// Why an assignment was not produced or not verified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GrowthError {
    /// A policy refusal. Python raised `ValueError` with exactly this message.
    Refused(String),
    /// Malformed input or an I/O failure. Python raised some other exception
    /// (`KeyError`, `TypeError`, `AttributeError`, `OSError`, ...).
    Malformed(String),
}

impl fmt::Display for GrowthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GrowthError::Refused(message) => write!(f, "{message}"),
            GrowthError::Malformed(message) => write!(f, "malformed growth evidence: {message}"),
        }
    }
}

impl std::error::Error for GrowthError {}

type Outcome<T> = std::result::Result<T, GrowthError>;

fn refused(message: impl Into<String>) -> GrowthError {
    GrowthError::Refused(message.into())
}

fn malformed(message: impl Into<String>) -> GrowthError {
    GrowthError::Malformed(message.into())
}

// ---- Python value semantics ------------------------------------------------------------------

/// Python truthiness of a JSON value.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => match (n.as_i64(), n.as_u64(), n.as_f64()) {
            (Some(i), _, _) => i != 0,
            (_, Some(u), _) => u != 0,
            (_, _, Some(f)) => f != 0.0,
            _ => true,
        },
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// A number as Python compares it: `True == 1 == 1.0`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Num {
    Int(i128),
    Float(f64),
}

fn number(value: &Value) -> Option<Num> {
    match value {
        Value::Bool(b) => Some(Num::Int(i128::from(*b))),
        Value::Number(n) => Some(if let Some(i) = n.as_i64() {
            Num::Int(i128::from(i))
        } else if let Some(u) = n.as_u64() {
            Num::Int(i128::from(u))
        } else {
            let f = n.as_f64()?;
            if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e38 {
                Num::Int(f as i128)
            } else {
                Num::Float(f)
            }
        }),
        _ => None,
    }
}

/// Python `==` between two JSON values.
pub fn py_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| py_eq(v, w)))
        }
        _ => match (number(a), number(b)) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        },
    }
}

/// A hashable JSON scalar as a Python `set`/`dict` key (`1`, `1.0` and `True` are one key).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Key {
    Null,
    Int(i128),
    Float(u64),
    Str(String),
}

fn key(value: &Value) -> Outcome<Key> {
    Ok(match value {
        Value::Null => Key::Null,
        Value::String(s) => Key::Str(s.clone()),
        Value::Array(_) | Value::Object(_) => {
            return Err(malformed("unhashable list or object used as a key"))
        }
        _ => match number(value) {
            Some(Num::Int(i)) => Key::Int(i),
            Some(Num::Float(f)) => Key::Float(f.to_bits()),
            None => return Err(malformed("unhashable value")),
        },
    })
}

/// `iter(value)`: list elements, dict keys, or the characters of a string.
fn iterate(value: &Value) -> Outcome<Vec<Value>> {
    match value {
        Value::Array(items) => Ok(items.clone()),
        Value::Object(map) => Ok(map.keys().cloned().map(Value::String).collect()),
        Value::String(s) => Ok(s.chars().map(|c| Value::String(c.to_string())).collect()),
        _ => Err(malformed("value is not iterable")),
    }
}

/// `len(value)` for the iterable shapes.
fn length(value: &Value) -> Outcome<usize> {
    match value {
        Value::Array(items) => Ok(items.len()),
        Value::Object(map) => Ok(map.len()),
        Value::String(s) => Ok(s.chars().count()),
        _ => Err(malformed("value has no length")),
    }
}

/// `item in container` for the container shapes Python allows.
fn contains(container: &Value, item: &Value) -> Outcome<bool> {
    match container {
        Value::Array(items) => Ok(items.iter().any(|x| py_eq(x, item))),
        Value::Object(map) => {
            key(item)?;
            Ok(item.as_str().is_some_and(|k| map.contains_key(k)))
        }
        Value::String(haystack) => match item {
            Value::String(needle) => Ok(haystack.contains(needle.as_str())),
            _ => Err(malformed("'in <string>' requires a string")),
        },
        _ => Err(malformed("argument is not iterable")),
    }
}

/// `dict.get(name)`; a missing key is `None`.
fn get<'a>(object: &'a Map<String, Value>, name: &str) -> &'a Value {
    object.get(name).unwrap_or(&Value::Null)
}

fn object<'a>(value: &'a Value, what: &str) -> Outcome<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| malformed(format!("{what} is not an object")))
}

fn index<'a>(value: &'a Value, name: &str, what: &str) -> Outcome<&'a Value> {
    object(value, what)?
        .get(name)
        .ok_or_else(|| malformed(format!("{what} has no {name:?}")))
}

/// Python's `str.isspace` set, which adds U+001C..U+001F to Unicode `White_Space`.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn py_strip(s: &str) -> &str {
    s.trim_matches(py_space)
}

// ---- digests ---------------------------------------------------------------------------------

/// HMAC-SHA256 (RFC 2104). Hand-written over `sha2` so the engine takes no new dependency;
/// checked against the RFC 4231 vectors in the unit tests.
pub fn hmac_sha256_hex(key: &[u8], message: &[u8]) -> String {
    const BLOCK: usize = 64;
    let mut block = [0u8; BLOCK];
    if key.len() > BLOCK {
        block[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(block.map(|b| b ^ 0x36));
    inner.update(message);
    let mut outer = Sha256::new();
    outer.update(block.map(|b| b ^ 0x5c));
    outer.update(inner.finalize());
    hex::encode(outer.finalize())
}

/// `hmac.compare_digest` on two ASCII strings: no early exit on the first differing byte.
fn compare_digest(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut differ = u8::from(a.len() != b.len());
    for (i, x) in a.iter().enumerate() {
        differ |= x ^ b.get(i).copied().unwrap_or(!x);
    }
    differ == 0
}

/// `stable_assignment_v2`: the 80/20 bucket on the canonical component id; no sealed test.
pub fn stable_assignment_v2(component: &str) -> (String, u64, &'static str) {
    let digest = hash(format!("{SEED}:component:{component}").as_bytes());
    let bucket = u64::from_str_radix(&digest[..16], 16).expect("SHA-256 hex prefix") % 1_000_000;
    let partition = if bucket < 800_000 {
        "meta_train"
    } else {
        "development_validation"
    };
    (digest, bucket, partition)
}

// ---- paths -----------------------------------------------------------------------------------

fn path_text(value: &Value, what: &str) -> Outcome<PathBuf> {
    let text = value
        .as_str()
        .ok_or_else(|| malformed(format!("{what} is not a string path")))?;
    if text.starts_with("//") && !text.starts_with("///") {
        return Err(refused(format!(
            "{what} spelled with a '//' root is not supported"
        )));
    }
    Ok(PathBuf::from(if text.is_empty() { "." } else { text }))
}

fn resolve_strict(path: &Path) -> Outcome<PathBuf> {
    path.canonicalize()
        .map_err(|e| malformed(format!("resolve {}: {e}", path.display())))
}

/// `path.is_symlink() or not path.resolve().is_relative_to(root)`, failing closed: custody that
/// cannot be established is never taken as custody kept.
///
/// - The path itself cannot be inspected (`lstat` fails: missing, a loop, a directory that may not
///   be searched): malformed, as Python raised there or on the read that followed
///   (`PermissionError`, `FileNotFoundError`, pathlib's `RuntimeError` for a loop).
/// - A symlink escapes.
/// - It resolves: it escapes unless the resolved path is inside `root`.
/// - It cannot be resolved because the resolved name is too long (`ENAMETOOLONG`): it escapes.
///   Python's non-strict `resolve()` built that name anyway and found it outside the root (a
///   symlink out of the workspace to a deep directory); inside the root it would have been read,
///   so refusing it is stricter there.
/// - Any other resolution error: malformed.
fn escapes(path: &Path, root: &Path) -> Outcome<bool> {
    const ENAMETOOLONG: i32 = 36; // Linux
    let custody = |e: std::io::Error| malformed(format!("custody of {}: {e}", path.display()));
    if std::fs::symlink_metadata(path)
        .map_err(custody)?
        .file_type()
        .is_symlink()
    {
        return Ok(true);
    }
    match path.canonicalize() {
        Ok(resolved) => Ok(!resolved.starts_with(root)),
        Err(e) if e.raw_os_error() == Some(ENAMETOOLONG) => Ok(true),
        Err(e) => Err(custody(e)),
    }
}

fn read(path: &Path) -> Outcome<Vec<u8>> {
    std::fs::read(path).map_err(|e| malformed(format!("read {}: {e}", path.display())))
}

fn parse(bytes: &[u8], what: &str) -> Outcome<Value> {
    if let Some(why) = wide_integer(bytes) {
        return Err(malformed(format!("parse {what}: {why}")));
    }
    serde_json::from_slice(bytes).map_err(|e| malformed(format!("parse {what}: {e}")))
}

/// Why the first integer literal outside `i64::MIN..=u64::MAX` cannot be read, or `None`.
/// `serde_json` reads such a literal as the nearest double, so two integers Python kept distinct
/// would compare equal here and an integer would become another one: the verifier refuses it
/// instead of rounding it (stricter; Python read it exactly). More than 4300 digits refuse in
/// CPython 3.12's words, as `json.loads` refused them. A literal with a fraction or an exponent
/// is a float, and text inside a string is not a number; anything malformed is left to
/// `serde_json`, which refuses it.
fn wide_integer(bytes: &[u8]) -> Option<String> {
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    i += if bytes[i] == b'\\' { 2 } else { 1 };
                }
                i += 1;
            }
            b'-' | b'0'..=b'9' => {
                let start = i;
                while i < bytes.len()
                    && matches!(bytes[i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
                {
                    i += 1;
                }
                let token = &bytes[start..i];
                let digits = token.strip_prefix(b"-").unwrap_or(token);
                if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
                    continue;
                }
                if digits.len() > 4300 {
                    return Some(format!(
                        "Exceeds the limit (4300 digits) for integer string conversion: value has {} digits; use sys.set_int_max_str_digits() to increase the limit",
                        digits.len()
                    ));
                }
                let text = std::str::from_utf8(token).expect("ASCII digits");
                let fits = if token.len() > digits.len() {
                    text.parse::<i64>().is_ok()
                } else {
                    text.parse::<u64>().is_ok()
                };
                if !fits {
                    return Some(format!(
                        "integer {text} is outside 64 bits, which is all this reader holds exactly; it is refused rather than rounded to a double (Python kept it exact)"
                    ));
                }
            }
            _ => i += 1,
        }
    }
    None
}

/// Read durable evidence named by `{path, sha256}` inside `root` and return its bytes.
fn artifact(root: &Path, reference: &Value) -> Outcome<Vec<u8>> {
    let Some(fields) = reference
        .as_object()
        .filter(|o| o.len() == 2 && o.contains_key("path") && o.contains_key("sha256"))
    else {
        return Err(refused("durable evidence path and hash required"));
    };
    let mut path = path_text(&fields["path"], "evidence path")?;
    if !path.is_absolute() {
        path = root.join(path);
    }
    let path = resolve_strict(&path)?;
    if !path.starts_with(root) {
        return Err(refused("growth evidence escapes workspace"));
    }
    let data = read(&path)?;
    if !py_eq(&Value::String(hash(&data)), &fields["sha256"]) {
        return Err(refused("growth evidence content changed"));
    }
    Ok(data)
}

mod policy;
pub use policy::{assign_new_components, verify_assignment, verify_assignment_file, Request};

#[cfg(test)]
#[path = "family_growth/tests.rs"]
mod tests;

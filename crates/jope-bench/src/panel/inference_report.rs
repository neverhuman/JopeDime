//! Read-only GPU timing-study reporting for v2 and v3 journals. Missing, corrupt and unsupported
//! are explicit: an absent attempt is unexecuted, never zero latency.
//!
//! Receipts are judged exactly as the retired Python reporter judged them, including its
//! equality and truthiness rules. JSON keeps its file member order and the summary is written
//! with Python `json.dumps(indent=2)` spelling (ASCII escapes, `repr` floats), so the report
//! bytes do not depend on the implementation language. A receipt member whose JSON type the
//! checks cannot inspect fails the report, as it did before, rather than being guessed at.

use anyhow::{bail, ensure, Context, Result};
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use sha2::{Digest, Sha256};
use std::{
    cmp::Ordering,
    fmt::{self, Write as _},
    fs,
    path::{Path, PathBuf},
};

const SUPPORTED: [&str; 2] = ["jope.inference-benchmark/v3", "jope.inference-benchmark/v2"];
/// Direct-request census of the v2 study when a plan predates an explicit census.
const DEFAULT_CENSUS: [(&str, u64); 4] = [("warm", 10), ("cold", 1), ("warmup", 2), ("profile", 1)];
const QUALIFICATION_ERRORS: [&str; 7] = [
    "hyperion",
    "foundation",
    "target",
    "context",
    "policy_logits",
    "reward_and_runtime",
    "cache_full_prefix",
];
const PREDICTION_BACKENDS: [&str; 3] = ["python", "rust", "cpp17"];
const BATCH_ROWS: [i128; 3] = [1, 32, 1024];
const REQUEST_COLUMNS: [&str; 6] = [
    "dataset",
    "mode",
    "seed",
    "elapsed_seconds",
    "accepted",
    "failure",
];
const BOUNDARY: &str = "Fitting data to first prediction, including preprocessing, encoders, policy, fitting, emission and compilation. Backends emitted sequentially; later milestones include earlier backend work. No LLM speed comparison.";

// ---------------------------------------------------------------------------------------------
// Order-preserving JSON with Python's comparison rules.

/// A JSON value that keeps member order, as Python's `json` module does.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Json {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Object),
}

/// Members in first-insertion order; a repeated key keeps its first position and last value.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Object(pub(super) Vec<(String, Json)>);

static NULL: Json = Json::Null;
static ZERO: Json = Json::Int(0);
pub(super) static EMPTY_ARRAY: Json = Json::Array(Vec::new());
pub(super) static EMPTY_OBJECT: Json = Json::Object(Object(Vec::new()));
static EMPTY_STRING: Json = Json::Str(String::new());

impl Object {
    pub(super) fn get(&self, key: &str) -> Option<&Json> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub(super) fn set(&mut self, key: &str, value: Json) {
        match self.0.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value,
            None => self.0.push((key.to_owned(), value)),
        }
    }
}

impl From<&str> for Json {
    fn from(value: &str) -> Self {
        Json::Str(value.to_owned())
    }
}
impl From<String> for Json {
    fn from(value: String) -> Self {
        Json::Str(value)
    }
}
impl From<bool> for Json {
    fn from(value: bool) -> Self {
        Json::Bool(value)
    }
}
impl From<u64> for Json {
    fn from(value: u64) -> Self {
        Json::Int(value.into())
    }
}
impl From<usize> for Json {
    fn from(value: usize) -> Self {
        Json::Int(value as i128)
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct JsonVisitor;
        impl<'de> Visitor<'de> for JsonVisitor {
            type Value = Json;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Json, E> {
                Ok(Json::Bool(v))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Json, E> {
                Ok(Json::Int(v.into()))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Json, E> {
                Ok(Json::Int(v.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Json, E> {
                if v.is_finite() {
                    Ok(Json::Float(v))
                } else {
                    Err(E::custom("nonfinite number"))
                }
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Json, E> {
                Ok(Json::Str(v.to_owned()))
            }
            fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Json, E> {
                Ok(Json::Str(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(Json::Array(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
                let mut members = Object::default();
                while let Some((key, value)) = map.next_entry::<String, Json>()? {
                    members.set(&key, value);
                }
                Ok(Json::Object(members))
            }
        }
        deserializer.deserialize_any(JsonVisitor)
    }
}

fn members<const N: usize>(entries: [(&str, Json); N]) -> Object {
    Object(
        entries
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
    )
}
fn object<const N: usize>(entries: [(&str, Json); N]) -> Json {
    Json::Object(members(entries))
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// `json.loads` of a file, or an empty object when it cannot be read or parsed.
fn read(path: &Path) -> Json {
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_else(|| EMPTY_OBJECT.clone())
}

fn kind(value: &Json) -> &'static str {
    match value {
        Json::Null => "null",
        Json::Bool(_) => "bool",
        Json::Int(_) => "int",
        Json::Float(_) => "float",
        Json::Str(_) => "string",
        Json::Array(_) => "array",
        Json::Object(_) => "object",
    }
}

#[derive(Clone, Copy, Debug)]
enum Number {
    Int(i128),
    Float(f64),
}

impl Number {
    fn to_f64(self) -> f64 {
        match self {
            Number::Int(i) => i as f64,
            Number::Float(f) => f,
        }
    }
    fn is_finite(self) -> bool {
        match self {
            Number::Int(_) => true,
            Number::Float(f) => f.is_finite(),
        }
    }
}

/// `isinstance(value, (int, float))`; a bool is an int.
fn number(value: &Json) -> Option<Number> {
    match value {
        Json::Bool(b) => Some(Number::Int(i128::from(*b))),
        Json::Int(i) => Some(Number::Int(*i)),
        Json::Float(f) => Some(Number::Float(*f)),
        _ => None,
    }
}

/// Exact numeric comparison across ints and floats, as Python compares them.
fn compare(a: Number, b: Number) -> Option<Ordering> {
    match (a, b) {
        (Number::Int(x), Number::Int(y)) => Some(x.cmp(&y)),
        (Number::Float(x), Number::Float(y)) => x.partial_cmp(&y),
        (Number::Int(x), Number::Float(y)) => int_float(x, y),
        (Number::Float(x), Number::Int(y)) => int_float(y, x).map(Ordering::reverse),
    }
}
fn int_float(i: i128, f: f64) -> Option<Ordering> {
    const LIMIT: f64 = 1.701_411_834_604_692_3e38; // 2^127
    if f.is_nan() {
        return None;
    }
    if f >= LIMIT {
        return Some(Ordering::Less);
    }
    if f < -LIMIT {
        return Some(Ordering::Greater);
    }
    let whole = f.trunc();
    Some(match i.cmp(&(whole as i128)) {
        Ordering::Equal if f > whole => Ordering::Less,
        Ordering::Equal if f < whole => Ordering::Greater,
        other => other,
    })
}
fn at_least_zero(n: Number) -> bool {
    compare(n, Number::Int(0)).is_some_and(Ordering::is_ge)
}
fn above_zero(n: Number) -> bool {
    compare(n, Number::Int(0)).is_some_and(Ordering::is_gt)
}

/// Python `==` between JSON values: numbers (and bools) by value, containers structurally,
/// objects regardless of member order.
pub(super) fn py_eq(a: &Json, b: &Json) -> bool {
    if let (Some(x), Some(y)) = (number(a), number(b)) {
        return compare(x, y) == Some(Ordering::Equal);
    }
    match (a, b) {
        (Json::Null, Json::Null) => true,
        (Json::Str(x), Json::Str(y)) => x == y,
        (Json::Array(x), Json::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q))
        }
        (Json::Object(x), Json::Object(y)) => {
            x.0.len() == y.0.len()
                && x.0
                    .iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| py_eq(v, w)))
        }
        _ => false,
    }
}
pub(super) fn is(value: &Json, text: &str) -> bool {
    matches!(value, Json::Str(s) if s == text)
}
/// `value is True`.
fn is_true(value: Option<&Json>) -> bool {
    matches!(value, Some(Json::Bool(true)))
}
fn truthy(value: &Json) -> bool {
    match value {
        Json::Null => false,
        Json::Bool(b) => *b,
        Json::Int(i) => *i != 0,
        Json::Float(f) => *f != 0.0,
        Json::Str(s) => !s.is_empty(),
        Json::Array(items) => !items.is_empty(),
        Json::Object(o) => !o.0.is_empty(),
    }
}
fn py_len(value: &Json) -> Result<usize> {
    Ok(match value {
        Json::Str(s) => s.chars().count(),
        Json::Array(items) => items.len(),
        Json::Object(o) => o.0.len(),
        other => bail!("TypeError: a JSON {} has no len()", kind(other)),
    })
}
/// Python iteration: array items, object keys or string characters.
pub(super) fn py_iter(value: &Json) -> Result<Vec<Json>> {
    Ok(match value {
        Json::Str(s) => s.chars().map(|c| Json::Str(c.to_string())).collect(),
        Json::Array(items) => items.clone(),
        Json::Object(o) => o.0.iter().map(|(k, _)| Json::Str(k.clone())).collect(),
        other => bail!("TypeError: a JSON {} is not iterable", kind(other)),
    })
}
/// `needle in container`.
fn py_contains(container: &Json, needle: &str) -> Result<bool> {
    Ok(match container {
        Json::Str(s) => s.contains(needle),
        Json::Array(items) => items.iter().any(|x| is(x, needle)),
        Json::Object(o) => o.get(needle).is_some(),
        other => bail!(
            "TypeError: argument of type {} is not iterable",
            kind(other)
        ),
    })
}
/// `all(needle in x for x in container)`.
fn all_contain(container: &Json, needle: &str) -> Result<bool> {
    for item in py_iter(container)? {
        if !py_contains(&item, needle)? {
            return Ok(false);
        }
    }
    Ok(true)
}
fn hashable(value: &Json) -> Result<()> {
    ensure!(
        !matches!(value, Json::Array(_) | Json::Object(_)),
        "TypeError: unhashable JSON {}",
        kind(value)
    );
    Ok(())
}
/// Distinct members of a Python set built from the values.
fn distinct(values: Vec<Json>) -> Result<Vec<Json>> {
    let mut seen: Vec<Json> = Vec::new();
    for value in values {
        hashable(&value)?;
        if !seen.iter().any(|s| py_eq(s, &value)) {
            seen.push(value);
        }
    }
    Ok(seen)
}
/// Python `<` ordering where it is defined.
fn py_order(a: &Json, b: &Json) -> Result<Ordering> {
    if let (Some(x), Some(y)) = (number(a), number(b)) {
        return compare(x, y).context("unordered numbers");
    }
    match (a, b) {
        (Json::Str(x), Json::Str(y)) => Ok(x.cmp(y)),
        (Json::Array(x), Json::Array(y)) => {
            for (p, q) in x.iter().zip(y) {
                if !py_eq(p, q) {
                    return py_order(p, q);
                }
            }
            Ok(x.len().cmp(&y.len()))
        }
        _ => bail!(
            "TypeError: '<' is not supported between JSON {} and {}",
            kind(a),
            kind(b)
        ),
    }
}

// ---------------------------------------------------------------------------------------------
// Member access with Python's failure modes.

/// `value.get(key)`: only an object has `get`; anything else fails the report.
fn get<'a>(value: &'a Json, key: &str) -> Result<Option<&'a Json>> {
    match value {
        Json::Object(o) => Ok(o.get(key)),
        other => bail!("AttributeError: a JSON {} has no get({key:?})", kind(other)),
    }
}
/// `value.get(key, default)`.
pub(super) fn get_or<'a>(value: &'a Json, key: &str, default: &'a Json) -> Result<&'a Json> {
    Ok(get(value, key)?.unwrap_or(default))
}
/// `value.get(key)`, absent as `None`.
pub(super) fn at<'a>(value: &'a Json, key: &str) -> Result<&'a Json> {
    get_or(value, key, &NULL)
}
/// `value[key]`: a missing member or a non-object fails the report.
pub(super) fn index<'a>(value: &'a Json, key: &str) -> Result<&'a Json> {
    match value {
        Json::Object(o) => o.get(key).with_context(|| format!("KeyError: {key:?}")),
        other => bail!(
            "TypeError: a JSON {} is not subscriptable by {key:?}",
            kind(other)
        ),
    }
}
/// `value[key]` inside a `try` that treats KeyError and TypeError as absence.
fn lookup<'a>(value: &'a Json, key: &str) -> Option<&'a Json> {
    match value {
        Json::Object(o) => o.get(key),
        _ => None,
    }
}
/// `value[key] = item` on a receipt.
fn assign(value: &mut Json, key: &str, item: Json) -> Result<()> {
    match value {
        Json::Object(o) => {
            o.set(key, item);
            Ok(())
        }
        other => bail!(
            "TypeError: a JSON {} does not support item assignment",
            kind(other)
        ),
    }
}

// ---------------------------------------------------------------------------------------------
// Artifact identity and exact-bit parity.

/// A pin names a readable file whose SHA-256 matches; anything else is unverified.
fn verified(pin: &Json) -> Result<bool> {
    let Some(Json::Str(path)) = lookup(pin, "path") else {
        return Ok(false);
    };
    ensure!(
        !path.contains('\0'),
        "ValueError: embedded null byte in {path:?}"
    );
    let Ok(bytes) = fs::read(path) else {
        return Ok(false);
    };
    Ok(
        matches!(lookup(pin, "sha256"), Some(Json::Str(expected)) if *expected == sha256_hex(&bytes)),
    )
}

/// The JSON behind an already verified pin, `read(Path(owner[key]['path']))`.
fn pinned(owner: &Json, key: &str) -> Result<Json> {
    match lookup(index(owner, key)?, "path") {
        Some(Json::Str(path)) => Ok(read(Path::new(path))),
        _ => bail!("TypeError: pin {key:?} has no path"),
    }
}

/// `struct.pack('>d', value)`: the IEEE-754 bits of a number; anything else fails the report.
fn pack(value: &Json) -> Result<u64> {
    match number(value) {
        Some(Number::Int(i)) => Ok((i as f64).to_bits()),
        Some(Number::Float(f)) => Ok(f.to_bits()),
        None => bail!(
            "struct.error: required argument is not a float (JSON {})",
            kind(value)
        ),
    }
}

/// A backend's predictions equal its qualification plan's expected values bit for bit, and
/// the plan is bound to the pinned bundle and its backend.
fn parity_verified(b: &Json) -> Result<bool> {
    if !verified(get_or(b, "qualification_plan", &EMPTY_OBJECT)?)?
        || !verified(get_or(b, "prediction", &EMPTY_OBJECT)?)?
    {
        return Ok(false);
    }
    let q = pinned(b, "qualification_plan")?;
    let actual = pinned(b, "prediction")?;
    let expected = get_or(&q, "expected", &EMPTY_OBJECT)?.clone();
    if !verified(get_or(b, "bundle", &EMPTY_OBJECT)?)? {
        return Ok(false);
    }
    let bundle = pinned(b, "bundle")?;
    let identity = Json::Str(sha256_hex(dumps(&bundle, Style::IDENTITY)?.as_bytes()));
    if !py_eq(at(&q, "bundle_sha256")?, &identity) {
        return Ok(false);
    }
    let backend = at(&q, "backend")?;
    if !py_eq(
        backend,
        at(get_or(&bundle, "manifest", &EMPTY_OBJECT)?, "backend")?,
    ) {
        return Ok(false);
    }
    // KeyError and TypeError in this comparison mean "not verified".
    let (Some(kind_actual), Some(kind_expected)) =
        (lookup(&actual, "kind"), lookup(&expected, "kind"))
    else {
        return Ok(false);
    };
    if !(py_eq(kind_actual, kind_expected) && is(kind_expected, "values")) {
        return Ok(false);
    }
    let (Some(data_actual), Some(data_expected)) =
        (lookup(&actual, "data"), lookup(&expected, "data"))
    else {
        return Ok(false);
    };
    let (Ok(n_actual), Ok(n_expected)) = (py_len(data_actual), py_len(data_expected)) else {
        return Ok(false);
    };
    if n_actual != n_expected || n_actual == 0 {
        return Ok(false);
    }
    for (a, e) in py_iter(data_actual)?
        .iter()
        .zip(py_iter(data_expected)?.iter())
    {
        let bits = pack(a)?;
        if bits != pack(e)? {
            return Ok(false);
        }
    }
    Ok(true)
}

// ---------------------------------------------------------------------------------------------
// Python `json.dumps`, `repr`/`str` and `csv` spelling.

#[derive(Clone, Copy)]
pub(super) struct Style {
    /// `ensure_ascii`.
    ascii: bool,
    /// `indent`; `None` writes one line.
    indent: Option<usize>,
    /// On one line, the default `(', ', ': ')` separators rather than `(',', ':')`.
    spaced: bool,
}

impl Style {
    /// `json.dumps(value, indent=2, allow_nan=False)`.
    pub(super) const REPORT: Style = Style {
        ascii: true,
        indent: Some(2),
        spaced: false,
    };
    /// `json.dumps(value, ensure_ascii=False, separators=(',', ':'))`.
    pub(super) const IDENTITY: Style = Style {
        ascii: false,
        indent: None,
        spaced: false,
    };
    /// `json.dumps(value)`.
    #[cfg_attr(not(any(test, feature = "inference-cuda")), allow(dead_code))]
    pub(super) const DEFAULT: Style = Style {
        ascii: true,
        indent: None,
        spaced: true,
    };
    fn item_separator(self) -> &'static str {
        if self.indent.is_none() && self.spaced {
            ", "
        } else {
            ","
        }
    }
    fn key_separator(self) -> &'static str {
        if self.indent.is_some() || self.spaced {
            ": "
        } else {
            ":"
        }
    }
    fn newline(self, level: usize, out: &mut String) {
        if let Some(width) = self.indent {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', width * level));
        }
    }
}

pub(super) fn dumps(value: &Json, style: Style) -> Result<String> {
    let mut out = String::new();
    dump(value, style, 0, &mut out)?;
    Ok(out)
}

fn dump(value: &Json, style: Style, level: usize, out: &mut String) -> Result<()> {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Int(i) => out.push_str(&i.to_string()),
        Json::Float(f) => {
            ensure!(
                f.is_finite(),
                "ValueError: Out of range float values are not JSON compliant"
            );
            out.push_str(&float_repr(*f));
        }
        Json::Str(s) => quote(s, style.ascii, out),
        Json::Array(items) if items.is_empty() => out.push_str("[]"),
        Json::Object(o) if o.0.is_empty() => out.push_str("{}"),
        Json::Array(items) => {
            out.push('[');
            for (position, item) in items.iter().enumerate() {
                if position > 0 {
                    out.push_str(style.item_separator());
                }
                style.newline(level + 1, out);
                dump(item, style, level + 1, out)?;
            }
            style.newline(level, out);
            out.push(']');
        }
        Json::Object(o) => {
            out.push('{');
            for (position, (key, item)) in o.0.iter().enumerate() {
                if position > 0 {
                    out.push_str(style.item_separator());
                }
                style.newline(level + 1, out);
                quote(key, style.ascii, out);
                out.push_str(style.key_separator());
                dump(item, style, level + 1, out)?;
            }
            style.newline(level, out);
            out.push('}');
        }
    }
    Ok(())
}

/// A JSON string literal, escaped as Python's encoder escapes it.
fn quote(text: &str, ascii: bool, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if u32::from(c) < 0x20 || (ascii && !(' '..='~').contains(&c)) => {
                for unit in c.encode_utf16(&mut [0; 2]) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: shortest round-trip digits, positional for exponents -4..=15.
pub(super) fn float_repr(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let scientific = format!("{:e}", x.abs());
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("LowerExp always writes an exponent");
    let exponent: i32 = exponent.parse().expect("LowerExp exponent is an integer");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let count = digits.len() as i32;
    let point = exponent + 1;
    let mut out = String::from(if x < 0.0 { "-" } else { "" });
    if -4 < point && point <= 16 {
        if point <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-point) as usize));
            out.push_str(&digits);
        } else if point < count {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        } else {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', (point - count) as usize));
            out.push_str(".0");
        }
    } else {
        out.push_str(&digits[..1]);
        if count > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(
            out,
            "e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        );
    }
    out
}

/// Python `str()` of a JSON value, as f-strings and `csv` render it.
pub(super) fn py_str(value: &Json) -> String {
    match value {
        Json::Str(s) => s.clone(),
        other => py_repr(other),
    }
}

pub(super) fn py_repr(value: &Json) -> String {
    match value {
        Json::Null => "None".into(),
        Json::Bool(true) => "True".into(),
        Json::Bool(false) => "False".into(),
        Json::Int(i) => i.to_string(),
        Json::Float(f) => float_repr(*f),
        Json::Str(s) => str_repr(s),
        Json::Array(items) => format!(
            "[{}]",
            items.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        Json::Object(o) => format!(
            "{{{}}}",
            o.0.iter()
                .map(|(k, v)| format!("{}: {}", str_repr(k), py_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Python `repr(str)`. Printability of non-ASCII characters follows the common Unicode
/// separator, format, control and private-use ranges rather than the full database.
pub(super) fn str_repr(text: &str) -> String {
    let delimiter = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::from(delimiter);
    for c in text.chars() {
        let code = u32::from(c);
        match c {
            c if c == delimiter || c == '\\' => {
                out.push('\\');
                out.push(c);
            }
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ if code < 0x20 || code == 0x7f => {
                let _ = write!(out, "\\x{code:02x}");
            }
            _ if code < 0x7f || printable(code) => out.push(c),
            _ if code <= 0xff => {
                let _ = write!(out, "\\x{code:02x}");
            }
            _ if code <= 0xffff => {
                let _ = write!(out, "\\u{code:04x}");
            }
            _ => {
                let _ = write!(out, "\\U{code:08x}");
            }
        }
    }
    out.push(delimiter);
    out
}

fn printable(code: u32) -> bool {
    !matches!(
        code,
        0x80..=0xa0
            | 0xad
            | 0x061c
            | 0x1680
            | 0x180e
            | 0x2000..=0x200f
            | 0x2028..=0x202f
            | 0x205f..=0x206f
            | 0x3000
            | 0xd800..=0xf8ff
            | 0xfeff
            | 0xfff0..=0xfffb
            | 0xfffe..=0xffff
            | 0xe0000..=0xe0fff
            | 0xf0000..=0x10ffff
    )
}

/// One `csv.writer` row (excel dialect, minimal quoting, CRLF). `None` is an empty field.
fn csv_row(fields: &[&Json], out: &mut String) {
    for (position, field) in fields.iter().enumerate() {
        if position > 0 {
            out.push(',');
        }
        let text = match field {
            Json::Null => String::new(),
            other => py_str(other),
        };
        if text.contains([',', '"', '\r', '\n']) {
            out.push('"');
            out.push_str(&text.replace('"', "\"\""));
            out.push('"');
        } else {
            out.push_str(&text);
        }
    }
    out.push_str("\r\n");
}

// ---------------------------------------------------------------------------------------------
// `statistics` with exact rational rounding.

/// An unsigned fixed-point magnitude in units of 2^-1074, wide enough for any finite f64 or
/// JSON integer and sums of them.
#[derive(Clone)]
struct Wide([u64; 40]);

impl Wide {
    const SCALE: u32 = 1074;
    fn zero() -> Self {
        Wide([0; 40])
    }
    fn add_shifted(&mut self, value: u64, shift: u32) {
        let (limb, offset) = ((shift / 64) as usize, shift % 64);
        let low = value << offset;
        let high = if offset == 0 {
            0
        } else {
            value >> (64 - offset)
        };
        self.add_at(limb, low);
        self.add_at(limb + 1, high);
    }
    fn add_at(&mut self, mut limb: usize, value: u64) {
        let mut carry = value;
        while carry != 0 {
            let (sum, overflow) = self.0[limb].overflowing_add(carry);
            self.0[limb] = sum;
            carry = u64::from(overflow);
            limb += 1;
        }
    }
    fn add_float(&mut self, x: f64) {
        let bits = x.abs().to_bits();
        let exponent = (bits >> 52) as u32;
        let fraction = bits & ((1 << 52) - 1);
        if exponent == 0 {
            self.add_shifted(fraction, 0);
        } else {
            self.add_shifted(fraction | (1 << 52), exponent - 1);
        }
    }
    fn add_int(&mut self, magnitude: u128) {
        self.add_shifted(magnitude as u64, Self::SCALE);
        self.add_shifted((magnitude >> 64) as u64, Self::SCALE + 64);
    }
    fn compare(&self, other: &Wide) -> Ordering {
        self.0.iter().rev().cmp(other.0.iter().rev())
    }
    /// `self - other`, for `self >= other`.
    fn minus(&self, other: &Wide) -> Wide {
        let mut result = Wide::zero();
        let mut borrow = false;
        for (limb, (a, b)) in self.0.iter().zip(&other.0).enumerate() {
            let (difference, first) = a.overflowing_sub(*b);
            let (difference, second) = difference.overflowing_sub(u64::from(borrow));
            result.0[limb] = difference;
            borrow = first || second;
        }
        result
    }
    /// Divide in place; returns the remainder.
    fn divide(&mut self, divisor: u64) -> u64 {
        let mut remainder = 0_u128;
        for limb in self.0.iter_mut().rev() {
            let current = (remainder << 64) | u128::from(*limb);
            *limb = (current / u128::from(divisor)) as u64;
            remainder = current % u128::from(divisor);
        }
        remainder as u64
    }
    fn bits(&self) -> u32 {
        self.0
            .iter()
            .rposition(|limb| *limb != 0)
            .map_or(0, |top| top as u32 * 64 + 64 - self.0[top].leading_zeros())
    }
    fn bit(&self, position: u32) -> bool {
        (self.0[(position / 64) as usize] >> (position % 64)) & 1 == 1
    }
    fn any_below(&self, position: u32) -> bool {
        let (limb, offset) = ((position / 64) as usize, position % 64);
        self.0[..limb].iter().any(|l| *l != 0) || self.0[limb] & ((1 << offset) - 1) != 0
    }
    /// The 64 bits starting at `shift`.
    fn window(&self, shift: u32) -> u64 {
        let (limb, offset) = ((shift / 64) as usize, shift % 64);
        let low = self.0[limb] >> offset;
        if offset == 0 || limb + 1 >= self.0.len() {
            low
        } else {
            low | (self.0[limb + 1] << (64 - offset))
        }
    }
    /// `self / divisor` × 2^-1074, rounded half to even.
    fn quotient(mut self, divisor: u64) -> f64 {
        let remainder = self.divide(divisor);
        let length = self.bits();
        let (mantissa, shift, up) = if length <= 53 {
            let mantissa = self.window(0);
            let twice = 2 * u128::from(remainder);
            let half = u128::from(divisor);
            (
                mantissa,
                0,
                twice > half || (twice == half && mantissa & 1 == 1),
            )
        } else {
            let shift = length - 53;
            let mantissa = self.window(shift) & ((1 << 53) - 1);
            let sticky = self.any_below(shift - 1) || remainder != 0;
            (
                mantissa,
                shift,
                self.bit(shift - 1) && (sticky || mantissa & 1 == 1),
            )
        };
        (mantissa + u64::from(up)) as f64 * power_of_two(shift as i32 - Self::SCALE as i32)
    }
}

fn power_of_two(exponent: i32) -> f64 {
    if exponent >= -1022 {
        f64::from_bits(((exponent + 1023) as u64) << 52)
    } else {
        f64::from_bits(1 << (exponent + 1074))
    }
}

/// The correctly rounded quotient of an exact signed sum and a count.
fn rational(values: &[Number], count: u64) -> f64 {
    let (mut positive, mut negative) = (Wide::zero(), Wide::zero());
    for value in values {
        match *value {
            Number::Int(i) if i < 0 => negative.add_int(i.unsigned_abs()),
            Number::Int(i) => positive.add_int(i.unsigned_abs()),
            Number::Float(f) if f.is_sign_negative() => negative.add_float(f),
            Number::Float(f) => positive.add_float(f),
        }
    }
    match positive.compare(&negative) {
        Ordering::Less => -negative.minus(&positive).quotient(count),
        _ => positive.minus(&negative).quotient(count),
    }
}

/// `statistics.mean`: an exact mean, an int when every value is an int and it divides.
fn mean(values: &[Json]) -> Result<Json> {
    let numbers = values
        .iter()
        .map(|v| number(v).context("TypeError: mean of a non-number"))
        .collect::<Result<Vec<_>>>()?;
    let count = numbers.len() as i128;
    if numbers.iter().all(|n| matches!(n, Number::Int(_))) {
        let total: i128 = numbers
            .iter()
            .map(|n| match n {
                Number::Int(i) => *i,
                Number::Float(_) => unreachable!(),
            })
            .sum();
        if total % count == 0 {
            return Ok(Json::Int(total / count));
        }
    }
    Ok(Json::Float(rational(&numbers, count as u64)))
}

/// `statistics.median` of an even-length sorted pair: `(a + b) / 2`.
fn midpoint(a: &Json, b: &Json) -> Result<Json> {
    let (Some(x), Some(y)) = (number(a), number(b)) else {
        bail!("TypeError: median of a non-number");
    };
    Ok(Json::Float(match (x, y) {
        (Number::Int(i), Number::Int(j)) => rational(&[Number::Int(i + j)], 2),
        _ => (x.to_f64() + y.to_f64()) / 2.0,
    }))
}

/// Sample count, median, 95th-percentile order statistic and mean, or nulls without samples.
fn distribution(values: &[Json]) -> Result<Json> {
    let mut sorted = values.to_vec();
    let mut failure = None;
    sorted.sort_by(|a, b| {
        py_order(a, b).unwrap_or_else(|error| {
            failure.get_or_insert(error);
            Ordering::Equal
        })
    });
    if let Some(error) = failure {
        return Err(error);
    }
    let n = sorted.len();
    if n == 0 {
        return Ok(object([
            ("n", 0_usize.into()),
            ("median_seconds", Json::Null),
            ("p95_seconds", Json::Null),
            ("mean_seconds", Json::Null),
        ]));
    }
    let median = if n % 2 == 1 {
        sorted[n / 2].clone()
    } else {
        midpoint(&sorted[n / 2 - 1], &sorted[n / 2])?
    };
    let p95 = sorted[(0.95 * n as f64).ceil() as usize - 1].clone();
    Ok(object([
        ("n", n.into()),
        ("median_seconds", median),
        ("p95_seconds", p95),
        ("mean_seconds", mean(&sorted)?),
    ]))
}

// ---------------------------------------------------------------------------------------------
// Journal inspection.

/// `Path.exists()`: absence (including a non-directory parent or a symlink loop) is false; any
/// other failure fails the report.
pub(super) fn exists(path: &Path) -> Result<bool> {
    match fs::metadata(path) {
        Ok(_) => Ok(true),
        // ENOENT, EBADF, ENOTDIR and ELOOP, and paths the OS cannot encode.
        Err(error)
            if matches!(error.raw_os_error(), Some(2 | 9 | 20 | 40))
                || error.kind() == std::io::ErrorKind::InvalidInput =>
        {
            Ok(false)
        }
        Err(error) => Err(error).with_context(|| format!("stat {}", path.display())),
    }
}

/// Directory entries whose names satisfy `matches`, or none when `dir` cannot be listed.
fn entries(dir: &Path, matches: impl Fn(&str) -> bool) -> Vec<(String, PathBuf)> {
    if !dir.is_dir() {
        return Vec::new();
    }
    let Ok(listing) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let Ok(listing) = listing.collect::<std::io::Result<Vec<_>>>() else {
        return Vec::new();
    };
    listing
        .into_iter()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            matches(&name).then(|| (name, entry.path()))
        })
        .collect()
}

/// `journal.glob(prefix + '*/receipt.json')`, as (directory name, receipt path).
fn receipts(journal: &Path, prefix: &str) -> Vec<(String, PathBuf)> {
    entries(journal, |name| name.starts_with(prefix))
        .into_iter()
        .filter(|(_, dir)| dir.is_dir())
        .filter(|(_, dir)| !entries(dir, |name| name == "receipt.json").is_empty())
        .map(|(name, dir)| (name, dir.join("receipt.json")))
        .collect()
}

fn journal_path(journal: &Path, name: String) -> PathBuf {
    journal.join(name)
}

// ---------------------------------------------------------------------------------------------
// The report.

/// The CUDA qualification receipt is bound to this plan and every case passed within tolerance.
fn qualified(plan: &Json, q: &Json) -> Result<bool> {
    let cases = get_or(q, "cases", &EMPTY_ARRAY)?;
    let experiment = || index(plan, "experiment");
    let accepted = py_eq(at(q, "schema")?, index(plan, "schema")?)
        && is_true(get(q, "passed")?)
        && py_eq(
            at(q, "plan_identity")?,
            index(experiment()?, "methods_sha256")?,
        )
        && py_eq(
            at(q, "binary_sha256")?,
            index(index(plan, "binary")?, "sha256")?,
        )
        && py_eq(at(q, "source_sha")?, index(experiment()?, "source_sha")?)
        && py_len(cases)? == 8
        && distinct(
            py_iter(cases)?
                .iter()
                .map(|c| at(c, "case").cloned())
                .collect::<Result<_>>()?,
        )?
        .len()
            == 8
        && py_len(get_or(q, "devices", &EMPTY_ARRAY)?)? == 3
        && all_contain(index(q, "devices")?, "Cuda")?
        && py_contains(get_or(q, "policy_device", &EMPTY_STRING)?, "Cuda")?;
    if !accepted {
        return Ok(false);
    }
    for case in py_iter(cases)? {
        if !is_true(get(&case, "pass")?) {
            return Ok(false);
        }
        let names = distinct(py_iter(get_or(&case, "errors", &EMPTY_OBJECT)?)?)?;
        if !(names.len() == QUALIFICATION_ERRORS.len()
            && QUALIFICATION_ERRORS
                .iter()
                .all(|key| names.iter().any(|name| is(name, key))))
        {
            return Ok(false);
        }
        let Json::Object(errors) = index(&case, "errors")? else {
            bail!("AttributeError: qualification errors have no values()");
        };
        let within = errors.0.iter().all(|(_, error)| {
            number(error).is_some_and(|n| {
                n.is_finite()
                    && at_least_zero(n)
                    && compare(n, Number::Float(3e-4)).is_some_and(Ordering::is_le)
            })
        });
        if !within {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Planned direct-request slots per mode, frozen before execution; every slot needs one
/// terminal outcome. A plan without an explicit census uses the v2 study's.
fn census(plan: &Json) -> Result<(Json, Vec<(String, u64)>)> {
    let counts = match get(plan, "census")? {
        Some(counts) => counts.clone(),
        None => Json::Object(Object(
            DEFAULT_CENSUS
                .iter()
                .map(|(mode, count)| (mode.to_string(), Json::from(*count)))
                .collect(),
        )),
    };
    let modes = distinct(py_iter(&counts)?)?;
    let known = modes.len() == DEFAULT_CENSUS.len()
        && DEFAULT_CENSUS
            .iter()
            .all(|(mode, _)| modes.iter().any(|m| is(m, mode)));
    ensure!(known, "census modes or counts invalid");
    let Json::Object(members) = &counts else {
        bail!("AttributeError: census has no values()");
    };
    let mut planned = Vec::new();
    for (mode, count) in &members.0 {
        let count = match count {
            Json::Int(i) if *i >= 0 => u64::try_from(*i).context("census count overflow")?,
            Json::Bool(b) => u64::from(*b),
            _ => bail!("census modes or counts invalid"),
        };
        planned.push((mode.clone(), count));
    }
    Ok((counts, planned))
}

/// A whole-request receipt is bound to the plan, the slot, the CUDA devices and the timer.
fn request_valid(plan: &Json, d: &Json, r: &Json, mode: &str, seed: u64) -> Result<bool> {
    let experiment = || index(plan, "experiment");
    Ok(py_eq(at(r, "schema")?, index(plan, "schema")?)
        && py_eq(
            at(r, "plan_identity")?,
            index(experiment()?, "methods_sha256")?,
        )
        && py_eq(
            at(r, "binary_sha256")?,
            index(index(plan, "binary")?, "sha256")?,
        )
        && py_eq(at(r, "source_sha")?, index(experiment()?, "source_sha")?)
        && py_eq(at(r, "dataset")?, index(d, "id")?)
        && py_eq(at(r, "seed")?, &Json::from(seed))
        && is(at(r, "mode")?, mode)
        && py_eq(at(r, "input_sha256")?, index(index(d, "raw")?, "sha256")?)
        && is(at(r, "selected_origin")?, "neural_policy")
        && is(at(r, "ranking_backend")?, "candle_cuda_float32")
        && py_len(get_or(r, "devices", &EMPTY_ARRAY)?)? == 3
        && all_contain(index(r, "devices")?, "Cuda")?
        && py_contains(get_or(r, "policy_device", &EMPTY_STRING)?, "Cuda")?
        && is_true(get(
            get_or(r, "export", &EMPTY_OBJECT)?,
            "validation_in_timer",
        )?))
}

/// A completed backend milestone with verified parity, artifacts, bundle and commands.
fn backend_good(b: &Json, t: &Json) -> Result<bool> {
    let qualification = || get_or(b, "qualification", &EMPTY_OBJECT);
    Ok(is(at(b, "status")?, "completed")
        && number(t).is_some_and(|n| n.is_finite() && above_zero(n))
        && parity_verified(b)?
        && {
            let values = get_or(qualification()?, "values", &ZERO)?;
            number(values)
                .map(above_zero)
                .with_context(|| format!("TypeError: '>' on a JSON {}", kind(values)))?
        }
        && py_eq(
            at(qualification()?, "maximum_absolute_error")?,
            &Json::Int(0),
        )
        && all_files_verified(get_or(b, "files", &EMPTY_OBJECT)?)?
        && truthy(at(b, "files")?)
        && verified(get_or(b, "bundle", &EMPTY_OBJECT)?)?
        && all_commands_succeeded(get_or(b, "commands", &EMPTY_ARRAY)?)?)
}

fn all_files_verified(files: &Json) -> Result<bool> {
    let Json::Object(files) = files else {
        bail!("AttributeError: a JSON {} has no values()", kind(files));
    };
    for (_, pin) in &files.0 {
        if !verified(pin)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn all_commands_succeeded(commands: &Json) -> Result<bool> {
    for command in py_iter(commands)? {
        if !py_eq(at(&command, "exit_code")?, &Json::Int(0)) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn all_verified(pins: &Json) -> Result<bool> {
    for pin in py_iter(pins)? {
        if !verified(&pin)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn any_parity(backends: &Json) -> Result<bool> {
    for backend in py_iter(backends)? {
        if parity_verified(&backend)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Every returned prediction equals the qualification plan's first expected value bit for bit.
/// KeyError and TypeError mean false; other failures fail the report, as before.
fn batch_matches(r: &Json, a: &Json, prediction_plan: &Json) -> Result<bool> {
    let Some(data) = lookup(a, "data") else {
        return Ok(false);
    };
    let Ok(rows) = py_len(data) else {
        return Ok(false);
    };
    let Some(batch_rows) = lookup(r, "batch_rows") else {
        return Ok(false);
    };
    if !py_eq(&Json::from(rows), batch_rows) {
        return Ok(false);
    }
    for x in py_iter(data)? {
        let bits = pack(&x)?;
        let Some(expected) = lookup(prediction_plan, "expected").and_then(|e| lookup(e, "data"))
        else {
            return Ok(false);
        };
        let first = match expected {
            Json::Array(items) => items.first().cloned().context("IndexError: empty data")?,
            Json::Str(text) => Json::Str(
                text.chars()
                    .next()
                    .context("IndexError: empty data")?
                    .to_string(),
            ),
            _ => return Ok(false),
        };
        if bits != pack(&first)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// A retained prediction-API receipt: bound to the plan, the warm export's verified bundle and
/// qualification, exact bits, a planned backend and batch size and ten finite samples.
fn prediction_valid(
    plan: &Json,
    inputs: &[Json],
    journal: &Path,
    qualification: bool,
    r: &Json,
) -> Result<bool> {
    let dataset = at(r, "dataset")?.clone();
    let mut d = None;
    for candidate in inputs {
        if py_eq(index(candidate, "id")?, &dataset) {
            d = Some(candidate);
            break;
        }
    }
    let experiment = || index(plan, "experiment");
    let mut valid = qualification
        && match d {
            None => false,
            Some(d) => {
                py_eq(at(r, "source_sha")?, index(experiment()?, "source_sha")?)
                    && py_eq(
                        at(r, "plan_identity")?,
                        index(experiment()?, "methods_sha256")?,
                    )
                    && py_eq(
                        at(r, "binary_sha256")?,
                        index(index(plan, "binary")?, "sha256")?,
                    )
                    && py_eq(at(r, "input_sha256")?, index(index(d, "raw")?, "sha256")?)
                    && verified(get_or(r, "source_qualification", &EMPTY_OBJECT)?)?
                    && verified(get_or(r, "prediction", &EMPTY_OBJECT)?)?
                    && is(at(r, "parity")?, "exact_bits")
            }
        };
    let source =
        read(&journal_path(journal, format!("warm-{}-0", py_str(&dataset))).join("receipt.json"));
    let mut entry = EMPTY_OBJECT.clone();
    for b in py_iter(get_or(
        get_or(&source, "export", &EMPTY_OBJECT)?,
        "backends",
        &EMPTY_ARRAY,
    )?)? {
        if py_eq(at(&b, "backend")?, at(r, "backend")?) {
            entry = b;
            break;
        }
    }
    valid = valid
        && parity_verified(&entry)?
        && py_eq(
            at(r, "bundle_sha256")?,
            at(get_or(&entry, "bundle", &EMPTY_OBJECT)?, "sha256")?,
        )
        && py_eq(
            at(r, "source_qualification")?,
            at(&entry, "qualification_plan")?,
        );
    if valid {
        let prediction_plan = pinned(r, "source_qualification")?;
        let a = pinned(r, "prediction")?;
        valid = batch_matches(r, &a, &prediction_plan)?;
    }
    // Timing and backend membership are part of receipt validity.
    let backend = at(r, "backend")?;
    let batch_rows = at(r, "batch_rows")?;
    let samples = get_or(r, "samples_seconds", &EMPTY_ARRAY)?;
    Ok(valid
        && PREDICTION_BACKENDS.iter().any(|name| is(backend, name))
        && BATCH_ROWS.iter().any(|n| py_eq(batch_rows, &Json::Int(*n)))
        && py_len(samples)? == 10
        && py_iter(samples)?
            .iter()
            .all(|x| number(x).is_some_and(|n| n.is_finite() && at_least_zero(n))))
}

/// Summarize a timing journal against its plan: the report and each direct-request record.
fn summarize(plan: &Json, journal: &Path) -> Result<(Json, Vec<Json>)> {
    let q = read(&journal.join("qualification.json"));
    let qualification = qualified(plan, &q)?;
    let launch = read(&journal.join("launch.json"));
    let exit_code = get(&launch, "exit_code")?;
    let controller = object([
        (
            "status",
            if exit_code.is_some_and(|code| py_eq(code, &ZERO)) {
                "completed"
            } else if exit_code.is_some_and(|code| *code != Json::Null) {
                "failed"
            } else {
                "unknown"
            }
            .into(),
        ),
        ("launch", launch.clone()),
        (
            "qualification_status",
            if qualification {
                "passed"
            } else if truthy(&q) {
                "failed"
            } else {
                "not_executed_or_interrupted"
            }
            .into(),
        ),
    ]);
    let (mut records, mut missing, mut failures) = (Vec::new(), Vec::new(), Vec::new());
    let mut backend_times: Vec<(Json, Vec<Json>)> = Vec::new();
    let (counts, modes) = census(plan)?;
    let inputs = py_iter(index(plan, "inputs")?)?;
    let mut slots = Vec::new();
    for d in &inputs {
        for (mode, count) in &modes {
            for seed in 0..*count {
                slots.push(format!("{mode}-{}-{seed}", py_str(index(d, "id")?)));
            }
        }
    }
    for d in &inputs {
        for (mode, count) in &modes {
            for seed in 0..*count {
                let id = index(d, "id")?;
                let slot = journal_path(journal, format!("{mode}-{}-{seed}", py_str(id)));
                let p = slot.join("receipt.json");
                if !exists(&p)? {
                    let status = if exists(&slot)? {
                        "unexecuted_or_interrupted"
                    } else {
                        "unexecuted"
                    };
                    missing.push(object([
                        ("dataset", id.clone()),
                        ("mode", mode.as_str().into()),
                        ("seed", seed.into()),
                        ("status", status.into()),
                    ]));
                    continue;
                }
                let mut r = read(&p);
                let mut valid = request_valid(plan, d, &r, mode, seed)?;
                let elapsed = at(&r, "elapsed_seconds")?;
                valid = valid && number(elapsed).is_some_and(|n| n.is_finite() && at_least_zero(n));
                let export = get_or(&r, "export", &EMPTY_OBJECT)?;
                let recipe = get_or(export, "recipe", &EMPTY_OBJECT)?;
                valid = valid
                    && py_eq(
                        at(recipe, "original_proposal")?,
                        at(&r, "selected_program")?,
                    )
                    && verified(get_or(export, "fitted_output", &EMPTY_OBJECT)?)?;
                let mut completions = Vec::new();
                for b in py_iter(get_or(export, "backends", &EMPTY_ARRAY)?)? {
                    let t = at(&b, "first_prediction_seconds")?;
                    if qualification && valid && backend_good(&b, t)? {
                        let backend = index(&b, "backend")?;
                        completions.push(backend.clone());
                        if mode == "warm" {
                            hashable(id)?;
                            hashable(backend)?;
                            let key = Json::Array(vec![id.clone(), backend.clone()]);
                            match backend_times.iter_mut().find(|(k, _)| py_eq(k, &key)) {
                                Some((_, times)) => times.push(t.clone()),
                                None => backend_times.push((key, vec![t.clone()])),
                            }
                        }
                    } else if !is(at(&b, "status")?, "unsupported") {
                        failures.push(object([
                            ("dataset", id.clone()),
                            ("mode", mode.as_str().into()),
                            ("seed", seed.into()),
                            ("backend", at(&b, "backend")?.clone()),
                            ("status", at(&b, "status")?.clone()),
                            (
                                "reason",
                                get_or(
                                    &b,
                                    "reason",
                                    &Json::from("receipt or artifact validation failed"),
                                )?
                                .clone(),
                            ),
                        ]));
                    }
                }
                let accepted = qualification
                    && valid
                    && is_true(get(&r, "success")?)
                    && ["python", "rust"]
                        .iter()
                        .all(|x| completions.iter().any(|c| is(c, x)));
                assign(&mut r, "accepted", accepted.into())?;
                if !accepted {
                    failures.push(object([
                        ("dataset", id.clone()),
                        ("mode", mode.as_str().into()),
                        ("seed", seed.into()),
                        (
                            "failure",
                            get_or(&r, "failure", &Json::from("required path incomplete"))?.clone(),
                        ),
                    ]));
                }
                records.push(r);
            }
        }
    }
    let mut warm = Vec::new();
    for r in &records {
        if is(index(r, "mode")?, "warm") && truthy(index(r, "accepted")?) {
            warm.push(r);
        }
    }
    let mut gp = Vec::new();
    for d in &inputs {
        if matches!(get(d, "gp")?, None | Some(Json::Null)) {
            continue;
        }
        for arm in ["neural", "generic"] {
            let id = index(d, "id")?;
            let p = journal_path(journal, format!("gp-{}", py_str(id)))
                .join(arm)
                .join("receipt.json");
            let mut r = read(&p);
            if !truthy(&r) {
                r = object([
                    ("dataset", id.clone()),
                    ("arm", arm.into()),
                    ("status", "unexecuted".into()),
                ]);
            }
            ensure!(
                matches!(r, Json::Object(_)),
                "TypeError: a JSON {} receipt does not support item assignment",
                kind(&r)
            );
            let experiment = || index(plan, "experiment");
            let outcome = || get_or(&r, "outcome", &EMPTY_OBJECT);
            let accepted = qualification
                && py_eq(at(&r, "source_sha")?, index(experiment()?, "source_sha")?)
                && py_eq(
                    at(&r, "plan_identity")?,
                    index(experiment()?, "methods_sha256")?,
                )
                && py_eq(
                    at(&r, "binary_sha256")?,
                    index(index(plan, "binary")?, "sha256")?,
                )
                && py_eq(at(&r, "input_sha256")?, index(index(d, "raw")?, "sha256")?)
                && verified(get_or(outcome()?, "search", &EMPTY_OBJECT)?)?
                && truthy(at(outcome()?, "execution_inventory")?)
                && all_verified(index(index(&r, "outcome")?, "execution_inventory")?)?;
            let export_qualified = accepted
                && any_parity(get_or(
                    get_or(outcome()?, "export", &EMPTY_OBJECT)?,
                    "backends",
                    &EMPTY_ARRAY,
                )?)?;
            assign(&mut r, "accepted", accepted.into())?;
            assign(&mut r, "export_qualified", export_qualified.into())?;
            gp.push(r);
        }
    }
    let mut prediction = Vec::new();
    let mut paths = receipts(journal, "prediction-");
    paths.sort();
    for (_, path) in paths {
        let mut r = read(&path);
        let valid = prediction_valid(plan, &inputs, journal, qualification, &r)?;
        let accepted = valid && is(at(&r, "status")?, "completed");
        assign(&mut r, "accepted", accepted.into())?;
        prediction.push(r);
    }
    let mut seen = Vec::new();
    for r in &prediction {
        let key = [
            at(r, "dataset")?.clone(),
            at(r, "backend")?.clone(),
            at(r, "batch_rows")?.clone(),
        ];
        for member in &key {
            hashable(member)?;
        }
        seen.push(key);
    }
    for d in &inputs {
        let id = index(d, "id")?;
        let source =
            read(&journal_path(journal, format!("warm-{}-0", py_str(id))).join("receipt.json"));
        let mut entries: Vec<(Json, Json)> = Vec::new();
        for b in py_iter(get_or(
            get_or(&source, "export", &EMPTY_OBJECT)?,
            "backends",
            &EMPTY_ARRAY,
        )?)? {
            let key = at(&b, "backend")?.clone();
            hashable(&key)?;
            match entries.iter_mut().find(|(k, _)| py_eq(k, &key)) {
                Some(slot) => slot.1 = b,
                None => entries.push((key, b)),
            }
        }
        for backend in PREDICTION_BACKENDS {
            for size in BATCH_ROWS {
                hashable(id)?;
                if seen.iter().any(|[dataset, name, rows]| {
                    py_eq(dataset, id) && is(name, backend) && py_eq(rows, &Json::Int(size))
                }) {
                    continue;
                }
                let entry = entries
                    .iter()
                    .find(|(k, _)| is(k, backend))
                    .map_or(&EMPTY_OBJECT, |(_, b)| b);
                let status = if is(at(entry, "status")?, "unsupported") {
                    "unsupported"
                } else if exists(&journal_path(
                    journal,
                    format!("prediction-{}-{backend}-{size}", py_str(id)),
                ))? {
                    "interrupted"
                } else {
                    "unexecuted"
                };
                prediction.push(object([
                    ("dataset", id.clone()),
                    ("backend", backend.into()),
                    ("batch_rows", Json::Int(size)),
                    ("status", status.into()),
                    ("accepted", false.into()),
                ]));
            }
        }
    }
    // Direct receipts outside the frozen census are never silently ignored.
    let mut extra: Vec<String> = DEFAULT_CENSUS
        .iter()
        .flat_map(|(mode, _)| receipts(journal, &format!("{mode}-")))
        .map(|(name, _)| name)
        .filter(|name| !slots.contains(name))
        .collect();
    extra.sort();
    let warm_count = match index(&counts, "warm")? {
        Json::Int(i) => *i,
        Json::Bool(b) => i128::from(*b),
        other => bail!("census warm count is a JSON {}", kind(other)),
    };
    let planned_warm = warm_count * inputs.len() as i128;
    let accepted = records
        .iter()
        .map(|r| index(r, "accepted").map(truthy))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .filter(|a| *a)
        .count();
    let reconciled = records.len() + missing.len() == slots.len();
    let reconciliation = object([
        ("planned", slots.len().into()),
        ("receipts", records.len().into()),
        ("accepted", accepted.into()),
        ("rejected", (records.len() - accepted).into()),
        ("missing", missing.len().into()),
        (
            "unplanned_receipts",
            Json::Array(extra.iter().map(|name| name.as_str().into()).collect()),
        ),
        ("reconciled", reconciled.into()),
        (
            "failure_or_missing_rate",
            if slots.is_empty() {
                Json::Null
            } else {
                Json::Float((slots.len() - accepted) as f64 / slots.len() as f64)
            },
        ),
    ]);
    let mut failure = None;
    backend_times.sort_by(|a, b| {
        py_order(&a.0, &b.0).unwrap_or_else(|error| {
            failure.get_or_insert(error);
            Ordering::Equal
        })
    });
    if let Some(error) = failure {
        return Err(error);
    }
    let mut milestones = Vec::new();
    for (key, times) in &backend_times {
        let Json::Array(key) = key else {
            unreachable!("milestone keys are pairs")
        };
        let Json::Object(statistics) = distribution(times)? else {
            unreachable!("a distribution is an object")
        };
        let mut milestone = members([("dataset", key[0].clone()), ("backend", key[1].clone())]);
        milestone.0.extend(statistics.0);
        milestones.push(Json::Object(milestone));
    }
    let mut cold = entries(journal, |name| {
        name.len() >= "cold-process-.json".len()
            && name.starts_with("cold-process-")
            && name.ends_with(".json")
    });
    cold.sort();
    let elapsed = warm
        .iter()
        .map(|r| index(r, "elapsed_seconds").cloned())
        .collect::<Result<Vec<_>>>()?;
    let summary = object([
        ("schema", index(plan, "schema")?.clone()),
        ("controller", controller),
        ("qualified", qualification.into()),
        ("qualification", q.clone()),
        (
            "publication_headline",
            (qualification
                && planned_warm > 0
                && warm.len() as i128 == planned_warm
                && extra.is_empty()
                && reconciled)
                .into(),
        ),
        ("completed_successfully", warm.len().into()),
        ("expected", Json::Int(planned_warm)),
        (
            "census",
            object([
                ("modes", counts.clone()),
                (
                    "source",
                    if get(plan, "census")?.is_some() {
                        "plan"
                    } else {
                        "default-v2"
                    }
                    .into(),
                ),
            ]),
        ),
        ("reconciliation", reconciliation),
        ("missing", Json::Array(missing)),
        ("failures", Json::Array(failures)),
        (
            "warm_cumulative_backend_milestones",
            Json::Array(milestones),
        ),
        (
            "cold_observations",
            Json::Array(cold.iter().map(|(_, path)| read(path)).collect()),
        ),
        ("full_gp", Json::Array(gp)),
        ("prediction", Json::Array(prediction)),
        ("window", read(&journal.join("window.json"))),
        (
            "warm_complete_request_distribution",
            distribution(&elapsed)?,
        ),
        ("boundary", BOUNDARY.into()),
    ]);
    Ok((summary, records))
}

/// Write `summary.json`, `requests.csv` and `methodology.md` for a journal. `plan` holds the
/// plan bytes whose pin the caller has already verified.
pub(super) fn write(plan: &[u8], journal: &Path, output: &Path) -> Result<()> {
    let plan: Json = serde_json::from_slice(plan).context("plan JSON")?;
    let schema = index(&plan, "schema")?;
    hashable(schema)?;
    ensure!(
        SUPPORTED.iter().any(|supported| is(schema, supported)),
        "receipt version unsupported"
    );
    let (summary, records) = summarize(&plan, journal)?;
    fs::create_dir_all(output)?;
    fs::write(
        output.join("summary.json"),
        dumps(&summary, Style::REPORT)? + "\n",
    )?;
    let mut requests = String::new();
    let header: Vec<Json> = REQUEST_COLUMNS.iter().map(|c| Json::from(*c)).collect();
    csv_row(&header.iter().collect::<Vec<_>>(), &mut requests);
    for r in &records {
        let fields = REQUEST_COLUMNS
            .iter()
            .map(|column| get_or(r, column, &EMPTY_STRING))
            .collect::<Result<Vec<_>>>()?;
        csv_row(&fields, &mut requests);
    }
    fs::write(output.join("requests.csv"), requests)?;
    let reconciliation = index(&summary, "reconciliation")?;
    let methodology = format!(
        "# GPU timing study\n\n{BOUNDARY}\n\nQualified: {}. Completed warm Python and Rust paths: {}/{}. Planned direct slots: {}, accepted {}, missing {}, unplanned receipts {}. Missing attempts are unexecuted, never zero latency. Cold observations are individual process wall times. Full GP uses five fitting-side OOF folds,96 candidate slots and120seconds per arm; exports occur after the winner freezes. Prediction API samples exclude process initialization, which is reported as a separate residual containing I/O and teardown.\n",
        py_str(index(&summary, "qualified")?),
        py_str(index(&summary, "completed_successfully")?),
        py_str(index(&summary, "expected")?),
        py_str(index(reconciliation, "planned")?),
        py_str(index(reconciliation, "accepted")?),
        py_str(index(reconciliation, "missing")?),
        py_len(index(reconciliation, "unplanned_receipts")?)?,
    );
    fs::write(output.join("methodology.md"), methodology)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Json {
        serde_json::from_str(text).unwrap()
    }
    fn member<'a>(value: &'a Json, key: &str) -> &'a Json {
        index(value, key).unwrap()
    }
    fn items(value: &Json) -> &Vec<Json> {
        match value {
            Json::Array(items) => items,
            other => panic!("not an array: {other:?}"),
        }
    }
    fn members_mut(value: &mut Json) -> &mut Object {
        match value {
            Json::Object(members) => members,
            other => panic!("not an object: {other:?}"),
        }
    }
    fn write_file(path: &Path, text: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }
    /// Write `text` under `root` and pin it.
    fn pin(root: &Path, name: &str, text: &str) -> Json {
        let path = root.join(name);
        write_file(&path, text);
        object([
            ("path", path.to_str().unwrap().into()),
            ("sha256", sha256_hex(text.as_bytes()).into()),
        ])
    }
    /// The v2 study's ten datasets with placeholder identities.
    fn plan() -> Json {
        let inputs: Vec<String> = (0..10)
            .map(|i| format!(r#"{{"id": "{i}", "raw": {{"sha256": "r"}}}}"#))
            .collect();
        parse(&format!(
            r#"{{"schema": "jope.inference-benchmark/v2", "experiment": {{"methods_sha256": "m", "source_sha": "s"}}, "binary": {{"sha256": "b"}}, "inputs": [{}]}}"#,
            inputs.join(", ")
        ))
    }
    fn summary(plan: &Json, journal: &Path) -> Json {
        summarize(plan, journal).unwrap().0
    }

    #[test]
    fn missing_never_claims_speed() {
        let root = tempfile::tempdir().unwrap();
        let s = summary(&plan(), root.path());
        assert_eq!(member(&s, "publication_headline"), &Json::Bool(false));
        let warm = items(member(&s, "missing"))
            .iter()
            .filter(|m| is(member(m, "mode"), "warm"))
            .count();
        assert_eq!(warm, 100);
        assert!(lookup(&s, "speedup").is_none());
    }

    #[test]
    fn controller_failure_before_qualification_is_explicit() {
        let root = tempfile::tempdir().unwrap();
        write_file(
            &root.path().join("launch.json"),
            r#"{"exit_code": 1, "reason": "bounded service failed"}"#,
        );
        let s = summary(&plan(), root.path());
        let controller = member(&s, "controller");
        assert!(is(member(controller, "status"), "failed"));
        assert!(is(
            member(controller, "qualification_status"),
            "not_executed_or_interrupted"
        ));
        assert_eq!(items(member(&s, "prediction")).len(), 90);
        assert_eq!(member(&s, "publication_headline"), &Json::Bool(false));
    }

    #[test]
    fn interrupted_direct_and_prediction_attempts_remain_missing() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("warm-0-0")).unwrap();
        fs::create_dir(root.path().join("prediction-0-rust-32")).unwrap();
        let s = summary(&plan(), root.path());
        let request = items(member(&s, "missing"))
            .iter()
            .find(|r| {
                is(member(r, "dataset"), "0")
                    && is(member(r, "mode"), "warm")
                    && py_eq(member(r, "seed"), &ZERO)
            })
            .unwrap();
        assert!(is(member(request, "status"), "unexecuted_or_interrupted"));
        let prediction = items(member(&s, "prediction"))
            .iter()
            .find(|r| {
                is(member(r, "dataset"), "0")
                    && is(member(r, "backend"), "rust")
                    && py_eq(member(r, "batch_rows"), &Json::Int(32))
            })
            .unwrap();
        assert!(is(member(prediction, "status"), "interrupted"));
        assert_eq!(member(prediction, "accepted"), &Json::Bool(false));
    }

    #[test]
    fn failed_and_wrong_device_remain_failures() {
        let root = tempfile::tempdir().unwrap();
        write_file(
            &root.path().join("warm-0-0/receipt.json"),
            r#"{"schema": "jope.inference-benchmark/v2", "dataset": "0", "seed": 0, "mode": "warm", "success": true, "elapsed_seconds": 1, "devices": ["Cpu", "Cpu", "Cpu"]}"#,
        );
        let s = summary(&plan(), root.path());
        assert_eq!(member(&s, "completed_successfully"), &Json::Int(0));
        assert!(!items(member(&s, "failures")).is_empty());
    }

    #[test]
    fn qualification_flag_without_evidence_refused() {
        let root = tempfile::tempdir().unwrap();
        let cases = [r#"{"pass": true}"#; 8].join(", ");
        write_file(
            &root.path().join("qualification.json"),
            &format!(r#"{{"passed": true, "cases": [{cases}]}}"#),
        );
        let s = summary(&plan(), root.path());
        assert_eq!(member(&s, "qualified"), &Json::Bool(false));
    }

    #[test]
    fn every_planned_slot_reconciles_to_one_outcome() {
        let root = tempfile::tempdir().unwrap();
        let s = summary(&plan(), root.path());
        let census = member(&s, "reconciliation");
        assert_eq!(
            [
                member(census, "planned"),
                member(census, "missing"),
                member(census, "receipts")
            ],
            [&Json::Int(140), &Json::Int(140), &Json::Int(0)]
        );
        assert_eq!(member(census, "reconciled"), &Json::Bool(true));
        assert_eq!(member(census, "failure_or_missing_rate"), &Json::Float(1.0));
        assert!(is(member(member(&s, "census"), "source"), "default-v2"));
    }

    #[test]
    fn unplanned_receipts_are_listed_and_block_the_headline() {
        let root = tempfile::tempdir().unwrap();
        for name in ["warm-0-15", "warm-ghost-0"] {
            write_file(
                &root.path().join(name).join("receipt.json"),
                r#"{"schema": "jope.inference-benchmark/v2", "success": true}"#,
            );
        }
        let s = summary(&plan(), root.path());
        assert_eq!(
            member(member(&s, "reconciliation"), "unplanned_receipts"),
            &parse(r#"["warm-0-15", "warm-ghost-0"]"#)
        );
        assert_eq!(member(&s, "publication_headline"), &Json::Bool(false));
    }

    #[test]
    fn census_comes_from_the_frozen_plan() {
        let root = tempfile::tempdir().unwrap();
        let mut plan = plan();
        let first = items(member(&plan, "inputs"))[0].clone();
        members_mut(&mut plan).set("inputs", Json::Array(vec![first]));
        members_mut(&mut plan).set(
            "census",
            parse(r#"{"warm": 2, "cold": 0, "warmup": 0, "profile": 0}"#),
        );
        let s = summary(&plan, root.path());
        assert_eq!(
            (
                member(&s, "expected"),
                member(member(&s, "reconciliation"), "planned")
            ),
            (&Json::Int(2), &Json::Int(2))
        );
        assert!(is(member(member(&s, "census"), "source"), "plan"));
        members_mut(&mut plan).set("census", parse(r#"{"warm": 2}"#));
        let error = summarize(&plan, root.path()).unwrap_err();
        assert!(format!("{error:#}").contains("census"), "{error:#}");
    }

    #[test]
    fn exact_parity_requires_matching_bundle() {
        let root = tempfile::tempdir().unwrap();
        let root = root.path();
        // Python `json.dumps` file spelling; the identity is the compact encoding in file order.
        let bundle = r#"{"manifest": {"backend": "python"}, "files": {"model.py": "content"}}"#;
        let identity =
            sha256_hex(br#"{"manifest":{"backend":"python"},"files":{"model.py":"content"}}"#);
        let values = r#"{"kind": "values", "data": [1.0]}"#;
        let plan = format!(
            r#"{{"expected": {values}, "backend": "python", "bundle_sha256": "{identity}"}}"#
        );
        let mut pins = object([
            ("bundle", pin(root, "bundle", bundle)),
            ("qualification_plan", pin(root, "qualification_plan", &plan)),
            ("prediction", pin(root, "prediction", values)),
        ]);
        assert!(parity_verified(&pins).unwrap());
        members_mut(&mut pins).set(
            "prediction",
            pin(root, "prediction", r#"{"kind": "values", "data": [2.0]}"#),
        );
        assert!(!parity_verified(&pins).unwrap());
    }

    #[test]
    fn corruption_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let asset = pin(root.path(), "asset", "original");
        assert!(verified(&asset).unwrap());
        fs::write(root.path().join("asset"), "corrupted").unwrap();
        assert!(!verified(&asset).unwrap());
    }

    /// Pins whose bundle binding is valid, so only the value comparison can refuse them.
    fn bound(root: &Path, expected: &str, actual: &str) -> Json {
        let bundle = r#"{"manifest": {"backend": "rust"}, "files": {"model.rs": "content"}}"#;
        let identity =
            sha256_hex(br#"{"manifest":{"backend":"rust"},"files":{"model.rs":"content"}}"#);
        let plan = format!(
            r#"{{"expected": {{"kind": "values", "data": {expected}}}, "backend": "rust", "bundle_sha256": "{identity}"}}"#
        );
        object([
            ("bundle", pin(root, "bound-bundle", bundle)),
            ("qualification_plan", pin(root, "bound-plan", &plan)),
            (
                "prediction",
                pin(
                    root,
                    "bound-prediction",
                    &format!(r#"{{"kind": "values", "data": {actual}}}"#),
                ),
            ),
        ])
    }

    #[test]
    fn parity_rejects_wrong_selected_prediction() {
        let root = tempfile::tempdir().unwrap();
        let root = root.path();
        let pins = object([
            (
                "qualification_plan",
                pin(
                    root,
                    "qualification_plan",
                    r#"{"expected": {"kind": "values", "data": [1.0]}}"#,
                ),
            ),
            (
                "prediction",
                pin(root, "prediction", r#"{"kind": "values", "data": [2.0]}"#),
            ),
        ]);
        assert!(!parity_verified(&pins).unwrap());
        // With a valid bundle binding, the value comparison alone decides.
        assert!(!parity_verified(&bound(root, "[1.0]", "[2.0]")).unwrap());
        assert!(parity_verified(&bound(root, "[1.0]", "[1.0]")).unwrap());
        assert!(parity_verified(&bound(root, "[1.0]", "[1]")).unwrap());
    }

    #[test]
    fn relative_tolerance_does_not_pass_exact_bit_policy() {
        let root = tempfile::tempdir().unwrap();
        let root = root.path();
        let pins = object([
            (
                "qualification_plan",
                pin(
                    root,
                    "qualification_plan",
                    r#"{"expected": {"kind": "values", "data": [0.0]}}"#,
                ),
            ),
            (
                "prediction",
                pin(root, "prediction", r#"{"kind": "values", "data": [-0.0]}"#),
            ),
        ]);
        assert!(!parity_verified(&pins).unwrap());
        assert!(!parity_verified(&bound(root, "[0.0]", "[-0.0]")).unwrap());
        assert!(!parity_verified(&bound(root, "[1.0]", "[1.0000000000000002]")).unwrap());
        assert!(parity_verified(&bound(root, "[-0.0]", "[-0.0]")).unwrap());
    }

    /// A complete, qualified single-dataset journal: every link of the acceptance chain holds.
    fn qualified_journal(root: &Path) -> Json {
        let cuda = r#"["Cuda(0)", "Cuda(1)", "Cuda(2)"]"#;
        let errors = QUALIFICATION_ERRORS
            .iter()
            .map(|k| format!(r#""{k}": 0.0001"#))
            .collect::<Vec<_>>()
            .join(", ");
        let cases = (0..8)
            .map(|i| format!(r#"{{"case": {i}, "pass": true, "errors": {{{errors}}}}}"#))
            .collect::<Vec<_>>()
            .join(", ");
        write_file(
            &root.join("qualification.json"),
            &format!(
                r#"{{"schema": "jope.inference-benchmark/v2", "passed": true, "plan_identity": "m", "binary_sha256": "b", "source_sha": "s", "cases": [{cases}], "devices": {cuda}, "policy_device": "Cuda(0)"}}"#
            ),
        );
        let artifacts = root.join("artifacts");
        let dump = |value: &Json| dumps(value, Style::IDENTITY).unwrap();
        let mut backends = Vec::new();
        let mut python_pins = None;
        for (name, seconds) in [("python", 0.5), ("rust", 0.75)] {
            let bundle = format!(
                r#"{{"manifest": {{"backend": "{name}"}}, "files": {{"model": "{name}"}}}}"#
            );
            let bundle_pin = pin(&artifacts, &format!("{name}-bundle.json"), &bundle);
            let identity = sha256_hex(dump(&parse(&bundle)).as_bytes());
            let plan_pin = pin(
                &artifacts,
                &format!("{name}-plan.json"),
                &format!(
                    r#"{{"expected": {{"kind": "values", "data": [1.5, -0.0]}}, "backend": "{name}", "bundle_sha256": "{identity}"}}"#
                ),
            );
            let prediction_pin = pin(
                &artifacts,
                &format!("{name}-prediction.json"),
                r#"{"kind": "values", "data": [1.5, -0.0]}"#,
            );
            let file_pin = pin(&artifacts, &format!("{name}-model"), name);
            if name == "python" {
                python_pins = Some((plan_pin.clone(), bundle_pin.clone()));
            }
            backends.push(format!(
                r#"{{"backend": "{name}", "status": "completed", "first_prediction_seconds": {seconds}, "qualification_plan": {}, "prediction": {}, "bundle": {}, "qualification": {{"values": 2, "maximum_absolute_error": 0}}, "files": {{"model": {}}}, "commands": [{{"exit_code": 0}}]}}"#,
                dump(&plan_pin),
                dump(&prediction_pin),
                dump(&bundle_pin),
                dump(&file_pin)
            ));
        }
        backends.push(r#"{"backend": "cpp17", "status": "unsupported"}"#.into());
        let fitted = pin(&artifacts, "fitted.json", "{}");
        write_file(
            &root.join("warm-d-0/receipt.json"),
            &format!(
                r#"{{"schema": "jope.inference-benchmark/v2", "plan_identity": "m", "binary_sha256": "b", "source_sha": "s", "dataset": "d", "seed": 0, "mode": "warm", "input_sha256": "r", "selected_origin": "neural_policy", "ranking_backend": "candle_cuda_float32", "devices": {cuda}, "policy_device": "Cuda(0)", "selected_program": "(pipe X (ridge))", "success": true, "elapsed_seconds": 2.5, "export": {{"validation_in_timer": true, "recipe": {{"original_proposal": "(pipe X (ridge))"}}, "fitted_output": {}, "backends": [{}]}}}}"#,
                dump(&fitted),
                backends.join(", ")
            ),
        );
        let (plan_pin, bundle_pin) = python_pins.unwrap();
        for (rows, data) in [(1, "[1.5]"), (32, "[1.5]")] {
            let prediction = pin(
                &artifacts,
                &format!("batch-{rows}.json"),
                &format!(r#"{{"kind": "values", "data": {data}}}"#),
            );
            write_file(
                &root.join(format!("prediction-d-python-{rows}/receipt.json")),
                &format!(
                    r#"{{"dataset": "d", "backend": "python", "batch_rows": {rows}, "source_sha": "s", "plan_identity": "m", "binary_sha256": "b", "input_sha256": "r", "source_qualification": {}, "prediction": {}, "parity": "exact_bits", "bundle_sha256": "{}", "samples_seconds": [{}], "status": "completed"}}"#,
                    dump(&plan_pin),
                    dump(&prediction),
                    py_str(member(&bundle_pin, "sha256")),
                    ["0.001"; 10].join(", ")
                ),
            );
        }
        parse(
            r#"{"schema": "jope.inference-benchmark/v2", "experiment": {"methods_sha256": "m", "source_sha": "s"}, "binary": {"sha256": "b"}, "inputs": [{"id": "d", "raw": {"sha256": "r"}}], "census": {"warm": 1, "cold": 0, "warmup": 0, "profile": 0}}"#,
        )
    }

    #[test]
    fn complete_qualified_evidence_releases_the_headline() {
        let root = tempfile::tempdir().unwrap();
        let plan = qualified_journal(root.path());
        let (s, records) = summarize(&plan, root.path()).unwrap();
        assert_eq!(member(&s, "qualified"), &Json::Bool(true));
        assert!(is(
            member(member(&s, "controller"), "qualification_status"),
            "passed"
        ));
        assert_eq!(member(&s, "publication_headline"), &Json::Bool(true));
        assert_eq!(member(&s, "completed_successfully"), &Json::Int(1));
        assert_eq!(member(&s, "failures"), &parse("[]"));
        assert_eq!(member(&records[0], "accepted"), &Json::Bool(true));
        assert_eq!(
            member(&s, "warm_cumulative_backend_milestones"),
            &parse(
                r#"[{"dataset": "d", "backend": "python", "n": 1, "median_seconds": 0.5, "p95_seconds": 0.5, "mean_seconds": 0.5},
                    {"dataset": "d", "backend": "rust", "n": 1, "median_seconds": 0.75, "p95_seconds": 0.75, "mean_seconds": 0.75}]"#
            )
        );
        assert_eq!(
            member(member(&s, "reconciliation"), "failure_or_missing_rate"),
            &Json::Float(0.0)
        );
        let prediction = items(member(&s, "prediction"));
        assert_eq!(prediction.len(), 9);
        // One exact batch of one row is accepted; 32 rows claimed with one value is not.
        assert_eq!(member(&prediction[0], "accepted"), &Json::Bool(true));
        assert!(is(member(&prediction[1], "status"), "completed"));
        assert_eq!(member(&prediction[1], "accepted"), &Json::Bool(false));
        let cpp = prediction
            .iter()
            .filter(|p| is(member(p, "backend"), "cpp17"))
            .collect::<Vec<_>>();
        assert_eq!(cpp.len(), 3);
        assert!(cpp.iter().all(|p| is(member(p, "status"), "unsupported")));

        // Any unplanned direct receipt withholds the headline.
        write_file(&root.path().join("warmup-d-9/receipt.json"), "{}");
        let s = summary(&plan, root.path());
        assert_eq!(member(&s, "publication_headline"), &Json::Bool(false));
        assert_eq!(
            member(member(&s, "reconciliation"), "unplanned_receipts"),
            &parse(r#"["warmup-d-9"]"#)
        );
    }

    #[test]
    fn float_spelling_matches_python_repr() {
        for (value, spelling) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (0.1, "0.1"),
            (1.5, "1.5"),
            (100.0, "100.0"),
            (1.0 / 3.0, "0.3333333333333333"),
            (2.0 / 3.0, "0.6666666666666666"),
            (0.1 + 0.2, "0.30000000000000004"),
            (1e-4, "0.0001"),
            (0.00012, "0.00012"),
            (1e-5, "1e-05"),
            (-2.5e-7, "-2.5e-07"),
            (1e15, "1000000000000000.0"),
            (1234567890123456.8, "1234567890123456.8"),
            (1e16, "1e+16"),
            (12345678901234567.0, "1.2345678901234568e+16"),
            (1e22, "1e+22"),
            (1e23, "1e+23"),
            (5e-324, "5e-324"),
            (f64::MAX, "1.7976931348623157e+308"),
        ] {
            assert_eq!(float_repr(value), spelling, "{value:?}");
        }
        assert_eq!(
            py_str(&parse(
                r#"[1, "a'b", null, true, 1.0, {"k": "v"}, "\u00e9\u0085"]"#
            )),
            r#"[1, "a'b", None, True, 1.0, {'k': 'v'}, 'é\x85']"#
        );
    }

    #[test]
    fn statistics_are_exact_like_the_python_module() {
        let values = |text: &str| items(&parse(text)).clone();
        // The exact mean of the three doubles rounds to 0.2; summing in floats gives 0.2000…04.
        assert_eq!(mean(&values("[0.1, 0.2, 0.3]")).unwrap(), Json::Float(0.2));
        assert_ne!((0.1 + 0.2 + 0.3) / 3.0, 0.2);
        assert_eq!(mean(&values("[1e308, 1e308]")).unwrap(), Json::Float(1e308));
        assert_eq!(mean(&values("[1, 2]")).unwrap(), Json::Float(1.5));
        assert_eq!(mean(&values("[1, 3]")).unwrap(), Json::Int(2));
        assert_eq!(mean(&values("[1, 2.0]")).unwrap(), Json::Float(1.5));
        assert_eq!(mean(&values("[true, true]")).unwrap(), Json::Int(1));
        assert_eq!(mean(&values("[-0.0]")).unwrap(), Json::Float(0.0));
        assert_eq!(mean(&values("[5e-324, 0.0]")).unwrap(), Json::Float(0.0));
        assert_eq!(
            mean(&values("[5e-324, 1e-323]")).unwrap(),
            Json::Float(1e-323)
        );
        assert_eq!(
            mean(&values("[-5e-324, -1e-323]")).unwrap(),
            Json::Float(-1e-323)
        );
        assert_eq!(
            distribution(&values("[3, 1, 2, 4]")).unwrap(),
            parse(r#"{"n": 4, "median_seconds": 2.5, "p95_seconds": 4, "mean_seconds": 2.5}"#)
        );
        assert_eq!(
            distribution(&values("[3, 1]")).unwrap(),
            parse(r#"{"n": 2, "median_seconds": 2.0, "p95_seconds": 3, "mean_seconds": 2}"#)
        );
        assert_eq!(
            distribution(&values("[]")).unwrap(),
            parse(r#"{"n": 0, "median_seconds": null, "p95_seconds": null, "mean_seconds": null}"#)
        );
    }

    #[test]
    fn written_report_keeps_python_spelling() {
        let root = tempfile::tempdir().unwrap();
        let journal = root.path().join("journal");
        let output = root.path().join("output");
        write_file(
            &journal.join("launch.json"),
            r#"{"exit_code": 0, "t": 1e-05, "big": 1e16, "neg": -0.0, "n": 12345678901234567890, "s": "tab\there é"}"#,
        );
        write_file(
            &journal.join("window.json"),
            r#"{"started": 1.5, "list": [], "obj": {}}"#,
        );
        write_file(
            &journal.join("warm-é1-0/receipt.json"),
            r#"{"mode": "warm", "dataset": "é1", "seed": 0, "elapsed_seconds": 2.5, "failure": "bad, \"quoted\"\nline", "success": true}"#,
        );
        let plan = r#"{"schema": "jope.inference-benchmark/v2", "experiment": {"methods_sha256": "m", "source_sha": "s"}, "binary": {"sha256": "b"}, "inputs": [{"id": "é1", "raw": {"sha256": "r"}}], "census": {"warm": 1, "cold": 0, "warmup": 0, "profile": 0}}"#;
        write(plan.as_bytes(), &journal, &output).unwrap();

        let prediction = PREDICTION_BACKENDS
            .iter()
            .flat_map(|backend| {
                BATCH_ROWS.iter().map(move |rows| {
                    format!(
                        "    {{\n      \"dataset\": \"\\u00e91\",\n      \"backend\": \"{backend}\",\n      \"batch_rows\": {rows},\n      \"status\": \"unexecuted\",\n      \"accepted\": false\n    }}"
                    )
                })
            })
            .collect::<Vec<_>>()
            .join(",\n");
        let expected = format!(
            r#"{{
  "schema": "jope.inference-benchmark/v2",
  "controller": {{
    "status": "completed",
    "launch": {{
      "exit_code": 0,
      "t": 1e-05,
      "big": 1e+16,
      "neg": -0.0,
      "n": 12345678901234567890,
      "s": "tab\there \u00e9"
    }},
    "qualification_status": "not_executed_or_interrupted"
  }},
  "qualified": false,
  "qualification": {{}},
  "publication_headline": false,
  "completed_successfully": 0,
  "expected": 1,
  "census": {{
    "modes": {{
      "warm": 1,
      "cold": 0,
      "warmup": 0,
      "profile": 0
    }},
    "source": "plan"
  }},
  "reconciliation": {{
    "planned": 1,
    "receipts": 1,
    "accepted": 0,
    "rejected": 1,
    "missing": 0,
    "unplanned_receipts": [],
    "reconciled": true,
    "failure_or_missing_rate": 1.0
  }},
  "missing": [],
  "failures": [
    {{
      "dataset": "\u00e91",
      "mode": "warm",
      "seed": 0,
      "failure": "bad, \"quoted\"\nline"
    }}
  ],
  "warm_cumulative_backend_milestones": [],
  "cold_observations": [],
  "full_gp": [],
  "prediction": [
{prediction}
  ],
  "window": {{
    "started": 1.5,
    "list": [],
    "obj": {{}}
  }},
  "warm_complete_request_distribution": {{
    "n": 0,
    "median_seconds": null,
    "p95_seconds": null,
    "mean_seconds": null
  }},
  "boundary": "{BOUNDARY}"
}}
"#
        );
        assert_eq!(
            fs::read_to_string(output.join("summary.json")).unwrap(),
            expected
        );
        assert_eq!(
            fs::read_to_string(output.join("requests.csv")).unwrap(),
            "dataset,mode,seed,elapsed_seconds,accepted,failure\r\né1,warm,0,2.5,False,\"bad, \"\"quoted\"\"\nline\"\r\n"
        );
        let methodology = fs::read_to_string(output.join("methodology.md")).unwrap();
        assert!(methodology.starts_with(&format!("# GPU timing study\n\n{BOUNDARY}\n\n")));
        assert!(methodology.contains("Qualified: False. Completed warm Python and Rust paths: 0/1. Planned direct slots: 1, accepted 0, missing 0, unplanned receipts 0. Missing attempts are unexecuted, never zero latency."));
        assert!(methodology.ends_with("containing I/O and teardown.\n"));
        // An unsupported contract is refused before any output changes.
        let v1 = plan.replace("inference-benchmark/v2", "inference-benchmark/v1");
        let error = write(v1.as_bytes(), &journal, &root.path().join("v1")).unwrap_err();
        assert!(format!("{error:#}").contains("receipt version unsupported"));
        assert!(!root.path().join("v1").exists());
    }
}

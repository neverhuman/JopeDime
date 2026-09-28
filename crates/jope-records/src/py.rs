//! Small Python built-in semantics the verifiers depend on: `str.strip`, `int(text)`, hashing
//! equality for set membership, `in`, iteration, `Path.read_text` and `PurePosixPath.parts`.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::pyjson::{self, Value};
use crate::{Error, Result};

/// Python `str.isspace` for one character (Unicode White_Space plus U+001C..U+001F).
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python `str.strip()`.
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// Python `int(text)` for ASCII decimal text: surrounding whitespace, an optional sign and
/// single underscores between digits. Non-ASCII digits (which Python also accepts) are refused.
pub fn int(text: &str) -> Option<i64> {
    let body = strip(text);
    let (negative, digits) = match body.as_bytes().first() {
        Some(b'-') => (true, &body[1..]),
        Some(b'+') => (false, &body[1..]),
        _ => (false, body),
    };
    if digits.is_empty()
        || digits.starts_with('_')
        || digits.ends_with('_')
        || digits.contains("__")
        || !digits.bytes().all(|b| b.is_ascii_digit() || b == b'_')
    {
        return None;
    }
    let value: i64 = digits.replace('_', "").parse().ok()?;
    Some(if negative { -value } else { value })
}

/// A hashable JSON scalar under Python's hash equality (`1 == 1.0 == True`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Key {
    None,
    Int(i64),
    Float(u64),
    Str(String),
}

impl Key {
    /// `None` for an unhashable value (a list or a dict).
    pub fn of(value: &Value) -> Option<Key> {
        Some(match value {
            Value::Null => Key::None,
            Value::Bool(b) => Key::Int(i64::from(*b)),
            Value::Int(i) => Key::Int(*i),
            Value::Float(f) => {
                if value == &Value::Int(*f as i64) {
                    Key::Int(*f as i64)
                } else {
                    Key::Float(f.to_bits())
                }
            }
            Value::Str(s) => Key::Str(s.clone()),
            Value::List(_) | Value::Dict(_) => return None,
        })
    }

    pub fn str(s: &str) -> Key {
        Key::Str(s.to_owned())
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Key::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// Python `iter(value)` as a list of values: a list's items, a dict's keys, a string's
/// characters. `None` when Python would raise `TypeError`.
pub fn iterate(value: &Value) -> Option<Vec<Value>> {
    match value {
        Value::List(list) => Some(list.clone()),
        Value::Dict(dict) => Some(dict.keys().map(Value::from).collect()),
        Value::Str(s) => Some(s.chars().map(|c| Value::Str(c.to_string())).collect()),
        _ => None,
    }
}

/// `set(value)` as hash keys; `None` when not iterable or an item is unhashable.
pub fn key_set(value: &Value) -> Option<std::collections::HashSet<Key>> {
    iterate(value)?.iter().map(Key::of).collect()
}

/// Python `item in container`; `None` when Python would raise `TypeError`.
pub fn contains(container: &Value, item: &Value) -> Option<bool> {
    match container {
        Value::List(list) => Some(list.iter().any(|x| x == item)),
        Value::Dict(dict) => {
            let key = Key::of(item)?;
            Some(key.as_str().is_some_and(|k| dict.contains_key(k)))
        }
        Value::Str(s) => item.as_str().map(|needle| s.contains(needle)),
        _ => None,
    }
}

/// Python `item in container`, or the refusal for the pair Python could not search (see
/// [`exception::membership`]): which of the two values was the wrong type, and what it was.
pub fn contains_or_raise(container: &Value, item: &Value) -> std::result::Result<bool, String> {
    contains(container, item).ok_or_else(|| exception::membership(container, item))
}

/// `type(value).__name__` for a JSON value.
pub fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        Value::Str(_) => "str",
        Value::List(_) => "list",
        Value::Dict(_) => "dict",
    }
}

/// Refusals for the malformed values a retired Python tool crashed on (a traceback and exit
/// status 1, no report). A port that meets such a value refuses instead of skipping it or carrying
/// on with a default. The caller prefixes the field; each message names the operation that could
/// not be applied and the offending value's type. The wording is modelled on the exception
/// CPython raised so a reader can recognise the case, but it is not a copy of CPython's text and
/// no test compares it with CPython's: the negative parity fixtures assert only that both sides
/// refuse (exit status 1, nothing printed on stdout).
pub mod exception {
    use super::type_name;
    use crate::pyjson::{str_repr, Value};

    /// `item in container` where Python could not search `container` for `item`: a dict and an
    /// unhashable item, a string and an item that is not a string, or a container that is neither
    /// a list, a dict nor a string.
    pub fn membership(container: &Value, item: &Value) -> String {
        match container {
            Value::Dict(_) => unhashable(item),
            Value::Str(_) => format!(
                "TypeError: 'in <string>' requires string as left operand, not {}",
                type_name(item)
            ),
            other => not_container(other),
        }
    }

    /// `Path(value)` on a value that is not a string.
    pub fn path_argument(value: &Value) -> String {
        format!(
            "TypeError: a path must be a string, not '{}'",
            type_name(value)
        )
    }

    /// `os.path.*(value)` (`os.fspath`) on a value that is not a string.
    pub fn fspath(value: &Value) -> String {
        format!(
            "TypeError: expected str, bytes or os.PathLike object, not {}",
            type_name(value)
        )
    }

    /// `for x in value`, `set(value)` or `s.intersection(value)` on a value that is not iterable.
    pub fn not_iterable(value: &Value) -> String {
        format!("TypeError: '{}' object is not iterable", type_name(value))
    }

    /// `item in value` on a value that is not a container.
    pub fn not_container(value: &Value) -> String {
        format!(
            "TypeError: argument of type '{}' is not iterable",
            type_name(value)
        )
    }

    /// Hashing a list or a dict (set membership, dict lookup).
    pub fn unhashable(value: &Value) -> String {
        format!("TypeError: unhashable type: '{}'", type_name(value))
    }

    /// `value.<attribute>` on a type that has no such attribute.
    pub fn no_attribute(value: &Value, attribute: &str) -> String {
        format!(
            "AttributeError: '{}' object has no attribute '{attribute}'",
            type_name(value)
        )
    }

    /// `d[key]` with the key missing.
    pub fn key_error(key: &str) -> String {
        format!("KeyError: {}", str_repr(key))
    }

    /// `value[key]` with a string key on a value that is not a dict.
    pub fn not_subscriptable(value: &Value) -> String {
        match value {
            Value::Str(_) => "TypeError: string indices must be integers, not 'str'".to_owned(),
            Value::List(_) => {
                "TypeError: list indices must be integers or slices, not str".to_owned()
            }
            other => format!(
                "TypeError: '{}' object is not subscriptable",
                type_name(other)
            ),
        }
    }
}

/// Python `separator.join(value)`: the items `for x in value` yields, joined, or the exception
/// line Python raised (a non-iterable, or an item that is not a string).
pub fn join(separator: &str, value: &Value) -> std::result::Result<String, String> {
    let items = iterate(value).ok_or_else(|| "TypeError: can only join an iterable".to_owned())?;
    let mut parts = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        match item.as_str() {
            Some(text) => parts.push(text.to_owned()),
            None => {
                return Err(format!(
                    "TypeError: sequence item {i}: expected str instance, {} found",
                    type_name(item)
                ))
            }
        }
    }
    Ok(parts.join(separator))
}

/// Python `value[0]`: the first item (a list's, or a string's first character), or the exception
/// line Python raised.
pub fn first_item(value: &Value) -> std::result::Result<Value, String> {
    match value {
        Value::List(list) => list
            .first()
            .cloned()
            .ok_or_else(|| "IndexError: list index out of range".to_owned()),
        Value::Str(text) => text
            .chars()
            .next()
            .map(|c| Value::from(c.to_string()))
            .ok_or_else(|| "IndexError: string index out of range".to_owned()),
        Value::Dict(_) => Err("KeyError: 0".to_owned()),
        other => Err(format!(
            "TypeError: '{}' object is not subscriptable",
            type_name(other)
        )),
    }
}

/// Python `value[key]` for a string key: the item, or the refusal where Python raised (a missing
/// key, or a value that is not a dict).
pub fn subscript<'a>(value: &'a Value, key: &str) -> std::result::Result<&'a Value, String> {
    match value {
        Value::Dict(dict) => dict.get(key).ok_or_else(|| exception::key_error(key)),
        other => Err(exception::not_subscriptable(other)),
    }
}

/// Python `repr` of a tuple of values.
pub fn tuple_repr(items: &[Value]) -> String {
    let parts: Vec<String> = items.iter().map(Value::repr).collect();
    if parts.len() == 1 {
        format!("({},)", parts[0])
    } else {
        format!("({})", parts.join(", "))
    }
}

/// Python `repr` of a list of strings (as `sorted(...)[:n]` printed them).
pub fn str_list_repr<S: AsRef<str>>(items: &[S]) -> String {
    let parts: Vec<String> = items.iter().map(|s| pyjson::str_repr(s.as_ref())).collect();
    format!("[{}]", parts.join(", "))
}

/// `Path.read_bytes()`.
pub fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|e| Error::io(path, e))
}

/// `Path.read_text()`: strict UTF-8 with universal newlines (`\r\n` and `\r` become `\n`).
pub fn read_text(path: &Path) -> Result<String> {
    let bytes = read_bytes(path)?;
    let text = String::from_utf8(bytes)
        .map_err(|e| Error::parse(path, format!("'utf-8' codec can't decode: {e}")))?;
    if !text.contains('\r') {
        return Ok(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' {
            out.push('\n');
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
        } else {
            out.push(c);
        }
    }
    Ok(out)
}

/// `json.loads(path.read_text())`.
pub fn load_json(path: &Path) -> Result<Value> {
    let text = read_text(path)?;
    pyjson::parse_str(&text).map_err(|message| Error::parse(path, message))
}

/// `json.loads(path.read_bytes())`.
pub fn load_json_bytes(path: &Path, bytes: &[u8]) -> Result<Value> {
    pyjson::parse_bytes(bytes).map_err(|message| Error::parse(path, message))
}

/// Lower-case hexadecimal SHA-256 (`hashlib.sha256(data).hexdigest()`).
pub fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// `Path.is_symlink()`: false when the path cannot be inspected.
pub fn is_symlink(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink())
}

/// `re.fullmatch("[0-9a-f]{n}", value or "")` for a JSON value (a non-string never matches).
pub fn is_hex(value: Option<&Value>, len: usize) -> bool {
    value.and_then(Value::as_str).is_some_and(|s| {
        s.len() == len
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

/// `PurePosixPath(text)`: its root (`/`, or `//` exactly as POSIX keeps it) and its parts
/// without empty or `.` components.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PosixPath {
    pub root: Option<&'static str>,
    pub parts: Vec<String>,
}

impl PosixPath {
    pub fn parse(text: &str) -> Self {
        let root = if text.starts_with("//") && !text.starts_with("///") {
            Some("//")
        } else if text.starts_with('/') {
            Some("/")
        } else {
            None
        };
        let parts = text
            .split('/')
            .filter(|part| !part.is_empty() && *part != ".")
            .map(str::to_owned)
            .collect();
        PosixPath { root, parts }
    }

    pub fn is_absolute(&self) -> bool {
        self.root.is_some()
    }

    /// `".." in path.parts`.
    pub fn has_parent_reference(&self) -> bool {
        self.parts.iter().any(|part| part == "..")
    }

    /// `base / path` for a relative path.
    pub fn join_onto(&self, base: &Path) -> PathBuf {
        let mut out = base.to_path_buf();
        for part in &self.parts {
            out.push(part);
        }
        out
    }

    /// Whether the relative parts begin with `prefix`.
    pub fn starts_with(&self, prefix: &[&str]) -> bool {
        self.root.is_none()
            && self.parts.len() >= prefix.len()
            && self.parts.iter().zip(prefix).all(|(a, b)| a == b)
    }

    /// `self / other`: an absolute `other` replaces the path, as in Python.
    pub fn join(&self, other: &str) -> Self {
        let other = PosixPath::parse(other);
        if other.is_absolute() {
            return other;
        }
        let mut parts = self.parts.clone();
        parts.extend(other.parts);
        PosixPath {
            root: self.root,
            parts,
        }
    }

    /// `path.name`: the final component, or `""`.
    pub fn name(&self) -> &str {
        self.parts.last().map_or("", String::as_str)
    }

    /// `path.suffix` (Python 3.12): the final component's extension, if it has one that is not
    /// its whole name.
    pub fn suffix(&self) -> &str {
        let name = self.name();
        match name.rfind('.') {
            Some(i) if 0 < i && i < name.len() - 1 => &name[i..],
            _ => "",
        }
    }
}

/// `str(PurePosixPath(...))`.
impl std::fmt::Display for PosixPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let body = self.parts.join("/");
        match (self.root, body.is_empty()) {
            (Some(root), _) => write!(f, "{root}{body}"),
            (None, true) => f.write_str("."),
            (None, false) => f.write_str(&body),
        }
    }
}

/// `math.fsum`: the correctly rounded sum of finite values, ported from CPython's
/// `mathmodule.c` (Shewchuk partials plus its half-even correction across partials). Returns
/// `None` where CPython raises (intermediate overflow, or `-inf + inf`).
pub fn fsum(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut partials: Vec<f64> = Vec::new();
    let mut special_sum = 0.0_f64;
    let mut inf_sum = 0.0_f64;
    for value in values {
        let mut x = value;
        let xsave = x;
        let mut kept = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let yr = hi - x;
            let lo = y - yr;
            if lo != 0.0 {
                partials[kept] = lo;
                kept += 1;
            }
            x = hi;
        }
        partials.truncate(kept);
        if x != 0.0 {
            if !x.is_finite() {
                if xsave.is_finite() {
                    return None; // intermediate overflow
                }
                if xsave.is_infinite() {
                    inf_sum += xsave;
                }
                special_sum += xsave;
                partials.clear();
            } else {
                partials.push(x);
            }
        }
    }
    if special_sum != 0.0 {
        return if inf_sum.is_nan() {
            None
        } else {
            Some(special_sum)
        };
    }
    let mut n = partials.len();
    let mut hi = 0.0_f64;
    if n > 0 {
        n -= 1;
        hi = partials[n];
        let mut lo = 0.0_f64;
        while n > 0 {
            let x = hi;
            n -= 1;
            let y = partials[n];
            hi = x + y;
            let yr = hi - x;
            lo = y - yr;
            if lo != 0.0 {
                break;
            }
        }
        if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
            let y = lo * 2.0;
            let x = hi + y;
            let yr = x - hi;
            if y == yr {
                hi = x;
            }
        }
    }
    Some(hi)
}

/// Python `round(value, ndigits)` for a float: correctly rounded to `ndigits` decimals (ties to
/// even on the exact binary value), returned as the nearest double.
pub fn round(value: f64, ndigits: usize) -> f64 {
    if !value.is_finite() {
        return value;
    }
    format!("{value:.ndigits$}").parse().unwrap_or(value)
}

/// Python `float(text)` for `str` input: surrounding whitespace, a sign, `inf`/`infinity`/`nan`
/// in any case, and underscores between digits (PEP 515). Non-ASCII digits are refused.
pub fn float(text: &str) -> Option<f64> {
    let body = strip(text);
    let (negative, unsigned) = match body.as_bytes().first() {
        Some(b'-') => (true, &body[1..]),
        Some(b'+') => (false, &body[1..]),
        _ => (false, body),
    };
    let lower = unsigned.to_ascii_lowercase();
    let special = match lower.as_str() {
        "inf" | "infinity" => Some(f64::INFINITY),
        "nan" => Some(f64::NAN),
        _ => None,
    };
    if let Some(value) = special {
        return Some(if negative { -value } else { value });
    }
    // digitpart ("." digitpart?)? | "." digitpart, then an optional exponent; digitpart is digits
    // with single underscores between them.
    fn digitpart(s: &[u8], mut i: usize) -> usize {
        let start = i;
        while i < s.len() {
            let separator =
                s[i] == b'_' && i > start && s.get(i + 1).is_some_and(u8::is_ascii_digit);
            if !(s[i].is_ascii_digit() || separator) {
                break;
            }
            i += 1;
        }
        i
    }
    let s = unsigned.as_bytes();
    let mut i = digitpart(s, 0);
    let mut digits = i > 0;
    if i < s.len() && s[i] == b'.' {
        let after = digitpart(s, i + 1);
        digits |= after > i + 1;
        i = after;
    }
    if !digits {
        return None;
    }
    if i < s.len() && (s[i] == b'e' || s[i] == b'E') {
        let mut j = i + 1;
        if j < s.len() && (s[j] == b'+' || s[j] == b'-') {
            j += 1;
        }
        let end = digitpart(s, j);
        if end == j {
            return None;
        }
        i = end;
    }
    if i != s.len() {
        return None;
    }
    let value: f64 = unsigned.replace('_', "").parse().ok()?;
    Some(if negative { -value } else { value })
}

/// Python `str.splitlines()`: lines without their terminators.
pub fn splitlines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let end = match c {
            '\r' => {
                if let Some(&(_, '\n')) = chars.peek() {
                    chars.next();
                    i + 2
                } else {
                    i + 1
                }
            }
            '\n' | '\u{b}' | '\u{c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}'
            | '\u{2029}' => i + c.len_utf8(),
            _ => continue,
        };
        lines.push(&text[start..i]);
        start = end;
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_int_and_strip() {
        assert_eq!(int("01"), Some(1));
        assert_eq!(int(" +1_0 "), Some(10));
        assert_eq!(int("1__0"), None);
        assert_eq!(int(""), None);
        assert_eq!(strip("\u{1c} x \u{a0}"), "x");
    }

    #[test]
    fn pure_posix_path_parts() {
        let path = PosixPath::parse("evidence/./S03-G03//x.json");
        assert_eq!(path.parts, ["evidence", "S03-G03", "x.json"]);
        assert!(!path.is_absolute());
        assert!(PosixPath::parse("//a").root == Some("//"));
        assert!(PosixPath::parse("///a").root == Some("/"));
        assert!(PosixPath::parse("a/../b").has_parent_reference());
        assert!(PosixPath::parse("").parts.is_empty());
        let base = PosixPath::parse("assets/./track/");
        assert_eq!(
            base.join("tracks")
                .join("t1")
                .join("track.json")
                .to_string(),
            "assets/track/tracks/t1/track.json"
        );
        assert_eq!(base.join("/abs").to_string(), "/abs");
        assert_eq!(PosixPath::parse("").to_string(), ".");
        assert_eq!(PosixPath::parse("//x//y/").to_string(), "//x/y");
        assert_eq!(PosixPath::parse("a/scores.csv.zst").suffix(), ".zst");
        assert_eq!(PosixPath::parse("a/.zst").suffix(), "");
        assert_eq!(PosixPath::parse("a/b.").suffix(), "");
        assert_eq!(PosixPath::parse("a/b.json").name(), "b.json");
    }

    #[test]
    fn fsum_matches_cpython() {
        assert_eq!(fsum([0.1; 10]), Some(1.0));
        assert_eq!(fsum([1e-16, 1.0, 1e16]), Some(10000000000000002.0));
        assert_eq!(fsum([1e308, 1e308, -1e308]), None);
        assert_eq!(fsum([f64::INFINITY, 1.0]), Some(f64::INFINITY));
        assert_eq!(fsum([f64::INFINITY, f64::NEG_INFINITY]), None);
        assert_eq!(fsum([]), Some(0.0));
        assert_eq!(fsum([1.0, 1e100, 1.0, -1e100]), Some(2.0));
        // Plain left-to-right addition gives 0.9999999999999999 here.
        assert_ne!([0.1_f64; 10].iter().sum::<f64>(), 1.0);
    }

    #[test]
    fn python_round_and_float() {
        assert_eq!(round(2.675, 2), 2.67);
        assert_eq!(round(0.25, 1), 0.2);
        assert_eq!((round(0.125, 2), round(0.375, 2)), (0.12, 0.38));
        assert_eq!(round(-0.04, 1).to_bits(), (-0.0_f64).to_bits());
        assert_eq!(float(" 1_000.5 "), Some(1000.5));
        assert_eq!(float("1."), Some(1.0));
        assert_eq!(float(".5e-1"), Some(0.05));
        assert!(float("-Infinity").is_some_and(|v| v == f64::NEG_INFINITY));
        assert!(float("nan").is_some_and(f64::is_nan));
        for bad in [
            "", "1__0", "_1", "1_", ".", "e5", "1e", "0x10", "1,5", "abc",
        ] {
            assert_eq!(float(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn python_splitlines() {
        assert_eq!(
            splitlines("a\nb\r\nc\rd\u{2028}e\n"),
            ["a", "b", "c", "d", "e"]
        );
        assert_eq!(splitlines("a\n\n"), ["a", ""]);
        assert!(splitlines("").is_empty());
    }

    #[test]
    fn python_membership() {
        let list = pyjson::parse_str(r#"["a", 1]"#).unwrap();
        assert_eq!(contains(&list, &Value::Float(1.0)), Some(true));
        // Python `==`, not JSON-text equality: `True == 1 == 1.0`, `[1] == [1.0]`.
        assert_eq!(contains(&list, &Value::Bool(true)), Some(true));
        let nested = pyjson::parse_str("[[1]]").unwrap();
        assert_eq!(
            contains(&nested, &pyjson::parse_str("[1.0]").unwrap()),
            Some(true)
        );
        // Each refusal names the value of the wrong type.
        let dict = pyjson::parse_str(r#"{"a": 1}"#).unwrap();
        assert_eq!(
            contains_or_raise(&dict, &list),
            Err("TypeError: unhashable type: 'list'".to_owned())
        );
        assert_eq!(
            contains_or_raise(&Value::from("abc"), &Value::Int(1)),
            Err("TypeError: 'in <string>' requires string as left operand, not int".to_owned())
        );
        assert_eq!(
            contains_or_raise(&Value::Null, &Value::from("a")),
            Err("TypeError: argument of type 'NoneType' is not iterable".to_owned())
        );
        assert_eq!(contains_or_raise(&dict, &Value::Int(1)), Ok(false));
        assert_eq!(
            contains(&Value::from("abc"), &Value::from("bc")),
            Some(true)
        );
        assert_eq!(contains(&Value::Int(1), &Value::from("x")), None);
        assert_eq!(Key::of(&Value::Bool(true)), Key::of(&Value::Float(1.0)));
        assert_eq!(tuple_repr(&[Value::from("a"), Value::Null]), "('a', None)");
    }
}

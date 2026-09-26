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
    }

    #[test]
    fn python_membership() {
        let list = pyjson::parse_str(r#"["a", 1]"#).unwrap();
        assert_eq!(contains(&list, &Value::Float(1.0)), Some(true));
        assert_eq!(
            contains(&Value::from("abc"), &Value::from("bc")),
            Some(true)
        );
        assert_eq!(contains(&Value::Int(1), &Value::from("x")), None);
        assert_eq!(Key::of(&Value::Bool(true)), Key::of(&Value::Float(1.0)));
        assert_eq!(tuple_repr(&[Value::from("a"), Value::Null]), "('a', None)");
    }
}

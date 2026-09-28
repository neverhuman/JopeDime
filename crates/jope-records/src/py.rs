//! Small Python built-in semantics the verifiers depend on: `str.strip`, `int(text)`, hashing
//! equality for set membership, `in`, iteration, `Path.read_text` and `PurePosixPath.parts`, the
//! `re` classes of a `str` pattern, subprocess output read with `text=True`, and the text of the
//! `OSError` and `CalledProcessError` a tool printed.

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

/// CPython 3.12's default `sys.get_int_max_str_digits()`. `int()` of a decimal string, and so
/// `json.loads` and `tomllib` of an integer literal, raised `ValueError` for more digits than this
/// (the sign, surrounding whitespace and underscores are not digits; leading zeros are).
pub const INT_MAX_STR_DIGITS: usize = 4300;

/// The limit on decimal integer conversion: `digits` digits are converted, more than
/// [`INT_MAX_STR_DIGITS`] refuse with CPython's `ValueError` message.
pub fn check_int_digits(digits: usize) -> std::result::Result<(), String> {
    if digits > INT_MAX_STR_DIGITS {
        return Err(format!(
            "Exceeds the limit ({INT_MAX_STR_DIGITS} digits) for integer string conversion: value has {digits} digits; use sys.set_int_max_str_digits() to increase the limit"
        ));
    }
    Ok(())
}

/// Python `int(text)` for ASCII decimal text: surrounding whitespace, an optional sign and
/// single underscores between digits. Non-ASCII digits (which Python also accepts) are refused,
/// and so is anything beyond `i64` (Python refused only past [`INT_MAX_STR_DIGITS`] digits).
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
    /// An integer beyond `i64` (a [`Value::BigInt`], or a float equal to one), as its decimal.
    Big(String),
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
            Value::BigInt(i) => Key::Big(i.clone()),
            Value::Float(f) => {
                if value == &Value::Int(*f as i64) {
                    Key::Int(*f as i64)
                } else if let Some(integer) = pyjson::integral_float_text(*f) {
                    Key::Big(integer)
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

/// Python `item in container`; `None` when Python would raise `TypeError`. A list is searched
/// with identity first ([`pyjson::eq_or_identical`]), so a JSON `NaN` is in a list holding one.
pub fn contains(container: &Value, item: &Value) -> Option<bool> {
    match container {
        Value::List(list) => Some(list.iter().any(|x| pyjson::eq_or_identical(x, item))),
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
        Value::Int(_) | Value::BigInt(_) => "int",
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

/// `Path.read_text()`: strict UTF-8 (a bad byte refuses with CPython's `UnicodeDecodeError`
/// message) with universal newlines (`\r\n` and `\r` become `\n`).
pub fn read_text(path: &Path) -> Result<String> {
    let bytes = read_bytes(path)?;
    let text = pyjson::decode_utf8(&bytes).map_err(|message| Error::parse(path, message))?;
    Ok(universal_newlines(text))
}

/// `Path.read_text(errors="replace")`: each invalid UTF-8 sequence becomes U+FFFD (the same
/// maximal subparts CPython replaces), with universal newlines.
pub fn read_text_replace(path: &Path) -> Result<String> {
    Ok(universal_newlines(&String::from_utf8_lossy(&read_bytes(
        path,
    )?)))
}

/// Text-mode newline translation (`newline=None`): `\r\n` and a lone `\r` become `\n`.
pub fn universal_newlines(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// A subprocess stream read with `text=True`: strict UTF-8 with universal newlines, or CPython's
/// `UnicodeDecodeError` message (Python raised it, and did not return the output, even when the
/// command succeeded).
pub fn text_output(bytes: &[u8]) -> std::result::Result<String, String> {
    pyjson::decode_utf8(bytes).map(universal_newlines)
}

/// `subprocess.run(..., capture_output=True, text=True)`: stdout and stderr both decoded as
/// [`text_output`] (stdout first, as `communicate` decodes them), or the stream that could not be
/// decoded and CPython's message.
pub fn text_streams(
    output: &std::process::Output,
) -> std::result::Result<(String, String), String> {
    let stdout = text_output(&output.stdout).map_err(|e| format!("stdout: {e}"))?;
    let stderr = text_output(&output.stderr).map_err(|e| format!("stderr: {e}"))?;
    Ok((stdout, stderr))
}

/// `Path.exists()` (Python 3.12): `False` for a path that is missing (`ENOENT`, `ENOTDIR`,
/// `EBADF`, `ELOOP`) or holds a NUL byte (a `ValueError` there); any other `OSError` (a name too
/// long, a permission refused) was raised, and is returned here.
pub fn path_exists(path: &Path) -> std::io::Result<bool> {
    const IGNORED: [i32; 4] = [2, 9, 20, 40]; // ENOENT, EBADF, ENOTDIR, ELOOP (Linux)
    match std::fs::metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.raw_os_error().is_some_and(|code| IGNORED.contains(&code)) => Ok(false),
        Err(e) if e.raw_os_error().is_none() && e.kind() == std::io::ErrorKind::InvalidInput => {
            Ok(false)
        }
        Err(e) => Err(e),
    }
}

/// `CompletedProcess.returncode`: the exit code, or minus the signal that killed the process.
pub fn returncode(status: std::process::ExitStatus) -> i64 {
    use std::os::unix::process::ExitStatusExt;
    match (status.code(), status.signal()) {
        (Some(code), _) => i64::from(code),
        (None, Some(signal)) => -i64::from(signal),
        (None, None) => -1,
    }
}

/// `str(subprocess.CalledProcessError(returncode, argv))` for `command`, the `repr` of the argument
/// list: `Command '[...]' returned non-zero exit status N.`, or for a command killed by a signal
/// `died with <Signals.SIGKILL: 9>.` (Linux signal numbers; an alias prints under its enum name,
/// `SIGABRT`, `SIGCHLD`, `SIGIO`).
pub fn called_process_error(command: &str, returncode: i64) -> String {
    const SIGNALS: [&str; 31] = [
        "SIGHUP",
        "SIGINT",
        "SIGQUIT",
        "SIGILL",
        "SIGTRAP",
        "SIGABRT",
        "SIGBUS",
        "SIGFPE",
        "SIGKILL",
        "SIGUSR1",
        "SIGSEGV",
        "SIGUSR2",
        "SIGPIPE",
        "SIGALRM",
        "SIGTERM",
        "SIGSTKFLT",
        "SIGCHLD",
        "SIGCONT",
        "SIGSTOP",
        "SIGTSTP",
        "SIGTTIN",
        "SIGTTOU",
        "SIGURG",
        "SIGXCPU",
        "SIGXFSZ",
        "SIGVTALRM",
        "SIGPROF",
        "SIGWINCH",
        "SIGIO",
        "SIGPWR",
        "SIGSYS",
    ];
    if returncode >= 0 {
        return format!("Command '{command}' returned non-zero exit status {returncode}.");
    }
    let signal = -returncode;
    let name = match signal {
        1..=31 => Some(SIGNALS[(signal - 1) as usize]),
        34 => Some("SIGRTMIN"),
        64 => Some("SIGRTMAX"),
        _ => None,
    };
    match name {
        Some(name) => format!("Command '{command}' died with <Signals.{name}: {signal}>."),
        None => format!("Command '{command}' died with unknown signal {signal}."),
    }
}

/// `str(OSError)` for an error Python raised with a file name: `[Errno N] <strerror>: '<name>'`
/// (`strerror` is the C library's text, which Rust's `io::Error` carries before `(os error N)`).
pub fn os_error(error: &std::io::Error, filename: &str) -> String {
    match error.raw_os_error() {
        Some(code) => {
            let text = std::io::Error::from_raw_os_error(code).to_string();
            let strerror = text
                .strip_suffix(&format!(" (os error {code})"))
                .unwrap_or(&text);
            format!("[Errno {code}] {strerror}: {}", pyjson::str_repr(filename))
        }
        None => format!("{filename}: {error}"),
    }
}

/// A pattern written for Python 3.12's `re` (a `str` pattern, Unicode 15.0), compiled so that its
/// classes match what Python's matched. The `regex` crate's own `\s`, `\w` and `\d` are Unicode
/// 16.0 `White_Space`, `Alphabetic`+marks+connectors and `Nd`; Python's are `str.isspace`
/// (`White_Space` plus U+001C..U+001F), `str.isalnum` or `_` (general category `L*` or `N*`)
/// and `str.isdecimal` (`Nd`), all as of Unicode 15.0. Those three escapes are rewritten, inside
/// or outside a bracket class; `\S`, `\W`, `\D`, `\b` and `\B` are not supported (none of the
/// ported patterns uses them).
pub fn regex(pattern: &str) -> regex::Regex {
    const SPACE: &str = r"\s\x1C-\x1F";
    const WORD: &str = r"[\p{L}\p{N}_&&\p{Age=V15_0}]";
    const DIGIT: &str = r"[\p{Nd}&&\p{Age=V15_0}]";
    let mut out = String::with_capacity(pattern.len() * 2);
    let mut depth = 0usize;
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                let escaped = chars.next().expect("a complete escape");
                match (escaped, depth > 0) {
                    ('s', false) => {
                        out.push('[');
                        out.push_str(SPACE);
                        out.push(']');
                    }
                    ('s', true) => out.push_str(SPACE),
                    ('w', _) => out.push_str(WORD),
                    ('d', _) => out.push_str(DIGIT),
                    ('S' | 'W' | 'D' | 'b' | 'B', _) => {
                        panic!("\\{escaped} has no Python-compatible translation here")
                    }
                    (other, _) => {
                        out.push('\\');
                        out.push(other);
                    }
                }
            }
            '[' => {
                depth += 1;
                out.push(c);
                // A `]` or `^]` right after the opening bracket is literal in both dialects.
            }
            ']' if depth > 0 => {
                depth -= 1;
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    regex::Regex::new(&out).expect("translated pattern")
}

/// Python `int(text)` for a run of Unicode decimal digits that `\d+` captured: the value as ASCII
/// decimal without leading zeros ([`decimal_digits`]), or the `ValueError` Python raised: text that
/// is not decimal digits, or more than [`INT_MAX_STR_DIGITS`] of them (leading zeros count; CPython
/// maps each Unicode digit to one ASCII digit before it counts).
pub fn decimal_int(text: &str) -> std::result::Result<String, String> {
    let value = decimal_digits(text).ok_or_else(|| {
        format!(
            "invalid literal for int() with base 10: {}",
            pyjson::str_repr(text)
        )
    })?;
    check_int_digits(text.chars().count())?;
    Ok(value)
}

/// The value of a run of Unicode decimal digits (`str.isdecimal`, what `\d+` captures) as ASCII
/// decimal without leading zeros, of any length; `None` when a character is not a decimal digit.
/// This is the digit mapping only: [`decimal_int`] is Python's `int()` of such a run, with its
/// limit on the number of digits.
pub fn decimal_digits(text: &str) -> Option<String> {
    static DIGIT: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| regex(r"^\d$"));
    let is_digit = |c: char| {
        let mut buffer = [0u8; 4];
        DIGIT.is_match(c.encode_utf8(&mut buffer))
    };
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if !is_digit(c) {
            return None;
        }
        // Unicode encodes every decimal digit set as ten consecutive code points 0..9, and the
        // sets that abut each other (the mathematical digits) each start at a multiple of ten
        // from the run's first code point.
        let mut first = c as u32;
        while let Some(previous) = first.checked_sub(1).and_then(char::from_u32) {
            if !is_digit(previous) {
                break;
            }
            first -= 1;
        }
        out.push(char::from_digit((c as u32 - first) % 10, 10).expect("a digit"));
    }
    if out.is_empty() {
        return None;
    }
    let trimmed = out.trim_start_matches('0');
    Some(if trimmed.is_empty() {
        "0".to_owned()
    } else {
        trimmed.to_owned()
    })
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

    /// Python 3.12's `\s`, `\w` and `\d` (Unicode 15.0), not the `regex` crate's.
    #[test]
    fn regex_classes_are_pythons() {
        let space = regex(r"^\s$");
        for c in ['\u{1c}', '\u{1f}', '\u{a0}', '\u{2028}', '\u{3000}', '\t'] {
            assert!(space.is_match(&c.to_string()), "{c:?} is a Python space");
        }
        assert!(!space.is_match("\u{180e}") && !space.is_match("\u{200b}"));
        let word = regex(r"^\w$");
        for c in [
            '_', 'a', '\u{e9}', '\u{b2}', '\u{2167}', '\u{4e00}', '\u{661}',
        ] {
            assert!(
                word.is_match(&c.to_string()),
                "{c:?} is a Python word character"
            );
        }
        // Marks, connector punctuation, joiners and `Other_Alphabetic` symbols are not; nor is a
        // letter first assigned after Unicode 15.0 (U+2EBF0, CJK Extension I, 15.1).
        for c in ['\u{301}', '\u{203f}', '\u{200d}', '\u{24b6}', '\u{2ebf0}'] {
            assert!(
                !word.is_match(&c.to_string()),
                "{c:?} is not a Python word character"
            );
        }
        let class = regex(r"^[\w-]+\s*:$");
        assert!(class.is_match("a-b\u{1f}:") && !class.is_match("a\u{301}:"));
        let digit = regex(r"^\d+$");
        assert!(digit.is_match("0\u{665}\u{1d7ce}") && !digit.is_match("\u{b2}"));
        // U+1D7D9 is MATHEMATICAL DOUBLE-STRUCK DIGIT ONE, in the run after the bold digits.
        assert_eq!(decimal_digits("\u{665}\u{1d7d9}0"), Some("510".to_owned()));
        assert_eq!(decimal_digits("007"), Some("7".to_owned()));
        assert_eq!(decimal_digits("\u{b2}"), None);
    }

    /// Python 3.12's `int()` of a digit run: 4300 digits convert, 4301 refuse with its message;
    /// leading zeros and Unicode digits count, one digit each.
    #[test]
    fn int_conversion_limit_is_pythons() {
        let limit = |n: usize| {
            format!("Exceeds the limit (4300 digits) for integer string conversion: value has {n} digits; use sys.set_int_max_str_digits() to increase the limit")
        };
        assert_eq!(check_int_digits(4300), Ok(()));
        assert_eq!(check_int_digits(4301), Err(limit(4301)));
        let run =
            |n: usize, zero: char, one: char| format!("{}{one}", zero.to_string().repeat(n - 1));
        assert_eq!(decimal_int(&run(4300, '0', '1')), Ok("1".to_owned()));
        assert_eq!(decimal_int(&run(4301, '0', '1')), Err(limit(4301)));
        assert_eq!(
            decimal_int(&run(4300, '\u{660}', '\u{661}')),
            Ok("1".to_owned())
        );
        assert_eq!(
            decimal_int(&run(4302, '\u{660}', '\u{661}')),
            Err(limit(4302))
        );
        assert_eq!(
            decimal_int("\u{b2}"),
            Err("invalid literal for int() with base 10: '\u{b2}'".to_owned())
        );
        // The digit mapping alone has no limit (what `\d+` matches, before `int()`).
        assert_eq!(decimal_digits(&run(4301, '0', '7')), Some("7".to_owned()));
        // The 64-bit `int` refuses long before the limit.
        assert_eq!(int(&"1".repeat(20)), None);
        assert_eq!(int(&"1".repeat(4301)), None);
    }

    #[test]
    fn subprocess_text_and_errors_read_as_python_printed_them() {
        assert_eq!(text_output(b"a\r\nb\rc\n").unwrap(), "a\nb\nc\n");
        assert_eq!(
            text_output(b"ok\xff").unwrap_err(),
            "'utf-8' codec can't decode byte 0xff in position 2: invalid start byte"
        );
        assert_eq!(
            text_output(b"\xe2\x82x").unwrap_err(),
            "'utf-8' codec can't decode bytes in position 0-1: invalid continuation byte"
        );
        assert_eq!(
            text_output(b"x\xed\xa0\x80").unwrap_err(),
            "'utf-8' codec can't decode byte 0xed in position 1: invalid continuation byte"
        );
        assert_eq!(
            text_output(b"x\xe2\x82").unwrap_err(),
            "'utf-8' codec can't decode bytes in position 1-2: unexpected end of data"
        );
        let missing = std::fs::metadata("/nonexistent-q/a").unwrap_err();
        assert_eq!(
            os_error(&missing, "/nonexistent-q/a"),
            "[Errno 2] No such file or directory: '/nonexistent-q/a'"
        );
        let argv = str_list_repr(&["du", "-s"]);
        assert_eq!(
            called_process_error(&argv, 1),
            "Command '['du', '-s']' returned non-zero exit status 1."
        );
        assert_eq!(
            called_process_error(&argv, -9),
            "Command '['du', '-s']' died with <Signals.SIGKILL: 9>."
        );
        assert_eq!(
            called_process_error(&argv, -40),
            "Command '['du', '-s']' died with unknown signal 40."
        );
        assert!(!path_exists(Path::new("/nonexistent-q/a")).unwrap());
        assert!(!path_exists(Path::new("/nonexistent-q/a\0b")).unwrap());
        let long = format!("/{}", "x".repeat(300));
        assert_eq!(
            path_exists(Path::new(&long)).unwrap_err().raw_os_error(),
            Some(36)
        );
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
        // Every JSON NaN is one object: found by identity, though `NaN == NaN` is false.
        let nans = pyjson::parse_str("[NaN]").unwrap();
        assert_eq!(contains(&nans, &Value::Float(f64::NAN)), Some(true));
        assert_ne!(Value::Float(f64::NAN), Value::Float(f64::NAN));
        assert_eq!(nans, pyjson::parse_str("[NaN]").unwrap());
        // Integers beyond 64 bits (opted in) compare and hash exactly, with floats too.
        let big = pyjson::parse_str_big("[18446744073709551616, 18446744073709551617]").unwrap();
        let items = big.as_list().unwrap();
        assert_ne!(items[0], items[1]);
        assert_eq!(items[0], Value::Float(18446744073709551616.0));
        assert_eq!(
            Key::of(&items[0]),
            Key::of(&Value::Float(18446744073709551616.0))
        );
        assert_eq!(
            contains(&big, &Value::Float(18446744073709551616.0)),
            Some(true)
        );
        assert_eq!(type_name(&items[1]), "int");
        assert!(pyjson::parse_str("18446744073709551616").is_err());
        assert_eq!(tuple_repr(&[Value::from("a"), Value::Null]), "('a', None)");
    }
}

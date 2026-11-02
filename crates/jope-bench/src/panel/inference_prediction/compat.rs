use super::*;

pub(super) const BACKENDS: [&str; 3] = ["python", "rust", "cpp17"];
pub(super) const SIZES: [usize; 3] = [1, 32, 1024];
pub(super) const NOTE: &str = "Public API timing includes return serialization; process residual also includes initialization, parsing, printing and teardown. Repeated first fitting-feature row; no scoring labels.";

/// A failure recorded in the receipt, spelled as Python's `str(exception)`.
pub(super) type Attempt<T> = std::result::Result<T, String>;

pub(super) fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64() * 1000.0)
}

pub(super) fn load(path: &Path) -> Result<Json> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

pub(super) fn text<'a>(value: &'a Json, what: &str) -> Result<&'a str> {
    match value {
        Json::Str(s) => Ok(s),
        other => bail!("TypeError: {what} is a JSON {}", py_type(other)),
    }
}

/// `value[0]`: a failure stops the stage, as the uncaught exception did.
pub(super) fn first(value: &Json) -> Result<Json> {
    match value {
        Json::Array(items) => items
            .first()
            .cloned()
            .context("IndexError: list index out of range"),
        Json::Str(s) => s
            .chars()
            .next()
            .map(|c| Json::Str(c.to_string()))
            .context("IndexError: string index out of range"),
        Json::Object(_) => bail!("KeyError: 0"),
        other => bail!(
            "TypeError: '{}' object is not subscriptable",
            py_type(other)
        ),
    }
}

/// Python's type name of a JSON value.
pub(super) fn py_type(value: &Json) -> &'static str {
    match value {
        Json::Null => "NoneType",
        Json::Bool(_) => "bool",
        Json::Int(_) => "int",
        Json::Float(_) => "float",
        Json::Str(_) => "str",
        Json::Array(_) => "list",
        Json::Object(_) => "dict",
    }
}

/// `str(pathlib.Path(path))`: empty and `.` components dropped.
pub(super) fn python_path(path: &Path) -> String {
    let mut out = String::new();
    for component in path.components() {
        let part = match component {
            Component::RootDir => {
                out.push('/');
                continue;
            }
            Component::CurDir | Component::Prefix(_) => continue,
            Component::ParentDir => "..".into(),
            Component::Normal(part) => part.to_string_lossy(),
        };
        if !out.is_empty() && !out.ends_with('/') {
            out.push('/');
        }
        out.push_str(&part);
    }
    if out.is_empty() {
        ".".into()
    } else {
        out
    }
}

/// `str(OSError)`: `[Errno N] strerror: 'filename'`.
pub(super) fn os_error(error: &std::io::Error, filename: &str) -> String {
    match error.raw_os_error() {
        Some(code) => {
            let message = std::io::Error::from_raw_os_error(code).to_string();
            let strerror = message
                .strip_suffix(&format!(" (os error {code})"))
                .unwrap_or(&message);
            format!("[Errno {code}] {strerror}: {}", str_repr(filename))
        }
        None => error.to_string(),
    }
}

pub(super) fn write_file(path: &Path, bytes: &[u8]) -> Attempt<()> {
    fs::write(path, bytes).map_err(|error| os_error(&error, &python_path(path)))
}

/// Python's `repr(bytes)`.
pub(super) fn bytes_repr(bytes: &[u8]) -> String {
    let delimiter = if bytes.contains(&b'\'') && !bytes.contains(&b'"') {
        b'"'
    } else {
        b'\''
    };
    let mut out = String::from("b");
    out.push(char::from(delimiter));
    for &byte in bytes {
        match byte {
            b'\\' => out.push_str("\\\\"),
            b'\t' => out.push_str("\\t"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            _ if byte == delimiter => {
                out.push('\\');
                out.push(char::from(byte));
            }
            0x20..=0x7e => out.push(char::from(byte)),
            _ => out.push_str(&format!("\\x{byte:02x}")),
        }
    }
    out.push(char::from(delimiter));
    out
}

/// `bytes.splitlines()`: `\n`, `\r` and `\r\n` end a line; no empty final line.
pub(super) fn splitlines(bytes: &[u8]) -> Vec<&[u8]> {
    let (mut lines, mut start, mut i) = (Vec::new(), 0, 0);
    while i < bytes.len() {
        match bytes[i] {
            b'\n' => {
                lines.push(&bytes[start..i]);
                i += 1;
                start = i;
            }
            b'\r' => {
                lines.push(&bytes[start..i]);
                i += if bytes.get(i + 1) == Some(&b'\n') {
                    2
                } else {
                    1
                };
                start = i;
            }
            _ => i += 1,
        }
    }
    if start < bytes.len() {
        lines.push(&bytes[start..]);
    }
    lines
}

/// `float(line)` for a bytes line: ASCII whitespace trimmed and underscores allowed between
/// digits; the error text is Python's.
pub(super) fn python_float(line: &[u8]) -> Attempt<f64> {
    let failure = || format!("could not convert string to float: {}", bytes_repr(line));
    let space = |b: &u8| matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r');
    let start = line.iter().position(|b| !space(b)).unwrap_or(line.len());
    let end = line
        .iter()
        .rposition(|b| !space(b))
        .map_or(start, |p| p + 1);
    let trimmed = std::str::from_utf8(&line[start..end]).map_err(|_| failure())?;
    let digits = trimmed.as_bytes();
    for (i, byte) in digits.iter().enumerate() {
        let digit = |j: usize| digits.get(j).is_some_and(u8::is_ascii_digit);
        if *byte == b'_' && !(i > 0 && digit(i - 1) && digit(i + 1)) {
            return Err(failure());
        }
    }
    trimmed.replace('_', "").parse().map_err(|_| failure())
}

/// Python 3.12's built-in `sum` of floats: compensated (Neumaier) summation.
pub(super) fn python_sum(values: &[f64]) -> f64 {
    let Some((head, rest)) = values.split_first() else {
        return 0.0;
    };
    let (mut total, mut compensation) = (0.0 + head, 0.0_f64);
    for &x in rest {
        let next = total + x;
        compensation += if total.abs() >= x.abs() {
            (total - next) + x
        } else {
            (x - next) + total
        };
        total = next;
    }
    if compensation != 0.0 && compensation.is_finite() {
        total += compensation;
    }
    total
}

/// `min(30, max(.001, seconds_to_cutoff))`, with its Python spelling.
pub(super) fn timeout(cutoff_ms: u64) -> (Duration, String) {
    let remaining = (cutoff_ms as f64 - now_ms()) / 1000.0;
    let bounded = if remaining > 0.001 { remaining } else { 0.001 };
    if bounded < 30.0 {
        (Duration::from_secs_f64(bounded), float_repr(bounded))
    } else {
        (Duration::from_secs(30), "30".into())
    }
}

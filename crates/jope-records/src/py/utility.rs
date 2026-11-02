use super::*;

/// `int(m.group(1))` of what a pattern's `\d+` captured, through the strict grammar ([`decimal`]):
/// the value as ASCII decimal without leading zeros, of any length up to [`INT_MAX_STR_DIGITS`]
/// digits, or the refusal. Python's `\d` matched every Unicode decimal digit and its `int()` mapped
/// them to ASCII; the grammar reads ASCII digits only and refuses a run holding any other
/// (stricter), naming it. More than 4300 digits refuse in CPython's words.
pub fn decimal_int(text: &str) -> std::result::Result<String, String> {
    match decimal(text) {
        Ok(Decimal {
            negative: false,
            digits,
        }) if digits.len() == text.len() => Ok(Decimal {
            negative: false,
            digits,
        }
        .magnitude()
        .to_owned()),
        Err(IntError::Limit(message)) => Err(message),
        _ => Err(format!(
            "{} is not ASCII decimal digits, the only digits the port reads as a number",
            pyjson::str_repr(text)
        )),
    }
}

/// Whether `text` is a run of what Python 3.12's `\d` matched (`str.isdecimal`, Unicode 15.0):
/// `re.fullmatch(r"\d+", text)`. A match only; a number is read from such a run by
/// [`decimal_int`].
pub fn is_decimal_run(text: &str) -> bool {
    static RUN: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| regex(r"^\d+$"));
    RUN.is_match(text)
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

/// `Path.is_symlink()` read leniently: false when the path cannot be inspected. The P1 ports
/// (coverage, scope, parity, README, cargo-artifact, build-report) do not use it; they use
/// [`path_is_symlink`], which refuses instead. It remains for the record verifiers that predate
/// them.
pub fn is_symlink(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink())
}

/// `os.path.abspath` (`posixpath`): joined to the working directory and normalised as
/// `posixpath.normpath` normalises, lexically: empty and `.` components dropped, `..` removing
/// the previous component (and nothing at the root), and exactly two leading slashes kept (POSIX
/// leaves `//` to the implementation, so Python keeps it as its own root; three or more are `/`).
/// Names stay the bytes they are. Lexical only: a symlink followed by `..` is [`realpath`]'s.
/// A relative path needs the working directory: when it cannot be read (it was deleted, say),
/// `os.getcwd()` raised and so does this, instead of resolving against `/`.
pub fn abspath(path: &Path) -> std::io::Result<PathBuf> {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};
    let mut joined = vec![];
    if !path.as_os_str().as_bytes().starts_with(b"/") {
        let cwd = std::env::current_dir()?;
        joined.extend_from_slice(cwd.as_os_str().as_bytes());
        if !joined.ends_with(b"/") {
            joined.push(b'/');
        }
    }
    joined.extend_from_slice(path.as_os_str().as_bytes());
    let slashes = if joined.starts_with(b"//") && !joined.starts_with(b"///") {
        2
    } else {
        1
    };
    let mut parts: Vec<&[u8]> = vec![];
    for part in joined.split(|&b| b == b'/') {
        match part {
            b"" | b"." => {}
            b".." => {
                parts.pop();
            }
            _ => parts.push(part),
        }
    }
    let mut out = b"/".repeat(slashes);
    out.extend(parts.join(&b'/'));
    Ok(PathBuf::from(std::ffi::OsString::from_vec(out)))
}

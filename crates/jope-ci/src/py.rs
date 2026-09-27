//! The few Python built-in behaviours the ported tools' output depends on.

use std::fmt::Write as _;
use std::path::{Component, Path, PathBuf};

/// `str.splitlines()`: the line boundaries Python recognises, not just `\n`.
pub fn splitlines(text: &str) -> Vec<&str> {
    let mut lines = vec![];
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let boundary = matches!(
            c,
            '\n' | '\r'
                | '\u{0b}'
                | '\u{0c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if boundary {
            lines.push(&text[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' {
                if let Some((j, '\n')) = chars.peek().copied() {
                    chars.next();
                    end = j + 1;
                }
            }
            start = end;
        }
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// `repr(str)`: single quotes unless the text holds a single quote and no double quote.
pub fn repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::new();
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                write!(out, "\\x{:02x}", c as u32).unwrap();
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `json.dumps(str)` with the default `ensure_ascii=True`.
pub fn json_str(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            c => {
                for unit in c.encode_utf16(&mut [0u16; 2]) {
                    write!(out, "\\u{unit:04x}").unwrap();
                }
            }
        }
    }
    out.push('"');
    out
}

/// `pathlib.PurePosixPath(text)`: repeated separators and `.` components collapse, a trailing
/// separator is dropped, and the empty path is `.`.
pub fn pure_path(text: &str) -> PathBuf {
    let mut out = PathBuf::new();
    for component in Path::new(text).components() {
        if component != Component::CurDir {
            out.push(component.as_os_str());
        }
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitlines_matches_python() {
        assert_eq!(splitlines("a\nb\r\nc\rd"), ["a", "b", "c", "d"]);
        assert_eq!(splitlines("a\n"), ["a"]);
        assert_eq!(splitlines("a\n\nb"), ["a", "", "b"]);
        assert_eq!(splitlines("x\u{2028}y"), ["x", "y"]);
        assert!(splitlines("").is_empty());
    }

    #[test]
    fn repr_and_json_match_python() {
        assert_eq!(repr("demo"), "'demo'");
        assert_eq!(repr("it's"), "\"it's\"");
        assert_eq!(repr("a'b\"c"), "'a\\'b\"c'");
        assert_eq!(json_str("/t with spaces/é"), "\"/t with spaces/\\u00e9\"");
        assert_eq!(json_str("\u{1F600}"), "\"\\ud83d\\ude00\"");
    }

    #[test]
    fn pure_path_normalises_like_pathlib() {
        assert_eq!(pure_path("a//b/./c/"), PathBuf::from("a/b/c"));
        assert_eq!(pure_path("./a"), PathBuf::from("a"));
        assert_eq!(pure_path(""), PathBuf::from("."));
        assert_eq!(pure_path("/w/x"), PathBuf::from("/w/x"));
    }
}

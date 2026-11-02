use super::*;

// ---------------------------------------------------------------------------------------------
// Python `json.dumps`, `repr`/`str` and `csv` spelling.

#[derive(Clone, Copy)]
pub(crate) struct Style {
    /// `ensure_ascii`.
    ascii: bool,
    /// `indent`; `None` writes one line.
    indent: Option<usize>,
    /// On one line, the default `(', ', ': ')` separators rather than `(',', ':')`.
    spaced: bool,
}

impl Style {
    /// `json.dumps(value, indent=2, allow_nan=False)`.
    pub(crate) const REPORT: Style = Style {
        ascii: true,
        indent: Some(2),
        spaced: false,
    };
    /// `json.dumps(value, ensure_ascii=False, separators=(',', ':'))`.
    pub(crate) const IDENTITY: Style = Style {
        ascii: false,
        indent: None,
        spaced: false,
    };
    /// `json.dumps(value)`.
    #[cfg_attr(not(any(test, feature = "inference-cuda")), allow(dead_code))]
    pub(crate) const DEFAULT: Style = Style {
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

pub(crate) fn dumps(value: &Json, style: Style) -> Result<String> {
    let mut out = String::new();
    dump(value, style, 0, &mut out)?;
    Ok(out)
}

pub(crate) fn dump(value: &Json, style: Style, level: usize, out: &mut String) -> Result<()> {
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
pub(crate) fn quote(text: &str, ascii: bool, out: &mut String) {
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
pub(crate) fn float_repr(x: f64) -> String {
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
pub(crate) fn py_str(value: &Json) -> String {
    match value {
        Json::Str(s) => s.clone(),
        other => py_repr(other),
    }
}

pub(crate) fn py_repr(value: &Json) -> String {
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
pub(crate) fn str_repr(text: &str) -> String {
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

pub(crate) fn printable(code: u32) -> bool {
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
pub(crate) fn csv_row(fields: &[&Json], out: &mut String) {
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

#[path = "format/statistics.rs"]
mod statistics;
pub(crate) use statistics::{distribution, mean, midpoint, power_of_two, rational};

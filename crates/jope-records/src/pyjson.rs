//! JSON with Python's `json` module semantics.
//!
//! The records were written and checked by Python, so this port reads them the way
//! `json.loads` did and prints summaries the way `json.dumps` did:
//!
//! - objects keep document order (a repeated key keeps its first position and its last value),
//!   so "first failing record" is the same record Python reported;
//! - `NaN`, `Infinity` and `-Infinity` literals are accepted; integers stay integers;
//! - [`dumps`] reproduces `json.dumps(..., sort_keys=..., indent=...)` with the default
//!   `ensure_ascii=True` escaping and Python float `repr`.
//!
//! Deliberate limits: integers must fit in `i64`, and a lone UTF-16 surrogate escape (which
//! Python keeps as an unpaired code point) is rejected because Rust strings cannot hold it.

use std::collections::HashMap;
use std::fmt::Write as _;

/// A parsed JSON value.
#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    Dict(Dict),
}

/// An insertion-ordered JSON object (Python `dict`).
#[derive(Clone, Debug, Default)]
pub struct Dict {
    keys: Vec<String>,
    values: Vec<Value>,
    index: HashMap<String, usize>,
}

impl Dict {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.index.get(key).map(|&i| &self.values[i])
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        match self.index.get(key) {
            Some(&i) => Some(&mut self.values[i]),
            None => None,
        }
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.index.contains_key(key)
    }

    /// `d[key] = value`: an existing key keeps its position.
    pub fn insert(&mut self, key: impl Into<String>, value: Value) -> Option<Value> {
        let key = key.into();
        if let Some(&i) = self.index.get(&key) {
            return Some(std::mem::replace(&mut self.values[i], value));
        }
        self.index.insert(key.clone(), self.keys.len());
        self.keys.push(key);
        self.values.push(value);
        None
    }

    /// `d.pop(key)`: the remaining keys keep their order.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        let i = self.index.remove(key)?;
        self.keys.remove(i);
        let value = self.values.remove(i);
        for position in self.index.values_mut() {
            if *position > i {
                *position -= 1;
            }
        }
        Some(value)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.keys.iter().map(String::as_str).zip(self.values.iter())
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.keys.iter().map(String::as_str)
    }

    pub fn values(&self) -> impl Iterator<Item = &Value> {
        self.values.iter()
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut Value> {
        self.values.iter_mut()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut Value)> {
        self.keys
            .iter()
            .map(String::as_str)
            .zip(self.values.iter_mut())
    }
}

impl<K: Into<String>> FromIterator<(K, Value)> for Dict {
    fn from_iter<I: IntoIterator<Item = (K, Value)>>(iter: I) -> Self {
        let mut dict = Dict::new();
        for (key, value) in iter {
            dict.insert(key, value);
        }
        dict
    }
}

/// Python `==` on dicts: same keys and equal values, order ignored.
impl PartialEq for Dict {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .all(|(key, value)| other.get(key).is_some_and(|theirs| value == theirs))
    }
}

/// Python `==`: `True == 1 == 1.0`, lists element-wise, dicts ignoring order.
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        use Value::{Bool, Dict, Float, Int, List, Null, Str};
        match (self, other) {
            (Null, Null) => true,
            (Str(a), Str(b)) => a == b,
            (List(a), List(b)) => a == b,
            (Dict(a), Dict(b)) => a == b,
            (Float(a), Float(b)) => a == b,
            (Bool(_) | Int(_), Bool(_) | Int(_)) => self.integer() == other.integer(),
            (Float(f), Bool(_) | Int(_)) | (Bool(_) | Int(_), Float(f)) => {
                let i = self.integer().or(other.integer()).unwrap_or_default();
                float_equals_int(*f, i)
            }
            _ => false,
        }
    }
}

fn float_equals_int(f: f64, i: i64) -> bool {
    // Python compares an int and a float exactly, never through a lossy conversion.
    f.is_finite()
        && f.fract() == 0.0
        && (-9.223_372_036_854_776e18..9.223_372_036_854_776e18).contains(&f)
        && f as i64 == i
}

impl Value {
    /// `bool` and `int` as Python's integer value (`True` is `1`).
    fn integer(&self) -> Option<i64> {
        match self {
            Value::Bool(b) => Some(i64::from(*b)),
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// `d.get(key)` on a dict; `None` for a missing key or a non-dict.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Dict(dict) => dict.get(key),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// An integer that is not a boolean (see the crate's parity notes on `isinstance(x, int)`).
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&Vec<Value>> {
        match self {
            Value::List(list) => Some(list),
            _ => None,
        }
    }

    pub fn as_list_mut(&mut self) -> Option<&mut Vec<Value>> {
        match self {
            Value::List(list) => Some(list),
            _ => None,
        }
    }

    pub fn as_dict(&self) -> Option<&Dict> {
        match self {
            Value::Dict(dict) => Some(dict),
            _ => None,
        }
    }

    pub fn as_dict_mut(&mut self) -> Option<&mut Dict> {
        match self {
            Value::Dict(dict) => Some(dict),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Python truthiness.
    pub fn truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::List(list) => !list.is_empty(),
            Value::Dict(dict) => !dict.is_empty(),
        }
    }

    /// Python `repr(value)`.
    pub fn repr(&self) -> String {
        match self {
            Value::Null => "None".to_owned(),
            Value::Bool(true) => "True".to_owned(),
            Value::Bool(false) => "False".to_owned(),
            Value::Int(i) => i.to_string(),
            Value::Float(f) => float_repr(*f),
            Value::Str(s) => str_repr(s),
            Value::List(list) => {
                let items: Vec<String> = list.iter().map(Value::repr).collect();
                format!("[{}]", items.join(", "))
            }
            Value::Dict(dict) => {
                let items: Vec<String> = dict
                    .iter()
                    .map(|(key, value)| format!("{}: {}", str_repr(key), value.repr()))
                    .collect();
                format!("{{{}}}", items.join(", "))
            }
        }
    }

    /// Python `str(value)` (what an f-string `{value}` prints).
    pub fn py_str(&self) -> String {
        match self {
            Value::Str(s) => s.clone(),
            other => other.repr(),
        }
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::Str(s.to_owned())
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::Str(s)
    }
}

impl From<usize> for Value {
    fn from(n: usize) -> Self {
        Value::Int(i64::try_from(n).unwrap_or(i64::MAX))
    }
}

impl From<i64> for Value {
    fn from(n: i64) -> Self {
        Value::Int(n)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

impl From<Vec<Value>> for Value {
    fn from(list: Vec<Value>) -> Self {
        Value::List(list)
    }
}

impl From<Dict> for Value {
    fn from(dict: Dict) -> Self {
        Value::Dict(dict)
    }
}

/// Python `repr(float)`: shortest round-trip digits, exponent form outside `[1e-4, 1e16)`.
pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    let magnitude = value.abs();
    if magnitude != 0.0 && !(1e-4..1e16).contains(&magnitude) {
        let scientific = format!("{value:e}");
        let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
        let exponent: i32 = exponent.parse().unwrap_or(0);
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:02}", exponent.abs());
    }
    let plain = format!("{value}");
    if plain.contains('.') {
        plain
    } else {
        format!("{plain}.0")
    }
}

/// Python `repr(str)`.
///
/// Printable characters are kept; the non-printable set is Python's for ASCII and Latin-1 and
/// the common format/separator characters beyond (Python consults the full Unicode database,
/// so an unassigned code point may be escaped there and kept here).
pub fn str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if is_printable(c) => out.push(c),
            c if (c as u32) <= 0xff => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c if (c as u32) <= 0xffff => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => {
                let _ = write!(out, "\\U{:08x}", c as u32);
            }
        }
    }
    out.push(quote);
    out
}

fn is_printable(c: char) -> bool {
    let cp = c as u32;
    if cp < 0x7f {
        return cp >= 0x20;
    }
    !matches!(cp,
        0x7f..=0xa0 | 0xad | 0x600..=0x605 | 0x61c | 0x6dd | 0x70f | 0x890..=0x891 | 0x8e2
        | 0x1680 | 0x180e | 0x2000..=0x200f | 0x2028..=0x202f | 0x205f..=0x2064
        | 0x2066..=0x206f | 0x3000 | 0xd800..=0xf8ff | 0xfeff | 0xfff9..=0xfffb
        | 0x110bd | 0x110cd | 0x13430..=0x1343f | 0x1bca0..=0x1bca3 | 0x1d173..=0x1d17a
        | 0xe0001 | 0xe0020..=0xe007f | 0xf0000..)
}

/// `json.dumps(value, sort_keys=sort_keys, indent=indent)` with Python's defaults otherwise.
pub fn dumps(value: &Value, sort_keys: bool, indent: Option<usize>) -> String {
    let mut out = String::new();
    write_value(&mut out, value, sort_keys, indent, 0);
    out
}

fn newline(out: &mut String, indent: Option<usize>, level: usize) {
    if let Some(width) = indent {
        out.push('\n');
        out.extend(std::iter::repeat_n(' ', width * level));
    }
}

fn write_value(
    out: &mut String,
    value: &Value,
    sort_keys: bool,
    indent: Option<usize>,
    level: usize,
) {
    let item_separator = if indent.is_some() { "," } else { ", " };
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Int(i) => {
            let _ = write!(out, "{i}");
        }
        Value::Float(f) => out.push_str(&json_float(*f)),
        Value::Str(s) => write_string(out, s),
        Value::List(list) => {
            if list.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (n, item) in list.iter().enumerate() {
                if n > 0 {
                    out.push_str(item_separator);
                }
                newline(out, indent, level + 1);
                write_value(out, item, sort_keys, indent, level + 1);
            }
            newline(out, indent, level);
            out.push(']');
        }
        Value::Dict(dict) => {
            if dict.is_empty() {
                out.push_str("{}");
                return;
            }
            let mut items: Vec<(&str, &Value)> = dict.iter().collect();
            if sort_keys {
                items.sort_by(|a, b| a.0.cmp(b.0));
            }
            out.push('{');
            for (n, (key, item)) in items.into_iter().enumerate() {
                if n > 0 {
                    out.push_str(item_separator);
                }
                newline(out, indent, level + 1);
                write_string(out, key);
                out.push_str(": ");
                write_value(out, item, sort_keys, indent, level + 1);
            }
            newline(out, indent, level);
            out.push('}');
        }
    }
}

fn json_float(f: f64) -> String {
    if f.is_nan() {
        "NaN".to_owned()
    } else if f.is_infinite() {
        if f > 0.0 { "Infinity" } else { "-Infinity" }.to_owned()
    } else {
        float_repr(f)
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
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
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}

/// `json.loads(text)` for a `str` argument.
pub fn parse_str(text: &str) -> Result<Value, String> {
    if text.starts_with('\u{feff}') {
        return Err(
            "Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)".to_owned(),
        );
    }
    let mut parser = Parser {
        text,
        bytes: text.as_bytes(),
        pos: 0,
        depth: 0,
    };
    parser.skip_whitespace();
    let value = parser.value()?;
    parser.skip_whitespace();
    if parser.pos != parser.bytes.len() {
        return Err(parser.error("Extra data", parser.pos));
    }
    Ok(value)
}

/// `json.loads(data)` for a `bytes` argument: the encoding is detected as `json.detect_encoding`
/// does (UTF-8 with or without a signature, UTF-16 and UTF-32 by their BOM or zero pattern).
pub fn parse_bytes(data: &[u8]) -> Result<Value, String> {
    let text = decode_detected(data)?;
    // Python checks for a BOM only on `str` input; after decoding bytes a stray U+FEFF is an
    // ordinary invalid character, which the parser rejects on its own.
    let mut parser = Parser {
        text: &text,
        bytes: text.as_bytes(),
        pos: 0,
        depth: 0,
    };
    parser.skip_whitespace();
    let value = parser.value()?;
    parser.skip_whitespace();
    if parser.pos != parser.bytes.len() {
        return Err(parser.error("Extra data", parser.pos));
    }
    Ok(value)
}

fn decode_detected(b: &[u8]) -> Result<String, String> {
    const UTF32_BE: &[u8] = &[0, 0, 0xfe, 0xff];
    const UTF32_LE: &[u8] = &[0xff, 0xfe, 0, 0];
    if b.starts_with(UTF32_BE) {
        return decode_utf32(&b[4..], true);
    }
    if b.starts_with(UTF32_LE) {
        return decode_utf32(&b[4..], false);
    }
    if b.starts_with(&[0xfe, 0xff]) {
        return decode_utf16(&b[2..], true);
    }
    if b.starts_with(&[0xff, 0xfe]) {
        return decode_utf16(&b[2..], false);
    }
    if let Some(rest) = b.strip_prefix(&[0xef, 0xbb, 0xbf]) {
        return decode_utf8(rest);
    }
    if b.len() >= 4 {
        if b[0] == 0 {
            return if b[1] != 0 {
                decode_utf16(b, true)
            } else {
                decode_utf32(b, true)
            };
        }
        if b[1] == 0 {
            return if b[2] != 0 || b[3] != 0 {
                decode_utf16(b, false)
            } else {
                decode_utf32(b, false)
            };
        }
    } else if b.len() == 2 {
        if b[0] == 0 {
            return decode_utf16(b, true);
        }
        if b[1] == 0 {
            return decode_utf16(b, false);
        }
    }
    decode_utf8(b)
}

fn decode_utf8(b: &[u8]) -> Result<String, String> {
    String::from_utf8(b.to_vec()).map_err(|e| format!("'utf-8' codec can't decode bytes: {e}"))
}

fn decode_utf16(b: &[u8], big_endian: bool) -> Result<String, String> {
    if !b.len().is_multiple_of(2) {
        return Err("'utf-16' codec can't decode: truncated data".to_owned());
    }
    let units: Vec<u16> = b
        .chunks_exact(2)
        .map(|pair| {
            if big_endian {
                u16::from_be_bytes([pair[0], pair[1]])
            } else {
                u16::from_le_bytes([pair[0], pair[1]])
            }
        })
        .collect();
    String::from_utf16(&units).map_err(|e| format!("'utf-16' codec can't decode: {e}"))
}

fn decode_utf32(b: &[u8], big_endian: bool) -> Result<String, String> {
    if !b.len().is_multiple_of(4) {
        return Err("'utf-32' codec can't decode: truncated data".to_owned());
    }
    b.chunks_exact(4)
        .map(|quad| {
            let cp = if big_endian {
                u32::from_be_bytes([quad[0], quad[1], quad[2], quad[3]])
            } else {
                u32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]])
            };
            char::from_u32(cp)
                .ok_or_else(|| format!("'utf-32' codec can't decode code point {cp:#x}"))
        })
        .collect()
}

/// Nesting beyond this is refused, as Python's recursion limit (about 1000) refused deeper
/// input; the lower bound keeps debug-build parser recursion well inside a test thread's stack.
const MAX_DEPTH: usize = 512;

struct Parser<'a> {
    text: &'a str,
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl Parser<'_> {
    fn error(&self, message: &str, pos: usize) -> String {
        let before = &self.text[..pos.min(self.text.len())];
        let line = before.matches('\n').count() + 1;
        let line_start = before.rfind('\n').map_or(0, |i| i + 1);
        let column = before[line_start..].chars().count() + 1;
        let char_pos = before.chars().count();
        format!("{message}: line {line} column {column} (char {char_pos})")
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn starts_with(&self, literal: &str) -> bool {
        self.bytes[self.pos..].starts_with(literal.as_bytes())
    }

    fn value(&mut self) -> Result<Value, String> {
        let Some(c) = self.peek() else {
            return Err(self.error("Expecting value", self.pos));
        };
        match c {
            b'"' => {
                self.pos += 1;
                self.string().map(Value::Str)
            }
            b'{' => self.nested(Self::object),
            b'[' => self.nested(Self::array),
            b'n' if self.starts_with("null") => self.literal(4, Value::Null),
            b't' if self.starts_with("true") => self.literal(4, Value::Bool(true)),
            b'f' if self.starts_with("false") => self.literal(5, Value::Bool(false)),
            b'N' if self.starts_with("NaN") => self.literal(3, Value::Float(f64::NAN)),
            b'I' if self.starts_with("Infinity") => self.literal(8, Value::Float(f64::INFINITY)),
            b'-' if self.starts_with("-Infinity") => {
                self.literal(9, Value::Float(f64::NEG_INFINITY))
            }
            _ => self.number(),
        }
    }

    fn literal(&mut self, len: usize, value: Value) -> Result<Value, String> {
        self.pos += len;
        Ok(value)
    }

    fn nested(&mut self, parse: fn(&mut Self) -> Result<Value, String>) -> Result<Value, String> {
        if self.depth >= MAX_DEPTH {
            return Err(self.error(
                "maximum recursion depth exceeded while decoding JSON",
                self.pos,
            ));
        }
        self.depth += 1;
        self.pos += 1;
        let result = parse(self);
        self.depth -= 1;
        result
    }

    fn digits(&mut self) -> usize {
        let start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        self.pos - start
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'1'..=b'9') => {
                self.digits();
            }
            Some(b'0') => self.pos += 1,
            _ => return Err(self.error("Expecting value", start)),
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') && matches!(self.bytes.get(self.pos + 1), Some(b'0'..=b'9')) {
            self.pos += 1;
            self.digits();
            is_float = true;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            let before_exponent = self.pos;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if self.digits() > 0 {
                is_float = true;
            } else {
                self.pos = before_exponent;
            }
        }
        let literal = &self.text[start..self.pos];
        if is_float {
            literal
                .parse::<f64>()
                .map(Value::Float)
                .map_err(|_| self.error("Invalid number", start))
        } else {
            literal
                .parse::<i64>()
                .map(Value::Int)
                .map_err(|_| self.error("Integer outside the supported 64-bit range", start))
        }
    }

    fn hex4(&self, at: usize) -> Option<u32> {
        let digits = self.bytes.get(at..at + 4)?;
        if !digits.iter().all(u8::is_ascii_hexdigit) {
            return None;
        }
        u32::from_str_radix(std::str::from_utf8(digits).ok()?, 16).ok()
    }

    /// The body of a string whose opening quote has been consumed.
    fn string(&mut self) -> Result<String, String> {
        let begin = self.pos - 1;
        let mut out = String::new();
        loop {
            let chunk_start = self.pos;
            while let Some(c) = self.peek() {
                if c == b'"' || c == b'\\' || c < 0x20 {
                    break;
                }
                self.pos += 1;
            }
            out.push_str(&self.text[chunk_start..self.pos]);
            match self.peek() {
                None => return Err(self.error("Unterminated string starting at", begin)),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    let escape_at = self.pos;
                    self.pos += 1;
                    let Some(kind) = self.peek() else {
                        return Err(self.error("Unterminated string starting at", begin));
                    };
                    self.pos += 1;
                    match kind {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let Some(mut cp) = self.hex4(self.pos) else {
                                return Err(self.error("Invalid \\uXXXX escape", escape_at));
                            };
                            self.pos += 4;
                            if (0xd800..=0xdbff).contains(&cp)
                                && self.bytes.get(self.pos..self.pos + 2) == Some(b"\\u")
                            {
                                if let Some(low) = self.hex4(self.pos + 2) {
                                    if (0xdc00..=0xdfff).contains(&low) {
                                        cp = 0x10000 + (((cp - 0xd800) << 10) | (low - 0xdc00));
                                        self.pos += 6;
                                    }
                                }
                            }
                            match char::from_u32(cp) {
                                Some(c) => out.push(c),
                                None => {
                                    return Err(self.error(
                                        "Unpaired surrogate escape (not representable)",
                                        escape_at,
                                    ))
                                }
                            }
                        }
                        _ => return Err(self.error("Invalid \\escape", escape_at)),
                    }
                }
                Some(_) => return Err(self.error("Invalid control character at", self.pos)),
            }
        }
    }

    fn object(&mut self) -> Result<Value, String> {
        let mut dict = Dict::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Dict(dict));
        }
        loop {
            if self.peek() != Some(b'"') {
                return Err(self.error(
                    "Expecting property name enclosed in double quotes",
                    self.pos,
                ));
            }
            self.pos += 1;
            let key = self.string()?;
            self.skip_whitespace();
            if self.peek() != Some(b':') {
                return Err(self.error("Expecting ':' delimiter", self.pos));
            }
            self.pos += 1;
            self.skip_whitespace();
            let value = self.value()?;
            dict.insert(key, value);
            self.skip_whitespace();
            match self.peek() {
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Value::Dict(dict));
                }
                Some(b',') => {
                    self.pos += 1;
                    self.skip_whitespace();
                }
                _ => return Err(self.error("Expecting ',' delimiter", self.pos)),
            }
        }
    }

    fn array(&mut self) -> Result<Value, String> {
        let mut list = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::List(list));
        }
        loop {
            list.push(self.value()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Value::List(list));
                }
                Some(b',') => {
                    self.pos += 1;
                    self.skip_whitespace();
                }
                _ => return Err(self.error("Expecting ',' delimiter", self.pos)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(text: &str) -> String {
        dumps(&parse_str(text).unwrap(), false, None)
    }

    #[test]
    fn dumps_matches_python_spelling() {
        assert_eq!(
            roundtrip(r#"{"b": [1, 2.5, true, null], "a": {}}"#),
            r#"{"b": [1, 2.5, true, null], "a": {}}"#
        );
        let value = parse_str(r#"{"b": 1, "a": [1e16, 1.5e-05, 0.0001, -0.0, 100.0]}"#).unwrap();
        assert_eq!(
            dumps(&value, true, None),
            r#"{"a": [1e+16, 1.5e-05, 0.0001, -0.0, 100.0], "b": 1}"#
        );
        assert_eq!(
            dumps(&value, true, Some(2)),
            "{\n  \"a\": [\n    1e+16,\n    1.5e-05,\n    0.0001,\n    -0.0,\n    100.0\n  ],\n  \"b\": 1\n}"
        );
        assert_eq!(
            roundtrip(r#""caf\u00e9 \ud83d\ude00 \u007f \"q\" \\ \/""#),
            r#""caf\u00e9 \ud83d\ude00 \u007f \"q\" \\ /""#
        );
        assert_eq!(
            roundtrip("[NaN, Infinity, -Infinity]"),
            "[NaN, Infinity, -Infinity]"
        );
    }

    #[test]
    fn objects_keep_document_order_and_python_duplicate_semantics() {
        let value = parse_str(r#"{"z": 1, "a": 2, "z": 3}"#).unwrap();
        assert_eq!(dumps(&value, false, None), r#"{"z": 3, "a": 2}"#);
    }

    #[test]
    fn python_rejections() {
        for bad in [
            "",
            "01",
            "1.",
            ".5",
            "[1,]",
            "{\"a\":1,}",
            "\"\t\"",
            "{'a': 1}",
            "[1] x",
            "\u{feff}{}",
            "\"\\ud800\"",
        ] {
            assert!(parse_str(bad).is_err(), "{bad:?} must be rejected");
        }
        assert!(matches!(parse_str("1e5").unwrap(), Value::Float(f) if f == 1e5));
        assert!(matches!(parse_str("-0").unwrap(), Value::Int(0)));
        assert!(parse_str(&"[".repeat(MAX_DEPTH + 1)).is_err());
        let deep = format!("{}{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
        assert!(parse_str(&deep).is_ok());
    }

    #[test]
    fn bytes_detect_the_encoding_like_python() {
        let utf16: Vec<u8> = "{\"a\": 1}"
            .encode_utf16()
            .flat_map(u16::to_le_bytes)
            .collect();
        assert_eq!(
            dumps(&parse_bytes(&utf16).unwrap(), false, None),
            r#"{"a": 1}"#
        );
        let mut signed = vec![0xef, 0xbb, 0xbf];
        signed.extend_from_slice(b"[1]");
        assert_eq!(dumps(&parse_bytes(&signed).unwrap(), false, None), "[1]");
        assert!(parse_bytes(b"\xff not json").is_err());
    }

    #[test]
    fn python_equality_and_repr() {
        assert_eq!(Value::Bool(true), Value::Int(1));
        assert_eq!(Value::Int(3), Value::Float(3.0));
        assert_ne!(Value::Int(3), Value::Float(3.5));
        assert_ne!(Value::Str("1".into()), Value::Int(1));
        assert_eq!(Value::Str("it's".into()).repr(), "\"it's\"");
        assert_eq!(
            Value::Str("a'b\"c\n\u{7f}\u{e9}".into()).repr(),
            "'a\\'b\"c\\n\\x7f\u{e9}'"
        );
        let list = parse_str(r#"["x", null, 1, 2.0, true]"#).unwrap();
        assert_eq!(list.repr(), "['x', None, 1, 2.0, True]");
        assert_eq!(float_repr(1e16), "1e+16");
        assert_eq!(float_repr(123456789012345.0), "123456789012345.0");
    }
}

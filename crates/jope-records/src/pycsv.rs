//! Python's `csv` module for the default (`excel`) dialect: `csv.reader`, `csv.DictReader` and
//! `csv.writer(lineterminator="\n")`, ported from CPython 3.12's `_csv.c` state machine so that
//! records the Python replay read and wrote are read and written identically here.

use crate::pyjson::{Dict, Value};
use crate::{Error, Result};

/// `csv.field_size_limit()` default.
pub const FIELD_LIMIT: usize = 131_072;

/// The key under which `csv.DictReader` keeps the extra fields of a long row (Python's
/// `restkey=None`). A header cell spelling exactly this string is not supported.
pub const REST_KEY: &str = "\u{0}restkey";

/// How the text was split into lines before the reader saw it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Newlines {
    /// `io.StringIO(text, newline="")`: lines end at `\n`, `\r` or `\r\n`, untranslated.
    Universal,
    /// `io.StringIO(text)`: lines end at `\n` only.
    LineFeed,
}

fn lines(text: &str, mode: Newlines) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < bytes.len() {
        let end = match (mode, bytes[i]) {
            (_, b'\n') => Some(i + 1),
            (Newlines::Universal, b'\r') => Some(if bytes.get(i + 1) == Some(&b'\n') {
                i + 2
            } else {
                i + 1
            }),
            _ => None,
        };
        match end {
            Some(end) => {
                out.push(&text[start..end]);
                start = end;
                i = end;
            }
            None => i += 1,
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    StartRecord,
    StartField,
    InField,
    InQuotedField,
    QuoteInQuotedField,
    EatCrnl,
}

struct Parser {
    state: State,
    fields: Vec<String>,
    field: String,
    field_len: usize,
}

impl Parser {
    fn save_field(&mut self) {
        self.fields.push(std::mem::take(&mut self.field));
        self.field_len = 0;
    }

    fn add_char(&mut self, c: char) -> Result<()> {
        if self.field_len >= FIELD_LIMIT {
            return Err(Error::Shape(format!(
                "field larger than field limit ({FIELD_LIMIT})"
            )));
        }
        self.field.push(c);
        self.field_len += 1;
        Ok(())
    }

    /// End of a record: save the field, then expect the next record (at the end-of-line marker)
    /// or swallow the rest of a `\r\n` pair.
    fn end_record(&mut self, c: Option<char>) {
        self.save_field();
        self.state = if c.is_none() {
            State::StartRecord
        } else {
            State::EatCrnl
        };
    }

    /// One character, or the end-of-line marker (`None`).
    fn process(&mut self, c: Option<char>) -> Result<()> {
        match (self.state, c) {
            (State::StartRecord, None) => {}
            (State::StartRecord, Some('\n' | '\r')) => self.state = State::EatCrnl,
            (State::StartRecord, Some(_)) => {
                self.state = State::StartField;
                return self.process(c);
            }
            (State::StartField, None | Some('\n' | '\r')) => self.end_record(c),
            (State::StartField, Some('"')) => self.state = State::InQuotedField,
            (State::StartField, Some(',')) => self.save_field(),
            (State::StartField, Some(ch)) => {
                self.add_char(ch)?;
                self.state = State::InField;
            }
            (State::InField, None | Some('\n' | '\r')) => self.end_record(c),
            (State::InField, Some(',')) => {
                self.save_field();
                self.state = State::StartField;
            }
            (State::InField, Some(ch)) => self.add_char(ch)?,
            (State::InQuotedField, None) => {}
            (State::InQuotedField, Some('"')) => self.state = State::QuoteInQuotedField,
            (State::InQuotedField, Some(ch)) => self.add_char(ch)?,
            (State::QuoteInQuotedField, Some('"')) => {
                self.add_char('"')?;
                self.state = State::InQuotedField;
            }
            (State::QuoteInQuotedField, Some(',')) => {
                self.save_field();
                self.state = State::StartField;
            }
            (State::QuoteInQuotedField, None | Some('\n' | '\r')) => self.end_record(c),
            (State::QuoteInQuotedField, Some(ch)) => {
                self.add_char(ch)?;
                self.state = State::InField;
            }
            (State::EatCrnl, Some('\n' | '\r')) => {}
            (State::EatCrnl, None) => self.state = State::StartRecord,
            (State::EatCrnl, Some(_)) => {
                return Err(Error::Shape(
                    "new-line character seen in unquoted field - do you need to open the file \
                     with newline=''?"
                        .into(),
                ))
            }
        }
        Ok(())
    }
}

/// `list(csv.reader(io.StringIO(text, ...)))`. A blank line is an empty record, as in Python.
pub fn reader(text: &str, mode: Newlines) -> Result<Vec<Vec<String>>> {
    let mut records = Vec::new();
    let mut parser = Parser {
        state: State::StartRecord,
        fields: Vec::new(),
        field: String::new(),
        field_len: 0,
    };
    for line in lines(text, mode) {
        for c in line.chars() {
            parser.process(Some(c))?;
        }
        parser.process(None)?;
        if parser.state == State::StartRecord {
            records.push(std::mem::take(&mut parser.fields));
        }
    }
    // End of input inside a quoted field (or with a pending field) still yields the record.
    if parser.field_len != 0 || parser.state == State::InQuotedField {
        parser.save_field();
        records.push(parser.fields);
    }
    Ok(records)
}

/// `csv.DictReader`: its field names (`None` for empty input) and rows as dicts. Blank records
/// are skipped; a short row maps the missing names to `None`, a long row keeps its extra fields
/// as a list under [`REST_KEY`].
pub struct DictRows {
    pub fieldnames: Option<Vec<String>>,
    pub rows: Vec<Dict>,
}

pub fn dict_reader(text: &str, mode: Newlines) -> Result<DictRows> {
    let mut records = reader(text, mode)?.into_iter();
    let Some(fieldnames) = records.next() else {
        return Ok(DictRows {
            fieldnames: None,
            rows: Vec::new(),
        });
    };
    let mut rows = Vec::new();
    for record in records {
        if record.is_empty() {
            continue;
        }
        let mut row = Dict::new();
        for (name, value) in fieldnames.iter().zip(&record) {
            row.insert(name.clone(), Value::Str(value.clone()));
        }
        if record.len() > fieldnames.len() {
            let rest = record[fieldnames.len()..]
                .iter()
                .map(|v| Value::Str(v.clone()))
                .collect();
            row.insert(REST_KEY, Value::List(rest));
        } else {
            for name in &fieldnames[record.len()..] {
                row.insert(name.clone(), Value::Null);
            }
        }
        rows.push(row);
    }
    Ok(DictRows {
        fieldnames: Some(fieldnames),
        rows,
    })
}

/// A value as `csv.writer` writes it: `None` is empty, anything else is `str(value)`.
pub fn cell(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        other => other.py_str(),
    }
}

/// `csv.writer(out, lineterminator="\n").writerow(fields)` with minimal quoting.
pub fn write_row<S: AsRef<str>>(out: &mut String, fields: &[S]) {
    if fields.len() == 1 && fields[0].as_ref().is_empty() {
        // A record of one empty field is written quoted, so it reads back as [""], not [].
        out.push_str("\"\"\n");
        return;
    }
    for (n, field) in fields.iter().enumerate() {
        if n > 0 {
            out.push(',');
        }
        let field = field.as_ref();
        if field.contains([',', '"', '\n', '\r']) {
            out.push('"');
            out.push_str(&field.replace('"', "\"\""));
            out.push('"');
        } else {
            out.push_str(field);
        }
    }
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(text: &str) -> Vec<Vec<String>> {
        reader(text, Newlines::Universal).unwrap()
    }

    #[test]
    fn reader_matches_python() {
        assert_eq!(read("a,b\r\nc,d\n"), [vec!["a", "b"], vec!["c", "d"]]);
        assert_eq!(read("a\n\nb"), [vec!["a"], vec![], vec!["b"]]);
        assert_eq!(read("\"x,\"\"y\"\"\",z\n"), [vec!["x,\"y\"", "z"]]);
        assert_eq!(read("\"a\"b,c\n"), [vec!["ab", "c"]]);
        assert_eq!(read("a\"b,\n"), [vec!["a\"b", ""]]);
        assert_eq!(read("\"multi\nline\",x\n"), [vec!["multi\nline", "x"]]);
        assert_eq!(read("\"open"), [vec!["open"]]);
        assert_eq!(read(",\n"), [vec!["", ""]]);
        assert_eq!(read("a\rb\r"), [vec!["a"], vec!["b"]]);
        assert!(read("").is_empty());
        // Without universal newlines a lone carriage return inside a record is refused.
        assert!(reader("a\rb\n", Newlines::LineFeed).is_err());
        assert!(reader(&"x".repeat(FIELD_LIMIT + 1), Newlines::Universal).is_err());
    }

    #[test]
    fn dict_reader_matches_python() {
        let rows = dict_reader("a,b\n1\n\n1,2,3\n", Newlines::Universal).unwrap();
        assert_eq!(
            rows.fieldnames.as_deref(),
            Some(&["a".to_owned(), "b".to_owned()][..])
        );
        assert_eq!(rows.rows.len(), 2);
        assert_eq!(rows.rows[0].get("b"), Some(&Value::Null));
        assert_eq!(
            rows.rows[1].get(REST_KEY).map(Value::repr).as_deref(),
            Some("['3']")
        );
        assert!(dict_reader("", Newlines::Universal)
            .unwrap()
            .fieldnames
            .is_none());
    }

    #[test]
    fn writer_matches_python() {
        let mut out = String::new();
        write_row(&mut out, &["a", "b,c", "d\"e", "", "f\ng"]);
        write_row(&mut out, &[""]);
        write_row(&mut out, &["", ""]);
        assert_eq!(out, "a,\"b,c\",\"d\"\"e\",,\"f\ng\"\n\"\"\n,\n");
        let back = reader(&out, Newlines::Universal).unwrap();
        assert_eq!(back[0], ["a", "b,c", "d\"e", "", "f\ng"]);
        assert_eq!(back[1], [""]);
        assert_eq!(cell(&Value::Null), "");
        assert_eq!(cell(&Value::Float(1500.0)), "1500.0");
        assert_eq!(cell(&Value::Bool(true)), "True");
    }
}

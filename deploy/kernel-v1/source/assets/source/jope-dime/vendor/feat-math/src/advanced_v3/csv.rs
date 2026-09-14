use super::schema::DataSchema;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RawTable {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl RawTable {
    pub fn new(headers: Vec<String>, rows: Vec<Vec<String>>) -> Result<Self> {
        let table = Self { headers, rows };
        table.validate_rectangular()?;
        Ok(table)
    }

    /// Parse RFC-4180-style quoted CSV, including commas, CRLF, embedded
    /// newlines, and doubled quote escapes. The first record is the header.
    pub fn from_csv(csv: &str) -> Result<Self> {
        let records = parse_csv_records(csv)?;
        let (headers, rows) = records.split_first().context("raw CSV has no header")?;
        Self::new(headers.clone(), rows.to_vec())
    }

    pub fn validate_against(&self, schema: &DataSchema) -> Result<()> {
        self.validate_rectangular()?;
        schema.validate()?;
        let expected: Vec<&str> = schema.columns.iter().map(|c| c.name.as_str()).collect();
        let actual: Vec<&str> = self.headers.iter().map(String::as_str).collect();
        if actual != expected {
            bail!(
                "raw table headers do not match bound schema (expected {:?}, found {:?})",
                expected,
                actual
            );
        }
        Ok(())
    }

    fn validate_rectangular(&self) -> Result<()> {
        if self.headers.is_empty() {
            bail!("raw table has no columns");
        }
        let mut headers = BTreeSet::new();
        for header in &self.headers {
            if header.trim().is_empty() {
                bail!("raw table header names must not be blank");
            }
            if !headers.insert(header.as_str()) {
                bail!("duplicate raw table header '{header}'");
            }
        }
        for (row, values) in self.rows.iter().enumerate() {
            if values.len() != self.headers.len() {
                bail!(
                    "raw table row {} has {} fields; expected {}",
                    row + 1,
                    values.len(),
                    self.headers.len()
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CsvFieldState {
    Start,
    Unquoted,
    Quoted,
    AfterQuote,
}

fn parse_csv_records(csv: &str) -> Result<Vec<Vec<String>>> {
    let mut records = Vec::<Vec<String>>::new();
    let mut row = Vec::<String>::new();
    let mut field = String::new();
    let mut chars = csv.chars().peekable();
    let mut state = CsvFieldState::Start;
    while let Some(ch) = chars.next() {
        match state {
            CsvFieldState::Quoted => {
                if ch == '"' {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        field.push('"');
                    } else {
                        state = CsvFieldState::AfterQuote;
                    }
                } else {
                    field.push(ch);
                }
            }
            CsvFieldState::AfterQuote => match ch {
                ',' => {
                    row.push(std::mem::take(&mut field));
                    state = CsvFieldState::Start;
                }
                '\n' | '\r' => {
                    if ch == '\r' && chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    row.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut row));
                    state = CsvFieldState::Start;
                }
                other => bail!("junk after closing quote in CSV field: '{other}'"),
            },
            CsvFieldState::Start => match ch {
                '"' => state = CsvFieldState::Quoted,
                ',' => row.push(String::new()),
                '\n' | '\r' => {
                    if ch == '\r' && chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    if row.is_empty() {
                        records.push(Vec::new());
                    } else {
                        row.push(String::new());
                        records.push(std::mem::take(&mut row));
                    }
                }
                other => {
                    field.push(other);
                    state = CsvFieldState::Unquoted;
                }
            },
            CsvFieldState::Unquoted => match ch {
                ',' => {
                    row.push(std::mem::take(&mut field));
                    state = CsvFieldState::Start;
                }
                '\n' | '\r' => {
                    if ch == '\r' && chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    row.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut row));
                    state = CsvFieldState::Start;
                }
                '"' => bail!("quote in unquoted CSV field"),
                other => field.push(other),
            },
        }
    }
    match state {
        CsvFieldState::Quoted => bail!("unterminated quoted CSV field"),
        CsvFieldState::AfterQuote | CsvFieldState::Unquoted => {
            row.push(field);
            records.push(row);
        }
        CsvFieldState::Start if !row.is_empty() => {
            row.push(field);
            records.push(row);
        }
        CsvFieldState::Start => {}
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_parser_accepts_supported_quoted_csv() {
        let records =
            parse_csv_records("a,b,c\r\n1,\"x,y\",\"say \"\"hi\"\"\"\n2,\"two\r\nlines\",é\r3,4,")
                .unwrap();
        assert_eq!(records[0], ["a", "b", "c"]);
        assert_eq!(records[1], ["1", "x,y", "say \"hi\""]);
        assert_eq!(records[2], ["2", "two\r\nlines", "é"]);
        assert_eq!(records[3], ["3", "4", ""]);
    }

    #[test]
    fn empty_record_and_one_empty_field_are_distinct() {
        let records = parse_csv_records("\n\"\"\n").unwrap();
        assert_eq!(records, vec![Vec::<String>::new(), vec![String::new()]]);
    }

    #[test]
    fn malformed_quotes_are_rejected() {
        assert!(parse_csv_records("a\n\"unterminated").is_err());
        assert!(parse_csv_records("a\nvalue\"quote").is_err());
        assert!(parse_csv_records("a\n\"closed\"junk").is_err());
        assert!(parse_csv_records("a\n\"closed\" ").is_err());
    }

    #[test]
    fn raw_tables_reject_invalid_headers_and_ragged_rows() {
        assert!(RawTable::new(vec![" ".into()], vec![]).is_err());
        assert!(RawTable::new(vec!["a".into(), "a".into()], vec![]).is_err());
        assert!(RawTable::new(vec!["a".into(), "b".into()], vec![vec!["1".into()]]).is_err());
        assert!(RawTable::from_csv("a\n\n").is_err());
    }
}

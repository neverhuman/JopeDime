//! Small built-in lexer for the three terminal code-preview languages.

use super::ThemeToken;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Language {
    Rust,
    Python,
    JavaScript,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Span<'a> {
    pub text: &'a str,
    pub token: Option<ThemeToken>,
}

#[derive(Clone, Copy)]
struct MarkedRange {
    start: usize,
    end: usize,
    token: Option<ThemeToken>,
}

/// Split source into text spans tagged with the matching theme token.
///
/// This is a display lexer, not a parser. Unknown syntax is returned unchanged without a token.
pub fn highlight(source: &str, language: Language) -> Vec<Span<'_>> {
    let bytes = source.as_bytes();
    let mut ranges: Vec<MarkedRange> = Vec::new();
    let mut index = 0;

    while index < bytes.len() {
        let start = index;

        if let Some((end, token)) = scan_comment(bytes, index, language) {
            push_range(&mut ranges, start, end, Some(token));
            index = end;
            continue;
        }

        if let Some(end) = scan_raw_rust_string(bytes, index, language) {
            push_range(&mut ranges, start, end, Some(ThemeToken::CodeString));
            index = end;
            continue;
        }

        if let Some((end, is_string)) = scan_quoted_string(bytes, index, language) {
            if is_string {
                push_range(&mut ranges, start, end, Some(ThemeToken::CodeString));
                index = end;
                continue;
            }
        }

        if bytes[index].is_ascii_digit() {
            let end = scan_number(bytes, index);
            push_range(&mut ranges, start, end, Some(ThemeToken::CodeNumber));
            index = end;
            continue;
        }

        if is_identifier_start(bytes[index]) {
            let end = scan_identifier(bytes, index);
            let identifier = &source[start..end];
            let token = is_keyword(identifier, language).then_some(ThemeToken::CodeKeyword);
            push_range(&mut ranges, start, end, token);
            index = end;
            continue;
        }

        index += char_width(bytes[index]);
        push_range(&mut ranges, start, index, None);
    }

    ranges
        .into_iter()
        .map(|range| Span {
            text: &source[range.start..range.end],
            token: range.token,
        })
        .collect()
}

fn push_range(ranges: &mut Vec<MarkedRange>, start: usize, end: usize, token: Option<ThemeToken>) {
    if start == end {
        return;
    }
    if let Some(previous) = ranges.last_mut() {
        if previous.end == start && previous.token == token {
            previous.end = end;
            return;
        }
    }
    ranges.push(MarkedRange { start, end, token });
}

fn scan_comment(bytes: &[u8], index: usize, language: Language) -> Option<(usize, ThemeToken)> {
    let line_comment = match language {
        Language::Rust | Language::JavaScript => bytes.get(index..index + 2) == Some(b"//"),
        Language::Python => bytes.get(index) == Some(&b'#'),
    };
    if line_comment {
        let mut end = index;
        while end < bytes.len() && bytes[end] != b'\n' {
            end += char_width(bytes[end]);
        }
        return Some((end, ThemeToken::CodeComment));
    }

    let block_comment = matches!(language, Language::Rust | Language::JavaScript)
        && bytes.get(index..index + 2) == Some(b"/*");
    if !block_comment {
        return None;
    }

    let rust_nested = language == Language::Rust;
    let mut depth = 1usize;
    let mut end = index + 2;
    while end < bytes.len() {
        if rust_nested && bytes.get(end..end + 2) == Some(b"/*") {
            depth += 1;
            end += 2;
        } else if bytes.get(end..end + 2) == Some(b"*/") {
            depth -= 1;
            end += 2;
            if depth == 0 {
                break;
            }
        } else {
            end += char_width(bytes[end]);
        }
    }
    Some((end, ThemeToken::CodeComment))
}

fn scan_raw_rust_string(bytes: &[u8], index: usize, language: Language) -> Option<usize> {
    if language != Language::Rust || bytes.get(index) != Some(&b'r') {
        return None;
    }
    let mut cursor = index + 1;
    while bytes.get(cursor) == Some(&b'#') {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'"') {
        return None;
    }

    let hashes = cursor - index - 1;
    cursor += 1;
    while cursor < bytes.len() {
        if bytes[cursor] == b'"'
            && bytes.get(cursor + 1..cursor + 1 + hashes)
                == Some(&bytes[index + 1..index + 1 + hashes])
        {
            return Some(cursor + 1 + hashes);
        }
        cursor += char_width(bytes[cursor]);
    }
    Some(bytes.len())
}

fn scan_quoted_string(bytes: &[u8], index: usize, language: Language) -> Option<(usize, bool)> {
    let quote = *bytes.get(index)?;
    let is_quote =
        matches!(quote, b'\'' | b'"') || (language == Language::JavaScript && quote == b'`');
    if !is_quote {
        return None;
    }

    let triple =
        language == Language::Python && bytes.get(index..index + 3) == Some(&[quote, quote, quote]);
    if language == Language::Rust && quote == b'\'' {
        let end = scan_char_literal(bytes, index);
        return Some((end.unwrap_or(index + 1), end.is_some()));
    }

    let delimiter = if triple { 3 } else { 1 };
    let mut cursor = index + delimiter;
    while cursor < bytes.len() {
        if bytes[cursor] == b'\\' {
            cursor += 1;
            if cursor < bytes.len() {
                cursor += char_width(bytes[cursor]);
            }
            continue;
        }
        if triple {
            if bytes.get(cursor..cursor + 3) == Some(&[quote, quote, quote]) {
                return Some((cursor + 3, true));
            }
        } else if bytes[cursor] == quote {
            return Some((cursor + 1, true));
        }
        cursor += char_width(bytes[cursor]);
    }
    // Unterminated strings are still useful to color in a preview and never panic.
    Some((bytes.len(), true))
}

fn scan_char_literal(bytes: &[u8], index: usize) -> Option<usize> {
    let mut cursor = index + 1;
    while cursor < bytes.len() && bytes[cursor] != b'\n' {
        if bytes[cursor] == b'\\' {
            cursor += 1;
            if cursor < bytes.len() {
                cursor += char_width(bytes[cursor]);
            }
            continue;
        }
        if bytes[cursor] == b'\'' {
            return Some(cursor + 1);
        }
        cursor += char_width(bytes[cursor]);
    }
    None
}

fn scan_number(bytes: &[u8], index: usize) -> usize {
    let mut cursor = index;
    if bytes.get(cursor) == Some(&b'0') {
        if let Some(prefix) = bytes.get(cursor + 1).copied() {
            if matches!(prefix, b'x' | b'X' | b'o' | b'O' | b'b' | b'B') {
                cursor += 2;
                while bytes
                    .get(cursor)
                    .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
                {
                    cursor += 1;
                }
                return cursor;
            }
        }
    }

    while bytes
        .get(cursor)
        .is_some_and(|byte| byte.is_ascii_digit() || *byte == b'_')
    {
        cursor += 1;
    }
    if bytes.get(cursor) == Some(&b'.') && bytes.get(cursor + 1).is_some_and(u8::is_ascii_digit) {
        cursor += 1;
        while bytes
            .get(cursor)
            .is_some_and(|byte| byte.is_ascii_digit() || *byte == b'_')
        {
            cursor += 1;
        }
    }
    if bytes
        .get(cursor)
        .is_some_and(|byte| matches!(byte, b'e' | b'E'))
    {
        let exponent = cursor;
        cursor += 1;
        if bytes
            .get(cursor)
            .is_some_and(|byte| matches!(byte, b'+' | b'-'))
        {
            cursor += 1;
        }
        let digits = cursor;
        while bytes
            .get(cursor)
            .is_some_and(|byte| byte.is_ascii_digit() || *byte == b'_')
        {
            cursor += 1;
        }
        if digits == cursor {
            cursor = exponent;
        }
    }
    while bytes
        .get(cursor)
        .is_some_and(|byte| byte.is_ascii_alphabetic())
    {
        cursor += 1;
    }
    cursor
}

fn scan_identifier(bytes: &[u8], index: usize) -> usize {
    let mut cursor = index + 1;
    while bytes
        .get(cursor)
        .is_some_and(|byte| is_identifier_continue(*byte))
    {
        cursor += 1;
    }
    cursor
}

fn is_identifier_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn is_identifier_continue(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn char_width(first_byte: u8) -> usize {
    match first_byte {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

fn is_keyword(identifier: &str, language: Language) -> bool {
    match language {
        Language::Rust => matches!(
            identifier,
            "as" | "async"
                | "await"
                | "break"
                | "const"
                | "continue"
                | "crate"
                | "else"
                | "enum"
                | "extern"
                | "false"
                | "fn"
                | "for"
                | "if"
                | "impl"
                | "in"
                | "let"
                | "loop"
                | "match"
                | "mod"
                | "move"
                | "mut"
                | "pub"
                | "ref"
                | "return"
                | "self"
                | "Self"
                | "static"
                | "struct"
                | "super"
                | "trait"
                | "true"
                | "type"
                | "unsafe"
                | "use"
                | "where"
                | "while"
                | "i8"
                | "i16"
                | "i32"
                | "i64"
                | "i128"
                | "isize"
                | "u8"
                | "u16"
                | "u32"
                | "u64"
                | "u128"
                | "usize"
                | "f32"
                | "f64"
                | "bool"
                | "char"
                | "str"
        ),
        Language::Python => matches!(
            identifier,
            "and"
                | "as"
                | "assert"
                | "async"
                | "await"
                | "break"
                | "class"
                | "continue"
                | "def"
                | "del"
                | "elif"
                | "else"
                | "except"
                | "False"
                | "finally"
                | "for"
                | "from"
                | "global"
                | "if"
                | "import"
                | "in"
                | "is"
                | "lambda"
                | "nonlocal"
                | "not"
                | "or"
                | "pass"
                | "raise"
                | "return"
                | "True"
                | "try"
                | "while"
                | "with"
                | "yield"
                | "None"
        ),
        Language::JavaScript => matches!(
            identifier,
            "async"
                | "await"
                | "break"
                | "case"
                | "catch"
                | "class"
                | "const"
                | "continue"
                | "debugger"
                | "default"
                | "delete"
                | "do"
                | "else"
                | "export"
                | "extends"
                | "false"
                | "finally"
                | "for"
                | "from"
                | "function"
                | "if"
                | "import"
                | "in"
                | "instanceof"
                | "let"
                | "new"
                | "null"
                | "of"
                | "return"
                | "super"
                | "switch"
                | "this"
                | "throw"
                | "true"
                | "try"
                | "typeof"
                | "var"
                | "void"
                | "while"
                | "with"
                | "yield"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{highlight, Language, Span};
    use crate::terminal::ThemeToken;

    fn expected<'a>(items: &[(&'a str, Option<ThemeToken>)]) -> Vec<Span<'a>> {
        items
            .iter()
            .map(|(text, token)| Span {
                text,
                token: *token,
            })
            .collect()
    }

    #[test]
    fn rust_golden_keywords_numbers_strings_and_comments() {
        let source = "fn add(x: i32) -> i32 { // sum\n    let n = 42; \"ok\"\n}";
        assert_eq!(
            highlight(source, Language::Rust),
            expected(&[
                ("fn", Some(ThemeToken::CodeKeyword)),
                (" add(x: ", None),
                ("i32", Some(ThemeToken::CodeKeyword)),
                (") -> ", None,),
                ("i32", Some(ThemeToken::CodeKeyword)),
                (" { ", None),
                ("// sum", Some(ThemeToken::CodeComment)),
                ("\n    ", None),
                ("let", Some(ThemeToken::CodeKeyword)),
                (" n = ", None),
                ("42", Some(ThemeToken::CodeNumber)),
                ("; ", None),
                ("\"ok\"", Some(ThemeToken::CodeString)),
                ("\n}", None),
            ])
        );
    }

    #[test]
    fn python_golden_keywords_numbers_strings_and_comments() {
        let source = "def area(x):\n    return x * 2.5 # scale";
        assert_eq!(
            highlight(source, Language::Python),
            expected(&[
                ("def", Some(ThemeToken::CodeKeyword)),
                (" area(x):\n    ", None),
                ("return", Some(ThemeToken::CodeKeyword)),
                (" x * ", None),
                ("2.5", Some(ThemeToken::CodeNumber)),
                (" ", None),
                ("# scale", Some(ThemeToken::CodeComment)),
            ])
        );
    }

    #[test]
    fn javascript_golden_keywords_numbers_strings_and_block_comments() {
        let source = "const value = 12; /* cached */ return `ok`;";
        assert_eq!(
            highlight(source, Language::JavaScript),
            expected(&[
                ("const", Some(ThemeToken::CodeKeyword)),
                (" value = ", None),
                ("12", Some(ThemeToken::CodeNumber)),
                ("; ", None),
                ("/* cached */", Some(ThemeToken::CodeComment)),
                (" ", None),
                ("return", Some(ThemeToken::CodeKeyword)),
                (" ", None),
                ("`ok`", Some(ThemeToken::CodeString)),
                (";", None),
            ])
        );
    }

    #[test]
    fn unterminated_strings_are_returned_as_string_spans_without_panicking() {
        for (source, language) in [
            ("let text = \"open", Language::Rust),
            ("text = 'open", Language::Python),
            ("const text = `open", Language::JavaScript),
        ] {
            let spans = highlight(source, language);
            assert_eq!(
                spans.last().map(|span| span.token),
                Some(Some(ThemeToken::CodeString))
            );
            assert_eq!(
                spans.iter().map(|span| span.text).collect::<String>(),
                source
            );
        }
    }

    #[test]
    fn rust_raw_strings_nested_comments_and_lifetimes_are_safe() {
        let source = "let s = r##\"raw\"##; /* outer /* inner */ end */ 'a";
        let spans = highlight(source, Language::Rust);
        assert!(spans.iter().any(|span| {
            span.text == "r##\"raw\"##" && span.token == Some(ThemeToken::CodeString)
        }));
        assert!(spans.iter().any(|span| {
            span.text == "/* outer /* inner */ end */"
                && span.token == Some(ThemeToken::CodeComment)
        }));
        assert!(spans
            .iter()
            .any(|span| span.text.contains("'a") && span.token.is_none()));
    }
}

use super::*;

pub(super) fn push_range(
    ranges: &mut Vec<MarkedRange>,
    start: usize,
    end: usize,
    token: Option<ThemeToken>,
) {
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

pub(super) fn scan_comment(
    bytes: &[u8],
    index: usize,
    language: HighlightLanguage,
) -> Option<(usize, ThemeToken)> {
    let line_comment = match language {
        HighlightLanguage::Rust | HighlightLanguage::JavaScript => {
            bytes.get(index..index + 2) == Some(b"//")
        }
        HighlightLanguage::Python => bytes.get(index) == Some(&b'#'),
    };
    if line_comment {
        let mut end = index;
        while end < bytes.len() && bytes[end] != b'\n' {
            end += char_width(bytes[end]);
        }
        return Some((end, ThemeToken::CodeComment));
    }

    let block_comment = matches!(
        language,
        HighlightLanguage::Rust | HighlightLanguage::JavaScript
    ) && bytes.get(index..index + 2) == Some(b"/*");
    if !block_comment {
        return None;
    }

    let rust_nested = language == HighlightLanguage::Rust;
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

pub(super) fn scan_raw_rust_string(
    bytes: &[u8],
    index: usize,
    language: HighlightLanguage,
) -> Option<usize> {
    if language != HighlightLanguage::Rust || bytes.get(index) != Some(&b'r') {
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

pub(super) fn scan_quoted_string(
    bytes: &[u8],
    index: usize,
    language: HighlightLanguage,
) -> Option<(usize, bool)> {
    let quote = *bytes.get(index)?;
    let is_quote = matches!(quote, b'\'' | b'"')
        || (language == HighlightLanguage::JavaScript && quote == b'`');
    if !is_quote {
        return None;
    }

    let triple = language == HighlightLanguage::Python
        && bytes.get(index..index + 3) == Some(&[quote, quote, quote]);
    if language == HighlightLanguage::Rust && quote == b'\'' {
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

pub(super) fn scan_char_literal(bytes: &[u8], index: usize) -> Option<usize> {
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

pub(super) fn scan_number(bytes: &[u8], index: usize) -> usize {
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

pub(super) fn scan_identifier(bytes: &[u8], index: usize) -> usize {
    let mut cursor = index + 1;
    while bytes
        .get(cursor)
        .is_some_and(|byte| is_identifier_continue(*byte))
    {
        cursor += 1;
    }
    cursor
}

pub(super) fn is_identifier_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

pub(super) fn is_identifier_continue(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

pub(super) fn char_width(first_byte: u8) -> usize {
    match first_byte {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

pub(super) fn is_keyword(identifier: &str, language: HighlightLanguage) -> bool {
    match language {
        HighlightLanguage::Rust => matches!(
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
        HighlightLanguage::Python => matches!(
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
        HighlightLanguage::JavaScript => matches!(
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

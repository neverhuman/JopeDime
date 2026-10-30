//! Pinned source-token counts for telemetry, independent of model tokenizers.
//!
//! V1 counts identifiers/keywords, literals, maximal operators and punctuation.
//! Whitespace, Python layout tokens and comments contribute zero. Rust lifetimes
//! and raw identifiers each count once. Prefixed, formatted and template strings
//! each count as one literal (interpolation is opaque). JavaScript regex literals
//! use the expression-start lexical goal, including after control parentheses;
//! this lightweight counter is not a JavaScript parser. Counts, including these
//! conventions, must get a new version whenever their rules change.

use std::fmt;

pub const LEXER_VERSION: &str = "jope.source-lex/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceLanguage {
    Rust,
    Python,
    JavaScript,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LexError {
    /// UTF-8 byte offset at the opening delimiter.
    pub offset: usize,
    pub reason: &'static str,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at byte {}", self.reason, self.offset)
    }
}

impl std::error::Error for LexError {}

/// Count source tokens without combining languages or measuring elapsed time.
/// Unterminated comments/literals and excessively nested templates are errors.
pub fn count_tokens(language: SourceLanguage, source: &str) -> Result<usize, LexError> {
    let mut lexer = Lexer {
        language,
        source,
        pos: 0,
        expression_start: true,
        control: false,
        parens: Vec::new(),
        template_depth: 0,
        statement_start: true,
        objects: Vec::new(),
    };
    let mut count = 0;
    while lexer.next()?.is_some() {
        count += 1;
    }
    Ok(count)
}

struct Lexer<'a> {
    language: SourceLanguage,
    source: &'a str,
    pos: usize,
    expression_start: bool,
    control: bool,
    parens: Vec<bool>,
    template_depth: usize,
    statement_start: bool,
    objects: Vec<bool>,
}

impl<'a> Lexer<'a> {
    fn error(&self, offset: usize, reason: &'static str) -> LexError {
        LexError { offset, reason }
    }

    fn rest(&self) -> &'a str {
        &self.source[self.pos..]
    }

    fn next(&mut self) -> Result<Option<&'a str>, LexError> {
        self.trivia()?;
        if self.pos == self.source.len() {
            return Ok(None);
        }
        let start = self.pos;
        let first = self.rest().chars().next().unwrap();
        let mut is_literal = self.literal()?;
        if !is_literal {
            if first.is_ascii_digit()
                || (first == '.'
                    && self.language != SourceLanguage::Rust
                    && self
                        .rest()
                        .as_bytes()
                        .get(1)
                        .is_some_and(u8::is_ascii_digit))
            {
                self.number();
                is_literal = true;
            } else if identifier_start(first, self.language) {
                self.identifier();
            } else if self.language == SourceLanguage::JavaScript
                && first == '/'
                && self.expression_start
            {
                self.regex()?;
                is_literal = true;
            } else {
                self.operator();
            }
        }
        let token = &self.source[start..self.pos];
        self.js_context(token, is_literal);
        Ok(Some(token))
    }

    fn trivia(&mut self) -> Result<(), LexError> {
        loop {
            while let Some(c) = self
                .rest()
                .chars()
                .next()
                .filter(|c| c.is_whitespace() || *c == '\u{feff}')
            {
                self.pos += c.len_utf8();
            }
            let python = self.language == SourceLanguage::Python;
            if (python && self.rest().starts_with('#'))
                || (!python && self.rest().starts_with("//"))
                || (self.pos == 0
                    && self.rest().starts_with("#!")
                    && !(self.language == SourceLanguage::Rust && self.rest().starts_with("#![")))
            {
                self.pos += self.rest().find('\n').unwrap_or(self.rest().len());
            } else if !python && self.rest().starts_with("/*") {
                self.block_comment()?;
            } else if python
                && (self.rest().starts_with("\\\n") || self.rest().starts_with("\\\r\n"))
            {
                self.pos += if self.rest().starts_with("\\\r") {
                    3
                } else {
                    2
                };
            } else {
                return Ok(());
            }
        }
    }

    fn block_comment(&mut self) -> Result<(), LexError> {
        let start = self.pos;
        self.pos += 2;
        let mut depth = 1;
        while self.pos < self.source.len() {
            if self.rest().starts_with("*/") {
                self.pos += 2;
                depth -= 1;
                if depth == 0 {
                    return Ok(());
                }
            } else if self.language == SourceLanguage::Rust && self.rest().starts_with("/*") {
                depth += 1;
                self.pos += 2;
            } else {
                self.pos += self.rest().chars().next().unwrap().len_utf8();
            }
        }
        Err(self.error(start, "unterminated block comment"))
    }

    fn identifier(&mut self) {
        while let Some(c) = self.rest().chars().next().filter(|&c| {
            identifier_start(c, self.language)
                || c.is_numeric()
                || ('\u{300}'..='\u{36f}').contains(&c)
        }) {
            self.pos += c.len_utf8();
        }
    }

    fn literal(&mut self) -> Result<bool, LexError> {
        let start = self.pos;
        if self.language == SourceLanguage::Rust && self.rust_prefix()? {
            return Ok(true);
        }
        if self.language == SourceLanguage::Python {
            for prefix in ["br", "rb", "fr", "rf", "r", "b", "u", "f"] {
                if self
                    .rest()
                    .get(..prefix.len())
                    .is_some_and(|s| s.eq_ignore_ascii_case(prefix))
                    && self
                        .rest()
                        .as_bytes()
                        .get(prefix.len())
                        .is_some_and(|b| matches!(b, b'\'' | b'"'))
                {
                    self.pos += prefix.len();
                    break;
                }
            }
        }
        let quote = self.rest().as_bytes()[0];
        if matches!(quote, b'\'' | b'"') {
            self.quoted(start, quote)?;
            return Ok(true);
        }
        if self.language == SourceLanguage::JavaScript && quote == b'`' {
            self.template()?;
            return Ok(true);
        }
        Ok(false)
    }

    fn rust_prefix(&mut self) -> Result<bool, LexError> {
        let start = self.pos;
        for prefix in ["br", "cr", "r"] {
            if !self.rest().starts_with(prefix) {
                continue;
            }
            let hashes = self.rest()[prefix.len()..]
                .bytes()
                .take_while(|&b| b == b'#')
                .count();
            if self.rest().as_bytes().get(prefix.len() + hashes) == Some(&b'"') {
                self.pos += prefix.len() + hashes + 1;
                let closing = format!("\"{}", "#".repeat(hashes));
                let Some(end) = self.rest().find(&closing) else {
                    return Err(self.error(start, "unterminated raw string"));
                };
                self.pos += end + closing.len();
                return Ok(true);
            }
        }
        if self.rest().starts_with("r#")
            && self.rest()[2..]
                .chars()
                .next()
                .is_some_and(|c| identifier_start(c, self.language))
        {
            self.pos += 2;
            self.identifier();
            return Ok(true);
        }
        if self.rest().starts_with('\'')
            && self.rest()[1..]
                .chars()
                .next()
                .is_some_and(|c| identifier_start(c, self.language))
        {
            self.pos += 1;
            self.identifier();
            if self.rest().starts_with('\'') {
                self.pos += 1;
            }
            return Ok(true);
        }
        if self.rest().starts_with("b\"")
            || self.rest().starts_with("c\"")
            || self.rest().starts_with("b'")
        {
            self.pos += 1;
        }
        Ok(false)
    }

    fn quoted(&mut self, start: usize, quote: u8) -> Result<(), LexError> {
        let triple = self.language == SourceLanguage::Python
            && self.rest().as_bytes().starts_with(&[quote; 3]);
        let width = if triple { 3 } else { 1 };
        self.pos += width;
        while self.pos < self.source.len() {
            if self.rest().as_bytes().starts_with(&[quote; 3][..width]) {
                self.pos += width;
                return Ok(());
            }
            if self.rest().starts_with('\\') {
                self.pos += 1;
                if self.pos == self.source.len() {
                    break;
                }
            }
            self.pos += self.rest().chars().next().unwrap().len_utf8();
        }
        Err(self.error(start, "unterminated string or character literal"))
    }

    fn number(&mut self) {
        let start = self.pos;
        let bytes = self.source.as_bytes();
        let radix = bytes.get(start) == Some(&b'0')
            && bytes
                .get(start + 1)
                .is_some_and(|b| matches!(b, b'x' | b'X' | b'o' | b'O' | b'b' | b'B'));
        if radix {
            self.pos += 2;
            while bytes
                .get(self.pos)
                .is_some_and(|b| b.is_ascii_hexdigit() || *b == b'_')
            {
                self.pos += 1;
            }
        } else {
            while bytes
                .get(self.pos)
                .is_some_and(|b| b.is_ascii_digit() || *b == b'_')
            {
                self.pos += 1;
            }
            if bytes.get(self.pos) == Some(&b'.')
                && bytes.get(self.pos + 1) != Some(&b'.')
                && (self.language != SourceLanguage::Rust
                    || !self.source[self.pos + 1..]
                        .chars()
                        .next()
                        .is_some_and(|c| identifier_start(c, self.language)))
            {
                self.pos += 1;
                while bytes
                    .get(self.pos)
                    .is_some_and(|b| b.is_ascii_digit() || *b == b'_')
                {
                    self.pos += 1;
                }
            }
            if bytes
                .get(self.pos)
                .is_some_and(|b| matches!(b, b'e' | b'E'))
            {
                self.pos += 1;
                if bytes
                    .get(self.pos)
                    .is_some_and(|b| matches!(b, b'+' | b'-'))
                {
                    self.pos += 1;
                }
                while bytes
                    .get(self.pos)
                    .is_some_and(|b| b.is_ascii_digit() || *b == b'_')
                {
                    self.pos += 1;
                }
            }
        }
        if self.language == SourceLanguage::Rust {
            self.identifier();
        } else if (self.language == SourceLanguage::Python
            && bytes
                .get(self.pos)
                .is_some_and(|b| matches!(b, b'j' | b'J')))
            || (self.language == SourceLanguage::JavaScript && bytes.get(self.pos) == Some(&b'n'))
        {
            self.pos += 1;
        }
    }

    fn regex(&mut self) -> Result<(), LexError> {
        let start = self.pos;
        self.pos += 1;
        let mut class = false;
        while self.pos < self.source.len() {
            let c = self.rest().chars().next().unwrap();
            self.pos += c.len_utf8();
            match c {
                '\\' if self.pos < self.source.len() => {
                    self.pos += self.rest().chars().next().unwrap().len_utf8()
                }
                '[' => class = true,
                ']' => class = false,
                '/' if !class => {
                    self.identifier();
                    return Ok(());
                }
                '\n' | '\r' => break,
                _ => {}
            }
        }
        Err(self.error(start, "unterminated regular expression"))
    }

    fn template(&mut self) -> Result<(), LexError> {
        let start = self.pos;
        if self.template_depth == 128 {
            return Err(self.error(start, "template nesting exceeds v1 limit"));
        }
        self.template_depth += 1;
        self.pos += 1;
        while self.pos < self.source.len() {
            if self.rest().starts_with('`') {
                self.pos += 1;
                self.template_depth -= 1;
                return Ok(());
            }
            if self.rest().starts_with("${") {
                self.pos += 2;
                self.expression_start = true;
                self.statement_start = false;
                let mut braces = 1;
                while braces != 0 {
                    self.trivia()?;
                    // The hole's closing brace is not an object/block token.
                    if braces == 1 && self.rest().starts_with('}') {
                        self.pos += 1;
                        break;
                    }
                    match self.next()? {
                        Some("{") => braces += 1,
                        Some("}") => braces -= 1,
                        Some(_) => {}
                        None => return Err(self.error(start, "unterminated template expression")),
                    }
                }
            } else {
                if self.rest().starts_with('\\') {
                    self.pos += 1;
                }
                if self.pos < self.source.len() {
                    self.pos += self.rest().chars().next().unwrap().len_utf8();
                }
            }
        }
        Err(self.error(start, "unterminated template literal"))
    }

    fn operator(&mut self) {
        let operators: &[&str] = match self.language {
            SourceLanguage::Rust => &[
                "<<=", ">>=", "..=", "...", "::", "->", "=>", "..", "<<", ">>", "&&", "||", "==",
                "!=", "<=", ">=", "+=", "-=", "*=", "/=", "%=", "^=", "&=", "|=",
            ],
            SourceLanguage::Python => &[
                "**=", "//=", "<<=", ">>=", "...", "**", "//", "<<", ">>", ":=", "==", "!=", "<=",
                ">=", "+=", "-=", "*=", "/=", "%=", "@=", "&=", "|=", "^=", "->",
            ],
            SourceLanguage::JavaScript => &[
                ">>>=", "===", "!==", ">>>", "**=", "&&=", "||=", "??=", "<<=", ">>=", "...", "=>",
                "==", "!=", "<=", ">=", "++", "--", "&&", "||", "??", "**", "<<", ">>", "+=", "-=",
                "*=", "/=", "%=", "&=", "|=", "^=",
            ],
        };
        if self.language == SourceLanguage::JavaScript
            && self.rest().starts_with("?.")
            && !self
                .rest()
                .as_bytes()
                .get(2)
                .is_some_and(u8::is_ascii_digit)
        {
            self.pos += 2;
        } else if let Some(op) = operators.iter().find(|op| self.rest().starts_with(**op)) {
            self.pos += op.len();
        } else {
            self.pos += self.rest().chars().next().unwrap().len_utf8();
        }
    }

    fn js_context(&mut self, token: &str, literal: bool) {
        if self.language != SourceLanguage::JavaScript {
            return;
        }
        let (expression, statement) = if literal {
            (false, false)
        } else {
            match token {
                "(" => {
                    self.parens.push(self.control);
                    self.control = false;
                    (true, false)
                }
                ")" => {
                    let control = self.parens.pop().unwrap_or(false);
                    (control, control)
                }
                "{" => {
                    self.objects
                        .push(self.expression_start && !self.statement_start);
                    (true, true)
                }
                "}" => {
                    let object = self.objects.pop().unwrap_or(false);
                    (!object, !object)
                }
                "]" | "++" | "--" => (false, false),
                ";" | "=>" | "else" | "do" | "try" | "finally" => (true, true),
                "if" | "while" | "for" | "with" | "switch" | "catch" => {
                    self.control = true;
                    (true, false)
                }
                "return" | "throw" | "case" | "delete" | "void" | "typeof" | "new" | "in"
                | "instanceof" | "yield" | "await" => (true, false),
                _ => (
                    token.chars().next().is_some_and(|c| {
                        !identifier_start(c, self.language)
                            && !c.is_ascii_digit()
                            && !matches!(c, '\'' | '"' | '`')
                    }),
                    false,
                ),
            }
        };
        self.expression_start = expression;
        self.statement_start = statement;
    }
}

fn identifier_start(c: char, language: SourceLanguage) -> bool {
    c == '_' || c.is_alphabetic() || (language == SourceLanguage::JavaScript && c == '$')
}

#[cfg(test)]
mod tests {
    use super::{count_tokens, SourceLanguage::*, LEXER_VERSION};

    #[test]
    fn hand_counted_fixture_files_pin_v1() {
        // Each list spells out the counted tokens in file order.
        let files = [
            (
                Rust,
                "fixture.rs",
                "fn main() { let x = 1 + 2; } // done\n",
                vec![
                    "fn", "main", "(", ")", "{", "let", "x", "=", "1", "+", "2", ";", "}",
                ],
            ),
            (
                Python,
                "fixture.py",
                "def add(x, y):\n    # comment\n    return x + y\n",
                vec![
                    "def", "add", "(", "x", ",", "y", ")", ":", "return", "x", "+", "y",
                ],
            ),
            (
                JavaScript,
                "fixture.mjs",
                "export const square = (x) => x ** 2; /* done */",
                vec![
                    "export", "const", "square", "=", "(", "x", ")", "=>", "x", "**", "2", ";",
                ],
            ),
        ];
        assert_eq!(LEXER_VERSION, "jope.source-lex/v1");
        for (language, name, source, tokens) in files {
            assert_eq!(
                count_tokens(language, source),
                Ok(tokens.len()),
                "{name}: {tokens:?}"
            );
            println!(
                "{LEXER_VERSION} {name}: {} hand-counted tokens",
                tokens.len()
            );
        }
    }

    #[test]
    fn comments_and_whitespace_are_not_tokens() {
        for language in [Rust, Python, JavaScript] {
            assert_eq!(count_tokens(language, " \r\n\t"), Ok(0));
        }
        assert_eq!(
            count_tokens(Rust, "/* outer /* nested */ still outer */ x // end"),
            Ok(1)
        );
        assert_eq!(
            count_tokens(JavaScript, "/* not /* nested */ x // end"),
            Ok(1)
        );
        assert_eq!(count_tokens(Python, "# comment\nx \\\n + y # end"), Ok(3));
        assert_eq!(count_tokens(Python, "x // y"), Ok(3));
        assert_eq!(count_tokens(Rust, "x // y"), Ok(1));
        assert_eq!(count_tokens(Rust, "#[derive(Clone)] struct X;"), Ok(10));
    }

    #[test]
    fn rust_literals_lifetimes_and_raw_identifiers() {
        for source in [
            "r###\"/* not a comment */ \\\" #\"###",
            "br#\"// text\"#",
            "c\"text\"",
            "b'\\n'",
            "'\\u{3bb}'",
            "'λ'",
            "'static",
            "r#type",
        ] {
            assert_eq!(count_tokens(Rust, source), Ok(1), "{source}");
        }
        assert_eq!(
            count_tokens(Rust, "fn f<'a>(x: &'a str) -> &'a str { x }"),
            Ok(19)
        );
    }

    #[test]
    fn python_and_javascript_strings_preserve_comment_markers() {
        for source in [
            "r'// text'",
            "b'bytes'",
            "Fr\"{value!r} # text\"",
            "'''multi\n# literal\nline'''",
            "\"escaped \\\" # text\"",
        ] {
            assert_eq!(count_tokens(Python, source), Ok(1), "{source}");
        }
        for source in [
            "'// text'",
            "\"/* text */\"",
            "`value ${x + 1}`",
            "`outer ${`inner ${1 + 2}`} tail`",
            "`value ${ /[{}]/.test('}') }`",
        ] {
            assert_eq!(count_tokens(JavaScript, source), Ok(1), "{source}");
        }
    }

    #[test]
    fn numeric_literals_and_maximal_operators() {
        assert_eq!(count_tokens(Rust, "0x1e+2 1..=3 2.5e-4f64 1.field"), Ok(10));
        assert_eq!(
            count_tokens(Python, ".5 1.e+3 2j 0xff **= //= := ..."),
            Ok(8)
        );
        assert_eq!(
            count_tokens(JavaScript, ".5 1.e-3 12n x?.y x?.3:0 >>>= !=="),
            Ok(13)
        );
        assert_eq!(count_tokens(Rust, "a += b && c != d"), Ok(7));
    }

    #[test]
    fn javascript_regex_and_division_have_distinct_contexts() {
        assert_eq!(
            count_tokens(JavaScript, "const re = /a\\/[b/]/gi; x / 2 / y;"),
            Ok(11)
        );
        assert_eq!(count_tokens(JavaScript, "if (x) /a/.test(x);"), Ok(11));
        assert_eq!(count_tokens(JavaScript, "return /a/;"), Ok(3));
        assert_eq!(count_tokens(JavaScript, "(x + y) / 2;"), Ok(8));
        assert_eq!(count_tokens(JavaScript, ".5 / 2;"), Ok(4));
        assert_eq!(count_tokens(JavaScript, "const x = {} / 2;"), Ok(8));
        assert_eq!(count_tokens(JavaScript, "x / /a/.test(y);"), Ok(9));
    }

    #[test]
    fn template_interpolation_preserves_surrounding_object_context() {
        // const obj = { x : <template> , y : 2 } / 3 ; = 15 tokens.
        for source in [
            "const obj = {x: `hi ${1}`, y: 2} / 3;",
            "const obj = {x: `hi ${ {value: 1}.value }`, y: 2} / 3;",
            "const obj = {x: `hi ${`nested ${1}`}`, y: 2} / 3;",
            "const obj = {x: `hi ${ {} / 2 }`, y: 2} / 3;",
        ] {
            assert_eq!(count_tokens(JavaScript, source), Ok(15), "{source}");
        }
    }

    #[test]
    fn unicode_and_line_endings_are_stable() {
        for language in [Rust, Python, JavaScript] {
            assert_eq!(count_tokens(language, "λ = 2\r\n café += 1"), Ok(6));
            assert_eq!(count_tokens(language, "a\u{301} + β"), Ok(3));
        }
        assert_eq!(count_tokens(JavaScript, "$value + _other"), Ok(3));
    }

    #[test]
    fn unterminated_literals_and_comments_are_errors() {
        for (language, source) in [
            (Rust, "/* missing"),
            (Rust, "r#\"missing"),
            (Python, "'''missing"),
            (JavaScript, "`missing ${1"),
            (JavaScript, "/missing"),
        ] {
            assert!(count_tokens(language, source).is_err(), "{source}");
        }
        let error = count_tokens(Python, "x = 'missing").unwrap_err();
        assert_eq!(error.offset, 4);
        assert_eq!(
            error.to_string(),
            "unterminated string or character literal at byte 4"
        );
    }
}

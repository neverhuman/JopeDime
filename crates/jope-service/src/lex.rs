//! Pinned source-token counts for telemetry, independent of model tokenizers.
//!
//! V1 counts identifiers/keywords, literals, maximal operators and punctuation.
//! Whitespace, Python layout tokens and comments contribute zero. Rust lifetimes
//! and raw identifiers each count once. Prefixed, formatted and template strings
//! each count as one literal (interpolation is opaque). JavaScript regex literals
//! use the expression-start lexical goal, including after control parentheses;
//! this lightweight counter is not a JavaScript parser. Counts, including these
//! conventions, must get a new version whenever their rules change.

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

mod error;
mod trivia;
use trivia::identifier_start;

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

    fn literal(&mut self) -> Result<bool, LexError> {
        let start = self.pos;
        if self.language == SourceLanguage::Rust && self.rust_prefix()? {
            return Ok(true);
        }
        let mut formatted = false;
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
                    formatted = prefix.contains('f');
                    self.pos += prefix.len();
                    break;
                }
            }
        }
        let quote = self.rest().as_bytes()[0];
        if matches!(quote, b'\'' | b'"') {
            self.quoted(start, quote, formatted)?;
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

    fn quoted(&mut self, start: usize, quote: u8, formatted: bool) -> Result<(), LexError> {
        let triple = self.language == SourceLanguage::Python
            && self.rest().as_bytes().starts_with(&[quote; 3]);
        let width = if triple { 3 } else { 1 };
        self.pos += width;
        while self.pos < self.source.len() {
            if self.rest().as_bytes().starts_with(&[quote; 3][..width]) {
                self.pos += width;
                return Ok(());
            }
            if formatted && self.rest().starts_with("{{") {
                self.pos += 2;
                continue;
            }
            if formatted && self.rest().starts_with('{') {
                self.pos += 1;
                self.python_field(start)?;
                continue;
            }
            if self.rest().starts_with('\\') {
                self.pos += 1;
                if self.pos == self.source.len() {
                    break;
                }
                // A backslash in an f-string does not escape a replacement field.
                if formatted && self.rest().starts_with('{') {
                    continue;
                }
            }
            self.pos += self.rest().chars().next().unwrap().len_utf8();
        }
        Err(self.error(start, "unterminated string or character literal"))
    }

    fn python_field(&mut self, start: usize) -> Result<(), LexError> {
        if self.template_depth == 128 {
            return Err(self.error(start, "formatted-string nesting exceeds v1 limit"));
        }
        self.template_depth += 1;
        let mut brackets = 0_usize;
        loop {
            self.trivia()?;
            if brackets == 0 && self.rest().starts_with('}') {
                self.pos += 1;
                self.template_depth -= 1;
                return Ok(());
            }
            if brackets == 0 && self.rest().starts_with(':') {
                self.pos += 1;
                self.python_format(start)?;
                self.template_depth -= 1;
                return Ok(());
            }
            match self.next()? {
                Some("(" | "[" | "{") => brackets += 1,
                Some(")" | "]" | "}") if brackets > 0 => brackets -= 1,
                Some(_) => {}
                None => return Err(self.error(start, "unterminated formatted-string field")),
            }
        }
    }

    fn python_format(&mut self, start: usize) -> Result<(), LexError> {
        while let Some(c) = self.rest().chars().next() {
            self.pos += c.len_utf8();
            match c {
                '}' => return Ok(()),
                '{' => self.python_field(start)?,
                _ => {}
            }
        }
        Err(self.error(start, "unterminated formatted-string format specifier"))
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
                && (self.language != SourceLanguage::Rust
                    || (bytes.get(self.pos + 1) != Some(&b'.')
                        && !self.source[self.pos + 1..]
                            .chars()
                            .next()
                            .is_some_and(|c| identifier_start(c, self.language))))
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

#[cfg(test)]
#[path = "lex/tests.rs"]
mod tests;

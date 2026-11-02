use super::*;

impl Lexer {
    pub(crate) fn trivia(&mut self) -> Result<(), LexError> {
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

    pub(crate) fn block_comment(&mut self) -> Result<(), LexError> {
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

    pub(crate) fn identifier(&mut self) {
        while let Some(c) = self.rest().chars().next().filter(|&c| {
            identifier_start(c, self.language)
                || c.is_numeric()
                || ('\u{300}'..='\u{36f}').contains(&c)
        }) {
            self.pos += c.len_utf8();
        }
    }
}

pub(crate) fn identifier_start(c: char, language: SourceLanguage) -> bool {
    c == '_' || c.is_alphabetic() || (language == SourceLanguage::JavaScript && c == '$')
}

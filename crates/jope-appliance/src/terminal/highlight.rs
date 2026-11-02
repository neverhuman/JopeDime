//! Small built-in lexer for the three terminal code-preview languages.

use super::ThemeToken;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HighlightLanguage {
    Rust,
    Python,
    JavaScript,
}

/// Compatibility name for callers using the original lexer language type.
pub type Language = HighlightLanguage;

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
pub fn highlight(source: &str, language: HighlightLanguage) -> Vec<Span<'_>> {
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

#[path = "highlight/scanner.rs"]
mod scanner;
pub(super) use scanner::*;

#[cfg(test)]
mod tests {
    use super::{highlight, HighlightLanguage, Span};
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
            highlight(source, HighlightLanguage::Rust),
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
            highlight(source, HighlightLanguage::Python),
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
            highlight(source, HighlightLanguage::JavaScript),
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
            ("let text = \"open", HighlightLanguage::Rust),
            ("text = 'open", HighlightLanguage::Python),
            ("const text = `open", HighlightLanguage::JavaScript),
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
        let spans = highlight(source, HighlightLanguage::Rust);
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

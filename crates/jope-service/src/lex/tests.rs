use super::*;
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
fn python_formatted_fields_keep_same_delimiter_and_nested_literals_opaque() {
    for source in [
        r#"f"{data["key"]}""#,
        r#"f'{data['key']}'"#,
        r#"f"{f"{value}"}""#,
        r#"Fr"{data["key"]}""#,
        r#"f"{data["key"]!r:>{width}}""#,
        r#"f"{x:'>10}""#,
        r#"f"{x:{width}.{precision}f}""#,
        r#"f"{{literal}} {data["key"]}""#,
        r#"f"\{data["key"]}""#,
        "f\"\"\"{data[\"\"\"key\"\"\"]}\"\"\"",
    ] {
        assert_eq!(count_tokens(Python, source), Ok(1), "{source}");
    }
    assert!(count_tokens(Python, r#"f"{data["key"]""#).is_err());
}

#[test]
fn python_trailing_decimal_point_precedes_attribute_dot() {
    // NUMBER 1. / OP . / NAME real; Rust retains its range operator.
    assert_eq!(count_tokens(Python, "1..real"), Ok(3));
    assert_eq!(count_tokens(Python, "1..real + 2..imag"), Ok(7));
    assert_eq!(count_tokens(Python, "1. .real"), Ok(3));
    assert_eq!(count_tokens(Python, "..."), Ok(1));
    assert_eq!(count_tokens(Rust, "1..real"), Ok(3));
    assert_eq!(count_tokens(JavaScript, "1..toString()"), Ok(5));
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

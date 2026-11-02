#[test]
fn python_strip() {
    assert_eq!(strip("\u{1c} x \u{a0}"), "x");
    assert_eq!(strip("\u{1f}45"), "45");
}

/// The strict grammar reads what Python's `int()` read with ASCII whitespace, sign and digits.
#[test]
fn strict_int_reads_ascii_decimal() {
    assert_eq!(int("01"), Ok(1));
    assert_eq!(int("45"), Ok(45));
    assert_eq!(int(" \t\n\r\u{b}\u{c}+45 \t\n\r\u{b}\u{c}"), Ok(45));
    assert_eq!(int("-45"), Ok(-45));
    assert_eq!(int("-0"), Ok(0));
    assert_eq!(int(&i64::MAX.to_string()), Ok(i64::MAX));
    assert_eq!(int(&i64::MIN.to_string()), Ok(i64::MIN));
    let decimal = decimal(" -0045 ").unwrap();
    assert_eq!((decimal.negative, decimal.digits), (true, "0045"));
    assert_eq!(decimal.magnitude(), "45");
    assert_eq!(super::decimal("000").unwrap().magnitude(), "0");
}

/// Round four of the review of PR #117: every input Python's `int()` refused is refused, and
/// so are the forms it read that the strict grammar does not (stricter): U+001C..U+001F
/// (Python refused them though `str.strip()` removes them), Unicode whitespace at or above
/// U+0080, underscores, other Unicode digits and values beyond `i64`.
#[test]
fn strict_int_refuses_what_python_refused_and_more() {
    for refused_by_python in [
        "",
        " ",
        "+",
        "-",
        "+-1",
        "1 1",
        "4.0",
        "0x10",
        "1e3",
        "45x",
        "\u{1c}45",
        "\u{1d}45",
        "\u{1e}45",
        "\u{1f}45",
        "45\u{1f}",
        "\u{200b}45",
        "1__0",
        "_1",
        "1_",
    ] {
        assert_eq!(
            int(refused_by_python),
            Err(IntError::Invalid),
            "{refused_by_python:?}"
        );
    }
    for read_by_python in [
        "\u{a0}45",
        "\u{85}45",
        "\u{2003}45",
        "\u{3000}45",
        "45\u{2028}",
        "4_5",
        "+1_0",
        "\u{664}\u{665}",
        "\u{ff14}\u{ff15}",
        "\u{1d7d2}",
        "9223372036854775808",
        "-9223372036854775809",
    ] {
        assert_eq!(
            int(read_by_python),
            Err(IntError::Invalid),
            "{read_by_python:?}"
        );
    }
}

#[test]
fn fsum_matches_cpython() {
    assert_eq!(fsum([0.1; 10]), Some(1.0));
    assert_eq!(fsum([1e-16, 1.0, 1e16]), Some(10000000000000002.0));
    assert_eq!(fsum([1e308, 1e308, -1e308]), None);
    assert_eq!(fsum([f64::INFINITY, 1.0]), Some(f64::INFINITY));
    assert_eq!(fsum([f64::INFINITY, f64::NEG_INFINITY]), None);
    assert_eq!(fsum([]), Some(0.0));
    assert_eq!(fsum([1.0, 1e100, 1.0, -1e100]), Some(2.0));
    // Plain left-to-right addition gives 0.9999999999999999 here.
    assert_ne!([0.1_f64; 10].iter().sum::<f64>(), 1.0);
}

#[test]
fn python_round_and_float() {
    assert_eq!(round(2.675, 2), 2.67);
    assert_eq!(round(0.25, 1), 0.2);
    assert_eq!((round(0.125, 2), round(0.375, 2)), (0.12, 0.38));
    assert_eq!(round(-0.04, 1).to_bits(), (-0.0_f64).to_bits());
}

/// `float()` through the strict grammar: the ASCII forms Python read, the same values.
#[test]
fn strict_float_reads_ascii_decimal_and_exponent_forms() {
    assert_eq!(float("21.511"), Some(21.511));
    assert_eq!(float(" \t\n\r\u{b}\u{c}21.511 \u{c}"), Some(21.511));
    assert_eq!(float("+4.622"), Some(4.622));
    assert_eq!(float("-4.622"), Some(-4.622));
    assert_eq!(float("1."), Some(1.0));
    assert_eq!(float(".5e-1"), Some(0.05));
    assert_eq!(float("4622E-3"), Some(4.622));
    assert_eq!(float("0.1e+01"), Some(1.0));
    assert_eq!(float("0021.5"), Some(21.5));
    assert_eq!(float("1e999"), Some(f64::INFINITY));
    assert_eq!(float("-1e999"), Some(f64::NEG_INFINITY));
    assert_eq!(float("1e-999"), Some(0.0));
    assert_eq!(float("2.675"), Some(2.675));
}

/// Round four of the review of PR #117: every input Python's `float()` refused is refused, and
/// so are the forms it read that the strict grammar does not (stricter): `nan`, `inf` and
/// `infinity`, underscores, other Unicode digits and Unicode whitespace at or above U+0080.
#[test]
fn strict_float_refuses_what_python_refused_and_more() {
    for refused_by_python in [
        "",
        " ",
        ".",
        "e5",
        "1e",
        "1e+",
        "1.5e",
        "0x10",
        "0x1p2",
        "1,5",
        "abc",
        "1 1",
        "\u{1c}21.5",
        "\u{1f}21.511",
        "21.511\u{1f}",
        "\u{200b}1",
        "1__0",
        "_1",
        "1_",
        "1.5_",
        "--1",
        "+-1",
        "infinit",
        "nana",
    ] {
        assert_eq!(float(refused_by_python), None, "{refused_by_python:?}");
    }
    for read_by_python in [
        "nan",
        "NaN",
        "-nan",
        "inf",
        "+inf",
        "-Infinity",
        "INFINITY",
        " 1_000.5 ",
        "1_0.5",
        "\u{a0}21.5",
        "\u{2003}21.5",
        "\u{85}21.5",
        "\u{664}.\u{666}",
        "\u{ff14}.5",
    ] {
        assert_eq!(float(read_by_python), None, "{read_by_python:?}");
    }
}

/// Python 3.12's `\s`, `\w` and `\d` (Unicode 15.0), not the `regex` crate's.
#[test]
fn regex_classes_are_pythons() {
    let space = regex(r"^\s$");
    for c in ['\u{1c}', '\u{1f}', '\u{a0}', '\u{2028}', '\u{3000}', '\t'] {
        assert!(space.is_match(&c.to_string()), "{c:?} is a Python space");
    }
    assert!(!space.is_match("\u{180e}") && !space.is_match("\u{200b}"));
    let word = regex(r"^\w$");
    for c in [
        '_', 'a', '\u{e9}', '\u{b2}', '\u{2167}', '\u{4e00}', '\u{661}',
    ] {
        assert!(
            word.is_match(&c.to_string()),
            "{c:?} is a Python word character"
        );
    }
    // Marks, connector punctuation, joiners and `Other_Alphabetic` symbols are not; nor is a
    // letter first assigned after Unicode 15.0 (U+2EBF0, CJK Extension I, 15.1).
    for c in ['\u{301}', '\u{203f}', '\u{200d}', '\u{24b6}', '\u{2ebf0}'] {
        assert!(
            !word.is_match(&c.to_string()),
            "{c:?} is not a Python word character"
        );
    }
    let class = regex(r"^[\w-]+\s*:$");
    assert!(class.is_match("a-b\u{1f}:") && !class.is_match("a\u{301}:"));
    let digit = regex(r"^\d+$");
    assert!(digit.is_match("0\u{665}\u{1d7ce}") && !digit.is_match("\u{b2}"));
    // What `\d+` matches is a match only; reading it as a number is the strict grammar's.
    assert!(is_decimal_run("0\u{665}\u{1d7d9}") && is_decimal_run("007"));
    assert!(!is_decimal_run("\u{b2}") && !is_decimal_run("") && !is_decimal_run("5\n"));
}

/// `int()` of a `\d+` capture through the strict grammar: ASCII digits of any value up to 4300
/// digits (leading zeros count, as they did for CPython 3.12), 4301 refused in CPython's words;
/// a run holding any other Unicode digit (which Python read) refused, naming it.
#[test]
fn int_conversion_limit_is_pythons() {
    let limit = |n: usize| {
        format!("Exceeds the limit (4300 digits) for integer string conversion: value has {n} digits; use sys.set_int_max_str_digits() to increase the limit")
    };
    assert_eq!(check_int_digits(4300), Ok(()));
    assert_eq!(check_int_digits(4301), Err(limit(4301)));
    let run = |n: usize, zero: char, one: char| format!("{}{one}", zero.to_string().repeat(n - 1));
    assert_eq!(decimal_int(&run(4300, '0', '1')), Ok("1".to_owned()));
    assert_eq!(decimal_int(&run(4301, '0', '1')), Err(limit(4301)));
    assert_eq!(decimal_int(&"9".repeat(4300)), Ok("9".repeat(4300)));
    assert_eq!(decimal_int("000"), Ok("0".to_owned()));
    let not_ascii = |text: &str| {
        Err(format!(
            "{} is not ASCII decimal digits, the only digits the port reads as a number",
            pyjson::str_repr(text)
        ))
    };
    for read_by_python in [
        "\u{665}".to_owned(),
        "\u{ff10}\u{ff15}".to_owned(),
        "\u{1d7d3}".to_owned(),
        run(4300, '\u{660}', '\u{661}'),
        run(4302, '\u{660}', '\u{661}'),
    ] {
        assert_eq!(decimal_int(&read_by_python), not_ascii(&read_by_python));
    }
    for refused in ["\u{b2}", "", " 5", "+5", "-5", "5 "] {
        assert_eq!(decimal_int(refused), not_ascii(refused), "{refused:?}");
    }
    // `int` of text: beyond `i64` it refuses as invalid, past the limit in CPython's words.
    assert_eq!(int(&"1".repeat(20)), Err(IntError::Invalid));
    assert_eq!(int(&"1".repeat(4301)), Err(IntError::Limit(limit(4301))));
}

/// Python 3.12's `int()` of text: the limit is checked on a well-formed literal before it is
/// converted, so 4301 digits refuse even when they are leading zeros of a small value; the
/// sign and surrounding whitespace are not digits. The strict grammar refuses underscores
/// (which Python read, not counting them as digits) whatever their number, and the reviewer's
/// U+001F around the digits (which Python refused).
#[test]
fn int_of_text_counts_digits_as_python_did() {
    let limit = |n: usize| {
        IntError::Limit(format!("Exceeds the limit (4300 digits) for integer string conversion: value has {n} digits; use sys.set_int_max_str_digits() to increase the limit"))
    };
    let zeros = |n: usize| "0".repeat(n);
    let unders = |n: usize| "0_".repeat(n);
    // 4300 digits convert, whatever ASCII whitespace and sign surround them.
    for text in [
        format!("{}45", zeros(4298)),
        format!("+{}45", zeros(4298)),
        format!(" \t{}45\n", zeros(4298)),
    ] {
        assert_eq!(int(&text), Ok(45), "{} characters", text.len());
    }
    assert_eq!(int(&format!("-{}45", zeros(4298))), Ok(-45));
    assert_eq!(int(&zeros(4300)), Ok(0));
    // 4301 refuse, leading zeros counted.
    for text in [
        format!("{}45", zeros(4299)),
        format!("+{}45", zeros(4299)),
        format!("-{}45", zeros(4299)),
        format!(" {}45 ", zeros(4299)),
        zeros(4301),
    ] {
        assert_eq!(int(&text), Err(limit(4301)), "{} characters", text.len());
    }
    assert_eq!(int(&zeros(5000)), Err(limit(5000)));
    // Outside the grammar, however many digits: underscores (Python read 4300 of them as 45
    // and refused 4301 at the limit), U+00A0 and U+001F around the digits.
    for text in [
        format!("{}45", unders(4298)),
        format!("{}45", unders(4299)),
        format!("\u{a0}+{}45\u{1f}", unders(4298)),
        format!("\u{1f}{}45", zeros(4298)),
    ] {
        assert_eq!(
            int(&text),
            Err(IntError::Invalid),
            "{} characters",
            text.len()
        );
    }
    // A malformed literal is refused as such, however long: CPython validated it first.
    assert_eq!(int(&format!("{}45_", zeros(4299))), Err(IntError::Invalid));
    assert_eq!(int(&format!("{}4x", zeros(4299))), Err(IntError::Invalid));
    // A non-ASCII digit stays refused (Python mapped it to ASCII and read 5; stricter).
    assert_eq!(
        int(&format!("{}\u{665}", zeros(4299))),
        Err(IntError::Invalid)
    );
}

#[test]
fn subprocess_text_and_errors_read_as_python_printed_them() {
    assert_eq!(text_output(b"a\r\nb\rc\n").unwrap(), "a\nb\nc\n");
    assert_eq!(
        text_output(b"ok\xff").unwrap_err(),
        "'utf-8' codec can't decode byte 0xff in position 2: invalid start byte"
    );
    assert_eq!(
        text_output(b"\xe2\x82x").unwrap_err(),
        "'utf-8' codec can't decode bytes in position 0-1: invalid continuation byte"
    );
    assert_eq!(
        text_output(b"x\xed\xa0\x80").unwrap_err(),
        "'utf-8' codec can't decode byte 0xed in position 1: invalid continuation byte"
    );
    assert_eq!(
        text_output(b"x\xe2\x82").unwrap_err(),
        "'utf-8' codec can't decode bytes in position 1-2: unexpected end of data"
    );
    let missing = std::fs::metadata("/nonexistent-q/a").unwrap_err();
    assert_eq!(
        os_error(&missing, "/nonexistent-q/a"),
        "[Errno 2] No such file or directory: '/nonexistent-q/a'"
    );
    let argv = str_list_repr(&["du", "-s"]);
    assert_eq!(
        called_process_error(&argv, 1),
        "Command '['du', '-s']' returned non-zero exit status 1."
    );
    assert_eq!(
        called_process_error(&argv, -9),
        "Command '['du', '-s']' died with <Signals.SIGKILL: 9>."
    );
    assert_eq!(
        called_process_error(&argv, -40),
        "Command '['du', '-s']' died with unknown signal 40."
    );
    assert!(!path_exists(Path::new("/nonexistent-q/a")).unwrap());
    assert!(!path_exists(Path::new("/nonexistent-q/a\0b")).unwrap());
    const ENAMETOOLONG: i32 = if cfg!(target_os = "macos") { 63 } else { 36 };
    let long = format!("/{}", "x".repeat(300));
    assert_eq!(
        path_exists(Path::new(&long)).unwrap_err().raw_os_error(),
        Some(ENAMETOOLONG)
    );
}

/// Review round six of PR #117: a name that is not UTF-8 is rendered as Python rendered its
/// surrogate-escaped `str` (never U+FFFD), and printing it through a strict UTF-8 stream fails
/// with CPython's `UnicodeEncodeError` text: the first run of surrogates, positions in
/// characters (`é` is one).
#[test]
fn names_that_are_not_utf8_render_as_python_rendered_them() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    let name = |bytes: &[u8]| OsStr::from_bytes(bytes).to_owned();
    let ff = name(b"/r/\xff/demo");
    assert_eq!(os_repr(&ff), r"'/r/\udcff/demo'");
    assert_eq!(os_display(&ff), r"/r/\udcff/demo");
    let character = |at: usize, c: &str| {
        format!(
            "'utf-8' codec can't encode character '\\{c}' in position {at}: surrogates not allowed"
        )
    };
    let characters = |from: usize, to: usize| {
        format!(
            "'utf-8' codec can't encode characters in position {from}-{to}: surrogates not allowed"
        )
    };
    assert_eq!(utf8_encode_error(&ff), Some(character(3, "udcff")));
    // A run of undecodable bytes is one range; an incomplete sequence is one surrogate per byte.
    assert_eq!(
        utf8_encode_error(&name(b"/r/\xff\xfe/x")),
        Some(characters(3, 4))
    );
    assert_eq!(
        utf8_encode_error(&name(b"/r/\xe2\x82x/demo")),
        Some(characters(3, 4))
    );
    assert_eq!(
        fsdecode(&name(b"\xe2\x82x")),
        [Err(0xdce2), Err(0xdc82), Ok('x')]
    );
    // Positions count characters, and only the first run is reported.
    let mut accented = "/r/\u{e9}".as_bytes().to_vec();
    accented.extend_from_slice(b"\xff/demo");
    assert_eq!(
        utf8_encode_error(&name(&accented)),
        Some(character(4, "udcff"))
    );
    assert_eq!(
        utf8_encode_error(&name(b"/a\xffb\xfe")),
        Some(character(2, "udcff"))
    );
    assert_eq!(os_display(&name(b"/a\xffb\xfe")), r"/a\udcffb\udcfe");
    // A UTF-8 name, ASCII or not, is printed as it is.
    assert_eq!(utf8_encode_error(OsStr::new("/r/\u{e9}/demo")), None);
    assert_eq!(os_display(OsStr::new("/r/\u{e9}")), "/r/\u{e9}");
    assert_eq!(os_repr(OsStr::new("/r/it's")), "\"/r/it's\"");
}

#[test]
fn python_splitlines() {
    assert_eq!(
        splitlines("a\nb\r\nc\rd\u{2028}e\n"),
        ["a", "b", "c", "d", "e"]
    );
    assert_eq!(splitlines("a\n\n"), ["a", ""]);
    assert!(splitlines("").is_empty());
}

#[test]
fn python_membership() {
    let list = pyjson::parse_str(r#"["a", 1]"#).unwrap();
    assert_eq!(contains(&list, &Value::Float(1.0)), Some(true));
    // Python `==`, not JSON-text equality: `True == 1 == 1.0`, `[1] == [1.0]`.
    assert_eq!(contains(&list, &Value::Bool(true)), Some(true));
    let nested = pyjson::parse_str("[[1]]").unwrap();
    assert_eq!(
        contains(&nested, &pyjson::parse_str("[1.0]").unwrap()),
        Some(true)
    );
    // Each refusal names the value of the wrong type.
    let dict = pyjson::parse_str(r#"{"a": 1}"#).unwrap();
    assert_eq!(
        contains_or_raise(&dict, &list),
        Err("TypeError: unhashable type: 'list'".to_owned())
    );
    assert_eq!(
        contains_or_raise(&Value::from("abc"), &Value::Int(1)),
        Err("TypeError: 'in <string>' requires string as left operand, not int".to_owned())
    );
    assert_eq!(
        contains_or_raise(&Value::Null, &Value::from("a")),
        Err("TypeError: argument of type 'NoneType' is not iterable".to_owned())
    );
    assert_eq!(contains_or_raise(&dict, &Value::Int(1)), Ok(false));
    assert_eq!(
        contains(&Value::from("abc"), &Value::from("bc")),
        Some(true)
    );
    assert_eq!(contains(&Value::Int(1), &Value::from("x")), None);
    assert_eq!(Key::of(&Value::Bool(true)), Key::of(&Value::Float(1.0)));
    // Every JSON NaN is one object: found by identity, though `NaN == NaN` is false.
    let nans = pyjson::parse_str("[NaN]").unwrap();
    assert_eq!(contains(&nans, &Value::Float(f64::NAN)), Some(true));
    assert_ne!(Value::Float(f64::NAN), Value::Float(f64::NAN));
    assert_eq!(nans, pyjson::parse_str("[NaN]").unwrap());
    // Integers beyond 64 bits (opted in) compare and hash exactly, with floats too.
    let big = pyjson::parse_str_big("[18446744073709551616, 18446744073709551617]").unwrap();
    let items = big.as_list().unwrap();
    assert_ne!(items[0], items[1]);
    assert_eq!(items[0], Value::Float(18446744073709551616.0));
    assert_eq!(
        Key::of(&items[0]),
        Key::of(&Value::Float(18446744073709551616.0))
    );
    assert_eq!(
        contains(&big, &Value::Float(18446744073709551616.0)),
        Some(true)
    );
    assert_eq!(type_name(&items[1]), "int");
    assert!(pyjson::parse_str("18446744073709551616").is_err());
    assert_eq!(tuple_repr(&[Value::from("a"), Value::Null]), "('a', None)");
}

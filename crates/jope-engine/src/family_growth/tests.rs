use super::*;

use super::*;

/// RFC 4231 test cases 1, 2 and 6 (short key, short ASCII key, key longer than a block).
#[test]
fn hmac_matches_rfc_4231() {
    assert_eq!(
        hmac_sha256_hex(&[0x0b; 20], b"Hi There"),
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
    );
    assert_eq!(
        hmac_sha256_hex(b"Jefe", b"what do ya want for nothing?"),
        "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
    );
    assert_eq!(
        hmac_sha256_hex(
            &[0xaa; 131],
            b"Test Using Larger Than Block-Size Key - Hash Key First"
        ),
        "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
    );
}

#[test]
fn compare_digest_needs_equal_length_and_bytes() {
    assert!(compare_digest("abc", "abc"));
    assert!(!compare_digest("abc", "abd"));
    assert!(!compare_digest("abc", "ab"));
    assert!(!compare_digest("", "a"));
    assert!(compare_digest("", ""));
}

#[test]
fn python_equality_truthiness_and_membership() {
    assert!(py_eq(&json!(1), &json!(1.0)));
    assert!(py_eq(&json!(true), &json!(1)));
    assert!(!py_eq(&json!("1"), &json!(1)));
    assert!(py_eq(&json!({"a": [1, "x"]}), &json!({"a": [1.0, "x"]})));
    assert!(!py_eq(&json!({"a": 1}), &json!({"a": 1, "b": 2})));
    for falsy in [
        json!(null),
        json!(false),
        json!(0),
        json!(0.0),
        json!(""),
        json!([]),
        json!({}),
    ] {
        assert!(!truthy(&falsy), "{falsy}");
    }
    assert!(contains(&json!(["a", "b"]), &json!("b")).unwrap());
    assert!(contains(&json!("reviewer-a"), &json!("view")).unwrap());
    assert!(contains(&json!({"k": 1}), &json!("k")).unwrap());
    assert!(contains(&json!(3), &json!("k")).is_err());
    assert_eq!(iterate(&json!("ab")).unwrap(), vec![json!("a"), json!("b")]);
    assert_eq!(py_strip("\u{1c} x \u{a0}"), "x");
}

/// Python 3.12's `json.loads` refused an integer literal of more than 4300 digits, of either
/// sign, even in a field nothing reads, and so does the native reader, in its words. Below
/// that Python read every integer exactly; `serde_json` reads one outside
/// `i64::MIN..=u64::MAX` as the nearest double, so the native reader refuses it rather than
/// round it (review round eight; stricter): `2^64` and `2^64 + 1` would otherwise be one
/// value. The 64-bit edges, floats, and digits inside a string are read as before.
#[test]
fn integers_are_exact_or_refused() {
    let receipt =
        |field: &str| format!(r#"{{"schema": "{ASSIGNMENT_SCHEMA}", "unused": {field}}}"#);
    let refusal = |field: &str| match parse(receipt(field).as_bytes(), "growth assignment") {
        Err(GrowthError::Malformed(message)) => message,
        other => panic!("{field}: {other:?}"),
    };
    for sign in ["", "-"] {
        let message = refusal(&format!("{sign}{}", "9".repeat(4301)));
        assert!(
                message.ends_with("Exceeds the limit (4300 digits) for integer string conversion: value has 4301 digits; use sys.set_int_max_str_digits() to increase the limit"),
                "{message}"
            );
    }
    for wide in [
        "12345678901234567890123",
        "18446744073709551616",
        "18446744073709551617",
        "-9223372036854775809",
        "[1, {\"k\": -18446744073709551616}]",
    ] {
        assert!(refusal(wide).contains("is outside 64 bits"), "{wide}");
    }
    let exact = |field: &str| {
        parse(receipt(field).as_bytes(), "growth assignment").unwrap()["unused"].clone()
    };
    assert_eq!(exact("18446744073709551615"), json!(u64::MAX));
    assert_eq!(exact("-9223372036854775808"), json!(i64::MIN));
    assert_eq!(exact("1.8446744073709552e19"), json!(1.8446744073709552e19));
    assert_eq!(exact("1e25"), json!(1e25));
    assert_eq!(
        exact(r#""18446744073709551617 \" 99999999999999999999""#),
        json!("18446744073709551617 \" 99999999999999999999")
    );
    assert!(!py_eq(
        &exact("18446744073709551615"),
        &exact("1.8446744073709552e19")
    ));
}

/// Same seed and bucket rule as `families::stable_component_assignment`.
#[test]
fn component_bucket_is_the_registry_rule() {
    for id in ["canonical-cc", "alias-string", "development-0"] {
        let (digest, bucket, partition) = stable_assignment_v2(id);
        let (d2, b2, p2) = crate::families::stable_component_assignment(id);
        assert_eq!((digest, bucket, partition.to_owned()), (d2, b2, p2));
    }
}

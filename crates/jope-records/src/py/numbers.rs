/// Python `str.isspace` for one character (Unicode White_Space plus U+001C..U+001F).
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python `str.strip()`.
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// CPython 3.12's default `sys.get_int_max_str_digits()`. `int()` of a decimal string, and so
/// `json.loads` and `tomllib` of an integer literal, raised `ValueError` for more digits than this
/// (the sign, surrounding whitespace and underscores are not digits; leading zeros are).
pub const INT_MAX_STR_DIGITS: usize = 4300;

/// The limit on decimal integer conversion: `digits` digits are converted, more than
/// [`INT_MAX_STR_DIGITS`] refuse with CPython's `ValueError` message.
pub fn check_int_digits(digits: usize) -> std::result::Result<(), String> {
    if digits > INT_MAX_STR_DIGITS {
        return Err(format!(
            "Exceeds the limit ({INT_MAX_STR_DIGITS} digits) for integer string conversion: value has {digits} digits; use sys.set_int_max_str_digits() to increase the limit"
        ));
    }
    Ok(())
}

/// Why [`decimal`] or [`int`] refused its text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntError {
    /// Not a decimal integer of the strict grammar ([`decimal`]), or for [`int`] a value beyond
    /// `i64`. Each caller words this refusal.
    Invalid,
    /// A decimal integer of more than [`INT_MAX_STR_DIGITS`] digits: CPython's `ValueError`
    /// message ([`check_int_digits`]).
    Limit(String),
}

/// The whitespace the strict number grammar allows around a number: ASCII space, `\t`, `\n`,
/// `\r`, `\x0b` and `\x0c`, CPython's `Py_ISSPACE`. `int()` and `float()` of a `str` also stripped
/// the Unicode whitespace at or above U+0080 (U+0085, U+00A0, U+2003, ...), which the grammar
/// refuses; they refused U+001C..U+001F, which `str.strip()` removes, and so does the grammar.
pub fn is_number_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\u{b}' | '\u{c}')
}

/// A decimal integer read by the strict grammar ([`decimal`]): its sign and its ASCII digits as
/// written (leading zeros kept).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decimal<'a> {
    pub negative: bool,
    pub digits: &'a str,
}

impl Decimal<'_> {
    /// The digits without leading zeros (`0` for zero): `str(abs(int(text)))`, of any length.
    pub fn magnitude(&self) -> &str {
        let trimmed = self.digits.trim_start_matches('0');
        if trimmed.is_empty() {
            "0"
        } else {
            trimmed
        }
    }

    /// The value, when it fits `i64`.
    pub fn to_i64(&self) -> Option<i64> {
        let magnitude: i128 = self.magnitude().parse().ok()?;
        i64::try_from(if self.negative { -magnitude } else { magnitude }).ok()
    }
}

/// The one grammar every port reads a decimal integer from text with. It accepts a subset of what
/// Python's `int(text)` accepted, never more: optional surrounding whitespace of
/// [`is_number_space`] only, an optional ASCII sign (`+` or `-`), then one or more ASCII digits,
/// at most [`INT_MAX_STR_DIGITS`] of them (leading zeros count, as they did for CPython 3.12,
/// which refused more with the message [`check_int_digits`] words). Python also read single
/// underscores between digits (PEP 515), every other Unicode decimal digit (mapped to ASCII) and
/// Unicode whitespace at or above U+0080; the grammar refuses all three, and each port documents
/// that as stricter than the Python it replaced. The text is validated before its digits are
/// counted, as CPython validated a literal before it applied the limit.
pub fn decimal(text: &str) -> std::result::Result<Decimal<'_>, IntError> {
    let body = text.trim_matches(is_number_space);
    let (negative, digits) = match body.as_bytes().first() {
        Some(b'-') => (true, &body[1..]),
        Some(b'+') => (false, &body[1..]),
        _ => (false, body),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(IntError::Invalid);
    }
    check_int_digits(digits.len()).map_err(IntError::Limit)?;
    Ok(Decimal { negative, digits })
}

/// `int(text)` through the strict grammar ([`decimal`]): the value, or [`IntError::Invalid`] for
/// text outside the grammar and for a value beyond `i64` (which Python converted; stricter).
pub fn int(text: &str) -> std::result::Result<i64, IntError> {
    decimal(text)?.to_i64().ok_or(IntError::Invalid)
}

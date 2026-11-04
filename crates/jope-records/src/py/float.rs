use super::*;

/// `math.fsum`: the correctly rounded sum of finite values, ported from CPython's
/// `mathmodule.c` (Shewchuk partials plus its half-even correction across partials). Returns
/// `None` where CPython raises (intermediate overflow, or `-inf + inf`).
pub fn fsum(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut partials: Vec<f64> = Vec::new();
    let mut special_sum = 0.0_f64;
    let mut inf_sum = 0.0_f64;
    for value in values {
        let mut x = value;
        let xsave = x;
        let mut kept = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let yr = hi - x;
            let lo = y - yr;
            if lo != 0.0 {
                partials[kept] = lo;
                kept += 1;
            }
            x = hi;
        }
        partials.truncate(kept);
        if x != 0.0 {
            if !x.is_finite() {
                if xsave.is_finite() {
                    return None; // intermediate overflow
                }
                if xsave.is_infinite() {
                    inf_sum += xsave;
                }
                special_sum += xsave;
                partials.clear();
            } else {
                partials.push(x);
            }
        }
    }
    if special_sum != 0.0 {
        return if inf_sum.is_nan() {
            None
        } else {
            Some(special_sum)
        };
    }
    let mut n = partials.len();
    let mut hi = 0.0_f64;
    if n > 0 {
        n -= 1;
        hi = partials[n];
        let mut lo = 0.0_f64;
        while n > 0 {
            let x = hi;
            n -= 1;
            let y = partials[n];
            hi = x + y;
            let yr = hi - x;
            lo = y - yr;
            if lo != 0.0 {
                break;
            }
        }
        if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
            let y = lo * 2.0;
            let x = hi + y;
            let yr = x - hi;
            if y == yr {
                hi = x;
            }
        }
    }
    Some(hi)
}

/// Python `round(value, ndigits)` for a float: correctly rounded to `ndigits` decimals (ties to
/// even on the exact binary value), returned as the nearest double.
pub fn round(value: f64, ndigits: usize) -> f64 {
    if !value.is_finite() {
        return value;
    }
    format!("{value:.ndigits$}").parse().unwrap_or(value)
}

/// `float(text)` through the strict number grammar: optional surrounding whitespace of
/// [`is_number_space`] only, an optional ASCII sign, then `digits [. [digits]]` or `. digits` in
/// ASCII digits, then optionally `e` or `E`, a sign and digits; the value is correctly rounded (an
/// exponent beyond the range of a double gives an infinity or zero, as Python's did). `None` for
/// anything else. Python also read `nan`, `inf` and `infinity` in any case, underscores between
/// digits, other Unicode decimal digits and Unicode whitespace at or above U+0080; no committed
/// input the ports read with `float()` writes any of them, and the grammar refuses them all
/// (stricter).
pub fn float(text: &str) -> Option<f64> {
    let body = text.trim_matches(is_number_space);
    let unsigned = body.strip_prefix(['+', '-']).unwrap_or(body);
    let s = unsigned.as_bytes();
    let digits = |mut i: usize| {
        while s.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        i
    };
    let mut i = digits(0);
    let mut any = i > 0;
    if s.get(i) == Some(&b'.') {
        let after = digits(i + 1);
        any |= after > i + 1;
        i = after;
    }
    if !any {
        return None;
    }
    if matches!(s.get(i), Some(b'e' | b'E')) {
        let mut j = i + 1;
        if matches!(s.get(j), Some(b'+' | b'-')) {
            j += 1;
        }
        let end = digits(j);
        if end == j {
            return None;
        }
        i = end;
    }
    if i != s.len() {
        return None;
    }
    body.parse().ok()
}

/// Python `str.splitlines()`: lines without their terminators.
pub fn splitlines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let end = match c {
            '\r' => {
                if let Some(&(_, '\n')) = chars.peek() {
                    chars.next();
                    i + 2
                } else {
                    i + 1
                }
            }
            '\n' | '\u{b}' | '\u{c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}'
            | '\u{2029}' => i + c.len_utf8(),
            _ => continue,
        };
        lines.push(&text[start..i]);
        start = end;
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

use super::*;

// ---------------------------------------------------------------------------------------------
// Python `json.dumps`, `repr`/`str` and `csv` spelling.

#[derive(Clone, Copy)]
pub(super) struct Style {
    /// `ensure_ascii`.
    ascii: bool,
    /// `indent`; `None` writes one line.
    indent: Option<usize>,
    /// On one line, the default `(', ', ': ')` separators rather than `(',', ':')`.
    spaced: bool,
}

impl Style {
    /// `json.dumps(value, indent=2, allow_nan=False)`.
    pub(super) const REPORT: Style = Style {
        ascii: true,
        indent: Some(2),
        spaced: false,
    };
    /// `json.dumps(value, ensure_ascii=False, separators=(',', ':'))`.
    pub(super) const IDENTITY: Style = Style {
        ascii: false,
        indent: None,
        spaced: false,
    };
    /// `json.dumps(value)`.
    #[cfg_attr(not(any(test, feature = "inference-cuda")), allow(dead_code))]
    pub(super) const DEFAULT: Style = Style {
        ascii: true,
        indent: None,
        spaced: true,
    };
    fn item_separator(self) -> &'static str {
        if self.indent.is_none() && self.spaced {
            ", "
        } else {
            ","
        }
    }
    fn key_separator(self) -> &'static str {
        if self.indent.is_some() || self.spaced {
            ": "
        } else {
            ":"
        }
    }
    fn newline(self, level: usize, out: &mut String) {
        if let Some(width) = self.indent {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', width * level));
        }
    }
}

pub(super) fn dumps(value: &Json, style: Style) -> Result<String> {
    let mut out = String::new();
    dump(value, style, 0, &mut out)?;
    Ok(out)
}

pub(super) fn dump(value: &Json, style: Style, level: usize, out: &mut String) -> Result<()> {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Int(i) => out.push_str(&i.to_string()),
        Json::Float(f) => {
            ensure!(
                f.is_finite(),
                "ValueError: Out of range float values are not JSON compliant"
            );
            out.push_str(&float_repr(*f));
        }
        Json::Str(s) => quote(s, style.ascii, out),
        Json::Array(items) if items.is_empty() => out.push_str("[]"),
        Json::Object(o) if o.0.is_empty() => out.push_str("{}"),
        Json::Array(items) => {
            out.push('[');
            for (position, item) in items.iter().enumerate() {
                if position > 0 {
                    out.push_str(style.item_separator());
                }
                style.newline(level + 1, out);
                dump(item, style, level + 1, out)?;
            }
            style.newline(level, out);
            out.push(']');
        }
        Json::Object(o) => {
            out.push('{');
            for (position, (key, item)) in o.0.iter().enumerate() {
                if position > 0 {
                    out.push_str(style.item_separator());
                }
                style.newline(level + 1, out);
                quote(key, style.ascii, out);
                out.push_str(style.key_separator());
                dump(item, style, level + 1, out)?;
            }
            style.newline(level, out);
            out.push('}');
        }
    }
    Ok(())
}

/// A JSON string literal, escaped as Python's encoder escapes it.
pub(super) fn quote(text: &str, ascii: bool, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if u32::from(c) < 0x20 || (ascii && !(' '..='~').contains(&c)) => {
                for unit in c.encode_utf16(&mut [0; 2]) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: shortest round-trip digits, positional for exponents -4..=15.
pub(super) fn float_repr(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let scientific = format!("{:e}", x.abs());
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("LowerExp always writes an exponent");
    let exponent: i32 = exponent.parse().expect("LowerExp exponent is an integer");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let count = digits.len() as i32;
    let point = exponent + 1;
    let mut out = String::from(if x < 0.0 { "-" } else { "" });
    if -4 < point && point <= 16 {
        if point <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-point) as usize));
            out.push_str(&digits);
        } else if point < count {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        } else {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', (point - count) as usize));
            out.push_str(".0");
        }
    } else {
        out.push_str(&digits[..1]);
        if count > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(
            out,
            "e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        );
    }
    out
}

/// Python `str()` of a JSON value, as f-strings and `csv` render it.
pub(super) fn py_str(value: &Json) -> String {
    match value {
        Json::Str(s) => s.clone(),
        other => py_repr(other),
    }
}

pub(super) fn py_repr(value: &Json) -> String {
    match value {
        Json::Null => "None".into(),
        Json::Bool(true) => "True".into(),
        Json::Bool(false) => "False".into(),
        Json::Int(i) => i.to_string(),
        Json::Float(f) => float_repr(*f),
        Json::Str(s) => str_repr(s),
        Json::Array(items) => format!(
            "[{}]",
            items.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        Json::Object(o) => format!(
            "{{{}}}",
            o.0.iter()
                .map(|(k, v)| format!("{}: {}", str_repr(k), py_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Python `repr(str)`. Printability of non-ASCII characters follows the common Unicode
/// separator, format, control and private-use ranges rather than the full database.
pub(super) fn str_repr(text: &str) -> String {
    let delimiter = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::from(delimiter);
    for c in text.chars() {
        let code = u32::from(c);
        match c {
            c if c == delimiter || c == '\\' => {
                out.push('\\');
                out.push(c);
            }
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ if code < 0x20 || code == 0x7f => {
                let _ = write!(out, "\\x{code:02x}");
            }
            _ if code < 0x7f || printable(code) => out.push(c),
            _ if code <= 0xff => {
                let _ = write!(out, "\\x{code:02x}");
            }
            _ if code <= 0xffff => {
                let _ = write!(out, "\\u{code:04x}");
            }
            _ => {
                let _ = write!(out, "\\U{code:08x}");
            }
        }
    }
    out.push(delimiter);
    out
}

pub(super) fn printable(code: u32) -> bool {
    !matches!(
        code,
        0x80..=0xa0
            | 0xad
            | 0x061c
            | 0x1680
            | 0x180e
            | 0x2000..=0x200f
            | 0x2028..=0x202f
            | 0x205f..=0x206f
            | 0x3000
            | 0xd800..=0xf8ff
            | 0xfeff
            | 0xfff0..=0xfffb
            | 0xfffe..=0xffff
            | 0xe0000..=0xe0fff
            | 0xf0000..=0x10ffff
    )
}

/// One `csv.writer` row (excel dialect, minimal quoting, CRLF). `None` is an empty field.
pub(super) fn csv_row(fields: &[&Json], out: &mut String) {
    for (position, field) in fields.iter().enumerate() {
        if position > 0 {
            out.push(',');
        }
        let text = match field {
            Json::Null => String::new(),
            other => py_str(other),
        };
        if text.contains([',', '"', '\r', '\n']) {
            out.push('"');
            out.push_str(&text.replace('"', "\"\""));
            out.push('"');
        } else {
            out.push_str(&text);
        }
    }
    out.push_str("\r\n");
}

// ---------------------------------------------------------------------------------------------
// `statistics` with exact rational rounding.

/// An unsigned fixed-point magnitude in units of 2^-1074, wide enough for any finite f64 or
/// JSON integer and sums of them.
#[derive(Clone)]
struct Wide([u64; 40]);

impl Wide {
    const SCALE: u32 = 1074;
    fn zero() -> Self {
        Wide([0; 40])
    }
    fn add_shifted(&mut self, value: u64, shift: u32) {
        let (limb, offset) = ((shift / 64) as usize, shift % 64);
        let low = value << offset;
        let high = if offset == 0 {
            0
        } else {
            value >> (64 - offset)
        };
        self.add_at(limb, low);
        self.add_at(limb + 1, high);
    }
    fn add_at(&mut self, mut limb: usize, value: u64) {
        let mut carry = value;
        while carry != 0 {
            let (sum, overflow) = self.0[limb].overflowing_add(carry);
            self.0[limb] = sum;
            carry = u64::from(overflow);
            limb += 1;
        }
    }
    fn add_float(&mut self, x: f64) {
        let bits = x.abs().to_bits();
        let exponent = (bits >> 52) as u32;
        let fraction = bits & ((1 << 52) - 1);
        if exponent == 0 {
            self.add_shifted(fraction, 0);
        } else {
            self.add_shifted(fraction | (1 << 52), exponent - 1);
        }
    }
    fn add_int(&mut self, magnitude: u128) {
        self.add_shifted(magnitude as u64, Self::SCALE);
        self.add_shifted((magnitude >> 64) as u64, Self::SCALE + 64);
    }
    fn compare(&self, other: &Wide) -> Ordering {
        self.0.iter().rev().cmp(other.0.iter().rev())
    }
    /// `self - other`, for `self >= other`.
    fn minus(&self, other: &Wide) -> Wide {
        let mut result = Wide::zero();
        let mut borrow = false;
        for (limb, (a, b)) in self.0.iter().zip(&other.0).enumerate() {
            let (difference, first) = a.overflowing_sub(*b);
            let (difference, second) = difference.overflowing_sub(u64::from(borrow));
            result.0[limb] = difference;
            borrow = first || second;
        }
        result
    }
    /// Divide in place; returns the remainder.
    fn divide(&mut self, divisor: u64) -> u64 {
        let mut remainder = 0_u128;
        for limb in self.0.iter_mut().rev() {
            let current = (remainder << 64) | u128::from(*limb);
            *limb = (current / u128::from(divisor)) as u64;
            remainder = current % u128::from(divisor);
        }
        remainder as u64
    }
    fn bits(&self) -> u32 {
        self.0
            .iter()
            .rposition(|limb| *limb != 0)
            .map_or(0, |top| top as u32 * 64 + 64 - self.0[top].leading_zeros())
    }
    fn bit(&self, position: u32) -> bool {
        (self.0[(position / 64) as usize] >> (position % 64)) & 1 == 1
    }
    fn any_below(&self, position: u32) -> bool {
        let (limb, offset) = ((position / 64) as usize, position % 64);
        self.0[..limb].iter().any(|l| *l != 0) || self.0[limb] & ((1 << offset) - 1) != 0
    }
    /// The 64 bits starting at `shift`.
    fn window(&self, shift: u32) -> u64 {
        let (limb, offset) = ((shift / 64) as usize, shift % 64);
        let low = self.0[limb] >> offset;
        if offset == 0 || limb + 1 >= self.0.len() {
            low
        } else {
            low | (self.0[limb + 1] << (64 - offset))
        }
    }
    /// `self / divisor` × 2^-1074, rounded half to even.
    fn quotient(mut self, divisor: u64) -> f64 {
        let remainder = self.divide(divisor);
        let length = self.bits();
        let (mantissa, shift, up) = if length <= 53 {
            let mantissa = self.window(0);
            let twice = 2 * u128::from(remainder);
            let half = u128::from(divisor);
            (
                mantissa,
                0,
                twice > half || (twice == half && mantissa & 1 == 1),
            )
        } else {
            let shift = length - 53;
            let mantissa = self.window(shift) & ((1 << 53) - 1);
            let sticky = self.any_below(shift - 1) || remainder != 0;
            (
                mantissa,
                shift,
                self.bit(shift - 1) && (sticky || mantissa & 1 == 1),
            )
        };
        (mantissa + u64::from(up)) as f64 * power_of_two(shift as i32 - Self::SCALE as i32)
    }
}

pub(super) fn power_of_two(exponent: i32) -> f64 {
    if exponent >= -1022 {
        f64::from_bits(((exponent + 1023) as u64) << 52)
    } else {
        f64::from_bits(1 << (exponent + 1074))
    }
}

/// The correctly rounded quotient of an exact signed sum and a count.
pub(super) fn rational(values: &[Number], count: u64) -> f64 {
    let (mut positive, mut negative) = (Wide::zero(), Wide::zero());
    for value in values {
        match *value {
            Number::Int(i) if i < 0 => negative.add_int(i.unsigned_abs()),
            Number::Int(i) => positive.add_int(i.unsigned_abs()),
            Number::Float(f) if f.is_sign_negative() => negative.add_float(f),
            Number::Float(f) => positive.add_float(f),
        }
    }
    match positive.compare(&negative) {
        Ordering::Less => -negative.minus(&positive).quotient(count),
        _ => positive.minus(&negative).quotient(count),
    }
}

/// `statistics.mean`: an exact mean, an int when every value is an int and it divides.
pub(super) fn mean(values: &[Json]) -> Result<Json> {
    let numbers = values
        .iter()
        .map(|v| number(v).context("TypeError: mean of a non-number"))
        .collect::<Result<Vec<_>>>()?;
    let count = numbers.len() as i128;
    if numbers.iter().all(|n| matches!(n, Number::Int(_))) {
        let total: i128 = numbers
            .iter()
            .map(|n| match n {
                Number::Int(i) => *i,
                Number::Float(_) => unreachable!(),
            })
            .sum();
        if total % count == 0 {
            return Ok(Json::Int(total / count));
        }
    }
    Ok(Json::Float(rational(&numbers, count as u64)))
}

/// `statistics.median` of an even-length sorted pair: `(a + b) / 2`.
pub(super) fn midpoint(a: &Json, b: &Json) -> Result<Json> {
    let (Some(x), Some(y)) = (number(a), number(b)) else {
        bail!("TypeError: median of a non-number");
    };
    Ok(Json::Float(match (x, y) {
        (Number::Int(i), Number::Int(j)) => rational(&[Number::Int(i + j)], 2),
        _ => (x.to_f64() + y.to_f64()) / 2.0,
    }))
}

/// Sample count, median, 95th-percentile order statistic and mean, or nulls without samples.
pub(super) fn distribution(values: &[Json]) -> Result<Json> {
    let mut sorted = values.to_vec();
    let mut failure = None;
    sorted.sort_by(|a, b| {
        py_order(a, b).unwrap_or_else(|error| {
            failure.get_or_insert(error);
            Ordering::Equal
        })
    });
    if let Some(error) = failure {
        return Err(error);
    }
    let n = sorted.len();
    if n == 0 {
        return Ok(object([
            ("n", 0_usize.into()),
            ("median_seconds", Json::Null),
            ("p95_seconds", Json::Null),
            ("mean_seconds", Json::Null),
        ]));
    }
    let median = if n % 2 == 1 {
        sorted[n / 2].clone()
    } else {
        midpoint(&sorted[n / 2 - 1], &sorted[n / 2])?
    };
    let p95 = sorted[(0.95 * n as f64).ceil() as usize - 1].clone();
    Ok(object([
        ("n", n.into()),
        ("median_seconds", median),
        ("p95_seconds", p95),
        ("mean_seconds", mean(&sorted)?),
    ]))
}

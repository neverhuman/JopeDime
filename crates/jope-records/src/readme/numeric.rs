use super::*;

/// A Python number (`bool` counts as `int`), as the renderer adds, divides and compares evidence
/// values. An `int` is exact, as Python's is: this reader's JSON integers fit `i64` (it refuses the
/// rest, stricter), and every sum of them is held in `i128`, which fewer than 2^64 of them cannot
/// leave (one that would is refused, never rounded or wrapped). A `float` is an IEEE double. An
/// `int` meets a `float` only as Python lets it: converted to the nearest double, ties to even,
/// for `+` and true division ([`Num::as_f64`]), and compared exactly, never through that
/// conversion ([`py_cmp`]).

#[derive(Clone, Copy, Debug)]
pub(crate) enum Num {
    Int(i128),
    Float(f64),
}

impl Num {
    fn of(value: &Value) -> Option<Num> {
        match value {
            Value::Bool(b) => Some(Num::Int(i128::from(*b))),
            Value::Int(i) => Some(Num::Int(i128::from(*i))),
            Value::Float(f) => Some(Num::Float(*f)),
            _ => None,
        }
    }

    fn type_name(self) -> &'static str {
        match self {
            Num::Int(_) => "int",
            Num::Float(_) => "float",
        }
    }

    /// `float(self)`: an `int` rounded to the nearest double, ties to even (what `i128 as f64`
    /// does, and CPython's `PyLong_AsDouble`), as Python converts one for `+` with a float, for a
    /// float format and for `int / int` of small operands.
    fn as_f64(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(f) => f,
        }
    }

    fn is_nan(self) -> bool {
        matches!(self, Num::Float(f) if f.is_nan())
    }

    /// Python's `self + other`: two ints add exactly; an int meeting a float is converted first
    /// and the two doubles added once (`int.__add__` returns `NotImplemented`, `float.__radd__`
    /// converts).
    fn plus(self, other: Num) -> std::result::Result<Num, String> {
        Ok(match (self, other) {
            (Num::Int(a), Num::Int(b)) => Num::Int(a.checked_add(b).ok_or_else(|| {
                "the exact integer sum leaves the 128-bit range the port holds it in".to_owned()
            })?),
            (a, b) => Num::Float(a.as_f64() + b.as_f64()),
        })
    }

    /// `self + value`, or the `TypeError` Python raised for a value that is not a number.
    fn add(self, value: &Value) -> std::result::Result<Num, String> {
        let other = Num::of(value).ok_or_else(|| {
            format!(
                "TypeError: unsupported operand type(s) for +: '{}' and '{}'",
                self.type_name(),
                py::type_name(value)
            )
        })?;
        self.plus(other)
    }

    /// `str(self)`: an int's decimal digits at any size, a float's `repr`.
    fn py_str(self) -> String {
        match self {
            Num::Int(i) => i.to_string(),
            Num::Float(f) => Value::Float(f).repr(),
        }
    }
}

/// A JSON number read as Python read it, or the refusal naming what was found instead (the Python
/// raised `ValueError` or `TypeError` where it formatted, added or compared it).
pub(crate) fn num_of(value: &Value, what: &str) -> Result<Num> {
    Num::of(value).ok_or_else(|| {
        Error::Shape(format!(
            "{what}: expected a number, found {} ({})",
            value.repr(),
            py::type_name(value)
        ))
    })
}

/// Python's comparison of two numbers: exact, an `int` against a `float` included (CPython's
/// `float_richcompare` never rounds the integer to a double first), `None` when either is a NaN,
/// with which every comparison is false.
pub(crate) fn py_cmp(a: Num, b: Num) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Num::Int(x), Num::Int(y)) => Some(x.cmp(&y)),
        (Num::Float(x), Num::Float(y)) => x.partial_cmp(&y),
        (Num::Int(i), Num::Float(f)) => int_float_cmp(i, f),
        (Num::Float(f), Num::Int(i)) => int_float_cmp(i, f).map(std::cmp::Ordering::reverse),
    }
}

/// An integer against a double, exactly: a double of magnitude 2^127 or more is beyond every
/// `i128`; any other is split into its integral part, which `as i128` holds exactly, and its
/// fraction, which decides between an integer and the double just past it.
pub(crate) fn int_float_cmp(i: i128, f: f64) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    const TWO_127: f64 = 170_141_183_460_469_231_731_687_303_715_884_105_728.0;
    if f.is_nan() {
        return None;
    }
    if f >= TWO_127 {
        return Some(Ordering::Less);
    }
    if f < -TWO_127 {
        return Some(Ordering::Greater);
    }
    let whole = f.trunc();
    Some(i.cmp(&(whole as i128)).then_with(|| {
        0.0_f64
            .partial_cmp(&(f - whole))
            .expect("a finite fraction")
    }))
}

/// `a < b` for two numbers, as Python compares them ([`py_cmp`]).
pub(crate) fn py_lt(a: Num, b: Num) -> bool {
    py_cmp(a, b) == Some(std::cmp::Ordering::Less)
}

/// `a == b` for two numbers, as Python compares them ([`py_cmp`]).
pub(crate) fn py_eq(a: Num, b: Num) -> bool {
    py_cmp(a, b) == Some(std::cmp::Ordering::Equal)
}

/// `a / b` for two Python integers (true division, `b > 0`): the double nearest the exact
/// quotient, ties to even, as CPython's `long_true_divide` rounds it, never the integer rounded to
/// a double first and the quotient rounded again. The quotient is developed in binary from the
/// exact integer division until it holds at least 55 significant bits, and the remainder that is
/// left decides a tie.
pub(crate) fn int_true_div(a: i128, b: u64) -> f64 {
    assert!(b > 0, "a count of at least one");
    let magnitude = a.unsigned_abs();
    if magnitude == 0 {
        return 0.0;
    }
    let b = u128::from(b);
    let (mut quotient, mut remainder) = (magnitude / b, magnitude % b);
    let mut scale = 0_i32;
    while quotient >> 54 == 0 {
        // `remainder < b <= u64::MAX`, so doubling it cannot overflow.
        remainder <<= 1;
        quotient <<= 1;
        scale += 1;
        if remainder >= b {
            quotient |= 1;
            remainder -= b;
        }
    }
    let shift = (128 - quotient.leading_zeros()) as i32 - 53;
    let mut kept = quotient >> shift;
    let dropped = quotient & ((1_u128 << shift) - 1);
    let half = 1_u128 << (shift - 1);
    if dropped > half || (dropped == half && (remainder != 0 || kept & 1 == 1)) {
        kept += 1;
    }
    // `kept <= 2^53` is exact as a double, and so is the power of two scaling it (a normal
    // exponent: the quotient lies between 2^-64 and 2^127).
    let exponent = shift - scale;
    let power = f64::from_bits(u64::try_from(exponent + 1023).expect("a normal exponent") << 52);
    let value = kept as f64 * power;
    if a < 0 {
        -value
    } else {
        value
    }
}

/// `format(x, ".Nf")`: `nan` for any NaN (Python ignores its sign), `inf` and `-inf`.
pub(crate) fn fixed(x: f64, precision: usize) -> String {
    if x.is_finite() {
        format!("{x:.precision$}")
    } else {
        py_nonfinite(x)
    }
}

/// `sorted(values)` for numbers, as CPython 3.12's `list.sort` orders them, every comparison
/// exact ([`py_cmp`]: an int and a float, or two ints, that round to the same double are still
/// ordered). Without a NaN every comparison is a total order and any stable sort agrees with it.
/// With a NaN (every comparison with it false) the result depends on the algorithm, which is
/// reproduced for fewer than 64 values (one `count_run` then `binarysort`, all timsort does below
/// its minimum run); a longer list holding a NaN is refused rather than ordered differently.
pub(crate) fn py_sorted(values: &[Num], what: &str) -> Result<Vec<Num>> {
    let mut out = values.to_vec();
    let lt = py_lt;
    if !out.iter().any(|v| v.is_nan()) {
        out.sort_by(|a, b| py_cmp(*a, *b).expect("no NaN"));
        return Ok(out);
    }
    if out.len() >= 64 {
        return Err(Error::Shape(format!(
            "{what}: {} values including a NaN, which the port orders as Python did only below 64",
            out.len()
        )));
    }
    // count_run: the longest strictly descending or non-descending prefix.
    let mut run = 1;
    if out.len() > 1 {
        let descending = lt(out[1], out[0]);
        run = 2;
        while run < out.len() && lt(out[run], out[run - 1]) == descending {
            run += 1;
        }
        if descending {
            out[..run].reverse();
        }
    }
    // binarysort from the end of the run.
    for start in run..out.len() {
        let pivot = out[start];
        let (mut l, mut r) = (0, start);
        while l < r {
            let p = l + (r - l) / 2;
            if lt(pivot, out[p]) {
                r = p;
            } else {
                l = p + 1;
            }
        }
        out.copy_within(l..start, l + 1);
        out[l] = pivot;
    }
    Ok(out)
}

/// `max(values)`: the first value no later value is greater than, compared exactly ([`py_cmp`];
/// `>` is false for a NaN).
pub(crate) fn py_max(values: &[Num]) -> Num {
    let (&first, rest) = values.split_first().expect("at least one report");
    let mut best = first;
    for &value in rest {
        if py_lt(best, value) {
            best = value;
        }
    }
    best
}

/// CPython 3.12's `sum(values)` (start `0`, `builtin_sum_impl`) of JSON numbers, path for path.
/// Integers first add in a C `long` (64 bits). An integer that would leave it hands the exact sum
/// to generic `+` for itself and every item after it: integers stay exact, and a float is added
/// plainly, without compensation. A float met on the `long` path is added to the integer sum
/// instead and starts the float path: floats add with Neumaier's compensation, integers add
/// plainly, and the compensation is applied once at the end when it is finite.
pub(crate) fn py_sum(values: &[Num], what: &str) -> Result<Num> {
    let refused = |e: String| Error::Shape(format!("{what}: {e}"));
    let mut rest = values.iter().copied();
    let mut long: i64 = 0;
    let item = loop {
        match rest.next() {
            None => return Ok(Num::Int(i128::from(long))),
            Some(Num::Int(b)) => match i64::try_from(b).ok().and_then(|b| long.checked_add(b)) {
                Some(sum) => long = sum,
                None => break Num::Int(b),
            },
            Some(float) => break float,
        }
    };
    let mut result = Num::Int(i128::from(long)).plus(item).map_err(refused)?;
    if let Num::Float(mut sum) = result {
        let mut compensation = 0.0_f64;
        loop {
            match rest.next() {
                None => {
                    if compensation != 0.0 && compensation.is_finite() {
                        sum += compensation;
                    }
                    return Ok(Num::Float(sum));
                }
                Some(Num::Float(x)) => {
                    let t = sum + x;
                    if sum.abs() >= x.abs() {
                        compensation += (sum - t) + x;
                    } else {
                        compensation += (x - t) + sum;
                    }
                    sum = t;
                }
                // `f_result += (double)value` for an int that fits a C `long`, which every JSON
                // integer this reader accepts does.
                Some(Num::Int(i)) => match i64::try_from(i) {
                    Ok(value) => sum += value as f64,
                    Err(_) => {
                        // Beyond a `long`, generic `+` from here, the compensation unapplied.
                        result = Num::Float(sum).plus(Num::Int(i)).map_err(refused)?;
                        break;
                    }
                },
            }
        }
    }
    for item in rest {
        result = result.plus(item).map_err(refused)?;
    }
    Ok(result)
}

/// `total / n` (true division) for a [`py_sum`] result: a float divided by `float(n)`, an int
/// divided exactly and rounded once ([`int_true_div`]).
pub(crate) fn py_div(total: Num, n: usize) -> f64 {
    match total {
        Num::Float(f) => f / n as f64,
        Num::Int(i) => int_true_div(i, u64::try_from(n).expect("a count fits 64 bits")),
    }
}

/// `format(n, ",")` for an integer.
pub(crate) fn thousands(n: i64) -> String {
    let digits = n.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 {
        format!("-{out}")
    } else {
        out
    }
}

/// `format(x, ".4g")` (Python's general format, precision 4).
pub(crate) fn general4(x: f64) -> String {
    const P: i32 = 4;
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.to_owned();
    }
    if !x.is_finite() {
        return py_nonfinite(x);
    }
    // Round to P significant digits in scientific form, then choose the presentation.
    let scientific = format!("{:.*e}", (P - 1) as usize, x);
    let (mantissa, exponent) = scientific.split_once('e').expect("scientific form");
    let exponent: i32 = exponent.parse().expect("exponent");
    if (-4..P).contains(&exponent) {
        let decimals = (P - 1 - exponent).max(0) as usize;
        strip_zeros(&format!("{x:.decimals$}"))
    } else {
        let mantissa = strip_zeros(mantissa);
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exponent.abs())
    }
}

pub(crate) fn py_nonfinite(x: f64) -> String {
    if x.is_nan() {
        "nan".to_owned()
    } else if x > 0.0 {
        "inf".to_owned()
    } else {
        "-inf".to_owned()
    }
}

pub(crate) fn strip_zeros(text: &str) -> String {
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        text.to_owned()
    }
}

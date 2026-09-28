//! `readme-benchmarks --check|--write`: render the README's measured-results blocks from
//! committed evidence.
//!
//! Rust port of `deploy/ci/readme-benchmarks.py` (archived at `8a55c4fef`; see
//! `docs/archive/python-archive-manifest.json`) producing the same bytes. Three independent marker
//! pairs, so regenerating one can never clobber another:
//!
//! ```text
//! <!-- arena:begin --> ... <!-- arena:end -->       the Golden45 ladders
//! <!-- samecut:begin --> ... <!-- samecut:end -->   the same-cut ladder this tree reproduces
//! <!-- bench:begin --> ... <!-- bench:end -->       emit-path latency
//! ```
//!
//! Nothing here is hand-typed. Every number is read from a file that is already under test:
//! `crates/jope-arena/tests/elo_regression.rs` asserts that both ladders recompute from
//! `scores.csv`, so this renderer does not re-derive them — it formats what those tests already
//! guarantee, records where every number came from, and refuses to render when the committed
//! evidence disagrees with itself. **benchmark-valid** is the number this repository claims;
//! **published** is what the benchmark ranked, rendered as the best known program per dataset
//! and labelled as not a held-out result.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use regex::Regex;

use crate::py;
use crate::pycsv::{self, Newlines};
use crate::pyjson::{Dict, Value};
use crate::{Error, Result};

const REGRESSION: &str = "assets/regression";
const ARENA: &str = "assets/regression/sotarena-tabular-regression-v2";
const TRACK: &str =
    "assets/regression/sotarena-tabular-regression-v2/tracks/tabular-regression-v2/track.json";
const LEADERBOARD: &str =
    "assets/regression/sotarena-tabular-regression-v2/results/tabular-regression-v2/leaderboard.csv";
const SNAPSHOT: &str = "assets/regression/elo-tabular-regression-v2.json";
const VALID_LEADERBOARD: &str =
    "assets/regression/leaderboard-tabular-regression-v2.benchmark-valid.csv";
const VALID_SNAPSHOT: &str = "assets/regression/elo-tabular-regression-v2.benchmark-valid.json";
const CORRECTIONS: &str = "assets/regression/protocol-corrections-tabular-regression-v2.json";
const BENCH: &str = "assets/regression/bench-baseline.json";
const README: &str = "README.md";
const SAMECUT: &str = "assets/same-cut/v2";
const SAMECUT_ADAPTERS: [&str; 4] = ["catboost", "lightgbm", "rf", "xgboost"];
const SAMECUT_TRACK_ID: &str = "jope-same-cut-v1";
const SAMECUT_DOC: &str = "docs/same-cut-bench.md";
const REPORT_SCHEMA: &str = "jope.bench-report/v2";
const SEARCHED: &str = "afm.lite-race.v1.resampled";
const LLM_REFERENCE_HOURS: i64 = 1;
const LLM_REFERENCE_TOKENS: i64 = 10_000_000;
const SNAPSHOT_SCHEMA: &str = "jope.arena-elo-snapshot/v2";
const CORRECTIONS_SCHEMA: &str = "jope.arena-protocol-corrections/v1";
const OURS: [&str; 2] = ["jope-dime", "jope-dime-regressor"];
const ARENA_BEGIN: &str = "<!-- arena:begin -->";
const ARENA_END: &str = "<!-- arena:end -->";
const SAMECUT_BEGIN: &str = "<!-- samecut:begin -->";
const SAMECUT_END: &str = "<!-- samecut:end -->";
const BENCH_BEGIN: &str = "<!-- bench:begin -->";
const BENCH_END: &str = "<!-- bench:end -->";
const ANCHOR: &str = "## Install and check";
const TOP_N: usize = 5;

fn samecut_path(name: &str) -> String {
    format!("{SAMECUT}/{name}")
}

struct Paths {
    root: PathBuf,
}

impl Paths {
    fn at(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }
    fn sha256(&self, rel: &str) -> Result<String> {
        Ok(py::sha256_hex(&py::read_bytes(&self.at(rel))?))
    }
    fn json(&self, rel: &str) -> Result<Value> {
        py::load_json(&self.at(rel))
    }
}

/// `refuse(why)`: the committed evidence disagrees with itself.
fn refuse<T>(why: impl Into<String>) -> Result<T> {
    Err(Error::Invalid(format!(
        "{}\nThe committed benchmark evidence disagrees with itself; refusing to publish.",
        why.into()
    )))
}

fn shape(what: &str) -> Error {
    Error::Shape(format!("{what} missing"))
}

/// `value[key]`: a missing key (Python's `KeyError`) or a value that is not a dict (its
/// `TypeError`) refuses, naming the key and, for a value that is not a dict, its type.
fn at<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    match value {
        Value::Dict(dict) => dict.get(key).ok_or_else(|| shape(&pyrepr_key(key))),
        other => Err(raised(
            pyrepr_key(key),
            py::exception::not_subscriptable(other),
        )),
    }
}

fn pyrepr_key(key: &str) -> String {
    crate::pyjson::str_repr(key)
}

fn get<'a>(value: &'a Value, key: &str) -> &'a Value {
    static NULL: Value = Value::Null;
    value.get(key).unwrap_or(&NULL)
}

/// `str(value)` as an f-string prints it.
fn s(value: &Value) -> String {
    value.py_str()
}

fn cell<'a>(row: &'a Dict, key: &str) -> Result<&'a str> {
    row.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| shape(&pyrepr_key(key)))
}

/// `int(row[key])` for a CSV cell through the strict number grammar ([`py::int`]): text that is
/// not an integer refuses (Python's `ValueError`), and so does one of more than 4300 digits,
/// leading zeros counted, in CPython's words. Underscores, Unicode digits and Unicode whitespace,
/// which Python read, are refused too (stricter), as are U+001C..U+001F around the digits, which
/// Python refused although `str.strip()` removes them.
fn int_cell(row: &Dict, key: &str) -> Result<i64> {
    let text = cell(row, key)?;
    py::int(text).map_err(|e| match e {
        py::IntError::Limit(message) => raised(format_args!("CSV column {key}"), message),
        py::IntError::Invalid => Error::Shape(format!(
            "CSV column {key}: expected an integer, found {}",
            pyrepr_key(text)
        )),
    })
}

/// `float(row[key])` for a CSV cell through the strict number grammar ([`py::float`]); text that
/// is not a number refuses (Python's `ValueError`). `nan`, `inf`, underscores, Unicode digits and
/// Unicode whitespace, which Python read, are refused too (stricter; no committed ladder writes
/// them).
fn float_cell(row: &Dict, key: &str) -> Result<f64> {
    let text = cell(row, key)?;
    py::float(text).ok_or_else(|| {
        Error::Shape(format!(
            "CSV column {key}: expected a number, found {}",
            pyrepr_key(text)
        ))
    })
}

/// A JSON number formatted with a float format (`f"{value:.2f}"`, which converts an `int` to the
/// nearest double); any other type refuses (the Python raised `ValueError` or `TypeError` in
/// `format`).
fn float_of(value: &Value, what: &str) -> Result<f64> {
    Ok(num_of(value, what)?.as_f64())
}

/// `f"{valid['rankable_rows']:,}"`: an integer. A float refuses, which is stricter than the
/// Python (it printed `format(8718.0, ",")`, `8,718.0`); any other type refuses as it did.
fn rankable_rows(valid: &Value) -> Result<i64> {
    let value = at(valid, "rankable_rows")?;
    value.as_int().ok_or_else(|| {
        Error::Shape(format!(
            "{VALID_SNAPSHOT} rankable_rows: expected an integer, found {} ({})",
            value.repr(),
            py::type_name(value)
        ))
    })
}

/// Malformed evidence the Python crashed on (a traceback, exit status 1): the field, then what
/// could not be done with it and its type (see [`py::exception`]; the text is not CPython's).
fn raised(context: impl std::fmt::Display, exception: String) -> Error {
    Error::Shape(format!("{context}: {exception}"))
}

/// `value[key]` (a string key); refuses where the Python raised (missing key, not a dict).
fn sub<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Value> {
    py::subscript(value, key).map_err(|e| raised(context, e))
}

/// `for x in value`: the items (a list's, a dict's keys, a string's characters), or Python's
/// `TypeError`.
fn items(value: &Value, context: &str) -> Result<Vec<Value>> {
    py::iterate(value).ok_or_else(|| raised(context, py::exception::not_iterable(value)))
}

/// `value.get(key)` on what the Python used as a dict (an `AttributeError` otherwise).
fn dict_get<'a>(value: &'a Value, key: &str, context: &str) -> Result<Option<&'a Value>> {
    match value {
        Value::Dict(dict) => Ok(dict.get(key)),
        other => Err(raised(context, py::exception::no_attribute(other, "get"))),
    }
}

/// `f"{value[:9]}"`: a string's first nine characters, or a list's first nine items.
fn first_nine(value: &Value, context: &str) -> Result<String> {
    match value {
        Value::Str(text) => Ok(text.chars().take(9).collect()),
        Value::List(list) => Ok(Value::List(list.iter().take(9).cloned().collect()).repr()),
        Value::Dict(_) => Err(Error::Shape(format!(
            "{context}: expected a string, found {}",
            value.repr()
        ))),
        other => Err(raised(
            context,
            format!(
                "TypeError: '{}' object is not subscriptable",
                py::type_name(other)
            ),
        )),
    }
}

/// `new < current` for two method ids that tie on RMSE, as the tuple comparison inside `min`
/// made it: equal ids are not less; strings compare, and numbers compare exactly ([`py_cmp`]: an
/// int and a float that round to the same double are still ordered); anything else raised.
fn id_less(new: &Value, current: &Value) -> Result<bool> {
    if new == current {
        return Ok(false);
    }
    match (new, current) {
        (Value::Str(a), Value::Str(b)) => Ok(a < b),
        _ => match (Num::of(new), Num::of(current)) {
            (Some(a), Some(b)) => Ok(py_lt(a, b)),
            _ => Err(raised(
                format_args!("{SAMECUT} method id"),
                format!(
                    "TypeError: '<' not supported between instances of '{}' and '{}'",
                    py::type_name(new),
                    py::type_name(current)
                ),
            )),
        },
    }
}

/// A dict built as `{key: value for ...}`: a repeated key keeps its first position (and its first
/// key object) and takes the last value; keys compare as Python hashes them.
type PyDict = Vec<(py::Key, Value, Value)>;

fn py_insert(dict: &mut PyDict, key: &Value, value: Value, context: &str) -> Result<()> {
    let hashed = py::Key::of(key).ok_or_else(|| raised(context, py::exception::unhashable(key)))?;
    match dict.iter_mut().find(|(k, _, _)| *k == hashed) {
        Some(entry) => entry.2 = value,
        None => dict.push((hashed, key.clone(), value)),
    }
    Ok(())
}

fn py_lookup<'a>(dict: &'a PyDict, key: &Value) -> Option<&'a Value> {
    let hashed = py::Key::of(key)?;
    dict.iter()
        .find(|(k, _, _)| *k == hashed)
        .map(|(_, _, v)| v)
}

/// `{a["id"]: a["rows"] for a in meta["applied"]}`.
fn applied_rows(meta: &Value, context: &str) -> Result<PyDict> {
    let mut applied = PyDict::new();
    for a in items(
        sub(meta, "applied", context)?,
        &format!("{context} applied"),
    )? {
        let id = sub(&a, "id", &format!("{context} applied"))?;
        let rows = sub(&a, "rows", &format!("{context} applied"))?.clone();
        py_insert(&mut applied, id, rows, &format!("{context} applied id"))?;
    }
    Ok(applied)
}

/// A Python number (`bool` counts as `int`), as the renderer adds, divides and compares evidence
/// values. An `int` is exact, as Python's is: this reader's JSON integers fit `i64` (it refuses the
/// rest, stricter), and every sum of them is held in `i128`, which fewer than 2^64 of them cannot
/// leave (one that would is refused, never rounded or wrapped). A `float` is an IEEE double. An
/// `int` meets a `float` only as Python lets it: converted to the nearest double, ties to even,
/// for `+` and true division ([`Num::as_f64`]), and compared exactly, never through that
/// conversion ([`py_cmp`]).
#[derive(Clone, Copy, Debug)]
enum Num {
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
fn num_of(value: &Value, what: &str) -> Result<Num> {
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
fn py_cmp(a: Num, b: Num) -> Option<std::cmp::Ordering> {
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
fn int_float_cmp(i: i128, f: f64) -> Option<std::cmp::Ordering> {
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
fn py_lt(a: Num, b: Num) -> bool {
    py_cmp(a, b) == Some(std::cmp::Ordering::Less)
}

/// `a == b` for two numbers, as Python compares them ([`py_cmp`]).
fn py_eq(a: Num, b: Num) -> bool {
    py_cmp(a, b) == Some(std::cmp::Ordering::Equal)
}

/// `a / b` for two Python integers (true division, `b > 0`): the double nearest the exact
/// quotient, ties to even, as CPython's `long_true_divide` rounds it, never the integer rounded to
/// a double first and the quotient rounded again. The quotient is developed in binary from the
/// exact integer division until it holds at least 55 significant bits, and the remainder that is
/// left decides a tie.
fn int_true_div(a: i128, b: u64) -> f64 {
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
fn fixed(x: f64, precision: usize) -> String {
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
fn py_sorted(values: &[Num], what: &str) -> Result<Vec<Num>> {
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
fn py_max(values: &[Num]) -> Num {
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
fn py_sum(values: &[Num], what: &str) -> Result<Num> {
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
fn py_div(total: Num, n: usize) -> f64 {
    match total {
        Num::Float(f) => f / n as f64,
        Num::Int(i) => int_true_div(i, u64::try_from(n).expect("a count fits 64 bits")),
    }
}

/// `format(n, ",")` for an integer.
fn thousands(n: i64) -> String {
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
fn general4(x: f64) -> String {
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

fn py_nonfinite(x: f64) -> String {
    if x.is_nan() {
        "nan".to_owned()
    } else if x > 0.0 {
        "inf".to_owned()
    } else {
        "-inf".to_owned()
    }
}

fn strip_zeros(text: &str) -> String {
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        text.to_owned()
    }
}

/// A snapshot and the CSV it hashes, cross-checked. Fails closed on any disagreement.
fn load_ladder(p: &Paths, snapshot: &str, csv: &str, ladder: &str) -> Result<(Value, Vec<Dict>)> {
    let snap = p.json(snapshot)?;
    let text = py::read_text(&p.at(csv))?;
    let rows = pycsv::dict_reader(&py::splitlines(&text).join("\n"), Newlines::LineFeed)?.rows;
    // `snap.get("schema")`
    dict_get(&snap, "schema", snapshot)?;
    if get(&snap, "schema") != &Value::from(SNAPSHOT_SCHEMA) {
        return refuse(format!(
            "{snapshot}: schema {} is not {SNAPSHOT_SCHEMA}",
            get(&snap, "schema").repr()
        ));
    }
    if get(&snap, "ladder") != &Value::from(ladder) {
        return refuse(format!(
            "{snapshot}: ladder {} is not {}",
            get(&snap, "ladder").repr(),
            crate::pyjson::str_repr(ladder)
        ));
    }
    // The snapshot records the hash of the ladder it was derived from. If the committed CSV has
    // moved since, the two files disagree and neither is safe to publish.
    let actual = p.sha256(csv)?;
    if get(&snap, "leaderboard_sha256") != &Value::from(actual.as_str()) {
        return refuse(format!(
            "{csv} sha256 {actual}\ndoes not match {snapshot} leaderboard_sha256 {}.",
            s(get(&snap, "leaderboard_sha256"))
        ));
    }
    if at(&snap, "methods")? != &Value::from(rows.len()) {
        return refuse(format!(
            "{snapshot}: {} methods but the CSV has {} rows",
            s(at(&snap, "methods")?),
            rows.len()
        ));
    }
    Ok((snap, rows))
}

struct Evidence {
    track: Value,
    published: Value,
    published_rows: Vec<Dict>,
    valid: Value,
    valid_rows: Vec<Dict>,
    corrections: Value,
}

fn load_evidence(p: &Paths) -> Result<Evidence> {
    let track = p.json(TRACK)?;
    let (published, published_rows) = load_ladder(p, SNAPSHOT, LEADERBOARD, "published")?;
    let (valid, valid_rows) = load_ladder(p, VALID_SNAPSHOT, VALID_LEADERBOARD, "benchmark-valid")?;
    let corrections = p.json(CORRECTIONS)?;
    if !at(&published, "corrections")?.is_null()
        || at(&published, "excluded_rows")? != &Value::Int(0)
    {
        return refuse(format!(
            "{SNAPSHOT}: the published ladder must apply no corrections"
        ));
    }
    // `corrections.get("schema")`
    dict_get(&corrections, "schema", CORRECTIONS)?;
    if get(&corrections, "schema") != &Value::from(CORRECTIONS_SCHEMA) {
        return refuse(format!("{CORRECTIONS}: schema is not {CORRECTIONS_SCHEMA}"));
    }
    // `a == b == c` is `a == b and b == c`: `c` is read only when `a == b`.
    let (a, b) = (
        at(&corrections, "scores_sha256")?,
        at(&published, "scores_sha256")?,
    );
    if !(a == b && b == at(&valid, "scores_sha256")?) {
        return refuse("the two ladders and the corrections do not bind the same scores.csv");
    }
    let meta = at(&valid, "corrections")?;
    if meta.is_null() || at(meta, "sha256")? != &Value::from(p.sha256(CORRECTIONS)?) {
        return refuse(format!(
            "{VALID_SNAPSHOT} was not computed with the committed {CORRECTIONS}"
        ));
    }
    if at(&valid, "excluded_rows")? != at(meta, "rows_corrected")? {
        // Equal while every published row is True. If the benchmark ever publishes a False row,
        // the ladder excludes it too and this sentence must be rewritten, not silently
        // under-counted.
        return refuse(format!(
            "the benchmark-valid ladder excluded {} rows but the corrections demoted {}; the README prose assumes they are the same set",
            s(at(&valid, "excluded_rows")?),
            s(at(meta, "rows_corrected")?)
        ));
    }
    for row in &valid_rows {
        // `r["benchmark_valid"] != "True"`: a short row's missing cell is None, which differs.
        let flag = row.get("benchmark_valid").ok_or_else(|| {
            raised(
                VALID_LEADERBOARD,
                py::exception::key_error("benchmark_valid"),
            )
        })?;
        if flag != &Value::from("True") {
            return refuse("a benchmark-invalid method survived into the benchmark-valid ladder");
        }
    }
    // Every correction must show up as fewer scored datasets for its method, so re-adding the
    // rows, or editing the corrections without regenerating the ladder, fails here. The ids,
    // methods and row counts are read as the Python read them: iterated, hashed and added as they
    // are, never skipped or defaulted, and added as Python adds them ([`Num::plus`]: integers
    // exact at any size, a float only once a float operand appears).
    let applied = applied_rows(meta, &format!("{VALID_SNAPSHOT} corrections"))?;
    let listed = items(
        sub(&corrections, "corrections", CORRECTIONS)?,
        &format!("{CORRECTIONS} corrections"),
    )?;
    let context = format!("{CORRECTIONS} corrections");
    let mut ids = std::collections::HashSet::new();
    for c in &listed {
        let id = sub(c, "id", &context)?;
        ids.insert(py::Key::of(id).ok_or_else(|| raised(&context, py::exception::unhashable(id)))?);
    }
    let applied_ids: std::collections::HashSet<py::Key> =
        applied.iter().map(|(k, _, _)| k.clone()).collect();
    if applied_ids != ids {
        return refuse("the corrections applied to the ladder are not the committed corrections");
    }
    // `corrected_by_method[m] = corrected_by_method.get(m, 0) + applied[c["id"]]`, in insertion
    // order like the Python dict.
    let mut by_method: Vec<(py::Key, Value, Num)> = vec![];
    for c in &listed {
        let method = sub(c, "method_id", &context)?;
        let key = py::Key::of(method)
            .ok_or_else(|| raised(&context, py::exception::unhashable(method)))?;
        let rows = py_lookup(&applied, sub(c, "id", &context)?).expect("the id sets are equal");
        let position = by_method.iter().position(|(k, _, _)| *k == key);
        let total = position.map_or(Num::Int(0), |i| by_method[i].2);
        let total = total
            .add(rows)
            .map_err(|e| raised(format_args!("{VALID_SNAPSHOT} corrections applied rows"), e))?;
        match position {
            Some(i) => by_method[i].2 = total,
            None => by_method.push((key, method.clone(), total)),
        }
    }
    let scored = |rows: &[Dict]| -> Result<BTreeMap<String, i64>> {
        rows.iter()
            .map(|r| {
                Ok((
                    cell(r, "method_id")?.to_owned(),
                    int_cell(r, "datasets_scored")?,
                ))
            })
            .collect()
    };
    let (scored_published, scored_valid) = (scored(&published_rows)?, scored(&valid_rows)?);
    for (key, method, n) in &by_method {
        // `scored.get(method, 0)`: the ladders are keyed by the CSV's strings.
        let count = |scored: &BTreeMap<String, i64>| {
            key.as_str()
                .and_then(|m| scored.get(m))
                .copied()
                .unwrap_or(0)
        };
        let (before, after) = (count(&scored_published), count(&scored_valid));
        // `before - after != n`: the difference of two `i64` counts exact in `i128` (it needs 65
        // bits at the extremes, where `i64` arithmetic wrapped), compared with the total exactly.
        let difference = Num::Int(i128::from(before) - i128::from(after));
        if !py_eq(difference, *n) {
            return refuse(format!(
                "{}: {} rows corrected but scored datasets went {before} -> {after}",
                method.py_str(),
                n.py_str()
            ));
        }
    }
    Ok(Evidence {
        track,
        published,
        published_rows,
        valid,
        valid_rows,
        corrections,
    })
}

fn table_row(r: &Dict) -> Result<String> {
    let ours = OURS.contains(&cell(r, "method_id")?);
    let bold = |text: &str| {
        if ours {
            format!("**{text}**")
        } else {
            text.to_owned()
        }
    };
    Ok(format!(
        "| {} | {} | {} | {} | {} | {}/{} | `{}` |",
        bold(cell(r, "elo_rank")?),
        bold(cell(r, "method")?),
        bold(cell(r, "elo")?),
        fixed(float_cell(r, "average_rank")?, 2),
        cell(r, "dataset_wins")?,
        cell(r, "datasets_scored")?,
        cell(r, "datasets_total")?,
        cell(r, "protocols")?
    ))
}

/// The top N plus every one of our entries, in rank order, with an ellipsis row per gap.
fn ladder_table(rows: &[Dict]) -> Result<Vec<String>> {
    let mut shown: std::collections::BTreeSet<usize> = (0..TOP_N).collect();
    for (i, r) in rows.iter().enumerate() {
        if OURS.contains(&cell(r, "method_id")?) {
            shown.insert(i);
        }
    }
    let mut out = vec![
        "| rank | method | Elo | avg rank | dataset wins | coverage | protocol |".to_owned(),
        "| ---: | --- | ---: | ---: | ---: | --- | --- |".to_owned(),
    ];
    let mut previous: i64 = -1;
    for i in shown {
        if i as i64 > previous + 1 {
            out.push("| … | | | | | | |".to_owned());
        }
        out.push(table_row(rows.get(i).ok_or_else(|| shape("ladder row"))?)?);
        previous = i as i64;
    }
    Ok(out)
}

fn render_arena(p: &Paths) -> Result<String> {
    let e = load_evidence(p)?;
    let (track, corrections, published, valid) = (&e.track, &e.corrections, &e.published, &e.valid);
    let empty = Value::Dict(Dict::new());
    let no_list = Value::List(vec![]);
    // `track.get("split", {})`, `track.get("leaderboard_policy", {})`; each is then used as a dict.
    let split = dict_get(track, "split", TRACK)?.unwrap_or(&empty);
    let policy = dict_get(track, "leaderboard_policy", TRACK)?.unwrap_or(&empty);
    let split_at = format!("{TRACK} split");
    let policy_at = format!("{TRACK} leaderboard_policy");
    // `", ".join(split.get("excluded_datasets", [])) or "none"`
    let excluded = dict_get(split, "excluded_datasets", &split_at)?.unwrap_or(&no_list);
    let excluded = py::join(", ", excluded)
        .map_err(|e| raised(format_args!("{split_at} excluded_datasets"), e))?;
    let excluded = if excluded.is_empty() {
        "none".to_owned()
    } else {
        excluded
    };
    let excluded_rows = s(at(valid, "excluded_rows")?);
    let find = |rows: &[Dict], id: &str| -> Result<Option<Dict>> {
        let mut found = None;
        for r in rows {
            if cell(r, "method_id")? == id {
                found = Some(r.clone());
            }
        }
        Ok(found)
    };
    let applied = applied_rows(
        at(valid, "corrections")?,
        &format!("{VALID_SNAPSHOT} corrections"),
    )?;
    // `{c["id"]: c for c in corrections["corrections"]}`
    let context = format!("{CORRECTIONS} corrections");
    let mut by_id = PyDict::new();
    for c in items(sub(corrections, "corrections", CORRECTIONS)?, &context)? {
        let id = sub(&c, "id", &context)?.clone();
        py_insert(&mut by_id, &id, c, &context)?;
    }
    let (Some(golden46), Some(golden42)) = (
        py_lookup(
            &by_id,
            &Value::from("golden46-new3-20260723-v4-test-adaptive"),
        ),
        py_lookup(
            &by_id,
            &Value::from("golden42-evolution-envelope-test-selected"),
        ),
    ) else {
        return refuse(
            "the README prose is written for the two committed corrections; update both together",
        );
    };
    // `c["evidence"][0].split(" ")[0]`
    let first_ref = |c: &Value| -> Result<String> {
        let evidence = sub(c, "evidence", &context)?;
        let first =
            py::first_item(evidence).map_err(|e| raised(format_args!("{context} evidence"), e))?;
        let text = first.as_str().ok_or_else(|| {
            raised(
                format_args!("{context} evidence"),
                py::exception::no_attribute(&first, "split"),
            )
        })?;
        Ok(text.split(' ').next().unwrap_or_default().to_owned())
    };
    let public = if get(split, "test_labels_public").truthy() {
        "public"
    } else {
        "**not public**"
    };
    // `track['primary_metric'].upper()`
    let metric = at(track, "primary_metric")?;
    let metric = metric
        .as_str()
        .ok_or_else(|| {
            raised(
                format_args!("{TRACK} primary_metric"),
                py::exception::no_attribute(metric, "upper"),
            )
        })?
        .to_uppercase();
    let mut out = vec![
        ARENA_BEGIN.to_owned(),
        format!(
            "### {} — {}, {} datasets",
            s(at(track, "public_name")?),
            s(at(track, "task")?),
            s(at(published, "datasets_scored")?)
        ),
        String::new(),
        format!(
            "`{}` · primary metric **{metric}** ({}) · Elo by `{}` · test labels are {public}; {} datasets are restricted; excluded datasets: {excluded}.",
            s(at(track, "id")?),
            s(at(track, "metric_direction")?),
            dict_get(policy, "elo_method", &policy_at)?.map_or_else(|| "unknown".to_owned(), s),
            split.get("restricted_dataset_count").map_or_else(|| "0".to_owned(), s),
        ),
        String::new(),
        "#### Benchmark-valid — the ladder this repository claims".to_owned(),
        String::new(),
        format!(
            "**{} methods** · {} rankable scores · {excluded_rows} published rows excluded as test-selected (see the next block). Bold rows are this repository's entries. *Benchmark-valid* here means that no artefact in this repository contradicts the row's published flag; it is not a claim that every comparator's protocol was verified.",
            s(at(valid, "methods")?),
            thousands(rankable_rows(valid)?),
        ),
        String::new(),
    ];
    out.extend(ladder_table(&e.valid_rows)?);
    let together = policy
        .get("protocols_ranked_together")
        .map_or_else(|| "False".to_owned(), s)
        .to_lowercase();
    out.extend([
        String::new(),
        format!(
            "**Read the protocol column before the Elo column.** `leaderboard_policy.protocols_ranked_together` is `{together}`, so entries produced under different protocols share one ladder. `jope-dime-regressor` is one fixed-seed reproducible run of this repository's engine under `public-development`, the same protocol as every other comparator. The same-cut benchmark is what turns this into a like-for-like comparison."
        ),
        String::new(),
        "#### Best known program per dataset — as published; test-selected, not a held-out result"
            .to_owned(),
        String::new(),
    ]);
    let Some(pr) = find(&e.published_rows, "jope-dime")? else {
        return refuse("jope-dime is missing from the published ladder");
    };
    if find(&e.valid_rows, "jope-dime")?.is_some() {
        // The prose below is written for the case where every jope-dime row is corrected.
        return refuse(
            "jope-dime still has benchmark-valid rows; the best-known block's prose assumes it has none",
        );
    }
    // `applied[c['id']]`
    let applied_of = |c: &Value| -> Result<String> {
        let id = sub(c, "id", &context)?;
        py_lookup(&applied, id).map(s).ok_or_else(|| {
            raised(
                format_args!("{VALID_SNAPSHOT} corrections applied"),
                format!("KeyError: {}", id.repr()),
            )
        })
    };
    let track_id = s(at(track, "id")?);
    out.extend([
        format!(
            "The benchmark published `jope-dime` at rank {}, Elo {}, {} dataset wins, {}/{} coverage, every row `benchmark_valid: True`. This repository's own code says otherwise about every one of those rows, so the benchmark-valid ladder above carries none of them:",
            cell(&pr, "elo_rank")?, cell(&pr, "elo")?, cell(&pr, "dataset_wins")?,
            cell(&pr, "datasets_scored")?, cell(&pr, "datasets_total")?
        ),
        String::new(),
        format!(
            "- **{} Golden42 rows** are each the per-dataset minimum RMSE over the ledger's test-scored candidate evaluations (63,108 across the 42 datasets), not the score of one program frozen before the test targets were opened. The red-team lane in this tree names that quantity the *best-test ASPFM envelope*, role `oracle_reference_only`, and keeps it out of its fair ladder (`{}`, `crates/jope-campaigns/src/commands/train_only_redteam.rs:402-407`).",
            applied_of(golden42)?, first_ref(golden42)?
        ),
        format!(
            "- **{} Golden46 rows** come from run `{}`, whose artefacts this repository's campaign code writes with `benchmark_valid: false, test_adaptive: true` (`{}`); the acceptance manifest relabelled the replayed winners `true`.",
            applied_of(golden46)?, s(at(golden46, "source_revision")?), first_ref(golden46)?
        ),
        String::new(),
        "Both are the best program found per dataset when candidates were scored on the test targets: reported here as best-known results, never as a held-out claim. The reproducible fixed-seed run of the same engine, `jope-dime-regressor`, is what stands on the ladder above. The same-cut benchmark replaces every row on this page with numbers this tree can reproduce.".to_owned(),
        String::new(),
        "| ladder | `jope-dime` | Elo | dataset wins | coverage |".to_owned(),
        "| --- | ---: | ---: | ---: | --- |".to_owned(),
        format!(
            "| as published | rank {} | {} | {} | {}/{} |",
            cell(&pr, "elo_rank")?, cell(&pr, "elo")?, cell(&pr, "dataset_wins")?,
            cell(&pr, "datasets_scored")?, cell(&pr, "datasets_total")?
        ),
        format!(
            "| benchmark-valid | not ranked | — | — | 0/{} (every row test-selected) |",
            cell(&pr, "datasets_total")?
        ),
        String::new(),
        format!(
            "Sources: published `{LEADERBOARD}` (sha256 `{}…`), recomputed byte-for-byte from `scores.csv`; benchmark-valid `{VALID_LEADERBOARD}` (sha256 `{}…`), the same `scores.csv` with `{CORRECTIONS}` (sha256 `{}…`) applied. Both are asserted by `crates/jope-arena/tests/elo_regression.rs`. Reproduce offline:",
            &p.sha256(LEADERBOARD)?[..16], &p.sha256(VALID_LEADERBOARD)?[..16], &p.sha256(CORRECTIONS)?[..16]
        ),
        String::new(),
        "```sh".to_owned(),
        "cargo run -p jope-arena --bin jope-arena -- elo \\".to_owned(),
        format!("  --track {track_id} --root {ARENA} --check"),
        "cargo run -p jope-arena --bin jope-arena -- elo \\".to_owned(),
        format!("  --track {track_id} --root {ARENA} \\"),
        format!("  --ladder benchmark-valid --corrections {CORRECTIONS} \\"),
        format!("  --check --committed {VALID_LEADERBOARD}"),
        "```".to_owned(),
        ARENA_END.to_owned(),
    ]);
    let _ = REGRESSION;
    Ok(out.join("\n"))
}

struct SameCut {
    deep: Value,
    deep_rows: Vec<Dict>,
    reports: Vec<Value>,
    ours: Dict,
    engines: BTreeMap<String, Dict>,
    versions: BTreeMap<String, String>,
    specs: Vec<Value>,
}

fn kind_of(m: &Value) -> Result<String> {
    Ok(s(at(at(m, "method")?, "kind")?))
}

fn engine_specs(r: &Value) -> Result<Vec<Value>> {
    let mut specs = vec![];
    for m in &items(at(r, "methods")?, &format!("{SAMECUT} report methods"))? {
        if kind_of(m)? == "adapter" {
            specs.push(at(at(m, "method")?, "adapter")?.clone());
        }
    }
    let mut keyed: Vec<(String, Value)> = specs
        .into_iter()
        .map(|spec| Ok((s(at(&spec, "name")?), spec)))
        .collect::<Result<_>>()?;
    // `sorted(..., key=name)` is stable.
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(keyed.into_iter().map(|(_, spec)| spec).collect())
}

/// The committed same-cut evidence, cross-checked. Fails closed on any disagreement.
fn load_samecut(p: &Paths) -> Result<SameCut> {
    let deep_csv = samecut_path(&format!("leaderboard-{SAMECUT_TRACK_ID}-deep.csv"));
    let deep_snapshot = samecut_path(&format!("elo-{SAMECUT_TRACK_ID}-deep.json"));
    let screen_csv = samecut_path(&format!("leaderboard-{SAMECUT_TRACK_ID}-screen.csv"));
    let screen_snapshot = samecut_path(&format!("elo-{SAMECUT_TRACK_ID}-screen.json"));
    let (deep, deep_rows) = load_ladder(p, &deep_snapshot, &deep_csv, "published")?;
    let (screen, _screen_rows) = load_ladder(p, &screen_snapshot, &screen_csv, "published")?;
    let track = p.json(&samecut_path(&format!(
        "track/tracks/{SAMECUT_TRACK_ID}-deep/track.json"
    )))?;
    let rel = SAMECUT;
    let scores = samecut_path(&format!("track/results/{SAMECUT_TRACK_ID}-deep/scores.csv"));
    if at(&deep, "scores_sha256")? != &Value::from(p.sha256(&scores)?) {
        return refuse(format!(
            "{rel}: the deep snapshot was not computed from the committed scores.csv"
        ));
    }
    if at(&track, "id")? != &Value::from(format!("{SAMECUT_TRACK_ID}-deep"))
        || at(&track, "primary_metric")? != &Value::from("rmse")
    {
        return refuse(format!(
            "{rel}: the deep track is not {SAMECUT_TRACK_ID}-deep ranked by rmse"
        ));
    }
    if !at(&deep, "corrections")?.is_null() || at(&deep, "excluded_rows")? != &Value::Int(0) {
        return refuse(format!("{rel}: the same-cut ladder applies no corrections"));
    }
    // `sorted(reports.glob("*.bench-report.json"))`: every entry whose name matches, of any kind
    // (pathlib's `*` matches a leading dot too, and a name that is not UTF-8), each then read as
    // JSON. A directory or an entry that cannot be listed refuses (pathlib listed nothing and the
    // Python refused with "no reports"). A matching name that is not UTF-8 refuses (stricter): the
    // Python read that report too, and the port's paths and messages are text.
    let reports_dir = p.at(&samecut_path("reports"));
    let mut report_paths: Vec<PathBuf> = vec![];
    for entry in std::fs::read_dir(&reports_dir).map_err(|e| Error::io(&reports_dir, e))? {
        let path = entry.map_err(|e| Error::io(&reports_dir, e))?.path();
        let name = path.file_name().expect("a listed entry has a name");
        if !std::os::unix::ffi::OsStrExt::as_bytes(name).ends_with(b".bench-report.json") {
            continue;
        }
        if name.to_str().is_none() {
            return Err(Error::Shape(format!(
                "{rel}: report {} is not UTF-8 (the Python read it as one of the reports)",
                py::os_display(path.as_os_str())
            )));
        }
        report_paths.push(path);
    }
    report_paths.sort();
    let reports: Vec<Value> = report_paths
        .iter()
        .map(|path| py::load_json(path))
        .collect::<Result<_>>()?;
    if reports.is_empty() {
        return refuse(format!("{rel}: no reports"));
    }
    let n = Value::from(reports.len());
    if !(&n == at(&deep, "datasets_scored")?
        && at(&deep, "datasets_scored")? == at(&screen, "datasets_scored")?)
    {
        return refuse(format!(
            "{rel}: {} reports but the ladders scored other dataset counts",
            reports.len()
        ));
    }
    let first = &reports[0];
    for r in &reports {
        // `r['dataset_id']`, read only when a refusal names it (as the Python formatted it).
        let id = || -> Result<String> { Ok(s(sub(r, "dataset_id", rel)?)) };
        if at(r, "schema")? != &Value::from(REPORT_SCHEMA) {
            return refuse(format!(
                "{rel}: report schema {} is not {REPORT_SCHEMA}",
                at(r, "schema")?.repr()
            ));
        }
        if !(at(r, "task")? == &Value::from("regression")
            && at(r, "ladder_metric")? == &Value::from("rmse")
            && at(r, "metric_direction")? == &Value::from("minimize"))
        {
            return refuse(format!(
                "{rel}: {} is not a regression report ranked by rmse",
                id()?
            ));
        }
        for key in ["protocol", "search", "git_sha", "host", "evaluator"] {
            if get(r, key) != get(first, key) || get(r, key).is_null() {
                return refuse(format!(
                    "{rel}: {} differs from the first report in {key}",
                    id()?
                ));
            }
        }
        // The plan key names the dataset; seed and fold count are what one run shares.
        let plan = at(r, "plan")?;
        let first_plan = at(first, "plan")?;
        if at(plan, "seed")? != at(first_plan, "seed")? || at(plan, "k")? != at(first_plan, "k")? {
            return refuse(format!(
                "{rel}: {} differs from the first report in fold seed or k",
                id()?
            ));
        }
        // `for m in r["methods"]`, iterated as the Python iterated it.
        let methods = items(at(r, "methods")?, &format!("{rel} report methods"))?;
        let mut searched = vec![];
        for m in &methods {
            if kind_of(m)? == "search" {
                searched.push(m.clone());
            }
        }
        if searched.len() != 1 || at(at(&searched[0], "method")?, "id")? != &Value::from(SEARCHED) {
            return refuse(format!(
                "{rel}: {} does not carry exactly one {SEARCHED} row",
                id()?
            ));
        }
        if get(at(&searched[0], "method")?, "resample") != at(r, "search")? {
            return refuse(format!(
                "{rel}: {}: the searched row is not the resampled procedure",
                id()?
            ));
        }
        let mut adapters = vec![];
        for m in &methods {
            if kind_of(m)? == "adapter" {
                adapters.push(s(at(at(at(m, "method")?, "adapter")?, "name")?));
            }
        }
        adapters.sort();
        if adapters != SAMECUT_ADAPTERS {
            return refuse(format!(
                "{rel}: {} carries adapters {}, not {}",
                id()?,
                py::str_list_repr(&adapters),
                py::str_list_repr(&SAMECUT_ADAPTERS)
            ));
        }
        for m in &methods {
            if kind_of(m)? == "adapter"
                && !get(at(at(m, "method")?, "adapter")?, "version").truthy()
            {
                return refuse(format!(
                    "{rel}: {}: {} records no engine version",
                    id()?,
                    s(at(at(m, "method")?, "id")?)
                ));
            }
        }
    }
    let row_for = |id: &str| -> Result<Option<Dict>> {
        for row in &deep_rows {
            if cell(row, "method_id")? == id {
                return Ok(Some(row.clone()));
            }
        }
        Ok(None)
    };
    let Some(ours) = row_for(SEARCHED)? else {
        return refuse(format!("{rel}: {SEARCHED} is missing from the deep ladder"));
    };
    let mut engines = BTreeMap::new();
    for name in SAMECUT_ADAPTERS {
        let Some(row) = row_for(&format!("adapter.{name}"))? else {
            return refuse(format!(
                "{rel}: adapter.{name} is missing from the deep ladder"
            ));
        };
        engines.insert(name.to_owned(), row);
    }
    // The engine specs come from the first report; every other report must have recorded the
    // same spec per engine, or the directory is not one run.
    let specs = engine_specs(first)?;
    for r in &reports {
        if engine_specs(r)? != specs {
            return refuse(format!(
                "{rel}: {} recorded different engine specs or versions",
                s(sub(r, "dataset_id", rel)?)
            ));
        }
    }
    let versions = specs
        .iter()
        .map(|spec| Ok((s(at(spec, "name")?), s(at(spec, "version")?))))
        .collect::<Result<_>>()?;
    Ok(SameCut {
        deep,
        deep_rows,
        reports,
        ours,
        engines,
        versions,
        specs,
    })
}

/// The `--adapter` argument that recorded this engine, rendered so a shell reads it back
/// exactly; a token that would need quoting refuses rather than print a wrong command.
fn adapter_flag(spec: &Value) -> Result<String> {
    // `re.compile(r"^[A-Za-z0-9_./=:+-]+$").match`: Python's `$` also matches before a final
    // newline, so a token ending in one newline passed there, and passes here.
    let safe = Regex::new(r"^[A-Za-z0-9_./=:+-]+\n?\z").expect("static pattern");
    let context = format!("{SAMECUT} adapter spec");
    // `tokens = [spec["name"], *spec["command"]]`: every token checked as it is, none skipped.
    let name = sub(spec, "name", &context)?;
    let command = sub(spec, "command", &context)?;
    let unpacked = py::iterate(command).ok_or_else(|| {
        raised(
            &context,
            format!(
                "TypeError: Value after * must be an iterable, not {}",
                py::type_name(command)
            ),
        )
    })?;
    for token in std::iter::once(name).chain(unpacked.iter()) {
        // `SAFE_TOKEN.match(token)`
        let text = token.as_str().ok_or_else(|| {
            raised(
                &context,
                format!(
                    "TypeError: expected string or bytes-like object, got '{}'",
                    py::type_name(token)
                ),
            )
        })?;
        if !safe.is_match(text) {
            return refuse(format!(
                "adapter {}: token {} is not shell-safe to render",
                name.repr(),
                token.repr()
            ));
        }
    }
    let command = py::join(" ", command).map_err(|e| raised(&context, e))?;
    Ok(format!("--adapter \"{}={command}\"", name.py_str()))
}

fn samecut_row(r: &Dict, versions: &BTreeMap<String, String>) -> Result<String> {
    let method_id = cell(r, "method_id")?;
    let b = if method_id == SEARCHED { "**" } else { "" };
    let mut name = cell(r, "method")?.to_owned();
    if let Some(engine) = method_id.strip_prefix("adapter.") {
        let version = versions
            .get(engine)
            .map_or("version unknown", String::as_str);
        name = format!("{name} ({version})");
    }
    Ok(format!(
        "| {b}{}{b} | {b}{name}{b} | {b}{}{b} | {} | {} | {}/{} |",
        cell(r, "elo_rank")?,
        cell(r, "elo")?,
        fixed(float_cell(r, "average_rank")?, 2),
        cell(r, "dataset_wins")?,
        cell(r, "datasets_scored")?,
        cell(r, "datasets_total")?
    ))
}

fn render_samecut(p: &Paths) -> Result<String> {
    let e = load_samecut(p)?;
    let (deep, reports, ours) = (&e.deep, &e.reports, &e.ours);
    let first = &reports[0];
    let (plan, protocol, search, host) = (
        at(first, "plan")?,
        at(first, "protocol")?,
        at(first, "search")?,
        at(first, "host")?,
    );
    let n = reports.len();
    let mut per_dataset = vec![];
    let mut search_seconds: Vec<Num> = vec![];
    let (mut finalist, mut best) = (0usize, 0usize);
    // `sorted(reports, key=lambda r: r["dataset_id"])` (stable). The ids are strings; ids of any
    // other type (which Python sorted only when they were mutually comparable numbers) are
    // refused rather than ordered by their text.
    let mut ordered: Vec<(&str, &Value)> = vec![];
    for r in reports {
        let id = sub(r, "dataset_id", SAMECUT)?;
        match (id.as_str(), reports.len()) {
            (Some(id), _) => ordered.push((id, r)),
            (None, 1) => ordered.push(("", r)),
            (None, _) => {
                return Err(Error::Shape(format!(
                    "{SAMECUT}: report dataset_id {} is not a string, so the reports cannot be ordered as the Python ordered them",
                    id.repr()
                )))
            }
        }
    }
    ordered.sort_by(|a, b| a.0.cmp(b.0));
    for (_, r) in ordered {
        let methods = items(at(r, "methods")?, &format!("{SAMECUT} report methods"))?;
        // `next(m for m in r["methods"] if m["method"]["kind"] == "search")`: every method before
        // the first search one is read, and one that cannot be read refuses.
        let mut s_index = None;
        for (i, m) in methods.iter().enumerate() {
            if kind_of(m)? == "search" {
                s_index = Some(i);
                break;
            }
        }
        let s_index = s_index.ok_or_else(|| shape("search method"))?;
        let sm = &methods[s_index];
        // "Best" is the ladder's own rule: the lowest pooled RMSE among the finalists that
        // scored, ties by method id: `min(scored, key=lambda m: (rmse, id))`, which keeps the
        // first of equal keys and compares the ids only on a tie. The RMSEs are compared as
        // Python compared them, exactly ([`py_cmp`]: an int RMSE and a float one that round to the
        // same double are not a tie).
        let mut top: Option<(usize, Num, Value)> = None;
        for (i, m) in methods.iter().enumerate() {
            let deep_stage = get(m, "deep");
            if !(deep_stage.truthy() && sub(deep_stage, "status", SAMECUT)? == &Value::from("ok")) {
                continue;
            }
            let rmse = num_of(
                sub(sub(deep_stage, "oof", SAMECUT)?, "rmse", SAMECUT)?,
                &format!("{SAMECUT} report deep oof rmse"),
            )?;
            let id = sub(at(m, "method")?, "id", SAMECUT)?.clone();
            let better = match &top {
                None => true,
                // Tuple `<`: the first pair that is neither identical nor equal decides, so the ids
                // decide between equal RMSEs, and between two NaNs (every JSON `NaN` is one object).
                Some((_, best_rmse, best_id)) => {
                    if py_eq(rmse, *best_rmse) || (rmse.is_nan() && best_rmse.is_nan()) {
                        id_less(&id, best_id)?
                    } else {
                        py_lt(rmse, *best_rmse)
                    }
                }
            };
            if better {
                top = Some((i, rmse, id));
            }
        }
        let Some((top_index, rmse, top_id)) = top else {
            return refuse(format!(
                "{SAMECUT}: {} has no scored finalist",
                s(sub(r, "dataset_id", SAMECUT)?)
            ));
        };
        let top_id = top_id.py_str();
        let produce = num_of(
            at(at(sm, "method")?, "produce_seconds")?,
            &format!("{SAMECUT} report search method produce_seconds"),
        )?;
        search_seconds.push(produce);
        let produce = produce.as_f64();
        let rank = get(sm, "deep_rank");
        if !rank.is_null() {
            finalist += 1;
        }
        let is_search = top_index == s_index;
        if is_search {
            best += 1;
        }
        let marker = if is_search { "**" } else { "" };
        let rank_text = if rank.is_null() {
            // `f'screen {s['screen_rank']}'`
            format!("screen {}", s(sub(sm, "screen_rank", SAMECUT)?))
        } else {
            s(rank)
        };
        per_dataset.push(format!(
            "| `{}` | {} | {} | {marker}{top_id}{marker} | {} | {rank_text} | {} s |",
            s(at(r, "dataset_id")?),
            s(at(r, "rows")?),
            s(at(r, "features")?),
            general4(rmse.as_f64()),
            fixed(produce, 1)
        ));
    }
    // `sum(s) / n`, the median of `sorted(s)` (the middle value, or `sum` of the middle two over
    // 2) and `max(s)`, with Python's arithmetic and its ordering of a NaN.
    let what = format!("{SAMECUT} report search method produce_seconds");
    let mean_s = py_div(py_sum(&search_seconds, &what)?, n);
    let sorted_s = py_sorted(&search_seconds, &what)?;
    let median_s = if n % 2 == 1 {
        sorted_s[n / 2].as_f64()
    } else {
        py_div(py_sum(&sorted_s[n / 2 - 1..n / 2 + 1], &what)?, 2)
    };
    let max_s = py_max(&search_seconds).as_f64();
    let git_sha = first_nine(at(first, "git_sha")?, &format!("{SAMECUT} report git_sha"))?;
    let mut out = vec![
        SAMECUT_BEGIN.to_owned(),
        "### Same-cut — every method on identical held-out rows, reproducible from this tree".to_owned(),
        String::new(),
        format!(
            "`{}` · {n} regression datasets from `datasets/test` · primary metric **RMSE** (minimize) · fold plan k={}, seed {}, screen fold {}, top {} deepened · search budget {} programs × {} rounds, re-run on every cut · recorded at `{}` with {} workers on {} cores.",
            s(at(deep, "track_id")?), s(at(plan, "k")?), s(at(plan, "seed")?),
            s(at(protocol, "screen_fold")?), s(at(protocol, "top")?), s(at(search, "pop")?),
            s(at(search, "gens")?), git_sha,
            s(at(host, "workers")?), s(at(host, "cores")?)
        ),
        String::new(),
        format!(
            "Every row here is a number this tree produced and can reproduce: the ten single-head baselines, four external engines (XGBoost, LightGBM, CatBoost and a scikit-learn random forest, each run as a subprocess with its library defaults, a fixed seed and one thread, handed a cut's training rows and its held-out features only, and scored by this benchmark from the predictions it returns) and the program the search wrote for each dataset, all fitted on the same training rows and scored on the same held-out rows by one evaluator (`{SAMECUT_DOC}`). The search sees only a cut's training rows, and its deep row is the *procedure* re-run on every cut, not one program chosen with knowledge of the other cuts' held-out rows. Bold is the searched program."
        ),
        String::new(),
        "**The datasets are a favourable selection.** They are the `test-10` collection, chosen from the published ladder where the historical engine ranked first (`docs/datasets.md`), and they are used here because they are the only committed regression data. Identical cuts make the comparison between methods fair on these ten; they say nothing about a broader panel. The frozen 30-dataset PMLB panel of the experiment contract (`docs/asm.md`) excludes them on purpose; a same-cut run over it is the next evidence step.".to_owned(),
        String::new(),
        "| rank | method | Elo | avg rank | dataset wins | coverage |".to_owned(),
        "| ---: | --- | ---: | ---: | ---: | --- |".to_owned(),
    ];
    for r in &e.deep_rows {
        out.push(samecut_row(r, &e.versions)?);
    }
    let ours_rank = int_cell(ours, "elo_rank")?;
    let mut standing = vec![];
    for name in SAMECUT_ADAPTERS {
        let engine = &e.engines[name];
        let side = if ours_rank < int_cell(engine, "elo_rank")? {
            "above"
        } else {
            "below"
        };
        standing.push(format!(
            "{side} {} (rank {})",
            cell(engine, "method")?,
            cell(engine, "elo_rank")?
        ));
    }
    let top = s(at(protocol, "top")?);
    out.extend([
        String::new(),
        format!(
            "**Where the search stands today:** rank {} of {} on the deep ladder, Elo {}, lowest pooled RMSE on {best} of {n} datasets (the ladder's own rule), a finalist on {finalist}; on the other {} it did not make the top-{top} screen. Against the four external engines it ranks {}. Coverage counts datasets where a method reached the deep stage, so the ladder ranks finalists only. The screen ladder (every method, one cut) is committed beside it.",
            cell(ours, "elo_rank")?, s(at(deep, "methods")?), cell(ours, "elo")?,
            n - finalist, standing.join(", ")
        ),
        String::new(),
        format!(
            "**Seconds per program.** Writing the program took the search {} s per dataset on average (median {} s, longest {} s) on the screen cut's training rows, wall time under whatever else the host was running. The owner's stated reference point for a large language model writing a program for the same task, not measured here, is about {LLM_REFERENCE_HOURS} hour and {} tokens; that comparison is one of cost, not of quality, and the quality standing is the ladder above.",
            fixed(mean_s, 1), fixed(median_s, 1), fixed(max_s, 1), thousands(LLM_REFERENCE_TOKENS)
        ),
        String::new(),
        "| dataset | rows | features | lowest pooled RMSE | RMSE | search rank (penalised score) | search |".to_owned(),
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |".to_owned(),
    ]);
    out.extend(per_dataset);
    let deep_csv = samecut_path(&format!("leaderboard-{SAMECUT_TRACK_ID}-deep.csv"));
    let deep_scores = samecut_path(&format!("track/results/{SAMECUT_TRACK_ID}-deep/scores.csv"));
    out.extend([
        String::new(),
        format!(
            "Sources: `{deep_csv}` (sha256 `{}…`) recomputed byte-for-byte from `{deep_scores}`, itself re-derived from the {n} reports under `{}` by `crates/jope-bench/tests/committed_evidence.rs`, which also re-evaluates one dataset's baselines on the same cuts. Provenance and the exact commands: `{SAMECUT}/SOURCES.md`. Reproduce:",
            &p.sha256(&deep_csv)?[..16], samecut_path("reports")
        ),
        String::new(),
        "```sh".to_owned(),
        "cargo run --release -p jope-bench -- run --datasets datasets/test --out target/same-cut/reports \\".to_owned(),
    ]);
    for spec in &e.specs {
        out.push(format!("  {} \\", adapter_flag(spec)?));
    }
    out.extend([
        format!(
            "  --top {top} --search --search-pop {} --search-gens {} --resample-search --workers {}",
            s(at(search, "pop")?), s(at(search, "gens")?), s(at(host, "workers")?)
        ),
        "cargo run --release -p jope-bench -- ladder --reports target/same-cut/reports \\".to_owned(),
        format!("  --track-root target/same-cut/track --track {SAMECUT_TRACK_ID}"),
        format!(
            "cargo run -p jope-arena -- elo --track {} --root target/same-cut/track \\",
            s(at(deep, "track_id")?)
        ),
        format!("  --check --committed {deep_csv}"),
        "```".to_owned(),
        SAMECUT_END.to_owned(),
    ]);
    Ok(out.join("\n"))
}

fn render_bench(p: &Paths) -> Result<String> {
    let b = p.json(BENCH)?;
    let num = |key: &str| -> Result<f64> { float_of(at(&b, key)?, &format!("{BENCH} {key}")) };
    let git_sha = first_nine(at(&b, "git_sha")?, &format!("{BENCH} git_sha"))?;
    Ok([
        BENCH_BEGIN.to_owned(),
        "### Emit-path latency".to_owned(),
        String::new(),
        format!(
            "`{}` — {} runs of the release `emit` example on `{}`.",
            s(at(&b, "bench")?),
            s(at(&b, "runs")?),
            s(at(&b, "genome")?)
        ),
        String::new(),
        "| min | median | mean | p95 |".to_owned(),
        "| ---: | ---: | ---: | ---: |".to_owned(),
        format!(
            "| {} ms | {} ms | {} ms | {} ms |",
            fixed(num("min_ms")?, 2),
            fixed(num("median_ms")?, 2),
            fixed(num("mean_ms")?, 2),
            fixed(num("p95_ms")?, 2)
        ),
        String::new(),
        format!(
            "**This is not an engine-speed number.** Its own `what` field says it measures \"{}\". The min-to-p95 spread is dominated by process spawn and scheduling, not by emission. Treat it as a regression tripwire (alert at {}% drift), not as a performance claim.",
            s(at(&b, "what")?),
            fixed(num("alert_pct")?, 0)
        ),
        String::new(),
        format!(
            "Recorded {} on `{}` ({}, {} cores, kernel {}) at `{}` with {}. Baselines are host-pinned; `bash deploy/ci/regression-suite.sh bench` compares against this file.",
            s(at(&b, "recorded")?),
            s(at(&b, "host")?),
            s(at(&b, "cpu")?),
            s(at(&b, "nproc")?),
            s(at(&b, "kernel")?),
            git_sha,
            s(at(&b, "rustc")?)
        ),
        BENCH_END.to_owned(),
    ]
    .join("\n"))
}

fn splice(text: &str, begin: &str, end: &str, block: &str) -> Result<String> {
    match (text.find(begin), text.find(end)) {
        (Some(i), Some(j)) => Ok(format!("{}{block}{}", &text[..i], &text[j + end.len()..])),
        _ => Err(Error::Invalid(format!(
            "markers {begin}/{end} not found; cannot splice"
        ))),
    }
}

/// The README with every block re-rendered (inserting the section on first use).
pub fn build(root: &Path, text: &str) -> Result<String> {
    let p = Paths {
        root: root.to_path_buf(),
    };
    let mut text = text.to_owned();
    if !text.contains(ARENA_BEGIN) {
        let Some(anchor) = text.find(ANCHOR) else {
            return Err(Error::Invalid(format!(
                "anchor {} not found in README",
                crate::pyjson::str_repr(ANCHOR)
            )));
        };
        let section = format!(
            "## Measured results\n\nGenerated by `jope-records readme-benchmarks` from committed evidence. Do not edit between the markers by hand.\n\n{ARENA_BEGIN}\n{ARENA_END}\n\n{SAMECUT_BEGIN}\n{SAMECUT_END}\n\n{BENCH_BEGIN}\n{BENCH_END}\n\n"
        );
        text = format!("{}{section}{}", &text[..anchor], &text[anchor..]);
    }
    if !text.contains(SAMECUT_BEGIN) {
        // The same-cut block was added after the other two: slot it between them.
        let Some(i) = text.find(BENCH_BEGIN) else {
            return Err(Error::Invalid(format!(
                "markers {BENCH_BEGIN} not found; cannot place the same-cut block"
            )));
        };
        text = format!(
            "{}{SAMECUT_BEGIN}\n{SAMECUT_END}\n\n{}",
            &text[..i],
            &text[i..]
        );
    }
    let text = splice(&text, ARENA_BEGIN, ARENA_END, &render_arena(&p)?)?;
    let text = splice(&text, SAMECUT_BEGIN, SAMECUT_END, &render_samecut(&p)?)?;
    splice(&text, BENCH_BEGIN, BENCH_END, &render_bench(&p)?)
}

/// Every input the render reads; a missing one refuses before anything is read.
pub fn required_inputs() -> Vec<String> {
    vec![
        LEADERBOARD.to_owned(),
        TRACK.to_owned(),
        SNAPSHOT.to_owned(),
        VALID_LEADERBOARD.to_owned(),
        VALID_SNAPSHOT.to_owned(),
        CORRECTIONS.to_owned(),
        BENCH.to_owned(),
        README.to_owned(),
        samecut_path(&format!("leaderboard-{SAMECUT_TRACK_ID}-deep.csv")),
        samecut_path(&format!("elo-{SAMECUT_TRACK_ID}-deep.json")),
        samecut_path(&format!("leaderboard-{SAMECUT_TRACK_ID}-screen.csv")),
        samecut_path(&format!("elo-{SAMECUT_TRACK_ID}-screen.json")),
        samecut_path(&format!("track/tracks/{SAMECUT_TRACK_ID}-deep/track.json")),
        samecut_path(&format!("track/results/{SAMECUT_TRACK_ID}-deep/scores.csv")),
        samecut_path("reports"),
        SAMECUT_DOC.to_owned(),
    ]
}

/// `--check` (write = false) or `--write`: the message, and whether the command succeeded.
pub fn run(root: &Path, write: bool) -> Result<(String, bool)> {
    for input in required_inputs() {
        let path = root.join(&input);
        // `p.exists()`: a missing input refuses with the Python's message; any other error (a
        // directory that may not be searched) refuses as the Python raised it.
        if !py::path_exists(&path).map_err(|e| Error::io(&path, e))? {
            return Err(Error::Invalid(format!(
                "missing required input: {}",
                py::os_display(path.as_os_str())
            )));
        }
    }
    let readme = root.join(README);
    let current = py::read_text(&readme)?;
    let updated = build(root, &current)?;
    if write {
        if updated != current {
            std::fs::write(&readme, &updated).map_err(|e| Error::io(&readme, e))?;
            return Ok(("README.md updated\n".to_owned(), true));
        }
        return Ok(("README.md already current\n".to_owned(), true));
    }
    if updated != current {
        return Ok((
            "README.md is out of date with the committed benchmark evidence.\nRun: cargo run -q --locked -p jope-records -- readme-benchmarks --write\n".to_owned(),
            false,
        ));
    }
    Ok(("README.md benchmark blocks are current\n".to_owned(), true))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python's `format` for the specs the blocks use.
    #[test]
    fn number_formats_match_python() {
        assert_eq!(fixed(0.125, 2), "0.12");
        assert_eq!(fixed(0.375, 2), "0.38");
        assert_eq!(fixed(2.5, 0), "2");
        assert_eq!(fixed(1.05, 1), "1.1");
        assert_eq!(thousands(10_000_000), "10,000,000");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(8763), "8,763");
        assert_eq!(general4(0.123456), "0.1235");
        assert_eq!(general4(1234.5), "1234");
        assert_eq!(general4(12345.0), "1.234e+04");
        assert_eq!(general4(0.0001234), "0.0001234");
        assert_eq!(general4(0.00001234), "1.234e-05");
        assert_eq!(general4(2.0), "2");
        assert_eq!(general4(0.5), "0.5");
        assert_eq!(general4(9999.5), "1e+04");
    }

    /// Integer true division rounds once, to the double nearest the exact quotient, ties to even
    /// (CPython's `long_true_divide`). For operands a double holds exactly that is one IEEE
    /// division; beyond 2^53 the integer is not rounded first.
    #[test]
    fn integer_true_division_rounds_once() {
        let mut seed: u64 = 0x2545_f491_4f6c_dd1d;
        for _ in 0..20_000 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let a = i128::from((seed >> 11) as i64) - (1_i128 << 52);
            let b = (seed & 0xffff) + 1;
            assert_eq!(int_true_div(a, b), a as f64 / b as f64, "{a} / {b}");
        }
        let two_53 = 1_i128 << 53;
        // The captured same-cut mean (`seconds-int-true-division`), and its negative.
        assert_eq!(
            int_true_div(5_000_000_000_000_000_321, 10),
            500_000_000_000_000_064.0
        );
        assert_eq!(
            int_true_div(-5_000_000_000_000_000_321, 10),
            -500_000_000_000_000_064.0
        );
        // 2^53 + 1 and 2^53 + 3 are ties, to the even neighbour; 2^53 + 1.5 is not.
        assert_eq!(int_true_div(2 * (two_53 + 1), 2), 9_007_199_254_740_992.0);
        assert_eq!(int_true_div(2 * (two_53 + 3), 2), 9_007_199_254_740_996.0);
        assert_eq!(int_true_div(2 * two_53 + 3, 2), 9_007_199_254_740_994.0);
        assert_eq!(
            int_true_div(i128::from(i64::MAX) * 2 - 1, 2),
            9_223_372_036_854_775_808.0
        );
        assert_eq!(int_true_div(0, 7), 0.0);
        assert_eq!(
            int_true_div(1, u64::MAX),
            1.0 / 18_446_744_073_709_551_615.0
        );
    }

    /// Numbers compare as Python compares them: exactly, an int against a float included.
    #[test]
    fn numbers_compare_exactly() {
        use std::cmp::Ordering::{Equal, Greater, Less};
        let two_53 = 1_i128 << 53;
        let two_127 = 2.0_f64.powi(127);
        for (a, b, want) in [
            (
                Num::Int(two_53 + 1),
                Num::Float(9_007_199_254_740_992.0),
                Some(Greater),
            ),
            (
                Num::Int(two_53),
                Num::Float(9_007_199_254_740_992.0),
                Some(Equal),
            ),
            (
                Num::Int(-two_53 - 1),
                Num::Float(-9_007_199_254_740_992.0),
                Some(Less),
            ),
            (Num::Int(3), Num::Float(2.5), Some(Greater)),
            (Num::Int(2), Num::Float(2.5), Some(Less)),
            (Num::Int(-2), Num::Float(-2.5), Some(Greater)),
            (Num::Int(-3), Num::Float(-2.5), Some(Less)),
            (Num::Int(0), Num::Float(-0.0), Some(Equal)),
            (Num::Int(i128::MAX), Num::Float(two_127), Some(Less)),
            (Num::Int(i128::MIN), Num::Float(-two_127), Some(Equal)),
            (
                Num::Int(i128::MIN),
                Num::Float(f64::NEG_INFINITY),
                Some(Greater),
            ),
            (Num::Int(i128::MAX), Num::Float(f64::INFINITY), Some(Less)),
            (Num::Int(1), Num::Float(f64::NAN), None),
            (Num::Int(two_53 + 1), Num::Int(two_53), Some(Greater)),
            (Num::Float(0.5), Num::Float(0.25), Some(Greater)),
        ] {
            assert_eq!(py_cmp(a, b), want, "{a:?} {b:?}");
            assert_eq!(
                py_cmp(b, a),
                want.map(std::cmp::Ordering::reverse),
                "{b:?} {a:?}"
            );
        }
    }

    /// `sum()`'s three paths: the C `long` fast path, exact integers and plain float addition
    /// after it overflows, and the compensated float path (integers added plainly).
    #[test]
    fn sums_follow_cpython() {
        let max = i128::from(i64::MAX);
        let sum = |values: &[Num]| py_sum(values, "test").unwrap();
        let int = |n: Num| match n {
            Num::Int(i) => i,
            Num::Float(f) => panic!("a float {f}"),
        };
        let float = |n: Num| match n {
            Num::Float(f) => f.to_bits(),
            Num::Int(i) => panic!("an int {i}"),
        };
        assert_eq!(int(sum(&[Num::Int(max), Num::Int(1), Num::Int(-max)])), 1);
        assert_eq!(
            int(sum(&[Num::Int(max), Num::Int(max), Num::Int(2)])),
            1 << 64
        );
        let (big, one) = (Num::Float(1e16), Num::Float(1.0));
        let generic = [
            Num::Int(max),
            Num::Int(1),
            Num::Int(-max),
            big,
            one,
            Num::Float(-1e16),
        ];
        assert_eq!(float(sum(&generic)), 0.0_f64.to_bits());
        let compensated = [Num::Int(1), big, one, Num::Float(-1e16)];
        assert_eq!(float(sum(&compensated)), 1.0_f64.to_bits());
        let ints_plain = [big, Num::Int(1), Num::Int(1), Num::Float(-1e16)];
        assert_eq!(float(sum(&ints_plain)), 0.0_f64.to_bits());
        assert_eq!(float(sum(&[Num::Float(-0.0)])), 0.0_f64.to_bits());
    }
}

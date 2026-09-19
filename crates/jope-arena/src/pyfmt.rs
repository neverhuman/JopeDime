//! Python-compatible float rounding and `repr`, so CSV cells written here are byte-identical to
//! the ones `csv.DictWriter` wrote from `round(x, n)` in the benchmark's `leaderboard.py`.
//!
//! Both languages round *correctly* (on the exact binary value, ties to even) and print the
//! shortest round-tripping decimal; the only differences are cosmetic (`1500` vs `1500.0`,
//! exponent spelling), handled in [`py_repr`].

/// Python's `round(value, ndigits)` for a finite `f64`.
pub fn py_round(value: f64, ndigits: usize) -> f64 {
    if !value.is_finite() {
        return value;
    }
    format!("{value:.ndigits$}").parse().unwrap_or(value)
}

/// Python's `repr(float)`.
pub fn py_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    let magnitude = value.abs();
    if magnitude != 0.0 && !(1e-4..1e16).contains(&magnitude) {
        let scientific = format!("{value:e}");
        let (mantissa, exponent) = scientific
            .split_once('e')
            .expect("Rust LowerExp always contains 'e'");
        let exponent: i32 = exponent.parse().expect("integer exponent");
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:02}", exponent.abs());
    }
    let plain = format!("{value}");
    if plain.contains('.') {
        plain
    } else {
        format!("{plain}.0")
    }
}

/// Python's `str(bool)`.
pub fn py_bool(value: bool) -> &'static str {
    if value {
        "True"
    } else {
        "False"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_matches_python() {
        assert_eq!(py_repr(py_round(2259.8123, 1)), "2259.8");
        assert_eq!(py_repr(py_round(1500.0, 1)), "1500.0");
        assert_eq!(py_repr(py_round(61.5, 3)), "61.5");
        assert_eq!(py_repr(py_round(4.6222222, 3)), "4.622");
        assert_eq!(py_repr(py_round(100.0 * 42.0 / 45.0, 1)), "93.3");
        assert_eq!(py_repr(py_round(0.25, 1)), "0.2");
        assert_eq!(py_repr(py_round(0.35, 1)), "0.3");
        assert_eq!(py_repr(py_round(2.675, 2)), "2.67");
        assert_eq!(py_repr(py_round(-0.04, 1)), "-0.0");
    }

    #[test]
    fn repr_matches_python_spelling() {
        assert_eq!(py_repr(0.0), "0.0");
        assert_eq!(py_repr(1e16), "1e+16");
        assert_eq!(py_repr(1.5e-5), "1.5e-05");
        assert_eq!(py_repr(0.0001), "0.0001");
        assert_eq!(py_repr(123456789012345.0), "123456789012345.0");
        assert_eq!(py_repr(0.028_304_828_353_688_47), "0.02830482835368847");
        assert_eq!(py_repr(f64::NAN), "nan");
        assert_eq!(py_bool(true), "True");
        assert_eq!(py_bool(false), "False");
    }
}

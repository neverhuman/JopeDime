//! Dataset registration and the versioned, target-independent matrix format.

/// A finite `f64`. Blank, non-numeric, NaN, and infinite text are `None`.
pub(crate) fn finite_number(value: &str) -> Option<f64> {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|number| number.is_finite())
}

pub mod baseline;
pub mod group_leak;
pub mod jmat;
pub mod leakage;
pub mod profile;
pub mod register;

#[cfg(test)]
mod tests {
    use super::finite_number;

    #[test]
    fn finite_number_accepts_plain_text_and_rejects_non_finite_text() {
        assert_eq!(finite_number(" 2.5 ").unwrap().to_bits(), 2.5_f64.to_bits());
        assert!(finite_number("").is_none());
        assert!(finite_number("nan").is_none());
        assert!(finite_number("inf").is_none());
        assert!(finite_number("1e309").is_none());
    }
}

/// FIT row count for a random split: the real-valued floor of
/// `rows * (1 - holdout_fraction)`.
///
/// f64 multiplication lands a few ulps under an integer for common decimals.
/// Five rows at holdout 0.8 produce `0.9999999999999998`, and plain `floor`
/// keeps zero FIT rows. When the product is within a few ulps of an integer,
/// that integer is the count. A genuine fraction, such as five rows at 0.5
/// (`2.5`), stays on the floor.
pub(super) fn fit_row_count(rows: u64, holdout_fraction: f64) -> u64 {
    let raw = (rows as f64) * (1.0 - holdout_fraction);
    let nearest = raw.round();
    let scale = raw.abs().max(1.0);
    let count = if (raw - nearest).abs() <= scale * f64::EPSILON * 32.0 {
        nearest
    } else {
        raw.floor()
    };
    count as u64
}

#[cfg(test)]
mod tests {
    use super::fit_row_count;

    #[test]
    fn five_rows_at_eight_tenths_keep_one_fit_row() {
        let raw = 5_f64 * (1.0 - 0.8);
        assert!(raw < 1.0, "{raw}");
        assert_eq!(raw.floor() as u64, 0, "{raw}");
        assert_eq!(fit_row_count(5, 0.8), 1);
    }

    #[test]
    fn a_genuine_half_stays_on_the_floor() {
        let raw = 5_f64 * (1.0 - 0.5);
        assert!(raw > 2.0 && raw < 3.0, "{raw}");
        assert_eq!(raw.floor() as u64, 2, "{raw}");
        assert_eq!(fit_row_count(5, 0.5), 2);
        assert_eq!(fit_row_count(3, 0.5), 1);
        assert_eq!(fit_row_count(7, 0.2), 5);
    }

    #[test]
    fn hundredths_match_integer_floor_through_the_row_limit() {
        for rows in [2_u64, 5, 10, 25, 100, 1_000, 100_000] {
            for percent in 1_u64..100 {
                let holdout = percent as f64 / 100.0;
                let exact = rows * (100 - percent) / 100;
                assert_eq!(
                    fit_row_count(rows, holdout),
                    exact,
                    "rows {rows} holdout {percent}%"
                );
            }
        }
    }
}

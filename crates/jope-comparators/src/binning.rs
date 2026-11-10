//! Quantile histogram binning shared by tree learners.
//!
//! Each column gets at most `max_bins` bins separated by cut points learned from the (imputed)
//! training values. A value `v` falls in bin `b` exactly when `v <= cut[b]` and `v > cut[b - 1]`,
//! so a split "bin ≤ b" is the same rule as the raw-value test `v <= cut[b]` used at prediction.

/// Largest supported bin count; codes are stored as `u8`.
pub(crate) const MAX_BINS: usize = 255;

/// `BinnedMatrix` is a struct for the binned training matrix contract in this module: the
/// column-major bin code of every training cell plus the cut points that define the bins.
#[derive(Clone, Debug)]
pub(crate) struct BinnedMatrix {
    codes: Vec<u8>,
    rows: usize,
    cuts: Vec<Vec<f64>>,
}

impl BinnedMatrix {
    /// Bin a dense row-major matrix with no missing values. `max_bins` is clamped to `2..=255`.
    pub(crate) fn build(dense: &[f64], rows: usize, cols: usize, max_bins: usize) -> Self {
        let max_bins = max_bins.clamp(2, MAX_BINS);
        let mut codes = vec![0_u8; rows * cols];
        let mut cuts = Vec::with_capacity(cols);
        let mut column = Vec::with_capacity(rows);
        for col in 0..cols {
            column.clear();
            column.extend((0..rows).map(|row| dense[row * cols + col]));
            let column_cuts = cut_points(&mut column, max_bins);
            for row in 0..rows {
                let value = dense[row * cols + col];
                // At most `max_bins - 1` cuts, so the code fits in `u8`.
                codes[col * rows + row] = column_cuts.partition_point(|cut| *cut < value) as u8;
            }
            cuts.push(column_cuts);
        }
        Self { codes, rows, cuts }
    }

    pub(crate) fn cols(&self) -> usize {
        self.cuts.len()
    }

    /// Bin codes of one column, indexed by training row.
    pub(crate) fn column(&self, col: usize) -> &[u8] {
        &self.codes[col * self.rows..(col + 1) * self.rows]
    }

    /// Number of bins in a column (`1` for a constant column).
    pub(crate) fn bin_count(&self, col: usize) -> usize {
        self.cuts[col].len() + 1
    }

    /// The raw-value threshold of the split "bin ≤ `bin`" in `col`.
    pub(crate) fn threshold(&self, col: usize, bin: usize) -> f64 {
        self.cuts[col][bin]
    }
}

fn cut_points(values: &mut Vec<f64>, max_bins: usize) -> Vec<f64> {
    values.sort_by(f64::total_cmp);
    values.dedup();
    if values.len() < 2 {
        return Vec::new();
    }
    if values.len() <= max_bins {
        return values
            .windows(2)
            .map(|pair| midpoint(pair[0], pair[1]))
            .collect();
    }
    let mut cuts: Vec<f64> = Vec::with_capacity(max_bins - 1);
    for step in 1..max_bins {
        let index = step * values.len() / max_bins;
        let cut = midpoint(values[index - 1], values[index]);
        if cuts.last().is_none_or(|last| cut > *last) {
            cuts.push(cut);
        }
    }
    cuts
}

fn midpoint(low: f64, high: f64) -> f64 {
    low + (high - low) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn few_unique_values_get_midpoint_cuts() {
        let dense = [3.0, 1.0, 2.0, 1.0];
        let binned = BinnedMatrix::build(&dense, 4, 1, 255);
        assert_eq!(binned.bin_count(0), 3);
        assert_eq!(binned.threshold(0, 0), 1.5);
        assert_eq!(binned.threshold(0, 1), 2.5);
        assert_eq!(binned.column(0), &[2, 0, 1, 0]);
    }

    #[test]
    fn constant_column_has_one_bin() {
        let binned = BinnedMatrix::build(&[4.0, 4.0, 4.0], 3, 1, 16);
        assert_eq!(binned.bin_count(0), 1);
        assert_eq!(binned.column(0), &[0, 0, 0]);
    }

    #[test]
    fn many_values_are_capped_and_consistent_with_thresholds() {
        let dense: Vec<f64> = (0..1000).map(|value| f64::from(value) * 0.37).collect();
        let binned = BinnedMatrix::build(&dense, 1000, 1, 16);
        assert_eq!(binned.bin_count(0), 16);
        for (row, value) in dense.iter().enumerate() {
            let code = usize::from(binned.column(0)[row]);
            if code > 0 {
                assert!(*value > binned.threshold(0, code - 1));
            }
            if code + 1 < binned.bin_count(0) {
                assert!(*value <= binned.threshold(0, code));
            }
        }
    }
}

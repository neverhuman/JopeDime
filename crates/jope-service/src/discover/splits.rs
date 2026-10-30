//! Deterministic row-index splits for discovery evaluation.
//!
//! Indices returned by these helpers refer to the input row order. Split membership is randomized
//! from `SHA256(seed_le_bytes || label_utf8)` and each result is sorted back into input order.

use std::collections::BTreeMap;
use std::fmt;

use sha2::{Digest, Sha256};

/// Invalid population or fold parameters for a split operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitError {
    /// The requested FIT size exceeds the number of rows.
    FitRowsExceedPopulation { fit_rows: usize, row_count: usize },
    /// A k-fold split needs at least two folds and at least one row per fold.
    InvalidFoldCount { folds: usize, row_count: usize },
}

impl fmt::Display for SplitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FitRowsExceedPopulation {
                fit_rows,
                row_count,
            } => write!(
                formatter,
                "FIT row count {fit_rows} exceeds population size {row_count}"
            ),
            Self::InvalidFoldCount { folds, row_count } => write!(
                formatter,
                "fold count {folds} must be between 2 and the row count {row_count}"
            ),
        }
    }
}

impl std::error::Error for SplitError {}

/// Randomly partitions `0..row_count` into FIT and HOLDOUT, with exactly `fit_rows` FIT rows.
///
/// The same seed and label always return the same partition. Use a distinct label for each split
/// purpose so the same run seed does not reuse a row ordering accidentally.
pub fn random_split(
    row_count: usize,
    fit_rows: usize,
    seed: u64,
    label: &str,
) -> Result<(Vec<usize>, Vec<usize>), SplitError> {
    if fit_rows > row_count {
        return Err(SplitError::FitRowsExceedPopulation {
            fit_rows,
            row_count,
        });
    }

    let mut indices: Vec<usize> = (0..row_count).collect();
    let mut stream = SeededStream::new(seed, label);
    shuffle(&mut indices, &mut stream);
    let mut fit = indices[..fit_rows].to_vec();
    let mut holdout = indices[fit_rows..].to_vec();
    fit.sort_unstable();
    holdout.sort_unstable();
    Ok((fit, holdout))
}

/// Stratifies `0..labels.len()` into FIT and HOLDOUT, allocating FIT rows proportionally by class.
///
/// Largest-remainder allocation makes the requested FIT size exact; each class allocation is
/// within one row of its ideal proportional share. Labels are grouped in their `Ord` order.
pub fn stratified_split<T: Ord>(
    labels: &[T],
    fit_rows: usize,
    seed: u64,
    label: &str,
) -> Result<(Vec<usize>, Vec<usize>), SplitError> {
    let row_count = labels.len();
    if fit_rows > row_count {
        return Err(SplitError::FitRowsExceedPopulation {
            fit_rows,
            row_count,
        });
    }

    let groups = class_groups(labels);
    let mut allocations = Vec::with_capacity(groups.len());
    let mut allocated = 0usize;
    for rows in groups.values() {
        let scaled = (rows.len() as u128) * (fit_rows as u128);
        let denominator = row_count as u128;
        let base = (scaled / denominator) as usize;
        let remainder = scaled % denominator;
        allocated += base;
        allocations.push((base, remainder));
    }

    let mut order: Vec<usize> = (0..allocations.len()).collect();
    order.sort_by(|left, right| {
        allocations[*right]
            .1
            .cmp(&allocations[*left].1)
            .then_with(|| left.cmp(right))
    });
    for class_index in order.into_iter().take(fit_rows - allocated) {
        allocations[class_index].0 += 1;
    }

    let mut fit = Vec::with_capacity(fit_rows);
    let mut holdout = Vec::with_capacity(row_count - fit_rows);
    for (class_index, rows) in groups.values().enumerate() {
        let mut shuffled = rows.clone();
        let mut stream = SeededStream::new(seed, &format!("{label}/class/{class_index}"));
        shuffle(&mut shuffled, &mut stream);
        let class_fit_rows = allocations[class_index].0;
        fit.extend_from_slice(&shuffled[..class_fit_rows]);
        holdout.extend_from_slice(&shuffled[class_fit_rows..]);
    }
    fit.sort_unstable();
    holdout.sort_unstable();
    Ok((fit, holdout))
}

/// Creates shuffled, disjoint folds covering every row in `0..row_count`.
///
/// Fold sizes differ by at most one row. Each fold is sorted into input order after assignment.
pub fn k_fold(
    row_count: usize,
    folds: usize,
    seed: u64,
    label: &str,
) -> Result<Vec<Vec<usize>>, SplitError> {
    validate_fold_count(row_count, folds)?;
    let mut indices: Vec<usize> = (0..row_count).collect();
    let mut stream = SeededStream::new(seed, label);
    shuffle(&mut indices, &mut stream);
    let mut result = vec![Vec::new(); folds];
    for (position, row) in indices.into_iter().enumerate() {
        result[position % folds].push(row);
    }
    for fold in &mut result {
        fold.sort_unstable();
    }
    Ok(result)
}

/// Creates class-stratified folds covering every row in `0..labels.len()`.
///
/// Every class contributes either `floor(class_rows / folds)` or
/// `ceil(class_rows / folds)` rows to each fold. Overall fold sizes also differ by at most one.
pub fn stratified_k_fold<T: Ord>(
    labels: &[T],
    folds: usize,
    seed: u64,
    label: &str,
) -> Result<Vec<Vec<usize>>, SplitError> {
    let row_count = labels.len();
    validate_fold_count(row_count, folds)?;
    let groups = class_groups(labels);
    let mut result = vec![Vec::new(); folds];
    let mut assigned_rows = 0usize;
    for (class_index, rows) in groups.values().enumerate() {
        let mut shuffled = rows.clone();
        let mut stream = SeededStream::new(seed, &format!("{label}/class/{class_index}"));
        shuffle(&mut shuffled, &mut stream);
        let start_fold = assigned_rows % folds;
        for (position, row) in shuffled.into_iter().enumerate() {
            result[(start_fold + position) % folds].push(row);
        }
        assigned_rows += rows.len();
    }
    for fold in &mut result {
        fold.sort_unstable();
    }
    Ok(result)
}

/// Returns a deterministic sample of at most `cap` rows, preserving their input order.
///
/// When the input is larger than `cap`, exactly `cap` row positions are selected using the
/// seeded stream. An input no larger than the cap is returned unchanged.
pub fn rung0_subsample(row_indices: &[usize], cap: usize, seed: u64, label: &str) -> Vec<usize> {
    if row_indices.len() <= cap {
        return row_indices.to_vec();
    }
    let mut positions: Vec<usize> = (0..row_indices.len()).collect();
    let mut stream = SeededStream::new(seed, label);
    shuffle(&mut positions, &mut stream);
    positions.truncate(cap);
    positions.sort_unstable();
    positions
        .into_iter()
        .map(|position| row_indices[position])
        .collect()
}

fn validate_fold_count(row_count: usize, folds: usize) -> Result<(), SplitError> {
    if folds < 2 || folds > row_count {
        return Err(SplitError::InvalidFoldCount { folds, row_count });
    }
    Ok(())
}

fn class_groups<T: Ord>(labels: &[T]) -> BTreeMap<&T, Vec<usize>> {
    let mut groups = BTreeMap::new();
    for (row, class) in labels.iter().enumerate() {
        groups.entry(class).or_insert_with(Vec::new).push(row);
    }
    groups
}

struct SeededStream {
    seed_digest: [u8; 32],
    counter: u64,
}

impl SeededStream {
    fn new(seed: u64, label: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(seed.to_le_bytes());
        hasher.update(label.as_bytes());
        Self {
            seed_digest: hasher.finalize().into(),
            counter: 0,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut hasher = Sha256::new();
        hasher.update(self.seed_digest);
        hasher.update(self.counter.to_be_bytes());
        self.counter = self.counter.wrapping_add(1);
        let digest = hasher.finalize();
        u64::from_be_bytes([
            digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
        ])
    }

    fn below(&mut self, upper_exclusive: usize) -> usize {
        let upper = upper_exclusive as u64;
        let zone = u64::MAX - u64::MAX % upper;
        loop {
            let candidate = self.next_u64();
            if candidate < zone {
                return (candidate % upper) as usize;
            }
        }
    }
}

fn shuffle<T>(items: &mut [T], stream: &mut SeededStream) {
    for index in (1..items.len()).rev() {
        let other = stream.below(index + 1);
        items.swap(index, other);
    }
}

#[cfg(test)]
mod tests {
    use super::{k_fold, random_split, rung0_subsample, stratified_k_fold, stratified_split};
    use std::collections::BTreeSet;

    #[test]
    fn seeded_random_split_is_repeatable_and_covers_every_row() {
        let first = random_split(31, 24, 17, "holdout").unwrap();
        assert_eq!(first, random_split(31, 24, 17, "holdout").unwrap());
        assert_eq!(first.0.len(), 24);
        assert_eq!(first.1.len(), 7);
        assert!(first.0.iter().all(|row| !first.1.contains(row)));
        assert_eq!(
            first
                .0
                .iter()
                .chain(&first.1)
                .copied()
                .collect::<BTreeSet<_>>(),
            (0..31).collect()
        );
    }

    #[test]
    fn stratified_split_is_repeatable_and_preserves_class_shares() {
        let labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2];
        let first = stratified_split(&labels, 12, 29, "holdout").unwrap();
        assert_eq!(first, stratified_split(&labels, 12, 29, "holdout").unwrap());
        assert_eq!(first.0.len(), 12);
        for class in 0..=2 {
            let total = labels.iter().filter(|label| **label == class).count();
            let fit = first.0.iter().filter(|row| labels[**row] == class).count();
            let ideal = (total * 12) as f64 / labels.len() as f64;
            assert!((fit as f64 - ideal).abs() < 1.0);
        }
    }

    #[test]
    fn plain_k_fold_is_repeatable_disjoint_and_complete() {
        let first = k_fold(37, 5, 41, "rung-1").unwrap();
        assert_eq!(first, k_fold(37, 5, 41, "rung-1").unwrap());
        assert_eq!(
            first.iter().map(Vec::len).max().unwrap() - first.iter().map(Vec::len).min().unwrap(),
            1
        );
        assert_partition(&first, 37);
    }

    #[test]
    fn stratified_k_fold_is_repeatable_and_balances_each_class() {
        let labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2];
        let first = stratified_k_fold(&labels, 4, 53, "rung-0").unwrap();
        assert_eq!(first, stratified_k_fold(&labels, 4, 53, "rung-0").unwrap());
        assert_partition(&first, labels.len());
        for class in 0..=2 {
            let counts: Vec<usize> = first
                .iter()
                .map(|fold| fold.iter().filter(|row| labels[**row] == class).count())
                .collect();
            assert!(counts.iter().max().unwrap() - counts.iter().min().unwrap() <= 1);
        }
    }

    #[test]
    fn rung0_subsample_is_repeatable_and_exactly_capped() {
        let rows: Vec<usize> = (100..180).collect();
        let first = rung0_subsample(&rows, 23, 67, "rung-0");
        assert_eq!(first, rung0_subsample(&rows, 23, 67, "rung-0"));
        assert_eq!(first.len(), 23);
        assert!(first.iter().all(|row| rows.contains(row)));
        assert_eq!(rung0_subsample(&rows[..10], 23, 67, "rung-0"), rows[..10]);
    }

    fn assert_partition(folds: &[Vec<usize>], row_count: usize) {
        let all: Vec<usize> = folds.iter().flatten().copied().collect();
        assert_eq!(all.len(), row_count);
        assert_eq!(
            all.iter().copied().collect::<BTreeSet<_>>().len(),
            row_count
        );
        assert_eq!(
            all.into_iter().collect::<BTreeSet<_>>(),
            (0..row_count).collect()
        );
    }
}

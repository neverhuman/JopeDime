//! Deterministic FIT / HOLDOUT row partitioning and bounded `<= 1,024`-row FIT sampling.
//!
//! All landmark conditioning and candidate fitting must read strictly from `fit_rows`;
//! `holdout_rows` remain untouched until held-out scoring.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{PrepError, PrepResult};

/// Maximum FIT rows admitted into `fit_transform_and_extract_bundle_high`.
pub const MAX_LANDMARK_FIT_ROWS: usize = 1_024;

/// Default holdout fraction (`0.20`) when callers request standard Carbon preparation.
pub const DEFAULT_HOLDOUT_FRACTION: f64 = 0.20;

/// `SplitPlan` is a struct for the deterministic row-partition contract in `jope-prep`: it binds
/// sorted, non-empty, disjoint `fit_rows` and `holdout_rows`, a `<= 1,024`-row FIT conditioning
/// sample drawn strictly from `fit_rows`, and SHA-256 digests of both row-index sequences.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitPlan {
    total_rows: usize,
    seed: u64,
    fit_rows: Vec<usize>,
    holdout_rows: Vec<usize>,
    sampled_fit_rows: Vec<usize>,
    fit_row_digest: [u8; 32],
    holdout_row_digest: [u8; 32],
}

impl SplitPlan {
    /// Construct a deterministic `FIT` / `HOLDOUT` partition of `0..total_rows`.
    pub fn deterministic(total_rows: usize, seed: u64, holdout_fraction: f64) -> PrepResult<Self> {
        if total_rows < 2 {
            return Err(PrepError::InsufficientRows { rows: total_rows });
        }
        if !holdout_fraction.is_finite() || holdout_fraction <= 0.0 || holdout_fraction >= 1.0 {
            return Err(PrepError::InvalidHoldoutFraction {
                fraction: holdout_fraction,
            });
        }

        let desired_holdout = ((total_rows as f64) * holdout_fraction).round() as usize;
        let holdout_count = desired_holdout.clamp(1, total_rows - 1);

        let mut indices: Vec<usize> = (0..total_rows).collect();
        let mut rng = SplitMix64::new(seed ^ 0x4A4F_5045_5052_4550);
        for i in (1..total_rows).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            indices.swap(i, j);
        }

        let mut holdout_rows = indices[..holdout_count].to_vec();
        let mut fit_rows = indices[holdout_count..].to_vec();
        holdout_rows.sort_unstable();
        fit_rows.sort_unstable();

        let sampled_fit_rows = sample_fit_indices(&fit_rows, seed);
        let fit_row_digest = hash_index_slice("jope.prep.fit_rows.v1", &fit_rows);
        let holdout_row_digest = hash_index_slice("jope.prep.holdout_rows.v1", &holdout_rows);

        Ok(Self {
            total_rows,
            seed,
            fit_rows,
            holdout_rows,
            sampled_fit_rows,
            fit_row_digest,
            holdout_row_digest,
        })
    }

    /// Construct a forward chronological `FIT` / `HOLDOUT` partition preserving row order:
    /// `0..(total_rows - holdout_count)` form `FIT`, and the trailing `holdout_count` rows form
    /// `HOLDOUT` (required for order-dependent objectives such as `mase`, `sharpe`, and `prediction_turnover`).
    pub fn chronological(total_rows: usize, seed: u64, holdout_fraction: f64) -> PrepResult<Self> {
        if total_rows < 2 {
            return Err(PrepError::InsufficientRows { rows: total_rows });
        }
        if !holdout_fraction.is_finite() || holdout_fraction <= 0.0 || holdout_fraction >= 1.0 {
            return Err(PrepError::InvalidHoldoutFraction {
                fraction: holdout_fraction,
            });
        }

        let desired_holdout = ((total_rows as f64) * holdout_fraction).round() as usize;
        let holdout_count = desired_holdout.clamp(1, total_rows - 1);
        let fit_end = total_rows - holdout_count;

        let fit_rows: Vec<usize> = (0..fit_end).collect();
        let holdout_rows: Vec<usize> = (fit_end..total_rows).collect();
        let sampled_fit_rows = sample_fit_indices(&fit_rows, seed);
        let fit_row_digest = hash_index_slice("jope.prep.fit_rows.v1", &fit_rows);
        let holdout_row_digest = hash_index_slice("jope.prep.holdout_rows.v1", &holdout_rows);

        Ok(Self {
            total_rows,
            seed,
            fit_rows,
            holdout_rows,
            sampled_fit_rows,
            fit_row_digest,
            holdout_row_digest,
        })
    }

    /// Construct a deterministic class-stratified `FIT` / `HOLDOUT` partition over a binary target
    /// (`0` vs non-zero `1`), guaranteeing both classes appear in `FIT` and `HOLDOUT` whenever
    /// each class has at least `2` rows and `total_holdout >= 2` / `total_fit >= 2`.
    pub fn stratified_binary(
        y_binary: &[u8],
        seed: u64,
        holdout_fraction: f64,
    ) -> PrepResult<Self> {
        let total_rows = y_binary.len();
        if total_rows < 2 {
            return Err(PrepError::InsufficientRows { rows: total_rows });
        }
        if !holdout_fraction.is_finite() || holdout_fraction <= 0.0 || holdout_fraction >= 1.0 {
            return Err(PrepError::InvalidHoldoutFraction {
                fraction: holdout_fraction,
            });
        }

        let mut idx_0: Vec<usize> = Vec::new();
        let mut idx_1: Vec<usize> = Vec::new();
        for (i, &lbl) in y_binary.iter().enumerate() {
            if lbl == 0 {
                idx_0.push(i);
            } else {
                idx_1.push(i);
            }
        }
        if idx_0.is_empty() || idx_1.is_empty() {
            return Self::deterministic(total_rows, seed, holdout_fraction);
        }

        let mut rng = SplitMix64::new(seed ^ 0x4A4F_5045_4249_4E31);
        for pool in [&mut idx_0, &mut idx_1] {
            for i in (1..pool.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                pool.swap(i, j);
            }
        }

        let desired_holdout = ((total_rows as f64) * holdout_fraction).round() as usize;
        let total_holdout = desired_holdout.clamp(1, total_rows - 1);
        let total_fit = total_rows - total_holdout;
        let n0 = idx_0.len();
        let n1 = idx_1.len();

        // Allocate holdout counts `(h0, h1)` preserving `h0 + h1 == total_holdout` while keeping
        // at least 1 row of each class in `FIT` (and in `HOLDOUT` when `n_k >= 2` and `total_holdout >= 2`).
        let min_h0 = if n0 >= 2 && total_holdout >= 2 && total_fit >= 2 {
            1
        } else {
            0
        }
        .max(
            total_holdout.saturating_sub(n1.saturating_sub(if n1 >= 1 && total_fit >= 2 {
                1
            } else {
                0
            })),
        );
        let max_h0 = n0
            .saturating_sub(if n0 >= 1 && total_fit >= 2 { 1 } else { 0 })
            .min(total_holdout.saturating_sub(
                if n1 >= 2 && total_holdout >= 2 && total_fit >= 2 {
                    1
                } else {
                    0
                },
            ));
        let prop_h0 = ((n0 as f64) * holdout_fraction).round() as usize;
        let h0 = if min_h0 <= max_h0 {
            prop_h0.clamp(min_h0, max_h0)
        } else {
            prop_h0.min(n0).min(total_holdout)
        };
        let h1 = total_holdout.saturating_sub(h0).min(n1);
        let h0 = total_holdout.saturating_sub(h1).min(n0);

        let mut holdout_rows = Vec::with_capacity(h0 + h1);
        holdout_rows.extend_from_slice(&idx_0[..h0]);
        holdout_rows.extend_from_slice(&idx_1[..h1]);
        holdout_rows.sort_unstable();

        let mut fit_rows = Vec::with_capacity(total_rows - holdout_rows.len());
        fit_rows.extend_from_slice(&idx_0[h0..]);
        fit_rows.extend_from_slice(&idx_1[h1..]);
        fit_rows.sort_unstable();

        let sampled_fit_rows = sample_fit_indices(&fit_rows, seed);
        let fit_row_digest = hash_index_slice("jope.prep.fit_rows.v1", &fit_rows);
        let holdout_row_digest = hash_index_slice("jope.prep.holdout_rows.v1", &holdout_rows);

        Ok(Self {
            total_rows,
            seed,
            fit_rows,
            holdout_rows,
            sampled_fit_rows,
            fit_row_digest,
            holdout_row_digest,
        })
    }

    /// Total row count of the unsplit source table.
    pub fn total_rows(&self) -> usize {
        self.total_rows
    }

    /// Deterministic seed used to construct this split.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Sorted FIT row indices (strictly disjoint from `holdout_rows`).
    pub fn fit_rows(&self) -> &[usize] {
        &self.fit_rows
    }

    /// Sorted HOLDOUT row indices (never admitted into landmark conditioning).
    pub fn holdout_rows(&self) -> &[usize] {
        &self.holdout_rows
    }

    /// Sorted FIT row indices (`<= 1,024`) selected for high-resolution landmark extraction.
    pub fn sampled_fit_rows(&self) -> &[usize] {
        &self.sampled_fit_rows
    }

    /// Domain-separated SHA-256 digest of `fit_rows`.
    pub fn fit_row_digest(&self) -> &[u8; 32] {
        &self.fit_row_digest
    }

    /// Domain-separated SHA-256 digest of `holdout_rows`.
    pub fn holdout_row_digest(&self) -> &[u8; 32] {
        &self.holdout_row_digest
    }

    /// Validate that all indices in `self` are strictly ordered, disjoint, and bounded by `expected_total_rows`.
    pub fn validate_for_table(&self, expected_total_rows: usize) -> PrepResult<()> {
        if self.total_rows != expected_total_rows || expected_total_rows < 2 {
            return Err(PrepError::ShapeMismatch {
                expected: self.total_rows,
                actual: expected_total_rows,
            });
        }
        if self.fit_rows.is_empty() || self.holdout_rows.is_empty() {
            return Err(PrepError::InsufficientRows {
                rows: expected_total_rows,
            });
        }
        verify_strictly_sorted(&self.fit_rows, expected_total_rows)?;
        verify_strictly_sorted(&self.holdout_rows, expected_total_rows)?;
        verify_strictly_sorted(&self.sampled_fit_rows, expected_total_rows)?;
        if self.sampled_fit_rows.is_empty() || self.sampled_fit_rows.len() > MAX_LANDMARK_FIT_ROWS {
            return Err(PrepError::ShapeMismatch {
                expected: MAX_LANDMARK_FIT_ROWS,
                actual: self.sampled_fit_rows.len(),
            });
        }
        // Verify sampled_fit_rows is a subset of fit_rows.
        for &idx in &self.sampled_fit_rows {
            if self.fit_rows.binary_search(&idx).is_err() {
                return Err(PrepError::InvalidSplitIndex {
                    index: idx,
                    total_rows: expected_total_rows,
                });
            }
        }
        Ok(())
    }
}

fn verify_strictly_sorted(indices: &[usize], total_rows: usize) -> PrepResult<()> {
    for (pos, &idx) in indices.iter().enumerate() {
        if idx >= total_rows || (pos > 0 && indices[pos - 1] >= idx) {
            return Err(PrepError::InvalidSplitIndex {
                index: idx,
                total_rows,
            });
        }
    }
    Ok(())
}

fn sample_fit_indices(fit_rows: &[usize], seed: u64) -> Vec<usize> {
    if fit_rows.len() <= MAX_LANDMARK_FIT_ROWS {
        return fit_rows.to_vec();
    }
    let mut pool = fit_rows.to_vec();
    let mut rng = SplitMix64::new(seed ^ 0x4C4D_4B5F_5341_4D50);
    for i in (0..MAX_LANDMARK_FIT_ROWS).rev() {
        let span = pool.len() - i;
        let pick = i + ((rng.next_u64() as usize) % span);
        pool.swap(i, pick);
    }
    pool.truncate(MAX_LANDMARK_FIT_ROWS);
    pool.sort_unstable();
    pool
}

fn hash_index_slice(domain: &str, rows: &[usize]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0u8]);
    hasher.update((rows.len() as u64).to_le_bytes());
    for &idx in rows {
        hasher.update((idx as u64).to_le_bytes());
    }
    hasher.finalize().into()
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

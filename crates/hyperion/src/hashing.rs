//! Stable, deterministic hashing for run receipts and cache keys.
//!
//! We deliberately avoid `std::DefaultHasher` (not stable across toolchains). SplitMix64 mixing over
//! quantized floats matches the project's ChaCha8 determinism discipline: the same content always
//! hashes to the same value, on any platform, across versions.
use crate::data::Matrix;

/// SplitMix64 finalizer — strong avalanche, fully deterministic.
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[inline]
fn combine(acc: u64, x: u64) -> u64 {
    // order-sensitive fold (like a stream hash)
    mix64(acc.rotate_left(5) ^ x.wrapping_add(0x9e37_79b9_7f4a_7c15))
}

/// Quantize a float so tiny FP noise does not change the hash; NaN/inf -> a fixed sentinel.
#[inline]
fn q(v: f32) -> u64 {
    if v.is_finite() {
        (((v as f64) * 1e6).round() as i64) as u64
    } else {
        0xDEAD_BEEF_DEAD_BEEF
    }
}

fn hash_str(acc: u64, s: &str) -> u64 {
    let mut a = acc;
    for b in s.bytes() {
        a = combine(a, b as u64);
    }
    a
}

/// Stable hash of dataset content: shape + names + (subsampled) quantized columns + target.
/// Columns are strided for very large matrices so this stays cheap while remaining content-sensitive.
pub fn dataset_hash(x: &Matrix, y: &[f32], names: &[String]) -> u64 {
    let mut a = 0x1234_5678_9abc_def0u64;
    a = combine(a, x.n as u64);
    a = combine(a, x.c as u64);
    for nm in names {
        a = hash_str(a, nm);
    }
    for j in 0..x.c {
        let col = x.col(j);
        let stride = (col.len() / 4096).max(1);
        let mut i = 0;
        while i < col.len() {
            a = combine(a, q(col[i]));
            i += stride;
        }
    }
    let ystride = (y.len() / 4096).max(1);
    let mut i = 0;
    while i < y.len() {
        a = combine(a, q(y[i]));
        i += ystride;
    }
    a
}

/// Stable hash of a fold partition: policy tag + seed + active set + per-fold validation membership.
pub fn fold_hash(
    policy_tag: u64,
    seed: u64,
    active_sorted: &[usize],
    folds: &[(Vec<usize>, Vec<usize>)],
) -> u64 {
    let mut a = 0x0f0f_0f0f_0f0f_0f0fu64;
    a = combine(a, policy_tag);
    a = combine(a, seed);
    a = combine(a, active_sorted.len() as u64);
    for &r in active_sorted {
        a = combine(a, r as u64);
    }
    for (fi, (_, va)) in folds.iter().enumerate() {
        a = combine(a, fi as u64);
        a = combine(a, va.len() as u64);
        for &r in va {
            a = combine(a, r as u64);
        }
    }
    a
}

/// Stable hash of one row's quantized feature vector (for cross-fold duplicate detection).
pub fn row_hash(x: &Matrix, row: usize) -> u64 {
    let mut a = 0xa5a5_a5a5_a5a5_a5a5u64;
    for j in 0..x.c {
        a = combine(a, q(x.at(row, j)));
    }
    a
}

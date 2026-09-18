//! The frozen jope-dime encoder sampling recipes.
//!
//! * Foundation (`parallel/tools/export_foundation.py`, schema
//!   `parallel.foundation_sampling.v1`): one `default_rng(20260707)`
//!   permutation per cut length split into context and query rows.
//! * Target (`parallel/tools/export_target.py`, schema
//!   `parallel.target_sampling.v1`): `target_maker.data.stable_seed`,
//!   `target_maker.data.sample_indices` and
//!   `target_maker.encode._window_indices` with seed 17, window 32, 2 views.

use std::fmt::Display;

use crate::blake2b::blake2b;
use crate::generator::{default_rng, Generator, SampleError};

pub const FOUNDATION_SCHEMA: &str = "parallel.foundation_sampling.v1";
pub const FOUNDATION_SEED: u64 = 20_260_707;
pub const TARGET_SCHEMA: &str = "parallel.target_sampling.v1";
pub const TARGET_SEED: u64 = 17;
pub const TARGET_WINDOW: usize = 32;
pub const TARGET_VIEWS: usize = 2;
/// The numpy release whose semantics this crate reproduces.
pub const NUMPY_VERSION: &str = "1.26.4";

/// A foundation row-index recipe for one cut length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoundationRecipe {
    pub context: Vec<u64>,
    pub query: Vec<u64>,
}

/// Foundation recipe for a cut with `n` training rows.
pub fn foundation_recipe(n: usize) -> FoundationRecipe {
    let idx = default_rng(u128::from(FOUNDATION_SEED)).permutation(n);
    let n_i = n as i64;
    let nc = 768.min(8.max(3 * n_i / 4));
    let nq = 256.min(n_i - nc);
    let clamp = |v: i64| v.clamp(0, n_i) as usize;
    let context = idx[..clamp(nc)].to_vec();
    let query = if nq < 4 {
        idx[clamp(n_i - 8)..].to_vec()
    } else {
        idx[clamp(nc)..clamp(nc + nq)].to_vec()
    };
    FoundationRecipe { context, query }
}

/// `target_maker.data.stable_seed(*parts)`: the low 63 bits of an 8-byte
/// BLAKE2b digest of the `|`-joined `str()` of each part.
pub fn stable_seed(parts: &[&dyn Display]) -> u64 {
    let text = parts
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join("|");
    let digest = blake2b(text.as_bytes(), 8);
    u64::from_be_bytes(digest.try_into().expect("8-byte digest")) & 0x7fff_ffff_ffff_ffff
}

/// `target_maker.data.sample_indices(length, window_size, rng)[0]`.
pub fn sample_indices(
    length: usize,
    window_size: usize,
    rng: &mut Generator,
) -> Result<Vec<u64>, SampleError> {
    if length == 0 {
        return Err(SampleError::EmptyPopulation);
    }
    rng.choice(length as u64, window_size, length < window_size)
}

/// `target_maker.encode._window_indices(length, window_size, stride,
/// max_windows, seed)`.
pub fn window_indices(
    length: usize,
    window_size: usize,
    stride: usize,
    max_windows: Option<usize>,
    seed: u64,
) -> Result<Vec<Vec<u64>>, SampleError> {
    if length == 0 {
        return Err(SampleError::EmptyPopulation);
    }
    let mut rng = default_rng(u128::from(stable_seed(&[
        &"encode",
        &seed,
        &length,
        &window_size,
        &stride,
    ])));
    if length < window_size {
        return Ok(vec![rng.choice(length as u64, window_size, true)?]);
    }
    let shuffled = rng.permutation(length);
    let stride = stride.max(1);
    let mut windows = Vec::new();
    let mut start = 0;
    while start + window_size <= length {
        windows.push(shuffled[start..start + window_size].to_vec());
        if max_windows.is_some_and(|max| windows.len() >= max) {
            break;
        }
        start += stride;
    }
    if windows.is_empty() {
        windows.push(shuffled[..window_size].to_vec());
    }
    Ok(windows)
}

/// A target row-index recipe for one `(task, length)` pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetRecipe {
    pub classical: Vec<Vec<u64>>,
    pub neural: Vec<Vec<u64>>,
}

/// The `sampling.json` key of a target recipe.
pub fn target_key(task: u32, n: usize) -> String {
    format!("{task}:{n}")
}

/// Target recipe for `task` (0 binary, 1 regression) and `n` rows.
pub fn target_recipe(task: u32, n: usize) -> Result<TargetRecipe, SampleError> {
    let classical = (0..TARGET_VIEWS)
        .map(|i| {
            let seed = stable_seed(&[
                &"fusion-signature-inference",
                &TARGET_SEED,
                &task,
                &n,
                &i,
                &TARGET_WINDOW,
            ]);
            sample_indices(n, TARGET_WINDOW, &mut default_rng(u128::from(seed)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let windows = window_indices(
        n,
        TARGET_WINDOW,
        TARGET_WINDOW,
        Some(TARGET_VIEWS),
        TARGET_SEED,
    )?;
    let neural = windows
        .iter()
        .chain(windows.iter())
        .take(TARGET_VIEWS)
        .cloned()
        .collect();
    Ok(TargetRecipe { classical, neural })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_seed_joins_display_parts() {
        let joined = stable_seed(&[&"encode", &17, &91, &32, &32]);
        let digest = blake2b(b"encode|17|91|32|32", 8);
        assert_eq!(
            joined,
            u64::from_be_bytes(digest.try_into().unwrap()) & (u64::MAX >> 1)
        );
    }

    #[test]
    fn short_cuts_query_the_permutation_tail() {
        let recipe = foundation_recipe(9);
        assert_eq!(recipe.context.len(), 8);
        assert_eq!(recipe.query.len(), 8);
        let tiny = foundation_recipe(5);
        assert_eq!(tiny.context.len(), 5);
        assert_eq!(tiny.query, tiny.context);
    }

    #[test]
    fn foundation_split_sizes() {
        for (n, nc, nq) in [(96, 72, 24), (800, 600, 200), (2000, 768, 256)] {
            let recipe = foundation_recipe(n);
            assert_eq!((recipe.context.len(), recipe.query.len()), (nc, nq));
        }
    }

    #[test]
    fn short_targets_repeat_the_single_neural_window() {
        let recipe = target_recipe(0, 7).unwrap();
        assert_eq!(recipe.neural[0], recipe.neural[1]);
        assert!(recipe.classical.iter().all(|w| w.len() == TARGET_WINDOW));
        assert_eq!(target_recipe(1, 0), Err(SampleError::EmptyPopulation));
    }
}

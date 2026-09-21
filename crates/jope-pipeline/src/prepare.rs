//! Stage 2: cut construction, owned by this crate.
//!
//! A *cut* is one fold of a fixed [`FOLDS`]-way partition of the dataset's
//! training rows: the fold itself is held out and the remaining folds are the
//! training rows. Cut `i` holds out fold `i`, so `--cuts k` searches the first
//! `k` folds and every cut has a disjoint held-out set.
//!
//! Fold membership comes from [`feat_math::folds::fold_of`], the workspace's own
//! deterministic FNV-1a assignment, keyed on `<dataset hash>:<seed>` so the
//! partition depends on the dataset and the `--seed` and on nothing else — not
//! on row order, not on how many cuts were asked for, not on the search.
//!
//! This stage deliberately does **not** call `jope_parallel::prepare::from_rows`.
//! That builds a `parallel.prepared.v1` view for the engine's GP search, which
//! this pipeline no longer drives (see [`crate::search`]); and the engine's
//! `crates/jope-engine/src/prepare.rs` has its bytes hashed into
//! `preprocessing_source_sha256` by `eval_registry/query.rs` via `include_bytes!`
//! — three producers and one verifier — so extending it to serve this pipeline
//! would silently invalidate every receipt the engine has already issued.
//! Nothing here touches that file.
//!
//! No scaling, centring or imputation happens here. The rows are handed to
//! `feat_math::fit` exactly as the dataset committed them, and every genome in
//! the search grammar carries its own preprocessing stages. That is why the
//! RMSE the search reports is already in the dataset's own target units.

use anyhow::{ensure, Context, Result};
use feat_math::{folds::fold_of, Rows};
use serde::Serialize;
use std::path::Path;

use crate::sha256_hex;

/// The partition every cut is one fold of.
pub const FOLDS: usize = 5;

/// One fold of the partition: fit on `train`, score on `hold`, never the reverse.
#[derive(Clone, Debug)]
pub struct Cut {
    pub index: usize,
    pub train_x: Rows,
    pub train_y: Vec<f64>,
    pub hold_x: Rows,
    pub hold_y: Vec<f64>,
    /// The dataset row numbers held out here, for the evidence record.
    pub hold_indices: Vec<usize>,
}

impl Cut {
    pub fn train_rows(&self) -> usize {
        self.train_x.len()
    }

    pub fn hold_rows(&self) -> usize {
        self.hold_x.len()
    }
}

/// What the stage published, plus the sha256 of `prepared.json` — the identity
/// every search, observation and receipt refers back to.
#[derive(Clone, Debug)]
pub struct Preparation {
    pub cuts: Vec<Cut>,
    pub seed: u64,
    pub rows: usize,
    pub sha256: String,
}

#[derive(Serialize)]
struct PreparedCutRecord {
    cut: usize,
    fold: usize,
    train_rows: usize,
    hold_rows: usize,
    hold_indices: Vec<usize>,
}

#[derive(Serialize)]
struct PreparedRecord {
    schema: &'static str,
    dataset: String,
    seed: u64,
    folds: usize,
    rows: usize,
    columns: usize,
    fold_key: String,
    cuts: Vec<PreparedCutRecord>,
}

/// The schema of the `prepared.json` this stage writes.
pub const PREPARED_SCHEMA_ID: &str = "jope.pipeline-prepared/v1";

/// Partition `rows` and materialise `<directory>/prepared.json`.
///
/// `rows` are complete observations, `features..., target`, as
/// [`crate::dataset::training_rows`] loaded them.
pub fn build(
    rows: &[Vec<f64>],
    dataset: &str,
    seed: u64,
    cuts: usize,
    directory: &Path,
) -> Result<Preparation> {
    ensure!(
        (1..=FOLDS).contains(&cuts),
        "--cuts must be 1..={FOLDS}, got {cuts}"
    );
    ensure!(!rows.is_empty(), "the dataset has no training rows");
    let columns = rows[0].len();
    ensure!(
        columns >= 2,
        "{columns} column(s); features and a target are required"
    );

    // One fold number per row, from the dataset and the seed alone.
    let fold_key = format!("{dataset}:{seed}");
    let assignment: Vec<usize> = (0..rows.len())
        .map(|row| fold_of(&fold_key, row, FOLDS))
        .collect();

    let mut built = Vec::with_capacity(cuts);
    let mut records = Vec::with_capacity(cuts);
    for index in 0..cuts {
        let (mut train_x, mut train_y) = (Rows::new(), Vec::new());
        let (mut hold_x, mut hold_y) = (Rows::new(), Vec::new());
        let mut hold_indices = Vec::new();
        for (row_index, row) in rows.iter().enumerate() {
            ensure!(
                row.len() == columns,
                "row {row_index}: {} columns, the first row has {columns}",
                row.len()
            );
            let (features, target) = row.split_at(columns - 1);
            if assignment[row_index] == index {
                hold_x.push(features.to_vec());
                hold_y.push(target[0]);
                hold_indices.push(row_index);
            } else {
                train_x.push(features.to_vec());
                train_y.push(target[0]);
            }
        }
        // A cut that cannot be fitted or cannot be scored is not a cut. R^2 is
        // undefined on a constant held-out target, so that is refused here
        // rather than surfacing later as a NaN every candidate "fails" on.
        ensure!(
            train_x.len() >= 2,
            "cut {index}: {} training row(s); at least 2 are required",
            train_x.len()
        );
        ensure!(
            hold_x.len() >= 2,
            "cut {index}: {} held-out row(s); at least 2 are required",
            hold_x.len()
        );
        ensure!(
            hold_y.iter().any(|value| *value != hold_y[0]),
            "cut {index}: the held-out target is constant ({}); R^2 is undefined on it",
            hold_y[0]
        );
        records.push(PreparedCutRecord {
            cut: index,
            fold: index,
            train_rows: train_x.len(),
            hold_rows: hold_x.len(),
            hold_indices: hold_indices.clone(),
        });
        built.push(Cut {
            index,
            train_x,
            train_y,
            hold_x,
            hold_y,
            hold_indices,
        });
    }

    let record = PreparedRecord {
        schema: PREPARED_SCHEMA_ID,
        dataset: dataset.to_owned(),
        seed,
        folds: FOLDS,
        rows: rows.len(),
        columns,
        fold_key,
        cuts: records,
    };
    std::fs::create_dir_all(directory)
        .with_context(|| format!("create {}", directory.display()))?;
    let path = directory.join("prepared.json");
    let bytes = serde_json::to_vec_pretty(&record)?;
    std::fs::write(&path, &bytes).with_context(|| format!("write {}", path.display()))?;
    Ok(Preparation {
        cuts: built,
        seed,
        rows: rows.len(),
        sha256: sha256_hex(&bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `y = 2*a + b`, so the target is not constant in any fold.
    fn rows(n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                let (a, b) = (i as f64, (i % 7) as f64);
                vec![a, b, 2.0 * a + b]
            })
            .collect()
    }

    #[test]
    fn a_cut_never_shares_a_row_between_its_fit_and_its_score() {
        let temp = tempfile::tempdir().unwrap();
        let prepared = build(&rows(120), "dataset", 11, FOLDS, temp.path()).unwrap();
        assert_eq!(prepared.cuts.len(), FOLDS);
        for cut in &prepared.cuts {
            assert_eq!(cut.train_rows() + cut.hold_rows(), 120);
            assert!(cut.hold_rows() > 0 && cut.train_rows() > 0);
            // Every held-out feature vector is absent from the training rows.
            for held in &cut.hold_x {
                assert!(
                    !cut.train_x.contains(held),
                    "cut {}: a held-out row is in the fit",
                    cut.index
                );
            }
        }
    }

    #[test]
    fn the_held_out_folds_are_disjoint_and_cover_the_dataset() {
        let temp = tempfile::tempdir().unwrap();
        let prepared = build(&rows(120), "dataset", 11, FOLDS, temp.path()).unwrap();
        let mut seen: Vec<usize> = prepared
            .cuts
            .iter()
            .flat_map(|c| c.hold_indices.clone())
            .collect();
        let total = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), total, "a row is held out by two cuts");
        assert_eq!(seen, (0..120).collect::<Vec<_>>());
    }

    #[test]
    fn the_partition_is_a_function_of_the_dataset_and_the_seed_only() {
        let temp = tempfile::tempdir().unwrap();
        let a = build(&rows(120), "dataset", 11, 2, &temp.path().join("a")).unwrap();
        let b = build(&rows(120), "dataset", 11, 2, &temp.path().join("b")).unwrap();
        assert_eq!(a.sha256, b.sha256);
        let other_seed = build(&rows(120), "dataset", 12, 2, &temp.path().join("c")).unwrap();
        let other_data = build(&rows(120), "elsewhere", 11, 2, &temp.path().join("d")).unwrap();
        assert_ne!(a.sha256, other_seed.sha256);
        assert_ne!(a.sha256, other_data.sha256);
        assert_ne!(a.cuts[0].hold_indices, other_seed.cuts[0].hold_indices);
        // Asking for fewer cuts does not move any row between folds.
        let all = build(&rows(120), "dataset", 11, FOLDS, &temp.path().join("e")).unwrap();
        for (few, many) in a.cuts.iter().zip(all.cuts.iter()) {
            assert_eq!(few.hold_indices, many.hold_indices);
        }
    }

    #[test]
    fn cut_counts_are_bounded_by_the_partition() {
        let temp = tempfile::tempdir().unwrap();
        assert!(build(&rows(60), "d", 1, 0, temp.path()).is_err());
        assert!(build(&rows(60), "d", 1, FOLDS + 1, temp.path()).is_err());
    }

    #[test]
    fn a_constant_held_out_target_is_refused_rather_than_scored_as_nan() {
        let temp = tempfile::tempdir().unwrap();
        let flat: Vec<Vec<f64>> = (0..60).map(|i| vec![i as f64, 1.0, 5.0]).collect();
        let error = format!("{:#}", build(&flat, "d", 1, 1, temp.path()).unwrap_err());
        assert!(error.contains("constant"), "{error}");
    }
}

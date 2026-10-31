//! Cuts: which training rows are held out, and a stable identity for each cut.
//!
//! Fold membership is `feat_math::fold_of(key, row, k)` with `key = "<dataset_hash>:<seed>"`,
//! exactly what the product pipeline's `jope_pipeline::prepare::build` uses, so a cut here and a
//! cut there are the same rows. What the pipeline lacks is an identity per cut;
//! [`crate::cuts::Cut::cut_id`] is the SHA-256
//! of a canonical JSON record naming the dataset, the training bytes, the key, `k`, the fold and
//! the held-out indices, so two runs that hold out the same rows of the same bytes agree, and any
//! change to any of those changes the id.

use anyhow::{ensure, Result};
use feat_math::fold_of;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Schema id of the record a cut id hashes.
pub const CUT_SCHEMA: &str = "jope.cut/v1";

/// How rows are assigned to folds.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FoldPlan {
    /// `"<dataset_hash>:<seed>"`.
    pub key: String,
    /// The seed inside `key`.
    pub seed: u64,
    /// Number of folds.
    pub k: usize,
}

impl FoldPlan {
    /// The plan the product pipeline would use for this dataset and seed.
    pub fn new(dataset_hash: &str, seed: u64, k: usize) -> Self {
        Self {
            key: format!("{dataset_hash}:{seed}"),
            seed,
            k,
        }
    }
}

/// One cut: fold `fold` held out, the rest trained on.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Cut {
    /// Held-out fold index.
    pub fold: usize,
    /// [`CUT_SCHEMA`] identity.
    pub cut_id: String,
    /// Training row indices, ascending.
    pub train: Vec<usize>,
    /// Held-out row indices, ascending.
    pub hold: Vec<usize>,
}

#[derive(Serialize)]
struct CutIdentity<'a> {
    schema: &'static str,
    dataset_hash: &'a str,
    train_sha256: &'a str,
    key: &'a str,
    seed: u64,
    k: usize,
    fold: usize,
    rows: usize,
    hold_sha256: String,
}

/// SHA-256 of `serde_json::to_vec(value)` (field order as declared).
pub fn canonical_sha256<T: Serialize>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn indices_sha256(indices: &[usize]) -> String {
    let mut hasher = Sha256::new();
    for index in indices {
        hasher.update((*index as u64).to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Every cut of `plan` over `rows` training rows. Fails when a fold would be empty or leave
/// fewer than eight training rows.
pub fn build(
    plan: &FoldPlan,
    dataset_hash: &str,
    train_sha256: &str,
    rows: usize,
) -> Result<Vec<Cut>> {
    ensure!(plan.k >= 2, "a fold plan needs at least two folds");
    let mut cuts = Vec::with_capacity(plan.k);
    for fold in 0..plan.k {
        let mut train = Vec::new();
        let mut hold = Vec::new();
        for row in 0..rows {
            if fold_of(&plan.key, row, plan.k) == fold {
                hold.push(row);
            } else {
                train.push(row);
            }
        }
        ensure!(
            !hold.is_empty() && train.len() >= 8,
            "fold {fold} of {} holds out {} rows and trains on {}; the dataset is too small for k={}",
            plan.key,
            hold.len(),
            train.len(),
            plan.k
        );
        let cut_id = canonical_sha256(&CutIdentity {
            schema: CUT_SCHEMA,
            dataset_hash,
            train_sha256,
            key: &plan.key,
            seed: plan.seed,
            k: plan.k,
            fold,
            rows,
            hold_sha256: indices_sha256(&hold),
        })?;
        cuts.push(Cut {
            fold,
            cut_id,
            train,
            hold,
        });
    }
    Ok(cuts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuts_partition_rows_and_ids_follow_every_input() {
        let plan = FoldPlan::new("0123456789abcdef", 7, 5);
        let cuts = build(&plan, "0123456789abcdef", "s", 100).unwrap();
        assert_eq!(cuts.len(), 5);
        let mut seen = vec![0usize; 100];
        for cut in &cuts {
            assert_eq!(cut.train.len() + cut.hold.len(), 100);
            for &row in &cut.hold {
                seen[row] += 1;
            }
            assert!(cut.train.windows(2).all(|w| w[0] < w[1]));
        }
        assert!(
            seen.iter().all(|&n| n == 1),
            "every row held out exactly once"
        );
        assert_eq!(cuts[0].cut_id.len(), 64);
        assert_ne!(cuts[0].cut_id, cuts[1].cut_id);

        let again = build(&plan, "0123456789abcdef", "s", 100).unwrap();
        assert_eq!(again, cuts, "deterministic");
        let other_seed = build(
            &FoldPlan::new("0123456789abcdef", 8, 5),
            "0123456789abcdef",
            "s",
            100,
        )
        .unwrap();
        assert_ne!(other_seed[0].cut_id, cuts[0].cut_id);
        let other_bytes = build(&plan, "0123456789abcdef", "t", 100).unwrap();
        assert_eq!(other_bytes[0].hold, cuts[0].hold, "same rows");
        assert_ne!(
            other_bytes[0].cut_id, cuts[0].cut_id,
            "different bytes, different id"
        );
        let other_k = build(
            &FoldPlan::new("0123456789abcdef", 7, 4),
            "0123456789abcdef",
            "s",
            100,
        )
        .unwrap();
        assert_eq!(other_k.len(), 4);
        assert_ne!(
            other_k[0].cut_id, cuts[0].cut_id,
            "different k, different id"
        );
        let other_rows = build(&plan, "0123456789abcdef", "s", 99).unwrap();
        assert_ne!(
            other_rows[0].cut_id, cuts[0].cut_id,
            "different row count, different id"
        );
        assert!(
            build(&FoldPlan::new("x", 1, 5), "x", "s", 10).is_err(),
            "too small"
        );
    }
}

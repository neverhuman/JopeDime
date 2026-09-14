//! A7 — missingness FeatureBank (target-free, fold-safe).
//!
//! The CSV loader maps empty/non-numeric cells to NaN, and the augmented matrix later median-fills them —
//! which throws away the *missingness pattern*. In clinical/survey data (e.g. STS, ~21% missing) the set
//! of recorded fields is itself signal. Here we surface that pattern as explicit base features: a
//! `miss::<col>` indicator (1 if the cell is NaN) for every column whose TRAIN missing-rate >= min_rate,
//! plus row-level `miss_count` / `miss_frac` over the original base columns.
//! Which columns qualify is decided on TRAIN only; the identical plan is applied to the holdout. Nothing
//! here reads the label, so it cannot leak.
use crate::data::{Dataset, Matrix};

const ROW_AGG: usize = 2; // miss_count, miss_frac

fn col_miss_rate(col: &[f32]) -> f32 {
    let m = col.iter().filter(|v| !v.is_finite()).count();
    m as f32 / col.len().max(1) as f32
}

/// Columns (train missing-rate >= `min_rate`) that get an explicit indicator. Decided on TRAIN only.
pub fn qualifying_columns(train: &Dataset, min_rate: f32) -> Vec<usize> {
    (0..train.p())
        .filter(|&j| col_miss_rate(train.x.col(j)) >= min_rate)
        .collect()
}

fn build_one(ds: &Dataset, base_p: usize, keep: &[usize]) -> Dataset {
    let n = ds.n();
    let p_new = base_p + keep.len() + ROW_AGG;
    let mut flat: Vec<f32> = Vec::with_capacity(n * p_new);
    // original base columns (still carrying NaN; the augmented-matrix fill happens later)
    for j in 0..base_p {
        flat.extend_from_slice(ds.x.col(j));
    }
    // miss indicators for the qualifying columns
    for &j in keep {
        let col = ds.x.col(j);
        flat.extend(col.iter().map(|v| if v.is_finite() { 0.0 } else { 1.0 }));
    }
    // row aggregates over the ORIGINAL base columns
    let mut miss_count = vec![0f32; n];
    for j in 0..base_p {
        for (i, v) in ds.x.col(j).iter().enumerate() {
            if !v.is_finite() {
                miss_count[i] += 1.0;
            }
        }
    }
    flat.extend_from_slice(&miss_count);
    let inv = 1.0 / base_p.max(1) as f32;
    flat.extend(miss_count.iter().map(|c| c * inv));

    let mut names = ds.names.clone();
    for &j in keep {
        names.push(format!("miss::{}", ds.names[j]));
    }
    names.push("miss_count".to_string());
    names.push("miss_frac".to_string());

    Dataset {
        x: Matrix::new(flat, n, p_new),
        y: ds.y.clone(),
        y_multi: ds.y_multi.clone(),
        names,
    }
}

/// Return augmented (train, hold) datasets with missingness indicators + row aggregates appended as extra
/// base features. The qualifying-column set is fit on TRAIN and applied to both.
pub fn augment_missing(train: &Dataset, hold: &Dataset, min_rate: f32) -> (Dataset, Dataset) {
    let base_p = train.p();
    let keep = qualifying_columns(train, min_rate);
    (
        build_one(train, base_p, &keep),
        build_one(hold, base_p, &keep),
    )
}

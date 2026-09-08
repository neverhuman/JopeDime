//! Math primitives shared across phases: standardize, correlation, rank-AUC, stratified
//! subsampling and fixed k-fold splits.
use crate::rng::Rng64;
use rand::seq::SliceRandom;
use rand::Rng;

pub const EPS: f32 = 1e-12;

/// Sum of squared deviations from the mean (variance * n). Used by GP split-gain.
pub fn sse(vals: &[f32]) -> f32 {
    let n = vals.len() as f32;
    if n < 1.0 {
        return 0.0;
    }
    let s: f32 = vals.iter().sum();
    let ss: f32 = vals.iter().map(|v| v * v).sum();
    ss - s * s / n
}

/// Standardize over FINITE values (NaN -> mean -> 0 after centering). Returns (vec, ok).
/// `ok` is false if fewer than 2 finite values or near-zero std (degenerate feature).
pub fn standardize(v: &[f32]) -> (Vec<f32>, bool) {
    let mut sum = 0.0f64;
    let mut cnt = 0usize;
    for &x in v {
        if x.is_finite() {
            sum += x as f64;
            cnt += 1;
        }
    }
    if cnt < 2 {
        return (vec![0.0; v.len()], false);
    }
    let mean = (sum / cnt as f64) as f32;
    let mut var = 0.0f64;
    for &x in v {
        if x.is_finite() {
            let d = (x - mean) as f64;
            var += d * d;
        }
    }
    let std = (var / cnt as f64).sqrt() as f32;
    if !(std > 1e-9) {
        return (vec![0.0; v.len()], false);
    }
    let out: Vec<f32> = v
        .iter()
        .map(|&x| if x.is_finite() { (x - mean) / std } else { 0.0 })
        .collect();
    (out, true)
}

/// |Pearson| of two already-standardized vectors (mean 0 / std 1 over finite, NaN->0).
pub fn corr_std(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    (dot / n).abs().min(1.0)
}

/// Rank-based ROC AUC, O(n log n), with correct tie-averaging. `y_true` in {0,1}.
pub fn auc(y_true: &[f32], score: &[f32]) -> f64 {
    let n = y_true.len();
    if n == 0 {
        return 0.5;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        score[a]
            .partial_cmp(&score[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut rank = vec![0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && score[idx[j + 1]] == score[idx[i]] {
            j += 1;
        }
        // average of 1-based ranks (i+1)..=(j+1)
        let avg = ((i + 1 + j + 1) as f64) / 2.0;
        for &k in &idx[i..=j] {
            rank[k] = avg;
        }
        i = j + 1;
    }
    let (mut sum_pos, mut n_pos, mut n_neg) = (0f64, 0usize, 0usize);
    for k in 0..n {
        if y_true[k] > 0.5 {
            sum_pos += rank[k];
            n_pos += 1;
        } else {
            n_neg += 1;
        }
    }
    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }
    (sum_pos - (n_pos * (n_pos + 1)) as f64 / 2.0) / (n_pos as f64 * n_neg as f64)
}

// ----------------------- task-aware metrics (regression / multiclass) -----------------------

pub fn rmse(y: &[f32], pred: &[f32]) -> f64 {
    let n = y.len().max(1) as f64;
    let s: f64 = y
        .iter()
        .zip(pred)
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum();
    (s / n).sqrt()
}
pub fn mae(y: &[f32], pred: &[f32]) -> f64 {
    let n = y.len().max(1) as f64;
    y.iter()
        .zip(pred)
        .map(|(&a, &b)| (a - b).abs() as f64)
        .sum::<f64>()
        / n
}
/// Coefficient of determination (higher is better; ≤1). Returns 0 for a degenerate (zero-variance) target.
pub fn r2(y: &[f32], pred: &[f32]) -> f64 {
    let n = y.len() as f64;
    if n < 1.0 {
        return 0.0;
    }
    let mean = y.iter().map(|&v| v as f64).sum::<f64>() / n;
    let ss_tot: f64 = y.iter().map(|&v| (v as f64 - mean).powi(2)).sum();
    let ss_res: f64 = y
        .iter()
        .zip(pred)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum();
    if ss_tot <= 1e-12 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}
/// Binary logloss; `proba` clamped away from 0/1.
pub fn logloss(y: &[f32], proba: &[f32]) -> f64 {
    let n = y.len().max(1) as f64;
    let mut s = 0.0;
    for (&yy, &p) in y.iter().zip(proba) {
        let p = (p as f64).clamp(1e-9, 1.0 - 1e-9);
        s -= if yy > 0.5 { p.ln() } else { (1.0 - p).ln() };
    }
    s / n
}
/// Multi-class logloss. `proba` is row-major rows×k (index `r*k + c`); `y_idx` holds class indices.
pub fn multi_logloss(y_idx: &[f32], proba: &[f32], k: usize) -> f64 {
    let n = y_idx.len().max(1) as f64;
    let mut s = 0.0;
    for (r, &yi) in y_idx.iter().enumerate() {
        let c = (yi as usize).min(k.saturating_sub(1));
        let p = (proba[r * k + c] as f64).clamp(1e-9, 1.0);
        s -= p.ln();
    }
    s / n
}
/// One-vs-rest macro-averaged AUC for multi-class. `proba` row-major rows×k; `y_idx` class indices.
pub fn ovr_macro_auc(y_idx: &[f32], proba: &[f32], k: usize) -> f64 {
    let rows = y_idx.len();
    let mut sum = 0.0;
    let mut cnt = 0;
    for c in 0..k {
        let yc: Vec<f32> = y_idx
            .iter()
            .map(|&v| if (v as usize) == c { 1.0 } else { 0.0 })
            .collect();
        let sc: Vec<f32> = (0..rows).map(|r| proba[r * k + c]).collect();
        if yc.iter().any(|&v| v > 0.5) && yc.iter().any(|&v| v < 0.5) {
            sum += auc(&yc, &sc);
            cnt += 1;
        }
    }
    if cnt == 0 {
        0.5
    } else {
        sum / cnt as f64
    }
}

/// The single "higher is better" scalar every selector/ensemble optimizes — keeps `MaskScorer`
/// task-agnostic. `pred` shape: rows×1 for Binary/MultiLabel(primary)/Regression, rows×`n_classes`
/// for MultiClass.
pub fn task_score(kind: crate::task::TaskKind, y: &[f32], pred: &[f32], n_classes: usize) -> f64 {
    use crate::task::TaskKind as TK;
    match kind {
        TK::Binary | TK::MultiLabel => auc(y, pred),
        TK::Regression => r2(y, pred),
        TK::MultiClass => ovr_macro_auc(y, pred, n_classes),
    }
}

/// Human/UI label for the scalar [`task_score`] returns — so progress events can label the number
/// carried in `best_val_auc`/`best_holdout_auc` correctly per task (it is `r2`, not AUC, for
/// regression). Kept in lockstep with `task_score`.
pub fn metric_name(kind: crate::task::TaskKind) -> &'static str {
    use crate::task::TaskKind as TK;
    match kind {
        TK::Binary | TK::MultiLabel | TK::MultiClass => "auc",
        TK::Regression => "r2",
    }
}

/// Sample `k` distinct elements from `pool` without replacement (partial Fisher-Yates).
fn sample_n(pool: &[usize], k: usize, rng: &mut Rng64) -> Vec<usize> {
    let k = k.min(pool.len());
    let mut p = pool.to_vec();
    let len = p.len();
    for i in 0..k {
        let j = rng.gen_range(i..len);
        p.swap(i, j);
    }
    p.truncate(k);
    p
}

/// Stratified row subsample preserving the positive rate. Returns shuffled indices of size ~`n`.
pub fn stratified_subsample(y: &[f32], n: usize, rng: &mut Rng64) -> Vec<usize> {
    let total = y.len();
    let i1: Vec<usize> = (0..total).filter(|&i| y[i] > 0.5).collect();
    let i0: Vec<usize> = (0..total).filter(|&i| y[i] <= 0.5).collect();
    if total == 0 {
        return Vec::new();
    }
    let n1 = ((n as f64 * i1.len() as f64 / total as f64).round() as usize).min(i1.len());
    let n0 = n.saturating_sub(n1).min(i0.len());
    let mut out = sample_n(&i0, n0, rng);
    out.extend(sample_n(&i1, n1, rng));
    out.shuffle(rng);
    out
}

/// Stratified k-fold over `rows`. Returns (train_idx, val_idx) per fold. Seeded & fixed:
/// drawing once and reusing across generations is what makes CV-mean track the blind holdout.
pub fn stratified_kfold(
    rows: &[usize],
    y: &[f32],
    k: usize,
    seed: u64,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut rng = crate::rng::seeded(seed);
    let mut pos: Vec<usize> = rows.iter().copied().filter(|&i| y[i] > 0.5).collect();
    let mut neg: Vec<usize> = rows.iter().copied().filter(|&i| y[i] <= 0.5).collect();
    pos.shuffle(&mut rng);
    neg.shuffle(&mut rng);
    let mut folds: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (c, &i) in pos.iter().enumerate() {
        folds[c % k].push(i);
    }
    for (c, &i) in neg.iter().enumerate() {
        folds[c % k].push(i);
    }
    (0..k)
        .map(|f| {
            let val = folds[f].clone();
            let mut train = Vec::new();
            for (g, fold) in folds.iter().enumerate() {
                if g != f {
                    train.extend_from_slice(fold);
                }
            }
            (train, val)
        })
        .collect()
}

/// Tie-averaged rank scaled to [0,1] (same rank machinery as `auc`). Label-free, deterministic.
pub fn rank01(col: &[f32]) -> Vec<f32> {
    let n = col.len();
    if n == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        col[a]
            .partial_cmp(&col[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut rank = vec![0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && col[idx[j + 1]] == col[idx[i]] {
            j += 1;
        }
        let avg = ((i + j) as f32) / 2.0; // average 0-based index over the tie block
        for &k in &idx[i..=j] {
            rank[k] = avg;
        }
        i = j + 1;
    }
    let denom = n.saturating_sub(1).max(1) as f32;
    for v in &mut rank {
        *v /= denom;
    }
    rank
}

/// Mean/std/SE/LCB of per-fold AUC for one prediction vector over a scope's folds.
pub struct FoldMetricSummary {
    pub fold: Vec<f64>,
    pub mean: f64,
    pub std: f64,
    pub se: f64,
    pub lcb: f64,
}

/// Per-fold AUC of `pred` on each fold's validation rows, summarized with `lcb = mean - t*se`.
pub fn fold_auc_lcb(
    pred: &[f32],
    y: &[f32],
    folds: &[(Vec<usize>, Vec<usize>)],
    t: f64,
) -> FoldMetricSummary {
    let fold: Vec<f64> = folds
        .iter()
        .map(|(_, va)| {
            let yv: Vec<f32> = va.iter().map(|&i| y[i]).collect();
            let pv: Vec<f32> = va.iter().map(|&i| pred[i]).collect();
            auc(&yv, &pv)
        })
        .collect();
    let k = fold.len().max(1) as f64;
    let mean = fold.iter().sum::<f64>() / k;
    let var = fold.iter().map(|a| (a - mean) * (a - mean)).sum::<f64>() / k;
    let std = var.sqrt();
    let se = std / k.sqrt();
    FoldMetricSummary {
        fold,
        mean,
        std,
        se,
        lcb: mean - t * se,
    }
}

/// |Pearson| correlation of two prediction vectors (ensemble diversity).
pub fn prediction_corr(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    if n == 0 || n != b.len() {
        return 0.0;
    }
    let (mut sa, mut sb) = (0f64, 0f64);
    for i in 0..n {
        sa += a[i] as f64;
        sb += b[i] as f64;
    }
    let (ma, mb) = (sa / n as f64, sb / n as f64);
    let (mut cov, mut va, mut vb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        let da = a[i] as f64 - ma;
        let db = b[i] as f64 - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 0.0 || vb <= 0.0 {
        return 0.0;
    }
    (cov / (va.sqrt() * vb.sqrt())).abs().min(1.0)
}

#[cfg(test)]
mod metric_label_tests {
    use super::metric_name;
    use crate::task::TaskKind;

    #[test]
    fn metric_name_matches_task_score_family() {
        // Classification families are scored by AUC; regression by R^2. Keep this in lockstep with
        // `task_score` so progress events label `best_val_auc` correctly.
        assert_eq!(metric_name(TaskKind::Binary), "auc");
        assert_eq!(metric_name(TaskKind::MultiLabel), "auc");
        assert_eq!(metric_name(TaskKind::MultiClass), "auc");
        assert_eq!(metric_name(TaskKind::Regression), "r2");
    }
}

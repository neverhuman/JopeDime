//! Metrics in the corpus's convention (sklearn `r2_score`): 1 - SS_res/SS_tot with
//! test-mean centering.

pub fn r2(y: &[f64], p: &[f64]) -> f64 {
    let n = y.len().min(p.len());
    if n == 0 {
        return 0.0;
    }
    let mean = y[..n].iter().sum::<f64>() / n as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for i in 0..n {
        let d = y[i] - p[i];
        ss_res += d * d;
        let t = y[i] - mean;
        ss_tot += t * t;
    }
    if ss_tot <= 0.0 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

pub fn rmse(y: &[f64], p: &[f64]) -> f64 {
    let n = y.len().min(p.len());
    if n == 0 {
        return 0.0;
    }
    let mut ss = 0.0;
    for i in 0..n {
        let d = y[i] - p[i];
        ss += d * d;
    }
    (ss / n as f64).sqrt()
}

pub fn mae(y: &[f64], p: &[f64]) -> f64 {
    let n = y.len().min(p.len());
    if n == 0 {
        return 0.0;
    }
    let mut sa = 0.0;
    for i in 0..n {
        sa += (y[i] - p[i]).abs();
    }
    sa / n as f64
}

pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let ma = a[..n].iter().sum::<f64>() / n as f64;
    let mb = b[..n].iter().sum::<f64>() / n as f64;
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return 0.0;
    }
    sab / (saa * sbb).sqrt()
}

// Binary-classification metrics. A row is positive iff y == 1.0 exactly; every
// other target value counts as negative. Scores/probabilities are ordered with
// `total_cmp`, so NaN scores sort deterministically instead of poisoning ranks.

/// Probability clip for `logloss`: caps one confidently-wrong row's
/// contribution at -ln(1e-6) ~= 13.8 so a single row cannot dominate a small
/// fold, while staying far below any calibration resolution achievable from
/// <= 1000 training rows.
pub const LOGLOSS_EPS: f64 = 1e-6;

fn positives(y: &[f64]) -> usize {
    y.iter().filter(|&&v| v == 1.0).count()
}

/// Exact Mann-Whitney AUC with midrank tie handling, O(n log n).
/// Degenerate inputs (empty, or a single class) return 0.5.
pub fn auc(y: &[f64], p: &[f64]) -> f64 {
    let n = y.len().min(p.len());
    let n_pos = positives(&y[..n]);
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| p[a].total_cmp(&p[b]).then(a.cmp(&b)));
    let mut rank_pos = 0.0;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && p[idx[j + 1]].total_cmp(&p[idx[i]]).is_eq() {
            j += 1;
        }
        let midrank = (i + j + 2) as f64 / 2.0;
        let group_pos = idx[i..=j].iter().filter(|&&k| y[k] == 1.0).count();
        rank_pos += midrank * group_pos as f64;
        i = j + 1;
    }
    let u = rank_pos - n_pos as f64 * (n_pos as f64 + 1.0) / 2.0;
    u / (n_pos as f64 * n_neg as f64)
}

pub fn logloss(y: &[f64], p: &[f64]) -> f64 {
    let n = y.len().min(p.len());
    if n == 0 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let pi = p[i].clamp(LOGLOSS_EPS, 1.0 - LOGLOSS_EPS);
        s -= y[i] * pi.ln() + (1.0 - y[i]) * (1.0 - pi).ln();
    }
    s / n as f64
}

pub fn brier(y: &[f64], p: &[f64]) -> f64 {
    let n = y.len().min(p.len());
    if n == 0 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let d = p[i] - y[i];
        s += d * d;
    }
    s / n as f64
}

/// Indices sorted descending by score, ties broken by ascending index. Tie
/// groups are contiguous, so block-walking callers stay order-independent.
fn desc_order(p: &[f64], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    idx
}

/// Average precision as a step-sum over score-descending tie blocks; equal
/// scores are processed as one block, so the value never depends on row order.
/// No positives -> 0.0.
pub fn average_precision(y: &[f64], p: &[f64]) -> f64 {
    let n = y.len().min(p.len());
    let n_pos = positives(&y[..n]);
    if n_pos == 0 {
        return 0.0;
    }
    let idx = desc_order(p, n);
    let (mut tp, mut fp) = (0usize, 0usize);
    let mut ap = 0.0;
    let mut recall_prev = 0.0;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && p[idx[j + 1]].total_cmp(&p[idx[i]]).is_eq() {
            j += 1;
        }
        let group_pos = idx[i..=j].iter().filter(|&&k| y[k] == 1.0).count();
        tp += group_pos;
        fp += (j - i + 1) - group_pos;
        let precision = tp as f64 / (tp + fp) as f64;
        let recall = tp as f64 / n_pos as f64;
        ap += (recall - recall_prev) * precision;
        recall_prev = recall;
        i = j + 1;
    }
    ap
}

/// Lift of the top ceil(frac*n) rows by score over the base positive rate.
/// A tie block straddling the cut contributes fractionally (slots * block
/// positive rate), so the value never depends on row order. No positives or
/// empty input -> 0.0.
pub fn lift_at(y: &[f64], p: &[f64], frac: f64) -> f64 {
    let n = y.len().min(p.len());
    let n_pos = positives(&y[..n]);
    if n == 0 || n_pos == 0 {
        return 0.0;
    }
    let k = ((frac * n as f64).ceil() as usize).clamp(1, n);
    let idx = desc_order(p, n);
    let mut taken = 0usize;
    let mut credited = 0.0;
    let mut i = 0;
    while i < n && taken < k {
        let mut j = i;
        while j + 1 < n && p[idx[j + 1]].total_cmp(&p[idx[i]]).is_eq() {
            j += 1;
        }
        let group = j - i + 1;
        let group_pos = idx[i..=j].iter().filter(|&&k| y[k] == 1.0).count();
        let slots = k - taken;
        if group <= slots {
            credited += group_pos as f64;
            taken += group;
        } else {
            credited += slots as f64 * group_pos as f64 / group as f64;
            taken = k;
        }
        i = j + 1;
    }
    let base = n_pos as f64 / n as f64;
    (credited / k as f64) / base
}

/// A row is predicted positive iff its score is >= the threshold.
pub fn accuracy_at(y: &[f64], p: &[f64], thr: f64) -> f64 {
    let n = y.len().min(p.len());
    if n == 0 {
        return 0.0;
    }
    let mut correct = 0usize;
    for i in 0..n {
        if (p[i] >= thr) == (y[i] == 1.0) {
            correct += 1;
        }
    }
    correct as f64 / n as f64
}

pub fn f1_at(y: &[f64], p: &[f64], thr: f64) -> f64 {
    let n = y.len().min(p.len());
    let (mut tp, mut fp, mut fneg) = (0usize, 0usize, 0usize);
    for i in 0..n {
        let pred = p[i] >= thr;
        let actual = y[i] == 1.0;
        match (pred, actual) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fneg += 1,
            (false, false) => {}
        }
    }
    let denom = 2 * tp + fp + fneg;
    if denom == 0 {
        return 0.0;
    }
    2.0 * tp as f64 / denom as f64
}

/// Candidate thresholds for the sweeps: midpoints of consecutive distinct
/// scores, the smallest distinct score (predicts everything positive under the
/// >= convention), and 0.5. Sorted ascending, deduped.
fn threshold_candidates(p: &[f64], n: usize) -> Vec<f64> {
    let mut sorted: Vec<f64> = p[..n].to_vec();
    sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    sorted.dedup_by(|a, b| a.total_cmp(b).is_eq());
    let mut cands = Vec::with_capacity(sorted.len() + 1);
    if let Some(&lo) = sorted.first() {
        cands.push(lo);
    }
    for w in sorted.windows(2) {
        cands.push((w[0] + w[1]) / 2.0);
    }
    cands.push(0.5);
    cands.sort_unstable_by(|a, b| a.total_cmp(b));
    cands.dedup_by(|a, b| a.total_cmp(b).is_eq());
    cands
}

fn best_threshold_by(y: &[f64], p: &[f64], score: impl Fn(f64) -> f64) -> (f64, f64) {
    let n = y.len().min(p.len());
    if n == 0 {
        return (0.5, 0.0);
    }
    let mut best_thr = 0.5;
    let mut best = f64::NEG_INFINITY;
    for thr in threshold_candidates(p, n) {
        let v = score(thr);
        // Strictly-better wins; ties keep the smallest threshold (candidates
        // are visited in ascending order).
        if v > best {
            best = v;
            best_thr = thr;
        }
    }
    (best_thr, best)
}

/// (threshold, f1) maximizing F1 over the candidate sweep; ties -> smallest
/// threshold. Degenerate inputs return (0.5, 0.0).
pub fn best_f1_threshold(y: &[f64], p: &[f64]) -> (f64, f64) {
    let n = y.len().min(p.len());
    if positives(&y[..n]) == 0 {
        return (0.5, 0.0);
    }
    best_threshold_by(&y[..n], &p[..n], |thr| f1_at(&y[..n], &p[..n], thr))
}

/// (threshold, J) maximizing Youden's J = TPR - FPR over the candidate sweep;
/// ties -> smallest threshold. Single-class inputs return (0.5, 0.0).
pub fn youden_threshold(y: &[f64], p: &[f64]) -> (f64, f64) {
    let n = y.len().min(p.len());
    let n_pos = positives(&y[..n]);
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return (0.5, 0.0);
    }
    best_threshold_by(&y[..n], &p[..n], |thr| {
        let (mut tp, mut fp) = (0usize, 0usize);
        for i in 0..n {
            if p[i] >= thr {
                if y[i] == 1.0 {
                    tp += 1;
                } else {
                    fp += 1;
                }
            }
        }
        tp as f64 / n_pos as f64 - fp as f64 / n_neg as f64
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_hand_value() {
        let y = [0.0, 0.0, 1.0, 1.0];
        let p = [0.1, 0.4, 0.35, 0.8];
        assert_eq!(auc(&y, &p), 0.75);
    }

    #[test]
    fn auc_midrank_ties() {
        let y = [0.0, 1.0, 1.0];
        let p = [0.5, 0.5, 0.9];
        assert_eq!(auc(&y, &p), 0.75);
    }

    #[test]
    fn auc_all_tied_is_half() {
        let y = [0.0, 1.0, 0.0, 1.0, 1.0];
        let p = [0.3; 5];
        assert_eq!(auc(&y, &p), 0.5);
    }

    #[test]
    fn auc_perfect_inverted_degenerate() {
        let y = [0.0, 0.0, 1.0, 1.0];
        assert_eq!(auc(&y, &[0.1, 0.2, 0.8, 0.9]), 1.0);
        assert_eq!(auc(&y, &[0.9, 0.8, 0.2, 0.1]), 0.0);
        assert_eq!(auc(&[1.0, 1.0], &[0.1, 0.9]), 0.5);
        assert_eq!(auc(&[], &[]), 0.5);
    }

    #[test]
    fn auc_is_order_independent() {
        let y = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let p = [0.2, 0.2, 0.5, 0.5, 0.5, 0.9];
        let base = auc(&y, &p);
        let perm = [5usize, 3, 1, 4, 2, 0];
        let yp: Vec<f64> = perm.iter().map(|&i| y[i]).collect();
        let pp: Vec<f64> = perm.iter().map(|&i| p[i]).collect();
        assert_eq!(auc(&yp, &pp), base);
    }

    #[test]
    fn auc_monotone_transform_invariant() {
        let y = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let p = [0.1, 0.7, 0.3, 0.9, 0.6, 0.2];
        let q: Vec<f64> = p.iter().map(|v: &f64| (5.0 * v).exp()).collect();
        assert_eq!(auc(&y, &p), auc(&y, &q));
    }

    #[test]
    fn logloss_hand_value_and_clip() {
        let y = [0.0, 1.0];
        let p = [0.1, 0.8];
        let expected = -(0.9f64.ln() + 0.8f64.ln()) / 2.0;
        assert!((logloss(&y, &p) - expected).abs() < 1e-15);
        let clipped = logloss(&[1.0], &[0.0]);
        assert!(clipped.is_finite());
        assert!((clipped + LOGLOSS_EPS.ln()).abs() < 1e-9);
    }

    #[test]
    fn brier_is_mse() {
        let y = [0.0, 1.0, 1.0];
        let p = [0.25, 0.5, 1.0];
        let expected = (0.0625 + 0.25 + 0.0) / 3.0;
        assert!((brier(&y, &p) - expected).abs() < 1e-15);
    }

    #[test]
    fn average_precision_hand_value_with_tie_block() {
        let y = [1.0, 0.0, 1.0];
        let p = [0.9, 0.8, 0.7];
        assert!((average_precision(&y, &p) - 5.0 / 6.0).abs() < 1e-15);
        // Tie block: both 0.8-scored rows enter as one block.
        let y2 = [1.0, 1.0, 0.0];
        let p2 = [0.9, 0.8, 0.8];
        // Block 1: tp=1 fp=0 -> P=1, R=1/2, ap += 1/2.
        // Block 2 (tie): tp=2 fp=1 -> P=2/3, R=1, ap += (1/2)*(2/3).
        assert!((average_precision(&y2, &p2) - (0.5 + 0.5 * 2.0 / 3.0)).abs() < 1e-15);
        assert_eq!(average_precision(&[0.0, 0.0], &[0.1, 0.2]), 0.0);
    }

    #[test]
    fn lift_hand_values() {
        let y = [1.0, 1.0, 0.0, 0.0];
        let p = [0.9, 0.8, 0.2, 0.1];
        assert_eq!(lift_at(&y, &p, 0.5), 2.0);
        // Fractional credit inside the cut tie block.
        let y2 = [1.0, 0.0];
        let p2 = [0.5, 0.5];
        assert_eq!(lift_at(&y2, &p2, 0.5), 1.0);
        assert_eq!(lift_at(&[0.0, 0.0], &[0.5, 0.4], 0.5), 0.0);
    }

    #[test]
    fn f1_accuracy_and_threshold_sweeps() {
        let y = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let p = [0.1, 0.3, 0.45, 0.7, 0.9, 0.6];
        assert!((accuracy_at(&y, &p, 0.5) - 4.0 / 6.0).abs() < 1e-15);
        // At thr 0.5: tp=2 fp=1 fn=1 -> F1 = 4/6.
        assert!((f1_at(&y, &p, 0.5) - 4.0 / 6.0).abs() < 1e-15);
        // At thr in (0.3, 0.45]: tp=3 fp=1 fn=0 -> F1 = 6/7 (best).
        let (thr, f1) = best_f1_threshold(&y, &p);
        assert!((f1 - 6.0 / 7.0).abs() < 1e-15);
        assert!(thr > 0.3 && thr <= 0.45);
        // Youden: J = 2/3 both at thr in (0.3, 0.45] (TPR=1, FPR=1/3) and at
        // thr in (0.6, 0.7] (TPR=2/3, FPR=0); the tie keeps the smaller.
        let (jthr, j) = youden_threshold(&y, &p);
        assert!((j - 2.0 / 3.0).abs() < 1e-15);
        assert!(jthr > 0.3 && jthr <= 0.45);
        assert_eq!(best_f1_threshold(&[0.0, 0.0], &[0.2, 0.8]), (0.5, 0.0));
        assert_eq!(youden_threshold(&[1.0, 1.0], &[0.2, 0.8]), (0.5, 0.0));
    }

    #[test]
    fn threshold_ties_prefer_smallest() {
        // Two thresholds achieve the same F1; the sweep must return the
        // smaller one.
        let y = [1.0, 1.0];
        let p = [0.2, 0.8];
        let (thr, f1) = best_f1_threshold(&y, &p);
        assert_eq!(f1, 1.0);
        assert_eq!(thr, 0.2);
    }
}

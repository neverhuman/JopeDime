//! Meta-fusion math: Bayesian model averaging (Yao 2018 stacking-of-predictive-distributions, EM simplex)
//! and BBC-CV (Tsamardinos 2018 bootstrap-bias-corrected selection). The BBC selector is the formal core
//! of the dominance guarantee: choosing among candidates by per-bootstrap in-bag selection + out-of-bag
//! scoring removes the winner's-curse optimism, so a blend is chosen over the best single member ONLY if
//! it honestly out-performs it on held-out bootstraps.
use crate::metrics::task_score;
use crate::rng::seeded;
use crate::task::TaskKind;
use rand::Rng;

/// Bayesian model-averaging simplex weights maximizing the OOF log-score of the linear opinion pool.
/// `member_oof[m]` = member m's OOF probabilities (length n); `y` binary {0,1}. EM fixed-point.
pub fn bma_weights(member_oof: &[Vec<f32>], y: &[f32], iters: usize) -> Vec<f64> {
    let m = member_oof.len();
    let n = y.len();
    if m == 0 {
        return vec![];
    }
    // p_m(y_i): each member's predicted probability of the OBSERVED class.
    let pm: Vec<Vec<f64>> = member_oof
        .iter()
        .map(|oof| {
            (0..n)
                .map(|i| {
                    let p = (oof[i] as f64).clamp(1e-6, 1.0 - 1e-6);
                    if y[i] > 0.5 {
                        p
                    } else {
                        1.0 - p
                    }
                })
                .collect()
        })
        .collect();
    let mut w = vec![1.0 / m as f64; m];
    for _ in 0..iters {
        let mut nw = vec![0.0; m];
        for i in 0..n {
            let denom: f64 = w
                .iter()
                .zip(pm.iter())
                .map(|(&wk, pmk)| wk * pmk[i])
                .sum::<f64>()
                .max(1e-12);
            for (nwk, (&wk, pmk)) in nw.iter_mut().zip(w.iter().zip(pm.iter())) {
                *nwk += wk * pmk[i] / denom;
            }
        }
        for v in &mut nw {
            *v /= n as f64;
        }
        w = nw;
    }
    w
}

pub struct BbcResult {
    pub winner: usize,
    pub corrected: f64, // mean out-of-bag score of the winner (winner's-curse-free)
    pub ci_lo: f64,
    pub ci_hi: f64,
    pub win_freq: Vec<f64>,
}

/// BBC-CV selection among candidate OOF prediction vectors. For each of `b` bootstraps: pick the candidate
/// maximizing `task_score` on the IN-BAG rows, then score that candidate on the OUT-OF-BAG rows. The winner
/// is the most-frequently-selected candidate (ties favour the EARLIER index — put best-single first so the
/// simpler model wins ties); `corrected` is the winner's mean OOB score. With best-single included as a
/// candidate, a blend wins only if it honestly beats the best single member ⇒ the dominance guarantee.
pub fn bbc_cv_select(
    cand_oof: &[Vec<f32>],
    y: &[f32],
    task: TaskKind,
    n_classes: usize,
    b: usize,
    seed: u64,
) -> BbcResult {
    let c = cand_oof.len();
    let n = y.len();
    let mut rng = seeded(seed);
    let mut win_count = vec![0.0f64; c];
    let mut oob_scores: Vec<Vec<f64>> = vec![Vec::new(); c];
    for _ in 0..b {
        let mut inbag = vec![false; n];
        let rows: Vec<usize> = (0..n)
            .map(|_| {
                let r = rng.gen_range(0..n);
                inbag[r] = true;
                r
            })
            .collect();
        let oob: Vec<usize> = (0..n).filter(|&i| !inbag[i]).collect();
        if oob.is_empty() {
            continue;
        }
        let yib: Vec<f32> = rows.iter().map(|&i| y[i]).collect();
        // pick best candidate on in-bag (ties favour earlier index)
        let (mut best_s, mut best_k) = (f64::NEG_INFINITY, 0usize);
        for (k, cand) in cand_oof.iter().enumerate() {
            let pib: Vec<f32> = rows.iter().map(|&i| cand[i]).collect();
            let s = task_score(task, &yib, &pib, n_classes);
            if s > best_s + 1e-12 {
                best_s = s;
                best_k = k;
            }
        }
        win_count[best_k] += 1.0;
        let yoob: Vec<f32> = oob.iter().map(|&i| y[i]).collect();
        let poob: Vec<f32> = oob.iter().map(|&i| cand_oof[best_k][i]).collect();
        oob_scores[best_k].push(task_score(task, &yoob, &poob, n_classes));
    }
    // winner = most-frequently selected (ties favour earlier index)
    let mut winner = 0usize;
    for k in 1..c {
        if win_count[k] > win_count[winner] + 1e-12 {
            winner = k;
        }
    }
    let s = &oob_scores[winner];
    let (corrected, ci_lo, ci_hi) = if s.is_empty() {
        (0.5, 0.5, 0.5)
    } else {
        let mut sorted = s.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let lo = sorted[((sorted.len() as f64 * 0.025) as usize).min(sorted.len() - 1)];
        let hi = sorted[((sorted.len() as f64 * 0.975) as usize).min(sorted.len() - 1)];
        (mean, lo, hi)
    };
    let total: f64 = win_count.iter().sum::<f64>().max(1.0);
    BbcResult {
        winner,
        corrected,
        ci_lo,
        ci_hi,
        win_freq: win_count.iter().map(|&v| v / total).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bma_weights_sum_to_one_and_favor_better_member() {
        let n = 400;
        let y: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect();
        // member 0: good (aligned), member 1: noise at 0.5
        let good: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 0.2 } else { 0.8 }).collect();
        let noise: Vec<f32> = vec![0.5; n];
        let w = bma_weights(&[good, noise], &y, 100);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-6, "{w:?}");
        assert!(w[0] > w[1], "BMA up-weights the better member: {w:?}");
    }

    #[test]
    fn bbc_picks_dominant_candidate() {
        // AUC is rank-based, so a "better" candidate must rank BETTER (not just be more confident).
        let n = 700;
        let y: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect();
        // cand 0 = best-single with ~1/5 ranking errors (lower AUC); cand 1 = a perfectly-ranked blend.
        let single: Vec<f32> = (0..n)
            .map(|i| {
                let correct = if i % 2 == 0 { 0.3 } else { 0.7 };
                let flipped = if i % 2 == 0 { 0.7 } else { 0.3 };
                if i % 5 == 0 {
                    flipped
                } else {
                    correct
                }
            })
            .collect();
        let blend: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 0.1 } else { 0.9 }).collect();
        let r = bbc_cv_select(&[single, blend], &y, TaskKind::Binary, 2, 200, 1);
        assert_eq!(
            r.winner, 1,
            "BBC picks the honestly-higher-AUC blend; freq {:?}",
            r.win_freq
        );
        assert!(r.corrected > 0.9);
    }

    #[test]
    fn bbc_keeps_best_single_when_blend_is_noise() {
        let n = 600;
        let y: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect();
        let single: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 0.1 } else { 0.9 }).collect();
        let noise: Vec<f32> = vec![0.5; n];
        let r = bbc_cv_select(&[single, noise], &y, TaskKind::Binary, 2, 200, 1);
        assert_eq!(r.winner, 0, "BBC keeps best-single over a noise blend");
    }
}

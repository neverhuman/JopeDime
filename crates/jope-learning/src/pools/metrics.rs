//! Within-context selection metrics on one frozen pool.
//!
//! Every metric reads measured outcomes only after a selector has produced its scores. A pool
//! member is either a complete measurement (`Some(q)`, higher is better) or a recorded failure
//! (`None`): a failure consumes its attempt slot and earns nothing. Unknown outcomes never reach
//! this module; `pool::freeze` refuses them.

use std::cmp::Ordering;

/// The fixed attempt budget of CLOVER §5.1 (`S_8`).
pub const DEFAULT_BUDGET: usize = 8;

/// Candidate order for a selector: descending score, ties broken by ascending program sha256.
/// `keys[i]` is the tie-break key (program sha) of candidate `i`.
pub fn selection_order(scores: &[f64], keys: &[&str]) -> Vec<usize> {
    assert_eq!(scores.len(), keys.len(), "one score per candidate");
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(Ordering::Equal)
            .then_with(|| keys[a].cmp(keys[b]))
    });
    order
}

/// Incumbent-relative gain of one outcome: `max(q - h, 0)`; a failure gains nothing.
pub fn gain(outcome: Option<f64>, incumbent: f64) -> f64 {
    outcome.map_or(0.0, |q| (q - incumbent).max(0.0))
}

/// `S_B(c) = max{h_c, q(a_1..a_B)} - h_c` for the first `budget` candidates of `order`.
pub fn realized_gain(
    outcomes: &[Option<f64>],
    incumbent: f64,
    order: &[usize],
    budget: usize,
) -> f64 {
    order
        .iter()
        .take(budget)
        .map(|&i| gain(outcomes[i], incumbent))
        .fold(0.0, f64::max)
}

/// The best gain available anywhere in the pool (the within-pool oracle for any `B >= 1`).
pub fn oracle_gain(outcomes: &[Option<f64>], incumbent: f64) -> f64 {
    outcomes
        .iter()
        .map(|&o| gain(o, incumbent))
        .fold(0.0, f64::max)
}

/// Exact expectation of `S_B` when `B` distinct members are drawn uniformly without
/// replacement. With gains sorted descending `g_(1) >= ... >= g_(n)`, the draw's maximum is
/// `g_(i)` with probability `C(n-i, B-1) / C(n, B)`; the ratio is formed iteratively so it never
/// overflows. This is a deterministic number, not a sampled baseline.
pub fn uniform_expected_gain(outcomes: &[Option<f64>], incumbent: f64, budget: usize) -> f64 {
    let n = outcomes.len();
    if n == 0 || budget == 0 {
        return 0.0;
    }
    let mut gains: Vec<f64> = outcomes.iter().map(|&o| gain(o, incumbent)).collect();
    gains.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    if budget >= n {
        return gains[0];
    }
    let (nf, bf) = (n as f64, budget as f64);
    let mut p = bf / nf;
    let mut expected = 0.0;
    for (i, g) in gains.iter().enumerate() {
        if p <= 0.0 {
            break;
        }
        expected += p * g;
        // P(max = g_(i+2)) / P(max = g_(i+1)) = (n - (i+1) - B + 1) / (n - (i+1)).
        let rank = (i + 1) as f64;
        let remaining = nf - rank;
        p = if remaining > 0.0 {
            p * ((remaining - bf + 1.0) / remaining).max(0.0)
        } else {
            0.0
        };
    }
    expected
}

/// Fraction of concordant pairs among complete members with distinct `q`. A score tie counts
/// one half, so a constant selector scores exactly 0.5. `None` when no comparable pair exists.
/// Failures are excluded: execution status is a separate target (CLOVER §4.3).
pub fn pairwise_accuracy(outcomes: &[Option<f64>], scores: &[f64]) -> Option<f64> {
    let measured: Vec<(f64, f64)> = outcomes
        .iter()
        .zip(scores)
        .filter_map(|(o, &s)| o.map(|q| (q, s)))
        .collect();
    let (mut credit, mut pairs) = (0.0, 0u64);
    for (i, &(qi, si)) in measured.iter().enumerate() {
        for &(qj, sj) in &measured[i + 1..] {
            if qi == qj {
                continue;
            }
            pairs += 1;
            let truth = qi > qj;
            credit += if si == sj {
                0.5
            } else if (si > sj) == truth {
                1.0
            } else {
                0.0
            };
        }
    }
    (pairs > 0).then(|| credit / pairs as f64)
}

/// Probability that `B` distinct members drawn uniformly without replacement include at least one
/// member strictly above the reference: `1 - C(n-k, B) / C(n, B)` with `k` improving members and
/// `B` capped at `n`. Failures and ties consume slots without improving.
pub fn uniform_beats_probability(outcomes: &[Option<f64>], incumbent: f64, budget: usize) -> f64 {
    let n = outcomes.len();
    let k = outcomes
        .iter()
        .filter(|&&o| gain(o, incumbent) > 0.0)
        .count();
    if n == 0 || budget == 0 || k == 0 {
        return 0.0;
    }
    let misses = n - k;
    // C(n-k, B) / C(n, B) = prod_{i<B} (n-k-i) / (n-i); zero once the draw exhausts the misses.
    let mut all_miss = 1.0;
    for i in 0..budget.min(n) {
        if i >= misses {
            return 1.0;
        }
        all_miss *= (misses - i) as f64 / (n - i) as f64;
    }
    1.0 - all_miss
}

fn normalized(oracle: f64, realized: f64) -> Option<f64> {
    (oracle > 0.0).then(|| (oracle - realized) / oracle)
}

/// Per-context metrics of one selector on one pool.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct ContextMetrics {
    /// Realized gain recorded for this context metrics.
    pub realized_gain: f64,
    /// Top1 gain recorded for this context metrics.
    pub top1_gain: f64,
    /// Regret recorded for this context metrics.
    pub regret: f64,
    /// `regret / oracle_gain`, in [0, 1]; `None` when the pool offers no gain at all.
    pub normalized_regret: Option<f64>,
    /// Probability that the selected `B` include a member strictly above the reference: 0 or 1
    /// for a deterministic selector, the exact draw probability for `uniform_exact`.
    pub beats_reference: f64,
    /// Optional pairwise accuracy value; `None` means it is not available.
    pub pairwise_accuracy: Option<f64>,
}

/// Performs the score context operation.
pub fn score_context(
    outcomes: &[Option<f64>],
    incumbent: f64,
    scores: &[f64],
    keys: &[&str],
    budget: usize,
) -> ContextMetrics {
    let order = selection_order(scores, keys);
    let realized = realized_gain(outcomes, incumbent, &order, budget);
    ContextMetrics {
        realized_gain: realized,
        top1_gain: realized_gain(outcomes, incumbent, &order, 1),
        regret: oracle_gain(outcomes, incumbent) - realized,
        normalized_regret: normalized(oracle_gain(outcomes, incumbent), realized),
        beats_reference: if realized > 0.0 { 1.0 } else { 0.0 },
        pairwise_accuracy: pairwise_accuracy(outcomes, scores),
    }
}

/// The uniform baseline's exact expected metrics (pairwise accuracy is 0.5 by symmetry).
pub fn uniform_context(outcomes: &[Option<f64>], incumbent: f64, budget: usize) -> ContextMetrics {
    let realized = uniform_expected_gain(outcomes, incumbent, budget);
    let has_pair = {
        let mut seen: Option<f64> = None;
        outcomes.iter().flatten().any(|&q| match seen {
            Some(first) => first != q,
            None => {
                seen = Some(q);
                false
            }
        })
    };
    ContextMetrics {
        realized_gain: realized,
        top1_gain: uniform_expected_gain(outcomes, incumbent, 1),
        regret: oracle_gain(outcomes, incumbent) - realized,
        normalized_regret: normalized(oracle_gain(outcomes, incumbent), realized),
        beats_reference: uniform_beats_probability(outcomes, incumbent, budget),
        pairwise_accuracy: has_pair.then_some(0.5),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_uniform(outcomes: &[Option<f64>], h: f64, b: usize) -> f64 {
        // Enumerate every b-subset of indices.
        let n = outcomes.len();
        let (mut total, mut count) = (0.0, 0u64);
        for mask in 0u32..(1 << n) {
            if mask.count_ones() as usize != b {
                continue;
            }
            let best = (0..n)
                .filter(|i| mask & (1 << i) != 0)
                .map(|i| gain(outcomes[i], h))
                .fold(0.0, f64::max);
            total += best;
            count += 1;
        }
        total / count as f64
    }

    #[test]
    fn uniform_expectation_matches_enumeration() {
        let outcomes = [
            Some(0.31),
            None,
            Some(0.52),
            Some(0.12),
            Some(0.52),
            None,
            Some(0.47),
            Some(0.05),
            Some(0.66),
        ];
        for h in [0.0, 0.3, 0.5, 0.7] {
            for b in 1..=outcomes.len() {
                let exact = uniform_expected_gain(&outcomes, h, b);
                let brute = brute_uniform(&outcomes, h, b);
                assert!(
                    (exact - brute).abs() < 1e-12,
                    "h={h} b={b}: {exact} vs {brute}"
                );
            }
        }
    }

    fn brute_beats(outcomes: &[Option<f64>], h: f64, b: usize) -> f64 {
        let n = outcomes.len();
        let b = b.min(n);
        let (mut hits, mut count) = (0u64, 0u64);
        for mask in 0u32..(1 << n) {
            if mask.count_ones() as usize != b {
                continue;
            }
            count += 1;
            if (0..n).any(|i| mask & (1 << i) != 0 && gain(outcomes[i], h) > 0.0) {
                hits += 1;
            }
        }
        hits as f64 / count as f64
    }

    #[test]
    fn uniform_beats_probability_matches_enumeration() {
        // Review case (clover R2): q=[0.2, 0.8], h=0.5, B=1 -> 0.5, while the expected gain is 0.15.
        let review = [Some(0.2), Some(0.8)];
        assert!((uniform_beats_probability(&review, 0.5, 1) - 0.5).abs() < 1e-15);
        assert!((uniform_expected_gain(&review, 0.5, 1) - 0.15).abs() < 1e-12);
        assert!((uniform_context(&review, 0.5, 1).beats_reference - 0.5).abs() < 1e-15);
        // Ties at the reference (0.5) and failures (None) occupy slots without improving.
        let mixed = [
            Some(0.5),
            None,
            Some(0.9),
            Some(0.1),
            Some(0.5),
            None,
            Some(0.7),
        ];
        for h in [0.0, 0.5, 0.95] {
            for b in 0..=mixed.len() + 2 {
                let exact = uniform_beats_probability(&mixed, h, b);
                let brute = if b == 0 {
                    0.0
                } else {
                    brute_beats(&mixed, h, b)
                };
                assert!(
                    (exact - brute).abs() < 1e-12,
                    "h={h} b={b}: {exact} vs {brute}"
                );
            }
        }
        // No improving member, and B >= n with one improving member.
        assert_eq!(uniform_beats_probability(&mixed, 0.95, 3), 0.0);
        assert_eq!(uniform_beats_probability(&[Some(1.0), None], 0.5, 5), 1.0);
        assert_eq!(uniform_beats_probability(&[], 0.5, 1), 0.0);
    }

    #[test]
    fn deterministic_beats_reference_is_zero_or_one() {
        let outcomes = [Some(0.2), Some(0.8)];
        let keys = ["a", "b"];
        assert_eq!(
            score_context(&outcomes, 0.5, &[1.0, 0.0], &keys, 1).beats_reference,
            0.0
        );
        assert_eq!(
            score_context(&outcomes, 0.5, &[0.0, 1.0], &keys, 1).beats_reference,
            1.0
        );
    }

    #[test]
    fn realized_gain_reads_only_the_first_budget_slots() {
        let outcomes = [Some(0.9), Some(0.1), None, Some(0.4)];
        let keys = ["d", "c", "b", "a"];
        // Scores put the 0.9 program last.
        let scores = [0.0, 3.0, 2.0, 1.0];
        let order = selection_order(&scores, &keys);
        assert_eq!(order, vec![1, 2, 3, 0]);
        assert_eq!(realized_gain(&outcomes, 0.2, &order, 2), 0.0);
        assert!((realized_gain(&outcomes, 0.2, &order, 3) - 0.2).abs() < 1e-12);
        assert!((oracle_gain(&outcomes, 0.2) - 0.7).abs() < 1e-12);
    }

    #[test]
    fn ties_break_by_program_sha() {
        let order = selection_order(&[1.0, 1.0, 1.0], &["c", "a", "b"]);
        assert_eq!(order, vec![1, 2, 0]);
    }

    #[test]
    fn pairwise_accuracy_counts_concordance_and_half_ties() {
        let outcomes = [Some(3.0), Some(2.0), Some(1.0), None];
        assert_eq!(
            pairwise_accuracy(&outcomes, &[3.0, 2.0, 1.0, 9.0]),
            Some(1.0)
        );
        assert_eq!(
            pairwise_accuracy(&outcomes, &[1.0, 2.0, 3.0, 9.0]),
            Some(0.0)
        );
        assert_eq!(
            pairwise_accuracy(&outcomes, &[5.0, 5.0, 5.0, 5.0]),
            Some(0.5)
        );
        assert_eq!(
            pairwise_accuracy(&[Some(1.0), Some(1.0), None], &[0.0, 1.0, 2.0]),
            None
        );
    }

    #[test]
    fn uniform_context_has_half_pairwise_and_exact_gain() {
        let outcomes = [Some(0.2), Some(0.8), None];
        let m = uniform_context(&outcomes, 0.5, 1);
        assert!((m.realized_gain - 0.1).abs() < 1e-12);
        assert_eq!(m.pairwise_accuracy, Some(0.5));
        assert!((m.regret - 0.2).abs() < 1e-12);
    }
}

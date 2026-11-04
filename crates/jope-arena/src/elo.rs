//! Regularized Bradley–Terry ratings on the Elo scale.
//!
//! Two fits live here because two implementations produced committed numbers:
//!
//! * [`fit_elo`] / [`fit_elo_arrays`] — the benchmark clone's `sotarena/elo.py` (numpy), which
//!   wrote every `results/<track>/leaderboard.csv` (jope-dime 2259.8 on `tabular-regression-v2`).
//!   One equal-weight match per (dataset, method pair); the accumulation order is the Python
//!   order (sorted datasets, sorted method pairs, `np.bincount` sequential adds) and the final
//!   centering uses numpy's pairwise summation, so ratings match to the last bit up to the
//!   platform's `exp`.
//! * [`ratings`] — the SOTArena reporter's `src/main.rs::ratings`, the Rust reporter's version that
//!   aggregates each pair's sufficient statistics (match count, outcome sum) before the same
//!   diagonal-Newton iteration. Identical in exact arithmetic; differs by floating-point noise.
//!
//! Both use `center = 1500`, `iterations = 250`, `regularization = 1e-4`, `tolerance = 1e-8`,
//! a per-iteration update clamp of ±40 (the reporter calls it `k_factor`) and a relative tie
//! tolerance of `1e-12`.

use std::collections::BTreeMap;

use serde::Deserialize;

/// Ratings are mean-centred here.
pub const CENTER: f64 = 1500.0;
/// Maximum diagonal-Newton iterations.
pub const ITERATIONS: usize = 250;
/// L2 pull towards [`CENTER`].
pub const REGULARIZATION: f64 = 1e-4;
/// Largest-update threshold that stops the iteration.
pub const TOLERANCE: f64 = 1e-8;
/// Per-iteration clamp on every rating update (Elo points).
pub const UPDATE_CLAMP: f64 = 40.0;
/// Two scores tie when they differ by at most this fraction of `max(|a|, |b|, 1)`.
pub const TIE_RELATIVE_TOLERANCE: f64 = 1e-12;

/// `ln(10) / 400`: one Elo point in natural-log odds.
fn elo_scale() -> f64 {
    std::f64::consts::LN_10 / 400.0
}

/// Whether lower or higher metric values win.
pub use jope_api::Direction;
/// Compatibility path; all runtime objective directions share the API enum.
pub type EloDirection = Direction;

/// The reporter's `config.json` `elo` block.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EloConfig {
    /// Start value and centring target.
    pub initial_rating: f64,
    /// Per-iteration update clamp.
    pub k_factor: f64,
    /// Relative tie tolerance.
    pub tie_tolerance: f64,
    /// Iteration cap.
    pub iterations: usize,
    /// L2 regularization.
    pub regularization: f64,
    /// Convergence threshold on the largest update.
    pub convergence_tolerance: f64,
}

impl Default for EloConfig {
    fn default() -> Self {
        Self {
            initial_rating: CENTER,
            k_factor: UPDATE_CLAMP,
            tie_tolerance: TIE_RELATIVE_TOLERANCE,
            iterations: ITERATIONS,
            regularization: REGULARIZATION,
            convergence_tolerance: TOLERANCE,
        }
    }
}

/// Method → dataset → score.
pub type TaskResults = BTreeMap<String, BTreeMap<String, f64>>;

/// One pairwise result: `outcome_a` is 1 when `method_a` won, 0.5 on a tie, 0 when it lost.
#[derive(Clone, Debug, PartialEq)]
pub struct Match {
    /// The dataset the pair was compared on.
    pub dataset_id: String,
    /// Lexicographically smaller method id.
    pub method_a: String,
    /// Lexicographically larger method id.
    pub method_b: String,
    /// 1.0 / 0.5 / 0.0 from `method_a`'s point of view.
    pub outcome_a: f64,
}

/// `matches_from_scores` from `sotarena/elo.py`: every dataset contributes one equal-weight match
/// per method pair, lower scores win (the benchmark's tracks are all `rmse`), and the relative tie
/// rule is `|a − b| ≤ tol · max(|a|, |b|, 1)`. Datasets are visited in sorted order, pairs in
/// sorted upper-triangle order.
pub fn matches_from_scores(
    scores: &BTreeMap<String, BTreeMap<String, f64>>,
    tie_relative_tolerance: f64,
) -> Vec<Match> {
    matches_from_scores_directed(scores, Direction::Minimize, tie_relative_tolerance)
}

/// [`matches_from_scores`] in either direction. `Minimize` is exactly the benchmark's rule (the
/// published `rmse` ladders reproduce bit for bit); `Maximize` lets a track ranked by `roc_auc`
/// or `macro_f1` use the same fit, with the same tie rule and the same visiting order.
pub fn matches_from_scores_directed(
    scores: &BTreeMap<String, BTreeMap<String, f64>>,
    direction: Direction,
    tie_relative_tolerance: f64,
) -> Vec<Match> {
    let mut matches = Vec::new();
    for (dataset_id, by_method) in scores {
        let methods: Vec<(&String, f64)> = by_method.iter().map(|(m, v)| (m, *v)).collect();
        for (left, (method_a, score_a)) in methods.iter().enumerate() {
            for (method_b, score_b) in &methods[left + 1..] {
                matches.push(Match {
                    dataset_id: dataset_id.clone(),
                    method_a: (*method_a).clone(),
                    method_b: (*method_b).clone(),
                    outcome_a: outcome(*score_a, *score_b, direction, tie_relative_tolerance),
                });
            }
        }
    }
    matches
}

/// Result of [`fit_elo_arrays`].
#[derive(Clone, Debug, PartialEq)]
pub struct Fit {
    /// Un-centred ratings, indexed like `method_count`.
    pub ratings: Vec<f64>,
    /// Iterations actually run.
    pub iterations_run: usize,
    /// Whether the update tolerance was reached before the cap.
    pub converged: bool,
}

/// `fit_elo_arrays` from `sotarena/elo.py`: a diagonal-Newton Bradley–Terry fit over prebuilt
/// match arrays, in the numpy accumulation order.
#[allow(clippy::too_many_arguments)]
mod fit;
pub use fit::{fit_elo, fit_elo_arrays, numpy_pairwise_sum};

/// The reporter's `outcome`: 1 when `left` beats `right` in `direction`, 0.5 within the relative
/// tie tolerance, else 0.
pub fn outcome(left: f64, right: f64, direction: Direction, tolerance: f64) -> f64 {
    let scale = left.abs().max(right.abs()).max(1.0);
    if (left - right).abs() <= tolerance * scale {
        return 0.5;
    }
    match direction {
        Direction::Maximize if left > right => 1.0,
        Direction::Minimize if left < right => 1.0,
        _ => 0.0,
    }
}

/// The reporter's `metric_better`.
pub fn metric_better(candidate: f64, current: f64, direction: Direction) -> bool {
    match direction {
        Direction::Maximize => candidate > current,
        Direction::Minimize => candidate < current,
    }
}

/// `ratings` from the SOTArena reporter's `src/main.rs`, verbatim: pair statistics are aggregated
/// (count, mean outcome) before the diagonal-Newton iteration, which is exact for the
/// Bradley–Terry likelihood and avoids materialising millions of repeated pairwise matches.
pub fn ratings(
    direction: Direction,
    elo: &EloConfig,
    methods: &TaskResults,
) -> BTreeMap<String, f64> {
    let method_ids: Vec<_> = methods.keys().map(String::as_str).collect();
    if method_ids.is_empty() {
        return BTreeMap::new();
    }
    let method_index: BTreeMap<_, _> = method_ids
        .iter()
        .enumerate()
        .map(|(index, method)| (*method, index))
        .collect();
    let mut by_dataset: BTreeMap<&str, Vec<(&str, f64)>> = BTreeMap::new();
    for (method, results) in methods {
        for (dataset, value) in results {
            by_dataset
                .entry(dataset)
                .or_default()
                .push((method, *value));
        }
    }

    let mut pairs: BTreeMap<(usize, usize), (f64, f64)> = BTreeMap::new();
    for scores in by_dataset.values() {
        for left in 0..scores.len() {
            for right in (left + 1)..scores.len() {
                let (a, av) = scores[left];
                let (b, bv) = scores[right];
                let entry = pairs
                    .entry((method_index[a], method_index[b]))
                    .or_insert((0.0, 0.0));
                entry.0 += 1.0;
                entry.1 += outcome(av, bv, direction, elo.tie_tolerance);
            }
        }
    }

    let pair_rows: Vec<_> = pairs
        .into_iter()
        .map(|((left, right), (count, sum))| (left, right, sum / count, count))
        .collect();
    let mut ratings = vec![elo.initial_rating; method_ids.len()];
    let elo_scale = elo_scale();
    for _ in 0..elo.iterations {
        let mut gradient = vec![0.0; method_ids.len()];
        let mut information = vec![0.0; method_ids.len()];
        for &(left, right, observed, weight) in &pair_rows {
            let delta = (elo_scale * (ratings[left] - ratings[right])).clamp(-700.0, 700.0);
            let probability = 1.0 / (1.0 + (-delta).exp());
            let residual = (observed - probability) * weight;
            let curvature = probability * (1.0 - probability) * weight;
            gradient[left] += residual;
            gradient[right] -= residual;
            information[left] += curvature;
            information[right] += curvature;
        }
        let mut largest_update = 0.0_f64;
        for method in 0..method_ids.len() {
            let regularized_gradient = elo_scale * gradient[method]
                - elo.regularization * (ratings[method] - elo.initial_rating);
            let regularized_information =
                elo_scale * elo_scale * information[method] + elo.regularization;
            let update =
                (regularized_gradient / regularized_information).clamp(-elo.k_factor, elo.k_factor);
            ratings[method] += update;
            largest_update = largest_update.max(update.abs());
        }
        if largest_update < elo.convergence_tolerance {
            break;
        }
    }
    let mean = ratings.iter().sum::<f64>() / ratings.len() as f64;
    let shift = elo.initial_rating - mean;
    method_ids
        .into_iter()
        .zip(ratings)
        .map(|(method, rating)| (method.to_owned(), rating + shift))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(rows: &[(&str, &str, f64)]) -> BTreeMap<String, BTreeMap<String, f64>> {
        let mut out: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
        for (dataset, method, value) in rows {
            out.entry((*dataset).to_owned())
                .or_default()
                .insert((*method).to_owned(), *value);
        }
        out
    }

    /// `tests/test_elo.py::test_better_method_gets_higher_rating`.
    #[test]
    fn better_method_gets_higher_rating() {
        let ratings = fit_elo(&matches_from_scores(
            &scores(&[
                ("a", "strong", 1.0),
                ("a", "weak", 2.0),
                ("b", "strong", 2.0),
                ("b", "weak", 3.0),
            ]),
            TIE_RELATIVE_TOLERANCE,
        ));
        assert!(ratings["strong"] > ratings["weak"]);
        let mean = ratings.values().sum::<f64>() / ratings.len() as f64;
        assert_eq!(mean, 1500.0);
    }

    /// `tests/test_elo.py::test_ties_remain_centered`.
    #[test]
    fn ties_remain_centered() {
        let ratings = fit_elo(&matches_from_scores(
            &scores(&[("a", "left", 1.0), ("a", "right", 1.0)]),
            TIE_RELATIVE_TOLERANCE,
        ));
        assert_eq!(ratings["left"], 1500.0);
        assert_eq!(ratings["right"], 1500.0);
    }

    #[test]
    fn matches_follow_sorted_upper_triangle_order_with_relative_ties() {
        let matches = matches_from_scores(
            &scores(&[
                ("z", "m2", 0.5),
                ("z", "m1", 0.5 + 1e-14),
                ("a", "m1", 2.0),
                ("a", "m3", 1.0),
                ("a", "m2", 3.0),
            ]),
            TIE_RELATIVE_TOLERANCE,
        );
        let flat: Vec<(&str, &str, &str, f64)> = matches
            .iter()
            .map(|m| {
                (
                    m.dataset_id.as_str(),
                    m.method_a.as_str(),
                    m.method_b.as_str(),
                    m.outcome_a,
                )
            })
            .collect();
        assert_eq!(
            flat,
            vec![
                ("a", "m1", "m2", 1.0),
                ("a", "m1", "m3", 0.0),
                ("a", "m2", "m3", 0.0),
                ("z", "m1", "m2", 0.5),
            ]
        );
    }

    #[test]
    fn methods_without_a_match_have_no_rating() {
        let ratings = fit_elo(&matches_from_scores(
            &scores(&[("a", "solo", 1.0), ("b", "x", 1.0), ("b", "y", 2.0)]),
            TIE_RELATIVE_TOLERANCE,
        ));
        assert_eq!(ratings.len(), 2);
        assert!(!ratings.contains_key("solo"));
    }

    #[test]
    fn pairwise_sum_matches_numpy_structure() {
        let small: Vec<f64> = (1..=7).map(f64::from).collect();
        assert_eq!(numpy_pairwise_sum(&small), 28.0);
        let block: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(numpy_pairwise_sum(&block), 5050.0);
        let big: Vec<f64> = (1..=305).map(|i| f64::from(i) * 0.1).collect();
        let naive: f64 = big.iter().sum();
        assert!((numpy_pairwise_sum(&big) - naive).abs() < 1e-9);
        assert_eq!(numpy_pairwise_sum(&[]), 0.0);
    }

    /// The two fits are the same estimator: ratings agree to floating-point noise.
    #[test]
    fn aggregated_and_per_match_fits_agree() {
        let mut rows = Vec::new();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for dataset in 0..30 {
            for method in 0..12 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let noise = (state % 1000) as f64 / 1000.0;
                rows.push((
                    format!("d{dataset:02}"),
                    format!("m{method:02}"),
                    method as f64 * 0.05 + noise,
                ));
            }
        }
        let by_dataset = scores(
            &rows
                .iter()
                .map(|(d, m, v)| (d.as_str(), m.as_str(), *v))
                .collect::<Vec<_>>(),
        );
        let per_match = fit_elo(&matches_from_scores(&by_dataset, TIE_RELATIVE_TOLERANCE));
        let mut by_method: TaskResults = BTreeMap::new();
        for (dataset, method, value) in &rows {
            by_method
                .entry(method.clone())
                .or_default()
                .insert(dataset.clone(), *value);
        }
        let aggregated = ratings(Direction::Minimize, &EloConfig::default(), &by_method);
        for (method, rating) in &per_match {
            assert!(
                (rating - aggregated[method]).abs() < 1e-6,
                "{method}: {rating} vs {}",
                aggregated[method]
            );
        }
        assert!(per_match["m00"] > per_match["m11"]);
    }

    /// The reporter's `direction_and_tolerance_are_deterministic`.
    #[test]
    fn direction_and_tolerance_are_deterministic() {
        assert_eq!(outcome(0.9, 0.8, Direction::Maximize, 1e-12), 1.0);
        assert_eq!(outcome(0.9, 0.8, Direction::Minimize, 1e-12), 0.0);
        assert_eq!(outcome(1.0, 1.0 + 1e-13, Direction::Minimize, 1e-12), 0.5);
        assert!(metric_better(0.1, 0.2, Direction::Minimize));
        assert!(!metric_better(0.1, 0.2, Direction::Maximize));
    }
}

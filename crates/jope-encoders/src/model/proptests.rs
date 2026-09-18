//! Property tests for the pure math helpers behind Starforge inference.
//!
//! The ensemble heads promise row-stable scoring (a query row scores identically
//! alone or inside any batch), so the aggregation helpers must hold structural
//! invariants for *any* input, not just the fixtures in the unit tests. This
//! module proves those invariants with `proptest` over randomized inputs:
//!
//! - `median_of_views`: view-order (permutation) invariance, identity when all
//!   views agree, row-count preservation, per-row bounds, and single-vs-batch
//!   row stability.
//! - `average_ranks_one_based`: rank-sum conservation, one-based range, and
//!   permutation equivariance (the fix for the historical rank-tie leak).
//! - `regression_query_chunk`: chunk bounds and no-op behavior when the request
//!   already fits the row budget.
//! - `tabicl_common` shape/softmax helpers: `split_heads`/`merge_heads`
//!   round-trip bit-exactly and `softmax_last` yields shift-invariant
//!   probability rows.

use super::tabicl_common::{merge_heads, softmax_last, split_heads};
use super::{
    average_ranks_one_based, median_of_views, regression_query_chunk, REGRESSION_MAX_TOTAL_ROWS,
};
use candle_core::{Device, Tensor};
use proptest::prelude::*;

/// Exactly 6 feature views (the ensemble contract) with a shared row count.
fn six_views() -> impl Strategy<Value = Vec<Vec<f64>>> {
    (1usize..24).prop_flat_map(|rows| {
        proptest::collection::vec(proptest::collection::vec(-1e6f64..1e6, rows), 6)
    })
}

/// A permutation of `0..n` derived from random sort keys (argsort).
fn permutation(n: usize) -> impl Strategy<Value = Vec<usize>> {
    proptest::collection::vec(any::<u64>(), n).prop_map(|keys| {
        let mut order: Vec<usize> = (0..keys.len()).collect();
        order.sort_by_key(|&i| keys[i]);
        order
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn median_of_views_ignores_view_order_and_preserves_rows(
        (views, perm) in six_views().prop_flat_map(|views| {
            let n_views = views.len();
            (Just(views), permutation(n_views))
        }),
    ) {
        let rows = views[0].len();
        let base = median_of_views(&views, rows);
        // Row-count preservation: one aggregated score per query row.
        prop_assert_eq!(base.len(), rows);
        // Permutation invariance: the median only sees each row's multiset of
        // view values, so reordering the views must not move a single bit.
        let shuffled: Vec<Vec<f64>> = perm.iter().map(|&v| views[v].clone()).collect();
        prop_assert_eq!(median_of_views(&shuffled, rows), base.clone());
        // Per-row bounds: a median never escapes that row's own view values.
        for (row, agg) in base.iter().enumerate() {
            let lo = views.iter().map(|v| v[row]).fold(f64::INFINITY, f64::min);
            let hi = views.iter().map(|v| v[row]).fold(f64::NEG_INFINITY, f64::max);
            prop_assert!((lo..=hi).contains(agg), "row {} median {} outside [{}, {}]", row, agg, lo, hi);
        }
    }

    #[test]
    fn median_of_views_is_identity_when_all_views_agree(
        row_values in proptest::collection::vec(-1e6f64..1e6, 1..24),
    ) {
        // 6 identical views: the aggregate must reproduce the view exactly
        // (0.5 * (x + x) is bit-exact for finite doubles in range).
        let views: Vec<Vec<f64>> = (0..6).map(|_| row_values.clone()).collect();
        prop_assert_eq!(median_of_views(&views, row_values.len()), row_values);
    }

    #[test]
    fn median_of_views_scores_rows_identically_alone_and_in_batch(views in six_views()) {
        // The row-stability contract behind commit "aggregate classifier feature
        // views by per-row median": batch membership must not change a score.
        let rows = views[0].len();
        let batched = median_of_views(&views, rows);
        for row in 0..rows {
            let alone: Vec<Vec<f64>> = views.iter().map(|view| vec![view[row]]).collect();
            prop_assert_eq!(median_of_views(&alone, 1)[0], batched[row]);
        }
    }

    #[test]
    fn average_ranks_conserve_sum_range_and_permutation_equivariance(
        (values, perm) in proptest::collection::vec(-1e3f64..1e3, 1..40)
            .prop_flat_map(|values| {
                let n = values.len();
                (Just(values), permutation(n))
            }),
    ) {
        let n = values.len();
        let ranks = average_ranks_one_based(&values);
        prop_assert_eq!(ranks.len(), n);
        // Tie-averaging redistributes ranks inside a tie group but never
        // creates or destroys rank mass: the sum stays 1 + 2 + ... + n.
        let sum: f64 = ranks.iter().sum();
        let expected = (n * (n + 1)) as f64 * 0.5;
        prop_assert!((sum - expected).abs() < 1e-9, "rank sum {} != {}", sum, expected);
        for r in &ranks {
            prop_assert!((1.0..=n as f64).contains(r));
        }
        // Equivariance: a rank depends only on the value and the multiset, so
        // permuting the input permutes the ranks the same way (average ties
        // make this exact; this is the guard against the rank-tie leak).
        let shuffled: Vec<f64> = perm.iter().map(|&v| values[v]).collect();
        let shuffled_ranks = average_ranks_one_based(&shuffled);
        for (i, &v) in perm.iter().enumerate() {
            prop_assert_eq!(shuffled_ranks[i], ranks[v]);
        }
    }

    #[test]
    fn regression_query_chunk_is_bounded_and_lossless_when_fitting(
        n_context in 0usize..4096,
        n_query in 0usize..4096,
    ) {
        let chunk = regression_query_chunk(n_context, n_query);
        // Always a usable, bounded chunk: at least one row, never more than
        // the request itself.
        prop_assert!(chunk >= 1);
        prop_assert!(chunk <= n_query.max(1));
        // When context + query already fit the row budget there is nothing to
        // chunk: the request passes through whole.
        if n_query >= 1 && n_context + n_query <= REGRESSION_MAX_TOTAL_ROWS {
            prop_assert_eq!(chunk, n_query);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn split_then_merge_heads_round_trips_bit_exactly(
        (dims, nhead, data) in (1usize..3, 1usize..5, 1usize..4, 1usize..5)
            .prop_flat_map(|(b, t, nhead, head_dim)| {
                let embed = nhead * head_dim;
                (
                    Just((b, t, embed)),
                    Just(nhead),
                    proptest::collection::vec(-10f32..10.0, b * t * embed),
                )
            }),
    ) {
        let (b, t, embed) = dims;
        let dev = Device::Cpu;
        let x = Tensor::from_vec(data.clone(), (b, t, embed), &dev).unwrap();
        let split = split_heads(&x, nhead).unwrap();
        // Shape contract: (..., T, E) -> (..., nhead, T, E / nhead).
        prop_assert_eq!(split.dims(), &[b, nhead, t, embed / nhead]);
        // Transpose + reshape only move values, so the inverse restores every
        // byte of the original tensor.
        let merged = merge_heads(&split).unwrap();
        prop_assert_eq!(merged.dims(), &[b, t, embed]);
        let round_trip: Vec<f32> = merged.flatten_all().unwrap().to_vec1().unwrap();
        prop_assert_eq!(round_trip, data);
    }

    #[test]
    fn softmax_last_rows_are_shift_invariant_probability_distributions(
        (dims, data) in (1usize..3, 1usize..4, 2usize..6)
            .prop_flat_map(|(b, t, c)| {
                (Just((b, t, c)), proptest::collection::vec(-30f32..30.0, b * t * c))
            }),
    ) {
        let (b, t, c) = dims;
        let dev = Device::Cpu;
        let x = Tensor::from_vec(data, (b, t, c), &dev).unwrap();
        let sm: Vec<f32> = softmax_last(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for row in sm.chunks(c) {
            // Each last-dim row is a probability distribution...
            let sum: f32 = row.iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-4, "row sums to {}", sum);
            for p in row {
                prop_assert!(*p > 0.0 && *p <= 1.0 + 1e-6);
            }
        }
        // ...and the max-subtraction stabilizer makes it shift invariant:
        // adding a constant to the logits must not move the probabilities.
        let shifted: Vec<f32> = softmax_last(&x.affine(1.0, 7.5).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (a, s) in sm.iter().zip(&shifted) {
            prop_assert!((a - s).abs() < 1e-5, "shift moved {} -> {}", a, s);
        }
    }
}

use feat_core::metrics::{auc, logloss, rmse};
use proptest::prelude::*;

proptest! {
    #[test]
    fn auc_is_invariant_under_reversal(scores in prop::collection::vec(-1000.0f32..1000.0, 2..128)) {
        let n = scores.len();
        let labels: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let mut rev_labels = labels.clone();
        let mut rev_scores = scores.clone();
        rev_labels.reverse();
        rev_scores.reverse();

        prop_assert!((auc(&labels, &scores) - auc(&rev_labels, &rev_scores)).abs() < 1e-9);
    }

    #[test]
    fn perfect_regression_has_zero_rmse(values in prop::collection::vec(-1000.0f32..1000.0, 1..128)) {
        prop_assert!(rmse(&values, &values) <= 1e-12);
    }

    #[test]
    fn logloss_is_finite_for_probabilities(labels in prop::collection::vec(0u8..2, 1..128), probs in prop::collection::vec(0.0f32..1.0, 1..128)) {
        let n = labels.len().min(probs.len());
        let y: Vec<f32> = labels.into_iter().take(n).map(|v| v as f32).collect();
        let p: Vec<f32> = probs.into_iter().take(n).collect();
        prop_assert!(logloss(&y, &p).is_finite());
    }
}

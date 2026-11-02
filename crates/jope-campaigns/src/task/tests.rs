use super::*;
use std::collections::BTreeSet;

#[test]
fn one_registry_owns_all_task_names_directories_and_metrics() {
    assert_eq!(
        Task::CANONICAL_ORDER.map(Task::as_str),
        ["regression", "binary", "multiclass", "timeseries",]
    );
    assert_eq!(Task::Timeseries.corpus_dir_name(), "timeseries_reduced_v2");
    assert_eq!(Task::Multiclass.corpus_dir_name(), "multi_label");
    assert_eq!(Task::Timeseries.primary_metric(), "negative_mase");
    assert_eq!(Task::Multiclass.native_error_metric(), "one_minus_macro_f1");
    assert_eq!(Task::Regression.reward_expression(), "clamp(r2,-1,1)");
    assert_eq!(Task::Binary.reward_expression(), "clamp(auc,-1,1)");
    assert_eq!(Task::parse("multi_label"), None);
}

#[test]
fn legacy_derived_order_is_frozen_separately_from_canonical_order() {
    let ordered = [
        Task::Multiclass,
        Task::Timeseries,
        Task::Regression,
        Task::Binary,
    ]
    .into_iter()
    .collect::<BTreeSet<_>>()
    .into_iter()
    .collect::<Vec<_>>();
    assert_eq!(
        ordered,
        [
            Task::Binary,
            Task::Regression,
            Task::Timeseries,
            Task::Multiclass,
        ]
    );
    assert_eq!(
        Task::CANONICAL_ORDER,
        [
            Task::Regression,
            Task::Binary,
            Task::Multiclass,
            Task::Timeseries,
        ]
    );
}

#[test]
fn typed_scores_reject_wrong_domains_and_timeseries_parity_is_zero() {
    assert!(NativeScore::<AucMetric>::new(1.01).is_err());
    assert!(NativeScore::<MacroF1Metric>::new(-0.01).is_err());
    assert!(NativeScore::<NegativeMaseMetric>::new(0.01).is_err());
    assert_eq!(normalize_reward(Task::Timeseries, -1.0).unwrap().get(), 0.0);
}

#[test]
fn embedded_and_rust_task_semantics_match() {
    verify_embedded_task_semantics().unwrap();
}

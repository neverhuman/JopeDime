//! The roster the appliance offers is the frozen panel's: 17 regression recipes over 16
//! families, five binary and four multiclass, every recipe a registered head and valid for its
//! task. The benchmark adapter's own test (`jope-bench` panel::methods) proves it re-exports
//! this registry rather than a copy.
use jope_runtime::methods::{records, registry, resolve, AUTO};
use jope_runtime::task::TaskKind;
use std::collections::BTreeSet;

#[test]
fn roster_counts_families_and_aliases() {
    let regression = registry::rust_methods(TaskKind::Regression).unwrap();
    assert_eq!(regression.len(), 17);
    assert_eq!(
        regression
            .iter()
            .map(|m| &m.family)
            .collect::<BTreeSet<_>>()
            .len(),
        16
    );
    assert_eq!(registry::rust_methods(TaskKind::Binary).unwrap().len(), 5);
    assert_eq!(
        registry::rust_methods(TaskKind::Multiclass).unwrap().len(),
        4
    );
    assert!(registry::rust_methods(TaskKind::Timeseries).is_err());
    for head in [
        "ridge",
        "ridge_gcv",
        "elasticnet",
        "histgbm",
        "histgbm2",
        "krr",
        "gpr",
        "ksmooth",
        "hurdle2",
        "stack",
        "portfolio",
    ] {
        assert!(feat_math::heads::is_head_op(head));
        assert!(regression.iter().any(|m| m.family == head), "{head}");
    }
    assert_eq!(
        resolve(TaskKind::Regression, "gbm").unwrap().id,
        "rust.histgbm"
    );
    assert_eq!(resolve(TaskKind::Regression, "gpr").unwrap().id, "rust.gpr");
    assert_eq!(
        resolve(TaskKind::Regression, "gpr.rbf_linear").unwrap().id,
        "rust.gpr.rbf_linear"
    );
    assert!(resolve(TaskKind::Regression, "gpr.rbf")
        .unwrap_err()
        .to_string()
        .contains("no regression method"));
    assert_eq!(
        resolve(TaskKind::Regression, "rust.gpr.rbf_linear")
            .unwrap()
            .configuration,
        "rbf_linear"
    );
    assert_eq!(
        resolve(TaskKind::Binary, "logistic").unwrap().id,
        "rust.logit_ridge"
    );
    assert!(resolve(TaskKind::Multiclass, "ridge").is_err());
}

#[test]
fn records_distinguish_native_and_uncompiled_neural_methods() {
    let records = records().unwrap();
    assert_eq!(records.iter().filter(|r| r.kind == "native").count(), 26);
    for record in &records {
        assert_eq!(record.executable, record.kind == "native", "{}", record.id);
        assert_eq!(record.reason.is_some(), !record.executable);
    }
    let neural = resolve(TaskKind::Regression, "proposal.step60000.regression").unwrap_err();
    assert_eq!(
        jope_runtime::error::kind_of(&neural),
        jope_runtime::FailureKind::Backend
    );
    let missing = jope_runtime::methods::unexecutable(&[
        "rust.ridge".into(),
        AUTO.into(),
        "neural.chimera.binary".into(),
        "rust.unknown".into(),
    ])
    .unwrap();
    assert_eq!(
        missing
            .iter()
            .map(|(id, _)| id.as_str())
            .collect::<Vec<_>>(),
        ["neural.chimera.binary", "rust.unknown"]
    );
}

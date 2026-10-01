use feat_math::kernel::capabilities::legacy_recipe_axis;
use feat_math::kernel::ordered_axis::{
    admit_legacy_genome, Axis, Causality, FitScope, FoldBoundary, FoldPhase, GroupKey,
    OrderedInputIdentity, StageAxis, TaskMode, ORDERED_AXIS_VERSION,
};
use feat_math::ops::{apply_op, FittedOp};
use feat_math::{ColumnSpec, DataSchema, GenomeIr, Objective, RawTable, RowSemantics, TaskSpec};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct PoisonFixture {
    cut_id: String,
    train_end: usize,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    poisoned_rows: Vec<Vec<String>>,
}

fn fixture() -> PoisonFixture {
    serde_json::from_str(include_str!(
        "fixtures/ordered_axis/held_out_poison.v1.json"
    ))
    .unwrap()
}

fn ordered_schema() -> DataSchema {
    DataSchema {
        columns: vec![
            ColumnSpec::categorical("group"),
            ColumnSpec::numeric("time"),
            ColumnSpec::numeric("value"),
        ],
        row_semantics: RowSemantics::Ordered {
            group_by: vec!["group".into()],
            order_by: vec!["time".into()],
        },
    }
}

fn iid_schema() -> DataSchema {
    DataSchema {
        columns: vec![ColumnSpec::numeric("value")],
        row_semantics: RowSemantics::Iid,
    }
}

fn causal_stage() -> StageAxis {
    StageAxis::new(Axis::SampleTime, Causality::Causal, FitScope::TrainFold)
}

#[test]
fn legacy_lag_roll_are_within_row_and_keep_frozen_bits() {
    let task = TaskSpec::regression(Objective::Rmse).unwrap();
    let schema = iid_schema();
    for name in ["lag", "roll"] {
        let metadata = legacy_recipe_axis(name).unwrap();
        assert_eq!(metadata.axis, Axis::FeatureWithinRow);
        assert_eq!(metadata.fit_scope, FitScope::Stateless);
        task.admit_ordered_stage(&schema, metadata).unwrap();
    }
    let genome = GenomeIr::parse("(pipe X (lag k=1) (roll k=1) (ridge))").unwrap();
    admit_legacy_genome(&genome, &task, &schema).unwrap();

    let rows = vec![vec![1.0, -0.0, 3.0], vec![4.0, 5.0, 6.0]];
    let lag = apply_op(
        &FittedOp::Lag {
            k: 1,
            append: false,
        },
        &rows,
    )
    .unwrap();
    let roll = apply_op(
        &FittedOp::Roll {
            k: 1,
            append: false,
        },
        &rows,
    )
    .unwrap();
    assert_eq!(
        lag[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [0.0f64.to_bits(), 1.0f64.to_bits(), (-0.0f64).to_bits()]
    );
    assert_eq!(
        roll[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [3.0f64.to_bits(), 1.0f64.to_bits(), (-0.0f64).to_bits()]
    );
}

#[test]
fn metadata_is_versioned_and_unknown_versions_fail_closed() {
    let metadata = causal_stage();
    assert_eq!(metadata.version, ORDERED_AXIS_VERSION);
    let json = serde_json::to_string(&metadata).unwrap();
    assert!(json.contains("\"sample_time\""));
    assert!(json.contains("\"train_fold\""));
    assert_eq!(serde_json::from_str::<StageAxis>(&json).unwrap(), metadata);
    let mut unknown = metadata;
    unknown.version += 1;
    assert!(unknown
        .admit(
            TaskMode::Ordered,
            Some(&OrderedInputIdentity::new("group", "time"))
        )
        .unwrap_err()
        .to_string()
        .contains("version"));
}

#[test]
fn sample_time_is_refused_for_iid_and_without_ordered_identity() {
    let metadata = causal_stage();
    let task = TaskSpec::regression(Objective::Rmse).unwrap();
    assert!(metadata.admit(TaskMode::IID, None).is_err());
    assert!(task
        .admit_ordered_stage(&iid_schema(), metadata)
        .unwrap_err()
        .to_string()
        .contains("ordered row semantics"));
    assert!(metadata.admit(TaskMode::Ordered, None).is_err());
    assert!(task
        .admit_ordered_stage(&ordered_schema(), metadata)
        .unwrap()
        .is_some());
}

#[test]
fn offline_stage_is_refused_for_forecasting() {
    let offline = StageAxis::new(Axis::SampleTime, Causality::Offline, FitScope::TrainFold);
    let forecast = TaskSpec::timeseries(Objective::Rmse).unwrap();
    assert!(forecast
        .admit_ordered_stage(&ordered_schema(), offline)
        .unwrap_err()
        .to_string()
        .contains("offline stage cannot enter forecasting"));
    assert!(TaskSpec::regression(Objective::Rmse)
        .unwrap()
        .admit_ordered_stage(&ordered_schema(), offline)
        .is_ok());
}

#[test]
fn input_identity_requires_one_group_and_one_numeric_timestamp() {
    let schema = ordered_schema();
    let identity = schema.ordered_input_identity_v1().unwrap();
    assert_eq!(identity.group_key_column, "group");
    assert_eq!(identity.timestamp_column, "time");
    assert_eq!(identity.version, ORDERED_AXIS_VERSION);

    let mut bad = schema.clone();
    bad.row_semantics = RowSemantics::Ordered {
        group_by: vec![],
        order_by: vec!["time".into()],
    };
    assert!(bad.ordered_input_identity_v1().is_err());
    bad.row_semantics = RowSemantics::Ordered {
        group_by: vec!["group".into()],
        order_by: vec!["time".into(), "value".into()],
    };
    assert!(bad.ordered_input_identity_v1().is_err());
    bad.row_semantics = schema.row_semantics;
    bad.columns[1] = ColumnSpec::categorical("time");
    assert!(bad.ordered_input_identity_v1().is_err());
}

fn probe(
    table: &RawTable,
    admissions: &[feat_math::kernel::ordered_axis::RowAdmission],
) -> (f64, Vec<f64>) {
    let mut history = BTreeMap::<GroupKey, f64>::new();
    let mut fitted_sum = 0.0;
    let mut before = Vec::new();
    for (row, admitted) in table.rows.iter().zip(admissions) {
        let value = row[2].parse::<f64>().unwrap();
        let prior = *history.get(&admitted.group).unwrap_or(&0.0);
        if admitted.group_reset {
            assert_eq!(prior, 0.0);
        }
        before.push(prior);
        if admitted.may_fit {
            fitted_sum += value;
        }
        // A causal stage emits from `prior` before this observation is carried.
        *history.entry(admitted.group.clone()).or_default() += value;
    }
    (fitted_sum, before)
}

#[test]
fn fold_certificate_seals_fit_and_carries_only_prior_group_state() {
    let fixture = fixture();
    let schema = ordered_schema();
    let task = TaskSpec::timeseries(Objective::Rmse).unwrap();
    let clean = RawTable::new(fixture.headers.clone(), fixture.rows).unwrap();
    let poison = RawTable::new(fixture.headers, fixture.poisoned_rows).unwrap();
    let fold = FoldBoundary::new(fixture.cut_id, fixture.train_end, clean.rows.len()).unwrap();
    assert_eq!(fold.fit_rows(), 0..4);
    let clean_rows = schema
        .certify_ordered_fold(&task, causal_stage(), &clean, fold.clone())
        .unwrap();
    let poisoned_rows = schema
        .certify_ordered_fold(&task, causal_stage(), &poison, fold)
        .unwrap();
    assert_eq!(clean_rows, poisoned_rows);
    assert_eq!(clean_rows.len(), 8);
    assert_eq!(clean_rows.iter().filter(|row| row.may_fit).count(), 4);
    assert_eq!(clean_rows[3].prior_in_group, 2); // equal A timestamps retain input order
    assert_eq!(clean_rows[4].phase, FoldPhase::HeldOut);
    assert_eq!(clean_rows[4].prior_train_in_group, 3);
    assert!(!clean_rows[4].group_reset);
    assert!(clean_rows[6].group_reset); // held-out group C has no A/B history
    assert_eq!(clean_rows[6].prior_in_group, 0);

    let (clean_fit, clean_before) = probe(&clean, &clean_rows);
    let (poison_fit, poison_before) = probe(&poison, &poisoned_rows);
    assert_eq!(clean_fit.to_bits(), poison_fit.to_bits());
    assert_eq!(clean_fit, 151.0);
    assert_eq!(&clean_before[..7], &poison_before[..7]);
    assert_eq!(clean_before[4], 51.0);
    assert_eq!(clean_before[5], 100.0);
    assert_eq!(clean_before[6], 0.0);
    assert_ne!(clean_before[7], poison_before[7]); // later held-out rows may see prior observations
}

#[test]
fn malformed_order_and_fold_boundaries_are_rejected() {
    let schema = ordered_schema();
    let task = TaskSpec::timeseries(Objective::Rmse).unwrap();
    let headers = vec!["group".into(), "time".into(), "value".into()];
    let reverse = RawTable::new(
        headers.clone(),
        vec![
            vec!["A".into(), "2".into(), "1".into()],
            vec!["A".into(), "1".into(), "2".into()],
        ],
    )
    .unwrap();
    let fold = FoldBoundary::new("cut", 1, 2).unwrap();
    assert!(schema
        .certify_ordered_fold(&task, causal_stage(), &reverse, fold.clone())
        .unwrap_err()
        .to_string()
        .contains("monotone"));
    let nonfinite =
        RawTable::new(headers, vec![vec!["A".into(), "NaN".into(), "1".into()]]).unwrap();
    assert!(schema
        .certify_ordered_fold(
            &task,
            causal_stage(),
            &nonfinite,
            FoldBoundary::new("cut", 1, 1).unwrap()
        )
        .is_err());
    assert!(FoldBoundary::new("", 1, 2).is_err());
    assert!(FoldBoundary::new("cut", 0, 2).is_err());
    assert!(FoldBoundary::new("cut", 3, 2).is_err());
    assert!(schema
        .certify_ordered_fold(
            &task,
            causal_stage(),
            &reverse,
            FoldBoundary::new("cut", 1, 3).unwrap()
        )
        .is_err());
}

#[test]
fn numeric_group_keys_canonicalize_zero_and_equal_spellings() {
    assert_eq!(
        GroupKey::numeric(-0.0).unwrap(),
        GroupKey::numeric(0.0).unwrap()
    );
    assert_eq!(
        GroupKey::numeric(1.0).unwrap(),
        GroupKey::numeric(1.0).unwrap()
    );
    let schema = DataSchema {
        columns: vec![ColumnSpec::numeric("group"), ColumnSpec::numeric("time")],
        row_semantics: RowSemantics::Ordered {
            group_by: vec!["group".into()],
            order_by: vec!["time".into()],
        },
    };
    let table = RawTable::new(
        vec!["group".into(), "time".into()],
        vec![vec!["1".into(), "1".into()], vec!["1.0".into(), "2".into()]],
    )
    .unwrap();
    let rows = schema
        .certify_ordered_fold(
            &TaskSpec::timeseries(Objective::Rmse).unwrap(),
            causal_stage(),
            &table,
            FoldBoundary::new("cut", 1, 2).unwrap(),
        )
        .unwrap();
    assert!(rows[0].group_reset);
    assert!(!rows[1].group_reset);
    assert_eq!(rows[1].prior_train_in_group, 1);
}

//! Public-boundary deterministic property lane for the current v3 prototype.
//!
//! These checks intentionally avoid asserting planned K1 behavior. They cover
//! invariants already promised by the public API: deterministic fitting,
//! prediction and emission, stable schema identity, and balanced deterministic
//! fold assignment.

use feat_math::advanced_v3::stratified_fold_assignments;
use feat_math::transpile::EmitOptions;
use feat_math::{
    emit_bound, fit_table, predict_table, ColumnSpec, DataSchema, GenomeIr, Objective, RawTable,
    RowSemantics, TaskSpec,
};

fn schema() -> DataSchema {
    DataSchema {
        columns: vec![
            ColumnSpec::numeric("amount"),
            ColumnSpec::categorical("country"),
        ],
        row_semantics: RowSemantics::Iid,
    }
}

fn training_table() -> RawTable {
    RawTable::new(
        vec!["amount".into(), "country".into()],
        vec![
            vec!["0".into(), "US".into()],
            vec!["0.1".into(), "US".into()],
            vec!["0.2".into(), "CA".into()],
            vec!["0.3".into(), "CA".into()],
            vec!["1.0".into(), "FR".into()],
            vec!["1.1".into(), "FR".into()],
            vec!["1.2".into(), "DE".into()],
            vec!["".into(), "DE".into()],
        ],
    )
    .unwrap()
}

#[test]
fn public_fit_predict_and_emission_are_deterministic() {
    let schema = schema();
    let table = training_table();
    let task = TaskSpec::binary(Objective::LogLoss).unwrap();
    let genome = GenomeIr::parse(
        "(pipe X (input numeric=median missing=indicator categorical=frequency) (logit_ridge l2=0.1))",
    )
    .unwrap();
    let target = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    assert_eq!(schema.sha256().unwrap(), schema.sha256().unwrap());
    assert_eq!(
        RawTable::from_csv("a,b\n1,\"x,y\"\n").unwrap(),
        RawTable::from_csv("a,b\n1,\"x,y\"\n").unwrap()
    );

    let first = fit_table(&genome, &task, &schema, &table, &target).unwrap();
    let second = fit_table(&genome, &task, &schema, &table, &target).unwrap();
    let first_predictions = first.predict_table(&table).unwrap();
    let free_predictions = predict_table(&first, &table).unwrap();
    let second_predictions = second.predict_table(&table).unwrap();
    assert_eq!(first_predictions.len(), second_predictions.len());
    for ((first, free), second) in first_predictions
        .iter()
        .zip(&free_predictions)
        .zip(&second_predictions)
    {
        assert_eq!(first.to_bits(), free.to_bits());
        assert_eq!(first.to_bits(), second.to_bits());
        assert!((0.0..=1.0).contains(first));
    }

    let first_emission = emit_bound(&first, &EmitOptions::default()).unwrap();
    let second_emission = emit_bound(&second, &EmitOptions::default()).unwrap();
    assert_eq!(first_emission.python, second_emission.python);
    assert_eq!(first_emission.rust, second_emission.rust);
    assert_eq!(first_emission.manifest, second_emission.manifest);
}

#[test]
fn public_fold_assignments_are_repeatable_bounded_and_balanced() {
    let binary_target = [
        0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    ];
    for folds in 2..=7 {
        let assignments = stratified_fold_assignments(&binary_target, folds);
        assert_eq!(
            assignments,
            stratified_fold_assignments(&binary_target, folds)
        );
        assert!(assignments.iter().all(|fold| *fold < folds));

        for class in [0.0, 1.0] {
            let mut counts = vec![0usize; folds];
            for (row, target) in binary_target.iter().enumerate() {
                if *target == class {
                    counts[assignments[row]] += 1;
                }
            }
            let minimum = counts.iter().min().unwrap();
            let maximum = counts.iter().max().unwrap();
            assert!(maximum - minimum <= 1, "folds={folds} class={class}");
        }
    }

    let regression_target = [0.1, 0.4, -0.2, 0.8, 1.3, -0.7, 2.0, 0.0, 0.6];
    for folds in 2..=7 {
        let assignments = stratified_fold_assignments(&regression_target, folds);
        let expected: Vec<usize> = (0..regression_target.len())
            .map(|row| row % folds)
            .collect();
        assert_eq!(assignments, expected);
    }
}

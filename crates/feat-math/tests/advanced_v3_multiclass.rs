use feat_math::{
    fit_table, macro_f1, multiclass_accuracy, predict_labels_table, predict_multiclass_table,
    predict_probabilities_table, predict_table, ColumnSpec, DataSchema, GenomeIr, Objective,
    RawTable, RowSemantics, TaskSpec, MULTICLASS_MAX_TRANSFORMED_FEATURES,
};

fn fixture(
    classes: usize,
    rows_per_class: usize,
    features: usize,
) -> (DataSchema, RawTable, Vec<f64>) {
    let schema = DataSchema {
        columns: (0..features)
            .map(|feature| ColumnSpec::numeric(format!("f{feature}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for class in 0..classes {
        for repeat in 0..rows_per_class {
            rows.push(
                (0..features)
                    .map(|feature| {
                        let value = if feature == class % features {
                            2.0
                        } else {
                            ((class * 17 + feature * 7 + repeat) % 19) as f64 / 50.0
                        };
                        value.to_string()
                    })
                    .collect(),
            );
            labels.push(class as f64);
        }
    }
    let table = RawTable {
        headers: schema
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect(),
        rows,
    };
    (schema, table, labels)
}

fn genome() -> GenomeIr {
    GenomeIr::parse(
        "(pipe X (input numeric=median) (standardize) \
         (multiclass_ridge class_weight=balanced l2=0.02))",
    )
    .unwrap()
}

fn genome_with_head(head: &str, feature_budget: usize) -> GenomeIr {
    GenomeIr::parse(&format!(
        "(pipe X (input numeric=median) (standardize) ({head} feature_budget={feature_budget}))"
    ))
    .unwrap()
}

#[test]
fn task_contract_accepts_only_multiclass_objectives() {
    assert!(TaskSpec::multiclass(Objective::MacroF1).is_ok());
    assert!(TaskSpec::multiclass(Objective::Accuracy).is_ok());
    assert!(TaskSpec::multiclass(Objective::RocAuc).is_err());
    assert!(TaskSpec::binary(Objective::MacroF1).is_err());
}

#[test]
fn aligned_probabilities_are_finite_normalized_and_deterministic() {
    for classes in [3usize, 12, 100] {
        let (schema, table, labels) = fixture(classes, 3, 8);
        let task = TaskSpec::multiclass(Objective::MacroF1).unwrap();
        let first = fit_table(&genome(), &task, &schema, &table, &labels).unwrap();
        let second = fit_table(&genome(), &task, &schema, &table, &labels).unwrap();
        let left = predict_multiclass_table(&first, &table).unwrap();
        let right = predict_multiclass_table(&second, &table).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.classes, (0..classes as i64).collect::<Vec<_>>());
        assert_eq!(left.labels.len(), table.rows.len());
        assert_eq!(left.probabilities.len(), table.rows.len());
        for row in &left.probabilities {
            assert_eq!(row.len(), classes);
            assert!(row.iter().all(|value| value.is_finite() && *value >= 0.0));
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        assert_eq!(predict_labels_table(&first, &table).unwrap(), left.labels);
        assert_eq!(
            predict_probabilities_table(&first, &table).unwrap(),
            (left.classes.clone(), left.probabilities.clone())
        );
        assert_eq!(
            predict_table(&first, &table).unwrap(),
            left.labels
                .iter()
                .map(|label| *label as f64)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn every_native_multiclass_head_is_probability_capable_and_deterministic() {
    let (schema, table, labels) = fixture(4, 3, 12);
    let task = TaskSpec::multiclass(Objective::MacroF1).unwrap();
    for head in [
        "multiclass_ridge",
        "multiclass_ovr_logistic",
        "multiclass_diagonal_lda",
        "multiclass_nearest_centroid",
    ] {
        for budget in [32, 64, 128, 256] {
            let genome = genome_with_head(head, budget);
            let first = fit_table(&genome, &task, &schema, &table, &labels).unwrap();
            let second = fit_table(&genome, &task, &schema, &table, &labels).unwrap();
            let left = predict_multiclass_table(&first, &table).unwrap();
            let right = predict_multiclass_table(&second, &table).unwrap();
            assert_eq!(left, right, "head={head} budget={budget}");
            assert!(left.probabilities.iter().all(|row| {
                row.len() == 4
                    && row.iter().all(|value| value.is_finite() && *value >= 0.0)
                    && (row.iter().sum::<f64>() - 1.0).abs() < 1e-12
            }));
        }
    }
}

#[test]
fn transformed_feature_width_is_deterministically_capped() {
    let (schema, table, labels) = fixture(3, 4, MULTICLASS_MAX_TRANSFORMED_FEATURES + 17);
    let task = TaskSpec::multiclass(Objective::MacroF1).unwrap();
    let model = fit_table(&genome(), &task, &schema, &table, &labels).unwrap();
    let prediction = model.predict_multiclass_table(&table).unwrap();
    assert_eq!(prediction.probabilities[0].len(), 3);
}

#[test]
fn labels_must_be_integer_with_three_covered_classes() {
    let (schema, table, mut labels) = fixture(3, 3, 4);
    let task = TaskSpec::multiclass(Objective::MacroF1).unwrap();
    labels[0] = 0.5;
    assert!(fit_table(&genome(), &task, &schema, &table, &labels)
        .unwrap_err()
        .to_string()
        .contains("integer"));

    let (two_schema, two_table, two_classes) = fixture(2, 3, 4);
    assert!(
        fit_table(&genome(), &task, &two_schema, &two_table, &two_classes)
            .unwrap_err()
            .to_string()
            .contains("at least three")
    );
}

#[test]
fn macro_f1_and_accuracy_match_the_hand_computed_fixture() {
    let truth = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
    let predicted = [0.0, 1.0, 1.0, 1.0, 2.0, 0.0];
    let expected_macro_f1 = (0.5 + 0.8 + 2.0 / 3.0) / 3.0;
    assert!((macro_f1(&truth, &predicted) - expected_macro_f1).abs() < 1e-12);
    assert!((multiclass_accuracy(&truth, &predicted) - 4.0 / 6.0).abs() < 1e-12);
}

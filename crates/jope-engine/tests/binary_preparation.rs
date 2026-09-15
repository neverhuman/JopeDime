use jope_parallel::prepare::{folds_with_seed, reconstruct_classification, Preprocessing, Table};
use serde_json::json;
use std::collections::BTreeSet;

#[test]
fn recorded_classification_mapping_recovers_exact_membership_and_raw_categories() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("original.csv");
    std::fs::write(&source, "colour,target\nred,2\nblue,1\n,2\n").unwrap();
    let m = json!({"source_path":source,"feature_columns":["colour"],"target_column":"target",
        "feature_encodings":{"colour":{"kind":"categorical_codes","mapping":{"__MISSING__":0,"blue":1,"red":2}}},
        "feature_scaling":{"min":[0],"max":[2]},"target_source_encoding":{"kind":"numeric"},
        "target_scaling":{"mode":"classification","n_classes":2,"class_to_code":{"1.0":1,"2.0":0},"scaled_code_formula":"code / max(1, n_classes - 1)"}});
    let table = Table {
        rows: vec![vec!["0.5".into(), "1".into()], vec!["0".into(), "0".into()]],
        width: 2,
        header: false,
    };
    let (raw, receipt) = reconstruct_classification(&table, &m).unwrap();
    assert_eq!(receipt["source_row_indices"], json!([1, 2]));
    assert_eq!(raw.rows, vec![vec!["blue", "1"], vec!["", "2"]]);
    let mut corrupted = m.clone();
    corrupted["target_scaling"]["class_to_code"]["2.0"] = json!(1);
    assert!(reconstruct_classification(&table, &corrupted).is_err());
    let mut corrupt_table = table.clone();
    corrupt_table.rows[0][1] = "0.3".into();
    assert!(reconstruct_classification(&corrupt_table, &m).is_err());
}

#[test]
fn repeated_binary_folds_are_deterministic_grouped_and_class_covering() {
    let mut rows = (0..30)
        .map(|i| vec![i.to_string(), (i % 2).to_string()])
        .collect::<Vec<_>>();
    rows.push(rows[0].clone());
    rows.push(vec!["0".into(), "1".into()]); // Conflicting duplicate features stay together.
    let groups = (0..rows.len())
        .map(|i| {
            if i == 2 || i == 4 {
                "same_subject".into()
            } else {
                format!("subject-{i}")
            }
        })
        .collect::<Vec<_>>();
    let cuts = folds_with_seed(&rows, "binary", 1, Some(&groups), 123).unwrap();
    assert_eq!(
        cuts,
        folds_with_seed(&rows, "binary", 1, Some(&groups), 123).unwrap()
    );
    assert_ne!(
        cuts,
        folds_with_seed(&rows, "binary", 1, Some(&groups), 124).unwrap()
    );
    for (train, test) in cuts {
        for partition in [&train, &test] {
            assert_eq!(
                partition
                    .iter()
                    .map(|&i| rows[i][1].clone())
                    .collect::<BTreeSet<_>>()
                    .len(),
                2
            );
            assert_eq!(partition.contains(&0), partition.contains(&30));
            assert_eq!(partition.contains(&0), partition.contains(&31));
            assert_eq!(partition.contains(&2), partition.contains(&4));
        }
    }
}

#[test]
fn too_few_minority_groups_is_explicit_and_preprocessing_uses_training_only() {
    let rows = (0..20)
        .map(|i| vec![i.to_string(), usize::from(i < 4).to_string()])
        .collect::<Vec<_>>();
    assert!(folds_with_seed(&rows, "binary", 1, None, 42)
        .unwrap_err()
        .to_string()
        .contains("insufficient_class_groups"));
    let training = vec![
        vec!["red".into(), "0".into(), "0".into()],
        vec!["blue".into(), "10".into(), "1".into()],
    ];
    let pp = Preprocessing::fit(&training, "binary").unwrap();
    let heldout = vec![vec!["green".into(), "1000".into(), "1".into()]];
    let transformed = pp.transform(&heldout).unwrap();
    assert_eq!(pp.columns[1].mean, 5.0);
    assert_eq!(transformed[0][0], -1.0);
    assert_eq!(transformed[0][1], 199.0);
}

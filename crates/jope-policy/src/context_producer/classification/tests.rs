use super::*;
fn fixture() -> ClassificationInput {
    let rows = (0..8)
        .map(|i| TrainingRow {
            id: format!("row{i}"),
            values: vec![
                if i == 0 { None } else { Some(i.to_string()) },
                [
                    None,
                    Some("".into()),
                    Some("NA".into()),
                    Some("null".into()),
                ][i % 4]
                    .clone(),
            ],
            label: if i % 2 == 0 { "red" } else { "blue" }.into(),
        })
        .collect::<Vec<_>>();
    ClassificationInput {
        schema: CLASSIFICATION_INPUT_SCHEMA.into(),
        task: ClassificationTask::Binary,
        classes: vec!["blue".into(), "red".into()],
        split_sha256: "c".repeat(64),
        identity: TrainingIdentity {
            dataset_sha256: "a".repeat(64),
            prepared_sha256: "b".repeat(64),
            training_sha256: training_rows_sha256(&rows).unwrap(),
            fitting_rows_sha256: sha(&serde_json::to_vec(
                &rows.iter().map(|r| &r.id).collect::<Vec<_>>(),
            )
            .unwrap()),
            cut: 0,
        },
        features: vec![
            Feature {
                name: "amount".into(),
                kind: FeatureKind::Numeric,
            },
            Feature {
                name: "category".into(),
                kind: FeatureKind::Categorical,
            },
        ],
        rows,
    }
}
#[test]
fn label_renaming_never_changes_feature_or_neutral_conditioning_bytes() {
    let input = fixture();
    let before = input.prepare().unwrap();
    let mut renamed = input.clone();
    renamed.classes = vec!["999-not-a-number".into(), "null".into()];
    for row in &mut renamed.rows {
        row.label = if row.label == "blue" {
            "999-not-a-number"
        } else {
            "null"
        }
        .into();
    }
    renamed.identity.training_sha256 = training_rows_sha256(&renamed.rows).unwrap();
    let after = renamed.prepare().unwrap();
    assert_eq!(
        serde_json::to_vec(&before.preprocessing).unwrap(),
        serde_json::to_vec(&after.preprocessing).unwrap()
    );
    assert_eq!(
        input.preprocessing_sha256().unwrap(),
        renamed.preprocessing_sha256().unwrap()
    );
    assert_eq!(
        serde_json::to_vec(&before.rows).unwrap(),
        serde_json::to_vec(&after.rows).unwrap()
    );
    assert_eq!(
        neutral_foundation_rows(&before.rows),
        neutral_foundation_rows(&after.rows)
    );
    assert!(neutral_foundation_rows(&before.rows)
        .iter()
        .all(|r| r.last() == Some(&0.)));
    assert_ne!(
        input.identity.training_sha256,
        renamed.identity.training_sha256
    );
    assert_eq!(
        before.preprocessing.columns[1].categories,
        vec!["", "NA", "null"]
    );
    assert!(before.rows.iter().any(|r| r[1] == -1.));
}
#[test]
fn classification_input_cannot_enter_accepted_regression_parser() {
    let input = fixture();
    assert!(serde_json::from_slice::<TrainingInput>(&serde_json::to_vec(&input).unwrap()).is_err());
    assert_eq!(input.task, ClassificationTask::Binary);
}
#[test]
fn classification_rejects_changed_labels_ids_or_declared_task() {
    let mut v = fixture();
    v.rows[0].label = "scoring-label".into();
    assert!(v.prepare().is_err());
    let mut v = fixture();
    v.identity.fitting_rows_sha256 = "e".repeat(64);
    assert!(v.prepare().is_err());
    let mut v = fixture();
    v.task = ClassificationTask::Multiclass;
    assert!(v.prepare().is_err());
    let mut v = fixture();
    v.classes[0].clear();
    assert!(v.prepare().is_err());
}

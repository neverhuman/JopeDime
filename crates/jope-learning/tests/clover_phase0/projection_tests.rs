use super::*;

fn visibility(labels: bool) -> Visibility {
    Visibility {
        task: "regression".into(),
        family: "family-a".into(),
        split_sha256: digest("split"),
        fold: 0,
        available_through: 10,
        fitting_rows: vec!["fit-a".into(), "fit-b".into()],
        forbidden_rows: BTreeSet::from(["held".into(), "future".into()]),
        training_labels: labels,
    }
}
fn observations() -> BTreeMap<String, Observation> {
    ["fit-a", "fit-b", "held", "future"]
        .into_iter()
        .enumerate()
        .map(|(i, id)| {
            (
                id.into(),
                Observation {
                    task: "regression".into(),
                    family: "family-a".into(),
                    role: if i < 2 { "training" } else { "confirmation" }.into(),
                    split_sha256: digest("split"),
                    fold: 0,
                    available_at: if i == 3 { 11 } else { 10 },
                    features: vec![i as f64, (i * i) as f64],
                    label: i as f64 + 100.,
                },
            )
        })
        .collect()
}

#[test]
fn feature_only_projection_never_reads_labels_heldout_or_future_payloads() {
    let mut rows = observations();
    let scope = visibility(false);
    let before = canonical_json(&project_encoder_input(&scope, &rows).unwrap()).unwrap();
    for row in rows.values_mut() {
        row.label = f64::NAN;
    }
    rows.get_mut("held").unwrap().features = vec![f64::NAN];
    rows.remove("future");
    assert_eq!(
        before,
        canonical_json(&project_encoder_input(&scope, &rows).unwrap()).unwrap()
    );
    assert!(project_encoder_input(&scope, &rows)
        .unwrap()
        .labels
        .is_none());
}

#[test]
fn supervised_projection_preserves_only_authorized_training_labels() {
    let mut rows = observations();
    let scope = visibility(true);
    let before = project_encoder_input(&scope, &rows).unwrap();
    assert_eq!(before.labels, Some(vec![100., 101.]));
    rows.get_mut("held").unwrap().label = 9e100;
    rows.get_mut("future").unwrap().features = vec![f64::NAN];
    assert_eq!(
        canonical_sha256(&before).unwrap(),
        canonical_sha256(&project_encoder_input(&scope, &rows).unwrap()).unwrap()
    );
    rows.get_mut("fit-a").unwrap().label = 123.;
    let after = project_encoder_input(&scope, &rows).unwrap();
    assert_eq!(before.features, after.features);
    assert_ne!(before.labels, after.labels);
}

#[test]
fn encoder_rejects_task_role_split_fold_family_time_and_membership_leakage() {
    let changes: Vec<fn(&mut Observation)> = vec![
        |r| r.task = "binary".into(),
        |r| r.family = "sealed-family".into(),
        |r| r.split_sha256 = digest("other split"),
        |r| r.fold = 1,
        |r| r.role = "validation".into(),
        |r| r.role = "confirmation".into(),
        |r| r.role = "delivery".into(),
        |r| r.available_at = 11,
    ];
    for change in changes {
        let mut rows = observations();
        change(rows.get_mut("fit-a").unwrap());
        assert!(project_encoder_input(&visibility(true), &rows).is_err());
    }
    for ids in [
        vec!["fit-a", "fit-a"],
        vec!["fit-a", "held"],
        vec!["fit-a", "missing"],
    ] {
        let mut scope = visibility(true);
        scope.fitting_rows = ids.into_iter().map(str::to_owned).collect();
        assert!(project_encoder_input(&scope, &observations()).is_err());
    }
}

#[test]
fn actual_preprocessing_and_target_copy_guard_respect_projected_inputs() {
    use jope_parallel::prepare::{reject_target_copies, Preprocessing};
    let mut rows = observations();
    // The production preprocessing implementation receives the projected training cut only.
    let prepare = |rows: &BTreeMap<String, Observation>| {
        let input = project_encoder_input(&visibility(true), rows).unwrap();
        let labels = input.labels.unwrap();
        let raw = input
            .features
            .into_iter()
            .zip(labels)
            .map(|(x, y)| {
                x.into_iter()
                    .chain([y])
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let pp = Preprocessing::fit(&raw, "regression").unwrap();
        (canonical_json(&pp).unwrap(), pp.transform(&raw).unwrap())
    };
    let before = prepare(&rows);
    rows.get_mut("held").unwrap().label = 99999.;
    rows.get_mut("future").unwrap().features = vec![1e90];
    assert_eq!(before, prepare(&rows));
    let copied = (0..4)
        .map(|i| vec![i.to_string(), i.to_string()])
        .collect::<Vec<_>>();
    assert!(reject_target_copies(&["label_copy".into()], &copied).is_err());
}

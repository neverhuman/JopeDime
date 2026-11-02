use super::*;

fn rows() -> Vec<String> {
    (0..10).map(|index| format!("row-{index}")).collect()
}

fn manifest() -> SplitViewManifestV1 {
    build_split_view_manifest(
        "dataset-1",
        "family-1",
        Task::Binary,
        OuterSplit::Train,
        &rows(),
        &"a".repeat(64),
        Some(&"b".repeat(64)),
    )
    .expect("valid fixture manifest")
}

#[test]
fn family_split_matches_the_deterministic_python_contract() {
    assert_eq!(source_family_campaign_split("synthetic:15"), "validation");
    assert_eq!(source_family_campaign_split("synthetic:abc"), "train");
    assert_eq!(
        source_family_campaign_split("synthetic:15"),
        source_family_campaign_split("synthetic:15")
    );
    let families = BTreeMap::from([
        ("d0".to_owned(), "f0".to_owned()),
        ("d1".to_owned(), "f1".to_owned()),
        ("d2".to_owned(), "f1".to_owned()),
        ("d3".to_owned(), "f2".to_owned()),
        ("d4".to_owned(), "f3".to_owned()),
        ("d5".to_owned(), "f3".to_owned()),
        ("d6".to_owned(), "f4".to_owned()),
        ("d7".to_owned(), "f5".to_owned()),
    ]);
    let (train, holdout) = freeze_source_family_split(&families, 4, 4, 17).unwrap();
    assert_eq!(train, ["d3", "d4", "d5", "d6"]);
    assert_eq!(holdout, ["d0", "d1", "d2", "d7"]);
    for family in ["f0", "f1", "f2", "f3", "f4", "f5"] {
        let in_train = families
            .iter()
            .any(|(dataset, candidate)| candidate == family && train.contains(dataset));
        let in_holdout = families
            .iter()
            .any(|(dataset, candidate)| candidate == family && holdout.contains(dataset));
        assert!(
            !(in_train && in_holdout),
            "family {family} crossed the split"
        );
    }
}

#[test]
fn split_view_identities_match_the_python_wire_contract() {
    let manifest = manifest();
    assert_eq!(
        manifest.sha256().unwrap(),
        "36f098be3131051e0ed2ccb7a16512a88417a15c8eb034377af1d68fd1c5be81"
    );
    assert_eq!(
        manifest.outer_split_hash,
        "8d47d7bbd1309c9296de4bc6cfe293e91da8885d9a4c963de3193ae69f2a5ad1"
    );
    assert_eq!(
        manifest.fold_plan_hash,
        "4d5cdc31ca43c6343ee6e73f7d15862f46de7c476ddb4f960863d1ee50547ef2"
    );
    assert_eq!(
        manifest.views[0].row_assignment_hash,
        "29a70d6de809a5109fbabe5d45761c52bfbf2e1aa0ef85dea514cab936a44939"
    );
    assert_eq!(manifest.views[0].train_row_ids().len(), 8);
    assert_eq!(manifest.views[0].test_row_ids().len(), 2);
}

#[test]
fn binary_summaries_are_bound_to_exact_train_rows_and_python_f32_values() {
    let manifest = manifest();
    let summaries = build_binary_target_summaries(
        &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        &rows(),
        &manifest,
    )
    .unwrap();
    assert_eq!(
        summaries
            .iter()
            .map(|summary| summary.positive_row_count)
            .collect::<Vec<_>>(),
        [5, 4, 4, 3, 4]
    );
    assert_eq!(
        summaries[0].split_view_manifest_hash,
        "36f098be3131051e0ed2ccb7a16512a88417a15c8eb034377af1d68fd1c5be81"
    );
    assert_eq!(summaries[0].values[0], 0.8888888955116272);
    assert_eq!(
        summaries[0].values[1..7],
        [
            0.625,
            0.375,
            0.6615632176399231,
            0.5108256340026855,
            0.800000011920929,
            1.3333333730697632
        ]
    );
    assert!(summaries[0].values[7..].iter().all(|value| *value == 0.0));
}

#[test]
fn malformed_or_non_train_only_inputs_are_rejected() {
    assert!(build_split_view_manifest(
        "",
        "f",
        Task::Binary,
        OuterSplit::Train,
        &rows(),
        &"a".repeat(64),
        None
    )
    .is_err());
    assert!(build_split_view_manifest(
        "d",
        "f",
        Task::Binary,
        OuterSplit::Train,
        &rows(),
        "bad",
        None
    )
    .is_err());
    let manifest = manifest();
    assert!(
        binary_target_summary(&[0.0; 8], manifest.views[0].test_row_ids(), &manifest, 0).is_err()
    );
    assert!(binary_target_summary(
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        manifest.views[0].train_row_ids(),
        &manifest,
        0
    )
    .is_err());
}

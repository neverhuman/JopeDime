use super::*;
#[test]
fn previous_generation_targets_are_not_relabelled_as_fresh() {
    let t = tempfile::tempdir().unwrap();
    let mut labels = Vec::new();
    for (record, generation) in [(0, 0), (1, 1)] {
        let c = frozen(
            t.path(),
            &t.path().join(format!("fixtures/c{record}.json")),
            &json!({"generation":generation}),
        )
        .unwrap();
        let target = frozen(
            t.path(),
            &t.path().join(format!("fixtures/t{record}.json")),
            &json!({"candidate":c}),
        )
        .unwrap();
        serde_json::to_writer(
            &mut labels,
            &json!({"record":record,"context":0,"target":target}),
        )
        .unwrap();
        labels.push(b'\n');
    }
    let groups = vec![json!({"context":0,"records":[0,1]})];
    assert_eq!(
        fresh_record_groups(t.path(), &labels, &groups, 1).unwrap(),
        BTreeMap::from([("0".to_owned(), vec![1])])
    );
    assert!(fresh_record_groups(t.path(), &labels, &groups, 2)
        .unwrap()
        .is_empty());
}

use super::*;

#[test]
fn pre_d5_ten_member_catalog_is_rejected() {
    let pre_d5: Vec<&str> = PRE_D5_FIXTURES.iter().map(|(path, _)| *path).collect();
    // A silent downgrade to the pre-D5 catalog: only the ten old members remain listed.
    let (_temporary, root) = corpus();
    rewrite_catalog(&root, |catalog| {
        catalog
            .datasets
            .retain(|entry| pre_d5.contains(&entry.path.as_str()));
        assert_eq!(catalog.datasets.len(), 10);
    });
    let error = verify_inputs(&root).unwrap_err().to_string();
    assert!(error.contains("expected exactly 40 datasets"), "{error}");
    // 39 and 41 members.
    for alteration in 0..2 {
        let (_temporary, root) = corpus();
        rewrite_catalog(&root, |catalog| {
            if alteration == 0 {
                catalog.datasets.pop();
            } else {
                let mut extra = catalog.datasets[0].clone();
                extra.id = "binary.extra".into();
                extra.path = "binary/extra.json".into();
                catalog.datasets.push(extra);
            }
        });
        let error = verify_inputs(&root).unwrap_err().to_string();
        assert!(
            error.contains("expected exactly 40 datasets"),
            "alteration {alteration}: {error}"
        );
    }
    // Forty members, but nine in one family and eleven in another.
    let (_temporary, root) = corpus();
    rewrite_catalog(&root, |catalog| {
        let entry = catalog
            .datasets
            .iter_mut()
            .find(|entry| entry.id == "binary.nonlinear")
            .unwrap();
        entry.task = Task::Regression;
    });
    let error = verify_inputs(&root).unwrap_err().to_string();
    assert!(error.contains("expected ten binary datasets"), "{error}");
}

use super::*;

#[test]
fn missing_extra_duplicate_and_untrusted_catalog_entries_fail() {
    for alteration in 0..6 {
        let (_temporary, root) = corpus();
        rewrite_catalog(&root, |catalog| match alteration {
            0 => {
                catalog.datasets.pop();
            }
            1 => catalog.datasets.push(catalog.datasets[0].clone()),
            2 => catalog.datasets[1] = catalog.datasets[0].clone(),
            3 => catalog.provenance.kind = "downloaded".into(),
            4 => catalog
                .provenance
                .external_inputs
                .push("external.csv".into()),
            _ => catalog.datasets[0].task = Task::Regression,
        });
        assert!(verify_inputs(&root).is_err(), "alteration {alteration}");
    }
}

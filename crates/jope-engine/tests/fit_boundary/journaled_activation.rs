use super::*;

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn journaled_activation_row_with_current_epoch_and_coverage_cannot_enable_shipped_fitting() {
    let data = Data::new();
    assert!(SCOPE.starts_with("synthetic-"));
    forge_journaled_activation(&data.authority);
    let active: bool = read_only(&data.authority)
        .query_row(ACTIVATION_SQL, [SCOPE], |r| r.get(0))
        .unwrap();
    assert!(active, "forged row does not satisfy the activation query");
    assert_every_entry_refused_without_verified_activation(&data, "forged");
}

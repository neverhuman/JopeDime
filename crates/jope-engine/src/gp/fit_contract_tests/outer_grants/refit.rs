use super::*;

/// The grant names the model identity its final-refit check verified, never
/// the handoff's unverified record of it.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn an_outer_grant_names_the_model_its_final_refit_verified() {
    let case = outer_case();
    let honest = case.grant("honest", &case.sealed).unwrap();
    let mut handoff: Value = read_json(&case.handoff()).unwrap();
    let verified = handoff["final_refit"]["model_sha256"].clone();
    assert_eq!(honest["subject"]["model_sha256"], verified);
    handoff["final_refit"]["model_sha256"] = json!("e".repeat(64));
    write_json(&case.handoff(), &handoff).unwrap();
    let granted = case.grant("misrecorded", &case.sealed).unwrap();
    assert_eq!(
        granted["subject"]["model_sha256"], verified,
        "the grant names a model identity its final refit never verified: {granted}"
    );
}

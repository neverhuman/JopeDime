use super::*;

/// The reviewed finding [P1]: `grant_body` hashed `prepared.json` against the
/// sealed selection, then reopened it. A preparation substituted between the
/// two supplied the grant's sealed-label identity and metadata.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn an_outer_grant_takes_its_labels_and_metadata_from_the_verified_preparation() {
    let case = outer_case();
    let prepared = case.prepared.join("prepared.json");
    let bound: Value = read_json(&prepared).unwrap();
    // Positive control: an honest grant binds the preparation's sealed labels.
    let mut honest = Value::Null;
    let nth = verifications(&prepared, || {
        honest = case.grant("honest", &case.sealed).unwrap();
    });
    assert_eq!(
        honest["scoring_labels"]["sha256"],
        hash(SEALED_TEST.as_bytes())
    );
    assert_eq!(honest["preparation"]["dataset"], bound["dataset"]);
    // A preparation naming another label file and other metadata replaces
    // the verified one right after its check.
    let mut substituted = bound.clone();
    substituted["original_test_sha256"] = json!(hash(OTHER_TEST.as_bytes()));
    substituted["dataset"] = json!("substituted-dataset");
    substituted["family"] = json!("substituted-family");
    substituted["clean_status"] = json!("substituted");
    let _hook = swap_at(
        &prepared,
        nth,
        serde_json::to_vec_pretty(&substituted).unwrap(),
    );
    let granted = case.grant("swapped", &case.other).unwrap();
    assert_eq!(
        granted["scoring_labels"]["sha256"],
        hash(SEALED_TEST.as_bytes()),
        "a grant was issued for labels the sealed preparation never bound: {granted}"
    );
    for key in ["dataset", "family", "clean_status"] {
        assert_eq!(granted["preparation"][key], bound[key], "{key}: {granted}");
    }
    // The scorer checks the named file against the verified identity, so the
    // substituted labels are never parsed.
    let outer = outer_release::Authority::open(&case.root.join("swapped")).unwrap();
    let grant_path = case
        .root
        .join("swapped/grants")
        .join(format!("{}.json", granted["grant_id"].as_str().unwrap()));
    let receipt = case.root.join("swapped-score.json");
    let error = refusal(outer.score(&grant_path, &case.predictor, &case.other, &receipt));
    assert!(
        error.contains("differ from the granted identity"),
        "{error}"
    );
}

/// The same receiver, for the sealed selection: the seal the handoff records
/// is the seal that is compared with the replayed selection.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn an_outer_grant_is_bound_to_the_seal_it_checked() {
    let case = outer_case();
    let mut handoff: Value = read_json(&case.handoff()).unwrap();
    let seal = case.root.join(handoff["seal"]["path"].as_str().unwrap());
    let replayed = fs::read(&seal).unwrap();
    // Positive control, counting the seal's verifications.
    let nth = verifications(&seal, || {
        case.grant("honest", &case.sealed).unwrap();
    });
    // The handoff records a seal that is not the replayed selection.
    let mut tampered: Value = serde_json::from_slice(&replayed).unwrap();
    tampered["stop_reason"] = json!("tampered");
    let tampered = serde_json::to_vec_pretty(&tampered).unwrap();
    fs::write(&seal, &tampered).unwrap();
    handoff["seal"]["sha256"] = json!(hash(&tampered));
    write_json(&case.handoff(), &handoff).unwrap();
    // Negative control: refused without a replacement.
    let error = refusal(case.grant("tampered", &case.sealed));
    assert!(
        error.contains("differs from the replayed selection"),
        "{error}"
    );
    // The replayed seal replaces it right after its check against the handoff.
    let _hook = swap_at(&seal, nth, replayed);
    let error = refusal(case.grant("swapped", &case.sealed));
    assert!(
        error.contains("differs from the replayed selection"),
        "a grant was issued for a handoff whose seal is not the replayed selection: {error}"
    );
}

/// The same receiver, for the fitted subject: the predictor the grant names
/// is the predictor verified as the selected final refit.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn an_outer_grant_is_for_the_predictor_it_verified() {
    let case = outer_case();
    let mut handoff: Value = read_json(&case.handoff()).unwrap();
    let exported = fs::read(&case.predictor).unwrap();
    // Positive control, counting the predictor's verifications.
    let mut honest = Value::Null;
    let nth = verifications(&case.predictor, || {
        honest = case.grant("honest", &case.sealed).unwrap();
    });
    assert_eq!(honest["subject"]["predictor"]["sha256"], hash(&exported));
    // The handoff records a predictor that declares another training scope.
    let mut forged: Value = serde_json::from_slice(&exported).unwrap();
    forged["training_sha256"] = json!("0".repeat(64));
    let forged = serde_json::to_vec_pretty(&forged).unwrap();
    fs::write(&case.predictor, &forged).unwrap();
    handoff["predictor"]["sha256"] = json!(hash(&forged));
    write_json(&case.handoff(), &handoff).unwrap();
    // Negative control: refused without a replacement.
    let error = refusal(case.grant("forged", &case.sealed));
    assert!(error.contains("another training scope"), "{error}");
    // The verified export replaces it right after the subject check.
    let _hook = swap_at(&case.predictor, nth, exported);
    let result = case.grant("swapped", &case.sealed);
    assert!(
        matches!(&result, Err(e) if format!("{e:#}").contains("another training scope")),
        "a grant was issued for a subject never verified as the final refit: {result:?}"
    );
}

/// The same receiver, for the handoff: the digest the grant records is the
/// digest of the handoff whose content was checked and recorded.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
pub(crate) fn an_outer_grant_records_the_handoff_it_checked() {
    let case = outer_case();
    let recorded = fs::read(case.handoff()).unwrap();
    // Positive control, counting the handoff's verifications.
    let mut honest = Value::Null;
    let nth = verifications(&case.handoff(), || {
        honest = case.grant("honest", &case.sealed).unwrap();
    });
    assert_eq!(honest["subject"]["handoff"]["sha256"], hash(&recorded));
    // Another handoff replaces it right after it was read.
    let mut other: Value = serde_json::from_slice(&recorded).unwrap();
    other["final_refit"]["model_sha256"] = json!("f".repeat(64));
    let _hook = swap_at(
        &case.handoff(),
        nth,
        serde_json::to_vec_pretty(&other).unwrap(),
    );
    let granted = case.grant("swapped", &case.sealed).unwrap();
    assert_eq!(
        granted["subject"]["handoff"]["sha256"],
        hash(&recorded),
        "the grant records a handoff other than the one it checked: {granted}"
    );
    assert_eq!(
        granted["subject"]["model_sha256"],
        honest["subject"]["model_sha256"]
    );
}

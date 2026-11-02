use super::*;

/// Route R8 at `label_admission::validate_preverified` (label_admission.rs
/// :475-481 before the fix): the target scale an observation is admitted with
/// is read from exactly the preparation state whose digest was checked.
#[test]
fn an_honest_scale_is_admitted_against_the_verified_state_even_if_replaced_after_its_check() {
    let c = scale_case();
    // Positive control.
    c.validate(&c.rows[1]).unwrap();
    let _hook = replace_on_verification(&c.state, rescaled());
    let admitted = c.validate(&c.rows[1]).map_err(|e| format!("{e:#}"));
    assert_eq!(
        admitted,
        Ok(()),
        "the admission consumed a preparation state other than the verified one"
    );
}

/// The same receiver, other direction: an observation claiming the
/// replacement's scale, with its own committed completion, is refused with or
/// without the replacement.
#[test]
fn a_forged_scale_is_refused_even_if_the_state_is_replaced_after_its_check() {
    let c = scale_case();
    let mut forged = c.rows[1].clone();
    forged["target_scale"] = json!(3.);
    forged["evaluation_identity_sha256"] = json!(hash(
        &serde_json::to_vec(&admission::identity(&forged)).unwrap()
    ));
    attach_completion(
        c.s.f.root(),
        c.s.f.ledger("family-a").parent().unwrap(),
        900,
        &mut forged,
    );
    // Negative control: refused without a replacement.
    let refused = failure(c.validate(&forged));
    assert!(refused.contains("preprocessing/scale"), "{refused}");
    let _hook = replace_on_verification(&c.state, rescaled());
    let refused = failure(c.validate(&forged));
    assert!(
        refused.contains("preprocessing/scale"),
        "an observation was admitted with the scale of a state other than the verified one: \
         {refused}"
    );
}

/// Route R8 at `label_admission::verify_inputs` (label_admission.rs:240-266
/// before the fix), the receiver behind native_genome.rs:347-349 and the
/// teaching store's preparation map: provenance is checked against the hashed
/// preparation bytes, never against a caller's separate parse.
#[test]
fn the_teachers_input_admission_binds_the_callers_preparation_to_the_hashed_bytes() {
    let s = strict();
    let root = s.f.root();
    let (mut context, prepared, _) = family_a(&s);
    let path = root.join("family-a/prepared.json");
    // Positive control.
    admission::verify_inputs(root, &context, &prepared).unwrap();
    // The bound preparation is not a clean evaluation.
    let mut unclean: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    unclean["clean_evaluation"] = json!(false);
    let unclean = serde_json::to_vec(&unclean).unwrap();
    fs::write(&path, &unclean).unwrap();
    context.prepared_sha256 = hash(&unclean);
    // Negative control: the caller's parse of the bound bytes is refused.
    let honest: Prepared = serde_json::from_slice(&unclean).unwrap();
    let refused = failure(admission::verify_inputs(root, &context, &honest));
    assert!(refused.contains("provenance conflict"), "{refused}");
    // A caller's parse of other bytes (what native_genome.rs:347-349 held
    // after an unverified read) claims a clean evaluation.
    let refused = failure(admission::verify_inputs(root, &context, &prepared));
    assert!(
        refused.contains("differs from the bound"),
        "provenance was checked against a parse other than the hashed bytes: {refused}"
    );
}

/// The same receiver: the preparation is replaced after its digest check, and
/// the admitted training identity is still the one of the verified
/// preparation (its views are not resolved from a reopened `prepared.json`).
#[test]
fn a_preparation_replaced_after_its_check_never_decides_the_admitted_views() {
    let s = strict();
    let root = s.f.root();
    let (context, prepared, _) = family_a(&s);
    let path = root.join("family-a/prepared.json");
    // Positive control.
    let admitted = admission::verify_inputs(root, &context, &prepared).unwrap();
    let mut other: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    other["cuts"][0]["training_sha256"] = json!("0".repeat(64));
    let _hook = replace_on_verification(&path, serde_json::to_vec(&other).unwrap());
    let raced = admission::verify_inputs(root, &context, &prepared).map_err(|e| format!("{e:#}"));
    assert_eq!(
        raced,
        Ok(admitted),
        "the admitted views were resolved from a preparation other than the verified one"
    );
}

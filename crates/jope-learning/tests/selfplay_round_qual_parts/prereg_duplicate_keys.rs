use super::*;

#[test]
fn rejects_duplicate_prereg_keys() {
    let fixture = prereg_fixture();
    let PreregFixture {
        identity,
        parsed_committed,
        pinned_prereg,
        parent_map,
        ratchets,
        scores,
        parity,
        pinned_bytes,
        pinned_plan,
        ..
    } = fixture;
    // 5. Duplicate-key JSON smuggling rejection (both orders, raw and canonical SHA-256):
    let base_pinned_str = String::from_utf8(pinned_bytes.clone()).unwrap();
    let dup_false_then_true = base_pinned_str.replacen(
        "\"authorized\": true",
        "\"authorized\": false,\n    \"authorized\": true",
        1,
    );
    let dup_true_then_false = base_pinned_str.replacen(
        "\"authorized\": true",
        "\"authorized\": true,\n    \"authorized\": false",
        1,
    );
    for (dup_label, dup_str) in [
        ("dup_false_then_true", &dup_false_then_true),
        ("dup_true_then_false", &dup_true_then_false),
    ] {
        let dup_bytes = dup_str.as_bytes();
        for sha in [
            sha256_hex(dup_bytes),
            canonical_sha256(&pinned_prereg).unwrap(),
            canonical_sha256(&parsed_committed).unwrap(),
        ] {
            let mut dup_plan = pinned_plan.clone();
            dup_plan.prereg_sha256 = sha;
            assert!(
                qualify_rounds_with_prereg(
                    RoundQualInputs {
                        plan: &dup_plan,
                        identity: &identity,
                        ratchet_outputs: &ratchets,
                        test_scores: &scores,
                        parity: &parity,
                        parent_of_context: &parent_map,
                    },
                    Some(dup_bytes),
                )
                .is_err(),
                "expected duplicate-key payload `{dup_label}` to be rejected"
            );
        }
    }
}

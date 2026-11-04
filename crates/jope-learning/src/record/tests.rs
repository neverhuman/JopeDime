use super::*;

fn identity() -> HwmIdentity {
    HwmIdentity {
        comparison: None,
        task: "regression".into(),
        dataset_hash: "4e9ab20dc7413158".into(),
        cut_id: "0".repeat(64),
        evaluator_sha256: "b5ccee8e33af34313c51896f52466f87f2f8544dcb01672e6d6b4832d1ac4393".into(),
        vocabulary_sha256: "8d55fee4366acd2c175cf55bfd769cebc904770f5227a6fd50a613723dc85787"
            .into(),
        protocol: "adaptive_cut_heldout_teacher".into(),
    }
}

#[test]
fn identity_hash_is_sha256_of_sorted_compact_json() {
    // Fixed vector: the six keys in bytewise order, compact, no whitespace.
    let expected_text = format!(
        r#"{{"cut_id":"{}","dataset_hash":"4e9ab20dc7413158","evaluator_sha256":"b5ccee8e33af34313c51896f52466f87f2f8544dcb01672e6d6b4832d1ac4393","protocol":"adaptive_cut_heldout_teacher","task":"regression","vocabulary_sha256":"8d55fee4366acd2c175cf55bfd769cebc904770f5227a6fd50a613723dc85787"}}"#,
        "0".repeat(64)
    );
    assert_eq!(
        String::from_utf8(crate::canon::canonical_json(&identity()).unwrap()).unwrap(),
        expected_text
    );
    assert_eq!(
        identity().sha256().unwrap(),
        crate::canon::sha256_hex(expected_text.as_bytes())
    );
    // Independently computed vector for a fully literal identity.
    let literal = HwmIdentity {
        comparison: None,
        task: "binary".into(),
        dataset_hash: "0259e0b6af043f58".into(),
        cut_id: "a".repeat(64),
        evaluator_sha256: UNBOUND.into(),
        vocabulary_sha256: UNBOUND.into(),
        protocol: "genebank-gen1".into(),
    };
    assert_eq!(
        literal.sha256().unwrap(),
        crate::canon::sha256_hex(
            format!(
                r#"{{"cut_id":"{}","dataset_hash":"0259e0b6af043f58","evaluator_sha256":"unbound","protocol":"genebank-gen1","task":"binary","vocabulary_sha256":"unbound"}}"#,
                "a".repeat(64)
            )
            .as_bytes()
        )
    );
}

#[test]
fn unknown_fields_are_rejected() {
    let mut v = serde_json::to_value(identity()).unwrap();
    v["extra"] = 1.into();
    assert!(serde_json::from_value::<HwmIdentity>(v).is_err());
}

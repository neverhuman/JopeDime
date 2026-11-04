use super::*;

#[test]
fn positive_vectors_canonical_bytes_and_domain_separated_ids_match() {
    let raw = fs::read_to_string(fixture_dir().join("positive_vectors.json"))
        .expect("read positive_vectors.json");
    let doc = pyjson::parse_str(&raw).expect("parse positive_vectors.json");
    let vectors = doc
        .get("vectors")
        .and_then(PyValue::as_list)
        .expect("vectors array");
    assert_eq!(
        vectors.len(),
        10,
        "expected 10 positive conformance vectors"
    );

    let mut seen_digests = std::collections::BTreeMap::new();
    for vec_val in vectors {
        let id = vec_val
            .get("id")
            .and_then(PyValue::as_str)
            .expect("vector id");
        let schema_id = vec_val
            .get("schema_id")
            .and_then(PyValue::as_str)
            .expect("schema_id");
        let raw_input = vec_val
            .get("raw_input_json")
            .and_then(PyValue::as_str)
            .expect("raw_input_json");
        let expected_canon = vec_val
            .get("expected_canonical_payload")
            .and_then(PyValue::as_str)
            .expect("expected_canonical_payload");
        let expected_len = vec_val
            .get("expected_payload_byte_len")
            .and_then(PyValue::as_int)
            .expect("expected_payload_byte_len") as u64;
        let expected_sha = vec_val
            .get("expected_domain_separated_sha256")
            .and_then(PyValue::as_str)
            .expect("expected_domain_separated_sha256");

        let payload = CanonicalPayload::parse_and_canonicalize(raw_input.as_bytes())
            .unwrap_or_else(|e| panic!("{id}: failed to canonicalize: {e}"));
        assert_eq!(payload.schema_id(), schema_id, "{id}: schema_id mismatch");
        assert_eq!(
            payload.as_str(),
            expected_canon,
            "{id}: canonical payload mismatch"
        );
        assert_eq!(
            payload.as_bytes().len() as u64,
            expected_len,
            "{id}: canonical byte length mismatch"
        );

        // Verify from_canonical_bytes accepts the canonical output.
        let verified_canon = CanonicalPayload::from_canonical_bytes(payload.as_bytes())
            .unwrap_or_else(|e| {
                panic!("{id}: from_canonical_bytes rejected canonical output: {e}")
            });
        assert_eq!(verified_canon, payload);

        // If raw_input differed from expected_canon, from_canonical_bytes must reject raw_input.
        if raw_input != expected_canon {
            assert!(
                matches!(
                    CanonicalPayload::from_canonical_bytes(raw_input.as_bytes()),
                    Err(WireContractError::NonCanonicalByteEncoding { .. })
                ),
                "{id}: from_canonical_bytes must reject non-canonical raw input"
            );
        }

        let dom_id = payload
            .domain_separated_id()
            .unwrap_or_else(|e| panic!("{id}: domain_separated_id failed: {e}"));
        assert_eq!(dom_id.formula(), DOMAIN_SEPARATED_ID_FORMULA);
        assert_eq!(dom_id.payload_byte_len(), expected_len);
        assert_eq!(
            dom_id.sha256_hex(),
            expected_sha,
            "{id}: domain-separated sha256 mismatch"
        );

        seen_digests.insert(id.to_string(), dom_id.sha256_hex().to_string());
    }

    // Precomposed U+00E9 vs combining U+0065 U+0301 must remain distinct (no Unicode normalization).
    assert_ne!(
        seen_digests["V02_UNICODE_PRECOMPOSED_E_ACUTE"],
        seen_digests["V03_UNICODE_COMBINING_E_ACUTE"]
    );
    // Reversed array [3, 2, 1] must produce a distinct digest from [1, 2, 3].
    assert_ne!(
        seen_digests["V04_ARRAY_ORDER_FORWARD"],
        seen_digests["V05_ARRAY_ORDER_REVERSED"]
    );
    // +0.0 vs -0.0/subnormal must produce distinct digests.
    assert_ne!(
        seen_digests["V06_EXACT_FLOAT_POSITIVE_ZERO_WITH_DISPLAY"],
        seen_digests["V07_EXACT_FLOAT_NEGATIVE_ZERO_AND_SUBNORMAL"]
    );
}

#[test]
fn negative_vectors_lexical_numeric_cycle_and_authority_refusals() {
    let raw = fs::read_to_string(fixture_dir().join("negative_vectors.json"))
        .expect("read negative_vectors.json");
    let doc = pyjson::parse_str(&raw).expect("parse negative_vectors.json");
    let vectors = doc
        .get("vectors")
        .and_then(PyValue::as_list)
        .expect("vectors array");
    assert_eq!(
        vectors.len(),
        18,
        "expected 18 negative conformance vectors"
    );

    for vec_val in vectors {
        let id = vec_val
            .get("id")
            .and_then(PyValue::as_str)
            .expect("vector id");
        let raw_input = vec_val
            .get("raw_input_json")
            .and_then(PyValue::as_str)
            .expect("raw_input_json");
        let expected_kind = vec_val
            .get("expected_error_kind")
            .and_then(PyValue::as_str)
            .expect("expected_error_kind");

        let err = CanonicalPayload::parse_and_canonicalize(raw_input.as_bytes()).expect_err(
            &format!("{id}: expected refusal {expected_kind}, but succeeded"),
        );
        let actual_kind = match err {
            WireContractError::DuplicateObjectKey { .. } => "DuplicateObjectKey",
            WireContractError::UnpairedSurrogate { .. } => "UnpairedSurrogate",
            WireContractError::NegativeZeroInteger { .. } => "NegativeZeroInteger",
            WireContractError::LeadingZeroNumber { .. } => "LeadingZeroNumber",
            WireContractError::IntegerOverflow { .. } => "IntegerOverflow",
            WireContractError::RawFloatInIdentityPayload { .. } => "RawFloatInIdentityPayload",
            WireContractError::NonFiniteFloatBits { .. } => "NonFiniteFloatBits",
            WireContractError::InvalidHexEncoding { .. } => "InvalidHexEncoding",
            WireContractError::ForbiddenCycleField { .. } => "ForbiddenCycleField",
            WireContractError::UnknownEnumVariant { .. } => "UnknownEnumVariant",
            WireContractError::UnknownField { .. } => "UnknownField",
            WireContractError::InvalidOutcomeAuthority { .. } => "InvalidOutcomeAuthority",
            other => panic!("{id}: unexpected error variant {other:?}"),
        };
        assert_eq!(
            actual_kind, expected_kind,
            "{id}: error kind mismatch for input {raw_input}"
        );
    }

    // Byte-level BOM, invalid UTF-8, and unescaped control byte refusals.
    let mut bom_bytes = vec![0xEF, 0xBB, 0xBF];
    bom_bytes.extend_from_slice(b"{\"schema\":\"jope.selfplay.test/v1\"}");
    assert_eq!(
        CanonicalPayload::parse_and_canonicalize(&bom_bytes),
        Err(WireContractError::ByteOrderMark)
    );

    let bad_utf8 = b"{\"schema\":\"jope.selfplay.test/v1\",\"bad\":\"\xff\"}";
    assert_eq!(
        CanonicalPayload::parse_and_canonicalize(bad_utf8),
        Err(WireContractError::InvalidUtf8)
    );

    let unescaped_ctrl = b"{\"schema\":\"jope.selfplay.test/v1\",\"bad\":\"\x01\"}";
    assert!(matches!(
        CanonicalPayload::parse_and_canonicalize(unescaped_ctrl),
        Err(WireContractError::UnescapedControlChar { byte: 0x01, .. })
    ));
}

#[test]
fn exact_ieee754_float_bits_and_safe_integer_boundaries() {
    let pos_zero_f64 = ExactF64::from_f64(0.0).unwrap();
    let neg_zero_f64 = ExactF64::from_f64(-0.0).unwrap();
    let min_subnormal_f64 = ExactF64::from_bits(1).unwrap();
    assert_eq!(pos_zero_f64.to_hex(), "0000000000000000");
    assert_eq!(neg_zero_f64.to_hex(), "8000000000000000");
    assert_eq!(min_subnormal_f64.to_hex(), "0000000000000001");
    assert!(!pos_zero_f64.is_negative_zero());
    assert!(neg_zero_f64.is_negative_zero());
    assert!(min_subnormal_f64.is_subnormal());
    assert!(!pos_zero_f64.is_subnormal());
    assert_ne!(pos_zero_f64, neg_zero_f64);

    let pos_zero_f32 = ExactF32::from_f32(0.0).unwrap();
    let neg_zero_f32 = ExactF32::from_f32(-0.0).unwrap();
    let min_subnormal_f32 = ExactF32::from_bits(1).unwrap();
    assert_eq!(pos_zero_f32.to_hex(), "00000000");
    assert_eq!(neg_zero_f32.to_hex(), "80000000");
    assert_eq!(min_subnormal_f32.to_hex(), "00000001");
    assert!(neg_zero_f32.is_negative_zero());
    assert!(min_subnormal_f32.is_subnormal());

    assert!(matches!(
        ExactF64::from_f64(f64::NAN),
        Err(WireContractError::NonFiniteFloatBits { .. })
    ));
    assert!(matches!(
        ExactF64::from_f64(f64::INFINITY),
        Err(WireContractError::NonFiniteFloatBits { .. })
    ));
    assert!(matches!(
        ExactF64::from_f64(f64::NEG_INFINITY),
        Err(WireContractError::NonFiniteFloatBits { .. })
    ));
    assert!(matches!(
        ExactF32::from_f32(f32::NAN),
        Err(WireContractError::NonFiniteFloatBits { .. })
    ));
    assert!(matches!(
        ExactF32::from_f32(f32::INFINITY),
        Err(WireContractError::NonFiniteFloatBits { .. })
    ));

    let seed = HexU64::from_hex("/seed_u64_hex", "deadbeefcafebabe").unwrap();
    assert_eq!(seed.0, 0xdead_beef_cafe_babe);
    assert_eq!(seed.to_hex(), "deadbeefcafebabe");

    assert_eq!(SAFE_INT_MIN, -9_007_199_254_740_991);
    assert_eq!(SAFE_INT_MAX, 9_007_199_254_740_991);

    // Changing display_f64 must never alter canonical payload bytes or DomainSeparatedId.
    let p1 = CanonicalPayload::parse_and_canonicalize(
        br#"{"schema":"jope.selfplay.float-bits/v1","score_f64":{"f64_bits":"3fec000000000000","display_f64":0.875}}"#,
    )
    .unwrap();
    let p2 = CanonicalPayload::parse_and_canonicalize(
        br#"{"schema":"jope.selfplay.float-bits/v1","score_f64":{"f64_bits":"3fec000000000000","display_f64":0.875000001}}"#,
    )
    .unwrap();
    let p3 = CanonicalPayload::parse_and_canonicalize(
        br#"{"schema":"jope.selfplay.float-bits/v1","score_f64":{"f64_bits":"3fec000000000000"}}"#,
    )
    .unwrap();
    assert_eq!(p1.as_bytes(), p2.as_bytes());
    assert_eq!(p2.as_bytes(), p3.as_bytes());
    assert_eq!(
        p1.domain_separated_id().unwrap(),
        p2.domain_separated_id().unwrap()
    );
}

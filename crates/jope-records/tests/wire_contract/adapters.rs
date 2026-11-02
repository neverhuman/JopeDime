use super::*;

#[test]
fn native_inference_and_storage_adapters_emit_same_golden_payload_bytes_and_ids() {
    let raw = fs::read_to_string(fixture_dir().join("positive_vectors.json"))
        .expect("read positive_vectors.json");
    let doc = pyjson::parse_str(&raw).expect("parse positive_vectors.json");
    let vectors = doc
        .get("vectors")
        .and_then(PyValue::as_list)
        .expect("vectors array");

    for vec_val in vectors {
        let id = vec_val.get("id").and_then(PyValue::as_str).unwrap();
        let raw_input = vec_val
            .get("raw_input_json")
            .and_then(PyValue::as_str)
            .unwrap();
        let expected_canon = vec_val
            .get("expected_canonical_payload")
            .and_then(PyValue::as_str)
            .unwrap();
        let expected_sha = vec_val
            .get("expected_domain_separated_sha256")
            .and_then(PyValue::as_str)
            .unwrap();

        let native_frame =
            NativeEnvelopeAdapter::seal(raw_input.as_bytes(), None, Some(format!("native:{id}")))
                .unwrap();
        let inference_frame = InferenceEnvelopeAdapter::seal(
            raw_input.as_bytes(),
            None,
            Some(format!("inference:{id}")),
        )
        .unwrap();
        let storage_frame =
            StorageEnvelopeAdapter::seal(raw_input.as_bytes(), None, Some(format!("storage:{id}")))
                .unwrap();

        // All three domain adapters produce identical golden payload bytes and identical own_id.
        assert_eq!(native_frame.payload().as_str(), expected_canon);
        assert_eq!(
            native_frame.payload().as_bytes(),
            inference_frame.payload().as_bytes()
        );
        assert_eq!(
            inference_frame.payload().as_bytes(),
            storage_frame.payload().as_bytes()
        );
        assert_eq!(native_frame.own_id(), inference_frame.own_id());
        assert_eq!(inference_frame.own_id(), storage_frame.own_id());
        assert_eq!(native_frame.own_id().sha256_hex(), expected_sha);

        // Round-trip serialize and open each adapter's envelope bytes.
        let native_bytes = native_frame.to_canonical_bytes();
        let inference_bytes = inference_frame.to_canonical_bytes();
        let storage_bytes = storage_frame.to_canonical_bytes();

        let opened_native = NativeEnvelopeAdapter::open(&native_bytes).unwrap();
        let opened_inference = InferenceEnvelopeAdapter::open(&inference_bytes).unwrap();
        let opened_storage = StorageEnvelopeAdapter::open(&storage_bytes).unwrap();
        assert_eq!(opened_native, native_frame);
        assert_eq!(opened_inference, inference_frame);
        assert_eq!(opened_storage, storage_frame);

        // Cross-domain opening fails closed.
        assert!(matches!(
            NativeEnvelopeAdapter::open(&inference_bytes),
            Err(WireContractError::UnknownEnumVariant { .. })
        ));
        assert!(matches!(
            InferenceEnvelopeAdapter::open(&storage_bytes),
            Err(WireContractError::UnknownEnumVariant { .. })
        ));
        assert!(matches!(
            StorageEnvelopeAdapter::open(&native_bytes),
            Err(WireContractError::UnknownEnumVariant { .. })
        ));
    }
}

#[test]
fn envelope_separation_and_no_self_referential_cycle() {
    let payload_a = CanonicalPayload::parse_and_canonicalize(
        br#"{"schema":"jope.selfplay.draw-request/v1","step_index":7}"#,
    )
    .unwrap();
    let own_id_a = payload_a.domain_separated_id().unwrap();

    let sig_1 =
        SignatureEnvelope::new("native_worker", own_id_a.sha256_hex(), &"11".repeat(32)).unwrap();
    let sig_2 =
        SignatureEnvelope::new("storage_committer", own_id_a.sha256_hex(), &"22".repeat(32))
            .unwrap();

    let frame_1 = EnvelopeFrame::new(
        EnvelopeDomain::Native,
        payload_a.clone(),
        Some(sig_1),
        Some("first display note".to_string()),
    )
    .unwrap();
    let frame_2 = EnvelopeFrame::new(
        EnvelopeDomain::Storage,
        payload_a.clone(),
        Some(sig_2),
        Some("second display note".to_string()),
    )
    .unwrap();

    // Mutating signature, display note, or domain never alters payload bytes or own_id.
    assert_eq!(frame_1.payload().as_bytes(), frame_2.payload().as_bytes());
    assert_eq!(frame_1.own_id(), frame_2.own_id());

    // Changing schema_id or payload length/bytes changes DomainSeparatedId.
    let id_other_schema =
        DomainSeparatedId::compute("jope.selfplay.draw-request/v2", payload_a.as_bytes()).unwrap();
    assert_ne!(own_id_a.sha256_hex(), id_other_schema.sha256_hex());

    let payload_b = CanonicalPayload::parse_and_canonicalize(
        br#"{"schema":"jope.selfplay.draw-request/v1","step_index":8}"#,
    )
    .unwrap();
    assert_ne!(
        own_id_a.sha256_hex(),
        payload_b.domain_separated_id().unwrap().sha256_hex()
    );

    // Signature binding a wrong own_id is refused.
    let bad_sig =
        SignatureEnvelope::new("native_worker", &"ff".repeat(32), &"11".repeat(32)).unwrap();
    assert!(matches!(
        EnvelopeFrame::new(
            EnvelopeDomain::Native,
            payload_a.clone(),
            Some(bad_sig),
            None
        ),
        Err(WireContractError::SignatureMismatch { .. })
    ));

    // Tampering with own_id in serialized envelope bytes is refused.
    let valid_env = String::from_utf8(frame_1.to_canonical_bytes()).unwrap();
    let tampered_env = valid_env.replace(own_id_a.sha256_hex(), &"00".repeat(32));
    assert!(matches!(
        EnvelopeFrame::parse_and_verify(tampered_env.as_bytes()),
        Err(WireContractError::EnvelopeIdMismatch { .. })
    ));
}

#[test]
fn measured_authority_boundary_rejects_predicted_synthetic_missing_and_unverified_claims() {
    let receipt_sha = "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90";
    let measured_payload = CanonicalPayload::parse_and_canonicalize(
        br#"{"schema":"jope.selfplay.outcome-envelope/v1","outcome":{"authority":"verified_measured","metric_name":"cv_r2","evidence_kind":"measured_cpu_cv","evidence_receipt_sha256":"a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90","metric":{"f64_bits":"3fec000000000000"}}}"#,
    )
    .unwrap();
    let outcome_val = measured_payload
        .value()
        .as_object()
        .unwrap()
        .get("outcome")
        .unwrap();
    let parsed = parse_outcome_wire("/outcome", outcome_val).unwrap();
    let ParsedOutcomeWire::UnverifiedMeasured(claim) = parsed else {
        panic!("expected UnverifiedMeasured");
    };

    // Unverified proof (independent_verifier_passed == false) must be refused.
    let unverified_proof = MeasuredEvidenceProof {
        receipt_sha256: receipt_sha.to_string(),
        evidence_kind: MeasuredEvidenceKind::MeasuredCpuCv,
        independent_verifier_passed: false,
    };
    assert!(matches!(
        VerifiedMeasuredOutcome::verify_from_claim(&claim, &unverified_proof),
        Err(WireContractError::InvalidOutcomeAuthority { .. })
    ));

    // Mismatched receipt digest must be refused.
    let wrong_sha_proof = MeasuredEvidenceProof {
        receipt_sha256: "bb".repeat(32),
        evidence_kind: MeasuredEvidenceKind::MeasuredCpuCv,
        independent_verifier_passed: true,
    };
    assert!(matches!(
        VerifiedMeasuredOutcome::verify_from_claim(&claim, &wrong_sha_proof),
        Err(WireContractError::InvalidOutcomeAuthority { .. })
    ));

    // All-zero receipt digest must be refused.
    let zero_claim = ParsedMeasuredClaim {
        metric_name: "cv_r2".to_string(),
        metric: ExactF64::from_f64(0.875).unwrap(),
        evidence_receipt_sha256: "00".repeat(32),
        evidence_kind: MeasuredEvidenceKind::MeasuredCpuCv,
    };
    let zero_proof = MeasuredEvidenceProof {
        receipt_sha256: "00".repeat(32),
        evidence_kind: MeasuredEvidenceKind::MeasuredCpuCv,
        independent_verifier_passed: true,
    };
    assert!(matches!(
        VerifiedMeasuredOutcome::verify_from_claim(&zero_claim, &zero_proof),
        Err(WireContractError::InvalidOutcomeAuthority { .. })
    ));

    // Mismatched evidence_kind must be refused.
    let wrong_kind_proof = MeasuredEvidenceProof {
        receipt_sha256: receipt_sha.to_string(),
        evidence_kind: MeasuredEvidenceKind::MeasuredHoldoutEval,
        independent_verifier_passed: true,
    };
    assert!(matches!(
        VerifiedMeasuredOutcome::verify_from_claim(&claim, &wrong_kind_proof),
        Err(WireContractError::InvalidOutcomeAuthority { .. })
    ));

    // Valid proof upgrades the claim to VerifiedMeasuredOutcome and enters accept_measured_claim.
    let valid_proof = MeasuredEvidenceProof {
        receipt_sha256: receipt_sha.to_string(),
        evidence_kind: MeasuredEvidenceKind::MeasuredCpuCv,
        independent_verifier_passed: true,
    };
    let verified = VerifiedMeasuredOutcome::verify_from_claim(&claim, &valid_proof).unwrap();
    let receipt = accept_measured_claim(&verified);
    assert_eq!(receipt.metric_name, "cv_r2");
    assert_eq!(receipt.metric_f64_bits, "3fec000000000000");
    assert_eq!(receipt.evidence_receipt_sha256, receipt_sha);
    assert_eq!(receipt.evidence_kind, MeasuredEvidenceKind::MeasuredCpuCv);

    // Missing outcome parses to MissingOutcome with typed MissingReason::SealedHoldout.
    let missing_payload = CanonicalPayload::parse_and_canonicalize(
        br#"{"schema":"jope.selfplay.outcome-envelope/v1","outcome":{"authority":"missing","metric_name":"outer_holdout_r2","missing_reason":"sealed_holdout"}}"#,
    )
    .unwrap();
    let missing_parsed = parse_outcome_wire(
        "/outcome",
        missing_payload
            .value()
            .as_object()
            .unwrap()
            .get("outcome")
            .unwrap(),
    )
    .unwrap();
    assert!(matches!(
        missing_parsed,
        ParsedOutcomeWire::Missing(ref m) if m.missing_reason == MissingReason::SealedHoldout
    ));
}

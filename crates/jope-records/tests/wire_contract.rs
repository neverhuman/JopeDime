//! Integration and conformance tests for `jope_records::wire_contract` (`jope.canonical-json/v1`, `I02`).

use jope_records::pyjson::{self, Value as PyValue};
use jope_records::wire_contract::{
    accept_measured_claim, parse_outcome_wire, CanonicalPayload, DomainSeparatedId, EnvelopeDomain,
    EnvelopeFrame, ExactF32, ExactF64, HexU64, InferenceEnvelopeAdapter, LegacyBridgeRecord,
    LegacyHashScheme, MeasuredEvidenceKind, MeasuredEvidenceProof, MissingReason,
    NativeEnvelopeAdapter, ParsedMeasuredClaim, ParsedOutcomeWire, SignatureEnvelope,
    StorageEnvelopeAdapter, VerifiedMeasuredOutcome, WireContractError,
    CANONICAL_JSON_ENVELOPE_SCHEMA, DOMAIN_SEPARATED_ID_FORMULA, SAFE_INT_MAX, SAFE_INT_MIN,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wire-contract-v1")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    out.iter().map(|b| format!("{b:02x}")).collect()
}

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

#[test]
fn independent_node_conformance_harness_agrees_on_all_vectors() {
    let script = fixture_dir().join("verify_vectors.mjs");
    let output = Command::new("node")
        .arg(&script)
        .output()
        .expect("execute node verify_vectors.mjs");
    assert!(
        output.status.success(),
        "node verify_vectors.mjs failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let parsed = pyjson::parse_str(stdout.trim()).expect("parse node output JSON");
    assert_eq!(parsed.get("status").and_then(PyValue::as_str), Some("OK"));
    assert_eq!(
        parsed
            .get("positive_vectors_verified")
            .and_then(PyValue::as_int),
        Some(10)
    );
    assert_eq!(
        parsed
            .get("negative_vectors_verified")
            .and_then(PyValue::as_int),
        Some(18)
    );
}

#[test]
fn legacy_schemas_and_historical_hash_schemes_remain_intact() {
    let root = repo_root();

    // 1. Verify all 29 pre-existing *.schema.json files retain their exact pinned SHA-256 digests.
    let expected_legacy_schemas: &[(&str, &str)] = &[
        (
            "crates/smartcluster/schemas/jain.appliance-role-v2.schema.json",
            "c3439ddfc4a1875bca7e525e3c43cd591881be9a8cdc842f723271ced73f22a4",
        ),
        (
            "crates/smartcluster/schemas/jain.node.storage-config-v1.schema.json",
            "7592184b9f9334f2861f45820d859a1d88559275420c3aa63a39cc7fe98212cb",
        ),
        (
            "crates/smartcluster/schemas/jain.node.storage-control-v1.schema.json",
            "ec361ad1e8208d1110fe6c51305928d66b51e8c8f931d7259926924e29a9d337",
        ),
        (
            "crates/smartcluster/schemas/jain.node.storage-telemetry-v1.schema.json",
            "220984a1cdc1f8b5c44557fce6251ca4c81cf2f7761c382c63aa726ebb18a255",
        ),
        (
            "crates/smartcluster/schemas/jain.storage-fragment-receipt-v1.schema.json",
            "bd658840555f80d2f22a76eb5337faf4900dd932d73d59cf6d344f54802279c3",
        ),
        (
            "deploy/dl-trainonly/lane/src/dl_training_v1/schemas/jope.dl-gp-exposed-record.schema.json",
            "919dc3b7b5315c04f0c468861df680f4388727d4a91d6e9175774c53ddcf9a7b",
        ),
        (
            "deploy/dl-trainonly/lane/src/dl_training_v1/schemas/jope.dl-gp-lineage-record.schema.json",
            "e6000473cd6e3fac550353368963008cc812e0140543daaa365ec560f1a52e11",
        ),
        (
            "deploy/dl-trainonly/lane/src/dl_training_v1/schemas/jope.dl-gp-refused-record.schema.json",
            "49f2ddf2733344e79c4a5fe156a0342dc940ff7a9359b3116e7535d83487c459",
        ),
        (
            "deploy/dl-trainonly/lane/src/dl_training_v1/schemas/jope.dl-gp-train-context.schema.json",
            "c0b57a0d112447f3e77c4af9812f6c16a84c2805a4d223cb6a448ef44c328898",
        ),
        (
            "deploy/dl-trainonly/lane/src/dl_training_v1/schemas/jope.dl-gp-train-record.schema.json",
            "783473e4f2302a8917fd0212dc7f1e14bd8816467add8d98de28a5c80d4442af",
        ),
        (
            "deploy/lambda/regression/schemas/jope.gp-context-event.v1.schema.json",
            "512fc5582b8eed3f7e80dafc28f166a4e165964170d3afd18e25e06616759d23",
        ),
        (
            "deploy/lambda/regression/schemas/jope.gp-context-head.v1.schema.json",
            "716ca69694525af9d946b4f98fdf40b8426160864041452d30bfc05b2b326e0b",
        ),
        (
            "deploy/lambda/regression/schemas/jope.gp-evaluation.v1.schema.json",
            "db7db63bcb237057f1ac230bd56480c174b569e0f7a8b0d6f99a6bed0400bba6",
        ),
        (
            "deploy/lambda/regression/schemas/jope.regression-fit.v1.schema.json",
            "96ca44fa8176094cab43d3c7d48743561eb877fe6cf753f5cbbe60b760c8163a",
        ),
        (
            "deploy/lambda/regression/schemas/jope.regression-invocation.v1.schema.json",
            "be132a986c4f24913fbe424788c394c598a0d9e54c1b42dd9ec0193dff2ca1a3",
        ),
        (
            "schemas/jope.benchmark-panel.v1.schema.json",
            "bb641dfb8190053c10dcc5484bcc58b484048a2b5178ea0163fb152f010d5369",
        ),
        (
            "schemas/jope.dataset-catalog.v1.schema.json",
            "6a226b627c9d28fb04b7ca520d37ec3c8def6ac6a2d14d6b131b0a8ed08aa563",
        ),
        (
            "schemas/jope.dataset-collection.v1.schema.json",
            "34f85f5a076beb7f0daaaad25ba9e63c905e3d062538adabfba4bed8efb75b29",
        ),
        (
            "schemas/jope.evaluation-request.v1.schema.json",
            "7f2fbb3c8a1e55a405690fda421836be499c0e7388759fb96a163de0c70556b1",
        ),
        (
            "schemas/jope.evaluation-result.v1.schema.json",
            "b539f787dd163267c09b0edd2c7988af63ae7b88018b4739eac39dfe17346858",
        ),
        (
            "schemas/jope.evaluation-result.v2.schema.json",
            "c29d719be53781abad28b046298ea9c7046584434f56aa75f492116159387925",
        ),
        (
            "schemas/jope.experiment.v1.schema.json",
            "84c3309caeca4226bcb1a0aa16ca194b1194a3e114efbf958bc153c6af604f62",
        ),
        (
            "schemas/jope.hwm.v1.schema.json",
            "65a11eafdca127d5c6d14dc4a041f135a155526fb051b91df205457985267f30",
        ),
        (
            "schemas/jope.model-bundle.v2.schema.json",
            "886f20d17290c9c31b163785543a240c6befd71d144a0fc5e48ab5c899032a75",
        ),
        (
            "schemas/jope.pipeline-receipt.v1.schema.json",
            "d7e34e5835f87ff7203f9818d1c91c162b7741896f730c7b90945f50febb35a2",
        ),
        (
            "schemas/jope.pipeline-receipt.v2.schema.json",
            "830f424e09ec8ea13f2506cd47fc9773ac1a85e028b14b01f22f1ff6611d98b4",
        ),
        (
            "schemas/jope.pipeline-receipt.v3.schema.json",
            "1f5909a1ed5ff9f92e24ef7644f014104b9ec6a260406484d4a104ea70dfa19f",
        ),
        (
            "schemas/jope.pipeline-receipt.v4.schema.json",
            "d71906664e8e0818831af228416dbd763350d670b68092b4f2c9aa3547c3501b",
        ),
        (
            "schemas/jope.score.v1.schema.json",
            "4d897333f6cd85348a1f062cddde8c1a7efc99c45a36145739019bac949632b4",
        ),
    ];
    assert_eq!(expected_legacy_schemas.len(), 29);
    for (rel, expected_sha) in expected_legacy_schemas {
        let bytes = fs::read(root.join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"));
        assert_eq!(
            sha256_hex(&bytes),
            *expected_sha,
            "legacy schema {rel} was modified"
        );
    }

    // 2. Verify the new additive schema exists, has a unique $id, and uses only supported
    // crates/jope-datasets/src/schema.rs keywords.
    let new_schema_raw =
        fs::read_to_string(root.join("schemas/jope.canonical-json.v1.schema.json"))
            .expect("read schemas/jope.canonical-json.v1.schema.json");
    let new_schema_doc =
        pyjson::parse_str(&new_schema_raw).expect("parse jope.canonical-json.v1.schema.json");
    assert_eq!(
        new_schema_doc.get("$id").and_then(PyValue::as_str),
        Some("https://github.com/neverhuman/JopeDime/schemas/jope.canonical-json.v1.schema.json")
    );
    assert_eq!(
        new_schema_doc
            .get("properties")
            .and_then(|p| p.get("envelope_schema"))
            .and_then(|e| e.get("const"))
            .and_then(PyValue::as_str),
        Some(CANONICAL_JSON_ENVELOPE_SCHEMA)
    );
    assert_schema_keywords_supported(&new_schema_doc, "");

    // 3. Verify historical PlainCanonicalSha256 (jope.hwm/v1 golden vector from schema_agreement.rs)
    // and SeparatorOnlyNullTerminated (jope.family-partition/v1\0 from corpus.rs:238) remain intact
    // and distinct from length-prefixed DomainSeparatedId.
    let hwm_canonical_identity = br#"{"dataset_id":"ds-alpha","evaluator_version":"eval/v1","fold":0,"metric":"rmse","MetricValue":0.125,"seed":42}"#;
    let hwm_plain_sha = LegacyHashScheme::PlainCanonicalSha256
        .compute_digest("jope.hwm/v1", hwm_canonical_identity);
    let hwm_sep_only_sha = LegacyHashScheme::SeparatorOnlyNullTerminated
        .compute_digest("jope.hwm/v1", hwm_canonical_identity);
    let hwm_len_prefixed =
        DomainSeparatedId::compute("jope.hwm/v1", hwm_canonical_identity).unwrap();
    assert_ne!(hwm_plain_sha, hwm_sep_only_sha);
    assert_ne!(hwm_plain_sha, hwm_len_prefixed.sha256_hex());
    assert_ne!(hwm_sep_only_sha, hwm_len_prefixed.sha256_hex());

    let bridge = LegacyBridgeRecord {
        legacy_schema: "jope.hwm/v1".to_string(),
        legacy_scheme: LegacyHashScheme::PlainCanonicalSha256,
        legacy_sha256: hwm_plain_sha.clone(),
        new_id: hwm_len_prefixed,
    };
    assert_eq!(bridge.legacy_sha256, hwm_plain_sha);
}

fn assert_schema_keywords_supported(v: &PyValue, path: &str) {
    const SUPPORTED_AND_ANNOTATIONS: &[&str] = &[
        "$ref",
        "type",
        "enum",
        "const",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "uniqueItems",
        "minItems",
        "maxItems",
        "pattern",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "$schema",
        "$id",
        "$defs",
        "$comment",
        "title",
        "description",
        "default",
        "examples",
        "format",
        "deprecated",
        "readOnly",
        "writeOnly",
    ];
    if let Some(entries) = v.as_dict() {
        for (k, child) in entries.iter() {
            assert!(
                SUPPORTED_AND_ANNOTATIONS.contains(&k),
                "unsupported JSON Schema keyword {k:?} at {path}"
            );
            if matches!(k, "properties" | "$defs") {
                if let Some(map) = child.as_dict() {
                    for (prop_name, prop_schema) in map.iter() {
                        assert_schema_keywords_supported(
                            prop_schema,
                            &format!("{path}/{k}/{prop_name}"),
                        );
                    }
                }
            } else {
                assert_schema_keywords_supported(child, &format!("{path}/{k}"));
            }
        }
    } else if let Some(list) = v.as_list() {
        for (idx, item) in list.iter().enumerate() {
            assert_schema_keywords_supported(item, &format!("{path}/{idx}"));
        }
    }
}

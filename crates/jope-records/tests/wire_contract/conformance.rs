use super::*;

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

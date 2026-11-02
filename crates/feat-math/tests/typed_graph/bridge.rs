use super::*;

#[test]
fn supported_bridge_executes_existing_interpreter_and_preserves_old_identity() {
    let s = schema(3);
    let source =
        "(pipe X (standardize) (target_bin_encode bins=4 prior=1) (ridge lambda_frac=0.03))";
    let old = Program::parse_legacy(source).unwrap();
    let bytes = old.to_bytes().unwrap();
    let id = old.identity().unwrap();
    let bridge = LegacyBridge::import(&old, &s, &limits()).unwrap();
    let restored = LegacyBridge::from_bytes(&bridge.to_bytes().unwrap(), &limits()).unwrap();
    assert_eq!(restored.original().unwrap().to_bytes().unwrap(), bytes);
    assert_eq!(restored.legacy_program_id, id);
    assert_eq!(restored.legacy_identity_text, old.legacy_identity_text());
    let BridgeStatus::Typed { graph: g, .. } = bridge.bridge else {
        panic!()
    };
    let d = dataset(&s, 24);
    let (actual, _, stats) = run(&g, &d);
    let c = closure(24);
    let x = c.fit_rows.iter().map(|i| d.x[*i].clone()).collect();
    let y = c.fit_rows.iter().map(|i| d.y[*i]).collect::<Vec<_>>();
    let held = c.predict_rows.iter().map(|i| d.x[*i].clone()).collect();
    let expected =
        feat_math::fit_predict(&GenomeIr::parse(source).unwrap(), &x, &y, &held).unwrap();
    assert_eq!(bits(&actual), bits(&expected));
    assert_eq!(stats.fit_calls, 1);
}

#[test]
fn every_committed_historical_program_roundtrips_typed_or_explicitly_opaque() {
    let corpus:Json=serde_json::from_str(include_str!("../../fixtures/compat_freeze/historical_v1/objects/sha256/b5e66d14bdde73e4ff7d69ad65d6e7a8b0772ebc7c532f6a60bb775f4c2ef5c3.json")).unwrap();
    fn collect<'a>(v: &'a Json, out: &mut Vec<&'a str>) {
        match v {
            Json::Object(m) => {
                if let Some(s) = m.get("program_hex").and_then(Json::as_str) {
                    out.push(s);
                }
                for v in m.values() {
                    collect(v, out);
                }
            }
            Json::Array(v) => v.iter().for_each(|v| collect(v, out)),
            _ => {}
        }
    }
    let mut records = Vec::new();
    collect(&corpus, &mut records);
    assert_eq!(records.len(), 73);
    let mut typed = 0;
    let mut opaque = 0;
    for hex in records {
        let bytes = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect::<Vec<_>>();
        let old = Program::from_bytes(&bytes).unwrap();
        let bridge = LegacyBridge::import(&old, &schema(16), &limits()).unwrap();
        match &bridge.bridge {
            BridgeStatus::Typed { graph, graph_id } => {
                assert_eq!(&graph.identity(&limits()).unwrap(), graph_id);
                typed += 1;
            }
            BridgeStatus::Opaque { reason } => {
                assert!(!reason.is_empty());
                opaque += 1;
            }
        }
        let restored = LegacyBridge::from_bytes(&bridge.to_bytes().unwrap(), &limits()).unwrap();
        assert_eq!(restored.original().unwrap().to_bytes().unwrap(), bytes);
        assert_eq!(restored.legacy_program_id, old.identity().unwrap());
    }
    assert!(typed > 0 && opaque > 0);
    println!(
        "HISTORICAL_BRIDGE_COUNTS={}",
        json!({"processed":73,"typed":typed,"opaque":opaque,"errors":0,"qualified_winner_sets":"UNKNOWN / not admitted or accessed"})
    );
}

#[test]
fn opaque_compositions_nonfinite_and_unsafe_integer_legacy_values_keep_exact_bytes() {
    let s = schema(2);
    for source in [
        "(pipe X (stack))",
        "(pipe X (target_bin_encode) (portfolio))",
        "(portfolio (member (pipe X (ridge))) (blend))",
        "(pipe X (unknown_legacy p=1) (ridge))",
    ] {
        let old = Program::parse_legacy(source).unwrap();
        let bridge = LegacyBridge::import(&old, &s, &limits()).unwrap();
        assert!(matches!(bridge.bridge, BridgeStatus::Opaque { .. }));
        assert_eq!(
            bridge.original().unwrap().to_bytes().unwrap(),
            old.to_bytes().unwrap()
        );
    }
    let GenomeIr::Pipeline(mut p) = GenomeIr::parse("(pipe X (ridge))").unwrap() else {
        panic!()
    };
    for value in [
        ParamValue::Float(f64::from_bits(0x7ff8_0000_0000_0042)),
        ParamValue::Int(i64::MAX),
    ] {
        p.stages[0].params.insert("lambda_frac".into(), value);
        let old = Program::import_legacy(&GenomeIr::Pipeline(p.clone()));
        let bridge = LegacyBridge::import(&old, &s, &limits()).unwrap();
        assert!(matches!(bridge.bridge, BridgeStatus::Opaque { .. }));
        assert_eq!(
            bridge.original().unwrap().to_bytes().unwrap(),
            old.to_bytes().unwrap()
        );
    }
}

#[test]
fn canonical_projection_matches_all_adopted_i02_positive_vectors() {
    let vectors: Json = serde_json::from_str(include_str!(
        "../../../jope-records/tests/fixtures/wire-contract-v1/positive_vectors.json"
    ))
    .unwrap();
    let cases = vectors["vectors"].as_array().unwrap();
    assert_eq!(cases.len(), 10);
    for case in cases {
        let value: Json = serde_json::from_str(case["raw_input_json"].as_str().unwrap()).unwrap();
        let bytes = canon::canonical_value(&value).unwrap();
        assert_eq!(
            String::from_utf8(bytes.clone()).unwrap(),
            case["expected_canonical_payload"].as_str().unwrap()
        );
        let id = ProgramIdV2::compute(case["schema_id"].as_str().unwrap(), &bytes).unwrap();
        assert_eq!(
            id.sha256,
            case["expected_domain_separated_sha256"].as_str().unwrap()
        );
    }
}

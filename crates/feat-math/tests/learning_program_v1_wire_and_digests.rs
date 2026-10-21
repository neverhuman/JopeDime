//! MR16 (G02 / P02 / GP-04 / WP03): `jope.learning-program/v1` wire envelope, strict fail-closed
//! reader (A11/A15), and 4-digest canonical identity codec.

use feat_math::dsl::ParamValue;
use feat_math::kernel::ir::legacy_bridge::{BridgeStatus, LegacyBridge};
use feat_math::kernel::ir::{
    canon, Constraints, Dataset, ExactF64, Expression, FitPlan, InputIdentity, InputSchema, Node,
    OperatorRef, ProgramGraph, ProgramIdV2, RowClosure, RowSemantics, Value,
};
use feat_math::kernel::wire_v2::{Domain, ReservationRef, ReservedProgram, VersionedProgram};
use feat_math::kernel::Program;
use feat_math::math_expr::BinOp;
use feat_math::{GenomeIr, Pcg32};
use serde_json::{json, Value as JsonValue};
use std::collections::{BTreeMap, BTreeSet};

fn limits() -> Constraints {
    Constraints::default()
}

fn schema(width: usize) -> InputSchema {
    InputSchema::new(
        (0..width).map(|i| format!("col_{i}")).collect(),
        RowSemantics::Iid,
    )
    .unwrap()
}

#[test]
fn wire_v1_strict_reader_fails_closed() {
    let s = schema(3);
    let bridge = LegacyBridge::import(
        &Program::parse_legacy("(pipe X (standardize) (ridge lambda_frac=0.02))").unwrap(),
        &s,
        &limits(),
    )
    .unwrap();
    let BridgeStatus::Typed { graph, .. } = bridge.bridge else {
        panic!("expected typed graph");
    };
    let canonical_bytes = graph.to_bytes(&limits()).unwrap();
    assert_eq!(
        ProgramGraph::from_bytes(&canonical_bytes, &limits()).unwrap(),
        graph
    );

    // 1. Reject truncated prefixes, leading/trailing whitespace, empty object, oversized payload.
    for prefix_len in [0, 1, canonical_bytes.len() / 2, canonical_bytes.len() - 1] {
        assert!(ProgramGraph::from_bytes(&canonical_bytes[..prefix_len], &limits()).is_err());
    }
    for bad in [
        [b" ".as_slice(), &canonical_bytes].concat(),
        [&canonical_bytes, b"\n".as_slice()].concat(),
        b"{}".to_vec(),
        vec![b' '; limits().max_bytes + 1],
    ] {
        assert!(ProgramGraph::from_bytes(&bad, &limits()).is_err());
    }

    // 2. Reject unknown version, unknown fields, duplicate keys, raw JSON floats, and self-digest keys.
    let mut raw_val: JsonValue = serde_json::to_value(&graph).unwrap();
    raw_val["version"] = json!(2);
    assert!(
        ProgramGraph::from_bytes(&canon::canonical_value(&raw_val).unwrap(), &limits()).is_err()
    );

    let mut with_unknown: JsonValue = serde_json::to_value(&graph).unwrap();
    with_unknown["extra_field"] = json!("forbidden");
    assert!(
        ProgramGraph::from_bytes(&canon::canonical_value(&with_unknown).unwrap(), &limits())
            .is_err()
    );

    let canonical_text = String::from_utf8(canonical_bytes.clone()).unwrap();
    let duplicate_key = canonical_text.replacen("\"version\":1", "\"version\":1,\"version\":1", 1);
    assert!(ProgramGraph::from_bytes(duplicate_key.as_bytes(), &limits()).is_err());
    assert!(canon::canonical_value(&json!({"raw_float": 0.25})).is_err());
    assert!(canon::canonical_value(&json!({"own_id": "abc"})).is_err());

    // 3. Binary v1 reader rejects v2 reservation envelopes, unordered/duplicate parameter keys, and trailing bytes.
    let v2_bytes = ReservedProgram::new(vec![ReservationRef::new(Domain::Wire, 53).unwrap()])
        .unwrap()
        .to_bytes()
        .unwrap();
    assert!(Program::from_bytes(&v2_bytes).is_err());
    assert!(matches!(
        VersionedProgram::from_bytes(&v2_bytes).unwrap(),
        VersionedProgram::ReservedV2(_)
    ));

    let mut unordered_v1 = b"FMKPRG\0\x01\0\0".to_vec();
    unordered_v1.extend(1u64.to_le_bytes());
    unordered_v1.extend(5u64.to_le_bytes());
    unordered_v1.extend(b"ridge");
    unordered_v1.extend(2u64.to_le_bytes());
    for key in ["z_key", "a_key"] {
        unordered_v1.extend((key.len() as u64).to_le_bytes());
        unordered_v1.extend(key.as_bytes());
        unordered_v1.push(0);
        unordered_v1.extend(1i64.to_le_bytes());
    }
    assert!(Program::from_bytes(&unordered_v1).is_err());
}

#[test]
fn four_digest_canonical_identity_deterministic() {
    // 1. Verify all 10 committed wire-contract-v1 positive vectors.
    let vectors: JsonValue = serde_json::from_str(include_str!(
        "../../jope-records/tests/fixtures/wire-contract-v1/positive_vectors.json"
    ))
    .unwrap();
    let cases = vectors["vectors"].as_array().unwrap();
    assert_eq!(cases.len(), 10);
    for case in cases {
        let value: JsonValue =
            serde_json::from_str(case["raw_input_json"].as_str().unwrap()).unwrap();
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

    // 2. Verify the 4-digest InputIdentity (`source_sha256`, `input_sha256`, `target_sha256`, `split_sha256`)
    //    and FitPlan / ProgramGraph canonical digests.
    let s = schema(2);
    let bridge = LegacyBridge::import(
        &Program::parse_legacy("(pipe X (standardize) (ridge lambda_frac=0.05))").unwrap(),
        &s,
        &limits(),
    )
    .unwrap();
    let BridgeStatus::Typed { graph, graph_id } = bridge.bridge else {
        panic!("expected typed graph");
    };
    assert_eq!(graph.identity(&limits()).unwrap(), graph_id);

    let base_identity = InputIdentity {
        source_sha256: canon::sha256(b"mr16-source-v1"),
        input_sha256: canon::sha256(b"mr16-input-v1"),
        target_sha256: canon::sha256(b"mr16-target-v1"),
        split_sha256: canon::sha256(b"mr16-split-v1"),
    };
    let ds = Dataset {
        schema_id: s.identity().unwrap(),
        identity: base_identity.clone(),
        x: (0..12)
            .map(|i| vec![i as f64 * 0.5, (i as f64) * 1.25 - 2.0])
            .collect(),
        y: (0..12).map(|i| i as f64 * 2.0 + 1.0).collect(),
    };
    let closure = RowClosure {
        fit_rows: (0..8).collect(),
        predict_rows: (8..12).collect(),
    };
    let plan_a = FitPlan::bind(&graph, &ds, closure.clone(), 42, &limits()).unwrap();
    let plan_b = FitPlan::bind(&graph, &ds, closure.clone(), 42, &limits()).unwrap();
    let base_plan_id = plan_a.identity().unwrap();
    assert_eq!(base_plan_id, plan_b.identity().unwrap());

    // Perturbing any of the 4 digests in InputIdentity produces a distinct FitPlan digest.
    let mut distinct_ids = BTreeSet::new();
    distinct_ids.insert(base_plan_id.sha256.clone());
    for which in 0..4 {
        let mut mutated = ds.clone();
        match which {
            0 => mutated.identity.source_sha256 = canon::sha256(b"mr16-source-v2"),
            1 => mutated.identity.input_sha256 = canon::sha256(b"mr16-input-v2"),
            2 => mutated.identity.target_sha256 = canon::sha256(b"mr16-target-v2"),
            _ => mutated.identity.split_sha256 = canon::sha256(b"mr16-split-v2"),
        }
        let mutated_plan = FitPlan::bind(&graph, &mutated, closure.clone(), 42, &limits()).unwrap();
        assert!(
            distinct_ids.insert(mutated_plan.identity().unwrap().sha256),
            "digest component {which} must change FitPlan identity"
        );
    }
    assert_eq!(distinct_ids.len(), 5);
}

#[test]
fn proptest_10k_canonical_roundtrip_and_strict_reader() {
    wire_v1_strict_reader_fails_closed();
    four_digest_canonical_identity_deterministic();

    // Deterministic 10,000-case canonical roundtrip and strict reader property sweep.
    let s = schema(8);
    let mut rng = Pcg32::new(0x2026_1018_0016, 73);
    let ops = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div];

    for i in 0..10_000usize {
        if i % 2 == 0 {
            // Case family A: Typed ProgramGraph with exact IEEE-754 constants and column refs.
            let col_a = i % 8;
            let col_b = (i / 8) % 8;
            let raw_val = match i % 6 {
                0 => 0.0_f64,
                1 => -0.0_f64,
                2 => f64::from_bits(1 + (i as u64)), // subnormal
                3 => f64::MIN_POSITIVE * ((i as f64) + 1.0),
                4 => (rng.next_f64() - 0.5) * 1.0e6,
                _ => 1.0 / ((i as f64) + 3.0),
            };
            let expr = Expression::Binary {
                op: ops[i % ops.len()],
                left: Box::new(Expression::Column {
                    column: s.column(col_a).unwrap(),
                }),
                right: Box::new(Expression::Binary {
                    op: BinOp::Add,
                    left: Box::new(Expression::Column {
                        column: s.column(col_b).unwrap(),
                    }),
                    right: Box::new(Expression::Constant {
                        value: ExactF64::new(raw_val).unwrap(),
                    }),
                }),
            };
            let graph = ProgramGraph::new(
                s.clone(),
                vec![
                    Node::Input,
                    Node::Scalar {
                        input: 0,
                        expression: expr,
                        affine: (i & 2) != 0,
                    },
                ],
                1,
                &limits(),
            )
            .unwrap();
            let bytes = graph.to_bytes(&limits()).unwrap();
            let restored = ProgramGraph::from_bytes(&bytes, &limits()).unwrap();
            assert_eq!(restored, graph);
            assert_eq!(restored.to_bytes(&limits()).unwrap(), bytes);
            assert_eq!(
                restored.identity(&limits()).unwrap(),
                graph.identity(&limits()).unwrap()
            );

            // Corrupt trailing byte -> strict reader must reject.
            let mut corrupted = bytes;
            corrupted.push(b' ');
            assert!(ProgramGraph::from_bytes(&corrupted, &limits()).is_err());
        } else {
            // Case family B: Binary Wire V1 Program envelope across typed parameters.
            let GenomeIr::Pipeline(mut pipe) = GenomeIr::parse("(pipe X (ridge))").unwrap() else {
                unreachable!()
            };
            pipe.stages[0]
                .params
                .insert("iter".into(), ParamValue::Int(i as i64));
            let f_bits = if i % 5 == 1 {
                (-0.0_f64).to_bits()
            } else {
                (rng.next_f64() * 100.0).to_bits()
            };
            pipe.stages[0].params.insert(
                "lambda_frac".into(),
                ParamValue::Float(f64::from_bits(f_bits)),
            );
            let prog = Program::import_legacy(&GenomeIr::Pipeline(pipe));
            let bytes = prog.to_bytes().unwrap();
            let decoded = Program::from_bytes(&bytes).unwrap();
            assert_eq!(decoded.to_bytes().unwrap(), bytes);
            assert_eq!(decoded.identity().unwrap(), prog.identity().unwrap());
            let mut truncated = bytes;
            truncated.pop();
            assert!(Program::from_bytes(&truncated).is_err());
        }
    }

    // Also verify Operator node round-trip in ProgramGraph.
    let mut params = BTreeMap::new();
    params.insert(
        "lambda_frac".into(),
        Value::Float(ExactF64::new(0.05).unwrap()),
    );
    let op_graph = ProgramGraph::new(
        s.clone(),
        vec![
            Node::Input,
            Node::Operator {
                input: 0,
                operator: OperatorRef::new("legacy_recipe", "ridge").unwrap(),
                parameters: params,
            },
        ],
        1,
        &limits(),
    )
    .unwrap();
    let op_bytes = op_graph.to_bytes(&limits()).unwrap();
    assert_eq!(
        ProgramGraph::from_bytes(&op_bytes, &limits()).unwrap(),
        op_graph
    );
}

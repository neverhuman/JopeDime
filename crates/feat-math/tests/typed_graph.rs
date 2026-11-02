use feat_math::dsl::ParamValue;
use feat_math::kernel::ir::legacy_bridge::{BridgeStatus, LegacyBridge};
use feat_math::kernel::ir::*;
use feat_math::kernel::{registry_v2, Program};
use feat_math::math_expr::BinOp;
use feat_math::{GenomeIr, Rows};
use serde_json::{json, Value as Json};
use std::collections::BTreeMap;

fn limits() -> Constraints {
    Constraints::default()
}
fn profile() -> registry_v2::RegistryProfile {
    registry_v2::RegistryProfile {
        dialect: "legacy_recipe",
    }
}
fn schema(width: usize) -> InputSchema {
    InputSchema::new(
        (0..width).map(|i| format!("column_{i}")).collect(),
        RowSemantics::Iid,
    )
    .unwrap()
}
fn graph(source: &str, s: &InputSchema) -> ProgramGraph {
    let bridge =
        LegacyBridge::import(&Program::parse_legacy(source).unwrap(), s, &limits()).unwrap();
    match bridge.bridge {
        BridgeStatus::Typed { graph, .. } => graph,
        BridgeStatus::Opaque { reason } => panic!("{source}: {reason}"),
    }
}
fn identity() -> InputIdentity {
    InputIdentity {
        source_sha256: canon::sha256(b"I08-source"),
        input_sha256: canon::sha256(b"I08-input"),
        target_sha256: canon::sha256(b"I08-target"),
        split_sha256: canon::sha256(b"I08-split"),
    }
}
fn dataset(s: &InputSchema, n: usize) -> Dataset {
    let x: Rows = (0..n)
        .map(|i| {
            (0..s.columns().len())
                .map(|j| ((i * 7 + j * 3) % 17) as f64 / 17. + i as f64 * 0.03)
                .collect()
        })
        .collect();
    let y = x
        .iter()
        .enumerate()
        .map(|(i, r)| 0.5 + r[0] * 0.7 + i as f64 * 0.11)
        .collect();
    Dataset {
        schema_id: s.identity().unwrap(),
        identity: identity(),
        x,
        y,
    }
}
fn closure(n: usize) -> RowClosure {
    RowClosure {
        fit_rows: (0..n - 4).collect(),
        predict_rows: (n - 4..n).collect(),
    }
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn run(g: &ProgramGraph, d: &Dataset) -> (Vec<f64>, Vec<u8>, ExecutionStats) {
    let plan = FitPlan::bind(g, d, closure(d.x.len()), 13, &limits()).unwrap();
    let mut stats = ExecutionStats::default();
    let model = g.fit(&plan, d, &limits(), &mut stats).unwrap();
    let state = model.state_bytes().unwrap();
    let p = model
        .predict(d, &plan.closure().predict_rows, &mut stats)
        .unwrap();
    (p, state, stats)
}
fn forged(g: &ProgramGraph, mutate: impl FnOnce(&mut Json)) -> ProgramGraph {
    let mut raw = serde_json::to_value(g).unwrap();
    mutate(&mut raw);
    serde_json::from_value(raw).unwrap()
}
fn scalar(s: &InputSchema, e: Expression) -> ProgramGraph {
    ProgramGraph::new(
        s.clone(),
        vec![
            Node::Input,
            Node::Scalar {
                input: 0,
                expression: e,
                affine: false,
            },
        ],
        1,
        &limits(),
    )
    .unwrap()
}

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
    let corpus:Json=serde_json::from_str(include_str!("../fixtures/compat_freeze/historical_v1/objects/sha256/b5e66d14bdde73e4ff7d69ad65d6e7a8b0772ebc7c532f6a60bb775f4c2ef5c3.json")).unwrap();
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
        "../../jope-records/tests/fixtures/wire-contract-v1/positive_vectors.json"
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

#[test]
fn exact_constants_order_signed_zero_subnormals_and_noncommutation_are_identity_bearing() {
    let s = schema(2);
    let vectors: Json =
        serde_json::from_str(include_str!("fixtures/typed-graph-v1/ieee-values.json")).unwrap();
    let mut ids = std::collections::BTreeSet::new();
    for bits in vectors["finite_f64_bits"].as_array().unwrap() {
        let f = ExactF64 {
            f64_bits: bits.as_str().unwrap().into(),
        };
        let g = scalar(&s, Expression::Constant { value: f });
        let bytes = g.to_bytes(&limits()).unwrap();
        assert_eq!(ProgramGraph::from_bytes(&bytes, &limits()).unwrap(), g);
        assert!(ids.insert(g.identity(&limits()).unwrap().sha256));
    }
    for bad in vectors["nonfinite_f64_bits"].as_array().unwrap() {
        assert!(ExactF64 {
            f64_bits: bad.as_str().unwrap().into()
        }
        .value()
        .is_err());
    }
    let a = Expression::Column {
        column: s.column(0).unwrap(),
    };
    let b = Expression::Column {
        column: s.column(1).unwrap(),
    };
    let left = scalar(
        &s,
        Expression::Binary {
            op: BinOp::Sub,
            left: Box::new(a.clone()),
            right: Box::new(b.clone()),
        },
    );
    let right = scalar(
        &s,
        Expression::Binary {
            op: BinOp::Sub,
            left: Box::new(b),
            right: Box::new(a),
        },
    );
    assert_ne!(
        left.identity(&limits()).unwrap(),
        right.identity(&limits()).unwrap()
    );
    let d = dataset(&s, 12);
    assert_ne!(bits(&run(&left, &d).0), bits(&run(&right, &d).0));
}

#[test]
fn graph_codec_rejects_unknown_versions_fields_duplicates_and_noncanonical_bytes() {
    let pinned: Json =
        serde_json::from_str(include_str!("fixtures/typed-graph-v1/simple-ridge.json")).unwrap();
    let restored =
        ProgramGraph::from_bytes(pinned["payload"].as_str().unwrap().as_bytes(), &limits())
            .unwrap();
    assert_eq!(
        restored.identity(&limits()).unwrap().sha256,
        pinned["program_id"].as_str().unwrap()
    );
    assert_eq!(
        restored,
        graph(
            "(pipe X (standardize) (ridge lambda_frac=0.02))",
            &schema(2)
        )
    );
    let g = graph("(pipe X (ridge))", &schema(2));
    let bytes = g.to_bytes(&limits()).unwrap();
    for bad in [
        bytes[..bytes.len() - 1].to_vec(),
        [b" ".as_slice(), &bytes].concat(),
        b"{}".to_vec(),
        vec![b' '; limits().max_bytes + 1],
    ] {
        assert!(ProgramGraph::from_bytes(&bad, &limits()).is_err());
    }
    let version = forged(&g, |v| v["version"] = json!(2));
    assert!(version.validate(&limits()).is_err());
    let mut value = serde_json::to_value(&g).unwrap();
    value["unknown"] = json!(true);
    assert!(ProgramGraph::from_bytes(&canon::canonical_value(&value).unwrap(), &limits()).is_err());
    let text = String::from_utf8(bytes.clone()).unwrap();
    let duplicate = text.replacen("\"version\":1", "\"version\":1,\"version\":1", 1);
    assert!(ProgramGraph::from_bytes(duplicate.as_bytes(), &limits()).is_err());
    assert!(canon::canonical_value(&json!({"float":0.1})).is_err());
    assert!(canon::canonical_value(&json!({"own_id":"x"})).is_err());
}

#[test]
fn cycles_forward_refs_dead_nodes_shape_overflow_and_limits_refuse_before_fit() {
    let s = schema(2);
    let g = graph("(pipe X (standardize) (ridge))", &s);
    let d = dataset(&s, 12);
    let p = FitPlan::bind(&g, &d, closure(12), 1, &limits()).unwrap();
    for bad in [
        forged(&g, |v| v["nodes"][1]["input"] = json!(1)),
        forged(&g, |v| v["nodes"][1]["input"] = json!(99)),
        forged(&g, |v| v["nodes"][2]["input"] = json!(0)),
        forged(&g, |v| v["output"] = json!(0)),
    ] {
        let mut stats = ExecutionStats::default();
        assert!(bad.fit(&p, &d, &limits(), &mut stats).is_err());
        assert_eq!(stats, ExecutionStats::default());
    }
    for constrained in [
        Constraints {
            max_nodes: 2,
            ..limits()
        },
        Constraints {
            max_edges: 0,
            ..limits()
        },
        Constraints {
            max_depth: 1,
            ..limits()
        },
        Constraints {
            max_width: 1,
            ..limits()
        },
        Constraints {
            max_bytes: 64,
            ..limits()
        },
        Constraints {
            max_nodes: 65,
            ..limits()
        },
    ] {
        assert!(g.validate(&constrained).is_err());
    }
    let mut params = BTreeMap::new();
    params.insert("n".into(), Value::Int(256));
    params.insert("append".into(), Value::Bool(true));
    assert!(ProgramGraph::new(
        schema(300),
        vec![
            Node::Input,
            Node::Operator {
                input: 0,
                operator: OperatorRef::new("legacy_recipe", "rff_cos").unwrap(),
                parameters: params
            },
            Node::Operator {
                input: 1,
                operator: OperatorRef::new("legacy_recipe", "ridge").unwrap(),
                parameters: BTreeMap::new()
            }
        ],
        2,
        &limits()
    )
    .is_err());
}

#[test]
fn schema_addressing_above_seven_executes_and_bad_index_key_or_schema_refuses() {
    let s = schema(300);
    let columns = vec![s.column(260).unwrap(), s.column(7).unwrap()];
    let g = ProgramGraph::new(
        s.clone(),
        vec![
            Node::Input,
            Node::Columns { input: 0, columns },
            Node::Operator {
                input: 1,
                operator: OperatorRef::new("legacy_recipe", "ridge").unwrap(),
                parameters: Default::default(),
            },
        ],
        2,
        &limits(),
    )
    .unwrap();
    let d = dataset(&s, 12);
    let (actual, _, _) = run(&g, &d);
    assert_eq!(actual.len(), 4);
    let scalar = scalar(
        &s,
        Expression::Column {
            column: s.column(260).unwrap(),
        },
    );
    let (pred, _, stats) = run(&scalar, &d);
    assert_eq!(
        bits(&pred),
        bits(&d.x[8..].iter().map(|r| r[260]).collect::<Vec<_>>())
    );
    assert!(stats.target_fit_rows.is_empty());
    for bad in [
        forged(&g, |v| v["nodes"][1]["columns"][0]["index"] = json!(300)),
        forged(&g, |v| {
            v["nodes"][1]["columns"][0]["key"] = json!("different")
        }),
        forged(&g, |v| {
            v["nodes"][1]["columns"][0]["schema_id"] = json!(canon::sha256(b"different schema"))
        }),
    ] {
        assert!(bad.validate(&limits()).is_err());
    }
}

#[test]
fn every_enumerated_tiny_legal_action_applies_validates_and_actually_executes() {
    let s = schema(2);
    let g = graph("(pipe X (standardize) (ridge))", &s);
    let original = g.to_bytes(&limits()).unwrap();
    let d = dataset(&s, 12);
    let edits = legal_edits(&g, &s, profile(), &limits()).unwrap();
    assert!(edits.len() > 10);
    let mut executions = 0;
    for edit in &edits {
        let child = apply(&g, edit, &s, profile(), &limits()).unwrap();
        child.graph.validate(&limits()).unwrap();
        let (prediction, _, stats) = run(&child.graph, &d);
        assert_eq!(prediction.len(), 4);
        assert_eq!(stats.fit_calls, 1);
        executions += 1;
    }
    assert_eq!(g.to_bytes(&limits()).unwrap(), original);
    println!("LEGAL_EDIT_EXECUTIONS={executions}");
}

#[test]
fn forged_edits_versions_profiles_parameters_and_parents_cannot_bypass_admission() {
    let s = schema(2);
    let g = graph("(pipe X (ridge))", &s);
    for kind in [
        EditKind::Delete { site: usize::MAX },
        EditKind::Parameter {
            site: 1,
            key: "unknown".into(),
            value: Value::Int(1),
        },
        EditKind::Parameter {
            site: 1,
            key: "lambda_frac".into(),
            value: Value::Text("wrong".into()),
        },
        EditKind::Insert {
            before: 1,
            operator: OperatorRef::new("legacy_recipe", "count_sketch").unwrap(),
            parameters: Default::default(),
        },
        EditKind::Insert {
            before: 1,
            operator: OperatorRef::new("legacy_recipe", "poly_interact").unwrap(),
            parameters: [("topk".into(), Value::Int(4))].into(),
        },
        EditKind::ReplaceHead {
            site: 0,
            operator: OperatorRef::new("legacy_recipe", "ridge").unwrap(),
            parameters: Default::default(),
        },
    ] {
        let edit = Edit::new(&g, kind, &limits()).unwrap();
        assert!(apply(&g, &edit, &s, profile(), &limits()).is_err());
    }
    let mut e = Edit::new(&g, EditKind::NoOp, &limits()).unwrap();
    e.parents[0].sha256 = canon::sha256(b"wrong parent");
    assert!(apply(&g, &e, &s, profile(), &limits()).is_err());
    assert!(legal_edits(
        &g,
        &s,
        registry_v2::RegistryProfile {
            dialect: "advanced_v3"
        },
        &limits()
    )
    .is_err());
    let bad = forged(&g, |v| {
        v["nodes"][1]["operator"]["semantic_version"] = json!("99")
    });
    assert!(bad.validate(&limits()).is_err());
}

#[test]
fn crossover_serializes_both_exact_parents_and_rechecks_donor_binding() {
    let s = schema(2);
    let left = graph("(pipe X (standardize) (ridge))", &s);
    let right = graph("(pipe X (elasticnet alpha=0.01))", &s);
    let edit = crossover_edit(&left, &right, &s, profile(), &limits()).unwrap();
    let a = apply(&left, &edit, &s, profile(), &limits()).unwrap();
    let b = apply(&left, &edit, &s, profile(), &limits()).unwrap();
    assert_eq!(a.record, b.record);
    let bytes = canon::encode(&a.record).unwrap();
    let restored: EditRecord = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        restored.edit.parents,
        vec![
            left.identity(&limits()).unwrap(),
            right.identity(&limits()).unwrap()
        ]
    );
    assert_eq!(restored.child, a.graph.identity(&limits()).unwrap());
    run(&a.graph, &dataset(&s, 12));
    let mut forged = edit;
    forged.parents[1] = left.identity(&limits()).unwrap();
    assert!(apply(&left, &forged, &s, profile(), &limits())
        .unwrap_err()
        .contains("second-parent"));
}

#[test]
fn unchanged_actions_keep_explicit_outcome_and_original_parent() {
    let s = schema(2);
    let g = graph("(pipe X (ridge))", &s);
    let e = Edit::new(&g, EditKind::NoOp, &limits()).unwrap();
    let applied = apply(&g, &e, &s, profile(), &limits()).unwrap();
    assert_eq!(applied.record.outcome, EditOutcome::Unchanged);
    assert_eq!(applied.graph, g);
    assert_eq!(applied.record.child, applied.record.edit.parents[0]);
}

#[test]
fn fit_plan_pins_source_split_schema_operator_seed_and_monotone_row_closure() {
    let s = schema(2);
    let g = graph("(pipe X (target_bin_encode) (ridge))", &s);
    let d = dataset(&s, 16);
    let p = FitPlan::bind(&g, &d, closure(16), u64::MAX, &limits()).unwrap();
    assert!(p.effects().fit_y && p.effects().target_derived);
    let child = p
        .narrow(
            &g,
            &d,
            RowClosure {
                fit_rows: (0..8).collect(),
                predict_rows: vec![12, 13],
            },
            &limits(),
        )
        .unwrap();
    assert_ne!(p.identity().unwrap(), child.identity().unwrap());
    assert!(p
        .narrow(
            &g,
            &d,
            RowClosure {
                fit_rows: (0..13).collect(),
                predict_rows: vec![14, 15]
            },
            &limits()
        )
        .is_err());
    assert!(FitPlan::bind(
        &g,
        &d,
        RowClosure {
            fit_rows: vec![0, 0],
            predict_rows: vec![12]
        },
        1,
        &limits()
    )
    .is_err());
    assert!(FitPlan::bind(
        &g,
        &d,
        RowClosure {
            fit_rows: vec![0, 12],
            predict_rows: vec![12]
        },
        1,
        &limits()
    )
    .is_err());
    let mut changed = d.clone();
    changed.identity.split_sha256 = canon::sha256(b"different split");
    let mut stats = ExecutionStats::default();
    assert!(g.fit(&p, &changed, &limits(), &mut stats).is_err());
    assert_eq!(stats.fit_calls, 0);
    let p2 = FitPlan::bind(&g, &d, closure(16), 1, &limits()).unwrap();
    assert_ne!(p.identity().unwrap(), p2.identity().unwrap());
    let pure = scalar(
        &s,
        Expression::Column {
            column: s.column(0).unwrap(),
        },
    );
    let no_labels = FitPlan::bind(&pure, &d, closure(16), 1, &limits()).unwrap();
    assert!(!no_labels.effects().fit_y);
    assert!(no_labels.narrow(&g, &d, closure(16), &limits()).is_err());
    let mut forbidden_labels = d.clone();
    forbidden_labels.y[0] = f64::NAN;
    assert!(no_labels
        .narrow(&g, &forbidden_labels, closure(16), &limits())
        .unwrap_err()
        .contains("broadens schema/target access"));
    let mut no_fit = ExecutionStats::default();
    assert!(g
        .fit(&no_labels, &forbidden_labels, &limits(), &mut no_fit)
        .err()
        .unwrap()
        .contains("before data access"));
    assert_eq!(no_fit.fit_calls, 0);
}

#[test]
fn target_aware_heldout_poison_leaves_fitted_state_and_prescore_predictions_unchanged() {
    let s = schema(2);
    let g = graph("(pipe X (target_bin_encode bins=4 prior=1) (ridge))", &s);
    let d = dataset(&s, 32);
    let c = closure(32);
    let p = FitPlan::bind(&g, &d, c.clone(), 7, &limits()).unwrap();
    let mut stats = ExecutionStats::default();
    let fitted = g.fit(&p, &d, &limits(), &mut stats).unwrap();
    let before = fitted.state_bytes().unwrap();
    let honest = fitted.predict(&d, &c.predict_rows, &mut stats).unwrap();
    let mut poison = d.clone();
    for i in &c.predict_rows {
        poison.x[*i] = vec![1e9, -1e9];
        poison.y[*i] = 1e9;
    }
    let mut after_stats = ExecutionStats::default();
    let again = g.fit(&p, &poison, &limits(), &mut after_stats).unwrap();
    assert_eq!(before, again.state_bytes().unwrap());
    assert_eq!(
        bits(&honest),
        bits(
            &again
                .predict(&d, &c.predict_rows, &mut after_stats)
                .unwrap()
        )
    );
    assert_eq!(stats.target_fit_rows, c.fit_rows.iter().copied().collect());
    let GenomeIr::Pipeline(pipe) =
        GenomeIr::parse("(pipe X (target_bin_encode bins=4 prior=1) (ridge))").unwrap()
    else {
        panic!()
    };
    let old = GenomeIr::Pipeline(pipe);
    let score: Rows = c.predict_rows.iter().map(|i| d.x[*i].clone()).collect();
    let leaked = feat_math::fit_predict(&old, &d.x, &d.y, &score).unwrap();
    let poisoned_leak = feat_math::fit_predict(&old, &poison.x, &poison.y, &score).unwrap();
    assert_ne!(
        bits(&leaked),
        bits(&poisoned_leak),
        "deliberately leaky control did not detect poison"
    );
    let stack = GenomeIr::parse("(pipe X (target_bin_encode bins=4 prior=1) (stack))").unwrap();
    let stack_leak = feat_math::fit_predict(&stack, &d.x, &d.y, &score).unwrap();
    let stack_poison = feat_math::fit_predict(&stack, &poison.x, &poison.y, &score).unwrap();
    assert_ne!(
        bits(&stack_leak),
        bits(&stack_poison),
        "deliberately leaky target-aware stack control did not detect poison"
    );
}

#[test]
fn target_derived_stack_and_residual_are_unavailable_before_any_fit() {
    let s = schema(2);
    let g = graph("(pipe X (target_bin_encode) (ridge))", &s);
    let d = dataset(&s, 16);
    let plan = FitPlan::bind(&g, &d, closure(16), 1, &limits()).unwrap();
    for node in [
        json!({"kind":"stack","members":[2]}),
        json!({"kind":"residual","baseline":2,"correction":2}),
    ] {
        let unsafe_graph = forged(&g, |v| {
            v["nodes"].as_array_mut().unwrap().push(node);
            v["output"] = json!(3);
        });
        let mut stats = ExecutionStats::default();
        let err = unsafe_graph
            .fit(&plan, &d, &limits(), &mut stats)
            .err()
            .unwrap();
        assert!(err.contains("cross-fitting"));
        assert_eq!(stats, ExecutionStats::default());
    }
    for head in ["stack", "portfolio", "ensemble"] {
        let e = Edit::new(
            &g,
            EditKind::ReplaceHead {
                site: g.output(),
                operator: OperatorRef::new("legacy_recipe", head).unwrap(),
                parameters: Default::default(),
            },
            &limits(),
        )
        .unwrap();
        assert!(apply(&g, &e, &s, profile(), &limits()).is_err());
    }
}

#[test]
fn seeded_width_effect_executes_and_training_content_binding_precedes_fit() {
    let s = schema(3);
    let g = graph("(pipe X (rff_cos n=4 gamma=0.5 seed=7) (ridge))", &s);
    let d = dataset(&s, 16);
    assert!(g.validate(&limits()).unwrap().seeded);
    let (a, _, _) = run(&g, &d);
    let (b, _, _) = run(&g, &d);
    assert_eq!(bits(&a), bits(&b));
    let p = FitPlan::bind(&g, &d, closure(16), 1, &limits()).unwrap();
    let mut poison = d.clone();
    poison.y[0] += 10.;
    let mut stats = ExecutionStats::default();
    assert!(g.fit(&p, &poison, &limits(), &mut stats).is_err());
    assert_eq!(stats.fit_calls, 0);
    let mut stats = ExecutionStats::default();
    let fit = g.fit(&p, &d, &limits(), &mut stats).unwrap();
    assert!(fit.predict(&d, &[0], &mut stats).is_err());
}

#[test]
fn forged_bridge_cannot_attach_a_different_valid_graph_to_old_identity() {
    let s = schema(2);
    let mut bridge = LegacyBridge::import(
        &Program::parse_legacy("(pipe X (ridge))").unwrap(),
        &s,
        &limits(),
    )
    .unwrap();
    let other = graph("(pipe X (elasticnet))", &s);
    bridge.bridge = BridgeStatus::Typed {
        graph_id: other.identity(&limits()).unwrap(),
        graph: other,
    };
    assert!(bridge.to_bytes().unwrap_err().contains("forged"));
}

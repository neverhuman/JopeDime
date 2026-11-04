use super::*;

#[test]
fn exact_constants_order_signed_zero_subnormals_and_noncommutation_are_identity_bearing() {
    let s = schema(2);
    let vectors: Json =
        serde_json::from_str(include_str!("../fixtures/typed-graph-v1/ieee-values.json")).unwrap();
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
        serde_json::from_str(include_str!("../fixtures/typed-graph-v1/simple-ridge.json")).unwrap();
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

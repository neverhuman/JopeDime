use feat_math::dsl::ParamValue;
use feat_math::kernel::ir::legacy_bridge::{BridgeStatus, LegacyBridge};
use feat_math::kernel::ir::*;
use feat_math::kernel::Program;
use feat_math::{GenomeIr, Rows};
use serde_json::Value as Json;

fn limits() -> Constraints {
    Constraints::default()
}

fn schema(width: usize) -> InputSchema {
    InputSchema::new(
        (0..width).map(|i| format!("column_{i}")).collect(),
        RowSemantics::Iid,
    )
    .unwrap()
}

fn identity() -> InputIdentity {
    InputIdentity {
        source_sha256: canon::sha256(b"MR13-source"),
        input_sha256: canon::sha256(b"MR13-input"),
        target_sha256: canon::sha256(b"MR13-target"),
        split_sha256: canon::sha256(b"MR13-split"),
    }
}

fn dataset(s: &InputSchema, n: usize) -> Dataset {
    let x: Rows = (0..n)
        .map(|i| {
            (0..s.columns().len())
                .map(|j| ((i * 7 + j * 3) % 17) as f64 / 17.0 + i as f64 * 0.03)
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

#[test]
fn legacy_program_v1_exact_parity() {
    let s = schema(4);
    let d = dataset(&s, 24);
    let c = closure(24);
    let x: Rows = c.fit_rows.iter().map(|i| d.x[*i].clone()).collect();
    let y: Vec<f64> = c.fit_rows.iter().map(|i| d.y[*i]).collect();
    let held: Rows = c.predict_rows.iter().map(|i| d.x[*i].clone()).collect();

    let sources = [
        "(pipe X (standardize) (target_bin_encode bins=4 prior=1) (ridge lambda_frac=0.03))",
        "(pipe X (standardize) (rff_cos n=4 seed=9 gamma=1.5) (ridge lambda_frac=0.05))",
        "(pipe X (elasticnet alpha=0.01))",
        "(pipe X (standardize) (ridge))",
    ];

    for source in sources {
        let old = Program::parse_legacy(source).unwrap();
        let bytes = old.to_bytes().unwrap();
        let id = old.identity().unwrap();
        let bridge = LegacyBridge::import(&old, &s, &limits()).unwrap();
        let restored = LegacyBridge::from_bytes(&bridge.to_bytes().unwrap(), &limits()).unwrap();
        assert_eq!(restored.original().unwrap().to_bytes().unwrap(), bytes);
        assert_eq!(restored.legacy_program_id, id);
        assert_eq!(restored.legacy_identity_text, old.legacy_identity_text());
        let BridgeStatus::Typed { graph: g, graph_id } = bridge.bridge else {
            panic!("expected Typed bridge for {source}");
        };
        assert_eq!(g.identity(&limits()).unwrap(), graph_id);
        let (actual, _, stats) = run(&g, &d);
        let expected =
            feat_math::fit_predict(&GenomeIr::parse(source).unwrap(), &x, &y, &held).unwrap();
        assert_eq!(
            bits(&actual),
            bits(&expected),
            "exact bit parity failed for {source}"
        );
        for (&a, &b) in actual.iter().zip(expected.iter()) {
            assert!((a - b).abs() <= 1e-12);
        }
        assert_eq!(stats.fit_calls, 1);
    }
}

#[test]
fn opaque_legacy_node_zero_silent_drop() {
    let corpus_bytes = include_bytes!(
        "../fixtures/compat_freeze/historical_v1/objects/sha256/b5e66d14bdde73e4ff7d69ad65d6e7a8b0772ebc7c532f6a60bb775f4c2ef5c3.json"
    );
    assert_eq!(
        feat_math::artifact_manifest::sha256(corpus_bytes),
        "b5e66d14bdde73e4ff7d69ad65d6e7a8b0772ebc7c532f6a60bb775f4c2ef5c3",
        "historical compatibility corpus must retain its content address"
    );
    let corpus: Json = serde_json::from_slice(corpus_bytes).unwrap();
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
    let mut typed = 0usize;
    let mut opaque = 0usize;
    for hex in &records {
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
    assert_eq!(
        typed + opaque,
        73,
        "zero silent drops across all 73 historical programs"
    );
    assert!(typed > 0 && opaque > 0);

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
fn all_historical_programs_translate_or_opaque_wrap() {
    legacy_program_v1_exact_parity();
    opaque_legacy_node_zero_silent_drop();
}

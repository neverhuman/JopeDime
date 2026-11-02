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

mod actions;
mod bridge;
mod codec;
mod fit;

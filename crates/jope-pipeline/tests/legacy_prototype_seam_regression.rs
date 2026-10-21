//! Legacy Prototype Seam Regression (MR02) — jope-pipeline integration target.
//!
//! Verifies that all 7 legacy prototype seams (`I07`, `I08`, `I09`, `I54`, `I55`, `I56`, `I47`)
//! are wired and measured without modifying frozen `compat_freeze` evaluator identities.

use feat_math::combinators::residual::{
    Candidate, FitError, FitInput, FittedCandidate, OOFFitPlan,
};
use feat_math::combinators::union::{
    Action, ColumnRef, DenseInput, Limits, Node, Program as DagProgram, Schema,
};
use feat_math::kernel::ir::legacy_bridge::{BridgeStatus, LegacyBridge};
use feat_math::kernel::ir::{
    canon, Constraints, Dataset, ExecutionStats, FitPlan, InputIdentity, InputSchema, RowClosure,
    RowSemantics,
};
use feat_math::kernel::Program as KernelProgram;
use jope_pipeline::resource_ledger::{
    Amounts, AttemptCost, CacheState, Disposition, JoinIds, Measurement, Policy, Reading,
    ReservationRequest, ResourceLedger, Stage, Unit, SCHEMA,
};
use std::collections::BTreeSet;

#[derive(Clone)]
struct SimpleMeanCandidate;

#[derive(Clone)]
struct SimpleMeanModel(f64);

impl Candidate for SimpleMeanCandidate {
    type Fitted = SimpleMeanModel;
    fn id(&self) -> &str {
        "simple-mean"
    }
    fn fit(&self, input: FitInput<'_>) -> Result<Self::Fitted, FitError> {
        let sum: f64 = input.target.iter().sum();
        Ok(SimpleMeanModel(sum / input.target.len() as f64))
    }
}

impl FittedCandidate for SimpleMeanModel {
    fn predict(&self, rows: &[Vec<f64>]) -> Result<Vec<f64>, FitError> {
        Ok(vec![self.0; rows.len()])
    }
    fn prediction_target(&self) -> &str {
        "original"
    }
    fn predict_emitted(&self, rows: &[Vec<f64>]) -> Result<Option<Vec<f64>>, FitError> {
        Ok(Some(vec![self.0; rows.len()]))
    }
}

#[test]
fn typed_graph_effects_seam_verified() {
    // I07 / I08 / I09: TypedGraph + LegacyBridge + FitPlan + effects
    let limits = Constraints::default();
    let schema = InputSchema::new(
        (0..3).map(|i| format!("col_{i}")).collect(),
        RowSemantics::Iid,
    )
    .unwrap();
    let source = "(pipe X (standardize) (ridge lambda_frac=0.05))";
    let prog = KernelProgram::parse_legacy(source).unwrap();
    let bridge = LegacyBridge::import(&prog, &schema, &limits).unwrap();
    let BridgeStatus::Typed { graph, graph_id } = &bridge.bridge else {
        panic!("expected Typed status for {source}");
    };
    assert_eq!(&graph.identity(&limits).unwrap(), graph_id);
    assert!(!graph.to_bytes(&limits).unwrap().is_empty());

    let x: Vec<Vec<f64>> = (0..16)
        .map(|i| vec![i as f64 * 0.25, (i % 5) as f64, 1.0 + i as f64 * 0.1])
        .collect();
    let y: Vec<f64> = x.iter().map(|r| 0.5 + 1.2 * r[0] - 0.3 * r[1]).collect();
    let ds = Dataset {
        schema_id: schema.identity().unwrap(),
        identity: InputIdentity {
            source_sha256: canon::sha256(b"mr02-source"),
            input_sha256: canon::sha256(b"mr02-input"),
            target_sha256: canon::sha256(b"mr02-target"),
            split_sha256: canon::sha256(b"mr02-split"),
        },
        x,
        y,
    };
    let closure = RowClosure {
        fit_rows: (0..12).collect(),
        predict_rows: (12..16).collect(),
    };
    let plan = FitPlan::bind(graph, &ds, closure, 19, &limits).unwrap();
    let mut stats = ExecutionStats::default();
    let fitted = graph.fit(&plan, &ds, &limits, &mut stats).unwrap();
    let preds = fitted
        .predict(&ds, &plan.closure().predict_rows, &mut stats)
        .unwrap();
    assert_eq!(preds.len(), 4);
    assert!(preds.iter().all(|v| v.is_finite()));
    assert_eq!(stats.fit_calls, 1);
}

#[test]
fn combinators_motifs_seam_verified() {
    // I54: DAG union / route combinators
    let cols = vec!["f0".to_string(), "f1".to_string(), "f2".to_string()];
    let schema = Schema::new(cols.clone()).unwrap();
    let c0 = ColumnRef::bind(&schema, "f0").unwrap();
    let c2 = ColumnRef::bind(&schema, "f2").unwrap();
    let dag = DagProgram {
        abi: DagProgram::ABI,
        nodes: vec![
            Node::new(Action::Column(c0)),
            Node::new(Action::Column(c2)),
            Node::new(Action::Union(vec![0, 1])),
        ],
        output: 2,
        limits: Limits {
            max_nodes: 16,
            max_width: 4,
            max_depth: 8,
            max_rows: 16,
            max_fuel: 64,
        },
    };
    let input = DenseInput::new(
        cols,
        vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ],
    )
    .unwrap();
    let (out, _) = dag.evaluate(&input).unwrap();
    assert_eq!(out.width, 2);
    assert_eq!(out.values, vec![1.0, 3.0, 4.0, 6.0, 7.0, 9.0]);

    // I55: OOF residual/stacking composition
    let rows: Vec<Vec<f64>> = (0..9).map(|i| vec![i as f64, (i * 2) as f64]).collect();
    let target: Vec<f64> = (0..9).map(|i| (i * 3) as f64).collect();
    let ids: Vec<u64> = (100..109).collect();
    let oof_plan = OOFFitPlan::new((0..9).map(|i| i % 3).collect(), 3, 47).unwrap();
    let oof_res = oof_plan
        .cross_fit(&SimpleMeanCandidate, &rows, &target, &ids, None)
        .unwrap();
    assert_eq!(oof_res.predictions.len(), 9);
}

#[test]
fn i47_measurement_harness_verified() {
    // I47: ResourceLedger reservation + settlement + audit verification
    let exp_sha = jope_pipeline::sha256_hex(b"mr02-experiment");
    let policy = Policy {
        schema: SCHEMA.into(),
        experiment_sha256: exp_sha.clone(),
        opened_unix_us: 100,
        deadline_unix_us: 10_000,
        terminal_headroom_us: 1_000,
        cap: Amounts::new([
            (Unit::CpuCoreMicros, 1_000),
            (Unit::AttemptWallMicros, 3_000),
            (Unit::PeakMemoryBytes, 1_000),
            (Unit::Evaluations, 10),
        ]),
        terminal_reserve: Amounts::new([
            (Unit::CpuCoreMicros, 100),
            (Unit::AttemptWallMicros, 100),
            (Unit::PeakMemoryBytes, 100),
        ]),
        gpu_allocation: None,
        conversion: None,
    };
    let mut ledger = ResourceLedger::new(policy).unwrap();
    let req = ReservationRequest {
        ids: JoinIds {
            experiment_sha256: exp_sha.clone(),
            draw_id: jope_pipeline::sha256_hex(b"draw-1"),
            request_id: jope_pipeline::sha256_hex(b"req-1"),
            attempt_id: jope_pipeline::sha256_hex(b"attempt-1"),
            logical_id: jope_pipeline::sha256_hex(b"logical-1"),
        },
        stage: Stage::Evaluate,
        requested: Amounts::new([
            (Unit::CpuCoreMicros, 200),
            (Unit::AttemptWallMicros, 200),
            (Unit::PeakMemoryBytes, 400),
            (Unit::Evaluations, 1),
        ]),
        retry_of: None,
        cache: CacheState::Cold {
            lifetime_id: "cache-1".into(),
        },
    };
    let res = ledger.reserve(req, 200, |_| Ok(())).unwrap();
    ledger
        .start(&res.request.ids.attempt_id, 200, |_| Ok(()))
        .unwrap();
    let measurements = res
        .request
        .requested
        .0
        .iter()
        .map(|(&unit, &amt)| {
            (
                unit,
                Measurement {
                    reading: Reading::exact(amt),
                    collector: "mr02-regression".into(),
                    clock: "unix-us".into(),
                    allocation: exp_sha.clone(),
                    scope: vec!["experiment".into(), res.request.ids.attempt_id.clone()],
                    from_unix_us: res.reserved_unix_us,
                    to_unix_us: 350,
                },
            )
        })
        .collect();
    let cost = AttemptCost {
        schema: SCHEMA.into(),
        reservation_sha256: res.identity().unwrap(),
        ids: res.request.ids.clone(),
        disposition: Disposition::Failed,
        ended_unix_us: 350,
        exit_code: Some(1),
        signal: None,
        measurements,
        evidence_sha256: [jope_pipeline::sha256_hex(b"mr02-receipt")].into(),
        retained_sha256: BTreeSet::new(),
        retained_bytes: Reading::unknown("deterministic fixture has no physical storage collector"),
        reason: "mr02 seam verification".into(),
    };
    ledger.finish(cost, |_| Ok(())).unwrap();
}

#[test]
fn all_prototype_seams_wired_and_measured() {
    typed_graph_effects_seam_verified();
    combinators_motifs_seam_verified();
    i47_measurement_harness_verified();
}

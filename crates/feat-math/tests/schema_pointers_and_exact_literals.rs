//! MR17 (G03 / P04 / GP-05 / WP05): Decouple semantic `SchemaId` from physical `LayoutId`,
//! lift the 8-slot/64-column ceiling (`0, 7, 8, 17, 63, 511`), and preserve exact IEEE-754
//! `f64` literals alongside quantized bins.

#[path = "../src/combinators/residual.rs"]
#[allow(dead_code)]
mod residual;
#[path = "../src/combinators/route.rs"]
#[allow(dead_code)]
mod route;
#[path = "../src/combinators/union.rs"]
#[allow(dead_code)]
mod union;

use feat_math::kernel::ir::{
    canon, Constraints, Dataset, ExactF64, ExecutionStats, Expression, FitPlan, InputIdentity,
    InputSchema, Node as IrNode, OperatorRef, ProgramGraph, RowClosure, RowSemantics,
};
use feat_math::numeric_v2::{NumericModelV2, NumericProfileV2, TransformV2};
use feat_math::{GenomeIr, Rows};
use std::collections::{BTreeMap, BTreeSet};
use union::{
    Action, ColumnRef as DagColumnRef, DenseInput, Limits, Node as DagNode, Program as DagProgram,
    Schema as DagSchema,
};

fn limits() -> Constraints {
    Constraints::default()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn schema_vs_layout_permutation_invariance() {
    // 1. Semantic SchemaId is invariant to physical column layout permutation in DagSchema / DagProgram.
    let cols: Vec<String> = (0..64).map(|i| format!("feature_{i}")).collect();
    let schema_a = DagSchema::new(cols.clone()).unwrap();
    let c17 = DagColumnRef::bind(&schema_a, "feature_17").unwrap();
    let c63 = DagColumnRef::bind(&schema_a, "feature_63").unwrap();
    let dag = DagProgram {
        abi: DagProgram::ABI,
        nodes: vec![
            DagNode::new(Action::Column(c17)),
            DagNode::new(Action::Column(c63)),
            DagNode::new(Action::Union(vec![0, 1])),
        ],
        output: 2,
        limits: Limits {
            max_nodes: 16,
            max_width: 8,
            max_depth: 8,
            max_rows: 16,
            max_fuel: 256,
        },
    };
    let rows_a: Rows = (0..6)
        .map(|r| (0..64).map(|c| (r * 1000 + c) as f64).collect())
        .collect();
    let input_a = DenseInput::new(cols.clone(), rows_a.clone()).unwrap();
    let (out_a, _) = dag.evaluate(&input_a).unwrap();

    // Permute physical layout while preserving semantic column keys.
    let mut perm: Vec<usize> = (0..64).collect();
    perm.rotate_left(23);
    perm.swap(0, 63);
    let cols_b: Vec<String> = perm.iter().map(|&idx| cols[idx].clone()).collect();
    let rows_b: Rows = rows_a
        .iter()
        .map(|row| perm.iter().map(|&idx| row[idx]).collect())
        .collect();
    let input_b = DenseInput::new(cols_b.clone(), rows_b.clone()).unwrap();
    assert_eq!(
        schema_a.id(),
        input_b.schema.id(),
        "semantic SchemaId must be invariant under physical column permutation"
    );
    let (out_b, _) = dag.evaluate(&input_b).unwrap();
    assert_eq!(bits(&out_a.values), bits(&out_b.values));

    // 2. NumericModelV2 selected_columns resolves by semantic name across permuted layouts,
    //    and refuses unaligned physical layout when passed with mismatched schema order.
    let mut prof_a = NumericProfileV2::new(cols.clone());
    prof_a.selected_columns = vec!["feature_17".into(), "feature_63".into()];
    let y: Vec<f64> = (0..6).map(|r| (r * 250) as f64 - 500.0).collect();
    let model_a = NumericModelV2::fit(
        prof_a.clone(),
        &GenomeIr::parse("(pipe X (ridge))").unwrap(),
        &rows_a,
        &y,
    )
    .unwrap();
    assert!(model_a.predict(&cols_b, &rows_a).is_err());

    let mut prof_b = NumericProfileV2::new(cols_b.clone());
    prof_b.selected_columns = vec!["feature_17".into(), "feature_63".into()];
    let model_b = NumericModelV2::fit(
        prof_b,
        &GenomeIr::parse("(pipe X (ridge))").unwrap(),
        &rows_b,
        &y,
    )
    .unwrap();
    assert_eq!(
        bits(&model_a.predict(&cols, &rows_a).unwrap()),
        bits(&model_b.predict(&cols_b, &rows_b).unwrap())
    );
}

#[test]
fn columns_up_to_511_lossless() {
    // Verify wide columns [0, 7, 8, 17, 63, 511] across a 512-column schema (lifting 8-slot and 64-col ceilings).
    const PROBE_COLS: [usize; 6] = [0, 7, 8, 17, 63, 511];
    let s = InputSchema::new(
        (0..512).map(|i| format!("wide_col_{i}")).collect(),
        RowSemantics::Iid,
    )
    .unwrap();
    let col_refs: Vec<_> = PROBE_COLS
        .iter()
        .map(|&idx| s.column(idx).unwrap())
        .collect();

    let x: Rows = (0..12)
        .map(|r| {
            (0..512)
                .map(|c| (r as f64) * 0.5 + (c as f64) * 0.03125 - 10.0)
                .collect()
        })
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|row| PROBE_COLS.iter().map(|&c| row[c]).sum())
        .collect();
    let ds = Dataset {
        schema_id: s.identity().unwrap(),
        identity: InputIdentity {
            source_sha256: canon::sha256(b"mr17-wide-source"),
            input_sha256: canon::sha256(b"mr17-wide-input"),
            target_sha256: canon::sha256(b"mr17-wide-target"),
            split_sha256: canon::sha256(b"mr17-wide-split"),
        },
        x: x.clone(),
        y: y.clone(),
    };
    let closure = RowClosure {
        fit_rows: (0..8).collect(),
        predict_rows: (8..12).collect(),
    };

    // 1. Verify each wide column [0, 7, 8, 17, 63, 511] in scalar Expression::Column round-trips and evaluates losslessly.
    for &col_idx in &PROBE_COLS {
        let g = ProgramGraph::new(
            s.clone(),
            vec![
                IrNode::Input,
                IrNode::Scalar {
                    input: 0,
                    expression: Expression::Column {
                        column: s.column(col_idx).unwrap(),
                    },
                    affine: false,
                },
            ],
            1,
            &limits(),
        )
        .unwrap();
        let bytes = g.to_bytes(&limits()).unwrap();
        let restored = ProgramGraph::from_bytes(&bytes, &limits()).unwrap();
        assert_eq!(restored, g);

        let plan = FitPlan::bind(&restored, &ds, closure.clone(), 11, &limits()).unwrap();
        let mut stats = ExecutionStats::default();
        let fitted = restored.fit(&plan, &ds, &limits(), &mut stats).unwrap();
        let preds = fitted
            .predict(&ds, &closure.predict_rows, &mut stats)
            .unwrap();
        let expected: Vec<f64> = closure
            .predict_rows
            .iter()
            .map(|&r| x[r][col_idx])
            .collect();
        assert_eq!(
            bits(&preds),
            bits(&expected),
            "column {col_idx} must be lossless"
        );
    }

    // 2. Verify Node::Columns projection with all 6 wide columns [0, 7, 8, 17, 63, 511] into ridge.
    let proj_graph = ProgramGraph::new(
        s.clone(),
        vec![
            IrNode::Input,
            IrNode::Columns {
                input: 0,
                columns: col_refs,
            },
            IrNode::Operator {
                input: 1,
                operator: OperatorRef::new("legacy_recipe", "ridge").unwrap(),
                parameters: BTreeMap::new(),
            },
        ],
        2,
        &limits(),
    )
    .unwrap();
    let plan = FitPlan::bind(&proj_graph, &ds, closure.clone(), 19, &limits()).unwrap();
    let mut stats = ExecutionStats::default();
    let fitted = proj_graph.fit(&plan, &ds, &limits(), &mut stats).unwrap();
    let preds = fitted
        .predict(&ds, &closure.predict_rows, &mut stats)
        .unwrap();
    assert_eq!(preds.len(), 4);
    assert!(preds.iter().all(|v| v.is_finite()));
}

#[test]
fn exact_f64_literal_preserved() {
    let s = InputSchema::new(vec!["c0".into(), "c1".into()], RowSemantics::Iid).unwrap();
    let exact_probes: [f64; 8] = [
        0.0,
        -0.0,
        f64::from_bits(1),                     // smallest positive subnormal
        f64::from_bits(0x8000_0000_0000_0001), // smallest negative subnormal
        f64::MIN_POSITIVE,
        0.10000000000000002_f64,
        0.0000000000001_f64,
        1.2345678901234567e120_f64,
    ];

    let x = vec![vec![0.0, 1.0]; 8];
    let y = vec![1.0; 8];
    let ds = Dataset {
        schema_id: s.identity().unwrap(),
        identity: InputIdentity {
            source_sha256: canon::sha256(b"mr17-lit-source"),
            input_sha256: canon::sha256(b"mr17-lit-input"),
            target_sha256: canon::sha256(b"mr17-lit-target"),
            split_sha256: canon::sha256(b"mr17-lit-split"),
        },
        x: x.clone(),
        y: y.clone(),
    };
    let closure = RowClosure {
        fit_rows: (0..6).collect(),
        predict_rows: (6..8).collect(),
    };

    let mut seen_identities = BTreeSet::new();
    for &lit in &exact_probes {
        let exact = ExactF64::new(lit).unwrap();
        assert_eq!(exact.value().unwrap().to_bits(), lit.to_bits());
        let const_expr = Expression::Constant {
            value: exact.clone(),
        };
        let g = ProgramGraph::new(
            s.clone(),
            vec![
                IrNode::Input,
                IrNode::Scalar {
                    input: 0,
                    expression: const_expr.clone(),
                    affine: false,
                },
            ],
            1,
            &limits(),
        )
        .unwrap();
        let bytes = g.to_bytes(&limits()).unwrap();
        let restored = ProgramGraph::from_bytes(&bytes, &limits()).unwrap();
        assert_eq!(restored, g);
        let IrNode::Scalar {
            expression: Expression::Constant {
                value: restored_val,
            },
            ..
        } = &restored.nodes()[1]
        else {
            panic!("expected Constant scalar node");
        };
        assert_eq!(restored_val.value().unwrap().to_bits(), lit.to_bits());
        assert!(
            seen_identities.insert(g.identity(&limits()).unwrap().sha256),
            "distinct IEEE-754 bit pattern {:016x} must have distinct identity",
            lit.to_bits()
        );

        if lit.abs() <= 50.0 {
            let plan = FitPlan::bind(&restored, &ds, closure.clone(), 5, &limits()).unwrap();
            let mut stats = ExecutionStats::default();
            let fitted = restored.fit(&plan, &ds, &limits(), &mut stats).unwrap();
            let preds = fitted
                .predict(&ds, &closure.predict_rows, &mut stats)
                .unwrap();
            if lit.to_bits() != (-0.0_f64).to_bits() {
                assert_eq!(preds[0].to_bits(), lit.to_bits());
            }
        }
    }

    // Verify quantized bins (`quantile_bins`) coexist alongside exact f64 edges in NumericModelV2.
    let a = 0.10000000000000002_f64;
    let b = f64::from_bits(a.to_bits() + 1);
    let q_rows = vec![vec![a], vec![a], vec![b], vec![b], vec![1.0]];
    let q_y = [-100.0, -50.0, 0.0, 50.0, 100.0];
    let q_model = NumericModelV2::fit(
        NumericProfileV2::new(vec!["c0".into()]),
        &GenomeIr::parse("(pipe X (quantile_bins bins=4) (ridge))").unwrap(),
        &q_rows,
        &q_y,
    )
    .unwrap();
    let TransformV2::Quantile { edges, bins, .. } = &q_model.transforms[0] else {
        panic!("expected Quantile transform");
    };
    assert_eq!(*bins, 4);
    assert!(edges[0].windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn w02_wide_columns_permutation_invariant_and_exact_f64() {
    schema_vs_layout_permutation_invariance();
    columns_up_to_511_lossless();
    exact_f64_literal_preserved();
}

//! MR18 (G05 / P05 / GP-05 / WP06 / W03): Region-based lazy scalar conditional (`LazyIf`)
//! and partitioned batch routing (`RouteMask` / `Partition` / `Merge`) with poison-branch
//! non-execution proof.

use feat_math::combinators::route::{
    partition_route_lazy, route_rows, LazyIf, Merge, Partition, RouteMask,
};
use feat_math::combinators::union::{GraphError, Matrix};
use std::sync::atomic::{AtomicUsize, Ordering};

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn lazy_if_poison_branch_never_evaluated() {
    let true_calls = AtomicUsize::new(0);
    let false_calls = AtomicUsize::new(0);
    let poison_calls = AtomicUsize::new(0);

    // 1. Scalar LazyIf: when predicate == 1.0, false branch (poison trap) has 0 invocations.
    let val_true = LazyIf::eval_scalar(
        1.0,
        || {
            true_calls.fetch_add(1, Ordering::SeqCst);
            Ok(42.5)
        },
        || {
            poison_calls.fetch_add(1, Ordering::SeqCst);
            Err(GraphError::Evaluation(
                "poison false branch executed".into(),
            ))
        },
    )
    .unwrap();
    assert_eq!(val_true.to_bits(), 42.5_f64.to_bits());
    assert_eq!(true_calls.load(Ordering::SeqCst), 1);
    assert_eq!(poison_calls.load(Ordering::SeqCst), 0);

    // 2. Scalar LazyIf: when predicate == 0.0, true branch (poison trap) has 0 invocations.
    let val_false = LazyIf::eval_scalar(
        0.0,
        || {
            poison_calls.fetch_add(1, Ordering::SeqCst);
            Err(GraphError::Evaluation("poison true branch executed".into()))
        },
        || {
            false_calls.fetch_add(1, Ordering::SeqCst);
            Ok(-17.25)
        },
    )
    .unwrap();
    assert_eq!(val_false.to_bits(), (-17.25_f64).to_bits());
    assert_eq!(false_calls.load(Ordering::SeqCst), 1);
    assert_eq!(poison_calls.load(Ordering::SeqCst), 0);

    // 3. Row-wise LazyIf with per-row poison traps on unselected branch rows.
    let predicates = [1.0, 0.0, 1.0, 1.0, 0.0];
    let x = [10.0, 20.0, 30.0, 40.0, 50.0];
    let row_true_calls = AtomicUsize::new(0);
    let row_false_calls = AtomicUsize::new(0);
    let out = LazyIf::eval_rows(
        &predicates,
        |r| {
            if predicates[r] != 1.0 {
                poison_calls.fetch_add(1, Ordering::SeqCst);
                return Err(GraphError::Evaluation("poison row in true branch".into()));
            }
            row_true_calls.fetch_add(1, Ordering::SeqCst);
            Ok(x[r] * 2.0)
        },
        |r| {
            if predicates[r] != 0.0 {
                poison_calls.fetch_add(1, Ordering::SeqCst);
                return Err(GraphError::Evaluation("poison row in false branch".into()));
            }
            row_false_calls.fetch_add(1, Ordering::SeqCst);
            Ok(x[r] + 5.0)
        },
    )
    .unwrap();
    assert_eq!(bits(&out), bits(&[20.0, 25.0, 60.0, 80.0, 55.0]));
    assert_eq!(row_true_calls.load(Ordering::SeqCst), 3);
    assert_eq!(row_false_calls.load(Ordering::SeqCst), 2);
    assert_eq!(poison_calls.load(Ordering::SeqCst), 0);

    // 4. Invalid predicates (0.5, NaN) fail closed without invoking either branch.
    for bad_pred in [0.5, -1.0, f64::NAN, f64::INFINITY] {
        let res = LazyIf::eval_scalar(
            bad_pred,
            || {
                poison_calls.fetch_add(1, Ordering::SeqCst);
                Ok(1.0)
            },
            || {
                poison_calls.fetch_add(1, Ordering::SeqCst);
                Ok(0.0)
            },
        );
        assert!(res.is_err());
    }
    assert_eq!(poison_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn partitioned_batch_routing_exact_merge() {
    // 1. Mixed mask: RouteMask -> Partition::split -> branch evaluation -> Merge::reconstruct
    //    matches eager route_rows bit-for-bit while evaluating each branch only on its partition.
    let predicate = Matrix {
        rows: 6,
        width: 1,
        values: vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    };
    let input = Matrix {
        rows: 6,
        width: 2,
        values: vec![
            1.0, 2.0, // row 0 (true)
            3.0, 4.0, // row 1 (false)
            5.0, 6.0, // row 2 (true)
            7.0, 8.0, // row 3 (false)
            9.0, 10.0, // row 4 (false)
            11.0, 12.0, // row 5 (true)
        ],
    };
    let mask = RouteMask::from_predicate(&predicate).unwrap();
    assert_eq!(mask.true_rows, vec![0, 2, 5]);
    assert_eq!(mask.false_rows, vec![1, 3, 4]);

    let (true_part, false_part) = Partition::split(&input, &mask).unwrap();
    assert_eq!(true_part.rows, 3);
    assert_eq!(false_part.rows, 3);
    assert_eq!(true_part.values, vec![1.0, 2.0, 5.0, 6.0, 11.0, 12.0]);
    assert_eq!(false_part.values, vec![3.0, 4.0, 7.0, 8.0, 9.0, 10.0]);

    let true_rows_evaluated = AtomicUsize::new(0);
    let false_rows_evaluated = AtomicUsize::new(0);
    let merged = partition_route_lazy(
        &input,
        &mask,
        2,
        |sub| {
            true_rows_evaluated.fetch_add(sub.rows, Ordering::SeqCst);
            Ok(Matrix {
                rows: sub.rows,
                width: sub.width,
                values: sub.values.iter().map(|v| v * 10.0).collect(),
            })
        },
        |sub| {
            false_rows_evaluated.fetch_add(sub.rows, Ordering::SeqCst);
            Ok(Matrix {
                rows: sub.rows,
                width: sub.width,
                values: sub.values.iter().map(|v| -v).collect(),
            })
        },
    )
    .unwrap();
    assert_eq!(true_rows_evaluated.load(Ordering::SeqCst), 3);
    assert_eq!(false_rows_evaluated.load(Ordering::SeqCst), 3);

    // Compare against eager route_rows.
    let eager_true = Matrix {
        rows: 6,
        width: 2,
        values: input.values.iter().map(|v| v * 10.0).collect(),
    };
    let eager_false = Matrix {
        rows: 6,
        width: 2,
        values: input.values.iter().map(|v| -v).collect(),
    };
    let eager_out = route_rows(&predicate, &eager_true, &eager_false).unwrap();
    assert_eq!(bits(&merged.values), bits(&eager_out.values));

    // Direct Merge::reconstruct check.
    let direct_merge = Merge::reconstruct(
        &mask,
        &Matrix {
            rows: 3,
            width: 2,
            values: true_part.values.iter().map(|v| v * 10.0).collect(),
        },
        &Matrix {
            rows: 3,
            width: 2,
            values: false_part.values.iter().map(|v| -v).collect(),
        },
    )
    .unwrap();
    assert_eq!(bits(&direct_merge.values), bits(&merged.values));

    // 2. All-true mask: false branch has 0 rows and is NEVER invoked even with a poison trap.
    let all_true_pred = Matrix {
        rows: 4,
        width: 1,
        values: vec![1.0; 4],
    };
    let all_true_mask = RouteMask::from_predicate(&all_true_pred).unwrap();
    let poison_invocations = AtomicUsize::new(0);
    let all_true_out = partition_route_lazy(
        &Matrix {
            rows: 4,
            width: 2,
            values: (0..8).map(|i| i as f64).collect(),
        },
        &all_true_mask,
        2,
        |sub| Ok(sub.clone()),
        |_sub| {
            poison_invocations.fetch_add(1, Ordering::SeqCst);
            Err(GraphError::Evaluation(
                "unselected false partition executed".into(),
            ))
        },
    )
    .unwrap();
    assert_eq!(poison_invocations.load(Ordering::SeqCst), 0);
    assert_eq!(all_true_out.rows, 4);
}

#[test]
fn w03_unselected_branch_zero_invocations_on_poison_trap() {
    lazy_if_poison_branch_never_evaluated();
    partitioned_batch_routing_exact_merge();
}

//! Step-6: order-invariance guards. These lock in this session's rank-leak fix — the order-sensitive
//! primitives must operate on VALUES, never on row position, so a target-ordered dataset cannot leak.
use feat_core::data::Matrix;
use feat_core::dsl::{Expr, Un};
use feat_core::hashing::mix64;
use feat_core::metrics::{auc, rank01};

fn perm(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (mix64(i as u64 ^ seed) as usize) % (i + 1);
        idx.swap(i, j);
    }
    idx
}

#[test]
fn auc_is_permutation_invariant() {
    let y: Vec<f32> = (0..200).map(|i| (i % 2) as f32).collect();
    let s: Vec<f32> = (0..200).map(|i| (mix64(i as u64) % 1000) as f32).collect();
    let p = perm(200, 7);
    let yp: Vec<f32> = p.iter().map(|&i| y[i]).collect();
    let sp: Vec<f32> = p.iter().map(|&i| s[i]).collect();
    assert!(
        (auc(&y, &s) - auc(&yp, &sp)).abs() < 1e-9,
        "AUC is a set statistic"
    );
}

#[test]
fn rank01_is_equivariant() {
    let c: Vec<f32> = (0..100).map(|i| (mix64(i as u64) % 50) as f32).collect(); // ties present
    let r = rank01(&c);
    let p = perm(100, 11);
    let cp: Vec<f32> = p.iter().map(|&i| c[i]).collect();
    let rp = rank01(&cp);
    for k in 0..100 {
        assert!(
            (rp[k] - r[p[k]]).abs() < 1e-6,
            "rank01 must permute with its input"
        );
    }
}

#[test]
fn dsl_rank_has_no_row_position_leak() {
    // A near-constant column (the RaceDocumented failure mode): rank() must give tied values ONE shared
    // rank, so it cannot encode row index — which, on a target-sorted file, was the original leak.
    let n = 200;
    let mut col = vec![1.0f32; n];
    col[50] = 2.0;
    col[150] = 0.0;
    let x = Matrix::new(col.clone(), n, 1);
    let r = Expr::Un(Un::Rank, Box::new(Expr::Col(0))).eval(&x);
    let tied: Vec<f32> = (0..n).filter(|&i| col[i] == 1.0).map(|i| r[i]).collect();
    let first = tied[0];
    assert!(
        tied.iter().all(|&v| (v - first).abs() < 1e-6),
        "tied values must share one averaged rank (no position leak)"
    );
    // Equivariance under a row permutation ⇒ no absolute-position dependence.
    let p = perm(n, 5);
    let colp: Vec<f32> = p.iter().map(|&i| col[i]).collect();
    let rp = Expr::Un(Un::Rank, Box::new(Expr::Col(0))).eval(&Matrix::new(colp, n, 1));
    for k in 0..n {
        assert!(
            (rp[k] - r[p[k]]).abs() < 1e-6,
            "DSL rank is permutation-equivariant"
        );
    }
}

#[test]
fn adversarial_auc_self_is_half() {
    // Train vs a holdout drawn from the SAME distribution ⇒ ~0.5 (no shift). Use the Mock backend.
    use feat_core::backend::{MockBackend, TrainSpec};
    let n = 300;
    let mk = |seed: u64, rows: usize| {
        let p = 4;
        let mut d = vec![0f32; rows * p];
        for j in 0..p {
            for i in 0..rows {
                d[j * rows + i] = (mix64((i * p + j) as u64 ^ seed) % 1000) as f32 / 1000.0;
            }
        }
        Matrix::new(d, rows, p)
    };
    let a = feat_core::advval::adversarial_auc(
        &mk(1, n),
        &mk(2, 100),
        &MockBackend,
        &TrainSpec::from_basic_fields(10, 3, 0.1, 1, 0),
        9,
    );
    assert!(
        a < 0.62,
        "same-distribution adversarial AUC should be near 0.5, got {a}"
    );
}

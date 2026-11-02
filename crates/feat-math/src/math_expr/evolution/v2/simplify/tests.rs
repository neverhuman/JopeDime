use super::*;
use crate::Pcg32;

fn grid(lo: f64, hi: f64, points: usize, f: impl Fn(f64) -> f64) -> Vec<(f64, f64)> {
    (0..points)
        .map(|i| {
            let v = lo + (hi - lo) * i as f64 / (points - 1) as f64;
            (v, f(v))
        })
        .collect()
}

fn close(actual: f64, expected: f64, relative: f64) -> bool {
    (actual - expected).abs() <= relative * expected.abs()
}
fn native(expr: &Expr, v: f64) -> (f64, FaultSet) {
    MathExpr::from_ast(expr.clone())
        .unwrap()
        .eval_row_traced(&[v])
}

#[path = "tests/certificate.rs"]
mod certificate;
#[path = "tests/linearization.rs"]
mod linearization;
#[path = "tests/minimax.rs"]
mod minimax;

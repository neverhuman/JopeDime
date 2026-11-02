use super::*;
fn near(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-10, "{a} != {b}");
}
fn error_expr() -> Expression {
    Expression::Sub {
        left: Box::new(Expression::Y {}),
        right: Box::new(Expression::Pred {}),
    }
}
fn value(f: &Fitness, y: &[f64], p: &[f64]) -> f64 {
    f.evaluate(y, p).unwrap().value
}
mod custom_expressions;
mod financial_catalog;
mod metric_semantics;

use feat_math::kernel::profile::{profile, SearchProfile};
use feat_math::kernel::search::admit_child;
use feat_math::math_expr::{Expr, ExpressionScope, MathExpr};

pub(super) fn config() -> SearchProfile {
    profile("kernel2-baseline", 2).unwrap()
}

pub(super) fn scope(width: usize, elementwise: bool) -> ExpressionScope {
    ExpressionScope::search_profile(width.min(8), elementwise)
}

pub(super) fn size(node: &Expr) -> usize {
    1 + match node {
        Expr::Unary(_, a) => size(a),
        Expr::Binary(_, a, b) => size(a) + size(b),
        Expr::Call(_, args) => args.iter().map(size).sum(),
        _ => 0,
    }
}

pub(super) fn admitted(expr: &MathExpr, p: &SearchProfile, width: usize, elementwise: bool) {
    let text = expr.to_source().unwrap();
    let reparsed = MathExpr::parse(&text).unwrap();
    scope(width, elementwise).validate(&reparsed).unwrap();
    let carrier = if elementwise { "map" } else { "derive" };
    admit_child(
        &format!("(pipe X ({carrier} expr=\"{text}\") (ridge))"),
        p,
        width,
    )
    .unwrap();
}

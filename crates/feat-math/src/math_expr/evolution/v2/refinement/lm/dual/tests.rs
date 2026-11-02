use super::*;
use crate::Pcg32;

fn sites_of(source: &str) -> (MathExpr, Vec<Vec<usize>>) {
    let expression = MathExpr::parse(source).unwrap();
    let sites = literal_paths(&expression.root);
    (expression, sites)
}

fn theta_of(expression: &MathExpr, sites: &[Vec<usize>]) -> Vec<f64> {
    sites
        .iter()
        .map(|site| literal_value(at(&expression.root, site)).unwrap())
        .collect()
}

fn native(expression: &MathExpr, row: &[f64]) -> (f64, FaultSet) {
    expression.compile().unwrap().eval_row_traced(row)
}

/// Seeded random row expressions over every function, operator and variable.
fn random_expr(rng: &mut Pcg32, depth: usize, columns: usize) -> Expr {
    let leaf = |rng: &mut Pcg32| -> Expr {
        match rng.next_u32() % 4 {
            0 => Expr::Var(Var::Col(rng.next_u32() as usize % columns)),
            1 => Expr::Var(
                [
                    Var::RowMean,
                    Var::RowSum,
                    Var::RowL1,
                    Var::RowL2,
                    Var::RowMin,
                    Var::RowMax,
                    Var::RowStd,
                ][rng.next_u32() as usize % 7],
            ),
            2 => Expr::Const(
                [0.0, -0.0, 0.5, 1.0, 2.0, 3.0, 8.0, 30.0, 1e-12, 1e12]
                    [rng.next_u32() as usize % 10],
            ),
            _ => Expr::Const(rng.range_f64(-3.0, 3.0)),
        }
    };
    if depth == 0 || rng.next_u32().is_multiple_of(4) {
        return leaf(rng);
    }
    let funcs = [
        Func::Abs,
        Func::Sqrt,
        Func::Log1p,
        Func::Sin,
        Func::Cos,
        Func::Tanh,
        Func::Sigmoid,
        Func::Min,
        Func::Max,
        Func::Clip,
        Func::Step,
        Func::Exp,
        Func::Log,
        Func::Asinh,
        Func::Atan,
        Func::Erf,
        Func::Sign,
        Func::Floor,
        Func::Round,
    ];
    match rng.next_u32() % 3 {
        0 => {
            let op = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow]
                [rng.next_u32() as usize % 5];
            Expr::Binary(
                op,
                Box::new(random_expr(rng, depth - 1, columns)),
                Box::new(random_expr(rng, depth - 1, columns)),
            )
        }
        1 => Expr::Unary(UnOp::Neg, Box::new(random_expr(rng, depth - 1, columns))),
        _ => {
            let func = funcs[rng.next_u32() as usize % funcs.len()];
            let arity = (1..=3).find(|n| validate_arity(func, *n).is_ok()).unwrap();
            Expr::Call(
                func,
                (0..arity)
                    .map(|_| random_expr(rng, depth - 1, columns))
                    .collect(),
            )
        }
    }
}

fn edge_rows() -> Vec<Vec<f64>> {
    let specials = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        2.0,
        -2.5,
        30.0,
        -30.0,
        8.0,
        1e-300,
        1e300,
        f64::MIN_POSITIVE,
        5e-324,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let mut rows = Vec::new();
    for (i, a) in specials.iter().enumerate() {
        rows.push(vec![*a, specials[(i + 3) % specials.len()], 1.5]);
    }
    rows.push(vec![0.25, -1.75, 3.0]);
    rows.push(vec![]);
    rows
}

mod numerical;
mod program;

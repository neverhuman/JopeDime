//! Additive expression operators: no revision-one implementation or RNG calls enter here.
use super::*;
use crate::kernel::capabilities::{Tier, BINOPS, FUNCTIONS, VARIABLES};
use crate::kernel::profile::{ExprPool, SearchProfile};

mod grow;
mod mutation;
mod refinement;
// B19 Remez compression: test-only, no caller until its move is wired (Lane D kickoff).
#[cfg(test)]
mod simplify;
// B20 e-graph simplification: opt-in through `simplify_v2` (egraph.rs).
mod egraph;

#[derive(Clone, Copy)]
enum Operator {
    Function(Func, usize),
    Binary(BinOp),
}

impl Operator {
    fn arity(self) -> usize {
        match self {
            Self::Function(_, n) => n,
            Self::Binary(_) => 2,
        }
    }
    fn build(self, mut children: Vec<Expr>) -> Expr {
        match self {
            Self::Function(f, _) => Expr::Call(f, children),
            Self::Binary(op) => {
                let b = children.pop().unwrap();
                let a = children.pop().unwrap();
                Expr::Binary(op, Box::new(a), Box::new(b))
            }
        }
    }
}

struct Search<'a> {
    pool: &'a ExprPool,
    scope: ExpressionScope,
    tier: Tier,
    operators: Vec<Operator>,
    variables: Vec<Var>,
}

impl<'a> Search<'a> {
    fn new(profile: &'a SearchProfile, mut scope: ExpressionScope) -> Result<Self> {
        let tier = match (profile.id, profile.revision) {
            ("kernel2-baseline", 2 | 3) => Tier::Baseline,
            ("kernel2-experimental", 2 | 3) => Tier::Experimental,
            _ => bail!("version-two kernel2 expression profile required"),
        };
        let pool = profile.expr.as_ref().context("expression pool required")?;
        if pool.weight_each == 0
            || pool.max_nodes == 0
            || pool.max_nodes > 15
            || pool.max_depth == 0
            || pool.max_depth > 4
            || scope.max_nodes == 0
            || scope.max_depth == 0
            || pool.birth_constants.iter().any(|v| !v.is_finite())
        {
            bail!("invalid expression pool or search limits");
        }
        scope.columns = scope.columns.min(pool.column_upper_exclusive).min(8);
        scope.max_nodes = scope.max_nodes.min(pool.max_nodes);
        scope.max_depth = scope.max_depth.min(pool.max_depth);
        let mut operators = Vec::new();
        for name in &pool.functions {
            let (func, _, _) = FUNCTIONS
                .iter()
                .find(|(_, n, c)| n == name && c.class.admits(tier))
                .context("function outside manifest search pool")?;
            let arity = (1..=3).find(|n| validate_arity(*func, *n).is_ok()).unwrap();
            operators.push(Operator::Function(*func, arity));
        }
        for name in &pool.binops {
            let (op, _, _) = BINOPS
                .iter()
                .find(|(_, n, c)| n == name && c.class.admits(tier))
                .context("binary operator outside manifest search pool")?;
            operators.push(Operator::Binary(*op));
        }
        let mut variables = Vec::new();
        for name in &pool.variables {
            let (var, _, _) = VARIABLES
                .iter()
                .find(|(_, n, c)| n == name && c.class.admits(tier))
                .context("variable outside manifest search pool")?;
            if *var != Var::X || scope.elementwise {
                variables.push(*var);
            }
        }
        variables.extend((0..scope.columns).map(Var::Col));
        if variables.is_empty() {
            bail!("expression scope has no variable leaf");
        }
        Ok(Self {
            pool,
            scope,
            tier,
            operators,
            variables,
        })
    }

    /// Check the printed/reparsed representation too: negative literals consume two nodes.
    fn finish(&self, root: Expr) -> Result<MathExpr> {
        let source = MathExpr::from_ast(normalize(root))?.to_source()?;
        let expression = MathExpr::parse(&source)?;
        self.scope.validate(&expression)?;
        crate::kernel::search::check_expression(
            &expression,
            self.pool,
            self.scope.elementwise,
            self.tier,
        )?;
        Ok(expression)
    }

    fn variable(&self, rng: &mut Pcg32) -> Expr {
        Expr::Var(self.variables[rng.next_u32() as usize % self.variables.len()])
    }
}

fn nodes(expr: &Expr) -> usize {
    1 + match expr {
        Expr::Unary(_, a) => nodes(a),
        Expr::Binary(_, a, b) => nodes(a) + nodes(b),
        Expr::Call(_, args) => args.iter().map(nodes).sum(),
        _ => 0,
    }
}

fn literal(value: f64) -> Expr {
    if value.is_sign_negative() {
        Expr::Unary(UnOp::Neg, Box::new(Expr::Const(-value)))
    } else {
        Expr::Const(value)
    }
}

fn literal_value(node: &Expr) -> Option<f64> {
    match node {
        Expr::Const(v) => Some(*v),
        Expr::Unary(UnOp::Neg, a) => literal_value(a).map(|v| -v),
        _ => None,
    }
}

fn normalize(node: Expr) -> Expr {
    if let Some(v) = literal_value(&node) {
        return literal(v);
    }
    match node {
        Expr::Unary(op, a) => Expr::Unary(op, Box::new(normalize(*a))),
        Expr::Binary(op, a, b) => {
            Expr::Binary(op, Box::new(normalize(*a)), Box::new(normalize(*b)))
        }
        Expr::Call(f, args) => Expr::Call(f, args.into_iter().map(normalize).collect()),
        other => other,
    }
}

fn literal_paths(root: &Expr) -> Vec<Vec<usize>> {
    paths(root)
        .into_iter()
        .filter(|path| {
            literal_value(at(root, path)).is_some()
                && (path.is_empty() || literal_value(at(root, &path[..path.len() - 1])).is_none())
        })
        .collect()
}

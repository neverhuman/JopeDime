use super::*;
use crate::Pcg32;

/// Separate search contract; existing mutation RNG streams never call this.
#[derive(Clone, Copy, Debug)]
pub struct ExpressionScope {
    pub columns: usize,
    pub elementwise: bool,
    pub max_nodes: usize,
    pub max_depth: usize,
}

#[derive(Clone, Debug)]
pub struct ExpressionEdit {
    pub expression: MathExpr,
    pub changed: bool,
}

impl ExpressionScope {
    pub fn validate(&self, expression: &MathExpr) -> Result<()> {
        if self.max_nodes == 0
            || self.max_nodes > MAX_NODES
            || self.max_depth == 0
            || self.max_depth > MAX_DEPTH
        {
            bail!("invalid expression search limits");
        }
        expression.check_structure(self.max_nodes, self.max_depth)?;
        expression.validate_columns(self.columns, self.elementwise)
    }
}

impl MathExpr {
    /// Type-preserving local edit over scalar F64 blocks. Candidates are
    /// constructed at an in-scope site and checked against the complete tree.
    pub fn mutate_v1(&self, seed: u64, scope: ExpressionScope) -> Result<ExpressionEdit> {
        scope.validate(self)?;
        let mut rng = Pcg32::new(seed, 0x657870727631);
        let paths = paths(&self.root);
        let path = &paths[rng.next_u32() as usize % paths.len()];
        let node = at(&self.root, path);
        let leaf = |rng: &mut Pcg32| -> Expr {
            let choices = 2 + usize::from(scope.elementwise) + usize::from(scope.columns > 0);
            match rng.next_u32() as usize % choices {
                0 => Expr::Const(rng.range_f64(-2.0, 2.0)),
                1 => Expr::Var(Var::RowMean),
                2 if scope.elementwise => Expr::Var(Var::X),
                _ => Expr::Var(Var::Col(rng.next_u32() as usize % scope.columns)),
            }
        };
        let mut choices = vec![leaf(&mut rng)];
        match node {
            Expr::Const(v) if v.is_finite() => {
                let changed = *v + rng.range_f64(-1.0, 1.0) * v.abs().max(1.0) * 0.25;
                if changed.is_finite() {
                    choices.push(Expr::Const(changed));
                }
            }
            Expr::Unary(_, a) => choices.push((**a).clone()),
            Expr::Binary(_, a, b) => {
                choices.extend([(**a).clone(), (**b).clone()]);
                for op in [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow] {
                    choices.push(Expr::Binary(op, a.clone(), b.clone()));
                }
            }
            Expr::Call(_, args) => choices.extend(args.iter().cloned()),
            _ => {}
        }
        for func in [
            Func::Sin,
            Func::Cos,
            Func::Tanh,
            Func::Abs,
            Func::Sqrt,
            Func::Sigmoid,
            Func::Log1p,
            Func::Exp,
            Func::Asinh,
            Func::Atan,
            Func::Erf,
            Func::Sign,
            Func::Floor,
            Func::Round,
            Func::Log,
        ] {
            choices.push(Expr::Call(func, vec![node.clone()]));
        }
        for op in [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow] {
            choices.push(Expr::Binary(
                op,
                Box::new(node.clone()),
                Box::new(leaf(&mut rng)),
            ));
        }
        for func in [Func::Min, Func::Max, Func::Step] {
            choices.push(Expr::Call(func, vec![node.clone(), leaf(&mut rng)]));
        }
        choices.push(Expr::Call(
            Func::Clip,
            vec![node.clone(), leaf(&mut rng), leaf(&mut rng)],
        ));
        let candidates = choices
            .into_iter()
            .filter_map(|replacement| {
                let mut root = self.root.clone();
                replace(&mut root, path, replacement);
                let candidate = MathExpr::from_ast(root).ok()?;
                scope.validate(&candidate).ok()?;
                Some(candidate)
            })
            .collect::<Vec<_>>();
        choose(self, candidates, &mut rng)
    }

    pub fn crossover_v1(
        &self,
        donor: &MathExpr,
        seed: u64,
        scope: ExpressionScope,
    ) -> Result<ExpressionEdit> {
        scope.validate(self)?;
        // Donor blocks may have a wider surrounding scope; only closed,
        // compatible subtrees enter the recipient.
        donor.check_structure(MAX_NODES, MAX_DEPTH)?;
        let mut rng = Pcg32::new(seed, 0x63726f73737631);
        let recipient = paths(&self.root);
        let path = &recipient[rng.next_u32() as usize % recipient.len()];
        let candidates = paths(&donor.root)
            .iter()
            .filter_map(|source| {
                let mut root = self.root.clone();
                replace(&mut root, path, at(&donor.root, source).clone());
                let candidate = MathExpr::from_ast(root).ok()?;
                scope.validate(&candidate).ok()?;
                Some(candidate)
            })
            .collect();
        choose(self, candidates, &mut rng)
    }

    /// Deterministic coordinate fitting on TRAINING rows only. The returned
    /// count is actual objective evaluations, including the initial baseline.
    /// Numerical errors reject a trial instead of fabricating zero predictions.
    pub fn optimize_constants_v1(
        &self,
        rows: &[Vec<f64>],
        targets: &[f64],
        max_evaluations: usize,
    ) -> Result<(Self, usize)> {
        self.check_structure(MAX_NODES, MAX_DEPTH)?;
        if rows.is_empty() || rows.len() != targets.len() || max_evaluations == 0 {
            bail!("constant fitting needs aligned nonempty training data and a budget");
        }
        let loss = |expr: &MathExpr| -> Result<f64> {
            let compiled = expr.compile()?;
            let mut sum = 0.0;
            for (row, target) in rows.iter().zip(targets) {
                let residual = compiled.eval_row(row)? - target;
                sum += residual * residual / rows.len() as f64;
            }
            if !sum.is_finite() {
                bail!("nonfinite constant-fitting objective");
            }
            Ok(sum)
        };
        let mut best = self.clone();
        let mut score = loss(&best)?;
        let mut evaluated = 1;
        let constants: Vec<_> = paths(&self.root)
            .into_iter()
            .filter(|path| matches!(at(&self.root, path), Expr::Const(_)))
            .collect();
        if constants.is_empty() {
            return Ok((best, evaluated));
        }
        let mut step = 1.0;
        while evaluated < max_evaluations {
            let mut improved = false;
            for path in &constants {
                let Expr::Const(value) = at(&best.root, path) else {
                    unreachable!()
                };
                let value = *value;
                for delta in [-step, step] {
                    if evaluated == max_evaluations {
                        return Ok((best, evaluated));
                    }
                    let mut candidate = best.clone();
                    replace(&mut candidate.root, path, Expr::Const(value + delta));
                    evaluated += 1;
                    if let Ok(trial) = loss(&candidate) {
                        if trial < score {
                            best = candidate;
                            score = trial;
                            improved = true;
                        }
                    }
                }
            }
            if !improved {
                step *= 0.5;
            }
        }
        Ok((best, evaluated))
    }
}

fn choose(parent: &MathExpr, candidates: Vec<MathExpr>, rng: &mut Pcg32) -> Result<ExpressionEdit> {
    let mut before = crate::kernel::codec::Writer::default();
    parent.write_exact(&mut before)?;
    let before = before.finish();
    let mut changed = Vec::new();
    for candidate in candidates {
        let mut after = crate::kernel::codec::Writer::default();
        candidate.write_exact(&mut after)?;
        if before != after.finish() {
            changed.push(candidate);
        }
    }
    if changed.is_empty() {
        return Ok(ExpressionEdit {
            expression: parent.clone(),
            changed: false,
        });
    }
    Ok(ExpressionEdit {
        expression: changed[rng.next_u32() as usize % changed.len()].clone(),
        changed: true,
    })
}

fn paths(root: &Expr) -> Vec<Vec<usize>> {
    fn visit(node: &Expr, path: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        out.push(path.clone());
        let children: Vec<&Expr> = match node {
            Expr::Unary(_, a) => vec![a],
            Expr::Binary(_, a, b) => vec![a, b],
            Expr::Call(_, args) => args.iter().collect(),
            _ => vec![],
        };
        for (i, child) in children.into_iter().enumerate() {
            path.push(i);
            visit(child, path, out);
            path.pop();
        }
    }
    let mut result = Vec::new();
    visit(root, &mut Vec::new(), &mut result);
    result
}
fn at<'a>(root: &'a Expr, path: &[usize]) -> &'a Expr {
    if path.is_empty() {
        return root;
    }
    let child = match root {
        Expr::Unary(_, a) => a.as_ref(),
        Expr::Binary(_, a, b) => {
            if path[0] == 0 {
                a
            } else {
                b
            }
        }
        Expr::Call(_, args) => &args[path[0]],
        _ => unreachable!(),
    };
    at(child, &path[1..])
}
fn replace(root: &mut Expr, path: &[usize], replacement: Expr) {
    if path.is_empty() {
        *root = replacement;
        return;
    }
    let child = match root {
        Expr::Unary(_, a) => a.as_mut(),
        Expr::Binary(_, a, b) => {
            if path[0] == 0 {
                a
            } else {
                b
            }
        }
        Expr::Call(_, args) => &mut args[path[0]],
        _ => unreachable!(),
    };
    replace(child, &path[1..], replacement);
}

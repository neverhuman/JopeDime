use super::*;
use crate::kernel::expression_search::{ExpressionArm, ExpressionMutation};
use std::collections::BTreeSet;

fn draw_arm(profile: &SearchProfile, rng: &mut Pcg32) -> Result<ExpressionArm> {
    let weights = profile
        .expression_edits
        .as_ref()
        .context("expression arm weights required")?;
    let mut seen = BTreeSet::new();
    let mut total = 0u64;
    for row in weights {
        if !seen.insert(row.arm) {
            bail!("duplicate expression arm");
        }
        total += u64::from(row.weight);
    }
    if total == 0 {
        bail!("expression arm weights have no mass");
    }
    let mut draw = ((u64::from(rng.next_u32()) << 32) | u64::from(rng.next_u32())) % total;
    for row in weights {
        if draw < u64::from(row.weight) {
            return Ok(row.arm);
        }
        draw -= u64::from(row.weight);
    }
    unreachable!()
}

impl MathExpr {
    /// Draw the arm first. An edit that cannot fit returns the legal parent with
    /// that same selected arm, so rejection cannot distort arm probabilities.
    pub fn mutate_v2(
        &self,
        profile: &SearchProfile,
        scope: ExpressionScope,
        seed: u64,
    ) -> Result<ExpressionMutation> {
        let search = Search::new(profile, scope)?;
        let parent = search.finish(self.root.clone())?;
        let mut rng = Pcg32::new(seed, 0x6d7574617632);
        let arm = draw_arm(profile, &mut rng)?;
        let sites = if arm == ExpressionArm::ConstantJitter {
            literal_paths(&parent.root)
        } else {
            paths(&parent.root)
        };
        let mut child = parent.clone();
        if !sites.is_empty() {
            let path = &sites[rng.next_u32() as usize % sites.len()];
            let node = at(&parent.root, path);
            let budget = search.scope.max_nodes - (nodes(&parent.root) - nodes(node));
            let depth = search.scope.max_depth - path.len();
            if let Some(replacement) = search.edit(node, arm, budget, depth, &mut rng) {
                let mut root = parent.root.clone();
                replace(&mut root, path, replacement);
                if let Ok(candidate) = search.finish(root) {
                    child = candidate;
                }
            }
        }
        Ok(ExpressionMutation {
            changed: child.to_source()? != parent.to_source()?,
            expression: child,
            arm,
        })
    }

    /// Only recipient-scope-compatible F64 subtrees satisfying the size-fair
    /// bound enter the recipient; the surrounding donor may have a wider scope.
    pub fn crossover_v2(
        &self,
        donor: &MathExpr,
        profile: &SearchProfile,
        scope: ExpressionScope,
        seed: u64,
    ) -> Result<ExpressionEdit> {
        let search = Search::new(profile, scope)?;
        let parent = search.finish(self.root.clone())?;
        donor.check_structure(MAX_NODES, MAX_DEPTH)?;
        let mut rng = Pcg32::new(seed, 0x63726f73737632);
        let sites = paths(&parent.root);
        let path = &sites[rng.next_u32() as usize % sites.len()];
        let fair = 1 + 2 * nodes(at(&parent.root, path));
        let candidates: Vec<_> = paths(&donor.root)
            .iter()
            .filter_map(|source| {
                let replacement = normalize(at(&donor.root, source).clone());
                if nodes(&replacement) > fair {
                    return None;
                }
                let mut root = parent.root.clone();
                replace(&mut root, path, replacement);
                search.finish(root).ok()
            })
            .collect();
        let child = if candidates.is_empty() {
            parent.clone()
        } else {
            candidates[rng.next_u32() as usize % candidates.len()].clone()
        };
        Ok(ExpressionEdit {
            changed: child.to_source()? != parent.to_source()?,
            expression: child,
        })
    }
}

impl Search<'_> {
    fn point(&self, node: &Expr, budget: usize, depth: usize, rng: &mut Pcg32) -> Option<Expr> {
        match node {
            Expr::Var(_) => Some(self.variable(rng)),
            Expr::Const(_) | Expr::Unary(_, _) => {
                let constants: Vec<_> = self
                    .pool
                    .birth_constants
                    .iter()
                    .copied()
                    .filter(|v| !v.is_sign_negative() || (budget >= 2 && depth >= 2))
                    .collect();
                (!constants.is_empty())
                    .then(|| literal(constants[rng.next_u32() as usize % constants.len()]))
            }
            Expr::Binary(_, a, b) => {
                let ops: Vec<_> = self
                    .operators
                    .iter()
                    .filter_map(|o| match o {
                        Operator::Binary(op) => Some(*op),
                        _ => None,
                    })
                    .collect();
                (!ops.is_empty()).then(|| {
                    Expr::Binary(
                        ops[rng.next_u32() as usize % ops.len()],
                        a.clone(),
                        b.clone(),
                    )
                })
            }
            Expr::Call(_, args) => {
                let ops: Vec<_> = self
                    .operators
                    .iter()
                    .filter_map(|o| match o {
                        Operator::Function(f, n) if *n == args.len() => Some(*f),
                        _ => None,
                    })
                    .collect();
                (!ops.is_empty())
                    .then(|| Expr::Call(ops[rng.next_u32() as usize % ops.len()], args.clone()))
            }
        }
    }

    fn edit(
        &self,
        node: &Expr,
        arm: ExpressionArm,
        budget: usize,
        depth: usize,
        rng: &mut Pcg32,
    ) -> Option<Expr> {
        match arm {
            ExpressionArm::Point => self.point(node, budget, depth, rng),
            ExpressionArm::Subtree => {
                Some(self.grow(budget, depth, rng.next_u32() & 1 == 0, true, rng))
            }
            ExpressionArm::Hoist => {
                let sites = paths(node);
                (sites.len() > 1).then(|| {
                    at(
                        node,
                        &sites[1 + rng.next_u32() as usize % (sites.len() - 1)],
                    )
                    .clone()
                })
            }
            ExpressionArm::Shrink => Some(self.variable(rng)),
            ExpressionArm::ConstantJitter => {
                let value = literal_value(node)?;
                let trial = value + rng.gaussian() * value.abs().max(1.0) * 0.25;
                trial.is_finite().then(|| literal(trial))
            }
            ExpressionArm::Wrap => {
                let ops: Vec<_> = self
                    .operators
                    .iter()
                    .copied()
                    .filter(|o| matches!(o, Operator::Function(_, 1)))
                    .collect();
                (!ops.is_empty() && depth > 1 && nodes(node) < budget)
                    .then(|| ops[rng.next_u32() as usize % ops.len()].build(vec![node.clone()]))
            }
            ExpressionArm::InsertBinary => {
                let ops: Vec<_> = self
                    .operators
                    .iter()
                    .copied()
                    .filter(|o| matches!(o, Operator::Binary(_)))
                    .collect();
                if ops.is_empty() || depth <= 1 || budget <= nodes(node) + 1 {
                    return None;
                }
                let right = self.grow(budget - nodes(node) - 1, depth - 1, false, true, rng);
                Some(ops[rng.next_u32() as usize % ops.len()].build(vec![node.clone(), right]))
            }
        }
    }
}

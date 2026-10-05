//! Typed scalar edits reuse existing EEC draws and private edit implementations.
use super::*;
use crate::kernel::expression_search::ExpressionArm;
use crate::math_expr::Expr;

#[derive(Clone, Debug)]
pub struct ScalarEdit {
    pub candidate: ScalarCandidate,
    pub arm: ExpressionArm,
    pub changed: bool,
    pub opt_in_drawn: bool,
    pub opt_in_admitted: bool,
    pub proposal_rejected: bool,
}

fn children(node: &ScalarNode) -> Vec<&ScalarNode> {
    match node {
        ScalarNode::Unary(_, a) => vec![a],
        ScalarNode::Binary(_, a, b) => vec![a, b],
        ScalarNode::Call(_, args) => args.iter().collect(),
        _ => vec![],
    }
}

fn size(node: &ScalarNode) -> usize {
    1 + children(node).iter().map(|node| size(node)).sum::<usize>()
}

fn paths(node: &ScalarNode) -> Vec<Vec<usize>> {
    fn visit(node: &ScalarNode, path: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        out.push(path.clone());
        for (index, child) in children(node).iter().enumerate() {
            path.push(index);
            visit(child, path, out);
            path.pop();
        }
    }
    let mut out = vec![];
    visit(node, &mut vec![], &mut out);
    out
}

fn at<'a>(mut node: &'a ScalarNode, path: &[usize]) -> &'a ScalarNode {
    for &index in path {
        node = match node {
            ScalarNode::Unary(_, child) if index == 0 => child,
            ScalarNode::Binary(_, a, _) if index == 0 => a,
            ScalarNode::Binary(_, _, b) if index == 1 => b,
            ScalarNode::Call(_, args) => &args[index],
            _ => unreachable!("path drawn from the same typed projection"),
        };
    }
    node
}

fn replace(root: &mut ScalarNode, path: &[usize], replacement: ScalarNode) {
    if let Some((&index, rest)) = path.split_first() {
        let child = match root {
            ScalarNode::Unary(_, child) if index == 0 => child.as_mut(),
            ScalarNode::Binary(_, a, _) if index == 0 => a.as_mut(),
            ScalarNode::Binary(_, _, b) if index == 1 => b.as_mut(),
            ScalarNode::Call(_, args) => &mut args[index],
            _ => unreachable!("path drawn from the same typed projection"),
        };
        replace(child, rest, replacement);
    } else {
        *root = replacement;
    }
}

fn from_old(node: Expr) -> Result<ScalarNode, String> {
    let expression = MathExpr::from_ast(node).map_err(|e| e.to_string())?;
    ScalarExpressionProgram::from_v1(&expression)
        .map(|program| program.ast().clone())
        .map_err(|e| e.to_string())
}

impl ScalarCandidate {
    pub fn mutate(&self, seed: u64) -> Result<ScalarEdit, String> {
        self.profile.mutate(self.program(), seed)
    }
}

impl ScalarProducerProfile {
    fn mutate(&self, parent: &ScalarExpressionProgram, seed: u64) -> Result<ScalarEdit, String> {
        self.admit(parent)?;
        let projection = self.projection(parent)?;
        let (arm, site, mut old_rng) = projection
            .scalar_mutation_site_v2(&self.base, self.scope(), seed)
            .map_err(|e| e.to_string())?;
        let mut root = parent.ast().clone();
        let mut policy = parent.numerical_policy();
        let mut opt_rng = Pcg32::new(seed, OPT_IN_EDIT_STREAM);
        let drawn = opt_rng.next_u32().is_multiple_of(4);
        let mut opt_admitted = false;
        let mut rejected = false;
        if let Some(path) = site {
            let node = at(parent.ast(), &path);
            let mut scope = self.scope();
            scope.max_nodes = self.declaration.max_nodes - (size(parent.ast()) - size(node));
            scope.max_depth = self.declaration.max_depth - path.len();
            let mut proposal = None;
            let mut opt_proposed = false;
            // A fixed catalog preserves the exact probabilities of unrelated
            // choices when a component is disabled. No failed choice redraws.
            if drawn && matches!(arm, ExpressionArm::Wrap | ExpressionArm::Point) {
                let mut choices: Vec<_> = self
                    .declaration
                    .function_catalog
                    .iter()
                    .copied()
                    .map(Some)
                    .collect();
                if !self.declaration.policy_birth_choices.is_empty() {
                    choices.push(None);
                }
                if !choices.is_empty() {
                    match choices[opt_rng.next_u32() as usize % choices.len()] {
                        Some(id) if self.function_enabled(id) => {
                            opt_proposed = true;
                            let child = if arm == ExpressionArm::Point {
                                match node {
                                    ScalarNode::Call(_, args) if args.len() == 1 => {
                                        Some(args[0].clone())
                                    }
                                    _ => None,
                                }
                            } else {
                                Some(node.clone())
                            };
                            if let Some(child) = child {
                                proposal = Some(ScalarNode::Call(
                                    ScalarFunction::from_versioned_id(2, id)
                                        .map_err(|e| e.to_string())?,
                                    vec![child],
                                ));
                                if id == 13 {
                                    policy = NumericalPolicy::StableV2;
                                }
                            }
                        }
                        None if self.declaration.enable_numerical_policy => {
                            opt_proposed = true;
                            // Policy toggle is a complete-program edit; variable
                            // point edits select only a declared stable reduction.
                            if arm == ExpressionArm::Point && matches!(node, ScalarNode::Var(_)) {
                                let choices = [Var::RowSum, Var::RowMean, Var::RowL1, Var::RowL2];
                                proposal = Some(ScalarNode::Var(
                                    choices[opt_rng.next_u32() as usize % choices.len()],
                                ));
                                policy = NumericalPolicy::StableV2;
                            } else {
                                proposal = Some(node.clone());
                                policy = if policy == NumericalPolicy::LegacyV1 {
                                    NumericalPolicy::StableV2
                                } else {
                                    NumericalPolicy::LegacyV1
                                };
                            }
                        }
                        _ => {}
                    }
                }
            }
            if !opt_proposed {
                proposal = match arm {
                    ExpressionArm::Hoist => {
                        let sites = paths(node);
                        (sites.len() > 1).then(|| {
                            at(
                                node,
                                &sites[1 + old_rng.next_u32() as usize % (sites.len() - 1)],
                            )
                            .clone()
                        })
                    }
                    ExpressionArm::Wrap | ExpressionArm::InsertBinary => {
                        // Existing selection and subtree growth see the actual
                        // remaining budget after subtracting this opaque subtree.
                        scope.max_nodes -= size(node) - 1;
                        let proxy = MathExpr::from_ast(Expr::Var(Var::RowMean))
                            .map_err(|e| e.to_string())?;
                        proxy
                            .edit_scalar_proxy_v2(&self.base, scope, arm, &mut old_rng)
                            .map_err(|e| e.to_string())?
                            .map(|edited| -> Result<ScalarNode, String> {
                                let mut graft = from_old(edited)?;
                                match &mut graft {
                                    ScalarNode::Call(_, args) if args.len() == 1 => {
                                        args[0] = node.clone()
                                    }
                                    ScalarNode::Binary(_, left, _) => **left = node.clone(),
                                    _ => return Err("unexpected existing wrapper shape".into()),
                                }
                                Ok(graft)
                            })
                            .transpose()?
                    }
                    ExpressionArm::Point => {
                        let proxy = self.projection(
                            &ScalarExpressionProgram::new(node.clone())
                                .map_err(|e| e.to_string())?
                                .with_numerical_policy(parent.numerical_policy()),
                        )?;
                        proxy
                            .edit_scalar_proxy_v2(&self.base, scope, arm, &mut old_rng)
                            .map_err(|e| e.to_string())?
                            .map(|edited| -> Result<ScalarNode, String> {
                                let mut graft = from_old(edited)?;
                                match (&mut graft, node) {
                                    (ScalarNode::Call(_, args), ScalarNode::Call(_, old_args)) => {
                                        *args = old_args.clone()
                                    }
                                    (
                                        ScalarNode::Binary(_, a, b),
                                        ScalarNode::Binary(_, old_a, old_b),
                                    ) => {
                                        *a = old_a.clone();
                                        *b = old_b.clone();
                                    }
                                    _ => {}
                                }
                                Ok(graft)
                            })
                            .transpose()?
                    }
                    _ => {
                        let proxy = self.projection(
                            &ScalarExpressionProgram::new(node.clone())
                                .map_err(|e| e.to_string())?
                                .with_numerical_policy(parent.numerical_policy()),
                        )?;
                        proxy
                            .edit_scalar_proxy_v2(&self.base, scope, arm, &mut old_rng)
                            .map_err(|e| e.to_string())?
                            .map(from_old)
                            .transpose()?
                    }
                };
            }
            if let Some(proposal) = proposal {
                replace(&mut root, &path, proposal);
                let candidate = ScalarExpressionProgram::new(root.clone())
                    .map_err(|e| e.to_string())?
                    .with_numerical_policy(policy);
                if self.admit(&candidate).is_ok() {
                    opt_admitted = opt_proposed;
                } else {
                    rejected = true;
                    root = parent.ast().clone();
                    policy = parent.numerical_policy();
                }
            } else {
                // An enabled typed Point choice without a unary child is a
                // refused proposal, not an unobserved successful edit.
                rejected = opt_proposed;
                root = parent.ast().clone();
                policy = parent.numerical_policy();
            }
        }
        let program = ScalarExpressionProgram::new(root)
            .map_err(|e| e.to_string())?
            .with_numerical_policy(policy);
        self.admit(&program)?;
        let changed = program.to_bytes().map_err(|e| e.to_string())?
            != parent.to_bytes().map_err(|e| e.to_string())?;
        Ok(ScalarEdit {
            candidate: ScalarCandidate {
                program,
                profile: self.clone(),
                seed,
                opt_in_drawn: drawn,
                opt_in_admitted: opt_admitted,
                proposal_rejected: rejected,
            },
            arm,
            changed,
            opt_in_drawn: drawn,
            opt_in_admitted: opt_admitted,
            proposal_rejected: rejected,
        })
    }
}

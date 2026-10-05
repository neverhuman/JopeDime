use super::*;
use crate::kernel::numerical_policy::NumericalPolicy;
use crate::kernel::scalar_program::{ScalarExpressionProgram, ScalarFunction, ScalarNode};
use crate::math_expr::{Expr, ExpressionScope, Func, MathExpr, Var};
impl ScalarProducerProfile {
    pub(in crate::kernel::scalar_program) fn validate_program(
        &self,
        program: &ScalarExpressionProgram,
        carrier: ScalarCarrier,
        width: usize,
    ) -> Result<(), String> {
        if carrier != self.carrier() || width != self.input_width() {
            return Err("scalar producer/model scope mismatch".into());
        }
        self.admit(program)
    }

    pub(super) fn scope(&self) -> ExpressionScope {
        ExpressionScope::search_profile(
            self.declaration.input_width.min(8),
            self.declaration.carrier == ScalarCarrier::Elementwise,
        )
    }

    pub(super) fn projection(&self, program: &ScalarExpressionProgram) -> Result<MathExpr, String> {
        fn project(
            node: &ScalarNode,
            program: &ScalarExpressionProgram,
            declaration: &ProducerDeclaration,
        ) -> Result<Expr, String> {
            Ok(match node {
                ScalarNode::Const(value) => Expr::Const(*value),
                ScalarNode::Var(var @ (Var::RowSum | Var::RowL1)) => {
                    let kind = if *var == Var::RowSum { "sum" } else { "l1" };
                    if program.numerical_policy() != NumericalPolicy::StableV2
                        || !declaration
                            .reduction_catalog
                            .iter()
                            .any(|name| name == kind)
                    {
                        return Err("reduction outside scalar producer".into());
                    }
                    Expr::Var(Var::RowMean)
                }
                ScalarNode::Var(var) => Expr::Var(*var),
                ScalarNode::Unary(op, child) => {
                    Expr::Unary(*op, Box::new(project(child, program, declaration)?))
                }
                ScalarNode::Binary(op, a, b) => Expr::Binary(
                    *op,
                    Box::new(project(a, program, declaration)?),
                    Box::new(project(b, program, declaration)?),
                ),
                ScalarNode::Call(function, args) => {
                    let function = match function {
                        ScalarFunction::V1(Func::Asinh)
                            if program.numerical_policy() == NumericalPolicy::StableV2 =>
                        {
                            if !declaration.function_catalog.contains(&13) {
                                return Err("asinh outside scalar producer".into());
                            }
                            Func::Abs
                        }
                        ScalarFunction::V1(function) => *function,
                        function => {
                            if !declaration.function_catalog.contains(&function.id()) {
                                return Err("function outside scalar producer".into());
                            }
                            Func::Abs
                        }
                    };
                    Expr::Call(
                        function,
                        args.iter()
                            .map(|node| project(node, program, declaration))
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                }
            })
        }
        MathExpr::from_ast(project(program.ast(), program, &self.declaration)?)
            .map_err(|e| e.to_string())
    }

    pub(in crate::kernel::scalar_program) fn admit(
        &self,
        program: &ScalarExpressionProgram,
    ) -> Result<(), String> {
        if program.numerical_policy() == NumericalPolicy::StableV2
            && !matches!(
                self.declaration.kind,
                ScalarProducerKind::NumericalPolicy | ScalarProducerKind::All
            )
        {
            return Err("policy outside scalar producer".into());
        }
        fn visit(node: &ScalarNode, scope: ExpressionScope) -> Result<(usize, usize), String> {
            let children = match node {
                // Existing growth/edit normalizes every negative literal to
                // Neg(Const(abs)). Reject aliases that would change paths when
                // the existing mutation seam normalizes its projection.
                ScalarNode::Const(value) if !value.is_finite() || value.is_sign_negative() => {
                    return Err("scalar producer requires canonical finite literals".into())
                }
                ScalarNode::Var(Var::X) if !scope.elementwise => return Err("scalar row x".into()),
                ScalarNode::Var(Var::Col(i)) if *i >= scope.columns => {
                    return Err("scalar profile column".into())
                }
                ScalarNode::Unary(_, child) => vec![child.as_ref()],
                ScalarNode::Binary(_, a, b) => vec![a.as_ref(), b.as_ref()],
                ScalarNode::Call(_, args) => args.iter().collect(),
                _ => vec![],
            };
            let mut nodes = 1;
            let mut depth = 1;
            for child in children {
                let (n, d) = visit(child, scope)?;
                nodes += n;
                depth = depth.max(d + 1);
            }
            Ok((nodes, depth))
        }
        let (nodes, depth) = visit(program.ast(), self.scope())?;
        if nodes > 15 || depth > 4 {
            return Err("scalar search caps".into());
        }
        // Reuse strict existing pipeline/profile admission for every retained
        // operator, constant, scope, real leaf and printed/reparsed size cap.
        // Typed unary projections preserve their actual children and node count.
        let projection = self
            .projection(program)?
            .to_source()
            .map_err(|e| e.to_string())?;
        let stage = if self.declaration.carrier == ScalarCarrier::Row {
            "derive"
        } else {
            "map"
        };
        let pipe = format!("(pipe X ({stage} expr={projection:?}) (ridge lambda=0.02))");
        crate::kernel::search::admit_child(&pipe, &self.base, self.declaration.input_width)
            .map_err(|e| e.to_string())?;
        let bytes = program.to_bytes().map_err(|e| e.to_string())?;
        if ScalarExpressionProgram::from_bytes(&bytes)
            .map_err(|e| e.to_string())?
            .to_bytes()
            .map_err(|e| e.to_string())?
            != bytes
        {
            return Err("scalar byte fixed point".into());
        }
        let versioned = crate::kernel::wire_v2::VersionedProgram::ScalarV2(program.clone());
        let text = versioned.to_lossless_text().map_err(|e| e.to_string())?;
        if crate::kernel::wire_v2::VersionedProgram::parse_lossless_text(&text)
            .map_err(|e| e.to_string())?
            .to_bytes()
            .map_err(|e| e.to_string())?
            != bytes
        {
            return Err("scalar text fixed point".into());
        }
        Ok(())
    }
}

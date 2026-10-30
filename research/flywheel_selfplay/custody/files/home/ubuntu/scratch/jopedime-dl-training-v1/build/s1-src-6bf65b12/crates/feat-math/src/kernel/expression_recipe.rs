//! Opt-in expression training recipes using the historical numeric policy.
//!
//! This profile changes backend admission, not operator identities or arithmetic.
//! The generated C ABI fits a numeric table; it does not embed fitted state or
//! accept ordered-state metadata. Non-finite features are sanitized to zero.
use super::{Error, Result};
use crate::transpile::{lint_shipped_text, validate_math_kernel, EmitOptions};
use crate::GenomeIr;

pub const PROFILE: &str = "feat-math.cpp17-expression-recipe.v2";

#[derive(Clone, Debug)]
pub struct Cpp17Recipe {
    pub profile: &'static str,
    pub source: String,
}

/// Emit symbolic raw/affine or map/derive stages with Ridge/ElasticNet.
/// Other stages retain the standalone C++17 recipe's existing admission.
/// Target transforms, nested portfolios and portable math remain refused.
pub fn emit_cpp17(genome: &GenomeIr, opts: &EmitOptions) -> Result<Cpp17Recipe> {
    validate_math_kernel(opts).map_err(|e| Error::Unsupported(e.to_string()))?;
    if let GenomeIr::Symbolic { source, expr, .. } = genome {
        // The historical planner reparses source; refuse an IR whose tree disagrees.
        let tree = expr
            .to_source()
            .map_err(|e| Error::Unsupported(e.to_string()))?;
        let parsed =
            crate::math_expr::MathExpr::parse(source).map_err(|e| Error::Syntax(e.to_string()))?;
        let text = parsed
            .to_source()
            .map_err(|e| Error::Syntax(e.to_string()))?;
        if tree != text {
            return Err(Error::Syntax("symbolic source and tree disagree".into()));
        }
    }
    let plan = crate::spec::build_plan(genome)
        .map_err(|missing| Error::Unsupported(missing.join("; ")))?;
    let body = opts
        .cpp_expression_recipe(&plan)
        .map_err(|missing| Error::Unsupported(missing.join(", ")))?;
    let source = format!("// {PROFILE}\n{body}");
    let hits = lint_shipped_text(&source);
    if !hits.is_empty() {
        return Err(Error::Unsupported(format!(
            "artifact vocabulary: {}",
            hits.join(", ")
        )));
    }
    Ok(Cpp17Recipe {
        profile: PROFILE,
        source,
    })
}

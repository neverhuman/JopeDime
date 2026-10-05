//! Explicit library training recipes. The inherited raw CSV main is not this
//! entry point. Clip trees wait for an exported typed hard-error trace.
use super::{validate, Error, LinearScaledWinner, MathExpr, Result};
use crate::math_expr::{Expr, Func};
use crate::transpile::{emit, lint_shipped_text, EmitOptions};

pub const RECIPE_PROFILE: &str = "feat-math.scalar-linear-scaling-recipe.v2";

#[derive(Clone, Debug)]
pub struct TrainingRecipeArtifacts {
    pub profile: &'static str,
    /// Explicit API: scaled_train(training_rows, training_targets), followed by
    /// the returned model's predict/to_bytes/from_bytes. No inference labels.
    pub rust: String,
    pub python: String,
    pub cpp17: String,
}

pub fn emit_training_recipe(
    expression: &MathExpr,
    input_width: usize,
) -> Result<TrainingRecipeArtifacts> {
    let template = LinearScaledWinner {
        expression: expression.clone(),
        input_width,
        slope: 0.0,
        intercept: 0.0,
    };
    validate(&template)?;
    if contains_clip(expression.ast()) {
        return Err(Error::Unsupported(
            "linear-scaling recipe needs a typed clip error trace".into(),
        ));
    }
    let super::Program::LegacyV1(genome) = template.raw_program()?;
    let opts = EmitOptions::default();
    let raw = emit(&genome, &opts).map_err(|e| Error::Unsupported(e.to_string()))?;
    let raw_rust = super::emit::rust_scalar_library(&raw.rust)?;
    let raw_python = super::emit::python_scalar_library(&raw.python);
    let cpp = super::super::expression_recipe::emit_cpp17(&genome, &opts)?;
    let raw_cpp = super::emit::cpp_scalar_library(&cpp.source);
    let bytes = template
        .to_bytes()
        .map_err(|e| Error::Unsupported(e.to_string()))?;
    let words = bytes
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let render = |source: &str| {
        source
            .replace("@WIDTH@", &input_width.to_string())
            .replace("@BYTES@", &words)
    };
    let rust = format!(
        "{}\nmod calibration {{\n{}\n}}\n{}",
        raw_rust,
        include_str!("calibration.rs"),
        render(include_str!("recipe_rust.txt"))
    );
    let python = format!(
        "{}\n{}",
        raw_python,
        render(include_str!("recipe_python.txt"))
    );
    let cpp17 = format!("{}\n{}", raw_cpp, render(include_str!("recipe_cpp17.txt")));
    for source in [&rust, &python, &cpp17] {
        let hits = lint_shipped_text(source);
        if !hits.is_empty() {
            return Err(Error::Unsupported(format!(
                "artifact vocabulary: {}",
                hits.join(", ")
            )));
        }
    }
    Ok(TrainingRecipeArtifacts {
        profile: RECIPE_PROFILE,
        rust,
        python,
        cpp17,
    })
}

fn contains_clip(node: &Expr) -> bool {
    match node {
        Expr::Call(Func::Clip, _) => true,
        Expr::Call(_, args) => args.iter().any(contains_clip),
        Expr::Unary(_, child) => contains_clip(child),
        Expr::Binary(_, left, right) => contains_clip(left) || contains_clip(right),
        _ => false,
    }
}

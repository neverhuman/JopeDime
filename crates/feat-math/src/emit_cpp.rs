//! C++ emitter: a single std-only `.cpp` file that compiles with
//! `g++ -O2 -std=c++17 -ffp-contract=off -fno-fast-math model.cpp` and mirrors
//! the Rust and Python numeric behavior on the certified operator set.
//!
//! This is the deterministic CPU rung beneath the optional CUDA build: the same
//! source, compiled without fused-multiply-add contraction so every `a*b + c`
//! stays two separately rounded IEEE-754 operations, exactly as the reference
//! runtime evaluates it. Transcendental ops (`cos`/`ln`/`exp`) reach the same
//! glibc libm as the Rust artifact, applied to arguments built in the identical
//! fixed order, so the three languages agree to the last bit. Coverage grows one
//! certified op at a time; unsupported ops are reported so a broad
//! half-implemented surface never ships silently.

use crate::mathexpr::fmt_f64;
use crate::spec::{helper_set, Helper, Plan, PlannedHead, PlannedOp, PlannedStage};
// B05–B07: explicit scalar-v2 recipe assembly.
use crate::kernel::scalar_program::emit::{self as scalar_emit, ScalarLanguage};
use crate::transpile::{EmitOptions, MathKernel};

/// Emit a standalone C++ program for `plan`, or list the elements this rung does
/// not yet support (mirrors `supports()` semantics for the Rust/Python paths).
pub fn emit_cpp(plan: &Plan, opts: &EmitOptions) -> Result<String, Vec<String>> {
    emit_cpp_inner(plan, opts, false)
}

impl EmitOptions {
    pub(crate) fn cpp_expression_recipe(&self, plan: &Plan) -> Result<String, Vec<String>> {
        emit_cpp_inner(plan, self, true)
    }
}

fn emit_cpp_inner(
    plan: &Plan,
    opts: &EmitOptions,
    expressions: bool,
) -> Result<String, Vec<String>> {
    let (stages, head) = match plan {
        Plan::Pipeline {
            target: None,
            stages,
            head,
        } => (stages, head),
        Plan::Pipeline {
            target: Some(_), ..
        } => return Err(vec!["target transforms".to_string()]),
        Plan::Symbolic { expr, affine, .. } if expressions => {
            let model = model::symbolic_cpp(expr, *affine)?;
            // Symbolic recipes have no ridge/stage use of max_num. Mark only
            // this opt-in copy; historical/default emitted bytes stay frozen.
            let preamble = PREAMBLE.replacen(
                "static double max_num(",
                "[[maybe_unused]] static double max_num(",
                1,
            );
            return Ok([
                preamble.as_str(),
                crate::FittedCppArtifact::expression_helpers(),
                H_SANITIZE,
                EXPRESSION_SHAPE,
                &model,
                IO_BLOCK,
            ]
            .join("\n\n")
                + "\n");
        }
        Plan::Symbolic { .. } => return Err(vec!["symbolic formula programs".to_string()]),
    };

    let mut unsupported: Vec<String> = Vec::new();
    for stage in stages {
        if op_cpp(&stage.op).is_none()
            && !(expressions
                && matches!(
                    stage.op,
                    PlannedOp::MapExpr { .. }
                        | PlannedOp::DeriveExpr { .. }
                        | PlannedOp::ScalarV2 { .. }
                ))
        {
            unsupported.push(format!("transform {}", op_label(&stage.op)));
        }
    }
    if !head_supported(head) {
        unsupported.push(format!("head {}", head_label(head)));
    }
    if !unsupported.is_empty() {
        return Err(unsupported);
    }

    if scalar_emit::raw_input(plan) && opts.math_kernel == MathKernel::Portable {
        return Err(vec![
            "scalar-v2 retained operations require PlatformLibm".into()
        ]);
    }
    let mut sections: Vec<String> = Vec::new();
    sections.push(PREAMBLE.to_string());
    if expressions {
        sections.push(crate::FittedCppArtifact::expression_helpers().to_string());
        sections.push(EXPRESSION_SHAPE.to_string());
    }
    if opts.math_kernel == MathKernel::Portable {
        sections.push(PORTABLE_WARNING_BEGIN.to_string());
        sections.push(PORTABLE_MATH.to_string());
        sections.push(PORTABLE_WARNING_END.to_string());
    }
    for helper in helper_set(plan) {
        // A standalone scalar stage preserves raw inputs and replaces its
        // features through its own epilogue; neither generic helper is called.
        if stages.len() == 1
            && scalar_emit::raw_input(plan)
            && matches!(helper, Helper::Sanitize | Helper::FinishStage)
        {
            continue;
        }
        if expressions && matches!(helper, Helper::MathSigmoid) {
            // The shared expression helper block supplies this same scalar.
            continue;
        }
        let body = helper_cpp(helper, opts.partition_point_quantile_bounds);
        if !body.is_empty() {
            sections.push(body.to_string());
        }
    }
    let scalar_programs = scalar_emit::programs(plan);
    if !scalar_programs.is_empty() {
        sections.push(
            scalar_emit::emit_v2_helpers(&scalar_programs, ScalarLanguage::Cpp17)
                .map_err(|e| vec![e.to_string()])?,
        );
    }
    let mut seen: Vec<&'static str> = Vec::new();
    for (index, stage) in stages.iter().enumerate() {
        if expressions {
            match &stage.op {
                PlannedOp::ScalarV2 {
                    program,
                    carrier,
                    width,
                } => {
                    sections.push(
                        scalar_emit::emit_stage(
                            program,
                            *carrier,
                            *width,
                            index + 1,
                            ScalarLanguage::Cpp17,
                        )
                        .map_err(|e| vec![e.to_string()])?,
                    );
                }
                PlannedOp::MapExpr { expr, .. } | PlannedOp::DeriveExpr { expr, .. } => {
                    sections.push(model::expression_stage_cpp(
                        index + 1,
                        expr,
                        matches!(stage.op, PlannedOp::MapExpr { .. }),
                    )?);
                }
                _ => {}
            }
        }
        if let Some((name, src)) = op_cpp(&stage.op) {
            if !seen.contains(&name) {
                seen.push(name);
                sections.push(src.to_string());
            }
        }
    }
    sections.push(head_src_cpp(head).to_string());
    sections.push(model::model_cpp(stages, head, expressions));
    sections.push(IO_BLOCK.to_string());
    let mut source = sections.join("\n\n") + "\n";
    if opts.math_kernel == MathKernel::Portable {
        for (platform, portable) in [
            ("std::exp(", "xexp("),
            ("std::log(", "xln("),
            ("std::sin(", "xsin("),
            ("std::cos(", "xcos("),
        ] {
            source = source.replace(platform, portable);
        }
    }
    Ok(source)
}

#[path = "emit_cpp/helpers.rs"]
mod helpers;
#[path = "emit_cpp/model.rs"]
mod model;
#[path = "emit_cpp/templates_advanced_ops.rs"]
mod templates_advanced_ops;
#[path = "emit_cpp/templates_base.rs"]
mod templates_base;
#[path = "emit_cpp/templates_basic_ops.rs"]
mod templates_basic_ops;
#[path = "emit_cpp/templates_heads.rs"]
mod templates_heads;
#[path = "emit_cpp/templates_io.rs"]
mod templates_io;
#[path = "emit_cpp/templates_portable.rs"]
mod templates_portable;

use helpers::*;
use templates_advanced_ops::*;
use templates_base::*;
use templates_basic_ops::*;
use templates_heads::*;
use templates_io::*;
#[cfg(test)]
pub(crate) use templates_portable::portable_math_matches_verified_fixture;
use templates_portable::*;

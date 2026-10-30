//! Public, sanitized capability report for a program's available routes.

use super::capabilities::{Manifest, Support, BINOPS, FUNCTIONS, UNOPS, VARIABLES};
use crate::dsl::{DslExpr, ParamValue};
use crate::math_expr::{Expr, MathExpr, Var};
use crate::{transpile, GenomeIr};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Route {
    Ok,
    Unsupported(Vec<String>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FittedBasis {
    Declared,
    Verified,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportReport {
    pub rust_recipe: Route,
    pub python_recipe: Route,
    pub cpp17_recipe: Route,
    pub cpp17_fitted: Route,
    pub cpp17_fitted_basis: FittedBasis,
    pub javascript: Route,
}

fn clean_kind(raw: &str) -> String {
    let cut = raw.find(['(', '{']).unwrap_or(raw.len());
    let kind = raw[..cut].trim_end();
    if kind.is_empty() || kind.contains('"') || !transpile::lint_shipped_text(kind).is_empty() {
        "unknown op".to_owned()
    } else {
        kind.to_owned()
    }
}

fn unique_kinds(raw: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut out = Vec::new();
    for reason in raw {
        let kind = clean_kind(&reason);
        if !out.contains(&kind) {
            out.push(kind);
        }
    }
    out
}

fn planning_kind(manifest: &Manifest, problem: &str) -> String {
    if problem.starts_with("portfolio:") || problem.starts_with("portfolio ") {
        return "portfolio".to_owned();
    }
    let kind = problem.split_once(':').map_or(problem, |(kind, _)| kind);
    if manifest.op(kind).is_some()
        || matches!(
            kind,
            "portfolio" | "target" | "gpr kernel" | "symbolic expression"
        )
    {
        kind.to_owned()
    } else {
        "unknown op".to_owned()
    }
}

fn route(reasons: std::result::Result<(), Vec<String>>) -> Route {
    match reasons {
        Ok(()) => Route::Ok,
        Err(reasons) => Route::Unsupported(unique_kinds(reasons)),
    }
}

fn require_expression(manifest: &Manifest, kind: &str, name: &str, unsupported: &mut Vec<String>) {
    let available = manifest.expression.iter().any(|row| {
        row.kind == kind
            && row.name == name
            && row.capability.emitters.cpp17_fitted == Support::Supported
    });
    if !available {
        unsupported.push(format!("{kind} {name}"));
    }
}

fn visit_expression(manifest: &Manifest, expr: &Expr, unsupported: &mut Vec<String>) {
    match expr {
        Expr::Const(_) => {}
        Expr::Var(value) => {
            let name = match value {
                Var::Col(index) if *index < 8 => "col_under_8",
                Var::Col(_) => "col_8_plus",
                other => VARIABLES
                    .iter()
                    .find(|(sample, _, _)| sample == other)
                    .map(|(_, name, _)| *name)
                    .expect("every variable has a capability row"),
            };
            require_expression(manifest, "variable", name, unsupported);
        }
        Expr::Unary(op, inner) => {
            let name = UNOPS
                .iter()
                .find(|(sample, _, _)| sample == op)
                .map(|(_, name, _)| *name)
                .expect("every unary operation has a capability row");
            require_expression(manifest, "unary", name, unsupported);
            visit_expression(manifest, inner, unsupported);
        }
        Expr::Binary(op, left, right) => {
            let name = BINOPS
                .iter()
                .find(|(sample, _, _)| sample == op)
                .map(|(_, name, _)| *name)
                .expect("every binary operation has a capability row");
            require_expression(manifest, "binary", name, unsupported);
            visit_expression(manifest, left, unsupported);
            visit_expression(manifest, right, unsupported);
        }
        Expr::Call(function, args) => {
            let name = FUNCTIONS
                .iter()
                .find(|(sample, _, _)| sample == function)
                .map(|(_, name, _)| *name)
                .expect("every function has a capability row");
            require_expression(manifest, "function", name, unsupported);
            for arg in args {
                visit_expression(manifest, arg, unsupported);
            }
        }
    }
}

fn visit_pipeline(manifest: &Manifest, pipeline: &DslExpr, unsupported: &mut Vec<String>) {
    for stage in &pipeline.stages {
        match manifest.op(&stage.op) {
            Some(row) if row.emitters.cpp17_fitted == Support::Supported => {}
            Some(row) => unsupported.push(format!(
                "{} {}",
                if matches!(row.kind, super::capabilities::OpKind::Head) {
                    "head"
                } else {
                    "transform"
                },
                row.name
            )),
            None => unsupported.push("unknown op".to_owned()),
        }
        if matches!(stage.op.as_str(), "map" | "derive") {
            if let Some(ParamValue::Str(source)) = stage.params.get("expr") {
                if let Ok(expression) = MathExpr::parse(source) {
                    visit_expression(manifest, expression.ast(), unsupported);
                }
            }
        }
    }
}

fn declared_fitted(manifest: &Manifest, program: &GenomeIr) -> Route {
    let mut unsupported = Vec::new();
    let form = match program {
        GenomeIr::Pipeline(_) => "pipeline",
        GenomeIr::Symbolic { .. } => "symbolic",
        GenomeIr::Portfolio(_) => "nested_portfolio",
    };
    if !manifest
        .form(form)
        .is_some_and(|row| row.emitters.cpp17_fitted == Support::Supported)
    {
        unsupported.push(if form == "nested_portfolio" {
            "portfolio".to_owned()
        } else {
            form.to_owned()
        });
    }
    match program {
        GenomeIr::Pipeline(pipeline) => visit_pipeline(manifest, pipeline, &mut unsupported),
        GenomeIr::Symbolic { expr, .. } => visit_expression(manifest, expr.ast(), &mut unsupported),
        GenomeIr::Portfolio(portfolio) => {
            for member in &portfolio.members {
                visit_pipeline(manifest, member, &mut unsupported);
            }
        }
    }
    let unsupported = unique_kinds(unsupported);
    if unsupported.is_empty() {
        Route::Ok
    } else {
        Route::Unsupported(unsupported)
    }
}

pub fn export_report_with(manifest: &Manifest, program: &GenomeIr) -> ExportReport {
    let planning = transpile::supports(program).err();
    let (rust_recipe, python_recipe, cpp17_recipe, javascript) = if let Some(problems) = planning {
        let kinds = unique_kinds(
            problems
                .iter()
                .map(|problem| planning_kind(manifest, problem)),
        );
        let unsupported = Route::Unsupported(kinds);
        (
            unsupported.clone(),
            unsupported.clone(),
            unsupported.clone(),
            unsupported,
        )
    } else {
        (
            Route::Ok,
            Route::Ok,
            route(transpile::supports_cpp17(program)),
            route(transpile::supports_javascript(program)),
        )
    };
    ExportReport {
        rust_recipe,
        python_recipe,
        cpp17_recipe,
        cpp17_fitted: declared_fitted(manifest, program),
        cpp17_fitted_basis: FittedBasis::Declared,
        javascript,
    }
}

pub fn export_report(program: &GenomeIr) -> ExportReport {
    export_report_with(&Manifest::current(), program)
}

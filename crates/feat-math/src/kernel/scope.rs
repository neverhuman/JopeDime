//! Static scope and width check for programs, before any evaluation (kernel2 WP-1.6).
//!
//! [`validate_scope`] walks a program's stages with the width interval of the pipeline slots'
//! width rules, the rules `PipelineSlots::validate_legacy_widths` applies, and checks every `map`
//! and `derive` expression against the lower bound of the width that reaches its stage: every
//! column it reads must exist at that bound, and it may read `x` only in a `map`. The check reads
//! no data, so it can run before a program is fitted or evaluated.
//!
//! It differs from `validate_legacy_widths` in two ways, both on purpose:
//!
//! - It takes no training-row count. `nystrom` is the only stage whose width depends on the rows
//!   (`min(k, rows)`), and the walk takes its width for one row, the fewest a fit can have, so
//!   the lower bound holds for every row count.
//! - It accepts inputs wider than [`crate::MAX_WIDTH`], up to [`MAX_INPUT_WIDTH`]. The runtime
//!   truncates a row to `MAX_WIDTH` only after a stage, so the first stage reads every input
//!   column.
//!
//! The members of an explicit portfolio do not see the input as it is: the portfolio first drops
//! every column that is constant in training, so a member's rows hold between one column and the
//! input width, and its column indices count the kept columns. Each member is walked from one
//! column, the same lower-bound rule.
//!
//! The search's non-legacy fault policies (`crate::evolve::FaultPolicy`) call it on every
//! candidate before fitting it. The legacy search does not, so no existing search stream changes.

use super::pipeline::{next_width, PipelineSlots};
use super::Error;
use crate::dsl::{DslExpr, ParamValue};
use crate::math_expr::{Expr, FaultClass, MathExpr, Var};
use crate::GenomeIr;
use std::fmt;

/// The widest input [`validate_scope`] accepts: 2^24 columns. Every stage's width arithmetic is
/// exact up to it, and one row that wide already holds 128 MiB of `f64` values.
pub const MAX_INPUT_WIDTH: usize = 1 << 24;

/// The training rows the width walk assumes: one, the fewest a fit can have. Only `nystrom`
/// depends on it, and its width never shrinks as rows are added.
const ROWS: usize = 1;

/// The columns a member of an explicit portfolio sees, at the fewest: one. The portfolio drops
/// every column that is constant in training before its members see the rows (`fit_portfolio`,
/// and the same mask at prediction), and it fits them only when at least one column is left.
const MEMBER_MIN_COLUMNS: usize = 1;

/// Why a program fails [`validate_scope`], and where.
///
/// Its `Display` form is `[member M, ][stage S: ][CLASS: ]reason`, for example
/// `stage 1: COLUMN_OOB: column index 3 out of bounds for width 2`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScopeError {
    /// The failing member of an explicit portfolio program; `None` for every other program.
    pub member: Option<usize>,
    /// The failing stage's index in its pipeline's stages, a target stage included. `None` when
    /// the error belongs to the program as a whole: an input width out of bounds, a stage layout
    /// the pipeline slots refuse (the reason names the stage), or a closed-form expression, which
    /// has no stages.
    pub stage: Option<usize>,
    /// The fault class of a scope fault: [`FaultClass::IllegalScope`] when an expression outside a
    /// `map` reads `x`, otherwise [`FaultClass::ColumnOob`] for a column that may not exist. `None`
    /// when the program cannot be checked: an expression that is not a string or does not parse, a
    /// stage layout the pipeline slots refuse, an input encoding stage, a stage whose width cannot
    /// be computed, or an input width out of bounds.
    pub class: Option<FaultClass>,
    /// The exact reason. For a scope fault it is the message of [`MathExpr::validate_columns`].
    pub reason: String,
}

impl ScopeError {
    fn program(reason: String) -> Self {
        Self {
            member: None,
            stage: None,
            class: None,
            reason,
        }
    }

    fn at_stage(stage: usize, class: Option<FaultClass>, reason: String) -> Self {
        Self {
            member: None,
            stage: Some(stage),
            class,
            reason,
        }
    }
}

impl fmt::Display for ScopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.member, self.stage) {
            (Some(member), Some(stage)) => write!(f, "member {member}, stage {stage}: ")?,
            (Some(member), None) => write!(f, "member {member}: ")?,
            (None, Some(stage)) => write!(f, "stage {stage}: ")?,
            (None, None) => {}
        }
        if let Some(class) = self.class {
            write!(f, "{class}: ")?;
        }
        f.write_str(&self.reason)
    }
}

impl std::error::Error for ScopeError {}

/// Checks, before any evaluation, that every expression of `genome` reads only what exists when
/// its stage runs on inputs of `input_width` columns.
///
/// - A pipeline is walked stage by stage. A `map` or `derive` expression is checked against the
///   lower bound of the width that reaches its stage; the width then moves through the stage by
///   the pipeline slots' width rules.
/// - A closed-form program evaluates its expression per row, so the expression is checked in row
///   context at `input_width`.
/// - Each member of an explicit portfolio is a pipeline and is checked as one; the error names the
///   member. A member sees only the input columns that vary in training, between one and
///   `input_width` of them, and its column indices count those columns, so its walk starts from
///   one column.
///
/// The first failure is returned. The result depends only on the arguments.
pub fn validate_scope(genome: &GenomeIr, input_width: usize) -> Result<(), ScopeError> {
    if input_width == 0 || input_width > MAX_INPUT_WIDTH {
        return Err(ScopeError::program(format!(
            "input width {input_width} is outside 1..={MAX_INPUT_WIDTH}"
        )));
    }
    match genome {
        GenomeIr::Pipeline(pipe) => check_pipeline(pipe, (input_width, input_width)),
        GenomeIr::Symbolic { expr, .. } => {
            check_expression(expr, input_width, false).map_err(|(class, reason)| ScopeError {
                class: Some(class),
                ..ScopeError::program(reason)
            })
        }
        GenomeIr::Portfolio(portfolio) => {
            for (member, pipe) in portfolio.members.iter().enumerate() {
                check_pipeline(pipe, (MEMBER_MIN_COLUMNS, input_width)).map_err(|error| {
                    ScopeError {
                        member: Some(member),
                        ..error
                    }
                })?;
            }
            Ok(())
        }
    }
}

/// One pipeline whose input has between `width.0` and `width.1` columns, in the order and with
/// the reasons of `validate_legacy_widths`.
fn check_pipeline(pipe: &DslExpr, mut width: (usize, usize)) -> Result<(), ScopeError> {
    let slots = PipelineSlots::from_pipeline(pipe).map_err(|error| {
        ScopeError::program(match error {
            // The slots name an op they cannot classify by its bare name.
            Error::Unsupported(op) => format!("unknown stage {op}"),
            other => message(other),
        })
    })?;
    // The slots admit a target stage only first and an input stage only right after it.
    let offset = usize::from(slots.target.is_some());
    if slots.input.is_some() {
        return Err(ScopeError::at_stage(
            offset,
            None,
            "input encoding needs a table schema binding".into(),
        ));
    }
    for (i, stage) in slots.transforms.iter().enumerate() {
        let index = offset + i;
        if matches!(stage.op.as_str(), "map" | "derive") {
            let Some(ParamValue::Str(source)) = stage.params.get("expr") else {
                return Err(ScopeError::at_stage(
                    index,
                    None,
                    format!("{} needs a string expression", stage.op),
                ));
            };
            let expr = MathExpr::parse(source)
                .map_err(|error| ScopeError::at_stage(index, None, error.to_string()))?;
            check_expression(&expr, width.0, stage.op == "map")
                .map_err(|(class, reason)| ScopeError::at_stage(index, Some(class), reason))?;
        }
        width = next_width(stage, ROWS, width)
            .map_err(|error| ScopeError::at_stage(index, None, message(error)))?;
    }
    Ok(())
}

/// One expression at `width` columns, in a `map` (`elementwise`) or in row context.
fn check_expression(
    expr: &MathExpr,
    width: usize,
    elementwise: bool,
) -> Result<(), (FaultClass, String)> {
    expr.validate_columns(width, elementwise).map_err(|error| {
        // `validate_columns` reports a read of `x` outside a map before any column.
        let class = if !elementwise && reads_x(expr.ast()) {
            FaultClass::IllegalScope
        } else {
            FaultClass::ColumnOob
        };
        (class, error.to_string())
    })
}

/// Whether the tree reads `x`. Iterative, so no depth can exhaust the stack.
fn reads_x(root: &Expr) -> bool {
    let mut pending = vec![root];
    while let Some(node) = pending.pop() {
        match node {
            Expr::Var(Var::X) => return true,
            Expr::Const(_) | Expr::Var(_) => {}
            Expr::Unary(_, child) => pending.push(child),
            Expr::Binary(_, left, right) => pending.extend([&**left, &**right]),
            Expr::Call(_, args) => pending.extend(args),
        }
    }
    false
}

/// The message inside a kernel error, without the variant name its `Display` prints.
fn message(error: Error) -> String {
    match error {
        Error::Syntax(text)
        | Error::Domain(text)
        | Error::Bounds(text)
        | Error::Shape(text)
        | Error::Unsupported(text)
        | Error::Resource(text)
        | Error::NonFinite(text) => text,
        Error::IterationLimit => "iteration limit".into(),
        Error::TimeLimit => "time limit".into(),
    }
}

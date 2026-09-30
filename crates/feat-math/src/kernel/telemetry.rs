//! Per-stage fault telemetry beside unchanged values (kernel2 WP-1.4), and the zero-width refusal
//! of the traced entry points (WP-1.8).
//!
//! [`apply_op_traced`], [`fit_traced`] and [`predict_traced`] return what
//! [`crate::ops::apply_op`], [`crate::model::fit`] and [`Model::predict`] return: the same rows bit
//! for bit, a model with the same serialized bytes (apart from the wall-clock fit times that a
//! portfolio records), the same prediction bits. Beside them they count, per stage, the faults
//! each `map` or `derive` expression raised (the classes of [`FaultClass`]) and the non-finite
//! values each stage's epilogue replaced with 0.0.
//!
//! Nothing here changes a value. The expression stages run the traced evaluators, which return
//! the legacy values bit for bit, through the same branches and fallbacks as
//! `apply_op_generated`'s `MapExpr` and `DeriveExpr` arms. Every other op runs
//! `apply_op_generated` itself. The stage epilogue (`finish`) and its append flag are private to
//! `ops.rs`, and `model.rs` is an identity-bearing file, so this module mirrors that code line for
//! line rather than editing it; `crates/feat-math/tests/stage_telemetry.rs` holds each mirror to
//! its original. The definitions of every count are in `docs/internal/kernel2/math-domain-table.md`.
//!
//! One behaviour is deliberately stricter than the legacy entry points: [`fit_traced`] and
//! [`predict_traced`] refuse a zero-width input, a row with no columns ([`ZeroWidthInput`]).

use crate::genome::GenomeIr;
use crate::heads::fit_head;
use crate::math_expr::{FaultClass, FaultSet, FaultSeverity, MathExpr};
use crate::model::{split_stages, split_target, Model};
use crate::ops::{apply_op_generated, fit_op, FittedOp};
use crate::target_transform::fit_target;
use crate::{sanitize, Rows, MAX_WIDTH};
use anyhow::{bail, Result};
use std::fmt;

/// The slots of [`StageFaultCounts::faulted_cells`]: one per bit of a [`FaultSet`], so a class
/// appended later has a slot without changing the field's type.
pub const FAULT_SLOTS: usize = 32;

const _: () = assert!(FaultClass::ALL.len() <= FAULT_SLOTS);

/// What one stage did on one pass over a set of rows: the faults its expression raised, and the
/// non-finite values its epilogue replaced.
///
/// The expression counts (`evaluated_cells`, `faulted_cells`, `hard_rows`, `compile_failed` and
/// `constant_output`) cover the cells a `map` or `derive` expression generated, before the append
/// merge, the width cap and the epilogue. For every other op they are zero or false.
/// `epilogue_sanitized` covers every op.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StageFaultCounts {
    /// Cells the expression generated: every cell of every `map` row (a row with no cells
    /// generates none), and one cell per `derive` row.
    pub evaluated_cells: u64,
    /// Per class, at the class's [`FaultClass::index`]: the generated cells whose faults include
    /// that class. A cell with two classes counts once under each.
    pub faulted_cells: [u64; FAULT_SLOTS],
    /// Generated cells with at least one Domain class, counted once per cell even when multiple
    /// Domain classes occur together.
    #[serde(default)]
    pub domain_cells: u64,
    /// Generated cells with at least one Saturation class, counted once per cell.
    #[serde(default)]
    pub saturation_cells: u64,
    /// Rows with at least one generated cell whose faults include a Hard class. Each such cell is
    /// the fallback 0.0 of an evaluation error (the legacy `unwrap_or(0.0)`) or of a compile
    /// failure.
    pub hard_rows: u64,
    /// `compile()` refused the expression, so the stage generated its fallback zeros: `map`
    /// writes 0.0 for every cell of the row and `derive` one 0.0 per row. Each of those cells
    /// carries `COMPILE_FAILED`.
    pub compile_failed: bool,
    /// The expression generated at least one cell, and every generated cell has the bits of the
    /// first. A single cell is trivially constant; +0.0 and -0.0 differ.
    pub constant_output: bool,
    /// Non-finite values that the stage epilogue (`finish` in `ops.rs`) replaced with 0.0. It
    /// runs after the append merge and the width cap, so a merged input cell counts and a cell
    /// cut by the cap does not; the 0.0 that pads an empty row is not a replacement.
    pub epilogue_sanitized: u64,
}

impl StageFaultCounts {
    /// The generated cells whose faults include `class`.
    pub fn faulted(&self, class: FaultClass) -> u64 {
        self.faulted_cells[class.index()]
    }
}

/// The stage counts of one pass through a program: the training pass of [`fit_traced`], or the
/// inference pass of [`predict_traced`].
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PipelineTelemetry {
    /// Non-finite input cells that a pipeline's entry sanitize replaced with 0.0 before its first
    /// stage. It is 0 for a symbolic program, which runs no such sanitize: its expression reads
    /// the raw row. It is also 0 for an explicit portfolio, whatever its input holds:
    /// `portfolio.rs` replaces the non-finite cells it reads with 0.0 too, but that code is not
    /// traced.
    pub input_sanitized: u64,
    /// A pipeline has one entry per fitted transform, in order: entry `i` is the stage of
    /// `ops[i]` in [`Model::Pipeline`]. The target stage and the head have none. A symbolic
    /// program has one entry, its expression, with one cell per row. An explicit portfolio has
    /// none: its members are fitted and applied inside `portfolio.rs`.
    pub stages: Vec<StageFaultCounts>,
}

/// The refusal of a zero-width input by [`fit_traced`] and [`predict_traced`] (WP-1.8): a row with
/// no columns. The stages never produce one, because the epilogue pads an empty row with 0.0, so
/// it can only come from the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZeroWidthInput {
    /// The index of the first row with no columns.
    pub row: usize,
}

impl ZeroWidthInput {
    /// The reason, which starts the refusal's message.
    pub const REASON: &'static str = "zero-width input";

    /// The refusal of `x`, if one of its rows has no columns. A set with no rows is not
    /// zero-width.
    pub fn find(x: &Rows) -> Option<ZeroWidthInput> {
        x.iter()
            .position(Vec::is_empty)
            .map(|row| ZeroWidthInput { row })
    }
}

impl fmt::Display for ZeroWidthInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: row {} has no columns", Self::REASON, self.row)
    }
}

impl std::error::Error for ZeroWidthInput {}

/// [`crate::ops::apply_op`] with the stage's fault counts. The rows are `apply_op`'s, bit for bit.
///
/// A `map` or `derive` stage takes the same branch as `apply_op_generated`: the compiled form when
/// the expression reads a row summary, the interpreter otherwise, and the fallback zeros when
/// `compile()` refuses the expression. Its cells come from the traced evaluators, whose values
/// equal the legacy ones, fallbacks included. Every other op runs `apply_op_generated` itself and
/// reports only `epilogue_sanitized`. A symbolic program is a model, not an op: [`fit_traced`] and
/// [`predict_traced`] trace it.
pub fn apply_op_traced(op: &FittedOp, x: &Rows) -> (Rows, StageFaultCounts) {
    let mut tally = Tally::new();
    let generated = match op {
        FittedOp::MapExpr { expr, .. } => generate_map(expr, x, &mut tally),
        FittedOp::DeriveExpr { expr, .. } => generate_derive(expr, x, &mut tally),
        _ => apply_op_generated(op, x),
    };
    let mut counts = tally.finish();
    let rows = finish_counted(op, x, generated, &mut counts.epilogue_sanitized);
    (rows, counts)
}

/// [`crate::model::fit`] with the training pass's telemetry. Apart from the zero-width refusal
/// below, an error is `fit`'s error and a model serializes to the bytes `fit` produces, with one
/// exception: a model with a portfolio in it, explicit or a compact head, records wall-clock fit
/// times (`fit_ms`) in its member diagnostics, which differ between two calls of `fit` itself.
///
/// It refuses what `fit` refuses, first with `fit`'s own check, and also a zero-width input
/// ([`ZeroWidthInput`]). `fit` accepts, refuses or panics on a zero-width input depending on the
/// program and on which row is empty (the domain table lists the cases). A pipeline's transforms
/// are fitted and applied in `fit`'s order; an explicit portfolio is fitted by `fit` itself and
/// reports no stage.
pub fn fit_traced(program: &GenomeIr, x: &Rows, y: &[f64]) -> Result<(Model, PipelineTelemetry)> {
    if x.is_empty() || y.len() < x.len() {
        bail!("training data is empty or target is short");
    }
    if let Some(refusal) = ZeroWidthInput::find(x) {
        return Err(refusal.into());
    }
    match program {
        GenomeIr::Pipeline(dsl) => {
            let (target_stage, rest) = split_target(&dsl.stages)?;
            let (transform_stages, head_stage) = split_stages(rest);
            let target = match target_stage {
                Some(stage) => Some(fit_target(stage, y)?),
                None => None,
            };
            let y_owned: Vec<f64>;
            let y_eff: &[f64] = match &target {
                Some(t) => {
                    y_owned = t.forward_all(y);
                    &y_owned
                }
                None => y,
            };
            let input_sanitized = count_nonfinite(x);
            let mut cur = x.clone();
            sanitize(&mut cur);
            let mut ops = Vec::with_capacity(transform_stages.len());
            let mut stages = Vec::with_capacity(transform_stages.len());
            for stage in transform_stages {
                let op = fit_op(stage, &cur, y_eff)?;
                // `fit` applies `apply_op_train`, which forwards every op to `apply_op`.
                let (next, counts) = apply_op_traced(&op, &cur);
                cur = next;
                ops.push(op);
                stages.push(counts);
            }
            let head = Box::new(fit_head(&head_stage, &cur, y_eff)?);
            Ok((
                Model::Pipeline { target, ops, head },
                PipelineTelemetry {
                    input_sanitized,
                    stages,
                },
            ))
        }
        GenomeIr::Portfolio(_) => Ok((
            crate::model::fit(program, x, y)?,
            PipelineTelemetry::default(),
        )),
        GenomeIr::Symbolic { expr, affine, .. } => {
            let mut tally = Tally::new();
            let mut sanitized = 0;
            let f: Vec<f64> = x
                .iter()
                .map(|row| {
                    let (v, faults) = expr.eval_row_traced(row);
                    let hard = tally.cell(v, faults);
                    tally.row(hard);
                    v
                })
                .map(|v| {
                    if v.is_finite() {
                        v
                    } else {
                        sanitized += 1;
                        0.0
                    }
                })
                .collect();
            // `fit`'s scaling, verbatim: the order of every floating-point operation is kept.
            let n = f.len() as f64;
            let (a, b) = if *affine {
                let fm = f.iter().sum::<f64>() / n;
                let ym = y[..f.len()].iter().sum::<f64>() / n;
                let mut cov = 0.0;
                let mut var = 0.0;
                for i in 0..f.len() {
                    let d = f[i] - fm;
                    cov += d * (y[i] - ym);
                    var += d * d;
                }
                if var > 1e-24 {
                    let a = cov / var;
                    (a, ym - a * fm)
                } else {
                    (0.0, ym)
                }
            } else {
                (1.0, 0.0)
            };
            let mut counts = tally.finish();
            counts.epilogue_sanitized = sanitized;
            Ok((
                Model::Symbolic {
                    expr: expr.clone(),
                    a,
                    b,
                },
                PipelineTelemetry {
                    input_sanitized: 0,
                    stages: vec![counts],
                },
            ))
        }
    }
}

/// [`Model::predict`] with the inference pass's telemetry. On every input it accepts, the
/// predictions have `predict`'s bits.
///
/// It refuses a zero-width input ([`ZeroWidthInput`]), which `predict` does not check. A set with
/// no rows is accepted, as `predict` accepts it.
pub fn predict_traced(model: &Model, x: &Rows) -> Result<(Vec<f64>, PipelineTelemetry)> {
    if let Some(refusal) = ZeroWidthInput::find(x) {
        return Err(refusal.into());
    }
    Ok(match model {
        Model::Pipeline { target, ops, head } => {
            let input_sanitized = count_nonfinite(x);
            let mut cur = x.clone();
            sanitize(&mut cur);
            let mut stages = Vec::with_capacity(ops.len());
            for op in ops {
                let (next, counts) = apply_op_traced(op, &cur);
                cur = next;
                stages.push(counts);
            }
            let preds = head.predict(&cur);
            let preds = match target {
                None => preds,
                Some(t) => preds.iter().map(|p| t.final_clamp(t.inverse(*p))).collect(),
            };
            (
                preds,
                PipelineTelemetry {
                    input_sanitized,
                    stages,
                },
            )
        }
        Model::Portfolio(m) => (m.predict(x), PipelineTelemetry::default()),
        Model::Symbolic { expr, a, b } => {
            let mut tally = Tally::new();
            let mut sanitized = 0;
            let preds: Vec<f64> = x
                .iter()
                .map(|row| {
                    let (v, faults) = expr.eval_row_traced(row);
                    let hard = tally.cell(v, faults);
                    tally.row(hard);
                    let v = if v.is_finite() {
                        v
                    } else {
                        sanitized += 1;
                        0.0
                    };
                    let z = a * v + b;
                    if z.is_finite() {
                        z.clamp(-60.0, 60.0)
                    } else {
                        0.0
                    }
                })
                .collect();
            let mut counts = tally.finish();
            counts.epilogue_sanitized = sanitized;
            (
                preds,
                PipelineTelemetry {
                    input_sanitized: 0,
                    stages: vec![counts],
                },
            )
        }
    })
}

/// Counts the expression cells of one stage as they are generated.
struct Tally {
    counts: StageFaultCounts,
    hard: FaultSet,
    first: Option<u64>,
    varied: bool,
}

impl Tally {
    fn new() -> Self {
        Self {
            counts: StageFaultCounts::default(),
            hard: FaultSeverity::Hard.classes(),
            first: None,
            varied: false,
        }
    }

    /// One generated cell, with the faults of its evaluation. Returns whether they include a Hard
    /// class.
    fn cell(&mut self, value: f64, faults: FaultSet) -> bool {
        self.counts.evaluated_cells += 1;
        let mut bits = faults.bits();
        while bits != 0 {
            self.counts.faulted_cells[bits.trailing_zeros() as usize] += 1;
            bits &= bits - 1;
        }
        if faults.intersects(FaultSeverity::Domain.classes()) {
            self.counts.domain_cells += 1;
        }
        if faults.intersects(FaultSeverity::Saturation.classes()) {
            self.counts.saturation_cells += 1;
        }
        match self.first {
            None => self.first = Some(value.to_bits()),
            Some(first) => self.varied |= value.to_bits() != first,
        }
        faults.intersects(self.hard)
    }

    /// The end of one generated row; `hard` when one of its cells had a Hard class.
    fn row(&mut self, hard: bool) {
        if hard {
            self.counts.hard_rows += 1;
        }
    }

    fn finish(self) -> StageFaultCounts {
        let mut counts = self.counts;
        counts.constant_output = self.first.is_some() && !self.varied;
        counts
    }
}

/// `apply_op_generated`'s `MapExpr` arm with the traced evaluators: out[i][j] = expr(x = r[j],
/// row = r). The compiled `map_row_traced` returns `map_row`'s values, and the interpreter's
/// `eval_element_traced` returns `eval_element(..).unwrap_or(0.0)`.
fn generate_map(expr: &MathExpr, x: &Rows, tally: &mut Tally) -> Rows {
    match expr.compile() {
        Ok(compiled) if compiled.uses_row_summaries() => x
            .iter()
            .map(|r| {
                let (values, faults) = compiled.map_row_traced(r);
                let mut hard = false;
                for (value, faults) in values.iter().zip(faults) {
                    hard |= tally.cell(*value, faults);
                }
                tally.row(hard);
                values
            })
            .collect(),
        Ok(_) => x
            .iter()
            .map(|r| {
                let mut hard = false;
                let mut values = Vec::with_capacity(r.len());
                for v in r {
                    let (value, faults) = expr.eval_element_traced(*v, r);
                    hard |= tally.cell(value, faults);
                    values.push(value);
                }
                tally.row(hard);
                values
            })
            .collect(),
        Err(_) => {
            tally.counts.compile_failed = true;
            x.iter()
                .map(|r| {
                    let values = vec![0.0; r.len()];
                    let mut hard = false;
                    for value in &values {
                        hard |= tally.cell(*value, FaultSet::COMPILE_FAILED);
                    }
                    tally.row(hard);
                    values
                })
                .collect()
        }
    }
}

/// `apply_op_generated`'s `DeriveExpr` arm with the traced evaluators: out[i] = [expr(row = r)].
/// Both `eval_row_traced` forms return `eval_row(..).unwrap_or(0.0)`.
fn generate_derive(expr: &MathExpr, x: &Rows, tally: &mut Tally) -> Rows {
    let mut derived = |(value, faults): (f64, FaultSet)| {
        let hard = tally.cell(value, faults);
        tally.row(hard);
        vec![value]
    };
    match expr.compile() {
        Ok(compiled) if compiled.uses_row_summaries() => x
            .iter()
            .map(|r| derived(compiled.eval_row_traced(r)))
            .collect(),
        Ok(_) => x.iter().map(|r| derived(expr.eval_row_traced(r))).collect(),
        Err(_) => {
            let rows: Rows = x
                .iter()
                .map(|_| derived((0.0, FaultSet::COMPILE_FAILED)))
                .collect();
            tally.counts.compile_failed = true;
            rows
        }
    }
}

/// `finish` in `ops.rs`, line for line, counting the non-finite values it replaces: the append
/// merge, then per row the width cap, the replacement of non-finite values and the pad of an
/// empty row.
fn finish_counted(op: &FittedOp, input: &Rows, mut out: Rows, sanitized: &mut u64) -> Rows {
    if appends(op) {
        for (i, row) in out.iter_mut().enumerate() {
            let mut merged = input[i].clone();
            merged.append(row);
            *row = merged;
        }
    }
    for row in out.iter_mut() {
        if row.len() > MAX_WIDTH {
            row.truncate(MAX_WIDTH);
        }
        for v in row.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
                *sanitized += 1;
            }
        }
        if row.is_empty() {
            row.push(0.0);
        }
    }
    out
}

/// `op_append` in `ops.rs`, verbatim: every variant is listed and there is no wildcard, so a new
/// op does not compile until its append flag is read here too.
fn appends(op: &FittedOp) -> bool {
    match op {
        FittedOp::Standardize { append, .. }
        | FittedOp::RobustScale { append, .. }
        | FittedOp::Rank { append, .. }
        | FittedOp::Blom { append, .. }
        | FittedOp::QuantileBins { append, .. }
        | FittedOp::BinOnehot { append, .. }
        | FittedOp::Hinge { append, .. }
        | FittedOp::WinsorClip { append, .. }
        | FittedOp::Cheb { append, .. }
        | FittedOp::PolyInteract { append, .. }
        | FittedOp::RffCos { append, .. }
        | FittedOp::RffSin { append, .. }
        | FittedOp::RandProj { append, .. }
        | FittedOp::LandmarkDist { append, .. }
        | FittedOp::ObliqueHinge { append, .. }
        | FittedOp::RowStats { append }
        | FittedOp::PairProd { append, .. }
        | FittedOp::TargetBinEncode { append, .. }
        | FittedOp::Sigmoid { append }
        | FittedOp::Step { append, .. }
        | FittedOp::NormRow { append }
        | FittedOp::Polar { append }
        | FittedOp::Lag { append, .. }
        | FittedOp::Roll { append, .. }
        | FittedOp::FftMag { append, .. }
        | FittedOp::RowSpectral { append }
        | FittedOp::RowDiffStats { append }
        | FittedOp::SvdProject { append, .. }
        | FittedOp::Nystrom { append, .. }
        | FittedOp::MapExpr { append, .. }
        | FittedOp::DeriveExpr { append, .. }
        | FittedOp::Symlog { append, .. }
        | FittedOp::SelectCorr { append, .. }
        | FittedOp::Impute { append, .. }
        | FittedOp::PairRatio { append, .. }
        | FittedOp::Spline { append, .. } => *append,
    }
}

/// The non-finite cells of `x`: what `sanitize` replaces.
fn count_nonfinite(x: &Rows) -> u64 {
    x.iter().flatten().filter(|v| !v.is_finite()).count() as u64
}

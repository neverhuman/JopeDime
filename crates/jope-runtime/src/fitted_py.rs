//! Fitted Python for a feat-math pipeline: the program's Python 3 recipe from
//! `feat_math::transpile::emit`, followed by the fitted state and `fitted_model()`, which returns
//! the recipe's own `Model` with every stage and the head already set, so no caller retrains.
//!
//! The recipe is generated from the program's plan (`feat_math::spec::build_plan`), so the plan is
//! what the fitted model must match: each fitted stage must carry the planned kind in the same
//! position, every parameter the recipe bakes into its prediction calls must equal the fitted one,
//! and each state is written in the shape that kind's `*_apply`/`*_predict` function reads.
//! Anything outside the recipe's slice is refused by name. [`emit_math_pipeline`] adds the
//! `math-pipeline/v1` input contract (`prediction_version: 2`, f64) and `predict_rows`.
//!
//! Floats are written with Rust's shortest round-trip form, which CPython parses back to the same
//! bits; a non-finite value anywhere in a state is refused.

use crate::calibration::OutputTransform;
use crate::error::Failure;
use crate::fitted_js::ExportCalibration;
use crate::math_pipeline::{MathPipeline, F64_PREDICTION_VERSION};
use anyhow::Result;
use feat_math::genome::GenomeIr;
use feat_math::heads::FittedHead;
use feat_math::model::Model;
use feat_math::ops::FittedOp;
use feat_math::spec::{build_plan, Plan, PlannedHead, PlannedOp, PlannedStage};
use feat_math::transpile::{emit as emit_recipe, lint_shipped_text, EmitOptions};

/// Engine output for one caller-supplied parity row.
#[derive(Clone, Debug)]
pub struct CheckRow {
    pub inputs: Vec<f64>,
    pub expected: f64,
}

/// The recipe's script guard; the fitted state goes in front of it.
const MAIN_GUARD: &str = "\n\n\nif __name__ == \"__main__\":";

fn is_identity_transform(transform: &OutputTransform) -> bool {
    transform.intercept == 0.0 && transform.slope == 1.0 && transform.lower.is_none()
}

fn python_hex(value: f64) -> String {
    if value.is_nan() {
        return "\"nan\"".into();
    }
    let bits = value.to_bits();
    let sign = if bits >> 63 == 0 { "" } else { "-" };
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    if exponent == 0 {
        if fraction == 0 {
            format!("\"{sign}0x0.0p+0\"")
        } else {
            format!("\"{sign}0x0.{fraction:013x}p-1022\"")
        }
    } else {
        format!("\"{sign}0x1.{fraction:013x}p{:+}\"", exponent - 1023)
    }
}

/// Emit standalone Python 3 that predicts with `model`, fitted from `genome` on rows `width` wide.
pub fn emit(genome: &GenomeIr, model: &Model, width: usize) -> Result<String> {
    emit_with_platt(genome, model, width, &ExportCalibration::default())
}

/// Emit a predictor with up to 64 caller-supplied engine check rows.
pub fn emit_with_checks(
    genome: &GenomeIr,
    model: &Model,
    width: usize,
    checks: &[CheckRow],
) -> Result<String> {
    emit_with_checks_and_platt(genome, model, width, &ExportCalibration::default(), checks)
}

/// Emit standalone Python with the shared optional regression and Platt calibration.
pub fn emit_with_platt(
    genome: &GenomeIr,
    model: &Model,
    width: usize,
    calibration: &ExportCalibration,
) -> Result<String> {
    emit_with_checks_and_platt(genome, model, width, calibration, &[])
}

/// Emit Python with shared calibration and caller-supplied engine check rows.
pub fn emit_with_checks_and_platt(
    genome: &GenomeIr,
    model: &Model,
    width: usize,
    calibration: &ExportCalibration,
    checks: &[CheckRow],
) -> Result<String> {
    let mut source =
        emit_calibrated_with_checks(genome, model, width, calibration.output.as_ref(), checks)?;
    if let Some(platt) = &calibration.platt {
        platt
            .validate()
            .map_err(|e| Failure::integrity(format!("invalid Platt calibration: {e:#}")))?;
        let sigmoid = &platt.sigmoid;
        source.push_str(&format!(
            r#"
PLATT_CLASS_ORDER = {}
def fitted_platt(score):
    import math
    if not math.isfinite(score) or score < 0 or score > 1:
        raise ValueError("invalid Platt score")
    z = {} if {} == 0 else {} + {} * (score / {} - {})
    if z >= 0:
        p = 1 / (1 + math.exp(-z))
    else:
        e = math.exp(z)
        p = e / (1 + e)
    return [1 - p, p]
"#,
            serde_json::to_string(&platt.class_order)?,
            py_number(sigmoid.intercept),
            py_number(sigmoid.slope),
            py_number(sigmoid.intercept),
            py_number(sigmoid.slope),
            py_number(sigmoid.scale),
            py_number(sigmoid.center),
        ));
    }
    lint(&source)?;
    Ok(source)
}

fn py_number(value: f64) -> String {
    serde_json::to_string(&value).expect("validated finite Platt coefficient")
}

/// Emit standalone Python with the optional fitted regression output transform.
pub fn emit_calibrated(
    genome: &GenomeIr,
    model: &Model,
    width: usize,
    output_transform: Option<&OutputTransform>,
) -> Result<String> {
    emit_calibrated_with_checks(genome, model, width, output_transform, &[])
}

/// Emit standalone Python with calibration and caller-supplied engine check rows.
pub fn emit_calibrated_with_checks(
    genome: &GenomeIr,
    model: &Model,
    width: usize,
    output_transform: Option<&OutputTransform>,
    checks: &[CheckRow],
) -> Result<String> {
    if checks.len() > 64 || checks.iter().any(|row| row.inputs.len() != width) {
        return Err(Failure::integrity("invalid Python self-check rows"));
    }
    let Model::Pipeline { target, ops, head } = model else {
        return Err(Failure::backend("fitted Python needs a pipeline model"));
    };
    feat_math::kernel::width::check_input_width(model, width)
        .map_err(|e| Failure::integrity(format!("fitted Python input width: {e}")))?;
    let plan = build_plan(genome).map_err(|missing| {
        Failure::backend(format!(
            "fitted Python refused: unsupported program elements: {}",
            missing.join("; ")
        ))
    })?;
    let Plan::Pipeline {
        target: planned_target,
        stages,
        head: planned_head,
    } = &plan
    else {
        return Err(Failure::backend("fitted Python needs a pipeline plan"));
    };
    if stages.len() != ops.len() {
        return Err(Failure::backend(format!(
            "fitted Python refused: {} fitted stages for {} planned stages",
            ops.len(),
            stages.len()
        )));
    }
    let mut problems = Vec::new();
    let target_state = match (planned_target, target) {
        (None, None) => None,
        (Some(planned), Some(fitted)) => target_state(planned, fitted, &mut problems),
        (Some(_), None) => {
            problems.push("no fitted target transform where the plan has one".into());
            None
        }
        (None, Some(_)) => {
            problems.push("a fitted target transform where the plan has none".into());
            None
        }
    };
    let mut states = Vec::with_capacity(ops.len());
    for (index, (planned, op)) in stages.iter().zip(ops).enumerate() {
        states.push(stage_state(index, planned, op, &mut problems));
    }
    let head_state = head_state(planned_head, head, &mut problems);
    if !problems.is_empty() {
        return Err(Failure::backend(format!(
            "fitted Python refused: {}",
            problems.join(", ")
        )));
    }
    let recipe = emit_recipe(genome, &EmitOptions::default())
        .map_err(|e| Failure::backend(format!("{e:#}")))?
        .python;
    let at = recipe
        .rfind(MAIN_GUARD)
        .ok_or_else(|| Failure::integrity("Python recipe has no script guard"))?;
    let mut block = String::from(
        "\n\n\n# Fitted state. Prediction only: these constants are what the pipeline learned.\n",
    );
    if let Some(state) = &target_state {
        block.push_str(&format!("FITTED_TARGET = {}\n", render(state)?));
    }
    let stage_states: Vec<Py> = states
        .iter()
        .map(|state| state.clone().unwrap_or(Py::None))
        .collect();
    block.push_str(&format!(
        "FITTED_STAGES = {}\nFITTED_HEAD = {}\nFITTED_WIDTH = {width}\n\n\n",
        render(&Py::Tuple(stage_states))?,
        render(&head_state.unwrap_or(Py::None))?,
    ));
    block.push_str(
        "def fitted_model():\n    \"\"\"The recipe's Model with its fitted state set. Use predict(), never fit().\"\"\"\n    model = Model()\n",
    );
    if target_state.is_some() {
        block.push_str("    model.target = FITTED_TARGET\n");
    }
    for (index, state) in states.iter().enumerate() {
        if state.is_some() {
            block.push_str(&format!(
                "    model.stage{} = FITTED_STAGES[{index}]\n",
                index + 1
            ));
        }
    }
    block.push_str(
        "    model.head = FITTED_HEAD\n    return model\n\n\n\
         def fitted_predict(rows):\n    \
         \"\"\"Predict rows of FITTED_WIDTH floats with the fitted model.\"\"\"\n    \
         rows = [list(row) for row in rows]\n    \
         for r, row in enumerate(rows):\n        \
         if len(row) != FITTED_WIDTH:\n            \
         raise ValueError(\"row %d has %d values; the model reads %d\" % (r, len(row), FITTED_WIDTH))\n    \
         if not rows:\n        \
         return []\n    \
         return fitted_model().predict(rows)",
    );
    if let Some(transform) = output_transform
        .copied()
        .filter(|t| !is_identity_transform(t))
    {
        if !transform.intercept.is_finite()
            || !transform.slope.is_finite()
            || transform.lower.is_some_and(|lower| !lower.is_finite())
        {
            return Err(Failure::integrity("non-finite regression output transform"));
        }
        let floor_code = transform.lower.map_or_else(String::new, |v| {
            format!(
                "        if math.isnan(value) or value < {v:?}:\n            value = {v:?}\n        elif value == 0.0 and {v:?} == 0.0 and (math.copysign(1.0, value) > 0.0 or math.copysign(1.0, {v:?}) > 0.0):\n            value = 0.0\n"
            )
        });
        block.push_str(&format!(
            r#"
_native_fitted_predict = fitted_predict
def fitted_predict(rows):
    predictions = _native_fitted_predict(rows)
    calibrated = []
    for prediction in predictions:
        value = {:?} + {:?} * prediction
{floor_code}        if not math.isfinite(value):
            raise ArithmeticError("calibrated prediction is not finite")
        calibrated.append(value)
    return calibrated
"#,
            transform.intercept,
            transform.slope,
            floor_code = floor_code
        ));
    }
    let rows = if checks.is_empty() {
        "()".to_owned()
    } else {
        format!(
            "({},)",
            checks
                .iter()
                .map(|row| {
                    let inputs = row
                        .inputs
                        .iter()
                        .map(|v| python_hex(*v))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("(({inputs}), {})", python_hex(row.expected))
                })
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    block.push_str(&format!(r#"
SELF_CHECK_ROWS = {rows}
SELF_CHECK_ABS_TOL = 1e-12
SELF_CHECK_REL_TOL = 1e-10
def _self_check_f64(value):
    return float("nan") if value == "nan" else float.fromhex(value)
def self_check():
    for index, (inputs, expected_hex) in enumerate(SELF_CHECK_ROWS):
        expected = _self_check_f64(expected_hex)
        actual = fitted_predict([[_self_check_f64(value) for value in inputs]])[0]
        if not (math.isnan(expected) and math.isnan(actual)) and not (
            math.isfinite(actual) and abs(actual-expected) <= SELF_CHECK_ABS_TOL + SELF_CHECK_REL_TOL*abs(expected)
        ):
            raise AssertionError("self_check failed at row %d" % index)
    return True
import sys as _self_check_sys
if __name__ == "__main__" and _self_check_sys.argv[1:] == ["--self-check"]:
    self_check()
    raise SystemExit(0)
"#));
    let mut source = String::with_capacity(recipe.len() + block.len());
    source.push_str(&recipe[..at]);
    source.push_str(&block);
    source.push_str(&recipe[at..]);
    lint(&source)?;
    Ok(source)
}

/// Standalone Python for a decoded `math-pipeline/v1` object: the fitted pipeline, the input
/// contract's encoding and `predict_rows`, with the decoder's f64 semantics
/// (`prediction_version: 2`): missing and unparseable cells encode as 0, and a non-finite
/// normalized input or prediction raises instead of becoming a number.
pub fn emit_math_pipeline(pipeline: &MathPipeline) -> Result<Vec<u8>> {
    let input = &pipeline.input;
    if input.prediction_version != F64_PREDICTION_VERSION {
        return Err(Failure::backend(format!(
            "fitted Python refused: input contract prediction_version {} (Python exports the f64 version {F64_PREDICTION_VERSION} only)",
            input.prediction_version
        )));
    }
    let genome = GenomeIr::parse(&pipeline.state.dsl)
        .map_err(|e| Failure::integrity(format!("pipeline dsl: {e:#}")))?;
    // Every exported math-pipeline carries 64 engine-derived numeric self-checks.
    let rows: Vec<Vec<f64>> = (0..64)
        .map(|i| {
            (0..input.selected.len())
                .map(|j| ((i + 17 * j) % 64) as f64 / 63.0)
                .collect()
        })
        .collect();
    let expected = pipeline.state.model.predict(&rows)?;
    let checks: Vec<CheckRow> = rows
        .into_iter()
        .zip(expected)
        .map(|(inputs, expected)| CheckRow { inputs, expected })
        .collect();
    let mut source = emit_with_checks(
        &genome,
        &pipeline.state.model,
        input.selected.len(),
        &checks,
    )?;
    let at = source
        .rfind(MAIN_GUARD)
        .ok_or_else(|| Failure::integrity("Python recipe has no script guard"))?;
    let columns = Py::Tuple(
        input
            .columns
            .iter()
            .map(|c| Py::Str(c.name.clone()))
            .collect(),
    );
    let scalers = Py::Tuple(
        input
            .columns
            .iter()
            .map(|c| Py::Tuple(vec![Py::F(c.min), Py::F(c.max), Py::B(c.constant)]))
            .collect(),
    );
    let selected = Py::Tuple(input.selected.iter().map(|&j| Py::I(j)).collect());
    let target = Py::Tuple(vec![Py::F(input.target.min), Py::F(input.target.scale)]);
    let block = format!(
        "\n\n\n# Input contract prediction_version=2: f64 arithmetic, finite results required.\n\
         MODEL_COLUMNS = {}\nINPUT_SCALERS = {}\nINPUT_SELECTED = {}\nTARGET_SCALE = {}\n{PY_CONTRACT_F64}",
        render(&columns)?,
        render(&scalers)?,
        render(&selected)?,
        render(&target)?,
    );
    source.insert_str(at, &block);
    lint(&source)?;
    Ok(source.into_bytes())
}

/// The decoder's `parse_cell_f64`/`encode_f64`/`inverse_f64` in Python. `_RUST_SPACE` is Rust's
/// `char::is_whitespace` set (`str.strip()` would also strip U+001C..U+001F), lower-casing is
/// ASCII-only like `to_ascii_lowercase`, and `_DECIMAL` is the finite subset of Rust's `f64`
/// grammar with ASCII digits (a Python `\d` would also match other scripts' digits).
const PY_CONTRACT_F64: &str = r##"_RUST_SPACE = "\t\n\x0b\x0c\r \x85\xa0                　"
_MISSING = frozenset(("", "na", "n/a", "null", "none", "nan", "?", "#n/a"))


def _is_decimal(text):
    """True for [+-]digits[.digits][e[+-]digits] or [+-].digits[e[+-]digits], ASCII digits only."""
    i = 0
    n = len(text)
    if i < n and text[i] in "+-":
        i += 1
    start = i
    while i < n and "0" <= text[i] <= "9":
        i += 1
    whole = i - start
    fraction = 0
    if i < n and text[i] == ".":
        i += 1
        mark = i
        while i < n and "0" <= text[i] <= "9":
            i += 1
        fraction = i - mark
    if whole == 0 and fraction == 0:
        return False
    if i < n and text[i] in "eE":
        i += 1
        if i < n and text[i] in "+-":
            i += 1
        mark = i
        while i < n and "0" <= text[i] <= "9":
            i += 1
        if i == mark:
            return False
    return i == n


def _parse_cell(cell):
    """A cell's value, or None when it is missing or not a finite number."""
    if cell is None:
        return None
    if isinstance(cell, (int, float)):
        try:
            value = float(cell)
        except OverflowError:
            return None
        return value if math.isfinite(value) else None
    if not isinstance(cell, str):
        raise TypeError("a cell is a str, a number or None")
    text = cell.strip(_RUST_SPACE)
    lower = "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in text)
    if lower in _MISSING:
        return None
    if lower == "true" or lower == "yes":
        return 1.0
    if lower == "false" or lower == "no":
        return 0.0
    if not _is_decimal(text):
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def _encode_cell(cell, j):
    value = _parse_cell(cell)
    low, high, constant = INPUT_SCALERS[j]
    if value is None or constant:
        return 0.0
    scaled = (value - low) / (high - low)
    if scaled < 0.0:
        scaled = 0.0
    elif scaled > 1.0:
        scaled = 1.0
    if not math.isfinite(scaled):
        raise ArithmeticError("prediction undefined: non-finite normalized input %r" % MODEL_COLUMNS[j])
    return scaled


def predict_rows(rows):
    """Predict rows of raw cells in MODEL_COLUMNS order. A cell is a str, a number or None."""
    x = []
    for r, row in enumerate(rows):
        row = list(row)
        if len(row) != len(MODEL_COLUMNS):
            raise ValueError("row %d has %d cells; the model reads %d" % (r, len(row), len(MODEL_COLUMNS)))
        x.append([_encode_cell(row[j], j) for j in INPUT_SELECTED])
    if not x:
        return []
    low, scale = TARGET_SCALE
    out = []
    for r, value in enumerate(fitted_predict(x)):
        if not math.isfinite(value):
            raise ArithmeticError("prediction undefined: row %d: non-finite fitted prediction" % r)
        result = low if scale <= 1e-12 else low + value * scale
        if not math.isfinite(result):
            raise ArithmeticError("prediction undefined: row %d: non-finite target prediction" % r)
        out.append(result)
    return out"##;

fn lint(source: &str) -> Result<()> {
    let hits = lint_shipped_text(source);
    if hits.is_empty() {
        Ok(())
    } else {
        Err(Failure::integrity(format!(
            "banned vocabulary in Python artifact: {}",
            hits.join(", ")
        )))
    }
}

/// A Python literal.
#[derive(Clone, Debug)]
enum Py {
    None,
    B(bool),
    I(usize),
    /// A tree node's leaf marker.
    Leaf,
    F(f64),
    Str(String),
    List(Vec<Py>),
    Tuple(Vec<Py>),
    Dict(Vec<(Py, Py)>),
}

impl Py {
    fn floats(values: &[f64]) -> Self {
        Self::List(values.iter().copied().map(Self::F).collect())
    }

    fn rows(rows: &[Vec<f64>]) -> Self {
        Self::List(rows.iter().map(|row| Self::floats(row)).collect())
    }

    fn ints(values: &[usize]) -> Self {
        Self::List(values.iter().copied().map(Self::I).collect())
    }

    fn pairs(pairs: &[(usize, usize)]) -> Self {
        Self::List(
            pairs
                .iter()
                .map(|&(a, b)| Self::Tuple(vec![Self::I(a), Self::I(b)]))
                .collect(),
        )
    }
}

fn render(value: &Py) -> Result<String> {
    let mut out = String::new();
    write_py(value, &mut out)?;
    Ok(out)
}

fn write_py(value: &Py, out: &mut String) -> Result<()> {
    match value {
        Py::None => out.push_str("None"),
        Py::B(b) => out.push_str(if *b { "True" } else { "False" }),
        Py::I(i) => out.push_str(&i.to_string()),
        Py::Leaf => out.push_str("-1"),
        Py::F(v) => {
            if !v.is_finite() {
                return Err(Failure::integrity("non-finite fitted state"));
            }
            // `{:?}` is the shortest form that round-trips, always with a `.` or an exponent.
            out.push_str(&format!("{v:?}"));
        }
        Py::Str(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    ' '..='~' => out.push(c),
                    _ => out.push_str(&format!("\\U{:08x}", u32::from(c))),
                }
            }
            out.push('"');
        }
        Py::Dict(items) => {
            out.push('{');
            for (i, (key, value)) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py(key, out)?;
                out.push_str(": ");
                write_py(value, out)?;
            }
            out.push('}');
        }
        Py::List(items) | Py::Tuple(items) => {
            let tuple = matches!(value, Py::Tuple(_));
            out.push(if tuple { '(' } else { '[' });
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py(item, out)?;
            }
            if tuple && items.len() == 1 {
                out.push(',');
            }
            out.push(if tuple { ')' } else { ']' });
        }
    }
    Ok(())
}

/// The state the recipe's `*_apply` reads for one fitted stage (None for a stateless kind), after
/// checking it against the planned stage. Problems are pushed and the stage is skipped.
fn stage_state(
    index: usize,
    planned: &PlannedStage,
    op: &FittedOp,
    problems: &mut Vec<String>,
) -> Option<Py> {
    use FittedOp as F;
    use PlannedOp as P;
    if fitted_append(op) != planned.append {
        problems.push(format!(
            "stage {index} append parameter differs from fitted state"
        ));
    }
    let mut differs = |name: &str| {
        problems.push(format!(
            "stage {index} {name} parameter differs from fitted state"
        ));
    };
    let same = |a: f64, b: f64| a.is_finite() && a.to_bits() == b.to_bits();
    let state = match (&planned.op, op) {
        (P::Standardize, F::Standardize { means, scales, .. }) => {
            Some(Py::Tuple(vec![Py::floats(means), Py::floats(scales)]))
        }
        (
            P::RobustScale,
            F::RobustScale {
                medians, scales, ..
            },
        ) => Some(Py::Tuple(vec![Py::floats(medians), Py::floats(scales)])),
        (P::Rank, F::Rank { sorted_cols, .. }) | (P::Blom, F::Blom { sorted_cols, .. }) => {
            Some(Py::rows(sorted_cols))
        }
        (
            P::QuantileBins { bins },
            F::QuantileBins {
                edges,
                bins: fitted,
                ..
            },
        )
        | (
            P::BinOnehot { bins },
            F::BinOnehot {
                edges,
                bins: fitted,
                ..
            },
        ) => {
            if bins != fitted {
                differs("bins");
            }
            Some(Py::rows(edges))
        }
        (P::Hinge { .. }, F::Hinge { knots, .. }) | (P::Spline { .. }, F::Spline { knots, .. }) => {
            Some(Py::rows(knots))
        }
        (P::WinsorClip { .. }, F::WinsorClip { lo, hi, .. }) => {
            Some(Py::Tuple(vec![Py::floats(lo), Py::floats(hi)]))
        }
        (P::Cheb { degree }, F::Cheb { degree: fitted, .. }) => {
            if degree != fitted {
                differs("degree");
            }
            None
        }
        (P::PolyInteract, F::PolyInteract { pairs, .. })
        | (P::PairProd { .. }, F::PairProd { pairs, .. }) => Some(Py::pairs(pairs)),
        (P::RffCos { .. }, F::RffCos { w, b, .. }) | (P::RffSin { .. }, F::RffSin { w, b, .. }) => {
            Some(Py::Tuple(vec![Py::rows(w), Py::floats(b)]))
        }
        (P::RandProj { .. }, F::RandProj { w, .. }) => Some(Py::rows(w)),
        (
            P::LandmarkDist { gamma, .. },
            F::LandmarkDist {
                landmarks,
                gamma: fitted,
                ..
            },
        )
        | (
            P::Nystrom { gamma, .. },
            F::Nystrom {
                landmarks,
                gamma: fitted,
                ..
            },
        ) => {
            if !same(*fitted, *gamma) {
                differs("gamma");
            }
            Some(Py::rows(landmarks))
        }
        (P::ObliqueHinge { .. }, F::ObliqueHinge { w, t, .. }) => {
            Some(Py::Tuple(vec![Py::rows(w), Py::floats(t)]))
        }
        (P::TargetBinEncode { .. }, F::TargetBinEncode { edges, enc, .. }) => {
            Some(Py::Tuple(vec![Py::rows(edges), Py::rows(enc)]))
        }
        (P::Step { t }, F::Step { t: fitted, .. }) => {
            if !same(*fitted, *t) {
                differs("t");
            }
            None
        }
        (P::Symlog { c }, F::Symlog { c: fitted, .. }) => {
            if !same(*fitted, *c) {
                differs("c");
            }
            None
        }
        (P::Lag { k }, F::Lag { k: fitted, .. }) | (P::Roll { k }, F::Roll { k: fitted, .. }) => {
            if k != fitted {
                differs("k");
            }
            None
        }
        // The recipe reads the fit-time width-capped k from the stage state.
        (P::FftMag { .. }, F::FftMag { k, .. }) => Some(Py::I(*k)),
        (
            P::SvdProject { .. },
            F::SvdProject {
                means, components, ..
            },
        ) => Some(Py::Tuple(vec![Py::floats(means), Py::rows(components)])),
        (P::SelectCorr { .. }, F::SelectCorr { indices, .. }) => Some(Py::ints(indices)),
        (
            P::Impute { .. },
            F::Impute {
                fill,
                indicator_cols,
                ..
            },
        ) => Some(Py::Tuple(vec![Py::floats(fill), Py::ints(indicator_cols)])),
        (P::PairRatio { .. }, F::PairRatio { pairs, eps, .. }) => {
            Some(Py::Tuple(vec![Py::pairs(pairs), Py::floats(eps)]))
        }
        (P::MapExpr { source, .. }, F::MapExpr { source: fitted, .. })
        | (P::DeriveExpr { source, .. }, F::DeriveExpr { source: fitted, .. }) => {
            if source != fitted {
                differs("expr");
            }
            None
        }
        (P::RowStats, F::RowStats { .. })
        | (P::Sigmoid, F::Sigmoid { .. })
        | (P::NormRow, F::NormRow { .. })
        | (P::Polar, F::Polar { .. })
        | (P::RowSpectral, F::RowSpectral { .. })
        | (P::RowDiffStats, F::RowDiffStats { .. }) => None,
        (planned, _) => {
            problems.push(format!(
                "fitted {} where the plan has {}",
                fitted_kind(op),
                planned_kind(planned)
            ));
            None
        }
    };
    state
}

fn fitted_append(op: &FittedOp) -> bool {
    use FittedOp as F;
    match op {
        F::Standardize { append, .. }
        | F::RobustScale { append, .. }
        | F::Rank { append, .. }
        | F::Blom { append, .. }
        | F::QuantileBins { append, .. }
        | F::BinOnehot { append, .. }
        | F::Hinge { append, .. }
        | F::WinsorClip { append, .. }
        | F::Cheb { append, .. }
        | F::PolyInteract { append, .. }
        | F::RffCos { append, .. }
        | F::RffSin { append, .. }
        | F::RandProj { append, .. }
        | F::LandmarkDist { append, .. }
        | F::ObliqueHinge { append, .. }
        | F::RowStats { append }
        | F::PairProd { append, .. }
        | F::TargetBinEncode { append, .. }
        | F::Sigmoid { append }
        | F::Step { append, .. }
        | F::NormRow { append }
        | F::Polar { append }
        | F::Lag { append, .. }
        | F::Roll { append, .. }
        | F::FftMag { append, .. }
        | F::RowSpectral { append }
        | F::RowDiffStats { append }
        | F::SvdProject { append, .. }
        | F::Nystrom { append, .. }
        | F::MapExpr { append, .. }
        | F::DeriveExpr { append, .. }
        | F::Symlog { append, .. }
        | F::SelectCorr { append, .. }
        | F::Impute { append, .. }
        | F::PairRatio { append, .. }
        | F::Spline { append, .. } => *append,
    }
}

/// The program name of a fitted stage, for refusal messages.
fn fitted_kind(op: &FittedOp) -> &'static str {
    use FittedOp as F;
    match op {
        F::Standardize { .. } => "standardize",
        F::RobustScale { .. } => "robust_scale",
        F::Rank { .. } => "rank",
        F::Blom { .. } => "blom",
        F::QuantileBins { .. } => "quantile_bins",
        F::BinOnehot { .. } => "bin_onehot",
        F::Hinge { .. } => "hinge",
        F::WinsorClip { .. } => "winsor_clip",
        F::Cheb { .. } => "cheb",
        F::PolyInteract { .. } => "poly_interact",
        F::RffCos { .. } => "rff_cos",
        F::RffSin { .. } => "rff_sin",
        F::RandProj { .. } => "rand_proj",
        F::LandmarkDist { .. } => "landmark_dist",
        F::ObliqueHinge { .. } => "oblique_hinge",
        F::RowStats { .. } => "row_stats",
        F::PairProd { .. } => "pair_prod",
        F::TargetBinEncode { .. } => "target_bin_encode",
        F::Sigmoid { .. } => "sigmoid",
        F::Step { .. } => "step",
        F::NormRow { .. } => "norm_row",
        F::Polar { .. } => "polar",
        F::Lag { .. } => "lag",
        F::Roll { .. } => "roll",
        F::FftMag { .. } => "fft_mag",
        F::RowSpectral { .. } => "row_spectral",
        F::RowDiffStats { .. } => "row_diff_stats",
        F::SvdProject { .. } => "svd_project",
        F::Nystrom { .. } => "nystrom",
        F::MapExpr { .. } => "map",
        F::DeriveExpr { .. } => "derive",
        F::Symlog { .. } => "symlog",
        F::SelectCorr { .. } => "select_corr",
        F::Impute { .. } => "impute",
        F::PairRatio { .. } => "pair_ratio",
        F::Spline { .. } => "spline",
    }
}

/// The program name of a planned stage, for refusal messages.
fn planned_kind(op: &PlannedOp) -> &'static str {
    use PlannedOp as P;
    match op {
        P::Standardize => "standardize",
        P::RobustScale => "robust_scale",
        P::Rank => "rank",
        P::Blom => "blom",
        P::QuantileBins { .. } => "quantile_bins",
        P::BinOnehot { .. } => "bin_onehot",
        P::Hinge { .. } => "hinge",
        P::WinsorClip { .. } => "winsor_clip",
        P::Cheb { .. } => "cheb",
        P::PolyInteract => "poly_interact",
        P::RffCos { .. } => "rff_cos",
        P::RffSin { .. } => "rff_sin",
        P::RandProj { .. } => "rand_proj",
        P::LandmarkDist { .. } => "landmark_dist",
        P::ObliqueHinge { .. } => "oblique_hinge",
        P::RowStats => "row_stats",
        P::PairProd { .. } => "pair_prod",
        P::TargetBinEncode { .. } => "target_bin_encode",
        P::Sigmoid => "sigmoid",
        P::Step { .. } => "step",
        P::NormRow => "norm_row",
        P::Polar => "polar",
        P::Lag { .. } => "lag",
        P::Roll { .. } => "roll",
        P::FftMag { .. } => "fft_mag",
        P::RowSpectral => "row_spectral",
        P::RowDiffStats => "row_diff_stats",
        P::SvdProject { .. } => "svd_project",
        P::Nystrom { .. } => "nystrom",
        P::MapExpr { .. } => "map",
        P::DeriveExpr { .. } => "derive",
        P::Symlog { .. } => "symlog",
        P::SelectCorr { .. } => "select_corr",
        P::Impute { .. } => "impute",
        P::PairRatio { .. } => "pair_ratio",
        P::Spline { .. } => "spline",
        P::RowVectorV2 { .. } => "a row-vector v2 stage",
        P::RowVectorInputV2 { .. } => "a row-vector v2 input",
        P::ScalarV2 { .. } => "a scalar v2 stage",
    }
}

/// The state the recipe's head `*_predict` reads, after checking it against the planned head.
fn head_state(planned: &PlannedHead, head: &FittedHead, problems: &mut Vec<String>) -> Option<Py> {
    use FittedHead as H;
    use PlannedHead as P;
    Some(match (planned, head) {
        (P::Ridge { .. }, H::Ridge(r)) => ridge(r),
        (P::RidgeGcv, H::RidgeGcv(r)) => Py::Tuple(vec![
            Py::ints(&r.cols),
            Py::floats(&r.means),
            Py::floats(&r.scales),
            Py::floats(&r.w),
        ]),
        (P::ElasticNet { .. }, H::ElasticNet(e)) => elastic_net(e),
        (P::Gbm(_), H::Gbm(g)) => gbm(g),
        (
            P::Blend { .. },
            H::Ensemble {
                ridge: r,
                gbm: g,
                w,
            },
        ) => Py::Tuple(vec![ridge(r), gbm(g), Py::F(*w)]),
        (P::HistGbm2 { .. }, H::HistGbm2(h)) => hist_gbm2(h),
        (P::Krr { .. }, H::Krr(k)) => krr(k),
        (P::Gpr { .. }, H::Gpr(g)) => Py::Tuple(vec![
            Py::floats(&g.means),
            Py::floats(&g.scales),
            Py::rows(&g.basis),
            Py::floats(&g.alpha),
            Py::floats(&g.ls),
            Py::F(g.sf2),
            Py::F(g.y_mean),
        ]),
        // The kernel terms are baked into the predict call; `theta` keeps its trailing noise entry,
        // which the recipe's decoder indexes.
        (P::GprKernel { spec, .. }, H::GprKernel(g)) => {
            if *spec != g.spec {
                problems.push("head kernel parameter differs from fitted state".into());
            }
            Py::Tuple(vec![
                Py::floats(&g.means),
                Py::floats(&g.scales),
                Py::rows(&g.basis),
                Py::floats(&g.alpha),
                Py::floats(&g.theta),
                Py::F(g.y_mean),
            ])
        }
        // k and gamma are baked into the predict call.
        (P::Ksmooth { k, gamma }, H::Ksmooth(m)) => {
            if *k != m.k {
                problems.push("head k parameter differs from fitted state".into());
            }
            if !(m.gamma.is_finite() && m.gamma.to_bits() == gamma.to_bits()) {
                problems.push("head gamma parameter differs from fitted state".into());
            }
            Py::Tuple(vec![
                Py::floats(&m.means),
                Py::floats(&m.scales),
                Py::rows(&m.z),
                Py::floats(&m.ys),
                Py::F(m.y_mean),
            ])
        }
        // The library is fixed: (weights, ridge_a, ridge_b, net, trees, kernel), None if unused.
        (P::Stack, H::Stack(m)) => Py::Tuple(vec![
            Py::floats(&m.weights),
            m.ridge_a.as_ref().map_or(Py::None, ridge),
            m.ridge_b.as_ref().map_or(Py::None, ridge),
            m.net.as_ref().map_or(Py::None, elastic_net),
            m.trees.as_ref().map_or(Py::None, hist_gbm2),
            m.kernel.as_ref().map_or(Py::None, krr),
        ]),
        (P::Portfolio(plan), H::Portfolio(m)) => portfolio(plan, m, problems)?,
        _ => {
            problems.push(format!(
                "fitted head {} does not match the planned head {}",
                fitted_head_kind(head),
                planned_head_kind(planned)
            ));
            return None;
        }
    })
}

fn ridge(r: &feat_math::heads::RidgeModel) -> Py {
    Py::Tuple(vec![
        Py::floats(&r.means),
        Py::floats(&r.scales),
        Py::floats(&r.w),
    ])
}

fn elastic_net(e: &feat_math::heads::ElasticNetModel) -> Py {
    Py::Tuple(vec![
        Py::floats(&e.means),
        Py::floats(&e.scales),
        Py::F(e.w0),
        Py::floats(&e.w),
    ])
}

fn krr(k: &feat_math::krr::KrrModel) -> Py {
    Py::Tuple(vec![
        Py::floats(&k.means),
        Py::floats(&k.scales),
        Py::rows(&k.basis),
        Py::floats(&k.alpha),
        Py::F(k.gamma),
        Py::F(k.y_mean),
    ])
}

/// `(init, lr, bags)`. An empty bag stays: the recipe divides by the number of bags.
fn hist_gbm2(h: &feat_math::boost2::HistGbm2Model) -> Py {
    Py::Tuple(vec![
        Py::F(h.init),
        Py::F(h.lr),
        Py::List(
            h.bags
                .iter()
                .map(|bag| Py::List(bag.iter().map(leafwise_tree).collect()))
                .collect(),
        ),
    ])
}

/// `(mean_only, y_mean, width, keep, proj, entries, clip_lo, clip_hi)`, one entry
/// `(component, projected, weight, state)` per fitted member. The component number is the
/// member's 1-based position in the planned roster, which names its `component<n>_predict`; a
/// member's state lists its stateful stages in order, then its head.
fn portfolio(
    plan: &feat_math::spec::PortfolioPlan,
    m: &feat_math::portfolio::PortfolioModel,
    problems: &mut Vec<String>,
) -> Option<Py> {
    use feat_math::portfolio::FittedMemberState as S;
    use feat_math::spec::PlannedMemberKind as K;
    if m.members.len() != m.weights.len() {
        problems.push("portfolio members and weights differ in length".into());
        return None;
    }
    let proj = match &m.proj {
        None => Py::None,
        Some(FittedOp::SvdProject {
            means, components, ..
        }) => Py::Tuple(vec![Py::floats(means), Py::rows(components)]),
        Some(_) => {
            problems.push("portfolio projection is not a principal component projection".into());
            return None;
        }
    };
    let mut entries = Vec::with_capacity(m.members.len());
    for (member, &weight) in m.members.iter().zip(&m.weights) {
        let Some(index) = plan.members.iter().position(|p| p.name == member.name) else {
            problems.push("a fitted portfolio member is not in the planned roster".into());
            continue;
        };
        let state = match (&plan.members[index].kind, &member.state) {
            (_, S::Mean | S::Skipped) => Py::None,
            (K::GroupMean, S::GroupMean(group)) => group_mean(group, problems)?,
            (K::Pipeline { stages, head }, S::Fitted(model)) => {
                let Model::Pipeline {
                    target: None,
                    ops,
                    head: fitted,
                } = model.as_ref()
                else {
                    problems.push("a portfolio member is not a plain pipeline".into());
                    continue;
                };
                if ops.len() != stages.len() {
                    problems.push(format!(
                        "portfolio member: {} fitted stages for {} planned stages",
                        ops.len(),
                        stages.len()
                    ));
                    continue;
                }
                let mut states: Vec<Py> = stages
                    .iter()
                    .zip(ops)
                    .enumerate()
                    .filter_map(|(i, (planned, op))| stage_state(i, planned, op, problems))
                    .collect();
                states.extend(head_state(head, fitted, problems));
                Py::Tuple(states)
            }
            _ => {
                problems.push("a fitted portfolio member does not match its planned kind".into());
                continue;
            }
        };
        entries.push(Py::Tuple(vec![
            Py::I(index + 1),
            Py::B(member.proj_view),
            Py::F(weight),
            state,
        ]));
    }
    Some(Py::Tuple(vec![
        Py::B(m.mean_only),
        Py::F(m.y_mean),
        Py::I(m.width),
        Py::ints(&m.keep),
        proj,
        Py::List(entries),
        Py::F(m.clip_lo),
        Py::F(m.clip_hi),
    ]))
}

/// `(table, g_mean)`, the table keyed by the recipe's `row_signature` tuples. The model's fields
/// are crate-private, so its serialized form is read: `table` is `[(key bits, mean), ...]`.
fn group_mean(
    group: &feat_math::portfolio::GroupMeanModel,
    problems: &mut Vec<String>,
) -> Option<Py> {
    #[derive(serde::Deserialize)]
    struct GroupMean {
        table: Vec<(Vec<u64>, f64)>,
        g_mean: f64,
    }
    let Ok(group) = serde_json::to_value(group).and_then(serde_json::from_value::<GroupMean>)
    else {
        problems.push("unreadable group-mean member state".into());
        return None;
    };
    Some(Py::Tuple(vec![
        Py::Dict(
            group
                .table
                .into_iter()
                .map(|(key, mean)| {
                    let key = key
                        .into_iter()
                        .map(|bits| Py::F(f64::from_bits(bits)))
                        .collect();
                    (Py::Tuple(key), Py::F(mean))
                })
                .collect(),
        ),
        Py::F(group.g_mean),
    ]))
}

/// `(edges, trees, init, lr, n_feat, fallback)`; a node is `(feature, bin, left, right, value)`
/// with feature -1 for a leaf, in the fitted pre-order.
fn gbm(g: &feat_math::heads::GbmModel) -> Py {
    use feat_math::heads::GbmNode;
    let trees = g
        .trees
        .iter()
        .map(|tree| {
            Py::List(
                tree.iter()
                    .map(|node| match node {
                        GbmNode::Leaf(v) => {
                            Py::Tuple(vec![Py::Leaf, Py::I(0), Py::I(0), Py::I(0), Py::F(*v)])
                        }
                        GbmNode::Split {
                            feature,
                            thr_bin,
                            left,
                            right,
                        } => Py::Tuple(vec![
                            Py::I(*feature),
                            Py::I(usize::from(*thr_bin)),
                            Py::I(*left),
                            Py::I(*right),
                            Py::F(0.0),
                        ]),
                    })
                    .collect(),
            )
        })
        .collect();
    Py::Tuple(vec![
        Py::rows(&g.edges),
        Py::List(trees),
        Py::F(g.init),
        Py::F(g.lr),
        Py::I(g.n_feat),
        Py::F(g.constant_value),
    ])
}

/// A leaf-wise tree: nodes `(feature, threshold, left, right, value)`, feature -1 for a leaf.
fn leafwise_tree(tree: &feat_math::boost2::BoostTree) -> Py {
    use feat_math::boost2::BoostNode;
    Py::List(
        tree.iter()
            .map(|node| match node {
                BoostNode::Leaf(v) => {
                    Py::Tuple(vec![Py::Leaf, Py::F(0.0), Py::I(0), Py::I(0), Py::F(*v)])
                }
                BoostNode::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => Py::Tuple(vec![
                    Py::I(*feature),
                    Py::F(*threshold),
                    Py::I(*left),
                    Py::I(*right),
                    Py::F(0.0),
                ]),
            })
            .collect(),
    )
}

fn fitted_head_kind(head: &FittedHead) -> &'static str {
    use FittedHead as H;
    match head {
        H::Ridge(_) => "ridge",
        H::RidgeGcv(_) => "ridge_gcv",
        H::ElasticNet(_) => "elasticnet",
        H::Gbm(_) => "gbm",
        H::Ensemble { .. } => "ensemble",
        H::HistGbm2(_) => "histgbm2",
        H::Krr(_) => "krr",
        H::Gpr(_) => "gpr",
        H::GprKernel(_) => "gpr with a kernel",
        H::Ksmooth(_) => "ksmooth",
        H::Hurdle2(_) => "hurdle2",
        H::Stack(_) => "stack",
        H::Portfolio(_) => "portfolio",
    }
}

fn planned_head_kind(head: &PlannedHead) -> &'static str {
    use PlannedHead as P;
    match head {
        P::Ridge { .. } => "ridge",
        P::RidgeGcv => "ridge_gcv",
        P::ElasticNet { .. } => "elasticnet",
        P::Gbm(_) => "gbm",
        P::Blend { .. } => "ensemble",
        P::HistGbm2 { .. } => "histgbm2",
        P::Krr { .. } => "krr",
        P::Gpr { .. } => "gpr",
        P::GprKernel { .. } => "gpr with a kernel",
        P::Ksmooth { .. } => "ksmooth",
        P::Stack => "stack",
        P::Portfolio(_) => "portfolio",
    }
}

/// The target transform's `self.target` state, after checking its kind against the plan.
fn target_state(
    planned: &feat_math::spec::PlannedTarget,
    fitted: &feat_math::target_transform::FittedTarget,
    problems: &mut Vec<String>,
) -> Option<Py> {
    use feat_math::spec::TargetKind as K;
    use feat_math::target_transform::FittedTarget as T;
    match (planned.kind, fitted) {
        (
            K::RankGauss,
            T::RankGauss {
                sorted_values,
                table_values,
                table_scores,
            },
        ) => Some(Py::Tuple(vec![
            Py::floats(sorted_values),
            Py::floats(table_values),
            Py::floats(table_scores),
        ])),
        // The recipe's log1p inverse reads no state.
        (K::Log1p, T::Log1p) => None,
        (K::Winsor, T::Winsor { lo, hi }) => Some(Py::Tuple(vec![Py::F(*lo), Py::F(*hi)])),
        (
            K::ZScore,
            T::ZScore {
                mu,
                sigma,
                clamp_lo,
                clamp_hi,
            },
        ) => Some(Py::Tuple(vec![
            Py::F(*mu),
            Py::F(*sigma),
            Py::F(*clamp_lo),
            Py::F(*clamp_hi),
        ])),
        _ => {
            problems.push("fitted target transform kind differs from the planned one".into());
            None
        }
    }
}

#[cfg(test)]
mod tests {
    //! Gate E2 for Python: every operation and head the capability registry marks Python-supported,
    //! fitted in Rust, exported with its fitted state and run under `python3` without refitting.
    //! Each prediction must agree with `Model::predict` to `abs 1e-12 + rel 1e-10` (spec §10.5).
    //! A missing `python3` is a skip locally and a failure under `JOPE_PARITY_STRICT=1`, as in
    //! `tests/fitted_js_qualification.rs`.

    use super::*;
    use feat_math::evolve::api::capability::{CapabilityKind, Language, Registry};
    use feat_math::model::fit;
    use std::io::ErrorKind;
    use std::path::Path;
    use std::process::Command;

    const WIDTH: usize = 5;

    /// Hand-picked programs beyond the registry probes: non-default parameters, appended stages,
    /// every target kind and multi-stage chains.
    const PROGRAMS: &[&str] = &[
        "(pipe X (ridge))",
        "(pipe X (standardize append=true) (ridge lambda_frac=0.1))",
        "(pipe X (robust_scale append=true) (ridge))",
        "(pipe X (rank) (ridge))",
        "(pipe X (blom append=true) (ridge))",
        "(pipe X (quantile_bins bins=8) (ridge))",
        "(pipe X (bin_onehot bins=4 append=true) (ridge))",
        "(pipe X (hinge knots=3) (ridge))",
        "(pipe X (winsor_clip q=0.05) (ridge))",
        "(pipe X (cheb degree=3 append=true) (ridge))",
        "(pipe X (poly_interact append=true) (ridge))",
        "(pipe X (rff_cos n=8) (ridge))",
        "(pipe X (rff_sin n=16 seed=2) (ridge))",
        "(pipe X (rand_proj k=4) (ridge))",
        "(pipe X (landmark_dist k=4 gamma=0.25) (ridge))",
        "(pipe X (oblique_hinge n=4) (ridge))",
        "(pipe X (row_stats append=true) (ridge))",
        "(pipe X (pair_prod append=true topk=4) (ridge))",
        "(pipe X (target_bin_encode bins=4 prior=1) (ridge))",
        "(pipe X (sigmoid append=true) (ridge))",
        "(pipe X (step t=0.25 append=true) (ridge))",
        "(pipe X (norm_row append=true) (ridge))",
        "(pipe X (row_diff_stats append=true) (ridge))",
        "(pipe X (polar append=true) (ridge))",
        "(pipe X (lag k=2 append=true) (ridge))",
        "(pipe X (roll k=3 append=true) (ridge))",
        "(pipe X (fft_mag k=9 append=true) (ridge))",
        "(pipe X (row_spectral append=true) (ridge))",
        "(pipe X (svd_project k=3 iters=10) (ridge))",
        "(pipe X (nystrom k=8 gamma=0.5) (ridge))",
        "(pipe X (map expr=\"sigmoid(col(0))\") (ridge))",
        "(pipe X (derive expr=\"row_mean + c0\") (ridge))",
        "(pipe X (symlog c=0.3 append=true) (ridge))",
        "(pipe X (select_corr k=3) (ridge))",
        "(pipe X (select_corr k=2 by=var append=true) (ridge))",
        "(pipe X (impute strategy=mean add_indicator=true) (ridge))",
        "(pipe X (impute strategy=const value=0.5) (ridge))",
        "(pipe X (pair_ratio topk=4 append=true) (ridge))",
        "(pipe X (spline knots=4 append=true) (ridge))",
        "(pipe X (robust_scale) (hinge knots=3 append=true) (elasticnet))",
        "(pipe X (standardize) (rff_cos n=8 append=true) (ridge_gcv))",
        "(pipe X (elasticnet alpha=0.001 l1_ratio=0.5 sweeps=60))",
        "(pipe X (gbm rounds=20 depth=3))",
        "(pipe X (ensemble lambda_frac=0.05 rounds=10 depth=2))",
        "(pipe X (histgbm2 rounds=20 leaves=8 bags=2 loss=huber))",
        "(pipe X (krr m=50))",
        "(pipe X (gpr m=50 steps=5))",
        "(pipe X (gpr kernel=rq+linear m=50 steps=5))",
        "(pipe X (ksmooth k=16 gamma=2.0))",
        "(pipe X (stack))",
        "(pipe X (portfolio))",
        "(pipe X (portfolio preset=none add1=group_mean add2=knn_loo repeats=1))",
        "(pipe X (target kind=rank_gauss) (ridge))",
        "(pipe X (target kind=log1p) (standardize) (ridge))",
        "(pipe X (target kind=winsor q=0.1) (ridge))",
        "(pipe X (target kind=zscore) (gbm rounds=10 depth=2))",
    ];

    /// Deterministic rows in [0, 1) from a fixed linear congruential sequence.
    fn rows(n: usize, width: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                (0..width)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        (state >> 11) as f64 / (1u64 << 53) as f64
                    })
                    .collect()
            })
            .collect()
    }

    fn train() -> (Vec<Vec<f64>>, Vec<f64>) {
        let x = rows(160, WIDTH, 7);
        let y = x
            .iter()
            .map(|r| 2.0 * r[0] - r[1] * r[2] + (3.0 * r[3]).sin() + 0.5 * r[4] + 1.5)
            .collect();
        (x, y)
    }

    /// Held rows, three training rows, values outside the training range on both sides, signed
    /// zero, exact zeros (the imputer's missing marker), a constant row and non-finite cells (the
    /// engine zeroes them).
    fn test_rows() -> Vec<Vec<f64>> {
        let mut test = rows(40, WIDTH, 99);
        test.extend(train().0.into_iter().take(3));
        test.push(vec![-0.5; WIDTH]);
        test.push(vec![1.5; WIDTH]);
        test.push(vec![0.0; WIDTH]);
        test.push(vec![-0.0; WIDTH]);
        test.push(vec![0.25; WIDTH]);
        test.push(vec![1e6, -1e6, 0.5, 0.0, 1.0]);
        test.push(vec![f64::NAN, 0.5, f64::INFINITY, f64::NEG_INFINITY, 0.3]);
        test
    }

    fn py_float(v: f64) -> String {
        if v.is_nan() {
            "float(\"nan\")".into()
        } else if v.is_infinite() {
            format!("float(\"{}inf\")", if v < 0.0 { "-" } else { "" })
        } else {
            format!("{v:?}")
        }
    }

    fn py_rows(rows: &[Vec<f64>]) -> String {
        let rows: Vec<String> = rows
            .iter()
            .map(|r| {
                format!(
                    "[{}]",
                    r.iter()
                        .map(|&v| py_float(v))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
            .collect();
        format!("[{}]", rows.join(", "))
    }

    /// `JOPE_PYTHON` selects another interpreter, e.g. the oldest supported one (3.9).
    fn python() -> std::ffi::OsString {
        std::env::var_os("JOPE_PYTHON").unwrap_or_else(|| "python3".into())
    }

    /// Run `call` (an expression over the loaded module namespace `m`) under `python3 -I -B` and
    /// read one `repr` float per line. `None` when python3 is absent and strict mode is off.
    fn run_python(dir: &Path, source: &str, call: &str) -> Option<Result<Vec<f64>, String>> {
        let predictor = dir.join("predictor.py");
        let driver = dir.join("driver.py");
        std::fs::write(&predictor, source).unwrap();
        std::fs::write(
            &driver,
            format!(
                "import runpy, sys\nm = runpy.run_path(sys.argv[1], run_name=\"predictor\")\n\
                 out = {call}\nsys.stdout.write(\"\\n\".join(repr(v) for v in out))\n"
            ),
        )
        .unwrap();
        let output = match Command::new(python())
            .args(["-I", "-B"])
            .arg(&driver)
            .arg(&predictor)
            .output()
        {
            Ok(output) => output,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                assert!(
                    std::env::var_os("JOPE_PARITY_STRICT").is_none_or(|v| v != "1"),
                    "fitted Python: python3 is unavailable and JOPE_PARITY_STRICT=1"
                );
                eprintln!("fitted Python SKIP: python3 is unavailable");
                return None;
            }
            Err(error) => panic!("could not start python3: {error}"),
        };
        if !output.status.success() {
            return Some(Err(String::from_utf8_lossy(&output.stderr).into_owned()));
        }
        let text = String::from_utf8(output.stdout).unwrap();
        Some(Ok(text
            .lines()
            .map(|line| {
                line.parse::<f64>()
                    .unwrap_or_else(|e| panic!("{line:?}: {e}"))
            })
            .collect()))
    }

    fn within_tolerance(actual: f64, expected: f64) -> bool {
        if expected.is_nan() {
            return actual.is_nan();
        }
        (actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs()
    }

    /// The registry's probe program for one Python-supported operation or head.
    fn probe(kind: CapabilityKind, name: &str) -> String {
        match (kind, name) {
            (_, "target") => "(pipe X (target kind=winsor) (ridge))".into(),
            (_, "map" | "derive") => format!("(pipe X ({name} expr=\"col(0)\") (ridge))"),
            (CapabilityKind::Head, _) => format!("(pipe X ({name}))"),
            _ => format!("(pipe X ({name}) (ridge))"),
        }
    }

    /// Fit, export and run one program; returns `(rows, bit-identical rows, max abs error)`.
    fn check(dir: &Path, dsl: &str) -> Option<(usize, usize, f64)> {
        let (x, y) = train();
        let test = test_rows();
        let genome = GenomeIr::parse(dsl).unwrap_or_else(|e| panic!("{dsl}: {e:#}"));
        let model = fit(&genome, &x, &y).unwrap_or_else(|e| panic!("{dsl}: {e:#}"));
        let expected = model.predict(&test).unwrap();
        let source = emit(&genome, &model, WIDTH).unwrap_or_else(|e| panic!("{dsl}: {e:#}"));
        assert!(
            source.contains("\ndef fitted_model():\n"),
            "{dsl}: no fitted model"
        );
        let actual = run_python(
            dir,
            &source,
            &format!("m[\"fitted_predict\"]({})", py_rows(&test)),
        )?
        .unwrap_or_else(|stderr| panic!("{dsl}: python3 failed:\n{stderr}"));
        assert_eq!(actual.len(), expected.len(), "{dsl}");
        let mut exact = 0;
        let mut worst = 0.0_f64;
        for (row, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                within_tolerance(*a, *e),
                "{dsl}: row {row}: Python {a:?} vs Rust {e:?}"
            );
            exact += usize::from(a.to_bits() == e.to_bits());
            worst = worst.max((a - e).abs());
        }
        Some((expected.len(), exact, worst))
    }

    #[test]
    fn fitted_python_matches_rust_for_every_registry_supported_operation_and_head() {
        let dir = tempfile::tempdir().unwrap();
        let registry = Registry::current();
        let mut programs: Vec<String> = registry
            .entries
            .iter()
            .filter(|row| matches!(row.kind, CapabilityKind::Operation | CapabilityKind::Head))
            .filter(|row| registry.supports(row.kind, row.name, Language::Python))
            .map(|row| probe(row.kind, row.name))
            .collect();
        let probes = programs.len();
        assert!(probes >= 50, "only {probes} Python-supported registry rows");
        programs.extend(PROGRAMS.iter().map(|dsl| dsl.to_string()));
        for dsl in &programs {
            let Some((rows, exact, worst)) = check(dir.path(), dsl) else {
                return;
            };
            eprintln!(
                "E2 python {dsl}: {exact}/{rows} rows bit-identical, max abs error {worst:e}"
            );
        }
    }

    /// Spec §10.5 decision flips for tree and threshold heads: on randomized rows, the share of
    /// rows whose Python prediction leaves the numeric tolerance must stay at or under 0.1%.
    #[test]
    fn tree_and_threshold_programs_do_not_flip_on_randomized_rows() {
        const TREES: &[&str] = &[
            "(pipe X (gbm rounds=20 depth=3))",
            "(pipe X (ensemble rounds=10 depth=2))",
            "(pipe X (histgbm2 rounds=20 leaves=8 bags=2))",
            "(pipe X (stack))",
            "(pipe X (step t=0.5) (quantile_bins bins=8 append=true) (ridge))",
            "(pipe X (bin_onehot bins=4) (target_bin_encode bins=4 append=true) (ridge))",
        ];
        let dir = tempfile::tempdir().unwrap();
        let (x, y) = train();
        // Rows over [-0.25, 1.25): the training range and both sides of it.
        let random: Vec<Vec<f64>> = rows(2000, WIDTH, 2026)
            .into_iter()
            .map(|r| r.into_iter().map(|v| 1.5 * v - 0.25).collect())
            .collect();
        for dsl in TREES {
            let genome = GenomeIr::parse(dsl).unwrap();
            let model = fit(&genome, &x, &y).unwrap();
            let expected = model.predict(&random).unwrap();
            let source = emit(&genome, &model, WIDTH).unwrap();
            let Some(actual) = run_python(
                dir.path(),
                &source,
                &format!("m[\"fitted_predict\"]({})", py_rows(&random)),
            ) else {
                return;
            };
            let actual = actual.unwrap_or_else(|stderr| panic!("{dsl}: python3 failed:\n{stderr}"));
            assert_eq!(actual.len(), expected.len(), "{dsl}");
            let flips = actual
                .iter()
                .zip(&expected)
                .filter(|(a, e)| !within_tolerance(**a, **e))
                .count();
            let exact = actual
                .iter()
                .zip(&expected)
                .filter(|(a, e)| a.to_bits() == e.to_bits())
                .count();
            eprintln!(
                "E2 python flips {dsl}: {flips}/{} rows, {exact} bit-identical",
                random.len()
            );
            assert!(flips * 1000 <= random.len(), "{dsl}: {flips} flips");
        }
    }

    /// The portfolio state keeps every active member; duplicated training rows give the
    /// group-mean member weight, so its table is exported and looked up row by row.
    #[test]
    fn portfolio_exports_its_active_members_including_group_means() {
        use feat_math::portfolio::FittedMemberState as S;
        let distinct = rows(40, WIDTH, 11);
        let x: Vec<Vec<f64>> = (0..4).flat_map(|_| distinct.clone()).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, r)| (7.0 * r[0]).sin() + r[1] * r[4] + 0.01 * (i % 4) as f64)
            .collect();
        let mut duplicated_test = distinct.clone();
        duplicated_test.extend(rows(10, WIDTH, 12));
        let (plain_x, plain_y) = train();
        let dir = tempfile::tempdir().unwrap();
        for (dsl, x, y, test, min_active) in [
            ("(pipe X (portfolio))", &plain_x, &plain_y, test_rows(), 2),
            (
                "(pipe X (portfolio preset=none add1=group_mean add2=knn_loo repeats=1))",
                &x,
                &y,
                duplicated_test,
                2,
            ),
        ] {
            let genome = GenomeIr::parse(dsl).unwrap();
            let model = fit(&genome, x, y).unwrap();
            let Model::Pipeline { head, .. } = &model else {
                panic!("{dsl}: not a pipeline");
            };
            let FittedHead::Portfolio(p) = head.as_ref() else {
                panic!("{dsl}: not a portfolio head");
            };
            assert!(!p.mean_only, "{dsl}: mean-only portfolio");
            let active = p
                .members
                .iter()
                .zip(&p.weights)
                .filter(|(m, w)| **w > 0.0 && matches!(m.state, S::Fitted(_) | S::GroupMean(_)))
                .count();
            assert!(active >= min_active, "{dsl}: {active} active members");
            if dsl.contains("group_mean") {
                assert!(
                    p.members
                        .iter()
                        .zip(&p.weights)
                        .any(|(m, w)| *w > 0.0 && matches!(m.state, S::GroupMean(_))),
                    "{dsl}: no weighted group-mean member"
                );
            }
            let expected = model.predict(&test).unwrap();
            let source = emit(&genome, &model, WIDTH).unwrap();
            assert!(source.contains("FITTED_HEAD = (False, "), "{dsl}");
            let Some(actual) = run_python(
                dir.path(),
                &source,
                &format!("m[\"fitted_predict\"]({})", py_rows(&test)),
            ) else {
                return;
            };
            let actual = actual.unwrap_or_else(|stderr| panic!("{dsl}: python3 failed:\n{stderr}"));
            assert_eq!(actual.len(), expected.len(), "{dsl}");
            for (row, (a, e)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    e.to_bits(),
                    "{dsl}: row {row}: Python {a:?} vs Rust {e:?}"
                );
            }
            eprintln!(
                "E2 python {dsl}: {active} active members, {} rows bit-identical",
                test.len()
            );
        }
    }

    /// `emit_math_pipeline` against `MathPipeline::predict_cells`: raw cells through the f64 input
    /// contract (missing tokens, booleans, Rust's number grammar and whitespace set, clamping, a
    /// constant and an unselected column, a permuted selection), then the target inverse.
    #[test]
    fn math_pipeline_contract_matches_the_decoder() {
        use crate::math_pipeline::{InputContract, NumericColumn, PipelineState, TargetScale};
        use crate::math_pipeline::{INPUT_PART, PIPELINE_PART};
        use std::collections::BTreeMap;
        let column = |name: &str, min: f64, max: f64, constant: bool| NumericColumn {
            name: name.into(),
            min,
            max,
            constant,
        };
        let columns = vec![
            column("a", -2.0, 3.0, false),
            column("b", 0.0, 10.0, false),
            column("c", 1.0, 2.0, false),
            column("d", -1e3, 1e3, false),
            column("e", 0.0, 1.0, false),
            column("f", 5.0, 5.0, true),
            column("g", 0.0, 1.0, false),
        ];
        let decode = |dsl: &str, target: TargetScale| {
            let (x, y) = train();
            let model = fit(&GenomeIr::parse(dsl).unwrap(), &x, &y).unwrap();
            let input = InputContract::new(columns.clone(), vec![4, 0, 5, 2, 1], target).unwrap();
            let parts = BTreeMap::from([
                (
                    PIPELINE_PART.to_string(),
                    serde_json::to_vec(&PipelineState {
                        dsl: dsl.into(),
                        model,
                    })
                    .unwrap(),
                ),
                (INPUT_PART.to_string(), serde_json::to_vec(&input).unwrap()),
            ]);
            MathPipeline::decode(&parts, PIPELINE_PART).unwrap()
        };
        let cells: &[&str] = &[
            "0.5",
            "",
            "NA",
            " nan ",
            "?",
            "#N/A",
            "None",
            "null",
            "true",
            "Yes",
            "FALSE",
            "no",
            "1e3",
            ".5",
            "5.",
            "+2",
            "-0",
            "-0.0",
            "1_000",
            "\u{a0}0.25\u{3000}",
            "\u{1c}3",
            "\u{661}\u{662}",
            "inf",
            "-Infinity",
            "1e400",
            "1e-400",
            "0x10",
            "2.5e-1",
            "1e",
            "e5",
            ".",
            "+",
            "  -1.5  ",
            "\t7\n",
            "3.0\u{85}",
            "K",
            "1,5",
            "1.5.2",
            "TRUE",
            "nAn",
            "\u{feff}1",
            "-2",
            "3",
            "1.75",
            "-1000",
            "999.5",
            "0.999",
            "1e-3",
        ];
        let mut owned: Vec<Vec<Option<String>>> = Vec::new();
        for r in 0..cells.len() {
            owned.push(
                (0..7)
                    .map(|j| {
                        let k = (r * 7 + j * 3) % (cells.len() + 1);
                        cells.get(k).map(|c| c.to_string())
                    })
                    .collect(),
            );
        }
        let rows: Vec<Vec<Option<&str>>> = owned
            .iter()
            .map(|row| row.iter().map(|c| c.as_deref()).collect())
            .collect();
        let literal = render(&Py::List(
            owned
                .iter()
                .map(|row| {
                    Py::List(
                        row.iter()
                            .map(|c| c.clone().map_or(Py::None, Py::Str))
                            .collect(),
                    )
                })
                .collect(),
        ))
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        for dsl in [
            "(pipe X (standardize) (ridge))",
            "(pipe X (target kind=zscore) (hinge knots=3) (gbm rounds=10 depth=2))",
        ] {
            let pipeline = decode(
                dsl,
                TargetScale {
                    name: "y".into(),
                    min: -4.0,
                    scale: 2.5,
                },
            );
            let expected = pipeline.predict_cells(&rows).unwrap();
            let source = String::from_utf8(emit_math_pipeline(&pipeline).unwrap()).unwrap();
            let Some(actual) = run_python(
                dir.path(),
                &source,
                &format!("m[\"predict_rows\"]({literal})"),
            ) else {
                return;
            };
            let actual = actual.unwrap_or_else(|stderr| panic!("{dsl}: python3 failed:\n{stderr}"));
            assert_eq!(actual.len(), expected.len(), "{dsl}");
            for (row, (a, e)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    a.to_bits() == e.to_bits(),
                    "{dsl}: row {row} {:?}: Python {a:?} vs Rust {e:?}",
                    owned[row]
                );
            }
            // Numbers are accepted as cells and read as their value.
            let numbers = run_python(
                dir.path(),
                &source,
                "m[\"predict_rows\"]([[0.5, 1, None, 2.5, -1, 7, True]]) + m[\"predict_rows\"]([[\"0.5\", \"1\", None, \"2.5\", \"-1\", \"7\", \"true\"]])",
            )
            .unwrap()
            .unwrap();
            assert_eq!(numbers[0].to_bits(), numbers[1].to_bits(), "{dsl}");
            // A constant target scale predicts its minimum; an empty input predicts nothing.
            let flat = decode(
                dsl,
                TargetScale {
                    name: "y".into(),
                    min: 7.0,
                    scale: 0.0,
                },
            );
            let source = String::from_utf8(emit_math_pipeline(&flat).unwrap()).unwrap();
            let out = run_python(
                dir.path(),
                &source,
                "m[\"predict_rows\"]([[None] * 7]) + m[\"predict_rows\"]([])",
            )
            .unwrap()
            .unwrap();
            assert_eq!(out, flat.predict_cells(&[vec![None; 7]]).unwrap(), "{dsl}");
        }
        // Failures agree: a short row, and a target inverse that overflows f64.
        let pipeline = decode(
            "(pipe X (standardize) (ridge))",
            TargetScale {
                name: "y".into(),
                min: 1e308,
                scale: 1e308,
            },
        );
        let source = String::from_utf8(emit_math_pipeline(&pipeline).unwrap()).unwrap();
        let overflow = vec![Some("1e9"); 7];
        for (rust, call, python) in [
            (
                pipeline.predict_cells(std::slice::from_ref(&overflow)),
                "m[\"predict_rows\"]([[\"1e9\"] * 7])",
                "ArithmeticError: prediction undefined: row 0: non-finite target prediction",
            ),
            (
                pipeline.predict_cells(&[vec![Some("1"); 6]]),
                "m[\"predict_rows\"]([[\"1\"] * 6])",
                "ValueError: row 0 has 6 cells; the model reads 7",
            ),
        ] {
            let rust = format!("{:#}", rust.unwrap_err());
            assert!(
                rust.contains("non-finite target prediction") || rust.contains("row has 6 cells"),
                "{call}: {rust}"
            );
            let stderr = run_python(dir.path(), &source, call).unwrap().unwrap_err();
            assert!(stderr.contains(python), "{call}: {stderr}");
        }
    }

    #[test]
    fn fitted_python_refuses_a_model_from_another_plan() {
        let (x, y) = train();
        let fitted_on = GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap();
        let model = fit(&fitted_on, &x, &y).unwrap();
        for (other, message) in [
            (
                "(pipe X (robust_scale) (ridge))",
                "fitted standardize where the plan has robust_scale",
            ),
            (
                "(pipe X (standardize) (rank) (ridge))",
                "1 fitted stages for 2 planned stages",
            ),
            (
                "(pipe X (standardize) (elasticnet))",
                "fitted head ridge does not match the planned head elasticnet",
            ),
            (
                "(pipe X (target kind=zscore) (standardize) (ridge))",
                "no fitted target transform where the plan has one",
            ),
        ] {
            let error = emit(&GenomeIr::parse(other).unwrap(), &model, WIDTH).unwrap_err();
            assert!(format!("{error:#}").contains(message), "{other}: {error:#}");
        }
        let target = fit(
            &GenomeIr::parse("(pipe X (target kind=zscore) (ridge))").unwrap(),
            &x,
            &y,
        )
        .unwrap();
        let error = emit(
            &GenomeIr::parse("(pipe X (target kind=winsor) (ridge))").unwrap(),
            &target,
            WIDTH,
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("target transform kind differs"),
            "{error:#}"
        );
    }

    #[test]
    fn fitted_python_refuses_contradictory_prediction_parameters() {
        let (x, y) = train();
        let cases = [
            (
                "(pipe X (standardize append=true) (ridge))",
                "(pipe X (standardize) (ridge))",
                "append",
            ),
            (
                "(pipe X (step t=0.1) (ridge))",
                "(pipe X (step t=0.9) (ridge))",
                "t",
            ),
            (
                "(pipe X (cheb degree=3) (ridge))",
                "(pipe X (cheb degree=4) (ridge))",
                "degree",
            ),
            (
                "(pipe X (quantile_bins bins=4) (ridge))",
                "(pipe X (quantile_bins bins=8) (ridge))",
                "bins",
            ),
            (
                "(pipe X (bin_onehot bins=3) (ridge))",
                "(pipe X (bin_onehot bins=4) (ridge))",
                "bins",
            ),
            (
                "(pipe X (landmark_dist k=4 gamma=0.25) (ridge))",
                "(pipe X (landmark_dist k=4 gamma=1) (ridge))",
                "gamma",
            ),
            (
                "(pipe X (nystrom k=4 gamma=0.25) (ridge))",
                "(pipe X (nystrom k=4 gamma=1) (ridge))",
                "gamma",
            ),
            (
                "(pipe X (symlog c=0.5) (ridge))",
                "(pipe X (symlog c=2) (ridge))",
                "c",
            ),
            (
                "(pipe X (lag k=1) (ridge))",
                "(pipe X (lag k=2) (ridge))",
                "k",
            ),
            (
                "(pipe X (roll k=1) (ridge))",
                "(pipe X (roll k=2) (ridge))",
                "k",
            ),
            (
                "(pipe X (map expr=\"col(0)\") (ridge))",
                "(pipe X (map expr=\"sigmoid(col(0))\") (ridge))",
                "expr",
            ),
            (
                "(pipe X (ksmooth k=8))",
                "(pipe X (ksmooth k=16))",
                "head k",
            ),
            (
                "(pipe X (ksmooth gamma=0.5))",
                "(pipe X (ksmooth gamma=2))",
                "head gamma",
            ),
            (
                "(pipe X (gpr kernel=rq+linear m=50 steps=2))",
                "(pipe X (gpr kernel=rbf+linear m=50 steps=2))",
                "head kernel",
            ),
        ];
        for (fitted, declared, parameter) in cases {
            let model = fit(&GenomeIr::parse(fitted).unwrap(), &x, &y).unwrap();
            let error = emit(&GenomeIr::parse(declared).unwrap(), &model, WIDTH).unwrap_err();
            assert!(
                format!("{error:#}").contains(&format!("{parameter} parameter differs")),
                "{fitted} / {declared}: {error:#}"
            );
        }
        let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
        let model = fit(&genome, &x, &y).unwrap();
        assert!(
            emit(&genome, &model, WIDTH - 1).is_err(),
            "wrong input width accepted"
        );
        assert!(
            emit(&genome, &model, WIDTH).is_ok(),
            "zero-stage matching model refused"
        );
    }

    #[test]
    fn fitted_python_refuses_non_finite_state_symbolic_models_and_f32_contracts() {
        let (x, y) = train();
        let genome = GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap();
        let mut model = fit(&genome, &x, &y).unwrap();
        if let Model::Pipeline { ops, .. } = &mut model {
            if let FittedOp::Standardize { means, .. } = &mut ops[0] {
                means[1] = f64::NAN;
            }
        }
        let error = emit(&genome, &model, WIDTH).unwrap_err();
        assert!(
            format!("{error:#}").contains("non-finite fitted state"),
            "{error:#}"
        );

        let symbolic = Model::Symbolic {
            expr: feat_math::math_expr::MathExpr::parse("col(0)").unwrap(),
            a: 0.0,
            b: 1.0,
        };
        let error = emit(&genome, &symbolic, WIDTH).unwrap_err();
        assert!(
            format!("{error:#}").contains("needs a pipeline model"),
            "{error:#}"
        );

        let input = include_bytes!("../tests/fixtures/math-pipeline/input.json");
        let state = include_bytes!("../tests/fixtures/math-pipeline/pipeline.json");
        let parts = std::collections::BTreeMap::from([
            (crate::math_pipeline::INPUT_PART.into(), input.to_vec()),
            (crate::math_pipeline::PIPELINE_PART.into(), state.to_vec()),
        ]);
        let legacy = MathPipeline::decode(&parts, crate::math_pipeline::PIPELINE_PART).unwrap();
        let error = emit_math_pipeline(&legacy).unwrap_err();
        assert!(
            format!("{error:#}").contains("prediction_version 1"),
            "{error:#}"
        );
    }

    /// Spec §10.5 determinism: the same model and emitter produce byte-identical source.
    #[test]
    fn fitted_python_is_byte_identical_for_the_same_model() {
        let (x, y) = train();
        for dsl in [
            "(pipe X (rank) (pair_prod topk=4 append=true) (gbm rounds=5))",
            "(pipe X (portfolio))",
        ] {
            let genome = GenomeIr::parse(dsl).unwrap();
            let first = emit(&genome, &fit(&genome, &x, &y).unwrap(), WIDTH).unwrap();
            let again = emit(&genome, &fit(&genome, &x, &y).unwrap(), WIDTH).unwrap();
            assert_eq!(
                crate::sha256_hex(first.as_bytes()),
                crate::sha256_hex(again.as_bytes()),
                "{dsl}"
            );
        }
    }
}

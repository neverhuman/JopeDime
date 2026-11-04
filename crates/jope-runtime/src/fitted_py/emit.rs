use super::head::{head_state, target_state};
use super::render::{render, Py};
use super::stage::stage_state;
use crate::calibration::OutputTransform;
use crate::error::Failure;
use crate::fitted_js::ExportCalibration;
use crate::math_pipeline::{MathPipeline, F64_PREDICTION_VERSION};
use anyhow::Result;
use feat_math::genome::GenomeIr;
use feat_math::model::Model;
use feat_math::spec::{build_plan, Plan};
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

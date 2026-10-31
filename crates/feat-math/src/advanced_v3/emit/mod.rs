use super::calibration::CalibrationModel;
use super::heads::{ExtraNode, V3Head};
use super::input::{CategoricalMode, FittedColumn, FittedInput};
use super::pipeline::{BoundModel, BoundPredictor};
use super::transforms::{GroupTail, SemanticKey, V3Op};
use super::types::{text_hash, TaskKind, TaskSpec, ARTIFACT_SCHEMA};
use crate::heads::FittedHead;
use crate::ops::FittedOp;
use crate::target_transform::FittedTarget;
use anyhow::{bail, Result};
use serde::Serialize;

// ---------------------------- bound artifacts -----------------------------

#[derive(Clone, Debug)]
pub struct BoundEmitted {
    pub python: String,
    pub rust: String,
    pub manifest: String,
    pub py_tokens: crate::transpile::TokenCount,
    pub rs_tokens: crate::transpile::TokenCount,
    pub timing: crate::transpile::EmitTiming,
}

/// The bound artifact's sidecar. The JSON key is `program_sha256`; the
/// `BoundModel` field it comes from keeps its Rust name because the fitted
/// model codec persists it.
#[derive(Serialize)]
struct BoundManifest<'a> {
    schema: &'a str,
    task: &'a TaskSpec,
    task_sha256: &'a str,
    schema_sha256: &'a str,
    vocabulary_sha256: &'a str,
    program_sha256: &'a str,
    emitter_version: &'a str,
    python_sha256: String,
    rust_sha256: String,
}

/// The strings a bound or fitted artifact carries that come from the caller's
/// data rather than from the emitter: the column names and every categorical
/// vocabulary entry. The shipped-text lint runs with them masked
/// ([`lint_shipped_masked`]), so a column called `island` or a category named
/// `population` cannot fail emission, while everything the emitter itself
/// writes is still checked.
///
/// Only strings that would trip the lint on their own are masked (an innocent
/// name never touches emitter text), and only in the spellings the artifacts
/// embed them in: raw, JSON-escaped (the C++ codec), Rust `Debug`-escaped (the
/// Rust artifact's string literals) and the JSON quotation of the JSON form (the Python
/// artifact embeds its JSON literals through valid JSON string quotation). Entries shorter than three
/// bytes cannot contain a banned token and are skipped. Longest first, so a name
/// that contains another is masked whole. One edge fails closed: a name that
/// does not trip raw but whose escaped spelling completes a token (a leading
/// newline followed by `nls`, say) is not masked, and the artifact is refused
/// rather than shipped.
pub(crate) fn user_text(model: &BoundModel) -> Vec<String> {
    let mut raw: Vec<String> = model
        .schema
        .columns
        .iter()
        .map(|column| column.name.clone())
        .collect();
    for column in &model.input.columns {
        if let FittedColumn::Categorical { vocabulary, .. } = column {
            raw.extend(vocabulary.iter().cloned());
        }
    }
    let trips = |text: &str| {
        !crate::transpile::lint_vocabulary_portfolio(text).is_empty()
            || !crate::transpile::lint_shipped_phrases(text).is_empty()
    };
    let mut forms = Vec::new();
    for text in raw {
        if text.len() < 3 || !trips(&text) {
            continue;
        }
        let json = serde_json::to_string(&text).unwrap_or_default();
        let json_inner = json.get(1..json.len().saturating_sub(1)).unwrap_or("");
        let debug = format!("{text:?}");
        let debug_inner = &debug[1..debug.len() - 1];
        let debug_of_json = serde_json::to_string(json_inner).unwrap_or_default();
        let debug_of_json_inner = &debug_of_json[1..debug_of_json.len() - 1];
        for form in [json_inner, debug_inner, debug_of_json_inner] {
            if !form.is_empty() && form != text {
                forms.push(form.to_owned());
            }
        }
        forms.push(text);
    }
    forms.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    forms.dedup();
    forms
}

/// The shipped-text lint over `text` with every user-supplied string
/// ([`user_text`]) replaced by underscores of the same length. Portfolio heads
/// get the frozen emitter's stricter portfolio rule (`oof` is banned too).
pub(crate) fn lint_shipped_masked(text: &str, user: &[String], portfolio: bool) -> Vec<String> {
    let mut masked = text.to_owned();
    for entry in user {
        if masked.contains(entry.as_str()) {
            masked = masked.replace(entry.as_str(), &"_".repeat(entry.len()));
        }
    }
    let mut hits = if portfolio {
        crate::transpile::lint_vocabulary_portfolio(&masked)
    } else {
        crate::transpile::lint_vocabulary(&masked)
    };
    hits.extend(crate::transpile::lint_shipped_phrases(&masked));
    hits
}

/// Emit a fitted, raw-CSV prediction program. This is deliberately separate
/// from `transpile::emit`: the latter remains the frozen artifact-v1 training
/// program (pipeline artifacts are byte-identical across emitter versions
/// 2.1 and 2.2; see `EMITTER_VERSION`). Every artifact is linted with the
/// caller's column names and categorical vocabularies masked before
/// `lint_shipped_masked` runs.
pub fn emit_bound(
    model: &BoundModel,
    options: &crate::transpile::EmitOptions,
) -> Result<BoundEmitted> {
    crate::transpile::validate_math_kernel(options)?;
    model.validate_binding()?;
    let BoundPredictor::Pipeline { ops, head, .. } = &model.predictor else {
        bail!("bound emission currently requires a v3 pipeline model");
    };
    for op in ops {
        op.validate_fitted_state(&model.schema)?;
    }
    let portfolio = matches!(head, V3Head::ClassificationPortfolio(_));
    let user = user_text(model);
    let start = std::time::Instant::now();
    let python = emit_bound_python(model, options.partition_point_quantile_bounds)?;
    let rust = emit_bound_rust(model)?;
    let elapsed = start.elapsed().as_nanos().max(1);
    for (lang, text) in [("Python", &python), ("Rust", &rust)] {
        let hits = lint_shipped_masked(text, &user, portfolio);
        if !hits.is_empty() {
            bail!(
                "banned vocabulary in bound {lang} artifact: {}",
                hits.join(", ")
            );
        }
    }
    let py_tokens = crate::tokens::count_tokens(&python, crate::tokens::Lang::Python);
    let rs_tokens = crate::tokens::count_tokens(&rust, crate::tokens::Lang::Rust);
    let py_total = py_tokens.code_tokens + py_tokens.doc_tokens;
    let rs_total = rs_tokens.code_tokens + rs_tokens.doc_tokens;
    let emitter_version = if options.partition_point_quantile_bounds {
        "feat-math-quantile-partition-point-v1"
    } else {
        crate::transpile::EMITTER_VERSION
    };
    let manifest = serde_json::to_string(&BoundManifest {
        schema: ARTIFACT_SCHEMA,
        task: &model.task,
        task_sha256: &model.task_sha256,
        schema_sha256: &model.schema_sha256,
        vocabulary_sha256: &model.vocabulary_sha256,
        program_sha256: &model.genome_sha256,
        emitter_version,
        python_sha256: text_hash(&python),
        rust_sha256: text_hash(&rust),
    })?;
    let hits = lint_shipped_masked(&manifest, &user, portfolio);
    if !hits.is_empty() {
        bail!(
            "banned vocabulary in bound artifact manifest: {}",
            hits.join(", ")
        );
    }
    Ok(BoundEmitted {
        python,
        rust,
        manifest,
        py_tokens,
        rs_tokens,
        timing: crate::transpile::EmitTiming {
            emit_ns: elapsed,
            py_tokens: py_total,
            rs_tokens: rs_total,
            tokens_per_sec: (py_total + rs_total) as f64 / (elapsed as f64 / 1e9),
        },
    })
}

/// Fast-path Rust-only bound emitter for `jope-micro`: emits and lints the
/// standalone Rust prediction program (`bound_model.rs`) without generating or
/// linting the unused Python twin or JSON manifest.
///
/// Contract note: because Python emission, Python vocabulary linting, and
/// manifest construction are intentionally skipped (`python` and `manifest` are
/// returned empty), this function validates only the bound Rust artifact and
/// can succeed on models where [`emit_bound`] would fail closed in the Python
/// emitter or Python linter.
pub fn emit_bound_rust_only(
    model: &BoundModel,
    options: &crate::transpile::EmitOptions,
) -> Result<BoundEmitted> {
    crate::transpile::validate_math_kernel(options)?;
    model.validate_binding()?;
    let BoundPredictor::Pipeline { ops, head, .. } = &model.predictor else {
        bail!("bound emission currently requires a v3 pipeline model");
    };
    for op in ops {
        op.validate_fitted_state(&model.schema)?;
    }
    let portfolio = matches!(head, V3Head::ClassificationPortfolio(_));
    let user = user_text(model);
    let start = std::time::Instant::now();
    let rust = emit_bound_rust(model)?;
    let elapsed = start.elapsed().as_nanos().max(1);
    let hits = lint_shipped_masked(&rust, &user, portfolio);
    if !hits.is_empty() {
        bail!(
            "banned vocabulary in bound Rust artifact: {}",
            hits.join(", ")
        );
    }
    let py_tokens = crate::transpile::TokenCount {
        code_tokens: 0,
        doc_tokens: 0,
        bytes: 0,
    };
    let rs_tokens = crate::tokens::count_tokens(&rust, crate::tokens::Lang::Rust);
    let rs_total = rs_tokens.code_tokens + rs_tokens.doc_tokens;
    Ok(BoundEmitted {
        python: String::new(),
        rust,
        manifest: String::new(),
        py_tokens,
        rs_tokens,
        timing: crate::transpile::EmitTiming {
            emit_ns: elapsed,
            py_tokens: 0,
            rs_tokens: rs_total,
            tokens_per_sec: rs_total as f64 / (elapsed as f64 / 1e9),
        },
    })
}

fn emit_bound_python(model: &BoundModel, partition_point_quantile_bounds: bool) -> Result<String> {
    let BoundPredictor::Pipeline {
        target,
        ops,
        head,
        calibration,
        missing_aware,
    } = &model.predictor
    else {
        bail!("bound emission currently requires a v3 pipeline model");
    };
    let headers = serde_json::to_string(
        &model
            .schema
            .columns
            .iter()
            .map(|column| &column.name)
            .collect::<Vec<_>>(),
    )?;
    let input = serde_json::to_string(&model.input)?;
    let head_json = serde_json::to_string(head)?;
    let ops_json = serde_json::to_string(ops)?;
    let calibration_json = serde_json::to_string(calibration)?;
    let target_json = serde_json::to_string(target)?;
    let headers_literal = serde_json::to_string(&headers)?;
    let input_literal = serde_json::to_string(&input)?;
    let head_json_literal = serde_json::to_string(&head_json)?;
    let ops_json_literal = serde_json::to_string(&ops_json)?;
    let calibration_json_literal = serde_json::to_string(&calibration_json)?;
    let target_json_literal = serde_json::to_string(&target_json)?;
    let kind = match head {
        V3Head::LogitRidge(_) => "logit",
        V3Head::TabBoost(_) => "tab_boost",
        V3Head::ExtraTrees(_) => "extra_trees",
        V3Head::GamBoost(_) => "gam_boost",
        V3Head::Pls(_) => "pls",
        V3Head::ClassificationPortfolio(_) => "classification_portfolio",
        V3Head::MulticlassRidge(_)
        | V3Head::MulticlassOvrLogistic(_)
        | V3Head::MulticlassDiagonalLda(_)
        | V3Head::MulticlassNearestCentroid(_) => {
            bail!("multiclass probability matrices are not supported by scalar bound emission")
        }
        V3Head::Legacy(FittedHead::Ridge(_)) => "ridge",
        V3Head::Legacy(_) => bail!("this fitted legacy head is not supported by bound emission"),
    };
    let (
        lower_bound_helper,
        tbe_index_expr,
        gam_index_expr,
        isotonic_index_expr,
        rankgauss_hi_expr,
    ) = if partition_point_quantile_bounds {
        (
            "\ndef lower_bound(sorted_values, v):\n    size = len(sorted_values)\n    if size == 0:\n        return 0\n    base = 0\n    while size > 1:\n        half = size // 2\n        mid = base + half\n        if sorted_values[mid] < v:\n            base = mid\n        size -= half\n    return base + (1 if sorted_values[base] < v else 0)\n",
            "lower_bound(edges, value)",
            "lower_bound(term[\"edges\"], item)",
            "lower_bound(spec[\"upper\"], value)",
            "lower_bound(scores, value)",
        )
    } else {
        (
            "",
            "sum(edge < value for edge in edges)",
            "sum(edge < item for edge in term[\"edges\"])",
            "sum(bound < value for bound in spec[\"upper\"])",
            "sum(score < value for score in scores)",
        )
    };
    Ok(format!(
        r#"#!/usr/bin/env python3
"""Schema-bound, dependency-free prediction artifact."""
import csv
import json
import math
import struct
import sys

HEADERS = json.loads({headers_literal})
INPUT = json.loads({input_literal})
OPS = json.loads({ops_json_literal})
HEAD = json.loads({head_json_literal})
CALIBRATION = json.loads({calibration_json_literal})
TARGET = json.loads({target_json_literal})
HEAD_KIND = {kind:?}
MISSING_AWARE = {missing_aware}

# O(1) categorical lookup (list.index would be O(vocab) per cell). Pure index
# bookkeeping — no float semantics — and json.loads round-trips the
# missing sentinel into the identical dict key.
LOOKUP = []
for state in INPUT["columns"]:
    if "Categorical" in state:
        table = dict()
        for position, value in enumerate(state["Categorical"]["vocabulary"]):
            table[value] = position
        LOOKUP.append(table)
    else:
        LOOKUP.append(None)

def missing(text):
    return text.strip().lower() in ("", "na", "nan", "null")

def category(text):
    return "\0MISSING" if text.strip().lower() in ("", "na", "null") else text

def clamp(value, lo, hi):
    # Branch form mirrors Rust f64::clamp exactly: NaN passes through
    # (min/max chains would pin NaN to a bound and silently diverge).
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value

def encode(row):
    out = []
    masks = []
    prior = INPUT["config"]["prior"]
    for j, state in enumerate(INPUT["columns"]):
        raw = row[j]
        if "Numeric" in state:
            spec = state["Numeric"]
            absent = missing(raw)
            if absent:
                value = spec["fill"]
            else:
                try:
                    value = float(raw)
                except ValueError:
                    raise SystemExit("invalid numeric value " + repr(raw))
                if not math.isfinite(value):
                    absent = True
                    value = spec["fill"]
            out.append(float("nan") if absent and MISSING_AWARE else (value if math.isfinite(value) else spec["fill"]))
            masks.append(absent)
            if spec["indicator"]:
                out.append(1.0 if absent else 0.0)
                masks.append(False)
            continue
        spec = state["Categorical"]
        key = category(raw)
        index = LOOKUP[j].get(key)
        mode = spec["mode"]
        if mode == "Native":
            out.append(-1.0 if index is None else float(index))
            masks.append(key == "\0MISSING")
        elif mode == "Frequency":
            out.append(0.0 if index is None else spec["counts"][index] / spec["total"])
            masks.append(key == "\0MISSING")
        elif mode == "Onehot":
            out.extend(1.0 if index == item else 0.0 for item in spec["onehot"])
            masks.extend(False for _ in spec["onehot"])
        else:
            ordered = spec["global"] if index is None else (
                spec["sums"][index] + prior * spec["global"]
            ) / (spec["counts"][index] + prior)
            if mode == "Ordered":
                out.append(ordered)
                masks.append(key == "\0MISSING")
            else:
                out.append(0.0 if index is None else spec["counts"][index] / spec["total"])
                masks.append(key == "\0MISSING")
                out.append(ordered)
                masks.append(False)
    return out, masks

def yj_pow(base, exponent):
    # Mirrors Rust f64::powf on overflow: the base here is always >= 1.0, so
    # the only Python-side surprise is OverflowError where Rust returns inf.
    try:
        return base ** exponent
    except OverflowError:
        return float("inf")

def yj(value, lam):
    if value >= 0.0:
        return math.log1p(value) if abs(lam) < 1e-12 else (yj_pow(value + 1.0, lam) - 1.0) / lam
    if abs(lam - 2.0) < 1e-12:
        return -math.log1p(-value)
    return -(yj_pow(1.0 - value, 2.0 - lam) - 1.0) / (2.0 - lam)

def category_missing(text):
    return text.strip().lower() in ("", "na", "null")

def semantic_key(text, kind):
    if kind == "numeric":
        stripped = text.strip()
        if stripped.lower() in ("", "na", "nan", "null"):
            return (0, None)
        try:
            value = float(stripped)
        except ValueError:
            raise SystemExit("invalid numeric value " + repr(text))
        return (1, 0.0 if value == 0.0 else value) if math.isfinite(value) else (0, None)
    return (0, None) if category_missing(text) else (2, text)

def stored_key(items):
    result = []
    for item in items:
        if item == "Missing":
            result.append((0, None))
        elif "Numeric" in item:
            result.append((1, struct.unpack("!d", int(item["Numeric"]).to_bytes(8, "big"))[0]))
        else:
            result.append((2, item["Categorical"]))
    return tuple(result)

def raw_keys(row, order):
    group = tuple(semantic_key(row[index], kind) for index, kind in order["group_columns"])
    ordered = tuple(semantic_key(row[index], kind) for index, kind in order["order_columns"])
    return group, ordered
{lower_bound_helper}
def apply_ops(rows, masks, raw_rows):
    for wrapped in OPS:
        name, state = next(iter(wrapped.items()))
        if name == "Legacy":
            name, state = next(iter(state.items()))
        append = state["append"]
        if name in ("Lag", "Roll"):
            width = len(rows[0]) if rows else 0
            generated = [[0.0] * width for _ in rows]
            generated_masks = [[True] * width for _ in rows]
            histories = dict()
            fitted_cutoffs = dict()
            for tail in state["tails"]:
                key = stored_key(tail["key"])
                histories[key] = [
                    (item["values"][:], item["missing"][:]) for item in tail["rows"]
                ]
                fitted_cutoffs[key] = stored_key(tail["last_order"])
            keys = [raw_keys(row, state["order"]) for row in raw_rows]
            for group, ordered in keys:
                if group in fitted_cutoffs and ordered <= fitted_cutoffs[group]:
                    raise ValueError("prediction row order must be strictly after fitted order")
            chronological = sorted(range(len(rows)), key=lambda index: (keys[index][0], keys[index][1], index))
            keep = state["k"] if name == "Lag" else max(0, state["k"] - 1)
            for index in chronological:
                history = histories.setdefault(keys[index][0], [])
                if name == "Lag":
                    if len(history) >= state["k"]:
                        generated[index] = history[-state["k"]][0][:]
                        generated_masks[index] = history[-state["k"]][1][:]
                else:
                    for feature in range(width):
                        total = 0.0
                        count = 0
                        for item in history[-max(0, state["k"]-1):]:
                            if not item[1][feature] and math.isfinite(item[0][feature]):
                                total += item[0][feature]
                                count += 1
                        if not masks[index][feature] and math.isfinite(rows[index][feature]):
                            total += rows[index][feature]
                            count += 1
                        if count:
                            generated[index][feature] = total / count
                            generated_masks[index][feature] = False
                history.append((rows[index][:], masks[index][:]))
                if len(history) > keep:
                    del history[:len(history)-keep]
        else:
            generated = []
            generated_masks = []
            for x, mask in zip(rows, masks):
                if name == "MaxAbs":
                    out = [value / scale for value, scale in zip(x, state["scales"])]
                    out_mask = mask[:]
                elif name == "YeoJohnson":
                    out = [yj(value, lam) for value, lam in zip(x, state["lambdas"])]
                    out_mask = mask[:]
                elif name == "CorrScreen" or name == "SelectCorr":
                    out = [x[j] for j in state["indices"]]
                    out_mask = [mask[j] for j in state["indices"]]
                elif name == "PiecewiseLinear":
                    out, out_mask = [], []
                    for value, absent, edges in zip(x, mask, state["edges"]):
                        for lo, hi in zip(edges, edges[1:]):
                            out.append(float(value > lo) if abs(hi-lo) <= 1e-15 else clamp((value-lo)/(hi-lo), 0.0, 1.0))
                            out_mask.append(absent)
                elif name == "PlsProject":
                    out, out_mask = [], [False] * len(state["weights"])
                    for weight in state["weights"]:
                        total = 0.0
                        for w, item, mean in zip(weight, x, state["means"]):
                            total += w * (item - mean)
                        out.append(total)
                elif name == "Standardize":
                    out = [clamp((value-mean)/scale, -20.0, 20.0) for value, mean, scale in zip(x, state["means"], state["scales"])]
                    out_mask = mask[:]
                elif name == "RobustScale":
                    out = [(value-median)/scale for value, median, scale in zip(x, state["medians"], state["scales"])]
                    out_mask = mask[:]
                elif name == "TargetBinEncode":
                    out = []
                    for value, edges, encoded in zip(x, state["edges"], state["enc"]):
                        index = {tbe_index_expr}
                        out.append(encoded[min(index, len(encoded)-1)])
                    out_mask = mask[:]
                elif name == "Sigmoid":
                    out = [1.0 / (1.0 + math.exp(-clamp(value, -30.0, 30.0))) for value in x]
                    out_mask = mask[:]
                elif name == "Step":
                    out = [1.0 if value > state["t"] else 0.0 for value in x]
                    out_mask = mask[:]
                elif name == "NormRow":
                    total = sum(item * item for item in x)
                    inv = 1.0 / max(1e-9, math.sqrt(total))
                    out = [value*inv for value in x]
                    out_mask = mask[:]
                elif name == "WinsorClip":
                    out = [clamp(value, lo, hi) for value, lo, hi in zip(x, state["lo"], state["hi"])]
                    out_mask = mask[:]
                elif name == "Impute":
                    out = [fill if value == 0.0 else value for value, fill in zip(x, state["fill"])]
                    out.extend(1.0 if x[j] == 0.0 else 0.0 for j in state["indicator_cols"])
                    out_mask = [False] * len(out)
                else:
                    raise ValueError("unsupported operation " + name)
                generated.append(out)
                generated_masks.append(out_mask)
        if append:
            rows = [original + added for original, added in zip(rows, generated)]
            masks = [original + added for original, added in zip(masks, generated_masks)]
        else:
            rows, masks = generated, generated_masks
        rows = [[0.0 if absent or not math.isfinite(value) else value for value, absent in zip(row, mask)]
                for row, mask in zip(rows, masks)]
    return rows

def sigmoid(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(value, 40.0)))
    z = math.exp(max(value, -40.0))
    return z / (1.0 + z)

def ridge(model, x, probability):
    linear = model["weights"][0] if probability else model["w"][0]
    weights = model["weights"] if probability else model["w"]
    for j, (mean, scale) in enumerate(zip(model["means"], model["scales"])):
        linear += weights[j+1] * clamp((x[j]-mean)/scale, -30.0 if probability else -20.0, 30.0 if probability else 20.0)
    return sigmoid(linear) if probability else clamp(linear, -60.0, 60.0)

def tree(node, x):
    if "Leaf" in node:
        return node["Leaf"]
    split = node["Split"]
    value = x[split["feature"]]
    left = value <= split["threshold"] if math.isfinite(value) else split["missing_left"]
    return tree(split["left"] if left else split["right"], x)

def predict_head(wrapped, x):
    tag, model = next(iter(wrapped.items()))
    if tag == "LogitRidge":
        value = ridge(model, x, True)
    elif tag == "Legacy":
        value = ridge(model["Ridge"], x, False)
    elif tag == "TabBoost":
        raw = model["base"]
        for stump in model["stumps"]:
            item = x[stump["feature"]]
            left = item <= stump["threshold"] if math.isfinite(item) else stump["missing_left"]
            raw += model["learning_rate"] * (stump["left"] if left else stump["right"])
        value = sigmoid(raw) if model["binary"] else (raw if math.isfinite(raw) else 0.0)
    elif tag == "ExtraTrees":
        total = 0.0
        for node in model["trees"]:
            total += tree(node, x)
        value = total / max(1, len(model["trees"]))
        value = clamp(value, 0.0, 1.0) if model["binary"] else (value if math.isfinite(value) else 0.0)
    elif tag == "GamBoost":
        raw = model["base"]
        for term in model["terms"]:
            item = x[term["feature"]]
            index = {gam_index_expr}
            raw += model["learning_rate"] * term["values"][min(index, len(term["values"])-1)]
        value = sigmoid(raw) if model["binary"] else (raw if math.isfinite(raw) else 0.0)
    elif tag == "ClassificationPortfolio":
        value = 0.0
        for weight, member in zip(model["weights"], model["members"]):
            value += weight * predict_head(member, x)
        value = clamp(value, 0.0, 1.0)
    elif tag == "Pls":
        projection = model["projection"]["PlsProject"]
        projected = []
        for weight in projection["weights"]:
            total = 0.0
            for w, item, mean in zip(weight, x, projection["means"]):
                total += w * (item - mean)
            projected.append(total)
        value = ridge(model["ridge"], projected, False)
    else:
        raise ValueError("unsupported fitted head " + tag)
    return value

def predict_numeric(x):
    value = predict_head(HEAD, x)
    cal = CALIBRATION
    if "Platt" in cal:
        spec = cal["Platt"]
        p = clamp(value, 1e-12, 1.0-1e-12)
        value = sigmoid(spec["slope"] * math.log(p/(1.0-p)) + spec["intercept"])
    elif "Isotonic" in cal:
        spec = cal["Isotonic"]
        index = {isotonic_index_expr}
        value = spec["values"][min(index, len(spec["values"])-1)]
    if TARGET is not None:
        if TARGET == "Log1p":
            value = math.exp(clamp(value, -745.0, 700.0)) - 1.0
            value = clamp(value, -60.0, 60.0)
        elif "Winsor" in TARGET:
            value = clamp(value, -60.0, 60.0)
        elif "ZScore" in TARGET:
            spec = TARGET["ZScore"]
            value = clamp(value * spec["sigma"] + spec["mu"], spec["clamp_lo"], spec["clamp_hi"])
        else:
            spec = TARGET["RankGauss"]
            scores, values = spec["table_scores"], spec["table_values"]
            if not (value > scores[0]):
                # NaN routes to the low end, mirroring the reference inverse.
                value = values[0]
            elif value >= scores[-1]:
                value = values[-1]
            else:
                hi = {rankgauss_hi_expr}
                lo = hi - 1
                value = values[lo] + (value-scores[lo])/(scores[hi]-scores[lo])*(values[hi]-values[lo])
            value = clamp(value, -60.0, 60.0)
    if {binary}:
        return clamp(value, 0.0, 1.0) if math.isfinite(value) else 0.5
    return value if math.isfinite(value) else 0.0

def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: model.py INPUT.csv [OUTPUT.csv]")
    with open(sys.argv[1], newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    if not rows or rows[0] != HEADERS:
        raise SystemExit("CSV header does not match bound schema")
    encoded = [encode(row) for row in rows[1:]]
    features = apply_ops([item[0] for item in encoded], [item[1] for item in encoded], rows[1:])
    predictions = [predict_numeric(row) for row in features]
    text = "".join(format(value, ".17g") + "\n" for value in predictions)
    if len(sys.argv) == 3:
        with open(sys.argv[2], "w", encoding="utf-8", newline="") as target:
            target.write(text)
    else:
        sys.stdout.write(text)

if __name__ == "__main__":
    main()
"#,
        binary = if model.task.kind == TaskKind::Binary {
            "True"
        } else {
            "False"
        },
        missing_aware = if *missing_aware { "True" } else { "False" },
    ))
}

fn rust_f64(value: f64) -> String {
    format!("{value:.17e}")
}

fn rust_f64_slice(values: &[f64]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| rust_f64(*value))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn rust_usize_slice(values: &[usize]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn rust_string(value: &str) -> String {
    format!("{value:?}")
}

fn rust_input_body(input: &FittedInput, missing_aware: bool) -> String {
    let mut source = String::from(
        "    let mut x: Vec<f64> = Vec::new();\n    let mut missing_mask: Vec<bool> = Vec::new();\n",
    );
    for (j, column) in input.columns.iter().enumerate() {
        match column {
            FittedColumn::Numeric { fill, indicator } => {
                source.push_str(&format!(
                    "    let raw_missing_{j} = is_missing(&row[{j}]);\n    let parsed_{j} = if raw_missing_{j} {{ {fill} }} else {{ row[{j}].trim().parse::<f64>().map_err(|_| format!(\"invalid numeric value {{}}\", row[{j}]))? }};\n    let absent_{j} = raw_missing_{j} || !parsed_{j}.is_finite();\n    let value_{j} = if parsed_{j}.is_finite() {{ parsed_{j} }} else {{ {fill} }};\n    x.push(if absent_{j} && {missing_aware} {{ f64::NAN }} else {{ value_{j} }});\n    missing_mask.push(absent_{j});\n",
                    fill = rust_f64(*fill)
                ));
                if *indicator {
                    source.push_str(&format!(
                        "    x.push(if absent_{j} {{ 1.0 }} else {{ 0.0 }});\n    missing_mask.push(false);\n"
                    ));
                }
            }
            FittedColumn::Categorical {
                mode,
                vocabulary,
                counts,
                sums,
                total,
                global,
                onehot,
            } => {
                let vocab = vocabulary
                    .iter()
                    .map(|value| rust_string(value))
                    .collect::<Vec<_>>()
                    .join(",");
                source.push_str(&format!(
                    "    let absent_{j} = is_category_missing(&row[{j}]);\n    let key_{j} = category(&row[{j}]);\n    let vocabulary_{j}: &[&str] = &[{vocab}];\n    let index_{j} = vocabulary_{j}.binary_search(&key_{j}.as_str()).ok();\n"
                ));
                match mode {
                    CategoricalMode::Native => source.push_str(&format!(
                        "    x.push(index_{j}.map_or(-1.0, |index| index as f64));\n    missing_mask.push(absent_{j});\n"
                    )),
                    CategoricalMode::Frequency => source.push_str(&format!(
                        "    let counts_{j}: &[usize] = &{counts};\n    x.push(index_{j}.map_or(0.0, |index| counts_{j}[index] as f64 / {total}.0));\n    missing_mask.push(absent_{j});\n",
                        counts = rust_usize_slice(counts)
                    )),
                    CategoricalMode::Onehot => {
                        for category in onehot {
                            source.push_str(&format!(
                                "    x.push(if index_{j} == Some({category}) {{ 1.0 }} else {{ 0.0 }});\n    missing_mask.push(false);\n"
                            ));
                        }
                    }
                    CategoricalMode::Ordered | CategoricalMode::Hybrid => {
                        source.push_str(&format!(
                            "    let counts_{j}: &[usize] = &{counts};\n    let sums_{j}: &[f64] = &{sums};\n    let ordered_{j} = index_{j}.map_or({global}, |index| (sums_{j}[index] + {prior} * {global}) / (counts_{j}[index] as f64 + {prior}));\n",
                            counts = rust_usize_slice(counts),
                            sums = rust_f64_slice(sums),
                            global = rust_f64(*global),
                            prior = rust_f64(input.config.prior),
                        ));
                        if *mode == CategoricalMode::Hybrid {
                            source.push_str(&format!(
                                "    x.push(index_{j}.map_or(0.0, |index| counts_{j}[index] as f64 / {total}.0));\n    missing_mask.push(absent_{j});\n"
                            ));
                        }
                        source.push_str(&format!(
                            "    x.push(ordered_{j});\n    missing_mask.push({});\n",
                            if *mode == CategoricalMode::Ordered {
                                format!("absent_{j}")
                            } else {
                                "false".to_string()
                            }
                        ));
                    }
                }
            }
        }
    }
    source.push_str("    Ok((x, missing_mask))\n");
    source
}

fn rust_numeric_ops_body(ops: &[V3Op]) -> Result<String> {
    let mut source = String::new();
    for (index, op) in ops.iter().enumerate() {
        let append = match op {
            V3Op::Legacy(op) => match op {
                FittedOp::Standardize { append, .. }
                | FittedOp::RobustScale { append, .. }
                | FittedOp::TargetBinEncode { append, .. }
                | FittedOp::Sigmoid { append }
                | FittedOp::Step { append, .. }
                | FittedOp::NormRow { append }
                | FittedOp::Lag { append, .. }
                | FittedOp::Roll { append, .. }
                | FittedOp::SelectCorr { append, .. }
                | FittedOp::WinsorClip { append, .. }
                | FittedOp::Impute { append, .. } => *append,
                _ => bail!("this legacy transform is not supported by bound emission"),
            },
            V3Op::MaxAbs { append, .. }
            | V3Op::YeoJohnson { append, .. }
            | V3Op::PiecewiseLinear { append, .. }
            | V3Op::CorrScreen { append, .. }
            | V3Op::PlsProject { append, .. } => *append,
            V3Op::Lag { .. } | V3Op::Roll { .. } => {
                bail!("table-aware row operation requires batch emission")
            }
        };
        source.push_str(&format!("    let generated_{index}: Vec<f64> = "));
        match op {
            V3Op::MaxAbs { scales, .. } => source.push_str(&format!(
                "x.iter().zip({scales}.iter()).map(|(value, scale)| value / scale).collect();\n",
                scales = rust_f64_slice(scales)
            )),
            V3Op::YeoJohnson { lambdas, .. } => source.push_str(&format!(
                "x.iter().zip({lambdas}.iter()).map(|(value, lambda)| yj(*value, *lambda)).collect();\n",
                lambdas = rust_f64_slice(lambdas)
            )),
            V3Op::CorrScreen { indices, .. } => source.push_str(&format!(
                "{indices}.iter().map(|index| x[*index]).collect();\n",
                indices = rust_usize_slice(indices)
            )),
            V3Op::PiecewiseLinear { edges, .. } => {
                source.push_str("{ let mut generated = Vec::new();\n");
                for (j, column) in edges.iter().enumerate() {
                    for interval in column.windows(2) {
                        source.push_str(&format!(
                            "        generated.push(piece(x[{j}], {lo}, {hi}));\n",
                            lo = rust_f64(interval[0]),
                            hi = rust_f64(interval[1])
                        ));
                    }
                }
                source.push_str("        generated };\n");
            }
            V3Op::PlsProject { means, weights, .. } => {
                source.push_str("{ let mut generated = Vec::new();\n");
                for weight in weights {
                    source.push_str(&format!(
                        "        generated.push(dot_centered(&x, &{means}, &{weight}));\n",
                        means = rust_f64_slice(means),
                        weight = rust_f64_slice(weight)
                    ));
                }
                source.push_str("        generated };\n");
            }
            V3Op::Legacy(op) => match op {
                FittedOp::Standardize { means, scales, .. } => source.push_str(&format!(
                    "x.iter().zip({means}.iter()).zip({scales}.iter()).map(|((value, mean), scale)| ((value-mean)/scale).clamp(-20.0, 20.0)).collect();\n",
                    means = rust_f64_slice(means),
                    scales = rust_f64_slice(scales),
                )),
                FittedOp::RobustScale {
                    medians, scales, ..
                } => source.push_str(&format!(
                    "x.iter().zip({medians}.iter()).zip({scales}.iter()).map(|((value, median), scale)| (value-median)/scale).collect();\n",
                    medians = rust_f64_slice(medians),
                    scales = rust_f64_slice(scales),
                )),
                FittedOp::TargetBinEncode { edges, enc, .. } => {
                    source.push_str("{ let mut generated = Vec::new();\n");
                    for (j, (column_edges, column_enc)) in edges.iter().zip(enc).enumerate() {
                        source.push_str(&format!(
                            "        let edges: &[f64] = &{edges}; let encoded: &[f64] = &{encoded}; let bin = edges.partition_point(|edge| *edge < x[{j}]).min(encoded.len()-1); generated.push(encoded[bin]);\n",
                            edges = rust_f64_slice(column_edges),
                            encoded = rust_f64_slice(column_enc),
                        ));
                    }
                    source.push_str("        generated };\n");
                }
                // Mirrors ops.rs apply_op exactly (single-branch form) — the
                // artifact's two-branch `sigmoid` is the HEAD formula and
                // differs from the transform by up to 1 ULP for negatives.
                FittedOp::Sigmoid { .. } => source.push_str(
                    "x.iter().map(|value| 1.0 / (1.0 + (-value.clamp(-30.0, 30.0)).exp())).collect();\n",
                ),
                FittedOp::Step { t, .. } => source.push_str(&format!(
                    "x.iter().map(|value| if *value > {} {{ 1.0 }} else {{ 0.0 }}).collect();\n",
                    rust_f64(*t)
                )),
                FittedOp::NormRow { .. } => source.push_str(
                    "{ let inv = 1.0 / x.iter().map(|value| value*value).sum::<f64>().sqrt().max(1e-9); x.iter().map(|value| value*inv).collect() };\n",
                ),
                FittedOp::Lag { .. } | FittedOp::Roll { .. } => {
                    bail!("legacy row operation is invalid in advanced-v3 emission")
                }
                FittedOp::SelectCorr { indices, .. } => source.push_str(&format!(
                    "{indices}.iter().map(|j| x[*j]).collect();\n",
                    indices = rust_usize_slice(indices)
                )),
                FittedOp::WinsorClip { lo, hi, .. } => source.push_str(&format!(
                    "x.iter().zip({lo}.iter()).zip({hi}.iter()).map(|((value, lo), hi)| value.clamp(*lo, *hi)).collect();\n",
                    lo = rust_f64_slice(lo),
                    hi = rust_f64_slice(hi),
                )),
                FittedOp::Impute {
                    fill,
                    indicator_cols,
                    ..
                } => {
                    source.push_str(&format!(
                        "{{ let fill: &[f64] = &{}; let mut generated: Vec<f64> = x.iter().zip(fill).map(|(value, fill)| if *value == 0.0 {{ *fill }} else {{ *value }}).collect();",
                        rust_f64_slice(fill)
                    ));
                    for column in indicator_cols {
                        source.push_str(&format!(
                            " generated.push(if x[{column}] == 0.0 {{ 1.0 }} else {{ 0.0 }});"
                        ));
                    }
                    source.push_str(" generated };\n");
                }
                _ => unreachable!(),
            },
            V3Op::Lag { .. } | V3Op::Roll { .. } => {
                bail!("table-aware row operation requires batch emission")
            }
        }
        if append {
            source.push_str(&format!("    x.extend(generated_{index});\n"));
        } else {
            source.push_str(&format!("    x = generated_{index};\n"));
        }
        source.push_str("    for value in &mut x { if !value.is_finite() { *value = 0.0; } }\n");
    }
    source.push_str("    x\n");
    Ok(source)
}

fn rust_bool_slice(values: &[bool]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(bool::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn v3_append(op: &V3Op) -> Result<bool> {
    Ok(match op {
        V3Op::Legacy(op) => match op {
            FittedOp::Standardize { append, .. }
            | FittedOp::RobustScale { append, .. }
            | FittedOp::TargetBinEncode { append, .. }
            | FittedOp::Sigmoid { append }
            | FittedOp::Step { append, .. }
            | FittedOp::NormRow { append }
            | FittedOp::SelectCorr { append, .. }
            | FittedOp::WinsorClip { append, .. }
            | FittedOp::Impute { append, .. } => *append,
            FittedOp::Lag { .. } | FittedOp::Roll { .. } => {
                bail!("legacy row operation is invalid in advanced-v3 emission")
            }
            _ => bail!("legacy transform is not supported by bound emission"),
        },
        V3Op::MaxAbs { append, .. }
        | V3Op::YeoJohnson { append, .. }
        | V3Op::PiecewiseLinear { append, .. }
        | V3Op::CorrScreen { append, .. }
        | V3Op::PlsProject { append, .. }
        | V3Op::Lag { append, .. }
        | V3Op::Roll { append, .. } => *append,
    })
}

fn rust_generated_mask_expr(op: &V3Op) -> Result<String> {
    let expression = match op {
        V3Op::MaxAbs { .. } | V3Op::YeoJohnson { .. } => "missing.clone()".to_string(),
        V3Op::CorrScreen { indices, .. } => format!(
            "{}.iter().map(|index| missing[*index]).collect()",
            rust_usize_slice(indices)
        ),
        V3Op::PiecewiseLinear { edges, .. } => {
            let items = edges
                .iter()
                .enumerate()
                .flat_map(|(index, column)| {
                    std::iter::repeat_n(format!("missing[{index}]"), column.len().saturating_sub(1))
                })
                .collect::<Vec<_>>();
            format!("vec![{}]", items.join(","))
        }
        V3Op::PlsProject { weights, .. } => format!("vec![false; {}]", weights.len()),
        V3Op::Legacy(op) => match op {
            FittedOp::Standardize { .. }
            | FittedOp::RobustScale { .. }
            | FittedOp::TargetBinEncode { .. }
            | FittedOp::Sigmoid { .. }
            | FittedOp::Step { .. }
            | FittedOp::NormRow { .. }
            | FittedOp::WinsorClip { .. } => "missing.clone()".to_string(),
            FittedOp::SelectCorr { indices, .. } => format!(
                "{}.iter().map(|index| missing[*index]).collect()",
                rust_usize_slice(indices)
            ),
            FittedOp::Impute { indicator_cols, .. } => {
                format!("vec![false; missing.len() + {}]", indicator_cols.len())
            }
            FittedOp::Lag { .. } | FittedOp::Roll { .. } => {
                bail!("legacy row operation is invalid in advanced-v3 emission")
            }
            _ => bail!("legacy transform is not supported by bound emission"),
        },
        V3Op::Lag { .. } | V3Op::Roll { .. } => {
            bail!("row-operation masks require batch emission")
        }
    };
    Ok(expression)
}

fn rust_semantic_key(key: &SemanticKey) -> String {
    match key {
        SemanticKey::Missing => "SemanticKey::Missing".to_string(),
        SemanticKey::Numeric(value) => format!("SemanticKey::Numeric(FiniteKey({}))", value.0),
        SemanticKey::Categorical(value) => {
            format!(
                "SemanticKey::Categorical({}.to_string())",
                rust_string(value)
            )
        }
    }
}

fn rust_group_tail(tail: &GroupTail) -> String {
    let key = tail
        .key
        .iter()
        .map(rust_semantic_key)
        .collect::<Vec<_>>()
        .join(",");
    let last_order = tail
        .last_order
        .iter()
        .map(rust_semantic_key)
        .collect::<Vec<_>>()
        .join(",");
    let rows = tail
        .rows
        .iter()
        .map(|row| {
            format!(
                "({}.to_vec(),{}.to_vec())",
                rust_f64_slice(&row.values),
                rust_bool_slice(&row.missing)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("histories.insert(vec![{key}], vec![{rows}]);\n    fitted_cutoffs.insert(vec![{key}], vec![{last_order}]);\n")
}

fn rust_key_vector(columns: &[(usize, super::schema::ColumnKind)]) -> String {
    columns
        .iter()
        .map(|(index, kind)| {
            format!(
                "semantic_key(&row[{index}], {})?",
                matches!(kind, super::schema::ColumnKind::Numeric)
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn rust_row_operation_body(index: usize, op: &V3Op) -> Result<String> {
    let (name, k, append, order, tails) = match op {
        V3Op::Lag {
            k,
            append,
            order,
            tails,
        } => ("lag", *k, *append, order, tails),
        V3Op::Roll {
            k,
            append,
            order,
            tails,
        } => ("roll", *k, *append, order, tails),
        _ => bail!("expected a row operation"),
    };
    let group_keys = rust_key_vector(&order.group_columns);
    let order_keys = rust_key_vector(&order.order_columns);
    let mut source = format!(
        "    let group_keys_{index}: Vec<Vec<SemanticKey>> = raw.iter().map(|row| -> Result<Vec<SemanticKey>, String> {{ Ok(vec![{group_keys}]) }}).collect::<Result<Vec<_>, _>>()?;\n    let order_keys_{index}: Vec<Vec<SemanticKey>> = raw.iter().map(|row| -> Result<Vec<SemanticKey>, String> {{ Ok(vec![{order_keys}]) }}).collect::<Result<Vec<_>, _>>()?;\n    let mut chronological_{index}: Vec<usize> = (0..rows.len()).collect();\n    chronological_{index}.sort_by(|left, right| group_keys_{index}[*left].cmp(&group_keys_{index}[*right]).then(order_keys_{index}[*left].cmp(&order_keys_{index}[*right])).then(left.cmp(right)));\n    let width_{index} = rows.first().map_or(0, Vec::len);\n    let mut generated_{index} = vec![vec![0.0; width_{index}]; rows.len()];\n    let mut generated_missing_{index} = vec![vec![true; width_{index}]; rows.len()];\n    let mut histories: std::collections::BTreeMap<Vec<SemanticKey>, Vec<(Vec<f64>, Vec<bool>)>> = std::collections::BTreeMap::new();\n    let mut fitted_cutoffs: std::collections::BTreeMap<Vec<SemanticKey>, Vec<SemanticKey>> = std::collections::BTreeMap::new();\n"
    );
    for tail in tails {
        source.push_str("    ");
        source.push_str(&rust_group_tail(tail));
    }
    source.push_str(&format!(
        "    for row_index in 0..rows.len() {{ if let Some(cutoff) = fitted_cutoffs.get(&group_keys_{index}[row_index]) {{ if order_keys_{index}[row_index].as_slice() <= cutoff.as_slice() {{ return Err(\"prediction row order must be strictly after fitted order\".to_string()); }} }} }}\n"
    ));
    source.push_str(&format!(
        "    for row_index in chronological_{index} {{\n        let history = histories.entry(group_keys_{index}[row_index].clone()).or_default();\n"
    ));
    if name == "lag" {
        source.push_str(&format!(
            "        if history.len() >= {k} {{ generated_{index}[row_index].clone_from(&history[history.len()-{k}].0); generated_missing_{index}[row_index].clone_from(&history[history.len()-{k}].1); }}\n"
        ));
    } else {
        source.push_str(&format!(
            "        for feature in 0..width_{index} {{ let mut sum = 0.0; let mut count = 0usize; let start = history.len().saturating_sub({previous}); for prior in &history[start..] {{ if !prior.1[feature] && prior.0[feature].is_finite() {{ sum += prior.0[feature]; count += 1; }} }} if !masks[row_index][feature] && rows[row_index][feature].is_finite() {{ sum += rows[row_index][feature]; count += 1; }} if count > 0 {{ generated_{index}[row_index][feature] = sum / count as f64; generated_missing_{index}[row_index][feature] = false; }} }}\n",
            previous = k.saturating_sub(1)
        ));
    }
    let keep = if name == "lag" {
        k
    } else {
        k.saturating_sub(1)
    };
    source.push_str(&format!(
        "        history.push((rows[row_index].clone(), masks[row_index].clone()));\n        if history.len() > {keep} {{ history.drain(..history.len()-{keep}); }}\n    }}\n"
    ));
    if append {
        source.push_str(&format!(
            "    for row_index in 0..rows.len() {{ rows[row_index].extend(std::mem::take(&mut generated_{index}[row_index])); masks[row_index].extend(std::mem::take(&mut generated_missing_{index}[row_index])); }}\n"
        ));
    } else {
        source.push_str(&format!(
            "    rows = generated_{index};\n    masks = generated_missing_{index};\n"
        ));
    }
    source.push_str(
        "    for (row, mask) in rows.iter_mut().zip(&masks) { for (value, absent) in row.iter_mut().zip(mask) { if *absent || !value.is_finite() { *value = 0.0; } } }\n",
    );
    Ok(source)
}

fn rust_batch_ops_source(ops: &[V3Op]) -> Result<(String, String)> {
    let mut helpers = String::new();
    let mut body = String::new();
    for (index, op) in ops.iter().enumerate() {
        if matches!(op, V3Op::Lag { .. } | V3Op::Roll { .. }) {
            body.push_str(&rust_row_operation_body(index, op)?);
            continue;
        }
        let numeric_body = rust_numeric_ops_body(std::slice::from_ref(op))?;
        helpers.push_str(&format!(
            "fn numeric_op_{index}(mut x: Vec<f64>) -> Vec<f64> {{\n{numeric_body}}}\n\n"
        ));
        let mask_expr = rust_generated_mask_expr(op)?;
        let append = v3_append(op)?;
        body.push_str(&format!(
            "    for row_index in 0..rows.len() {{\n        let missing = masks[row_index].clone();\n        let generated_missing: Vec<bool> = {mask_expr};\n        rows[row_index] = numeric_op_{index}(std::mem::take(&mut rows[row_index]));\n"
        ));
        if append {
            body.push_str("        masks[row_index].extend(generated_missing);\n");
        } else {
            body.push_str("        masks[row_index] = generated_missing;\n");
        }
        body.push_str(
            "        for (value, absent) in rows[row_index].iter_mut().zip(&masks[row_index]) { if *absent || !value.is_finite() { *value = 0.0; } }\n    }\n",
        );
    }
    body.push_str("    Ok(rows)\n");
    Ok((helpers, body))
}

fn rust_linear_predictor(
    means: &[f64],
    scales: &[f64],
    weights: &[f64],
    probability: bool,
    input: &str,
) -> String {
    format!(
        "{{ let means: &[f64] = &{means}; let scales: &[f64] = &{scales}; let weights: &[f64] = &{weights}; let mut value = weights[0]; for j in 0..means.len() {{ value += weights[j+1] * (({input}[j]-means[j])/scales[j]).clamp({lo}, {hi}); }} {finish} }}",
        means = rust_f64_slice(means),
        scales = rust_f64_slice(scales),
        weights = rust_f64_slice(weights),
        lo = if probability { "-30.0" } else { "-20.0" },
        hi = if probability { "30.0" } else { "20.0" },
        finish = if probability { "sigmoid(value)" } else { "value.clamp(-60.0, 60.0)" },
    )
}

fn rust_tree_expr(node: &ExtraNode, input: &str) -> String {
    match node {
        ExtraNode::Leaf(value) => rust_f64(*value),
        ExtraNode::Split {
            feature,
            threshold,
            missing_left,
            left,
            right,
        } => format!(
            "if if {input}[{feature}].is_finite() {{ {input}[{feature}] <= {threshold} }} else {{ {missing_left} }} {{ {left} }} else {{ {right} }}",
            threshold = rust_f64(*threshold),
            left = rust_tree_expr(left, input),
            right = rust_tree_expr(right, input),
        ),
    }
}

fn rust_head_body(head: &V3Head) -> Result<String> {
    let source = match head {
        V3Head::LogitRidge(model) => {
            rust_linear_predictor(&model.means, &model.scales, &model.weights, true, "x")
        }
        V3Head::Legacy(FittedHead::Ridge(model)) => {
            rust_linear_predictor(&model.means, &model.scales, &model.w, false, "x")
        }
        V3Head::TabBoost(model) => {
            let mut body = format!("{{ let mut value = {};", rust_f64(model.base));
            for stump in &model.stumps {
                body.push_str(&format!(
                    " value += {lr} * if if x[{feature}].is_finite() {{ x[{feature}] <= {threshold} }} else {{ {missing_left} }} {{ {left} }} else {{ {right} }};",
                    lr = rust_f64(model.learning_rate),
                    feature = stump.feature,
                    threshold = rust_f64(stump.threshold),
                    missing_left = stump.missing_left,
                    left = rust_f64(stump.left),
                    right = rust_f64(stump.right),
                ));
            }
            body.push_str(if model.binary {
                " sigmoid(value) }"
            } else {
                " finite(value) }"
            });
            body
        }
        V3Head::ExtraTrees(model) => {
            let trees = model
                .trees
                .iter()
                .map(|tree| format!("({})", rust_tree_expr(tree, "x")))
                .collect::<Vec<_>>();
            let sum = if trees.is_empty() {
                "0.0".to_string()
            } else {
                trees.join(" + ")
            };
            if model.binary {
                format!(
                    "(({sum}) / {}.0_f64).clamp(0.0_f64, 1.0_f64)",
                    trees.len().max(1)
                )
            } else {
                format!("finite(({sum}) / {}.0)", trees.len().max(1))
            }
        }
        V3Head::GamBoost(model) => {
            let mut body = format!("{{ let mut value = {};", rust_f64(model.base));
            for (index, term) in model.terms.iter().enumerate() {
                body.push_str(&format!(
                    " let edges_{index}: &[f64] = &{edges}; let values_{index}: &[f64] = &{values}; let bin_{index} = edges_{index}.partition_point(|edge| *edge < x[{feature}]).min(values_{index}.len()-1); value += {lr} * values_{index}[bin_{index}];",
                    edges = rust_f64_slice(&term.edges),
                    values = rust_f64_slice(&term.values),
                    feature = term.feature,
                    lr = rust_f64(model.learning_rate),
                ));
            }
            body.push_str(if model.binary {
                " sigmoid(value) }"
            } else {
                " finite(value) }"
            });
            body
        }
        V3Head::Pls(model) => {
            let V3Op::PlsProject { means, weights, .. } = &model.projection else {
                unreachable!()
            };
            let mut body = String::from("{ let projected: Vec<f64> = vec![");
            body.push_str(
                &weights
                    .iter()
                    .map(|weight| {
                        format!(
                            "dot_centered(x, &{}, &{})",
                            rust_f64_slice(means),
                            rust_f64_slice(weight)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(","),
            );
            body.push_str("]; ");
            body.push_str(&rust_linear_predictor(
                &model.ridge.means,
                &model.ridge.scales,
                &model.ridge.w,
                false,
                "projected",
            ));
            body.push_str(" }");
            body
        }
        V3Head::ClassificationPortfolio(model) => {
            let mut terms = Vec::with_capacity(model.members.len());
            for (weight, member) in model.weights.iter().zip(&model.members) {
                terms.push(format!(
                    "{} * ({})",
                    rust_f64(*weight),
                    rust_head_body(member)?
                ));
            }
            format!("({}).clamp(0.0, 1.0)", terms.join(" + "))
        }
        V3Head::MulticlassRidge(_)
        | V3Head::MulticlassOvrLogistic(_)
        | V3Head::MulticlassDiagonalLda(_)
        | V3Head::MulticlassNearestCentroid(_) => {
            bail!("multiclass probability matrices are not supported by scalar bound emission")
        }
        V3Head::Legacy(_) => bail!("this fitted legacy head is not supported by bound emission"),
    };
    Ok(source)
}

fn rust_calibration_body(calibration: &CalibrationModel, binary: bool) -> String {
    let calibrated = match calibration {
        CalibrationModel::None => "value".to_string(),
        CalibrationModel::Platt { slope, intercept } => format!(
            "{{ let p = value.clamp(1e-12, 1.0-1e-12); sigmoid({} * (p/(1.0-p)).ln() + {}) }}",
            rust_f64(*slope),
            rust_f64(*intercept)
        ),
        CalibrationModel::Isotonic { upper, values } => format!(
            "{{ let upper: &[f64] = &{}; let values: &[f64] = &{}; let index = upper.partition_point(|bound| *bound < value).min(values.len()-1); values[index] }}",
            rust_f64_slice(upper),
            rust_f64_slice(values)
        ),
    };
    if binary {
        format!("({calibrated}).clamp(0.0, 1.0)")
    } else {
        format!("finite({calibrated})")
    }
}

fn rust_target_body(target: &Option<FittedTarget>, input: &str) -> String {
    match target {
        None => input.to_string(),
        Some(FittedTarget::Log1p) => {
            format!("({input}.clamp(-745.0, 700.0).exp() - 1.0).clamp(-60.0, 60.0)")
        }
        Some(FittedTarget::Winsor { .. }) => format!("{input}.clamp(-60.0, 60.0)"),
        Some(FittedTarget::ZScore {
            mu,
            sigma,
            clamp_lo,
            clamp_hi,
        }) => format!(
            "({input} * {} + {}).clamp({}, {})",
            rust_f64(*sigma),
            rust_f64(*mu),
            rust_f64(*clamp_lo),
            rust_f64(*clamp_hi)
        ),
        Some(FittedTarget::RankGauss {
            table_values,
            table_scores,
            ..
        }) => format!(
            "rank_inverse({input}, &{}, &{}).clamp(-60.0, 60.0)",
            rust_f64_slice(table_scores),
            rust_f64_slice(table_values)
        ),
    }
}

fn emit_bound_rust(model: &BoundModel) -> Result<String> {
    let BoundPredictor::Pipeline {
        target,
        ops,
        head,
        calibration,
        missing_aware,
    } = &model.predictor
    else {
        bail!("bound emission currently requires a v3 pipeline model");
    };
    let headers = model
        .schema
        .columns
        .iter()
        .map(|column| rust_string(&column.name))
        .collect::<Vec<_>>()
        .join(",");
    let input_body = rust_input_body(&model.input, *missing_aware);
    let (ops_helpers, ops_body) = rust_batch_ops_source(ops)?;
    let head_body = rust_head_body(head)?;
    let calibration_body = rust_calibration_body(calibration, model.task.kind == TaskKind::Binary);
    let target_body = rust_target_body(target, "calibrated");
    let mut source = String::from(
        r#"//! Schema-bound, standard-library-only prediction artifact.
use std::env;
use std::fs;

fn parse_csv(text: &str) -> Result<Vec<Vec<String>>, String> {
    let mut records = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut chars = text.chars().peekable();
    let mut quoted = false;
    let mut at_start = true;
    while let Some(ch) = chars.next() {
        if quoted {
            if ch == '"' {
                if chars.peek() == Some(&'"') { chars.next(); field.push('"'); }
                else { quoted = false; }
            } else { field.push(ch); }
            continue;
        }
        match ch {
            '"' if at_start => { quoted = true; at_start = false; }
            ',' => { row.push(std::mem::take(&mut field)); at_start = true; }
            '\n' => { row.push(std::mem::take(&mut field)); records.push(std::mem::take(&mut row)); at_start = true; }
            '\r' if chars.peek() == Some(&'\n') => {}
            '\r' => { row.push(std::mem::take(&mut field)); records.push(std::mem::take(&mut row)); at_start = true; }
            '"' => return Err("quote in unquoted CSV field".to_string()),
            other => { field.push(other); at_start = false; }
        }
    }
    if quoted { return Err("unterminated quoted CSV field".to_string()); }
    if !field.is_empty() || !row.is_empty() { row.push(field); records.push(row); }
    Ok(records)
}

fn is_missing(text: &str) -> bool {
    let value = text.trim().to_ascii_lowercase();
    matches!(value.as_str(), "" | "na" | "nan" | "null")
}

fn is_category_missing(text: &str) -> bool {
    let value = text.trim().to_ascii_lowercase();
    matches!(value.as_str(), "" | "na" | "null")
}

fn category(text: &str) -> String {
    if is_category_missing(text) { "\0MISSING".to_string() } else { text.to_string() }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FiniteKey(u64);

impl FiniteKey { fn value(self) -> f64 { f64::from_bits(self.0) } }
impl PartialOrd for FiniteKey { fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) } }
impl Ord for FiniteKey { fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.value().total_cmp(&other.value()) } }

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SemanticKey { Missing, Numeric(FiniteKey), Categorical(String) }

fn semantic_key(text: &str, numeric: bool) -> Result<SemanticKey, String> {
    if numeric {
        if is_missing(text) { return Ok(SemanticKey::Missing); }
        let value = text.trim().parse::<f64>().map_err(|_| format!("invalid numeric value {text}"))?;
        if !value.is_finite() { return Ok(SemanticKey::Missing); }
        return Ok(SemanticKey::Numeric(FiniteKey(if value == 0.0 { 0 } else { value.to_bits() })));
    }
    Ok(if is_category_missing(text) { SemanticKey::Missing } else { SemanticKey::Categorical(text.to_string()) })
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 { 1.0 / (1.0 + (-value.min(40.0)).exp()) }
    else { let z = value.max(-40.0).exp(); z / (1.0 + z) }
}

fn finite(value: f64) -> f64 { if value.is_finite() { value } else { 0.0 } }

fn yj(value: f64, lambda: f64) -> f64 {
    if value >= 0.0 {
        if lambda.abs() < 1e-12 { value.ln_1p() } else { ((value+1.0).powf(lambda)-1.0)/lambda }
    } else if (lambda-2.0).abs() < 1e-12 { -(-value).ln_1p() }
    else { -((1.0-value).powf(2.0-lambda)-1.0)/(2.0-lambda) }
}

fn piece(value: f64, lo: f64, hi: f64) -> f64 {
    if (hi-lo).abs() <= 1e-15 { if value > lo { 1.0 } else { 0.0 } }
    else { ((value-lo)/(hi-lo)).clamp(0.0, 1.0) }
}

fn dot_centered(x: &[f64], means: &[f64], weights: &[f64]) -> f64 {
    x.iter().zip(means).zip(weights).map(|((value, mean), weight)| weight*(value-mean)).sum()
}

fn rank_inverse(value: f64, scores: &[f64], values: &[f64]) -> f64 {
    // NaN routes to the low end (mirrors the reference inverse); `<=` would
    // fall through and underflow `hi - 1`.
    if !(value > scores[0]) { return values[0]; }
    if value >= scores[scores.len()-1] { return values[values.len()-1]; }
    let hi = scores.partition_point(|score| *score < value);
    let lo = hi - 1;
    values[lo] + (value-scores[lo])/(scores[hi]-scores[lo])*(values[hi]-values[lo])
}

fn encode(row: &[String]) -> Result<(Vec<f64>, Vec<bool>), String> {
"#,
    );
    source.push_str(&input_body);
    source.push_str("}\n\n");
    source.push_str(&ops_helpers);
    source.push_str("fn apply_ops(mut rows: Vec<Vec<f64>>, mut masks: Vec<Vec<bool>>, raw: &[Vec<String>]) -> Result<Vec<Vec<f64>>, String> {\n");
    source.push_str(&ops_body);
    source.push_str("}\n\nfn predict_numeric(x: &[f64]) -> f64 {\n    let value = ");
    source.push_str(&head_body);
    source.push_str(";\n    let calibrated = ");
    source.push_str(&calibration_body);
    source.push_str(";\n    let mapped = ");
    source.push_str(&target_body);
    // The final task mapping mirrors BoundModel::predict_table exactly:
    // binary NaN -> 0.5 chance, regression NaN -> 0.0.
    if model.task.kind == TaskKind::Binary {
        source.push_str(";\n    if mapped.is_finite() { mapped.clamp(0.0, 1.0) } else { 0.5 }");
    } else {
        source.push_str(";\n    if mapped.is_finite() { mapped } else { 0.0 }");
    }
    source.push_str("\n}\n\nfn run() -> Result<(), String> {\n");
    source.push_str(&format!(
        "    let expected: Vec<String> = vec![{headers}].into_iter().map(str::to_string).collect();\n"
    ));
    source.push_str(
        r#"    let args: Vec<String> = env::args().collect();
    if args.len() != 2 && args.len() != 3 { return Err("usage: model INPUT.csv [OUTPUT.csv]".to_string()); }
    let text = fs::read_to_string(&args[1]).map_err(|error| error.to_string())?;
    let records = parse_csv(&text)?;
    if records.first() != Some(&expected) { return Err("CSV header does not match bound schema".to_string()); }
    let raw = &records[1..];
    let mut features = Vec::with_capacity(raw.len());
    let mut masks = Vec::with_capacity(raw.len());
    for row in raw {
        if row.len() != expected.len() { return Err("CSV row width mismatch".to_string()); }
        let (values, missing) = encode(row)?;
        features.push(values);
        masks.push(missing);
    }
    let transformed = apply_ops(features, masks, raw)?;
    let mut output = String::new();
    for row in &transformed {
        let prediction = predict_numeric(row);
        // Debug formatting is shortest-round-trip: parses back to the exact
        // f64, unlike fixed-point {:.17} which zeroes small magnitudes.
        output.push_str(&format!("{prediction:?}\n"));
    }
    if args.len() == 3 { fs::write(&args[2], output).map_err(|error| error.to_string())?; }
    else { print!("{output}"); }
    Ok(())
}

fn main() {
    if let Err(error) = run() { eprintln!("{error}"); std::process::exit(2); }
}
"#,
    );
    Ok(source)
}

#[cfg(test)]
mod typescript_support {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/typescript_support/mod.rs"
    ));
}

#[cfg(test)]
mod go_support {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/go_support/mod.rs"
    ));
}

#[cfg(test)]
mod tests {
    use super::super::csv::RawTable;
    use super::super::pipeline::fit_table;
    use super::super::schema::{ColumnSpec, DataSchema, RowSemantics};
    use super::super::types::{Objective, TaskSpec};
    use super::*;
    use crate::GenomeIr;
    use std::process::Command;

    fn ordered_schema() -> DataSchema {
        DataSchema {
            columns: vec![
                ColumnSpec::categorical("group"),
                ColumnSpec::numeric("time"),
                ColumnSpec::numeric("value"),
            ],
            row_semantics: RowSemantics::Ordered {
                group_by: vec!["group".to_string()],
                order_by: vec!["time".to_string()],
            },
        }
    }

    fn predictions(output: &[u8]) -> Vec<f64> {
        String::from_utf8(output.to_vec())
            .unwrap()
            .lines()
            .map(|line| line.parse().unwrap())
            .collect()
    }

    fn assert_row_op_artifacts(op: &str, index: usize) {
        let schema = ordered_schema();
        let training = RawTable::new(
            vec!["group".into(), "time".into(), "value".into()],
            vec![
                vec!["A".into(), "1".into(), "1".into()],
                vec!["A".into(), "2".into(), "2".into()],
                vec!["B".into(), "1".into(), "10".into()],
                vec!["B".into(), "2".into(), "20".into()],
                vec!["nan".into(), "1".into(), "5".into()],
                vec!["nan".into(), "2".into(), "6".into()],
            ],
        )
        .unwrap();
        let task = TaskSpec::regression(Objective::R2).unwrap();
        let genome = GenomeIr::parse(&format!(
            "(pipe X (input numeric=median categorical=ordered prior=0) ({op}) (ridge))"
        ))
        .unwrap();
        let model = fit_table(
            &genome,
            &task,
            &schema,
            &training,
            &[1.0, 2.0, 10.0, 20.0, 5.0, 6.0],
        )
        .unwrap();
        // Reverse chronological caller order, equal-order A rows (ordinal
        // tie-break), fitted A/B tails, and an unseen all-missing C value.
        let csv = "group,time,value\nA,4,4\nA,3,3\nA,3,5\nB,3,30\nC,1,\nD,1,10000000000000000\nD,2,1\nD,3,-10000000000000000\n,1,\n,2,\nnan,3,inf\nnan,4,nan\n";
        let test = RawTable::from_csv(csv).unwrap();
        let runtime = model.predict_table(&test).unwrap();
        let emitted = emit_bound(&model, &crate::transpile::EmitOptions::default()).unwrap();
        let portable_error = emit_bound(
            &model,
            &crate::transpile::EmitOptions {
                math_kernel: crate::transpile::MathKernel::Portable,
                ..crate::transpile::EmitOptions::default()
            },
        )
        .expect_err("bound emission must not silently ignore Portable");
        assert!(portable_error.to_string().contains("portable"));

        let scratch = tempfile::Builder::new()
            .prefix(&format!("feat-math-v3-row-op-emission-{index}-"))
            .tempdir()
            .unwrap();
        let temp = scratch.path();
        if super::typescript_support::enabled() {
            let bundle =
                crate::typescript::emit_bound(&model, &crate::transpile::EmitOptions::default())
                    .unwrap();
            super::typescript_support::run(
                &temp.join("typescript"),
                &bundle.source,
                &bundle.kernel,
                &serde_json::json!({"rows":test.rows}),
                &runtime,
                true,
            );
        }
        if super::go_support::enabled() {
            let bundle =
                crate::go::emit_bound(&model, &crate::transpile::EmitOptions::default()).unwrap();
            super::go_support::run(
                &temp.join("go"),
                &bundle.source,
                &bundle.kernel,
                &serde_json::json!({"rows":test.rows}),
                &runtime,
                true,
            );
        }
        let csv_path = temp.join("test.csv");
        let python_path = temp.join("model.py");
        let rust_path = temp.join("model.rs");
        let binary_path = temp.join("model");
        std::fs::write(&csv_path, csv).unwrap();
        std::fs::write(&python_path, emitted.python).unwrap();
        std::fs::write(&rust_path, emitted.rust).unwrap();
        let compilation = Command::new("rustc")
            .args(["--edition=2021", "-O"])
            .arg(&rust_path)
            .arg("-o")
            .arg(&binary_path)
            .output()
            .unwrap();
        assert!(
            compilation.status.success(),
            "generated Rust failed: {}",
            String::from_utf8_lossy(&compilation.stderr)
        );
        let python = Command::new("python3")
            .arg(&python_path)
            .arg(&csv_path)
            .output()
            .unwrap();
        let rust = Command::new(&binary_path).arg(&csv_path).output().unwrap();
        assert!(
            python.status.success(),
            "generated Python failed: {}",
            String::from_utf8_lossy(&python.stderr)
        );
        assert!(
            rust.status.success(),
            "generated Rust failed: {}",
            String::from_utf8_lossy(&rust.stderr)
        );
        let python_predictions = predictions(&python.stdout);
        let rust_predictions = predictions(&rust.stdout);
        assert_eq!(python_predictions.len(), runtime.len());
        assert_eq!(rust_predictions.len(), runtime.len());
        for (row, ((python, rust), runtime)) in python_predictions
            .iter()
            .zip(&rust_predictions)
            .zip(&runtime)
            .enumerate()
        {
            assert_eq!(python.to_bits(), runtime.to_bits(), "Python row {row}");
            assert_eq!(rust.to_bits(), runtime.to_bits(), "Rust row {row}");
        }
        if op.starts_with("roll") {
            assert_eq!(
                python_predictions[7].to_bits(),
                runtime[7].to_bits(),
                "Python cancellation-window row must use the runtime's sequential fold"
            );
            assert_eq!(
                rust_predictions[7].to_bits(),
                runtime[7].to_bits(),
                "Rust cancellation-window row must use the runtime's sequential fold"
            );
        }

        let stale_csv = "group,time,value\nA,2,99\n";
        let stale_table = RawTable::from_csv(stale_csv).unwrap();
        assert!(model.predict_table(&stale_table).is_err());
        let stale_path = temp.join("stale.csv");
        std::fs::write(&stale_path, stale_csv).unwrap();
        let stale_python = Command::new("python3")
            .arg(&python_path)
            .arg(&stale_path)
            .output()
            .unwrap();
        let stale_rust = Command::new(&binary_path)
            .arg(&stale_path)
            .output()
            .unwrap();
        assert!(
            !stale_python.status.success(),
            "generated Python accepted a non-future fitted-tail row"
        );
        assert!(
            !stale_rust.status.success(),
            "generated Rust accepted a non-future fitted-tail row"
        );

        let mut corrupt_model = model.clone();
        let BoundPredictor::Pipeline { ops, .. } = &mut corrupt_model.predictor else {
            unreachable!()
        };
        match &mut ops[0] {
            V3Op::Lag { tails, .. } | V3Op::Roll { tails, .. } => {
                tails[0].last_order[0] = SemanticKey::Missing;
            }
            _ => unreachable!(),
        }
        assert!(
            emit_bound(&corrupt_model, &crate::transpile::EmitOptions::default())
                .unwrap_err()
                .to_string()
                .contains("tail state hash mismatch"),
            "bound emission must reject corrupt fitted-tail state"
        );
    }

    #[test]
    fn emitted_lag_and_roll_use_fitted_group_tails() {
        assert_row_op_artifacts("lag k=2", 0);
        assert_row_op_artifacts("roll k=3", 1);
    }
}

#[cfg(test)]
mod quotation_regression_tests {
    use super::*;
    use crate::advanced_v3::{
        fit_table, predict_table, ColumnSpec, DataSchema, Objective, RawTable, RowSemantics,
        TaskSpec,
    };
    #[test]
    fn bound_python_control_unicode_json_is_valid_and_preserves_fitted_bits() {
        let table = RawTable::new(
            vec!["label\u{2028}\u{301}\u{7f}".into()],
            (0..12)
                .map(|i| {
                    vec![format!(
                        "category-{}#![forbid(unsafe_code)]\u{2028}\u{301}\u{7f}",
                        i % 3
                    )]
                })
                .collect(),
        )
        .unwrap();
        let schema = DataSchema {
            columns: vec![ColumnSpec::categorical(table.headers[0].clone())],
            row_semantics: RowSemantics::Iid,
        };
        let model = fit_table(
            &crate::GenomeIr::parse("(pipe X (input categorical=onehot) (ridge))").unwrap(),
            &TaskSpec::regression(Objective::Rmse).unwrap(),
            &schema,
            &table,
            &(0..12).map(|i| i as f64).collect::<Vec<_>>(),
        )
        .unwrap();
        if super::typescript_support::enabled() {
            let bundle =
                crate::typescript::emit_bound(&model, &crate::transpile::EmitOptions::default())
                    .unwrap();
            let scratch = tempfile::tempdir().unwrap();
            super::typescript_support::run(
                scratch.path(),
                &bundle.source,
                &bundle.kernel,
                &serde_json::json!({"rows":table.rows}),
                &model.predict_table(&table).unwrap(),
                true,
            );
        }
        if super::go_support::enabled() {
            let bundle =
                crate::go::emit_bound(&model, &crate::transpile::EmitOptions::default()).unwrap();
            let scratch = tempfile::tempdir().unwrap();
            super::go_support::run(
                scratch.path(),
                &bundle.source,
                &bundle.kernel,
                &serde_json::json!({"rows":table.rows}),
                &model.predict_table(&table).unwrap(),
                true,
            );
        }
        let source = emit_bound(&model, &crate::transpile::EmitOptions::default())
            .unwrap()
            .python;
        let rows = serde_json::to_string(&table.rows).unwrap();
        let script=format!("{source}\nimport struct\nrows=json.loads({})\nfor row in rows:\n    print(struct.pack('>d',predict_numeric(encode(row)[0])).hex())\n",serde_json::to_string(&rows).unwrap());
        let mut child = std::process::Command::new("python3")
            .args([
                "-c",
                "import sys;exec(compile(sys.stdin.read(),'<fitted>','exec'), {'__name__':'fitted_test'})",
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        use std::io::Write;
        child
            .stdin
            .take()
            .unwrap()
            .write_all(script.as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let actual = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(|line| u64::from_str_radix(line, 16).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            predict_table(&model, &table)
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

/// Opt-in v3 binding around unchanged v2 bound sources. The caller supplies the
/// selected canonical program; its digest must match the fitted model's binding.
pub fn emit_bound_targets_v3(
    model: &BoundModel,
    program: &crate::GenomeIr,
    options: &crate::transpile::EmitOptions,
    requested: &[crate::artifact_manifest::Target],
) -> Result<crate::artifact_manifest::TargetEmission> {
    use crate::artifact_manifest::{
        sha256, ArtifactManifestV3, FittedIdentity, Target, TargetEmission, TargetRecord,
    };
    anyhow::ensure!(
        sha256(program.canonical().as_bytes()) == model.genome_sha256,
        "bound model does not identify selected program"
    );
    let bound = emit_bound(model, options)?;
    let old: serde_json::Value = serde_json::from_str(&bound.manifest)?;
    let emitter = old["emitter_version"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("missing bound emitter identity"))?;
    let fitted = FittedIdentity {
        state_sha256: sha256(&serde_json::to_vec(model)?),
        input_sha256: sha256(&serde_json::to_vec(&model.input)?),
        task_sha256: model.task_sha256.clone(),
        schema_sha256: model.schema_sha256.clone(),
        vocabulary_sha256: model.vocabulary_sha256.clone(),
    };
    let mut records = Vec::new();
    let mut sources = std::collections::BTreeMap::new();
    for &target in requested {
        if target == Target::Cuda {
            let bundle = crate::cuda::emit_bound(model, options)?;
            let inputs: Vec<_> = bundle
                .files
                .iter()
                .map(|(p, s)| (p.as_str(), s.as_str()))
                .collect();
            records.push(TargetRecord::supported(
                target,
                "advanced_v3",
                crate::cuda::EMITTER,
                &inputs,
                Some(fitted.clone()),
            ));
            sources.extend(bundle.files);
            continue;
        }
        if target == Target::TypeScript {
            let bundle = crate::typescript::emit_bound(model, options)?;
            records.push(TargetRecord::supported(
                target,
                "advanced_v3",
                crate::typescript::EMITTER_VERSION,
                &[
                    ("model.ts", &bundle.source),
                    ("model.kernel.rs", &bundle.kernel),
                ],
                Some(fitted.clone()),
            ));
            sources.insert("model.ts".into(), bundle.source);
            sources.insert("model.kernel.rs".into(), bundle.kernel);
            continue;
        }
        if target == Target::Go {
            let bundle = crate::go::emit_bound(model, options)?;
            records.push(TargetRecord::supported(
                target,
                "advanced_v3",
                crate::go::EMITTER_VERSION,
                &crate::go::sources(&bundle),
                Some(fitted.clone()),
            ));
            for (path, source) in crate::go::sources(&bundle) {
                sources.insert(path.into(), source.into());
            }
            continue;
        }
        let source = match target {
            Target::Python => Some(&bound.python),
            Target::Rust => Some(&bound.rust),
            _ => None,
        };
        if let Some(source) = source {
            records.push(TargetRecord::supported(
                target,
                "advanced_v3",
                emitter,
                &[(target.path(), source)],
                Some(fitted.clone()),
            ));
            sources.insert(target.path().into(), source.clone());
        } else {
            records.push(TargetRecord::unsupported(
                target,
                "bound-v2 adapter supports Python, Rust, TypeScript, Go and CUDA C++".into(),
            ));
        }
    }
    Ok(TargetEmission {
        manifest: ArtifactManifestV3::new(
            &program.canonical(),
            "advanced_v3",
            options,
            requested,
            records,
        )?,
        sources,
    })
}

impl BoundEmitted {
    /// Versioned target binding without changing the existing bound emitter API.
    pub fn emit_targets_v3(
        model: &BoundModel,
        program: &crate::GenomeIr,
        options: &crate::transpile::EmitOptions,
        requested: &[crate::artifact_manifest::Target],
    ) -> Result<crate::artifact_manifest::TargetEmission> {
        emit_bound_targets_v3(model, program, options, requested)
    }
}

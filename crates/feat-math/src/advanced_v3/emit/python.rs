use super::*;

pub(crate) fn emit_bound_python(
    model: &BoundModel,
    partition_point_quantile_bounds: bool,
) -> Result<String> {
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

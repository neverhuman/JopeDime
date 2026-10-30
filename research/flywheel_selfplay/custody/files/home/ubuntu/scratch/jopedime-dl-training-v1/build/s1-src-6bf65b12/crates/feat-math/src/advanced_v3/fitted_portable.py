"""Fitted, standard-library-only matrix inference."""
import json
import math

STATE = json.loads(__STATE_JSON__)
_LOOKUPS = [{value: i for i, value in enumerate(wrapped["Onehot"]["vocabulary"])} if "Onehot" in wrapped else None for wrapped in STATE["input"]]


def _finite(value):
    return value if math.isfinite(value) else 0.0


def _clamp(value, low, high):
    return low if value < low else (high if value > high else value)


def _sigmoid(value):
    if math.isnan(value):
        value = -40.0
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(value, 40.0)))
    z = math.exp(max(value, -40.0))
    return z / (1.0 + z)


def _linear(model, row):
    value = model["weights"][0]
    for j, mean in enumerate(model["means"]):
        raw = _finite(row[j]) if model["finite_input"] else row[j]
        value += model["weights"][j + 1] * _clamp(
            (raw - mean) / model["scales"][j], -model["bound"], model["bound"])
    return value


def _softmax(scores):
    maximum = max(scores)
    values = [math.exp(_clamp(_finite(value) - maximum, -40.0, 0.0)) for value in scores]
    total = 0.0
    for value in values:
        total += value
    total = max(total, 1e-300)
    return [value / total for value in values]


def _normalize(values):
    total = 0.0
    for value in values:
        total += value
    if not math.isfinite(total) or total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def _encode(row):
    if len(row) != len(STATE["input"]) or any(not isinstance(value, str) for value in row):
        raise ValueError("input shape differs")
    out = []
    for raw, wrapped, lookup in zip(row, STATE["input"], _LOOKUPS):
        kind, state = next(iter(wrapped.items()))
        if kind == "Numeric":
            absent = raw.strip().lower() in ("", "na", "nan", "null")
            value = state["fill"] if absent else float(raw)
            absent = absent or not math.isfinite(value)
            out.append(float("nan") if absent and state["restore_missing"] else (state["fill"] if absent else value))
            if state["indicator"]:
                out.append(1.0 if absent else 0.0)
        else:
            key = "\0MISSING" if raw.strip().lower() in ("", "na", "null") else raw
            found = lookup.get(key)
            out.extend(1.0 if found == index else 0.0 for index in state["indices"])
    return out


def _distance(row, center, scales, squared_scale):
    total = 0.0
    for raw, mean, scale in zip(row, center, scales):
        delta = _finite(raw) - mean
        if squared_scale:
            total += delta * delta / scale
        else:
            delta /= scale
            total += delta * delta
    return total


def _branch(node, row):
    kind, state = next(iter(node.items()))
    if kind == "Leaf":
        return state
    raw = row[state["feature"]]
    left = raw <= state["threshold"] if math.isfinite(raw) else state["absent_left"]
    return _branch(state["left"] if left else state["right"], row)


def _scalar(wrapped, row):
    kind, state = next(iter(wrapped.items()))
    if kind == "Probability":
        return _sigmoid(_linear(state, row))
    if kind == "Rules":
        value = state["base"]
        for rule in state["rules"]:
            raw = row[rule["feature"]]
            left = raw <= rule["threshold"] if math.isfinite(raw) else rule["absent_left"]
            value += state["rate"] * (rule["left"] if left else rule["right"])
        return _sigmoid(value) if state["binary"] else _finite(value)
    if kind == "Average":
        value = -0.0
        for branch in state["branches"]:
            value += _branch(branch, row)
        value /= max(1, len(state["branches"]))
        return _clamp(value, 0.0, 1.0) if state["binary"] else _finite(value)
    if kind == "Bins":
        value = state["base"]
        for term in state["terms"]:
            lo, hi = 0, len(term["edges"])
            while lo < hi:
                mid = lo + (hi - lo) // 2
                if term["edges"][mid] < row[term["feature"]]:
                    lo = mid + 1
                else:
                    hi = mid
            value += state["rate"] * term["values"][min(lo, len(term["values"]) - 1)]
        return _sigmoid(value) if state["binary"] else _finite(value)
    if kind == "Projection":
        projected = []
        for weights in state["weights"]:
            value = -0.0
            for j, weight in enumerate(weights):
                value += weight * (row[j] - state["means"][j])
            projected.append(_finite(value))
        return _clamp(_linear(state["output"], projected), -60.0, 60.0)
    if kind == "Combination":
        value = -0.0
        for weight, member in zip(state["weights"], state["members"]):
            value += weight * _scalar(member, row)
        return _clamp(value, 0.0, 1.0)
    raise ValueError("unsupported fitted scalar state")


def _predict(row):
    kind, state = next(iter(STATE["head"].items()))
    if kind == "Scalar":
        value = _scalar(state["state"], row)
        return [1.0 - value, value] if state["binary"] else [_finite(value)]
    if kind == "Regression":
        return [_finite(_clamp(_linear(state, row), -60.0, 60.0))]
    if kind == "Binary":
        probability = _sigmoid(_linear(state, row))
        return [1.0 - probability, probability]
    if kind == "Softmax":
        return _softmax([_finite(_linear(model, row)) for model in state])
    if kind == "Ovr":
        return _normalize([max(0.0, _finite(_sigmoid(_linear(model, row)))) for model in state])
    if kind == "Lda":
        return _softmax([prior - 0.5 * _distance(row, center, state["scales"], True)
                         for center, prior in zip(state["centers"], state["priors"])])
    return _softmax([-_distance(row, center, state["scales"], False) for center in state["centers"]])


def predict_encoded(rows):
    return [_predict(_encode(row)) for row in rows]

"""Private raw-table panel adapter v2; same-cut adapter v1 is unchanged.

The caller supplies training labels and prediction features only. Median imputation,
missing indicators and capped one-hot dictionaries are fitted on training rows. The
complete estimator and preprocessing state are retained in a checksummed private
pickle; never load a pickle obtained from an untrusted source.
"""
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path

CONTRACT = "jope.bench-adapter/v2"
ARTIFACT = "jope.panel-fitted-python/v1"
INPUT_POLICY = "median-onehot256-indicator/v1"
MAX_ONEHOT = 256
MAX_REQUEST_BYTES = 128 * 1024 * 1024
MAX_COLUMNS = 4096
MAX_ROWS = 1_000_000
# float64 train + prediction matrices together; model-internal allocations are
# separate and remain subject to the caller's process/resource limits.
MAX_TRANSFORMED_CELLS = 16 * 1024 * 1024


def parse_request(raw):
    """Decode one bounded JSON request with no ambiguous fields or nonfinite values."""
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("fit input must be nonempty bytes within 128 MiB")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key!r}")
            result[key] = value
        return result

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("JSON number must be finite")
        return result

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_float=finite_float,
                      parse_constant=invalid_constant)


def _matrix_budget(width, *row_counts):
    if (type(width) is not int or width <= 0
            or any(type(rows) is not int or rows < 0 for rows in row_counts)
            or width * sum(row_counts) > MAX_TRANSFORMED_CELLS):
        raise ValueError("transformed matrix exceeds the 128 MiB float64 cell budget")


def _keys(value, expected, where):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{where}: expected fields {sorted(expected)}")


def _numeric(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("raw feature must be a string or null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("numeric feature must be finite or null")
    return number


def _category(value):
    if value is None:
        return "\0MISSING"
    if not isinstance(value, str):
        raise ValueError("raw category must be a string or null")
    # Same lossless namespace as panel/data.rs, including literal missing-like strings.
    return "jope.category.v1:" + json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_request(request):
    _keys(request, ("contract", "task", "columns", "classes", "train", "predict"), "request")
    if request["contract"] != CONTRACT:
        raise ValueError("unsupported adapter contract")
    task = request["task"]
    classes = request["classes"]
    if task not in ("regression", "binary", "multiclass"):
        raise ValueError("unsupported panel task")
    if not isinstance(classes, list) or any(not isinstance(c, str) or not c for c in classes):
        raise ValueError("class vocabulary must contain original string labels")
    if len(set(classes)) != len(classes):
        raise ValueError("duplicate class label")
    if not ((task == "regression" and not classes) or (task == "binary" and len(classes) == 2)
            or (task == "multiclass" and len(classes) >= 3)):
        raise ValueError("class vocabulary is incompatible with task")
    columns = request["columns"]
    if not isinstance(columns, list) or not 0 < len(columns) <= MAX_COLUMNS:
        raise ValueError("feature column count must be within 1..4096")
    names = set()
    for column in columns:
        _keys(column, ("name", "kind"), "column")
        name = column["name"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("empty or duplicate feature name")
        names.add(name)
        if column["kind"] not in ("numeric", "categorical"):
            raise ValueError("unsupported column kind")
    _keys(request["train"], ("features", "targets"), "train")
    _keys(request["predict"], ("features",), "predict")
    for part in ("train", "predict"):
        rows = request[part]["features"]
        if not isinstance(rows, list) or not 0 < len(rows) <= MAX_ROWS:
            raise ValueError(f"{part}: row count must be within 1..1000000")
        _matrix_budget(len(columns), len(rows))
        for row in rows:
            if not isinstance(row, list) or len(row) != len(columns):
                raise ValueError(f"{part}: feature width mismatch")
            for column, value in zip(columns, row):
                (_numeric if column["kind"] == "numeric" else _category)(value)
    labels = request["train"]["targets"]
    if not isinstance(labels, list) or len(labels) != len(request["train"]["features"]):
        raise ValueError("training target length mismatch")
    if any(not isinstance(label, str) for label in labels):
        raise ValueError("training labels must be original strings")
    if task == "regression":
        for label in labels:
            _numeric(label)
    elif set(labels) != set(classes):
        raise ValueError("training labels differ from declared class vocabulary")


def fit_preprocessing(columns, rows):
    """Retain full dictionaries; select the same max256 count-ranked levels as v3 input."""
    fitted = []
    width = 0
    for j, column in enumerate(columns):
        if column["kind"] == "numeric":
            finite = sorted((value for row in rows if (value := _numeric(row[j])) is not None),
                            key=lambda value: (value, math.copysign(1.0, value)))
            if finite:
                # Match feat_math::ops::quantile_sorted, including odd subnormal medians.
                position = 0.5 * (len(finite) - 1)
                lo = math.floor(position)
                hi = min(lo + 1, len(finite) - 1)
                fraction = position - lo
                fill = finite[lo] * (1.0 - fraction) + finite[hi] * fraction
            else:
                fill = 0.0
            fitted.append({"kind": "numeric", "fill": fill})
            width += 2
        else:
            counts = {}
            for row in rows:
                key = _category(row[j])
                counts[key] = counts.get(key, 0) + 1
            vocabulary = sorted(counts)
            selected = sorted(sorted(vocabulary, key=lambda key: (-counts[key], key))[:MAX_ONEHOT])
            fitted.append({"kind": "categorical", "vocabulary": vocabulary,
                           "counts": [counts[key] for key in vocabulary], "selected": selected})
            width += len(selected)
    return {"policy": INPUT_POLICY, "columns": columns, "fitted": fitted, "width": width,
            "unknown_category": "all-zero-onehot", "max_onehot": MAX_ONEHOT, "min_count": 1}


def transform(state, rows):
    import numpy as np

    if state["policy"] != INPUT_POLICY:
        raise ValueError("unsupported preprocessing policy")
    _matrix_budget(state["width"], len(rows))
    result = np.zeros((len(rows), state["width"]), dtype=np.float64)
    offset = 0
    for j, column in enumerate(state["fitted"]):
        if column["kind"] == "numeric":
            for i, row in enumerate(rows):
                value = _numeric(row[j])
                result[i, offset] = column["fill"] if value is None else value
                result[i, offset + 1] = float(value is None)
            offset += 2
        else:
            index = {key: k for k, key in enumerate(column["selected"])}
            for i, row in enumerate(rows):
                found = index.get(_category(row[j]))
                if found is not None:
                    result[i, offset + found] = 1.0
            offset += len(index)
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite transformed feature")
    return result


def predictions(model, x, task, classes):
    import numpy as np

    if task == "regression":
        values = np.asarray(model.predict(x), dtype=np.float64).reshape(-1)
        if values.shape != (len(x),) or not np.all(np.isfinite(values)):
            raise ValueError("malformed regression predictions")
        return {"kind": "values", "data": values.tolist()}
    _matrix_budget(len(classes), len(x))
    probabilities = np.asarray(model.predict_proba(x), dtype=np.float64)
    model_classes = np.asarray(model.classes_).reshape(-1).tolist()
    if len(model_classes) != len(classes) or set(model_classes) != set(range(len(classes))):
        raise ValueError("fitted model class columns differ from declared vocabulary")
    if probabilities.shape != (len(x), len(classes)):
        raise ValueError("malformed probability matrix")
    probabilities = probabilities[:, [model_classes.index(i) for i in range(len(classes))]]
    if (not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
            or not np.all(np.abs(probabilities.sum(axis=1) - 1.0) <= 1e-6)):
        raise ValueError("invalid probabilities")
    return {"kind": "probabilities", "data": {"classes": list(classes), "values": probabilities.tolist()}}


def _load_trusted_artifact(artifact_bytes):
    """Load the private artifact emitted by this invocation's fit call."""
    return pickle.loads(artifact_bytes)


def _replay_trusted_artifact(artifact_bytes, expected_sha256, rows):
    """Replay an internally produced artifact after verifying its complete digest.

    The pin must come from the caller's trusted custody record, never from an
    untrusted artifact's accompanying JSON. SHA-256 provides integrity, not
    authenticity; Python estimator pickle is executable and is not a sandbox.
    Only fit calls this helper, on bytes just produced by the same invocation.
    """
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("fitted artifact SHA-256 pin is missing or invalid")
    # Snapshot mutable byte buffers before hashing and deserializing the same bytes.
    artifact_bytes = bytes(artifact_bytes)
    if hashlib.sha256(artifact_bytes).hexdigest() != expected_sha256:
        raise ValueError("fitted artifact checksum mismatch")
    state = _load_trusted_artifact(artifact_bytes)
    if state["schema"] != ARTIFACT:
        raise ValueError("unsupported fitted artifact")
    if any(len(row) != len(state["preprocessing"]["columns"]) for row in rows):
        raise ValueError("prediction feature width mismatch")
    return predictions(state["model"], transform(state["preprocessing"], rows), state["task"], state["classes"])


def fit(request, create_model, engine_version):
    import numpy as np

    validate_request(request)
    preprocessing = fit_preprocessing(request["columns"], request["train"]["features"])
    _matrix_budget(preprocessing["width"], len(request["train"]["features"]),
                   len(request["predict"]["features"]))
    if request["task"] != "regression":
        _matrix_budget(len(request["classes"]), len(request["predict"]["features"]))
    x = transform(preprocessing, request["train"]["features"])
    labels = request["train"]["targets"]
    classes = request["classes"]
    if request["task"] == "regression":
        y = np.asarray([float(value) for value in labels], dtype=np.float64)
    else:
        codes = {label: i for i, label in enumerate(classes)}
        y = np.asarray([codes[label] for label in labels], dtype=np.int64)
    model = create_model(request["task"])
    model.fit(x, y)
    output = predictions(model, transform(preprocessing, request["predict"]["features"]), request["task"], classes)
    artifact = pickle.dumps({"schema": ARTIFACT, "engine_version": engine_version,
                             "task": request["task"], "classes": list(classes),
                             "preprocessing": preprocessing, "model": model}, protocol=5)
    checksum = hashlib.sha256(artifact).hexdigest()
    if _replay_trusted_artifact(artifact, checksum, request["predict"]["features"]) != output:
        raise ValueError("fitted artifact reload changed predictions")
    return output, artifact, checksum


def read_request_bytes(path, limit=MAX_REQUEST_BYTES):
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("fit input exceeds 128 MiB")
    return data


def write_private(path, data):
    """Create one immutable private artifact; refusing a stale path preserves its evidence."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def run(create_model, version):
    """--panel-v2 request.json result.json; dependency unavailability has exit status69."""
    if len(sys.argv) != 4 or sys.argv[1] != "--panel-v2":
        sys.stderr.write("usage: <adapter> --panel-v2 <request.json> <result.json>\n")
        return 2
    try:
        request_path, result_path = map(Path, sys.argv[2:])
        artifact_path = result_path.with_name(result_path.name + ".fitted.pkl")
        if os.path.lexists(result_path) or os.path.lexists(artifact_path):
            raise ValueError("private output already exists")
        request_bytes = read_request_bytes(request_path)
        request = parse_request(request_bytes)
        engine_version = version()
        output, artifact, checksum = fit(request, create_model, engine_version)
        result = {"contract": CONTRACT, "engine_version": engine_version,
                  "input_sha256": hashlib.sha256(request_bytes).hexdigest(), "predictions": output,
                  "fitted": {"path": artifact_path.name, "sha256": checksum, "bytes": len(artifact),
                             "format": ARTIFACT, "preprocessing_policy": INPUT_POLICY, "reload_verified": True}}
        result_bytes = (json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        write_private(artifact_path, artifact)
        write_private(result_path, result_bytes)
        return 0
    except (ImportError, ModuleNotFoundError) as error:
        sys.stderr.write(f"unavailable dependency: {error}\n")
        return 69
    except Exception as error:
        sys.stderr.write(f"panel adapter failed: {error}\n")
        return 1

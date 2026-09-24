
import json
import math
import re

_RAW_CATEGORICAL = json.loads(__KINDS_JSON__)
_RAW_CLASSES = json.loads(__CLASSES_JSON__)
_RAW_REGRESSION = __REGRESSION__
_RAW_SPACE = "\u0009\u000a\u000b\u000c\u000d\u0020\u0085\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
_RAW_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")

__IMPLEMENTATION__

def predict_raw(rows):
    """Original string/null features in schema order; complete aligned predictions."""
    if not isinstance(rows, list):
        raise ValueError("rows must be an array")
    encoded = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(_RAW_CATEGORICAL):
            raise ValueError("input shape differs")
        values = []
        for value, categorical in zip(row, _RAW_CATEGORICAL):
            if value is None:
                values.append("")
            elif not isinstance(value, str):
                raise ValueError("features must be strings or null")
            elif categorical:
                value.encode("utf-8")
                values.append("jope.category.v1:" + json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            else:
                trimmed = value.strip(_RAW_SPACE)
                if not _RAW_NUMBER.fullmatch(trimmed) or not math.isfinite(float(trimmed)):
                    raise ValueError("numeric features must be finite or null")
                values.append(value)
        encoded.append(values)
    matrix = _matrix(encoded)
    width = 1 if _RAW_REGRESSION else len(_RAW_CLASSES)
    if len(matrix) != len(rows) or any(len(row) != width or any(not math.isfinite(v) for v in row) for row in matrix):
        raise ValueError("fitted output shape differs")
    if _RAW_REGRESSION:
        return {"kind": "values", "data": [row[0] for row in matrix]}
    if any(any(v < 0.0 or v > 1.0 for v in row) or abs(sum(row)-1.0) > 1e-6 for row in matrix):
        raise ValueError("invalid fitted probability matrix")
    return {"kind": "probabilities", "data": {"classes": list(_RAW_CLASSES), "values": matrix}}

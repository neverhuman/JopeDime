import ctypes
from pathlib import Path
_TYPED_CODEC = json.loads(__CODEC_JSON__)
_TYPED_LOOKUPS = [{entry["value"]: entry["index"] for entry in column["entries"]} if column["kind"] == "categorical" else None for column in _TYPED_CODEC["columns"]]

def _matrix(rows):
    values = []
    missing = []
    for row in rows:
        for raw, column, lookup in zip(row, _TYPED_CODEC["columns"], _TYPED_LOOKUPS):
            absent = raw == ""
            missing.append(int(absent))
            if column["kind"] == "numeric":
                values.append(0.0 if absent else float(raw))
            else:
                key = "\0MISSING" if absent else raw
                values.append(float(lookup.get(key, -1)))
    library = ctypes.CDLL(str(Path(__file__).resolve().with_name("model.so")))
    entry = library.jope_fitted_predict
    entry.restype = ctypes.c_int
    entry.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                     ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                     ctypes.c_size_t, ctypes.c_size_t,
                     ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    dense = (ctypes.c_double * len(values))(*values)
    mask = (ctypes.c_uint8 * len(missing))(*missing)
    output = (ctypes.c_double * len(rows))()
    status = entry(dense, len(values), mask, len(missing), len(rows), len(_RAW_CATEGORICAL), output, len(rows))
    if status != 0:
        raise ValueError("fitted library refused input: " + str(status))
    return [[value] if _RAW_REGRESSION else [1.0-value, value] for value in output]

"""Derive the three real R882 MHA projections from captured ATen operations.

Call only after the raw parameter and tensor-DAG validation gates. These records
describe actual ``aten.addmm`` operations and saved parameter views; they do not
add parameters, replace raw observations, or claim additional nn.Linear hooks.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

WEIGHT_ID = "base.mm_cross_attn.in_proj_weight"
BIAS_ID = "base.mm_cross_attn.in_proj_bias"
SCHEMA = "jain.h256-stage2-capture/v1"


def _require(condition, message):
    if not condition:
        raise ValueError("MHA functional projection: " + message)


def _one(records, message):
    _require(len(records) == 1, message)
    return records[0]


def _f32(record, shape=None):
    _require(record.get("dtype") == "float32", "tensor/parameter must be float32")
    actual_shape = record["shape"]
    if shape is not None:
        _require(actual_shape == list(shape), "unexpected tensor/parameter shape")
    _require(isinstance(actual_shape, list) and all(type(v) is int and v > 0 for v in actual_shape),
             "invalid matrix/vector shape")
    value = np.asarray(record["values"], dtype=np.float32)
    _require(value.size == int(np.prod(actual_shape)) and bool(np.isfinite(value).all()),
             "invalid numeric cells")
    return value.reshape(actual_shape)


def _bytes(value):
    return np.ascontiguousarray(value, dtype="<f4").tobytes()


def _sha(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def derived_mha_affines(data, case):
    """Return three legacy-like functional records and their per-case proof receipt.

    Q/K/V use rows [0:32], [32:64], [64:96] of the *existing* saved joint
    in-projection parameter. The checker follows the captured split -> transpose
    -> addmm graph, checks parameter-view bytes, and checks x @ W.T + b closure.
    No argument or raw trace list is mutated.
    """
    _require(data.get("schema") == SCHEMA, "wrong raw trace schema")
    parameters = {v["id"]: v for v in data["parameters"]}
    tensors = {v["id"]: v for v in case["tensors"]}
    operations = {v["id"]: v for v in case["operations"]}
    _require(len(parameters) == len(data["parameters"]), "duplicate parameter identity")
    _require(len(tensors) == len(case["tensors"]), "duplicate tensor identity")
    _require(len(operations) == len(case["operations"]), "duplicate operation identity")
    _require(WEIGHT_ID in parameters and BIAS_ID in parameters, "joint saved parameters missing")
    weight_parameter, bias_parameter = parameters[WEIGHT_ID], parameters[BIAS_ID]
    _require(weight_parameter.get("role") == bias_parameter.get("role") == "solution_encoder",
             "wrong saved parameter role")
    full_weight = _f32(weight_parameter, [96, 32])
    full_bias = _f32(bias_parameter, [96])
    _require(bool(weight_parameter.get("checkpoint_origin")) and bool(bias_parameter.get("checkpoint_origin")),
             "joint parameters lack saved origins")
    weight_split = _one([v for v in case["operations"] if v["kind"] == "aten.split.Tensor"
                         and v["parameter_ids"] == [WEIGHT_ID]], "missing/ambiguous joint weight split")
    bias_split = _one([v for v in case["operations"] if v["kind"] == "aten.split.Tensor"
                       and v["parameter_ids"] == [BIAS_ID]], "missing/ambiguous joint bias split")
    _require(len(weight_split["output_tensor_ids"]) == len(bias_split["output_tensor_ids"]) == 3,
             "joint parameter split must have three outputs")
    records, proofs = [], []
    for index, label in enumerate(("Q", "K", "V")):
        start, stop = index * 32, (index + 1) * 32
        weight_tid = weight_split["output_tensor_ids"][index]
        bias_tid = bias_split["output_tensor_ids"][index]
        _require(weight_tid in tensors and bias_tid in tensors, "split tensor missing")
        _require(tensors[weight_tid]["producer_op_id"] == weight_split["id"]
                 and tensors[bias_tid]["producer_op_id"] == bias_split["id"], "split producer differs")
        weight = _f32(tensors[weight_tid], [32, 32])
        bias = _f32(tensors[bias_tid], [32])
        _require(_bytes(weight) == _bytes(full_weight[start:stop])
                 and _bytes(bias) == _bytes(full_bias[start:stop]), "saved parameter row view differs")
        transpose = _one([v for v in case["operations"] if v["kind"] == "aten.t.default"
                          and v["input_tensor_ids"] == [weight_tid]], "missing/ambiguous weight transpose")
        _require(len(transpose["output_tensor_ids"]) == 1
                 and weight_split["id"] in transpose["parent_op_ids"], "transpose lineage differs")
        transpose_tid = transpose["output_tensor_ids"][0]
        _require(transpose_tid in tensors and tensors[transpose_tid]["producer_op_id"] == transpose["id"],
                 "transposed weight producer differs")
        transposed = _f32(tensors[transpose_tid], [32, 32])
        _require(_bytes(transposed) == _bytes(weight.T), "captured matmul weight is not the actual transpose")
        addmm = _one([v for v in case["operations"] if v["kind"] == "aten.addmm.default"
                      and len(v["input_tensor_ids"]) == 3 and v["input_tensor_ids"][0] == bias_tid
                      and v["input_tensor_ids"][2] == transpose_tid], "missing/ambiguous actual addmm")
        _require(addmm.get("role") == "solution_encoder"
                 and addmm.get("observation") == "native-functional-dispatch", "addmm is not the captured solution operation")
        _require(transpose["id"] in addmm["parent_op_ids"] and bias_split["id"] in addmm["parent_op_ids"],
                 "addmm does not consume the actual parameter views")
        input_tid = addmm["input_tensor_ids"][1]
        _require(input_tid in tensors and len(addmm["output_tensor_ids"]) == 1, "addmm operands missing")
        output_tid = addmm["output_tensor_ids"][0]
        _require(output_tid in tensors and tensors[output_tid]["producer_op_id"] == addmm["id"],
                 "addmm output producer differs")
        x, y = _f32(tensors[input_tid]), _f32(tensors[output_tid])
        _require(x.ndim == y.ndim == 2 and x.shape[1] == 32 and y.shape == (x.shape[0], 32),
                 "actual addmm activation shape differs")
        x64, w64, b64, y64 = (v.astype(np.float64) for v in (x, weight, bias, y))
        denominator = np.abs(x64) @ np.abs(w64).T + np.abs(b64) + 1e-9
        error = float(np.max(np.abs(x64 @ w64.T + b64 - y64) / denominator))
        _require(error <= 1e-5, "actual x @ W.T + b does not close")
        name = "base.mm_cross_attn." + label
        slices = {"weight": {"parameter_id": WEIGHT_ID, "axis": 0, "start": start, "stop": stop},
                  "bias": {"parameter_id": BIAS_ID, "axis": 0, "start": start, "stop": stop}}
        record = {"network": "base", "role": "solution_encoder", "name": name,
            "functional_kind": "captured aten.addmm; saved joint-parameter row views",
            "operation_id": addmm["id"], "parameter_ids": [WEIGHT_ID, BIAS_ID],
            "source_parameter_slices": slices, "source_weight_row_slice": [start, stop],
            "source_bias_row_slice": [start, stop], "weight_layout": "out,in",
            "input_shape": list(x.shape), "weight_shape": [32, 32], "output_shape": list(y.shape),
            "input": x.reshape(-1).tolist(), "weights": weight.reshape(-1).tolist(),
            "bias": bias.tolist(), "output": y.reshape(-1).tolist(),
            "execution_scope": addmm.get("execution_scope"), "call_source": addmm.get("call_source")}
        records.append(record)
        proofs.append({"name": name, "operation_id": addmm["id"], "functional_kind": record["functional_kind"],
            "input_tensor_id": input_tid, "weight_tensor_id": weight_tid, "bias_tensor_id": bias_tid,
            "matmul_weight_tensor_id": transpose_tid, "output_tensor_id": output_tid,
            "input_shape": list(x.shape), "weight_shape": [32, 32], "output_shape": list(y.shape),
            "source_parameter_ids": [WEIGHT_ID, BIAS_ID], "source_parameter_slices": slices,
            "source_checkpoint_origins": [weight_parameter["checkpoint_origin"], bias_parameter["checkpoint_origin"]],
            "parameter_split_operation_ids": [weight_split["id"], bias_split["id"]],
            "weight_transpose_operation_id": transpose["id"], "weight_view_sha256": _sha(weight),
            "bias_view_sha256": _sha(bias), "input_sha256": _sha(x), "output_sha256": _sha(y),
            "saved_parameter_view_exact": True, "max_normalized_affine_error": error})
    receipt = {"schema": "jain.h256-derived-mha-affines/v1", "status": "PASS", "case_id": case["id"],
        "matrix_count": 3, "canonical_parameter_count_unchanged": len(data["parameters"]),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "matrices": proofs,
        "scope": "Three actual captured ATen projections; raw trace and canonical parameter inventory are unchanged"}
    return records, receipt

#!/usr/bin/env python3
"""Verify the accepted policy tensor inventory (W2-T06).

Checks the safetensors header against the documented native geometry, the payload layout
(contiguous, non-overlapping, no trailing bytes, byte lengths equal to shapes), every tensor's
payload digest, and a role accounting that partitions the tensors into decode, rank and
training-only roles. The Candle forward in crates/jope-policy/src/model.rs must reference exactly
the inference roles. An optional optimizer archive shows which stored parameters carry AdamW
moments. Scope: the stored object and its documented roles; not a forward-pass, parity or
training qualification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
RECORD = ROOT / "data/tensors.json"
MODEL_SOURCE = REPO / "crates/jope-policy/src/model.rs"
DTYPE_BYTES = {"F32": 4}
EXPECTED_PARAMETERS = 27796494
EXPECTED_TENSORS = 136
ROLES = {
    "decode": ("encoder", "ctx_proj", "token_emb", "pos_emb", "decoder", "out_norm"),
    "rank": ("reward_encoder", "reward_token_emb", "reward_norm", "z_gamma", "z_proj",
             "reward_heads", "runtime_head"),
    "training_only": ("stability_head", "reward_head_ablation", "loss_log_vars"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_geometry(source, width=512, embedding_rows=657, positions=129, layers=6,
                      ffn=2048, context=2048, heads=4):
    """Documented mirror of model.rs expected_inventory for the step-60000 configuration."""
    expected = {}

    def dense(name, inputs, outputs):
        expected[name + ".weight"] = [outputs, inputs]
        expected[name + ".bias"] = [outputs]

    def norm(name):
        expected[name + ".weight"] = [width]
        expected[name + ".bias"] = [width]

    for name in ("encoder", "reward_encoder"):
        dense(name + ".0", context, width)
        norm(name + ".2")
        dense(name + ".4", width, width)
        norm(name + ".6")
    dense("ctx_proj", width, width)
    expected["token_emb.weight"] = [embedding_rows, width]
    expected["pos_emb.weight"] = [positions, width]
    for i in range(layers):
        p = f"decoder.layers.{i}"
        expected[p + ".self_attn.in_proj_weight"] = [3 * width, width]
        expected[p + ".self_attn.in_proj_bias"] = [3 * width]
        dense(p + ".self_attn.out_proj", width, width)
        dense(p + ".linear1", width, ffn)
        dense(p + ".linear2", ffn, width)
        norm(p + ".norm1")
        norm(p + ".norm2")
    norm("out_norm")
    expected["reward_token_emb.weight"] = [embedding_rows, width]
    norm("reward_norm")
    dense("z_gamma.0", width, width)
    dense("z_proj", width, width)
    for i in range(heads):
        dense(f"reward_heads.{i}.0", 3 * width, width)
        dense(f"reward_heads.{i}.3", width, 1)
    for name, inputs, index in [("runtime_head", 2 * width, 2), ("stability_head", 3 * width, 2),
                                ("reward_head_ablation", width, 3)]:
        dense(name + ".0", inputs, width)
        dense(name + f".{index}", width, 1)
    block = source.split("const LOSS_LOG_VARS:", 1)[1].split("];", 1)[0]
    for name in re.findall(r'"([a-z_]+)"', block):
        expected["loss_log_vars." + name] = []
    return expected


def parse_safetensors(raw):
    """Header and payload; every byte of the payload belongs to exactly one tensor."""
    require(len(raw) >= 8, "truncated header length")
    size = struct.unpack("<Q", raw[:8])[0]
    require(8 + size <= len(raw), "truncated header")
    header = json.loads(raw[8:8 + size])
    header.pop("__metadata__", None)
    payload = raw[8 + size:]
    spans = []
    for name, entry in header.items():
        require(set(entry) == {"dtype", "shape", "data_offsets"}, f"{name}: header fields")
        require(entry["dtype"] in DTYPE_BYTES, f"{name}: dtype {entry['dtype']}")
        begin, end = entry["data_offsets"]
        require(0 <= begin <= end <= len(payload), f"{name}: offsets outside payload")
        require(end - begin == math.prod(entry["shape"]) * DTYPE_BYTES[entry["dtype"]],
                f"{name}: byte length differs from shape")
        spans.append((begin, end, name))
    cursor = 0
    for begin, end, name in sorted(spans):
        require(begin == cursor, f"{name}: payload gap or overlap at byte {begin}")
        cursor = end
    require(cursor == len(payload), "trailing payload bytes")
    return header, payload


def role_of(name):
    root = name.split(".", 1)[0]
    roles = [role for role, roots in ROLES.items() if root in roots]
    require(len(roles) == 1, f"{name}: no single role")
    return roles[0]


def forward_roots(source):
    """Tensor roots the Candle inference methods reference (string literals only)."""
    region = source.split("fn encode_context(", 1)[1].split('#[cfg(feature = "inference-cuda")]', 1)[0]
    return set(re.findall(r'"([a-z_]+)(?=[."])', region))


def optimizer_moment_bytes(path):
    """Payload bytes of the torch archive's tensor entries (AdamW moments m and v)."""
    with zipfile.ZipFile(path) as archive:
        return sum(info.file_size for info in archive.infolist() if "/data/" in info.filename)


def verify(record, raw, source, optimizer=None):
    require(hashlib.sha256(raw).hexdigest() == record["sha256"], "asset digest differs from record")
    header, payload = parse_safetensors(raw)
    actual = {name: entry["shape"] for name, entry in header.items()}
    require(all(entry["dtype"] == "F32" for entry in header.values()), "non-F32 tensor")
    require(actual == expected_geometry(source), "header differs from documented expected_inventory geometry")
    tensors = {t["name"]: t for t in record["tensors"]}
    require(set(tensors) == set(actual), "record tensor names differ from header")
    parameters = {"stored": 0, **{role: 0 for role in ROLES}}
    for name, entry in header.items():
        begin, end = entry["data_offsets"]
        digest = hashlib.sha256(payload[begin:end]).hexdigest()
        item = tensors[name]
        require(item["shape"] == entry["shape"], f"{name}: record shape")
        require(item["payload_sha256"] == digest, f"{name}: payload digest differs")
        role = role_of(name)
        require(item["role"] == role, f"{name}: record role")
        count = math.prod(entry["shape"])
        require(item["parameters"] == count, f"{name}: record parameters")
        parameters["stored"] += count
        parameters[role] += count
    require(parameters["stored"] == record["parameters"] == EXPECTED_PARAMETERS, "stored parameter total")
    require(len(actual) == record["tensor_count"] == EXPECTED_TENSORS, "tensor count")
    require(record["role_parameters"] == {role: parameters[role] for role in ROLES},
            "record role accounting")
    referenced = forward_roots(source) & {name.split(".", 1)[0] for name in actual}
    inference = set(ROLES["decode"]) | set(ROLES["rank"])
    require(referenced == inference,
            f"Candle forward roots {sorted(referenced)} differ from inference roles {sorted(inference)}")
    summary = {
        "tensor_count": len(actual), "parameters": parameters, "dtype": "F32",
        "asset_sha256": record["sha256"], "payload_layout": "contiguous, no gaps, overlaps or trailing bytes",
        "per_tensor_payload_digests": "match", "forward_references": sorted(referenced),
        "scope": "stored object and documented roles; not a forward-pass, parity or training qualification",
    }
    if optimizer is not None:
        moments = optimizer_moment_bytes(optimizer)
        require(moments == 2 * 4 * parameters["stored"],
                f"optimizer moment bytes {moments} are not two FP32 moments per stored parameter")
        summary["trainable"] = {"source": str(optimizer), "moment_bytes": moments,
                                "all_stored_parameters_have_adamw_moments": True}
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--optimizer", type=Path, help="optional torch optimizer archive of the same checkpoint")
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    print(json.dumps(verify(record, Path(record["path"]).read_bytes(), MODEL_SOURCE.read_text(),
                            args.optimizer), sort_keys=True))

#!/usr/bin/env python3
"""Reference parity for actual pointer views, including Rust preprocessing."""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

r = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(r / "assets/source/target_maker"))
sys.path.insert(0, str(r / "assets/source/fable_god"))
os.environ["FABLE_REGRESSOR_SOURCE"] = str(r / "assets/source/fable_god/provenance/regressor_source")
from python.elo_finetune.model_io import load_regressor
from target_maker.fused import FusedTargetEncoder

torch.set_num_threads(2)
torch._C._set_graph_executor_optimize(False)
torch.backends.cuda.matmul.allow_tf32 = False
f = load_regressor(
    r / "assets/models/starlight/starlight_core_regressor.safetensors",
    r / "assets/models/starlight/starlight_config.json",
    "cuda",
)
for module in f.modules():
    if hasattr(module, "num_kv_heads_test"):
        module.num_kv_heads_test = None
target = FusedTargetEncoder(r / "assets/models/target-fused", device="cuda")
results = []
for path in sorted((r / "data/prepared/views-expanded-20").glob("*/prepared.json")):
    m = json.loads(path.read_text())
    c = m["cuts"][0]
    s = m["storage"]
    pp = json.loads((path.parent / "cut-0/preprocessing.json").read_text())
    source = list(csv.DictReader((r / s["source"]).open()))
    selected = [source[s["original_training_row_map"][i]] for i in c["train_indices"]]
    X = np.empty((len(selected), len(s["feature_columns"])), dtype="float64")
    for i, row in enumerate(selected):
        for j, name in enumerate(s["feature_columns"]):
            try:
                v = float(row[name])
            except ValueError:
                v = float("nan")
            if not np.isfinite(v):
                v = pp["columns"][j]["impute"]
            X[i, j] = (v - pp["columns"][j]["mean"]) / pp["columns"][j]["scale"]
    y = (np.array([float(row[s["target_column"]]) for row in selected]) - pp["target_mean"]) / pp[
        "target_scale"
    ]
    target_y = ((y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y * 0).astype("float32")
    tv = target.encode_values(target_y, 1)
    rust_t = np.fromfile(r / "data/encodings/target-cuts" / m["dataset"] / "cut-0/target.f32", dtype="<f4")
    terr = float(np.abs(tv - rust_t).max())
    lo = X.min(0)
    span = X.max(0) - lo
    span = np.where(span > 1e-12, span, 1)
    X = ((X - lo) / span).astype("float32")
    y = y.astype("float32")
    idx = np.random.default_rng(20260707).permutation(len(y))
    nc = min(768, max(8, 3 * len(y) // 4))
    nq = min(256, len(y) - nc)
    ctx = idx[:nc]
    qry = idx[max(0, len(y) - 8) :] if nq < 4 else idx[nc : nc + nq]
    cy = y[ctx]
    cy = (cy - float(cy.mean())) / (float(cy.std()) + 1e-8)
    with torch.no_grad():
        x = torch.from_numpy(np.concatenate([X[ctx], X[qry]])).cuda()[:, None, :]
        cy = torch.from_numpy(cy).cuda()[:, None]
        h = f(x, cy, only_return_standard_out=False)["test_embeddings"].float().cpu().numpy()[:, 0]
    fv = np.concatenate(
        [h.mean(0), h.std(0), np.percentile(h, 10, axis=0), np.percentile(h, 90, axis=0)]
    ).astype("float32")
    rust_f = np.fromfile(
        r / "data/encodings/foundation-cuts" / m["dataset"] / "cut-0/foundation_raw.f32", dtype="<f4"
    )
    ferr = float(np.abs(fv - rust_f).max())
    result = dict(
        dataset=m["dataset"],
        cut=0,
        training_sha256=c["training_sha256"],
        target_max_abs_error=terr,
        foundation_max_abs_error=ferr,
        passed=max(terr, ferr) < 3e-4,
    )
    results.append(result)
    print(json.dumps(result), flush=True)
report = dict(
    schema="parallel.real_cut_component_parity.v1",
    passed=all(x["passed"] for x in results),
    tolerance=3e-4,
    cuts=results,
)
(r / "runs/parity/component-views.json").write_text(json.dumps(report, indent=2) + "\n")
assert report["passed"]

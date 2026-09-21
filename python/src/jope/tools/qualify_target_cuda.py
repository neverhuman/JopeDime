#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from export_target import FusedGraph

r = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(r / "assets/source/target_maker"))
from target_maker.fused import FusedTargetEncoder

original = r / "assets/target-export-v1"
out = r / "assets/target-export-cuda-v1"
out.mkdir(parents=True, exist_ok=True)
# Imported bundle classes require the matching acquisition environment only for
# this one-time conversion, never for the Rust service.
enc = FusedTargetEncoder(r / "assets/models/target-fused", device="cpu")
graph = FusedGraph(enc).eval().cuda()
torch._C._set_graph_executor_optimize(False)
torch.set_num_threads(2)
torch.backends.mha.set_fastpath_enabled(False)
d = np.load(original / "parity-inputs.npz")
inputs = [torch.from_numpy(d[k]).cuda() for k in ["classical", "neural", "lengths", "task"]]
with torch.no_grad():
    if "--reuse-export" in sys.argv:
        traced = torch.jit.load(str(out / "target.pt"), map_location="cuda:0").eval()
    else:
        traced = torch.jit.trace(graph, tuple(t[:2] for t in inputs), check_trace=False)
        traced = torch.jit.freeze(traced)
        traced.save(str(out / "target.pt"))
    actual = traced(*inputs).cpu().numpy()
    error = float(np.abs(actual - d["reference"]).max())
    batch_errors = [
        float(np.abs(traced(*(t[:n] for t in inputs)).cpu().numpy() - actual[:n]).max()) for n in [1, 7, 36]
    ]
(out / "sampling.json").write_bytes((original / "sampling.json").read_bytes())
report = dict(
    schema="parallel.target_cuda_parity.v1",
    passed=error < 3e-4,
    max_abs_error=error,
    tolerance=3e-4,
    module_sha256=hashlib.sha256((out / "target.pt").read_bytes()).hexdigest(),
    sampling_sha256=hashlib.sha256((out / "sampling.json").read_bytes()).hexdigest(),
    reference_parity_sha256=hashlib.sha256((original / "parity.json").read_bytes()).hexdigest(),
    export_source_sha256=hashlib.sha256((r / "tools/export_target.py").read_bytes()).hexdigest(),
    numerical_backend="LibTorch 2.7 CUDA sm80 FP32",
    cases=len(actual),
    dynamic_batch_errors=batch_errors,
    graph_executor_optimize=False,
)
(out / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
assert report["passed"]

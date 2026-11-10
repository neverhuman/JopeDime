"""Private H297 CPU runtime, preserving t1845 scorer and frozen dependencies."""

from pathlib import Path
import os
import sys
import json
import hashlib
import importlib.util
import time
import datetime
import numpy as np
import torch

ROOT = Path(os.environ.get("H297_ROOT", "/home/ubuntu/jope-data/h297-astra"))
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
BASE = 1.1389572403748425
IDENTITY = "e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available()
sys.path.insert(0, "/home/ubuntu/JopeDimeMicro")
sys.path.insert(0, "/home/ubuntu/JopeDimeMicro/python")
sys.path.insert(0, str(ROOT / "closure"))
import model  # noqa: E402, F401 -- preload pinned private module
import phase12_multimodal_model  # noqa: E402, F401 -- preload pinned private module
import rust_evaluator as evaluator  # noqa: E402
from dev_incumbent import warm_restart_adapter, batch_seed  # noqa: E402, F401


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


guard = module("h297_dope_guard", ROOT / "closure/dope_freeze_guard.py")
native = module("jope._telemetry", next((ROOT / "closure").glob("_telemetry*.so")))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def event(kind, **kw):
    rec = dict(
        epoch=time.time(),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        event=kind,
        **kw,
    )
    with (ROOT / "events.jsonl").open("a") as f:
        f.write(json.dumps(rec, allow_nan=False) + "\n")
    print(json.dumps(rec), flush=True)


def state_sha(values):
    h = hashlib.sha256()
    for name in sorted(values):
        t = values[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(np.asarray(t.shape, dtype=np.int64).tobytes())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def verify_inputs():
    manifest = json.loads((ROOT / "input-copy-manifest.json").read_text())
    for r in manifest["files"]:
        assert sha(ROOT / r["path"]) == r["sha256"], r["path"]
    selected = json.loads((ROOT / "input/selected.json").read_text())
    assert len(selected["train"]) == 1219 and len(selected["dev"]) == 91
    contract = dict(
        protocol="fixed-dev-cost-argmin-negative-fitness/v1",
        selected_packet_sha256=sha(ROOT / "input/selected.json"),
        dope_sha256=DOPE,
        source_sha256=sha(ROOT / "input/cpu_exact_fused.py"),
        model_source_sha256=sha(ROOT / "closure/model.py"),
        adapter_source_sha256=sha(ROOT / "closure/phase12_multimodal_model.py"),
        contexts=[
            dict(
                cid=r["context_sha256"],
                pool_sha256=r["pool_sha256"],
                training_table_sha256=r["train_only_jtt1_sha256"],
            )
            for r in selected["dev"]
        ],
    )
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    assert identity == IDENTITY
    return selected, contract


def build():
    fused = module("h297_fused", ROOT / "input/cpu_exact_fused.py")
    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    for p in net.parameters():
        p.requires_grad_(False)
    for n, p in net.adapter_10tok.named_parameters():
        if not n.startswith("anchor."):
            p.requires_grad_(True)
    guard.lock_dope_weights_and_eval_mode(net)
    path = ROOT / "checkpoints/round-3175.pt"
    assert (
        sha(path) == "1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b"
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["state_dict"], strict=True)
    assert (
        ckpt["dope_sha256"] == DOPE
        and guard.snapshot_dope_weights(net).composite_sha256 == DOPE
    )
    assert sum(p.numel() for p in net.parameters() if p.requires_grad) == 29211
    return net, ckpt


def load_pools(rows):
    pools = {}
    for r in rows:
        c = r["context_sha256"]
        path = ROOT / "input/pools" / (c + ".npz")
        assert sha(path) == r["pool_sha256"]
        with np.load(path, allow_pickle=False) as z:
            prog = z["program"].astype(np.float32)
            p = dict(
                rec=r,
                program=prog,
                pids=[
                    v.decode() if isinstance(v, bytes) else str(v)
                    for v in z["program_sha256"].tolist()
                ],
                fitness=z["fitness"].astype(np.float64),
                gate=z["gate"].astype(bool),
            )
        p["proj16"] = (
            prog.astype(np.float64) @ evaluator.get_w1_sign_matrix() / np.sqrt(412)
        )
        pools[c] = p
    return pools


def training_indices(pool, slate_size=192):
    # Exact t1845 automatic_promotion.training_indices policy.
    order = np.argsort(pool["fitness"])
    k = min(48, slate_size, len(order))
    rest_pool = order[k:] if len(order) > k else order
    return np.concatenate(
        [
            order[:k],
            np.random.choice(
                rest_pool, slate_size - k, replace=len(rest_pool) < slate_size - k
            ),
        ]
    )


def predict(net, pools, tokens, cids, indexes):
    prog = np.stack([pools[c]["program"][ix] for c, ix in zip(cids, indexes)])
    proj = np.stack([pools[c]["proj16"][ix] for c, ix in zip(cids, indexes)])
    b, m = prog.shape[:2]
    histf = np.stack(
        [pools[c]["fitness"][ix[: min(4, m)]] for c, ix in zip(cids, indexes)]
    )
    histg = np.stack(
        [pools[c]["gate"][ix[: min(4, m)]] for c, ix in zip(cids, indexes)]
    )
    return net.adapter_10tok.predict_costs(
        ctx_feat=torch.zeros((b, 584)),
        prog_412=torch.as_tensor(prog),
        proj_16=torch.as_tensor(proj),
        base_prior_cost=torch.zeros((b, m)),
        scale=torch.full((b,), 0.4208),
        sos_prior=torch.zeros((b, m, 2)),
        mm_tokens_10x32=torch.stack([tokens[c]["token"] for c in cids]),
        mm_mask_10=torch.stack([tokens[c]["mask"] for c in cids]),
        revealed_indices=torch.arange(min(4, m)).reshape(1, -1).expand(b, -1),
        revealed_raw_f=torch.as_tensor(histf, dtype=torch.float32),
        revealed_gate=torch.as_tensor(histg, dtype=torch.float32),
    )


def reporter(label):
    d = ROOT / "scorer" / label
    d.mkdir(exist_ok=False)
    return native.FullFlywheelReporter(
        ROOT / "input/selected.json", d, d / "history.jsonl"
    )


def evaluate(net, rows, pools, tokens, report):
    net.eval()
    result = []
    for r in rows:
        c = r["context_sha256"]
        p = pools[c]
        with torch.no_grad():
            cost = (
                predict(net, pools, tokens, [c], [np.arange(len(p["pids"]))])[
                    "total_cost"
                ][0]
                .float()
                .numpy()
                .copy()
            )
        assert np.isfinite(cost).all()
        j = int(np.argmin(cost))
        result.append(
            dict(
                cid=c,
                family=r["family_id"],
                choice=j,
                selected_program=p["pids"][j],
                utility=-float(p["fitness"][j]) if p["gate"][j] else -2.0,
                fitness=float(p["fitness"][j]),
                gate=bool(p["gate"][j]),
            )
        )
    u = report.utility(
        [r["cid"] for r in result],
        [r["fitness"] for r in result],
        [r["gate"] for r in result],
        [r["selected_program"] for r in result],
        "dev",
    )
    return dict(
        mean=u["mean"],
        selected_programs=[r["selected_program"] for r in result],
        per_context=[r["utility"] for r in result],
        rows=result,
        canonical_utility=u,
    )

"""One frozen H138 inference/readout stage; no optimizer or gradient operation."""

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from unittest.mock import patch
import numpy as np
from science import ARMS, ROUNDS, folds, primary, secondary
from model_ops import average, tensor_hash, verify_dope

HERE = Path(__file__).resolve().parent
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def dependencies(c):
    manifest = json.loads((HERE / "manifest.json").read_text())
    repo = HERE.parents[2]
    for p, h in manifest["source_sha256"].items():
        if sha(repo / p) != h:
            raise RuntimeError("source pin mismatch: " + p)
    h = load(
        "h138_h124", repo / "research/flywheel_selfplay/h124_gbm_shrink_blend/runner.py"
    )
    native = load(
        "h138_native",
        repo / "research/flywheel_selfplay/h135_blend025_flywheel_val/science.py",
    )
    h.INPUT = c / "inputs/data/h50"
    h.CACHE = c / "inputs/data/h58"
    h.R882 = c / "inputs/round-882.pt"
    for rel, key in [
        ("context_features.npz", h.CACHE),
        ("pool_priors.npz", h.CACHE),
        ("cpu_exact_fused.py", h.INPUT),
        ("selected.json", h.INPUT),
    ]:
        original = next(p for p in h.FILES if p.name == rel)
        h.FILES[key / rel] = h.FILES[original]
    h.FILES[h.R882] = manifest["checkpoints"]["882"]["sha256"]
    return h, native, manifest


def build(c, h, manifest, arm):
    import torch

    checkpoints = {
        r: torch.load(
            c / f"inputs/round-{r}.pt", map_location="cpu", weights_only=False
        )
        for r in ROUNDS
    }
    for r, v in checkpoints.items():
        if (
            sha(c / f"inputs/round-{r}.pt") != manifest["checkpoints"][str(r)]["sha256"]
            or v["dope_sha256"] != DOPE
        ):
            raise RuntimeError("checkpoint identity changed")
    states = {r: v["state_dict"] for r, v in checkpoints.items()}
    fused = load("h138_fused", h.INPUT / "cpu_exact_fused.py")
    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    names = tuple(
        n
        for n, p in net.named_parameters()
        if n.startswith("adapter_10tok.") and not n.startswith("adapter_10tok.anchor.")
    )
    if (
        len(names) != 47
        or sum(dict(net.named_parameters())[n].numel() for n in names) != 29211
    ):
        raise RuntimeError("adapter identity mismatch")
    guard = h.source_module(
        "python/src/jope/north_star/dope_freeze_guard.py", "h138_guard"
    )
    dope_keys = [n for n in states[882] if n.startswith("dope_model.")] + list(
        guard.DOPE_NORMALIZER_BUFFERS
    )
    verify_dope(states, dope_keys)
    loaded = net.load_state_dict(states[882], strict=False)
    if loaded.unexpected_keys or any(
        n.startswith(("adapter_10tok.", "dope_model.")) for n in loaded.missing_keys
    ):
        raise RuntimeError("inexact checkpoint restoration")
    for p in net.parameters():
        p.requires_grad_(False)
    guard.lock_dope_weights_and_eval_mode(net)
    net.eval()
    if guard.snapshot_dope_weights(net).composite_sha256 != DOPE:
        raise RuntimeError("r882 DOPE composite mismatch")
    state = average(states, names, arm)
    net.load_state_dict(state, strict=False)
    if guard.snapshot_dope_weights(net).composite_sha256 != DOPE:
        raise RuntimeError("averaged DOPE mismatch")
    frozen = tensor_hash({n: t for n, t in net.state_dict().items() if n not in names})
    receipt = dict(
        arm=arm,
        adapter_parameters=29211,
        adapter_tensors=47,
        adapter_names=list(names),
        adapter_sha256=tensor_hash({n: state[n] for n in names}),
        reference_adapter_sha256=tensor_hash({n: states[882][n] for n in names}),
        frozen_state_sha256=frozen,
        dope_before=DOPE,
        dope_bitwise_tensors=len(dope_keys),
        DOPE_bitwise_equal=True,
        buffers_from="r882",
        optimizer_updates=0,
        CUDA_initialized=torch.cuda.is_initialized(),
        missing_keys=loaded.missing_keys,
    )
    return net, guard, names, frozen, receipt


def pools(h, rows):
    result = {}
    for r in rows:
        cid = r["context_sha256"]
        p = h.INPUT / "pools" / f"{cid}.npz"
        if sha(p) != r["pool_sha256"]:
            raise RuntimeError("pool hash mismatch")
        with np.load(p, allow_pickle=False) as z:
            fit = z["fitness"].astype(np.float64)
            gate = z["gate"].astype(bool)
        if (
            fit.shape != (r["n_candidates"],)
            or gate.shape != fit.shape
            or not np.isfinite(fit[gate]).all()
        ):
            raise RuntimeError("invalid pool labels")
        result[cid] = dict(
            path=p,
            pool_sha256=r["pool_sha256"],
            fitness=fit,
            gate=gate,
            u=np.where(gate, -fit, -2.0),
        )
    return result


def evaluate(c, out, h, native, manifest, rows, arm):
    import torch

    net, guard, names, frozen, receipt = build(c, h, manifest, arm)
    context, reads = h.train_context(rows, [])
    projection = h.train_projection(rows, [])
    pool = pools(h, rows)
    setter = torch.set_num_threads
    with patch.object(torch, "set_num_threads", lambda requested: setter(1)):
        predictions, net = h.baseline(out, rows, context, pool, projection, net=net)
    values = []
    choices = []
    for r in rows:
        cid = r["context_sha256"]
        v, i = native.native(predictions[cid], pool[cid]["fitness"], pool[cid]["gate"])
        values.append(v)
        choices.append(i)
    if frozen != tensor_hash(
        {n: t for n, t in net.state_dict().items() if n not in names}
    ):
        raise RuntimeError("frozen state changed during inference")
    if receipt["adapter_sha256"] != tensor_hash(
        {n: net.state_dict()[n] for n in names}
    ):
        raise RuntimeError("adapter changed during inference")
    receipt.update(
        dope_after=guard.snapshot_dope_weights(net).composite_sha256,
        values=values,
        choices=choices,
        context_ids=[r["context_sha256"] for r in rows],
        cache_reads=reads,
        CUDA_initialized=torch.cuda.is_initialized(),
    )
    if receipt["dope_after"] != DOPE or receipt["CUDA_initialized"]:
        raise RuntimeError("DOPE/CPU invariant failed")
    np.savez_compressed(
        out / "adapter.npz", **{n: net.state_dict()[n].detach().numpy() for n in names}
    )
    save(out / "receipt.json", receipt)
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--custody", type=Path, required=True)
    p.add_argument("--stage", required=True)
    args = p.parse_args()
    c = args.custody.resolve()
    stage = args.stage
    root = c / "run"
    out = root / stage
    out.mkdir()
    launch = json.loads((c / "launch.json").read_text())
    manifest = json.loads((HERE / "manifest.json").read_text())
    if not launch["merged_source_verified"] or os.getpriority(os.PRIO_PROCESS, 0) < 19:
        raise RuntimeError("merged source/nice19 launch required")
    for name, h in manifest["executor_sha256"].items():
        if sha(HERE / name) != h:
            raise RuntimeError("executor changed")
    packet = json.loads((c / "inputs/data/h50/selected.json").read_text())
    if sha(c / "inputs/data/h50/selected.json") != manifest["selected_packet_sha256"]:
        raise RuntimeError("packet identity changed")
    train, val = packet["train"], packet["dev"]
    for rows, split in [(train, "DL-Train"), (val, "DL-Val")]:
        if any(
            r["dl_split"] != split
            or r["task"] != "regression"
            or r["source_split"] not in ("train", "inner")
            for r in rows
        ):
            raise RuntimeError("unauthorized row")
    if (
        len(train) != 1219
        or len(val) != 91
        or {r["family_id"] for r in train} & {r["family_id"] for r in val}
    ):
        raise RuntimeError("cohort overlap/count changed")
    if sha(HERE / "fold_manifest.json") != manifest["fold_manifest_sha256"]:
        raise RuntimeError("fold manifest changed")
    if (
        folds(train)
        != json.loads((HERE / "fold_manifest.json").read_text())["assignments"]
    ):
        raise RuntimeError("family partition identity changed")
    selected = None
    if stage == "secondary":
        selected = json.loads((root / "primary/metrics.json").read_text())[
            "selected_arm"
        ]
    rows = val if stage == "baseline" or (stage == "secondary" and selected) else train
    permitted = {
        str(c / "inputs/data/h50/pools" / f"{r['context_sha256']}.npz") for r in rows
    }
    opened = set()
    refusals = []

    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        s = str(path)
        mode = args[1]
        write = isinstance(mode, str) and any(x in mode for x in "wax+")
        if any(
            x in s.lower()
            for x in [
                "/sealed",
                "/holdout",
                "meta-test",
                "meta_test",
                "source_manifest.json",
                "qualification.lock",
            ]
        ) or path.is_relative_to(Path("/home/ubuntu/jope-data")):
            refusals.append(s)
            raise RuntimeError("protected/outside input refused")
        if path.is_relative_to(c / "inputs"):
            if write:
                raise RuntimeError("input mutation refused")
            if path.parent == c / "inputs/data/h50/pools" and s not in permitted:
                refusals.append(s)
                raise RuntimeError("unadmitted phase pool")
            opened.add(s)

    sys.addaudithook(audit)
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from threadpoolctl import threadpool_limits

    _limit = threadpool_limits(limits=1)  # Retain the limiter for process lifetime.

    def no_gradient(*a, **kw):
        raise RuntimeError("zero-update H138 forbids gradient/optimizer operations")

    torch.Tensor.backward = no_gradient
    torch.autograd.grad = no_gradient
    torch.optim.Optimizer.step = no_gradient
    h, native, manifest = dependencies(c)
    try:
        if stage == "baseline":
            reference = json.loads((c / "inputs/reference.json").read_text())
            contract = reference["evaluation_contract"]
            if (
                sha(c / "inputs/reference.json") != manifest["reference_sha256"]
                or reference["evaluation_identity"] != manifest["evaluation_identity"]
                or hashlib.sha256(
                    json.dumps(contract, sort_keys=True).encode()
                ).hexdigest()
                != reference["evaluation_identity"]
            ):
                raise RuntimeError("evaluation identity mismatch")
            expected = [
                dict(
                    cid=r["context_sha256"],
                    pool_sha256=r["pool_sha256"],
                    training_table_sha256=r["train_only_jtt1_sha256"],
                )
                for r in val
            ]
            if contract["contexts"] != expected:
                raise RuntimeError("91-context contract changed")
            r = evaluate(c, out, h, native, manifest, val, "R882")
            sanity = native.sanity(r["values"], reference)
            if sanity["max_context_error"] != 0 or sanity["mean_error"] != 0:
                raise RuntimeError("r882 exact91-context regression gate failed")
            save(out / "sanity.json", sanity)
        else:
            if (
                json.loads((root / "baseline/sanity.json").read_text())["status"]
                != "PASS"
            ):
                raise RuntimeError("baseline gate first")
            if stage.startswith("train_"):
                arm = stage.removeprefix("train_")
                if arm not in ("R882", *ARMS):
                    raise ValueError("unregistered arm")
                evaluate(c, out, h, native, manifest, train, arm)
            elif stage == "primary":
                values = {
                    a: json.loads((root / f"train_{a}/receipt.json").read_text())[
                        "values"
                    ]
                    for a in ("R882", *ARMS)
                }
                save(out / "metrics.json", primary(train, values))
            elif stage == "secondary":
                if selected is None:
                    save(
                        out / "receipt.json",
                        dict(status="NOT_RUN_PRIMARY_FAILED", arm_payload_reads=0),
                    )
                else:
                    r = evaluate(c, out, h, native, manifest, val, selected)
                    base = json.loads((root / "baseline/receipt.json").read_text())[
                        "values"
                    ]
                    save(
                        out / "metrics.json",
                        secondary(val, base, r["values"], selected),
                    )
            elif stage == "summary":
                prim = json.loads((root / "primary/metrics.json").read_text())
                sec = (
                    json.loads((root / "secondary/metrics.json").read_text())
                    if (root / "secondary/metrics.json").exists()
                    else None
                )
                metrics = dict(
                    wave="H138",
                    status="COMPLETE",
                    optimizer_updates=0,
                    primary=prim,
                    secondary=sec,
                    verdict=sec["verdict"] if sec else "NO_ADVANCE",
                    serving_promotion=False,
                )
                if len(json.dumps(metrics).encode()) >= 100000:
                    raise RuntimeError("metrics exceed100KB")
                save(root / "metrics.json", metrics)
            else:
                raise ValueError("unknown stage")
    finally:
        save(
            out / "read-audit.json",
            dict(
                stage=stage,
                metadata_scope="Selected packet train+dev metadata is read; no source_manifest/protected metadata. Only authorized phase pool payloads are permitted; cached numeric rows are range-selected.",
                opened=sorted(opened),
                protected_reads=0,
                refusals=refusals,
                optimizer_updates=0,
            ),
        )
    print("COMPLETE", stage, flush=True)


if __name__ == "__main__":
    main()

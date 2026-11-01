"""Prospective H155, immutable H153 primitives, TRAIN-family calibration only."""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types

import numpy as np
import psutil
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = Path("/home/ubuntu/JopeDime")
PARENT = "ea6347d5cc642cd0d7ae847c4e95f309471a5e72"
BASE = "research/flywheel_selfplay/h153_sam/"
INPUT = Path("/home/ubuntu/jope-data/h50/input")
CACHE = Path("/home/ubuntu/jope-data/h58/cache/context_features.npz")
SEEDS = (155101, 155102, 155103)
ARMS = {
    "CTRL": (0.0, False),
    "LS05": (0.05, False),
    "LS10": (0.1, False),
    "LS05_TEMP": (0.05, True),
    "LS10_TEMP": (0.1, True),
}
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def pinned_sources():
    """No foreign working files are imported: immutable, hash-bound Git blobs."""
    manifest = json.loads((HERE / "manifest.json").read_text())
    code = {}
    for name, expected in manifest["parent_sources"].items():
        raw = subprocess.check_output(
            ["git", "show", PARENT + ":" + BASE + name], cwd=REPO
        )
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError("parent source changed: " + name)
        code[name] = raw
    runtime = types.ModuleType("h155_parent_runtime")
    runtime.__file__ = str(REPO / BASE / "runtime.py")
    sys.modules[runtime.__name__] = runtime
    exec(compile(code["runtime.py"], runtime.__file__, "exec"), runtime.__dict__)
    runtime.HERE = REPO  # Guard Git reads survive removal of the parent's source export.
    primitives = {
        "np": np,
        "torch": torch,
        "hashlib": hashlib,
        "random": random,
        "SEEDS": SEEDS,
        "__name__": "h155_primitives",
    }
    for name, names in [
        ("runner.py", {"predict", "evaluate", "baseline_gate"}),
        ("science.py", {"batch_plan"}),
    ]:
        tree = ast.parse(code[name])
        nodes = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        if len(nodes) != len(names):
            raise RuntimeError("primitive inventory mismatch")
        exec(
            compile(
                ast.Module(body=nodes, type_ignores=[]), str(REPO / BASE / name), "exec"
            ),
            primitives,
        )
    return runtime, primitives


def split_families(rows):
    families = sorted({r["family_id"] for r in rows})
    groups = [
        sorted(x.tolist())
        for x in np.array_split(np.random.default_rng(155140).permutation(families), 5)
    ]
    assert len(families) == 205
    assert not any(
        set(a) & set(b) for i, a in enumerate(groups) for b in groups[i + 1 :]
    )
    return groups


def target_cost(fitness, gate):
    return torch.where(gate, fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0))


def pairwise(pred, target, eps, temperature):
    """Lower cost is better; a positive logit prefers the first candidate."""
    i, j = torch.triu_indices(
        pred.shape[-1], pred.shape[-1], offset=1, device=pred.device
    )
    mask = target[..., i] != target[..., j]
    if not bool(mask.any()):
        return pred.sum() * 0
    logits = ((pred[..., j] - pred[..., i]) / temperature)[mask]
    label = (target[..., i] < target[..., j]).to(pred.dtype)[mask]
    smooth = eps + (1 - 2 * eps) * label
    return F.binary_cross_entropy_with_logits(logits, smooth)


def objective(pred, fitness, gate, eps, temperature, parent_loss):
    return parent_loss(pred, fitness, gate) + 0.1 * pairwise(
        pred, target_cost(fitness, gate), eps, temperature
    )


def fit_temperature(logits, labels, eps):
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([log_t], lr=0.05)
    targets = eps + (1 - 2 * eps) * labels.double()
    losses = []
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        value = F.binary_cross_entropy_with_logits(
            logits.double() / log_t.exp(), targets
        )
        value.backward()
        optimizer.step()
        with torch.no_grad():
            log_t.clamp_(math.log(0.05), math.log(20))
        losses.append(float(value.detach()))
    return float(log_t.detach().exp()), losses


def guard():
    """CPU-only worker: pause its own work for load, abort at disk hard stop."""
    for path in [HERE, Path("/home/ubuntu/jope-data/h155")]:
        v = os.statvfs(path)
        if (
            v.f_bavail * v.f_frsize < 40_000_000_000
            or 1 - v.f_bfree / v.f_blocks >= 0.95
        ):
            raise RuntimeError("DISKGUARD")
    p = psutil.Process()
    if p.num_threads() > 4 or p.memory_info().rss >= 6_000_000_000:
        raise RuntimeError("PROCESS_RESOURCE_GUARD")
    while (
        max(
            psutil.cpu_percent(interval=0.05),
            100 * os.getloadavg()[0] / (os.cpu_count() or 1),
        )
        > 85
    ):
        print("PAUSED own CPU work for host pressure", flush=True)
        time.sleep(5)
    if psutil.virtual_memory().available < 16 * 1024**3:
        raise RuntimeError("MEMORY_GUARD")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only worker initialized CUDA")


def summarize(rows, values, reference):
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    families = sorted({r["family_id"] for r in rows})
    delta = np.stack(
        [
            (values - reference)[
                :, [i for i, r in enumerate(rows) if r["family_id"] == family]
            ].mean(1)
            for family in families
        ],
        1,
    )
    gains = delta.mean(1)
    ix = np.random.default_rng(155141).integers(
        len(families), size=(10000, len(families))
    )
    draws = np.median(delta[:, ix].mean(2), axis=0)
    scores = np.where(delta > 0, 1.0, np.where(delta < 0, 0.0, 0.5))
    wins = scores.sum(1)
    elo = 1500 + 400 * np.log10((wins + 0.5) / (len(families) - wins + 0.5))
    bw = scores[:, ix].sum(2)
    be = np.median(
        1500 + 400 * np.log10((bw + 0.5) / (len(families) - bw + 0.5)), axis=0
    )
    return dict(
        contexts=len(rows),
        families=len(families),
        context_mean_by_seed=values.mean(1).tolist(),
        family_gain_by_seed=gains.tolist(),
        median_family_gain=float(np.median(gains)),
        gain_ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
        p_one_sided=float((1 + (draws <= 0).sum()) / 10001),
        seeds_beating=int((gains > 0).sum()),
        elo_by_seed=elo.tolist(),
        median_elo=float(np.median(elo)),
        elo_ci95=np.quantile(be, [0.025, 0.975]).tolist(),
        family_ids=families,
        family_delta_by_seed=delta.tolist(),
    )


def decisions(inner):
    treatment = [a for a in ARMS if a != "CTRL"]
    result, adjusted = {}, 0.0
    for rank, arm in enumerate(
        sorted(treatment, key=lambda a: (inner[a]["vs_CTRL"]["p_one_sided"], a))
    ):
        c, r = inner[arm]["vs_CTRL"], inner[arm]["vs_r882"]
        adjusted = max(adjusted, min(1.0, c["p_one_sided"] * (4 - rank)))
        gates = dict(
            ctrl_gain=c["median_family_gain"] > 0,
            ctrl_ci=c["gain_ci95"][0] > 0,
            r882_gain=r["median_family_gain"] > 0,
            r882_ci=r["gain_ci95"][0] > 0,
            holm=adjusted < 0.05,
            seeds=c["seeds_beating"] >= 2 and r["seeds_beating"] >= 2,
        )
        result[arm] = dict(holm_p=adjusted, gates=gates, promising=all(gates.values()))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir()  # Exactly one attempt, never consumes cached outcome cells.
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pins = json.loads((HERE / "manifest.json").read_text())
    for name, expected in pins["inputs"].items():
        if digest(name) != expected:
            raise RuntimeError("input hash mismatch: " + name)
    for name, expected in pins["assets"].items():
        if digest(name) != expected:
            raise RuntimeError("model/source asset mismatch: " + name)
    for name, expected in pins["executor_sha256"].items():
        if digest(HERE / name) != expected:
            raise RuntimeError("executor hash mismatch: " + name)
    packet = json.loads((INPUT / "selected.json").read_text())
    train, dev = packet["train"], packet["dev"]
    assert (len(train), len(dev)) == (1219, 91)
    assert len({r["family_id"] for r in dev}) == 19
    assert not ({r["family_id"] for r in train} & {r["family_id"] for r in dev})
    for r in train + dev:
        assert (
            r["task"] == "regression"
            and r["source_split"] in ("train", "inner")
            and not r.get("has_meta_test")
        )
    assert len({r["context_sha256"] for r in train + dev}) == 1310
    assert all(r["dl_split"] == "DL-Train" for r in train)
    assert all(r["dl_split"] == "DL-Val" for r in dev)
    groups = split_families(train)
    assert groups == json.loads((HERE / "fold_manifest.json").read_text())["families"]
    fit = [r for r in train if r["family_id"] not in set(groups[0] + groups[1])]
    cal = [r for r in train if r["family_id"] in groups[1]]
    inner = [r for r in train if r["family_id"] in groups[0]]
    runtime, primitives = pinned_sources()
    allowed = set(pins["inputs"]) | set(pins["assets"])
    allowed.update(
        str(INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + dev
    )
    audit = runtime.Audit(allowed, out)
    sys.addaudithook(audit.check)
    updates, cells = 0, {}
    start = time.monotonic()
    try:
        guard()
        net, freeze_guard, names, frozen, fused = runtime.build(
            Path(pins["checkpoint"])
        )
        params = [p for p in net.parameters() if p.requires_grad]
        assert len(params) == 47 and sum(p.numel() for p in params) == 29211
        assert not any(
            n.startswith("dope") and p.requires_grad for n, p in net.named_parameters()
        )
        if args.smoke:
            runtime.check_frozen(net, freeze_guard, names, frozen)
            save(
                out / "smoke.json",
                dict(
                    status="PASS",
                    optimizer_updates=0,
                    scored_contexts=0,
                    dope_sha256=DOPE,
                    adapter_parameters=29211,
                    adapter_tensors=47,
                    trainable_names=list(names),
                    frozen_sha256=frozen,
                    cuda_initialized=torch.cuda.is_initialized(),
                    audit=audit.receipt(),
                ),
            )
            return
        launch = json.loads((HERE / "launch.json").read_text())
        assert launch["prereg_merged"] and launch["executor_merged"]
        assert launch["manifest_sha256"] == digest(HERE / "manifest.json")
        pools, tokens, masks = {}, {}, {}
        projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)
        for r in train + dev:
            cid = r["context_sha256"]
            path = INPUT / "pools" / (cid + ".npz")
            if digest(path) != r["pool_sha256"]:
                raise RuntimeError("pool identity mismatch: " + cid)
            with np.load(path, allow_pickle=False) as z:
                program = z["program"].astype(np.float32)
                pools[cid] = dict(
                    program=program,
                    proj16=(program.astype(np.float64) @ projection).astype(np.float32),
                    fitness=z["fitness"].astype(np.float64),
                    gate=z["gate"].astype(bool),
                )
        with np.load(CACHE, allow_pickle=False) as z:
            ids = list(map(str, z["cids"]))
            assert len(ids) == 1310 and set(ids) == set(pools)
            for i, cid in enumerate(ids):
                tokens[cid] = torch.as_tensor(
                    z["tokens_10x32"][i].copy(), dtype=torch.float32
                )
                masks[cid] = torch.as_tensor(z["mask_10"][i].copy(), dtype=torch.bool)
        evaluate, predict = primitives["evaluate"], primitives["predict"]

        def assess(rows):
            guard()
            result = evaluate(net, pools, tokens, masks, rows)
            guard()
            return result

        reference = json.loads(Path(pins["reference"]).read_text())
        expected_contract = [
            dict(
                cid=r["context_sha256"],
                pool_sha256=r["pool_sha256"],
                training_table_sha256=r["train_only_jtt1_sha256"],
            )
            for r in dev
        ]
        assert reference["evaluation_contract"]["contexts"] == expected_contract
        assert reference["best_utility"] == 1.1377941898164525
        baseline_dev = assess(
            dev
        )  # First scored action, before any optimization/calibration.
        sanity = primitives["baseline_gate"](baseline_dev["values"], reference)
        save(out / "baseline_dev.json", dict(**baseline_dev, sanity=sanity))
        print("BASELINE_EXACT_PASS", sanity, flush=True)
        baseline_train = assess(train)
        save(out / "baseline_train.json", baseline_train)
        by_cid = {c: i for i, c in enumerate(baseline_train["context_ids"])}
        baseline_inner = {
            "values": [
                baseline_train["values"][by_cid[r["context_sha256"]]] for r in inner
            ]
        }
        # Calibrator only sees group1 labels; no outcomes enter adapter batch generation.
        logits, labels = [], []
        with torch.no_grad():
            for r in cal:
                cid = r["context_sha256"]
                size = len(pools[cid]["fitness"])
                pred = predict(net, pools, tokens, masks, [cid], [np.arange(size)])[0]
                pos = torch.as_tensor(np.linspace(0, size - 1, 16, dtype=int))
                target = target_cost(
                    torch.as_tensor(pools[cid]["fitness"], dtype=torch.float32),
                    torch.as_tensor(pools[cid]["gate"]),
                )[pos]
                i, j = torch.triu_indices(16, 16, offset=1)
                unequal = target[i] != target[j]
                logits.append((pred[pos][j] - pred[pos][i])[unequal])
                labels.append((target[i] < target[j])[unequal].float())
        temperatures = {}
        for eps in (0.05, 0.1):
            temperature, losses = fit_temperature(
                torch.cat(logits), torch.cat(labels), eps
            )
            temperatures[str(eps)] = dict(
                temperature=temperature,
                losses=losses,
                calibration_contexts=len(cal),
                families=groups[1],
                pairs=sum(x.numel() for x in labels),
            )
        save(out / "calibration.json", temperatures)
        print(
            "CALIBRATION_COMPLETE",
            {e: x["temperature"] for e, x in temperatures.items()},
            flush=True,
        )
        runtime.check_frozen(net, freeze_guard, names, frozen)
        initial = {n: t.detach().clone() for n, t in net.state_dict().items()}
        initial_vec = torch.cat([initial[n].reshape(-1) for n in names])
        orders = {
            r["context_sha256"]: np.argsort(pools[r["context_sha256"]]["fitness"])
            for r in fit
        }
        for seed in SEEDS:
            plan = primitives["batch_plan"](list(orders), orders, seed)
            assert len(plan) == 40
            ph = hashlib.sha256()
            for stream, cids, ixs in plan:
                ph.update(str(stream).encode())
                for cid, ix in zip(cids, ixs, strict=True):
                    ph.update(cid.encode())
                    ph.update(ix.astype("<i8").tobytes())
            for arm, (eps, calibrated) in ARMS.items():
                net.load_state_dict(initial, strict=True)
                optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
                assert {
                    id(p) for group in optimizer.param_groups for p in group["params"]
                } == {id(p) for p in params}
                temperature = (
                    temperatures[str(eps)]["temperature"] if calibrated else 1.0
                )
                losses = []
                for stream, cids, ixs in plan:
                    guard()
                    torch.manual_seed(stream)
                    fitness = torch.as_tensor(
                        np.stack(
                            [
                                pools[c]["fitness"][ix]
                                for c, ix in zip(cids, ixs, strict=True)
                            ]
                        ),
                        dtype=torch.float32,
                    )
                    gate = torch.as_tensor(
                        np.stack(
                            [
                                pools[c]["gate"][ix]
                                for c, ix in zip(cids, ixs, strict=True)
                            ]
                        ),
                        dtype=torch.bool,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    pred = predict(net, pools, tokens, masks, cids, ixs)
                    value = objective(
                        pred, fitness, gate, eps, temperature, runtime.loss
                    )
                    if not bool(torch.isfinite(value)):
                        raise RuntimeError("nonfinite objective")
                    value.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                    runtime.check_frozen(net, freeze_guard, names, frozen)
                    losses.append(float(value.detach()))
                drift = float(
                    torch.linalg.vector_norm(
                        torch.cat(
                            [
                                dict(net.named_parameters())[n].detach().reshape(-1)
                                for n in names
                            ]
                        )
                        - initial_vec
                    )
                )
                assert drift > 0
                # All scheduled updates have completed before these cell diagnostics.
                dev_result = assess(dev)
                train_result = assess(train)
                inner_result = {
                    "values": [
                        train_result["values"][by_cid[r["context_sha256"]]]
                        for r in inner
                    ]
                }
                runtime.check_frozen(net, freeze_guard, names, frozen)
                cell = dict(
                    seed=seed,
                    arm=arm,
                    eps=eps,
                    temperature=temperature,
                    optimizer_updates=40,
                    adapter_l2_drift=drift,
                    adapter_sha256=runtime.tensor_hash(
                        {n: net.state_dict()[n] for n in names}
                    ),
                    frozen_sha256=frozen,
                    dope_sha256=DOPE,
                    losses=losses,
                    batch_plan_sha256=ph.hexdigest(),
                    inner=inner_result,
                    dev=dev_result,
                    train=train_result,
                )
                save(out / f"{arm}-{seed}.json", cell)
                np.savez_compressed(
                    out / f"{arm}-{seed}-adapter.npz",
                    **{n: net.state_dict()[n].numpy() for n in names},
                )
                cells[arm, seed] = cell
                print(
                    "CELL_COMPLETE",
                    arm,
                    seed,
                    "updates",
                    updates,
                    "drift",
                    drift,
                    flush=True,
                )
        result = {}
        for panel, rows, baseline in [
            ("inner", inner, baseline_inner),
            ("dev91", dev, baseline_dev),
            ("train1219", train, baseline_train),
        ]:
            field = (
                "dev"
                if panel == "dev91"
                else ("train" if panel == "train1219" else "inner")
            )
            result[panel] = {}
            for arm in ARMS:
                values = np.asarray(
                    [cells[arm, seed][field]["values"] for seed in SEEDS]
                )
                ctrl = np.asarray(
                    [cells["CTRL", seed][field]["values"] for seed in SEEDS]
                )
                result[panel][arm] = dict(
                    vs_r882=summarize(rows, values, baseline["values"]),
                    vs_CTRL=summarize(rows, values, ctrl),
                )
        decision = decisions(result["inner"])
        metrics = dict(
            wave="H155",
            status="COMPLETE",
            verdict="PROMISING_EXPLORATORY"
            if any(x["promising"] for x in decision.values())
            else "NO_ADVANCE",
            panels=result,
            decisions=decision,
            baseline=sanity,
            optimizer_updates=updates,
            seeds=list(SEEDS),
            arms=ARMS,
            dope_sha256=DOPE,
            frozen_sha256=frozen,
            calibration_temperatures={
                k: v["temperature"] for k, v in temperatures.items()
            },
            fold_sizes=dict(
                fit_contexts=len(fit),
                calibration_contexts=len(cal),
                inner_contexts=len(inner),
                fit_families=123,
                calibration_families=41,
                inner_families=41,
            ),
            manifest_sha256=digest(HERE / "manifest.json"),
            source=launch,
            CPU_only=True,
            GPU_jobs=0,
            protected_reads=0,
            synthetic_training_datasets=0,
            elapsed_seconds=time.monotonic() - start,
            limitations=[
                "r882 previously trained on all1219 TRAIN contexts",
                "91-context DL-Val repeatedly exposed; exploratory only",
                "No serving admission or sealed-test qualification",
                "S3 custody is exact H144 archive; original H50/H58 derivative versions not established",
            ],
        )
        save(out / "metrics.json", metrics)
        print("STUDY_COMPLETE", metrics["verdict"], updates, flush=True)
    finally:
        save(
            out / "access-audit.json",
            dict(
                **audit.receipt(),
                optimizer_updates=updates,
                CUDA_initialized=torch.cuda.is_initialized(),
                process_threads=psutil.Process().num_threads(),
                nice=os.getpriority(os.PRIO_PROCESS, 0),
            ),
        )


if __name__ == "__main__":
    main()

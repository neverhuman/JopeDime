"""Single H201 attempt: exact native DL-Val first, zero updates, frozen history replay."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from monitor import admit_host

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = Path("/home/ubuntu/jope-data/h201")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def native_modules():
    """Reuse the exact Engine/runtime used by H197 and the H199 native endpoint."""
    old_path, saved = (
        sys.path[:],
        {
            k: sys.modules.pop(k, None)
            for k in ["analysis", "treatment", "h182_source_treatment"]
        },
    )
    try:
        sys.path.insert(0, str(HERE.parent / "h189_swa_ema"))
        base = load_module(
            "h201_h197_native_base", HERE.parent / "h189_swa_ema/h182_base.py"
        )
    finally:
        sys.path[:] = old_path
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    native = load_module(
        "h201_h199_native_endpoint", HERE.parent / "h139_selection_audit/runner.py"
    )
    return base, native


def main():
    host = admit_host()  # before any scientific input read or numerical-library import
    if os.getpriority(os.PRIO_PROCESS, 0) != 19 or len(os.sched_getaffinity(0)) > 2:
        raise RuntimeError("H201 requires nice19, affinity <=2 cores")
    import numpy as np
    import torch
    import analysis as stat

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)

    def forbidden(*args, **kwargs):
        raise RuntimeError("H201 forbids optimizer construction, updates and backward")

    torch.optim.Optimizer.__init__ = forbidden
    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output != DATA / "attempt-001" or args.output.exists():
        raise RuntimeError("fresh prescribed attempt-001 only; no retries")
    pins = json.loads((HERE / "manifest.json").read_text())
    launch = json.loads(args.launch.read_text())
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package pin mismatch: " + name)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    if (
        launch["source_commit"] != head
        or not launch["merged_source_verified"]
        or launch["manifest_sha256"] != sha(HERE / "manifest.json")
    ):
        raise RuntimeError("launch must bind reviewed merged source")
    for group in ["source_sha256", "parent_sources", "assets", "inputs"]:
        for name, expected in pins[group].items():
            path = (
                HERE / name
                if group == "source_sha256"
                else REPO / name
                if group == "parent_sources"
                else Path(name)
            )
            if sha(path) != expected:
                raise RuntimeError("source/input pin mismatch: " + str(path))
    args.output.mkdir()
    save(
        args.output / "launch.json",
        dict(
            host=host,
            source_commit=head,
            launch=launch,
            optimizer_updates=0,
            backward_calls=0,
        ),
    )
    base, endpoint = native_modules()
    rt = base.runtime()
    packet = json.loads((rt.INPUT / "selected.json").read_text())
    train, rows = packet["train"], packet["dev"]
    if len(rows) != 91 or len({r["family_id"] for r in rows}) != 19:
        raise RuntimeError("fixed native DL-Val cohort is 91 contexts /19 families")
    if {r["family_id"] for r in rows} & {r["family_id"] for r in train}:
        raise RuntimeError("TRAIN/Val family overlap")
    if any(
        r["dl_split"] != "DL-Val"
        or r["source_split"] not in ("train", "inner")
        or r.get("has_meta_test", False)
        for r in rows
    ):
        raise RuntimeError("unapproved input role")
    allowed = set(pins["inputs"]) | set(pins["assets"]) | {str(args.launch)}
    allowed |= {str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in rows}
    audit = rt.Audit(allowed, args.output)
    sys.addaudithook(audit.check)
    net, guard, names, frozen, fused = rt.build(str(DATA / "ckpt/round-882.pt"))
    for p in net.parameters():
        p.requires_grad_(False)
    engine = base.Engine(net, rt, names, frozen, guard, fused, train + rows)
    engine.load(rows)
    initial = rt.tensor_hash(net.state_dict())
    baseline, costs = engine.evaluate(rows, native=True, prove_identity=True)
    ids, families = [r["context_sha256"] for r in rows], [r["family_id"] for r in rows]
    vector = np.asarray([baseline[c] for c in ids])
    reference = json.loads((DATA / "inputs/r882-reference.json").read_text())
    if (
        not np.array_equal(vector, np.asarray(reference["utility"]["per_context"]))
        or float(vector.mean()) != 1.1377941898164525
    ):
        raise RuntimeError("native R882 regression is not bit-exact")
    with np.load(pins["baseline_cost_reference"], allow_pickle=False) as z:
        if any(
            costs[c].dtype != z[c].dtype or costs[c].tobytes() != z[c].tobytes()
            for c in ids
        ):
            raise RuntimeError("native R882 candidate cost bytes changed")
    save(
        args.output / "baseline.json",
        dict(
            status="PASS",
            contexts=91,
            families=19,
            mean=float(vector.mean()),
            cost_bytes_exact=True,
            optimizer_updates=0,
        ),
    )
    print("BASELINE PASS 91/91 native exact; zero updates", flush=True)
    catalog = {
        r["id"]: r
        for r in map(
            json.loads, (DATA / "inputs/real-catalog.jsonl").read_text().splitlines()
        )
    }
    if any(catalog[r["dataset_id"]]["origin"] != "real" for r in rows):
        raise RuntimeError("real S3 catalog join failed")
    panels, arm_names = [vector], ["R882", "R3100", "R3175"]
    np.savez_compressed(args.output / "R882-costs.npz", **costs)
    for round_id in [3100, 3175]:
        state = torch.load(
            DATA / f"ckpt/round-{round_id}.pt", map_location="cpu", weights_only=True
        )
        if state["round"] != round_id or state["dope_sha256"] != rt.DOPE:
            raise RuntimeError("checkpoint round/DOPE identity")
        net.load_state_dict(state["state_dict"], strict=True)
        engine.check()
        values, raw = engine.evaluate(rows, native=True, prove_identity=True)
        for row in rows:
            c = row["context_sha256"]
            pool = engine.pools[c]
            if values[c] != endpoint.native(raw[c], pool["fitness"], pool["gate"]):
                raise RuntimeError("H197/H199 native endpoint disagreement")
        panels.append(np.asarray([values[c] for c in ids]))
        np.savez_compressed(args.output / f"R{round_id}-costs.npz", **raw)
        print("NATIVE REPLAY COMPLETE", round_id, flush=True)
    panel = np.stack(panels)
    family_names, family = stat.family_panel(panel, families)
    history = [
        json.loads(line)
        for line in (DATA / "inputs/adapter_admissions.jsonl").read_text().splitlines()
    ]
    if any(
        r["evaluation_identity"] != reference["evaluation_identity"] for r in history
    ):
        raise RuntimeError("history differs from pinned native evaluation identity")
    archive = [
        json.loads(line)
        for line in (DATA / "inputs/round_dev.complete.jsonl").read_text().splitlines()
    ]
    if [r["round"] for r in history] != [r["round"] for r in archive]:
        raise RuntimeError("incomplete historical per-context coverage")
    contract = json.loads((DATA / "inputs/evaluation_contract.json").read_text())
    if [r["cid"] for r in contract["contexts"]] != ids:
        raise RuntimeError("historical/native context order mismatch")
    historical = np.asarray(
        [r["candidate"]["per_context"] for r in archive], dtype=np.float64
    )
    _, hf = stat.family_panel(historical, families)
    initial_context = np.asarray(archive[0]["initial_incumbent"]["per_context"])
    _, initial_f = stat.family_panel(initial_context[None], families)
    previous = initial_context.copy()
    parity = {}
    for record, row, v in zip(archive, history, historical, strict=True):
        if (
            record["admission"]["evaluation_identity"] != row["evaluation_identity"]
            or abs(float(v.mean()) - row["candidate_utility"]) > 1e-12
            or abs(float(previous.mean()) - row["previous_best_utility"]) > 1e-12
        ):
            raise RuntimeError("history identity, scalar or incumbent-vector mismatch")
        if row["accepted"]:
            previous = v.copy()
        if row["round"] in [882, 3100, 3175]:
            arm = arm_names.index("R" + str(row["round"]))
            parity[str(row["round"])] = bool(np.array_equal(panel[arm], v))
    if not all(parity.values()) or set(parity) != {"882", "3100", "3175"}:
        raise RuntimeError("native replay differs from archived candidate utilities")
    comparisons = {
        name: stat.interval(family[a] - family[b])
        for name, a, b in [
            ("R3175_minus_R882", 2, 0),
            ("R3175_minus_R3100", 2, 1),
            ("R3100_minus_R882", 1, 0),
        ]
    }
    before3175 = np.asarray([r["round"] <= 3175 for r in history])
    null = stat.null_max(hf[before3175], family[0])
    split = stat.split_selection(
        hf[before3175],
        family[0],
        [r["round"] for r in history if r["round"] <= 3175],
        family_names,
    )
    ratings = stat.elo(panel, families, arm_names)
    sma, decisions = stat.sma_replay(
        [r["round"] for r in history], hf, initial_f[0], [r["action"] for r in history]
    )
    save(args.output / "SMA-decisions.json", decisions)
    np.savez_compressed(
        args.output / "panels.npz",
        native=panel,
        historical=historical,
        initial=initial_context,
        context_ids=np.asarray(ids),
        family_ids=np.asarray(families),
        rounds=np.asarray([r["round"] for r in history]),
    )
    final_verdict = stat.verdict(
        comparisons["R3175_minus_R882"]["ci95"][0],
        null["p_max"],
        split["all_three_B_positive"],
        ratings["R3175"]["ci95"][0],
    )
    net.load_state_dict(
        torch.load(DATA / "ckpt/round-882.pt", map_location="cpu", weights_only=True)[
            "state_dict"
        ],
        strict=True,
    )
    engine.check()
    if (
        initial != rt.tensor_hash(net.state_dict())
        or any(p.requires_grad for p in net.parameters())
        or torch.cuda.is_initialized()
        or audit.refused
    ):
        raise RuntimeError("final frozen state/zero-update boundary")
    receipt = audit.receipt()
    save(args.output / "audit.json", receipt)
    metrics = dict(
        status="COMPLETE",
        wave="H201",
        verdict=final_verdict,
        host=host,
        source_commit=head,
        optimizer_updates=0,
        backward_calls=0,
        DOPE_before=rt.DOPE,
        DOPE_after=rt.DOPE,
        nonadapter_sha256=frozen,
        restored_model_sha256=initial,
        contexts=91,
        families=19,
        order_family_count_discrepancy="order20; pinned H197/H199/native cohort19; no padding",
        baseline_mean=1.1377941898164525,
        native_archive_parity=parity,
        context_mean={a: float(panel[j].mean()) for j, a in enumerate(arm_names)},
        equal_family_utility={
            a: float(family[j].mean()) for j, a in enumerate(arm_names)
        },
        comparisons=comparisons,
        Elo=ratings,
        winner_null=null,
        selection_split=split,
        harm={
            a: dict(
                family_fraction=float(np.mean(family[j] < family[0])),
                context_fraction=float(np.mean(panel[j] < panel[0])),
                minus2_count=int((panel[j] == -2).sum()),
            )
            for j, a in enumerate(arm_names)
        },
        history=dict(
            rows=len(history),
            first_round=history[0]["round"],
            last_round=history[-1]["round"],
            initial_incumbent_round=774,
            complete_per_context_coverage=True,
            SMA=sma,
        ),
        audit=dict(
            opened_count=len(receipt["opened"]),
            refused=receipt["refused"],
            protected_reads=0,
        ),
        interpretation="retrospective repeatedly observed DL-Val; centered empirical null and split checks do not establish independent generalization",
    )
    save(args.output / "metrics.json", metrics)
    print("H201 COMPLETE", final_verdict, "updates=0", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""H143 shadow replay of the strict-margin admission rule over the logged flywheel trajectory (zero updates, CPU).

Stage `replay` re-decides every logged admission (rounds 775-2530, evaluation identity e79a7715) from the per-round
receipts' per-context 91-context validation utilities, under S_LIVE (the live single-mean rule, a regression check),
R_STRICT and R_STRICT_NOINV with three bootstrap RNG seeds. Stage `dltrain` scores each distinct final incumbent and
R882 on DL-Train with the merged H124 CPU forward pass (byte-pinned) and writes the readout and verdict.
The candidate stream is the logged one: candidates the live loop would have trained from a different incumbent are
not reproducible offline (stated limitation).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
H124_PATH = HERE.parent / "h124_gbm_shrink_blend" / "runner.py"
H124_SHA = "9ebce9652a7caadc7e50896a9c1b667de5aea53f1e3cf4f8797f815d8a16285a"
H143_PREREG_COMMIT = "245f59f97bed67e98e1f88bc9d0c6952540aa30a"
INPUTS = Path("/home/ubuntu/jope-data/h143/inputs")
H50 = Path("/home/ubuntu/jope-data/h50/input")
RECEIPTS_SUMS_SHA = "b7d3a5a5f8c9f3ce320dd4ad81038dd0c5ac3bfbd26d9f1155b1d6d1909a4b76"
ADMISSIONS = (
    "adapter_admissions_775_2530.jsonl",
    "833d27cd453af4e2a17ce7583388bf4c9cfaf9e393975170d41c9e15fa3819e9",
)
REFERENCE = (
    "best_dev_checkpoint.json",
    "30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a",
)
SELECTED_SHA = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
EVALUATION_IDENTITY = "e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83"
ROUNDS = (775, 2530)
CHECKPOINTS = {  # round: (file, sha256) — copied at claim time 2026-11-01T01:03:15Z
    774: (
        "R774.pt",
        "60430aa6466f7ff9be95daf98b8e06f10c284cf752580e67b4e39b33b5d1f7aa",
    ),
    782: (
        "R782.pt",
        "427b0d4608e94f60f5f97d79b5c1c4ce9a7efd59d17e7853f714e901f3a4af0c",
    ),
    785: (
        "R785.pt",
        "ef6696f6369122a162d58b9b522fd83cae110a5576ef3760e3b43ba85d129e0a",
    ),
    859: (
        "R859.pt",
        "5802245f447b1852610576cde08b8a299a9e1e99eb82991aac63c958284fa715",
    ),
    882: (
        "R882.pt",
        "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64",
    ),
}
SEEDS = (143101, 143102, 143103)
ARMS = ("R_STRICT", "R_STRICT_NOINV")
DRAWS = 2000
MARGIN = 0.001
INVALID = -2.0
DLTRAIN_DRAWS = 10_000
DLTRAIN_SEED = 143000
ELO_DRAWS = 1_000
ELO_SEED = 143001
DISK_FREE_MIN_BYTES = 40 * 10**9
MIN_NICE = 19
METRICS_LIMIT_BYTES = 100_000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ admission rules (pure, unit-tested)


def family_gains(cand, inc, families, drop_invalid):
    """Per-family mean of candidate - incumbent per-context utility. With drop_invalid, contexts where either side
    picked an invalid program (-2) are excluded and families left empty are dropped."""
    sums, counts = defaultdict(float), defaultdict(int)
    for c, i, f in zip(cand, inc, families):
        if drop_invalid and (c == INVALID or i == INVALID):
            continue
        sums[f] += c - i
        counts[f] += 1
    return np.asarray([sums[f] / counts[f] for f in sorted(counts)])


def strict_accept(cand, inc, families, rng, drop_invalid=False):
    """R_STRICT: accept iff the median over DRAWS family-cluster resamples of the equal-family mean gain > MARGIN.
    Exact ties (zero gain everywhere) are therefore rejected."""
    g = family_gains(cand, inc, families, drop_invalid)
    if len(g) == 0:
        return False, 0.0
    draws = g[rng.integers(0, len(g), size=(DRAWS, len(g)))].mean(axis=1)
    med = float(np.median(draws))
    return med > MARGIN, med


def live_accept(cand_mean, inc_mean):
    """S_LIVE as logged: accept an improvement or an exact tie of the 91-context mean."""
    if cand_mean > inc_mean:
        return True, "accept_improvement"
    if cand_mean == inc_mean:
        return True, "accept_tie"
    return False, "restore_best_model_and_optimizer"


def replay(stream, start, families, rule, seed=None):
    """stream: list of dicts(round, per_context, mean). start: (round, per_context, mean). Returns the trajectory."""
    inc_round, inc_pc, inc_mean = start
    rng = np.random.default_rng(seed) if seed is not None else None
    admissions, log = [], []
    for s in stream:
        cand_inv, inc_inv = (
            sum(v == INVALID for v in s["per_context"]),
            sum(v == INVALID for v in inc_pc),
        )
        if rule == "S_LIVE":
            ok, action = live_accept(s["mean"], inc_mean)
            stat = s["mean"] - inc_mean
        else:
            ok, stat = strict_accept(
                s["per_context"],
                inc_pc,
                families,
                rng,
                drop_invalid=(rule == "R_STRICT_NOINV"),
            )
            action = "accept" if ok else "reject"
        log.append(
            {
                "round": s["round"],
                "accepted": ok,
                "action": action,
                "stat": stat,
                "candidate_invalid": cand_inv,
                "incumbent_invalid": inc_inv,
            }
        )
        if ok:
            admissions.append(s["round"])
            inc_round, inc_pc, inc_mean = s["round"], s["per_context"], s["mean"]
    return {
        "admissions": admissions,
        "final_round": inc_round,
        "final_val_mean": float(np.mean(inc_pc)),
        "final_val_invalid": int(sum(v == INVALID for v in inc_pc)),
        "log": log,
    }


def cliff_driven(stream_by_round, admissions, start_pc, families):
    """Admissions whose invalid-excluded family gain does not clear the margin (deterministic median of family
    gains, a descriptive check)."""
    inc, out = start_pc, []
    for r in admissions:
        cand = stream_by_round[r]["per_context"]
        g = family_gains(cand, inc, families, drop_invalid=True)
        if len(g) == 0 or float(np.mean(g)) <= MARGIN:
            out.append(r)
        inc = cand
    return out


def verdict(dl):
    """dl: {seed: {"available": bool, "ci95": [lo, hi]}} for R_STRICT's final incumbent vs R882."""
    avail = [s for s, v in dl.items() if v["available"]]
    if len(avail) < 2:
        return "INCOMPLETE_COVERAGE"
    helps = sum(1 for s in avail if dl[s]["ci95"][0] > 0)
    hurts = sum(1 for s in avail if dl[s]["ci95"][1] < 0)
    if helps >= 2:
        return "RULE_HELPS"
    if hurts >= 2:
        return "RULE_HURTS"
    return "RULE_NEUTRAL"


# ------------------------------------------------------------------ inputs


def load_stream():
    if (
        sha(INPUTS / "receipts.SHA256SUMS") != RECEIPTS_SUMS_SHA
        or sha(INPUTS / ADMISSIONS[0]) != ADMISSIONS[1]
    ):
        raise RuntimeError("pinned receipts or admissions log changed")
    sums = dict(
        line.split()[::-1]
        for line in (INPUTS / "receipts.SHA256SUMS").read_text().splitlines()
    )
    adm = [
        json.loads(line) for line in (INPUTS / ADMISSIONS[0]).read_text().splitlines()
    ]
    if [a["round"] for a in adm] != list(range(ROUNDS[0], ROUNDS[1] + 1)):
        raise RuntimeError("admission window is not contiguous 775-2530")
    stream, dev_after = [], {}
    first_initial = None
    for a in adm:
        path = INPUTS / "receipts" / f"round-{a['round']}.json"
        if sha(path) != sums[path.name]:
            raise RuntimeError(f"receipt changed: {path.name}")
        r = json.loads(path.read_text())
        if (
            r["evaluation_identity"] != EVALUATION_IDENTITY
            or a["evaluation_identity"] != EVALUATION_IDENTITY
        ):
            raise RuntimeError("evaluation identity mismatch")
        c = r["candidate_dev_utility"]
        if len(c["per_context"]) != 91 or c["mean"] != a["candidate_utility"]:
            raise RuntimeError(
                f"round {a['round']}: candidate utility does not match the admissions log"
            )
        stream.append(
            {
                "round": a["round"],
                "per_context": [float(v) for v in c["per_context"]],
                "mean": c["mean"],
                "logged_action": a["action"],
                "logged_accepted": a["accepted"],
            }
        )
        dev_after[a["round"]] = r["dev_utility"]
        if first_initial is None:
            first_initial = r["initial_dev_utility"]
    start = (
        774,
        [float(v) for v in first_initial["per_context"]],
        adm[0]["previous_best_utility"],
    )
    return stream, start, dev_after


def val_families():
    selected = H50 / "selected.json"
    if sha(selected) != SELECTED_SHA:
        raise RuntimeError("selected packet changed")
    packet = json.loads(selected.read_text())
    ref = json.loads((INPUTS / REFERENCE[0]).read_text())
    if (
        sha(INPUTS / REFERENCE[0]) != REFERENCE[1]
        or ref["evaluation_identity"] != EVALUATION_IDENTITY
    ):
        raise RuntimeError("reference contract changed")
    if [c["cid"] for c in ref["evaluation_contract"]["contexts"]] != [
        r["context_sha256"] for r in packet["dev"]
    ]:
        raise RuntimeError("validation contract order mismatch")
    return [r["family_id"] for r in packet["dev"]], ref, packet


def guards(path):
    if os.getpriority(os.PRIO_PROCESS, 0) < MIN_NICE:
        raise RuntimeError("H143 requires nice >= 19")
    free = shutil.disk_usage(path).free
    if free < DISK_FREE_MIN_BYTES:
        raise RuntimeError(f"host free {free} B < 40 GB")
    return {"disk_free_bytes": free}


def verify_prereg_committed():
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", H143_PREREG_COMMIT, source],
        cwd=REPO,
        check=True,
    )
    expected = subprocess.check_output(
        [
            "git",
            "show",
            f"{H143_PREREG_COMMIT}:research/flywheel_selfplay/h143_strict_admission_replay/PREREG.md",
        ],
        cwd=REPO,
    )
    if expected != (HERE / "PREREG.md").read_bytes():
        raise RuntimeError("H143 PREREG bytes changed after the preregistration commit")
    return source, hashlib.sha256(expected).hexdigest()


# ------------------------------------------------------------------ stage 1: replay


def stage_replay(root):
    root = Path(root)
    guards(root.parent)
    source, prereg_sha = verify_prereg_committed()
    stream, start, dev_after = load_stream()
    families, ref, _ = val_families()
    live = replay(stream, start, families, "S_LIVE")
    mismatches = [
        s["round"]
        for s, l in zip(stream, live["log"])
        if (s["logged_action"], s["logged_accepted"]) != (l["action"], l["accepted"])
    ]
    expected = [float(v) for v in ref["utility"]["per_context"]]
    by_round = {s["round"]: s for s in stream}
    final_pc = (
        by_round[live["final_round"]]["per_context"]
        if live["final_round"] in by_round
        else start[1]
    )
    regression = {
        "logged_action_mismatches": mismatches,
        "final_round": live["final_round"],
        "final_matches_reference_91": int(
            sum(abs(a - b) <= 1e-6 for a, b in zip(final_pc, expected))
        ),
        "final_mean": float(np.mean(final_pc)),
        "admissions": live["admissions"],
    }
    if (
        mismatches
        or live["final_round"] != 882
        or regression["final_matches_reference_91"] != 91
        or abs(regression["final_mean"] - 1.1377942) > 1e-6
    ):
        raise RuntimeError(f"S_LIVE regression FAIL: {regression}")
    arms = {}
    for arm in ARMS:
        for seed in SEEDS:
            t = replay(stream, start, families, arm, seed)
            arms[f"{arm}/{seed}"] = {
                "admissions": t["admissions"],
                "n_admissions": len(t["admissions"]),
                "final_round": t["final_round"],
                "final_val_mean": t["final_val_mean"],
                "final_val_invalid": t["final_val_invalid"],
                "cliff_driven_admissions": cliff_driven(
                    by_round, t["admissions"], start[1], families
                ),
                "max_median_gain": max(x["stat"] for x in t["log"]),
            }
    agreement = {
        arm: len({tuple(arms[f"{arm}/{s}"]["admissions"]) for s in SEEDS}) == 1
        for arm in ARMS
    }
    out = {
        "source_commit": source,
        "prereg_sha256": prereg_sha,
        "window": list(ROUNDS),
        "rounds": len(stream),
        "regression_s_live": regression,
        "s_live": {
            "admissions": live["admissions"],
            "final_round": live["final_round"],
            "final_val_mean": live["final_val_mean"],
            "final_val_invalid": live["final_val_invalid"],
            "cliff_driven_admissions": cliff_driven(
                by_round, live["admissions"], start[1], families
            ),
        },
        "arms": arms,
        "trajectory_identical_across_seeds": agreement,
        "start": {"round": 774, "val_mean": start[2]},
    }
    root.mkdir(parents=True, exist_ok=False)
    (root / "replay.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(
        "REPLAY",
        json.dumps({k: (v["final_round"], v["n_admissions"]) for k, v in arms.items()}),
        flush=True,
    )
    return out


# ------------------------------------------------------------------ stage 2: DL-Train


def native(cost, fitness, gate):
    cost, fitness, gate = np.asarray(cost), np.asarray(fitness), np.asarray(gate)
    if (
        cost.ndim != 1
        or cost.size == 0
        or cost.shape != fitness.shape
        or not np.isfinite(cost).all()
    ):
        raise ValueError("invalid complete pool")
    i = int(np.argmin(cost))
    return float(-fitness[i] if gate[i] else -2.0)


def load_h124():
    if sha(H124_PATH) != H124_SHA:
        raise RuntimeError("H124 module differs from the pinned bytes")
    spec = importlib.util.spec_from_file_location("h143_h124", H124_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stage_dltrain(root):
    root = Path(root)
    replayed = json.loads((root / "replay.json").read_text())
    source, prereg_sha = verify_prereg_committed()
    if source != replayed["source_commit"]:
        raise RuntimeError("dltrain must run from the same source commit as replay")
    h124 = load_h124()
    launch = dict(h124.preflight(root.parent), **guards(root.parent))
    t0, cpu0 = time.monotonic(), time.process_time()
    families_val, ref, packet = val_families()
    finals = sorted({882} | {v["final_round"] for v in replayed["arms"].values()})
    available = {r: (INPUTS / f, s) for r, (f, s) in CHECKPOINTS.items()}
    extra = (
        INPUTS / "extra_checkpoints.json"
    )  # archived rejected candidates fetched after replay (PREREG step 3)
    if extra.exists():
        for r, e in json.loads(extra.read_text()).items():
            if int(r) not in {v["final_round"] for v in replayed["arms"].values()}:
                raise RuntimeError("extra checkpoint is not a replayed final incumbent")
            available[int(r)] = (INPUTS / e["file"], e["sha256"])
    for p, digest in h124.FILES.items():
        if p.name in ("source_manifest.json", "cache_manifest.json", "round-882.pt"):
            continue
        if sha(p) != digest:
            raise RuntimeError(f"dependency hash mismatch: {p}")
    train, val = packet["train"], packet["dev"]
    rows = val + train
    with h124.RssWatchdog(root) as watchdog:
        context, reads = h124.train_context(rows, [])
        projection = h124.train_projection(rows, [])
        pools = {}
        for r in rows:
            path = H50 / "pools" / f"{r['context_sha256']}.npz"
            if sha(path) != r["pool_sha256"]:
                raise RuntimeError("pool identity mismatch")
            with np.load(path, allow_pickle=False) as z:
                fit, gate = z["fitness"].astype(np.float64), z["gate"].astype(bool)
            pools[r["context_sha256"]] = dict(
                path=path,
                pool_sha256=r["pool_sha256"],
                fitness=fit,
                gate=gate,
                u=np.where(gate, -fit, -2.0),
            )
        scored, dope = {}, {}
        for rnd in finals:
            if rnd not in available:
                continue
            guards(root)
            h124.headroom(root)
            path, digest = available[rnd]
            if sha(path) != digest:
                raise RuntimeError(f"checkpoint {rnd} changed")
            h124.R882 = path
            h124.FILES[path] = digest
            arm_root = root / f"R{rnd}"
            arm_root.mkdir()
            costs, net = h124.baseline(arm_root, rows, context, pools, projection)
            del net
            h124.release_memory()
            (arm_root / "r882-copy.pt").unlink()
            dope[rnd] = json.loads((arm_root / "dope-secondary.json").read_text())
            pv = [
                native(
                    costs[r["context_sha256"]],
                    pools[r["context_sha256"]]["fitness"],
                    pools[r["context_sha256"]]["gate"],
                )
                for r in val
            ]
            scored[rnd] = {
                "val": pv,
                "train": [
                    native(
                        costs[r["context_sha256"]],
                        pools[r["context_sha256"]]["fitness"],
                        pools[r["context_sha256"]]["gate"],
                    )
                    for r in train
                ],
                "u4": [
                    h124.endpoints(
                        costs[r["context_sha256"]], pools[r["context_sha256"]]["u"]
                    )["u4"]
                    for r in train
                ],
            }
            if rnd == 882:
                expected = np.asarray(ref["utility"]["per_context"], np.float64)
                if int(np.sum(np.abs(np.asarray(pv) - expected) <= 1e-6)) != 91:
                    raise RuntimeError(
                        "R882 rescoring does not reproduce the 91-context reference"
                    )
        result = readout(h124, replayed, train, scored)
        if any(
            d["before"] != h124.DOPE or d["after"] != h124.DOPE for d in dope.values()
        ):
            raise RuntimeError("DOPE identity changed")
        per_raw = json.dumps(
            {str(k): v for k, v in scored.items()}, sort_keys=True
        ).encode()
        import sklearn
        import torch

        metrics = {
            "schema": "jope.h143.strict-admission-replay/v1",
            "status": "COMPLETE",
            "verdict": result.pop("verdict"),
            "source_commit": replayed["source_commit"],
            "prereg_commit": H143_PREREG_COMMIT,
            "prereg_sha256": replayed["prereg_sha256"],
            "runner_sha256": sha(Path(__file__)),
            "h124_module_sha256": H124_SHA,
            "window": replayed["window"],
            "rounds": replayed["rounds"],
            "regression_s_live": replayed["regression_s_live"],
            "s_live": replayed["s_live"],
            "replay_arms": replayed["arms"],
            "trajectory_identical_across_seeds": replayed[
                "trajectory_identical_across_seeds"
            ],
            **result,
            "per_incumbent_sha256": hashlib.sha256(per_raw).hexdigest(),
            "dope": {
                str(k): {"before": d["before"], "after": d["after"]}
                for k, d in dope.items()
            },
            "read_boundary": {
                "dl_train_contexts": len(train),
                "dl_val_flywheel_contract_contexts": len(val),
                "sealed_payloads_opened": 0,
                "holdout_payloads_opened": 0,
                "meta_test_payloads_opened": 0,
                "live_dir_writes": 0,
            },
            "runtime": {
                "wall_seconds": time.monotonic() - t0,
                "process_cpu_seconds": time.process_time() - cpu0,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "rss_watchdog": watchdog.receipt(),
                "launch": launch,
            },
            "versions": {
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
                "python": platform.python_version(),
            },
            "optimizer_updates": 0,
            "GPU_used": False,
            "CUDA_initialized": torch.cuda.is_initialized(),
        }
        raw = json.dumps(metrics, indent=1, sort_keys=True, allow_nan=False) + "\n"
        if len(raw.encode()) >= METRICS_LIMIT_BYTES:
            raise RuntimeError("H143 metrics would exceed 100 KB")
        (root / "per_incumbent.json").write_bytes(per_raw)
        (root / "metrics.json").write_text(raw)
    print("COMPLETE", metrics["verdict"], flush=True)
    return metrics


def readout(h124, replayed, train, scored):
    families = [r["family_id"] for r in train]
    order = sorted(set(families))
    idx = {f: k for k, f in enumerate(order)}

    def fam(v):
        s, c = np.zeros(len(order)), np.zeros(len(order))
        for x, f in zip(v, families):
            s[idx[f]] += x
            c[idx[f]] += 1
        return s / c

    samples = np.random.default_rng(DLTRAIN_SEED).integers(
        0, len(order), size=(DLTRAIN_DRAWS, len(order))
    )
    ref = fam(scored[882]["train"])
    incumbents = {}
    for rnd, v in scored.items():
        g = fam(v["train"]) - ref
        boot = g[samples].mean(axis=1)
        incumbents[str(rnd)] = {
            "dl_train_mean_gain": float(g.mean()),
            "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "families_win_tie_loss": [
                int((g > 1e-12).sum()),
                int((np.abs(g) <= 1e-12).sum()),
                int((g < -1e-12).sum()),
            ],
            "u4_mean_gain_descriptive": float(
                (fam(v["u4"]) - fam(scored[882]["u4"])).mean()
            ),
            "dl_train_invalid_picks": int(sum(x == INVALID for x in v["train"])),
            "val_mean_rescored": float(np.mean(v["val"])),
        }
    per_arm = {}
    for key, a in replayed["arms"].items():
        r = str(a["final_round"])
        per_arm[key] = {
            "final_round": a["final_round"],
            "available": r in incumbents,
            **(
                {
                    "ci95": incumbents[r]["ci95"],
                    "dl_train_mean_gain": incumbents[r]["dl_train_mean_gain"],
                }
                if r in incumbents
                else {"ci95": None}
            ),
        }
    strict = {
        k.split("/")[1]: v for k, v in per_arm.items() if k.startswith("R_STRICT/")
    }
    return {
        "verdict": verdict(strict),
        "dl_train_by_incumbent": incumbents,
        "dl_train_by_arm": per_arm,
        "elo": elo(h124, train, scored),
    }


def elo(h124, train, scored):
    helper = h124.source_module("python/src/jope/north_star/elo_tracker.py", "h143_elo")
    contenders = sorted(scored)
    rows_by_family = defaultdict(list)
    for j, r in enumerate(train):
        rows_by_family[r["family_id"]].append((j, r))
    if len(contenders) < 2:
        return {"ratings": {"R882": 1500.0}, "note": "only R882 scored"}

    def ratings(fams):
        recs = [
            helper.DatasetEvalRecord(
                f"{i}:{r['context_sha256']}",
                f"{i}:{f}",
                "regression",
                {f"R{c}": scored[c]["train"][j] for c in contenders},
            )
            for i, f in enumerate(fams)
            for j, r in rows_by_family[f]
        ]
        return helper.compute_family_grouped_bradley_terry_elo(
            recs, "R882", anchor_elo=1500.0, tie_margin=1e-4, prior=0.01, n_iters=400
        )["ratings"]

    order = sorted(rows_by_family)
    point = ratings(order)
    rng = np.random.default_rng(ELO_SEED)
    draws = defaultdict(list)
    for _ in range(ELO_DRAWS):
        for a, v in ratings(
            [order[k] for k in rng.integers(0, len(order), len(order))]
        ).items():
            draws[a].append(v)
    return {
        "ratings": point,
        "ci95": {
            a: [float(x) for x in np.quantile(v, [0.025, 0.975])]
            for a, v in draws.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("replay", "dltrain"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "replay":
        stage_replay(args.root)
    else:
        stage_dltrain(args.root)


if __name__ == "__main__":
    main()

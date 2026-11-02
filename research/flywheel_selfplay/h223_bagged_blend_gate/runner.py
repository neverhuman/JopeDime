"""One monitored H223 CPU attempt: GATED_BAG3 (r882 + 0.25 x (calibrated mean of 3 HistGBM members - r882), gated by
the outcome-free zgap signal with H210's frozen FIT-only threshold, else r882) vs R882 on the flywheel native
first-argmin protocol, 205-family pooled cross-fit (H199/H210/H217 machinery, fresh seeds/folds), with GATED_025 as
the frozen reference, the H129 disagreement gate on the bagged blend as placebo, and the DL-Val flywheel91 plateau as a
co-primary.

Reuses, byte-pinned and unchanged: the merged H124 module (load_data, train_context/projection, baseline forward,
calibration, endpoints, program; via the merged H139 loader) and the merged H139 roster/pools/native helpers (the H152
regression path; `native` is the flywheel protocol fixed-dev-cost-argmin-negative-fitness/v1). `fit_unit` is H217's
`fit_gbm` (H124 `fit_one` with the H195 exclusion sets) building the design matrix once and fitting three members on
it: member 0 is the unweighted H217 fit (GATED_025's component, bit for bit), members 1-2 refit with family-bootstrap
multiplicities as sample weights. H217 operations carry over (effective-affinity limits, absolute deadline, pre-claim
source check); hw_t1325 sets the resources: <= 8 worker cores, RSS < 4 GiB, load5 <= 60, free disk >= 300 GB.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import socket
import subprocess
import sys
import time

import numpy as np

import analysis as stat

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = Path.home() / "jope-data/h223"
INPUTS = DATA / "inputs"
PREREG_SHA = "dd7538a62e9c06c6853400640fd631cabf83a5d935e375ebfd09c01f518f4186"
H139_PATH = REPO / "research/flywheel_selfplay/h139_selection_audit/runner.py"
H139_SHA = "7a37821db82c89ddf8e48266383c681b20535ff31b95504e259c74dbe5a9c686"
BOUND_FILES = (
    "research/flywheel_selfplay/h223_bagged_blend_gate/PREREG.md",
    "research/flywheel_selfplay/h223_bagged_blend_gate/runner.py",
    "research/flywheel_selfplay/h223_bagged_blend_gate/analysis.py",
    "research/flywheel_selfplay/h223_bagged_blend_gate/monitor.py",
    "research/flywheel_selfplay/h139_selection_audit/runner.py",
    "research/flywheel_selfplay/h124_gbm_shrink_blend/runner.py",
    "python/src/jope/north_star/dope_freeze_guard.py",
)
HOST_RULES = {
    "xbabe2": dict(min_free=300_000_000_000, max_used=None),  # hw_t1325: free >= 300 GB
}
MEM_MIN = 20 * 2**30  # hw_t1325: xbabe2 swap is full; stop below 20 GiB available
LOAD5_MAX = 60.0  # amendment A1 (hw_t1400): pause before a unit while load5 > 60
LOAD5_RESUME = 55.0  # ... until load5 <= 55 (rechecked every 60 s)
LOAD_POLL = 60.0
CORES = len(
    os.sched_getaffinity(0)
)  # effective allocation (H210 E1), not the online count
DEADLINE_ENV = "H223_DEADLINE_EPOCH"  # absolute deadline from the attempt claim, set by the monitor
SCRATCH_MAX = 1_000_000_000  # 135 prediction files (H217: 45 in 155 MB)
RSS_LIMIT = 4 * 2**30  # hw_t1325: process-tree RSS < 4 GiB
METRICS_LIMIT = 100_000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jsonable(o):
    """Gate thresholds include -inf (closed) and +inf (always open): write them as strings; NaN stays an error."""
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, float) and math.isinf(o):
        return "inf" if o > 0 else "-inf"
    return o


def arm_costs(stat_mod, base_cost, priors, taus):
    """Every H223 arm's cost for one context and the gate decisions (outcome-free: costs only). `priors` maps SINGLE
    (member 0) and BAG (member mean) to calibrated costs; each prior gets H217's blend and signals unchanged."""
    blended = {k: stat_mod.blend(base_cost, p) for k, p in priors.items()}
    signal = {
        k: dict(
            zgap=stat_mod.zgap(base_cost, blended[k]),
            disagreement=stat_mod.disagreement(p, base_cost),
        )
        for k, p in priors.items()
    }
    opened = {
        arm: stat_mod.gated(signal[stat_mod.PRIORS[arm]][name], taus[arm])
        for arm, name in stat_mod.GATES.items()
    }
    costs = {"R882": base_cost}
    costs.update(
        {
            arm: blended[stat_mod.PRIORS[arm]]
            for arm in stat_mod.ARMS
            if arm.startswith("BLEND_")
        }
    )
    costs.update(
        {
            arm: blended[stat_mod.PRIORS[arm]] if opened[arm] else base_cost
            for arm in stat_mod.GATES
        }
    )
    return costs, opened, signal


def save(path, obj, exclusive=False):
    with Path(path).open("x" if exclusive else "w") as f:
        f.write(
            json.dumps(jsonable(obj), indent=1, sort_keys=True, allow_nan=False) + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def load(path, name, digest=None):
    if digest is not None and sha(path) != digest:
        raise RuntimeError(f"pinned module bytes changed: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def staged(path):
    """Map a pinned ~/jope-data input path into the H223 staging layout (same relative path under DATA/inputs)."""
    base = Path.home() / "jope-data"
    return (
        INPUTS / Path(path).relative_to(base)
        if Path(path).is_relative_to(base)
        else Path(path)
    )


def load_modules():
    """Pinned H139/H124 bytes, pointed at the H223 staging layout (~/jope-data/h223/inputs on xbabe2, the only admitted host).
    Every staged file keeps its pinned SHA-256 (verified by the inherited loaders); only the directory changes."""
    h139 = load(H139_PATH, "h223_h139", H139_SHA)
    h124 = h139.load_h124()
    h124.RSS_LIMIT_KIB = RSS_LIMIT // 1024
    h124.INPUT, h124.CACHE, h124.R882 = (
        staged(p) for p in (h124.INPUT, h124.CACHE, h124.R882)
    )
    h124.FILES = {staged(p): d for p, d in h124.FILES.items()}
    h139.INPUT = h124.INPUT
    h139.ARM_DIR = INPUTS  # pinned best_dev_checkpoint.json copy (H139 REFERENCE sha)
    return h139, h124


def tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def admit_host():
    """Preregistered host boundary: xbabe2 only, at nice 19. Called before any read, write or launch."""
    host = socket.gethostname().split(".")[0]
    if host not in HOST_RULES or os.getpriority(os.PRIO_PROCESS, 0) < 19:
        raise RuntimeError(f"H223 runs on xbabe2 at nice 19 only (host {host!r})")
    return host


def deadline():
    """The absolute deadline (epoch seconds) the monitor derived from the attempt claim; None outside a monitored run."""
    value = os.environ.get(DEADLINE_ENV)
    return None if value is None else float(value)


def guards(root):
    """xbabe2 disk rule (hw_t1325: free >= 300 GB), MemAvailable >= 20 GiB, nice 19, own root < 1 GB, and the
    absolute deadline: hard aborts, checked before every unit and stage. Host load is not an abort here (amendment
    A1, hw_t1400): see wait_for_load."""
    host = admit_host()
    rule = HOST_RULES[host]
    end = deadline()
    if end is not None and time.time() >= end:
        raise RuntimeError(f"absolute deadline {end} passed")
    usage = shutil.disk_usage(Path.home())
    used = usage.used / (usage.used + usage.free)
    if usage.free < rule["min_free"] or (
        rule["max_used"] is not None and used >= rule["max_used"]
    ):
        raise RuntimeError(f"disk guard: free {usage.free} B, used {used:.5f}")
    mem = (
        next(
            int(p[1])
            for p in (
                ln.split() for ln in Path("/proc/meminfo").read_text().splitlines()
            )
            if p and p[0] == "MemAvailable:"
        )
        * 1024
    )
    load1, load5, _ = os.getloadavg()
    if mem < MEM_MIN:
        raise RuntimeError(f"host guard: MemAvailable {mem} B, load5 {load5}")
    scratch = tree_bytes(root) if Path(root).exists() else 0
    if scratch >= SCRATCH_MAX:
        raise RuntimeError(f"own output root {scratch} B >= 1 GB")
    return dict(
        host=host,
        effective_cores=CORES,
        deadline_epoch=end,
        free=usage.free,
        used=round(used, 5),
        mem_available=mem,
        load1=load1,
        load5=load5,
        scratch=scratch,
    )


def wait_for_load():
    """Amendment A1 (hw_t1400): before a unit, if load5 > 60, wait and recheck every 60 s until load5 <= 55. The
    absolute deadline still ends the attempt while it waits. Returns the seconds waited."""
    load5 = os.getloadavg()[1]
    if load5 <= LOAD5_MAX:
        return 0.0
    t0 = time.monotonic()
    print("LOAD PAUSE load5", round(load5, 1), flush=True)
    while os.getloadavg()[1] > LOAD5_RESUME:
        end = deadline()
        if end is not None and time.time() >= end:
            raise RuntimeError(f"absolute deadline {end} passed while paused for load")
        time.sleep(LOAD_POLL)
    waited = time.monotonic() - t0
    print("LOAD RESUME after", round(waited), "s", flush=True)
    return waited


def bind_source(launch):
    """Squash-compatible binding (H152): PREREG hash, and every bound file equals its blob at the reviewed head AND at
    the merged source commit."""
    if sha(REPO / BOUND_FILES[0]) != PREREG_SHA:
        raise RuntimeError("PREREG bytes differ from the preregistration")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    if head != launch["source_commit"]:
        raise RuntimeError("checkout HEAD is not the merged source commit")
    bound = {}
    for rel in BOUND_FILES:
        local = (REPO / rel).read_bytes()
        for commit in (launch["reviewed_head"], launch["source_commit"]):
            if (
                subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=REPO)
                != local
            ):
                raise RuntimeError(f"{rel}: executed bytes differ from {commit}")
        bound[rel] = hashlib.sha256(local).hexdigest()
    return bound


def bounded_threads(fn, *args, **kwargs):
    """The pinned H124 baseline hardcodes torch.set_num_threads(4); hold it at 1 (as H152)."""
    from unittest.mock import patch

    import torch

    setter = torch.set_num_threads

    def capped(requested):
        setter(1)

    setter(1)
    with patch.object(torch, "set_num_threads", capped):
        return fn(*args, **kwargs)


def unit_name(tag, seed, excluded, member):
    return f"{tag}-seed{seed}-exclude{''.join(map(str, excluded))}-m{member}"


def fit_unit(
    h124, tag, seed, excluded, rows, mapping, context, pools, samples, root, val=None
):
    """H217 fit_gbm with three members on one design matrix: exclusion size 2 (outer {A,B}) or 3 (inner {A,B,j}).
    Member 0 is H217's unweighted fit (same features, targets, model and random_state); members 1-2 are refits with
    family-bootstrap multiplicities as sample weights. Predictions for the excluded folds' contexts (and DL-Val
    contexts when given), one file per member."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    if (
        seed not in stat.SEEDS
        or tag != "BLEND"
        or len(excluded) not in (2, 3)
        or tuple(sorted(set(excluded))) != excluded
    ):
        raise ValueError("unregistered fit")
    outs = [
        root / (unit_name(tag, seed, excluded, m) + ".npz") for m in range(stat.MEMBERS)
    ]
    if any(out.exists() for out in outs):
        raise RuntimeError("fit already exists")
    t0 = time.monotonic()
    fit_rows = [r for r in rows if mapping[r["context_sha256"]] not in excluded]
    x = np.empty((len(fit_rows) * 192, 1089), dtype=np.float64)
    targets, fams = [], []
    for i, row in enumerate(fit_rows):
        cid = row["context_sha256"]
        p, index = pools[cid], samples[cid]
        cf = np.concatenate([context[cid]["lm48_z"], context[cid]["dope629"]]).astype(
            np.float32
        )
        x[i * 192 : (i + 1) * 192, :412] = h124.program(pools, cid)[index]
        x[i * 192 : (i + 1) * 192, 412:] = cf
        targets.append((p["u"][index] - p["u"].mean()).astype(np.float32))
        fams.extend([row["family_id"]] * 192)
    y = np.concatenate(targets)
    models, walls, weights_sha = [], [], []
    for member in range(stat.MEMBERS):
        t1 = time.monotonic()
        weights = stat.member_weights(fams, seed, excluded, member)
        h124.check_rss()
        model = HistGradientBoostingRegressor(**h124.MODEL, random_state=seed)
        models.append(
            model.fit(x, y)
            if weights is None
            else model.fit(x, y, sample_weight=weights)
        )
        walls.append(time.monotonic() - t1)
        weights_sha.append(
            None if weights is None else hashlib.sha256(weights.tobytes()).hexdigest()
        )
        del weights
    del x, y
    h124.release_memory()
    predictions = [{} for _ in models]
    targets_rows = [
        (r, pools, context) for r in rows if mapping[r["context_sha256"]] in excluded
    ]
    if val is not None:
        targets_rows += [(r, val[1], val[0]) for r in val[2]]
    for row, pl, ctx in targets_rows:
        cid = row["context_sha256"]
        cf = np.concatenate([ctx[cid]["lm48_z"], ctx[cid]["dope629"]]).astype(
            np.float32
        )
        xp = h124.program(pl, cid)
        xp = np.concatenate([xp, np.repeat(cf[None], len(xp), axis=0)], axis=1)
        for member, model in enumerate(models):
            predictions[member][cid] = model.predict(xp).astype(np.float32)
    unit_wall = time.monotonic() - t0
    receipts = []
    for member, (out, model) in enumerate(zip(outs, models)):
        np.savez_compressed(out, **predictions[member])
        receipt = dict(
            tag=tag,
            seed=seed,
            member=member,
            excluded_folds=list(excluded),
            fit_contexts=len(fit_rows),
            fit_rows=len(fit_rows) * 192,
            sample_weight_sha256=weights_sha[member],
            iterations=int(model.n_iter_),
            prediction_sha256=sha(out),
            predicted_contexts=len(predictions[member]),
            fit_wall_seconds=walls[member],
            unit_wall_seconds=unit_wall,
            shuffled=False,
        )
        save(out.with_suffix(".json"), receipt)
        receipts.append(receipt)
    print(
        "FIT",
        unit_name(tag, seed, excluded, "*"),
        round(time.monotonic() - t0, 1),
        [round(w, 1) for w in walls],
        flush=True,
    )
    return receipts


def verified_unit(root, tag, seed, excluded):
    """Single-resume rule: an existing unit is reused only if every member's receipt and npz hash agree."""
    receipts = []
    for member in range(stat.MEMBERS):
        name = unit_name(tag, seed, excluded, member)
        npz, js = root / (name + ".npz"), root / (name + ".json")
        if npz.exists() and js.exists():
            receipt = json.loads(js.read_text())
            if receipt["prediction_sha256"] == sha(npz):
                receipts.append(receipt)
                continue
        break
    if len(receipts) == stat.MEMBERS:
        return receipts
    for member in range(stat.MEMBERS):
        name = unit_name(tag, seed, excluded, member)
        for p in (root / (name + ".npz"), root / (name + ".json")):
            if p.exists():
                p.rename(
                    p.with_name(p.name + ".incomplete")
                )  # kept for custody, never deleted
    return None


def run(root, launch_path, resume=False):
    admit_host()  # before any import, read, root creation or claim
    if deadline() is None:
        raise RuntimeError(
            "H223 runs only under its monitor (absolute deadline missing)"
        )
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    root = Path(root)
    if root.parent != DATA or not root.name.startswith("run-"):
        raise RuntimeError("H223 requires a run-* root under ~/jope-data/h223")
    launch = json.loads(Path(launch_path).read_text())
    bound = bind_source(launch)
    if resume:
        if (
            not root.exists()
            or (root / "resume-identity.json").exists()
            or (root / "metrics.json").exists()
        ):
            raise RuntimeError("single resume only, of an existing incomplete root")
        save(
            root / "resume-identity.json",
            dict(utc=time.time(), bound=bound),
            exclusive=True,
        )
    else:
        if root.exists():
            raise RuntimeError("fresh root required")
        root.mkdir(parents=True)
    launch_guard = guards(root)
    h139, h124 = load_modules()
    t0, cpu0 = time.monotonic(), time.process_time()
    save(
        root
        / ("execution-identity-resume.json" if resume else "execution-identity.json"),
        dict(
            launch=launch,
            bound_sha256=bound,
            launch_guard=launch_guard,
            prereg_sha256=PREREG_SHA,
            host=socket.gethostname(),
            environment={
                k: os.environ.get(k)
                for k in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "CUDA_VISIBLE_DEVICES",
                )
            },
        ),
    )

    # 1. Regression first: exact native r882 on the 91 DL-Val contexts (H152 path).
    train_rows, val, reference = h139.roster(h124)
    # The cache closure admits only the full roster (as H152): read val + train, keep the 91 DL-Val rows.
    val_ctx, _ = h124.train_context(val + train_rows, [])
    val_ctx = {r["context_sha256"]: val_ctx[r["context_sha256"]] for r in val}
    val_proj = h124.train_projection(val + train_rows, [])
    val_pools = h139.pools_for(val)
    vroot = root / ("regression-resume" if resume else "regression")
    vroot.mkdir()
    val_costs, net = bounded_threads(
        h124.baseline, vroot, val, val_ctx, val_pools, val_proj
    )
    got = np.array(
        [
            h139.native(
                val_costs[r["context_sha256"]],
                val_pools[r["context_sha256"]]["fitness"],
                val_pools[r["context_sha256"]]["gate"],
            )
            for r in val
        ]
    )
    if (
        not np.array_equal(got, np.asarray(reference["utility"]["per_context"]))
        or float(got.mean()) != reference["best_utility"]
    ):
        raise RuntimeError("R882 91/91 native regression failed")
    del net
    h124.release_memory()
    print("BASELINE PASS: 91/91 native r882 exact", flush=True)

    # 2. TRAIN r882 costs through the exact H129 path.
    h124.FOLD_SEED, h124.SAMPLE_SEED = stat.FOLD_SEEDS[stat.SEEDS[0]], stat.SAMPLE_SEED
    troot = root / ("train-resume" if resume else "train")
    troot.mkdir()
    selected, context, pools, samples, _, reads = h124.load_data(troot)
    rows = selected["train"]
    if [r["context_sha256"] for r in rows] != [r["context_sha256"] for r in train_rows]:
        raise RuntimeError("roster mismatch")
    projection = h124.train_projection(rows, selected["dev"])
    base, net = bounded_threads(h124.baseline, troot, rows, context, pools, projection)
    del net
    h124.release_memory()
    dope = json.loads((troot / "dope-train.json").read_text())
    if dope["before"] != h124.DOPE or dope["after"] != h124.DOPE:
        raise RuntimeError("DOPE changed")
    fams = [r["family_id"] for r in rows]
    folds = {s: stat.seed_folds(fams, s) for s in stat.SEEDS}
    if json.loads((HERE / "fold_manifest.json").read_text())["families_by_seed"] != {
        str(s): g for s, g in folds.items()
    }:
        raise RuntimeError("fold manifest identity")
    if len({json.dumps(g) for g in folds.values()}) != 3:
        raise RuntimeError("seed partitions must differ")
    mapping = {
        s: {
            r["context_sha256"]: next(
                k for k, g in enumerate(folds[s]) if r["family_id"] in g
            )
            for r in rows
        }
        for s in stat.SEEDS
    }

    # 3. Fits (15 units per seed x 3 members = 135; no SHUF), checkpointed per unit; DL-Val predicted by the outer units.
    outers, inners = stat.fit_plan()
    fits = {}
    load_wait = 0.0
    for seed in stat.FIT_SEEDS["BLEND"]:
        for ex in outers + inners:
            load_wait += wait_for_load()
            guards(root)
            receipts = verified_unit(root, "BLEND", seed, ex) if resume else None
            if receipts is None:
                receipts = fit_unit(
                    h124,
                    "BLEND",
                    seed,
                    ex,
                    rows,
                    mapping[seed],
                    context,
                    pools,
                    samples,
                    root,
                    val=(val_ctx, val_pools, val) if ex in outers else None,
                )
            fits[(seed, ex)] = receipts
    if len(fits) != 45 or sum(len(v) for v in fits.values()) != 45 * stat.MEMBERS:
        raise RuntimeError("45 units x 3 member receipts required")

    def preds(seed, ex):
        """Raw predictions of one unit: SINGLE = member 0 (H217's fit), BAG = the mean of the 3 members."""
        members = []
        for m in range(stat.MEMBERS):
            with np.load(
                root / (unit_name("BLEND", seed, ex, m) + ".npz"), allow_pickle=False
            ) as z:
                members.append({k: z[k].astype(np.float64) for k in z.files})
        return dict(
            SINGLE=members[0],
            BAG={k: np.mean([m[k] for m in members], axis=0) for k in members[0]},
        )

    def fitness_native(cost, pool):
        return h139.native(cost, pool["fitness"], pool["gate"])

    # 4. Stage 1: FIT-only calibration (per prior) and gate thresholds (per gated arm) per cell, and the B panels;
    # lock fsynced before any A scoring.
    family_of = {r["context_sha256"]: r["family_id"] for r in rows}
    blend_arm = {"SINGLE": "BLEND_025", "BAG": "BLEND_BAG3"}
    cells = []
    for seed in stat.SEEDS:
        for k in range(5):
            a, b, fit = stat.cell(k)
            inner = {kind: {} for kind in blend_arm}
            for j in fit:
                p = preds(seed, stat.inner_model_for(k, j))
                for kind in inner:
                    inner[kind].update(
                        {c: v for c, v in p[kind].items() if mapping[seed][c] == j}
                    )
            fit_ids = sorted(inner["SINGLE"])
            cal = {
                kind: list(
                    h124.calibration(
                        np.concatenate([inner[kind][c][4:] for c in fit_ids]),
                        np.concatenate([pools[c]["fitness"][4:] for c in fit_ids]),
                        np.concatenate([pools[c]["gate"][4:] for c in fit_ids]),
                    )
                )
                for kind in inner
            }
            # Gate thresholds from FIT contexts only: inner out-of-fold calibrated priors, FIT native utilities.
            sig = {arm: [] for arm in stat.GATES}
            fit_r882, fit_fams = [], []
            fit_blend = {kind: [] for kind in blend_arm}
            for c in fit_ids:
                costs, _, signal = arm_costs(
                    stat,
                    base[c],
                    {
                        kind: cal[kind][0] - cal[kind][1] * inner[kind][c]
                        for kind in inner
                    },
                    {arm: 0.0 for arm in stat.GATES},
                )
                for arm, name in stat.GATES.items():
                    sig[arm].append(signal[stat.PRIORS[arm]][name])
                fit_r882.append(fitness_native(base[c], pools[c]))
                for kind, arm in blend_arm.items():
                    fit_blend[kind].append(fitness_native(costs[arm], pools[c]))
                fit_fams.append(family_of[c])
            gate_choice = {
                arm: stat.choose_tau(
                    sig[arm], fit_r882, fit_blend[stat.PRIORS[arm]], fit_fams
                )
                for arm in stat.GATES
            }
            pair = tuple(sorted((a, b)))
            outer = preds(seed, pair)
            b_ids = [
                r["context_sha256"]
                for r in rows
                if mapping[seed][r["context_sha256"]] == b
            ]
            b_panel = {
                kind: hashlib.sha256(
                    np.array(
                        [
                            fitness_native(
                                stat.blend(
                                    base[c],
                                    cal[kind][0] - cal[kind][1] * outer[kind][c],
                                ),
                                pools[c],
                            )
                            for c in b_ids
                        ]
                    ).tobytes()
                ).hexdigest()
                for kind in blend_arm
            }
            cells.append(
                dict(
                    seed=seed,
                    outer=k,
                    A=a,
                    B=b,
                    FIT=list(fit),
                    calibration=cal,
                    gate_choice=gate_choice,
                    fit_contexts=len(fit_ids),
                    b_panel_blend_native_sha256=b_panel,
                    fit_prediction_sha256={
                        f"{''.join(map(str, ex))}-m{m}": fits[(seed, ex)][m][
                            "prediction_sha256"
                        ]
                        for ex in [pair] + [stat.inner_model_for(k, j) for j in fit]
                        for m in range(stat.MEMBERS)
                    },
                )
            )
    lock = root / ("stage1_lock-resume.json" if resume else "stage1_lock.json")
    save(
        lock,
        dict(written_before_any_A_or_DL_Val_scoring=True, cells=cells),
        exclusive=True,
    )
    lock_sha = sha(lock)
    print("LOCK", lock_sha[:12], flush=True)

    # 5. Stage 2: pooled A, every arm: flywheel native first-argmin (primary) and U4 (report-only).
    index = {r["context_sha256"]: i for i, r in enumerate(rows)}
    u4 = {a: np.full((3, len(rows)), np.nan) for a in stat.ARMS}
    native = {a: np.full((3, len(rows)), np.nan) for a in stat.ARMS}
    opened_a = {arm: np.zeros((3, len(rows)), bool) for arm in stat.GATES}
    switched_a = {kind: np.zeros((3, len(rows)), bool) for kind in blend_arm}
    val_cells = []
    for c in cells:
        seed, si = c["seed"], stat.SEEDS.index(c["seed"])
        ob = preds(seed, tuple(sorted((c["A"], c["B"]))))
        cal = c["calibration"]
        taus = {arm: c["gate_choice"][arm]["tau"] for arm in stat.GATES}
        for r in rows:
            cid = r["context_sha256"]
            if mapping[seed][cid] != c["A"]:
                continue
            costs, opened, signal = arm_costs(
                stat,
                base[cid],
                {kind: cal[kind][0] - cal[kind][1] * ob[kind][cid] for kind in ob},
                taus,
            )
            for arm in stat.GATES:
                opened_a[arm][si, index[cid]] = opened[arm]
            for kind in blend_arm:
                switched_a[kind][si, index[cid]] = signal[kind]["zgap"] != 0.0
            for arm, cost in costs.items():
                e = h124.endpoints(cost, pools[cid]["u"])
                nat = fitness_native(cost, pools[cid])
                if nat != e["legacy"]:
                    raise RuntimeError("flywheel native != first-argmin utility")
                u4[arm][si, index[cid]] = e["u4"]
                native[arm][si, index[cid]] = nat
        val_cells.append(
            (
                taus,
                {
                    kind: {
                        r["context_sha256"]: cal[kind][0]
                        - cal[kind][1] * ob[kind][r["context_sha256"]]
                        for r in val
                    }
                    for kind in ob
                },
            )
        )
    gate_rates = {
        arm: dict(
            open_rate_per_seed=opened_a[arm].mean(axis=1).tolist(),
            open_rate_among_switching_per_seed=[
                float(opened_a[arm][s][sw[s]].mean()) if sw[s].any() else 0.0
                for s in range(3)
            ],
        )
        for arm in stat.GATES
        for sw in [switched_a[stat.PRIORS[arm]]]
    }
    gate_rates["switching_rate_per_seed"] = {
        kind: v.mean(axis=1).tolist() for kind, v in switched_a.items()
    }

    def panel(values, arm):
        """Seed rows of an arm (all H223 arms have three seeds; R882 is seed-invariant)."""
        n = len(stat.ARM_SEEDS.get(arm, stat.SEEDS))
        return values[arm][:n]

    if any(not np.isfinite(panel(v, a)).all() for v in (u4, native) for a in stat.ARMS):
        raise RuntimeError("pooled A coverage incomplete")
    if not all(
        np.array_equal(v["R882"][s], v["R882"][0])
        for v in (u4, native)
        for s in range(3)
    ):
        raise RuntimeError("R882 panel must be seed-invariant")

    def versus(values, arm):
        p = panel(values, arm)
        return p, values["R882"][: len(p)]

    contrasts = {a: stat.contrast(*versus(native, a), fams) for a in stat.ARMS[1:]}
    u4_contrasts = {a: stat.contrast(*versus(u4, a), fams) for a in stat.ARMS[1:]}
    elo = {a: stat.elo_vs_reference(*versus(native, a), fams) for a in stat.ARMS[1:]}
    elo_u4 = {a: stat.elo_vs_reference(*versus(u4, a), fams) for a in stat.ARMS[1:]}
    minus2 = {
        a: [int(np.count_nonzero(row == -2.0)) for row in panel(native, a)]
        for a in stat.ARMS
    }

    # 6. DL-Val descriptive: each of the 15 cells applies its own calibrated outer prior and FIT-chosen taus to the 91
    # contexts; per-context utilities are averaged over the 15 cells.
    dl_val = {}
    per_arm = {arm: {"u4": [], "native": []} for arm in stat.ARMS}
    for r in val:
        cid = r["context_sha256"]
        acc = {arm: {"u4": [], "native": []} for arm in stat.ARMS}
        for taus, prior in val_cells:
            costs, _, _ = arm_costs(
                stat, val_costs[cid], {kind: v[cid] for kind, v in prior.items()}, taus
            )
            for arm, cost in costs.items():
                acc[arm]["u4"].append(h124.endpoints(cost, val_pools[cid]["u"])["u4"])
                acc[arm]["native"].append(
                    h139.native(cost, val_pools[cid]["fitness"], val_pools[cid]["gate"])
                )
        for arm in stat.ARMS:
            for key in ("u4", "native"):
                per_arm[arm][key].append(float(np.mean(acc[arm][key])))
    for arm in stat.ARMS:
        u4v, nat = per_arm[arm]["u4"], per_arm[arm]["native"]
        vf = [r["family_id"] for r in val]
        dl_val[arm] = dict(
            native_context_mean=float(
                np.mean(nat)
            ),  # the flywheel91 statistic (R882 = 1.1377941898164525; H135)
            native_equal_family=float(stat.family_matrix(np.array(nat), vf)[1].mean()),
            u4_equal_family=float(stat.family_matrix(np.array(u4v), vf)[1].mean()),
            native_minus2_in_every_cell=int(sum(x == -2.0 for x in nat)),
        )
    # R882's cell-averaged values must reproduce the exact regression; the plateau reference is the regression itself.
    if not np.allclose(per_arm["R882"]["native"], got, rtol=0, atol=1e-12):
        raise RuntimeError(
            "DL-Val R882 per-context utilities differ from the regression"
        )
    vf = [r["family_id"] for r in val]
    plateau = {
        arm: stat.plateau(per_arm[arm]["native"], got, vf) for arm in stat.ARMS[1:]
    }
    if plateau[stat.PRIMARY]["r882_context_mean"] != stat.R882_DL_VAL:
        raise RuntimeError("DL-Val r882 context mean differs from the pinned contract")
    decision = stat.verdict(contrasts[stat.PRIMARY], plateau[stat.PRIMARY])
    # Report-only: the bagged gate against the frozen GATED_025 reference, paired on the same contexts and cells.
    paired = dict(
        native=stat.contrast(
            panel(native, stat.PRIMARY), panel(native, stat.REFERENCE), fams
        ),
        elo_native=stat.elo_vs_reference(
            panel(native, stat.PRIMARY), panel(native, stat.REFERENCE), fams
        ),
        dl_val_plateau=stat.plateau(
            per_arm[stat.PRIMARY]["native"], per_arm[stat.REFERENCE]["native"], vf
        ),
    )
    import sklearn

    metrics = dict(
        schema="jope.h223.bagged-blend-gate/v1",
        status="COMPLETE",
        verdict=decision["verdict"],
        decision=decision,
        primary_endpoint="flywheel native fixed-dev-cost-argmin-negative-fitness/v1 (full-pool first argmin; -fitness valid, -2.0 invalid)",
        contrasts_native=contrasts,
        contrasts_u4_report_only=u4_contrasts,
        elo_native_vs_r882=elo,
        elo_u4_vs_r882_report_only=elo_u4,
        minus2_native_per_seed=minus2,
        dl_val_descriptive=dl_val,
        dl_val_plateau_co_primary=plateau[stat.PRIMARY],
        dl_val_plateau_report_only={a: plateau[a] for a in stat.ARMS[2:]},
        paired_primary_vs_reference_report_only=paired,
        gate_rates_pooled_a=gate_rates,
        gate_choice_by_cell={
            f"{c['seed']}/{c['outer']}": {
                arm: {k: v for k, v in g.items() if k != "grid"}
                for arm, g in c["gate_choice"].items()
            }
            for c in cells
        },
        fold_jaccard=stat.fold_jaccard(folds),
        calibration_by_cell={
            f"{c['seed']}/{c['outer']}": c["calibration"] for c in cells
        },
        stage1_lock_sha256=lock_sha,
        fit_units=len(fits),
        fits=sum(len(v) for v in fits.values()),
        fit_wall_seconds=sum(v[0]["unit_wall_seconds"] for v in fits.values()),
        resumed=resume,
        dope_before=dope["before"],
        dope_after=dope["after"],
        bound_sha256=bound,
        read_boundary=dict(
            train_contexts=len(rows),
            dl_val_contexts_for_regression_and_descriptive=len(val),
            sealed=0,
            holdout=0,
            meta_test=0,
            live_loop=0,
            train_cache_dl_val_rows_decoded=sum(
                v["overlapping_reads"] for v in reads.values()
            ),
        ),
        runtime=dict(
            wall_seconds=time.monotonic() - t0,
            process_cpu_seconds=time.process_time() - cpu0,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            scratch_bytes=tree_bytes(root),
            effective_cores=CORES,
            deadline_epoch=deadline(),
            runner_load_wait_seconds=load_wait,
        ),
        versions=dict(
            numpy=np.__version__,
            scikit_learn=sklearn.__version__,
            torch=torch.__version__,
            python=platform.python_version(),
        ),
        optimizer_updates=0,
        GPU_used=False,
        CUDA_initialized=torch.cuda.is_initialized(),
        serving_promotion=False,
    )
    raw = (
        json.dumps(jsonable(metrics), indent=1, sort_keys=True, allow_nan=False) + "\n"
    )
    if len(raw.encode()) >= METRICS_LIMIT:
        raise RuntimeError("metrics would exceed 100 KB")
    np.savez_compressed(
        root / "panels.npz",
        **{f"u4_{a}": v for a, v in u4.items()},
        **{f"native_{a}": v for a, v in native.items()},
        context_ids=np.array([r["context_sha256"] for r in rows]),
        family_ids=np.array(fams),
    )
    (root / "metrics.json").write_text(raw)
    print("COMPLETE", metrics["verdict"], flush=True)
    return metrics


EARLIER_WAVES = {
    "H203": HERE.parent / "h203_blend025_gated/fold_manifest.json",
    "H210": HERE.parent / "h210_blend025_gated_retry/fold_manifest.json",
    "H217": HERE.parent / "h217_gated025_confirm/fold_manifest.json",
    "H218": HERE.parent / "h218_gated_lambda_nest/fold_manifest.json",
}


def write_fold_manifest():
    """Outcome-free: TRAIN context/family identities from the pinned selected.json only, plus the overlap with the
    registered H203/H210/H217/H218 partitions."""
    h124 = load(H139_PATH, "h223_h139", H139_SHA).load_h124()
    selected = h124.INPUT / "selected.json"
    if sha(selected) != h124.FILES[selected]:
        raise RuntimeError("selected packet changed")
    rows = json.loads(selected.read_text())["train"]
    (HERE / "fold_manifest.json").write_text(
        json.dumps(
            stat.fold_manifest(
                rows,
                {
                    w: json.loads(p.read_text())["families_by_seed"]
                    for w, p in EARLIER_WAVES.items()
                },
            ),
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--launch", type=Path)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="the single preregistered resume after a guard abort",
    )
    ap.add_argument("--write-fold-manifest", action="store_true")
    ap.add_argument(
        "--check-launch",
        action="store_true",
        help="host and source binding only (no read, root or claim); the monitor runs it before claiming",
    )
    a = ap.parse_args()
    if a.write_fold_manifest:
        write_fold_manifest()
    elif a.check_launch:
        if a.launch is None:
            ap.error("--launch is required")
        admit_host()
        bind_source(json.loads(a.launch.read_text()))
        print("SOURCE OK", flush=True)
    elif a.root is None or a.launch is None:
        ap.error("--root and --launch are required")
    else:
        run(a.root, a.launch, resume=a.resume)


if __name__ == "__main__":
    main()

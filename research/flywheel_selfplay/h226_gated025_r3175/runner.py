"""One monitored H226 CPU attempt: fresh-fold confirmation of frozen H210/H217 GATED_025 on the retained-best r3175
base (G3175: BLEND_025 on r3175 gated by outcome-free zgap(r3175, blend_3175) with FIT-selected threshold over R3175,
else r3175) vs R882 and R3175 on the flywheel native first-argmin protocol, 205-family pooled cross-fit (fresh seeds
226101-226103, fresh fold seeds 226141-226143), with G882 as the replication anchor, G3175_DIS as the H129 disagreement
placebo on r3175, and the DL-Val flywheel91 plateau (requiring both vs-R882 gain > +0.002 & LCB95 > 0 AND vs-R3175
LCB95 > 0) as co-primary.

Reuses, byte-pinned and unchanged: the merged H124 module (load_data, train_context/projection, baseline forward,
calibration, endpoints, program; via the merged H139 loader) and the merged H139 roster/pools/native helpers (the H152
regression path; `native` is the flywheel protocol fixed-dev-cost-argmin-negative-fitness/v1). `fit_gbm` is H217's
`fit_gbm` (H124 `fit_one` with the H195 exclusion sets). The gate and its FIT-only tau procedure are H210/H217's,
frozen. Operations (hw_t1435): xbabe2 CPU only, nice 19, whole-tree affinity <= 4 cores, RSS < 4 GiB, MemAvailable >=
20 GiB, disk free >= 300 GB, load5 > 60 pauses (recheck every 60 s, resume at <= 55), compute stop 18:30Z.
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
DATA = Path.home() / "jope-data/h226"
INPUTS = DATA / "inputs"
CKPT_DIR = DATA / "ckpt"
CHECKPOINTS = {
    "R882": (
        CKPT_DIR / "round-882.pt",
        "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64",
    ),
    "R3175": (
        CKPT_DIR / "round-3175.pt",
        "1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b",
    ),
}
PREREG_SHA = "fc89fa3264c66baf41e8ac07bbcd9e3d349c6eded238a76068a734de341213da"
H139_PATH = REPO / "research/flywheel_selfplay/h139_selection_audit/runner.py"
H139_SHA = "7a37821db82c89ddf8e48266383c681b20535ff31b95504e259c74dbe5a9c686"
BOUND_FILES = (
    "research/flywheel_selfplay/h226_gated025_r3175/PREREG.md",
    "research/flywheel_selfplay/h226_gated025_r3175/runner.py",
    "research/flywheel_selfplay/h226_gated025_r3175/analysis.py",
    "research/flywheel_selfplay/h226_gated025_r3175/monitor.py",
    "research/flywheel_selfplay/h139_selection_audit/runner.py",
    "research/flywheel_selfplay/h124_gbm_shrink_blend/runner.py",
    "python/src/jope/north_star/dope_freeze_guard.py",
)
HOST_RULES = {
    "xbabe2": dict(min_free=300_000_000_000, max_used=None),  # hw_t1435: free >= 300 GB
}
MEM_MIN = 20 * 2**30  # hw_t1435: xbabe2 swap is full; stop below 20 GiB available
LOAD5_MAX = 60.0  # hw_t1435: pause before a fit while load5 > 60
LOAD5_RESUME = 55.0  # ... until load5 <= 55 (rechecked every 60 s)
LOAD_POLL = 60.0
CORES = len(
    os.sched_getaffinity(0)
)  # effective allocation at import time (pinned to <= 4 by monitor/runner)
WORKER_CORES = 4
DEADLINE_ENV = "H226_DEADLINE_EPOCH"  # absolute deadline from the attempt claim, set by the monitor
SCRATCH_MAX = 600_000_000
RSS_LIMIT = 4 * 2**30  # hw_t1435: process-tree RSS < 4 GiB
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


def arm_costs(stat_mod, base_costs, gbm_cost, taus):
    """Every H226 arm's cost for one context and the gate decisions (outcome-free: costs only).
    `base_costs` maps R882 and R3175 to their cost vectors; each base gets H210/H217's blend and signals."""
    blended = {b: stat_mod.blend(base_costs[b], gbm_cost) for b in stat_mod.BASES}
    signal = {
        b: dict(
            zgap=stat_mod.zgap(base_costs[b], blended[b]),
            disagreement=stat_mod.disagreement(gbm_cost, base_costs[b]),
        )
        for b in stat_mod.BASES
    }
    opened = {
        arm: stat_mod.gated(signal[stat_mod.BASE_OF[arm]][name], taus[arm])
        for arm, name in stat_mod.GATES.items()
    }
    costs = {
        "R882": base_costs["R882"],
        "R3175": base_costs["R3175"],
        "B3175": blended["R3175"],
        "B882": blended["R882"],
    }
    costs.update(
        {
            arm: blended[stat_mod.BASE_OF[arm]]
            if opened[arm]
            else base_costs[stat_mod.BASE_OF[arm]]
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
    """Map a pinned ~/jope-data input path into the H226 staging layout (same relative path under DATA/inputs)."""
    base = Path.home() / "jope-data"
    return (
        INPUTS / Path(path).relative_to(base)
        if Path(path).is_relative_to(base)
        else Path(path)
    )


def load_modules():
    """Pinned H139/H124 bytes, pointed at the H226 staging layout (~/jope-data/h226/inputs on xbabe2, the only admitted host).
    Every staged file keeps its pinned SHA-256 (verified by the inherited loaders); only the directory changes."""
    h139 = load(H139_PATH, "h226_h139", H139_SHA)
    h124 = h139.load_h124()
    h124.RSS_LIMIT_KIB = RSS_LIMIT // 1024
    h124.INPUT, h124.CACHE = (staged(p) for p in (h124.INPUT, h124.CACHE))
    h124.R882 = CHECKPOINTS["R882"][0]
    h124.FILES = {
        (CHECKPOINTS["R882"][0] if Path(p).name == "round-882.pt" else staged(p)): d
        for p, d in h124.FILES.items()
    }
    h139.INPUT = h124.INPUT
    h139.ARM_DIR = INPUTS  # pinned best_dev_checkpoint.json copy (H139 REFERENCE sha)
    return h139, h124


def structural_check(h124):
    """Verify both R882 and R3175 checkpoints: pinned SHA-256, DOPE identity field, and non-adapter equality."""
    import torch

    states = {}
    for name, (path, digest) in CHECKPOINTS.items():
        if sha(path) != digest:
            raise RuntimeError(f"{name} checkpoint SHA-256 mismatch: {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("dope_sha256") != h124.DOPE:
            raise RuntimeError(f"{name} DOPE identity mismatch")
        states[name] = ck["state_dict"]
    ref, cand = states["R882"], states["R3175"]
    if set(ref) != set(cand):
        raise RuntimeError("R3175 parameter set differs from R882")
    nonadapter = [
        k
        for k in cand
        if not k.startswith("adapter_10tok.") and not torch.equal(cand[k], ref[k])
    ]
    if nonadapter:
        raise RuntimeError("R3175 non-adapter tensors differ from R882")
    diff_elems = sum(
        int((cand[k] != ref[k]).sum()) for k in cand if k.startswith("adapter_10tok.")
    )
    if diff_elems <= 0:
        raise RuntimeError("R3175 adapter tensors must differ from R882")
    return dict(
        r882_sha256=CHECKPOINTS["R882"][1],
        r3175_sha256=CHECKPOINTS["R3175"][1],
        dope_sha256=h124.DOPE,
        nonadapter_tensors_equal=True,
        r3175_adapter_elements_differing_from_r882=diff_elems,
    )


def tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def admit_host():
    """Preregistered host boundary: xbabe2 only, at nice 19. Called before any read, write or launch."""
    host = socket.gethostname().split(".")[0]
    if host not in HOST_RULES or os.getpriority(os.PRIO_PROCESS, 0) < 19:
        raise RuntimeError(f"H226 runs on xbabe2 at nice 19 only (host {host!r})")
    return host


def deadline():
    """The absolute deadline (epoch seconds) the monitor derived from the attempt claim; None outside a monitored run."""
    value = os.environ.get(DEADLINE_ENV)
    return None if value is None else float(value)


def guards(root):
    """xbabe2 disk rule (hw_t1435: free >= 300 GB), MemAvailable >= 20 GiB, nice 19, own root < 600 MB, and the
    absolute deadline: hard aborts, checked before every fit and stage. Host load is a pause (see wait_for_load)."""
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
        raise RuntimeError(f"own output root {scratch} B >= 600 MB")
    return dict(
        host=host,
        effective_cores=CORES,
        worker_affinity_cores=len(os.sched_getaffinity(0)),
        deadline_epoch=end,
        free=usage.free,
        used=round(used, 5),
        mem_available=mem,
        load1=load1,
        load5=load5,
        scratch=scratch,
    )


def wait_for_load():
    """hw_t1435: before a fit/stage, if load5 > 60, wait and recheck every 60 s until load5 <= 55. The absolute
    deadline still ends the attempt while it waits. Returns the seconds waited."""
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
    """The pinned H124 baseline hardcodes torch.set_num_threads(4); hold it at 1 (as H152/H217)."""
    from unittest.mock import patch

    import torch

    setter = torch.set_num_threads

    def capped(requested):
        setter(1)

    setter(1)
    with patch.object(torch, "set_num_threads", capped):
        return fn(*args, **kwargs)


def fit_gbm(
    h124, tag, seed, excluded, rows, mapping, context, pools, samples, root, val=None
):
    """H124 fit_one, H195/H217/H226 rules: exclusion size 2 (outer {A,B}) or 3 (inner {A,B,j}); SHUF permutes the
    training targets within family. Predictions for the excluded folds' contexts (and DL-Val contexts when given)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    if (
        seed not in stat.SEEDS
        or tag not in ("BLEND", "SHUF")
        or len(excluded) not in (2, 3)
        or tuple(sorted(set(excluded))) != excluded
    ):
        raise ValueError("unregistered fit")
    name = f"{tag}-seed{seed}-exclude{''.join(map(str, excluded))}"
    out = root / (name + ".npz")
    if out.exists():
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
    if tag == "SHUF":
        y = stat.shuffle_within_family(
            y, fams, stat.stream_seed(stat.SHUF_SEED, seed, *excluded)
        )
    h124.check_rss()
    model = HistGradientBoostingRegressor(**h124.MODEL, random_state=seed).fit(x, y)
    del x, y
    h124.release_memory()
    predictions = {}
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
        predictions[cid] = model.predict(
            np.concatenate([xp, np.repeat(cf[None], len(xp), axis=0)], axis=1)
        ).astype(np.float32)
    np.savez_compressed(out, **predictions)
    receipt = dict(
        tag=tag,
        seed=seed,
        excluded_folds=list(excluded),
        fit_contexts=len(fit_rows),
        fit_rows=len(fit_rows) * 192,
        iterations=int(model.n_iter_),
        prediction_sha256=sha(out),
        predicted_contexts=len(predictions),
        wall_seconds=time.monotonic() - t0,
        shuffled=tag == "SHUF",
    )
    save(out.with_suffix(".json"), receipt)
    print("FIT", name, round(receipt["wall_seconds"], 1), flush=True)
    return receipt


def verified_fit(root, tag, seed, excluded):
    """Single-resume rule: an existing fit is reused only if its receipt and npz hash agree."""
    name = f"{tag}-seed{seed}-exclude{''.join(map(str, excluded))}"
    npz, js = root / (name + ".npz"), root / (name + ".json")
    if npz.exists() and js.exists():
        receipt = json.loads(js.read_text())
        if receipt["prediction_sha256"] == sha(npz):
            return receipt
    for p in (npz, js):
        if p.exists():
            p.rename(
                p.with_name(p.name + ".incomplete")
            )  # kept for custody, never deleted
    return None


def eval_base_checkpoint(
    h124,
    base_name,
    root,
    val,
    val_ctx,
    val_pools,
    val_proj,
    rows,
    context,
    pools,
    projection,
    resume=False,
):
    """Evaluate a frozen base checkpoint (R882 or R3175) on DL-Val91 and 1,219 DL-Train contexts."""
    ckpt_path, ckpt_sha = CHECKPOINTS[base_name]
    if sha(ckpt_path) != ckpt_sha:
        raise RuntimeError(f"{base_name} checkpoint hash mismatch before forward")
    h124.R882 = ckpt_path
    h124.FILES[ckpt_path] = ckpt_sha
    vroot = root / (
        f"regression-{base_name.lower()}-resume"
        if resume
        else f"regression-{base_name.lower()}"
    )
    vroot.mkdir()
    val_costs, net = bounded_threads(
        h124.baseline, vroot, val, val_ctx, val_pools, val_proj
    )
    (vroot / "r882-copy.pt").unlink()
    troot = root / (
        f"train-{base_name.lower()}-resume" if resume else f"train-{base_name.lower()}"
    )
    troot.mkdir()
    train_costs, net = bounded_threads(
        h124.baseline, troot, rows, context, pools, projection, net=net
    )
    del net
    h124.release_memory()
    dope_val = json.loads((vroot / "dope-secondary.json").read_text())
    dope_train = json.loads((troot / "dope-train.json").read_text())
    if (
        dope_val["before"] != h124.DOPE
        or dope_val["after"] != h124.DOPE
        or dope_train["before"] != h124.DOPE
        or dope_train["after"] != h124.DOPE
        or sha(ckpt_path) != ckpt_sha
    ):
        raise RuntimeError(f"{base_name} DOPE or checkpoint hash changed")
    return val_costs, train_costs, dict(val=dope_val, train=dope_train)


def run(root, launch_path, resume=False, smoke=False):
    admit_host()  # before any import, read, root creation or claim
    if deadline() is None:
        raise RuntimeError(
            "H226 runs only under its monitor (absolute deadline missing)"
        )
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) > WORKER_CORES:
        os.sched_setaffinity(0, allowed[:WORKER_CORES])
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    root = Path(root)
    valid_prefix = "smoke-" if smoke else "run-"
    if root.parent != DATA or not root.name.startswith(valid_prefix):
        raise RuntimeError(
            f"H226 requires a {valid_prefix}* root under ~/jope-data/h226"
        )
    launch = json.loads(Path(launch_path).read_text())
    bound = bind_source(launch)
    if resume:
        if (
            smoke
            or not root.exists()
            or (root / "resume-identity.json").exists()
            or (root / "metrics.json").exists()
        ):
            raise RuntimeError(
                "single resume only, of an existing incomplete run-* root"
            )
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
            smoke=smoke,
            worker_affinity=sorted(os.sched_getaffinity(0)),
            environment={
                k: os.environ.get(k)
                for k in (
                    "OMP_NUM_THREADS",
                    "OMP_THREAD_LIMIT",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "CUDA_VISIBLE_DEVICES",
                )
            },
        ),
    )

    # 1. Structural & exact DL-Val91 regression gates first (R882 and R3175), plus TRAIN base costs.
    structure = structural_check(h124)
    train_rows, val, reference = h139.roster(h124)
    val_ctx, _ = h124.train_context(val + train_rows, [])
    val_ctx = {r["context_sha256"]: val_ctx[r["context_sha256"]] for r in val}
    val_proj = h124.train_projection(val + train_rows, [])
    val_pools = h139.pools_for(val)

    h124.FOLD_SEED, h124.SAMPLE_SEED = stat.FOLD_SEEDS[stat.SEEDS[0]], stat.SAMPLE_SEED
    droot = root / ("data-resume" if resume else "data")
    droot.mkdir()
    selected, context, pools, samples, _, reads = h124.load_data(droot)
    rows = selected["train"]
    if [r["context_sha256"] for r in rows] != [r["context_sha256"] for r in train_rows]:
        raise RuntimeError("roster mismatch")
    projection = h124.train_projection(rows, selected["dev"])

    val_costs_882, base_882, dope_882 = eval_base_checkpoint(
        h124,
        "R882",
        root,
        val,
        val_ctx,
        val_pools,
        val_proj,
        rows,
        context,
        pools,
        projection,
        resume=resume,
    )
    got_882 = np.array(
        [
            h139.native(
                val_costs_882[r["context_sha256"]],
                val_pools[r["context_sha256"]]["fitness"],
                val_pools[r["context_sha256"]]["gate"],
            )
            for r in val
        ]
    )
    if (
        not np.array_equal(got_882, np.asarray(reference["utility"]["per_context"]))
        or float(got_882.mean()) != reference["best_utility"]
        or float(got_882.mean()) != stat.R882_DL_VAL
    ):
        raise RuntimeError("R882 91/91 native regression failed")
    print("BASELINE PASS: 91/91 native r882 exact", float(got_882.mean()), flush=True)

    val_costs_3175, base_3175, dope_3175 = eval_base_checkpoint(
        h124,
        "R3175",
        root,
        val,
        val_ctx,
        val_pools,
        val_proj,
        rows,
        context,
        pools,
        projection,
        resume=resume,
    )
    got_3175 = np.array(
        [
            h139.native(
                val_costs_3175[r["context_sha256"]],
                val_pools[r["context_sha256"]]["fitness"],
                val_pools[r["context_sha256"]]["gate"],
            )
            for r in val
        ]
    )
    if not math.isclose(
        float(got_3175.mean()), stat.R3175_DL_VAL, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(
            f"R3175 91/91 native regression failed: got {float(got_3175.mean())!r}, expected {stat.R3175_DL_VAL!r}"
        )
    print("BASELINE PASS: 91/91 native r3175 exact", float(got_3175.mean()), flush=True)

    base_train = {"R882": base_882, "R3175": base_3175}
    base_val = {"R882": val_costs_882, "R3175": val_costs_3175}
    got_val = {"R882": got_882, "R3175": got_3175}

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

    outers, inners = stat.fit_plan()
    if smoke:
        wait_for_load()
        guards(root)
        smoke_receipt = fit_gbm(
            h124,
            "BLEND",
            stat.SEEDS[0],
            outers[0],
            rows,
            mapping[stat.SEEDS[0]],
            context,
            pools,
            samples,
            root,
            val=(val_ctx, val_pools, val),
        )
        smoke_out = dict(
            schema="jope.h226.smoke/v1",
            status="SMOKE_PASS",
            r882_dl_val_mean=float(got_882.mean()),
            r3175_dl_val_mean=float(got_3175.mean()),
            structure=structure,
            dope_882=dope_882,
            dope_3175=dope_3175,
            smoke_fit=smoke_receipt,
            runtime=dict(
                wall_seconds=time.monotonic() - t0,
                process_cpu_seconds=time.process_time() - cpu0,
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scratch_bytes=tree_bytes(root),
                worker_affinity=sorted(os.sched_getaffinity(0)),
            ),
        )
        save(root / "smoke-metrics.json", smoke_out, exclusive=True)
        print("SMOKE PASS", flush=True)
        return smoke_out

    # 3. Fits (15 BLEND per seed, 45 in total; no SHUF), checkpointed per fit; DL-Val predicted by the outer models.
    fits = {}
    load_wait = 0.0
    for tag in ("BLEND", "SHUF"):
        for seed in stat.FIT_SEEDS[tag]:
            for ex in outers + inners:
                load_wait += wait_for_load()
                guards(root)
                receipt = verified_fit(root, tag, seed, ex) if resume else None
                if receipt is None:
                    receipt = fit_gbm(
                        h124,
                        tag,
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
                fits[(tag, seed, ex)] = receipt
    if len(fits) != 45:
        raise RuntimeError("45 fit receipts required")

    def preds(tag, seed, ex):
        with np.load(
            root / f"{tag}-seed{seed}-exclude{''.join(map(str, ex))}.npz",
            allow_pickle=False,
        ) as z:
            return {k: z[k].astype(np.float64) for k in z.files}

    # 4. Stage 1: FIT-only calibration and gate thresholds per cell (chosen on each arm's own base), and B panels;
    # lock fsynced before any A or DL-Val scoring.
    family_of = {r["context_sha256"]: r["family_id"] for r in rows}
    cells = []
    for seed in stat.SEEDS:
        for k in range(5):
            a, b, fit = stat.cell(k)
            cal, inner_by_tag = {}, {}
            tags = [t for t in ("BLEND", "SHUF") if seed in stat.FIT_SEEDS[t]]
            for tag in tags:
                inner = {}
                for j in fit:
                    p = preds(tag, seed, stat.inner_model_for(k, j))
                    inner.update({c: p[c] for c in p if mapping[seed][c] == j})
                order = sorted(inner)
                cal[tag] = list(
                    h124.calibration(
                        np.concatenate([inner[c][4:] for c in order]),
                        np.concatenate([pools[c]["fitness"][4:] for c in order]),
                        np.concatenate([pools[c]["gate"][4:] for c in order]),
                    )
                )
                inner_by_tag[tag] = inner
            # Gate thresholds from FIT contexts only: inner out-of-fold calibrated priors, FIT native utilities.
            fa, fb_ = cal["BLEND"]
            fit_inner = inner_by_tag["BLEND"]
            fit_ids = sorted(fit_inner)
            sig = {arm: [] for arm in stat.GATES}
            fit_base = {bname: [] for bname in stat.BASES}
            fit_blend = {bname: [] for bname in stat.BASES}
            fit_fams = []
            for c in fit_ids:
                costs, _, signal = arm_costs(
                    stat,
                    {bname: base_train[bname][c] for bname in stat.BASES},
                    fa - fb_ * fit_inner[c],
                    {arm: 0.0 for arm in stat.GATES},
                )
                for arm, sname in stat.GATES.items():
                    sig[arm].append(signal[stat.BASE_OF[arm]][sname])
                for bname in stat.BASES:
                    fit_base[bname].append(
                        h139.native(
                            base_train[bname][c], pools[c]["fitness"], pools[c]["gate"]
                        )
                    )
                    fit_blend[bname].append(
                        h139.native(
                            costs[stat.BLEND_OF[bname]],
                            pools[c]["fitness"],
                            pools[c]["gate"],
                        )
                    )
                fit_fams.append(family_of[c])
            gate_choice = {
                arm: stat.choose_tau(
                    sig[arm],
                    fit_base[stat.BASE_OF[arm]],
                    fit_blend[stat.BASE_OF[arm]],
                    fit_fams,
                )
                for arm in stat.GATES
            }
            outer = preds("BLEND", seed, tuple(sorted((a, b))))
            b_rows = [r for r in rows if mapping[seed][r["context_sha256"]] == b]
            b_panel = {
                bname: hashlib.sha256(
                    np.array(
                        [
                            h139.native(
                                stat.blend(
                                    base_train[bname][r["context_sha256"]],
                                    cal["BLEND"][0]
                                    - cal["BLEND"][1] * outer[r["context_sha256"]],
                                ),
                                pools[r["context_sha256"]]["fitness"],
                                pools[r["context_sha256"]]["gate"],
                            )
                            for r in b_rows
                        ]
                    ).tobytes()
                ).hexdigest()
                for bname in stat.BASES
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
                    b_panel_blend025_native_sha256=b_panel,
                    fit_prediction_sha256={
                        f"{t}-{''.join(map(str, ex))}": fits[(t, seed, ex)][
                            "prediction_sha256"
                        ]
                        for t in tags
                        for ex in [tuple(sorted((a, b)))]
                        + [stat.inner_model_for(k, j) for j in fit]
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
    switched_a = {bname: np.zeros((3, len(rows)), bool) for bname in stat.BASES}
    val_cells = []
    for c in cells:
        seed, si = c["seed"], stat.SEEDS.index(c["seed"])
        pair = tuple(sorted((c["A"], c["B"])))
        ob = preds("BLEND", seed, pair)
        ab, bb = c["calibration"]["BLEND"]
        taus = {arm: c["gate_choice"][arm]["tau"] for arm in stat.GATES}
        for r in rows:
            cid = r["context_sha256"]
            if mapping[seed][cid] != c["A"]:
                continue
            costs, opened, signal = arm_costs(
                stat,
                {bname: base_train[bname][cid] for bname in stat.BASES},
                ab - bb * ob[cid],
                taus,
            )
            for arm in stat.GATES:
                opened_a[arm][si, index[cid]] = opened[arm]
            for bname in stat.BASES:
                switched_a[bname][si, index[cid]] = signal[bname]["zgap"] != 0.0
            for arm, cost in costs.items():
                e = h124.endpoints(cost, pools[cid]["u"])
                nat = h139.native(cost, pools[cid]["fitness"], pools[cid]["gate"])
                if nat != e["legacy"]:
                    raise RuntimeError("flywheel native != first-argmin utility")
                u4[arm][si, index[cid]] = e["u4"]
                native[arm][si, index[cid]] = nat
        val_cells.append(
            (
                taus,
                {r["context_sha256"]: ab - bb * ob[r["context_sha256"]] for r in val},
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
        for sw in [switched_a[stat.BASE_OF[arm]]]
    }
    gate_rates["switching_rate_per_seed"] = {
        bname: v.mean(axis=1).tolist() for bname, v in switched_a.items()
    }

    def panel(values, arm):
        """Seed rows of an arm (all H226 arms have three seeds; R882 and R3175 are seed-invariant)."""
        n = len(stat.ARM_SEEDS.get(arm, stat.SEEDS))
        return values[arm][:n]

    if any(not np.isfinite(panel(v, a)).all() for v in (u4, native) for a in stat.ARMS):
        raise RuntimeError("pooled A coverage incomplete")
    for bname in stat.BASES:
        if not all(
            np.array_equal(v[bname][s], v[bname][0])
            for v in (u4, native)
            for s in range(3)
        ):
            raise RuntimeError(f"{bname} panel must be seed-invariant")

    def versus(values, arm, ref="R882"):
        p = panel(values, arm)
        return p, values[ref][: len(p)]

    contrasts = {
        a: stat.contrast(*versus(native, a, "R882"), fams) for a in stat.ARMS[1:]
    }
    contrasts_vs_r3175 = {
        a: stat.contrast(*versus(native, a, "R3175"), fams)
        for a in ("G3175", "G3175_DIS", "B3175", "G882", "R882")
    }
    paired_g3175_vs_g882 = dict(
        native=stat.contrast(*versus(native, "G3175", "G882"), fams),
        u4=stat.contrast(*versus(u4, "G3175", "G882"), fams),
        elo_native=stat.elo_vs_reference(*versus(native, "G3175", "G882"), fams),
    )
    u4_contrasts = {
        a: stat.contrast(*versus(u4, a, "R882"), fams) for a in stat.ARMS[1:]
    }
    u4_contrasts_vs_r3175 = {
        a: stat.contrast(*versus(u4, a, "R3175"), fams)
        for a in ("G3175", "G3175_DIS", "B3175", "G882", "R882")
    }
    elo = {
        a: stat.elo_vs_reference(*versus(native, a, "R882"), fams)
        for a in stat.ARMS[1:]
    }
    elo_vs_r3175 = {
        a: stat.elo_vs_reference(*versus(native, a, "R3175"), fams)
        for a in ("G3175", "G3175_DIS", "B3175", "G882", "R882")
    }
    elo_u4 = {
        a: stat.elo_vs_reference(*versus(u4, a, "R882"), fams) for a in stat.ARMS[1:]
    }
    minus2 = {
        a: [int(np.count_nonzero(row == -2.0)) for row in panel(native, a)]
        for a in stat.ARMS
    }

    # 6. DL-Val descriptive & co-primary: each of the 15 cells applies its own calibrated outer prior and FIT-chosen
    # taus to the 91 contexts; per-context utilities are averaged over the 15 cells, and top-1 picks are recorded.
    dl_val = {}
    per_arm = {arm: {"u4": [], "native": []} for arm in stat.ARMS}
    val_picks = {
        arm: np.zeros((len(val_cells), len(val)), dtype=np.int64) for arm in stat.ARMS
    }
    for ci_idx, r in enumerate(val):
        cid = r["context_sha256"]
        acc = {arm: {"u4": [], "native": []} for arm in stat.ARMS}
        for cell_idx, (taus, prior) in enumerate(val_cells):
            costs, _, _ = arm_costs(
                stat,
                {bname: base_val[bname][cid] for bname in stat.BASES},
                prior[cid],
                taus,
            )
            for arm, cost in costs.items():
                val_picks[arm][cell_idx, ci_idx] = int(np.argmin(cost))
                acc[arm]["u4"].append(h124.endpoints(cost, val_pools[cid]["u"])["u4"])
                acc[arm]["native"].append(
                    h139.native(cost, val_pools[cid]["fitness"], val_pools[cid]["gate"])
                )
        for arm in stat.ARMS:
            for key in ("u4", "native"):
                per_arm[arm][key].append(float(np.mean(acc[arm][key])))
    vf = [r["family_id"] for r in val]
    for arm in stat.ARMS:
        u4v, nat = per_arm[arm]["u4"], per_arm[arm]["native"]
        dl_val[arm] = dict(
            native_context_mean=float(np.mean(nat)),
            native_equal_family=float(stat.family_matrix(np.array(nat), vf)[1].mean()),
            u4_equal_family=float(stat.family_matrix(np.array(u4v), vf)[1].mean()),
            native_minus2_in_every_cell=int(sum(x == -2.0 for x in nat)),
        )
    for bname in stat.BASES:
        if not np.allclose(
            per_arm[bname]["native"], got_val[bname], rtol=0, atol=1e-12
        ):
            raise RuntimeError(
                f"DL-Val {bname} per-context utilities differ from the regression"
            )
    plateau_vs_r882 = {
        arm: stat.plateau(per_arm[arm]["native"], got_882, vf) for arm in stat.ARMS[1:]
    }
    plateau_vs_r3175 = {
        arm: stat.plateau(per_arm[arm]["native"], got_3175, vf)
        for arm in ("G3175", "G3175_DIS", "B3175", "G882", "R882")
    }
    if plateau_vs_r882[stat.PRIMARY]["r882_context_mean"] != stat.R882_DL_VAL:
        raise RuntimeError("DL-Val r882 context mean differs from the pinned contract")
    if not math.isclose(
        plateau_vs_r3175[stat.PRIMARY]["ref_context_mean"],
        stat.R3175_DL_VAL,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("DL-Val r3175 context mean differs from the pinned contract")

    dl_val_top1_flips = dict(
        G3175_vs_R3175=stat.top1_flips(val_picks["G3175"], val_picks["R3175"]),
        G3175_vs_R882=stat.top1_flips(val_picks["G3175"], val_picks["R882"]),
        R3175_vs_R882=stat.top1_flips(val_picks["R3175"], val_picks["R882"]),
        G882_vs_R882=stat.top1_flips(val_picks["G882"], val_picks["R882"]),
        G3175_vs_G882=stat.top1_flips(val_picks["G3175"], val_picks["G882"]),
    )
    decision = stat.verdict(
        contrasts[stat.PRIMARY],
        plateau_vs_r882[stat.PRIMARY],
        plateau_vs_r3175[stat.PRIMARY],
    )
    import sklearn

    metrics = dict(
        schema="jope.h226.gated025-r3175/v1",
        status="COMPLETE",
        verdict=decision["verdict"],
        decision=decision,
        primary_endpoint="flywheel native fixed-dev-cost-argmin-negative-fitness/v1 (full-pool first argmin; -fitness valid, -2.0 invalid)",
        structure=structure,
        contrasts_native=contrasts,
        contrasts_native_vs_r3175=contrasts_vs_r3175,
        paired_g3175_vs_g882=paired_g3175_vs_g882,
        contrasts_u4_report_only=u4_contrasts,
        contrasts_u4_vs_r3175_report_only=u4_contrasts_vs_r3175,
        elo_native_vs_r882=elo,
        elo_native_vs_r3175=elo_vs_r3175,
        elo_u4_vs_r882_report_only=elo_u4,
        minus2_native_per_seed=minus2,
        dl_val_descriptive=dl_val,
        dl_val_plateau_co_primary=dict(
            vs_R882=plateau_vs_r882[stat.PRIMARY],
            vs_R3175=plateau_vs_r3175[stat.PRIMARY],
        ),
        dl_val_plateau_vs_r882_report_only={
            a: plateau_vs_r882[a] for a in stat.ARMS[2:]
        },
        dl_val_plateau_vs_r3175_report_only={
            a: plateau_vs_r3175[a] for a in ("G3175_DIS", "B3175", "G882", "R882")
        },
        dl_val_top1_flips=dl_val_top1_flips,
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
        fits=len(fits),
        fit_wall_seconds=sum(f["wall_seconds"] for f in fits.values()),
        resumed=resume,
        dope_before=dope_882["train"]["before"],
        dope_after=dope_3175["train"]["after"],
        dope_receipts=dict(R882=dope_882, R3175=dope_3175),
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
            worker_affinity_cores=len(os.sched_getaffinity(0)),
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
        **{f"val_picks_{a}": v for a, v in val_picks.items()},
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
    "H222": HERE.parent / "h222_lipschitz_adapter/fold_manifest.json",
    "H223": HERE.parent / "h223_bagged_blend_gate/fold_manifest.json",
    "H224": HERE.parent / "h224_learned_switch_gate/fold_manifest.json",
    "H225": HERE.parent / "h225_multistep_regret/fold_manifest.json",
}


def write_fold_manifest():
    """Outcome-free: TRAIN context/family identities from the pinned selected.json only, plus the overlap with the
    registered earlier waves (H203, H210, H217, H218, H222, H223, H224, H225)."""
    h124 = load(H139_PATH, "h226_h139", H139_SHA).load_h124()
    selected = staged(h124.INPUT) / "selected.json"
    if sha(selected) != h124.FILES[h124.INPUT / "selected.json"]:
        raise RuntimeError("selected packet changed")
    rows = json.loads(selected.read_text())["train"]
    earlier = {}
    for w, p in EARLIER_WAVES.items():
        d = json.loads(p.read_text())
        earlier[w] = d.get("families_by_seed") or d.get("folds")
    (HERE / "fold_manifest.json").write_text(
        json.dumps(
            stat.fold_manifest(rows, earlier),
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
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="run exact R882/R3175 DL-Val91 regression gates and 1 fit on a smoke-* root",
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
        run(a.root, a.launch, resume=a.resume, smoke=a.smoke)


if __name__ == "__main__":
    main()

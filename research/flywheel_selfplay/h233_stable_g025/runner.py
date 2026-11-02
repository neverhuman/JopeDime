"""One monitored H233 CPU attempt: STABLE_G025 (apply the frozen H217 GATED_025 blend to a context ONLY when (a) the
H217 zgap gate passes AND (b) the blended first-argmin pick is unanimous across 3 outer HistGBM members — member 0 is
the H217 unweighted fit, members 1-2 are FIT-family bootstrap refits with the same recipe — else fall back to r882)
vs R882 on the flywheel native first-argmin protocol, 205-family / 1,219-context pooled cross-fit (fresh seeds
233101-233103, fold seeds 233141-233143), with G025 (frozen H217 recipe) as the replication anchor, STABLE_SHUF
(same vote rate, member agreement mask permuted across contexts within each FIT-calibrated cell) as placebo,
BAG_VOTE_UNGATED (unanimity vote without zgap) as report-only, and DL-Val91 (19 families, 91 contexts) as secondary.

Budget (hw_t2205): 5 outer x 3 members + 10 inner x 1 member = 25 HistGBM fits per seed, 75 total across 3 seeds.
Resources (hw_t2205): xbabe2 CPU only at nice 19, taskset <= 4 cores (up to 8 if load5 < 50), RSS < 4 GiB,
MemAvailable >= 20 GiB, disk free >= 300 GB, load5 > 60 pause / <= 55 resume, no GPU.
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
DATA = Path.home() / "jope-data/h233"
INPUTS = DATA / "inputs"
PREREG_SHA = "6f2bec113c1fc3161db0ff822c8472881ffbd83262772eb597f0d4e9114fddad"
H139_PATH = REPO / "research/flywheel_selfplay/h139_selection_audit/runner.py"
H139_SHA = "7a37821db82c89ddf8e48266383c681b20535ff31b95504e259c74dbe5a9c686"
BOUND_FILES = (
    "research/flywheel_selfplay/h233_stable_g025/PREREG.md",
    "research/flywheel_selfplay/h233_stable_g025/runner.py",
    "research/flywheel_selfplay/h233_stable_g025/analysis.py",
    "research/flywheel_selfplay/h233_stable_g025/monitor.py",
    "research/flywheel_selfplay/h139_selection_audit/runner.py",
    "research/flywheel_selfplay/h124_gbm_shrink_blend/runner.py",
    "python/src/jope/north_star/dope_freeze_guard.py",
)
HOST_RULES = {
    "xbabe2": dict(min_free=300_000_000_000, max_used=None),
}
MEM_MIN = 20 * 2**30  # hw_t2205: xbabe2 swap is full; stop below 20 GiB available
LOAD5_FAST_MAX = 50.0  # hw_t2205: taskset <= 4 cores (up to 8 if load5 < 50)
LOAD5_MAX = 60.0  # hw_t2205: pause before a unit while load5 > 60
LOAD5_RESUME = 55.0  # ... until load5 <= 55 (rechecked every 60 s)
LOAD_POLL = 60.0
CORES = len(os.sched_getaffinity(0))
DEADLINE_ENV = "H233_DEADLINE_EPOCH"
SCRATCH_MAX = 1_000_000_000  # 75 prediction files (< 150 MB)
RSS_LIMIT = 4 * 2**30  # hw_t2205: process-tree RSS < 4 GiB
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


def arm_costs(stat_mod, base_cost, member_priors, tau, shuf_agree):
    """Compute all 5 H233 arms' cost vectors, gate decisions, and outcome-free signals for one context."""
    return stat_mod.context_arm_costs(base_cost, member_priors, tau, shuf_agree)


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
    """Map a pinned ~/jope-data input path into the H233 staging layout (same relative path under DATA/inputs)."""
    base = Path.home() / "jope-data"
    return (
        INPUTS / Path(path).relative_to(base)
        if Path(path).is_relative_to(base)
        else Path(path)
    )


def load_modules():
    """Pinned H139/H124 bytes, pointed at the H233 staging layout (~/jope-data/h233/inputs on xbabe2).
    Every staged file keeps its pinned SHA-256 (verified by the inherited loaders); only the directory changes."""
    h139 = load(H139_PATH, "h233_h139", H139_SHA)
    h124 = h139.load_h124()
    h124.RSS_LIMIT_KIB = RSS_LIMIT // 1024
    h124.INPUT, h124.CACHE, h124.R882 = (
        staged(p) for p in (h124.INPUT, h124.CACHE, h124.R882)
    )
    h124.FILES = {staged(p): d for p, d in h124.FILES.items()}
    h139.INPUT = h124.INPUT
    h139.ARM_DIR = INPUTS
    return h139, h124


def tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def admit_host():
    """Preregistered host boundary: xbabe2 only, at nice 19. Called before any read, write or launch."""
    host = socket.gethostname().split(".")[0]
    if host not in HOST_RULES or os.getpriority(os.PRIO_PROCESS, 0) < 19:
        raise RuntimeError(f"H233 runs on xbabe2 at nice 19 only (host {host!r})")
    return host


def deadline():
    """The absolute deadline (epoch seconds) the monitor derived from the attempt claim; None outside a monitored run."""
    value = os.environ.get(DEADLINE_ENV)
    return None if value is None else float(value)


def guards(root):
    """xbabe2 disk rule (hw_t2205: free >= 300 GB), MemAvailable >= 20 GiB, nice 19, own root < 1 GB, and the
    absolute deadline: hard aborts, checked before every unit and stage. Host load is a pause: see wait_for_load."""
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
    """hw_t2205: pause between fit units while load5 > 60, resuming only once load5 <= 55.
    The absolute deadline still terminates a paused run."""
    load5 = os.getloadavg()[1]
    if load5 <= LOAD5_MAX:
        return 0.0
    t0 = time.monotonic()
    print("LOAD PAUSE", round(load5, 2), flush=True)
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
    """H233 fit unit: outer exclusion sets (len==2, exclude {A,B}) fit 3 members (0 = unweighted H217 fit, 1-2 =
    FIT-family bootstrap refits); inner exclusion sets (len==3, exclude {A,B,j}) fit member 0 only (H217 FIT
    calibration and tau selection). Total: 5*3 + 10*1 = 25 HistGBM fits per seed, 75 total."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    if (
        seed not in stat.SEEDS
        or tag != "BLEND"
        or len(excluded) not in (2, 3)
        or tuple(sorted(set(excluded))) != excluded
    ):
        raise ValueError("unregistered fit")
    members = stat.members_for_exclusion(excluded)
    outs = [
        root / (unit_name(tag, seed, excluded, m) + ".npz") for m in members
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
    omp_threads = 7 if os.getloadavg()[1] < LOAD5_FAST_MAX else 3
    with threadpool_limits(omp_threads):
        for member in members:
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
                None
                if weights is None
                else hashlib.sha256(weights.tobytes()).hexdigest()
            )
            del weights
        del x, y
        h124.release_memory()
        predictions = [{} for _ in models]
        targets_rows = [
            (r, pools, context)
            for r in rows
            if mapping[r["context_sha256"]] in excluded
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
            for idx, model in enumerate(models):
                predictions[idx][cid] = model.predict(xp).astype(np.float32)
    unit_wall = time.monotonic() - t0
    receipts = []
    for idx, (member, out, model) in enumerate(zip(members, outs, models)):
        np.savez_compressed(out, **predictions[idx])
        receipt = dict(
            tag=tag,
            seed=seed,
            member=member,
            excluded_folds=list(excluded),
            fit_contexts=len(fit_rows),
            fit_rows=len(fit_rows) * 192,
            sample_weight_sha256=weights_sha[idx],
            iterations=int(model.n_iter_),
            prediction_sha256=sha(out),
            predicted_contexts=len(predictions[idx]),
            fit_wall_seconds=walls[idx],
            unit_wall_seconds=unit_wall,
            omp_threads=omp_threads,
            shuffled=False,
        )
        save(out.with_suffix(".json"), receipt)
        receipts.append(receipt)
    print(
        "FIT",
        unit_name(tag, seed, excluded, "*" if len(members) > 1 else "0"),
        round(time.monotonic() - t0, 1),
        [round(w, 1) for w in walls],
        flush=True,
    )
    return receipts


def verified_unit(root, tag, seed, excluded):
    """Single-resume rule: an existing unit is reused only if every required member's receipt and npz hash agree."""
    members = stat.members_for_exclusion(excluded)
    receipts = []
    for member in members:
        name = unit_name(tag, seed, excluded, member)
        npz, js = root / (name + ".npz"), root / (name + ".json")
        if npz.exists() and js.exists():
            receipt = json.loads(js.read_text())
            if receipt["prediction_sha256"] == sha(npz):
                receipts.append(receipt)
                continue
        break
    if len(receipts) == len(members):
        return receipts
    for member in members:
        name = unit_name(tag, seed, excluded, member)
        for p in (root / (name + ".npz"), root / (name + ".json")):
            if p.exists():
                p.rename(p.with_name(p.name + ".incomplete"))
    return None


def run_smoke(root, launch_path):
    """TRAIN-only smoke check (0 DL-Val reads): fits 1 outer unit (3 members) + 1 inner unit (1 member) on a 25-context
    TRAIN subset (5 contexts per fold of seed 233101) to verify the end-to-end pipeline and guards before primary."""
    admit_host()
    if deadline() is None:
        raise RuntimeError(
            "H233 runs only under its monitor (absolute deadline missing)"
        )
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    root = Path(root)
    if root.parent != DATA or not root.name.startswith("smoke-"):
        raise RuntimeError("H233 smoke requires a smoke-* root under ~/jope-data/h233")
    if root.exists():
        raise RuntimeError("fresh smoke root required")
    launch = json.loads(Path(launch_path).read_text())
    bound = bind_source(launch)
    root.mkdir(parents=True)
    launch_guard = guards(root)
    h139, h124 = load_modules()
    t0 = time.monotonic()
    h124.FOLD_SEED, h124.SAMPLE_SEED = stat.FOLD_SEEDS[stat.SEEDS[0]], stat.SAMPLE_SEED
    troot = root / "train"
    troot.mkdir()
    selected, context, pools, samples, _, reads = h124.load_data(troot)
    all_rows = selected["train"]
    fams = [r["family_id"] for r in all_rows]
    folds0 = stat.seed_folds(fams, stat.SEEDS[0])
    map0 = {
        r["context_sha256"]: next(
            k for k, g in enumerate(folds0) if r["family_id"] in g
        )
        for r in all_rows
    }
    sub_rows = []
    for k in range(5):
        sub_rows.extend([r for r in all_rows if map0[r["context_sha256"]] == k][:5])
    projection = h124.train_projection(all_rows, selected["dev"])
    base, net = bounded_threads(
        h124.baseline, troot, sub_rows, context, pools, projection
    )
    del net
    h124.release_memory()
    dope = json.loads((troot / "dope-train.json").read_text())
    if dope["before"] != h124.DOPE or dope["after"] != h124.DOPE:
        raise RuntimeError("DOPE changed during smoke")
    r_outer = fit_unit(
        h124, "BLEND", stat.SEEDS[0], (0, 1), sub_rows, map0, context, pools, samples, root
    )
    r_inner = fit_unit(
        h124, "BLEND", stat.SEEDS[0], (0, 1, 2), sub_rows, map0, context, pools, samples, root
    )
    with np.load(
        root / (unit_name("BLEND", stat.SEEDS[0], (0, 1, 2), 0) + ".npz"),
        allow_pickle=False,
    ) as z:
        inner_m0 = {k: z[k].astype(np.float64) for k in z.files}
    fit_ids = sorted(c for c in inner_m0 if map0[c] == 2)
    cal = list(
        h124.calibration(
            np.concatenate([inner_m0[c][4:] for c in fit_ids]),
            np.concatenate([pools[c]["fitness"][4:] for c in fit_ids]),
            np.concatenate([pools[c]["gate"][4:] for c in fit_ids]),
        )
    )
    outer_m = []
    for m in range(stat.MEMBERS):
        with np.load(
            root / (unit_name("BLEND", stat.SEEDS[0], (0, 1), m) + ".npz"),
            allow_pickle=False,
        ) as z:
            outer_m.append({k: z[k].astype(np.float64) for k in z.files})
    a_ids = sorted(r["context_sha256"] for r in sub_rows if map0[r["context_sha256"]] == 0)
    agree_a = {}
    for cid in a_ids:
        priors = [cal[0] - cal[1] * outer_m[m][cid] for m in range(stat.MEMBERS)]
        agree_a[cid], _ = stat.unanimous_pick([stat.blend(base[cid], p) for p in priors])
    shuf_a = stat.shuffle_agreement_mask(agree_a, stat.SEEDS[0], 0, "A")
    for cid in a_ids:
        priors = [cal[0] - cal[1] * outer_m[m][cid] for m in range(stat.MEMBERS)]
        costs, opened, _ = arm_costs(stat, base[cid], priors, 0.5, shuf_a[cid])
        if set(costs) != set(stat.ARMS):
            raise RuntimeError("smoke arm set mismatch")
    smoke_receipt = dict(
        schema="jope.h233.smoke/v1",
        status="PASS",
        subset_train_contexts=len(sub_rows),
        outer_members_fitted=len(r_outer),
        inner_members_fitted=len(r_inner),
        total_smoke_fits=len(r_outer) + len(r_inner),
        dl_val_contexts_read=0,
        protected_reads=0,
        dope_before=dope["before"],
        dope_after=dope["after"],
        bound_sha256=bound,
        launch_guard=launch_guard,
        wall_seconds=time.monotonic() - t0,
    )
    save(root / "smoke-metrics.json", smoke_receipt, exclusive=True)
    print("SMOKE PASS", round(smoke_receipt["wall_seconds"], 1), "s", flush=True)
    return smoke_receipt


def run(root, launch_path, resume=False):
    admit_host()
    if deadline() is None:
        raise RuntimeError(
            "H233 runs only under its monitor (absolute deadline missing)"
        )
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    root = Path(root)
    if root.parent != DATA or not root.name.startswith("run-"):
        raise RuntimeError("H233 requires a run-* root under ~/jope-data/h233")
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

    # 3. Fits (5 outer x 3 members + 10 inner x 1 member = 25 fits/seed, 75 total), checkpointed per unit;
    # DL-Val predicted by the outer units.
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
    if len(fits) != 45 or sum(len(v) for v in fits.values()) != 75:
        raise RuntimeError("45 units (15 outer x 3 + 30 inner x 1 = 75 member receipts) required")

    def load_unit_members(seed, ex):
        members = stat.members_for_exclusion(ex)
        out = []
        for m in members:
            with np.load(
                root / (unit_name("BLEND", seed, ex, m) + ".npz"), allow_pickle=False
            ) as z:
                out.append({k: z[k].astype(np.float64) for k in z.files})
        return out

    def fitness_native(cost, pool):
        return h139.native(cost, pool["fitness"], pool["gate"])

    # 4. Stage 1: FIT-only calibration (member 0 inner OOF), FIT-only zgap threshold tau (H217 choose_tau),
    # outcome-free 3-member unanimity masks and deterministic per-cell STABLE_SHUF permutations, and B panels;
    # lock fsynced before any A or DL-Val scoring.
    family_of = {r["context_sha256"]: r["family_id"] for r in rows}
    val_ids = sorted(r["context_sha256"] for r in val)
    cells = []
    cell_runtime = {}
    for seed in stat.SEEDS:
        for k in range(5):
            a, b, fit = stat.cell(k)
            inner_m0 = {}
            for j in fit:
                p0 = load_unit_members(seed, stat.inner_model_for(k, j))[0]
                inner_m0.update({c: v for c, v in p0.items() if mapping[seed][c] == j})
            fit_ids = sorted(inner_m0)
            cal = list(
                h124.calibration(
                    np.concatenate([inner_m0[c][4:] for c in fit_ids]),
                    np.concatenate([pools[c]["fitness"][4:] for c in fit_ids]),
                    np.concatenate([pools[c]["gate"][4:] for c in fit_ids]),
                )
            )
            # FIT-only tau selection on member 0 (frozen H217 GATED_025 rule)
            zgap_fit, fit_r882, fit_blend0, fit_fams = [], [], [], []
            for c in fit_ids:
                b0 = stat.blend(base[c], cal[0] - cal[1] * inner_m0[c])
                zgap_fit.append(stat.zgap(base[c], b0))
                fit_r882.append(fitness_native(base[c], pools[c]))
                fit_blend0.append(fitness_native(b0, pools[c]))
                fit_fams.append(family_of[c])
            gate_choice = stat.choose_tau(zgap_fit, fit_r882, fit_blend0, fit_fams)
            tau = gate_choice["tau"]

            pair = tuple(sorted((a, b)))
            outer_members = load_unit_members(seed, pair)  # 3 members: 0, 1, 2
            a_ids = sorted(
                r["context_sha256"]
                for r in rows
                if mapping[seed][r["context_sha256"]] == a
            )
            b_ids = sorted(
                r["context_sha256"]
                for r in rows
                if mapping[seed][r["context_sha256"]] == b
            )
            # Outcome-free member agreement masks on A and DL-Val, plus deterministic per-cell permutation (STABLE_SHUF)
            agree_a = {}
            for c in a_ids:
                priors = [cal[0] - cal[1] * outer_members[m][c] for m in range(stat.MEMBERS)]
                agree_a[c], _ = stat.unanimous_pick([stat.blend(base[c], p) for p in priors])
            shuf_a = stat.shuffle_agreement_mask(agree_a, seed, k, "A")

            agree_val = {}
            for c in val_ids:
                priors = [
                    cal[0] - cal[1] * outer_members[m][c] for m in range(stat.MEMBERS)
                ]
                agree_val[c], _ = stat.unanimous_pick(
                    [stat.blend(val_costs[c], p) for p in priors]
                )
            shuf_val = stat.shuffle_agreement_mask(agree_val, seed, k, "VAL")

            b_panel_sha = hashlib.sha256(
                np.array(
                    [
                        fitness_native(
                            stat.blend(base[c], cal[0] - cal[1] * outer_members[0][c]),
                            pools[c],
                        )
                        for c in b_ids
                    ]
                ).tobytes()
            ).hexdigest()

            fit_pred_shas = {}
            for m in range(stat.MEMBERS):
                fit_pred_shas[f"{''.join(map(str, pair))}-m{m}"] = fits[(seed, pair)][m][
                    "prediction_sha256"
                ]
            for j in fit:
                in_ex = stat.inner_model_for(k, j)
                fit_pred_shas[f"{''.join(map(str, in_ex))}-m0"] = fits[(seed, in_ex)][0][
                    "prediction_sha256"
                ]

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
                    a_contexts=len(a_ids),
                    a_vote_unanimous_count=int(sum(agree_a.values())),
                    a_vote_shuf_count=int(sum(shuf_a.values())),
                    a_vote_rate=float(np.mean(list(agree_a.values()))),
                    b_panel_blend025_native_sha256=b_panel_sha,
                    fit_prediction_sha256=fit_pred_shas,
                )
            )
            cell_runtime[(seed, k)] = (outer_members, cal, tau, agree_a, shuf_a, agree_val, shuf_val)

    lock = root / ("stage1_lock-resume.json" if resume else "stage1_lock.json")
    save(
        lock,
        dict(written_before_any_A_or_DL_Val_scoring=True, cells=cells),
        exclusive=True,
    )
    lock_sha = sha(lock)
    print("LOCK", lock_sha[:12], flush=True)

    # 5. Stage 2: pooled A (205 families, 1,219 contexts), every arm: flywheel native first-argmin (primary) and U4 (report-only).
    index = {r["context_sha256"]: i for i, r in enumerate(rows)}
    u4 = {a: np.full((3, len(rows)), np.nan) for a in stat.ARMS}
    native = {a: np.full((3, len(rows)), np.nan) for a in stat.ARMS}
    opened_a = {arm: np.zeros((3, len(rows)), bool) for arm in stat.GATES}
    switched_a = np.zeros((3, len(rows)), bool)
    unanimous_a = np.zeros((3, len(rows)), bool)
    shuf_unanimous_a = np.zeros((3, len(rows)), bool)

    for c in cells:
        seed, k = c["seed"], c["outer"]
        si = stat.SEEDS.index(seed)
        outer_members, cal, tau, agree_a, shuf_a, _, _ = cell_runtime[(seed, k)]
        for r in rows:
            cid = r["context_sha256"]
            if mapping[seed][cid] != c["A"]:
                continue
            priors = [cal[0] - cal[1] * outer_members[m][cid] for m in range(stat.MEMBERS)]
            costs, opened, signal = arm_costs(
                stat, base[cid], priors, tau, shuf_a[cid]
            )
            for arm in stat.GATES:
                opened_a[arm][si, index[cid]] = opened[arm]
            switched_a[si, index[cid]] = signal["zgap"] != 0.0
            unanimous_a[si, index[cid]] = signal["unanimous"]
            shuf_unanimous_a[si, index[cid]] = signal["shuf_unanimous"]
            for arm, cost in costs.items():
                e = h124.endpoints(cost, pools[cid]["u"])
                nat = fitness_native(cost, pools[cid])
                if nat != e["legacy"]:
                    raise RuntimeError("flywheel native != first-argmin utility")
                u4[arm][si, index[cid]] = e["u4"]
                native[arm][si, index[cid]] = nat

    if not np.array_equal(unanimous_a.sum(axis=1), shuf_unanimous_a.sum(axis=1)):
        raise RuntimeError("STABLE_SHUF vote rate diverged from STABLE_G025 member unanimity rate")

    gate_rates = {
        arm: dict(
            open_rate_per_seed=opened_a[arm].mean(axis=1).tolist(),
            open_rate_among_switching_per_seed=[
                float(opened_a[arm][s][switched_a[s]].mean()) if switched_a[s].any() else 0.0
                for s in range(3)
            ],
        )
        for arm in stat.GATES
    }
    gate_rates["switching_rate_per_seed"] = switched_a.mean(axis=1).tolist()
    gate_rates["unanimous_vote_rate_per_seed"] = unanimous_a.mean(axis=1).tolist()
    gate_rates["shuf_vote_rate_per_seed"] = shuf_unanimous_a.mean(axis=1).tolist()

    def panel(values, arm):
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

    # Paired contrasts required for admission: STABLE_G025 vs STABLE_SHUF and STABLE_G025 vs G025
    stable_vs_shuf = stat.contrast(
        panel(native, "STABLE_G025"), panel(native, "STABLE_SHUF"), fams
    )
    stable_vs_g025 = stat.contrast(
        panel(native, "STABLE_G025"), panel(native, "G025"), fams
    )
    elo_stable_vs_shuf = stat.elo_vs_reference(
        panel(native, "STABLE_G025"), panel(native, "STABLE_SHUF"), fams
    )
    elo_stable_vs_g025 = stat.elo_vs_reference(
        panel(native, "STABLE_G025"), panel(native, "G025"), fams
    )

    # 6. Secondary DL-Val91 (19 families, 91 contexts): both 3-seed equal-family contrasts (averaging the 5 outer
    # cells per seed) and 15-cell plateau readout.
    vf = [r["family_id"] for r in val]
    val_index = {r["context_sha256"]: i for i, r in enumerate(val)}
    val_native_cells = {a: np.zeros((3, 5, len(val)), np.float64) for a in stat.ARMS}
    val_u4_cells = {a: np.zeros((3, 5, len(val)), np.float64) for a in stat.ARMS}

    for c in cells:
        seed, k = c["seed"], c["outer"]
        si = stat.SEEDS.index(seed)
        outer_members, cal, tau, _, _, _, shuf_val = cell_runtime[(seed, k)]
        for r in val:
            cid = r["context_sha256"]
            vi = val_index[cid]
            priors = [
                cal[0] - cal[1] * outer_members[m][cid] for m in range(stat.MEMBERS)
            ]
            costs, _, _ = arm_costs(
                stat, val_costs[cid], priors, tau, shuf_val[cid]
            )
            for arm, cost in costs.items():
                val_u4_cells[arm][si, k, vi] = h124.endpoints(cost, val_pools[cid]["u"])["u4"]
                val_native_cells[arm][si, k, vi] = h139.native(
                    cost, val_pools[cid]["fitness"], val_pools[cid]["gate"]
                )

    val_native_by_seed = {a: val_native_cells[a].mean(axis=1) for a in stat.ARMS}
    per_arm = {
        a: {
            "native": val_native_cells[a].mean(axis=(0, 1)).tolist(),
            "u4": val_u4_cells[a].mean(axis=(0, 1)).tolist(),
        }
        for a in stat.ARMS
    }
    dl_val = {}
    for arm in stat.ARMS:
        u4v, nat = per_arm[arm]["u4"], per_arm[arm]["native"]
        dl_val[arm] = dict(
            native_context_mean=float(np.mean(nat)),
            native_equal_family=float(stat.family_matrix(np.array(nat), vf)[1].mean()),
            u4_equal_family=float(stat.family_matrix(np.array(u4v), vf)[1].mean()),
            native_minus2_in_every_cell=int(sum(x == -2.0 for x in nat)),
        )
    if not np.allclose(per_arm["R882"]["native"], got, rtol=0, atol=1e-12):
        raise RuntimeError(
            "DL-Val R882 per-context utilities differ from the regression"
        )
    val_contrasts_native = {
        a: stat.contrast(val_native_by_seed[a], val_native_by_seed["R882"], vf)
        for a in stat.ARMS[1:]
    }
    val_elo_native_vs_r882 = {
        a: stat.elo_vs_reference(val_native_by_seed[a], val_native_by_seed["R882"], vf)
        for a in stat.ARMS[1:]
    }
    val_stable_vs_shuf = stat.contrast(
        val_native_by_seed["STABLE_G025"], val_native_by_seed["STABLE_SHUF"], vf
    )
    val_stable_vs_g025 = stat.contrast(
        val_native_by_seed["STABLE_G025"], val_native_by_seed["G025"], vf
    )
    plateau = {
        arm: stat.plateau(per_arm[arm]["native"], got, vf) for arm in stat.ARMS[1:]
    }
    if plateau[stat.PRIMARY]["r882_context_mean"] != stat.R882_DL_VAL:
        raise RuntimeError("DL-Val r882 context mean differs from the pinned contract")

    decision = stat.verdict(
        contrasts[stat.PRIMARY],
        stable_vs_shuf,
        stable_vs_g025,
        val_contrasts_native[stat.PRIMARY],
        plateau[stat.PRIMARY],
    )
    paired = dict(
        stable_vs_g025=dict(
            native=stable_vs_g025,
            elo_native=elo_stable_vs_g025,
            dl_val_equal_family=val_stable_vs_g025,
            dl_val_plateau=stat.plateau(
                per_arm["STABLE_G025"]["native"], per_arm["G025"]["native"], vf
            ),
        ),
        stable_vs_shuf=dict(
            native=stable_vs_shuf,
            elo_native=elo_stable_vs_shuf,
            dl_val_equal_family=val_stable_vs_shuf,
            dl_val_plateau=stat.plateau(
                per_arm["STABLE_G025"]["native"], per_arm["STABLE_SHUF"]["native"], vf
            ),
        ),
    )

    overlapping_reads = int(sum(v["overlapping_reads"] for v in reads.values()))
    if overlapping_reads != 0:
        raise RuntimeError("forbidden split rows decoded during train cache read")

    import sklearn

    metrics = dict(
        schema="jope.h233.stable-g025/v1",
        status="COMPLETE",
        verdict=decision["verdict"],
        combined_verdict=decision["combined"],
        decision=decision,
        holm=stat.holm({stat.PRIMARY: contrasts[stat.PRIMARY]["one_sided_p_gt_margin"]}),
        primary_endpoint="flywheel native fixed-dev-cost-argmin-negative-fitness/v1 (full-pool first argmin; -fitness valid, -2.0 invalid)",
        contrasts_native=contrasts,
        contrasts_u4_report_only=u4_contrasts,
        elo_native_vs_r882=elo,
        elo_u4_vs_r882_report_only=elo_u4,
        paired_contrasts=paired,
        minus2_native_per_seed=minus2,
        dl_val_secondary=dict(
            descriptive_by_arm=dl_val,
            equal_family_contrasts_native=val_contrasts_native,
            elo_native_vs_r882=val_elo_native_vs_r882,
            plateau_by_arm=plateau,
        ),
        gate_rates_pooled_a=gate_rates,
        gate_choice_by_cell={
            f"{c['seed']}/{c['outer']}": {
                k: v for k, v in c["gate_choice"].items() if k != "grid"
            }
            for c in cells
        },
        vote_summary_by_cell={
            f"{c['seed']}/{c['outer']}": dict(
                a_contexts=c["a_contexts"],
                unanimous_count=c["a_vote_unanimous_count"],
                shuf_count=c["a_vote_shuf_count"],
                vote_rate=c["a_vote_rate"],
            )
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
        protected_reads=0,
        read_boundary=dict(
            train_contexts=len(rows),
            dl_val_contexts_for_regression_and_secondary=len(val),
            sealed=0,
            holdout=0,
            meta_test=0,
            live_loop=0,
            protected_reads=0,
            train_cache_dl_val_rows_decoded=overlapping_reads,
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
        **{f"val_native_{a}": v for a, v in val_native_by_seed.items()},
        context_ids=np.array([r["context_sha256"] for r in rows]),
        family_ids=np.array(fams),
        val_context_ids=np.array([r["context_sha256"] for r in val]),
        val_family_ids=np.array(vf),
    )
    (root / "metrics.json").write_text(raw)
    print("COMPLETE", metrics["combined_verdict"], flush=True)
    return metrics


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
        help="TRAIN-only 4-fit smoke verification before primary launch",
    )
    ap.add_argument(
        "--check-launch",
        action="store_true",
        help="verify xbabe2 nice-19 admission and exact git/PREREG binding, then exit",
    )
    args = ap.parse_args()
    if args.check_launch:
        admit_host()
        if not args.launch:
            ap.error("--check-launch requires --launch")
        bind_source(json.loads(args.launch.read_text()))
        print("SOURCE OK", flush=True)
        return
    if not (args.root and args.launch):
        ap.error("--root and --launch required")
    if args.smoke:
        run_smoke(args.root, args.launch)
    else:
        run(args.root, args.launch, resume=args.resume)


if __name__ == "__main__":
    main()

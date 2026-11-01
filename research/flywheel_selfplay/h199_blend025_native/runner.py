"""One monitored H199 CPU attempt: BLEND_025 vs R882 on the flywheel native first-argmin protocol, powered by the
205-family pooled cross-fit (H195 machinery, fresh seeds and fold partitions).

Reuses, byte-pinned and unchanged: the merged H124 module (load_data, train_context/projection, baseline forward,
calibration, endpoints, program; via the merged H139 loader) and the merged H139 roster/pools/native helpers (the H152
regression path; `native` is the flywheel protocol fixed-dev-cost-argmin-negative-fitness/v1). `fit_gbm` is H124
`fit_one` with the H195 exclusion sets ({A,B} outer pairs and {A,B,j} inner triples), a per-seed fold map, the SHUF
within-family target permutation and DL-Val prediction for the descriptive readout.
"""

import argparse
import hashlib
import importlib.util
import json
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
DATA = Path.home() / "jope-data/h199"
INPUTS = DATA / "inputs"
PREREG_SHA = "cffb4463bc049cd5a3015f6a93550d24a4d1d9e57c3bf31bda012adc2c5453fb"
H139_PATH = REPO / "research/flywheel_selfplay/h139_selection_audit/runner.py"
H139_SHA = "7a37821db82c89ddf8e48266383c681b20535ff31b95504e259c74dbe5a9c686"
BOUND_FILES = (
    "research/flywheel_selfplay/h199_blend025_native/PREREG.md",
    "research/flywheel_selfplay/h199_blend025_native/runner.py",
    "research/flywheel_selfplay/h199_blend025_native/analysis.py",
    "research/flywheel_selfplay/h199_blend025_native/monitor.py",
    "research/flywheel_selfplay/h139_selection_audit/runner.py",
    "research/flywheel_selfplay/h124_gbm_shrink_blend/runner.py",
    "python/src/jope/north_star/dope_freeze_guard.py",
)
HOST_RULES = {
    "xbabe1": dict(min_free=40_000_000_000, max_used=None),
    "xbabe2": dict(min_free=40_000_000_000, max_used=0.938),
}
MEM_MIN = 16 * 2**30
SCRATCH_MAX = 500_000_000
RSS_LIMIT = 6_000_000_000
METRICS_LIMIT = 100_000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(path, obj, exclusive=False):
    with Path(path).open("x" if exclusive else "w") as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n")
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
    """Map a pinned ~/jope-data input path into the H199 staging layout (same relative path under DATA/inputs)."""
    base = Path.home() / "jope-data"
    return (
        INPUTS / Path(path).relative_to(base)
        if Path(path).is_relative_to(base)
        else Path(path)
    )


def load_modules():
    """Pinned H139/H124 bytes, pointed at the H199 staging layout so the run is host-portable (xbabe1 or xbabe2).
    Every staged file keeps its pinned SHA-256 (verified by the inherited loaders); only the directory changes."""
    h139 = load(H139_PATH, "h199_h139", H139_SHA)
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


def guards(root):
    """Host-specific disk rule (xbabe1: free >= 40 GB; xbabe2: used <= 93.8% and free >= 40 GB), MemAvailable >= 16
    GiB, load1 <= 85% of cores, nice 19, own root < 500 MB. Checked before every fit and stage."""
    host = socket.gethostname().split(".")[0]
    if host not in HOST_RULES or os.getpriority(os.PRIO_PROCESS, 0) < 19:
        raise RuntimeError("H199 runs on xbabe1/xbabe2 at nice 19 only")
    rule = HOST_RULES[host]
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
    load = os.getloadavg()[0]
    if mem < MEM_MIN or load > 0.85 * os.cpu_count():
        raise RuntimeError(f"host guard: MemAvailable {mem} B, load {load}")
    scratch = tree_bytes(root) if Path(root).exists() else 0
    if scratch >= SCRATCH_MAX:
        raise RuntimeError(f"own output root {scratch} B >= 500 MB")
    return dict(
        host=host,
        free=usage.free,
        used=round(used, 5),
        mem_available=mem,
        load1=load,
        scratch=scratch,
    )


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


def fit_gbm(
    h124, tag, seed, excluded, rows, mapping, context, pools, samples, root, val=None
):
    """H124 fit_one, H195/H199 rules: exclusion size 2 (outer {A,B}) or 3 (inner {A,B,j}); SHUF permutes the training
    targets within family. Predictions for the excluded folds' contexts (and DL-Val contexts when given)."""
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


def run(root, launch_path, resume=False):
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    root = Path(root)
    if root.parent != DATA or not root.name.startswith("run-"):
        raise RuntimeError("H199 requires a run-* root under ~/jope-data/h199")
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

    # 3. Fits (15 per seed and type: BLEND x 3 seeds, SHUF x 1 seed), checkpointed per fit; DL-Val by the outer models.
    outers, inners = stat.fit_plan()
    fits = {}
    for tag in ("BLEND", "SHUF"):
        for seed in stat.FIT_SEEDS[tag]:
            for ex in outers + inners:
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
    if len(fits) != 60:
        raise RuntimeError("60 fit receipts required")

    def preds(tag, seed, ex):
        with np.load(
            root / f"{tag}-seed{seed}-exclude{''.join(map(str, ex))}.npz",
            allow_pickle=False,
        ) as z:
            return {k: z[k].astype(np.float64) for k in z.files}

    # 4. Stage 1: fit-only calibration per cell and the B panels; lock fsynced before any A scoring.
    cells = []
    for seed in stat.SEEDS:
        for k in range(5):
            a, b, fit = stat.cell(k)
            cal = {}
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
            outer = preds("BLEND", seed, tuple(sorted((a, b))))
            b_rows = [r for r in rows if mapping[seed][r["context_sha256"]] == b]
            b_native = np.array(
                [
                    h139.native(
                        stat.blend(
                            base[r["context_sha256"]],
                            cal["BLEND"][0]
                            - cal["BLEND"][1] * outer[r["context_sha256"]],
                        ),
                        pools[r["context_sha256"]]["fitness"],
                        pools[r["context_sha256"]]["gate"],
                    )
                    for r in b_rows
                ]
            )
            cells.append(
                dict(
                    seed=seed,
                    outer=k,
                    A=a,
                    B=b,
                    FIT=list(fit),
                    calibration=cal,
                    b_panel_blend025_native_sha256=hashlib.sha256(
                        b_native.tobytes()
                    ).hexdigest(),
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
    val_gbm = {"BLEND": [], "SHUF": []}
    for c in cells:
        seed, si = c["seed"], stat.SEEDS.index(c["seed"])
        pair = tuple(sorted((c["A"], c["B"])))
        ob = preds("BLEND", seed, pair)
        ab, bb = c["calibration"]["BLEND"]
        shuf = "SHUF" in c["calibration"]
        if shuf:
            osh = preds("SHUF", seed, pair)
            asf, bsf = c["calibration"]["SHUF"]
        for r in rows:
            cid = r["context_sha256"]
            if mapping[seed][cid] != c["A"]:
                continue
            gb = ab - bb * ob[cid]
            costs = {"R882": base[cid]}
            costs.update(
                {
                    arm: stat.blend(base[cid], gb, lam)
                    for arm, lam in stat.LAMBDAS.items()
                }
            )
            if shuf:
                costs["BLEND_025_SHUF"] = stat.blend(base[cid], asf - bsf * osh[cid])
            for arm, cost in costs.items():
                e = h124.endpoints(cost, pools[cid]["u"])
                nat = h139.native(cost, pools[cid]["fitness"], pools[cid]["gate"])
                if nat != e["legacy"]:
                    raise RuntimeError("flywheel native != first-argmin utility")
                u4[arm][si, index[cid]] = e["u4"]
                native[arm][si, index[cid]] = nat
        val_gbm["BLEND"].append(
            {r["context_sha256"]: ab - bb * ob[r["context_sha256"]] for r in val}
        )
        if shuf:
            val_gbm["SHUF"].append(
                {r["context_sha256"]: asf - bsf * osh[r["context_sha256"]] for r in val}
            )

    def panel(values, arm):
        """Seed rows of an arm: the SHUF placebo has the first seed only (R882 is seed-invariant)."""
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
    decision = stat.verdict(contrasts[stat.PRIMARY])

    # 6. DL-Val descriptive (mean calibrated outer-model prior over its cells: 15 BLEND, 5 SHUF).
    dl_val = {}
    for arm in stat.ARMS:
        u4v, nat = [], []
        for r in val:
            cid = r["context_sha256"]
            gb = np.mean([d[cid] for d in val_gbm["BLEND"]], axis=0)
            gs = np.mean([d[cid] for d in val_gbm["SHUF"]], axis=0)
            if arm == "R882":
                cost = val_costs[cid]
            elif arm == "BLEND_025_SHUF":
                cost = stat.blend(val_costs[cid], gs)
            else:
                cost = stat.blend(val_costs[cid], gb, stat.LAMBDAS[arm])
            u4v.append(h124.endpoints(cost, val_pools[cid]["u"])["u4"])
            nat.append(
                h139.native(cost, val_pools[cid]["fitness"], val_pools[cid]["gate"])
            )
        vf = [r["family_id"] for r in val]
        dl_val[arm] = dict(
            native_context_mean=float(
                np.mean(nat)
            ),  # the flywheel91 statistic (R882 = 1.1377941898164525; H135)
            native_equal_family=float(stat.family_matrix(np.array(nat), vf)[1].mean()),
            u4_equal_family=float(stat.family_matrix(np.array(u4v), vf)[1].mean()),
            native_minus2=int(sum(x == -2.0 for x in nat)),
        )
    import sklearn

    metrics = dict(
        schema="jope.h199.blend025-native/v1",
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
        fold_jaccard=stat.fold_jaccard(folds),
        calibration_by_cell={
            f"{c['seed']}/{c['outer']}": c["calibration"] for c in cells
        },
        stage1_lock_sha256=lock_sha,
        fits=len(fits),
        fit_wall_seconds=sum(f["wall_seconds"] for f in fits.values()),
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
    raw = json.dumps(metrics, indent=1, sort_keys=True, allow_nan=False) + "\n"
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


def write_fold_manifest():
    """Outcome-free: TRAIN context/family identities from the pinned selected.json only."""
    h124 = load(H139_PATH, "h199_h139", H139_SHA).load_h124()
    selected = h124.INPUT / "selected.json"
    if sha(selected) != h124.FILES[selected]:
        raise RuntimeError("selected packet changed")
    rows = json.loads(selected.read_text())["train"]
    (HERE / "fold_manifest.json").write_text(
        json.dumps(stat.fold_manifest(rows), indent=1, sort_keys=True) + "\n"
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
    a = ap.parse_args()
    if a.write_fold_manifest:
        write_fold_manifest()
    elif a.root is None or a.launch is None:
        ap.error("--root and --launch are required")
    else:
        run(a.root, a.launch, resume=a.resume)


if __name__ == "__main__":
    main()

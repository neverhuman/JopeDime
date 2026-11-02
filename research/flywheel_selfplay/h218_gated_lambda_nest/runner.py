"""One monitored H218 CPU attempt: nested blend-weight (lambda in {0.125, 0.25, 0.5}) + outcome-free zgap gate
chosen jointly on FIT only (inner 3-fold family CV out-of-fold predictions, maximize FIT native gain s.t. FIT harm <= 5%,
else r882) vs R882 on the flywheel native first-argmin protocol, 205-family pooled cross-fit on fresh seeds/folds, with
the DL-Val flywheel91 plateau as a co-primary.

Reuses, byte-pinned and unchanged: the merged H124 module (load_data, train_context/projection, baseline forward,
calibration, endpoints, program; via the merged H139 loader) and the merged H139 roster/pools/native helpers (the H152
regression path; `native` is the flywheel protocol fixed-dev-cost-argmin-negative-fitness/v1). `fit_gbm` is H124
`fit_one` with the H195 exclusion sets ({A,B} outer pairs and {A,B,j} inner triples), a per-seed fold map, and DL-Val
prediction. The functions `zgap`, `disagreement`, `tau_grid`, `gated`, `choose_tau`, and `blend` are byte-identical to
H210/H217; only `lambda` becomes the grid `{0.125, 0.25, 0.5}` nested on FIT only (`GATED_NEST`), compared against a
context-permuted `zgap` placebo (`GATED_NEST_SHUF`) and fixed `lambda = 0.25` (`GATED_025`, report-only reference).
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
DATA = Path.home() / "jope-data/h218"
INPUTS = DATA / "inputs"
PREREG_SHA = "453abed33b56a8807ab3d6374e97505197f5601d262d4112971d6a09f58e07f8"
H139_PATH = REPO / "research/flywheel_selfplay/h139_selection_audit/runner.py"
H139_SHA = "7a37821db82c89ddf8e48266383c681b20535ff31b95504e259c74dbe5a9c686"
BOUND_FILES = (
    "research/flywheel_selfplay/h218_gated_lambda_nest/PREREG.md",
    "research/flywheel_selfplay/h218_gated_lambda_nest/runner.py",
    "research/flywheel_selfplay/h218_gated_lambda_nest/analysis.py",
    "research/flywheel_selfplay/h218_gated_lambda_nest/monitor.py",
    "research/flywheel_selfplay/h139_selection_audit/runner.py",
    "research/flywheel_selfplay/h124_gbm_shrink_blend/runner.py",
    "python/src/jope/north_star/dope_freeze_guard.py",
)
HOST_RULES = {
    "xbabe2": dict(
        min_free=200_000_000_000, max_used=None
    ),  # hw_t0400: stop if xbabe2 free disk < 200 GB
}
MEM_MIN = 20 * 2**30  # hw_t0400: xbabe2 swap is full; stop below 20 GiB available
CORES = len(
    os.sched_getaffinity(0)
)  # effective allocation (H210 E1), not the online count
LOAD_LIMIT = min(85.0, 0.85 * CORES)  # hw_t0400: load guard 85
DEADLINE_ENV = "H218_DEADLINE_EPOCH"  # absolute deadline from the attempt claim, set by the monitor
SCRATCH_MAX = 500_000_000
RSS_LIMIT = 6_000_000_000
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


def arm_costs(stat_mod, base_cost, gbm_cost, gate_choice, zgap_shuf_override=None):
    """Every H218 arm's cost for one context and the gate decisions (outcome-free: costs only)."""
    lam_nest = float(gate_choice["GATED_NEST"]["lam"])
    tau_nest = float(gate_choice["GATED_NEST"]["tau"])
    lam_shuf = float(gate_choice["GATED_NEST_SHUF"]["lam"])
    tau_shuf = float(gate_choice["GATED_NEST_SHUF"]["tau"])
    tau_025 = float(gate_choice["GATED_025"]["tau"])

    blend_nest = stat_mod.blend(base_cost, gbm_cost, lam_nest)
    blend_shuf = stat_mod.blend(base_cost, gbm_cost, lam_shuf)
    blend_025 = stat_mod.blend(base_cost, gbm_cost, 0.25)

    sig_nest = stat_mod.zgap(base_cost, blend_nest)
    sig_shuf = (
        stat_mod.zgap(base_cost, blend_shuf)
        if zgap_shuf_override is None
        else float(zgap_shuf_override)
    )
    sig_025 = stat_mod.zgap(base_cost, blend_025)

    signal = dict(
        zgap_nest=sig_nest,
        zgap_nest_shuf=sig_shuf,
        zgap_025=sig_025,
    )
    opened = dict(
        GATED_NEST=stat_mod.gated(sig_nest, tau_nest),
        GATED_NEST_SHUF=stat_mod.gated(sig_shuf, tau_shuf),
        GATED_025=stat_mod.gated(sig_025, tau_025),
    )
    costs = dict(
        R882=base_cost,
        GATED_NEST=blend_nest if opened["GATED_NEST"] else base_cost,
        GATED_NEST_SHUF=blend_shuf if opened["GATED_NEST_SHUF"] else base_cost,
        GATED_025=blend_025 if opened["GATED_025"] else base_cost,
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
    """Map a pinned ~/jope-data input path into the H218 staging layout (same relative path under DATA/inputs)."""
    base = Path.home() / "jope-data"
    return (
        INPUTS / Path(path).relative_to(base)
        if Path(path).is_relative_to(base)
        else Path(path)
    )


def load_modules():
    """Pinned H139/H124 bytes, pointed at the H218 staging layout (~/jope-data/h218/inputs on xbabe2).
    Every staged file keeps its pinned SHA-256 (verified by the inherited loaders); only the directory changes."""
    h139 = load(H139_PATH, "h218_h139", H139_SHA)
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
        raise RuntimeError(f"H218 runs on xbabe2 at nice 19 only (host {host!r})")
    return host


def deadline():
    """The absolute deadline (epoch seconds) the monitor derived from the attempt claim; None outside a monitored run."""
    value = os.environ.get(DEADLINE_ENV)
    return None if value is None else float(value)


def guards(root):
    """xbabe2 disk rule (free >= 200 GB), MemAvailable >= 20 GiB, load5 <= LOAD_LIMIT (85), nice 19, own root < 500 MB,
    and the absolute deadline. Checked before every fit and stage."""
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
    if mem < MEM_MIN or load5 > LOAD_LIMIT:
        raise RuntimeError(f"host guard: MemAvailable {mem} B, load5 {load5}")
    scratch = tree_bytes(root) if Path(root).exists() else 0
    if scratch >= SCRATCH_MAX:
        raise RuntimeError(f"own output root {scratch} B >= 500 MB")
    return dict(
        host=host,
        effective_cores=CORES,
        load_limit=LOAD_LIMIT,
        deadline_epoch=end,
        free=usage.free,
        used=round(used, 5),
        mem_available=mem,
        load1=load1,
        load5=load5,
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
    """H124 fit_one, H195/H217/H218 rules: exclusion size 2 (outer {A,B}) or 3 (inner {A,B,j}); predictions for the
    excluded folds' contexts (and DL-Val contexts when given)."""
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
    """Single-resume / post-smoke rule: an existing fit is reused only if its receipt and npz hash agree."""
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


def compute_cell_stage1(
    seed, k, rows, mapping, base, pools, family_of, preds_fn, h124, h139, fits
):
    """Stage-1 FIT-only calibration and nested (lambda, tau) selection for one cell (seed, k)."""
    a, b, fit = stat.cell(k)
    cal, inner_by_tag = {}, {}
    tags = [t for t in ("BLEND", "SHUF") if seed in stat.FIT_SEEDS[t]]
    for tag in tags:
        inner = {}
        for j in fit:
            p = preds_fn(tag, seed, stat.inner_model_for(k, j))
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
    fa, fb_ = cal["BLEND"]
    fit_inner = inner_by_tag["BLEND"]
    fit_ids = sorted(fit_inner)
    fit_r882 = [
        h139.native(base[c], pools[c]["fitness"], pools[c]["gate"]) for c in fit_ids
    ]
    fit_fams = [family_of[c] for c in fit_ids]
    sig_by_lam = {lam: [] for lam in stat.LAMBDA_GRID}
    fit_blend_by_lam = {lam: [] for lam in stat.LAMBDA_GRID}
    for c in fit_ids:
        gbm_cost = fa - fb_ * fit_inner[c]
        for lam in stat.LAMBDA_GRID:
            blended_c = stat.blend(base[c], gbm_cost, lam)
            sig_by_lam[lam].append(stat.zgap(base[c], blended_c))
            fit_blend_by_lam[lam].append(
                h139.native(blended_c, pools[c]["fitness"], pools[c]["gate"])
            )
    shuf_sig_by_lam = {
        lam: stat.permute_zgap(
            sig_by_lam[lam],
            stat.stream_seed(stat.SHUF_SEED, "FIT", seed, k, int(round(lam * 1000))),
        )
        for lam in stat.LAMBDA_GRID
    }
    nest_choice = stat.choose_lambda_tau(
        sig_by_lam, fit_r882, fit_blend_by_lam, fit_fams
    )
    shuf_choice = stat.choose_lambda_tau(
        shuf_sig_by_lam, fit_r882, fit_blend_by_lam, fit_fams
    )
    g025_choice = stat.choose_tau(
        sig_by_lam[0.25], fit_r882, fit_blend_by_lam[0.25], fit_fams
    )
    g025_choice["lam"] = 0.25
    g025_choice["selected_lambda"] = (
        "R882" if g025_choice["tau"] == -math.inf else "0.25"
    )
    gate_choice = dict(
        GATED_NEST=nest_choice,
        GATED_NEST_SHUF=shuf_choice,
        GATED_025=g025_choice,
    )
    outer = preds_fn("BLEND", seed, tuple(sorted((a, b))))
    b_rows = [r for r in rows if mapping[seed][r["context_sha256"]] == b]
    b_native = np.array(
        [
            h139.native(
                stat.blend(
                    base[r["context_sha256"]],
                    cal["BLEND"][0] - cal["BLEND"][1] * outer[r["context_sha256"]],
                    0.25,
                ),
                pools[r["context_sha256"]]["fitness"],
                pools[r["context_sha256"]]["gate"],
            )
            for r in b_rows
        ]
    )
    return dict(
        seed=seed,
        outer=k,
        A=a,
        B=b,
        FIT=list(fit),
        calibration=cal,
        gate_choice=gate_choice,
        fit_contexts=len(fit_ids),
        b_panel_blend025_native_sha256=hashlib.sha256(b_native.tobytes()).hexdigest(),
        fit_prediction_sha256={
            f"{t}-{''.join(map(str, ex))}": fits[(t, seed, ex)]["prediction_sha256"]
            for t in tags
            for ex in [tuple(sorted((a, b)))]
            + [stat.inner_model_for(k, j) for j in fit]
        },
    )


def run(root, launch_path, resume=False, smoke=False, after_smoke=False):
    admit_host()  # before any import, read, root creation or claim
    if deadline() is None:
        raise RuntimeError(
            "H218 runs only under its monitor (absolute deadline missing)"
        )
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    root = Path(root)
    if root.parent != DATA or not root.name.startswith("run-"):
        raise RuntimeError("H218 requires a run-* root under ~/jope-data/h218")
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
    elif after_smoke:
        if (
            not root.exists()
            or not (root / "smoke_cell_0.json").exists()
            or (root / "metrics.json").exists()
        ):
            raise RuntimeError(
                "after_smoke requires an existing root with smoke_cell_0.json"
            )
        save(
            root / "after-smoke-identity.json",
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
    id_name = (
        "execution-identity-resume.json"
        if resume
        else ("execution-identity-smoke.json" if smoke else "execution-identity.json")
    )
    save(
        root / id_name,
        dict(
            launch=launch,
            bound_sha256=bound,
            launch_guard=launch_guard,
            prereg_sha256=PREREG_SHA,
            host=socket.gethostname(),
            smoke=smoke,
            after_smoke=after_smoke,
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
    vroot = root / (
        "regression-resume"
        if resume
        else ("regression-after-smoke" if after_smoke else "regression")
    )
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
    troot = root / (
        "train-resume" if resume else ("train-after-smoke" if after_smoke else "train")
    )
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

    def preds(tag, seed, ex):
        with np.load(
            root / f"{tag}-seed{seed}-exclude{''.join(map(str, ex))}.npz",
            allow_pickle=False,
        ) as z:
            return {k: z[k].astype(np.float64) for k in z.files}

    family_of = {r["context_sha256"]: r["family_id"] for r in rows}

    # Smoke mode: 1 cell (seed=218101, outer=0) non-scored (4 fits + Stage-1 FIT selection only; no A or DL-Val arm scoring).
    if smoke:
        s0 = stat.SEEDS[0]
        a0, b0, fit0 = stat.cell(0)
        smoke_exclusions = [tuple(sorted((a0, b0)))] + [
            stat.inner_model_for(0, j) for j in fit0
        ]
        smoke_fits = {}
        for ex in smoke_exclusions:
            guards(root)
            receipt = fit_gbm(
                h124,
                "BLEND",
                s0,
                ex,
                rows,
                mapping[s0],
                context,
                pools,
                samples,
                root,
                val=(val_ctx, val_pools, val)
                if ex == tuple(sorted((a0, b0)))
                else None,
            )
            smoke_fits[("BLEND", s0, ex)] = receipt
        cell0 = compute_cell_stage1(
            s0, 0, rows, mapping, base, pools, family_of, preds, h124, h139, smoke_fits
        )
        smoke_receipt = dict(
            schema="jope.h218.smoke-cell/v1",
            status="SMOKE_PASS_NON_SCORED",
            seed=s0,
            outer=0,
            fits=len(smoke_fits),
            fit_wall_seconds=sum(f["wall_seconds"] for f in smoke_fits.values()),
            wall_seconds=time.monotonic() - t0,
            dope_before=dope["before"],
            dope_after=dope["after"],
            r882_dl_val_mean=float(got.mean()),
            cell_stage1={k: v for k, v in cell0.items() if k != "gate_choice"},
            gate_choice_summary={
                arm: {gk: gv for gk, gv in g.items() if gk != "grid"}
                for arm, g in cell0["gate_choice"].items()
            },
            scored_a_contexts=0,
            scored_dl_val_arms=0,
            optimizer_updates=0,
        )
        save(root / "smoke_cell_0.json", smoke_receipt, exclusive=True)
        print("SMOKE PASS", s0, 0, round(smoke_receipt["wall_seconds"], 1), flush=True)
        return smoke_receipt

    # 3. Fits (15 BLEND per seed, 45 in total; no SHUF), checkpointed per fit; DL-Val predicted by the outer models.
    outers, inners = stat.fit_plan()
    fits = {}
    for tag in ("BLEND", "SHUF"):
        for seed in stat.FIT_SEEDS[tag]:
            for ex in outers + inners:
                guards(root)
                receipt = (
                    verified_fit(root, tag, seed, ex)
                    if (resume or after_smoke)
                    else None
                )
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

    # 4. Stage 1: FIT-only calibration and nested (lambda, tau) selection per cell, and B panels; lock fsynced before any A scoring.
    cells = []
    for seed in stat.SEEDS:
        for k in range(5):
            cells.append(
                compute_cell_stage1(
                    seed,
                    k,
                    rows,
                    mapping,
                    base,
                    pools,
                    family_of,
                    preds,
                    h124,
                    h139,
                    fits,
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
    switched_a = {arm: np.zeros((3, len(rows)), bool) for arm in stat.GATES}
    val_cells = []
    for c in cells:
        seed, si = c["seed"], stat.SEEDS.index(c["seed"])
        pair = tuple(sorted((c["A"], c["B"])))
        ob = preds("BLEND", seed, pair)
        ab, bb = c["calibration"]["BLEND"]
        gc = c["gate_choice"]
        a_rows = [r for r in rows if mapping[seed][r["context_sha256"]] == c["A"]]
        lam_shuf = float(gc["GATED_NEST_SHUF"]["lam"])
        raw_zgap_a = [
            stat.zgap(
                base[r["context_sha256"]],
                stat.blend(
                    base[r["context_sha256"]],
                    ab - bb * ob[r["context_sha256"]],
                    lam_shuf,
                ),
            )
            for r in a_rows
        ]
        shuf_zgap_a = stat.permute_zgap(
            raw_zgap_a, stat.stream_seed(stat.SHUF_SEED, "A", seed, c["outer"])
        )
        for r, z_shuf in zip(a_rows, shuf_zgap_a):
            cid = r["context_sha256"]
            costs, opened, signal = arm_costs(
                stat,
                base[cid],
                ab - bb * ob[cid],
                gc,
                zgap_shuf_override=z_shuf,
            )
            for arm, sig_key in stat.GATES.items():
                opened_a[arm][si, index[cid]] = opened[arm]
                switched_a[arm][si, index[cid]] = signal[sig_key] != 0.0
            for arm, cost in costs.items():
                e = h124.endpoints(cost, pools[cid]["u"])
                nat = h139.native(cost, pools[cid]["fitness"], pools[cid]["gate"])
                if nat != e["legacy"]:
                    raise RuntimeError("flywheel native != first-argmin utility")
                u4[arm][si, index[cid]] = e["u4"]
                native[arm][si, index[cid]] = nat
        raw_zgap_val = [
            stat.zgap(
                val_costs[r["context_sha256"]],
                stat.blend(
                    val_costs[r["context_sha256"]],
                    ab - bb * ob[r["context_sha256"]],
                    lam_shuf,
                ),
            )
            for r in val
        ]
        shuf_zgap_val = stat.permute_zgap(
            raw_zgap_val, stat.stream_seed(stat.SHUF_SEED, "VAL", seed, c["outer"])
        )
        val_cells.append(
            (
                gc,
                {r["context_sha256"]: ab - bb * ob[r["context_sha256"]] for r in val},
                {r["context_sha256"]: z for r, z in zip(val, shuf_zgap_val)},
            )
        )
    gate_rates = {
        arm: dict(
            open_rate_per_seed=opened_a[arm].mean(axis=1).tolist(),
            open_rate_among_switching_per_seed=[
                float(opened_a[arm][s][switched_a[arm][s]].mean())
                if switched_a[arm][s].any()
                else 0.0
                for s in range(3)
            ],
            switching_rate_per_seed=switched_a[arm].mean(axis=1).tolist(),
        )
        for arm in stat.GATES
    }

    def panel(values, arm):
        """Seed rows of an arm (all H218 arms have three seeds; R882 is seed-invariant)."""
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

    # 6. DL-Val descriptive & co-primary plateau: each of the 15 cells applies its own calibrated outer prior and
    # FIT-chosen (lambda, tau) to the 91 contexts; per-context utilities are averaged over the 15 cells.
    dl_val = {}
    per_arm = {arm: {"u4": [], "native": []} for arm in stat.ARMS}
    for r in val:
        cid = r["context_sha256"]
        acc = {arm: {"u4": [], "native": []} for arm in stat.ARMS}
        for gc, prior, shuf_val_map in val_cells:
            costs, _, _ = arm_costs(
                stat,
                val_costs[cid],
                prior[cid],
                gc,
                zgap_shuf_override=shuf_val_map[cid],
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
    lambda_counts = {
        arm: stat.lambda_selection_counts(cells, arm=arm) for arm in stat.GATES
    }
    import sklearn

    metrics = dict(
        schema="jope.h218.gated-lambda-nest/v1",
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
        gate_rates_pooled_a=gate_rates,
        lambda_selection_counts=lambda_counts,
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
        continued_after_smoke=after_smoke,
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
            load_limit=LOAD_LIMIT,
            deadline_epoch=deadline(),
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
}


def write_fold_manifest():
    """Outcome-free: TRAIN context/family identities from the pinned selected.json only, plus the overlap with the
    registered H203/H210/H217 partitions."""
    h124 = load(H139_PATH, "h218_h139", H139_SHA).load_h124()
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
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="1-cell non-scored smoke test (seed 218101, outer 0; 4 fits + FIT selection only)",
    )
    ap.add_argument(
        "--after-smoke",
        action="store_true",
        help="continue full run from a verified 1-cell smoke root",
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
        run(
            a.root,
            a.launch,
            resume=a.resume,
            smoke=a.smoke,
            after_smoke=a.after_smoke,
        )


if __name__ == "__main__":
    main()

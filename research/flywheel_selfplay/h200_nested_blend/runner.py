"""One monitored H200 CPU attempt: nested family-grouped lambda choice and rank blend.

Uses pinned H124/H139 baseline, data loader, calibration, HistGBM recipe and native utility helpers. The H200 fit
plan adds one-fold calibrators to the outer and inner cross-fit models. No parameters are updated.
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
DATA = Path.home() / "jope-data/h200"
INPUTS = Path.home() / "jope-data/h195/inputs"
PREREG_SHA = "4ce48078de380462d0dd41a102155f5a358dcdd79b2c812404fdd35994af1e31"
H139_PATH = REPO / "research/flywheel_selfplay/h139_selection_audit/runner.py"
H139_SHA = "7a37821db82c89ddf8e48266383c681b20535ff31b95504e259c74dbe5a9c686"
BOUND_FILES = (
    "research/flywheel_selfplay/h200_nested_blend/PREREG.md",
    "research/flywheel_selfplay/h200_nested_blend/runner.py",
    "research/flywheel_selfplay/h200_nested_blend/analysis.py",
    "research/flywheel_selfplay/h200_nested_blend/monitor.py",
    "research/flywheel_selfplay/h200_nested_blend/manifest.json",
    "research/flywheel_selfplay/h200_nested_blend/test_h200.py",
    "research/flywheel_selfplay/h139_selection_audit/runner.py",
    "research/flywheel_selfplay/h124_gbm_shrink_blend/runner.py",
    "python/src/jope/north_star/dope_freeze_guard.py",
)
HOST_RULES = {"xbabe1": dict(min_free=40_000_000_000, max_used=None)}
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
    """Map a pinned H124 input into the already-staged H195 input tree."""
    base = Path.home() / "jope-data"
    return (
        INPUTS / Path(path).relative_to(base)
        if Path(path).is_relative_to(base)
        else Path(path)
    )


def load_modules():
    """Load pinned H139/H124 bytes against the H195 staging layout on xbabe1.
    Every staged file keeps its pinned SHA-256 (verified by the inherited loaders); only the directory changes."""
    h139 = load(H139_PATH, "h195_h139", H139_SHA)
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
    """xbabe1 only; free >=40 GB, MemAvailable >=16 GiB, load <=85%, nice 19, output <500 MB."""
    host = socket.gethostname().split(".")[0]
    if host not in HOST_RULES or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("H200 runs on xbabe1 at nice 19 only")
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
    """Pinned H124 HistGBM recipe for one registered H200 exclusion set."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    if (
        seed not in stat.SEEDS
        or tag != "GBM"
        or len(excluded) not in (2, 3, 4)
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
    if tree_bytes(root) + sum(value.nbytes for value in predictions.values()) >= SCRATCH_MAX:
        raise RuntimeError("next prediction file could exceed the 500 MB H200 output limit")
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
        shuffled=False,
    )
    save(out.with_suffix(".json"), receipt)
    print("FIT", name, round(receipt["wall_seconds"], 1), flush=True)
    return receipt


def calibration_cost(h124, raw_predictions, context_ids, pools):
    ordered = sorted(context_ids)
    return list(h124.calibration(
        np.concatenate([raw_predictions[c][4:] for c in ordered]),
        np.concatenate([pools[c]["fitness"][4:] for c in ordered]),
        np.concatenate([pools[c]["gate"][4:] for c in ordered]),
    ))


def calibrated(raw, params):
    intercept, slope = params
    return intercept - slope * np.asarray(raw, dtype=np.float64)


def run(root, launch_path):
    # This must precede reading launch, manifest, source, or data.
    require_host(socket.gethostname())
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    root, launch_path = Path(root), Path(launch_path)
    if root != DATA / "attempt-001":
        raise RuntimeError("H200 permits exactly ~/jope-data/h200/attempt-001")
    launch = json.loads(launch_path.read_text())
    bound = bind_source(launch)
    launch_guard = guards(root)
    if root.exists():
        allowed = {"supervisor.stdout", "supervisor.resources.jsonl"}
        if {p.name for p in root.iterdir()} - allowed:
            raise RuntimeError("one fresh attempt only; unexpected existing output")
    else:
        root.mkdir(parents=True)
    t0, cpu0 = time.monotonic(), time.process_time()
    save(root / "execution-identity.json", dict(
        launch=launch, bound_sha256=bound, launch_guard=launch_guard,
        host=socket.gethostname(), prereg_sha256=PREREG_SHA,
        environment={k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "CUDA_VISIBLE_DEVICES"
        )}
    ))

    h139, h124 = load_modules()
    h124.FOLD_SEED, h124.SAMPLE_SEED = stat.FOLD_SEEDS[stat.SEEDS[0]], stat.SAMPLE_SEED
    # Exact baseline regression precedes any TRAIN outcome-pool read or treatment.
    train_rows, val_rows, reference = h139.roster(h124)
    if len(train_rows) != 1219 or len(val_rows) != 91:
        raise RuntimeError("pinned flywheel cohort size mismatch")
    val_ctx, _ = h124.train_context(val_rows + train_rows, [])
    val_ctx = {r["context_sha256"]: val_ctx[r["context_sha256"]] for r in val_rows}
    val_proj = h124.train_projection(val_rows + train_rows, [])
    val_pools = h139.pools_for(val_rows)
    val_root = root / "regression"
    val_root.mkdir()
    val_costs, net = bounded_threads(h124.baseline, val_root, val_rows, val_ctx, val_pools, val_proj)
    baseline = np.asarray([
        h139.native(val_costs[r["context_sha256"]],
                    val_pools[r["context_sha256"]]["fitness"],
                    val_pools[r["context_sha256"]]["gate"])
        for r in val_rows
    ])
    expected = np.asarray(reference["utility"]["per_context"])
    if not np.array_equal(baseline, expected) or float(baseline.mean()) != 1.1377941898164525:
        raise RuntimeError("R882 91/91 native regression failed")
    del net
    h124.release_memory()
    print("BASELINE PASS: 91/91 native r882 exact", flush=True)

    troot = root / "train"
    troot.mkdir()
    selected, context, pools, samples, _, read_receipts = h124.load_data(troot)
    rows = selected["train"]
    if [r["context_sha256"] for r in rows] != [r["context_sha256"] for r in train_rows]:
        raise RuntimeError("H200 roster differs from pinned TRAIN identities")
    if any(r.get("dl_split") != "DL-Train" or r.get("task") != "regression"
           or r.get("has_meta_test") is True for r in rows):
        raise RuntimeError("TRAIN eligibility guard failed")
    projection = h124.train_projection(rows, selected["dev"])
    base, net = bounded_threads(h124.baseline, troot, rows, context, pools, projection)
    del net
    h124.release_memory()
    dope = json.loads((troot / "dope-train.json").read_text())
    if dope["before"] != h124.DOPE or dope["after"] != h124.DOPE:
        raise RuntimeError("DOPE changed")
    families = [r["family_id"] for r in rows]
    family_by_id = {r["context_sha256"]: r["family_id"] for r in rows}
    folds = {s: stat.seed_folds(families, s) for s in stat.SEEDS}
    mapping = {s: {
        r["context_sha256"]: next(k for k, group in enumerate(folds[s]) if r["family_id"] in group)
        for r in rows
    } for s in stat.SEEDS}
    fold_manifest = dict(
        wave="H200", family_count=len(set(families)), context_count=len(rows),
        fold_seeds={str(s): stat.FOLD_SEEDS[s] for s in stat.SEEDS},
        families_by_seed={str(s): group for s, group in folds.items()},
        fold_context_counts_by_seed={str(s): [
            sum(mapping[s][r["context_sha256"]] == k for r in rows) for k in range(5)
        ] for s in stat.SEEDS},
        algorithm="sorted 205 family IDs; default_rng(fold_seed).permutation; numpy.array_split into five",
    )
    save(root / "fold_manifest.json", fold_manifest, exclusive=True)

    outers, inners, calibrators = stat.fit_plan()
    fit_receipts = {}
    for seed in stat.SEEDS:
        for excluded in outers + inners + calibrators:
            guards(root)
            receipt = fit_gbm(
                h124, "GBM", seed, excluded, rows, mapping[seed], context, pools,
                samples, root
            )
            fit_receipts[(seed, excluded)] = receipt
    if len(fit_receipts) != 60:
        raise RuntimeError("H200 requires exactly 60 fit receipts")

    def preds(seed, excluded):
        path = root / f"GBM-seed{seed}-exclude{''.join(map(str, excluded))}.npz"
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].astype(np.float64) for k in z.files}

    # FIT-only nested choice and final outer calibration. No A/B treatment score is computed here.
    stage1 = []
    for seed in stat.SEEDS:
        for k in range(5):
            a, b, fit_folds = stat.cell(k)
            all_inner_raw = {}
            for j in fit_folds:
                p = preds(seed, stat.inner_model_for(k, j))
                all_inner_raw.update({c: p[c] for c in p if mapping[seed][c] == j})
            fit_ids = set(all_inner_raw)
            final_cal = calibration_cost(h124, all_inner_raw, fit_ids, pools)
            grid_scores = {arm: {lam: {} for lam in stat.LAMBDA_GRID}
                           for arm in ("NEST_L", "NEST_SHUF")}
            for j in fit_folds:
                held = preds(seed, stat.inner_model_for(k, j))
                u, v = [q for q in fit_folds if q != j]
                crossfit = {}
                for train_fold, held_fold in ((u, v), (v, u)):
                    cp = preds(seed, stat.calibrator_model_for(train_fold))
                    crossfit.update({c: cp[c] for c in cp if mapping[seed][c] == held_fold})
                inner_cal = calibration_cost(h124, crossfit, set(crossfit), pools)
                held_ids = [r["context_sha256"] for r in rows if mapping[seed][r["context_sha256"]] == j]
                for cid in held_ids:
                    gbm_cost = calibrated(held[cid], inner_cal)
                    shuffled_cost = stat.permute_scores(
                        gbm_cost, stat.stream_seed(stat.SHUF_SEED, seed, "inner", k, cid)
                    )
                    ref = base[cid]
                    utility0 = h139.native(ref, pools[cid]["fitness"], pools[cid]["gate"])
                    for lam in stat.LAMBDA_GRID:
                        c_l = stat.blend(ref, gbm_cost, lam)
                        c_s = stat.blend(ref, shuffled_cost, lam)
                        grid_scores["NEST_L"][lam][cid] = (
                            h139.native(c_l, pools[cid]["fitness"], pools[cid]["gate"]) - utility0
                        )
                        grid_scores["NEST_SHUF"][lam][cid] = (
                            h139.native(c_s, pools[cid]["fitness"], pools[cid]["gate"]) - utility0
                        )
            gains = {}
            chosen = {}
            for arm in ("NEST_L", "NEST_SHUF"):
                gains[arm] = {}
                chosen[arm] = {}
                for lam, values in grid_scores[arm].items():
                    ids = sorted(values)
                    _, family_values = stat.family_matrix(
                        np.asarray([values[c] for c in ids]),
                        [family_by_id[c] for c in ids]
                    )
                    gains[arm][lam] = float(family_values.mean())
                chosen[arm] = stat.choose_lambda(gains[arm])
            stage1.append(dict(seed=seed, outer=k, A=a, B=b, FIT=list(fit_folds),
                               calibration=final_cal, lambda_mean_gain=gains,
                               selected_lambda=chosen,
                               fit_prediction_sha256={
                                   ''.join(map(str, ex)): fit_receipts[(seed, ex)]["prediction_sha256"]
                                   for ex in [tuple(sorted((a, b)))]
                                   + [stat.inner_model_for(k, j) for j in fit_folds]
                                   + [stat.calibrator_model_for(j) for j in range(5)]
                               }))
    lock_path = root / "stage1_lock.json"
    save(lock_path, dict(written_before_any_A_treatment_scoring=True, cells=stage1), exclusive=True)
    lock_sha = sha(lock_path)
    print("LOCK", lock_sha[:12], flush=True)

    # Score pooled A only after the FIT-only lambda/calibration lock is durable.
    index = {r["context_sha256"]: i for i, r in enumerate(rows)}
    panel = {arm: np.full((len(stat.SEEDS), len(rows)), np.nan) for arm in stat.ARMS}
    lambdas = {}
    for c in stage1:
        seed, si, k = c["seed"], stat.SEEDS.index(c["seed"]), c["outer"]
        outer_ex = tuple(sorted((c["A"], c["B"])))
        outer_raw = preds(seed, outer_ex)
        lambdas[f"{seed}/{k}"] = c["selected_lambda"]
        for r in rows:
            cid = r["context_sha256"]
            if mapping[seed][cid] != c["A"]:
                continue
            ref, p = base[cid], outer_raw[cid]
            gbm_cost = calibrated(p, c["calibration"])
            shuffled = stat.permute_scores(
                gbm_cost, stat.stream_seed(stat.SHUF_SEED, seed, "outer", k, cid)
            )
            values = {
                "R882": ref,
                "FIX025": stat.blend(ref, gbm_cost, 0.25),
                "NEST_L": stat.blend(ref, gbm_cost, c["selected_lambda"]["NEST_L"]),
                "RANK025": stat.rank_blend(ref, -p),
                "NEST_SHUF": stat.blend(ref, shuffled, c["selected_lambda"]["NEST_SHUF"]),
            }
            for arm, cost in values.items():
                panel[arm][si, index[cid]] = h139.native(
                    cost, pools[cid]["fitness"], pools[cid]["gate"]
                )
    if any(not np.isfinite(v).all() for v in panel.values()):
        raise RuntimeError("pooled A coverage incomplete")
    if not all(np.array_equal(panel["R882"][s], panel["R882"][0]) for s in range(3)):
        raise RuntimeError("R882 utility must be seed-invariant")
    contrasts = {arm: stat.contrast(panel[arm], panel["R882"], families)
                 for arm in stat.ARMS[1:]}
    nestl_vs_fix025 = stat.contrast(panel["NEST_L"], panel["FIX025"], families)
    adjusted = stat.holm({a: contrasts[a]["p_one_sided_vs_margin"] for a in ("NEST_L", "RANK025")})
    elo = {arm: stat.elo_vs_reference(panel[arm], panel["R882"], families)
           for arm in stat.ARMS[1:]}
    nestl_vs_fix025_elo = stat.elo_vs_reference(panel["NEST_L"], panel["FIX025"], families)
    decision = stat.verdict(contrasts["NEST_L"], adjusted["NEST_L"])
    import sklearn
    metrics = dict(
        schema="jope.h200.nested-blend/v1", status="COMPLETE",
        verdict=decision["verdict"], decision=decision,
        primary_endpoint="H139 native first-argmin utility (full candidate pool)",
        contrasts_native=contrasts, holm_adjusted_p=adjusted, elo_vs_r882=elo,
        contrast_nestl_vs_fix025=nestl_vs_fix025,
        elo_nestl_vs_fix025=nestl_vs_fix025_elo,
        selected_lambdas=lambdas, stage1_lock_sha256=lock_sha,
        fits=len(fit_receipts), fit_wall_seconds=sum(f["wall_seconds"] for f in fit_receipts.values()),
        read_boundary=dict(train_contexts=len(rows), train_families=len(set(families)),
                           dl_val_contexts_for_baseline_only=len(val_rows),
                           sealed=0, holdout=0, meta_test=0, live_loop=0,
                           train_cache_dl_val_rows_decoded=sum(
                               v["overlapping_reads"] for v in read_receipts.values()
                           )),
        dope_before=dope["before"], dope_after=dope["after"], bound_sha256=bound,
        runtime=dict(wall_seconds=time.monotonic()-t0, process_cpu_seconds=time.process_time()-cpu0,
                     max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     scratch_bytes=0),
        versions=dict(numpy=np.__version__, scikit_learn=sklearn.__version__,
                      torch=torch.__version__, python=platform.python_version()),
        optimizer_updates=0, GPU_used=False, CUDA_initialized=torch.cuda.is_initialized(),
        serving_promotion=False
    )
    np.savez_compressed(root / "panels.npz",
                        **{f"native_{a}": v for a, v in panel.items()},
                        context_ids=np.array([r["context_sha256"] for r in rows]),
                        family_ids=np.array(families))
    metrics["runtime"]["scratch_bytes"] = tree_bytes(root)
    raw = json.dumps(metrics, indent=1, sort_keys=True, allow_nan=False) + "\n"
    if len(raw.encode()) >= METRICS_LIMIT:
        raise RuntimeError("metrics exceed 100 KB")
    (root / "metrics.json").write_text(raw)
    if tree_bytes(root) >= SCRATCH_MAX:
        raise RuntimeError("H200 output reached 500 MB limit")
    print("COMPLETE", metrics["verdict"], flush=True)
    return metrics


def require_host(host):
    if host.split(".")[0] != "xbabe1":
        raise RuntimeError("H200 host guard rejects all hosts except xbabe1")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--launch", type=Path)
    args = ap.parse_args()
    require_host(socket.gethostname())
    if args.root is None or args.launch is None:
        ap.error("--root and --launch required")
    run(args.root, args.launch)


if __name__ == "__main__":
    main()

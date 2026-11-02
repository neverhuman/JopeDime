"""H224 immutable-source CPU run; nested FIT gates lock before A/DL-Val scoring."""

import argparse
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
import numpy as np

REPO = Path("/home/ubuntu/JopeDime")
PREFIX = "research/flywheel_selfplay/h224_learned_switch_gate/"


def blob(commit, path):
    return subprocess.check_output(
        ["git", "-C", str(REPO), "show", commit + ":" + path]
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(commit, path, name, expected):
    raw = blob(commit, path)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError("source identity mismatch: " + path)
    m = types.ModuleType(name)
    m.__file__ = str(REPO / path)
    sys.modules[name] = m
    exec(compile(raw, m.__file__, "exec"), m.__dict__)
    return m


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=1, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def state_hash(net):
    import torch

    h = hashlib.sha256()
    for n, t in sorted(net.state_dict().items()):
        h.update((n + str(t.dtype) + str(tuple(t.shape))).encode())
        h.update(
            t.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return h.hexdigest()


class Store:
    def __init__(self, root, prefix):
        import boto3

        self.root, self.prefix, self.receipts = root, prefix, {}
        self.client = boto3.Session(profile_name="veox-jepson-prod").client(
            "s3", region_name="us-east-2"
        )

    def put(self, path):
        raw = path.read_bytes()
        name = str(path.relative_to(self.root))
        digest = hashlib.sha256(raw).hexdigest()
        key = self.prefix + name
        bucket = "veox-jopedime-learning-use2"
        version = self.client.put_object(
            Bucket=bucket,
            Key=key,
            Body=raw,
            IfNoneMatch="*",
            Metadata={"sha256": digest},
        )["VersionId"]
        response = self.client.get_object(Bucket=bucket, Key=key, VersionId=version)
        got = response["Body"].read()
        response["Body"].close()
        if (
            response["VersionId"] != version
            or hashlib.sha256(got).hexdigest() != digest
        ):
            raise RuntimeError("S3 recovery mismatch")
        receipt = dict(
            s3_uri="s3://" + bucket + "/" + key,
            version_id=version,
            sha256=digest,
            bytes=len(raw),
        )
        save(self.root / ("receipt-" + name.replace("/", "_") + ".json"), receipt)
        self.receipts[name] = receipt
        return receipt


def score_vector(h139, cost, pool):
    return h139.native(cost, pool["fitness"], pool["gate"])


def fit_cell(stat, gate, h124, h139, seed, outer, rows, mapping, base, pools, preds):
    a, b, fit_groups = stat.cell(outer)
    inner = {}
    for j in fit_groups:
        prediction = preds(seed, stat.inner_model_for(outer, j))
        inner.update(
            {cid: value for cid, value in prediction.items() if mapping[cid] == j}
        )
    ids = sorted(inner)
    byid = {r["context_sha256"]: r for r in rows}
    if any(mapping[cid] not in fit_groups for cid in ids):
        raise RuntimeError("FIT prediction closure")
    cal = list(
        h124.calibration(
            np.concatenate([inner[c][4:] for c in ids]),
            np.concatenate([pools[c]["fitness"][4:] for c in ids]),
            np.concatenate([pools[c]["gate"][4:] for c in ids]),
        )
    )
    x, switching, refs, blends, families, groups, signals = [], [], [], [], [], [], []
    for cid in ids:
        prior = cal[0] - cal[1] * inner[cid]
        blended = stat.blend(base[cid], prior)
        feature, switch = gate.features(base[cid], prior, blended, stat.zgap)
        x.append(feature)
        switching.append(switch)
        signals.append(stat.zgap(base[cid], blended))
        refs.append(score_vector(h139, base[cid], pools[cid]))
        blends.append(score_vector(h139, blended, pools[cid]))
        families.append(byid[cid]["family_id"])
        groups.append(mapping[cid])
    delta = np.asarray(blends) - refs
    choices = {"GATED_025": stat.choose_tau(signals, refs, blends, families)}
    for arm, shuffled in [("LGATE", False), ("LGATE_SHUF", True)]:
        choices[arm] = gate.train(
            x,
            delta,
            families,
            groups,
            switching,
            stat.stream_seed(224160, seed, outer),
            shuffled=shuffled,
        )
    return dict(
        seed=seed,
        outer=outer,
        A=a,
        B=b,
        FIT=list(fit_groups),
        calibration=cal,
        choices=choices,
        fit_contexts=len(ids),
        fit_context_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        fit_family_count=len(set(families)),
    )


def costs_for(stat, gate, base, prior, cell):
    blended = stat.blend(base, prior)
    x, switch = gate.features(base, prior, blended, stat.zgap)
    choices = cell["choices"]
    result = {"R882": base}
    result["GATED_025"] = (
        blended
        if stat.gated(stat.zgap(base, blended), choices["GATED_025"]["tau"])
        else base
    )
    for arm in ("LGATE", "LGATE_SHUF"):
        result[arm] = (
            blended if gate.opens(choices[arm], x[None], [switch])[0] else base
        )
    return result


def run(args):
    import torch
    import psutil

    pins = json.loads(blob(args.commit, PREFIX + "manifest.json"))
    for name, expected in pins["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError("package version mismatch: " + name)
    claim = json.loads((args.root.parent / "attempt1-claim.json").read_text())
    if claim["source_commit"] != args.commit or args.root.exists():
        raise RuntimeError("fresh exact-source attempt required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if (
        torch.cuda.is_initialized()
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or len(os.sched_getaffinity(0)) > 4
    ):
        raise RuntimeError("CPU-only four-core boundary")
    args.root.mkdir()
    started = time.monotonic()
    stage = "setup"
    fits = {}
    cells = []
    monitor = module(
        args.commit,
        PREFIX + "monitor.py",
        "h224_monitor_runtime",
        pins["source_sha256"]["monitor.py"],
    )

    def check():
        if time.time() >= claim["deadline_epoch"]:
            raise RuntimeError("absolute attempt deadline")
        monitor.resources(psutil.Process(), args.root)

    stat = module(
        args.commit,
        PREFIX + "analysis.py",
        "analysis",
        pins["source_sha256"]["analysis.py"],
    )
    gate = module(
        args.commit, PREFIX + "gate.py", "h224_gate", pins["source_sha256"]["gate.py"]
    )
    parent = pins["parent_commit"]

    def load_parent(path, name):
        return module(parent, path, name, pins["parent_sources"][path])

    helper = load_parent(
        "research/flywheel_selfplay/h217_gated025_confirm/runner.py", "h224_h217"
    )
    h139 = load_parent(
        "research/flywheel_selfplay/h139_selection_audit/runner.py", "h224_h139"
    )
    h124 = load_parent(
        "research/flywheel_selfplay/h124_gbm_shrink_blend/runner.py", "h224_h124"
    )
    h124.REPO = REPO
    h124.RSS_LIMIT_KIB = 4 * 1024 * 1024

    def staged(path):
        if Path(path).is_relative_to("/home/ubuntu/jope-data"):
            return Path(pins["staging"]) / Path(path).relative_to(
                "/home/ubuntu/jope-data"
            )
        return Path(path)

    h124.INPUT, h124.CACHE, h124.R882 = map(staged, (h124.INPUT, h124.CACHE, h124.R882))
    h124.FILES = {staged(k): v for k, v in h124.FILES.items()}
    h139.INPUT = h124.INPUT
    h139.ARM_DIR = Path(pins["staging"])
    store = Store(args.root, pins["s3_prefix"] + "attempt1/")
    save(
        args.root / "launch.json",
        dict(
            source_commit=args.commit,
            claim=claim,
            manifest_sha256=hashlib.sha256(
                blob(args.commit, PREFIX + "manifest.json")
            ).hexdigest(),
            optimizer_updates=0,
            CUDA_initialized=False,
        ),
    )
    store.put(args.root / "launch.json")
    try:
        check()
        for path, expected in h124.FILES.items():
            if h124.sha(path) != expected:
                raise RuntimeError("input pin mismatch: " + str(path))
        provenance = json.loads(blob(args.commit, PREFIX + "provenance.json"))
        packet = json.loads((h124.INPUT / "selected.json").read_text())
        if provenance["zero_synthetic"] is not True or set(
            provenance["dataset_ids"]
        ) != {r["dataset_id"] for r in packet["train"] + packet["dev"]}:
            raise RuntimeError("real-S3 provenance closure")
        train_rows, val, reference = h139.roster(h124)
        val_ctx, _ = h124.train_context(val + train_rows, [])
        val_ctx = {r["context_sha256"]: val_ctx[r["context_sha256"]] for r in val}
        val_proj = h124.train_projection(val + train_rows, [])
        val_pools = h139.pools_for(val)
        vroot = args.root / "regression"
        vroot.mkdir()
        val_base, val_net = helper.bounded_threads(
            h124.baseline, vroot, val, val_ctx, val_pools, val_proj
        )
        got = np.array(
            [
                score_vector(
                    h139, val_base[r["context_sha256"]], val_pools[r["context_sha256"]]
                )
                for r in val
            ]
        )
        if (
            not np.array_equal(got, np.asarray(reference["utility"]["per_context"]))
            or float(got.mean()) != 1.1377941898164525
        ):
            raise RuntimeError("native91 bit-exact regression")
        del val_net
        h124.release_memory()
        save(
            args.root / "baseline.json",
            dict(
                bit_exact=True,
                mean=float(got.mean()),
                utilities=got.tolist(),
                before_any_fit=True,
            ),
        )
        store.put(args.root / "baseline.json")
        print("BASELINE PASS 91/91", flush=True)
        h124.FOLD_SEED, h124.SAMPLE_SEED = 224141, 224140
        troot = args.root / "train"
        troot.mkdir()
        selected, context, pools, samples, _, reads = h124.load_data(troot)
        rows = selected["train"]
        if rows != train_rows:
            raise RuntimeError("TRAIN roster mismatch")
        projection = h124.train_projection(rows, selected["dev"])
        base, net = helper.bounded_threads(
            h124.baseline, troot, rows, context, pools, projection
        )
        frozen = state_hash(net)
        fams = [r["family_id"] for r in rows]
        folds = {s: stat.seed_folds(fams, s) for s in stat.SEEDS}
        registered = json.loads(blob(args.commit, PREFIX + "fold_manifest.json"))
        if registered["families_by_seed"] != {str(s): g for s, g in folds.items()}:
            raise RuntimeError("fold identity")
        mapping = {
            s: {
                r["context_sha256"]: next(
                    k for k, g in enumerate(folds[s]) if r["family_id"] in g
                )
                for r in rows
            }
            for s in stat.SEEDS
        }
        fitroot = args.root / "fits"
        fitroot.mkdir()

        def preds(seed, excluded):
            path = (
                fitroot / f"BLEND-seed{seed}-exclude{''.join(map(str, excluded))}.npz"
            )
            if sha(path) != fits[(seed, excluded)]["prediction_sha256"]:
                raise RuntimeError("prediction artifact changed")
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].astype(np.float64) for k in z.files}

        outers, inners = stat.fit_plan()
        first = stat.SEEDS[0]
        smoke = [(first, tuple(sorted(stat.cell(0)[:2])))] + [
            (first, stat.inner_model_for(0, j)) for j in stat.cell(0)[2]
        ]
        plan = smoke + [
            (s, e) for s in stat.SEEDS for e in outers + inners if (s, e) not in smoke
        ]
        if len(plan) != 45 or len(set(plan)) != 45:
            raise RuntimeError("unique fit inventory")
        for number, (seed, excluded) in enumerate(plan, 1):
            stage = "fits"
            check()
            receipt = helper.fit_gbm(
                h124,
                "BLEND",
                seed,
                excluded,
                rows,
                mapping[seed],
                context,
                pools,
                samples,
                fitroot,
                val=(val_ctx, val_pools, val) if excluded in outers else None,
            )
            fits[(seed, excluded)] = receipt
            file = (
                fitroot / f"BLEND-seed{seed}-exclude{''.join(map(str, excluded))}.npz"
            )
            store.put(file)
            store.put(file.with_suffix(".json"))
            if number == 4:
                smoke_cell = fit_cell(
                    stat,
                    gate,
                    h124,
                    h139,
                    first,
                    0,
                    rows,
                    mapping[first],
                    base,
                    pools,
                    preds,
                )
                save(
                    args.root / "smoke.json",
                    helper.jsonable(
                        dict(
                            status="PASS_NON_SCORED",
                            fits=4,
                            A_scored=0,
                            DL_Val_treatment_scored=0,
                            cell=smoke_cell,
                            reuse="Exact four immutable fits reused in full45; no extra fits or budget reset",
                        )
                    ),
                )
                store.put(args.root / "smoke.json")
                print("SMOKE PASS 4 FITS; continuing fixed full plan", flush=True)
        stage = "lock"
        for seed in stat.SEEDS:
            for outer in range(5):
                check()
                cells.append(
                    fit_cell(
                        stat,
                        gate,
                        h124,
                        h139,
                        seed,
                        outer,
                        rows,
                        mapping[seed],
                        base,
                        pools,
                        preds,
                    )
                )
        lock = args.root / "stage1_lock.json"
        save(
            lock,
            helper.jsonable(
                dict(
                    source_commit=args.commit,
                    cells=cells,
                    fit_receipts=list(fits.values()),
                    A_scored=0,
                    DL_Val_treatment_scored=0,
                )
            ),
        )
        store.put(lock)
        locksha = sha(lock)
        print("LOCK", locksha, flush=True)
        stage = "score"
        index = {r["context_sha256"]: i for i, r in enumerate(rows)}
        panels = {a: np.full((3, len(rows)), np.nan) for a in stat.ARMS}
        val_panels = {a: np.full((3, 5, len(val)), np.nan) for a in stat.ARMS}
        adoption = {a: [0, 0, 0] for a in stat.ARMS[1:]}
        for cell in cells:
            check()
            if sha(lock) != locksha:
                raise RuntimeError("stage1 lock changed")
            seed, k = cell["seed"], cell["outer"]
            si = stat.SEEDS.index(seed)
            outer = preds(seed, tuple(sorted((cell["A"], cell["B"]))))
            ca, cb = cell["calibration"]
            for row in rows:
                cid = row["context_sha256"]
                if mapping[seed][cid] != cell["A"]:
                    continue
                costs = costs_for(stat, gate, base[cid], ca - cb * outer[cid], cell)
                for arm, cost in costs.items():
                    panels[arm][si, index[cid]] = score_vector(h139, cost, pools[cid])
                    if arm != "R882":
                        adoption[arm][si] += int(
                            np.argmin(cost) != np.argmin(base[cid])
                        )
            for vi, row in enumerate(val):
                cid = row["context_sha256"]
                costs = costs_for(stat, gate, val_base[cid], ca - cb * outer[cid], cell)
                for arm, cost in costs.items():
                    val_panels[arm][si, k, vi] = score_vector(
                        h139, cost, val_pools[cid]
                    )
        if any(
            not np.isfinite(x).all()
            for x in list(panels.values()) + list(val_panels.values())
        ):
            raise RuntimeError("incomplete scored coverage")
        contrasts = {
            a: stat.contrast(panels[a], panels["R882"], fams) for a in stat.ARMS[1:]
        }
        contrasts["LGATE_vs_GATED_025"] = stat.contrast(
            panels["LGATE"], panels["GATED_025"], fams
        )
        vf = [r["family_id"] for r in val]
        plateau = {
            a: stat.plateau(val_panels[a].mean((0, 1)), got, vf) for a in stat.ARMS[1:]
        }
        decision = stat.verdict(contrasts["LGATE"], plateau["LGATE"])
        guard = h124.source_module(
            "python/src/jope/north_star/dope_freeze_guard.py", "h224_final_guard"
        )
        dope_after = guard.snapshot_dope_weights(net).composite_sha256
        if (
            state_hash(net) != frozen
            or dope_after != h124.DOPE
            or torch.cuda.is_initialized()
        ):
            raise RuntimeError("frozen/CPU boundary")
        for path, expected in h124.FILES.items():
            if h124.sha(path) != expected:
                raise RuntimeError("postrun input drift")
        np.savez_compressed(
            args.root / "panels.npz",
            **{"A_" + k: v for k, v in panels.items()},
            **{"DLVal_" + k: v for k, v in val_panels.items()},
            context_ids=np.array([r["context_sha256"] for r in rows]),
            family_ids=np.array(fams),
        )
        store.put(args.root / "panels.npz")
        result = dict(
            wave="H224",
            status="COMPLETE",
            verdict=decision["verdict"],
            decision=decision,
            source_commit=args.commit,
            contrasts_native=contrasts,
            dl_val_plateau=plateau,
            dl_val_per_seed={
                a: [stat.plateau(val_panels[a][si].mean(0), got, vf) for si in range(3)]
                for a in stat.ARMS[1:]
            },
            elo_native_vs_r882={
                a: stat.elo_vs_reference(panels[a], panels["R882"], fams)
                for a in stat.ARMS[1:]
            },
            adoption_counts=adoption,
            optimizer_updates=0,
            fits=len(fits),
            logical_fit_uses=60,
            stage1_lock_sha256=locksha,
            baseline91_bitexact=True,
            native_mean=float(got.mean()),
            dope_before=h124.DOPE,
            dope_after=dope_after,
            frozen_state_hash=frozen,
            CUDA_initialized=False,
            read_boundary=dict(
                sealed=0,
                holdout=0,
                meta_test=0,
                train_cache_dl_val_rows_decoded=sum(
                    v["overlapping_reads"] for v in reads.values()
                ),
            ),
            elapsed_seconds=time.monotonic() - started,
            U4=None,
            U4_reason="Not measured in H224",
            serving_promotion=False,
        )
        check()
        save(args.root / "metrics.json", result)
        store.put(args.root / "metrics.json")
        check()
        print("COMPLETE", result["verdict"], flush=True)
    except BaseException as exc:
        failure = dict(
            status="INCOMPLETE",
            verdict=None,
            error=repr(exc),
            stage=stage,
            durable_fits=len(fits),
            source_commit=args.commit,
            elapsed_seconds=time.monotonic() - started,
        )
        save(args.root / "failure.json", failure)
        try:
            store.put(args.root / "failure.json")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--root", type=Path, required=True)
    run(ap.parse_args())

"""H284 runner:
1. --smoke: Native91 (91/91) + bit-exact reproduction of H279 cell KM276140-S279101-C0.
2. --mode export: before any fresh pool exists or is read, fit H279 BAG_SWITCH (q=6), BAG_SWITCH_NESTED (q* in {5,6,7}),
   and paired RERANK8_VETO proposer on all 205 DL-Train families (1,219 contexts) for seeds 279101-279103, verify 64-context
   reload bit-exactness of all 30 exported LightGBM models, and lock fit_lock.json.
3. --mode transfer: apply frozen H284 exports (279101-279103) and frozen H278 RERANK8_VETO exports (278101-278103,
   fetched by S3 VersionId) unchanged to H282's genuinely distinct r782/r785 GPU pools (descriptive) and to fresh pools
   from H283 (EVEN half) and/or H285 (ODD half) and their union."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import analysis_h279 as stat279
import analysis_h284 as stat
import bagged
import freeze_bagged as fb
import h265_core as core
import topk

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
H276_INPUTS = Path("/home/ubuntu/jope-data/h276/inputs")
H284_DATA = Path("/home/ubuntu/jope-data/h284")
H282_DATA = Path("/home/ubuntu/jope-data/h282")
POOLS = Path("/home/ubuntu/jope-data/h50/input")
CACHE = Path("/home/ubuntu/jope-data/h58/cache")
NATIVE91 = 1.1377941898164525
TOP4_CTX, TOP4_FAM = 1.2403601682250796, 1.2519997168684853
H279_REGRESSION_CELL = "KM276140-S279101-C0"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    return fb.save_json(Path(path), obj)


def audit_hook(event, args):
    """Refuse any open of a sealed, holdout or meta-test path."""
    if event == "open" and isinstance(args[0], (str, bytes)):
        s = os.fsdecode(args[0]).lower()
        if any(
            x in s
            for x in (
                "/sealed",
                "/holdout",
                "meta-test",
                "meta_test",
                "qualification.lock",
            )
        ):
            raise RuntimeError("protected path: " + s)


def dope_of(path):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    value = str(ckpt.get("dope_sha256", ""))
    del ckpt
    return value


def verify(pins):
    for path, digest in pins["inputs"].items():
        if sha(path) != digest:
            raise RuntimeError("input pin mismatch: " + path)
    for path, digest in pins["parent_sources"].items():
        if sha(REPO / path) != digest:
            raise RuntimeError("parent pin mismatch: " + path)
    for path, digest in pins["source_sha256"].items():
        if sha(HERE / path) != digest:
            raise RuntimeError("source pin mismatch: " + path)
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package mismatch: " + name)


def native91(val, costs, ref):
    ref_map = {
        c["cid"]: float(u)
        for c, u in zip(
            ref["evaluation_contract"]["contexts"], ref["utility"]["per_context"]
        )
    }
    single, top4, exact = [], [], 0
    for r in val:
        cid = r["context_sha256"]
        with np.load(POOLS / "pools" / f"{cid}.npz", allow_pickle=False) as z:
            fit, gate = z["fitness"].astype(np.float64), z["gate"].astype(bool)
        u = np.where(gate, -fit, -2.0)
        rc = costs[cid]
        u0 = float(u[int(np.argmin(rc))])
        single.append(u0)
        exact += u0 == ref_map[cid]
        t4 = np.argsort(rc, kind="mergesort")[:4]
        top4.append(float(u[int(t4[int(np.argmin(np.where(gate[t4], fit[t4], 2.0)))])]))
    fams = np.array([r["family_id"] for r in val])
    t4a = np.asarray(top4)
    fam_top4 = float(np.mean([t4a[fams == f].mean() for f in sorted(set(fams))]))
    if exact != 91 or float(np.mean(single)) != NATIVE91:
        raise RuntimeError("Native91 regression")
    if abs(float(np.mean(top4)) - TOP4_CTX) > 1e-12 or abs(fam_top4 - TOP4_FAM) > 1e-12:
        raise RuntimeError("TOP4_PICK@r882 regression")
    return dict(
        native91=float(np.mean(single)),
        matches=exact,
        top4_ctx=float(np.mean(top4)),
        top4_fam=fam_top4,
    )


def priors_for(rows):
    ids = {r["context_sha256"] for r in rows}
    aprior, proj16, ctxf = {}, {}, {}
    with np.load(CACHE / "pool_priors.npz", allow_pickle=False) as zp:
        cids, offsets = [str(c) for c in zp["cids"]], zp["offsets"]
        base, proj = zp["base_prior_cost"], zp["proj16"]
        for i, cid in enumerate(cids):
            if cid in ids:
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                aprior[cid] = base[lo:hi].astype(np.float32).copy()
                proj16[cid] = proj[lo:hi].astype(np.float32).copy()
    with np.load(CACHE / "context_features.npz", allow_pickle=False) as zc:
        for i, cid in enumerate(str(c) for c in zc["cids"]):
            if cid in ids:
                ctxf[cid] = {
                    "lm48_z": zc["lm48_z"][i].astype(np.float32).copy(),
                    "dope629": zc["dope629"][i].astype(np.float32).copy(),
                }
    return aprior, proj16, ctxf


def h279_cell_regression(train, P8, fam_by_cid):
    """Re-run H279 cell KM276140-S279101-C0 and verify bit-exact equality against H279 attempt-001."""
    ref_path = H284_DATA / "inputs" / f"h279-{H279_REGRESSION_CELL}.json"
    ref = json.loads(ref_path.read_text())
    cm = json.loads((HERE / "fold_manifest.json").read_text())
    parts = stat279.partitions_from_manifest(cm)
    clusters = parts["KM276140"]
    cbf = stat279.cluster_out_folds(train, clusters, 0, 279101)
    assess = cbf[0]
    got = bagged.run_cell_bagged(
        279101, 0, cbf, P8, fam_by_cid, 8, salt="KM276140-0", store_models=False
    )
    inv = {
        a: int(sum(not P8[c]["gate_k"][j] for c, j in zip(assess, got["assess_picks"][a])))
        for a in stat279.ALL_ARMS
    }
    chg = {
        a: int(sum(j != 0 for j in got["assess_picks"][a]))
        for a in stat279.ALL_ARMS
    }
    ok = (
        json.loads(json.dumps(got["receipts"])) == ref["receipts"]
        and inv == ref["invalid"]
        and chg == ref["changed"]
        and got["assess_picks"] == ref["assess_picks"]
        and got["outer_votes"] == ref["outer_votes"]
        and got["shuf_votes"] == ref["shuf_votes"]
    )
    if not ok:
        raise RuntimeError("H279 regression cell KM276140-S279101-C0 differs")
    return {
        "cell": H279_REGRESSION_CELL,
        "bit_exact": True,
        "receipts_equal": True,
        "assess_picks_equal": True,
        "outer_votes_equal": True,
        "shuf_votes_equal": True,
        "invalid": inv,
        "changed": chg,
    }


def build_stored_round_packets(round_json_path: Path, gpu_costs_npz_path: Path, val_rows: list[dict], vap, vpj, vcf) -> dict:
    """Build K=8 feature and outcome packets for a stored round JSON (r782, r785) + GPU-rescored costs npz."""
    round_obj = json.loads(Path(round_json_path).read_text())["context_outcomes"]
    costs = {}
    with np.load(gpu_costs_npz_path, allow_pickle=False) as z:
        for r in val_rows:
            cid = r["context_sha256"]
            costs[cid] = z[cid].astype(np.float32)

    out = {}
    for r in val_rows:
        cid = r["context_sha256"]
        cands = round_obj[cid]["candidates"]
        val_gain = np.array(
            [
                np.nan if c.get("val_gain") is None else float(c["val_gain"])
                for c in cands
            ],
            dtype=np.float64,
        )
        gate = np.isfinite(val_gain) & (val_gain > -2.0)
        utility = np.where(gate, val_gain, -2.0).astype(np.float64)
        fitness = (-utility).astype(np.float64)

        rc = costs[cid]
        order = np.argsort(rc, kind="mergesort")
        tk = order[: fb.K]
        X = core.build_candidate_features(
            cid=cid,
            cand_indices=tk,
            r882_cost=rc,
            base_prior_cost=vap[cid],
            proj16=vpj[cid],
            ctx_feat=vcf[cid],
            pool_npz=round_obj[cid],
            full_order=order,
        )
        if not np.all(np.isfinite(X)):
            raise RuntimeError(f"non-finite features in stored round for {cid}")
        out[cid] = {
            "X": X,
            "utility_k": utility[tk],
            "gate_k": gate[tk],
            "fitness_k": fitness[tk],
            "slot0_u": float(utility[int(order[0])]),
        }
    return out


def score_packet_collection(
    cids: list[str],
    fams: list[str],
    packets: dict[str, dict],
    h284_export_dir: Path,
    h284_cells: list[dict],
    h278_export_dir: Path,
    h278_cells: list[dict],
    seed_base: int = stat.BOOT_SEED,
) -> dict:
    """Score a collection of K=8 packets with frozen H284 (279101-279103) and frozen H278 (278101-278103) models."""
    n_ctx = len(cids)
    X = np.concatenate([packets[c]["X"] for c in cids])
    n_seeds = len(h284_cells)
    panel = np.full((n_seeds, n_ctx, len(stat.ALL_ARMS)), np.nan, dtype=np.float64)
    invalid_by_seed = {a: [] for a in stat.ALL_ARMS}
    changed_by_seed = {a: [] for a in stat.ALL_ARMS}
    picks_by_seed = []

    for si in range(n_seeds):
        c284 = h284_cells[si]
        c278 = h278_cells[si]
        s284 = fb.score_frozen_bagged_cell(h284_export_dir, c284, X, n_ctx, fb.K)
        p278 = fb.score_frozen_h278_cell(h278_export_dir, c278, X, n_ctx, fb.K)
        picks = {
            "slot0": s284["slot0"],
            "RERANK8_VETO": s284["RERANK8_VETO"],
            "H278_RERANK8_VETO": p278,
            "BAG_SWITCH": s284["BAG_SWITCH"],
            "BAG_SWITCH_NESTED": s284["BAG_SWITCH_NESTED"],
        }
        picks_by_seed.append(
            {
                "h284_seed": int(c284["seed"]),
                "h278_seed": int(c278["seed"]),
                "q_selected": int(c284["q_selected"]),
                "picks": picks,
                "votes": s284["votes"],
            }
        )
        for ai, a in enumerate(stat.ALL_ARMS):
            pa = picks[a]
            panel[si, :, ai] = [packets[c]["utility_k"][j] for c, j in zip(cids, pa)]
            inv_cnt = int(
                sum(
                    (not bool(packets[c]["gate_k"][j])) or float(packets[c]["utility_k"][j]) <= -2.0 + 1e-9
                    for c, j in zip(cids, pa)
                )
            )
            chg_cnt = int(sum(int(j) != 0 for j in pa))
            invalid_by_seed[a].append(inv_cnt)
            changed_by_seed[a].append(chg_cnt)

    summary = stat.summarize_pool_panel(panel, fams, invalid_by_seed, changed_by_seed, seed_base=seed_base)
    return {
        "summary": summary,
        "panel": panel,
        "cids": cids,
        "fams": fams,
        "picks_by_seed": picks_by_seed,
    }


def s3_fetch_versioned(bucket: str, key: str, version_id: str, dst: Path, expected_sha256: str | None = None) -> str:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws",
        "s3api",
        "get-object",
        "--profile",
        "veox-jepson-prod",
        "--bucket",
        bucket,
        "--key",
        key,
    ]
    if version_id and version_id not in ("", "null", "None"):
        cmd.extend(["--version-id", str(version_id)])
    cmd.append(str(dst))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    got_sha = sha(dst)
    if expected_sha256 and got_sha != expected_sha256:
        raise RuntimeError(f"S3 fetched sha256 mismatch for {key}: {got_sha} != {expected_sha256}")
    return got_sha


def load_fresh_pool_receipt(receipt_path: Path, cache_dir: Path, cohort_map: dict[str, str]) -> dict:
    """Load a fresh pool receipt (fresh_pools_H283.json or fresh_pools_H285.json), fetching S3 artifacts by VersionId
    when needed, and return normalized {cids, fams, packets, provenance}."""
    rec = json.loads(Path(receipt_path).read_text())
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Support either local_path or S3 bucket/key/VersionId
    npz_path = None
    s3_meta = {}
    if isinstance(rec, dict):
        # Case 1: top-level packet npz URI/key or local path
        bucket = rec.get("bucket") or rec.get("s3_bucket") or "veox-jopedime-learning-use2"
        key = rec.get("s3_key") or rec.get("key") or rec.get("packets_s3_key") or rec.get("features_s3_key")
        vid = rec.get("VersionId") or rec.get("version_id") or rec.get("s3_version_id") or rec.get("packets_version_id")
        exp_sha = rec.get("sha256") or rec.get("packets_sha256")
        if not key and isinstance(rec.get("packets"), dict):
            pk = rec["packets"]
            bucket = pk.get("bucket") or pk.get("s3_bucket") or bucket
            key = pk.get("s3_key") or pk.get("key")
            vid = pk.get("VersionId") or pk.get("version_id") or pk.get("s3_version_id")
            exp_sha = pk.get("sha256")
        if not key and isinstance(rec.get("artifacts"), list):
            for art in rec["artifacts"]:
                if isinstance(art, dict) and str(art.get("key") or art.get("s3_key") or art.get("file", "")).endswith(".npz"):
                    bucket = art.get("bucket") or art.get("s3_bucket") or bucket
                    key = art.get("s3_key") or art.get("key")
                    vid = art.get("VersionId") or art.get("version_id") or art.get("s3_version_id")
                    exp_sha = art.get("sha256")
                    break
        if key:
            if key.startswith("s3://"):
                parts = key[5:].split("/", 1)
                bucket, key = parts[0], parts[1]
            dst = cache_dir / Path(key).name
            got_sha = s3_fetch_versioned(bucket, key, str(vid or ""), dst, exp_sha)
            npz_path = dst
            s3_meta = {"bucket": bucket, "key": key, "VersionId": vid, "sha256": got_sha}
        elif rec.get("local_path") and Path(rec["local_path"]).exists():
            npz_path = Path(rec["local_path"])
            s3_meta = {"local_path": str(npz_path), "sha256": sha(npz_path)}

    if npz_path is None:
        raise RuntimeError(f"Could not resolve fresh packet npz from receipt {receipt_path}")

    packets = {}
    cids = []
    fams = []
    with np.load(npz_path, allow_pickle=False) as z:
        files = set(z.files)
        if "X" in files and ("cids" in files or "context_ids" in files):
            raw_cids = [str(c) for c in (z["cids"] if "cids" in files else z["context_ids"])]
            X_all = z["X"].astype(np.float32)
            if X_all.ndim == 2:
                X_all = X_all.reshape(len(raw_cids), fb.K, 568)
            u_key = "utility_k" if "utility_k" in files else ("utility" if "utility" in files else "val_gain_k")
            g_key = "gate_k" if "gate_k" in files else ("gate" if "gate" in files else None)
            u_all = z[u_key].astype(np.float64)
            if g_key and g_key in files:
                g_all = z[g_key].astype(bool)
            else:
                g_all = np.isfinite(u_all) & (u_all > -2.0 + 1e-9)
            raw_fams = (
                [str(f) for f in z["family_ids"]]
                if "family_ids" in files
                else ([str(f) for f in z["fams"]] if "fams" in files else [cohort_map[c] for c in raw_cids])
            )
            for i, cid in enumerate(raw_cids):
                cids.append(cid)
                fams.append(raw_fams[i])
                packets[cid] = {
                    "X": X_all[i],
                    "utility_k": np.where(g_all[i], u_all[i], -2.0).astype(np.float64),
                    "gate_k": g_all[i],
                }
        else:
            raise RuntimeError(f"Unsupported npz layout in {npz_path}: {sorted(files)[:10]}")

    return {
        "cids": cids,
        "fams": fams,
        "packets": packets,
        "s3_meta": s3_meta,
        "receipt_sha256": sha(receipt_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=("export", "transfer"), default="export")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--export-dir", type=Path)
    ap.add_argument("--h278-export-dir", type=Path, default=H284_DATA / "h278_export")
    ap.add_argument("--fresh-cohort", type=Path, default=Path("/home/ubuntu/jope_plan/ns/receipts/fresh_cohort_v1.json"))
    ap.add_argument("--fresh-h283", type=Path, default=Path("/home/ubuntu/jope_plan/ns/receipts/fresh_pools_H283.json"))
    ap.add_argument("--fresh-h285", type=Path, default=Path("/home/ubuntu/jope_plan/ns/receipts/fresh_pools_H285.json"))
    args = ap.parse_args()

    os.environ["H265_PAUSE_FILE"] = str(args.output.with_suffix(".pause.json"))
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
    ):
        raise RuntimeError("CPU xbabe2 nice19 required")

    sys.addaudithook(audit_hook)
    core.INPUT_POOLS_DIR = POOLS
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)

    started = time.monotonic()
    r882_sha_before, dope_before = sha(pins["checkpoint"]), dope_of(pins["checkpoint"])
    if dope_before != pins["dope_composite_sha256"]:
        raise RuntimeError("DOPE identity before")

    selected = json.loads((POOLS / "selected.json").read_text())
    train, val = selected["train"], selected["dev"]
    if (
        len(train) != 1219
        or len(val) != 91
        or len({r["family_id"] for r in train}) != 205
    ):
        raise RuntimeError("cohort size")
    if any(
        r.get("has_meta_test") is True or r["task"] != "regression" for r in train + val
    ):
        raise RuntimeError("protected context")

    costs = {}
    with (
        np.load(H276_INPUTS / "r882-train-costs.npz") as zt,
        np.load(H276_INPUTS / "r882-secondary-costs.npz") as zv,
    ):
        for r in train + val:
            cid = r["context_sha256"]
            costs[cid] = (zt[cid] if cid in zt.files else zv[cid]).astype(np.float32)

    base = native91(val, costs, json.loads(Path(pins["reference"]).read_text()))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    save(out / "baseline.json", dict(status="PASS", **base))
    print("BASELINE PASS", base, flush=True)

    ap_, pj, cf = priors_for(train)
    fam_by_cid = {r["context_sha256"]: r["family_id"] for r in train}
    P8 = topk.load_packets(train, costs, ap_, pj, cf, 8)
    reg279 = h279_cell_regression(train, P8, fam_by_cid)
    print("H279 REGRESSION PASS", reg279["cell"], flush=True)

    if args.smoke:
        save(
            out / "smoke.json",
            dict(
                status="PASS",
                optimizer_updates=0,
                baseline=base,
                h279_regression=reg279,
                r882_sha_before=r882_sha_before,
                r882_sha_after=sha(pins["checkpoint"]),
                dope_before=dope_before,
                dope_after=dope_of(pins["checkpoint"]),
            ),
        )
        return

    if args.mode == "export":
        export_dir = args.export_dir or (out / "export")
        code_hashes = {k: sha(HERE / k) for k in pins["source_sha256"]}
        head = subprocess.check_output(
            ["git", "--git-dir=/home/ubuntu/JopeDime/.git", "rev-parse", "refs/remotes/origin/v1/next"],
            text=True,
        ).strip()
        lock = fb.fit_full_train_bagged(
            train_rows=train,
            packets=P8,
            fam_by_cid=fam_by_cid,
            export_dir=export_dir,
            code_hashes=code_hashes,
            source_commit=head,
            seeds=fb.SEEDS,
        )
        # Also write outcome-free DL-Val91 dev feature packets for r882 and r3175 so frozen scoring is self-contained
        vap, vpj, vcf = priors_for(val)
        V8_r882 = topk.load_packets(val, costs, vap, vpj, vcf, 8)
        vids = [r["context_sha256"] for r in val]
        vfams = [r["family_id"] for r in val]
        np.savez_compressed(
            export_dir / "dev-features-r882-top8.npz",
            X=np.stack([V8_r882[c]["X"] for c in vids], axis=0),
            utility_k=np.stack([V8_r882[c]["utility_k"] for c in vids], axis=0),
            gate_k=np.stack([V8_r882[c]["gate_k"] for c in vids], axis=0),
            cids=np.array(vids),
            family_ids=np.array(vfams),
        )
        save(
            out / "export_summary.json",
            dict(
                status="EXPORT_COMPLETE",
                elapsed_seconds=time.monotonic() - started,
                baseline=base,
                h279_regression=reg279,
                fit_lock_sha256=lock["fit_lock_sha256"],
                cells=[
                    {
                        "seed": c["seed"],
                        "selected_lr": c["selected_lr"],
                        "selected_n_estimators": c["selected_n_estimators"],
                        "strict_margin": c["strict_margin"],
                        "strict_admitted": c["strict_admitted"],
                        "q_fixed": c["q_fixed"],
                        "q_selected": c["q_selected"],
                        "q_selection_tier": c["q_selection_tier"],
                        "ranker_sha256": c["ranker"]["sha256"],
                        "veto_sha256": c["veto"]["sha256"],
                        "bag_sha256s": [b["sha256"] for b in c["bags"]],
                    }
                    for c in lock["cells"]
                ],
                fresh_pools_read=0,
                dl_val_reads_before_lock=0,
                optimizer_updates=0,
                r882_sha_before=r882_sha_before,
                r882_sha_after=sha(pins["checkpoint"]),
                dope_before=dope_before,
                dope_after=dope_of(pins["checkpoint"]),
            ),
        )
        print("EXPORT COMPLETE", lock["fit_lock_sha256"], flush=True)
        return

    # Mode == "transfer"
    del P8
    core.trim_malloc()
    export_dir = args.export_dir or (H284_DATA / "export")
    h284_em = json.loads((export_dir / "export_manifest.json").read_text())
    h278_em = json.loads((args.h278_export_dir / "export_manifest.json").read_text())

    vap, vpj, vcf = priors_for(val)
    vids = [r["context_sha256"] for r in val]
    vfams = [r["family_id"] for r in val]

    # 1. Descriptive: H282's genuinely distinct r782 and r785 GPU pools on DL-Val91 (plus r882 and r3175 reference)
    descriptive = {}
    V8_r882 = topk.load_packets(val, costs, vap, vpj, vcf, 8)
    res_r882 = score_packet_collection(
        vids, vfams, V8_r882, export_dir, h284_em["cells"], args.h278_export_dir, h278_em["cells"], seed_base=stat.BOOT_SEED + 10
    )
    descriptive["r882_dlval91"] = res_r882["summary"]

    # r3175 GPU DL-Val91 (from H281)
    r3175_gpu_npz = Path("/home/ubuntu/jope-data/h281/gpu-attempt-001/r3175-gpu-costs.npz")
    if r3175_gpu_npz.exists():
        costs_3175 = {}
        with np.load(r3175_gpu_npz, allow_pickle=False) as z3:
            for r in val:
                cid = r["context_sha256"]
                costs_3175[cid] = z3[cid].astype(np.float32)
        V8_r3175 = topk.load_packets(val, costs_3175, vap, vpj, vcf, 8)
        res_r3175 = score_packet_collection(
            vids, vfams, V8_r3175, export_dir, h284_em["cells"], args.h278_export_dir, h278_em["cells"], seed_base=stat.BOOT_SEED + 20
        )
        descriptive["r3175_gpu_dlval91"] = res_r3175["summary"]

    # H282 distinct r782 and r785 GPU pools
    r782_res = None
    r785_res = None
    for tag, rnum, sb in (("r782", 782, 30), ("r785", 785, 40)):
        r_json = H282_DATA / "stored" / f"round-{rnum}.json"
        r_npz = H282_DATA / "gpu-attempt-001" / tag / "r3175-gpu-costs.npz"
        if r_json.exists() and r_npz.exists():
            pkts = build_stored_round_packets(r_json, r_npz, val, vap, vpj, vcf)
            res_tag = score_packet_collection(
                vids, vfams, pkts, export_dir, h284_em["cells"], args.h278_export_dir, h278_em["cells"], seed_base=stat.BOOT_SEED + sb
            )
            descriptive[f"{tag}_gpu_dlval91"] = res_tag["summary"]
            if tag == "r782":
                r782_res = res_tag
            else:
                r785_res = res_tag

    if r782_res is not None and r785_res is not None:
        # Combined mean across the two genuinely distinct stored GPU pools (r782, r785)
        panel_782_785 = 0.5 * (r782_res["panel"] + r785_res["panel"])
        inv_comb = {
            a: [int(x + y) for x, y in zip(r782_res["summary"]["invalid_by_seed"][a], r785_res["summary"]["invalid_by_seed"][a])]
            for a in stat.ALL_ARMS
        }
        chg_comb = {
            a: [int(x + y) for x, y in zip(r782_res["summary"]["changed_by_seed"][a], r785_res["summary"]["changed_by_seed"][a])]
            for a in stat.ALL_ARMS
        }
        descriptive["r782_r785_distinct_mean_dlval91"] = stat.summarize_pool_panel(
            panel_782_785, vfams, inv_comb, chg_comb, seed_base=stat.BOOT_SEED + 50
        )

    # 2. Fresh pools: H283 (EVEN half) and/or H285 (ODD half)
    cohort_map = {}
    if args.fresh_cohort.exists():
        fc = json.loads(args.fresh_cohort.read_text())
        for item in fc.get("eligible", []):
            if isinstance(item, dict) and item.get("buildable") and isinstance(item.get("context"), dict):
                cid = item["context"].get("context_sha256")
                fkey = item.get("family_key")
                if cid and fkey:
                    cohort_map[str(cid)] = str(fkey)
        for item in fc.get("contexts", fc.get("rows", [])):
            if isinstance(item, dict) and "context_sha256" in item and "family_id" in item:
                cohort_map[item["context_sha256"]] = item["family_id"]

    fresh_sets = {}
    fresh_summaries = {}
    fresh_provenance = {}
    for label, rpath, sb in (
        ("H283_even", args.fresh_h283, 100),
        ("H285_odd", args.fresh_h285, 200),
    ):
        if rpath.exists():
            loaded = load_fresh_pool_receipt(rpath, out / "fresh_cache" / label, cohort_map)
            scored = score_packet_collection(
                loaded["cids"],
                loaded["fams"],
                loaded["packets"],
                export_dir,
                h284_em["cells"],
                args.h278_export_dir,
                h278_em["cells"],
                seed_base=stat.BOOT_SEED + sb,
            )
            fresh_sets[label] = (loaded, scored)
            fresh_summaries[label] = {
                "summary": scored["summary"],
                "gates": {
                    "BAG_SWITCH": stat.evaluate_arm_fresh_gates(scored["summary"], "BAG_SWITCH"),
                    "BAG_SWITCH_NESTED": stat.evaluate_arm_fresh_gates(scored["summary"], "BAG_SWITCH_NESTED"),
                },
            }
            fresh_provenance[label] = loaded["s3_meta"]

    union_summary = None
    if fresh_sets:
        u_cids, u_fams, u_packets = [], [], {}
        for label in ("H283_even", "H285_odd"):
            if label in fresh_sets:
                loaded, _ = fresh_sets[label]
                for c, f in zip(loaded["cids"], loaded["fams"]):
                    if c not in u_packets:
                        u_cids.append(c)
                        u_fams.append(f)
                        u_packets[c] = loaded["packets"][c]
        u_scored = score_packet_collection(
            u_cids,
            u_fams,
            u_packets,
            export_dir,
            h284_em["cells"],
            args.h278_export_dir,
            h278_em["cells"],
            seed_base=stat.BOOT_SEED + 300,
        )
        union_summary = u_scored["summary"]
        np.savez_compressed(out / "fresh_union_panel.npz", panel=u_scored["panel"])
        save(out / "fresh_union_picks.json", {"cids": u_cids, "fams": u_fams, "by_seed": u_scored["picks_by_seed"]})

    decision = stat.decide_fresh(union_summary)
    result = {
        "wave": "H284",
        "status": "COMPLETE",
        "verdict": decision["verdict"],
        "decision": decision,
        "elapsed_seconds": time.monotonic() - started,
        "baseline": base,
        "h279_regression": reg279,
        "export_manifest_sha256": sha(export_dir / "export_manifest.json"),
        "h278_export_manifest_sha256": sha(args.h278_export_dir / "export_manifest.json"),
        "fresh_provenance": fresh_provenance,
        "fresh_union": union_summary,
        "fresh_by_pool_set": fresh_summaries,
        "descriptive_dlval91": descriptive,
        "optimizer_updates": 0,
        "serving_promotion": False,
        "r882_sha_before": r882_sha_before,
        "r882_sha_after": sha(pins["checkpoint"]),
        "dope_before": dope_before,
        "dope_after": dope_of(pins["checkpoint"]),
    }
    save(out / "metrics_full.json", result)

    # Build compact <50 KB metrics.json
    def strip_fams(s):
        if s is None:
            return None
        s2 = {k: v for k, v in s.items() if k != "family_ids"}
        s2["contrasts"] = {
            ck: {k: v for k, v in cv.items() if k != "harmed_001_families_by_seed"}
            for ck, cv in s.get("contrasts", {}).items()
        }
        return s2

    compact = dict(result)
    compact["fresh_union"] = strip_fams(result["fresh_union"])
    compact["fresh_by_pool_set"] = {
        k: {"summary": strip_fams(v["summary"]), "gates": v["gates"]}
        for k, v in result["fresh_by_pool_set"].items()
    }
    compact["descriptive_dlval91"] = {
        k: strip_fams(v) for k, v in result["descriptive_dlval91"].items()
    }
    save(out / "metrics.json", compact)
    print("TRANSFER COMPLETE", result["verdict"], flush=True)


if __name__ == "__main__":
    main()

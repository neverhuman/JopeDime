"""H287 runner:
1. --smoke: Native91 (91/91, 1.1377941898164525) + bit-exact reproduction of H279 cell KM276140-S279101-C0.
   fresh_pools_read = 0 enforced by sys.addaudithook for the entire wave.
2. Full run:
   - Step 1: Reproduce Native91 and H279 cell KM276140-S279101-C0 bit-exact.
   - Step 2 & 3: Run 4-fold family-grouped inner CV on DL-Train (205 families, 1,219 contexts; same folds as H284;
     refit seeds 287101-287103) for BAG_SWITCH, RERANK8_VETO, ENS_MEAN, ENS_MEAN_PROPOSER, ENS_VOTE2, ENS_VOTE3,
     CLIP, and CLIP_ENS_VOTE2.
   - Step 4: Freeze and export all models + ensemble_spec.json + fit_lock.json to export_dir, and verify 64-context
     DL-Train reload bit-exactness before any DL-Val91 read.
   - Step 3 (descriptive): Score reused DL-Val91 (r882, r3175 GPU, and r782/r785 GPU, explicitly labeled 'reused').
   - Step 5: Compute H287 inner-CV verdict (EXPORTED_ADVANCE / EXPORTED_NO_ADVANCE)."""

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
import analysis_h287 as stat
import bagged
import ensemble as ens
import h265_core as core
import topk

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
H276_INPUTS = Path("/home/ubuntu/jope-data/h276/inputs")
H284_DATA = Path("/home/ubuntu/jope-data/h284")
H287_DATA = Path("/home/ubuntu/jope-data/h287")
H282_DATA = Path("/home/ubuntu/jope-data/h282")
POOLS = Path("/home/ubuntu/jope-data/h50/input")
CACHE = Path("/home/ubuntu/jope-data/h58/cache")
NATIVE91 = 1.1377941898164525
TOP4_CTX, TOP4_FAM = 1.2403601682250796, 1.2519997168684853
H279_REGRESSION_CELL = "KM276140-S279101-C0"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    return ens.save_json(Path(path), obj)


def audit_hook(event, args):
    """Refuse any open of a sealed, holdout, meta-test, or fresh-pool path (fresh_pools_read = 0)."""
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
                "fresh_pools",
                "fresh_cohort",
                "fresh_top8",
                "fresh-top8",
                "fresh-build",
            )
        ):
            raise RuntimeError("protected or fresh-pool path blocked in H287: " + s)


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


def score_reused_dlval_collection(
    cids: list[str],
    fams: list[str],
    packets: dict[str, dict],
    export_dir: Path,
    spec: dict,
    label: str,
    seed_base: int = stat.BOOT_SEED + 100,
) -> dict:
    """Score reused DL-Val91 packets (descriptive only, explicitly labeled reused=True)."""
    n_ctx = len(cids)
    X = np.concatenate([packets[c]["X"] for c in cids])
    scored = ens.score_all_h287_arms(export_dir, X, n_ctx, k=ens.K, spec=spec)

    n_seeds = 3
    panel = np.full((n_seeds, n_ctx, len(stat.ALL_ARMS)), np.nan, dtype=np.float64)
    invalid_by_seed = {a: [] for a in stat.ALL_ARMS}
    changed_by_seed = {a: [] for a in stat.ALL_ARMS}

    for si in range(n_seeds):
        picks_si = {
            "slot0": scored["slot0"],
            "RERANK8_VETO": scored["per_seed"]["RERANK8_VETO"][si],
            "BAG_SWITCH": scored["per_seed"]["BAG_SWITCH"][si],
            "ENS_MEAN": scored["ENS_MEAN"],
            "ENS_MEAN_PROPOSER": scored["ENS_MEAN_PROPOSER"],
            "ENS_VOTE2": scored["ENS_VOTE2"],
            "ENS_VOTE3": scored["ENS_VOTE3"],
            "CLIP": scored["per_seed"]["CLIP"][si],
            "CLIP_ENS_VOTE2": scored["CLIP_ENS_VOTE2"],
        }
        for ai, arm in enumerate(stat.ALL_ARMS):
            pa = picks_si[arm]
            panel[si, :, ai] = [packets[c]["utility_k"][j] for c, j in zip(cids, pa)]
            inv_cnt = int(
                sum(
                    (not bool(packets[c]["gate_k"][j])) or float(packets[c]["utility_k"][j]) <= -2.0 + 1e-9
                    for c, j in zip(cids, pa)
                )
            )
            chg_cnt = int(sum(int(j) != 0 for j in pa))
            invalid_by_seed[arm].append(inv_cnt)
            changed_by_seed[arm].append(chg_cnt)

    summary = stat.summarize_panel(
        panel,
        fams,
        invalid_by_seed,
        changed_by_seed,
        arm_names=stat.ALL_ARMS,
        seed_base=seed_base,
        label=label,
        reused=True,
    )
    return {
        "summary": summary,
        "panel": panel,
    }


def build_stored_round_packets(round_json_path: Path, gpu_costs_npz_path: Path, val_rows: list[dict], vap, vpj, vcf) -> dict:
    stored = json.loads(Path(round_json_path).read_text())["dev_utility"]
    vids = [r["context_sha256"] for r in val_rows]
    with np.load(gpu_costs_npz_path, allow_pickle=False) as z:
        costs = {c: z[c].astype(np.float32) for c in vids}
    V = topk.load_packets(val_rows, costs, vap, vpj, vcf, ens.K)
    slot0_matches = int(
        sum(float(V[c]["utility_k"][0]) == float(s) for c, s in zip(vids, stored["per_context"]))
    )
    if slot0_matches != 91:
        raise RuntimeError(f"stored round {round_json_path.name} slot 0 mismatch: {slot0_matches}/91")
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--export-dir", type=Path, default=H287_DATA / "export")
    ap.add_argument("--h284-export-dir", type=Path, default=H284_DATA / "export")
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
                fresh_pools_read=0,
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

    h284_em = json.loads((args.h284_export_dir / "export_manifest.json").read_text())

    # Step 2 & 3: 4-fold family-grouped inner CV on DL-Train (205 families, 1,219 contexts)
    inner_cv = ens.run_inner_cv_dl_train(train, P8, fam_by_cid, h284_em["cells"])
    np.savez_compressed(out / "inner_cv_panel.npz", panel=inner_cv["panel"])
    print("INNER CV COMPLETE", flush=True)

    # Step 4: Freeze and export all models + ensemble_spec.json + fit_lock.json before reading DL-Val91
    code_hashes = {k: sha(HERE / k) for k in pins["source_sha256"]}
    head = subprocess.check_output(
        ["git", "--git-dir=/home/ubuntu/JopeDime/.git", "rev-parse", "refs/remotes/origin/v1/next"],
        text=True,
    ).strip()
    exported = ens.export_h287_models_and_spec(
        train_rows=train,
        packets=P8,
        fam_by_cid=fam_by_cid,
        h284_export_dir=args.h284_export_dir,
        export_dir=args.export_dir,
        code_hashes=code_hashes,
        source_commit=head,
    )
    print("EXPORT COMPLETE", exported["lock"]["fit_lock_sha256"], flush=True)

    # Free DL-Train packets before scoring reused DL-Val91
    del P8
    core.trim_malloc()

    # Step 3 (descriptive): Reused DL-Val91 (explicitly labeled "reused")
    vap, vpj, vcf = priors_for(val)
    vids = [r["context_sha256"] for r in val]
    vfams = [r["family_id"] for r in val]

    reused_dlval91 = {}
    V8_r882 = topk.load_packets(val, costs, vap, vpj, vcf, 8)
    res_r882 = score_reused_dlval_collection(
        vids, vfams, V8_r882, args.export_dir, exported["spec"], "reused_r882_dlval91", seed_base=stat.BOOT_SEED + 100
    )
    reused_dlval91["reused_r882_dlval91"] = res_r882["summary"]

    r3175_gpu_npz = Path("/home/ubuntu/jope-data/h281/gpu-attempt-001/r3175-gpu-costs.npz")
    if r3175_gpu_npz.exists():
        costs_3175 = {}
        with np.load(r3175_gpu_npz, allow_pickle=False) as z3:
            for r in val:
                cid = r["context_sha256"]
                costs_3175[cid] = z3[cid].astype(np.float32)
        V8_r3175 = topk.load_packets(val, costs_3175, vap, vpj, vcf, 8)
        res_r3175 = score_reused_dlval_collection(
            vids, vfams, V8_r3175, args.export_dir, exported["spec"], "reused_r3175_gpu_dlval91", seed_base=stat.BOOT_SEED + 200
        )
        reused_dlval91["reused_r3175_gpu_dlval91"] = res_r3175["summary"]

    r782_res = None
    r785_res = None
    for tag, rnum, sb in (("r782", 782, 300), ("r785", 785, 400)):
        r_json = H282_DATA / "stored" / f"round-{rnum}.json"
        r_npz = H282_DATA / "gpu-attempt-001" / tag / "r3175-gpu-costs.npz"
        if r_json.exists() and r_npz.exists():
            pkts = build_stored_round_packets(r_json, r_npz, val, vap, vpj, vcf)
            res_tag = score_reused_dlval_collection(
                vids, vfams, pkts, args.export_dir, exported["spec"], f"reused_{tag}_gpu_dlval91", seed_base=stat.BOOT_SEED + sb
            )
            reused_dlval91[f"reused_{tag}_gpu_dlval91"] = res_tag["summary"]
            if tag == "r782":
                r782_res = res_tag
            else:
                r785_res = res_tag

    if r782_res is not None and r785_res is not None:
        panel_782_785 = 0.5 * (r782_res["panel"] + r785_res["panel"])
        inv_comb = {
            a: [int(x + y) for x, y in zip(r782_res["summary"]["invalid_by_seed"][a], r785_res["summary"]["invalid_by_seed"][a])]
            for a in stat.ALL_ARMS
        }
        chg_comb = {
            a: [int(x + y) for x, y in zip(r782_res["summary"]["changed_by_seed"][a], r785_res["summary"]["changed_by_seed"][a])]
            for a in stat.ALL_ARMS
        }
        reused_dlval91["reused_r782_r785_distinct_mean_dlval91"] = stat.summarize_panel(
            panel_782_785,
            vfams,
            inv_comb,
            chg_comb,
            arm_names=stat.ALL_ARMS,
            seed_base=stat.BOOT_SEED + 500,
            label="reused_r782_r785_distinct_mean_dlval91",
            reused=True,
        )

    decision = stat.decide_h287(inner_cv["summary"])
    result = {
        "wave": "H287",
        "slug": "h287_bagswitch_seed_ensemble_export",
        "status": "COMPLETE",
        "verdict": decision["verdict"],
        "decision": decision,
        "elapsed_seconds": time.monotonic() - started,
        "fresh_pools_read": 0,
        "dl_val_reads_before_lock": 0,
        "baseline": base,
        "h279_regression": reg279,
        "ensemble_spec_sha256": exported["spec"]["ensemble_spec_sha256"],
        "fit_lock_sha256": exported["lock"]["fit_lock_sha256"],
        "reload_bit_exact_check_on_64_dl_train_contexts": exported["reload_check_64"],
        "inner_cv_replicates_meta": inner_cv["replicates_meta"],
        "inner_cv_dl_train": inner_cv["summary"],
        "reused_descriptive_dlval91": reused_dlval91,
        "optimizer_updates": 0,
        "serving_promotion": False,
        "r882_sha_before": r882_sha_before,
        "r882_sha_after": sha(pins["checkpoint"]),
        "dope_before": dope_before,
        "dope_after": dope_of(pins["checkpoint"]),
    }
    save(out / "metrics_full.json", result)

    def strip_compact(s):
        if s is None:
            return None
        s2 = {k: v for k, v in s.items() if k != "family_ids"}
        s2["contrasts"] = {
            ck: {k: v for k, v in cv.items() if k != "harmed_001_families_by_seed"}
            for ck, cv in s.get("contrasts", {}).items()
        }
        return s2

    compact = dict(result)
    compact["inner_cv_dl_train"] = strip_compact(result["inner_cv_dl_train"])
    compact["reused_descriptive_dlval91"] = {
        k: strip_compact(v) for k, v in result["reused_descriptive_dlval91"].items()
    }
    save(out / "metrics.json", compact)
    print("H287 COMPLETE", result["verdict"], flush=True)


if __name__ == "__main__":
    main()

"""One CPU-only H278 wave. Part A: freeze and export one RERANK8_VETO model per model seed, fit on ALL DL-Train with the
unchanged H276 science (topk.py, h265_core.py, veto.py byte-identical; freeze.py adds no science). Part B: a
pre-registered transfer evaluation: P1 on DL-Val91 r882 top-8; P2 on the r3175 top-8 pool, gated on an exact
re-score of r3175's stored utility (hw_t2140)."""

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

import analysis as stat

import topk
import veto
import freeze

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INPUTS = Path("/home/ubuntu/jope-data/h278/inputs")
POOLS = Path("/home/ubuntu/jope-data/h50/input")
CACHE = Path("/home/ubuntu/jope-data/h58/cache")
NATIVE91 = 1.1377941898164525
TOP4_CTX, TOP4_FAM = 1.2403601682250796, 1.2519997168684853
OUT = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=False, default=float)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return sha(path)


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


def validate_launch(
    launch, head, manifest_sha256, prereg_sha256, fold_sha256, smoke=False
):
    if launch.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("launch manifest binding")
    if smoke:
        return
    if (
        not launch.get("merged_source_verified")
        or launch.get("source_commit") != head
        or launch.get("publication_ref") != "v1/next"
        or not launch.get("published_ancestor_verified")
        or len(str(launch.get("review_receipt_sha256", ""))) != 64
        or launch.get("prereg_sha256") != prereg_sha256
        or launch.get("fold_manifest_sha256") != fold_sha256
    ):
        raise RuntimeError("external published/reviewed launch binding")


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


def load_core():
    import importlib.util

    spec = importlib.util.spec_from_file_location("h265_core", HERE / "h265_core.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules["h265_core"] = core
    spec.loader.exec_module(core)
    core.INPUT_POOLS_DIR = POOLS  # same shared read-only pools as H265
    return core


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


def predict_picks(model_text, X, n_ctx, k, n_est, margin, core):
    import lightgbm as lgb

    sc = np.asarray(
        lgb.Booster(model_str=model_text).predict(X, num_iteration=int(n_est)),
        dtype=np.float64,
    ).reshape(n_ctx, k)
    return [core.select_from_top4_scores(sc[i], margin) for i in range(n_ctx)]


def oracle_summary(rows, packets):
    fams = np.array([r["family_id"] for r in rows])
    out = {}
    for key in ("r882_u", "oracle_top4_u", "oracle_top8_u", "oracle_pool_u"):
        u = np.array([packets[r["context_sha256"]][key] for r in rows])
        out[key] = dict(
            context_mean=float(u.mean()),
            family_mean=float(
                np.mean([u[fams == f].mean() for f in sorted(set(fams))])
            ),
        )
    out["invalid_in_top8_slice"] = int(
        sum(packets[r["context_sha256"]]["invalid_in_slice"] for r in rows)
    )
    return out


REGRESSION_CELL = "F272140-S272101-K0"


def regression(core, train, costs, P8):
    """H272 cell reproduced exactly with the byte-identical science (receipt, invalid, changed, deep-pick counts)."""
    import analysis_h272 as a272

    ref = json.loads((INPUTS / f"h272-{REGRESSION_CELL}.json").read_text())
    fam = {r["context_sha256"]: r["family_id"] for r in train}
    cbf = a272.cids_by_fold(train, 272140)
    got = topk.run_cell(272101, 0, "RERANK8", False, cbf, P8, fam, 8, salt=272140)
    inv = int(sum(not P8[c]["gate_k"][j] for c, j in zip(cbf[0], got["assess_picks"])))
    chg = int(sum(j != 0 for j in got["assess_picks"]))
    deep = int(sum(j >= 4 for j in got["assess_picks"]))
    if not (
        got["receipt"] == ref["receipts"]["RERANK8"]
        and inv == ref["invalid"]["RERANK8"]
        and chg == ref["changed"]["RERANK8"]
        and deep == ref["deep_picks"]
    ):
        raise RuntimeError("H272 regression cell differs")
    return dict(
        cell=REGRESSION_CELL,
        receipt_equal=True,
        invalid=inv,
        changed=chg,
        deep_picks=deep,
    )


def r3175_rescore(val, costs3175):
    """Re-score r3175's stored DL-Val91 utility from the CPU-reproducible r3175 costs (argmin; utility as Native91)."""
    stored = json.loads((INPUTS / "r3175" / "round-3175.json").read_text())[
        "dev_utility"
    ]
    u = []
    for r in val:
        cid = r["context_sha256"]
        with np.load(POOLS / "pools" / f"{cid}.npz", allow_pickle=False) as z:
            f, g = z["fitness"].astype(np.float64), z["gate"].astype(bool)
        j = int(np.argmin(costs3175[cid]))
        u.append(float(-f[j] if g[j] else -2.0))
    equal = int(sum(a == b for a, b in zip(u, stored["per_context"])))
    return dict(
        cpu_mean=float(np.mean(u)),
        stored_mean=stored["mean"],
        per_context_equal=equal,
        exact=bool(equal == 91 and float(np.mean(u)) == stored["mean"]),
    )


def features_packet(path, packets, rows):
    """Outcome-free export of the dev feature rows (no fitness, gate or utility): X (contexts, 8, 568) and context ids."""
    ids = [r["context_sha256"] for r in rows]
    np.savez_compressed(
        path, X=np.stack([packets[c]["X"] for c in ids]), cids=np.asarray(ids)
    )
    return sha(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume-from", type=Path)
    args = ap.parse_args()
    os.environ["H265_PAUSE_FILE"] = str(args.output.with_suffix(".pause.json"))
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
    ):
        raise RuntimeError("CPU xbabe2 nice19 required")
    out = args.output.resolve()
    out.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)
    launch = json.loads(args.launch.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    validate_launch(
        launch,
        head,
        sha(HERE / "manifest.json"),
        sha(HERE / "PREREG.md"),
        sha(HERE / "fold_manifest.json"),
        args.smoke,
    )
    if args.resume_from is not None:
        raise RuntimeError("H278 has no resume path")
    sys.addaudithook(audit_hook)
    core = load_core()
    started = time.monotonic()
    r882_sha_before, dope_before = sha(pins["checkpoint"]), dope_of(pins["checkpoint"])
    r3175_sha = sha(INPUTS / "r3175" / "round-3175.pt")
    if (
        dope_before != pins["dope_composite_sha256"]
        or dope_of(INPUTS / "r3175" / "round-3175.pt") != pins["dope_composite_sha256"]
    ):
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
    costs, costs3175 = {}, {}
    with (
        np.load(INPUTS / "r882-train-costs.npz") as zt,
        np.load(INPUTS / "r882-secondary-costs.npz") as zv,
        np.load(INPUTS / "r3175-secondary-costs.npz") as z3,
    ):
        for r in train + val:
            cid = r["context_sha256"]
            costs[cid] = (zt[cid] if cid in zt.files else zv[cid]).astype(np.float32)
        for r in val:
            costs3175[r["context_sha256"]] = z3[r["context_sha256"]].astype(np.float32)
    base = native91(val, costs, json.loads(Path(pins["reference"]).read_text()))
    rescore = r3175_rescore(val, costs3175)
    save(
        out / "baseline.json",
        dict(
            status="PASS",
            **base,
            r3175_rescore=rescore,
            r3175_checkpoint_sha256=r3175_sha,
        ),
    )
    print("BASELINE PASS", base, "r3175 rescore", rescore, flush=True)
    fam = {r["context_sha256"]: r["family_id"] for r in train}
    ap_, pj, cf = priors_for(train)
    P8 = topk.load_packets(train, costs, ap_, pj, cf, 8)
    if args.smoke:
        reg = regression(core, train, costs, P8)
        save(
            out / "smoke.json",
            dict(
                status="PASS",
                optimizer_updates=0,
                baseline=base,
                r3175_rescore=rescore,
                h272_regression=reg,
            ),
        )
        print("REGRESSION PASS", reg, flush=True)
        return
    # ---- Part A: freeze, export, reload regression, fit-lock (no DL-Val read before the lock) ----
    exp = out / "export"
    exp.mkdir()
    family_ids = sorted({r["family_id"] for r in train})
    sample = [r["context_sha256"] for r in train[:64]]
    X_sample = np.concatenate([P8[c]["X"] for c in sample])
    cells = []
    for s in freeze.SEEDS:
        folds = freeze.family_folds(train, s)
        res = freeze.freeze_veto(s, folds, P8, fam)
        files = {}
        for role, text, obj in (
            ("ranker", res["ranker"], res["ranker_obj"]),
            ("veto", res["gate"], res["gate_obj"]),
        ):
            p = exp / f"{role}-seed{s}.txt"
            p.write_text(text)
            reloaded = __import__("lightgbm").Booster(model_file=str(p))
            mem = obj.booster_.predict(
                X_sample,
                num_iteration=res["n_estimators"] if role == "ranker" else None,
            )
            got = reloaded.predict(
                X_sample,
                num_iteration=res["n_estimators"] if role == "ranker" else None,
            )
            if not np.array_equal(mem, got) or reloaded.num_feature() != 568:
                raise RuntimeError("export reload regression: " + p.name)
            files[role] = dict(
                file=p.name,
                sha256=sha(p),
                num_feature=568,
                reload_bit_exact_on_64_train_contexts=True,
            )
        cells.append(
            dict(
                seed=s,
                selected_lr=res["receipt"]["lr"],
                selected_n_estimators=res["n_estimators"],
                strict_margin=res["receipt"]["strict_margin"],
                unconstrained_margin=res["receipt"]["margin"],
                selected_tau=veto.TAU,
                strict_admitted=res["receipt"]["strict_admitted"],
                inner_fit_family_gain=res["receipt"]["inner_fit_family_gain"],
                inner_fit_ci95=res["receipt"]["inner_fit_ci95"],
                inner_folds={
                    str(k): sorted({fam[c] for c in v}) for k, v in folds.items() if v
                },
                **files,
            )
        )
        print("FROZEN", s, res["receipt"], flush=True)
    code = {
        f: sha(HERE / f)
        for f in (
            "topk.py",
            "h265_core.py",
            "veto.py",
            "freeze.py",
            "runner.py",
            "analysis.py",
        )
    }
    lock = save(
        exp / "fit_lock.json",
        dict(
            schema="jope.h278.fit-lock/v1",
            arm="RERANK8_VETO",
            fit_split="DL-Train",
            dl_val_reads_before_lock=0,
            training_family_ids=family_ids,
            training_contexts=1219,
            cells=cells,
            code_sha256=code,
            source_commit=launch["source_commit"],
        ),
    )
    print("FIT LOCK", lock, flush=True)
    # ---- Part B: DL-Val91 after the lock ----
    vap, vpj, vcf = priors_for(val)
    V882 = topk.load_packets(val, costs, vap, vpj, vcf, 8)
    V3175 = topk.load_packets(val, costs3175, vap, vpj, vcf, 8)
    feat = {
        "r882": features_packet(exp / "dev-features-r882-top8.npz", V882, val),
        "r3175": features_packet(exp / "dev-features-r3175-top8.npz", V3175, val),
    }
    vids = [r["context_sha256"] for r in val]
    panels, invalid, picks_out = {}, {}, {}
    for name, V in (("P1_r882", V882), ("P2_r3175", V3175)):
        X = np.concatenate([V[c]["X"] for c in vids])
        u = np.zeros((3, 91, 2))
        inv = np.zeros((3, 91, 2), dtype=bool)
        picks_out[name] = []
        for si, c in enumerate(cells):
            picks, _, _ = freeze.frozen_picks(
                (exp / c["ranker"]["file"]).read_text(),
                (exp / c["veto"]["file"]).read_text(),
                X,
                91,
                c["selected_n_estimators"],
                float("inf") if c["strict_margin"] == "inf" else c["strict_margin"],
            )
            for i, (cid, j) in enumerate(zip(vids, picks)):
                u[si, i] = [V[cid]["utility_k"][0], V[cid]["utility_k"][j]]
                inv[si, i] = [not V[cid]["gate_k"][0], not V[cid]["gate_k"][j]]
            picks_out[name].append(
                dict(
                    seed=c["seed"],
                    picks=[int(j) for j in picks],
                    changed=int(sum(j != 0 for j in picks)),
                )
            )
        panels[name], invalid[name] = u, inv
    save(out / "dev_picks.json", picks_out)
    np.savez_compressed(
        out / "dev_panels.npz",
        **{k + "_utility": v for k, v in panels.items()},
        **{k + "_invalid": v for k, v in invalid.items()},
    )
    result = dict(
        wave="H278", status="COMPUTED", elapsed_seconds=time.monotonic() - started
    )
    try:
        p1 = stat.summarize(val, panels["P1_r882"], invalid["P1_r882"])
        p2 = stat.summarize(val, panels["P2_r3175"], invalid["P2_r3175"])
        p2_gate_open = rescore["exact"]
        verdict = (
            ("ADOPT_SHADOW_PASS" if stat.adopt_gate(p2) else "NOT_TRANSFERRED")
            if p2_gate_open
            else "P2_BLOCKED"
        )
        result.update(
            status="COMPLETE",
            verdict=verdict,
            p1_r882_supporting=dict(**p1, gate_3of3_positive=stat.p1_gate(p1)),
            p2_r3175=dict(
                status="GATE" if p2_gate_open else "BLOCKED",
                block_reason=None if p2_gate_open else stat.P2_BLOCK_REASON,
                label="ADOPTION GATE"
                if p2_gate_open
                else "EXPLORATORY (not a gate; CPU-reproducible r3175 costs)",
                **p2,
                adopt_gate=stat.adopt_gate(p2),
            ),
            r3175_rescore=rescore,
            export=dict(
                fit_lock_sha256=lock,
                cells=cells,
                dev_feature_packets_sha256=feat,
                code_sha256=code,
            ),
            picks_changed={k: [x["changed"] for x in v] for k, v in picks_out.items()},
        )
    except Exception as exc:
        result["packaging_error"] = repr(exc)
    result.update(
        optimizer_updates=0,
        baseline=base,
        source_commit=launch["source_commit"],
        manifest_sha256=sha(HERE / "manifest.json"),
        serving_promotion=False,
        r882_sha_before=r882_sha_before,
        r882_sha_after=sha(pins["checkpoint"]),
        r3175_sha_before=r3175_sha,
        r3175_sha_after=sha(INPUTS / "r3175" / "round-3175.pt"),
        dope_before=dope_before,
        dope_after=dope_of(pins["checkpoint"]),
        dlval_reuse_note="DL-Val91 is heavily reused (H241-H277).",
    )
    save(out / "metrics.json", result)
    print("COMPLETE", result.get("verdict", result["status"]), flush=True)


if __name__ == "__main__":
    main()

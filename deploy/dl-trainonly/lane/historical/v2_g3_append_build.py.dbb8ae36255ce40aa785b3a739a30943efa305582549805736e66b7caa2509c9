#!/usr/bin/env python3
"""Campaign-2 context tree for the g3-appended catalog-only datasets (owner ENG:watcher 13:58 MT; g3 decisions 796e2724).

For every appended dataset whose DECIDED role is train_fit, train_nonpool, dev or heldout (never sealed_excluded, never a
role-less bridge), build all folds through tools/v2_rowsel_build.py's own T1 v2-split functions, exactly as the
qualification receipts did (universe, target-free split v2 + gates, train-only view + counterfactual, assemble =
t1prep.prepare/refit + teacher JTT1, downstream counterfactual, target_linear check, compose_context_v2), and write
create-only context dirs (encoder.jtt1.gz, teacher.jtt1.gz, a0.f32, context.json; '.done' last, dataset-wide) under
build-v2/ROWSEL-T1-g3append-v2split/shard-0/. A dataset is written only when every fold passes (builder rule).
New datasets have no epoch-0 lineage: epoch0_replaced lists none, the cut replaces nothing, and every lineage field that
only applies to epoch-0 cohorts is null with a note. Rights: research-only (G3-Q3, ALL-DATA-RESEARCH). Nothing is
published, assigned to S1 or charged here.
FOLD-1 (owner 07:22Z): the LINEAR-DETERMINED check refuses only the context whose own outer-train rows are linear
(v2_rowsel_build.context_refusal); another fold's verdict never removes a context. Moved into the lane from
tools-gen/g3_context_build.py so the fold-isolation tests cover it.
Usage: v2_g3_append_build.py DECISIONS_JSON [--limit K]
"""
import argparse, collections, hashlib, json, os, sys, time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]                                # the lane source root (tools/..)
sys.path[:0] = [str(SRC), str(SRC / "tools")]
import numpy as np  # noqa: E402
import v2_rowsel_build as B  # noqa: E402
from dltrain import rowsel, rowsel_raw, target_linear  # noqa: E402

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
OUT = RUN / "build-v2/ROWSEL-T1-g3append-v2split"
COHORTS = [(RUN / "receipts/data-expansion/small-data-500-999-manifest-20261006T174515Z.json", RUN / "inputs/data-expansion-small500-v1/blobs"),
           (RUN / "receipts/data-expansion/intake-complete-52-manifest-20261006T174515Z.json", RUN / "inputs/data-expansion-intake52-v1/blobs")]
ROLES = {"train_fit", "train_nonpool", "dev", "heldout"}
OWNER_DEC = "/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261006T202049Z.json"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def atomic(p: Path, b: bytes):
    tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
    with open(tmp, "xb") as f:
        f.write(b); f.flush(); os.fsync(f.fileno())
    os.link(tmp, p)        # create-only: fails if the name exists
    os.unlink(tmp)


def build_one(item, blobs, role, codes, shard):
    h = item["dataset_hash"]
    bodies = {k: (blobs / item["catalog_files"][k]["sha256"]).read_bytes() for k in ("meta", "train", "test")}
    for k, b in bodies.items():
        assert sha(b) == item["catalog_files"][k]["sha256"], (h, k)
    meta = json.loads(bodies["meta"])
    U = B.t1_universe(h, bodies["train"], bodies["test"], int(item["shape"]["columns"]))
    prep = B.t1_prep(U)
    folds = B.split_v2_t1(h, prep.positions, prep.values, prep.specs)
    rep = B.split_v2_t1_report(prep, folds)
    bad = B.split_gate_failures(rep)
    if bad:
        return {"status": "failed", "reason": "split_gate:" + ",".join(bad)}
    cut = {"cut_id": B.CUT_PREFIX + rep["sha256"], "schema": B.SPLIT_V2, "comparable_with_replaced": False, "folds": len(rep["fold_sizes"]),
           "replaces_cut": [], "rule": "cut_id = 'trainonly-split-v2:' + rowsel.split.sha256 (new catalog-only dataset: no registry split to replace)"}
    built_folds = [f for f in sorted(folds) if len(folds[f])]
    e0 = {"level": "dataset", "contexts": [], "count": 0, "fold_correspondence": "none: new catalog-only dataset appended by g3 (no epoch-0 contexts)",
          "new_contexts": len(built_folds), "context_count_changed": False}
    t1 = {p: {"sha256": item["catalog_files"][p]["sha256"], "bytes": len(bodies[p]), "lines": len(rowsel.lines(bodies[p]))} for p in ("train", "test")}
    lineage = {"manifest_sha256": None, "membership_sha256": None, "data_sha256": None, "membership_selected_sha256_matches_manifest": None,
               "port_reproduces_v1_selection": None, "membership_remap_matches_epoch0_folds": {}, "port_reproduces_v1_selection_not_run": "new catalog-only dataset: no v1 sample exists"}
    ds = {"h": h, "U": U, "meta": meta, "meta_sha256": item["catalog_files"]["meta"]["sha256"], "cohort": "append-g3", "t1": t1, "lineage": lineage,
          "split": {"kind": "none", "note": "catalog-only dataset appended by g3: no registry split object"}}
    refs = {"owner_line": "g3 append (owner ENG:watcher 13:58 MT; ALL-DATA-RESEARCH)",
            "context_extra": {"rights": "research-only", "rights_decision": f"{OWNER_DEC}#G3-Q3", "g3_role": role, "catalog_row_sha256": item["catalog_row_sha256"]},
            "row_selection_leak": None, "acq_selection": "catalog v246 T1 (min-max/impute over all source rows pre-split; train-only refit cancels the affine scaling)"}
    results, refused = [], []
    for f in built_folds:
        view = rowsel.select_view(U, folds, f, B.rowsel.DEFAULT_BUDGET, None, quotas=None)
        cf = rowsel.counterfactual_report(rowsel.select_view, U, folds, f, B.rowsel.DEFAULT_BUDGET)
        if not (cf["train_side_invariant"] and cf["test_draw_invariant"] and cf["base_view_identity"] == view.identity):
            return {"status": "failed", "reason": f"f{f}:counterfactual_failed"}
        built = B.assemble(U, view, meta)
        dcf = B.downstream_counterfactual(U, built, meta, h)
        if not dcf["train_side_invariant"]:
            return {"status": "failed", "reason": f"f{f}:downstream_counterfactual_failed"}
        ctx = B.compose_context_v2(ds, f, "clean", {"kind": "catalog_t1"}, view, cf, dcf, built, codes, refs, B.rowsel.DEFAULT_BUDGET, folds, rep, cut, e0,
                                   {"cohort": "append-g3"}, profile=B.PROFILE_V2)
        ctx["target_linear"] = target_linear.check_jtt1(built["art"]["split"]["jtt1"])
        r = B.context_refusal(ctx["target_linear"])                    # this context only (FOLD-1)
        if r:
            refused.append({"fold": f, "refused": r, "ols_rel_rmse": ctx["target_linear"]["ols_rel_rmse"]})
            continue
        results.append((ctx, built["art"]))
    if refused and not results:
        return {"status": "failed", "reason": "target_linear_in_features", "refused_contexts": refused}
    for ctx, art in results:
        d = shard / ctx["view_id"]
        d.mkdir(parents=True, exist_ok=False)
        atomic(d / "encoder.jtt1.gz", art["enc_gz"]); atomic(d / "teacher.jtt1.gz", art["tea_gz"]); atomic(d / "a0.f32", art["a0"])
        atomic(d / "context.json", json.dumps(dict(ctx, utc_us=int(time.time() * 1e6)), sort_keys=True).encode() + b"\n")
    for ctx, _ in results:
        atomic(shard / ctx["view_id"] / ".done", b"ok\n")
    return {"status": "done", "contexts": [{"fold": c["fold"], "view_id": c["view_id"], "context_sha256": c["context_sha256"], "n_train": c["n_train"], "n_test": c["n_test"]} for c, _ in results],
            **({"refused_contexts": refused} if refused else {})}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("decisions"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dec = json.loads(Path(a.decisions).read_bytes())
    role_of = {p["dataset_hash"]: p["role"] for p in dec["per_identity"]}
    codes = {p: sha((SRC / p).read_bytes()) for p in B.CODE_FILES}
    shard = OUT / "shard-0"; (shard / "_datasets").mkdir(parents=True, exist_ok=True)
    stat, n = collections.Counter(), 0
    for man, blobs in COHORTS:
        for it in json.loads(man.read_bytes())["items"]:
            h = it["dataset_hash"]; role = role_of.get(h)
            if role not in ROLES:
                stat[f"skipped:{role}"] += 1; continue
            rp = shard / "_datasets" / f"{h}.json"
            if rp.exists() and json.loads(rp.read_bytes()).get("status") == "done":
                stat["already_done"] += 1; continue
            if a.limit and n >= a.limit:
                break
            n += 1; t0 = time.time()
            try:
                rec = build_one(it, blobs, role, codes, shard)
            except Exception as e:  # recorded, the build continues
                rec = {"status": "failed", "reason": f"{type(e).__name__}: {e}"[:400]}
            rec.update(dataset_hash=h, role=role, wall_s=round(time.time() - t0, 2), utc_us=int(time.time() * 1e6), code_sha256=codes,
                       decisions_sha256=sha(Path(a.decisions).read_bytes()))
            if not rp.exists():
                atomic(rp, json.dumps(rec, sort_keys=True).encode() + b"\n")
            stat[rec["status"]] += 1
            print(json.dumps({"h": h, "role": role, "status": rec["status"], "reason": rec.get("reason"), "contexts": len(rec.get("contexts") or [])}), flush=True)
    print(json.dumps({"summary": dict(stat)}), flush=True)


if __name__ == "__main__":
    main()

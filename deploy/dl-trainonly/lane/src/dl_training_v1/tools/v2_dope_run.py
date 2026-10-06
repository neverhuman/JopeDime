"""Dope attention-small (4168-d) encodings for a v2 build tree, from outer-train rows only (owner L6(c)).

For each context dir OUT/shard-*/<view_id>/ with .done and context.json (not published), the encoder JTT1's train block
(jtt1.train_view: outer-train rows only, already train-only scaled) is written as a one-dataset Dope corpus
(<stage>/<view>/corpus/<dataset_hash>/{meta.json, train.csv}); meta.json carries the headers of the Dope stage spec for
this view (tools/stage_dope.py) when their width matches, else positional names. dope-kernel embed-regression-corpus runs
with --determinism-checks 1; its JSON 'vector' (4168 float32) becomes dope.npy, which is added to context.json 'files'
with sha256 and bytes, plus a 'dope' block: binary and router sha256, determinism, rows/features seen, and exposure
(dataset in the router's training cohort v2; dev family excluded from Dope dev metrics; held-out families unexposed per
dope-holdout-exposure-20261005T185521Z.json). Contexts with dope.npy already listed are skipped (resumable). The stage
dir of a finished context is removed (own scratch). CPU guard pauses apply. No S3, network or GPU.

Usage (lane slice): v2_dope_run.py --build TREE --shard I --of N [--limit K]
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from dltrain import cpuguard, jtt1

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
KERNEL = RUN / "build/dope-target/release/dope-kernel"
ROUTER = Path("/home/ubuntu/dope/runs/full-rust-metrics-20260718/router/attention-small")
COHORT = Path("/home/ubuntu/dope/runs/full-rust-metrics-20260718/training-cohort.v2.json")
# Per-context pretraining overlap (20:39Z inventory). The 18:55Z dope-holdout-exposure receipt measured the full-* routers'
# 2,096-record cohort, not attention-small, and is superseded for this encoder.
OVERLAP = RUN / "receipts/trainonly-v2/pretraining-overlap-20261005T203956Z.jsonl"
OVERLAP_SHA = "5f16216d2350ef30c06c570c0f33cbf8957771d1023f11f6f11ba0523405d24a"
DOPE_KEY = "dope_attention_small_router@63696a5d"
DIM = 4168


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def atomic(path: Path, data: bytes):
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def stage_headers():
    specs = {}
    for d in sorted((RUN / "stage").glob("dope-v2-*/dope-stage-specs.jsonl")):
        for line in open(d):
            s = json.loads(line)
            specs[s["view_id"]] = s["meta_json"]["output_headers"]
    return specs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    kernel_sha = sha(KERNEL.read_bytes())
    router_sha = {n: sha((ROUTER / n).read_bytes()) for n in ("router.bundle.json", "router-evidence.json")}
    ob = OVERLAP.read_bytes()
    if sha(ob) != OVERLAP_SHA:
        raise SystemExit("pretraining-overlap receipt sha mismatch")
    overlap, ds_status = {}, {}
    rank = {"Exposed": 2, "Unknown": 1, "Unseen": 0}
    for line in ob.decode().splitlines():
        x = json.loads(line); overlap[x["view_id"]] = x
        st = ((x.get("encoders") or {}).get(DOPE_KEY) or {})
        cur = ds_status.get(x["dataset_hash"])
        if cur is None or rank.get(st.get("status"), 1) > rank.get(cur.get("status"), 1):
            ds_status[x["dataset_hash"]] = st   # exposure is dataset-level; keep the most conservative fold record
    headers = stage_headers()
    reg = json.load(open(RUN / "families/g2/registry.json"))
    ctx2fam = {c: fid for fid, f in reg["families"].items() for c in (f.get("contexts") or [])}
    ds2fam = {m["dataset_hash"]: fid for fid, f in reg["families"].items() for m in (f.get("members") or []) if m.get("dataset_hash")}
    assign = json.load(open(RUN / "holdout/holdout.json"))["assignment"]
    build = Path(a.build)
    ctxs = sorted(p.parent for p in build.glob("shard-*/*/context.json") if (p.parent / ".done").exists())
    mine = [d for i, d in enumerate(ctxs) if i % a.of == a.shard]
    if a.limit:
        mine = mine[: a.limit]
    stage_root = RUN / "tmp" / "dope-stage-v2" / f"{build.name}-s{a.shard}"
    t0 = time.time(); done = skipped = 0; failed = []
    for n, d in enumerate(mine):
        ctx = json.loads((d / "context.json").read_text())
        if ctx.get("published") or "dope.npy" in (ctx.get("files") or {}):
            skipped += 1; continue
        if n % 25 == 0:
            ok, reasons, _ = cpuguard.check()
            while not ok:
                time.sleep(15); ok, reasons, _ = cpuguard.check()
        view, h = ctx["view_id"], ctx["dataset_hash"]
        try:
            raw = gzip.decompress((d / "encoder.jtt1.gz").read_bytes())
            if sha(raw) != ctx["files"]["encoder.jtt1.gz"]["raw_sha256"]:
                raise RuntimeError("encoder JTT1 differs from context.json")
            _, _, X, y = jtt1.train_view(raw)
            block = np.column_stack([X, y])
            hdr = headers.get(view)
            if not hdr or len(hdr) != block.shape[1]:
                hdr = [f"x{j}" for j in range(block.shape[1] - 1)] + ["target"]
            corpus = stage_root / view / "corpus"
            ds = corpus / h
            if ds.parent.parent.exists():
                shutil.rmtree(ds.parent.parent)
            ds.mkdir(parents=True)
            (ds / "meta.json").write_text(json.dumps({"task_type": "regression", "target_column": hdr[-1], "output_headers": hdr,
                                                       "target_is_final_column": True}))
            buf = io.StringIO(); w = csv.writer(buf, lineterminator="\n")
            for row in block:
                w.writerow(["%.17g" % v for v in row])
            (ds / "train.csv").write_text(buf.getvalue())
            out = stage_root / view / "out"; man = stage_root / view / "manifest.json"
            r = subprocess.run([str(KERNEL), "embed-regression-corpus", "--dataset-root", str(corpus), "--output-dir", str(out),
                                "--router-bundle", str(ROUTER / "router.bundle.json"), "--router-evidence", str(ROUTER / "router-evidence.json"),
                                "--method", "attention-small", "--expected-dimension", str(DIM), "--shard-index", "0", "--shard-count", "1",
                                "--jobs", "1", "--determinism-checks", "1", "--manifest-out", str(man)], capture_output=True, text=True, timeout=1800)
            m = json.loads(man.read_text()) if man.exists() else {}
            if r.returncode != 0 or m.get("success") != 1 or m.get("determinism_checked") != 1:
                raise RuntimeError(f"dope exit {r.returncode} success {m.get('success')} det {m.get('determinism_checked')}: {r.stderr[-300:]}")
            emb = json.loads((out / f"{h}_attention-small_{DIM}.json").read_text())
            vec = np.asarray(emb["vector"], dtype="<f4")
            if vec.shape != (DIM,) or not np.all(np.isfinite(vec)):
                raise RuntimeError(f"vector shape {vec.shape} or non-finite")
            if int(emb.get("rows", -1)) != X.shape[0]:
                raise RuntimeError(f"Dope saw {emb.get('rows')} rows, expected {X.shape[0]} outer-train rows")
            # Rebuilt contexts (row-selection rebuilds) carry new view_ids and context shas: fall back to dataset level.
            if view in overlap:
                ov, basis = (overlap[view].get("encoders") or {}).get(DOPE_KEY), "view_id"
            else:
                ov, basis = ds_status.get(h), "dataset_level"
            fam = ctx2fam.get(ctx["context_sha256"]) or ds2fam.get(h)
            npy = io.BytesIO(); np.save(npy, vec, allow_pickle=False); body = npy.getvalue()
            atomic(d / "dope.npy", body)
            ctx["files"]["dope.npy"] = {"sha256": sha(body), "bytes": len(body)}
            ctx["dope"] = {"kernel_sha256": kernel_sha, "commit": "63696a5db246abf9c85343c12a5d8c5522fb864a", "router_sha256": router_sha,
                           "method": "attention-small", "dimension": DIM, "rows_seen": int(emb["rows"]), "features_seen": int(emb["features"]),
                           "determinism_checked": True, "embedding_json_sha256": sha((out / f"{h}_attention-small_{DIM}.json").read_bytes()),
                           "exposure": {"pretraining_overlap": ov, "basis": basis,
                                        "family_id": fam,
                                        "holdout_assignment": assign.get(fam, "no_family"),
                                        "lift_eligible": (ov or {}).get("status") == "Unseen",
                                        "overlap_receipt": f"{OVERLAP}#sha256={OVERLAP_SHA}",
                                        "limitation": "dataset-level evidence; Unknown is never treated as Unseen"},
                           "utc_us": int(time.time() * 1e6)}
            atomic(d / "context.json", (json.dumps(ctx, sort_keys=True) + "\n").encode())
            shutil.rmtree(stage_root / view)
            done += 1
        except Exception as e:
            failed.append({"view_id": view, "error": f"{type(e).__name__}: {e}"[:400]})
    rec = {"schema": "jope.dl-v2-dope-run/v1", "build": str(build), "shard": a.shard, "of": a.of, "expected": len(mine), "done": done,
           "skipped_already": skipped, "failed": failed, "kernel_sha256": kernel_sha, "router_sha256": router_sha,
           "code_sha256": sha(Path(__file__).read_bytes()), "elapsed_s": round(time.time() - t0, 1), "utc_us": int(time.time() * 1e6)}
    (build / f"dope-shard-{a.shard}-of-{a.of}.receipt.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: rec[k] for k in ("shard", "expected", "done", "skipped_already", "elapsed_s")} | {"failed": len(failed)}))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

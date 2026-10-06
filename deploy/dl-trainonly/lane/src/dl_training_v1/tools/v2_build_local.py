"""Provisional v2 train-only build, local and unpublished (L7b values: T1 decoded and re-fitted on outer-train rows).

Per epoch-0 context (scale_check.resolve_context lineage), writes under OUT/<shard>/<view_id>/:
  encoder.jtt1.gz   train-only min-max JTT1 (t1prep.prepare): the encoders' training-side input
  teacher.jtt1.gz   same rows, columns and inner folds, values re-fitted (categories re-coded, target copies dropped)
                    but NOT min-max scaled: the evaluator in patch 0001 scales per fit on its own fit rows
  a0.f32            A0 meta-features of encoder.jtt1 (metafeatures.from_raw)
  context.json      identities, sha256 of every file, t1prep strata, code sha256, values mode, published=false
  .done             written last; a context with .done is skipped (resumable, idempotent)
Per shard: shard-<i>-of-<n>.receipt.json (expected vs done vs failed). No S3, network or GPU; nothing is published.

Usage (inside the lane slice): v2_build_local.py --out DIR --shard I --of N
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from dltrain import cpuguard, metafeatures, scale_check, t1prep, trainonly

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
META_DIR = RUN / "inputs" / "t1-meta"
DATASETS_INV = RUN / "tmp" / "trainonly-v1" / "epoch0-datasets.json"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def atomic(path: Path, data: bytes):
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def code_shas():
    root = Path(__file__).resolve().parents[1]
    return {f: sha((root / f).read_bytes()) for f in ("tools/v2_build_local.py", "dltrain/t1prep.py", "dltrain/trainonly.py",
                                                        "dltrain/metafeatures.py", "dltrain/scale_check.py", "dltrain/jtt1.py")}


def build_one(c, meta, meta_sha, d: Path, codes):
    A, R, T = c["A"], [int(i) for i in c["R"]], [int(i) for i in c["T"]]
    split = t1prep.prepare(A, meta, R, T)
    fitted, rep = t1prep.refit(A, meta, R)
    teacher = trainonly.jtt1_bytes(fitted[R], fitted[T].reshape(-1, fitted.shape[1]), split["inner_labels"])
    enc_gz = gzip.compress(split["jtt1"], compresslevel=3, mtime=0)
    tea_gz = gzip.compress(teacher, compresslevel=3, mtime=0)
    a0 = metafeatures.from_raw(split["jtt1"])[0]
    d.mkdir(parents=True, exist_ok=True)
    atomic(d / "encoder.jtt1.gz", enc_gz)
    atomic(d / "teacher.jtt1.gz", tea_gz)
    atomic(d / "a0.f32", a0)
    row = c["row"]
    ctx = {"schema": "jope.dl-v2-local-context/v1", "view_id": row["view_id"], "dataset_hash": row["dataset_hash"], "fold": int(row["fold"]),
           "context_sha256": row["context_sha256"], "outer_train_indices_sha256": sha(json.dumps(R).encode()),
           "outer_test_indices_sha256": sha(json.dumps(T).encode()), "n_train": len(R), "n_test": len(T),
           "files": {"encoder.jtt1.gz": {"sha256": sha(enc_gz), "raw_sha256": sha(split["jtt1"]), "bytes": len(enc_gz)},
                     "teacher.jtt1.gz": {"sha256": sha(tea_gz), "raw_sha256": sha(teacher), "bytes": len(tea_gz)},
                     "a0.f32": {"sha256": sha(a0), "bytes": len(a0)}},
           "scaler_sha256": sha(trainonly.scaler_bytes(split)), "inner_labels_sha256": sha(np.asarray(split["inner_labels"], np.uint8).tobytes()),
           "inner_loads": split["inner_cv"]["loads"], "t1prep": rep, "t1_meta_sha256": meta_sha, "values_mode": "L7b",
           "published": False, "code_sha256": codes, "utc_us": int(time.time() * 1e6)}
    atomic(d / "context.json", (json.dumps(ctx, sort_keys=True) + "\n").encode())
    atomic(d / ".done", b"ok\n")
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    a = ap.parse_args(argv)
    cpuguard.set_threads(2)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    codes = code_shas()
    src = scale_check.Sources(load_index=False)
    cache = scale_check.DataCache(max_items=4)
    inv = json.loads(DATASETS_INV.read_text())
    rows = sorted(src.included(), key=lambda r: (r["dataset_hash"], int(r["fold"])))
    mine = [r for r in rows if int(r["dataset_hash"][:8], 16) % a.of == a.shard]
    t0 = time.time(); done = skipped = 0; failed = []
    for n, r in enumerate(mine):
        d = out / f"shard-{a.shard}" / r["view_id"]
        if (d / ".done").exists():
            skipped += 1; continue
        if n % 25 == 0:
            ok, reasons, _ = cpuguard.check()
            while not ok:
                print(json.dumps({"guard": "PAUSE", "reasons": reasons}), flush=True); time.sleep(15); ok, reasons, _ = cpuguard.check()
        try:
            c = scale_check.resolve_context(r, src, cache)
            k = inv["meta_key"][r["dataset_hash"]].rsplit("/", 1)[1]
            mb = (META_DIR / f"{k}.json").read_bytes()
            if sha(mb) != k:
                raise RuntimeError("meta sha mismatch")
            build_one(c, json.loads(mb), k, d, codes)
            done += 1
        except Exception as e:  # recorded per context; the shard continues
            failed.append({"view_id": r["view_id"], "error": f"{type(e).__name__}: {e}"[:300]})
        if (done + len(failed)) % 100 == 0:
            print(json.dumps({"shard": a.shard, "progress": done + skipped + len(failed), "of": len(mine), "failed": len(failed),
                              "elapsed_s": round(time.time() - t0)}), flush=True)
    rec = {"schema": "jope.dl-v2-local-shard/v1", "shard": a.shard, "of": a.of, "expected": len(mine), "built_now": done,
           "already_done": skipped, "failed": failed, "complete": done + skipped == len(mine), "elapsed_s": round(time.time() - t0, 1),
           "values_mode": "L7b", "published": False, "code_sha256": codes, "utc_us": int(time.time() * 1e6)}
    atomic(out / f"shard-{a.shard}-of-{a.of}.receipt.json", (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps({k: rec[k] for k in ("shard", "expected", "built_now", "already_done", "complete", "elapsed_s")} | {"failed": len(failed)}))
    return 0 if rec["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())

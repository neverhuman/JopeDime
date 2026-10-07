#!/usr/bin/env python3
"""Fetch (exact VersionId + sha256) and prep every admitted jtt1-encoder-s0 context into one packed store.

Output DIR: store.npz-like files X.f16 (concat rows x p per context, row-major), y.f32, miss.f32, index.json (offsets,
shapes, manifest records). Read-only on S3. Admission: inventory rows with an encoder-s0 object, research_only false,
context not in the G3 list. Fetched gz kept under DIR/jtt1/ (create-only) so a rerun never refetches.
"""
import argparse, concurrent.futures as cf, gzip, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ta_common as C

B = "veox-jopedime-learning-use2"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


_s3 = None
def s3():
    global _s3
    if _s3 is None:
        import boto3, botocore
        _s3 = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "veox-jepson-prod")).client("s3", config=botocore.config.Config(retries={"max_attempts": 8, "mode": "adaptive"}))
    return _s3


def work(args):
    row, jdir = args
    o = row["inputs"]["jtt1-encoder-s0"]; cid = row["context_sha256"]
    path = os.path.join(jdir, o["sha256"] + ".jtt1.gz")
    if os.path.exists(path):
        gz = open(path, "rb").read()
    else:
        gz = s3().get_object(Bucket=B, Key=o["key"], VersionId=o["version_id"])["Body"].read()
        tmp = path + f".tmp{os.getpid()}"; open(tmp, "wb").write(gz); os.replace(tmp, path)
    if C.sha256(gz) != o["sha256"]:
        return cid, None, {"cid": cid, "skipped": "sha256 mismatch"}
    try:
        arrs, rec = C.prep_context(gzip.decompress(gz), cid)
    except Exception as e:
        return cid, None, {"cid": cid, "skipped": f"error {e!r}"[:300]}
    rec.update(dataset_hash=row.get("dataset_hash"), fold=row.get("fold"), rights=row.get("rights"), cut_id=row.get("cut_id"),
               key=o["key"], version_id=o["version_id"], gz_sha256=o["sha256"])
    return cid, arrs, rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inv", required=True); ap.add_argument("--g3", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "jtt1"), exist_ok=True)
    g3 = set(json.load(open(a.g3))); inv = json.load(open(a.inv))
    rows = [r for r in inv if "jtt1-encoder-s0" in r["inputs"]]
    excl = {"g3": sum(r["context_sha256"] in g3 for r in rows), "research_only_object": sum(bool(r["inputs"]["jtt1-encoder-s0"].get("research_only")) for r in rows)}
    rows = [r for r in rows if r["context_sha256"] not in g3 and not r["inputs"]["jtt1-encoder-s0"].get("research_only")]
    rows.sort(key=lambda r: r["context_sha256"])
    if a.limit: rows = rows[: a.limit]
    log("admitted", len(rows), "excluded", excl)
    Xs, ys, ms, recs, skipped = [], [], [], [], []
    with cf.ProcessPoolExecutor(a.workers) as ex:
        for i, (cid, arrs, rec) in enumerate(ex.map(work, [(r, os.path.join(a.out, "jtt1")) for r in rows], chunksize=4)):
            if arrs is None:
                skipped.append(rec)
            else:
                X, y, m = arrs; Xs.append(np.nan_to_num(X, nan=0.0).astype(np.float16).ravel()); ys.append(y); ms.append(m); recs.append(rec)
            if i % 500 == 0: log("prep", i, "/", len(rows))
    xo = np.cumsum([0] + [len(x) for x in Xs]); yo = np.cumsum([0] + [len(y) for y in ys]); mo = np.cumsum([0] + [len(m) for m in ms])
    for r, a0, b0, c0 in zip(recs, xo, yo, mo):
        r["x_off"], r["y_off"], r["m_off"] = int(a0), int(b0), int(c0)
    np.save(os.path.join(a.out, "X.npy"), np.concatenate(Xs)); np.save(os.path.join(a.out, "y.npy"), np.concatenate(ys).astype(np.float32))
    np.save(os.path.join(a.out, "miss.npy"), np.concatenate(ms).astype(np.float32))
    stats = {"admitted": len(rows), "prepped": len(recs), "skipped": len(skipped), "excluded": excl,
             "rows_total_outer_train": int(sum(r["n_train"] for r in recs)), "rows_stored": int(sum(r["n"] for r in recs)),
             "datasets": len({r["dataset_hash"] for r in recs}), "reduced_contexts": sum(r["reduction"] is not None for r in recs),
             "nstore": C.NSTORE, "pstore": C.PSTORE, "phv": C.PHV, "seed": C.SEED, "n_test_nonzero": sum(r["n_test"] != 0 for r in recs)}
    json.dump({"stats": stats, "contexts": recs, "skipped": skipped}, open(os.path.join(a.out, "index.json"), "w"))
    log("DONE", json.dumps(stats))


if __name__ == "__main__":
    main()

"""Create-only filtered copy of a publish, GPU or custody tree, without the contexts of a local SUPERSEDED record.

The publisher publishes every context in a tree, so contexts that must never be published (GROUP-UNION-B: the as-built
v2split contexts of the group-annotated datasets, receipts/trainonly-v2/build/superseded-local-groupdrop-v2split-*.json) are left
out by copying the remaining context directories, byte for byte, into a new tree. Datasets are kept or dropped as a whole
(dataset-level replacement), the source tree is never modified, and a receipt lists every kept and dropped view_id.

Usage: v2_pubset.py --src TREE --dst NEW_TREE --exclude RECORD#SHA256
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--exclude", required=True, help="PATH#SHA256 of a jope.dl-v2-superseded-local/v1 record")
    a = ap.parse_args(argv)
    src, dst = Path(a.src), Path(a.dst)
    rp, rs = a.exclude.split("#", 1)
    rb = Path(rp).read_bytes()
    if sha(rb) != rs:
        raise SystemExit(f"{rp}: sha256 mismatch")
    rec = json.loads(rb)
    if rec.get("schema") != "jope.dl-v2-superseded-local/v1":
        raise SystemExit(f"{rp}: not a superseded-local record")
    drop_views = {r["view_id"] for r in rec["rows"]}
    drop_ds = {r["dataset_hash"] for r in rec["rows"]}
    if dst.exists():
        raise SystemExit(f"{dst}: exists (create-only)")
    kept, dropped = [], []
    ctxs = sorted(glob.glob(f"{src}/shard-*/*/context.json"))
    for p in ctxs:
        d = Path(p).parent
        c = json.loads(Path(p).read_bytes())
        if c["view_id"] in drop_views or c["dataset_hash"] in drop_ds:
            if c["view_id"] not in drop_views:
                raise SystemExit(f"{c['view_id']}: dataset {c['dataset_hash']} is superseded but this view is not in the record")
            dropped.append(c["view_id"]); continue
        if not (d / ".done").exists():
            raise SystemExit(f"{d}: not done")
        kept.append((d, c["view_id"]))
    for d, _ in kept:
        out = dst / d.parent.name / d.name
        shutil.copytree(d, out)
        for f in sorted(os.listdir(d)):
            if sha((d / f).read_bytes()) != sha((out / f).read_bytes()):
                raise SystemExit(f"{out / f}: copy differs")
    receipt = {"schema": "jope.dl-v2-pubset/v1", "src": str(src), "dst": str(dst), "exclude": {"path": rp, "sha256": rs},
               "contexts_in_src": len(ctxs), "kept": len(kept), "dropped": len(dropped), "dropped_view_ids": sorted(dropped),
               "kept_view_ids": sorted(v for _, v in kept), "code_sha256": sha(Path(__file__).read_bytes()),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    body = (json.dumps(receipt, indent=1, sort_keys=True) + "\n").encode()
    with open(dst.parent / f"{dst.name}.pubset-receipt.json", "xb") as fh:      # create-only
        fh.write(body)
    print(json.dumps({"dst": str(dst), "kept": len(kept), "dropped": len(dropped), "receipt_sha256": sha(body)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

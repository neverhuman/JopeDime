"""Create-only SUPERSEDED document for published v2 manifests (RAW-SPLIT-RULING, NONRAW-SPLIT-A).

Lists every context and object key of the named manifests (local copies verified against the publish ledger), bound to the
owner ruling file and the evidence receipt by sha256. Objects stay in S3 untouched; loaders refuse them through the holdout
denylist built with --superseded. Replaces the inline script that built superseded-raw-split-v1 (cd51465f): with the same
inputs this tool reproduces that document byte-exactly (see --check-against).

Usage (lane slice): v2_superseded.py --name superseded-nonraw-split-v1 --manifests a,b,c --reason TEXT --rule TEXT
                    --ruling-name TEXT --ruling-file PATH --evidence PATH [--check-against PATH] [--publish]
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
PUB = RUN / "receipts/trainonly-v2/publish"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--manifests", required=True, help="comma-separated published manifest names")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--rule", required=True)
    ap.add_argument("--ruling-name", required=True)
    ap.add_argument("--ruling-file", required=True)
    ap.add_argument("--evidence", required=True, help="receipt path (absolute, or relative to RUN)")
    ap.add_argument("--check-against", help="an earlier local document this run must reproduce byte-exactly")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location("vp", Path(__file__).resolve().parent / "v2_publish.py")
    vp = importlib.util.module_from_spec(spec); spec.loader.exec_module(vp)
    names = [x for x in a.manifests.split(",") if x]
    ver = {}
    for line in vp.DEFAULT_PUBLISH_LEDGER.read_bytes().splitlines():
        r = json.loads(line)
        if r.get("event") == "verified" and "/manifests/" in r["key"]:
            ver[r["key"]] = r
    ms, ctx, keys, seen = [], set(), set(), set()
    for key, r in sorted(ver.items()):
        name, f = key.split("/manifests/", 1)[1].split("/")
        if name not in names:
            continue
        local = PUB / f"v2-publish-manifest-{name}-{f[:-5]}.json"
        b = local.read_bytes()
        if sha(b) != r["sha256"]:
            raise SystemExit(f"{local}: differs from the ledger's verified record")
        m = json.loads(b); seen.add(name)
        ms.append({"name": name, "key": key, "version_id": r["version_id"], "sha256": r["sha256"], "contexts": len(m["contexts"])})
        for c in m["contexts"]:
            ctx.add(c["view_id"])
            for o in c["objects"]:
                keys.add(o["key"]); keys.add(o["receipt_key"])
    if set(names) - seen:
        raise SystemExit(f"not verified in the publish ledger: {sorted(set(names) - seen)}")
    ruling, ev = Path(a.ruling_file), Path(a.evidence)
    evp = ev if ev.is_absolute() else RUN / ev
    doc = {"schema": "jope.dl-v2-superseded/v1", "status": "SUPERSEDED", "reason": a.reason,
           "ruling": {"name": a.ruling_name, "decision_file": ruling.name, "sha256": sha(ruling.read_bytes())},
           "evidence": {"redteam_receipt" if "redteam" in str(ev) else "receipt": str(ev), ("redteam_sha256" if "redteam" in str(ev) else "sha256"): sha(evp.read_bytes())},
           "rule": a.rule, "manifests": ms, "contexts": sorted(ctx), "object_keys": sorted(keys)}
    if not a.check_against:                      # new documents carry the tool's own hash (PROVENANCE-01)
        doc["code_sha256"] = sha(Path(__file__).read_bytes())
    body = (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode(); dg = sha(body)
    if a.check_against:
        want = Path(a.check_against).read_bytes()
        print(json.dumps({"reproduces": body == want, "sha256": dg, "want_sha256": sha(want)}))
        return 0 if body == want else 1
    local = PUB / f"v2-{a.name}-{dg[:12]}.json"
    if not local.exists():
        local.write_bytes(body)
    out = {"local": str(local), "sha256": dg, "manifests": len(ms), "contexts": len(ctx), "object_keys": len(keys)}
    if a.publish:
        writer, reader = vp.real_clients(); ledger = vp.PublishLedger(vp.DEFAULT_PUBLISH_LEDGER)
        try:
            out["published"] = vp.Publisher(writer, reader, ledger).publish_bytes(
                f"{vp.s3io.V2_PREFIX_KEY}manifests/{a.name}/{dg[:12]}.json", body, "application/json", "manifest", "superseded")
        finally:
            ledger.close()
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

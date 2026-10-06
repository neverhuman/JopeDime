"""Additive holdout-assignment supplement (owner ruling SEALED-LIFT, ENG:watcher delegated 23:38:54Z).

Every published v2 manifest (found as a verified `.../manifests/<name>/<sha12>.json` key in the publish ledger; its local copy
receipts/trainonly-v2/publish/v2-publish-manifest-<name>-<sha12>.json must hash to the ledger's sha256) gets, per context,
the P3 family_id and holdout assignment (train_fit / train_nonpool / dev / heldout / sealed_excluded) from the frozen
holdout.json and the families registry, both pinned by sha256. GPU manifests also get their context2048 leak-control status
(CTX2048-CONTROL): a supplied pilot receipt that verified the composed control, or UNVERIFIED. Two documents are written,
one for production manifests and one for research-only ones; --publish PUTs them create-only (production under
manifests/, research under research-only/manifests/). No existing object is rewritten.

Usage (lane slice): v2_holdout_supplement.py --name holdout-supplement-v1 [--ctx2048 MANIFEST_NAME=PATH#SHA ...] [--publish]
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
PUB = RUN / "receipts" / "trainonly-v2" / "publish"
HOLDOUT = (RUN / "holdout" / "holdout.json", "5b2ecc63")
REGISTRY = (RUN / "families" / "g2" / "registry.json", "1a4f1d41")
SCHEMA = "jope.dl-v2-holdout-assignment-supplement/v1"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_vp():
    spec = importlib.util.spec_from_file_location("v2_publish", Path(__file__).resolve().parent / "v2_publish.py")
    vp = importlib.util.module_from_spec(spec); spec.loader.exec_module(vp)
    return vp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ctx2048", action="append", default=[], help="MANIFEST_NAME=PATH#SHA256 of its passing context2048 control pilot")
    ap.add_argument("--skip", default="", help="comma-separated manifest names left out (e.g. SUPERSEDED sets; listed in the document)")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args(argv)
    skip = {x for x in a.skip.split(",") if x}
    vp = load_vp()
    hb, rb = HOLDOUT[0].read_bytes(), REGISTRY[0].read_bytes()
    if not sha(hb).startswith(HOLDOUT[1]) or not sha(rb).startswith(REGISTRY[1]):
        raise SystemExit("holdout or registry sha256 differs from the frozen P2/P3 pins")
    assign = json.loads(hb)["assignment"]
    reg = json.loads(rb)
    ds2fam = {m["dataset_hash"]: fid for fid, f in reg["families"].items() for m in (f.get("members") or []) if m.get("dataset_hash")}
    ctl = {}
    for spec in a.ctx2048:
        n, rest = spec.split("=", 1); p, want, *flag = rest.split("#")
        if sha(Path(p).read_bytes()) != want:
            raise SystemExit(f"{p}: sha256 mismatch")
        ctl[n] = {"status": "VERIFIED", "pilot_receipt": f"{p}#sha256={want}"}
        if "retro" in flag:
            ctl[n]["note"] = "pilot (with the fixed context2048 control) ran AFTER this set's production encode; same code path, disclosed"
        elif "proxy" in flag:
            ctl[n]["note"] = "fixed control verified on this set's original 3 pilot contexts after publication"
    verified = {}
    for line in vp.DEFAULT_PUBLISH_LEDGER.read_bytes().splitlines():
        r = json.loads(line)
        if r.get("event") == "verified" and "/manifests/" in r["key"]:
            verified[r["key"]] = r
    docs = {"production": [], "research": []}
    for key, r in sorted(verified.items()):
        name, file = key.split("/manifests/", 1)[1].split("/")
        local = PUB / f"v2-publish-manifest-{name}-{file[:-5]}.json"
        if name.startswith(("holdout-supplement", "holdout-denylist", "s1-input-list")) or not local.exists():
            continue                                   # not a context manifest (supplements, denylists, S1 lists, provenance)
        if name in skip:
            continue                                   # superseded set: not described again (see skipped_manifests)
        mb = local.read_bytes()
        if sha(mb) != r["sha256"]:
            raise SystemExit(f"{local}: sha256 differs from the ledger's verified record for {key}")
        m = json.loads(mb)
        rows, counts = [], {}
        for c in m["contexts"]:
            fam = ds2fam.get(c["dataset_hash"])
            asg = assign.get(fam, "no_family")
            counts[asg] = counts.get(asg, 0) + 1
            rows.append({"view_id": c["view_id"], "dataset_hash": c["dataset_hash"], "fold": c.get("fold"), "family_id": fam, "assignment": asg})
        has_ctx = any(o["component"] == "context2048" for c in m["contexts"] for o in c.get("objects", []))
        entry = {"manifest_key": key, "manifest_version_id": r["version_id"], "manifest_sha256": r["sha256"], "contexts": len(rows),
                 "assignment_counts": counts, "rows": rows}
        if has_ctx:
            entry["context2048_leak_control"] = ctl.get(name) or {"status": "UNVERIFIED at publish",
                                                                  "note": "pilot must-fail control did not cover context2048; see CTX2048-CONTROL"}
        docs["research" if "/research-only/" in key or name.startswith("rights-research") else "production"].append(entry)
    out = {}
    for kind, entries in docs.items():
        doc = {"schema": SCHEMA, "name": f"{a.name}-{kind}", "kind": kind, "rule": "SEALED-LIFT: sealed_excluded contexts never count in any "
               "development lift number; trainers refuse them; scored once in the final P3 held-out evaluation",
               "sources": {"holdout": {"path": str(HOLDOUT[0]), "sha256": sha(hb)}, "registry": {"path": str(REGISTRY[0]), "sha256": sha(rb)}},
               "manifests": entries, "code_sha256": sha(Path(__file__).read_bytes())}
        if skip:
            doc["skipped_manifests"] = sorted(skip)
        body = (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode()
        digest = sha(body)
        local = PUB / f"v2-holdout-supplement-{a.name}-{kind}-{digest[:12]}.json"
        if not local.exists():
            local.write_bytes(body)
        out[kind] = {"local": str(local), "sha256": digest, "manifests": len(entries), "contexts": sum(e["contexts"] for e in entries),
                     "sealed_excluded": sum(e["assignment_counts"].get("sealed_excluded", 0) for e in entries), "body": body}
    if a.publish:
        writer, reader = vp.real_clients()
        ledger = vp.PublishLedger(vp.DEFAULT_PUBLISH_LEDGER)
        try:
            pub = vp.Publisher(writer, reader, ledger)
            for kind, o in out.items():
                key = f"{vp.s3io.V2_PREFIX_KEY}{'research-only/' if kind == 'research' else ''}manifests/{a.name}-{kind}/{o['sha256'][:12]}.json"
                o["published"] = pub.publish_bytes(key, o["body"], "application/json", "manifest", f"supplement:{kind}")
        finally:
            ledger.close()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "body"} for k, v in out.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""PAYLOAD-STRIP-PUBLISH quarantine list: every unstripped JTT1 payload already published (inputs/jtt1-encoder|teacher/).

From the local v2 publish ledger (append-only JSONL, read only): every `verified` object record whose key matches the
quarantine rule of dltrain/quarantine.py, with VersionId, sha256, bytes, view_id, ledger time, its receipt object (key,
VersionId) and the publish run receipt (receipts/trainonly-v2/publish/v2-publish-real-*.json) whose time window holds it.
Writes ONE create-only local document (schema jope.dl-v2-quarantine/v1) named v2-quarantine-<name>-<sha256[:12]>.json under
RUN/receipts/trainonly-v2/quarantine/ (dltrain/quarantine.QUARANTINE_DIR, where the publisher and loaders read it).
LOCAL ONLY: it is not published until the owner rules; nothing in S3 is read, written, moved or deleted.

Usage (lane slice, offline): v2_quarantine_unstripped.py --name unstripped-v1 [--ledger PATH] [--out-dir DIR]
"""
import argparse
import collections
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dltrain import quarantine  # noqa: E402

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
PUB = RUN / "receipts/trainonly-v2/publish"
LEDGER = PUB / "v2-publish-ledger.jsonl"
RULING = {"name": "PAYLOAD-STRIP-PUBLISH (ENG:watcher delegated, 2026-10-06 06:52 MT)",
          "file": "/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261006T125305Z.json",
          "sha256": "240e0af5521c03f95157c447916d61f3e49dffa2e564a72bc097f811df957baf"}


def sha(b) -> str:
    return hashlib.sha256(b).hexdigest()


def runs_index(pub: Path):
    out = []
    for p in sorted(pub.glob("v2-publish-real-*.json")):
        b = p.read_bytes()
        try:
            d = json.loads(b)
        except ValueError:
            continue
        if isinstance(d.get("started_utc_us"), int) and isinstance(d.get("finished_utc_us"), int):
            out.append({"receipt": p.name, "sha256": sha(b), "build": d.get("build"), "start": d["started_utc_us"], "end": d["finished_utc_us"]})
    return out


def build(ledger: Path, pub: Path, name: str) -> dict:
    data = ledger.read_bytes()
    if data and not data.endswith(b"\n"):
        raise SystemExit(f"{ledger}: ends in a partial line")
    objs, receipts = {}, {}
    for i, line in enumerate(data.splitlines()):
        r = json.loads(line)
        if r.get("event") != "verified" or not isinstance(r.get("key"), str):
            continue
        k = r["key"]
        if quarantine.rule_match(k):
            prev = objs.get(k)
            if prev is not None and (prev["version_id"], prev["sha256"]) != (r["version_id"], r["sha256"]):
                raise SystemExit(f"{k}: two different verified records")
            objs[k] = dict(r, line=i)
        elif "/receipts/jtt1-encoder/" in k or "/receipts/jtt1-teacher/" in k:
            receipts[k] = r
    runs = runs_index(pub)
    entries = []
    for k, r in sorted(objs.items()):
        comp = next(c for c in quarantine.QUARANTINED_COMPONENTS if f"/inputs/{c}/" in k)
        rk = k.replace(f"/inputs/{comp}/", f"/receipts/{comp}/").rsplit(".jtt1.gz", 1)[0] + ".json"
        rr = receipts.get(rk)
        run = [x for x in runs if x["start"] <= r["utc_us"] <= x["end"]]
        entries.append({"key": k, "version_id": r["version_id"], "sha256": r["sha256"], "bytes": r["bytes"], "component": comp,
                        "view_id": r.get("view_id"), "outcome": r.get("outcome"), "ledger_utc_us": r["utc_us"], "ledger_line": r["line"],
                        "receipt": {"key": rk, "version_id": rr["version_id"], "sha256": rr["sha256"]} if rr else None,
                        "run": {"receipt": run[0]["receipt"], "sha256": run[0]["sha256"], "build": run[0]["build"]} if len(run) == 1
                        else ({"ambiguous": [x["receipt"] for x in run]} if run else None)})
    by_comp = collections.Counter(e["component"] for e in entries)
    by_run = collections.Counter((e["run"] or {}).get("receipt", "unattributed") for e in entries)
    return {"schema": quarantine.SCHEMA, "name": name, "reason": quarantine.REASON, "ruling": RULING,
            "rule": "every verified object under v2-trainonly/(research-only/)?inputs/jtt1-encoder|jtt1-teacher/ (dltrain/quarantine.rule_match); "
                    "the publisher and the loaders refuse these keys; stripped payloads publish only under inputs/jtt1-*-s0/",
            "status": "LOCAL ONLY: not published until the owner rules; no S3 object is read, written, moved or deleted",
            "ledger": {"path": str(ledger), "sha256": sha(data), "bytes": len(data), "lines": data.count(b"\n")},
            "counts": {"entries": len(entries), "by_component": dict(by_comp), "bytes": sum(e["bytes"] for e in entries),
                       "with_receipt": sum(1 for e in entries if e["receipt"]), "by_run": dict(by_run),
                       "first_utc_us": min((e["ledger_utc_us"] for e in entries), default=None),
                       "last_utc_us": max((e["ledger_utc_us"] for e in entries), default=None)},
            "entries": entries, "code_sha256": {"tools/v2_quarantine_unstripped.py": sha(Path(__file__).read_bytes()),
                                                "dltrain/quarantine.py": sha((Path(__file__).resolve().parents[1] / "dltrain/quarantine.py").read_bytes())}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--pub", default=str(PUB))
    ap.add_argument("--out-dir", default=str(quarantine.QUARANTINE_DIR))
    a = ap.parse_args(argv)
    doc = build(Path(a.ledger), Path(a.pub), a.name)
    doc["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    body = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    out = Path(a.out_dir) / f"v2-quarantine-{a.name}-{sha(body)[:12]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "xb") as f:      # create-only
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(out, 0o444)
    print(json.dumps({"path": str(out), "sha256": sha(body), "counts": {k: v for k, v in doc["counts"].items() if k != "by_run"}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

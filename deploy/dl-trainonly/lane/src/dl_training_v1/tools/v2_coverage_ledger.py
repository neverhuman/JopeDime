"""Coverage ledger: one row per epoch-0 context (4,873) with its bucket, train-only route, replacement contexts, published
sets and current state, plus per-dataset and per-bucket rollups. Built only from local receipts, manifests and trees; no
network. States: published (named manifests), publishing / encoding / building (artifacts present, manifest not yet),
rights_held, not_rebuildable (with the builder's reason).

Usage: v2_coverage_ledger.py [--out-dir RUN/receipts/trainonly-v2/coverage]
"""
import argparse
import collections
import csv
import glob
import hashlib
import json
import time
from pathlib import Path

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
TV2 = RUN / "receipts/trainonly-v2"
PUB = TV2 / "publish"
CAP = {"row_policy:stratified_cap_to_final_max", "original_size.downsampled"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def manifests():
    """name -> set of view_ids, from the local copies of every manifest verified in the publish ledger."""
    out, keys = {}, {}
    for r in jl(PUB / "v2-publish-ledger.jsonl"):
        if r.get("event") == "verified" and "/manifests/" in r["key"] and "/manifests/holdout-supplement" not in r["key"] \
                and "/manifests/s1-input-list" not in r["key"]:
            name, f = r["key"].split("/manifests/", 1)[1].split("/")
            local = PUB / f"v2-publish-manifest-{name}-{f[:-5]}.json"
            if local.exists() and sha(local.read_bytes()) == r["sha256"]:
                out[name] = {c["view_id"] for c in json.loads(local.read_text())["contexts"]}
                keys[name] = r["key"]
    return out, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(TV2 / "coverage"))
    a = ap.parse_args()
    lab = {x["view_id"]: x for x in jl(TV2 / "row-selection-leak-v2-20261005T200546Z.jsonl")}
    acq = json.loads((TV2 / "acquisition-selection-20261005T2045Z.json").read_text())
    held = {x["dataset_hash"]: x["host"] for x in jl(TV2 / "rights-gate-no-license-20261005T180254Z.jsonl")}
    scope = set(json.loads((RUN / "tmp/rights-research-datasets.json").read_text()))
    man, mkeys = manifests()
    # replacements: T1 overlay (per context) and raw builds (per dataset)
    t1_rep = {}
    for r in jl(RUN / "epochs/epoch-001-rowsel-overlay.jsonl"):
        if r.get("kind") == "replace":
            t1_rep[r["key"]["view_id"]] = r["replacement"]["context_json"]["path"].split("/")[-2]
    raw_rep = collections.defaultdict(list)          # epoch-0 view -> new raw views
    raw_trees = {"ROWSEL-RAW-v1": "raw", "ROWSEL-RAW-v1-coverage": "raw-research", "ROWSEL-RAW-v1-nonnumeric": "raw-nonnumeric"}
    raw_ds_tree = {}
    for tree, kind in raw_trees.items():
        for p in glob.glob(str(RUN / "build-v2" / tree / "shard-*/*/context.json")):
            if not (Path(p).parent / ".done").exists():
                continue
            c = json.loads(Path(p).read_text())
            raw_ds_tree[c["dataset_hash"]] = kind
            for e in (c.get("epoch0_replaced") or {}).get("contexts", []):
                raw_rep[e["view_id"]].append(c["view_id"])
    cov = json.loads(sorted(TV2.glob("build/rowsel-raw-coverage-summary-*.json"))[-1].read_text())
    reasons = {}
    for b in cov["buckets"].values():
        for x in b.get("datasets_not_rebuilt") or []:
            if isinstance(x, dict):
                reasons[x["dataset_hash"]] = x.get("reason")
    nn = {}
    nnp = TV2 / "coreset-nonnumeric/all185-nonnumeric.jsonl"
    if nnp.exists():
        for x in jl(nnp):
            nn[x.get("dataset_hash")] = x.get("outcome") or x.get("tier") or x.get("status")
    t1fb = {x["dataset_hash"] for x in cov["buckets"]["incoming_acquisition_capped_kmeans"]["datasets_not_rebuilt"]}
    gpu_out = {d: {Path(p).parent.name for p in glob.glob(str(RUN / "build-v2" / d / "out/*/.done"))}
               for d in ("GPU-rowsel-raw-v1", "GPU-rowsel-t1-fallback-v1", "RIGHTS-RESEARCH-raw-v1/gpu")}
    rows = []
    for v, x in sorted(lab.items()):
        h, cohort, labels = x["dataset_hash"], x["cohort"], tuple(x["row_selection_leak"])
        cap = bool(CAP & set((acq.get(h) or {}).get("selection_markers") or []))
        if "all_rows_coreset" in labels:
            bucket = "frozen_coreset"
        elif "kmeans" in labels:
            bucket = ("incoming_kmeans_capped" if cap else "incoming_kmeans_uncapped") if cohort == "incoming" else "frozen_kmeans"
        else:
            bucket = f"acq_capped_{cohort}" if cap else "clean"
        is_held, in_scope = h in held, h in scope
        route, state, reason, repl = None, None, None, []
        pubs = sorted(n for n, vs in man.items() if v in vs)
        if v in t1_rep:
            repl = [t1_rep[v]]; route = "T1 rowsel" + (" (research-only)" if in_scope and is_held else "")
            if h in t1fb:
                route = "T1 fallback (acquisition_cap)"
            pubs = sorted(n for n, vs in man.items() if repl[0] in vs)
        if v in raw_rep:
            repl = sorted(raw_rep[v]); route = {"raw": "raw", "raw-research": "raw (research-only)", "raw-nonnumeric": "raw + nonnumeric"}[raw_ds_tree[h]]
            pubs = sorted(n for n, vs in man.items() if any(r in vs for r in repl))
        if bucket == "clean":
            route = "none needed" + (" (research-only)" if is_held else "")
        if pubs:
            gpu = any("gpu" in n for n in pubs)
            state = "published (CPU+GPU)" if gpu else "published (CPU); GPU pending"
        elif repl:
            if route.startswith("T1 fallback"):
                state = "encoding (GPU chain, xbabe1)" if not (set(repl) & gpu_out["GPU-rowsel-t1-fallback-v1"]) else "encoded, publish pending"
            elif route == "raw (research-only)":
                state = "encoding (GPU chain, xbabe1)" if not (set(repl) & gpu_out["RIGHTS-RESEARCH-raw-v1/gpu"]) else "encoded, publish pending"
            elif route == "raw + nonnumeric":
                state = "built (nonnumeric extension); Dope/GPU pending"
            else:
                state = "publishing (CPU) / GPU encoded, red-team gate" if set(repl) & gpu_out["GPU-rowsel-raw-v1"] else "publishing"
        elif is_held and not in_scope:
            state, reason = "rights_held", f"no established licence ({held[h]})"
        elif bucket == "frozen_coreset" and nn.get(h) and nn.get(h) != "unmatched" and h not in raw_ds_tree:
            state, route, reason = "building (nonnumeric extension)", "raw + nonnumeric", f"nonnumeric tier: {nn.get(h)}"
        else:
            state, reason = "not_rebuildable", reasons.get(h) or (f"nonnumeric: {nn.get(h)}" if nn.get(h) else "see coverage receipt")
        rows.append({"view_id": v, "dataset_hash": h, "fold": x["fold"], "cohort": cohort, "labels": "|".join(labels), "bucket": bucket,
                     "rights": "research-only" if (is_held and in_scope) else ("held" if is_held else "licensed"), "route": route,
                     "replacements": "|".join(repl), "published_in": "|".join(pubs), "state": state, "reason": reason})
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    with open(out / f"coverage-ledger-{stamp}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    by_bucket = collections.defaultdict(collections.Counter)
    for r in rows:
        by_bucket[r["bucket"]][r["state"].split(" (")[0] if r["state"].startswith("published") else r["state"]] += 1
    ds = collections.defaultdict(lambda: {"contexts": 0, "states": collections.Counter()})
    for r in rows:
        ds[r["dataset_hash"]]["contexts"] += 1; ds[r["dataset_hash"]]["states"][r["state"]] += 1
    doc = {"schema": "jope.dl-v2-coverage-ledger/v1", "utc": stamp, "epoch0_contexts": len(rows),
           "by_bucket": {b: dict(c) for b, c in sorted(by_bucket.items())},
           "by_state": dict(collections.Counter(r["state"] for r in rows)),
           "datasets": len(ds), "manifests": mkeys, "csv": str(out / f"coverage-ledger-{stamp}.csv"),
           "code_sha256": sha(Path(__file__).read_bytes())}
    (out / f"coverage-ledger-{stamp}.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: doc[k] for k in ("epoch0_contexts", "by_state", "by_bucket", "datasets")}, indent=1))


if __name__ == "__main__":
    main()

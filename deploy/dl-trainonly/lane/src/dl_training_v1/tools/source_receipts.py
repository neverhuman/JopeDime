"""Per-dataset source receipts for the v2 train-only rebuild (G22 packet item 2, Codex root 16:38Z/16:57Z asks).

For every epoch-0 included context, scale_check.resolve_context re-walks the verified lineage (prepared.json ->
research manifest -> data object -> membership -> outer split; every hop sha256- and byte-checked). Per dataset the
receipt then records: the T1 meta.json identity and the fitted steps it declares, the T1 train/test blob identities,
the original source's declared identity and whether that file exists on this host (size and sha256 of the local copy),
and the transform scopes as v1 built them next to the v2 train-only plan.

Bounded and local: no S3 call, no network, no GPU. Writes create-only files under --out:
  source-receipts.jsonl   one JSON line per dataset
  source-summary.json     counts, availability inventory and the sha256 of the jsonl
"""
import argparse
import collections
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dltrain import common, cpuguard, scale_check

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
META_DIR = RUN / "inputs" / "t1-meta"
DATASETS_INV = RUN / "tmp" / "trainonly-v1" / "epoch0-datasets.json"
FROZEN_DISCOVERY = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-20261001/discovery/admitted.jsonl.gz")
FROZEN_BLOBS = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-20261001/cache/blobs")
INCOMING_ITEMS = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-incoming-intake-v1/items")

V2_PLAN = {
    "feature_scaling": "re-fitted per dataset and outer fold on outer-train rows (trainonly.fit_minmax)",
    "target_scaling": "re-fitted per dataset and outer fold on outer-train rows",
    "category_codes": "decoded through meta.json and re-coded on outer-train categories (t1prep)",
    "feature_selection": "exact/affine target copies detected on outer-train rows (t1prep.target_copy_columns)",
    "inner_folds": "assigned from train-only-scaled outer-train rows; each inner fold refits on its inner-train rows",
    "numeric_imputation": "not removable from T1 values: counted per context (L7b) or refit from original sources (L7a)",
}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def t1_blobs():
    """dataset_hash -> {train, test} blob sha256 from the frozen discovery and the incoming catalog rows."""
    import gzip
    out = {}
    with gzip.open(FROZEN_DISCOVERY, "rt") as fh:
        for line in fh:
            d = json.loads(line)
            files = (d.get("source") or {}).get("files") or {}
            out[d["dataset_hash"]] = {k: (files.get(k) or {}).get("key", "").rsplit("/", 1)[-1] or None for k in ("train", "test", "meta")}
    for p in INCOMING_ITEMS.glob("*/catalog-row.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        f = d.get("files") or {}
        out.setdefault(d.get("dataset_hash"), {k: (f.get(k) or {}).get("sha256") for k in ("train", "test", "meta")})
    return out


def meta_summary(meta):
    fe = meta.get("feature_encodings") or {}
    acq = meta.get("acquisition") or {}
    return {
        "feature_scaling_method": (meta.get("feature_scaling") or {}).get("method"),
        "target_scaling": bool(meta.get("target_scaling")),
        "numeric_impute_columns": sum(1 for v in fe.values() if v.get("kind") == "numeric" and v.get("impute_value") is not None),
        "categorical_columns": sum(1 for v in fe.values() if v.get("kind") == "categorical_codes"),
        "downsampling": (meta.get("downsampling") or {}).get("method"),
        "t1_split": (meta.get("sampling") or {}).get("method"),
        "acquisition_tool": acq.get("tool"),
        "source_platform": meta.get("source_platform") or acq.get("source_platform"),
        "source_url": meta.get("source_url") or acq.get("source_url"),
        "source_size_bytes": meta.get("source_size_bytes") or acq.get("source_size_bytes"),
        "source_rows": meta.get("source_rows") or acq.get("source_rows"),
        "source_path": meta.get("source_path") or meta.get("staging_path") or acq.get("staging_path"),
    }


def source_availability(ms):
    p = ms.get("source_path")
    if not p:
        return {"checked": ["this host"], "local": False, "reason": "no source_path in meta"}
    if not os.path.isfile(p):
        return {"checked": ["this host"], "local": False, "reason": "absent on this host"}
    size = os.path.getsize(p)
    want = ms.get("source_size_bytes")
    return {"checked": ["this host"], "local": True, "bytes": size, "size_matches_meta": (want is None) or size == want,
            "sha256": sha_file(p)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N datasets only (smoke run)")
    a = ap.parse_args(argv)
    cpuguard.set_threads(4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    src = scale_check.Sources(load_index=False)
    cache = scale_check.DataCache(max_items=4)
    inv = json.loads(DATASETS_INV.read_text())
    blobs = t1_blobs()
    rows = sorted((r for r in src.included()), key=lambda r: (r["dataset_hash"], int(r["fold"])))
    by_ds = collections.defaultdict(list)
    for r in rows:
        by_ds[r["dataset_hash"]].append(r)
    datasets = sorted(by_ds)
    if a.limit:
        datasets = datasets[: a.limit]
    counts = collections.Counter()
    jl = out / "source-receipts.jsonl"
    with open(jl, "x") as fh:
        for n, h in enumerate(datasets):
            if n % 50 == 0:
                ok, reasons, _ = cpuguard.check()
                while not ok:
                    print(json.dumps({"guard": "PAUSE", "reasons": reasons}), flush=True); time.sleep(15); ok, reasons, _ = cpuguard.check()
            mkey = inv["meta_key"][h].rsplit("/", 1)[1]
            meta_b = (META_DIR / f"{mkey}.json").read_bytes()
            if common.sha256_bytes(meta_b) != mkey:
                raise SystemExit(f"{h}: cached meta.json sha mismatch")
            ms = meta_summary(json.loads(meta_b))
            folds, errors = [], []
            for r in by_ds[h]:
                try:
                    c = scale_check.resolve_context(r, src, cache)
                except Exception as e:  # recorded, never swallowed: the dataset is marked unresolved
                    errors.append({"view_id": r["view_id"], "error": f"{type(e).__name__}: {e}"[:300]})
                    continue
                folds.append({"view_id": r["view_id"], "fold": int(r["fold"]), "context_sha256": r["context_sha256"],
                              "n_train": len(c["R"]), "n_test": len(c["T"]),
                              "outer_train_indices_sha256": hashlib.sha256(json.dumps([int(i) for i in c["R"]]).encode()).hexdigest(),
                              "outer_test_indices_sha256": hashlib.sha256(json.dumps([int(i) for i in c["T"]]).encode()).hexdigest(),
                              "refs": c["refs"], "v1_input_superseded": True})
            avail = source_availability(ms)
            rec = {"schema": "jope.dl-trainonly-source-receipt/v1", "dataset_hash": h, "cohort": by_ds[h][0]["cohort"],
                   "folds": folds, "unresolved": errors,
                   "t1": {"meta_blob_sha256": mkey, "train_blob_sha256": (blobs.get(h) or {}).get("train"),
                          "test_blob_sha256": (blobs.get(h) or {}).get("test"),
                          "frozen_blobs_local": all((FROZEN_BLOBS / str((blobs.get(h) or {}).get(k))).is_file() for k in ("train", "test"))},
                   "declared": ms, "source_availability": avail,
                   "v1_scopes": {"t1_feature_scaling": ms["feature_scaling_method"], "t1_imputation": "acquisition tool, fitted before sampling",
                                 "t1_category_codes": "acquisition tool, all source rows", "research_2k": "T1 lines verbatim; sampler selects rows using all rows",
                                 "feature_selection": "audit_target_leakage.py over train and test rows",
                                 "inner_folds": "prepare_fold0 digest tie-break on all-row-scaled bytes",
                                 "gp_search": "test-primary fitness (TEST-STEER-01)"},
                   "v2_plan": V2_PLAN}
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            counts["datasets"] += 1; counts["contexts_resolved"] += len(folds); counts["contexts_unresolved"] += len(errors)
            counts["source_local"] += bool(avail.get("local")); counts["cohort_" + rec["cohort"]] += 1
            counts["has_categorical"] += ms["categorical_columns"] > 0; counts["has_numeric_impute"] += ms["numeric_impute_columns"] > 0
            counts["downsampled"] += bool(ms["downsampling"]); counts["scaled_all_rows"] += "cleaned source rows" in str(ms["feature_scaling_method"])
            counts["source_url"] += bool(ms["source_url"])
    summary = {"schema": "jope.dl-trainonly-source-summary/v1", "utc_us": int(time.time() * 1e6), "elapsed_s": round(time.time() - t0, 1),
               "counts": dict(counts), "jsonl": {"path": str(jl), "sha256": sha_file(jl), "bytes": jl.stat().st_size},
               "inputs": {"epoch": src.epoch_ref, "codex_packet": src.packet_ref, "datasets_inventory": str(DATASETS_INV)},
               "availability_scope": "this host only (xbabe2 local paths); other hosts and archive buckets were NOT checked",
               "policy": {"s3_calls": 0, "network": False, "gpu": False},
               "code_sha256": {f: sha_file(Path(__file__).resolve().parents[1] / f) for f in
                               ("tools/source_receipts.py", "dltrain/scale_check.py", "dltrain/common.py")}}
    (out / "source-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("counts", "elapsed_s")}))
    return 0 if counts["contexts_unresolved"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

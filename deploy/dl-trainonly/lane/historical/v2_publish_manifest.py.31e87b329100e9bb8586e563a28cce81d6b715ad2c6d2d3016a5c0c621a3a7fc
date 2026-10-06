"""Publish-set manifest for a v2 train-only publish: binds every published context to the row-selection, acquisition-cap,
rights and pretraining-exposure receipts it was admitted under, and reconciles the full 4,873-context epoch-0 cohort.

The publishable set is recomputed here from the receipts, independently of the tree assembly:
  labels      row-selection-leak-v2 (per context) plus acquisition_cap when the dataset's T1 metadata records an
              acquisition-time row cap (acq_selection.json markers row_policy:stratified_cap_to_final_max or
              original_size.downsampled)
  held        dataset listed in the rights gate (no licence; L7a rights hold, excluded and counted)
  publishable labels == ["clean"] and not held
It must equal the build tree's context set exactly, every published context's labels must equal its context.json labels,
and every listed object and receipt must have a verified record in the publish ledger (read read-only, not locked).
Superseded label receipts are pinned alongside the current one; nothing is edited or dropped.

--publish additionally PUTs the manifest create-only to v2-trainonly/manifests/<name>/<sha12>.json through the v2 writer
(publish ledger + byte-budget ledger, exact-version readback). Without it, nothing leaves the host.

Usage (lane slice): v2_publish_manifest.py --build TREE --name clean-v1 [--publish]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
TV2 = RUN / "receipts" / "trainonly-v2"
AUDIT = Path("/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts")
SCHEMA = "jope.dl-v2-publish-manifest/v1"
CAP_MARKERS = {"row_policy:stratified_cap_to_final_max", "original_size.downsampled"}

# Inputs pinned by content. A mismatch stops the run; a new receipt needs a code change, not a silent swap.
PINS = {
    "row_selection_leak_v2": (TV2 / "row-selection-leak-v2-20261005T200546Z.jsonl",
                              "2196d4e8a504f9a3a41c5fc910d588353ac54fb6ac538dfccc8658780c1f6905"),
    "acquisition_selection": (TV2 / "acquisition-selection-20261005T2045Z.json",
                              "2757fef11544a5326c9f9375a754628ac8b53a9bfe5d5a7f04cbd57dacd0f2d2"),
    "rights_gate_no_license": (TV2 / "rights-gate-no-license-20261005T180254Z.jsonl",
                               "23fc60527a6ef6616d990042393359057ab32472a2d1dd929d81ed33ffe34519"),
    "pretraining_overlap": (TV2 / "pretraining-overlap-20261005T203956Z.jsonl",
                            "5f16216d2350ef30c06c570c0f33cbf8957771d1023f11f6f11ba0523405d24a"),
    "l7a_build_summary": (TV2 / "l7a-build-summary-20261005T204452Z.json",
                          "672d1db7d89bdb73c4c8c15adef84a457a0f7120cd14f9db4453e6fce24290fb"),
    "owner_decisions_typed": (AUDIT / "owner-decisions-20261005T183423Z.json", None),
    "owner_decisions_rowsel": (AUDIT / "owner-decisions-20261005T203324Z.json", None),
}
SUPERSEDED = {  # retained, not used for admission
    "row_selection_leak_v1_1853": TV2 / "row-selection-leak-20261005T185318Z.jsonl",
    "row_selection_leak_v1_1855": TV2 / "row-selection-leak-20261005T185553Z.jsonl",
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pinned():
    out, blobs = {}, {}
    for name, (p, want) in PINS.items():
        b = p.read_bytes()
        got = sha(b)
        if want and got != want:
            raise SystemExit(f"{name}: {p} sha256 {got} != pinned {want}")
        out[name] = {"path": str(p), "sha256": got, "bytes": len(b)}
        blobs[name] = b
    sup = {n: {"path": str(p), "sha256": sha(p.read_bytes()), "status": "superseded_retained"} for n, p in SUPERSEDED.items()}
    return out, sup, blobs


RIGHTS_SCOPE = RUN / "tmp" / "rights-research-datasets.json"   # RIGHTS-RESEARCH: Rdatasets/UCI/CDC rights-held datasets


def classify(blobs, rights_research=None):
    rows = [json.loads(l) for l in blobs["row_selection_leak_v2"].decode().splitlines() if l.strip()]
    acq = json.loads(blobs["acquisition_selection"])
    held = {json.loads(l)["dataset_hash"] for l in blobs["rights_gate_no_license"].decode().splitlines() if l.strip()}
    cls = {}
    for r in rows:
        h = r["dataset_hash"]
        labels = list(r["row_selection_leak"])
        if CAP_MARKERS & set((acq.get(h) or {}).get("selection_markers") or []) and "acquisition_cap" not in labels:
            labels.append("acquisition_cap")
        is_held = h in held
        if labels != ["clean"]:
            disp = "not_clean_row_selection" if set(labels) - {"clean", "acquisition_cap"} else "acquisition_cap"
        elif rights_research is not None:   # research-only manifest: only rights-held datasets inside the RIGHTS-RESEARCH scope
            disp = "published" if is_held and h in rights_research else ("held_no_license" if is_held else "licensed_published_elsewhere")
        elif is_held:
            disp = "held_no_license"
        else:
            disp = "published"
        cls[r["view_id"]] = {"dataset_hash": h, "fold": r["fold"], "cohort": r.get("cohort"), "labels": labels, "held": is_held,
                             "disposition": disp}
    return cls


def read_ledger(path: Path):
    verified, done = {}, set()
    for line in path.read_bytes().splitlines():
        rec = json.loads(line)
        if rec.get("event") == "verified":
            prev = verified.get(rec["key"])
            if prev and (prev["sha256"], prev["bytes"], prev["version_id"]) != (rec["sha256"], rec["bytes"], rec["version_id"]):
                raise SystemExit(f"ledger has two verified records for {rec['key']}")
            verified[rec["key"]] = rec
        elif rec.get("event") == "context_done":
            done.add(rec["view_id"])
    return verified, done


def build(tree: Path, name: str, ledger_path: Path, rights_research: bool = False):
    import importlib.util
    spec = importlib.util.spec_from_file_location("v2_publish", Path(__file__).resolve().parent / "v2_publish.py")
    vp = importlib.util.module_from_spec(spec); spec.loader.exec_module(vp)
    pins, sup, blobs = pinned()
    rr = None
    if rights_research:
        rb = RIGHTS_SCOPE.read_bytes(); rr = set(json.loads(rb))
        pins["rights_research_scope"] = {"path": str(RIGHTS_SCOPE), "sha256": sha(rb), "bytes": len(rb)}
    cls = classify(blobs, rr)
    plans, refused = vp.scan_build(tree)
    if refused:
        raise SystemExit(f"{len(refused)} contexts in the tree are refused by the publisher scan; first {refused[0]}")
    tree_views = {p["view_id"] for p in plans}
    want = {v for v, c in cls.items() if c["disposition"] == "published"}
    if tree_views != want:
        raise SystemExit(f"tree/recomputed mismatch: only-tree {sorted(tree_views - want)[:5]} only-receipts {sorted(want - tree_views)[:5]}"
                         f" ({len(tree_views - want)}/{len(want - tree_views)})")
    verified, done = read_ledger(ledger_path)
    entries, missing = [], []
    for p in sorted(plans, key=lambda p: p["view_id"]):
        ctx, c = p["ctx"], cls[p["view_id"]]
        if ctx.get("row_selection_leak") != c["labels"]:
            raise SystemExit(f"{p['view_id']}: context.json labels {ctx.get('row_selection_leak')} != receipts {c['labels']}")
        objs = []
        for o in p["objects"]:
            v, rv = verified.get(o["key"]), verified.get(o["receipt_key"])
            if not v or not rv or (v["sha256"], v["bytes"]) != (o["sha256"], o["bytes"]):
                missing.append(o["key"]); continue
            objs.append({"component": o["component"], "key": o["key"], "version_id": v["version_id"], "sha256": v["sha256"],
                         "bytes": v["bytes"], "receipt_key": o["receipt_key"], "receipt_version_id": rv["version_id"],
                         "receipt_sha256": rv["sha256"], "research_only": o["research_only"]})
        exp = (ctx.get("dope") or {}).get("exposure") or {}
        entries.append({"view_id": p["view_id"], "dataset_hash": ctx["dataset_hash"], "fold": ctx["fold"], "cohort": c["cohort"],
                        "context_sha256": ctx["context_sha256"], "outer_train_indices_sha256": ctx["outer_train_indices_sha256"],
                        "outer_test_indices_sha256": ctx["outer_test_indices_sha256"], "inner_labels_sha256": ctx["inner_labels_sha256"],
                        "scaler_sha256": ctx["scaler_sha256"], "values_mode": ctx["values_mode"],
                        "local_context_json_sha256": p["context_json_sha256"], "row_selection_leak": c["labels"],
                        "context_done_in_ledger": p["view_id"] in done,
                        "dope": {"lift_eligible": exp.get("lift_eligible"), "family_id": exp.get("family_id"),
                                 "holdout_assignment": exp.get("holdout_assignment"),
                                 "overlap_status": (exp.get("pretraining_overlap") or {}).get("status")},
                        "objects": objs}
                       | ({"label_exposure": ctx["label_exposure"]} if "label_exposure" in ctx else {})
                       | ({"gpu_hosts": sorted({(c_ or {}).get("host") for c_ in (ctx["gpu"].get("components") or {}).values()})} if "gpu" in ctx else {}))
    disp = {}
    for c in cls.values():
        disp[c["disposition"]] = disp.get(c["disposition"], 0) + 1
    by_mode, lift = {}, {}
    for e in entries:
        by_mode[e["values_mode"]] = by_mode.get(e["values_mode"], 0) + 1
        k = str(e["dope"]["overlap_status"]); lift[k] = lift.get(k, 0) + 1
    doc = {"schema": SCHEMA, "name": name, "prefix": "s3://veox-jopedime-learning-use2/experiments/gp-policy-dl/v2-trainonly/",
           "inputs": pins, "superseded_inputs": sup,
           "rule": "published iff row_selection_leak == ['clean'] after adding acquisition_cap from T1 acquisition markers "
                   f"{sorted(CAP_MARKERS)}, and the dataset is not rights-held; recomputed from the pinned receipts and equal to the tree",
           "cohort_contexts": len(cls), "dispositions": disp, "published_by_values_mode": by_mode,
           "published_dope_overlap_status": lift, "published_contexts": len(entries),
           "published_objects": sum(len(e["objects"]) for e in entries), "missing_in_ledger": len(missing),
           "not_published": [{"view_id": v} | cls[v] for v in sorted(cls) if cls[v]["disposition"] != "published"],
           "contexts": entries, "code_sha256": sha(Path(__file__).read_bytes()),
           "publisher_code_sha256": sha((Path(__file__).resolve().parent / "v2_publish.py").read_bytes())}
    if sum(disp.values()) != len(cls):
        raise SystemExit("dispositions do not sum to the cohort")
    return doc, missing, vp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ledger", default=str(TV2 / "publish" / "v2-publish-ledger.jsonl"))
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--rights-research", action="store_true", help="manifest of RIGHTS-RESEARCH (research-only) contexts")
    a = ap.parse_args(argv)
    doc, missing, vp = build(Path(a.build), a.name, Path(a.ledger), a.rights_research)
    body = (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = sha(body)
    out = TV2 / "publish" / f"v2-publish-manifest-{a.name}-{digest[:12]}.json"
    if not out.exists():
        out.write_bytes(body)
    summary = {k: doc[k] for k in ("cohort_contexts", "dispositions", "published_by_values_mode", "published_dope_overlap_status",
                                   "published_contexts", "published_objects", "missing_in_ledger")} | {"local": str(out), "sha256": digest}
    if a.publish:
        if missing:
            raise SystemExit(f"refusing to publish the manifest: {len(missing)} objects lack a verified ledger record")
        # research-only indexes live under research-only/ too (the first two RIGHTS-RESEARCH manifests, 22:52Z, predate this)
        key = f"{vp.s3io.V2_PREFIX_KEY}{'research-only/' if a.rights_research else ''}manifests/{a.name}/{digest[:12]}.json"
        writer, reader = vp.real_clients()
        ledger = vp.PublishLedger(Path(a.ledger))
        try:
            r = vp.Publisher(writer, reader, ledger).publish_bytes(key, body, "application/json", "manifest", f"manifest:{a.name}")
        finally:
            ledger.close()
        summary["published"] = r
    print(json.dumps(summary, sort_keys=True))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())

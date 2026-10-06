"""Binding manifest for raw-universe row-selection rebuilds (ROWSEL-SPLIT): dataset-level replacement.

A raw rebuild draws a new value-free outer split, so a dataset's epoch-0 contexts are replaced as a group
(context.json `epoch0_replaced`, level "dataset"; new fold f is not epoch-0 fold f). A dataset is published only if ALL its
new contexts are in the tree (the builder's `new_contexts` count) and each one's inputs match the build tree it came from:
  CPU trees  encoder/teacher/A0 sha256 equal the build-tree context.json (which may differ only by the Dope additions)
  GPU trees  the GPU input encoder JTT1 sha256 equals the build-tree encoder.
Datasets the builder could not rebuild are listed from the coverage receipt with their reasons. --publish PUTs the manifest
create-only (research-only contexts route the manifest under research-only/manifests/).

Usage (lane slice): v2_rowsel_raw_manifest.py --build TREE --name NAME --source BUILD_TREE --summary PATH#SHA
                    [--coverage PATH#SHA] [--rights-research] [--publish]
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
TV2 = RUN / "receipts" / "trainonly-v2"
SCHEMA = "jope.dl-v2-rowsel-raw-publish-manifest/v1"
INPUT_FILES = ("encoder.jtt1.gz", "teacher.jtt1.gz", "a0.f32")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pinned(spec):
    p, want = spec.split("#", 1)
    b = Path(p).read_bytes()
    if sha(b) != want:
        raise SystemExit(f"{p}: sha256 mismatch")
    return b, {"path": p, "sha256": want, "bytes": len(b)}


def source_ctx(src_tree: Path):
    out = {}
    for p in glob.glob(f"{src_tree}/shard-*/*/context.json"):
        d = Path(p).parent
        if (d / ".done").exists():
            c = json.loads(Path(p).read_text())
            out[c["view_id"]] = c
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--source", required=True, action="append", help="raw build tree(s) the contexts came from")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--coverage", default=None)
    ap.add_argument("--rights-research", action="store_true")
    ap.add_argument("--ledger", default=str(TV2 / "publish" / "v2-publish-ledger.jsonl"))
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location("v2_publish", Path(__file__).resolve().parent / "v2_publish.py")
    vp = importlib.util.module_from_spec(spec); spec.loader.exec_module(vp)
    _, sref = pinned(a.summary)
    inputs = {"summary": sref}
    cov = None
    if a.coverage:
        cb, inputs["coverage"] = pinned(a.coverage)
        cov = json.loads(cb)
    src = {}
    for s in a.source:
        src.update(source_ctx(Path(s)))
    plans, refused = vp.scan_build(Path(a.build))
    if refused:
        raise SystemExit(f"{len(refused)} contexts refused by the publisher scan; first {refused[0]}")
    verified = {}
    for line in Path(a.ledger).read_bytes().splitlines():
        r = json.loads(line)
        if r.get("event") == "verified":
            verified[r["key"]] = r
    by_ds = defaultdict(list)
    for p in plans:
        by_ds[p["ctx"]["dataset_hash"]].append(p)
    entries, missing, datasets = [], [], []
    for h, ps in sorted(by_ds.items()):
        rep = ps[0]["ctx"]["epoch0_replaced"]
        if any(p["ctx"]["epoch0_replaced"] != rep for p in ps):
            raise SystemExit(f"{h}: contexts disagree on epoch0_replaced")
        if len(ps) != rep["new_contexts"]:
            raise SystemExit(f"{h}: {len(ps)} contexts in tree, builder made {rep['new_contexts']}; datasets replace as a group")
        for p in ps:
            ctx, sc = p["ctx"], src.get(p["view_id"])
            if sc is None:
                raise SystemExit(f"{p['view_id']}: not found in the source build tree")
            if ctx["context_sha256"] != sc["context_sha256"] or ctx.get("rights") != sc.get("rights"):
                raise SystemExit(f"{p['view_id']}: identity or rights differ from the source build")
            if "gpu" in ctx:
                if (ctx["gpu"].get("input") or {}).get("encoder_jtt1_sha256") != sc["files"]["encoder.jtt1.gz"]["sha256"]:
                    raise SystemExit(f"{p['view_id']}: GPU input differs from the source encoder")
            else:
                for f in INPUT_FILES:
                    got = ctx["files"].get(f) or (ctx.get("row_level_inputs_local_custody") or {}).get(f) or {}
                    if got.get("sha256") != sc["files"][f]["sha256"]:
                        raise SystemExit(f"{p['view_id']}: {f} differs from the source build")
            if a.rights_research != (ctx.get("rights") == "research-only"):
                raise SystemExit(f"{p['view_id']}: rights {ctx.get('rights')!r} does not match this manifest kind")
            objs = []
            for o in p["objects"]:
                v, rv = verified.get(o["key"]), verified.get(o["receipt_key"])
                if not v or not rv or (v["sha256"], v["bytes"]) != (o["sha256"], o["bytes"]):
                    missing.append(o["key"]); continue
                objs.append({"component": o["component"], "key": o["key"], "version_id": v["version_id"], "sha256": v["sha256"], "bytes": v["bytes"],
                             "receipt_key": o["receipt_key"], "receipt_version_id": rv["version_id"], "receipt_sha256": rv["sha256"],
                             "research_only": o["research_only"]})
            entries.append({"view_id": p["view_id"], "dataset_hash": h, "fold": ctx["fold"], "context_sha256": ctx["context_sha256"],
                            "values_mode": ctx["values_mode"], "row_selection_leak": ctx["row_selection_leak"], "class": ctx.get("class"),
                            "outer_train_indices_sha256": ctx["outer_train_indices_sha256"], "outer_test_indices_sha256": ctx["outer_test_indices_sha256"],
                            "inner_labels_sha256": ctx["inner_labels_sha256"], "scaler_sha256": ctx["scaler_sha256"],
                            "rowsel_identity": (ctx.get("rowsel") or {}).get("identity"), "rights": ctx.get("rights"),
                            "label_exposure": ctx.get("label_exposure"), "holdout_assignment": ctx.get("holdout_assignment"),
                            "local_context_json_sha256": p["context_json_sha256"], "objects": objs})
        datasets.append({"dataset_hash": h, "class": ps[0]["ctx"].get("class"), "replaces_epoch0": rep})
    not_rebuilt = None
    if cov is not None:
        not_rebuilt = cov.get("not_rebuilt") or cov.get("unrebuildable") or cov.get("datasets_not_rebuilt")
    doc = {"schema": SCHEMA, "name": a.name, "prefix": "s3://veox-jopedime-learning-use2/experiments/gp-policy-dl/v2-trainonly/",
           "inputs": inputs, "rule": "dataset-level replacement: a dataset's epoch-0 contexts are replaced by its full new context set "
           "(new value-free outer split; fold f does not correspond); published only if every new context is present and input-verified",
           "datasets": datasets, "published_datasets": len(datasets), "published_contexts": len(entries),
           "epoch0_contexts_replaced": sum(d["replaces_epoch0"]["count"] for d in datasets),
           "published_objects": sum(len(e["objects"]) for e in entries), "missing_in_ledger": len(missing),
           "not_rebuilt_from_coverage_receipt": not_rebuilt, "contexts": entries, "code_sha256": sha(Path(__file__).read_bytes()),
           "publisher_code_sha256": sha((Path(__file__).resolve().parent / "v2_publish.py").read_bytes())}
    body = (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = sha(body)
    out = TV2 / "publish" / f"v2-publish-manifest-{a.name}-{digest[:12]}.json"
    if not out.exists():
        out.write_bytes(body)
    summary = {k: doc[k] for k in ("published_datasets", "published_contexts", "epoch0_contexts_replaced", "published_objects", "missing_in_ledger")}
    summary |= {"local": str(out), "sha256": digest}
    if a.publish:
        if missing:
            raise SystemExit(f"refusing to publish: {len(missing)} objects lack a verified ledger record")
        key = f"{vp.s3io.V2_PREFIX_KEY}{'research-only/' if a.rights_research else ''}manifests/{a.name}/{digest[:12]}.json"
        writer, reader = vp.real_clients(); ledger = vp.PublishLedger(Path(a.ledger))
        try:
            summary["published"] = vp.Publisher(writer, reader, ledger).publish_bytes(key, body, "application/json", "manifest", f"manifest:{a.name}")
        finally:
            ledger.close()
    print(json.dumps(summary, sort_keys=True))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())

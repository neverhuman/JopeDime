"""S1 teacher-rescoring input list (owner ruling S1-UNDER-G22, ENG:watcher delegated 2026-10-06T00:02Z).

One row per v2 train-only context with its teacher input: the train-only teacher JTT1 (outer-train rows, values re-fitted on
outer-train rows, inner-fold labels from outer-train rows only; outer-test rows present for post-selection scoring only).
  production  published contexts: teacher JTT1 by S3 key, VersionId, sha256 and bytes, taken from the binding manifests
  research    RIGHTS-RESEARCH contexts: local custody path and sha256 (row data is never uploaded), rights=research-only
Every row carries dataset_hash, fold, context_sha256, n_train/n_test, values_mode, inner_labels_sha256, the epoch-0 context it
replaces (row-selection rebuilds) and the P3 family/assignment, so S1 can bind its fixed packet and its sealed handling.
--publish PUTs both lists create-only (production under manifests/, research under research-only/manifests/).

Usage (lane slice): v2_s1_input_list.py --name s1-input-list-v1 [--publish]
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
PUB = RUN / "receipts" / "trainonly-v2" / "publish"
SCHEMA = "jope.dl-v2-s1-input-list/v1"
# production binding manifests that carry teacher inputs (name, local copy)
PRODUCTION_NAMES = ["clean-v1", "rowsel-t1-clean-v1", "rowsel-raw-clean-v1", "rowsel-t1-fallback-v1", "rowsel-raw-nonnumeric-v1"]
RESEARCH_TREES = [RUN / "build-v2/RIGHTS-RESEARCH-clean-v1/custody", RUN / "build-v2/ROWSEL-T1-frozen-kmeans-research-v1",
                  RUN / "build-v2/RIGHTS-RESEARCH-raw-v1/custody"]


def production_manifests(names=None):
    """name -> local copy, for each production CPU manifest verified in the publish ledger (latest if several)."""
    names = names or PRODUCTION_NAMES
    out = {}
    for line in (PUB / "v2-publish-ledger.jsonl").read_bytes().splitlines():
        r = json.loads(line)
        if r.get("event") == "verified" and "/manifests/" in r["key"] and "/research-only/" not in r["key"]:
            name, f = r["key"].split("/manifests/", 1)[1].split("/")
            local = PUB / f"v2-publish-manifest-{name}-{f[:-5]}.json"
            if name in names and local.exists() and sha(local.read_bytes()) == r["sha256"]:
                out[name] = local
    return [(n, out[n]) for n in names if n in out]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


CUT_PREFIXES = ("trainonly-split-v2:", "trainonly-split-v2g:")


def cut_of(ctx):
    """The context's outer-split identity for HWM comparisons: the cut block of a re-split context, or, for raw v2 contexts
    (no cut block), "trainonly-split-v2:" + rowsel.split.sha256 (the formula of rowsel-raw-v2-cut-index-*.json). None otherwise."""
    c = ctx.get("cut") or {}
    if str(c.get("cut_id", "")).startswith(CUT_PREFIXES):
        return c["cut_id"]
    sp = (ctx.get("rowsel") or {}).get("split") or {}
    if sp.get("schema") == "rowsel-raw-split-v2" and len(str(sp.get("sha256", ""))) == 64:
        return "trainonly-split-v2:" + sp["sha256"]
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--denylist", required=True, help="PATH#SHA256 of the holdout denylist (HOLDOUT-INPUTS-140); refused contexts are excluded")
    ap.add_argument("--production-names", default=",".join(PRODUCTION_NAMES), help="comma-separated production manifest names")
    ap.add_argument("--research-trees", default=",".join(map(str, RESEARCH_TREES)), help="comma-separated research custody trees")
    ap.add_argument("--supersedes", default="", help="comma-separated published keys of earlier S1 lists this one replaces")
    ap.add_argument("--require-cut", action="store_true", help="every row must carry a split-v2/v2g cut_id (HWM identity)")
    a = ap.parse_args(argv)
    names = [x for x in a.production_names.split(",") if x]
    trees = [Path(x) for x in a.research_trees.split(",") if x]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dltrain.admission import Admission, AdmissionRefused
    dpath, dsha = a.denylist.split("#", 1)
    adm = Admission(dpath, dsha, purpose="teacher")
    hb = (RUN / "holdout/holdout.json").read_bytes(); rb = (RUN / "families/g2/registry.json").read_bytes()
    assign = json.loads(hb)["assignment"]
    ds2fam = {m["dataset_hash"]: fid for fid, f in json.loads(rb)["families"].items() for m in (f.get("members") or []) if m.get("dataset_hash")}

    def p3(h):
        fam = ds2fam.get(h)
        return {"family_id": fam, "assignment": assign.get(fam, "no_family")}

    prod, sources = [], {}
    pms = production_manifests(names)
    prod_names = {n for n, _ in pms}
    for name, path in pms:
        mb = path.read_bytes(); m = json.loads(mb)
        sources[name] = {"path": str(path), "sha256": sha(mb)}
        ds_rep = {d["dataset_hash"]: d["replaces_epoch0"] for d in m.get("datasets", [])}   # raw route: dataset-level replacement
        for c in m["contexts"]:
            t = [o for o in c["objects"] if o["component"] == "jtt1-teacher"]
            if len(t) != 1:
                raise SystemExit(f"{name}/{c['view_id']}: expected one teacher object")
            t = t[0]
            prod.append({"view_id": c["view_id"], "dataset_hash": c["dataset_hash"], "fold": c["fold"], "context_sha256": c["context_sha256"],
                         "values_mode": c["values_mode"], "inner_labels_sha256": c["inner_labels_sha256"],
                         "outer_train_indices_sha256": c["outer_train_indices_sha256"], "outer_test_indices_sha256": c["outer_test_indices_sha256"],
                         "replaces_epoch0": c.get("replaces_epoch0") or ds_rep.get(c["dataset_hash"]), "row_selection_leak": c["row_selection_leak"], "set": name,
                         "cut_id": c.get("cut_id"),
                         "teacher_input": {"key": t["key"], "version_id": t["version_id"], "sha256": t["sha256"], "bytes": t["bytes"]}, "p3": p3(c["dataset_hash"])})
    research = []
    for tree in trees:
        for p in sorted(glob.glob(f"{tree}/shard-*/*/context.json")):
            d = Path(p).parent
            if not (d / ".done").exists():
                continue
            ctx = json.loads(Path(p).read_text())
            tf = d / "teacher.jtt1.gz"; tb = tf.read_bytes()
            if sha(tb) != ctx["files"]["teacher.jtt1.gz"]["sha256"]:
                raise SystemExit(f"{tf}: sha256 differs from context.json")
            research.append({"view_id": ctx["view_id"], "dataset_hash": ctx["dataset_hash"], "fold": ctx["fold"], "context_sha256": ctx["context_sha256"],
                             "values_mode": ctx["values_mode"], "inner_labels_sha256": ctx["inner_labels_sha256"],
                             "outer_train_indices_sha256": ctx["outer_train_indices_sha256"], "outer_test_indices_sha256": ctx["outer_test_indices_sha256"],
                             "replaces_epoch0": {k: (ctx.get("epoch0") or {}).get(k) for k in ("view_id", "context_sha256")} if ctx.get("epoch0") else None,
                             "row_selection_leak": ctx.get("row_selection_leak"), "set": tree.parent.name if tree.name == "custody" else tree.name,
                             "cut_id": cut_of(ctx),
                             "rights": ctx.get("rights"), "rights_decision": ctx.get("rights_decision"),
                             "teacher_input": {"local_custody_path": str(tf), "sha256": sha(tb), "bytes": len(tb), "uploaded": False},
                             "p3": p3(ctx["dataset_hash"])})
    if a.require_cut:
        bad = [r["view_id"] for r in prod + research if not str(r.get("cut_id") or "").startswith(CUT_PREFIXES)]
        if bad:
            raise SystemExit(f"{len(bad)} rows without a split-v2/v2g cut_id, e.g. {bad[:3]}")
    excluded = []
    for rows in (prod, research):
        keep = []
        for r in rows:
            try:
                adm.check(r["view_id"], assignment=r["p3"]["assignment"])
                keep.append(r)
            except AdmissionRefused as e:
                excluded.append({"view_id": r["view_id"], "set": r["set"], "assignment": r["p3"]["assignment"], "reason": str(e)[:160]})
        rows[:] = keep
    out = {}
    for kind, rows in (("production", prod), ("research", research)):
        counts = {}
        for r in rows:
            counts[r["p3"]["assignment"]] = counts.get(r["p3"]["assignment"], 0) + 1
        doc = {"schema": SCHEMA, "name": f"{a.name}-{kind}", "kind": kind, "contexts": len(rows), "assignment_counts": counts,
               "sources": sources if kind == "production" else {"trees": [str(t) for t in trees]},
               "supersedes": [k for k in a.supersedes.split(",") if k and ("/research-only/" in k) == (kind == "research")],
               "p3": {"holdout_sha256": sha(hb), "registry_sha256": sha(rb)},
               "owner_decisions": ["S1-UNDER-G22 owner-decisions-20261006T000114Z.json", "S1-BUDGET-18 owner-decisions-20261005T183423Z.json",
                                   "SEALED-LIFT owner-decisions-20261005T233942Z.json"] + (["RIGHTS-RESEARCH owner-decisions-20261005T223750Z.json"] if kind == "research" else []),
               "denylist": {"path": dpath, "sha256": dsha},
               "excluded_by_admission": [e for e in excluded if (e["set"] in prod_names) == (kind == "production")],
               "note": "HOLDOUT-INPUTS-140: sealed_excluded and heldout contexts are refused by dltrain.admission and listed under excluded_by_admission",
               "rows": rows, "code_sha256": sha(Path(__file__).read_bytes())}
        body = (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode()
        digest = sha(body)
        local = PUB / f"v2-s1-input-list-{a.name}-{kind}-{digest[:12]}.json"
        if not local.exists():
            local.write_bytes(body)
        out[kind] = {"local": str(local), "sha256": digest, "contexts": len(rows), "assignment_counts": counts, "body": body}
    if a.publish:
        spec = importlib.util.spec_from_file_location("v2_publish", Path(__file__).resolve().parent / "v2_publish.py")
        vp = importlib.util.module_from_spec(spec); spec.loader.exec_module(vp)
        writer, reader = vp.real_clients(); ledger = vp.PublishLedger(vp.DEFAULT_PUBLISH_LEDGER)
        try:
            pub = vp.Publisher(writer, reader, ledger)
            for kind, o in out.items():
                key = f"{vp.s3io.V2_PREFIX_KEY}{'research-only/' if kind == 'research' else ''}manifests/{a.name}-{kind}/{o['sha256'][:12]}.json"
                o["published"] = pub.publish_bytes(key, o["body"], "application/json", "manifest", f"s1-input-list:{kind}")
        finally:
            ledger.close()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "body"} for k, v in out.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

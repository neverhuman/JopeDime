"""H288 fresh cohort v3 (hw_t0005): H286's cohort_v2.py with the half switched to ODD, H285's first 60 ODD families
excluded instead of H283's 60, and 300 families taken. Metadata only, outcome-blind, built BEFORE any forward pass.
ODD families beyond the first 300 qualifying ones are reserved (untouched) for a later wave.

H286 text (unchanged rule): From fresh_cohort_v1.json (sha256 12d2b1f3...; H283's shared cohort): the EVEN half only, minus H283's assigned 60
EVEN families, in the same sha256(family key) order. For each family, the candidate contexts are the admitted S1-prod
train-side contexts (P3 train_fit / train_nonpool) of the family's catalog datasets that have a summary.tar.gz and a
teacher.jtt1.gz (H283's buildability rule, re-derived with H283's fresh_cohort.py helpers). A context qualifies if its
summary holds >= 2 accept records (a line count of accepts.jsonl; no record is parsed, no fitness is read). Per family:
prefer train_fit, then the smallest catalog id, then the smallest context sha256. The first 200 qualifying families form
the cohort (a smaller outcome-blind prefix, floor 100, only if build+score would not fit the window). ODD families
(H285) and H283's 60 are disjoint by construction."""

import hashlib
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = Path("/home/ubuntu/jope_plan/ns/receipts/fresh_cohort_v1.json")
V1_SHA = "12d2b1f37bf0e3c48ee61a24949b4f504251fe56b9b26b3081398623982f323c"
TAKE = 300
HALF = "ODD"
PRIOR_WAVE = "H285"  # its first 60 ODD families are excluded
MIN_ACCEPTS = 2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_v1_helpers():
    path = HERE.parent / "h283_rerank8_veto_fresh_cohort_gpu" / "fresh_cohort.py"
    spec = importlib.util.spec_from_file_location("fresh_cohort_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, sha(path)


def count_accepts(tar_path):
    """Number of non-empty lines across accepts.jsonl members (metadata: pool size; records are not parsed)."""
    n = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            if m.name.endswith("/accepts.jsonl") or m.name == "accepts.jsonl":
                f = tf.extractfile(m)
                if f is not None:
                    n += sum(1 for line in f if line.strip())
    return n


def main(out):
    if sha(V1) != V1_SHA:
        raise RuntimeError("fresh_cohort_v1 hash")
    v1 = json.loads(V1.read_text())
    fc, fc_sha = load_v1_helpers()
    cat = json.loads(fc.CATALOG.read_text())["datasets"]
    members = {}
    for d in cat:
        if d["task"] == "regression" and fc.fam_key(d):
            members.setdefault(fc.fam_key(d), []).append(d["id"])
    s1, _ = fc.s1_entries()
    contexts = {}
    for r in s1:
        p3 = r["_p3"]
        cid = r["context_sha256"]
        if (
            r["_status"] == "admitted"
            and p3.get("assignment") in fc.TRAIN_SIDE
            and (fc.S1 / "contexts" / cid / "summary.tar.gz").exists()
            and (Path(r["context_dir"]) / "teacher.jtt1.gz").exists()
        ):
            contexts.setdefault(r["dataset_hash"], {})[cid] = dict(
                context_sha256=cid,
                context_dir=r["context_dir"],
                cut_id=r.get("cut_id"),
                fold=r.get("fold"),
                p3_family_id=p3.get("family_id"),
                p3_assignment=p3.get("assignment"),
            )
    prior = set(v1["assignment"][PRIOR_WAVE]["families"])
    reasons = dict(
        other_half=0, h285_assigned=0, no_buildable_context=0, fewer_than_2_accepts=0, reserved_beyond_take=0
    )
    rows, scanned = [], 0
    for e in v1["eligible"]:
        if e["half"] != HALF:
            reasons["other_half"] += 1
            continue
        if e["family_key"] in prior:
            reasons["h285_assigned"] += 1
            continue
        if len(rows) >= TAKE:
            reasons["reserved_beyond_take"] += 1
            continue
        scanned += 1
        cands = [
            dict(c, dataset_id=ds)
            for ds in sorted(members.get(e["family_key"], []))
            for c in contexts.get(ds, {}).values()
        ]
        if not cands:
            reasons["no_buildable_context"] += 1
            continue
        cands.sort(key=lambda c: (c["p3_assignment"] != "train_fit", c["dataset_id"], c["context_sha256"]))
        pick = None
        for c in cands:
            n = count_accepts(fc.S1 / "contexts" / c["context_sha256"] / "summary.tar.gz")
            if n >= MIN_ACCEPTS:
                pick = dict(c, accepts=n)
                break
        if pick is None:
            reasons["fewer_than_2_accepts"] += 1
            continue
        rows.append(dict(e, dataset_id=pick["dataset_id"], context=pick, buildable=True))
    doc = dict(
        schema="jope.fresh-cohort/v3",
        created_by="rd_opus (H288, hw_t0005)",
        decided_before_any_outcome=True,
        rules=__doc__,
        sources=dict(fresh_cohort_v1=dict(path=str(V1), sha256=V1_SHA), v1_helpers_sha256=fc_sha),
        exclusion_receipts=reasons,
        counts=dict(scanned_families_in_half=scanned, qualifying=len(rows), taken=min(len(rows), TAKE)),
        assignment=dict(H288=dict(owner="rd_opus", path="GPU", half="ODD", families=[r["family_key"] for r in rows])),
        eligible=rows,
    )
    Path(out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(dict(counts=doc["counts"], exclusions=reasons)))


if __name__ == "__main__":
    main(sys.argv[1])

"""Shared FRESH regression cohort v1 for H283/H284/H285 (hw_t2305), outcome-blind, metadata only.

Source: S3_DATASET_CATALOG.json (catalog v246), task=regression, origin=real, datasets bucket only. Family key =
provenance.dedupe_fingerprint (`family_key`), else the provisional `srcfile:<stem>` (`family_key_derived`).

Exclusions (a family is dropped if ANY of its datasets is excluded):
  sealed:  the 14 sealed-matching catalog ids; any record mentioning BeyondArena or "sealed"; any record outside the
           datasets bucket; no family key -> excluded (none occur). The sealed bucket is never read.
  P3/S1:   every dataset that any S1-prod input list (the GP runs that produced every corpus pool) assigns to P3
           heldout / sealed_excluded / dev, and every dataset sharing a coarse P3 family (fam-...) with one of them.
  seen:    every dataset in the 1,780-context admitted corpus (DL-Train, DL-Val, DL-Holdout, inner/outer contexts), the
           approved 1,219/91 packet and the writer input (selected.json byte-identical, sha 981960ad...); every
           dataset sharing a catalog family key or a coarse corpus family (fam-...) with them; conservatively also any
           dataset sharing a source URL, source file, normalised name or train-data sha256 with a seen one.

Buildable = an admitted S1-prod context with P3 assignment train_fit/train_nonpool (train side), a summary.tar.gz (the
candidate pool, extracted exactly as research/flywheel_selfplay/dataset_builder.py does) and a teacher.jtt1.gz (the
train-only table = teacher with the test rows stripped, byte-identical to the corpus tables). No final.json, fitness or
any other outcome is read here.

Order: eligible family keys sorted by sha256(key). EVEN first hex nibble -> H283 (rd_opus, GPU path); ODD -> H285
(rd_astra, CPU path). Each wave takes the first <= 60 BUILDABLE families of its half. Per family: the smallest catalog
id with a buildable context, then its smallest context_sha256 (one context per family)."""

import hashlib
import json
import re
import sys
from pathlib import Path

CATALOG = Path("/home/ubuntu/jope_plan/master/source/S3_DATASET_CATALOG.json")
ADMITTED = Path("/home/ubuntu/jope-evidence/dope-lift-20261021/deep-learning-dl03/admitted-input-contexts.json")
PACKET = Path("/home/ubuntu/jope-data/h50/input/selected.json")
S1 = Path("/home/ubuntu/scratch/jopedime-dl-training-v1/receipts/trainonly-v2/s1-prod")
SEALED_IDS = {
    "0a8428ef25a69175", "24713c7f7bfa7807", "36ef1e99b0f383cb", "48bcaee521b95d1b", "525fc7ef6e7d6d64",
    "5d21a759a92617cb", "6a3edcd68b228f1f", "a882d8b478db2ecb", "c4b1f6dfb2a8605f", "ca307077904135b9",
    "d2cf9a9eddfd7d35", "ee74ac8192a9f455", "f2fa5e94ae02ab7c", "8366db9ae9698880",
}  # fmt: skip
TRAIN_SIDE = {"train_fit", "train_nonpool"}
BAD_ROLES = {"heldout", "sealed_excluded", "dev"}
PER_WAVE = 60


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm_name(name):
    if not name:
        return None
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\(.*?\)", "", name.lower())) or None


def fam_key(d):
    return d.get("family_key") or d.get("family_key_derived")


def s1_entries():
    rows, files = [], {}
    for f in sorted([S1 / "input-list.json", *S1.glob("input-list-join-*.json")]):
        doc = json.loads(f.read_text())
        files[f.name] = sha(f)
        for status in ("admitted", "refused"):
            for r in doc.get(status) or []:
                rows.append(dict(r, _status=status, _p3=(r.get("p3") or {})))
    return rows, files


def main(out):
    cat = json.loads(CATALOG.read_text())["datasets"]
    by_id = {d["id"]: d for d in cat}
    admitted = json.loads(ADMITTED.read_text())
    packet = json.loads(PACKET.read_text())
    s1, s1_files = s1_entries()

    seen_ids = {r["dataset_hash"] for r in admitted}
    seen_ids |= {r["dataset_id"] for r in packet["train"] + packet["dev"]}
    corpus_fams = {r["family_id"] for r in admitted}
    corpus_fams |= {r["family_id"] for r in packet["train"] + packet["dev"]}
    seen = [by_id[i] for i in seen_ids if i in by_id]
    seen_keys = {fam_key(d) for d in seen} - {None}
    seen_urls = {d.get("source_url") for d in seen} - {None}
    seen_files = {d.get("source_file") for d in seen} - {None}
    seen_names = {norm_name(d.get("name")) for d in seen} - {None}
    seen_data = {d.get("train_sha256") for d in seen} - {None}

    coarse, bad_ds, bad_fams, buildable = {}, set(), set(), {}
    for r in s1:
        ds, p3 = r["dataset_hash"], r["_p3"]
        coarse.setdefault(ds, set()).add(p3.get("family_id"))
        if p3.get("assignment") in BAD_ROLES:
            bad_ds.add(ds)
            bad_fams.add(p3.get("family_id"))
        cid = r["context_sha256"]
        if (
            r["_status"] == "admitted"
            and p3.get("assignment") in TRAIN_SIDE
            and (S1 / "contexts" / cid / "summary.tar.gz").exists()
            and (Path(r["context_dir"]) / "teacher.jtt1.gz").exists()
        ):
            buildable.setdefault(ds, {})[cid] = dict(
                context_sha256=cid,
                context_dir=r["context_dir"],
                cut_id=r.get("cut_id"),
                fold=r.get("fold"),
                p3_family_id=p3.get("family_id"),
                p3_assignment=p3.get("assignment"),
            )
    bad_fams.discard(None)

    reasons, fam_excluded, candidates = {}, {}, []

    def mark(d, why):
        reasons[why] = reasons.get(why, 0) + 1
        if fam_key(d):
            fam_excluded.setdefault(fam_key(d), set()).add(why)

    for d in cat:
        blob = json.dumps([d.get("name"), d.get("source_url"), d.get("source_file"), d.get("collections")]).lower()
        cf = coarse.get(d["id"], set())
        if d["task"] != "regression":
            reasons["task_not_regression"] = reasons.get("task_not_regression", 0) + 1
        elif d["id"] in SEALED_IDS:
            mark(d, "sealed_matching_id")
        elif "beyondarena" in blob or "sealed" in blob:
            mark(d, "sealed_or_beyondarena_text")
        elif d.get("s3_bucket") != "veox-jopedime-datasets-use2":
            mark(d, "not_train_side_bucket")
        elif not fam_key(d):
            reasons["family_key_unknown"] = reasons.get("family_key_unknown", 0) + 1
        elif d["id"] in bad_ds:
            mark(d, "p3_heldout_sealed_excluded_or_dev")
        elif cf & bad_fams:
            mark(d, "p3_family_heldout_sealed_excluded_or_dev")
        elif d["id"] in seen_ids:
            mark(d, "dataset_seen_corpus_or_writer_input")
        elif fam_key(d) in seen_keys:
            mark(d, "family_key_seen_corpus_or_writer_input")
        elif cf & corpus_fams:
            mark(d, "p3_family_in_flywheel_corpus")
        elif d.get("source_url") in seen_urls or d.get("source_file") in seen_files:
            mark(d, "source_shared_with_seen")
        elif norm_name(d.get("name")) in seen_names:
            mark(d, "name_shared_with_seen")
        elif d.get("train_sha256") in seen_data:
            mark(d, "data_identical_to_seen")
        else:
            candidates.append(d)

    fams = {}
    for d in candidates:
        if fam_key(d) in fam_excluded:
            reasons["family_has_excluded_member"] = reasons.get("family_has_excluded_member", 0) + 1
            continue
        fams.setdefault(fam_key(d), []).append(d["id"])

    rows = []
    for k in sorted(fams, key=lambda k: hashlib.sha256(k.encode()).hexdigest()):
        h = hashlib.sha256(k.encode()).hexdigest()
        ds = sorted(fams[k])
        built = [i for i in ds if buildable.get(i)]
        pick = built[0] if built else ds[0]
        ctx = sorted(buildable.get(pick, {}).values(), key=lambda c: c["context_sha256"])
        d = by_id[pick]
        rows.append(
            dict(
                family_key=k,
                family_key_sha256=h,
                half="EVEN" if int(h[0], 16) % 2 == 0 else "ODD",
                key_kind="dedupe_fingerprint" if d.get("family_key") else "srcfile_provisional",
                dataset_id=pick,
                n_eligible_datasets=len(ds),
                buildable=bool(ctx),
                n_buildable_contexts=len(ctx),
                context=ctx[0] if ctx else None,
                rows=d["rows"],
                cols=d["cols"],
                manifest_key=d["s3_objects"]["manifest.json"]["key"],
                manifest_etag=d["s3_objects"]["manifest.json"]["etag"],
                native_split_registry_key=d.get("native_split_registry_key"),
                train_sha256=d.get("train_sha256"),
            )
        )
    even = [r for r in rows if r["half"] == "EVEN" and r["buildable"]]
    odd = [r for r in rows if r["half"] == "ODD" and r["buildable"]]
    doc = dict(
        schema="jope.fresh-cohort/v1",
        created_by="rd_opus (H283, hw_t2305)",
        decided_before_any_outcome=True,
        rules=__doc__,
        sources=dict(
            catalog=dict(path=str(CATALOG), sha256=sha(CATALOG), catalog_version="v246"),
            admitted_corpus=dict(path=str(ADMITTED), sha256=sha(ADMITTED), contexts=len(admitted)),
            packet=dict(path=str(PACKET), sha256=sha(PACKET), note="byte-identical to writer input selected.json"),
            s1_prod_input_lists=s1_files,
        ),
        seen=dict(dataset_ids=len(seen_ids), catalog_family_keys=len(seen_keys), corpus_fam_ids=len(corpus_fams)),
        p3=dict(bad_role_datasets=len(bad_ds), bad_role_fam_ids=len(bad_fams)),
        exclusion_receipts=dict(sorted(reasons.items())),
        sealed_roster_note=(
            "The catalog's P3 keyed_subset_fold_manifest has sealed_excluded_applied=false (no roster); the S1-prod input "
            "lists do carry P3 sealed_excluded/heldout/dev assignments, and those datasets and coarse families are "
            "excluded, as are the catalog's 14 sealed-matching ids and any BeyondArena/sealed text match."
        ),
        counts=dict(
            eligible_families=len(rows),
            buildable_families=sum(r["buildable"] for r in rows),
            even_buildable=len(even),
            odd_buildable=len(odd),
        ),
        assignment=dict(
            H283=dict(owner="rd_opus", path="GPU", half="EVEN", families=[r["family_key"] for r in even[:PER_WAVE]]),
            H285=dict(owner="rd_astra", path="CPU", half="ODD", families=[r["family_key"] for r in odd[:PER_WAVE]]),
        ),
        eligible=rows,
    )
    Path(out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(dict(counts=doc["counts"], exclusions=doc["exclusion_receipts"], seen=doc["seen"], p3=doc["p3"])))


if __name__ == "__main__":
    main(sys.argv[1])

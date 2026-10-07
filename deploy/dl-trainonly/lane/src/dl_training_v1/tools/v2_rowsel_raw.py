"""Raw-universe row-selection rebuild, local and unpublished (ROWSEL-TRAINONLY + ROWSEL-SPLIT; dltrain/rowsel_raw.py).

Scope (recounted from the receipts, never from the plan's prose):
  (a) all_rows_coreset   acquisition-selection class all_rows_coreset (frozen), every epoch-0 context labelled
                         ["all_rows_coreset"] in row-selection-leak-v2, L7a dataset record status "matched"
  (b) acquisition_capped acquisition-selection class kmeans, cohort incoming, reduced=true (original_size.downsampled),
                         every epoch-0 context labelled ["kmeans"], L7a dataset record status "matched"
Rights-held datasets (rights-gate receipt) are refused before any read; their raw files are never opened.

Per dataset: the L7a match is re-run on the same raw bytes and must reproduce the L7a record (values_sha256, read
report); then universe -> pre-cap -> 5-fold split -> per fold selection, L7a values, both counterfactual gates. Contexts
are written only when all 5 folds pass (files, context.json, then .done last); otherwise nothing is written for the
dataset, the failure goes to the dataset record and the shard receipt, and the dataset keeps its epoch-0 contexts.
A context with .done is never rewritten; its recorded identity must equal the recomputed one.

Usage (inside the lane slice, RUN/tmp/p2fix3/py.sh). --split-version defaults to rowsel-raw-split-v2 (target-free);
rowsel-raw-split-v1 runs only when passed explicitly (v1 replay/verification; PR #218 finding 10):
  v2_rowsel_raw.py plan      [--out DIR] [--of 3]
  v2_rowsel_raw.py build     [--out DIR] --shard I --of N [--datasets h1,h2]
  v2_rowsel_raw.py summarize [--out DIR] --of N --tests-log LOG --mutants JSON [--pilot JSON] [--receipt PATH]
No S3, network, GPU or git; nothing is published.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

from dltrain import (cpuguard, jtt1, nonnumeric, rowsel, rowsel_raw, rowsel_raw_split_v2, rowsel_split_v2_groups, scale_check,
                     t1raw, target_linear)

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
META_DIR = RUN / "inputs" / "t1-meta"
RAW_DIR = RUN / "inputs" / "t1-raw"
DATASETS_INV = RUN / "tmp" / "trainonly-v1" / "epoch0-datasets.json"
RECEIPTS = RUN / "receipts" / "trainonly-v2"
ACQ = RECEIPTS / "acquisition-selection-20261005T2045Z.json"
LEAK = RECEIPTS / "row-selection-leak-v2-20261005T200546Z.jsonl"
L7A_SUMMARY = RECEIPTS / "l7a-build-summary-20261005T204452Z.json"
L7A_OUT = RUN / "build-v2" / "L7a"
OUT = RUN / "build-v2" / "ROWSEL-RAW-v1"
OWNER = {"ROWSEL-SPLIT": "/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261005T210659Z.json",
         "ROWSEL-TRAINONLY": "/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261005T203324Z.json"}
CODE_FILES = ("tools/v2_rowsel_raw.py", "dltrain/rowsel_raw.py", "dltrain/rowsel.py", "dltrain/t1raw.py", "dltrain/t1prep.py",
              "dltrain/trainonly.py", "dltrain/metafeatures.py", "dltrain/scale_check.py", "dltrain/jtt1.py", "dltrain/cpuguard.py",
              "dltrain/nonnumeric.py", "dltrain/rowsel_raw_split_v2.py", "dltrain/rowsel_split_v2_groups.py",
              "dltrain/target_linear.py")
LATER_CODE_FILES = frozenset({"dltrain/nonnumeric.py",       # added after ROWSEL-RAW-v1 was built; not a dependency of it
                              "dltrain/rowsel_raw_split_v2.py",    # added after the v1 trees; RAW-SPLIT-RULING
                              "dltrain/rowsel_split_v2_groups.py", "dltrain/target_linear.py"})   # RAW-GROUP / LINEAR-DETERMINED
SPLIT_V1 = rowsel_raw.SPLIT_SCHEMA               # "rowsel-raw-split-v1" (18eac154): kept to verify the v1 trees only
SPLIT_V2 = rowsel_raw_split_v2.SPLIT_SCHEMA      # "rowsel-raw-split-v2": ROWSEL-SPLIT as written (RAW-SPLIT-RULING)
SPLIT_V2G = rowsel_split_v2_groups.SPLIT_SCHEMA   # "rowsel-split-v2-groups": RAW-GROUP (a), per-entry group column
SPLIT_VERSIONS = {SPLIT_V1: (rowsel_raw.outer_split, rowsel_raw.split_report),
                  SPLIT_V2: (rowsel_raw_split_v2.outer_split, rowsel_raw_split_v2.split_report),
                  SPLIT_V2G: (None, None)}            # built per entry by v2g_functions (the entry names its group column)
CLASSES = ("all_rows_coreset", "acquisition_capped")
LABEL_OF = {"all_rows_coreset": ["all_rows_coreset"], "acquisition_capped": ["kmeans"]}
PLAN_SCHEMA = "jope.dl-v2-rowsel-raw-plan/v1"
DATASET_SCHEMA = "jope.dl-v2-rowsel-raw-dataset/v1"
SHARD_SCHEMA = "jope.dl-v2-rowsel-raw-shard/v1"
SUMMARY_SCHEMA = "jope.dl-v2-rowsel-raw-summary/v1"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ref(p) -> dict:
    p = Path(p)
    return {"path": str(p), "sha256": sha_file(p), "bytes": p.stat().st_size}


def atomic(path: Path, data: bytes):
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def jdump(o) -> bytes:
    return (json.dumps(o, sort_keys=True) + "\n").encode()


def code_shas():
    root = Path(__file__).resolve().parents[1]
    return {f: sha((root / f).read_bytes()) for f in CODE_FILES}


def split_functions(e, split_version: str | None = None) -> tuple:
    """(version, outer_split, split_report) of a plan entry. Nothing defaults to v1 (PR #218 finding 10; the owner's
    standing ruling: defaults are target-free): an entry names its version, or (a v1 plan, byte-identical to the plans
    built before versions were stamped) carries none and is read as v1 ONLY when the caller passes split_version=v1
    explicitly. A version the caller passes must equal the entry's; an unknown version is refused."""
    v = e.get("split_version")
    if v is None:
        if split_version != SPLIT_V1:
            raise rowsel_raw.RowselRawError("split_version_not_given",
                                            f"entry without split_version; the v1 replay must pass {SPLIT_V1!r} explicitly, got {split_version!r}")
        v = SPLIT_V1
    elif split_version is not None and split_version != v:
        raise rowsel_raw.RowselRawError("split_version_mismatch", f"entry {v!r} != requested {split_version!r}")
    if v not in SPLIT_VERSIONS:
        raise rowsel_raw.RowselRawError("unknown_split_version", repr(v))
    if v == SPLIT_V2G:
        return (v, *v2g_functions(e))
    return (v, *SPLIT_VERSIONS[v])


def stamp_split_version(doc: dict, version: str) -> dict:
    """A v1 plan stays byte-identical to the plans already built (no field); any other version is named in the plan and
    in every dataset entry, which is what run_dataset reads."""
    if version not in SPLIT_VERSIONS:
        raise SystemExit(f"unknown split version {version!r}")
    if version != SPLIT_V1:
        doc["split_version"] = version
        for e in doc["datasets"]:
            e["split_version"] = version
    return doc


def plan_split_version(pl: dict, declared: str | None = None) -> str:
    """The plan's split version. A plan without the field (the v1 plans) is v1 only when `declared` is v1 explicitly;
    otherwise it is refused (no silent v1 default). A declared version must equal the plan's."""
    v = pl.get("split_version")
    if v is None:
        if declared != SPLIT_V1:
            raise SystemExit(f"plan has no split_version; a v1 replay must pass --split-version {SPLIT_V1} explicitly (got {declared!r})")
        return SPLIT_V1
    if declared is not None and declared != v:
        raise SystemExit(f"plan split version {v} != --split-version {declared}")
    return v


def default_out(route: str, version: str) -> Path:
    trees = {("l7a", SPLIT_V1): OUT, ("nonnumeric", SPLIT_V1): OUT_NONNUMERIC,
             ("l7a", SPLIT_V2): RUN / "build-v2" / "ROWSEL-RAW-v2", ("nonnumeric", SPLIT_V2): RUN / "build-v2" / "ROWSEL-RAW-v2-nonnumeric",
             ("l7a", SPLIT_V2G): RUN / "build-v2" / "ROWSEL-RAW-v2gsplit",
             ("nonnumeric", SPLIT_V2G): RUN / "build-v2" / "ROWSEL-RAW-v2gsplit-nonnumeric"}
    return trees[(route, version)]


def should_mark_complete(complete: bool, version: str) -> bool:
    """Only a finished tree of a NEW split version gets a .complete marker: the v1 trees are never written to."""
    return bool(complete) and version in SPLIT_VERSIONS and version != SPLIT_V1


def write_complete(out: Path, receipt: Path, counts: dict, version: str) -> Path:
    """Create-only marker at the tree root once its summary verified every planned dataset (done or failed)."""
    m = out / ".complete"
    body = jdump({"schema": "jope.dl-v2-rowsel-raw-tree-complete/v1", "tree": str(out), "split_version": version,
                  "summary": ref(receipt), "counts": counts, "published": False, "utc_us": int(time.time() * 1e6)})
    with open(m, "xb") as f:
        f.write(body); f.flush(); os.fsync(f.fileno())
    return m


def code_known(recorded, codes) -> bool:
    """Recorded code equals the current code, except that the tool may be an earlier version kept read-only as
    receipts/trainonly-v2/build/v2_rowsel_raw.build-<sha256>.py (the bytes that built it)."""
    if not isinstance(recorded, dict) or not set(recorded) <= set(codes) or set(codes) - set(recorded) - LATER_CODE_FILES:
        return False
    t = "tools/v2_rowsel_raw.py"
    if any(recorded[k] != codes[k] for k in recorded if k != t):
        return False
    saved = RECEIPTS / "build" / f"v2_rowsel_raw.build-{recorded[t]}.py"
    return recorded[t] == codes[t] or (saved.exists() and sha_file(saved) == recorded[t])


def cpu_s() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def maxrss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


# --------------------------------------------------------------------------------------------------- scope

def held_set():
    paths = sorted(glob.glob(str(RECEIPTS / "rights-gate-no-license-*.jsonl")))
    if len(paths) != 1:
        raise SystemExit(f"expected one rights-gate receipt, found {paths}")
    rows = [json.loads(l) for l in Path(paths[0]).read_bytes().decode().splitlines() if l.strip()]
    return {r["dataset_hash"] for r in rows}, ref(paths[0])


def l7a_records():
    """dataset_hash -> (record, path, sha256), bound to the L7a summary's shard receipts."""
    summ = json.loads(L7A_SUMMARY.read_bytes())
    for s in summ["shard_receipts"]:
        if sha_file(s["path"]) != s["sha256"]:
            raise SystemExit(f"L7a shard receipt drift: {s['path']}")
    out = {}
    for p in sorted(glob.glob(str(L7A_OUT / "shard-*" / "_datasets" / "*.json"))):
        b = Path(p).read_bytes()
        r = json.loads(b)
        if r["dataset_hash"] in out:
            raise SystemExit(f"two L7a records for {r['dataset_hash']}")
        out[r["dataset_hash"]] = (r, p, sha(b))
    return out


def recount(rows):
    """The scope per class from the receipts. Returns (in-scope entries, per-class accounting)."""
    acq = json.loads(ACQ.read_bytes())
    leak = {}
    for l in LEAK.read_bytes().decode().splitlines():
        if l.strip():
            r = json.loads(l)
            leak[r["view_id"]] = r["row_selection_leak"]
    held, _ = held_set()
    recs = l7a_records()
    by = collections.defaultdict(list)
    for r in rows:
        by[r["dataset_hash"]].append(r)
    acct = {c: collections.Counter() for c in CLASSES}
    scope = []
    for h in sorted(acq):
        a = acq[h]
        if a["class"] == "all_rows_coreset":
            cls = "all_rows_coreset"
        elif a["class"] == "kmeans" and a["cohort"] == "incoming" and a["reduced"]:
            cls = "acquisition_capped"
        else:
            continue
        ctx = sorted(by[h], key=lambda r: int(r["fold"]))
        n = len(ctx)
        acct[cls]["class_datasets"] += 1
        acct[cls]["class_contexts"] += n
        labels = {json.dumps(leak.get(r["view_id"])) for r in ctx}
        if labels != {json.dumps(LABEL_OF[cls])}:
            raise SystemExit(f"{h}: epoch-0 leak labels {labels} != {LABEL_OF[cls]} for class {cls}")
        if h in held:
            acct[cls]["held_no_license_datasets"] += 1
            acct[cls]["held_no_license_contexts"] += n
            continue
        rec = recs.get(h)
        st = rec[0]["status"] if rec else "no_l7a_record"
        if st != "matched":
            key = f"not_matched:{(rec[0].get('reason') if rec else st)}"
            acct[cls][key + "_datasets"] += 1
            acct[cls][key + "_contexts"] += n
            continue
        r, p, s = rec
        m = r["match"]
        scope.append({"dataset_hash": h, "class": cls, "cohort": a["cohort"],
                      "epoch0": [{"view_id": c["view_id"], "context_sha256": c["context_sha256"], "fold": int(c["fold"])}
                                 for c in ctx],
                      "epoch0_labels": LABEL_OF[cls], "match_record": {"path": p, "sha256": s},
                      "raw_rows": m["raw_rows"], "raw_rows_with_target": m["raw_rows"] - m["raw_rows_without_target"],
                      "header_width": r["read"]["header_width"], "acq_pre_selection_rows": a["pre_selection_rows"]})
        acct[cls]["in_scope_datasets"] += 1
        acct[cls]["in_scope_epoch0_contexts"] += n
    return scope, {c: dict(v) for c, v in acct.items()}


def est_cost(e) -> float:
    n = min(e["raw_rows_with_target"], rowsel_raw.PRECAP_ROWS)
    w = e["header_width"]
    return 50 * n * (min(w, 60) + 30) * 0.42e-6 + e["raw_rows"] * w * 1.5e-6


def plan(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    src = scale_check.Sources(load_index=False)
    scope, acct = recount(src.included())
    loads = [0.0] * a.of
    for e in sorted(scope, key=lambda e: (-est_cost(e), e["dataset_hash"])):     # LPT: deterministic balance
        i = min(range(a.of), key=lambda k: (loads[k], k))
        e["shard"] = i
        e["est_cpu_s"] = round(est_cost(e), 1)
        loads[i] += est_cost(e)
    scope.sort(key=lambda e: e["dataset_hash"])
    held, held_ref = held_set()
    doc = {"schema": PLAN_SCHEMA, "of": a.of, "datasets": scope, "accounting": acct, "epoch": src.epoch_ref,
           "inputs": {"acquisition_selection": ref(ACQ), "row_selection_leak": ref(LEAK), "l7a_summary": ref(L7A_SUMMARY),
                      "rights_gate": held_ref, "owner_decisions": {k: ref(v) for k, v in OWNER.items() if Path(v).exists()}},
           "shard_est_cpu_s": [round(x, 1) for x in loads],
           "precap_rows": rowsel_raw.PRECAP_ROWS, "outer_folds": rowsel_raw.OUTER_FOLDS, "budget": rowsel_raw.BUDGET}
    stamp_split_version(doc, a.split_version)
    p = out / "plan.json"
    body = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    if p.exists() and p.read_bytes() != body:
        raise SystemExit(f"{p} exists with different content; refusing to replace a pinned plan")
    if not p.exists():
        atomic(p, body)
    print(json.dumps({"plan": str(p), "sha256": sha(body), "accounting": acct, "shard_est_cpu_s": doc["shard_est_cpu_s"],
                      "in_scope": collections.Counter(e["class"] for e in scope)}))
    return 0


def load_plan(out: Path):
    b = (out / "plan.json").read_bytes()
    return json.loads(b), sha(b)


# --------------------------------------------------------------------------------------------------- one dataset

def load_inputs(e, src, cache, inv, held):
    """Every input of one dataset, each sha256-checked against its receipt. Raises RowselRawError on a mismatch."""
    h = e["dataset_hash"]
    if h in held:
        raise rowsel_raw.RowselRawError("held_no_license")                      # before any read of its source
    mb_path = Path(e["match_record"]["path"])
    mb = mb_path.read_bytes()
    if sha(mb) != e["match_record"]["sha256"]:
        raise rowsel_raw.RowselRawError("match_record_drift", str(mb_path))
    rec = json.loads(mb)
    k = inv["meta_key"][h].rsplit("/", 1)[1]
    meta_b = (META_DIR / f"{k}.json").read_bytes()
    if sha(meta_b) != k or rec["t1_meta_sha256"] != k:
        raise rowsel_raw.RowselRawError("meta_sha_mismatch", k)
    fdir = RAW_DIR / h
    if not (fdir / ".done").exists():
        raise rowsel_raw.RowselRawError("raw_not_done")
    fj = json.loads((fdir / "fetch.json").read_bytes())
    raw_path = Path(fj["file"])
    if raw_path.parent.resolve() != fdir.resolve():
        raise rowsel_raw.RowselRawError("raw_outside_dataset_dir", str(raw_path))
    got = sha_file(raw_path)
    if got != fj.get("sha256") or got != rec["raw"]["sha256"]:
        raise rowsel_raw.RowselRawError("raw_custody_mismatch", f"{got} fetch {fj.get('sha256')} l7a {rec['raw']['sha256']}")
    rows = [r for r in src.included() if r["dataset_hash"] == h]
    rows.sort(key=lambda r: int(r["fold"]))
    if [(r["view_id"], r["context_sha256"]) for r in rows] != [(c["view_id"], c["context_sha256"]) for c in e["epoch0"]]:
        raise rowsel_raw.RowselRawError("epoch0_contexts_differ_from_plan")
    c0 = scale_check.resolve_context(rows[0], src, cache)
    if c0["refs"]["data_sha256"] != rec["data_sha256"]:
        raise rowsel_raw.RowselRawError("data_object_differs_from_matched", c0["refs"]["data_sha256"])
    raw_ref = {"file": str(raw_path), "sha256": got, "bytes": raw_path.stat().st_size, "fetch_status": fj.get("status"),
               "url": fj.get("url"), "byte_identical_to_original": got.startswith(h)}
    return {"meta": json.loads(meta_b), "meta_sha256": k, "raw_path": raw_path, "raw_ref": raw_ref, "record": rec,
            "A": c0["A"], "rows": rows, "sealed": sum(1 for r in rows if r["context_sha256"] in src.sealed_ctx)}


def rematch(inp):
    """read_source + match on the same raw bytes; must reproduce the L7a dataset record. Returns (values, specs, read)."""
    rec = inp["record"]
    specs = t1raw.columns(inp["meta"], inp["A"].shape[1])
    got = t1raw.read_source(inp["raw_path"], inp["meta"], specs)
    m = t1raw.match(inp["A"], specs, got["values"])
    if not m["matched"]:
        raise rowsel_raw.RowselRawError("rematch_unmatched", json.dumps({k: m["stats"][k] for k in ("unmatched_rows", "ambiguous_rows", "oversubscribed_rows")}))
    vs = sha(np.ascontiguousarray(t1raw.matched_values(m), "<f8").tobytes())
    if vs != rec["values_sha256"]:
        raise rowsel_raw.RowselRawError("rematch_values_differ", vs)
    if got["report"] != rec["read"]:
        raise rowsel_raw.RowselRawError("read_report_differs")
    return got["values"], specs, {"values_sha256": vs, "read_report_equal": True, "match_stats_equal": m["stats"] == rec["match"]}


def match_summary(st):
    keys = ("rows", "matched_rows", "raw_rows", "raw_rows_without_target", "raw_value_classes", "print_class", "residual", "tol_rel")
    return {k: st.get(k) for k in keys}


def run_dataset(e, out_shard: Path, codes, src, cache, inv, held, guard, select_fn=None, log=print, loader=None, matcher=None,
                post_gate=None, split_version=None):
    """Build one dataset. Returns its dataset record (also written to out_shard/_datasets/<h>.json). `loader` and
    `matcher` replace load_inputs/rematch for a scope without an L7a match record (tools/v2_rowsel_raw_coverage.py);
    e['context_extra'] (plan data, e.g. rights) is added to the dataset record and to every context.json. `post_gate`
    (inp, prep, folds, universe_ref, split_ref, results) -> dict runs after every fold passed and before anything is
    written; its dict goes to the dataset record, and passed=False refuses the dataset like a failed fold.
    `split_version`: the version the caller requested; required (= v1) for an entry that names none (split_functions)."""
    h = e["dataset_hash"]
    t0, c0 = time.time(), cpu_s()
    rec = {"schema": DATASET_SCHEMA, "dataset_hash": h, "class": e["class"], "cohort": e["cohort"], "code_sha256": codes,
           "epoch0_replaced": None, "status": "failed", "published": False}
    rec.update(e.get("context_extra") or {})
    contexts = []
    try:
        split_version, split_fn, split_report_fn = split_functions(e, split_version)   # before any read
        rec["split_version"] = split_version
        inp = (loader or load_inputs)(e, src, cache, inv, held)
        values, specs, rematch_rep = (matcher or rematch)(inp)
        rec["rematch"] = rematch_rep
        prep = rowsel_raw.prepare_universe(h, values, specs)
        del values
        inp["A"] = None
        rec["universe"] = prep.report
        if split_version == SPLIT_V2G:                                   # dataset-level admission BEFORE any split
            rec["group_admission"] = v2g_admission(e, prep)
        folds = split_fn(h, prep.positions, prep.values, specs)
        srep = split_report_fn(prep, folds)
        srep["target_check"] = rowsel_raw.split_target_check(split_fn, prep)
        if split_version == SPLIT_V2G:
            srep["group_value_check_fold0"] = v2g_group_value_check(split_fn, prep, e["group_column"]["index"])
        rec["split"] = srep
        if srep["groups_straddling_folds"] or not srep["target_check"]["target_invariant"]:
            raise rowsel_raw.RowselRawError("split_check_failed", json.dumps(srep["target_check"]))
        if split_version == SPLIT_V2G:
            v2g_gates(srep, e)
        elif min(srep["fold_sizes"]) == 0:
            raise rowsel_raw.RowselRawError("empty_outer_fold", json.dumps(srep["fold_sizes"]))
        built_folds = [f for f in range(rowsel_raw.OUTER_FOLDS) if srep["fold_sizes"][f]]      # v2g: trailing empty folds
        mrec = e["match_record"]
        universe_ref = {"raw_sha256s": [inp["raw_ref"]["sha256"]], "match_receipt_sha256": mrec["sha256"],
                        "match_receipt": mrec["path"], "partition": rowsel_raw.PARTITION,
                        "rows_before_precap": prep.report["precap"]["rows_before"],
                        "rows_after_precap": prep.report["precap"]["rows_after"], "precap_applied": prep.report["precap"]["applied"],
                        "positions_sha256": prep.report["positions_sha256"], "data_rows": prep.report["cleanup"]["data_rows"],
                        "cleanup": {k: prep.report["cleanup"][k] for k in ("no_target", "unmapped_category_with_target")}}
        split_ref = {k: srep[k] for k in ("schema", "sha256", "fold_sizes", "groups", "largest_group", "groups_straddling_folds",
                                          "domain", "rule")} | {"target_invariant": srep["target_check"]["target_invariant"]}
        if split_version == SPLIT_V2G:
            split_ref |= {k: srep[k] for k in V2G_SPLIT_KEYS} | {"group_value_invariant": srep["group_value_check_fold0"]["invariant"]}
        e0 = {"level": "dataset", "contexts": e["epoch0"], "count": len(e["epoch0"]),
              "fold_correspondence": "none: a new value-free outer split (ROWSEL-SPLIT); new fold f is not epoch-0 fold f",
              "new_contexts": len(built_folds), "context_count_changed": len(e["epoch0"]) != len(built_folds),
              "epoch0_sealed_contexts": inp["sealed"]}
        rec["epoch0_replaced"] = e0
        common = {"t1_meta_sha256": inp["meta_sha256"], "data_sha256": inp["record"]["data_sha256"], "raw_source": inp["raw_ref"],
                  "match": match_summary(inp["record"]["match"]), "match_receipt": mrec, "class": e["class"],
                  "cohort": e["cohort"], "epoch0_row_selection_leak": e["epoch0_labels"], "epoch0_replaced": e0,
                  "code_sha256": codes, "rowsel_universe": universe_ref, "rowsel_split": split_ref}
        common.update(e.get("context_extra") or {})
        if split_version == SPLIT_V2G:
            common["cut"] = v2g_cut(srep, e)
            common["split_version"] = SPLIT_V2G
        results = []
        for f in built_folds:
            guard.wait()
            tf, cf = time.time(), cpu_s()
            kw = {} if select_fn is None else {"select_fn": select_fn}
            r = rowsel_raw.build_fold(prep, folds, f, universe_ref, split_ref, **kw)
            files = rowsel_raw.context_files(r)
            crec = rowsel_raw.context_record(r, prep, files, common)
            tlg = None
            if split_version != SPLIT_V1 and r["passed"]:                 # LINEAR-DETERMINED gate (v1 is a byte-identical replay)
                tlg = target_linear.check_jtt1(r["artifacts"]["encoder_raw"])
                crec["target_linear"] = tlg
            results.append((r, files, crec))
            contexts.append({"fold": f, "view_id": r["view_id"], "identity": r["identity"], "passed": r["passed"],
                             "failures": r["failures"], "quotas": list(r["view"].quotas), "n_train": crec["n_train"],
                             "n_test": crec["n_test"], "context_sha256": crec["context_sha256"],
                             "cpu_s": round(cpu_s() - cf, 2), "wall_s": round(time.time() - tf, 2)}
                            | ({"target_linear": {"verdict": tlg["verdict"], "ols_rel_rmse": tlg["ols_rel_rmse"]}} if tlg else {}))
            log(json.dumps({"dataset": h, "fold": f, "passed": r["passed"], "failures": r["failures"],
                            "cpu_s": contexts[-1]["cpu_s"], "maxrss_mb": maxrss_mb()}))
            if not r["passed"]:
                break                                                    # the dataset cannot be admitted any more
        rec["contexts"] = contexts
        failed = [c for c in contexts if not c["passed"]]
        if failed:
            raise rowsel_raw.RowselRawError("counterfactual_failed", json.dumps({c["fold"]: c["failures"] for c in failed}))
        lin = {c["fold"]: c["target_linear"]["ols_rel_rmse"] for c in contexts
               if (c.get("target_linear") or {}).get("verdict") == target_linear.REFUSE}
        if lin:                                                          # refused: no context of the dataset is written
            raise rowsel_raw.RowselRawError(target_linear.REFUSE, json.dumps(lin))
        if post_gate is not None:
            guard.wait()
            g = post_gate(inp, prep, folds, universe_ref, split_ref, results)
            rec["post_gate"] = g
            if not g.get("passed"):
                raise rowsel_raw.RowselRawError("post_gate_failed", json.dumps(g)[:1500])
        pending = []                                                     # every fold passed
        for r, files, crec in results:
            d = out_shard / r["view_id"]
            if (d / ".done").exists():
                old = json.loads((d / "context.json").read_bytes())
                if old.get("rowsel", {}).get("identity") != r["identity"] or old.get("files") != crec["files"]:
                    raise rowsel_raw.RowselRawError("done_context_differs", r["view_id"])
                continue
            pending.append((d, files, crec))
        for d, files, crec in pending:                                   # phase 1: every file of every context
            d.mkdir(parents=True, exist_ok=True)
            for name, body in files.items():
                atomic(d / name, body)
            atomic(d / "context.json", jdump(dict(crec, utc_us=int(time.time() * 1e6))))
        for d, _, _ in pending:                                          # phase 2: .done last, dataset-wide
            atomic(d / ".done", b"ok\n")
        rec.update(status="done", contexts_written=len(pending), contexts_already_done=len(results) - len(pending), reason=None)
    except (rowsel_raw.RowselRawError, t1raw.T1RawError) as err:
        rec.update(reason=err.reason, detail=err.detail[:2000])
    except Exception as err:  # recorded; the shard continues
        rec.update(reason="exception", detail=f"{type(err).__name__}: {err}"[:2000])
    rec.setdefault("contexts", contexts)
    rec.update(cpu_s=round(cpu_s() - c0, 2), wall_s=round(time.time() - t0, 2), maxrss_mb=maxrss_mb(), utc_us=int(time.time() * 1e6))
    (out_shard / "_datasets").mkdir(parents=True, exist_ok=True)
    atomic(out_shard / "_datasets" / f"{h}.json", jdump(rec))
    return rec


def dataset_done(out_shard: Path, h: str, codes) -> bool:
    p = out_shard / "_datasets" / f"{h}.json"
    if not p.exists():
        return False
    r = json.loads(p.read_bytes())
    return (r.get("status") == "done" and r.get("code_sha256") == codes
            and all((out_shard / c["view_id"] / ".done").exists() for c in r.get("contexts", [])))


def build(a):
    cpuguard.set_threads(2)
    out = Path(a.out)
    pl, plan_sha = load_plan(out)
    if pl["of"] != a.of:
        raise SystemExit(f"plan is for --of {pl['of']}")
    plan_split_version(pl, a.split_version)
    out_shard = out / f"shard-{a.shard}"
    out_shard.mkdir(parents=True, exist_ok=True)
    codes = code_shas()
    src = scale_check.Sources(load_index=False)
    cache = scale_check.DataCache(max_items=2)
    inv = json.loads(DATASETS_INV.read_text())
    held, held_ref = held_set()
    mine = [e for e in pl["datasets"] if e.get("disposition", "build") == "build" and e["shard"] == a.shard]
    if a.datasets:
        keep = set(a.datasets.split(","))
        mine = [e for e in mine if e["dataset_hash"] in keep]
    guard = cpuguard.Guard(journal_path=out / f"shard-{a.shard}.guard.jsonl")
    t0, c0 = time.time(), cpu_s()
    counts, failed, done_now, skipped = collections.Counter(), [], [], []
    for n, e in enumerate(sorted(mine, key=lambda e: -e["est_cpu_s"])):
        h = e["dataset_hash"]
        if dataset_done(out_shard, h, codes):
            skipped.append(h)
            counts["already_done"] += 1
            continue
        guard.wait()
        rec = run_dataset(e, out_shard, codes, src, cache, inv, held, guard, split_version=a.split_version, **route_hooks(a, out, codes))
        if rec["status"] == "done":
            done_now.append(h)
            counts["done"] += 1
        else:
            failed.append({"dataset_hash": h, "class": e["class"], "reason": rec.get("reason"), "detail": rec.get("detail", "")[:500],
                           "epoch0_contexts_kept": len(e["epoch0"])})
            counts["failed"] += 1
        print(json.dumps({"shard": a.shard, "n": n + 1, "of": len(mine), "dataset": h, "status": rec["status"],
                          "reason": rec.get("reason"), "cpu_s": rec["cpu_s"], "elapsed_s": round(time.time() - t0)}), flush=True)
    by_class = collections.defaultdict(collections.Counter)
    for e in mine:
        p = out_shard / "_datasets" / f"{e['dataset_hash']}.json"
        st = json.loads(p.read_bytes())["status"] if p.exists() else "missing"
        by_class[e["class"]][st] += 1
    rec = {"schema": SHARD_SCHEMA, "shard": a.shard, "of": a.of, "plan_sha256": plan_sha, "datasets_filter": a.datasets,
           "expected_datasets": len(mine), "done_now": len(done_now), "already_done": len(skipped), "failed": failed,
           "status_by_class": {k: dict(v) for k, v in by_class.items()},
           "complete": not failed and len(done_now) + len(skipped) == len(mine),
           "cpu_s": round(cpu_s() - c0, 1), "wall_s": round(time.time() - t0, 1), "maxrss_mb": maxrss_mb(),
           "guard": guard.summary(), "published": False, "code_sha256": codes, "rights_gate": held_ref,
           "utc_us": int(time.time() * 1e6)}
    name = f"shard-{a.shard}-of-{a.of}.receipt.json" if not a.datasets else f"shard-{a.shard}-of-{a.of}.partial-{int(time.time())}.receipt.json"
    atomic(out / name, (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps({k: rec[k] for k in ("shard", "expected_datasets", "done_now", "already_done", "complete", "cpu_s", "wall_s")}
                     | {"failed": len(failed)}))
    return 0 if rec["complete"] else 1


# --------------------------------------------------------------------------------------------------- summary

def summarize(a):
    out = Path(a.out)
    pl, plan_sha = load_plan(out)
    codes = code_shas()
    version = plan_split_version(pl, a.split_version)
    per_class = {c: collections.Counter() for c in CLASSES}
    failures = collections.defaultdict(list)
    datasets, cpu_ds, ctx_dirs = [], 0.0, set()
    orphan = []
    for e in pl["datasets"]:
        h, cls = e["dataset_hash"], e["class"]
        sd = out / f"shard-{e['shard']}"
        p = sd / "_datasets" / f"{h}.json"
        pc = per_class[cls]
        pc["expected_datasets"] += 1
        pc["expected_epoch0_contexts"] += len(e["epoch0"])
        pc["expected_new_contexts"] += rowsel_raw.OUTER_FOLDS
        if not p.exists():
            pc["not_built_datasets"] += 1
            failures[cls].append({"dataset_hash": h, "reason": "not_built"})
            continue
        r = json.loads(p.read_bytes())
        cpu_ds += r.get("cpu_s", 0.0)
        if r["status"] == "done":
            ok = code_known(r.get("code_sha256"), codes)
            if (r.get("split") or {}).get("schema") != version or r.get("split_version", SPLIT_V1) != version:
                raise SystemExit(f"{h}: split {(r.get('split') or {}).get('schema')} is not the plan's {version}")
            for c in r["contexts"]:
                d = sd / c["view_id"]
                ctx_dirs.add(str(d))
                cj = json.loads((d / "context.json").read_bytes()) if (d / ".done").exists() else None
                if not cj or cj["rowsel"]["identity"] != c["identity"] or cj["context_sha256"] != c["context_sha256"]:
                    ok = False
                for name, meta in (cj or {}).get("files", {}).items():
                    if sha_file(d / name) != meta["sha256"]:
                        ok = False
            if not ok:
                raise SystemExit(f"{h}: done record does not match its context files or the current code")
            pc["done_datasets"] += 1
            pc["done_new_contexts"] += len(r["contexts"])
            pc["epoch0_contexts_replaced"] += len(e["epoch0"])
            pc["precap_applied_datasets"] += int(r["universe"]["precap"]["applied"])
            pc["context_count_changed_datasets"] += int(r["epoch0_replaced"]["context_count_changed"])
        else:
            pc["failed_datasets"] += 1
            pc["failed_epoch0_contexts_kept"] += len(e["epoch0"])
            pc[f"failed_reason:{r.get('reason')}"] += 1
            failures[cls].append({"dataset_hash": h, "reason": r.get("reason"), "detail": (r.get("detail") or "")[:400],
                                  "epoch0_contexts_kept": len(e["epoch0"])})
        datasets.append({"dataset_hash": h, "class": cls, "shard": e["shard"], "status": r["status"], "reason": r.get("reason"),
                         "record_sha256": sha_file(p), "universe_rows": (r.get("universe") or {}).get("rows"),
                         "rows_before_precap": ((r.get("universe") or {}).get("precap") or {}).get("rows_before"),
                         "fold_sizes": (r.get("split") or {}).get("fold_sizes"), "split_sha256": (r.get("split") or {}).get("sha256"),
                         "epoch0_contexts": len(e["epoch0"]), "new_view_ids": [c["view_id"] for c in r.get("contexts", [])] if r["status"] == "done" else [],
                         "cpu_s": r.get("cpu_s"), "maxrss_mb": r.get("maxrss_mb")})
    for sd in sorted(out.glob("shard-*")):
        if sd.is_dir():
            for d in sd.iterdir():
                if d.is_dir() and d.name != "_datasets" and str(d) not in ctx_dirs:
                    orphan.append(str(d))
    shard_receipts = []
    cpu_shards = 0.0
    for p in sorted(out.glob(f"shard-*-of-{a.of}.receipt.json")):
        r = json.loads(p.read_bytes())
        cpu_shards += r["cpu_s"]
        shard_receipts.append({"path": str(p), "sha256": sha_file(p), "expected_datasets": r["expected_datasets"],
                               "done_now": r["done_now"], "already_done": r["already_done"], "failed": len(r["failed"]),
                               "complete": r["complete"], "cpu_s": r["cpu_s"], "wall_s": r["wall_s"], "maxrss_mb": r["maxrss_mb"],
                               "guard": r["guard"], "plan_sha256": r["plan_sha256"]})
    mut = json.loads(Path(a.mutants).read_bytes())
    root = Path(__file__).resolve().parents[1]
    doc = {"schema": SUMMARY_SCHEMA, "utc_us": int(time.time() * 1e6), "out": str(out), "of": a.of, "split_version": version,
           "plan": {"path": str(out / "plan.json"), "sha256": plan_sha, "accounting": pl["accounting"], "inputs": pl["inputs"]},
           "counts_per_class": {c: dict(v) for c, v in per_class.items()}, "failures": dict(failures),
           "orphan_context_dirs": orphan, "datasets": datasets,
           "cpu": {"dataset_cpu_s": round(cpu_ds, 1), "shard_process_cpu_s": round(cpu_shards, 1),
                   "note": "getrusage user+sys of the shard processes (lane slice)"},
           "shard_receipts": shard_receipts,
           "tests": {"log": ref(a.tests_log), "tail": Path(a.tests_log).read_text().strip().splitlines()[-3:]},
           "mutants": {"path": a.mutants, "sha256": sha_file(a.mutants), "totals": mut.get("totals"),
                       "per_mutant": {k: {"killed": v["killed"], "failing": v["failing"], "leak_mutant": v.get("leak_mutant"),
                                          "reason": v.get("reason")} for k, v in mut.get("mutants", {}).items()},
                       "excluded_equivalent": mut.get("excluded_equivalent")},
           "pilot": json.loads(Path(a.pilot).read_bytes()) if a.pilot else None,
           "code_sha256": codes | {"tests/test_v2_rowsel_raw.py": sha_file(root / "tests/test_v2_rowsel_raw.py")},
           "policy": {"s3_calls": 0, "network": False, "gpu": False, "git": False, "published": False,
                      "held_sources_read": False, "credential_files_read": False}}
    stem = "rowsel-raw-summary" if version == SPLIT_V1 else "rowsel-raw-v2-summary"
    rp = Path(a.receipt) if a.receipt else RECEIPTS / "build" / f"{stem}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    data = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    with open(rp, "xb") as f:
        f.write(data)
    complete = not orphan and all(not v.get("not_built_datasets") for v in per_class.values())
    marker = write_complete(out, rp, {c: dict(v) for c, v in per_class.items()}, version) if should_mark_complete(complete, version) else None
    print(json.dumps({"receipt": str(rp), "sha256": sha(data), "counts_per_class": doc["counts_per_class"], "orphans": len(orphan),
                      "complete_marker": str(marker) if marker else None}))
    return 0


# --------------------------------------------------------------------------------------------------- route: nonnumeric

OUT_NONNUMERIC = RUN / "build-v2" / "ROWSEL-RAW-v1-nonnumeric"
NN_STUDY = RECEIPTS / "coreset-nonnumeric" / "all185-nonnumeric.jsonl"
NN_STUDY_SHA256 = "30b5241254ef05477b842d126743e06b7c35368e2d1fc46004f6ea05bcca7bee"
NN_DECISION = Path("/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261006T000114Z.json")
NN_DECISION_SHA256 = "673025afd4b043df38d7fa52017449b3636a345654814be2bfaaadbe32a01dc1"
NN_TIERS = ("matched", "matched_set_level", "matched_covered")
NN_MATCH_SCHEMA = "jope.dl-v2-rowsel-raw-nonnumeric-match/v1"
NN_PLAN_SCHEMA = "jope.dl-v2-rowsel-raw-nonnumeric-plan/v1"
NN_SUMMARY_SCHEMA = "jope.dl-v2-rowsel-raw-nonnumeric-summary/v1"
NN_CONTROL_SALT = "rowsel-raw-nonnumeric-text-cf-v1"


def plan_nonnumeric(a):
    """CORESET-NONNUMERIC scope from the study's per-dataset outcomes (pinned by sha256): every dataset whose outcome is
    one of NN_TIERS is built; the others are listed with the study's reason. Each must be a frozen all_rows_coreset
    dataset whose L7a match failed with rows_unmatched_nonnumeric_raw."""
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if sha_file(NN_STUDY) != NN_STUDY_SHA256 or sha_file(NN_DECISION) != NN_DECISION_SHA256:
        raise SystemExit("CORESET-NONNUMERIC study or owner record drift")
    study = [json.loads(l) for l in NN_STUDY.read_bytes().decode().splitlines() if l.strip()]
    acq = json.loads(ACQ.read_bytes())
    held, held_ref = held_set()
    recs = l7a_records()
    src = scale_check.Sources(load_index=False)
    by = collections.defaultdict(list)
    for r in src.included():
        by[r["dataset_hash"]].append(r)
    entries = []
    for s in sorted(study, key=lambda r: r["dataset_hash"]):
        h = s["dataset_hash"]
        rec = recs[h][0]
        if acq[h]["class"] != "all_rows_coreset" or rec["status"] != "unmatched" or rec["reason"] != "rows_unmatched_nonnumeric_raw":
            raise SystemExit(f"{h}: not a rows_unmatched_nonnumeric_raw coreset dataset")
        ctx = sorted(by[h], key=lambda r: int(r["fold"]))
        e = {"dataset_hash": h, "class": "all_rows_coreset", "cohort": acq[h]["cohort"], "epoch0_labels": LABEL_OF["all_rows_coreset"],
             "epoch0": [{"view_id": c["view_id"], "context_sha256": c["context_sha256"], "fold": int(c["fold"])} for c in ctx],
             "study": {"status": s["status"], "reason": s["reason"], "raw_sha256": s["raw_sha256"]},
             "l7a_record": {"path": recs[h][1], "sha256": recs[h][2], "raw_sha256": (rec.get("raw") or {}).get("sha256")},
             "raw_rows": (s.get("stats") or {}).get("raw_rows"), "header_width": (rec.get("read") or {}).get("header_width") or 20,
             "match_record": None}
        if h in held:
            e.update(disposition="not_built", reason="held_no_license")
        elif s["status"] in NN_TIERS:
            e.update(disposition="build", nonnumeric_tier=s["status"],
                     context_extra={"values_route": "nonnumeric", "nonnumeric_tier": s["status"],
                                    "nonnumeric_decision": f"{NN_DECISION}#sha256={NN_DECISION_SHA256}"})
        else:
            e.update(disposition="not_built", reason=s["reason"] or s["status"])
        entries.append(e)
    loads = [0.0] * a.of
    for e in sorted((e for e in entries if e["disposition"] == "build"),
                    key=lambda e: (-est_cost({"raw_rows": e["raw_rows"], "raw_rows_with_target": e["raw_rows"],
                                              "header_width": e["header_width"]}), e["dataset_hash"])):
        c = est_cost({"raw_rows": e["raw_rows"], "raw_rows_with_target": e["raw_rows"], "header_width": e["header_width"]})
        i = min(range(a.of), key=lambda k: (loads[k], k))
        e["shard"], e["est_cpu_s"] = i, round(c, 1)
        loads[i] += c
    cnt = collections.Counter(f"{e['disposition']}:{e.get('nonnumeric_tier') or e.get('reason')}" for e in entries)
    doc = {"schema": NN_PLAN_SCHEMA, "route": "nonnumeric", "of": a.of, "datasets": entries,
           "counts": {"datasets": len(entries), "contexts": sum(len(e["epoch0"]) for e in entries), "by": dict(sorted(cnt.items())),
                      "build_datasets": sum(e["disposition"] == "build" for e in entries),
                      "build_epoch0_contexts": sum(len(e["epoch0"]) for e in entries if e["disposition"] == "build")},
           "inputs": {"study": ref(NN_STUDY), "owner_decision": ref(NN_DECISION), "acquisition_selection": ref(ACQ),
                      "l7a_summary": ref(L7A_SUMMARY), "rights_gate": held_ref},
           "epoch": src.epoch_ref, "shard_est_cpu_s": [round(x, 1) for x in loads], "admitted_tiers": list(NN_TIERS),
           "precap_rows": rowsel_raw.PRECAP_ROWS, "outer_folds": rowsel_raw.OUTER_FOLDS, "budget": rowsel_raw.BUDGET}
    stamp_split_version(doc, a.split_version)
    p = out / "plan.json"
    body = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    if p.exists() and p.read_bytes() != body:
        raise SystemExit(f"{p} exists with different content; refusing to replace a pinned plan")
    if not p.exists():
        atomic(p, body)
    print(json.dumps({"plan": str(p), "sha256": sha(body), "counts": doc["counts"], "shard_est_cpu_s": doc["shard_est_cpu_s"]}))
    return 0


def make_nonnumeric_loader(out: Path, codes):
    """load_inputs for the nonnumeric route: the same rights gate (first, before any read), meta / raw / epoch-0 checks
    (raw sha256 == fetch.json == L7a record == study), then nonnumeric.match_dataset. Its match receipt (tier, per-column
    readings, values sha256) is written to out/_match/<h>.json and pinned in e['match_record']. Refused unless the tier
    is admitted and equals the study's."""
    def load(e, src, cache, inv, held):
        h = e["dataset_hash"]
        if h in held:
            raise rowsel_raw.RowselRawError("held_no_license")                      # before any read of its source
        k = inv["meta_key"][h].rsplit("/", 1)[1]
        meta_b = (META_DIR / f"{k}.json").read_bytes()
        if sha(meta_b) != k:
            raise rowsel_raw.RowselRawError("meta_sha_mismatch", k)
        meta = json.loads(meta_b)
        fdir = RAW_DIR / h
        if not (fdir / ".done").exists():
            raise rowsel_raw.RowselRawError("raw_not_done")
        fj = json.loads((fdir / "fetch.json").read_bytes())
        raw_path = Path(fj["file"])
        if raw_path.parent.resolve() != fdir.resolve():
            raise rowsel_raw.RowselRawError("raw_outside_dataset_dir", str(raw_path))
        got = sha_file(raw_path)
        pinned = {"fetch.json": fj.get("sha256"), "l7a_record": (e.get("l7a_record") or {}).get("raw_sha256"),
                  "study": (e.get("study") or {}).get("raw_sha256")}
        if any(v is not None and v != got for v in pinned.values()) or pinned["fetch.json"] is None:
            raise rowsel_raw.RowselRawError("raw_custody_mismatch", f"{got} != {pinned}")
        rows = sorted((r for r in src.included() if r["dataset_hash"] == h), key=lambda r: int(r["fold"]))
        if [(r["view_id"], r["context_sha256"]) for r in rows] != [(c["view_id"], c["context_sha256"]) for c in e["epoch0"]]:
            raise rowsel_raw.RowselRawError("epoch0_contexts_differ_from_plan")
        ctxs = [scale_check.resolve_context(r, src, cache) for r in rows]
        data = {c["refs"]["data_sha256"] for c in ctxs}
        if len(data) != 1:
            raise rowsel_raw.RowselRawError("epoch0_contexts_on_different_data_objects", json.dumps(sorted(data)))
        A = ctxs[0]["A"]
        raw_ref = {"file": str(raw_path), "sha256": got, "bytes": raw_path.stat().st_size, "fetch_status": fj.get("status"),
                   "url": fj.get("url"), "byte_identical_to_original": got.startswith(h)}
        m = nonnumeric.match_dataset(raw_path, meta, A)
        readings = {n: c["reading"] for n, c in (m.get("text_columns") or {}).items()}
        mrec = {"schema": NN_MATCH_SCHEMA, "dataset_hash": h, "tier": m["status"], "status": m["status"], "reason": m["reason"],
                "detail": m.get("detail"), "data_sha256": data.pop(), "t1_meta_sha256": k, "raw": raw_ref, "read": m.get("read"),
                "match": m.get("stats"), "attempts": m.get("attempts"), "ambiguity": m.get("ambiguity"),
                "text_columns": m.get("text_columns"), "readings": readings, "target_name_conflict": m.get("target_name_conflict"),
                "staging_rule": nonnumeric.STAGING_RULE, "universe_rule": nonnumeric.UNIVERSE_RULE, "universe": m.get("universe"),
                "values_sha256": sha(np.ascontiguousarray(m["values"], "<f8").tobytes()) if "values" in m else None,
                "study_tier": (e.get("study") or {}).get("status"), "code_sha256": codes}   # no timestamp: deterministic,
        (out / "_match").mkdir(parents=True, exist_ok=True)                                  # so view ids reproduce
        mp = out / "_match" / f"{h}.json"
        body = jdump(mrec)
        atomic(mp, body)
        e["match_record"] = {"path": str(mp), "sha256": sha(body)}
        if m["status"] not in NN_TIERS:
            raise rowsel_raw.RowselRawError(m["reason"] or "unmatched", json.dumps(m.get("stats"))[:500])
        if m["status"] != e.get("nonnumeric_tier"):
            raise rowsel_raw.RowselRawError("nonnumeric_tier_differs_from_study", f"{m['status']} != {e.get('nonnumeric_tier')}")
        return {"meta": meta, "meta_sha256": k, "raw_path": raw_path, "raw_ref": raw_ref, "record": mrec, "A": A, "rows": rows,
                "sealed": sum(1 for r in rows if r["context_sha256"] in src.sealed_ctx), "_width": int(A.shape[1])}
    return load


def nonnumeric_matcher(inp):
    """The universe re-derived from the raw file and the receipt's pinned readings (nonnumeric.read_universe); refused
    unless its bytes equal the receipt's values_sha256."""
    rec = inp["record"]
    V, uspecs, urep = nonnumeric.read_universe(inp["raw_path"], inp["meta"], inp["_width"], rec["readings"])
    vs = sha(np.ascontiguousarray(V, "<f8").tobytes())
    if vs != rec["values_sha256"]:
        raise rowsel_raw.RowselRawError("nonnumeric_universe_differs", vs)
    return V, uspecs, {"values_sha256": vs, "nonnumeric_tier": rec["tier"], "readings": rec["readings"], "universe": urep}


def all_row_rank_universe(values, specs, text, readings):
    """MUST-FAIL CONTROL ONLY: the acquisition's all-row staging rank kept as the column's value (a numeric spec)."""
    V = np.array(values, dtype=np.float64, copy=True)
    out = [dict(s) for s in specs]
    for n, cells in text.items():
        if readings[n] != nonnumeric.RANK:
            continue
        j = next(i for i, s in enumerate(out) if s["name"] == n)
        rank = nonnumeric.staging_rank(cells)
        V[:, out[j]["index"]] = [np.nan if nonnumeric.is_na(t) else float(rank[t]) for t in cells]
        out[j] = {"index": out[j]["index"], "name": n, "source_name": out[j]["source_name"], "kind": "numeric",
                  "lo": 0.0, "hi": float(max(len(rank) - 1, 1)), "impute_value": None, "derive": None}
    return V, out


def _ctx_train_side(r):
    art, v = r["artifacts"], r["view"]
    return {"train_order": r["train_order"], "train_fit_identity": v.train_fit.identity, "quotas": list(v.quotas),
            "encoder_train_rows": jtt1.train_rows_sha256(art["encoder_raw"]),
            "teacher_train_rows": jtt1.train_rows_sha256(art["teacher_raw"]), "scaler": art["scaler_sha256"],
            "inner_labels": art["inner_labels_sha256"], "a0": sha(art["a0"]), "sampler_category_maps": r["sampler_category_maps"]}


def nonnumeric_text_gate(inp, prep, folds, universe_ref, split_ref, results, fold: int = 0):
    """Text-level counterfactual on the real raw text: every outer-test (fold `fold`) cell of every rank text column is
    changed (1/3 -> a new text sorting before every outer-train text, 1/3 -> a new text sorting just after a middle
    outer-train text, 1/3 -> NA), the universe is rebuilt from the text and the context rebuilt with the split held fixed:
    the train side must be bit-identical (passed). MUST-FAIL CONTROL: the same perturbation with the all-row staging rank
    as the value must change the train side; the first-sorting text shifts every outer-train rank, so the control has
    power on a two-text column too."""
    h, rec = prep.dataset_hash, inp["record"]
    readings = rec["readings"]
    specs0 = t1raw.columns(inp["meta"], inp["_width"])
    src = t1raw.read_source(inp["raw_path"], inp["meta"], specs0)
    cand = nonnumeric.text_columns(specs0, src["report"])
    text = nonnumeric.confirm_text(specs0, nonnumeric.read_text(inp["raw_path"], inp["meta"], specs0, src["report"], cand))
    rank_cols = sorted(n for n in text if readings.get(n) == nonnumeric.RANK)
    if not rank_cols:
        return {"passed": True, "skipped": "no_rank_text_columns", "control_fails_as_required": None}
    base = results[fold][0]
    F = [int(p) for p in folds[fold]]
    R = base["train_order"]
    text2 = {n: list(c) for n, c in text.items()}
    inserted = {}
    for n in rank_cols:
        tt = sorted({text[n][p] for p in R if not nonnumeric.is_na(text[n][p])})
        first = "\x00test-only-first"
        middle = (tt[len(tt) // 2] + "\x00test-only") if tt else "test-only"
        if tt and not first < tt[0]:
            raise rowsel_raw.RowselRawError("text_gate_insert_not_first", n)
        inserted[n] = {"first": repr(first), "after_middle": repr(middle), "outer_train_texts": len(tt)}
        for k, p in enumerate(F):
            text2[n][p] = ("", first, middle)[k % 3]

    def ctx(V, sp):
        p2 = rowsel_raw.prepare_universe(h, V, sp)
        if p2.positions.tolist() != prep.positions.tolist():
            raise rowsel_raw.RowselRawError("text_gate_universe_moved")
        return rowsel_raw.build_fold(p2, folds, fold, universe_ref, split_ref, counterfactual=False)
    V2, us2, _ = nonnumeric.universe(src["values"], specs0, text2, readings)
    clean = ctx(V2, us2)
    want = _ctx_train_side(base)
    got = _ctx_train_side(clean)
    clean_diff = sorted(k for k in want if got[k] != want[k])
    Va, sa = all_row_rank_universe(src["values"], specs0, text, readings)
    Vb, sb = all_row_rank_universe(src["values"], specs0, text2, readings)
    ca, cb = _ctx_train_side(ctx(Va, sa)), _ctx_train_side(ctx(Vb, sb))
    ctrl_diff = sorted(k for k in ca if ca[k] != cb[k])
    return {"fold": fold, "rank_text_columns": rank_cols, "outer_test_rows_perturbed": len(F),
            "perturbation": "outer-test text: 1/3 -> '' (NA), 1/3 -> a new text sorting before every outer-train text, "
                            "1/3 -> a new text sorting just after a middle outer-train text", "inserted": inserted,
            "clean_train_side_differs": clean_diff, "clean_test_draw_identical": clean["test_order"] == base["test_order"],
            "control": "all-row staging rank as the value (the acquisition's encoder)", "control_train_side_differs": ctrl_diff,
            "control_fails_as_required": bool(ctrl_diff),
            "passed": not clean_diff and clean["test_order"] == base["test_order"]}


def route_hooks(a, out: Path, codes) -> dict:
    if getattr(a, "route", "l7a") == "nonnumeric":
        return {"loader": make_nonnumeric_loader(out, codes), "matcher": nonnumeric_matcher, "post_gate": nonnumeric_text_gate}
    return {}


def summarize_nonnumeric(a):
    out = Path(a.out)
    pl, plan_sha = load_plan(out)
    codes = code_shas()
    version = plan_split_version(pl, a.split_version)
    rows, cnt = [], collections.Counter()
    for e in pl["datasets"]:
        h, n0 = e["dataset_hash"], len(e["epoch0"])
        row = {"dataset_hash": h, "disposition": e["disposition"], "study_status": e["study"]["status"],
               "study_reason": e["study"]["reason"], "epoch0_contexts": n0, "epoch0_view_ids": [c["view_id"] for c in e["epoch0"]]}
        if e["disposition"] != "build":
            row.update(status="not_built", reason=e["reason"])
        else:
            sd = out / f"shard-{e['shard']}"
            p = sd / "_datasets" / f"{h}.json"
            if not p.exists():
                row.update(status="not_built", reason="no_dataset_record")
            else:
                r = json.loads(p.read_bytes())
                u = r.get("universe") or {}
                row.update(status=r["status"], reason=r.get("reason"), detail=(r.get("detail") or "")[:600], tier=e["nonnumeric_tier"],
                           record={"path": str(p), "sha256": sha_file(p)}, cpu_s=r.get("cpu_s"), maxrss_mb=r.get("maxrss_mb"),
                           raw_rows=(u.get("cleanup") or {}).get("data_rows"),
                           rows_lost_no_target=(u.get("cleanup") or {}).get("no_target"),
                           rows_lost_unmapped_number_keyed_category=(u.get("cleanup") or {}).get("unmapped_category_with_target"),
                           universe_rows=u.get("rows"), precap_applied=(u.get("precap") or {}).get("applied"),
                           text_gate={k: (r.get("post_gate") or {}).get(k) for k in ("rank_text_columns", "clean_train_side_differs",
                                                                                     "clean_test_draw_identical", "control_fails_as_required",
                                                                                     "control_train_side_differs", "passed", "skipped")})
                mp = out / "_match" / f"{h}.json"
                if mp.exists():
                    row["match_receipt"] = {"path": str(mp), "sha256": sha_file(mp)}
                if r["status"] == "done":
                    if not code_known(r.get("code_sha256"), codes):
                        raise SystemExit(f"{h}: built with other code")
                    if r["split"]["schema"] != version or r.get("split_version", SPLIT_V1) != version:
                        raise SystemExit(f"{h}: split {r['split']['schema']} is not the plan's {version}")
                    for c in r["contexts"]:
                        d = sd / c["view_id"]
                        cj = json.loads((d / "context.json").read_bytes())
                        if not (d / ".done").exists() or cj["rowsel"]["identity"] != c["identity"] or cj["row_selection_leak"] != ["clean"]:
                            raise SystemExit(f"{h}: {c['view_id']} not done, identity or label differs")
                        if (cj.get("values_route"), cj.get("nonnumeric_tier")) != ("nonnumeric", e["nonnumeric_tier"]):
                            raise SystemExit(f"{h}: {c['view_id']} lacks its nonnumeric route fields")
                        for name, meta in cj["files"].items():
                            if sha_file(d / name) != meta["sha256"]:
                                raise SystemExit(f"{h}: {name} hash mismatch")
                    row["new_view_ids"] = [c["view_id"] for c in r["contexts"]]
        cnt[f"datasets|{row['status']}|{row.get('tier') or row.get('reason')}"] += 1
        cnt[f"epoch0_contexts|{row['status']}"] += n0
        rows.append(row)
    done = [r for r in rows if r["status"] == "done"]
    shard_receipts = [ref(p) | {k: json.loads(Path(p).read_bytes())[k] for k in ("expected_datasets", "done_now", "already_done", "complete",
                                                                                "cpu_s", "wall_s", "maxrss_mb")}
                      for p in sorted(glob.glob(str(out / f"shard-*-of-{pl['of']}.receipt.json")))]
    mut = json.loads(Path(a.mutants).read_bytes())
    doc = {"schema": NN_SUMMARY_SCHEMA, "utc_us": int(time.time() * 1e6), "out": str(out), "split_version": version,
           "plan": {"path": str(out / "plan.json"), "sha256": plan_sha, "counts": pl["counts"], "inputs": pl["inputs"]},
           "counts": {"expected_build_datasets": pl["counts"]["build_datasets"],
                      "expected_build_epoch0_contexts": pl["counts"]["build_epoch0_contexts"],
                      "expected_new_contexts": rowsel_raw.OUTER_FOLDS * pl["counts"]["build_datasets"],
                      "done_datasets": len(done), "done_new_contexts": sum(len(r.get("new_view_ids", [])) for r in done),
                      "epoch0_contexts_replaced": sum(r["epoch0_contexts"] for r in done),
                      "epoch0_contexts_kept": sum(r["epoch0_contexts"] for r in rows if r["status"] != "done"),
                      "done_by_tier": dict(collections.Counter(r["tier"] for r in done)),
                      "failed_datasets": sum(r["status"] == "failed" for r in rows),
                      "not_built": dict(collections.Counter(r["reason"] for r in rows if r["status"] == "not_built")),
                      "by": dict(sorted(cnt.items())),
                      "text_gate_control_failed_as_required": sum(1 for r in done if (r.get("text_gate") or {}).get("control_fails_as_required")),
                      "text_gate_control_no_power": sorted(r["dataset_hash"] for r in done
                                                           if not (r.get("text_gate") or {}).get("control_fails_as_required"))},
           "raw_row_losses": sorted(({"dataset_hash": r["dataset_hash"], "raw_rows": r["raw_rows"],
                                      "unmapped_number_keyed_category": r["rows_lost_unmapped_number_keyed_category"],
                                      "no_target": r["rows_lost_no_target"]}
                                     for r in rows if r.get("rows_lost_unmapped_number_keyed_category")),
                                    key=lambda x: -x["unmapped_number_keyed_category"]),
           "datasets": rows, "shard_receipts": shard_receipts,
           "cpu": {"dataset_cpu_s": round(sum(r.get("cpu_s") or 0 for r in rows), 1),
                   "shard_process_cpu_s": round(sum(s["cpu_s"] for s in shard_receipts), 1)},
           "tests": {"log": ref(a.tests_log), "tail": Path(a.tests_log).read_text().strip().splitlines()[-3:]},
           "mutants": {"path": a.mutants, "sha256": sha_file(a.mutants), "totals": mut.get("totals")},
           "pilot": json.loads(Path(a.pilot).read_bytes()) if a.pilot else None,
           "code_sha256": codes, "policy": {"s3_calls": 0, "network": False, "gpu": False, "git": False, "published": False,
                                            "held_sources_read": False, "credential_files_read": False}}
    stem = "rowsel-raw-nonnumeric-summary" if version == SPLIT_V1 else "rowsel-raw-v2-nonnumeric-summary"
    rp = Path(a.receipt) if a.receipt else RECEIPTS / "build" / f"{stem}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    data = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    with open(rp, "xb") as f:
        f.write(data)
    complete = not any(r.get("reason") == "no_dataset_record" for r in rows)
    counts = {k: v for k, v in doc["counts"].items() if k != "by"}
    marker = write_complete(out, rp, counts, version) if should_mark_complete(complete, version) else None
    print(json.dumps({"receipt": str(rp), "sha256": sha(data), "counts": counts, "complete_marker": str(marker) if marker else None}))
    return 0


# --------------------------------------------------------------------------------------------------- split v2-groups (RAW-GROUP)
#
# Owner ruling RAW-GROUP (owner-decisions-20261006T103543Z.json; relay DELEGATED-RELAY-raw-group-20261006T103543Z.txt):
#   (a) the raw-route datasets whose registry group column carries EXACTLY from T1 onto the raw universe (fill receipt
#       raw-route-group-annotated-20261006T105200Z.json, carry verdict 'yes') are rebuilt on rowsel-split-v2-groups
#       (dltrain/rowsel_split_v2_groups.py: union of exact-duplicate feature rows and equal group values; same v2 ranking
#       and greedy placement) into new create-only trees, under new cut_ids 'trainonly-split-v2g:' + split sha256;
#       cut.replaces_cut names the as-built plain-v2 contexts, which are superseded locally (nothing deleted);
#   (b) where the group does not carry exactly (verdict 'near'): plain split v2 is kept and the as-built contexts are
#       flagged group_annotation_dropped BY REFERENCE in a create-only record (contexts are never rewritten): they stay in
#       training/S1 and are excluded from every headline outer-test number.
# Group value of a universe row = canonical '<f8' bytes of its value in the group column (the T1 column's raw reading,
# spec index); NaN or t1raw.MISSING_CODE = the one missing group (as the fill receipt's carry/straddle scan keyed it).
# Admission BEFORE any split or union-find: rowsel_split_v2_groups.target_tied_group for EVERY candidate column first;
# then (ambiguous registry columns) every candidate must give the identical component structure; lowest index taken.
# Gates after the split: no duplicate or annotated group straddles folds, the target never moves the split, group- and
# duplicate-preserving outer-test changes never move it, empty folds only trailing and only below 5 components (the
# lane-wide convention), equal to the plan's. LINEAR-DETERMINED applies as on every non-v1 route.

RAW_GROUP_SOURCE = RECEIPTS / "fill" / "raw-route-group-annotated-20261006T105200Z.json"
RAW_GROUP_SOURCE_SHA256 = "7ac257708f15e9139c84276d296a9b9226662343c9ab7a57b72c1d1864fbba95"
RAW_GROUP_DECISION = Path("/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261006T103543Z.json")
RAW_GROUP_DECISION_SHA256 = "f2e13f9263b386cac223f4dbe1c9206be49cfef79588ceccdedfa661218e3778"
RAW_GROUP_RELAY = RUN / "receipts" / "owner-relays" / "DELEGATED-RELAY-raw-group-20261006T103543Z.txt"
V2G_PLAN_SCHEMA = "jope.dl-v2-rowsel-raw-v2gsplit-plan/v1"
V2G_SUMMARY_SCHEMA = "jope.dl-v2-rowsel-raw-v2gsplit-summary/v1"
V2G_DROPPED_SCHEMA = "jope.dl-v2-rowsel-raw-group-annotation-dropped/v1"
V2G_SUPERSEDED_SCHEMA = "jope.dl-v2-rowsel-raw-v2-superseded-by-v2g/v1"
V2G_SPLIT_KEYS = ("partition", "group_column", "components", "annotation_groups", "annotation_groups_straddling_folds",
                  "empty_folds", "effective_folds")
RAW_MISSING_GROUP = b"\x00missing-cell"        # not 8 bytes: never read as a numeric '<f8' group value


def raw_group_values(values, specs, j: int, name: str) -> list:
    """Group value per universe row from T1 column j's raw reading (specs[j]; never the target)."""
    j = int(j)
    if not 0 <= j < len(specs) - 1:
        raise rowsel_raw.RowselRawError("group_column_not_a_feature", f"{j} of {len(specs) - 1} features")
    if specs[j].get("name") != name:
        raise rowsel_raw.RowselRawError("group_column_name_mismatch", f"spec {j} is {specs[j].get('name')!r}, plan says {name!r}")
    col = np.asarray(values, dtype=np.float64)[:, specs[j]["index"]]
    return [RAW_MISSING_GROUP if (np.isnan(x) or x == t1raw.MISSING_CODE) else np.float64(x + 0.0).tobytes() for x in col.tolist()]


def v2g_functions(e) -> tuple:
    """(split_fn, split_report_fn) of a v2-groups entry: its group column {index, name}."""
    gc = e.get("group_column") or {}
    if gc.get("index") is None or not gc.get("name"):
        raise rowsel_raw.RowselRawError("group_column_not_given", "a rowsel-split-v2-groups entry must name its group column")
    j, name = int(gc["index"]), str(gc["name"])
    spec = rowsel_split_v2_groups.group_spec(j, name)

    def split_fn(h, positions, values, specs, folds=rowsel_raw.OUTER_FOLDS):
        return rowsel_split_v2_groups.split(h, [int(p) for p in np.asarray(positions).tolist()], rowsel_raw.feature_bytes(values, specs),
                                            raw_group_values(values, specs, j, name), rowsel_raw.PARTITION, folds)

    def report_fn(prep, folds):
        rep = rowsel_raw.split_report(prep, folds)
        pos = prep.positions.tolist()
        gv = raw_group_values(prep.values, prep.specs, j, name)
        comps = rowsel_split_v2_groups.components(pos, rowsel_raw.feature_bytes(prep.values, prep.specs), gv)
        empty = [f for f, n in enumerate(rep["fold_sizes"]) if n == 0]
        rep.update(schema=SPLIT_V2G, sha256=rowsel_split_v2_groups.split_sha256(prep.dataset_hash, rowsel_raw.PARTITION, spec, folds),
                   domain=rowsel_split_v2_groups.SPLIT_DOMAIN, rule=rowsel_split_v2_groups.RULE, partition=rowsel_raw.PARTITION,
                   group_column=spec, components=len(comps), annotation_groups=len(set(gv)),
                   annotation_groups_straddling_folds=rowsel_split_v2_groups.straddling_groups(folds, pos, gv),
                   empty_folds=empty, effective_folds=len(rep["fold_sizes"]) - len(empty))
        return rep
    return split_fn, report_fn


def v2g_admission(e, prep) -> dict:
    """Dataset-level admission BEFORE any split or union-find (PR #218 finding 8 order): target_tied_group on EVERY
    candidate column first; only when none is tied, the component structures of the candidates (ambiguous registry
    columns) must be identical."""
    gc = e["group_column"]
    cands = [int(c) for c in e.get("group_candidates") or [gc["index"]]]
    names = list(e.get("group_candidate_names") or [gc["name"]])
    if len(names) != len(cands) or int(gc["index"]) != min(cands) or names[cands.index(int(gc["index"]))] != gc["name"]:
        raise rowsel_raw.RowselRawError("group_candidates_inconsistent", json.dumps({"candidates": cands, "names": names, "group_column": gc}))
    y = prep.values[:, -1]
    gvs = {c: raw_group_values(prep.values, prep.specs, c, n) for c, n in zip(cands, names)}
    ties = {str(c): rowsel_split_v2_groups.target_tied_group(gvs[c], y) for c in cands}      # FIRST, for every candidate
    if any(ties.values()):
        raise rowsel_raw.RowselRawError("group_column_tied_to_target", json.dumps({c: t for c, t in ties.items() if t}))
    rep = {"candidates": cands, "names": names, "target_tie": ties, "rule": "target_tied_group first for every candidate; "
           "then identical component structure across candidates; lowest index is the group column"}
    if len(cands) > 1:
        pos = prep.positions.tolist()
        fb = rowsel_raw.feature_bytes(prep.values, prep.specs)
        structures = {str(c): sha(json.dumps(rowsel_split_v2_groups.component_partition(pos, fb, gvs[c])).encode()) for c in cands}
        rep["component_structure_sha256"] = structures
        if len(set(structures.values())) != 1:
            raise rowsel_raw.RowselRawError("ambiguous_group_columns_with_different_components", json.dumps(structures))
    return rep


def v2g_group_value_check(split_fn, prep, j: int, fold: int = 0) -> dict:
    """The fold assignment must not move when one outer-test fold's features (the group column included: remapped to
    fresh values), text and targets change; validity = the component structure (duplicates UNION groups) unchanged."""
    h, pos = prep.dataset_hash, prep.positions
    name = prep.specs[int(j)]["name"]
    base = split_fn(h, pos, prep.values, prep.specs)
    rows = prep.rows_of(base[fold]) if len(base[fold]) else np.zeros(0, dtype=np.int64)
    part0 = rowsel_split_v2_groups.component_partition(pos.tolist(), rowsel_raw.feature_bytes(prep.values, prep.specs),
                                                       raw_group_values(prep.values, prep.specs, j, name))
    fold_of = {p: f for f, v in base.items() for p in v}
    cases = {}
    for kind in rowsel_raw_split_v2.CF_KINDS:
        V2, s2 = rowsel_raw_split_v2.perturb_fold_values(prep.values, prep.specs, rows, kind)
        part = rowsel_split_v2_groups.component_partition(pos.tolist(), rowsel_raw.feature_bytes(V2, s2), raw_group_values(V2, s2, j, name))
        got = split_fn(h, pos, V2, s2)
        moved = sum(1 for f, v in got.items() for p in v if fold_of[p] != f)
        cases[kind] = {"valid": part == part0, "split_identical": moved == 0, "rows_moved": int(moved)}
    return {"fold": int(fold), "cases": cases, "valid": all(c["valid"] for c in cases.values()),
            "invariant": all(c["valid"] and c["split_identical"] for c in cases.values())}


def v2g_empty_folds_allowed(rep: dict) -> bool:
    """Lane-wide convention: with c < 5 components the 5-fold greedy fill leaves exactly folds c..4 empty (c >= 2)."""
    c, sizes = rep.get("components"), rep["fold_sizes"]
    return c is not None and 2 <= c < len(sizes) and [f for f, n in enumerate(sizes) if n == 0] == list(range(c, len(sizes)))


def v2g_gates(srep: dict, e) -> None:
    if srep["annotation_groups_straddling_folds"]:
        raise rowsel_raw.RowselRawError("annotation_groups_straddle_folds", str(srep["annotation_groups_straddling_folds"]))
    gvc = srep["group_value_check_fold0"]
    if not (gvc["valid"] and gvc["invariant"]):
        raise rowsel_raw.RowselRawError("split_moves_under_group_preserving_changes", json.dumps(gvc["cases"]))
    if srep["empty_folds"] and not v2g_empty_folds_allowed(srep):
        raise rowsel_raw.RowselRawError("empty_outer_fold", json.dumps(srep["fold_sizes"]))
    if e.get("empty_folds") is not None and srep["empty_folds"] != list(e["empty_folds"]):
        raise rowsel_raw.RowselRawError("empty_folds_differ_from_plan", f"{srep['empty_folds']} != {e['empty_folds']}")


def v2g_cut(srep: dict, e) -> dict:
    rv = e["replaces_v2"]
    return {"cut_id": rowsel_split_v2_groups.cut_id(srep["sha256"]), "schema": SPLIT_V2G, "comparable_with_replaced": False,
            "group_column": srep["group_column"], "folds": len(srep["fold_sizes"]), "empty_folds": srep["empty_folds"],
            "effective_folds": srep["effective_folds"],
            "replaces_cut": [{"kind": "rowsel-raw-split-v2 (as built, superseded; no cut_id was recorded)", "tree": rv["tree"],
                              "split_sha256": rv["split_sha256"], "view_ids": list(rv["view_ids"]), "dataset_record": rv["dataset_record"]}],
            "rule": "cut_id = 'trainonly-split-v2g:' + rowsel_split_v2_groups.split_sha256(h, 'raw', group_column, the 5-fold "
                    "assignment incl. empty folds); never compared with the replaced as-built v2 split"}


def plan_v2g(a) -> int:
    """The v2gsplit plan of one raw route: the fill receipt's 'yes' datasets of the route's v2 tree, each the base v2 plan
    entry with split v2-groups, its group column (all reproducing candidates), the as-built v2 contexts it supersedes
    and the expected trailing empty folds; plus the create-only group-annotation-dropped record of the 'near' datasets."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base_out = default_out(a.route, SPLIT_V2)
    base_pl, base_sha = load_plan(base_out)
    if plan_split_version(base_pl, SPLIT_V2) != SPLIT_V2:
        raise SystemExit("base plan is not split v2")
    base = {e["dataset_hash"]: e for e in base_pl["datasets"]}
    srcb = RAW_GROUP_SOURCE.read_bytes()
    if sha(srcb) != RAW_GROUP_SOURCE_SHA256:
        raise SystemExit("raw-route group source receipt drift")
    entries, dropped, skipped = [], [], []
    for x in json.loads(srcb)["datasets"]:
        if x.get("tree") != base_out.name:
            continue
        h = x["dataset_hash"]
        verdict = ((x.get("carry") or {}).get("verdict") or "").split(":")[0]
        rec_ref = x.get("dataset_record") or {}
        if x.get("status") != "built" or rec_ref.get("status") != "done":
            skipped.append({"dataset_hash": h, "reason": x.get("reason") or "not built as v2"})
            continue
        if sha_file(rec_ref["path"]) != rec_ref["sha256"]:
            raise SystemExit(f"{h}: as-built v2 record drift")
        replaces = {"tree": str(base_out), "dataset_record": {"path": rec_ref["path"], "sha256": rec_ref["sha256"]},
                    "view_ids": list(x["view_ids"]), "split_sha256": x["recorded_split"]["sha256"]}
        if verdict == "near":
            dropped.append({"dataset_hash": h, "flag": "group_annotation_dropped", "as_built": replaces,
                            "carry_verdict": x["carry"]["verdict"], "registry_group_column": {k: v for k, v in x["registry_group_column"].items() if k != "evidence"},
                            "training": "included (training/S1)", "headline_outer_test": "excluded from every headline outer-test number"})
            continue
        if verdict != "yes":
            skipped.append({"dataset_hash": h, "reason": f"carry verdict {verdict!r}"})
            continue
        rg = x["registry_group_column"]
        cands = [int(c) for c in rg["candidates_reproducing"]]
        names = list(rg["t1_name"])
        c = min(cands)
        groups = {s["t1_index"]: s["groups"] for s in x["as_built_group_straddling"]}[c]
        e = dict(base[h])
        e.update(split_version=SPLIT_V2G, group_column={"index": c, "name": names[cands.index(c)]}, group_candidates=cands,
                 group_candidate_names=names, empty_folds=list(range(groups, rowsel_raw.OUTER_FOLDS)) if groups < rowsel_raw.OUTER_FOLDS else [],
                 replaces_v2=replaces, group_evidence={"source": {"path": str(RAW_GROUP_SOURCE), "sha256": RAW_GROUP_SOURCE_SHA256},
                                                       "carry": x["carry"], "ambiguous": bool(rg.get("ambiguous")),
                                                       "as_built_group_straddling": x["as_built_group_straddling"]})
        e["effective_folds"] = rowsel_raw.OUTER_FOLDS - len(e["empty_folds"])
        entries.append(e)
    loads = [0.0] * a.of
    for e in sorted(entries, key=lambda e: (-float(e.get("est_cpu_s") or 1.0), e["dataset_hash"])):
        i = min(range(a.of), key=lambda k: (loads[k], k))
        e["shard"] = i
        loads[i] += float(e.get("est_cpu_s") or 1.0)
    entries.sort(key=lambda e: e["dataset_hash"])
    decision = {"owner_decisions": {"path": str(RAW_GROUP_DECISION), "sha256": RAW_GROUP_DECISION_SHA256}, "relay": str(RAW_GROUP_RELAY),
                "keys": ["RAW-GROUP", "RAW-GROUP-PR", "S1-RAW-GROUP"]}
    doc = {"schema": V2G_PLAN_SCHEMA, "route": a.route, "of": a.of, "base_tree": str(base_out), "base_plan_sha256": base_sha,
           "datasets": entries, "counts": {"datasets": len(entries), "expected_new_contexts": sum(e["effective_folds"] for e in entries),
                                           "replaced_v2_contexts": sum(len(e["replaces_v2"]["view_ids"]) for e in entries),
                                           "group_annotation_dropped_datasets": len(dropped), "skipped": len(skipped)},
           "skipped": skipped, "shard_est_cpu_s": [round(x, 1) for x in loads], "decision": decision,
           "inputs": {"raw_group_source": {"path": str(RAW_GROUP_SOURCE), "sha256": RAW_GROUP_SOURCE_SHA256}} | base_pl.get("inputs", {}),
           "precap_rows": rowsel_raw.PRECAP_ROWS, "outer_folds": rowsel_raw.OUTER_FOLDS, "budget": rowsel_raw.BUDGET}
    stamp_split_version(doc, SPLIT_V2G)
    drop = {"schema": V2G_DROPPED_SCHEMA, "route": a.route, "datasets": dropped, "count": len(dropped),
            "contexts": sum(len(d["as_built"]["view_ids"]) for d in dropped), "decision": decision,
            "rule": "RAW-GROUP (b): plain split v2 kept; flag group_annotation_dropped by reference (contexts are never rewritten); "
                    "included in training/S1; excluded from every headline outer-test number"}
    for name, body in (("plan.json", doc), ("group-annotation-dropped.json", drop)):
        data = (json.dumps(body, indent=1, sort_keys=True) + "\n").encode()
        p = out / name
        if p.exists() and p.read_bytes() != data:
            raise SystemExit(f"{p} exists with different content; refusing to replace a pinned record")
        if not p.exists():
            atomic(p, data)
    print(json.dumps({"plan": str(out / "plan.json"), "counts": doc["counts"], "shard_est_cpu_s": doc["shard_est_cpu_s"]}))
    return 0


def verify_v2g_context(d: Path, e, r) -> list:
    """Problems of one built v2-groups context directory against its dataset record and plan entry."""
    p = []
    cj = json.loads((d / "context.json").read_bytes()) if (d / ".done").exists() else None
    if cj is None:
        return ["no .done"]
    for name, meta in cj.get("files", {}).items():
        if not (d / name).is_file() or sha_file(d / name) != meta["sha256"]:
            p.append(f"file:{name}")
    cut, sref = cj.get("cut") or {}, (cj.get("rowsel") or {}).get("split") or {}
    if (cj.get("split_version") != SPLIT_V2G or sref.get("schema") != SPLIT_V2G or cut.get("cut_id") != rowsel_split_v2_groups.cut_id(sref.get("sha256", ""))
            or cut.get("comparable_with_replaced") is not False or cut.get("group_column") != e["group_column"]
            or [x.get("view_ids") for x in cut.get("replaces_cut") or []] != [e["replaces_v2"]["view_ids"]]):
        p.append("cut")
    if sref.get("annotation_groups_straddling_folds") != 0 or not sref.get("group_value_invariant") or not sref.get("target_invariant"):
        p.append("split_gates")
    if sref.get("empty_folds") != e["empty_folds"] or cj["fold"] in e["empty_folds"]:
        p.append("empty_folds")
    if (cj.get("target_linear") or {}).get("verdict") == target_linear.REFUSE or "target_linear" not in cj:
        p.append("target_linear")
    if cj["view_id"] in e["replaces_v2"]["view_ids"]:
        p.append("view_id_reused")
    return p


def summarize_v2g(a) -> int:
    """Verify a v2gsplit tree; write its summary receipt, the create-only superseded record (as-built v2 contexts ->
    their v2g replacements, for the publisher's denylist/superseded flow) and, when complete, .complete."""
    out = Path(a.out)
    pl, plan_sha = load_plan(out)
    plan_split_version(pl, SPLIT_V2G)
    codes = code_shas()
    rows, problems, ctx_dirs, superseded, review = [], [], set(), [], []
    for e in pl["datasets"]:
        h = e["dataset_hash"]
        sd = out / f"shard-{e['shard']}"
        p = sd / "_datasets" / f"{h}.json"
        row = {"dataset_hash": h, "class": e["class"], "group_column": e["group_column"], "effective_folds": e["effective_folds"],
               "empty_folds": e["empty_folds"], "replaced_v2_view_ids": e["replaces_v2"]["view_ids"]}
        if not p.exists():
            rows.append(row | {"status": "missing_record"})
            continue
        r = json.loads(p.read_bytes())
        row.update(status=r["status"], reason=r.get("reason"), detail=(r.get("detail") or "")[:400], record={"path": str(p), "sha256": sha_file(p)})
        if r["status"] == "done":
            if not code_known(r.get("code_sha256"), codes):
                problems.append({"dataset": h, "problem": "code_not_current"})
            if (r.get("split") or {}).get("schema") != SPLIT_V2G or r.get("split_version") != SPLIT_V2G:
                problems.append({"dataset": h, "problem": "split_version"})
            if len(r["contexts"]) != e["effective_folds"]:
                problems.append({"dataset": h, "problem": "context_count", "contexts": len(r["contexts"])})
            ids = []
            for c in r["contexts"]:
                d = sd / c["view_id"]
                ctx_dirs.add(str(d))
                pr = verify_v2g_context(d, e, r)
                if pr:
                    problems.append({"dir": str(d), "problems": pr})
                ids.append(c["view_id"])
                if (c.get("target_linear") or {}).get("verdict") == target_linear.REVIEW:
                    review.append({"view_id": c["view_id"], "ols_rel_rmse": c["target_linear"]["ols_rel_rmse"]})
            row.update(new_view_ids=ids, cut_id=rowsel_split_v2_groups.cut_id(r["split"]["sha256"]))
            superseded.append({"dataset_hash": h, "superseded_view_ids": e["replaces_v2"]["view_ids"], "superseded_by": ids,
                               "cut_id": row["cut_id"], "as_built_record": e["replaces_v2"]["dataset_record"]})
        rows.append(row)
    orphans = [str(d) for sd in out.glob("shard-*") if sd.is_dir() for d in sd.iterdir()
               if d.is_dir() and d.name != "_datasets" and str(d) not in ctx_dirs]
    done = [x for x in rows if x["status"] == "done"]
    counts = {"expected_datasets": len(rows), "done_datasets": len(done), "failed_datasets": sum(x["status"] == "failed" for x in rows),
              "missing_records": sum(x["status"] == "missing_record" for x in rows),
              "failed_by_reason": dict(collections.Counter(x.get("reason") for x in rows if x["status"] == "failed")),
              "expected_new_contexts": sum(x["effective_folds"] for x in rows), "done_new_contexts": sum(len(x["new_view_ids"]) for x in done),
              "superseded_v2_contexts": sum(len(s["superseded_view_ids"]) for s in superseded),
              "target_linear_near_review_contexts": len(review), "verification_problems": len(problems), "orphan_context_dirs": len(orphans)}
    shard_recs = [ref(q) for q in sorted(out.glob(f"shard-*-of-{pl['of']}.receipt.json"))]
    mut = json.loads(Path(a.mutants).read_bytes())
    dropped_p = out / "group-annotation-dropped.json"
    doc = {"schema": V2G_SUMMARY_SCHEMA, "utc_us": int(time.time() * 1e6), "out": str(out), "route": pl["route"], "split_version": SPLIT_V2G,
           "plan": {"path": str(out / "plan.json"), "sha256": plan_sha, "counts": pl["counts"]}, "counts": counts, "datasets": rows,
           "problems": problems[:100], "orphans": orphans[:50], "shard_receipts": shard_recs, "target_linear_near_review": review,
           "group_annotation_dropped": ref(dropped_p) if dropped_p.exists() else None, "decision": pl["decision"],
           "tests": {"log": ref(a.tests_log), "tail": Path(a.tests_log).read_text().strip().splitlines()[-3:]},
           "mutants": {"path": a.mutants, "sha256": sha_file(a.mutants), "totals": mut.get("totals")}, "code_sha256": codes,
           "policy": {"s3_calls": 0, "network": False, "gpu": False, "git": False, "published": False}}
    rp = Path(a.receipt) if a.receipt else RECEIPTS / "build" / f"rowsel-raw-v2gsplit-{pl['route']}-summary-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    data = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    with open(rp, "xb") as fh:
        fh.write(data)
    complete = counts["missing_records"] == 0 and not problems and not orphans and len(shard_recs) == pl["of"]
    marker = None
    if complete:
        sup = (json.dumps({"schema": V2G_SUPERSEDED_SCHEMA, "tree": str(out), "summary": {"path": str(rp), "sha256": sha(data)},
                           "datasets": superseded, "count": len(superseded), "contexts": counts["superseded_v2_contexts"],
                           "rule": "the as-built plain-v2 contexts listed are superseded locally by their v2g replacements; nothing is deleted"},
                          indent=1, sort_keys=True) + "\n").encode()
        sp = out / "superseded-v2.json"
        if not sp.exists():
            atomic(sp, sup)
        marker = write_complete(out, rp, counts, SPLIT_V2G) if not (out / ".complete").exists() else None
    print(json.dumps({"receipt": str(rp), "sha256": sha(data), "complete": complete, "marker": str(marker) if marker else None, "counts": counts}))
    return 0 if complete else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "build", "summarize"))
    ap.add_argument("--route", choices=("l7a", "nonnumeric"), default="l7a")
    ap.add_argument("--split-version", choices=tuple(SPLIT_VERSIONS), default=SPLIT_V2,
                    help=f"default {SPLIT_V2} (target-free); {SPLIT_V1} only when passed explicitly (v1 replay/verification)")
    ap.add_argument("--out")
    ap.add_argument("--shard", type=int)
    ap.add_argument("--of", type=int, default=3)
    ap.add_argument("--datasets", help="comma-separated dataset hashes (pilot / smoke runs only)")
    ap.add_argument("--tests-log")
    ap.add_argument("--mutants")
    ap.add_argument("--pilot")
    ap.add_argument("--receipt")
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = str(default_out(a.route, a.split_version))
    if a.cmd == "plan":
        if a.split_version == SPLIT_V2G:
            return plan_v2g(a)
        return plan_nonnumeric(a) if a.route == "nonnumeric" else plan(a)
    if a.cmd == "summarize":
        if a.split_version == SPLIT_V2G:
            return summarize_v2g(a)
        return summarize_nonnumeric(a) if a.route == "nonnumeric" else summarize(a)
    if a.shard is None:
        ap.error("--shard is required for build")
    return build(a)


if __name__ == "__main__":
    sys.exit(main())

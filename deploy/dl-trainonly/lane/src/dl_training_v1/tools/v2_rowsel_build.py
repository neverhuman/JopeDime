"""ROWSEL-TRAINONLY T1 rebuild of the kmeans contexts: local, CPU only, unpublished.

Two scopes (--scope), both labelled ["kmeans"] in row-selection-leak-v2:
  incoming-kmeans (owner line ROWSEL-TRAINONLY): every incoming kmeans context, 375 datasets and 1,749 contexts:
    810 contexts without an acquisition cap        -> row_selection_leak ["clean"]
    939 acquisition-capped contexts (acq_selection reduced=true) -> ["acquisition_cap"]; the cap is not undone.
    Under ROWSEL-SPLIT, a capped context whose raw source matched is superseded by another lane's raw build. Its T1
    build here is the fallback (rowsel.raw_route_superseded_if_matched = true).
  frozen-kmeans-research (owner line RIGHTS-RESEARCH): the frozen kmeans contexts whose dataset is in the pinned
    rights-research list (Rdatasets, UCI, CDC). That is 42 datasets and 209 contexts. Each context.json carries
    rights "research-only" and rights_decision "<owner-decisions file>#sha256=<sha>". Every dataset's host in the
    rights-gate receipt must be an admitted host; the city portals stay held.
    The parent folds are the registry split objects fetched to RUN/inputs/rowsel-frozen-splits. Each is
    sha256-checked at use and cross-checked against the fetch receipt (key, VersionId, bytes).
Coreset, frozen capped and incoming clean contexts are outside both scopes and are never read. Every context is built
on the T1 universe (rowsel.route "t1"). The acquisition-cap label must agree with the cap markers in the T1 meta.json.

Per dataset (lineage of every epoch-0 context through scale_check.resolve_context; sha256 on every hop):
  U       the T1 train lines then the test lines (manifest.source.{train,test}: INTAKE verified-objects for incoming,
          FROZEN/cache/blobs for frozen); position is the concatenation index and the row id is
          rowsel.positional_row_id. The line split must equal materialize.py's.
  folds   the registry split object manifest.source.split (jope.virtual-splits/v1, grouped-stratified-kfold/v1). Its
          repeat-0 test_indices must partition U. Remapping them onto the v1 sample must reproduce each epoch-0 context's
          membership fold, so the fold numbering is the one epoch-0 used.
  port    a fit on every row with the v1 seed is compared with manifest.selected_source_indices_sha256. For frozen
          datasets, FROZEN/work/<h>/{input.csv,sampler/result.json} (the v1 sampler I/O) is compared too. This is
          recorded and does not gate.
Per context (fold f):
  view    rowsel.select_view(U, folds, f, 2000). The train rows come from the sampler fitted on U minus F_f only. The
          test rows come from draw_test_rows over the ids of F_f. The quotas come from the parent-fold sizes.
  gate 1  rowsel.counterfactual_report must be train-side invariant and test-draw invariant.
  A       the T1 lines at the sorted positions of train and test (np.loadtxt, as resolve_context reads v1 data
          objects), checked bit for bit against U. R and T are the indices of the train and test positions in A.
  files   encoder.jtt1.gz, teacher.jtt1.gz and a0.f32, built exactly as tools/v2_build_local.build_one builds them.
  gate 2  the downstream counterfactual. The values of A's outer-test rows are replaced by other T1 rows and the build
          is repeated. The encoder and teacher train rows, the scaler, the inner labels, A0 and the train-side t1prep
          report must not change.
  context.json  schema jope.dl-v2-rowsel-context/v1 (see compose_context). .done is written last.
A context that fails a gate or raises writes nothing. It is listed in the shard receipt, and its epoch-0 version stays.

Outputs:
  OUT/shard-<i>/<view_id>/         the context files (nothing else sits in a shard directory)
  OUT/_datasets/<dataset_hash>.json  the per-dataset lineage record
  OUT/shard-<i>-of-<n>.receipt.json  expected, done, failed and counterfactual failures, labels and CPU time
  --overlay   appends to RUN/epochs/epoch-001-rowsel-overlay.jsonl (append-only; the header pins epoch-000.json)
  --summarize writes the build summary receipt under RUN/receipts/trainonly-v2/build/

No S3, no network, no GPU. Nothing is published.

Usage (lane slice, through py.sh; --scope defaults to incoming-kmeans):
  v2_rowsel_build.py --out DIR --shard I --of N [--scope S] [--datasets h1,h2]
  v2_rowsel_build.py --out DIR --of N --overlay [--overlay-path P]
  v2_rowsel_build.py --out DIR --of N --summarize [--scope S] --tests-log LOG --mutants JSON [--pilot-out DIR] [--receipt P]
"""
import argparse
import collections
import glob
import gzip
import hashlib
import io
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

from dltrain import cpuguard, jtt1, metafeatures, rowsel, scale_check, t1prep, trainonly

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
META_DIR = RUN / "inputs" / "t1-meta"
DATASETS_INV = RUN / "tmp" / "trainonly-v1" / "epoch0-datasets.json"
RECEIPTS = RUN / "receipts" / "trainonly-v2"
LEAK = RECEIPTS / "row-selection-leak-v2-20261005T200546Z.jsonl"
LEAK_SHA256 = "2196d4e8a504f9a3a41c5fc910d588353ac54fb6ac538dfccc8658780c1f6905"
ACQ = RUN / "tmp" / "rowsel" / "acq_selection.json"
ACQ_SHA256 = "2757fef11544a5326c9f9375a754628ac8b53a9bfe5d5a7f04cbd57dacd0f2d2"
PLAN = RECEIPTS / "rowsel-rebuild-plan.md"
OWNER_DECISIONS = Path("/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261005T210659Z.json")
EPOCH0 = scale_check.EPOCH_PATH
EPOCH0_SHA256 = scale_check.EPOCH_SHA256
OVERLAY = RUN / "epochs" / "epoch-001-rowsel-overlay.jsonl"
INTAKE_OBJECTS = scale_check.INTAKE_OBJECTS
L7B = RUN / "build-v2" / "L7b"
L7B_OF = 3

SCHEMA = "jope.dl-v2-rowsel-context/v1"
IDENTITY_SCHEMA = "jope.dl-v2-rowsel-context-identity/v1"
SHARD_SCHEMA = "jope.dl-v2-rowsel-shard/v1"
DATASET_SCHEMA = "jope.dl-v2-rowsel-dataset/v1"
OVERLAY_SCHEMA = "jope.dl-epoch-overlay/v1"
SUMMARY_SCHEMA = "jope.dl-v2-rowsel-build-summary/v1"
OWNER_LINE = "ROWSEL-TRAINONLY"
VIEW_TAG = "rowsel1"
ROUTE = "t1"
SPLIT_SCHEMA = "jope.virtual-splits/v1"
SPLIT_METHOD = "grouped-stratified-kfold/v1"
DOWNSTREAM_DOMAIN = "trainonly-rowsel-downstream-cf-v1"
EXPECTED = {"datasets": 375, "contexts": 1749, "clean": 810, "acquisition_cap": 939}
FROZEN_ROOT = scale_check.FROZEN_ROOT
FROZEN_KEY_PREFIX = scale_check.FROZEN_KEY_PREFIX
FROZEN_BLOBS = FROZEN_ROOT / "cache" / "blobs"
FROZEN_SPLITS = RUN / "inputs" / "rowsel-frozen-splits"
FROZEN_SPLIT_FETCH = RECEIPTS / "rowsel-frozen-split-fetch-20261005T211334Z.json"
FROZEN_SPLIT_FETCH_SHA256 = "274d73c06a677c27e6b4691cdcc84ee5efb1f7691f0d4e48a08292f0e2a1a557"
RIGHTS_DATASETS = RUN / "tmp" / "rights-research-datasets.json"
RIGHTS_DATASETS_SHA256 = "cc33da25b0b8dfe51f0a9edc7c0966fa588fc16f3d3c52fde0c5577692474ed7"
RIGHTS_DECISION = Path("/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261005T223750Z.json")
RIGHTS_DECISION_SHA256 = "75217a2a50010bde04f27c4977ee9c3d462986d4ea0d9b6f340043895e983924"
RIGHTS_GATE = RECEIPTS / "rights-gate-no-license-20261005T180254Z.jsonl"
RIGHTS_GATE_SHA256 = "23fc60527a6ef6616d990042393359057ab32472a2d1dd929d81ed33ffe34519"
RIGHTS_HOSTS = frozenset({"vincentarelbundock.github.io", "archive.ics.uci.edu", "data.cdc.gov"})
SCOPES = {
    "incoming-kmeans": {"cohort": "incoming", "owner_line": OWNER_LINE, "expected": EXPECTED, "rights": None},
    "frozen-kmeans-research": {"cohort": "frozen", "owner_line": "ROWSEL-TRAINONLY+RIGHTS-RESEARCH",
                               "expected": {"datasets": 42, "contexts": 209, "clean": 209, "acquisition_cap": 0},
                               "rights": "research-only"},
}
CODE_FILES = ("tools/v2_rowsel_build.py", "tools/v2_build_local.py", "dltrain/rowsel.py", "dltrain/t1prep.py",
              "dltrain/trainonly.py", "dltrain/metafeatures.py", "dltrain/scale_check.py", "dltrain/jtt1.py",
              "dltrain/cpuguard.py")
QUOTA_RULE = ("largest remainder of budget*|F_f|/|U| over the repeat-0 parent folds (ties to the lower fold); "
              "n_train = budget - n_test; |U| <= budget keeps every row (rowsel.fold_quotas)")


class RebuildError(ValueError):
    """A dataset or context cannot be rebuilt; `reason` is a stable token for the receipts."""

    def __init__(self, reason, detail=""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail = reason, detail


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_ref(p: Path) -> dict:
    p = Path(p)
    return {"path": str(p), "sha256": sha_file(p), "bytes": p.stat().st_size}


def atomic(path: Path, data: bytes):
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def jdump(o) -> bytes:
    return (json.dumps(o, sort_keys=True) + "\n").encode()


def code_shas(root: Path | None = None) -> dict:
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    return {f: sha((root / f).read_bytes()) for f in CODE_FILES}


def read_verified(path: Path, digest: str, size=None) -> bytes:
    body = Path(path).read_bytes()
    if sha(body) != digest or (size is not None and len(body) != int(size)):
        raise RebuildError("lineage_sha_mismatch", f"{path}")
    return body


def cpu_s() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


# ------------------------------------------------------------------------------------------------- scope

def select_scope(rows, leak: dict, acq: dict, cohort: str = "incoming", datasets=None) -> list:
    """(epoch-0 row, label, acquisition record) for the admitted class only: contexts of `cohort` whose
    row-selection-leak-v2 label is exactly ["kmeans"] and, when `datasets` is given, whose dataset is in it. The label
    after the rebuild is "acquisition_cap" for datasets whose T1 rows were capped at acquisition (acq_selection
    reduced=true), otherwise "clean"."""
    out = []
    for r in rows:
        lab = leak.get(r["view_id"])
        if lab is None:
            raise RebuildError("scope_unlabelled", r["view_id"])
        if lab["cohort"] != cohort or lab["row_selection_leak"] != ["kmeans"]:
            continue
        if datasets is not None and lab["dataset_hash"] not in datasets:
            continue
        if r.get("cohort") != cohort or (lab["dataset_hash"], int(lab["fold"])) != (r["dataset_hash"], int(r["fold"])):
            raise RebuildError("scope_label_mismatch", r["view_id"])
        a = acq.get(r["dataset_hash"])
        if a is None or a.get("cohort") != cohort or a.get("class") != "kmeans":
            raise RebuildError("scope_acquisition_record", r["dataset_hash"])
        capped = bool(a["reduced"])
        if capped != bool(a.get("selection_markers")):
            raise RebuildError("scope_acquisition_ambiguous", r["dataset_hash"])
        out.append((r, "acquisition_cap" if capped else "clean", a))
    return sorted(out, key=lambda t: (t[0]["dataset_hash"], int(t[0]["fold"])))


def scope_counts(scope) -> dict:
    labels = collections.Counter(lab for _, lab, _ in scope)
    return {"datasets": len({r["dataset_hash"] for r, _, _ in scope}), "contexts": len(scope),
            "clean": labels["clean"], "acquisition_cap": labels["acquisition_cap"]}


def check_rights_hosts(datasets, gate_rows) -> dict:
    """Every dataset of a research-only scope must be a rights-gate dataset on an admitted host (RIGHTS-RESEARCH lifts
    the hold for Rdatasets, UCI and CDC only); anything else, the city portals included, is refused."""
    host = {r["dataset_hash"]: r.get("host") for r in gate_rows}
    bad = sorted(h for h in datasets if host.get(h) not in RIGHTS_HOSTS)
    if bad:
        raise RebuildError("rights_host_not_admitted", ",".join(f"{h}:{host.get(h)}" for h in bad[:10]))
    return dict(collections.Counter(host[h] for h in datasets))


def rights_decision_ref(path=RIGHTS_DECISION, digest=RIGHTS_DECISION_SHA256) -> str:
    body = Path(path).read_bytes()
    if sha(body) != digest or "RIGHTS-RESEARCH" not in json.loads(body).get("decisions", {}):
        raise RebuildError("rights_decision_drift", str(path))
    return f"{path}#sha256={digest}"


def load_scope(rows, leak_path=LEAK, acq_path=ACQ, expected="scope", scope="incoming-kmeans"):
    cfg = SCOPES[scope]
    if expected == "scope":
        expected = cfg["expected"]
    lb, ab = Path(leak_path).read_bytes(), Path(acq_path).read_bytes()
    if expected is not None and (sha(lb) != LEAK_SHA256 or sha(ab) != ACQ_SHA256):
        raise RebuildError("scope_input_drift", f"leak {sha(lb)} acq {sha(ab)}")
    leak = {}
    for line in lb.decode().splitlines():
        if line.strip():
            r = json.loads(line)
            leak[r["view_id"]] = r
    datasets, extra_refs, extra = None, {}, {}
    if cfg["rights"] is not None:
        db, gb = RIGHTS_DATASETS.read_bytes(), RIGHTS_GATE.read_bytes()
        if sha(db) != RIGHTS_DATASETS_SHA256 or sha(gb) != RIGHTS_GATE_SHA256:
            raise RebuildError("scope_input_drift", f"rights list {sha(db)} gate {sha(gb)}")
        datasets = set(json.loads(db))
        gate = [json.loads(l) for l in gb.decode().splitlines() if l.strip()]
        extra = {"rights": cfg["rights"], "rights_decision": rights_decision_ref()}
        extra_refs = {"rights_datasets": {"path": str(RIGHTS_DATASETS), "sha256": sha(db), "datasets": len(datasets)},
                      "rights_gate": {"path": str(RIGHTS_GATE), "sha256": sha(gb), "rows": len(gate)},
                      "rights_decision": extra["rights_decision"]}
    selected = select_scope(rows, leak, json.loads(ab), cfg["cohort"], datasets)
    counts = scope_counts(selected)
    if expected is not None and counts != expected:
        raise RebuildError("scope_counts", f"{counts} != {expected}")
    if cfg["rights"] is not None:
        extra_refs["rights_hosts"] = check_rights_hosts({r["dataset_hash"] for r, _, _ in selected}, gate)
    refs = {"scope": scope, "row_selection_leak": {"path": str(leak_path), "sha256": sha(lb), "rows": len(leak)},
            "acq_selection": {"path": str(acq_path), "sha256": sha(ab)}, **extra_refs}
    if extra:
        refs["context_extra"] = extra
    return selected, refs, counts


# ------------------------------------------------------------------------------------------------- universe and lineage

def materialize_lines(body: bytes) -> list:
    """research_2k/materialize.py lines(): the line split that defined the registry positions."""
    return [line.removesuffix(b"\r") for line in body.split(b"\n") if line]


def t1_universe(h: str, train_body: bytes, test_body: bytes, columns: int) -> rowsel.Universe:
    """rowsel.universe_from_t1, refused unless its line split equals materialize.py's for both blobs (positions are
    registry indices only when both splits agree)."""
    for name, body in (("train", train_body), ("test", test_body)):
        if rowsel.lines(body) != materialize_lines(body):
            raise RebuildError("line_split_differs", f"{name} blob: rowsel.lines != materialize.lines")
    return rowsel.universe_from_t1(h, train_body, test_body, columns)


def parent_folds_from_split(split: dict, n: int) -> dict:
    folds = sorted((f for f in split["folds"] if int(f["repeat"]) == 0), key=lambda f: int(f["fold"]))
    if [int(f["fold"]) for f in folds] != list(range(len(folds))) or len(folds) != int(split["effective_folds"]):
        raise RebuildError("split_folds", "repeat-0 folds are not 0..effective_folds-1")
    out = {int(f["fold"]): [int(i) for i in f["test_indices"]] for f in folds}
    allidx = sorted(i for v in out.values() for i in v)
    if allidx != list(range(n)):
        raise RebuildError("split_partition", "repeat-0 test_indices do not partition the universe")
    return out


def meta_cap_markers(meta) -> list:
    """Acquisition-time row selection recorded in a T1 meta.json (the rule of tmp/rowsel/acq_selection.py)."""
    a, osz, ds = meta.get("acquisition") or {}, meta.get("original_size") or {}, meta.get("downsampling")
    out = []
    if isinstance(ds, dict) and ds.get("method"):
        out.append("downsampling:" + ds["method"])
    if a.get("row_policy") and a.get("row_policy") != "all":
        out.append("row_policy:" + a["row_policy"])
    if osz.get("downsampled"):
        out.append("original_size.downsampled")
    return out


def check_cap_label(ds: dict, acq_rec: dict):
    """The scope label comes from acq_selection; it must agree with the cap markers of the meta.json actually used."""
    got = ds["lineage"].get("meta_cap_markers")
    if got is not None and sorted(got) != sorted(acq_rec.get("selection_markers") or []):
        raise RebuildError("acquisition_cap_disagrees", f"meta {got} acq_selection {acq_rec.get('selection_markers')}")


def lineage_route(cohort: str, row: dict, msha: str, memb_sha: str):
    """(manifest path, T1 blob directory, membership path) of one cohort, as scale_check.resolve_context walks it."""
    if cohort == "incoming":
        return INTAKE_OBJECTS / msha, INTAKE_OBJECTS, INTAKE_OBJECTS / memb_sha
    if cohort == "frozen":
        pref = row["prepared_ref"]
        sm = json.loads(read_verified(Path(pref["path"]), pref["sha256"], pref.get("bytes")))["source_manifest"]
        if sm.get("sha256") != msha or not str(sm.get("key", "")).startswith(FROZEN_KEY_PREFIX):
            raise RebuildError("frozen_manifest_route", msha)
        return (FROZEN_ROOT / sm["key"][len(FROZEN_KEY_PREFIX):], FROZEN_BLOBS,
                FROZEN_ROOT / "objects" / "memberships" / "sha256" / f"{memb_sha}.json.gz")
    raise RebuildError("unknown_cohort", str(cohort))


def frozen_split_fetch_index(path=FROZEN_SPLIT_FETCH, digest=FROZEN_SPLIT_FETCH_SHA256) -> dict:
    return {o["sha256"]: o for o in json.loads(read_verified(Path(path), digest))["objects"]}


def load_split(cohort: str, h: str, sref: dict, split_dir=FROZEN_SPLITS, fetch_index=None):
    """The registry split object named by manifest.source.split, sha256- and byte-checked at use. Frozen objects come
    from the read-only fetch directory and must match their fetch-receipt entry (dataset, key, VersionId, bytes)."""
    if cohort == "incoming":
        path = INTAKE_OBJECTS / sref["sha256"]
        return read_verified(path, sref["sha256"], sref.get("bytes")), {"path": str(path), "source": "intake verified-objects"}
    fetch_index = frozen_split_fetch_index() if fetch_index is None else fetch_index
    o = fetch_index.get(sref["sha256"])
    if o is None or (o.get("dataset_hash"), o.get("key"), o.get("version_id"), int(o.get("bytes", -1))) != \
            (h, sref.get("key"), sref.get("version_id"), int(sref.get("bytes", -2))):
        raise RebuildError("split_fetch_receipt", f"{h} {sref['sha256']}")
    path = Path(split_dir) / f"{sref['sha256']}.json"
    return read_verified(path, sref["sha256"], sref.get("bytes")), {
        "path": str(path), "source": "read-only exact-VersionId fetch", "fetch_receipt": str(FROZEN_SPLIT_FETCH),
        "fetch_receipt_sha256": FROZEN_SPLIT_FETCH_SHA256}


def combine_csv(train: bytes, test: bytes) -> bytes:
    """research_2k/materialize.py combine_csv: the input.csv handed to the v1 sampler."""
    return train.rstrip(b"\r\n") + b"\n" + test.lstrip(b"\r\n").rstrip(b"\r\n") + b"\n"


def v1_work_check(h: str, bodies: dict, manifest: dict, n: int, root=FROZEN_ROOT) -> dict:
    """Frozen v1 sampler I/O (FROZEN/work/<h>): input.csv is the T1 concatenation and result.json names the manifest's
    selection with the v1 seed. Recorded only."""
    inp, res_p = Path(root) / "work" / h / "input.csv", Path(root) / "work" / h / "sampler" / "result.json"
    if not (inp.is_file() and res_p.is_file()):
        return {"present": False}
    res = json.loads(res_p.read_bytes())
    in_sha = sha_file(inp)
    return {"present": True, "input_csv_sha256": in_sha,
            "input_csv_equals_t1_concatenation": in_sha == sha(combine_csv(bodies["train"], bodies["test"]))
            == res.get("input_csv_sha256"),
            "input_rows_equal_universe": int(res.get("input_rows", -1)) == n,
            "result_seed_is_v1_seed": (res.get("sampler") or {}).get("seed") == rowsel.v1_seed(h),
            "result_selected_sha256_matches_manifest":
                res.get("selected_source_indices_sha256") == manifest["selected_source_indices_sha256"],
            "result_json_sha256": sha_file(res_p)}


def load_dataset(h: str, ctx_rows: list, src, cache, inv: dict, budget: int = rowsel.DEFAULT_BUDGET) -> dict:
    """Every input of one dataset, each sha256-verified, plus the dataset-level lineage checks."""
    t0 = cpu_s()
    refs = []
    for r in ctx_rows:
        try:
            c = scale_check.resolve_context(r, src, cache)
        except scale_check.ScaleCheckError as e:
            raise RebuildError("epoch0_lineage", str(e)[:200]) from None
        refs.append(c["refs"])
    if len({(x["manifest_sha256"], x["membership_sha256"], x["data_sha256"]) for x in refs}) != 1:
        raise RebuildError("epoch0_lineage", "contexts of one dataset name different manifests")
    cohorts = {r["cohort"] for r in ctx_rows}
    if len(cohorts) != 1:
        raise RebuildError("epoch0_lineage", f"mixed cohorts {sorted(cohorts)}")
    cohort = cohorts.pop()
    msha, memb_sha = refs[0]["manifest_sha256"], refs[0]["membership_sha256"]
    manifest_path, blob_dir, memb_path = lineage_route(cohort, ctx_rows[0], msha, memb_sha)
    manifest = json.loads(read_verified(manifest_path, msha))
    if manifest.get("dataset_hash") != h:
        raise RebuildError("manifest_dataset_hash", msha)
    src_ref = manifest["source"]
    bodies = {p: read_verified(blob_dir / src_ref[p]["sha256"], src_ref[p]["sha256"], src_ref[p].get("bytes"))
              for p in ("train", "test")}
    sref = src_ref["split"]
    split_body, split_src = load_split(cohort, h, sref)
    split = json.loads(split_body)
    s_src = split.get("source") or {}
    bad = [k for k, ok in {
        "schema": split.get("schema") == SPLIT_SCHEMA, "method": split.get("method") == SPLIT_METHOD,
        "dataset_hash": s_src.get("dataset_hash") == h,
        "train_sha256": s_src.get("train_sha256") == src_ref["train"]["sha256"],
        "test_sha256": s_src.get("test_sha256") == src_ref["test"]["sha256"],
        "parent_split": ((manifest.get("split_policy") or {}).get("parent_split") or {}).get("sha256") == sref["sha256"],
    }.items() if not ok]
    if bad:
        raise RebuildError("split_identity", ",".join(bad))
    columns = int(manifest["shape"]["columns"])
    U = t1_universe(h, bodies["train"], bodies["test"], columns)
    n_tr, n_te = len(rowsel.lines(bodies["train"])), len(rowsel.lines(bodies["test"]))
    if not (U.n == int(manifest["shape"]["source_rows"]) == int(s_src["train_rows"]) + int(s_src["test_rows"])
            and n_tr == int(s_src["train_rows"])):
        raise RebuildError("universe_rows", f"U {U.n} train {n_tr}+{n_te} manifest {manifest['shape']} split {s_src}")
    folds = parent_folds_from_split(split, U.n)
    rowsel.validate_parent_folds(U, folds)
    # v1 lineage: the membership's rows resolve to U positions, and remapping the parent folds reproduces every
    # epoch-0 membership fold (so registry fold f is epoch-0 fold f).
    mref = manifest["objects"]["membership"]
    if mref["sha256"] != memb_sha:
        raise RebuildError("membership_ref", memb_sha)
    memb = json.loads(gzip.decompress(read_verified(memb_path, memb_sha, mref.get("bytes"))))
    pos_of = {rid: i for i, rid in enumerate(rowsel.t1_v1_source_row_ids(h, bodies["train"], bodies["test"]))}
    try:
        selected = [pos_of[x] for x in memb["ordered_source_row_ids"]]
    except KeyError:
        raise RebuildError("membership_ids_unresolved", memb_sha) from None
    sel_ok = rowsel.selected_source_indices_sha256(selected) == manifest["selected_source_indices_sha256"]
    position = {s: i for i, s in enumerate(selected)}
    mfold = {int(f["fold"]): f for f in memb["folds"] if int(f.get("repeat", 0)) == 0}
    remap = {}
    for r in ctx_rows:
        f = int(r["fold"])
        remap[f] = (f in folds and f in mfold and sorted(position[i] for i in folds[f] if i in position)
                    == [int(i) for i in mfold[f]["test_indices"]])
    if not sel_ok or not all(remap.values()):
        raise RebuildError("membership_remap", f"selected_sha_ok={sel_ok} remap={remap}")
    tp = time.process_time()
    port = rowsel.sampler_select(U, U.positions.tolist(), budget, rowsel.v1_seed(h))
    port_ok = rowsel.selected_source_indices_sha256(port.output_order) == manifest["selected_source_indices_sha256"]
    port_s = time.process_time() - tp
    k = inv["meta_key"][h].rsplit("/", 1)[1]
    mb = (META_DIR / f"{k}.json").read_bytes()
    if sha(mb) != k:
        raise RebuildError("meta_sha_mismatch", k)
    meta = json.loads(mb)
    lineage = {"schema": DATASET_SCHEMA, "dataset_hash": h, "cohort": cohort, "manifest_sha256": msha,
               "membership_sha256": memb_sha, "manifest_path": str(manifest_path), "t1_blob_dir": str(blob_dir),
               "split_source": split_src, "meta_cap_markers": meta_cap_markers(meta),
               "v1_sampler_io": v1_work_check(h, bodies, manifest, U.n) if cohort == "frozen" else None,
               "data_sha256": refs[0]["data_sha256"], "epoch0_contexts": len(ctx_rows),
               "universe_rows": U.n, "features": U.features, "t1_train_lines": n_tr, "t1_test_lines": n_te,
               "line_split_equals_materialize": True, "parent_fold_sizes": {str(f): len(v) for f, v in folds.items()},
               "membership_selected_sha256_matches_manifest": sel_ok,
               "membership_remap_matches_epoch0_folds": {str(f): ok for f, ok in remap.items()},
               "port_reproduces_v1_selection": port_ok, "port_fit_cpu_s": round(port_s, 3),
               "t1_meta_sha256": k}
    return {"h": h, "U": U, "folds": folds, "meta": meta, "meta_sha256": k, "cohort": cohort,
            "t1": {p: {"sha256": src_ref[p]["sha256"], "bytes": len(bodies[p]), "lines": n} for p, n
                   in (("train", n_tr), ("test", n_te))},
            "split": {"kind": "registry", "sha256": sref["sha256"], "version_id": sref.get("version_id"),
                      "key": sref.get("key"), "bytes": len(split_body), "schema": split["schema"],
                      "method": split["method"], "track": split.get("track"),
                      "effective_folds": int(split["effective_folds"]), "repeat": 0, "custody": split_src},
            "lineage": lineage, "load_cpu_s": cpu_s() - t0}


# ------------------------------------------------------------------------------------------------- one context

def a_matrix(U: rowsel.Universe, apos) -> np.ndarray:
    """The T1 lines at `apos`, parsed as resolve_context parses a v1 data object and checked bit for bit against the
    universe values the sampler saw."""
    idx = U.index_of(apos)
    body = b"\n".join(U.row_bytes[i] for i in idx) + b"\n"
    A = np.loadtxt(io.BytesIO(body), delimiter=",", dtype="float64", ndmin=2)
    ref = np.ascontiguousarray(np.column_stack([U.X[idx], U.y[idx]]), dtype=np.float64)
    if A.shape != ref.shape or not bool(np.isfinite(A).all()) or A.tobytes() != ref.tobytes():
        raise RebuildError("a_matrix_mismatch", f"A {A.shape} vs universe {ref.shape}")
    return A


def build_files(A, R, T, meta) -> dict:
    """tools/v2_build_local.build_one without the writes (same calls, same order, same compression)."""
    split = t1prep.prepare(A, meta, R, T)
    fitted, rep = t1prep.refit(A, meta, R)
    teacher = trainonly.jtt1_bytes(fitted[R], fitted[T].reshape(-1, fitted.shape[1]), split["inner_labels"])
    enc_gz = gzip.compress(split["jtt1"], compresslevel=3, mtime=0)
    tea_gz = gzip.compress(teacher, compresslevel=3, mtime=0)
    a0 = metafeatures.from_raw(split["jtt1"])[0]
    return {"split": split, "t1prep": rep, "teacher": teacher, "enc_gz": enc_gz, "tea_gz": tea_gz, "a0": a0}


def assemble(U: rowsel.Universe, view, meta) -> dict:
    """A, R, T and the files of one view (no gate)."""
    train_set = set(int(p) for p in view.train_positions)
    test_set = set(int(p) for p in view.test_positions)
    apos = sorted(train_set | test_set)
    A = a_matrix(U, apos)
    R = [i for i, p in enumerate(apos) if p in train_set]
    T = [i for i, p in enumerate(apos) if p in test_set]
    return {"apos": apos, "A": A, "R": R, "T": T, "art": build_files(A, R, T, meta)}


def train_side_fields(art: dict) -> dict:
    s, rep = art["split"], art["t1prep"]
    return {"encoder_train_rows_sha256": jtt1.train_rows_sha256(s["jtt1"]),
            "teacher_train_rows_sha256": jtt1.train_rows_sha256(art["teacher"]),
            "scaler_sha256": sha(trainonly.scaler_bytes(s)),
            "inner_labels_sha256": sha(np.asarray(s["inner_labels"], np.uint8).tobytes()),
            "a0_sha256": sha(art["a0"]),
            "kept_columns": list(rep["kept_columns"]), "target_copy_dropped": list(rep["target_copy_dropped"]),
            "category_train_maps": {k: v["train"] for k, v in sorted(rep["categorical"].items())}}


def downstream_counterfactual(U: rowsel.Universe, built: dict, meta, h: str) -> dict:
    """Replace the values of A's outer-test rows by other finite T1 rows of U (ranked by a value-free hash, the test
    rows themselves excluded), rebuild the files and list every train-side field that changed."""
    apos, A, R, T = built["apos"], built["A"], built["R"], built["T"]
    test_pos = {apos[i] for i in T}
    finite = np.isfinite(U.X).all(axis=1)
    cands = sorted((int(p) for p, ok in zip(U.positions.tolist(), finite.tolist()) if ok and int(p) not in test_pos),
                   key=lambda p: (hashlib.sha256(f"{h}\0{DOWNSTREAM_DOMAIN}\0{p}".encode()).digest(), p))
    if not cands or not T:
        return {"train_side_invariant": False, "train_side_differs": ["no_replacement_rows"], "test_rows_changed": 0}
    repl = [cands[j % len(cands)] for j in range(len(T))]
    idx = U.index_of(repl)
    A2 = A.copy()
    A2[T] = np.column_stack([U.X[idx], U.y[idx]])
    art2 = build_files(A2, R, T, meta)
    base, got = train_side_fields(built["art"]), train_side_fields(art2)
    differs = sorted(k for k in base if base[k] != got[k])
    return {"rule": f"outer-test rows of A replaced by U rows ranked by sha256(h NUL {DOWNSTREAM_DOMAIN} NUL position)",
            "test_rows_changed": int(np.count_nonzero(np.any(A2[T] != A[T], axis=1))),
            "encoder_raw_changed": sha(art2["split"]["jtt1"]) != sha(built["art"]["split"]["jtt1"]),
            "train_side_fields": sorted(base), "train_side_differs": differs, "train_side_invariant": not differs}


def new_view_id(h: str, fold: int, view_identity: str) -> str:
    return f"{h}__r0__f{int(fold)}__{VIEW_TAG}_{view_identity[:12]}"


def l7b_pointer(view_id: str, h: str):
    p = L7B / f"shard-{int(h[:8], 16) % L7B_OF}" / view_id / "context.json"
    return {"path": str(p), "sha256": sha_file(p)} if p.exists() and (p.parent / ".done").exists() else None


def compose_context(ds, row, label, acq_rec, view, cf, dcf, built, codes, refs, budget) -> dict:
    h, f = ds["h"], int(row["fold"])
    U, art, apos, R, T = ds["U"], built["art"], built["apos"], built["R"], built["T"]
    split = art["split"]
    vid = new_view_id(h, f, view.identity)
    files = {"encoder.jtt1.gz": {"sha256": sha(art["enc_gz"]), "raw_sha256": sha(split["jtt1"]), "bytes": len(art["enc_gz"])},
             "teacher.jtt1.gz": {"sha256": sha(art["tea_gz"]), "raw_sha256": sha(art["teacher"]), "bytes": len(art["tea_gz"])},
             "a0.f32": {"sha256": sha(art["a0"]), "bytes": len(art["a0"])}}
    tr_sha, te_sha = sha(json.dumps(R).encode()), sha(json.dumps(T).encode())
    apos_sha = sha(json.dumps(apos).encode())
    scaler, inner = sha(trainonly.scaler_bytes(split)), sha(np.asarray(split["inner_labels"], np.uint8).tobytes())
    epoch0 = {"view_id": row["view_id"], "context_sha256": row["context_sha256"]}
    identity = {"schema": IDENTITY_SCHEMA, "dataset_hash": h, "fold": f, "repeat": 0, "view_id": vid,
                "rowsel_view_identity": view.identity, "values_mode": "L7b", "route": ROUTE,
                "universe": {"t1_train_sha256": ds["t1"]["train"]["sha256"], "t1_test_sha256": ds["t1"]["test"]["sha256"],
                             "rows": U.n},
                "parent_split": {"sha256": ds["split"]["sha256"], "version_id": ds["split"]["version_id"]},
                "t1_meta_sha256": ds["meta_sha256"], "a_positions_sha256": apos_sha,
                "outer_train_indices_sha256": tr_sha, "outer_test_indices_sha256": te_sha,
                "files_raw_sha256": {"encoder.jtt1": files["encoder.jtt1.gz"]["raw_sha256"],
                                     "teacher.jtt1": files["teacher.jtt1.gz"]["raw_sha256"],
                                     "a0.f32": files["a0.f32"]["sha256"]},
                "scaler_sha256": scaler, "inner_labels_sha256": inner, "epoch0": epoch0}
    fp = view.train_fit.fingerprints
    quotas_all = rowsel.fold_quotas({k: len(v) for k, v in ds["folds"].items()}, budget)
    block = {
        "route": ROUTE,
        "view": {"schema": rowsel.SCHEMA, "identity": view.identity, "port": rowsel.PORT,
                 "train_fit_identity": view.train_fit.identity, "passthrough": view.train_fit.passthrough,
                 "fit_rows": fp["fit_rows"], "fit_positions_sha256": fp["fit_positions_sha256"],
                 "train_positions_sha256": sha(rowsel._u64(view.train_positions)),
                 "test_positions_sha256": sha(rowsel._u64(view.test_positions)),
                 "test_draw_sha256": view.test_draw_sha256, "draw_domain": rowsel.DRAW_DOMAIN,
                 "train_values_sha256": view.train_values_sha256, "test_values_sha256": view.test_values_sha256,
                 "train_output_order_sha256": rowsel.selected_source_indices_sha256(view.train_fit.output_order)},
        "universe": {"kind": "t1", "dataset_hash": h, "rows": U.n, "features": U.features,
                     "train": ds["t1"]["train"], "test": ds["t1"]["test"],
                     "position_rule": "T1 train lines then test lines; position = concatenation index (materialize.py)",
                     "row_id_rule": "rowsel.positional_row_id(dataset_hash, partition, partition_index)"},
        "parent_split": dict(ds["split"], fold_sizes={str(k): len(v) for k, v in sorted(ds["folds"].items())},
                             fold_size=len(ds["folds"][f])),
        "quotas": {"budget": int(budget), "n_train": int(view.quotas[0]), "n_test": int(view.quotas[1]),
                   "quota_met": len(view.train_positions) == view.quotas[0] and len(view.test_positions) == view.quotas[1],
                   "rule": QUOTA_RULE, "all_folds": {str(k): list(v) for k, v in sorted(quotas_all.items())}},
        "seed": view.seed, "seed_rule": "sha256(dataset_hash NUL 'trainonly-rowsel-v1') (rowsel.derive_seed)",
        "rowsel_py_sha256": codes["dltrain/rowsel.py"],
        "outer_split_changed": False,
        "a_order": "A rows = T1 lines at the ascending union of train and test positions; R/T index A",
        "a_positions_sha256": apos_sha,
        "train_positions": [int(p) for p in view.train_positions],
        "test_positions": [int(p) for p in view.test_positions],
        "counterfactual": cf,
        "downstream_counterfactual": dcf,
        "lineage": {k: ds["lineage"][k] for k in ("manifest_sha256", "membership_sha256", "data_sha256",
                                                  "membership_selected_sha256_matches_manifest",
                                                  "port_reproduces_v1_selection")}
                   | {"membership_remap_matches_epoch0_fold": ds["lineage"]["membership_remap_matches_epoch0_folds"][str(f)]},
    }
    if label == "acquisition_cap" and row.get("cohort") == "incoming":   # ROWSEL-SPLIT raw route (incoming capped)
        block["raw_route_superseded_if_matched"] = True
    return {"schema": SCHEMA, "view_id": vid, "dataset_hash": h, "fold": f, "repeat": 0, "cohort": row.get("cohort"),
            "owner_line": refs.get("owner_line", OWNER_LINE), **refs.get("context_extra", {}),
            "context_sha256": sha(rowsel._canon(identity)), "identity": identity,
            "epoch0": dict(epoch0, epoch_sha256=EPOCH0_SHA256, l7b_context=l7b_pointer(row["view_id"], h)),
            "row_selection_leak": [label], "row_selection_leak_epoch0": ["kmeans"],
            "row_selection_leak_receipt": refs.get("row_selection_leak"),
            "acquisition": dict(acq_rec, source=refs.get("acq_selection")),
            "rowsel": block,
            "outer_train_indices_sha256": tr_sha, "outer_test_indices_sha256": te_sha, "n_train": len(R), "n_test": len(T),
            "files": files, "scaler_sha256": scaler, "inner_labels_sha256": inner,
            "inner_loads": split["inner_cv"]["loads"], "t1prep": art["t1prep"], "t1_meta_sha256": ds["meta_sha256"],
            "values_mode": "L7b", "published": False, "code_sha256": codes}


def rebuild_context(ds, row, label, acq_rec, shard_dir: Path, codes, refs, *, budget=rowsel.DEFAULT_BUDGET,
                    select_fn=rowsel.select_view) -> dict:
    """One context: view, gate 1, A/R/T and files, gate 2, then the writes (.done last). Returns
    {"status": "done", "view_id", "dir", "context"} or {"status": "failed", "reason", "detail"}; a failed context
    writes nothing."""
    t0 = time.process_time()
    U, folds, f = ds["U"], ds["folds"], int(row["fold"])
    view = select_fn(U, folds, f, budget, None, quotas=None)
    cf = rowsel.counterfactual_report(select_fn, U, folds, f, budget)
    cf_ok = cf["train_side_invariant"] and cf["test_draw_invariant"] and cf["base_view_identity"] == view.identity
    if not cf_ok:
        bad = {k: v for k, v in cf["cases"].items() if v.get("train_side_differs") or v.get("test_draw_differs")}
        return {"status": "failed", "reason": "counterfactual_failed",
                "detail": json.dumps({"train_side_invariant": cf["train_side_invariant"],
                                      "test_draw_invariant": cf["test_draw_invariant"], "cases": bad}, sort_keys=True)[:600]}
    held = set(int(p) for p in folds[f])
    if set(view.train_positions) & held or not set(view.test_positions) <= held:
        return {"status": "failed", "reason": "view_outside_parent_fold", "detail": f"fold {f}"}
    built = assemble(U, view, ds["meta"])
    dcf = downstream_counterfactual(U, built, ds["meta"], ds["h"])
    if not dcf["train_side_invariant"]:
        return {"status": "failed", "reason": "downstream_counterfactual_failed",
                "detail": json.dumps(dcf["train_side_differs"])}
    ctx = compose_context(ds, row, label, acq_rec, view, cf, dcf, built, codes, refs, budget)
    ctx["build_cpu_s"] = round(time.process_time() - t0, 3)
    ctx["utc_us"] = int(time.time() * 1e6)
    d = shard_dir / ctx["view_id"]
    d.mkdir(parents=True, exist_ok=True)
    art = built["art"]
    atomic(d / "encoder.jtt1.gz", art["enc_gz"])
    atomic(d / "teacher.jtt1.gz", art["tea_gz"])
    atomic(d / "a0.f32", art["a0"])
    atomic(d / "context.json", jdump(ctx))
    atomic(d / ".done", b"ok\n")
    return {"status": "done", "view_id": ctx["view_id"], "dir": d, "context": ctx}


# ------------------------------------------------------------------------------------------------- shard loop

def guard():
    pauses = 0
    ok, reasons, _ = cpuguard.check()
    while not ok:
        pauses += 1
        print(json.dumps({"guard": "PAUSE", "reasons": reasons}), flush=True)
        time.sleep(15)
        ok, reasons, _ = cpuguard.check()
    return pauses


def done_index(shard_dir: Path) -> dict:
    """epoch-0 view_id -> rebuilt context dir, over the contexts with .done (resume)."""
    out = {}
    if shard_dir.is_dir():
        for d in shard_dir.iterdir():
            if (d / ".done").exists() and (d / "context.json").exists():
                out[json.loads((d / "context.json").read_bytes())["epoch0"]["view_id"]] = d
    return out


def build_shard(out: Path, shard: int, of: int, items: list, loader, codes: dict, refs: dict, *,
                budget=rowsel.DEFAULT_BUDGET, select_fn=rowsel.select_view, guard_fn=guard) -> dict:
    """Build every scope item of this shard (dataset_hash[:8] % of == shard). `loader(h, rows)` returns the dataset
    inputs (load_dataset in production). Resumable: a context whose epoch-0 view_id already has a .done is skipped."""
    out = Path(out)
    shard_dir = out / f"shard-{shard}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    mine = [it for it in items if int(it[0]["dataset_hash"][:8], 16) % of == shard]
    groups = collections.OrderedDict()
    for it in mine:
        groups.setdefault(it[0]["dataset_hash"], []).append(it)
    t0, c0 = time.time(), cpu_s()
    done = done_index(shard_dir)
    counts = collections.Counter()
    failed, datasets, pauses = [], {}, 0
    for n, (h, its) in enumerate(groups.items()):
        todo = [it for it in its if it[0]["view_id"] not in done]
        counts["already_done"] += len(its) - len(todo)
        if not todo:
            continue
        pauses += guard_fn()
        try:
            ds = loader(h, [it[0] for it in its])
            check_cap_label(ds, its[0][2])
        except Exception as e:  # a dataset-level failure: every context not yet done fails with the dataset's reason
            reason = e.reason if isinstance(e, RebuildError) else "dataset_error"
            for r, lab, _ in todo:
                failed.append({"view_id": r["view_id"], "dataset_hash": h, "fold": int(r["fold"]), "label": lab,
                               "reason": reason, "detail": f"{type(e).__name__}: {e}"[:400]})
            datasets[h] = {"status": "failed", "reason": reason}
            continue
        datasets[h] = {"status": "loaded", "load_cpu_s": round(ds.get("load_cpu_s", 0.0), 3),
                       **{k: ds["lineage"].get(k) for k in ("universe_rows", "features", "port_reproduces_v1_selection")}}
        dsdir = out / "_datasets"
        dsdir.mkdir(parents=True, exist_ok=True)
        atomic(dsdir / f"{h}.json", jdump(ds["lineage"]))
        for r, lab, acq_rec in todo:
            pauses += guard_fn()
            try:
                res = rebuild_context(ds, r, lab, acq_rec, shard_dir, codes, refs, budget=budget, select_fn=select_fn)
            except Exception as e:  # recorded per context; the shard continues
                res = {"status": "failed", "reason": e.reason if isinstance(e, RebuildError) else "context_error",
                       "detail": f"{type(e).__name__}: {e}"[:400]}
            if res["status"] == "done":
                counts["built_now"] += 1
                counts[f"built_now_{lab}"] += 1
            else:
                failed.append({"view_id": r["view_id"], "dataset_hash": h, "fold": int(r["fold"]), "label": lab,
                               "reason": res["reason"], "detail": res["detail"]})
        print(json.dumps({"shard": shard, "datasets": n + 1, "of": len(groups), "built_now": counts["built_now"],
                          "failed": len(failed), "elapsed_s": round(time.time() - t0)}), flush=True)
    done = done_index(shard_dir)
    done_labels = collections.Counter(json.loads((d / "context.json").read_bytes())["row_selection_leak"][0]
                                      for v, d in done.items() if v in {it[0]["view_id"] for it in mine})
    expected_labels = collections.Counter(lab for _, lab, _ in mine)
    ndone = sum(done_labels.values())
    rec = {"schema": SHARD_SCHEMA, "shard": shard, "of": of, "out": str(out),
           "expected": len(mine), "expected_by_label": dict(expected_labels), "expected_datasets": len(groups),
           "built_now": counts["built_now"], "already_done": counts["already_done"],
           "done": ndone, "done_by_label": dict(done_labels),
           "failed": failed, "failed_by_reason": dict(collections.Counter(x["reason"] for x in failed)),
           "counterfactual_failures": sum(1 for x in failed if x["reason"] in ("counterfactual_failed",
                                                                                "downstream_counterfactual_failed")),
           "complete": ndone == len(mine) and not failed, "datasets": datasets,
           "cpu_s": round(cpu_s() - c0, 2), "elapsed_s": round(time.time() - t0, 1), "guard_pauses": pauses,
           "budget": int(budget), "values_mode": "L7b", "route": ROUTE, "published": False, "owner_line": OWNER_LINE,
           "inputs": refs, "code_sha256": codes, "utc_us": int(time.time() * 1e6)}
    atomic(out / f"shard-{shard}-of-{of}.receipt.json", (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode())
    return rec


def input_refs(scope_refs: dict) -> dict:
    refs = dict(scope_refs)
    refs["epoch0"] = {"path": str(EPOCH0), "sha256": EPOCH0_SHA256}
    refs["datasets_inventory"] = file_ref(DATASETS_INV)
    refs["plan"] = file_ref(PLAN) if PLAN.exists() else None
    refs["owner_decisions"] = file_ref(OWNER_DECISIONS) if OWNER_DECISIONS.exists() else None
    return refs


def build(a) -> int:
    cpuguard.set_threads(2)
    codes = code_shas()
    src = scale_check.Sources(load_index=False)
    cache = scale_check.DataCache(max_items=4)
    inv = json.loads(DATASETS_INV.read_text())
    scope, scope_refs, counts = load_scope(src.included(), scope=a.scope)
    if a.datasets:                                    # pilot runs only; a full build leaves this unset
        keep = set(a.datasets.split(","))
        scope = [it for it in scope if it[0]["dataset_hash"] in keep]
        if {it[0]["dataset_hash"] for it in scope} != keep:
            raise SystemExit(f"--datasets outside the admitted scope: {sorted(keep - {it[0]['dataset_hash'] for it in scope})}")
    refs = input_refs(scope_refs) | {"scope_counts": counts, "owner_line": SCOPES[a.scope]["owner_line"]}
    rec = build_shard(Path(a.out), a.shard, a.of, scope, lambda h, rows: load_dataset(h, rows, src, cache, inv), codes, refs)
    print(json.dumps({k: rec[k] for k in ("shard", "expected", "built_now", "already_done", "done", "done_by_label",
                                          "failed_by_reason", "complete", "cpu_s", "elapsed_s")}))
    return 0 if rec["complete"] else 1


# ------------------------------------------------------------------------------------------------- overlay

def verify_context_dir(d: Path) -> dict:
    """A done context: schema, identity recompute, view_id suffix, gates passed and file sha256/bytes on disk."""
    ctx = json.loads((d / "context.json").read_bytes())
    problems = []
    if ctx.get("schema") != SCHEMA or ctx.get("published") is not False or ctx.get("values_mode") != "L7b":
        problems.append("schema/published/values_mode")
    if ctx.get("view_id") != d.name or ctx["identity"]["view_id"] != d.name:
        problems.append("view_id")
    if sha(rowsel._canon(ctx["identity"])) != ctx["context_sha256"]:
        problems.append("context_sha256")
    if d.name != new_view_id(ctx["dataset_hash"], ctx["fold"], ctx["rowsel"]["view"]["identity"]):
        problems.append("view_id_suffix")
    cf, dcf = ctx["rowsel"]["counterfactual"], ctx["rowsel"]["downstream_counterfactual"]
    if not (cf["train_side_invariant"] and cf["test_draw_invariant"] and dcf["train_side_invariant"]):
        problems.append("gates")
    for name, meta in ctx["files"].items():
        p = d / name
        if not p.is_file() or p.stat().st_size != meta["bytes"] or sha_file(p) != meta["sha256"]:
            problems.append(f"file:{name}")
    if "rights" in ctx and (ctx["rights"] != "research-only" or "#sha256=" not in str(ctx.get("rights_decision"))):
        problems.append("rights")
    unlisted = sorted({p.name for p in d.iterdir()} - set(ctx["files"]) - {"context.json", ".done"})
    if unlisted:
        problems.append(f"unlisted:{unlisted}")
    return {"context": ctx, "problems": problems}


def overlay_record(d: Path, ctx: dict) -> dict:
    rs = ctx["rowsel"]
    rec = {"kind": "replace",
           "key": {"dataset_hash": ctx["dataset_hash"], "fold": ctx["fold"], "view_id": ctx["epoch0"]["view_id"],
                   "context_sha256": ctx["epoch0"]["context_sha256"]},
           "replacement": {"view_id": ctx["view_id"], "context_sha256": ctx["context_sha256"], "schema": ctx["schema"],
                           "values_mode": ctx["values_mode"],
                           "context_json": {"path": str(d / "context.json"), "sha256": sha_file(d / "context.json"),
                                            "bytes": (d / "context.json").stat().st_size},
                           "files": {k: v["sha256"] for k, v in sorted(ctx["files"].items())}},
           "row_selection_leak": ctx["row_selection_leak"], "row_selection_leak_epoch0": ctx["row_selection_leak_epoch0"],
           "rowsel": {"route": rs["route"], "view_identity": rs["view"]["identity"],
                      "universe": {"kind": rs["universe"]["kind"], "t1_train_sha256": rs["universe"]["train"]["sha256"],
                                   "t1_test_sha256": rs["universe"]["test"]["sha256"], "rows": rs["universe"]["rows"]},
                      "parent_split": {"kind": rs["parent_split"]["kind"], "sha256": rs["parent_split"]["sha256"],
                                       "version_id": rs["parent_split"]["version_id"]},
                      "outer_split_changed": rs["outer_split_changed"], "rowsel_py_sha256": rs["rowsel_py_sha256"],
                      "quotas": [rs["quotas"]["n_train"], rs["quotas"]["n_test"]]},
           "admission": {"counterfactual": "pass", "downstream_counterfactual": "pass",
                         "v2_publish": "pending: the publisher admits only the local and l7a schemas and the clean label"}}
    if rs.get("raw_route_superseded_if_matched"):
        rec["rowsel"]["raw_route_superseded_if_matched"] = True
    for k in ("rights", "rights_decision"):
        if k in ctx:
            rec[k] = ctx[k]
    return rec


def write_overlay(out: Path, overlay_path: Path = OVERLAY, epoch_path: Path = EPOCH0,
                  epoch_sha256: str | None = EPOCH0_SHA256) -> dict:
    """Append one `replace` record per done context not yet in the overlay. The file is created with a header that
    pins sha256(epoch-000.json); an existing file must carry that header and is only ever appended to. A context whose
    key is already present with a different replacement refuses the whole append."""
    eb = Path(epoch_path).read_bytes()
    esha = sha(eb)
    if epoch_sha256 is not None and esha != epoch_sha256:
        raise RebuildError("epoch_drift", f"{epoch_path} {esha} != {epoch_sha256}")
    header = {"kind": "header", "schema": OVERLAY_SCHEMA, "epoch": 1, "owner_line": OWNER_LINE,
              "overlay_of": {"path": str(epoch_path), "sha256": esha, "bytes": len(eb)},
              "key": ["dataset_hash", "fold", "view_id", "context_sha256"],
              "rule": "epoch-000.json is never edited; a keyed epoch-0 context is replaced by the record's replacement; "
                      "an epoch-0 context without a record stays as it is; append-only"}
    recs, bad = [], []
    for d in sorted(Path(out).glob("shard-*/*")):
        if not d.is_dir() or not (d / ".done").exists():
            continue
        v = verify_context_dir(d)
        if v["problems"]:
            bad.append({"dir": str(d), "problems": v["problems"]})
            continue
        recs.append(overlay_record(d, v["context"]))
    if bad:
        raise RebuildError("overlay_unverified_contexts", json.dumps(bad[:5]))
    recs.sort(key=lambda r: (r["key"]["dataset_hash"], r["key"]["fold"], r["key"]["view_id"]))
    keys = [r["key"]["view_id"] for r in recs]
    if len(keys) != len(set(keys)):
        raise RebuildError("overlay_duplicate_key", "two rebuilt contexts replace one epoch-0 context")
    existing = {}
    path = Path(overlay_path)
    if path.exists():
        lines = [json.loads(l) for l in path.read_bytes().decode().splitlines() if l.strip()]
        if not lines or {k: lines[0].get(k) for k in ("kind", "schema", "epoch", "overlay_of")} != \
                {k: header[k] for k in ("kind", "schema", "epoch", "overlay_of")}:
            raise RebuildError("overlay_header_mismatch", str(path))
        for r in lines[1:]:
            existing[r["key"]["view_id"]] = r
    new = []
    for r in recs:
        old = existing.get(r["key"]["view_id"])
        if old is None:
            new.append(r)
        elif (old["key"], old["replacement"]["view_id"], old["replacement"]["context_sha256"]) != \
                (r["key"], r["replacement"]["view_id"], r["replacement"]["context_sha256"]):
            raise RebuildError("overlay_conflict", r["key"]["view_id"])
    body = b"".join(json.dumps(r, sort_keys=True, separators=(",", ":")).encode() + b"\n" for r in new)
    if not path.exists():
        hdr = json.dumps(dict(header, created_utc_us=int(time.time() * 1e6)), sort_keys=True,
                         separators=(",", ":")).encode() + b"\n"
        with open(path, "xb") as fh:
            fh.write(hdr + body); fh.flush(); os.fsync(fh.fileno())
    elif body:
        with open(path, "ab") as fh:
            fh.write(body); fh.flush(); os.fsync(fh.fileno())
    total = len(existing) + len(new)
    return {"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size, "appended": len(new),
            "records": total, "overlay_of": header["overlay_of"]}


# ------------------------------------------------------------------------------------------------- summary

def summarize(a) -> int:
    out = Path(a.out)
    src = scale_check.Sources(load_index=False)
    scope, scope_refs, counts = load_scope(src.included(), scope=a.scope)
    cfg = SCOPES[a.scope]
    by_view = {r["view_id"]: (r, lab) for r, lab, _ in scope}
    done, problems = {}, []
    for d in sorted(out.glob("shard-*/*")):
        if d.is_dir() and (d / ".done").exists():
            v = verify_context_dir(d)
            if v["problems"]:
                problems.append({"dir": str(d), "problems": v["problems"]})
            done[v["context"]["epoch0"]["view_id"]] = v["context"]
    shard_recs = []
    for p in sorted(out.glob(f"shard-*-of-{a.of}.receipt.json")):
        r = json.loads(p.read_bytes())
        shard_recs.append({"path": str(p), "sha256": sha_file(p), "expected": r["expected"], "built_now": r["built_now"],
                           "already_done": r["already_done"], "done": r["done"], "failed": len(r["failed"]),
                           "counterfactual_failures": r["counterfactual_failures"], "complete": r["complete"],
                           "cpu_s": r["cpu_s"], "elapsed_s": r["elapsed_s"], "guard_pauses": r["guard_pauses"]})
    failed = {}
    for p in sorted(out.glob(f"shard-*-of-{a.of}.receipt.json")):
        for x in json.loads(p.read_bytes())["failed"]:
            if x["view_id"] not in done:
                failed[x["view_id"]] = x
    missing = sorted(v for v in by_view if v not in done and v not in failed)
    done_labels = collections.Counter(c["row_selection_leak"][0] for c in done.values())
    label_ok = all(c["row_selection_leak"] == [by_view[v][1]] for v, c in done.items() if v in by_view)
    want_extra = scope_refs.get("context_extra", {})
    rights_ok = all({k: c.get(k) for k in ("rights", "rights_decision")} == {k: want_extra.get(k) for k in ("rights", "rights_decision")}
                    for c in done.values())
    outside = sorted(v for v in done if v not in by_view)
    ds_rec = [json.loads(p.read_bytes()) for p in sorted((out / "_datasets").glob("*.json"))]
    port = collections.Counter(bool(d["port_reproduces_v1_selection"]) for d in ds_rec)
    cf_cpu = sum(c.get("build_cpu_s", 0.0) for c in done.values())
    tests = {"log": file_ref(Path(a.tests_log)), "tail": Path(a.tests_log).read_text().strip().splitlines()[-3:]}
    mut = json.loads(Path(a.mutants).read_bytes())
    mutants = {"result": file_ref(Path(a.mutants)), "totals": mut.get("totals"),
               "survived": {k: v.get("why_survived") for k, v in (mut.get("mutants") or {}).items() if not v.get("killed")}}
    overlay = file_ref(Path(a.overlay_path)) if Path(a.overlay_path).exists() else None
    if overlay:
        lines = Path(a.overlay_path).read_bytes().decode().splitlines()
        overlay["records"] = len(lines) - 1
        overlay["header"] = json.loads(lines[0])
    pilot = None
    if a.pilot_out:
        pr = [json.loads(p.read_bytes()) for p in sorted(Path(a.pilot_out).glob("shard-*-of-*.receipt.json"))]
        pilot = {"out": a.pilot_out, "receipts": [file_ref(p) for p in sorted(Path(a.pilot_out).glob("shard-*-of-*.receipt.json"))],
                 "expected": sum(r["expected"] for r in pr), "done": sum(r["done"] for r in pr),
                 "failed": sum(len(r["failed"]) for r in pr), "cpu_s": sum(r["cpu_s"] for r in pr),
                 "datasets": sorted({h for r in pr for h in r["datasets"]})}
    root = Path(__file__).resolve().parents[1]
    codes = code_shas() | {f: sha_file(root / f) for f in ("tests/test_v2_rowsel_build.py", "tests/test_rowsel.py")}
    doc = {"schema": SUMMARY_SCHEMA, "utc_us": int(time.time() * 1e6), "owner_line": cfg["owner_line"], "out": str(out),
           "of": a.of, "scope_name": a.scope, "cohort": cfg["cohort"], "rights": cfg["rights"],
           "scope": counts, "expected": cfg["expected"], "rights_fields_match_scope": rights_ok,
           "counts": {"expected": len(by_view), "done": len(done), "failed": len(failed), "missing": len(missing),
                      "done_by_label": dict(done_labels),
                      "expected_by_label": dict(collections.Counter(lab for _, lab in by_view.values())),
                      "datasets_done": len({c["dataset_hash"] for c in done.values()}),
                      "datasets_fully_done": len({h for h in {r["dataset_hash"] for r, _ in by_view.values()}
                                                  if all(v in done for v, (r, _) in by_view.items() if r["dataset_hash"] == h)})},
           "sums_to_expected": len(done) + len(failed) == len(by_view) and not missing,
           "labels_match_scope": label_ok, "done_outside_scope": outside, "verification_problems": problems,
           "failed": sorted(failed.values(), key=lambda x: x["view_id"]),
           "failed_by_reason": dict(collections.Counter(x["reason"] for x in failed.values())),
           "missing": missing[:50],
           "port_reproduces_v1_selection_datasets": {str(k): v for k, v in port.items()},
           "port_mismatch_datasets": sorted(d["dataset_hash"] for d in ds_rec if not d["port_reproduces_v1_selection"]),
           "v1_sampler_io_all_checks_pass": dict(collections.Counter(
               str(all(v for k, v in (d.get("v1_sampler_io") or {}).items() if isinstance(v, bool)))
               if d.get("v1_sampler_io") else "not_applicable" for d in ds_rec)),
           "meta_cap_markers": dict(collections.Counter("|".join(d.get("meta_cap_markers") or []) or "-" for d in ds_rec)),
           "cpu": {"shard_process_cpu_s": round(sum(r["cpu_s"] for r in shard_recs), 1),
                   "context_build_cpu_s_sum": round(cf_cpu, 1),
                   "wall_s_max_shard": max((r["elapsed_s"] for r in shard_recs), default=None)},
           "shard_receipts": shard_recs, "pilot": pilot, "tests": tests, "mutants": mutants, "overlay": overlay,
           "inputs": input_refs(scope_refs), "code_sha256": codes,
           "policy": {"s3_calls": 0, "network": False, "gpu": False, "git": False, "published": False,
                      "raw_sources_read": False, "epoch0_edited": False}}
    stem = "rowsel-t1-kmeans-build-summary" if a.scope == "incoming-kmeans" else f"rowsel-t1-{a.scope}-build-summary"
    rp = Path(a.receipt) if a.receipt else RECEIPTS / "build" / f"{stem}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    data = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    with open(rp, "xb") as fh:
        fh.write(data)
    print(json.dumps({"receipt": str(rp), "sha256": sha(data), "counts": doc["counts"], "failed_by_reason": doc["failed_by_reason"],
                      "sums_to_expected": doc["sums_to_expected"], "problems": len(problems), "overlay": overlay and overlay["sha256"]}))
    return 0 if doc["sums_to_expected"] and not problems and rights_ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--datasets", help="comma-separated dataset hashes (pilot only)")
    ap.add_argument("--scope", choices=sorted(SCOPES), default="incoming-kmeans")
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--overlay-path", default=str(OVERLAY))
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--tests-log")
    ap.add_argument("--mutants")
    ap.add_argument("--pilot-out")
    ap.add_argument("--receipt")
    a = ap.parse_args(argv)
    if a.overlay:
        res = write_overlay(Path(a.out), Path(a.overlay_path))
        print(json.dumps(res))
        return 0
    if a.summarize:
        return summarize(a)
    if a.shard is None:
        ap.error("--shard is required for a build")
    return build(a)


if __name__ == "__main__":
    sys.exit(main())

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

Usage (lane slice, through py.sh). --split-version is REQUIRED (PR #218 finding 10; no default): rowsel-raw-split-v2 or
rowsel-split-v2-groups with --tree; 'registry' (the routes above) only as an explicit historical replay:
  v2_rowsel_build.py --split-version registry --out DIR --shard I --of N [--scope S] [--datasets h1,h2]
  v2_rowsel_build.py --split-version registry --out DIR --of N --overlay [--overlay-path P]
  v2_rowsel_build.py --split-version registry --out DIR --of N --summarize [--scope S] --tests-log LOG --mutants JSON [--receipt P]
  v2_rowsel_build.py --split-version rowsel-raw-split-v2|rowsel-split-v2-groups --tree T (--plan|--shard I|--summarize) --of N
The 'port' check above (a v1 sampler fit on every row, i.e. every row's target; finding 3) runs only on the registry
replay; v2/v2g builds record it as not run.
"""
import argparse
import collections
import glob
import gzip
import hashlib
import importlib.util
import io
import json
import os
import resource
import sys
import time
from pathlib import Path

from dltrain import replacement

import numpy as np

from dltrain import (cpuguard, jtt1, metafeatures, rowsel, rowsel_raw, rowsel_raw_split_v2, rowsel_split_v2_groups, scale_check, t1prep,
                     target_linear, trainonly)

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
    # DRIFTED-T1-FALLBACK (owner-decisions-20261006T112914Z.json): an EXPLICIT opt-in scope, never a default. Selected
    # only with the t1_fallback record (rights-held fill completion 3aae3100, its 'drifted' list) passed by the caller.
    "frozen-capped-drifted": {"cohort": "frozen", "owner_line": "ROWSEL-TRAINONLY+DRIFTED-T1-FALLBACK",
                              "expected": {"datasets": 23, "contexts": 115, "clean": 0, "acquisition_cap": 115},
                              "rights": "research-only", "selector": "drifted"},
}
CODE_FILES = ("tools/v2_rowsel_build.py", "tools/v2_build_local.py", "dltrain/rowsel.py", "dltrain/t1prep.py",
              "dltrain/trainonly.py", "dltrain/metafeatures.py", "dltrain/scale_check.py", "dltrain/jtt1.py",
              "dltrain/cpuguard.py", "dltrain/rowsel_raw.py", "dltrain/rowsel_raw_split_v2.py", "dltrain/rowsel_split_v2_groups.py",
              "dltrain/target_linear.py", "dltrain/replacement.py")
LATER_CODE_FILES = frozenset({"dltrain/rowsel_split_v2_groups.py",    # added after the v2split trees were built
                              "dltrain/target_linear.py", "dltrain/replacement.py"})            # LINEAR-DETERMINED gate (after the v2gsplit trees)
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

DRIFTED_RECORD = RECEIPTS / "fill" / "rights-held-fill-completion-20261006T114619Z.json"
DRIFTED_RECORD_SHA256 = "3aae31003bdba6719a9d03802b872d789e40b6541d267df5d4a0054d30d03fd2"
DRIFTED_RULING = Path("/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261006T112914Z.json")
DRIFTED_RULING_SHA256 = "e413a92d9493d996ee571ebe7047e2b6a0229de15ba252c132b55e647ab1b61f"
ALL_DATA_RESEARCH = Path("/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261006T102636Z.json")
ALL_DATA_RESEARCH_SHA256 = "c7ab09322a8f4bb762083716c5f5dba210b7989eb49fc95a007dd16a8903d94c"
CONTENT_SPLIT_DIR = Path("/home/ubuntu/scratch/jopedime-regression-neural-encoding-v1/source-objects")


def drifted_fallback_record(path) -> dict:
    """The explicit t1_fallback record of DRIFTED-T1-FALLBACK: the 'drifted' list of the rights-held fill completion,
    sha256-pinned. Refused when not given (the fallback is never a default) or when its bytes differ."""
    if path is None:
        raise RebuildError("t1_fallback_record_required", "the drifted T1 fallback runs only with --t1-fallback-record")
    body = Path(path).read_bytes()
    if sha(body) != DRIFTED_RECORD_SHA256:
        raise RebuildError("t1_fallback_record_drift", f"{path} sha256 {sha(body)} != {DRIFTED_RECORD_SHA256}")
    ruling = DRIFTED_RULING.read_bytes()
    if sha(ruling) != DRIFTED_RULING_SHA256 or "DRIFTED-T1-FALLBACK" not in json.loads(ruling)["decisions"]:
        raise RebuildError("t1_fallback_ruling_drift", str(DRIFTED_RULING))
    drifted = {x["dataset_hash"]: x for x in json.loads(body)["drifted"]}
    ref_ = {"path": str(path), "sha256": DRIFTED_RECORD_SHA256}
    ruling_ref = {"path": str(DRIFTED_RULING), "sha256": DRIFTED_RULING_SHA256, "key": "DRIFTED-T1-FALLBACK"}
    reasons = {h: {"reason": x["raw_route_reason"], "why": "drifted: the raw content match fails against today's live source",
                   "detail": json.dumps(x.get("study"), sort_keys=True)[:600], "disposition": x.get("disposition"),
                   "ruling": ruling_ref} for h, x in drifted.items()}
    return {"datasets": drifted, "ref": ref_, "reasons": reasons, "ruling": ruling_ref}


def all_data_research_ref() -> str:
    body = ALL_DATA_RESEARCH.read_bytes()
    if sha(body) != ALL_DATA_RESEARCH_SHA256:
        raise RebuildError("rights_decision_drift", str(ALL_DATA_RESEARCH))
    return f"{ALL_DATA_RESEARCH}#sha256={ALL_DATA_RESEARCH_SHA256}"


def select_drifted(rows, leak: dict, acq: dict, drifted) -> list:
    """(epoch-0 row, 'acquisition_cap', acquisition record) of the drifted datasets only: frozen contexts labelled
    ["clean"] whose acquisition record is class clean, reduced (capped at acquisition, with its selection markers)."""
    out = []
    for r in rows:
        if r.get("dataset_hash") not in drifted:
            continue
        lab = leak.get(r["view_id"])
        if lab is None:
            raise RebuildError("scope_unlabelled", r["view_id"])
        if lab["cohort"] != "frozen" or lab["row_selection_leak"] != ["clean"] or r.get("cohort") != "frozen" \
                or (lab["dataset_hash"], int(lab["fold"])) != (r["dataset_hash"], int(r["fold"])):
            raise RebuildError("scope_label_mismatch", r["view_id"])
        a = acq.get(r["dataset_hash"])
        if a is None or (a.get("cohort"), a.get("class")) != ("frozen", "clean") or not a.get("reduced") or not a.get("selection_markers"):
            raise RebuildError("scope_acquisition_record", r["dataset_hash"])
        out.append((r, "acquisition_cap", a))
    return sorted(out, key=lambda t: (t[0]["dataset_hash"], int(t[0]["fold"])))


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


def load_scope(rows, leak_path=LEAK, acq_path=ACQ, expected="scope", scope="incoming-kmeans", fallback_record=None):
    cfg = SCOPES[scope]
    if cfg.get("selector") == "drifted":
        return load_drifted_scope(rows, leak_path, acq_path, expected, scope, fallback_record)
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


def load_drifted_scope(rows, leak_path, acq_path, expected, scope, fallback_record):
    cfg = SCOPES[scope]
    rec = drifted_fallback_record(fallback_record)                    # explicit opt-in: refused without the record
    if expected == "scope":
        expected = cfg["expected"]
    lb, ab = Path(leak_path).read_bytes(), Path(acq_path).read_bytes()
    if sha(lb) != LEAK_SHA256 or sha(ab) != ACQ_SHA256:
        raise RebuildError("scope_input_drift", f"leak {sha(lb)} acq {sha(ab)}")
    leak = {}
    for line in lb.decode().splitlines():
        if line.strip():
            r = json.loads(line)
            leak[r["view_id"]] = r
    selected = select_drifted(rows, leak, json.loads(ab), rec["datasets"])
    counts = scope_counts(selected)
    if expected is not None and counts != expected:
        raise RebuildError("scope_counts", f"{counts} != {expected}")
    extra = {"rights": cfg["rights"], "rights_decision": all_data_research_ref()}
    refs = {"scope": scope, "row_selection_leak": {"path": str(leak_path), "sha256": sha(lb), "rows": len(leak)},
            "acq_selection": {"path": str(acq_path), "sha256": sha(ab)}, "t1_fallback_record": rec["ref"],
            "t1_fallback_ruling": rec["ruling"], "context_extra": extra, "rights_decision": extra["rights_decision"],
            "t1_fallback_reasons": rec["reasons"]}
    return selected, refs, counts


def content_addressed_split(cohort: str, h: str, sref: dict, split_dir=CONTENT_SPLIT_DIR):
    """The registry split object by its sha256 from a local content-addressed object directory (verified at use; no
    fetch). Used only by the drifted T1 fallback, whose objects are not in the frozen split-fetch receipt."""
    path = Path(split_dir) / sref["sha256"]
    return read_verified(path, sref["sha256"], sref.get("bytes")), {"path": str(path), "source": "content-addressed local object (sha256 verified)"}


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


PORT_NOT_RUN = "not run: registry replay only (all-row targets; #218 finding 3)"


def load_dataset(h: str, ctx_rows: list, src, cache, inv: dict, budget: int = rowsel.DEFAULT_BUDGET, *,
                 v1_port_check: bool | None = None, split_loader=None) -> dict:
    """Every input of one dataset, each sha256-verified, plus the dataset-level lineage checks.

    v1_port_check (required; None is refused): True ONLY for the explicit registry replay (--split-version registry).
    It re-runs the v1 research sampler over EVERY universe row -- rowsel.sampler_select(U, U.positions) reads the
    target of each row, outer-test rows of any fold included (PR #218 finding 3) -- to check that the port reproduces
    the epoch-0 selection (lineage only: the boolean feeds no fold, view, file or gate). The v2/v2g builds pass False:
    the port is not run and the lineage records port_reproduces_v1_selection = port_fit_cpu_s = None with the reason
    PORT_NOT_RUN. Contexts already built by c0b989b6/6e1e166e/fcfe3828 keep the bit those versions computed."""
    if v1_port_check is None:
        raise ValueError("load_dataset: v1_port_check must be given (True only for the explicit registry replay)")
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
    split_body, split_src = (split_loader or load_split)(cohort, h, sref)
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
    port_ok = port_s = None
    if v1_port_check:
        # REGISTRY REPLAY ONLY: the v1 sampler fit over all universe rows (every row's target; finding 3). Reached only
        # through build() of --split-version registry, which passes v1_port_check=True.
        tp = time.process_time()
        port = rowsel.sampler_select(U, U.positions.tolist(), budget, rowsel.v1_seed(h))
        port_ok = rowsel.selected_source_indices_sha256(port.output_order) == manifest["selected_source_indices_sha256"]
        port_s = round(time.process_time() - tp, 3)
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
               "port_reproduces_v1_selection": port_ok, "port_fit_cpu_s": port_s,
               "t1_meta_sha256": k}
    if not v1_port_check:
        lineage["port_reproduces_v1_selection_not_run"] = PORT_NOT_RUN
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
        pending_views = [it for it in its if it[0]["view_id"] not in done]
        counts["already_done"] += len(its) - len(pending_views)
        if not pending_views:
            continue
        pauses += guard_fn()
        try:
            ds = loader(h, [it[0] for it in its])
            check_cap_label(ds, its[0][2])
        except Exception as e:  # a dataset-level failure: every context not yet done fails with the dataset's reason
            reason = e.reason if isinstance(e, RebuildError) else "dataset_error"
            for r, lab, _ in pending_views:
                failed.append({"view_id": r["view_id"], "dataset_hash": h, "fold": int(r["fold"]), "label": lab,
                               "reason": reason, "detail": f"{type(e).__name__}: {e}"[:400]})
            datasets[h] = {"status": "failed", "reason": reason}
            continue
        datasets[h] = {"status": "loaded", "load_cpu_s": round(ds.get("load_cpu_s", 0.0), 3),
                       **{k: ds["lineage"].get(k) for k in ("universe_rows", "features", "port_reproduces_v1_selection")}}
        dsdir = out / "_datasets"
        dsdir.mkdir(parents=True, exist_ok=True)
        atomic(dsdir / f"{h}.json", jdump(ds["lineage"]))
        for r, lab, acq_rec in pending_views:
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
    rec = build_shard(Path(a.out), a.shard, a.of, scope, lambda h, rows: load_dataset(h, rows, src, cache, inv, v1_port_check=True),
                      codes, refs)
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


# ------------------------------------------------------------------------------------------------- split v2 (NONRAW-SPLIT-A)
#
# --split-version rowsel-raw-split-v2 (owner ruling NONRAW-SPLIT-A, owner-decisions-20261006T053215Z.json): the registry
# parent folds (grouped-stratified-kfold/v1 hashes the WHOLE row, target included, into its duplicate key, so y reaches
# fold assignment) are replaced by dltrain/rowsel_raw_split_v2 over this route's universe U:
#   values      U's T1 values; every T1 column (categorical codes included: in T1 they are numbers) is a numeric
#               feature for the grouping; the target column is never part of a group key
#   partition   "t1": row rank = sha256(h NUL 'trainonly-rowsel-split-v1' NUL 't1' NUL decimal(position)), position = the
#               concatenation index of U (T1 train lines then test lines, as materialize.py)
#   split id    sha256(canonical JSON {schema rowsel-raw-split-v2, dataset_hash, partition 't1', folds})
#   per dataset gates: no duplicate group straddles folds, split_target_check (targets permuted / replaced) and
#               rowsel_raw_split_v2.value_check (one outer-test fold's features and targets changed with the duplicate
#               structure kept) must leave the folds identical, and no fold may be empty
#   per fold    rowsel.select_view on the v2 folds (quotas from the new fold sizes), both counterfactual gates, files
#   replacement DATASET level: the 5 new contexts replace all epoch-0 contexts of the dataset as a set (new fold f is not
#               registry fold f); written only when every fold passes (files and context.json first, .done last)
#   cut         every context carries cut = {cut_id: 'trainonly-split-v2:' + split sha256 (= rowsel.split.sha256),
#               schema, replaces_cut: the registry split object it replaces, comparable_with_replaced: false}
# The registry route is a historical replay, run only when --split-version registry is passed explicitly (no default:
# PR #218 finding 10); the registry split is still loaded and lineage-verified here, because it is the cut being
# replaced. The v1-port lineage check (all-row targets) runs only on that replay (load_dataset v1_port_check).

SPLIT_REGISTRY = "registry"
SPLIT_V2 = rowsel_raw_split_v2.SPLIT_SCHEMA
SPLIT_V2G = rowsel_split_v2_groups.SPLIT_SCHEMA
SPLIT_VERSIONS = (SPLIT_REGISTRY, SPLIT_V2, SPLIT_V2G)
VIEW_TAG_V2G = "rowsel2g"
T1_PARTITION = "t1"
CUT_PREFIX = "trainonly-split-v2:"
SCHEMA_V2 = "jope.dl-v2-rowsel-context/v2"
IDENTITY_SCHEMA_V2 = "jope.dl-v2-rowsel-context-identity/v2"
DATASET_SCHEMA_V2 = "jope.dl-v2-rowsel-dataset/v2"
PLAN_SCHEMA_V2 = "jope.dl-v2-rowsel-v2split-plan/v1"
SHARD_SCHEMA_V2 = "jope.dl-v2-rowsel-v2split-shard/v1"
SUMMARY_SCHEMA_V2 = "jope.dl-v2-rowsel-v2split-summary/v1"
VIEW_TAG_V2 = "rowsel2s"
NONRAW_DECISION = Path("/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261006T053215Z.json")
NONRAW_DECISION_SHA256 = "8c455c418ec985d64b07063b73aa15c6f1a59161adfe8e0cbb0ac95c1e13a2ea"
FALLBACK_SOURCE = RECEIPTS / "build" / "rowsel-raw-v2-coverage-summary-20261006T052815Z.json"
FALLBACK_SOURCE_SHA256 = "b3f1a9e024b27e8f7bd583ea8f0283925f1f6194ea02afc271119160614126c9"
RAW_V2_PLAN = RUN / "build-v2" / "ROWSEL-RAW-v2" / "plan.json"
CENSUS = RUN / "tmp" / "split-scope" / "census-all.jsonl"
CENSUS_SHA256 = "290b81e9c40c3f4e558483c02210afb63fdb026b216a182f1460bba7bf5a1d5d"
TREES = {
    "kmeans-clean": {"scope": "incoming-kmeans", "label": "clean", "fallback": False,
                     "out": RUN / "build-v2" / "ROWSEL-T1-kmeans-v2split", "v1_tree": RUN / "build-v2" / "ROWSEL-T1-kmeans-v1",
                     "expected": {"datasets": 173, "epoch0_contexts": 810}},
    "kmeans-fallback": {"scope": "incoming-kmeans", "label": "acquisition_cap", "fallback": True,
                        "out": RUN / "build-v2" / "ROWSEL-T1-fallback-v2split", "v1_tree": RUN / "build-v2" / "ROWSEL-T1-kmeans-v1",
                        "expected": {"datasets": 18, "epoch0_contexts": 75}},
    "frozen-kmeans-research": {"scope": "frozen-kmeans-research", "label": "clean", "fallback": False,
                               "out": RUN / "build-v2" / "ROWSEL-T1-frozen-kmeans-research-v2split",
                               "v1_tree": RUN / "build-v2" / "ROWSEL-T1-frozen-kmeans-research-v1",
                               "expected": {"datasets": 42, "epoch0_contexts": 209}},
}
TREES["frozen-capped-drifted-fallback"] = {
    "scope": "frozen-capped-drifted", "label": "acquisition_cap", "fallback": True, "fallback_kind": "drifted", "content_split": True,
    "opt_in": "--t1-fallback-record", "out": RUN / "build-v2" / "ROWSEL-T1-frozen-capped-fallback-v2split",
    "v1_tree": RUN / "build-v2" / "ROWSEL-T1-frozen-capped-fallback-has-no-v1-tree", "expected": {"datasets": 23, "epoch0_contexts": 115}}
SPLIT_V2_PARTITION_RULE = ("partition 't1': position = concatenation index of U (T1 train lines then test lines, "
                           "materialize.py); row rank sha256(h NUL 'trainonly-rowsel-split-v1' NUL 't1' NUL decimal(position))")


def t1_split_inputs(U: rowsel.Universe) -> tuple:
    """(values, specs) for rowsel_raw_split_v2 over a T1 universe: every T1 column a numeric feature, target last."""
    values = np.ascontiguousarray(np.column_stack([U.X, U.y]), dtype=np.float64)
    specs = [{"index": j, "name": f"t1_col{j}", "source_name": f"t1_col{j}", "kind": "numeric"} for j in range(U.features)]
    specs.append({"index": U.features, "name": "target", "source_name": "target", "kind": "numeric", "target": True})
    return values, specs


def t1_prep(U: rowsel.Universe):
    values, specs = t1_split_inputs(U)
    return rowsel_raw.Prep(U.dataset_hash, specs, np.asarray(U.positions, dtype=np.int64), values, U.row_ids, U.row_bytes, {})


def split_v2_t1(h, positions, values, specs, folds=rowsel_raw.OUTER_FOLDS):
    return rowsel_raw_split_v2.outer_split(h, positions, values, specs, folds, partition=T1_PARTITION)


def split_v2_t1_sha256(h: str, folds) -> str:
    return sha(rowsel._canon({"schema": SPLIT_V2, "dataset_hash": h, "partition": T1_PARTITION,
                              "folds": [[int(p) for p in folds[f]] for f in sorted(folds)]}))


def split_v2_t1_report(prep, folds) -> dict:
    rep = rowsel_raw.split_report(prep, folds)
    rep.update(schema=SPLIT_V2, sha256=split_v2_t1_sha256(prep.dataset_hash, folds), domain=rowsel_raw_split_v2.SPLIT_DOMAIN,
               rule=rowsel_raw_split_v2.RULE, partition=T1_PARTITION, partition_rule=SPLIT_V2_PARTITION_RULE)
    rep["target_check"] = rowsel_raw.split_target_check(split_v2_t1, prep)
    rep["value_check_fold0"] = rowsel_raw_split_v2.value_check(split_v2_t1, prep, 0)
    return rep


def empty_fold_list(fold_sizes) -> list:
    return [i for i, n in enumerate(fold_sizes) if n == 0]


def empty_folds_allowed(rep: dict) -> bool:
    """Lane-wide convention for rowsel-split-v2-groups (shared with the clean-route builder): the split always uses
    OUTER_FOLDS folds; with c < OUTER_FOLDS components the greedy fill puts one component in each of folds 0..c-1 and
    leaves folds c..OUTER_FOLDS-1 empty. Empty folds are allowed only then: a v2g report (it has 'components'), fewer
    components than folds, and exactly the trailing folds c.. empty. Split v2 (no 'components') never allows one."""
    c, sizes = rep.get("components"), rep["fold_sizes"]
    return c is not None and 2 <= c < len(sizes) and empty_fold_list(sizes) == list(range(c, len(sizes)))


def split_gate_failures(rep: dict) -> list:
    out = []
    if rep["groups_straddling_folds"]:
        out.append("groups_straddle_folds")
    if not rep["target_check"]["target_invariant"]:
        out.append("split_reads_target")
    if not (rep["value_check_fold0"]["valid"] and rep["value_check_fold0"]["invariant"]):
        out.append("split_moves_under_outer_test_values")
    if min(rep["fold_sizes"]) == 0 and not empty_folds_allowed(rep):
        out.append("empty_outer_fold")
    if rep.get("annotation_groups_straddling_folds"):
        out.append("annotation_groups_straddle_folds")
    if "group_value_check_fold0" in rep and not (rep["group_value_check_fold0"]["valid"] and rep["group_value_check_fold0"]["invariant"]):
        out.append("split_moves_under_group_preserving_changes")
    return out


def new_view_id_v2(h: str, fold: int, view_identity: str, split_sha256: str, tag: str = VIEW_TAG_V2) -> str:
    return f"{h}__r0__f{int(fold)}__{tag}_{sha(rowsel._canon([view_identity, split_sha256]))[:12]}"


def cut_block(split_sha256: str, ds: dict, epoch0_rows: list) -> dict:
    return {"cut_id": CUT_PREFIX + split_sha256, "schema": SPLIT_V2, "comparable_with_replaced": False,
            "replaces_cut": {"kind": "registry-split", "split_sha256": ds["split"]["sha256"], "version_id": ds["split"]["version_id"],
                             "schema": ds["split"]["schema"], "method": ds["split"]["method"], "repeat": 0,
                             "epoch0_contexts": [{"view_id": r["view_id"], "context_sha256": r["context_sha256"], "fold": int(r["fold"])}
                                                 for r in epoch0_rows],
                             "note": "epoch-0 records carry no explicit cut_id; their outer cut is this registry split object"},
            "rule": "cut_id = 'trainonly-split-v2:' + rowsel.split.sha256; GP-teacher HWM comparisons use this cut_id only"}


def census_index(path=CENSUS, digest=CENSUS_SHA256) -> dict:
    """dataset_hash -> how the registry split grouped rows (split-scope census): annotation-group feature columns that
    reproduce the split, and full-row duplicate components. Informational: v2 groups by exact feature rows only."""
    body = read_verified(Path(path), digest)
    out = {}
    for l in body.decode().splitlines():
        if l.strip():
            r = json.loads(l)
            out[r["dataset_hash"]] = {"registry_group_columns": r.get("reproducing_group_columns") or [],
                                      "registry_grouped_by_feature_column": not r.get("match_group_none", True),
                                      "registry_full_row_duplicate_components": r.get("dup_multi_components"),
                                      "registry_rows_in_full_row_duplicates": r.get("rows_in_dup_multi")}
    return out


def fallback_datasets(path=FALLBACK_SOURCE, digest=FALLBACK_SOURCE_SHA256, raw_plan=RAW_V2_PLAN) -> dict:
    """The incoming acquisition-capped kmeans datasets whose raw source did not match (raw v2 route summary): built here
    on T1 with the acquisition_cap label. None may be a dataset the raw v2 tree rebuilt."""
    doc = json.loads(read_verified(Path(path), digest))
    nb = {d["dataset_hash"]: d for d in doc["buckets"]["incoming_acquisition_capped_kmeans"]["datasets_not_rebuilt"]}
    rebuilt = {e["dataset_hash"] for e in json.loads(Path(raw_plan).read_bytes())["datasets"]}
    clash = sorted(set(nb) & rebuilt)
    if clash:
        raise RebuildError("fallback_dataset_rebuilt_on_raw", ",".join(clash))
    return {"datasets": nb, "ref": {"path": str(path), "sha256": digest, "raw_v2_plan": file_ref(Path(raw_plan))}}


def load_tree_scope(rows, tree: str, fallback_record=None) -> tuple:
    """(dataset_hash -> [(epoch-0 row, label, acquisition record)], refs, counts) of one v2split tree. The drifted T1
    fallback tree (an explicit opt-in) needs the t1_fallback record; every other tree ignores it."""
    t = TREES[tree]
    if t.get("fallback_kind") == "drifted":
        selected, refs, _ = load_scope(rows, scope=t["scope"], fallback_record=fallback_record)
    else:
        selected, refs, _ = load_scope(rows, scope=t["scope"])
    sel = [it for it in selected if it[1] == t["label"]]
    if t["fallback"] and t.get("fallback_kind") != "drifted":
        fb = fallback_datasets()
        sel = [it for it in sel if it[0]["dataset_hash"] in fb["datasets"]]
        refs["t1_fallback_source"] = fb["ref"]
        refs["t1_fallback_reasons"] = {h: {k: d.get(k) for k in ("reason", "detail")} for h, d in fb["datasets"].items()}
    groups = collections.OrderedDict()
    for it in sel:
        groups.setdefault(it[0]["dataset_hash"], []).append(it)
    counts = {"datasets": len(groups), "epoch0_contexts": len(sel)}
    if counts != t["expected"]:
        raise RebuildError("tree_scope_counts", f"{tree}: {counts} != {t['expected']}")
    if t["fallback"] and set(groups) != set(refs["t1_fallback_reasons"]):
        raise RebuildError("fallback_scope", "the fallback list and the scope differ")
    return groups, refs, counts


def v1_cost(tree: str) -> dict:
    """dataset_hash -> measured v1 CPU (context build_cpu_s per epoch-0 context, scaled to 5 folds, plus the lineage fit)."""
    t = TREES[tree]
    per, n = collections.Counter(), collections.Counter()
    for f in glob.glob(str(t["v1_tree"] / "shard-*" / "*" / "context.json")):
        c = json.loads(Path(f).read_bytes())
        per[c["dataset_hash"]] += c.get("build_cpu_s", 0.0)
        n[c["dataset_hash"]] += 1
    port = {}
    for f in glob.glob(str(t["v1_tree"] / "_datasets" / "*.json")):
        d = json.loads(Path(f).read_bytes())
        port[d["dataset_hash"]] = d.get("port_fit_cpu_s", 0.0)
    return {h: per[h] * 5.0 / max(n[h], 1) + port.get(h, 0.0) for h in per}


def plan_v2(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = scale_check.Sources(load_index=False)
    groups, refs, counts = load_tree_scope(src.included(), a.tree, getattr(a, "t1_fallback_record", None))
    census = census_index()
    cost = v1_cost(a.tree)
    t = TREES[a.tree]
    entries = []
    for h, its in groups.items():
        e = {"dataset_hash": h, "label": t["label"], "cohort": its[0][0]["cohort"], "acquisition": its[0][2],
             "epoch0": [{"view_id": r["view_id"], "context_sha256": r["context_sha256"], "fold": int(r["fold"])} for r, _, _ in its],
             "est_cpu_s": round(cost.get(h, 60.0), 1), "registry_grouping": census.get(h)}
        if t["fallback"] and t.get("fallback_kind") == "drifted":
            e["t1_fallback"] = dict(refs["t1_fallback_reasons"][h], rule="DRIFTED-T1-FALLBACK: raw content match fails against "
                                    "today's live source; rebuilt on the frozen T1 (acquisition-capped) data keeping acquisition_cap, "
                                    "research-only, split v2, train-only", source=refs["t1_fallback_record"]["path"],
                                    source_sha256=refs["t1_fallback_record"]["sha256"])
            if (census.get(h) or {}).get("registry_grouped_by_feature_column"):
                e.update(disposition="not_built", reason="registry_group_annotated_awaiting_v2g_decision",
                         detail="the registry split grouped rows by a feature column (GROUP-UNION-B); on T1 the group maps exactly, so "
                                "it belongs on rowsel-split-v2-groups (or plain v2 flagged group_annotation_dropped): owner/coordinator call")
        elif t["fallback"]:
            e["t1_fallback"] = dict(refs["t1_fallback_reasons"][h], rule="ROWSEL-SPLIT: raw source unmatched; rebuilt on T1 "
                                    "keeping acquisition_cap", source=refs["t1_fallback_source"]["path"],
                                    source_sha256=refs["t1_fallback_source"]["sha256"])
        entries.append(e)
    loads = [0.0] * a.of
    for e in sorted((x for x in entries if x.get("disposition", "build") == "build"),
                    key=lambda e: (-e["est_cpu_s"], e["dataset_hash"])):     # LPT, deterministic
        i = min(range(a.of), key=lambda k: (loads[k], k))
        e["shard"] = i
        loads[i] += e["est_cpu_s"]
    entries.sort(key=lambda e: e["dataset_hash"])
    doc = {"schema": PLAN_SCHEMA_V2, "tree": a.tree, "split_version": SPLIT_V2, "of": a.of, "datasets": entries, "counts": counts,
           "expected_new_contexts": rowsel_raw.OUTER_FOLDS * sum(1 for x in entries if x.get("disposition", "build") == "build"),
           "shard_est_cpu_s": [round(x, 1) for x in loads],
           "inputs": {k: v for k, v in refs.items() if k != "t1_fallback_reasons"} | {"census": {"path": str(CENSUS), "sha256": CENSUS_SHA256},
                      "owner_decision": file_ref(NONRAW_DECISION), "epoch0": {"path": str(EPOCH0), "sha256": EPOCH0_SHA256}},
           "owner_line": SCOPES[t["scope"]]["owner_line"] + "+NONRAW-SPLIT-A", "budget": rowsel.DEFAULT_BUDGET}
    p = out / "plan.json"
    body = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    if p.exists() and p.read_bytes() != body:
        raise SystemExit(f"{p} exists with different content; refusing to replace a pinned plan")
    if not p.exists():
        atomic(p, body)
    print(json.dumps({"plan": str(p), "sha256": sha(body), "counts": counts, "shard_est_cpu_s": doc["shard_est_cpu_s"]}))
    return 0


# ------------------------------------------------------------------------------------------------- split v2 groups (GROUP-UNION-B)
#
# --split-version rowsel-split-v2-groups: split v2 amended so annotated groups are kept together (owner ruling GROUP-UNION-B):
# components = exact-duplicate feature rows UNION rows with an equal value in the dataset's group column; ranking and
# placement as v2 (dltrain/rowsel_split_v2_groups.py, shared with the clean-route builder). Only the datasets whose
# registry split grouped rows by one feature column are rebuilt, into new trees.
#   group column  the registry annotation 'group' is not in any local pinned object (the split objects pin only the
#                 train/test blobs and the T1 meta), so the column is the single feature column that, with the registry
#                 split port (RUN/tmp/split-scope/gsk.py, sha256-pinned), reproduces EVERY repeat and fold of the registry
#                 split object. Several such columns are accepted only if each gives the identical component structure;
#                 the lowest index is then the spec. Group value = the column's float64 value (canonical bytes; every
#                 missing cell one group, as the registry's equal empty fields are); the plan checks that this grouping is
#                 the registry's field-bytes grouping on the universe, else the dataset is refused.
#   identity      split sha = rowsel_split_v2_groups.split_sha256(h, 't1', {index, name}, folds); cut_id 'trainonly-split-v2g:'+sha
#   replaces      the registry split AND the as-built v2split cut of the dataset (both listed in cut.replaces_cut)

CUT_PREFIX_V2G = rowsel_split_v2_groups.CUT_PREFIX
GSK = RUN / "tmp" / "split-scope" / "gsk.py"
GSK_SHA256 = "83183db43a42740c3929c6de302e9633ac2c2731fdad143a9dac5e020027c599"
PLAN_SCHEMA_V2G = "jope.dl-v2-rowsel-v2gsplit-plan/v1"
TREES_V2G = {
    "kmeans-clean-groups": {"base_tree": "kmeans-clean", "out": RUN / "build-v2" / "ROWSEL-T1-kmeans-v2gsplit"},
    "kmeans-fallback-groups": {"base_tree": "kmeans-fallback", "out": RUN / "build-v2" / "ROWSEL-T1-fallback-v2gsplit"},
    "frozen-kmeans-research-groups": {"base_tree": "frozen-kmeans-research", "out": RUN / "build-v2" / "ROWSEL-T1-frozen-kmeans-research-v2gsplit"},
}

PROFILE_V2 = {"version": SPLIT_V2, "split_fn": split_v2_t1, "sha_fn": split_v2_t1_sha256, "rule": rowsel_raw_split_v2.RULE,
              "view_tag": VIEW_TAG_V2, "cut_fn": None, "group_column": None}


MISSING_GROUP = b"\x00missing-cell"       # NOT 8 bytes: never read as a '<f8' group value (target_tied_group's affine test).
                                         # Only its equality matters to the split: 6e1e166e/fcfe3828 used b"\x00missing"
                                         # (8 bytes), which gives the identical partition, components and split.


def t1_group_values(values, index: int) -> list:
    """Group value per row = canonical float64 bytes of feature column `index` (-0.0 == 0.0); every missing (NaN) cell
    is the one group MISSING_GROUP, as the registry port joins rows whose group fields are equal empty strings."""
    col = np.asarray(values, dtype=np.float64)[:, int(index)]
    return [MISSING_GROUP if np.isnan(x) else np.float64(x + 0.0).tobytes() for x in col.tolist()]


def group_partition(group_values) -> list:
    """The partition of row numbers by group value alone (canonical): what two group-value rules must agree on."""
    return rowsel_split_v2_groups.component_partition(list(range(len(group_values))),
                                                      [str(i).encode() for i in range(len(group_values))], list(group_values))


def group_value_check(split_fn, prep, index: int, fold: int = 0) -> dict:
    """As rowsel_raw_split_v2.value_check, with validity = the COMPONENT structure (feature duplicates UNION groups) of
    the whole universe unchanged: one outer-test fold's features (the group column included: remapped injectively to
    fresh values), text and targets changed; the folds must not move."""
    h, pos = prep.dataset_hash, prep.positions
    base = split_fn(h, pos, prep.values, prep.specs)
    rows = prep.rows_of(base[fold])
    part0 = rowsel_split_v2_groups.component_partition(pos.tolist(), rowsel_raw.feature_bytes(prep.values, prep.specs),
                                                       t1_group_values(prep.values, index))
    fold_of = {p: f for f, v in base.items() for p in v}
    cases = {}
    for kind in rowsel_raw_split_v2.CF_KINDS:
        V2, s2 = rowsel_raw_split_v2.perturb_fold_values(prep.values, prep.specs, rows, kind)
        part = rowsel_split_v2_groups.component_partition(pos.tolist(), rowsel_raw.feature_bytes(V2, s2), t1_group_values(V2, index))
        got = split_fn(h, pos, V2, s2)
        moved = sum(1 for f, v in got.items() for p in v if fold_of[p] != f)
        cases[kind] = {"valid": part == part0, "split_identical": moved == 0, "rows_moved": int(moved)}
    return {"fold": int(fold), "cases": cases, "valid": all(c["valid"] for c in cases.values()),
            "invariant": all(c["valid"] and c["split_identical"] for c in cases.values())}


def cut_block_v2g(split_rep: dict, ds: dict, epoch0_rows: list, spec: dict, replaced_v2split: dict) -> dict:
    split_sha256 = split_rep["sha256"]
    return {"cut_id": CUT_PREFIX_V2G + split_sha256, "schema": SPLIT_V2G, "comparable_with_replaced": False, "group_column": spec,
            "folds": len(split_rep["fold_sizes"]), "empty_folds": split_rep["empty_folds"], "effective_folds": split_rep["effective_folds"],
            "empty_folds_rule": "the split always uses 5 folds; with c < 5 components folds c..4 stay empty and get no context; "
                                "cut_id binds the 5-fold assignment including the empty folds",
            "replaces_cut": [
                {"kind": "registry-split", "split_sha256": ds["split"]["sha256"], "version_id": ds["split"]["version_id"],
                 "schema": ds["split"]["schema"], "method": ds["split"]["method"], "repeat": 0,
                 "epoch0_contexts": [{"view_id": r["view_id"], "context_sha256": r["context_sha256"], "fold": int(r["fold"])} for r in epoch0_rows],
                 "note": "epoch-0 records carry no explicit cut_id; their outer cut is this registry split object"},
                dict(replaced_v2split, kind="rowsel-raw-split-v2 (as built, superseded)")],
            "rule": "cut_id = 'trainonly-split-v2g:' + rowsel.split.sha256; GP-teacher HWM comparisons use this cut_id only and are "
                    "never compared with any replaced cut"}


def profile_v2g(index: int, name: str, replaced_v2split: dict, expected_empty_folds=None) -> dict:
    spec = rowsel_split_v2_groups.group_spec(index, name)

    def split_fn(h, positions, values, specs, folds=rowsel_raw.OUTER_FOLDS):
        if int(index) >= len(specs) - 1:
            raise RebuildError("group_column_is_not_a_feature", f"{index} of {len(specs) - 1} features")
        return rowsel_split_v2_groups.split(h, np.asarray(positions).tolist(), rowsel_raw.feature_bytes(values, specs),
                                            t1_group_values(values, index), T1_PARTITION, folds)

    def group_report(prep, folds):
        gv = t1_group_values(prep.values, index)
        comps = rowsel_split_v2_groups.components(prep.positions.tolist(), rowsel_raw.feature_bytes(prep.values, prep.specs), gv)
        return {"group_column": spec, "components": len(comps), "largest_component": max(len(c) for c in comps),
                "annotation_groups": len({x for x in gv if x is not None}),
                "annotation_groups_straddling_folds": rowsel_split_v2_groups.straddling_groups(folds, prep.positions.tolist(), gv)}

    def fold_target_tie(prep, folds, f):
        """Target-tie admission of context f (PR #246 review 5437848458 fold isolation): the group column must not copy or
        determine the target on fold f's OWN outer-train rows (every universe row not in fold f); no outer-test row and no
        other fold's decision enters. A tie refuses this context only; the split itself never receives y."""
        if int(index) >= len(prep.specs) - 1:
            raise RebuildError("group_column_is_not_a_feature", f"{index} of {len(prep.specs) - 1} features")
        test = {int(p) for p in folds[f]}
        rows = prep.rows_of([p for p in prep.positions.tolist() if p not in test])
        V = prep.values[rows]
        return rowsel_split_v2_groups.target_tied_group(t1_group_values(V, index), V[:, -1])

    return {"version": SPLIT_V2G, "split_fn": split_fn, "fold_target_tie_fn": fold_target_tie,
            "sha_fn": lambda h, folds: rowsel_split_v2_groups.split_sha256(h, T1_PARTITION, spec, folds),
            "rule": rowsel_split_v2_groups.RULE, "view_tag": VIEW_TAG_V2G, "group_column": spec, "group_report": group_report,
            "group_value_check": lambda fn, prep, fold: group_value_check(fn, prep, index, fold),
            "cut_fn": lambda split_rep, ds, rows: cut_block_v2g(split_rep, ds, rows, spec, replaced_v2split),
            "expected_empty_folds": None if expected_empty_folds is None else [int(f) for f in expected_empty_folds],
            "e0_note": "none: a new target-free outer split with annotated groups (GROUP-UNION-B); new fold f is not registry fold f "
                       "nor v2split fold f"}


def load_gsk():
    body = read_verified(GSK, GSK_SHA256)
    spec = importlib.util.spec_from_loader("gsk_registry_port", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(body, str(GSK), "exec"), mod.__dict__)
    return mod


def registry_group_evidence(h: str, U: rowsel.Universe, split_body: bytes, train_body: bytes, test_body: bytes, candidates: list) -> dict:
    """Which candidate group columns reproduce EVERY repeat/fold of the registry split object (registry split port, as
    the census ran it: its own line split and field parser, base seed = the split's first fold seed), and whether all
    reproducing candidates give the identical v2g component structure on this universe.
    Fold isolation (PR #246 review 5437848458): the census reads features only; the target-tie guard is not decided here
    but per context at build, over that context's own outer-train rows (TIE_PER_CONTEXT)."""
    gsk = load_gsk()
    split = json.loads(split_body)
    if split["source"]["dataset_hash"] != h or (split["source"]["train_sha256"], split["source"]["test_sha256"]) != (sha(train_body), sha(test_body)):
        raise RebuildError("split_source", "registry split object does not bind these T1 blobs")
    rows = [gsk.csv_fields(l) for body in (train_body, test_body) for l in gsk.csv_lines(body)]
    if len(rows) != U.n:
        raise RebuildError("port_line_split", f"port rows {len(rows)} != universe rows {U.n}")
    base_seed = int(split["folds"][0]["seed"])
    keys = [gsk.dup_key(r) for r in rows]
    want = split["folds"]
    fb = rowsel_raw.feature_bytes(np.column_stack([U.X, U.y]), t1_split_inputs(U)[1])
    pos = U.positions.tolist()
    per = {}
    for c in candidates:
        if int(c) >= U.features:
            per[str(c)] = {"reproduces_registry": False, "reason": "not a feature column"}
            continue
        k, got, _ = gsk.generate_folds(keys, base_seed, groups=[r[int(c)] for r in rows])
        gv = t1_group_values(np.column_stack([U.X, U.y]), int(c))
        row = {"reproduces_registry": k == int(split["effective_folds"]) and got == want, "effective_folds": k,
               "missing_cells": sum(1 for x in gv if x == MISSING_GROUP), "distinct_values": len(set(gv)),
               "target_tie": TIE_PER_CONTEXT}                            # never decided here: no y is read (fold isolation)
        comps = rowsel_split_v2_groups.component_partition(pos, fb, gv)
        row.update(field_bytes_equals_value_grouping=group_partition([r[int(c)] for r in rows]) == group_partition(gv),
                   v2g_components=len(comps), component_structure_sha256=sha(json.dumps(comps).encode()))
        per[str(c)] = row
    repro = [int(c) for c in candidates if per[str(c)].get("reproduces_registry")]
    unstructured = [c for c in repro if per[str(c)].get("component_structure_sha256") is None]
    structures = {per[str(c)]["component_structure_sha256"] for c in repro}
    return {"candidates": [int(c) for c in candidates], "per_candidate": per, "reproducing": repro,
            "identical_components_across_reproducing": None if unstructured else len(structures) == 1,
            **({"components_not_computed_for": unstructured} if unstructured else {}), "base_seed": base_seed,
            "port": {"path": str(GSK), "sha256": GSK_SHA256}, "registry_folds_compared": len(want)}


TIE_NOT_COMPUTED = "target_tied_group refused before partition"
TIE_PER_CONTEXT = ("target_tied_group runs per context at build, over that context's own outer-train rows only "
                   "(profile fold_target_tie_fn); the census reads features only")
GROUP_SOURCE_A = ("registry annotation not available in local pinned objects (the split objects pin only the train/test "
                  "blobs and the T1 meta; no annotations reference); nothing fetched")


def replacement_membership(built_folds, admitted, refusals) -> dict:
    """The replacement record's fold membership after context-specific refusals (FOLD-1; PR #251 review 5441124504):
    new_contexts counts the WRITTEN contexts and the record names the built folds, the admitted (written) folds and each
    refused fold with its own reason, so the manifest publishes exactly the admitted set and a refused fold never blocks
    the others. Set only when a fold is refused; without refusals the record is unchanged (new_contexts = every built fold)."""
    return {"new_contexts": len(admitted), "built_folds": sorted(int(f) for f in built_folds),
            "admitted_folds": sorted(int(f) for f in admitted),
            "refused_folds": sorted(({"fold": int(c["fold"]), "reason": c["refused"]} for c in refusals), key=lambda x: x["fold"])}


def context_refusal(tlg) -> str | None:
    """The refusal reason of ONE context from its own LINEAR-DETERMINED report (outer-train rows only), else None. Never
    aggregated across folds (FOLD-1, owner 07:22Z; PR #246 review 5437848458)."""
    return target_linear.REFUSE if (tlg or {}).get("verdict") == target_linear.REFUSE else None


def group_disposition(ev: dict, names: list) -> dict:
    """Build only with exactly one group structure: some column reproduces the registry split, every reproducing column
    gives the identical v2g component structure (then the lowest index is the spec), its value grouping is the
    registry's field-bytes grouping, and it has at least 2 components. With c < OUTER_FOLDS components the 5-fold split
    leaves folds c.. empty (empty_folds_allowed); the entry pins them. Otherwise not built, with the reason.
    FOLD-1 (owner 07:22Z, supersedes #218 finding 8): planning reads features only; the target-tie guard is never a
    dataset-level disposition but runs per context at build, over that context's own outer-train rows."""
    if not ev["reproducing"]:
        return {"disposition": "not_built", "reason": "no_group_column_reproduces_registry"}
    if ev["identical_components_across_reproducing"] is not True:       # None (no structure) is never identical
        return {"disposition": "not_built", "reason": "ambiguous_group_columns_with_different_components"}
    if not all(ev["per_candidate"][str(c)]["field_bytes_equals_value_grouping"] for c in ev["reproducing"]):
        return {"disposition": "not_built", "reason": "group_value_grouping_differs_from_registry_field_grouping"}
    c = min(ev["reproducing"])
    k = ev["per_candidate"][str(c)]["v2g_components"]
    if k < 2:
        return {"disposition": "not_built", "reason": "fewer_than_two_v2g_components",
                "detail": f"{k} component: no outer fold can have training rows", "candidate_group_column": {"index": c, "name": names[c]}}
    empty = list(range(k, rowsel_raw.OUTER_FOLDS)) if k < rowsel_raw.OUTER_FOLDS else []
    return {"disposition": "build", "group_column": {"index": c, "name": names[c]},
            "empty_folds": empty, "effective_folds": rowsel_raw.OUTER_FOLDS - len(empty),
            "group_source": "inferred: the single feature column whose registry-port grouping reproduces every repeat and fold of "
                            "the registry split object" + (" (ambiguous: every reproducing column gives the identical component "
                                                          "structure; lowest index taken)" if len(ev["reproducing"]) > 1 else ""),
            "group_source_a": GROUP_SOURCE_A}


def plan_v2g(a) -> int:
    """The v2gsplit plan: the base v2split tree's datasets whose registry split grouped rows by a feature column; per
    dataset the group column with its evidence (or a refusal reason) and the as-built v2split cut it replaces."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = TREES_V2G[a.tree]
    base_out = TREES[cfg["base_tree"]]["out"]
    base_plan = json.loads((base_out / "plan.json").read_bytes())
    inv = json.loads(DATASETS_INV.read_text())
    census = {}
    for l in read_verified(CENSUS, CENSUS_SHA256).decode().splitlines():
        if l.strip():
            r = json.loads(l)
            census[r["dataset_hash"]] = r
    entries = []
    for e in base_plan["datasets"]:
        if not (e.get("registry_grouping") or {}).get("registry_grouped_by_feature_column"):
            continue
        h = e["dataset_hash"]
        rp = base_out / f"shard-{e['shard']}" / "_datasets" / f"{h}.json"
        r = json.loads(rp.read_bytes())
        if r.get("status") != "done":
            raise SystemExit(f"{h}: the base v2split dataset is not done")
        c0 = json.loads((base_out / f"shard-{e['shard']}" / r["contexts"][0]["view_id"] / "context.json").read_bytes())
        uni = c0["rowsel"]["universe"]
        blob = Path(r["lineage"]["t1_blob_dir"])
        tr = read_verified(blob / uni["train"]["sha256"], uni["train"]["sha256"])
        te = read_verified(blob / uni["test"]["sha256"], uni["test"]["sha256"])
        U = t1_universe(h, tr, te, uni["features"] + 1)
        split_body = read_verified(Path(r["registry_split_replaced"]["custody"]["path"]), r["registry_split_replaced"]["sha256"])
        cands = list(census[h].get("reproducing_group_columns") or [])
        ev = registry_group_evidence(h, U, split_body, tr, te, cands)
        mk = r["lineage"]["t1_meta_sha256"]
        if mk != inv["meta_key"][h].rsplit("/", 1)[1]:
            raise SystemExit(f"{h}: v2split record meta {mk} != inventory meta")
        meta = json.loads(read_verified(META_DIR / f"{mk}.json", mk))
        names = meta["output_headers"]
        ev["census"] = {"path": str(CENSUS), "sha256": CENSUS_SHA256, "reproducing_group_columns": cands,
                        "group_column_candidates": census[h].get("group_column_candidates")}
        ev["names"] = {str(c): names[c] for c in cands if c < len(names)}
        entry = {k: e[k] for k in ("dataset_hash", "label", "cohort", "acquisition", "epoch0", "registry_grouping") if k in e}
        if e.get("t1_fallback"):
            entry["t1_fallback"] = e["t1_fallback"]
        entry["est_cpu_s"] = round(float(r.get("cpu_s") or e.get("est_cpu_s") or 60.0), 1)
        entry["group_evidence"] = ev
        entry["replaces_v2split"] = {"tree": str(base_out), "cut_id": r["cut"]["cut_id"], "split_sha256": r["split"]["sha256"],
                                     "dataset_record": {"path": str(rp), "sha256": sha_file(rp)},
                                     "view_ids": [c["view_id"] for c in r["contexts"]],
                                     "context_sha256s": [c["context_sha256"] for c in r["contexts"]]}
        entry.update(group_disposition(ev, names))
        entries.append(entry)
    loads = [0.0] * a.of
    for e in sorted((x for x in entries if x["disposition"] == "build"), key=lambda e: (-e["est_cpu_s"], e["dataset_hash"])):
        i = min(range(a.of), key=lambda k: (loads[k], k))
        e["shard"] = i
        loads[i] += e["est_cpu_s"]
    entries.sort(key=lambda e: e["dataset_hash"])
    built = [e for e in entries if e["disposition"] == "build"]
    doc = {"schema": PLAN_SCHEMA_V2G, "tree": a.tree, "base_tree": cfg["base_tree"], "split_version": SPLIT_V2G, "of": a.of,
           "datasets": entries,
           "counts": {"datasets": len(entries), "epoch0_contexts": sum(len(e["epoch0"]) for e in entries),
                      "build_datasets": len(built), "build_epoch0_contexts": sum(len(e["epoch0"]) for e in built),
                      "replaced_v2split_contexts": sum(len(e["replaces_v2split"]["view_ids"]) for e in built),
                      "not_built": dict(collections.Counter(e["reason"] for e in entries if e["disposition"] != "build"))},
           "expected_new_contexts": rowsel_raw.OUTER_FOLDS * len(built), "shard_est_cpu_s": [round(x, 1) for x in loads],
           "inputs": base_plan["inputs"] | {"base_plan": file_ref(base_out / "plan.json"), "registry_port": {"path": str(GSK), "sha256": GSK_SHA256}},
           "owner_line": base_plan["owner_line"] + "+GROUP-UNION-B", "budget": rowsel.DEFAULT_BUDGET,
           "owner_decision_note": "GROUP-UNION-B (ENG:watcher, delegated, relayed by the coordinator); not yet in claude/receipts at plan time"}
    p = out / "plan.json"
    body = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    if p.exists() and p.read_bytes() != body:
        raise SystemExit(f"{p} exists with different content; refusing to replace a pinned plan")
    if not p.exists():
        atomic(p, body)
    print(json.dumps({"plan": str(p), "sha256": sha(body), "counts": doc["counts"], "shard_est_cpu_s": doc["shard_est_cpu_s"]}))
    return 0


SUPPLEMENT_EMPTY_FOLDS = "plan-supplement-empty-folds.json"
OWNER_DECISIONS_V2G = Path("/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261006T062557Z.json")
OWNER_RELAY_V2G = RUN / "receipts" / "owner-relays" / "DELEGATED-RELAY-group-union-b-20261006T062557Z.txt"
EMPTY_FOLDS_DECISION = ("coordinator decision 2026-10-06 (amended the same day for lane-wide consistency with the clean-route builder): "
                        "group datasets with fewer than 5 v2g components are built on the unchanged 5-fold split; the greedy fill "
                        "leaves the trailing folds empty; split.empty_folds and split.effective_folds are recorded; a context is built "
                        "only for each non-empty fold; cut_id binds the 5-fold assignment including the empty folds. Reported to "
                        "the owner as an implementation decision under GROUP-UNION-B")


def owner_ruling_v2g() -> dict:
    body = OWNER_DECISIONS_V2G.read_bytes()
    return {"id": "GROUP-UNION-B", "owner_decisions": file_ref(OWNER_DECISIONS_V2G),
            "key_present": "GROUP-UNION-B" in (json.loads(body).get("decisions") or {}), "relay": file_ref(OWNER_RELAY_V2G),
            "recorded_in_owner_decisions": True}


def plan_v2g_supplement(a) -> int:
    """A create-only supplement to a v2gsplit tree's pinned plan.json: the datasets plan.json refused for having fewer
    than 5 components, now planned on the 5-fold split with trailing empty folds (EMPTY_FOLDS_DECISION). Their shards
    are tagged 'e' (shard-e0); summarize_v2 merges the supplement. Refused once the tree is .complete."""
    out = Path(a.out)
    base_p = out / "plan.json"
    base = json.loads(base_p.read_bytes())
    if base.get("split_version") != SPLIT_V2G or base.get("tree") != a.tree:
        raise SystemExit(f"{base_p} is not the {a.tree} v2gsplit plan")
    if (out / ".complete").exists():
        raise SystemExit(f"{out} is complete; a supplement goes into an additional tree")
    entries = []
    for e in base["datasets"]:
        if e.get("disposition") == "build" or e.get("reason") != "fewer_v2g_components_than_outer_folds":
            continue
        ev = e["group_evidence"]
        n = {k: v for k, v in e.items() if k not in ("disposition", "reason", "detail", "candidate_group_column")}
        n.update(group_disposition(ev, {int(k): v for k, v in ev["names"].items()}))
        n["supersedes_plan_disposition"] = {k: e.get(k) for k in ("disposition", "reason", "detail")}
        if n["disposition"] == "build":
            n["shard"] = 0
        entries.append(n)
    built = [e for e in entries if e["disposition"] == "build"]
    doc = {"schema": PLAN_SCHEMA_V2G, "kind": "supplement", "tree": a.tree, "base_tree": base["base_tree"], "split_version": SPLIT_V2G,
           "of": 1, "shard_tag": "e", "supplements": file_ref(base_p), "datasets": entries,
           "counts": {"datasets": len(entries), "build_datasets": len(built), "build_epoch0_contexts": sum(len(e["epoch0"]) for e in built),
                      "replaced_v2split_contexts": sum(len(e["replaces_v2split"]["view_ids"]) for e in built),
                      "not_built": dict(collections.Counter(e["reason"] for e in entries if e["disposition"] != "build"))},
           "expected_new_contexts": sum(e["effective_folds"] for e in built),
           "shard_est_cpu_s": [round(sum(e["est_cpu_s"] for e in built), 1)],
           "inputs": base["inputs"], "owner_line": base["owner_line"], "budget": base["budget"],
           "decision": {"text": EMPTY_FOLDS_DECISION, "owner_ruling": owner_ruling_v2g()}}
    p = out / SUPPLEMENT_EMPTY_FOLDS
    body = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    if p.exists() and p.read_bytes() != body:
        raise SystemExit(f"{p} exists with different content; refusing to replace a pinned plan")
    if not p.exists():
        atomic(p, body)
    print(json.dumps({"plan": str(p), "sha256": sha(body), "counts": doc["counts"], "expected_new_contexts": doc["expected_new_contexts"]}))
    return 0


def tree_cfg(tree: str) -> dict:
    """(version, out, scope tree) of a v2split or v2gsplit tree."""
    if tree in TREES:
        return {"version": SPLIT_V2, "out": TREES[tree]["out"], "scope_tree": tree}
    return {"version": SPLIT_V2G, "out": TREES_V2G[tree]["out"], "scope_tree": TREES_V2G[tree]["base_tree"]}


def entry_profile(e: dict, version: str) -> dict:
    if version == SPLIT_V2:
        return PROFILE_V2
    gc = e["group_column"]
    return profile_v2g(gc["index"], gc["name"], {k: e["replaces_v2split"][k] for k in ("cut_id", "split_sha256", "tree", "view_ids",
                                                                                      "context_sha256s", "dataset_record")},
                       expected_empty_folds=e.get("empty_folds"))


def compose_context_v2(ds, f, label, acq_rec, view, cf, dcf, built, codes, refs, budget, folds, split_rep, cut, e0, extra,
                       profile=None) -> dict:
    prof = profile or PROFILE_V2
    version = prof["version"]
    h = ds["h"]
    U, art, apos, R, T = ds["U"], built["art"], built["apos"], built["R"], built["T"]
    split = art["split"]
    vid = new_view_id_v2(h, f, view.identity, split_rep["sha256"], prof["view_tag"])
    files = {"encoder.jtt1.gz": {"sha256": sha(art["enc_gz"]), "raw_sha256": sha(split["jtt1"]), "bytes": len(art["enc_gz"])},
             "teacher.jtt1.gz": {"sha256": sha(art["tea_gz"]), "raw_sha256": sha(art["teacher"]), "bytes": len(art["tea_gz"])},
             "a0.f32": {"sha256": sha(art["a0"]), "bytes": len(art["a0"])}}
    tr_sha, te_sha = sha(json.dumps(R).encode()), sha(json.dumps(T).encode())
    apos_sha = sha(json.dumps(apos).encode())
    scaler, inner = sha(trainonly.scaler_bytes(split)), sha(np.asarray(split["inner_labels"], np.uint8).tobytes())
    identity = {"schema": IDENTITY_SCHEMA_V2, "dataset_hash": h, "fold": int(f), "repeat": 0, "view_id": vid,
                "rowsel_view_identity": view.identity, "values_mode": "L7b", "route": ROUTE,
                "universe": {"t1_train_sha256": ds["t1"]["train"]["sha256"], "t1_test_sha256": ds["t1"]["test"]["sha256"], "rows": U.n},
                "split": {"schema": version, "sha256": split_rep["sha256"], "partition": T1_PARTITION,
                          **({"group_column": prof["group_column"]} if prof.get("group_column") else {})}, "cut_id": cut["cut_id"],
                "t1_meta_sha256": ds["meta_sha256"], "a_positions_sha256": apos_sha,
                "outer_train_indices_sha256": tr_sha, "outer_test_indices_sha256": te_sha,
                "files_raw_sha256": {"encoder.jtt1": files["encoder.jtt1.gz"]["raw_sha256"],
                                     "teacher.jtt1": files["teacher.jtt1.gz"]["raw_sha256"], "a0.f32": files["a0.f32"]["sha256"]},
                "scaler_sha256": scaler, "inner_labels_sha256": inner, "epoch0_replaced": [c["view_id"] for c in e0["contexts"]]}
    fp = view.train_fit.fingerprints
    quotas_all = rowsel.fold_quotas({k: len(v) for k, v in folds.items()}, budget)
    block = {
        "route": ROUTE, "split_version": version,
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
        "split": {k: split_rep[k] for k in ("schema", "sha256", "partition", "partition_rule", "fold_sizes", "groups", "largest_group",
                                            "duplicate_rows", "groups_straddling_folds", "domain", "rule")}
                 | {"fold_size": len(folds[f]), "target_invariant": split_rep["target_check"]["target_invariant"],
                    "value_invariant": split_rep["value_check_fold0"]["invariant"]}
                 | {k: split_rep[k] for k in ("group_column", "components", "annotation_groups", "annotation_groups_straddling_folds",
                                              "empty_folds", "effective_folds") if k in split_rep}
                 | ({"group_value_invariant": split_rep["group_value_check_fold0"]["invariant"]}
                    if "group_value_check_fold0" in split_rep else {}),
        "registry_split_replaced": ds["split"],
        "registry_grouping": extra.get("registry_grouping"),
        "quotas": {"budget": int(budget), "n_train": int(view.quotas[0]), "n_test": int(view.quotas[1]),
                   "quota_met": len(view.train_positions) == view.quotas[0] and len(view.test_positions) == view.quotas[1],
                   "rule": QUOTA_RULE.replace("repeat-0 parent folds", f"{version} folds"),
                   "all_folds": {str(k): list(v) for k, v in sorted(quotas_all.items())}},
        "seed": view.seed, "seed_rule": "sha256(dataset_hash NUL 'trainonly-rowsel-v1') (rowsel.derive_seed)",
        "rowsel_py_sha256": codes["dltrain/rowsel.py"], "outer_split_changed": True,
        "a_order": "A rows = T1 lines at the ascending union of train and test positions; R/T index A",
        "a_positions_sha256": apos_sha,
        "train_positions": [int(p) for p in view.train_positions], "test_positions": [int(p) for p in view.test_positions],
        "counterfactual": cf, "downstream_counterfactual": dcf,
        "lineage": {k: ds["lineage"][k] for k in ("manifest_sha256", "membership_sha256", "data_sha256",
                                                  "membership_selected_sha256_matches_manifest", "port_reproduces_v1_selection")}
                   | {"registry_remap_matches_epoch0_folds": all(ds["lineage"]["membership_remap_matches_epoch0_folds"].values())}
                   | ({"port_reproduces_v1_selection_not_run": ds["lineage"]["port_reproduces_v1_selection_not_run"]}
                      if "port_reproduces_v1_selection_not_run" in ds["lineage"] else {}),
    }
    ctx = {"schema": SCHEMA_V2, "view_id": vid, "dataset_hash": h, "fold": int(f), "repeat": 0, "cohort": extra.get("cohort"),
           "owner_line": refs.get("owner_line", OWNER_LINE), **refs.get("context_extra", {}),
           "split_version": version, "cut": cut, "epoch0_replaced": e0,
           "context_sha256": sha(rowsel._canon(identity)), "identity": identity,
           "row_selection_leak": [label], "row_selection_leak_epoch0": ["kmeans"],
           "row_selection_leak_receipt": refs.get("row_selection_leak"),
           "acquisition": dict(acq_rec, source=refs.get("acq_selection")), "rowsel": block,
           "outer_train_indices_sha256": tr_sha, "outer_test_indices_sha256": te_sha, "n_train": len(R), "n_test": len(T),
           "files": files, "scaler_sha256": scaler, "inner_labels_sha256": inner,
           "inner_loads": split["inner_cv"]["loads"], "t1prep": art["t1prep"], "t1_meta_sha256": ds["meta_sha256"],
           "values_mode": "L7b", "published": False, "code_sha256": codes}
    if extra.get("t1_fallback"):
        ctx["t1_fallback"] = extra["t1_fallback"]
    if extra.get("group_annotation"):
        ctx["group_annotation"] = extra["group_annotation"]
    return ctx


def build_dataset_v2(ds, its, shard_dir: Path, codes, refs, *, budget=rowsel.DEFAULT_BUDGET, select_fn=rowsel.select_view,
                     split_fn=None, extra=None, profile=None) -> dict:
    """All 5 folds of one dataset on the v2 split, gated; written only when every fold passes (.done last, dataset-wide).
    Returns the dataset record (also written to shard_dir/_datasets/<h>.json)."""
    t0, c0 = time.time(), cpu_s()
    h, U = ds["h"], ds["U"]
    extra = extra or {}
    prof = profile or PROFILE_V2
    version = prof["version"]
    split_fn = prof["split_fn"] if split_fn is None else split_fn
    label, acq_rec = its[0][1], its[0][2]
    rows = [it[0] for it in its]
    extra = dict(extra, cohort=rows[0].get("cohort"))
    rec = {"schema": DATASET_SCHEMA_V2, "dataset_hash": h, "split_version": version, "label": label, "status": "failed",
           "code_sha256": codes, "lineage": ds["lineage"], "registry_split_replaced": ds["split"],
           "registry_grouping": extra.get("registry_grouping"), "published": False}
    contexts = []
    try:
        prep = t1_prep(U)
        if prof.get("fold_target_tie_fn"):                              # never dataset-level: per context, own train rows
            rec["group_target_tie"] = "per context on its own outer-train rows (fold_target_tie_fn)"
        folds = split_fn(h, prep.positions, prep.values, prep.specs)
        rep = rowsel_raw.split_report(prep, folds)
        rep.update(schema=version, sha256=prof["sha_fn"](h, folds), domain=rowsel_raw_split_v2.SPLIT_DOMAIN,
                   rule=prof["rule"], partition=T1_PARTITION, partition_rule=SPLIT_V2_PARTITION_RULE)
        rep["target_check"] = rowsel_raw.split_target_check(split_fn, prep)
        rep["value_check_fold0"] = rowsel_raw_split_v2.value_check(split_fn, prep, 0)
        if prof.get("group_column"):
            rep.update(prof["group_report"](prep, folds))
            rep["group_value_check_fold0"] = prof["group_value_check"](split_fn, prep, 0)
            rep["empty_folds"] = empty_fold_list(rep["fold_sizes"])
            rep["effective_folds"] = len(rep["fold_sizes"]) - len(rep["empty_folds"])
        rec["split"] = rep
        bad = split_gate_failures(rep)
        if bad:
            raise RebuildError("split_gate_failed", ",".join(bad))
        if prof.get("expected_empty_folds") is not None and rep["empty_folds"] != prof["expected_empty_folds"]:
            raise RebuildError("empty_folds_differ_from_plan", f"{rep['empty_folds']} != {prof['expected_empty_folds']}")
        cut = prof["cut_fn"](rep, ds, rows) if prof.get("cut_fn") else cut_block(rep["sha256"], ds, rows)
        built_folds = [f for f in sorted(folds) if len(folds[f])]           # contexts only for non-empty folds
        e0 = {"level": "dataset", "contexts": [{"view_id": r["view_id"], "context_sha256": r["context_sha256"], "fold": int(r["fold"])}
                                               for r in rows], "count": len(rows),
              "fold_correspondence": prof.get("e0_note", "none: a new target-free outer split (NONRAW-SPLIT-A); new fold f is not registry fold f"),
              "new_contexts": len(built_folds), "context_count_changed": len(rows) != len(built_folds)}
        rec.update(cut=cut, epoch0_replaced=e0)
        results = []
        for f in built_folds:
            tf = time.process_time()
            tie_rec = None
            if prof.get("fold_target_tie_fn"):                          # target-tie: fold f's OWN outer-train rows only
                tie = prof["fold_target_tie_fn"](prep, folds, f)
                if tie:
                    contexts.append({"fold": f, "refused": "group_column_tied_to_target", "target_tie": tie})
                    continue
                tie_rec = {"result": "admit", "rows": "this context's own outer-train rows", "rule": "FOLD-1",
                           "reasons_checked": list(rowsel_split_v2_groups.TARGET_TIE_REASONS)}
            view = select_fn(U, folds, f, budget, None, quotas=None)
            cf = rowsel.counterfactual_report(select_fn, U, folds, f, budget)
            failures = []
            if not (cf["train_side_invariant"] and cf["test_draw_invariant"] and cf["base_view_identity"] == view.identity):
                failures.append("counterfactual_failed")
            held = set(int(p) for p in folds[f])
            if set(view.train_positions) & held or not set(view.test_positions) <= held:
                failures.append("view_outside_parent_fold")
            ctx = None
            if not failures:
                built = assemble(U, view, ds["meta"])
                dcf = downstream_counterfactual(U, built, ds["meta"], h)
                if not dcf["train_side_invariant"]:
                    failures.append("downstream_counterfactual_failed")
                else:
                    ctx = compose_context_v2(ds, f, label, acq_rec, view, cf, dcf, built, codes, refs, budget, folds, rep, cut, e0, extra,
                                             profile=prof)
                    ctx["target_linear"] = target_linear.check_jtt1(built["art"]["split"]["jtt1"])    # outer-train rows only
                    if tie_rec is not None:
                        ctx["group_target_tie"] = tie_rec
                    ctx["build_cpu_s"] = round(time.process_time() - tf, 3)
                    results.append((ctx, built["art"]))
            contexts.append({"fold": f, "view_id": ctx["view_id"] if ctx else None, "context_sha256": ctx["context_sha256"] if ctx else None,
                             "passed": not failures, "failures": failures, "quotas": list(view.quotas),
                             "cpu_s": round(time.process_time() - tf, 2)}
                            | ({"target_linear": {k: ctx["target_linear"][k] for k in ("verdict", "ols_rel_rmse")}} if ctx else {}))
            if failures:
                raise RebuildError("fold_gate_failed", json.dumps({f: failures}))
        for c in contexts:                                               # LINEAR-DETERMINED: this context's own verdict only
            r = context_refusal(c.get("target_linear"))
            if r and not c.get("refused"):
                c["refused"] = r
        refused = [c for c in contexts if c.get("refused")]
        results = [(ctx, art) for ctx, art in results if ctx["fold"] not in {c["fold"] for c in refused}]
        if refused and not results:                                      # every context refused on its own rows
            reasons = sorted({c["refused"] for c in refused})
            raise RebuildError(reasons[0] if len(reasons) == 1 else "all_contexts_refused",
                               json.dumps({c["fold"]: (c.get("target_linear") or {}).get("ols_rel_rmse", c.get("target_tie")) for c in refused}))
        if refused:
            rec["refused_contexts"] = refused
            contexts = [c for c in contexts if not c.get("refused")]
            e0.update(replacement_membership(built_folds, [ctx["fold"] for ctx, _ in results], refused))
            e0["context_count_changed"] = len(rows) != e0["new_contexts"]
            for ctx, _ in results:                                       # every written context names the admitted set
                ctx["epoch0_replaced"] = e0
        pending = []
        for ctx, art in results:
            d = shard_dir / ctx["view_id"]
            if (d / ".done").exists():
                old = json.loads((d / "context.json").read_bytes())
                if old.get("context_sha256") != ctx["context_sha256"] or old.get("files") != ctx["files"]:
                    raise RebuildError("done_context_differs", ctx["view_id"])
                if not replacement.done_context_matches(old, ctx):
                    raise RebuildError("done_context_replacement_differs", replacement.recovery_detail(ctx["view_id"]))
                continue
            pending.append((d, ctx, art))
        for d, ctx, art in pending:                                     # phase 1: every file of every context
            d.mkdir(parents=True, exist_ok=True)
            atomic(d / "encoder.jtt1.gz", art["enc_gz"])
            atomic(d / "teacher.jtt1.gz", art["tea_gz"])
            atomic(d / "a0.f32", art["a0"])
            atomic(d / "context.json", jdump(dict(ctx, utc_us=int(time.time() * 1e6))))
        for d, _, _ in pending:                                         # phase 2: .done last, dataset-wide
            atomic(d / ".done", b"ok\n")
        rec.update(status="done", reason=None, contexts_written=len(pending), contexts_already_done=len(results) - len(pending))
    except RebuildError as e:
        rec.update(reason=e.reason, detail=e.detail[:1500])
    except Exception as e:  # recorded; the shard continues
        rec.update(reason="dataset_error", detail=f"{type(e).__name__}: {e}"[:1500])
    rec.update(contexts=contexts, cpu_s=round(cpu_s() - c0, 2), wall_s=round(time.time() - t0, 2),
               maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, utc_us=int(time.time() * 1e6))
    (shard_dir / "_datasets").mkdir(parents=True, exist_ok=True)
    atomic(shard_dir / "_datasets" / f"{h}.json", jdump(rec))
    return rec


def dataset_done_v2(shard_dir: Path, h: str, codes, version: str = None) -> bool:
    p = shard_dir / "_datasets" / f"{h}.json"
    if not p.exists():
        return False
    r = json.loads(p.read_bytes())
    return (r.get("status") == "done" and r.get("code_sha256") == codes and r.get("split_version") == (version or SPLIT_V2)
            and replacement.dataset_done_matches(shard_dir, r))


def build_v2(a) -> int:
    cpuguard.set_threads(2)
    out = Path(a.out)
    plan_path = out / (getattr(a, "plan_file", None) or "plan.json")
    plan = json.loads(plan_path.read_bytes())
    plan_sha = sha_file(plan_path)
    if plan.get("kind") == "supplement":
        if plan["supplements"]["sha256"] != sha_file(out / "plan.json"):
            raise SystemExit(f"{plan_path} supplements a different plan.json")
        if (out / ".complete").exists():
            raise SystemExit(f"{out} is complete; a supplement goes into an additional tree")
    tag = plan.get("shard_tag", "")
    tc = tree_cfg(a.tree)
    version = tc["version"]
    if plan.get("split_version") != version or plan["tree"] != a.tree or plan["of"] != a.of or a.split_version != version:
        raise SystemExit(f"plan {plan.get('tree')}/{plan.get('split_version')}/of {plan.get('of')} != {a.tree}/{version}/of {a.of}")
    codes = code_shas()
    src = scale_check.Sources(load_index=False)
    cache = scale_check.DataCache(max_items=4)
    inv = json.loads(DATASETS_INV.read_text())
    groups, refs, counts = load_tree_scope(src.included(), tc["scope_tree"], getattr(a, "t1_fallback_record", None))
    split_loader = content_addressed_split if TREES[tc["scope_tree"]].get("content_split") else None
    if version == SPLIT_V2 and counts != plan["counts"]:
        raise SystemExit(f"scope {counts} != plan {plan['counts']}")
    if version == SPLIT_V2G and not {e["dataset_hash"] for e in plan["datasets"]} <= set(groups):  # (supplements too)
        raise SystemExit("v2gsplit plan names datasets outside its base scope")
    refs = input_refs(refs) | {"owner_line": plan["owner_line"], "owner_decision_nonraw": file_ref(NONRAW_DECISION)}
    mine = [e for e in plan["datasets"] if e.get("disposition", "build") == "build" and e["shard"] == a.shard]
    if a.datasets:
        keep = set(a.datasets.split(","))
        mine = [e for e in mine if e["dataset_hash"] in keep]
    shard_dir = out / f"shard-{tag}{a.shard}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    t0, c0 = time.time(), cpu_s()
    counts_run, failed, pauses = collections.Counter(), [], 0
    for n, e in enumerate(sorted(mine, key=lambda e: -e["est_cpu_s"])):
        h = e["dataset_hash"]
        if dataset_done_v2(shard_dir, h, codes, version):
            counts_run["already_done"] += 1
            continue
        pauses += guard()
        its = groups[h]
        if [(it[0]["view_id"], it[0]["context_sha256"]) for it in its] != [(c["view_id"], c["context_sha256"]) for c in e["epoch0"]]:
            raise SystemExit(f"{h}: epoch-0 contexts differ from the plan")
        try:
            ds = load_dataset(h, [it[0] for it in its], src, cache, inv, v1_port_check=False,      # no all-row v1 port
                              split_loader=split_loader)
            check_cap_label(ds, its[0][2])
        except Exception as err:
            rec = {"schema": DATASET_SCHEMA_V2, "dataset_hash": h, "split_version": version, "status": "failed",
                   "reason": err.reason if isinstance(err, RebuildError) else "dataset_error", "detail": f"{type(err).__name__}: {err}"[:1500],
                   "code_sha256": codes, "contexts": [], "utc_us": int(time.time() * 1e6)}
            (shard_dir / "_datasets").mkdir(parents=True, exist_ok=True)
            atomic(shard_dir / "_datasets" / f"{h}.json", jdump(rec))
        else:
            ga = {k: e[k] for k in ("group_column", "group_source", "group_source_a", "group_evidence") if k in e} or None
            rec = build_dataset_v2(ds, its, shard_dir, codes, refs, profile=entry_profile(e, version),
                                   extra={"registry_grouping": e.get("registry_grouping"), "t1_fallback": e.get("t1_fallback"),
                                          "group_annotation": ga})
        if rec["status"] == "done":
            counts_run["done"] += 1
        else:
            counts_run["failed"] += 1
            failed.append({"dataset_hash": h, "reason": rec.get("reason"), "detail": (rec.get("detail") or "")[:500],
                           "epoch0_contexts_kept": len(e["epoch0"])})
        print(json.dumps({"tree": a.tree, "shard": a.shard, "n": n + 1, "of": len(mine), "dataset": h, "status": rec["status"],
                          "reason": rec.get("reason"), "cpu_s": rec.get("cpu_s"), "elapsed_s": round(time.time() - t0)}), flush=True)
    done = sum(1 for e in mine if dataset_done_v2(shard_dir, e["dataset_hash"], codes, version))
    rec = {"schema": SHARD_SCHEMA_V2, "tree": a.tree, "shard": f"{tag}{a.shard}" if tag else a.shard, "of": a.of, "plan_sha256": plan_sha,
           **({"plan_file": plan_path.name} if tag else {}), "datasets_filter": a.datasets,
           "expected_datasets": len(mine), "done": done, "done_now": counts_run["done"], "already_done": counts_run["already_done"],
           "failed": failed, "complete": done == len(mine) and not failed, "cpu_s": round(cpu_s() - c0, 1),
           "elapsed_s": round(time.time() - t0, 1), "guard_pauses": pauses, "code_sha256": codes, "published": False,
           "utc_us": int(time.time() * 1e6)}
    name = f"shard-{tag}{a.shard}-of-{a.of}.receipt.json" if not a.datasets else f"shard-{tag}{a.shard}-of-{a.of}.partial-{int(time.time())}.receipt.json"
    atomic(out / name, (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps({k: rec[k] for k in ("tree", "shard", "expected_datasets", "done", "done_now", "already_done", "complete", "cpu_s",
                                          "elapsed_s")} | {"failed": len(failed)}))
    return 0 if rec["complete"] else 1


def verify_context_dir_v2(d: Path, rights_extra: dict | None = None) -> dict:
    ctx = json.loads((d / "context.json").read_bytes())
    p = []
    if ctx.get("schema") != SCHEMA_V2 or ctx.get("published") is not False or ctx.get("values_mode") != "L7b":
        p.append("schema/published/values_mode")
    if ctx.get("view_id") != d.name or ctx["identity"]["view_id"] != d.name:
        p.append("view_id")
    if sha(rowsel._canon(ctx["identity"])) != ctx["context_sha256"]:
        p.append("context_sha256")
    rs = ctx["rowsel"]
    ver = ctx.get("split_version")
    tag, prefix = (VIEW_TAG_V2G, CUT_PREFIX_V2G) if ver == SPLIT_V2G else (VIEW_TAG_V2, CUT_PREFIX)
    if d.name != new_view_id_v2(ctx["dataset_hash"], ctx["fold"], rs["view"]["identity"], rs["split"]["sha256"], tag):
        p.append("view_id_suffix")
    cut = ctx.get("cut") or {}
    rc = cut.get("replaces_cut")
    if ver == SPLIT_V2G:
        rc_ok = (isinstance(rc, list) and len(rc) == 2 and rc[0].get("split_sha256") == rs["registry_split_replaced"]["sha256"]
                 and str(rc[1].get("cut_id", "")).startswith(CUT_PREFIX) and cut.get("group_column") == rs["split"].get("group_column"))
    else:
        rc_ok = (rc or {}).get("split_sha256") == rs["registry_split_replaced"]["sha256"]
    if (cut.get("cut_id") != prefix + rs["split"]["sha256"] or cut.get("schema") != ver
            or cut.get("comparable_with_replaced") is not False or ctx["identity"]["cut_id"] != cut.get("cut_id") or not rc_ok):
        p.append("cut")
    if ver not in (SPLIT_V2, SPLIT_V2G) or (rs["split"]["schema"], rs["split"]["partition"]) != (ver, T1_PARTITION):
        p.append("split")
    if rs["split"]["groups_straddling_folds"] or not rs["split"]["target_invariant"] or not rs["split"]["value_invariant"]:
        p.append("split_gates")
    if ver == SPLIT_V2G and (rs["split"].get("annotation_groups_straddling_folds") != 0 or not rs["split"].get("group_value_invariant")):
        p.append("group_gates")
    if ver == SPLIT_V2G:
        sizes = rs["split"]["fold_sizes"]
        empty = empty_fold_list(sizes)
        if (empty != (rs["split"].get("empty_folds") or []) or empty != (cut.get("empty_folds") or [])
                or (empty and not empty_folds_allowed(rs["split"])) or ctx["fold"] in empty
                or rs["split"].get("effective_folds", len(sizes)) != len(sizes) - len(empty)
                or cut.get("effective_folds", len(sizes)) != len(sizes) - len(empty) or cut.get("folds", len(sizes)) != len(sizes)):
            p.append("empty_folds")
    cf, dcf = rs["counterfactual"], rs["downstream_counterfactual"]
    if not (cf["train_side_invariant"] and cf["test_draw_invariant"] and dcf["train_side_invariant"]):
        p.append("gates")
    for name, meta in ctx["files"].items():
        f = d / name
        if not f.is_file() or f.stat().st_size != meta["bytes"] or sha_file(f) != meta["sha256"]:
            p.append(f"file:{name}")
    if rights_extra is not None and {k: ctx.get(k) for k in ("rights", "rights_decision")} != \
            {k: rights_extra.get(k) for k in ("rights", "rights_decision")}:
        p.append("rights")
    unlisted = sorted({x.name for x in d.iterdir()} - set(ctx["files"]) - {"context.json", ".done"})
    if unlisted:
        p.append(f"unlisted:{unlisted}")
    return {"context": ctx, "problems": p}


CODE_CUSTODY = RECEIPTS / "code-custody"
CODE_WHY_DEFAULT = "the live file changed after the build; the recorded bytes are verified in code custody"


def code_custody_check(recorded: dict, root: Path | None = None, custody: Path = CODE_CUSTODY, why: dict | None = None) -> dict:
    """Can every recorded code hash be produced from exact bytes? A hash is producible from the live file under SRC when
    the live bytes hash to it, else from <custody>/<file name>.<sha256> when THOSE bytes hash to it (the name alone proves
    nothing). Files whose live bytes differ are listed with the reason given in `why` (or the default)."""
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    why = why or {}
    files = {}
    for f, h in sorted((recorded or {}).items()):
        live_p = root / f
        live = sha_file(live_p) if live_p.is_file() else None
        src = str(live_p) if live == h else None
        if src is None:
            c = Path(custody) / f"{Path(f).name}.{h}"
            if c.is_file() and sha_file(c) == h:
                src = str(c)
        row = {"recorded": h, "live": live, "same_as_live": live == h, "bytes_from": src, "producible": src is not None}
        if live != h:
            row["why"] = why.get(f, CODE_WHY_DEFAULT)
        files[f] = row
    expected = set(CODE_FILES)
    return {"ok": bool(files) and all(x["producible"] for x in files.values()) and set(files) <= expected
                  and not (expected - set(files) - LATER_CODE_FILES),
            "files": files, "missing_files": sorted(expected - set(files)), "unexpected_files": sorted(set(files) - expected),
            "differs_from_live": sorted(f for f, x in files.items() if not x["same_as_live"]), "custody": str(custody)}


def summarize_v2(a) -> int:
    """Verify the tree, write its summary receipt and, when every planned dataset has a record, its create-only
    .complete marker (an existing marker is kept). Recorded code is checked against exact bytes (code_custody_check),
    not against the live tree, which other owners may edit after a build."""
    out = Path(a.out)
    plan = json.loads((out / "plan.json").read_bytes())
    codes = code_shas()
    why = dict(x.split("=", 1) for x in (getattr(a, "code_why", None) or []))
    custody = Path(getattr(a, "code_custody", None) or CODE_CUSTODY)
    code_checks = {}
    tc = tree_cfg(a.tree)
    version = tc["version"]
    if plan.get("split_version") != version or plan.get("tree") != a.tree:
        raise SystemExit(f"plan {plan.get('tree')}/{plan.get('split_version')} != {a.tree}/{version}")
    want_extra = {}
    if TREES[tc["scope_tree"]]["scope"] == "frozen-kmeans-research":
        want_extra = {"rights": "research-only", "rights_decision": rights_decision_ref()}
    elif TREES[tc["scope_tree"]]["scope"] == "frozen-capped-drifted":
        want_extra = {"rights": "research-only", "rights_decision": all_data_research_ref()}
    sups = []                                                            # create-only supplements of plan.json
    for sp in sorted(out.glob("plan-supplement-*.json")):
        d = json.loads(sp.read_bytes())
        if d.get("kind") != "supplement" or d["supplements"]["sha256"] != sha_file(out / "plan.json") or d.get("split_version") != version:
            raise SystemExit(f"{sp} does not supplement this plan.json")
        sups.append((sp, d))
    entries = {e["dataset_hash"]: dict(e, _tag="", _plan="plan.json") for e in plan["datasets"]}
    for sp, d in sups:
        for e in d["datasets"]:
            if entries[e["dataset_hash"]].get("disposition", "build") == "build":
                raise SystemExit(f"{sp} re-plans {e['dataset_hash']}, which plan.json builds")
            entries[e["dataset_hash"]] = dict(e, _tag=d["shard_tag"], _plan=sp.name)
    rows, problems, ctx_dirs = [], [], set()
    for e in [entries[h] for h in sorted(entries)]:
        h = e["dataset_hash"]
        rg = e.get("registry_grouping") or {}
        row = {"dataset_hash": h, "label": e["label"], "epoch0_contexts": len(e["epoch0"]),
               "registry_grouped_by_feature_column": rg.get("registry_grouped_by_feature_column"),
               "registry_group_columns": rg.get("registry_group_columns") or [],
               "group_annotation_dropped": bool(rg.get("registry_grouped_by_feature_column")) and version == SPLIT_V2,
               "plan": e["_plan"], "effective_folds": e.get("effective_folds", rowsel_raw.OUTER_FOLDS),
               "empty_folds": e.get("empty_folds", [])}
        if version == SPLIT_V2G:
            ev = e["group_evidence"]
            row.update(group_column=e.get("group_column"), group_source=e.get("group_source"),
                       group_evidence={"reproducing": ev["reproducing"], "candidates": ev["candidates"],
                                       "identical_components_across_reproducing": ev["identical_components_across_reproducing"],
                                       "registry_folds_compared": ev["registry_folds_compared"], "base_seed": ev["base_seed"]},
                       replaces_v2split_cut=e["replaces_v2split"]["cut_id"])
        if e.get("disposition", "build") != "build":
            row.update(status="not_built", reason=e.get("reason"))
            rows.append(row)
            continue
        sd = out / f"shard-{e['_tag']}{e['shard']}"
        p = sd / "_datasets" / f"{h}.json"
        if not p.exists():
            row["status"] = "missing_record"
        else:
            r = json.loads(p.read_bytes())
            row.update(status=r["status"], reason=r.get("reason"), detail=(r.get("detail") or "")[:400], cpu_s=r.get("cpu_s"),
                       record={"path": str(p), "sha256": sha_file(p)})
            if r["status"] == "done":
                key = json.dumps(r.get("code_sha256"), sort_keys=True)
                if key not in code_checks:
                    code_checks[key] = code_custody_check(r.get("code_sha256"), custody=custody, why=why)
                if not code_checks[key]["ok"]:
                    problems.append({"dataset": h, "problem": "code_not_producible",
                                     "files": sorted(f for f, x in code_checks[key]["files"].items() if not x["producible"])})
                ids = []
                for c in r["contexts"]:
                    d = sd / c["view_id"]
                    if not (d / ".done").exists():
                        problems.append({"dir": str(d), "problems": ["no .done"]}); continue
                    v = verify_context_dir_v2(d, want_extra if want_extra else None)
                    if v["problems"]:
                        problems.append({"dir": str(d), "problems": v["problems"]})
                    if v["context"]["context_sha256"] != c["context_sha256"] or v["context"]["cut"]["cut_id"] != r["cut"]["cut_id"]:
                        problems.append({"dir": str(d), "problems": ["record_mismatch"]})
                    if bool(v["context"].get("t1_fallback")) != bool(e.get("t1_fallback")):
                        problems.append({"dir": str(d), "problems": ["t1_fallback"]})
                    if v["context"]["row_selection_leak"] != [e["label"]]:
                        problems.append({"dir": str(d), "problems": ["label"]})
                    tlv = (v["context"].get("target_linear") or {}).get("verdict")
                    if tlv == target_linear.REFUSE:
                        problems.append({"dir": str(d), "problems": ["target_linear_in_features"]})
                    elif tlv == target_linear.REVIEW:
                        row.setdefault("target_linear_near_review", []).append(c["view_id"])
                    if v["context"].get("split_version") != version:
                        problems.append({"dir": str(d), "problems": ["split_version"]})
                    if version == SPLIT_V2G:
                        rc = v["context"]["cut"]["replaces_cut"]
                        if (rc[1].get("cut_id") != e["replaces_v2split"]["cut_id"] or v["context"]["cut"]["group_column"] != e["group_column"]
                                or set(v["context"]["identity"]["epoch0_replaced"]) != {x["view_id"] for x in e["epoch0"]}
                                or c["view_id"] in e["replaces_v2split"]["view_ids"]):
                            problems.append({"dir": str(d), "problems": ["replaces_cut_or_group_column"]})
                    ctx_dirs.add(str(d)); ids.append(c["view_id"])
                if len(ids) + len(r.get("refused_contexts") or []) != row["effective_folds"] or len(r["contexts"]) != len(ids):
                    problems.append({"dataset": h, "problems": ["context_count"], "contexts": len(ids), "planned": row["effective_folds"]})
                if version == SPLIT_V2G and r["split"].get("empty_folds", []) != row["empty_folds"]:
                    problems.append({"dataset": h, "problems": ["empty_folds_differ_from_plan"]})
                rc = r["cut"]["replaces_cut"]
                row.update(new_view_ids=ids, cut_id=r["cut"]["cut_id"], split_sha256=r["split"]["sha256"],
                           fold_sizes=r["split"]["fold_sizes"],
                           replaces_cut=[x.get("cut_id") or x["split_sha256"] for x in rc] if isinstance(rc, list) else rc["split_sha256"])
                if version == SPLIT_V2G:
                    row.update(components=r["split"].get("components"), annotation_groups=r["split"].get("annotation_groups"),
                               annotation_groups_straddling_folds=r["split"].get("annotation_groups_straddling_folds"),
                               group_value_invariant=r["split"]["group_value_check_fold0"]["invariant"],
                               value_invariant=r["split"]["value_check_fold0"]["invariant"],
                               target_invariant=r["split"]["target_check"]["target_invariant"])
        rows.append(row)
    orphans = [str(d) for sd in out.glob("shard-*") if sd.is_dir() for d in sd.iterdir()
               if d.is_dir() and d.name != "_datasets" and str(d) not in ctx_dirs]
    not_built = [r for r in rows if r["status"] == "not_built"]
    rows_b = [r for r in rows if r["status"] != "not_built"]
    done = [r for r in rows if r["status"] == "done"]
    counts = {"expected_datasets": len(rows_b), "expected_epoch0_contexts": sum(r["epoch0_contexts"] for r in rows_b),
              "expected_new_contexts": sum(r["effective_folds"] for r in rows_b), "done_datasets": len(done),
              "done_new_contexts": sum(len(r["new_view_ids"]) for r in done),
              "epoch0_contexts_replaced": sum(r["epoch0_contexts"] for r in done),
              "epoch0_contexts_kept": sum(r["epoch0_contexts"] for r in rows_b if r["status"] != "done"),
              "failed_datasets": sum(r["status"] == "failed" for r in rows), "missing_records": sum(r["status"] == "missing_record" for r in rows),
              "failed_by_reason": dict(collections.Counter(r.get("reason") for r in rows if r["status"] == "failed")),
              "registry_grouped_by_feature_column_datasets": sum(1 for r in rows if r.get("registry_grouped_by_feature_column")),
              "group_annotation_dropped_datasets": sorted(r["dataset_hash"] for r in rows if r["group_annotation_dropped"] and r["status"] == "done"),
              "registry_group_annotated_not_built": sorted(r["dataset_hash"] for r in rows
                                                           if r.get("registry_grouped_by_feature_column") and r["status"] == "not_built"),
              "verification_problems": len(problems), "orphan_context_dirs": len(orphans)}
    rec_names = [f"shard-{i}-of-{plan['of']}.receipt.json" for i in range(plan["of"])]
    rec_names += [f"shard-{d['shard_tag']}{i}-of-{d['of']}.receipt.json" for _, d in sups for i in range(d["of"])]
    shard_recs = [file_ref(out / n) | {k: json.loads((out / n).read_bytes())[k] for k in ("expected_datasets", "done", "complete", "cpu_s", "elapsed_s", "guard_pauses")}
                  for n in rec_names if (out / n).exists()]
    mut = json.loads(Path(a.mutants).read_bytes())
    if version == SPLIT_V2G:
        counts["empty_folds_datasets"] = {r["dataset_hash"]: r["empty_folds"] for r in rows_b if r["empty_folds"]}
        counts["not_built_datasets"] = len(not_built)
        counts["not_built_by_reason"] = dict(collections.Counter(r.get("reason") for r in not_built))
        done_h = {r["dataset_hash"] for r in done}
        counts["replaced_v2split_contexts"] = sum(len(e["replaces_v2split"]["view_ids"]) for e in entries.values()
                                                  if e.get("disposition") == "build" and e["dataset_hash"] in done_h)
        counts["annotation_groups_straddling_folds"] = sum(r.get("annotation_groups_straddling_folds") or 0 for r in done)
        counts["group_source"] = dict(collections.Counter("inferred-ambiguous-identical" if len(r["group_evidence"]["reproducing"]) > 1
                                                          else "inferred-single" for r in rows_b))
    doc = {"schema": SUMMARY_SCHEMA_V2, "utc_us": int(time.time() * 1e6), "tree": a.tree, "out": str(out), "split_version": version,
           "plan": file_ref(out / "plan.json"), "plan_counts": plan["counts"], "inputs": plan["inputs"], "counts": counts,
           **({"plan_supplements": [file_ref(sp) | {"counts": d["counts"], "decision": d["decision"]["text"]} for sp, d in sups]} if sups else {}),
           "datasets": rows, "problems": problems[:100], "orphans": orphans[:50], "shard_receipts": shard_recs,
           "cpu": {"shard_process_cpu_s": round(sum(s["cpu_s"] for s in shard_recs), 1),
                   "dataset_cpu_s": round(sum(r.get("cpu_s") or 0 for r in rows), 1)},
           "tests": {"log": file_ref(Path(a.tests_log)), "tail": Path(a.tests_log).read_text().strip().splitlines()[-3:]},
           "mutants": {"result": file_ref(Path(a.mutants)), "totals": mut.get("totals")},
           "pilot": json.loads(Path(a.pilot).read_bytes()) if getattr(a, "pilot", None) else None,
           "code_sha256": codes,
           "code_custody": {"rule": "each recorded hash is produced from exact bytes: the live file, or code custody "
                                    "<file name>.<sha256> whose bytes hash to it; only an unproducible hash is a problem",
                            "custody": str(custody), "recorded_code_sets": [dict(c, recorded=json.loads(k)) for k, c in code_checks.items()]},
           "owner_decision": file_ref(NONRAW_DECISION),
           **({"owner_ruling": owner_ruling_v2g()} if version == SPLIT_V2G else {}),
           "policy": {"s3_calls": 0, "network": False, "gpu": False, "git": False, "published": False, "old_trees_touched": False}}
    stem = {"kmeans-clean": "rowsel-t1-kmeans-v2split-summary", "kmeans-fallback": "rowsel-t1-fallback-v2split-summary",
            "frozen-kmeans-research": "rowsel-t1-frozen-kmeans-research-v2split-summary",
            "kmeans-clean-groups": "rowsel-t1-kmeans-v2gsplit-summary", "kmeans-fallback-groups": "rowsel-t1-fallback-v2gsplit-summary",
            "frozen-kmeans-research-groups": "rowsel-t1-frozen-kmeans-research-v2gsplit-summary",
            "frozen-capped-drifted-fallback": "rowsel-t1-frozen-capped-fallback-v2split-summary"}[a.tree]
    rp = Path(a.receipt) if a.receipt else RECEIPTS / "build" / f"{stem}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    data = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    with open(rp, "xb") as fh:
        fh.write(data)
    complete = (counts["missing_records"] == 0 and not problems and not orphans and len(shard_recs) == len(rec_names)
                and all(s["complete"] for s in shard_recs))
    marker = None
    if complete and not (out / ".complete").exists():
        marker = out / ".complete"
        body = jdump({"schema": "jope.dl-v2-rowsel-tree-complete/v1", "tree": str(out), "split_version": version,
                      **({"plan_supplements": [file_ref(sp) for sp, _ in sups]} if sups else {}),
                      "summary": {"path": str(rp), "sha256": sha(data), "bytes": len(data)}, "counts": counts, "published": False,
                      "utc_us": int(time.time() * 1e6)})
        with open(marker, "xb") as fh:
            fh.write(body); fh.flush(); os.fsync(fh.fileno())
    print(json.dumps({"receipt": str(rp), "sha256": sha(data), "complete": complete, "marker": str(marker) if marker else None,
                      "counts": counts}))
    return 0 if complete else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="required for the registry route; a v2split tree defaults to TREES[tree]['out']")
    ap.add_argument("--split-version", choices=SPLIT_VERSIONS, required=True,
                    help=f"required, no default (PR #218 finding 10): {SPLIT_V2} or {SPLIT_V2G}; '{SPLIT_REGISTRY}' only as the "
                         "explicit historical replay")
    ap.add_argument("--tree", choices=sorted(TREES) + sorted(TREES_V2G),
                    help="v2split tree (--split-version rowsel-raw-split-v2) or v2gsplit tree (--split-version rowsel-split-v2-groups)")
    ap.add_argument("--plan", action="store_true", help="v2split: write the tree's pinned plan.json")
    ap.add_argument("--plan-supplement", action="store_true", help="v2gsplit: write the empty-folds supplement of plan.json")
    ap.add_argument("--plan-file", help="v2gsplit build: plan file in the tree (default plan.json; a supplement's name)")
    ap.add_argument("--t1-fallback-record", help="REQUIRED (and only allowed) for the opt-in drifted T1 fallback tree: the "
                                                 "rights-held fill completion record whose 'drifted' list it builds")
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
    ap.add_argument("--pilot", help="v2split summarize: the pilot record JSON")
    ap.add_argument("--code-custody", help="v2split summarize: code custody dir (default RUN/receipts/trainonly-v2/code-custody)")
    ap.add_argument("--code-why", action="append", help="v2split summarize: FILE=REASON for a file whose live bytes differ")
    ap.add_argument("--receipt")
    a = ap.parse_args(argv)
    if a.split_version in (SPLIT_V2, SPLIT_V2G):
        if a.tree is None:
            ap.error(f"--tree is required with --split-version {a.split_version}")
        tc = tree_cfg(a.tree)
        if tc["version"] != a.split_version:
            ap.error(f"tree {a.tree} is a {tc['version']} tree")
        opt_in = TREES.get(tc["scope_tree"], {}).get("fallback_kind") == "drifted"
        if opt_in and not a.t1_fallback_record:
            ap.error(f"tree {a.tree} is an explicit opt-in: --t1-fallback-record is required")
        if a.t1_fallback_record and not opt_in:
            ap.error("--t1-fallback-record belongs to the drifted T1 fallback tree only")
        if a.out is None:
            a.out = str(tc["out"])
        if a.overlay:
            ap.error("the overlay is the registry route's; v2split contexts replace epoch-0 at dataset level")
        if a.plan:
            return plan_v2(a) if a.split_version == SPLIT_V2 else plan_v2g(a)
        if a.plan_supplement:
            if a.split_version != SPLIT_V2G:
                ap.error("--plan-supplement is for v2gsplit trees")
            return plan_v2g_supplement(a)
        if a.summarize:
            return summarize_v2(a)
        if a.shard is None:
            ap.error("--shard is required for a build")
        return build_v2(a)
    if a.out is None or a.tree is not None or a.plan:
        ap.error("the registry route needs --out and takes no --tree / --plan")
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

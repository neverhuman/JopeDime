"""PLAN §5 P6: train-only context encodings for the dltrain lane. CPU only: no GPU, no encoder inference, no model weights.

Subcommands. Exit codes: 0 PASS or DRY_RUN, 2 FAIL or refused input, 3 BLOCKED (gate, budget or guard), 4 STOP
(closure, POISONED/immutable conflict, test-row dependence).

- map-audit: the read-only adoption mapping audit against Codex's neural-encoding-queue-v1 output index. It joins every
  included epoch-0 context to Codex outcomes on (dataset_hash, sample_identity_sha256, fold), with repeat 0, and requires
  exact equality of the outer-train and outer-test index hashes and row counts between the context's own membership
  object and Codex's outcome. Same CLI and output as before; it now runs on the shared join/binding helpers below.
- run --arm A0: A0 train-only meta-features (the dltrain A0 producer module) per admitted context from JTT1 outer-train
  bytes only, one create-only '<f4' array per context, the block normalizer refit on train_fit rows only, and the
  jope.dl-context-encoding/v1 manifests (train_fit, dev, diag; heldout under sealed/). Every train_fit family of the
  scope must be closed (its mapped contexts train_fit or diag) before anything is written, and the fitted normalizer
  passes normalizers.assert_fit_closure before the normalizer or a manifest is written (else STOP). Manifests label
  present contexts train_only=STRUCTURAL_JTT1_TRAIN_BYTES, scoped to JTT1 bytes (UPSTREAM-SCALE-01) with each context's
  deterministic upstream exposure; VERIFIED_BY_COUNTERFACTUAL exists only in a PASS verify receipt that binds the
  manifest sha256 values. RUN/encodings/<arm> needs a bound p3 partition map (bind_partition_map), the P3 holdout
  binding and holdout.json, a P3 PASS receipt and a P5 PASS receipt for this epoch and map; otherwise OUT must lie under
  RUN/tmp/p6-sample-*/. run --arm A1|A2|A3|LB-replica exits 3 (BLOCKED) with the reasons.
- adopt: binds Codex queue outputs to the DL arms by reference only (bucket, key, VersionId, sha256). Tier I classifies
  every included context with no GET; tier O fetches outcomes and tier A arrays inside the GET budgets. A dry run may
  restrict tiers O/A to a --contexts subset. Production mode (no --dry-run) refuses --contexts, needs --content-checks,
  --jtt1-cache, a terminal index whose release names it (index_ref on all five pin fields; a terminal release's
  original launch release must be the PLAN pin), an exact-VersionId index fetch receipt, release qualification flags or
  an owner waiver, and PASSes only when every hyperion/foundation/target row is final with JTT1-derived evidence at
  tier A. The production output gate (RUN/encodings/adoption/...) also needs the bound p3 map, holdout binding and the
  P3/P5 receipts. Adopted vectors are labelled CLAIMED train-only: structural and content evidence is recorded, but
  upgrading the label would need a re-encode counterfactual (encoder inference), which is out of scope. Nothing is
  written to S3 and no array bytes are written to disk.
- verify --arm A0: coverage, fit closure, poison test, JTT1-byte counterfactual, affine invariance, the deterministic
  upstream exposure check and the cross-fold column-set audit; a p3 manifest's map binding is recomputed. The receipt
  carries the train_only promotion. Never GETs.
- sample, fake-partition-map, bench-a0: helpers for the bounded real-sample measurement (outputs under RUN/tmp only).
- schemas: writes the embedded JSON Schemas (Draft 2020-12) create-only into a directory.

JTT1 bytes come only from jtt1.load_raw (exact-VersionId GET into the create-only cache, verified on every read) and are
decoded only by jtt1.train_view inside the A0 producer. dltrain.s3io is imported only inside functions; every S3 call is
a read through the guarded lane read client, which refuses PutObject.
"""
import argparse
import ast
import collections
import concurrent.futures as cf
import functools
import io
import json
import math
import os
import platform
import random
import re
import resource
import struct
import sys
import tempfile
import time
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path

import numpy as np

from dltrain import common, cpuguard, jtt1

# ----------------------------------------------------------------------------------------------------- constants
EXPERIMENT_ID = "gp-policy-dl-v1-e0"
RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
RUN_TMP = RUN / "tmp"
RUN_ENCODINGS = RUN / "encodings"
SAMPLE_DIR_PREFIX = "p6-sample-"
DEFAULT_MEMBERSHIPS = RUN_ENCODINGS / "memberships.jsonl"

PARTITIONS = ("train_fit", "dev", "diag", "heldout")
EXCLUDED_ROLES = ("sealed", "prohibited")
OPEN_PARTITIONS = ("train_fit", "dev", "diag")
ARMS = ("A0", "A1", "A2", "A3", "LB-replica")
COMPONENTS = ("hyperion", "foundation", "target", "tabpfn35", "context2048")
A0_BLOCK = "m"

EXIT_PASS, EXIT_FAIL, EXIT_BLOCKED, EXIT_STOP = 0, 2, 3, 4

CONTEXT_ENCODING_SCHEMA = "jope.dl-context-encoding/v1"
ARM_SCHEMA = "jope.dl-encoding-arm/v1"
ADOPTION_SCHEMA = "jope.dl-encoding-adoption/v1"
VERIFY_SCHEMA = "jope.dl-encoding-verify/v1"
PACKET_RECEIPT_SCHEMA = "jope.dl-packet-receipt/v1"
ADOPTION_RECEIPT_SCHEMA = "jope.dl-encoding-adoption-receipt/v1"
LEDGER_SCHEMA = "jope.dl-get-ledger/v1"
SAMPLE_SCHEMA = "jope.dl-p6-sample/v1"
FAKE_MAP_META_SCHEMA = "jope.dl-fake-partition-map-meta/v1"
BENCH_SCHEMA = "jope.dl-a0-bench/v1"
SEALED_CLOSURE_SCHEMA = "jope.dl-encoding-sealed-closure/v1"
FIT_CONTEXT_SET_SCHEMA = "jope.dl-fit-context-set/v1"
FIT_FAMILY_SET_SCHEMA = "jope.dl-fit-family-set/v1"
NORMALIZER_SCHEMA = "jope.dl-normalizer/v1"
# dltrain.normalizers.ARRAY_ORDER and METHOD keys, restated so this module keeps its light top-level imports (a test ties them)
NORMALIZER_ARRAY_ORDER = ("mean", "scale", "rotation", "proj_mean", "proj_scale", "eigenvalues")
NORMALIZER_METHOD_KEYS = ("standardize", "pca", "projection_scale", "clip_abs", "rank_rtol", "dtype_compute", "dtype_stored",
                          "gram", "eigh")

# run labels present A0 rows structurally; only a PASS verify receipt bound to the manifest sha256 values promotes them
A0_TRAIN_ONLY = "STRUCTURAL_JTT1_TRAIN_BYTES"
A0_TRAIN_ONLY_VERIFIED = "VERIFIED_BY_COUNTERFACTUAL"
ADOPTED_TRAIN_ONLY = "CLAIMED"
UPSTREAM_FINDING = "UPSTREAM-SCALE-01"
TRAIN_ONLY_SCOPE = "jtt1_bytes"
TRAIN_ONLY_CLAIM = "train-only relative to JTT1 bytes, not to raw data (UPSTREAM-SCALE-01)"
TRAIN_ONLY_RULE = ("train_only=STRUCTURAL_JTT1_TRAIN_BYTES: computed from jtt1.train_view bytes only, scoped to JTT1 bytes "
                   f"({TRAIN_ONLY_CLAIM}); VERIFIED_BY_COUNTERFACTUAL is granted only by a create-only PASS verify receipt "
                   "(jope.dl-encoding-verify/v1, train_only_promotion.granted) that binds these manifests' sha256 values "
                   "after coverage, fit closure, poison (scope all), counterfactual, affine invariance and upstream "
                   "exposure; a context with upstream exposure stays OPEN under UPSTREAM-SCALE-01 and is never promoted; "
                   "run plans that read these manifests need that receipt")
# v -> (v + d) / (1 + d): a downward all-row minimum shift of d range units, as upstream min-max scaling over train and
# test rows applies when an outer-test row holds the minimum; a column whose distinct train values merge under one of these
# shifts has A0 values that can depend on the outer-test minimum (TOL-04)
UPSTREAM_SHIFTS = (1e-6, 1e-3, 1.0)
EXPOSURE_LIST_MAX = 32
PROMOTION_CHECKS = ("coverage", "fit_closure", "poison_test", "counterfactual", "affine_invariance", "upstream_exposure")
GATING_STATUSES = ("PASS", "FAIL", "STOP")
GUARD_MAX_PAUSE_S = 1800.0
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
COUNTERFACTUAL_SEED = 20261005
AFFINE_SEED = 20261005
POISON_SEED = 20261005
AFFINE_ATOL = AFFINE_RTOL = 1e-6
NAN_PATTERNS = (0x7FF8000000000000, 0x7FF0000000000001, 0xFFF8000000000000, 0x7FF0000000000000, 0xFFF0000000000000,
                0x7FFFFFFFFFFFFFFF)
COUNTERFACTUAL_VARIANTS = ("R1a_fill_00", "R1b_fill_ff", "R2_random_finite_f64", "R3_nan_inf_patterns", "R4_test_rows_removed")
POISON_SYNTHETIC = {"dev": 2, "heldout": 2, "diag": 1}
DIAG_NOTE = ("diag (NONINDEPENDENT_SPLIT_DIAGNOSTIC, PLAN §6.2) is excluded from every fit as JTT1 files, not as rows: a diag "
             "context's outer-test rows are outer-train rows of its train_fit sibling folds, which do feed the normalizer; "
             "poison variant (a) shows the fit never reads a diag JTT1 file, not that diag rows cannot reach it")
# dltrain.normalizers.DEV_ONLY_ARMS and RESEARCH_ONLY_ARMS, restated to keep the light top-level imports (a test ties them)
DEV_ONLY_ARMS = ("LB-replica",)
RESEARCH_ONLY_ARMS = ("A2", "A3", "LB-replica")

# P3 holdout (dltrain.holdout HOLDOUT_SCHEMA, BINDING_SCHEMA and ASSIGNMENTS, restated; a test ties them)
HOLDOUT_SCHEMA = "jope.dl-family-holdout/v1"
HOLDOUT_BINDING_SCHEMA = "jope.dl-binding/v1"
P3_ASSIGNMENT_ROLES = {"heldout": "heldout", "dev": "dev", "train_fit": "train_fit", "train_nonpool": "train_fit",
                       "sealed_excluded": "sealed", "prohibited_excluded": "prohibited"}
MAP_BINDING_SCHEMA = "jope.dl-partition-map-binding/v1"
MAP_BINDING_RULE = ("every partition-map context's split_role equals the role recomputed from the bound holdout.json (family "
                    "assignment heldout/dev/train_fit|train_nonpool/sealed_excluded/prohibited_excluded -> heldout/dev/train_fit/"
                    "sealed/prohibited; a train context listed in holdout.diag -> diag); holdout.json sha256 and bytes equal the "
                    "binding's holdout_ref; the P3 receipt (and the P5 receipt, when given) is PASS and binds this map's sha256")
OWNER_WAIVER_SCHEMA = "jope.dl-owner-waiver/v1"

# Codex neural-encoding-queue-v1: read-only peer evidence (formats per Codex publish.py, worker.py, gpu_encoders.py)
CODEX_UPSTREAM = "codex-neural-encoding-queue-v1"
CODEX_BUCKET = "veox-jopedime-learning-use2"
RELEASE_SCHEMA = "jope.neural-encoding-release/v1"                      # the launch release (a8382194)
TERMINAL_RELEASE_SCHEMA = "jope.neural-encoding-terminal-release/v1"    # the terminal release (names its index)
RELEASE_SCHEMAS = (RELEASE_SCHEMA, TERMINAL_RELEASE_SCHEMA)
# PLAN §5 P6 step 1: the launch release by exact VersionId (Codex 03:15:12Z HANDOFF)
PLAN_LAUNCH_RELEASE_REF = {
    "bucket": CODEX_BUCKET,
    "key": "experiments/gp-teacher-v6/intake/neural-encodings-v1/releases/"
           "a8382194d981d4b89692561adc28e9ac2c8c016a98c293412699de92d6cc5d0b.json",
    "version_id": "is_2hlXj5EcYec0_ZqMJjlF.4GzlqwCZ",
    "sha256": "a8382194d981d4b89692561adc28e9ac2c8c016a98c293412699de92d6cc5d0b", "bytes": 2591}
INDEX_SCHEMA = "jope.neural-encoding-output-index/v1"
OUTCOME_SCHEMA = "jope.regression-neural-encoding-outcome/v1"
JOB_SCHEMA = "neural-encoding-job/v1"
HYPERION_RECIPE = "gpu-v5-row-sweep-pcg-v1"
HYPERION_SCALES = [4, 7, 11, 15]
FOUNDATION_SEED = 20260707
TARGET_SEED = 17
TARGET_STRUCTURAL_ZEROS = 376
TABPFN35_CHECKPOINT = {"sha256": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3", "bytes": 876027932}
PIN_FIELDS = ("bucket", "key", "version_id", "sha256", "bytes")
# component -> ((packet.model_refs key, inventory section, inventory id, inventory field), ...); None = no inventory pin
COMPONENT_PINS = {
    "hyperion": (("hyperion_v5_arm_b_identity", "components", "hyperion_v5_arm_b", "identity"),
                 ("hyperion_v5_arm_b_runtime_manifest", "components", "hyperion_v5_arm_b", "runtime_manifest")),
    "foundation": (("starlight_core_regressor_identity", "components", "starlight_core_regressor", "identity"),),
    "target": (("target_encoder_fused_v1_identity", "components", "target_encoder_fused_v1", "identity"),
               ("target_encoder_fused_v1_metadata", "components", "target_encoder_fused_v1", "metadata")),
    "tabpfn35": None,
    "context2048": (("context_transforms", "fitted_normalizers", "context_transforms_lineage_b", "identity"),),
}
ARMS_ELIGIBLE = {"hyperion": ["A1", "A2", "A3", "LB-replica"], "foundation": ["A2", "A3", "LB-replica"],
                 "target": ["LB-replica"], "tabpfn35": ["A3"], "context2048": []}
# component -> (inventory section, inventory id) of its encoder licence block (DL-D3a); None = not in the inventory
COMPONENT_LICENSE_SOURCE = {"hyperion": ("components", "hyperion_v5_arm_b"),
                            "foundation": ("components", "starlight_core_regressor"),
                            "target": ("components", "target_encoder_fused_v1"), "tabpfn35": None,
                            "context2048": ("fitted_normalizers", "context_transforms_lineage_b")}
UNRESOLVED_LICENSE = {"class": "unresolved", "redistribution": "unknown", "policy_eligible": False,
                      "note": "no licence block in encoding-inventory.json (TabPFN 3.5 licence unresolved, G03)"}
# JTT1-derived evidence an adopted row needs before a production adoption can PASS (TOL-07/AB-3)
REQUIRED_JTT1_EVIDENCE = {"hyperion": ("shape", "content", "array_content"), "foundation": ("shape", "content", "array_content"),
                          "target": ("shape", "content")}
JTT1_BINDING_NOTE = ("statistical, not a hash equality: JTT1 n_train/n_test, the kept-feature recompute, context y moments "
                     "(foundation), target min/span (target), feature q01/q99 (hyperion) and min/span (foundation) recomputed "
                     "from the pinned JTT1 outer-train rows; Codex train_row_ids_sha256 has no JTT1 counterpart (JTT1 holds no "
                     "row ids) and is compared with the packet fold and the outcome input_binding only")
RECIPE_IDENTITY = {
    "hyperion": ("gpu-v5-row-sweep-pcg-v1; NEW identity per DL-D13; native RNG equivalence unqualified; "
                 "accepted-producer parity not applicable"),
    "foundation": ("Starlight core regressor (TabPFNv3) inner context/query probe, seed 20260707; raw 2048 retained; "
                   "ResearchOnly (DL-D3a)"),
    "target": "fused target encoder v1, seed 17, 376 structural zero dimensions; LB-replica only, never A0-A3",
    "tabpfn35": "TabPFN 3.5 inner context/query probe; checkpoint unpinned in encoding-inventory.json (A3 blocked)",
    "context2048": "historical 2048 ABI assembly through the frozen context_transforms 1b779c6f; legacy projection",
}
DRY_RUN_OUTCOME_COMPONENTS = ("hyperion", "foundation")
PRODUCTION_OUTCOME_COMPONENTS = ("hyperion", "foundation", "target")
DRY_RUN_ARRAY_COMPONENTS = ("hyperion",)
PRODUCTION_ARRAY_COMPONENTS = ("hyperion", "foundation", "target")
OUTCOME_EQUAL_FIELDS = ("job_id", "component", "identity", "fold", "packet_ref", "array_ref", "claim_ref")

_GATES = [
    "terminal_index_and_production_adoption: Codex neural-encoding-queue-v1 must publish a terminal index "
    "(all_shards_terminal=true) and `adopt` must PASS in production mode at the production output gate "
    "(RUN/encodings/adoption/<release12>-<index12>, bound p3 partition map)",
    "npz_read_line: no admitted line yet to read adopted Codex NPZ arrays (exact-VersionId GETs) into encoding arrays",
    "P5_P3: real P6 outputs need a P5 PASS receipt for this epoch and map, the P3 holdout binding and holdout.json, and a P3 "
    "PASS receipt binding the partition map (a fake map is a demonstration)",
    "adopted_train_only_counterfactual: a sampled re-encode with the outer-test rows perturbed (encoder inference, admitted "
    "separately) must give identical vectors before any adopted block feeds a run plan; until then adopted blocks are "
    "CLAIMED train-only, and test-row dependence -> STOP (PLAN §5 P6 step 1 and Failure)",
]
_RESIDUAL = ("DL-D13_residual_producer: the producer for residual contexts (45 feature-sanitized views and absent "
             "outputs) is undecided")
_TABPFN_PIN = ("tabpfn35_pin: the TabPFN 3.5 checkpoint ece4d67e... (876,027,932 B) has no pin in encoding-inventory.json "
               "or packet.model_refs; it needs an owner-approved inventory addendum")
BLOCKED_ARM_REASONS = {"A1": _GATES + [_RESIDUAL], "A2": _GATES + [_RESIDUAL], "A3": _GATES + [_RESIDUAL, _TABPFN_PIN],
                       "LB-replica": list(_GATES)}
ARM_NOTES = {"A2": "ResearchOnly (DL-D3a)", "A3": "ResearchOnly (DL-D3a; TabPFN 3.5 licence unresolved, G03)",
             "LB-replica": "dev only; never holdout-reported (fused target block likely exposed, G06)"}

_HEX64 = re.compile(r"[0-9a-f]{64}")
_PACKAGE = "dltrain"


# ----------------------------------------------------------------------------------------------------- errors
class EncodingError(RuntimeError):
    """A P6 outcome other than PASS. reason is a short machine token; detail is free text."""
    exit_code = EXIT_FAIL
    status = "FAIL"

    def __init__(self, reason, detail=""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail = reason, str(detail)


class Refused(EncodingError):
    """Refused or failed input (exit 2)."""


class Blocked(EncodingError):
    """A gate, budget or guard refusal (exit 3)."""
    exit_code = EXIT_BLOCKED
    status = "BLOCKED"


class Stop(EncodingError):
    """Closure failure, immutable conflict, sha mismatch of pinned bytes or test-row dependence (exit 4)."""
    exit_code = EXIT_STOP
    status = "STOP"


class BudgetExhausted(Blocked):
    """A GET cap was reached; raised before the GET is sent."""


class PartitionMapError(Refused):
    """The partition map is not {64-hex context_sha256: {family_id, split_role}}."""


# ----------------------------------------------------------------------------------------------------- small helpers
def _sha(data: bytes) -> str:
    return common.sha256_bytes(data)


def _is_hex64(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and _HEX64.fullmatch(v) is not None


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _read_bytes(path) -> bytes:
    """Every input file read of run/adopt/verify goes through here."""
    return Path(path).read_bytes()


def _loads(raw: bytes, what: str):
    try:
        return json.loads(raw)
    except ValueError as err:
        raise Refused(f"{what}_unparseable", str(err)) from None


def _real(p) -> Path:
    return Path(os.path.realpath(os.fspath(p)))


def _under(p, root) -> bool:
    p, root = _real(p), _real(root)
    return p == root or root in p.parents


def _stamp(us=None) -> str:
    us = common.utc_us() if us is None else us
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime(us // 1_000_000)) + f"{us % 1_000_000:06d}Z"


def _ceil_frac(fraction, n) -> int:
    """ceil(fraction * n) with exact decimal semantics (no binary floating-point artefacts)."""
    return math.ceil(Fraction(str(fraction)) * n)


def sample_root(out):
    """The RUN/tmp/p6-sample-*/ directory that OUT is or lies under, else None."""
    o, t = _real(out), _real(RUN_TMP)
    try:
        rel = o.relative_to(t)
    except ValueError:
        return None
    if not rel.parts:
        return None
    head = rel.parts[0]
    if not head.startswith(SAMPLE_DIR_PREFIX) or len(head) == len(SAMPLE_DIR_PREFIX):
        return None
    return t / head


def _require_run_tmp(path, what):
    if not _under(path, RUN_TMP) or _real(path) == _real(RUN_TMP):
        raise Refused(f"{what}_outside_run_tmp", f"{path} must lie under {RUN_TMP}")


def _quantiles(values):
    if not values:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {"n": int(arr.size), "p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max())}


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def runtime_identity() -> dict:
    """python, numpy, BLAS/LAPACK build strings (np.show_config) and the thread environment."""
    try:
        cfg = np.show_config(mode="dicts") or {}
    except Exception:  # noqa: BLE001 (an unreadable build config is recorded as such)
        cfg = {}
    deps = cfg.get("Build Dependencies", {}) if isinstance(cfg, dict) else {}
    return {"python": sys.version, "numpy": np.__version__, "machine": platform.machine(),
            "blas": json.dumps(deps.get("blas", {}), sort_keys=True, default=str),
            "lapack": json.dumps(deps.get("lapack", {}), sort_keys=True, default=str),
            "threads_env": {k: os.environ.get(k) for k in THREAD_ENV}}


def _runtime_key(rt) -> tuple:
    rt = rt or {}
    return (rt.get("numpy"), rt.get("blas"), rt.get("machine"), json.dumps(rt.get("threads_env"), sort_keys=True))


# ----------------------------------------------------------------------------------------------- source closure
def _package_imports(tree, package_dir) -> set:
    """dltrain modules a parsed module imports anywhere in its body (the same rule as train.py and holdout.py)."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if (node.level == 0 and module == _PACKAGE) or (node.level == 1 and not module):
                found.update(alias.name for alias in node.names)
            elif node.level == 0 and module.startswith(_PACKAGE + "."):
                found.add(module.split(".")[1])
            elif node.level == 1:
                found.add(module.split(".")[0])
        elif isinstance(node, ast.Import):
            found.update(alias.name.split(".")[1] for alias in node.names if alias.name.startswith(_PACKAGE + "."))
    return {name for name in found if (package_dir / f"{name}.py").is_file()}


@functools.cache
def _closure(root: str) -> tuple:
    package_dir = Path(__file__).resolve().parent
    digests, pending_modules = {}, [root]
    while pending_modules:
        name = pending_modules.pop()
        if f"{name}.py" in digests:
            continue
        raw = (package_dir / f"{name}.py").read_bytes()
        digests[f"{name}.py"] = common.sha256_bytes(raw)
        pending_modules.extend(sorted(_package_imports(ast.parse(raw, filename=f"{name}.py"), package_dir)))
    digests["__init__.py"] = common.sha256_file(package_dir / "__init__.py")
    return tuple((f"{_PACKAGE}/{name}", digest) for name, digest in sorted(digests.items()))


def source_tree(root: str = "encodings") -> list:
    return [list(x) for x in _closure(root)]


def source_tree_sha256(root: str = "encodings") -> str:
    """sha256 over 'path NUL sha256 LF' lines of the module's static dltrain import closure (holdout.py convention)."""
    return common.sha256_bytes("".join(f"{p}\0{d}\n" for p, d in _closure(root)).encode("utf-8"))


# ----------------------------------------------------------------------------------------------- producer modules
def _a0():
    """The A0 producer: dltrain/metafeatures.py (the lane file) or dltrain/a0.py (the spec name)."""
    here = Path(__file__).resolve().parent
    if (here / "metafeatures.py").is_file():
        from dltrain import metafeatures as mod
    elif (here / "a0.py").is_file():
        from dltrain import a0 as mod
    else:
        raise Blocked("a0_producer_absent", "neither dltrain/metafeatures.py nor dltrain/a0.py exists")
    return mod


def _norm():
    from dltrain import normalizers
    return normalizers


def _a0_error_type():
    """The producer's A0Error class (a bug: STOP), or an empty tuple when the producer is absent."""
    try:
        return _a0().A0Error
    except EncodingError:
        return ()


def _a0_producer_tree(a0m) -> dict:
    name = a0m.__name__.split(".")[-1]
    fn = getattr(a0m, "source_tree_sha256", None)
    sha = fn() if callable(fn) else source_tree_sha256(name)
    return {"module": a0m.__name__, "source_tree_sha256": sha, "files": source_tree(name)}


# ----------------------------------------------------------------------------------------------- GET ledger
class GetLedger:
    """Per-kind GET caps plus combined caps, checked before every GET (BudgetExhausted). One entry per GET with the
    expected and received sha256/bytes."""
    KINDS = ("jtt1", "release", "outcome", "array")

    def __init__(self, caps: dict, combined: dict | None = None):
        self.caps = {k: int(caps.get(k, 0)) for k in self.KINDS}
        self.combined = {name: (tuple(kinds), int(cap)) for name, (kinds, cap) in (combined or {}).items()}
        self.entries = []

    def used(self, kind=None) -> int:
        return sum(1 for e in self.entries if kind is None or e["kind"] == kind)

    def check(self, kind):
        if kind not in self.caps:
            raise ValueError(f"unknown GET kind {kind}")
        if self.used(kind) >= self.caps[kind]:
            raise BudgetExhausted(f"budget_exhausted:{kind}", f"cap {self.caps[kind]} reached before the next GET")
        for name, (kinds, cap) in self.combined.items():
            if kind in kinds and sum(self.used(k) for k in kinds) >= cap:
                raise BudgetExhausted(f"budget_exhausted:{name}", f"combined cap {cap} reached before the next GET")

    def open(self, kind, ref) -> dict:
        entry = {"kind": kind, "bucket": ref.get("bucket"), "key": ref.get("key"), "version_id": ref.get("version_id"),
                 "expected_sha256": ref.get("sha256"), "expected_bytes": ref.get("bytes"), "got_sha256": None,
                 "got_bytes": None, "got_version_id": None, "attempts": 1, "utc_us": common.utc_us(), "ok": False}
        self.entries.append(entry)
        return entry

    def doc(self) -> dict:
        used = {k: self.used(k) for k in self.KINDS}
        return {"schema": LEDGER_SCHEMA, "caps": dict(self.caps),
                "combined": {n: {"kinds": list(k), "cap": c, "used": sum(used[x] for x in k)} for n, (k, c) in self.combined.items()},
                "used": used, "gets": len(self.entries), "bytes": sum(e["got_bytes"] or 0 for e in self.entries),
                "all_sha_verified": all(e["ok"] for e in self.entries), "puts": 0,
                "attempts_note": "attempts counts get_exact calls; the lane read client retries up to 3 times inside one call",
                "entries": list(self.entries)}


def _read_client():
    """The guarded lane read client (s3io.make_client, writes=False): exact-VersionId GETs; PutObject is refused."""
    from dltrain import s3io
    return s3io.make_client()


class _Fetcher:
    """Exact-VersionId GETs under a ledger. The read client is created lazily, only when a GET is needed."""

    def __init__(self, ledger: GetLedger, cache_dir=None):
        self.ledger, self.cache_dir, self._client = ledger, cache_dir, None

    def client(self):
        if self._client is None:
            self._client = _read_client()
        return self._client

    def jtt1_cached(self, ref) -> bool:
        return self.cache_dir is not None and jtt1.cache_path(self.cache_dir, ref).exists()

    def jtt1(self, ref) -> bytes:
        """Verified raw JTT1 bytes through jtt1.load_raw: the cache, else one exact GET within the jtt1 cap."""
        if self.cache_dir is None:
            raise Refused("jtt1_cache_absent", "no --jtt1-cache given")
        if self.jtt1_cached(ref):
            return jtt1.load_raw(ref, self.cache_dir)
        self.ledger.check("jtt1")
        entry = self.ledger.open("jtt1", ref)
        counters = {}
        try:
            raw = jtt1.load_raw(ref, self.cache_dir, client=self.client(), ledger=counters)
        except Exception as err:
            entry.update(error=f"{type(err).__name__}: {err}", got_bytes=counters.get("bytes"))
            raise
        gz = _read_bytes(jtt1.cache_path(self.cache_dir, ref))
        entry.update(got_sha256=_sha(gz), got_bytes=len(gz), got_version_id=ref.get("version_id"),
                     ok=_sha(gz) == ref["sha256"] and len(gz) == int(ref["bytes"]))
        return raw

    def get(self, kind, ref) -> bytes:
        self.ledger.check(kind)
        entry = self.ledger.open(kind, ref)
        from dltrain import s3io
        try:
            body, meta = s3io.get_exact(self.client(), ref["key"], ref["version_id"], bucket=ref["bucket"])
        except Exception as err:
            entry["error"] = f"{type(err).__name__}: {err}"
            raise
        entry.update(got_sha256=_sha(body), got_bytes=len(body), got_version_id=(meta or {}).get("VersionId"))
        entry["ok"] = entry["got_sha256"] == ref.get("sha256") and entry["got_bytes"] == int(ref.get("bytes", -1))
        return body


# ----------------------------------------------------------------------------------------------- inputs
def load_partition_map(src):
    """programs._load_partition_map rules, reimplemented: a JSON object {64-hex context_sha256: {family_id,
    split_role}} with split_role in PARTITIONS + EXCLUDED_ROLES. Returns (map, pin)."""
    if isinstance(src, Mapping):
        path, data = None, dict(src)
        raw = common.canonical(data)
    else:
        path = Path(src)
        raw = _read_bytes(path)
        try:
            data = json.loads(raw)
        except ValueError as err:
            raise PartitionMapError("partition_map_unparseable", f"{path}: {err}") from None
    if not isinstance(data, dict):
        raise PartitionMapError("partition_map_malformed", "must be a JSON object context_sha256 -> {family_id, split_role}")
    out = {}
    for cid, v in data.items():
        fam = v.get("family_id") if isinstance(v, dict) else None
        role = v.get("split_role") if isinstance(v, dict) else None
        if not _is_hex64(cid) or not isinstance(fam, str) or not fam or role not in PARTITIONS + EXCLUDED_ROLES:
            raise PartitionMapError("partition_map_entry", f"{str(cid)[:16]} needs a 64-hex key, a family_id and a "
                                                           f"split_role in {PARTITIONS + EXCLUDED_ROLES}")
        out[cid] = {"family_id": fam, "split_role": role}
    return out, {"path": None if path is None else str(path), "bytes": len(raw), "sha256": _sha(raw), "entries": len(out)}


def resolve_map_kind(path, pmap, kind, binding=None, map_sha256=None) -> str:
    """kind is 'fake' or 'p3'. 'p3' is a positive property: the map must carry no fake family ids or fake sidecar AND
    hold a bind_partition_map record for this map's sha256 (roles recomputed from the bound P3 holdout.json); 'not
    fake-looking' is not enough. map_sha256 defaults to the sha256 of the map file (or of the canonical mapping)."""
    if kind not in ("fake", "p3"):
        raise Refused("partition_map_kind", f"{kind!r} is not fake|p3")
    fake = any(v["family_id"].startswith("fake-fam-") for v in pmap.values())
    if path is not None:
        meta = Path(str(path) + ".meta.json")
        if meta.is_file():
            try:
                fake = fake or json.loads(_read_bytes(meta)).get("kind") == "fake"
            except (ValueError, AttributeError):
                pass
    if kind == "p3" and fake:
        raise Refused("fake_map_labelled_p3", "the partition map carries fake-fam- ids or a fake sidecar")
    if kind == "p3" and not (isinstance(binding, Mapping) and binding.get("status") == "BOUND"
                             and binding.get("record", {}).get("partition_map_sha256")
                             == (map_sha256 if map_sha256 is not None else _map_sha(pmap, path))):
        raise Refused("p3_map_unbound", "kind p3 needs a partition-map binding (--holdout, --holdout-binding, --p3-receipt)")
    if kind == "fake" and binding is not None:
        raise Refused("binding_with_fake_map", "a fake partition map cannot carry a P3 binding")
    return kind


def _map_sha(pmap, path) -> str:
    """The partition map's pinned sha256: of the file bytes when it came from a file, else of its canonical form."""
    return _sha(_read_bytes(path)) if path is not None else _sha(common.canonical(dict(pmap)))


def load_epoch(path):
    raw = _read_bytes(path)
    doc = _loads(raw, "epoch")
    if not isinstance(doc, dict) or not isinstance(doc.get("contexts"), list):
        raise Refused("epoch_malformed", f"{path} has no contexts list")
    seen = set()
    for c in doc["contexts"]:
        cid = c.get("context_sha256") if isinstance(c, dict) else None
        if not _is_hex64(cid) or cid in seen:
            raise Refused("epoch_context_malformed", f"context {str(cid)[:16]} is malformed or repeated")
        seen.add(cid)
    return doc, {"path": str(path), "sha256": _sha(raw), "bytes": len(raw), "experiment_id": doc.get("experiment_id"),
                 "epoch": doc.get("epoch"), "contexts": len(doc["contexts"])}


def included_contexts(epoch_doc) -> list:
    return [c for c in epoch_doc["contexts"] if c.get("status") == "included"]


def load_contexts_file(path):
    """A sample file ({contexts: [...]}) or a JSON list of context_sha256 strings (or {context_sha256} objects)."""
    raw = _read_bytes(path)
    doc = _loads(raw, "contexts")
    items = doc.get("contexts") if isinstance(doc, dict) else doc
    if not isinstance(items, list):
        raise Refused("contexts_malformed", f"{path}: no contexts list")
    ids = []
    for it in items:
        cid = it.get("context_sha256") if isinstance(it, dict) else it
        if not _is_hex64(cid):
            raise Refused("contexts_malformed", f"{path}: entry {str(cid)[:16]} is not a 64-hex context_sha256")
        ids.append(cid)
    if len(set(ids)) != len(ids):
        raise Refused("contexts_duplicated", f"{path} lists a context twice")
    return ids, {"kind": "subset", "path": str(path), "sha256": _sha(raw), "n": len(ids),
                 "contexts_sha256": _sha(common.canonical(sorted(ids)))}


def load_memberships(path):
    """RUN/encodings/memberships.jsonl rows (fields extracted at fetch time plus membership_sha256)."""
    raw = _read_bytes(path)
    rows = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError as err:
            raise Refused("memberships_unparseable", str(err)) from None
        cid = rec.get("context_sha256") if isinstance(rec, dict) else None
        if not _is_hex64(cid):
            raise Refused("memberships_malformed", "a row lacks a 64-hex context_sha256")
        if cid in rows and rows[cid] != rec:
            raise Stop("membership_cache_conflict", f"{cid[:16]} has two different rows")
        rows[cid] = rec
    return rows, {"path": str(path), "sha256": _sha(raw), "bytes": len(raw), "rows": len(rows)}


def link_memberships(contexts, rows, require=True) -> dict:
    """Each context's cached row must carry the epoch's membership_ref.sha256 (else STOP cache_stale)."""
    out = {}
    for c in contexts:
        cid = c["context_sha256"]
        rec = rows.get(cid)
        if rec is None:
            if require:
                raise Stop("membership_cache_incomplete", f"{cid[:16]} has no cached membership row")
            continue
        if rec.get("membership_sha256") != (c.get("membership_ref") or {}).get("sha256"):
            raise Stop("cache_stale", f"{cid[:16]}: cached membership_sha256 differs from the epoch membership_ref")
        out[cid] = rec
    return out


def load_holdout_binding(path) -> dict:
    raw = _read_bytes(path)
    doc = _loads(raw, "holdout_binding")
    s3ref = None
    if isinstance(doc, dict):
        for cand in (doc, doc.get("ref"), doc.get("binding_ref"), doc.get("s3ref")):
            if isinstance(cand, dict) and all(k in cand for k in PIN_FIELDS):
                s3ref = {k: cand[k] for k in PIN_FIELDS}
                break
    return {"path": str(path), "sha256": _sha(raw), "bytes": len(raw), "s3ref": s3ref}


def bound_shas(doc, name) -> set:
    """Every sha256 a receipt binds for `name` (epoch, partition_map, holdout): <name>_sha256, <name>.sha256 or
    inputs.<name>.sha256 (the corpus_build._receipt_ok convention, plus the inputs block)."""
    found = set()
    if not isinstance(doc, Mapping):
        return found
    if isinstance(doc.get(f"{name}_sha256"), str):
        found.add(doc[f"{name}_sha256"])
    for holder in (doc, doc.get("inputs") if isinstance(doc.get("inputs"), Mapping) else {}):
        e = holder.get(name)
        if isinstance(e, Mapping) and isinstance(e.get("sha256"), str):
            found.add(e["sha256"])
    return found


def _receipt_status_pass(doc) -> bool:
    return doc.get("verdict", doc.get("status")) == "PASS" or doc.get("pass") is True


def p5_receipt_pass(path, epoch_sha256, map_sha256=None) -> tuple:
    """(ok, why): a P5 receipt with status/verdict PASS that binds exactly this epoch sha256 and, when map_sha256 is
    given, exactly this partition map (P5's own output gate binds its map the same way)."""
    try:
        doc = json.loads(_read_bytes(path))
    except (OSError, ValueError) as err:
        return False, f"unreadable: {err}"
    if not isinstance(doc, dict):
        return False, "not a JSON object"
    status = doc.get("status", doc.get("verdict"))
    if status != "PASS":
        return False, f"status {status!r}"
    packet = doc.get("packet")
    if packet is not None and not str(packet).startswith("P5"):
        return False, f"packet {packet!r}"
    if bound_shas(doc, "epoch") != {epoch_sha256}:
        return False, "epoch sha256 differs"
    if map_sha256 is not None and bound_shas(doc, "partition_map") != {map_sha256}:
        return False, "partition map sha256 differs (or is not bound)"
    return True, "PASS"


def p3_receipt_pass(path, map_sha256, holdout_sha256=None) -> tuple:
    """(ok, why): a P3 receipt with verdict/status PASS (or pass: true) that binds exactly this partition map and, when
    it names a holdout, exactly this holdout.json."""
    try:
        doc = json.loads(_read_bytes(path))
    except (OSError, ValueError) as err:
        return False, f"unreadable: {err}"
    if not isinstance(doc, dict):
        return False, "not a JSON object"
    if not _receipt_status_pass(doc):
        return False, f"status {doc.get('verdict', doc.get('status'))!r}"
    if bound_shas(doc, "partition_map") != {map_sha256}:
        return False, "partition map sha256 differs (or is not bound)"
    named = bound_shas(doc, "holdout")
    if holdout_sha256 is not None and named and named != {holdout_sha256}:
        return False, "holdout sha256 differs"
    return True, "PASS"


def p3_roles(holdout_doc, pmap) -> tuple:
    """(roles recomputed for every map context from holdout.json, problems). The family comes from the map; its role
    from holdout.assignment, and a train context listed in holdout.diag is diag."""
    assignment = holdout_doc.get("assignment")
    diag = holdout_doc.get("diag")
    if not isinstance(assignment, dict) or not isinstance(diag, list) or not all(_is_hex64(c) for c in diag):
        return {}, [{"problem": "holdout_malformed", "detail": "assignment must be an object and diag a list of 64-hex ids"}]
    diag_set = set(diag)
    roles, problems = {}, []
    for cid in sorted(pmap):
        fam, mapped = pmap[cid]["family_id"], pmap[cid]["split_role"]
        a = assignment.get(fam)
        if a not in P3_ASSIGNMENT_ROLES:
            problems.append({"context": cid, "family_id": fam, "problem": "family_unassigned", "assignment": a})
            continue
        role = P3_ASSIGNMENT_ROLES[a]
        if cid in diag_set:
            if role != "train_fit":
                problems.append({"context": cid, "family_id": fam, "problem": "diag_outside_train_family", "assignment": a})
                continue
            role = "diag"
        roles[cid] = role
        if role != mapped:
            problems.append({"context": cid, "family_id": fam, "problem": "role_differs", "map": mapped, "holdout": role})
    return roles, problems


def bind_partition_map(pmap, map_pin, epoch_pin, *, holdout, holdout_binding, p3_receipt, p5_receipt=None) -> dict:
    """Bind a partition map to the P3 freeze (TOL-01). Needs holdout.json, its binding document and a P3 PASS receipt
    binding this map's sha256; a P5 receipt, when given, must PASS for this epoch and this map. Every context's role
    is recomputed from holdout.json and must equal the map's. Absent or non-PASS inputs are BLOCKED; contradictions
    (holdout bytes vs the binding, a role difference) are a STOP. Returns {status: BOUND, sha256, record, pins}; the
    record holds content hashes only (no paths), and its sha256 goes into the normalizer header."""
    missing = [name for name, v in (("--holdout", holdout), ("--holdout-binding", holdout_binding),
                                    ("--p3-receipt", p3_receipt)) if not v]
    if missing:
        raise Blocked("map_binding_inputs", f"a p3 partition map needs {', '.join(missing)}")
    hraw = _read_bytes(holdout)
    hdoc = _loads(hraw, "holdout")
    hsha = _sha(hraw)
    if not isinstance(hdoc, dict) or hdoc.get("schema") != HOLDOUT_SCHEMA or hdoc.get("experiment_id") != EXPERIMENT_ID:
        raise Refused("holdout_schema", f"holdout.json must be {HOLDOUT_SCHEMA} for {EXPERIMENT_ID}")
    braw = _read_bytes(holdout_binding)
    bdoc = _loads(braw, "holdout_binding")
    if not isinstance(bdoc, dict) or bdoc.get("schema") != HOLDOUT_BINDING_SCHEMA or bdoc.get("kind") != "holdout" \
            or bdoc.get("experiment_id") != EXPERIMENT_ID:
        raise Refused("holdout_binding_schema", f"the holdout binding must be {HOLDOUT_BINDING_SCHEMA} kind holdout")
    href = bdoc.get("holdout_ref")
    if not (isinstance(href, dict) and all(href.get(k) is not None for k in PIN_FIELDS)):
        raise Refused("holdout_binding_ref", "the binding has no complete holdout_ref")
    if href.get("sha256") != hsha or href.get("bytes") != len(hraw):
        raise Stop("holdout_binding_mismatch", f"holdout.json {hsha[:16]} ({len(hraw)} B) is not the bound holdout "
                                               f"{str(href.get('sha256'))[:16]} ({href.get('bytes')} B)")
    differing = [k for k in ("pool_sha256", "N", "k", "tau_hex", "tau_dev_hex") if k in bdoc and bdoc[k] != hdoc.get(k)]
    if differing:
        raise Stop("holdout_binding_mismatch", f"binding fields differ from holdout.json: {differing}")
    ok, why = p3_receipt_pass(p3_receipt, map_pin["sha256"], hsha)
    if not ok:
        raise Blocked("p3_receipt", f"P3 receipt not usable: {why}")
    p5_pin = None
    if p5_receipt:
        ok, why = p5_receipt_pass(p5_receipt, epoch_pin["sha256"], map_pin["sha256"])
        if not ok:
            raise Blocked("p5_receipt", f"P5 receipt not usable: {why}")
        p5raw = _read_bytes(p5_receipt)
        p5_pin = {"path": str(p5_receipt), "sha256": _sha(p5raw), "bytes": len(p5raw)}
    roles, problems = p3_roles(hdoc, pmap)
    if problems:
        raise Stop("partition_map_roles_differ", f"{len(problems)} map contexts differ from holdout.json "
                                                 f"(first {json.dumps(problems[0], sort_keys=True)[:300]})")
    p3raw = _read_bytes(p3_receipt)
    record = {"schema": MAP_BINDING_SCHEMA, "experiment_id": EXPERIMENT_ID, "partition_map_sha256": map_pin["sha256"],
              "epoch_sha256": epoch_pin["sha256"], "holdout_sha256": hsha, "holdout_bytes": len(hraw),
              "holdout_binding_sha256": _sha(braw), "holdout_ref": {k: href[k] for k in PIN_FIELDS},
              "p3_receipt_sha256": _sha(p3raw), "p5_receipt_sha256": None if p5_pin is None else p5_pin["sha256"],
              "contexts": len(roles), "families": len({v["family_id"] for v in pmap.values()}),
              "roles": dict(sorted(collections.Counter(roles.values()).items())), "rule": MAP_BINDING_RULE}
    return {"status": "BOUND", "sha256": _sha(common.canonical(record)), "record": record,
            "pins": {"holdout": {"path": str(holdout), "sha256": hsha, "bytes": len(hraw)},
                     "holdout_binding": {"path": str(holdout_binding), "sha256": _sha(braw), "bytes": len(braw),
                                         "s3ref": load_holdout_binding(holdout_binding)["s3ref"]},
                     "p3_receipt": {"path": str(p3_receipt), "sha256": _sha(p3raw), "bytes": len(p3raw)},
                     "p5_receipt": p5_pin}}


def binding_manifest_entry(binding):
    """The manifests' inputs.partition_map.binding: the content record and its sha256 (null for a fake map)."""
    return None if binding is None else dict(binding["record"], sha256=binding["sha256"])


# --------------------------------------------------------------------------------- shared join/binding helpers
def output_join_key(x) -> tuple:
    return (x["identity"][0], x["identity"][2], x["fold"]["fold"], x["fold"].get("repeat", 0))


def context_join_key(c) -> tuple:
    return (c["dataset_hash"], c["sample_identity_sha256"], c["fold"], 0)


def index_join(outputs) -> dict:
    """Complete index outputs with an identity, keyed (identity[0], identity[2], fold.fold, fold.repeat)."""
    outs = collections.defaultdict(list)
    for x in outputs:
        if x.get("identity") and x.get("state") == "complete":
            outs[output_join_key(x)].append(x)
    return outs


def identity_prefix_ok(identity) -> bool:
    """Codex identity [dataset_hash, canonical_sha256, sample_identity_sha256]: dataset_hash is the canonical's 16-hex
    prefix (the join uses identity[0]; this binds identity[1] to it)."""
    try:
        return isinstance(identity[0], str) and len(identity[0]) == 16 and str(identity[1])[:16] == identity[0]
    except (IndexError, TypeError):
        return False


def membership_disagrees(mem, c) -> bool:
    return (mem["sample_identity_sha256"] != c["sample_identity_sha256"] or mem["outer_fold"] != c["fold"]
            or mem["dataset_hash"] != c["dataset_hash"])


def fold_binds_membership(f, mem) -> bool:
    """Exact equality of the outer train/test index sha256 and row counts (the map-audit binding rule)."""
    return (f["train_indices_sha256"] == mem["outer_train_indices_sha256"] and f["test_indices_sha256"] == mem["outer_test_indices_sha256"]
            and f["train_rows"] == mem["outer_train_rows"] and f["test_rows"] == mem["outer_test_rows"])


def le_u64_sha256(indices) -> str:
    """sha256 of b''.join(i.to_bytes(8, 'little') for i in indices), computed through numpy."""
    arr = np.asarray(indices)
    if arr.ndim != 1 or (arr.size and arr.dtype.kind not in "iu"):
        raise ValueError("indices must be a flat integer list")
    if arr.size and int(arr.min()) < 0:
        raise ValueError("negative index")
    return _sha(arr.astype("<u8").tobytes())


def train_index_ok(fold) -> bool:
    """LE u64 sha256 recompute; len == train_rows; strictly increasing; 0 <= index < train_rows + test_rows."""
    try:
        idx = fold["train_indices"]
        if not isinstance(idx, list) or le_u64_sha256(idx) != fold["train_indices_sha256"]:
            return False
        n_tr, n_te = fold["train_rows"], fold["test_rows"]
        if not (_is_int(n_tr) and _is_int(n_te)) or len(idx) != n_tr:
            return False
        if idx:
            arr = np.asarray(idx, dtype=np.int64)
            if np.any(np.diff(arr) <= 0) or int(arr[0]) < 0 or int(arr[-1]) >= n_tr + n_te:
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def codex_canonical(obj) -> bytes:
    """Codex's canonical form (custody.canonical): sorted keys, compact separators, NaN refused and the json default
    ensure_ascii. Never common.canonical, which writes non-ASCII characters raw."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def codex_job_id(dataset_entry, fold, component, packet) -> str:
    """worker.job_id: sha256 of the Codex-canonical job preimage."""
    return _sha(codex_canonical({
        "schema": JOB_SCHEMA, "identity": dataset_entry["identity"], "manifest_ref": dataset_entry["manifest_ref"],
        "train_indices_sha256": fold["train_indices_sha256"], "train_row_ids_sha256": fold["train_row_ids_sha256"],
        "fold": fold["fold"], "component": component, "model_refs": packet["model_refs"],
        "encoder_source_sha256": packet["source_pins"]["gpu_encoders.py"]["sha256"]}))


def _pin_equal(a, b) -> bool:
    return (isinstance(a, dict) and isinstance(b, dict) and all(a.get(k) is not None for k in PIN_FIELDS)
            and all(a.get(k) == b.get(k) for k in PIN_FIELDS))


def _inventory_entry(inventory, section, inv_id):
    for e in inventory.get(section) or []:
        if isinstance(e, dict) and e.get("id") == inv_id:
            return e
    return {}


def component_license(inventory, comp) -> dict:
    """The inventory's encoder licence block of a component (DL-D3a), copied; unresolved or unrecorded otherwise."""
    src = COMPONENT_LICENSE_SOURCE[comp]
    if src is None:
        return dict(UNRESOLVED_LICENSE)
    lic = _inventory_entry(inventory, *src).get("license")
    if not isinstance(lic, dict):
        return {"class": "unrecorded", "redistribution": "unknown", "policy_eligible": False,
                "note": f"{src[0]}[{src[1]}] carries no licence block"}
    return json.loads(json.dumps(lic))


def research_only(comp, license_block) -> bool:
    """ResearchOnly (never served or redistributed): a licence that is not policy-eligible, or a component whose
    eligible arms are all ResearchOnly arms (A2, A3, LB-replica)."""
    arms = ARMS_ELIGIBLE[comp]
    return license_block.get("policy_eligible") is not True or (bool(arms) and all(a in RESEARCH_ONLY_ARMS for a in arms))


def component_pins(packet, inventory) -> dict:
    """Each component's packet.model_refs pins against encoding-inventory.json (bucket, key, version_id, sha256, bytes),
    with the component's encoder licence block and its research_only flag."""
    model_refs = packet.get("model_refs") or {}
    out = {}
    for comp in COMPONENTS:
        spec = COMPONENT_PINS[comp]
        lic = component_license(inventory, comp)
        if spec is None:
            out[comp] = {"status": "unpinned", "pins": [], "required_pin": dict(TABPFN35_CHECKPOINT),
                         "reason": "no pin in encoding-inventory.json or packet.model_refs; Codex worker.py checks the "
                                   "checkpoint only at runtime; an owner-approved inventory addendum must pin it",
                         "encoder_license": lic, "research_only": research_only(comp, lic)}
            continue
        pins, ok = [], True
        for mr_key, section, inv_id, field in spec:
            got = model_refs.get(mr_key)
            want = _inventory_entry(inventory, section, inv_id).get(field)
            eq = _pin_equal(got, want)
            ok = ok and eq
            pins.append({"model_ref": mr_key, "inventory": f"{section}[{inv_id}].{field}", "equal": eq,
                         "ref": {k: (got or {}).get(k) for k in PIN_FIELDS}})
        out[comp] = {"status": "pinned" if ok else "mismatch", "pins": pins, "encoder_license": lic,
                     "research_only": research_only(comp, lic)}
    return out


def rights_for(packet_entry) -> dict:
    """P2 registry rights are not available yet; the packet licence status stands in (source codex_packet)."""
    if packet_entry is None:
        return {"rights_status": "unknown", "source": "pending_P2"}
    status = (packet_entry.get("license") or {}).get("status") or "unknown"
    return {"rights_status": status, "source": "codex_packet"}


# ------------------------------------------------------------------------------------------------- map-audit
def load_index(path, sha256):
    raw = Path(path).read_bytes()
    if common.sha256_bytes(raw) != sha256:
        raise RuntimeError("index sha256 mismatch")
    return json.loads(raw)


def fetch_memberships(contexts, cache_path, threads=8):
    """Exact-version GETs of each context's membership object (cached locally as JSONL; sha256 checked). The read
    client is built only when some context is uncached, so a complete cache runs with no S3 client or credentials."""
    from dltrain import s3io
    cache = {}
    cp = Path(cache_path)
    if cp.exists():
        for line in cp.read_text().splitlines():
            rec = json.loads(line)
            cache[rec["context_sha256"]] = rec
    pending_contexts = [c for c in contexts if c["context_sha256"] not in cache]
    client = s3io.make_client() if pending_contexts else None

    def one(c):
        m = c["membership_ref"]
        body, _ = s3io.get_exact(client, m["key"], m["version_id"], bucket=m["bucket"])
        if common.sha256_bytes(body) != m["sha256"] or len(body) != m["bytes"]:
            raise RuntimeError(f"membership sha/bytes mismatch for {c['context_sha256']}")
        d = json.loads(body)
        return {"context_sha256": c["context_sha256"], "dataset_hash": d.get("dataset_hash"), "outer_fold": d.get("outer_fold"),
                "outer_repeat": d.get("outer_repeat"), "sample_identity_sha256": d.get("sample_identity_sha256"),
                "outer_train_indices_sha256": d.get("outer_train_indices_sha256"), "outer_train_rows": d.get("outer_train_rows"),
                "outer_test_indices_sha256": d.get("outer_test_indices_sha256"), "outer_test_rows": d.get("outer_test_rows"),
                "membership_sha256": m["sha256"]}

    if pending_contexts:
        with cf.ThreadPoolExecutor(threads) as ex, open(cp, "a") as f:
            for rec in ex.map(one, pending_contexts):
                cache[rec["context_sha256"]] = rec
                f.write(json.dumps(rec, sort_keys=True) + "\n")
    return cache, len(pending_contexts)


def map_audit(epoch_path, index_path, index_sha256, cache_path):
    e = json.loads(Path(epoch_path).read_text())
    contexts = [c for c in e["contexts"] if c["status"] == "included"]
    index = load_index(index_path, index_sha256)
    outs = index_join(index["outputs"])
    members, fetched = fetch_memberships(contexts, cache_path)
    rows, reasons = [], collections.Counter()
    comp_cov = collections.Counter()
    for c in contexts:
        mem = members[c["context_sha256"]]
        problems = []
        if membership_disagrees(mem, c):
            problems.append("membership_disagrees_with_epoch")
        matched = {}
        for x in outs.get(context_join_key(c), []):
            ok = fold_binds_membership(x["fold"], mem)
            if c.get("canonical_sha256") and x["identity"][1] != c["canonical_sha256"]:
                ok = False
                problems.append(f"canonical_mismatch:{x['component']}")
            if not identity_prefix_ok(x["identity"]):
                ok = False
                problems.append(f"identity_prefix_mismatch:{x['component']}")
            if not ok:
                problems.append(f"train_or_test_rows_mismatch:{x['component']}")
                continue
            if x["component"] in matched:
                problems.append(f"duplicate_component:{x['component']}")
            matched[x["component"]] = {"job_id": x["job_id"], "array_ref": x["array_ref"], "outcome_ref": x["outcome_ref"]}
        for comp in matched:
            comp_cov[comp] += 1
        status = "complete" if set(matched) >= set(COMPONENTS) else ("partial" if matched else "none")
        reasons[status] += 1
        for p in problems:
            reasons[p.split(":")[0]] += 1
        rows.append({"context_sha256": c["context_sha256"], "dataset_hash": c["dataset_hash"], "fold": c["fold"], "cohort": c["cohort"],
                     "outer_train_indices_sha256": mem["outer_train_indices_sha256"], "status": status, "components": sorted(matched),
                     "problems": problems, "matched": matched})
    summary = {"included_contexts": len(contexts), "memberships_fetched_now": fetched, "status_counts": dict(reasons),
               "component_coverage": {k: comp_cov.get(k, 0) for k in COMPONENTS},
               "index": {"path": str(index_path), "sha256": index_sha256, "utc_us": index["utc_us"], "all_shards_terminal": index["all_shards_terminal"],
                         "outputs": len(index["outputs"]), "packet_ref": index.get("packet_ref")},
               "binding_rule": "join (dataset_hash, sample_identity_sha256, fold, repeat 0); require outer train/test index sha256 and row counts equal"}
    return summary, rows


# ------------------------------------------------------------------------------------------ A0 run (per context)
def _entry_base(c, role, fam, mem) -> dict:
    ref = c["input_ref"]
    return {"context_sha256": c["context_sha256"], "cohort": c["cohort"], "fold": c["fold"], "dataset_hash": c["dataset_hash"],
            "family_id": fam, "split_role": role, "rights_status": "unknown", "rights_source": "pending_P2",
            "input": {k: ref.get(k) for k in ("bucket", "key", "version_id", "sha256", "bytes", "raw_sha256", "raw_bytes")},
            "membership_sha256": (c.get("membership_ref") or {}).get("sha256"),
            "outer_train_indices_sha256": (mem or {}).get("outer_train_indices_sha256")}


def _array_rel(role, sha) -> str:
    return ("sealed/arrays/" if role == "heldout" else "arrays/") + f"{sha}.f32le"


def merges_under_shift(col, shifts=UPSTREAM_SHIFTS) -> bool:
    """True when some shift d maps two distinct finite values of the column to one: v -> (v + d) / (1 + d) is weakly
    monotone in floating point, so a merge shows as equal neighbours among the sorted distinct values."""
    u = np.unique(np.asarray(col, dtype=np.float64)[np.isfinite(col)])
    if u.size < 2:
        return False
    with np.errstate(over="ignore", invalid="ignore"):
        for d in shifts:
            m = (u + d) / (1.0 + d)
            if bool(np.any(m[1:] <= m[:-1])):
                return True
    return False


def upstream_exposure(X, y) -> dict:
    """Deterministic, train-only (decoded outer-train rows only) exposure of one context to UPSTREAM-SCALE-01.
    scale: columns (and y) whose distinct train values merge under a downward all-row minimum shift, so their A0 values
    can depend on the outer-test minimum (TOL-04). column_set: valid columns constant on the outer-train rows; if upstream
    encoding drops all-row-constant columns or one-hot encodes levels over all rows, such a column exists only because an
    outer-test row differs, and p / m_x_const_col_frac / x_active_col_frac carry that fact (TOL-05, rule unconfirmed)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cols = [j for j in range(X.shape[1]) if merges_under_shift(X[:, j])]
    y_exp = merges_under_shift(y)
    finite = np.isfinite(X).all(axis=0)
    const = [j for j in range(X.shape[1]) if finite[j] and X[:, j].min() == X[:, j].max()]
    return {"finding": UPSTREAM_FINDING, "shifts": list(UPSTREAM_SHIFTS),
            "scale": {"exposed": bool(cols) or y_exp, "y": y_exp, "x_columns_exposed": len(cols),
                      "x_columns": cols[:EXPOSURE_LIST_MAX]},
            "column_set": {"exposed": bool(const), "train_constant_columns": len(const),
                           "columns": const[:EXPOSURE_LIST_MAX]}}


def train_only_scope(exposure) -> str:
    """'jtt1_bytes', plus '; upstream_scale_exposed' and/or '; upstream_column_set_exposed' (UPSTREAM-SCALE-01)."""
    scope = TRAIN_ONLY_SCOPE
    if exposure["scale"]["exposed"]:
        scope += "; upstream_scale_exposed"
    if exposure["column_set"]["exposed"]:
        scope += "; upstream_column_set_exposed"
    return scope


def encode_context(a0m, c, role, fam, raw, mem=None):
    """(manifest entry, vector bytes | None) for one context. A refusal (A0Refused) is a missing context, never imputed.
    A present context is labelled train_only=STRUCTURAL_JTT1_TRAIN_BYTES with its scope and upstream exposure (computed
    from jtt1.train_view only); VERIFIED_BY_COUNTERFACTUAL is granted only by a PASS verify receipt."""
    h = jtt1.parse_header(raw)
    train_row_sha256 = jtt1.train_rows_sha256(raw)
    entry = _entry_base(c, role, fam, mem)
    entry.update(train_rows=h["n_train"], features=h["p"], train_row_sha256=train_row_sha256)
    unlabelled = {"train_only": None, "train_only_scope": None, "upstream_exposure": None}
    if mem is not None and (mem.get("outer_train_rows") != h["n_train"] or mem.get("outer_test_rows") != h["n_test"]):
        entry.update(status="missing", missing_reason="jtt1_membership_shape_mismatch", arrays=[], **unlabelled)
        return entry, None
    try:
        vec, meta = a0m.from_raw(raw)
    except a0m.A0Refused as err:
        entry.update(status="missing", missing_reason=str(err.reason), arrays=[], **unlabelled)
        return entry, None
    width = len(a0m.FEATURES)
    if not isinstance(vec, (bytes, bytearray)) or len(vec) != 4 * width:
        raise Stop("a0_vector_length", f"{c['context_sha256'][:16]}: {len(vec)} bytes != {4 * width}")
    if meta.get("train_row_sha256") != train_row_sha256 or meta.get("n_train") != h["n_train"] or meta.get("p") != h["p"]:
        raise Stop("a0_meta_mismatch", f"{c['context_sha256'][:16]}: producer meta differs from the pinned JTT1")
    vec = bytes(vec)
    vsha = _sha(vec)
    _h, _g, X, y = jtt1.train_view(raw)
    exposure = upstream_exposure(X, y)
    entry.update(status="present", missing_reason=None, train_only=A0_TRAIN_ONLY, train_only_scope=train_only_scope(exposure),
                 upstream_exposure=exposure,
                 arrays=[{"block_id": A0_BLOCK, "sha256": vsha, "bytes": len(vec), "dtype": "<f4", "shape": [width],
                          "path": _array_rel(role, vsha)}])
    return entry, vec


def fit_a0_normalizer(fit_entries, vectors, pmap, *, epoch_sha256, partition_map_sha256, partition_map_kind,
                      holdout_binding_sha256=None, partition_map_binding_sha256=None):
    """Refit the A0 block normalizer on the given train_fit rows (sorted by context_sha256)."""
    N, a0m = _norm(), _a0()
    rows = [N.FitRow(context_sha256=e["context_sha256"], family_id=e["family_id"], split_role=e["split_role"],
                     vector=np.frombuffer(vectors[e["context_sha256"]], dtype="<f4").astype(np.float64))
            for e in sorted(fit_entries, key=lambda e: e["context_sha256"])]
    return N.fit_block(A0_BLOCK, rows, partition_map=pmap, arm="A0", block_version=a0m.VERSION,
                       feature_list_sha256=a0m.FEATURE_LIST_SHA256, epoch_sha256=epoch_sha256,
                       partition_map_sha256=partition_map_sha256, partition_map_kind=partition_map_kind,
                       holdout_binding_sha256=holdout_binding_sha256,
                       partition_map_binding_sha256=partition_map_binding_sha256)


def map_family_closure(pmap, contexts) -> list:
    """Problems with the map's family closure for the train_fit contexts given: every context of a train_fit context's
    family must be train_fit or diag (diag is a context-level carve-out). Checked before anything is written (TOL-02)."""
    fam_roles = collections.defaultdict(set)
    for v in pmap.values():
        fam_roles[v["family_id"]].add(v["split_role"])
    bad = {}
    for c in contexts:
        v = pmap.get(c["context_sha256"] if isinstance(c, Mapping) else c)
        if v is not None and v["split_role"] == "train_fit":
            other = fam_roles[v["family_id"]] - {"train_fit", "diag"}
            if other:
                bad[v["family_id"]] = sorted(other)
    return [{"family_id": f, "other_roles": r} for f, r in sorted(bad.items())]


def _assert_run_fit_closure(n, pmap, present_train_fit):
    """The run-side fit closure (verify --fit-closure repeats it independently): normalizers.assert_fit_closure on the
    fitted normalizer before the normalizer or any manifest is written; a violation is a STOP."""
    N = _norm()
    try:
        return N.assert_fit_closure(n, pmap, present_train_fit)
    except N.NormalizerFitRefused as err:
        raise Stop("fit_closure", str(err)) from None


def select_fit_entries(entries_by_partition) -> list:
    """The only rows a normalizer is ever fit on: train_fit contexts whose vector is present."""
    return [e for e in entries_by_partition.get("train_fit", []) if e["status"] == "present"]


def fit_from_entries(entries_by_partition, vectors, pmap, **params):
    """Selection plus fit, shared by run and the verify poison test (so the test exercises the run's selection)."""
    return fit_a0_normalizer(select_fit_entries(entries_by_partition), vectors, pmap, **params)


def normalizer_bytes(n) -> tuple:
    jb, blob = _norm().to_bytes(n)
    return bytes(jb), bytes(blob)


def fit_set_hashes(contexts, families) -> tuple:
    """Independent recompute of the fit-set hashes (the verify side does not trust the normalizer module's helpers)."""
    return (_sha(common.canonical({"schema": FIT_CONTEXT_SET_SCHEMA, "contexts": sorted(contexts)})),
            _sha(common.canonical({"schema": FIT_FAMILY_SET_SCHEMA, "families": sorted(set(families))})))


def _manifest_rel(partition) -> str:
    return "sealed/heldout.json" if partition == "heldout" else f"manifests/{partition}.json"


def _validate(doc, schema_file):
    try:
        from jsonschema import Draft202012Validator
    except ImportError:  # the lane venv carries jsonschema; without it the manifests are still written
        return
    errors = sorted(Draft202012Validator(SCHEMAS[schema_file]).iter_errors(doc), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        raise Refused("schema_violation", f"{schema_file}: {'/'.join(map(str, e.path))}: {e.message}")


def build_manifests(arm, entries, norm_rec, abi, width_full, inputs, scope, block) -> dict:
    if arm in DEV_ONLY_ARMS:
        raise Stop("dev_only_arm_heldout", f"{arm} is dev only (G06): it never gets a heldout manifest")
    out = {}
    for p in PARTITIONS:
        ents = sorted(entries.get(p, []), key=lambda e: e["context_sha256"])
        missing = collections.Counter(e["missing_reason"] for e in ents if e["status"] == "missing")
        doc = {"schema": CONTEXT_ENCODING_SCHEMA, "experiment_id": EXPERIMENT_ID, "arm": arm, "partition": p, "abi_id": abi,
               "width_full": width_full, "inputs": inputs, "scope": scope, "blocks": [block], "normalizers": [norm_rec],
               "contexts": ents, "coverage": {"admitted": len(ents), "present": sum(1 for e in ents if e["status"] == "present"),
                                              "missing": dict(sorted(missing.items()))},
               "train_only_rule": TRAIN_ONLY_RULE}
        if p == "heldout":
            doc["encoder_weight_exposure"] = {fam: {} for fam in sorted({e["family_id"] for e in ents})}
        if p == "diag":
            doc["partition_note"] = DIAG_NOTE
        out[p] = doc
    return out


def production_gate_reasons(a, epoch_sha256) -> list:
    """The real-output gate shared by run and adopt (TOL-01, AB-1): a p3 map with every P3 binding input and a P5 PASS
    receipt for this epoch. bind_partition_map then checks their content (roles, holdout bytes, the map sha256 that
    the P3 and P5 receipts bind) once the map is read."""
    reasons = []
    if a.partition_map_kind != "p3":
        reasons.append("partition_map_kind is not p3 (a fake map never reaches a real output root)")
    for flag, value in (("--holdout", a.holdout), ("--holdout-binding", a.holdout_binding), ("--p3-receipt", a.p3_receipt)):
        if not value:
            reasons.append(f"no {flag} (P3)")
    if not a.p5_receipt:
        reasons.append("no --p5-receipt")
    else:
        ok, why = p5_receipt_pass(a.p5_receipt, epoch_sha256)
        if not ok:
            reasons.append(f"P5 receipt not usable: {why}")
    return reasons


def _run_output_gate(a, epoch_sha256) -> str:
    out = _real(a.out)
    if sample_root(out) is not None:
        return "sample"
    reasons = []
    if out != _real(RUN_ENCODINGS) / a.arm:
        reasons.append(f"OUT must lie under {RUN_TMP}/{SAMPLE_DIR_PREFIX}*/ or be exactly {RUN_ENCODINGS}/{a.arm}")
    reasons += production_gate_reasons(a, epoch_sha256)
    if reasons:
        raise Blocked("output_gate", "; ".join(reasons))
    return "production"


def _map_binding(a, pmap, map_pin, epoch_pin, mode):
    """bind_partition_map for a p3 map (always at the production gate); a fake map takes no binding inputs."""
    if a.partition_map_kind == "p3" or mode == "production":
        return bind_partition_map(pmap, map_pin, epoch_pin, holdout=a.holdout, holdout_binding=a.holdout_binding,
                                  p3_receipt=a.p3_receipt, p5_receipt=a.p5_receipt)
    if a.holdout or a.holdout_binding or a.p3_receipt:
        raise Refused("binding_with_fake_map", "--holdout, --holdout-binding and --p3-receipt bind a p3 map; a fake map "
                                               "takes none")
    return None


def _binding_inputs(binding) -> dict:
    """The input pins a binding contributes to receipts and adoption.json (empty for a fake map)."""
    if binding is None:
        return {"holdout_binding": None}
    pins = binding["pins"]
    return {"holdout": pins["holdout"], "holdout_binding": pins["holdout_binding"], "p3_receipt": pins["p3_receipt"],
            "p5_receipt": pins["p5_receipt"]}


def _run_a0(a) -> tuple:
    a0m = _a0()
    N = _norm()
    for name in ("epoch", "partition_map", "jtt1_cache", "out"):
        if not getattr(a, name):
            raise Refused("missing_argument", f"--{name.replace('_', '-')} is required for --arm A0")
    epoch_doc, epoch_pin = load_epoch(a.epoch)
    mode = _run_output_gate(a, epoch_pin["sha256"])
    out = _real(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    started = common.utc_us()
    guard = cpuguard.Guard(journal_path=out / "guard-journal.jsonl", max_pause_s=GUARD_MAX_PAUSE_S)
    ledger = GetLedger({"jtt1": a.jtt1_get_budget})
    fetch = _Fetcher(ledger, a.jtt1_cache)
    rec = {"status": None, "failure": None, "counts": None, "outputs": {}, "writes": [], "timing": {}}
    inputs_pin = {"epoch": epoch_pin}
    try:
        guard.wait()
        pmap, map_pin = load_partition_map(a.partition_map)
        binding = _map_binding(a, pmap, map_pin, epoch_pin, mode)
        kind = resolve_map_kind(a.partition_map, pmap, a.partition_map_kind, binding, map_pin["sha256"])
        inputs_pin["partition_map"] = dict(map_pin, kind=kind, binding=binding_manifest_entry(binding))
        inputs_pin.update(_binding_inputs(binding))
        hb = inputs_pin["holdout_binding"]
        if a.p5_receipt and binding is None:
            p5raw = _read_bytes(a.p5_receipt)
            inputs_pin["p5_receipt"] = {"path": str(a.p5_receipt), "sha256": _sha(p5raw), "bytes": len(p5raw)}
        included = included_contexts(epoch_doc)
        by_id = {c["context_sha256"]: c for c in included}
        if a.contexts:
            ids, scope = load_contexts_file(a.contexts)
            unknown = [cid for cid in ids if cid not in by_id]
            if unknown:
                raise Refused("contexts_not_included", f"{len(unknown)} --contexts entries are not included epoch contexts "
                                                       f"(first {unknown[0][:16]})")
            scope_ctx = [by_id[cid] for cid in ids]
            inputs_pin["contexts"] = scope
        else:
            scope_ctx, scope = included, {"kind": "all", "n": len(included)}
        unmapped = [c["context_sha256"] for c in scope_ctx if c["context_sha256"] not in pmap]
        if unmapped:
            raise Stop("partition_map_closure", f"{len(unmapped)} contexts have no partition-map entry (first {unmapped[0][:16]})")
        mem_path = a.memberships or (DEFAULT_MEMBERSHIPS if DEFAULT_MEMBERSHIPS.is_file() else None)
        mems = {}
        if mem_path:
            mem_rows, mem_pin = load_memberships(mem_path)
            inputs_pin["memberships"] = mem_pin
            mems = link_memberships(scope_ctx, mem_rows, require=False)
        excluded = collections.Counter()
        admitted = []
        for c in scope_ctx:
            role = pmap[c["context_sha256"]]["split_role"]
            if role in EXCLUDED_ROLES:
                excluded[role] += 1
            else:
                admitted.append(c)
        admitted.sort(key=lambda c: (PARTITIONS.index(pmap[c["context_sha256"]]["split_role"]), c["context_sha256"]))
        closure = map_family_closure(pmap, admitted)
        if closure:                                     # before any JTT1 read or write (TOL-02)
            raise Stop("fit_closure", f"{len(closure)} train_fit families of the scope also hold contexts with other roles "
                                      f"(first {json.dumps(closure[0], sort_keys=True)})")
        entries = {p: [] for p in PARTITIONS}
        vectors, load_us, a0_us, writes = {}, [], [], []
        for c in admitted:
            guard.tick()
            cid = c["context_sha256"]
            role, fam = pmap[cid]["split_role"], pmap[cid]["family_id"]
            t0 = time.perf_counter_ns()
            raw = fetch.jtt1(c["input_ref"])
            t1 = time.perf_counter_ns()
            entry, vec = encode_context(a0m, c, role, fam, raw, mems.get(cid))
            t2 = time.perf_counter_ns()
            del raw
            load_us.append((t1 - t0) / 1000.0)
            a0_us.append((t2 - t1) / 1000.0)
            if vec is not None:
                res = common.immutable_write(out / entry["arrays"][0]["path"], vec, mode=0o444)
                writes.append({"path": entry["arrays"][0]["path"], "existed": res["existed"]})
                vectors[cid] = vec
            entries[role].append(entry)
        n = fit_from_entries(entries, vectors, pmap, epoch_sha256=epoch_pin["sha256"], partition_map_sha256=map_pin["sha256"],
                             partition_map_kind=kind, holdout_binding_sha256=hb["sha256"] if hb else None,
                             partition_map_binding_sha256=binding["sha256"] if binding else None)
        # the run-side fit closure, before the normalizer or any manifest is written (TOL-02)
        _assert_run_fit_closure(n, pmap, sorted(e["context_sha256"] for e in select_fit_entries(entries)))
        jb, blob = normalizer_bytes(n)
        header = json.loads(jb)
        _validate(header, "jope.dl-normalizer.schema.json")
        nsha, bsha = _sha(jb), _sha(blob)
        if nsha != n.sha256 or bsha != n.blob_sha256:
            raise Stop("normalizer_bytes", "to_bytes differs from the fitted normalizer identity")
        wres = N.write(n, out / "normalizers")
        writes.append({"path": f"normalizers/{nsha}.json", "existed": wres["header"]["existed"]})
        writes.append({"path": f"normalizers/{bsha}.f32le", "existed": wres["blob"]["existed"]})
        norm_rec = n.manifest_entry()
        width_full = int(norm_rec["width_full"])
        abi = N.abi_id("A0", [n], {A0_BLOCK: width_full})
        producer_tree = _a0_producer_tree(a0m)
        block = {"block_id": A0_BLOCK, "producer": {"kind": "computed", "module": a0m.__name__, "version": a0m.VERSION,
                                                     "feature_list_sha256": a0m.FEATURE_LIST_SHA256,
                                                     "source_tree_sha256": producer_tree["source_tree_sha256"]},
                 "components": [], "dtype": "<f4", "shape": [len(a0m.FEATURES)], "pre_normalization": True}
        m_inputs = {"epoch": {"path": epoch_pin["path"], "sha256": epoch_pin["sha256"], "s3ref": None},
                    "partition_map": {"path": map_pin["path"], "sha256": map_pin["sha256"], "kind": kind,
                                      "binding": binding_manifest_entry(binding)},
                    "holdout_binding": hb, "adoption_receipt": None}
        manifests = build_manifests("A0", entries, norm_rec, abi, width_full, m_inputs, scope, block)
        man_refs = {}
        for p in PARTITIONS:
            _validate(manifests[p], "jope.dl-context-encoding.schema.json")
            data = common.canonical(manifests[p])
            res = common.immutable_write(out / _manifest_rel(p), data, mode=0o444)
            writes.append({"path": _manifest_rel(p), "existed": res["existed"]})
            man_refs[p] = {"path": _manifest_rel(p), "sha256": _sha(data), "bytes": len(data),
                           "contexts": len(manifests[p]["contexts"])}
        arm_doc = {"schema": ARM_SCHEMA, "experiment_id": EXPERIMENT_ID, "arm": "A0", "abi_id": abi, "width_full": width_full,
                   "blocks": [{"block_id": A0_BLOCK, "informative_dims": len(a0m.FEATURES), "width_full": width_full,
                               "normalizer_sha256": nsha, "producer": block["producer"]}],
                   "widths_full": {A0_BLOCK: width_full}, "inputs": m_inputs, "scope": scope,
                   "manifests": {p: man_refs[p] for p in OPEN_PARTITIONS},
                   "heldout": "sealed: referenced only by sealed/closure.json (P10-P12 after unsealing)",
                   "train_only_rule": TRAIN_ONLY_RULE}
        _validate(arm_doc, "jope.dl-encoding-arm.schema.json")
        arm_bytes = common.canonical(arm_doc)
        res = common.immutable_write(out / "arm.json", arm_bytes, mode=0o444)
        writes.append({"path": "arm.json", "existed": res["existed"]})
        sealed_doc = {"schema": SEALED_CLOSURE_SCHEMA, "experiment_id": EXPERIMENT_ID, "arm": "A0", "abi_id": abi,
                      "heldout_manifest": man_refs["heldout"], "arrays_dir": "sealed/arrays",
                      "rule": "run plans reference only the train_fit, dev and diag manifests; this closure is read by P10-P12 after unsealing"}
        sealed_bytes = common.canonical(sealed_doc)
        res = common.immutable_write(out / "sealed/closure.json", sealed_bytes, mode=0o444)
        writes.append({"path": "sealed/closure.json", "existed": res["existed"]})
        counts = {}
        for p in PARTITIONS:
            cov = manifests[p]["coverage"]
            counts[p] = {"admitted": cov["admitted"], "present": cov["present"], "missing": cov["missing"]}
        counts["excluded_by_role"] = {r: excluded.get(r, 0) for r in EXCLUDED_ROLES}
        counts["upstream_exposure"] = {p: {"scale": sum(1 for e in entries[p] if (e["upstream_exposure"] or {}).get("scale", {}).get("exposed")),
                                           "column_set": sum(1 for e in entries[p]
                                                             if (e["upstream_exposure"] or {}).get("column_set", {}).get("exposed"))}
                                       for p in PARTITIONS}
        missing_total = sum(sum(counts[p]["missing"].values()) for p in PARTITIONS)
        rec.update(counts=counts, writes=writes,
                   outputs={"manifests": {p: man_refs[p] for p in OPEN_PARTITIONS}, "sealed_closure": {"path": "sealed/closure.json",
                            "sha256": _sha(sealed_bytes), "bytes": len(sealed_bytes)},
                            "arm": {"path": "arm.json", "sha256": _sha(arm_bytes), "bytes": len(arm_bytes)},
                            "normalizer": dict(norm_rec, constant_dims=(header.get("dims") or {}).get("constant_dims"),
                                               floored_components=(header.get("dims") or {}).get("floored_components")),
                            "abi_id": abi, "arrays": len(vectors)},
                   timing={"load_us": _quantiles(load_us), "a0_us": _quantiles(a0_us)})
        rec["status"] = "PASS" if missing_total == 0 else "FAIL"
        if missing_total:
            rec["failure"] = {"reason": "coverage_incomplete", "detail": f"{missing_total} admitted contexts missing (never imputed)"}
        code = EXIT_PASS if missing_total == 0 else EXIT_FAIL
    except EncodingError as err:
        rec["status"], rec["failure"], code = err.status, {"reason": err.reason, "detail": err.detail}, err.exit_code
    except cpuguard.GuardTimeout as err:
        rec["status"], rec["failure"], code = "BLOCKED", {"reason": "guard_timeout", "detail": str(err)}, EXIT_BLOCKED
    except common.ImmutableConflict as err:
        rec["status"], rec["failure"], code = "STOP", {"reason": "immutable_conflict", "detail": str(err)}, EXIT_STOP
    except jtt1.JTTError as err:
        rec["status"], rec["failure"], code = "STOP", {"reason": "jtt1_refused", "detail": str(err)}, EXIT_STOP
    except a0m.A0Error as err:
        rec["status"], rec["failure"], code = "STOP", {"reason": "a0_error", "detail": str(err)}, EXIT_STOP
    except N.NormalizerError as err:
        rec["status"], rec["failure"], code = "FAIL", {"reason": f"normalizer:{type(err).__name__}", "detail": str(err)}, EXIT_FAIL
    except Exception as err:  # noqa: BLE001 (an unexpected error still leaves a FAIL receipt)
        rec["status"], rec["failure"], code = "FAIL", {"reason": f"error:{type(err).__name__}", "detail": str(err)}, EXIT_FAIL
    ledger_doc = ledger.doc()
    lres = common.write_receipt(out / f"ledger-{stamp}.json", ledger_doc)
    packet = "P6-A0-sample" if mode == "sample" else "P6-A0"
    receipt = {"schema": PACKET_RECEIPT_SCHEMA, "packet": packet, "experiment_id": EXPERIMENT_ID, "arm": "A0", "label": a.label,
               "mode": mode, "status": rec["status"], "verdict": rec["status"], "exit_code": code, "failure": rec["failure"],
               "claim": ("A0 meta-features computed on CPU from JTT1 outer-train bytes only (jtt1.train_view), labelled "
                         f"{A0_TRAIN_ONLY} ({TRAIN_ONLY_CLAIM}); {A0_TRAIN_ONLY_VERIFIED} needs a PASS verify receipt bound "
                         "to these manifests; normalizer refit on train_fit rows only, fit closure checked before the "
                         "normalizer and manifests were written; no GPU, no encoder inference, no S3 writes. Partition map "
                         f"kind {inputs_pin.get('partition_map', {}).get('kind', a.partition_map_kind)}"
                         " (a fake map makes every role-dependent result a demonstration only)."),
               "started_utc_us": started, "utc_us": common.utc_us(),
               "source_tree": {"encodings": {"sha256": source_tree_sha256("encodings"), "files": source_tree("encodings")},
                               "a0": _a0_producer_tree(a0m)},
               "runtime": runtime_identity(), "inputs": inputs_pin, "outputs": rec["outputs"], "writes": rec["writes"],
               "counts": rec["counts"],
               "jtt1_ledger": {"path": lres["path"], "sha256": lres["sha256"], "used": ledger_doc["used"], "caps": ledger_doc["caps"],
                               "gets": ledger_doc["gets"], "bytes": ledger_doc["bytes"], "all_sha_verified": ledger_doc["all_sha_verified"]},
               "guard": guard.summary(), "timing": dict(rec["timing"], peak_rss_bytes=peak_rss_bytes())}
    rres = common.write_receipt(out / f"receipt-{stamp}.json", receipt)
    summary = {"status": rec["status"], "exit_code": code, "packet": packet, "failure": rec["failure"], "counts": rec["counts"],
               "receipt": {"path": rres["path"], "sha256": rres["sha256"]}, "ledger": {"path": lres["path"], "gets": ledger_doc["gets"]},
               "outputs": rec["outputs"], "writes": rec["writes"]}
    return code, summary


def cmd_run(a) -> tuple:
    try:
        cpuguard.require_slice_limits()
    except Exception as err:  # noqa: BLE001 (any failure of the slice check blocks the launch)
        return EXIT_BLOCKED, {"status": "BLOCKED", "reason": "slice_limits", "detail": str(err)}
    if a.arm != "A0":
        return EXIT_BLOCKED, {"status": "BLOCKED", "arm": a.arm, "reasons": list(BLOCKED_ARM_REASONS[a.arm]),
                              "note": ARM_NOTES.get(a.arm)}
    try:
        return _run_a0(a)
    except EncodingError as err:
        return err.exit_code, {"status": err.status, "reason": err.reason, "detail": err.detail}


# ----------------------------------------------------------------------------------------------- perturbations
def _train_stop(raw) -> int:
    return jtt1.train_range(jtt1.parse_header(raw))[1]


def perturb_test_fill(raw, byte: int) -> bytes:
    """R1a/R1b: every byte at or after the train-range stop replaced by one constant byte. Built from raw[:stop] and the
    length only: the test region is never read (TOL-11). Any test byte a mutant reads differs from 0x00 or from 0xFF."""
    stop = _train_stop(raw)
    return bytes(raw[:stop]) + bytes([byte]) * (len(raw) - stop)


def perturb_test_removed(raw) -> bytes:
    """R4: the outer-test rows removed (header n_test := 0), built from the header and raw[:stop] only."""
    h = jtt1.parse_header(raw)
    stop = jtt1.train_range(h)[1]
    return jtt1.MAGIC + jtt1.HEADER.pack(h["n_train"], 0, h["p"], h["inner_folds"]) + bytes(raw[jtt1.HEADER_BYTES:stop])


def perturb_test_random(raw, rng: random.Random) -> bytes:
    """R2: the test region replaced by random finite float64 values."""
    stop = _train_stop(raw)
    n = (len(raw) - stop) // 8
    return bytes(raw[:stop]) + struct.pack(f"<{n}d", *(rng.uniform(-1e6, 1e6) for _ in range(n)))


def perturb_test_nan(raw) -> bytes:
    """R3: the test region replaced by repeated NaN/Inf bit patterns."""
    stop = _train_stop(raw)
    n = (len(raw) - stop) // 8
    pats = np.array(NAN_PATTERNS, dtype="<u8")
    return bytes(raw[:stop]) + np.resize(pats, n).astype("<u8").tobytes()


def perturb_after_header(raw, rng: random.Random) -> bytes:
    """Poison (a): inner labels and every train and test value after the header replaced (finite values)."""
    h = jtt1.parse_header(raw)
    start = jtt1.HEADER_BYTES
    labels = bytes(rng.randrange(2) for _ in range(h["n_train"]))
    n = (len(raw) - start - h["n_train"]) // 8
    return bytes(raw[:start]) + labels + struct.pack(f"<{n}d", *(rng.uniform(-1e3, 1e3) for _ in range(n)))


def perturb_train_y0(raw) -> bytes:
    """Control: train y[0] += 10*s + 1 with s = labels.robust_scale(y_train)."""
    from dltrain import labels
    h = jtt1.parse_header(raw)
    start, _ = jtt1.train_range(h)
    y = jtt1.train_view(raw)[3]
    s = float(labels.robust_scale(y))
    off = start + 8 * h["p"]
    (y0,) = struct.unpack_from("<d", raw, off)
    buf = bytearray(raw)
    struct.pack_into("<d", buf, off, y0 + 10.0 * s + 1.0)
    return bytes(buf)


def _a0_outcome(a0m, raw) -> tuple:
    try:
        vec, _ = a0m.from_raw(raw)
        return ("ok", bytes(vec))
    except a0m.A0Refused as err:
        return ("refused", str(err.reason))


# ----------------------------------------------------------------------------------------------------- verify
def _load_manifests(out) -> tuple:
    """(manifests, pins): the four manifests and the sha256/bytes of the exact bytes read (bound by the promotion)."""
    mans, pins = {}, {}
    for p in PARTITIONS:
        path = out / _manifest_rel(p)
        if not path.is_file():
            raise Refused("manifest_absent", str(path))
        raw = _read_bytes(path)
        mans[p] = _loads(raw, f"manifest_{p}")
        pins[p] = {"path": _manifest_rel(p), "sha256": _sha(raw), "bytes": len(raw)}
    return mans, pins


def _latest_run_runtime(out):
    best = None
    for path in sorted(out.glob("receipt-*.json")):
        try:
            doc = json.loads(_read_bytes(path))
        except ValueError:
            continue
        if str(doc.get("packet", "")).startswith("P6-A0") and doc.get("status") in ("PASS", "FAIL"):
            best = doc.get("runtime")
    return best


def _public_id(cid, role) -> str:
    return "<sealed>" if role == "heldout" else cid


def _blob_arrays(header, blob) -> dict:
    out = {}
    for a in header.get("arrays") or []:
        count = int(np.prod(a["shape"])) if a["shape"] else 1
        out[a["name"]] = np.frombuffer(blob, dtype="<f4", count=count, offset=int(a["offset"])).reshape(a["shape"])
    return out


def verify_coverage(out, manifests, admitted_roles) -> dict:
    seen = collections.Counter()
    problems, missing, present = [], collections.Counter(), 0
    for p, m in manifests.items():
        for e in m.get("contexts") or []:
            cid = e["context_sha256"]
            seen[cid] += 1
            role = admitted_roles.get(cid)
            if e.get("split_role") != p or (role is not None and role != p):
                problems.append({"context": _public_id(cid, p), "problem": "partition_mismatch"})
            if e.get("status") == "present":
                ok = True
                for arr in e.get("arrays") or []:
                    path = out / arr["path"]
                    if not path.is_file():
                        problems.append({"context": _public_id(cid, p), "problem": "array_absent"})
                        ok = False
                        continue
                    data = _read_bytes(path)
                    if _sha(data) != arr["sha256"] or len(data) != arr["bytes"]:
                        problems.append({"context": _public_id(cid, p), "problem": "array_sha_or_bytes_mismatch"})
                        ok = False
                present += 1 if ok else 0
            else:
                missing[e.get("missing_reason") or "unknown"] += 1
                problems.append({"context": _public_id(cid, p), "problem": f"missing:{e.get('missing_reason')}"})
    dup = [c for c, k in seen.items() if k > 1]
    absent = sorted(set(admitted_roles) - set(seen))
    extra = sorted(set(seen) - set(admitted_roles))
    for c in dup:
        problems.append({"context": _public_id(c, admitted_roles.get(c)), "problem": "duplicated_across_manifests"})
    for c in absent:
        problems.append({"context": _public_id(c, admitted_roles.get(c)), "problem": "admitted_but_absent"})
    for c in extra:
        problems.append({"context": c if c in admitted_roles else "<outside-scope>", "problem": "listed_but_not_admitted"})
    ok = not problems and present == len(admitted_roles)
    return {"status": "PASS" if ok else "FAIL", "admitted": len(admitted_roles), "present": present,
            "missing": dict(sorted(missing.items())), "duplicates": len(dup), "absent": len(absent), "extra": len(extra),
            "problems": problems[:200], "problems_total": len(problems)}


def _stored_normalizer(out, manifests):
    rec = (manifests["train_fit"].get("normalizers") or [None])[0]
    if not rec:
        raise Refused("normalizer_record_absent", "train_fit manifest has no normalizer record")
    for p in PARTITIONS:
        if (manifests[p].get("normalizers") or [None])[0] != rec:
            raise Refused("normalizer_record_differs", f"{p} manifest names another normalizer")
    jb = _read_bytes(out / f"normalizers/{rec['normalizer_sha256']}.json")
    blob = _read_bytes(out / f"normalizers/{rec['blob_sha256']}.f32le")
    return rec, jb, blob


def verify_fit_closure(out, manifests, pmap, run_runtime) -> dict:
    N = _norm()
    res = {"status": "FAIL", "checks": {}}
    try:
        rec, jb, blob = _stored_normalizer(out, manifests)
    except (OSError, EncodingError) as err:
        res["error"] = str(err)
        return res
    checks = res["checks"]
    checks["normalizer_sha256"] = _sha(jb) == rec["normalizer_sha256"]
    checks["blob_sha256"] = _sha(blob) == rec["blob_sha256"]
    header = json.loads(jb)
    fit = header.get("fit") or {}
    try:
        n = N.from_bytes(jb, blob)
        checks["from_bytes"] = True
    except N.NormalizerError as err:
        checks["from_bytes"] = False
        res["error"] = f"from_bytes: {err}"
        return res
    tf = [e for e in manifests["train_fit"]["contexts"] if e["status"] == "present"]
    present_tf = sorted(e["context_sha256"] for e in tf)
    fams = sorted({pmap[c]["family_id"] for c in present_tf if c in pmap})
    ctx_hash, fam_hash = fit_set_hashes(present_tf, fams)
    checks["fit_context_set_sha256"] = ctx_hash == fit.get("fit_context_set_sha256") == rec.get("fit_context_set_sha256")
    checks["fit_family_set_sha256"] = fam_hash == fit.get("fit_family_set_sha256") == rec.get("fit_family_set_sha256")
    checks["fit_n_contexts"] = fit.get("n_contexts") == len(present_tf)
    checks["fit_n_families"] = fit.get("n_families") == len(fams)
    checks["fit_partition_role"] = fit.get("partition_role") == "train_fit"
    fam_roles = collections.defaultdict(set)
    for v in pmap.values():
        fam_roles[v["family_id"]].add(v["split_role"])
    # a train_fit family: every mapped context of it is train_fit or diag (diag is a context-level carve-out)
    checks["families_are_train_fit"] = all(fam_roles[f] <= {"train_fit", "diag"} for f in fams)
    checks["roles_are_train_fit"] = all(pmap.get(c, {}).get("split_role") == "train_fit" for c in present_tf)
    try:
        N.assert_fit_closure(n, pmap, present_tf)
        checks["assert_fit_closure"] = True
    except N.NormalizerError as err:
        checks["assert_fit_closure"] = False
        res["closure_error"] = str(err)
    vectors = {}
    for e in tf:
        data = _read_bytes(out / e["arrays"][0]["path"])
        vectors[e["context_sha256"]] = data
    try:
        refit = fit_a0_normalizer(tf, vectors, pmap, epoch_sha256=fit.get("epoch_sha256"),
                                  partition_map_sha256=fit.get("partition_map_sha256"),
                                  partition_map_kind=fit.get("partition_map_kind"),
                                  holdout_binding_sha256=fit.get("holdout_binding_sha256"),
                                  partition_map_binding_sha256=fit.get("partition_map_binding_sha256"))
        jb2, blob2 = normalizer_bytes(refit)
    except N.NormalizerError as err:
        checks["refit"] = False
        res["error"] = f"refit refused: {err}"
        return res
    identical = jb2 == jb and blob2 == blob
    same_runtime = run_runtime is not None and _runtime_key(run_runtime) == _runtime_key(runtime_identity())
    res["refit_identical_bytes"] = identical
    res["same_runtime_identity"] = same_runtime
    if same_runtime:
        checks["refit"] = identical
    else:
        h2 = json.loads(jb2)
        a1, a2 = _blob_arrays(header, blob), _blob_arrays(h2, blob2)
        tol = (all(k in a1 and k in a2 for k in ("mean", "scale", "eigenvalues"))
               and np.allclose(a1["mean"], a2["mean"], rtol=1e-9, atol=0) and np.allclose(a1["scale"], a2["scale"], rtol=1e-9, atol=0)
               and a1["eigenvalues"].shape == a2["eigenvalues"].shape
               and np.allclose(a1["eigenvalues"], a2["eigenvalues"], rtol=1e-6, atol=0))
        checks["refit"] = bool(identical or tol)
        res["refit_rule"] = "different runtime identity: mean/scale rtol 1e-9, eigenvalues rtol 1e-6"
    res["status"] = "PASS" if all(checks.values()) else "FAIL"
    res["fit_contexts"] = len(present_tf)
    res["fit_families"] = len(fams)
    return res


_DATA_FIELDS = ("status", "missing_reason", "train_rows", "features", "train_row_sha256", "arrays", "train_only",
                "train_only_scope", "upstream_exposure")


def _data_fields(entry) -> dict:
    return {k: entry.get(k) for k in _DATA_FIELDS}


def _poison_synthetic(base, pmap, tf):
    """Variant (e): POISON_SYNTHETIC dev, heldout and diag entries added (new ids; dev/heldout in new families, diag in
    existing train_fit families), each a copy of a present train_fit context's entry and vector. Returns (results, map)."""
    donors = [c for c in tf if base[c][1] is not None]
    results, pm = dict(base), dict(pmap)
    if not donors:
        return results, pm
    tf_fams = sorted({pmap[c]["family_id"] for c in tf})
    i = 0
    for role, count in POISON_SYNTHETIC.items():
        for j in range(count):
            cid = _sha(f"p6-poison-synthetic\0{role}\0{j}".encode())
            fam = tf_fams[j % len(tf_fams)] if role == "diag" else f"p6-poison-synthetic-fam:{role}:{j}"
            donor = donors[i % len(donors)]
            i += 1
            results[cid] = (dict(base[donor][0], context_sha256=cid, family_id=fam, split_role=role), base[donor][1])
            pm[cid] = {"family_id": fam, "split_role": role}
    return results, pm


POISON_ROTATION = {"dev": "heldout", "heldout": "diag", "diag": "dev"}


def verify_poison(a0m, out, manifests, ctx_by_id, pmap, admitted, raw_of, guard, scope_mode="all") -> dict:
    """(a) every heldout/dev/diag JTT1 perturbed after the header; (b) the test bytes of every train_fit context replaced
    by 0x00 and by 0xFF fills (built without reading them); (c) control: one train y of one train_fit context; (d) every
    non-train_fit context dropped from scope; (e) synthetic dev, heldout and diag entries added; (f) the roles of the
    non-train_fit contexts rotated (dev -> heldout -> diag -> dev). (a), (b), (d), (e) and (f) must leave the normalizer
    bytes identical (and, for (a) and (b), the train_fit arrays and rows), else STOP; (c) must move it, else FAIL. Each
    variant reruns the run's selection plus fit (fit_from_entries) over the contexts in scope. (a) covers diag JTT1
    files, not rows (DIAG_NOTE)."""
    _rec, jb_stored, blob_stored = _stored_normalizer(out, manifests)
    fit = (json.loads(jb_stored).get("fit") or {})
    params = {"epoch_sha256": fit.get("epoch_sha256"), "partition_map_sha256": fit.get("partition_map_sha256"),
              "partition_map_kind": fit.get("partition_map_kind"), "holdout_binding_sha256": fit.get("holdout_binding_sha256"),
              "partition_map_binding_sha256": fit.get("partition_map_binding_sha256")}
    tf = sorted(c for c in admitted if pmap[c]["split_role"] == "train_fit")
    others = sorted(c for c in admitted if pmap[c]["split_role"] != "train_fit")
    if scope_mode.startswith("sample:"):
        k = int(scope_mode.split(":", 1)[1])
        by_part = collections.defaultdict(list)
        for c in others:
            by_part[pmap[c]["split_role"]].append(c)
        others = sorted(c for p in by_part for c in by_part[p][:k])
        perturb_tf = tf[:k]
    else:
        perturb_tf = tf
    stored_rows = {e["context_sha256"]: e for e in manifests["train_fit"]["contexts"]}

    def encode(cid, raw):
        guard.tick()
        return encode_context(a0m, ctx_by_id[cid], pmap[cid]["split_role"], pmap[cid]["family_id"], raw)

    def fit_of(results, pm=pmap):
        entries = collections.defaultdict(list)
        for cid in sorted(results):
            entries[pm[cid]["split_role"]].append(results[cid][0])
        vecs = {cid: r[1] for cid, r in results.items() if r[1] is not None}
        return normalizer_bytes(fit_from_entries(entries, vecs, pm, **params))

    def same_tf(variant, base):
        return {"normalizer_identical": fit_of(variant) == base_norm,
                "train_fit_arrays_identical": all(variant[c][1] == base[c][1] for c in tf),
                "train_fit_rows_identical": all(_data_fields(variant[c][0]) == _data_fields(base[c][0]) for c in tf)}

    base = {c: encode(c, raw_of(c)) for c in tf + others}
    base_norm = fit_of(base)
    res = {"scope": scope_mode, "train_fit": len(tf), "perturbed_non_train_fit": len(others),
           "perturbed_train_fit_test_bytes": len(perturb_tf), "baseline_reproduces_stored": base_norm == (jb_stored, blob_stored),
           "baseline_rows_equal_stored": all(_data_fields(base[c][0]) == _data_fields(stored_rows.get(c, {})) for c in tf)}
    rng = random.Random(POISON_SEED)
    other_set, perturb_set = set(others), set(perturb_tf)
    variant_a = {c: (encode(c, perturb_after_header(raw_of(c), rng)) if c in other_set else base[c]) for c in tf + others}
    res["a"] = same_tf(variant_a, base)
    b_parts = []
    for fill in (0x00, 0xFF):
        variant_b = {c: (encode(c, perturb_test_fill(raw_of(c), fill)) if c in perturb_set else base[c]) for c in tf + others}
        b_parts.append(same_tf(variant_b, base))
    res["b"] = {k: all(part[k] for part in b_parts) for k in b_parts[0]}
    res["b_fills"] = ["0x00", "0xff"]
    present_tf = [c for c in tf if base[c][1] is not None]
    if present_tf:
        c0 = present_tf[0]
        variant_c = dict(base)
        variant_c[c0] = encode(c0, perturb_train_y0(raw_of(c0)))
        res["c"] = {"context": c0, "vector_differs": variant_c[c0][1] != base[c0][1], "normalizer_differs": fit_of(variant_c) != base_norm}
    else:
        res["c"] = {"context": None, "vector_differs": False, "normalizer_differs": False}
    res["d"] = {"normalizer_identical": fit_of({c: base[c] for c in tf}) == base_norm, "dropped": len(others)}
    syn, pm_e = _poison_synthetic(base, pmap, tf)
    res["e"] = {"normalizer_identical": fit_of(syn, pm_e) == base_norm, "added": len(syn) - len(base)}
    pm_f, variant_f = dict(pmap), dict(base)
    for c in others:
        role = POISON_ROTATION[pmap[c]["split_role"]]
        pm_f[c] = {"family_id": pmap[c]["family_id"], "split_role": role}
        variant_f[c] = (dict(base[c][0], split_role=role), base[c][1])
    res["f"] = {"normalizer_identical": fit_of(variant_f, pm_f) == base_norm, "relabelled": len(others)}
    stop = not (all(res["a"].values()) and all(res["b"].values())
                and all(res[v]["normalizer_identical"] for v in ("d", "e", "f")))
    res["status"] = "STOP" if stop else ("PASS" if res["c"]["normalizer_differs"] else "FAIL")
    res["diag_note"] = DIAG_NOTE
    return res


def counterfactual_inputs(raw) -> tuple:
    """The JTT1-byte counterfactual variants of one input, each built without reading the outer-test region (TOL-11)."""
    return tuple(zip(COUNTERFACTUAL_VARIANTS, (perturb_test_fill(raw, 0x00), perturb_test_fill(raw, 0xFF),
                                               perturb_test_random(raw, random.Random(COUNTERFACTUAL_SEED)),
                                               perturb_test_nan(raw), perturb_test_removed(raw))))


def verify_counterfactual(a0m, admitted, roles, raw_of, guard) -> dict:
    variants = COUNTERFACTUAL_VARIANTS
    runs = identical = controls = controls_differ = 0
    diffs, refused = [], collections.Counter()
    for cid in sorted(admitted):
        guard.tick()
        raw = raw_of(cid)
        base = _a0_outcome(a0m, raw)
        trs = jtt1.train_rows_sha256(raw)
        if base[0] == "refused":
            refused[base[1]] += 1
        for name, v in counterfactual_inputs(raw):
            runs += 1
            same = _a0_outcome(a0m, v) == base and jtt1.train_rows_sha256(v) == trs
            identical += same
            if not same:
                diffs.append({"context": _public_id(cid, roles.get(cid)), "variant": name})
        if base[0] == "ok":
            controls += 1
            controls_differ += _a0_outcome(a0m, perturb_train_y0(raw)) != base
    status = "STOP" if diffs else ("PASS" if controls_differ == controls else "FAIL")
    return {"status": status, "contexts": len(admitted), "runs": runs, "identical": identical, "differences": diffs[:200],
            "controls": controls, "controls_differ": controls_differ, "refused_baselines": dict(refused),
            "variants": list(variants), "seed": COUNTERFACTUAL_SEED}


def _distinct_finite(a) -> int:
    return int(np.unique(a[np.isfinite(a)]).size)


def _faithful(a, b) -> bool:
    """A float map of a column is faithful when it keeps finiteness cell by cell and keeps every distinct value
    distinct. A positive affine map is monotone non-decreasing after rounding, so equal distinct counts also mean the
    order is kept."""
    return bool(np.array_equal(np.isfinite(a), np.isfinite(b))) and _distinct_finite(a) == _distinct_finite(b)


AFFINE_MAP_KINDS = ("drawn", "no_shift", "pow2_scale", "identity")


def faithful_affine(col, scale, shift) -> tuple:
    """(mapped column, kind). The drawn map scale*col + shift when it is faithful; else the first faithful fallback:
    scale*col (no shift), then 2**round(log2(scale))*col (exact power-of-two scaling), then the identity. A float map
    that merges distinct values (a shift next to values like 1e-86, or y values of 1e-22 beside 1.0) has changed the
    input's information content, so comparing A0 across it would not test invariance."""
    col = np.asarray(col, dtype=np.float64)
    for kind, s, b in (("drawn", scale, shift), ("no_shift", scale, 0.0), ("pow2_scale", 2.0 ** round(math.log2(scale)), 0.0)):
        with np.errstate(over="ignore", invalid="ignore"):
            mapped = col * s + b
        if _faithful(col, mapped):
            return mapped, kind
    return col.copy(), "identity"


def verify_affine(a0m, admitted, raw_of, guard, roles=None) -> dict:
    """Random positive affine maps of each decoded X column and of y (the draw sequence of default_rng(20261005) over
    the sorted admitted contexts). Each drawn map is checked for faithfulness on that column; an unfaithful one is
    replaced by its first faithful fallback (faithful_affine). The status is the invariance claim under faithful maps.
    The upstream min-shift channel (UPSTREAM-SCALE-01) is not inferred from these random draws: verify_upstream_exposure
    tests it deterministically and lists the exposed contexts as OPEN."""
    roles = roles or {}
    rng = np.random.default_rng(AFFINE_SEED)
    names = list(a0m.FEATURES)
    maxd = np.zeros(len(names))
    bad_contexts, evaluated, refused = 0, 0, 0
    kinds = collections.Counter()
    adjustments, adjusted_contexts = [], 0
    for cid in sorted(admitted):
        guard.tick()
        _h, g, X, y = jtt1.train_view(raw_of(cid))
        try:
            f0 = np.asarray(a0m.features(X, y, g), dtype=np.float64)
        except a0m.A0Refused:
            refused += 1
            continue
        p = X.shape[1]
        sx = 10.0 ** rng.uniform(-2, 2, size=p)
        bx = sx * rng.uniform(-10, 10, size=p)
        sy = 10.0 ** rng.uniform(-2, 2)
        by = sy * rng.uniform(-10, 10)
        X1 = np.empty_like(X)
        notes = []
        for j in range(p):
            X1[:, j], kind = faithful_affine(X[:, j], sx[j], bx[j])
            kinds[kind] += 1
            if kind != "drawn":
                with np.errstate(over="ignore", invalid="ignore"):
                    merged = _distinct_finite(X[:, j] * sx[j] + bx[j])
                notes.append({"column": j, "kind": kind, "distinct": _distinct_finite(X[:, j]), "distinct_under_drawn_map": merged,
                              "scale": float(sx[j]), "shift_in_scale_units": float(bx[j] / sx[j])})
        y1, kind_y = faithful_affine(y, sy, by)
        kinds[kind_y] += 1
        if kind_y != "drawn":
            with np.errstate(over="ignore", invalid="ignore"):
                merged = _distinct_finite(y * sy + by)
            notes.append({"column": "y", "kind": kind_y, "distinct": _distinct_finite(y), "distinct_under_drawn_map": merged,
                          "scale": float(sy), "shift_in_scale_units": float(by / sy)})
        f1 = np.asarray(a0m.features(X1, y1, g), dtype=np.float64)
        d = np.abs(f1 - f0)
        maxd = np.maximum(maxd, d)
        evaluated += 1
        bad_contexts += int(np.any(d > AFFINE_ATOL + AFFINE_RTOL * np.abs(f0)))
        if notes:
            adjusted_contexts += 1
            adjustments.append({"context": _public_id(cid, roles.get(cid)), "maps": notes})
    exceed = [names[i] for i in range(len(names)) if maxd[i] > AFFINE_ATOL]
    return {"status": "PASS" if bad_contexts == 0 and evaluated > 0 else "FAIL", "contexts": evaluated,
            "refused": refused, "contexts_outside_tolerance": bad_contexts, "atol": AFFINE_ATOL, "rtol": AFFINE_RTOL,
            "max_abs_delta": float(maxd.max()) if maxd.size else 0.0,
            "per_feature_max_abs_delta": {names[i]: float(maxd[i]) for i in range(len(names))},
            "features_with_max_abs_delta_above_atol": exceed,
            "maps": "x_j -> a_j*x_j + a_j*u_j, a_j = 10**U(-2,2), u_j = U(-10,10); the same for y; numpy default_rng(20261005)",
            "faithfulness": {"rule": "a drawn map is used only if it keeps cell finiteness and every distinct value distinct; else "
                                     "the first faithful fallback of no_shift (a*x), pow2_scale (2**round(log2 a)*x), identity",
                             "map_kinds": {k: kinds.get(k, 0) for k in AFFINE_MAP_KINDS},
                             "contexts_with_adjusted_maps": adjusted_contexts, "adjustments": adjustments[:200]},
            "claim_scope": "invariance of the A0 function under faithful per-column positive affine maps; the upstream "
                           "min-shift channel is reported by the upstream_exposure check, not by these random draws"}


def verify_upstream_exposure(admitted, roles, raw_of, guard, manifests) -> tuple:
    """(result, {cid: channels}). The deterministic train-only exposure (upstream_exposure) of every admitted context,
    recomputed from jtt1.train_view and compared with each manifest entry's recorded exposure and train_only_scope
    (a difference FAILs). Exposed contexts are listed OPEN under UPSTREAM-SCALE-01: a PASS never covers them (TOL-04/05)."""
    recorded = {e["context_sha256"]: e for m in manifests.values() for e in m.get("contexts") or []}
    mismatches, listed, exposed = [], {"scale": [], "column_set": []}, {}
    for cid in sorted(admitted):
        guard.tick()
        _h, _g, X, y = jtt1.train_view(raw_of(cid))
        exp = upstream_exposure(X, y)
        entry = recorded.get(cid) or {}
        if entry.get("status") == "present" and (entry.get("upstream_exposure") != json.loads(json.dumps(exp))
                                                 or entry.get("train_only_scope") != train_only_scope(exp)):
            mismatches.append(_public_id(cid, roles.get(cid)))
        channels = [ch for ch in ("scale", "column_set") if exp[ch]["exposed"]]
        if channels:
            exposed[cid] = channels
        pid = _public_id(cid, roles.get(cid))
        if exp["scale"]["exposed"]:
            listed["scale"].append({"context": pid, "partition": roles.get(cid), "y": exp["scale"]["y"],
                                    "x_columns_exposed": exp["scale"]["x_columns_exposed"], "x_columns": exp["scale"]["x_columns"]})
        if exp["column_set"]["exposed"]:
            listed["column_set"].append({"context": pid, "partition": roles.get(cid),
                                         "train_constant_columns": exp["column_set"]["train_constant_columns"]})
    res = {"status": "PASS" if not mismatches else "FAIL", "finding": UPSTREAM_FINDING, "contexts": len(admitted),
           "shifts": list(UPSTREAM_SHIFTS), "manifest_mismatch_total": len(mismatches), "manifest_mismatches": mismatches[:200],
           "scale": {"status": "OPEN" if listed["scale"] else "NONE", "contexts": len(listed["scale"]), "listed": listed["scale"][:200]},
           "column_set": {"status": "OPEN" if listed["column_set"] else "NONE", "contexts": len(listed["column_set"]),
                          "listed": listed["column_set"][:200]},
           "rule": "scale: a column (or y) whose distinct outer-train values merge under v -> (v + d)/(1 + d) for a listed d "
                   "(a downward all-row minimum shift of d range units, as upstream min-max scaling over train and test rows "
                   "applies when an outer-test row holds the minimum); column_set: a valid column constant on the outer-train "
                   "rows (its presence can come from outer-test rows if upstream drops all-row-constant columns or one-hot "
                   "encodes levels over all rows; rule unconfirmed). Status PASS means the manifests record exactly this "
                   "exposure; exposed contexts stay OPEN under UPSTREAM-SCALE-01 and are never promoted"}
    return res, exposed


def column_set_audit(admitted, ctx_by_id, roles, raw_of, guard) -> dict:
    """The cross-fold column-set audit (TOL-05), decoding outer-train rows only: for admitted sibling folds of one
    sample (same dataset_hash and sample_identity_sha256, same p), a column constant on fold k's train rows that takes
    another finite value on a sibling's train rows must vary on fold k's outer-test rows (every row not in train(k) is
    in test(k)). Confirmed columns make A0 v1's column counts depend on outer-test rows if upstream drops all-row-constant
    columns. Reported, never gating; the epoch-wide audit needs every context's JTT1 in the cache."""
    groups = collections.defaultdict(list)
    for cid in sorted(admitted):
        c = ctx_by_id[cid]
        groups[(c["dataset_hash"], c.get("sample_identity_sha256"))].append(cid)
    siblings = {k: v for k, v in groups.items() if len(v) >= 2}
    checked = const_cols = confirmed = pairs = 0
    listed = []
    for key in sorted(siblings):
        cids = siblings[key]
        pairs += len(cids) * (len(cids) - 1) // 2
        views = {}
        for cid in cids:
            guard.tick()
            views[cid] = jtt1.train_view(raw_of(cid))[2]
        for cid in cids:
            X = views[cid]
            checked += 1
            hits = []
            finite = np.isfinite(X).all(axis=0)
            for j in np.flatnonzero(finite):
                col = X[:, j]
                if col.min() != col.max():
                    continue
                const_cols += 1
                for other in cids:
                    if other == cid or views[other].shape[1] != X.shape[1]:
                        continue
                    o = views[other][:, j]
                    o = o[np.isfinite(o)]
                    if o.size and bool(np.any(o != col[0])):
                        hits.append(int(j))
                        break
            if hits:
                confirmed += len(hits)
                listed.append({"context": _public_id(cid, roles.get(cid)), "partition": roles.get(cid),
                               "columns": hits[:EXPOSURE_LIST_MAX], "columns_confirmed": len(hits)})
    return {"status": "OPEN" if listed else "NONE", "finding": UPSTREAM_FINDING, "samples_with_siblings": len(siblings),
            "contexts_checked": checked, "sibling_pairs": pairs, "train_constant_columns": const_cols,
            "confirmed_test_varying_columns": confirmed, "contexts_confirmed": len(listed), "listed": listed[:200],
            "scope_note": "verify scope only (admitted contexts with sibling folds in scope); not gating"}


def verify_map_binding(a, out, manifests, pmap, map_pin, epoch_pin) -> dict:
    """A p3 manifest's partition-map binding, recomputed from the given P3 inputs and compared with the manifests and
    the normalizer header (TOL-01: verify no longer takes whatever map it is given). A fake map: NOT_APPLICABLE."""
    recorded = (manifests["train_fit"].get("inputs") or {}).get("partition_map") or {}
    if recorded.get("kind") != "p3":
        return {"status": "NOT_APPLICABLE", "kind": recorded.get("kind"),
                "note": "fake partition map: every role-dependent result (fit set, normalizer, poison) is a demonstration"}
    try:
        b = bind_partition_map(pmap, map_pin, epoch_pin, holdout=a.holdout, holdout_binding=a.holdout_binding,
                               p3_receipt=a.p3_receipt, p5_receipt=a.p5_receipt)
    except EncodingError as err:
        return {"status": "FAIL", "kind": "p3", "failure": {"reason": err.reason, "detail": err.detail}}
    try:
        _rec, jb, _blob = _stored_normalizer(out, manifests)
        header_sha = (json.loads(jb).get("fit") or {}).get("partition_map_binding_sha256")
    except (OSError, ValueError, EncodingError):
        header_sha = None
    entry = binding_manifest_entry(b)
    checks = {"manifests_record_this_binding": all(((m.get("inputs") or {}).get("partition_map") or {}).get("binding") == entry
                                                   for m in manifests.values()),
              "normalizer_header_binding": header_sha == b["sha256"]}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "kind": "p3", "binding_sha256": b["sha256"],
            "checks": checks}


def train_only_promotion(results, manifests, manifest_pins, roles, exposed, kind) -> dict:
    """VERIFIED_BY_COUNTERFACTUAL for the present contexts of exactly these manifest bytes, except the contexts with
    upstream exposure (OPEN under UPSTREAM-SCALE-01). Granted only when every PROMOTION_CHECKS result ran with PASS, the
    poison scope was all and the map binding is PASS (p3) or NOT_APPLICABLE (fake: a demonstration)."""
    missing = [k for k in PROMOTION_CHECKS if k not in results]
    failed = [k for k in PROMOTION_CHECKS if k in results and results[k].get("status") != "PASS"]
    if "poison_test" in results and results["poison_test"].get("scope") != "all":
        failed.append("poison_test:scope_not_all")
    if results.get("partition_map_binding", {}).get("status") not in ("PASS", "NOT_APPLICABLE"):
        failed.append("partition_map_binding")
    granted = not missing and not failed
    present = sorted(e["context_sha256"] for m in manifests.values() for e in m.get("contexts") or [] if e.get("status") == "present")
    open_ = [{"context": _public_id(c, roles.get(c)), "partition": roles.get(c), "finding": UPSTREAM_FINDING,
              "channels": exposed[c]} for c in present if c in exposed]
    promoted = [c for c in present if c not in exposed] if granted else []
    return {"label": A0_TRAIN_ONLY_VERIFIED, "from_label": A0_TRAIN_ONLY, "scope": TRAIN_ONLY_SCOPE, "claim": TRAIN_ONLY_CLAIM,
            "granted": granted, "requires": list(PROMOTION_CHECKS) + ["poison scope all",
                                                                       "partition_map_binding PASS (p3) or NOT_APPLICABLE (fake)"],
            "missing": missing, "failed": failed, "partition_map_kind": kind, "demonstration_only": kind != "p3",
            "manifests": manifest_pins, "contexts_present": len(present), "contexts_promoted": len(promoted),
            "contexts_open": len(open_), "open": open_[:200],
            "rule": "the label applies to the present contexts of exactly the bound manifest bytes, minus the OPEN ones; run "
                    "plans that read the manifests need this receipt with status PASS and granted true"}


def _verify(a, out, guard) -> tuple:
    a0m = _a0()
    epoch_doc, epoch_pin = load_epoch(a.epoch)
    pmap, map_pin = load_partition_map(a.partition_map)
    manifests, manifest_pins = _load_manifests(out)
    for p, m in manifests.items():
        if m.get("arm") != a.arm or m.get("partition") != p:
            raise Refused("manifest_identity", f"{p} manifest is arm {m.get('arm')} partition {m.get('partition')}")
        if (m.get("inputs") or {}).get("epoch", {}).get("sha256") != epoch_pin["sha256"]:
            raise Refused("verify_epoch_differs", f"{p} manifest was built from another epoch")
        if (m.get("inputs") or {}).get("partition_map", {}).get("sha256") != map_pin["sha256"]:
            raise Refused("verify_partition_map_differs", f"{p} manifest was built with another partition map")
    included = included_contexts(epoch_doc)
    by_id = {c["context_sha256"]: c for c in included}
    scope = manifests["train_fit"].get("scope") or {"kind": "all"}
    if a.contexts:
        ids, _pin = load_contexts_file(a.contexts)
    elif scope.get("kind") == "subset":
        ids, pin = load_contexts_file(scope["path"])
        if pin["sha256"] != scope.get("sha256"):
            raise Refused("verify_scope_differs", "the recorded --contexts file changed since the run")
    else:
        ids = [c["context_sha256"] for c in included]
    unknown = [c for c in ids if c not in by_id]
    if unknown:
        raise Refused("contexts_not_included", f"{len(unknown)} scope contexts are not included epoch contexts")
    unmapped = [c for c in ids if c not in pmap]
    if unmapped:
        raise Stop("partition_map_closure", f"{len(unmapped)} scope contexts have no partition-map entry")
    admitted = sorted(c for c in ids if pmap[c]["split_role"] in PARTITIONS)
    roles = {c: pmap[c]["split_role"] for c in admitted}

    def raw_of(cid):
        try:
            return jtt1.load_raw(by_id[cid]["input_ref"], a.jtt1_cache)
        except jtt1.JTTError as err:
            raise Refused("jtt1_not_cached_or_invalid", f"{cid[:16]}: {err} (verify never GETs)") from None

    kind = ((manifests["train_fit"].get("inputs") or {}).get("partition_map") or {}).get("kind")
    results = {"partition_map_binding": verify_map_binding(a, out, manifests, pmap, map_pin, epoch_pin)}
    exposed = {}
    if a.coverage:
        results["coverage"] = verify_coverage(out, manifests, roles)
    if a.fit_closure:
        results["fit_closure"] = verify_fit_closure(out, manifests, pmap, _latest_run_runtime(out))
    if a.poison_test:
        results["poison_test"] = verify_poison(a0m, out, manifests, by_id, pmap, admitted, raw_of, guard, a.poison_scope)
    if a.counterfactual:
        results["counterfactual"] = verify_counterfactual(a0m, admitted, roles, raw_of, guard)
    if a.affine_invariance:
        results["affine_invariance"] = verify_affine(a0m, admitted, raw_of, guard, roles)
    if a.upstream_exposure:
        results["upstream_exposure"], exposed = verify_upstream_exposure(admitted, roles, raw_of, guard, manifests)
    if a.column_set_audit:
        results["column_set_audit"] = column_set_audit(admitted, by_id, roles, raw_of, guard)
    states = [r["status"] for r in results.values() if r["status"] in GATING_STATUSES]
    status = "STOP" if "STOP" in states else ("FAIL" if "FAIL" in states else "PASS")
    code = {"STOP": EXIT_STOP, "FAIL": EXIT_FAIL, "PASS": EXIT_PASS}[status]
    promotion = train_only_promotion(results, manifests, manifest_pins, roles, exposed, kind)
    open_findings = []
    ue = results.get("upstream_exposure")
    if ue:
        open_findings += [{"finding": UPSTREAM_FINDING, "channel": ch, "contexts": ue[ch]["contexts"]}
                          for ch in ("scale", "column_set") if ue[ch]["status"] == "OPEN"]
    csa = results.get("column_set_audit")
    if csa and csa["status"] == "OPEN":
        open_findings.append({"finding": UPSTREAM_FINDING, "channel": "column_set_confirmed_cross_fold",
                              "contexts": csa["contexts_confirmed"]})
    doc = {"schema": VERIFY_SCHEMA, "experiment_id": EXPERIMENT_ID, "arm": a.arm, "status": status, "exit_code": code,
           "utc_us": common.utc_us(), "inputs": {"epoch": epoch_pin, "partition_map": dict(map_pin, kind=kind), "scope": scope,
                                                 "admitted": len(admitted), "manifests": manifest_pins},
           "results": results, "train_only_promotion": promotion, "open_findings": open_findings,
           "source_tree": {"encodings": source_tree_sha256("encodings"), "a0": _a0_producer_tree(a0m)["source_tree_sha256"]},
           "runtime": runtime_identity(), "guard": guard.summary(), "peak_rss_bytes": peak_rss_bytes(),
           "gets": 0, "rule": "verify reads only local files and the JTT1 cache; it never GETs. Statuses OPEN, NONE and "
                              "NOT_APPLICABLE do not gate; open_findings lists what a PASS does not cover"}
    return code, doc


def cmd_verify(a) -> tuple:
    try:
        cpuguard.require_slice_limits()
    except Exception as err:  # noqa: BLE001
        return EXIT_BLOCKED, {"status": "BLOCKED", "reason": "slice_limits", "detail": str(err)}
    if a.arm != "A0":
        return EXIT_BLOCKED, {"status": "BLOCKED", "arm": a.arm, "reasons": list(BLOCKED_ARM_REASONS.get(a.arm, []))}
    if not (a.coverage or a.fit_closure or a.poison_test or a.counterfactual or a.affine_invariance or a.upstream_exposure
            or a.column_set_audit):
        return EXIT_FAIL, {"status": "FAIL", "reason": "nothing_to_verify"}
    if not (a.poison_scope == "all" or re.fullmatch(r"sample:[1-9][0-9]*", a.poison_scope or "")):
        return EXIT_FAIL, {"status": "FAIL", "reason": "poison_scope", "detail": a.poison_scope}
    out = _real(a.out)
    if not out.is_dir():
        return EXIT_FAIL, {"status": "FAIL", "reason": "out_absent", "detail": str(out)}
    guard = cpuguard.Guard(journal_path=out / "guard-journal.jsonl", max_pause_s=GUARD_MAX_PAUSE_S)
    try:
        guard.wait()
        code, doc = _verify(a, out, guard)
    except EncodingError as err:
        code, doc = err.exit_code, {"schema": VERIFY_SCHEMA, "status": err.status, "exit_code": err.exit_code,
                                    "failure": {"reason": err.reason, "detail": err.detail}, "utc_us": common.utc_us()}
    except cpuguard.GuardTimeout as err:
        code, doc = EXIT_BLOCKED, {"schema": VERIFY_SCHEMA, "status": "BLOCKED", "exit_code": EXIT_BLOCKED,
                                   "failure": {"reason": "guard_timeout", "detail": str(err)}, "utc_us": common.utc_us()}
    except Exception as err:  # noqa: BLE001 (an A0Error is a bug and a STOP; anything else still leaves a FAIL document)
        stop = isinstance(err, _a0_error_type())
        code = EXIT_STOP if stop else EXIT_FAIL
        doc = {"schema": VERIFY_SCHEMA, "status": "STOP" if stop else "FAIL", "exit_code": code, "utc_us": common.utc_us(),
               "failure": {"reason": "a0_error" if stop else f"error:{type(err).__name__}", "detail": str(err)}}
    res = common.write_receipt(out / f"verify-{_stamp()}.json", doc)
    promo = doc.get("train_only_promotion") or {}
    return code, {"status": doc["status"], "exit_code": code, "verify": {"path": res["path"], "sha256": res["sha256"]},
                  "results": {k: v.get("status") for k, v in (doc.get("results") or {}).items()}, "failure": doc.get("failure"),
                  "train_only_promotion": {k: promo.get(k) for k in ("granted", "contexts_promoted", "contexts_open", "missing",
                                                                     "failed")} if promo else None,
                  "open_findings": doc.get("open_findings")}


# ------------------------------------------------------------------------------------------------------ adopt
def _load_ref(value, what, default_bucket=None) -> dict:
    """A ref given as a JSON file path or inline JSON: {bucket, key, version_id, sha256, bytes}."""
    if value is None:
        raise Refused(f"{what}_absent")
    text = value.strip()
    raw = text.encode() if text.startswith("{") else _read_bytes(value)
    ref = _loads(raw, what)
    if not isinstance(ref, dict):
        raise Refused(f"{what}_malformed", "not a JSON object")
    ref = dict(ref)
    if default_bucket and not ref.get("bucket"):
        ref["bucket"] = default_bucket
    if not all(ref.get(k) is not None for k in PIN_FIELDS) or not _is_hex64(ref.get("sha256")):
        raise Refused(f"{what}_malformed", f"needs {PIN_FIELDS} with a 64-hex sha256")
    return {k: ref[k] for k in PIN_FIELDS}


def _adopt_output_gate(a, release_ref, index_ref) -> str:
    """sample: OUT under RUN/tmp/p6-sample-*/. production: OUT exactly RUN/encodings/adoption/<release12>-<index12>,
    without --dry-run, with the same real-output gate as run (AB-1): a p3 map with --holdout, --holdout-binding and
    --p3-receipt, and a P5 PASS receipt for this epoch (their content is bound once the map is read)."""
    out = _real(a.out)
    if sample_root(out) is not None:
        return "sample"
    want = _real(RUN_ENCODINGS) / "adoption" / f"{release_ref['sha256'][:12]}-{index_ref['sha256'][:12]}"
    if a.dry_run or out != want:
        raise Blocked("output_gate", f"OUT must lie under {RUN_TMP}/{SAMPLE_DIR_PREFIX}*/ or, without --dry-run, be exactly "
                                     f"{want}")
    try:
        epoch_sha256 = _sha(_read_bytes(a.epoch))
    except OSError as err:
        raise Refused("epoch_unreadable", str(err)) from None
    reasons = production_gate_reasons(a, epoch_sha256)
    if reasons:
        raise Blocked("output_gate", "; ".join(reasons))
    return "production"


def release_qualification(release) -> dict:
    """Codex's qualification flags: adoption_qualified and every summary.*_qualified of a terminal release."""
    flags = {}
    if "adoption_qualified" in release or release.get("schema") == TERMINAL_RELEASE_SCHEMA:
        flags["adoption_qualified"] = release.get("adoption_qualified")
    summary = release.get("summary") if isinstance(release.get("summary"), dict) else {}
    for k in sorted(summary):
        if k.endswith("_qualified"):
            flags[f"summary.{k}"] = summary[k]
    return flags


def load_owner_waiver(path, release_ref, needed) -> dict:
    """An owner waiver citation (jope.dl-owner-waiver/v1) for the release's unqualified flags: it must name this release's
    sha256 and waive every flag in `needed`. Without one, production adoption of an unqualified release is BLOCKED."""
    if not path:
        raise Blocked("release_not_adoption_qualified", f"Codex qualification flags not true: {needed}; production adoption "
                                                        "needs them true or an --owner-waiver citation")
    raw = _read_bytes(path)
    doc = _loads(raw, "owner_waiver")
    if not isinstance(doc, dict) or doc.get("schema") != OWNER_WAIVER_SCHEMA or doc.get("release_sha256") != release_ref["sha256"]:
        raise Blocked("owner_waiver_mismatch", f"the waiver must be {OWNER_WAIVER_SCHEMA} for release {release_ref['sha256'][:16]}")
    waived = doc.get("waived")
    if not isinstance(waived, list) or not set(needed) <= set(waived):
        raise Blocked("owner_waiver_incomplete", f"the waiver does not waive {sorted(set(needed) - set(waived or []))}")
    return {"path": str(path), "sha256": _sha(raw), "bytes": len(raw), "waived": sorted(needed),
            "decision": doc.get("decision")}


def load_fetch_receipt(path, index_ref, release_ref) -> dict:
    """An exact-VersionId fetch receipt (e.g. jope.dl-codex-terminal-fetch/v1) whose refs.index equals --index-ref on all
    five pin fields: the index's key and VersionId are bound to a recorded fetch, not to CLI text (AB-2)."""
    raw = _read_bytes(path)
    doc = _loads(raw, "index_fetch_receipt")
    refs = doc.get("refs") if isinstance(doc, dict) else None
    if not isinstance(refs, dict) or not _pin_equal(refs.get("index"), index_ref):
        raise Refused("index_fetch_receipt_mismatch", "refs.index must equal --index-ref on bucket, key, version_id, sha256 "
                                                      "and bytes")
    local = ((doc.get("local") or {}).get("index") or {}) if isinstance(doc.get("local"), dict) else {}
    if local.get("bytes") is not None and local.get("bytes") != index_ref["bytes"]:
        raise Refused("index_fetch_receipt_mismatch", "local.index.bytes differs from --index-ref")
    rel = refs.get("release")
    return {"path": str(path), "sha256": _sha(raw), "bytes": len(raw), "schema": doc.get("schema"), "index_ref_equal": True,
            "release_ref_equal": None if rel is None else _pin_equal(rel, release_ref)}


def _ulp_close_f32(a, b, n=2) -> bool:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return False
    tol = n * np.spacing(np.maximum(np.abs(a), np.abs(b))).astype(np.float64)
    return bool(np.all(np.abs(a.astype(np.float64) - b.astype(np.float64)) <= tol))


def _rel_close(a, b, rtol) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= rtol * max(abs(a), abs(b)) + 1e-300


def codex_target_proxy_removal(X, y) -> tuple:
    """worker.py records(): train-only exact/affine target-proxy detection, reproduced operation for operation."""
    removed = []
    if np.ptp(y) > 1e-12:
        centered = y - y.mean()
        den = float(centered @ centered)
        for i in range(X.shape[1]):
            v = X[:, i]
            slope = float(centered @ (v - v.mean())) / den
            if abs(slope) > 1e-12 and np.max(abs(v - (slope * y + (v.mean() - slope * y.mean())))) <= 1e-10 * max(1, float(np.max(abs(v)))):
                removed.append(i)
    kept = [i for i in range(X.shape[1]) if i not in removed]
    return kept, removed


def pool_f32(rows) -> np.ndarray:
    """gpu_encoders.pool: mean | std | p10 | p90 over rows, float32."""
    return np.concatenate([rows.mean(0), rows.std(0), np.percentile(rows, 10, axis=0), np.percentile(rows, 90, axis=0)]).astype("float32")


def probe_sizes(n) -> tuple:
    nc = min(768, max(1, 3 * n // 4), n - 1)
    return nc, min(256, n - nc)


def array_contract(component, n, n_kept) -> dict:
    _nc, nq = probe_sizes(n)
    return {"hyperion": {"rows": [n, 128], "vector": [512], "feature_q01": [n_kept], "feature_q99": [n_kept]},
            "foundation": {"rows": [nq, 512], "vector": [2048], "feature_min": [n_kept], "feature_span": [n_kept]},
            "target": {"vector": [512]},
            "tabpfn35": {"rows": [nq, 1024], "vector": [2048]},
            "context2048": {"foundation_projected": [1024], "vector": [2048]}}[component]


def array_table_violation(meta, contract):
    """The first array name (sorted) whose outcome metadata differs from the contract (all float32), else None."""
    if not isinstance(meta, dict):
        return "arrays"
    for name in sorted(set(meta) | set(contract)):
        got, want = meta.get(name), contract.get(name)
        if want is None or not isinstance(got, dict) or got.get("dtype") != "float32" or list(got.get("shape") or []) != want:
            return name
    return None


def _int_list(v) -> bool:
    return isinstance(v, list) and all(_is_int(i) for i in v)


def recipe_violation(component, recipe, n, n_kept, context_transforms_pin):
    """(first violating field or None, informational dict)."""
    info = {}
    if not isinstance(recipe, dict):
        return "recipe", info
    if component == "hyperion":
        if recipe.get("recipe") != HYPERION_RECIPE:
            return "recipe", info
        if recipe.get("sweeps") != 3 or not _is_int(recipe.get("sweeps")):
            return "sweeps", info
        if recipe.get("scales") != HYPERION_SCALES:
            return "scales", info
        if recipe.get("native_rng_equivalence_qualified") is not False:
            return "native_rng_equivalence_qualified", info
        perms = recipe.get("permutations")
        if not isinstance(perms, list) or len(perms) != 3:
            return "permutations", info
        rows_id, cols_id = list(range(n)), list(range(n_kept))
        for s, pr in enumerate(perms):
            rows = pr.get("rows") if isinstance(pr, dict) else None
            cols = pr.get("columns") if isinstance(pr, dict) else None
            if not (_int_list(rows) and _int_list(cols)):
                return "permutations", info
            if s == 0 and (rows != rows_id or cols != cols_id):
                return "permutations", info
            if s > 0 and (sorted(rows) != rows_id or sorted(cols) != cols_id):
                return "permutations", info
    elif component in ("foundation", "tabpfn35"):
        if component == "foundation" and (recipe.get("seed") != FOUNDATION_SEED or not _is_int(recipe.get("seed"))):
            return "seed", info
        nc, nq = probe_sizes(n)
        ctx, qry = recipe.get("context_local_indices"), recipe.get("query_local_indices")
        if not _int_list(ctx) or len(ctx) != nc or any(not 0 <= i < n for i in ctx) or len(set(ctx)) != len(ctx):
            return "context_local_indices", info
        if not _int_list(qry) or len(qry) != nq or any(not 0 <= i < n for i in qry) or len(set(qry)) != len(qry):
            return "query_local_indices", info
        if set(ctx) & set(qry):
            return "context_query_overlap", info
        if component == "foundation":
            perm = np.random.default_rng(FOUNDATION_SEED).permutation(n)
            info["rng_permutation_recompute_matches"] = bool(ctx == perm[:nc].tolist() and qry == perm[nc:nc + nq].tolist())
            if not info["rng_permutation_recompute_matches"]:         # the recipe identity claims seed 20260707 (AB-6)
                return "rng_permutation", info
    elif component == "target":
        if recipe.get("seed") != TARGET_SEED or not _is_int(recipe.get("seed")):
            return "seed", info
        if recipe.get("structural_zero_dimensions") != TARGET_STRUCTURAL_ZEROS:
            return "structural_zero_dimensions", info
    elif component == "context2048":
        if not _pin_equal(recipe.get("normalizer_ref"), context_transforms_pin):
            return "normalizer_ref", info
        if recipe.get("training_holdout_qualified") is not False:
            return "training_holdout_qualified", info
    return None, info


class _Adoption:
    """Tier I / O / A classification state for one adopt invocation."""

    def __init__(self, a, out, guard, fetch, *, release, index, packet, pins, ctx_norm_pin, pmap, map_pin, kind,
                 included, subset, mems, s1, terminal, binding_sha256=None):
        self.a, self.out, self.guard, self.fetch = a, out, guard, fetch
        self.release, self.index, self.packet, self.pins, self.ctx_norm_pin = release, index, packet, pins, ctx_norm_pin
        self.pmap, self.map_pin, self.kind, self.binding_sha256 = pmap, map_pin, kind, binding_sha256
        self.included, self.subset, self.subset_set, self.mems, self.s1 = included, subset, set(subset), mems, s1
        self.terminal = terminal
        self.dry = bool(a.dry_run)
        self.join = collections.defaultdict(lambda: collections.defaultdict(list))
        for key, xs in index_join(index["outputs"]).items():
            for x in xs:
                self.join[key][x.get("component")].append(x)
        self.packet_ix, self.packet_dups = {}, 0
        for d in packet.get("datasets") or []:
            ident = d.get("identity") if isinstance(d, dict) else None
            if isinstance(ident, list):
                if tuple(ident) in self.packet_ix:
                    self.packet_dups += 1
                    continue
                self.packet_ix[tuple(ident)] = d
        self.headers = {}
        self.prepared = {}
        self.rows = {}
        self.budget = {"tier_O": None, "tier_A": None}
        self.job_id_checked = self.job_id_ok = 0
        self.slot_report, self.retired_keys = self._account_slots()

    def _account_slots(self) -> tuple:
        """(report, retired context join keys): the packet's slots (dataset x fold x component) against the index (AB-5).
        An index row with an identity accounts for its (identity, fold, component) slot. A null-identity row is resolved
        through the packet by job_id and retires its whole fold (Codex retires folds, so the fold's other components may
        have no row). In a terminal index, a slot with no row outside a retired fold, or a null-identity row that resolves
        to no packet slot, is unaccounted (the caller STOPs); retired slots are reconciled with the terminal release."""
        comps = list(self.packet.get("components") or COMPONENTS)
        by_job, slots = {}, set()
        for i, d in enumerate(self.packet.get("datasets") or []):
            if i % 64 == 0:
                self.guard.tick()
            ident = d.get("identity") if isinstance(d, dict) else None
            if not isinstance(ident, list):
                continue
            for f in d.get("folds") or []:
                if not isinstance(f, dict):
                    continue
                fold_key = (f.get("fold"), f.get("repeat", 0))
                for k in comps:
                    slot = (tuple(ident), fold_key, k)
                    slots.add(slot)
                    try:
                        by_job[codex_job_id(d, f, k, self.packet)] = slot
                    except (KeyError, TypeError, ValueError):
                        pass
        with_identity, retired_folds = set(), set()
        null_rows = unresolved = 0
        for x in self.index.get("outputs") or []:
            ident = x.get("identity")
            if ident:
                fold = x.get("fold") if isinstance(x.get("fold"), dict) else {}
                with_identity.add((tuple(ident), (fold.get("fold"), fold.get("repeat", 0)), x.get("component")))
                continue
            null_rows += 1
            slot = by_job.get(x.get("job_id"))
            if slot is None:
                unresolved += 1
            else:
                retired_folds.add(slot[:2])
        retired_slots = sum(1 for s in slots if s[:2] in retired_folds)
        unaccounted = sum(1 for s in slots if s not in with_identity and s[:2] not in retired_folds)
        summary = self.release.get("summary") if isinstance(self.release.get("summary"), dict) else {}
        expected = summary.get("retired_component_slots")
        report = {"packet_slots": len(slots), "index_slots_with_identity": len(with_identity & slots),
                  "index_slots_outside_packet": len(with_identity - slots), "null_identity_rows": null_rows,
                  "null_identity_unresolved": unresolved, "retired_folds": len(retired_folds),
                  "retired_component_slots": retired_slots, "release_retired_component_slots": expected,
                  "reconciled_with_release": None if expected is None else expected == retired_slots,
                  "unaccounted_slots": unaccounted,
                  "rule": "a slot is accounted by an index row with its identity, fold and component, or by a retired fold "
                          "(a null-identity row resolved through the packet by job_id); terminal: unaccounted or unresolved "
                          "-> STOP"}
        return report, {(ident[0], ident[2], fk[0], fk[1]) for ident, fk in retired_folds}

    def index_level(self) -> dict:
        """Over every complete index output (not only the epoch-joined ones): a packet entry by identity, the job_id
        recompute, fold dict equality with the packet fold, and the LE u64 train-index recompute."""
        out = collections.Counter()
        for i, xs in enumerate(self.join.values()):
            if i % 512 == 0:
                self.guard.tick()
            for rows in xs.values():
                for x in rows:
                    out["complete_outputs"] += 1
                    d = self.packet_ix.get(tuple(x["identity"]))
                    if d is None:
                        continue
                    out["packet_entry_found"] += 1
                    f = x["fold"]
                    try:
                        out["job_id_equal"] += codex_job_id(d, f, x.get("component"), self.packet) == x.get("job_id")
                    except (KeyError, TypeError, ValueError):
                        pass
                    out["fold_equals_packet_fold"] += any(g == f for g in d.get("folds") or [])
                    out["train_index_hash_ok"] += train_index_ok(f)
        return {k: out.get(k, 0) for k in ("complete_outputs", "packet_entry_found", "job_id_equal", "fold_equals_packet_fold",
                                            "train_index_hash_ok")}

    # -- helpers
    def _jtt1_raw(self, c):
        return self.fetch.jtt1(c["input_ref"])

    def _jtt1_header(self, c):
        cid = c["context_sha256"]
        if cid not in self.headers:
            raw = self._jtt1_raw(c)
            h = jtt1.parse_header(raw)
            self.headers[cid] = {"n_train": h["n_train"], "n_test": h["n_test"], "p": h["p"],
                                 "train_row_sha256": jtt1.train_rows_sha256(raw)}
        return self.headers[cid]

    def _prepared_manifest(self, c):
        """The source_manifest pin of the context's sha-verified prepared_ref (a local read, cached per context), or None
        when it cannot be verified (AB-4: binds the Codex packet's manifest_ref, hence identity[1], to the epoch)."""
        cid = c["context_sha256"]
        if cid not in self.prepared:
            ref = c.get("prepared_ref") if isinstance(c.get("prepared_ref"), dict) else {}
            got = None
            try:
                raw = _read_bytes(ref["path"])
                if _sha(raw) == ref.get("sha256") and len(raw) == ref.get("bytes"):
                    sm = json.loads(raw).get("source_manifest")
                    if isinstance(sm, dict) and all(sm.get(k) is not None for k in PIN_FIELDS):
                        got = {k: sm[k] for k in PIN_FIELDS}
            except (KeyError, OSError, ValueError, TypeError, AttributeError):
                got = None
            self.prepared[cid] = got
        return self.prepared[cid]

    def _row(self, c, k, role, fam):
        rights = {"rights_status": "unknown", "source": "pending_P2"}
        # a heldout context is never eligible for a dev-only arm (LB-replica: never holdout-reported, G06; TOL-09)
        arms = [x for x in ARMS_ELIGIBLE[k] if not (role == "heldout" and x in DEV_ONLY_ARMS)]
        return {"context_sha256": c["context_sha256"], "component": k, "cohort": c["cohort"], "fold": c["fold"],
                "dataset_hash": c["dataset_hash"], "sample_identity_sha256": c["sample_identity_sha256"], "split_role": role,
                "family_id": fam, "status": None, "tier": "I", "train_only": None, "job_id": None, "array_ref": None,
                "outcome_ref": None, "claim_ref": None, "index_fold": None,
                "binding_evidence": {"index_join": None, "job_id": None, "manifest": None, "membership": None, "index_hash": None,
                                     "jtt1_rows": None, "outcome": None, "content": None, "array": None},
                "jtt1_evidence": [], "residual_detail": None,
                "recipe_identity": RECIPE_IDENTITY[k], "legacy_projection": k == "context2048",
                "raw_foundation_2048_retained": None, "pre_normalization": None,
                "arms_eligible": arms, "rights": rights,
                "encoder_license": self.pins[k]["encoder_license"], "research_only": self.pins[k]["research_only"],
                "family_provenance": {"family_id": fam, "split_role": role,
                                      "partition_map": {"sha256": self.map_pin["sha256"], "kind": self.kind,
                                                        "binding_sha256": self.binding_sha256}},
                "encoder_weight_exposure": "Unknown"}

    def finalize_labels(self):
        """Labels that depend on the final status (TOL-10, AB-7): train_only CLAIMED only for bound_index_only and
        adopted rows; raw_foundation_2048_retained and pre_normalization only once the tier-O array contract (and at tier A
        the NPZ) showed the raw vector, i.e. for adopted rows; the JTT1-derived evidence each row carries."""
        for r in self.rows.values():
            st = r["status"]
            adopted = st == "adopted"
            r["train_only"] = ADOPTED_TRAIN_ONLY if st in ("bound_index_only", "adopted") else None
            r["raw_foundation_2048_retained"] = True if adopted and r["component"] == "foundation" else None
            r["pre_normalization"] = True if adopted else None
            r["jtt1_evidence"] = jtt1_evidence(r)

    # -- tier I
    def tier_i(self, c):
        cid = c["context_sha256"]
        role, fam = self.pmap[cid]["split_role"], self.pmap[cid]["family_id"]
        mem = self.mems.get(cid)
        for k in COMPONENTS:
            row = self._row(c, k, role, fam)
            row["status"] = self._classify_i(c, k, role, mem, row)
            self.rows[(cid, k)] = row

    def _classify_i(self, c, k, role, mem, row):
        if role in EXCLUDED_ROLES:
            return f"excluded_by_role:{role}"
        if c["context_sha256"] in self.s1:
            row["residual_detail"] = {"view_id": c.get("view_id"), "view_kind": view_kind(c.get("view_id"))}
            return "residual:feature_sanitized_view"
        jkey = context_join_key(c)
        cands = self.join.get(jkey, {}).get(k, [])
        if not cands:
            if jkey in self.retired_keys:
                return "residual:absent_or_retired_before_dispatch"
            return "residual:no_complete_output" if self.terminal else "pending:queue_nonterminal"
        if len(cands) > 1:
            return "rejected:duplicate_component_rows"
        x = cands[0]
        f = x["fold"]
        row["_x"] = x
        row.update(job_id=x.get("job_id"), array_ref=x.get("array_ref"), outcome_ref=x.get("outcome_ref"),
                   claim_ref=x.get("claim_ref"),
                   index_fold={key: f.get(key) for key in ("fold", "repeat", "train_indices_sha256", "train_row_ids_sha256",
                                                           "test_indices_sha256", "train_rows", "test_rows")})
        ev = row["binding_evidence"]
        ev["index_join"] = {"rows": 1, "key": list(jkey), "identity": list(x["identity"])}
        if not identity_prefix_ok(x["identity"]):
            return "rejected:identity_prefix_mismatch"
        pin = self.pins[k]["status"]
        if pin == "unpinned":
            return "rejected:component_unpinned_in_inventory"
        if pin != "pinned":
            return "rejected:component_pin_mismatch"
        d = self.packet_ix.get(tuple(x["identity"]))
        if d is None:
            return "rejected:not_in_codex_packet"
        row["_d"] = d
        row["rights"] = rights_for(d)
        if d.get("cohort") != c["cohort"] or (c["cohort"] == "incoming" and d.get("item_id") != c.get("item_id")) \
                or (c.get("canonical_sha256") and x["identity"][1] != c["canonical_sha256"]):
            return "rejected:packet_identity_mismatch"
        manifest = self._prepared_manifest(c)
        if manifest is None:
            return "rejected:prepared_ref_unverifiable"
        if not _pin_equal(d.get("manifest_ref"), manifest):
            return "rejected:packet_manifest_mismatch"
        ev["manifest"] = "packet manifest_ref equals the source_manifest of the sha-verified prepared_ref"
        if f.get("repeat") != 0 or "repeat" not in f:
            return "rejected:index_fold_repeat"
        pf = next((g for g in d.get("folds") or [] if isinstance(g, dict) and g.get("fold") == f.get("fold")
                   and g.get("repeat", 0) == f.get("repeat", 0)), None)
        if pf != f:
            return "rejected:packet_fold_mismatch"
        self.job_id_checked += 1
        try:
            jid = codex_job_id(d, f, k, self.packet)
        except (KeyError, TypeError, ValueError):
            jid = None
        if jid != x.get("job_id"):
            return "rejected:job_id_mismatch"
        self.job_id_ok += 1
        ev["job_id"] = "recomputed_equal"
        if mem is None or not fold_binds_membership(f, mem):
            return "rejected:membership_mismatch"
        if mem.get("outer_repeat") != 0:
            return "rejected:membership_repeat"
        ev["membership"] = "outer train/test index sha256 and rows equal; outer_repeat 0"
        if not train_index_ok(f):
            return "rejected:index_hash_recompute_mismatch"
        ev["index_hash"] = "le_u64_sha256 recomputed equal; strictly increasing; in range"
        if row["rights"]["rights_status"] == "prohibited":
            return "rejected:rights_prohibited"
        if c["context_sha256"] in self.subset_set and self.fetch.cache_dir is not None:
            h = self._jtt1_header(c)
            ev["jtt1_rows"] = dict(h)
            if h["n_train"] != f.get("train_rows") or h["n_test"] != f.get("test_rows"):
                return "rejected:jtt1_shape_mismatch"
        if k == "context2048":
            return "not_for_arms:legacy_projection"
        if not row["arms_eligible"]:
            return "not_for_arms:dev_only_arm"
        return "bound_index_only"

    # -- tier O
    def tier_o(self, row, c):
        x, d = row["_x"], row["_d"]
        f = x["fold"]
        k = row["component"]
        ref = x["outcome_ref"]
        body = self.fetch.get("outcome", ref)
        row["tier"] = "O"               # only once the GET happened: a budget refusal leaves the row at tier I
        if _sha(body) != ref.get("sha256") or len(body) != ref.get("bytes"):
            return "rejected:outcome_sha_mismatch"
        try:
            o = json.loads(body)
        except ValueError:
            return "rejected:outcome_field_mismatch:json"
        if not isinstance(o, dict):
            return "rejected:outcome_field_mismatch:json"
        row["_o"] = o
        ev = row["binding_evidence"]
        ev["outcome"] = {"sha256": ref.get("sha256"), "bytes": ref.get("bytes"), "version_id": ref.get("version_id")}
        if o.get("schema") != OUTCOME_SCHEMA:
            return "rejected:outcome_field_mismatch:schema"
        if o.get("state") != "complete":
            return "rejected:outcome_field_mismatch:state"
        for field in OUTCOME_EQUAL_FIELDS:
            want = self.index.get("packet_ref") if field == "packet_ref" else x.get(field)
            if o.get(field) != want:
                return f"rejected:outcome_field_mismatch:{field}"
        if o.get("training_admitted") is not False:
            return "rejected:outcome_field_mismatch:training_admitted"
        if o.get("benchmark_fit") is not False:
            return "rejected:outcome_field_mismatch:benchmark_fit"
        if not (_is_int(o.get("aws_lambda_calls")) and o.get("aws_lambda_calls") == 0):
            return "rejected:outcome_field_mismatch:aws_lambda_calls"
        ib = o.get("input_binding") if isinstance(o.get("input_binding"), dict) else {}
        if ib.get("selection_fit") != "outer_train_only":
            return "rejected:selection_fit_not_outer_train_only"
        mem = self.mems.get(c["context_sha256"]) or {}
        if not (ib.get("outer_train_indices_sha256") == f.get("train_indices_sha256") == mem.get("outer_train_indices_sha256")
                and ib.get("outer_train_row_ids_sha256") == f.get("train_row_ids_sha256")):
            return "rejected:input_binding_hash_mismatch"
        kept, removed = ib.get("kept_feature_indices"), ib.get("removed_target_feature_indices")
        if self.fetch.cache_dir is not None:
            p = self._jtt1_header(c)["p"]
        else:
            p = (d.get("shape") or {}).get("features")
        if not (_int_list(kept) and _int_list(removed) and kept and _is_int(p) and not set(kept) & set(removed)
                and len(set(kept)) == len(kept) and len(set(removed)) == len(removed) and sorted(kept + removed) == list(range(p))):
            return "rejected:feature_partition_invalid"
        n = f.get("train_rows")
        bad = array_table_violation(o.get("arrays"), array_contract(k, n, len(kept)))
        if bad is not None:
            return f"rejected:array_contract_violation:{bad}"
        field, info = recipe_violation(k, o.get("recipe"), n, len(kept), self.ctx_norm_pin)
        if field is not None:
            return f"rejected:recipe_contract_violation:{field}"
        ev["outcome"].update(checks="fields, selection_fit, input_binding hashes, feature partition, array contract, recipe",
                             **info)
        if self.a.content_checks:
            which = self._content_o(row, c, o, kept, removed)
            if which is not None:
                return f"rejected:content_check_failed:{which}"
        return "adopted"

    def _content_o(self, row, c, o, kept, removed):
        _h, _g, X, y = jtt1.train_view(self._jtt1_raw(c))
        k = row["component"]
        content = {}
        rk, rr = codex_target_proxy_removal(X, y)
        if rk != kept or rr != removed:
            row["binding_evidence"]["content"] = {"kept_features": False}
            return "kept_features"
        content["kept_features"] = True
        recipe = o.get("recipe") or {}
        if k == "foundation":
            y32 = y.astype(np.float32)[np.asarray(recipe["context_local_indices"], dtype=np.int64)]
            mu, sd = float(y32.mean()), float(y32.std()) + 1e-8
            ok = _rel_close(mu, recipe.get("context_y_mean"), 1e-6) and _rel_close(sd, recipe.get("context_y_std"), 1e-6)
            content["context_y_moments"] = ok
            if not ok:
                row["binding_evidence"]["content"] = content
                return "context_y_moments"
        if k == "target":
            y32 = y.astype(np.float32)
            lo = y32.min()
            span = y32.max() - lo
            ok = _ulp_close_f32(lo, recipe.get("train_target_min", np.nan)) and _ulp_close_f32(span, recipe.get("train_target_span", np.nan))
            content["target_min_span"] = ok
            if not ok:
                row["binding_evidence"]["content"] = content
                return "target_min_span"
        row["binding_evidence"]["content"] = content
        return None

    # -- tier A
    def tier_a(self, row, c):
        x, o = row["_x"], row["_o"]
        ref = x["array_ref"]
        k = row["component"]
        body = self.fetch.get("array", ref)
        row["tier"] = "A"               # only once the GET happened: a budget refusal leaves the row at tier O
        if _sha(body) != ref.get("sha256") or len(body) != ref.get("bytes"):
            return "rejected:array_sha_mismatch"
        try:
            with np.load(io.BytesIO(body), allow_pickle=False) as z:
                arrays = {name: np.asarray(z[name]) for name in z.files}
        except Exception:  # noqa: BLE001 (any refusal to load is the classification)
            return "rejected:array_load_refused"
        meta = o.get("arrays") or {}
        for name in sorted(set(arrays) | set(meta)):
            got, want = arrays.get(name), meta.get(name)
            if got is None or not isinstance(want, dict) or str(got.dtype) != want.get("dtype") or list(got.shape) != list(want.get("shape") or []):
                return f"rejected:array_contract_violation:{name}"
            if not np.isfinite(got).all():
                return f"rejected:array_contract_violation:{name}"
        checks = {}
        kept = (o.get("input_binding") or {}).get("kept_feature_indices") or []
        if k in ("hyperion", "foundation") and self.fetch.cache_dir is not None:
            _h, _g, X, _y = jtt1.train_view(self._jtt1_raw(c))
            X32 = X[:, kept].astype("float32")
            if k == "hyperion":
                q01 = np.percentile(X32, 1, axis=0).astype("float32")
                q99 = np.percentile(X32, 99, axis=0).astype("float32")
                ok = _ulp_close_f32(q01, arrays["feature_q01"]) and _ulp_close_f32(q99, arrays["feature_q99"])
                checks["feature_quantiles"] = ok
                if not ok:
                    row["binding_evidence"]["array"] = {"sha256": ref.get("sha256"), "checks": checks}
                    return "rejected:content_check_failed:feature_quantiles"
            else:
                lo = X32.min(0)
                span = X32.max(0) - lo
                span = np.where(span > 1e-12, span, 1)
                ok = _ulp_close_f32(lo, arrays["feature_min"]) and _ulp_close_f32(span, arrays["feature_span"])
                checks["feature_min_span"] = ok
                if not ok:
                    row["binding_evidence"]["array"] = {"sha256": ref.get("sha256"), "checks": checks}
                    return "rejected:content_check_failed:feature_min_span"
        if "rows" in arrays:
            rows = arrays["rows"]
            pooled = pool_f32(rows)
            scale = max(1.0, float(np.max(np.abs(rows)))) if rows.size else 1.0
            ok = pooled.shape == arrays["vector"].shape and bool(np.allclose(arrays["vector"], pooled, rtol=1e-5, atol=1e-6 * scale))
            checks["pooled_vector"] = ok
            if not ok:
                row["binding_evidence"]["array"] = {"sha256": ref.get("sha256"), "checks": checks}
                return "rejected:content_check_failed:pooled_vector"
        checks["all_finite_float32"] = True
        row["binding_evidence"]["array"] = {"sha256": ref.get("sha256"), "bytes": ref.get("bytes"), "checks": checks,
                                            "held": "in memory only; never written"}
        return "adopted"


def _counts(rows) -> dict:
    out = {k: collections.Counter() for k in COMPONENTS}
    for r in rows:
        out[r["component"]][r["status"]] += 1
    return {k: dict(sorted(v.items())) for k, v in out.items()}


def _row_public(row) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def _rows_bytes(rows) -> bytes:
    return b"".join(common.canonical(_row_public(r)) + b"\n"
                    for r in sorted(rows, key=lambda r: (r["context_sha256"], COMPONENTS.index(r["component"]))))


def view_kind(view_id) -> str:
    """The suffix kind of a feature-sanitized view id <dataset>__r<k>__f<k>__<suffix>: clean_target_leak_v1 (fold 0),
    test_adaptive (folds 1-4: __test_adaptive_<hash>), another suffix as is, or unsuffixed."""
    parts = str(view_id or "").split("__")
    suffix = "__".join(parts[3:])
    if not suffix:
        return "unsuffixed"
    return "test_adaptive" if suffix.startswith("test_adaptive_") else suffix


def jtt1_evidence(row) -> list:
    """The JTT1-derived evidence a row carries: shape (JTT1 n_train/n_test equal the fold), content (the tier-O
    recomputes: kept features, context y moments, target min/span) and array_content (feature q01/q99 or min/span)."""
    ev = row["binding_evidence"]
    out = []
    if ev.get("jtt1_rows"):
        out.append("shape")
    content = ev.get("content")
    if isinstance(content, dict) and content and all(v is True for v in content.values()):
        out.append("content")
    checks = (ev.get("array") or {}).get("checks") or {}
    if any(checks.get(k) is True for k in ("feature_quantiles", "feature_min_span")) and all(v is True for v in checks.values()):
        out.append("array_content")
    return out


def production_completion(rows) -> dict:
    """A production adoption PASSes only when no hyperion/foundation/target row is left bound_index_only and every
    adopted row reached tier A with its REQUIRED_JTT1_EVIDENCE (AB-3, TOL-07)."""
    unfinished = sum(1 for r in rows if r["component"] in PRODUCTION_OUTCOME_COMPONENTS and r["status"] == "bound_index_only")
    unevidenced = sum(1 for r in rows if r["status"] == "adopted" and (
        r["tier"] != "A" or not set(REQUIRED_JTT1_EVIDENCE.get(r["component"], ("shape",))) <= set(r["jtt1_evidence"])))
    return {"unfinished_bound_rows": unfinished, "adopted_without_jtt1_evidence": unevidenced,
            "required_jtt1_evidence": {k: list(v) for k, v in REQUIRED_JTT1_EVIDENCE.items()},
            "rule": "production PASS needs zero unfinished rows of the tier-O components and every adopted row at tier A with "
                    "its required JTT1-derived evidence; otherwise FAIL (BLOCKED when a GET budget ran out)"}


def residual_totals(rows) -> dict:
    """Residual rows by reason, over all components and per component (AB-5)."""
    total, per = collections.Counter(), collections.defaultdict(collections.Counter)
    for r in rows:
        if r["status"].startswith("residual:"):
            total[r["status"]] += 1
            per[r["component"]][r["status"]] += 1
    return {"total": dict(sorted(total.items())), "by_component": {k: dict(sorted(v.items())) for k, v in sorted(per.items())}}


def _adopt(a, out, guard, fetch, ledger, state) -> tuple:
    mode = "dry_run" if a.dry_run else "production"
    state["mode"] = mode
    gate = state["gate"]
    release_ref = state["release_ref"]
    index_ref = state["index_ref"]
    # b. release: one exact-VersionId GET unless OUT/release.json already holds the same bytes
    rel_path = out / "release.json"
    if rel_path.is_file() and _sha(_read_bytes(rel_path)) == release_ref["sha256"]:
        body, fetched = _read_bytes(rel_path), False
    else:
        body, fetched = fetch.get("release", release_ref), True
    if _sha(body) != release_ref["sha256"] or len(body) != int(release_ref["bytes"]):
        raise Refused("release_sha_mismatch", f"got {_sha(body)[:16]} / {len(body)} B")
    if fetched:
        common.immutable_write(rel_path, body, mode=0o444)
    release = _loads(body, "release")
    if not isinstance(release, dict) or release.get("schema") not in RELEASE_SCHEMAS:
        raise Refused("release_schema", f"expected one of {RELEASE_SCHEMAS}")
    if release.get("optimizer_training") is not False:
        raise Refused("release_optimizer_training", "optimizer_training must be false")
    state["release_fetched_now"] = fetched          # per invocation: receipt only, never in the deterministic adoption.json
    rel_index = release.get("index_ref")
    terminal_release = release.get("schema") == TERMINAL_RELEASE_SCHEMA
    launch_ref = release.get("original_launch_release_ref") if terminal_release else release_ref
    flags = release_qualification(release)
    state["inputs"]["release"] = {"ref": release_ref, "keys": sorted(release), "schema": release.get("schema"),
                                  "terminal_release": terminal_release, "state": release.get("state"),
                                  "packet_ref": release.get("packet_ref"), "index_ref": rel_index,
                                  "index_ref_is_null": rel_index is None,
                                  # a dry run records it; production needs equality on all five pin fields (AB-2)
                                  "index_ref_equals_adopted_index": None if rel_index is None else _pin_equal(rel_index, index_ref),
                                  "original_launch_release_ref": release.get("original_launch_release_ref"),
                                  "launch_release_equals_plan_pin": _pin_equal(launch_ref, PLAN_LAUNCH_RELEASE_REF),
                                  "qualification": flags, "custody_ref": release.get("custody_ref"),
                                  "progress_ref": release.get("progress_ref"), "docs": release.get("docs")}
    # c. index (local only; never re-fetched)
    idx_path = Path(a.index)
    raw = _read_bytes(idx_path)
    isha = _sha(raw)
    if isha != index_ref["sha256"] or idx_path.stem != isha or len(raw) != int(index_ref["bytes"]):
        raise Refused("index_sha_mismatch", f"local {isha[:16]} ({len(raw)} B, file {idx_path.name}) vs ref {index_ref['sha256'][:16]}")
    index = _loads(raw, "index")
    nbytes = len(raw)
    del raw
    if not isinstance(index, dict) or index.get("schema") != INDEX_SCHEMA:
        raise Refused("index_schema", f"expected {INDEX_SCHEMA}")
    if index.get("optimizer_training") is not False:
        raise Refused("index_optimizer_training", "optimizer_training must be false")
    if index.get("errors") != []:
        raise Refused("index_errors", "index.errors must be []")
    if index.get("packet_ref") != release.get("packet_ref"):
        raise Stop("packet_ref_mismatch", "index.packet_ref differs from release.packet_ref")
    terminal = index.get("all_shards_terminal") is True
    outputs = index.get("outputs") or []
    state["inputs"]["index"] = {"path": str(idx_path), "sha256": isha, "bytes": nbytes, "ref": index_ref, "utc_us": index.get("utc_us"),
                                "all_shards_terminal": terminal, "outputs": len(outputs),
                                "complete": sum(1 for x in outputs if x.get("state") == "complete"),
                                "null_identity": sum(1 for x in outputs if not x.get("identity")),
                                "packet_ref": index.get("packet_ref")}
    if not terminal and mode == "production":
        raise Blocked("queue_not_terminal", "production adoption needs all_shards_terminal=true; use --dry-run")
    # c2. the index bound to the release and to a recorded exact-VersionId fetch, and Codex's qualification (AB-2)
    if a.index_fetch_receipt:
        state["inputs"]["index"]["fetch_receipt"] = load_fetch_receipt(a.index_fetch_receipt, index_ref, release_ref)
    elif mode == "production":
        raise Refused("index_fetch_receipt_absent", "production adoption binds the index key and VersionId to an exact-VersionId "
                                                    "fetch receipt (--index-fetch-receipt), not to --index-ref text")
    else:
        state["inputs"]["index"]["fetch_receipt"] = None
    state["inputs"]["owner_waiver"] = None
    if mode == "production":
        rel_rec = state["inputs"]["release"]
        if rel_rec["index_ref_equals_adopted_index"] is not True:
            raise Refused("release_index_ref_differs", "production adoption needs release.index_ref equal to --index-ref on "
                                                       "bucket, key, version_id, sha256 and bytes (a terminal release)")
        if not rel_rec["launch_release_equals_plan_pin"]:
            raise Refused("launch_release_not_plan_pin", "the (original) launch release must be the PLAN §5 P6 pin "
                                                         f"{PLAN_LAUNCH_RELEASE_REF['sha256'][:16]}@{PLAN_LAUNCH_RELEASE_REF['version_id']}")
        needed = sorted({k for k, v in flags.items() if v is not True} | ({"adoption_qualified"} - set(flags)))
        if needed:
            state["inputs"]["owner_waiver"] = load_owner_waiver(a.owner_waiver, release_ref, needed)
    # d. packet (local copy; the 44.8 MB packet is never fetched)
    praw = _read_bytes(a.packet)
    pref = index["packet_ref"]
    if _sha(praw) != pref.get("sha256") or len(praw) != pref.get("bytes"):
        raise Refused("packet_sha_mismatch", "local packet.json differs from index.packet_ref")
    packet = _loads(praw, "packet")
    state["inputs"]["packet"] = {"path": str(a.packet), "sha256": _sha(praw), "bytes": len(praw)}
    del praw
    # e. inventory
    iraw = _read_bytes(a.inventory)
    if _sha(iraw) != a.inventory_sha256:
        raise Refused("inventory_sha_mismatch", f"{_sha(iraw)[:16]} != {a.inventory_sha256[:16]}")
    inventory = _loads(iraw, "inventory")
    state["inputs"]["inventory"] = {"path": str(a.inventory), "sha256": _sha(iraw), "bytes": len(iraw)}
    # f. component pins
    pins = component_pins(packet, inventory)
    state["component_pins"] = pins
    ctx_norm_pin = _inventory_entry(inventory, "fitted_normalizers", "context_transforms_lineage_b").get("identity")
    # g. epoch, partition map and memberships
    epoch_doc, epoch_pin = load_epoch(a.epoch)
    state["inputs"]["epoch"] = epoch_pin
    pmap, map_pin = load_partition_map(a.partition_map)
    binding = _map_binding(a, pmap, map_pin, epoch_pin, gate)
    kind = resolve_map_kind(a.partition_map, pmap, a.partition_map_kind, binding, map_pin["sha256"])
    state["inputs"]["partition_map"] = dict(map_pin, kind=kind, binding=binding_manifest_entry(binding))
    state["inputs"].update(_binding_inputs(binding))
    included = included_contexts(epoch_doc)
    by_id = {c["context_sha256"]: c for c in included}
    unmapped = [c["context_sha256"] for c in included if c["context_sha256"] not in pmap]
    if unmapped:
        raise Stop("partition_map_closure", f"{len(unmapped)} included contexts have no partition-map entry (first {unmapped[0][:16]})")
    join_keys = collections.Counter(context_join_key(c) for c in included)
    duplicated = sorted(k for k, n in join_keys.items() if n > 1)
    if duplicated:                                  # one Codex output must map to one context (AB-4)
        raise Stop("context_join_key_duplicate", f"{len(duplicated)} join keys (dataset_hash, sample_identity, fold, 0) "
                                                 f"name more than one included context (first {list(duplicated[0])})")
    if a.contexts:
        subset, cpin = load_contexts_file(a.contexts)
        unknown = [cid for cid in subset if cid not in by_id]
        if unknown:
            raise Refused("contexts_not_included", f"{len(unknown)} --contexts entries are not included epoch contexts")
        state["inputs"]["contexts"] = cpin
    else:
        subset = sorted(by_id)
        state["inputs"]["contexts"] = {"kind": "all", "n": len(subset)}
    mem_rows, mem_pin = load_memberships(a.memberships)
    state["inputs"]["memberships"] = mem_pin
    mems = link_memberships(included, mem_rows, require=True)
    # h. quarantine
    lq = epoch_doc.get("leakage_quarantine") or {}
    q = {v.split("__")[0] for v in lq.get("source_view_ids") or []}
    s1 = {c["context_sha256"] for c in included if c["dataset_hash"] in q}
    s2 = {c["context_sha256"] for c in included if membership_disagrees(mems[c["context_sha256"]], c)}
    state["quarantine"] = {"quarantined_datasets": sorted(q), "S1": len(s1), "S2": len(s2), "equal": s1 == s2,
                           "rule": "S1 = included contexts of quarantined datasets; S2 = contexts whose membership disagrees with the epoch"}
    if s1 != s2:
        raise Stop("quarantine_sets_differ", f"S1={len(s1)} S2={len(s2)}")
    ad = _Adoption(a, out, guard, fetch, release=release, index=index, packet=packet, pins=pins, ctx_norm_pin=ctx_norm_pin,
                   pmap=pmap, map_pin=map_pin, kind=kind, included=included, subset=subset, mems=mems, s1=s1, terminal=terminal,
                   binding_sha256=binding["sha256"] if binding else None)
    slots = ad.slot_report
    state["slots"] = slots
    if terminal and (slots["unaccounted_slots"] or slots["null_identity_unresolved"]):
        raise Stop("unaccounted_packet_slots", f"terminal index: {slots['unaccounted_slots']} packet slots have no index row "
                                               f"outside a retired fold; {slots['null_identity_unresolved']} null-identity rows "
                                               "resolve to no packet slot")
    if mode == "production" and slots["reconciled_with_release"] is False:
        raise Stop("retired_slots_reconciliation", f"{slots['retired_component_slots']} retired slots != release summary "
                                                   f"retired_component_slots {slots['release_retired_component_slots']}")
    # tier I: every included context, no GET (JTT1 headers come from the cache for the subset)
    index_level = ad.index_level()
    for c in sorted(included, key=lambda c: c["context_sha256"]):
        guard.tick()
        ad.tier_i(c)
    tier_i_counts = _counts(ad.rows.values())
    # tier O: outcomes for the subset, in sample order
    o_comps = DRY_RUN_OUTCOME_COMPONENTS if ad.dry else PRODUCTION_OUTCOME_COMPONENTS
    o_rows, a_rows = [], []
    try:
        for cid in subset:
            guard.tick()
            for k in o_comps:
                row = ad.rows[(cid, k)]
                if row["status"] != "bound_index_only":
                    continue
                row["status"] = ad.tier_o(row, by_id[cid])
                o_rows.append(row)
    except BudgetExhausted as err:
        ad.budget["tier_O"] = {"reason": err.reason, "detail": err.detail}
    # tier A: arrays with whatever outcome+array budget remains, in sample order, for tier-O-adopted rows
    a_comps = DRY_RUN_ARRAY_COMPONENTS if ad.dry else PRODUCTION_ARRAY_COMPONENTS
    if ad.budget["tier_O"] is None:
        try:
            for cid in subset:
                guard.tick()
                for k in a_comps:
                    row = ad.rows[(cid, k)]
                    if row["status"] != "adopted" or "_o" not in row:
                        continue
                    row["status"] = ad.tier_a(row, by_id[cid])
                    a_rows.append(row)
        except BudgetExhausted as err:
            ad.budget["tier_A"] = {"reason": err.reason, "detail": err.detail}
    ad.finalize_labels()
    # outputs
    rows = list(ad.rows.values())
    by_part = collections.defaultdict(list)
    for r in rows:
        by_part[r["split_role"]].append(r)
    if any(set(r["arms_eligible"]) & set(DEV_ONLY_ARMS) for r in by_part.get("heldout", [])):
        raise Stop("dev_only_arm_heldout", "a heldout row lists a dev-only arm (LB-replica is never holdout-reported, G06)")
    row_refs, sealed_refs, writes = {}, {}, []
    for p in OPEN_PARTITIONS:
        data = _rows_bytes(by_part.get(p, []))
        rel = f"rows/{p}.jsonl"
        res = common.immutable_write(out / rel, data, mode=0o444)
        writes.append({"path": rel, "existed": res["existed"]})
        row_refs[p] = {"path": rel, "sha256": _sha(data), "bytes": len(data), "rows": len(by_part.get(p, []))}
    for name, parts in (("heldout", ("heldout",)), ("excluded", EXCLUDED_ROLES)):
        rs = [r for p in parts for r in by_part.get(p, [])]
        data = _rows_bytes(rs)
        rel = f"sealed/rows-{name}.jsonl"
        res = common.immutable_write(out / rel, data, mode=0o444)
        writes.append({"path": rel, "existed": res["existed"]})
        sealed_refs[name] = {"path": rel, "sha256": _sha(data), "bytes": len(data), "rows": len(rs)}
    sealed_bytes = common.canonical({"schema": SEALED_CLOSURE_SCHEMA, "experiment_id": EXPERIMENT_ID, "kind": "adoption",
                                     "rows": sealed_refs, "rule": "heldout and role-excluded rows are read only after unsealing"})
    res = common.immutable_write(out / "sealed/closure.json", sealed_bytes, mode=0o444)
    writes.append({"path": "sealed/closure.json", "existed": res["existed"]})
    incomplete = (ad.budget["tier_O"] is not None or ad.budget["tier_A"] is not None) and not ad.dry
    completion = production_completion(rows)
    if ad.dry:
        status = "DRY_RUN"
    elif incomplete:
        status = "BLOCKED"
    elif completion["unfinished_bound_rows"] or completion["adopted_without_jtt1_evidence"]:
        status = "FAIL"
    else:
        status = "PASS"
    roles = collections.Counter(ad.pmap[c["context_sha256"]]["split_role"] for c in included)
    doc = {"schema": ADOPTION_SCHEMA, "experiment_id": EXPERIMENT_ID, "mode": mode, "gate": gate, "status": status,
           "upstream": CODEX_UPSTREAM,
           "scope": {"tier_I": "every included epoch context (no GET)",
                     "tier_O": "outcome GETs for the --contexts subset (dry run) or every included context (production), in order",
                     "tier_A": "array GETs with the remaining outcome+array budget", "tier_O_components": list(o_comps),
                     "tier_A_components": list(a_comps), "subset": len(subset)},
           "inputs": state["inputs"], "component_pins": pins, "quarantine": state["quarantine"], "slots": slots,
           "completion": completion,
           "counts": {"tier_I": tier_i_counts, "tier_O": _counts(o_rows), "tier_A": _counts(a_rows), "final": _counts(rows),
                      "residual_by_reason": residual_totals(rows),
                      "contexts": {"included": len(included), "subset": len(subset), "by_role": dict(sorted(roles.items())),
                                   "job_id_recomputed": ad.job_id_checked, "job_id_equal": ad.job_id_ok,
                                   "packet_duplicate_identities": ad.packet_dups},
                      "index_level": index_level},
           "labels": {"train_only": "CLAIMED only for bound_index_only and adopted rows (null otherwise); structural and content "
                                    "evidence is recorded; upgrading the label needs a re-encode counterfactual (encoder "
                                    "inference), out of scope (arm gate adopted_train_only_counterfactual)",
                      "recipe_identity": RECIPE_IDENTITY, "arms_eligible": ARMS_ELIGIBLE,
                      "arms_eligible_rule": "a heldout row drops the dev-only arms (LB-replica, G06); a heldout target row is "
                                            "not_for_arms:dev_only_arm and is never fetched",
                      "legacy_projection": {"context2048": True},
                      "raw_foundation_2048_retained": "true only for adopted foundation rows (the tier-O array contract, and at "
                                                      "tier A the NPZ, show the raw [2048] vector); null otherwise",
                      "pre_normalization": "true only for adopted rows; null otherwise",
                      "jtt1_binding": JTT1_BINDING_NOTE,
                      "encoder_license": "per row and in component_pins, copied from encoding-inventory.json (DL-D3a); "
                                         "research_only marks components never to be served or redistributed",
                      "encoder_weight_exposure": "Unknown until P2", "rights_source": "codex_packet licence status until P2"},
           "budgets": {"caps": dict(ledger.caps), "combined": {n: {"kinds": list(k), "cap": c} for n, (k, c) in ledger.combined.items()},
                       "budget_exhausted": dict(ad.budget)},
           "rows": row_refs, "sealed": {"closure": {"path": "sealed/closure.json", "sha256": _sha(sealed_bytes), "bytes": len(sealed_bytes)}},
           "rule": "reference only: no array bytes on disk, nothing written to S3; release, index and packet bound by sha256"}
    _validate(doc, "jope.dl-encoding-adoption.schema.json")
    data = common.receipt_bytes(doc)
    res = common.immutable_write(out / "adoption.json", data, mode=0o444)
    writes.append({"path": "adoption.json", "existed": res["existed"]})
    state["writes"] = writes
    state["adoption"] = {"path": "adoption.json", "sha256": _sha(data), "bytes": len(data)}
    state["counts"] = doc["counts"]
    code = {"BLOCKED": EXIT_BLOCKED, "FAIL": EXIT_FAIL}.get(status, EXIT_PASS)
    return code, status


def cmd_adopt(a) -> tuple:
    try:
        cpuguard.require_slice_limits()
    except Exception as err:  # noqa: BLE001
        return EXIT_BLOCKED, {"status": "BLOCKED", "reason": "slice_limits", "detail": str(err)}
    try:
        release_ref = _load_ref(a.release_ref, "release_ref", default_bucket=CODEX_BUCKET)
        index_ref = _load_ref(a.index_ref, "index_ref", default_bucket=CODEX_BUCKET)
        if a.content_checks and not a.jtt1_cache:
            raise Refused("content_checks_need_jtt1_cache", "--content-checks reads JTT1 train rows from --jtt1-cache")
        if not a.dry_run and a.contexts:
            raise Refused("contexts_in_production", "production adoption covers every included context (tiers O and A); "
                                                    "--contexts is for --dry-run (AB-3)")
        if not a.dry_run and not (a.content_checks and a.jtt1_cache):
            raise Refused("production_needs_content_checks", "production adoption needs --content-checks and --jtt1-cache: "
                                                             "every adopted row carries JTT1-derived evidence (TOL-07, AB-3)")
        mode = _adopt_output_gate(a, release_ref, index_ref)
    except EncodingError as err:
        return err.exit_code, {"status": err.status, "reason": err.reason, "detail": err.detail}
    out = _real(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp, started = _stamp(), common.utc_us()
    guard = cpuguard.Guard(journal_path=out / "guard-journal.jsonl", max_pause_s=GUARD_MAX_PAUSE_S)
    ledger = GetLedger({"release": a.release_get_budget, "outcome": a.outcome_get_budget, "array": a.outcome_get_budget,
                        "jtt1": a.jtt1_get_budget}, combined={"outcome+array": (("outcome", "array"), a.outcome_get_budget)})
    fetch = _Fetcher(ledger, a.jtt1_cache)
    state = {"release_ref": release_ref, "index_ref": index_ref, "inputs": {}, "writes": [], "gate": mode}
    failure = None
    try:
        guard.wait()
        code, status = _adopt(a, out, guard, fetch, ledger, state)
    except EncodingError as err:
        code, status, failure = err.exit_code, err.status, {"reason": err.reason, "detail": err.detail}
    except cpuguard.GuardTimeout as err:
        code, status, failure = EXIT_BLOCKED, "BLOCKED", {"reason": "guard_timeout", "detail": str(err)}
    except common.ImmutableConflict as err:
        code, status, failure = EXIT_STOP, "STOP", {"reason": "immutable_conflict", "detail": str(err)}
    except jtt1.JTTError as err:
        code, status, failure = EXIT_STOP, "STOP", {"reason": "jtt1_refused", "detail": str(err)}
    except Exception as err:  # noqa: BLE001 (an unexpected error still leaves a FAIL receipt)
        code, status, failure = EXIT_FAIL, "FAIL", {"reason": f"error:{type(err).__name__}", "detail": str(err)}
    ledger_doc = ledger.doc()
    lres = common.write_receipt(out / f"ledger-{stamp}.json", ledger_doc)
    receipt = {"schema": ADOPTION_RECEIPT_SCHEMA, "packet": "P6-adopt" + ("-dry-run" if a.dry_run else ""),
               "experiment_id": EXPERIMENT_ID, "status": status, "exit_code": code, "failure": failure, "gate": mode,
               "started_utc_us": started, "utc_us": common.utc_us(), "adoption": state.get("adoption"), "counts": state.get("counts"),
               "release_fetched_now": state.get("release_fetched_now"), "slots": state.get("slots"),
               "inputs": state["inputs"], "writes": state["writes"],
               "budgets": {"caps": ledger_doc["caps"], "used": ledger_doc["used"], "combined": ledger_doc["combined"]},
               "ledger": {"path": lres["path"], "sha256": lres["sha256"], "gets": ledger_doc["gets"], "bytes": ledger_doc["bytes"],
                          "all_sha_verified": ledger_doc["all_sha_verified"], "puts": 0},
               "source_tree": {"encodings": {"sha256": source_tree_sha256("encodings"), "files": source_tree("encodings")}},
               "runtime": runtime_identity(), "guard": guard.summary(), "peak_rss_bytes": peak_rss_bytes()}
    rres = common.write_receipt(out / f"adoption-receipt-{stamp}.json", receipt)
    return code, {"status": status, "exit_code": code, "failure": failure, "counts": state.get("counts"),
                  "adoption": state.get("adoption"), "receipt": {"path": rres["path"], "sha256": rres["sha256"]},
                  "ledger": {"path": lres["path"], "used": ledger_doc["used"], "gets": ledger_doc["gets"]}, "writes": state["writes"]}


# ------------------------------------------------------------------------------------------------- helpers CLI
def sample_contexts(epoch_doc, seed=20261005, per_stratum=4) -> dict:
    """Strata = sorted (cohort, fold) groups of included contexts; within a stratum contexts are sorted by context_sha256;
    one random.Random(seed) is shared across strata and rng.sample(pool, per_stratum) is drawn per stratum."""
    strata = collections.defaultdict(list)
    for c in included_contexts(epoch_doc):
        strata[(c["cohort"], c["fold"])].append(c["context_sha256"])
    rng = random.Random(seed)
    picked, table = [], []
    for key in sorted(strata):
        pool = sorted(strata[key])
        take = rng.sample(pool, min(per_stratum, len(pool)))
        picked.extend(take)
        table.append({"cohort": key[0], "fold": key[1], "pool": len(pool), "contexts": take})
    return {"contexts": picked, "strata": table, "sha256": _sha(common.canonical(picked))}


def fake_partition_map(epoch_doc, seed="p6-fake-map/v1", heldout=0.2, dev=0.2, diag=0.1) -> tuple:
    """Deterministic fake map in the P4a format {context_sha256: {family_id, split_role}} (demonstration only)."""
    inc = included_contexts(epoch_doc)

    def prio(domain, unit):
        return common.sha256_bytes(f"{domain}\0{seed}\0{unit}".encode("utf-8"))

    fam_of = {c["context_sha256"]: "fake-fam-" + c["dataset_hash"] for c in inc}
    fams = sorted(set(fam_of.values()))
    held = set(sorted(fams, key=lambda f: (prio("heldout", f), f))[:_ceil_frac(heldout, len(fams))])
    rest = [f for f in fams if f not in held]
    devs = set(sorted(rest, key=lambda f: (prio("dev", f), f))[:_ceil_frac(dev, len(rest))])
    role = {cid: ("heldout" if f in held else "dev" if f in devs else "train_fit") for cid, f in fam_of.items()}
    eligible = sorted((c["context_sha256"] for c in inc if role[c["context_sha256"]] == "train_fit"
                       and not (c["cohort"] == "frozen" and c["fold"] == 0)), key=lambda cid: (prio("diag", cid), cid))
    for cid in eligible[:_ceil_frac(diag, len(eligible))]:
        role[cid] = "diag"
    pmap = {cid: {"family_id": fam_of[cid], "split_role": role[cid]} for cid in sorted(fam_of)}
    counts = collections.Counter(role.values())
    meta = {"schema": FAKE_MAP_META_SCHEMA, "kind": "fake", "seed": seed, "fractions": {"heldout": heldout, "dev": dev, "diag": diag},
            "counts": {"families": len(fams), "heldout_families": len(held), "dev_families": len(devs), "contexts": len(pmap),
                       "diag_eligible": len(eligible), **{p: counts.get(p, 0) for p in PARTITIONS}},
            "rule": ("family = 'fake-fam-' + dataset_hash; prio(domain, unit) = sha256(domain NUL seed NUL unit); heldout = first "
                     "ceil(f*F) families by (prio('heldout'), family); dev = first ceil(f*|rest|) of the rest by prio('dev'); diag = "
                     "first ceil(f*|eligible|) remaining train_fit contexts excluding frozen fold 0 by prio('diag'); else train_fit")}
    return pmap, meta


def bench_a0(n=1700, ps=(9, 64, 344, 1025, 2048), repeats=3, seed=20261005) -> dict:
    a0m = _a0()
    guard = cpuguard.Guard()
    rng = np.random.default_rng(seed)
    out = []
    for p in ps:
        X = rng.normal(size=(n, p))
        w = rng.normal(size=p) / math.sqrt(p)
        y = np.einsum("ij,j->i", X, w, optimize=False) + 0.1 * rng.normal(size=n)
        g = (np.arange(n) % 2).astype(np.uint8)
        times = []
        for _ in range(repeats):
            guard.tick()
            t0 = time.perf_counter()
            v = a0m.features(X, y, g)
            times.append(time.perf_counter() - t0)
        out.append({"n": n, "p": p, "min_s": min(times), "all_s": times, "peak_rss_bytes": peak_rss_bytes(),
                    "vector_sha256": _sha(a0m.to_f32le(v))})
        del X
    return {"schema": BENCH_SCHEMA, "producer": _a0_producer_tree(a0m), "repeats": repeats, "seed": seed, "results": out,
            "runtime": runtime_identity(), "guard": guard.summary(), "data": "synthetic normal X, linear y; no corpus data"}


def cmd_sample(a) -> tuple:
    _require_run_tmp(a.out, "sample")
    epoch_doc, epoch_pin = load_epoch(a.epoch)
    seed = int(a.seed)
    doc = sample_contexts(epoch_doc, seed=seed, per_stratum=a.per_stratum)
    doc = {"schema": SAMPLE_SCHEMA, "epoch": {"path": epoch_pin["path"], "sha256": epoch_pin["sha256"]}, "seed": seed,
           "per_stratum": a.per_stratum, **doc}
    data = common.canonical(doc)
    res = common.immutable_write(a.out, data, mode=0o444)
    return EXIT_PASS, {"status": "PASS", "path": res["path"], "sha256": res["sha256"], "contexts": len(doc["contexts"]),
                       "contexts_sha256": doc["sha256"], "strata": [[s["cohort"], s["fold"], s["pool"]] for s in doc["strata"]]}


def cmd_fake_map(a) -> tuple:
    _require_run_tmp(a.out, "fake_partition_map")
    epoch_doc, epoch_pin = load_epoch(a.epoch)
    pmap, meta = fake_partition_map(epoch_doc, seed=a.seed, heldout=a.heldout, dev=a.dev, diag=a.diag)
    data = common.canonical(pmap)
    meta = dict(meta, epoch={"path": epoch_pin["path"], "sha256": epoch_pin["sha256"]}, map={"sha256": _sha(data), "bytes": len(data)})
    res = common.immutable_write(a.out, data, mode=0o444)
    mres = common.immutable_write(str(a.out) + ".meta.json", common.canonical(meta), mode=0o444)
    return EXIT_PASS, {"status": "PASS", "path": res["path"], "sha256": res["sha256"], "bytes": res["bytes"], "meta": mres["path"],
                       "counts": meta["counts"]}


def cmd_bench(a) -> tuple:
    _require_run_tmp(a.out, "bench")
    ps = tuple(int(x) for x in str(a.p).split(",") if x.strip())
    doc = bench_a0(n=a.n, ps=ps, repeats=a.repeats, seed=a.seed)
    res = common.write_receipt(a.out, doc)
    return EXIT_PASS, {"status": "PASS", "path": res["path"], "results": [{k: r[k] for k in ("p", "min_s", "peak_rss_bytes")} for r in doc["results"]]}


def cmd_schemas(a) -> tuple:
    written = []
    for name, schema in sorted(SCHEMAS.items()):
        res = common.immutable_write(Path(a.out) / name, (json.dumps(schema, indent=1, sort_keys=True) + "\n").encode(), mode=0o444)
        written.append({"path": res["path"], "sha256": res["sha256"], "existed": res["existed"]})
    return EXIT_PASS, {"status": "PASS", "written": written}


def cmd_map_audit(a) -> tuple:
    summary, rows = map_audit(a.epoch, a.index, a.index_sha256, a.cache)
    doc = {"schema": "jope.dl-encoding-map-audit/v1", "utc_us": common.utc_us(), "epoch": {"path": a.epoch, "sha256": common.sha256_file(a.epoch)},
           "summary": summary, "rows": rows}
    res = common.write_receipt(a.out, doc)
    return EXIT_PASS, {"receipt": {k: res[k] for k in ("path", "sha256", "bytes")}, "summary": summary}


# ---------------------------------------------------------------------------------------------- JSON Schemas
_REF = {"type": "object", "required": list(PIN_FIELDS),
        "properties": {"bucket": {"type": "string"}, "key": {"type": "string"}, "version_id": {"type": "string"},
                       "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}, "bytes": {"type": "integer", "minimum": 0}}}
_HEX = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
_HEX_OR_NULL = {"anyOf": [_HEX, {"type": "null"}]}
_ARM_ENUM = {"enum": list(ARMS)}
_ABI = {"type": "string", "pattern": "^dl-v1-e0:(A0|A1|A2|A3|LB-replica):[0-9a-f]{16}$"}
_SCOPE = {"type": "object", "required": ["kind", "n"], "properties": {"kind": {"enum": ["all", "subset"]}, "n": {"type": "integer", "minimum": 0}}}
_BINDING_REC = {"type": "object",
                "required": ["schema", "sha256", "partition_map_sha256", "epoch_sha256", "holdout_sha256", "holdout_binding_sha256",
                             "holdout_ref", "p3_receipt_sha256", "p5_receipt_sha256"],
                "properties": {"schema": {"const": MAP_BINDING_SCHEMA}, "sha256": _HEX, "partition_map_sha256": _HEX,
                               "epoch_sha256": _HEX, "holdout_sha256": _HEX, "holdout_binding_sha256": _HEX, "holdout_ref": _REF,
                               "p3_receipt_sha256": _HEX, "p5_receipt_sha256": _HEX_OR_NULL}}
_KIND_P3 = {"properties": {"kind": {"const": "p3"}}, "required": ["kind"]}
_INPUTS = {"type": "object", "required": ["epoch", "partition_map", "holdout_binding", "adoption_receipt"], "additionalProperties": False,
           "properties": {"epoch": {"type": "object", "required": ["path", "sha256", "s3ref"],
                                    "properties": {"path": {"type": "string"}, "sha256": _HEX, "s3ref": {"anyOf": [_REF, {"type": "null"}]}}},
                          "partition_map": {"type": "object", "required": ["path", "sha256", "kind", "binding"],
                                            "properties": {"path": {"type": ["string", "null"]}, "sha256": _HEX, "kind": {"enum": ["p3", "fake"]},
                                                           "binding": {"anyOf": [{"type": "null"}, _BINDING_REC]}},
                                            # p3 is a positive property: bound to the P3 freeze (TOL-01); a fake map carries no binding
                                            "allOf": [{"if": _KIND_P3, "then": {"properties": {"binding": _BINDING_REC}},
                                                       "else": {"properties": {"binding": {"type": "null"}}}}]},
                          "holdout_binding": {"anyOf": [{"type": "null"}, {"type": "object", "required": ["sha256"], "properties": {"sha256": _HEX}}]},
                          "adoption_receipt": _HEX_OR_NULL},
           "allOf": [{"if": {"properties": {"partition_map": _KIND_P3}, "required": ["partition_map"]},
                      "then": {"properties": {"holdout_binding": {"type": "object", "required": ["sha256"]}}}}]}
_LICENSE = {"type": "object", "required": ["class"], "properties": {"class": {"type": "string"}}}
# an adopted block's component: every COMPONENT_PINS pin, seeds, row caps, the encoder licence and research_only (AB-8, AB-9)
_COMPONENT = {"type": "object",
              "required": ["component", "weights", "recipe", "pins", "seeds", "row_caps", "encoder_license", "research_only"],
              "properties": {"component": {"enum": list(COMPONENTS)}, "weights": _REF, "recipe": {"type": ["string", "object"]},
                             "pins": {"type": "array", "minItems": 1,
                                      "items": {"type": "object", "required": ["model_ref", "ref"],
                                                "properties": {"model_ref": {"type": "string"}, "ref": _REF}}},
                             "seeds": {"type": "object", "additionalProperties": {"type": "integer"}},
                             "row_caps": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 1}},
                             "encoder_license": _LICENSE, "research_only": {"type": "boolean"}},
              "allOf": [{"if": {"properties": {"component": {"const": comp}}, "required": ["component"]},
                         "then": {"properties": {"pins": {"allOf": [
                             {"contains": {"properties": {"model_ref": {"const": spec_item[0]}}, "required": ["model_ref"]}}
                             for spec_item in spec]}}}}
                        for comp, spec in COMPONENT_PINS.items() if spec]}
_ADOPTED = {"properties": {"producer": {"properties": {"kind": {"const": "adopted"}}, "required": ["kind"]}}, "required": ["producer"]}
_NORM_REC = {"type": "object", "additionalProperties": False,
             "required": ["block_id", "normalizer_sha256", "blob_sha256", "fit_family_set_sha256", "fit_context_set_sha256", "rank", "width_full"],
             "properties": {"block_id": {"type": "string"}, "normalizer_sha256": _HEX, "blob_sha256": _HEX, "fit_family_set_sha256": _HEX,
                            "fit_context_set_sha256": _HEX, "rank": {"type": "integer", "minimum": 0}, "width_full": {"type": "integer", "minimum": 1}}}
_PRODUCER = {"oneOf": [
    {"type": "object", "required": ["kind", "module", "version", "feature_list_sha256", "source_tree_sha256"],
     "properties": {"kind": {"const": "computed"}, "module": {"type": "string"}, "version": {"type": "string"},
                    "feature_list_sha256": _HEX, "source_tree_sha256": _HEX}},
    {"type": "object", "required": ["kind", "upstream", "release_ref", "index_ref", "packet_ref"],
     "properties": {"kind": {"const": "adopted"}, "upstream": {"const": CODEX_UPSTREAM}, "release_ref": _REF, "index_ref": _REF,
                    "packet_ref": _REF}}]}
_BLOCK = {"type": "object", "additionalProperties": False, "required": ["block_id", "producer", "components", "dtype", "shape", "pre_normalization"],
          "properties": {"block_id": {"type": "string"}, "producer": _PRODUCER,
                         "components": {"type": "array", "items": _COMPONENT},
                         "dtype": {"const": "<f4"}, "shape": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 1},
                         "pre_normalization": {"const": True}},
          "allOf": [{"if": _ADOPTED, "then": {"properties": {"components": {"minItems": 1}}}}]}
_ARRAY = {"type": "object", "required": ["block_id"],
          "properties": {"block_id": {"type": "string"}, "sha256": _HEX, "bytes": {"type": "integer", "minimum": 1}, "dtype": {"const": "<f4"},
                         "shape": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                         "path": {"type": "string", "pattern": "^(sealed/)?arrays/[0-9a-f]{64}\\.f32le$"},
                         "npz_ref": _REF, "member": {"type": "string"}},
          "oneOf": [{"required": ["sha256", "bytes", "dtype", "shape", "path"]}, {"required": ["npz_ref", "member", "dtype", "shape"]}]}
_EXPOSURE = {"type": "object", "required": ["finding", "shifts", "scale", "column_set"],
             "properties": {"finding": {"const": UPSTREAM_FINDING}, "shifts": {"type": "array", "items": {"type": "number"}},
                            "scale": {"type": "object", "required": ["exposed", "y", "x_columns_exposed", "x_columns"]},
                            "column_set": {"type": "object", "required": ["exposed", "train_constant_columns", "columns"]}}}
_CONTEXT = {"type": "object", "additionalProperties": False,
            "required": ["context_sha256", "cohort", "fold", "dataset_hash", "family_id", "split_role", "rights_status", "rights_source",
                         "input", "membership_sha256", "outer_train_indices_sha256", "train_rows", "features", "train_row_sha256",
                         "arrays", "status", "missing_reason", "train_only", "train_only_scope", "upstream_exposure"],
            "properties": {"context_sha256": _HEX, "cohort": {"enum": ["frozen", "incoming"]}, "fold": {"type": "integer", "minimum": 0},
                           "dataset_hash": {"type": "string"}, "family_id": {"type": "string", "minLength": 1},
                           "split_role": {"enum": list(PARTITIONS)}, "rights_status": {"type": "string"}, "rights_source": {"type": "string"},
                           "input": {"type": "object", "required": ["key", "version_id", "sha256", "raw_sha256", "raw_bytes"],
                                     "properties": {"bucket": {"type": "string"}, "key": {"type": "string"}, "version_id": {"type": "string"},
                                                    "sha256": _HEX, "bytes": {"type": "integer"}, "raw_sha256": _HEX, "raw_bytes": {"type": "integer"}}},
                           "membership_sha256": _HEX, "outer_train_indices_sha256": _HEX_OR_NULL,
                           "train_rows": {"type": "integer", "minimum": 1}, "features": {"type": "integer", "minimum": 1},
                           "train_row_sha256": _HEX, "arrays": {"type": "array", "items": _ARRAY},
                           "status": {"enum": ["present", "missing"]}, "missing_reason": {"type": ["string", "null"]},
                           # a manifest never says VERIFIED: that label lives only in a PASS verify receipt (TOL-03)
                           "train_only": {"enum": [A0_TRAIN_ONLY, ADOPTED_TRAIN_ONLY, None]},
                           "train_only_scope": {"anyOf": [{"type": "null"},
                                                          {"type": "string", "pattern": f"^{TRAIN_ONLY_SCOPE}(; [a-z_]+)*$"}]},
                           "upstream_exposure": {"anyOf": [{"type": "null"}, _EXPOSURE]}},
            "allOf": [{"if": {"properties": {"status": {"const": "present"}}},
                       "then": {"properties": {"arrays": {"minItems": 1}, "missing_reason": {"type": "null"}}},
                       "else": {"properties": {"arrays": {"maxItems": 0}, "missing_reason": {"type": "string"},
                                               "train_only": {"type": "null"}}}},
                      {"if": {"properties": {"train_only": {"const": A0_TRAIN_ONLY}}, "required": ["train_only"]},
                       "then": {"properties": {"train_only_scope": {"type": "string"}, "upstream_exposure": _EXPOSURE}}}]}
SCHEMAS = {
    "jope.dl-context-encoding.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "jope.dl-context-encoding/v1", "title": CONTEXT_ENCODING_SCHEMA,
        "type": "object", "additionalProperties": False,
        "required": ["schema", "experiment_id", "arm", "partition", "abi_id", "width_full", "inputs", "scope", "blocks", "normalizers",
                     "contexts", "coverage", "train_only_rule"],
        "properties": {"schema": {"const": CONTEXT_ENCODING_SCHEMA}, "experiment_id": {"const": EXPERIMENT_ID}, "arm": _ARM_ENUM,
                       "partition": {"enum": list(PARTITIONS)}, "abi_id": _ABI, "width_full": {"type": "integer", "minimum": 1},
                       "inputs": _INPUTS, "scope": _SCOPE, "blocks": {"type": "array", "minItems": 1, "items": _BLOCK},
                       "normalizers": {"type": "array", "minItems": 1, "items": _NORM_REC}, "contexts": {"type": "array", "items": _CONTEXT},
                       "encoder_weight_exposure": {"type": "object", "additionalProperties": {
                           "type": "object", "additionalProperties": {"enum": ["Unknown", "Exposed", "Unseen"]}}},
                       "coverage": {"type": "object", "required": ["admitted", "present", "missing"], "additionalProperties": False,
                                    "properties": {"admitted": {"type": "integer", "minimum": 0}, "present": {"type": "integer", "minimum": 0},
                                                   "missing": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 1}}}},
                       "train_only_rule": {"type": "string"}, "partition_note": {"type": "string"}},
        "allOf": [{"if": {"properties": {"partition": {"const": "heldout"}}}, "then": {"required": ["encoder_weight_exposure"]},
                   "else": {"not": {"required": ["encoder_weight_exposure"]}}},
                  {"if": {"properties": {"partition": {"const": "diag"}}}, "then": {"required": ["partition_note"]}},
                  {"if": {"properties": {"partition": {"const": "heldout"}}},
                   "then": {"properties": {"arm": {"not": {"enum": list(DEV_ONLY_ARMS)}}}}},
                  # an adopted block needs the adoption receipt it came from (AB-9)
                  {"if": {"properties": {"blocks": {"contains": _ADOPTED}}, "required": ["blocks"]},
                   "then": {"properties": {"inputs": {"properties": {"adoption_receipt": _HEX}}}}}]},
    "jope.dl-encoding-arm.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "jope.dl-encoding-arm/v1", "title": ARM_SCHEMA,
        "type": "object", "additionalProperties": False,
        "required": ["schema", "experiment_id", "arm", "abi_id", "width_full", "blocks", "widths_full", "inputs", "scope", "manifests", "heldout",
                     "train_only_rule"],
        "properties": {"schema": {"const": ARM_SCHEMA}, "experiment_id": {"const": EXPERIMENT_ID}, "arm": _ARM_ENUM, "abi_id": _ABI,
                       "width_full": {"type": "integer", "minimum": 1},
                       "blocks": {"type": "array", "minItems": 1, "items": {
                           "type": "object", "required": ["block_id", "informative_dims", "width_full", "normalizer_sha256", "producer"],
                           "properties": {"block_id": {"type": "string"}, "informative_dims": {"type": "integer", "minimum": 1},
                                          "width_full": {"type": "integer", "minimum": 1}, "normalizer_sha256": _HEX, "producer": _PRODUCER,
                                          "research_only": {"type": "boolean"},
                                          "encoder_licenses": {"type": "object", "additionalProperties": _LICENSE}},
                           # an adopted block carries its research_only flag and encoder licences (AB-8)
                           "allOf": [{"if": _ADOPTED, "then": {"required": ["research_only", "encoder_licenses"]}}]}},
                       "widths_full": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 1}},
                       "inputs": _INPUTS, "scope": _SCOPE,
                       "manifests": {"type": "object", "additionalProperties": False, "required": list(OPEN_PARTITIONS),
                                     "properties": {p: {"type": "object", "required": ["path", "sha256", "bytes", "contexts"],
                                                        "properties": {"path": {"type": "string", "pattern": "^manifests/"}, "sha256": _HEX,
                                                                       "bytes": {"type": "integer"}, "contexts": {"type": "integer"}}}
                                                    for p in OPEN_PARTITIONS}},
                       "heldout": {"type": "string"}, "train_only_rule": {"type": "string"}},
        "allOf": [{"if": {"properties": {"blocks": {"contains": _ADOPTED}}, "required": ["blocks"]},
                   "then": {"properties": {"inputs": {"properties": {"adoption_receipt": _HEX}}}}}]},
    "jope.dl-normalizer.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "jope.dl-normalizer/v1", "title": NORMALIZER_SCHEMA,
        "description": "The canonical JSON header <normalizer_sha256>.json of one block normalizer (dltrain.normalizers); its blob "
                       "<blob_sha256>.f32le holds the arrays in the table order. Fit sets are train_fit only (DL-D4a).",
        "type": "object", "additionalProperties": False,
        "required": ["schema", "experiment_id", "arm", "block_id", "block_version", "feature_list_sha256", "method", "dims", "fit",
                     "arrays", "blob"],
        "properties": {
            "schema": {"const": NORMALIZER_SCHEMA}, "experiment_id": {"const": EXPERIMENT_ID}, "arm": _ARM_ENUM,
            "block_id": {"type": "string", "minLength": 1}, "block_version": {"type": "string", "minLength": 1},
            "feature_list_sha256": _HEX_OR_NULL,
            "method": {"type": "object", "additionalProperties": False, "required": list(NORMALIZER_METHOD_KEYS),
                       "properties": {"standardize": {"type": "string"}, "pca": {"type": "string"},
                                      "projection_scale": {"type": "string"}, "clip_abs": {"type": "number", "exclusiveMinimum": 0},
                                      "rank_rtol": {"type": "number", "exclusiveMinimum": 0}, "dtype_compute": {"const": "float64"},
                                      "dtype_stored": {"const": "<f4"}, "gram": {"type": "string"}, "eigh": {"type": "string"}}},
            "dims": {"type": "object", "additionalProperties": False,
                     "required": ["in", "rank", "width_full", "constant_dims", "floored_components"],
                     "properties": {"in": {"type": "integer", "minimum": 1}, "rank": {"type": "integer", "minimum": 1},
                                    "width_full": {"type": "integer", "minimum": 1},
                                    "constant_dims": {"type": "array", "uniqueItems": True, "items": {"type": "integer", "minimum": 0}},
                                    "floored_components": {"type": "integer", "minimum": 0}}},
            "fit": {"type": "object", "additionalProperties": False,
                    "required": ["partition_role", "n_contexts", "n_families", "fit_context_set_sha256", "fit_family_set_sha256",
                                 "partition_map_sha256", "partition_map_kind", "partition_map_binding_sha256", "epoch_sha256",
                                 "holdout_binding_sha256"],
                    "properties": {"partition_role": {"const": "train_fit"}, "n_contexts": {"type": "integer", "minimum": 2},
                                   "n_families": {"type": "integer", "minimum": 1}, "fit_context_set_sha256": _HEX,
                                   "fit_family_set_sha256": _HEX, "partition_map_sha256": _HEX,
                                   "partition_map_kind": {"enum": ["p3", "fake"]}, "partition_map_binding_sha256": _HEX_OR_NULL,
                                   "epoch_sha256": _HEX, "holdout_binding_sha256": _HEX_OR_NULL},
                    # p3: bound (both binding hashes); fake: neither (TOL-01)
                    "allOf": [{"if": {"properties": {"partition_map_kind": {"const": "p3"}}, "required": ["partition_map_kind"]},
                               "then": {"properties": {"partition_map_binding_sha256": _HEX, "holdout_binding_sha256": _HEX}},
                               "else": {"properties": {"partition_map_binding_sha256": {"type": "null"},
                                                       "holdout_binding_sha256": {"type": "null"}}}}]},
            "arrays": {"type": "array", "minItems": len(NORMALIZER_ARRAY_ORDER), "maxItems": len(NORMALIZER_ARRAY_ORDER),
                       "prefixItems": [{"properties": {"name": {"const": name}}} for name in NORMALIZER_ARRAY_ORDER],
                       "items": {"type": "object", "additionalProperties": False, "required": ["name", "dtype", "shape", "offset", "bytes"],
                                 "properties": {"name": {"enum": list(NORMALIZER_ARRAY_ORDER)}, "dtype": {"const": "<f4"},
                                                "shape": {"type": "array", "minItems": 1, "maxItems": 2,
                                                          "items": {"type": "integer", "minimum": 1}},
                                                "offset": {"type": "integer", "minimum": 0}, "bytes": {"type": "integer", "minimum": 4}}}},
            "blob": {"type": "object", "additionalProperties": False, "required": ["sha256", "bytes"],
                     "properties": {"sha256": _HEX, "bytes": {"type": "integer", "minimum": 4}}}}},
    "jope.dl-encoding-adoption.schema.json": {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "jope.dl-encoding-adoption/v1", "title": ADOPTION_SCHEMA,
        "type": "object", "additionalProperties": False,
        "required": ["schema", "experiment_id", "mode", "gate", "status", "upstream", "scope", "inputs", "component_pins", "quarantine",
                     "slots", "completion", "counts", "labels", "budgets", "rows", "sealed", "rule"],
        "properties": {"schema": {"const": ADOPTION_SCHEMA}, "experiment_id": {"const": EXPERIMENT_ID},
                       "mode": {"enum": ["dry_run", "production"]}, "gate": {"enum": ["sample", "production"]},
                       "status": {"enum": ["PASS", "DRY_RUN", "BLOCKED", "FAIL"]},
                       "upstream": {"const": CODEX_UPSTREAM}, "scope": {"type": "object"},
                       "inputs": {"type": "object", "required": ["release", "index", "packet", "inventory", "epoch", "partition_map",
                                                                 "memberships", "contexts", "holdout_binding", "owner_waiver"],
                                  "properties": {"partition_map": {"type": "object", "required": ["kind", "sha256", "binding"]},
                                                 "index": {"type": "object", "required": ["fetch_receipt"]}}},
                       "component_pins": {"type": "object", "required": list(COMPONENTS), "additionalProperties": {
                           "type": "object", "required": ["status", "pins", "encoder_license", "research_only"],
                           "properties": {"status": {"enum": ["pinned", "mismatch", "unpinned"]}, "encoder_license": _LICENSE,
                                          "research_only": {"type": "boolean"}}}},
                       "quarantine": {"type": "object", "required": ["S1", "S2", "equal"]},
                       "slots": {"type": "object", "required": ["packet_slots", "retired_component_slots", "unaccounted_slots"]},
                       "completion": {"type": "object", "required": ["unfinished_bound_rows", "adopted_without_jtt1_evidence"]},
                       "counts": {"type": "object", "required": ["tier_I", "tier_O", "tier_A", "final", "contexts", "residual_by_reason"]},
                       "labels": {"type": "object", "required": ["train_only", "recipe_identity", "arms_eligible"]},
                       "budgets": {"type": "object", "required": ["caps", "combined"]},
                       "rows": {"type": "object", "additionalProperties": False, "required": list(OPEN_PARTITIONS),
                                "properties": {p: {"type": "object", "required": ["path", "sha256", "bytes", "rows"],
                                                   "properties": {"path": {"const": f"rows/{p}.jsonl"}, "sha256": _HEX}} for p in OPEN_PARTITIONS}},
                       "sealed": {"type": "object"}, "rule": {"type": "string"}}},
}


# ------------------------------------------------------------------------------------------------------- CLI
def _parser():
    ap = argparse.ArgumentParser(prog="dltrain.encodings")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ma = sub.add_parser("map-audit")
    ma.add_argument("--epoch", required=True)
    ma.add_argument("--index", required=True)
    ma.add_argument("--index-sha256", required=True)
    ma.add_argument("--cache", required=True)
    ma.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, choices=ARMS)
    r.add_argument("--epoch")
    r.add_argument("--partition-map")
    r.add_argument("--partition-map-kind", choices=("p3", "fake"), default="fake")
    r.add_argument("--holdout", help="P3 holdout.json (p3 map binding)")
    r.add_argument("--holdout-binding", help="P3 binding document bindings/<experiment>/holdout.json (p3 map binding)")
    r.add_argument("--p3-receipt", help="P3 PASS receipt binding this partition map's sha256 (p3 map binding)")
    r.add_argument("--jtt1-cache")
    r.add_argument("--jtt1-get-budget", type=int, default=0)
    r.add_argument("--out")
    r.add_argument("--contexts")
    r.add_argument("--memberships")
    r.add_argument("--p5-receipt")
    r.add_argument("--label")
    ad = sub.add_parser("adopt")
    ad.add_argument("--release-ref", required=True)
    ad.add_argument("--index", required=True)
    ad.add_argument("--index-ref", required=True)
    ad.add_argument("--packet", required=True)
    ad.add_argument("--inventory", required=True)
    ad.add_argument("--inventory-sha256", required=True)
    ad.add_argument("--epoch", required=True)
    ad.add_argument("--memberships", required=True)
    ad.add_argument("--partition-map", required=True)
    ad.add_argument("--partition-map-kind", choices=("p3", "fake"), required=True)      # no default (AB-1)
    ad.add_argument("--holdout")
    ad.add_argument("--holdout-binding")
    ad.add_argument("--p3-receipt")
    ad.add_argument("--p5-receipt")
    ad.add_argument("--index-fetch-receipt", help="exact-VersionId fetch receipt whose refs.index equals --index-ref")
    ad.add_argument("--owner-waiver", help="jope.dl-owner-waiver/v1 citation for unqualified release flags")
    ad.add_argument("--out", required=True)
    ad.add_argument("--contexts")
    ad.add_argument("--dry-run", action="store_true")
    ad.add_argument("--release-get-budget", type=int, default=1)
    ad.add_argument("--outcome-get-budget", type=int, default=0)
    ad.add_argument("--jtt1-cache")
    ad.add_argument("--jtt1-get-budget", type=int, default=0)
    ad.add_argument("--content-checks", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--arm", required=True, choices=ARMS)
    v.add_argument("--out", required=True)
    v.add_argument("--epoch", required=True)
    v.add_argument("--partition-map", required=True)
    v.add_argument("--jtt1-cache", required=True)
    v.add_argument("--contexts")
    v.add_argument("--holdout")
    v.add_argument("--holdout-binding")
    v.add_argument("--p3-receipt")
    v.add_argument("--p5-receipt")
    v.add_argument("--coverage", action="store_true")
    v.add_argument("--fit-closure", action="store_true")
    v.add_argument("--poison-test", action="store_true")
    v.add_argument("--poison-scope", default="all")
    v.add_argument("--counterfactual", action="store_true")
    v.add_argument("--affine-invariance", action="store_true")
    v.add_argument("--upstream-exposure", action="store_true")
    v.add_argument("--column-set-audit", action="store_true")
    s = sub.add_parser("sample")
    s.add_argument("--epoch", required=True)
    s.add_argument("--seed", default="20261005")
    s.add_argument("--per-stratum", type=int, default=4)
    s.add_argument("--out", required=True)
    f = sub.add_parser("fake-partition-map")
    f.add_argument("--epoch", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--seed", default="p6-fake-map/v1")
    f.add_argument("--heldout", type=float, default=0.2)
    f.add_argument("--dev", type=float, default=0.2)
    f.add_argument("--diag", type=float, default=0.1)
    b = sub.add_parser("bench-a0")
    b.add_argument("--n", type=int, default=1700)
    b.add_argument("--p", default="9,64,344,1025,2048")
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--seed", type=int, default=20261005)
    b.add_argument("--out", required=True)
    sc = sub.add_parser("schemas")
    sc.add_argument("--out", required=True)
    return ap


_COMMANDS = {"map-audit": cmd_map_audit, "run": cmd_run, "adopt": cmd_adopt, "verify": cmd_verify, "sample": cmd_sample,
             "fake-partition-map": cmd_fake_map, "bench-a0": cmd_bench, "schemas": cmd_schemas}


def run_cli(argv=None) -> tuple:
    a = _parser().parse_args(argv)
    try:
        return _COMMANDS[a.cmd](a)
    except EncodingError as err:
        return err.exit_code, {"status": err.status, "reason": err.reason, "detail": err.detail}
    except common.ImmutableConflict as err:
        return EXIT_STOP, {"status": "STOP", "reason": "immutable_conflict", "detail": str(err)}


def _main(argv=None) -> int:
    code, summary = run_cli(argv)
    print(json.dumps(summary, default=str))
    return code


if __name__ == "__main__":
    raise SystemExit(_main())

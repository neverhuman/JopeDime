"""PLAN §5 P3: freeze the 5% family holdout, the development split and the split-only diagnostic (PLAN §6.2).

The selection code (sha256, prio, derive_seed, freeze, assign_appended) is PLAN §6.2 verbatim in semantics. The only
additions are explicit STOPs where the verbatim code would otherwise raise IndexError or KeyError (a pool smaller than
two families, an empty dev split, a pool family missing from the registry). The commitment's ALGORITHM carries
code_sha256, a sha256 over the exact source of that code, the pool predicate and their constants, so a change to the
selection code after the commitment makes freeze, verify and publish STOP.

Selectable pool: independence_qualified in {auto, reviewed}, rights_status != prohibited, not sealed and not
seed_selection_exposed. The DL-D4b normalizer-exposure clause does not apply, because the owner chose DL-D4a (the commit
gate checks it). N is the pool size and k = ceil(0.05·N). Estimand: "qualified, unsealed, non-seed-exposed families only".

Run order: commit -> freeze -> verify --assignment-poison -> closure -> publish. PLAN §5 P3 lists the closure after the
binding. The closure is offline and changes no selection, so running it first only moves a cross-partition STOP ahead
of the irreversible binding. Each command writes a create-only receipt (jope.dl-holdout-receipt/v1, with the
source_tree_sha256 of this module's import closure and the runtime identity) under <dir>/receipts/ unless --receipt is
given, and exits 0 PASS, 2 STOP, 3 ERROR or 4 CONFLICT.
  commit  --registry R --registry-sha256 H --decisions D [--review-csv C] [--p2-receipt P [--p2-receipt-sha256 PS]]
          [--owner-decisions O --owner-decisions-sha256 OS] --out RUN/holdout
      Before any S3 call, two gates run. The review gate needs phase=final, a decision file the registry binds, all 10
      seed rows confirm_mapping or add_members, and no reject_mapping. It applies the P2 provenance rule itself (review
      fix P3I-1: families.check_provenance with real=True and RUN/receipts/P2): no DRAFT or SIMULATION marker, an
      ISO-8601 UTC message time and a pinned owner record. P3 accepts that record only directly in RUN/receipts/P2
      (the stored owner message dl-d5-owner-message-<sha16>.txt hashing to message_sha256, or a record there) or, for
      an owner_directive, at the canonical OWNER_DIRECTIVES.md path (resolved, not by basename) or named by an owner
      message stored there (D-MAP-5). The final registry's own marker, decision_deltas.provenance, must read draft
      false and simulation false with that same record. The P3 gate (PLAN §5 P3 Gate) needs a PASS P2
      packet receipt (default RUN/receipts/P2/receipt.json) that carries this registry's S3 ref at
      objects/family-registry/sha256/<H>.json, and pinned owner decisions that record DL-D2 = a, DL-D4 = a,
      DL-D6 = commitment-derived and DL-D12 = keep sealed. The first S3 call then GETs the registry at exactly that
      ref's VersionId and needs this registry's sha256 and bytes (review fix P3I-2): a version that does not resolve
      or holds other bytes is a STOP. The commitment key's version history must be empty, or one
      version with no delete marker. The commitment {schema, experiment_id, pool_sha256, registry_sha256,
      review_decisions_sha256, algorithm, frac} is then PUT create-only at PREFIX bindings/<experiment>/
      holdout-commitment.json. A 412, 409, 5xx or transport failure is resolved by re-reading the key (PLAN §6.6
      publish step 1): equal bytes adopt that VersionId, different bytes are CONFLICT, and an absent key gets one more
      create-only PUT. Afterwards the history must be exactly the adopted version. seed = derive_seed(commitment_bytes,
      VersionId) goes to seed.hex before anything else, then commitment.json, commitment-ref.json and gate.json.
  freeze  --registry R --registry-sha256 H --seed-file RUN/holdout/seed.hex --out RUN/holdout
          [--context-stats S --context-stats-sha256 HS]
      Checks the seed against the commitment, the commitment against this registry, pool and selection code, and
      gate.json against the P2 receipt and owner decisions it names. Then runs freeze() and its invariant checks, which
      include an independent recomputation of the diag draw, and writes holdout.json: canonical JSON and a pure function
      of the registry bytes, the seed, the commitment and the gate. It carries the estimand, realized fractions at the
      family, dataset and fold grains, and the pool-versus-training composition. The receipt's expect block lists the
      PLAN §5 P3 Expect items that need per-context statistics: program and record grains, and n, p and target-scale
      quantiles. Only --context-stats reports them (holdout-stats-<sha16>.json); without it, expect.complete is false.
  verify  --holdout H --registry R --seed-file S [--registry-sha256 X] [--assignment-poison]
      Recomputes freeze() (k == ceil(0.05·N), exactly one assignment per registry family, every §6.2 invariant) and
      byte-compares holdout.json. --assignment-poison perturbs held-family records: contexts, frozen_fold0, every
      member leaf and the member list, every other leaf, and pool-predicate values that keep the family in the pool. It
      applies them one at a time and cumulatively, plus dev-family contexts, and requires dev, diag, tau, tau_dev and
      pool_sha256 (and held, train_fit, assignment) unchanged. Any change is a STOP.
  closure --holdout H --registry R --epoch E --master M --registry-db DB --alias-sources A [--registry-sha256 X]
          [--epoch-sha256 E2] [--master-sha256 M2] [--alias-sources-sha256 A2] [--out D]
      Every input is pinned, by argv or by the registry's epoch_ref and alias_sources_sha256 and the epoch's
      quarantine_check master manifest. An input without a pin is a STOP. For each held, dev and sealed family and each
      diag context the closure lists identities, contexts of every schema (epoch rows, excluded rows, the registry
      contexts table read with mode=ro and query_only, master equivalents, transitively), aliases, history refs and
      seeds. Whatever these inputs cannot enumerate, seed programs included, is closure_unknown. Every closure context
      names its owning registry family. A held, dev or sealed scope that reaches a context of a family with another
      assignment, or a diag scope that reaches a context of a non-training family, is a blocking cross-family link:
      closure.json is still written, with status STOP, and the command STOPs (cross_partition_closure).
  publish --holdout RUN/holdout/holdout.json [--verify-receipt V] [--closure C]
      Needs a PASS `verify --assignment-poison` receipt for these exact bytes, and a PASS closure.json for them whose
      cross-family links publish re-derives from the holdout assignment. GETs the bound commitment by exact version,
      re-derives the seed and recomputes N, k, held, dev, tau and tau_dev from it. Then GETs the registry at the exact
      VersionId of holdout.json's registry ref and needs its sha256 and bytes (P3I-2), so the binding never carries a
      registry ref nobody read. The commitment key must hold exactly
      that version, and the binding key at most one version, neither with a delete marker. Then, each create-only with
      exact-version readback and a single-version history: the manifest at objects/holdout/sha256/<sha>.json and at
      campaigns/<experiment>/manifests/<sha>.json, closure.json at objects/exclusion-closure/sha256/<sha>.json, and the
      deterministic binding bindings/<experiment>/holdout.json. A binding that differs from ours in anything but its
      rule text, a second binding version or a delete marker is CONFLICT: STOP, and the experiment is CONTAMINATED.
freeze, verify and closure make no S3 call. commit and publish do write to S3: commit the commitment, publish the two
manifests, the closure and the binding. Their version-history checks use ListObjectVersions on one exact key
(Prefix = the key, MaxKeys = HISTORY_MAX, filtered to Key == key), the bounded form s3io's lane guard allows (LP-1); a
refused or truncated listing is a STOP before the first write (fail closed). A DRAFT dry run (tools/p2_decisions.py
dryrun) reaches commit and publish only through the non-CLI keyword _main(..., draft_dry_run=True) (review fixes
P3I-1, D-MAP-6): commit then applies families.check_provenance(real=False) without the owner-record and registry-marker
rules, both commands need injected fake clients (any botocore client is refused), commit never writes into RUN/holdout,
and every receipt says draft_dry_run. The real P3 run is gated (PLAN §5 P3 gate, §7.2); nothing in this module
admits it.
"""
import argparse
import ast
import base64
import copy
import csv
import dataclasses
import functools
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from dltrain import common

EXPERIMENT_ID = "gp-policy-dl-v1-e0"
FRAC = 0.05
MIN_POOL = 2
ESTIMAND = "qualified, unsealed, non-seed-exposed families only"
HOLDOUT_SCHEMA = "jope.dl-family-holdout/v1"
COMMITMENT_SCHEMA = "jope.dl-holdout-commitment/v1"
COMMITMENT_REF_SCHEMA = "jope.dl-holdout-commitment-ref/v1"
GATE_SCHEMA = "jope.dl-holdout-gate/v1"
BINDING_SCHEMA = "jope.dl-binding/v1"
CLOSURE_SCHEMA = "jope.dl-exclusion-closure/v1"
STATS_SCHEMA = "jope.dl-holdout-stats/v1"
CONTEXT_STATS_SCHEMA = "jope.dl-context-stats/v1"
RECEIPT_SCHEMA = "jope.dl-holdout-receipt/v1"
REGISTRY_SCHEMA = "jope.dl-family-registry/v1"
DECISIONS_SCHEMA = "jope.dl-family-review-decisions/v1"
OWNER_DECISIONS_SCHEMA = "jope.dl-owner-decisions/v1"
P2_RECEIPT_SCHEMAS = ("jope.dl-packet-receipt/v1", "jope.dl-p2-receipt/v1")
ALIAS_SOURCES_SCHEMA = "jope.dl-family-alias-sources/v1"
EPOCH_SCHEMA = "jope.dl-corpus-epoch/v1"
COMMITMENT_REL = f"bindings/{EXPERIMENT_ID}/holdout-commitment.json"
BINDING_REL = f"bindings/{EXPERIMENT_ID}/holdout.json"
REGISTRY_OBJECT_REL = "objects/family-registry/sha256/{sha}.json"
MANIFEST_REL = "objects/holdout/sha256/{sha}.json"
CAMPAIGN_MANIFEST_REL = "campaigns/" + EXPERIMENT_ID + "/manifests/{sha}.json"
CLOSURE_OBJECT_REL = "objects/exclusion-closure/sha256/{sha}.json"
# The owner decisions P1 and P2 pinned (epoch.DECISIONS_SHA256, family_pin.default_fixed); --owner-decisions overrides.
OWNER_DECISIONS_PATH = ("/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/"
                        "owner-decisions-20261005T044116Z.json")
OWNER_DECISIONS_SHA256 = "77a115c78e927000e77f052c86c2fde5cc66c149d4a16814172b91426aa3a602"
GATE_DECISIONS = ("DL-D2", "DL-D4", "DL-D6", "DL-D12")
GATE_DECISION_VALUES = {"DL-D2": "a", "DL-D4": "a", "DL-D6": "commitment-derived", "DL-D12": "keep_sealed"}
# PLAN §7.2 owner records (review fixes P3I-1, D-MAP-5, D-MAP-6). The stored owner message lives directly in
# RUN/receipts/P2 (families.RECEIPTS_DIR); an owner_directive is the canonical OWNER_DIRECTIVES.md or a file named by an
# owner message stored there. A DRAFT dry run never writes into HOLDOUT_DIR.
RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
P2_RECEIPTS_DIR = RUN / "receipts" / "P2"
HOLDOUT_DIR = RUN / "holdout"
OWNER_DIRECTIVES_PATH = Path("/home/ubuntu/JopeDime/.agent/plans/OWNER_DIRECTIVES.md")
# A registry ref whose exact-version GET returns one of these does not resolve (P3I-2): 400 a malformed VersionId,
# 404 no such version or key, 405 the version is a delete marker.
REF_UNRESOLVED_STATUS = (400, 404, 405)

ASSIGNMENTS = ("heldout", "dev", "train_fit", "train_nonpool", "sealed_excluded", "prohibited_excluded")
POOL_ASSIGNMENTS = frozenset({"heldout", "dev", "train_fit"})
TRAIN_ASSIGNMENTS = frozenset({"train_fit", "train_nonpool"})
CLOSURE_FAMILY_ASSIGNMENTS = ("heldout", "dev", "sealed_excluded")
QUALIFICATIONS = ("auto", "reviewed", "unresolved")
POOL_QUALIFIED = frozenset({"auto", "reviewed"})
REQUIRED_FAMILY_FIELDS = ("family_id", "sealed", "rights_status", "independence_qualified", "seed_selection_exposed",
                          "contexts", "frozen_fold0", "members")
POOL_FIXED_FIELDS = ("family_id", "independence_qualified", "rights_status", "sealed", "seed_selection_exposed")
FREEZE_KEYS = ("schema", "N", "k", "held", "dev", "diag", "tau_hex", "tau_dev_hex", "p_tau", "assignment", "train_fit",
               "pool_sha256")
POISON_SPEC_KEYS = ("dev", "diag", "tau_hex", "tau_dev_hex", "pool_sha256")          # PLAN §5 P3 Validation
POISON_KEYS = POISON_SPEC_KEYS + ("held", "N", "k", "p_tau", "train_fit", "assignment")
SEED_ROW_KIND = "seed_source_mapping"
SEED_ROW_OK = frozenset({"confirm_mapping", "add_members"})
EXPECTED_SEED_ROWS = 10
EPOCH_REF_FIELDS = ("view_id", "cohort", "fold", "dataset_hash", "item_id", "context_schema", "status",
                    "exclusion_reason", "history_ref", "telemetry_ref", "native_source", "membership_ref", "input_ref",
                    "prepared_ref")
MEMBER_FIELDS = ("kind", "dataset_hash", "canonical_sha256", "sample_identity_sha256", "item_id", "cohort",
                 "epoch_status", "registry_state")
STAT_INT_FIELDS = ("programs", "records", "n", "p")
STAT_FLOAT_FIELDS = ("target_scale",)
QUANTILES = (0, 25, 50, 75, 100)
COMPOSITION_GROUPS = ("pool", "heldout", "dev", "train_fit_pool", "train_nonpool", "training", "sealed_excluded",
                      "prohibited_excluded")
# PLAN §5 P3 Expect items that need per-context statistics, keyed by the jope.dl-context-stats/v1 field that feeds them.
EXPECT_STATS_ITEMS = {"programs": "realized_fractions.programs", "records": "realized_fractions.records",
                      "n": "composition.n_quantiles", "p": "composition.p_quantiles",
                      "target_scale": "composition.target_scale_quantiles"}
EXPECT_REGISTRY_ITEMS = ("k_exact", "every_family_exactly_one_assignment", "estimand", "realized_fractions.families",
                         "realized_fractions.datasets", "realized_fractions.folds", "composition.cohort",
                         "composition.platform", "composition.family_size")
SEED_UNKNOWN_REASON = ("per-context seed pools (prepare_launch seed-pool.jsonl: 96 broad seeds plus up to 16 "
                       "dataset-conditioned prior champions) and the SEED192 program_sha256 values sourced from the "
                       "family are not P3 inputs")
REF_KEYS = ("bucket", "key", "version_id", "sha256", "bytes")
HISTORY_MAX = 16
BINDING_DESCRIPTIVE_FIELDS = frozenset({"rule"})
EXIT = {"PASS": 0, "STOP": 2, "ERROR": 3, "CONFLICT": 4}
HEX64 = re.compile(r"[0-9a-f]{64}")
FAMILY_ID_RE = re.compile(r"fam-[0-9a-f]{64}")
RIGHTS_RE = re.compile(r"recorded:\S+")
SEED_FILE_RE = re.compile(r"[0-9a-f]{64}\n")
OPTION_RE = re.compile(r"\s*\(?([A-Za-z])(?:\)|:|\s*$)")
POOL_PREDICATE = ("independence_qualified in {auto, reviewed} and rights_status != prohibited and not sealed and not "
                  "seed_selection_exposed (DL-D4a: no normalizer-exposure clause)")
_ALGORITHM_PROSE = {
    "name": "jope.dl-family-holdout/v1",
    "source": "PLAN §6.2 freeze, prio, derive_seed and assign_appended",
    "prio": "int.from_bytes(sha256(domain || 0x00 || seed || 0x00 || utf8(unit)), 'big')",
    "domains": {"heldout": "jope.dl-family-holdout/v1", "dev": "jope.dl-family-dev/v1",
                "diag": "jope.dl-split-diagnostic/v1"},
    "seed": "sha256(sha256(commitment_bytes) || utf8(commitment_version_id))",
    "heldout": "the first ceil(frac*N) pool families ordered by (prio_heldout, family_id)",
    "dev": "the first ceil(frac*|rest|) remaining pool families ordered by (prio_dev, family_id)",
    "train_fit": "(rest - dev) | unsealed, non-prohibited non-pool families",
    "diag": "the first ceil(frac*|ctx_rest|) train_fit contexts outside frozen fold 0 ordered by "
            "(prio_diag, context_sha256); drawn once",
    "pool": POOL_PREDICATE,
    "appends": "assign_appended: inherit; else prio <= tau is heldout, prio_dev <= tau_dev is dev, else train_fit",
}
SELECTION_CODE_UNITS = ("sha256", "prio", "derive_seed", "freeze", "assign_appended", "pool_predicate", "pool_members",
                        "pool_sha256")
_PACKAGE = "dltrain"


class HoldoutStop(RuntimeError):
    """A P3 STOP condition (PLAN §5 P3 Failure): k mismatch, gate failure, poison difference, inconsistent inputs."""


class HoldoutConflict(HoldoutStop):
    """A create-only commitment or binding key holds something else (CONFLICT; a binding conflict is CONTAMINATED)."""

    def __init__(self, message: str, **details):
        super().__init__(message)
        self.details = details


class RegistryError(HoldoutStop):
    """The family registry is missing, mis-pinned or malformed for what P3 reads."""


class Contaminated(RuntimeError):
    """PLAN §6.2 assign_appended: a cross-partition merge, or a sealed or prohibited member joining another family."""


# --------------------------------------------------------------------------------------- PLAN §6.2 (verbatim semantics)
def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


DOMAIN = b"jope.dl-family-holdout/v1"
DEV_DOMAIN = b"jope.dl-family-dev/v1"
DIAG_DOMAIN = b"jope.dl-split-diagnostic/v1"


def prio(seed: bytes, unit: str, domain: bytes = DOMAIN) -> int:
    return int.from_bytes(hashlib.sha256(domain + b"\0" + seed + b"\0" + unit.encode()).digest(), "big")


def derive_seed(commitment_bytes: bytes, commitment_version_id: str) -> bytes:   # §5 P3 step 1
    return hashlib.sha256(hashlib.sha256(commitment_bytes).digest() + commitment_version_id.encode()).digest()


def freeze(registry: dict, pool: list[str], seed: bytes, frac: float = 0.05) -> dict:
    fams = sorted(set(pool)); N = len(fams); k = math.ceil(frac * N)
    if not 0 < frac < 1 or N < MIN_POOL:
        raise HoldoutStop(f"freeze needs 0 < frac < 1 and a pool of at least {MIN_POOL} families (N={N}, frac={frac})")
    missing = [f for f in fams if f not in registry]
    if missing:
        raise HoldoutStop(f"{len(missing)} pool families are missing from the registry")
    order = sorted(fams, key=lambda f: (prio(seed, f), f))
    held = order[:k]; rest = order[k:]
    tau = prio(seed, held[-1])
    dev_order = sorted(rest, key=lambda f: (prio(seed, f, DEV_DOMAIN), f))
    dev = dev_order[:math.ceil(frac * len(rest))]
    if not dev:
        raise HoldoutStop(f"empty dev split (N={N}, k={k})")
    tau_dev = prio(seed, dev[-1], DEV_DOMAIN)
    pool_set = set(fams)
    trainable_nonpool = {f for f in registry if f not in pool_set and not registry[f]["sealed"]
                         and registry[f]["rights_status"] != "prohibited"}   # DL-D5: unresolved / seed-exposed families train, never held
    train_fit = sorted((set(rest) - set(dev)) | trainable_nonpool)
    ctx_rest = sorted(c for f in train_fit for c in registry[f]["contexts"] if not registry[f]["frozen_fold0"].get(c))
    diag = sorted(ctx_rest, key=lambda c: (prio(seed, c, DIAG_DOMAIN), c))[:math.ceil(frac * len(ctx_rest))]
    held_set, dev_set = set(held), set(dev)
    assignment = {f: ("heldout" if f in held_set else "dev" if f in dev_set else "train_fit") for f in fams}
    for f in registry:                                   # non-pool families
        assignment.setdefault(f, "sealed_excluded" if registry[f]["sealed"]
                              else "prohibited_excluded" if registry[f]["rights_status"] == "prohibited"
                              else "train_nonpool")      # in train_fit: trains, never held or dev
    return {"schema": HOLDOUT_SCHEMA, "N": N, "k": k, "held": held, "dev": dev, "diag": diag,
            "tau_hex": f"{tau:064x}", "tau_dev_hex": f"{tau_dev:064x}", "p_tau": (tau + 1) / 2**256,
            "assignment": assignment, "train_fit": train_fit, "pool_sha256": sha256(common.canonical(fams))}


def assign_appended(member_families: set[str], new_family: str, pool_eligible: bool, sealed: bool, prohibited: bool,
                    seed, tau, tau_dev, assignment):
    if member_families:                    # joins existing families: inherits
        inherited = {assignment[f] for f in member_families}
        if (len(inherited) > 1 or (sealed and inherited != {"sealed_excluded"})
                or (prohibited and inherited != {"prohibited_excluded"})):
            raise Contaminated("CONTAMINATED")   # cross-partition merge, or a sealed/prohibited member joining another family
        return inherited.pop()
    if sealed:
        return "sealed_excluded"
    if prohibited:
        return "prohibited_excluded"
    if not pool_eligible:
        return "train_nonpool"   # pool_eligible = the P3 pool predicate; DL-D5: trains, never held
    if prio(seed, new_family) <= tau:
        return "heldout"
    if prio(seed, new_family, DEV_DOMAIN) <= tau_dev:
        return "dev"
    return "train_fit"


def append_assignments(holdout_doc: dict, seed: bytes, appends: dict) -> dict:
    """Assign post-freeze families with assign_appended. appends = {family_id: {member_families, pool_eligible, sealed,
    prohibited}}. The frozen held, dev and diag lists are never recomputed: the diagnostic was drawn once at P3, so
    appended contexts never join it. A new family whose id already has an assignment is a STOP (stable IDs, §6.2).
    Appended families that join other appended families are resolved after them (dependency order), and each result
    is visible to the later ones. A member family that is neither frozen nor appended, a new family naming itself, or
    a cycle among the appends is a STOP."""
    tau, tau_dev = int(holdout_doc["tau_hex"], 16), int(holdout_doc["tau_dev_hex"], 16)
    frozen = holdout_doc["assignment"]
    members_of = {}
    for fid in sorted(appends):
        members = set(appends[fid].get("member_families") or ())
        if fid in frozen and fid not in members:
            raise HoldoutStop(f"{fid} is already assigned: recomputing an existing family_id is a STOP")
        if fid in members and fid not in frozen:
            raise HoldoutStop(f"{fid} is new but names itself as a member family")
        missing = sorted(m for m in members if m not in frozen and m not in appends)
        if missing:
            raise HoldoutStop(f"{fid}: {len(missing)} member families are neither frozen nor appended")
        members_of[fid] = members
    deps = {fid: {m for m in members_of[fid] if m in appends and m != fid} for fid in members_of}
    order, done, pending = [], set(), sorted(appends)
    while pending:
        ready = [f for f in pending if deps[f] <= done]
        if not ready:
            raise HoldoutStop(f"{len(pending)} appended families join each other in a cycle")
        order += ready
        done.update(ready)
        pending = [f for f in pending if f not in done]
    known = dict(frozen)
    out = {}
    for fid in order:
        a = appends[fid]
        out[fid] = assign_appended(members_of[fid], fid, bool(a["pool_eligible"]), bool(a["sealed"]),
                                   bool(a["prohibited"]), seed, tau, tau_dev, known)
        known[fid] = out[fid]
    return {"assignment": out, "order": order, "held": list(holdout_doc["held"]), "dev": list(holdout_doc["dev"]),
            "diag": list(holdout_doc["diag"])}


# ----------------------------------------------------------------------------------------------- small helpers
def _s3():
    from dltrain import s3io
    return s3io


def _families():
    from dltrain import families     # the P2 provenance rule itself (CL-2), so P3 applies exactly that rule (P3I-1)
    return families


def _resolved(path) -> Path | None:
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _key(rel: str) -> str:
    return _s3().PREFIX_KEY + rel


def _hex64(v) -> bool:
    return isinstance(v, str) and bool(HEX64.fullmatch(v))


def _usable_vid(v) -> bool:
    return isinstance(v, str) and bool(v) and v != "null"


def _checksum_b64(body: bytes) -> str:
    return base64.b64encode(hashlib.sha256(body).digest()).decode()


def _http_status(err) -> int:
    return int((getattr(err, "response", None) or {}).get("ResponseMetadata", {}).get("HTTPStatusCode") or 0)


def _no_duplicate_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate JSON key (sha256 {sha256(str(k).encode())[:16]})")
        out[k] = v
    return out


def _no_constants(name):
    raise ValueError(f"non-finite JSON constant {name}")


def loads_strict(raw: bytes):
    """JSON with duplicate keys and NaN/Infinity refused."""
    return json.loads(raw, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constants)


def _ratio(a, b):
    return a / b if b else None


def _quantiles(values):
    v = sorted(values)
    return {f"p{q}": v[(len(v) - 1) * q // 100] for q in QUANTILES} if v else None


def _safe_identity(identity: str, sealed: bool) -> str:
    """Registry-item identities are hashes; any other identity of a sealed family is replaced by its sha256."""
    if not sealed or identity.startswith("item:"):
        return identity
    return "sha256:" + sha256(identity.encode())


def _safe_key(u, sealed: bool) -> dict:
    u = u if isinstance(u, dict) else {"value": u}
    out = {"kind": u.get("kind")}
    if "value_sha256" in u:
        out["value_sha256"] = u["value_sha256"]
    elif sealed:
        out["value_sha256"] = sha256(str(u.get("value")).encode())
    else:
        out["value"] = u.get("value")
    return out


def _fsync_dir(directory) -> None:
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _create_file(path, data: bytes, mode: int = 0o444) -> dict:
    """Crash-atomic create-only write. The bytes go to a same-directory temp file (write, fsync, chmod), which is then
    hard-linked to the final name and unlinked. The final name therefore never holds torn bytes, so a crash leaves
    either nothing or the complete file. An existing file with equal bytes is success; different bytes raise
    common.ImmutableConflict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        try:
            os.link(tmp, path)
        except FileExistsError:
            existing = path.read_bytes()
            if existing != data:
                raise common.ImmutableConflict(
                    f"{path} exists with different bytes ({sha256(existing)} != {sha256(data)})") from None
            return {"path": str(path), "sha256": sha256(data), "bytes": len(data), "existed": True}
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
    _fsync_dir(path.parent)
    return {"path": str(path), "sha256": sha256(data), "bytes": len(data), "existed": False}


def _write_json(path, doc) -> dict:
    return _create_file(path, common.canonical(doc))


def _load_pinned(path, expected_sha256=None) -> tuple[bytes, dict]:
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        raise HoldoutStop(f"{path}: missing file") from None
    got = sha256(raw)
    if expected_sha256 is not None and got != expected_sha256:
        raise HoldoutStop(f"{path}: sha256 {got} != pinned {expected_sha256}")
    return raw, {"path": str(path), "sha256": got, "bytes": len(raw)}


def _walk_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_dicts(v)


# ------------------------------------------------------------------------------------------- registry adapter
@dataclasses.dataclass(frozen=True)
class Registry:
    path: str
    sha256: str
    bytes: int
    doc: dict
    families: dict


def _family_problems(fid, rec) -> list[str]:
    if not isinstance(rec, dict):
        return [f"{fid}: record is {type(rec).__name__}, not an object"]
    p = [f"{fid}: missing {k}" for k in REQUIRED_FAMILY_FIELDS if k not in rec]
    if p:
        return p
    if not isinstance(fid, str) or not FAMILY_ID_RE.fullmatch(fid):
        p.append(f"{fid}: family_id must be 'fam-' + 64 lowercase hex")
    if rec["family_id"] != fid:
        p.append(f"{fid}: family_id field differs from its key")
    for k in ("sealed", "seed_selection_exposed"):
        if not isinstance(rec[k], bool):
            p.append(f"{fid}: {k} must be a bool, not {type(rec[k]).__name__}")
    rs = rec["rights_status"]
    if not isinstance(rs, str) or not (rs in ("unknown", "prohibited") or RIGHTS_RE.fullmatch(rs)):
        p.append(f"{fid}: rights_status must be recorded:<spdx>, unknown or prohibited")
    iq = rec["independence_qualified"]
    if not isinstance(iq, str) or iq not in QUALIFICATIONS:
        p.append(f"{fid}: independence_qualified must be one of {list(QUALIFICATIONS)}")
    ctx = rec["contexts"]
    ctx_ok = isinstance(ctx, list) and all(isinstance(c, str) and HEX64.fullmatch(c) for c in ctx)
    if not ctx_ok:
        p.append(f"{fid}: contexts must be a list of 64-hex context_sha256 values")
    elif ctx != sorted(set(ctx)):
        p.append(f"{fid}: contexts must be sorted and unique")
    ff = rec["frozen_fold0"]
    if not isinstance(ff, dict) or not all(isinstance(v, bool) for v in ff.values()):
        p.append(f"{fid}: frozen_fold0 must map context_sha256 to bool")
    elif ctx_ok and set(ff) != set(ctx):
        p.append(f"{fid}: frozen_fold0 keys must equal contexts")
    mem = rec["members"]
    if not isinstance(mem, list) or not mem:
        p.append(f"{fid}: members must be a non-empty list")
    else:
        ids = [m.get("identity") if isinstance(m, dict) else None for m in mem]
        if not all(isinstance(i, str) and i for i in ids):
            p.append(f"{fid}: every member needs a non-empty string identity")
        elif len(set(ids)) != len(ids):
            p.append(f"{fid}: duplicate member identity")
    if isinstance(iq, str) and iq in POOL_QUALIFIED and ctx_ok and not ctx:
        p.append(f"{fid}: {iq} family without contexts (P2 qualifies contextless families as unresolved)")
    return p


def validate_registry(doc, require_final: bool = True) -> list[str]:
    """Every problem with the fields P3 reads (PLAN §5 P2 'Per family'; P2 spec per_family_fields)."""
    if not isinstance(doc, dict):
        return ["registry is not a JSON object"]
    p = []
    if doc.get("schema") != REGISTRY_SCHEMA:
        p.append(f"schema {doc.get('schema')!r} != {REGISTRY_SCHEMA}")
    if doc.get("experiment_id") != EXPERIMENT_ID:
        p.append(f"experiment_id {doc.get('experiment_id')!r} != {EXPERIMENT_ID}")
    phase = doc.get("phase")
    if require_final and phase != "final":
        p.append(f"phase {phase!r}: P3 needs the final (post-review) registry")
    elif phase not in ("pre_review", "final"):
        p.append(f"phase {phase!r} is not pre_review or final")
    rds = doc.get("review_decisions_sha256")
    if phase == "final" and not (isinstance(rds, str) and HEX64.fullmatch(rds)):
        p.append("a final registry must carry review_decisions_sha256 (64 hex)")
    fams = doc.get("families")
    if not isinstance(fams, dict) or not fams:
        return p + ["families must be a non-empty object"]
    ctx_owner, id_owner = {}, {}
    for fid in sorted(fams):
        fp = _family_problems(fid, fams[fid])
        p += fp
        if fp:
            continue
        for c in fams[fid]["contexts"]:
            if c in ctx_owner:
                p.append(f"context {c} is in families {ctx_owner[c]} and {fid}")
            ctx_owner.setdefault(c, fid)
        for m in fams[fid]["members"]:
            i = m["identity"]
            if i in id_owner:
                p.append(f"identity {_safe_identity(i, True)} is in families {id_owner[i]} and {fid}")
            id_owner.setdefault(i, fid)
    ids = doc.get("identities")
    if ids is not None:
        if not isinstance(ids, dict):
            p.append("identities must map identity -> family_id")
        elif ids != id_owner:
            bad = sum(1 for i in set(ids) | set(id_owner) if ids.get(i) != id_owner.get(i))
            p.append(f"identities map disagrees with family members for {bad} identities")
    retired = doc.get("retired_family_ids")
    if retired is not None:
        if not isinstance(retired, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in retired.items()):
            p.append("retired_family_ids must map old family_id -> new family_id")
        else:
            if set(retired) & set(fams):
                p.append("a retired family_id is still a family")
            if any(v not in fams and v not in retired for v in retired.values()):
                p.append("a retired family_id maps to an unknown family")
    return p


def load_registry(path, sha256_expected: str, require_final: bool = True) -> Registry:
    """Adapter for the P2 registry (jope.dl-family-registry/v1): pins the bytes by sha256 and validates the fields
    freeze() and the pool predicate read. Raises RegistryError on any missing or ill-typed field."""
    if not isinstance(sha256_expected, str) or not HEX64.fullmatch(sha256_expected):
        raise RegistryError("the registry sha256 pin must be 64 lowercase hex")
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        raise RegistryError(f"registry not found at {path}") from None
    got = sha256(raw)
    if got != sha256_expected:
        raise RegistryError(f"registry sha256 {got} != pinned {sha256_expected}")
    try:
        doc = loads_strict(raw)
    except ValueError as err:
        raise RegistryError(f"registry is not strict JSON: {err}") from None
    problems = validate_registry(doc, require_final)
    if problems:
        raise RegistryError(f"{len(problems)} registry problem(s): " + "; ".join(problems[:12]))
    return Registry(path=str(path), sha256=got, bytes=len(raw), doc=doc, families=doc["families"])


def pool_predicate(fam: dict) -> bool:
    """The P3 selectable pool (PLAN §5 P3; DL-D4a, so no normalizer-exposure clause)."""
    return (fam["independence_qualified"] in POOL_QUALIFIED and fam["rights_status"] != "prohibited"
            and not fam["sealed"] and not fam["seed_selection_exposed"])


def pool_members(families: dict) -> list[str]:
    return sorted(f for f, rec in families.items() if pool_predicate(rec))


def pool_sha256(pool) -> str:
    return sha256(common.canonical(sorted(set(pool))))


# ------------------------------------------------------------------------------------ code and runtime identity
def selection_code() -> dict:
    """The exact source of the selection code and the constants it reads. Its sha256 is ALGORITHM['code_sha256'],
    which the commitment carries, so freeze, verify and publish STOP when this code changed after the commitment."""
    g = globals()
    units = {name: inspect.getsource(g[name]) for name in SELECTION_CODE_UNITS}
    units["common.canonical"] = inspect.getsource(common.canonical)
    constants = {"DOMAIN": DOMAIN.decode(), "DEV_DOMAIN": DEV_DOMAIN.decode(), "DIAG_DOMAIN": DIAG_DOMAIN.decode(),
                 "POOL_QUALIFIED": sorted(POOL_QUALIFIED), "MIN_POOL": MIN_POOL, "FRAC": FRAC,
                 "HOLDOUT_SCHEMA": HOLDOUT_SCHEMA, "POOL_PREDICATE": POOL_PREDICATE}
    return {"units": units, "constants": constants}


def selection_code_sha256() -> str:
    return sha256(common.canonical(selection_code()))


ALGORITHM = _ALGORITHM_PROSE | {"code_units": list(SELECTION_CODE_UNITS) + ["common.canonical"],
                                "code_sha256": selection_code_sha256()}


def _package_imports(tree: ast.AST, package_dir: Path) -> set[str]:
    """The dltrain modules a parsed module imports anywhere in its body (pattern copied from train.py)."""
    found: set[str] = set()
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
def _source_tree() -> tuple[tuple[str, str], ...]:
    package_dir = Path(__file__).resolve().parent
    digests: dict[str, str] = {}
    todo = ["holdout"]
    while todo:
        name = todo.pop()
        if f"{name}.py" in digests:
            continue
        raw = (package_dir / f"{name}.py").read_bytes()
        digests[f"{name}.py"] = sha256(raw)
        todo.extend(sorted(_package_imports(ast.parse(raw, filename=f"{name}.py"), package_dir)))
    digests["__init__.py"] = common.sha256_file(package_dir / "__init__.py")
    return tuple((f"{_PACKAGE}/{name}", digest) for name, digest in sorted(digests.items()))


def source_tree_sha256() -> str:
    """sha256 over 'path NUL sha256 LF' lines of this module's static dltrain import closure (PLAN §5 Format)."""
    return sha256("".join(f"{path}\0{digest}\n" for path, digest in _source_tree()).encode("utf-8"))


@functools.cache
def _runtime() -> tuple[tuple[str, object], ...]:
    doc = {"python": sys.version, "executable_sha256": common.sha256_file(os.path.realpath(sys.executable))}
    for dist in ("boto3", "botocore"):
        try:
            doc[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            doc[dist] = None
    return tuple(sorted(doc.items()))


def runtime_identity() -> dict:
    return dict(_runtime())


def runtime_sha256() -> str:
    return sha256(common.canonical(runtime_identity()))


# ----------------------------------------------------------------------------------------------- review gate
def _effective_decision(row: dict, group: dict, classes: list) -> str | None:
    g = group.get(row.get("review_group_id"))
    if g is not None:
        return g
    want = f"recommendation=={row.get('recommendation', '')}"
    matches = {c["decision"] for c in classes if c.get("kind") == row.get("kind")
               and isinstance(c.get("when"), str) and re.sub(r"\s+", "", c["when"]) == want}
    if len(matches) > 1:
        return "ambiguous"
    return matches.pop() if matches else None


def _record_location_problems(prov: dict, record, receipts_dir) -> list[str]:
    """Where P3 accepts the pinned owner record that families.check_provenance(real=True) resolved (P3I-1, D-MAP-5): a
    message record directly in RUN/receipts/P2 (the stored owner message, or an explicit record there; never a staging,
    simulation or other directory); an owner_directive at the canonical OWNER_DIRECTIVES.md path (resolved paths, not
    the basename families compares), or a file named by an owner message stored directly in RUN/receipts/P2."""
    rdir = _resolved(receipts_dir)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return ["the decision provenance resolves to no pinned owner record"]
    path = _resolved(record["path"])
    if record.get("form") != "owner_directive":
        if path is None or rdir is None or path.parent != rdir:
            return ["the pinned owner record is not directly in RUN/receipts/P2 (staging, simulation and other "
                    "directories never hold an owner record)"]
        return []
    if path is not None and path == _resolved(OWNER_DIRECTIVES_PATH):
        return []
    named = prov.get("named_by_owner") if isinstance(prov, dict) else None
    if not isinstance(named, dict):
        return ["an owner_directive must be the canonical OWNER_DIRECTIVES.md (resolved path) or a file named by an "
                "owner message stored directly in RUN/receipts/P2"]
    if named.get("record") is not None:
        msg = _resolved((named["record"] if isinstance(named["record"], dict) else {}).get("path"))
    else:
        name = _families().OWNER_MESSAGE_NAME.format(sha16=str(named.get("message_sha256"))[:16])
        msg = _resolved(Path(receipts_dir) / name)
    if msg is None or rdir is None or msg.parent != rdir:
        return ["the owner message that names the directive file is not stored directly in RUN/receipts/P2"]
    return []


def _registry_provenance_problems(reg: Registry, record) -> list[str]:
    """The final registry's own CL-2 marker (families writes decision_deltas.provenance = {draft, simulation, record}):
    draft and simulation must be false and record a pinned record, the one the decision file resolves to (P3I-1)."""
    dd = reg.doc.get("decision_deltas")
    rp = dd.get("provenance") if isinstance(dd, dict) else None
    if not isinstance(rp, dict):
        return ["the final registry carries no decision_deltas.provenance (the P2 CL-2 marker)"]
    p = []
    if rp.get("draft") is not False or rp.get("simulation") is not False:
        p.append("the final registry was built from DRAFT or SIMULATION decisions (decision_deltas.provenance)")
    rec = rp.get("record")
    if not (isinstance(rec, dict) and _hex64(rec.get("sha256")) and isinstance(rec.get("path"), str)):
        p.append("the final registry's decision_deltas.provenance.record is not a pinned record with a 64-hex sha256")
    elif isinstance(record, dict) and not (rec.get("sha256") == record.get("sha256")
                                           and rec.get("form") == record.get("form")
                                           and _resolved(rec["path"]) == _resolved(record.get("path"))):
        p.append("the final registry's decision_deltas.provenance.record is not the owner record this decision file "
                 "resolves to")
    return p


def review_gate(reg: Registry, decisions_bytes: bytes, review_csv_bytes: bytes, *, draft_dry_run: bool = False) -> dict:
    """P3 gate on the P2 owner review (P2 spec 'What P3 needs'): the decision file is the one the final registry binds,
    binds this review.csv, has no reject_mapping, and every one of the 10 seed rows resolves to confirm_mapping or
    add_members.

    Provenance (review fix P3I-1): the P2 rule itself, families.check_provenance(real=True, receipts_dir=
    RUN/receipts/P2): a known kind with its fields, non-degenerate 64-hex hashes, no DRAFT or SIMULATION marker, an
    ISO-8601 UTC message time, a message_text (when given) that hashes to message_sha256, and a pinned owner record
    (the stored owner message dl-d5-owner-message-<sha16>.txt hashing to message_sha256, an explicit record, or the
    owner_directive file). P3 adds where that record may live (_record_location_problems) and needs the final
    registry's decision_deltas.provenance to read draft false and simulation false with that same record.
    draft_dry_run (non-CLI, the DRAFT dry run with fake clients only) applies families.check_provenance(real=False)
    and skips the owner-record and registry-marker rules."""
    try:
        dec = loads_strict(decisions_bytes)
    except ValueError as err:
        raise HoldoutStop(f"review decisions are not strict JSON: {err}") from None
    if not isinstance(dec, dict):
        raise HoldoutStop("review decisions must be a JSON object")
    p = []
    d_sha, c_sha = sha256(decisions_bytes), sha256(review_csv_bytes)
    if dec.get("schema") != DECISIONS_SCHEMA:
        p.append(f"decisions schema {dec.get('schema')!r} != {DECISIONS_SCHEMA}")
    if dec.get("reviewer") != "owner":
        p.append("decisions reviewer must be 'owner'")
    fam, prov, info = _families(), dec.get("provenance"), None
    try:
        info = fam.check_provenance(prov, real=not draft_dry_run, receipts_dir=P2_RECEIPTS_DIR)
    except fam.FamilyStop as err:
        p.append(f"decisions provenance (P2 rule, families.check_provenance): {err}")
    if not draft_dry_run:
        if info is not None:
            p += _record_location_problems(prov, info.get("record"), P2_RECEIPTS_DIR)
        p += _registry_provenance_problems(reg, None if info is None else info.get("record"))
    if reg.doc.get("review_decisions_sha256") != d_sha:
        p.append("the registry's review_decisions_sha256 is not this decision file")
    binds = dec.get("binds") if isinstance(dec.get("binds"), dict) else {}
    if binds.get("review_csv_sha256") != c_sha:
        p.append("decisions binds.review_csv_sha256 is not this review.csv")
    if reg.doc.get("review_csv_sha256") is not None and reg.doc.get("review_csv_sha256") != c_sha:
        p.append("the registry's review_csv_sha256 is not this review.csv")
    if reg.doc.get("alias_sources_sha256") is not None and binds.get("alias_sources_sha256") != reg.doc["alias_sources_sha256"]:
        p.append("decisions binds.alias_sources_sha256 differs from the registry's")
    group, classes = {}, []
    for d in dec.get("decisions") or []:
        if not isinstance(d, dict) or not isinstance(d.get("review_group_id"), str) or not isinstance(d.get("decision"), str):
            p.append("every group decision needs review_group_id and decision strings")
            continue
        if d["review_group_id"] in group:
            p.append(f"two decisions for {d['review_group_id']}")
        group[d["review_group_id"]] = d["decision"]
    for c in dec.get("class_decisions") or []:
        if not isinstance(c, dict) or not isinstance(c.get("decision"), str):
            p.append("every class decision needs a decision string")
            continue
        classes.append(c)
    if "reject_mapping" in set(group.values()) | {c["decision"] for c in classes}:
        p.append("a reject_mapping decision makes P3 STOP")
    try:
        rows = list(csv.DictReader(io.StringIO(review_csv_bytes.decode("utf-8-sig"))))
    except (UnicodeDecodeError, csv.Error) as err:
        raise HoldoutStop(f"review.csv is unreadable: {err}") from None
    if rows and not {"review_group_id", "kind", "recommendation"} <= set(rows[0]):
        p.append("review.csv lacks review_group_id, kind or recommendation columns")
    seed_rows = [r for r in rows if r.get("kind") == SEED_ROW_KIND]
    if len(seed_rows) != EXPECTED_SEED_ROWS:
        p.append(f"{len(seed_rows)} seed_source_mapping rows != {EXPECTED_SEED_ROWS}")
    for r in seed_rows:
        eff = _effective_decision(r, group, classes)
        if eff not in SEED_ROW_OK:
            p.append(f"seed row {r.get('review_group_id')}: decision {eff!r} is not confirm_mapping or add_members")
    if p:
        raise HoldoutStop(f"review gate: {len(p)} problem(s): " + "; ".join(p[:10]))
    rule = ("families.check_provenance(real=False): DRAFT dry run, fake S3 only" if draft_dry_run else
            "families.check_provenance(real=True, receipts_dir=RUN/receipts/P2), the record directly in "
            "RUN/receipts/P2 (or the canonical OWNER_DIRECTIVES.md), registry decision_deltas.provenance not draft")
    return {"review_decisions_sha256": d_sha, "review_csv_sha256": c_sha, "review_rows": len(rows),
            "seed_rows": len(seed_rows), "group_decisions": len(group), "class_decisions": len(classes),
            "provenance": {"kind": prov.get("kind"), "draft": info["draft"], "record": info["record"], "rule": rule,
                           "draft_dry_run": bool(draft_dry_run)}}


# -------------------------------------------------------------------------------------------------- P3 gate
def _option(text: str) -> str | None:
    m = OPTION_RE.match(text)
    return m.group(1).lower() if m else None


def _gate_decision_problems(decisions: dict) -> list[str]:
    p = [f"owner decisions do not record {d}" for d in GATE_DECISIONS
         if not isinstance(decisions.get(d), str) or not decisions[d].strip()]
    if p:
        return p
    if _option(decisions["DL-D2"]) != "a":
        p.append("DL-D2 is not option a (freeze() trains unknown-licence families; prohibited ones never train)")
    if _option(decisions["DL-D4"]) != "a":
        p.append("DL-D4 is not option a (the pool predicate has no DL-D4b normalizer-exposure clause)")
    if "commitment-derived" not in decisions["DL-D6"].lower():
        p.append("DL-D6 is not the commitment-derived seed (the only seed this module derives)")
    if not re.match(r"(?i)\s*keep\b", decisions["DL-D12"]):
        p.append("DL-D12 is not 'keep sealed' (freeze() excludes sealed families)")
    return p


def check_gate(reg: Registry, p2_receipt_path, owner_decisions_path, owner_decisions_sha256,
               p2_receipt_sha256=None) -> dict:
    """PLAN §5 P3 Gate, enforced before the irreversible commitment PUT (mirrors family_pin.check_gate).

    The P2 packet receipt must have status (or verdict) PASS for packet P2 and carry an S3 ref
    {bucket, key, version_id, sha256, bytes} for exactly this registry at objects/family-registry/sha256/<sha>.json.
    The owner decisions file must be pinned by sha256, have provenance (PLAN §7.2), and record DL-D2 = a, DL-D4 = a,
    DL-D6 = commitment-derived and DL-D12 = keep sealed, the decisions this module's code assumes. Returns the
    gate record that commit writes to gate.json and freeze puts into holdout.json."""
    p = []
    rp = Path(p2_receipt_path).resolve()
    raw, p2_meta = _load_pinned(rp, p2_receipt_sha256)
    try:
        rec = loads_strict(raw)
    except ValueError as err:
        raise HoldoutStop(f"P3 gate: the P2 receipt is not strict JSON: {err}") from None
    if not isinstance(rec, dict):
        raise HoldoutStop("P3 gate: the P2 receipt is not a JSON object")
    if rec.get("schema") not in P2_RECEIPT_SCHEMAS:
        p.append(f"P2 receipt schema {rec.get('schema')!r} is not one of {list(P2_RECEIPT_SCHEMAS)}")
    if rec.get("packet") != "P2":
        p.append("the P2 receipt does not name packet P2")
    verdicts = [rec[k] for k in ("status", "verdict") if k in rec]
    if not verdicts or any(v != "PASS" for v in verdicts):
        p.append("the P2 receipt is not PASS")
    want_key = _key(REGISTRY_OBJECT_REL.format(sha=reg.sha256))
    refs = [d for d in _walk_dicts(rec) if d.get("key") == want_key]
    good = [d for d in refs if d.get("bucket") == _s3().BUCKET and d.get("sha256") == reg.sha256
            and d.get("bytes") == reg.bytes and _usable_vid(d.get("version_id"))]
    if not refs:
        p.append("the P2 receipt carries no S3 ref for this registry (objects/family-registry/sha256/<sha>.json)")
    elif len(good) != len(refs) or len({d["version_id"] for d in good}) != 1:
        p.append("the P2 receipt's S3 refs for this registry are incomplete or disagree")
    if not _hex64(owner_decisions_sha256):
        raise HoldoutStop("P3 gate: the owner decisions need a 64-hex sha256 pin")
    op = Path(owner_decisions_path).resolve()
    o_raw, o_meta = _load_pinned(op, owner_decisions_sha256)
    try:
        od = loads_strict(o_raw)
    except ValueError as err:
        raise HoldoutStop(f"P3 gate: the owner decisions are not strict JSON: {err}") from None
    if not isinstance(od, dict) or od.get("schema") != OWNER_DECISIONS_SCHEMA:
        p.append(f"owner decisions are not {OWNER_DECISIONS_SCHEMA}")
        od = od if isinstance(od, dict) else {}
    if not isinstance(od.get("provenance"), dict) or not od["provenance"]:
        p.append("owner decisions need a provenance block (PLAN §7.2 human-authored citation)")
    decisions = od.get("decisions") if isinstance(od.get("decisions"), dict) else {}
    p += _gate_decision_problems(decisions)
    if p:
        raise HoldoutStop(f"P3 gate: {len(p)} problem(s): " + "; ".join(p))
    return {"schema": GATE_SCHEMA, "experiment_id": EXPERIMENT_ID, "registry_sha256": reg.sha256,
            "registry_s3": {k: good[0][k] for k in REF_KEYS},
            "p2_receipt": p2_meta | {"packet": "P2", "verdict": "PASS"},
            "owner_decisions": o_meta | {"schema": OWNER_DECISIONS_SCHEMA},
            "decisions": dict(GATE_DECISION_VALUES)}


def load_gate(holdout_dir, reg: Registry) -> dict:
    """gate.json as commit wrote it, re-checked against the P2 receipt and owner decisions it names."""
    p = Path(holdout_dir) / "gate.json"
    try:
        raw = p.read_bytes()
    except FileNotFoundError:
        raise HoldoutStop(f"{p} is missing: commit records the P3 gate there") from None
    try:
        g = loads_strict(raw)
    except ValueError as err:
        raise HoldoutStop(f"gate.json is not strict JSON: {err}") from None
    if not isinstance(g, dict) or g.get("schema") != GATE_SCHEMA or common.canonical(g) != raw:
        raise HoldoutStop(f"gate.json is not a canonical {GATE_SCHEMA} document")
    if g.get("registry_sha256") != reg.sha256:
        raise HoldoutStop("gate.json was recorded for another registry")
    try:
        again = check_gate(reg, g["p2_receipt"]["path"], g["owner_decisions"]["path"], g["owner_decisions"]["sha256"],
                           g["p2_receipt"]["sha256"])
    except (KeyError, TypeError):
        raise HoldoutStop("gate.json lacks the P2 receipt or owner-decisions refs") from None
    if again != g:
        raise HoldoutStop("gate.json no longer matches the P2 receipt and owner decisions it names")
    return g


# ------------------------------------------------------------------------------------------------ commitment
def commitment_doc(registry_sha256: str, review_decisions_sha256: str, pool) -> dict:
    return {"schema": COMMITMENT_SCHEMA, "experiment_id": EXPERIMENT_ID, "pool_sha256": pool_sha256(pool),
            "registry_sha256": registry_sha256, "review_decisions_sha256": review_decisions_sha256,
            "algorithm": ALGORITHM, "frac": FRAC}


def commitment_bytes(registry_sha256: str, review_decisions_sha256: str, pool) -> bytes:
    """Deterministic: no clock or host field, so a crash-rerun reproduces the same bytes (and hence the same seed)."""
    return common.canonical(commitment_doc(registry_sha256, review_decisions_sha256, pool))


def read_seed(path) -> bytes:
    text = Path(path).read_bytes().decode("ascii", "replace")
    if not SEED_FILE_RE.fullmatch(text):
        raise HoldoutStop(f"{path} must hold 64 lowercase hex digits and one newline")
    return bytes.fromhex(text[:64])


def load_commitment(holdout_dir) -> dict:
    d = Path(holdout_dir)
    body = (d / "commitment.json").read_bytes()
    ref = loads_strict((d / "commitment-ref.json").read_bytes())
    doc = loads_strict(body)
    s3 = _s3()
    p = []
    if common.canonical(doc) != body:
        p.append("commitment.json is not canonical JSON")
    if not isinstance(ref, dict) or ref.get("schema") != COMMITMENT_REF_SCHEMA:
        raise HoldoutStop(f"commitment-ref.json is not {COMMITMENT_REF_SCHEMA}")
    if ref.get("bucket") != s3.BUCKET or ref.get("key") != _key(COMMITMENT_REL):
        p.append("commitment-ref names another bucket or key")
    if ref.get("sha256") != sha256(body) or ref.get("bytes") != len(body):
        p.append("commitment-ref sha256/bytes differ from commitment.json")
    vid = ref.get("version_id")
    if not isinstance(vid, str) or not vid or vid == "null":
        p.append("commitment-ref has no usable VersionId")
    if p:
        raise HoldoutStop("commitment files: " + "; ".join(p))
    return {"bytes": body, "doc": doc, "ref": ref}


def check_commitment(com: dict, reg: Registry, seed: bytes, pool) -> None:
    d = com["doc"]
    p = []
    expected = {"schema": COMMITMENT_SCHEMA, "experiment_id": EXPERIMENT_ID, "registry_sha256": reg.sha256,
                "review_decisions_sha256": reg.doc.get("review_decisions_sha256"), "pool_sha256": pool_sha256(pool),
                "algorithm": ALGORITHM, "frac": FRAC}
    for k, v in expected.items():
        if d.get(k) != v:
            p.append(f"commitment {k} does not match")
    alg = d.get("algorithm")
    if isinstance(alg, dict) and alg.get("code_sha256") != ALGORITHM["code_sha256"]:
        p.append("the selection code changed after the commitment (algorithm.code_sha256)")
    if derive_seed(com["bytes"], com["ref"]["version_id"]) != seed:
        p.append("seed.hex does not derive from the commitment bytes and VersionId")
    if p:
        raise HoldoutStop("commitment check: " + "; ".join(p))


# --------------------------------------------------------------------------------------- S3 create-only + history
def _ambiguous(err) -> bool:
    """PLAN §6.6 publish step 1: a 412, 409, timeout, reset or 5xx is resolved by re-reading the key."""
    status = _http_status(err)
    if status in (409, 412) or status >= 500:
        return True
    try:
        from botocore import exceptions as bx
    except ImportError:  # pragma: no cover (botocore is in VENV)
        return False
    return isinstance(err, (bx.ConnectionError, bx.HTTPClientError))


def _get_current(reader, key: str) -> dict | None:
    try:
        cur = reader.get_object(Bucket=_s3().BUCKET, Key=key)
    except Exception as err:  # noqa: BLE001 (only 'absent' is resolved here)
        if _http_status(err) == 404:
            return None
        raise
    return {"version_id": cur.get("VersionId"), "body": cur["Body"].read()}


def _create_only(writer, reader, key: str, body: bytes) -> dict:
    """PLAN §6.6 publish step 1. PutObject(IfNoneMatch='*', SHA256 checksum); a 200 is read back by the returned
    VersionId. A 412, 409, 5xx or transport failure is resolved by re-reading the key. Equal or different bytes are
    returned for the caller (equal is success; different is CONFLICT or POISONED_KEY). An absent key gets exactly one
    more create-only PUT, and a second ambiguous outcome with the key still absent is a STOP. Other errors propagate."""
    s3 = _s3()
    attempts = []
    for attempt in (1, 2):
        try:
            resp = writer.put_object(Bucket=s3.BUCKET, Key=key, Body=body, IfNoneMatch="*", ContentType="application/json",
                                     ChecksumAlgorithm="SHA256", ChecksumSHA256=_checksum_b64(body))
        except Exception as err:  # noqa: BLE001
            if not _ambiguous(err):
                raise
            attempts.append({"attempt": attempt, "status": _http_status(err) or None, "error": type(err).__name__})
            cur = _get_current(reader, key)
            if cur is None:
                if attempt == 1:
                    continue
                raise HoldoutStop(f"create-only PUT of {key} is unresolved: the key is still absent after a second "
                                  "ambiguous attempt") from None
            existing = cur["body"]
            return {"key": key, "outcome": "exists", "equal": existing == body, "existing": existing,
                    "version_id": cur["version_id"], "sha256": sha256(existing), "bytes": len(existing),
                    "attempts": attempts}
        vid = resp.get("VersionId")
        if not _usable_vid(vid):
            raise HoldoutStop(f"PutObject returned no usable VersionId for {key}")
        rb = s3.verify_readback(reader, key, vid, body)
        return {"key": key, "outcome": "created", "equal": True, "existing": None, "version_id": vid,
                "sha256": rb["sha256"], "bytes": rb["bytes"], "attempts": attempts}
    raise AssertionError("unreachable")


def key_history(reader, key: str) -> dict:
    """Every version and delete marker of exactly this key: ListObjectVersions with Prefix=key, MaxKeys=HISTORY_MAX,
    filtered to the exact key; a truncated listing is a STOP. s3io's lane guard allows exactly this form (LP-1: Bucket,
    Prefix = one full key inside PREFIX, 1 <= MaxKeys <= s3io.HISTORY_MAX), so commit and publish do reach their S3
    writes. A guard that refuses it is still a STOP here (fail closed): P3 cannot show that a commitment or binding
    key holds a single version without it."""
    s3 = _s3()
    try:
        resp = reader.list_object_versions(Bucket=s3.BUCKET, Prefix=key, MaxKeys=HISTORY_MAX)
    except s3.S3Refused:
        raise HoldoutStop("version history unavailable: the s3io lane guard refuses ListObjectVersions for this call, "
                          f"so P3 cannot show that {key} holds a single version (fail closed: no commitment or binding "
                          "is written without an exact-key, bounded ListObjectVersions)") from None
    if resp.get("IsTruncated"):
        raise HoldoutStop(f"the version history of {key} exceeds {HISTORY_MAX} entries")
    versions = [{"version_id": v.get("VersionId"), "is_latest": bool(v.get("IsLatest"))}
                for v in resp.get("Versions") or [] if v.get("Key") == key]
    markers = [{"version_id": m.get("VersionId"), "is_latest": bool(m.get("IsLatest"))}
               for m in resp.get("DeleteMarkers") or [] if m.get("Key") == key]
    if not all(_usable_vid(v["version_id"]) for v in versions + markers):
        raise HoldoutStop(f"the version history of {key} has an unusable VersionId")
    return {"key": key, "versions": versions, "delete_markers": markers}


def _single_version(hist: dict, version_id: str) -> bool:
    return (not hist["delete_markers"] and len(hist["versions"]) == 1
            and hist["versions"][0]["version_id"] == version_id and hist["versions"][0]["is_latest"])


def _at_most_one_version(hist: dict) -> bool:
    return not hist["delete_markers"] and len(hist["versions"]) <= 1


def check_registry_object(reader, ref, registry_sha256: str, registry_bytes) -> dict:
    """Review fix P3I-2: GET the registry object at exactly the VersionId its S3 ref records (the P2 receipt's ref,
    which gate.json and holdout.json carry) and require this registry's sha256 and byte count and the echoed
    VersionId. commit runs it before the commitment key's history and the irreversible commitment PUT, publish before
    its first PUT, so neither commits to nor binds a registry ref nobody read (a DRAFT dry run's fake VersionId does
    not resolve in S3). A version that does not resolve (REF_UNRESOLVED_STATUS) or holds other bytes is a STOP; other
    errors propagate (ERROR)."""
    s3 = _s3()
    key = _key(REGISTRY_OBJECT_REL.format(sha=registry_sha256))
    if not (isinstance(ref, dict) and ref.get("bucket") == s3.BUCKET and ref.get("key") == key
            and ref.get("sha256") == registry_sha256 and ref.get("bytes") == registry_bytes
            and _usable_vid(ref.get("version_id"))):
        raise HoldoutStop("the registry S3 ref is incomplete or does not name this registry's object")
    try:
        body, meta = s3.get_exact(reader, key, ref["version_id"])
    except Exception as err:  # noqa: BLE001 (only a version that does not resolve is a STOP; the rest propagates)
        status = _http_status(err)
        if status in REF_UNRESOLVED_STATUS:
            raise HoldoutStop(f"the registry ref does not resolve in S3 (HTTP {status} for its exact VersionId): no S3 "
                              "write happens for a registry version nobody can read") from None
        raise
    if (meta.get("VersionId"), sha256(body), len(body)) != (ref["version_id"], registry_sha256, registry_bytes):
        raise HoldoutStop(f"the registry object at the ref's exact VersionId is not this registry (sha256 "
                          f"{sha256(body)}, {len(body)} bytes, VersionId echoed "
                          f"{'as requested' if meta.get('VersionId') == ref['version_id'] else 'differently'})")
    return {k: ref[k] for k in REF_KEYS} | {"readback": "GetObject at the exact VersionId", "etag": meta.get("ETag")}


def _draft_dry_run_guard(clients, out_dir=None) -> None:
    """Review fixes P3I-1/D-MAP-6: the DRAFT dry run (tools/p2_decisions.py dryrun) reaches commit and publish only
    through the non-CLI keyword draft_dry_run. It needs injected fake clients: any botocore client, real or stubbed, is
    refused, so a dry run never reaches S3. commit never writes its outputs into the real holdout directory."""
    try:
        from botocore.client import BaseClient
    except ImportError:  # pragma: no cover (botocore is in VENV)
        BaseClient = ()
    if (not isinstance(clients, dict) or not clients
            or any(c is None or isinstance(c, BaseClient) for c in clients.values())):
        raise HoldoutStop("a DRAFT dry run needs injected fake S3 clients: a botocore client (real or stubbed) is "
                          "refused")
    if out_dir is not None:
        out, real = _resolved(out_dir), _resolved(HOLDOUT_DIR)
        if out is None or real is None or out == real or real in out.parents:
            raise HoldoutStop("a DRAFT dry run never writes into the real holdout directory (RUN/holdout)")


def commit(registry_path, registry_sha256: str, decisions_path, out_dir, clients: dict, review_csv_path=None,
           p2_receipt=None, owner_decisions=OWNER_DECISIONS_PATH, owner_decisions_sha256=OWNER_DECISIONS_SHA256,
           p2_receipt_sha256=None, *, draft_dry_run: bool = False) -> dict:
    if draft_dry_run:
        _draft_dry_run_guard(clients, out_dir)
    reg = load_registry(registry_path, registry_sha256)
    dec_raw = Path(decisions_path).read_bytes()
    csv_path = Path(review_csv_path) if review_csv_path else Path(registry_path).with_name("review.csv")
    if not csv_path.is_file():
        raise HoldoutStop(f"review.csv not found at {csv_path}: the seed-row gate needs it")
    review = review_gate(reg, dec_raw, csv_path.read_bytes(), draft_dry_run=draft_dry_run)
    out = Path(out_dir)
    p2_path = Path(p2_receipt) if p2_receipt else out.resolve().parent / "receipts" / "P2" / "receipt.json"
    gate = check_gate(reg, p2_path, owner_decisions, owner_decisions_sha256, p2_receipt_sha256)
    pool = pool_members(reg.families)
    if len(pool) < MIN_POOL:
        raise HoldoutStop(f"pool has {len(pool)} families; freeze needs at least {MIN_POOL}")
    body = commitment_bytes(reg.sha256, sha256(dec_raw), pool)
    gate_body = common.canonical(gate)
    for name, want in (("commitment.json", body), ("gate.json", gate_body)):
        local = out / name
        if local.exists() and local.read_bytes() != want:
            raise HoldoutStop(f"{local} holds different bytes; refusing before any S3 call")
    s3 = _s3()
    key = _key(COMMITMENT_REL)
    w, r = clients["writer"], clients["reader"]
    registry_object = check_registry_object(r, gate["registry_s3"], reg.sha256, reg.bytes)     # P3I-2: first S3 call
    before = key_history(r, key)
    if not _at_most_one_version(before):
        raise HoldoutConflict("CONFLICT: the commitment key has been rewritten or deleted (more than one version or a "
                              "delete marker), so its first version cannot be identified; STOP for owner review",
                              stop_reason="commitment_history", commitment_history=before)
    put = _create_only(w, r, key, body)
    if not put["equal"]:
        raise HoldoutConflict("CONFLICT: holdout-commitment.json already holds a different commitment; STOP",
                              stop_reason="commitment_conflict",
                              existing_commitment={k: put[k] for k in ("key", "version_id", "sha256", "bytes")})
    vid = put["version_id"]
    if not _usable_vid(vid):
        raise HoldoutStop("the commitment has no usable VersionId")
    after = key_history(r, key)
    if not _single_version(after, vid):
        raise HoldoutConflict("CONFLICT: the commitment key does not hold exactly the adopted version; the seed would "
                              "be ambiguous; STOP for owner review", stop_reason="commitment_history",
                              commitment_history=after, adopted_version_id=vid)
    seed = derive_seed(body, vid)
    w_seed = _create_file(out / "seed.hex", (seed.hex() + "\n").encode())   # before anything else
    w_com = _create_file(out / "commitment.json", body)
    ref = {"schema": COMMITMENT_REF_SCHEMA, "bucket": s3.BUCKET, "key": key, "version_id": vid, "sha256": sha256(body),
           "bytes": len(body)}
    w_ref = _write_json(out / "commitment-ref.json", ref)
    w_gate = _create_file(out / "gate.json", gate_body)
    return {"status": "PASS", "commitment_ref": ref,
            "s3_outcome": "created" if put["outcome"] == "created" else "already_present_equal_bytes",
            "s3_attempts": put["attempts"], "commitment_history": {"before": before, "after": after},
            "seed_sha256": sha256(seed), "pool_N": len(pool), "k_expected": math.ceil(FRAC * len(pool)),
            "review_gate": review, "p3_gate": gate,
            "registry": {"path": reg.path, "sha256": reg.sha256, "bytes": reg.bytes, "s3": gate["registry_s3"]},
            "registry_object": registry_object, "draft_dry_run": bool(draft_dry_run),
            "outputs": {"seed": w_seed, "commitment": w_com, "commitment_ref": w_ref, "gate": w_gate}}


# ------------------------------------------------------------------------------------------- freeze checks/doc
def freeze_checks(families: dict, pool, res: dict, seed: bytes, frac: float = FRAC) -> dict:
    """PLAN §5 P3 Expect and §6.2 rules, checked against the result (not by trusting freeze()). held, dev, train_fit
    and diag are recomputed here from the registry and the seed, independently of the result."""
    fams = sorted(set(pool))
    pool_set = set(fams)
    N = len(fams)
    k = math.ceil(frac * N)
    held, dev, diag, a, tf = res["held"], res["dev"], res["diag"], res["assignment"], res["train_fit"]
    hs, ds, tfs = set(held), set(dev), set(tf)
    expected = {}
    for f, rec in families.items():
        expected[f] = ("heldout" if f in hs else "dev" if f in ds else "train_fit" if f in pool_set
                       else "sealed_excluded" if rec["sealed"]
                       else "prohibited_excluded" if rec["rights_status"] == "prohibited" else "train_nonpool")
    ctx_owner = {c: f for f, rec in families.items() for c in rec["contexts"]}
    ctx_rest = [c for f in tf if f in families for c in families[f]["contexts"] if not families[f]["frozen_fold0"].get(c)]
    held_order = sorted(fams, key=lambda f: (prio(seed, f), f))[:k]
    dev_order = sorted(pool_set - set(held_order), key=lambda f: (prio(seed, f, DEV_DOMAIN), f))
    n_dev = math.ceil(frac * (N - k))
    trainable_nonpool = {f for f, rec in families.items()
                         if f not in pool_set and not rec["sealed"] and rec["rights_status"] != "prohibited"}
    tf_expected = sorted((pool_set - set(held_order) - set(dev_order[:n_dev])) | trainable_nonpool)
    ctx_rest_expected = [c for f in tf_expected for c in families[f]["contexts"] if not families[f]["frozen_fold0"].get(c)]
    diag_expected = sorted(ctx_rest_expected, key=lambda c: (prio(seed, c, DIAG_DOMAIN), c))[
        :math.ceil(frac * len(ctx_rest_expected))]
    return {
        "k_exact": res["N"] == N and res["k"] == k == math.ceil(Fraction(str(frac)) * N) and len(hs) == len(held) == k,
        "heldout_is_prio_prefix": held == held_order,
        "dev_exact": len(ds) == len(dev) == n_dev and dev == dev_order[:len(dev)],
        "tau": bool(held) and res["tau_hex"] == f"{prio(seed, held[-1]):064x}"
               and res["p_tau"] == (int(res["tau_hex"], 16) + 1) / 2**256,
        "tau_dev": bool(dev) and res["tau_dev_hex"] == f"{prio(seed, dev[-1], DEV_DOMAIN):064x}",
        "every_family_exactly_one_assignment": set(a) == set(families) and len(a) == len(families)
                                               and all(v in ASSIGNMENTS for v in a.values()),
        "assignment_matches_partition": a == expected,
        "train_fit_excludes_heldout_and_dev": not tfs & (hs | ds),
        "train_fit_is_train_assignments": tf == sorted(tfs) and tfs == {f for f, v in a.items() if v in TRAIN_ASSIGNMENTS},
        "train_fit_includes_train_nonpool": {f for f, v in a.items() if v == "train_nonpool"} <= tfs,
        "train_fit_exact": tf == tf_expected,
        "sealed_and_prohibited_never_train": not any(families[f]["sealed"] or families[f]["rights_status"] == "prohibited"
                                                     for f in tfs if f in families),
        "diag_size_exact": len(diag) == len(set(diag)) == math.ceil(frac * len(ctx_rest)),
        "diag_is_prio_prefix": diag == diag_expected,
        "diag_within_train_fit": all(ctx_owner.get(c) in tfs for c in diag),
        "diag_excludes_frozen_fold0": not any(families[ctx_owner[c]]["frozen_fold0"].get(c) for c in diag if c in ctx_owner),
        "pool_sha256": res["pool_sha256"] == pool_sha256(fams),
    }


def _groups(res: dict) -> dict:
    a = res["assignment"]
    by = defaultdict(list)
    for f in sorted(a):
        by[a[f]].append(f)
    return {"pool": sorted(f for f in a if a[f] in POOL_ASSIGNMENTS), "heldout": by["heldout"], "dev": by["dev"],
            "train_fit_pool": by["train_fit"], "train_nonpool": by["train_nonpool"], "training": list(res["train_fit"]),
            "sealed_excluded": by["sealed_excluded"], "prohibited_excluded": by["prohibited_excluded"], "all": sorted(a)}


def _datasets(families: dict, fids, included_only: bool = True) -> set:
    """Member dataset_hash values; included_only keeps members whose epoch_status is 'included' (datasets that have
    included epoch contexts), dropping withdrawn-only, registry-sibling and external members."""
    return {m["dataset_hash"] for f in fids for m in families[f]["members"] if isinstance(m.get("dataset_hash"), str)
            and (not included_only or m.get("epoch_status") == "included")}


def _realized_fractions(families: dict, res: dict, groups: dict) -> dict:
    diag = set(res["diag"])

    def folds(g):
        drop = diag if g == "training" else set()
        return sum(1 for f in groups[g] for c in families[f]["contexts"] if c not in drop)
    measures = (("families", lambda g: len(groups[g])),
                ("datasets", lambda g: len(_datasets(families, groups[g]))),
                ("member_datasets", lambda g: len(_datasets(families, groups[g], included_only=False))),
                ("folds", folds))
    out = {}
    for grain, measure in measures:
        n = {g: measure(g) for g in ("heldout", "dev", "pool", "training", "all")}
        out[grain] = n | {"heldout_over_pool": _ratio(n["heldout"], n["pool"]), "heldout_over_all": _ratio(n["heldout"], n["all"]),
                          "dev_over_pool": _ratio(n["dev"], n["pool"]), "training_over_all": _ratio(n["training"], n["all"])}
    out["datasets"]["unit"] = "member dataset_hash with epoch_status == included (datasets with included contexts)"
    out["member_datasets"]["unit"] = "every member dataset_hash, withdrawn-only, sibling and contextless members included"
    out["folds"]["unit"] = "context = (dataset, outer fold)"
    out["folds"]["training_unit"] = "train_fit contexts minus the split-only diagnostic (excluded from the optimizer)"
    out["folds"]["diag"] = len(diag)
    for grain in ("programs", "records"):
        out[grain] = {"status": "not_computed", "reason": "needs per-context counts: freeze --context-stats (P4a/P5 inputs)"}
    return out


def _family_composition(families: dict, fids) -> dict:
    cohort, platform, size, n_ctx = Counter(), Counter(), Counter(), []
    for f in fids:
        rec = families[f]
        coh = rec.get("cohorts")
        if not (isinstance(coh, list) and all(isinstance(c, str) for c in coh)):
            coh = [m.get("cohort") for m in rec["members"] if isinstance(m.get("cohort"), str)]
        cohort["+".join(sorted(set(coh))) or "unknown"] += 1
        disp = rec.get("display")
        plats = disp.get("platforms") if isinstance(disp, dict) else None
        plats = sorted({p for p in plats if isinstance(p, str)}) if isinstance(plats, list) else []
        platform["withheld" if rec["sealed"] else ("+".join(plats) or "unknown")] += 1
        size[str(len(_datasets(families, [f])))] += 1
        n_ctx.append(len(rec["contexts"]))
    return {"families": len(fids), "datasets": len(_datasets(families, fids)),
            "member_datasets": len(_datasets(families, fids, included_only=False)), "folds": sum(n_ctx),
            "cohort": dict(sorted(cohort.items())), "platform": dict(sorted(platform.items())),
            "family_datasets_histogram": dict(sorted(size.items())), "family_folds_quantiles": _quantiles(n_ctx)}


def build_holdout_doc(reg: Registry, res: dict, com: dict, seed: bytes, gate: dict) -> dict:
    """holdout.json: a pure function of the registry bytes, the seed, the commitment and the gate (no clock), so verify
    can rebuild it byte for byte and a rerun of freeze reproduces it."""
    fams = reg.families
    groups = _groups(res)
    member_assignment = {}
    for f in sorted(fams):
        for m in fams[f]["members"]:
            member_assignment[_safe_identity(m["identity"], fams[f]["sealed"])] = res["assignment"][f]
    n_ctx_rest = sum(1 for f in res["train_fit"] for c in fams[f]["contexts"] if not fams[f]["frozen_fold0"].get(c))
    doc = {k: res[k] for k in FREEZE_KEYS}
    doc.update({
        "experiment_id": EXPERIMENT_ID, "frac": FRAC, "estimand": ESTIMAND, "algorithm": ALGORITHM,
        "pool_predicate": POOL_PREDICATE,
        "registry": {"sha256": reg.sha256, "bytes": reg.bytes, "phase": reg.doc.get("phase"),
                     "review_decisions_sha256": reg.doc.get("review_decisions_sha256"),
                     "review_csv_sha256": reg.doc.get("review_csv_sha256"),
                     "alias_sources_sha256": reg.doc.get("alias_sources_sha256"), "s3": gate["registry_s3"]},
        "gate": {"p2_receipt_sha256": gate["p2_receipt"]["sha256"],
                 "owner_decisions_sha256": gate["owner_decisions"]["sha256"], "decisions": gate["decisions"]},
        "commitment": {k: com["ref"][k] for k in REF_KEYS},
        "seed_sha256": sha256(seed),
        "counts": {"families": len(fams), "pool_N": res["N"], "heldout_k": res["k"], "dev": len(res["dev"]),
                   "train_fit_families": len(res["train_fit"]), "ctx_rest": n_ctx_rest, "diag_contexts": len(res["diag"]),
                   "member_identities": len(member_assignment),
                   "assignment": dict(sorted(Counter(res["assignment"].values()).items()))},
        "member_assignment": member_assignment,
        "realized_fractions": _realized_fractions(fams, res, groups),
        "composition": {g: _family_composition(fams, groups[g]) for g in COMPOSITION_GROUPS},
        "rules": {"diag": "diagnostic_only (NONINDEPENDENT_SPLIT_DIAGNOSTIC); excluded from the optimizer; drawn once",
                  "heldout": "read only at the frozen confirmatory unsealings (PLAN §6.2)",
                  "train_fit": "every fit uses train_fit only; train_nonpool families train but are never held or dev"},
    })
    return doc


def holdout_doc_problems(doc) -> list[str]:
    """Structural self-check of a holdout.json without the registry (publish runs it before any S3 call)."""
    if not isinstance(doc, dict):
        return ["holdout.json is not an object"]
    p = []
    for k, v in (("schema", HOLDOUT_SCHEMA), ("experiment_id", EXPERIMENT_ID), ("estimand", ESTIMAND), ("frac", FRAC),
                 ("algorithm", ALGORITHM)):
        if doc.get(k) != v:
            p.append(f"{k} does not match")
    N, k = doc.get("N"), doc.get("k")
    held, dev, a, tf = doc.get("held"), doc.get("dev"), doc.get("assignment"), doc.get("train_fit")
    if not (isinstance(N, int) and isinstance(k, int) and isinstance(held, list) and isinstance(dev, list)
            and isinstance(a, dict) and isinstance(tf, list) and isinstance(doc.get("diag"), list)):
        return p + ["N, k, held, dev, diag, assignment or train_fit is missing or ill-typed"]
    if N < MIN_POOL or k != math.ceil(FRAC * N) or len(held) != k or len(set(held)) != k:
        p.append("k != ceil(0.05·N)")
    if len(dev) != math.ceil(FRAC * (N - k)) or len(set(dev)) != len(dev) or set(held) & set(dev):
        p.append("dev size or disjointness")
    if any(v not in ASSIGNMENTS for v in a.values()):
        p.append("unknown assignment label")
    labels = Counter(a.values())
    if any(a.get(f) != "heldout" for f in held) or labels["heldout"] != k:
        p.append("heldout assignment")
    if any(a.get(f) != "dev" for f in dev) or labels["dev"] != len(dev):
        p.append("dev assignment")
    if sum(labels[x] for x in POOL_ASSIGNMENTS) != N:
        p.append("pool assignments != N")
    if set(tf) != {f for f, v in a.items() if v in TRAIN_ASSIGNMENTS}:
        p.append("train_fit != train_fit + train_nonpool assignments")
    for key in ("tau_hex", "tau_dev_hex", "pool_sha256", "seed_sha256"):
        if not _hex64(doc.get(key)):
            p.append(f"{key} is not 64 hex")
    com = doc.get("commitment")
    if not isinstance(com, dict) or com.get("key") != _key(COMMITMENT_REL) or not _usable_vid(com.get("version_id")):
        p.append("commitment ref missing or names another key")
    reg = doc.get("registry")
    if not isinstance(reg, dict) or not _hex64(reg.get("sha256")):
        p.append("registry sha256 missing")
    else:
        s3ref = reg.get("s3")
        if not (isinstance(s3ref, dict) and s3ref.get("key") == _key(REGISTRY_OBJECT_REL.format(sha=reg["sha256"]))
                and s3ref.get("sha256") == reg["sha256"] and _usable_vid(s3ref.get("version_id"))):
            p.append("registry S3 ref missing (the P3 gate records it)")
    gate = doc.get("gate")
    if not (isinstance(gate, dict) and _hex64(gate.get("p2_receipt_sha256")) and _hex64(gate.get("owner_decisions_sha256"))
            and gate.get("decisions") == GATE_DECISION_VALUES):
        p.append("P3 gate record missing or incomplete")
    return p


# ------------------------------------------------------------------------------------------- context statistics
def load_context_stats(path, expected_sha256: str) -> tuple[dict, dict]:
    if not isinstance(expected_sha256, str) or not HEX64.fullmatch(expected_sha256):
        raise HoldoutStop("--context-stats needs --context-stats-sha256 (64 hex)")
    raw, meta = _load_pinned(path, expected_sha256)
    doc = loads_strict(raw)
    if not isinstance(doc, dict) or doc.get("schema") != CONTEXT_STATS_SCHEMA or not isinstance(doc.get("contexts"), dict):
        raise HoldoutStop(f"context stats must be {CONTEXT_STATS_SCHEMA} with a contexts object")
    for c, v in doc["contexts"].items():
        if not HEX64.fullmatch(str(c)) or not isinstance(v, dict) or set(v) - set(STAT_INT_FIELDS + STAT_FLOAT_FIELDS):
            raise HoldoutStop("context stats: keys must be context_sha256 and fields programs/records/n/p/target_scale")
        for fld, x in v.items():
            ok = (isinstance(x, int) and not isinstance(x, bool) and x >= 0) if fld in STAT_INT_FIELDS else \
                 (isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0)
            if not ok:
                raise HoldoutStop(f"context stats: {fld} must be a finite non-negative number")
    return doc, meta


def context_stats_report(families: dict, holdout_doc: dict, holdout_sha256: str, stats: dict, stats_meta: dict) -> dict:
    a = holdout_doc["assignment"]
    diag = set(holdout_doc["diag"])

    def ctx_of(pred):
        return sorted(c for f in sorted(families) if pred(f) for c in families[f]["contexts"])
    groups = {"heldout": ctx_of(lambda f: a[f] == "heldout"), "dev": ctx_of(lambda f: a[f] == "dev"), "diag": sorted(diag),
              "training": [c for c in ctx_of(lambda f: a[f] in TRAIN_ASSIGNMENTS) if c not in diag],
              "pool": ctx_of(lambda f: a[f] in POOL_ASSIGNMENTS), "all": ctx_of(lambda f: True)}
    st = stats["contexts"]
    fractions = {}
    for grain in ("programs", "records"):
        n = {g: sum(st[c][grain] for c in v if grain in st.get(c, {})) for g, v in groups.items()}
        fractions[grain] = n | {"heldout_over_pool": _ratio(n["heldout"], n["pool"]), "heldout_over_all": _ratio(n["heldout"], n["all"]),
                                "dev_over_pool": _ratio(n["dev"], n["pool"]), "diag_over_all": _ratio(n["diag"], n["all"]),
                                "training_over_all": _ratio(n["training"], n["all"])}
    quant = {g: {fld: _quantiles([st[c][fld] for c in v if fld in st.get(c, {})]) for fld in ("n", "p", "target_scale")}
             for g, v in groups.items()}
    return {"schema": STATS_SCHEMA, "experiment_id": EXPERIMENT_ID, "holdout_sha256": holdout_sha256,
            "context_stats": stats_meta, "estimand": ESTIMAND,
            "coverage": {g: {"contexts": len(v), "with_stats": sum(c in st for c in v)} for g, v in groups.items()},
            "field_coverage": {fld: sum(1 for c in groups["all"] if fld in st.get(c, {})) for fld in EXPECT_STATS_ITEMS},
            "realized_fractions": fractions, "composition_quantiles": quant}


def expect_report(checks: dict, stats_report: dict | None = None) -> dict:
    """Which PLAN §5 P3 Expect items this run reports. Items that need per-context statistics count only when every
    registry context has the feeding field; complete is false until then, so no receipt claims the Expect is met."""
    missing = []
    for fld, item in EXPECT_STATS_ITEMS.items():
        if stats_report is None or stats_report["field_coverage"].get(fld) != stats_report["coverage"]["all"]["contexts"]:
            missing.append(item)
    reported = [i for i in EXPECT_REGISTRY_ITEMS] + [i for i in EXPECT_STATS_ITEMS.values() if i not in missing]
    ok = bool(checks.get("k_exact")) and bool(checks.get("every_family_exactly_one_assignment"))
    return {"estimand": ESTIMAND, "k_exact": bool(checks.get("k_exact")),
            "every_family_exactly_one_assignment": bool(checks.get("every_family_exactly_one_assignment")),
            "reported": reported, "missing": missing, "complete": ok and not missing,
            "rule": "items in missing need freeze --context-stats covering every registry context"}


def freeze_cmd(registry_path, registry_sha256: str, seed_file, out_dir, context_stats=None,
               context_stats_sha256=None) -> dict:
    reg = load_registry(registry_path, registry_sha256)
    seed = read_seed(seed_file)
    hdir = Path(seed_file).parent
    com = load_commitment(hdir)
    gate = load_gate(hdir, reg)
    pool = pool_members(reg.families)
    check_commitment(com, reg, seed, pool)
    res = freeze(reg.families, pool, seed, FRAC)
    checks = freeze_checks(reg.families, pool, res, seed)
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise HoldoutStop(f"freeze checks failed: {failed}")
    body = common.canonical(build_holdout_doc(reg, res, com, seed, gate))
    out = Path(out_dir)
    w = _create_file(out / "holdout.json", body)
    outputs = {"holdout": w}
    report = None
    if context_stats is not None:
        stats, meta = load_context_stats(context_stats, context_stats_sha256)
        report = context_stats_report(reg.families, loads_strict(body), w["sha256"], stats, meta)
        outputs["stats"] = _write_json(out / f"holdout-stats-{meta['sha256'][:16]}.json", report)
    return {"status": "PASS", "checks": checks, "expect": expect_report(checks, report), "N": res["N"], "k": res["k"],
            "dev": len(res["dev"]), "diag": len(res["diag"]), "train_fit": len(res["train_fit"]),
            "holdout_sha256": w["sha256"], "registry_sha256": reg.sha256, "commitment_sha256": com["ref"]["sha256"],
            "outputs": outputs}


# --------------------------------------------------------------------------------------------- verify + poison
def _poison_hex(tag: str) -> str:
    return sha256(b"jope.dl-holdout-poison/v1\0" + tag.encode())


def _rewrite_leaves(obj, tag: str):
    """Every leaf replaced by a different value of the same JSON type (dict keys kept, None becomes a string)."""
    if isinstance(obj, dict):
        return {k: _rewrite_leaves(v, f"{tag}/{k}") for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite_leaves(v, f"{tag}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, bool):
        return not obj
    if isinstance(obj, int):
        return obj + 1 + int(_poison_hex(tag)[:4], 16)
    if isinstance(obj, float):
        return obj * 2.0 + 1.0
    return _poison_hex(f"{tag}:{obj}")


def _p_contexts_replaced(rec, fid):
    new = sorted(_poison_hex(f"replace:{fid}:{i}") for i in range(max(1, len(rec["contexts"]))))
    rec["contexts"], rec["frozen_fold0"] = new, {c: i % 2 == 0 for i, c in enumerate(new)}


def _p_contexts_added(rec, fid):
    add = [_poison_hex(f"add:{fid}:{i}") for i in range(3 * len(rec["contexts"]) + 7)]
    rec["contexts"] = sorted(set(rec["contexts"]) | set(add))
    rec["frozen_fold0"] = dict(rec["frozen_fold0"]) | {c: False for c in add}


def _p_contexts_removed(rec, fid):
    rec["contexts"], rec["frozen_fold0"] = [], {}


def _p_fold0_flipped(rec, fid):
    rec["frozen_fold0"] = {c: not v for c, v in rec["frozen_fold0"].items()}


def _new_member(fid: str) -> dict:
    return {"identity": "item:" + _poison_hex("member:" + fid), "kind": "registry_item",
            "dataset_hash": _poison_hex("new-dataset:" + fid)[:16], "canonical_sha256": _poison_hex("new-canonical:" + fid),
            "sample_identity_sha256": _poison_hex("new-sample:" + fid), "item_id": _poison_hex("new-item:" + fid),
            "cohort": "incoming", "epoch_status": "included", "registry_state": "validated", "display": None,
            "union_keys": [{"kind": "dataset_hash", "value": _poison_hex("new-dataset:" + fid)[:16]}]}


def _p_members_changed(rec, fid):
    for m in rec["members"]:
        m["dataset_hash"] = _poison_hex("dataset:" + m["identity"])[:16]
        m["canonical_sha256"] = _poison_hex("canonical:" + m["identity"])
    rec["members"].append(_new_member(fid))


def _p_members_rewritten(rec, fid):
    """Every member leaf (identity, item_id, hashes, cohort, kind, statuses, display, union_keys) rewritten, the member
    order reversed, one member dropped (when there are several) and one added."""
    members = [_rewrite_leaves(m, f"member:{fid}:{i}") for i, m in enumerate(rec["members"])]
    members.reverse()
    if len(members) > 1:
        members.pop()
    rec["members"] = members + [_new_member(fid)]


def _p_metadata_changed(rec, fid):
    for k in sorted(set(rec) - set(POOL_FIXED_FIELDS) - {"contexts", "frozen_fold0", "members"}):
        rec[k] = {"poison": _poison_hex(f"meta:{fid}:{k}")}
    rec["poison_note"] = _poison_hex("note:" + fid)


def _p_pool_fields_changed(rec, fid):
    """Pool-predicate values changed without leaving the pool: rights unknown <-> recorded:<spdx>, auto <-> reviewed."""
    rec["rights_status"] = ("recorded:Poison-" + _poison_hex("spdx:" + fid)[:8]) if rec["rights_status"] == "unknown" \
        else "unknown"
    rec["independence_qualified"] = "reviewed" if rec["independence_qualified"] == "auto" else "auto"


def _p_all_leaves_rewritten(rec, fid):
    """Generic: every leaf of every field outside the pool-predicate fields rewritten."""
    for k in sorted(set(rec) - set(POOL_FIXED_FIELDS)):
        rec[k] = _rewrite_leaves(rec[k], f"leaf:{fid}:{k}")


POISON_VARIANTS = (
    ("held_contexts_replaced", "heldout", (_p_contexts_replaced,)),
    ("held_contexts_added", "heldout", (_p_contexts_added,)),
    ("held_contexts_removed", "heldout", (_p_contexts_removed,)),
    ("held_frozen_fold0_flipped", "heldout", (_p_fold0_flipped,)),
    ("held_members_changed", "heldout", (_p_members_changed,)),
    ("held_members_rewritten", "heldout", (_p_members_rewritten,)),
    ("held_metadata_changed", "heldout", (_p_metadata_changed,)),
    ("held_pool_fields_changed", "heldout", (_p_pool_fields_changed,)),
    ("held_all_leaves_rewritten", "heldout", (_p_all_leaves_rewritten,)),
    ("held_cumulative", "heldout", (_p_contexts_replaced, _p_contexts_added, _p_fold0_flipped, _p_members_changed,
                                     _p_members_rewritten, _p_metadata_changed, _p_pool_fields_changed,
                                     _p_all_leaves_rewritten)),
    ("dev_contexts_added", "dev", (_p_contexts_added, _p_fold0_flipped)),
)


def poison_variants(families: dict, base: dict):
    """Honest perturbations of held-family records (and, as an extra, dev-family contexts). Family ids and pool
    membership stay fixed, so the pool is unchanged by construction; only the perturbed records are copied."""
    for name, target, fns in POISON_VARIANTS:
        targets = base["held"] if target == "heldout" else base["dev"]
        fams = dict(families)
        for f in targets:
            rec = copy.deepcopy(families[f])
            for fn in fns:
                fn(rec, f)
            fams[f] = rec
        yield name, target, len(targets), fams


def run_assignment_poison(families: dict, seed: bytes, freeze_fn=None, frac: float = FRAC) -> dict:
    freeze_fn = freeze_fn or freeze
    pool = pool_members(families)
    base = freeze_fn(families, pool, seed, frac)
    variants = []
    for name, target, n, fams in poison_variants(families, base):
        if pool_members(fams) != pool:
            raise HoldoutStop(f"poison variant {name} changed the pool (bug in the perturbation)")
        out = freeze_fn(fams, pool, seed, frac)
        changed = [k for k in POISON_KEYS if out.get(k) != base.get(k)]
        variants.append({"variant": name, "perturbed": target, "families": n, "changed": changed,
                         "spec_keys_changed": [k for k in changed if k in POISON_SPEC_KEYS]})
    return {"ran": True, "pass": all(not v["changed"] for v in variants), "invariant_keys": list(POISON_KEYS),
            "spec_keys": list(POISON_SPEC_KEYS), "variants": variants}


def verify(holdout_path, registry_path, seed_file, registry_sha256=None, assignment_poison: bool = False,
           freeze_fn=None) -> dict:
    freeze_fn = freeze_fn or freeze
    h_raw = Path(holdout_path).read_bytes()
    h_doc = loads_strict(h_raw)
    bound = (h_doc.get("registry") or {}).get("sha256") if isinstance(h_doc, dict) else None
    if registry_sha256 is not None and registry_sha256 != bound:
        raise HoldoutStop("--registry-sha256 differs from the registry sha256 recorded in holdout.json")
    reg = load_registry(registry_path, bound)
    seed = read_seed(seed_file)
    hdir = Path(seed_file).parent
    com = load_commitment(hdir)
    gate = load_gate(hdir, reg)
    pool = pool_members(reg.families)
    check_commitment(com, reg, seed, pool)
    res = freeze_fn(reg.families, pool, seed, FRAC)
    checks = freeze_checks(reg.families, pool, res, seed)
    diffs = [k for k in FREEZE_KEYS if res.get(k) != h_doc.get(k)]
    bytes_equal = common.canonical(build_holdout_doc(reg, res, com, seed, gate)) == h_raw
    poison = run_assignment_poison(reg.families, seed, freeze_fn) if assignment_poison else {"ran": False}
    failed = [k for k, v in checks.items() if not v]
    ok = not failed and not diffs and bytes_equal and (not assignment_poison or poison["pass"])
    return {"status": "PASS" if ok else "STOP", "holdout_sha256": sha256(h_raw), "registry_sha256": reg.sha256,
            "N": res["N"], "k": res["k"], "checks": checks, "failed_checks": failed, "field_diffs": diffs,
            "bytes_equal": bytes_equal, "expect": expect_report(checks), "assignment_poison": poison}


# ----------------------------------------------------------------------------------------------------- closure
def _ro_connect(db_path):
    """sqlite3 mode=ro plus PRAGMA query_only=1 (PLAN §2: the intake registry is never written)."""
    p = str(Path(db_path).resolve())
    if "?" in p or "#" in p:
        raise HoldoutStop("registry DB path must not contain '?' or '#'")
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    con.execute("PRAGMA query_only=1")
    return con


def read_registry_contexts(db_path, item_ids) -> dict:
    ids = sorted(set(item_ids))
    con = _ro_connect(db_path)
    try:
        con.execute("BEGIN")                       # one read snapshot for poll_seq and the rows
        poll_seq = con.execute("select max(seq) from polls").fetchone()[0]
        rows = []
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            q = "select context, item_id from contexts where item_id in (%s)" % ",".join("?" * len(chunk))
            rows += con.execute(q, chunk).fetchall()
        con.execute("COMMIT")
    finally:
        con.close()
    rows = sorted({(str(c), str(i)) for c, i in rows})
    by_item = defaultdict(list)
    for c, i in rows:
        by_item[i].append(c)
    return {"poll_seq": poll_seq, "by_item": dict(by_item), "rows": len(rows), "items_queried": len(ids),
            "snapshot_sha256": sha256(common.canonical([list(r) for r in rows]))}


class _EpochIndex:
    def __init__(self, rows):
        self.by_ctx, self.by_dataset, self.by_item = {}, defaultdict(list), defaultdict(list)
        for row in rows:
            c = row.get("context_sha256") if isinstance(row, dict) else None
            if not isinstance(c, str) or c in self.by_ctx:
                raise HoldoutStop("epoch rows must carry unique context_sha256 values")
            self.by_ctx[c] = row
            if isinstance(row.get("dataset_hash"), str):
                self.by_dataset[row["dataset_hash"]].append(c)
            if isinstance(row.get("item_id"), str):
                self.by_item[row["item_id"]].append(c)


def _master_equivalents(master_doc) -> dict:
    views = master_doc.get("views") if isinstance(master_doc, dict) else None
    if not isinstance(views, list):
        raise HoldoutStop("input-manifest-master must carry a views list")
    eq = {}
    for v in views:
        c = v.get("context_sha256") if isinstance(v, dict) else None
        if not isinstance(c, str):
            raise HoldoutStop("a master view lacks context_sha256")
        lst = [{"context_sha256": e.get("context_sha256"), "prepared_path": e.get("prepared_path"),
                "compatibility_identity": e.get("compatibility_identity")}
               for e in v.get("equivalent_evaluation_contexts") or [] if isinstance(e, dict)]
        eq[c] = sorted(lst, key=lambda x: (str(x["context_sha256"]), str(x["prepared_path"])))
    return eq


def _expand_equivalents(sources: dict, master_eq: dict) -> None:
    frontier = sorted(sources)
    while frontier:
        c = frontier.pop()
        for e in master_eq.get(c) or []:
            x = e["context_sha256"]
            if not isinstance(x, str):
                continue
            if x not in sources:
                frontier.append(x)
            sources[x].add("master_equivalent")


def _context_entries(scope: str, sources: dict, epoch: _EpochIndex, master_eq: dict, fold0: dict, unknown: list,
                     owner) -> list:
    out = []
    for c in sorted(sources):
        row = epoch.by_ctx.get(c)
        entry = {"context_sha256": c, "sources": sorted(sources[c]), "frozen_fold0": fold0.get(c), "owner": owner(c),
                 "epoch": {k: row[k] for k in EPOCH_REF_FIELDS if k in row} if row else None,
                 "equivalents": master_eq.get(c)}
        if row is None:
            unknown.append({"kind": "context_not_in_epoch", "scope": scope, "context_sha256": c,
                            "sources": entry["sources"], "missing": ["context_schema", "histories", "seeds"]})
        if c not in master_eq:
            cohort = row.get("cohort") if row else None
            unknown.append({"kind": "equivalents_unknown", "scope": scope, "context_sha256": c,
                            "reason": "no input-manifest-master view" + (" (incoming cohort)" if cohort == "incoming" else "")})
        out.append(entry)
    return out


def _seed_unknown(scope: str, n_contexts: int, rec: dict) -> dict:
    return {"kind": "seed_programs", "scope": scope, "contexts": n_contexts,
            "seed_selection_exposed": rec["seed_selection_exposed"], "seeds_from_sources": rec.get("seeds_from_sources"),
            "reason": SEED_UNKNOWN_REASON}


def _family_closure(fid, rec, assignment, epoch, master_eq, db_by_item, unknown, owner) -> dict:
    sealed, scope = rec["sealed"], f"family:{fid}"
    identities, item_ids, keys = [], [], []
    for m in sorted(rec["members"], key=lambda m: m["identity"]):
        ident = _safe_identity(m["identity"], sealed)
        identities.append({"identity": ident} | {k: m[k] for k in MEMBER_FIELDS if m.get(k) is not None})
        if m["identity"].startswith("item:"):
            item_ids.append(m["identity"][len("item:"):])
        else:
            unknown.append({"kind": "external_identity_contexts", "scope": scope, "identity": ident,
                            "reason": "not a registry item: its contexts outside epoch-000 are not P3 inputs"})
        keys += [_safe_key(u, sealed) | {"identity": ident} for u in m.get("union_keys") or []]
    sources = defaultdict(set)
    for c in rec["contexts"]:
        sources[c].add("registry_contexts")
    excluded = rec.get("excluded_contexts")
    for c in sorted(excluded) if isinstance(excluded, dict) else []:
        sources[c].add("registry_excluded_contexts")
    for c in rec.get("registry_only_contexts") or []:
        sources[c].add("registry_only_contexts")
    for m in rec["members"]:
        for c in epoch.by_dataset.get(m.get("dataset_hash"), []):
            sources[c].add("epoch_dataset_hash")
        for c in epoch.by_item.get(m.get("item_id"), []):
            sources[c].add("epoch_item_id")
    no_rows = []                                   # normal for frozen items: their contexts come from the epoch rows
    for iid in item_ids:
        rows = db_by_item.get(iid)
        if rows is None:
            no_rows.append("item:" + iid)
        for c in rows or []:
            sources[c].add("registry_db_contexts")
    _expand_equivalents(sources, master_eq)
    contexts = _context_entries(scope, sources, epoch, master_eq, rec["frozen_fold0"], unknown, owner)
    unknown.append(_seed_unknown(scope, len(contexts), rec))
    external = []
    for a in rec.get("external_aliases") or []:
        a = a if isinstance(a, dict) else {"external_id": a}
        x = {"source": a.get("source"), "key_kind": a.get("key_kind"), "role": a.get("role"),
             "external_id_sha256": sha256(str(a.get("external_id")).encode())}
        if not sealed and a.get("external_id") is not None:
            x["external_id"] = a["external_id"]
        external.append(x)
        unknown.append({"kind": "alias_source_contexts", "scope": scope, "source": x["source"], "key_kind": x["key_kind"],
                        "external_id_sha256": x["external_id_sha256"],
                        "reason": "contexts of alias-source campaigns are not enumerated by P3 inputs"})
    edges = []
    for e in rec.get("union_edges") or []:
        if isinstance(e, dict):
            edges.append(_safe_key(e, sealed) | {"a": _safe_identity(str(e.get("a")), sealed),
                                                 "b": _safe_identity(str(e.get("b")), sealed)})
    seeds = rec.get("seed_sources") if isinstance(rec.get("seed_sources"), list) else []
    return {"assignment": assignment, "sealed": sealed, "identities": identities,
            "registry_db_items_without_context_rows": sorted(no_rows),
            "aliases": {"union_keys": sorted(keys, key=common.canonical), "union_edges": sorted(edges, key=common.canonical),
                        "external_aliases": sorted(external, key=common.canonical)},
            "seeds": {"seed_selection_exposed": rec["seed_selection_exposed"],
                      "seed_sources": sorted(_safe_identity(str(s), sealed) for s in seeds),
                      "seeds_from_sources": rec.get("seeds_from_sources"),
                      "seed_programs": "closure_unknown (kind seed_programs)"},
            "contexts": contexts}


def cross_family_links(families: dict, diag: dict, assignment: dict, diag_set) -> list[dict]:
    """Closure contexts owned by another registry family. Blocking (a cross-partition leak, PLAN §6.2/§5 P5): a held,
    dev or sealed scope reaching a context of a family with a different assignment (a train family, or a diag context,
    among them), or a diag scope reaching a context of a non-training family. Same-assignment links are recorded only."""
    links = []
    for f in sorted(families):
        scope_a = assignment.get(f)
        for e in families[f]["contexts"]:
            o = e["owner"]
            if o is None or o["family_id"] == f:
                continue
            owner_a = assignment.get(o["family_id"])
            links.append({"scope": f"family:{f}", "scope_assignment": scope_a, "context_sha256": e["context_sha256"],
                          "owner_family": o["family_id"], "owner_assignment": owner_a, "sources": e["sources"],
                          "in_diag": e["context_sha256"] in diag_set, "blocking": owner_a != scope_a})
    for c in sorted(diag):
        f = diag[c]["family_id"]
        for e in diag[c]["contexts"]:
            o = e["owner"]
            if o is None or o["family_id"] == f:
                continue
            owner_a = assignment.get(o["family_id"])
            links.append({"scope": f"diag:{c}", "scope_assignment": "diag", "context_sha256": e["context_sha256"],
                          "owner_family": o["family_id"], "owner_assignment": owner_a, "sources": e["sources"],
                          "in_diag": e["context_sha256"] in diag_set, "blocking": owner_a not in TRAIN_ASSIGNMENTS})
    return links


def build_closure(holdout_doc: dict, holdout_sha256: str, reg: Registry, epoch_rows, master_doc, db: dict,
                  inputs: dict) -> dict:
    fams, a = reg.families, holdout_doc["assignment"]
    if set(a) != set(fams):
        raise HoldoutStop("the registry's families are not the holdout's assignment keys")
    epoch, master_eq, unknown = _EpochIndex(epoch_rows), _master_equivalents(master_doc), []
    ctx_owner = {c: f for f, rec in fams.items() for c in rec["contexts"]}

    def owner(c):
        f = ctx_owner.get(c)
        return None if f is None else {"family_id": f, "assignment": a[f]}
    families = {f: _family_closure(f, fams[f], a[f], epoch, master_eq, db["by_item"], unknown, owner)
                for f in sorted(fams) if a[f] in CLOSURE_FAMILY_ASSIGNMENTS}
    diag = {}
    for c in holdout_doc["diag"]:
        f = ctx_owner.get(c)
        if f is None or a[f] not in TRAIN_ASSIGNMENTS:
            raise HoldoutStop(f"diag context {c} does not belong to a train_fit family")
        sources = defaultdict(set, {c: {"diag"}})
        _expand_equivalents(sources, master_eq)
        scope = f"diag:{c}"
        diag[c] = {"family_id": f, "contexts": _context_entries(scope, sources, epoch, master_eq,
                                                                {c: fams[f]["frozen_fold0"].get(c)}, unknown, owner)}
        unknown.append(_seed_unknown(scope, len(diag[c]["contexts"]), fams[f]))
    unknown.append({"kind": "reuse_cache_reachable_contexts", "scope": "global",
                    "reason": "contexts of other campaigns reachable through the reuse cache (for example other "
                              "ten-case campaigns) are not P3 inputs"})
    uniq = {common.canonical(u): u for u in unknown}
    unknown = [uniq[k] for k in sorted(uniq)]
    links = cross_family_links(families, diag, a, set(holdout_doc["diag"]))
    blocking = sum(1 for x in links if x["blocking"])
    return {"schema": CLOSURE_SCHEMA, "experiment_id": EXPERIMENT_ID, "holdout_sha256": holdout_sha256,
            "registry_sha256": reg.sha256, "inputs": inputs, "status": "STOP" if blocking else "PASS",
            "rule": "PLAN §5 P3 step 4: held, dev, diag and sealed closure; anything these inputs cannot enumerate is "
                    "closure_unknown; a blocking cross-family link (cross_partition_closure) is a STOP",
            "families": families, "diag": diag, "closure_unknown": unknown, "cross_family_links": links,
            "counts": {"families": dict(sorted(Counter(v["assignment"] for v in families.values()).items())),
                       "family_contexts": sum(len(v["contexts"]) for v in families.values()),
                       "diag_contexts": len(diag),
                       "diag_closure_contexts": sum(len(v["contexts"]) for v in diag.values()),
                       "closure_unknown": dict(sorted(Counter(u["kind"] for u in unknown).items())),
                       "cross_family_links": len(links), "blocking_links": blocking}}


def _resolve_pin(what: str, cli, derived, derived_from: str) -> tuple[str, str]:
    """The sha256 pin of a closure input, from argv or from a validated registry/epoch field. None is a STOP."""
    if cli is not None and not _hex64(cli):
        raise HoldoutStop(f"--{what}-sha256 must be 64 lowercase hex")
    if cli is not None and derived is not None and cli != derived:
        raise HoldoutStop(f"--{what}-sha256 differs from the {derived_from} pin")
    if cli is None and derived is None:
        raise HoldoutStop(f"the {what} input has no pin: pass --{what}-sha256 (there is no {derived_from} pin)")
    return (cli, "argv") if cli is not None else (derived, derived_from)


def _registry_epoch_pin(reg: Registry):
    ref = reg.doc.get("epoch_ref")
    if ref is None:
        return None
    if not isinstance(ref, dict) or not _hex64(ref.get("sha256")):
        raise HoldoutStop("registry epoch_ref must be an object with a 64-hex sha256")
    if ref.get("key") is not None and not (isinstance(ref["key"], str) and ref["key"].endswith(f"/{ref['sha256']}.json")):
        raise HoldoutStop("registry epoch_ref key is not the epoch's content-addressed key")
    if ref.get("version_id") is not None and not _usable_vid(ref["version_id"]):
        raise HoldoutStop("registry epoch_ref has no usable version_id")
    return ref["sha256"]


def _epoch_master_pin(epoch_doc: dict):
    qc = epoch_doc.get("quarantine_check")
    if qc is None:
        return None
    mm = qc.get("master_manifest") if isinstance(qc, dict) else None
    if not isinstance(mm, dict) or not _hex64(mm.get("sha256")):
        raise HoldoutStop("epoch quarantine_check.master_manifest lacks a 64-hex sha256")
    return mm["sha256"]


def _registry_alias_pin(reg: Registry):
    v = reg.doc.get("alias_sources_sha256")
    if v is None:
        return None
    if not _hex64(v):
        raise HoldoutStop("registry alias_sources_sha256 must be 64 lowercase hex")
    return v


def closure(holdout_path, registry_path, epoch_path, master_path, registry_db, alias_sources_path, out_dir=None,
            registry_sha256=None, epoch_sha256=None, master_sha256=None, alias_sources_sha256=None) -> dict:
    h_raw = Path(holdout_path).read_bytes()
    h_doc = loads_strict(h_raw)
    problems = holdout_doc_problems(h_doc)
    if problems:
        raise HoldoutStop(f"holdout.json self-check: {problems}")
    bound = h_doc["registry"]["sha256"]
    if registry_sha256 is not None and registry_sha256 != bound:
        raise HoldoutStop("--registry-sha256 differs from the registry sha256 recorded in holdout.json")
    reg = load_registry(registry_path, bound)
    e_pin, e_by = _resolve_pin("epoch", epoch_sha256, _registry_epoch_pin(reg), "registry epoch_ref")
    e_raw, e_meta = _load_pinned(epoch_path, e_pin)
    epoch_doc = loads_strict(e_raw)
    if not isinstance(epoch_doc, dict) or epoch_doc.get("schema") != EPOCH_SCHEMA or not isinstance(epoch_doc.get("contexts"), list):
        raise HoldoutStop(f"the epoch must be {EPOCH_SCHEMA} with a contexts list")
    m_pin, m_by = _resolve_pin("master", master_sha256, _epoch_master_pin(epoch_doc), "epoch quarantine_check")
    m_raw, m_meta = _load_pinned(master_path, m_pin)
    master_doc = loads_strict(m_raw)
    a_pin, a_by = _resolve_pin("alias-sources", alias_sources_sha256, _registry_alias_pin(reg),
                               "registry alias_sources_sha256")
    a_raw, a_meta = _load_pinned(alias_sources_path, a_pin)
    alias_doc = loads_strict(a_raw)
    if not isinstance(alias_doc, dict) or alias_doc.get("schema") != ALIAS_SOURCES_SCHEMA:
        raise HoldoutStop(f"alias sources must be {ALIAS_SOURCES_SCHEMA}")
    alias_inputs = [{"name": i.get("name"), "role": i.get("role"), "sha256": i.get("sha256")}
                    for i in alias_doc.get("inputs") or [] if isinstance(i, dict)]
    a = h_doc["assignment"]
    items = [m["identity"][len("item:"):] for f, rec in reg.families.items() if a.get(f) in CLOSURE_FAMILY_ASSIGNMENTS
             for m in rec["members"] if m["identity"].startswith("item:")]
    db = read_registry_contexts(registry_db, items)
    inputs = {"holdout": {"path": str(holdout_path), "sha256": sha256(h_raw), "bytes": len(h_raw)},
              "registry": {"path": reg.path, "sha256": reg.sha256, "bytes": reg.bytes},
              "epoch": e_meta | {"pinned_by": e_by}, "master": m_meta | {"pinned_by": m_by},
              "registry_db": {"path": str(registry_db), "mode": "ro+query_only",
                              **{k: db[k] for k in ("poll_seq", "rows", "items_queried", "snapshot_sha256")}},
              "alias_sources": a_meta | {"pinned_by": a_by, "inputs": sorted(alias_inputs, key=common.canonical)}}
    doc = build_closure(h_doc, sha256(h_raw), reg, epoch_doc["contexts"], master_doc, db, inputs)
    body = common.canonical(doc)
    out = Path(out_dir) if out_dir else Path(holdout_path).parent
    w = _create_file(out / "closure.json", body)
    res = {"status": doc["status"], "closure": w, "counts": doc["counts"], "holdout_sha256": sha256(h_raw)}
    if doc["status"] != "PASS":
        keep = ("scope", "scope_assignment", "context_sha256", "owner_family", "owner_assignment", "sources", "in_diag")
        res.update(stop_reason="cross_partition_closure", experiment_state_if_bound="CONTAMINATED",
                   blocking_links=[{k: x[k] for k in keep} for x in doc["cross_family_links"] if x["blocking"]][:50])
    return res


# ----------------------------------------------------------------------------------------------------- publish
def binding_doc(holdout_doc: dict, holdout_ref: dict, closure_ref: dict) -> dict:
    """Deterministic (no clock), so a crash-rerun re-sends the same bytes and a 412 resolves to equal bytes."""
    return {"schema": BINDING_SCHEMA, "experiment_id": EXPERIMENT_ID, "kind": "holdout", "holdout_ref": holdout_ref,
            "commitment_ref": holdout_doc["commitment"], "registry_sha256": holdout_doc["registry"]["sha256"],
            "registry_ref": holdout_doc["registry"]["s3"], "closure_ref": closure_ref,
            "pool_sha256": holdout_doc["pool_sha256"], "N": holdout_doc["N"], "k": holdout_doc["k"],
            "tau_hex": holdout_doc["tau_hex"], "tau_dev_hex": holdout_doc["tau_dev_hex"], "estimand": ESTIMAND,
            "rule": "first freeze wins; a second binding is CONFLICT and the experiment is CONTAMINATED"}


def find_verify_receipt(holdout_dir, holdout_sha256: str, explicit=None) -> dict:
    paths = [Path(explicit)] if explicit else sorted(Path(holdout_dir, "receipts").glob("verify-*.json"))
    for p in paths:
        try:
            r = json.loads(p.read_bytes())
        except (OSError, ValueError):
            continue
        poison = r.get("assignment_poison") if isinstance(r, dict) else None
        if (isinstance(poison, dict) and r.get("schema") == RECEIPT_SCHEMA and r.get("command") == "verify"
                and r.get("status") == "PASS" and r.get("holdout_sha256") == holdout_sha256
                and poison.get("ran") is True and poison.get("pass") is True):
            return {"path": str(p), "sha256": common.sha256_file(p)}
    raise HoldoutStop(f"publish needs a PASS `verify --assignment-poison` receipt for holdout sha256 {holdout_sha256}")


def closure_gate(closure_path, holdout_sha256: str, holdout_doc: dict) -> tuple[bytes, dict]:
    """publish needs closure.json for exactly this holdout, covering every held, dev, sealed family and diag context,
    with status PASS and no blocking cross-family link when re-derived from holdout.json's assignment."""
    try:
        raw = Path(closure_path).read_bytes()
    except FileNotFoundError:
        raise HoldoutStop(f"publish needs closure.json ({closure_path} is missing): run closure first") from None
    try:
        cdoc = loads_strict(raw)
    except ValueError as err:
        raise HoldoutStop(f"closure.json is not strict JSON: {err}") from None
    if not isinstance(cdoc, dict) or common.canonical(cdoc) != raw:
        raise HoldoutStop("closure.json is not canonical JSON")
    p = []
    for k, v in (("schema", CLOSURE_SCHEMA), ("experiment_id", EXPERIMENT_ID), ("holdout_sha256", holdout_sha256),
                 ("registry_sha256", holdout_doc["registry"]["sha256"])):
        if cdoc.get(k) != v:
            p.append(f"closure {k} does not match")
    a = holdout_doc["assignment"]
    fams, diag = cdoc.get("families"), cdoc.get("diag")
    if not isinstance(fams, dict) or set(fams) != {f for f, v in a.items() if v in CLOSURE_FAMILY_ASSIGNMENTS}:
        p.append("closure families are not the holdout's held, dev and sealed families")
    if not isinstance(diag, dict) or set(diag) != set(holdout_doc["diag"]):
        p.append("closure diag contexts are not the holdout's diag")
    if p:
        raise HoldoutStop("closure gate: " + "; ".join(p))
    try:
        links = cross_family_links(fams, diag, a, set(holdout_doc["diag"]))
    except (KeyError, TypeError):
        raise HoldoutStop("closure gate: closure contexts lack owner records") from None
    blocking = [x for x in links if x["blocking"]]
    if blocking or cdoc.get("status") != "PASS":
        raise HoldoutStop(f"closure gate: cross_partition_closure ({len(blocking)} blocking cross-family links, "
                          f"closure status {cdoc.get('status')!r}); the holdout must not be bound")
    return raw, {"path": str(closure_path), "sha256": sha256(raw), "bytes": len(raw), "cross_family_links": len(links)}


def recheck_selection(holdout_doc: dict, commitment_body: bytes, version_id: str) -> dict:
    """Recompute, from the bound commitment and the holdout's own pool labels, everything that does not need the
    registry: seed, pool_sha256, N, k, held, dev, tau, tau_dev and p_tau (the verify receipt covers diag and
    train_fit, which need the registry)."""
    try:
        cdoc = loads_strict(commitment_body)
    except ValueError:
        raise HoldoutStop("the bound commitment is not strict JSON") from None
    p = []
    if not isinstance(cdoc, dict) or common.canonical(cdoc) != commitment_body:
        raise HoldoutStop("the bound commitment is not a canonical JSON object")
    reg = holdout_doc["registry"]
    want = {"schema": COMMITMENT_SCHEMA, "experiment_id": EXPERIMENT_ID, "algorithm": ALGORITHM, "frac": FRAC,
            "registry_sha256": reg["sha256"], "review_decisions_sha256": reg.get("review_decisions_sha256"),
            "pool_sha256": holdout_doc["pool_sha256"]}
    p += [f"commitment {k} does not match holdout.json" for k, v in want.items() if cdoc.get(k) != v]
    seed = derive_seed(commitment_body, version_id)
    if sha256(seed) != holdout_doc["seed_sha256"]:
        p.append("seed_sha256 does not derive from the bound commitment")
    pool = sorted(f for f, v in holdout_doc["assignment"].items() if v in POOL_ASSIGNMENTS)
    if pool_sha256(pool) != holdout_doc["pool_sha256"]:
        p.append("the pool labelled in the assignment does not hash to pool_sha256")
    N = len(pool)
    k = math.ceil(FRAC * N)
    if N < MIN_POOL:
        raise HoldoutStop("publish recheck: " + "; ".join(p + [f"pool of {N} families"]))
    order = sorted(pool, key=lambda f: (prio(seed, f), f))
    held = order[:k]
    dev = sorted(order[k:], key=lambda f: (prio(seed, f, DEV_DOMAIN), f))[:math.ceil(FRAC * (N - k))]
    tau = prio(seed, held[-1])
    recomputed = {"N": N, "k": k, "held": held, "dev": dev, "tau_hex": f"{tau:064x}", "p_tau": (tau + 1) / 2**256,
                  "tau_dev_hex": f"{prio(seed, dev[-1], DEV_DOMAIN):064x}" if dev else None}
    p += [f"{key} differs from the selection recomputed from the bound commitment"
          for key, v in recomputed.items() if holdout_doc.get(key) != v]
    if p:
        raise HoldoutStop("publish recheck: " + "; ".join(p))
    return {"seed_sha256": sha256(seed), "N": N, "k": k,
            "recomputed": sorted(recomputed) + ["pool_sha256", "seed_sha256"]}


def binding_mismatch(existing: bytes, expected: dict) -> list[str]:
    """Fields of an existing binding that differ from ours (strict JSON; BINDING_DESCRIPTIVE_FIELDS ignored)."""
    try:
        doc = loads_strict(existing)
    except ValueError:
        return ["<not strict JSON>"]
    if not isinstance(doc, dict):
        return ["<not an object>"]
    missing = object()
    keys = (set(doc) | set(expected)) - BINDING_DESCRIPTIVE_FIELDS
    return sorted(k for k in keys if doc.get(k, missing) != expected.get(k, missing))


def _their_ref(existing: bytes):
    try:
        doc = loads_strict(existing)
    except ValueError:
        return None
    ref = doc.get("holdout_ref") if isinstance(doc, dict) else None
    return {k: ref.get(k) for k in REF_KEYS} if isinstance(ref, dict) else None


def _publish_object(writer, reader, key: str, body: bytes) -> dict:
    """A create-only dependency (§6.6 step 1) whose key must then hold exactly one version: the ref we bind is unique."""
    res = _create_only(writer, reader, key, body)
    if not res["equal"]:
        raise HoldoutStop(f"POISONED_KEY: {key} holds different bytes")
    hist = key_history(reader, key)
    if not _single_version(hist, res["version_id"]):
        raise HoldoutStop(f"{key} does not hold exactly one version (object_history: {len(hist['versions'])} versions, "
                          f"{len(hist['delete_markers'])} delete markers); refusing to bind a ref that is not unique")
    return {"bucket": _s3().BUCKET, "key": key, "version_id": res["version_id"], "sha256": sha256(body),
            "bytes": len(body), "outcome": "created" if res["outcome"] == "created" else "already_present_equal_bytes",
            "attempts": res["attempts"]}


def publish(holdout_path, clients: dict, verify_receipt=None, closure_path=None, *,
            draft_dry_run: bool = False) -> dict:
    if draft_dry_run:
        _draft_dry_run_guard(clients)
    hp = Path(holdout_path)
    body = hp.read_bytes()
    doc = loads_strict(body)
    if common.canonical(doc) != body:
        raise HoldoutStop("holdout.json is not canonical JSON")
    problems = holdout_doc_problems(doc)
    if problems:
        raise HoldoutStop(f"holdout.json self-check: {problems}")
    h_sha = sha256(body)
    gate = find_verify_receipt(hp.parent, h_sha, verify_receipt)
    cl_body, cl_meta = closure_gate(Path(closure_path) if closure_path else hp.parent / "closure.json", h_sha, doc)
    s3 = _s3()
    w, r = clients["writer"], clients["reader"]
    com = doc["commitment"]
    c_body, _ = s3.get_exact(r, com["key"], com["version_id"])
    if sha256(c_body) != com["sha256"] or len(c_body) != com["bytes"]:
        raise HoldoutStop("the bound commitment differs from the commitment named by holdout.json")
    recheck = recheck_selection(doc, c_body, com["version_id"])
    reg_meta = doc["registry"]
    registry_object = check_registry_object(r, reg_meta.get("s3"), reg_meta["sha256"], reg_meta.get("bytes"))  # P3I-2
    c_hist = key_history(r, com["key"])
    if not _single_version(c_hist, com["version_id"]):
        raise HoldoutConflict("CONFLICT: the commitment key does not hold exactly the bound version (a re-upload, a "
                              "second commitment or a delete marker); the seed is ambiguous; STOP for owner review",
                              stop_reason="commitment_history", commitment_history=c_hist)
    b_key = _key(BINDING_REL)
    b_before = key_history(r, b_key)
    if not _at_most_one_version(b_before):
        raise HoldoutConflict("CONFLICT: the holdout binding key has been rewritten or deleted; STOP, experiment "
                              "CONTAMINATED", experiment_state="CONTAMINATED", stop_reason="binding_history",
                              binding_history=b_before)
    m_key = _key(MANIFEST_REL.format(sha=h_sha))
    if b_before["versions"]:                         # bound before: refuse early when it names another holdout
        existing, _ = s3.get_exact(r, b_key, b_before["versions"][0]["version_id"])
        theirs = _their_ref(existing)
        if theirs is None or theirs.get("sha256") != h_sha or theirs.get("key") != m_key:
            raise HoldoutConflict("CONFLICT: the holdout binding names a different holdout; STOP, experiment CONTAMINATED",
                                  experiment_state="CONTAMINATED", stop_reason="binding_conflict",
                                  existing_binding={"key": b_key, "version_id": b_before["versions"][0]["version_id"],
                                                    "sha256": sha256(existing), "bytes": len(existing),
                                                    "holdout_ref": theirs},
                                  attempted_holdout_ref={"bucket": s3.BUCKET, "key": m_key, "version_id": None,
                                                         "sha256": h_sha, "bytes": len(body)})
    manifest = _publish_object(w, r, m_key, body)
    campaign = _publish_object(w, r, _key(CAMPAIGN_MANIFEST_REL.format(sha=h_sha)), body)
    closure_obj = _publish_object(w, r, _key(CLOSURE_OBJECT_REL.format(sha=cl_meta["sha256"])), cl_body)
    holdout_ref = {k: manifest[k] for k in REF_KEYS}
    closure_ref = {k: closure_obj[k] for k in REF_KEYS}
    expected = binding_doc(doc, holdout_ref, closure_ref)
    b = _create_only(w, r, b_key, common.canonical(expected))
    if b["equal"]:
        outcome = "created" if b["outcome"] == "created" else "already_present_equal_bytes"
    else:
        mismatch = binding_mismatch(b["existing"], expected)
        existing_binding = {"key": b_key, "version_id": b["version_id"], "sha256": b["sha256"], "bytes": b["bytes"],
                            "holdout_ref": _their_ref(b["existing"])}
        if "holdout_ref" in mismatch or "<not strict JSON>" in mismatch or "<not an object>" in mismatch:
            raise HoldoutConflict("CONFLICT: the holdout binding names a different holdout; STOP, experiment CONTAMINATED",
                                  experiment_state="CONTAMINATED", stop_reason="binding_conflict",
                                  existing_binding=existing_binding, attempted_holdout_ref=holdout_ref)
        if mismatch:
            raise HoldoutConflict("CONFLICT: the holdout binding names this holdout but differs in other fields; STOP, "
                                  "experiment CONTAMINATED", experiment_state="CONTAMINATED",
                                  stop_reason="binding_fields_differ", differing_fields=mismatch,
                                  existing_binding=existing_binding, attempted_holdout_ref=holdout_ref)
        outcome = "already_bound_same_ref"
    b_after = key_history(r, b_key)
    if not _single_version(b_after, b["version_id"]):
        raise HoldoutConflict("CONFLICT: the holdout binding key does not hold exactly one version; STOP, experiment "
                              "CONTAMINATED", experiment_state="CONTAMINATED", stop_reason="binding_history",
                              binding_history=b_after)
    return {"status": "PASS", "holdout_sha256": h_sha, "verify_receipt": gate, "closure": cl_meta, "recheck": recheck,
            "registry_object": registry_object, "draft_dry_run": bool(draft_dry_run),
            "commitment_ref": {k: com[k] for k in ("key", "version_id", "sha256", "bytes")},
            "manifest": manifest, "campaign_manifest": campaign, "closure_object": closure_obj,
            "binding": {"key": b_key, "version_id": b["version_id"], "sha256": b["sha256"], "bytes": b["bytes"],
                        "outcome": outcome, "attempts": b["attempts"]},
            "histories": {"commitment": c_hist, "binding_before": b_before, "binding_after": b_after}}


# --------------------------------------------------------------------------------------------------------- CLI
def _default_clients() -> dict:
    s3 = _s3()
    return {"writer": s3.make_client(writes=True), "reader": s3.make_client()}


def _err(err: BaseException) -> dict:
    out = {"type": type(err).__name__}
    if isinstance(err, (HoldoutStop, common.ImmutableConflict, ValueError, KeyError, OSError)):
        out["message"] = str(err)[:500]
    elif "dltrain.s3io" in sys.modules:
        out.update(sys.modules["dltrain.s3io"].safe_err(err))
    return out


def _identity() -> dict:
    """Code and runtime identity for every receipt (PLAN §5 Format)."""
    try:
        return {"source_tree_sha256": source_tree_sha256(), "source_tree": [list(x) for x in _source_tree()],
                "runtime": runtime_identity(), "runtime_sha256": runtime_sha256(),
                "selection_code_sha256": ALGORITHM["code_sha256"]}
    except Exception as err:  # noqa: BLE001 (recorded; never hides the command's own outcome)
        return {"identity_error": _err(err)}


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dltrain.holdout", description="PLAN §5 P3 holdout, dev split and diagnostic")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commit")
    c.add_argument("--registry", required=True)
    c.add_argument("--registry-sha256", required=True)
    c.add_argument("--decisions", required=True)
    c.add_argument("--review-csv", help="default: review.csv next to the registry")
    c.add_argument("--p2-receipt", help="default: <out>/../receipts/P2/receipt.json")
    c.add_argument("--p2-receipt-sha256")
    c.add_argument("--owner-decisions", default=OWNER_DECISIONS_PATH)
    c.add_argument("--owner-decisions-sha256", default=OWNER_DECISIONS_SHA256)
    c.add_argument("--out", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--registry", required=True)
    f.add_argument("--registry-sha256", required=True)
    f.add_argument("--seed-file", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--context-stats")
    f.add_argument("--context-stats-sha256")
    p = sub.add_parser("publish")
    p.add_argument("--holdout", required=True)
    p.add_argument("--verify-receipt", help="default: a PASS verify receipt under <holdout dir>/receipts/")
    p.add_argument("--closure", help="default: closure.json next to holdout.json")
    cl = sub.add_parser("closure")
    cl.add_argument("--holdout", required=True)
    cl.add_argument("--registry", required=True)
    cl.add_argument("--registry-sha256")
    cl.add_argument("--epoch", required=True)
    cl.add_argument("--epoch-sha256")
    cl.add_argument("--master", required=True)
    cl.add_argument("--master-sha256")
    cl.add_argument("--registry-db", required=True)
    cl.add_argument("--alias-sources", required=True)
    cl.add_argument("--alias-sources-sha256")
    cl.add_argument("--out")
    v = sub.add_parser("verify")
    v.add_argument("--holdout", required=True)
    v.add_argument("--registry", required=True)
    v.add_argument("--registry-sha256")
    v.add_argument("--seed-file", required=True)
    v.add_argument("--assignment-poison", action="store_true")
    for s in (c, f, p, cl, v):
        s.add_argument("--receipt", help="default: <dir>/receipts/<command>-<utc_us>.json")
    return ap


def _run(args, clients, draft_dry_run: bool = False) -> dict:
    if draft_dry_run and args.cmd in ("commit", "publish") and clients is None:
        raise HoldoutStop("a DRAFT dry run needs injected fake S3 clients; it never uses the real client factory")
    if args.cmd == "commit":
        return commit(args.registry, args.registry_sha256, args.decisions, args.out, clients or _default_clients(),
                      args.review_csv, args.p2_receipt, args.owner_decisions, args.owner_decisions_sha256,
                      args.p2_receipt_sha256, draft_dry_run=draft_dry_run)
    if args.cmd == "freeze":
        return freeze_cmd(args.registry, args.registry_sha256, args.seed_file, args.out, args.context_stats,
                          args.context_stats_sha256)
    if args.cmd == "publish":
        return publish(args.holdout, clients or _default_clients(), args.verify_receipt, args.closure,
                       draft_dry_run=draft_dry_run)
    if args.cmd == "closure":
        return closure(args.holdout, args.registry, args.epoch, args.master, args.registry_db, args.alias_sources,
                       args.out, args.registry_sha256, args.epoch_sha256, args.master_sha256, args.alias_sources_sha256)
    res = verify(args.holdout, args.registry, args.seed_file, args.registry_sha256, args.assignment_poison)
    res["assignment_poison_requested"] = bool(args.assignment_poison)
    return res


def _receipt_dir(args) -> Path:
    if args.cmd in ("commit", "freeze"):
        return Path(args.out) / "receipts"
    if args.cmd == "closure" and args.out:
        return Path(args.out) / "receipts"
    return Path(args.holdout).parent / "receipts"


def _main(argv=None, clients=None, *, draft_dry_run: bool = False) -> int:
    """The CLI. clients and draft_dry_run are Python-only, with no command-line flag: draft_dry_run serves the DRAFT
    dry run of tools/p2_decisions.py (fake clients only, see _draft_dry_run_guard) and is stamped into the receipt."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)
    started = common.utc_us()
    draft_dry_run = bool(draft_dry_run)
    base = {"schema": RECEIPT_SCHEMA, "command": args.cmd, "experiment_id": EXPERIMENT_ID, "started_utc_us": started,
            "argv": common.redact_argv(["dltrain.holdout"] + argv)} | _identity()
    if draft_dry_run:
        base |= {"draft_dry_run": True,
                 "notice": "DRAFT dry run: fake S3 only; never a P3 commitment, seed, holdout or binding"}
    try:
        result = _run(args, clients, draft_dry_run)
    except HoldoutConflict as err:
        result = {"status": "CONFLICT", "error": _err(err)} | err.details
    except (HoldoutStop, common.ImmutableConflict) as err:
        result = {"status": "STOP", "error": _err(err)}
    except Exception as err:  # noqa: BLE001 (recorded in the receipt; exit 3)
        result = {"status": "ERROR", "error": _err(err)}
    doc = base | result | {"finished_utc_us": common.utc_us()}
    path = Path(args.receipt) if args.receipt else _receipt_dir(args) / f"{args.cmd}-{started}.json"
    try:
        rec = common.write_receipt(path, doc)
    except Exception as err:  # noqa: BLE001
        print(json.dumps({"status": "ERROR", "receipt_error": _err(err)}), file=sys.stderr)
        return EXIT["ERROR"]
    summary = {k: doc.get(k) for k in ("command", "status", "draft_dry_run", "experiment_state", "stop_reason", "N",
                                       "k", "holdout_sha256", "seed_sha256", "s3_outcome", "failed_checks", "error")
               if k in doc}
    print(json.dumps(summary | {"receipt": {k: rec[k] for k in ("path", "sha256", "bytes")}}, sort_keys=True))
    return EXIT.get(doc["status"], EXIT["ERROR"])


if __name__ == "__main__":
    raise SystemExit(_main())

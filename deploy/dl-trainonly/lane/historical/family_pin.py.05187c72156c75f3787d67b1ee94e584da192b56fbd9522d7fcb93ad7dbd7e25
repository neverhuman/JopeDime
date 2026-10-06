"""PLAN §5 P2 step 1 (`pin`): the only P2 step that touches SQLite or S3.

Spec: RUN/receipts/P2/survey/wf_a43390a0-83c.output.json result.spec.algorithm step 1 (PLAN §5 P2, §6.2).
`families.py pin --out RUN/families/` calls pin(); `build`, `verify` and `publish` never call S3 or SQLite and read
only what this step pinned. Every check runs before the first write, and every local check runs before the first
GET, so a STOP leaves OUT untouched. The S3 reads already made at a STOP stay in the ledger journal.

Steps (spec numbering):
 1. epoch: sha256 == bcfb5230…; P1 PASS and its S3 epoch/binding refs; DL-D5/DL-D12 recorded in the owner decisions.
    Every prepared.json (all 5,017 rows) re-hashed against prepared_ref.
 2. registry snapshot: sqlite3 'file:<db>?mode=ro' + PRAGMA query_only=1 + one BEGIN..COMMIT; items whose dataset is in
    the epoch and their contexts; identity rule id == sha256(intake_canonical([dataset, canonical, sample])) and
    dataset == canonical[:16]; exactly one 'frozen' item per frozen dataset with sample == source_manifest key[-2];
    every epoch item_id exists with the same dataset and sample. Payloads keep only PAYLOAD_KEEP.
 3. incoming items: catalog-row.json bytes == intake_canonical(catalog_row)+b'\\n' and
    sha256(intake_canonical(catalog_row)) == catalog_row_sha256; materialization.json pinned; the local data object
    and research manifest re-hashed (file name == sha256 == published sha256).
 4. frozen datasets: research manifest (== prepared source_manifest), data object, readiness record, catalog v1 row
    (canonical, provenance and files.meta must agree), dataset-records/<h>.json (meta pin = record.meta).
 5. S3 metas: frozen exact GETs (record.meta.version), incoming content-addressed GETs (D-P2-1; response VersionId
    recorded), body sha256 == key suffix and len == bytes; inputs/meta/<sha>.json keeps only META_KEEP.
 6. sealed: v1..v5 re-hashed with the parent chain; S = members of SEALED_PARTITIONS families; for S - E with a
    dataset-record: T1 manifest, train and meta (exact) and test (content-addressed, pinned by the manifest) into
    inputs/sealed-ref/; an id without a record is 'content_unavailable'. Plus a 32-object sampled exact readback of
    frozen datasets/<h>/manifest.json at the readiness VersionIds.
 7. seeds and aliases: seed pools, receipts and scripts by sha; the 10 seed-source pointers and parquets
    (sha == pointer); all BeyondArena pointers (sha == pointer-receipt) and the index; v6 names = source-cache dirs
    whose parquet sha == pointer; the TabPFN-unseen root comes from the config only, never argv;
    inputs/classic-aliases.json.
 8. exposure inputs: split manifest, assembly, registry-members, qualification v4, index, target metadata, data.py,
    and the encoder weight identities (recorded, never read).
 9. alias-sources.json, written last (create-only); STOP if GETs > 1,200 or bytes > 20 MB (checked before every GET).

Disclosure rule: a sealed member's display name, source keys, URLs and lineage-B family names are never printed or
written in clear. Metas, snapshot catalog rows and sealed-ref manifests of base-sealed datasets (S, the Rdatasets
alias keys of S, or a sealed metadata_sha256) keep column names and numbers but carry {'value_sha256': …} in place of
every identifying string. STOP messages carry only ids, hashes, paths and field names. The live pin knows only the
base-sealed set; the build's seal set is larger (provisional seals, near-duplicates), so a live-pinned generation is
re-pinned locally (repin_local, below) before its inputs count as redacted (review fixes CL-5, SC-4).

Outputs under OUT (= RUN/families/), all create-only (common.immutable_write); relative 'local_path' values are
relative to OUT:
  inputs/registry-snapshot.json             jope.dl-family-registry-snapshot/v1
  inputs/meta/<blob sha256>.json            flat META_KEEP extract (non-finite numbers -> null)
  inputs/sealed-ref/<sha256>                raw T1 train/test blobs (content-addressed)
  inputs/sealed-ref/manifest/<sha256>.json  redacted T1 manifest extract
  inputs/sealed-ref/meta/<sha256>.json      redacted META_KEEP extract
  inputs/classic-aliases.json               jope.dl-family-classic-aliases/v1 (curated, agent-drafted, owner-reviewed)
  alias-sources.json                        jope.dl-family-alias-sources/v1 (inputs, s3_reads ledger, policy, …)

Every alias-sources input entry is {name, role, sha256, bytes} plus 'path' (an external local file whose bytes hash
to sha256), and/or 's3' {bucket, key, version_id} (the same bytes in S3), and/or 'local_path'/'local_sha256'/
'local_bytes' (a derived copy under OUT). Names follow INPUT_NAMES (alias-sources.json['input_names']).

API: pin(out, policy, config, clients, guard, ledger_path, receipt_path, preflight) runs the step; preflight=True runs
every local check and the S3 plan/budget with 0 S3 requests and no writes. Build-side helpers: load_alias_sources(),
input_index(), rehash_inputs() (re-checks every 'path'/'local_path'; build STOPs 're-pin' on a mismatch), CpuGuard,
intake_canonical(), identity_key(), value_sha256(). CLI (RUN phase): python -m dltrain.family_pin --out RUN/families
[--preflight] [--ledger RUN/receipts/P2/<attempt>.jsonl] [--receipt RUN/receipts/P2/<attempt>.json]; the policy comes
from dltrain.families.POLICY; the CLI refuses outputs outside RUN/families and RUN/receipts/P2.

Local re-pin (review fixes CL-5, SC-4, SC-3, SC-8; `families pin --from-local`): repin_local(src, out, from_receipt)
makes no S3 request and opens no SQLite database. verify_generation() re-checks the source generation against its pin
receipt and ledger journal and re-hashes every pinned file; families.run_build over the source inputs (current code and
POLICY, in memory) gives the seal set (members of sealed or withheld families); the new generation's inputs are
written redacted: META_LOSSLESS hashed in every meta (derive() never reads those fields), every META_SENSITIVE field
hashed for sealed or withheld members, ROW_SENSITIVE removed from frozen and incoming snapshot rows (build reads those
rows from the pinned catalog and item files) and hashed in sealed or withheld siblings' rows, and a pinned input under
RUN/receipts/P2 that holds canaries (the survey) re-pinned as a redacted copy. A second in-memory build over the new
inputs must reproduce the review rows, screen files, partition, seals and counts, and every written file must scan
clean, before alias-sources.json (with a repin block: source, verification, seal floor, redaction, withhold) is
written. load_withhold() reads a jope.dl-family-withhold/v1 list for --withhold.

Never imports jope, gp_worker, local_gateway, reuse_cache, intake_registry or family_protocol; their rules are copied.
"""
import argparse
import gzip
import json
import math
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from dltrain import common

EXPERIMENT_ID = "gp-policy-dl-v1-e0"
ALIAS_SOURCES_SCHEMA = "jope.dl-family-alias-sources/v1"
SNAPSHOT_SCHEMA = "jope.dl-family-registry-snapshot/v1"
CLASSIC_ALIASES_SCHEMA = "jope.dl-family-classic-aliases/v1"
RECEIPT_SCHEMA = "jope.dl-family-pin-receipt/v1"
REPIN_RECEIPT_SCHEMA = "jope.dl-family-repin-receipt/v1"
WITHHOLD_SCHEMA = "jope.dl-family-withhold/v1"
DATASETS_BUCKET = "veox-jopedime-datasets-use2"
MODELS_BUCKET = "veox-jopedime-models-use2"
LEARNING_BUCKET = "veox-jopedime-learning-use2"
RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
EPOCH_SHA256 = "bcfb5230bc0f5f32f6b42edbfa3a6bf76b89e0593b593b9042290e8c077a87ba"
SEALED_PARTITIONS = ("meta_validation", "meta_test")          # DL-D12; cross-checked against the exclusion policy
MAX_GETS = 1200
MAX_BYTES = 20_000_000
READBACK_SAMPLE = 32
READBACK_DOMAIN = b"jope.dl-family-pin-readback/v1\0"
GUARD_EVERY_GETS = 50
CPU_ADMISSION = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-per-fold-hwm-20261001/gp/cpu-admission.json")
PAYLOAD_KEEP = ("catalog_row", "catalog_row_sha256", "catalog_ref", "manifest_ref", "objects", "identity_key")
META_KEEP = ("feature_columns", "original_headers", "output_headers", "target_column", "dropped_feature_columns",
             "feature_scaling", "target_scaling", "source_family", "source_dataset_id", "source_platform", "source_url",
             "dedupe_fingerprint")
META_DEDUP_PATH = ("quality_checks", "gates", "deduplication")
META_SENSITIVE = ("source_family", "source_dataset_id", "source_platform", "source_url", "dedupe_fingerprint", "quality_checks")
# META_SENSITIVE fields family_keys.derive() never reads: hashed in every meta by the local re-pin (lossless; CL-5)
META_LOSSLESS = ("source_url", "source_dataset_id", "source_platform", "dedupe_fingerprint")
ROW_SENSITIVE = ("display_name", "provenance", "ingest", "aliases", "collections")
ROW_EXTRACT_KEEP = ("schema", "id", "dataset_hash", "canonical_sha256", "task", "origin", "shape", "split", "target", "files")
LINEAGE_B_VERSIONS = tuple(f"lineage_b_qualification_v{i}" for i in range(1, 6))
HEX16 = re.compile(r"[0-9a-f]{16}")
HEX64 = re.compile(r"[0-9a-f]{64}")
RDATASETS_URL = re.compile(r"vincentarelbundock\.github\.io/Rdatasets/(?:csv|doc)/([^/]+)/([^/?#]+?)(?:\.csv|\.html)?$",
                           re.IGNORECASE)
FORBIDDEN_IMPORTS = ("jope", "gp_worker", "local_gateway", "reuse_cache", "intake_registry", "family_protocol")
LAYOUT = {"alias_sources": "alias-sources.json", "registry_snapshot": "inputs/registry-snapshot.json", "meta": "inputs/meta/<blob sha256>.json",
          "sealed_ref_blob": "inputs/sealed-ref/<sha256>", "sealed_ref_manifest": "inputs/sealed-ref/manifest/<sha256>.json",
          "sealed_ref_meta": "inputs/sealed-ref/meta/<sha256>.json", "classic_aliases": "inputs/classic-aliases.json",
          "local_path": "relative to the directory holding alias-sources.json"}
INPUT_NAMES = {"frozen": ["frozen_manifest:<h>", "frozen_data:<h>", "dataset_record:<h>"],
               "incoming": ["incoming_catalog_row:<item_id>", "incoming_materialization:<item_id>", "incoming_data:<item_id>",
                            "incoming_manifest:<item_id>"],
               "meta": "meta:<blob sha256> (entry.datasets lists every dataset_hash using it)",
               "sealed": ["sealed_ref_manifest:<h>", "sealed_ref_train:<h>", "sealed_ref_test:<h>", "sealed_ref_meta:<h>", "dataset_record:<h>"],
               "beyondarena": ["ba_pointer:<name>", "ba_parquet:<name>"], "seeds": ["seed_candidates:<name>", "tencase_v3_prepared:<name>"],
               "single_files": "the default_fixed() names (epoch, readiness, catalog_v1, lineage_b_qualification_v1..v5, ...)",
               "encoder_identities": "encoder:<id> (identity_path, never read)", "aggregates": ["epoch_prepared_set", "registry_snapshot",
                                                                                                    "classic_aliases"]}


class PinStop(RuntimeError):
    """A P2 pin STOP: nothing is written under OUT; S3 reads already made stay in the ledger journal."""


# ----------------------------------------------------------------------------------------------- copied rules
def intake_canonical(value) -> bytes:
    """intake_contract.canonical (copied, never imported): sorted keys, compact, allow_nan=False, ensure_ascii=True.
    It differs from common.canonical for non-ASCII text; catalog-row.json, catalog_row_sha256 and item ids use it."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity_key(dataset: str, canonical_sha256: str, sample: str) -> str:
    """intake_contract.identity_key(identity(manifest)) (copied): the registry item id."""
    return common.sha256_bytes(intake_canonical([dataset, canonical_sha256, sample]))


def value_sha256(value) -> dict:
    """The only form a sealed member's identifying text takes in pin outputs."""
    text = value if isinstance(value, str) else common.canonical(value).decode("utf-8")
    return {"value_sha256": common.sha256_bytes(text.encode("utf-8"))}


def _hash_leaves(obj):
    """Every string or number leaf -> value_sha256 (keys, booleans and nulls kept)."""
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, dict):
        return {k: _hash_leaves(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_hash_leaves(v) for v in obj]
    return value_sha256(obj)


def _finite(obj) -> tuple[object, int]:
    """Non-finite floats (json.loads accepts NaN/Infinity) become null so canonical() can serialize; returns the count."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None, 1
    if isinstance(obj, dict):
        out, n = {}, 0
        for k, v in obj.items():
            out[k], m = _finite(v)
            n += m
        return out, n
    if isinstance(obj, list):
        out, n = [], 0
        for v in obj:
            w, m = _finite(v)
            out.append(w)
            n += m
        return out, n
    return obj, 0


def rdatasets_keys(provenance) -> set[str]:
    """Lower-cased 'pkg/name' keys from provenance (platform 'rdatasets' + dataset_id, and Rdatasets csv/doc URLs).
    Used here only to decide which metas to redact; family_keys.derive() owns the authoritative key derivation."""
    keys = set()
    if not isinstance(provenance, dict):
        return keys
    platform, dsid, url = provenance.get("platform"), provenance.get("dataset_id"), provenance.get("url")
    if isinstance(platform, str) and platform.strip().lower() == "rdatasets" and isinstance(dsid, str) and "/" in dsid:
        keys.add(dsid.strip().lower())
    if isinstance(url, str):
        m = RDATASETS_URL.search(url.strip())
        if m:
            keys.add(f"{m.group(1)}/{m.group(2)}".lower())
    return keys


def lineage_b_rdatasets_keys(dataset_entry: dict) -> set[str]:
    """Lineage-B side: source_key 'rdatasets:<pkg>/<name>' plus every evidence[].source_url on an Rdatasets csv/doc page."""
    keys = set()
    sk = dataset_entry.get("source_key")
    if isinstance(sk, str) and sk.startswith("rdatasets:"):
        keys.add(sk[len("rdatasets:"):].strip().lower())
    for ev in dataset_entry.get("evidence") or []:
        url = ev.get("source_url") if isinstance(ev, dict) else None
        if isinstance(url, str):
            m = RDATASETS_URL.search(url.strip())
            if m:
                keys.add(f"{m.group(1)}/{m.group(2)}".lower())
    return keys


def meta_extract(body: bytes, redact: bool = False) -> tuple[dict, dict]:
    """META_KEEP fields plus quality_checks.gates.deduplication; everything else (acquisition.agent_id, display
    names, …) is dropped. redact=True replaces META_SENSITIVE values with value_sha256 leaves."""
    try:
        doc = json.loads(body)
    except (ValueError, UnicodeDecodeError) as err:
        raise PinStop(f"meta blob is not JSON ({type(err).__name__})") from None
    if not isinstance(doc, dict):
        raise PinStop("meta blob is not a JSON object")
    out = {k: doc[k] for k in META_KEEP if k in doc}
    node = doc
    for part in META_DEDUP_PATH:
        node = node.get(part) if isinstance(node, dict) else None
    if node is not None:
        out["quality_checks"] = {"gates": {"deduplication": node}}
    out, nonfinite = _finite(out)
    if redact:
        for k in META_SENSITIVE:
            if k in out:
                out[k] = _hash_leaves(out[k])
    info = {"kept_fields": sorted(out), "dropped_top_level_fields": len(set(doc) - set(out)), "nonfinite_to_null": nonfinite,
            "redacted": bool(redact)}
    return out, info


def catalog_row_redacted(row: dict) -> dict:
    """A catalog row (or T1 manifest) with every ROW_SENSITIVE subtree hashed; column-level fields kept."""
    return {k: (_hash_leaves(v) if k in ROW_SENSITIVE else v) for k, v in row.items()}


def manifest_extract(row: dict) -> dict:
    """Redacted T1 manifest extract for a sealed reference: ROW_EXTRACT_KEEP plus hashed ROW_SENSITIVE fields."""
    out = {k: row[k] for k in ROW_EXTRACT_KEEP if k in row}
    for k in ROW_SENSITIVE:
        if k in row:
            out[k] = _hash_leaves(row[k])
    return out


def _hash_leaves_once(obj):
    """_hash_leaves that keeps an existing {'value_sha256': <64 hex>} leaf as it is (re-redaction is idempotent)."""
    if isinstance(obj, dict) and set(obj) == {"value_sha256"} and _is_hex64(obj.get("value_sha256")):
        return obj
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, dict):
        return {k: _hash_leaves_once(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_hash_leaves_once(v) for v in obj]
    return value_sha256(obj)


def redact_meta_doc(doc: dict, full: bool) -> dict:
    """A pinned META_KEEP extract with identifying fields hashed (review fixes CL-5/SC-4): META_LOSSLESS in every meta
    (derive() never reads them), every META_SENSITIVE field when full (a member of a sealed or withheld family)."""
    out = dict(doc)
    for k in (META_SENSITIVE if full else META_LOSSLESS):
        if k in out:
            out[k] = _hash_leaves_once(out[k])
    return out


def redact_snapshot_doc(snap: dict, hashed_items: set) -> tuple[dict, dict]:
    """The registry snapshot with ROW_SENSITIVE (display_name, provenance, ...) removed from the catalog_row of every
    frozen and incoming item (build reads those rows from the pinned catalog v1 and item files, never from the
    snapshot) and hashed in the catalog_row of each sibling in hashed_items (siblings of sealed or withheld families);
    other sibling rows stay (build derives sibling keys from them)."""
    items, counts = [], {"stripped": 0, "hashed": 0, "kept": 0, "no_row": 0}
    for it in snap.get("items") or []:
        it = dict(it)
        payload = dict(it.get("payload") or {})
        row = payload.get("catalog_row")
        if not isinstance(row, dict):
            counts["no_row"] += 1
        elif it.get("epoch_role") in ("frozen", "incoming"):
            payload["catalog_row"] = {k: v for k, v in row.items() if k not in ROW_SENSITIVE}
            it["stripped"] = sorted(k for k in row if k in ROW_SENSITIVE)
            counts["stripped"] += 1
        elif "item:" + str(it.get("id")) in hashed_items:
            payload["catalog_row"] = {k: (_hash_leaves_once(v) if k in ROW_SENSITIVE else v) for k, v in row.items()}
            it["redacted"] = True
            counts["hashed"] += 1
        else:
            counts["kept"] += 1
        it["payload"] = payload
        items.append(it)
    out = dict(snap, items=items, redaction={"rule": "frozen/incoming catalog_row: ROW_SENSITIVE removed; sibling of a "
                                                    "sealed or withheld family: ROW_SENSITIVE hashed",
                                            "row_sensitive": list(ROW_SENSITIVE), "counts": counts})
    return out, counts


def jsonable(obj, where: str = "policy"):
    """Deterministic JSON form of a policy object: tuples -> lists, sets -> sorted lists, bytes -> UTF-8 text,
    compiled regex -> its pattern. Callables and other types STOP (a policy must be data)."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise PinStop(f"{where}: non-finite number")
        return obj
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            raise PinStop(f"{where}: bytes are not UTF-8") from None
    if isinstance(obj, re.Pattern):
        return obj.pattern
    if isinstance(obj, (list, tuple)):
        return [jsonable(v, f"{where}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, (set, frozenset)):
        return sorted((jsonable(v, f"{where}{{}}") for v in obj), key=common.canonical)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise PinStop(f"{where}: non-string key {k!r}")
            out[k] = jsonable(v, f"{where}.{k}")
        return out
    raise PinStop(f"{where}: {type(obj).__name__} is not JSON-representable (a policy must be data)")


# ----------------------------------------------------------------------------------------------- config
@dataclass(frozen=True)
class FixedInput:
    path: Path
    sha256: str
    role: str
    s3: dict | None = None          # the same bytes in S3, as recorded by the survey/PLAN (pin does not read them)
    note: str | None = None


@dataclass(frozen=True)
class Identity:
    """A recorded identity (encoder weights): never read; only stat()ed when a local copy exists."""
    path: Path | None
    sha256: str
    role: str
    s3: dict | None = None
    note: str | None = None


@dataclass(frozen=True)
class PinConfig:
    fixed: dict
    identities: dict
    registry_db: Path
    incoming_items: Path
    frozen_research_root: Path
    dataset_records: Path
    ba_pointers: Path
    ba_source_cache: Path
    tencase_v3_contexts: Path
    seed_view_files: dict
    classic_aliases: dict
    cpu_admission: Path = CPU_ADMISSION
    frozen_research_prefix: str = "research/regression-2k/v1/"
    datasets_bucket: str = DATASETS_BUCKET
    expected_seed_sources: int = 10
    max_gets: int = MAX_GETS
    max_bytes: int = MAX_BYTES
    readback_sample: int = READBACK_SAMPLE
    lineage_b_versions: tuple = LINEAGE_B_VERSIONS


_H = Path("/home/ubuntu")
_SCRATCH = _H / "scratch"
_LB = _H / ("jope-evidence/xbabe2-migration-20260914/accepted60000-restored/parallel/runs/"
            "xbabe2-native-20260914-60000/scientific-evidence")
_LB_S3 = "staging/lineage-b/accepted60000-restored/parallel/runs/xbabe2-native-20260914-60000/scientific-evidence/"
_ASSEMBLY = _H / "jain-split/jope-dime/parallel/data/encodings/contexts-assembly-v2"
_R2K = _H / "JopeDime/.agent/experiments/lambda-training/local_lambda/full_regression/research_2k"
_PFH = _SCRATCH / "jopedime-regression-research-2k-per-fold-hwm-20261001"
_INTAKE = _SCRATCH / "jopedime-regression-research-2k-incoming-intake-v1"
_INV = _SCRATCH / "jopedime-lambda-gp-20260928/inventory"
_BA = _H / "beyondarena-import-20260927"
_TABPFN = _SCRATCH / "tabpfn35-encoder-20260927"
_TABPFN_UNSEEN_ROOT = _SCRATCH / "jopedime-regression-research-2k-tabpfn35-unseen-20261001"   # config only, never argv
_MODELS = _SCRATCH / "jopedime-regression-neural-encoding-v1/models"
_V3 = _SCRATCH / "jopedime-ten-case-test-primary-2cv-v3"
_HYB = _SCRATCH / "jopedime-ten-case-hybrid-v1"


def _lb(sha: str, suffix: str, role: str, version_id: str | None = None) -> FixedInput:
    name = f"{sha}-{suffix}.json"
    s3 = {"bucket": LEARNING_BUCKET, "key": _LB_S3 + name, "version_id": version_id, "recorded_by": "survey"} if version_id else None
    return FixedInput(_LB / name, sha, role, s3)


def default_fixed() -> dict:
    """Every single-file input with its full sha256 (re-verified read-only on 2026-10-05; a mismatch at pin is a STOP)."""
    F = FixedInput
    return {
        # gate, spec and plan
        "epoch": F(RUN / "epochs/epoch-000.json", EPOCH_SHA256, "epoch"),
        "p1_receipt": F(RUN / "receipts/P1/receipt.json", "9784ae6e2e95ea128389a20850ca08e371c229b4bf246777307a1f5231044a64", "gate"),
        "p1_publish": F(RUN / "receipts/P1/publish.json", "a4b8b540d66f982c377576cfcc71601e3e973454f87ea7f6e90f4d9aa513af21", "gate"),
        "owner_decisions": F(_SCRATCH / "jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261005T044116Z.json",
                             "77a115c78e927000e77f052c86c2fde5cc66c149d4a16814172b91426aa3a602", "gate"),
        "plan": F(_SCRATCH / "jopedime-dltrain-audit-v1/claude/PLAN.md",
                  "fb0628c4793f7d7a50a1d7eb98039d15f7132334e5b1fda8cba5ea65fe2c37f5", "plan"),
        "p2_survey": F(RUN / "receipts/P2/survey/wf_a43390a0-83c.output.json",
                       "723e18988e28ae92fa0c3113fd47187814f4325e77e5c1d295b30f0c3008eed8", "spec"),
        # frozen provenance and cross-checks
        "readiness": F(_SCRATCH / "jopedime-next100-gp-audit-v1/readiness-v1/family-readiness.json",
                       "0b27b80e4bbec8526e8432f47d509b8680bd7f3189c4732c4a67176e7a86d233", "frozen_provenance"),
        "catalog_v1": F(_INV / "catalog.jsonl", "ab9fda8d2dea46067b70a42812e9d3c1d9dc6ba025780100df34377e81aa1120", "catalog_v1",
                        {"bucket": DATASETS_BUCKET, "key": "catalog/v1/catalog.jsonl", "version_id": "ZcqyZUbRv6fn1GU8Uenl013vZ0jQ0TDz",
                         "recorded_by": "PLAN"}),
        "catalog_v1_binding": F(_INV / "catalog-binding.json", "61b1ed71d4a0bef9f320200bc05e808e79e9d2694e87c752ac3889a90f4b5c72",
                                "catalog_v1"),
        "legacy_inventory_tsv": F(_H / "super_cluster/dataset_audit_20260908/regression_inventory.tsv",
                                  "40024ba17b4d37ecc0fcec7db4a814469f3c2e89654a38f3370387d71596e544", "legacy_source_family"),
        "expansion_inventory_v7": F(_H / "super_cluster/dataset_audit_20260908/v7/expansion-inventory.json",
                                    "3fcfbfce8c72b8d823ac003b3ae913abe3d7962189b348e2c5c34add2404ed92", "expansion_family"),
        # sealed set (lineage-B; DL-D12) and cross-checks
        "lineage_b_qualification_v1": _lb("f92b1c4aeadee290c957bc77ee5bba675d952800c80851d3ebb1a5d1af612b2d", "qualification-v1",
                                          "sealed_registry", "lrR2Q.6lMvT51TWpf3UA1ww0xnlCNKe5"),
        "lineage_b_qualification_v2": _lb("87fd7ab02feb532869be0be2b8f05e4e71ad2d3e1fe37d3b98891abdb39ecebc", "qualification-v2",
                                          "sealed_registry", "Npza4KUfESIzX37lOKVEMtPeUkvWlzGd"),
        "lineage_b_qualification_v3": _lb("6ae76ce12a0c04e137e1442864da6a917c7734afcf650a70e2259d1af9b992a9", "qualification-v3",
                                          "sealed_registry", "OEvJXfgOXIcMK9O1QgiK81duXKVC_2lw"),
        "lineage_b_qualification_v4": _lb("22b5de5f2f847fb6e086d125d6fec945a1b4edb7c00b94b49e1ad95e7540cfc4", "qualification-v4",
                                          "sealed_registry", "S7VYyHFp7h8gDlO_.24NcXLDM.03RSsl"),
        "lineage_b_qualification_v5": _lb("f22125d892c953bab7b0c74b9855ae39edd33072caa2c69f3aff1420c02eb296",
                                          "qualification-v5-expansion-20260909", "sealed_registry", "CF5R1LcCtD.6MC1tsCzOyuxfzl3MscRs"),
        "lineage_b_partition_freeze": _lb("553b933f592585bc48676e50dd910b99516137b8942fbc043e5f5dad30723b30", "partition-freeze",
                                          "sealed_policy", "U0n_WOHvAKIw5zZqecb1jxgzU.o.x1lC"),
        "lineage_b_family_exclusion_policy": _lb("44f9afb53225fbce5c7e768e58e9ad1ea98111f814e8c462365109681b325660",
                                                 "family-exclusion-policy", "sealed_policy", "arGvrItnI3ViYl.fTzJ9BwWSuee1TzZX"),
        "lineage_b_scientific_policy": _lb("77dd0c3cfeaa37c24f7e0a783a52bbbb4e90b7fe6b56b6a1ea14ad95a41999c4", "scientific-policy",
                                           "sealed_policy_not_used", "t39rjcN42KnbvqGCzhqSpebN7KHO9Ukm"),
        "native_index": F(_ASSEMBLY / "index.json", "6c81da15d75dff6abeaf1d40fb46ec77e98d6146f535392905bfaf1153574b9c",
                          "sealed_crosscheck", note="another repository's git-ignored data; no S3 version; cross-check only"),
        "canonical_exclusions": F(_INV / "canonical-exclusions.jsonl", "44724ee8189f47d74d4cdcb31fb75c221de8ac1da7ec58ae91a5382d22860133",
                                  "sealed_crosscheck"),
        "corpus_audit_v2": F(_SCRATCH / "jopedime-real-regression-audit-20260930/20260930T203328Z/datasets.jsonl.gz",
                             "b8ce4d6c1863656c40fe5e7fafc7669b5c9cfbb996d17208d36d00861cf342ad", "sealed_crosscheck",
                             {"bucket": LEARNING_BUCKET, "key": "experiments/gp-teacher-v6/corpus-audits/v2/objects/inventories/sha256/"
                              "b8ce4d6c1863656c40fe5e7fafc7669b5c9cfbb996d17208d36d00861cf342ad.jsonl.gz",
                              "version_id": "AyU9YPiN2p2CxxWWdJT_IEe.lNjooxqv", "recorded_by": "survey"}),
        "split_registry_native": F(_SCRATCH / "jopedime-real-regression-audit-20260930/split-registry-20260930T210222Z/native-splits.jsonl.gz",
                                   "c307cb607227e6dc1292425b6c2831e292dba8b66120ade01a377c675eeb2405", "sealed_crosscheck",
                                   {"bucket": DATASETS_BUCKET, "key": "split-registry/v1/catalogs/sha256/"
                                    "c307cb607227e6dc1292425b6c2831e292dba8b66120ade01a377c675eeb2405.jsonl.gz",
                                    "version_id": "vSBcAP4_L_0J7OrKuthheaD.H0Ul1tMD", "recorded_by": "survey"}),
        # seeds (SEED192) and their provenance
        "seed_pool_recovery192": F(_PFH / "gp/recovery-seed-pool.jsonl", "a469fb8263f9fb40bedc409808e5548ba3f2eac9859e62cdfc065911d5bd74d3", "seed_pool"),
        "seed_pool_broad96": F(_PFH / "gp/seed-pool.jsonl", "89f6cccc9f08a94ccdfaf6ddc224a41e13bce068470a06566e58bcc21a0d838e", "seed_pool"),
        "seed_pool_supplemental": F(_SCRATCH / "jopedime-regression-research-2k-20261001-recovery1/gp/seed-pool-supplemental.jsonl",
                                    "2d13b3c5d0e26ed593203b3b5f7d16525313c188ed0b3bd2e9272a0a47324228", "seed_pool"),
        "seed_recovery_preparation": F(_SCRATCH / "jopedime-regression-research-2k-20261001-recovery1/gp/recovery-preparation.json",
                                       "24360569462b3aafecce7d5bfc225e96bb39f0f5bfcf21ed5106437ee5f9f87f", "seed_receipt"),
        "seed_pool_summary": F(_SCRATCH / "jopedime-regression-research-2k-20261001/gp/seed-pool-summary.json",
                               "f8961eaf28eb75f4c00b72ea74471de026e04d052e0eb539142903d30636a5f5", "seed_receipt"),
        "seed_script_build_seed_pool": F(_R2K / "build_seed_pool.py", "a36399af78a1b2e7d49575469464c12d0dbb79e7c0af7e26bbb5b9f7081dc81b",
                                         "seed_script"),
        "seed_script_prepare_recovery": F(_R2K / "prepare_recovery.py", "3f83ec436ca9a7bbe3909a4f6a67da955831eeb75aff191d74f238ba3ffcda2e",
                                          "seed_script"),
        "seed_script_test_primary_prepare": F(_R2K.parent / "test_primary_multifidelity/prepare.py",
                                              "0b5d4fb89d30747e01f68ec50b228825534d87883eec74d1e1bf62d041765302", "seed_script"),
        "tencase_v3_input_manifest": F(_V3 / "input-manifest.json", "70228a9d97515299bdb1391662ad25d7c318f47177caf7344f9c550111403cb4",
                                       "seed_campaign"),
        "tencase_v3_campaign_manifest": F(_V3 / "campaign-manifest.json", "042285c0958e94b42aea6bdc790d26f6f47f743c549d194b58c19cc199fac5db",
                                          "seed_campaign"),
        "tencase_v3_source_manifest": F(_V3 / "source-manifest.json", "ac47e9257f7d6d1e24830b42fb2aa657a503ba271102e9987b084452e448f5e1",
                                        "seed_campaign"),
        "tencase_hybrid_input_manifest": F(_HYB / "input-manifest.json", "891902e5eed3b0fa99f211a72328a7c6525504bc2fd2e3a8b98364132de717dc",
                                           "seed_campaign",
                                           {"bucket": LEARNING_BUCKET, "key": "experiments/gp-teacher-v6/objects/ten-case-input-manifests/sha256/"
                                            "891902e5eed3b0fa99f211a72328a7c6525504bc2fd2e3a8b98364132de717dc.json",
                                            "version_id": "_bwhObbmAbtRbgVPnNfOQ5eXFrqKlDef", "recorded_by": "survey (not HEAD-checked)"}),
        "tencase_hybrid_campaign_manifest": F(_HYB / "campaign-manifest.json", "eb89738e57bd485da8a56530a2565620002e0808f049a1d8145cc0da7576e8ee",
                                              "seed_campaign"),
        "tencase_hybrid_input_pointer": F(_HYB / "input-pointer.json", "0f89cb11e3ad106e606d4f0677a4df9d83a6bede470a35070ced6f76fe65554e",
                                          "seed_campaign"),
        # BeyondArena and TabPFN 3.5 (external alias annotations)
        "beyondarena_pointer_receipt": F(_BA / "pointer-receipt.json", "830542f8037c42f9e529fd2ddbe5da5d6ed73a74cb7f19cc36f71d8bfc2be5d1",
                                         "beyondarena"),
        "beyondarena_index": F(_BA / "pointers/index.json", "b8b2268d220f50436fa710171fab519c8d5c10e5383b5b89e02e2ef0dfbdf7e1", "beyondarena"),
        "beyondarena_overlap_candidates": F(_BA / "source-overlap-candidates.json",
                                            "7076cb2cb09d4178de31880444680a91381d1ec932e42ce53fe19eea63018aa3", "beyondarena",
                                            note="name-only, incomplete; annotation"),
        "tabpfn35_encoder_spec": F(_TABPFN / "encoder-spec.json", "68caa596e5feef86b779bda2a59c7b31afcbde6bc1d6226ac3dd8bc4ff722bdc",
                                   "tabpfn35"),
        "tabpfn35_coverage_summary": F(_TABPFN / "coverage-summary.json", "9c93bcd54158c99705d2702b60a2d67df2fa960edb18ac30b74fc06d24522cbd",
                                       "tabpfn35"),
        "tabpfn35_study_manifest": F(_TABPFN / "study-manifest.json", "fd55eccc97973a6ef37bc3d5f427688c344b147ea7c84958bf6ca3e46289f963",
                                     "tabpfn35"),
        "tabpfn35_unseen_audit": F(_TABPFN_UNSEEN_ROOT / "s3-audit-final.json",
                                   "170281079d38f5cef58382b1fdf3b29e287cb0b5b0f43aa62e5434c1a969a2fc", "tabpfn35_unseen",
                                   note="TabPFN-unseen root from this config only, never argv"),
        # encoder and normalizer fit sets (exposure)
        "lineage_a_split_manifest": F(_H / "jope/data/processed/active_v3_dual_20260706/split_manifest.json",
                                      "3213496d09d058a4f0e57c7061c1792a441cda17fee46fd765bdf66c13af9062", "exposure_fit_set"),
        "lineage_a_pca": F(_H / "jope/data/processed/active_v3_dual_20260706/foundation_projection.npz",
                           "3df4615971e7974b09f91d6ebcf22672e02b4900d4f71a18616dae8dc37b8ff9", "encoder_normalizer"),
        "lineage_a_standardizer": F(_H / "jope/data/processed/active_v3_dual_20260706/standardization.json",
                                    "6c07305fc1de841ea387fe5b01d0074b9ba386d9d15089fe2a3c10acdf623145", "encoder_normalizer"),
        "lineage_b_assembly": F(_ASSEMBLY / "assembly.json", "6582df23f5c1ee048d11040a5f2bd1447032932fc23f6c5a3aa9cb4f2d6896cb",
                                "exposure_fit_set",
                                {"bucket": LEARNING_BUCKET, "key": "staging/lineage-b/accepted60000-restored/parallel/data/encodings/"
                                 "contexts-assembly-v2/assembly.json", "version_id": "AQnRv4kn5IFQ_FF17lk4320gLZ8ev8P9",
                                 "recorded_by": "source-evidence"}),
        "lineage_b_registry_members": F(_ASSEMBLY / "registry-members.json",
                                        "5e6dd030090aa6ee06aa81def53e041772bba53fa32db5aca57716f6fa376181", "exposure_fit_set"),
        "lineage_b_transforms_ot": F(_ASSEMBLY / "transforms.ot", "eb4dce65c7fe3a72f13700ce7667183745b531744fe59229718f0c0458d834b6",
                                     "encoder_normalizer"),
        "lineage_b_context_transforms": F(_MODELS / "context-transforms", "1b779c6ff7a1a812e3001342d663353d3c7ff54eb93392a600f01813f4c66e22",
                                          "encoder_normalizer"),
        "target_metadata": F(_MODELS / "target_encoder_fused_v1_metadata", "3849615a7c8f9e750d543eb07e02eceb167e648775fe1dd5c041f8cdecfe7043",
                             "exposure_fit_set",
                             {"bucket": MODELS_BUCKET, "key": "blobs/sha256/3849615a7c8f9e750d543eb07e02eceb167e648775fe1dd5c041f8cdecfe7043",
                              "version_id": "q46RFzbRy2Jk7PKe27PCoS7bk0G5.KwJ", "recorded_by": "source-evidence"}),
        "target_data_py": F(_H / "target_maker/target_maker/data.py", "e20c00a8c2ce5894e2ad7dab9b48e05c353a4736aba471e6d481b7c0931bf0fd",
                            "exposure_rule"),
        # copied rules (never imported)
        "rule_family_protocol": F(_R2K / "per_fold_hwm/worst50_v1/next100_v1/family_protocol.py",
                                  "20381727b39035b73dde59c2058ec44005f27fa1e72d7c0ea90694cf65738026", "copied_rule"),
        "rule_intake_contract": F(_R2K / "per_fold_hwm/incoming_intake_v1/intake_contract.py",
                                  "5f360064a61bf39fc024c058d753cf3619e9c35ac4b9586f2b92fe90ec87d6d8", "copied_rule"),
        "rule_content_identity": F(_H / "hyper/data/real_world_additions/tools/content_identity.py",
                                   "f2aa3c4c33a2f9f4db97c131eebf1ec451e0df278a16af8ce7e9db0bec2c0c0a", "copied_rule"),
    }


def default_identities() -> dict:
    """Encoder weight identities: recorded for encoder_weight_exposure, never read by P2."""
    blob = lambda sha, vid: {"bucket": MODELS_BUCKET, "key": f"blobs/sha256/{sha}", "version_id": vid, "recorded_by": "survey"}  # noqa: E731
    return {
        "encoder:hyperion_v5_arm_b": Identity(_MODELS / "hyperion_v5_arm_b_identity",
                                              "88b9e8345e44d3ea609a8286215618a762f14e607fcdada6d27447b1f74668be", "encoder_weights",
                                              blob("88b9e8345e44d3ea609a8286215618a762f14e607fcdada6d27447b1f74668be",
                                                   "6_zLiTSpAypF_ed34kqL2d0992lurWxO")),
        "encoder:starlight": Identity(_MODELS / "starlight_core_regressor_identity",
                                      "f663ae49cc0f1504f07800a69dd6c31ca22fc473f26bbc0684eac25129a04128", "encoder_weights",
                                      blob("f663ae49cc0f1504f07800a69dd6c31ca22fc473f26bbc0684eac25129a04128",
                                           "wU5wqKRNL1IRgtBuS.9JnqnN9eVpHZHm")),
        "encoder:target_fused_v1": Identity(_MODELS / "target_encoder_fused_v1_identity",
                                            "be96adc6549a478c6a1759e40ce8c403fa529694b54a74a7d93a6dd56d35a40f", "encoder_weights",
                                            blob("be96adc6549a478c6a1759e40ce8c403fa529694b54a74a7d93a6dd56d35a40f",
                                                 "XTTiITZb5MZVhsBmUVv1mEMyPQxfWjYe"),
                                            note="must equal target_metadata.artifact_sha256"),
        "encoder:tabpfn35": Identity(_TABPFN / "models/tabpfn-v3.5-20260909.safetensors",
                                     "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3", "encoder_weights",
                                     note="must equal tabpfn35_encoder_spec.checkpoint_sha256; base weights unmodified"),
    }


SEED_VIEW_FILES = {   # ten-case v3 contexts/<name>__r0__f0/{bootstrap-candidates.jsonl, prepared.json}
    "airfoil_self_noise": ("95725bff1853a5683b6d3ac08232d16f5f482022393519fb72990de2273f65c6",
                           "829a37971a1950952b0615b5307be4fa5aafcdefa4189a5073f0050a5b366695"),
    "california_house_prices_2020": ("94119e995884e64cda8d37dd63e2042d61d08ef0bbd829b1bc39f42e24ce7033",
                                     "94d330569514d8ad4b91455ca33f8827e83c5e0469c1a9ffab9afd1c86d09308"),
    "concrete_compressive_strength": ("fa370f5594354f1cc84e2f1412e69002cb6bc8ccb770a04fe036f72dbf078797",
                                      "ec7cc008754a9143803e8e479f5792d10a0e9ef907b66b9892ec21aa1cf842ae"),
    "diamonds": ("ee45432b15f31a737471ab60eba8d058195ba82baf86cfc6ec9769649d3bfef0",
                 "3ed2c4c594b733369c0a2b4126ab950f82e4788448f1094534a0b9ef5437e251"),
    "houses": ("401c2fcdd4aeaa94e9558e1d137c823af37d72bcfb44e09510e3125e1770b932",
               "896e8f2695a8b09a08b85abbf4fbc905c7e9fdc327f7ffd6ae877825cdab8e70"),
    "immoscout_german_house_prices": ("4d748abc0d5ff8b28860eec346da687eeeda94c9a394ee98151f4d3c0ab1aebc",
                                      "7a54b90ea4f53b843d24d6459a77522c670e3d3257b9a599d57498f4dd387117"),
    "miami_housing": ("4e3c391493505da52af1a4504868d6605516d23dcbff4f1e8939b97bfe88a987",
                      "d6d7f797137b54c03f5a113133f519296d704f9ad64dbe16f5ad934a2d860f3d"),
    "superconductivity": ("cfb138a9b47c65b9fa0885946ec590bcdeb21d52dca1913e0d1b89f802e83f2c",
                          "4e876fd8caf8578120921790d3daf8134569f16ddf9a1318e449b2b508421e5f"),
    "wine_quality": ("bfd3311108b54d65aaadcf5624cbb0b59f75c7d5f79be1a8056f4c8c3e67ae81",
                     "6a8cd280ec8cd6d36f1a355bde8a96b8b08e345d8a33bba7de498d764f4a77df"),
    "wine_world_cost": ("f5896c64a97a992e683afcfff6348a408e44252aa0b8edf982fe403e17d5a220",
                        "86e2d1c44d04fba80b7cc0c2995b8fe18540f37696a4d80014bd6fb91ffa5f1e"),
}

CLASSIC_ALIASES = {
    "schema": CLASSIC_ALIASES_SCHEMA,
    "provenance": {"drafted_by": "agent (P2 pin builder) from the P2 survey wf_a43390a0-83c (seed-exposure and near-duplicate-screen areas)",
                   "review": "owner (DL-D5 reviewer = owner) through review.csv; a union entry is applied only when the P2 value screen "
                             "confirms a strong or content link to every listed member, else STOP"},
    "entries": [
        {"name": "concrete_compressive_strength", "action": "union", "members": ["485079e140459631"], "seed_source": True,
         "evidence": "9 column multisets match after min-max within 0.0016; 4-column row join 830/1030, column-shuffled null 1"},
        {"name": "diamonds", "action": "union", "members": ["525fc7ef6e7d6d64", "ee74ac8192a9f455"], "seed_source": True,
         "evidence": "6-column row join 1999/1999 for each member, null 0"},
        {"name": "houses", "action": "union", "members": ["0a8428ef25a69175", "5f91a00bd3b66742"], "seed_source": True,
         "evidence": "4-column row join 1837/1999 and 1900/1999, null 0 (965 capped rows dropped upstream)"},
        {"name": "saratoga_houses", "action": "review", "members": ["6a3edcd68b228f1f"], "related_seed_sources": ["houses"],
         "note": "name-only near match; not an alias"},
        {"name": "ames_house_prices", "action": "review", "members": ["5978a22875af98cd", "6614606fc60bbbfc", "ab721eb8fcf22f1f"],
         "related_seed_sources": ["houses", "california_house_prices_2020"], "note": "name-only near match; not an alias"},
        {"name": "pmlb_218_house_8L", "action": "review", "members": ["80589368fa78f21a"], "related_seed_sources": ["houses"],
         "note": "name-only near match; not an alias"},
        {"name": "pmlb_574_house_16H", "action": "review", "members": ["ab279eded3b3c52b"], "related_seed_sources": ["houses"],
         "note": "name-only near match; not an alias"},
        {"name": "recreate_concrete", "action": "review", "members": ["2e1853094bdea505", "8366db9ae9698880"],
         "related_seed_sources": ["concrete_compressive_strength"], "note": "different measurements (UPV/RH); not an alias"},
        {"name": "ttb_wine", "action": "review", "members": ["0cca94ee62ec093d"], "related_seed_sources": ["wine_quality", "wine_world_cost"],
         "note": "name-only near match; not an alias"},
        {"name": "california_house_prices_2020_vs_1990", "action": "review", "members": ["5f91a00bd3b66742"],
         "related_seed_sources": ["california_house_prices_2020"], "note": "different data (2020-21 sales vs 1990 census); not an alias"},
    ] + [{"name": f"keyword:{k}", "action": "review", "keyword": k, "members": []}
         for k in ("boston", "abalone", "wine_quality", "bike_sharing", "ccpp", "insurance", "flights", "taxi")],
    "uci_legacy_folder_map": {"status": "UNKNOWN", "map": {},
                              "note": "the 5 legacy /ml/machine-learning-databases/<folder>/ paths have no verified uci:<id>; "
                                      "build keeps uci_legacy_folder keys as-is"},
}


def default_config() -> PinConfig:
    return PinConfig(fixed=default_fixed(), identities=default_identities(), registry_db=_INTAKE / "intake.sqlite3",
                     incoming_items=_INTAKE / "items", frozen_research_root=_SCRATCH / "jopedime-regression-research-2k-20261001",
                     dataset_records=_INV / "dataset-records", ba_pointers=_BA / "pointers/sources",
                     ba_source_cache=_SCRATCH / "jopedime-gp-teacher-v6-20260930/source-cache",
                     tencase_v3_contexts=_V3 / "contexts", seed_view_files=dict(SEED_VIEW_FILES), classic_aliases=CLASSIC_ALIASES)


# ----------------------------------------------------------------------------------------------- CPU guard and ledger
class CpuGuard:
    """Pause while cpu-admission.json is older than max_age_s, admit is not true, or the node's cpu_rolling_percent
    >= max_cpu (an unreadable file counts as not admitted). STOP after max_wait_s of continuous pause."""

    def __init__(self, path=CPU_ADMISSION, node: str = "xbabe2", max_age_s: float = 45.0, max_cpu: float = 75.0,
                 poll_s: float = 15.0, max_wait_s: float = 3600.0, min_interval_s: float = 5.0, clock=time.time, sleep=time.sleep):
        self.path, self.node, self.max_age_s, self.max_cpu = Path(path), node, max_age_s, max_cpu
        self.poll_s, self.max_wait_s, self.min_interval_s, self.clock, self.sleep = poll_s, max_wait_s, min_interval_s, clock, sleep
        self.checks, self.pauses, self.last_ok = 0, [], None

    def state(self) -> tuple[bool, str]:
        try:
            doc = json.loads(self.path.read_bytes())
            ts = doc.get("sampled_unix_seconds")
            if not isinstance(ts, (int, float)) or isinstance(ts, bool):
                ts = self.path.stat().st_mtime
        except (OSError, ValueError, AttributeError) as err:
            return False, f"admission file unreadable ({type(err).__name__})"
        if not isinstance(doc, dict):
            return False, "admission file is not an object"
        age = self.clock() - ts
        if age > self.max_age_s:
            return False, f"admission sample is {age:.0f}s old (> {self.max_age_s:.0f}s)"
        if doc.get("admit") is not True:
            return False, "admit is not true"
        cpu = ((doc.get("nodes") or {}).get(self.node) or {}).get("cpu_rolling_percent")
        if not isinstance(cpu, (int, float)) or isinstance(cpu, bool):
            return False, f"nodes.{self.node}.cpu_rolling_percent missing"
        if cpu >= self.max_cpu:
            return False, f"nodes.{self.node}.cpu_rolling_percent {cpu:.1f} >= {self.max_cpu:.0f}"
        return True, "admitted"

    def wait(self, stage: str, force: bool = False) -> None:
        now = self.clock()
        if not force and self.last_ok is not None and now - self.last_ok < self.min_interval_s:
            return
        start, first = now, None
        while True:
            self.checks += 1
            ok, why = self.state()
            if ok:
                self.last_ok = self.clock()
                if first is not None:
                    self.pauses.append({"stage": stage, "reason": first, "paused_s": round(self.last_ok - start, 3)})
                return
            first = first or why
            if self.clock() - start >= self.max_wait_s:
                raise PinStop(f"CPU guard: not admitted for {self.max_wait_s:.0f}s at stage {stage} ({why})")
            self.sleep(self.poll_s)

    def summary(self) -> dict:
        return {"path": str(self.path), "node": self.node, "max_age_s": self.max_age_s, "max_cpu_percent": self.max_cpu,
                "checks": self.checks, "pauses": len(self.pauses), "paused_s": round(sum(p["paused_s"] for p in self.pauses), 3),
                "pause_log": self.pauses[:200]}


class NoGuard:
    """Offline tests only."""
    checks, pauses = 0, []

    def wait(self, stage: str, force: bool = False) -> None:
        self.checks += 1

    def summary(self) -> dict:
        return {"guard": "none (offline)", "checks": self.checks, "pauses": 0}


class Ledger:
    """Every S3 request: {seq, op, mode, role, bucket, key, requested_version_id, version_id, bytes, sha256, status,
    utc_us, done_utc_us}. With a path, each entry is appended to a create-only JSONL journal and fsynced before the
    next request, so a STOP or crash still accounts for every GET. reserve() refuses a GET that would exceed a cap."""

    def __init__(self, path=None, max_gets: int = MAX_GETS, max_bytes: int = MAX_BYTES):
        self.entries, self.max_gets, self.max_bytes = [], max_gets, max_bytes
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            os.close(os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644))

    @property
    def gets(self) -> int:
        return len(self.entries)

    @property
    def bytes(self) -> int:
        return sum(e.get("bytes") or 0 for e in self.entries)

    def reserve(self, expected_bytes) -> None:
        if self.gets + 1 > self.max_gets:
            raise PinStop(f"S3 budget: GET {self.gets + 1} would exceed {self.max_gets}")
        if self.bytes + int(expected_bytes or 0) > self.max_bytes:
            raise PinStop(f"S3 budget: {self.bytes} + {int(expected_bytes or 0)} B would exceed {self.max_bytes} B")

    def record(self, entry: dict) -> dict:
        entry = dict(entry, seq=len(self.entries) + 1)
        self.entries.append(entry)
        if self.path:
            common.assert_no_secrets_obj(entry)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
        return entry

    def totals(self) -> dict:
        by_role = {}
        for e in self.entries:
            r = by_role.setdefault(e["role"], {"gets": 0, "bytes": 0, "ok": 0})
            r["gets"] += 1
            r["bytes"] += e.get("bytes") or 0
            r["ok"] += e.get("status") == "ok"
        return {"gets": self.gets, "bytes": self.bytes, "ok": sum(e.get("status") == "ok" for e in self.entries),
                "failed": sum(e.get("status") != "ok" for e in self.entries), "by_role": by_role,
                "limits": {"gets": self.max_gets, "bytes": self.max_bytes}, "journal": str(self.path) if self.path else None}


def s3_get(item: dict, clients: dict, ledger: Ledger) -> bytes:
    """One guarded GET through dltrain.s3io: 'exact' = get_exact(VersionId) with a lane read client,
    'content_addressed' = get_content_addressed with a cas-read client (D-P2-1). The body must hash to the pinned
    sha256, have the pinned length and (exact) come back at the requested VersionId; anything else is a STOP."""
    from botocore.exceptions import BotoCoreError, ClientError

    from dltrain import s3io
    ledger.reserve(item["bytes"])
    entry = {"op": "GetObject", "mode": item["mode"], "role": item["role"], "bucket": item["bucket"], "key": item["key"],
             "requested_version_id": item.get("version_id"), "expected_sha256": item["sha256"], "expected_bytes": item["bytes"],
             "version_id": None, "bytes": None, "sha256": None, "utc_us": common.utc_us()}
    label = f"{item['role']} s3://{item['bucket']}/{item['key']}"
    try:
        if item["mode"] == "exact":
            body, meta = s3io.get_exact(clients["exact"], item["key"], item["version_id"], bucket=item["bucket"])
        elif item["mode"] == "content_addressed":
            body, meta = s3io.get_content_addressed(clients["cas"], item["bucket"], item["key"])
        else:
            raise PinStop(f"unknown read mode {item['mode']!r}")
    except s3io.ReadbackMismatch:
        ledger.record(entry | {"status": "sha_mismatch", "done_utc_us": common.utc_us()})
        raise PinStop(f"{label}: body sha256 does not match the content-addressed key") from None
    except s3io.S3Refused as err:
        ledger.record(entry | {"status": "refused", "done_utc_us": common.utc_us()})
        raise PinStop(f"{label}: guard refused ({str(err)[:200]})") from None
    except ClientError as err:
        code = str(err.response.get("Error", {}).get("Code") or "")
        status = err.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        ledger.record(entry | {"status": f"error:{code}:{status}", "done_utc_us": common.utc_us()})
        raise PinStop(f"{label}: S3 error {code} ({status})") from None
    except BotoCoreError as err:
        ledger.record(entry | {"status": f"error:{type(err).__name__}", "done_utc_us": common.utc_us()})
        raise PinStop(f"{label}: transport error {type(err).__name__} (outcome recorded; the request may have been served)") from None
    except PinStop:
        raise
    except BaseException as err:
        ledger.record(entry | {"status": f"unknown:{type(err).__name__}", "done_utc_us": common.utc_us()})
        raise
    got = common.sha256_bytes(body)
    entry.update(version_id=meta.get("VersionId"), etag=meta.get("ETag"), bytes=len(body), sha256=got, done_utc_us=common.utc_us())
    problems = []
    if got != item["sha256"]:
        problems.append("sha256")
    if len(body) != item["bytes"]:
        problems.append("bytes")
    if item["mode"] == "exact" and meta.get("VersionId") != item["version_id"]:
        problems.append("version_id")
    if item["mode"] == "content_addressed" and not meta.get("VersionId"):
        problems.append("no_version_id")
    entry["status"] = "ok" if not problems else "mismatch:" + ",".join(problems)
    ledger.record(entry)
    if problems:
        raise PinStop(f"{label}: {', '.join(problems)} mismatch")
    return body


def default_clients() -> dict:
    """Live clients (RUN phase only): a lane read client for exact-version GETs and a cas-read client (D-P2-1)."""
    from dltrain import s3io
    return {"exact": s3io.make_client(), "cas": s3io.make_client(mode="cas-read")}


# ----------------------------------------------------------------------------------------------- helpers
def _read(path, what: str) -> bytes:
    p = Path(path)
    if not p.is_file():
        raise PinStop(f"{what}: missing file {p}")
    return p.read_bytes()


def _json(path, what: str):
    raw = _read(path, what)
    try:
        return json.loads(raw)
    except ValueError:
        raise PinStop(f"{what}: {path} is not JSON") from None


def _pin_file(name: str, role: str, path, sha256: str | None = None, nbytes: int | None = None, **extra) -> dict:
    """Re-hash a local file (streaming). With sha256/nbytes given, a mismatch is a STOP."""
    p = Path(path)
    if not p.is_file():
        raise PinStop(f"{name}: missing file {p}")
    got, size = common.sha256_file(p), p.stat().st_size
    if sha256 is not None and got != sha256:
        raise PinStop(f"{name}: sha256 {got} != pinned {sha256} ({p})")
    if nbytes is not None and size != nbytes:
        raise PinStop(f"{name}: {size} B != pinned {nbytes} B ({p})")
    return {"name": name, "role": role, "path": str(p), "sha256": got, "bytes": size} | {k: v for k, v in extra.items() if v is not None}


def _s3ref(ref: dict | None) -> dict | None:
    if not isinstance(ref, dict):
        return None
    return {k: ref.get(k) for k in ("bucket", "key", "version_id") if ref.get(k) is not None} or None


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise PinStop(msg)


def _is_hex64(v) -> bool:
    return isinstance(v, str) and bool(HEX64.fullmatch(v))


def _local_from_key(root: Path, prefix: str, key: str, what: str) -> Path:
    _need(isinstance(key, str) and key.startswith(prefix) and ".." not in key.split("/"), f"{what}: key {key!r} is not under {prefix}")
    return Path(root) / key[len(prefix):]


def _loads(body: bytes, what: str):
    try:
        return json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise PinStop(f"{what}: body is not JSON") from None


def _set_sha256(values) -> str:
    return common.sha256_bytes(common.canonical(sorted(values)))


# ----------------------------------------------------------------------------------------------- step 0/1: fixed, gate, epoch
def pin_fixed(cfg: PinConfig, guard) -> dict:
    out = {}
    for name in sorted(cfg.fixed):
        fi = cfg.fixed[name]
        guard.wait(f"fixed:{name}")
        out[name] = _pin_file(name, fi.role, fi.path, fi.sha256, s3=fi.s3, note=fi.note)
    for name in ("epoch", "p1_receipt", "p1_publish", "owner_decisions", "readiness", "catalog_v1", "seed_pool_recovery192",
                 "lineage_b_family_exclusion_policy", "beyondarena_pointer_receipt", "beyondarena_index", "tabpfn35_encoder_spec",
                 "target_metadata") + tuple(cfg.lineage_b_versions):
        _need(name in out, f"config lacks the fixed input {name}")
    return out


def check_gate(cfg: PinConfig) -> dict:
    """PLAN P2 gate: P1 PASS on this epoch, its S3 epoch and binding refs, DL-D5 and DL-D12 recorded."""
    epoch_sha = cfg.fixed["epoch"].sha256
    p1 = _json(cfg.fixed["p1_receipt"].path, "P1 receipt")
    _need(p1.get("verdict") == "PASS", "P1 receipt verdict is not PASS")
    _need(((p1.get("outputs") or {}).get("epoch") or {}).get("sha256") == epoch_sha, "P1 receipt epoch sha256 differs from the pinned epoch")
    refs = (_json(cfg.fixed["p1_publish"].path, "P1 publish receipt").get("refs") or {})
    e, b = refs.get("epoch") or {}, refs.get("binding") or {}
    _need(e.get("sha256") == epoch_sha and str(e.get("key", "")).endswith(f"/{epoch_sha}.json") and e.get("version_id"),
          "P1 publish receipt does not carry the epoch's S3 ref")
    _need(bool(b.get("key") and b.get("version_id") and _is_hex64(b.get("sha256"))), "P1 publish receipt lacks the binding ref")
    od = _json(cfg.fixed["owner_decisions"].path, "owner decisions")
    dec = od.get("decisions") or {}
    _need("DL-D5" in dec and "DL-D12" in dec, "owner decisions lack DL-D5 or DL-D12")
    keep = ("bucket", "key", "version_id", "sha256", "bytes")
    return {"p1_verdict": "PASS", "epoch_s3": {k: e.get(k) for k in keep}, "binding_s3": {k: b.get(k) for k in keep},
            "owner_decisions_sha256": cfg.fixed["owner_decisions"].sha256, "decisions_cited": ["DL-D2", "DL-D5", "DL-D7", "DL-D12"]}


def load_epoch(path, expected_sha256: str) -> dict:
    raw = _read(path, "epoch")
    got = common.sha256_bytes(raw)
    _need(got == expected_sha256, f"epoch sha256 {got} != {expected_sha256}")
    doc = json.loads(raw)
    rows = doc.get("contexts")
    _need(isinstance(rows, list) and rows, "epoch has no contexts")
    seen, cohort_of, item_dataset = set(), {}, {}
    for r in rows:
        c, h, ctx = r.get("cohort"), r.get("dataset_hash"), r.get("context_sha256")
        _need(c in ("frozen", "incoming"), f"epoch row {ctx}: cohort {c!r}")
        _need(isinstance(h, str) and bool(HEX16.fullmatch(h)), f"epoch row {ctx}: dataset_hash {h!r}")
        _need(_is_hex64(ctx) and ctx not in seen, f"epoch row: context_sha256 {ctx!r} missing or duplicated")
        seen.add(ctx)
        _need(cohort_of.setdefault(h, c) == c, f"dataset {h} appears in both cohorts")
        if c == "incoming":
            i = r.get("item_id")
            _need(_is_hex64(i), f"incoming row {ctx} lacks item_id")
            _need(item_dataset.setdefault(i, h) == h, f"item {i} maps to two datasets")
        else:
            _need(not r.get("item_id"), f"frozen row {ctx} carries an item_id")
        _need(isinstance(r.get("prepared_ref"), dict), f"epoch row {ctx} lacks prepared_ref")
    E = set(cohort_of)
    return {"path": str(path), "sha256": got, "bytes": len(raw), "rows": rows, "E": E, "ITEMS": set(item_dataset),
            "cohort_of": cohort_of, "item_dataset": item_dataset,
            "frozen": {h for h, c in cohort_of.items() if c == "frozen"}, "incoming": {h for h, c in cohort_of.items() if c == "incoming"},
            "experiment_id": doc.get("experiment_id"), "epoch": doc.get("epoch"), "created_utc_us": doc.get("created_utc_us")}


def check_prepared(ep: dict, guard) -> dict:
    """Re-hash every row's prepared.json against prepared_ref; one source manifest per frozen dataset / incoming item."""
    frozen_sm, item_sm, digest = {}, {}, []
    included = 0
    for n, r in enumerate(sorted(ep["rows"], key=lambda x: x["context_sha256"])):
        if n % 500 == 0:
            guard.wait("prepared")
        ref, ctx = r["prepared_ref"], r["context_sha256"]
        raw = _read(ref.get("path"), f"prepared.json of {ctx}")
        _need(common.sha256_bytes(raw) == ref.get("sha256") and len(raw) == ref.get("bytes"), f"prepared.json of {ctx}: sha256/bytes differ from the epoch")
        d = json.loads(raw)
        c = d.get("context") or {}
        _need(c.get("context_sha256") == ctx and c.get("dataset_hash") == r["dataset_hash"], f"prepared.json of {ctx} disagrees with the epoch row")
        raw_sha = (r.get("input_ref") or {}).get("raw_sha256")
        _need(raw_sha is None or c.get("input_raw_sha256") == raw_sha, f"prepared.json of {ctx}: input_raw_sha256 differs from input_ref")
        sm = d.get("source_manifest")
        _need(isinstance(sm, dict) and isinstance(sm.get("key"), str) and _is_hex64(sm.get("sha256")), f"prepared.json of {ctx} lacks source_manifest")
        sm = {k: sm.get(k) for k in ("bucket", "key", "version_id", "sha256", "bytes")}
        if r["cohort"] == "frozen":
            _need(frozen_sm.setdefault(r["dataset_hash"], sm) == sm, f"frozen dataset {r['dataset_hash']} has two source manifests")
        else:
            _need(item_sm.setdefault(r["item_id"], sm) == sm, f"incoming item {r['item_id']} has two source manifests")
        included += r.get("status") == "included"
        digest.append([ctx, ref["sha256"], ref["bytes"]])
    return {"frozen_source_manifest": frozen_sm, "incoming_source_manifest": item_sm, "files": len(digest), "included_rows": included,
            "set_sha256": _set_sha256(digest), "bytes": sum(x[2] for x in digest)}


# ----------------------------------------------------------------------------------------------- step 2: registry
def connect_ro(db_path) -> sqlite3.Connection:
    """sqlite3 'file:<path>?mode=ro' (URI) + PRAGMA query_only=1; autocommit so the caller owns BEGIN..COMMIT.
    A write through this connection raises sqlite3.OperationalError."""
    p = Path(db_path)
    _need(p.is_file(), f"registry database missing: {p}")
    con = sqlite3.connect(f"file:{quote(str(p))}?mode=ro", uri=True, isolation_level=None)
    con.execute("PRAGMA query_only=1")
    if con.execute("PRAGMA query_only").fetchone()[0] != 1:
        con.close()
        raise PinStop("PRAGMA query_only did not take effect")
    return con


def registry_snapshot(db_path, datasets: set) -> dict:
    """One BEGIN..COMMIT read: max(polls.seq), the items whose dataset is in `datasets`, and their contexts."""
    con = connect_ro(db_path)
    try:
        con.execute("BEGIN")
        poll_seq = con.execute("select max(seq) from polls").fetchone()[0]
        rows = con.execute("select id, dataset, canonical, sample, state, reason, admitted_ns, retired_ns, payload from items").fetchall()
        items = [r for r in rows if r[1] in datasets]
        ids = {r[0] for r in items}
        contexts = sorted([c, i] for c, i in con.execute("select context, item_id from contexts") if i in ids)
        con.execute("COMMIT")
    finally:
        con.close()
    out = []
    for r in sorted(items):
        try:
            payload = json.loads(r[8]) if r[8] else {}
        except ValueError:
            raise PinStop(f"registry item {r[0]}: payload is not JSON") from None
        _need(isinstance(payload, dict), f"registry item {r[0]}: payload is not an object")
        out.append({"id": r[0], "dataset": r[1], "canonical": r[2], "sample": r[3], "state": r[4], "reason": r[5],
                    "admitted_ns": r[6], "retired_ns": r[7], "payload_full": payload})
    return {"db_path": str(db_path), "poll_seq": poll_seq, "utc_us": common.utc_us(), "items": out, "contexts": contexts,
            "items_total_in_db": len(rows)}


def check_registry(snap: dict, ep: dict, prep: dict) -> dict:
    by_id, frozen_items = {}, {}
    for it in snap["items"]:
        i, h, can, s = it["id"], it["dataset"], it["canonical"], it["sample"]
        _need(all(isinstance(x, str) for x in (i, h, can, s)), f"registry item {i}: non-text identity field")
        _need(i == identity_key(h, can, s), f"registry item {i}: id != sha256(canonical([dataset, canonical, sample]))")
        _need(h == can[:16], f"registry item {i}: dataset != canonical[:16]")
        by_id[i] = it
        if it["state"] == "frozen":
            frozen_items.setdefault(h, []).append(it)
    frozen_item = {}
    for h in sorted(ep["frozen"]):
        got = frozen_items.get(h, [])
        _need(len(got) == 1, f"frozen dataset {h}: {len(got)} registry items in state 'frozen' (need exactly 1)")
        sm = prep["frozen_source_manifest"].get(h)
        _need(sm is not None, f"frozen dataset {h}: no source manifest in its prepared.json")
        _need(got[0]["sample"] == sm["key"].split("/")[-2], f"frozen dataset {h}: registry sample != source_manifest sample")
        frozen_item[h] = got[0]
    for r in ep["rows"]:
        i = r.get("item_id")
        if not i:
            continue
        it = by_id.get(i)
        _need(it is not None, f"epoch item {i} is not in the registry snapshot")
        _need(it["dataset"] == r["dataset_hash"] and it["sample"] == r.get("sample_identity_sha256"),
              f"epoch item {i}: registry dataset/sample differ from the epoch row")
    epoch_ids = set(ep["ITEMS"]) | {it["id"] for it in frozen_item.values()}
    siblings = sorted(i for i in by_id if i not in epoch_ids)
    states = {}
    for it in snap["items"]:
        states[it["state"]] = states.get(it["state"], 0) + 1
    return {"by_id": by_id, "frozen_item": frozen_item, "siblings": siblings, "states": states}


def snapshot_doc(snap: dict, ep: dict, reg: dict, redact_datasets: set) -> dict:
    items, frozen_ids = [], {x["id"] for x in reg["frozen_item"].values()}
    for it in snap["items"]:
        p = it["payload_full"]
        payload = {k: p[k] for k in PAYLOAD_KEEP if k in p}
        rec = {k: it[k] for k in ("id", "dataset", "canonical", "sample", "state", "reason", "admitted_ns", "retired_ns")}
        role = "frozen" if it["id"] in frozen_ids else ("incoming" if it["id"] in ep["ITEMS"] else "sibling")
        rec["epoch_role"] = role
        if it["dataset"] in redact_datasets and isinstance(payload.get("catalog_row"), dict):
            payload["catalog_row"] = catalog_row_redacted(payload["catalog_row"])
            rec["redacted"] = True
        rec["payload"] = payload
        items.append(rec)
    return {"schema": SNAPSHOT_SCHEMA, "experiment_id": EXPERIMENT_ID, "db_path": snap["db_path"], "mode": "ro+query_only",
            "transaction": "one BEGIN..COMMIT", "poll_seq": snap["poll_seq"], "utc_us": snap["utc_us"],
            "filter": {"rule": "items.dataset in the epoch-000 dataset_hash set", "datasets": len(ep["E"]), "datasets_sha256": _set_sha256(ep["E"])},
            "payload_keep": list(PAYLOAD_KEEP), "items": items, "contexts": snap["contexts"],
            "counts": {"items": len(items), "contexts": len(snap["contexts"]), "items_total_in_db": snap["items_total_in_db"],
                       "siblings": len(reg["siblings"]), "states": reg["states"]}}


# ----------------------------------------------------------------------------------------------- step 3: incoming
def check_incoming(cfg: PinConfig, ep: dict, reg: dict, prep: dict, guard) -> dict:
    entries, metas = [], []
    for n, item_id in enumerate(sorted(ep["ITEMS"])):
        if n % 50 == 0:
            guard.wait("incoming")
        it = reg["by_id"][item_id]
        h, pay = it["dataset"], it["payload_full"]
        d = Path(cfg.incoming_items) / item_id
        row = pay.get("catalog_row")
        _need(isinstance(row, dict), f"incoming item {item_id}: payload has no catalog_row")
        raw = _read(d / "catalog-row.json", f"incoming item {item_id} catalog-row.json")
        _need(raw == intake_canonical(row) + b"\n", f"incoming item {item_id}: catalog-row.json bytes != canonical(payload.catalog_row)+LF")
        _need(common.sha256_bytes(intake_canonical(row)) == pay.get("catalog_row_sha256"), f"incoming item {item_id}: catalog_row_sha256 mismatch")
        _need(row.get("dataset_hash") == h and row.get("canonical_sha256") == it["canonical"], f"incoming item {item_id}: catalog_row identity differs")
        entries.append({"name": f"incoming_catalog_row:{item_id}", "role": "incoming_catalog_row", "path": str(d / "catalog-row.json"),
                        "sha256": common.sha256_bytes(raw), "bytes": len(raw), "dataset_hash": h, "item_id": item_id})
        mraw = _read(d / "materialization.json", f"incoming item {item_id} materialization.json")
        m = json.loads(mraw)
        _need(m.get("dataset_hash") == h and m.get("sample_identity_sha256") == it["sample"], f"incoming item {item_id}: materialization identity differs")
        entries.append({"name": f"incoming_materialization:{item_id}", "role": "incoming_materialization", "path": str(d / "materialization.json"),
                        "sha256": common.sha256_bytes(mraw), "bytes": len(mraw), "dataset_hash": h, "item_id": item_id})
        lo = ((m.get("local_objects") or {}).get("data") or {})
        po = ((m.get("published_objects") or {}).get("data") or {})
        dp = Path(str(lo.get("path") or ""))
        _need(_is_hex64(lo.get("sha256")) and dp.name == lo["sha256"] == po.get("sha256"),
              f"incoming item {item_id}: data object file name / sha256 / published sha256 disagree")
        entries.append(_pin_file(f"incoming_data:{item_id}", "incoming_data", dp, lo["sha256"], lo.get("bytes"), dataset_hash=h,
                                 item_id=item_id, s3=_s3ref(po)))
        pm = m.get("published_manifest") or {}
        mp = Path(str(m.get("manifest") or ""))
        _need(_is_hex64(pm.get("sha256")) and mp.name == pm["sha256"], f"incoming item {item_id}: research manifest path/sha disagree")
        sm = prep["incoming_source_manifest"].get(item_id)
        _need(sm is None or sm["sha256"] == pm["sha256"], f"incoming item {item_id}: prepared source_manifest != materialization manifest")
        mr = pay.get("manifest_ref")
        _need(not isinstance(mr, dict) or mr.get("sha256") == pm["sha256"], f"incoming item {item_id}: payload manifest_ref differs")
        entries.append(_pin_file(f"incoming_manifest:{item_id}", "incoming_manifest", mp, pm["sha256"], pm.get("bytes"), dataset_hash=h,
                                 item_id=item_id, s3=_s3ref(pm)))
        man = json.loads(mp.read_bytes())
        _need(man.get("dataset_hash") == h and man.get("canonical_sha256") == it["canonical"]
              and man.get("sample_identity_sha256") == it["sample"], f"incoming item {item_id}: research manifest identity differs")
        meta = ((row.get("files") or {}).get("meta") or {})
        _need(_is_hex64(meta.get("sha256")) and isinstance(meta.get("bytes"), int) and meta["bytes"] > 0,
              f"incoming item {item_id}: catalog_row.files.meta lacks sha256/bytes")
        metas.append({"sha256": meta["sha256"], "bytes": meta["bytes"], "dataset_hash": h, "item_id": item_id, "cohort": "incoming"})
    return {"entries": entries, "metas": metas}


def sibling_metas(reg: dict) -> list:
    out = []
    for i in reg["siblings"]:
        it = reg["by_id"][i]
        meta = (((it["payload_full"].get("catalog_row") or {}).get("files") or {}).get("meta") or {})
        if _is_hex64(meta.get("sha256")) and isinstance(meta.get("bytes"), int) and meta["bytes"] > 0:
            out.append({"sha256": meta["sha256"], "bytes": meta["bytes"], "dataset_hash": it["dataset"], "item_id": i, "cohort": "sibling"})
    return out


# ----------------------------------------------------------------------------------------------- step 4: frozen
def load_catalog(path) -> dict:
    out, dup = {}, set()
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            h = r.get("dataset_hash")
            if h in out:
                dup.add(h)
            out[h] = r
    return {"rows": out, "duplicates": dup}


def check_frozen(cfg: PinConfig, ep: dict, reg: dict, prep: dict, catalog: dict, guard) -> dict:
    readiness = _json(cfg.fixed["readiness"].path, "readiness")
    rr, dup = {}, set()
    for rec in readiness.get("records") or []:
        if rec.get("cohort") == "frozen":
            h = rec.get("dataset_hash")
            if h in rr:
                dup.add(h)
            rr[h] = rec
    entries, metas, provenance, catalog_refs = [], [], {}, {}
    prefix, root = cfg.frozen_research_prefix, Path(cfg.frozen_research_root)
    for n, h in enumerate(sorted(ep["frozen"])):
        if n % 50 == 0:
            guard.wait("frozen")
        it = reg["frozen_item"][h]
        sm = prep["frozen_source_manifest"][h]
        mpath = _local_from_key(root, prefix, sm["key"], f"frozen dataset {h} source manifest")
        entries.append(_pin_file(f"frozen_manifest:{h}", "frozen_manifest", mpath, sm["sha256"], sm.get("bytes"), dataset_hash=h, s3=_s3ref(sm)))
        man = json.loads(mpath.read_bytes())
        _need(man.get("dataset_hash") == h and man.get("canonical_sha256") == it["canonical"] and man.get("sample_identity_sha256") == it["sample"],
              f"frozen dataset {h}: research manifest identity differs from the registry item")
        dobj = ((man.get("objects") or {}).get("data") or {})
        _need(_is_hex64(dobj.get("sha256")), f"frozen dataset {h}: research manifest lacks objects.data")
        dpath = _local_from_key(root, prefix, dobj.get("key"), f"frozen dataset {h} data object")
        entries.append(_pin_file(f"frozen_data:{h}", "frozen_data", dpath, dobj["sha256"], dobj.get("bytes"), dataset_hash=h, s3=_s3ref(dobj)))
        R = rr.get(h)
        _need(R is not None and h not in dup, f"frozen dataset {h}: {'duplicate' if h in dup else 'no'} readiness record")
        cref = R.get("catalog_ref") or {}
        _need(R.get("canonical_sha256") == it["canonical"] == cref.get("canonical_sha256"), f"frozen dataset {h}: readiness canonical differs")
        _need(R.get("provenance") == cref.get("provenance"), f"frozen dataset {h}: readiness provenance != catalog_ref provenance")
        C = catalog["rows"].get(h)
        _need(C is not None and h not in catalog["duplicates"], f"frozen dataset {h}: no unique catalog v1 row")
        _need(C.get("canonical_sha256") == it["canonical"], f"frozen dataset {h}: catalog v1 canonical differs")
        _need(C.get("provenance") == R.get("provenance"), f"frozen dataset {h}: catalog v1 provenance != readiness provenance")
        rpath = Path(cfg.dataset_records) / f"{h}.json"
        entries.append(_pin_file(f"dataset_record:{h}", "dataset_record", rpath, dataset_hash=h))
        rec = json.loads(rpath.read_bytes())
        _need(rec.get("dataset_hash") == h, f"dataset-record {h}: dataset_hash differs")
        meta = rec.get("meta") or {}
        _need(_is_hex64(meta.get("sha256")) and meta.get("key") == f"blobs/sha256/{meta['sha256']}" and isinstance(meta.get("bytes"), int)
              and isinstance(meta.get("version"), str) and meta["version"] not in ("", "null"), f"dataset-record {h}: incomplete meta pin")
        cm = ((C.get("files") or {}).get("meta") or {})
        _need(cm.get("sha256") == meta["sha256"] and cm.get("bytes") == meta["bytes"], f"frozen dataset {h}: catalog v1 files.meta != record.meta")
        rman = rec.get("manifest") or {}
        _need(rman.get("sha256") == cref.get("sha256") and rman.get("key") == cref.get("key"), f"frozen dataset {h}: record manifest != readiness catalog_ref")
        metas.append({"sha256": meta["sha256"], "bytes": meta["bytes"], "version_id": meta["version"], "dataset_hash": h,
                      "item_id": it["id"], "cohort": "frozen"})
        provenance[h] = R.get("provenance")
        catalog_refs[h] = {k: cref.get(k) for k in ("bucket", "key", "version_id", "sha256", "bytes")}
    return {"entries": entries, "metas": metas, "provenance": provenance, "catalog_refs": catalog_refs}


# ----------------------------------------------------------------------------------------------- step 6 (local): sealed
def sealed_inputs(cfg: PinConfig, ep: dict, catalog: dict) -> dict:
    chain, prev = [], None
    S, SK, MS = set(), set(), set()
    part_of, changes = {}, 0
    for name in cfg.lineage_b_versions:
        fi = cfg.fixed[name]
        _need(Path(fi.path).name.startswith(fi.sha256), f"{name}: file name does not carry its sha256")
        doc = _json(fi.path, name)
        parent = doc.get("parent")
        psha = parent.get("sha256") if isinstance(parent, dict) else parent
        _need(psha == prev, f"{name}: parent sha256 {psha} != previous registry {prev}")
        fams = doc.get("families")
        fams = list(fams.values()) if isinstance(fams, dict) else fams
        _need(isinstance(fams, list), f"{name}: no families")
        sealed_here = set()
        for f in fams:
            part = f.get("partition")
            for ds in f.get("datasets") or []:
                h = ds.get("dataset")
                _need(isinstance(h, str) and bool(HEX16.fullmatch(h)), f"{name}: a dataset id is not 16 hex")
                if part_of.get(h, part) != part:
                    changes += 1
                part_of[h] = part
                if part in SEALED_PARTITIONS:
                    sealed_here.add(h)
                    SK |= lineage_b_rdatasets_keys(ds)
                    if _is_hex64(ds.get("metadata_sha256")):
                        MS.add(ds["metadata_sha256"])
        S |= sealed_here
        chain.append({"name": name, "sha256": fi.sha256, "parent_sha256": psha, "sealed_datasets": len(sealed_here)})
        prev = fi.sha256
    pol = _json(cfg.fixed["lineage_b_family_exclusion_policy"].path, "family exclusion policy")
    _need(sorted(pol.get("sealed_partitions") or []) == sorted(SEALED_PARTITIONS),
          "family exclusion policy sealed_partitions differ from DL-D12 (meta_validation, meta_test)")
    S_in, S_out = S & ep["E"], S - ep["E"]
    refs, unavailable, plan = {}, [], []
    for h in sorted(S_out):
        rpath = Path(cfg.dataset_records) / f"{h}.json"
        if not rpath.is_file():
            unavailable.append(h)
            continue
        rentry = _pin_file(f"dataset_record:{h}", "sealed_dataset_record", rpath, dataset_hash=h)
        rec = json.loads(rpath.read_bytes())
        _need(rec.get("dataset_hash") == h, f"sealed dataset-record {h}: dataset_hash differs")
        pins = {"manifest": rec.get("manifest") or {}, "meta": rec.get("meta") or {},
                "train": {"key": rec.get("key"), "sha256": rec.get("sha256"), "bytes": rec.get("bytes"), "version": rec.get("version")}}
        for part, p in pins.items():
            _need(isinstance(p.get("key"), str) and _is_hex64(p.get("sha256")) and isinstance(p.get("bytes"), int)
                  and isinstance(p.get("version"), str) and p["version"] not in ("", "null"), f"sealed dataset-record {h}: incomplete {part} pin")
            if part != "manifest":
                _need(p["key"] == f"blobs/sha256/{p['sha256']}", f"sealed dataset-record {h}: {part} key is not content-addressed")
        crow = catalog["rows"].get(h)
        ctest = ((crow or {}).get("files") or {}).get("test") or {}
        refs[h] = {"record": rentry, "pins": pins, "catalog_test": {"sha256": ctest.get("sha256"), "bytes": ctest.get("bytes")} if crow else None}
        for part in ("manifest", "train", "meta"):
            p = pins[part]
            plan.append({"role": f"sealed_{part}", "mode": "exact", "bucket": cfg.datasets_bucket, "key": p["key"], "version_id": p["version"],
                         "sha256": p["sha256"], "bytes": p["bytes"], "dataset_hash": h})
    return {"S": S, "S_in": S_in, "S_out": S_out, "SK": SK, "MS": MS, "chain": chain, "partition_changes": changes,
            "refs": refs, "unavailable": unavailable, "plan": plan}


def base_sealed(ep: dict, sealed: dict, frozen: dict, incoming_rows: dict, metas: list) -> dict:
    """dataset -> basis list, for the base sealed rules (exact id, Rdatasets alias, sealed metadata_sha256).
    Used to decide redaction; families.build owns the authoritative sealed computation."""
    meta_of = {}
    for m in metas:
        meta_of.setdefault(m["dataset_hash"], set()).add(m["sha256"])
    out = {}
    for h in sorted(ep["E"]):
        basis = []
        if h in sealed["S"]:
            basis.append("lineage_b_member")
        prov = frozen["provenance"].get(h) if h in frozen["provenance"] else ((incoming_rows.get(h) or {}).get("provenance"))
        if rdatasets_keys(prov) & sealed["SK"]:
            basis.append("lineage_b_rdatasets_alias")
        if meta_of.get(h, set()) & sealed["MS"]:
            basis.append("lineage_b_metadata_sha256")
        if basis:
            out[h] = basis
    return out


# ----------------------------------------------------------------------------------------------- step 7/8: seeds, aliases, exposure
def seed_inputs(cfg: PinConfig, guard) -> dict:
    rows = []
    for line in _read(cfg.fixed["seed_pool_recovery192"].path, "seed pool").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    srcs = set()
    for r in rows:
        try:
            srcs.add(r["source_evidence"][0]["source_view"].split("__")[0])
        except (KeyError, IndexError, TypeError, AttributeError):
            raise PinStop("seed pool row lacks source_evidence[0].source_view") from None
    srcs = sorted(srcs)
    _need(len(srcs) == cfg.expected_seed_sources, f"seed pool names {len(srcs)} seed sources (expected {cfg.expected_seed_sources})")
    entries = []
    for name in srcs:
        _need(name in cfg.seed_view_files, f"seed source {name} has no pinned ten-case view files")
        cand_sha, prep_sha = cfg.seed_view_files[name]
        vdir = Path(cfg.tencase_v3_contexts) / f"{name}__r0__f0"
        entries.append(_pin_file(f"seed_candidates:{name}", "seed_view_candidates", vdir / "bootstrap-candidates.jsonl", cand_sha, seed_source=name))
        entries.append(_pin_file(f"tencase_v3_prepared:{name}", "seed_view_prepared", vdir / "prepared.json", prep_sha, seed_source=name))
    receipt = _json(cfg.fixed["beyondarena_pointer_receipt"].path, "BeyondArena pointer receipt")
    objects = {o.get("key"): o for o in receipt.get("objects") or [] if isinstance(o, dict)}
    idx_obj = [o for k, o in objects.items() if isinstance(k, str) and k.endswith("/index.json")]
    _need(len(idx_obj) == 1 and idx_obj[0].get("sha256") == cfg.fixed["beyondarena_index"].sha256,
          "BeyondArena index sha256 differs from the pointer receipt")
    pointers = {}
    for p in sorted(Path(cfg.ba_pointers).glob("*.json")):
        raw = p.read_bytes()
        doc = json.loads(raw)
        name = p.stem
        _need(doc.get("name") == name, f"BeyondArena pointer {p.name}: name field differs from the file name")
        key = f"benchmarks/BeyondArena/{doc.get('revision')}/sources/{name}.json"
        o = objects.get(key)
        sha = common.sha256_bytes(raw)
        _need(o is not None and o.get("sha256") == sha and o.get("bytes") == len(raw), f"BeyondArena pointer {name}: not equal to the pointer receipt")
        pointers[name] = doc
        entries.append({"name": f"ba_pointer:{name}", "role": "ba_pointer", "path": str(p), "sha256": sha, "bytes": len(raw),
                        "s3": {"bucket": doc.get("bucket") or DATASETS_BUCKET, "key": key, "version_id": o.get("version_id")},
                        "seed_source": name in srcs})
    hashed = {}

    def parquet(name: str):
        if name not in hashed:
            lp = Path(cfg.ba_source_cache) / name / "dataset.parquet"
            if not lp.is_file():
                hashed[name] = None
            else:
                guard.wait(f"parquet:{name}", force=True)
                hashed[name] = (lp, common.sha256_file(lp), lp.stat().st_size)
        return hashed[name]

    for name in srcs:
        _need(name in pointers, f"seed source {name} has no BeyondArena pointer")
        pf = (pointers[name].get("files") or {}).get("dataset.parquet") or {}
        got = parquet(name)
        _need(got is not None and got[1] == pf.get("sha256") and got[2] == pf.get("bytes"), f"seed source {name}: parquet differs from its pointer")
    v6, v6_mismatch = [], []
    cache = Path(cfg.ba_source_cache)
    for d in sorted(x.name for x in cache.iterdir() if x.is_dir()) if cache.is_dir() else []:
        pf = ((pointers.get(d) or {}).get("files") or {}).get("dataset.parquet") or {}
        got = parquet(d)
        if got is not None and pf and got[1] == pf.get("sha256") and got[2] == pf.get("bytes"):
            v6.append(d)
        else:
            v6_mismatch.append(d)
    for d in sorted(set(v6) | set(srcs)):
        lp, sha, size = hashed[d]
        pf = pointers[d]["files"]["dataset.parquet"]
        entries.append({"name": f"ba_parquet:{d}", "role": "seed_source_parquet" if d in srcs else "v6_parquet", "path": str(lp),
                        "sha256": sha, "bytes": size, "s3": (_s3ref(pf) or {}) | {"bucket": pointers[d].get("bucket") or DATASETS_BUCKET},
                        "target": pointers[d].get("target"), "seed_source": d in srcs})
    return {"entries": entries, "seed_sources": srcs, "seed_rows": len(rows), "pointers": sorted(pointers), "v6_names": v6,
            "v6_mismatch": v6_mismatch, "revision": receipt.get("revision")}


def classic_aliases(cfg: PinConfig, ep: dict, seed_sources: list) -> tuple[dict, dict]:
    doc = jsonable(cfg.classic_aliases, "classic_aliases")
    _need(doc.get("schema") == CLASSIC_ALIASES_SCHEMA, "classic aliases: wrong schema")
    names, union_members, review_in, review_out, keywords = set(), 0, 0, 0, 0
    for e in doc.get("entries") or []:
        _need(e.get("name") not in names, f"classic aliases: duplicate entry {e.get('name')}")
        names.add(e.get("name"))
        _need(e.get("action") in ("union", "review"), f"classic alias {e.get('name')}: action {e.get('action')!r}")
        if e["action"] == "union":
            _need(e["name"] in seed_sources, f"classic alias {e['name']}: a union entry must name a seed source")
            _need(bool(e.get("members")), f"classic alias {e['name']}: no members")
            for m in e["members"]:
                _need(m in ep["E"], f"classic alias {e['name']}: member {m} is not an epoch dataset")
            union_members += len(e["members"])
        else:
            keywords += bool(e.get("keyword"))
            for m in e.get("members") or []:
                if m in ep["E"]:
                    review_in += 1
                else:
                    review_out += 1
    check = {"entries": len(names), "union_entries": sum(e["action"] == "union" for e in doc["entries"]), "union_members": union_members,
             "review_members_in_epoch": review_in, "review_members_not_in_epoch": review_out, "keyword_entries": keywords}
    return doc, check


def exposure_inputs(cfg: PinConfig) -> dict:
    tm = _json(cfg.fixed["target_metadata"].path, "target metadata")
    tid = cfg.identities.get("encoder:target_fused_v1")
    _need(tid is None or tm.get("artifact_sha256") == tid.sha256, "target metadata artifact_sha256 differs from the recorded encoder identity")
    spec = _json(cfg.fixed["tabpfn35_encoder_spec"].path, "TabPFN 3.5 encoder spec")
    pid = cfg.identities.get("encoder:tabpfn35")
    _need(pid is None or spec.get("checkpoint_sha256") == pid.sha256, "TabPFN 3.5 checkpoint_sha256 differs from the recorded encoder identity")
    entries = []
    for name in sorted(cfg.identities):
        ide = cfg.identities[name]
        p = Path(ide.path) if ide.path else None
        e = {"name": name, "role": ide.role, "sha256": ide.sha256, "bytes": p.stat().st_size if p and p.is_file() else None,
             "identity_path": str(p) if p else None, "read": False, "verified": "stat_only (recorded identity; P2 never reads weights)"}
        if ide.s3:
            e["s3"] = ide.s3
        if ide.note:
            e["note"] = ide.note
        entries.append(e)
    return {"entries": entries, "fit_records": len(tm.get("fit_records") or []), "tabpfn35_base_weights_modified": spec.get("base_weights_modified")}


# ----------------------------------------------------------------------------------------------- the S3 phase
def readback_sample(frozen: dict, n: int) -> list:
    order = sorted(frozen["catalog_refs"], key=lambda h: (common.sha256_bytes(READBACK_DOMAIN + h.encode()), h))
    return order[:n]


def build_plan(cfg: PinConfig, frozen: dict, metas: list, sealed: dict) -> tuple[list, dict]:
    """Ordered GET plan: metas (frozen exact, then incoming/sibling content-addressed; one GET per distinct sha),
    sealed refs (manifest, train, meta exact; test content-addressed after its manifest), then the readback sample."""
    plan, by_sha = [], {}
    for m in sorted(metas, key=lambda x: (x["cohort"] != "frozen", x["dataset_hash"], x["sha256"])):
        prev = by_sha.get(m["sha256"])
        if prev is not None:
            _need(prev["bytes"] == m["bytes"], f"meta {m['sha256']}: two byte counts pinned")
            prev["datasets"].append(m["dataset_hash"])
            if m.get("version_id") and not prev.get("version_id"):
                prev.update(mode="exact", version_id=m["version_id"], role="meta_frozen")
            continue
        item = {"role": f"meta_{m['cohort']}", "mode": "exact" if m.get("version_id") else "content_addressed", "bucket": cfg.datasets_bucket,
                "key": f"blobs/sha256/{m['sha256']}", "version_id": m.get("version_id"), "sha256": m["sha256"], "bytes": m["bytes"],
                "datasets": [m["dataset_hash"]]}
        by_sha[m["sha256"]] = item
        plan.append(item)
    tests_planned, tests_bytes = 0, 0
    for h in sorted(sealed["refs"]):
        plan.extend(x for x in sealed["plan"] if x["dataset_hash"] == h)
        ct = sealed["refs"][h]["catalog_test"]
        tests_planned += 1
        tests_bytes += (ct or {}).get("bytes") or 0
    for h in readback_sample(frozen, cfg.readback_sample):
        ref = frozen["catalog_refs"][h]
        _need(isinstance(ref.get("key"), str) and isinstance(ref.get("version_id"), str) and _is_hex64(ref.get("sha256"))
              and isinstance(ref.get("bytes"), int), f"frozen dataset {h}: incomplete readiness catalog_ref")
        plan.append({"role": "readback_frozen_manifest", "mode": "exact", "bucket": ref.get("bucket") or cfg.datasets_bucket, "key": ref["key"],
                     "version_id": ref["version_id"], "sha256": ref["sha256"], "bytes": ref["bytes"], "dataset_hash": h})
    gets = len(plan) + tests_planned
    nbytes = sum(x["bytes"] for x in plan) + tests_bytes
    _need(gets <= cfg.max_gets, f"S3 plan: {gets} GETs > {cfg.max_gets}")
    _need(nbytes <= cfg.max_bytes, f"S3 plan: {nbytes} B > {cfg.max_bytes} B")
    return plan, {"planned_gets": gets, "planned_bytes": nbytes, "sealed_test_gets": tests_planned, "distinct_metas": len(by_sha)}


def run_reads(cfg: PinConfig, plan: list, sealed: dict, catalog: dict, redact_meta: set, clients: dict, ledger: Ledger, guard) -> dict:
    metas, sealed_files, readback = {}, {}, {"sampled": 0, "ok": 0}
    test_after = {}
    for n, item in enumerate(plan):
        if n % GUARD_EVERY_GETS == 0:
            guard.wait("s3", force=True)
        body = s3_get(item, clients, ledger)
        entry = ledger.entries[-1]
        role, h = item["role"], item.get("dataset_hash")
        if role.startswith("meta_"):
            ext, info = meta_extract(body, redact=item["sha256"] in redact_meta)
            metas[item["sha256"]] = {"extract": ext, "info": info, "get": entry, "datasets": sorted(set(item["datasets"]))}
        elif role == "readback_frozen_manifest":
            row = _loads(body, f"readback of datasets/{h}/manifest.json")
            _need(isinstance(row, dict), f"readback of datasets/{h}/manifest.json is not an object")
            crow = catalog["rows"].get(h) or {}
            _need(row.get("dataset_hash") == h and row.get("canonical_sha256") == crow.get("canonical_sha256")
                  and ((row.get("files") or {}).get("meta") or {}).get("sha256") == ((crow.get("files") or {}).get("meta") or {}).get("sha256"),
                  f"readback of datasets/{h}/manifest.json disagrees with catalog v1")
            readback["sampled"] += 1
            readback["ok"] += 1
        elif role == "sealed_manifest":
            row = _loads(body, f"sealed ref {h} manifest")
            _need(isinstance(row, dict), f"sealed ref {h}: manifest is not an object")
            pins = sealed["refs"][h]["pins"]
            files = row.get("files") or {}
            test = files.get("test") or {}
            _need(row.get("dataset_hash") == h, f"sealed ref {h}: manifest dataset_hash differs")
            _need((files.get("train") or {}).get("sha256") == pins["train"]["sha256"]
                  and (files.get("meta") or {}).get("sha256") == pins["meta"]["sha256"], f"sealed ref {h}: manifest train/meta pins differ from the record")
            _need(_is_hex64(test.get("sha256")) and isinstance(test.get("bytes"), int), f"sealed ref {h}: manifest lacks a test pin")
            ct = sealed["refs"][h]["catalog_test"]
            _need(ct is None or (ct["sha256"] == test["sha256"] and ct["bytes"] == test["bytes"]), f"sealed ref {h}: manifest test pin != catalog v1")
            sealed_files[(h, "manifest")] = {"extract": manifest_extract(row), "get": entry}
            sealed["refs"][h]["train_has_header"] = (files.get("train") or {}).get("has_header")
            tv = test.get("version_id") or test.get("version")      # a CatalogRow carries none: then D-P2-1 content-addressed
            tv = tv if isinstance(tv, str) and tv not in ("", "null") else None
            test_after[h] = {"role": "sealed_test", "mode": "exact" if tv else "content_addressed", "bucket": item["bucket"],
                             "key": f"blobs/sha256/{test['sha256']}", "version_id": tv, "sha256": test["sha256"], "bytes": test["bytes"],
                             "dataset_hash": h, "has_header": test.get("has_header")}
        elif role in ("sealed_train", "sealed_test"):
            sealed_files[(h, role[len("sealed_"):])] = {"raw": body, "get": entry}
        elif role == "sealed_meta":
            ext, info = meta_extract(body, redact=True)
            sealed_files[(h, "meta")] = {"extract": ext, "info": info, "get": entry}
        # the dependent test GET follows the last planned part of its reference
        nxt = plan[n + 1] if n + 1 < len(plan) else None
        if h in test_after and (nxt is None or nxt.get("dataset_hash") != h or not nxt["role"].startswith("sealed_")):
            body_t = s3_get(test_after[h], clients, ledger)
            sealed_files[(h, "test")] = {"raw": body_t, "get": ledger.entries[-1], "has_header": test_after.pop(h)["has_header"]}
    _need(not test_after, "a sealed reference test GET was not executed")
    return {"metas": metas, "sealed_files": sealed_files, "readback": readback}


# ----------------------------------------------------------------------------------------------- write
def _precheck_out(out: Path) -> None:
    _need(not (out / "alias-sources.json").exists(), f"{out}/alias-sources.json exists: already pinned (pin once per OUT)")
    inp = out / "inputs"
    _need(not (inp.exists() and any(inp.iterdir())), f"{inp} holds files from an earlier attempt without alias-sources.json; move it aside")


def _write(out: Path, rel: str, data: bytes) -> dict:
    try:
        res = common.immutable_write(out / rel, data)
    except common.ImmutableConflict:
        raise PinStop(f"{rel}: create-only path already holds different bytes") from None
    return {"local_path": rel, "local_sha256": res["sha256"], "local_bytes": res["bytes"]}


def pin(out, policy=None, config: PinConfig | None = None, clients: dict | None = None, guard=None, ledger_path=None,
        receipt_path=None, preflight: bool = False) -> dict:
    """Run P2 step 1 and return a counts-only summary. Raises PinStop on any failed check (OUT untouched).
    policy: the families POLICY (copied into alias-sources.json['policy'] through jsonable()); clients: {'exact', 'cas'}
    S3 clients (default: dltrain.s3io guarded live clients, built only after every local check passed); guard: a
    CpuGuard (default on cpu-admission.json); ledger_path: create-only JSONL journal of every GET; receipt_path:
    create-only receipt (verdict PASS, or STOP with the stage and the S3 totals so far). preflight=True runs every
    local check and the S3 plan/budget, then returns counts: no S3 client is built and nothing is written."""
    cfg = config or default_config()
    started = common.utc_us()
    state = {"stage": "precheck", "ledger": None}
    try:
        _need(ledger_path is None or not Path(ledger_path).exists(), f"ledger journal {ledger_path} exists (one journal per attempt)")
        _need(receipt_path is None or not Path(receipt_path).exists(), f"receipt {receipt_path} exists (one receipt per attempt)")
        summary = _pin(Path(out), policy, cfg, clients, guard, ledger_path, started, state, preflight)
    except PinStop as err:
        if receipt_path and not Path(receipt_path).exists():
            led = state["ledger"]
            common.write_receipt(receipt_path, {
                "schema": RECEIPT_SCHEMA, "experiment_id": EXPERIMENT_ID, "verdict": "STOP", "stop": str(err)[:500],
                "stage": state["stage"], "started_utc_us": started, "utc_us": common.utc_us(),
                "s3_totals": led.totals() if led is not None else {"gets": 0, "bytes": 0, "journal": None}})
        raise
    if receipt_path and not preflight:
        common.write_receipt(receipt_path, {"schema": RECEIPT_SCHEMA, "experiment_id": EXPERIMENT_ID, "utc_us": common.utc_us(),
                                            "started_utc_us": started, **summary})
    return summary


def _pin(out: Path, policy, cfg: PinConfig, clients, guard, ledger_path, started: int, state: dict, preflight: bool = False) -> dict:
    if policy is None:
        policy = _families_policy()
    pol = jsonable(policy)
    _need(isinstance(pol, dict) and bool(pol), "policy must be a non-empty mapping")
    for k, v in pol.items():
        if k.lower() == "sealed_partitions":
            _need(isinstance(v, list) and sorted(v) == sorted(SEALED_PARTITIONS), "policy sealed partitions differ from DL-D12")
    _precheck_out(out)
    guard = guard or CpuGuard(cfg.cpu_admission)
    guard.wait("start", force=True)
    # ---- local checks (no S3, no writes)
    state["stage"] = "local"
    fixed = pin_fixed(cfg, guard)
    gate = check_gate(cfg)
    ep = load_epoch(cfg.fixed["epoch"].path, cfg.fixed["epoch"].sha256)
    prep = check_prepared(ep, guard)
    guard.wait("registry", force=True)
    snap = registry_snapshot(cfg.registry_db, ep["E"])
    reg = check_registry(snap, ep, prep)
    inc = check_incoming(cfg, ep, reg, prep, guard)
    catalog = load_catalog(cfg.fixed["catalog_v1"].path)
    fro = check_frozen(cfg, ep, reg, prep, catalog, guard)
    sealed = sealed_inputs(cfg, ep, catalog)
    seeds = seed_inputs(cfg, guard)
    aliases_doc, alias_check = classic_aliases(cfg, ep, seeds["seed_sources"])
    expo = exposure_inputs(cfg)
    metas_needed = fro["metas"] + inc["metas"] + sibling_metas(reg)
    incoming_rows = {reg["by_id"][i]["dataset"]: reg["by_id"][i]["payload_full"].get("catalog_row") or {} for i in ep["ITEMS"]}
    base = base_sealed(ep, sealed, fro, incoming_rows, metas_needed)
    redact_meta = {m["sha256"] for m in metas_needed if m["dataset_hash"] in base}
    plan, plan_info = build_plan(cfg, fro, metas_needed, sealed)
    snap_doc = snapshot_doc(snap, ep, reg, set(base))
    if preflight:
        state["stage"] = "preflight-done"
        return {"verdict": "PREFLIGHT_PASS", "s3_plan": plan_info, "counts": {
            "epoch_rows": len(ep["rows"]), "epoch_datasets": len(ep["E"]), "epoch_items": len(ep["ITEMS"]), "prepared_files": prep["files"],
            "poll_seq": snap["poll_seq"], "snapshot_items": len(snap["items"]), "snapshot_contexts": len(snap["contexts"]),
            "siblings": len(reg["siblings"]), "fixed_inputs": len(fixed), "sealed_S": len(sealed["S"]), "sealed_in_epoch": len(sealed["S_in"]),
            "sealed_refs": len(sealed["refs"]), "sealed_content_unavailable": len(sealed["unavailable"]), "base_sealed_epoch_datasets": len(base),
            "metas_to_redact": len(redact_meta), "seed_sources": len(seeds["seed_sources"]), "ba_pointers": len(seeds["pointers"]),
            "v6_names": len(seeds["v6_names"]), "v6_mismatch": len(seeds["v6_mismatch"]), "snapshot_bytes": len(common.canonical(snap_doc))},
            "classic_aliases": alias_check, "cpu_guard": guard.summary()}
    # ---- S3 (budget-checked before every GET; every request journaled)
    state["stage"] = "s3"
    guard.wait("s3-start", force=True)
    ledger = state["ledger"] = Ledger(ledger_path, cfg.max_gets, cfg.max_bytes)
    clients = clients or default_clients()
    got = run_reads(cfg, plan, sealed, catalog, redact_meta, clients, ledger, guard)
    _need(ledger.gets <= cfg.max_gets and ledger.bytes <= cfg.max_bytes, "S3 budget exceeded")
    # ---- writes (create-only), alias-sources.json last
    state["stage"] = "write"
    guard.wait("write", force=True)
    inputs = list(fixed.values()) + inc["entries"] + fro["entries"] + seeds["entries"] + expo["entries"]
    inputs.append({"name": "epoch_prepared_set", "role": "prepared_set", "sha256": prep["set_sha256"], "bytes": prep["bytes"],
                   "files": prep["files"], "rule": "sha256(canonical(sorted [context_sha256, prepared sha256, bytes])) over every epoch row; "
                                                   "each file re-hashed against epoch prepared_ref at pin"})
    snap_bytes = common.canonical(snap_doc)
    w = _write(out, "inputs/registry-snapshot.json", snap_bytes)
    inputs.append({"name": "registry_snapshot", "role": "registry_snapshot", "sha256": w["local_sha256"], "bytes": w["local_bytes"], **w,
                   "db_path": str(cfg.registry_db), "poll_seq": snap["poll_seq"]})
    for sha in sorted(got["metas"]):
        m = got["metas"][sha]
        w = _write(out, f"inputs/meta/{sha}.json", common.canonical(m["extract"]))
        g = m["get"]
        inputs.append({"name": f"meta:{sha}", "role": "meta", "sha256": sha, "bytes": g["bytes"], **w, "datasets": m["datasets"],
                       "s3": {"bucket": g["bucket"], "key": g["key"], "version_id": g["version_id"]}, "read_mode": g["mode"],
                       "redacted": m["info"]["redacted"], "nonfinite_to_null": m["info"]["nonfinite_to_null"]})
    sealed_index = {}
    for (h, part) in sorted(got["sealed_files"]):
        f = got["sealed_files"][(h, part)]
        g = f["get"]
        if "raw" in f:
            w = _write(out, f"inputs/sealed-ref/{g['sha256']}", f["raw"])
        else:
            w = _write(out, f"inputs/sealed-ref/{part}/{g['sha256']}.json", common.canonical(f["extract"]))
        e = {"name": f"sealed_ref_{part}:{h}", "role": f"sealed_ref_{part}", "sha256": g["sha256"], "bytes": g["bytes"], **w,
             "dataset_hash": h, "s3": {"bucket": g["bucket"], "key": g["key"], "version_id": g["version_id"]}, "read_mode": g["mode"],
             "redacted": "raw" not in f}
        hh = f.get("has_header") if part == "test" else sealed["refs"][h].get("train_has_header") if part == "train" else None
        if hh is not None:
            e["has_header"] = hh
        inputs.append(e)
        sealed_index.setdefault(h, {})[part] = e["name"]
    for h in sorted(sealed["refs"]):
        inputs.append(sealed["refs"][h]["record"])
    w = _write(out, "inputs/classic-aliases.json", common.canonical(aliases_doc))
    inputs.append({"name": "classic_aliases", "role": "classic_aliases", "sha256": w["local_sha256"], "bytes": w["local_bytes"], **w})
    names = [e["name"] for e in inputs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    _need(not dupes, f"duplicate input names: {dupes[:5]}")
    inputs.sort(key=lambda e: e["name"])
    redacted_metas = sorted(s for s, m in got["metas"].items() if m["info"]["redacted"])
    counts = {
        "epoch_rows": len(ep["rows"]), "epoch_datasets": len(ep["E"]), "epoch_items": len(ep["ITEMS"]), "frozen_datasets": len(ep["frozen"]),
        "incoming_datasets": len(ep["incoming"]), "prepared_files": prep["files"], "poll_seq": snap["poll_seq"],
        "snapshot_items": len(snap["items"]), "snapshot_contexts": len(snap["contexts"]), "siblings": len(reg["siblings"]),
        "metas": len(got["metas"]), "metas_redacted": len(redacted_metas),
        "metas_nonfinite_to_null": sum(m["info"]["nonfinite_to_null"] for m in got["metas"].values()),
        "sealed_S": len(sealed["S"]), "sealed_in_epoch": len(sealed["S_in"]), "sealed_out_of_epoch": len(sealed["S_out"]),
        "sealed_refs": len(sealed["refs"]), "sealed_content_unavailable": len(sealed["unavailable"]), "sealed_rdatasets_keys": len(sealed["SK"]),
        "base_sealed_epoch_datasets": len(base), "readback_ok": got["readback"]["ok"], "seed_sources": len(seeds["seed_sources"]),
        "seed_rows": seeds["seed_rows"], "ba_pointers": len(seeds["pointers"]), "v6_names": len(seeds["v6_names"]),
        "v6_mismatch": len(seeds["v6_mismatch"]), "inputs": len(inputs), "s3_gets": ledger.gets, "s3_bytes": ledger.bytes,
    }
    doc = {
        "schema": ALIAS_SOURCES_SCHEMA, "experiment_id": EXPERIMENT_ID, "created_utc_us": common.utc_us(), "started_utc_us": started,
        "policy": pol,
        "pin_rules": {"payload_keep": list(PAYLOAD_KEEP), "meta_keep": list(META_KEEP) + [".".join(META_DEDUP_PATH)],
                      "meta_redacted_fields": list(META_SENSITIVE), "row_redacted_fields": list(ROW_SENSITIVE),
                      "sealed_partitions": list(SEALED_PARTITIONS), "s3_limits": {"gets": cfg.max_gets, "bytes": cfg.max_bytes},
                      "readback": {"sample": cfg.readback_sample, "domain": READBACK_DOMAIN.decode()},
                      "canonical_for_registry_rows": "intake_canonical (ensure_ascii=True)", "nonfinite_meta_numbers": "null"},
        "gate": gate,
        "epoch": {"path": ep["path"], "sha256": ep["sha256"], "bytes": ep["bytes"], "s3": gate["epoch_s3"], "binding": gate["binding_s3"],
                  "rows": len(ep["rows"]), "datasets": len(ep["E"]), "items": len(ep["ITEMS"]), "datasets_sha256": _set_sha256(ep["E"]),
                  "items_sha256": _set_sha256(ep["ITEMS"])},
        "registry_snapshot": {"path": "inputs/registry-snapshot.json", "sha256": common.sha256_bytes(snap_bytes), "bytes": len(snap_bytes),
                              "poll_seq": snap["poll_seq"], "utc_us": snap["utc_us"], "items": len(snap["items"]), "contexts": len(snap["contexts"]),
                              "siblings": len(reg["siblings"]), "db_path": str(cfg.registry_db)},
        "sealed": {"partitions": list(SEALED_PARTITIONS), "chain": sealed["chain"], "partition_changes": sealed["partition_changes"],
                   "S": sorted(sealed["S"]), "S_in_epoch": sorted(sealed["S_in"]), "S_out_of_epoch": sorted(sealed["S_out"]),
                   "refs": sealed_index, "content_unavailable": sealed["unavailable"],
                   "rdatasets_keys": {"count": len(sealed["SK"]), "set_sha256": _set_sha256(sealed["SK"])},
                   "metadata_sha256": {"count": len(sealed["MS"]), "set_sha256": _set_sha256(sealed["MS"])},
                   "base_sealed_epoch_datasets": base, "redacted_meta_sha256": redacted_metas,
                   "note": "base sealed set computed at pin for redaction only; families.build owns sealing"},
        "seeds": {"seed_sources": seeds["seed_sources"], "seed_rows": seeds["seed_rows"]},
        "beyondarena": {"revision": seeds["revision"], "pointers": len(seeds["pointers"]), "seed_sources": seeds["seed_sources"],
                        "v6_names": seeds["v6_names"], "v6_mismatch": seeds["v6_mismatch"]},
        "classic_aliases": alias_check,
        "exposure": {"target_fit_records": expo["fit_records"], "tabpfn35_base_weights_modified": expo["tabpfn35_base_weights_modified"]},
        "readback": got["readback"] | {"sample_rule": "first N frozen datasets by sha256(domain + dataset_hash)"},
        "s3_plan": plan_info,
        "s3_reads": ledger.entries,
        "s3_totals": ledger.totals(),
        "layout": LAYOUT,
        "input_names": INPUT_NAMES,
        "inputs": inputs,
        "counts": counts,
        "cpu_guard": guard.summary(),
        "pin_code": {"module": "dltrain.family_pin", "sha256": common.sha256_file(__file__)},
    }
    body = common.canonical(doc)
    res = common.immutable_write(out / "alias-sources.json", body)
    state["stage"] = "done"
    return {"verdict": "PASS", "alias_sources": {"path": str(out / "alias-sources.json"), "sha256": res["sha256"], "bytes": res["bytes"]},
            "counts": counts, "s3_totals": ledger.totals(), "cpu_guard": guard.summary()}


def _families_policy():
    try:
        from dltrain import families
    except ImportError as err:
        raise PinStop(f"no policy given and dltrain.families is not importable ({type(err).__name__})") from None
    pol = getattr(families, "POLICY", None)
    if callable(pol):
        pol = pol()
    _need(isinstance(pol, dict) and bool(pol), "no policy given and dltrain.families.POLICY is not a non-empty mapping")
    return pol


# ----------------------------------------------------------------------------------------------- local re-pin (no network)
def verify_generation(src_dir, from_receipt, guard=None) -> dict:
    """Re-verify a pinned generation without any network or SQLite access. Its pin receipt (verdict PASS) names
    alias-sources.json (sha256, bytes) and the S3 ledger journal; the journal must equal alias-sources s3_reads and the
    receipt totals, every request 'ok'. Every pinned file is re-hashed: each external 'path' against sha256/bytes (all
    roles, the v6 parquets included) and each local copy under the generation against local_sha256/local_bytes; each
    meta and sealed-reference copy needs its GET in the journal with the same sha256 and bytes. The epoch's prepared.json
    set is re-hashed by the build that follows (families.load_inputs)."""
    guard = guard or NoGuard()
    src = Path(src_dir)
    rec_raw = _read(from_receipt, "source pin receipt")
    rec = _loads(rec_raw, "source pin receipt")
    _need(isinstance(rec, dict) and rec.get("schema") == RECEIPT_SCHEMA and rec.get("verdict") == "PASS",
          "the source pin receipt is not a PASS jope.dl-family-pin-receipt/v1")
    ref = rec.get("alias_sources") or {}
    alias = src / "alias-sources.json"
    _need(Path(str(ref.get("path"))).resolve() == alias.resolve(), "the pin receipt names another alias-sources.json")
    raw = _read(alias, "source alias-sources.json")
    _need(common.sha256_bytes(raw) == ref.get("sha256") and len(raw) == ref.get("bytes"),
          "source alias-sources.json differs from its pin receipt")
    doc = _loads(raw, "source alias-sources.json")
    _need(isinstance(doc, dict) and doc.get("schema") == ALIAS_SOURCES_SCHEMA, "source alias-sources.json: wrong schema")
    totals = rec.get("s3_totals") or {}
    jpath = Path(str(totals.get("journal")))
    jraw = _read(jpath, "S3 ledger journal")
    entries = [json.loads(line) for line in jraw.splitlines() if line.strip()]
    _need(entries == doc.get("s3_reads"), "the ledger journal differs from alias-sources s3_reads")
    _need(len(entries) == totals.get("gets") == (rec.get("counts") or {}).get("s3_gets")
          and sum(e.get("bytes") or 0 for e in entries) == totals.get("bytes"),
          "the ledger journal totals differ from the pin receipt")
    _need(all(e.get("status") == "ok" for e in entries), "the ledger journal holds a request that is not 'ok'")
    got_by_sha = {}
    for e in entries:
        got_by_sha.setdefault(e["sha256"], e)
    checked = {"external": 0, "local": 0, "bytes": 0, "ledger_matched": 0}
    for n, e in enumerate(doc.get("inputs") or []):
        if n % 100 == 0:
            guard.wait("repin-verify")
        if e.get("path"):
            p = Path(e["path"])
            _need(p.is_file(), f"{e['name']}: pinned file is missing")
            _need(common.sha256_file(p) == e.get("sha256") and (e.get("bytes") is None or p.stat().st_size == e["bytes"]),
                  f"{e['name']}: pinned file changed")
            checked["external"] += 1
            checked["bytes"] += p.stat().st_size
        if e.get("local_path"):
            p = src / e["local_path"]
            _need(".." not in Path(e["local_path"]).parts and p.is_file(), f"{e['name']}: local copy is missing")
            _need(common.sha256_file(p) == e.get("local_sha256") and p.stat().st_size == e.get("local_bytes"),
                  f"{e['name']}: local copy changed")
            checked["local"] += 1
            checked["bytes"] += p.stat().st_size
        role = e.get("role") or ""
        if role == "meta" or role.startswith("sealed_ref_"):
            g = got_by_sha.get(e.get("sha256"))
            _need(g is not None and g.get("bytes") == e.get("bytes") and g.get("status") == "ok",
                  f"{e['name']}: no 'ok' ledger GET with its sha256 and bytes")
            checked["ledger_matched"] += 1
    snap = doc.get("registry_snapshot") or {}
    sp = src / str(snap.get("path"))
    _need(sp.is_file() and common.sha256_file(sp) == snap.get("sha256"), "registry snapshot differs from alias-sources")
    return {"doc": doc, "raw": raw, "receipt": {"path": str(from_receipt), "sha256": common.sha256_bytes(rec_raw)},
            "alias_sources": {"path": str(alias), "sha256": common.sha256_bytes(raw), "bytes": len(raw)},
            "journal": {"path": str(jpath), "sha256": common.sha256_bytes(jraw), "entries": len(entries),
                        "bytes": totals.get("bytes")},
            "checked": checked}


def load_withhold(path, sha256: str | None) -> dict:
    """A withhold list (review fix SC-3): {schema, identities: [...], reason}, pinned by sha256."""
    raw = _read(path, "withhold list")
    _need(_is_hex64(sha256) and common.sha256_bytes(raw) == sha256, "the withhold list needs a matching --withhold-sha256")
    doc = _loads(raw, "withhold list")
    ids = doc.get("identities") if isinstance(doc, dict) else None
    _need(isinstance(doc, dict) and doc.get("schema") == WITHHOLD_SCHEMA and isinstance(ids, list) and ids
          and all(isinstance(i, str) and re.fullmatch(r"(item:[0-9a-f]{64}|beyondarena:[A-Za-z0-9_.-]+)", i) for i in ids),
          f"the withhold list is not a {WITHHOLD_SCHEMA} document with identity ids")
    return {"path": str(path), "sha256": sha256, "identities": sorted(set(ids))}


def _canary_regex(strings) -> re.Pattern | None:
    """Every canary as a case-insensitive word sequence across any separator run (long canaries also inside words)."""
    pats = []
    for s in sorted({s for s in strings if isinstance(s, str)}, key=lambda x: (-len(x), x)):
        words = re.findall(r"[a-z0-9]+", s.lower())
        if not words:
            continue
        body = r"[^a-z0-9]+".join(re.escape(w) for w in words)
        if len(s) < 12:
            body = r"(?<![a-z0-9])" + body + r"(?![a-z0-9])"
        pats.append(body)
    return re.compile("|".join(pats), re.IGNORECASE) if pats else None


def redact_text_doc(raw: bytes, strings) -> bytes:
    """A redacted copy of a JSON (or text) receipt: every canary occurrence inside a string value or key becomes
    '[sealed:<sha12>]' (review fix SC-8). JSON stays valid (each string is redacted on its own)."""
    rx = _canary_regex(strings)
    sub = (lambda s: s) if rx is None else (
        lambda s: rx.sub(lambda m: "[sealed:" + common.sha256_bytes(m.group(0).lower().encode())[:12] + "]", s))

    def walk(o):
        if isinstance(o, str):
            return sub(o)
        if isinstance(o, dict):
            return {sub(k): walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o
    try:
        doc = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return sub(raw.decode("utf-8", errors="replace")).encode("utf-8")
    return common.canonical(walk(doc))


def _compare_models(a, b) -> list:
    """Differences (names only) between the plaintext pass and the redacted pass of a local re-pin."""
    out = []
    for name in ("review.csv",) + tuple(k for k in a.outputs if k.startswith("screen/")):
        if a.outputs.get(name) != b.outputs.get(name):
            out.append(f"output {name}")
    pa = {f: sorted(F["_members"]) for f, F in a.pre.families.items()}
    pb = {f: sorted(F["_members"]) for f, F in b.pre.families.items()}
    if pa != pb:
        out.append("family partition")
    for key in ("sealed", "sealed_basis", "withheld", "independence_qualified"):
        if {f: F.get(key) for f, F in a.pre.families.items()} != {f: F.get(key) for f, F in b.pre.families.items()}:
            out.append(f"family {key}")
    if a.summary.get("counts") != b.summary.get("counts"):
        out.append("counts")
    return out


def _texts_for_scan(files: dict) -> tuple[dict, list]:
    texts, undecodable = {}, []
    for rel, raw in sorted(files.items()):
        if raw[:2] == b"\x1f\x8b":
            try:
                raw = gzip.decompress(raw)
            except (OSError, EOFError):
                undecodable.append(rel)
                continue
        try:
            texts[rel] = raw.decode("utf-8")
        except UnicodeDecodeError:
            undecodable.append(rel)
    return texts, undecodable


def repin_local(src_dir, out, *, from_receipt, policy=None, guard=None, receipt_path=None, withhold=None,
                expect=None, receipts_root=None) -> dict:
    """Network-free re-pin of an existing generation into a fresh generation directory (review fixes CL-5, SC-4, SC-3).

    1. verify_generation(): the source generation's pin receipt, alias-sources.json, ledger journal and every pinned
       file (no S3, no SQLite).
    2. Plaintext pass: families.run_build over the source inputs with the current code and POLICY (in memory) gives the
       pre-review seal set: every member of a sealed (hard or provisional) or withheld family (withheld = the withhold
       list plus the families a recommended union would join with a sealed family).
    3. Redacted inputs: every meta loses its META_LOSSLESS values (hashed); the metas of those members lose every
       META_SENSITIVE value; the snapshot drops ROW_SENSITIVE from frozen and incoming catalog rows and hashes it in their
       siblings' rows; sealed-reference copies and classic aliases are copied byte-identical.
    4. The redacted inputs are written create-only; a second in-memory build over them (alias-sources bytes held in
       memory, with repin.seal_floor and repin.redaction) must reproduce review.csv, the screen files, the partition,
       the seals and the counts, else STOP before alias-sources.json exists.
    5. alias-sources.json is written last (create-only), then the receipt. S3 requests: 0.
    A pinned input that sits under receipts_root (RUN/receipts/P2, e.g. the P2 survey) and holds canary strings is
    re-pinned as a redacted local copy under inputs/spec/ (review fix SC-8: the original can then move to restricted
    custody without breaking the generation). Before alias-sources.json is written, every written file and the
    alias-sources bytes are canary-scanned (the plaintext and redacted passes' canaries); a hit is a STOP."""
    from dltrain import families as fam         # families imports this module; the re-pin needs its build
    started = common.utc_us()
    src, out = Path(src_dir), Path(out)
    receipts_root = Path(receipts_root) if receipts_root is not None else RUN / "receipts" / "P2"
    _need(src.resolve() != out.resolve(), "the re-pin needs a fresh generation directory (OUT != the source)")
    _need(not out.resolve().is_relative_to(src.resolve() / "inputs"), "OUT must not sit inside the source inputs")
    _need(receipt_path is None or not Path(receipt_path).exists(), f"receipt {receipt_path} exists (one per attempt)")
    _precheck_out(out)
    guard = guard or CpuGuard()
    policy = fam.POLICY if policy is None else policy
    guard.wait("repin-start", force=True)
    gen = verify_generation(src, from_receipt, guard)
    cfg1 = gen["doc"]
    wh = withhold or {"identities": []}
    guard.wait("repin-plaintext", force=True)
    res1 = fam.run_build(src / "alias-sources.json", None, guard=guard, expect=expect, policy_from_code=True,
                         extra_withhold=frozenset(wh["identities"]), rehash=False)
    hidden = sorted(fam._hidden_members(res1.pre))
    idents = res1.base.idents
    full_metas = {idents[m].meta_sha256 for m in hidden if idents[m].meta_sha256}
    files, inputs, mcount = {}, [], {"full": 0, "lossless": 0, "already_redacted": 0}
    cans1 = list(res1.cans)
    spec_copies = []
    for e in cfg1.get("inputs") or []:
        e2 = dict(e)
        if e.get("path") and Path(e["path"]).resolve().is_relative_to(receipts_root.resolve()):
            raw = Path(e["path"]).read_bytes()
            if any(fam.canary_scan({"x": raw.decode("utf-8", errors="replace")}, cans1).values()):
                red = redact_text_doc(raw, cans1)
                _need(not any(fam.canary_scan({"x": red.decode("utf-8")}, cans1).values()),
                      f"{e['name']}: the redacted copy still holds a canary (STOP)")
                rel = f"inputs/spec/{common.sha256_bytes(red)}{Path(e['path']).suffix}"
                files[rel] = red
                e2 = {k: v for k, v in e2.items() if k not in ("path", "s3")}
                e2.update(sha256=common.sha256_bytes(red), bytes=len(red), local_path=rel,
                          local_sha256=common.sha256_bytes(red), local_bytes=len(red), redacted=True,
                          redacted_from={"path": e["path"], "sha256": e.get("sha256"), "bytes": e.get("bytes")})
                spec_copies.append({"name": e["name"], "original_sha256": e.get("sha256"), "copy": rel})
            inputs.append(e2)
            continue
        if e.get("role") == "meta":
            doc = _loads((src / e["local_path"]).read_bytes(), e["name"])
            full = e["name"].split(":", 1)[1] in full_metas
            red = redact_meta_doc(doc, full)
            b = common.canonical(red)
            mcount["full" if full else "lossless"] += 1
            mcount["already_redacted"] += bool(e.get("redacted"))
            files[e["local_path"]] = b
            e2.update(local_sha256=common.sha256_bytes(b), local_bytes=len(b), redacted=bool(full or e.get("redacted")),
                      redaction="all_sensitive" if (full or e.get("redacted")) else "lossless_fields")
        elif e.get("local_path") and e.get("name") != "registry_snapshot":
            files[e["local_path"]] = (src / e["local_path"]).read_bytes()
        inputs.append(e2)
    snap_rel = (cfg1.get("registry_snapshot") or {}).get("path")
    snap = _loads((src / snap_rel).read_bytes(), "registry snapshot")
    snap2, scount = redact_snapshot_doc(snap, set(hidden))
    snap_b = common.canonical(snap2)
    files[snap_rel] = snap_b
    for e2 in inputs:
        if e2.get("name") == "registry_snapshot":
            e2.update(sha256=common.sha256_bytes(snap_b), bytes=len(snap_b), local_sha256=common.sha256_bytes(snap_b),
                      local_bytes=len(snap_b))
    seal_floor = {"identities": hidden, "sha256": common.sha256_bytes(common.canonical(hidden)),
                  "rule": "members of the families the plaintext pass sealed (hard or provisional) or withheld"}
    repin = {
        "mode": "local", "s3_requests": 0, "sqlite": False,
        "source": {"generation_dir": str(src), "alias_sources": gen["alias_sources"], "pin_receipt": gen["receipt"],
                   "ledger": gen["journal"], "policy_sha256": res1.base.inp.pinned_policy_sha256},
        "verified": gen["checked"],
        "withhold": {"identities": wh["identities"], "file": {k: wh.get(k) for k in ("path", "sha256")} if withhold else None},
        "seal_floor": seal_floor,
        "spec_copies": spec_copies,
        "redaction": {"identities": hidden, "metas": mcount, "snapshot": scount,
                      "meta_fields_every_meta": list(META_LOSSLESS), "meta_fields_sealed_or_withheld": list(META_SENSITIVE),
                      "row_fields": list(ROW_SENSITIVE)},
        "code": {"family_pin": common.sha256_file(__file__), "families": common.sha256_file(fam.__file__)},
    }
    doc2 = dict(cfg1)
    doc2.update(
        created_utc_us=common.utc_us(), started_utc_us=started, policy=jsonable(policy), inputs=sorted(inputs, key=lambda x: x["name"]),
        registry_snapshot=dict(cfg1.get("registry_snapshot") or {}, sha256=common.sha256_bytes(snap_b), bytes=len(snap_b)),
        sealed=dict(cfg1.get("sealed") or {}, redacted_meta_sha256=sorted(
            e["name"].split(":", 1)[1] for e in inputs if e.get("role") == "meta" and e.get("redaction") == "all_sensitive")),
        pin_rules=dict(cfg1.get("pin_rules") or {}, meta_redacted_fields={"every_meta": list(META_LOSSLESS),
                                                                            "sealed_or_withheld_members": list(META_SENSITIVE)},
                       row_redacted_fields={"frozen_and_incoming_removed": list(ROW_SENSITIVE),
                                            "sealed_or_withheld_siblings_hashed": list(ROW_SENSITIVE)}),
        repin=repin, cpu_guard=guard.summary(), pin_code={"module": "dltrain.family_pin", "sha256": common.sha256_file(__file__)})
    doc2["counts"] = dict(cfg1.get("counts") or {}, inputs=len(inputs),
                          metas_redacted=sum(1 for e in inputs if e.get("role") == "meta" and e.get("redaction") == "all_sensitive"),
                          metas_lossless_redacted=mcount["lossless"], repin_s3_gets=0)
    body = common.canonical(doc2)
    # writes: the inputs first (create-only), then the in-memory check, then alias-sources.json last
    written = 0
    for rel in sorted(files):
        _write(out, rel, files[rel])
        written += 1
    guard.wait("repin-check", force=True)
    res2 = fam.run_build(out / "alias-sources.json", None, guard=guard, expect=expect, raw_config=body)
    diff = _compare_models(res1, res2)
    _need(not diff, f"the redacted inputs do not reproduce the plaintext pre-review ({diff[:5]}): STOP, alias-sources.json "
                    "not written")
    texts, undecodable = _texts_for_scan(dict(files, **{"alias-sources.json": body}))
    hits = fam.canary_scan(texts, sorted(set(res1.cans) | set(res2.cans)))
    bad = sorted(k for k, n in hits.items() if n)
    _need(not bad and not undecodable, f"the redacted generation still holds canary strings in {len(bad)} file(s) "
                                       f"({bad[:3]}) or {len(undecodable)} undecodable file(s): STOP, alias-sources.json "
                                       "not written")
    res = common.immutable_write(out / "alias-sources.json", body)
    summary = {"verdict": "PASS", "mode": "local_repin",
               "alias_sources": {"path": str(out / "alias-sources.json"), "sha256": res["sha256"], "bytes": res["bytes"]},
               "source": repin["source"], "verified": gen["checked"], "files_written": written + 1,
               "redaction": {"identities": len(hidden), "metas": mcount, "snapshot": scount,
                             "spec_copies": len(spec_copies)},
               "canary_scan": {"files": len(texts), "hits": 0, "strings": len(set(res1.cans) | set(res2.cans))},
               "withhold": len(wh["identities"]), "check": {"plaintext_vs_redacted": "equal", "counts": res2.summary["counts"]},
               "s3_requests": 0, "cpu_guard": guard.summary()}
    if receipt_path:
        common.write_receipt(receipt_path, {"schema": REPIN_RECEIPT_SCHEMA, "experiment_id": EXPERIMENT_ID,
                                            "started_utc_us": started, "utc_us": common.utc_us(), **summary,
                                            "code": repin["code"]})
    return summary


# ----------------------------------------------------------------------------------------------- build-side helpers
def load_alias_sources(path) -> dict:
    raw = _read(path, "alias-sources.json")
    doc = json.loads(raw)
    _need(doc.get("schema") == ALIAS_SOURCES_SCHEMA, "alias-sources.json: wrong schema")
    return doc


def input_index(doc: dict) -> dict:
    """name -> input entry (names are unique; see INPUT_NAMES for the per-dataset scheme)."""
    return {e["name"]: e for e in doc.get("inputs") or []}


def rehash_inputs(doc: dict, base_dir, skip_roles=(), guard=None) -> dict:
    """Re-check every pinned 'path' (external file) and 'local_path' (copy under base_dir). build STOPs ('re-pin') on
    any mismatch. skip_roles lets a caller skip heavy annotation-only roles (e.g. 'v6_parquet')."""
    base, bad, checked = Path(base_dir), [], 0
    for e in doc.get("inputs") or []:
        if e.get("role") in skip_roles:
            continue
        if guard is not None and checked % 200 == 0:
            guard.wait("rehash")
        for pk, sk, bk in (("path", "sha256", "bytes"), ("local_path", "local_sha256", "local_bytes")):
            if not e.get(pk):
                continue
            p = Path(e[pk]) if pk == "path" else base / e[pk]
            checked += 1
            if not p.is_file():
                bad.append({"name": e["name"], "field": pk, "problem": "missing"})
                continue
            if common.sha256_file(p) != e[sk] or (e.get(bk) is not None and p.stat().st_size != e[bk]):
                bad.append({"name": e["name"], "field": pk, "problem": "sha256/bytes differ"})
    snap = doc.get("registry_snapshot") or {}
    if snap.get("path"):
        p = base / snap["path"]
        if not p.is_file() or common.sha256_file(p) != snap.get("sha256"):
            bad.append({"name": "registry_snapshot", "field": "path", "problem": "missing or sha256 differs"})
    return {"checked": checked, "mismatches": bad, "ok": not bad}


# ----------------------------------------------------------------------------------------------- CLI
def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dltrain.family_pin", description="P2 step 1 (pin). Prefer `dltrain.families pin`.")
    ap.add_argument("--out", required=True, help="RUN/families/")
    ap.add_argument("--receipt", help="create-only pin receipt (RUN/receipts/P2/...)")
    ap.add_argument("--ledger", help="create-only JSONL journal of every S3 GET (RUN/receipts/P2/...)")
    ap.add_argument("--preflight", action="store_true", help="every local check and the S3 plan; 0 S3 requests, nothing written")
    a = ap.parse_args(argv)
    fam, rec = (RUN / "families").resolve(), (RUN / "receipts" / "P2").resolve()
    for path, root in ((a.out, fam), (a.receipt, rec), (a.ledger, rec)):
        if path is not None and not Path(path).resolve().is_relative_to(root):
            print(json.dumps({"verdict": "STOP", "stop": f"{path} is outside {root} (live outputs go only there)"}), file=sys.stderr)
            return 3
    try:
        summary = pin(a.out, ledger_path=None if a.preflight else a.ledger, receipt_path=None if a.preflight else a.receipt,
                      preflight=a.preflight)
    except PinStop as err:
        print(json.dumps({"verdict": "STOP", "stop": str(err)[:500]}), file=sys.stderr)
        return 3
    print(json.dumps({k: summary.get(k) for k in ("verdict", "alias_sources", "counts", "s3_totals", "s3_plan")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

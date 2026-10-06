"""P6 encodings tests (spec T21-T41): adopt, run --arm A0, manifests, verify, sample, fake-partition-map and hygiene.

Offline: synthetic epochs and JTT1 inputs, Codex release/index/packet/outcome/array objects held in a tests_p3 FakeBucket
(every GetObject passes the real s3io lane guard), a stub cpuguard (no admission waits, no slice query). Every test is a
tests_p3.fixtures_p3.P3Case, whose armed audit hook fails the test on any write outside TMPDIR; all temporary
directories are tempfile.TemporaryDirectory(dir=$TMPDIR).

Run inside the lane slice: python -m unittest discover -s tests_p6 -t . -v
"""
import ast
import collections
import gzip
import hashlib
import inspect
import io
import json
import os
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dltrain import common, cpuguard, jtt1, programs, s3io
from dltrain import encodings as E
from dltrain import metafeatures as A0
from dltrain import normalizers as N
from tests_p3 import fixtures_p3 as fx

TMPDIR = os.environ["TMPDIR"]
SRC = Path(E.__file__).resolve().parent.parent
CODEX_BUCKET = "veox-jopedime-learning-use2"
CODEX_PREFIX = "experiments/gp-teacher-v6/intake/neural-encodings-v1/"
COMPS = ("hyperion", "foundation", "target", "tabpfn35", "context2048")
UNPINNED = "rejected:component_unpinned_in_inventory"
LEGACY = "not_for_arms:legacy_projection"
DEV_ONLY = "not_for_arms:dev_only_arm"
BOUND = "bound_index_only"
TIER_I_REASONS = {"duplicate_component_rows", "component_unpinned_in_inventory", "component_pin_mismatch", "not_in_codex_packet",
                  "packet_identity_mismatch", "packet_fold_mismatch", "job_id_mismatch", "membership_mismatch",
                  "index_hash_recompute_mismatch", "rights_prohibited", "jtt1_shape_mismatch", "identity_prefix_mismatch",
                  "prepared_ref_unverifiable", "packet_manifest_mismatch", "index_fold_repeat", "membership_repeat"}

# Records file opens while armed (T36/T37: nothing is read before the slice check).
_OPENS = {"armed": False, "events": []}


def _open_hook(event, args):
    if _OPENS["armed"] and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENS["events"].append(("open", os.fsdecode(args[0])))


sys.addaudithook(_open_hook)


def h(tag: str) -> str:
    return hashlib.sha256(("p6-encodings-test:" + tag).encode()).hexdigest()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def le_u64(indices) -> str:
    """The spec formula, in pure Python: sha256(b''.join(i.to_bytes(8, 'little') for i in indices))."""
    return sha(b"".join(int(i).to_bytes(8, "little") for i in indices))


def jload(path):
    return json.loads(Path(path).read_bytes())


def rows_of(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def writable(path):
    os.chmod(path, 0o644)
    return Path(path)


# ------------------------------------------------------------------------------ Codex code, copied verbatim
def codex_canonical(x):  # custody.canonical
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def codex_sha(x):  # custody.sha
    return hashlib.sha256(x).hexdigest()


def codex_job_id(d, f, component, p):  # worker.job_id
    return codex_sha(codex_canonical(dict(schema='neural-encoding-job/v1', identity=d['identity'], manifest_ref=d['manifest_ref'], train_indices_sha256=f['train_indices_sha256'], train_row_ids_sha256=f['train_row_ids_sha256'], fold=f['fold'], component=component, model_refs=p['model_refs'], encoder_source_sha256=p['source_pins']['gpu_encoders.py']['sha256'])))


def codex_pool(rows):  # gpu_encoders.pool
    return np.concatenate([rows.mean(0), rows.std(0), np.percentile(rows, 10, axis=0), np.percentile(rows, 90, axis=0)]).astype('float32')


def codex_sampling(n):  # gpu_encoders.sampling
    p = np.random.default_rng(20260707).permutation(n)
    nc = min(768, max(1, 3 * n // 4), n - 1)
    nq = min(256, n - nc)
    return p[:nc], p[nc:nc + nq]


def codex_removal(X, y):  # worker.records(): train-only target-proxy detection
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


# ---------------------------------------------------------------------------------------------- test base
class StubGuard:
    """Counts wait()/tick() (T37); never blocks."""
    created = []

    def __init__(self, journal_path=None, **_kw):
        self.journal_path = journal_path
        self.events = []
        StubGuard.created.append(self)

    def wait(self):
        self.events.append("wait")
        return 0.0

    def tick(self):
        self.events.append("tick")
        return 0.0

    def summary(self):
        return {"checks": len(self.events), "pauses": 0, "paused_s": 0.0, "reasons": [], "rule": "stub guard (offline tests)"}


class P6Case(fx.P3Case):
    """P3Case (write guard armed, TMPDIR-only temp dirs) plus a p6-sample-* root, a stub guard, a mocked slice check and
    no S3 client unless a test hands one in."""

    def setUp(self):
        super().setUp()
        self._t6 = tempfile.TemporaryDirectory(dir=TMPDIR, prefix=E.SAMPLE_DIR_PREFIX)
        self.addCleanup(self._t6.cleanup)
        self.root = Path(self._t6.name)
        StubGuard.created = []
        self.slice = mock.Mock(return_value={})
        for target, attr, value in ((cpuguard, "Guard", StubGuard), (cpuguard, "require_slice_limits", self.slice),
                                    (E, "_read_client", mock.Mock(side_effect=AssertionError("no S3 client expected here")))):
            p = mock.patch.object(target, attr, value)
            p.start()
            self.addCleanup(p.stop)

    def use_bucket(self, bucket):
        p = mock.patch.object(E, "_read_client", lambda: fx.FakeS3Client(bucket))
        p.start()
        self.addCleanup(p.stop)

    def cli(self, *argv):
        return E.run_cli([str(a) for a in argv])


# ------------------------------------------------------------------------------------------- run fixture
def make_raw(n_train, n_test, p, seed, *, nan_y=False):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    X = rng.normal(size=(n, p))
    y = np.einsum("ij,j->i", X, rng.normal(size=p), optimize=False) + 0.5 * rng.normal(size=n)
    if nan_y:
        y[1] = np.nan
    labels = (np.arange(n_train) % 2).astype(np.uint8)
    mat = np.column_stack([X, y]).astype("<f8")
    return jtt1.MAGIC + struct.pack("<IIIB", n_train, n_test, p, 2) + labels.tobytes() + mat.tobytes()


RUN_SPECS = (("tf1", "train_fit", "A", "frozen", 0), ("tf2", "train_fit", "B", "incoming", 1), ("tf3", "train_fit", "C", "frozen", 2),
             ("dv1", "dev", "D", "frozen", 0), ("dv2", "dev", "E", "incoming", 3), ("dg1", "diag", "A", "frozen", 1),
             ("hd1", "heldout", "F", "frozen", 0), ("hd2", "heldout", "F", "frozen", 1), ("sl1", "sealed", "G", "frozen", 0),
             ("pr1", "prohibited", "H", "incoming", 0))


class RunWorld:
    """8 admitted contexts across the 4 partitions (3 train_fit, 2 dev, 1 diag, 2 heldout) plus 1 sealed and 1
    prohibited, one epoch-excluded context; JTT1 inputs in a FakeBucket and (optionally) in the local cache."""

    def __init__(self, root: Path, *, nan_tag=None, cache=True, role_override=None, name="run", raw_override=None):
        self.root = root / f"world-{name}"
        self.root.mkdir()
        self.cache = self.root / "jtt1"
        self.cache.mkdir()
        self.bucket = fx.FakeBucket()
        self.contexts, self.pmap, self.raws, mems = [], {}, {}, []
        self.tag = {}
        for i, (tag, role, fam, cohort, fold) in enumerate(RUN_SPECS):
            role = (role_override or {}).get(tag, role)
            raw = (raw_override or {}).get(tag) or make_raw(40, 10, 4, seed=i, nan_y=(tag == nan_tag))
            gz = gzip.compress(raw, compresslevel=3, mtime=0)
            key = f"experiments/gp-teacher-v6/objects/inputs/sha256/{sha(raw)}.jtt1.gz"
            vid = self.bucket.preload(key, gz)
            ref = {"bucket": s3io.BUCKET, "key": key, "version_id": vid, "sha256": sha(gz), "bytes": len(gz),
                   "raw_sha256": sha(raw), "raw_bytes": len(raw), "compression": "gzip-3-mtime0"}
            if cache:
                (self.cache / f"{sha(raw)}.jtt1.gz").write_bytes(gz)
            dh = h("ds:" + fam)[:16]
            cid = h("ctx:" + tag)
            c = {"context_sha256": cid, "status": "included", "cohort": cohort, "fold": fold, "dataset_hash": dh,
                 "sample_identity_sha256": h("sid:" + fam), "view_id": f"{dh}__r0__f{fold}", "input_ref": ref,
                 "membership_ref": {"bucket": s3io.BUCKET, "key": f"experiments/gp-teacher-v6/objects/memberships/sha256/{h('mk' + tag)}.json",
                                    "version_id": "vm-" + tag, "sha256": h("mem:" + tag), "bytes": 1000}}
            if cohort == "incoming":
                c["item_id"] = h("item:" + tag)
            self.contexts.append(c)
            self.raws[cid] = raw
            self.tag[tag] = cid
            self.pmap[cid] = {"family_id": "fam-" + fam, "split_role": role}
            mems.append({"context_sha256": cid, "dataset_hash": dh, "outer_fold": fold, "outer_repeat": 0,
                         "sample_identity_sha256": c["sample_identity_sha256"], "outer_train_indices_sha256": h("tri:" + tag),
                         "outer_train_rows": 40, "outer_test_indices_sha256": h("tei:" + tag), "outer_test_rows": 10,
                         "membership_sha256": c["membership_ref"]["sha256"]})
        excluded = dict(self.contexts[0], context_sha256=h("ctx:excluded"), status="excluded", exclusion_reason="withdrawn")
        self.epoch = {"schema": "jope.dl-corpus-epoch/v1", "experiment_id": E.EXPERIMENT_ID, "epoch": 0,
                      "contexts": self.contexts + [excluded], "leakage_quarantine": {"source_view_ids": []}}
        self.epoch_path = self.root / "epoch.json"
        self.epoch_path.write_bytes(common.canonical(self.epoch))
        self.map_path = self.root / "map.json"
        self.map_path.write_bytes(common.canonical(self.pmap))
        self.mem_path = self.root / "memberships.jsonl"
        self.mem_path.write_bytes(b"".join(json.dumps(m, sort_keys=True).encode() + b"\n" for m in mems))
        self.by_id = {c["context_sha256"]: c for c in self.contexts}

    def write_map(self, pmap, name):
        path = self.root / name
        path.write_bytes(common.canonical(pmap))
        return path

    def run_args(self, out, *extra, budget=0, map_path=None):
        return ["run", "--arm", "A0", "--epoch", self.epoch_path, "--partition-map", map_path or self.map_path,
                "--partition-map-kind", "fake", "--jtt1-cache", self.cache, "--jtt1-get-budget", budget, "--out", out,
                "--memberships", self.mem_path, *extra]

    def verify_args(self, out, *flags, map_path=None):
        return ["verify", "--arm", "A0", "--out", out, "--epoch", self.epoch_path, "--partition-map", map_path or self.map_path,
                "--jtt1-cache", self.cache, *flags]

    def admitted(self):
        return [c for c, v in self.pmap.items() if v["split_role"] in E.PARTITIONS]


# ----------------------------------------------------------------------------------------- adopt fixture
def model_refs():
    names = ("hyperion_v5_arm_b_identity", "hyperion_v5_arm_b_runtime_manifest", "starlight_core_regressor_identity",
             "target_encoder_fused_v1_identity", "target_encoder_fused_v1_metadata", "context_transforms")
    return {n: {"bucket": "veox-jopedime-models-use2", "key": f"blobs/sha256/{h(n)}", "version_id": "v-" + n[:12],
                "bytes": 1000 + i, "sha256": h(n), "server_checksum": "FULL_OBJECT"} for i, n in enumerate(names)}


LICENSES = {"hyperion": {"class": "eligible_non_open", "redistribution": "permitted", "policy_eligible": True},
            "foundation": {"class": "restricted", "spdx": "LicenseRef-Jain-ThirdParty-Provenance-Pending", "redistribution": "prohibited",
                           "policy_eligible": False},
            "target": {"class": "eligible_non_open", "redistribution": "permitted", "policy_eligible": True}}


def inventory_for(mr, tamper=None):
    """The encoding-inventory.json shape: pins plus the licence blocks of the real inventory (context_transforms has none)."""
    inv = {"schema": "jope.dl-encoding-inventory/v1",
           "components": [{"id": "hyperion_v5_arm_b", "identity": dict(mr["hyperion_v5_arm_b_identity"]),
                           "runtime_manifest": dict(mr["hyperion_v5_arm_b_runtime_manifest"]), "license": dict(LICENSES["hyperion"])},
                          {"id": "hyperion_v7", "identity": {"bucket": "b", "key": "k", "version_id": "v", "sha256": h("v7"), "bytes": 1}},
                          {"id": "starlight_core_regressor", "identity": dict(mr["starlight_core_regressor_identity"]),
                           "license": dict(LICENSES["foundation"])},
                          {"id": "target_encoder_fused_v1", "identity": dict(mr["target_encoder_fused_v1_identity"]),
                           "metadata": dict(mr["target_encoder_fused_v1_metadata"]), "license": dict(LICENSES["target"])}],
           "fitted_normalizers": [{"id": "context_transforms_lineage_b", "identity": dict(mr["context_transforms"])}]}
    if tamper == "hyperion_runtime_manifest":
        inv["components"][0]["runtime_manifest"]["version_id"] = "another-version"
    return inv


class AdoptWorld:
    """Contexts with Codex outputs for all five components. Each add() names a role, the --contexts subset membership
    and mutations; finalize() writes the epoch (with a sha-pinned prepared.json per context), memberships, partition
    map, packet, inventory, index, release (a terminal release naming the index when terminal, else a launch release)
    and an exact-VersionId index fetch receipt (local files and FakeBucket objects). Missing outputs: missing=(...) has
    no index row (pending in a non-terminal index, unaccounted in a terminal one), failed=(...) has a non-complete row,
    retired=True leaves only Codex's null-identity row of a retired fold."""

    def __init__(self, root: Path, *, name, terminal=True, tamper_pin=None, qualified=True, retired_slots=None):
        self.root = root / f"world-{name}"
        self.root.mkdir()
        self.cache = self.root / "jtt1"
        self.cache.mkdir()
        self.bucket = fx.FakeBucket()
        self.terminal, self.tamper_pin, self.qualified, self.retired_slots = terminal, tamper_pin, qualified, retired_slots
        self.mr = model_refs()
        self.source_pins = {"gpu_encoders.py": {"bytes": 9818, "sha256": h("gpu_encoders.py")},
                            "worker.py": {"bytes": 13410, "sha256": h("worker.py")}}
        self.specs, self.cid, self.expect = [], {}, {}

    def add(self, tag, role="train_fit", *, cohort="frozen", fold=0, subset=False, expect=None, **mut):
        self.specs.append({"tag": tag, "role": role, "cohort": cohort, "fold": fold, "subset": subset, "mut": mut})
        self.cid[tag] = h("ctx:" + tag)
        self.expect[tag] = expect or {}

    def finalize(self, subset_order=None, unmapped=(), extra_null_row=False):
        ctxs, entries, mems, pmap, qviews, built = [], [], [], {}, [], []
        for i, s in enumerate(self.specs):
            tag, m = s["tag"], s["mut"]
            seed = 1000 + i
            rng = np.random.default_rng(seed)
            X = rng.normal(size=(50, 4))
            y = np.einsum("ij,j->i", X, rng.normal(size=4), optimize=False) + 0.3 * rng.normal(size=50)
            if m.get("proxy"):
                X[:, 2] = 3.0 * y + 1.0
            data = np.column_stack([X, y]).astype("<f8")
            tr = sorted(random.Random(seed).sample(range(50), 40))
            te = [j for j in range(50) if j not in set(tr)]
            test_block = data[te]
            if m.get("jtt1_extra_test_row"):
                test_block = np.vstack([test_block, data[te[:1]]])
            labels = (np.arange(40) % 2).astype(np.uint8)
            raw = jtt1.MAGIC + struct.pack("<IIIB", 40, len(test_block), 4, 2) + labels.tobytes() + data[tr].tobytes() + test_block.tobytes()
            gz = gzip.compress(raw, compresslevel=3, mtime=0)
            key = f"experiments/gp-teacher-v6/objects/inputs/sha256/{sha(raw)}.jtt1.gz"
            vid = self.bucket.preload(key, gz)
            ref = {"bucket": s3io.BUCKET, "key": key, "version_id": vid, "sha256": sha(gz), "bytes": len(gz),
                   "raw_sha256": sha(raw), "raw_bytes": len(raw), "compression": "gzip-3-mtime0"}
            if not m.get("uncached"):
                (self.cache / f"{sha(raw)}.jtt1.gz").write_bytes(gz)
            dh, sid = h("ds:" + tag)[:16], h("sid:" + tag)
            identity = [dh, (h("canon:" + tag) if m.get("identity_prefix_wrong") else dh + h("canon:" + tag)[16:]), sid]
            cid = self.cid[tag]
            fold = {"fold": s["fold"], "repeat": 0, "test_indices_sha256": le_u64(te), "test_rows": 10, "train_indices": list(tr),
                    "train_indices_sha256": le_u64(tr), "train_row_ids_sha256": h("rowids:" + tag), "train_rows": 40}
            if m.get("bad_train_indices"):
                fold["train_indices"] = tr[:-2] + [tr[-1], tr[-2]]
            c = {"context_sha256": cid, "status": "included", "cohort": s["cohort"], "fold": s["fold"], "dataset_hash": dh,
                 "sample_identity_sha256": sid, "view_id": f"{dh}__r0__f{s['fold']}", "input_ref": ref,
                 "membership_ref": {"bucket": s3io.BUCKET, "key": f"experiments/gp-teacher-v6/objects/memberships/sha256/{h('mk' + tag)}.json",
                                    "version_id": "vm", "sha256": h("mem:" + tag), "bytes": 1000}}
            if s["cohort"] == "incoming":
                c["item_id"] = h("item:" + tag)
            mem = {"context_sha256": cid, "dataset_hash": dh, "outer_fold": s["fold"], "outer_repeat": 0, "sample_identity_sha256": sid,
                   "outer_train_indices_sha256": fold["train_indices_sha256"], "outer_train_rows": 40,
                   "outer_test_indices_sha256": fold["test_indices_sha256"], "outer_test_rows": 10,
                   "membership_sha256": c["membership_ref"]["sha256"]}
            if m.get("membership_test_sha_wrong"):
                mem["outer_test_indices_sha256"] = h("wrong-test:" + tag)
            if m.get("membership_repeat"):
                mem["outer_repeat"] = 1
            if m.get("quarantined"):
                mem["sample_identity_sha256"] = h("parent-sid:" + tag)
                qviews.append(f"{dh}__r0__f0")
                c["view_id"] = f"{dh}__r0__f{s['fold']}__clean_target_leak_v1"
            if m.get("membership_disagrees"):
                mem["sample_identity_sha256"] = h("other-sid:" + tag)
            if tag not in unmapped:
                pmap[cid] = {"family_id": "fam-" + dh, "split_role": s["role"]}
            manifest_ref = {"bucket": "veox-jopedime-datasets-use2", "bytes": 3000, "key": f"research/regression-2k/v1/datasets/{dh}/{sid}/manifest.json",
                            "sha256": h("manifest:" + tag), "version_id": "vmf"}
            # the epoch's sha-pinned prepared.json names the source manifest that Codex's packet manifest_ref must equal
            source_manifest = dict(manifest_ref, sha256=h("other-manifest:" + tag)) if m.get("manifest_mismatch") else dict(manifest_ref)
            prepared = common.canonical({"schema": "jope.regression-research-gp-context-preparation/v1", "view_id": c["view_id"],
                                         "source_manifest": dict(source_manifest, reused=False)})
            (self.root / "prepared").mkdir(exist_ok=True)
            ppath = self.root / "prepared" / f"{tag}.json"
            ppath.write_bytes(prepared)
            if not m.get("prepared_absent"):
                c["prepared_ref"] = {"path": str(ppath), "sha256": sha(prepared), "bytes": len(prepared)}
            d = {"cohort": s["cohort"], "identity": identity, "item_id": h("item:" + tag),
                 "license": {"status": "prohibited" if m.get("rights_prohibited") else "unknown"}, "manifest_ref": manifest_ref,
                 "folds": [json.loads(json.dumps(fold))], "shape": {"columns": 5, "features": 4, "sample_rows": 50, "source_rows": 50}}
            if m.get("packet_cohort_wrong"):
                d["cohort"] = "incoming" if s["cohort"] == "frozen" else "frozen"
            if m.get("packet_fold_wrong"):
                d["folds"][0]["train_row_ids_sha256"] = h("pfold:" + tag)
            if not m.get("packet_absent"):
                entries.append(d)
            ctxs.append(c)
            mems.append(mem)
            built.append({"spec": s, "fold": fold, "identity": identity, "manifest_ref": manifest_ref, "X": data[tr][:, :-1],
                          "y": data[tr][:, -1], "seed": seed})
        self.packet = {"schema": "jope.regression-neural-encoding-packet/v1", "components": list(COMPS), "datasets": entries,
                       "model_refs": self.mr, "source_pins": self.source_pins, "limits": {"attempts_per_component": 1}}
        pb = codex_canonical(self.packet)
        self.packet_path = self.root / "packet.json"
        self.packet_path.write_bytes(pb)
        self.packet_ref = {"bucket": CODEX_BUCKET, "bytes": len(pb), "key": CODEX_PREFIX + f"objects/packets/sha256/{sha(pb)}.json",
                           "sha256": sha(pb), "version_id": "vpacket"}
        outputs = []
        for b in built:
            outputs += self._outputs(b)
        if extra_null_row:            # a null-identity row whose job_id names no packet slot
            outputs.append({"array_ref": None, "claim_ref": None, "component": "hyperion", "fold": None, "identity": None,
                            "job_id": h("unknown-job"), "outcome_ref": None, "state": "absent_or_retired_before_dispatch"})
        self.index = {"all_shards_terminal": self.terminal, "errors": [], "optimizer_training": False, "outputs": outputs,
                      "packet_ref": self.packet_ref, "schema": "jope.neural-encoding-output-index/v1", "utc_us": 1791184334301465}
        ib = codex_canonical(self.index)
        (self.root / "codex-index").mkdir()
        self.index_path = self.root / "codex-index" / f"{sha(ib)}.json"
        self.index_path.write_bytes(ib)
        self.index_ref = {"bucket": CODEX_BUCKET, "key": CODEX_PREFIX + f"objects/indexes/sha256/{sha(ib)}.json", "version_id": "vidx",
                          "sha256": sha(ib), "bytes": len(ib)}
        self.index_ref_path = self.root / "index-ref.json"
        self.index_ref_path.write_text(json.dumps(self.index_ref))
        self.retired_component_slots = sum(len(COMPS) for s in self.specs if s["mut"].get("retired"))
        self.release = self.terminal_release() if self.terminal else self.launch_release()
        self.release_ref, self.release_ref_path = self.publish_release(self.release, "release-ref.json")
        self.fetch_receipt_path = self.write_fetch_receipt(self.index_ref, self.release_ref, "fetch-receipt.json")
        inv = codex_canonical(inventory_for(self.mr, self.tamper_pin))
        self.inventory_path = self.root / "inventory.json"
        self.inventory_path.write_bytes(inv)
        self.inventory_sha = sha(inv)
        excluded = dict(ctxs[0], context_sha256=h("ctx:excluded"), status="excluded", exclusion_reason="withdrawn")
        self.epoch = {"schema": "jope.dl-corpus-epoch/v1", "experiment_id": E.EXPERIMENT_ID, "epoch": 0, "contexts": ctxs + [excluded],
                      "leakage_quarantine": {"source_view_ids": qviews}}
        self.epoch_path = self.root / "epoch.json"
        self.epoch_path.write_bytes(common.canonical(self.epoch))
        self.mem_path = self.root / "memberships.jsonl"
        self.mem_path.write_bytes(b"".join(json.dumps(m, sort_keys=True).encode() + b"\n" for m in mems))
        self.pmap = pmap
        self.map_path = self.root / "map.json"
        self.map_path.write_bytes(common.canonical(pmap))
        order = subset_order or [s["tag"] for s in self.specs if s["subset"]]
        self.subset = [self.cid[t] for t in order]
        self.contexts_path = self.root / "contexts.json"
        self.contexts_path.write_text(json.dumps({"contexts": self.subset}))
        self.by_cid = {self.cid[s["tag"]]: s for s in self.specs}
        return self

    def launch_release(self, **over):
        """Codex's launch release form (a8382194...): index_ref names no adopted index."""
        doc = {"schema": "jope.neural-encoding-release/v1", "packet_ref": self.packet_ref, "docs": {"README.md": {"sha256": h("readme")}},
               "custody_ref": {"bucket": CODEX_BUCKET, "key": CODEX_PREFIX + "custody/x.json", "sha256": h("custody"), "bytes": 1, "version_id": "vc"},
               "index_ref": None, "progress_ref": {"bucket": CODEX_BUCKET, "key": CODEX_PREFIX + "progress/x.json", "sha256": h("progress"),
                                                   "bytes": 1, "version_id": "vp"},
               "readonly_dashboard": "http://127.0.0.1:8765/", "optimizer_training": False}
        doc.update(over)
        return doc

    def terminal_release(self, **over):
        """Codex's terminal release form (cab46d85...): it names the terminal index, the original launch release and the
        qualification flags."""
        q = self.qualified
        summary = {"schema": "jope.encoding-final-dependencies/v1", "index_ref": dict(self.index_ref), "packet_ref": dict(self.packet_ref),
                   "all_output_array_claim_index_dependencies_qualified": True, "all_shards_terminal_metadata_qualified": True,
                   "context_gp_join_qualified": q, "raw_data_independence_qualified": q, "rights_family_adoption_qualified": q,
                   "retired_component_slots": self.retired_component_slots if self.retired_slots is None else self.retired_slots,
                   "retired_folds": self.retired_component_slots // len(COMPS), "optimizer_training": False}
        doc = {"schema": "jope.neural-encoding-terminal-release/v1", "state": "TERMINAL_DEPENDENCIES_VERIFIED", "adoption_qualified": q,
               "custody_ref": {"bucket": CODEX_BUCKET, "key": CODEX_PREFIX + "terminal-v1/custody/x.json", "sha256": h("tcustody"), "bytes": 1,
                               "version_id": "vtc"},
               "index_ref": dict(self.index_ref), "original_launch_release_ref": dict(E.PLAN_LAUNCH_RELEASE_REF),
               "packet_ref": self.packet_ref, "optimizer_training": False, "summary": summary, "utc_us": 1791196638968378}
        doc.update(over)
        return doc

    def publish_release(self, doc, name):
        """(release_ref, local ref file) of a release preloaded into the FakeBucket."""
        rb = codex_canonical(doc)
        rkey = CODEX_PREFIX + f"releases/{sha(rb)}.json"
        ref = {"bucket": CODEX_BUCKET, "key": rkey, "version_id": self.bucket.preload(rkey, rb), "sha256": sha(rb), "bytes": len(rb)}
        path = self.root / name
        path.write_text(json.dumps(ref))
        return ref, path

    def write_fetch_receipt(self, index_ref, release_ref, name):
        """An exact-VersionId fetch receipt (jope.dl-codex-terminal-fetch/v1 form)."""
        path = self.root / name
        path.write_text(json.dumps({"schema": "jope.dl-codex-terminal-fetch/v1", "refs": {"index": index_ref, "release": release_ref},
                                    "local": {"index": {"bytes": index_ref["bytes"], "fetched": True, "path": str(self.index_path)}},
                                    "utc_us": 1791196999880210}))
        return path

    def _outputs(self, b):
        m = b["spec"]["mut"]
        X, y = b["X"], b["y"]
        kept0, removed0 = codex_removal(X, y)
        out = []
        fold = b["fold"]
        if m.get("retired"):          # Codex's retired fold: one null-identity row (hyperion), no row for the others
            jid = codex_job_id({"identity": b["identity"], "manifest_ref": b["manifest_ref"]}, fold, "hyperion", self.packet)
            return [{"array_ref": None, "claim_ref": None, "component": "hyperion", "fold": None, "identity": None, "job_id": jid,
                     "outcome_ref": None, "state": "absent_or_retired_before_dispatch"}]
        for ci, comp in enumerate(COMPS):
            if comp in m.get("missing", ()):
                continue
            if comp in m.get("failed", ()):
                out.append({"component": comp, "fold": fold, "identity": b["identity"], "state": "failed",
                            "job_id": codex_job_id({"identity": b["identity"], "manifest_ref": b["manifest_ref"]}, fold, comp, self.packet)})
                continue
            cm = m.get(comp, {})
            kept, removed = cm.get("kept", kept0), cm.get("removed", removed0)
            X32, y32 = X[:, kept].astype("float32"), y.astype("float32")
            n = 40
            ctx_i, qry_i = codex_sampling(n)
            rng = np.random.default_rng(b["seed"] * 10 + ci)
            if comp == "hyperion":
                rows = rng.random((n, 128)).astype("float32")
                arrays = {"rows": rows, "vector": codex_pool(rows), "feature_q01": np.percentile(X32, 1, axis=0).astype("float32"),
                          "feature_q99": np.percentile(X32, 99, axis=0).astype("float32")}
                prng = np.random.default_rng(20260623)
                perms = [{"rows": list(range(n)), "columns": list(range(len(kept)))}]
                perms += [{"rows": prng.permutation(n).tolist(), "columns": prng.permutation(len(kept)).tolist()} for _ in range(2)]
                recipe = {"permutations": perms, "scales": [4, 7, 11, 15], "sweeps": 3, "batch": 256, "aggregation": "cell mean",
                          "recipe": "gpu-v5-row-sweep-pcg-v1", "native_rng_equivalence_qualified": False}
            elif comp == "foundation":
                rows = rng.normal(size=(len(qry_i), 512)).astype("float32")
                lo = X32.min(0)
                span = X32.max(0) - lo
                span = np.where(span > 1e-12, span, 1)
                arrays = {"rows": rows, "vector": codex_pool(rows), "feature_min": lo, "feature_span": span}
                cy = y32[ctx_i]
                recipe = {"context_local_indices": ctx_i.tolist(), "query_local_indices": qry_i.tolist(), "context_y_mean": float(cy.mean()),
                          "context_y_std": float(cy.std()) + 1e-8, "seed": 20260707, "query_chunk": 256, "provider": "Starlight"}
            elif comp == "target":
                arrays = {"vector": rng.normal(size=512).astype("float32")}
                lo = float(y32.min())
                recipe = {"classical_local_indices": [[0, 1]], "neural_local_indices": [[0]], "train_target_min": lo,
                          "train_target_span": float(y32.max() - lo), "seed": 17, "structural_zero_dimensions": 376}
            elif comp == "tabpfn35":
                rows = rng.normal(size=(len(qry_i), 1024)).astype("float32")
                arrays = {"rows": rows, "vector": rng.normal(size=2048).astype("float32")}
                recipe = {"context_local_indices": ctx_i.tolist(), "query_local_indices": qry_i.tolist(), "seed": 1729}
            else:
                arrays = {"vector": rng.normal(size=2048).astype("float32"), "foundation_projected": rng.normal(size=1024).astype("float32")}
                recipe = {"parent_refs": {}, "normalizer_ref": dict(self.mr["context_transforms"]), "training_holdout_qualified": False}
            meta = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}
            if "npz_mutate" in cm:
                cm["npz_mutate"](arrays)
            buf = io.BytesIO()
            np.savez_compressed(buf, **arrays)
            npz = cm.get("npz_bytes", buf.getvalue())
            akey = CODEX_PREFIX + f"objects/arrays/sha256/{sha(npz)}.npz"
            array_ref = {"bucket": CODEX_BUCKET, "bytes": len(npz), "key": akey, "sha256": sha(npz), "version_id": self.bucket.preload(akey, npz)}
            if cm.get("array_sha_wrong"):
                array_ref["sha256"] = h("wrong-array:" + b["spec"]["tag"])
            fold = b["fold"]
            jid = codex_job_id({"identity": b["identity"], "manifest_ref": b["manifest_ref"]}, fold, comp, self.packet)
            claim_ref = {"bucket": CODEX_BUCKET, "bytes": 500, "key": CODEX_PREFIX + f"jobs/{jid}/claim.json", "sha256": h("claim:" + jid),
                         "version_id": "vclaim"}
            binding = {"removed_target_feature_indices": removed, "kept_feature_indices": kept,
                       "selection_fit": cm.get("selection_fit", "outer_train_only"),
                       "outer_train_indices_sha256": fold["train_indices_sha256"],
                       "outer_train_row_ids_sha256": cm.get("row_ids", fold["train_row_ids_sha256"])}
            if "meta_mutate" in cm:
                cm["meta_mutate"](meta)
            if "recipe_mutate" in cm:
                cm["recipe_mutate"](recipe)
            outcome = {"schema": "jope.regression-neural-encoding-outcome/v1", "job_id": jid, "state": "complete", "component": comp,
                       "identity": b["identity"], "fold": fold, "packet_ref": self.packet_ref, "claim_ref": claim_ref, "array_ref": array_ref,
                       "arrays": meta, "recipe": recipe, "input_binding": binding, "device": "cuda:0", "gpu": {"uuid": "GPU-test"},
                       "host": "xbabe1", "pid": 1, "starttick": "1", "started_utc_us": 1, "finished_utc_us": 2,
                       "training_admitted": cm.get("training_admitted", False), "benchmark_fit": False, "aws_lambda_calls": 0}
            if cm.get("outcome_claim_ref_wrong"):
                outcome["claim_ref"] = dict(claim_ref, sha256=h("other-claim"))
            ob = codex_canonical(outcome)
            okey = CODEX_PREFIX + f"jobs/{jid}/outcome.json"
            outcome_ref = {"bucket": CODEX_BUCKET, "bytes": len(ob), "key": okey, "sha256": sha(ob), "version_id": self.bucket.preload(okey, ob)}
            if cm.get("outcome_sha_wrong"):
                outcome_ref["sha256"] = h("wrong-outcome:" + b["spec"]["tag"])
            row_fold = {k: v for k, v in fold.items() if k != "repeat"} if cm.get("no_repeat") else fold
            row = {"array_ref": array_ref, "claim_ref": claim_ref, "component": comp, "fold": row_fold, "identity": b["identity"],
                   "job_id": h("wrong-job") if cm.get("job_id_wrong") else jid, "outcome_ref": outcome_ref, "state": "complete"}
            out.append(row)
            if cm.get("duplicate"):
                out.append(dict(row))
        return out

    def args(self, out, *, dry_run, budget=1000, release_budget=1, jtt1_budget=0, content=True, contexts=None, index_path=None,
             index_ref_path=None, packet_path=None, inventory_sha=None, release_ref_path=None, fetch_receipt=True, kind="fake",
             p3=None, waiver=None, map_path=None):
        """adopt argv. contexts defaults to the dry run (production refuses --contexts); fetch_receipt True passes this
        world's exact-VersionId index fetch receipt (or a given path); p3 is a make_p3_inputs() dict."""
        contexts = dry_run if contexts is None else contexts
        a = ["adopt", "--release-ref", release_ref_path or self.release_ref_path, "--index", index_path or self.index_path,
             "--index-ref", index_ref_path or self.index_ref_path, "--packet", packet_path or self.packet_path,
             "--inventory", self.inventory_path, "--inventory-sha256", inventory_sha or self.inventory_sha, "--epoch", self.epoch_path,
             "--memberships", self.mem_path, "--partition-map", map_path or self.map_path, "--partition-map-kind", kind, "--out", out,
             "--release-get-budget", release_budget, "--outcome-get-budget", budget, "--jtt1-cache", self.cache,
             "--jtt1-get-budget", jtt1_budget]
        if contexts:
            a += ["--contexts", self.contexts_path]
        if dry_run:
            a.append("--dry-run")
        if content:
            a.append("--content-checks")
        if fetch_receipt:
            a += ["--index-fetch-receipt", self.fetch_receipt_path if fetch_receipt is True else fetch_receipt]
        for flag, key in (("--holdout", "holdout"), ("--holdout-binding", "binding"), ("--p3-receipt", "p3"), ("--p5-receipt", "p5")):
            if p3 and p3.get(key):
                a += [flag, p3[key]]
        if waiver:
            a += ["--owner-waiver", waiver]
        return a

    def expected(self, *, production):
        """(cid, component) -> final status from each add()'s defaults and overrides. Production processes every included
        context at tiers O and A; a dry run only the --contexts subset (hyperion and foundation). A heldout target row is
        not_for_arms:dev_only_arm (TOL-09); a retired fold is residual for every component."""
        out = {}
        for s in self.specs:
            tag, role = s["tag"], s["role"]
            if role in E.EXCLUDED_ROLES:
                base = {k: f"excluded_by_role:{role}" for k in COMPS}
            elif s["mut"].get("retired"):
                base = {k: "residual:absent_or_retired_before_dispatch" for k in COMPS}
            elif production or s["subset"]:
                base = {"hyperion": "adopted", "foundation": "adopted", "target": "adopted" if production else BOUND,
                        "tabpfn35": UNPINNED, "context2048": LEGACY}
            else:
                base = {"hyperion": BOUND, "foundation": BOUND, "target": BOUND, "tabpfn35": UNPINNED, "context2048": LEGACY}
            if role == "heldout" and base["target"] in (BOUND, "adopted"):
                base["target"] = DEV_ONLY
            base.update(self.expect[tag])
            for k in COMPS:
                out[(self.cid[tag], k)] = base[k]
        return out


def counts_of(expected):
    out = {k: collections.Counter() for k in COMPS}
    for (_cid, k), st in expected.items():
        out[k][st] += 1
    return {k: dict(sorted(v.items())) for k, v in out.items()}


def tier_i_of(status):
    if status.startswith(("excluded_by_role:", "residual:", "pending:", "not_for_arms:")) or status == BOUND:
        return status
    if status.startswith("rejected:") and status.split(":", 1)[1] in TIER_I_REASONS:
        return status
    return BOUND


def four(status):
    """The same status for hyperion, foundation, target and context2048 (tabpfn35 is rejected earlier as unpinned)."""
    return {"hyperion": status, "foundation": status, "target": status, "context2048": status}


def bump(field, delta):
    """An npz_mutate hook: add delta to the first element of one stored array."""
    def f(arrays):
        arr = arrays[field].copy()
        arr.flat[0] += np.float32(delta)
        arrays[field] = arr
    return f


def build_main_world(root, *, name="P", terminal=True):
    """The T21 production world: every tier-I/O/A rejection reason, residuals, role exclusions and adoptions."""
    w = AdoptWorld(root, name=name, terminal=terminal)
    w.add("ok1", "train_fit", subset=True, proxy=True)
    w.add("ok2", "dev", cohort="incoming", fold=1, subset=True)
    w.add("held", "heldout", fold=2, subset=True)
    w.add("diag", "diag", fold=3)
    w.add("sealed", "sealed", expect={k: "excluded_by_role:sealed" for k in COMPS})
    w.add("prohib", "prohibited", cohort="incoming", expect={k: "excluded_by_role:prohibited" for k in COMPS})
    w.add("quar", "train_fit", subset=True, quarantined=True, expect={k: "residual:feature_sanitized_view" for k in COMPS})
    w.add("nofound", "dev", failed=("foundation",), expect={"foundation": "residual:no_complete_output"})
    w.add("dup", "train_fit", hyperion={"duplicate": True}, expect={"hyperion": "rejected:duplicate_component_rows"})
    w.add("nopacket", "train_fit", packet_absent=True, expect=four("rejected:not_in_codex_packet"))
    w.add("pcohort", "train_fit", packet_cohort_wrong=True, expect=four("rejected:packet_identity_mismatch"))
    w.add("pfold", "train_fit", packet_fold_wrong=True, expect=four("rejected:packet_fold_mismatch"))
    w.add("jobid", "train_fit", hyperion={"job_id_wrong": True}, expect={"hyperion": "rejected:job_id_mismatch"})
    w.add("mem", "train_fit", membership_test_sha_wrong=True, expect=four("rejected:membership_mismatch"))
    w.add("idx", "train_fit", bad_train_indices=True, expect=four("rejected:index_hash_recompute_mismatch"))
    w.add("rights", "train_fit", rights_prohibited=True, expect=four("rejected:rights_prohibited"))
    w.add("shape", "train_fit", subset=True, jtt1_extra_test_row=True, expect=four("rejected:jtt1_shape_mismatch"))
    w.add("osha", "train_fit", subset=True, hyperion={"outcome_sha_wrong": True}, expect={"hyperion": "rejected:outcome_sha_mismatch"})
    w.add("ofield", "train_fit", subset=True, hyperion={"training_admitted": True}, foundation={"outcome_claim_ref_wrong": True},
          expect={"hyperion": "rejected:outcome_field_mismatch:training_admitted", "foundation": "rejected:outcome_field_mismatch:claim_ref"})
    w.add("sel", "train_fit", subset=True, hyperion={"selection_fit": "all_rows"}, expect={"hyperion": "rejected:selection_fit_not_outer_train_only"})
    w.add("ib", "train_fit", subset=True, foundation={"row_ids": h("wrong-row-ids")}, expect={"foundation": "rejected:input_binding_hash_mismatch"})
    w.add("part", "train_fit", subset=True, hyperion={"kept": [0, 1, 2, 3], "removed": [3]}, expect={"hyperion": "rejected:feature_partition_invalid"})
    w.add("acon", "train_fit", subset=True, hyperion={"meta_mutate": lambda m: m["vector"].update(shape=[511])},
          expect={"hyperion": "rejected:array_contract_violation:vector"})
    w.add("recipe", "train_fit", subset=True, hyperion={"recipe_mutate": lambda r: r.update(sweeps=2)},
          foundation={"recipe_mutate": lambda r: r.update(seed=1)},
          expect={"hyperion": "rejected:recipe_contract_violation:sweeps", "foundation": "rejected:recipe_contract_violation:seed"})
    w.add("kept", "train_fit", subset=True, hyperion={"kept": [0, 1, 2], "removed": [3]},
          expect={"hyperion": "rejected:content_check_failed:kept_features"})
    w.add("ymom", "train_fit", subset=True, foundation={"recipe_mutate": lambda r: r.update(context_y_mean=r["context_y_mean"] + 1.0)},
          expect={"foundation": "rejected:content_check_failed:context_y_moments"})
    w.add("tspan", "train_fit", subset=True, target={"recipe_mutate": lambda r: r.update(train_target_min=r["train_target_min"] - 1.0)},
          expect={"target": "rejected:content_check_failed:target_min_span"})
    w.add("asha", "train_fit", subset=True, hyperion={"array_sha_wrong": True}, expect={"hyperion": "rejected:array_sha_mismatch"})
    w.add("aload", "train_fit", subset=True, hyperion={"npz_bytes": b"not an npz archive"}, expect={"hyperion": "rejected:array_load_refused"})
    w.add("aquant", "train_fit", subset=True, hyperion={"npz_mutate": bump("feature_q01", 0.5)},
          expect={"hyperion": "rejected:content_check_failed:feature_quantiles"})
    w.add("apool", "train_fit", subset=True, hyperion={"npz_mutate": bump("vector", 0.5)},
          expect={"hyperion": "rejected:content_check_failed:pooled_vector"})
    w.add("fminspan", "train_fit", subset=True, foundation={"npz_mutate": bump("feature_min", -0.5)},
          expect={"foundation": "rejected:content_check_failed:feature_min_span"})
    return w.finalize()


def build_small_world(root, *, name, terminal=False, tamper_pin=None, uncached_tag=None, s2_extra=False, unmapped=(), qualified=True):
    """dB (dev) and dA (train_fit) in the subset (sample order dB, dA), dC without a complete foundation output (no row in
    a non-terminal index, a failed row in a terminal one), dS sealed."""
    w = AdoptWorld(root, name=name, terminal=terminal, tamper_pin=tamper_pin, qualified=qualified)
    w.add("dA", "train_fit", subset=True, uncached=(uncached_tag == "dA"))
    w.add("dB", "dev", cohort="incoming", fold=1, subset=True)
    w.add("dC", "train_fit", membership_disagrees=s2_extra, **({"failed": ("foundation",)} if terminal else {"missing": ("foundation",)}),
          expect={"foundation": "residual:no_complete_output" if terminal else "pending:queue_nonterminal"})
    w.add("dS", "sealed", expect={k: "excluded_by_role:sealed" for k in COMPS})
    return w.finalize(subset_order=["dB", "dA"], unmapped=unmapped)


# =========================================================================================== adopt (T21-T28)
class AdoptClassification(P6Case):
    def test_t21_every_status_and_reason_production(self):
        w = build_main_world(self.root)
        self.use_bucket(w.bucket)
        out = self.root / "adopt-P"
        code, s = self.cli(*w.args(out, dry_run=False))
        self.assertEqual(code, 0, s)
        doc = jload(out / "adoption.json")
        self.assertEqual(doc["status"], "PASS")
        self.assertEqual((doc["mode"], doc["gate"]), ("production", "sample"))
        self.assertEqual(doc["inputs"]["contexts"], {"kind": "all", "n": 32})          # production covers every context
        expected = w.expected(production=True)
        self.assertEqual(doc["counts"]["final"], counts_of(expected))
        self.assertEqual(doc["counts"]["tier_I"], counts_of({k: tier_i_of(v) for k, v in expected.items()}))
        rows = {}
        for rel in ("rows/train_fit.jsonl", "rows/dev.jsonl", "rows/diag.jsonl", "sealed/rows-heldout.jsonl", "sealed/rows-excluded.jsonl"):
            for r in rows_of(out / rel):
                self.assertNotIn((r["context_sha256"], r["component"]), rows)
                rows[(r["context_sha256"], r["component"])] = r
        self.assertEqual({k: r["status"] for k, r in rows.items()}, expected)
        for (cid, comp), r in rows.items():
            role = w.pmap[cid]["split_role"]
            self.assertEqual(r["arms_eligible"], [a for a in E.ARMS_ELIGIBLE[comp] if not (role == "heldout" and a in E.DEV_ONLY_ARMS)])
            if comp == "target":
                self.assertEqual(r["arms_eligible"], [] if role == "heldout" else ["LB-replica"])
            self.assertEqual(r["encoder_weight_exposure"], "Unknown")
            self.assertEqual(r["family_provenance"]["split_role"], role)
            self.assertEqual(r["family_provenance"]["partition_map"], {"sha256": sha(w.map_path.read_bytes()), "kind": "fake",
                                                                       "binding_sha256": None})
            self.assertEqual((r["encoder_license"], r["research_only"]),
                             (doc["component_pins"][comp]["encoder_license"], doc["component_pins"][comp]["research_only"]))
        held = w.cid["held"]
        self.assertEqual((rows[(held, "target")]["status"], rows[(held, "target")]["arms_eligible"]), (DEV_ONLY, []))
        self.assertEqual(rows[(held, "hyperion")]["arms_eligible"], ["A1", "A2", "A3"])
        ledger = jload(s["ledger"]["path"])
        held_target = next(x for x in w.index["outputs"] if x.get("identity") and x["identity"][2] == h("sid:held")
                           and x["component"] == "target")
        self.assertNotIn(held_target["outcome_ref"]["key"], [e["key"] for e in ledger["entries"]])   # never fetched
        ok1 = w.cid["ok1"]
        hy = rows[(ok1, "hyperion")]
        self.assertEqual(hy["binding_evidence"]["job_id"], "recomputed_equal")
        self.assertTrue(hy["binding_evidence"]["content"]["kept_features"])
        self.assertTrue(hy["binding_evidence"]["array"]["checks"]["feature_quantiles"])
        self.assertTrue(hy["binding_evidence"]["array"]["checks"]["pooled_vector"])
        self.assertEqual(hy["binding_evidence"]["jtt1_rows"]["n_train"], 40)
        self.assertIn("prepared_ref", hy["binding_evidence"]["manifest"])
        self.assertEqual(hy["jtt1_evidence"], ["shape", "content", "array_content"])
        self.assertEqual(rows[(ok1, "target")]["jtt1_evidence"], ["shape", "content"])
        self.assertEqual(rows[(ok1, "foundation")]["binding_evidence"]["content"]["context_y_moments"], True)
        self.assertEqual(rows[(ok1, "target")]["binding_evidence"]["content"]["target_min_span"], True)
        self.assertEqual(rows[(ok1, "hyperion")]["rights"], {"rights_status": "unknown", "source": "codex_packet"})
        self.assertEqual({k: (doc["component_pins"][k]["encoder_license"]["class"], doc["component_pins"][k]["research_only"])
                          for k in COMPS},
                         {"hyperion": ("eligible_non_open", False), "foundation": ("restricted", True),
                          "target": ("eligible_non_open", True), "tabpfn35": ("unresolved", True), "context2048": ("unrecorded", True)})
        self.assertEqual((doc["completion"]["unfinished_bound_rows"], doc["completion"]["adopted_without_jtt1_evidence"]), (0, 0))
        self.assertEqual((doc["slots"]["unaccounted_slots"], doc["slots"]["retired_component_slots"], doc["slots"]["reconciled_with_release"]),
                         (0, 0, True))
        self.assertTrue(doc["inputs"]["release"]["terminal_release"])
        self.assertTrue(doc["inputs"]["index"]["fetch_receipt"]["index_ref_equal"])
        self.assertEqual(doc["component_pins"]["tabpfn35"]["status"], "unpinned")
        self.assertEqual({k: doc["component_pins"][k]["status"] for k in ("hyperion", "foundation", "target", "context2048")},
                         dict.fromkeys(("hyperion", "foundation", "target", "context2048"), "pinned"))
        self.assertEqual(doc["quarantine"]["S1"], 1)
        self.assertTrue(doc["quarantine"]["equal"])
        # 32 contexts x 5 outputs, minus nofound's foundation, plus dup's second hyperion row; nopacket has no entry (5);
        # jobid's hyperion job_id is wrong (1); pfold's packet fold differs (5); idx's train indices fail the recompute (5)
        self.assertEqual(doc["counts"]["index_level"], {"complete_outputs": 160, "packet_entry_found": 155, "job_id_equal": 154,
                                                        "fold_equals_packet_fold": 150, "train_index_hash_ok": 150})

    def test_t21_component_pin_mismatch(self):
        w = build_small_world(self.root, name="M", terminal=True, tamper_pin="hyperion_runtime_manifest")
        w.expect["dA"]["hyperion"] = w.expect["dB"]["hyperion"] = w.expect["dC"]["hyperion"] = "rejected:component_pin_mismatch"
        self.use_bucket(w.bucket)
        out = self.root / "adopt-M"
        code, s = self.cli(*w.args(out, dry_run=False))
        self.assertEqual(code, 0, s)
        doc = jload(out / "adoption.json")
        self.assertEqual(doc["component_pins"]["hyperion"]["status"], "mismatch")
        self.assertEqual(doc["counts"]["final"], counts_of(w.expected(production=True)))

    def test_t21_pending_in_a_nonterminal_dry_run(self):
        w = build_small_world(self.root, name="D")
        self.use_bucket(w.bucket)
        out = self.root / "adopt-D"
        code, s = self.cli(*w.args(out, dry_run=True))
        self.assertEqual(code, 0, s)
        doc = jload(out / "adoption.json")
        self.assertEqual((doc["status"], doc["mode"]), ("DRY_RUN", "dry_run"))
        expected = w.expected(production=False)
        self.assertEqual(doc["counts"]["final"], counts_of(expected))
        self.assertEqual(expected[(w.cid["dC"], "foundation")], "pending:queue_nonterminal")
        self.assertEqual(doc["counts"]["tier_A"]["hyperion"], {"adopted": 2})
        self.assertEqual(doc["counts"]["tier_O"]["foundation"], {"adopted": 2})
        self.assertEqual(doc["counts"]["tier_O"]["target"], {})


class AdoptGates(P6Case):
    def test_t22_terminal_gate(self):
        w = build_small_world(self.root, name="T")
        self.use_bucket(w.bucket)
        out = self.root / "adopt-prod"
        code, s = self.cli(*w.args(out, dry_run=False))
        self.assertEqual(code, 3, s)
        self.assertEqual((s["status"], s["failure"]["reason"]), ("BLOCKED", "queue_not_terminal"))
        self.assertFalse((out / "rows").exists())
        self.assertFalse((out / "adoption.json").exists())
        self.assertFalse((out / "sealed").exists())
        code, s = self.cli(*w.args(self.root / "adopt-dry", dry_run=True))
        self.assertEqual((code, s["status"]), (0, "DRY_RUN"))
        self.assertFalse(jload(self.root / "adopt-dry" / "adoption.json")["inputs"]["index"]["all_shards_terminal"])

    def _refused(self, w, out, expect_code, reason, **kw):
        code, s = self.cli(*w.args(out, dry_run=True, **kw))
        self.assertEqual(code, expect_code, s)
        self.assertEqual(s["failure"]["reason"], reason, s)
        self.assertFalse((out / "rows").exists())
        self.assertFalse((out / "adoption.json").exists())

    def test_t23_binding_refusals(self):
        w = build_small_world(self.root, name="R")
        self.use_bucket(w.bucket)
        bad_release = self.root / "bad-release-ref.json"
        bad_release.write_text(json.dumps(dict(w.release_ref, sha256=h("other-release"))))
        self._refused(w, self.root / "o1", 2, "release_sha_mismatch", release_ref_path=bad_release)
        bad_ref = self.root / "bad-index-ref.json"
        bad_ref.write_text(json.dumps(dict(w.index_ref, sha256=h("other-index"))))
        self._refused(w, self.root / "o2", 2, "index_sha_mismatch", index_ref_path=bad_ref)
        renamed = self.root / "renamed-index.json"
        renamed.write_bytes(w.index_path.read_bytes())
        self._refused(w, self.root / "o3", 2, "index_sha_mismatch", index_path=renamed)
        other_packet = self.root / "packet-other.json"
        other_packet.write_bytes(w.packet_path.read_bytes() + b" ")
        self._refused(w, self.root / "o4", 2, "packet_sha_mismatch", packet_path=other_packet)
        self._refused(w, self.root / "o5", 2, "inventory_sha_mismatch", inventory_sha=h("other-inventory"))
        # release.packet_ref != index.packet_ref is a STOP
        rel = dict(w.release, packet_ref=dict(w.packet_ref, version_id="another"))
        rb = codex_canonical(rel)
        key = CODEX_PREFIX + f"releases/{sha(rb)}.json"
        ref = {"bucket": CODEX_BUCKET, "key": key, "version_id": w.bucket.preload(key, rb), "sha256": sha(rb), "bytes": len(rb)}
        stop_ref = self.root / "stop-release-ref.json"
        stop_ref.write_text(json.dumps(ref))
        self._refused(w, self.root / "o6", 4, "packet_ref_mismatch", release_ref_path=stop_ref)

    def test_release_index_ref_is_recorded_not_gated(self):
        """The real release a8382194 names an earlier index (40f217ee...), not the adopted one; adopt records whether the
        release's index_ref equals the adopted index (null -> None) and keeps packet_ref equality as the binding."""
        w = build_small_world(self.root, name="RI")
        self.use_bucket(w.bucket)
        seen = {}
        for tag, index_ref in (("null", None), ("other", dict(w.index_ref, sha256=h("earlier-index"), bytes=7)), ("same", dict(w.index_ref))):
            rel = dict(w.release, index_ref=index_ref)
            rb = codex_canonical(rel)
            key = CODEX_PREFIX + f"releases/{sha(rb)}.json"
            ref = {"bucket": CODEX_BUCKET, "key": key, "version_id": w.bucket.preload(key, rb), "sha256": sha(rb), "bytes": len(rb)}
            ref_path = self.root / f"release-ref-{tag}.json"
            ref_path.write_text(json.dumps(ref))
            code, s = self.cli(*w.args(self.root / f"adopt-RI-{tag}", dry_run=True, release_ref_path=ref_path))
            self.assertEqual((code, s["status"]), (0, "DRY_RUN"), s)
            rec = jload(s["receipt"]["path"])["inputs"]["release"]
            seen[tag] = (rec["index_ref_is_null"], rec["index_ref_equals_adopted_index"])
        self.assertEqual(seen, {"null": (True, None), "other": (False, False), "same": (False, True)})

    def test_t24_quarantine(self):
        w = build_main_world(self.root, name="Q")
        self.use_bucket(w.bucket)
        out = self.root / "adopt-Q"
        code, _ = self.cli(*w.args(out, dry_run=False))
        self.assertEqual(code, 0)
        quar = w.cid["quar"]
        self.assertIn(quar, w.subset)
        rows =[r for r in rows_of(out / "rows/train_fit.jsonl") if r["context_sha256"] == quar]
        self.assertEqual({r["status"] for r in rows}, {"residual:feature_sanitized_view"})
        joinable = [x for x in w.index["outputs"] if x.get("identity") and x["identity"][2] == h("sid:quar")]
        self.assertEqual(len(joinable), 5)          # parent rows match by dataset_hash, sample identity and fold
        self.assertFalse(any(r["binding_evidence"]["outcome"] for r in rows))
        # a per-context residual reason (the view kind), and residual totals by reason (AB-5)
        self.assertEqual({r["residual_detail"]["view_kind"] for r in rows}, {"clean_target_leak_v1"})
        self.assertEqual(E.view_kind("07b95b5a26a4df4e__r0__f3__test_adaptive_0f1e2d"), "test_adaptive")
        self.assertEqual(E.view_kind("07b95b5a26a4df4e__r0__f0"), "unsuffixed")
        totals = jload(out / "adoption.json")["counts"]["residual_by_reason"]
        self.assertEqual(totals["total"], {"residual:feature_sanitized_view": 5, "residual:no_complete_output": 1})
        self.assertEqual(totals["by_component"]["foundation"], {"residual:feature_sanitized_view": 1, "residual:no_complete_output": 1})
        s2 = build_small_world(self.root, name="S", s2_extra=True)
        self.use_bucket(s2.bucket)
        code, s = self.cli(*s2.args(self.root / "adopt-S", dry_run=True))
        self.assertEqual((code, s["failure"]["reason"]), (4, "quarantine_sets_differ"))
        self.assertFalse((self.root / "adopt-S" / "rows").exists())

    def test_t25_labels(self):
        w = build_main_world(self.root, name="L")
        self.use_bucket(w.bucket)
        out = self.root / "adopt-L"
        self.assertEqual(self.cli(*w.args(out, dry_run=False))[0], 0)
        # no row and no adoption label claims VERIFIED (inputs may quote Codex's own release state string)
        texts = [p.read_bytes() for p in sorted(out.rglob("*.jsonl"))]
        texts.append(json.dumps(jload(out / "adoption.json")["labels"]).encode())
        self.assertTrue(texts)
        for t in texts:
            self.assertNotIn(b"VERIFIED", t)
        rows = [r for p in sorted(out.rglob("*.jsonl")) for r in rows_of(p)]
        adopted = [r for r in rows if r["status"] == "adopted"]
        self.assertTrue(adopted)
        self.assertEqual({r["train_only"] for r in adopted}, {"CLAIMED"})
        statuses = collections.Counter(r["status"].split(":")[0] for r in rows)
        self.assertTrue(statuses["rejected"] and statuses["not_for_arms"] and statuses["excluded_by_role"] and statuses["residual"])
        for r in rows:
            if r["component"] == "hyperion":
                self.assertIn("gpu-v5-row-sweep-pcg-v1", r["recipe_identity"])
                self.assertIn("NEW identity", r["recipe_identity"])
                self.assertIn("parity not applicable", r["recipe_identity"])
            # TOL-10/AB-7: labels follow the final status, for every status
            bound = r["status"] in ("bound_index_only", "adopted")
            self.assertEqual(r["train_only"], "CLAIMED" if bound else None, r["status"])
            self.assertEqual(r["raw_foundation_2048_retained"], True if r["status"] == "adopted" and r["component"] == "foundation" else None)
            self.assertEqual(r["pre_normalization"], True if r["status"] == "adopted" else None)
            self.assertEqual(r["legacy_projection"], r["component"] == "context2048")
        labels = jload(out / "adoption.json")["labels"]
        self.assertIn("statistical, not a hash equality", labels["jtt1_binding"])
        self.assertIn("null otherwise", labels["train_only"])

    def test_t26_reference_only(self):
        w = build_small_world(self.root, name="F", terminal=True)
        made = []

        def spy(*a, **kw):
            made.append(kw)
            client = fx.FakeS3Client(w.bucket, writes=kw.get("writes", False))
            clients.append(client)
            return client
        clients = []
        for target, attr, value in ((s3io, "make_client", spy), (E, "_read_client", ORIGINAL_READ_CLIENT)):
            p = mock.patch.object(target, attr, value)
            p.start()
            self.addCleanup(p.stop)
        out = self.root / "adopt-F"
        code, s = self.cli(*w.args(out, dry_run=False))
        self.assertEqual(code, 0, s)
        self.assertEqual(len(made), 1)                       # one lazily created client, through the real _read_client
        self.assertFalse(made[0].get("writes", False))
        self.assertFalse(clients[0]._dltrain_guard.writes)
        with self.assertRaises(s3io.S3Refused):              # in PREFIX, so only the read-only guard refuses it
            clients[0].put_object(Bucket=CODEX_BUCKET, Key=s3io.PREFIX_KEY + "x.json", Body=b"{}")
        self.assertEqual(w.bucket.puts(), [])
        for p_ in out.rglob("*"):
            if p_.is_file():
                self.assertNotIn(p_.suffix, (".npz", ".f32le", ".npy"))
                self.assertFalse(p_.read_bytes().startswith(b"PK\x03\x04"), p_)
                self.assertNotIn(b"\x93NUMPY", p_.read_bytes())

    def test_t27_get_budgets_and_ledger(self):
        w = build_small_world(self.root, name="G")
        self.use_bucket(w.bucket)
        out = self.root / "adopt-G"
        code, s = self.cli(*w.args(out, dry_run=True))
        self.assertEqual(code, 0, s)
        ledger = jload(s["ledger"]["path"])
        dA, dB = w.cid["dA"], w.cid["dB"]

        def key_of(cid, comp, kind):
            row = next(x for x in w.index["outputs"] if x.get("identity") and x["identity"][2] == h("sid:" + w.by_cid[cid]["tag"])
                       and x["component"] == comp)
            return row["outcome_ref" if kind == "outcome" else "array_ref"]["key"]
        expect = [("release", w.release_ref["key"]), ("outcome", key_of(dB, "hyperion", "outcome")), ("outcome", key_of(dB, "foundation", "outcome")),
                  ("outcome", key_of(dA, "hyperion", "outcome")), ("outcome", key_of(dA, "foundation", "outcome")),
                  ("array", key_of(dB, "hyperion", "array")), ("array", key_of(dA, "hyperion", "array"))]
        self.assertEqual([(e["kind"], e["key"]) for e in ledger["entries"]], expect)
        for e in ledger["entries"]:
            for f in ("kind", "bucket", "key", "version_id", "expected_sha256", "expected_bytes", "got_sha256", "got_bytes", "attempts", "utc_us"):
                self.assertIn(f, e)
            self.assertTrue(e["ok"])
            self.assertEqual((e["got_sha256"], e["got_bytes"]), (e["expected_sha256"], e["expected_bytes"]))
        # rerun: release.json present, so no release GET; rows identical (existed)
        code, s2 = self.cli(*w.args(out, dry_run=True))
        self.assertEqual(code, 0, s2)
        self.assertEqual(jload(s2["ledger"]["path"])["used"]["release"], 0)
        # outcome+array cap: 3 GETs, then BudgetExhausted before the 4th
        out3 = self.root / "adopt-G3"
        code, s3 = self.cli(*w.args(out3, dry_run=True, budget=3))
        self.assertEqual(code, 0, s3)
        l3 = jload(s3["ledger"]["path"])
        self.assertEqual((l3["used"]["outcome"], l3["used"]["array"]), (3, 0))
        doc3 = jload(out3 / "adoption.json")
        self.assertEqual(doc3["budgets"]["budget_exhausted"]["tier_O"]["reason"], "budget_exhausted:outcome")
        r3 = {(r["context_sha256"], r["component"]): r for r in rows_of(out3 / "rows/train_fit.jsonl") + rows_of(out3 / "rows/dev.jsonl")}
        refused_o = r3[(dA, "foundation")]                    # its outcome GET was refused: no tier O label
        self.assertEqual((refused_o["tier"], refused_o["status"], refused_o["binding_evidence"]["outcome"]), ("I", BOUND, None))
        # outcome+array cap 5: the 4 outcomes and dB's array, then the cap refuses dA's array GET
        out5 = self.root / "adopt-G5"
        code, s5a = self.cli(*w.args(out5, dry_run=True, budget=5))
        self.assertEqual(code, 0, s5a)
        self.assertEqual(jload(s5a["ledger"]["path"])["used"], {"array": 1, "jtt1": 0, "outcome": 4, "release": 1})
        r5 = {(r["context_sha256"], r["component"]): r for r in rows_of(out5 / "rows/train_fit.jsonl") + rows_of(out5 / "rows/dev.jsonl")}
        self.assertEqual((r5[(dB, "hyperion")]["tier"], r5[(dB, "hyperion")]["status"]), ("A", "adopted"))
        self.assertIsNotNone(r5[(dB, "hyperion")]["binding_evidence"]["array"])
        refused_a = r5[(dA, "hyperion")]                      # its array GET was refused: it stays a tier O adoption
        self.assertEqual((refused_a["tier"], refused_a["status"], refused_a["binding_evidence"]["array"]), ("O", "adopted", None))
        self.assertEqual(jload(out5 / "adoption.json")["counts"]["tier_A"]["hyperion"], {"adopted": 1})
        # production stops BLOCKED on the same cap
        wt = build_small_world(self.root, name="GT", terminal=True)
        self.use_bucket(wt.bucket)
        code, s4 = self.cli(*wt.args(self.root / "adopt-GT", dry_run=False, budget=3))
        self.assertEqual((code, s4["status"]), (3, "BLOCKED"))
        # release cap 0: refused before any GET
        calls_before = len(w.bucket.calls)
        self.use_bucket(w.bucket)
        code, s5 = self.cli(*w.args(self.root / "adopt-G0", dry_run=True, release_budget=0))
        self.assertEqual((code, s5["failure"]["reason"]), (3, "budget_exhausted:release"))
        self.assertEqual(len(w.bucket.calls), calls_before)
        # jtt1 cap 0 with an uncached subset JTT1: BudgetExhausted before any JTT1 GET
        wu = build_small_world(self.root, name="GU", uncached_tag="dA")
        self.use_bucket(wu.bucket)
        code, s6 = self.cli(*wu.args(self.root / "adopt-GU", dry_run=True))
        self.assertEqual((code, s6["failure"]["reason"]), (3, "budget_exhausted:jtt1"))
        self.assertFalse(any(op == "GetObject" and "objects/inputs/" in (k or "") for op, k in wu.bucket.calls))

    def test_t28_closure_and_sealed_rows(self):
        w = build_small_world(self.root, name="C", unmapped=("dB",))
        self.use_bucket(w.bucket)
        code, s = self.cli(*w.args(self.root / "adopt-C", dry_run=True))
        self.assertEqual((code, s["failure"]["reason"]), (4, "partition_map_closure"))
        self.assertFalse((self.root / "adopt-C" / "rows").exists())
        wm = build_main_world(self.root, name="H")
        self.use_bucket(wm.bucket)
        out = self.root / "adopt-H"
        self.assertEqual(self.cli(*wm.args(out, dry_run=False))[0], 0)
        held = {c for c, v in wm.pmap.items() if v["split_role"] == "heldout"}
        self.assertTrue(held)
        for p in ("train_fit", "dev", "diag"):
            self.assertFalse({r["context_sha256"] for r in rows_of(out / f"rows/{p}.jsonl")} & held)
            self.assertNotIn(next(iter(held)).encode(), (out / f"rows/{p}.jsonl").read_bytes())
        self.assertEqual({r["context_sha256"] for r in rows_of(out / "sealed/rows-heldout.jsonl")}, held)
        self.assertNotIn(next(iter(held)).encode(), (out / "adoption.json").read_bytes())


ORIGINAL_READ_CLIENT = E._read_client


class CodexForms(P6Case):
    def test_t30_codex_canonical_job_id(self):
        packet = {"model_refs": model_refs(), "source_pins": {"gpu_encoders.py": {"bytes": 1, "sha256": h("src")}}}
        d = {"identity": ["0123456789abcdef", h("canon"), h("sid")],
             "manifest_ref": {"bucket": "b", "key": "datasets/café/manifest.json", "version_id": "v", "sha256": h("m"), "bytes": 3}}
        f = {"fold": 2, "train_indices_sha256": h("tri"), "train_row_ids_sha256": h("rid")}
        for comp in COMPS:
            self.assertEqual(E.codex_job_id(d, f, comp, packet), codex_job_id(d, f, comp, packet))
        pre = dict(schema="neural-encoding-job/v1", identity=d["identity"], manifest_ref=d["manifest_ref"], train_indices_sha256=f["train_indices_sha256"],
                   train_row_ids_sha256=f["train_row_ids_sha256"], fold=f["fold"], component="hyperion", model_refs=packet["model_refs"],
                   encoder_source_sha256=h("src"))
        self.assertEqual(E.codex_canonical(pre), codex_canonical(pre))
        self.assertNotEqual(E.codex_canonical(pre), common.canonical(pre))     # ensure_ascii differs on the non-ASCII key
        self.assertNotEqual(E.codex_job_id(d, f, "hyperion", packet), sha(common.canonical(pre)))
        for fn in (E.codex_job_id, E.codex_canonical):          # no reference to common.canonical in the code (docstrings aside)
            tree = ast.parse(inspect.getsource(fn))
            attrs = {(n.value.id, n.attr) for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
            self.assertNotIn(("common", "canonical"), attrs)
            self.assertNotIn("common", {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})

    def test_index_hash_formula(self):
        rng = random.Random(7)
        for _ in range(20):
            idx = sorted(rng.sample(range(5000), rng.randrange(0, 300)))
            self.assertEqual(E.le_u64_sha256(idx), le_u64(idx))
        fold = {"train_indices": [0, 2, 5], "train_indices_sha256": le_u64([0, 2, 5]), "train_rows": 3, "test_rows": 3}
        self.assertTrue(E.train_index_ok(fold))
        self.assertFalse(E.train_index_ok(dict(fold, train_indices=[0, 5, 2])))
        self.assertFalse(E.train_index_ok(dict(fold, train_indices=[0, 2, 5], train_rows=4)))
        self.assertFalse(E.train_index_ok(dict(fold, train_indices=[0, 2, 6], train_indices_sha256=le_u64([0, 2, 6]))))
        self.assertFalse(E.train_index_ok(dict(fold, train_indices=[-1, 2, 5])))


# ================================================================================= map-audit regression (T29)
def _old_load_index(path, sha256):
    raw = Path(path).read_bytes()
    if common.sha256_bytes(raw) != sha256:
        raise RuntimeError("index sha256 mismatch")
    return json.loads(raw)


def _old_fetch_memberships(contexts, cache_path, threads=8):
    """Copied from the pre-refactor encodings.py (sha256 c454e1d2...)."""
    import concurrent.futures as cf
    cache = {}
    cp = Path(cache_path)
    if cp.exists():
        for line in cp.read_text().splitlines():
            rec = json.loads(line)
            cache[rec["context_sha256"]] = rec
    todo = [c for c in contexts if c["context_sha256"] not in cache]
    client = s3io.make_client()

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

    if todo:
        with cf.ThreadPoolExecutor(threads) as ex, open(cp, "a") as f:
            for rec in ex.map(one, todo):
                cache[rec["context_sha256"]] = rec
                f.write(json.dumps(rec, sort_keys=True) + "\n")
    return cache, len(todo)


def _old_map_audit(epoch_path, index_path, index_sha256, cache_path):
    """Copied from the pre-refactor encodings.py (sha256 c454e1d2...)."""
    COMPONENTS = ("hyperion", "foundation", "target", "tabpfn35", "context2048")
    e = json.loads(Path(epoch_path).read_text())
    contexts = [c for c in e["contexts"] if c["status"] == "included"]
    index = _old_load_index(index_path, index_sha256)
    outs = collections.defaultdict(list)
    for x in index["outputs"]:
        if x.get("identity") and x.get("state") == "complete":
            outs[(x["identity"][0], x["identity"][2], x["fold"]["fold"], x["fold"].get("repeat", 0))].append(x)
    members, fetched = _old_fetch_memberships(contexts, cache_path)
    rows, reasons = [], collections.Counter()
    comp_cov = collections.Counter()
    for c in contexts:
        mem = members[c["context_sha256"]]
        key = (c["dataset_hash"], c["sample_identity_sha256"], c["fold"], 0)
        problems = []
        if mem["sample_identity_sha256"] != c["sample_identity_sha256"] or mem["outer_fold"] != c["fold"] or mem["dataset_hash"] != c["dataset_hash"]:
            problems.append("membership_disagrees_with_epoch")
        matched = {}
        for x in outs.get(key, []):
            f = x["fold"]
            ok = (f["train_indices_sha256"] == mem["outer_train_indices_sha256"] and f["test_indices_sha256"] == mem["outer_test_indices_sha256"]
                  and f["train_rows"] == mem["outer_train_rows"] and f["test_rows"] == mem["outer_test_rows"])
            if c.get("canonical_sha256") and x["identity"][1] != c["canonical_sha256"]:
                ok = False
                problems.append(f"canonical_mismatch:{x['component']}")
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


class MapAuditRegression(P6Case):
    def test_t29_same_summary_and_rows(self):
        w = AdoptWorld(self.root, name="MA", terminal=False)
        w.add("full", "train_fit")
        w.add("partial", "dev", missing=("tabpfn35", "context2048"))
        w.add("none", "train_fit", missing=COMPS)
        w.add("dup", "train_fit", hyperion={"duplicate": True})
        w.add("rows", "train_fit", membership_test_sha_wrong=True)
        w.add("disagree", "train_fit", membership_disagrees=True)
        w.add("canon", "train_fit")
        w.finalize()
        epoch = jload(w.epoch_path)
        self.assertEqual(epoch["contexts"][6]["context_sha256"], w.cid["canon"])
        epoch["contexts"][6]["canonical_sha256"] = h("other-canonical")      # canonical mismatch on one context
        w.epoch_path.write_bytes(common.canonical(epoch))
        idx_sha = sha(w.index_path.read_bytes())
        with mock.patch.object(s3io, "make_client", lambda *a, **k: fx.FakeS3Client(fx.FakeBucket())):
            old = _old_map_audit(w.epoch_path, w.index_path, idx_sha, w.mem_path)
            new = E.map_audit(w.epoch_path, w.index_path, idx_sha, w.mem_path)
            self.assertEqual(new, old)
            self.assertTrue(old[0]["status_counts"].get("complete") and old[0]["status_counts"].get("partial") and old[0]["status_counts"].get("none"))
            for reason in ("membership_disagrees_with_epoch", "canonical_mismatch", "train_or_test_rows_mismatch", "duplicate_component"):
                self.assertIn(reason, old[0]["status_counts"])
            out = self.root / "map-audit.json"
            code, s = self.cli("map-audit", "--epoch", w.epoch_path, "--index", w.index_path, "--index-sha256", idx_sha, "--cache", w.mem_path, "--out", out)
        self.assertEqual(code, 0)
        doc = jload(out)
        self.assertEqual(doc["schema"], "jope.dl-encoding-map-audit/v1")
        self.assertEqual([doc["summary"], doc["rows"]], json.loads(json.dumps(list(old))))
        self.assertEqual(sorted(s), ["receipt", "summary"])

    def test_complete_membership_cache_needs_no_s3_client(self):
        """With every included context already in the memberships cache, map-audit builds no S3 client at all (it runs
        under the no-AWS prefix) and appends nothing to the cache."""
        w = AdoptWorld(self.root, name="MB", terminal=False)
        w.add("full", "train_fit")
        w.add("none", "dev", missing=COMPS)
        w.finalize()
        before = w.mem_path.read_bytes()
        idx_sha = sha(w.index_path.read_bytes())

        def refuse(*_a, **_k):
            raise AssertionError("map-audit built an S3 client although the membership cache is complete")

        with mock.patch.object(s3io, "make_client", refuse):
            summary, rows = E.map_audit(w.epoch_path, w.index_path, idx_sha, w.mem_path)
        self.assertEqual(summary["memberships_fetched_now"], 0)
        self.assertEqual(len(rows), 2)
        self.assertEqual(w.mem_path.read_bytes(), before)


# ============================================================================================ run (T31-T38)
class RunA0(P6Case):
    def test_t31_end_to_end(self):
        w = RunWorld(self.root, cache=False)
        self.use_bucket(w.bucket)
        out = self.root / "a0"
        code, s = self.cli(*w.run_args(out, budget=64))
        self.assertEqual(code, 0, s)
        self.assertEqual(s["status"], "PASS")
        self.assertEqual(s["ledger"]["gets"], 10 - 2)          # sealed and prohibited are never loaded
        for p in ("manifests/train_fit.json", "manifests/dev.json", "manifests/diag.json", "sealed/heldout.json", "arm.json", "sealed/closure.json"):
            self.assertTrue((out / p).is_file(), p)
            self.assertEqual(os.stat(out / p).st_mode & 0o777, 0o444)
        self.assertFalse((out / "manifests/heldout.json").exists())
        arrays = sorted((out / "arrays").glob("*.f32le"))
        sealed_arrays = sorted((out / "sealed/arrays").glob("*.f32le"))
        self.assertEqual((len(arrays), len(sealed_arrays)), (6, 2))
        for a in arrays + sealed_arrays:
            self.assertEqual(os.stat(a).st_mode & 0o777, 0o444)
            self.assertEqual(a.stat().st_size, 352)
            self.assertEqual(a.name, sha(a.read_bytes()) + ".f32le")
        norm_files = sorted((out / "normalizers").iterdir())
        self.assertEqual([p.suffix for p in norm_files].count(".json"), 1)
        self.assertEqual([p.suffix for p in norm_files].count(".f32le"), 1)
        from jsonschema import Draft202012Validator
        schema = E.SCHEMAS["jope.dl-context-encoding.schema.json"]
        file_schema = SRC / "schemas" / "jope.dl-context-encoding.schema.json"
        schemas = [schema] + ([jload(file_schema)] if file_schema.is_file() else [])
        if file_schema.is_file():
            self.assertEqual(jload(file_schema), json.loads(json.dumps(schema)))
        for p in ("manifests/train_fit.json", "manifests/dev.json", "manifests/diag.json", "sealed/heldout.json"):
            for sc in schemas:
                Draft202012Validator(sc).validate(jload(out / p))
        Draft202012Validator(E.SCHEMAS["jope.dl-encoding-arm.schema.json"]).validate(jload(out / "arm.json"))
        receipt = jload(s["receipt"]["path"])
        self.assertEqual(receipt["packet"], "P6-A0-sample")
        self.assertEqual(receipt["counts"]["excluded_by_role"], {"sealed": 1, "prohibited": 1})
        self.assertEqual({p: receipt["counts"][p]["present"] for p in E.PARTITIONS}, {"train_fit": 3, "dev": 2, "diag": 1, "heldout": 2})
        listed = {e["context_sha256"] for p in ("manifests/train_fit.json", "manifests/dev.json", "manifests/diag.json", "sealed/heldout.json")
                  for e in jload(out / p)["contexts"]}
        self.assertEqual(listed, set(w.admitted()))
        for tag in ("sl1", "pr1"):
            for p in out.rglob("*.json"):
                self.assertNotIn(w.tag[tag].encode(), p.read_bytes(), p)
        held = {w.tag["hd1"], w.tag["hd2"]}
        for p in ("manifests/train_fit.json", "manifests/dev.json", "manifests/diag.json", "arm.json"):
            for cid in held:
                self.assertNotIn(cid.encode(), (out / p).read_bytes())
        self.assertNotIn("sealed/heldout.json", (out / "arm.json").read_text())

    def test_t32_manifest_fields(self):
        w = RunWorld(self.root)
        out = self.root / "a0"
        code, s = self.cli(*w.run_args(out))
        self.assertEqual(code, 0, s)
        mans = {p: jload(out / E._manifest_rel(p)) for p in E.PARTITIONS}
        tf = mans["train_fit"]
        rec = tf["normalizers"][0]
        n = N.load(out / "normalizers", rec["normalizer_sha256"])
        self.assertEqual(rec, n.manifest_entry())
        present_tf = sorted(e["context_sha256"] for e in tf["contexts"] if e["status"] == "present")
        self.assertEqual(rec["fit_context_set_sha256"],
                         sha(common.canonical({"schema": "jope.dl-fit-context-set/v1", "contexts": present_tf})))
        fams = sorted({w.pmap[c]["family_id"] for c in present_tf})
        self.assertEqual(rec["fit_family_set_sha256"], sha(common.canonical({"schema": "jope.dl-fit-family-set/v1", "families": fams})))
        mems = {json.loads(line)["context_sha256"]: json.loads(line) for line in w.mem_path.read_text().splitlines()}
        for p, m in mans.items():
            self.assertEqual((m["schema"], m["arm"], m["partition"], m["experiment_id"]), ("jope.dl-context-encoding/v1", "A0", p, E.EXPERIMENT_ID))
            self.assertEqual(m["abi_id"], N.abi_id("A0", [n], {"m": n.width_full}))
            self.assertEqual(m["width_full"], n.width_full)
            prod = m["blocks"][0]["producer"]
            self.assertEqual((prod["module"], prod["version"], prod["feature_list_sha256"], prod["source_tree_sha256"]),
                             ("dltrain.metafeatures", A0.VERSION, A0.FEATURE_LIST_SHA256, A0.source_tree_sha256()))
            self.assertEqual(m["blocks"][0]["components"], [])
            self.assertEqual(m["normalizers"], [rec])
            self.assertEqual(m["inputs"]["partition_map"], {"path": str(w.map_path), "sha256": sha(w.map_path.read_bytes()), "kind": "fake",
                                                            "binding": None})
            self.assertEqual(m["train_only_rule"], E.TRAIN_ONLY_RULE)
            self.assertEqual(m.get("partition_note"), E.DIAG_NOTE if p == "diag" else None)
            self.assertEqual([e["context_sha256"] for e in m["contexts"]], sorted(e["context_sha256"] for e in m["contexts"]))
            for e in m["contexts"]:
                c, raw = w.by_id[e["context_sha256"]], w.raws[e["context_sha256"]]
                self.assertEqual(e["train_row_sha256"], jtt1.train_rows_sha256(raw))
                self.assertEqual(e["input"], {k: c["input_ref"][k] for k in ("bucket", "key", "version_id", "sha256", "bytes", "raw_sha256", "raw_bytes")})
                self.assertEqual(e["membership_sha256"], c["membership_ref"]["sha256"])
                self.assertEqual(e["outer_train_indices_sha256"], mems[c["context_sha256"]]["outer_train_indices_sha256"])
                self.assertEqual((e["family_id"], e["split_role"]), (w.pmap[c["context_sha256"]]["family_id"], p))
                # TOL-03: a structural label scoped to JTT1 bytes; VERIFIED exists only in a PASS verify receipt
                self.assertEqual((e["rights_status"], e["rights_source"], e["status"], e["train_only"], e["train_only_scope"]),
                                 ("unknown", "pending_P2", "present", "STRUCTURAL_JTT1_TRAIN_BYTES", "jtt1_bytes"))
                _h, _g, Xc, yc = jtt1.train_view(raw)
                self.assertEqual(e["upstream_exposure"], json.loads(json.dumps(E.upstream_exposure(Xc, yc))))
                self.assertFalse(e["upstream_exposure"]["scale"]["exposed"] or e["upstream_exposure"]["column_set"]["exposed"])
                self.assertEqual((e["train_rows"], e["features"]), (40, 4))
                arr = e["arrays"][0]
                data = (out / arr["path"]).read_bytes()
                vec, _ = A0.from_raw(raw)
                self.assertEqual(data, vec)
                self.assertEqual((arr["sha256"], arr["bytes"], arr["dtype"], arr["shape"], arr["block_id"]), (sha(data), 352, "<f4", [88], "m"))
                self.assertEqual(arr["path"].startswith("sealed/"), p == "heldout")
            self.assertEqual(m["coverage"], {"admitted": len(m["contexts"]), "present": len(m["contexts"]), "missing": {}})
        self.assertEqual(mans["heldout"]["encoder_weight_exposure"], {"fam-F": {}})
        for p in E.OPEN_PARTITIONS:
            self.assertNotIn("encoder_weight_exposure", mans[p])

    def test_t36_run_refusals(self):
        w = RunWorld(self.root)
        p5 = self.root / "p5.json"
        p5.write_text(json.dumps({"packet": "P5", "status": "PASS", "epoch": {"sha256": h("another-epoch")}}))
        hb = self.root / "holdout-binding.json"
        hb.write_text(json.dumps({"schema": "jope.dl-binding/v1", "ref": {"bucket": "b", "key": "k", "version_id": "v", "sha256": h("hb"), "bytes": 1}}))
        real = E.RUN_ENCODINGS / "A0"
        before = real.exists()
        for extra in ((), ("--p5-receipt", p5), ("--p5-receipt", p5, "--holdout-binding", hb)):
            code, s = self.cli(*w.run_args(real, *extra))
            self.assertEqual((code, s["status"], s["reason"]), (3, "BLOCKED", "output_gate"), s)
        ok_p5 = self.root / "p5-ok.json"
        ok_p5.write_text(json.dumps({"packet": "P5", "status": "PASS", "epoch": {"sha256": sha(w.epoch_path.read_bytes())}}))
        code, s = self.cli(*w.run_args(real, "--p5-receipt", ok_p5, "--holdout-binding", hb))      # kind fake: still refused
        self.assertEqual((code, s["reason"]), (3, "output_gate"))
        self.assertIn("partition_map_kind", s["detail"])
        self.assertEqual(real.exists(), before)
        code, s = self.cli(*w.run_args(self.root.parent / "not-a-sample-dir" / "a0"))
        self.assertEqual((code, s["reason"]), (3, "output_gate"))
        # the slice check fails: exit before any file read
        self.slice.side_effect = RuntimeError("slice aggregate differs")
        _OPENS["events"] = []
        _OPENS["armed"] = True
        try:
            code, s = self.cli(*w.run_args(self.root / "a0-slice"))
        finally:
            _OPENS["armed"] = False
        self.assertEqual((code, s["reason"]), (3, "slice_limits"))
        self.assertEqual([e for e in _OPENS["events"] if e[1].startswith(str(self.root))], [])
        self.slice.side_effect = None
        # uncached JTT1 with --jtt1-get-budget 0: BudgetExhausted before any GET
        wu = RunWorld(self.root, cache=False, name="uncached")
        self.use_bucket(wu.bucket)
        code, s = self.cli(*wu.run_args(self.root / "a0-budget", budget=0))
        self.assertEqual((code, s["status"], s["failure"]["reason"]), (3, "BLOCKED", "budget_exhausted:jtt1"))
        self.assertEqual([op for op, _ in wu.bucket.calls], [])
        self.assertFalse((self.root / "a0-budget" / "manifests").exists())

    def test_blocked_arms(self):
        for arm in ("A1", "A2", "A3", "LB-replica"):
            code, s = self.cli("run", "--arm", arm)
            self.assertEqual((code, s["status"]), (3, "BLOCKED"))
            text = " ".join(s["reasons"])
            self.assertIn("terminal_index_and_production_adoption", text)
            self.assertIn("npz_read_line", text)
            self.assertIn("P5", text)
            self.assertIn("adopted_train_only_counterfactual", text)          # TOL-08
            self.assertIn("test-row dependence -> STOP", text)
            self.assertEqual("DL-D13" in text, arm in ("A1", "A2", "A3"))
            self.assertEqual("tabpfn35_pin" in text, arm == "A3")

    def test_t37_guard_and_slice_order(self):
        w = RunWorld(self.root)
        order = []
        self.slice.side_effect = lambda *a, **k: order.append(("slice",)) or {}
        _OPENS["events"] = order
        _OPENS["armed"] = True
        try:
            code, s = self.cli(*w.run_args(self.root / "a0"))
        finally:
            _OPENS["armed"] = False
        self.assertEqual(code, 0, s)
        self.assertEqual(order[0], ("slice",))
        g = StubGuard.created[-1]
        self.assertEqual(g.events[0], "wait")
        self.assertGreaterEqual(g.events.count("tick"), len(w.admitted()))
        # adopt
        wa = build_small_world(self.root, name="guard-adopt")
        self.use_bucket(wa.bucket)
        order2 = []
        self.slice.side_effect = lambda *a, **k: order2.append(("slice",)) or {}
        _OPENS["events"] = order2
        _OPENS["armed"] = True
        try:
            code, s = self.cli(*wa.args(self.root / "adopt", dry_run=True))
        finally:
            _OPENS["armed"] = False
        self.assertEqual(code, 0, s)
        self.assertEqual(order2[0], ("slice",))
        g = StubGuard.created[-1]
        self.assertEqual(g.events[0], "wait")
        self.assertGreaterEqual(g.events.count("tick"), len([c for c in wa.epoch["contexts"] if c["status"] == "included"]))
        # verify
        StubGuard.created = []
        code, s = self.cli(*w.verify_args(self.root / "a0", "--coverage", "--counterfactual"))
        self.assertEqual(code, 0, s)
        g = StubGuard.created[-1]
        self.assertEqual(g.events[0], "wait")
        self.assertGreaterEqual(g.events.count("tick"), len(w.admitted()))

    def test_t38_idempotent_rerun_and_conflict(self):
        w = RunWorld(self.root)
        out = self.root / "a0"
        code, s1 = self.cli(*w.run_args(out))
        self.assertEqual(code, 0, s1)
        before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file() and not p.name.startswith(("receipt-", "ledger-", "guard-"))}
        code, s2 = self.cli(*w.run_args(out))
        self.assertEqual(code, 0, s2)
        self.assertTrue(s2["writes"])
        self.assertTrue(all(x["existed"] for x in s2["writes"]), s2["writes"])
        self.assertTrue(not any(x["existed"] for x in s1["writes"]))
        after = {p: p.read_bytes() for p in before}
        self.assertEqual(before, after)
        pm = dict(w.pmap)
        pm[w.tag["dv1"]] = {"family_id": "fam-D", "split_role": "train_fit"}
        code, s3 = self.cli(*w.run_args(out, map_path=w.write_map(pm, "map2.json")))
        self.assertEqual((code, s3["status"], s3["failure"]["reason"]), (4, "STOP", "immutable_conflict"))
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        # adopt: identical rerun reports existed for every create-only output
        wa = build_small_world(self.root, name="idem")
        self.use_bucket(wa.bucket)
        aout = self.root / "adopt"
        code, a1 = self.cli(*wa.args(aout, dry_run=True))
        self.assertEqual(code, 0, a1)
        code, a2 = self.cli(*wa.args(aout, dry_run=True))
        self.assertEqual(code, 0, a2)
        self.assertTrue(a2["writes"] and all(x["existed"] for x in a2["writes"]), a2["writes"])

    def test_missing_context_is_reported_never_imputed(self):
        w = RunWorld(self.root, nan_tag="dv1")
        out = self.root / "a0"
        code, s = self.cli(*w.run_args(out))
        self.assertEqual((code, s["status"], s["failure"]["reason"]), (2, "FAIL", "coverage_incomplete"))
        dev = jload(out / "manifests/dev.json")
        e = next(x for x in dev["contexts"] if x["context_sha256"] == w.tag["dv1"])
        self.assertEqual((e["status"], e["missing_reason"], e["arrays"], e["train_only"]), ("missing", "nonfinite_target", [], None))
        self.assertEqual(dev["coverage"]["missing"], {"nonfinite_target": 1})


# ========================================================================================== verify (T33-T35)
class VerifyA0(P6Case):
    def setUp(self):
        super().setUp()
        self.w = RunWorld(self.root)
        self.out = self.root / "a0"
        code, s = self.cli(*self.w.run_args(self.out))
        self.assertEqual(code, 0, s)

    def verify(self, *flags, map_path=None):
        code, s = self.cli(*self.w.verify_args(self.out, *flags, map_path=map_path))
        return code, s, jload(s["verify"]["path"])

    def test_t33_coverage(self):
        code, s, doc = self.verify("--coverage")
        self.assertEqual((code, doc["results"]["coverage"]["status"]), (0, "PASS"), doc)
        self.assertEqual(doc["results"]["coverage"]["present"], 8)
        e = next(x for x in jload(self.out / "manifests/dev.json")["contexts"])
        arr = self.out / e["arrays"][0]["path"]
        data = arr.read_bytes()
        arr.unlink()
        code, s, doc = self.verify("--coverage")
        self.assertEqual(code, 2)
        self.assertIn({"context": e["context_sha256"], "problem": "array_absent"}, doc["results"]["coverage"]["problems"])
        arr.write_bytes(data[:-4] + b"\0\0\0\0")
        code, s, doc = self.verify("--coverage")
        self.assertEqual(code, 2)
        self.assertIn({"context": e["context_sha256"], "problem": "array_sha_or_bytes_mismatch"}, doc["results"]["coverage"]["problems"])
        writable(arr).write_bytes(data)
        code, _, _ = self.verify("--coverage")
        self.assertEqual(code, 0)
        dev_path = writable(self.out / "manifests/dev.json")
        dev = jload(dev_path)
        tf_entry = jload(self.out / "manifests/train_fit.json")["contexts"][0]
        dev["contexts"].append(tf_entry)
        dev_path.write_bytes(common.canonical(dev))
        code, s, doc = self.verify("--coverage")
        self.assertEqual(code, 2)
        probs = doc["results"]["coverage"]["problems"]
        self.assertIn({"context": tf_entry["context_sha256"], "problem": "duplicated_across_manifests"}, probs)

    def test_t33_nonfinite_target_makes_coverage_fail(self):
        w = RunWorld(self.root, nan_tag="dv2", name="nan")
        out = self.root / "a0-nan"
        self.assertEqual(self.cli(*w.run_args(out))[0], 2)
        code, s = self.cli(*w.verify_args(out, "--coverage"))
        doc = jload(s["verify"]["path"])
        self.assertEqual(code, 2)
        cov = doc["results"]["coverage"]
        self.assertEqual((cov["status"], cov["missing"], cov["present"], cov["admitted"]), ("FAIL", {"nonfinite_target": 1}, 7, 8))
        self.assertIn({"context": w.tag["dv2"], "problem": "missing:nonfinite_target"}, cov["problems"])

    def test_t34_fit_closure(self):
        code, s, doc = self.verify("--fit-closure")
        fc = doc["results"]["fit_closure"]
        self.assertEqual((code, fc["status"]), (0, "PASS"), fc)
        self.assertTrue(fc["refit_identical_bytes"])
        self.assertTrue(fc["same_runtime_identity"])
        # tampered fit hash, consistently in every manifest
        saved = {p: (self.out / E._manifest_rel(p)).read_bytes() for p in E.PARTITIONS}
        for p in E.PARTITIONS:
            path = writable(self.out / E._manifest_rel(p))
            m = jload(path)
            m["normalizers"][0]["fit_context_set_sha256"] = h("tampered")
            path.write_bytes(common.canonical(m))
        code, s, doc = self.verify("--fit-closure")
        self.assertEqual((code, doc["results"]["fit_closure"]["status"]), (2, "FAIL"))
        self.assertFalse(doc["results"]["fit_closure"]["checks"]["fit_context_set_sha256"])
        for p, b in saved.items():
            (self.out / E._manifest_rel(p)).write_bytes(b)
        # a normalizer whose fit set includes a dev context
        orig = E.fit_a0_normalizer

        def leaky(entries_by_partition, vectors, pmap, **kw):
            ents = [dict(e, split_role="train_fit") for p in ("train_fit", "dev") for e in entries_by_partition.get(p, []) if e["status"] == "present"]
            pm = {c: (dict(v, split_role="train_fit") if v["split_role"] == "dev" else v) for c, v in pmap.items()}
            return orig(ents, vectors, pm, **kw)
        out1 = self.root / "a0-leaky-run"
        with mock.patch.object(E, "fit_from_entries", leaky):
            code, s = self.cli(*self.w.run_args(out1))
        # TOL-02: the run-side closure STOPs before the normalizer or any manifest is written
        self.assertEqual((code, s["status"], s["failure"]["reason"]), (4, "STOP", "fit_closure"), s)
        self.assertFalse((out1 / "normalizers").exists())
        self.assertFalse((out1 / "manifests").exists() or (out1 / "sealed" / "heldout.json").exists() or (out1 / "arm.json").exists())
        # verify stays an independent second check: with the run-side check disabled, verify still fails the output
        out2 = self.root / "a0-leaky"
        with mock.patch.object(E, "fit_from_entries", leaky), mock.patch.object(E, "_assert_run_fit_closure", lambda *a, **k: {}):
            self.assertEqual(self.cli(*self.w.run_args(out2))[0], 0)
        code, s = self.cli(*self.w.verify_args(out2, "--fit-closure"))
        doc = jload(s["verify"]["path"])
        self.assertEqual(code, 2)
        fc = doc["results"]["fit_closure"]
        self.assertEqual(fc["status"], "FAIL")
        self.assertFalse(fc["checks"]["fit_context_set_sha256"])
        self.assertFalse(fc["checks"]["assert_fit_closure"])

    def test_t35_poison(self):
        code, s, doc = self.verify("--poison-test", "--poison-scope", "all")
        pt = doc["results"]["poison_test"]
        self.assertEqual((code, pt["status"]), (0, "PASS"), pt)
        self.assertTrue(all(pt["a"].values()) and all(pt["b"].values()))
        self.assertTrue(pt["c"]["normalizer_differs"] and pt["c"]["vector_differs"])
        self.assertTrue(pt["baseline_reproduces_stored"] and pt["baseline_rows_equal_stored"])
        self.assertEqual((pt["perturbed_non_train_fit"], pt["perturbed_train_fit_test_bytes"]), (5, 3))
        # TOL-06: scope-level variants (drop, add synthetic, relabel non-train_fit contexts) and the diag scope note
        self.assertTrue(pt["d"]["normalizer_identical"] and pt["e"]["normalizer_identical"] and pt["f"]["normalizer_identical"])
        self.assertEqual((pt["d"]["dropped"], pt["e"]["added"], pt["f"]["relabelled"]), (5, sum(E.POISON_SYNTHETIC.values()), 5))
        self.assertEqual((pt["diag_note"], pt["b_fills"]), (E.DIAG_NOTE, ["0x00", "0xff"]))
        # a fit that depends on how many non-train_fit contexts are in scope -> STOP through (d) and (e)
        orig_ffe = E.fit_from_entries

        def counting(entries_by_partition, vectors, pmap, **kw):
            n_other = sum(len(v) for p, v in entries_by_partition.items() if p != "train_fit")
            scaled = {c: (np.frombuffer(v, dtype="<f4") * np.float32(1 + n_other)).astype("<f4").tobytes() for c, v in vectors.items()}
            return orig_ffe(entries_by_partition, scaled, pmap, **kw)
        with mock.patch.object(E, "fit_from_entries", counting):
            code, s, doc = self.verify("--poison-test")
        cpt = doc["results"]["poison_test"]
        self.assertEqual((code, cpt["status"]), (4, "STOP"))
        self.assertEqual((cpt["d"]["normalizer_identical"], cpt["e"]["normalizer_identical"], cpt["f"]["normalizer_identical"]),
                         (False, False, True))
        self.assertTrue(all(cpt["a"].values()))
        code, s, doc = self.verify("--poison-test", "--poison-scope", "sample:1")
        self.assertEqual(code, 0)
        self.assertEqual(doc["results"]["poison_test"]["perturbed_non_train_fit"], 3)
        # A0 that reads one test byte -> STOP
        orig_from_raw = A0.from_raw

        def leaky_from_raw(raw):
            vec, meta = orig_from_raw(raw)
            v = np.frombuffer(vec, dtype="<f4").copy()
            v[0] = np.float32(v[0] + raw[-1])
            return v.tobytes(), meta
        with mock.patch.object(A0, "from_raw", leaky_from_raw):
            code, s, doc = self.verify("--poison-test")
        self.assertEqual((code, doc["results"]["poison_test"]["status"], doc["status"]), (4, "STOP", "STOP"))
        self.assertFalse(doc["results"]["poison_test"]["b"]["train_fit_arrays_identical"])
        # a fit that includes dev rows -> STOP
        orig = E.fit_a0_normalizer

        def leaky_fit(entries_by_partition, vectors, pmap, **kw):
            ents = [dict(e, split_role="train_fit") for p in ("train_fit", "dev") for e in entries_by_partition.get(p, []) if e["status"] == "present"]
            pm = {c: (dict(v, split_role="train_fit") if v["split_role"] == "dev" else v) for c, v in pmap.items()}
            return orig(ents, vectors, pm, **kw)
        with mock.patch.object(E, "fit_from_entries", leaky_fit):
            code, s, doc = self.verify("--poison-test")
        self.assertEqual((code, doc["results"]["poison_test"]["status"]), (4, "STOP"))
        self.assertFalse(doc["results"]["poison_test"]["a"]["normalizer_identical"])
        # a fit blind to the control context -> FAIL
        orig_sel = E.select_fit_entries

        def blind(entries_by_partition):
            return orig_sel(entries_by_partition)[1:]
        with mock.patch.object(E, "select_fit_entries", blind):
            code, s, doc = self.verify("--poison-test")
        self.assertEqual((code, doc["results"]["poison_test"]["status"]), (2, "FAIL"))
        self.assertFalse(doc["results"]["poison_test"]["c"]["normalizer_differs"])

    def test_counterfactual_and_affine(self):
        code, s, doc = self.verify("--counterfactual", "--affine-invariance")
        cf, af = doc["results"]["counterfactual"], doc["results"]["affine_invariance"]
        self.assertEqual((code, cf["status"], af["status"]), (0, "PASS", "PASS"), doc)
        self.assertEqual((cf["runs"], cf["identical"], cf["controls"], cf["controls_differ"]), (40, 40, 8, 8))
        self.assertEqual(cf["variants"], list(E.COUNTERFACTUAL_VARIANTS))
        self.assertNotIn("as_drawn_unfaithful", af)
        self.assertEqual(len(af["per_feature_max_abs_delta"]), 88)
        self.assertLessEqual(af["max_abs_delta"], 1e-6)
        self.assertEqual(doc["gets"], 0)
        orig_from_raw = A0.from_raw

        def leaky_from_raw(raw):
            vec, meta = orig_from_raw(raw)
            v = np.frombuffer(vec, dtype="<f4").copy()
            v[3] = np.float32(raw[-3])
            return v.tobytes(), meta
        with mock.patch.object(A0, "from_raw", leaky_from_raw):
            code, s, doc = self.verify("--counterfactual")
        self.assertEqual((code, doc["results"]["counterfactual"]["status"]), (4, "STOP"))

    def test_affine_failure_and_a0_error_are_reported(self):
        orig_features = A0.features

        def located(X, y, g):                       # depends on absolute location: not affine invariant
            v = np.array(orig_features(X, y, g), dtype=np.float64)
            v[0] += float(np.mean(X))
            return v
        with mock.patch.object(A0, "features", located):
            code, s, doc = self.verify("--affine-invariance")
        af = doc["results"]["affine_invariance"]
        self.assertEqual((code, af["status"]), (2, "FAIL"))
        self.assertIn(A0.FEATURES[0], af["features_with_max_abs_delta_above_atol"])
        self.assertGreater(af["per_feature_max_abs_delta"][A0.FEATURES[0]], 1e-6)

        def broken(raw):
            raise A0.A0Error("synthetic bug")
        with mock.patch.object(A0, "from_raw", broken):
            code, s, doc = self.verify("--counterfactual")
        self.assertEqual((code, doc["status"], doc["failure"]["reason"]), (4, "STOP", "a0_error"))

    def test_verify_never_gets(self):
        for p in self.w.cache.iterdir():
            p.unlink()
        code, s, doc = self.verify("--counterfactual")
        self.assertEqual((code, doc["status"], doc["failure"]["reason"]), (2, "FAIL", "jtt1_not_cached_or_invalid"))


class NormalizerSchema(P6Case):
    """The fourth spec schema, jope.dl-normalizer.schema.json (embedded in encodings.SCHEMAS like the other three)."""

    def test_run_normalizer_header_validates_and_tampering_is_refused(self):
        from jsonschema import Draft202012Validator
        self.assertEqual(E.NORMALIZER_ARRAY_ORDER, N.ARRAY_ORDER)
        self.assertEqual(set(E.NORMALIZER_METHOD_KEYS), set(N.METHOD))
        schema = E.SCHEMAS["jope.dl-normalizer.schema.json"]
        Draft202012Validator.check_schema(schema)
        w = RunWorld(self.root)
        out = self.root / "a0"
        code, s = self.cli(*w.run_args(out))
        self.assertEqual(code, 0, s)
        rec = jload(out / "manifests/train_fit.json")["normalizers"][0]
        header = jload(out / f"normalizers/{rec['normalizer_sha256']}.json")
        v = Draft202012Validator(schema)
        self.assertEqual(list(v.iter_errors(header)), [])
        bad = {
            "dev_fit_role": dict(header, fit=dict(header["fit"], partition_role="dev")),
            "extra_key": dict(header, timestamp=1),
            "array_order": dict(header, arrays=[header["arrays"][1], header["arrays"][0]] + header["arrays"][2:]),
            "unknown_map_kind": dict(header, fit=dict(header["fit"], partition_map_kind="p4")),
            "rank_zero": dict(header, dims=dict(header["dims"], rank=0)),
            "short_blob_sha": dict(header, blob=dict(header["blob"], sha256="abc")),
        }
        for name, doc in bad.items():
            self.assertTrue(list(v.iter_errors(doc)), f"{name} must be refused")


class AffineFaithfulness(P6Case):
    """verify --affine-invariance only compares A0 across maps that keep the input's information: a drawn float map
    that merges distinct values (real JTT1 columns hold values like 1e-86 beside 0.01) is replaced by a faithful
    fallback, and the as-drawn comparison is still reported."""

    @staticmethod
    def tiny_column_raw():
        rng = np.random.default_rng(5)
        n_train, n_test, p = 200, 20, 3
        n = n_train + n_test
        X = rng.normal(size=(n, p))
        X[:, 2] = np.where(rng.random(n) < 0.5, 1e-80 * rng.integers(1, 60, n), rng.random(n))
        y = X[:, 0] + rng.normal(size=n)
        labels = (np.arange(n_train) % 2).astype(np.uint8)
        mat = np.column_stack([X, y]).astype("<f8")
        return jtt1.MAGIC + struct.pack("<IIIB", n_train, n_test, p, 2) + labels.tobytes() + mat.tobytes()

    def test_unfaithful_drawn_maps_fall_back_and_are_reported(self):
        raw = self.tiny_column_raw()
        cid = h("ctx:affine-tiny")
        res = E.verify_affine(A0, [cid], lambda c: raw, StubGuard(), roles={cid: "dev"})
        self.assertEqual(res["status"], "PASS", res)
        self.assertLessEqual(res["max_abs_delta"], 1e-6)
        fa = res["faithfulness"]
        self.assertEqual(fa["contexts_with_adjusted_maps"], 1)
        self.assertEqual(fa["map_kinds"], {"drawn": 3, "no_shift": 1, "pow2_scale": 0, "identity": 0})
        adj = fa["adjustments"][0]
        self.assertEqual(adj["context"], cid)
        self.assertEqual([(m["column"], m["kind"]) for m in adj["maps"]], [(2, "no_shift")])
        self.assertLess(adj["maps"][0]["distinct_under_drawn_map"], adj["maps"][0]["distinct"])
        # TOL-04: the random as-drawn detector is gone; the channel is the deterministic train-only exposure test
        self.assertNotIn("as_drawn_unfaithful", res)
        _h, _g, X, y = jtt1.train_view(raw)
        exp = E.upstream_exposure(X, y)
        self.assertEqual((exp["scale"]["exposed"], exp["scale"]["x_columns"], exp["scale"]["y"]), (True, [2], False))
        ue, exposed = E.verify_upstream_exposure([cid], {cid: "dev"}, lambda c: raw, StubGuard(), {})
        self.assertEqual((ue["scale"]["status"], ue["scale"]["contexts"], exposed), ("OPEN", 1, {cid: ["scale"]}))
        sealed = E.verify_affine(A0, [cid], lambda c: raw, StubGuard(), roles={cid: "heldout"})
        self.assertEqual(sealed["faithfulness"]["adjustments"][0]["context"], "<sealed>")
        self.assertNotIn(cid, json.dumps(sealed))

    def test_faithful_affine_fallback_order(self):
        tiny = np.array([0.0, 1e-86, 2e-86, 0.01, 0.5])
        m, kind = E.faithful_affine(tiny, 3.7, 7.3 * 3.7)
        self.assertEqual(kind, "no_shift")
        np.testing.assert_array_equal(m, tiny * 3.7)
        m, kind = E.faithful_affine(np.array([0.1, 0.2, 0.3]), 3.7, 2.0)
        self.assertEqual(kind, "drawn")
        np.testing.assert_array_equal(m, np.array([0.1, 0.2, 0.3]) * 3.7 + 2.0)
        ulps = 1.9 + np.arange(64) * np.spacing(1.9)          # consecutive doubles: x*1.1 merges some of them
        m, kind = E.faithful_affine(ulps, 1.1, 5.0)
        self.assertEqual(kind, "pow2_scale")
        np.testing.assert_array_equal(m, ulps)                 # 2**round(log2 1.1) == 1
        m, kind = E.faithful_affine(np.array([1e308, 1.5e308, 0.0]), 100.0, 50.0)
        self.assertEqual(kind, "identity")                     # every map overflows
        np.testing.assert_array_equal(m, np.array([1e308, 1.5e308, 0.0]))


# ===================================================================================== helpers (T39-T41)
def _forbidden_epoch(root, n_datasets=30):
    ctxs = []
    for d in range(n_datasets):
        cohort = "frozen" if d % 3 else "incoming"
        dh = h(f"fds:{d}")[:16]
        for fold in range(5):
            if cohort == "incoming" and fold == 4 and d % 2:
                continue
            ctxs.append({"context_sha256": h(f"fctx:{d}:{fold}"), "status": "included", "cohort": cohort, "fold": fold, "dataset_hash": dh,
                         "sample_identity_sha256": h(f"fsid:{d}")})
    ctxs.append({"context_sha256": h("fctx:excluded"), "status": "excluded", "cohort": "frozen", "fold": 0, "dataset_hash": h("x")[:16],
                 "sample_identity_sha256": h("x")})
    path = root / "epoch-helpers.json"
    path.write_bytes(common.canonical({"schema": "jope.dl-corpus-epoch/v1", "contexts": ctxs}))
    return path, ctxs


class Helpers(P6Case):
    def test_t40_fake_partition_map(self):
        epath, ctxs = _forbidden_epoch(self.root)
        inc = [c for c in ctxs if c["status"] == "included"]
        out1, out2 = self.root / "map1.json", self.root / "map2.json"
        self.assertEqual(self.cli("fake-partition-map", "--epoch", epath, "--out", out1)[0], 0)
        self.assertEqual(self.cli("fake-partition-map", "--epoch", epath, "--out", out2)[0], 0)
        self.assertEqual(out1.read_bytes(), out2.read_bytes())
        pmap = jload(out1)
        self.assertEqual(out1.read_bytes(), common.canonical(pmap))
        self.assertEqual(set(pmap), {c["context_sha256"] for c in inc})
        meta = jload(str(out1) + ".meta.json")
        self.assertEqual((meta["kind"], meta["seed"], meta["fractions"]), ("fake", "p6-fake-map/v1", {"heldout": 0.2, "dev": 0.2, "diag": 0.1}))
        fam_roles = collections.defaultdict(set)
        for c in inc:
            v = pmap[c["context_sha256"]]
            self.assertEqual(v["family_id"], "fake-fam-" + c["dataset_hash"])
            fam_roles[v["family_id"]].add(v["split_role"])
            if v["split_role"] == "diag":
                self.assertFalse(c["cohort"] == "frozen" and c["fold"] == 0)
        for roles in fam_roles.values():
            self.assertTrue(len(roles) == 1 or roles == {"train_fit", "diag"}, roles)
        F = len(fam_roles)
        held = sum(1 for r in fam_roles.values() if r == {"heldout"})
        dev = sum(1 for r in fam_roles.values() if r == {"dev"})
        self.assertEqual(held, -(-F * 2 // 10))
        self.assertEqual(dev, -(-(F - held) * 2 // 10))
        eligible = [c for c in inc if pmap[c["context_sha256"]]["split_role"] in ("train_fit", "diag")
                    and not (c["cohort"] == "frozen" and c["fold"] == 0)]
        diag = sum(1 for v in pmap.values() if v["split_role"] == "diag")
        self.assertEqual(diag, -(-len(eligible) // 10))
        self.assertEqual(meta["counts"]["diag"], diag)
        # loaders agree, on this map and on refusals
        mine, pin = E.load_partition_map(out1)
        theirs, tpin = programs._load_partition_map(out1)
        self.assertEqual((mine, pin["sha256"]), (theirs, tpin["sha256"]))
        good = next(iter(pmap))
        bad_maps = [b"[1, 2]", b"not json", json.dumps({"abc": {"family_id": "f", "split_role": "dev"}}).encode(),
                    json.dumps({good: {"family_id": "", "split_role": "dev"}}).encode(),
                    json.dumps({good: {"family_id": "f", "split_role": "holdout"}}).encode(),
                    json.dumps({good: {"family_id": "f"}}).encode(), json.dumps({good: "dev"}).encode(),
                    json.dumps({good.upper(): {"family_id": "f", "split_role": "dev"}}).encode()]
        for i, b in enumerate(bad_maps):
            path = self.root / f"bad-{i}.json"
            path.write_bytes(b)
            with self.assertRaises(E.PartitionMapError):
                E.load_partition_map(path)
            with self.assertRaises(programs.ReconcileError):
                programs._load_partition_map(path)
        ok_maps = [{good: {"family_id": "f", "split_role": r}} for r in E.PARTITIONS + E.EXCLUDED_ROLES]
        for i, mp in enumerate(ok_maps):
            path = self.root / f"ok-{i}.json"
            path.write_bytes(json.dumps(mp).encode())
            self.assertEqual(E.load_partition_map(path)[0], programs._load_partition_map(path)[0])
        # outside RUN/tmp: refused, nothing written
        code, s = self.cli("fake-partition-map", "--epoch", epath, "--out", E.RUN_ENCODINGS / "fake-map-must-not-exist.json")
        self.assertEqual((code, s["reason"]), (2, "fake_partition_map_outside_run_tmp"))
        self.assertFalse((E.RUN_ENCODINGS / "fake-map-must-not-exist.json").exists())
        # a fake map can never be labelled p3
        with self.assertRaises(E.Refused):
            E.resolve_map_kind(out1, pmap, "p3")

    def test_t41_sample(self):
        epath, ctxs = _forbidden_epoch(self.root)
        out = self.root / "sample.json"
        code, s = self.cli("sample", "--epoch", epath, "--seed", "20261005", "--per-stratum", "4", "--out", out)
        self.assertEqual(code, 0, s)
        doc = jload(out)
        strata = collections.defaultdict(list)
        for c in ctxs:
            if c["status"] == "included":
                strata[(c["cohort"], c["fold"])].append(c["context_sha256"])
        rng = random.Random(20261005)
        want = []
        for key in sorted(strata):
            want += rng.sample(sorted(strata[key]), 4)
        self.assertEqual(doc["contexts"], want)
        self.assertEqual(len(want), 4 * len(strata))
        self.assertEqual(doc["sha256"], sha(common.canonical(want)))
        self.assertEqual([(x["cohort"], x["fold"]) for x in doc["strata"]], sorted(strata))
        self.assertEqual(E.sample_contexts(jload(epath), seed=20261005, per_stratum=4)["contexts"], want)
        out2 = self.root / "sample2.json"
        self.assertEqual(self.cli("sample", "--epoch", epath, "--out", out2)[0], 0)
        self.assertEqual(out.read_bytes(), out2.read_bytes())

    def test_bench_and_schemas_commands(self):
        out = self.root / "bench.json"
        code, s = self.cli("bench-a0", "--n", "64", "--p", "3,9", "--repeats", "1", "--out", out)
        self.assertEqual(code, 0, s)
        doc = jload(out)
        self.assertEqual([r["p"] for r in doc["results"]], [3, 9])
        sdir = self.root / "schemas"
        code, s = self.cli("schemas", "--out", sdir)
        self.assertEqual(code, 0)
        from jsonschema import Draft202012Validator
        for name in E.SCHEMAS:
            Draft202012Validator.check_schema(jload(sdir / name))


class Hygiene(P6Case):
    FORBIDDEN = {"jope", "gp_worker", "local_gateway", "reuse_cache", "intake_registry"}
    TOP_LEVEL_FORBIDDEN = {"boto3", "botocore", "dltrain.s3io"}

    def test_t39_modules(self):
        canary = "jopedime-regression-research-2k-" + "tabpfn35-unseen-20261001"
        for mod in (A0, N, E):
            src = Path(mod.__file__).read_text()
            self.assertNotIn(canary, src, mod.__name__)
            tree = ast.parse(src)
            names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.add(node.module.split(".")[0])
            self.assertFalse(names & self.FORBIDDEN, mod.__name__)
            top = set()
            for node in tree.body:
                if isinstance(node, ast.Import):
                    top |= {a.name for a in node.names}
                elif isinstance(node, ast.ImportFrom):
                    top |= {node.module or ""} | {f"{node.module}.{a.name}" for a in node.names}
            self.assertFalse({t for t in top if t in self.TOP_LEVEL_FORBIDDEN or t.split(".")[0] in ("boto3", "botocore")}, mod.__name__)

    def test_t39_every_test_is_a_p3case_under_tmpdir(self):
        mod = sys.modules[__name__]
        classes = [v for v in vars(mod).values() if isinstance(v, type) and issubclass(v, unittest.TestCase) and v is not P6Case]
        self.assertTrue(classes)
        for cls in classes:
            self.assertTrue(issubclass(cls, fx.P3Case), cls)
        self.assertTrue(str(self.root.resolve()).startswith(os.path.realpath(TMPDIR) + os.sep))
        self.assertTrue(self.root.name.startswith(E.SAMPLE_DIR_PREFIX))
        self.assertIsNotNone(E.sample_root(self.root / "x"))
        self.assertIsNone(E.sample_root(Path(TMPDIR) / "other" / "x"))
        self.assertIsNone(E.sample_root(E.RUN_ENCODINGS / "A0"))

    def test_encodings_top_level_imports_stay_light(self):
        tree = ast.parse(Path(E.__file__).read_text())
        mods = {n.module for n in tree.body if isinstance(n, ast.ImportFrom)} | {a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
        self.assertNotIn("dltrain.s3io", mods)
        imported = {a.name for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "dltrain" for a in n.names}
        self.assertEqual(imported, {"common", "cpuguard", "jtt1"})


if __name__ == "__main__":
    unittest.main()

"""Version-bound TRAIN transport shared by H206 native search and learning."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import struct
from dataclasses import asdict, dataclass

import numpy as np


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    return sha(canonical(value))


def require_hash(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("required SHA256 identity missing/malformed")


def array_hash(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    return sha(canonical({"dtype": a.dtype.str, "shape": list(a.shape)}) + a.tobytes())


@dataclass(frozen=True)
class ObjectRef:
    bucket: str
    key: str
    version_id: str
    sha256: str

    def __post_init__(self):
        require_hash(self.sha256)
        if not all(
            isinstance(v, str) and v and v != "null" for v in (self.bucket, self.key, self.version_id)
        ):
            raise ValueError("exact bucket/key/non-null VersionId required")

    def read(self, store) -> bytes:
        data, version = store.get(self)
        if version != self.version_id or sha(data) != self.sha256:
            raise ValueError("object version/body SHA mismatch")
        return data


@dataclass(frozen=True)
class ArraySpec:
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    def __post_init__(self):
        require_hash(self.sha256)

    def check(self, value):
        if tuple(value.shape) != tuple(self.shape) or value.dtype.str != self.dtype:
            raise ValueError("tensor shape/dtype mismatch")
        if array_hash(value) != self.sha256 or not np.isfinite(value).all():
            raise ValueError("tensor content hash/nonfinite mismatch")


@dataclass(frozen=True)
class Context:
    context_id: str
    dataset_id: str
    family_id: str
    task: str
    role: str
    train: ObjectRef
    cut_sha256: str
    tensors: ObjectRef
    arrays: dict[str, ArraySpec]
    programs: tuple[str, ...]
    seed: int
    actor_sha256: str
    proposal_source_sha256: str
    evaluator_sha256: str
    native_binary_sha256: str
    environment_sha256: str
    protocol_sha256: str
    encoder_sha256: str
    encoder_model_sha256: str
    original_train: ObjectRef | None = None
    derivation: ObjectRef | None = None
    native_encoding: str = "jtt1-gzip"

    def __post_init__(self):
        if self.native_encoding not in ("jtt1-gzip", "jtt1-raw"):
            raise ValueError("explicit approved native TRAIN encoding required")
        for name in (
            "context_id",
            "cut_sha256",
            "actor_sha256",
            "proposal_source_sha256",
            "evaluator_sha256",
            "native_binary_sha256",
            "environment_sha256",
            "protocol_sha256",
            "encoder_sha256",
            "encoder_model_sha256",
        ):
            require_hash(getattr(self, name))
        if self.task != "regression" or self.role not in ("FIT", "B", "A"):
            raise ValueError("regression FIT/B/A identity required")
        if not self.dataset_id or not self.family_id or self.family_id.lower() in ("none", "null", "unknown"):
            raise ValueError("resolved dataset/family required")
        if type(self.seed) is not int or not self.programs or any(not p.strip() for p in self.programs):
            raise ValueError("seed and nonempty ordered DSL slate required")
        if len(set(self.programs)) != len(self.programs):
            raise ValueError("distinct DSL slate required")

    @classmethod
    def from_dict(cls, value):
        row = dict(value)
        row["train"] = ObjectRef(**row["train"])
        row["tensors"] = ObjectRef(**row["tensors"])
        for name in ("original_train", "derivation"):
            if row.get(name) is not None:
                row[name] = ObjectRef(**row[name])
        row["arrays"] = {
            k: ArraySpec(tuple(v["shape"]), v["dtype"], v["sha256"]) for k, v in row["arrays"].items()
        }
        row["programs"] = tuple(row["programs"])
        return cls(**row)

    @property
    def identity(self):
        return digest(asdict(self))


class Scope:
    """Normalized owner-produced role + transitive-exclusion artifacts, hash pinned.

    This is a consumer contract, not a replacement admission/qualification receipt.
    The actual normalized artifacts and their approved hashes remain upstream pins.
    """

    def __init__(self, roles: bytes, exclusion: bytes, *, roles_sha256, exclusion_sha256):
        require_hash(roles_sha256)
        require_hash(exclusion_sha256)
        if sha(roles) != roles_sha256 or sha(exclusion) != exclusion_sha256:
            raise ValueError("role/exclusion artifact SHA mismatch")
        self.roles = json.loads(roles)
        closure = json.loads(exclusion)
        if (
            self.roles["schema"] != "h206-role-contract-v1"
            or closure["schema"] != "h206-exclusion-contract-v1"
        ):
            raise ValueError("normalized role/exclusion contract required")
        self.sha256 = digest({"roles": roles_sha256, "exclusion": exclusion_sha256})
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        for a, b in closure["links"]:
            parent[find(a)] = find(b)
        self.component = find
        self._parent = parent
        self.protected = {find(x) for x in closure["protected"]}
        self.exclusion_sha256 = exclusion_sha256
        self.reference_families = tuple(self.roles["reference_families"])
        if not self.reference_families or any(
            not isinstance(x, str) or not x for x in self.reference_families
        ):
            raise ValueError("pinned reference family inventory required for novelty accounting")
        seen = {}
        for row in self.roles["contexts"].values():
            component = find("family:" + row["family_id"])
            if component in seen and seen[component] != row["role"]:
                raise ValueError("transitive family leakage across FIT/B/A roles")
            seen[component] = row["role"]

    def component_hash(self, family_id):
        root = self.component("family:" + family_id)
        members = sorted(k for k in list(self._parent) if self.component(k) == root)
        return digest(members)

    def admit_metadata(self, context_id, role):
        """Authorize role/alias metadata before opening even a template body."""
        row = self.roles["contexts"].get(context_id)
        if row is None or row["role"] != role:
            raise ValueError("context role differs from approved role manifest")
        dataset, family = "dataset:" + row["dataset_id"], "family:" + row["family_id"]
        if self.component(dataset) != self.component(family) or self.component(family) in self.protected:
            raise ValueError("unjoined or transitively protected family")
        return row

    def verify(self, context: Context, role: str):
        data_fields = ["context_id", "dataset_id", "family_id", "task", "role", "train", "cut_sha256"]
        data_fields += [
            name for name in ("original_train", "derivation") if getattr(context, name) is not None
        ]
        if context.native_encoding != "jtt1-gzip":
            data_fields.append("native_encoding")
        data_identity = {k: v for k, v in json.loads(canonical(asdict(context))).items() if k in data_fields}
        if context.role != role or self.roles["contexts"].get(context.context_id) != data_identity:
            raise ValueError("context role/identity differs from approved role manifest")
        for name in (
            "proposal_source_sha256",
            "evaluator_sha256",
            "native_binary_sha256",
            "environment_sha256",
            "protocol_sha256",
            "encoder_sha256",
            "encoder_model_sha256",
        ):
            if self.roles["sources"].get(name) != getattr(context, name):
                raise ValueError("source/evaluator/environment/protocol differs from approved pins")
        dataset, family = "dataset:" + context.dataset_id, "family:" + context.family_id
        if self.component(dataset) != self.component(family) or self.component(family) in self.protected:
            raise ValueError("unjoined or transitively protected family")
        return True


SHAPES = {
    "vec256": (1, 256),
    "col_lm": (1, 64, 17),
    "tokens_10x32": (1, 10, 32),
    "mask_10": (1, 10),
    "ctx_feat": (1, 584),
    "scale": (1,),
    "revealed_indices": (1, 4),
    "revealed_raw_f": (1, 4),
    "revealed_gate": (1, 4),
}


@dataclass
class LoadedContext:
    context: Context
    tensors: dict[str, np.ndarray]
    train_bytes: bytes
    scope_sha256: str
    reference_families: tuple[str, ...]
    loaded_identity: str
    family_component_sha256: str

    def verify(self):
        if (
            self.context.identity != self.loaded_identity
            or sha(self.train_bytes) != self.context.train.sha256
        ):
            raise ValueError("loaded context/TRAIN identity changed after admission")
        if set(self.tensors) != set(self.context.arrays):
            raise ValueError("missing/unexpected tensors")
        for key, value in self.tensors.items():
            self.context.arrays[key].check(value)


def native_train_body(context, data):
    if context.native_encoding == "jtt1-raw":
        return data
    if context.native_encoding == "jtt1-gzip":
        return gzip.decompress(data)
    raise ValueError("explicit approved native TRAIN encoding required")


def admitted_original_derivation(context, scope, store):
    """Verify approved original bytes/linkage; semantic CSV derivation stays with ARGON.

    The role authority must already pin this actual producer-qualified witness.
    Matching hashes/booleans alone never qualify the producer or arbitrary CSV
    decoding. The actual input qualification and upstream row/dtype proof remain
    mandatory, separate from this consumer byte/shape/cut/linkage verification.
    """
    if context.original_train is None or context.derivation is None:
        raise ValueError("complete original TRAIN/derivation refs required before payload access")
    doc = json.loads(context.derivation.read(store))
    expected = {
        "schema": "h206-original-train-derivation-v1",
        "context_id": context.context_id,
        "dataset_id": context.dataset_id,
        "family_id": context.family_id,
        "task": "regression",
        "original_train": asdict(context.original_train),
        "native_train": asdict(context.train),
        "tensors": asdict(context.tensors),
        "cut_sha256": context.cut_sha256,
        "encoder_sha256": context.encoder_sha256,
        "encoder_model_sha256": context.encoder_model_sha256,
        "native_dtype": "<f8",
        "native_encoding": context.native_encoding,
        "query_rows": 0,
    }
    if any(doc.get(key) != value for key, value in expected.items()):
        raise ValueError("original TRAIN/derivation identity or cut mismatch")
    for key in ("train_derivation_source_sha256", "tensor_derivation_source_sha256"):
        require_hash(doc.get(key))
        if doc[key] != scope.roles["sources"].get(key):
            raise ValueError("original TRAIN/encoder derivation source is not approved")
    require_hash(doc.get("original_rows_sha256"))
    require_hash(doc.get("native_matrix_sha256"))
    context.original_train.read(store)  # Actual named VersionId GET/body SHA, never a URI label.
    return doc


def load_context(
    context: Context, scope: Scope, store, *, role="FIT", require_original_provenance=False
) -> LoadedContext:
    scope.verify(context, role)  # BEFORE any object body access.
    provenance = admitted_original_derivation(context, scope, store) if require_original_provenance else None
    payload = context.tensors.read(store)
    with np.load(io.BytesIO(payload), allow_pickle=False) as bundle:
        tensors = {k: bundle[k].copy() for k in bundle.files}
    loaded = LoadedContext(
        context,
        tensors,
        context.train.read(store),
        scope.sha256,
        tuple(scope.component_hash(f) for f in scope.reference_families),
        context.identity,
        scope.component_hash(context.family_id),
    )
    loaded.verify()
    if provenance is not None:
        body = native_train_body(context, loaded.train_bytes)
        if len(body) < 17 or body[:4] != b"JTT1":
            raise ValueError("derived native TRAIN is not the approved JTT1 body")
        nt, nq, columns, folds = struct.unpack_from("<IIIB", body, 4)
        row_ids = provenance.get("source_row_ids", [])
        if (
            nq != 0
            or folds != 2
            or nt < 50
            or not 1 <= columns <= 512
            or len(body) != 17 + nt + 8 * nt * (columns + 1)
            or provenance.get("native_shape") != [nt, columns + 1]
            or not isinstance(row_ids, list)
            or len(row_ids) != nt
            or any(type(i) is not int or i < 0 for i in row_ids)
            or len(set(row_ids)) != nt
            or (nt + 10) * columns > 16_000_000
            or sha(body[17 : 17 + nt]) != context.cut_sha256
            or sha(body[17 + nt :]) != provenance["native_matrix_sha256"]
            or not np.isfinite(np.frombuffer(body, dtype="<f8", offset=17 + nt)).all()
        ):
            raise ValueError("derived native TRAIN row/dtype/cut witness mismatch")
    k = len(context.programs)
    shapes = SHAPES | {
        "program": (k, 412),
        "projection": (k, 16),
        "base_prior_cost": (1, k),
        "sos_prior": (1, k, 2),
    }
    if set(tensors) != set(shapes):
        raise ValueError("incomplete native/encoder tensor closure")
    for name, shape in shapes.items():
        if tensors[name].shape != shape:
            raise ValueError("invalid native tensor shape: " + name)
        dtype = (
            np.dtype("bool")
            if name == "mask_10"
            else np.dtype("int64")
            if name == "revealed_indices"
            else np.dtype("float32")
        )
        if tensors[name].dtype != dtype:
            raise ValueError("invalid native tensor dtype: " + name)
    revealed = tensors["revealed_indices"]
    if np.any(revealed < 0) or np.any(revealed >= k) or len(set(revealed[0])) != 4:
        raise ValueError("revealed candidate indices invalid")
    if not tensors["mask_10"].any():
        raise ValueError("empty encoder mask")
    return loaded


def guard_cpu(storage_path):
    """Current CPU/numeric/memory/disk preflight; no GPU probe or queue release."""
    import os
    from pathlib import Path

    import psutil
    import torch

    from research.flywheel_selfplay.hwave_numeric_threads import (
        admit_numeric_threads,
        capture_numeric_threads,
    )

    process = psutil.Process()
    cpu = psutil.cpu_percent(interval=0.05)
    disk = psutil.disk_usage(Path(storage_path).resolve())
    available = psutil.virtual_memory().available
    numeric = admit_numeric_threads([capture_numeric_threads(torch)], os_threads=process.num_threads())
    affinity = len(os.sched_getaffinity(0))
    rss = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)] if p.is_running())
    if (
        os.getpriority(os.PRIO_PROCESS, 0) != 19
        or cpu > 85
        or available < 16 * 2**30
        or rss >= 16 * 2**30
        or disk.used / (disk.used + disk.free) >= 0.938
        or numeric["numeric_threads_upper_bound"] > min(4, affinity)
    ):
        raise RuntimeError("current H206 CPU/numeric/memory/disk resource refusal")
    return {
        "cpu_percent": cpu,
        "available_bytes": available,
        "disk_used_fraction": disk.used / (disk.used + disk.free),
        "effective_affinity_cores": affinity,
        "numeric": numeric,
        "nice": 19,
        "tree_rss_bytes": rss,
    }

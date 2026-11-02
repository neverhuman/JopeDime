"""Pinned --fit-s1 calls, exact-version persistence and semantic outcome cache."""

from __future__ import annotations

import gzip
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from research.flywheel_selfplay.rust_evaluator import (
    compute_s1_metrics_from_inner_cv,
    propose_flywheel_candidates,
)

from .contracts import ObjectRef, canonical, digest, guard_cpu, sha


class S3Store:
    def __init__(self, client, bucket: str, prefix: str):
        self.client, self.bucket, self.prefix = client, bucket, prefix.rstrip("/")

    def get(self, ref: ObjectRef):
        result = self.client.get_object(Bucket=ref.bucket, Key=ref.key, VersionId=ref.version_id)
        return result["Body"].read(), result.get("VersionId")

    def put(self, name: str, data: bytes) -> ObjectRef:
        key = self.prefix + "/" + name
        result = self.client.put_object(Bucket=self.bucket, Key=key, Body=data, IfNoneMatch="*")
        ref = ObjectRef(self.bucket, key, result.get("VersionId"), sha(data))
        ref.read(self)  # Commit only after an actual exact-version GET and byte check.
        return ref


class OutcomeCache:
    def __init__(self, root: Path, cap_bytes=40_000_000_000):
        self.root = Path(root)
        repo = Path(__file__).resolve().parents[3]
        if (
            self.root.is_symlink()
            or self.root.resolve().is_relative_to(repo)
            or self.root.resolve().is_relative_to(Path("/tmp"))
        ):
            raise ValueError("owned cache outside checkout and /tmp required")
        self.cap_bytes = cap_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = self.hit_bytes = self.miss_bytes = 0

    def get(self, identity):
        path = self.root / (digest(identity) + ".json")
        if path.is_symlink():
            raise ValueError("native cache cannot follow symlinks")
        if not path.exists():
            self.misses += 1
            return None
        data = path.read_bytes()
        doc = json.loads(data)
        if doc.get("identity") != identity or doc.get("records_sha256") != digest(doc.get("records")):
            raise ValueError("cached native outcome identity/content mismatch")
        self.hits += 1
        self.hit_bytes += len(data)
        return doc["records"]

    def put(self, identity, records):
        data = canonical({"identity": identity, "records": records, "records_sha256": digest(records)})
        path = self.root / (digest(identity) + ".json")
        if sum(p.stat().st_size for p in self.root.glob("*.json")) + len(data) > self.cap_bytes:
            raise RuntimeError("owned native cache cap exhausted")
        with path.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        self.miss_bytes += len(data)


class RustS1Evaluator:
    """The actual native CLI adapter. Fixture binaries never qualify real labels."""

    def __init__(self, binary: Path, *, binary_sha256: str, scratch: Path, evaluator_sha256: str):
        self.binary = Path(binary)
        self.binary_sha256 = binary_sha256
        if sha(self.binary.read_bytes()) != binary_sha256:
            raise ValueError("native evaluator binary SHA mismatch")
        if (
            sha(Path(inspect.getsourcefile(compute_s1_metrics_from_inner_cv)).read_bytes())
            != evaluator_sha256
        ):
            raise ValueError("native metric/evaluator source SHA mismatch")
        self.evaluator_sha256 = evaluator_sha256
        self.environment = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "binary_sha256": binary_sha256,
            "numeric_threads": 1,
            "metric_source_sha256": evaluator_sha256,
        }
        self.environment_sha256 = digest(self.environment)
        self.protocol_sha256 = digest(
            {
                "schema": "h206-native-S1-TRAIN-v1",
                "folds": 2,
                "query_padding": "first-10-TRAIN-rows-cyclic-zero-query-target-v1",
                "min_train_rows": 50,
                "min_inner_fold_rows": 25,
                "max_columns": 512,
                "max_cells": 16_000_000,
                "metric_source_sha256": evaluator_sha256,
            }
        )
        self.scratch = Path(scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.calls = self.failures = 0

    def evaluate(self, loaded, index: int, *, timeout_s: float):
        loaded.verify()
        resources = guard_cpu(self.scratch)
        context = loaded.context
        if (
            context.role != "FIT"
            or context.evaluator_sha256 != self.evaluator_sha256
            or context.native_binary_sha256 != self.binary_sha256
            or context.environment_sha256 != self.environment_sha256
            or context.protocol_sha256 != self.protocol_sha256
        ):
            raise ValueError("native TRAIN role/evaluator identity mismatch")
        if sha(self.binary.read_bytes()) != self.binary_sha256:
            raise ValueError("native binary changed before call")
        # Consume ONLY the admitted TRAIN body. Never derive/read an outer payload.
        body = gzip.decompress(loaded.train_bytes)
        if body[:4] != b"JTT1":
            raise ValueError("admitted TRAIN body is not JTT1")
        import struct

        nt, nq, p, folds = struct.unpack_from("<IIIB", body, 4)
        expected = 17 + nt + 8 * (nt + nq) * (p + 1)
        if folds != 2 or nt < 50 or not 1 <= p <= 512 or (nt + 10) * p > 16_000_000 or len(body) != expected:
            raise ValueError("TRAIN-only JTT1 shape/cut mismatch")
        # A body with query rows is not a TRAIN-only source, even with zero labels.
        if nq != 0:
            raise ValueError("native input contains query/outer rows")
        fold_bytes = body[17 : 17 + nt]
        if (
            sha(fold_bytes) != context.cut_sha256
            or not set(fold_bytes) <= {0, 1}
            or min(fold_bytes.count(0), fold_bytes.count(1)) < 25
        ):
            raise ValueError("native TRAIN cut/fold identity mismatch")
        # Existing S1 expects >=10 query rows even for --fit-s1. Construct these
        # solely from admitted TRAIN X, with ZERO query targets; never fetch outer
        # data. This prospective bridge still needs exact native parity proof.
        train_matrix = np.frombuffer(body, dtype="<f8", offset=17 + nt).reshape(nt, p + 1)
        if not np.isfinite(train_matrix).all():
            raise ValueError("nonfinite TRAIN table")
        queries = train_matrix[np.arange(10) % nt].copy()
        queries[:, -1] = 0.0
        body = (
            struct.pack("<4sIIIB", b"JTT1", nt, 10, p, 2)
            + fold_bytes
            + train_matrix.tobytes()
            + queries.tobytes()
        )
        dsl = context.programs[index]
        job = self.scratch / (
            digest({"context": context.identity, "program": dsl}) + "-" + str(time.time_ns())
        )
        job.mkdir()
        input_path, candidate_path, output_path = (
            job / "train.jtt1",
            job / "candidate.json",
            job / "result.json",
        )
        input_path.write_bytes(body)
        candidate_path.write_bytes(
            canonical(
                {
                    "dsl": dsl,
                    "program_sha256": sha(dsl.encode()),
                    "birth_seq": 0,
                    "parents": [],
                    "operator": "H206_SEARCH",
                }
            )
        )
        env = os.environ | {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "TMPDIR": str(job),
        }
        started = time.monotonic()
        self.calls += 1
        raw = b""
        returncode, stderr = None, b""
        try:
            proc = subprocess.run(
                [
                    "nice",
                    "-n",
                    "19",
                    str(self.binary),
                    "--fit-s1",
                    str(input_path),
                    str(candidate_path),
                    str(output_path),
                ],
                env=env,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
            returncode, stderr = proc.returncode, proc.stderr
            if proc.returncode != 0:
                raise RuntimeError("native exit " + str(proc.returncode))
            raw = output_path.read_bytes()
            metrics = compute_s1_metrics_from_inner_cv(dsl, json.loads(raw).get("inner_cv", []))
            if metrics["status"] != "completed":
                raise ValueError("native invalid inner-CV result")
        except (subprocess.TimeoutExpired, RuntimeError, ValueError, OSError) as exc:
            self.failures += 1
            metrics = {"status": "failed", "fitness": None, "cv_gate_passed": False, "error": str(exc)}
        return {
            "index": index,
            "scratch_job": str(job),
            "program_sha256": sha(dsl.encode()),
            "metrics": metrics,
            "returncode": returncode,
            "stderr": stderr.decode(errors="replace")[:4096],
            "stderr_sha256": sha(stderr),
            "raw_output_sha256": sha(raw),
            "raw_output": raw.decode(errors="replace"),
            "binary_sha256": self.binary_sha256,
            "input_sha256": sha(body),
            "original_train_sha256": context.train.sha256,
            "protocol_sha256": self.protocol_sha256,
            "environment_sha256": self.environment_sha256,
            "resources": resources,
            "wall_seconds": time.monotonic() - started,
        }


def collect_teacher(loaded, actor_costs, k, native, cache, store, prefix, budget, *, timeout_s=25.0):
    """Persist the exact labels that the learner will consume, including failures."""
    loaded.verify()
    context = loaded.context
    if context.role != "FIT" or k not in (1, 16, 64, 256) or len(context.programs) != 256:
        raise ValueError("fixed H206 FIT slate/k required")
    costs = np.asarray(actor_costs)
    if costs.shape != (256,) or not np.isfinite(costs).all():
        raise ValueError("finite full actor slate required")
    order = sorted(range(256), key=lambda i: (float(costs[i]), sha(context.programs[i].encode())))
    identity = {
        "context": context.identity,
        "scope": loaded.scope_sha256,
        "order": order,
        "k": k,
        "native_binary": native.binary_sha256,
        "evaluator": native.evaluator_sha256,
    }
    records = cache.get(identity)
    if records is None:
        records = []
        try:
            for index in order[:k]:
                with budget.measure("native_search"):
                    records.append(native.evaluate(loaded, index, timeout_s=timeout_s))
        except BaseException:
            store.put(
                prefix + "/partial-outcomes.json", canonical({"identity": identity, "records": records})
            )
            raise
        cache.put(identity, records)
    if len(records) != k or [r["index"] for r in records] != order[:k]:
        raise ValueError("native cache slate/candidate coverage mismatch")
    for record in records:
        index = record["index"]
        if (
            record["program_sha256"] != sha(context.programs[index].encode())
            or record["binary_sha256"] != native.binary_sha256
        ):
            raise ValueError("native program/binary receipt mismatch")
        raw = record["raw_output"].encode()
        if sha(raw) != record["raw_output_sha256"]:
            raise ValueError("cached raw native result hash mismatch")
        if record["metrics"]["status"] == "completed":
            expected = compute_s1_metrics_from_inner_cv(
                context.programs[index], json.loads(raw).get("inner_cv", [])
            )
            if record["metrics"] != expected:
                raise ValueError("cached fitness differs from raw native output")
    eligible = [r for r in records if r["metrics"]["status"] == "completed"]
    chosen = (
        min(eligible, key=lambda r: (r["metrics"]["fitness"], order.index(r["index"])))["index"]
        if eligible
        else None
    )
    teacher = {
        "schema": "h206-native-teacher-v1",
        "identity": identity,
        "context_identity": context.identity,
        "records": records,
        "records_sha256": digest(records),
        "target_index": chosen,
    }
    with budget.measure("teacher_s3_flush_verify"):
        ref = store.put(prefix + "/teacher.json", canonical(teacher))
    if chosen is None:
        error = RuntimeError("all native prefix candidates failed; persisted, no teacher substitution")
        error.receipt = ref
        raise error
    # This is an actual persisted/read-back teacher, not a disconnected target array.
    if json.loads(ref.read(store)) != teacher:
        raise ValueError("persisted teacher differs")
    return ref


def propose_slate(parents, landmark48, champions, family_id, seed, *, source_sha256):
    """Actual inherited GP proposal path; no cached-pool resampling substitution."""
    path = Path(inspect.getsourcefile(propose_flywheel_candidates))
    if sha(path.read_bytes()) != source_sha256:
        raise ValueError("proposal source content identity mismatch")
    if np.asarray(landmark48).shape != (48,) or not np.isfinite(landmark48).all():
        raise ValueError("TRAIN-derived landmark48 required")
    distinct = {}
    attempts = 0
    while len(distinct) < 256 and attempts < 4096:
        batch = propose_flywheel_candidates(
            parents,
            landmark48,
            champions,
            family_id,
            n_proposals=256,
            seed=seed + attempts,
        )
        if not batch:
            break
        for proposal in batch:
            attempts += 1
            dsl = proposal["dsl"].strip()
            distinct.setdefault(sha(dsl.encode()), proposal)
            if len(distinct) == 256 or attempts == 4096:
                break
    if len(distinct) != 256:
        raise RuntimeError(
            f"could not obtain 256 distinct native GP candidates within4096 attempts: unique={len(distinct)} attempts={attempts}"
        )
    return {
        "programs": list(distinct.values()),
        "attempts": attempts,
        "proposal_source_sha256": source_sha256,
        "seed": seed,
    }

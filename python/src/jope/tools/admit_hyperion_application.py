#!/usr/bin/env python3
"""Bind a numerically qualified Hyperion backend to an unchanged context assembly."""

import argparse
import hashlib
import json
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--jobs", required=True)
p.add_argument("--reference-spec", required=True)
p.add_argument("--candidate-spec", required=True)
p.add_argument("--parity", required=True)
p.add_argument("--assembly", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads((a.root / path).read_text())


def binding(path):
    return dict(path=path, sha256=sha(a.root / path))


old, new = read(a.reference_spec), read(a.candidate_spec)
changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
assert changed == ["persistent_binary", "persistent_binary_sha256"], changed
for key in [
    "binary",
    "weights",
    "config",
    "source_manifest",
    "loader",
    "runtime_manifest",
    "persistent_binary",
]:
    assert sha(a.root / new[key]) == new[key + "_sha256"]
parity = read(a.parity)
assert parity["passed"] and parity["frozen_contract_unchanged"]
assert parity["reference_spec_sha256"] == sha(a.root / a.reference_spec)
assert parity["candidate_spec_sha256"] == sha(a.root / a.candidate_spec)
assert parity["jobs_sha256"] == sha(a.root / a.jobs)
assert parity["original_final_test_targets_opened"] is False
assert parity["shared_input_comparisons"] == len(parity["comparisons"]) >= 71
assert parity["max_abs_error"] <= parity["tolerance_max_abs"] <= 3e-4
assert len(parity["verified_encodings"]) == len(read(a.jobs)["jobs"])
for row in parity["verified_encodings"]:
    path = a.root / row["path"]
    assert sha(path) == row["sha256"] and sha(path.with_name("hyperion.json")) == row["receipt_sha256"]
assembly = read(a.assembly)
assert assembly["final_test_targets_used"] is False
registry = assembly["registry"]
assert sha(a.root / registry["path"]) == registry["sha256"]
result = dict(
    schema="parallel.context_backend_admission.v1",
    timestamp=int(time.time()),
    passed=True,
    jobs=binding(a.jobs),
    assembly=binding(a.assembly),
    registry=registry,
    reference_hyperion_spec=binding(a.reference_spec),
    candidate_hyperion_spec=binding(a.candidate_spec),
    numerical_parity=binding(a.parity),
    changed_spec_fields=changed,
    unchanged_semantic_channels=True,
    frozen_weights_unchanged=True,
    max_abs_error=parity["max_abs_error"],
    shared_input_comparisons=parity["shared_input_comparisons"],
    projection_refitted=False,
    cohort_statistics_refitted=False,
    qualification=True,
    final_test_targets_used=False,
)
out = a.root / a.output
out.parent.mkdir(parents=True, exist_ok=True)
assert not out.exists(), "Admission receipts are versioned"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))

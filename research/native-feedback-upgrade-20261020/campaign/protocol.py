"""Standard research records. Importable without NumPy, torch, or a device."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = "jope.research-result/v1"
STATES = {"planned", "blocked", "running", "complete", "failed", "stopped"}
BASELINES = {"native17773_trie", "native17773_qhead", "native429_trie", "native429_qhead"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def publish(path, value):
    """Create-only immutable record; a failed or completed run is never overwritten."""
    with Path(path).open("xb") as f:
        f.write(canonical(value) + b"\n")
        f.flush()
        import os
        os.fsync(f.fileno())


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def validate_result(r):
    if r.get("schema") != SCHEMA or r.get("state") not in STATES:
        raise ValueError("unknown result schema/state")
    for key in ("run_id", "branch", "source_commit", "preregistration_sha256", "concept"):
        if not isinstance(r.get(key), str) or not r[key]:
            raise ValueError("missing identity: " + key)
    identity = r.get("comparison_identity")
    if not isinstance(identity, dict) or not all(identity.get(k) for k in (
        "task", "input_archive_sha256", "features_sha256", "split_sha256",
        "evaluation_role", "evaluator_sha256", "vocabulary_sha256", "budget")):
        raise ValueError("incomplete comparison identity")
    if r.get("comparison_identity_sha256") != digest(identity):
        raise ValueError("comparison identity digest mismatch")
    if not re.fullmatch('[0-9a-f]{40}', r['source_commit']) or not re.fullmatch('[0-9a-f]{64}', r['preregistration_sha256']):
        raise ValueError('full source/preregistration hashes required')
    for key in ('input_archive_sha256', 'features_sha256', 'split_sha256', 'evaluator_sha256', 'vocabulary_sha256'):
        if not isinstance(identity[key], str) or not re.fullmatch('[0-9a-f]{64}', identity[key]):
            raise ValueError('full comparison hash required: ' + key)
    counts = r.get("counts", {})
    for key in ("optimizer_updates", "fresh_gp_fits", "historical_replay_slots", "cache_reads", "failures"):
        if type(counts.get(key)) is not int or counts[key] < 0:
            raise ValueError("missing/nonintegral counter: " + key)
    if r.get("hwm_promotion"):
        raise ValueError("research runner cannot promote a production HWM")
    if r.get("scope") == "historical_replay" and counts["fresh_gp_fits"]:
        raise ValueError("historical replay cannot claim fresh GP fits")
    if r["state"] == "complete":
        if not BASELINES.issubset(r.get("metrics", {})):
            raise ValueError("both native checkpoint controls required")
        for name, m in r["metrics"].items():
            if not all(finite(m.get(k)) for k in ("parent_mean_regret", "parent_mean_quality")):
                raise ValueError("nonfinite/unsupported metric " + name)
            if not 0 <= m["parent_mean_regret"] <= 1:
                raise ValueError("regret outside declared normalization")
        if counts["optimizer_updates"] > 0:
            hashes = r.get("parameter_hashes", {})
            if not hashes or any(v["before"] == v["after"] for v in hashes.values()):
                raise ValueError("nonzero updates require changed parameter hashes")
            if any(not re.fullmatch('[0-9a-f]{64}', v[k]) for v in hashes.values() for k in ('before', 'after')):
                raise ValueError('full parameter hashes required')
        if r.get("primary", {}).get("parents", 0) < 2:
            raise ValueError("parent-level uncertainty is required")
    return r


def comparable(a, b):
    """Never strip native identity fields or compare unlike cuts/roles/budgets."""
    validate_result(a)
    validate_result(b)
    if a["comparison_identity"] != b["comparison_identity"]:
        raise ValueError("incompatible comparison identities; retain separate rows")
    if a["state"] != "complete" or b["state"] != "complete":
        raise ValueError("unfinished results are not quality measurements")
    return True


def next_action(state, *, admitted, reviewed, failed_attempts=0):
    """Scheduling decisions only. No launcher, unit, process, or signal operation."""
    if state not in STATES:
        raise ValueError("unknown state")
    if state == "running":
        return "verify_optimizer_checkpoint_guard_and_cost_progress"
    if state in ("complete", "failed", "stopped"):
        return "study_immutable_results_and_freeze_new_hypothesis"
    if not admitted or not reviewed or failed_attempts:
        return "blocked_name_owner_and_exact_dependency"
    return "ready_for_existing_owner_launch_once"


def validate_plan(p):
    if p.get("schema") != "jope.research-wave/v1":
        raise ValueError("unknown wave schema")
    if p.get("concept") not in ("composite_control", "top_listwise", "bootstrap_thompson"):
        raise ValueError("unknown concept")
    if p.get("seeds") != [41, 73, 109] or p.get("arms") != ["continuation", "uniform_feedback", "targeted_feedback"]:
        raise ValueError("unmatched seed/control plan")
    if p.get("evaluation_partition") != "calibration" or p.get("scope") != "historical_replay":
        raise ValueError("this wave is historical development, not fresh GP or sealed confirmation")
    if p.get("evaluation_exposure") != "development_once_per_frozen_wave":
        raise ValueError("assessment exposure must be explicit")
    if p.get("microbatch_contexts", 0) < 1 or p["microbatch_contexts"] > p.get("effective_batch_contexts", 0):
        raise ValueError("invalid accumulation/packing")
    for k in ("initial_updates", "updates_per_wave", "waves", "wall_seconds"):
        if type(p.get(k)) is not int or p[k] <= 0:
            raise ValueError("invalid finite work bound " + k)
    if p["initial_updates"] > 192 or p["updates_per_wave"] > 192 or p["waves"] > 3 or p["wall_seconds"] > 3600:
        raise ValueError("exceeds first-wave scientific envelope")
    for k in ("input_pins", "source_sha256", "native_checkpoint_parameters"):
        if not isinstance(p.get(k), dict) or not p[k]:
            raise ValueError("missing immutable " + k)
    return p

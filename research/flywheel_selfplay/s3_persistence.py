#!/usr/bin/env python3
"""
Phase 4b: S3 Schema Persistence (`jope.hwm/v1`, `jope.score/v1`, `gp_distilled/v1`)
and Smart Data Retention / Garbage Pruning (`jope-learning::retention` parity).

Ensures that as self-play generates new GP trees across rounds:
  1. Only value-add records are persisted:
     - New High-Water Mark (`jope.hwm/v1`) winners that beat or match the incumbent HWM.
     - Top-K (`K=4`) Pareto-optimal gate-passing candidates per context (`jope.score/v1`).
     - Up to `M=4` informative hard-negative contrast candidates per context (for contrastive training).
     - All duplicate `program_sha256` evaluations, timeout/syntax failures, and dominated garbage
       are pruned before S3 upload.
  2. Persists canonical artifacts locally under `/mnt/jope-archive/flywheel-20261023/s3_mirror/`
     and uploads them to `s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/`
     via `AWS_PROFILE=veox-jepson-prod`.
"""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

LOCAL_S3_MIRROR = Path("/mnt/jope-archive/flywheel-20261023/s3_mirror")
EXTERNAL_SYNC_RECEIPTS_DIR = Path("/mnt/jope-archive/flywheel-20261023/s3_sync_receipts")
S3_BUCKET_PREFIX = "s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023"
AWS_PROFILE = "veox-jepson-prod"


def apply_smart_retention_filter(
    context_sha256: str,
    incumbent_best_fitness: float,
    candidates: list[dict],
    keep_top_k: int = 4,
    keep_hard_negatives: int = 4,
) -> dict:
    """
    Filter a batch of evaluated GP candidates for `context_sha256` using `jope-learning::retention` rules:
      - Deduplicate by `program_sha256`.
      - Prune failed / non-finite evaluations (`status != 'completed'`).
      - Identify if a new HWM (`fitness < incumbent_best_fitness`) was achieved.
      - Keep at most `keep_top_k` gate-passing Pareto winners and `keep_hard_negatives` informative
        contrastive negatives, pruning all other dominated candidates.
    """
    total_raw = len(candidates)
    dedup = {}
    pruned_failed = 0
    pruned_duplicates = 0

    for c in candidates:
        if c.get("status") != "completed":
            pruned_failed += 1
            continue
        psha = c.get("program_sha256") or hashlib.sha256(c["dsl"].encode()).hexdigest()
        if psha in dedup:
            pruned_duplicates += 1
            continue
        dedup[psha] = c

    valid = sorted(dedup.values(), key=lambda x: (float(x["fitness"]), x["program_sha256"]))
    passing = [c for c in valid if c.get("cv_gate_passed")]
    failing = [c for c in valid if not c.get("cv_gate_passed")]

    retained_top = passing[:keep_top_k]
    # Hard negatives: closest to passing the gate (lowest fitness among failing or bottom of passing)
    retained_neg = failing[:keep_hard_negatives]
    pruned_dominated = len(valid) - len(retained_top) - len(retained_neg)

    new_hwm_record = None
    if retained_top and float(retained_top[0]["fitness"]) < incumbent_best_fitness - 1e-6:
        best = retained_top[0]
        new_hwm_record = {
            "schema": "jope.hwm/v1",
            "context_sha256": context_sha256,
            "program_sha256": best["program_sha256"],
            "dsl": best["dsl"],
            "operator": best.get("operator", "edit_zero"),
            "parents": best.get("parents", []),
            "previous_best_fitness": float(incumbent_best_fitness),
            "new_best_fitness": float(best["fitness"]),
            "fitness_improvement": float(incumbent_best_fitness - float(best["fitness"])),
            "cv_geomean_rmse_ratio": float(best["cv_geomean_rmse_ratio"]),
            "cv_gate_passed": True,
            "test_rmse_ratio": best.get("test_rmse_ratio"),
            "cv_test_agreed": bool(best.get("cv_test_agreed", False)),
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    score_records = []
    for role_tag, subset in (("top_k_winner", retained_top), ("hard_negative", retained_neg)):
        for c in subset:
            score_records.append({
                "schema": "jope.score/v1",
                "retention_tier": role_tag,
                "context_sha256": context_sha256,
                "program_sha256": c["program_sha256"],
                "dsl": c["dsl"],
                "operator": c.get("operator", "edit_zero"),
                "parents": c.get("parents", []),
                "fitness": float(c["fitness"]),
                "logratio": float(c["logratio"]),
                "cv_geomean_rmse_ratio": float(c["cv_geomean_rmse_ratio"]),
                "cv_gate_passed": bool(c["cv_gate_passed"]),
                "test_rmse_ratio": c.get("test_rmse_ratio"),
                "cv_test_agreed": bool(c.get("cv_test_agreed", False)),
                "wall_s": float(c.get("wall_s", 0.0)),
            })

    return {
        "context_sha256": context_sha256,
        "new_hwm": new_hwm_record,
        "retained_scores": score_records,
        "stats": {
            "total_evaluated": total_raw,
            "pruned_failed": pruned_failed,
            "pruned_duplicates": pruned_duplicates,
            "pruned_dominated": max(0, pruned_dominated),
            "retained_top_k": len(retained_top),
            "retained_hard_negatives": len(retained_neg),
            "total_retained": len(score_records),
        },
    }


def persist_round_artifacts_to_s3(
    round_idx: int,
    retention_batches: list[dict],
    round_metrics: dict,
    emitted_rust_files: list[str],
    upload_to_s3: bool = True,
) -> dict:
    """
    Write canonical `jope.hwm/v1`, `jope.score/v1`, `gp_distilled/v1`, and `retention_report.json`
    for `round_idx` to local custody and sync to `s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/`.
    """
    round_dir = LOCAL_S3_MIRROR / f"round-{round_idx:02d}"
    hwm_dir = round_dir / "hwm"
    scores_dir = round_dir / "scores"
    distilled_dir = round_dir / "gp_distilled"
    hwm_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    distilled_dir.mkdir(parents=True, exist_ok=True)

    agg_stats = {
        "total_evaluated": 0,
        "pruned_failed": 0,
        "pruned_duplicates": 0,
        "pruned_dominated": 0,
        "retained_top_k": 0,
        "retained_hard_negatives": 0,
        "total_retained": 0,
        "new_hwm_count": 0,
    }

    all_scores = []
    new_hwms = []
    for b in retention_batches:
        cid = b["context_sha256"]
        st = b["stats"]
        for k in ("total_evaluated", "pruned_failed", "pruned_duplicates", "pruned_dominated", "retained_top_k", "retained_hard_negatives", "total_retained"):
            agg_stats[k] += st[k]
        if b["new_hwm"] is not None:
            agg_stats["new_hwm_count"] += 1
            new_hwms.append(b["new_hwm"])
            (hwm_dir / f"{cid}.json").write_text(json.dumps(b["new_hwm"], indent=2))
        all_scores.extend(b["retained_scores"])

    # Write compact JSONL shard of retained scores (`jope.score/v1`)
    scores_jsonl_path = scores_dir / f"retained_scores_round_{round_idx:02d}.jsonl"
    with scores_jsonl_path.open("w") as f:
        for srec in all_scores:
            f.write(json.dumps(srec, sort_keys=True) + "\n")

    # Write distilled self-play lesson bundle (`gp_distilled/v1`)
    distilled_manifest = {
        "schema": "gp_distilled/v1",
        "round": round_idx,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "new_hwm_winners": new_hwms,
        "emitted_rust_models": emitted_rust_files,
        "round_metrics": round_metrics,
        "retention_summary": agg_stats,
    }
    distilled_path = distilled_dir / f"distilled_round_{round_idx:02d}.json"
    distilled_path.write_text(json.dumps(distilled_manifest, indent=2))

    pruned_total = agg_stats["pruned_failed"] + agg_stats["pruned_duplicates"] + agg_stats["pruned_dominated"]
    agg_stats["pruned_garbage_total"] = pruned_total
    agg_stats["storage_reduction_pct"] = round(
        100.0 * pruned_total / max(1, agg_stats["total_evaluated"]), 2
    )

    s3_target_uri = f"{S3_BUCKET_PREFIX}/round-{round_idx:02d}/"
    receipt = {
        "schema": "jope.round-persistence-manifest/v2",
        "round": round_idx,
        "local_round_dir": str(round_dir),
        "scores_jsonl_path": str(scores_jsonl_path),
        "scores_jsonl_sha256": hashlib.sha256(scores_jsonl_path.read_bytes()).hexdigest(),
        "distilled_path": str(distilled_path),
        "distilled_sha256": hashlib.sha256(distilled_path.read_bytes()).hexdigest(),
        "retention_summary": agg_stats,
        "s3_target_uri": s3_target_uri,
    }
    receipt_path = round_dir / "persistence_receipt.json"
    # Write immutable round manifest ONCE prior to S3 sync; never mutate in-place after upload
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")

    s3_status = {"uploaded": False, "s3_uri": s3_target_uri}
    if upload_to_s3:
        env = os.environ.copy()
        env["AWS_PROFILE"] = AWS_PROFILE
        try:
            res = subprocess.run(
                [
                    "aws", "s3", "sync",
                    str(round_dir),
                    s3_target_uri,
                    "--only-show-errors",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            s3_status["uploaded"] = (res.returncode == 0)
            s3_status["returncode"] = res.returncode
            if res.returncode != 0:
                s3_status["stderr"] = res.stderr[:300]
        except Exception as e:
            s3_status["error"] = str(e)

        EXTERNAL_SYNC_RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
        ext_sync_path = EXTERNAL_SYNC_RECEIPTS_DIR / f"round-{round_idx:02d}-s3-sync-receipt.json"
        ext_sync_payload = {
            "round": round_idx,
            "round_manifest_path": str(receipt_path),
            "round_manifest_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "s3_sync": s3_status,
            "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        ext_sync_path.write_text(json.dumps(ext_sync_payload, indent=2) + "\n")

    return {**receipt, "s3_sync": s3_status}


def sync_directory_to_s3(local_dir: Path, s3_uri: str) -> dict:
    """Sync any local directory (`RESULTS_DIR`, `EMITTED_RUST_DIR`) to `s3_uri` via `AWS_PROFILE=veox-jepson-prod`."""
    env = os.environ.copy()
    env["AWS_PROFILE"] = AWS_PROFILE
    status = {"uploaded": False, "s3_uri": s3_uri}
    try:
        res = subprocess.run(
            ["aws", "s3", "sync", str(local_dir), s3_uri, "--only-show-errors"],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        status["uploaded"] = res.returncode == 0
        status["returncode"] = res.returncode
        if res.returncode != 0:
            status["stderr"] = res.stderr[:300]
    except Exception as e:
        status["error"] = str(e)
    return status

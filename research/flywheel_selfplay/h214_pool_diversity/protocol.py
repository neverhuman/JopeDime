"""H214 capability and resource admission; no study or data reads at import."""

from __future__ import annotations

import math

from research.flywheel_selfplay.h214_pool_diversity import analysis as stat

HOST = "xbabe1"
MAX_WALL = 7200


def require_smoke(receipt: dict, source_commit: str) -> None:
    if (
        receipt.get("status") != "PASS"
        or receipt.get("source_commit") != source_commit
        or receipt.get("widths") != [16, 32, 64]
        or receipt.get("pool_record_count") != 3
        or receipt.get("rust_evaluation_completed") is not True
        or receipt.get("analysis_row_written") is not True
    ):
        raise ValueError(
            "merged H212 end-to-end smoke PASS at exact launch head required"
        )


def propose_arm(proposer, arguments: dict, arm: str):
    if arm not in stat.ARMS:
        raise ValueError("unregistered H214 arm; TEMP16 is NOT_RUN by owner ruling")
    options = dict(arguments, n_proposals=64 if arm == "FPS16" else 16)
    result = proposer(**options)
    if len(result) != options["n_proposals"]:
        raise ValueError("native proposal output omitted registered generation width")
    return result


def resource_guard(observation: dict, now: float, high_since: float | None):
    numeric = [
        observation[key]
        for key in (
            "disk_used_fraction",
            "disk_free_bytes",
            "mem_available_bytes",
            "rss_bytes",
            "load5",
            "load1",
            "cores",
            "cpu_pct",
        )
    ]
    if (
        not all(math.isfinite(value) and value >= 0 for value in numeric)
        or observation["cores"] <= 0
    ):
        raise RuntimeError("invalid resource observation")
    if observation["host"] != HOST or observation["nice"] != 19:
        raise RuntimeError("H214 requires xbabe1 CPU at nice19")
    if (
        observation["disk_used_fraction"] > 0.938
        or observation["disk_free_bytes"] < 40_000_000_000
        or observation["mem_available_bytes"] < 16 * 2**30
        or observation["rss_bytes"] >= 6_000_000_000
    ):
        raise RuntimeError("instantaneous disk/free/memory/RSS guard")
    if observation["load5"] > 0.85 * observation["cores"]:
        raise RuntimeError("load5 guard")
    high = (
        observation["load1"] > 0.85 * observation["cores"]
        or observation["cpu_pct"] > 85.0
    )
    high_since = (now if high_since is None else high_since) if high else None
    if high_since is not None and now - high_since >= 300:
        raise RuntimeError("CPU/load1 above85% sustained300s")
    return high_since

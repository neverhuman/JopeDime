"""v35 dev dataset pools + validation-integrity guard.

Adopted BYTE-IDENTICAL from the fable benchmark harness
(the configured fable harness pools.json, sha256 bbba666d…5b20fd) so v35
dev numbers are directly comparable to the fable's own dev artifacts
(e.g. experiments/v2_devA.jsonl). Pools were sampled once (seed 42) from the
2090 train-split hashes; validation hashes excluded by construction.

Protocol (locked 2026-07-10):
  DEV-A (300): unlimited iteration.
  DEV-B (300): <=1 honest confirm per milestone candidate.
  DEV-C (500): untouched until the single M4.5 preflight.
  validation-232: ship gates only (<=2 runs in all of v35), never via this module.

Every loader call re-asserts validation disjointness; `assert_dev_hashes` is the
hard gate any dev-scoring path must call (ported from the fable harness).
"""
from __future__ import annotations

import json
from pathlib import Path

from .prime_config import load_config

_PRIME_CONFIG = load_config()
POOLS_PATH = Path(__file__).resolve().parents[2] / "data" / "dev_pools_v35.json"
FABLE_POOLS_PATH = _PRIME_CONFIG.fable_root / "harness" / "pools.json"
VALIDATION_JSON = _PRIME_CONFIG.validation_dataset_manifest

_POOLS: dict[str, list[str]] | None = None
_VAL: set[str] | None = None


def validation_hashes() -> set[str]:
    global _VAL
    if _VAL is None:
        d = json.loads(VALIDATION_JSON.read_text())
        _VAL = {x["dataset_hash"] for x in d["datasets"]}
    return _VAL


def assert_dev_hashes(hashes) -> None:
    """Hard integrity gate: no validation hash may ever be scored during development."""
    bad = set(hashes) & validation_hashes()
    if bad:
        raise RuntimeError(
            f"INTEGRITY VIOLATION: validation hashes in dev scoring: {sorted(bad)[:5]}"
        )


def get_pools() -> dict[str, list[str]]:
    global _POOLS
    if _POOLS is None:
        pools = json.loads(POOLS_PATH.read_text())
        if FABLE_POOLS_PATH.exists():
            fable = json.loads(FABLE_POOLS_PATH.read_text())
            if fable != pools:
                raise RuntimeError(
                    "dev_pools_v35.json has drifted from the fable pools.json"
                )
        val = validation_hashes()
        for k, v in pools.items():
            if set(v) & val:
                raise RuntimeError(f"validation leak in pool {k}")
        combined = [h for vs in pools.values() for h in vs]
        if len(set(combined)) != len(combined):
            raise RuntimeError("duplicate hashes across dev pools")
        _POOLS = pools
    return _POOLS


if __name__ == "__main__":
    p = get_pools()
    print({k: len(v) for k, v in p.items()})

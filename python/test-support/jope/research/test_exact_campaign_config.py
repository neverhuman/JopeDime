from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from test_exact_common import (
    EXPECTED_METRIC,
    atomic_json,
    compose_seed,
    interleave_tasks,
    rank_common_winners,
    selfplay_authority,
    sha256_file,
    sha256_text,
    structural_key,
)

GENEBANK = Path("/home/ubuntu/jope-flywheel/genebank/genebank.db")
ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
CATALOG_POINTER = ROOT / "catalog" / "current.json"
PROPOSALS = ROOT / "proposals" / "dl-proposals.jsonl"
CHECKPOINT_HASH = "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592"
GOLDEN42 = Path("/home/ubuntu/golden42/golden42_foundry/data/panel.json")
SHARD_SCHEMA = "JopeFlywheelCorpusEvalShard.v1"
HEAD_PREFIXES = {
    "ridge",
    "elasticnet",
    "gpr",
    "histgbm2",
    "ensemble_ridge_gbm",
    "tab_boost",
    "extra_trees",
    "gam_boost",
    "pls",
    "logit_ridge",
}


def database(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con

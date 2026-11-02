#!/usr/bin/env python3
"""Generate 64 greedy+nucleus policy proposals for every frozen dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path

import torch

if __package__:
    from .test_exact_common import sha256_file, sha256_text
    from .test_exact_dl_generate_main import main
    from .test_exact_dl_generate_support import (
        BINARY_HEADS,
        CLASSIFIER_ONLY_HEADS,
        DEFAULT_CATALOG,
        DEFAULT_CHECKPOINT,
        DEFAULT_OUTPUT,
        MULTICLASS_HEADS,
        PROPOSAL_SELECTION_POLICY_SCHEMA,
        STRIDE_FLOATS,
        decode_batch,
        default_proposal_selection_policy,
        deterministic_normal,
        filter_exact_dataset_manifest,
        head_valid_for_task,
        load_proposal_selection_policy,
        rank_scored_proposals,
        score_context_candidates,
    )
else:
    from test_exact_common import sha256_file, sha256_text
    from test_exact_dl_generate_main import main
    from test_exact_dl_generate_support import (
        BINARY_HEADS,
        CLASSIFIER_ONLY_HEADS,
        DEFAULT_CATALOG,
        DEFAULT_CHECKPOINT,
        DEFAULT_OUTPUT,
        MULTICLASS_HEADS,
        PROPOSAL_SELECTION_POLICY_SCHEMA,
        STRIDE_FLOATS,
        decode_batch,
        default_proposal_selection_policy,
        deterministic_normal,
        filter_exact_dataset_manifest,
        head_valid_for_task,
        load_proposal_selection_policy,
        rank_scored_proposals,
        score_context_candidates,
    )

__all__ = [
    "DEFAULT_CATALOG",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_OUTPUT",
    "STRIDE_FLOATS",
    "PROPOSAL_SELECTION_POLICY_SCHEMA",
    "BINARY_HEADS",
    "CLASSIFIER_ONLY_HEADS",
    "MULTICLASS_HEADS",
    "default_proposal_selection_policy",
    "load_proposal_selection_policy",
    "head_valid_for_task",
    "rank_scored_proposals",
    "deterministic_normal",
    "filter_exact_dataset_manifest",
    "decode_batch",
    "score_context_candidates",
    "sha256_file",
    "sha256_text",
    "main",
    "argparse",
    "hashlib",
    "json",
    "math",
    "os",
    "struct",
    "time",
    "Path",
    "torch",
]

if __name__ == "__main__":
    main()

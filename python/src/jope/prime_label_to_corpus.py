"""Adapter: realized-label / aggregate rows -> policy_v3 ``TrainingRow`` corpus.

This is the data seam of the self-improvement loop's brain. The loop realizes
champions on held-out data and writes labels; this module turns those labels into
the exact corpus that ``jope.policy.train_v3.train_policy_v3_from_jsonl`` consumes,
so the REAL transformer proposer can train on the loop's realized outcomes.

Supported input schemas (dispatched on the ``schema`` field, with fallbacks):

* ``jope-prime-realization-aggregate-v1`` (the primary target) — one row per
  (dataset_hash, semantic_plan_hash) group, carrying ``canonical_genome``,
  ``reward`` (== ``mean_realized``), ``p_bad``, ``reward_std``, ``n_evidence``.
* ``jope-prime-model-realization-v2`` — a single realized champion; ``realized_score``
  is the reward, plus ``oof_r2`` and ``failure``.
* ``LeagueLabel.v1`` (see ``rust/jope-league/src/contracts.rs``) — the league's
  wire label; ``robust_score`` is the reward, ``failure`` marks a negative.

The two load-bearing mappings:

* ``canonical_genome`` -> genome tokens. IMPORTANT: the realization rows are tagged
  ``engine="prime"`` meaning the *Prime fork*, NOT the policy grammar engine. Their
  genomes are ordinary ``featmath`` (and occasionally ``newmath``) pipelines. So we
  resolve the true *grammar* engine by probing which grammar can encode the genome,
  because ``train_v3`` tokenizes with ``row.engine`` as a grammar key.
* ``reward``/``mean_realized`` -> the ranking target. ``train_v3`` recomputes the
  within-dataset listwise labels (``oof_delta_to_incumbent``, ``rank_pct``) from
  ``_score_label`` = ``metadata.score_robust`` | ``metadata.oof_r2_raw`` | ``row.score``,
  grouped by ``metadata.reward_pair_identity`` when present, otherwise the base dataset.
  League labels derive that identity from dataset/task/fold plan/evaluator authority;
  ``dataset_id`` remains ``dataset_hash`` for conditioning, splits, and routing.

Dataset conditioning (the model's ``embedding`` / ``target_embedding``): produced the
same way ``self_play``/``gp_factory`` build it — ``propose.embedding_from_vs`` /
``target_summary_from_vs`` over the per-dataset ``.vs`` signature files named in a
split manifest. When those signatures are unavailable the adapter falls back to a
deterministic per-dataset placeholder of the requested dimensionality and flags the
row (``metadata.embedding_synthetic``); a production checkpoint requires the real
signatures (documented on ``PlaceholderEmbedder``).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .schema import TrainingRow
from .split_integrity import (
    DEVELOPMENT_SPLIT,
    TEMPORAL_AUDIT_SPLIT,
    assignment_for_dataset,
    is_training_authority_label,
    validate_split_manifest,
)


from .prime_label_to_corpus_identity import (
    AGGREGATE_SCHEMA,
    DEFAULT_PROVENANCE,
    DUAL_EMBEDDING_DIM,
    GRAMMAR_ENGINE_PRIORITY,
    LEAGUE_LABEL_SCHEMA,
    REALIZATION_SCHEMA,
    TARGET_SUMMARY_DIM,
    GenomeEngineResolver,
    _finite,
    _league_label_evidence,
    _reward_pair_identity,
    _sha256_file,
)
from .prime_label_to_corpus_embeddings import PlaceholderEmbedder, SignatureEmbedder
from .prime_label_to_corpus_rows import (
    build_corpus,
    label_to_training_row,
    write_corpus,
    _parse_row,
)

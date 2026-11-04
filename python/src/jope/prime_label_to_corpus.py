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

from .prime_label_to_corpus_embeddings import (
    PlaceholderEmbedder as PlaceholderEmbedder,
)
from .prime_label_to_corpus_embeddings import (
    SignatureEmbedder as SignatureEmbedder,
)
from .prime_label_to_corpus_identity import (
    AGGREGATE_SCHEMA as AGGREGATE_SCHEMA,
)
from .prime_label_to_corpus_identity import (
    DEFAULT_PROVENANCE as DEFAULT_PROVENANCE,
)
from .prime_label_to_corpus_identity import (
    DUAL_EMBEDDING_DIM as DUAL_EMBEDDING_DIM,
)
from .prime_label_to_corpus_identity import (
    GRAMMAR_ENGINE_PRIORITY as GRAMMAR_ENGINE_PRIORITY,
)
from .prime_label_to_corpus_identity import (
    LEAGUE_LABEL_SCHEMA as LEAGUE_LABEL_SCHEMA,
)
from .prime_label_to_corpus_identity import (
    REALIZATION_SCHEMA as REALIZATION_SCHEMA,
)
from .prime_label_to_corpus_identity import (
    TARGET_SUMMARY_DIM as TARGET_SUMMARY_DIM,
)
from .prime_label_to_corpus_identity import (
    GenomeEngineResolver as GenomeEngineResolver,
)
from .prime_label_to_corpus_identity import (
    _finite as _finite,
)
from .prime_label_to_corpus_identity import (
    _league_label_evidence as _league_label_evidence,
)
from .prime_label_to_corpus_identity import (
    _reward_pair_identity as _reward_pair_identity,
)
from .prime_label_to_corpus_identity import (
    _sha256_file as _sha256_file,
)
from .prime_label_to_corpus_rows import (
    _parse_row as _parse_row,
)
from .prime_label_to_corpus_rows import (
    build_corpus as build_corpus,
)
from .prime_label_to_corpus_rows import (
    label_to_training_row as label_to_training_row,
)
from .prime_label_to_corpus_rows import (
    write_corpus as write_corpus,
)
from .schema import TrainingRow as TrainingRow
from .split_integrity import (
    DEVELOPMENT_SPLIT as DEVELOPMENT_SPLIT,
)
from .split_integrity import (
    TEMPORAL_AUDIT_SPLIT as TEMPORAL_AUDIT_SPLIT,
)
from .split_integrity import (
    assignment_for_dataset as assignment_for_dataset,
)
from .split_integrity import (
    is_training_authority_label as is_training_authority_label,
)
from .split_integrity import (
    validate_split_manifest as validate_split_manifest,
)

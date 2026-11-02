from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ..grammars.spec import GrammarSpec
from .tokens import (
    GENERATION31_SNAPSHOT_SHA256, GENERATION31_TOKEN_SHA256, GENERATION31_VOCAB_SIZE,
    ActionVocab, append_missing_task_tokens, build_action_vocab,
    extend_action_vocab_append_only, extend_engine_ops_append_only,
    reconcile_head_tasks_from_grammar,
)

def generation31_vocab_snapshot_path() -> Path:
    """Return the Prime-owned frozen grammar snapshot for generation 31."""
    if configured := os.environ.get("JOPE_GENERATION31_VOCAB"):
        return Path(configured).expanduser().resolve(strict=False)
    from ..prime_config import load_config

    return load_config().generation31_vocab_snapshot


def build_generation31_action_vocab(path: str | Path | None = None) -> ActionVocab:
    """Rehydrate and verify generation 31's immutable 1,103-token vocabulary."""
    from ..grammars.spec import GrammarSpec

    snapshot_path = Path(path) if path is not None else generation31_vocab_snapshot_path()
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot_hash = hashlib.sha256(snapshot_bytes).hexdigest()
    if snapshot_hash != GENERATION31_SNAPSHOT_SHA256:
        raise ValueError(
            "generation-31 grammar snapshot drift: "
            f"sha256={snapshot_hash}"
        )
    snapshot = json.loads(snapshot_bytes)
    if snapshot.get("schema_version") != 4 or snapshot.get("syntax") != "active_multi_engine_v3":
        raise ValueError("generation-31 vocabulary snapshot has an unsupported schema")
    engines = snapshot.get("engines")
    if not isinstance(engines, dict) or tuple(engines) != ("featmath", "newmath"):
        raise ValueError("generation-31 vocabulary snapshot must contain featmath then newmath")
    specs = [GrammarSpec.from_json(engines[name]) for name in ("featmath", "newmath")]
    vocab = build_action_vocab(specs)
    payload = json.dumps(
        vocab.tokens, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    token_hash = hashlib.sha256(payload).hexdigest()
    if vocab.size != GENERATION31_VOCAB_SIZE or token_hash != GENERATION31_TOKEN_SHA256:
        raise ValueError(
            "generation-31 vocabulary snapshot drift: "
            f"size={vocab.size}, token_sha256={token_hash}"
        )
    if (
        vocab.tokens[59] != "VAL=featmath:bin_onehot:bins=15"
        or vocab.tokens[-2:] != ["TASK=regression", "TASK=binary"]
    ):
        raise ValueError("generation-31 vocabulary boundary fixture drift")
    return vocab


def build_prime_action_vocab() -> ActionVocab:
    """Append Prime after generation 31's exact checkpoint vocabulary."""
    from ..grammars import grammar_for, prime_grammar
    from ..grammars.prime import registry_snapshot

    vocab = extend_action_vocab_append_only(
        build_generation31_action_vocab(), prime_grammar()
    )
    compatibility_names = [
        str(entry["name"])
        for entry in registry_snapshot()
        if entry.get("role") == "head"
        and "compatibility" in entry.get("tags", [])
    ]
    compatibility_names.extend(
        ("logit_ridge", "tab_boost", "gam_boost", "extra_trees")
    )
    # Four-task program: the multiclass decode heads. These are genuinely NEW
    # tokens (the snapshot has no multiclass head at all), appended here — after
    # every pre-existing op and before the TASK tokens — so every id below the
    # previous vocab size is unchanged and existing checkpoints keep loading via
    # `_validate_action_vocab_compatibility`'s append-only prefix rule. A vocab
    # token can never be renamed once appended, so these names are pinned to what
    # feat-math main actually executes (`corpus_eval` + `fit_v3_head`).
    compatibility_names.extend(
        (
            "multiclass_ridge",
            "multiclass_ovr_logistic",
            "multiclass_diagonal_lda",
            "multiclass_nearest_centroid",
        )
    )
    vocab = extend_engine_ops_append_only(
        vocab,
        grammar_for("featmath"),
        list(dict.fromkeys(compatibility_names)),
    )
    # Four-task program: grant TIMESERIES capability to the featmath heads the
    # snapshot froze at ("regression", "binary"). Appends no tokens — see
    # reconcile_head_tasks_from_grammar. `logistic`/`logit_ridge` are excluded
    # because corpus_eval rejects classifier heads for timeseries, so granting them
    # would emit genomes that decode and then fail evaluation. Multiclass is NOT
    # granted here: only the four multiclass_* engine heads implement
    # `multiclass_probabilities`, so every other head fails closed with "bound model
    # does not carry a multiclass head".
    vocab = reconcile_head_tasks_from_grammar(
        vocab,
        grammar_for("featmath"),
        grant=("timeseries",),
        exclude=("logistic", "logit_ridge"),
    )
    # TASK=timeseries / TASK=multiclass are appended AFTER every engine/op/compat
    # token so the gen-31 prefix (and every later id) is stable.
    return append_missing_task_tokens(vocab)

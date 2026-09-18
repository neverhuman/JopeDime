"""Masked decoding for policy_v3: batch sampling, greedy one-shot, and propose_v3."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from ..engines import EngineRouter
from ..pipeline_text import format_pipeline_actions
from ..schema import normalize_embedding
from ..selector_calibration import load_selector_calibration, score_candidate_with_calibration
from ..selector_reranker import load_selector_reranker, score_candidate_with_reranker
from .automaton import DecoderState
from .model_v3 import JopePolicyModelV3, pad_sequences
from .tokens import BOS, decode_tokens

# Transforms that blow up dimensionality/instability when repeated (risk guard).
_EXPLOSIVE = {
    "featmath": {"cheb", "poly_interact", "pair_prod", "bin_onehot", "rff_cos", "rff_sin"},
    "newmath": {"cheb", "poly_interact", "rff_cos", "rff_sin", "phase_interference", "iterate", "logistic_map"},
}


def _require_legal_actions(legal: Sequence[int], state: DecoderState) -> None:
    """Fail loudly, and by name, when a decode step has no legal action.

    Both sampling loops build ``mask = full(-inf); mask[legal] = 0``. With an
    empty ``legal`` every logit stays ``-inf``, so softmax yields NaN and
    ``multinomial`` raises something unrelated to the real cause; with
    ``temperature <= 0`` argmax silently returns index 0 (``PAD``) and
    ``state.advance(0)`` raises a bare ``IllegalToken``. Neither names the
    engine, the task, or the phase.

    This is the failure mode a task with no generatable head produces, and it is
    exactly how multiclass presented before the ``multiclass_*`` decode heads
    existed: the sampler had an empty legal set at the HEAD phase and nothing
    said so. Any future task added without heads should surface here on the
    first decode rather than as a NaN.
    """
    if len(legal) == 0:
        raise ValueError(
            "no legal action at decode phase "
            f"{state.phase} for engine={state.engine!r} task={state.task!r}; "
            "the grammar declares no generatable op for this (engine, task) pair"
        )


def candidate_risk_reason_v3(engine: str, actions: dict[str, Any]) -> str | None:
    counts: dict[str, int] = {}
    for stage in actions.get("transforms") or []:
        op = str(stage.get("op"))
        counts[op] = counts.get(op, 0) + 1
    for op, count in counts.items():
        if count >= 3 and op in _EXPLOSIVE.get(engine, set()):
            return f"repeated_explosive_transform:{op}"
    return None


@torch.no_grad()
def sample_sequences(
    model: JopePolicyModelV3,
    embedding: torch.Tensor,  # (512,)
    target_summary: torch.Tensor | None,
    *,
    n: int,
    engine: str | None = None,
    task: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 0,
    max_len: int = 120,
    device: str | torch.device = "cpu",
) -> list[list[int]]:
    """Sample n grammar-masked token sequences for one dataset context.

    When ``engine``/``task`` are given they are force-decoded (the model then samples the
    stages). ``task`` masks task-incompatible heads (e.g. featmath gpr for binary).
    """
    model.eval()
    vocab = model.vocab
    emb = embedding.float().reshape(1, -1).to(device)
    tgt = None if target_summary is None else target_summary.float().reshape(1, -1).to(device)
    z = model.context(emb, tgt)  # (1, d)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    states = [DecoderState(vocab) for _ in range(n)]
    seqs: list[list[int]] = [[BOS] for _ in range(n)]
    if engine is not None:
        eid = vocab.engine_ids[engine]
        for state, seq in zip(states, seqs):
            state.advance(eid)
            seq.append(eid)
        # ENGINE is followed by TASK in the fused vocab; force it when requested.
        if vocab.task_ids and task is not None:
            tid = vocab.task_ids.get(task, vocab.task_ids[vocab.tasks[0]])
            for state, seq in zip(states, seqs):
                state.advance(tid)
                seq.append(tid)
    live = [i for i, s in enumerate(states) if not s.is_done()]
    while live:
        token_ids, pad_mask = pad_sequences([seqs[i] for i in live], device=device)
        zk = z.expand(len(live), -1)
        step_logits = model.next_token_logits(zk, token_ids, pad_mask)  # (L, V)
        next_live: list[int] = []
        for row_idx, i in enumerate(live):
            legal = states[i].legal_ids()
            _require_legal_actions(legal, states[i])
            row = step_logits[row_idx].detach().cpu()
            mask = torch.full((vocab.size,), float("-inf"))
            mask[legal] = 0.0
            masked = row + mask
            if temperature <= 0.0:
                tid = int(masked.argmax().item())
            else:
                probs = torch.softmax(masked / max(temperature, 1e-6), dim=-1)
                if 0.0 < top_p < 1.0:
                    sorted_probs, order = probs.sort(descending=True)
                    keep = sorted_probs.cumsum(-1) - sorted_probs < top_p
                    keep[0] = True
                    filtered = torch.zeros_like(probs)
                    filtered[order[keep]] = probs[order[keep]]
                    probs = filtered / filtered.sum()
                tid = int(torch.multinomial(probs, 1, generator=generator).item())
            states[i].advance(tid)
            seqs[i].append(tid)
            if not states[i].is_done() and len(seqs[i]) < max_len:
                next_live.append(i)
        live = next_live
    return seqs


@torch.no_grad()
def sample_sequences_batched(
    model: JopePolicyModelV3,
    embeddings: torch.Tensor,
    target_summaries: torch.Tensor | None,
    *,
    n: int,
    engine: str | None = None,
    task: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 0,
    max_len: int = 120,
    device: str | torch.device = "cpu",
) -> list[list[list[int]]]:
    """Sample ``n`` sequences for every context in one model batch."""

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a nonempty two-dimensional batch")
    if n <= 0:
        raise ValueError("n must be positive")
    if target_summaries is not None and (
        target_summaries.ndim != 2
        or target_summaries.shape[0] != embeddings.shape[0]
    ):
        raise ValueError("target summaries must match the embedding batch")

    model.eval()
    vocab = model.vocab
    emb = embeddings.float().to(device)
    tgt = None if target_summaries is None else target_summaries.float().to(device)
    z = model.context(emb, tgt)
    batch_size = embeddings.shape[0]
    context = z.repeat_interleave(n, dim=0)
    total = batch_size * n
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    states = [DecoderState(vocab) for _ in range(total)]
    seqs: list[list[int]] = [[BOS] for _ in range(total)]
    if engine is not None:
        eid = vocab.engine_ids[engine]
        for state, seq in zip(states, seqs):
            state.advance(eid)
            seq.append(eid)
        if vocab.task_ids and task is not None:
            tid = vocab.task_ids.get(task, vocab.task_ids[vocab.tasks[0]])
            for state, seq in zip(states, seqs):
                state.advance(tid)
                seq.append(tid)

    live = [index for index, state in enumerate(states) if not state.is_done()]
    while live:
        token_ids, pad_mask = pad_sequences(
            [seqs[index] for index in live], device=device
        )
        live_index = torch.tensor(live, dtype=torch.long, device=context.device)
        step_logits = model.next_token_logits(
            context.index_select(0, live_index), token_ids, pad_mask
        )
        next_live: list[int] = []
        for row_index, sequence_index in enumerate(live):
            legal = states[sequence_index].legal_ids()
            _require_legal_actions(legal, states[sequence_index])
            row = step_logits[row_index].detach().cpu()
            mask = torch.full((vocab.size,), float("-inf"))
            mask[legal] = 0.0
            masked = row + mask
            if temperature <= 0.0:
                token = int(masked.argmax().item())
            else:
                probabilities = torch.softmax(
                    masked / max(temperature, 1e-6), dim=-1
                )
                if 0.0 < top_p < 1.0:
                    sorted_probabilities, order = probabilities.sort(descending=True)
                    keep = (
                        sorted_probabilities.cumsum(-1) - sorted_probabilities < top_p
                    )
                    keep[0] = True
                    filtered = torch.zeros_like(probabilities)
                    filtered[order[keep]] = probabilities[order[keep]]
                    probabilities = filtered / filtered.sum()
                token = int(
                    torch.multinomial(
                        probabilities, 1, generator=generator
                    ).item()
                )
            states[sequence_index].advance(token)
            seqs[sequence_index].append(token)
            if (
                not states[sequence_index].is_done()
                and len(seqs[sequence_index]) < max_len
            ):
                next_live.append(sequence_index)
        live = next_live

    return [seqs[index * n : (index + 1) * n] for index in range(batch_size)]


@torch.no_grad()
def greedy_sequence(
    model: JopePolicyModelV3,
    embedding: torch.Tensor,
    target_summary: torch.Tensor | None,
    *,
    engine: str | None = None,
    task: str | None = None,
    device: str | torch.device = "cpu",
) -> list[int]:
    return sample_sequences(
        model, embedding, target_summary, n=1, engine=engine, task=task, temperature=0.0, device=device
    )[0]


def rank_candidates(valid, n, stability_threshold, selection_key):
    """Final candidate ordering. BUG-A fix (all 7 external reviews): the deployed
    rank-0 slot is SACRED — always the global selection_key argmax. The engine
    diversity floor previously got PREPENDED (engines alphabetical → featmath
    preempted a better newmath candidate at slot 0, structurally defeating the
    routing objective and poisoning self-play credit assignment). Diversity now
    only fills slots >= 1."""
    ranked_order = sorted(range(len(valid)), key=selection_key, reverse=True)
    top = ranked_order[0]
    floor: list[int] = []
    if n >= len({row["engine"] for row in valid}):
        for engine in sorted({row["engine"] for row in valid}):
            if valid[top]["engine"] == engine:
                continue  # already represented by the global best at slot 0
            viable = [
                i
                for i in ranked_order
                if valid[i]["engine"] == engine
                and i != top
                and i not in floor
                and float(valid[i].get("p_bad") or 0.0) <= stability_threshold
                and not valid[i].get("candidate_risk_reason")
            ]
            if viable:
                floor.append(viable[0])
    return [top] + floor + [i for i in ranked_order if i != top and i not in floor]


def propose_v3(
    embedding: Sequence[float] | np.ndarray,
    *,
    target_summary: Sequence[float] | np.ndarray | None = None,
    model: JopePolicyModelV3,
    router: EngineRouter,
    n: int = 1,
    seed: int = 0,
    temperature: float = 0.7,
    top_p: float = 0.9,
    engines: tuple[str, ...] = ("featmath", "newmath"),
    task: str | None = None,
    samples_per_engine: int | None = None,
    historical: list[tuple[str, str]] | None = None,
    risk_aversion: float = 0.5,
    stability_threshold: float = 0.5,
    candidate_audit: list[dict[str, Any]] | None = None,
    selector_calibration: str | dict[str, Any] | None = None,
    selector_reranker: str | dict[str, Any] | None = None,
    return_all_candidates: bool = False,
    gene_feed: Any = None,
    gene_feed_dataset: str | None = None,
    device: str | torch.device = "cpu",
) -> list[dict[str, Any]]:
    """Generate, validate, reward-rank, and select (engine, genome) proposals.

    Every emitted genome is masked-decoded (valid by construction); the bridge is the
    canonicalization authority. Selection score = reward_mean - risk_aversion * reward_std.
    """
    samples_per_engine = samples_per_engine or max(n * 8, 24)
    calibration = load_selector_calibration(selector_calibration)
    reranker = load_selector_reranker(selector_reranker)
    emb = torch.tensor(normalize_embedding(list(np.asarray(embedding).reshape(-1).astype(float))))
    tgt = None
    if target_summary is not None:
        values = list(np.asarray(target_summary).reshape(-1).astype(float))[:512]
        values += [0.0] * (512 - len(values))
        tgt = torch.tensor(values)

    candidates: list[dict[str, Any]] = []
    for engine in engines:
        # For a fused model, force the dataset's task so only task-appropriate genomes
        # (and heads) are proposed; task=None lets the model sample the task freely.
        greedy = greedy_sequence(model, emb, tgt, engine=engine, task=task, device=device)
        sampled = sample_sequences(
            model,
            emb,
            tgt,
            n=samples_per_engine,
            engine=engine,
            task=task,
            temperature=temperature,
            top_p=top_p,
            seed=seed + int.from_bytes(__import__("hashlib").sha256(engine.encode()).digest()[:2], "big"),
            device=device,
        )
        for decoder, seq in [("greedy", greedy)] + [("sample", s) for s in sampled]:
            decoded_engine, actions = decode_tokens(seq, model.vocab)
            candidates.append(
                {
                    "engine": decoded_engine,
                    "genome": format_pipeline_actions(actions),
                    "actions": actions,
                    "tokens": seq,
                    "decoder": decoder,
                }
            )

    validations = router.validate_many([(c["engine"], c["genome"]) for c in candidates])
    seen: set[tuple[str, str]] = set()
    historical_keys = {(engine, genome) for engine, genome in (historical or [])}
    valid: list[dict[str, Any]] = []
    disagreements = 0
    for cand, meta in zip(candidates, validations):
        row = {**cand, **{k: meta.get(k) for k in ("valid", "canonical_genome", "head_type", "errors", "family_bitmap")}}
        row.pop("actions", None)
        if not meta.get("valid"):
            disagreements += 1
            row["automaton_bridge_disagreement"] = True
            if candidate_audit is not None:
                candidate_audit.append({k: row.get(k) for k in ("engine", "genome", "decoder", "valid", "errors")})
            continue
        canonical = str(meta.get("canonical_genome"))
        key = (cand["engine"], canonical)
        risk = candidate_risk_reason_v3(cand["engine"], cand["actions"])
        row["candidate_risk_reason"] = risk
        row["candidate_diversity_key"] = _candidate_diversity_key_v3(
            cand["actions"], row.get("head_type"), row.get("family_bitmap")
        )
        row["novel_vs_historical"] = key not in historical_keys
        if key in seen:
            row["candidate_filter"] = "duplicate_canonical"
            if candidate_audit is not None:
                candidate_audit.append({k: row.get(k) for k in ("engine", "canonical_genome", "decoder", "candidate_filter")})
            continue
        seen.add(key)
        valid.append(row)

    if not valid:
        fallback_engine = engines[0]
        meta = router.validate_many([(fallback_engine, router.fallback(fallback_engine))])[0]
        fallback = [
            {
                **meta,
                "engine": fallback_engine,
                "fallback": True,
                "reward": None,
                "automaton_bridge_disagreements": disagreements,
            }
        ]
        if return_all_candidates:
            return fallback, []
        return fallback

    with torch.no_grad():
        mean, std = model.score_token_sequences(emb, tgt, [row["tokens"] for row in valid], device=device)
        token_ids, pad = pad_sequences([row["tokens"] for row in valid], device=device)
        # dataset-conditioned instability (v3.1): the same z the reward heads see.
        # v3.4 un-shared the context encoder, so "the same z the reward heads see" is now
        # reward_context(), NOT context(). Feeding the sequence-path z here would score
        # instability through a representation the stability head was never trained on.
        # reward_context falls back to the single encoder on a v3.3 checkpoint.
        z_ctx = model.reward_context(
            emb.float().reshape(1, -1).to(device),
            None if tgt is None else tgt.float().reshape(1, -1).to(device),
        )
        p_bad = torch.sigmoid(
            model.instability_logit(token_ids, pad, z_ctx.expand(token_ids.shape[0], -1))
        ).cpu()

    # RF1: raw BT rewards are shift/scale-free AND engine-identity leaks into them
    # (featmath medians sat ~5 points above newmath while newmath seeds actually
    # outscored featmath seeds). Rank candidates WITHIN their engine and mix by
    # percentile so engine choice reflects candidate quality, not token identity.
    scores = [float(mean[i]) - risk_aversion * float(std[i]) for i in range(len(valid))]
    percentile: dict[int, float] = {}
    for engine in {row["engine"] for row in valid}:
        idxs = [i for i, row in enumerate(valid) if row["engine"] == engine]
        ranked = sorted(idxs, key=lambda i: scores[i])
        for rank, i in enumerate(ranked):
            percentile[i] = (rank + 1) / len(ranked)

    for i, row in enumerate(valid):
        row.update(
            {
                "reward": float(mean[i]),
                "reward_mean": float(mean[i]),
                "reward_std": float(std[i]),
                "reward_lcb": float(mean[i]) - risk_aversion * float(std[i]),
                "reward_percentile_in_engine": percentile[i],
                "p_bad": float(p_bad[i]),
                "predicted_oof_delta": float(mean[i]),
                "uncertainty": float(std[i]),
            }
        )
        if calibration is not None:
            row.update(score_candidate_with_calibration(row, calibration))
        if reranker is not None:
            rerank_score = score_candidate_with_reranker(row, reranker)
            if rerank_score is not None:
                row["selector_reranker_score"] = rerank_score
        if gene_feed is not None:
            row["gene_id"] = gene_feed.proposed(
                engine=row["engine"],
                canonical=str(row.get("canonical_genome")),
                dataset_hash=str(gene_feed_dataset or ""),
                head=row.get("head_type"),
                families=[k for k, v in (row.get("family_bitmap") or {}).items() if v],
                decoder=row.get("decoder"),
                mu=float(mean[i]),
                sigma=float(std[i]),
                p_bad=float(p_bad[i]),
                percentile=percentile[i],
                calibrated_win_rate=row.get("calibrated_win_rate"),
                calibrated_expected_delta=row.get("calibrated_expected_delta"),
                novel=row.get("novel_vs_historical"),
                risk=row.get("candidate_risk_reason"),
            )

    def selection_key(i: int) -> float:
        row = valid[i]
        calibrated_win = row.get("calibrated_win_rate")
        calibrated_delta = row.get("calibrated_expected_delta")
        calibrated_signal = 0.0
        if isinstance(calibrated_win, (int, float)):
            calibrated_signal += 0.35 * (float(calibrated_win) - 0.5)
        if isinstance(calibrated_delta, (int, float)) and math.isfinite(float(calibrated_delta)):
            calibrated_signal += 0.25 * math.tanh(float(calibrated_delta))
        key = percentile[i] + calibrated_signal
        if valid[i]["candidate_risk_reason"]:
            key -= 1.0
        key -= float(p_bad[i])
        if float(p_bad[i]) > stability_threshold:
            key -= 1.0
        return key

    for i, row in enumerate(valid):
        row["selection_key"] = selection_key(i)

    if reranker is not None:
        order = _rank_candidates_with_reranker(valid, n, stability_threshold)
    else:
        order = rank_candidates(valid, n, stability_threshold, lambda i: float(valid[i].get("selection_key", 0.0)))
    proposals: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    for rank, i in enumerate(order):
        row = dict(valid[i])
        row.pop("tokens", None)
        row.update(
            {
                "model_version": "policy_v3",
                "reward_rank": rank,
                "fallback": False,
                "candidate_selected": rank < n,
                "candidate_filter": "selected" if rank < n else "ranked_not_selected",
                "automaton_bridge_disagreements": disagreements,
            }
        )
        if candidate_audit is not None:
            candidate_audit.append(
                {
                    k: row.get(k)
                    for k in (
                        "engine",
                        "canonical_genome",
                        "decoder",
                        "reward",
                        "reward_std",
                        "reward_rank",
                        "reward_percentile_in_engine",
                        "p_bad",
                        "selection_key",
                        "selector_reranker_score",
                        "selection_score",
                        "selection_policy",
                        "calibrated_expected_delta",
                        "calibrated_win_rate",
                        "candidate_risk_reason",
                    )
                }
            )
        all_candidates.append(row)
        if rank < n:
            proposals.append(row)
    if return_all_candidates:
        return proposals, all_candidates
    return proposals


def _rank_candidates_with_reranker(
    valid: list[dict[str, Any]],
    n: int,
    stability_threshold: float,
) -> list[int]:
    """Selector 2.0 portfolio: top reranker rows first, then uncertainty/diversity/engine slots."""
    limit = max(0, int(n))
    base_order = sorted(
        range(len(valid)),
        key=lambda i: (
            _row_numeric(valid[i], "selector_reranker_score", default=_row_numeric(valid[i], "selection_key")),
            _row_numeric(valid[i], "selection_key"),
            _row_numeric(valid[i], "reward"),
            -int(valid[i].get("proposal_index") if isinstance(valid[i].get("proposal_index"), int) else 10**9),
        ),
        reverse=True,
    )
    if limit <= 0:
        return base_order
    selected: list[int] = []
    selected_set: set[int] = set()

    def safe(i: int) -> bool:
        return (
            not valid[i].get("candidate_risk_reason")
            and _row_numeric(valid[i], "p_bad", default=0.0) <= stability_threshold
        )

    def add(i: int) -> bool:
        if i in selected_set:
            return False
        selected.append(i)
        selected_set.add(i)
        return len(selected) >= limit

    for i in base_order:
        if safe(i) and add(i) and len(selected) >= min(3, limit):
            break
        if len(selected) >= min(3, limit):
            break
    if len(selected) >= limit:
        return selected + [i for i in base_order if i not in selected_set]

    uncertainty = sorted(
        [i for i in base_order if i not in selected_set and safe(i)],
        key=lambda i: (_row_numeric(valid[i], "reward_std"), _row_numeric(valid[i], "selection_key")),
        reverse=True,
    )
    if uncertainty and add(uncertainty[0]):
        return selected + [i for i in base_order if i not in selected_set]

    seen_diversity = {str(valid[i].get("candidate_diversity_key") or "") for i in selected}
    diverse = [
        i
        for i in base_order
        if i not in selected_set
        and safe(i)
        and bool(valid[i].get("novel_vs_historical", True))
        and str(valid[i].get("candidate_diversity_key") or "") not in seen_diversity
    ]
    if diverse and add(diverse[0]):
        return selected + [i for i in base_order if i not in selected_set]

    engine_counts: dict[str, int] = {}
    for i in selected:
        engine = str(valid[i].get("engine") or "")
        engine_counts[engine] = engine_counts.get(engine, 0) + 1
    engines = sorted({str(row.get("engine") or "") for row in valid})
    underrepresented = sorted(engines, key=lambda e: (engine_counts.get(e, 0), e))
    for engine in underrepresented:
        bucket = [i for i in base_order if i not in selected_set and safe(i) and str(valid[i].get("engine") or "") == engine]
        if bucket and add(bucket[0]):
            return selected + [i for i in base_order if i not in selected_set]

    for i in base_order:
        if add(i):
            break
    return selected + [i for i in base_order if i not in selected_set]


def _candidate_diversity_key_v3(
    actions: dict[str, Any],
    head_type: Any,
    family_bitmap: Any,
) -> str:
    transforms = [stage for stage in actions.get("transforms") or [] if isinstance(stage, dict)]
    first = str(transforms[0].get("op") if transforms else "none")
    head = str(head_type or ((actions.get("head") or {}).get("op") if isinstance(actions.get("head"), dict) else "") or "unknown")
    families = family_bitmap if isinstance(family_bitmap, dict) else {}
    active = sorted(str(k) for k, v in families.items() if v)
    family = active[0] if active else first
    return f"{head}:{family}"


def _row_numeric(row: dict[str, Any], key: str, default: float = float("-inf")) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default

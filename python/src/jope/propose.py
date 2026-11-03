from __future__ import annotations

import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data.active import TARGET_SUMMARY_DIM, format_pipeline_actions, parse_pipeline_actions
from .data.target_encoding import target_vs_embedding
from .data.vs import DATASET_SIGNATURE_DIM, dataset_signature_from_vs, dataset_vector_from_vs
from .grammar import HEADS, KNOWN_VALID_FALLBACK, TRANSFORMS, DecodeChoice, decode_choice, sample_genome
from .model import (
    HEAD_LIST,
    TRANSFORM_LIST,
    JopeActiveModelV2,
    JopeOneShotModel,
    decode_active_v2_output,
    genome_features_from_actions,
)
from .schema import EMBEDDING_DIM, normalize_embedding
from .selector_calibration import load_selector_calibration, score_candidate_with_calibration
from .validator import RustValidator


def load_embedding(path: str | Path) -> list[float]:
    path = Path(path)
    if path.suffix == ".npy":
        values = np.load(path).reshape(-1).astype(float).tolist()
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        values = data.get("embedding", data if isinstance(data, list) else [])
    return normalize_embedding(values)


def load_standardization(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path).resolve()
    standardization = json.loads(path.read_text(encoding="utf-8"))
    from .encoders.projection import resolve_foundation_paths

    return resolve_foundation_paths(standardization, path.parent)


def embedding_from_vs(path: str | Path, standardization: dict[str, Any] | None = None) -> list[float]:
    if standardization and int(standardization.get("dim") or 0) == DATASET_SIGNATURE_DIM:
        values = dataset_signature_from_vs(path).astype(np.float32)
    else:
        values = dataset_vector_from_vs(path).astype(np.float32)
    if standardization is not None:
        mean = np.asarray(standardization.get("mean", []), dtype=np.float32)
        scale = np.asarray(standardization.get("scale", []), dtype=np.float32)
        if mean.shape == values.shape and scale.shape == values.shape:
            scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
            values = (values - mean) / scale
    # v3.2: append the foundation-expert block when the corpus was built with one
    # (standardization.json carries the projector spec; dataset hash = dir name).
    if standardization is not None and standardization.get("foundation"):
        from .encoders.projection import FoundationProjector  # noqa: PLC0415

        proj = FoundationProjector.from_spec(standardization["foundation"])
        block = proj.block_for_hash(Path(path).parent.name)
        values = np.concatenate([values, block])
    return normalize_embedding(values.tolist())


def target_summary_from_vs(path: str | Path | None) -> list[float]:
    if path is None:
        return [0.0] * TARGET_SUMMARY_DIM
    path = Path(path)
    if not path.exists():
        return [0.0] * TARGET_SUMMARY_DIM
    values = target_vs_embedding(path, dim=TARGET_SUMMARY_DIM)
    return _normalize_target_summary(values)


def propose_one(
    embedding: list[float] | np.ndarray,
    *,
    model: JopeOneShotModel | None = None,
    validator: RustValidator | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    validator = validator or RustValidator()
    rng = random.Random(seed)
    prediction: dict[str, float] = {}
    if model is not None:
        genome, prediction = _model_genome(embedding, model, rng)
    else:
        genome = sample_genome(rng)
    result = validator.validate_one(genome)
    result.update(prediction)
    if result.get("valid"):
        result["fallback"] = False
        return result
    fallback = validator.validate_one(KNOWN_VALID_FALLBACK)
    fallback.update(prediction)
    fallback["fallback"] = True
    fallback["failed_genome"] = genome
    fallback["failed_errors"] = result.get("errors", [])
    return fallback


def load_model(path: str | Path, map_location: str = "cpu") -> JopeOneShotModel:
    model = JopeOneShotModel()
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()
    return model


def load_any_model(path: str | Path, map_location: str = "cpu"):
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and str(state.get("model_version", "")).startswith("policy_v3"):
        from .policy.model_v3 import load_policy_v3

        return load_policy_v3(state)
    if isinstance(state, dict) and (state.get("model_version") == "active_v2" or state.get("vocab")):
        model = JopeActiveModelV2(
            state.get("vocab"),
            dataset_dim=int(state.get("dataset_dim") or EMBEDDING_DIM),
            target_dim=int(state.get("target_dim") or TARGET_SUMMARY_DIM),
            hidden_dim=int(state.get("hidden_dim") or 256),
            dropout=float(state.get("dropout") if state.get("dropout") is not None else 0.1),
        )
        # strict=False so legacy checkpoints (predating the reward head / loss log-vars)
        # still load; their new params stay at init and should be retrained.
        model.load_state_dict(state["model"] if "model" in state else state, strict=False)
        model.eval()
        return model
    model = JopeOneShotModel()
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()
    return model


def propose_active_v2(
    embedding: list[float] | np.ndarray,
    *,
    target_summary: list[float] | np.ndarray | None = None,
    model: JopeActiveModelV2,
    validator: RustValidator | None = None,
    n: int = 1,
    seed: int = 0,
    temperature: float = 0.0,
    beam: bool = True,
    historical_genomes: list[str] | None = None,
    candidate_audit: list[dict[str, Any]] | None = None,
    selector_calibration: str | Path | dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    validator = validator or RustValidator()
    calibration = load_selector_calibration(selector_calibration)
    x = torch.tensor(
        [normalize_embedding(list(np.asarray(embedding).reshape(-1).astype(float)))], dtype=torch.float32
    )
    target = torch.tensor([_normalize_target_summary(target_summary)], dtype=torch.float32)
    with torch.no_grad():
        out = model(x, target)
    candidates = _active_v2_candidate_genomes(
        out,
        model.vocab,
        n=n,
        seed=seed,
        temperature=temperature,
        beam=beam,
        historical_genomes=historical_genomes,
    )
    base_summary = _active_v2_prediction_summary(out)
    valid: list[dict[str, Any]] = []
    valid_features: list[list[float]] = []
    invalid: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen: set[str] = set()
    validation_rows = validator.validate_many([genome for genome, _ in candidates])
    for idx, ((genome, decode_meta), result) in enumerate(zip(candidates, validation_rows, strict=True)):
        result.update(base_summary)
        result.update(decode_meta)
        result["model_version"] = "active_v2"
        result["proposal_index"] = idx
        result["decoder_temperature"] = temperature
        if result.get("valid"):
            canonical = str(result.get("canonical_genome") or genome)
            if canonical in seen:
                result["candidate_filter"] = "duplicate_canonical"
                result["candidate_selected"] = False
                duplicates.append(result)
                continue
            seen.add(canonical)
            actions = parse_pipeline_actions(canonical)
            risk = candidate_risk_reason(actions)
            result["candidate_risk_reason"] = risk
            result["candidate_diversity_key"] = _candidate_diversity_key(actions)
            valid_features.append(
                genome_features_from_actions(actions, result.get("family_bitmap"), model.vocab)
            )
            valid.append(result)
        else:
            result["candidate_filter"] = "invalid"
            result["candidate_selected"] = False
            invalid.append(result)

    proposals: list[dict[str, Any]] = []
    if valid:
        features = torch.tensor(valid_features, dtype=torch.float32)
        with torch.no_grad():
            rewards = model.score_genomes(x, target, features).reshape(-1).tolist()
        for result, reward in zip(valid, rewards, strict=True):
            result["reward"] = float(reward)
        if calibration is not None:
            for result in valid:
                result.update(score_candidate_with_calibration(result, calibration))
        for result in valid:
            result["selection_slot"] = None
        selected = _select_portfolio_candidates(
            valid,
            n=min(max(1, n), 3),
            calibrated_score_key="selection_score" if calibration is not None else None,
        )
        for rank, result in enumerate(selected):
            result = dict(result)
            result["reward_rank"] = rank
            result["fallback"] = False
            result["candidate_selected"] = True
            result["candidate_filter"] = "selected"
            proposals.append(result)
        selected_ids = {id(row) for row in selected}
        for row in valid:
            if id(row) in selected_ids:
                row["candidate_selected"] = True
                row["candidate_filter"] = "selected"
            elif row.get("candidate_risk_reason"):
                row["candidate_selected"] = False
                row["candidate_filter"] = "risk_guard"
            else:
                row["candidate_selected"] = False
                row["candidate_filter"] = "reward_or_diversity"
    if candidate_audit is not None:
        for row in sorted(
            valid + duplicates + invalid, key=lambda item: int(item.get("proposal_index") or 0)
        ):
            candidate_audit.append(_audit_candidate_row(row))
    if not proposals:
        fallback = validator.validate_one(KNOWN_VALID_FALLBACK)
        fallback.update(base_summary)
        fallback["model_version"] = "active_v2"
        fallback["fallback"] = True
        fallback["selection_slot"] = "fallback_fill"
        if calibration is not None:
            fallback.update(score_candidate_with_calibration(fallback, calibration))
        if invalid:
            fallback["failed_genome"] = invalid[0].get("input_genome")
            fallback["failed_errors"] = invalid[0].get("errors", [])
        proposals.append(fallback)
    return proposals


def _model_genome(
    embedding: list[float] | np.ndarray,
    model: JopeOneShotModel,
    rng: random.Random,
) -> tuple[str, dict[str, float]]:
    values = normalize_embedding(list(np.asarray(embedding).reshape(-1).astype(float)))
    x = torch.tensor([values], dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    count = int(out["transform_count_logits"].argmax(dim=-1).item())
    count = max(0, min(count, 6))
    transform_order = torch.argsort(out["transform_logits"][0], descending=True).tolist()
    transforms = tuple(TRANSFORM_LIST[i] for i in transform_order[:count] if TRANSFORM_LIST[i] in TRANSFORMS)
    head_idx = int(out["head_logits"].argmax(dim=-1).item())
    head = HEAD_LIST[head_idx] if HEAD_LIST[head_idx] in HEADS else "ridge"
    prediction = {
        "predicted_oof_delta": float(out["value"].item()),
        "uncertainty": float(out["uncertainty"].item()),
    }
    return decode_choice(DecodeChoice(transforms=transforms, head=head), rng), prediction


def _normalize_target_summary(values: list[float] | np.ndarray | None) -> list[float]:
    if values is None:
        return [0.0] * TARGET_SUMMARY_DIM
    out = [float(value) for value in list(np.asarray(values).reshape(-1))[:TARGET_SUMMARY_DIM]]
    if len(out) < TARGET_SUMMARY_DIM:
        out.extend([0.0] * (TARGET_SUMMARY_DIM - len(out)))
    return out


def _active_v2_candidate_genomes(
    out: dict[str, Any],
    vocab: dict[str, Any],
    *,
    n: int,
    seed: int,
    temperature: float,
    beam: bool,
    historical_genomes: list[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    if beam:
        count_scores = out["transform_count_logits"][0]
        head_scores = out["head_logits"][0]
        count_top = torch.topk(count_scores, k=min(3, count_scores.numel())).indices.tolist()
        head_top = torch.topk(head_scores, k=min(max(3, n), head_scores.numel())).indices.tolist()
        for count_idx in count_top:
            for head_idx in head_top:
                genome, _ = decode_active_v2_output(
                    out,
                    vocab,
                    row=0,
                    temperature=0.0,
                    seed=seed,
                    head_index=int(head_idx),
                    count_index=int(count_idx),
                )
                candidates.append(
                    (
                        genome,
                        {
                            "decoder": "beam",
                            "beam_count": int(count_idx),
                            "beam_head": int(head_idx),
                        },
                    )
                )
    attempts = max(n * 12, 32)
    sample_temperature = temperature if temperature > 0.0 else 0.7
    for attempt in range(attempts):
        genome, _ = decode_active_v2_output(
            out, vocab, row=0, temperature=sample_temperature, seed=seed + attempt + 1
        )
        candidates.append((genome, {"decoder": "sample", "sample_attempt": attempt}))
    candidates.extend(
        _historical_mutation_candidates(
            historical_genomes or [], vocab=vocab, seed=seed, limit=max(n * 4, 12)
        )
    )
    return candidates


def _active_v2_prediction_summary(out: dict[str, Any]) -> dict[str, float]:
    return {
        "predicted_oof_delta": float(out["value"][0].detach().cpu()),
        "predicted_rank_percentile": float(out["rank_percentile"][0].detach().cpu()),
        "predicted_top_candidate_prob": float(torch.sigmoid(out["top_candidate_logit"][0]).detach().cpu()),
        "predicted_runtime": float(out["runtime"][0].detach().cpu()),
        "predicted_complexity": float(out["complexity"][0].detach().cpu()),
        "uncertainty": float(out["uncertainty"][0].detach().cpu()),
    }


def _select_reward_ranked_candidates(
    valid: list[dict[str, Any]], *, n: int, score_key: str = "reward"
) -> list[dict[str, Any]]:
    ranked = sorted(
        valid,
        key=lambda row: float(
            row.get(score_key)
            if isinstance(row.get(score_key), (int, float))
            else row.get("reward")
            if isinstance(row.get("reward"), (int, float))
            else float("-inf")
        ),
        reverse=True,
    )
    safe = [row for row in ranked if not row.get("candidate_risk_reason")]
    selected: list[dict[str, Any]] = []
    diversity_seen: set[str] = set()
    for row in safe:
        key = str(row.get("candidate_diversity_key") or "")
        if key in diversity_seen:
            continue
        selected.append(row)
        diversity_seen.add(key)
        if len(selected) >= n:
            return selected
    for row in safe:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= n:
            return selected
    return selected


_PORTFOLIO_PREFERRED_HEADS = ("histgbm2", "gpr", "ridge")
_PORTFOLIO_DEFERRED_HEADS = {"ksmooth", "histgbm"}


def _select_portfolio_candidates(
    valid: list[dict[str, Any]],
    *,
    n: int,
    calibrated_score_key: str | None = None,
) -> list[dict[str, Any]]:
    limit = min(max(0, int(n)), 3)
    if limit <= 0:
        return []
    safe = [row for row in valid if not row.get("candidate_risk_reason")]
    if not safe:
        return []

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    selected_canonicals: set[str] = set()
    selected_diversity: set[str] = set()

    def add(row: dict[str, Any], slot: str) -> None:
        row["selection_slot"] = slot
        selected.append(row)
        selected_ids.add(id(row))
        selected_canonicals.add(_selection_canonical_key(row))
        selected_diversity.add(str(row.get("candidate_diversity_key") or ""))

    raw_best = _rank_candidates(safe, "reward")[0]
    add(raw_best, "raw_reward")
    if len(selected) >= limit:
        return selected

    if calibrated_score_key is not None:
        for row in _rank_candidates(safe, calibrated_score_key):
            if id(row) in selected_ids:
                continue
            if _selection_canonical_key(row) in selected_canonicals:
                continue
            if str(row.get("candidate_diversity_key") or "") in selected_diversity:
                continue
            add(row, "calibrated")
            break
        if len(selected) >= limit:
            return selected

    preferred_selected = False
    for head in _PORTFOLIO_PREFERRED_HEADS:
        bucket = [row for row in safe if id(row) not in selected_ids and _selection_head(row) == head]
        if not bucket:
            continue
        diverse = [
            row
            for row in bucket
            if str(row.get("candidate_diversity_key") or "") not in selected_diversity
            and _selection_canonical_key(row) not in selected_canonicals
        ]
        pick = _rank_candidates(diverse or bucket, calibrated_score_key or "reward")[0]
        add(pick, "preferred_head")
        preferred_selected = True
        break
    if len(selected) >= limit:
        return selected

    fallback_pool = [row for row in safe if id(row) not in selected_ids]
    if preferred_selected:
        fallback_pool = [
            row for row in fallback_pool if _selection_head(row) not in _PORTFOLIO_DEFERRED_HEADS
        ] or fallback_pool
    else:
        non_deferred = [row for row in fallback_pool if _selection_head(row) not in _PORTFOLIO_DEFERRED_HEADS]
        fallback_pool = non_deferred or fallback_pool

    for pool in (
        [
            row
            for row in fallback_pool
            if str(row.get("candidate_diversity_key") or "") not in selected_diversity
            and _selection_canonical_key(row) not in selected_canonicals
        ],
        fallback_pool,
    ):
        for row in _rank_candidates(pool, calibrated_score_key or "reward"):
            if id(row) in selected_ids:
                continue
            add(row, "fallback_fill")
            if len(selected) >= limit:
                return selected
    return selected


def _rank_candidates(rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -_candidate_numeric(row, score_key),
            -_candidate_numeric(row, "reward"),
            int(row.get("proposal_index") if isinstance(row.get("proposal_index"), int) else 10**9),
            _selection_canonical_key(row),
        ),
    )


def _candidate_numeric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return float(value)
    value = row.get("reward")
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return float(value)
    return float("-inf")


def _selection_canonical_key(row: dict[str, Any]) -> str:
    return str(row.get("canonical_genome") or row.get("input_genome") or row.get("genome") or "")


def _selection_head(row: dict[str, Any]) -> str:
    head = row.get("head_type")
    if isinstance(head, str) and head:
        return head
    try:
        actions = parse_pipeline_actions(_selection_canonical_key(row))
        return str((actions.get("head") or {}).get("op") or "")
    except Exception:
        return ""


_EXPANSIVE_TRANSFORMS = {
    "bin_onehot",
    "cheb",
    "hinge",
    "oblique_hinge",
    "pair_prod",
    "poly_interact",
    "quantile_bins",
    "rand_proj",
    "rff_cos",
    "rff_sin",
}
_REPEATED_EXPLOSIVE_TRANSFORMS = _EXPANSIVE_TRANSFORMS | {"sigmoid"}


def candidate_risk_reason(actions: dict[str, Any]) -> str | None:
    transforms = [stage for stage in actions.get("transforms") or [] if isinstance(stage, dict)]
    head = actions.get("head") if isinstance(actions.get("head"), dict) else {}
    head_op = str(head.get("op") or "")
    counts = Counter(str(stage.get("op") or "") for stage in transforms)
    for op, count in counts.items():
        if op in _REPEATED_EXPLOSIVE_TRANSFORMS and count > 2:
            return f"repeated_explosive_transform:{op}"
    expansion_count = sum(1 for stage in transforms if str(stage.get("op") or "") in _EXPANSIVE_TRANSFORMS)
    appended_expansion_count = sum(
        1
        for stage in transforms
        if str(stage.get("op") or "") in _EXPANSIVE_TRANSFORMS
        and bool((stage.get("params") or {}).get("append"))
    )
    if head_op in {"gpr", "ridge", "elasticnet"} and (expansion_count >= 5 or appended_expansion_count >= 3):
        return f"excessive_feature_expansion_before:{head_op}"
    if head_op == "gpr":
        kernel = str((head.get("params") or {}).get("kernel") or "rbf")
        if "periodic" in kernel and expansion_count >= 2:
            return "unstable_gpr_periodic_with_expansion"
        if len(transforms) > 4 and expansion_count >= 3:
            return "unstable_gpr_large_expansion"
    return None


def _candidate_diversity_key(actions: dict[str, Any]) -> str:
    transforms = [stage for stage in actions.get("transforms") or [] if isinstance(stage, dict)]
    head = actions.get("head") if isinstance(actions.get("head"), dict) else {}
    head_op = str(head.get("op") or "?")
    first = str(transforms[0].get("op") if transforms else "none")
    return f"{head_op}:{first}"


def _audit_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    audit = {
        key: value
        for key, value in row.items()
        if key
        in {
            "canonical_genome",
            "input_genome",
            "valid",
            "errors",
            "decoder",
            "proposal_index",
            "reward",
            "reward_rank",
            "selection_score",
            "selection_policy",
            "calibrated_expected_delta",
            "calibrated_win_rate",
            "calibrated_tail_risk",
            "selection_score_components",
            "selection_slot",
            "predicted_oof_delta",
            "uncertainty",
            "head_type",
            "candidate_selected",
            "candidate_filter",
            "candidate_risk_reason",
            "candidate_diversity_key",
            "beam_count",
            "beam_head",
            "sample_attempt",
            "mutation_attempt",
            "mutation_source",
            "model_version",
        }
    }
    audit.setdefault("candidate_selected", False)
    audit.setdefault("candidate_filter", "unknown")
    audit.setdefault("selection_slot", None)
    return audit


def _historical_mutation_candidates(
    historical_genomes: list[str],
    *,
    vocab: dict[str, Any],
    seed: int,
    limit: int,
) -> list[tuple[str, dict[str, Any]]]:
    if not historical_genomes or limit <= 0:
        return []
    rng = random.Random(seed + 7919)
    out: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for source in historical_genomes[: max(1, min(len(historical_genomes), limit))]:
        for attempt in range(3):
            if len(out) >= limit:
                return out
            try:
                actions = parse_pipeline_actions(source)
                mutated = _mutate_actions(actions, vocab=vocab, rng=rng)
                genome = format_pipeline_actions(mutated)
            except Exception:
                continue
            if genome in seen:
                continue
            seen.add(genome)
            out.append(
                (
                    genome,
                    {
                        "decoder": "mutation",
                        "mutation_attempt": attempt,
                        "mutation_source": source,
                    },
                )
            )
    return out


def _mutate_actions(actions: dict[str, Any], *, vocab: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    mutated = copy.deepcopy(actions)
    transforms = list(mutated.get("transforms") or [])
    head = mutated.get("head") if isinstance(mutated.get("head"), dict) else {"op": "ridge", "params": {}}
    choices = ["toggle_append", "replace_head", "append_safe_transform", "drop_transform"]
    if not transforms:
        choices = ["replace_head", "append_safe_transform"]
    choice = rng.choice(choices)
    if choice == "drop_transform" and transforms:
        transforms.pop(rng.randrange(len(transforms)))
    elif choice == "append_safe_transform" and len(transforms) < 6:
        op = rng.choice(["robust_scale", "rank", "blom", "quantile_bins"])
        transforms.append(
            {"op": op, "params": _sample_stage_params(TRANSFORMS.get(op, {}), rng, allow_append=True)}
        )
    elif choice == "toggle_append" and transforms:
        stage = transforms[rng.randrange(len(transforms))]
        params = dict(stage.get("params") or {})
        params["append"] = not bool(params.get("append"))
        stage["params"] = params
    else:
        head_ops = [op for op in (vocab.get("head_ops") or sorted(HEADS)) if op in HEADS]
        if not head_ops:
            head_ops = sorted(HEADS)
        op = rng.choice(head_ops)
        head = {"op": op, "params": _sample_stage_params(HEADS.get(op, {}), rng, allow_append=False)}
    mutated["transforms"] = transforms
    mutated["head"] = head
    mutated["transform_count"] = len(transforms)
    return mutated


def _sample_stage_params(
    space: dict[str, list[object]],
    rng: random.Random,
    *,
    allow_append: bool,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, values in space.items():
        if values:
            params[key] = rng.choice(list(values))
    if allow_append and rng.random() < 0.35:
        params["append"] = True
    return params

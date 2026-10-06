"""Teacher-forced development and split-diagnostic metrics on fixtures (PLAN §5 P10 Metrics, §6.2, §6.7, §5 P7).

``dev_metrics`` reads only the ``dev`` split (role ``training_validation``) or the split-only diagnostic
``diag`` (role ``diagnostic_only``). Any other split, including every heldout name, is refused before the data
object is touched: holdout reads happen only at unsealing milestones (§5 P10), never here.

Metrics, all unsmoothed and teacher-forced:
- CE per token, family-macro (primary): mean over families of the mean over contexts of the context's
  token-mean masked CE over its teachers (group-stream contexts only, §6.1 teachers);
- CE per token, token-micro (secondary);
- token validity: the fraction of predicted positions whose unmasked argmax is a legal next token;
- within-context pairwise accuracy, family-macro, over pairs with distinct ``q_grid`` (§6.3 ties).

The §6.3.2 calibration readout: ``prior_shift_probs`` applies ``logit += log(pi_nat / pi_group)`` in float64;
``pi_group_from_journal`` pools the consumption journal's ``feas_class_counts`` (the members that entered L_feas,
i.e. the class prior the feasibility head was trained under); ``feasibility_readout`` reports raw and corrected
probabilities for every labelled program of a dev or diag context, with ``pi_nat`` per backend from the fit.

Evaluation runs in ``model.eval()`` under ``torch.no_grad()``, in fixed chunks, with no augmentation, restores the
previous training mode, and never consumes the global torch, Python or numpy RNG (asserted).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from dltrain import objective as objective_mod
from dltrain import sampler as sampler_mod

SCHEMA = "jope.dl-eval-dev/v1"
FEASIBILITY_SCHEMA = "jope.dl-eval-feasibility/v1"
ALLOWED_SPLITS = {"dev": "training_validation", "diag": "diagnostic_only"}
DEV_VALIDATION_SPLIT = "dev-family-v1"
EVAL_CHUNK = 64
PAD_ID = 0
N_FEAS_CLASSES = 3


class EvaluateRefused(RuntimeError):
    """The requested split may not be read by this evaluator."""


def _rng_snapshot():
    return torch.get_rng_state().clone(), random.getstate(), np.random.get_state()


def _rng_unchanged(before) -> bool:
    t0, p0, n0 = before
    t1, p1, n1 = _rng_snapshot()
    return (torch.equal(t0, t1) and p0 == p1 and n0[0] == n1[0] and np.array_equal(n0[1], n1[1])
            and tuple(n0[2:]) == tuple(n1[2:]))


def _check_split(split) -> None:
    if not isinstance(split, str) or split not in ALLOWED_SPLITS:
        raise EvaluateRefused(f"split {split!r} refused: only {sorted(ALLOWED_SPLITS)} are readable here; heldout "
                              "data is read only at unsealing milestones (PLAN §5 P10)")


def _split_contexts(corpus, split: str) -> list[str]:
    out = []
    for sha, row in corpus.contexts.items():
        if row["split_role"] != split:
            continue
        if split == "dev" and corpus.assignment.get(row["family_id"]) != "dev":
            continue
        out.append(sha)
    return sorted(out)


def _forward_programs(model, data, ctx: str, programs: list[str]) -> dict:
    """Per program: (per-position NLL list, legal-argmax hits, predicted positions, q_hat, feasibility logits as
    float64), in fixed chunks."""
    corpus = data.corpus
    z = torch.from_numpy(np.array(corpus.encodings[ctx], dtype=np.float32, copy=True)).reshape(1, -1)
    results = {}
    for start in range(0, len(programs), EVAL_CHUNK):
        chunk = programs[start:start + EVAL_CHUNK]
        seqs = [tuple(int(t) for t in corpus.tokens(ctx, p)) for p in chunk]
        n, width = len(seqs), max(len(s) for s in seqs)
        tokens = torch.full((n, width), PAD_ID, dtype=torch.long)
        legal = torch.zeros((n, width - 1, data.vocab_size), dtype=torch.bool)
        for i, seq in enumerate(seqs):
            tokens[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            legal[i, :len(seq) - 1] = torch.from_numpy(np.asarray(data.legal_mask_fn(seq), dtype=bool))
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        out = model(tokens, lengths, z, torch.zeros(n, dtype=torch.long))
        valid = objective_mod.valid_positions(lengths, width)
        nll = objective_mod.masked_token_nll(out.logits, tokens[:, 1:], legal, valid)
        hit = legal.gather(-1, out.logits.argmax(-1).unsqueeze(-1)).squeeze(-1)
        nll64 = nll.detach().double().numpy()
        feas64 = out.feas_logits.detach().double().numpy()
        for i, prog in enumerate(chunk):
            k = len(seqs[i]) - 1
            results[prog] = ([float(x) for x in nll64[i, :k]], int(hit[i, :k].sum()), k, float(out.q_hat[i]),
                             [float(x) for x in feas64[i]])
    return results


def _mean(values) -> float | None:
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def dev_metrics(model, data, *, split: str = "dev", max_pairs_programs: int = 64) -> dict:
    """Teacher-forced metrics for ``split`` in {'dev', 'diag'}; every other split raises EvaluateRefused first."""
    _check_split(split)
    if isinstance(max_pairs_programs, bool) or not isinstance(max_pairs_programs, int) or max_pairs_programs < 0:
        raise ValueError("max_pairs_programs must be a non-negative integer")
    corpus = data.corpus
    contexts = _split_contexts(corpus, split)
    rng_before = _rng_snapshot()
    was_training = model.training
    family_ce: dict[str, list[float]] = defaultdict(list)
    family_pairs: dict[str, list[float]] = defaultdict(list)
    families: set[str] = set()
    nll_terms: list[float] = []
    n_tokens = n_hits = n_sequences = n_pairs = 0
    model.eval()
    try:
        with torch.no_grad():
            for ctx in contexts:
                family = corpus.contexts[ctx]["family_id"]
                families.add(family)
                plan = sampler_mod.context_plan(corpus, ctx, data.config)
                ce_programs = list(plan.teachers) if plan.stream == "group" else []
                known = sorted((lab for lab in corpus.labels.records.get(ctx, ()) if lab.q_status == "known"),
                               key=lambda lab: (-lab.q_grid, lab.program_sha256))[:max_pairs_programs]
                programs = list(dict.fromkeys(ce_programs + [lab.program_sha256 for lab in known]))
                if not programs:
                    continue
                res = _forward_programs(model, data, ctx, programs)
                n_sequences += len(programs)
                if ce_programs:
                    ctx_terms = [x for p in ce_programs for x in res[p][0]]
                    ctx_count = sum(res[p][2] for p in ce_programs)
                    family_ce[family].append(math.fsum(ctx_terms) / ctx_count)
                    nll_terms.extend(ctx_terms)
                    n_tokens += ctx_count
                    n_hits += sum(res[p][1] for p in ce_programs)
                correct = total = 0
                for i in range(len(known)):
                    for k in range(i + 1, len(known)):
                        a, b = known[i], known[k]
                        if a.q_grid == b.q_grid:
                            continue
                        total += 1
                        correct += (res[a.program_sha256][3] - res[b.program_sha256][3]) * (a.q_grid - b.q_grid) > 0
                if total:
                    family_pairs[family].append(correct / total)
                    n_pairs += total
    finally:
        model.train(was_training)
    if not _rng_unchanged(rng_before):
        raise AssertionError("evaluation consumed global RNG state")
    return {
        "schema": SCHEMA,
        "split": split,
        "role": ALLOWED_SPLITS[split],
        "validation_split": DEV_VALIDATION_SPLIT if split == "dev" else None,
        "contexts": len(contexts),
        "families": len(families),
        "sequences": n_sequences,
        "tokens": n_tokens,
        "ce_per_token_family_macro": _mean(_mean(v) for v in family_ce.values()) if family_ce else None,
        "ce_per_token_token_micro": math.fsum(nll_terms) / n_tokens if n_tokens else None,
        "token_validity": n_hits / n_tokens if n_tokens else None,
        "pairwise_accuracy_family_macro": (_mean(_mean(v) for v in family_pairs.values())
                                           if family_pairs else None),
        "pairs": n_pairs,
    }


# ----------------------------------------------------------------------------- calibration readout (PLAN §6.3.2)

def _class_prior(values: Sequence[float], what: str, *, positive: bool) -> tuple[float, float, float]:
    """A 3-class prior as frequencies: finite, >= 0 (> 0 when ``positive``), with a positive sum."""
    vals = [float(v) for v in values]
    if len(vals) != N_FEAS_CLASSES or not all(math.isfinite(v) and v >= 0.0 for v in vals):
        raise ValueError(f"{what} must hold {N_FEAS_CLASSES} finite values >= 0, got {values!r}")
    if positive and min(vals) <= 0.0:
        raise ValueError(f"{what} needs every class > 0 (log({what}) of an absent class is undefined): {values!r}")
    total = math.fsum(vals)
    if total <= 0.0:
        raise ValueError(f"{what} has no mass: {values!r}")
    return tuple(v / total for v in vals)


def _softmax64(z: Sequence[float]) -> list[float]:
    """Float64 softmax; -inf entries get probability 0 (at least one entry must be finite)."""
    top = max(z)
    e = [math.exp(x - top) if math.isfinite(x) else 0.0 for x in z]
    total = math.fsum(e)
    return [x / total for x in e]


def prior_shift_probs(logits: Sequence[float], pi_nat: Sequence[float], pi_group: Sequence[float]) -> list[float]:
    """softmax(logits + log(pi_nat) - log(pi_group)) in float64 (PLAN §6.3.2 calibration readout).

    ``pi_group`` is the class prior the feasibility head was trained under and must be > 0 for every class; a class
    with ``pi_nat`` 0 gets probability 0. Both priors are normalized first (their scale cancels in the softmax)."""
    z = [float(x) for x in logits]
    if len(z) != N_FEAS_CLASSES or not all(math.isfinite(x) for x in z):
        raise ValueError(f"logits must hold {N_FEAS_CLASSES} finite values, got {logits!r}")
    nat = _class_prior(pi_nat, "pi_nat", positive=False)
    group = _class_prior(pi_group, "pi_group", positive=True)
    return _softmax64([x + math.log(n) - math.log(g) if n > 0.0 else -math.inf for x, n, g in zip(z, nat, group)])


def pi_group_from_journal(records: Iterable[Mapping], *, context_sha256: str | None = None) -> tuple[float, ...]:
    """pi_group (PLAN §6.3.2): the class frequencies of the members actually sampled that entered L_feas, pooled
    over the ``feas_class_counts`` of every consumption-journal slot, or of one context's slots. Remote and unknown
    backend members never enter it (§6.3 backend rule). ValueError when no such member was sampled."""
    counts = [0] * N_FEAS_CLASSES
    for rec in records:
        for slot in rec["slots"]:
            if context_sha256 is not None and slot["context_sha256"] != context_sha256:
                continue
            fc = slot["feas_class_counts"]
            if (not isinstance(fc, (list, tuple)) or len(fc) != N_FEAS_CLASSES
                    or any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in fc)):
                raise ValueError(f"journal step {rec.get('step')} slot {slot.get('slot')}: invalid feas_class_counts")
            counts = [a + b for a, b in zip(counts, fc)]
    total = sum(counts)
    if total == 0:
        raise ValueError("no L_feas member in the selected journal slots")
    return tuple(c / total for c in counts)


def feasibility_readout(model, data, *, split: str, pi_group: Sequence[float]) -> dict:
    """Raw and prior-shift corrected feasibility probabilities for every labelled program of each ``split`` context
    ('dev' or 'diag'; every other split raises EvaluateRefused before anything is read). The correction uses
    ``pi_nat`` of the row's backend (train_fit natural frequencies, ``fit.class_prior_nat``) and ``pi_group``;
    a backend without a natural prior gets ``probs_corrected`` None (unknown, never a guess)."""
    _check_split(split)
    group = _class_prior(pi_group, "pi_group", positive=True)
    corpus, prior_nat = data.corpus, data.fit.class_prior_nat
    contexts = _split_contexts(corpus, split)
    rng_before = _rng_snapshot()
    was_training = model.training
    rows: list[dict] = []
    model.eval()
    try:
        with torch.no_grad():
            for ctx in contexts:
                labs = [lab for lab in corpus.labels.records.get(ctx, ()) if lab.feas_class is not None]
                if not labs:
                    continue
                res = _forward_programs(model, data, ctx, [lab.program_sha256 for lab in labs])
                for lab in labs:
                    logits = res[lab.program_sha256][4]
                    nat = prior_nat.get(lab.backend)
                    rows.append({"context_sha256": ctx, "family_id": corpus.contexts[ctx]["family_id"],
                                 "program_sha256": lab.program_sha256, "backend": lab.backend,
                                 "feas_class": lab.feas_class, "feas_valid": lab.feas_valid, "logits": logits,
                                 "probs_raw": _softmax64(logits),
                                 "probs_corrected": None if nat is None else prior_shift_probs(logits, nat, group)})
    finally:
        model.train(was_training)
    if not _rng_unchanged(rng_before):
        raise AssertionError("evaluation consumed global RNG state")
    return {"schema": FEASIBILITY_SCHEMA, "split": split, "role": ALLOWED_SPLITS[split], "contexts": len(contexts),
            "pi_group": list(group), "pi_nat": {b: list(v) for b, v in sorted(prior_nat.items())}, "programs": rows}

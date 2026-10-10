"""Differentiable trie log-probability and IPO/DPO steps.

ell matches rlfw.trie_logprob.child_logprob: summed, EOS-inclusive, trie
children only, dropout off. q_hat is never read. lambda_kl=0 is pure IPO/DPO.
The KL term is the path KL on the pair, not the full-trie KL.
"""

from __future__ import annotations

import math

import torch

from .losses import DPO_BETA, IPO_TAUS


def child_logprob_torch(logits, choices, chosen):
    """One-step trie log-prob. A single child contributes 0."""
    choice_ids = tuple(sorted(int(token) for token in choices))
    if not choice_ids or int(chosen) not in choice_ids:
        raise ValueError("chosen token is outside trie children")
    if len(choice_ids) == 1:
        return logits.new_zeros(())
    index = torch.tensor(choice_ids, device=logits.device)
    values = logits[index]
    if not torch.isfinite(values).all():
        raise ValueError("finite supported logits required")
    peak = values.max()
    log_z = peak + torch.log(torch.exp(values - peak).sum())
    return logits[int(chosen)] - log_z


def sequence_logprob(logits_row, tokens, trie):
    """Sum child log-probs. logits_row[i] predicts tokens[i + 1]."""
    tokens = tuple(tokens)
    if tokens not in trie.programs:
        raise ValueError("complete registered trie leaf including EOS required")
    if logits_row.shape[0] < len(tokens) - 1:
        raise ValueError("logits do not cover the program")
    total = logits_row.new_zeros(())
    for position, chosen in enumerate(tokens[1:], 1):
        total = total + child_logprob_torch(
            logits_row[position - 1], trie.get(tokens[:position]), chosen
        )
    return total


def model_sequence_logprob(model, tokens, trie, ctx_z):
    """Teacher-forced trie ell. The forward's q_hat is ignored."""
    was_training = model.training
    model.eval()
    try:
        device = ctx_z.device
        if ctx_z.dim() == 1:
            ctx_z = ctx_z.view(1, -1)
        token_row = torch.tensor(tuple(tokens), dtype=torch.long, device=device).view(1, -1)
        length = torch.tensor([token_row.shape[1]], dtype=torch.long, device=device)
        seq_ctx = torch.zeros(1, dtype=torch.long, device=device)
        outputs = model(token_row, length, ctx_z, seq_ctx)
        return sequence_logprob(outputs.logits[0], tokens, trie)
    finally:
        model.train(was_training)


def _dpo_loss(margin):
    z = DPO_BETA * margin
    # -log sigmoid(z), stable for either sign.
    return torch.where(z >= 0, torch.nn.functional.softplus(-z), torch.nn.functional.softplus(z) - z)


def _path_kl(model, reference, tokens, trie, ctx_z):
    """Mean KL(policy || reference) on trie children along one program."""
    device = ctx_z.device
    if ctx_z.dim() == 1:
        ctx_z = ctx_z.view(1, -1)
    token_row = torch.tensor(tuple(tokens), dtype=torch.long, device=device).view(1, -1)
    length = torch.tensor([token_row.shape[1]], dtype=torch.long, device=device)
    seq_ctx = torch.zeros(1, dtype=torch.long, device=device)
    was_model = model.training
    was_ref = reference.training
    model.eval()
    reference.eval()
    try:
        policy_logits = model(token_row, length, ctx_z, seq_ctx).logits[0]
        with torch.no_grad():
            ref_logits = reference(token_row, length, ctx_z, seq_ctx).logits[0]
    finally:
        model.train(was_model)
        reference.train(was_ref)
    terms = []
    for position, _chosen in enumerate(tokens[1:], 1):
        children = tuple(sorted(int(token) for token in trie.get(tokens[:position])))
        if len(children) <= 1:
            continue
        index = torch.tensor(children, device=device)
        policy = torch.log_softmax(policy_logits[position - 1, index], dim=0)
        ref = torch.log_softmax(ref_logits[position - 1, index], dim=0)
        terms.append(torch.exp(policy).mul(policy - ref).sum())
    if not terms:
        return policy_logits.new_zeros(())
    return torch.stack(terms).mean()


def preference_loss(
    model, reference, winner, loser, trie, ctx_z, kind, tau=2.0, lambda_kl=0.0
):
    """IPO or DPO on one measured pair. No q, q_hat, or fitness argument."""
    if reference is model:
        raise ValueError("reference must be a frozen copy")
    if kind not in ("ipo", "dpo"):
        raise ValueError("kind must be ipo or dpo")
    if kind == "ipo" and tau not in IPO_TAUS:
        raise ValueError("tau_IPO must be one of 2, 5, 10")
    if not math.isfinite(lambda_kl) or lambda_kl < 0:
        raise ValueError("lambda_kl must be finite and >= 0")
    winner_logp = model_sequence_logprob(model, winner, trie, ctx_z)
    loser_logp = model_sequence_logprob(model, loser, trie, ctx_z)
    with torch.no_grad():
        ref_winner = model_sequence_logprob(reference, winner, trie, ctx_z)
        ref_loser = model_sequence_logprob(reference, loser, trie, ctx_z)
    margin = (winner_logp - loser_logp) - (ref_winner - ref_loser)
    if kind == "ipo":
        target = 1.0 / (2.0 * tau)
        base = (margin - target) ** 2
    else:
        base = _dpo_loss(margin)
    path_kl = margin.new_zeros(())
    if lambda_kl > 0:
        path_kl = 0.5 * (
            _path_kl(model, reference, winner, trie, ctx_z)
            + _path_kl(model, reference, loser, trie, ctx_z)
        )
    return {
        "loss": base + lambda_kl * path_kl,
        "base": base,
        "margin": margin,
        "path_kl": path_kl,
        "kind": kind,
        "tau": tau if kind == "ipo" else None,
        "lambda_kl": lambda_kl,
    }


def preference_step(
    model, reference, winner, loser, trie, ctx_z, optimizer, kind, tau=2.0, lambda_kl=0.0
):
    """One AdamW step in eval mode so dropout stays off while gradients flow."""
    model.eval()
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    terms = preference_loss(
        model, reference, winner, loser, trie, ctx_z, kind, tau=tau, lambda_kl=lambda_kl
    )
    optimizer.zero_grad(set_to_none=True)
    terms["loss"].backward()
    optimizer.step()
    return {
        "loss": float(terms["loss"].detach()),
        "base": float(terms["base"].detach()),
        "margin": float(terms["margin"].detach()),
        "path_kl": float(terms["path_kl"].detach()),
        "kind": kind,
        "tau": tau if kind == "ipo" else None,
        "lambda_kl": lambda_kl,
    }

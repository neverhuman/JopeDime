"""Default-zero objective extension and differentiable deployed-trie losses."""

from contextlib import contextmanager
import math
import time


class TensorPolicy:
    """Gradient-carrying scalar prefix forward around a verified PR1 policy."""

    def __init__(self, pinned):
        self.pinned = pinned
        self.net = pinned.net
        self.trie_sha256 = pinned.trie.identity

    @property
    def training(self):
        return self.net.training

    def eval(self):
        self.net.eval()

    def train(self, mode):
        self.net.train(mode)

    def connected_zero(self):
        return next(self.net.parameters()).sum() * 0

    def next_logits(self, context, prefix):
        torch = self.pinned.torch
        device = next(self.net.parameters()).device
        ctx = torch.as_tensor(context, dtype=torch.float32, device=device).reshape(
            1, -1
        )
        tokens = torch.tensor(
            [tuple(prefix) + (self.net.cfg.pad_id,)], dtype=torch.long, device=device
        )
        lengths = torch.tensor([tokens.shape[1]], dtype=torch.long, device=device)
        return self.net(
            tokens, lengths, ctx, torch.zeros(1, dtype=torch.long, device=device)
        ).logits[0, len(prefix) - 1]


@contextmanager
def dropout_off(policy):
    previous = policy.training
    policy.eval()
    try:
        yield
    finally:
        policy.train(previous)


def child_distribution(logits, choices, temperature=1.0):
    import torch

    choices = tuple(sorted(choices))
    if not choices or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("finite positive temperature and trie children required")
    values = logits[list(choices)].double() / temperature
    if not torch.isfinite(values).all():
        raise ValueError("finite supported logits required")
    return choices, torch.log_softmax(values, dim=0)


def trie_logprob(policy, context, tokens, trie, *, temperature=1.0, reference=None):
    """Gradient-carrying extension checked against PR1's scalar likelihood API."""
    if reference is None:
        from .trie_logprob import child_logprob as reference
    tokens = tuple(tokens)
    if tokens not in trie.programs:
        raise ValueError("complete registered trie leaf including EOS required")
    values = []
    with dropout_off(policy):
        for position, chosen in enumerate(tokens[1:], 1):
            logits = policy.next_logits(context, tokens[:position])
            choices, logp = child_distribution(
                logits, trie.get(tokens[:position]), temperature
            )
            if chosen not in choices:
                raise ValueError("chosen token outside deployed trie support")
            value = logp[choices.index(chosen)]
            expected = reference(
                logits.detach().double().cpu(), choices, chosen, temperature
            )
            if abs(float(value.detach()) - expected) > 1e-12:
                raise ValueError("differentiable likelihood differs from PR1")
            values.append(value)
    return sum(values), tuple(values)


def policy_imitation(
    policy, contexts, target_contexts, trie, *, check_train, reference=None
):
    """Prefix reach mass makes this summed sequence likelihood, never token mean."""
    losses = []
    with dropout_off(policy):
        for target in target_contexts:
            cid = target["context_id"]
            check_train(cid)
            if target["role"] != "train":
                raise ValueError("development target in policy objective")
            if not target["eligible"]:
                continue
            if reference is None:
                from .trie_logprob import child_logprob as reference
            total = policy.connected_zero()
            for item in target["prefixes"]:
                prefix = tuple(item["prefix"])
                logits = policy.next_logits(contexts[cid], prefix)
                choices, logp = child_distribution(logits, trie.get(prefix))
                for token, mass in item["children"]:
                    if token not in choices:
                        raise ValueError("target support differs from frozen trie")
                    expected = reference(
                        logits.detach().double().cpu(), choices, token, 1.0
                    )
                    if (
                        abs(float(logp[choices.index(token)].detach()) - expected)
                        > 1e-12
                    ):
                        raise ValueError("actor likelihood differs from PR1")
                    total = (
                        total - item["reach_mass"] * mass * logp[choices.index(token)]
                    )
            check_train(cid)
            losses.append(total)
    return sum(losses) / len(losses) if losses else policy.connected_zero()


def sampled_kl(policy, anchor, contexts, token_rows, trie, *, reference_fn=None):
    """Unbiased score-function KL gradient from fresh unselected theta draws.

    This is a training estimator; only exact_trie_kl qualifies the 64-context
    probe. The forward value is the sample mean log ratio and may be negative.
    """
    values = []
    for context, tokens in zip(contexts, token_rows):
        theta, _ = trie_logprob(policy, context, tokens, trie, reference=reference_fn)
        import torch

        with torch.no_grad():
            reference, _ = trie_logprob(
                anchor, context, tokens, trie, reference=reference_fn
            )
        ratio = theta - reference
        surrogate = theta * (ratio.detach() + 1)
        values.append(surrogate - surrogate.detach() + ratio.detach())
    if len(values) != len(contexts) or len(values) != len(token_rows) or not values:
        raise ValueError("complete fresh on-policy KL draw inventory required")
    return sum(values) / len(values)


def exact_trie_kl(policy, anchor, contexts, trie, *, deadline, check_train):
    """Exact full-support KL(theta||anchor), using the tree chain rule.

    Every conditional is visited. Singleton edges contribute zero without a
    forward. Timeout is an unavailable probe, never an approximate passing KL.
    """
    import torch

    if policy.trie_sha256 != trie.identity or anchor.trie_sha256 != trie.identity:
        raise ValueError("KL requires identical deployed support")
    losses = []
    with dropout_off(policy), dropout_off(anchor):
        for cid, context in contexts.items():
            check_train(cid)
            loss = policy.connected_zero().double()
            stack = [((1,), torch.ones((), dtype=torch.float64, device=loss.device))]
            terminal_mass = policy.connected_zero().double()
            while stack:
                if time.time() >= deadline:
                    raise TimeoutError(
                        "full-trie KL probe incomplete at fixed deadline"
                    )
                prefix, reach = stack.pop()
                children = tuple(sorted(trie.get(prefix)))
                if not children:
                    if prefix not in trie.programs or prefix[-1] != 2:
                        raise ValueError("nonterminal missing trie children")
                    terminal_mass = terminal_mass + reach
                    continue
                if len(children) == 1:
                    stack.append((prefix + (children[0],), reach))
                    continue
                choices, logp = child_distribution(
                    policy.next_logits(context, prefix), children
                )
                with torch.no_grad():
                    _, loga = child_distribution(
                        anchor.next_logits(context, prefix), children
                    )
                probabilities = logp.exp()
                loss = loss + reach * (probabilities * (logp - loga)).sum()
                stack.extend(
                    (prefix + (token,), reach * probabilities[i])
                    for i, token in enumerate(choices)
                )
            if abs(float(terminal_mass.detach()) - 1) > 1e-9:
                raise ValueError("full-trie probability closure failed")
            if not torch.isfinite(loss) or float(loss.detach()) < -1e-10:
                raise ValueError("invalid exact KL")
            check_train(cid)
            losses.append(loss)
    if not losses:
        raise ValueError("nonempty train probe set required")
    return sum(losses) / len(losses)


def objective(
    base_objective,
    out,
    batch,
    *,
    lambda_pi=0.0,
    lambda_kl=0.0,
    pi_loss=None,
    kl_loss=None,
):
    """Return the exact L6 result with both flags zero, before policy computation."""
    if any(not math.isfinite(x) or x < 0 for x in (lambda_pi, lambda_kl)):
        raise ValueError("finite nonnegative policy coefficients required")
    base = base_objective(out, batch)
    if lambda_pi == 0 and lambda_kl == 0:
        return base
    loss, metrics, parts = base
    terms = {}
    if lambda_pi:
        if pi_loss is None:
            raise ValueError("enabled policy imitation requires measured targets")
        terms["pi"] = pi_loss()
        loss = loss + lambda_pi * terms["pi"]
    if lambda_kl:
        if kl_loss is None:
            raise ValueError("enabled KL requires a frozen full-support anchor")
        terms["kl"] = kl_loss()
        loss = loss + lambda_kl * terms["kl"]
    if any(not math.isfinite(float(x.detach())) for x in (loss, *terms.values())):
        raise ValueError("finite extended objective required")
    return (
        loss,
        {
            **metrics,
            "policy_terms": {k: float(v.detach()) for k, v in terms.items()},
            "lambda_pi": lambda_pi,
            "lambda_kl": lambda_kl,
        },
        parts,
    )

"""Research-only native policy value adapter and audited feedback acquisition.

Frozen native representations preserve both original language models. Only this
adapter learns. Outcome access is restricted to explicitly acquired TRAIN pairs.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def parameter_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(canonical([name, str(value.dtype), list(value.shape)]))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cluster_split(rows, meta, gids, seed):
    """Union original parent, retained family, dataset and gid before splitting.

    Pool parent hashes alone do not capture all retained lineage equivalences.
    No measured quality is consulted.
    """
    index = {cid: i for i, cid in enumerate(meta["cids"])}
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    for row in rows:
        i = index[row["context_id"]]
        keys = ["parent:" + row["parent_id"], "family:" + meta["family"][i],
                "dataset:" + meta["dataset"][i], "gid:" + str(int(gids[i]))]
        for key in keys[1:]:
            union(keys[0], key)
    groups = {r["context_id"]: find("parent:" + r["parent_id"]) for r in rows}
    unique = sorted(set(groups.values()), key=lambda g: digest([seed, "split", g]))
    if len(unique) < 30:
        raise ValueError("fewer than 30 independent retained parent components")
    a, b = int(.65 * len(unique)), int(.80 * len(unique))
    roles = {g: "train" if i < a else "calibration" if i < b else "assessment"
             for i, g in enumerate(unique)}
    return {cid: {"group": group, "partition": roles[group]} for cid, group in groups.items()}


def trie_log_probs(logits, sequences):
    """Exact probabilities on the complete finite candidate trie, including EOS.

    logits[i, j] predicts sequence i's token j+1. Padding has no mass. A finite
    candidate pool is narrower than the original 6,640-program serving trie.
    """
    if len(sequences) != logits.shape[0] or len({tuple(x) for x in sequences}) != len(sequences):
        raise ValueError("unique candidate sequences required")
    children = {}
    for seq in sequences:
        if len(seq) < 2 or seq[0] != 1 or seq[-1] != 2 or 2 in seq[:-1]:
            raise ValueError("complete BOS/EOS sequences required")
        for j in range(1, len(seq)):
            children.setdefault(tuple(seq[:j]), set()).add(seq[j])
    out = []
    for i, seq in enumerate(sequences):
        terms = []
        for j in range(1, len(seq)):
            allowed = sorted(children[tuple(seq[:j])])
            terms.append(logits[i, j-1, seq[j]] - torch.logsumexp(logits[i, j-1, allowed], 0))
        out.append(torch.stack(terms).sum())
    return torch.stack(out)


@dataclass(frozen=True)
class Target:
    context_id: str
    member: int
    program_sha256: str
    quality: float
    outcome_sha256: str
    wave: int


class FeedbackLedger:
    """Create-only, idempotent selected-outcome boundary; assessment is forbidden."""

    def __init__(self, split, rows):
        self.split = split
        self.rows = {r["context_id"]: r for r in rows}
        self.targets = {}
        self.events = []

    def acquire(self, cid, member, *, wave, propensity):
        if cid not in self.split or self.split[cid]["partition"] != "train":
            raise ValueError("only TRAIN outcome acquisition is allowed")
        if not math.isfinite(propensity) or not 0 < propensity <= 1:
            raise ValueError("positive finite conditional propensity required")
        row = self.rows[cid]
        m = row["members"][member]
        key = (cid, member)
        if key in self.targets:
            return self.targets[key]
        # Failures retain an event and charged opportunity, but never become a
        # fabricated numerical target. The current archive contains complete rows.
        event = {"context_id": cid, "program_sha256": m["program_sha256"],
                 "status": m["status"], "wave": wave, "propensity": propensity,
                 "physical_new_fits": 0, "logical_replay_slots": 1}
        if m["status"] != "complete" or not isinstance(m.get("q"), (float, int)) or not math.isfinite(m["q"]):
            event["target"] = "refused_missing_or_failed_measurement"
            self.events.append(event)
            return None
        identity = {"context_id": cid, "parent_id": row["parent_id"],
                    "program_sha256": m["program_sha256"], "q": m["q"],
                    "protocol_sha256": row["protocol_sha256"], "status": m["status"]}
        target = Target(cid, member, m["program_sha256"], float(m["q"]), digest(identity), wave)
        self.targets[key] = target
        event.update(target="measured", outcome_sha256=target.outcome_sha256)
        self.events.append(event)
        return target

    def by_context(self):
        result = {}
        for target in self.targets.values():
            result.setdefault(target.context_id, []).append(target)
        return {cid: sorted(values, key=lambda t: t.member) for cid, values in result.items()}

    def identity(self):
        return digest([vars(t) for _, t in sorted(self.targets.items())])


def choose(available, mean, uncertainty, *, mode, rng):
    """No qualities, statuses or oracle ranks enter the acquisition function."""
    available = sorted(available)
    if not available:
        raise ValueError("candidate pool exhausted")
    if mode == "uniform":
        return int(rng.choice(available)), 1 / len(available)
    if mode != "targeted":
        raise ValueError("unknown acquisition policy")
    # 75% acquisition-function maximizer, 25% uniform. Explicit propensities
    # are conditional per draw, never claimed to be trajectory inclusion odds.
    scores = np.asarray(mean) + np.asarray(uncertainty)
    priority = max(available, key=lambda i: (scores[i], -i))
    chosen = priority if rng.random() < .75 else int(rng.choice(available))
    return chosen, .25 / len(available) + (.75 if chosen == priority else 0.)


class ValueAdapter(nn.Module):
    """A small residual MLP on frozen native representations and token counts."""

    def __init__(self, width):
        super().__init__()
        self.input_norm = nn.LayerNorm(width)
        self.hidden = nn.Sequential(nn.Linear(width, 192), nn.GELU(),
                                    nn.Linear(192, 96), nn.GELU())
        self.skip = nn.Linear(width, 1, bias=False)
        self.head = nn.Linear(96, 1)

    def forward(self, x):
        x = self.input_norm(x)
        return (self.head(self.hidden(x)) + self.skip(x)).squeeze(-1)


def measured_loss(prediction, quality):
    """Context-balanced measured ranking, listwise and robust value objectives."""
    if prediction.ndim != 1 or prediction.shape != quality.shape or len(quality) < 2:
        raise ValueError("at least two matched measured outcomes required")
    if not torch.isfinite(quality).all() or not torch.isfinite(prediction).all():
        raise ValueError("nonfinite measured values")
    difference = quality[:, None] - quality[None, :]
    mask = difference > 1e-8
    pair = F.softplus(prediction[None, :] - prediction[:, None])[mask].mean() if mask.any() else prediction.sum() * 0
    # Midrank handles exact ties; no winner is invented when all qualities tie.
    rank = ((difference > 1e-8).float().sum(1) + .5 * (difference.abs() <= 1e-8).float().sum(1)) / len(quality)
    target = torch.softmax(rank / .25, dim=0)
    listing = -(target * torch.log_softmax(prediction, dim=0)).sum()
    robust = F.huber_loss(torch.sigmoid(prediction), rank)
    return .5 * pair + .5 * listing + .25 * robust


def train_adapter(model, optimizer, features, ledger, *, steps, seed, batch_contexts=16):
    """All inputs with labels are acquired TRAIN targets; no full outcome table."""
    contexts = ledger.by_context()
    eligible = sorted(cid for cid, values in contexts.items() if len(values) >= 2)
    if not eligible:
        raise ValueError("no supported TRAIN contexts")
    rng = np.random.default_rng(seed)
    before = parameter_hash(model)
    receipts = []
    model.train()
    for step in range(steps):
        selected = rng.choice(eligible, size=min(batch_contexts, len(eligible)), replace=False).tolist()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for cid in selected:
            targets = contexts[cid]
            indices = [t.member for t in targets]
            quality = torch.tensor([t.quality for t in targets], dtype=torch.float32)
            losses.append(measured_loss(model(features[cid][indices]), quality))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite optimizer loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        receipts.append({"step": step + 1, "loss": float(loss.detach()),
                         "gradient_norm": float(norm), "contexts": selected,
                         "target_hashes": [t.outcome_sha256 for cid in selected for t in contexts[cid]],
                         "supported_contexts": len(selected), "lr": optimizer.param_groups[0]["lr"]})
    after = parameter_hash(model)
    if before == after:
        raise ValueError("optimizer did not change parameters")
    return {"parameter_before": before, "parameter_after": after,
            "snapshot_sha256": ledger.identity(), "optimizer_updates": steps, "steps": receipts}


@torch.no_grad()
def ensemble_predict(models, x):
    predictions = []
    for model in models:
        model.eval()
        predictions.append(model(x).numpy())
    values = np.stack(predictions)
    return values.mean(0), values.std(0)


def selection_metric(row, scores, budget):
    """Assessment only: best measured quality among equal-count selected programs."""
    if not np.isfinite(scores).all() or len(scores) != len(row["members"]):
        raise ValueError("finite score per candidate required")
    chosen = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), row["members"][i]["program_sha256"]))[:budget]
    qs = np.array([m["q"] for m in row["members"]], dtype=float)
    q = max(qs[chosen])
    span = qs.max() - qs.min()
    normalized = (q - qs.min()) / span if span > 1e-8 else 1.
    return {"quality": float(q), "normalized_quality": float(normalized),
            "regret": float(1 - normalized), "beats_incumbent": bool(q > row["incumbent_q"] + 1e-8),
            "best_found": bool(q >= qs.max() - 1e-8), "selected": chosen, "slots": len(chosen)}


def paired_interval(candidate, reference, groups, *, seed=7301, resamples=10000, alpha=.01):
    """Aggregate repeated contexts/seeds by retained parent, then cluster bootstrap."""
    ids = sorted(candidate)
    if ids != sorted(reference) or not ids:
        raise ValueError("same nonempty contexts required")
    by_parent = {}
    for cid in ids:
        by_parent.setdefault(groups[cid], []).append(candidate[cid] - reference[cid])
    delta = np.array([np.mean(v) for _, v in sorted(by_parent.items())])
    if len(delta) < 2:
        raise ValueError("at least two independent parents required")
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), size=(resamples, len(delta)))].mean(1)
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"effect": float(delta.mean()), "lower": float(low), "upper": float(high),
            "confidence": 1-alpha, "parents": len(delta), "contexts": len(ids), "resamples": resamples}

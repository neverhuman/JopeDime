"""Full finite-support policy/head readouts, without program execution.

CPU exact uses singleton active-row deployed prefix+PAD geometry, float32
MATH SDPA and float64 child logsumexp. Verification calls logprobs_many one
leaf at a time: arbitrary batched geometry is a different likelihood identity.
GPU KV is an analysis protocol requiring a declared and measured tolerance.
"""

import math
import random
import time
from collections import defaultdict

from .trie_logprob import child_logprob


def score_full_support(policy, context, *, batch_size=32):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive head batch size required")
    leaves = sorted(policy.trie.programs)
    started = time.perf_counter()
    heads = {}
    for offset in range(0, len(leaves), batch_size):
        chunk = leaves[offset : offset + batch_size]
        heads.update(zip(chunk, policy.score_heads([context] * len(chunk), chunk)))
    return heads, time.perf_counter() - started


def trie_enumerate(
    policy,
    context,
    *,
    mode="cpu_exact",
    temperature=1.0,
    head_batch_size=32,
    cache_bytes=64 * 1024**2,
):
    if mode not in ("cpu_exact", "kv_analysis"):
        raise ValueError("explicit CPU exact or KV analysis mode required")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive finite temperature required")
    if mode == "cpu_exact" and getattr(policy, "device", "cpu") != "cpu":
        raise ValueError("CPU exact requires CPU deployed geometry")
    started = time.perf_counter()
    policy.eval()
    engine = policy
    if mode == "kv_analysis":
        from .kv_decode import KVPolicy

        engine = KVPolicy(policy, context, cache_bytes=cache_bytes)
    leaves = sorted(policy.trie.programs)
    if not leaves or len({p[0] for p in leaves}) != 1:
        raise ValueError("nonempty trie with one BOS required")
    # Compress singleton chains. They consume no random choice or log mass.
    stack = [(0, (leaves[0][0],), ())]
    probabilities = {}
    calls = 0
    while stack:
        node, prefix, logterms = stack.pop()
        choices = policy.trie.nodes[node]
        while len(choices) == 1:
            token, node = next(iter(choices.items()))
            prefix += (token,)
            choices = policy.trie.nodes[node]
        if not choices:
            if prefix not in policy.trie.programs:
                raise ValueError("unregistered terminal")
            probabilities[prefix] = math.fsum(logterms)
            continue
        logits = engine.next_logits(context, prefix)
        calls += 1
        for token, child in sorted(choices.items(), reverse=True):
            stack.append(
                (
                    child,
                    prefix + (token,),
                    logterms + (child_logprob(logits, choices, token, temperature),),
                )
            )
    probability_seconds = time.perf_counter() - started
    if mode == "cpu_exact":
        heads, head_seconds = score_full_support(
            policy, context, batch_size=head_batch_size
        )
    else:
        start_heads = time.perf_counter()
        heads = {p: engine.heads(p) for p in leaves}
        head_seconds = time.perf_counter() - start_heads
    rows = [
        {
            "tokens": list(p),
            "program_sha256": policy.trie.programs[p]["program_sha256"],
            "length": len(p),
            "logprob": probabilities[p],
            "q_hat": heads[p]["q"],
            "p_complete": heads[p]["p_complete"],
            "runtime_seconds": heads[p]["runtime_seconds"],
        }
        for p in leaves
    ]
    mass = math.fsum(math.exp(r["logprob"]) for r in rows)
    if abs(mass - 1.0) > 1e-12:
        raise ValueError(f"finite-support mass is not normalized: {mass!r}")
    stats = {
        "mode": mode,
        "rows": len(rows),
        "mass": mass,
        "temperature": temperature,
        "probability_seconds": probability_seconds,
        "head_seconds": head_seconds,
        "total_seconds": time.perf_counter() - started,
        "branch_forwards": calls,
        "likelihood_geometry": "singleton_prefix_plus_PAD"
        if mode == "cpu_exact"
        else "incremental_MATH_SDPA",
        "head_batch_size": head_batch_size if mode == "cpu_exact" else 1,
    }
    if mode == "kv_analysis":
        stats.update(
            kv_forward_calls=engine.forward_calls,
            kv_tokens_processed=engine.tokens_processed,
            cache_peak_bytes=engine.cache.peak_bytes,
            cache_evictions=engine.cache.evictions,
        )
    return rows, stats


def verify_sample(
    policy, context, rows, *, seed=1729, fraction=0.01, temperature=1.0, tolerance=1e-12
):
    if not 0 < fraction <= 1 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(
            "positive sample fraction and finite nonnegative tolerance required"
        )
    selected = random.Random(seed).sample(rows, max(1, math.ceil(len(rows) * fraction)))
    delta = max(
        abs(
            policy.logprobs_many(
                [context], [tuple(r["tokens"])], temperature=temperature
            )[0].total
            - r["logprob"]
        )
        for r in selected
    )
    return {
        "sample_size": len(selected),
        "seed": seed,
        "fraction": fraction,
        "max_logprob_delta": delta,
        "tolerance": tolerance,
        "passed": delta <= tolerance,
        "reference_geometry": "singleton_active_row",
    }


def _ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    lo = 0
    while lo < len(ordered):
        hi = lo + 1
        while hi < len(ordered) and values[ordered[hi]] == values[ordered[lo]]:
            hi += 1
        for i in ordered[lo:hi]:
            ranks[i] = (lo + hi - 1) / 2
        lo = hi
    return ranks


def ranking_report(rows):
    """Qualified comparable labels only; no predicted value becomes a reward.

    Input rows must already join exact context/protocol/evaluator identities.
    Top-8 recall is restricted to the measured support and counts ties
    fractionally at the true-q boundary. Selection bias stays in the receipt.
    """
    groups = defaultdict(list)
    for row in rows:
        if row.get("qualified") is not True:
            continue
        if not all(math.isfinite(row[k]) for k in ("q", "q_hat")):
            raise ValueError("qualified finite measured/predicted q required")
        groups[row["context_id"]].append(row)
    eligible = []
    for context, group in groups.items():
        if len({r["program_sha256"] for r in group}) != len(group):
            raise ValueError("deduplicate comparable programs before ranking")
        if len(group) < 2:
            continue
        x, y = (_ranks([r[k] for r in group]) for k in ("q", "q_hat"))
        xm, ym = sum(x) / len(x), sum(y) / len(y)
        variance = math.sqrt(
            sum((v - xm) ** 2 for v in x) * sum((v - ym) ** 2 for v in y)
        )
        rho = (
            sum((a - xm) * (b - ym) for a, b in zip(x, y)) / variance
            if variance
            else None
        )
        k = min(8, len(group))
        boundary = sorted((r["q"] for r in group), reverse=True)[k - 1]
        above = sum(r["q"] > boundary for r in group)
        tied = sum(r["q"] == boundary for r in group)
        top = sorted(group, key=lambda r: (-r["q_hat"], r["program_sha256"]))[:k]
        recall = (
            sum(
                1.0
                if r["q"] > boundary
                else (k - above) / tied
                if r["q"] == boundary
                else 0.0
                for r in top
            )
            / k
        )
        eligible.append(
            {
                "context_id": context,
                "labels": len(group),
                "spearman": rho,
                "top8_recall_measured_support": recall,
                "top8_informative": len(group) > 8,
            }
        )
    measured = [r for c in eligible for r in groups[c["context_id"]]]
    bins = defaultdict(list)
    ordered = sorted(
        measured, key=lambda r: (r["q_hat"], r["context_id"], r["program_sha256"])
    )
    for i, r in enumerate(ordered):
        bins["decile:" + str(min(9, i * 10 // len(ordered)))].append(r)
        bins[
            "length:"
            + (
                "<128"
                if r["length"] < 128
                else "128-255"
                if r["length"] < 256
                else ">=256"
            )
        ].append(r)
    calibration = {
        key: {
            "n": len(v),
            "q_mean": sum(r["q"] for r in v) / len(v),
            "q_hat_mean": sum(r["q_hat"] for r in v) / len(v),
            "mae": sum(abs(r["q"] - r["q_hat"]) for r in v) / len(v),
        }
        for key, v in bins.items()
    }
    return {
        "contexts": eligible,
        "multi_label_contexts": len(eligible),
        "labels": len(measured),
        "calibration": calibration,
        "scope": "existing qualified measured support; selected executions may bias calibration; full-trie recall is unmeasured",
    }

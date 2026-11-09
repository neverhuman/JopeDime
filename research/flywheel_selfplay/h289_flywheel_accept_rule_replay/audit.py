"""Audit the recorded replay from saved draws and utility matrix; no new experiment."""

from pathlib import Path
import json
import hashlib
import numpy as np
import analysis as a

ROOT = Path(__file__).resolve().parent.parent


def main():
    out = ROOT / "attempt1"
    m = json.loads((out / "metrics-full.json").read_text())
    reg = json.loads((ROOT / "prepared/regression.json").read_text())
    assert reg["status"] == "PASS"
    with np.load(ROOT / "prepared/history.npz") as z:
        rounds = z["rounds"]
        values = z["utilities"]
        families = z["families"]
        invalid = z["invalid"]
    ids, groups, sizes = a.family_layout(families)
    fm = np.stack([values[:, g].mean(axis=1) for g in groups], axis=1)
    cm = np.stack([np.maximum(values[:, g], -1.0).mean(axis=1) for g in groups], axis=1)
    pos = {int(n): i for i, n in enumerate(rounds)}
    with np.load(out / "bootstrap-indices.npz") as z:
        draws = [z[str(s)] for s in a.SEEDS]
    expected, _ = a.make_draws(19)
    for x, y in zip(draws, expected):
        assert np.array_equal(x, y) and x.shape == (10000, 19)
    events = [
        json.loads(x) for x in (out / "replay-events.jsonl").read_text().splitlines()
    ]
    assert len(events) == 3 * 4 * (len(rounds) - 1)
    state = {(seed, rule): 0 for seed in a.SEEDS for rule in a.RULES}
    seen = {k: [] for k in state}
    last = {k: 3175 for k in state}
    for e in events:
        k = (e["seed"], e["rule"])
        i = pos[e["round"]]
        j = state[k]
        assert e["round"] > last[k] and e["previous_round"] == int(rounds[j])
        last[k] = e["round"]
        si = a.SEEDS.index(e["seed"])
        if e["rule"] == "R0":
            point = float(values[i].mean() - values[j].mean())
            accept = point > a.TIE
        else:
            raw = None
            if e["rule"] == "R1":
                delta = fm[i] - fm[j]
            elif e["rule"] == "R2":
                delta = cm[i] - cm[j]
            else:
                # Independent past-only rank reconstruction; no sorted-state mutation reused from replay.
                history = fm[:i]

                def ranks(target):
                    return (
                        (history < target - a.TIE).sum(axis=0)
                        + 0.5
                        * (
                            (history >= target - a.TIE) & (history <= target + a.TIE)
                        ).sum(axis=0)
                    ) / len(history)

                delta = ranks(fm[i]) - ranks(fm[j])
                raw = float(np.median(fm[i] - fm[j]))
                assert raw == e["raw_paired_family_median_gain"]
            point = float(np.median(delta))
            accept = False
            if point > a.MARGIN and (raw is None or raw > a.TIE):
                boot = np.median(delta[draws[si]], axis=1)
                ci = np.quantile(boot, [0.025, 0.975])
                assert np.allclose(ci, e["ci95"], atol=1e-14, rtol=0)
                assert e["bootstrap_draws"] == 10000
                accept = bool(ci[0] > 0)
            else:
                assert e["ci95"] is None and e["bootstrap_draws"] == 0
        assert abs(point - e["point"]) < 1e-14 and accept == e["accepted"]
        if accept:
            seen[k].append(e["round"])
            state[k] = i
    for seed in a.SEEDS:
        si = a.SEEDS.index(seed)
        for rule in a.RULES:
            assert (
                seen[(seed, rule)] == m["replay"][rule]["accepted_rounds_by_seed"][si]
                and int(rounds[state[(seed, rule)]])
                == m["replay"][rule]["terminal_round_by_seed"][si]
            )
    counts = []
    for x in draws:
        c = np.zeros((10000, 19))
        np.add.at(c, (np.arange(10000)[:, None], x), 1)
        counts.append(c)
    for n, record in m["accepted_round_comparisons"].items():
        check = a.comparison(values[pos[int(n)]], values[0], groups, sizes, counts)
        for key, value in check.items():
            assert value == record[key]
    runs = json.loads((out / "replays-full.json").read_text())
    assert (
        a.decide(runs, m["accepted_round_comparisons"], m["window"], len(rounds) - 1)
        == m["decision"]
    )
    window = np.flatnonzero(
        (rounds >= m["window"]["window_first_requested"])
        & (rounds <= m["window"]["window_last_requested"])
    )
    lead = values[0] - values[window].mean(axis=0)
    assert np.isclose(
        lead.mean(), m["window"]["baseline_minus_plateau_point"], atol=1e-15
    )
    for i, c in enumerate(counts):
        sums = np.array([lead[g].sum() for g in groups])
        bs = (c @ sums) / (c @ sizes)
        r = m["window"]["baseline_minus_plateau_bootstrap"]["by_seed"][i]
        assert np.allclose(np.quantile(bs, [0.025, 0.975]), r["ci95"], atol=1e-14)
    centered = values[window] - values[0]
    centered -= centered.mean(axis=1)[:, None]
    gs = np.stack([centered[:, g].sum(axis=1) for g in groups], axis=0)
    for i, c in enumerate(counts):
        maxima = np.max((c @ gs) / (c @ sizes)[:, None], axis=1)
        null = m["window"]["best_of_N_centered_null"][str(len(window))]["by_seed"][i]
        assert np.allclose(
            np.quantile(maxima, [0.025, 0.975]), null["ci95"], atol=1e-14
        )
        ci = m["window"]["baseline_minus_plateau_bootstrap"]["by_seed"][i]["ci95"]
        within = bool(ci[0] <= 0 <= ci[1] and lead.mean() <= null["ci95"][1])
        assert within == m["window"]["within_noise_by_seed"][i]
    attrs = json.loads((out / "round-tail-attribution.json").read_text())
    for r in attrs:
        i = pos[r["round"]]
        delta = values[0] - values[i]
        mask = invalid[0] | invalid[i]
        assert np.isclose(
            np.where(mask, delta, 0.0).mean(),
            r["invalid_tail_contribution"],
            atol=1e-14,
        )
        assert np.isclose(
            np.where(~mask, delta, 0.0).mean(), r["both_valid_contribution"], atol=1e-14
        )
        assert (
            abs(
                r["lead"]
                - r["invalid_tail_contribution"]
                - r["both_valid_contribution"]
            )
            < 1e-14
        )
    assert np.isclose(
        np.mean([r["lead"] for r in attrs]), m["attribution"]["mean_lead"]
    )
    for name, digest in json.loads((ROOT / "source/manifest.json").read_text())[
        "source_sha256"
    ].items():
        assert (
            hashlib.sha256((ROOT / "source" / name).read_bytes()).hexdigest() == digest
        )
    receipt = dict(
        status="PASS",
        events_verified=len(events),
        saved_family_draws_verified=3,
        draws_per_seed=10000,
        chronological_incumbents_verified=True,
        rank_transform_independently_reconstructed_past_only=True,
        accepted_round_raw_qualification_recomputed=True,
        baseline_and_tail_decomposition_verified=True,
        neural_forwards=0,
        neural_updates=0,
        metrics_sha256=hashlib.sha256(
            (out / "metrics-full.json").read_bytes()
        ).hexdigest(),
    )
    (out / "audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

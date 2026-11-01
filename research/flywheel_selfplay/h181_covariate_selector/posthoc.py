"""Closed-wave TRAIN-only step-40 diagnostics. Never re-adjudicate a closed wave."""

import json
from pathlib import Path
import numpy as np
import science as s


def paths():
    files = []
    for wave, seed0, base in [
        ("H170", 170101, Path("/home/ubuntu/jope-data/h170/attempt-001")),
        ("H171", 171101, Path("/home/ubuntu/jope_plan/evidence/H171/run")),
    ]:
        files.extend(
            (wave, seed, outer, base / f"A{outer}-{seed}.json")
            for seed in range(seed0, seed0 + 3)
            for outer in range(5)
        )
    return files


def run(rows, X, out):
    ids = [r["context_sha256"] for r in rows]
    lookup = {c: i for i, c in enumerate(ids)}
    panels = {}
    sources = []
    for wave, seed, outer, path in paths():
        record = json.loads(path.read_text())
        sources.append(str(path))
        for arm, steps in record["selector_panels"].items():
            # H171 CTRL_EX4 records logical steps; logical10 is actual40.
            pick = 1 if wave == "H171" and arm == "CTRL_EX4" else 4
            if len(steps) != 5:
                raise ValueError("expected steps0/10/20/30/40")
            key = (wave, arm)
            panels.setdefault(key, {})
            values = panels[key].setdefault(seed, np.full(len(rows), np.nan))
            base = steps[0]
            chosen = steps[pick]
            if base["context_ids"] != chosen["context_ids"]:
                raise ValueError("context alignment")
            for cid, b, y in zip(
                chosen["context_ids"], base["values"], chosen["values"], strict=True
            ):
                i = lookup[cid]
                if np.isfinite(values[i]):
                    raise ValueError("duplicate B context")
                values[i] = y - b
    path = Path("/home/ubuntu/jope-data/h177/attempt-002/panels.npz")
    sources.append(str(path))
    with np.load(path, allow_pickle=False) as z:
        panel = z["pooled_b"].copy()
    if panel.shape != (3, len(rows), 5) or not np.isfinite(panel).all():
        raise ValueError("H177 panel shape")
    for j, arm in enumerate(["CTRL", "ACC4", "ACC16", "OC15"], start=1):
        panels["H177", arm] = {
            177101 + i: panel[i, :, j] - panel[i, :, 0] for i in range(3)
        }
    weights = s.family_weights()
    group = s.groups()
    result = []
    raw = {}
    # All diagnostics use the frozen full-slate X; it is seed-invariant.
    if not np.array_equal(X[0], X[1]) or not np.array_equal(X[0], X[2]):
        raise ValueError("full X seed invariance")
    for (wave, arm), byseed in sorted(panels.items()):
        contexts = np.stack([byseed[k] for k in sorted(byseed)])
        if not np.isfinite(contexts).all():
            raise ValueError("incomplete closed-wave TRAIN coverage")
        Y = s.family_mean(contexts, rows)
        adjusted = []
        theta = []
        for i in range(3):
            allmethods, co, _ = s.adjusted_panel(X[i], Y[i : i + 1], group)
            adjusted.append(allmethods[2, 0])
            theta.append(co[1, :, 0])
        A = np.stack(adjusted)
        raw[wave + "_" + arm + "_gain"] = Y
        raw[wave + "_" + arm + "_CUPED_K"] = A
        raw[wave + "_" + arm + "_theta"] = np.stack(theta)
        result.append(
            dict(
                wave=wave,
                arm=arm,
                actual_updates=40,
                source_seeds=sorted(byseed),
                STRICT=s.summarize(Y, weights),
                CUPED_K=s.summarize(A, weights),
                bias_vs_STRICT=s.summarize(A - Y, weights),
                scope="POSTHOC DESCRIPTIVE ONLY; correlated reused TRAIN panels; no selector decision or re-verdict; H171 CTRL_EX4 logical10=actual40",
            )
        )
    np.savez_compressed(out / "posthoc.npz", **raw)
    return dict(
        scope="NO_REVERDICT; posthoc frozen closed-wave TRAIN-only step40; frozen full-slate r882 covariates, FIT123 exclusions and fold181140; no heldout outcomes or network updates",
        panels=result,
        sources=sources,
    )

"""Family-grouped trimmed objective; no outer-test information enters selection."""

import hashlib
import json
import numpy as np

TIE = 1e-4


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def family_profile(picks, cids, packets, family_by_cid):
    grouped = {}
    for c, j in zip(cids, picks):
        u = packets[c]["utility_k"]
        grouped.setdefault(family_by_cid[c], []).append(float(u[int(j)]) - float(u[0]))
    names = sorted(grouped)
    values = np.array([np.mean(grouped[f]) for f in names])
    assert len(names) > 5
    order = np.argsort(-values, kind="stable")
    return dict(
        families=names,
        gains=values.tolist(),
        trimmed_gain=float(np.delete(values, order[:5]).mean()),
        family_median=float(np.median(values)),
        dropped_families=[names[i] for i in order[:5]],
        picks=list(map(int, picks)),
        cids=list(cids),
    )


def choose(table):
    high = max(row["trimmed_gain"] for row in table)
    tied = [row for row in table if row["trimmed_gain"] >= high - TIE]
    median = max(row["family_median"] for row in tied)
    finalists = [row for row in tied if row["family_median"] >= median - TIE]
    chosen = min(
        finalists,
        key=lambda row: (row["inner_switches"], -row["k"], -row["a"], row["tau"]),
    )
    return chosen, len(tied), len(finalists)


def decide(paired, robust, baseline, config_agreement):
    checks = dict(
        trimmed_median_gt_001=paired["drop_top5"]["median"] > 0.001,
        trimmed_pooled_lcb95_gt_0=paired["drop_top5"]["pooled"]["lcb95"] > 0,
        mean_noninferior=paired["mean"]["median"] >= -0.005,
        drop_top5_lcb95_pos_3of3=all(
            v["drop_top5_lcb95_one_sided"] > 0 for v in robust["concentration"]
        ),
        harmed_within_one_pp=all(
            r <= h + 0.01
            for r, h in zip(
                robust["summary"]["harmed_share_by_seed"],
                baseline["summary"]["harmed_share_by_seed"],
            )
        ),
        invalid_excess_not_above_h303=all(
            r["excess"] <= h["excess"]
            for r, h in zip(robust["invalid_detail"], baseline["invalid_detail"])
        ),
    )
    verdict = (
        "ROBUSTSEL_NO_CHANGE"
        if config_agreement == 12
        else ("ROBUSTSEL_ADVANCE" if all(checks.values()) else "ROBUSTSEL_NOT_ADVANCED")
    )
    return dict(
        verdict=verdict,
        criteria=checks,
        failed=[k for k, v in checks.items() if not v],
        configs_equal_h303=config_agreement,
    )

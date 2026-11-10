"""Run only after H303's frozen closeout: independent replay and diagnostics."""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from independent_replay import replay303, SEEDS
from statistics import summarize
from audit_structure import chronology


def run(scratch, source, published_metrics, closeout):
    assert closeout.get("closed") is True and len(closeout["commit"]) == 40
    with np.load(scratch / "train.npz", allow_pickle=False) as data:
        picks, locks = replay303(scratch / "h303", data)
        statistics = summarize(data["utility"], picks, data["families"], data["gates"])
    np.save(scratch / "h303-picks.npy", picks)
    hashes = {
        str(seed): hashlib.sha256(
            np.asarray(picks[i], dtype="<i4").tobytes()
        ).hexdigest()
        for i, seed in enumerate(SEEDS)
    }
    expected = published_metrics["nested_hashes"]
    published_gains = published_metrics["arms"]["ENS_VOTE2_NESTED"]["summary"][
        "per_seed_family_mean_gain"
    ]
    match = hashes == expected and np.allclose(
        statistics["gains"], published_gains, rtol=0, atol=1e-14
    )
    lock_hashes = {f"{x['outer']}:{x['root']}": x["fit_lock_sha256"] for x in locks}
    assert lock_hashes == published_metrics["fit_lock_sha256"]
    regression = dict(
        status="PASS" if match else "H303_REPRO_FAIL",
        picks_sha256=hashes,
        expected_picks_sha256=expected,
        gains=statistics["gains"],
        published_gains=published_gains,
        locks=locks,
        closeout=closeout,
    )
    (source / "regression_h303.json").write_text(
        json.dumps(regression, indent=2) + "\n"
    )
    (source / "statistics_h303.json").write_text(
        json.dumps(statistics, indent=2, allow_nan=False) + "\n"
    )
    inv = json.loads((source.parent / "h303-archive-inventory.json").read_text())
    # H303's archive can also include the earlier H302 regression locks.
    nested_inventory = [
        x for x in inv if "/root_" in x["name"] and x["name"].endswith("/fit_lock.json")
    ]
    assert len(nested_inventory) == 12
    timing = chronology(nested_inventory, "43e00dd31f")
    (source / "chronology_h303.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                regression=regression["status"],
                gains=statistics["gains"],
                p=statistics["permutation"]["p_upper_plus_one"],
            )
        )
    )
    assert match


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--closeout", type=Path, required=True)
    a = ap.parse_args()
    run(
        a.scratch,
        a.source,
        json.loads(a.metrics.read_text()),
        json.loads(a.closeout.read_text()),
    )

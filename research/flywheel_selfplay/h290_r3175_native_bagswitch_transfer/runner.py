"""One frozen scoring attempt, only after the prereg is published."""

from dependencies import ROOT, core, bagged, veto, freeze_bagged, forbidden
import statistics_h290 as stat
import json
import hashlib
import os
import sys
import time
import numpy as np
import lightgbm as lgb


def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def write(f, v):
    with f.open("x") as s:
        json.dump(v, s, indent=2, allow_nan=False)
        s.write("\n")
        s.flush()
        os.fsync(s.fileno())


def protected(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = os.fsdecode(args[0]).lower()
        if any(
            s in path
            for s in [
                "/sealed",
                "/holdout",
                "/dl-test",
                "/dl_test",
                "fresh_cohort_v3",
                "/h288/",
                "/h285/",
                "qualification.lock",
            ]
        ) or path.endswith((".pt", ".jtt1", ".jtt1.gz")):
            raise RuntimeError("Forbidden H290 scoring input")


def placebo(cell, X, n):
    root = ROOT / "fit/models"
    limit = cell["selected_n_estimators"]
    margin = cell["raw_margin"]

    def pred(info, limit=None):
        f = root / info["file"]
        assert sha(f) == info["sha256"]
        return np.asarray(
            lgb.Booster(model_file=str(f)).predict(
                X, num_iteration=limit, num_threads=1
            )
        ).reshape(n, 8)

    gate = pred(cell["veto"])
    scores = veto.mask(pred(cell["placebo"], limit), gate)
    proposed = [core.select_from_top4_scores(x, margin) for x in scores]
    bags = np.stack([pred(info, limit) for info in cell["placebo_bags"]])
    votes = bagged.vote_counts(bags, proposed, margin)
    return bagged.switch_by_votes(proposed, votes, 6), votes.tolist()


def null_picks(gate, seed, cohort, arm):
    rng = np.random.default_rng(core.stream_seed(seed, "H290", cohort, arm))
    picks = []
    for row in gate:
        eligible = np.flatnonzero(row) if arm == "NULL_RV" else np.arange(8)
        picks.append(int(rng.choice(eligible)) if len(eligible) else 0)
    return picks


def main():
    sys.addaudithook(protected)
    lgb.LGBMRanker.fit = forbidden
    lgb.LGBMClassifier.fit = forbidden
    lgb.train = forbidden
    pub = json.loads((ROOT / "prereg/published.json").read_text())
    m = json.loads((ROOT / "source/manifest.json").read_text())
    for name, digest in m["source_sha256"].items():
        assert sha(ROOT / "source" / name) == digest, name
    fit = json.loads((ROOT / "fit/fit_lock.json").read_text())
    assert sha(ROOT / "fit/fit_lock.json") == m["fit_lock_sha256"]
    for name in ["fit.py", "dependencies.py", "monitor.py", "custody.py"]:
        assert sha(ROOT / "source" / name) == fit["source_sha256"][name]
    reference = json.loads((ROOT / "code-closure/export_manifest.json").read_text())
    out = ROOT / "attempt1"
    out.mkdir()
    write(
        out / "launch.json",
        dict(
            attempt=1,
            source_commit=pub["source_commit"],
            manifest_sha256=sha(ROOT / "source/manifest.json"),
            fit_lock_sha256=m["fit_lock_sha256"],
            epoch=time.time(),
            neural_forwards=0,
            neural_updates=0,
        ),
    )
    summaries = {}
    for cohort in ["H286", "H283"]:
        pin = next(x for x in m["preparation"]["cohorts"] if x["wave"] == cohort)
        inp = ROOT / "prepared" / pin["file"]
        assert sha(inp) == pin["sha256"]
        with np.load(inp) as z:
            X = z["X"].reshape(-1, 568)
            ids = z["cids"]
            families = z["families"]
            gate = z["gate"]
        n = len(ids)
        picks = np.zeros((3, n, 6), dtype=int)
        votes = []
        for i, cell in enumerate(fit["cells"]):
            assert cell["seed"] == stat.SEEDS[i]
            native = freeze_bagged.score_frozen_bagged_cell(
                ROOT / "fit/models", cell, X, n
            )
            old = freeze_bagged.score_frozen_bagged_cell(
                ROOT / "reference", reference["cells"][i], X, n
            )
            shuf, svotes = placebo(cell, X, n)
            picks[i, :, 1] = native["BAG_SWITCH"]
            picks[i, :, 2] = old["BAG_SWITCH"]
            picks[i, :, 3] = shuf
            for j, arm in [(4, "NULL_RV"), (5, "NULL_LF")]:
                picks[i, :, j] = null_picks(gate, stat.SEEDS[i], cohort, arm)
            votes.append(
                dict(
                    seed=cell["seed"],
                    native=native["votes"],
                    reference=old["votes"],
                    placebo=svotes,
                )
            )
        write(
            out / (cohort + "-picks.json"),
            dict(
                cids=ids.tolist(),
                arms=list(stat.ARMS),
                picks=picks.tolist(),
                votes=votes,
                learned_picks_use_features_only=True,
            ),
        )
        with np.load(inp) as z:
            u = z["utility"]
        values = np.stack([u[np.arange(n)[:, None], pick] for pick in picks])
        invalid = np.stack([~gate[np.arange(n)[:, None], pick] for pick in picks])
        summary, fm, draws = stat.summarize(values, invalid, families)
        np.savez_compressed(
            out / (cohort + "-panels.npz"),
            utility=values,
            invalid=invalid,
            picks=picks,
            families=families,
            cids=ids,
            family_utilities=fm,
            bootstrap_indices=draws,
        )
        summary["coverage"] = pin
        write(out / (cohort + "-metrics.json"), summary)
        summaries[cohort] = summary
        print(cohort + " SCORED_AND_SUMMARIZED", flush=True)
    decision = stat.verdict(summaries["H286"])
    result = dict(
        schema="jope.h290.metrics/v1",
        wave="H290",
        status="COMPLETE",
        verdict=decision["verdict"],
        source_commit=pub["source_commit"],
        decision=decision,
        scope="SEMI_FRESH, previously opened cohorts; shadow only",
        primary=summaries["H286"],
        secondary=summaries["H283"],
        fit_lock_sha256=m["fit_lock_sha256"],
        seeds=list(stat.SEEDS),
        reference_seeds=[279101, 279102, 279103],
        bootstrap_seed=290150,
        bootstrap_draws=10000,
        regression=m["regression"],
        cpu_gpu_drift=m["preparation"]["paired_cpu_gpu_drift"],
        neural_forwards=0,
        neural_updates=0,
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    write(out / "metrics-full.json", result)
    print(
        json.dumps(
            dict(
                status="COMPLETE",
                verdict=result["verdict"],
                native=result["primary"]["arms"]["NATIVE"]["gain"],
                contrast=result["primary"]["native_minus_r882_fit"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

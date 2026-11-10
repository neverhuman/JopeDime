"""Pinned-input regression, then SELECT locks and retrospective evaluation."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import reselection as a
from fetch_inputs import INPUTS

IDENTITY = "e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83"
PREREG = "0cbc429eb7f2e8092fe6146f7ea01633fb613def"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    raw = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with path.open("x") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return hashlib.sha256(raw.encode()).hexdigest()


def install_guard():
    def audit(event, args):
        if (
            event == "import"
            and args
            and str(args[0]).split(".")[0] in {"torch", "tensorflow", "jax"}
        ):
            raise RuntimeError("No neural frameworks permitted")
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0]).lower()
            if any(
                s in p
                for s in (
                    "/sealed",
                    "/holdout",
                    "fresh_cohort",
                    "fresh_pools",
                    "/dl-test",
                    "/dl_test",
                    "meta-test",
                    "meta_test",
                    "flywheel-gp-fused",
                    "flywheel-full-20261030",
                )
            ) or p.endswith((".pt", ".jtt1", ".jtt1.gz")):
                raise RuntimeError("Forbidden input: " + p)

    sys.addaudithook(audit)


def load_history(scratch):
    for archive, version, digest, member, member_sha in INPUTS:
        assert sha(scratch / member) == member_sha
    with np.load(scratch / "history.npz", allow_pickle=False) as z:
        rounds = z["rounds"].copy()
        utility = z["utilities"].copy()
        context_families = z["families"].astype(str)
    assert utility.shape == (1500, 91) and np.isfinite(utility).all()
    assert np.array_equal(rounds, np.array([r for r in range(3175, 4676) if r != 4331]))
    journal = {}
    for line in (scratch / "adapter_admissions.jsonl").read_text().splitlines():
        row = json.loads(line)
        r = int(row["round"])
        assert r not in journal or journal[r] == row
        journal[r] = row
    for r, u in zip(rounds, utility):
        row = journal[int(r)]
        assert row["evaluation_identity"] == IDENTITY
        assert abs(float(u.mean()) - row["candidate_utility"]) <= 1e-15
    families, matrix = a.family_matrix(utility, context_families)
    assert len(families) == 19
    return rounds, utility, context_families, families, matrix


def r0_replay(rounds, select_values, eval_values):
    selected = select_values.mean(1)
    evaluated = eval_values.mean(1)
    current = 0
    trajectory, promotions = [], []
    for i in range(len(rounds)):
        if selected[i] - selected[current] > a.TIE:
            promotions.append((i, current))
            current = i
        trajectory.append(current)
    n = len(promotions)
    return dict(
        n_promotions=n,
        final_incumbent_round=int(rounds[current]),
        promotion_precision=float(
            sum(evaluated[i] - evaluated[j] > a.TIE for i, j in promotions) / n
        )
        if n
        else 0.0,
        init_select_utility=float(selected[0]),
        final_select_utility=float(selected[current]),
        init_eval_family_equal_utility=float(evaluated[0]),
        final_eval_family_equal_utility=float(evaluated[current]),
        time_avg_eval_family_equal_utility=float(evaluated[trajectory].mean()),
        net_shrinkage_vs_init=float(
            selected[current] - selected[0] - evaluated[current] + evaluated[0]
        ),
    )


def regression(scratch, output):
    rounds, utility, context_families, families, matrix = load_history(scratch)
    assert abs(float(utility[0].mean()) - 1.1389572403748425) <= 1e-15
    assert abs(float(utility[1:].mean(1).mean()) - 1.1320995298853955) <= 1e-15
    assert (
        abs(float(np.quantile(utility[1:].mean(1), 0.95)) - 1.1354387884248622) <= 1e-15
    )
    reference = json.loads((scratch / "h304_reference.json").read_text())
    assert (
        sha(scratch / "h304_reference.json")
        == "60c5f879437bcfe5310b8e6e58cf652f81cd0c5f084dc138468950d0596cd84a"
    )
    select, evaluation = a.split_indices(families, 30401)
    cases = {}
    for label, values, key in [
        ("family_equal", matrix[:, select], "primary_r3175_init_family_equal_select"),
        (
            "context_mean",
            utility[:, np.isin(context_families, families[select])],
            "sensitivity_r3175_init_context_mean_select",
        ),
    ]:
        result = r0_replay(rounds, values, matrix[:, evaluation])
        expected = reference[key]["per_seed"]["30401"]["rules"]["R0_BASELINE"]
        for field, value in result.items():
            assert abs(value - expected[field]) <= 1e-12, (
                label,
                field,
                value,
                expected[field],
            )
        cases[label] = dict(
            observed=result,
            expected={k: expected[k] for k in result},
            max_abs_error=max(abs(v - expected[k]) for k, v in result.items()),
        )
    result = dict(
        status="PASS",
        prereg_commit=PREREG,
        reference_close_commit="3dccbc745ad48e77835c41380aa07b1f7acb9cb4",
        r3175_context_mean=float(utility[0].mean()),
        journal_cross_verified_rounds=len(rounds),
        history_sha256=sha(scratch / "history.npz"),
        admissions_sha256=sha(scratch / "adapter_admissions.jsonl"),
        evaluation_identity=IDENTITY,
        seed=30401,
        select_families=len(select),
        eval_families=len(evaluation),
        cases=cases,
        time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    save(output, result)
    print(json.dumps(result), flush=True)


def run(scratch, source):
    reg = json.loads((source / "regression_receipt.json").read_text())
    assert reg["status"] == "PASS" and reg["prereg_commit"] == PREREG
    rounds, utility, context_families, families, matrix = load_history(scratch)
    assert sha(scratch / "history.npz") == reg["history_sha256"]
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_lock = dict(
        prereg_commit=PREREG,
        started=start,
        source_sha256={f.name: sha(f) for f in source.glob("*.py")},
        regression_sha256=sha(source / "regression_receipt.json"),
    )
    save(scratch / "execution_lock.json", run_lock)
    splits = [a.split_indices(families, seed) for seed in a.SEEDS]
    all_picks, per_seed = [], {}
    best = None
    for seed, (select, evaluation) in zip(a.SEEDS, splits):
        picks, scores, eb = a.select_candidates(
            matrix[:, select], rounds, 304200 + seed
        )
        scorefile = scratch / f"scores_{seed}.npz"
        np.savez_compressed(
            scorefile,
            scores=scores,
            rounds=rounds,
            select=select,
            evaluation=evaluation,
        )
        lock = dict(
            seed=seed,
            select_families=families[select].tolist(),
            eval_families=families[evaluation].tolist(),
            picked_indices=picks.tolist(),
            picked_rounds=rounds[picks].tolist(),
            eb=eb,
            scores_sha256=sha(scorefile),
            eval_computed=False,
        )
        lock_sha = save(scratch / f"picks_{seed}.json", lock)
        results = a.eval_split(matrix, picks, select, evaluation)
        if seed == 30401:
            rule_gains = np.array(
                [results[name]["gain_vs_r3175"] for name in a.RULES[1:]]
            )
            best = a.RULES[1 + a.earliest_max(rule_gains, np.arange(4))]
            save(
                scratch / "rule_choice_lock.json",
                dict(
                    seed=30401,
                    rule=best,
                    gains=rule_gains.tolist(),
                    pick_lock_sha256=lock_sha,
                    other_primary_eval_computed=False,
                ),
            )
        per_seed[str(seed)] = dict(
            n_select_families=len(select),
            n_eval_families=len(evaluation),
            select_families=families[select].tolist(),
            eval_families=families[evaluation].tolist(),
            picked_rounds=dict(zip(a.RULES, map(int, rounds[picks]))),
            eb=eb,
            pick_lock_sha256=lock_sha,
            rules=results,
        )
        all_picks.append(picks)
        print(
            json.dumps(
                dict(
                    primary_seed=seed,
                    picked_rounds=rounds[picks].tolist(),
                    gains=[results[n]["gain_vs_r3175"] for n in a.RULES],
                )
            ),
            flush=True,
        )
    primary, bootstrap = a.summarize_primary(matrix, all_picks, splits, per_seed)
    verdict = a.decision(best, primary)
    repeated = []
    for j, (select, evaluation) in enumerate(a.balanced_splits()):
        picks, scores, eb = a.select_candidates(matrix[:, select], rounds, 307200 + j)
        # Every sensitivity split also gets an fsynced SELECT-only lock before EVAL.
        lock_sha = save(
            scratch / f"sensitivity_picks_{j:03d}.json",
            dict(
                index=j,
                select=select.tolist(),
                evaluation=evaluation.tolist(),
                picked_indices=picks.tolist(),
                picked_rounds=rounds[picks].tolist(),
                eb=eb,
                scores_sha256=hashlib.sha256(
                    scores.astype("<f8").tobytes()
                ).hexdigest(),
                eval_computed=False,
            ),
        )
        levels = matrix[picks][:, evaluation].mean(1)
        baseline = float(matrix[0, evaluation].mean())
        gains = levels - baseline
        vs_s0 = levels - levels[0]
        sg = matrix[picks][:, select].mean(1) - matrix[0, select].mean()
        repeated.append(
            dict(
                split=j,
                select=select.tolist(),
                evaluation=evaluation.tolist(),
                pick_lock_sha256=lock_sha,
                picked_rounds=rounds[picks].tolist(),
                eval_levels=levels.tolist(),
                baseline_level=baseline,
                gains_vs_r3175=gains.tolist(),
                gains_vs_s0=vs_s0.tolist(),
                shrinkage_vs_r3175=(sg - gains).tolist(),
                shrinkage_vs_s0=(sg - sg[0] - vs_s0).tolist(),
                eb_tau2=eb["tau2"],
            )
        )
        if (j + 1) % 25 == 0:
            print("sensitivity", j + 1, flush=True)
    save(scratch / "sensitivity_full.json", repeated)
    sensitivity = {}
    for i, name in enumerate(a.RULES):
        g = np.array([v["gains_vs_r3175"][i] for v in repeated])
        s0 = np.array([v["gains_vs_s0"][i] for v in repeated])
        picked = [v["picked_rounds"][i] for v in repeated]
        histogram = {str(r): picked.count(r) for r in sorted(set(picked))}
        sensitivity[name] = dict(
            mean_eval_level=float(np.mean([v["eval_levels"][i] for v in repeated])),
            median_eval_level=float(np.median([v["eval_levels"][i] for v in repeated])),
            mean_gain_vs_r3175=float(g.mean()),
            median_gain_vs_r3175=float(np.median(g)),
            mean_gain_vs_s0=float(s0.mean()),
            median_gain_vs_s0=float(np.median(s0)),
            splits_wins_ties_losses_vs_r3175=[
                int((g > a.TIE).sum()),
                int((np.abs(g) <= a.TIE).sum()),
                int((g < -a.TIE).sum()),
            ],
            splits_wins_ties_losses_vs_s0=[
                int((s0 > a.TIE).sum()),
                int((np.abs(s0) <= a.TIE).sum()),
                int((s0 < -a.TIE).sum()),
            ],
            mean_shrinkage_vs_r3175=float(
                np.mean([v["shrinkage_vs_r3175"][i] for v in repeated])
            ),
            mean_shrinkage_vs_s0=float(
                np.mean([v["shrinkage_vs_s0"][i] for v in repeated])
            ),
            picked_round_histogram=histogram,
            unique_rounds=len(histogram),
            modal_pick_share=max(histogram.values()) / 200,
            primary_picks=rounds[np.array(all_picks)[:, i]].tolist(),
        )
    metrics = dict(
        schema="jope.h307.metrics/v1",
        wave_id="H307",
        owner="rd_astra",
        status="COMPLETE",
        verdict=verdict["verdict"],
        prereg_commit=PREREG,
        execution_lock_sha256=sha(scratch / "execution_lock.json"),
        evaluation_identity=IDENTITY,
        n_rounds=1500,
        n_contexts=91,
        n_families=19,
        split_seeds=list(a.SEEDS),
        bootstrap_seed=304150,
        bootstrap_draws=10000,
        select_bootstrap_draws=1000,
        regression_status="PASS",
        optimizer_updates=0,
        neural_forwards=0,
        gpu_used=False,
        fresh_pools_read=0,
        dope_untouched=True,
        per_seed=per_seed,
        primary=primary,
        median_bootstrap=bootstrap,
        decision=verdict,
        sensitivity_200=sensitivity,
        baseline_context_mean=float(utility[0].mean()),
        baseline_family_equal_mean=float(matrix[0].mean()),
        generalization_adjusted_plateau=dict(
            rule=best,
            median_eval_level=primary[best]["median_eval_level"],
            interval=primary[best]["median_eval_level_interval"],
            interpretation="Retrospective historically exposed dev91 level; not external generalization",
        ),
        limitations=[
            "19 shared dev families; split seeds are not training seeds",
            "Seed30401 EVAL chooses the non-S0 rule and is included in the decision",
            "Historical candidates and incumbent already saw full dev91",
            "Conditional fixed-pick bootstrap does not remove candidate or rule-selection bias",
            "Family Elo with 0.01 decisive pseudocount; not H304 context-count Elo",
        ],
        started=start,
        finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    save(source / "metrics.json", metrics)
    assert (source / "metrics.json").stat().st_size < 50_000
    print(
        json.dumps(dict(decision=verdict, primary=primary, sensitivity=sensitivity)),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["regression", "run"])
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    install_guard()
    source = Path(__file__).parent
    if args.phase == "regression":
        try:
            regression(args.scratch, source / "regression_receipt.json")
        except Exception as error:
            save(
                source / "regression_receipt.json",
                dict(status="REGRESSION_FAIL", error=repr(error)),
            )
            raise
    else:
        run(args.scratch, source)

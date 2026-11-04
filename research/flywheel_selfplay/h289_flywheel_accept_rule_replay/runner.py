"""Single recorded-artifact analysis attempt; no checkpoints, neural frameworks or forwards."""

from pathlib import Path
import os
import sys
import json
import hashlib
import time
import numpy as np
import analysis as a

ROOT = Path(__file__).resolve().parent.parent


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protect(event, args):
    if event == "import" and args[0].split(".")[0] in ["torch", "tensorflow", "jax"]:
        raise RuntimeError("Neural framework forbidden in recorded replay")
    if event == "open" and isinstance(args[0], (str, bytes)):
        s = os.fsdecode(args[0]).lower()
        if any(
            x in s
            for x in [
                "fresh_cohort",
                "fresh_pools",
                "/sealed",
                "/holdout",
                "/dl-test",
                "/dl_test",
                "meta_test",
                "meta-test",
            ]
        ) or s.endswith((".pt", ".jtt1", ".jtt1.gz")):
            raise RuntimeError("forbidden input " + s)


def main():
    sys.addaudithook(protect)
    published = json.loads((ROOT / "prereg/published.json").read_text())
    source_commit = published["source_commit"]
    m = json.loads((ROOT / "source/manifest.json").read_text())
    reg = json.loads((ROOT / "prepared/regression.json").read_text())
    assert reg["status"] == "PASS" and reg["r3175_mean"] == 1.1389572403748425
    for name, digest in m["source_sha256"].items():
        assert sha(ROOT / "source" / name) == digest
    assert sha(ROOT / "prepared/history.npz") == reg["history_sha256"]
    out = ROOT / "attempt1"
    out.mkdir()
    write(
        out / "launch.json",
        dict(
            source_commit=source_commit,
            epoch=time.time(),
            manifest_sha256=sha(ROOT / "source/manifest.json"),
            protocol_sha256=sha(ROOT / "source/PROTOCOL.md"),
            attempt=1,
            seeds=list(a.SEEDS),
            draws=a.DRAWS,
            neural_updates=0,
            neural_forwards=0,
            source_input=reg["history_sha256"],
        ),
    )
    with np.load(ROOT / "prepared/history.npz", allow_pickle=False) as z:
        rounds = z["rounds"]
        values = z["utilities"]
        invalid = z["invalid"]
        families = z["families"]
        cids = z["cids"]
        programs = z["programs"]
    ids, groups, sizes = a.family_layout(families)
    draws, counts = a.make_draws(len(ids))
    np.savez_compressed(
        out / "bootstrap-indices.npz", **{str(s): d for s, d in zip(a.SEEDS, draws)}
    )
    with (out / "replay-events.jsonl").open("x") as f:
        runs = a.replay(
            values,
            rounds,
            families,
            draws,
            emit=lambda e: f.write(json.dumps(e, separators=(",", ":")) + "\n"),
        )
    write(out / "replays-full.json", runs)
    print("REPLAY complete", flush=True)
    winner, per_round, window = a.winner_analysis(
        values, rounds, groups, sizes, draws, counts, int(m["latest_complete_round"])
    )
    write(out / "last200-comparisons.json", per_round)
    accepted = sorted(
        {
            e["round"]
            for r in runs
            for rule in a.RULES
            for e in r["rules"][rule]["accepted"]
        }
    )
    position = {int(r): i for i, r in enumerate(rounds)}
    comparisons = {}
    for n in accepted:
        i = position[n]
        record = a.comparison(values[i], values[0], groups, sizes, counts)
        record.update(
            round=n,
            invalid_base=int(invalid[0].sum()),
            invalid_candidate=int(invalid[i].sum()),
            changed_programs=int((programs[i] != programs[0]).sum()),
            accepted_by=[
                dict(seed=r["seed"], rule=rule)
                for r in runs
                for rule in a.RULES
                if any(e["round"] == n for e in r["rules"][rule]["accepted"])
            ],
        )
        comparisons[str(n)] = record
    write(out / "accepted-comparisons.json", comparisons)
    attr, at_rows, contexts = a.attribution(
        values, invalid, rounds, cids, families, window
    )
    write(out / "round-tail-attribution.json", at_rows)
    write(out / "context-tail-attribution.json", contexts)
    decision = a.decide(runs, comparisons, winner, len(rounds) - 1)
    history = json.loads((ROOT / "prepared/round-metadata.json").read_text())
    literal = []
    current = float(values[0].mean())
    for n, v in zip(rounds[1:], values[1:]):
        if v.mean() > current:
            literal.append(int(n))
            current = float(v.mean())
    recorded = [
        r["round"] for r in history if r["round"] > 3175 and r["recorded_accepted"]
    ]
    policy_summaries = {}
    for rule in a.RULES:
        counts_accepted = [r["rules"][rule]["accept_count"] for r in runs]
        terminal = [r["rules"][rule]["terminal_round"] for r in runs]
        terminal_comparisons = [
            a.comparison(values[position[n]], values[0], groups, sizes, counts)
            for n in terminal
        ]
        policy_summaries[rule] = dict(
            accept_count_by_seed=counts_accepted,
            median_accept_count=float(np.median(counts_accepted)),
            accepted_rounds_by_seed=[
                [e["round"] for e in r["rules"][rule]["accepted"]] for r in runs
            ],
            terminal_round_by_seed=terminal,
            terminal_context_gain=[
                x["context_mean_gain"] for x in terminal_comparisons
            ],
            terminal_elo_base1500=[
                x["elo_base1500"]["point"] for x in terminal_comparisons
            ],
            terminal_raw_gain_ci95_by_seed=[
                terminal_comparisons[i]["bootstrap"]["by_seed"][i]["ci95"]
                for i in range(3)
            ],
        )
    metrics = dict(
        schema="jope.h289.metrics/v1",
        wave="H289",
        source_commit=source_commit,
        status="COMPLETE",
        verdict=decision["verdict"],
        decision=decision,
        seeds=list(a.SEEDS),
        bootstrap_draws=a.DRAWS,
        scope="reused DL-Val91; shadow-only recorded-path replay, not counterfactual optimization",
        regression=reg,
        records=len(rounds) - 1,
        window=winner,
        replay=policy_summaries,
        accepted_round_comparisons=comparisons,
        attribution=attr,
        current_rule_audit=dict(
            literal_mean_gt_best_accepts=literal,
            journal_accepts=recorded,
            match=literal == recorded,
            R0_owner_tie_tolerance=a.TIE,
        ),
        neural_updates=0,
        neural_forwards=0,
        DOPE="untouched; recorded identity c51a56d0 verified",
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    write(out / "metrics-full.json", metrics)
    print(
        json.dumps(
            dict(
                status="COMPLETE",
                verdict=decision["verdict"],
                accepts={
                    k: v["accept_count_by_seed"] for k, v in policy_summaries.items()
                },
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

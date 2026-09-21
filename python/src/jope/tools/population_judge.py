#!/usr/bin/env python3
"""Population judge: the pre-registered adoption rule over K parallel training arms of one block.

Inputs (all receipts already on disk; this tool computes and records, it never trains or promotes):
  --parent    the parent checkpoint's frozen-panel monitor receipt (parallel.frozen_monitoring.v1)
  --arm K=PATH[:IDENTITY]   each arm's terminal monitor receipt (same binding: panel, corpus, executable), K = arm index
                            as pre-registered; IDENTITY = the arm's ranking_backward identity (default dense_pairwise_v2)
  --baseline  the flywheel baseline receipt carrying panel_null.guards (g_nll, g_bt) — the F2 guards
  --tournament optional parallel.population_tournament.v1: per arm the 16-cut seed-only W/T/L against arm 0
  --history   optional prior judgments (for the 2-of-3 hysteresis of identity-changing arms)

Rule (PREREGISTRATION.md `population`): an arm is eligible iff none of sequence_nll, ranking_bt, ranking_listwise
exceeds the PARENT's receipt by the guards; eta = |ranking_bt(arm 0) - ranking_bt(arm 1)| (null pair); next parent =
arm 0 if eligible; a variant k replaces arm 0 only if ranking_bt_k <= ranking_bt_0 - max(eta, 0.25 * g_bt) AND, when a
tournament is supplied, wins_k >= losses_k against arm 0 on the 16 cuts; if arm 0 is ineligible and some variant is
eligible, the best eligible variant is adopted (population_rescue); if nothing is eligible the block is
diagnostic_rejected; an arm whose identity differs from the parent's is adopted only after winning 2 of 3 consecutive
judgments. The adopted arm's numbers are labelled selected_of_K with eta beside them. Output: immutable
parallel.population_judgment.v1.
"""

import argparse
import json
import math
from pathlib import Path

import promotion_gate
from resume_expired_cycle import read, require, sha

SCHEMA = "parallel.population_judgment.v1"
MONITOR_SCHEMA = "parallel.frozen_monitoring.v1"
TOURNAMENT_SCHEMA = "parallel.population_tournament.v1"
GUARDED = ("sequence_nll", "ranking_bt", "ranking_listwise")
DEFAULT_IDENTITY = "dense_pairwise_v2"
HYSTERESIS_WINS, HYSTERESIS_WINDOW = 2, 3
MARGIN_FRACTION = 0.25


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def load_monitor(path):
    value = read(path)
    require(
        isinstance(value, dict) and value.get("schema") == MONITOR_SCHEMA,
        "%s: schema is not %s" % (path, MONITOR_SCHEMA),
    )
    require(value.get("final_tests_opened") is False, "%s: final_tests_opened is not false" % path)
    require(
        all(finite(value.get(k)) for k in GUARDED + ("runtime_huber",)), "%s: non-finite panel value" % path
    )
    binding = value.get("binding") or {}
    require(
        all(
            isinstance(binding.get(k), str)
            for k in ("checkpoint_sha256", "corpus_sha256", "panel_sha256", "executable_sha256")
        ),
        "%s: binding incomplete" % path,
    )
    return dict(path=str(path), sha256=sha(path), value=value, binding=binding)


def guards_from(baseline_path):
    baseline = read(baseline_path)
    guards = ((baseline.get("metrics") or {}).get("panel_null") or {}).get("guards") or {}
    require(
        finite(guards.get("g_nll")) and finite(guards.get("g_bt")),
        "baseline receipt lacks finite panel_null.guards",
    )
    return dict(
        g_nll=guards["g_nll"],
        g_bt=guards["g_bt"],
        borrowed=((baseline.get("metrics") or {}).get("panel_null") or {}).get("guards_borrowed"),
        baseline=dict(path=str(baseline_path), sha256=sha(baseline_path)),
    )


def eligibility(arm, parent, guards):
    limits = dict(
        sequence_nll=parent["value"]["sequence_nll"] * (1 + guards["g_nll"]),
        ranking_bt=parent["value"]["ranking_bt"] * (1 + guards["g_bt"]),
        ranking_listwise=parent["value"]["ranking_listwise"] * (1 + guards["g_bt"]),
    )
    exceeded = {
        k: dict(value=arm["value"][k], limit=limits[k]) for k in GUARDED if arm["value"][k] > limits[k]
    }
    return dict(eligible=not exceeded, exceeded=exceeded, limits=limits)


def parse_arms(specs):
    arms = {}
    for spec in specs:
        require("=" in spec, "arm spec must be K=PATH[:IDENTITY]: " + spec)
        index, rest = spec.split("=", 1)
        path, _, identity = rest.partition(":")
        index = int(index)
        require(index not in arms, "duplicate arm %d" % index)
        arms[index] = dict(load_monitor(Path(path)), identity=identity or DEFAULT_IDENTITY)
    require(0 in arms, "arm 0 (the incumbent) is required")
    return arms


def load_tournament(path, arms):
    if path is None:
        return None
    value = read(path)
    require(
        isinstance(value, dict)
        and value.get("schema") == TOURNAMENT_SCHEMA
        and value.get("final_tests_opened") is False,
        "%s: not a %s" % (path, TOURNAMENT_SCHEMA),
    )
    rows = value.get("arms") or {}
    out = {}
    for key, row in rows.items():
        index = int(key)
        require(index in arms, "tournament names unknown arm %s" % key)
        require(
            all(type(row.get(k)) is int for k in ("wins", "ties", "losses")),
            "tournament arm %s lacks W/T/L" % key,
        )
        out[index] = dict(
            wins=row["wins"],
            ties=row["ties"],
            losses=row["losses"],
            mean_log_ratio=row.get("mean_log_ratio"),
            ci90=row.get("ci90"),
        )
    return dict(path=str(path), sha256=sha(path), arms=out, contexts=value.get("contexts"))


def hysteresis(history, index, identity, parent_identity):
    """An identity-changing arm needs 2 wins in the last 3 judgments (this one counted) before it becomes the parent."""
    if identity == parent_identity:
        return dict(required=False, satisfied=True, wins=None)
    recent = [
        h
        for h in (history or [])
        if h.get("candidate_arm") == index and h.get("candidate_identity") == identity
    ][-(HYSTERESIS_WINDOW - 1) :]
    wins = sum(1 for h in recent if h.get("candidate_won")) + 1
    return dict(
        required=True,
        satisfied=wins >= HYSTERESIS_WINS,
        wins=wins,
        window=HYSTERESIS_WINDOW,
        needed=HYSTERESIS_WINS,
    )


def judge(parent, arms, guards, tournament=None, history=None, cycle=None):
    parent_identity = parent.get("identity", DEFAULT_IDENTITY)
    for index, arm in arms.items():
        require(
            arm["binding"]["panel_sha256"] == parent["binding"]["panel_sha256"]
            and arm["binding"]["corpus_sha256"] == parent["binding"]["corpus_sha256"]
            and arm["binding"]["executable_sha256"] == parent["binding"]["executable_sha256"],
            "arm %d monitored on a different panel/corpus/executable than the parent" % index,
        )
        require(
            arm["binding"]["checkpoint_sha256"] != parent["binding"]["checkpoint_sha256"],
            "arm %d is the parent checkpoint" % index,
        )
    checks = {index: eligibility(arm, parent, guards) for index, arm in arms.items()}
    eligible = sorted(index for index, c in checks.items() if c["eligible"])
    bt = {index: arm["value"]["ranking_bt"] for index, arm in arms.items()}
    eta = abs(bt[0] - bt[1]) if 1 in arms else None
    margin = max(eta if eta is not None else 0.0, MARGIN_FRACTION * guards["g_bt"])
    adopted, reason, rescue = None, None, False
    candidates = sorted((bt[i], i) for i in eligible if i != 0)
    if 0 in eligible:
        adopted, reason = 0, "incumbent eligible"
        for value, index in candidates:
            if value > bt[0] - margin:
                continue
            if tournament is not None:
                row = tournament["arms"].get(index)
                if row is None or row["wins"] < row["losses"]:
                    continue
            gate = hysteresis(history, index, arms[index]["identity"], parent_identity)
            if not gate["satisfied"]:
                continue
            adopted, reason = (
                index,
                "variant %d beats the incumbent by %.6f >= margin %.6f%s"
                % (index, bt[0] - value, margin, " with tournament wins >= losses" if tournament else ""),
            )
            break
    elif candidates:
        for value, index in candidates:
            gate = hysteresis(history, index, arms[index]["identity"], parent_identity)
            if gate["satisfied"]:
                adopted, reason, rescue = (
                    index,
                    "incumbent ineligible; best eligible variant adopted (population_rescue)",
                    True,
                )
                break
        if adopted is None:
            reason = "incumbent ineligible; eligible variants all await hysteresis"
    else:
        reason = "no eligible arm: block diagnostic_rejected"
    outcome = (
        "population_adopted"
        if adopted == 0
        else (
            "population_rescue"
            if rescue
            else ("population_variant_adopted" if adopted is not None else "population_rejected")
        )
    )
    return dict(
        schema=SCHEMA,
        cycle=cycle,
        outcome=outcome,
        adopted_arm=adopted,
        reason=reason,
        parent=dict(
            path=parent["path"],
            sha256=parent["sha256"],
            checkpoint_sha256=parent["binding"]["checkpoint_sha256"],
            identity=parent_identity,
            panel={k: parent["value"][k] for k in GUARDED + ("runtime_huber",)},
        ),
        guards=guards,
        eta=eta,
        margin=margin,
        margin_rule="max(eta, %g * g_bt)" % MARGIN_FRACTION,
        arms={
            str(index): dict(
                path=arm["path"],
                sha256=arm["sha256"],
                checkpoint_sha256=arm["binding"]["checkpoint_sha256"],
                identity=arm["identity"],
                panel={k: arm["value"][k] for k in GUARDED + ("runtime_huber",)},
                eligible=checks[index]["eligible"],
                exceeded=checks[index]["exceeded"],
                tournament=(tournament or {}).get("arms", {}).get(index) if tournament else None,
                hysteresis=hysteresis(history, index, arm["identity"], parent_identity),
            )
            for index, arm in arms.items()
        },
        tournament=dict(
            path=tournament["path"], sha256=tournament["sha256"], contexts=tournament.get("contexts")
        )
        if tournament
        else None,
        selected_of_k=len(arms),
        label="the adopted arm's panel numbers are selected_of_K with eta beside them (winner's curse); improvement claims remain F1/F8 only",
        candidate_arm=adopted,
        candidate_identity=arms[adopted]["identity"] if adopted is not None else None,
        candidate_won=adopted not in (None, 0),
        final_tests_opened=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-identity", default=DEFAULT_IDENTITY)
    parser.add_argument("--arm", action="append", default=[], help="K=PATH[:IDENTITY] (repeatable)")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--tournament", type=Path)
    parser.add_argument("--history", type=Path, help="json list of prior judgments (for hysteresis)")
    parser.add_argument("--cycle", type=int)
    parser.add_argument("--output", type=Path, help="immutable judgment (required unless --dry-run)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    require(args.output is not None or args.dry_run, "--output is required unless --dry-run")
    parent = dict(load_monitor(args.parent), identity=args.parent_identity)
    arms = parse_arms(args.arm)
    history = read(args.history) if args.history else None
    judgment = judge(
        parent, arms, guards_from(args.baseline), load_tournament(args.tournament, arms), history, args.cycle
    )
    if args.dry_run:
        print(json.dumps(judgment, sort_keys=True))
        return judgment
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = promotion_gate.immutable(args.output, judgment)
    print(
        json.dumps(
            dict(
                judgment=str(args.output),
                sha256=digest,
                outcome=judgment["outcome"],
                adopted_arm=judgment["adopted_arm"],
                reason=judgment["reason"],
            )
        )
    )
    return judgment


if __name__ == "__main__":
    main()

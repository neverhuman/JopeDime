"""Successor remaining-gap register and flywheel-evidence contract. No launch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import closed_loop
import useful_hour
from selfplay_milestones import world_class

SCHEMA = "parallel.beyond_reproach.v1"
FLYWHEEL_SCHEMA = "parallel.flywheel_evidence.v1"
AUDIT = Path(__file__).resolve().parents[2] / "docs/audits/worldclass-redteam-20260911/gap-register.json"

THEATER = (
    "league_episode.v1 schema without live random arm",
    "replay argv without corpus/baseline/checkpoint",
    "win-rate receipt without quality-report seed pairs",
    "H05 v3 identities with qualification_ready=false",
    "coverage in-flight weights without D04 census importer",
)
SUBSTANCE = (
    "new original admitted programs on a new corpus identity",
    "real verify_execution_protocol producer",
    "accepted challenger on two predeclared panels",
    "P2 original discoveries attributable to the successor",
    "useful hour joined to committed propose/search/pack outputs",
)
FAKE_CLOSE = {
    "R01": "inventory classified without unread/unresolved paths remaining explicit",
    "R02": "source fence claimed while root/vendor routes still start fits",
    "R03": "query tests passed treated as production query access",
    "R04": "mirror controller install treated as certified sync",
    "R05": "idle GPUs treated as a funding grant",
    "R06": "stub verify_execution_protocol",
    "R07": "108 AdamW resets counted as continuous optimizer age",
    "R08": "warm resident pool treated as qualified replacement",
    "R09": "interrupted 8-pass/1-fail query treated as a passing release",
    "R10": "restart PID 774452 or unfence legacy GP",
    "D01": "count 1930 triplets or 4048 as eligible",
    "D02": "CPU helper identity assumed from path names",
    "D03": "apply to live d77e856b contexts file",
    "D04": "run weighted sampler without strict census",
    "D05": "metadata disjointness treated as leakage qualification",
    "M01": "50/50 groups reported as 50% fresh programs",
    "M02": "world-size change compared across objective identities",
    "M03": "architecture existence treated as selection benefit",
    "M04": "identical nominal conditions fabricated as fresh support",
    "G01": "unreleased weighted-parent helper treated as production RNG",
    "G02": "grammar capability counted as measured reachability",
    "G03": "missing slots imputed as wins",
    "G04": "normalized RMSE reported as original-unit skill",
    "H01": "wall seconds equated to core-seconds",
    "H02": "layout rewrite without exact parity",
    "H03": "cache reuse without collision fixtures",
    "H04": "whole-device invalidation called incremental",
    "H05": "pack screen/proxy scores as teachers",
    "H06": "admit CUDA GPR after closed-negative",
    "H07": "idle paid GPUs credited as savings",
    "H08": "fast-math or identity-changing flags",
    "H09": "hot-path hashing added before custody throughput",
    "S01": "receipts or occupancy claimed as accepted promotion",
    "S02": "rank-seconds or VRAM residency without committed outputs",
    "L01": "schema-only episode treated as live games",
    "L02": "IS weights without logprobs",
    "L03": "materialize_view or unfunded freeze",
}
WOULD_CLOSE = {
    "L01": "live GP jobs mint a random legal-grammar arm with paired receipts",
    "L02": "seeds persist finite policy_logprob on teaching observations",
    "L03": "native freeze() on a funded new-identity corpus/baseline/checkpoint",
}
EXTRA = (
    dict(id="L01", owner="scheduler", priority="P1",
         problem="vs-random league arm is not minted in live GP jobs",
         closed=False, evidence=[]),
    dict(id="L02", owner="scheduler", priority="P1",
         problem="policy_logprob not persisted on observation seeds",
         closed=False, evidence=[]),
    dict(id="L03", owner="scheduler", priority="P1",
         problem="teacher_improved replay argv has unresolved corpus/baseline/checkpoint",
         closed=False, evidence=[]),
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_audit(path=AUDIT):
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("schema") != "jope.redteam-action-register.v1":
        raise ValueError("unknown audit gap-register schema")
    return path, data


def evidence_sha(paths):
    return digest(list(paths or []))


def annotate(items):
    rows = []
    for item in list(items) + [dict(extra) for extra in EXTRA]:
        ident = item["id"]
        evidence = list(item.get("evidence") or [])
        row = dict(
            id=ident,
            owner=item.get("owner"),
            priority=item.get("priority"),
            problem=item.get("problem"),
            closed=bool(item.get("closed")),
            would_close=item.get("acceptance") or WOULD_CLOSE.get(ident),
            fake_close=FAKE_CLOSE[ident],
            evidence=evidence,
            evidence_sha256=evidence_sha(evidence),
            this_lane_owns=False,
        )
        rows.append(row)
    return rows


def refuse_world_class(rates):
    return not world_class(rates)


def refuse_recovery_train(**kwargs):
    kwargs.setdefault("hold", "held")
    kwargs.setdefault("automatic_training", True)
    kwargs.setdefault("teachers_growth", True)
    kwargs.setdefault("new_teacher_beats", 3)
    kwargs.setdefault("corpus_is_recovery", True)
    ok, reason = closed_loop.launch_train_allowed(**kwargs)
    return (not ok), reason


def refuse_useful_hour(outputs=None):
    return useful_hour.committed(outputs) is False


def flywheel_effect(p1_accepted, p2_successor_discoveries, useful_hour_committed=False,
                    world_class_ok=False, vs_random="unavailable"):
    blocked = []
    if not p1_accepted:
        blocked.append("p1_not_accepted")
    if not p2_successor_discoveries:
        blocked.append("p2_no_successor_discoveries")
    if not useful_hour_committed:
        blocked.append("useful_hour_uncommitted")
    if not world_class_ok:
        blocked.append("world_class_false")
    if vs_random == "unavailable":
        blocked.append("vs_random_unavailable")
    return dict(
        schema=FLYWHEEL_SCHEMA,
        flywheel_effect=False if blocked else True,
        p1_accepted=bool(p1_accepted),
        p2_successor_discoveries=bool(p2_successor_discoveries),
        useful_hour=bool(useful_hour_committed),
        world_class=bool(world_class_ok),
        vs_random=vs_random,
        serving_written=False,
        train_launched=False,
        blocked_by=blocked,
        note="P1 is an accepted promotion with original producers; P2 is subsequent original discoveries on the successor. Receipts are not P1.",
    )


def register(audit_path=AUDIT):
    path, data = load_audit(audit_path)
    items = annotate(data.get("items") or [])
    closed_n = sum(1 for r in items if r["closed"])
    body = dict(
        schema=SCHEMA,
        source_path=str(path),
        source_sha256=file_sha(path),
        source_not_overwritten=True,
        theater=list(THEATER),
        substance=list(SUBSTANCE),
        items=items,
        n_items=len(items),
        n_closed=closed_n,
        n_open=len(items) - closed_n,
        overall_scientific_objective_complete=False,
        restart_774452=False,
        protocol_stubbed=False,
    )
    body["identity"] = digest({k: body[k] for k in body if k != "identity"})
    return body


def emit(dest, audit_path=AUDIT, winrate=None):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    reg = register(audit_path)
    (dest / "register.json").write_text(json.dumps(reg, sort_keys=True, indent=2) + "\n")
    rates = {"neural_seed_rate": 0.0, "final_gp_rate": 0.014285714285714285,
             "final_gp_margin_rate": 0.0}
    if winrate:
        wr = json.loads(Path(winrate).read_text())
        for key in rates:
            value = wr.get(key)
            if isinstance(value, (int, float)):
                rates[key] = value
    effect = flywheel_effect(False, False, False, world_class(rates), "unavailable")
    (dest / "flywheel.json").write_text(json.dumps(effect, sort_keys=True, indent=2) + "\n")
    summary = dict(
        schema="parallel.beyond_reproach_summary.v1",
        register_sha256=file_sha(dest / "register.json"),
        flywheel_sha256=file_sha(dest / "flywheel.json"),
        flywheel_effect=effect["flywheel_effect"],
        n_open=reg["n_open"],
        n_closed=reg["n_closed"],
        world_class=effect["world_class"],
        useful_hour=False,
        launch_train_allowed=False,
        train_launched=False,
        serving_written=False,
    )
    (dest / "SUMMARY.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", type=Path, default=AUDIT)
    p.add_argument("--winrate", type=Path)
    args = p.parse_args()
    print(json.dumps(emit(args.output, args.audit, args.winrate), indent=2))

#!/usr/bin/env python3
"""H75 point diagnostics from admitted, complete development score records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

ELO_SHA256 = "9ece98326c32453179b0c765eb45944b68ca9de944d2828a1d42dee1401d1835"
DOPE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
SEEDS = [701, 702, 703]
PANELS = {
    "legacy": ["A", "B", "C", "D", "replay_B", "replay_D"],
    "confirmed_U8": ["A", "B", "C", "D", "G", "R"],
}
IDENTITIES = ("run_id", "source_git_sha", "prereg_sha256", "execution_manifest_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value, length=64):
    require(
        isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value), "invalid source/input digest"
    )
    return value


def identifier(value):
    require(
        isinstance(value, str)
        and value.strip() == value
        and value.lower() not in {"", "none", "null", "unknown", "unresolved"},
        "unresolved identifier",
    )


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "utility must be finite numeric data")
    return float(value)


def load_reference(path):
    source = Path(path).read_bytes()
    require(hashlib.sha256(source).hexdigest() == ELO_SHA256, "Elo source hash mismatch")
    module = types.ModuleType("_h75_pinned_elo_reference")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def admit(packet):
    """Join explicit row identities to a frozen caller-supplied scoring roster."""
    manifest, rows = packet["manifest"], packet["rows"]
    require(manifest["schema"] == "jope.h75-closed-loop.score-admission/v1", "wrong admission schema")
    require(type(manifest["fixture_only"]) is bool, "fixture scope must be explicit")
    identifier(manifest["run_id"])
    digest(manifest["source_git_sha"], 40)
    for name in ("prereg_sha256", "execution_manifest_sha256"):
        digest(manifest[name])
    require(manifest["dope_sha256"] == DOPE_SHA256, "DOPE identity mismatch")
    panel = manifest["panel"]
    require(panel in PANELS, "unknown or mixed score panel")
    roster, contexts, family_folds = {}, {}, {}
    for cell in manifest["roster"]:
        cid, seed, fold, family = cell["context_sha256"], cell["seed"], cell["fold"], cell["family_id"]
        digest(cid)
        identifier(family)
        require(type(seed) is int and seed in SEEDS, "unexpected seed")
        require(type(fold) is int and 0 <= fold < 5, "invalid fold")
        require(cell["role"] == "DL-Train", "only DL-Train development folds are admitted")
        require(family not in family_folds or family_folds[family] == fold, "family split across folds")
        family_folds[family] = fold
        require(cid not in contexts or contexts[cid] == (family, fold), "conflicting context identity")
        contexts[cid] = (family, fold)
        require((cid, seed) not in roster, "duplicate roster cell")
        digest(cell["evaluation_identity_sha256"])
        if panel == "confirmed_U8":
            cuts = cell["confirmation_cut_sha256s"]
            require(
                isinstance(cuts, list) and len(cuts) == len(set(cuts)) == 3,
                "need three fixed confirmation identities",
            )
            for cut in cuts:
                digest(cut)
        roster[cid, seed] = cell
    require(
        bool(contexts) and set(roster) == {(cid, seed) for cid in contexts for seed in SEEDS},
        "incomplete admission roster",
    )
    observed, statuses = {}, Counter()
    for row in rows:
        require(row["schema"] == "jope.h75-closed-loop.score/v1", "wrong score schema")
        require(all(row[name] == manifest[name] for name in IDENTITIES), "foreign execution identity")
        require(row["panel"] == panel and row["policy_id"] in PANELS[panel], "mixed/foreign panel or policy")
        cid, seed = row["context_sha256"], row["seed"]
        require(type(seed) is int and (cid, seed) in roster, "foreign context/seed")
        cell = roster[cid, seed]
        require(
            type(row["fold"]) is int
            and all(
                row[name] == cell[name]
                for name in ("fold", "family_id", "role", "evaluation_identity_sha256")
            ),
            "score/roster identity mismatch",
        )
        digest(row["selected_program_sha256"])
        if panel == "confirmed_U8":
            require(
                row["confirmation_cut_sha256s"] == cell["confirmation_cut_sha256s"],
                "confirmation identities changed",
            )
        key = (cid, seed, row["policy_id"])
        require(key not in observed, "duplicate/conflicting score row")
        status = row["status"]
        require(status in {"completed", "failed", "timeout", "unknown"}, "unknown execution status")
        if row["utility"] is not None:
            finite(row["utility"])
        require(status != "completed" or row["utility"] is not None, "completed score lacks utility")
        statuses[status] += 1
        observed[key] = row
    expected = len(roster) * len(PANELS[panel])
    return manifest, contexts, observed, dict(statuses), expected


def score(packet, reference):
    manifest, contexts, observed, statuses, expected = admit(packet)
    panel = manifest["panel"]
    report = {
        "schema": "jope.h75-closed-loop.point-metrics/v1",
        "fixture_only": manifest["fixture_only"],
        **{name: manifest[name] for name in IDENTITIES},
        "panel": panel,
        "dope_sha256": DOPE_SHA256,
        "n_contexts": len(contexts),
        "n_families": len({v[0] for v in contexts.values()}),
        "expected_rows": expected,
        "observed_rows": len(observed),
        "row_status_counts": statuses,
        "missing_rows": expected - len(observed),
        "seeds": SEEDS,
        "contenders": PANELS[panel],
        "status": "INCOMPLETE",
        "elo": None,
        "raw_utility": None,
        "bootstrap": {
            "status": "NOT_COMPUTED",
            "required_draws": 10000,
            "executed_draws": 0,
            "seed": 20261031,
        },
        "admission_decision": None,
        "elo_source_sha256": ELO_SHA256,
        "scope": "Point diagnostics only. Caller-supplied roster provenance and preconfirmation selection remain unverified; r882/A continuity is conditional on prior exposure. No bootstrap or scientific admission.",
    }
    if len(observed) != expected or statuses.get("completed", 0) != expected:
        return report
    means = {
        cid: {
            policy: statistics.fmean(observed[cid, seed, policy]["utility"] for seed in SEEDS)
            for policy in PANELS[panel]
        }
        for cid in sorted(contexts)
    }
    for values in means.values():
        for value in values.values():
            finite(value - values["A"])
    records = [
        reference.DatasetEvalRecord(cid, contexts[cid][0], "regression", values)
        for cid, values in means.items()
    ]
    report["elo"] = reference.compute_family_grouped_bradley_terry_elo(
        records,
        anchor_contender="A",
        anchor_elo=1500.0,
        tie_margin=1e-4,
        prior=0.01,
        n_iters=400,
        allow_incomplete_panel=False,
    )
    families = defaultdict(lambda: defaultdict(list))
    for cid, values in means.items():
        for policy, value in values.items():
            families[contexts[cid][0]][policy].append(value)
    family_means = {
        family: {policy: statistics.fmean(values) for policy, values in sorted(policies.items())}
        for family, policies in sorted(families.items())
    }
    raw = {}
    for policy in PANELS[panel]:
        deltas = [finite(values[policy] - values["A"]) for values in family_means.values()]
        raw[policy] = {
            "val_reg_inner_context_mean": statistics.fmean(values[policy] for values in means.values())
            if panel == "legacy"
            else None,
            "context_mean_utility": statistics.fmean(values[policy] for values in means.values()),
            "equal_family_mean_utility": statistics.fmean(values[policy] for values in family_means.values()),
            "mean_paired_family_delta_vs_A": statistics.fmean(deltas),
            "median_paired_family_delta_vs_A": statistics.median(deltas),
            "fraction_families_delta_below_minus_0_01": sum(d < -0.01 for d in deltas) / len(deltas),
        }
    report.update(
        status="COMPLETE_POINT_DIAGNOSTICS",
        raw_utility=raw,
        per_context_seed_mean_utility=means,
        per_family_mean_utility=family_means,
        elo_parameters={
            "anchor": "A",
            "anchor_rating": 1500,
            "tie_margin": 1e-4,
            "prior": 0.01,
            "iterations": 400,
        },
    )
    return report


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--elo-source", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        source = args.packet.read_bytes()
        packet = json.loads(source, object_pairs_hook=unique_object)
        admit(packet)
        report = score(packet, load_reference(args.elo_source))
        report["packet_sha256"] = hashlib.sha256(source).hexdigest()
        report["admission_manifest_sha256"] = hashlib.sha256(
            json.dumps(packet["manifest"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        report["scorer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            text=True,
            capture_output=True,
            check=False,
        )
        report["scorer_commit"] = git.stdout.strip() if git.returncode == 0 else None
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError, OverflowError, ImportError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

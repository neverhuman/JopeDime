#!/usr/bin/env python3
"""Publish frozen H99/H100 readouts after the complete FIFO receipt panel exists."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882 = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
MANIFEST = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
SEEDS = {"H99": (36, 37, 38), "H100": (39, 40, 41)}
ARMS = {
    "H99": ("ESH-l2", "ESH-l4"),
    "H100": ("CONTROL", "ASAM05", "ASAM10", "DRIFT_MATCH"),
}
MARGIN = {"H99": 0.001, "H100": 0.001546}


class Pending(Exception):
    """A complete immutable terminal panel is not available yet."""


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def safe_path(path):
    path = Path(path).absolute()
    for part in path.parts:
        require(
            not any(
                x in part.lower()
                for x in (
                    "admiralty",
                    "sealed",
                    "holdout",
                    "meta-test",
                    "meta_test",
                    "flywheel-full-20261030",
                )
            ),
            "protected/live path rejected",
        )
    for parent in (path, *path.parents):
        require(not parent.is_symlink(), "receipt paths must not contain symlinks")
    return path.resolve()


def finite(value):
    require(
        type(value) in (int, float) and math.isfinite(value),
        "nonfinite or nonnumeric statistic",
    )
    return value


def near(left, right):
    return math.isclose(finite(left), finite(right), rel_tol=1e-10, abs_tol=1e-12)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def h99_fold_assignment_matches(cell, fold):
    cell_assignment = cell.get("fold_assignment")
    fold_assignment = fold.get("fold_assignment")
    if not isinstance(cell_assignment, dict) or not isinstance(fold_assignment, dict):
        return False
    if cell_assignment.get("seed") != 9900 or fold_assignment.get("seed") != 9900:
        return False
    keys = ("fold_a_families", "fold_b_families", "folds")
    return all(
        isinstance(cell_assignment.get(key), list)
        and isinstance(fold_assignment.get(key), list)
        and cell_assignment[key] == fold_assignment[key]
        for key in keys
    )


class Receipts:
    def __init__(self, root):
        self.root = safe_path(root)
        self.hashes = {}

    def read(self, path, expected_sha=None):
        path = safe_path(path)
        require(
            path.is_relative_to(self.root) and path.suffix == ".json",
            "receipt escapes run root or is not JSON",
        )
        if not path.exists():
            raise Pending(f"missing terminal receipt: {path.relative_to(self.root)}")
        require(
            path.is_file() and path.stat().st_size <= 16 * 1024**2,
            "invalid/oversized JSON receipt",
        )
        data = path.read_bytes()
        digest = sha(data)
        require(
            expected_sha is None or digest == expected_sha,
            "FIFO receipt digest mismatch",
        )
        value = json.loads(
            data,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
        require(isinstance(value, dict), "receipt must be an object")
        self.hashes[str(path.relative_to(self.root))] = digest
        return value

    def stable(self):
        for relative, digest in self.hashes.items():
            path = safe_path(self.root / relative)
            if not path.exists() or sha(path.read_bytes()) != digest:
                raise Pending("receipt snapshot changed during readout")


def terminal_panel(wave, ledger):
    cap = 9000 if wave == "H99" else 2700
    require(ledger.get("cap_seconds") == cap, "GPU cap identity mismatch")
    require(
        ledger.get("schema") == "jope.h99.gpu-budget/v1"
        if wave == "H99"
        else ledger.get("wave") == wave,
        "FIFO ledger identity mismatch",
    )
    if ledger.get("in_progress") is not None:
        raise Pending("FIFO action is still in progress; no outcome files opened")
    rows = ledger.get("attempts", [])
    require(isinstance(rows, list), "invalid FIFO attempts")
    expected = {(arm, seed) for arm in ARMS[wave] for seed in SEEDS[wave]}
    if wave == "H99":
        expected.add(("NULL", 36))
    complete, extra, spent = {}, [], 0.0
    for row in rows:
        wall = finite(row["wall_seconds"])
        require(
            wall >= 0 and finite(row["wait_seconds"]) >= 0,
            "negative/unknown resource charge",
        )
        spent += wall
        key = (row.get("arm"), row.get("seed"))
        special = (
            row.get("action") == "fold_b_scoring"
            if wave == "H99"
            else row.get("timing") is True
        )
        require(special or key in expected, "unregistered FIFO action")
        success = (
            row.get("status") == "complete" if wave == "H99" else not row.get("error")
        )
        if not success:
            # Preserve charged old pre-scoring failures. A failed scoring action is unknown.
            require(
                not special and (wave == "H99" or row.get("outcome_started") is False),
                "failed/unknown scoring action blocks readout",
            )
            continue
        require(wall > 0, "completed action has no positive wall charge")
        if special:
            extra.append(row)
        else:
            require(key not in complete, "duplicate completed FIFO cell")
            complete[key] = row
    require(spent <= cap, "cumulative GPU wall cap exceeded")
    require(len(extra) <= 1, "duplicate timing/scoring action")
    if set(complete) != expected or len(extra) != 1:
        raise Pending(
            "complete fixed FIFO panel and terminal scoring/timing action required"
        )
    return complete, extra[0], spent


def interval(values):
    require(
        isinstance(values, list) and len(values) == 2, "95% interval needs two bounds"
    )

    def bound(value):
        if value in ("Infinity", "-Infinity"):
            return float(value)
        return finite(value)

    require(bound(values[0]) <= bound(values[1]), "reversed Elo interval")
    return values


def elo_packet(point, ci, per_seed, seeds):
    interval([point, point])
    require([x["seed"] for x in per_seed] == list(seeds), "Elo seed inventory mismatch")
    for row in per_seed:
        interval([row["elo"], row["elo"]])
    return {
        "median": point,
        "ci95_family_cluster": interval(ci),
        "per_seed": per_seed,
        "reference": "r882",
        "reference_elo": 1500,
    }


def win_tie_loss(values, margin):
    wins = sum(x > margin for x in values)
    losses = sum(x < -margin for x in values)
    return {"wins": wins, "ties": len(values) - wins - losses, "losses": losses}


def holm_decisions(pvalues, strict):
    ordered = sorted(pvalues, key=lambda arm: (pvalues[arm], arm))
    passing, decisions = True, {}
    for rank, arm in enumerate(ordered):
        p = finite(pvalues[arm])
        require(0 <= p <= 1, "invalid native Holm tail area")
        threshold = 0.05 / (len(ordered) - rank)
        passing = passing and (p < threshold if strict else p <= threshold)
        decisions[arm] = passing
    return decisions


def gate(wave, item, lcb, point, harm, holm, diagnostic=False):
    require(0 <= finite(harm) <= 1, "invalid family harm fraction")
    checks = {
        "strict_lcb": finite(lcb) > MARGIN[wave],
        "nonnegative_point": finite(point) >= 0,
        "holm": holm,
        "family_harm_le_10pct": harm <= 0.10,
        "eligible": not diagnostic,
    }
    passed = all(checks.values())
    require(
        item["admitted"] is passed, "native admission contradicts common gate checks"
    )
    return {
        "verdict": "ADMIT" if passed else "NO_ADMIT",
        "checks": checks,
        "primary_reference": "r882" if wave == "H99" else "CONTROL",
        "strict_margin": MARGIN[wave],
        "lcb95": lcb,
        "point": point,
        "harm_reference": "r882",
        "harm_fraction": harm,
    }


def h99(store, cells, scoring):
    fold = store.read(store.root / "outcomes/fold_b.json")
    metrics = store.read(store.root / "outcomes/admission.json")
    require(
        set(metrics["arms"]) == set(ARMS["H99"]), "H99 native arm inventory mismatch"
    )
    require(
        scoring["result"]["fold_b"] == str(store.root / "outcomes/fold_b.json"),
        "FIFO scoring receipt path mismatch",
    )
    require(
        fold.get("schema") == "jope.h99.fold-b/v1"
        and fold.get("status") == "complete"
        and fold.get("fold_seed") == 9900,
        "H99 Fold-B identity mismatch",
    )
    require(
        metrics.get("schema") == "jope.h99.admission-metrics/v1"
        and metrics.get("status") == "complete"
        and metrics.get("fold_seed") == 9900,
        "H99 native metrics identity mismatch",
    )
    for key, expected in {
        "source_manifest_sha256": MANIFEST,
        "r882_checkpoint_sha256": R882,
        "dope_sha256_before": DOPE,
        "dope_sha256_after": DOPE,
        "candidate_count": 6,
        "fold_b_read_count": 6,
        "fold_b_family_count": 41,
        "null_cell_complete": True,
        "null_fold_b_scored": False,
        "validation_payloads_opened": False,
        "holdout_payloads_opened": 0,
        "outer_rows_delivered": 0,
    }.items():
        require(fold.get(key) == expected, f"H99 {key} mismatch")
    families = sorted(fold["baseline_family_utility"])
    require(
        len(families) == 41
        and metrics["fold_b_family_count"] == 41
        and metrics["fold_b_read_count"] == 6
        and metrics["bootstrap_draws"] == 10000
        and metrics["strict_margin"] == MARGIN["H99"],
        "H99 aggregate protocol mismatch",
    )
    require(
        metrics["source_manifest_sha256"] == MANIFEST
        and metrics["fold_assignment_sha256"] == fold["fold_assignment_sha256"],
        "H99 aggregate source/fold mismatch",
    )
    rows = {(x["arm"], x["seed"]): x for x in fold["candidates"]}
    require(
        len(rows) == len(fold["candidates"]) == 6
        and set(rows) == set(cells) - {("NULL", 36)},
        "H99 candidate panel mismatch",
    )
    for key, receipt in cells.items():
        summary = store.read(Path(receipt["result"]["summary"]))
        require(
            (summary["arm"], summary["seed"]) == key
            and summary["status"] == "complete"
            and summary["schema"] == "jope.h99.cell/v1",
            "H99 cell identity mismatch",
        )
        require(
            summary["fold_seed"] == 9900
            and h99_fold_assignment_matches(summary, fold)
            and summary["fold_b_scored"] is False,
            "H99 cell fold/scoring mismatch",
        )
        require(
            summary["dope_sha256_before"] == summary["dope_sha256_after"] == DOPE
            and summary["r882_checkpoint_sha256"] == R882
            and summary["source_manifest_sha256"] == MANIFEST,
            "H99 cell source/DOPE mismatch",
        )
        require(
            summary["candidate_adapter_artifact_sha256"]
            == receipt["result"]["artifact_sha256"],
            "H99 candidate digest mismatch",
        )
        if key in rows:
            require(
                rows[key]["candidate_adapter_sha256"]
                == receipt["result"]["artifact_sha256"],
                "H99 scored artifact differs from FIFO artifact",
            )
    pvalues = {
        arm: metrics["arms"][arm]["holm"]["raw_p_one_sided"] for arm in ARMS["H99"]
    }
    decisions = holm_decisions(pvalues, strict=False)
    result = {}
    for arm in ARMS["H99"]:
        item = metrics["arms"][arm]
        columns = []
        for seed in SEEDS["H99"]:
            row = rows[arm, seed]
            require(
                row["fold_b_scoring_passes"] == 1
                and sorted(row["family_gain_vs_r882"]) == families
                and sorted(row["family_utility"]) == families,
                "H99 family/score inventory mismatch",
            )
            values = [finite(row["family_gain_vs_r882"][f]) for f in families]
            require(
                all(
                    near(
                        row["family_utility"][f] - fold["baseline_family_utility"][f],
                        value,
                    )
                    for f, value in zip(families, values, strict=True)
                ),
                "H99 paired gains mismatch",
            )
            columns.append(values)
        medians = [statistics.median(x) for x in zip(*columns, strict=True)]
        harm = sum(x < -0.01 for x in medians) / len(families)
        require(
            near(
                item["pooled_mean_gain"],
                statistics.mean(x for col in columns for x in col),
            )
            and near(item["pooled_family_median_gain"], statistics.median(medians))
            and near(item["harm_fraction_below_minus_0_01"], harm),
            "H99 aggregate differs from scored families",
        )
        require(
            item["holm"]["holm_reject_alpha_0_05"] is decisions[arm],
            "H99 Holm decision mismatch",
        )
        per_seed = [
            {
                "seed": x["seed"],
                "elo": x["elo_vs_r882"],
                "ci95": interval(x["elo_ci95_family_cluster"]),
                "family_win_tie_loss": win_tie_loss(
                    columns[SEEDS["H99"].index(x["seed"])], 1e-4
                ),
            }
            for x in item["per_seed"]
        ]
        result[arm] = {
            "elo_vs_r882": elo_packet(
                item["elo_vs_r882_median_across_seeds"],
                item["elo_vs_r882_median_ci95_family_cluster"],
                per_seed,
                SEEDS["H99"],
            ),
            "common_gate": gate(
                "H99",
                item,
                item["mean_lcb95_one_sided"],
                item["pooled_family_median_gain"],
                harm,
                decisions[arm],
            ),
        }
    return result, 41


def h100(store, cells, timing, ledger):
    require(
        (timing["arm"], timing["seed"]) == ("ASAM10", 39),
        "H100 timing identity mismatch",
    )
    native = store.read(store.root / "metrics.json")
    require(
        set(native["arms"]) == set(ARMS["H100"]), "H100 native arm inventory mismatch"
    )
    require(
        native.get("schema") == "jope.h100.metrics/v1"
        and native.get("complete") is True
        and native.get("seeds") == list(SEEDS["H100"])
        and native.get("bootstrap_draws") == 10000,
        "H100 native metrics identity mismatch",
    )
    require(
        native.get("ledger") == ledger,
        "H100 aggregate is not bound to this FIFO ledger",
    )
    require(
        near(
            native["gpu_wall_seconds"],
            sum(x["wall_seconds"] for x in ledger["attempts"]),
        ),
        "H100 aggregate wall accounting mismatch",
    )
    store.read(
        store.root / f"timing-ASAM10-seed39-attempt{timing['attempt']}.json",
        timing["receipt_sha256"],
    )
    packets, identities = {}, set()
    for key, row in cells.items():
        require(
            row.get("outcome_started") is True and row.get("stage") == "complete",
            "H100 receipt has no complete scoring boundary",
        )
        path = store.root / f"{key[0]}-seed{key[1]}-attempt{row['attempt']}.json"
        cell = store.read(path, row["receipt_sha256"])
        require(
            (cell["arm"], cell["seed"]) == key
            and cell["wave"] == "H100"
            and cell["timing_only"] is False,
            "H100 cell identity mismatch",
        )
        require(
            cell["dope_before"] == cell["dope_after"] == DOPE
            and cell["checkpoint_sha256"] == R882
            and cell["manifest_sha256"] == MANIFEST
            and cell["trainable_count"] == 29211
            and cell["updates"] == 40
            and cell["holdout_payloads_opened"] == cell["outer_rows_delivered"] == 0,
            "H100 cell source/DOPE mismatch",
        )
        identities.add(
            json.dumps(
                [
                    cell["source_commit"],
                    cell["prereg_sha256"],
                    cell["initial_adapter_sha256"],
                    cell["source_hashes"],
                    cell["folds"],
                ],
                sort_keys=True,
            )
        )
        packets[key] = cell
    require(len(identities) == 1, "mixed H100 source/fold/reference identities")
    families = sorted(packets["CONTROL", 39]["dev"]["per_family"])
    require(len(families) == 19, "H100 requires nineteen dev families")
    for (arm, seed), cell in packets.items():
        require(
            sorted(cell["dev"]["per_family"])
            == sorted(cell["r882_dev"]["per_family"])
            == families
            and cell["r882_dev"] == packets["CONTROL", seed]["r882_dev"],
            "H100 paired family/reference mismatch",
        )
    pvalues = {
        a: native["holm"][a]["tail_area"] for a in ARMS["H100"] if a != "CONTROL"
    }
    decisions = holm_decisions(pvalues, strict=True)
    result = {}
    for arm in ARMS["H100"]:
        item = native["arms"][arm]
        gains, delta = [], []
        for seed in SEEDS["H100"]:
            utility = packets[arm, seed]["dev"]["per_family"]
            control = packets["CONTROL", seed]["dev"]["per_family"]
            reference = packets[arm, seed]["r882_dev"]["per_family"]
            gains.append([finite(utility[f]) - finite(control[f]) for f in families])
            delta.append([finite(utility[f]) - finite(reference[f]) for f in families])
        point = statistics.median(statistics.mean(x) for x in gains)
        harm = sum(statistics.mean(x) < -0.01 for x in zip(*delta, strict=True)) / len(
            families
        )
        require(
            near(item["median_gain"], point) and near(item["harm_fraction"], harm),
            "H100 aggregate differs from scored families",
        )
        eligible = arm != "CONTROL"
        if eligible:
            require(
                native["holm"][arm]["passes"] is decisions[arm],
                "H100 Holm decision mismatch",
            )
        elo = item["elo_vs_r882"]
        per_seed = [
            {
                "seed": x["seed"],
                "elo": x["elo"],
                "ci95": interval(x["elo_ci95"]),
                "family_win_tie_loss": win_tie_loss(
                    delta[SEEDS["H100"].index(x["seed"])], 0.0
                ),
            }
            for x in elo["per_seed"]
        ]
        result[arm] = {
            "elo_vs_r882": elo_packet(
                elo["median_elo"], elo["median_elo_ci95"], per_seed, SEEDS["H100"]
            ),
            "common_gate": gate(
                "H100",
                item,
                item["lcb95"],
                point,
                harm,
                decisions.get(arm, False),
                diagnostic=not eligible,
            ),
        }
    return result, 19


def readout(wave, root):
    result = {
        "schema": "jope.h99-h100.post-run-readout/v1",
        "wave": wave,
        "status": "PENDING",
        "verdict": "UNKNOWN",
        "arms": {},
        "readout_source_sha256": sha(Path(__file__).read_bytes()),
        "statistical_source": "unchanged native preregistered aggregate; no new bootstrap or gate",
        "serving_promotion": False,
        "native_elo_tie_margin": 1e-4 if wave == "H99" else 0.0,
    }
    store = Receipts(root)
    try:
        ledger = store.read(store.root / "budget.json")
        cells, special, spent = terminal_panel(wave, ledger)
        arms, families = (
            h99(store, cells, special)
            if wave == "H99"
            else h100(store, cells, special, ledger)
        )
        store.stable()
        result.update(
            status="COMPLETE",
            arms=arms,
            family_count=families,
            gpu_wall_seconds=spent,
            verdict="ADMIT"
            if any(x["common_gate"]["verdict"] == "ADMIT" for x in arms.values())
            else "NO_ADMIT",
        )
    except Pending as error:
        result["reason"] = str(error)
    except (ValueError, KeyError, TypeError, OSError) as error:
        result.update(status="INVALID", reason=str(error))
    result["receipt_sha256"] = store.hashes
    return result


def markdown(result):
    lines = [
        f"# {result['wave']} post-run readout",
        "",
        f"Status: **{result['status']}**. Verdict: **{result['verdict']}**.",
        "",
    ]
    if result["status"] == "COMPLETE":
        lines += [
            "Elo reference: r882 = 1500. Native paired family-bootstrap intervals; no pooling across waves.",
            "",
            "| Arm | Median Elo vs r882 | 95% family CI | Common-gate verdict |",
            "| --- | --- | --- | --- |",
        ]
        for arm, item in result["arms"].items():
            elo = item["elo_vs_r882"]
            lines.append(
                f"| {arm} | {elo['median']} | {elo['ci95_family_cluster']} | {item['common_gate']['verdict']} |"
            )
        lines += [
            "",
            "H99 primary: r882, margin >0.001. H100 primary: CONTROL, margin >0.001546. Both retain their own preregistered Holm and <=10% r882 family-harm gates.",
        ]
    else:
        lines.append(result["reason"])
    lines += [
        "",
        "Research readout only; no training, scoring, CUDA, FIFO/lock action or serving promotion.",
        "",
    ]
    return "\n".join(lines)


def publish(result, destination):
    destination = safe_path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for suffix, body in {
        "json": json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        "md": markdown(result),
    }.items():
        path = destination / f"{result['wave']}-readout.{suffix}"
        if path.exists():
            require(
                path.read_text() == body,
                "refusing to replace a prior readout; use a fresh output directory",
            )
        else:
            with path.open("x") as stream:
                stream.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", choices=SEEDS, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--timeout-seconds", type=float, default=21600)
    args = parser.parse_args()
    require(
        1 <= args.poll_seconds <= 60 and 1 <= args.timeout_seconds <= 86400,
        "invalid bounded watch timing",
    )
    root = safe_path(
        args.root
        or Path.home() / f"jope-data/{args.wave.lower()}/{args.wave.lower()}-wave"
    )
    destination = safe_path(args.out_dir)
    require(
        not destination.is_relative_to(root),
        "readout output must stay outside the input run root",
    )
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        if os.getloadavg()[0] > 85:
            if not args.watch:
                raise RuntimeError("CPU load exceeds85; retry later")
            result = {
                "status": "PENDING",
                "verdict": "UNKNOWN",
                "reason": "CPU load exceeds85; no receipt reads",
            }
        else:
            result = readout(args.wave, root)
        if (
            not args.watch
            or result["status"] != "PENDING"
            or time.monotonic() >= deadline
        ):
            if "schema" not in result:
                raise RuntimeError(result["reason"])
            publish(result, destination)
            print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
            return {"COMPLETE": 0, "INVALID": 2, "PENDING": 3}[result["status"]]
        print(
            json.dumps(
                {
                    "wave": args.wave,
                    "status": result["status"],
                    "verdict": "UNKNOWN",
                    "reason": result.get("reason"),
                }
            ),
            flush=True,
        )
        time.sleep(min(args.poll_seconds, max(0.0, deadline - time.monotonic())))


if __name__ == "__main__":
    raise SystemExit(main())

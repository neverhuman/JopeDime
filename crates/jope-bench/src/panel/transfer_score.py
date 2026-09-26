#!/usr/bin/env python3
"""Independent recomputation of the transfer KPI from the plan, labels and frozen predictions (W10-T09).

It does not reuse the Rust reporter's gains. From hash-verified assets it recomputes the
training-mean baseline and each arm's RMSE per search, the five-audit mean gain and the outer gain
per dataset/seed/outer unit, their per-unit minimum, and the equal-weight means over units within
datasets, datasets within families and families. Missing or failed pairs, zero or nonfinite
baselines and invalid values leave the score incomplete; nothing becomes a zero gain. With a
reporter output it compares every unit, family and the headline. Release and isolation evidence is
the reporter's responsibility and is not assessed here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

GAIN_FIELDS = ("audit", "outer", "contribution")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(asset):
    data = Path(asset["path"]).read_bytes()
    require(hashlib.sha256(data).hexdigest() == asset["sha256"], f"altered asset {asset['path']}")
    return json.loads(data)


def search_digest(search):
    """The planner's search identity: SHA-256 of its compact field-ordered JSON encoding."""
    fields = ("dataset", "seed", "outer", "audit", "train", "score", "inner_score")
    encoded = json.dumps({f: search[f] for f in fields}, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def rmse(actual, predicted):
    require(len(actual) == len(predicted) and actual, "prediction count differs from scoring rows")
    require(all(math.isfinite(x) for x in predicted), "nonfinite prediction")
    value = math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))
    require(math.isfinite(value), "nonfinite loss")
    return value


def mean(values):
    return math.fsum(values) / len(values)


def score(plan, freeze):
    specification = plan["specification"]
    arms = {arm["id"] for arm in specification["arms"]}
    require({"parent", "challenger"} <= arms, "parent and challenger arms required")
    searches = {search_digest(s): s for s in plan["searches"]}
    predictions = {}
    for asset in freeze["predictions"]:
        prediction = load(asset)
        key = (prediction["search_sha256"], prediction["arm"])
        require(key[0] in searches and key[1] in arms, "unplanned prediction")
        require(key not in predictions, "duplicate prediction")
        predictions[key] = prediction["predictions"]
    incomplete = [f"plan: {f}" for f in plan["failures"]]
    incomplete += [f"terminal {f['status']}: {f['search_sha256']} {f['arm']}" for f in freeze["failures"]]
    incomplete += [f"missing {sid} {arm}" for sid in searches for arm in sorted(arms)
                   if (sid, arm) not in predictions
                   and not any(f["search_sha256"] == sid and f["arm"] == arm for f in freeze["failures"])]
    if incomplete:
        return {"complete": False, "kpi_percentage_points": None, "reasons": incomplete}
    labels = {d["id"]: load(d["labels"]) for d in specification["datasets"]}
    losses = {}
    for sid, search in searches.items():
        table = labels[search["dataset"]]
        baseline_value = mean([table[r] for r in search["train"]])
        actual = [table[r] for r in search["score"]]
        baseline = rmse(actual, [baseline_value] * len(actual))
        for arm in sorted(arms):  # controls are scored as separate loss rows, never pooled
            losses[(sid, arm)] = (rmse(actual, predictions[(sid, arm)]), baseline)
    units, reasons, families = [], [], {}
    for dataset in specification["datasets"]:
        dataset_units = []
        for seed in specification["seeds"]:
            for outer in range(5):
                group = [sid for sid, s in searches.items()
                         if s["dataset"] == dataset["id"] and s["seed"] == seed and s["outer"] == outer]
                audit = [sid for sid in group if searches[sid]["audit"] is not None]
                final = [sid for sid in group if searches[sid]["audit"] is None]
                unit = {"dataset": dataset["id"], "family": dataset["family"], "seed": seed, "outer": outer}
                values = [(losses[(sid, "parent")][0], losses[(sid, "challenger")][0], losses[(sid, "parent")][1])
                          for sid in audit + final]
                if len(audit) != 5 or len(final) != 1 or any(b <= 0 or not math.isfinite(b) for _, _, b in values):
                    reasons.append(f"{dataset['id']} seed {seed} outer {outer}: invalid baseline or fold set")
                    units.append({**unit, "gains": None})
                    continue
                audit_gain = mean([(p - c) / b for p, c, b in values[:5]])
                p, c, b = values[5]
                outer_gain = (p - c) / b
                gains = {"audit": audit_gain, "outer": outer_gain, "contribution": min(audit_gain, outer_gain)}
                units.append({**unit, "gains": gains})
                dataset_units.append(gains)
        if len(dataset_units) == 5 * len(specification["seeds"]):
            families.setdefault(dataset["family"], []).append(
                {f: mean([u[f] for u in dataset_units]) for f in GAIN_FIELDS})
    family_means = {name: {f: mean([d[f] for d in ds]) for f in GAIN_FIELDS} for name, ds in families.items()}
    complete = not reasons and bool(family_means)
    kpi = 100 * mean([g["contribution"] for g in family_means.values()]) if complete else None
    return {"complete": complete, "kpi_percentage_points": kpi, "families": family_means,
            "units": units, "reasons": reasons, "losses": losses}


def close(a, b, tolerance=1e-9):
    return a is not None and b is not None and abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))


def compare(result, report):
    """Every disagreement between the independent score and the reporter, or an empty list."""
    problems = []
    if bool(report.get("ok")) != result["complete"]:
        problems.append(f"completeness: report ok={report.get('ok')} independent={result['complete']}")
    if not result["complete"]:
        if report.get("kpi_percentage_points") is not None:
            problems.append("report released a headline the independent score cannot complete")
        return problems
    if not close(result["kpi_percentage_points"], report.get("kpi_percentage_points")):
        problems.append(f"headline {report.get('kpi_percentage_points')} != {result['kpi_percentage_points']}")
    for name, gains in result["families"].items():
        theirs = report.get("families", {}).get(name, {})
        problems += [f"family {name} {f}" for f in GAIN_FIELDS if not close(gains[f], theirs.get(f))]
    reported = {(u["dataset"], u["seed"], u["outer"]): u.get("gains") for u in report.get("units", [])}
    for unit in result["units"]:
        theirs = reported.get((unit["dataset"], unit["seed"], unit["outer"]))
        if theirs is None or any(not close(unit["gains"][f], theirs.get(f)) for f in GAIN_FIELDS):
            problems.append(f"unit {unit['dataset']} seed {unit['seed']} outer {unit['outer']}")
    for row in report.get("losses", []):
        mine = result["losses"].get((row["search_sha256"], row["arm"]))
        if mine is not None and not (close(mine[0], row["rmse"]) and close(mine[1], row["training_mean_rmse"])):
            problems.append(f"loss {row['search_sha256']} {row['arm']}")
    if len(report.get("losses", [])) != len(result["losses"]):
        problems.append("loss row count differs")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = score(json.loads(args.plan.read_text()), json.loads(args.freeze.read_text()))
    output = {k: v for k, v in result.items() if k != "losses"}
    if args.report:
        output["disagreements"] = compare(result, json.loads(args.report.read_text()))
    print(json.dumps(output, sort_keys=True))
    return 1 if output.get("disagreements") else 0


if __name__ == "__main__":
    raise SystemExit(main())

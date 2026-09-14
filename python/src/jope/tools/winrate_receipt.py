"""Honest league win-rate receipt from existing paired rows. No serving writes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

from selfplay_milestones import BARS, descriptive_bars, existence_proof, win_rates, world_class
from scientific_metadata import immutable, metadata_bytes, read, read_with_digest

SCHEMA = "parallel.winrate_receipt.v2"
UNAVAILABLE = "unavailable"
MIN_PAIRS = 20


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha(path):
    return metadata_bytes(path)[1]


def load_json(path):
    return read(path)


def quality_rows(report):
    rows = report.get("rows") or report.get("allowlisted_rows") or []
    if isinstance(rows, dict):
        first = rows.get("first")
        return [first] if isinstance(first, dict) else []
    return [r for r in rows if isinstance(r, dict)]


def paired_summary(path):
    return paired_summary_with_digest(path)[0]


def paired_summary_with_digest(path):
    data, observed = metadata_bytes(path)
    counts = Counter()
    n = 0
    seen = {}
    with io.StringIO(data.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("paired diagnostic requires unique CSV column names")
        for row in reader:
            if None in row:
                raise ValueError("paired diagnostic has extra CSV values")
            keys = ("partition", "family", "cut", "rng_seed", "opponent")
            if any(not row.get(k) for k in keys):
                raise ValueError("paired diagnostic identity missing")
            key = tuple(row[k] for k in keys)
            if key in seen:
                if seen[key] != row:
                    raise ValueError("conflicting duplicate paired diagnostic")
                continue
            left, right = float(row["neural_rmse"]), float(row["opponent_rmse"])
            if not all(math.isfinite(v) and v >= 0 for v in (left, right)):
                raise ValueError("invalid paired diagnostic RMSE")
            expected = "win" if left < right else "loss" if left > right else "tie"
            if row.get("outcome") != expected:
                raise ValueError("paired diagnostic outcome conflicts with scores")
            seen[key] = row
            n += 1
            counts[(row.get("opponent"), row.get("outcome"))] += 1
    generic = dict(win=counts.get(("generic", "win"), 0),
                   loss=counts.get(("generic", "loss"), 0),
                   tie=counts.get(("generic", "tie"), 0))
    shuffled = dict(win=counts.get(("shuffled", "win"), 0),
                    loss=counts.get(("shuffled", "loss"), 0),
                    tie=counts.get(("shuffled", "tie"), 0))
    generic_n = sum(generic.values())
    return (dict(rows=n, generic=generic, shuffled=shuffled,
                 vs_generic_rate=(generic["win"] / generic_n) if generic_n else None), observed)


def later_vs_teacher(diagnostic, arm="neural"):
    block = ((diagnostic or {}).get("later_vs_teacher") or {}).get(arm) or {}
    pairs = block.get("pairs")
    better = block.get("final_better")
    equal, worse = block.get("final_equal"), block.get("final_worse")
    if (any(type(x) is not int or x < 0 for x in (pairs, better, equal, worse))
            or pairs <= 0 or better + equal + worse != pairs):
        return None
    return dict(pairs=pairs, final_better=better,
                final_equal=block.get("final_equal"),
                final_worse=block.get("final_worse"),
                rate=better / pairs)


def receipt(quality=None, diagnostic=None, paired=None, dest=None,
            quality_path=None, diagnostic_path=None, paired_path=None):
    def bound_report(value, path):
        if path is None:
            return value, None
        loaded, observed = read_with_digest(path)
        if value is not None and digest(value) != digest(loaded):
            raise ValueError("supplied report disagrees with its source bytes")
        return loaded, observed
    quality, quality_sha = bound_report(quality, quality_path)
    diagnostic, diagnostic_sha = bound_report(diagnostic, diagnostic_path)
    rows = quality_rows(quality or {})
    neural = [r for r in rows if r.get("arm") == "neural"]
    rates = win_rates(neural) if neural else None
    later = later_vs_teacher(diagnostic)
    paired_rows, paired_sha = paired_summary_with_digest(paired_path) if paired_path else (None, None)
    seed_rate = rates["neural_seed_rate"] if rates and rates["neural_seed_n"] else UNAVAILABLE
    final_rate = later["rate"] if later else (rates["final_gp_rate"] if rates else UNAVAILABLE)
    margin_rate = rates["final_gp_margin_rate"] if rates and rates["final_gp_n"] else UNAVAILABLE
    numeric = {
        "neural_seed_rate": seed_rate if seed_rate != UNAVAILABLE else 0.0,
        "final_gp_rate": final_rate if final_rate != UNAVAILABLE else 0.0,
        "final_gp_margin_rate": margin_rate if margin_rate != UNAVAILABLE else 0.0,
    }
    # world_class / existence require measured rates. Unavailable seed is not a 5% proof.
    measured_for_bars = rates if rates and rates["neural_seed_n"] and rates["final_gp_n"] else numeric
    enough = bool(rates and rates["neural_seed_n"] >= MIN_PAIRS and rates["final_gp_n"] >= MIN_PAIRS)
    if seed_rate == UNAVAILABLE or not enough:
        bars_ok = False
        exists = False
    else:
        bars_ok = world_class(measured_for_bars)
        exists = existence_proof(measured_for_bars)
    body = dict(
        schema=SCHEMA,
        sources=dict(
            quality_path=str(quality_path) if quality_path else None,
            quality_sha256=quality_sha,
            diagnostic_path=str(diagnostic_path) if diagnostic_path else None,
            diagnostic_sha256=diagnostic_sha,
            paired_path=str(paired_path) if paired_path else None,
            paired_sha256=paired_sha,
        ),
        quality_neural_rows=len(neural),
        win_rates=rates,
        later_vs_teacher=later,
        vs_generic=paired_rows["generic"] if paired_rows else None,
        vs_generic_rate=paired_rows["vs_generic_rate"] if paired_rows else None,
        vs_shuffled=paired_rows["shuffled"] if paired_rows else None,
        vs_random=UNAVAILABLE,
        neural_seed_rate=seed_rate,
        final_gp_rate=final_rate,
        final_gp_margin_rate=margin_rate,
        bars=BARS,
        existence_proof=exists,
        world_class=bars_ok,
        descriptive_bars_met=descriptive_bars(measured_for_bars),
        qualification_ready=False,
        unavailable_requirements=["original execution and selection/access verification",
                                  "independent family support and frozen scientific comparison"],
        independent_evaluation=False,
        serving_written=False,
        shuffled_retired=False,
        note=("final_gp_rate uses later_vs_teacher when quality rows are absent; "
              "margin and seed stay unavailable without paired teacher/seed/final magnitudes"),
    )
    body["identity"] = digest({k: body[k] for k in body if k != "identity"})
    if dest:
        dest = Path(dest)
        body["receipt_sha256"] = immutable(dest, body)
        body["receipt_path"] = str(dest)
    return body


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quality", type=Path)
    p.add_argument("--diagnostic", type=Path)
    p.add_argument("--paired", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = receipt(dest=args.output,
                  quality_path=args.quality, diagnostic_path=args.diagnostic,
                  paired_path=args.paired)
    print(json.dumps({k: out[k] for k in ("schema", "neural_seed_rate", "final_gp_rate",
                                          "final_gp_margin_rate", "vs_random",
                                          "world_class")}, indent=2))

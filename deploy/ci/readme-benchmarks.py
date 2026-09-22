#!/usr/bin/env python3
"""Render the README's measured-results blocks from committed evidence.

Two independent marker pairs, so regenerating one can never clobber the other:

    <!-- arena:begin --> ... <!-- arena:end -->     the Golden45 ladder
    <!-- bench:begin --> ... <!-- bench:end -->     emit-path latency

Nothing here is hand-typed. Every number is read from a file that is already under
test: `crates/jope-arena/tests/elo_regression.rs::committed_fixture_reproduces_the_published_ladder`
asserts the ladder recomputes from `scores.csv`, so this script does not re-derive it --
it formats what that test already guarantees, and records where it came from.

    readme-benchmarks.py --check    exit 1 if README.md is out of date (CI)
    readme-benchmarks.py --write    rewrite the blocks in place
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARENA = ROOT / "assets/regression/sotarena-tabular-regression-v2"
LEADERBOARD = ARENA / "results/tabular-regression-v2/leaderboard.csv"
TRACK = ARENA / "tracks/tabular-regression-v2/track.json"
SNAPSHOT = ROOT / "assets/regression/elo-tabular-regression-v2.json"
BENCH = ROOT / "assets/regression/bench-baseline.json"
README = ROOT / "README.md"

ARENA_BEGIN, ARENA_END = "<!-- arena:begin -->", "<!-- arena:end -->"
BENCH_BEGIN, BENCH_END = "<!-- bench:begin -->", "<!-- bench:end -->"

# Where the section is inserted the first time, if the markers are not present yet.
ANCHOR = "## Install and check"
TOP_N = 5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def render_arena() -> str:
    track = json.loads(TRACK.read_text())
    snap = json.loads(SNAPSHOT.read_text())
    rows = list(csv.DictReader(LEADERBOARD.read_text().splitlines()))

    # The Elo snapshot records the hash of the ladder it was derived from. If the committed
    # leaderboard has moved since, the two files disagree and neither is safe to publish.
    # Fail closed rather than render a number whose provenance is already broken.
    recorded = snap.get("leaderboard_sha256")
    actual = sha256(LEADERBOARD)
    if recorded and recorded != actual:
        raise SystemExit(
            f"leaderboard.csv sha256 {actual}\n"
            f"does not match {SNAPSHOT.relative_to(ROOT)} leaderboard_sha256 {recorded}.\n"
            "The committed benchmark evidence disagrees with itself; refusing to publish."
        )

    split = track.get("split", {})
    policy = track.get("leaderboard_policy", {})
    excluded = ", ".join(split.get("excluded_datasets", [])) or "none"

    out = [
        ARENA_BEGIN,
        f"### {track['public_name']} — {track['task']}, {snap['datasets_scored']} datasets",
        "",
        f"`{track['id']}` · primary metric **{track['primary_metric'].upper()}** "
        f"({track['metric_direction']}) · **{snap['methods']} methods** · "
        f"{snap['rankable_rows']:,} rankable scores · "
        f"Elo by `{policy.get('elo_method', 'unknown')}`.",
        "",
        "| rank | method | Elo | avg rank | dataset wins | coverage | protocol |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for r in rows[:TOP_N]:
        lead = r["method_id"] == "jope-dime"
        name = f"**{r['method']}**" if lead else r["method"]
        rank = f"**{r['elo_rank']}**" if lead else r["elo_rank"]
        elo = f"**{r['elo']}**" if lead else r["elo"]
        out.append(
            f"| {rank} | {name} | {elo} | {float(r['average_rank']):.2f} | "
            f"{r['dataset_wins']} | {r['datasets_scored']}/{r['datasets_total']} | "
            f"`{r['protocols']}` |"
        )

    out += [
        "",
        "**Read the protocol column before the Elo column.** "
        f"`leaderboard_policy.protocols_ranked_together` is "
        f"`{str(policy.get('protocols_ranked_together', False)).lower()}`, so entries produced under "
        "different protocols share one ladder. `jope-dime` is the only `aspfm-generated` entry; every "
        "comparator above is `public-development`. Read this as a standing on a shared ladder "
        "whose entries were generated under different protocols, not as a like-for-like "
        "held-out comparison.",
        "",
        f"Also true of this track: test labels are "
        f"{'public' if split.get('test_labels_public') else '**not public**'}; "
        f"{split.get('restricted_dataset_count', 0)} datasets are restricted; "
        f"excluded datasets: {excluded}.",
        "",
        f"Source `{LEADERBOARD.relative_to(ROOT)}` (sha256 `{sha256(LEADERBOARD)[:16]}…`), "
        "recomputed from `scores.csv` by "
        "`crates/jope-arena/tests/elo_regression.rs`. Reproduce offline:",
        "",
        "```sh",
        "cargo run -p jope-arena --bin jope-arena -- elo \\",
        f"  --track {track['id']} \\",
        f"  --root {ARENA.relative_to(ROOT)} --check",
        "```",
        ARENA_END,
    ]
    return "\n".join(out)


def render_bench() -> str:
    b = json.loads(BENCH.read_text())
    return "\n".join(
        [
            BENCH_BEGIN,
            "### Emit-path latency",
            "",
            f"`{b['bench']}` — {b['runs']} runs of the release `emit` example on "
            f"`{b['genome']}`.",
            "",
            "| min | median | mean | p95 |",
            "| ---: | ---: | ---: | ---: |",
            f"| {b['min_ms']:.2f} ms | {b['median_ms']:.2f} ms | {b['mean_ms']:.2f} ms "
            f"| {b['p95_ms']:.2f} ms |",
            "",
            "**This is not an engine-speed number.** Its own `what` field says it measures "
            f"\"{b['what']}\". The min-to-p95 spread is dominated by process spawn and "
            "scheduling, not by emission. Treat it as a regression tripwire "
            f"(alert at {b['alert_pct']:.0f}% drift), not as a performance claim.",
            "",
            f"Recorded {b['recorded']} on `{b['host']}` "
            f"({b['cpu']}, {b['nproc']} cores, kernel {b['kernel']}) at `{b['git_sha'][:9]}` "
            f"with {b['rustc']}. Baselines are host-pinned; "
            f"`bash deploy/ci/regression-suite.sh bench` compares against this file.",
            BENCH_END,
        ]
    )


def splice(text: str, begin: str, end: str, block: str) -> str:
    i, j = text.find(begin), text.find(end)
    if i == -1 or j == -1:
        raise SystemExit(f"markers {begin}/{end} not found; cannot splice")
    return text[:i] + block + text[j + len(end) :]


def build(text: str) -> str:
    if ARENA_BEGIN not in text:
        anchor = text.find(ANCHOR)
        if anchor == -1:
            raise SystemExit(f"anchor {ANCHOR!r} not found in README")
        head = git("rev-parse", "--short", "HEAD")
        section = (
            "## Measured results\n\n"
            f"Generated by `deploy/ci/readme-benchmarks.py` from committed evidence at `{head}`. "
            "Do not edit between the markers by hand.\n\n"
            f"{ARENA_BEGIN}\n{ARENA_END}\n\n{BENCH_BEGIN}\n{BENCH_END}\n\n"
        )
        text = text[:anchor] + section + text[anchor:]
    text = splice(text, ARENA_BEGIN, ARENA_END, render_arena())
    return splice(text, BENCH_BEGIN, BENCH_END, render_bench())


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    args = ap.parse_args()

    for p in (LEADERBOARD, TRACK, SNAPSHOT, BENCH, README):
        if not p.exists():
            raise SystemExit(f"missing required input: {p}")

    current = README.read_text()
    updated = build(current)

    if args.write:
        if updated != current:
            README.write_text(updated)
            print("README.md updated")
        else:
            print("README.md already current")
        return 0

    if updated != current:
        print(
            "README.md is out of date with the committed benchmark evidence.\n"
            "Run: python3 deploy/ci/readme-benchmarks.py --write",
            file=sys.stderr,
        )
        return 1
    print("README.md benchmark blocks are current")
    return 0


if __name__ == "__main__":
    sys.exit(main())

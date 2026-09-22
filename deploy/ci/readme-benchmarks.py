#!/usr/bin/env python3
"""Render the README's measured-results blocks from committed evidence.

Two independent marker pairs, so regenerating one can never clobber the other:

    <!-- arena:begin --> ... <!-- arena:end -->     the Golden45 ladders
    <!-- bench:begin --> ... <!-- bench:end -->     emit-path latency

Nothing here is hand-typed. Every number is read from a file that is already under
test: `crates/jope-arena/tests/elo_regression.rs` asserts that both ladders recompute from
`scores.csv` (the published one byte-for-byte, the benchmark-valid one with the committed
protocol corrections applied), so this script does not re-derive them -- it formats what those
tests already guarantee, records where every number came from, and refuses to render when the
committed evidence disagrees with itself.

Two ladders, deliberately:

* **benchmark-valid** is the number this repository claims. Rows whose source run this
  repository's own campaign code labels test-adaptive are excluded before the Elo fit
  (`assets/regression/protocol-corrections-tabular-regression-v2.json`, every entry with cited
  evidence).
* **published** is what the benchmark ranked, all rows included. It is rendered as the best
  known program per dataset and labelled as not a held-out result.

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
REGRESSION = ROOT / "assets/regression"
ARENA = REGRESSION / "sotarena-tabular-regression-v2"
TRACK = ARENA / "tracks/tabular-regression-v2/track.json"
# The published ladder: a byte-exact copy of the benchmark's own output.
LEADERBOARD = ARENA / "results/tabular-regression-v2/leaderboard.csv"
SNAPSHOT = REGRESSION / "elo-tabular-regression-v2.json"
# The benchmark-valid ladder: the same scores.csv with the corrections applied.
VALID_LEADERBOARD = REGRESSION / "leaderboard-tabular-regression-v2.benchmark-valid.csv"
VALID_SNAPSHOT = REGRESSION / "elo-tabular-regression-v2.benchmark-valid.json"
CORRECTIONS = REGRESSION / "protocol-corrections-tabular-regression-v2.json"
BENCH = REGRESSION / "bench-baseline.json"
README = ROOT / "README.md"

SNAPSHOT_SCHEMA = "jope.arena-elo-snapshot/v2"
CORRECTIONS_SCHEMA = "jope.arena-protocol-corrections/v1"
# This repository's own entries: the campaign method and the fixed-seed reproducible run.
OURS = ("jope-dime", "jope-dime-regressor")

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


def refuse(why: str) -> None:
    raise SystemExit(
        f"{why}\nThe committed benchmark evidence disagrees with itself; refusing to publish."
    )


def load_ladder(snapshot_path: Path, csv_path: Path, ladder: str) -> tuple[dict, list[dict]]:
    """A snapshot and the CSV it hashes, cross-checked. Fails closed on any disagreement."""
    snap = json.loads(snapshot_path.read_text())
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    rel = snapshot_path.relative_to(ROOT)
    if snap.get("schema") != SNAPSHOT_SCHEMA:
        refuse(f"{rel}: schema {snap.get('schema')!r} is not {SNAPSHOT_SCHEMA}")
    if snap.get("ladder") != ladder:
        refuse(f"{rel}: ladder {snap.get('ladder')!r} is not {ladder!r}")
    # The snapshot records the hash of the ladder it was derived from. If the committed CSV has
    # moved since, the two files disagree and neither is safe to publish.
    actual = sha256(csv_path)
    if snap.get("leaderboard_sha256") != actual:
        refuse(
            f"{csv_path.relative_to(ROOT)} sha256 {actual}\n"
            f"does not match {rel} leaderboard_sha256 {snap.get('leaderboard_sha256')}."
        )
    if len(rows) != snap["methods"]:
        refuse(f"{rel}: {snap['methods']} methods but the CSV has {len(rows)} rows")
    return snap, rows


def load_evidence() -> dict:
    track = json.loads(TRACK.read_text())
    published, published_rows = load_ladder(SNAPSHOT, LEADERBOARD, "published")
    valid, valid_rows = load_ladder(VALID_SNAPSHOT, VALID_LEADERBOARD, "benchmark-valid")
    corrections = json.loads(CORRECTIONS.read_text())

    if published["corrections"] is not None or published["excluded_rows"] != 0:
        refuse(f"{SNAPSHOT.relative_to(ROOT)}: the published ladder must apply no corrections")
    if corrections.get("schema") != CORRECTIONS_SCHEMA:
        refuse(f"{CORRECTIONS.relative_to(ROOT)}: schema is not {CORRECTIONS_SCHEMA}")
    if corrections["scores_sha256"] != published["scores_sha256"] != valid["scores_sha256"]:
        refuse("the two ladders and the corrections do not bind the same scores.csv")
    meta = valid["corrections"]
    if meta is None or meta["sha256"] != sha256(CORRECTIONS):
        refuse(
            f"{VALID_SNAPSHOT.relative_to(ROOT)} was not computed with the committed "
            f"{CORRECTIONS.relative_to(ROOT)}"
        )
    if valid["excluded_rows"] < meta["rows_corrected"]:
        refuse("the benchmark-valid ladder excluded fewer rows than the corrections demoted")
    if any(r["benchmark_valid"] != "True" for r in valid_rows):
        refuse("a benchmark-invalid method survived into the benchmark-valid ladder")

    # Every correction must show up as fewer scored datasets for its method, so re-adding the
    # rows, or editing the corrections without regenerating the ladder, fails here.
    applied = {a["id"]: a["rows"] for a in meta["applied"]}
    if set(applied) != {c["id"] for c in corrections["corrections"]}:
        refuse("the corrections applied to the ladder are not the committed corrections")
    corrected_by_method: dict[str, int] = {}
    for c in corrections["corrections"]:
        corrected_by_method[c["method_id"]] = (
            corrected_by_method.get(c["method_id"], 0) + applied[c["id"]]
        )
    scored_published = {r["method_id"]: int(r["datasets_scored"]) for r in published_rows}
    scored_valid = {r["method_id"]: int(r["datasets_scored"]) for r in valid_rows}
    for method, n in corrected_by_method.items():
        if scored_published.get(method, 0) - scored_valid.get(method, 0) != n:
            refuse(
                f"{method}: {n} rows corrected but scored datasets went "
                f"{scored_published.get(method, 0)} -> {scored_valid.get(method, 0)}"
            )

    return {
        "track": track,
        "published": published,
        "published_rows": published_rows,
        "valid": valid,
        "valid_rows": valid_rows,
        "corrections": corrections,
        "corrected_by_method": corrected_by_method,
    }


def table_row(r: dict) -> str:
    ours = r["method_id"] in OURS
    name = f"**{r['method']}**" if ours else r["method"]
    rank = f"**{r['elo_rank']}**" if ours else r["elo_rank"]
    elo = f"**{r['elo']}**" if ours else r["elo"]
    return (
        f"| {rank} | {name} | {elo} | {float(r['average_rank']):.2f} | "
        f"{r['dataset_wins']} | {r['datasets_scored']}/{r['datasets_total']} | "
        f"`{r['protocols']}` |"
    )


def ladder_table(rows: list[dict]) -> list[str]:
    """The top N plus every one of our entries, in rank order, with an ellipsis row per gap."""
    shown = set(range(TOP_N))
    for i, r in enumerate(rows):
        if r["method_id"] in OURS:
            shown.add(i)
    out = [
        "| rank | method | Elo | avg rank | dataset wins | coverage | protocol |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    previous = -1
    for i in sorted(shown):
        if i > previous + 1:
            out.append("| … | | | | | | |")
        out.append(table_row(rows[i]))
        previous = i
    return out


def render_arena() -> str:
    e = load_evidence()
    track, corrections = e["track"], e["corrections"]
    published, valid = e["published"], e["valid"]
    split = track.get("split", {})
    policy = track.get("leaderboard_policy", {})
    excluded = ", ".join(split.get("excluded_datasets", [])) or "none"
    corrected = valid["corrections"]["rows_corrected"]
    published_ours = {r["method_id"]: r for r in e["published_rows"] if r["method_id"] in OURS}
    valid_ours = {r["method_id"]: r for r in e["valid_rows"] if r["method_id"] in OURS}
    run_ids = sorted({c["source_revision"] for c in corrections["corrections"] if c.get("source_revision")})
    evidence = corrections["corrections"][0]["evidence"][0].split(" ")[0]

    out = [
        ARENA_BEGIN,
        f"### {track['public_name']} — {track['task']}, {published['datasets_scored']} datasets",
        "",
        f"`{track['id']}` · primary metric **{track['primary_metric'].upper()}** "
        f"({track['metric_direction']}) · Elo by `{policy.get('elo_method', 'unknown')}` · "
        f"test labels are {'public' if split.get('test_labels_public') else '**not public**'}; "
        f"{split.get('restricted_dataset_count', 0)} datasets are restricted; "
        f"excluded datasets: {excluded}.",
        "",
        "#### Held-out, benchmark-valid — the ladder this repository claims",
        "",
        f"**{valid['methods']} methods** · {valid['rankable_rows']:,} rankable scores · "
        f"{corrected} published rows excluded as test-adaptive (see the next block). "
        "Bold rows are this repository's entries.",
        "",
        *ladder_table(e["valid_rows"]),
        "",
        "**Read the protocol column before the Elo column.** "
        f"`leaderboard_policy.protocols_ranked_together` is "
        f"`{str(policy.get('protocols_ranked_together', False)).lower()}`, so entries produced under "
        "different protocols share one ladder. `jope-dime` is the only `aspfm-generated` entry "
        "(a program written per dataset); `jope-dime-regressor` is one fixed-seed reproducible run "
        "under `public-development`, the same protocol as every other comparator. Read the ladder "
        "as a standing among entries generated under different protocols, not as a like-for-like "
        "held-out comparison; the same-cut benchmark is what makes it one.",
        "",
        "#### Best known program per dataset — as published; test-adaptive, not a held-out result",
        "",
    ]
    p = published_ours.get("jope-dime")
    v = valid_ours.get("jope-dime")
    if p is None or v is None:
        refuse("jope-dime is missing from a ladder")
    out += [
        f"The benchmark published `jope-dime` at rank {p['elo_rank']}, Elo {p['elo']}, "
        f"{p['dataset_wins']} dataset wins, {p['datasets_scored']}/{p['datasets_total']} coverage, "
        f"every row `benchmark_valid: True`. {corrected} of those rows come from run "
        f"`{'`, `'.join(run_ids)}`, whose artefacts this repository's own campaign code writes with "
        f"`benchmark_valid: false, test_adaptive: true` (`{evidence}`); the acceptance manifest "
        "relabelled the replayed winners `true`. Those rows are the best program found per dataset "
        "when candidates were scored on the test targets. They are reported here as best-known "
        "results and are excluded from the ladder above; they are never a held-out claim.",
        "",
        f"The other {v['datasets_scored']} `jope-dime` rows are the best value of a per-dataset "
        "evolution ledger (`source_kind: complete-evolution-ledger-snapshot`). This repository "
        "holds no artefact contradicting their published flag, and none proving the selection was "
        "frozen before the test targets were opened, so they stay as published. The same-cut "
        "benchmark replaces every row above with numbers this tree can reproduce; until then, "
        "*benchmark-valid* means \"no contradicting evidence here\", not \"verified held-out\".",
        "",
        "| ladder | `jope-dime` rank | Elo | dataset wins | coverage |",
        "| --- | ---: | ---: | ---: | --- |",
        f"| as published | {p['elo_rank']} | {p['elo']} | {p['dataset_wins']} | "
        f"{p['datasets_scored']}/{p['datasets_total']} |",
        f"| benchmark-valid | {v['elo_rank']} | {v['elo']} | {v['dataset_wins']} | "
        f"{v['datasets_scored']}/{v['datasets_total']} |",
        "",
        f"Sources: published `{LEADERBOARD.relative_to(ROOT)}` (sha256 "
        f"`{sha256(LEADERBOARD)[:16]}…`), recomputed byte-for-byte from `scores.csv`; benchmark-valid "
        f"`{VALID_LEADERBOARD.relative_to(ROOT)}` (sha256 `{sha256(VALID_LEADERBOARD)[:16]}…`), "
        f"the same `scores.csv` with `{CORRECTIONS.relative_to(ROOT)}` (sha256 "
        f"`{sha256(CORRECTIONS)[:16]}…`) applied. Both are asserted by "
        "`crates/jope-arena/tests/elo_regression.rs`. Reproduce offline:",
        "",
        "```sh",
        "cargo run -p jope-arena --bin jope-arena -- elo \\",
        f"  --track {track['id']} --root {ARENA.relative_to(ROOT)} --check",
        "cargo run -p jope-arena --bin jope-arena -- elo \\",
        f"  --track {track['id']} --root {ARENA.relative_to(ROOT)} \\",
        f"  --ladder benchmark-valid --corrections {CORRECTIONS.relative_to(ROOT)} \\",
        f"  --check --committed {VALID_LEADERBOARD.relative_to(ROOT)}",
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

    for p in (
        LEADERBOARD,
        TRACK,
        SNAPSHOT,
        VALID_LEADERBOARD,
        VALID_SNAPSHOT,
        CORRECTIONS,
        BENCH,
        README,
    ):
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

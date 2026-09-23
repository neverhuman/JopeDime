#!/usr/bin/env python3
"""Render the README's measured-results blocks from committed evidence.

Three independent marker pairs, so regenerating one can never clobber another:

    <!-- arena:begin --> ... <!-- arena:end -->       the Golden45 ladders
    <!-- samecut:begin --> ... <!-- samecut:end -->   the same-cut ladder this tree reproduces
    <!-- bench:begin --> ... <!-- bench:end -->       emit-path latency

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

# The same-cut evidence: `crates/jope-bench/tests/committed_evidence.rs` asserts that the tracks
# re-derive from the reports and both ladders from the tracks, byte for byte.
SAMECUT = ROOT / "assets/same-cut/v1"
SAMECUT_REPORTS = SAMECUT / "reports"
SAMECUT_TRACK_ID = "jope-same-cut-v1"
SAMECUT_DEEP_CSV = SAMECUT / f"leaderboard-{SAMECUT_TRACK_ID}-deep.csv"
SAMECUT_DEEP_SNAPSHOT = SAMECUT / f"elo-{SAMECUT_TRACK_ID}-deep.json"
SAMECUT_SCREEN_CSV = SAMECUT / f"leaderboard-{SAMECUT_TRACK_ID}-screen.csv"
SAMECUT_SCREEN_SNAPSHOT = SAMECUT / f"elo-{SAMECUT_TRACK_ID}-screen.json"
SAMECUT_DEEP_TRACK = SAMECUT / f"track/tracks/{SAMECUT_TRACK_ID}-deep/track.json"
SAMECUT_DEEP_SCORES = SAMECUT / f"track/results/{SAMECUT_TRACK_ID}-deep/scores.csv"
SAMECUT_DOC = ROOT / "docs/same-cut-bench.md"
REPORT_SCHEMA = "jope.bench-report/v2"
# The searched program's ladder id (the deep stage re-runs the search on every cut).
SEARCHED = "afm.lite-race.v1.resampled"
# The owner's stated reference point for a large language model writing a program for the same
# task; rendered as a cost comparison, never as a measured or quality claim.
LLM_REFERENCE_HOURS = 1
LLM_REFERENCE_TOKENS = 10_000_000

SNAPSHOT_SCHEMA = "jope.arena-elo-snapshot/v2"
CORRECTIONS_SCHEMA = "jope.arena-protocol-corrections/v1"
# This repository's own entries: the campaign method and the fixed-seed reproducible run.
OURS = ("jope-dime", "jope-dime-regressor")

ARENA_BEGIN, ARENA_END = "<!-- arena:begin -->", "<!-- arena:end -->"
SAMECUT_BEGIN, SAMECUT_END = "<!-- samecut:begin -->", "<!-- samecut:end -->"
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
    if not (corrections["scores_sha256"] == published["scores_sha256"] == valid["scores_sha256"]):
        refuse("the two ladders and the corrections do not bind the same scores.csv")
    meta = valid["corrections"]
    if meta is None or meta["sha256"] != sha256(CORRECTIONS):
        refuse(
            f"{VALID_SNAPSHOT.relative_to(ROOT)} was not computed with the committed "
            f"{CORRECTIONS.relative_to(ROOT)}"
        )
    if valid["excluded_rows"] != meta["rows_corrected"]:
        # Equal while every published row is True. If the benchmark ever publishes a False row,
        # the ladder excludes it too and this sentence must be rewritten, not silently under-counted.
        refuse(
            f"the benchmark-valid ladder excluded {valid['excluded_rows']} rows but the corrections "
            f"demoted {meta['rows_corrected']}; the README prose assumes they are the same set"
        )
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
    excluded_rows = valid["excluded_rows"]
    published_ours = {r["method_id"]: r for r in e["published_rows"] if r["method_id"] in OURS}
    valid_ours = {r["method_id"]: r for r in e["valid_rows"] if r["method_id"] in OURS}
    applied = {a["id"]: a["rows"] for a in valid["corrections"]["applied"]}
    by_id = {c["id"]: c for c in corrections["corrections"]}
    golden46 = by_id.get("golden46-new3-20260723-v4-test-adaptive")
    golden42 = by_id.get("golden42-evolution-envelope-test-selected")
    if golden46 is None or golden42 is None:
        refuse("the README prose is written for the two committed corrections; update both together")
    first_ref = lambda c: c["evidence"][0].split(" ")[0]  # noqa: E731

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
        "#### Benchmark-valid — the ladder this repository claims",
        "",
        f"**{valid['methods']} methods** · {valid['rankable_rows']:,} rankable scores · "
        f"{excluded_rows} published rows excluded as test-selected (see the next block). "
        "Bold rows are this repository's entries. *Benchmark-valid* here means that no artefact in "
        "this repository contradicts the row's published flag; it is not a claim that every "
        "comparator's protocol was verified.",
        "",
        *ladder_table(e["valid_rows"]),
        "",
        "**Read the protocol column before the Elo column.** "
        f"`leaderboard_policy.protocols_ranked_together` is "
        f"`{str(policy.get('protocols_ranked_together', False)).lower()}`, so entries produced under "
        "different protocols share one ladder. `jope-dime-regressor` is one fixed-seed reproducible "
        "run of this repository's engine under `public-development`, the same protocol as every "
        "other comparator. The same-cut benchmark is what turns this into a like-for-like "
        "comparison.",
        "",
        "#### Best known program per dataset — as published; test-selected, not a held-out result",
        "",
    ]
    p = published_ours.get("jope-dime")
    if p is None:
        refuse("jope-dime is missing from the published ladder")
    v = valid_ours.get("jope-dime")
    if v is not None:
        # The prose below is written for the case where every jope-dime row is corrected. A
        # corrections file that leaves some of them valid needs the block rewritten, not rendered.
        refuse(
            "jope-dime still has benchmark-valid rows; the best-known block's prose assumes it "
            "has none"
        )
    out += [
        f"The benchmark published `jope-dime` at rank {p['elo_rank']}, Elo {p['elo']}, "
        f"{p['dataset_wins']} dataset wins, {p['datasets_scored']}/{p['datasets_total']} coverage, "
        f"every row `benchmark_valid: True`. This repository's own code says otherwise about every "
        "one of those rows, so the benchmark-valid ladder above carries none of them:",
        "",
        f"- **{applied[golden42['id']]} Golden42 rows** are each the per-dataset minimum RMSE over "
        "the ledger's test-scored candidate evaluations (63,108 across the 42 datasets), "
        "not the score of one program frozen before the test targets were opened. The red-team "
        "lane in this tree names "
        "that quantity the *best-test ASPFM envelope*, role `oracle_reference_only`, and keeps it "
        f"out of its fair ladder (`{first_ref(golden42)}`, "
        "`crates/jope-campaigns/src/commands/train_only_redteam.rs:402-407`).",
        f"- **{applied[golden46['id']]} Golden46 rows** come from run "
        f"`{golden46['source_revision']}`, whose artefacts this repository's campaign code writes "
        f"with `benchmark_valid: false, test_adaptive: true` (`{first_ref(golden46)}`); the "
        "acceptance manifest relabelled the replayed winners `true`.",
        "",
        "Both are the best program found per dataset when candidates were scored on the test "
        "targets: reported here as best-known results, never as a held-out claim. The reproducible "
        "fixed-seed run of the same engine, `jope-dime-regressor`, is what stands on the ladder "
        "above. The same-cut benchmark replaces every row on this page with numbers this tree can "
        "reproduce.",
        "",
        "| ladder | `jope-dime` | Elo | dataset wins | coverage |",
        "| --- | ---: | ---: | ---: | --- |",
        f"| as published | rank {p['elo_rank']} | {p['elo']} | {p['dataset_wins']} | "
        f"{p['datasets_scored']}/{p['datasets_total']} |",
        f"| benchmark-valid | not ranked | — | — | 0/{p['datasets_total']} (every row test-selected) |",
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


def load_samecut() -> dict:
    """The committed same-cut evidence, cross-checked. Fails closed on any disagreement."""
    deep, deep_rows = load_ladder(SAMECUT_DEEP_SNAPSHOT, SAMECUT_DEEP_CSV, "published")
    screen, screen_rows = load_ladder(SAMECUT_SCREEN_SNAPSHOT, SAMECUT_SCREEN_CSV, "published")
    track = json.loads(SAMECUT_DEEP_TRACK.read_text())
    rel = SAMECUT.relative_to(ROOT)
    if deep["scores_sha256"] != sha256(SAMECUT_DEEP_SCORES):
        refuse(f"{rel}: the deep snapshot was not computed from the committed scores.csv")
    if track["id"] != f"{SAMECUT_TRACK_ID}-deep" or track["primary_metric"] != "rmse":
        refuse(f"{rel}: the deep track is not {SAMECUT_TRACK_ID}-deep ranked by rmse")
    if deep["corrections"] is not None or deep["excluded_rows"] != 0:
        refuse(f"{rel}: the same-cut ladder applies no corrections")
    reports = [
        json.loads(p.read_text()) for p in sorted(SAMECUT_REPORTS.glob("*.bench-report.json"))
    ]
    if not reports:
        refuse(f"{rel}: no reports")
    if not (len(reports) == deep["datasets_scored"] == screen["datasets_scored"]):
        refuse(f"{rel}: {len(reports)} reports but the ladders scored other dataset counts")
    first = reports[0]
    for r in reports:
        if r["schema"] != REPORT_SCHEMA:
            refuse(f"{rel}: report schema {r['schema']!r} is not {REPORT_SCHEMA}")
        if (r["task"], r["ladder_metric"], r["metric_direction"]) != ("regression", "rmse", "minimize"):
            refuse(f"{rel}: {r['dataset_id']} is not a regression report ranked by rmse")
        for key in ("protocol", "search", "git_sha", "host", "evaluator"):
            if r.get(key) != first.get(key) or r.get(key) is None:
                refuse(f"{rel}: {r['dataset_id']} differs from the first report in {key}")
        # The plan key names the dataset; seed and fold count are what one run shares.
        if (r["plan"]["seed"], r["plan"]["k"]) != (first["plan"]["seed"], first["plan"]["k"]):
            refuse(f"{rel}: {r['dataset_id']} differs from the first report in fold seed or k")
        searched = [m for m in r["methods"] if m["method"]["kind"] == "search"]
        if len(searched) != 1 or searched[0]["method"]["id"] != SEARCHED:
            refuse(f"{rel}: {r['dataset_id']} does not carry exactly one {SEARCHED} row")
        if searched[0]["method"].get("resample") != r["search"]:
            refuse(f"{rel}: {r['dataset_id']}: the searched row is not the resampled procedure")
    ours = next((row for row in deep_rows if row["method_id"] == SEARCHED), None)
    if ours is None:
        refuse(f"{rel}: {SEARCHED} is missing from the deep ladder")
    return {
        "deep": deep,
        "deep_rows": deep_rows,
        "screen": screen,
        "screen_rows": screen_rows,
        "track": track,
        "reports": reports,
        "ours": ours,
    }


def samecut_row(r: dict) -> str:
    ours = r["method_id"] == SEARCHED
    b = "**" if ours else ""
    return (
        f"| {b}{r['elo_rank']}{b} | {b}{r['method']}{b} | {b}{r['elo']}{b} | "
        f"{float(r['average_rank']):.2f} | {r['dataset_wins']} | "
        f"{r['datasets_scored']}/{r['datasets_total']} |"
    )


def render_samecut() -> str:
    e = load_samecut()
    deep, reports, ours = e["deep"], e["reports"], e["ours"]
    first = reports[0]
    plan, protocol, search, host = first["plan"], first["protocol"], first["search"], first["host"]
    n = len(reports)
    per_dataset = []
    search_seconds = []
    finalist = 0
    best = 0
    for r in sorted(reports, key=lambda r: r["dataset_id"]):
        s = next(m for m in r["methods"] if m["method"]["kind"] == "search")
        # "Best" is the ladder's own rule: the lowest pooled RMSE among the finalists that
        # scored, ties by method id. (The protocol's deep rank orders by the penalised score,
        # which can disagree with RMSE on a close dataset; the table says which rule it uses.)
        scored = [m for m in r["methods"] if m.get("deep") and m["deep"]["status"] == "ok"]
        if not scored:
            refuse(f"{SAMECUT.relative_to(ROOT)}: {r['dataset_id']} has no scored finalist")
        top = min(scored, key=lambda m: (m["deep"]["oof"]["rmse"], m["method"]["id"]))
        rmse = top["deep"]["oof"]["rmse"]
        search_seconds.append(s["method"]["produce_seconds"])
        rank = s.get("deep_rank")
        if rank is not None:
            finalist += 1
        best += top is s
        per_dataset.append(
            f"| `{r['dataset_id']}` | {r['rows']} | {r['features']} | "
            f"{'**' if top is s else ''}{top['method']['id']}{'**' if top is s else ''} | "
            f"{rmse:.4g} | {rank if rank is not None else f'screen {s['screen_rank']}'} | "
            f"{s['method']['produce_seconds']:.1f} s |"
        )
    mean_s = sum(search_seconds) / n
    median_s = sorted(search_seconds)[n // 2] if n % 2 else sum(sorted(search_seconds)[n // 2 - 1 : n // 2 + 1]) / 2
    max_s = max(search_seconds)
    out = [
        SAMECUT_BEGIN,
        f"### Same-cut — every method on identical held-out rows, reproducible from this tree",
        "",
        f"`{deep['track_id']}` · {n} regression datasets from `datasets/test` · primary metric "
        f"**RMSE** (minimize) · fold plan k={plan['k']}, seed {plan['seed']}, screen fold "
        f"{protocol['screen_fold']}, top {protocol['top']} deepened · search budget {search['pop']} "
        f"programs × {search['gens']} rounds, re-run on every cut · recorded at "
        f"`{first['git_sha'][:9]}` with {host['workers']} workers on {host['cores']} cores.",
        "",
        "Every row here is a number this tree produced and can reproduce: the ten single-head "
        "baselines and the program the search wrote for each dataset, all fitted on the same "
        "training rows and scored on the same held-out rows by one evaluator "
        f"(`{SAMECUT_DOC.relative_to(ROOT)}`). The search sees only a cut's training rows, and its "
        "deep row is the *procedure* re-run on every cut, not one program chosen with knowledge "
        "of the other cuts' held-out rows. Bold is the searched program.",
        "",
        "**The datasets are a favourable selection.** They are the `test-10` collection, chosen "
        "from the published ladder where the historical engine ranked first "
        "(`docs/datasets.md`), and they are used here because they are the only committed "
        "regression data. Identical cuts make the comparison between methods fair on these ten; "
        "they say nothing about a broader panel. The frozen 30-dataset PMLB panel of the "
        "experiment contract (`docs/asm.md`) excludes them on purpose; a same-cut run over it is "
        "the next evidence step.",
        "",
        "| rank | method | Elo | avg rank | dataset wins | coverage |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
        *(samecut_row(r) for r in e["deep_rows"]),
        "",
        f"**Where the search stands today:** rank {ours['elo_rank']} of {deep['methods']} on the "
        f"deep ladder, Elo {ours['elo']}, lowest pooled RMSE on {best} of {n} datasets (the "
        f"ladder's own rule), a finalist on {finalist}; on the other {n - finalist} it did not "
        f"make the top-{protocol['top']} screen. Coverage counts datasets where a method reached "
        "the deep stage, so the ladder ranks finalists only. The screen ladder (every method, "
        "one cut) is committed beside it.",
        "",
        f"**Seconds per program.** Writing the program took the search {mean_s:.1f} s per dataset "
        f"on average (median {median_s:.1f} s, longest {max_s:.1f} s) on the screen cut's training "
        f"rows, wall time under whatever else the host was running. The owner's stated reference "
        f"point for a large language model writing a program for the same task, not measured "
        f"here, is about {LLM_REFERENCE_HOURS} hour and {LLM_REFERENCE_TOKENS:,} tokens; that "
        "comparison is one of cost, not of quality, and the quality standing is the ladder above.",
        "",
        "| dataset | rows | features | lowest pooled RMSE | RMSE | search rank (penalised score) | search |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
        *per_dataset,
        "",
        f"Sources: `{SAMECUT_DEEP_CSV.relative_to(ROOT)}` (sha256 `{sha256(SAMECUT_DEEP_CSV)[:16]}…`) "
        f"recomputed byte-for-byte from `{SAMECUT_DEEP_SCORES.relative_to(ROOT)}`, itself "
        f"re-derived from the {n} reports under `{SAMECUT_REPORTS.relative_to(ROOT)}` by "
        "`crates/jope-bench/tests/committed_evidence.rs`, which also re-evaluates one dataset's "
        f"baselines on the same cuts. Provenance and the exact commands: `{SAMECUT.relative_to(ROOT)}/SOURCES.md`. "
        "Reproduce:",
        "",
        "```sh",
        "cargo run --release -p jope-bench -- run --datasets datasets/test --out target/same-cut/reports \\",
        f"  --top {protocol['top']} --search --search-pop {search['pop']} --search-gens {search['gens']} "
        f"--resample-search --workers {host['workers']}",
        "cargo run --release -p jope-bench -- ladder --reports target/same-cut/reports \\",
        f"  --track-root target/same-cut/track --track {SAMECUT_TRACK_ID}",
        f"cargo run -p jope-arena -- elo --track {deep['track_id']} --root target/same-cut/track \\",
        f"  --check --committed {SAMECUT_DEEP_CSV.relative_to(ROOT)}",
        "```",
        SAMECUT_END,
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
        section = (
            "## Measured results\n\n"
            "Generated by `deploy/ci/readme-benchmarks.py` from committed evidence. "
            "Do not edit between the markers by hand.\n\n"
            f"{ARENA_BEGIN}\n{ARENA_END}\n\n{SAMECUT_BEGIN}\n{SAMECUT_END}\n\n"
            f"{BENCH_BEGIN}\n{BENCH_END}\n\n"
        )
        text = text[:anchor] + section + text[anchor:]
    if SAMECUT_BEGIN not in text:
        # The same-cut block was added after the other two: slot it between them.
        i = text.find(BENCH_BEGIN)
        if i == -1:
            raise SystemExit(f"markers {BENCH_BEGIN} not found; cannot place the same-cut block")
        text = text[:i] + f"{SAMECUT_BEGIN}\n{SAMECUT_END}\n\n" + text[i:]
    text = splice(text, ARENA_BEGIN, ARENA_END, render_arena())
    text = splice(text, SAMECUT_BEGIN, SAMECUT_END, render_samecut())
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
        SAMECUT_DEEP_CSV,
        SAMECUT_DEEP_SNAPSHOT,
        SAMECUT_SCREEN_CSV,
        SAMECUT_SCREEN_SNAPSHOT,
        SAMECUT_DEEP_TRACK,
        SAMECUT_DEEP_SCORES,
        SAMECUT_REPORTS,
        SAMECUT_DOC,
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

#!/usr/bin/env python3
"""Replay retained historical Elo ladders and keep unrecoverable history open (W12-T04).

Every history in `elo-history/histories.json` carries exactly one disposition:

- `reproducible-with-receipts`: the original scores, track and registries are retained with
  digests. This module re-implements the documented procedure independently (aggregated pair
  statistics in plain Python, not the per-match port in `crates/jope-arena/src/elo.rs`) and must
  reproduce the committed leaderboard bytes and snapshot rows from those inputs alone.
- `scorer-parity-only`: a current scorer agrees with a reference implementation on fixtures. That
  is implementation parity; no historical match graph was recovered.
- `irrecoverable`: original matches, participants or settings are not retained. The retained
  observations may be quoted as attributable history; ranking, rating and continuity claims are
  refused.

Every rating is historical-ranking-only and is refused to teaching, HWM, transfer and promotion
consumers. Ratings are relative to one method universe, so continuity across histories is refused.
Lineage tables are recomputed from their retained bytes: a parent edge whose parent is not retained
blocks continuity across it, and exposure totals that need unretained corpus membership are refused.
A digest authenticates retained bytes, not the truth of the producer that wrote them.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess

CORE = Path(__file__).resolve().parents[1]
REPO = CORE.parents[2]
RECORD = CORE / "elo-history" / "histories.json"
SCHEMA = "jope.core-ip-elo-histories/v1"
SNAPSHOT_SCHEMA = "jope.arena-elo-snapshot/v2"
CORRECTIONS_SCHEMA = "jope.arena-protocol-corrections/v1"
DISPOSITIONS = ("reproducible-with-receipts", "scorer-parity-only", "irrecoverable")
CONSUMER = "historical-ranking-only"
REFUSED_CONSUMERS = ("teaching", "hwm", "transfer", "promotion")
CLAIMS = ("ranking", "rating", "continuity")
SCORE_COLUMNS = ("track_id", "dataset_id", "method_id", "protocol", "metric", "value", "status",
                 "benchmark_valid", "source_kind", "source_revision", "source_sha256", "source_uri",
                 "notes")
LEADERBOARD_COLUMNS = ("elo_rank", "full_coverage_rank", "method_id", "method", "elo", "average_rank",
                       "dataset_wins", "datasets_scored", "datasets_total", "coverage_pct",
                       "full_coverage", "pending_cells", "coverage_status", "protocols",
                       "benchmark_valid", "scope")
SCOPES = {"published": "official unified benchmark comparison",
          "benchmark-valid": "benchmark-valid rows only; protocol corrections applied"}
# The documented procedure: monograph chapter 7, "How the retained Elo scale is constructed".
CENTER, ITERATIONS, REGULARIZATION, TOLERANCE, CLAMP, TIE = 1500.0, 250, 1e-4, 1e-8, 40.0, 1e-12
SCALE = math.log(10) / 400
DIGEST = re.compile(r"[0-9a-f]{64}")
COMMON = ("id", "disposition", "subject", "scope", "anchors", "limitations")


class Unavailable(Exception):
    """A retained input exists but cannot be decoded here; it is never counted as replayed."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve(root, reference):
    relative = PurePosixPath(reference)
    require(not relative.is_absolute() and ".." not in relative.parts, f"{reference}: path traversal")
    return root / relative


def load(root, item):
    """Digest-check one retained file and return its (decoded) bytes."""
    require(isinstance(item, dict) and isinstance(item.get("path"), str)
            and DIGEST.fullmatch(item.get("sha256") or ""), f"{item!r}: input reference")
    path = resolve(root, item["path"])
    require(path.is_file() and not path.is_symlink(), f"{item['path']}: retained input missing")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == item["sha256"], f"{item['path']}: digest mismatch")
    if path.suffix not in (".gz", ".zst"):
        return raw
    if path.suffix == ".gz":
        data = gzip.decompress(raw)
    else:
        tool = shutil.which("zstd")
        if tool is None:
            raise Unavailable(f"{item['path']}: no zstd decoder on this host")
        data = subprocess.run([tool, "-q", "-d", "-c"], input=raw, capture_output=True,
                              check=True).stdout
    require(hashlib.sha256(data).hexdigest() == item.get("decoded_sha256"),
            f"{item['path']}: decoded digest mismatch")
    return data


def rankable(row):
    return row["status"] == "ok" and row["value"] != ""


def parse_scores(text, track_id):
    """Rows of a `scores.csv`. A second delivery of a rankable cell is refused, never merged."""
    reader = csv.reader(io.StringIO(text, newline=""))
    require(tuple(next(reader, ())) == SCORE_COLUMNS, "scores.csv columns")
    rows, cells = [], set()
    for line, values in enumerate(reader, 2):
        require(len(values) == len(SCORE_COLUMNS), f"scores.csv line {line}: column count")
        row = dict(zip(SCORE_COLUMNS, values))
        if rankable(row):
            require(row["track_id"] == track_id, f"scores.csv line {line}: row of another track")
            cell = (row["dataset_id"], row["method_id"])
            require(cell not in cells, f"scores.csv line {line}: duplicate delivery of {cell} "
                                       "cannot add or replace a match")
            cells.add(cell)
            require(math.isfinite(float(row["value"])), f"scores.csv line {line}: non-finite value")
            require(row["benchmark_valid"] in ("True", "False"), f"scores.csv line {line}: validity")
        rows.append(row)
    return rows


def apply_corrections(rows, document, track_id, scores_sha256):
    """Demote the rows a digest-bound corrections file selects; corrections never promote."""
    require(document.get("schema") == CORRECTIONS_SCHEMA, "corrections schema")
    require(document.get("track_id") == track_id and document.get("scores_sha256") == scores_sha256,
            "corrections bind another track or scores.csv")
    applied, corrected = [], 0
    for correction in document["corrections"]:
        require(correction.get("benchmark_valid") is False and correction.get("evidence"),
                f"{correction.get('id')}: a correction only demotes and cites evidence")
        matched = 0
        for row in rows:
            if not rankable(row) or row["method_id"] != correction["method_id"]:
                continue
            if any(correction.get(field) is not None and row[field] != correction[field]
                   for field in ("source_kind", "source_revision")):
                continue
            if correction.get("dataset_ids") and row["dataset_id"] not in correction["dataset_ids"]:
                continue
            matched += 1
            if row["benchmark_valid"] == "True":
                row["benchmark_valid"] = "False"
                corrected += 1
        require(matched, f"{correction['id']}: selects no rankable row")
        applied.append({"id": correction["id"], "rows": matched})
    return applied, corrected


def outcome(left, right, minimize):
    if abs(left - right) <= TIE * max(abs(left), abs(right), 1.0):
        return 0.5
    return 1.0 if (left < right) == minimize else 0.0


def pair_statistics(by_dataset, minimize):
    """Match count and outcome sum for each unordered method pair (the likelihood's sufficient
    statistics): one equal-weight match per dataset and pair of scored methods."""
    pairs = {}
    for dataset in sorted(by_dataset):
        scores = sorted(by_dataset[dataset].items())
        for i, (left, left_value) in enumerate(scores):
            for right, right_value in scores[i + 1:]:
                count, total = pairs.get((left, right), (0, 0.0))
                pairs[(left, right)] = (count + 1, total + outcome(left_value, right_value, minimize))
    return pairs


def fit(pairs, cap=ITERATIONS):
    """Regularized Bradley–Terry on the Elo scale by synchronous diagonal Newton steps."""
    methods = sorted({method for pair in pairs for method in pair})
    index = {method: i for i, method in enumerate(methods)}
    table = [(index[a], index[b], count, total) for (a, b), (count, total) in sorted(pairs.items())]
    ratings = [CENTER] * len(methods)
    iterations, converged = 0, False
    for _ in range(cap):
        gradient, information = [0.0] * len(methods), [0.0] * len(methods)
        for a, b, count, total in table:
            delta = max(-700.0, min(700.0, SCALE * (ratings[a] - ratings[b])))
            probability = 1.0 / (1.0 + math.exp(-delta))
            residual = total - count * probability
            curvature = count * probability * (1.0 - probability)
            gradient[a] += residual
            gradient[b] -= residual
            information[a] += curvature
            information[b] += curvature
        steps = [max(-CLAMP, min(CLAMP, (SCALE * gradient[m] - REGULARIZATION * (ratings[m] - CENTER))
                                 / (SCALE * SCALE * information[m] + REGULARIZATION)))
                 for m in range(len(methods))]
        ratings = [rating + step for rating, step in zip(ratings, steps)]
        iterations += 1
        if max((abs(step) for step in steps), default=0.0) < TOLERANCE:
            converged = True
            break
    shift = CENTER - math.fsum(ratings) / len(ratings) if ratings else 0.0
    return {method: rating + shift for method, rating in zip(methods, ratings)}, iterations, converged


def exchangeable_groups(pairs):
    """Methods with equal match counts against every other method and equal total score.

    Bradley–Terry ratings depend on the data only through match counts and each method's total
    score, so such methods have equal ratings in exact arithmetic even when their per-opponent
    records differ. Their published order is decided by floating-point accumulation order and the
    name tie-break, never by a comparison.
    """
    counts, scores = {}, {}
    for (a, b), (count, total) in pairs.items():
        counts.setdefault(a, {})[b] = count
        counts.setdefault(b, {})[a] = count
        scores[a] = scores.get(a, 0.0) + total
        scores[b] = scores.get(b, 0.0) + count - total

    def related(m, n):
        return all(counts[m].get(k, 0) == counts[n].get(k, 0)
                   for k in (set(counts[m]) | set(counts[n])) - {m, n})

    groups = []
    for score in sorted(set(scores.values())):
        classes = []
        for method in sorted(m for m, s in scores.items() if s == score):
            home = next((c for c in classes if all(related(method, n) for n in c)), None)
            if home is None:
                classes.append([method])
            else:
                home.append(method)
        groups += [c for c in classes if len(c) > 1]
    return sorted(groups)


def components(pairs):
    """Connected components of the comparison graph. Order between components is imposed by the
    regularizer and centering, not by any comparison."""
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in pairs:
        parent[find(a)] = find(b)
    groups = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)
    return sorted(len(group) for group in groups.values())


def ladder(rows, *, track_id, ladder_kind, minimize, names, pending, total_datasets):
    """The leaderboard rows (every cell spelled as Python's `round`/`repr` write it)."""
    by_dataset, protocols, valid = {}, {}, {}
    rankable_rows = excluded = 0
    for row in rows:
        if not rankable(row):
            continue
        is_valid = row["benchmark_valid"] == "True"
        if ladder_kind == "benchmark-valid" and not is_valid:
            excluded += 1
            continue
        by_dataset.setdefault(row["dataset_id"], {})[row["method_id"]] = float(row["value"])
        protocols.setdefault(row["method_id"], set()).add(row["protocol"])
        valid[row["method_id"]] = valid.get(row["method_id"], True) and is_valid
        rankable_rows += 1
    pairs = pair_statistics(by_dataset, minimize)
    ratings, iterations, converged = fit(pairs)
    ranks, wins = {}, {}
    for dataset in sorted(by_dataset):
        ordered = sorted(by_dataset[dataset].items(),
                         key=lambda item: (item[1] if minimize else -item[1], item[0]))
        for rank, (method, _) in enumerate(ordered, 1):
            ranks.setdefault(method, []).append(rank)
        wins[ordered[0][0]] = wins.get(ordered[0][0], 0) + 1
    board, full = [], 0
    for position, (method, rating) in enumerate(sorted(ratings.items(), key=lambda i: (-i[1], i[0])), 1):
        scored = len(ranks.get(method, []))
        covered = scored == total_datasets
        full += covered
        cells = pending.get(method, 0)
        board.append({
            "elo_rank": position, "full_coverage_rank": str(full) if covered else "",
            "method_id": method, "method": names.get(method, method), "elo": repr(round(rating, 1)),
            "average_rank": repr(round(sum(ranks.get(method, [])) / scored, 3)),
            "dataset_wins": wins.get(method, 0), "datasets_scored": scored,
            "datasets_total": total_datasets,
            "coverage_pct": repr(round(100.0 * scored / total_datasets, 1)),
            "full_coverage": str(covered), "pending_cells": cells,
            "coverage_status": f"{scored}/{total_datasets}" + (f" · {cells} pending" if cells else ""),
            "protocols": "|".join(sorted(protocols.get(method, ()))),
            "benchmark_valid": str(valid.get(method, False)), "scope": SCOPES[ladder_kind]})
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(LEADERBOARD_COLUMNS)
    writer.writerows([row[column] for column in LEADERBOARD_COLUMNS] for row in board)
    signatures = {}
    for dataset in sorted(by_dataset):
        for method, value in by_dataset[dataset].items():
            signatures.setdefault(method, []).append((dataset, value))
    aliases = {}
    for method in sorted(ratings):
        aliases.setdefault(tuple(signatures[method]), []).append(method)
    return {"rows": board, "csv": buffer.getvalue(), "ratings": ratings, "iterations": iterations,
            "converged": converged, "rankable_rows": rankable_rows, "excluded_rows": excluded,
            "datasets_scored": len(by_dataset), "matches": sum(c for c, _ in pairs.values()),
            "components": components(pairs), "exchangeable": exchangeable_groups(pairs),
            "identical_scores": sorted(g for g in aliases.values() if len(g) > 1), "pairs": pairs}


def replay(root, spec):
    """Recompute one ladder from its pinned inputs; the track's own declarations bind the paths."""
    base = PurePosixPath(spec["root"])
    inputs = spec["inputs"]
    require(inputs["track"]["path"] == str(base / "tracks" / spec["track_id"] / "track.json"),
            "replay track path")
    track = json.loads(load(root, inputs["track"]))
    require(track.get("schema") == "sotarena.track.v1" and track.get("id") == spec["track_id"],
            "track identity")
    require(track.get("metric_direction") in ("minimize", "maximize"), "metric direction")
    declared = {"datasets": track["dataset_registry"], "methods": track.get("method_registry",
                                                                          "registry/methods.json"),
                "scores": track["scores"]}
    if track.get("resolution_ledger"):
        declared["ledger"] = track["resolution_ledger"]
    require(set(inputs) - {"track"} == set(declared), "replay inputs differ from the track's")
    for role, relative in declared.items():
        path = inputs[role]["path"]
        require(path in (str(base / relative), str(base / relative) + ".zst", str(base / relative) + ".gz"),
                f"replay {role} path is not the track's declared input")
    datasets = json.loads(load(root, inputs["datasets"]))["datasets"]
    names = {m["id"]: m.get("name", m["id"]) for m in json.loads(load(root, inputs["methods"]))["methods"]}
    scores_bytes = load(root, inputs["scores"])
    scores_sha256 = hashlib.sha256(scores_bytes).hexdigest()
    rows = parse_scores(scores_bytes.decode(), spec["track_id"])
    pending = {}
    if "ledger" in inputs:
        for line in load(root, inputs["ledger"]).decode().splitlines():
            if line.strip():
                cell = json.loads(line)
                require(isinstance(cell.get("resolved"), bool), "ledger cell without `resolved`")
                if not cell["resolved"]:
                    pending[cell["method_id"]] = pending.get(cell["method_id"], 0) + 1
    corrections = None
    if spec.get("corrections"):
        raw = load(root, spec["corrections"])
        applied, corrected = apply_corrections(rows, json.loads(raw), spec["track_id"], scores_sha256)
        corrections = {"file": PurePosixPath(spec["corrections"]["path"]).name,
                       "sha256": hashlib.sha256(raw).hexdigest(), "applied": applied,
                       "rows_corrected": corrected}
    require(spec["ladder"] in SCOPES, "ladder kind")
    result = ladder(rows, track_id=spec["track_id"], ladder_kind=spec["ladder"],
                    minimize=track["metric_direction"] == "minimize", names=names, pending=pending,
                    total_datasets=len(datasets))
    result.update(scores_sha256=scores_sha256, corrections=corrections)
    return result


POSITION = ("elo_rank", "full_coverage_rank")


def compare(label, committed, replayed, groups):
    """Every cell must agree, except that methods of one exchangeable group may trade places.

    Returns the groups whose committed order differs from the replay's; such an order was set by
    the original implementation's floating-point accumulation, not by data.
    """
    require(len(committed) == len(replayed), f"{label}: ladder length differs from replay")
    group_of = {method: i for i, group in enumerate(groups) for method in group}
    traded = set()
    for position, (theirs, ours) in enumerate(zip(committed, replayed), 1):
        require(all(theirs[f] == ours[f] for f in POSITION), f"{label}: rank {position} position cells")
        if theirs["method_id"] != ours["method_id"]:
            group = group_of.get(ours["method_id"])
            require(group is not None and group == group_of.get(theirs["method_id"]),
                    f"{label}: rank {position} is {theirs['method_id']} in the committed ladder but "
                    f"{ours['method_id']} in the independent replay")
            traded.add(group)
    strip = lambda rows: {r["method_id"]: {k: v for k, v in r.items() if k not in POSITION} for r in rows}
    theirs, ours = strip(committed), strip(replayed)
    require(theirs.keys() == ours.keys(), f"{label}: rated methods differ from replay")
    for method in sorted(ours):
        require(theirs[method] == ours[method], f"{label}: {method} differs from replay")
    return [groups[i] for i in sorted(traded)]


def check_replay(root, history):
    spec, hid = history["replay"], history["id"]
    result = replay(root, spec)
    committed = load(root, spec["outputs"]["leaderboard"])
    committed_rows = list(csv.DictReader(io.StringIO(committed.decode(), newline="")))
    as_text = [{k: str(v) for k, v in row.items()} for row in result["rows"]]
    traded = compare(f"{hid} leaderboard", committed_rows, as_text, result["exchangeable"])
    snapshot = json.loads(load(root, spec["outputs"]["snapshot"]))
    expected = {"schema": SNAPSHOT_SCHEMA, "track_id": spec["track_id"], "ladder": spec["ladder"],
                "scores_sha256": result["scores_sha256"], "corrections": result["corrections"],
                "excluded_rows": result["excluded_rows"],
                "leaderboard_sha256": hashlib.sha256(committed).hexdigest(),
                "rankable_rows": result["rankable_rows"], "datasets_scored": result["datasets_scored"],
                "methods": len(result["rows"])}
    for field, value in expected.items():
        require(snapshot.get(field) == value, f"{hid}: snapshot {field} differs from replay")
    require(compare(f"{hid} snapshot", snapshot.get("rows", []), result["rows"],
                    result["exchangeable"]) == traded, f"{hid}: snapshot and leaderboard orders differ")
    return {"byte_identical": result["csv"].encode() == committed,
            "exchangeable_groups": result["exchangeable"], "order_differs_from_replay": traded,
            "identical_score_groups": result["identical_scores"],
            "methods": len(result["rows"]), "rankable_rows": result["rankable_rows"],
            "matches": result["matches"], "iterations": result["iterations"],
            "converged": result["converged"], "components": result["components"],
            "cross_component_order": "imposed-by-regularization" if len(result["components"]) > 1
            else "data-connected", "consumer": CONSUMER}


def sensitivity(root, history, cap=2000):
    """How far the retained ratings (the iterate at the historical cap) sit from a longer fit."""
    result = replay(root, history["replay"])
    longer, iterations, converged = fit(result["pairs"], cap)
    order = lambda ratings: [m for m, _ in sorted(ratings.items(), key=lambda i: (-i[1], i[0]))]
    retained, extended = order(result["ratings"]), order(longer)
    return {"history": history["id"], "historical_cap": ITERATIONS,
            "historical_converged": result["converged"], "extended_iterations": iterations,
            "extended_converged": converged,
            "max_abs_rating_shift": max(abs(longer[m] - result["ratings"][m]) for m in longer),
            "positions_changed": sum(a != b for a, b in zip(retained, extended)),
            "top10_unchanged": retained[:10] == extended[:10]}


def check_history(root, history, seen):
    require(isinstance(history, dict) and all(history.get(f) for f in COMMON),
            f"{history.get('id') if isinstance(history, dict) else history}: incomplete history")
    hid = history["id"]
    require(hid not in seen, f"{hid}: duplicate history")
    seen.add(hid)
    require(history["disposition"] in DISPOSITIONS, f"{hid}: disposition")
    require(isinstance(history["limitations"], list), f"{hid}: limitations")
    for item in history.get("retained", []):
        load(root, item)
    if history["disposition"] == "reproducible-with-receipts":
        require(isinstance(history.get("replay"), dict), f"{hid}: reproducible history needs a replay")
        receipts = history.get("receipts")
        require(isinstance(receipts, dict) and receipts.get("original_execution")
                and receipts.get("algorithm"), f"{hid}: original execution and algorithm receipts")
        return
    require("replay" not in history, f"{hid}: {history['disposition']} carries no replay")
    require(isinstance(history.get("missing"), list) and history["missing"],
            f"{hid}: {history['disposition']} must name what is missing")
    require(not {"ratings", "ranking", "leaderboard"} & set(history),
            f"{hid}: {history['disposition']} history cannot carry ratings or rankings")
    if history["disposition"] == "scorer-parity-only":
        require(history.get("parity"), f"{hid}: scorer parity needs its parity test")


LINEAGE_COLUMNS = ("state", "parent", "label", "parent_label", "block", "corpus")


def lineage_graph(root, lineage):
    """Checkpoint nodes and parent edges from a retained table; one node per checkpoint state."""
    lid, columns = lineage["id"], lineage.get("columns", {})
    require(set(columns) == set(LINEAGE_COLUMNS), f"{lid}: lineage columns")
    nodes = {}
    for row in csv.DictReader(io.StringIO(load(root, lineage["table"]).decode(), newline="")):
        node = {role: row[column] for role, column in columns.items()}
        require(nodes.setdefault(node["state"], node) == node,
                f"{lid}: checkpoint {node['state'][:12]}… is recorded inconsistently")
    labels = {node["label"]: node for node in nodes.values()}
    require(len(labels) == len(nodes), f"{lid}: checkpoint labels are not unique")
    missing = {}
    for node in nodes.values():
        if node["parent"] not in nodes:
            key = (node["parent"], node["parent_label"])
            missing[key] = missing.get(key, 0) + 1
    counts = {"checkpoints": len(nodes), "blocks": len({n["block"] for n in nodes.values()}),
              "corpus_identities": len({n["corpus"] for n in nodes.values()}),
              "retained_parent_edges": len(nodes) - sum(missing.values()),
              "unretained_parent_edges": sum(missing.values())}
    edges = [{"parent_state_sha256": state, "parent_label": label, "children": children}
             for (state, label), children in sorted(missing.items())]
    return {"nodes": nodes, "labels": labels, "counts": counts, "missing_edges": edges}


def check_lineage(root, lineage, seen):
    lid = lineage.get("id")
    require(lid and lid not in seen, f"{lid}: lineage id")
    seen.add(lid)
    require(lineage.get("disposition") in DISPOSITIONS and lineage.get("limitations"),
            f"{lid}: lineage disposition and limitations")
    require(lineage.get("exposure_edges") in ("retained", "not-retained"), f"{lid}: exposure edges")
    if "table" not in lineage:
        require(lineage["disposition"] == "irrecoverable" and lineage.get("missing"),
                f"{lid}: a lineage without a retained table is irrecoverable and names what is missing")
        return None
    graph = lineage_graph(root, lineage)
    require(graph["counts"] == lineage.get("expected"), f"{lid}: recorded counts differ from the table")
    require(graph["missing_edges"] == lineage.get("missing_edges"),
            f"{lid}: recorded missing edges differ from the table")
    return graph["counts"]


def authorize(document, history_id, claim, consumer=CONSUMER):
    """Return the permitted scope of a claim about one history, or refuse it."""
    require(consumer not in REFUSED_CONSUMERS,
            f"{history_id}: Elo is historical-ranking-only; {consumer} consumers are refused")
    require(consumer == CONSUMER, f"{history_id}: unknown consumer {consumer!r}")
    require(claim in CLAIMS, f"{history_id}: unknown claim {claim!r}")
    history = next((h for h in document["histories"] if h["id"] == history_id), None)
    require(history is not None, f"{history_id}: unknown history")
    require(claim != "continuity", f"{history_id}: a rating is relative to its own method universe; "
                                   "continuity across ladders is not data-supported")
    require(history["disposition"] == "reproducible-with-receipts",
            f"{history_id}: {history['disposition']} history supports no {claim} claim")
    return {"history": history_id, "claim": claim, "consumer": CONSUMER, "scope": history["scope"]}


def continuity(root, document, lineage_id, ancestor, descendant):
    """Parameter-ancestry continuity between two retained checkpoints, through retained edges only."""
    lineage = next((entry for entry in document["lineages"] if entry["id"] == lineage_id), None)
    require(lineage is not None, f"{lineage_id}: unknown lineage")
    require("table" in lineage, f"{lineage_id}: irrecoverable lineage supports no continuity claim")
    graph = lineage_graph(root, lineage)
    require(ancestor in graph["labels"], f"{lineage_id}: {ancestor} is not a retained checkpoint")
    node = graph["labels"].get(descendant)
    require(node is not None, f"{lineage_id}: {descendant} is not a retained checkpoint")
    path, visited = [descendant], {node["state"]}
    while node["label"] != ancestor:
        parent = graph["nodes"].get(node["parent"])
        require(parent is not None, f"{lineage_id}: missing lineage edge above {node['label']} "
                                    f"(parent {node['parent_label']}, {node['parent'][:12]}… not retained)")
        require(parent["state"] not in visited, f"{lineage_id}: parent cycle at {parent['label']}")
        visited.add(parent["state"])
        node = parent
        path.append(node["label"])
    return {"lineage": lineage_id, "path": path, "meaning": "parameter ancestry through recorded "
            "block-initialization edges; each block trains on its own corpus"}


def exposure(document, lineage_id):
    lineage = next((entry for entry in document["lineages"] if entry["id"] == lineage_id), None)
    require(lineage is not None, f"{lineage_id}: unknown lineage")
    require(lineage["exposure_edges"] == "retained",
            f"{lineage_id}: corpus-to-dataset exposure edges are not retained; exposure totals refused")
    return lineage["exposure"]


def derived_view(root, view, histories):
    """A committed plot table must repeat its snapshots' rows; it adds no history of its own."""
    rows = list(csv.DictReader(io.StringIO(load(root, view["table"]).decode())))
    snapshots = {}
    for name, history_id in view["snapshots"].items():
        history = histories.get(history_id)
        require(history is not None and history["disposition"] == "reproducible-with-receipts",
                f"{view['id']}: {name} is not backed by a reproducible history")
        output = history["replay"]["outputs"]["snapshot"]
        snapshots[name] = (output["sha256"], {row["method_id"]: row for row in
                                              json.loads(load(root, output))["rows"]})
    for number, row in enumerate(rows, 2):
        require(row["snapshot"] in snapshots, f"{view['id']} line {number}: unbacked snapshot")
        digest, board = snapshots[row["snapshot"]]
        require(row["source_sha256"] == digest, f"{view['id']} line {number}: source digest")
        source = board.get(row["method"])
        require(source is not None and all(str(source[k]) == row[v] for k, v in (
            ("elo_rank", "rank"), ("elo", "elo"), ("datasets_scored", "datasets_scored"),
            ("datasets_total", "datasets_total"), ("coverage_pct", "coverage_pct"))),
            f"{view['id']} line {number}: differs from its snapshot")
    counts = {name: sum(row["snapshot"] == name for row in rows) for name in snapshots}
    require(all(counts[name] == len(board) for name, (_, board) in snapshots.items()),
            f"{view['id']}: a snapshot is only partly plotted")
    return len(rows)


def verify(path=RECORD, root=REPO):
    document = json.loads(Path(path).read_text())
    require(document.get("schema") == SCHEMA, "histories schema")
    require(document.get("numerical_policy"), "numerical policy")
    seen, replayed, unavailable = set(), {}, {}
    for history in document.get("histories", []):
        check_history(root, history, seen)
        if history["disposition"] == "reproducible-with-receipts":
            try:
                replayed[history["id"]] = check_replay(root, history)
            except Unavailable as reason:
                unavailable[history["id"]] = str(reason)
    histories = {h["id"]: h for h in document["histories"]}
    lineages, lineage_seen = {}, set()
    for lineage in document.get("lineages", []):
        lineages[lineage["id"]] = check_lineage(root, lineage, lineage_seen)
    views = {view["id"]: derived_view(root, view, histories) for view in document.get("derived_views", [])}
    by = {d: sorted(h["id"] for h in document["histories"] if h["disposition"] == d) for d in DISPOSITIONS}
    return {
        "replayed": sorted(replayed), "not_replayed_here": unavailable,
        "scorer_parity_only": by["scorer-parity-only"], "irrecoverable": by["irrecoverable"],
        "exact_totals": {"histories": len(replayed),
                         "rankable_rows": sum(r["rankable_rows"] for r in replayed.values()),
                         "pairwise_matches": sum(r["matches"] for r in replayed.values())},
        "excluded_from_exact_totals": sorted(set(histories) - set(replayed)),
        "replays": replayed, "lineages": lineages, "derived_view_rows": views,
        "consumer": CONSUMER, "refused_consumers": list(REFUSED_CONSUMERS),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--require-all", action="store_true",
                        help="fail when a reproducible history cannot be replayed on this host")
    parser.add_argument("--sensitivity", metavar="HISTORY",
                        help="only report the iteration-cap sensitivity of one reproducible history")
    args = parser.parse_args()
    if args.sensitivity:
        record = json.loads(args.record.read_text())
        chosen = next((h for h in record["histories"] if h["id"] == args.sensitivity), None)
        require(chosen is not None and "replay" in chosen, f"{args.sensitivity}: no replayable history")
        print(json.dumps(sensitivity(REPO, chosen), sort_keys=True))
        raise SystemExit(0)
    summary = verify(args.record)
    require(not (args.require_all and summary["not_replayed_here"]),
            f"not replayed here: {sorted(summary['not_replayed_here'])}")
    print(json.dumps(summary, sort_keys=True))

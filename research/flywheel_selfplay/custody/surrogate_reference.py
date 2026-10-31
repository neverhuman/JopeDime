#!/usr/bin/env python3
"""Independent S1 arithmetic replay and training-pool revealed-slot census.

Uses admitted local training artifacts only; never imports a model or fits a GP.
Run --self-test for refusal/boundary controls. See SURROGATE.md for provenance.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
SELECTED = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
CATALOG = "e5f732fd784bf67bfc16f4173c585b0861e6e8a6026808df857097a721bbf0ba"
EVALUATOR = "53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900"
JOURNAL_INDEX_SHA256 = "37c022236710166a88419a4e96a91f3366864baa92d7ea98ea7e0bd001dd1e51"
APPROVED_JOURNAL_ROOT = Path.home() / "jope-data/res-05/journals"
SCORE_ROW_KEYS = frozenset(
    {
        "complexity_open_parens",
        "cv_fold_wins",
        "cv_gate_passed",
        "cv_geomean_rmse_ratio",
        "cv_log_ratio_sd",
        "cv_mean_log_ratio",
        "cv_tail_log_ratio",
        "cv_test_agreed",
        "dsl",
        "fitness",
        "logratio",
        "operator",
        "parents",
        "program_sha256",
        "status",
        "test_rmse_ratio",
        "wall_s",
    }
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_json(path, expected=None):
    data = Path(path).read_bytes()
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(data)


def load_journal_index(path):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != JOURNAL_INDEX_SHA256:
        raise ValueError("journal index identity is not the approved RES-05 index")
    return json.loads(data), digest


def read_approved_journal(path, root=APPROVED_JOURNAL_ROOT):
    """Read one pinned journal only from the approved training-journal directory."""
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("journal path must be an absolute approved training path")
    if any(part in ("", ".", "..") for part in path.split("/")[1:]):
        raise ValueError("journal path is not canonical")
    candidate = Path(path)
    approved_root = Path(root).resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(approved_root):
        raise ValueError("journal path is outside approved training artifacts")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("symlink in approved journal path")
    if not resolved.is_file():
        raise ValueError("approved journal path is not a file")
    return resolved.read_bytes()


def validate_score_row(row):
    if not isinstance(row, dict) or set(row) != SCORE_ROW_KEYS:
        raise ValueError("unexpected score row schema")
    if row["test_rmse_ratio"] is not None:
        raise ValueError("forbidden outer score")


def finite(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("finite scalar required")
    return float(value)


def fitness(dsl, mean, tail, sd, wins):
    """From full-precision two-fold summary components, not stored fitness."""
    mean, tail, sd = map(finite, (mean, tail, sd))
    if type(wins) is not int or not 0 <= wins <= 2 or sd < 0:
        raise ValueError("invalid two-fold summary")
    if not isinstance(dsl, str):
        raise ValueError("DSL must be text")
    gate = wins == 2 and math.exp(mean) <= 0.90
    value = (
        0.8 * mean + 0.1 * tail + 0.1 * sd + 0.0005 * dsl.count("(")
        if gate
        else 100.0 + max(mean, 0.0) + max(tail, 0.0) + max(sd, 0.0)
    )
    return value, gate


def agrees(actual, recorded):
    actual, recorded = finite(actual), finite(recorded)
    return abs(actual - recorded) <= 1e-12 * abs(recorded)


def admit(packet, catalog_path):
    selected = checked_json(packet / "selected.json", SELECTED)
    catalog = checked_json(catalog_path, CATALOG)
    by_id = {r["id"]: r for r in catalog["datasets"]}
    excluded = set(catalog["inventory"]["sealed_name_matches_excluded"])
    rows = {}
    for r in selected["train"]:
        d = by_id.get(r["dataset_id"])
        if (
            not d
            or r["dataset_id"] in excluded
            or "beyondarena" in json.dumps(d).lower()
            or r["task"] != "regression"
            or d["task"] != "regression"
            or r["dl_split"] != "DL-Train"
            or r["source_split"] not in ("train", "inner")
            or not r["family_id"]
            or not d.get("family_key")
            or not d["canonical_sha256"].startswith(r["dataset_id"])
        ):
            raise ValueError("inadmissible training context")
        cid = r["context_sha256"]
        if (
            len(cid) != 64
            or any(c not in "0123456789abcdef" for c in cid)
            or cid in rows
        ):
            raise ValueError("duplicate/invalid context identity")
        rows[cid] = r
    if len(rows) != 1219 or len({r["family_id"] for r in rows.values()}) != 205:
        raise ValueError("frozen training inventory mismatch")
    if {r["family_id"] for r in rows.values()} & {
        r["family_id"] for r in selected["dev"]
    }:
        raise ValueError("training/validation family overlap")
    return rows


def replay(index, allowed):
    if index.get("evaluator_source_sha256") != EVALUATOR:
        raise ValueError("producer evaluator source mismatch")
    counts = Counter()
    max_relative = 0.0
    inputs = []
    for item in index["rounds"]:
        if item["input_hashes"]["selected.json"] != SELECTED:
            raise ValueError("producer input identity mismatch")
        expected = [
            (g["context_sha256"], pid)
            for g in item["groups"]
            for pid in g["selected_programs"]
        ]
        if any(cid not in allowed for cid, _ in expected):
            raise ValueError("non-training journal context")
        data = read_approved_journal(item["local_scores_path"])
        if hashlib.sha256(data).hexdigest() != item["scores_sha256"]:
            raise ValueError("journal digest mismatch")
        rows = [
            json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()
        ]
        if len(rows) != len(expected) or len(rows) != item["GP_counts"]["evaluated"]:
            raise ValueError("journal/round inventory mismatch")
        for r, (_, pid) in zip(rows, expected):
            counts["evaluations"] += 1
            validate_score_row(r)
            if (
                r["program_sha256"] != pid
                or hashlib.sha256(r["dsl"].encode()).hexdigest() != pid
            ):
                raise ValueError("program identity mismatch")
            recorded = finite(r["fitness"])
            if type(r["cv_gate_passed"]) is not bool:
                raise ValueError("gate must be boolean")
            if r["status"] == "failed":
                if recorded != 100.0 or r["cv_gate_passed"]:
                    raise ValueError("failure sentinel mismatch")
                counts["explicit_failure_policy_replays"] += 1
                continue
            if r["status"] != "completed":
                raise ValueError("unknown evaluation status")
            if r["complexity_open_parens"] != r["dsl"].count("("):
                raise ValueError("DSL complexity mismatch")
            value, gate = fitness(
                r["dsl"],
                r["cv_mean_log_ratio"],
                r["cv_tail_log_ratio"],
                r["cv_log_ratio_sd"],
                r["cv_fold_wins"],
            )
            if not agrees(value, recorded) or gate != r["cv_gate_passed"]:
                raise ValueError(f"surrogate mismatch at round {item['round']}")
            counts["full_precision_replays"] += 1
            counts["gate_pass" if gate else "gate_fail"] += 1
            error = abs(value - recorded) / abs(recorded) if recorded else 0.0
            max_relative = max(max_relative, error)
        inputs.append(
            {
                k: item[k]
                for k in ("round", "scores_sha256", "receipt_sha256", "manifest_sha256")
            }
        )
    if counts["full_precision_replays"] < 500:
        raise ValueError("fewer than 500 full-precision evaluations")
    return dict(
        counts=dict(counts),
        max_relative_error=max_relative,
        rounds=inputs,
        limit="Arithmetic parity from lossless summaries; no independent native refit.",
    )


def classify(pids, values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(pids) or not np.isfinite(values).all():
        raise ValueError("invalid pool fitness inventory")
    if len(values) < 4:
        return "short", False
    order = np.argsort(
        values
    )  # Exact training helper default; do not change tie policy.
    storage = tuple(pids[:4])
    sorted_four = tuple(pids[i] for i in order[:4])
    tied = len(values) > 4 and values[order[3]] == values[order[4]]
    category = (
        "same_order"
        if storage == sorted_four
        else "same_members_reordered"
        if Counter(storage) == Counter(sorted_four)
        else "different_members"
    )
    return category, bool(tied)


def pool_census(packet, allowed):
    totals, families = Counter(), {}
    headers = Counter()
    evidence = []
    for cid, r in sorted(allowed.items()):
        path = packet / "pools" / (cid + ".npz")
        if sha(path) != r["pool_sha256"]:
            raise ValueError("pool hash mismatch")
        with np.load(path, allow_pickle=False) as z:
            pids = [
                p.decode() if isinstance(p, bytes) else str(p)
                for p in z["program_sha256"].tolist()
            ]
            values = z["fitness"].astype(np.float64)
        if len(values) != r["n_candidates"]:
            raise ValueError("candidate count mismatch")
        category, tied = classify(pids, values)
        totals[category] += 1
        totals["boundary_ties"] += tied
        family = families.setdefault(r["family_id"], Counter())
        family[category] += 1
        table = packet / "tables" / (cid + ".jtt1")
        if sha(table) != r["train_only_jtt1_sha256"]:
            raise ValueError("training-table digest mismatch")
        with table.open("rb") as stream:
            import struct

            raw = stream.read(17)
            nt, nq, width = struct.unpack_from("<III", raw, 4)
            folds = stream.read(nt)
        if raw[:4] != b"JTT1" or nq or raw[16] != 2 or set(folds) != {0, 1}:
            raise ValueError("not a train-only two-fold packet")
        headers["contexts"] += 1
        evidence.append(
            dict(
                context=cid,
                dataset=r["dataset_id"],
                family=r["family_id"],
                category=category,
                boundary_tie=tied,
                rows=nt,
                columns=width,
                fold_counts=[folds.count(0), folds.count(1)],
                fold_sha256=hashlib.sha256(folds).hexdigest(),
                pool_sha256=r["pool_sha256"],
            )
        )
    totals["contexts"] = len(evidence)
    return dict(
        counts=dict(totals),
        families={k: dict(v) for k, v in sorted(families.items())},
        row_range=[min(r["rows"] for r in evidence), max(r["rows"] for r in evidence)],
        feature_range=[
            min(r["columns"] for r in evidence),
            max(r["columns"] for r in evidence),
        ],
        records=evidence,
        numpy_version=np.__version__,
        population="Initial 1219 DL-Train pools; not a pre-proposal snapshot census.",
    )


def proposal_census(packet, allowed, index):
    """Replay the closed promotion prefix before each observed proposal group."""
    meta = index["promotions"]
    if meta["first_later_round"] <= 782:
        raise ValueError("incomplete/corrupted promotion prefix")
    event_bytes = read_approved_journal(meta["local_path"])
    if hashlib.sha256(event_bytes).hexdigest() != meta["sha256"]:
        raise ValueError("incomplete/corrupted promotion prefix")
    events = [
        json.loads(line) for line in event_bytes.decode("utf-8").splitlines()
    ]
    if len(events) != meta["entries"] or [e["round"] for e in events] != sorted(
        e["round"] for e in events
    ):
        raise ValueError("promotion prefix inventory/order mismatch")
    for e in events:
        if (
            e["context_sha256"] not in allowed
            or e["cv_gate_passed"] is not True
            or hashlib.sha256(e["dsl"].encode()).hexdigest() != e["program_sha256"]
        ):
            raise ValueError("invalid training admission")
        finite(e["fitness"])
    contexts = {g["context_sha256"] for r in index["rounds"] for g in r["groups"]}
    pools = {}
    for cid in contexts:
        p = packet / "pools" / (cid + ".npz")
        if sha(p) != allowed[cid]["pool_sha256"]:
            raise ValueError("snapshot base hash mismatch")
        with np.load(p, allow_pickle=False) as z:
            ids = [
                v.decode() if isinstance(v, bytes) else str(v)
                for v in z["program_sha256"].tolist()
            ]
            pools[cid] = (ids, z["fitness"].astype(np.float64))
    position = 0
    counts, records = Counter(), []
    for r in index["rounds"]:
        while position < len(events) and events[position]["round"] < r["round"]:
            event = events[position]
            cid = event["context_sha256"]
            if cid in pools:
                ids, values = pools[cid]
                if event["program_sha256"] not in ids:
                    ids.append(event["program_sha256"])
                    pools[cid] = (ids, np.append(values, event["fitness"]))
            position += 1
        group_ids = [g["context_sha256"] for g in r["groups"]]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError(
                "repeated context within round needs a finer snapshot join"
            )
        for cid in group_ids:
            category, tied = classify(*pools[cid])
            counts[category] += 1
            counts["boundary_ties"] += tied
            records.append(
                dict(
                    round=r["round"], context=cid, category=category, boundary_tie=tied
                )
            )
    counts["snapshots"] = len(records)
    return dict(
        counts=dict(counts),
        records=records,
        promotion_prefix_sha256=meta["sha256"],
        population="Actual pre-proposal training pools in rounds775-782; repeated adaptive observations.",
    )


class Controls(unittest.TestCase):
    def test_passing_formula(self):
        value, gate = fitness("(a)", -1.0, -0.5, 0.5, 2)
        self.assertAlmostEqual(value, -0.7995, places=14)
        self.assertTrue(gate)

    def test_failed_gate_penalty(self):
        value, gate = fitness("(a)", 0.2, 0.4, 0.2, 1)
        self.assertAlmostEqual(value, 100.8, places=12)
        self.assertFalse(gate)

    def test_gate_boundary(self):
        self.assertTrue(fitness("", math.log(0.9), math.log(0.9), 0, 2)[1])
        self.assertFalse(fitness("", math.log(0.9000001), 0, 0, 2)[1])

    def test_relative_and_zero(self):
        self.assertTrue(agrees(0, 0))
        self.assertFalse(agrees(1e-30, 0))
        self.assertFalse(agrees(1.0 + 1e-10, 1.0))

    def test_invalid_components(self):
        for invalid in [math.inf, math.nan, True]:
            with self.assertRaises(ValueError):
                fitness("", invalid, 0, 0, 2)
        with self.assertRaises(ValueError):
            fitness("", 0, 0, -1, 2)

    def test_order_not_just_set(self):
        self.assertEqual(
            classify(list("abcde"), [2, 1, 3, 4, 5])[0], "same_members_reordered"
        )
        self.assertEqual(
            classify(list("abcde"), [5, 1, 2, 3, 4])[0], "different_members"
        )
        self.assertEqual(
            classify(list("abcde"), [1, 2, 3, 4, 5]), ("same_order", False)
        )

    def test_short_nan_ties(self):
        self.assertEqual(classify(["a"], [1]), ("short", False))
        with self.assertRaises(ValueError):
            classify(["a"], [math.nan])
        self.assertTrue(classify(list("abcde"), [1, 1, 1, 1, 1])[1])

    def test_journal_inventory_refusal(self):
        with self.assertRaises(ValueError):
            replay({"rounds": [], "evaluator_source_sha256": EVALUATOR}, {})

    def test_journal_index_must_match_approved_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "approved RES-05 index"):
                load_journal_index(path)

    def test_journal_path_must_be_under_approved_root_and_link_free(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "approved"
            root.mkdir()
            allowed = root / "scores.jsonl"
            allowed.write_text("training")
            self.assertEqual(read_approved_journal(str(allowed), root), b"training")

            outside = base / "outside.jsonl"
            outside.write_text("not admitted")
            with self.assertRaisesRegex(ValueError, "outside approved"):
                read_approved_journal(str(outside), root)

            inside_target = root / "target.jsonl"
            inside_target.write_text("target")
            link = root / "linked.jsonl"
            link.symlink_to(inside_target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                read_approved_journal(str(link), root)

    def test_score_row_rejects_unknown_or_missing_fields(self):
        row = {key: None for key in SCORE_ROW_KEYS}
        validate_score_row(row)
        with self.assertRaisesRegex(ValueError, "unexpected score row schema"):
            validate_score_row(dict(row, outer_rmse=0.0))
        missing = dict(row)
        missing.pop("wall_s")
        with self.assertRaisesRegex(ValueError, "unexpected score row schema"):
            validate_score_row(missing)
        leaked = dict(row, test_rmse_ratio=0.5)
        with self.assertRaisesRegex(ValueError, "forbidden outer score"):
            validate_score_row(leaked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--journal-index", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Controls)
        return (
            0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
        )
    if any(
        v is None for v in (args.packet, args.catalog, args.journal_index, args.output)
    ):
        parser.error("--packet, --catalog, --journal-index and --output are required")
    if args.output.resolve().is_relative_to(HERE.parents[2]):
        parser.error("bulk receipt must be outside the working tree")
    started = time.monotonic()
    allowed = admit(args.packet, args.catalog)
    index, index_digest = load_journal_index(args.journal_index)
    result = dict(
        schema="jope.res05.surrogate-reference/v1",
        source_sha256=sha(__file__),
        selected_sha256=SELECTED,
        catalog_sha256=CATALOG,
        index_sha256=index_digest,
        surrogate=replay(index, allowed),
        initial_pools=pool_census(args.packet, allowed),
        proposal_snapshots=proposal_census(args.packet, allowed, index),
        native_fits=0,
        model_forwards=0,
        GPU_work=False,
        OD13_owner_decision="PENDING",
    )
    result["wall_seconds"] = time.monotonic() - started
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(
        json.dumps(
            dict(
                surrogate=result["surrogate"],
                pool_counts=result["initial_pools"]["counts"],
                proposal_counts=result["proposal_snapshots"]["counts"],
                row_range=result["initial_pools"]["row_range"],
                wall_seconds=result["wall_seconds"],
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

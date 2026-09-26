import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import elo_history as eh

ZSTD = shutil.which("zstd") is not None
FAST = ("H-same-cut-v1-deep", "H-same-cut-v1-screen", "H-same-cut-v2-deep", "H-same-cut-v2-screen")


def record():
    return json.loads(eh.RECORD.read_text())


def pinned(node):
    """Every repository file a record pins, wherever it appears."""
    if isinstance(node, dict):
        if isinstance(node.get("path"), str) and "sha256" in node:
            yield node["path"]
        for value in node.values():
            yield from pinned(value)
    elif isinstance(node, list):
        for value in node:
            yield from pinned(value)


def fast(document):
    """The record without the zstd-compressed histories and the view that needs them."""
    document["histories"] = [h for h in document["histories"]
                             if h["disposition"] != "reproducible-with-receipts" or h["id"] in FAST]
    document["derived_views"] = []
    return document


class Copy:
    """A throwaway repository holding a record and exactly the files it pins."""

    def __init__(self, directory, document):
        self.root = Path(directory)
        self.document = document
        for path in sorted(set(pinned(document))):
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(eh.REPO / path, target)
        self.path = self.root / "histories.json"

    def item(self, history_id, *keys):
        node = next(h for h in self.document["histories"] if h["id"] == history_id)
        for key in keys:
            node = node[key]
        return node

    def rewrite(self, item, change):
        """Change a pinned file and re-pin it, as a publisher substituting evidence would."""
        path = self.root / item["path"]
        path.write_bytes(change(path.read_bytes()))
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def verify(self):
        self.path.write_text(json.dumps(self.document))
        return eh.verify(self.path, self.root)


def scores_csv(rows):
    lines = [",".join(eh.SCORE_COLUMNS)]
    for dataset, method, value in rows:
        lines.append(f"t,{dataset},{method},same-cut,rmse,{value},ok,True,k,r,,,")
    return "\n".join(lines) + "\n"


def board(rows, **options):
    parsed = eh.parse_scores(scores_csv(rows), "t")
    settings = dict(track_id="t", ladder_kind="published", minimize=True, names={}, pending={},
                    total_datasets=len({r[0] for r in rows}))
    settings.update(options)
    return eh.ladder(parsed, **settings)


class CommittedRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = eh.verify()

    def test_every_reproducible_history_is_replayed_or_explicitly_not_here(self):
        reproducible = {h["id"] for h in record()["histories"]
                        if h["disposition"] == "reproducible-with-receipts"}
        replayed, missing = set(self.summary["replayed"]), set(self.summary["not_replayed_here"])
        self.assertEqual(replayed | missing, reproducible)
        self.assertFalse(replayed & missing)
        self.assertTrue(set(FAST) <= replayed)
        if ZSTD:
            self.assertEqual(missing, set())

    def test_only_replayed_histories_enter_exact_totals(self):
        every = {h["id"] for h in record()["histories"]}
        self.assertEqual(set(self.summary["excluded_from_exact_totals"]),
                         every - set(self.summary["replayed"]))
        replays = self.summary["replays"]
        self.assertEqual(self.summary["exact_totals"]["histories"], len(replays))
        self.assertEqual(self.summary["exact_totals"]["rankable_rows"],
                         sum(r["rankable_rows"] for r in replays.values()))
        for irrecoverable in self.summary["irrecoverable"] + self.summary["scorer_parity_only"]:
            self.assertNotIn(irrecoverable, replays)

    def test_committed_order_of_exchangeable_methods_is_reported_not_trusted(self):
        replays = self.summary["replays"]
        self.assertTrue(replays["H-same-cut-v1-deep"]["byte_identical"])
        screen = replays["H-same-cut-v1-screen"]
        self.assertFalse(screen["byte_identical"])
        self.assertEqual(screen["order_differs_from_replay"], [["baseline.elasticnet", "baseline.ridge"]])
        self.assertIn(["baseline.elasticnet", "baseline.ridge"], screen["exchangeable_groups"])
        if ZSTD:
            published = replays["H-sotarena-golden45-published"]
            self.assertEqual(published["order_differs_from_replay"],
                             [["ngboost", "xgboost_hist_trees900_depth4_lr0p02"]])
            self.assertFalse(published["converged"])  # the retained ratings are the 250th iterate
            self.assertEqual(published["iterations"], eh.ITERATIONS)
            self.assertTrue(replays["H-sotarena-golden45-benchmark-valid"]["byte_identical"])

    def test_every_rating_is_historical_ranking_only(self):
        self.assertEqual(self.summary["consumer"], "historical-ranking-only")
        self.assertEqual(set(self.summary["refused_consumers"]), {"teaching", "hwm", "transfer", "promotion"})
        self.assertTrue(all(r["consumer"] == "historical-ranking-only"
                            for r in self.summary["replays"].values()))

    def test_lineage_edges_and_derived_view(self):
        a100 = self.summary["lineages"]["L-lineage-b-a100-recovery"]
        self.assertEqual((a100["checkpoints"], a100["retained_parent_edges"], a100["unretained_parent_edges"]),
                         (78, 74, 4))
        audit = self.summary["lineages"]["L-native-audit-108-blocks"]
        self.assertEqual((audit["checkpoints"], audit["retained_parent_edges"]), (108, 73))
        self.assertEqual(self.summary["derived_view_rows"], {"V-plot-historical-elo": 661})


class IndependentFitTests(unittest.TestCase):
    def test_positive_controls(self):
        strong = board([("a", "strong", 1.0), ("a", "weak", 2.0), ("b", "strong", 2.0), ("b", "weak", 3.0)])
        self.assertGreater(strong["ratings"]["strong"], strong["ratings"]["weak"])
        self.assertAlmostEqual(sum(strong["ratings"].values()) / 2, 1500.0, places=9)
        tied = board([("a", "left", 1.0), ("a", "right", 1.0 + 1e-13)])
        self.assertEqual(tied["ratings"], {"left": 1500.0, "right": 1500.0})
        alone = board([("a", "solo", 1.0), ("b", "x", 1.0), ("b", "y", 2.0)])
        self.assertNotIn("solo", alone["ratings"])
        flipped = board([("a", "strong", 1.0), ("a", "weak", 2.0)], minimize=False)
        self.assertGreater(flipped["ratings"]["weak"], flipped["ratings"]["strong"])

    def test_equal_counts_and_totals_are_exchangeable_despite_different_records(self):
        result = board([("d1", "a", 1), ("d1", "c", 2), ("d1", "b", 3), ("d1", "x", 4),
                        ("d2", "b", 1), ("d2", "x", 2), ("d2", "a", 3), ("d2", "c", 4)])
        self.assertEqual(result["exchangeable"], [["a", "b"], ["c", "x"]])
        self.assertNotEqual(result["pairs"][("a", "c")], result["pairs"][("b", "c")])
        self.assertAlmostEqual(result["ratings"]["a"], result["ratings"]["b"], places=9)

    def test_disconnected_groups_are_reported(self):
        result = board([("d1", "a", 1), ("d1", "b", 2), ("d2", "c", 1), ("d2", "d", 2)])
        self.assertEqual(result["components"], [2, 2])

    def test_identical_score_vectors_are_reported_as_possible_aliases(self):
        result = board([("d1", "a", 1), ("d1", "a2", 1), ("d1", "b", 2),
                        ("d2", "a", 3), ("d2", "a2", 3), ("d2", "b", 1)])
        self.assertEqual(result["identical_scores"], [["a", "a2"]])

    def test_duplicate_delivery_cannot_add_or_replace_a_match(self):
        rows = [("d1", "a", 1.0), ("d1", "b", 2.0)]
        for duplicate in [("d1", "a", 1.0), ("d1", "a", 3.0)]:
            with self.subTest(duplicate=duplicate):
                with self.assertRaisesRegex(ValueError, "duplicate delivery"):
                    eh.parse_scores(scores_csv(rows + [duplicate]), "t")
        missing = scores_csv(rows) + "t,d1,c,same-cut,rmse,,missing,True,k,r,,,\n" * 2
        self.assertEqual(len(eh.parse_scores(missing, "t")), 4)  # unrankable rows add no match

    def test_replay_is_deterministic(self):
        history = next(h for h in record()["histories"] if h["id"] == "H-same-cut-v2-screen")
        first, second = eh.replay(eh.REPO, history["replay"]), eh.replay(eh.REPO, history["replay"])
        self.assertEqual((first["csv"], first["ratings"]), (second["csv"], second["ratings"]))


class SubstitutionTests(unittest.TestCase):
    def test_positive_control_and_substitutions(self):
        v1 = "H-same-cut-v1-deep"

        def flip_winner(data):
            rows = list(csv.reader(io.StringIO(data.decode(), newline="")))
            ok = [r for r in rows[1:] if r[6] == "ok"]
            best = min(ok, key=lambda r: float(r[5]))
            best[5] = repr(float(best[5]) * 10)
            buffer = io.StringIO()
            csv.writer(buffer, lineterminator="\n").writerows(rows)
            return buffer.getvalue().encode()

        def swap_rows(data):
            lines = data.decode().splitlines(keepends=True)
            first, second = lines[1].split(",", 2), lines[2].split(",", 2)
            lines[1], lines[2] = ",".join(first[:2] + second[2:]), ",".join(second[:2] + first[2:])
            return "".join(lines).encode()

        def snapshot_elo(data):
            snapshot = json.loads(data)
            snapshot["rows"][0]["elo"] = "1700.0"
            return json.dumps(snapshot).encode()

        def stranger(c):
            c.item(v1, "replay", "inputs", "scores")["path"] = c.item(
                "H-same-cut-v2-deep", "replay", "inputs", "scores")["path"]

        mutations = {
            "unchanged bytes, wrong digest": (
                lambda c: c.item(v1, "replay", "inputs", "scores").update(sha256="0" * 64), "digest mismatch"),
            "re-pinned substituted scores": (
                lambda c: c.rewrite(c.item(v1, "replay", "inputs", "scores"), flip_winner),
                "differs from replay|in the committed ladder but"),
            "re-pinned duplicate delivery": (
                lambda c: c.rewrite(c.item(v1, "replay", "inputs", "scores"),
                                    lambda d: d + d.splitlines(keepends=True)[1]), "duplicate delivery"),
            "re-pinned reordered leaderboard": (
                lambda c: c.rewrite(c.item(v1, "replay", "outputs", "leaderboard"), swap_rows),
                "in the committed ladder but"),
            "re-pinned snapshot rating": (
                lambda c: c.rewrite(c.item(v1, "replay", "outputs", "snapshot"), snapshot_elo),
                "differs from replay"),
            "input other than the track declares": (stranger, "not the track's declared input"),
            "path traversal": (lambda c: c.item("H-legacy-round1-internal-elo", "retained")[0].update(
                path="../elo_trajectory.csv"), "path traversal"),
        }
        with tempfile.TemporaryDirectory() as directory:
            Copy(directory, fast(record())).verify()  # positive control
        for label, (mutate, reason) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                copy_ = Copy(directory, fast(record()))
                mutate(copy_)
                with self.assertRaisesRegex(ValueError, reason):
                    copy_.verify()


class DispositionTests(unittest.TestCase):
    def test_claims_are_refused_outside_reproducible_histories_and_historical_consumers(self):
        document = record()
        granted = eh.authorize(document, "H-same-cut-v2-deep", "ranking")
        self.assertEqual(granted["consumer"], "historical-ranking-only")
        refusals = {
            ("H-same-cut-v2-deep", "ranking", "hwm"): "hwm consumers are refused",
            ("H-same-cut-v2-deep", "rating", "teaching"): "teaching consumers are refused",
            ("H-same-cut-v2-deep", "rating", "transfer"): "transfer consumers are refused",
            ("H-same-cut-v2-deep", "ranking", "promotion"): "promotion consumers are refused",
            ("H-same-cut-v2-deep", "continuity", eh.CONSUMER): "continuity across ladders",
            ("H-legacy-round1-internal-elo", "ranking", eh.CONSUMER): "irrecoverable history supports no ranking",
            ("H-sotarena-golden42-v1", "rating", eh.CONSUMER): "irrecoverable history supports no rating",
            ("H-campaign-redteam-bradley-terry", "ranking", eh.CONSUMER): "scorer-parity-only history supports",
            ("H-unknown", "ranking", eh.CONSUMER): "unknown history",
        }
        for (history, claim, consumer), reason in refusals.items():
            with self.subTest(history=history, claim=claim, consumer=consumer):
                with self.assertRaisesRegex(ValueError, reason):
                    eh.authorize(document, history, claim, consumer)

    def test_record_cannot_upgrade_a_disposition(self):
        def history(document, history_id):
            return next(h for h in document["histories"] if h["id"] == history_id)

        legacy, parity = "H-legacy-round1-internal-elo", "H-campaign-redteam-bradley-terry"
        mutations = {
            "irrecoverable relabelled reproducible": (
                lambda d: history(d, legacy).update(disposition="reproducible-with-receipts"), "needs a replay"),
            "irrecoverable with a replay": (
                lambda d: history(d, legacy).update(replay=history(d, "H-same-cut-v1-deep")["replay"]),
                "carries no replay"),
            "irrecoverable with ratings": (
                lambda d: history(d, legacy).update(ratings={"gen30": 1119.2}), "cannot carry ratings"),
            "irrecoverable without what is missing": (
                lambda d: history(d, legacy).update(missing=[]), "must name what is missing"),
            "parity without its test": (lambda d: history(d, parity).pop("parity"), "parity test"),
            "parity with a ranking": (lambda d: history(d, parity).update(ranking=["a"]), "cannot carry"),
            "reproducible without receipts": (
                lambda d: history(d, "H-same-cut-v1-deep").pop("receipts"), "receipts"),
            "unknown disposition": (lambda d: history(d, legacy).update(disposition="recovered"), "disposition"),
            "duplicate history": (lambda d: d["histories"].append(copy.deepcopy(history(d, legacy))),
                                  "duplicate history"),
            "no limitations": (lambda d: history(d, legacy).update(limitations=[]), "incomplete history"),
            "altered retained observation": (
                lambda d: history(d, legacy)["retained"][0].update(sha256="1" * 64), "digest mismatch"),
        }
        for label, (mutate, reason) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                document = fast(record())
                mutate(document)
                with self.assertRaisesRegex(ValueError, reason):
                    Copy(directory, document).verify()


class LineageTests(unittest.TestCase):
    A100, AUDIT = "L-lineage-b-a100-recovery", "L-native-audit-108-blocks"

    def test_continuity_only_through_retained_edges(self):
        document = record()
        granted = eh.continuity(eh.REPO, document, self.A100, "6000", "60000")
        self.assertEqual(granted["path"][0], "60000")
        self.assertEqual(granted["path"][-1], "6000")
        self.assertEqual(len(granted["path"]), 19)  # one block-initialization edge per block
        audit = eh.continuity(eh.REPO, document, self.AUDIT,
                              "runs/continuous-learning-v2-20260909/cycle-000000/train-2",
                              "runs/continuous-learning-v2-20260909/cycle-000001/train-2")
        self.assertEqual(len(audit["path"]), 2)
        refusals = {
            (self.A100, "3000", "60000"): "3000 is not a retained checkpoint",
            (self.A100, "4000", "4477"): "missing lineage edge above 4477",
            (self.A100, "6000", "70000"): "70000 is not a retained checkpoint",
            ("L-lineage-a-xbabe2", "a", "b"): "irrecoverable lineage supports no continuity",
            ("L-unknown", "a", "b"): "unknown lineage",
        }
        for (lineage, ancestor, descendant), reason in refusals.items():
            with self.subTest(lineage=lineage, ancestor=ancestor, descendant=descendant):
                with self.assertRaisesRegex(ValueError, reason):
                    eh.continuity(eh.REPO, document, lineage, ancestor, descendant)
        for lineage in (self.A100, self.AUDIT, "L-lineage-a-xbabe2"):
            with self.assertRaisesRegex(ValueError, "exposure edges are not retained"):
                eh.exposure(document, lineage)

    def test_recorded_lineage_facts_must_match_the_table(self):
        def lineage(document, lineage_id):
            return next(entry for entry in document["lineages"] if entry["id"] == lineage_id)

        mutations = {
            "inflated retained edges": (lambda d: lineage(d, self.A100)["expected"].update(
                retained_parent_edges=78, unretained_parent_edges=0), "recorded counts differ"),
            "hidden missing edge": (lambda d: lineage(d, self.A100).update(missing_edges=[]),
                                    "recorded missing edges differ"),
            "table without columns": (lambda d: lineage(d, self.AUDIT).pop("columns"), "lineage columns"),
            "irrecoverable table-less lineage relabelled": (
                lambda d: lineage(d, "L-lineage-a-xbabe2").update(disposition="reproducible-with-receipts"),
                "is irrecoverable and names what is missing"),
        }
        for label, (mutate, reason) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                document = fast(record())
                mutate(document)
                with self.assertRaisesRegex(ValueError, reason):
                    Copy(directory, document).verify()

    def test_inconsistent_or_ambiguous_tables_are_refused(self):
        header = "state,parent,step,parent_step,cycle,corpus\n"
        tables = {
            "inconsistent node": header + "s1,p,1,0,0,c\ns1,q,1,0,0,c\n",
            "duplicate label": header + "s1,p,1,0,0,c\ns2,s1,1,1,0,c\n",
        }
        reasons = {"inconsistent node": "recorded inconsistently", "duplicate label": "labels are not unique"}
        for label, text in tables.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "table.csv"
                path.write_text(text)
                entry = {"id": "L-test", "columns": {"state": "state", "parent": "parent", "label": "step",
                                                     "parent_label": "parent_step", "block": "cycle",
                                                     "corpus": "corpus"},
                         "table": {"path": "table.csv", "sha256": hashlib.sha256(text.encode()).hexdigest()}}
                with self.assertRaisesRegex(ValueError, reasons[label]):
                    eh.lineage_graph(Path(directory), entry)


class DerivedViewTests(unittest.TestCase):
    def test_plot_rows_must_repeat_reproducible_snapshots(self):
        document = record()
        histories = {h["id"]: h for h in document["histories"]}
        view = document["derived_views"][0]

        def plot(change):
            def rewrite(data):
                rows = list(csv.DictReader(io.StringIO(data.decode(), newline="")))
                change(rows)
                buffer = io.StringIO()
                writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
                return buffer.getvalue().encode()
            return rewrite

        cases = {
            "raised rating": (plot(lambda rows: rows[0].update(elo="1999.9")), None, "differs from its snapshot"),
            "dropped row": (plot(lambda rows: rows.pop()), None, "only partly plotted"),
            "irrecoverable source": (None, "H-legacy-round1-internal-elo", "not backed by a reproducible history"),
        }
        with tempfile.TemporaryDirectory() as directory:
            copy_ = Copy(directory, copy.deepcopy(document))
            self.assertEqual(eh.derived_view(copy_.root, view, histories), 661)  # positive control
        for label, (rewrite, source, reason) in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                changed = copy.deepcopy(view)
                copy_ = Copy(directory, copy.deepcopy(document))
                if rewrite:
                    copy_.rewrite(changed["table"], rewrite)
                if source:
                    changed["snapshots"]["same-cut-v1-deep"] = source
                with self.assertRaisesRegex(ValueError, reason):
                    eh.derived_view(copy_.root, changed, histories)


if __name__ == "__main__":
    unittest.main()

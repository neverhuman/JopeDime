import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from rlfw.analytics.extract import Copies, digest, ledger_events
from rlfw.analytics.report import read_daily, render, s3_census, s3_daily, write_daily
from rlfw.analytics.windows import measured_windows


class CopiesTest(unittest.TestCase):
    def test_database_copy_cannot_write_or_create_live_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.sqlite"
            with sqlite3.connect(source) as db:
                db.execute("CREATE TABLE example(value)")
            before = (digest(source), source.stat().st_mtime_ns, set(root.iterdir()))
            copies = Copies(root / "copies")
            db = copies.database(source)
            with self.assertRaises(sqlite3.OperationalError):
                db.execute("INSERT INTO example VALUES(1)")
            db.close()
            self.assertEqual(before[:2], (digest(source), source.stat().st_mtime_ns))
            self.assertEqual(set(root.iterdir()) - before[2], {root / "copies"})

    def test_live_wal_and_symlinks_refused_before_sqlite_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "live.sqlite"
            src.write_bytes(b"not a snapshot")
            Path(str(src) + "-wal").write_bytes(b"active")
            c = Copies(root / "copies")
            with self.assertRaises(ValueError):
                c.database(src)
            link = root / "link"
            link.symlink_to(src)
            with self.assertRaises(ValueError):
                c.copy(link)

    def test_hash_and_length_pin_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "source.json"
            src.write_text("{}")
            c = Copies(root / "copies")
            with self.assertRaises(ValueError):
                c.json(src, expected_sha="0" * 64)
            with self.assertRaises(ValueError):
                c.json(src, expected_bytes=3)

    def test_merged_ledger_schema_keeps_first_label_and_has_no_invented_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "ledger.sqlite"
            with sqlite3.connect(src) as db:
                db.execute(
                    "CREATE TABLE events(hwm_key,version,event,reason,publication,best_q_grid,best_program_sha256)"
                )
                db.execute(
                    "INSERT INTO events VALUES(?,?,?,?,?,?,?)",
                    (
                        "key",
                        1,
                        "first_label",
                        None,
                        json.dumps({"q_grid": 3}),
                        3,
                        "sha",
                    ),
                )
            rows = ledger_events(Copies(root / "copies"), src)
            self.assertEqual(rows[0]["event"], "first_label")
            self.assertNotIn("at", rows[0])


class ParquetTest(unittest.TestCase):
    def test_provider_role_census_is_stock_and_must_match_receipt(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "census.parquet"
            rows = [
                {
                    "role": "evaluation",
                    "count": 2,
                    "bytes": 100,
                    "storage_class": "STANDARD",
                }
            ]
            pq.write_table(pa.Table.from_pylist(rows), path)
            receipt = {
                "schema": "jope.s3gp-dry-run/v1",
                "objects_prefix_census": {"bytes": 100, "object_count": 2},
                "parquet_export": {"rows": 1},
            }
            actual, census = s3_census(path, receipt)
            self.assertEqual(actual, rows)
            self.assertNotIn("usd_per_day", census)
            receipt["objects_prefix_census"]["bytes"] = 101
            with self.assertRaises(ValueError):
                s3_census(path, receipt)

    def test_null_roundtrip_dashboard_and_html_escaping(self):
        row = {
            "date": "2026-10-09",
            "kind": "summary",
            "metric": "T_feedback",
            "scope": "<script>",
            "status": "unmeasured",
            "value": None,
            "n": 0,
            "unit": "seconds",
            "note": "no exposure",
            "sources": "file.json:1",
            "source_commit": "a" * 40,
            "context_set_sha256": "b" * 64,
            "details": "{}",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = write_daily([row], root / "daily")
            readback = read_daily(files)
            self.assertEqual(readback, [row])
            render(readback, root / "dashboard.html")
            raw = (root / "dashboard.html").read_text()
            self.assertIn("unmeasured", raw)
            self.assertIn("&lt;script&gt;", raw)
            self.assertNotIn("<script>", raw)
            with self.assertRaises(ValueError):
                write_daily([row], root / "daily")

    def test_s3_export_estimate_and_unknown_dollars_preserved(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s3.parquet"
            rows = [
                {
                    "date": "2026-10-09",
                    "bytes": 100,
                    "usd": None,
                    "cost_kind": "unmeasured",
                    "source": "G-S3GP",
                }
            ]
            pq.write_table(pa.Table.from_pylist(rows), path)
            self.assertEqual(s3_daily(path), rows)
            pq.write_table(pa.Table.from_pylist([{"bytes": 1}]), path)
            with self.assertRaises(ValueError):
                s3_daily(path)


class WindowTest(unittest.TestCase):
    def test_missing_publication_clock_cannot_establish_zero_window_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def pinned(name, value):
                path = root / name
                path.write_text(json.dumps(value))
                return {
                    "path": str(path),
                    "sha256": digest(path),
                    "bytes": path.stat().st_size,
                }

            ledger = root / "ledger.sqlite"
            with sqlite3.connect(ledger) as db:
                db.execute(
                    "CREATE TABLE events(hwm_key,version,event,reason,publication,best_q_grid,best_program_sha256)"
                )
                for version, event in ((1, "first_label"), (2, "improvement")):
                    publication = {
                        "hwm_key": "key",
                        "source": "measured",
                        "program_sha256": str(version),
                        "q_grid": version * 1000000,
                        "outcome": "qualified_complete",
                    }
                    db.execute(
                        "INSERT INTO events VALUES(?,?,?,?,?,?,?)",
                        (
                            "key",
                            version,
                            event,
                            None,
                            json.dumps(publication),
                            publication["q_grid"],
                            str(version),
                        ),
                    )
            manifest = {
                "schema": "jope.rlfw-measurement-input/v1",
                "contexts": pinned(
                    "contexts.json",
                    [
                        {
                            "context_id": "c",
                            "parent": "p",
                            "hwm_key": "key",
                            "frozen_q_grid": 1000000,
                            "resolved": True,
                        }
                    ],
                ),
                "ledger": {
                    "path": str(ledger),
                    "sha256": digest(ledger),
                    "bytes": ledger.stat().st_size,
                },
                "windows": [
                    {
                        "name": "1h",
                        "start": 0,
                        "end": 3600,
                        "cpu": pinned(
                            "cpu.json",
                            {
                                "schema": "jope.evaluator-cgroup-window/v1",
                                "start": 0,
                                "end": 3600,
                                "parents": {"p": 3600},
                            },
                        ),
                    }
                ],
            }
            # Missing first-label and improvement clocks both leave an unknown
            # window membership. CPU receipts cannot turn that into zero.
            for included in ((1,), (2,), ()):
                with self.subTest(included=included):
                    manifest["timeline"] = pinned(
                        f"timeline-{included}.json",
                        [
                            {
                                "hwm_key": "key",
                                "version": v,
                                "context_id": "c",
                                "at": v,
                                "first_seen": True,
                            }
                            for v in included
                        ],
                    )
                    rows = {
                        r["metric"]: r
                        for r in measured_windows(
                            Copies(root / f"copies-{included}"),
                            manifest,
                            source_commit="a" * 40,
                        )
                    }
                    for metric in ("U_useful", "G_per_core_hour", "delta_H"):
                        self.assertIsNone(rows[metric]["value"])
                        self.assertEqual(rows[metric]["status"], "unmeasured")

    def test_real_ledger_schema_receipt_join_and_rolling_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def pinned(name, value):
                p = root / name
                p.write_text(json.dumps(value))
                return {"path": str(p), "sha256": digest(p), "bytes": p.stat().st_size}

            ledger = root / "ledger.sqlite"
            with sqlite3.connect(ledger) as db:
                db.execute(
                    "CREATE TABLE events(hwm_key,version,event,reason,publication,best_q_grid,best_program_sha256)"
                )
                for version, event, score in (
                    (1, "first_label", 1000000),
                    (2, "improvement", 2000000),
                ):
                    publication = {
                        "hwm_key": "key",
                        "source": "measured",
                        "program_sha256": str(version),
                        "q_grid": score,
                        "outcome": "qualified_complete",
                    }
                    db.execute(
                        "INSERT INTO events VALUES(?,?,?,?,?,?,?)",
                        (
                            "key",
                            version,
                            event,
                            None,
                            json.dumps(publication),
                            score,
                            str(version),
                        ),
                    )
            manifest = {
                "schema": "jope.rlfw-measurement-input/v1",
                "contexts": pinned(
                    "contexts.json",
                    [
                        {
                            "context_id": "c",
                            "parent": "p",
                            "hwm_key": "key",
                            "frozen_q_grid": 1000000,
                            "resolved": True,
                        }
                    ],
                ),
                "timeline": pinned(
                    "timeline.json",
                    [
                        {
                            "hwm_key": "key",
                            "version": v,
                            "context_id": "c",
                            "at": v,
                            "first_seen": True,
                        }
                        for v in (1, 2)
                    ],
                ),
                "ledger": {
                    "path": str(ledger),
                    "sha256": digest(ledger),
                    "bytes": ledger.stat().st_size,
                },
                "windows": [
                    {
                        "name": "1h",
                        "start": 0,
                        "end": 3600,
                        "cpu": pinned(
                            "cpu.json",
                            {
                                "schema": "jope.evaluator-cgroup-window/v1",
                                "start": 0,
                                "end": 3600,
                                "parents": {"p": 3600},
                            },
                        ),
                    }
                ],
            }
            copies = Copies(root / "copies")
            rows = measured_windows(copies, manifest, source_commit="a" * 40)
            result = {r["metric"]: r["value"] for r in rows}
            self.assertEqual(result["G_per_core_hour"], 1.0)
            self.assertEqual(result["U_useful"], 1 / 3600)
            self.assertEqual(result["delta_H"], 1.0)
            self.assertIsNone(result["T_feedback:p50"])
            # Only proposals in this window enter its cohort. Admission alone
            # keeps T unmeasured, even if completed-only p50 is available.
            manifest["feedback"] = pinned(
                "feedback.json",
                [
                    {
                        "join_key": key,
                        "context_id": "c",
                        "parent": "p",
                        "stage": stage,
                        "at": at,
                    }
                    for key, stage, at in (
                        ("older", "PROPOSAL_COMMITTED", -100),
                        ("older", "FIRST_EXPOSURE", 100),
                        ("in", "PROPOSAL_COMMITTED", 20),
                        ("in", "FIRST_EXPOSURE", 30),
                    )
                ],
            )
            result = {
                r["metric"]: r["value"]
                for r in measured_windows(copies, manifest, source_commit="a" * 40)
            }
            self.assertEqual(result["T_feedback"], 10)
            self.assertEqual(result["T_feedback:p50"], 10)
            feedback = json.loads(Path(manifest["feedback"]["path"]).read_text())
            feedback.extend(
                [
                    {
                        "join_key": "foreign",
                        "context_id": "foreign",
                        "parent": "foreign",
                        "stage": stage,
                        "at": at,
                    }
                    for stage, at in (
                        ("PROPOSAL_COMMITTED", 20),
                        ("FIRST_EXPOSURE", 2000),
                    )
                ]
            )
            manifest["feedback"] = pinned("foreign-feedback.json", feedback)
            result = {
                r["metric"]: r["value"]
                for r in measured_windows(copies, manifest, source_commit="a" * 40)
            }
            self.assertEqual(result["T_feedback:p50"], 10)
            manifest["feedback"] = pinned(
                "censored.json",
                [
                    {
                        "join_key": "pending",
                        "context_id": "c",
                        "parent": "p",
                        "stage": stage,
                        "at": at,
                    }
                    for stage, at in (("PROPOSAL_COMMITTED", 20), ("ADMITTED", 30))
                ],
            )
            result = {
                r["metric"]: r["value"]
                for r in measured_windows(copies, manifest, source_commit="a" * 40)
            }
            self.assertIsNone(result["T_feedback"])
            self.assertEqual(result["feedback_right_censored"], 1)
            manifest["windows"][0]["end"] = 3601
            with self.assertRaises(ValueError):
                measured_windows(copies, manifest, source_commit="a" * 40)

import copy
import gzip
import io
import json
import tarfile
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from custody_v2.read_only import ReadOnlyS3
from custody_v2.retention import (
    BYTE_CAP,
    KEEPER_FACTS,
    budget_status,
    classify,
    cost_report,
    digest,
    identity,
    pack_local,
    verify_local,
)

NOW = datetime(2026, 10, 13, tzinfo=timezone.utc)
PREFIX = "safe/objects/"


def row(key="rust_executable", version="v1", body=b"same bytes"):
    return {
        "bucket": "bucket",
        "key": PREFIX + key,
        "version_id": version,
        "bytes": len(body),
        "storage_class": "STANDARD",
        "is_latest": True,
        "etag": "not-a-sha",
        "last_modified": NOW.isoformat(),
    }


def facts(r):
    return dict(
        {k: False for k in KEEPER_FACTS},
        identity=list(identity(r)),
        legal_hold_checked=True,
        retention_checked=True,
        legal_hold="OFF",
        retention_until=None,
        audit_salt_commitment="a" * 64,
    )


class Events:
    def __init__(self):
        self.handlers = {}

    def register(self, name, callback):
        self.handlers[name] = callback


class FakeS3:
    def __init__(self, pages=None, body=b"same bytes"):
        self.pages = list(pages or [])
        self.body = body
        self.calls = []
        self.meta = SimpleNamespace(events=Events())

    def dispatch(self, operation, params, response):
        events = self.meta.events.handlers
        events["before-send.s3"](request=None, event_name="before-send.s3." + operation)
        self.calls.append((operation, params))
        response["ResponseMetadata"] = {"RequestId": "fixture", "RetryAttempts": 0}
        events["after-call.s3"](
            http_response=SimpleNamespace(status_code=200, content=b"listing"),
            parsed=response,
            model=SimpleNamespace(name=operation),
        )
        return response

    def list_object_versions(self, **params):
        return self.dispatch("ListObjectVersions", params, self.pages.pop(0))

    def get_object(self, **params):
        response = {
            "VersionId": params["VersionId"],
            "ContentLength": len(self.body),
            "Body": io.BytesIO(self.body),
        }
        return self.dispatch("GetObject", params, response)

    def delete_object(self, **params):
        return self.dispatch("DeleteObject", params, {})


def page(version="v1", truncated=False, **extra):
    return dict(
        Versions=[
            {
                "Key": PREFIX + "score_records",
                "VersionId": version,
                "Size": 10,
                "StorageClass": "STANDARD",
                "IsLatest": True,
                "LastModified": NOW,
                "ETag": "opaque",
            }
        ],
        IsTruncated=truncated,
        **extra,
    )


class RetentionTests(unittest.TestCase):
    def test_unknown_facts_and_null_versions_keep_everything(self):
        self.assertEqual(classify(row())["retention_class"], "keeper")
        r = row(version="null")
        self.assertEqual(
            classify(r, facts(r))["reason"], "immutable_version_unavailable"
        )

    def test_complete_negative_facts_are_only_a_non_applied_proposal(self):
        r = row()
        result = classify(r, facts(r))
        self.assertEqual(result["retention_class"], "other")
        self.assertEqual(result["proposed_retention_days"], 14)
        self.assertFalse(result["applied"] or result["qualified"])

    def test_each_keeper_reason_wins(self):
        r = row()
        for name in KEEPER_FACTS:
            with self.subTest(name=name):
                f = facts(r)
                f[name] = True
                self.assertEqual(classify(r, f)["retention_class"], "keeper")

    def test_missing_salt_or_hold_facts_cannot_release_bytes(self):
        r = row()
        for name in (
            "audit_salt_commitment",
            "legal_hold_checked",
            "retention_checked",
            *KEEPER_FACTS,
        ):
            f = facts(r)
            del f[name]
            self.assertEqual(classify(r, f)["retention_class"], "keeper")

    def test_keeper_facts_cannot_be_borrowed_from_another_version(self):
        with self.assertRaisesRegex(ValueError, "exact physical version"):
            classify(row(version="new"), facts(row(version="old")))

    def test_noncurrent_unknown_and_small_artifacts_stay_forever(self):
        for role in ("score_records", "input", "rust_source", "unknown"):
            self.assertEqual(classify(row(role))["retention_class"], "keeper")
        r = row()
        r["is_latest"] = False
        self.assertEqual(classify(r, facts(r))["reason"], "noncurrent_version_review")

    def test_alarm_at_80_percent_and_refusal_at_100_percent(self):
        self.assertFalse(budget_status(BYTE_CAP * 4 // 5 - 1, 0, 0)["alarm"])
        self.assertEqual(
            budget_status(BYTE_CAP * 4 // 5, 0, 0)["alarm"], ["new_version_bytes"]
        )
        self.assertTrue(budget_status(BYTE_CAP - 1, 0, 0)["admit_new_scoring"])
        self.assertFalse(budget_status(BYTE_CAP, 0, 0)["admit_new_scoring"])

    def test_missing_counters_and_boolean_counts_refuse(self):
        self.assertFalse(budget_status(0, None, None)["admit_new_scoring"])
        with self.assertRaises(ValueError):
            budget_status(True, 0, 0)

    def test_cost_uses_binary_gb_actual_request_counts_and_explicit_unknowns(self):
        r = row()
        r["bytes"] = 2**30
        rates = {
            "region": "us-east-2",
            "storage_bytes_per_gb": 2**30,
            "standard_gb_month": "0.023",
            "list_request": "0.000005",
            "get_request": "0.0000004",
        }
        cost = cost_report(
            [r],
            [{"operation": "ListObjectVersions"}, {"operation": "GetObject"}],
            rates,
            NOW,
        )
        self.assertEqual(cost["request_usd_estimate"], "0.0000054")
        self.assertEqual(cost["days_in_utc_month"], 31)
        self.assertIsNone(cost["actual_billed_usd"])
        r["storage_class"] = "GLACIER"
        self.assertFalse(cost_report([r], [], rates, NOW)["all_storage_priced"])


class ReaderTests(unittest.TestCase):
    def test_wrong_returned_version_refuses_without_consuming_body(self):
        body = io.BytesIO(b"same bytes")
        fake = FakeS3()
        reader = ReadOnlyS3(fake, "bucket", PREFIX)

        def wrong(**params):
            return fake.dispatch(
                "GetObject",
                params,
                {"VersionId": "different", "ContentLength": 10, "Body": body},
            )

        with patch.object(fake, "get_object", side_effect=wrong):
            with self.assertRaisesRegex(ValueError, "GET identity"):
                reader.capture(row("score_records"))
        self.assertTrue(body.closed)
        self.assertEqual(reader.receipt()["object_body_bytes"], 0)
        self.assertEqual(reader.receipt()["physical_request_count"], 1)

    def test_all_mutations_are_refused_before_any_fake_dispatch(self):
        fake = FakeS3()
        reader = ReadOnlyS3(fake, "bucket", PREFIX)
        for operation in (
            "delete_object",
            "delete_objects",
            "put_object",
            "copy_object",
            "put_bucket_lifecycle_configuration",
        ):
            with self.assertRaises(ValueError):
                reader.call(operation, Bucket="bucket", Key=PREFIX + "x")
        with self.assertRaises(ValueError):
            fake.delete_object(Bucket="bucket", Key=PREFIX + "x")
        self.assertEqual(fake.calls, [])

    def test_wrong_scope_and_unversioned_get_refuse_before_dispatch(self):
        fake = FakeS3()
        reader = ReadOnlyS3(fake, "bucket", PREFIX)
        for params in (
            {"Bucket": "other", "Prefix": PREFIX},
            {"Bucket": "bucket", "Prefix": "outside/"},
        ):
            with self.assertRaises(ValueError):
                reader.call("list_object_versions", **params)
        with self.assertRaises(ValueError):
            reader.call(
                "get_object", Bucket="bucket", Key=PREFIX + "x", VersionId="null"
            )
        self.assertEqual(fake.calls, [])

    def test_full_pagination_and_every_version_is_retained(self):
        fake = FakeS3(
            [
                page("old", True, NextKeyMarker="next", NextVersionIdMarker="old"),
                page("new"),
            ]
        )
        reader = ReadOnlyS3(fake, "bucket", PREFIX)
        result = reader.census()
        self.assertEqual(len(result["rows"]), 2)
        self.assertTrue(result["complete_listing"])
        self.assertFalse(result["point_in_time_snapshot"])
        self.assertEqual(reader.receipt()["physical_request_count"], 2)

    def test_repeated_cursor_duplicate_version_and_request_cap_refuse(self):
        for pages, cap in (
            (
                [
                    page("a", True, NextKeyMarker="x"),
                    page("b", True, NextKeyMarker="x"),
                ],
                10,
            ),
            ([page("a", True, NextKeyMarker="x"), page("a")], 10),
            ([page("a", True, NextKeyMarker="x"), page("b")], 1),
        ):
            with self.subTest(cap=cap):
                reader = ReadOnlyS3(FakeS3(pages), "bucket", PREFIX, max_requests=cap)
                with self.assertRaises(ValueError):
                    reader.census()

    def test_nonfinite_expired_and_body_budget_refuse(self):
        with self.assertRaises(ValueError):
            ReadOnlyS3(FakeS3(), "bucket", PREFIX, seconds=float("inf"))
        fake = FakeS3(body=b"same bytes")
        reader = ReadOnlyS3(fake, "bucket", PREFIX, max_object_bytes=1)
        with self.assertRaises(ValueError):
            reader.capture(row("score_records"))
        self.assertEqual(fake.calls, [])
        reader.end = 0
        with self.assertRaises(TimeoutError):
            reader.census()

    def test_capture_records_real_body_bytes_and_never_uses_etag_as_sha(self):
        body = b"same bytes"
        fake = FakeS3(body=body)
        reader = ReadOnlyS3(fake, "bucket", PREFIX)
        self.assertEqual(reader.capture(row("score_records", body=body)), body)
        self.assertEqual(reader.receipt()["object_body_bytes"], len(body))
        self.assertFalse(reader.receipt()["etag_is_sha256"])


class PackTests(unittest.TestCase):
    def test_payload_corruption_refuses_even_if_outer_archive_sha_is_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = pack_local(
                Path(tmp) / "pack", [(row("score_records"), b"same bytes")]
            )
            manifest = json.loads(Path(receipt["manifest"]).read_bytes())
            stream = io.BytesIO()
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w") as archive:
                    item = tarfile.TarInfo(manifest["blobs"][0]["name"])
                    item.size = 10
                    archive.addfile(item, io.BytesIO(b"different!"))
            raw = stream.getvalue()
            Path(receipt["pack"]).write_bytes(raw)
            manifest["pack_sha256"] = digest(raw)
            with self.assertRaisesRegex(ValueError, "member SHA"):
                verify_local(receipt["pack"], manifest)

    def test_deduplicated_pack_preserves_all_physical_refs_and_reopens(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = b"same bytes"
            members = [
                (row("score_records", body=body), body),
                (row("evaluation", body=body), body),
            ]
            receipt = pack_local(Path(tmp) / "pack", members)
            manifest = json.loads(Path(receipt["manifest"]).read_bytes())
            verified = verify_local(receipt["pack"], manifest)
            self.assertEqual(
                (verified["physical_refs"], verified["verified_blobs"]), (2, 1)
            )
            self.assertEqual(receipt["remote_writes"], 0)

    def test_corrupt_pack_bad_member_hash_and_unknown_refs_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = b"same bytes"
            receipt = pack_local(Path(tmp) / "pack", [(row("score_records"), body)])
            manifest = json.loads(Path(receipt["manifest"]).read_bytes())
            changed = copy.deepcopy(manifest)
            changed["blobs"][0]["sha256"] = "a" * 64
            with self.assertRaises(ValueError):
                verify_local(receipt["pack"], changed)
            changed = copy.deepcopy(manifest)
            changed["refs"][0]["sha256"] = "b" * 64
            with self.assertRaises(ValueError):
                verify_local(receipt["pack"], changed)
            Path(receipt["pack"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "pack SHA"):
                verify_local(receipt["pack"], manifest)

    def test_duplicate_physical_version_and_incomplete_body_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = b"same bytes"
            r = row("score_records")
            with self.assertRaises(ValueError):
                pack_local(Path(tmp) / "one", [(r, body), (r, body)])
            with self.assertRaises(ValueError):
                pack_local(Path(tmp) / "two", [(r, b"short")])

    def test_fsync_failure_never_returns_a_success_or_publishes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "pack"
            with patch(
                "custody_v2.retention.os.fsync",
                side_effect=OSError("injected persistence failure"),
            ):
                with self.assertRaises(OSError):
                    pack_local(dest, [(row("score_records"), b"same bytes")])
            self.assertFalse((dest / "manifest.json").exists())

    def test_existing_destination_and_qualified_manifest_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "pack"
            receipt = pack_local(dest, [(row("score_records"), b"same bytes")])
            with self.assertRaises(FileExistsError):
                pack_local(dest, [(row("score_records"), b"same bytes")])
            manifest = json.loads(Path(receipt["manifest"]).read_bytes())
            manifest["qualified"] = True
            with self.assertRaises(ValueError):
                verify_local(receipt["pack"], manifest)

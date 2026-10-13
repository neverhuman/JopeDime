import datetime
import hashlib
import io
import time
import unittest

from ingest_joined_roster.s3 import ReadOnlyS3


class Client:
    def __init__(self):
        self.calls = []
        self.body = b'{"existing":"metadata"}'
        self.version = "immutable-version"
        self.sha = hashlib.sha256(self.body).hexdigest()
        self.length = len(self.body)
        self.pages = [
            {
                "Contents": [
                    {
                        "Key": "allowed/record.json",
                        "Size": self.length,
                        "LastModified": datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
                    }
                ],
                "IsTruncated": False,
            }
        ]

    def list_objects_v2(self, **kwargs):
        self.calls.append(("ListObjectsV2", kwargs))
        return self.pages.pop(0)

    def head_object(self, **kwargs):
        self.calls.append(("HeadObject", kwargs))
        return {"VersionId": self.version, "ContentLength": self.length, "Metadata": {"sha256": self.sha}}

    def get_object(self, **kwargs):
        self.calls.append(("GetObject", kwargs))
        return {"VersionId": self.version, "ContentLength": self.length, "Body": io.BytesIO(self.body)}

    def put_object(self, **kwargs):
        raise AssertionError("S3 mutation called")

    delete_object = put_object
    put_bucket_lifecycle_configuration = put_object


class ReadOnlyTest(unittest.TestCase):
    def setUp(self):
        self.client = Client()
        self.reader = ReadOnlyS3(
            self.client,
            bucket="fixture",
            prefix="allowed/",
            deadline=time.monotonic() + 60,
            max_requests=20,
            max_bytes=1024,
        )

    def test_read_only_listing_version_and_digest_receipt(self):
        listing = self.reader.list_objects()
        body, ref = self.reader.read_current(listing["objects"][0]["key"], limit=1024)
        self.assertEqual(body, self.client.body)
        self.assertEqual(ref["sha256"], self.client.sha)
        self.assertEqual(self.client.calls[-1][1]["VersionId"], self.client.version)
        receipt = self.reader.receipt()
        self.assertEqual(receipt["s3_writes"], 0)
        self.assertEqual(receipt["verified_get_bytes"], len(body))
        self.assertEqual([c[0] for c in self.client.calls], ["ListObjectsV2", "HeadObject", "GetObject"])
        self.assertFalse(
            any(hasattr(self.reader, name) for name in ("put", "delete", "commit", "transition"))
        )

    def test_prefix_escape_refused_before_any_sdk_call(self):
        with self.assertRaisesRegex(ValueError, "namespace"):
            self.reader.read_current("outside/record.json", limit=1024)
        self.assertEqual(self.client.calls, [])

    def test_missing_version_refuses_get(self):
        self.client.version = "null"
        with self.assertRaisesRegex(ValueError, "VersionId"):
            self.reader.read_current("allowed/record.json", limit=1024)
        self.assertEqual(len(self.client.calls), 1)

    def test_metadata_hash_drift_is_refused(self):
        self.client.sha = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.reader.read_current("allowed/record.json", limit=1024)

    def test_body_length_mismatch_is_refused(self):
        self.client.length += 1
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            self.reader.read_current("allowed/record.json", limit=1024)

    def test_byte_cap_refuses_get(self):
        self.reader.max_bytes = 1
        with self.assertRaisesRegex(ValueError, "byte cap"):
            self.reader.read_current("allowed/record.json", limit=1024)
        self.assertEqual(len(self.client.calls), 1)

    def test_request_and_deadline_caps_refuse_before_sdk(self):
        self.reader.max_requests = 0
        with self.assertRaisesRegex(ValueError, "request cap"):
            self.reader.list_objects()
        self.reader.max_requests = 20
        self.reader.deadline = time.monotonic() - 1
        with self.assertRaises(TimeoutError):
            self.reader.list_objects()
        self.assertEqual(self.client.calls, [])

    def test_repeated_cursor_and_unmarked_partial_listing_refused(self):
        self.client.pages = [{"IsTruncated": True, "NextContinuationToken": "same"}] * 2
        with self.assertRaisesRegex(ValueError, "repeated"):
            self.reader.list_objects()
        self.client.pages = [{}]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.reader.list_objects()

    def test_listing_drift_and_boolean_size_refused(self):
        self.client.pages[0]["Contents"][0]["Size"] = True
        with self.assertRaisesRegex(ValueError, "size identity"):
            self.reader.list_objects()

    def test_mutation_dispatch_refused(self):
        for operation in ("PutObject", "DeleteObject", "PutBucketLifecycleConfiguration", "RestoreObject"):
            with self.assertRaisesRegex(ValueError, "read-only"):
                self.reader._request(operation)
        self.assertEqual(self.client.calls, [])

    def test_partial_or_escaping_prefix_inventory_is_refused(self):
        self.client.pages = [{"IsTruncated": True, "CommonPrefixes": [{"Prefix": "allowed/one/"}]}]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.reader.list_prefixes()
        self.client.pages = [{"IsTruncated": False, "CommonPrefixes": [{"Prefix": "outside/"}]}]
        with self.assertRaisesRegex(ValueError, "escape"):
            self.reader.list_prefixes()
        self.client.pages = [{"IsTruncated": False, "CommonPrefixes": [{"Prefix": "allowed/one/"}]}]
        self.assertEqual(self.reader.list_prefixes(), ["allowed/one/"])


if __name__ == "__main__":
    unittest.main()

"""Offline proofs for the s3io guard, publish helpers, G18 evaluator and the P0 probe (no network, fake keys)."""
import base64
import datetime
import hashlib
import io
import json
import os
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import boto3
import botocore
from botocore.response import StreamingBody
from botocore.stub import Stubber

from dltrain import common, s3io

B, P, V = s3io.BUCKET, s3io.PREFIX_KEY, s3io.V2_PREFIX_KEY


def _session():
    return boto3.session.Session(aws_access_key_id="testing", aws_secret_access_key="testing", region_name="us-east-2")


def _client(mode="lane", writes=False):
    c = s3io.make_client(mode=mode, writes=writes, session=_session())

    def tripwire(**_):
        raise AssertionError("a request reached before-send in an offline test")
    c.meta.events.register("before-send.*.*", tripwire)
    return c


def _body(data: bytes):
    return StreamingBody(io.BytesIO(data), len(data))


def _b64(body: bytes) -> str:
    return base64.b64encode(hashlib.sha256(body).digest()).decode()


def _etag(n: int) -> str:
    return '"' + f"{n:032x}" + '"'


def put_kw(key=P + "x", body=b"1", **kw):
    base = {"Bucket": B, "Key": key, "Body": body, "ChecksumAlgorithm": "SHA256", "ChecksumSHA256": _b64(body)}
    base.update(kw)
    return base


class ReadClientRefusals(unittest.TestCase):
    def setUp(self):
        self.c = _client()

    def refused(self, fn, regex=None, **kw):
        if regex:
            with self.assertRaisesRegex(s3io.S3Refused, regex):
                fn(**kw)
        else:
            with self.assertRaises(s3io.S3Refused):
                fn(**kw)

    def test_deletes_configuration_admin_refused(self):
        self.refused(self.c.delete_object, "not allowed", Bucket=B, Key=P + "x")
        self.refused(self.c.delete_objects, "not allowed", Bucket=B, Delete={"Objects": [{"Key": P + "x"}]})
        self.refused(self.c.put_bucket_lifecycle_configuration, "not allowed", Bucket=B, LifecycleConfiguration={"Rules": []})
        self.refused(self.c.put_bucket_policy, "not allowed", Bucket=B, Policy="{}")
        self.refused(self.c.get_bucket_lifecycle_configuration, "not allowed", Bucket=B)
        self.refused(self.c.list_buckets, "not allowed")
        self.refused(self.c.put_object_acl, "not allowed", Bucket=B, Key=P + "x", ACL="private")
        self.refused(self.c.copy_object, "not allowed", Bucket=B, Key=P + "y", CopySource={"Bucket": B, "Key": P + "x"})
        self.refused(self.c.put_object_retention, "not allowed", Bucket=B, Key=P + "x", Retention={"Mode": "GOVERNANCE"})
        self.refused(self.c.put_object_legal_hold, "not allowed", Bucket=B, Key=P + "x", LegalHold={"Status": "ON"})
        self.refused(self.c.create_multipart_upload, "not allowed", Bucket=B, Key=P + "x")

    def test_read_client_cannot_write(self):
        self.refused(self.c.put_object, "read-only", **put_kw(IfNoneMatch="*"))

    def test_unpinned_reads_outside_prefix_refused(self):
        self.refused(self.c.get_object, "without an exact VersionId", Bucket=B, Key="experiments/gp-teacher-v6/x.json")
        self.refused(self.c.head_object, "without an exact VersionId", Bucket="veox-jopedime-datasets-use2", Key="datasets/x")
        self.refused(self.c.get_object, "'null'", Bucket=B, Key=P + "x", VersionId="null")
        self.refused(self.c.get_object, "without an exact VersionId", Bucket="veox-jopedime-datasets-use2",
                     Key="research/regression-2k/incoming/v1/latest.json")

    def test_unbounded_list_refused(self):
        self.refused(self.c.list_objects_v2, "MaxKeys", Bucket=B, Prefix=P)
        self.refused(self.c.list_objects_v2, "MaxKeys", Bucket=B, Prefix=P, MaxKeys=5000)
        self.refused(self.c.list_objects_v2, "Prefix", Bucket=B, MaxKeys=10)

    def test_version_history_exact_key_only(self):
        key = P + "bindings/gp-policy-dl-v1-e0/holdout-commitment.json"
        with Stubber(self.c) as st:
            st.add_response("list_object_versions", {"Versions": [], "IsTruncated": False},
                            {"Bucket": B, "Prefix": key, "MaxKeys": 16})
            self.c.list_object_versions(Bucket=B, Prefix=key, MaxKeys=16)
            st.assert_no_pending_responses()
        self.refused(self.c.list_object_versions, "only Bucket, Prefix and MaxKeys", Bucket=B, Prefix=key, MaxKeys=16, KeyMarker="a")
        self.refused(self.c.list_object_versions, "only Bucket, Prefix and MaxKeys", Bucket="veox-jopedime-models-use2", Prefix=key, MaxKeys=16)
        self.refused(self.c.list_object_versions, "only Bucket, Prefix and MaxKeys", Bucket=B, Prefix=key)
        self.refused(self.c.list_object_versions, "one full key inside PREFIX", Bucket=B, Prefix=P, MaxKeys=16)
        self.refused(self.c.list_object_versions, "one full key inside PREFIX", Bucket=B, Prefix=P + "bindings/", MaxKeys=16)
        self.refused(self.c.list_object_versions, "one full key inside PREFIX", Bucket=B, Prefix="experiments/gp-teacher-v6/x.json", MaxKeys=16)
        self.refused(self.c.list_object_versions, "one full key inside PREFIX", Bucket=B, Prefix=P + "../x.json", MaxKeys=16)
        self.refused(self.c.list_object_versions, "MaxKeys", Bucket=B, Prefix=key, MaxKeys=101)
        self.refused(self.c.list_object_versions, "MaxKeys", Bucket=B, Prefix=key, MaxKeys=0)

    def test_presigning_refused_everywhere(self):
        self.refused(self.c.generate_presigned_url, "presigning", ClientMethod="get_object", Params={"Bucket": B, "Key": P + "x"})
        with self.assertRaisesRegex(s3io.S3Refused, "presigning"):
            self.c.generate_presigned_post(B, P + "x")
        with self.assertRaisesRegex(s3io.S3Refused, "presigned|signing"):  # class-level presigner hits before-sign
            type(self.c).generate_presigned_url(self.c, "get_object", Params={"Bucket": "veox-jopedime-models-use2", "Key": "b/x",
                                                                             "VersionId": "abc"}, HttpMethod="DELETE")


class WriteClientRefusals(unittest.TestCase):
    def setUp(self):
        self.c = _client(writes=True)

    def refused(self, regex, **kw):
        with self.assertRaisesRegex(s3io.S3Refused, regex):
            self.c.put_object(**kw)

    def test_condition_rules(self):
        self.refused("exactly one condition", **put_kw())
        self.refused("exactly one condition", **put_kw(IfNoneMatch="*", IfMatch=_etag(1)))
        self.refused("IfNoneMatch must be", **put_kw(IfNoneMatch="x"))
        self.refused("strong ETag", **put_kw(key=P + "runs/r/resume/HEAD.json", IfMatch="*"))
        self.refused("strong ETag", **put_kw(key=P + "runs/r/resume/HEAD.json", IfMatch=" "))
        self.refused("only on CAS pointer keys", **put_kw(key=P + "objects/epoch/sha256/x.json", IfMatch=_etag(1)))

    def test_prefix_and_bucket_rules(self):
        self.refused("outside PREFIX", **put_kw(key="experiments/gp-teacher-v6/x", IfNoneMatch="*"))
        self.refused("outside PREFIX", **put_kw(key=P + "../escape", IfNoneMatch="*"))
        with self.assertRaisesRegex(s3io.S3Refused, "outside PREFIX"):
            self.c.put_object(**(put_kw(IfNoneMatch="*") | {"Bucket": "veox-jopedime-models-use2"}))

    def test_allowlist_and_checksum_rules(self):
        self.refused("allowlist", **put_kw(IfNoneMatch="*", Tagging="a=b"))
        self.refused("allowlist", **put_kw(IfNoneMatch="*", StorageClass="GLACIER"))
        self.refused("allowlist", **put_kw(IfNoneMatch="*", ObjectLockMode="GOVERNANCE"))
        self.refused("allowlist", **put_kw(IfNoneMatch="*", SSECustomerAlgorithm="AES256", SSECustomerKey="k" * 32))
        self.refused("allowlist", **put_kw(IfNoneMatch="*", ServerSideEncryption="aws:kms"))
        with self.assertRaisesRegex(s3io.S3Refused, "Checksum"):
            self.c.put_object(Bucket=B, Key=P + "x", Body=b"1", IfNoneMatch="*")
        self.refused("Checksum", **put_kw(IfNoneMatch="*", ChecksumSHA256=_b64(b"other")))
        self.refused("Checksum", **put_kw(IfNoneMatch="*", ChecksumAlgorithm="CRC32"))

    def test_allowed_create_only_put_passes_the_guard(self):
        body = b'{"a":1}'
        with Stubber(self.c) as stub:
            stub.add_response("put_object", {"VersionId": "v1", "ETag": _etag(1)}, put_kw(P + "x", body, IfNoneMatch="*"))
            self.c.put_object(**put_kw(P + "x", body, IfNoneMatch="*"))
            stub.assert_no_pending_responses()


class GuardMechanics(unittest.TestCase):
    def test_request_stage_and_send_stage(self):
        class M:
            name = "PutObject"
        g = s3io.Guard(writes=True)
        with self.assertRaises(s3io.S3Refused):
            g.check_request(model=M(), params={"headers": {"Content-Type": "application/json"}})
        g.check_request(model=M(), params={"headers": {"If-None-Match": "*"}})

        class R:
            def __init__(self, url, method="GET"):
                self.url, self.method = url, method
        g.check_send(request=R(f"https://{B}.s3.us-east-2.amazonaws.com/{P}x"))
        g.check_send(request=R(f"https://s3.us-east-2.amazonaws.com/{B}/{P}x"))
        for bad in ("http://127.0.0.1:7450/x", "https://evil.example.com/x", "http://s3.us-east-2.amazonaws.com/x"):
            with self.assertRaises(s3io.S3Refused):
                g.check_send(request=R(bad))
        with self.assertRaises(s3io.S3Refused):
            g.check_sign(request=R("https://s3.us-east-2.amazonaws.com/", "DELETE"), operation_name="GetObject", signature_version="s3v4")
        with self.assertRaises(s3io.S3Refused):
            g.check_sign(request=R("https://s3.us-east-2.amazonaws.com/", "GET"), operation_name="GetObject", signature_version="s3v4-query")
        g.check_sign(request=R("https://s3.us-east-2.amazonaws.com/", "GET"), operation_name="GetObject", signature_version="s3v4")

    def test_configured_endpoint_overrides_are_ignored(self):
        with mock.patch.dict(os.environ, {"AWS_ENDPOINT_URL_S3": "http://127.0.0.1:7450", "AWS_ENDPOINT_URL": "http://127.0.0.1:7464"}):
            c = s3io.make_client(session=_session())
            self.assertEqual(c.meta.endpoint_url, "https://s3.us-east-2.amazonaws.com")
            s = s3io.sts_client(session=_session())
            self.assertEqual(s.meta.endpoint_url, s3io.STS_ENDPOINT)

    def test_retry_budgets(self):
        self.assertEqual(_client(writes=True).meta.config.retries.get("total_max_attempts"), 1)
        self.assertEqual(_client(mode="lifecycle-once").meta.config.retries.get("total_max_attempts"), 1)
        self.assertEqual(_client().meta.config.retries.get("total_max_attempts"), 3)

    def test_latest_pointer_head_is_the_only_unpinned_outside_read(self):
        c = _client()
        with Stubber(c) as stub:
            stub.add_response("head_object", {"VersionId": "lv1"}, {"Bucket": "veox-jopedime-datasets-use2",
                                                                    "Key": "research/regression-2k/incoming/v1/latest.json"})
            c.head_object(Bucket="veox-jopedime-datasets-use2", Key="research/regression-2k/incoming/v1/latest.json")


class LifecycleOnce(unittest.TestCase):
    def test_exactly_one_lifecycle_read(self):
        c = _client(mode="lifecycle-once")
        with self.assertRaisesRegex(s3io.S3Refused, "presigning"):
            c.generate_presigned_url("get_bucket_lifecycle_configuration", Params={"Bucket": B}, HttpMethod="DELETE")
        self.assertFalse(c._dltrain_guard.lifecycle_used)
        with Stubber(c) as stub:
            stub.add_response("get_bucket_lifecycle_configuration", {"Rules": []}, {"Bucket": B})
            c.get_bucket_lifecycle_configuration(Bucket=B)
            for fn, kw in ((c.get_bucket_lifecycle_configuration, {"Bucket": B}), (c.get_object, {"Bucket": B, "Key": P + "x"}),
                           (c.put_object, put_kw(IfNoneMatch="*"))):
                with self.assertRaises(s3io.S3Refused):
                    fn(**kw)

    def test_readback_normalizes_dates_and_records_http(self):
        c = _client(mode="lifecycle-once")
        rules = [{"ID": "dated", "Status": "Enabled", "Filter": {"Prefix": "staging/"},
                  "Expiration": {"Date": datetime.datetime(2027, 1, 1, tzinfo=datetime.timezone.utc)}},
                 {"ID": "abort-mpu", "Status": "Enabled", "Filter": {}, "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7}}]
        with Stubber(c) as stub:
            stub.add_response("get_bucket_lifecycle_configuration", {"Rules": rules}, {"Bucket": B})
            out = s3io.lifecycle_readback(client=c)
        self.assertEqual(out["verdict"], "PASS")
        self.assertEqual(out["raw_rules"][0]["Expiration"]["Date"], "2027-01-01T00:00:00Z")
        common.receipt_bytes({"schema": "x"} | out)   # serializable, scanner-clean

    def test_unexpected_error_is_unknown_not_pass(self):
        c = _client(mode="lifecycle-once")
        with Stubber(c) as stub:
            stub.add_client_error("get_bucket_lifecycle_configuration", service_error_code="AccessDenied", http_status_code=403)
            out = s3io.lifecycle_readback(client=c)
        self.assertEqual(out["verdict"], "UNKNOWN")


class LifecycleEvaluation(unittest.TestCase):
    def ev(self, rules):
        return s3io.evaluate_lifecycle(rules)["verdict"]

    def test_cases(self):
        self.assertEqual(self.ev([{"ID": "abort-mpu", "Status": "Enabled", "Filter": {}, "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7}}]), "PASS")
        self.assertEqual(self.ev([{"ID": "e", "Status": "Enabled", "Filter": {}, "Expiration": {"Days": 30}}]), "FAIL")
        self.assertEqual(self.ev([{"ID": "s", "Status": "Enabled", "Filter": {"Prefix": "staging/"}, "Transitions": [{"Days": 30}]}]), "PASS")
        self.assertEqual(self.ev([{"ID": "n", "Status": "Enabled", "Filter": {"Prefix": "experiments/"}, "NoncurrentVersionExpiration": {"NoncurrentDays": 1}}]), "FAIL")
        self.assertEqual(self.ev([{"ID": "c", "Status": "Enabled", "Filter": {"Prefix": P + "runs/"}, "Expiration": {"Days": 1}}]), "FAIL")
        self.assertEqual(self.ev([{"ID": "t", "Status": "Enabled", "Filter": {"Tag": {"Key": "a", "Value": "b"}}, "Transitions": [{"Days": 1}]}]), "FAIL")
        self.assertEqual(self.ev([{"ID": "dm", "Status": "Enabled", "Filter": {}, "Expiration": {"ExpiredObjectDeleteMarker": True}}]), "PASS")
        self.assertEqual(self.ev([{"ID": "and", "Status": "Enabled", "Filter": {"And": {"Prefix": "staging/", "Tags": [{"Key": "a", "Value": "b"}]}},
                                   "Expiration": {"Days": 1}}]), "PASS")

    def test_disabled_and_unknown_status_and_keys_fail_closed(self):
        self.assertEqual(self.ev([{"ID": "off", "Status": "Disabled", "Filter": {}, "Expiration": {"Days": 1}}]), "OWNER_REVIEW")
        self.assertEqual(self.ev([{"ID": "lower", "Status": "enabled", "Filter": {}, "Expiration": {"Days": 1}}]), "FAIL")
        self.assertEqual(self.ev([{"ID": "nostatus", "Filter": {}, "NoncurrentVersionExpiration": {"NoncurrentDays": 1}}]), "FAIL")
        self.assertEqual(self.ev([{"ID": "v1", "Status": "Enabled", "Prefix": "", "Transition": {"Days": 1}}]), "FAIL")
        self.assertEqual(self.ev([{"ID": "cfn", "Status": "Enabled", "Filter": {}, "ExpirationInDays": 1}]), "FAIL")


class PublishHelpers(unittest.TestCase):
    def test_create_only_then_exact_readback(self):
        w, r = _client(writes=True), _client()
        body = b'{"k":1}'
        with Stubber(w) as sw, Stubber(r) as sr:
            sw.add_response("put_object", {"VersionId": "v1", "ETag": _etag(1)})
            sr.add_response("get_object", {"Body": _body(body), "VersionId": "v1"}, {"Bucket": B, "Key": P + "k", "VersionId": "v1"})
            out = s3io.put_create_only(w, P + "k", body, r)
        self.assertEqual(out["outcome"], "created")

    def test_412_equal_bytes_is_success_and_different_is_poisoned(self):
        for existing, expect in ((b'{"k":1}', "ok"), (b"other", "poisoned")):
            w, r = _client(writes=True), _client()
            with Stubber(w) as sw, Stubber(r) as sr:
                sw.add_client_error("put_object", service_error_code="PreconditionFailed", http_status_code=412)
                sr.add_response("get_object", {"Body": _body(existing), "VersionId": "v0"}, {"Bucket": B, "Key": P + "k"})
                if expect == "ok":
                    self.assertEqual(s3io.put_create_only(w, P + "k", b'{"k":1}', r)["outcome"], "already_present_equal_bytes")
                else:
                    with self.assertRaises(s3io.PoisonedKey):
                        s3io.put_create_only(w, P + "k", b'{"k":1}', r)

    def test_readback_mismatch_and_null_version(self):
        w, r = _client(writes=True), _client()
        with Stubber(w) as sw, Stubber(r) as sr:
            sw.add_response("put_object", {"VersionId": "v1", "ETag": _etag(1)})
            sr.add_response("get_object", {"Body": _body(b"corrupt"), "VersionId": "v1"}, {"Bucket": B, "Key": P + "k", "VersionId": "v1"})
            with self.assertRaises(s3io.ReadbackMismatch):
                s3io.put_create_only(w, P + "k", b"intended", r)
        w = _client(writes=True)
        with Stubber(w) as sw:
            sw.add_response("put_object", {"VersionId": "null", "ETag": _etag(1)})
            with self.assertRaises(RuntimeError):
                s3io.put_create_only(w, P + "k", b"x", _client())


class ProbeEndToEnd(unittest.TestCase):
    """run_probe with stubbed clients: PASS baseline, a crashed racer (FAIL), a stray step-4 write (FAIL)."""
    NONCE = b"\x01" * 8
    ATTEMPT = P + "probes/T/attempt-1/"

    def bodies(self):
        n = self.NONCE.hex()
        return {"A": common.canonical({"probe": "A", "claim_ts": "T", "nonce": n}),
                "v1": common.canonical({"probe": "P", "v": 1, "nonce": n}), "v2": common.canonical({"probe": "P", "v": 2, "nonce": n}),
                "race": [common.canonical({"probe": "P", "v": f"race-{i}", "nonce": n}) for i in range(8)]}

    def build(self, step4_written=False, crash_racer=None, winner=3):
        bd = self.bodies()
        w, r = _client(writes=True), _client()
        racers = [_client(writes=True) for _ in range(8)]
        sw, sr = Stubber(w), Stubber(r)
        ok = lambda v, e: {"VersionId": v, "ETag": _etag(e), "ResponseMetadata": {"HTTPStatusCode": 200}}  # noqa: E731
        sw.add_response("put_object", ok("va1", 1))
        sw.add_client_error("put_object", service_error_code="PreconditionFailed", http_status_code=412)
        sw.add_response("put_object", ok("vp1", 2))
        sw.add_response("put_object", ok("vp2", 3))
        sw.add_client_error("put_object", service_error_code="PreconditionFailed", http_status_code=412)
        if step4_written:
            sw.add_response("put_object", ok("vq", 4))
        else:
            sw.add_client_error("put_object", service_error_code="NoSuchKey", http_status_code=404)
        sr.add_response("head_object", {"VersionId": "va1"})
        sr.add_response("get_object", {"Body": _body(bd["race"][winner]), "VersionId": "vw"})
        if step4_written:
            sr.add_response("head_object", {"VersionId": "vq"})
        else:
            sr.add_client_error("head_object", service_error_code="404", http_status_code=404)
        for _ in range(4):
            sr.add_response("head_object", {})
        for b, v in ((bd["A"], "va1"), (bd["v1"], "vp1"), (bd["v2"], "vp2"), (bd["race"][winner], "vw")):
            sr.add_response("get_object", {"Body": _body(b), "VersionId": v})
        stubs = [sw, sr]
        for i, rc in enumerate(racers):
            if i == crash_racer:
                rc.head_object = lambda **_: {}
                def boom(**_):
                    raise botocore.exceptions.ReadTimeoutError(endpoint_url="https://x")
                rc.put_object = boom
                continue
            s = Stubber(rc)
            s.add_response("head_object", {})
            if i == winner:
                s.add_response("put_object", ok("vw", 9))
            else:
                s.add_client_error("put_object", service_error_code="PreconditionFailed", http_status_code=412)
            stubs.append(s)
        for s in stubs:
            s.activate()
        return {"writer": w, "reader": r, "racers": racers}

    def run_probe(self, **kw):
        clients = self.build(**kw)
        with mock.patch.object(s3io.os, "urandom", return_value=self.NONCE):
            return s3io.run_probe("T", self.ATTEMPT, 1, clients, s3io.Journal())

    def test_pass_baseline(self):
        res = self.run_probe()
        self.assertEqual(res["verdict"], "PASS", json.dumps(res["steps"], default=str)[:2000])
        self.assertEqual(res["objects_written"]["versions"], 4)
        self.assertEqual(res["steps"]["3_race"]["winners"], 1)

    def test_crashed_racer_fails_closed(self):
        res = self.run_probe(crash_racer=5)
        self.assertEqual(res["verdict"], "FAIL")
        self.assertEqual(res["steps"]["3_race"]["unknown"], 1)
        self.assertEqual(res["objects_written"]["unknown_outcomes"], 1)

    def test_stray_step4_write_fails(self):
        res = self.run_probe(step4_written=True)
        self.assertEqual(res["verdict"], "FAIL")
        self.assertFalse(res["steps"]["4_ifmatch_missing_key"]["pass"])
        self.assertEqual(res["objects_written"]["versions"], 5)


class ContentAddressedReads(unittest.TestCase):
    KEY = "blobs/sha256/" + hashlib.sha256(b"meta").hexdigest()

    def test_cas_client_allows_only_content_addressed_gets(self):
        c = _client(mode="cas-read")
        for fn, kw in ((c.get_object, {"Bucket": "veox-jopedime-datasets-use2", "Key": "datasets/x/manifest.json"}),
                       (c.get_object, {"Bucket": B, "Key": self.KEY}),
                       (c.get_object, {"Bucket": "veox-jopedime-datasets-use2", "Key": self.KEY + ".json"}),
                       (c.get_object, {"Bucket": "veox-jopedime-datasets-use2", "Key": self.KEY, "VersionId": "v"}),
                       (c.head_object, {"Bucket": "veox-jopedime-datasets-use2", "Key": self.KEY}),
                       (c.put_object, put_kw(IfNoneMatch="*"))):
            with self.assertRaises(s3io.S3Refused):
                fn(**kw)
        with self.assertRaises(s3io.S3Refused):
            _client().get_object(Bucket="veox-jopedime-datasets-use2", Key=self.KEY)   # the lane client still refuses

    def test_body_must_hash_to_the_key(self):
        c = _client(mode="cas-read")
        with Stubber(c) as stub:
            stub.add_response("get_object", {"Body": _body(b"meta"), "VersionId": "v1"}, {"Bucket": "veox-jopedime-datasets-use2", "Key": self.KEY})
            stub.add_response("get_object", {"Body": _body(b"tampered"), "VersionId": "v2"}, {"Bucket": "veox-jopedime-datasets-use2", "Key": self.KEY})
            body, meta = s3io.get_content_addressed(c, "veox-jopedime-datasets-use2", self.KEY)
            self.assertEqual((body, meta["VersionId"]), (b"meta", "v1"))
            with self.assertRaises(s3io.ReadbackMismatch):
                s3io.get_content_addressed(c, "veox-jopedime-datasets-use2", self.KEY)
        with self.assertRaises(s3io.S3Refused):
            s3io.get_content_addressed(_client(), "veox-jopedime-datasets-use2", self.KEY)


# ------------------------------------------------------------------------------------------- v2 train-only (L6(e))
V2_KEY = V + "encodings/a0/0123456789abcdef/f0/aaaaaaaaaaaa-bbbbbbbbbbbb-cccccccccccc.f32"
V2_RESEARCH_KEY = V + "research-only/encodings/tabpfn35/0123456789abcdef/f0/aaaaaaaaaaaa-bbbbbbbbbbbb-cccccccccccc.npy"
CAP = s3io.V2_BUDGET_BYTES


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class _TmpLedger(unittest.TestCase):
    """Every v2 test owns a fresh ledger under TMPDIR; the real V2_LEDGER_PATH is never touched."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="s3io-v2-")
        self.ledger_path = os.path.join(self.tmp, "s3-v2-ledger.jsonl")
        self.assertNotEqual(os.path.realpath(self.ledger_path), str(s3io.V2_LEDGER_PATH))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def v2_guard(self):
        return s3io.Guard(writes=True, v2=True, ledger=self.ledger_path)

    def v2_client(self, writes=True):
        c = s3io.make_client(writes=writes, v2=True, ledger_path=self.ledger_path if writes else None, session=_session())

        def tripwire(**_):
            raise AssertionError("a request reached before-send in an offline test")
        c.meta.events.register("before-send.*.*", tripwire)
        return c

    def put(self, guard, **params):
        guard.check_params(params, types.SimpleNamespace(name="PutObject"))

    def usage(self):
        return s3io.V2Ledger(self.ledger_path).usage()


class V2WriterGuard(_TmpLedger):
    def test_accepts_create_only_puts_under_v2(self):
        c = self.v2_client()
        for key, body in ((V2_KEY, b"\x00\x01\x02\x03"), (V2_RESEARCH_KEY, b"npy-bytes"),
                          (V + "receipts/a0/0123456789abcdef/f0/x.json", b'{"r":1}')):
            with Stubber(c) as stub:
                stub.add_response("put_object", {"VersionId": "v1", "ETag": _etag(1)}, put_kw(key, body, IfNoneMatch="*"))
                c.put_object(**put_kw(key, body, IfNoneMatch="*"))
                stub.assert_no_pending_responses()
        u = self.usage()
        self.assertEqual((u["reservations"], u["used_bytes"]), (3, 4 + 9 + 7))

    def test_put_create_only_with_exact_version_readback(self):
        w, r = self.v2_client(), self.v2_client(writes=False)
        body = b"payload"
        with Stubber(w) as sw, Stubber(r) as sr:
            sw.add_response("put_object", {"VersionId": "v1", "ETag": _etag(1)})
            sr.add_response("get_object", {"Body": _body(body), "VersionId": "v1"}, {"Bucket": B, "Key": V2_KEY, "VersionId": "v1"})
            out = s3io.put_create_only(w, V2_KEY, body, r, content_type="application/octet-stream")
        self.assertEqual((out["outcome"], out["sha256"], out["bytes"]), ("created", _sha(body), len(body)))
        # 412 with equal bytes: the unpinned GET needs a v2 reader; the default reader refuses it
        for reader, ok in ((self.v2_client(writes=False), True), (_client(), False)):
            w = self.v2_client()
            with Stubber(w) as sw, Stubber(reader) as sr:
                sw.add_client_error("put_object", service_error_code="PreconditionFailed", http_status_code=412)
                if ok:
                    sr.add_response("get_object", {"Body": _body(body), "VersionId": "v1"}, {"Bucket": B, "Key": V2_KEY})
                    self.assertEqual(s3io.put_create_only(w, V2_KEY, body, reader)["outcome"], "already_present_equal_bytes")
                else:
                    with self.assertRaisesRegex(s3io.S3Refused, "without an exact VersionId"):
                        s3io.put_create_only(w, V2_KEY, body, reader)

    def test_refuses_v1_keys(self):
        g = self.v2_guard()
        for key in (P + "objects/receipt/sha256/" + "0" * 64 + ".json", P + "runs/r/resume/HEAD.json", P + "probes/T/attempt-1/A.json",
                    P + "encodings/a0/0123456789abcdef/f0/x.f32"):
            with self.assertRaisesRegex(s3io.S3Refused, "^PutObject outside V2_PREFIX"):
                self.put(g, **put_kw(key, b"1", IfNoneMatch="*"))
        with self.assertRaisesRegex(s3io.S3Refused, "^PutObject outside V2_PREFIX"):
            self.v2_client().put_object(**put_kw(P + "objects/x.json", b"1", IfNoneMatch="*"))
        self.assertEqual(self.usage()["reservations"], 0)

    def test_refuses_keys_outside_both_prefixes(self):
        g = self.v2_guard()
        bad = ["experiments/gp-teacher-v6/encodings/hyperion/0123456789abcdef/f0/x.npy",
               "experiments/gp-policy-dl/v2-trainonlyX/encodings/a0/0123456789abcdef/f0/x.f32",
               "experiments/gp-policy-dl/v3-trainonly/encodings/a0/0123456789abcdef/f0/x.f32",
               "staging/experiments/gp-policy-dl/v2-trainonly/encodings/a0/x.f32",
               "experiments/gp-policy-dl/v2-trainonly", V, V + "encodings//a0/x.f32", V + "encodings/a0/",
               V + "encodings/../../v1/objects/x.json", V + "encodings/./a0/x.f32", V + "encodings/a0/x y.f32", V + "a\\b.f32"]
        for key in bad:
            with self.assertRaisesRegex(s3io.S3Refused, "^PutObject outside V2_PREFIX", msg=key):
                self.put(g, **put_kw(key, b"1", IfNoneMatch="*"))
        for bucket in ("veox-jopedime-models-use2", "veox-jopedime-datasets-use2", "veox-jopedime-learning-usw2"):
            with self.assertRaisesRegex(s3io.S3Refused, "^PutObject outside V2_PREFIX"):
                self.put(g, **(put_kw(V2_KEY, b"1", IfNoneMatch="*") | {"Bucket": bucket}))
        self.assertEqual(self.usage()["reservations"], 0)

    def test_requires_if_none_match_and_checksum(self):
        g = self.v2_guard()
        head = V + "runs/r/resume/HEAD.json"
        for kw in ({}, {"IfMatch": _etag(1)}, {"IfNoneMatch": "*", "IfMatch": _etag(1)}, {"IfNoneMatch": "x"}, {"IfNoneMatch": None}):
            for key in (V2_KEY, head):
                with self.assertRaisesRegex(s3io.S3Refused, "create-only", msg=f"{key} {kw}"):
                    self.put(g, **put_kw(key, b"1", **kw))
        with self.assertRaisesRegex(s3io.S3Refused, "Checksum"):
            self.put(g, Bucket=B, Key=V2_KEY, Body=b"1", IfNoneMatch="*")
        with self.assertRaisesRegex(s3io.S3Refused, "Checksum"):
            self.put(g, Bucket=B, Key=V2_KEY, Body=b"1", IfNoneMatch="*", ChecksumAlgorithm="SHA256")
        for kw in ({"ChecksumSHA256": _b64(b"other")}, {"ChecksumAlgorithm": "CRC32"}):
            with self.assertRaisesRegex(s3io.S3Refused, "Checksum"):
                self.put(g, **put_kw(V2_KEY, b"1", IfNoneMatch="*", **kw))
        for kw in ({"Tagging": "a=b"}, {"StorageClass": "GLACIER"}, {"ObjectLockMode": "GOVERNANCE"}, {"ACL": "public-read"},
                   {"ServerSideEncryption": "aws:kms"}, {"Expires": datetime.datetime(2030, 1, 1)}):
            with self.assertRaisesRegex(s3io.S3Refused, "allowlist"):
                self.put(g, **put_kw(V2_KEY, b"1", IfNoneMatch="*", **kw))
        with self.assertRaisesRegex(s3io.S3Refused, "Body must be bytes"):
            self.put(g, **(put_kw(V2_KEY, b"1", IfNoneMatch="*") | {"Body": io.BytesIO(b"1")}))
        c = self.v2_client()   # the same refusals through a real (offline) client
        with self.assertRaisesRegex(s3io.S3Refused, "create-only"):
            c.put_object(**put_kw(head, b"1", IfMatch=_etag(1)))
        with self.assertRaisesRegex(s3io.S3Refused, "create-only"):
            c.put_object(**put_kw(V2_KEY, b"1"))
        with self.assertRaisesRegex(s3io.S3Refused, "Checksum"):
            c.put_object(Bucket=B, Key=V2_KEY, Body=b"1", IfNoneMatch="*")
        self.assertEqual(self.usage()["reservations"], 0, "refused PUTs must not be charged to the budget")

    def test_request_stage(self):
        class M:
            name = "PutObject"
        g = self.v2_guard()
        vh = f"https://{B}.{s3io.S3_HOST}/"
        ps = f"https://{s3io.S3_HOST}/{B}/"
        g.check_request(model=M(), params={"headers": {"If-None-Match": "*"}, "url": vh + V2_KEY})
        g.check_request(model=M(), params={"headers": {"If-None-Match": "*"}, "url": ps + V2_KEY})
        for params in ({"headers": {"If-Match": _etag(1)}, "url": vh + V2_KEY},
                       {"headers": {"If-None-Match": "*", "If-Match": _etag(1)}, "url": vh + V2_KEY},
                       {"headers": {"If-None-Match": "x"}, "url": vh + V2_KEY}):
            with self.assertRaisesRegex(s3io.S3Refused, "If-None-Match: \\* and no If-Match"):
                g.check_request(model=M(), params=params)
        for url in (vh + P + "objects/x.json", ps + "experiments/gp-teacher-v6/x", None, "https://evil.example.com/" + V2_KEY,
                    f"https://veox-jopedime-models-use2.{s3io.S3_HOST}/" + V2_KEY):
            with self.assertRaisesRegex(s3io.S3Refused, "not a V2_PREFIX key"):
                g.check_request(model=M(), params={"headers": {"If-None-Match": "*"}, "url": url})
        d = s3io.Guard(writes=True)
        d.check_request(model=M(), params={"headers": {"If-None-Match": "*"}, "url": vh + P + "x"})
        d.check_request(model=M(), params={"headers": {"If-None-Match": "*"}})
        for url in (vh + V2_KEY, ps + V2_RESEARCH_KEY, vh + V2_KEY.replace("v2-trainonly", "v2%2Dtrainonly")):
            with self.assertRaisesRegex(s3io.S3Refused, "default writer refuses"):
                d.check_request(model=M(), params={"headers": {"If-None-Match": "*"}, "url": url})

    def test_refuses_deletes_admin_and_other_writes(self):
        c = self.v2_client()
        calls = ((c.delete_object, {"Bucket": B, "Key": V2_KEY}),
                 (c.delete_object, {"Bucket": B, "Key": V2_KEY, "VersionId": "v1"}),
                 (c.delete_objects, {"Bucket": B, "Delete": {"Objects": [{"Key": V2_KEY}]}}),
                 (c.put_object_acl, {"Bucket": B, "Key": V2_KEY, "ACL": "private"}),
                 (c.put_object_tagging, {"Bucket": B, "Key": V2_KEY, "Tagging": {"TagSet": []}}),
                 (c.delete_object_tagging, {"Bucket": B, "Key": V2_KEY}),
                 (c.put_object_retention, {"Bucket": B, "Key": V2_KEY, "Retention": {"Mode": "GOVERNANCE"}}),
                 (c.put_object_legal_hold, {"Bucket": B, "Key": V2_KEY, "LegalHold": {"Status": "OFF"}}),
                 (c.copy_object, {"Bucket": B, "Key": V2_KEY, "CopySource": {"Bucket": B, "Key": P + "x"}}),
                 (c.create_multipart_upload, {"Bucket": B, "Key": V2_KEY}),
                 (c.restore_object, {"Bucket": B, "Key": V2_KEY}),
                 (c.put_bucket_lifecycle_configuration, {"Bucket": B, "LifecycleConfiguration": {"Rules": []}}),
                 (c.delete_bucket_lifecycle, {"Bucket": B}),
                 (c.put_bucket_versioning, {"Bucket": B, "VersioningConfiguration": {"Status": "Suspended"}}),
                 (c.put_bucket_policy, {"Bucket": B, "Policy": "{}"}),
                 (c.get_bucket_lifecycle_configuration, {"Bucket": B}))
        for fn, kw in calls:
            with self.assertRaisesRegex(s3io.S3Refused, "not allowed", msg=fn.__name__):
                fn(**kw)
        with self.assertRaisesRegex(s3io.S3Refused, "presigning"):
            c.generate_presigned_url("put_object", Params={"Bucket": B, "Key": V2_KEY})
        self.assertEqual(c.meta.config.retries.get("total_max_attempts"), 1)
        self.assertEqual(self.usage()["reservations"], 0)

    def test_v2_reads(self):
        r = self.v2_client(writes=False)
        with Stubber(r) as st:
            st.add_response("get_object", {"Body": _body(b"x"), "VersionId": "v1"}, {"Bucket": B, "Key": V2_KEY})
            st.add_response("head_object", {"VersionId": "v1"}, {"Bucket": B, "Key": V2_KEY})
            r.get_object(Bucket=B, Key=V2_KEY)
            r.head_object(Bucket=B, Key=V2_KEY)
        for key in ("experiments/gp-teacher-v6/x.json", V + "a/../../v1/x"):
            with self.assertRaisesRegex(s3io.S3Refused, "without an exact VersionId"):
                r.get_object(Bucket=B, Key=key)
        with self.assertRaisesRegex(s3io.S3Refused, "read-only"):
            r.put_object(**put_kw(V2_KEY, b"1", IfNoneMatch="*"))
        with self.assertRaisesRegex(s3io.S3Refused, "without an exact VersionId"):  # the default reader is unchanged
            _client().get_object(Bucket=B, Key=V2_KEY)


class DefaultWriterRefusesV2(_TmpLedger):
    def test_default_writer_refuses_v2_keys(self):
        g = s3io.Guard(writes=True)
        for key in (V2_KEY, V2_RESEARCH_KEY, V + "receipts/a0/x.json", V + "runs/r/resume/HEAD.json"):
            for kw in ({"IfNoneMatch": "*"}, {"IfMatch": _etag(1)}):
                with self.assertRaisesRegex(s3io.S3Refused, "^PutObject outside PREFIX", msg=key):
                    self.put(g, **put_kw(key, b"1", **kw))
        c = _client(writes=True)
        with self.assertRaisesRegex(s3io.S3Refused, "^PutObject outside PREFIX"):
            c.put_object(**put_kw(V2_KEY, b"1", IfNoneMatch="*"))
        with self.assertRaisesRegex(s3io.S3Refused, "^PutObject outside PREFIX"):
            s3io.put_create_only(c, V2_KEY, b"1", _client())
        self.assertFalse(os.path.exists(self.ledger_path))

    def test_flag_combinations(self):
        with self.assertRaisesRegex(ValueError, "ledger"):
            s3io.Guard(writes=True, v2=True)
        with self.assertRaisesRegex(ValueError, "ledger_path"):
            s3io.make_client(writes=True, v2=True, session=_session())
        with self.assertRaisesRegex(ValueError, "only a v2 writer"):
            s3io.Guard(writes=True, ledger=self.ledger_path)
        with self.assertRaisesRegex(ValueError, "only a v2 writer"):
            s3io.Guard(v2=True, ledger=self.ledger_path)
        for mode in ("cas-read", "lifecycle-once"):
            with self.assertRaisesRegex(ValueError, "lane-client flag"):
                s3io.Guard(mode=mode, v2=True)
        self.assertFalse(s3io.Guard(writes=True).v2)
        self.assertIsNone(_client(writes=True)._dltrain_guard.ledger)


class V2Budget(_TmpLedger):
    def seed(self, used):
        s3io.V2Ledger(self.ledger_path).reserve(V + "seed/big.bin", _sha(b"seed"), used)

    def test_put_that_crosses_the_cap_is_refused(self):
        self.seed(CAP - 10)
        g = self.v2_guard()
        with self.assertRaisesRegex(s3io.BudgetExceeded, "cap"):
            self.put(g, **put_kw(V + "a/k11", b"x" * 11, IfNoneMatch="*"))
        self.assertEqual(self.usage()["reservations"], 1, "a refused PUT is not charged")
        self.put(g, **put_kw(V + "a/k10", b"y" * 10, IfNoneMatch="*"))      # lands exactly on the cap: allowed
        self.assertEqual(self.usage()["used_bytes"], CAP)
        with self.assertRaises(s3io.BudgetExceeded):
            self.put(g, **put_kw(V + "a/k1", b"z", IfNoneMatch="*"))
        with self.assertRaises(s3io.BudgetExceeded):                         # different bytes on the same key count again
            self.put(g, **put_kw(V + "a/k10", b"w" * 10, IfNoneMatch="*"))
        self.put(g, **put_kw(V + "a/k10", b"y" * 10, IfNoneMatch="*"))      # a retry of the same bytes is not counted twice
        self.assertEqual(self.usage()["used_bytes"], CAP)

    def test_one_byte_over_is_refused_and_budget_is_decimal_40gb(self):
        self.assertEqual(CAP, 40 * 10 ** 9)
        self.seed(CAP - 5)
        with self.assertRaises(s3io.BudgetExceeded):
            self.put(self.v2_guard(), **put_kw(V + "a/k6", b"x" * 6, IfNoneMatch="*"))
        self.put(self.v2_guard(), **put_kw(V + "a/k5", b"x" * 5, IfNoneMatch="*"))
        with self.assertRaises(s3io.BudgetExceeded):
            s3io.V2Ledger(self.ledger_path).reserve(V + "a/k-1", _sha(b"q"), 1)

    def test_budget_survives_restart_and_is_shared(self):
        a, b = s3io.V2Ledger(self.ledger_path), s3io.V2Ledger(self.ledger_path)
        a.reserve(V + "a/1", _sha(b"1"), CAP - 3)
        with self.assertRaises(s3io.BudgetExceeded):     # b has not re-read yet; it must sync before deciding
            b.reserve(V + "a/2", _sha(b"2"), 4)
        b.reserve(V + "a/2", _sha(b"2"), 3)
        with self.assertRaises(s3io.BudgetExceeded):
            a.reserve(V + "a/3", _sha(b"3"), 1)
        del a, b
        # "restart": a fresh guard, client and ledger read the persisted file
        u = self.usage()
        self.assertEqual((u["used_bytes"], u["remaining_bytes"], u["reservations"]), (CAP, 0, 2))
        with self.assertRaises(s3io.BudgetExceeded):
            self.put(self.v2_guard(), **put_kw(V2_KEY, b"1", IfNoneMatch="*"))
        c = self.v2_client()
        with Stubber(c):
            with self.assertRaises(s3io.BudgetExceeded):
                c.put_object(**put_kw(V2_KEY, b"1", IfNoneMatch="*"))
        lines = Path(self.ledger_path).read_bytes().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual([json.loads(x)["seq"] for x in lines], [0, 1])
        self.assertEqual(json.loads(lines[1])["prev"], hashlib.sha256(lines[0]).hexdigest())

    def test_ledger_is_append_only_and_fails_closed(self):
        led = s3io.V2Ledger(self.ledger_path)
        led.reserve(V + "a/1", _sha(b"1"), 100)
        led.reserve(V + "a/2", _sha(b"2"), 200)
        good = Path(self.ledger_path).read_bytes()
        mutate = {"edited_bytes": good.replace(b'"bytes":100', b'"bytes":1'),
                  "partial_line": good + b'{"schema"',
                  "dropped_line": good.split(b"\n", 1)[1],
                  "bad_key": good.replace(V.encode() + b"a/1", P.encode() + b"a/1"),
                  "negative": good.replace(b'"bytes":200', b'"bytes":-200'),
                  "unknown_event": good.replace(b'"event":"reserve"', b'"event":"release"')}
        ts = str(json.loads(good.split(b"\n", 1)[0])["utc_us"]).encode()
        mutate["edited_timestamp_chain_only"] = good.replace(ts, str(int(ts) + 1).encode(), 1)   # only the hash chain sees it
        for name, data in mutate.items():
            with open(self.ledger_path, "wb") as f:
                f.write(data)
            with self.assertRaises(s3io.LedgerCorrupt, msg=name):
                s3io.V2Ledger(self.ledger_path)
        with open(self.ledger_path, "wb") as f:
            f.write(good)
        live = s3io.V2Ledger(self.ledger_path)
        with open(self.ledger_path, "wb") as f:   # shrinking under a live ledger
            f.write(good.split(b"\n", 1)[0] + b"\n")
        with self.assertRaisesRegex(s3io.LedgerCorrupt, "shrank"):
            live.reserve(V + "a/3", _sha(b"3"), 1)
        with self.assertRaisesRegex(s3io.S3Refused, "^v2 ledger refuses a key outside V2_PREFIX"):   # its own layer
            s3io.V2Ledger(os.path.join(self.tmp, "other.jsonl")).reserve(P + "x", _sha(b"x"), 1)
        self.assertEqual(s3io.safe_err(s3io.LedgerCorrupt("boom"))["message"], "boom")


class ProductionLoaderRefusesResearchOnly(unittest.TestCase):
    def test_research_only_keys_refused(self):
        for key in (V2_RESEARCH_KEY, V + "research-only/receipts/tabpfn35/0123456789abcdef/f0/x.json", V + "research-only/x",
                    V + "/research-only/encodings/tabpfn35/x.npy", V + "Research-Only/encodings/x.npy",
                    V + "encodings/tabpfn35/0123456789abcdef/f0/x.npy", V + "encodings/TabPFN35/x.npy", None, b"bytes-key"):
            with self.assertRaises(s3io.ResearchOnlyRefused, msg=repr(key)):
                s3io.assert_production_loadable(key)
            self.assertTrue(issubclass(s3io.ResearchOnlyRefused, s3io.S3Refused))
        for key in (V2_KEY, V + "inputs/jtt1-encoder/0123456789abcdef/f0/x.jtt1.gz", P + "objects/receipt/sha256/x.json"):
            self.assertEqual(s3io.assert_production_loadable(key), key)

    def test_production_get_exact(self):
        c = _client()
        with Stubber(c) as st:   # no response queued: a refused key must never reach the stubber
            with self.assertRaises(s3io.ResearchOnlyRefused):
                s3io.production_get_exact(c, V2_RESEARCH_KEY, "v1")
            st.assert_no_pending_responses()
        with Stubber(c) as st:
            st.add_response("get_object", {"Body": _body(b"ok"), "VersionId": "v1"}, {"Bucket": B, "Key": V2_KEY, "VersionId": "v1"})
            body, meta = s3io.production_get_exact(c, V2_KEY, "v1")
        self.assertEqual((body, meta["VersionId"]), (b"ok", "v1"))


if __name__ == "__main__":
    unittest.main()

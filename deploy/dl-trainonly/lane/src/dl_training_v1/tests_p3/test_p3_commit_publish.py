"""commit (review gate, P3 gate, version history, create-only commitment, crash-rerun, CONFLICT) and publish (verify and
closure gates, selection recheck, manifests, closure upload, strict binding, CONFLICT => CONTAMINATED). S3 is the
in-memory FakeS3 behind the real guard (holding the P2-published registry at the P2 receipt's VersionId, which commit and
publish now GET first: review fix P3I-2, see test_p3_interface), or the real guarded client under Stubber."""
import json

from botocore.exceptions import ConnectTimeoutError, ReadTimeoutError
from botocore.stub import Stubber

from dltrain import common, holdout, s3io
from tests_p3 import fixtures_p3 as fx

COMMITMENT_KEY = s3io.PREFIX_KEY + holdout.COMMITMENT_REL
BINDING_KEY = s3io.PREFIX_KEY + holdout.BINDING_REL


def commit_direct(inputs, out, clients):
    return holdout.commit(inputs.registry, inputs.registry_sha256, inputs.decisions, out, clients,
                          p2_receipt=inputs.p2_receipt, owner_decisions=inputs.owner_decisions,
                          owner_decisions_sha256=inputs.owner_decisions_sha256)


class Commit(fx.P3Case):
    def setUp(self):
        super().setUp()
        self.inputs = fx.write_inputs(self.tmp / "in")
        self.bucket = fx.registry_bucket(self.inputs)

    def commit(self, out, inputs=None, bucket=None):
        return fx.run_cli(fx.commit_args(inputs or self.inputs, out), clients=fx.fake_clients(bucket or self.bucket))

    def test_commit_writes_the_commitment_then_the_seed(self):
        out = self.tmp / "holdout"
        code, s = self.commit(out)
        self.assertEqual(code, 0, s)
        body = (out / "commitment.json").read_bytes()
        ref = json.loads((out / "commitment-ref.json").read_bytes())
        doc = json.loads(body)
        pool = holdout.pool_members(self.inputs.families)
        self.assertEqual(doc, {"schema": holdout.COMMITMENT_SCHEMA, "experiment_id": "gp-policy-dl-v1-e0",
                               "pool_sha256": fx.sha(common.canonical(pool)), "registry_sha256": self.inputs.registry_sha256,
                               "review_decisions_sha256": fx.sha(self.inputs.decisions.read_bytes()),
                               "algorithm": holdout.ALGORITHM, "frac": 0.05})
        self.assertEqual(common.canonical(doc), body)
        vid, stored = self.bucket.current(COMMITMENT_KEY)
        self.assertEqual((stored, ref["version_id"], ref["key"], ref["sha256"]), (body, vid, COMMITMENT_KEY, fx.sha(body)))
        seed = holdout.derive_seed(body, vid)
        self.assertEqual((out / "seed.hex").read_text(), seed.hex() + "\n")
        self.assertEqual(holdout.read_seed(out / "seed.hex"), seed)
        r = fx.receipt(s)
        self.assertEqual((r["status"], r["s3_outcome"], r["pool_N"], r["k_expected"]), ("PASS", "created", 60, 3))
        self.assertEqual(r["seed_sha256"], fx.sha(seed))
        self.assertEqual(self.bucket.puts(), [COMMITMENT_KEY])
        hist = r["commitment_history"]
        self.assertEqual((hist["before"]["versions"], hist["after"]["versions"]), ([], [{"version_id": vid, "is_latest": True}]))
        gate = json.loads((out / "gate.json").read_bytes())
        self.assertEqual(gate, r["p3_gate"])
        self.assertEqual(gate["registry_s3"], {"bucket": s3io.BUCKET, "key": fx.registry_s3_key(self.inputs.registry_sha256),
                                               "version_id": "vRegistry-fixture", "sha256": self.inputs.registry_sha256,
                                               "bytes": self.inputs.registry.stat().st_size})
        self.assertEqual(gate["owner_decisions"]["sha256"], self.inputs.owner_decisions_sha256)
        self.assertEqual(gate["decisions"], {"DL-D2": "a", "DL-D4": "a", "DL-D6": "commitment-derived",
                                             "DL-D12": "keep_sealed"})
        self.assertEqual(r["registry"]["s3"], gate["registry_s3"])

    def test_crash_rerun_reproduces_the_same_seed(self):
        first, second, third = self.tmp / "a", self.tmp / "b", self.tmp / "c"
        code, _ = self.commit(first)
        self.assertEqual(code, 0)
        # crash after the PUT but before any local write: a fresh directory, the same S3 state -> 412, equal bytes
        code, s = self.commit(second)
        self.assertEqual(code, 0, s)
        self.assertEqual(fx.receipt(s)["s3_outcome"], "already_present_equal_bytes")
        for name in ("seed.hex", "commitment.json", "commitment-ref.json", "gate.json"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)
        # crash after seed.hex only: the rerun completes the same files
        third.mkdir()
        (third / "seed.hex").write_bytes((first / "seed.hex").read_bytes())
        code, s = self.commit(third)
        self.assertEqual(code, 0, s)
        self.assertEqual((third / "commitment-ref.json").read_bytes(), (first / "commitment-ref.json").read_bytes())
        # a plain rerun into the finished directory is a no-op
        code, _ = self.commit(first)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.bucket.versions(COMMITMENT_KEY)), 1)

    def test_conflicting_commitment_stops(self):
        self.bucket.preload(COMMITMENT_KEY, b'{"experiment_id":"gp-policy-dl-v1-e0","someone":"else"}')
        out = self.tmp / "holdout"
        code, s = self.commit(out)
        self.assertEqual(code, 4, s)
        r = fx.receipt(s)
        self.assertEqual((r["status"], r["stop_reason"]), ("CONFLICT", "commitment_conflict"))
        self.assertFalse((out / "seed.hex").exists())
        self.assertFalse((out / "commitment.json").exists())
        self.assertEqual(len(self.bucket.versions(COMMITMENT_KEY)), 1)

    def test_local_file_mismatch_refuses_before_any_s3_call(self):
        for name in ("commitment.json", "gate.json"):
            out = self.tmp / ("holdout-" + name)
            out.mkdir()
            (out / name).write_bytes(b'{"other":1}')
            code, s = self.commit(out)
            self.assertEqual(code, 2, s)
            self.assertEqual(self.bucket.calls, [])

    def gate_stops(self, inputs, regex):
        out = self.tmp / ("gate-" + fx.sha(str(inputs.dir).encode())[:8])
        code, s = self.commit(out, inputs)
        self.assertEqual(code, 2, s)
        r = fx.receipt(s)
        self.assertEqual(r["status"], "STOP")
        self.assertRegex(r["error"]["message"], regex)
        self.assertEqual(self.bucket.calls, [])
        self.assertFalse((out / "seed.hex").exists())

    def test_review_gate(self):
        self.gate_stops(fx.write_inputs(self.tmp / "g1", seed_decision="reject_mapping"), "reject_mapping")
        self.gate_stops(fx.write_inputs(self.tmp / "g2", n_seed_rows=9), "9 seed_source_mapping rows")
        self.gate_stops(fx.write_inputs(self.tmp / "g3", drop_seed_rows=1), "seed row rg-.* decision None")
        self.gate_stops(fx.write_inputs(self.tmp / "g4", seed_decision="unresolved"), "decision 'unresolved'")
        self.gate_stops(fx.write_inputs(self.tmp / "g5", registry_decisions_sha=fx.h("other decisions")),
                        "review_decisions_sha256 is not this decision file")
        self.gate_stops(fx.write_inputs(self.tmp / "g6", csv_bytes_override=b"review_group_id,kind,recommendation\n"),
                        "review_csv_sha256")
        self.gate_stops(fx.write_inputs(self.tmp / "g7", provenance=False), "provenance")
        self.gate_stops(fx.write_inputs(self.tmp / "g8", phase="pre_review"), "final")
        dup = [{"review_group_id": "rg-" + fx.h("publisher-row")[:24], "decision": "union"}]
        self.gate_stops(fx.write_inputs(self.tmp / "g9", extra=dup), "two decisions")

    def test_p3_gate_needs_a_passing_p2_receipt_that_publishes_this_registry(self):
        """SC-01: P2 PASS bound to this registry and its S3 ref, before the irreversible commitment PUT."""
        other = fx.write_inputs(self.tmp / "other", fx.standard_families(n_pool=61))
        cases = (("p1", dict(p2=dict(status="FAIL")), "not PASS"),
                 ("p2", dict(p2=dict(packet="P1")), "packet P2"),
                 ("p3", dict(p2=dict(schema="jope.dl-p1-receipt/v1")), "schema"),
                 ("p4", dict(p2=dict(with_ref=False)), "no S3 ref for this registry"),
                 ("p5", dict(p2=dict(ref_overrides={"version_id": "null"})), "incomplete or disagree"),
                 ("p6", dict(p2=dict(ref_overrides={"bytes": 1})), "incomplete or disagree"),
                 ("p7", dict(p2=dict(ref_overrides={"bucket": "another-bucket"})), "incomplete or disagree"),
                 ("p8", dict(p2_bytes_override=other.p2_receipt.read_bytes()), "no S3 ref for this registry"),
                 ("p9", dict(p2_bytes_override=b"{not json"), "not strict JSON"))
        for d, kw, regex in cases:
            self.gate_stops(fx.write_inputs(self.tmp / d, **kw), regex)
        inputs = fx.write_inputs(self.tmp / "p10")
        inputs.p2_receipt.unlink()
        self.gate_stops(inputs, "missing file")
        both = fx.write_inputs(self.tmp / "p11")                              # status and verdict must both be PASS
        doc = json.loads(both.p2_receipt.read_bytes()) | {"verdict": "FAIL"}
        both.p2_receipt.write_bytes(common.canonical(doc))
        self.gate_stops(both, "not PASS")

    def test_p3_gate_needs_the_owner_decisions_the_code_assumes(self):
        """SC-01: DL-D2 = a, DL-D4 = a, DL-D6 = commitment-derived, DL-D12 = keep sealed, pinned, with provenance."""
        cases = (("o1", dict(drop=("DL-D6",)), "do not record DL-D6"),
                 ("o2", dict(drop=("DL-D2", "DL-D12")), "do not record DL-D2"),
                 ("o3", dict(decisions={"DL-D4": "b: reuse frozen 1b779c6f normalizers"}), "DL-D4 is not option a"),
                 ("o4", dict(decisions={"DL-D2": "(b) block until qualified"}), "DL-D2 is not option a"),
                 ("o5", dict(decisions={"DL-D6": "fixed number 20261004"}), "DL-D6 is not the commitment-derived"),
                 ("o6", dict(decisions={"DL-D12": "waive the lineage-B seal"}), "DL-D12 is not 'keep sealed'"),
                 ("o7", dict(provenance=False), "provenance"),
                 ("o8", dict(schema="jope.dl-owner-decisions/v0"), "owner decisions are not"))
        for d, owner, regex in cases:
            self.gate_stops(fx.write_inputs(self.tmp / d, owner=owner), regex)
        inputs = fx.write_inputs(self.tmp / "o9")
        inputs.owner_decisions_sha256 = fx.h("another pin")
        self.gate_stops(inputs, "sha256 .* != pinned")
        inputs.owner_decisions_sha256 = "not-a-pin"
        self.gate_stops(inputs, "64-hex sha256 pin")

    def test_default_p2_receipt_is_under_run_receipts(self):
        argv = fx.commit_args(self.inputs, self.inputs.dir / "holdout")
        i = argv.index("--p2-receipt")
        del argv[i:i + 2]
        code, s = fx.run_cli(argv, clients=fx.fake_clients(self.bucket))     # <out>/../receipts/P2/receipt.json
        self.assertEqual(code, 0, s)
        self.assertEqual(fx.receipt(s)["p3_gate"]["p2_receipt"]["path"], str(self.inputs.p2_receipt.resolve()))

    def test_review_gate_accepts_class_decisions_and_add_members(self):
        for d, kw in (("c1", {"use_class": True}), ("c2", {"seed_decision": "add_members"}),
                      ("c3", {"use_class": True, "when": "recommendation == confirm_mapping"})):
            inputs = fx.write_inputs(self.tmp / d, **kw)
            code, s = self.commit(self.tmp / (d + "-out"), inputs, fx.registry_bucket(inputs))
            self.assertEqual(code, 0, s)
            self.assertEqual(fx.receipt(s)["review_gate"]["seed_rows"], 10)
        self.gate_stops(fx.write_inputs(self.tmp / "c4", use_class=True, when="recommendation==independent"),
                        "decision None")

    def test_missing_review_csv_stops(self):
        self.inputs.review_csv.rename(self.tmp / "moved.csv")
        code, s = self.commit(self.tmp / "holdout")
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "review.csv not found")
        code, s = fx.run_cli(fx.commit_args(self.inputs, self.tmp / "holdout2") + ["--review-csv", self.tmp / "moved.csv"],
                             clients=fx.fake_clients(self.bucket))
        self.assertEqual(code, 0, s)

    def test_without_injected_clients_the_real_factory_is_the_only_route(self):
        code, s = fx.run_cli(fx.commit_args(self.inputs, self.tmp / "holdout"))
        self.assertEqual(code, 3, s)
        self.assertEqual(fx.receipt(s)["error"]["type"], "AssertionError")


class CommitVersionHistory(fx.P3Case):
    """LP-1: the seed derives from the commitment key's only version; a re-upload, a delete marker or a second version
    makes commit STOP before any seed exists."""

    def setUp(self):
        super().setUp()
        self.inputs = fx.write_inputs(self.tmp / "in")
        self.bucket = fx.registry_bucket(self.inputs)
        code, s = fx.run_cli(fx.commit_args(self.inputs, self.tmp / "a"), clients=fx.fake_clients(self.bucket))
        self.assertEqual(code, 0, s)
        self.body = (self.tmp / "a" / "commitment.json").read_bytes()

    def recommit(self, out):
        puts = len(self.bucket.puts())
        code, s = fx.run_cli(fx.commit_args(self.inputs, out), clients=fx.fake_clients(self.bucket))
        return code, s, len(self.bucket.puts()) - puts

    def test_equal_bytes_reupload_makes_a_rerun_conflict(self):
        self.bucket.preload(COMMITMENT_KEY, self.body)                       # e.g. an out-of-lane sync of bindings/
        code, s, new_puts = self.recommit(self.tmp / "b")
        self.assertEqual(code, 4, s)
        r = fx.receipt(s)
        self.assertEqual((r["status"], r["stop_reason"]), ("CONFLICT", "commitment_history"))
        self.assertEqual(len(r["commitment_history"]["versions"]), 2)
        self.assertEqual(new_puts, 0)
        self.assertFalse((self.tmp / "b" / "seed.hex").exists())

    def test_delete_marker_blocks_a_reroll(self):
        self.bucket.delete_marker(COMMITMENT_KEY)
        code, s, new_puts = self.recommit(self.tmp / "b")
        self.assertEqual(code, 4, s)
        self.assertEqual(fx.receipt(s)["stop_reason"], "commitment_history")
        self.assertEqual(new_puts, 0)                                         # no new version, so no re-rolled seed
        self.assertEqual(len(self.bucket.versions(COMMITMENT_KEY)), 2)
        self.assertFalse((self.tmp / "b" / "seed.hex").exists())

    def test_a_second_version_right_after_our_create_is_refused(self):
        bucket = fx.registry_bucket(self.inputs)
        bucket.after_put[COMMITMENT_KEY] = lambda b: b.preload(COMMITMENT_KEY, self.body)
        out = self.tmp / "c"
        code, s = fx.run_cli(fx.commit_args(self.inputs, out), clients=fx.fake_clients(bucket))
        self.assertEqual(code, 4, s)
        r = fx.receipt(s)
        self.assertEqual(r["stop_reason"], "commitment_history")
        self.assertEqual(r["adopted_version_id"], bucket.versions(COMMITMENT_KEY)[0][0])
        self.assertFalse((out / "seed.hex").exists())

    def test_a_guard_refusing_history_makes_commit_stop_before_any_put(self):
        bucket = fx.registry_bucket(self.inputs, history_supported=False)  # simulated refusing guard (fail closed)
        out = self.tmp / "d"
        code, s = fx.run_cli(fx.commit_args(self.inputs, out), clients=fx.fake_clients(bucket))
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "refuses ListObjectVersions")
        self.assertEqual(bucket.puts(), [])
        self.assertEqual(sorted(x.name for x in out.iterdir()), ["receipts"])      # no seed, commitment or gate file

    def test_real_lane_guard_admits_exact_key_history_only(self):
        r = fx.guarded_client()                                            # the real guarded read client
        with Stubber(r) as sr:
            sr.add_response("list_object_versions", {"Versions": [], "DeleteMarkers": [], "IsTruncated": False},
                            {"Bucket": s3io.BUCKET, "Prefix": COMMITMENT_KEY, "MaxKeys": holdout.HISTORY_MAX})
            hist = holdout.key_history(r, COMMITMENT_KEY)
            sr.assert_no_pending_responses()
        self.assertEqual(hist, {"key": COMMITMENT_KEY, "versions": [], "delete_markers": []})
        for bad in ({"Bucket": s3io.BUCKET, "Prefix": s3io.PREFIX_KEY, "MaxKeys": 16},
                    {"Bucket": s3io.BUCKET, "Prefix": COMMITMENT_KEY, "MaxKeys": 101},
                    {"Bucket": s3io.BUCKET, "Prefix": COMMITMENT_KEY, "MaxKeys": 16, "KeyMarker": "x"}):
            with self.assertRaises(s3io.S3Refused):
                r.list_object_versions(**bad)


class CreateOnlyProtocol(fx.P3Case):
    """SC-08, PLAN §6.6 publish step 1: a 412, 409, timeout, reset or 5xx re-reads the key; equal bytes are success,
    different bytes are returned for CONFLICT/POISONED_KEY, and an absent key gets exactly one more create-only PUT."""
    KEY = s3io.PREFIX_KEY + "objects/holdout/sha256/" + "a" * 64 + ".json"
    BODY = b'{"fixture":true}'

    def create(self, bucket):
        return holdout._create_only(fx.FakeS3Client(bucket, writes=True), fx.FakeS3Client(bucket), self.KEY, self.BODY)

    def test_ambiguous_failure_before_the_store_puts_once_more(self):
        for err in (fx.client_error(500, "InternalError", "PutObject"), fx.client_error(409, "ConditionalRequestConflict",
                                                                                         "PutObject"),
                    fx.client_error(412, "PreconditionFailed", "PutObject"), ConnectTimeoutError(endpoint_url="https://x")):
            bucket = fx.FakeBucket()
            bucket.fail_put(self.KEY, err, store=False)
            res = self.create(bucket)
            self.assertEqual((res["outcome"], res["equal"], len(res["attempts"])), ("created", True, 1), err)
            self.assertEqual(bucket.puts(), [self.KEY, self.KEY])
            self.assertEqual([b for _, b in bucket.versions(self.KEY)], [self.BODY])

    def test_lost_response_after_the_store_is_success_with_that_version(self):
        for err in (fx.client_error(503, "SlowDown", "PutObject"), ReadTimeoutError(endpoint_url="https://x")):
            bucket = fx.FakeBucket()
            bucket.fail_put(self.KEY, err, store=True)
            res = self.create(bucket)
            self.assertEqual((res["outcome"], res["equal"]), ("exists", True), err)
            self.assertEqual(res["version_id"], bucket.versions(self.KEY)[0][0])
            self.assertEqual(bucket.puts(), [self.KEY])

    def test_two_ambiguous_attempts_with_the_key_absent_stop(self):
        bucket = fx.FakeBucket()
        for _ in range(2):
            bucket.fail_put(self.KEY, fx.client_error(500, "InternalError", "PutObject"), store=False)
        with self.assertRaisesRegex(holdout.HoldoutStop, "unresolved"):
            self.create(bucket)
        self.assertEqual(bucket.puts(), [self.KEY, self.KEY])
        self.assertEqual(bucket.versions(self.KEY), [])

    def test_other_errors_propagate_without_a_retry(self):
        bucket = fx.FakeBucket()
        bucket.fail_put(self.KEY, fx.client_error(403, "AccessDenied", "PutObject"), store=False)
        with self.assertRaises(Exception) as ctx:
            self.create(bucket)
        self.assertNotIsInstance(ctx.exception, holdout.HoldoutStop)
        self.assertEqual(bucket.puts(), [self.KEY])

    def test_different_bytes_are_returned_unequal(self):
        bucket = fx.FakeBucket()
        bucket.preload(self.KEY, b"other")
        res = self.create(bucket)
        self.assertEqual((res["outcome"], res["equal"], res["existing"]), ("exists", False, b"other"))


class CreateOnlyThroughTheGuardedClient(fx.P3Case):
    """The same create-only paths through s3io.make_client (the real lane guard) with botocore Stubber; nothing is sent."""
    KEY = COMMITMENT_KEY
    BODY = b'{"fixture":"commitment"}'

    def put_params(self):
        return {"Bucket": s3io.BUCKET, "Key": self.KEY, "Body": self.BODY, "IfNoneMatch": "*",
                "ContentType": "application/json", "ChecksumAlgorithm": "SHA256",
                "ChecksumSHA256": holdout._checksum_b64(self.BODY)}

    def run_create(self, setup_writer, setup_reader):
        w, r = fx.guarded_client(writes=True), fx.guarded_client()
        with Stubber(w) as sw, Stubber(r) as sr:
            setup_writer(sw)
            setup_reader(sr)
            res = holdout._create_only(w, r, self.KEY, self.BODY)
            sw.assert_no_pending_responses()
            sr.assert_no_pending_responses()
        return res

    def readback(self, sr, vid):
        sr.add_response("get_object", {"Body": fx.streaming(self.BODY), "VersionId": vid},
                        {"Bucket": s3io.BUCKET, "Key": self.KEY, "VersionId": vid})

    def test_created_then_412_equal_bytes(self):
        res = self.run_create(
            lambda sw: sw.add_response("put_object", {"VersionId": "vFirst", "ETag": '"' + "a" * 32 + '"'}, self.put_params()),
            lambda sr: self.readback(sr, "vFirst"))
        self.assertEqual((res["outcome"], res["version_id"]), ("created", "vFirst"))
        res = self.run_create(
            lambda sw: sw.add_client_error("put_object", service_error_code="PreconditionFailed", http_status_code=412,
                                           expected_params=self.put_params()),
            lambda sr: sr.add_response("get_object", {"Body": fx.streaming(self.BODY), "VersionId": "vFirst"},
                                       {"Bucket": s3io.BUCKET, "Key": self.KEY}))
        self.assertEqual((res["outcome"], res["equal"], res["version_id"]), ("exists", True, "vFirst"))

    def test_412_with_different_bytes_is_unequal(self):
        res = self.run_create(
            lambda sw: sw.add_client_error("put_object", service_error_code="PreconditionFailed", http_status_code=412,
                                           expected_params=self.put_params()),
            lambda sr: sr.add_response("get_object", {"Body": fx.streaming(b'{"other":true}'), "VersionId": "vOther"},
                                       {"Bucket": s3io.BUCKET, "Key": self.KEY}))
        self.assertEqual((res["equal"], res["version_id"]), (False, "vOther"))

    def test_500_then_absent_then_one_more_put(self):
        def writer(sw):
            sw.add_client_error("put_object", service_error_code="InternalError", http_status_code=500,
                                expected_params=self.put_params())
            sw.add_response("put_object", {"VersionId": "vSecond", "ETag": '"' + "b" * 32 + '"'}, self.put_params())

        def reader(sr):
            sr.add_client_error("get_object", service_error_code="NoSuchKey", http_status_code=404,
                                expected_params={"Bucket": s3io.BUCKET, "Key": self.KEY})
            self.readback(sr, "vSecond")
        res = self.run_create(writer, reader)
        self.assertEqual((res["outcome"], res["version_id"], res["attempts"][0]["status"]), ("created", "vSecond", 500))

    def test_non_ambiguous_errors_propagate(self):
        with self.assertRaises(Exception) as ctx:
            self.run_create(
                lambda sw: sw.add_client_error("put_object", service_error_code="AccessDenied", http_status_code=403,
                                               expected_params=self.put_params()),
                lambda sr: None)
        self.assertNotIsInstance(ctx.exception, holdout.HoldoutStop)


class Publish(fx.P3Case):
    def setUp(self):
        super().setUp()
        self.inputs = fx.write_inputs(self.tmp / "in")

    def publish(self, out, bucket, *extra):
        return fx.run_cli(fx.publish_args(out, *extra), clients=fx.fake_clients(bucket))

    def keys(self, out):
        body = (out / "holdout.json").read_bytes()
        cl = (out / "closure.json").read_bytes()
        return {"manifest": s3io.PREFIX_KEY + f"objects/holdout/sha256/{fx.sha(body)}.json",
                "campaign": s3io.PREFIX_KEY + f"campaigns/gp-policy-dl-v1-e0/manifests/{fx.sha(body)}.json",
                "closure": s3io.PREFIX_KEY + f"objects/exclusion-closure/sha256/{fx.sha(cl)}.json"}

    def test_publish_binds_once_and_a_rerun_is_a_noop(self):
        out, bucket, _ = fx.publish_ready(self, self.inputs)
        code, s = self.publish(out, bucket)
        self.assertEqual(code, 0, s)
        body = (out / "holdout.json").read_bytes()
        cl_body = (out / "closure.json").read_bytes()
        doc = json.loads(body)
        k = self.keys(out)
        m_vid, stored = bucket.current(k["manifest"])
        self.assertEqual(stored, body)
        self.assertEqual(bucket.current(k["campaign"])[1], body)                  # SC-07: §6.6 campaign manifest
        c_vid, c_stored = bucket.current(k["closure"])
        self.assertEqual(c_stored, cl_body)                                       # SC-07: the closure is published
        b_vid, b_body = bucket.current(BINDING_KEY)
        binding = json.loads(b_body)
        holdout_ref = {"bucket": s3io.BUCKET, "key": k["manifest"], "version_id": m_vid, "sha256": fx.sha(body),
                       "bytes": len(body)}
        closure_ref = {"bucket": s3io.BUCKET, "key": k["closure"], "version_id": c_vid, "sha256": fx.sha(cl_body),
                       "bytes": len(cl_body)}
        self.assertEqual(binding, holdout.binding_doc(doc, holdout_ref, closure_ref))
        self.assertEqual(binding["commitment_ref"]["version_id"], json.loads((out / "commitment-ref.json").read_bytes())["version_id"])
        self.assertEqual(binding["registry_ref"], doc["registry"]["s3"])
        self.assertEqual(common.canonical(binding), b_body)
        self.assertEqual(bucket.puts(), [COMMITMENT_KEY, k["manifest"], k["campaign"], k["closure"], BINDING_KEY])
        r = fx.receipt(s)
        self.assertEqual((r["status"], r["binding"]["outcome"], r["manifest"]["outcome"], r["closure_object"]["outcome"]),
                         ("PASS", "created", "created", "created"))
        self.assertEqual(r["recheck"]["seed_sha256"], doc["seed_sha256"])
        self.assertEqual(r["histories"]["binding_after"]["versions"], [{"version_id": b_vid, "is_latest": True}])
        code, s = self.publish(out, bucket)
        self.assertEqual(code, 0, s)
        self.assertEqual(fx.receipt(s)["binding"]["outcome"], "already_present_equal_bytes")
        for key in list(k.values()) + [BINDING_KEY]:
            self.assertEqual(len(bucket.versions(key)), 1, key)

    def test_publish_needs_a_passing_assignment_poison_receipt(self):
        out, bucket = fx.pipeline(self, self.inputs, verify_too=False)
        code, s = self.publish(out, bucket)
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "verify --assignment-poison")
        code, _ = fx.run_cli(fx.verify_args(self.inputs, out, poison=False))
        self.assertEqual(code, 0)
        code, s = self.publish(out, bucket)              # a verify without the poison test is not enough
        self.assertEqual(code, 2, s)
        self.assertEqual(bucket.puts(), [COMMITMENT_KEY])

    def test_publish_needs_a_pass_closure_for_these_bytes(self):
        out, bucket = fx.pipeline(self, self.inputs)
        code, s = self.publish(out, bucket)
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "run closure first")
        self.assertEqual(bucket.puts(), [COMMITMENT_KEY])
        other = self.tmp / "other-closure.json"
        other.write_bytes(common.canonical({"schema": holdout.CLOSURE_SCHEMA, "holdout_sha256": fx.h("other")}))
        code, s = self.publish(out, bucket, "--closure", other)
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "closure gate")
        self.assertEqual(bucket.puts(), [COMMITMENT_KEY])

    def test_second_binding_with_a_different_ref_is_conflict_and_contaminated(self):
        out, bucket, _ = fx.publish_ready(self, self.inputs)
        other_ref = {"bucket": s3io.BUCKET, "key": s3io.PREFIX_KEY + "objects/holdout/sha256/" + fx.h("other") + ".json",
                     "version_id": "vOtherHoldout", "sha256": fx.h("other"), "bytes": 123}
        bucket.preload(BINDING_KEY, common.canonical({"schema": "jope.dl-binding/v1", "kind": "holdout",
                                                      "holdout_ref": other_ref}))
        code, s = self.publish(out, bucket)
        self.assertEqual(code, 4, s)
        self.assertEqual((s["status"], s["experiment_state"]), ("CONFLICT", "CONTAMINATED"))
        r = fx.receipt(s)
        self.assertEqual((r["status"], r["experiment_state"], r["stop_reason"]), ("CONFLICT", "CONTAMINATED", "binding_conflict"))
        self.assertEqual(r["existing_binding"]["holdout_ref"], other_ref)
        self.assertEqual(r["attempted_holdout_ref"]["sha256"], fx.sha((out / "holdout.json").read_bytes()))
        self.assertEqual(len(bucket.versions(BINDING_KEY)), 1)
        self.assertEqual(bucket.puts(), [COMMITMENT_KEY])                   # refused before any manifest is written

    def preload_objects(self, out, bucket):
        body, cl = (out / "holdout.json").read_bytes(), (out / "closure.json").read_bytes()
        k = self.keys(out)
        vids = {name: bucket.preload(k[name], cl if name == "closure" else body) for name in k}
        holdout_ref = {"bucket": s3io.BUCKET, "key": k["manifest"], "version_id": vids["manifest"], "sha256": fx.sha(body),
                       "bytes": len(body)}
        closure_ref = {"bucket": s3io.BUCKET, "key": k["closure"], "version_id": vids["closure"], "sha256": fx.sha(cl),
                       "bytes": len(cl)}
        return holdout.binding_doc(json.loads(body), holdout_ref, closure_ref)

    def test_existing_binding_that_differs_only_in_rule_text_is_accepted(self):
        out, bucket, _ = fx.publish_ready(self, self.inputs)
        expected = self.preload_objects(out, bucket)
        bucket.preload(BINDING_KEY, common.canonical(expected | {"rule": "first freeze wins (written by an older publisher)"}))
        code, s = self.publish(out, bucket)
        self.assertEqual(code, 0, s)
        r = fx.receipt(s)
        self.assertEqual((r["manifest"]["outcome"], r["binding"]["outcome"]),
                         ("already_present_equal_bytes", "already_bound_same_ref"))

    def test_existing_binding_with_the_same_ref_but_other_fields_is_conflict(self):
        """LP-5: every field but the rule text must equal ours, parsed strictly."""
        for name, mutate, reason in (
                ("note", lambda e: {"holdout_ref": e["holdout_ref"], "note": "written by an older publisher"},
                 "binding_fields_differ"),
                ("tau", lambda e: e | {"tau_hex": "0" * 64}, "binding_fields_differ"),
                ("closure", lambda e: {k: v for k, v in e.items() if k != "closure_ref"}, "binding_fields_differ")):
            inputs = fx.write_inputs(self.tmp / name)
            out, bucket, _ = fx.publish_ready(self, inputs)
            expected = self.preload_objects(out, bucket)
            bucket.preload(BINDING_KEY, common.canonical(mutate(expected)))
            code, s = self.publish(out, bucket)
            self.assertEqual(code, 4, (name, s))
            r = fx.receipt(s)
            self.assertEqual((r["experiment_state"], r["stop_reason"]), ("CONTAMINATED", reason), name)
            self.assertEqual(len(bucket.versions(BINDING_KEY)), 1)
        inputs = fx.write_inputs(self.tmp / "dup")
        out, bucket, _ = fx.publish_ready(self, inputs)
        expected = self.preload_objects(out, bucket)
        raw = common.canonical(expected)
        bucket.preload(BINDING_KEY, raw[:-1] + b',"tau_hex":"' + b"0" * 64 + b'"}')       # duplicate key, last wins
        code, s = self.publish(out, bucket)
        self.assertEqual(code, 4, s)
        self.assertEqual(fx.receipt(s)["stop_reason"], "binding_conflict")

    def test_poisoned_manifest_key_stops_before_the_binding(self):
        out, bucket, _ = fx.publish_ready(self, self.inputs)
        bucket.preload(self.keys(out)["manifest"], b"different bytes")
        code, s = self.publish(out, bucket)
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "POISONED_KEY")
        self.assertIsNone(bucket.current(BINDING_KEY))

    def test_publish_refuses_a_commitment_that_is_not_the_bound_one(self):
        out, bucket, _ = fx.publish_ready(self, self.inputs)
        other = fx.FakeBucket()                         # the bound commitment is absent from this bucket
        code, s = self.publish(out, other)
        self.assertEqual(code, 3, s)
        self.assertEqual(other.puts(), [])

    def tampered(self, out, change):
        """A copy of holdout.json changed by `change`, with a forged PASS verify receipt and a forged PASS closure for
        the new bytes: only publish's own recheck can refuse it."""
        doc = json.loads((out / "holdout.json").read_bytes())
        change(doc)
        t = self.tmp / ("tampered-" + fx.sha(common.canonical(doc))[:8])
        t.mkdir()
        body = common.canonical(doc)
        (t / "holdout.json").write_bytes(body)
        self.assertEqual(holdout.holdout_doc_problems(doc), [])
        (t / "receipts").mkdir()
        (t / "receipts" / "verify-forged.json").write_bytes(common.canonical(
            {"schema": holdout.RECEIPT_SCHEMA, "command": "verify", "status": "PASS", "holdout_sha256": fx.sha(body),
             "assignment_poison": {"ran": True, "pass": True}}))
        cl = json.loads((out / "closure.json").read_bytes()) | {"holdout_sha256": fx.sha(body)}
        (t / "closure.json").write_bytes(common.canonical(cl))
        return t

    def test_publish_recomputes_the_selection_from_the_bound_commitment(self):
        """LP-4: a hand-picked holdout.json with a forged verify receipt does not bind."""
        out, bucket, _ = fx.publish_ready(self, self.inputs)

        def swap(d):
            h0, d0 = d["held"][0], d["dev"][0]
            d["held"] = [d0] + d["held"][1:]
            d["dev"] = [h0] + d["dev"][1:]
            d["assignment"][h0], d["assignment"][d0] = "dev", "heldout"

        def zero_seed(d):
            d["seed_sha256"] = "0" * 64
        for change, regex in ((swap, "held differs"), (zero_seed, "seed_sha256 does not derive")):
            t = self.tampered(out, change)
            code, s = self.publish(t, bucket)
            self.assertEqual(code, 2, s)
            self.assertRegex(fx.receipt(s)["error"]["message"], regex)
            self.assertEqual(bucket.puts(), [COMMITMENT_KEY])
            self.assertIsNone(bucket.current(BINDING_KEY))

    def test_commitment_history_must_be_the_bound_version(self):
        out, bucket, _ = fx.publish_ready(self, self.inputs)
        bucket.preload(COMMITMENT_KEY, (out / "commitment.json").read_bytes())     # an equal-bytes re-upload
        code, s = self.publish(out, bucket)
        self.assertEqual(code, 4, s)
        self.assertEqual(fx.receipt(s)["stop_reason"], "commitment_history")
        self.assertEqual(bucket.puts(), [COMMITMENT_KEY])

    def test_a_deleted_binding_is_never_replaced(self):
        """LP-1: a delete marker on the binding key (an out-of-lane delete) must not let a second binding through."""
        out, bucket, _ = fx.publish_ready(self, self.inputs)
        code, s = self.publish(out, bucket)
        self.assertEqual(code, 0, s)
        bucket.delete_marker(BINDING_KEY)
        code, s = self.publish(out, bucket)
        self.assertEqual(code, 4, s)
        r = fx.receipt(s)
        self.assertEqual((r["experiment_state"], r["stop_reason"]), ("CONTAMINATED", "binding_history"))
        self.assertEqual(len(bucket.versions(BINDING_KEY)), 2)                     # the version and the marker only

    def test_a_second_binding_version_after_ours_is_contaminated(self):
        out, bucket, _ = fx.publish_ready(self, self.inputs)
        bucket.after_put[BINDING_KEY] = lambda b: b.preload(BINDING_KEY, b'{"another":"binding"}')
        code, s = self.publish(out, bucket)
        self.assertEqual(code, 4, s)
        r = fx.receipt(s)
        self.assertEqual((r["experiment_state"], r["stop_reason"]), ("CONTAMINATED", "binding_history"))

    def test_a_guard_refusing_history_makes_publish_write_nothing(self):
        """P3I-5: s3io's guard allows the exact-key history; a guard that refuses it still fails closed."""
        out, bucket, _ = fx.publish_ready(self, self.inputs)
        bucket.history_supported = False
        code, s = self.publish(out, bucket)
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "refuses ListObjectVersions")
        self.assertEqual(bucket.puts(), [COMMITMENT_KEY])

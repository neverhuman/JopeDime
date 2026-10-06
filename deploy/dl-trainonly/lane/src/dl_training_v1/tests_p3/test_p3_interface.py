"""Review fixes P3I-1, P3I-2 and P3I-5, with the holdout parts of D-MAP-5 and D-MAP-6.

- P3I-1: holdout.review_gate applies the P2 provenance rule itself (families.check_provenance with real=True and
  RUN/receipts/P2), adds where the pinned owner record may live, and needs the final registry's own CL-2 marker
  (decision_deltas.provenance) to read draft and simulation false with that same record. DRAFT decisions reach commit
  only through the non-CLI keyword draft_dry_run, with fake clients, never into RUN/holdout.
- P3I-2: commit and publish GET the registry at the exact VersionId of the P2 receipt's ref and need its sha256 and
  bytes before any S3 write.
- P3I-5: the docs describe the exact-key history s3io's lane guard allows.
Every location (RUN/receipts/P2, RUN/holdout, OWNER_DIRECTIVES.md) is moved under TMPDIR by fx.P3Case."""
import json

from botocore.stub import Stubber

from dltrain import common, families, holdout, s3io
from tests_p3 import fixtures_p3 as fx

COMMITMENT_KEY = s3io.PREFIX_KEY + holdout.COMMITMENT_REL
BINDING_KEY = s3io.PREFIX_KEY + holdout.BINDING_REL
DRAFT_SESSION = "DRAFT-UNCONFIRMED relayed answer (fixture)"


class GateCase(fx.P3Case):
    def commit(self, inputs, *, bucket=None, out=None, **kw):
        bucket = fx.registry_bucket(inputs) if bucket is None else bucket
        out = inputs.dir / "holdout" if out is None else out
        code, s = fx.run_cli(fx.commit_args(inputs, out), clients=fx.fake_clients(bucket), **kw)
        return code, s, bucket, out

    def passes(self, inputs, **kw):
        code, s, bucket, out = self.commit(inputs, **kw)
        self.assertEqual(code, 0, s)
        return fx.receipt(s), bucket, out

    def stops(self, inputs, regex, **kw):
        """A review-gate STOP: before any S3 call, no seed and no commitment file."""
        code, s, bucket, out = self.commit(inputs, **kw)
        self.assertEqual(code, 2, s)
        r = fx.receipt(s)
        self.assertEqual(r["status"], "STOP")
        self.assertRegex(r["error"]["message"], regex)
        self.assertEqual(bucket.calls, [])
        self.assertFalse((out / "seed.hex").exists())
        self.assertFalse((out / "commitment.json").exists())
        return r


class ReviewGateProvenance(GateCase):
    """P3I-1: P3 can tell a registry built from owner decisions from one built from a DRAFT."""

    def test_the_owner_record_passes_and_is_recorded(self):
        inputs = fx.write_inputs(self.tmp / "in")
        r, _, _ = self.passes(inputs)
        msg = fx.owner_message(fx.sha(inputs.review_csv.read_bytes()))
        prov = r["review_gate"]["provenance"]
        self.assertEqual((prov["kind"], prov["draft"], prov["draft_dry_run"]), ("owner_verbatim_message", False, False))
        self.assertEqual(prov["record"], {"path": str(self.p2_receipts / f"dl-d5-owner-message-{fx.sha(msg)[:16]}.txt"),
                                          "sha256": fx.sha(msg), "form": "stored_owner_message"})
        self.assertEqual(inputs.owner_message, self.p2_receipts / f"dl-d5-owner-message-{fx.sha(msg)[:16]}.txt")
        self.assertFalse(r["draft_dry_run"])
        self.assertNotIn("notice", r)

    def test_the_real_locations_follow_the_p2_rule(self):
        self.assertEqual(fx.REAL_LOCATIONS["P2_RECEIPTS_DIR"], families.RECEIPTS_DIR)
        self.assertEqual(fx.REAL_LOCATIONS["HOLDOUT_DIR"], families.RUN / "holdout")
        self.assertEqual(fx.REAL_LOCATIONS["OWNER_DIRECTIVES_PATH"].name, families.OWNER_DIRECTIVE_NAMES[0])
        self.assertEqual(families.OWNER_MESSAGE_NAME, "dl-d5-owner-message-{sha16}.txt")

    def test_draft_and_simulation_markers_stop(self):
        """families.provenance_marks_draft: session/kind/message_utc words, draft or simulation true, confirmed_by_owner
        false, a non-empty unconfirmed list, a note starting with DRAFT or SIMULATION."""
        for name, over in (("session", {"session": DRAFT_SESSION}), ("draft", {"draft": True}),
                           ("simulation", {"simulation": True}), ("confirmed", {"confirmed_by_owner": False}),
                           ("unconfirmed", {"unconfirmed": ["Q5"]}), ("note", {"note": "SIMULATION of the answer"}),
                           ("lower", {"session": "owner session (draft)"})):
            with self.subTest(name):
                self.stops(fx.write_inputs(self.tmp / name, prov_overrides=over), "marks DRAFT or SIMULATION")

    def test_the_stored_owner_message_must_exist_and_hash_to_message_sha256(self):
        self.stops(fx.write_inputs(self.tmp / "absent", store_message=False),
                   "does not resolve to a pinned owner record")
        inputs = fx.write_inputs(self.tmp / "edited")
        inputs.owner_message.write_bytes(b"DL-D5 confirmed: Q1=b\n")
        self.stops(inputs, "does not hash to message_sha256")

    def test_message_fields(self):
        for name, over, regex in (("utc", {"message_utc": "5 Oct 2026, 08:00"}, "ISO-8601"),
                                  ("local-time", {"message_utc": "2026-10-05T08:00:00+02:00"}, "ISO-8601"),
                                  ("text", {"message_text": "DL-D5 confirmed: Q1=b"}, "message_text"),
                                  ("kind", {"kind": "agent_summary"}, "known kind"),
                                  ("degenerate", {"message_sha256": "0" * 64}, "degenerate"),
                                  ("no-session", {"session": " "}, "lacks session")):
            with self.subTest(name):
                self.stops(fx.write_inputs(self.tmp / name, prov_overrides=over), regex)
        inputs = fx.write_inputs(self.tmp / "with-text")                  # a message_text equal to the stored bytes
        doc = json.loads(inputs.decisions.read_bytes())
        self.assertNotIn("message_text", doc["provenance"])
        text = fx.owner_message(fx.sha(inputs.review_csv.read_bytes())).decode()
        self.passes(fx.write_inputs(self.tmp / "text-ok", prov_overrides={"message_text": text}))

    def test_an_explicit_record_counts_only_directly_in_run_receipts_p2(self):
        msg = b"DL-D5 confirmed (explicit record fixture): Q1=a, Q2=a, Q3=a, Q4=a, Q5=a\n"
        session_receipt = (f'{{"owner_message_sha256": "{fx.sha(msg)}", "message_utc": "{fx.OWNER_MESSAGE_UTC}"}}\n'
                           .encode())
        for name, where, raw, ok in (("elsewhere", self.tmp / "elsewhere", msg, False),
                                     ("staging", self.p2_receipts / "staging", msg, False),
                                     ("simulation", self.p2_receipts / "simulation", msg, False),
                                     ("direct-bytes", self.p2_receipts, msg, True),
                                     ("direct-names", self.p2_receipts, session_receipt, True)):
            with self.subTest(name):
                where.mkdir(parents=True, exist_ok=True)
                path = where / f"owner-record-{name}.txt"
                path.write_bytes(raw)
                record = {"path": str(path), "sha256": fx.sha(raw)}
                inputs = fx.write_inputs(self.tmp / name, message=msg, store_message=False,
                                         prov_overrides={"record": record},
                                         registry_provenance=fx.cl2_marker(record | {"form": "pinned_record"}))
                if ok:
                    r, _, _ = self.passes(inputs)
                    self.assertEqual(r["review_gate"]["provenance"]["record"]["form"], "pinned_record")
                else:
                    self.stops(inputs, "not directly in RUN/receipts/P2")

    def test_the_registry_marker_must_say_owner_built(self):
        """families writes decision_deltas.provenance = {draft, simulation, record} into every final registry."""
        good = fx.write_inputs(self.tmp / "good")
        record = json.loads(good.registry.read_bytes())["decision_deltas"]["provenance"]["record"]
        for name, marker, regex in (
                ("draft", fx.cl2_marker(record, draft=True), "built from DRAFT or SIMULATION"),
                ("simulation", fx.cl2_marker(record, simulation=True), "built from DRAFT or SIMULATION"),
                ("draft-null", {"draft": None, "simulation": False, "record": record}, "built from DRAFT or SIMULATION"),
                ("no-record", fx.cl2_marker(None), "not a pinned record"),
                ("other-record", fx.cl2_marker(record | {"sha256": fx.h("other message")}),
                 "not the owner record this decision file resolves to"),
                ("other-path", fx.cl2_marker(record | {"path": str(self.tmp / "x.txt")}),
                 "not the owner record this decision file resolves to"),
                ("absent", False, "carries no decision_deltas.provenance")):
            with self.subTest(name):
                self.stops(fx.write_inputs(self.tmp / name, registry_provenance=marker), regex)

    def test_a_simulation_file_is_never_a_decision_file(self):
        inputs = fx.write_inputs(self.tmp / "in")
        reg = holdout.load_registry(inputs.registry, inputs.registry_sha256)
        doc = json.loads(inputs.decisions.read_bytes()) | {"schema": families.SIMULATION_SCHEMA}
        with self.assertRaisesRegex(holdout.HoldoutStop, "decisions schema"):
            holdout.review_gate(reg, common.canonical(doc), inputs.review_csv.read_bytes())

    def test_an_owner_directive_must_be_the_canonical_file_or_named_by_a_stored_message(self):
        """D-MAP-5: families accepts any OWNER_DIRECTIVES.md by basename; P3 compares resolved paths."""
        text = b"# Owner directives (fixture)\nDL-D5 confirmed: Q1=a, Q2=a, Q3=a, Q4=a, Q5=a\n"

        def write(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text)
            return path

        def directive(path, **extra):
            return {"kind": "owner_directive", "file": str(path), "version": "fixture-1", "lines": "1-2",
                    "file_sha256": fx.sha(path.read_bytes())} | extra

        def marker(path):
            return fx.cl2_marker({"path": str(path), "sha256": fx.sha(path.read_bytes()), "form": "owner_directive"})

        canonical = write(holdout.OWNER_DIRECTIVES_PATH)
        self.assertTrue(str(canonical).startswith(str(self.tmp)))
        r, _, _ = self.passes(fx.write_inputs(self.tmp / "canonical", provenance=directive(canonical),
                                              registry_provenance=marker(canonical)))
        self.assertEqual(r["review_gate"]["provenance"]["record"]["form"], "owner_directive")
        same_name = write(self.tmp / "elsewhere" / "OWNER_DIRECTIVES.md")
        self.stops(fx.write_inputs(self.tmp / "basename", provenance=directive(same_name),
                                   registry_provenance=marker(same_name)), "canonical OWNER_DIRECTIVES.md")
        notes = write(self.tmp / "notes" / "dl-d5-directive.md")
        named_msg = f"My DL-D5 answer is {notes.name} lines 1-2.\n".encode()
        fx.store_owner_message(named_msg)
        named = {"kind": "owner_verbatim_message", "message_sha256": fx.sha(named_msg),
                 "message_utc": fx.OWNER_MESSAGE_UTC}
        self.passes(fx.write_inputs(self.tmp / "named", provenance=directive(notes, named_by_owner=named),
                                    registry_provenance=marker(notes)))
        loose = write(self.tmp / "loose" / "named-message.txt")            # the naming message kept outside P2
        loose.write_bytes(named_msg)
        named_elsewhere = named | {"record": {"path": str(loose), "sha256": fx.sha(named_msg)}}
        self.stops(fx.write_inputs(self.tmp / "named-elsewhere",
                                   provenance=directive(notes, named_by_owner=named_elsewhere),
                                   registry_provenance=marker(notes)), "not stored directly in RUN/receipts/P2")


class DraftDryRun(GateCase):
    """P3I-1 and D-MAP-6: a DRAFT is structurally unpromotable into a real commit; the dry run is Python-only."""

    def draft_inputs(self, name):
        return fx.write_inputs(self.tmp / name, store_message=False,
                               prov_overrides={"session": DRAFT_SESSION, "draft": True, "confirmed_by_owner": False},
                               registry_provenance=fx.cl2_marker(None, draft=True))

    def test_the_cli_has_no_dry_run_switch(self):
        parser = holdout._parser()
        options = [o for group in parser._subparsers._group_actions for sub in group.choices.values()
                   for a in sub._actions for o in a.option_strings]
        self.assertIn("--registry", options)
        self.assertFalse([o for o in options if "draft" in o or "dry" in o])

    def test_draft_decisions_stop_a_real_commit(self):
        r = self.stops(self.draft_inputs("real"), "marks DRAFT or SIMULATION")
        self.assertRegex(r["error"]["message"], "built from DRAFT or SIMULATION decisions")
        self.assertNotIn("draft_dry_run", r)

    def test_unconfirmed_file_shapes_are_refused_in_both_modes(self):
        """The DL-D5 tool's DRAFT and what-if files (their own schema, provenance kind and reviewer) are refused by the
        review gate in its real mode and in its dry-run mode alike (tools/p2_decisions.py holdout_refuses relies on
        it); only the rehearsal variant with the decision schema reaches a dry run."""
        inputs = fx.write_inputs(self.tmp / "in")
        reg = holdout.load_registry(inputs.registry, inputs.registry_sha256)
        base = json.loads(inputs.decisions.read_bytes())
        for schema, kind, reviewer in (("jope.dl-p2-decisions-draft-unconfirmed/v1", "relay_draft_unconfirmed", "relay"),
                                       ("jope.dl-p2-decisions-what-if/v1", "agent_what_if_not_a_decision", "agent")):
            doc = base | {"schema": schema, "reviewer": reviewer,
                          "provenance": base["provenance"] | {"kind": kind, "session": DRAFT_SESSION}}
            for dry in (False, True):
                with self.subTest(schema=schema, draft_dry_run=dry):
                    with self.assertRaises(holdout.HoldoutStop) as ctx:
                        holdout.review_gate(reg, common.canonical(doc), inputs.review_csv.read_bytes(),
                                            draft_dry_run=dry)
                    for part in ("decisions schema", "reviewer must be 'owner'", "known kind"):
                        self.assertIn(part, str(ctx.exception))

    def test_the_dry_run_passes_with_fake_clients_and_is_stamped(self):
        inputs = self.draft_inputs("dry")
        code, s, bucket, out = self.commit(inputs, draft_dry_run=True)
        self.assertEqual(code, 0, s)
        r = fx.receipt(s)
        self.assertEqual((r["status"], r["draft_dry_run"], s["draft_dry_run"]), ("PASS", True, True))
        self.assertRegex(r["notice"], "DRAFT dry run")
        prov = r["review_gate"]["provenance"]
        self.assertEqual((prov["draft"], prov["record"], prov["draft_dry_run"]), (True, None, True))
        self.assertEqual(bucket.puts(), [COMMITMENT_KEY])
        self.assertEqual(r["registry_object"]["version_id"], fx.REGISTRY_VID)
        code, s = fx.run_cli(fx.freeze_args(inputs, out), draft_dry_run=True)     # offline steps run as usual
        self.assertEqual(code, 0, s)
        self.assertTrue(fx.receipt(s)["draft_dry_run"])

    def test_the_dry_run_refuses_real_clients_and_the_real_holdout_dir(self):
        inputs = self.draft_inputs("guard")
        code, s = fx.run_cli(fx.commit_args(inputs, inputs.dir / "h1"), draft_dry_run=True)    # no injected clients
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "injected fake S3 clients")
        guarded = {"writer": fx.guarded_client(writes=True), "reader": fx.guarded_client()}   # botocore: refused
        code, s = fx.run_cli(fx.commit_args(inputs, inputs.dir / "h2"), clients=guarded, draft_dry_run=True)
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "botocore client")
        mixed = {"writer": fx.FakeS3Client(fx.registry_bucket(inputs), writes=True), "reader": fx.guarded_client()}
        code, s = fx.run_cli(fx.commit_args(inputs, inputs.dir / "h3"), clients=mixed, draft_dry_run=True)
        self.assertEqual(code, 2, s)
        bucket = fx.registry_bucket(inputs)
        for out in (holdout.HOLDOUT_DIR, holdout.HOLDOUT_DIR / "attempt-1"):
            code, s = fx.run_cli(fx.commit_args(inputs, out) + ["--receipt", self.tmp / f"r-{out.name}.json"],
                                 clients=fx.fake_clients(bucket), draft_dry_run=True)
            self.assertEqual(code, 2, s)
            self.assertRegex(fx.receipt(s)["error"]["message"], "real holdout directory")
        self.assertEqual(bucket.calls, [])
        self.assertFalse(holdout.HOLDOUT_DIR.exists())

    def test_publish_in_a_dry_run_refuses_real_clients(self):
        inputs = fx.write_inputs(self.tmp / "in")
        out, bucket, _ = fx.publish_ready(self, inputs)
        guarded = {"writer": fx.guarded_client(writes=True), "reader": fx.guarded_client()}
        code, s = fx.run_cli(fx.publish_args(out), clients=guarded, draft_dry_run=True)
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "botocore client")
        code, s = fx.run_cli(fx.publish_args(out), draft_dry_run=True)
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "injected fake S3 clients")
        self.assertEqual(bucket.puts(), [COMMITMENT_KEY])
        code, s = fx.run_cli(fx.publish_args(out), clients=fx.fake_clients(bucket), draft_dry_run=True)
        self.assertEqual(code, 0, s)
        self.assertTrue(fx.receipt(s)["draft_dry_run"])


class RegistryExactVersion(GateCase):
    """P3I-2: no commitment and no binding for a registry ref nobody read at its exact VersionId."""

    def setUp(self):
        super().setUp()
        self.inputs = fx.write_inputs(self.tmp / "in")
        self.key = fx.registry_s3_key(self.inputs.registry_sha256)
        self.body = self.inputs.registry.read_bytes()

    def bucket_with(self, body, version_id=fx.REGISTRY_VID):
        bucket = fx.FakeBucket()
        bucket.plant(self.key, body, version_id)
        return bucket

    def stops_before_any_write(self, bucket, regex, inputs=None):
        code, s, bucket, out = self.commit(inputs or self.inputs, bucket=bucket)
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], regex)
        self.assertEqual(bucket.puts(), [])
        self.assertEqual([op for op, _ in bucket.calls], ["GetObject"])          # nothing after the registry GET
        self.assertFalse((out / "seed.hex").exists())
        self.assertFalse((out / "commitment.json").exists())

    def test_commit_reads_the_registry_at_its_version_first(self):
        r, bucket, _ = self.passes(self.inputs)
        ops = [op for op, _ in bucket.calls]
        self.assertEqual(bucket.calls[0], ("GetObject", self.key))
        self.assertLess(ops.index("GetObject"), ops.index("ListObjectVersions"))
        self.assertLess(ops.index("ListObjectVersions"), ops.index("PutObject"))
        ref = r["p3_gate"]["registry_s3"]
        self.assertEqual({k: r["registry_object"][k] for k in holdout.REF_KEYS}, ref)
        self.assertEqual((ref["version_id"], ref["sha256"], ref["bytes"]),
                         (fx.REGISTRY_VID, self.inputs.registry_sha256, len(self.body)))

    def test_commit_stops_when_the_version_does_not_resolve_or_holds_other_bytes(self):
        self.stops_before_any_write(fx.FakeBucket(), "does not resolve")             # never published
        self.stops_before_any_write(self.bucket_with(self.body, "vAnotherVersion"), "HTTP 404")
        self.stops_before_any_write(self.bucket_with(None), "HTTP 405")               # a delete marker version
        self.stops_before_any_write(self.bucket_with(self.body + b"\n"), "is not this registry")

    def test_a_draft_receipts_fake_version_never_reaches_a_commitment(self):
        """D-MAP-6 route 2: a P2 receipt from a fake bucket names a VersionId that S3 does not hold."""
        inputs = fx.write_inputs(self.tmp / "draft-ref", p2={"ref_overrides": {"version_id": "v0005-deadbeef"}})
        self.stops_before_any_write(fx.registry_bucket(inputs), "does not resolve", inputs=inputs)

    def test_publish_rereads_the_registry_before_its_first_put(self):
        out, bucket, _ = fx.publish_ready(self, self.inputs)
        for versions, regex in (([(fx.REGISTRY_VID, self.body + b"\n")], "is not this registry"),
                                ([], "does not resolve")):
            bucket.objects[(s3io.BUCKET, self.key)] = versions
            code, s = fx.run_cli(fx.publish_args(out), clients=fx.fake_clients(bucket))
            self.assertEqual(code, 2, s)
            self.assertRegex(fx.receipt(s)["error"]["message"], regex)
            self.assertEqual(bucket.puts(), [COMMITMENT_KEY])
            self.assertIsNone(bucket.current(BINDING_KEY))
        bucket.objects[(s3io.BUCKET, self.key)] = [(fx.REGISTRY_VID, self.body)]
        code, s = fx.run_cli(fx.publish_args(out), clients=fx.fake_clients(bucket))
        self.assertEqual(code, 0, s)
        r = fx.receipt(s)
        ref = {k: r["registry_object"][k] for k in holdout.REF_KEYS}
        self.assertEqual(json.loads(bucket.current(BINDING_KEY)[1])["registry_ref"], ref)
        self.assertEqual(ref["version_id"], fx.REGISTRY_VID)

    def test_the_guarded_client_sends_one_exact_version_get(self):
        ref = {"bucket": s3io.BUCKET, "key": self.key, "version_id": fx.REGISTRY_VID,
               "sha256": self.inputs.registry_sha256, "bytes": len(self.body)}
        params = {"Bucket": s3io.BUCKET, "Key": self.key, "VersionId": fx.REGISTRY_VID}
        reader = fx.guarded_client()

        def check():
            return holdout.check_registry_object(reader, ref, self.inputs.registry_sha256, len(self.body))
        with Stubber(reader) as sr:
            sr.add_response("get_object", {"Body": fx.streaming(self.body), "VersionId": fx.REGISTRY_VID}, params)
            got = check()
            sr.add_response("get_object", {"Body": fx.streaming(self.body), "VersionId": "vEchoedOther"}, params)
            with self.assertRaisesRegex(holdout.HoldoutStop, "is not this registry"):
                check()
            for status, code in ((404, "NoSuchVersion"), (400, "InvalidArgument"), (405, "MethodNotAllowed")):
                sr.add_client_error("get_object", service_error_code=code, http_status_code=status,
                                    expected_params=params)
                with self.assertRaisesRegex(holdout.HoldoutStop, f"HTTP {status}"):
                    check()
            sr.add_client_error("get_object", service_error_code="AccessDenied", http_status_code=403,
                                expected_params=params)
            with self.assertRaises(Exception) as ctx:
                check()
            self.assertNotIsInstance(ctx.exception, holdout.HoldoutStop)          # not a STOP: propagates (ERROR)
            sr.assert_no_pending_responses()
        self.assertEqual({k: got[k] for k in holdout.REF_KEYS}, ref)
        for bad in (ref | {"bucket": "another-bucket"}, ref | {"version_id": "null"}, ref | {"version_id": ""},
                    ref | {"bytes": 1}, ref | {"sha256": fx.h("other")}, ref | {"key": self.key + ".bak"}, None):
            with self.assertRaisesRegex(holdout.HoldoutStop, "incomplete or does not name"):
                holdout.check_registry_object(reader, bad, self.inputs.registry_sha256, len(self.body))


class HistoryDocs(fx.P3Case):
    """P3I-5: nothing says that the lane guard blocks P3's history checks; the docs name the form it allows."""

    def test_the_docs_match_the_guard(self):
        stale = r"(?i)does not allow that operation|today's s3io lane guard|until s3io permits|guard refuses the operation"
        for doc in (holdout.__doc__, holdout.key_history.__doc__, fx.__doc__, fx.proposed_history_guard.__doc__):
            self.assertNotRegex(doc, stale)
        self.assertIn("the bounded form s3io's lane guard allows", " ".join(holdout.__doc__.split()))
        self.assertIn("allows exactly this form", " ".join(holdout.key_history.__doc__.split()))
        self.assertIn("ListObjectVersions", s3io.READ_OPS)
        self.assertLessEqual(holdout.HISTORY_MAX, s3io.HISTORY_MAX)
        s3io.Guard().check_params({"Bucket": s3io.BUCKET, "Prefix": COMMITMENT_KEY, "MaxKeys": holdout.HISTORY_MAX},
                                  type("Model", (), {"name": "ListObjectVersions"}))

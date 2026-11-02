"""Core HWM publisher fixture cases, mixed into the legacy test module."""

from _hwm_publish_fixture import *  # noqa: F401,F403


class CoreCases:
    def test_valid_manifest_exact_version_and_no_label_data(self):
        result, bucket = self.publish()
        self.assertEqual(result["counts"], {"selection_reached": 1})
        for versions in bucket.objects.values():
            self.assertEqual(len(versions), 1)
            for _, body in versions:
                self.assertNotIn(b"PRIVATE_CANARY", body)
                self.assertNotIn(b"outer_test", body)
                self.assertNotIn(b"leader_test_rmse_field", body)
        reads = [c for c in bucket.calls if c[0] == "GetObject"]
        self.assertTrue(reads and all(c[2] for c in reads))
        obj = h.inventory(self.source, self.header_sha)[1][0]["body"]
        self.assertEqual(obj["authority"], "progress_evidence_only")
        self.assertFalse(obj["g0_qualification"]["attempt_accounting_verified"])

    def test_research_and_dev_scopes_preserved(self):
        self.lane, self.role = "research", "dev"
        self.emit()
        obj = h.inventory(self.source, self.header_sha)[1][0]
        self.assertTrue(obj["key"].startswith(h.s3io.V2_RESEARCH_ONLY_KEY))
        self.assertEqual(obj["body"]["scope"], "selection_only")
        result, bucket = self.publish()
        for key in bucket.keys():  # noqa: SIM118 - FakeBucket API, not a dict
            if "/gp-hwm/" in key:
                self.assertTrue(key.startswith(h.s3io.V2_RESEARCH_ONLY_KEY))
        self.assertEqual(result["completed"], 1)

    def assert_output_alias_refused(self, work_dir, receipt):
        plan, sha = self.freeze()
        before = {
            str(p.relative_to(self.source)): p.read_bytes()
            for p in self.source.rglob("*")
            if p.is_file()
        }
        with (
            mock.patch.object(h.transport, "fake_clients") as fake,
            mock.patch.object(h.transport, "real_clients") as real,
            mock.patch.object(h.transport, "PublishLedger") as ledger,
        ):
            with self.assertRaises(h.Refused):
                h.publish(plan, sha, work_dir, receipt, True)
            fake.assert_not_called()
            real.assert_not_called()
            ledger.assert_not_called()
        after = {
            str(p.relative_to(self.source)): p.read_bytes()
            for p in self.source.rglob("*")
            if p.is_file()
        }
        self.assertEqual(before, after)
        self.assertFalse((self.root / "publish").exists())

    def test_work_directory_parent_alias_refused_before_state(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.assert_output_alias_refused(
            outside / ".." / "source", self.root / "receipt.json"
        )

    def test_receipt_parent_alias_refused_before_state(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.assert_output_alias_refused(
            self.root / "publish", outside / ".." / "source" / "receipt.json"
        )

    def test_no_selection_is_a_terminal_record(self):
        self.final["outcome"] = self.event["outcome"] = "g0_not_ready"
        del self.final["selection"]
        for k in ("selected_program_sha256", "selected_cv_ratio", "selected_fitness"):
            self.event[k] = None
        self.emit()
        plan, objects = h.inventory(self.source, self.header_sha)
        self.assertEqual(plan["counts"], {"g0_not_ready": 1})
        self.assertIn("/gp-terminal/", objects[0]["key"])
        self.assertNotIn("selection", objects[0]["body"])

    def test_wrong_header_refused(self):
        with self.assertRaises(h.Refused):
            h.inventory(self.source, "0" * 64)

    def test_changed_dependency_refused(self):
        (self.source / "registry.json").write_bytes(b"{}")
        with self.assertRaises(h.Refused):
            h.inventory(self.source, self.header_sha)

    def test_final_hash_swap_refused_before_clients(self):
        plan, sha = self.freeze()
        (self.source / "contexts" / self.cid / "final.json").write_bytes(b"{}")
        with (
            mock.patch.object(h.transport, "fake_clients") as clients,
            self.assertRaises(h.Refused),
        ):
            h.publish(
                plan, sha, self.root / "publish", self.root / "receipt.json", True
            )
        clients.assert_not_called()
        self.assertFalse((self.root / "publish").exists())

    def test_wrong_cut_program_metrics_and_nonfinite_refused(self):
        for field, value in (
            ("cut_id", "trainonly-split-v2:" + "0" * 64),
            ("s1_input_sha256", "0" * 64),
        ):
            with self.subTest(field=field):
                saved = self.final[field]
                self.final[field] = value
                self.emit()
                with self.assertRaises(h.Refused):
                    h.inventory(self.source, self.header_sha)
                self.final[field] = saved
        for field, value in (
            ("leader_test_rmse_field", 0.01),
            ("program_sha256", "0" * 64),
            ("cv_geomean_rmse_ratio", 0.95),
            ("fitness", True),
        ):
            with self.subTest(field=field):
                old = self.final["selection"][field]
                self.final["selection"][field] = value
                self.emit()
                with self.assertRaises(h.Refused):
                    h.inventory(self.source, self.header_sha)
                self.final["selection"][field] = old
        with self.assertRaises(h.Refused):
            h.decode(b'{"fitness":NaN}')

    def test_rights_and_family_refusals(self):
        for role in ("heldout", "sealed_excluded", "unknown"):
            self.role = role
            self.emit()
            with self.assertRaises(h.Refused):
                h.inventory(self.source, self.header_sha)
        self.role, self.denied = "train_fit", True
        self.emit()
        with self.assertRaises(h.Refused):
            h.inventory(self.source, self.header_sha)

    def test_duplicate_and_torn_final_ledger_refused(self):
        p = self.source / "ledger-production.jsonl"
        initial = p.read_bytes()
        p.write_bytes(initial + h.common.canonical(self.event) + b"\n")
        with self.assertRaises(h.Refused):
            h.inventory(self.source, self.header_sha)
        p.write_bytes(initial.rstrip(b"\n"))
        with self.assertRaises(h.Refused):
            h.inventory(self.source, self.header_sha)

    def test_missing_committed_artifact_hard_fails(self):
        (self.source / "contexts" / self.cid / "summary.tar.gz").unlink()
        with self.assertRaises(FileNotFoundError):
            h.inventory(self.source, self.header_sha)

    def test_ledger_append_does_not_change_frozen_set(self):
        plan, sha = self.freeze()
        with (self.source / "ledger-production.jsonl").open("ab") as stream:
            stream.write(b'{"event":"defer","context_sha256":"new"}\n')
        self.assertEqual(h.prepare(plan, sha)[0]["artifact_count"], 1)

    def test_forbidden_only_perturbation_leaves_payload_unchanged(self):
        before = h.inventory(self.source, self.header_sha)
        forbidden = self.source / "outer-test" / "private.json"
        self.write(forbidden, {"targets": [1]})
        self.write(forbidden, {"targets": [999]})
        after = h.inventory(self.source, self.header_sha)
        self.assertEqual(before, after)
        with self.assertRaises(h.Refused):
            h.read_regular(forbidden)

    def test_authorized_program_change_invalidates_frozen_plan(self):
        plan, sha = self.freeze()
        self.final["selection"]["program"]["dsl"] = "(pipe X (ridge lambda_frac=0.02))"
        ph = h.digest(self.final["selection"]["program"]["dsl"].encode())
        self.final["selection"]["program"]["program_sha256"] = self.final["selection"][
            "program_sha256"
        ] = ph
        self.event["selected_program_sha256"] = ph
        self.emit()
        with self.assertRaises(h.Refused):
            h.prepare(plan, sha)

    def test_duplicate_json_keys_refused(self):
        with self.assertRaises(h.Refused):
            h.decode(b'{"a":1,"a":2}')

    def test_symlink_open_rejected_without_creating_a_link(self):
        # Host owner prohibits new symlinks; inject ELOOP at the O_NOFOLLOW boundary.
        with (
            mock.patch.object(h.os, "open", side_effect=OSError(40, "link refused")),
            self.assertRaises(OSError),
        ):
            h.read_regular(self.source / "ledger.header.json")

    def test_interrupted_put_resume_and_duplicate_delivery(self):
        plan, sha = self.freeze()
        _, objects = h.prepare(plan, sha)
        clients = h.transport.fake_clients(self.root / "budget.jsonl")
        bucket = clients[2]
        key = objects[0]["key"]
        bucket.put_faults[key] = [("after", RuntimeError("interrupted"))]
        with mock.patch.object(h.transport, "fake_clients", return_value=clients):
            with self.assertRaises(RuntimeError):
                h.publish(
                    plan, sha, self.root / "publish", self.root / "failed.json", True
                )
            self.assertFalse(any("/manifests/" in k for k in list(bucket.keys())))
            failed = json.loads((self.root / "failed.json").read_bytes())
            self.assertEqual((failed["status"], failed["remaining"]), ("failed", 1))
            h.publish(
                plan, sha, self.root / "publish", self.root / "resumed.json", True
            )
            put_count = len(bucket.puts())
            h.publish(
                plan, sha, self.root / "publish", self.root / "duplicate.json", True
            )
            self.assertEqual(len(bucket.puts()), put_count)
            self.assertEqual(len(bucket.objects[(h.s3io.BUCKET, key)]), 1)

    def test_corrupted_exact_readback_blocks_manifest(self):
        plan, sha = self.freeze()
        _, objects = h.prepare(plan, sha)
        clients = h.transport.fake_clients(self.root / "budget.jsonl")
        clients[2].corrupt_reads[objects[0]["key"]] = b"corrupt"
        with (
            mock.patch.object(h.transport, "fake_clients", return_value=clients),
            self.assertRaises(h.s3io.ReadbackMismatch),
        ):
            h.publish(
                plan, sha, self.root / "publish", self.root / "receipt.json", True
            )
        self.assertFalse(any("/manifests/" in k for k in list(clients[2].keys())))

    def test_existing_different_object_conflicts(self):
        plan, sha = self.freeze()
        _, objects = h.prepare(plan, sha)
        clients = h.transport.fake_clients(self.root / "budget.jsonl")
        clients[2].objects[(h.s3io.BUCKET, objects[0]["key"])] = [
            ("already", b"different")
        ]
        with (
            mock.patch.object(h.transport, "fake_clients", return_value=clients),
            self.assertRaises(h.s3io.PoisonedKey),
        ):
            h.publish(
                plan, sha, self.root / "publish", self.root / "receipt.json", True
            )

    def test_dry_run_never_constructs_aws_or_touches_original_budget(self):
        plan, sha = self.freeze()
        budget = self.root / "original-budget.jsonl"
        # Empty valid ledger; ensure the original stays byte-identical.
        budget.write_bytes(b"")
        with mock.patch.object(
            h.transport, "real_clients", side_effect=AssertionError("AWS forbidden")
        ):
            h.publish(
                plan,
                sha,
                self.root / "publish",
                self.root / "receipt.json",
                True,
                budget,
            )
        self.assertEqual(budget.read_bytes(), b"")
        self.assertNotEqual(
            (self.root / "publish/dry-run-budget.jsonl").read_bytes(), b""
        )



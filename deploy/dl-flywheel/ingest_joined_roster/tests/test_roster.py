import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from jope_flywheel.protocol import partition

from ingest_joined_roster.roster import (
    ARTIFACTS,
    assemble,
    canonical,
    digest,
    document,
    publish,
    read,
    read_committed,
)


def fixture(root):
    sources = {}

    def pin(name, body):
        p = root / name
        p.write_bytes(body)
        ref = {"path": str(p), "sha256": digest(body), "bytes": len(body)}
        sources[name] = ref
        return ref

    def array(value):
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        return stream.getvalue()

    split = "1" * 64
    parents = {}
    for number in range(2000):
        lineage = digest(str(number).encode())
        parents.setdefault(partition(lineage, split), lineage)
        if len(parents) == 2:
            break
    cids = ["a" * 64, "b" * 64]
    ds = ["1" * 16, "2" * 16]
    views = ["native-train", "native-val"]
    dsl = ["historical-train-dsl", "historical-val-dsl"]
    progs = [digest(p.encode()) for p in dsl]
    cut = "frozen-native-cut"
    inventory = [
        {
            "context_sha256": cid,
            "view_id": view,
            "dataset_hash": dataset,
            "fold": i,
            "cut_id": cut,
            "current_role": "training_validation_eligible",
            "gp_index": {"outcome": "selection_reached"},
        }
        for i, (cid, view, dataset) in enumerate(zip(cids, views, ds))
    ]
    inventory.append(
        {
            "context_sha256": None,
            "view_id": "native-missing",
            "dataset_hash": "3" * 16,
            "fold": 0,
            "cut_id": cut,
            "current_role": "training_validation_eligible",
            "gp_index": None,
        }
    )
    pin("inventory", canonical({"schema": "jope.full-native-corpus-inventory-audit/v1", "rows": inventory}))
    pin(
        "meta",
        canonical(
            {
                "schema": "dlhwm.data/v1",
                "n": 2,
                "cids": cids,
                "view_id": views,
                "dataset": ds,
                "fold": [0, 1],
                "program_sha256": progs,
            }
        ),
    )
    pin(
        "winners",
        canonical(
            {
                "programs": [
                    {"cid": cid, "program_sha256": ph, "dsl": text} for cid, ph, text in zip(cids, progs, dsl)
                ]
            }
        ),
    )
    pin("features", array(np.zeros((2, 2051), dtype="<f4")))
    pin("gid", array(np.array([1, 2], dtype="<i8")))
    pin(
        "bundle_manifest",
        canonical(
            {
                "files": {
                    "meta.json": sources["meta"]["sha256"],
                    "hwm_programs.json": sources["winners"]["sha256"],
                    "blocks_kumo_tabular_s.npy": sources["features"]["sha256"],
                    "gid.npy": sources["gid"]["sha256"],
                }
            }
        ),
    )
    pin(
        "parents",
        b"".join(
            canonical(
                {
                    "parent_lineage_sha256": parents[role],
                    "partition": role,
                    "split_namespace_sha256": split,
                    "bundle_manifest_sha256": sources["bundle_manifest"]["sha256"],
                    "gid_sha256": sources["gid"]["sha256"],
                    "retained_lineage_context_sha256": [cid],
                    "retained_group": i + 1,
                }
            )
            + b"\n"
            for i, (role, cid) in enumerate(zip(["train", "validation"], cids))
        ),
    )
    pin("target", canonical({"views": views}))
    prefix = "program-results/flywheel/v1/"
    cid = cids[0]
    program = "c" * 64
    artifacts = {
        role: {
            "bucket": "fixture",
            "key": prefix + "objects/" + (program if role == "program" else "d" * 64) + "/" + role,
            "sha256": program if role == "program" else "d" * 64,
            "version_id": "exact-version",
            "bytes": 8,
        }
        for role in ARTIFACTS
    }
    evaluation = "e" * 64
    record = {
        "schema": "jope.native-original-program-result/v1",
        "context": {
            "identity": cid,
            "dataset_hash": ds[0],
            "cut_id": cut,
            "input_sha256": "d" * 64,
            "target_sha256": "d" * 64,
            "split_sha256": "d" * 64,
            "evaluator_sha256": "d" * 64,
        },
        "program_sha256": program,
        "evaluation_key": evaluation,
        "status": "completed",
        "measurement": "executed",
        "artifacts": artifacts,
    }
    body = canonical(record)
    ref = pin("record", body)
    pin(
        "object_index",
        canonical(
            [
                {
                    "kind": "program_result",
                    "state": "copied",
                    "copy": ref["path"],
                    "ref": {
                        "bucket": "fixture",
                        "key": prefix + evaluation + "/record.json",
                        "version_id": "exact-version",
                        "sha256": ref["sha256"],
                        "bytes": ref["bytes"],
                    },
                }
            ]
        ),
    )
    sources.pop("record")
    plan = {"schema": "jope.native-ingest-plan/v1", "non_qualifying": True, "sources": sources}
    return plan


def update(plan, name, value):
    body = canonical(value)
    ref = plan["sources"][name]
    Path(ref["path"]).write_bytes(body)
    ref.update(sha256=digest(body), bytes=len(body))


class RosterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.plan = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_roles_rewards_and_native_identities_are_preserved(self):
        manifest, blobs = assemble(self.plan)
        train = document(blobs["train.jsonl"])
        val = document(blobs["validation.jsonl"])
        self.assertNotEqual(train["parent_lineage_sha256"], val["parent_lineage_sha256"])
        self.assertEqual(train["context_id"], "a" * 64)
        self.assertIsNone(train["measured_program_candidates"][0]["q"])
        self.assertFalse(train["learner_admitted"])
        self.assertIsNone(document(blobs["backlog.jsonl"])["context_id"])
        self.assertEqual(manifest["summary"]["family_overlap"], 0)
        self.assertEqual(train.pop("row_sha256"), digest(canonical(train)))

    def test_role_packs_bind_each_vector_without_validation_payload(self):
        manifest, blobs = assemble(self.plan)
        train = document(blobs["train.jsonl"])
        val = document(blobs["validation.jsonl"])
        self.assertEqual(train["features"]["path"], "train-vectors.f32")
        self.assertEqual(val["features"]["path"], "validation-vectors.f32")
        for row in (train, val):
            ref = row["features"]
            pack = blobs[ref["path"]]
            raw = pack[ref["offset"] : ref["offset"] + ref["bytes"]]
            self.assertEqual(digest(raw), ref["sha256"])
            self.assertEqual(digest(pack), ref["pack_sha256"])
            self.assertEqual(len(pack), ref["pack_bytes"])
            self.assertEqual(manifest["members"][ref["path"]]["sha256"], ref["pack_sha256"])

    def test_current_selection_evidence_does_not_invent_old_identity_or_inputs(self):
        inv = document(read(self.plan["sources"]["inventory"]))
        inv["rows"][-1]["s1_finals"] = [{"context_sha256": "f" * 64}]
        update(self.plan, "inventory", inv)
        dsl = "new-selection-evidence"
        row = {
            "schema": "jope.gp-hwm-evidence/v1",
            "context_sha256": "f" * 64,
            "dataset_hash": "3" * 16,
            "cut_id": "frozen-native-cut",
            "fold": 0,
            "outcome": "selection_reached",
            "authority": "progress_evidence_only",
            "selection": {"program": {"program_sha256": digest(dsl.encode()), "dsl": dsl}},
        }
        raw = canonical(row)
        p = self.root / "new-selection"
        p.write_bytes(raw)
        index = document(read(self.plan["sources"]["object_index"]))
        index.append(
            {
                "kind": "backlog_current_hwm",
                "view_id": "native-missing",
                "state": "copied",
                "copy": str(p),
                "ref": {
                    "bucket": "fixture",
                    "key": "hwm/frozen.json",
                    "version_id": "immutable",
                    "sha256": digest(raw),
                    "bytes": len(raw),
                },
            }
        )
        update(self.plan, "object_index", index)
        manifest, blobs = assemble(self.plan)
        pending = document(blobs["backlog.jsonl"])
        self.assertIsNone(pending["context_id"])
        self.assertIsNone(pending["q"])
        self.assertTrue(pending["current_selection_evidence_present"])
        self.assertIn("no_retained_2051_encoding", pending["reasons"])
        self.assertEqual(manifest["summary"]["backlog_with_current_selection_evidence"], 1)

    def test_missing_cut_outside_retained_inputs_stays_explicit(self):
        inv = document(read(self.plan["sources"]["inventory"]))
        inv["rows"][-1]["cut_id"] = None
        update(self.plan, "inventory", inv)
        _, blobs = assemble(self.plan)
        self.assertIsNone(document(blobs["backlog.jsonl"])["cut_id"])

    def test_dedup_rerun_and_output_root_independence(self):
        first = publish(self.plan, self.root / "one")
        second = publish(self.plan, self.root / "one")
        index = document(read(self.plan["sources"]["object_index"]))
        update(self.plan, "object_index", index * 2)
        _, blobs = assemble(self.plan)
        self.assertEqual(len(document(blobs["train.jsonl"])["measured_program_candidates"]), 1)
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertEqual(first["manifest_sha256"], publish(self.plan, self.root / "dedup")["manifest_sha256"])
        self.plan = fixture(self.root)
        self.assertEqual(first["manifest_sha256"], publish(self.plan, self.root / "two")["manifest_sha256"])

    def test_duplicate_refusals_preserve_manifest_identity(self):
        index = document(read(self.plan["sources"]["object_index"]))
        item = index[0]
        record = document(Path(item["copy"]).read_bytes())
        record["schema"] = "jope.flywheel-program-result/v1"
        raw = canonical(record)
        Path(item["copy"]).write_bytes(raw)
        item["ref"].update(sha256=digest(raw), bytes=len(raw))
        update(self.plan, "object_index", index)
        first = publish(self.plan, self.root / "refused-once")
        update(self.plan, "object_index", index * 2)
        duplicate = publish(self.plan, self.root / "refused-twice")
        self.assertEqual(first["manifest_sha256"], duplicate["manifest_sha256"])

    def test_manifest_absent_until_all_members_are_present(self):
        output = self.root / "out"

        def fault():
            self.assertTrue((output / "train.jsonl").exists())
            self.assertFalse((output / "manifest.json").exists())
            raise RuntimeError("crash before commit")

        with self.assertRaisesRegex(RuntimeError, "crash"):
            publish(self.plan, output, before_manifest=fault)
        self.assertFalse((output / "manifest.json").exists())
        result = publish(self.plan, output)
        self.assertEqual(digest((output / "manifest.json").read_bytes()), result["manifest_sha256"])

    def test_consumer_refuses_prepared_only_and_corrupted_committed_members(self):
        output = self.root / "consumer"

        def crash():
            with self.assertRaises(FileNotFoundError):
                read_committed(output, "0" * 64)
            raise RuntimeError("uncommitted output")

        with self.assertRaises(RuntimeError):
            publish(self.plan, output, before_manifest=crash)
        result = publish(self.plan, output)
        pinned = result["manifest_sha256"]
        committed = read_committed(output, pinned)
        self.assertEqual(len(committed["rows"]["train.jsonl"]), 1)
        vector = output / "train-vectors.f32"
        vector.write_bytes(b"corrupted member")
        with self.assertRaisesRegex(ValueError, "bytes/hash drift"):
            read_committed(output, pinned)

    def test_publication_refuses_symlinked_parent_and_lock_before_writing(self):
        actual = self.root / "actual"
        actual.mkdir()
        link = self.root / "link"
        link.symlink_to(actual, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            publish(self.plan, link / "out")
        self.assertFalse((actual / "out").exists())
        output = self.root / "bad-lock"
        output.mkdir()
        target = self.root / "protected-file"
        target.write_text("unchanged")
        (output / ".publish.lock").symlink_to(target)
        with self.assertRaisesRegex(ValueError, "lock symlink"):
            publish(self.plan, output)
        self.assertEqual(target.read_text(), "unchanged")
        self.assertFalse((output / "manifest.json").exists())

    def test_final_directory_failure_preserves_prepared_bytes_without_commit(self):
        import os

        output = self.root / "out"
        real = os.fsync

        def failing(fd):
            if (output / "manifest.json").exists():
                raise OSError("final directory fsync")
            return real(fd)

        with patch("ingest_joined_roster.roster.os.fsync", side_effect=failing):
            with self.assertRaisesRegex(OSError, "final directory"):
                publish(self.plan, output)
        self.assertFalse((output / "manifest.json").exists())
        self.assertEqual(len(list(output.glob(".manifest-*.prepared"))), 1)
        publish(self.plan, output)

    def test_current_exclusion_blocks_old_teacher(self):
        inv = document(read(self.plan["sources"]["inventory"]))
        inv["rows"][0]["current_role"] = "sealed_excluded"
        update(self.plan, "inventory", inv)
        manifest, blobs = assemble(self.plan)
        self.assertEqual(blobs["train.jsonl"], b"")
        self.assertEqual(manifest["summary"]["train_contexts"], 0)

    def test_role_change_cannot_reassign_descendants(self):
        ref = self.plan["sources"]["parents"]
        rows = [document(x) for x in read(ref).splitlines()]
        rows[0]["partition"] = "validation"
        body = b"".join(canonical(r) + b"\n" for r in rows)
        Path(ref["path"]).write_bytes(body)
        ref.update(sha256=digest(body), bytes=len(body))
        with self.assertRaisesRegex(ValueError, "family/role"):
            assemble(self.plan)

    def test_duplicate_lineage_crosswalk_is_refused(self):
        ref = self.plan["sources"]["parents"]
        rows = [document(x) for x in read(ref).splitlines()]
        rows[1]["retained_lineage_context_sha256"] = rows[0]["retained_lineage_context_sha256"]
        body = b"".join(canonical(r) + b"\n" for r in rows)
        Path(ref["path"]).write_bytes(body)
        ref.update(sha256=digest(body), bytes=len(body))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            assemble(self.plan)

    def test_boolean_fold_does_not_alias_integer(self):
        inv = document(read(self.plan["sources"]["inventory"]))
        inv["rows"][0]["fold"] = False
        update(self.plan, "inventory", inv)
        with self.assertRaisesRegex(ValueError, "fold type"):
            assemble(self.plan)

    def test_pinned_source_change_is_refused(self):
        Path(self.plan["sources"]["inventory"]["path"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "bytes/hash drift"):
            assemble(self.plan)

    def test_source_and_output_symlinks_are_refused(self):
        ref = self.plan["sources"]["inventory"]
        link = self.root / "link"
        link.symlink_to(ref["path"])
        ref["path"] = str(link)
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            assemble(self.plan)
        self.plan = fixture(self.root)
        output = self.root / "out"
        output.mkdir()
        (output / "train.jsonl").symlink_to(self.root / "inventory")
        with self.assertRaisesRegex(ValueError, "output member symlink"):
            publish(self.plan, output)

    def test_existing_committed_member_drift_is_refused(self):
        output = self.root / "out"
        publish(self.plan, output)
        (output / "train.jsonl").write_text("drift")
        with self.assertRaisesRegex(ValueError, "output drift"):
            publish(self.plan, output)

    def test_duplicate_json_keys_and_nonfinite_numbers_are_refused(self):
        for body in (b'{"k":1,"k":2}', b'{"k":NaN}'):
            with self.assertRaises(ValueError):
                document(body)

    def test_cross_context_program_is_refused(self):
        index = document(read(self.plan["sources"]["object_index"]))
        row = index[0]
        body = document(Path(row["copy"]).read_bytes())
        body["context"]["dataset_hash"] = "9" * 16
        raw = canonical(body)
        Path(row["copy"]).write_bytes(raw)
        row["ref"].update(sha256=digest(raw), bytes=len(raw))
        update(self.plan, "object_index", index)
        with self.assertRaisesRegex(ValueError, "cross-context"):
            assemble(self.plan)

    def test_artifact_namespace_escape_is_refused(self):
        index = document(read(self.plan["sources"]["object_index"]))
        row = index[0]
        body = document(Path(row["copy"]).read_bytes())
        body["artifacts"]["input"]["key"] = "different-prefix"
        raw = canonical(body)
        Path(row["copy"]).write_bytes(raw)
        row["ref"].update(sha256=digest(raw), bytes=len(raw))
        update(self.plan, "object_index", index)
        with self.assertRaisesRegex(ValueError, "namespace/hash/role"):
            assemble(self.plan)


if __name__ == "__main__":
    unittest.main()

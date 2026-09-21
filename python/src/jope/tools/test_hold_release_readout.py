import json
import tempfile
import unittest
from pathlib import Path

import hold_release_readout as hr
from resume_expired_cycle import atomic, sha

RUN = "runs/x"


def journal_line(**event):
    return json.dumps(event) + "\n"


class HoldReleaseReadoutTests(unittest.TestCase):
    """A synthetic run: release acknowledged, the first monitored block rejected, then the rollback happens (or not)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / RUN
        (self.run / "learning").mkdir(parents=True)
        self.decision = self.root / "decision.json"
        atomic(
            self.decision, dict(schema=hr.DECISION_SCHEMA, verdict="inconclusive", final_tests_opened=False)
        )
        self.nonce = "hold-release-" + sha(self.decision)[:16]
        self.accepted = self.checkpoint("cycle-000018", 60000)
        self.rejected = self.checkpoint("cycle-000021", 69000)
        self.manifest = self.write_manifest(9, automatic_training=True)
        monitor = self.run / "learning/monitoring/rejected.json"
        monitor.parent.mkdir(parents=True, exist_ok=True)
        atomic(
            monitor,
            dict(
                schema="parallel.frozen_monitoring.v1",
                evaluated_step=69000,
                sequence_nll=1.39,
                ranking_bt=0.5225,
                ranking_listwise=2.25,
                runtime_huber=0.07,
            ),
        )
        self.journal = [
            journal_line(
                timestamp=100.0,
                event="diagnostic_accepted",
                checkpoint_sha256=self.accepted["sha256"],
                receipt="runs/x/learning/monitoring/accepted.json",
            ),
            journal_line(
                timestamp=200.0,
                event="diagnostic_rejected",
                checkpoint_sha256="b" * 64,
                receipt="runs/x/learning/monitoring/older.json",
            ),
            journal_line(
                timestamp=300.0,
                event="diagnostic_hold_acknowledged",
                acknowledgement=dict(nonce=self.nonce, checkpoint_sha256="b" * 64),
            ),
            journal_line(
                timestamp=350.0,
                event="stage_started",
                kind="train",
                command=["bin", self.rejected["parent"], "--checkpoint-steps"],
            ),
            journal_line(
                timestamp=400.0,
                event="diagnostic_rejected",
                checkpoint_sha256=self.rejected["sha256"],
                receipt="runs/x/learning/monitoring/rejected.json",
            ),
        ]
        self.write_journal()
        self.write_state(learning=self.rejected["path"], manifest_sha256=self.manifest["sha256"])

    def checkpoint(self, cycle, step):
        directory = self.run / ("learning/%s/train-2/checkpoints/step-%d" % (cycle, step))
        directory.mkdir(parents=True)
        (directory / "state.json").write_text(json.dumps(dict(step=step)))
        parent = self.run / ("learning/%s/train-2/checkpoints/step-%d" % (cycle, step - 3000))
        return dict(
            path=str(directory.relative_to(self.root)),
            sha256=sha(directory / "state.json"),
            parent=str(parent),
        )

    def write_manifest(self, version, automatic_training):
        path = self.run / ("recovery-manifest-v%d.json" % version)
        atomic(path, dict(automatic_training=automatic_training, version=version))
        return dict(path=path, sha256=sha(path))

    def write_journal(self):
        (self.run / "learning/journal.jsonl").write_text("".join(self.journal))

    def write_state(self, learning, manifest_sha256):
        atomic(
            self.run / "learning/state.json",
            dict(
                learning_checkpoint=learning,
                learning_corpus="runs/x/learning/snapshot-1/corpus",
                manifest_sha256=manifest_sha256,
                training_diagnostic=dict(consecutive_rejections=1, held=False, last_step=69000),
            ),
        )

    def readout(self):
        output = self.root / "decisions/readout.json"
        result = hr.main(
            [
                "--root",
                str(self.root),
                "--run",
                RUN,
                "--decision",
                str(self.decision),
                "--output",
                str(output),
            ]
        )
        return output, result

    def verify(self):
        return hr.main(
            [
                "--root",
                str(self.root),
                "--run",
                RUN,
                "--verify-rollback",
                str(self.root / "decisions/readout.json"),
                "--output",
                str(self.root / "decisions/status.json"),
            ]
        )

    def test_readout_names_the_last_accepted_checkpoint_before_release(self):
        output, result = self.readout()
        self.assertEqual((result["status"], result["outcome"]), ("decided", "rollback_required"))
        self.assertEqual(result["reparent_to"]["checkpoint_sha256"], self.accepted["sha256"])
        self.assertEqual(result["verdict"]["checkpoint_sha256"], self.rejected["sha256"])
        self.assertEqual(result["monitor"]["ranking_bt"], 0.5225)
        digest = sha(output)
        hr.main(
            [
                "--root",
                str(self.root),
                "--run",
                RUN,
                "--decision",
                str(self.decision),
                "--output",
                str(output),
            ]
        )
        self.assertEqual(sha(output), digest)  # immutable

    def test_pending_until_training_is_disabled_then_reparented(self):
        self.readout()
        status = self.verify()
        self.assertEqual(status["status"], "pending")
        self.assertTrue(status["learning_checkpoint"]["is_rejected"])
        self.assertEqual(
            (
                status["manifest"]["automatic_training"],
                status["deviations"],
                status["revisions_after_verdict"],
            ),
            (True, [], []),
        )
        self.journal.append(
            journal_line(
                timestamp=480.0,
                event="operator_pause_committed",
                reason="F11 first post-release monitor rejected",
                evidence="successor/decisions/pause/receipt.json",
                training_restart_authorized=False,
                control_sha256="e" * 64,
            )
        )
        self.write_journal()
        status = self.verify()
        self.assertEqual(
            (
                status["status"],
                status["controller_stops_after_verdict"][0]["event"],
                status["controller_stops_after_verdict"][0]["training_restart_authorized"],
            ),
            ("paused", "operator_pause_committed", False),
        )
        disabled = self.write_manifest(10, automatic_training=False)
        self.journal.append(
            journal_line(
                timestamp=500.0,
                event="manifest_revision_committed",
                new_manifest_sha256=disabled["sha256"],
                receipt="runs/x/review/manifest-migration-v10.json",
            )
        )
        self.write_journal()
        self.write_state(learning=self.rejected["path"], manifest_sha256=disabled["sha256"])
        status = self.verify()
        self.assertEqual(
            (
                status["status"],
                status["manifest"]["automatic_training"],
                len(status["revisions_after_verdict"]),
            ),
            ("training_disabled", False, 1),
        )
        self.assertEqual(status["manifest"]["path"], "runs/x/recovery-manifest-v10.json")
        self.assertEqual(status["manifest"]["automatic_training_source"], "manifest")
        # a revision whose manifest file is not locatable is judged by its migration receipt
        (self.run / "review").mkdir(exist_ok=True)
        atomic(
            self.run / "review/manifest-migration-v11.json",
            dict(
                schema="parallel.f11_manifest_migration.v1",
                training_disabled=True,
                learning_head_reparented=False,
                new_manifest_sha256="1" * 64,
            ),
        )
        self.journal.append(
            journal_line(
                timestamp=520.0,
                event="manifest_revision_committed",
                new_manifest_sha256="1" * 64,
                receipt="runs/x/review/manifest-migration-v11.json",
            )
        )
        self.write_journal()
        self.write_state(learning=self.rejected["path"], manifest_sha256="1" * 64)
        status = self.verify()
        self.assertEqual(
            (status["status"], status["manifest"]["path"], status["manifest"]["automatic_training"]),
            ("training_disabled", None, False),
        )
        self.assertTrue(status["manifest"]["automatic_training_source"].startswith("migration receipt"))
        self.assertEqual(
            [r.get("training_disabled") for r in status["revisions_after_verdict"]], [None, True]
        )
        self.write_state(learning=self.accepted["path"], manifest_sha256=disabled["sha256"])
        status = self.verify()
        self.assertEqual((status["status"], status["learning_checkpoint"]["is_target"]), ("reparented", True))
        self.assertEqual(
            json.loads((self.root / "decisions/status.json").read_bytes())["schema"], hr.VERIFICATION_SCHEMA
        )

    def test_a_block_from_the_rejected_lineage_after_the_verdict_is_a_deviation(self):
        self.readout()
        self.journal.append(
            journal_line(
                timestamp=450.0,
                event="stage_started",
                kind="train",
                command=[
                    "bin",
                    str(self.run / "learning/cycle-000021/train-2/checkpoints/step-69000"),
                    "--checkpoint-steps",
                ],
            )
        )
        self.write_journal()
        status = self.verify()
        self.assertEqual(len(status["deviations"]), 1)
        self.assertEqual(status["deviations"][0]["parent_sha256"], self.rejected["sha256"])
        self.assertIn("f11_deviation", status["deviations"][0]["rule"])
        self.assertEqual(
            len(status["train_stages_after_verdict"]), 1
        )  # the pre-verdict block 22 start is not counted
        self.journal.append(
            journal_line(
                timestamp=460.0,
                event="stage_started",
                kind="train",
                command=[
                    "bin",
                    str(self.run / "learning/cycle-000018/train-2/checkpoints/step-60000"),
                    "--checkpoint-steps",
                ],
            )
        )
        self.write_journal()
        status = self.verify()
        self.assertEqual(
            (len(status["deviations"]), len(status["train_stages_after_verdict"])), (1, 2)
        )  # a block from the target is not a deviation

    def test_refusals(self):
        self.journal[-1] = journal_line(
            timestamp=400.0,
            event="diagnostic_accepted",
            checkpoint_sha256=self.rejected["sha256"],
            receipt="runs/x/learning/monitoring/rejected.json",
        )
        self.write_journal()
        output, result = self.readout()
        self.assertEqual(result["outcome"], "continue")
        with self.assertRaisesRegex(RuntimeError, "nothing to verify"):
            self.verify()
        with self.assertRaisesRegex(RuntimeError, "--decision is required"):
            hr.main(["--root", str(self.root), "--run", RUN])


if __name__ == "__main__":
    unittest.main()

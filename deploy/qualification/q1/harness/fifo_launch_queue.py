"""FIFO launch unittest mixins, imported by ``test_fifo_launch``."""
from pathlib import Path
import fcntl
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fifo_launch as launch
import q1
import q1_cut


class FIFOQueueTests:
    def test_consumed_or_aliased_ledger_refuses_without_new_run(self):
        self.args.ledger.mkdir()
        for suffix in ("started", "finished"):
            marker = self.args.ledger / (
                self.plan["pins"]["bundle_sha256"] + "." + suffix + ".json"
            )
            marker.write_text("existing evidence")
            with self.assertRaisesRegex(ValueError, "consumed"):
                self.run_launch()
            self.assertEqual(marker.read_text(), "existing evidence")
            marker.unlink()
        alias = self.root / "alias"
        alias.symlink_to(self.args.ledger, target_is_directory=True)
        self.args.ledger = alias
        with self.assertRaisesRegex(ValueError, "canonical owner-signed"):
            self.admit()
        self.assertFalse(self.args.run_dir.exists())

    def test_fifo_waits_for_earlier_owner_without_pruning_any_other_ticket(self):
        fifo = launch.fifo_module(self.paths)
        earlier = fifo.enqueue_ticket(
            self.queue, pid=os.getpid(), wave="W1fixture", epoch_ns=1
        )
        dead = self.queue / "2-999999999-deadfixture"
        dead.write_text("retained dead ticket")
        notes = []

        def sleep(_seconds):
            notes.append("waited")
            if len(notes) == 2:
                # First callback is the zero courtesy delay. The next callback
                # proves that the earlier live owner prevented admission.
                self.assertEqual(launch.oldest_readonly(fifo, self.queue), earlier)
                earlier.unlink()  # Fixture owner releases its own earlier ticket.

        with launch.lease(
            self.paths, fifo, 1, lambda: 0, sleep=sleep, poll=0.01, backoff=0
        ):
            self.assertEqual(len(notes), 2)
            self.assertTrue(dead.exists())
        self.assertEqual(list(self.queue.iterdir()), [dead])

    def test_real_busy_lock_and_wait_cap_cannot_be_preempted(self):
        fifo = launch.fifo_module(self.paths)
        with self.lock.open("rb") as w1:
            fcntl.flock(w1, fcntl.LOCK_EX)
            with self.assertRaisesRegex(TimeoutError, "wait cap"):
                with launch.lease(
                    self.paths, fifo, 0.04, lambda: 0, poll=0.01, backoff=0
                ):
                    self.fail("preempted W1")
            with self.lock.open("rb") as probe:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assertEqual(list(self.queue.iterdir()), [])

    def test_missing_or_symlinked_lock_refuses_without_replacing_it(self):
        fifo = launch.fifo_module(self.paths)
        self.lock.unlink()
        with self.assertRaises(FileNotFoundError):
            with launch.lease(self.paths, fifo, 1, lambda: 0):
                self.fail("missing lock admitted")
        real = self.root / "real-lock"
        real.touch()
        self.lock.symlink_to(real)
        with self.assertRaises(OSError):
            with launch.lease(self.paths, fifo, 1, lambda: 0):
                self.fail("symlinked lock admitted")
        self.assertTrue(self.lock.is_symlink())
        self.assertEqual(list(self.queue.iterdir()), [])

    def test_changed_inputs_after_wait_refuse_and_keep_failure_receipt(self):
        checks = []

        def source_check(*_):
            checks.append(1)
            if len(checks) == 2:
                self.backend.write_text("changed while waiting")

        with self.assertRaisesRegex(ValueError, "changed while waiting"):
            launch.launch(
                self.args,
                self.paths,
                source_check=source_check,
                load=lambda: 0,
                lease_options={"poll": 0.01, "backoff": 0},
                grace=0.02,
            )
        receipt = q1.load(self.args.run_dir / "launch-receipt.json")
        self.assertEqual(receipt["last_phase"], "locked")
        self.assertFalse((self.args.run_dir / "run.log").exists())
        self.assertFalse(self.args.ledger.exists())
        self.assertEqual(list(self.queue.iterdir()), [])

    def test_positive_run_and_analysis_hold_lease_and_do_not_promote_fail(self):
        result = self.run_launch()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["q1_decision"], "FAIL")
        self.assertFalse(result["default_changed"])
        self.assertTrue((self.args.run_dir / "receipts.jsonl").exists())
        self.assertTrue((self.args.run_dir / "analysis.json").exists())
        self.assertEqual(list(self.queue.iterdir()), [])
        with self.assertRaisesRegex(ValueError, "consumed"):
            self.run_launch()

    def test_source_or_receipt_mutation_blocks_analysis(self):
        original = launch.run_owned

        def mutate(command, *args):
            code = original(command, *args)
            if "run" in command:
                (self.args.run_dir / "receipts.jsonl").write_text(
                    "modified after producer finish"
                )
            return code

        with patch.object(launch, "run_owned", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "ledger/receipt binding"):
                self.run_launch()
        self.assertFalse((self.args.run_dir / "analysis.json").exists())
        self.assertTrue(self.args.ledger.exists())

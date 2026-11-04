"""Private lock/queue and signing fixtures; no study, real data, GPU or live queue."""

from dataclasses import replace
import fcntl
import math
import os
from pathlib import Path
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
from test_q1 import qualification_plan
from fifo_launch_admission import FIFOAdmissionTests
from fifo_launch_queue import FIFOQueueTests
from fifo_launch_process import FIFOProcessTests


class FIFOTests(FIFOAdmissionTests, FIFOQueueTests, FIFOProcessTests, unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = self.root / "queue"
        self.queue.mkdir()
        self.lock = self.root / "qualification.lock"
        self.lock.touch()
        self.backend = self.root / "backend"
        self.backend.write_text("#!/bin/sh\nexit 99\n")
        self.backend.chmod(0o700)
        key = self.root / "fixture-key"
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True
        )
        signers = self.root / "fixture-signers"
        signers.write_text("jepson " + key.with_suffix(".pub").read_text())
        self.prereg = self.root / "fixture-prereg"
        self.prereg.write_text(
            "Owner allowed-signers SHA-256: " + q1.sha_file(signers) + "\n"
        )
        self.driver = self.root / "fixture-driver.py"
        self.driver.write_text(self.fake_driver())
        self.paths = replace(
            launch.Paths(),
            driver=self.driver,
            prereg=self.prereg,
            lock=self.lock,
            queue=self.queue,
        )
        self.plan = qualification_plan()
        self.plan.update(
            preregistration_sha256=q1.sha_file(self.prereg),
            backend_sha256=q1.sha_file(self.backend),
            ledger_root=str(self.root / "ledger"),
        )
        self.plan["pins"]["dope_sha256"] = launch.DOPE
        self.planfile = self.root / "plan.json"
        self.planfile.write_bytes(q1.canonical(self.plan))
        message = self.root / "approval"
        message.write_bytes(q1.approval_payload(self.plan))
        subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "sign",
                "-f",
                str(key),
                "-n",
                "jopedime-q1",
                str(message),
            ],
            check=True,
            capture_output=True,
        )
        self.args = SimpleNamespace(
            plan=self.planfile,
            backend=self.backend,
            signature=message.with_suffix(".sig"),
            allowed_signers=signers,
            ledger=Path(self.plan["ledger_root"]),
            run_dir=self.root / "run",
            wait_seconds=2.0,
            wall_seconds=3.0,
            authorized_launch=True,
        )

    def fake_driver(self):
        return f"""import fcntl, hashlib, json, sys
from pathlib import Path
args = sys.argv[1:]
def value(flag): return Path(args[args.index(flag)+1])
plan = json.loads(value('--plan').read_text())
digest = hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
with Path({str(self.lock)!r}).open('rb') as other:
    try:
        fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        pass
    else:
        # jankurai:allow HLT-008-FALSE-GREEN-RISK reason="legitimate exit/skip check" expires=2026-12-31
        raise SystemExit('driver ran without the qualification lease')
out = value('--out')
if args[0]=='run':
    ledger=value('--ledger');ledger.mkdir()
    prefix=plan['pins']['bundle_sha256']
    (ledger/(prefix+'.started.json')).write_text(json.dumps({{'plan_sha256':digest,'status':'started_outcome_unknown'}}))
    out.write_text('{{"fixture_not_numerical":true}}\\n')
    finished={{'plan_sha256':digest,'status':'complete','failed_runs':0,'receipts_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}}
    (ledger/(prefix+'.finished.json')).write_text(json.dumps(finished))
else:
    out.write_text(json.dumps({{'schema':'jope.q1-analysis/v1','plan_sha256':digest,'decision':'FAIL','fixture_not_a_study':True}}))
"""

    def admit(self, **kwargs):
        return launch.admit(
            self.args,
            self.paths,
            source_check=lambda *_: None,
            load=lambda: 0,
            **kwargs,
        )

    def run_launch(self, **kwargs):
        return launch.launch(
            self.args,
            self.paths,
            source_check=lambda *_: None,
            load=lambda: 0,
            lease_options={"poll": 0.01, "backoff": 0},
            grace=0.02,
            **kwargs,
        )

    def resign(self):
        self.planfile.write_bytes(q1.canonical(self.plan))
        message = self.root / "changed-approval"
        message.write_bytes(q1.approval_payload(self.plan))
        subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "sign",
                "-f",
                str(self.root / "fixture-key"),
                "-n",
                "jopedime-q1",
                str(message),
            ],
            check=True,
            capture_output=True,
        )
        self.args.signature = message.with_suffix(".sig")



if __name__ == "__main__":
    unittest.main()

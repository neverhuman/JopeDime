#!/usr/bin/env python3
"""Scaled regression: DL offline shards run concurrently and share one deadline with --combine."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().with_name("dl-offline-group.sh")
# A shell fake, not Python. Import startup under load consumed the old sub-second
# slack and the group reported "passed before" instead of "expired during".
FAKE = """#!/bin/bash
set -euo pipefail
log=$FAKE_LOG
step=""
output=""
args=("$@")
i=0
while (( i < ${#args[@]} )); do
  case "${args[$i]}" in
    --combine) step=combine ;;
    --shard) i=$((i + 1)); step="shard ${args[$i]}" ;;
    --output) i=$((i + 1)); output="${args[$i]}" ;;
  esac
  i=$((i + 1))
done
printf 'start %s %s\\n' "$step" "$output" >> "$log"
if [[ $step == combine ]]; then sleep "$FAKE_COMBINE_S"; else sleep "$FAKE_SHARD_S"; fi
printf 'end %s\\n' "$step" >> "$log"
"""


class CommonDeadline(unittest.TestCase):
    def run_group(self, budget, shard_s, combine_s):
        tmp = Path(tempfile.mkdtemp(prefix="dl-offline-group-"))
        fake = tmp / "fake_runner.sh"
        fake.write_text(FAKE)
        log = tmp / "steps.log"
        env = dict(os.environ, DL_OFFLINE_RUNNER=f"bash {fake}", FAKE_LOG=str(log),
                   FAKE_SHARD_S=str(shard_s), FAKE_COMBINE_S=str(combine_s))
        r = subprocess.run(["bash", str(SCRIPT), str(budget), str(tmp / "out")], env=env, capture_output=True, text=True, timeout=60)
        steps = log.read_text().splitlines() if log.exists() else []
        return r, steps, tmp / "out"

    def test_parallel_shards_fit_when_their_serial_sum_does_not(self):
        r, steps, out = self.run_group(5, 2.0, 0.1)
        self.assertEqual(r.returncode, 0, r.stderr)
        starts = [s for s in steps if s.startswith("start")]
        self.assertEqual(3, sum(s.startswith("start shard") for s in starts), steps)
        self.assertEqual(3, sum(s.startswith("end shard") for s in steps), steps)
        self.assertEqual({f"start shard {i}/3 {out}/shard-{i}" for i in (1, 2, 3)}, set(starts[:3]))
        self.assertTrue(starts[3].startswith(f"start combine {out}"), starts)

    def test_combine_uses_only_the_remaining_common_deadline(self):
        # Parallel shards take 2 s; combine starts with 2 s left and is killed at the deadline.
        r, steps, _ = self.run_group(4, 2.0, 3.0)
        self.assertEqual(r.returncode, 124, r.stderr)
        self.assertIn("common 4s deadline expired during --combine", r.stderr)
        self.assertEqual(3, sum(s.startswith("end shard") for s in steps), steps)
        self.assertNotIn("end combine", steps)

    def test_a_bad_budget_is_refused(self):
        r, steps, _ = self.run_group("0", 0.1, 0.1)
        self.assertEqual(r.returncode, 2)
        self.assertEqual(steps, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

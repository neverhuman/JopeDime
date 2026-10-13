#!/usr/bin/env python3
"""Scaled regression for deploy/ci/dl-offline-group.sh: the strict DL offline shard group (3 shards + --combine) shares ONE
deadline. A fake runner (DL_OFFLINE_RUNNER) sleeps per step, with seconds standing in for the real 900 s budget:
  - each step fits the budget on its own but the group total does not  -> rejected (exit 124), later steps never start;
  - the shards fit but combine pushes the total past the deadline       -> rejected while combine runs;
  - the whole group fits                                                 -> passes, steps in order with the pinned outputs."""
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

    def test_each_step_fits_but_the_total_exceeds_the_deadline(self):
        # 3 s fits in the 8 s budget. Two shards leave 2 s, so shard 3 starts and the deadline fires during it.
        r, steps, _ = self.run_group(8, 3.0, 0.1)
        self.assertEqual(r.returncode, 124, r.stderr)
        self.assertIn("common 8s deadline expired during: --shard 3/3", r.stderr)
        self.assertIn("start shard 3/3", " ".join(steps))
        self.assertNotIn("end shard 3/3", steps)
        self.assertFalse(any(s.startswith("start combine") for s in steps), steps)

    def test_combine_runs_inside_the_same_deadline(self):
        # three 1.5 s shards fit in 8 s; combine at 4 s starts with time left and is killed during it.
        r, steps, _ = self.run_group(8, 1.5, 4.0)
        self.assertEqual(r.returncode, 124, r.stderr)
        self.assertIn("common 8s deadline expired during: --combine", r.stderr)
        self.assertIn("end shard 3/3", steps)
        self.assertNotIn("end combine", steps)

    def test_the_group_within_the_deadline_passes_in_order(self):
        r, steps, out = self.run_group(3, 0.1, 0.1)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([s for s in steps if s.startswith("start")],
                         [f"start shard {i}/3 {out}/shard-{i}" for i in (1, 2, 3)] + [f"start combine {out}"])
        self.assertIn("finished within the common 3s deadline", r.stdout)

    def test_a_bad_budget_is_refused(self):
        r, steps, _ = self.run_group("0", 0.1, 0.1)
        self.assertEqual(r.returncode, 2)
        self.assertEqual(steps, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

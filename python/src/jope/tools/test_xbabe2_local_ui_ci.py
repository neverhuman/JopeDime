import os
import subprocess
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "xbabe2-local-ui-ci.sh"


class Xbabe2LocalUiCiTests(unittest.TestCase):
    def test_refuses_github_ci(self):
        env = os.environ.copy()
        env["XBABE2_LOCAL_UI_CI"] = "1"
        env["CI"] = "true"
        env["GITHUB_ACTIONS"] = "true"
        proc = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("refused", (proc.stderr + proc.stdout).lower())

    def test_refuses_without_explicit_flag(self):
        env = {k: v for k, v in os.environ.items() if k not in {"XBABE2_LOCAL_UI_CI", "CI", "GITHUB_ACTIONS"}}
        proc = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()

"""Offline proofs for dltrain.common (PLAN §5 P0 step 3)."""
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from dltrain import common


class CommonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_canonical_is_sorted_compact_and_refuses_nan(self):
        self.assertEqual(common.canonical({"b": 1, "a": [1, 2]}), b'{"a":[1,2],"b":1}')
        with self.assertRaises(ValueError):
            common.canonical({"x": float("nan")})

    def test_atomic_write_leaves_no_temp_files(self):
        out = common.atomic_write(self.d / "f.json", b"abc")
        self.assertEqual((self.d / "f.json").read_bytes(), b"abc")
        self.assertEqual(out["bytes"], 3)
        self.assertEqual([p.name for p in self.d.iterdir()], ["f.json"])

    def test_immutable_write_is_create_only(self):
        p = self.d / "r.json"
        self.assertFalse(common.immutable_write(p, b"one")["existed"])
        self.assertTrue(common.immutable_write(p, b"one")["existed"])
        with self.assertRaises(common.ImmutableConflict):
            common.immutable_write(p, b"two")

    def test_receipts_refuse_secret_material(self):
        for bad in ({"env": "AWS_SECRET_ACCESS_KEY=x"}, {"k": "AKIA" + "A" * 16}, {"argv": ["--token", "x"]},
                    {"h": "Bearer " + "a" * 24}, {"t": "AWS_SESSION_TOKEN"}, {"low": "aws_secret_access_key = abc"},
                    {"SessionToken": "x"}, {"multi": "line1\nAKIA" + "B" * 16}, {"tab": "id:\tASIA" + "C" * 16},
                    {"url": "https://u:p@example.com/x"}, {"sig": "...&X-Amz-Signature=abc"}, {"flag": "--aws-secret-access-key=zz"}):
            with self.assertRaises(common.SecretRefused):
                common.write_receipt(self.d / "bad.json", bad)
        self.assertFalse((self.d / "bad.json").exists())
        res = common.write_receipt(self.d / "ok.json", {"arn": "arn:aws:iam::123456789012:user/x", "n": 1, "argv": ["p", "--token", "<redacted>"],
                                                        "sha": "a" * 64, "vid": "QjKmAjvVgrDoGZnksp7scksW8n3sSuLJ", "etag": '"' + "f" * 32 + '"'})
        self.assertEqual(json.loads((self.d / "ok.json").read_text())["n"], 1)
        self.assertEqual(res["bytes"], (self.d / "ok.json").stat().st_size)

    def test_redact_argv(self):
        self.assertEqual(common.redact_argv(["x", "--token", "abc", "--key=zz", "--other", "v"]),
                         ["x", "--token", "<redacted>", "--key=<redacted>", "--other", "v"])
        self.assertEqual(common.redact_argv(["--session-token", "s", "--aws-secret-access-key=k", "--receipt-dir", "d"]),
                         ["--session-token", "<redacted>", "--aws-secret-access-key=<redacted>", "--receipt-dir", "d"])

    def test_copy_verified(self):
        src = self.d / "bin"
        src.write_bytes(b"\x7fELF demo")
        sha = common.sha256_file(src)
        out = common.copy_verified(src, self.d / "out", sha)
        self.assertEqual(common.sha256_file(out["dst"]), sha)
        with self.assertRaises(RuntimeError):
            common.copy_verified(src, self.d / "out2", "0" * 64)

    def test_utc_iso_format(self):
        self.assertRegex(common.utc_iso(1791175128040293), r"^2026-10-05T04:38:48\.040293Z$")


if __name__ == "__main__":
    unittest.main()

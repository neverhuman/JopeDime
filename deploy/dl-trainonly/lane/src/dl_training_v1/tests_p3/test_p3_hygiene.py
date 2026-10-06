"""Lane hygiene for P3: the write guard is live (tests write nothing outside TMPDIR), forbidden imports are absent, and
offline commands never ask for an S3 client."""
import ast
import os
from pathlib import Path
from unittest import mock

from dltrain import holdout, s3io
from tests_p3 import fixtures_p3 as fx

FORBIDDEN = {"jope", "gp_worker", "local_gateway", "reuse_cache", "intake_registry", "family_protocol"}


class WriteGuard(fx.P3Case):
    def test_classifier_without_writing(self):
        inside = os.path.join(fx.ALLOWED_ROOT, "x", "y.json")
        outside = "/home/ubuntu/JopeDime/never-written-by-p3-tests.json"
        w = os.O_WRONLY | os.O_CREAT
        self.assertEqual(fx.write_target("open", (outside, "wb", w)), outside)
        self.assertEqual(fx.write_target("open", (outside, None, os.O_RDWR | os.O_CREAT | os.O_EXCL)), outside)
        self.assertEqual(fx.write_target("open", (outside, "a", 0)), outside)
        self.assertIsNone(fx.write_target("open", (outside, "rb", os.O_RDONLY)))
        self.assertIsNone(fx.write_target("open", (outside, None, os.O_RDONLY)))
        self.assertIsNone(fx.write_target("open", (3, "wb", w)))
        self.assertTrue(fx.under_root(fx.write_target("open", (inside, "w", w))))
        self.assertEqual(fx.write_target("os.rename", (inside, outside, -1, -1)), outside)
        self.assertEqual(fx.write_target("os.mkdir", (outside, 0o755, -1)), outside)
        self.assertEqual(fx.write_target("os.chmod", (outside, 0o444, -1)), outside)
        self.assertEqual(fx.write_target("shutil.copyfile", (inside, outside)), outside)
        self.assertIsNone(fx.write_target("sqlite3.connect", (f"file:{outside}?mode=ro",)))
        self.assertEqual(fx.write_target("sqlite3.connect", (outside,)), outside)
        self.assertIsNone(fx.write_target("os.listdir", (outside,)))
        self.assertFalse(fx.under_root(outside))
        self.assertFalse(fx.under_root(fx.ALLOWED_ROOT + "-sibling/x"))

    def test_the_audit_hook_is_live(self):
        allowed = self.tmp / "allowed"
        allowed.mkdir()
        with mock.patch.object(fx, "ALLOWED_ROOT", os.path.realpath(allowed)):
            (self.tmp / "sibling.txt").write_bytes(b"x")       # under TMPDIR, outside the narrowed root
            (allowed / "inside.txt").write_bytes(b"y")
            seen = list(fx._GUARD["violations"])
        fx._GUARD["violations"].clear()
        self.assertEqual([t for _, t in seen], [os.path.realpath(self.tmp / "sibling.txt")])

    def test_tmpdir_is_the_lane_tmp(self):
        self.assertEqual(fx.ALLOWED_ROOT, os.path.realpath(os.environ["TMPDIR"]))
        self.assertFalse(fx.ALLOWED_ROOT.startswith("/tmp"))
        self.assertTrue(str(self.tmp.resolve()).startswith(fx.ALLOWED_ROOT + os.sep))


class ModuleHygiene(fx.P3Case):
    def test_no_forbidden_imports_and_no_top_level_s3(self):
        src = Path(holdout.__file__).read_text()
        tree = ast.parse(src)
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names |= {(node.module or "").split(".")[0]} | {a.name for a in node.names}
        self.assertFalse(names & FORBIDDEN, names & FORBIDDEN)
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                top |= {f"{node.module}.{a.name}" for a in node.names}
        self.assertFalse({"boto3", "botocore", "dltrain.s3io"} & top, top)
        self.assertNotIn("jopedime-regression-research-2k-tabpfn35-unseen-20261001", src)

    def test_offline_commands_never_request_an_s3_client(self):
        inputs = fx.write_inputs(self.tmp / "in")
        out, bucket = fx.pipeline(self, inputs)        # freeze and verify ran with the real factory patched to raise
        calls_after_commit = list(bucket.calls)
        code, s = fx.run_cli(fx.freeze_args(inputs, out))
        self.assertEqual(code, 0, s)
        code, s = fx.run_cli(fx.verify_args(inputs, out))
        self.assertEqual(code, 0, s)
        paths = fx.closure_inputs(self.tmp / "closure-in", inputs)
        code, s = fx.run_cli(fx.closure_args(inputs, out, paths))
        self.assertEqual(code, 0, s)
        self.assertEqual(bucket.calls, calls_after_commit)


class CrashAtomicWrites(fx.P3Case):
    """LP-6: create-only files appear only complete; a crash leaves nothing, so a rerun never meets a torn seed."""

    def test_a_crash_before_the_link_leaves_neither_a_torn_file_nor_a_temp_file(self):
        p = self.tmp / "d" / "seed.hex"
        for target, err in (("link", OSError("simulated crash before link")), ("fsync", OSError("simulated crash in fsync"))):
            with mock.patch.object(holdout.os, target, side_effect=err):
                with self.assertRaises(OSError):
                    holdout._create_file(p, b"a" * 64 + b"\n")
            self.assertEqual(list(p.parent.iterdir()), [], target)
        w = holdout._create_file(p, b"a" * 64 + b"\n")
        self.assertEqual((w["existed"], p.read_bytes(), oct(p.stat().st_mode & 0o777)), (False, b"a" * 64 + b"\n", "0o444"))
        self.assertTrue(holdout._create_file(p, b"a" * 64 + b"\n")["existed"])
        with self.assertRaises(holdout.common.ImmutableConflict):
            holdout._create_file(p, b"b" * 64 + b"\n")
        self.assertEqual([x.name for x in p.parent.iterdir()], ["seed.hex"])

    def test_commit_crash_while_writing_the_seed_reruns_to_the_same_seed(self):
        inputs = fx.write_inputs(self.tmp / "in")
        bucket = fx.registry_bucket(inputs)
        out = self.tmp / "holdout"
        with mock.patch.object(holdout.os, "link", side_effect=OSError("simulated crash")):
            code, s = fx.run_cli(fx.commit_args(inputs, out), clients=fx.fake_clients(bucket))
        self.assertEqual(code, 3, s)
        self.assertEqual(sorted(x.name for x in out.iterdir()), ["receipts"])     # no torn seed.hex, no temp file
        code, s = fx.run_cli(fx.commit_args(inputs, out), clients=fx.fake_clients(bucket))
        self.assertEqual(code, 0, s)
        self.assertEqual(fx.receipt(s)["s3_outcome"], "already_present_equal_bytes")
        vid, body = bucket.current(s3io.PREFIX_KEY + holdout.COMMITMENT_REL)
        self.assertEqual(holdout.read_seed(out / "seed.hex"), holdout.derive_seed(body, vid))
        self.assertEqual(len(bucket.versions(s3io.PREFIX_KEY + holdout.COMMITMENT_REL)), 1)

    def test_a_torn_seed_from_an_older_writer_is_never_overwritten(self):
        inputs = fx.write_inputs(self.tmp / "in")
        bucket = fx.registry_bucket(inputs)
        out = self.tmp / "holdout"
        out.mkdir()
        (out / "seed.hex").write_bytes(b"")
        code, s = fx.run_cli(fx.commit_args(inputs, out), clients=fx.fake_clients(bucket))
        self.assertEqual(code, 2, s)
        self.assertEqual(fx.receipt(s)["error"]["type"], "ImmutableConflict")
        self.assertEqual((out / "seed.hex").read_bytes(), b"")

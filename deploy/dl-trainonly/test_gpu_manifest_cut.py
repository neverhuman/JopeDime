"""Tests for gpu_manifest_cut.source_bound_cut_of (owner 16:28 MT item 3): only the cut moves, only from an identity-equal source."""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("gmc", Path(__file__).absolute().with_name("gpu_manifest_cut.py"))
gmc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gmc)
M = gmc.load()

SPLIT = {"schema": "rowsel-raw-split-v2", "sha256": "a" * 64}
SRC = {"view_id": "v1", "context_sha256": "c" * 64, "rowsel": {"identity": "i", "split": SPLIT}}


def gpu_ctx(**kw):
    ctx = {"view_id": "v1", "context_sha256": "c" * 64, "rowsel": {"identity": "i", "split_schema": "rowsel-raw-split-v2"},
           "gpu": {"components": {}}}
    ctx.update(kw)
    return ctx


class SourceBoundCut(unittest.TestCase):
    def setUp(self):
        self.cut = gmc.source_bound_cut_of(M.cut_of, {"v1": SRC})

    def test_frozen_tool_alone_finds_no_cut(self):          # the reproduced blocker
        self.assertIsNone(M.cut_of(gpu_ctx()))

    def test_identity_equal_source_gives_its_split_cut(self):
        self.assertEqual(self.cut(gpu_ctx()), "trainonly-split-v2:" + "a" * 64)

    def test_context_sha_mismatch_stays_none(self):           # must-fail: a different context never inherits a cut
        self.assertIsNone(self.cut(gpu_ctx(context_sha256="d" * 64)))

    def test_missing_source_stays_none(self):
        self.assertIsNone(self.cut(gpu_ctx(view_id="v2")))

    def test_non_gpu_context_unchanged(self):                 # CPU contexts keep the frozen behavior exactly
        cpu = {"view_id": "v1", "context_sha256": "c" * 64, "rowsel": {"identity": "i"}}
        self.assertIsNone(self.cut(cpu))
        self.assertEqual(self.cut(dict(cpu, rowsel={"split": SPLIT})), M.cut_of(dict(cpu, rowsel={"split": SPLIT})))

    def test_own_cut_wins(self):
        own = gpu_ctx(cut={"cut_id": "trainonly-split-v2g:" + "b" * 64})
        self.assertEqual(self.cut(own), "trainonly-split-v2g:" + "b" * 64)

    def test_source_without_split_stays_none(self):
        cut = gmc.source_bound_cut_of(M.cut_of, {"v1": dict(SRC, rowsel={"identity": "i"})})
        self.assertIsNone(cut(gpu_ctx()))

    def test_only_split_identity_is_used(self):               # labels/rows in the source are never read: result depends on split only
        noisy = dict(SRC, inner_labels_sha256="x" * 64, n_train=10, outer_test_indices_sha256="y" * 64)
        cut = gmc.source_bound_cut_of(M.cut_of, {"v1": noisy})
        self.assertEqual(cut(gpu_ctx()), "trainonly-split-v2:" + "a" * 64)

    def test_local_dir_refused_with_publish(self):
        with self.assertRaises(SystemExit):
            gmc.main(["--build", "x", "--name", "n", "--source", "s", "--summary", "p#h", "--publish", "--local-dir", "/nonexistent"])


if __name__ == "__main__":
    unittest.main()

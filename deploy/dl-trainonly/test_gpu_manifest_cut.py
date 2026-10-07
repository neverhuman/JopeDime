"""Tests for gpu_manifest_cut (owner 16:28 MT item 3): only the cut moves, only from an identity-equal source with a valid
target-free split digest, and the executed adapter is bound in the manifest. Synthetic trees, fake ledger and fake bucket only."""

import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE / "lane/src/dl_training_v1"))
spec = importlib.util.spec_from_file_location("gmc", HERE / "gpu_manifest_cut.py")
gmc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gmc)
M = gmc.load(gmc.TOOL, "v2_rowsel_raw_manifest")
from dltrain import rowsel_raw_split_v2 as split  # noqa: E402
from tests.test_v2_publish import jtt_gz, make_context, vp  # noqa: E402

SPLIT = {"schema": "rowsel-raw-split-v2", "sha256": "a" * 64}
SRC = {
    "view_id": "v1",
    "context_sha256": "c" * 64,
    "rowsel": {"identity": "i", "split": SPLIT},
}
CANARY = ("OUTER_TEST_LABEL_CANARY_314159_" + "Z" * 64)[:64]


def gpu_ctx(**kw):
    ctx = {
        "view_id": "v1",
        "context_sha256": "c" * 64,
        "rowsel": {"identity": "i", "split_schema": "rowsel-raw-split-v2"},
        "gpu": {"components": {}},
    }
    ctx.update(kw)
    return ctx


class SourceBoundCut(unittest.TestCase):
    def setUp(self):
        self.cut = gmc.source_bound_cut_of(M.cut_of, {"v1": SRC})

    def test_frozen_tool_alone_finds_no_cut(self):  # the reproduced blocker
        self.assertIsNone(M.cut_of(gpu_ctx()))

    def test_identity_equal_source_gives_its_split_cut(self):
        self.assertEqual(self.cut(gpu_ctx()), "trainonly-split-v2:" + "a" * 64)

    def test_context_sha_mismatch_stays_none(
        self,
    ):  # a different context never inherits a cut
        self.assertIsNone(self.cut(gpu_ctx(context_sha256="d" * 64)))

    def test_missing_source_stays_none(self):
        self.assertIsNone(self.cut(gpu_ctx(view_id="v2")))

    def test_non_gpu_context_unchanged(
        self,
    ):  # CPU contexts keep the frozen behavior exactly
        cpu = {"view_id": "v1", "context_sha256": "c" * 64, "rowsel": {"identity": "i"}}
        self.assertIsNone(self.cut(cpu))
        self.assertEqual(
            self.cut(dict(cpu, rowsel={"split": SPLIT})),
            M.cut_of(dict(cpu, rowsel={"split": SPLIT})),
        )

    def test_own_cut_wins(self):
        own = gpu_ctx(cut={"cut_id": "trainonly-split-v2g:" + "b" * 64})
        self.assertEqual(self.cut(own), "trainonly-split-v2g:" + "b" * 64)

    def test_source_without_split_is_refused(self):
        cut = gmc.source_bound_cut_of(
            M.cut_of, {"v1": dict(SRC, rowsel={"identity": "i"})}
        )
        with self.assertRaises(SystemExit):
            cut(gpu_ctx())

    def test_malformed_source_digests_are_refused(
        self,
    ):  # canary, uppercase, short, wrong schema
        for bad in (
            {"schema": "rowsel-raw-split-v2", "sha256": CANARY},
            {"schema": "rowsel-raw-split-v2", "sha256": "A" * 64},
            {"schema": "rowsel-raw-split-v2", "sha256": "a" * 63},
            {"schema": "rowsel-split-v2-groups", "sha256": "a" * 64},
        ):
            cut = gmc.source_bound_cut_of(
                M.cut_of, {"v1": dict(SRC, rowsel={"split": bad})}
            )
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                cut(gpu_ctx())

    def test_only_split_identity_is_used(
        self,
    ):  # labels/rows in the source never change the result
        noisy = dict(
            SRC,
            inner_labels_sha256="x" * 64,
            n_train=10,
            outer_test_indices_sha256="y" * 64,
            outer_test_labels=[CANARY],
        )
        cut = gmc.source_bound_cut_of(M.cut_of, {"v1": noisy})
        self.assertEqual(cut(gpu_ctx()), "trainonly-split-v2:" + "a" * 64)

    def test_local_dir_refused_with_publish(self):
        with self.assertRaises(SystemExit):
            gmc.main(
                [
                    "--build",
                    "x",
                    "--name",
                    "n",
                    "--source",
                    "s",
                    "--summary",
                    "p#h",
                    "--publish",
                    "--local-dir",
                    "/nonexistent",
                ]
            )

    def test_adapter_identity_tracks_adapter_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "gpu_manifest_cut.py"
            p.write_bytes((HERE / "gpu_manifest_cut.py").read_bytes())
            a = gmc.adapter_identity(p)
            p.write_bytes(p.read_bytes() + b"# changed\n")
            b = gmc.adapter_identity(p)
        self.assertEqual(a["sha256"], gmc.adapter_identity()["sha256"])
        self.assertNotEqual(a["sha256"], b["sha256"])
        self.assertEqual(a["frozen_tool_sha256"], gmc.TOOL[1])


class ReceivingPath(unittest.TestCase):
    """gmc.main end to end on synthetic source and GPU trees (fake verified ledger; fake bucket for the publish case)."""

    DS = "0123456789abcdef"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gpucut-"))
        self.digest = split.split_sha256(
            self.DS, {0: [0, 5], 1: [1, 6], 2: [2, 7], 3: [3, 8], 4: [4, 9]}
        )

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def receive(self, name, change_source=None, publish_clients=None):
        base = self.tmp / name
        source, build = base / "source", base / "gpu"
        schema = "jope.dl-v2-rowsel-raw-context/v1"
        rep = {
            "level": "dataset",
            "count": 1,
            "new_contexts": 1,
            "contexts": [],
            "fold_correspondence": "none: target-free new split",
        }
        sd = make_context(
            source / "shard-0",
            self.DS,
            files={"encoder.jtt1.gz": jtt_gz(n_test=0)},
            schema=schema,
            values_mode="L7a",
            epoch0_replaced=rep,
            rowsel={
                "identity": "synthetic",
                "split": {"schema": split.SPLIT_SCHEMA, "sha256": self.digest},
            },
        )
        sc = json.loads((sd / "context.json").read_bytes())
        make_context(
            build / "shard-0",
            self.DS,
            files={"hyperion.npy": b"synthetic-train-only-vector"},
            schema=schema,
            values_mode="L7a",
            epoch0_replaced=rep,
            rowsel={"identity": "synthetic", "split_schema": split.SPLIT_SCHEMA},
            gpu={
                "input": {
                    "encoder_jtt1_sha256": sc["files"]["encoder.jtt1.gz"]["sha256"]
                },
                "components": {},
            },
        )
        if change_source:
            change_source(sc)
            (sd / "context.json").write_text(json.dumps(sc, sort_keys=True) + "\n")
        plans, refused = vp.scan_build(build)
        self.assertFalse(refused)
        rows = []
        for o in plans[0]["objects"]:
            common = {
                "schema": vp.LEDGER_SCHEMA,
                "outcome": "created",
                "utc_us": 1,
                "view_id": plans[0]["view_id"],
            }
            rows += [
                dict(
                    common,
                    event="verified",
                    kind="object",
                    key=o["key"],
                    sha256=o["sha256"],
                    bytes=o["bytes"],
                    version_id="FAKE-OBJECT",
                ),
                dict(
                    common,
                    event="verified",
                    kind="receipt",
                    key=o["receipt_key"],
                    sha256="f" * 64,
                    bytes=1,
                    version_id="FAKE-RECEIPT",
                ),
            ]
        ledger = base / "ledger.jsonl"
        ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary = base / "summary.json"
        summary.write_text("{}\n")
        args = [
            "--build",
            str(build),
            "--name",
            name,
            "--source",
            str(source),
            "--summary",
            f"{summary}#{gmc.sha(summary.read_bytes())}",
            "--ledger",
            str(ledger),
            "--require-cut",
        ]
        args += ["--publish"] if publish_clients else ["--local-dir", str(base / "out")]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = gmc.main(
                args,
                clients=publish_clients,
                out_root=(base / "out") if publish_clients else None,
            )
        return rc, json.loads(buf.getvalue().strip().splitlines()[-1]), base

    def test_authorized_target_free_cut_and_adapter_bound(self):
        rc, out, _ = self.receive("authorized")
        doc = json.loads(Path(out["local"]).read_bytes())
        self.assertEqual(rc, 0)
        self.assertEqual(
            doc["contexts"][0]["cut_id"], "trainonly-split-v2:" + self.digest
        )
        self.assertEqual(doc["code_sha256"], gmc.TOOL[1])  # frozen hash preserved
        self.assertEqual(doc["cut_adapter"]["sha256"], gmc.adapter_identity()["sha256"])
        self.assertEqual(
            doc["cut_adapter"]["inherited_cut_views"], [doc["contexts"][0]["view_id"]]
        )
        self.assertEqual(out["sha256"], gmc.sha(Path(out["local"]).read_bytes()))

    def test_label_canary_in_source_split_is_refused_before_any_manifest(self):
        with self.assertRaises(SystemExit):
            self.receive(
                "canary",
                change_source=lambda c: c["rowsel"]["split"].update(sha256=CANARY),
            )
        self.assertFalse((self.tmp / "canary" / "out").exists())

    def test_invalid_digests_are_refused(self):
        for i, bad in enumerate(("A" * 64, "a" * 63, "g" * 64)):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                self.receive(
                    f"bad{i}",
                    change_source=lambda c, b=bad: c["rowsel"]["split"].update(
                        sha256=b
                    ),
                )

    def test_identity_mismatches_are_refused(self):
        changes = {
            "ctx": lambda c: c.update(context_sha256="d" * 64),
            "encoder": lambda c: c["files"]["encoder.jtt1.gz"].update(sha256="e" * 64),
            "rights": lambda c: c.update(rights="research-only"),
        }
        for n, ch in changes.items():
            with self.subTest(n), self.assertRaises(SystemExit):
                self.receive(f"mismatch-{n}", change_source=ch)

    def test_outer_test_fields_in_source_are_ignored(self):
        _, a, _ = self.receive("plain")
        _, b, _ = self.receive(
            "noisy",
            change_source=lambda c: c.update(
                outer_test_labels=[CANARY], outer_test_targets=[CANARY]
            ),
        )
        da, db = (json.loads(Path(x["local"]).read_bytes()) for x in (a, b))
        self.assertEqual(da["contexts"][0]["cut_id"], db["contexts"][0]["cut_id"])
        self.assertNotIn(CANARY.encode(), Path(b["local"]).read_bytes())

    def test_publish_uses_frozen_transport_create_only(self):
        bucket = vp.FakeBucket()
        writer, reader, _ = vp.fake_clients(self.tmp / "budget.jsonl", bucket)
        rc, out, base = self.receive("published", publish_clients=(writer, reader))
        self.assertEqual(rc, 0)
        pub = out["published"]
        self.assertEqual(
            pub["key"],
            f"{vp.s3io.V2_PREFIX_KEY}manifests/published/{out['sha256'][:12]}.json",
        )
        self.assertEqual(pub["sha256"], out["sha256"])
        self.assertTrue(pub["version_id"])


if __name__ == "__main__":
    unittest.main()

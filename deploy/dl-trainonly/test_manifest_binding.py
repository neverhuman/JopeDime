"""Tests for manifest_binding (red-team 2026-10-07): each of the 20 faults that escaped every publish gate is reproduced against the
#239 adapter (it still writes a manifest) and refused by the binding adapter before any manifest file; clean trees of the five
published tree kinds pass and the manifest binds the executed adapter. Synthetic trees, fake ledger and fake bucket only."""

import contextlib
import gzip
import importlib.util
import io
import json
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE / "lane/src/dl_training_v1"))
spec = importlib.util.spec_from_file_location("manifest_binding", HERE / "manifest_binding.py")
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)
gmc = mb.load_cut_adapter()
from dltrain import jtt1  # noqa: E402
from tests.test_v2_publish import jtt_gz, make_context, vp  # noqa: E402

DS, OTHER = "0123456789abcdef", "fedcba9876543210"
RAW, ROWSEL = "jope.dl-v2-rowsel-raw-context/v1", "jope.dl-v2-rowsel-context/v2"
V2G, V2 = "rowsel-split-v2-groups", "rowsel-raw-split-v2"
# tree kind -> (context schema, values mode, split schema, cut prefix, research rights, GPU)
KINDS = {
    "rawgroup_cpu_raw": (RAW, "L7a", V2G, "trainonly-split-v2g:", False, False),
    "rawgroup_cpu_nonnumeric": (RAW, "L7a", V2G, "trainonly-split-v2g:", False, False),
    "rawgroup_gpu": (RAW, "L7a", V2G, "trainonly-split-v2g:", False, True),
    "g3_vectors": (ROWSEL, "L7b", V2, "trainonly-split-v2:", True, False),
    "g3_gpu": (ROWSEL, "L7b", V2, "trainonly-split-v2:", True, True),
}
# the 20 escapes of receipts/trainonly-v2/build/redteam-{rawgroup-v2g,g3append}-publish (tree/case), one fixture each
ESCAPES = {
    "rawgroup_cpu_raw": ("cut_prefix_wrong", "cut_hex_corrupted", "cut_from_other_fold_dataset", "split_schema_wrong"),
    "rawgroup_cpu_nonnumeric": ("cut_prefix_wrong", "cut_hex_corrupted", "cut_from_other_fold_dataset", "split_schema_wrong"),
    "rawgroup_gpu": ("gpu_npy_other_fold", "target_copy_kept", "cut_prefix_wrong", "cut_hex_corrupted",
                     "cut_from_other_fold_dataset", "split_schema_wrong"),
    "g3_vectors": ("cut_hex_corrupted", "cut_from_other_fold_dataset"),
    "g3_gpu": ("gpu_npy_other_fold", "target_copy_kept", "cut_hex_corrupted", "cut_from_other_fold_dataset"),
}


def rd(d):
    return json.loads((d / "context.json").read_bytes())


def wr(d, c):
    (d / "context.json").write_text(json.dumps(c, sort_keys=True) + "\n")


def binding_of(enc_gz):
    raw = gzip.decompress(enc_gz)
    h = jtt1.parse_header(raw)
    a, b = jtt1.train_range(h)
    return {"train_rows": h["n_train"], "features_in": h["p"], "selection_fit": "outer_train_only",
            "removed_target_copy_features": [], "train_block_sha256": mb.sha(raw[a:b])}


def fault(name, d, other, foreign):
    """Apply one red-team fault to context dir d (other = the next fold of its dataset, foreign = a context of another dataset)."""
    c = rd(d)
    if name == "cut_prefix_wrong":
        pre, hx = c["cut"]["cut_id"].split(":", 1)
        c["cut"]["cut_id"] = ("trainonly-split-v2:" if pre.endswith("v2g") else "trainonly-split-v2g:") + hx
    elif name == "cut_hex_corrupted":
        cid = c["cut"]["cut_id"]
        c["cut"]["cut_id"] = cid[:-1] + ("0" if cid[-1] != "0" else "1")
    elif name == "cut_from_other_fold_dataset":
        c["cut"] = rd(foreign)["cut"]
    elif name == "split_schema_wrong":   # the split block where present, else the GPU parent's split_schema (as the red-team)
        if "split" in c["rowsel"]:
            c["rowsel"]["split"]["schema"] = "grouped-stratified-kfold/v1"
        else:
            c["rowsel"]["split_schema"] = "grouped-stratified-kfold/v1"
    elif name == "gpu_npy_other_fold":
        shutil.copy2(other / "hyperion.npy", d / "hyperion.npy")
        c["files"]["hyperion.npy"] = rd(other)["files"]["hyperion.npy"]
    elif name == "target_copy_kept":
        b = c["gpu"]["input"]["binding"]
        b.update(features_in=b["features_in"] + 1, train_block_sha256=mb.sha(b"forged"))
    else:
        raise AssertionError(name)
    wr(d, c)


class Trees(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="manifest-binding-"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def build(self, kind, name):
        """Source build tree, publish tree (two datasets x two folds), runner records, verified fake ledger, summary."""
        schema, mode, split_schema, prefix, research, gpu = KINDS[kind]
        base = self.tmp / name
        source, tree, out = base / "source", base / "tree", base / "runner-out"
        rep = {"level": "dataset", "count": 2, "new_contexts": 2, "contexts": [], "fold_correspondence": "none: target-free new split"}
        dirs = {}
        for ds in (DS, OTHER):
            digest = mb.sha(f"split-{ds}".encode())
            for fold in (0, 1):
                common = dict(schema=schema, values_mode=mode, epoch0_replaced=rep, split_version=split_schema,
                              cut={"cut_id": prefix + digest, "comparable_with_replaced": False},
                              rowsel={"identity": f"id-{ds}", "split": {"schema": split_schema, "sha256": digest}},
                              rights="research-only" if research else None)
                if kind == "rawgroup_cpu_nonnumeric":
                    common["nonnumeric_tier"] = "text-dropped"
                enc = jtt_gz(n_train=10, n_test=0 if not gpu else 3, p=2 + fold)
                src_files = {"encoder.jtt1.gz": enc, "teacher.jtt1.gz": jtt_gz(n_train=10, n_test=0), "a0.f32": f"a0-{ds}{fold}".encode()}
                sd = make_context(source / "shard-0", ds, fold, files=src_files, **common)
                sc = rd(sd)
                if gpu:
                    npy = f"vector-{ds}-{fold}".encode()
                    pub = dict(common, rowsel={"identity": f"id-{ds}", "split_schema": split_schema}, gpu={
                        "input": {"encoder_jtt1_sha256": sc["files"]["encoder.jtt1.gz"]["sha256"], "binding": binding_of(enc)},
                        "components": {"hyperion": {"seconds": 1.0}}})
                    pd = make_context(tree / "shard-0", ds, fold, files={"hyperion.npy": npy}, **pub)
                    od = out / pd.name
                    od.mkdir(parents=True)
                    (od / "context.json").write_text(json.dumps({"view_id": pd.name, "files": rd(pd)["files"]}) + "\n")
                    (od / ".done").write_bytes(b"ok\n")
                elif research:
                    pd = make_context(tree / "shard-0", ds, fold, files={"a0.f32": src_files["a0.f32"]}, **common)
                    c = rd(pd)
                    c["row_level_inputs_local_custody"] = {f: sc["files"][f] for f in ("encoder.jtt1.gz", "teacher.jtt1.gz")}
                    wr(pd, c)
                else:
                    pd = make_context(tree / "shard-0", ds, fold, files=src_files, **common)
                for d in (pd,):
                    c = rd(d)
                    c["context_sha256"] = sc["context_sha256"]
                    wr(d, c)
                dirs[(ds, fold)] = pd
        self.seal(base)
        summary = base / "summary.json"
        summary.write_text("{}\n")
        args = ["--build", str(tree), "--name", name, "--source", str(source), "--summary", f"{summary}#{mb.sha(summary.read_bytes())}",
                "--ledger", str(base / "ledger.jsonl"), "--require-cut"] + (["--rights-research"] if research else [])
        gpu_args = ["--gpu-out", str(out)] if gpu else []
        return base, dirs, args, gpu_args

    def seal(self, base):
        """The verified ledger of the tree as it is now (the publish dry run runs after any fault, as in the red-team)."""
        plans, refused = vp.scan_build(base / "tree")
        self.assertFalse(refused, refused)
        rows = []
        for p in plans:
            for o in p["objects"]:
                common = {"schema": vp.LEDGER_SCHEMA, "outcome": "created", "utc_us": 1, "view_id": p["view_id"]}
                rows += [dict(common, event="verified", kind="object", key=o["key"], sha256=o["sha256"], bytes=o["bytes"], version_id="V-O"),
                         dict(common, event="verified", kind="receipt", key=o["receipt_key"], sha256="f" * 64, bytes=1, version_id="V-R")]
        (base / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def run_binding(self, base, args, gpu_args, publish_clients=None):
        a = args + gpu_args + (["--publish"] if publish_clients else ["--local-dir", str(base / "out")])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mb.main(a, clients=publish_clients, out_root=(base / "out") if publish_clients else None)
        return rc, json.loads(buf.getvalue().strip().splitlines()[-1])

    def run_cut_adapter(self, base, args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = gmc.main(args + ["--local-dir", str(base / "gmc")])
        return rc, json.loads(buf.getvalue().strip().splitlines()[-1])

    def faulted(self, kind, name):
        base, dirs, args, gpu_args = self.build(kind, f"{kind}-{name}")
        fault(name, dirs[(DS, 0)], dirs[(DS, 1)], dirs[(OTHER, 0)])
        self.seal(base)
        return base, args, gpu_args


def _escape_tests(kind, name):
    def reproduced(self):   # the gap: the #239 adapter (frozen tool + transport scan) still writes a manifest for this fault
        base, args, _ = self.faulted(kind, name)
        rc, out = self.run_cut_adapter(base, args)
        self.assertEqual(rc, 0)
        self.assertTrue(Path(out["local"]).exists())

    def refused(self):      # the fix: refused before any manifest file exists
        base, args, gpu_args = self.faulted(kind, name)
        with self.assertRaises(SystemExit) as cm:
            self.run_binding(base, args, gpu_args)
        self.assertIn("refused before any manifest", str(cm.exception))
        self.assertFalse((base / "out").exists())
    return reproduced, refused


class Escapes(Trees):
    pass


for _kind, _names in ESCAPES.items():
    for _name in _names:
        _rep, _ref = _escape_tests(_kind, _name)
        setattr(Escapes, f"test_{_kind}__{_name}__passes_the_current_gates", _rep)
        setattr(Escapes, f"test_{_kind}__{_name}__refused_by_binding", _ref)


class GpuAdditions(Trees):
    """Review 5440093136 [P1]: a GPU context may carry only the runner's outputs. Each added file below has no source, strip or runner
    binding; it must refuse before any manifest file while the runner's own vector is unchanged."""

    def add(self, kind, name, body, seal=True):
        base, dirs, args, gpu_args = self.build(kind, f"{kind}-add-{name}")
        d = dirs[(DS, 0)]
        (d / name).write_bytes(body)
        c = rd(d)
        c["files"][name] = {"sha256": mb.sha(body), "bytes": len(body)}
        if name.endswith(".jtt1.gz"):
            c["files"][name]["raw_sha256"] = mb.sha(gzip.decompress(body))
        wr(d, c)
        if seal:
            self.seal(base)
        return base, args, gpu_args

    def assert_refused(self, base, args, gpu_args, why="refused before any manifest"):
        with self.assertRaises(SystemExit) as cm:
            self.run_binding(base, args, gpu_args)
        self.assertIn(why, str(cm.exception))
        self.assertFalse((base / "out").exists())

    def test_zero_test_encoder_with_synthetic_heldout_target_is_refused(self):
        # Lane B's retained reproducer: n_train=10/n_test=0/p=2 encoder, an unrelated synthetic held-out target written into a
        # purported training target cell, files sha/bytes updated, normal verified ledger sealed; previously exit 0 + manifest entry
        raw = bytearray(gzip.decompress(jtt_gz(n_train=10, n_test=0, p=2)))
        h = jtt1.parse_header(bytes(raw))
        start, _ = jtt1.train_range(h)
        cell = start + 8 * (h["p"] + 1) * 0 + 8 * h["p"]          # row 0, target column
        raw[cell:cell + 8] = struct.pack("<d", 314159.0)          # synthetic held-out target, not derivable from the train rows
        self.assert_refused(*self.add("rawgroup_gpu", "encoder.jtt1.gz", gzip.compress(bytes(raw), mtime=0)))

    def test_added_row_level_and_a0_files_are_refused_in_every_gpu_kind(self):
        for kind in ("rawgroup_gpu", "g3_gpu"):
            research = KINDS[kind][4]
            for name, body in (("encoder.jtt1.gz", jtt_gz(n_train=10, n_test=0, p=2)), ("teacher.jtt1.gz", jtt_gz(n_train=10, n_test=0)),
                               ("a0.f32", b"a0-not-from-the-runner")):
                row_level = name.endswith(".jtt1.gz")
                with self.subTest(kind=kind, name=name):
                    # research rights: the publisher scan already refuses row-level files (rights_row_data); the adapter still refuses
                    # before any manifest. Production rights: the new binding rule refuses.
                    self.assert_refused(*self.add(kind, name, body, seal=not (research and row_level)),
                                        why="refused by the publisher scan" if research and row_level else "refused before any manifest")

    def test_a_gpu_context_missing_a_runner_output_is_refused(self):
        base, dirs, args, gpu_args = self.build("g3_gpu", "missing-output")
        rec = Path(gpu_args[1]) / dirs[(DS, 0)].name / "context.json"
        doc = json.loads(rec.read_bytes())
        doc["files"]["foundation.npy"] = {"sha256": "0" * 64, "bytes": 8}
        rec.write_text(json.dumps(doc) + "\n")
        self.assert_refused(base, args, gpu_args)


class CpuRowLevel(Trees):
    """The same Lane B payload (a zero-test encoder with a synthetic held-out target in a training target cell) in a NON-GPU context."""

    def synthetic_encoder(self, d):
        raw = bytearray(gzip.decompress((d / "encoder.jtt1.gz").read_bytes()))
        h = jtt1.parse_header(bytes(raw))
        start, _ = jtt1.train_range(h)
        raw[start + 8 * h["p"]:start + 8 * h["p"] + 8] = struct.pack("<d", 314159.0)   # row 0 target cell: synthetic held-out value
        return gzip.compress(bytes(raw), mtime=0)

    def place(self, name, forge_strip_record):
        base, dirs, args, gpu_args = self.build("rawgroup_cpu_raw", name)
        d = dirs[(DS, 0)]
        c = rd(d)
        old = dict(c["files"]["encoder.jtt1.gz"])
        body = self.synthetic_encoder(d)
        (d / "encoder.jtt1.gz").write_bytes(body)
        new = {"sha256": mb.sha(body), "bytes": len(body), "raw_sha256": mb.sha(gzip.decompress(body))}
        c["files"]["encoder.jtt1.gz"] = new
        if forge_strip_record:   # a self-declared record that binds the synthetic file to the untouched source file
            from dltrain import quarantine
            c["publication_strip"] = {"key_scheme": quarantine.key_scheme(), "files": {"encoder.jtt1.gz": {
                "original": {"sha256": old["sha256"], "raw_sha256": old.get("raw_sha256")},
                "stripped": {"sha256": new["sha256"], "raw_sha256": new["raw_sha256"]},
                "n_test_payload": 0, "train_rows_sha256_equal": True}}}
        wr(d, c)
        self.seal(base)
        return base, args, gpu_args

    def test_cpu_context_encoder_differing_from_its_source_is_refused(self):
        # without a strip record the frozen tool's own source check already refuses ("encoder.jtt1.gz differs from the source build");
        # the binding pre-pass now refuses first, on the recomputed training rows
        base, args, gpu_args = self.place("cpu-synthetic", False)
        with self.assertRaises(SystemExit) as cm:
            self.run_binding(base, args, gpu_args)
        self.assertRegex(str(cm.exception), "encoder.jtt1.gz (training rows are not the source build's training rows|differs from the source build)")
        self.assertFalse((base / "out").exists())

    def test_cpu_context_forged_strip_record_is_refused(self):
        base, args, gpu_args = self.place("cpu-forged-strip", True)
        rc, out = self.run_cut_adapter(base, args)            # the gap: the frozen tool + #239 adapter accept the forged record
        self.assertEqual(rc, 0)
        self.assertTrue(Path(out["local"]).exists())
        with self.assertRaises(SystemExit) as cm:
            self.run_binding(base, args, gpu_args)
        self.assertIn("encoder.jtt1.gz training rows are not the source build's training rows", str(cm.exception))
        self.assertFalse((base / "out").exists())


class PartitionIdentity(Trees):
    """Review 5440969832 [P1]: a context carrying another fold's partition identity, with --source, view_id, fold, context_sha256,
    cut, split, payload and runner record unchanged, refuses before any manifest (Lane B's reproducer, every kind)."""

    def relabel(self, kind, fields, name):
        base, dirs, args, gpu_args = self.build(kind, name)
        d, other = dirs[(DS, 0)], rd(dirs[(DS, 1)])
        c = rd(d)
        for f in fields:   # another fold's value; where the synthetic folds share one (inner_labels_sha256), a different hash
            c[f] = other[f] if other[f] != c[f] else "e" * 64
        wr(d, c)
        self.seal(base)
        return base, args, gpu_args

    def assert_refused(self, base, args, gpu_args, why):
        with self.assertRaises(SystemExit) as cm:
            self.run_binding(base, args, gpu_args)
        self.assertIn(why, str(cm.exception))
        self.assertIn("refused before any manifest", str(cm.exception))
        self.assertFalse((base / "out").exists())

    def test_other_folds_outer_index_hashes_are_refused(self):
        for kind in KINDS:
            for fields in (("outer_train_indices_sha256",), ("outer_test_indices_sha256",),
                           ("outer_train_indices_sha256", "outer_test_indices_sha256")):
                with self.subTest(kind=kind, fields=fields):
                    self.assert_refused(*self.relabel(kind, fields, f"{kind}-{'-'.join(f[:9] for f in fields)}"),
                                        f"{fields[0]} differs from the source build")

    def test_other_partition_identities_are_refused(self):
        for kind in ("rawgroup_cpu_raw", "g3_gpu"):
            for f in ("inner_labels_sha256", "scaler_sha256"):
                with self.subTest(kind=kind, field=f):
                    self.assert_refused(*self.relabel(kind, (f,), f"{kind}-{f}"), f"{f} differs from the source build")

    def test_rowsel_relabels_and_one_sided_absence_are_refused(self):
        base, dirs, args, gpu_args = self.build("rawgroup_cpu_raw", "rowsel-positions")
        c = rd(dirs[(DS, 0)])
        c["rowsel"]["selected_positions_sha256"] = "1" * 64            # a position hash the source does not carry
        wr(dirs[(DS, 0)], c)
        self.seal(base)
        self.assert_refused(base, args, gpu_args, "rowsel block differs from the source build")
        base, dirs, args, gpu_args = self.build("g3_gpu", "slim-rowsel")
        c = rd(dirs[(DS, 0)])
        c["rowsel"]["identity"] = rd(dirs[(OTHER, 0)])["rowsel"]["identity"]
        wr(dirs[(DS, 0)], c)
        self.seal(base)
        self.assert_refused(base, args, gpu_args, "rowsel identity differs from the source build")
        base, dirs, args, gpu_args = self.build("g3_vectors", "absent-test-index")
        c = rd(dirs[(DS, 0)])
        del c["outer_test_indices_sha256"]
        wr(dirs[(DS, 0)], c)
        self.seal(base)
        self.assert_refused(base, args, gpu_args, "outer_test_indices_sha256 differs from the source build")


class CleanAndContract(Trees):
    def test_clean_trees_of_every_kind_pass_and_bind_the_adapter(self):
        for kind, (_, _, _, _, _, gpu) in KINDS.items():
            with self.subTest(kind):
                base, _, args, gpu_args = self.build(kind, f"clean-{kind}")
                rc, out = self.run_binding(base, args, gpu_args)
                doc = json.loads(Path(out["local"]).read_bytes())
                self.assertEqual(rc, 0)
                self.assertEqual(out["contexts_bound"], 4)
                self.assertEqual(out["gpu_contexts_bound"], 4 if gpu else 0)
                self.assertEqual(doc["binding_adapter"]["sha256"], mb.adapter_identity()["sha256"])
                self.assertEqual(doc["binding_adapter"]["cut_adapter_sha256"], mb.CUT_ADAPTER[1])
                self.assertEqual(doc["cut_adapter"]["sha256"], gmc.adapter_identity()["sha256"])   # #239 block kept
                self.assertEqual(doc["code_sha256"], gmc.TOOL[1])                                   # frozen tool hash kept
                self.assertEqual(out["sha256"], mb.sha(Path(out["local"]).read_bytes()))

    def test_gpu_contexts_need_runner_records(self):
        base, _, args, _ = self.build("g3_gpu", "no-gpu-out")
        with self.assertRaises(SystemExit):
            self.run_binding(base, args, [])
        base, dirs, args, gpu_args = self.build("g3_gpu", "missing-record")
        (Path(gpu_args[1]) / dirs[(DS, 0)].name / ".done").unlink()
        with self.assertRaises(SystemExit):
            self.run_binding(base, args, gpu_args)

    def test_raw_gpu_context_inheriting_its_cut_still_binds(self):   # #239 compatibility: no own cut, cut from the source
        base, dirs, args, gpu_args = self.build("rawgroup_gpu", "inherit")
        for d in dirs.values():
            c = rd(d)
            c.pop("cut")
            wr(d, c)
        for p in (base / "source").glob("shard-0/*/context.json"):
            c = json.loads(p.read_bytes())
            c.pop("cut")
            c["rowsel"]["split"]["schema"] = V2
            wr(p.parent, c)
        for d in dirs.values():
            c = rd(d)
            c["rowsel"]["split_schema"] = V2
            wr(d, c)
        rc, out = self.run_binding(base, args, gpu_args)
        doc = json.loads(Path(out["local"]).read_bytes())
        self.assertEqual(rc, 0)
        self.assertEqual(len(doc["cut_adapter"]["inherited_cut_views"]), 4)
        self.assertTrue(all(e["cut_id"].startswith("trainonly-split-v2:") for e in doc["contexts"]))

    def test_injected_cut_on_an_inheriting_context_is_refused(self):
        base, dirs, args, gpu_args = self.build("rawgroup_gpu", "inject")
        for p in (base / "source").glob("shard-0/*/context.json"):
            c = json.loads(p.read_bytes())
            c.pop("cut")
            wr(p.parent, c)
        with self.assertRaises(SystemExit):
            self.run_binding(base, args, gpu_args)

    def test_local_dir_refused_with_publish(self):
        with self.assertRaises(SystemExit):
            mb.main(["--build", "x", "--name", "n", "--source", "s", "--summary", "p#h", "--publish", "--local-dir", "/nonexistent"])

    def test_adapter_identity_tracks_adapter_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "manifest_binding.py"
            p.write_bytes((HERE / "manifest_binding.py").read_bytes())
            a = mb.adapter_identity(p)
            p.write_bytes(p.read_bytes() + b"# changed\n")
            b = mb.adapter_identity(p)
        self.assertEqual(a["sha256"], mb.adapter_identity()["sha256"])
        self.assertNotEqual(a["sha256"], b["sha256"])

    def test_cut_adapter_pin_matches_the_file(self):
        self.assertEqual(mb.sha((HERE / "gpu_manifest_cut.py").read_bytes()), mb.CUT_ADAPTER[1])

    def test_publish_uses_frozen_transport_create_only(self):
        base, _, args, gpu_args = self.build("g3_gpu", "published")
        bucket = vp.FakeBucket()
        writer, reader, _ = vp.fake_clients(self.tmp / "budget.jsonl", bucket)
        rc, out = self.run_binding(base, args, gpu_args, publish_clients=(writer, reader))
        self.assertEqual(rc, 0)
        self.assertEqual(out["published"]["key"],
                         f"{vp.s3io.V2_PREFIX_KEY}research-only/manifests/published/{out['sha256'][:12]}.json")
        self.assertEqual(out["published"]["sha256"], out["sha256"])


if __name__ == "__main__":
    unittest.main()

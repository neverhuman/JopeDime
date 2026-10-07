"""FOLD-1 partial refusal from builder to manifest (PR #251 review 5441124504).

When a route refuses some folds on their own outer-train rows (LINEAR-DETERMINED here), the admitted folds must stay
publishable: the builder writes them with a replacement record (epoch0_replaced) whose new_contexts counts the WRITTEN
contexts and which names built_folds, admitted_folds and refused_folds; tools/v2_rowsel_raw_manifest.py publishes exactly
the admitted set, bound to the source build tree. Real builders (T1 v2g, G3, L7b, raw v2g), a vector-only publish copy, a
verified ledger and the unchanged manifest entry point; no gate or refusal function is mocked.

  Mixed      fold 1 refused, folds 0/2/3/4 written -> the manifest publishes the four (MUST FAIL on ed2ec4c6, where every
             written context still said new_contexts=5 and the manifest refused the dataset)
  Controls   a clean five-fold build keeps the old record (no membership fields) and publishes five; an incomplete tree
             (an admitted context missing) is refused; a publish copy whose membership differs from the source build is
             refused; a source tree holding a context of a refused fold is refused; every fold refused writes nothing
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dltrain import target_linear as tl
from tests import test_v2_build_local_fold_isolation as li
from tests import test_v2_rowsel_build_fold_isolation as ri
from tests import test_v2_rowsel_raw_fold_isolation as rr

SRC = Path(__file__).resolve().parents[1]
rb, tb, tg = ri.rb, ri.tb, ri.tg
ltool, ltv = li.tool, li.tv
rtool, rv2g, traw = rr.tool, rr.v2g, rr.traw
MEMBERSHIP = ("built_folds", "admitted_folds", "refused_folds")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, SRC / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(b):
    return hashlib.sha256(b).hexdigest()


def publish_copy(src_shard: Path, pub: Path, drop_fold=None, edit=None):
    """The vector-only publication copy of every written source context (A0 published; encoder/teacher kept in local
    custody), as the publish trees carry them."""
    shard = pub / "shard-0"
    shard.mkdir(parents=True)
    for d in sorted(src_shard.iterdir()):
        if not (d / ".done").is_file():
            continue
        c = json.loads((d / "context.json").read_bytes())
        if drop_fold is not None and c["fold"] == drop_fold:
            continue
        c["row_level_inputs_local_custody"] = {f: c["files"][f] for f in ("encoder.jtt1.gz", "teacher.jtt1.gz")}
        c["files"] = {"a0.f32": c["files"]["a0.f32"]}
        if edit:
            edit(c)
        p = shard / d.name
        p.mkdir()
        shutil.copyfile(d / "a0.f32", p / "a0.f32")
        (p / "context.json").write_text(json.dumps(c, sort_keys=True) + "\n")
        (p / ".done").write_bytes(b"ok\n")
    return pub


def manifest(root: Path, pub: Path, src_tree: Path):
    """tools/v2_rowsel_raw_manifest.py main() over a verified ledger; returns (rc, summary, doc) or raises SystemExit."""
    vm = _load("v2_rowsel_raw_manifest", "tools/v2_rowsel_raw_manifest.py")
    vp = _load("v2_publish", "tools/v2_publish.py")
    vm.TV2 = root / "tv2"
    (vm.TV2 / "publish").mkdir(parents=True, exist_ok=True)
    plans, refused = vp.scan_build(pub)
    assert not refused, refused
    ledger = root / f"ledger-{pub.name}.jsonl"
    with open(ledger, "w") as f:
        for p in plans:
            for o in p["objects"]:
                f.write(json.dumps({"event": "verified", "key": o["key"], "sha256": o["sha256"], "bytes": o["bytes"], "version_id": "v1"}) + "\n")
                f.write(json.dumps({"event": "verified", "key": o["receipt_key"], "sha256": "r" * 64, "bytes": 1, "version_id": "v2"}) + "\n")
    summ = root / "summary.json"
    if not summ.exists():
        summ.write_text("{}\n")
    args = ["--build", str(pub), "--name", pub.name, "--source", str(src_tree), "--summary", f"{summ}#{sha(summ.read_bytes())}",
            "--ledger", str(ledger), "--require-cut"]
    if plans and plans[0]["ctx"].get("rights") == "research-only":
        args.append("--rights-research")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = vm.main(args)
    summary = json.loads(out.getvalue().splitlines()[-1])
    return rc, summary, json.loads(Path(summary["local"]).read_text())


# ------------------------------------------------------------------------------------------------- builders (real)

def build_t1(shard: Path, m):
    ds = tb.dataset(m, folds=tg.t2.REGISTRY_FOLDS)
    return rb.build_dataset_v2(ds, tb.items(tb.H, "clean", folds=range(4)), shard, tb.CODES, dict(tb.REFS), budget=tb.BUDGET,
                               profile=tg.profile(),
                               extra={"registry_grouping": {"registry_grouped_by_feature_column": True, "registry_group_columns": [tg.GCOL]},
                                      "group_annotation": {"group_column": tg.SPEC, "group_source": "fixture"}})


def t1_matrix(mixed=True, all_linear=False):
    m = np.array(tg.M, copy=True)
    if not mixed and not all_linear:
        return tb._r9(m)
    m[:, ri.Y] = ri.linear_y(m)
    m = tb._r9(m)
    if all_linear:
        return m
    rows = tg.folds_of(m)[1]
    return ri.heldout_only(m, rows, ri.noise(len(rows), 11))


def build_g3(root: Path, shard: Path, mixed=True):
    g = ri.G3()
    g.blobs = root / "blobs"
    g.blobs.mkdir(exist_ok=True)
    m = g.base()[0] if mixed else tb._r9(np.array(tg.M, copy=True))
    return ri.g3.build_one(g.item(m), g.blobs, "train_fit", tb.CODES, shard)


def build_l7b(shard: Path, mixed=True, all_linear=False):
    case = li.L7b()
    if all_linear:
        M = np.array(ltv.A, copy=True); M[:, -1] = case.lin(M)
    else:
        M = case.base("linear", 1)[0] if mixed else ltv.A
    e = ltv.g_entry()
    return ltool.run_dataset_v2(e, shard, "L7b", lambda e, M=M: ltv.t1_inputs(M, e), set(), {"tools/v2_build_local.py": "c" * 64},
                                log=lambda s: None, version=ltool.SPLIT_V2G)


def build_raw(shard: Path, mixed=True):
    V = rr.base_linear(1) if mixed else np.array(rv2g.VALUES, copy=True)
    with mock.patch.object(rtool, "load_inputs", traw.fake_inputs), mock.patch.object(rtool, "rematch", rv2g.with_values(V)):
        return rtool.run_dataset(rv2g.entry(), shard, rv2g.CODES, None, None, None, set(), traw._Guard(), log=lambda s: None,
                                 split_version=rtool.SPLIT_V2G)


ROUTES = ("t1_v2g", "g3", "l7b", "raw_v2g")


class Mixed(unittest.TestCase):
    """One mixed build per route (fold 1 LINEAR-DETERMINED), shared by the acceptance and the control cases."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="fold1-partial-", dir=os.environ.get("TMPDIR")))
        cls.built = {}
        for route in ROUTES:
            src = cls.root / route / "src"; shard = src / "shard-0"; shard.mkdir(parents=True)
            if route == "t1_v2g":
                rec = build_t1(shard, t1_matrix())
            elif route == "g3":
                rec = build_g3(cls.root / route, shard)
            elif route == "l7b":
                rec = build_l7b(shard)
            else:
                rec = build_raw(shard)
            cls.built[route] = (src, shard, rec)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def written(self, shard):
        return [json.loads((d / "context.json").read_bytes()) for d in sorted(shard.iterdir()) if (d / ".done").is_file()]

    def test_the_admitted_folds_are_published(self):
        """MUST FAIL on ed2ec4c6: every written context said new_contexts=5 and the manifest refused the dataset."""
        for route in ROUTES:
            with self.subTest(route=route):
                src, shard, rec = self.built[route]
                self.assertEqual(rec["status"], "done", rec.get("detail"))
                self.assertEqual([(c["fold"], c["refused"]) for c in rec["refused_contexts"]], [(1, tl.REFUSE)])
                ctxs = self.written(shard)
                self.assertEqual(sorted(c["fold"] for c in ctxs), [0, 2, 3, 4])
                pub = publish_copy(shard, self.root / route / "pub-ok")
                rc, summary, doc = manifest(self.root / route, pub, src)   # ed2ec4c6: '4 contexts in tree, builder made 5'
                for c in ctxs:
                    e0 = c["epoch0_replaced"]
                    self.assertEqual({k: e0.get(k) for k in ("new_contexts",) + MEMBERSHIP},
                                     {"new_contexts": 4, "built_folds": [0, 1, 2, 3, 4], "admitted_folds": [0, 2, 3, 4],
                                      "refused_folds": [{"fold": 1, "reason": tl.REFUSE}]})
                self.assertEqual((rc, summary["published_datasets"], summary["published_contexts"], summary["missing_in_ledger"]), (0, 1, 4, 0))
                self.assertEqual(sorted(e["fold"] for e in doc["contexts"]), [0, 2, 3, 4])
                rep = doc["datasets"][0]["replaces_epoch0"]
                self.assertEqual({k: rep[k] for k in MEMBERSHIP + ("new_contexts",)},
                                 {"built_folds": [0, 1, 2, 3, 4], "admitted_folds": [0, 2, 3, 4],
                                  "refused_folds": [{"fold": 1, "reason": tl.REFUSE}], "new_contexts": 4})

    def test_control_an_incomplete_tree_is_refused(self):
        """An admitted context missing from the publish tree still refuses the dataset as a group."""
        for route in ROUTES:
            with self.subTest(route=route):
                src, shard, _ = self.built[route]
                pub = publish_copy(shard, self.root / route / "pub-missing", drop_fold=3)
                with self.assertRaises(SystemExit) as cm:
                    manifest(self.root / route, pub, src)
                self.assertIn("datasets replace as a group", str(cm.exception))

    def test_control_membership_must_equal_the_source_build(self):
        """A publish copy whose replacement record differs from the source build (membership dropped, count kept) is refused."""
        def strip_membership(c):
            for k in MEMBERSHIP:
                c["epoch0_replaced"].pop(k, None)
            c["epoch0_replaced"]["new_contexts"] = 4
        for route in ROUTES:
            with self.subTest(route=route):
                src, shard, _ = self.built[route]
                pub = publish_copy(shard, self.root / route / "pub-edited", edit=strip_membership)
                with self.assertRaises(SystemExit) as cm:
                    manifest(self.root / route, pub, src)
                self.assertIn("differs from the source build", str(cm.exception))

    def test_control_a_source_context_of_a_refused_fold_is_refused(self):
        """The source tree must hold no context of a refused fold (refused folds are never written)."""
        src, shard, _ = self.built["l7b"]
        other = self.root / "l7b" / "src-planted"
        shutil.copytree(src, other)
        d = sorted(p for p in (other / "shard-0").iterdir() if (p / ".done").is_file())[0]
        planted = d.with_name(d.name + "-fold1")
        shutil.copytree(d, planted)
        c = json.loads((planted / "context.json").read_bytes())
        c.update(fold=1, view_id=planted.name)
        (planted / "context.json").write_text(json.dumps(c))
        pub = publish_copy(shard, self.root / "l7b" / "pub-planted")
        with self.assertRaises(SystemExit) as cm:
            manifest(self.root / "l7b", pub, other)
        self.assertIn("refused fold", str(cm.exception))


class Controls(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="fold1-partial-ctl-", dir=os.environ.get("TMPDIR")))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_a_clean_build_keeps_the_old_record_and_publishes_every_fold(self):
        for route in ("t1_v2g", "l7b"):
            with self.subTest(route=route):
                src = self.root / route / "src"; shard = src / "shard-0"; shard.mkdir(parents=True)
                rec = build_t1(shard, t1_matrix(mixed=False)) if route == "t1_v2g" else build_l7b(shard, mixed=False)
                self.assertEqual((rec["status"], rec.get("refused_contexts")), ("done", None), rec.get("detail"))
                for d in shard.iterdir():
                    if (d / ".done").is_file():
                        e0 = json.loads((d / "context.json").read_bytes())["epoch0_replaced"]
                        self.assertEqual(e0["new_contexts"], 5)
                        self.assertFalse(set(MEMBERSHIP) & set(e0))
                rc, summary, _ = manifest(self.root / route, publish_copy(shard, self.root / route / "pub"), src)
                self.assertEqual((rc, summary["published_contexts"]), (0, 5))

    def test_every_fold_refused_writes_nothing(self):
        for route in ("t1_v2g", "l7b"):
            with self.subTest(route=route):
                shard = self.root / route / "shard-0"; shard.mkdir(parents=True)
                rec = build_t1(shard, t1_matrix(all_linear=True)) if route == "t1_v2g" else build_l7b(shard, all_linear=True)
                self.assertEqual((rec["status"], rec["reason"]), ("failed", tl.REFUSE), rec.get("detail"))
                self.assertEqual([d.name for d in shard.iterdir() if d.name != "_datasets"], [])


if __name__ == "__main__":
    unittest.main()

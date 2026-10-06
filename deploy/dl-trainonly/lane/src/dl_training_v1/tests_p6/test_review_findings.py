"""P6 tests for the 2026-10-05 review findings: train-only leakage (TOL-01..TOL-11) and adoption binding (AB-1..AB-9).

Offline, with the fixtures of tests_p6.test_encodings (imported as a module, so its test classes are not collected twice):
synthetic epochs, JTT1 inputs and Codex objects in a tests_p3 FakeBucket, a stub cpuguard, and the P3Case write guard
(every write under TMPDIR). Production output gates are driven by patching RUN/encodings to a TMPDIR directory outside
every p6-sample-* root.

Run inside the lane slice: python -m unittest discover -s tests_p6 -t . -v
"""
import collections
import json
import os
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dltrain import common, jtt1
from dltrain import encodings as E
from dltrain import metafeatures as A0
from dltrain import normalizers as N
from tests_p3 import fixtures_p3 as fx
from tests_p6 import test_encodings as T

TMPDIR = os.environ["TMPDIR"]
h, sha, jload, rows_of, writable = T.h, T.sha, T.jload, T.rows_of, T.writable
COMPS, UNPINNED, LEGACY, DEV_ONLY, BOUND = T.COMPS, T.UNPINNED, T.LEGACY, T.DEV_ONLY, T.BOUND
MAP_ROLE_TO_ASSIGNMENT = {"train_fit": "train_fit", "diag": "train_fit", "dev": "dev", "heldout": "heldout",
                          "sealed": "sealed_excluded", "prohibited": "prohibited_excluded"}


def make_p3_inputs(root, pmap, map_path, epoch_path, *, name="p3", assignment_override=None, diag=None, p3_map_sha=None,
                   p5_map_sha=None, p5_epoch_sha=None, binding_mutate=None, holdout_mutate=None):
    """P3-shaped holdout.json (assignment + diag recomputable to the map's roles), its jope.dl-binding/v1 document, a
    P3 PASS receipt binding the map sha256 and a P5 PASS receipt binding the epoch and the map."""
    d = Path(root) / f"p3-{name}"
    d.mkdir()
    fam_roles = collections.defaultdict(set)
    for v in pmap.values():
        fam_roles[v["family_id"]].add(v["split_role"])
    assignment = {f: MAP_ROLE_TO_ASSIGNMENT[sorted(roles - {"diag"} or {"diag"})[0]] for f, roles in fam_roles.items()}
    assignment.update(assignment_override or {})
    held = sorted(f for f, a in assignment.items() if a == "heldout")
    dev = sorted(f for f, a in assignment.items() if a == "dev")
    holdout = {"schema": E.HOLDOUT_SCHEMA, "experiment_id": E.EXPERIMENT_ID, "N": len(held) + len(dev) + 1, "k": len(held),
               "held": held, "dev": dev, "assignment": dict(sorted(assignment.items())),
               "diag": sorted(c for c, v in pmap.items() if v["split_role"] == "diag") if diag is None else diag,
               "train_fit": sorted(f for f, a in assignment.items() if a in ("train_fit", "train_nonpool")),
               "tau_hex": h("tau"), "tau_dev_hex": h("tau-dev"), "pool_sha256": h("pool")}
    if holdout_mutate:
        holdout_mutate(holdout)
    hb = common.canonical(holdout)
    (d / "holdout.json").write_bytes(hb)
    binding = {"schema": E.HOLDOUT_BINDING_SCHEMA, "experiment_id": E.EXPERIMENT_ID, "kind": "holdout",
               "holdout_ref": {"bucket": "veox-jopedime-learning-use2", "key": f"objects/holdout/sha256/{sha(hb)}.json",
                               "version_id": "vholdout", "sha256": sha(hb), "bytes": len(hb)},
               "pool_sha256": holdout["pool_sha256"], "N": holdout["N"], "k": holdout["k"], "tau_hex": holdout["tau_hex"],
               "tau_dev_hex": holdout["tau_dev_hex"], "rule": "first freeze wins"}
    if binding_mutate:
        binding_mutate(binding)
    (d / "binding.json").write_bytes(common.canonical(binding))
    map_sha = sha(Path(map_path).read_bytes())
    (d / "p3.json").write_text(json.dumps({"schema": "jope.dl-holdout-receipt/v1", "verdict": "PASS",
                                           "partition_map_sha256": p3_map_sha or map_sha, "holdout_sha256": sha(hb)}))
    (d / "p5.json").write_text(json.dumps({"packet": "P5", "status": "PASS",
                                           "inputs": {"epoch": {"sha256": p5_epoch_sha or sha(Path(epoch_path).read_bytes())},
                                                      "partition_map": {"sha256": p5_map_sha or map_sha}}}))
    return {"holdout": d / "holdout.json", "binding": d / "binding.json", "p3": d / "p3.json", "p5": d / "p5.json"}


def p3_flags(inputs, *, p5=True):
    out = ["--holdout", inputs["holdout"], "--holdout-binding", inputs["binding"], "--p3-receipt", inputs["p3"]]
    return out + (["--p5-receipt", inputs["p5"]] if p5 else [])


def p3_run_args(w, out, inputs, *extra, map_path=None, p5=True):
    args = w.run_args(out, *extra, map_path=map_path)
    args[args.index("--partition-map-kind") + 1] = "p3"
    return args + p3_flags(inputs, p5=p5)


class ProdRoot:
    """RUN/encodings patched to a TMPDIR directory outside every p6-sample-* root: OUT can then be exactly the
    production path and the production output gates run for real."""

    def __init__(self, case):
        self._t = tempfile.TemporaryDirectory(dir=TMPDIR, prefix="p6-prod-")
        case.addCleanup(self._t.cleanup)
        self.encodings = Path(self._t.name) / "encodings"
        p = mock.patch.object(E, "RUN_ENCODINGS", self.encodings)
        p.start()
        case.addCleanup(p.stop)


def raw_from(X, y, n_train, inner=2):
    """JTT1 bytes from full X (n x p) and y, the first n_train rows outer-train."""
    n, p = X.shape
    labels = (np.arange(n_train) % 2).astype(np.uint8)
    mat = np.column_stack([X, y]).astype("<f8")
    return jtt1.MAGIC + struct.pack("<IIIB", n_train, n - n_train, p, inner) + labels.tobytes() + mat.tobytes()


def exposed_raws():
    """dv1: a column whose tiny train values merge under an all-row minimum shift (scale channel); hd1: a column constant
    on the outer-train rows but varying on the outer-test rows (column-set channel)."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(50, 4))
    X[:, 2] = np.where(rng.random(50) < 0.5, 1e-80 * rng.integers(1, 60, 50), rng.random(50))
    y = X[:, 0] + 0.5 * rng.normal(size=50)
    scale = raw_from(X, y, 40)
    X2 = rng.normal(size=(50, 4))
    X2[:40, 3] = 0.25
    X2[40:, 3] = rng.random(10)
    y2 = X2[:, 0] + 0.5 * rng.normal(size=50)
    return {"dv1": scale, "hd1": raw_from(X2, y2, 40)}


# ============================================================================================ TOL-01 / AB-1
class MapBinding(T.P6Case):
    """TOL-01: 'p3' is bound to the P3 freeze (holdout.json roles, its binding, P3/P5 receipts binding this map)."""

    def test_bind_partition_map_recomputes_every_role(self):
        w = T.RunWorld(self.root)
        epoch_doc, epoch_pin = E.load_epoch(w.epoch_path)
        pmap, map_pin = E.load_partition_map(w.map_path)
        inp = make_p3_inputs(self.root, w.pmap, w.map_path, w.epoch_path, assignment_override={"fam-C": "train_nonpool"})

        def bind(i, pm=pmap, mp=map_pin, p5=True):
            return E.bind_partition_map(pm, mp, epoch_pin, holdout=i["holdout"], holdout_binding=i["binding"],
                                        p3_receipt=i["p3"], p5_receipt=i["p5"] if p5 else None)
        b = bind(inp)
        rec = b["record"]
        self.assertEqual(b["status"], "BOUND")
        self.assertEqual(b["sha256"], sha(common.canonical(rec)))
        self.assertEqual((rec["partition_map_sha256"], rec["epoch_sha256"], rec["holdout_sha256"]),
                         (map_pin["sha256"], epoch_pin["sha256"], sha(inp["holdout"].read_bytes())))
        self.assertEqual(rec["roles"], {"dev": 2, "diag": 1, "heldout": 2, "prohibited": 1, "sealed": 1, "train_fit": 3})
        self.assertNotIn(str(self.root), json.dumps(rec))                       # content hashes only, no paths
        self.assertIsNone(bind(inp, p5=False)["record"]["p5_receipt_sha256"])
        # a dev context relabelled train_fit in the map (its P3 receipt rebinding that wrong map): STOP
        bad = dict(w.pmap)
        bad[w.tag["dv1"]] = {"family_id": "fam-D", "split_role": "train_fit"}
        bad_path = w.write_map(bad, "map-bad.json")
        pm2, mp2 = E.load_partition_map(bad_path)
        rebound = make_p3_inputs(self.root, w.pmap, bad_path, w.epoch_path, name="rebound")
        with self.assertRaises(E.Stop) as cm:
            bind(rebound, pm2, mp2)
        self.assertEqual(cm.exception.reason, "partition_map_roles_differ")
        # contradictions and unusable receipts
        cases = {
            "holdout_binding_mismatch": make_p3_inputs(self.root, w.pmap, w.map_path, w.epoch_path, name="hb",
                                                       binding_mutate=lambda b: b["holdout_ref"].update(sha256=h("x"))),
            "binding_fields": make_p3_inputs(self.root, w.pmap, w.map_path, w.epoch_path, name="bf",
                                             binding_mutate=lambda b: b.update(k=99)),
            "unassigned": make_p3_inputs(self.root, w.pmap, w.map_path, w.epoch_path, name="un",
                                         holdout_mutate=lambda d: d["assignment"].pop("fam-B")),
            "diag_outside": make_p3_inputs(self.root, w.pmap, w.map_path, w.epoch_path, name="dg", diag=[w.tag["dv1"]]),
            "p3_other_map": make_p3_inputs(self.root, w.pmap, w.map_path, w.epoch_path, name="p3m", p3_map_sha=h("other-map")),
            "p5_other_map": make_p3_inputs(self.root, w.pmap, w.map_path, w.epoch_path, name="p5m", p5_map_sha=h("other-map")),
            "p5_other_epoch": make_p3_inputs(self.root, w.pmap, w.map_path, w.epoch_path, name="p5e", p5_epoch_sha=h("other-epoch")),
        }
        want = {"holdout_binding_mismatch": (E.Stop, "holdout_binding_mismatch"), "binding_fields": (E.Stop, "holdout_binding_mismatch"),
                "unassigned": (E.Stop, "partition_map_roles_differ"), "diag_outside": (E.Stop, "partition_map_roles_differ"),
                "p3_other_map": (E.Blocked, "p3_receipt"), "p5_other_map": (E.Blocked, "p5_receipt"),
                "p5_other_epoch": (E.Blocked, "p5_receipt")}
        for name, i in cases.items():
            with self.assertRaises(want[name][0], msg=name) as cm:
                bind(i)
            self.assertEqual(cm.exception.reason, want[name][1], name)
        with self.assertRaises(E.Blocked) as cm:
            E.bind_partition_map(pmap, map_pin, epoch_pin, holdout=None, holdout_binding=inp["binding"], p3_receipt=inp["p3"])
        self.assertEqual(cm.exception.reason, "map_binding_inputs")

    def test_p3_kind_is_a_positive_property(self):
        # the reviewer's probe: a P5-sample family id, no sidecar, labelled p3 -> no longer accepted without a binding
        cid = h("ctx:probe")
        probe = {cid: {"family_id": "p5-sample-fam:abc", "split_role": "train_fit"}}
        with self.assertRaises(E.Refused) as cm:
            E.resolve_map_kind(None, probe, "p3")
        self.assertEqual(cm.exception.reason, "p3_map_unbound")
        other = {"status": "BOUND", "record": {"partition_map_sha256": h("another-map")}}
        with self.assertRaises(E.Refused):
            E.resolve_map_kind(None, probe, "p3", other)
        mine = {"status": "BOUND", "record": {"partition_map_sha256": sha(common.canonical(probe))}}
        self.assertEqual(E.resolve_map_kind(None, probe, "p3", mine), "p3")
        with self.assertRaises(E.Refused) as cm:
            E.resolve_map_kind(None, probe, "fake", mine)
        self.assertEqual(cm.exception.reason, "binding_with_fake_map")

    def test_sample_run_with_p3_needs_the_binding_and_a_fake_map_takes_none(self):
        w = T.RunWorld(self.root)
        args = w.run_args(self.root / "a0-p3")
        args[args.index("--partition-map-kind") + 1] = "p3"
        code, s = self.cli(*args)
        self.assertEqual((code, s["status"], s["failure"]["reason"]), (3, "BLOCKED", "map_binding_inputs"))
        inp = make_p3_inputs(self.root, w.pmap, w.map_path, w.epoch_path)
        code, s = self.cli(*w.run_args(self.root / "a0-fake-bound", "--holdout-binding", inp["binding"]))
        self.assertEqual((code, s["failure"]["reason"]), (2, "binding_with_fake_map"))
        code, s = self.cli(*p3_run_args(w, self.root / "a0-p3-bound", inp))
        self.assertEqual((code, s["status"]), (0, "PASS"), s)
        self.assertEqual(s["packet"], "P6-A0-sample")

    def test_production_run_binds_the_map_and_verify_rechecks_it(self):
        prod = ProdRoot(self)
        w = T.RunWorld(self.root)
        inp = make_p3_inputs(self.root, w.pmap, w.map_path, w.epoch_path)
        out = prod.encodings / "A0"
        # missing P3 inputs: BLOCKED before anything is read or written
        code, s = self.cli(*w.run_args(out))
        self.assertEqual((code, s["reason"]), (3, "output_gate"))
        for absent in ("--holdout", "--p3-receipt"):
            self.assertIn(f"no {absent}", s["detail"])
        self.assertFalse(out.exists())
        # P5 receipt bound to another partition map: BLOCKED inside the run, no manifest written
        bad_p5 = make_p3_inputs(self.root, w.pmap, w.map_path, w.epoch_path, name="p5m", p5_map_sha=h("other-map"))
        code, s = self.cli(*p3_run_args(w, out, dict(inp, p5=bad_p5["p5"])))
        self.assertEqual((code, s["failure"]["reason"]), (3, "p5_receipt"))
        self.assertFalse((out / "manifests").exists())
        # a map whose roles differ from holdout.json (its P3 receipt rebinding it): STOP, nothing beyond receipts written
        bad = dict(w.pmap)
        bad[w.tag["dv1"]] = {"family_id": "fam-D", "split_role": "train_fit"}
        bad_path = w.write_map(bad, "map-bad.json")
        rebound = make_p3_inputs(self.root, w.pmap, bad_path, w.epoch_path, name="rebound")
        code, s = self.cli(*p3_run_args(w, out, rebound, map_path=bad_path))
        self.assertEqual((code, s["status"], s["failure"]["reason"]), (4, "STOP", "partition_map_roles_differ"))
        self.assertFalse((out / "arrays").exists() or (out / "manifests").exists())
        # bound: PASS at the production gate
        code, s = self.cli(*p3_run_args(w, out, inp))
        self.assertEqual((code, s["status"], s["packet"]), (0, "PASS", "P6-A0"), s)
        receipt = jload(s["receipt"]["path"])
        self.assertEqual(receipt["mode"], "production")
        man = jload(out / "manifests/train_fit.json")
        binding = man["inputs"]["partition_map"]["binding"]
        self.assertEqual((man["inputs"]["partition_map"]["kind"], binding["schema"]), ("p3", E.MAP_BINDING_SCHEMA))
        header = jload(out / f"normalizers/{man['normalizers'][0]['normalizer_sha256']}.json")
        self.assertEqual(header["fit"]["partition_map_binding_sha256"], binding["sha256"])
        self.assertEqual(header["fit"]["holdout_binding_sha256"], sha(inp["binding"].read_bytes()))
        self.assertEqual(man["inputs"]["holdout_binding"]["sha256"], sha(inp["binding"].read_bytes()))
        self.assertEqual(receipt["inputs"]["p3_receipt"]["sha256"], sha(inp["p3"].read_bytes()))
        # verify recomputes the binding from the P3 inputs it is given
        code, s = self.cli(*(w.verify_args(out, "--coverage") + p3_flags(inp)))
        doc = jload(s["verify"]["path"])
        self.assertEqual((code, doc["results"]["partition_map_binding"]["status"]), (0, "PASS"), doc["results"])
        code, s = self.cli(*w.verify_args(out, "--coverage"))                    # p3 manifests, no P3 inputs
        doc = jload(s["verify"]["path"])
        self.assertEqual((code, doc["results"]["partition_map_binding"]["status"]), (2, "FAIL"))
        other = make_p3_inputs(self.root, w.pmap, w.map_path, w.epoch_path, name="other",
                               holdout_mutate=lambda d: d.update(tau_hex=h("other-tau")),
                               binding_mutate=lambda b: b.update(tau_hex=h("other-tau")))
        code, s = self.cli(*(w.verify_args(out, "--coverage") + p3_flags(other)))
        doc = jload(s["verify"]["path"])
        self.assertEqual(doc["results"]["partition_map_binding"]["status"], "FAIL")
        self.assertFalse(doc["results"]["partition_map_binding"]["checks"]["manifests_record_this_binding"])

    def test_production_adopt_gate_is_the_run_gate(self):
        """AB-1: the production adopt gate needs a bound p3 map, the holdout binding and the P3/P5 receipts."""
        prod = ProdRoot(self)
        w = T.build_small_world(self.root, name="AP", terminal=True)
        self.use_bucket(w.bucket)
        out = prod.encodings / "adoption" / f"{w.release_ref['sha256'][:12]}-{w.index_ref['sha256'][:12]}"
        code, s = self.cli(*w.args(out, dry_run=False))                         # kind fake
        self.assertEqual((code, s["reason"]), (3, "output_gate"))
        self.assertIn("partition_map_kind is not p3", s["detail"])
        code, s = self.cli(*w.args(out, dry_run=False, kind="p3"))              # p3 without the P3 inputs
        self.assertEqual((code, s["reason"]), (3, "output_gate"))
        self.assertFalse(out.exists())
        inp = make_p3_inputs(self.root, w.pmap, w.map_path, w.epoch_path)
        code, s = self.cli(*w.args(out, dry_run=False, kind="p3", p3=inp))
        self.assertEqual((code, s["status"]), (0, "PASS"), s)
        doc = jload(out / "adoption.json")
        self.assertEqual((doc["mode"], doc["gate"], doc["inputs"]["partition_map"]["kind"]), ("production", "production", "p3"))
        bsha = doc["inputs"]["partition_map"]["binding"]["sha256"]
        self.assertEqual(doc["inputs"]["holdout_binding"]["sha256"], sha(inp["binding"].read_bytes()))
        self.assertEqual(doc["inputs"]["p5_receipt"]["sha256"], sha(inp["p5"].read_bytes()))
        rows = rows_of(out / "rows/train_fit.jsonl") + rows_of(out / "rows/dev.jsonl")
        self.assertEqual({r["family_provenance"]["partition_map"]["binding_sha256"] for r in rows}, {bsha})
        self.assertEqual(jload(s["receipt"]["path"])["gate"], "production")


# ===================================================================================================== TOL-02
class RunFitClosure(T.P6Case):
    def test_an_open_train_family_stops_before_any_write(self):
        w = T.RunWorld(self.root, role_override={"dg1": "heldout"})             # family A: train_fit tf1 + heldout dg1
        out = self.root / "a0"
        code, s = self.cli(*w.run_args(out))
        self.assertEqual((code, s["status"], s["failure"]["reason"]), (4, "STOP", "fit_closure"))
        self.assertIn("fam-A", s["failure"]["detail"])
        for rel in ("arrays", "sealed", "normalizers", "manifests", "arm.json"):
            self.assertFalse((out / rel).exists(), rel)
        self.assertEqual(E.map_family_closure(w.pmap, [w.tag["dv1"]]), [])     # a dev context alone is not checked
        self.assertEqual(E.map_family_closure(w.pmap, [w.tag["tf1"]]), [{"family_id": "fam-A", "other_roles": ["heldout"]}])


# ============================================================================================ TOL-03 / AB-10
class TrainOnlyPromotion(T.P6Case):
    ALL = ("--coverage", "--fit-closure", "--poison-test", "--counterfactual", "--affine-invariance", "--upstream-exposure")

    def test_verified_label_lives_only_in_a_bound_verify_receipt(self):
        w = T.RunWorld(self.root)
        out = self.root / "a0"
        self.assertEqual(self.cli(*w.run_args(out))[0], 0)
        for p in E.PARTITIONS:
            m = jload(out / E._manifest_rel(p))
            self.assertEqual({e["train_only"] for e in m["contexts"]}, {"STRUCTURAL_JTT1_TRAIN_BYTES"})
            self.assertEqual(m["train_only_rule"], E.TRAIN_ONLY_RULE)               # the label is promoted only by verify
        code, s = self.cli(*w.verify_args(out, *self.ALL))
        doc = jload(s["verify"]["path"])
        promo = doc["train_only_promotion"]
        self.assertEqual((code, doc["status"], promo["granted"]), (0, "PASS", True), promo)
        self.assertEqual((promo["label"], promo["from_label"], promo["contexts_promoted"], promo["contexts_open"]),
                         ("VERIFIED_BY_COUNTERFACTUAL", "STRUCTURAL_JTT1_TRAIN_BYTES", 8, 0))
        self.assertTrue(promo["demonstration_only"])                               # a fake map
        for p in E.PARTITIONS:
            data = (out / E._manifest_rel(p)).read_bytes()
            self.assertEqual(promo["manifests"][p], {"path": E._manifest_rel(p), "sha256": sha(data), "bytes": len(data)})
        self.assertEqual(os.stat(s["verify"]["path"]).st_mode & 0o777, 0o444)     # create-only receipt
        # a partial verify promotes nothing
        code, s = self.cli(*w.verify_args(out, "--coverage"))
        promo = jload(s["verify"]["path"])["train_only_promotion"]
        self.assertEqual((code, promo["granted"], promo["contexts_promoted"]), (0, False, 0))
        self.assertEqual(promo["missing"], ["fit_closure", "poison_test", "counterfactual", "affine_invariance", "upstream_exposure"])
        code, s = self.cli(*w.verify_args(out, *self.ALL, "--poison-scope", "sample:1"))
        promo = jload(s["verify"]["path"])["train_only_promotion"]
        self.assertEqual((promo["granted"], promo["failed"]), (False, ["poison_test:scope_not_all"]))


# ============================================================================================ TOL-04 / TOL-05
class UpstreamExposure(T.P6Case):
    def test_merges_under_shift(self):
        self.assertTrue(E.merges_under_shift(np.array([0.0, 5.89e-23, 0.5])))
        self.assertTrue(E.merges_under_shift(np.array([0.0, 1e-17, 1.0]), shifts=(1.0,)))
        self.assertFalse(E.merges_under_shift(np.array([0.1, 0.2, 0.3, np.nan])))
        self.assertFalse(E.merges_under_shift(np.array([7.0, 7.0])))

    def test_exposed_contexts_are_labelled_open_and_never_promoted(self):
        w = T.RunWorld(self.root, raw_override=exposed_raws())
        out = self.root / "a0"
        code, s = self.cli(*w.run_args(out))
        self.assertEqual(code, 0, s)
        self.assertEqual(s["counts"]["upstream_exposure"]["dev"], {"scale": 1, "column_set": 0})
        self.assertEqual(s["counts"]["upstream_exposure"]["heldout"], {"scale": 0, "column_set": 1})
        dev = {e["context_sha256"]: e for e in jload(out / "manifests/dev.json")["contexts"]}
        held = {e["context_sha256"]: e for e in jload(out / "sealed/heldout.json")["contexts"]}
        self.assertEqual(dev[w.tag["dv1"]]["train_only_scope"], "jtt1_bytes; upstream_scale_exposed")
        self.assertEqual(dev[w.tag["dv1"]]["upstream_exposure"]["scale"]["x_columns"], [2])
        self.assertEqual(held[w.tag["hd1"]]["train_only_scope"], "jtt1_bytes; upstream_column_set_exposed")
        self.assertEqual(held[w.tag["hd1"]]["upstream_exposure"]["column_set"]["train_constant_columns"], 1)
        code, s = self.cli(*w.verify_args(out, *TrainOnlyPromotion.ALL))
        doc = jload(s["verify"]["path"])
        ue, promo = doc["results"]["upstream_exposure"], doc["train_only_promotion"]
        self.assertEqual((code, ue["status"], ue["scale"]["status"], ue["column_set"]["status"]), (0, "PASS", "OPEN", "OPEN"))
        self.assertEqual(ue["scale"]["listed"][0]["context"], w.tag["dv1"])
        self.assertEqual(ue["column_set"]["listed"][0]["context"], "<sealed>")
        self.assertNotIn(w.tag["hd1"], json.dumps(doc))
        self.assertEqual((promo["granted"], promo["contexts_promoted"], promo["contexts_open"]), (True, 6, 2))
        self.assertEqual({(o["partition"], tuple(o["channels"])) for o in promo["open"]},
                         {("dev", ("scale",)), ("heldout", ("column_set",))})
        self.assertEqual(sorted(f["channel"] for f in doc["open_findings"]), ["column_set", "scale"])
        # a manifest whose recorded exposure differs from the recomputation FAILs the check
        path = writable(out / "manifests/dev.json")
        m = jload(path)
        for e in m["contexts"]:
            if e["context_sha256"] == w.tag["dv1"]:
                e["upstream_exposure"]["scale"]["x_columns"] = []
        path.write_bytes(common.canonical(m))
        code, s = self.cli(*w.verify_args(out, "--upstream-exposure"))
        ue = jload(s["verify"]["path"])["results"]["upstream_exposure"]
        self.assertEqual((code, ue["status"], ue["manifest_mismatch_total"]), (2, "FAIL", 1))

    def test_column_set_audit_confirms_test_varying_columns(self):
        rng = np.random.default_rng(3)
        n = 50
        X = rng.normal(size=(n, 3))
        X[:, 1] = 0.5
        X[45, 1] = 0.7                     # one row varies column 1
        y = X[:, 0] + rng.normal(size=n)
        order0 = list(range(n))                                  # fold 0: row 45 is an outer-test row
        order1 = [45] + [r for r in range(n) if r != 45]         # fold 1: row 45 is an outer-train row
        raws = {"f0": raw_from(X[order0], y[order0], 40), "f1": raw_from(X[order1], y[order1], 40),
                "solo": raw_from(X, y, 40)}
        ctx = {h(k): {"dataset_hash": "d" * 16 if k != "solo" else "e" * 16, "sample_identity_sha256": h("sid")} for k in raws}
        raw_of = {h(k): v for k, v in raws.items()}.__getitem__
        res = E.column_set_audit(sorted(ctx), ctx, {h("f0"): "train_fit", h("f1"): "train_fit", h("solo"): "dev"}, raw_of,
                                 T.StubGuard())
        self.assertEqual((res["status"], res["samples_with_siblings"], res["contexts_checked"]), ("OPEN", 1, 2))
        self.assertEqual((res["train_constant_columns"], res["confirmed_test_varying_columns"], res["contexts_confirmed"]), (1, 1, 1))
        self.assertEqual(res["listed"], [{"context": h("f0"), "partition": "train_fit", "columns": [1], "columns_confirmed": 1}])


# ============================================================================================= TOL-07 / AB-3
class ProductionAdoptScope(T.P6Case):
    def test_production_refuses_a_subset_and_missing_content_checks(self):
        w = T.build_small_world(self.root, name="PS", terminal=True)
        self.use_bucket(w.bucket)
        code, s = self.cli(*w.args(self.root / "o1", dry_run=False, contexts=True))
        self.assertEqual((code, s["reason"]), (2, "contexts_in_production"))
        code, s = self.cli(*w.args(self.root / "o2", dry_run=False, content=False))
        self.assertEqual((code, s["reason"]), (2, "production_needs_content_checks"))
        args = [a for a in w.args(self.root / "o3", dry_run=False)]
        i = args.index("--jtt1-cache")
        del args[i:i + 2]
        code, s = self.cli(*args)
        self.assertEqual((code, s["reason"]), (2, "content_checks_need_jtt1_cache"))
        for o in ("o1", "o2", "o3"):
            self.assertFalse((self.root / o).exists())

    def test_production_pass_needs_every_adopted_row_at_tier_a_with_jtt1_evidence(self):
        w = T.build_small_world(self.root, name="PE", terminal=True)
        self.use_bucket(w.bucket)
        with mock.patch.object(E, "PRODUCTION_ARRAY_COMPONENTS", ("hyperion",)):
            code, s = self.cli(*w.args(self.root / "adopt-PE", dry_run=False))
        doc = jload(self.root / "adopt-PE" / "adoption.json")
        self.assertEqual((code, doc["status"]), (2, "FAIL"))
        self.assertEqual(doc["completion"]["unfinished_bound_rows"], 0)
        self.assertEqual(doc["completion"]["adopted_without_jtt1_evidence"], 5)   # foundation dA, dB; target dA, dB, dC
        rows = rows_of(self.root / "adopt-PE" / "rows/train_fit.jsonl") + rows_of(self.root / "adopt-PE" / "rows/dev.jsonl")
        tier_o = [r for r in rows if r["status"] == "adopted" and r["tier"] == "O"]
        self.assertEqual(len(tier_o), 5)
        self.assertTrue(all("array_content" not in r["jtt1_evidence"] for r in tier_o))


# ===================================================================================================== TOL-09
class DevOnlyArms(T.P6Case):
    def test_no_heldout_manifest_for_a_dev_only_arm(self):
        with self.assertRaises(E.Stop) as cm:
            E.build_manifests("LB-replica", {}, {}, "dl-v1-e0:LB-replica:" + "0" * 16, 1, {}, {"kind": "all", "n": 0}, {})
        self.assertEqual(cm.exception.reason, "dev_only_arm_heldout")
        self.assertEqual(E.DEV_ONLY_ARMS, N.DEV_ONLY_ARMS)
        self.assertEqual(E.RESEARCH_ONLY_ARMS, N.RESEARCH_ONLY_ARMS)


# ===================================================================================================== TOL-11
class PerturbationsNeverReadTestBytes(T.P6Case):
    def test_outputs_depend_only_on_the_train_range(self):
        raw = T.make_raw(40, 10, 4, seed=3)
        stop = E._train_stop(raw)
        other = raw[:stop] + bytes(reversed(raw[stop:]))
        self.assertNotEqual(raw, other)
        fns = {"fill_00": lambda r: E.perturb_test_fill(r, 0x00), "fill_ff": lambda r: E.perturb_test_fill(r, 0xFF),
               "random": lambda r: E.perturb_test_random(r, random.Random(1)), "nan": E.perturb_test_nan,
               "removed": E.perturb_test_removed}
        for name, fn in fns.items():
            self.assertEqual(fn(raw), fn(other), name)
            self.assertEqual(jtt1.train_rows_sha256(fn(raw)), jtt1.train_rows_sha256(raw), name)
        self.assertEqual(jtt1.parse_header(E.perturb_test_removed(raw))["n_test"], 0)
        self.assertEqual(A0.from_raw(E.perturb_test_removed(raw))[0], A0.from_raw(raw)[0])
        self.assertEqual([name for name, _v in E.counterfactual_inputs(raw)], list(E.COUNTERFACTUAL_VARIANTS))
        self.assertFalse(hasattr(E, "perturb_test_xor"))


# ======================================================================================================= AB-2
class TerminalReleaseBinding(T.P6Case):
    def world(self, name, **kw):
        w = T.build_small_world(self.root, name=name, terminal=True, **kw)
        self.use_bucket(w.bucket)
        return w

    def test_terminal_release_is_accepted_and_bound(self):
        w = self.world("TR")
        code, s = self.cli(*w.args(self.root / "adopt-TR", dry_run=False))
        self.assertEqual((code, s["status"]), (0, "PASS"), s)
        rel = jload(self.root / "adopt-TR" / "adoption.json")["inputs"]["release"]
        self.assertEqual((rel["schema"], rel["terminal_release"], rel["index_ref_equals_adopted_index"],
                          rel["launch_release_equals_plan_pin"]), (E.TERMINAL_RELEASE_SCHEMA, True, True, True))
        self.assertTrue(rel["qualification"]["adoption_qualified"])
        self.assertIn("summary.raw_data_independence_qualified", rel["qualification"])
        # a dry run accepts the terminal schema and records the same binding
        code, s = self.cli(*w.args(self.root / "adopt-TR-dry", dry_run=True))
        self.assertEqual((code, s["status"]), (0, "DRY_RUN"))

    def test_production_refusals(self):
        w = self.world("TX")
        cases = {
            "release_index_ref_differs": w.terminal_release(index_ref=dict(w.index_ref, sha256=h("other-index"))),
            "launch_release_not_plan_pin": w.terminal_release(original_launch_release_ref=dict(E.PLAN_LAUNCH_RELEASE_REF, version_id="v2")),
        }
        for reason, doc in cases.items():
            ref, path = w.publish_release(doc, f"release-{reason}.json")
            out = self.root / f"adopt-{reason}"
            code, s = self.cli(*w.args(out, dry_run=False, release_ref_path=path,
                                       fetch_receipt=w.write_fetch_receipt(w.index_ref, ref, f"fetch-{reason}.json")))
            self.assertEqual((code, s["failure"]["reason"]), (2, reason), s)
            self.assertFalse((out / "adoption.json").exists())
        # the launch release names no adopted index: production refuses it
        ref, path = w.publish_release(w.launch_release(), "release-launch.json")
        code, s = self.cli(*w.args(self.root / "adopt-launch", dry_run=False, release_ref_path=path))
        self.assertEqual((code, s["failure"]["reason"]), (2, "release_index_ref_differs"))
        # the index key and VersionId come from a recorded exact-VersionId fetch, not from --index-ref text
        code, s = self.cli(*w.args(self.root / "adopt-nofetch", dry_run=False, fetch_receipt=False))
        self.assertEqual((code, s["failure"]["reason"]), (2, "index_fetch_receipt_absent"))
        bad = w.write_fetch_receipt(dict(w.index_ref, version_id="another-version"), w.release_ref, "fetch-bad.json")
        code, s = self.cli(*w.args(self.root / "adopt-badfetch", dry_run=False, fetch_receipt=bad))
        self.assertEqual((code, s["failure"]["reason"]), (2, "index_fetch_receipt_mismatch"))

    def test_unqualified_release_needs_an_owner_waiver(self):
        w = self.world("TQ", qualified=False)
        code, s = self.cli(*w.args(self.root / "adopt-q0", dry_run=False))
        self.assertEqual((code, s["failure"]["reason"]), (3, "release_not_adoption_qualified"))
        needed = ["adoption_qualified", "summary.context_gp_join_qualified", "summary.raw_data_independence_qualified",
                  "summary.rights_family_adoption_qualified"]

        def waiver(name, **doc):
            path = self.root / name
            path.write_text(json.dumps(dict({"schema": E.OWNER_WAIVER_SCHEMA, "release_sha256": w.release_ref["sha256"],
                                             "waived": needed, "decision": "owner-decision-ref"}, **doc)))
            return path
        code, s = self.cli(*w.args(self.root / "adopt-q1", dry_run=False, waiver=waiver("w-partial.json", waived=needed[:2])))
        self.assertEqual((code, s["failure"]["reason"]), (3, "owner_waiver_incomplete"))
        code, s = self.cli(*w.args(self.root / "adopt-q2", dry_run=False, waiver=waiver("w-other.json", release_sha256=h("x"))))
        self.assertEqual((code, s["failure"]["reason"]), (3, "owner_waiver_mismatch"))
        wpath = waiver("w-ok.json")
        code, s = self.cli(*w.args(self.root / "adopt-q3", dry_run=False, waiver=wpath))
        self.assertEqual((code, s["status"]), (0, "PASS"), s)
        doc = jload(self.root / "adopt-q3" / "adoption.json")
        self.assertEqual(doc["inputs"]["owner_waiver"]["sha256"], sha(wpath.read_bytes()))
        self.assertEqual(doc["inputs"]["owner_waiver"]["waived"], needed)
        self.assertFalse(doc["inputs"]["release"]["qualification"]["adoption_qualified"])
        code, s = self.cli(*w.args(self.root / "adopt-q-dry", dry_run=True))      # a dry run only records the flags
        self.assertEqual((code, s["status"]), (0, "DRY_RUN"))


# ======================================================================================================= AB-4
class IdentityAndManifestBinding(T.P6Case):
    def world(self, name, terminal=True):
        w = T.AdoptWorld(self.root, name=name, terminal=terminal)
        w.add("ok", "train_fit")
        w.add("idpre", "train_fit", identity_prefix_wrong=True, expect={k: "rejected:identity_prefix_mismatch" for k in COMPS})
        w.add("manif", "train_fit", manifest_mismatch=True, expect=T.four("rejected:packet_manifest_mismatch"))
        w.add("noprep", "dev", prepared_absent=True, expect=T.four("rejected:prepared_ref_unverifiable"))
        w.add("norep", "train_fit", hyperion={"no_repeat": True}, expect={"hyperion": "rejected:index_fold_repeat"})
        w.add("mrep", "dev", membership_repeat=True, expect=T.four("rejected:membership_repeat"))
        w.finalize()
        self.use_bucket(w.bucket)
        return w

    def test_identity_prefix_manifest_and_repeat_are_bound(self):
        w = self.world("IB")
        out = self.root / "adopt-IB"
        code, s = self.cli(*w.args(out, dry_run=False))
        self.assertEqual((code, s["status"]), (0, "PASS"), s)
        doc = jload(out / "adoption.json")
        expected = w.expected(production=True)
        self.assertEqual(doc["counts"]["final"], T.counts_of(expected))
        rows = {(r["context_sha256"], r["component"]): r for p in ("train_fit", "dev") for r in rows_of(out / f"rows/{p}.jsonl")}
        self.assertEqual({k: r["status"] for k, r in rows.items()}, expected)
        self.assertEqual(rows[(w.cid["ok"], "hyperion")]["binding_evidence"]["membership"],
                         "outer train/test index sha256 and rows equal; outer_repeat 0")

    def test_map_audit_reports_an_identity_prefix_mismatch(self):
        w = self.world("IM", terminal=False)
        with mock.patch.object(T.s3io, "make_client", mock.Mock(side_effect=AssertionError("no client"))):
            summary, rows = E.map_audit(w.epoch_path, w.index_path, sha(w.index_path.read_bytes()), w.mem_path)
        row = next(r for r in rows if r["context_sha256"] == w.cid["idpre"])
        self.assertEqual(row["status"], "none")
        self.assertIn("identity_prefix_mismatch:hyperion", row["problems"])
        self.assertEqual(summary["status_counts"]["identity_prefix_mismatch"], 5)
        ok = next(r for r in rows if r["context_sha256"] == w.cid["ok"])
        self.assertEqual((ok["status"], ok["problems"]), ("complete", []))

    def test_duplicate_context_join_keys_stop(self):
        w = T.build_small_world(self.root, name="DK")
        self.use_bucket(w.bucket)
        epoch = jload(w.epoch_path)
        twin = dict(epoch["contexts"][0], context_sha256=h("ctx:twin"))
        epoch["contexts"].insert(1, twin)
        w.epoch_path.write_bytes(common.canonical(epoch))
        mems = [json.loads(line) for line in w.mem_path.read_text().splitlines()]
        mems.append(dict(mems[0], context_sha256=twin["context_sha256"]))
        w.mem_path.write_text("".join(json.dumps(m, sort_keys=True) + "\n" for m in mems))
        pmap = dict(w.pmap, **{twin["context_sha256"]: dict(w.pmap[epoch["contexts"][0]["context_sha256"]])})
        w.map_path.write_bytes(common.canonical(pmap))
        code, s = self.cli(*w.args(self.root / "adopt-DK", dry_run=True))
        self.assertEqual((code, s["failure"]["reason"]), (4, "context_join_key_duplicate"))
        self.assertFalse((self.root / "adopt-DK" / "rows").exists())


# ======================================================================================================= AB-5
class SlotAccounting(T.P6Case):
    def test_retired_fold_is_residual_and_reconciled(self):
        w = T.AdoptWorld(self.root, name="RT", terminal=True)
        w.add("ok", "train_fit")
        w.add("ret", "dev", retired=True)
        w.finalize()
        self.use_bucket(w.bucket)
        out = self.root / "adopt-RT"
        code, s = self.cli(*w.args(out, dry_run=False))
        self.assertEqual((code, s["status"]), (0, "PASS"), s)
        doc = jload(out / "adoption.json")
        self.assertEqual(doc["counts"]["final"], T.counts_of(w.expected(production=True)))
        self.assertEqual({r["status"] for r in rows_of(out / "rows/dev.jsonl")}, {"residual:absent_or_retired_before_dispatch"})
        slots = doc["slots"]
        self.assertEqual((slots["packet_slots"], slots["null_identity_rows"], slots["retired_folds"], slots["retired_component_slots"],
                          slots["release_retired_component_slots"], slots["reconciled_with_release"], slots["unaccounted_slots"]),
                         (10, 1, 1, 5, 5, True, 0))
        self.assertEqual(doc["counts"]["residual_by_reason"]["total"], {"residual:absent_or_retired_before_dispatch": 5})

    def test_unaccounted_slots_stop_in_a_terminal_index(self):
        w = T.AdoptWorld(self.root, name="UA", terminal=True)
        w.add("ok", "train_fit")
        w.add("lost", "train_fit", missing=("target",))             # a packet slot with no index row
        w.finalize()
        self.use_bucket(w.bucket)
        for dry in (True, False):
            code, s = self.cli(*w.args(self.root / f"adopt-UA-{dry}", dry_run=dry))
            self.assertEqual((code, s["failure"]["reason"]), (4, "unaccounted_packet_slots"), s)
            self.assertFalse((self.root / f"adopt-UA-{dry}" / "rows").exists())
        u = T.AdoptWorld(self.root, name="UN", terminal=True)
        u.add("ok", "train_fit")
        u.finalize(extra_null_row=True)                              # a null-identity row resolving to no packet slot
        self.use_bucket(u.bucket)
        code, s = self.cli(*u.args(self.root / "adopt-UN", dry_run=True))
        self.assertEqual((code, s["failure"]["reason"]), (4, "unaccounted_packet_slots"))
        # a non-terminal index: missing rows are pending, never a STOP
        p = T.AdoptWorld(self.root, name="PN", terminal=False)
        p.add("ok", "train_fit", subset=True)
        p.add("lost", "train_fit", missing=("target",))
        p.finalize(extra_null_row=True)
        self.use_bucket(p.bucket)
        code, s = self.cli(*p.args(self.root / "adopt-PN", dry_run=True))
        self.assertEqual((code, s["status"]), (0, "DRY_RUN"), s)
        slots = jload(self.root / "adopt-PN" / "adoption.json")["slots"]
        self.assertEqual((slots["unaccounted_slots"], slots["null_identity_unresolved"]), (1, 1))

    def test_retired_slots_reconcile_with_the_terminal_release(self):
        w = T.AdoptWorld(self.root, name="RC", terminal=True, retired_slots=10)
        w.add("ok", "train_fit")
        w.add("ret", "train_fit", retired=True)
        w.finalize()
        self.use_bucket(w.bucket)
        code, s = self.cli(*w.args(self.root / "adopt-RC", dry_run=False))
        self.assertEqual((code, s["failure"]["reason"]), (4, "retired_slots_reconciliation"))
        code, s = self.cli(*w.args(self.root / "adopt-RC-dry", dry_run=True))
        self.assertEqual((code, s["status"]), (0, "DRY_RUN"))
        slots = jload(self.root / "adopt-RC-dry" / "adoption.json")["slots"]
        self.assertEqual((slots["retired_component_slots"], slots["release_retired_component_slots"], slots["reconciled_with_release"]),
                         (5, 10, False))


# ======================================================================================================= AB-6
class FoundationRecipe(T.P6Case):
    def test_the_permutation_must_be_the_seed_20260707_one(self):
        w = T.AdoptWorld(self.root, name="RP", terminal=True)
        w.add("perm", "train_fit", foundation={"recipe_mutate": lambda r: r.update(context_local_indices=list(range(30)),
                                                                                   query_local_indices=list(range(30, 40)))},
              expect={"foundation": "rejected:recipe_contract_violation:rng_permutation"})
        w.finalize()
        self.use_bucket(w.bucket)
        out = self.root / "adopt-RP"
        code, s = self.cli(*w.args(out, dry_run=False))
        self.assertEqual(code, 0, s)
        row = next(r for r in rows_of(out / "rows/train_fit.jsonl") if r["component"] == "foundation")
        self.assertEqual(row["status"], "rejected:recipe_contract_violation:rng_permutation")
        self.assertEqual(jload(out / "adoption.json")["counts"]["final"], T.counts_of(w.expected(production=True)))


# ======================================================================================================= AB-9
class AdoptedBlockSchema(T.P6Case):
    def test_adopted_manifest_fields_are_enforced(self):
        from jsonschema import Draft202012Validator
        w = T.RunWorld(self.root)
        out = self.root / "a0"
        self.assertEqual(self.cli(*w.run_args(out))[0], 0)
        base = jload(out / "manifests/train_fit.json")
        v = Draft202012Validator(E.SCHEMAS["jope.dl-context-encoding.schema.json"])
        self.assertEqual(list(v.iter_errors(base)), [])
        ref = {"bucket": "b", "key": "k", "version_id": "v", "sha256": h("ref"), "bytes": 1}
        comp = {"component": "hyperion", "weights": ref, "recipe": "gpu-v5-row-sweep-pcg-v1",
                "pins": [{"model_ref": "hyperion_v5_arm_b_identity", "ref": ref},
                         {"model_ref": "hyperion_v5_arm_b_runtime_manifest", "ref": ref}],
                "seeds": {"permutations": 20260623}, "row_caps": {}, "encoder_license": dict(T.LICENSES["hyperion"]),
                "research_only": False}
        block = {"block_id": "hyperion", "producer": {"kind": "adopted", "upstream": E.CODEX_UPSTREAM, "release_ref": ref,
                                                      "index_ref": ref, "packet_ref": ref},
                 "components": [comp], "dtype": "<f4", "shape": [512], "pre_normalization": True}
        good = json.loads(json.dumps(dict(base, arm="A1", abi_id="dl-v1-e0:A1:" + "0" * 16, blocks=[block],
                                          inputs=dict(base["inputs"], adoption_receipt=h("adoption")))))
        self.assertEqual(list(v.iter_errors(good)), [])

        def broken(mutate):
            doc = json.loads(json.dumps(good))
            mutate(doc)
            return list(v.iter_errors(doc))
        bad = {
            "adopted block without its adoption receipt": lambda d: d["inputs"].update(adoption_receipt=None),
            "component without pins": lambda d: d["blocks"][0]["components"][0].pop("pins"),
            "hyperion without its runtime-manifest pin": lambda d: d["blocks"][0]["components"][0]["pins"].pop(),
            "component without seeds": lambda d: d["blocks"][0]["components"][0].pop("seeds"),
            "component without row caps": lambda d: d["blocks"][0]["components"][0].pop("row_caps"),
            "component without its licence": lambda d: d["blocks"][0]["components"][0].pop("encoder_license"),
            "adopted block without components": lambda d: d["blocks"][0].update(components=[]),
            "npz member without shape": lambda d: d["contexts"][0].update(arrays=[{"block_id": "hyperion", "npz_ref": ref,
                                                                                  "member": "vector", "dtype": "<f4"}]),
            "p3 map without a binding": lambda d: d["inputs"]["partition_map"].update(kind="p3"),
            "p3 map without the holdout binding": lambda d: d["inputs"]["partition_map"].update(
                kind="p3", binding=dict(sha256=h("b"), schema=E.MAP_BINDING_SCHEMA, partition_map_sha256=h("m"), epoch_sha256=h("e"),
                                        holdout_sha256=h("hs"), holdout_binding_sha256=h("hb"), holdout_ref=ref,
                                        p3_receipt_sha256=h("p3"), p5_receipt_sha256=None)),
            "a fake map with a binding": lambda d: d["inputs"]["partition_map"].update(binding={"sha256": h("b")}),
            "a manifest claiming VERIFIED": lambda d: d["contexts"][0].update(train_only="VERIFIED_BY_COUNTERFACTUAL"),
            "a dev-only arm in a heldout manifest": lambda d: d.update(arm="LB-replica", partition="heldout",
                                                                        encoder_weight_exposure={}),
        }
        for name, mutate in bad.items():
            self.assertTrue(broken(mutate), name)
        arm_v = Draft202012Validator(E.SCHEMAS["jope.dl-encoding-arm.schema.json"])
        arm = jload(out / "arm.json")
        self.assertEqual(list(arm_v.iter_errors(arm)), [])
        adopted_arm = json.loads(json.dumps(dict(arm, inputs=good["inputs"], blocks=[dict(arm["blocks"][0], producer=block["producer"])])))
        self.assertTrue(list(arm_v.iter_errors(adopted_arm)))                     # no research_only / encoder_licenses
        adopted_arm["blocks"][0].update(research_only=False, encoder_licenses={"hyperion": dict(T.LICENSES["hyperion"])})
        self.assertEqual(list(arm_v.iter_errors(adopted_arm)), [])

    def test_normalizer_schema_binds_p3(self):
        from jsonschema import Draft202012Validator
        w = T.RunWorld(self.root)
        out = self.root / "a0"
        self.assertEqual(self.cli(*w.run_args(out))[0], 0)
        rec = jload(out / "manifests/train_fit.json")["normalizers"][0]
        header = jload(out / f"normalizers/{rec['normalizer_sha256']}.json")
        v = Draft202012Validator(E.SCHEMAS["jope.dl-normalizer.schema.json"])
        self.assertEqual(list(v.iter_errors(header)), [])
        self.assertIsNone(header["fit"]["partition_map_binding_sha256"])
        self.assertTrue(list(v.iter_errors(dict(header, fit=dict(header["fit"], partition_map_kind="p3")))))
        self.assertTrue(list(v.iter_errors(dict(header, fit=dict(header["fit"], partition_map_binding_sha256=h("b"))))))


# ======================================================================================================= ties
class RestatedConstants(T.P6Case):
    def test_p3_constants_match_dltrain_holdout(self):
        from dltrain import holdout as H
        self.assertEqual((E.HOLDOUT_SCHEMA, E.HOLDOUT_BINDING_SCHEMA, E.EXPERIMENT_ID), (H.HOLDOUT_SCHEMA, H.BINDING_SCHEMA, H.EXPERIMENT_ID))
        self.assertEqual(set(E.P3_ASSIGNMENT_ROLES), set(H.ASSIGNMENTS))
        self.assertEqual({k for k, v in E.P3_ASSIGNMENT_ROLES.items() if v == "train_fit"}, set(H.TRAIN_ASSIGNMENTS))
        self.assertEqual(E.PLAN_LAUNCH_RELEASE_REF["sha256"], "a8382194d981d4b89692561adc28e9ac2c8c016a98c293412699de92d6cc5d0b")

    def test_every_test_here_is_a_p3case(self):
        mod = sys.modules[__name__]
        classes = [v for v in vars(mod).values() if isinstance(v, type) and issubclass(v, unittest.TestCase)]
        self.assertTrue(classes)
        for cls in classes:
            self.assertTrue(issubclass(cls, fx.P3Case), cls)


if __name__ == "__main__":
    unittest.main()

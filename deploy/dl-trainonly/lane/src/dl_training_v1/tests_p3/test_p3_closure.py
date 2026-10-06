"""closure: identities, contexts of every schema, aliases, histories and seeds for held, dev, sealed and diag; anything the
inputs cannot enumerate is closure_unknown; every input is pinned; a cross-partition link is a STOP that also blocks
publish; the intake registry is opened read-only; sealed names never appear."""
import json
import sqlite3

from dltrain import common, holdout, s3io
from tests_p3 import fixtures_p3 as fx

COMMITMENT_KEY = s3io.PREFIX_KEY + holdout.COMMITMENT_REL


class Closure(fx.P3Case):
    def setUp(self):
        super().setUp()
        fams = fx.standard_families()
        x0 = fams[fx.fid("x0")]
        x0["members"].append({"identity": "beyondarena:" + fx.SEALED_CANARY, "kind": "beyondarena", "cohort": "external",
                              "display": {"name": fx.SEALED_CANARY}})
        self.inputs = fx.write_inputs(self.tmp / "in", fams)
        self.out, self.bucket = fx.pipeline(self, self.inputs)
        self.h_body = (self.out / "holdout.json").read_bytes()
        self.doc = json.loads(self.h_body)
        self.held0 = self.doc["held"][0]
        self.paths = fx.closure_inputs(self.tmp / "closure-in", self.inputs, extra_dataset_link_family=self.held0)
        self.master_ctx = {v["context_sha256"] for v in json.loads(self.paths["master"].read_bytes())["views"]}

    def args(self, *extra, **kw):
        return fx.closure_args(self.inputs, self.out, self.paths, *extra, **kw)

    def run_closure(self, *extra, **kw):
        code, s = fx.run_cli(self.args(*extra, **kw))
        self.assertEqual(code, 0, s)
        body = (self.out / "closure.json").read_bytes()
        return json.loads(body), body, s

    def test_closure_covers_held_dev_sealed_and_diag(self):
        db_before = fx.sha(self.paths["db"].read_bytes())
        c, body, _ = self.run_closure()
        self.assertEqual(common.canonical(c), body)
        self.assertEqual((c["status"], c["cross_family_links"], c["counts"]["blocking_links"]), ("PASS", [], 0))
        a = self.doc["assignment"]
        self.assertEqual(set(c["families"]), {f for f, v in a.items() if v in ("heldout", "dev", "sealed_excluded")})
        self.assertEqual(c["counts"]["families"], {"dev": 3, "heldout": 3, "sealed_excluded": 3})
        self.assertEqual(set(c["diag"]), set(self.doc["diag"]))
        self.assertEqual((c["holdout_sha256"], c["registry_sha256"]), (fx.sha(self.h_body), self.inputs.registry_sha256))
        for f, fam in c["families"].items():
            rec = self.inputs.families[f]
            got = {e["context_sha256"]: e for e in fam["contexts"]}
            for key in ("contexts", "excluded_contexts", "registry_only_contexts"):
                self.assertTrue(set(rec[key]) <= set(got), (f, key))
            for ctx in rec["contexts"]:
                e = got[ctx]
                self.assertEqual(e["owner"], {"family_id": f, "assignment": a[f]})
                self.assertEqual(e["epoch"]["history_ref"]["sha256"], fx.h("history:" + ctx))
                self.assertEqual(e["frozen_fold0"], rec["frozen_fold0"][ctx])
                self.assertTrue({"registry_contexts", "registry_db_contexts"} <= set(e["sources"]))
                if ctx in self.master_ctx:
                    eq = fx.h("equivalent:" + ctx)
                    self.assertEqual(e["equivalents"][0]["context_sha256"], eq)
                    self.assertIn("master_equivalent", got[eq]["sources"])
                    self.assertIsNone(got[eq]["owner"])                         # not a registry context
            self.assertEqual(sorted(i["identity"] for i in fam["identities"] if i["identity"].startswith("item:")),
                             sorted(m["identity"] for m in rec["members"] if m["identity"].startswith("item:")))
            self.assertEqual(fam["registry_db_items_without_context_rows"], [])
            self.assertEqual(fam["seeds"]["seed_selection_exposed"], False)
        sibling = fx.h("sibling-sample:" + self.held0)
        held = {e["context_sha256"]: e for e in c["families"][self.held0]["contexts"]}
        self.assertIn("epoch_dataset_hash", held[sibling]["sources"])
        self.assertEqual(fx.sha(self.paths["db"].read_bytes()), db_before)     # the registry DB was only read

    def test_closure_unknown_lists_what_the_inputs_cannot_enumerate(self):
        c, _, _ = self.run_closure()
        unknown = c["closure_unknown"]
        kinds = {u["kind"] for u in unknown}
        self.assertTrue({"reuse_cache_reachable_contexts", "context_not_in_epoch", "equivalents_unknown",
                         "external_identity_contexts", "alias_source_contexts", "seed_programs"} <= kinds, kinds)
        no_view = {u["context_sha256"] for u in unknown if u["kind"] == "equivalents_unknown"}
        not_epoch = {u["context_sha256"] for u in unknown if u["kind"] == "context_not_in_epoch"}
        all_entries = [e for fam in c["families"].values() for e in fam["contexts"]] + \
                      [e for d in c["diag"].values() for e in d["contexts"]]
        for e in all_entries:
            self.assertEqual(e["context_sha256"] in no_view, e["context_sha256"] not in self.master_ctx)
            self.assertEqual(e["context_sha256"] in not_epoch, e["epoch"] is None)
        self.assertEqual(c["counts"]["closure_unknown"], dict(sorted({k: sum(u["kind"] == k for u in unknown)
                                                                      for k in kinds}.items())))

    def test_seed_programs_are_closure_unknown_for_every_scope(self):
        """SC-06: per-context seed pools and SEED192 program identities are not P3 inputs, so every closure scope
        (family and diag context) records them as closure_unknown."""
        c, _, _ = self.run_closure()
        seeds = {u["scope"]: u for u in c["closure_unknown"] if u["kind"] == "seed_programs"}
        scopes = {f"family:{f}" for f in c["families"]} | {f"diag:{d}" for d in c["diag"]}
        self.assertEqual(set(seeds), scopes)
        for f, fam in c["families"].items():
            u = seeds[f"family:{f}"]
            self.assertEqual(u["contexts"], len(fam["contexts"]))
            self.assertEqual(u["seeds_from_sources"], self.inputs.families[f]["seeds_from_sources"])
            self.assertRegex(u["reason"], "seed-pool.jsonl")
        for d, entry in c["diag"].items():
            self.assertEqual(seeds[f"diag:{d}"]["contexts"], len(entry["contexts"]))

    def test_sealed_names_and_exposed_fields_never_appear(self):
        c, body, s = self.run_closure()
        self.assertNotIn(fx.SEALED_CANARY.encode(), body)
        self.assertNotIn(fx.SEALED_CANARY.encode(), self.h_body)
        self.assertNotIn(fx.SEALED_CANARY, json.dumps(s))
        for leak in (b"tabpfn35", b"rmse", b"/fixture/beyondarena", b"dataset x0"):
            self.assertNotIn(leak, body)
        x0 = c["families"][fx.fid("x0")]
        self.assertTrue(all("value" not in k and "value_sha256" in k for k in x0["aliases"]["union_keys"]))
        self.assertTrue(all("external_id" not in x for x in x0["aliases"]["external_aliases"]))
        self.assertIn("sha256:" + fx.sha(("beyondarena:" + fx.SEALED_CANARY).encode()),
                      [i["identity"] for i in x0["identities"]])

    def test_inputs_are_recorded_and_the_closure_is_deterministic(self):
        c, body, _ = self.run_closure()
        db = c["inputs"]["registry_db"]
        self.assertEqual((db["mode"], db["poll_seq"]), ("ro+query_only", 3))
        self.assertEqual((c["inputs"]["epoch"]["sha256"], c["inputs"]["epoch"]["pinned_by"]),
                         (fx.sha(self.paths["epoch"].read_bytes()), "argv"))
        self.assertEqual(c["inputs"]["master"]["pinned_by"], "argv")
        self.assertEqual((c["inputs"]["alias_sources"]["sha256"], c["inputs"]["alias_sources"]["pinned_by"]),
                         (self.inputs.doc["alias_sources_sha256"], "registry alias_sources_sha256"))
        self.assertEqual([i["name"] for i in c["inputs"]["alias_sources"]["inputs"]],
                         ["beyondarena-seed-pointers", "v6-named-datasets"])
        code, s = fx.run_cli(self.args())
        self.assertEqual(code, 0, s)
        self.assertTrue(fx.receipt(s)["closure"]["existed"])
        self.assertEqual((self.out / "closure.json").read_bytes(), body)

    def test_mismatched_inputs_stop(self):
        bad_alias = self.tmp / "alias-other.json"
        bad_alias.write_bytes(common.canonical({"schema": "jope.dl-family-alias-sources/v1", "inputs": []}))
        argv = self.args()
        argv[argv.index("--alias-sources") + 1] = bad_alias
        code, s = fx.run_cli(argv)
        self.assertEqual(code, 2, s)
        other = fx.write_inputs(self.tmp / "other", fx.standard_families(n_pool=58))
        argv = self.args()
        argv[argv.index("--registry") + 1] = other.registry
        code, s = fx.run_cli(argv)
        self.assertEqual(code, 2, s)
        code, s = fx.run_cli(self.args("--epoch-sha256", "0" * 64))
        self.assertEqual(code, 2, s)
        self.assertFalse((self.out / "closure.json").exists())

    def test_every_input_needs_a_pin(self):
        """SC-03: no unpinned input. The fixture registry has epoch_ref None and the fixture epoch no quarantine_check,
        so without argv pins the epoch and the master have none: STOP before closure.json exists."""
        for kw, regex in ((dict(pin_epoch=False), "epoch input has no pin"),
                          (dict(pin_master=False), "master input has no pin")):
            code, s = fx.run_cli(self.args(**kw))
            self.assertEqual(code, 2, s)
            self.assertRegex(fx.receipt(s)["error"]["message"], regex)
        code, s = fx.run_cli(self.args("--master-sha256", "NOT-HEX"))
        self.assertEqual(code, 2, s)
        self.assertFalse((self.out / "closure.json").exists())
        with self.assertRaisesRegex(holdout.HoldoutStop, "alias-sources input has no pin"):
            holdout._resolve_pin("alias-sources", None, None, "registry alias_sources_sha256")
        reg = holdout.Registry(path="x", sha256="0" * 64, bytes=1, doc={"alias_sources_sha256": "short"}, families={})
        with self.assertRaisesRegex(holdout.HoldoutStop, "alias_sources_sha256 must be 64"):
            holdout._registry_alias_pin(reg)

    def test_master_is_pinned_by_the_epoch(self):
        epoch = json.loads(self.paths["epoch"].read_bytes())
        for name, qc, want in (("wrong", {"master_manifest": {"sha256": fx.h("not the master")}}, 2),
                               ("malformed", {"master_manifest": {"path": "/x"}}, 2),
                               ("right", {"master_manifest": {"sha256": fx.sha(self.paths["master"].read_bytes())}}, 0)):
            p = self.tmp / f"epoch-{name}.json"
            p.write_bytes(common.canonical(epoch | {"quarantine_check": qc}))
            paths = dict(self.paths, epoch=p)
            code, s = fx.run_cli(fx.closure_args(self.inputs, self.out, paths, pin_master=False))
            self.assertEqual(code, want, (name, s))
        c = json.loads((self.out / "closure.json").read_bytes())
        self.assertEqual((c["inputs"]["master"]["sha256"], c["inputs"]["master"]["pinned_by"]),
                         (fx.sha(self.paths["master"].read_bytes()), "epoch quarantine_check"))

    def test_cross_partition_links_stop_the_closure_and_block_publish(self):
        """LP-2: a held family's closure reaching a train_fit (diag) context through a master equivalent, a train
        context through an epoch dataset_hash join, and a diag context reaching a held context are blocking links."""
        fams, a = self.inputs.families, self.doc["assignment"]
        epoch, master, db_rows = fx.closure_docs(fams, extra_dataset_link_family=self.held0)
        held = self.doc["held"][0]
        hc, hc2 = fams[held]["contexts"][0], fams[held]["contexts"][1]
        d1, d2 = self.doc["diag"][0], self.doc["diag"][1]
        owner = {c: f for f, rec in fams.items() for c in rec["contexts"]}
        train = next(f for f in self.doc["train_fit"] if a[f] == "train_fit" and fams[f]["contexts"]
                     and owner[d1] != f and owner[d2] != f)
        x = fams[train]["contexts"][0]
        views = {v["context_sha256"]: v for v in master["views"]}
        for src, dst in ((hc, d1), (d2, hc2)):
            v = views.setdefault(src, {"context_sha256": src, "equivalent_evaluation_contexts": []})
            v["equivalent_evaluation_contexts"].append({"context_sha256": dst, "prepared_path": "/fixture/x.json"})
        master["views"] = list(views.values())
        for row in epoch["contexts"]:
            if row["context_sha256"] == x:
                row["dataset_hash"] = fams[held]["members"][0]["dataset_hash"]
        paths = fx.write_closure_inputs(self.tmp / "leaky-in", epoch, master, db_rows)
        code, s = fx.run_cli(fx.closure_args(self.inputs, self.out, paths))
        self.assertEqual(code, 2, s)
        r = fx.receipt(s)
        self.assertEqual((r["status"], r["stop_reason"], r["experiment_state_if_bound"]),
                         ("STOP", "cross_partition_closure", "CONTAMINATED"))
        c = json.loads((self.out / "closure.json").read_bytes())
        self.assertEqual(c["status"], "STOP")
        blocking = {(x_["scope"], x_["context_sha256"]): x_ for x_ in c["cross_family_links"] if x_["blocking"]}
        link = blocking[(f"family:{held}", d1)]
        self.assertEqual((link["owner_family"], link["owner_assignment"], link["in_diag"]),
                         (owner[d1], a[owner[d1]], True))
        self.assertIn("master_equivalent", link["sources"])
        link = blocking[(f"family:{held}", x)]
        self.assertEqual((link["owner_family"], link["owner_assignment"]), (train, "train_fit"))
        self.assertIn("epoch_dataset_hash", link["sources"])
        link = blocking[(f"diag:{d2}", hc2)]
        self.assertEqual((link["owner_family"], link["owner_assignment"]), (held, "heldout"))
        self.assertEqual(c["counts"]["blocking_links"], len(blocking))
        code, s = fx.run_cli(fx.publish_args(self.out), clients=fx.fake_clients(self.bucket))
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "cross_partition_closure")
        self.assertEqual(self.bucket.puts(), [COMMITMENT_KEY])

    def test_publish_rederives_blocking_links_from_the_holdout_assignment(self):
        """A closure.json that claims PASS while its owner records show a held context is refused by publish."""
        c, _, _ = self.run_closure()
        d = next(iter(c["diag"]))
        entry = c["diag"][d]["contexts"][0]
        entry["owner"] = {"family_id": self.held0, "assignment": "train_fit"}      # a lie the assignment exposes
        forged = self.tmp / "forged-closure.json"
        forged.write_bytes(common.canonical(c))
        code, s = fx.run_cli(fx.publish_args(self.out, "--closure", forged), clients=fx.fake_clients(self.bucket))
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "cross_partition_closure")
        self.assertEqual(self.bucket.puts(), [COMMITMENT_KEY])

    def test_registry_db_is_opened_read_only(self):
        con = holdout._ro_connect(self.paths["db"])
        try:
            with self.assertRaises(sqlite3.OperationalError):
                con.execute("insert into contexts (context, item_id) values ('x', 'y')")
            self.assertEqual(con.execute("pragma query_only").fetchone()[0], 1)
        finally:
            con.close()
        missing = self.tmp / "absent.sqlite3"
        with self.assertRaises(sqlite3.OperationalError):
            holdout._ro_connect(missing)
        self.assertFalse(missing.exists())
        res = holdout.read_registry_contexts(self.paths["db"], [])
        self.assertEqual((res["poll_seq"], res["rows"]), (3, 0))


class ClosurePinnedByTheRegistry(fx.P3Case):
    """SC-03: the registry's epoch_ref pins the epoch when argv does not; a malformed epoch_ref or a disagreeing argv pin
    is a STOP."""

    def setup_with(self, epoch_ref_fn, base="case"):
        fams = fx.standard_families()
        epoch, master, db_rows = fx.closure_docs(fams)
        paths = fx.write_closure_inputs(self.tmp / base / "closure-in", epoch, master, db_rows)
        e_raw = paths["epoch"].read_bytes()
        inputs = fx.write_inputs(self.tmp / base / "in", fams, epoch_ref=epoch_ref_fn(fx.sha(e_raw), len(e_raw)))
        out, _ = fx.pipeline(self, inputs)
        return inputs, out, paths

    def test_registry_epoch_ref_pins_the_epoch(self):
        inputs, out, paths = self.setup_with(lambda s, n: {"bucket": s3io.BUCKET, "key": f"x/objects/epoch/sha256/{s}.json",
                                                           "version_id": "vEpoch", "sha256": s, "bytes": n})
        code, s = fx.run_cli(fx.closure_args(inputs, out, paths, pin_epoch=False))
        self.assertEqual(code, 0, s)
        c = json.loads((out / "closure.json").read_bytes())
        self.assertEqual(c["inputs"]["epoch"]["pinned_by"], "registry epoch_ref")
        code, s = fx.run_cli(fx.closure_args(inputs, out, paths, "--epoch-sha256", fx.h("other epoch"), pin_epoch=False))
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "differs from the registry epoch_ref pin")

    def test_malformed_epoch_ref_stops(self):
        for name, fn, regex in (("no-sha", lambda s, n: {"key": "x", "version_id": "v"}, "64-hex sha256"),
                                ("bad-key", lambda s, n: {"sha256": s, "key": "x/other.json"}, "content-addressed key"),
                                ("null-version", lambda s, n: {"sha256": s, "version_id": "null"}, "usable version_id"),
                                ("not-object", lambda s, n: s, "64-hex sha256")):
            with self.subTest(name):
                inputs, out, paths = self.setup_with(fn, base=name)
                code, s = fx.run_cli(fx.closure_args(inputs, out, paths, pin_epoch=False))
                self.assertEqual(code, 2, s)
                self.assertRegex(fx.receipt(s)["error"]["message"], regex)
                self.assertFalse((out / "closure.json").exists())

"""load_registry: the P2 registry adapter pins the bytes and rejects every missing or ill-typed field P3 reads."""
import copy

from dltrain import common, holdout
from tests_p3 import fixtures_p3 as fx


class LoadRegistry(fx.P3Case):
    def setUp(self):
        super().setUp()
        self.doc = fx.simple_registry_doc()
        self.fid = fx.fid("p001")          # an auto, incoming, single-dataset pool family

    def write(self, doc_or_bytes):
        raw = doc_or_bytes if isinstance(doc_or_bytes, bytes) else common.canonical(doc_or_bytes)
        p = self.tmp / f"registry-{fx.sha(raw)[:16]}.json"
        p.write_bytes(raw)
        return p, fx.sha(raw)

    def rejects(self, doc_or_bytes, regex, **kw):
        p, s = self.write(doc_or_bytes)
        with self.assertRaisesRegex(holdout.RegistryError, regex):
            holdout.load_registry(p, s, **kw)

    def mutated(self, fn):
        doc = copy.deepcopy(self.doc)
        fn(doc)
        return doc

    def test_valid_registry_loads(self):
        p, s = self.write(self.doc)
        reg = holdout.load_registry(p, s)
        self.assertEqual((reg.sha256, len(reg.families)), (s, 72))
        self.assertEqual(len(holdout.pool_members(reg.families)), 60)
        self.assertTrue(issubclass(holdout.RegistryError, holdout.HoldoutStop))

    def test_pin_and_parse(self):
        p, s = self.write(self.doc)
        with self.assertRaisesRegex(holdout.RegistryError, "sha256"):
            holdout.load_registry(p, "0" * 64)
        with self.assertRaisesRegex(holdout.RegistryError, "64 lowercase hex"):
            holdout.load_registry(p, s.upper())
        self.rejects(b"{not json", "not strict JSON")
        self.rejects(b'{"a":1,"a":2}', "duplicate JSON key")
        self.rejects(b'{"a":NaN}', "non-finite")
        self.rejects(b"[]", "not a JSON object")

    def test_top_level(self):
        self.rejects(self.mutated(lambda d: d.update(schema="jope.dl-family-registry/v0")), "schema")
        self.rejects(self.mutated(lambda d: d.update(experiment_id="other")), "experiment_id")
        pre = self.mutated(lambda d: d.update(phase="pre_review"))
        self.rejects(pre, "final")
        p, s = self.write(pre)
        self.assertEqual(holdout.load_registry(p, s, require_final=False).doc["phase"], "pre_review")
        self.rejects(self.mutated(lambda d: d.update(phase="draft")), "phase", require_final=False)
        self.rejects(self.mutated(lambda d: d.pop("review_decisions_sha256")), "review_decisions_sha256")
        self.rejects(self.mutated(lambda d: d.update(families={})), "non-empty")
        self.rejects(self.mutated(lambda d: d.update(families=[])), "non-empty")

    def test_each_required_field_missing(self):
        for field in holdout.REQUIRED_FAMILY_FIELDS:
            self.rejects(self.mutated(lambda d, k=field: d["families"][self.fid].pop(k)), f"missing {field}")

    def test_ill_typed_fields(self):
        bad = (("sealed", "false"), ("sealed", 0), ("sealed", None), ("seed_selection_exposed", 1),
               ("seed_selection_exposed", "no"), ("rights_status", "recorded:"), ("rights_status", "forbidden"),
               ("rights_status", None), ("rights_status", "recorded: MIT"), ("independence_qualified", "maybe"),
               ("independence_qualified", ["auto"]), ("independence_qualified", None), ("contexts", "abc"),
               ("contexts", None), ("frozen_fold0", []), ("members", []), ("members", "item:x"), ("family_id", 7))
        for field, value in bad:
            self.rejects(self.mutated(lambda d, k=field, v=value: d["families"][self.fid].update({k: v})), self.fid[:12])

    def test_contexts_and_frozen_fold0(self):
        rec = self.doc["families"][self.fid]
        c0 = rec["contexts"][0]
        self.rejects(self.mutated(lambda d: d["families"][self.fid].update(contexts=list(reversed(rec["contexts"])))),
                     "sorted and unique")
        self.rejects(self.mutated(lambda d: d["families"][self.fid].update(contexts=sorted(rec["contexts"] + [c0]))),
                     "sorted and unique")
        self.rejects(self.mutated(lambda d: d["families"][self.fid].update(contexts=[c0.upper()] + rec["contexts"][1:])),
                     "64-hex")
        self.rejects(self.mutated(lambda d: d["families"][self.fid]["frozen_fold0"].pop(c0)), "keys must equal contexts")
        self.rejects(self.mutated(lambda d: d["families"][self.fid]["frozen_fold0"].update({c0: "yes"})), "to bool")
        self.rejects(self.mutated(lambda d: d["families"][self.fid]["frozen_fold0"].update({fx.h("extra"): False})),
                     "keys must equal contexts")

    def test_members_and_identities(self):
        self.rejects(self.mutated(lambda d: d["families"][self.fid]["members"][0].pop("identity")), "identity")
        self.rejects(self.mutated(lambda d: d["families"][self.fid]["members"][0].update(identity="")), "identity")
        self.rejects(self.mutated(lambda d: d["families"][self.fid]["members"].append(
            copy.deepcopy(d["families"][self.fid]["members"][0]))), "duplicate member identity")
        other = fx.fid("p002")

        def steal_identity(d):
            d["families"][other]["members"].append(copy.deepcopy(d["families"][self.fid]["members"][0]))
        self.rejects(self.mutated(steal_identity), "is in families")
        self.rejects(self.mutated(lambda d: d["identities"].update({"item:" + fx.h("ghost"): self.fid})), "identities map")
        self.rejects(self.mutated(lambda d: d.update(identities=[])), "identities must map")

    def test_cross_family_context_and_ids(self):
        other = fx.fid("p002")

        def share_context(d):
            c = d["families"][self.fid]["contexts"][0]
            o = d["families"][other]
            o["contexts"] = sorted(o["contexts"] + [c])
            o["frozen_fold0"][c] = False
        self.rejects(self.mutated(share_context), "is in families")

        def rekey(d):
            d["families"][fx.fid("renamed")] = d["families"].pop(self.fid)
        self.rejects(self.mutated(rekey), "differs from its key")

        def bad_pattern(d):
            rec = d["families"].pop(self.fid)
            rec["family_id"] = "fam-00"
            d["families"]["fam-00"] = rec
        self.rejects(self.mutated(bad_pattern), "64 lowercase hex")

    def test_qualified_contextless_family_and_retired_ids(self):
        def contextless(d):
            d["families"][self.fid].update(contexts=[], frozen_fold0={})
        self.rejects(self.mutated(contextless), "without contexts")
        p, s = self.write(self.mutated(lambda d: (contextless(d), d["families"][self.fid].update(
            independence_qualified="unresolved"))))
        reg = holdout.load_registry(p, s)                        # an unresolved contextless family is legal
        self.assertNotIn(self.fid, holdout.pool_members(reg.families))
        self.rejects(self.mutated(lambda d: d.update(retired_family_ids={self.fid: fx.fid("p002")})), "retired")
        self.rejects(self.mutated(lambda d: d.update(retired_family_ids={fx.fid("old"): fx.fid("nowhere")})), "retired")
        self.rejects(self.mutated(lambda d: d.update(retired_family_ids=[])), "retired_family_ids")
        p, s = self.write(self.mutated(lambda d: d.update(retired_family_ids={fx.fid("old"): self.fid})))
        holdout.load_registry(p, s)

    def test_pool_predicate(self):
        fams = self.doc["families"]
        for tag, want in fx.EXPECTED_NONPOOL.items():
            self.assertFalse(holdout.pool_predicate(fams[fx.fid(tag)]), tag)
        self.assertTrue(holdout.pool_predicate(fams[self.fid]))
        reviewed = fams[fx.fid("p000")]
        self.assertEqual(reviewed["independence_qualified"], "reviewed")
        self.assertTrue(holdout.pool_predicate(reviewed))
        recorded = dict(fams[self.fid], rights_status="recorded:MIT")
        self.assertTrue(holdout.pool_predicate(recorded))

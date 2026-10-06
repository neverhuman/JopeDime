"""Read-only real-data checks of dltrain.scale_check: the spec A oracle fold replicates Codex bit for bit, the spec B lineage
proofs hold, and the replica reproduces spec B's independent 4-context probe (anchor, U-control, H-floor, F-severe) number
for number. Skipped when the local data is absent. Nothing is written."""
import unittest

from dltrain import scale_check as sc

ORACLE_VIEW = "147d3a6b5f70b309__r0__f3__test_adaptive_3249606db8ee"   # spec A FIDELITY fold (incoming, n=1614, k=5)
ANCHOR_VIEW = sc.ANCHOR_VIEW                                           # the UPSTREAM-SCALE-01 context
PROBE = {"U-control": "42cce7c47356", "H-floor": "2e0bc6eb25da", "F-severe": "256287f8bef8"}   # spec B EXPECTED


@unittest.skipUnless(sc.EPOCH_PATH.exists() and (sc.CODEX_ROOT / "packet.json").exists(), "local data absent")
class TestRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = sc.Sources()
        cls.rows = {r["view_id"]: r for r in cls.sources.included()}
        cls.by_prefix = {r["context_sha256"][:12]: r for r in cls.rows.values()}
        cls.cache = sc.DataCache()

    def analyse(self, row, **kw):
        ci = sc.resolve_context(row, self.sources, self.cache)
        return ci, sc.analyse(ci, self.sources, **kw)

    def assert_lineage(self, res):
        self.assertTrue(res["lineage"], res["lineage"])
        self.assertTrue(all(v is True for v in res["lineage"].values()), res["lineage"])

    def test_oracle_fold_replicates_codex(self):
        ci, res = self.analyse(self.rows[ORACLE_VIEW], positions=0, want_a0=False, want_replication=True)
        self.assertEqual(len(ci["R"]), 1614)
        self.assertTrue(res["replication"]["pass"], res["replication"])
        self.assertEqual(res["replication"]["checks"], 29)
        self.assertTrue(res["local"])
        self.assertEqual(ci["owner"], sc.LOCAL_NODE)
        self.assert_lineage(res)
        self.assertIn("packet_train_row_ids_match", res["lineage"])

    def test_anchor_reproduces_the_spec_b_pilot(self):
        ci, res = self.analyse(self.rows[ANCHOR_VIEW], positions=256, want_a0=True, want_replication=True)
        self.assertEqual((len(ci["R"]), len(ci["T"]), ci["A"].shape), (1600, 399, (1999, 21)))
        self.assert_lineage(res)
        self.assertTrue(res["lineage"]["jtt1_cache_equals_rebuild"])
        self.assertTrue(res["lineage"]["jtt1_cache_train_view_equal"])
        self.assertIsNone(res["replication"]["pass"])                       # outcomes live on another shard
        self.assertEqual(ci["owner"], "xbabe3")
        cmp = res["comparisons"]
        b = cmp["B_vs_A"]["components"]
        self.assertEqual(b["hyperion"]["class"], "IDENTICAL")                 # B leaves u and target v identical
        self.assertEqual(b["target"]["class"], "IDENTICAL")
        self.assertEqual(b["foundation"]["yz"]["differ"], 0)
        self.assertEqual(b["foundation"]["class"], "NUMERICAL")
        self.assertEqual((b["foundation"]["x_ctx"]["differ"], b["foundation"]["x_ctx"]["max_ulp"]), (216, 1))
        self.assertEqual((b["foundation"]["x_qry"]["differ"], b["foundation"]["x_qry"]["max_ulp"]), (67, 1))
        self.assertEqual(b["tabpfn35"]["class"], "STRUCTURAL")
        self.assertEqual(set(b["tabpfn35"]["X_ctx"]["differ_by_column"]), {"6"})   # only the flagged column moves
        c = cmp["C_vs_A"]["components"]
        self.assertEqual((c["hyperion"]["u"]["flips"], c["hyperion"]["u"]["flips_ge2"]), (1023, 2))
        self.assertEqual(c["hyperion"]["u"]["floor_column_flips"], 2)
        self.assertEqual(c["foundation"]["yz"]["max_ulp"], 358)
        self.assertEqual(max(c["target"]["classical"]["max_ulp"], c["target"]["neural"]["max_ulp"]), 2)
        self.assertEqual({k: v["class"] for k, v in cmp["C2_vs_A"]["components"].items()},
                         {k: v["class"] for k, v in cmp["B_vs_A"]["components"].items()})
        self.assertEqual(cmp["A1_vs_A"]["class"], "IDENTICAL")                # every column exactly [0,1] over all rows
        for k in sc.FIX_GATE_KEYS:
            self.assertEqual(cmp[k]["class"], "IDENTICAL", k)
        self.assertEqual(res["prescan"]["flagged_cols"], [6])
        self.assertAlmostEqual(res["prescan"]["severity"], 1 / 58, places=9)
        a0 = res["a0"]
        self.assertIn(a0["status"], ("computed", "skipped"))
        if a0["status"] == "computed":
            self.assertTrue(a0["from_raw_equals_direct"])
        train = set(int(r) for r in ci["R"])
        for key, comp in cmp.items():
            for rec in comp["components"].values():
                for arr in rec.values():
                    if isinstance(arr, dict):
                        for p in arr.get("positions", []):
                            self.assertIn(p[0], train, key)

    def test_r5_row_count_correction(self):
        ci = sc.resolve_context(self.rows[ANCHOR_VIEW], self.sources, self.cache)
        corr = sc.refs_correction(ci)
        self.assertEqual([(c["recorded"], c["observed"], c["correct"]) for c in corr["corrections"]],
                         [(1998, 1999, False), (1998, 1999, False)])
        self.assertEqual(corr["column_claims"]["columns_exactly_0_1_over_all_rows"], 21)
        self.assertEqual(corr["column_claims"]["fold0_train_columns_not_reaching_both_extremes"], [6])
        self.assertAlmostEqual(corr["column_claims"]["fold0_train_min_of_those_columns"][0], 1 / 59, places=8)

    def test_spec_b_probe_contexts(self):
        got = {}
        for cat, prefix in PROBE.items():
            ci, res = self.analyse(self.by_prefix[prefix], positions=0, want_a0=False, want_replication=True)
            self.assert_lineage(res)
            self.assertTrue(res["replication"]["pass"], (cat, res["replication"]))
            for k in sc.FIX_GATE_KEYS:
                self.assertEqual(res["comparisons"][k]["class"], "IDENTICAL", (cat, k))
            got[cat] = res["comparisons"]
        u = got["U-control"]
        self.assertEqual((u["B_vs_A"]["class"], u["C2_vs_A"]["class"]), ("IDENTICAL", "IDENTICAL"))
        self.assertNotEqual(u["C_vs_A"]["class"], "IDENTICAL")
        h = got["H-floor"]["B_vs_A"]["components"]["hyperion"]
        self.assertEqual((h["u"]["flips"], h["u"]["flips_ge2"], h["u"]["floor_column_flips"]), (1524, 14, 14))
        self.assertEqual(h["class"], "STRUCTURAL")
        f = got["F-severe"]
        self.assertEqual(f["B_vs_A"]["components"]["foundation"]["yz"]["max_ulp"], 11510)
        self.assertEqual(f["C_vs_A"]["components"]["foundation"]["x_ctx"]["max_ulp"], 32345)

    def test_s_floor_context_has_a_reachable_foundation_floor(self):
        ci = sc.resolve_context(self.rows[sc.S_FLOOR_SPEC[0]], self.sources, self.cache)
        ps = sc.prescan(ci, sc.codex_inputs(ci["A"][ci["R"]]))
        self.assertGreater(ps["foundation_floor_risk"], 0)
        self.assertNotIn(ci["row"]["context_sha256"], self.sources.sealed_ctx)

    def test_sealed_contexts_are_known_and_excluded_from_U_inputs(self):
        self.assertGreater(len(self.sources.sealed_ctx), 0)
        self.assertNotIn(sc.ANCHOR_CONTEXT, self.sources.sealed_ctx)
        self.assertEqual(len(sc.SPEC_B_SAMPLE_48), 48)
        self.assertFalse(set(sc.SPEC_B_SAMPLE_48) & self.sources.sealed_ctx)
        self.assertTrue(self.sources.registry_is_pre_review)
        self.assertEqual(self.sources.index_ref["keys"], sc.SPEC_B_INDEX["keys"])


if __name__ == "__main__":
    unittest.main()

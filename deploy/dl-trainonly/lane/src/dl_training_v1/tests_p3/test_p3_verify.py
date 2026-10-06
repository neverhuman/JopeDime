"""freeze and verify through the CLI, the holdout.json content, and the assignment-poison test (honest perturbations
pass; planted dependencies of dev, diag or tau on held-family records are detected)."""
import json
import math
from unittest import mock

from dltrain import common, holdout
from tests_p3 import fixtures_p3 as fx


def leaky_diag_freeze(registry, pool, seed, frac=0.05):
    """Planted dependency: the diagnostic is drawn over every pool context, held families included."""
    out = holdout.freeze(registry, pool, seed, frac)
    ctx = sorted(c for f in sorted(set(pool)) for c in registry[f]["contexts"] if not registry[f]["frozen_fold0"].get(c))
    out["diag"] = sorted(ctx, key=lambda c: (holdout.prio(seed, c, holdout.DIAG_DOMAIN), c))[:math.ceil(frac * len(ctx))]
    return out


def leaky_tau_freeze(registry, pool, seed, frac=0.05):
    """Planted dependency: tau mixes in the held families' context count."""
    out = holdout.freeze(registry, pool, seed, frac)
    n = sum(len(registry[f]["contexts"]) for f in out["held"])
    out["tau_hex"] = f"{holdout.prio(seed, out['held'][-1] + ':' + str(n)):064x}"
    return out


def leaky_dev_freeze(registry, pool, seed, frac=0.05):
    """Planted dependency: the dev draw is salted with the held families' member dataset hashes."""
    out = holdout.freeze(registry, pool, seed, frac)
    salt = "|".join(sorted(str(m.get("dataset_hash")) for f in out["held"] for m in registry[f]["members"]))
    rest = sorted(set(pool) - set(out["held"]))
    out["dev"] = sorted(rest, key=lambda f: (holdout.prio(seed, f + salt, holdout.DEV_DOMAIN), f))[:len(out["dev"])]
    return out


def pool_field_dev_freeze(registry, pool, seed, frac=0.05):
    """Planted dependency (SC-02): dev salted with held families' rights_status and independence_qualified, values that
    can change while the family stays in the pool."""
    out = holdout.freeze(registry, pool, seed, frac)
    salt = "|".join(registry[f]["rights_status"] + ":" + registry[f]["independence_qualified"] for f in out["held"])
    rest = sorted(set(pool) - set(out["held"]))
    out["dev"] = sorted(rest, key=lambda f: (holdout.prio(seed, f + salt, holdout.DEV_DOMAIN), f))[:len(out["dev"])]
    return out


def member_salted_diag_freeze(field):
    """Planted dependency (LP-3): the diag draw salted with one member field of the held families."""
    def fz(registry, pool, seed, frac=0.05):
        out = holdout.freeze(registry, pool, seed, frac)
        salt = "|".join(sorted(json.dumps(m.get(field), sort_keys=True) for f in out["held"]
                               for m in registry[f]["members"]))
        ctx = sorted(c for f in out["train_fit"] for c in registry[f]["contexts"] if not registry[f]["frozen_fold0"].get(c))
        out["diag"] = sorted(ctx, key=lambda c: (holdout.prio(seed, c + salt, holdout.DIAG_DOMAIN), c))[:len(out["diag"])]
        return out
    return fz


class FreezeAndVerify(fx.P3Case):
    def setUp(self):
        super().setUp()
        self.inputs = fx.write_inputs(self.tmp / "in")
        self.out, self.bucket = fx.pipeline(self, self.inputs, verify_too=False)
        self.body = (self.out / "holdout.json").read_bytes()
        self.doc = json.loads(self.body)

    def test_holdout_json_content(self):
        d = self.doc
        self.assertEqual(common.canonical(d), self.body)
        self.assertEqual((d["schema"], d["experiment_id"], d["estimand"]),
                         ("jope.dl-family-holdout/v1", "gp-policy-dl-v1-e0", "qualified, unsealed, non-seed-exposed families only"))
        self.assertEqual((d["N"], d["k"], len(d["dev"])), (60, 3, 3))
        self.assertEqual(d["counts"]["assignment"], {"dev": 3, "heldout": 3, "prohibited_excluded": 2,
                                                     "sealed_excluded": 3, "train_fit": 54, "train_nonpool": 7})
        self.assertEqual(d["registry"]["sha256"], self.inputs.registry_sha256)
        gate = json.loads((self.out / "gate.json").read_bytes())
        self.assertEqual(d["registry"]["s3"], gate["registry_s3"])                     # SC-01: the registry's S3 ref
        self.assertEqual(d["gate"], {"p2_receipt_sha256": fx.sha(self.inputs.p2_receipt.read_bytes()),
                                     "owner_decisions_sha256": self.inputs.owner_decisions_sha256,
                                     "decisions": holdout.GATE_DECISION_VALUES})
        ref = json.loads((self.out / "commitment-ref.json").read_bytes())
        self.assertEqual(d["commitment"], {k: ref[k] for k in ("bucket", "key", "version_id", "sha256", "bytes")})
        self.assertEqual(d["seed_sha256"], fx.sha(holdout.read_seed(self.out / "seed.hex")))
        rf = d["realized_fractions"]
        self.assertEqual(rf["families"]["heldout_over_pool"], 3 / 60)
        self.assertEqual(rf["folds"]["diag"], len(d["diag"]))
        train_ctx = sum(len(self.inputs.families[f]["contexts"]) for f in d["train_fit"])
        self.assertEqual(rf["folds"]["training"], train_ctx - len(d["diag"]))          # SC-04: diag is not training
        self.assertEqual(rf["programs"]["status"], "not_computed")
        self.assertEqual(set(d["composition"]), set(holdout.COMPOSITION_GROUPS))
        self.assertEqual(d["composition"]["sealed_excluded"]["platform"], {"withheld": 3})
        self.assertEqual(d["composition"]["pool"]["families"], 60)
        members = {m["identity"]: f for f, rec in self.inputs.families.items() for m in rec["members"]}
        self.assertEqual(d["member_assignment"], {i: d["assignment"][f] for i, f in members.items()})
        self.assertNotIn(fx.SEALED_CANARY.encode(), self.body)

    def test_freeze_receipt_says_which_expect_items_are_missing(self):
        """SC-04: without per-context statistics the receipt must not claim the PLAN §5 P3 Expect list is met."""
        code, s = fx.run_cli(fx.freeze_args(self.inputs, self.out))
        self.assertEqual(code, 0, s)
        e = fx.receipt(s)["expect"]
        self.assertEqual((e["complete"], e["k_exact"], e["every_family_exactly_one_assignment"]), (False, True, True))
        self.assertEqual(e["missing"], list(holdout.EXPECT_STATS_ITEMS.values()))
        self.assertEqual(e["estimand"], holdout.ESTIMAND)

    def test_dataset_grain_counts_datasets_with_included_contexts(self):
        """SC-04: withdrawn-only (or sibling, external) members do not count at the dataset grain; member_datasets
        reports every member."""
        fams = json.loads(common.canonical(self.inputs.families))
        f = self.doc["held"][0]
        fams[f]["members"].append({"identity": "item:" + fx.h("withdrawn-member"), "kind": "registry_item",
                                   "dataset_hash": fx.h("withdrawn-dataset")[:16], "epoch_status": "withdrawn_only"})
        res = holdout.freeze(fams, holdout.pool_members(fams), holdout.read_seed(self.out / "seed.hex"))
        rf = holdout._realized_fractions(fams, res, holdout._groups(res))
        self.assertEqual(rf["member_datasets"]["heldout"], rf["datasets"]["heldout"] + 1)
        self.assertEqual(rf["member_datasets"]["all"], rf["datasets"]["all"] + 1)

    def test_freeze_rerun_is_byte_identical_and_create_only(self):
        code, s = fx.run_cli(fx.freeze_args(self.inputs, self.out))
        self.assertEqual(code, 0, s)
        self.assertTrue(fx.receipt(s)["outputs"]["holdout"]["existed"])
        self.assertEqual((self.out / "holdout.json").read_bytes(), self.body)

    def test_freeze_refuses_a_seed_that_does_not_derive_from_the_commitment(self):
        alt = self.tmp / "alt"
        fx.commit_files(self.out, alt)
        (alt / "seed.hex").write_text("ab" * 32 + "\n")
        code, s = fx.run_cli(["freeze", "--registry", self.inputs.registry, "--registry-sha256", self.inputs.registry_sha256,
                              "--seed-file", alt / "seed.hex", "--out", alt])
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "does not derive")
        self.assertFalse((alt / "holdout.json").exists())
        (alt / "bad.hex").write_text("AB" * 32 + "\n")
        with self.assertRaises(holdout.HoldoutStop):
            holdout.read_seed(alt / "bad.hex")

    def test_freeze_refuses_a_registry_the_commitment_does_not_bind(self):
        other = fx.write_inputs(self.tmp / "other", fx.standard_families(n_pool=61))
        code, s = fx.run_cli(["freeze", "--registry", other.registry, "--registry-sha256", other.registry_sha256,
                              "--seed-file", self.out / "seed.hex", "--out", self.tmp / "o2"])
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "another registry|registry_sha256")

    def test_freeze_needs_the_recorded_gate(self):
        alt = self.tmp / "no-gate"
        fx.commit_files(self.out, alt)
        (alt / "seed.hex").write_bytes((self.out / "seed.hex").read_bytes())
        (alt / "gate.json").unlink()
        code, s = fx.run_cli(fx.freeze_args(self.inputs, alt))
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "gate.json is missing")
        self.inputs.owner_decisions.write_bytes(b"{}")                  # the named owner decisions changed afterwards
        for argv in (fx.freeze_args(self.inputs, self.out), fx.verify_args(self.inputs, self.out)):
            code, s = fx.run_cli(argv)
            self.assertEqual(code, 2, s)
            self.assertRegex(fx.receipt(s)["error"]["message"], "sha256 .* != pinned")

    def test_verify_with_assignment_poison_passes(self):
        code, s = fx.run_cli(fx.verify_args(self.inputs, self.out))
        self.assertEqual(code, 0, s)
        r = fx.receipt(s)
        self.assertEqual((r["status"], r["bytes_equal"], r["field_diffs"], r["failed_checks"]), ("PASS", True, [], []))
        self.assertEqual(r["holdout_sha256"], fx.sha(self.body))
        self.assertTrue(r["checks"]["diag_is_prio_prefix"] and r["checks"]["train_fit_exact"])
        poison = r["assignment_poison"]
        self.assertTrue(poison["ran"] and poison["pass"])
        self.assertEqual([v["variant"] for v in poison["variants"]], [v[0] for v in holdout.POISON_VARIANTS])
        self.assertTrue(all(v["changed"] == [] and v["families"] > 0 for v in poison["variants"]))
        self.assertEqual(poison["spec_keys"], ["dev", "diag", "tau_hex", "tau_dev_hex", "pool_sha256"])

    def test_verify_detects_a_tampered_holdout(self):
        d = json.loads(self.body)
        h0, d0 = d["held"][0], d["dev"][0]
        d["held"] = sorted(d["held"][1:] + [d0])
        d["dev"] = sorted(d["dev"][1:] + [h0])
        d["assignment"][h0], d["assignment"][d0] = "dev", "heldout"
        tampered = self.tmp / "tampered" / "holdout.json"
        tampered.parent.mkdir()
        tampered.write_bytes(common.canonical(d))
        self.assertEqual(holdout.holdout_doc_problems(d), [])              # structurally fine, so only a recompute sees it
        code, s = fx.run_cli(["verify", "--holdout", tampered, "--registry", self.inputs.registry,
                              "--seed-file", self.out / "seed.hex"])
        self.assertEqual(code, 2, s)
        r = fx.receipt(s)
        self.assertEqual(r["status"], "STOP")
        self.assertIn("held", r["field_diffs"])
        self.assertFalse(r["bytes_equal"])

    def test_verify_refuses_another_registry_or_seed(self):
        other = fx.write_inputs(self.tmp / "other", fx.standard_families(n_pool=59))
        code, s = fx.run_cli(["verify", "--holdout", self.out / "holdout.json", "--registry", other.registry,
                              "--seed-file", self.out / "seed.hex"])
        self.assertEqual(code, 2, s)
        code, s = fx.run_cli(["verify", "--holdout", self.out / "holdout.json", "--registry", self.inputs.registry,
                              "--registry-sha256", other.registry_sha256, "--seed-file", self.out / "seed.hex"])
        self.assertEqual(code, 2, s)
        alt = self.tmp / "alt"
        fx.commit_files(self.out, alt)
        (alt / "seed.hex").write_text("cd" * 32 + "\n")
        code, s = fx.run_cli(["verify", "--holdout", self.out / "holdout.json", "--registry", self.inputs.registry,
                              "--seed-file", alt / "seed.hex", "--assignment-poison"])
        self.assertEqual(code, 2, s)
        self.assertRegex(fx.receipt(s)["error"]["message"], "does not derive")

    def test_context_stats_add_program_and_record_grains(self):
        contexts = sorted(c for rec in self.inputs.families.values() for c in rec["contexts"])
        stats = {"schema": "jope.dl-context-stats/v1",
                 "contexts": {c: {"programs": 100 + i % 7, "records": 1000 + i % 11, "n": 50 + i % 13, "p": 3 + i % 5,
                                  "target_scale": 0.5 + (i % 9) / 4} for i, c in enumerate(contexts)}}
        sp = self.tmp / "stats.json"
        sp.write_bytes(common.canonical(stats))
        out2 = self.tmp / "with-stats"
        fx.commit_files(self.out, out2)
        (out2 / "seed.hex").write_bytes((self.out / "seed.hex").read_bytes())
        code, s = fx.run_cli(["freeze", "--registry", self.inputs.registry, "--registry-sha256", self.inputs.registry_sha256,
                              "--seed-file", out2 / "seed.hex", "--out", out2, "--context-stats", sp,
                              "--context-stats-sha256", fx.sha(sp.read_bytes())])
        self.assertEqual(code, 0, s)
        self.assertEqual((out2 / "holdout.json").read_bytes(), self.body)   # the bound manifest ignores optional stats
        e = fx.receipt(s)["expect"]
        self.assertEqual((e["complete"], e["missing"]), (True, []))          # SC-04: full coverage meets the Expect
        rep = json.loads((out2 / f"holdout-stats-{fx.sha(sp.read_bytes())[:16]}.json").read_bytes())
        self.assertEqual(rep["holdout_sha256"], fx.sha(self.body))
        held_ctx = [c for f in self.doc["held"] for c in self.inputs.families[f]["contexts"]]
        want = sum(stats["contexts"][c]["programs"] for c in held_ctx)
        self.assertEqual(rep["realized_fractions"]["programs"]["heldout"], want)
        self.assertEqual(rep["coverage"]["all"], {"contexts": len(contexts), "with_stats": len(contexts)})
        self.assertEqual(set(rep["composition_quantiles"]["pool"]), {"n", "p", "target_scale"})
        partial = {"schema": "jope.dl-context-stats/v1",
                   "contexts": {c: {k: v for k, v in x.items() if k != "records"} for c, x in stats["contexts"].items()}}
        pp = self.tmp / "partial.json"
        pp.write_bytes(common.canonical(partial))
        code, s = fx.run_cli(["freeze", "--registry", self.inputs.registry, "--registry-sha256", self.inputs.registry_sha256,
                              "--seed-file", out2 / "seed.hex", "--out", out2, "--context-stats", pp,
                              "--context-stats-sha256", fx.sha(pp.read_bytes())])
        self.assertEqual(code, 0, s)
        self.assertEqual(fx.receipt(s)["expect"]["missing"], ["realized_fractions.records"])
        code, s = fx.run_cli(["freeze", "--registry", self.inputs.registry, "--registry-sha256", self.inputs.registry_sha256,
                              "--seed-file", out2 / "seed.hex", "--out", out2, "--context-stats", sp,
                              "--context-stats-sha256", "0" * 64])
        self.assertEqual(code, 2, s)


class SelectionCodeIdentity(fx.P3Case):
    """SC-05: the commitment binds the selection code itself, and every receipt names the code and runtime."""

    def test_code_sha256_covers_the_selection_units(self):
        self.assertEqual(holdout.ALGORITHM["code_sha256"], fx.sha(common.canonical(holdout.selection_code())))
        units = holdout.selection_code()["units"]
        self.assertEqual(set(units), set(holdout.SELECTION_CODE_UNITS) | {"common.canonical"})
        self.assertIn("def freeze(registry", units["freeze"])
        self.assertIn("def pool_predicate(fam", units["pool_predicate"])
        real_freeze = holdout.freeze

        def edited_freeze(registry, pool, seed, frac=0.05):       # same behaviour, different source
            return real_freeze(registry, pool, seed, frac)
        with mock.patch.object(holdout, "freeze", edited_freeze):
            self.assertNotEqual(holdout.selection_code_sha256(), holdout.ALGORITHM["code_sha256"])
        self.assertEqual(holdout.selection_code_sha256(), holdout.ALGORITHM["code_sha256"])

    def test_a_code_change_after_the_commitment_stops_freeze_verify_and_publish(self):
        inputs = fx.write_inputs(self.tmp / "in")
        out, bucket, _ = fx.publish_ready(self, inputs)
        o2 = self.tmp / "o2"
        fx.commit_files(out, o2)
        (o2 / "seed.hex").write_bytes((out / "seed.hex").read_bytes())
        changed = holdout.ALGORITHM | {"code_sha256": fx.h("edited selection code")}
        with mock.patch.object(holdout, "ALGORITHM", changed):
            code, s = fx.run_cli(fx.freeze_args(inputs, o2))
            self.assertEqual(code, 2, s)
            self.assertRegex(fx.receipt(s)["error"]["message"], "selection code changed")
            self.assertFalse((o2 / "holdout.json").exists())
            code, s = fx.run_cli(fx.verify_args(inputs, out))
            self.assertEqual(code, 2, s)
            self.assertRegex(fx.receipt(s)["error"]["message"], "selection code changed")
            code, s = fx.run_cli(fx.publish_args(out), clients=fx.fake_clients(bucket))
            self.assertEqual(code, 2, s)
            self.assertRegex(fx.receipt(s)["error"]["message"], "algorithm does not match")
        self.assertEqual(bucket.puts(), [fx.s3io.PREFIX_KEY + holdout.COMMITMENT_REL])

    def test_every_receipt_names_the_source_tree_and_runtime(self):
        inputs = fx.write_inputs(self.tmp / "in")
        code, s = fx.run_cli(fx.commit_args(inputs, self.tmp / "holdout"),
                             clients=fx.fake_clients(fx.registry_bucket(inputs)))
        self.assertEqual(code, 0, s)
        r = fx.receipt(s)
        self.assertEqual(r["source_tree_sha256"], holdout.source_tree_sha256())
        # review fix P3I-1: the review gate applies families.check_provenance, so the P2 modules join the closure
        self.assertEqual([p for p, _ in r["source_tree"]],
                         ["dltrain/__init__.py", "dltrain/common.py", "dltrain/families.py", "dltrain/family_keys.py",
                          "dltrain/family_pin.py", "dltrain/family_screen.py", "dltrain/holdout.py", "dltrain/s3io.py"])
        self.assertEqual(r["source_tree_sha256"], fx.sha("".join(f"{p}\0{d}\n" for p, d in r["source_tree"]).encode()))
        self.assertEqual(r["runtime_sha256"], fx.sha(common.canonical(r["runtime"])))
        self.assertEqual(set(r["runtime"]), {"python", "executable_sha256", "boto3", "botocore"})
        self.assertEqual(r["selection_code_sha256"], holdout.ALGORITHM["code_sha256"])


class AssignmentPoison(fx.P3Case):
    SEED = bytes(range(32))

    def setUp(self):
        super().setUp()
        self.fams = fx.standard_families()

    def test_honest_perturbations_leave_the_split_unchanged(self):
        res = holdout.run_assignment_poison(self.fams, self.SEED)
        self.assertTrue(res["pass"], json.dumps(res["variants"]))
        self.assertEqual(len(res["variants"]), len(holdout.POISON_VARIANTS))
        self.assertIn("held_cumulative", [v["variant"] for v in res["variants"]])

    def test_perturbations_touch_only_the_targeted_families(self):
        base = holdout.freeze(self.fams, holdout.pool_members(self.fams), self.SEED)
        before = common.canonical(self.fams)
        for name, target, n, fams in holdout.poison_variants(self.fams, base):
            changed = {f for f in self.fams if common.canonical(fams[f]) != common.canonical(self.fams[f])}
            self.assertEqual(changed, set(base["held"] if target == "heldout" else base["dev"]), name)
            self.assertEqual(holdout.pool_members(fams), holdout.pool_members(self.fams), name)
        self.assertEqual(common.canonical(self.fams), before)

    def test_member_and_pool_field_perturbations_reach_every_leaf(self):
        """LP-3 and SC-02: the member variant rewrites every member leaf and reorders, drops and adds members; the
        pool-field variant changes rights_status and independence_qualified without leaving the pool."""
        base = holdout.freeze(self.fams, holdout.pool_members(self.fams), self.SEED)
        variants = {name: fams for name, _, _, fams in holdout.poison_variants(self.fams, base)}
        for f in base["held"]:
            old, new = self.fams[f], variants["held_members_rewritten"][f]
            old_leaves = {(i, k): json.dumps(v, sort_keys=True) for i, m in enumerate(old["members"]) for k, v in m.items()}
            for m in new["members"][:-1]:
                for k, v in m.items():
                    self.assertNotIn((k, json.dumps(v, sort_keys=True)), {(k2, v2) for (_, k2), v2 in old_leaves.items()})
            self.assertEqual(len(new["members"]), max(1, len(old["members"]) - 1) + 1)
            p = variants["held_pool_fields_changed"][f]
            self.assertNotEqual((p["rights_status"], p["independence_qualified"]),
                                (old["rights_status"], old["independence_qualified"]))
            self.assertTrue(holdout.pool_predicate(p))
            g = variants["held_all_leaves_rewritten"][f]
            for k in set(old) - set(holdout.POOL_FIXED_FIELDS):
                if old[k] not in (None, [], {}):
                    self.assertNotEqual(g[k], old[k], k)

    def assertDetected(self, freeze_fn, key):
        res = holdout.run_assignment_poison(self.fams, self.SEED, freeze_fn)
        self.assertFalse(res["pass"])
        hit = [v["variant"] for v in res["variants"] if key in v["changed"]]
        self.assertTrue(hit, json.dumps(res["variants"]))
        self.assertTrue(all(key in v["spec_keys_changed"] for v in res["variants"] if key in v["changed"]))
        return hit

    def test_detects_a_planted_diag_dependency(self):
        hit = self.assertDetected(leaky_diag_freeze, "diag")
        self.assertIn("held_contexts_added", hit)
        self.assertIn("held_cumulative", hit)

    def test_detects_a_planted_tau_dependency(self):
        hit = self.assertDetected(leaky_tau_freeze, "tau_hex")
        self.assertIn("held_contexts_removed", hit)

    def test_detects_a_planted_dev_dependency_on_held_members(self):
        hit = self.assertDetected(leaky_dev_freeze, "dev")
        self.assertIn("held_members_changed", hit)
        self.assertIn("held_cumulative", hit)

    def test_detects_a_planted_dev_dependency_on_held_pool_field_values(self):
        hit = self.assertDetected(pool_field_dev_freeze, "dev")
        self.assertIn("held_pool_fields_changed", hit)
        self.assertIn("held_cumulative", hit)

    def test_detects_planted_diag_dependencies_on_held_member_fields(self):
        """LP-3: identity, item_id and union_keys salts on the diag draw are seen by the poison test and by
        freeze_checks.diag_is_prio_prefix (an independent recomputation of the draw)."""
        pool = holdout.pool_members(self.fams)
        for field in ("identity", "item_id", "union_keys", "sample_identity_sha256", "registry_state"):
            fz = member_salted_diag_freeze(field)
            hit = self.assertDetected(fz, "diag")
            self.assertIn("held_members_rewritten", hit, field)
            self.assertIn("held_cumulative", hit, field)
            if field != "item_id":                         # frozen-cohort members have no item_id leaf to rewrite
                self.assertIn("held_all_leaves_rewritten", hit, field)
            checks = holdout.freeze_checks(self.fams, pool, fz(self.fams, pool, self.SEED), self.SEED)
            self.assertFalse(checks["diag_is_prio_prefix"], field)

    def test_freeze_checks_recompute_train_fit(self):
        pool = holdout.pool_members(self.fams)
        res = holdout.freeze(self.fams, pool, self.SEED)
        dropped = [f for f in res["train_fit"] if res["assignment"][f] == "train_nonpool"][0]
        bad = dict(res, train_fit=[f for f in res["train_fit"] if f != dropped])
        self.assertFalse(holdout.freeze_checks(self.fams, pool, bad, self.SEED)["train_fit_exact"])

    def test_verify_reports_stop_for_a_planted_dependency(self):
        inputs = fx.write_inputs(self.tmp / "in")
        out, _ = fx.pipeline(self, inputs, verify_too=False)
        res = holdout.verify(out / "holdout.json", inputs.registry, out / "seed.hex", assignment_poison=True,
                             freeze_fn=leaky_diag_freeze)
        self.assertEqual(res["status"], "STOP")
        self.assertFalse(res["assignment_poison"]["pass"])
        res = holdout.verify(out / "holdout.json", inputs.registry, out / "seed.hex", assignment_poison=True,
                             freeze_fn=member_salted_diag_freeze("identity"))
        self.assertEqual(res["status"], "STOP")
        self.assertIn("diag_is_prio_prefix", res["failed_checks"])
        self.assertFalse(res["assignment_poison"]["pass"])
        honest = holdout.verify(out / "holdout.json", inputs.registry, out / "seed.hex", assignment_poison=True)
        self.assertEqual(honest["status"], "PASS")

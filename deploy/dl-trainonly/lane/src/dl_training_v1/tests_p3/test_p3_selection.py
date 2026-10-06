"""PLAN §6.2 selection code: prio domain separation, derive_seed, freeze (exact k, determinism, one assignment per family,
train_fit, split diagnostic) and assign_appended (every branch)."""
import copy
import hashlib
import math
import random
from collections import Counter

from dltrain import common, holdout
from tests_p3 import fixtures_p3 as fx

SEED = hashlib.sha256(b"p3 selection seed").digest()
DOMAINS = (holdout.DOMAIN, holdout.DEV_DOMAIN, holdout.DIAG_DOMAIN)


def frozen(families, seed=SEED):
    return holdout.freeze(families, holdout.pool_members(families), seed)


class PrioAndSeed(fx.P3Case):
    def test_prio_is_the_plan_formula(self):
        for unit in ("fam-x", "e" * 64, "naïve unit", ""):
            for dom in DOMAINS:
                want = int.from_bytes(hashlib.sha256(dom + b"\0" + SEED + b"\0" + unit.encode()).digest(), "big")
                self.assertEqual(holdout.prio(SEED, unit, dom), want)
        self.assertEqual(holdout.prio(SEED, "u"), holdout.prio(SEED, "u", b"jope.dl-family-holdout/v1"))
        self.assertEqual(DOMAINS, (b"jope.dl-family-holdout/v1", b"jope.dl-family-dev/v1", b"jope.dl-split-diagnostic/v1"))

    def test_domain_separation(self):
        units = [fx.fid(f"d{i}") for i in range(400)]
        for u in units:
            self.assertEqual(len({holdout.prio(SEED, u, d) for d in DOMAINS}), 3)
        orders = [sorted(units, key=lambda u, d=d: (holdout.prio(SEED, u, d), u)) for d in DOMAINS]
        self.assertNotEqual(orders[0][:20], orders[1][:20])
        self.assertNotEqual(orders[0][:20], orders[2][:20])
        self.assertNotEqual(orders[1][:20], orders[2][:20])
        # 5% prefixes of independent orders barely overlap (expected overlap 20 * 0.05 = 1)
        self.assertLess(len(set(orders[0][:20]) & set(orders[1][:20])), 6)
        # the seed separates draws as well
        other = sorted(units, key=lambda u: (holdout.prio(b"\x01" * 32, u), u))
        self.assertNotEqual(orders[0][:20], other[:20])

    def test_derive_seed(self):
        body, vid = b'{"a":1}', "3HL4kqtJlcpXroDTDmJ.rmSpXd3dIbrH"
        want = hashlib.sha256(hashlib.sha256(body).digest() + vid.encode()).digest()
        self.assertEqual(holdout.derive_seed(body, vid), want)
        self.assertEqual(len(want), 32)
        self.assertEqual(holdout.derive_seed(body, vid), holdout.derive_seed(bytes(body), str(vid)))
        self.assertNotEqual(holdout.derive_seed(body, vid + "x"), want)
        self.assertNotEqual(holdout.derive_seed(body + b" ", vid), want)


class Freeze(fx.P3Case):
    def test_exact_k_and_dev_size_for_many_pool_sizes(self):
        for n in (2, 3, 19, 20, 21, 39, 40, 41, 60, 100, 199, 200, 201, 999):
            fams = fx.pool_only_families(n)
            res = frozen(fams)
            k = -(-n // 20)                                   # exact ceil(0.05·N) in integers
            self.assertEqual((res["N"], res["k"], len(res["held"])), (n, k, k), n)
            self.assertEqual(len(res["dev"]), -(-(n - k) // 20), n)
            checks = holdout.freeze_checks(fams, holdout.pool_members(fams), res, SEED)
            self.assertEqual([c for c, ok in checks.items() if not ok], [], n)

    def test_tiny_or_inconsistent_pools_stop(self):
        for n in (0, 1):
            fams = fx.pool_only_families(n)
            with self.assertRaises(holdout.HoldoutStop):
                holdout.freeze(fams, holdout.pool_members(fams), SEED)
        fams = fx.pool_only_families(5)
        with self.assertRaises(holdout.HoldoutStop):
            holdout.freeze(fams, holdout.pool_members(fams) + [fx.fid("not-in-registry")], SEED)

    def test_deterministic_and_order_independent(self):
        fams = fx.standard_families()
        a = common.canonical(frozen(fams))
        rng = random.Random(7)
        keys = list(fams)
        rng.shuffle(keys)
        shuffled = {k: copy.deepcopy(fams[k]) for k in keys}
        pool = holdout.pool_members(shuffled)
        rng.shuffle(pool)
        self.assertEqual(common.canonical(holdout.freeze(shuffled, pool + pool[:3], SEED)), a)
        self.assertEqual(common.canonical(frozen(fams)), a)
        self.assertNotEqual(frozen(fams, b"\x02" * 32)["held"], frozen(fams)["held"])

    def test_freeze_does_not_mutate_the_registry(self):
        fams = fx.standard_families()
        before = common.canonical(fams)
        frozen(fams)
        self.assertEqual(common.canonical(fams), before)

    def test_every_registry_family_has_exactly_one_assignment_including_non_pool(self):
        fams = fx.standard_families()
        res = frozen(fams)
        a = res["assignment"]
        self.assertEqual(set(a), set(fams))
        self.assertEqual(len(a), len(fams))
        self.assertEqual(Counter(a.values()), {"heldout": 3, "dev": 3, "train_fit": 54, "train_nonpool": 7,
                                               "sealed_excluded": 3, "prohibited_excluded": 2})
        for tag, want in fx.EXPECTED_NONPOOL.items():
            self.assertEqual(a[fx.fid(tag)], want, tag)
        self.assertEqual(sorted(f for f, v in a.items() if v == "heldout"), sorted(res["held"]))
        self.assertEqual(sorted(f for f, v in a.items() if v == "dev"), sorted(res["dev"]))
        checks = holdout.freeze_checks(fams, holdout.pool_members(fams), res, SEED)
        self.assertEqual([c for c, ok in checks.items() if not ok], [])

    def test_train_fit_excludes_dev_and_heldout_and_includes_train_nonpool(self):
        fams = fx.standard_families()
        res = frozen(fams)
        tf = set(res["train_fit"])
        self.assertFalse(tf & set(res["dev"]))
        self.assertFalse(tf & set(res["held"]))
        nonpool = {fx.fid(t) for t, v in fx.EXPECTED_NONPOOL.items() if v == "train_nonpool"}
        self.assertTrue(nonpool <= tf)
        pool = set(holdout.pool_members(fams))
        self.assertEqual(tf, (pool - set(res["held"]) - set(res["dev"])) | nonpool)
        self.assertFalse({fx.fid(t) for t in ("x0", "x1", "x2", "q0", "q1")} & tf)

    def test_split_diagnostic_excludes_frozen_fold0_and_is_drawn_once(self):
        fams = fx.standard_families()
        res = frozen(fams)
        owner = {c: f for f, rec in fams.items() for c in rec["contexts"]}
        fold0 = {c for rec in fams.values() for c, v in rec["frozen_fold0"].items() if v}
        train_fold0 = {c for c in fold0 if owner[c] in set(res["train_fit"])}
        self.assertTrue(train_fold0, "the fixture must offer fold-0 contexts to exclude")
        self.assertFalse(set(res["diag"]) & fold0)
        self.assertTrue(all(owner[c] in set(res["train_fit"]) for c in res["diag"]))
        ctx_rest = [c for f in res["train_fit"] for c in fams[f]["contexts"] if not fams[f]["frozen_fold0"][c]]
        self.assertEqual(len(res["diag"]), math.ceil(0.05 * len(ctx_rest)))
        by_prio = sorted(ctx_rest, key=lambda c: (holdout.prio(SEED, c, holdout.DIAG_DOMAIN), c))
        self.assertEqual(res["diag"], by_prio[:len(res["diag"])])         # §6.2: kept in diagnostic-priority order
        # appends never redraw it: new families are assigned, held/dev/diag stay exactly as frozen
        new = {fx.fid(f"append{i}"): {"member_families": [], "pool_eligible": True, "sealed": False, "prohibited": False}
               for i in range(40)}
        out = holdout.append_assignments(res, SEED, new)
        self.assertEqual((out["held"], out["dev"], out["diag"]), (res["held"], res["dev"], res["diag"]))
        self.assertEqual(set(out["assignment"]), set(new))
        appended_contexts = {fx.h(f"context:append{i}:0:{j}") for i in range(40) for j in range(5)}
        self.assertFalse(appended_contexts & set(out["diag"]))

    def test_freeze_checks_catch_wrong_results(self):
        fams = fx.standard_families()
        pool = holdout.pool_members(fams)
        res = frozen(fams)
        bad_k = dict(res, k=res["k"] + 1)
        self.assertFalse(holdout.freeze_checks(fams, pool, bad_k, SEED)["k_exact"])
        fold0 = next(c for f in res["train_fit"] for c, v in fams[f]["frozen_fold0"].items() if v)
        bad_diag = dict(res, diag=sorted(res["diag"][1:] + [fold0]))
        self.assertFalse(holdout.freeze_checks(fams, pool, bad_diag, SEED)["diag_excludes_frozen_fold0"])
        moved = res["dev"][0]
        bad_tf = dict(res, train_fit=sorted(res["train_fit"] + [moved]))
        self.assertFalse(holdout.freeze_checks(fams, pool, bad_tf, SEED)["train_fit_excludes_heldout_and_dev"])
        missing = dict(res, assignment={f: v for f, v in res["assignment"].items() if f != fx.fid("x0")})
        self.assertFalse(holdout.freeze_checks(fams, pool, missing, SEED)["every_family_exactly_one_assignment"])


class AssignAppended(fx.P3Case):
    def setUp(self):
        super().setUp()
        self.fams = fx.standard_families()
        self.res = frozen(self.fams)
        self.a = self.res["assignment"]
        self.tau, self.tau_dev = int(self.res["tau_hex"], 16), int(self.res["tau_dev_hex"], 16)

    def app(self, members, new="fam-new", pool_eligible=True, sealed=False, prohibited=False):
        return holdout.assign_appended(set(members), new, pool_eligible, sealed, prohibited, SEED, self.tau, self.tau_dev, self.a)

    def by(self, label):
        return sorted(f for f, v in self.a.items() if v == label)

    def test_joining_existing_families_inherits(self):
        for label in ("heldout", "dev", "train_fit", "train_nonpool", "sealed_excluded", "prohibited_excluded"):
            fams = self.by(label)
            self.assertEqual(self.app(fams[:2]), label)
        self.assertEqual(self.app(self.by("sealed_excluded")[:1], sealed=True), "sealed_excluded")
        self.assertEqual(self.app(self.by("prohibited_excluded")[:1], prohibited=True), "prohibited_excluded")
        self.assertEqual(self.app(self.by("heldout")[:1], pool_eligible=False), "heldout")

    def test_contaminated_cases(self):
        cases = ((self.by("heldout")[:1] + self.by("train_fit")[:1], {}),        # cross-partition merge
                 (self.by("dev")[:1] + self.by("heldout")[:1], {}),
                 (self.by("train_fit")[:1] + self.by("train_nonpool")[:1], {}),
                 (self.by("train_fit")[:1], {"sealed": True}),                    # sealed member joining a train family
                 (self.by("heldout")[:1], {"sealed": True}),
                 (self.by("dev")[:1], {"prohibited": True}),                     # prohibited member joining dev
                 (self.by("sealed_excluded")[:1], {"prohibited": True}))
        for members, kw in cases:
            with self.assertRaisesRegex(holdout.Contaminated, "^CONTAMINATED$"):
                self.app(members, **kw)
        self.assertTrue(issubclass(holdout.Contaminated, RuntimeError))

    def test_new_families_sealed_prohibited_nonpool(self):
        self.assertEqual(self.app([], sealed=True), "sealed_excluded")
        self.assertEqual(self.app([], sealed=True, prohibited=True), "sealed_excluded")
        self.assertEqual(self.app([], prohibited=True), "prohibited_excluded")
        self.assertEqual(self.app([], pool_eligible=False), "train_nonpool")

    def test_new_pool_families_follow_tau_and_tau_dev(self):
        seen = Counter()
        for i in range(3000):
            f = fx.fid(f"new-pool-{i}")
            p, pd = holdout.prio(SEED, f), holdout.prio(SEED, f, holdout.DEV_DOMAIN)
            want = "heldout" if p <= self.tau else "dev" if pd <= self.tau_dev else "train_fit"
            self.assertEqual(self.app([], new=f), want)
            seen[want] += 1
        self.assertGreater(seen["heldout"], 0)
        self.assertGreater(seen["dev"], 0)
        self.assertGreater(seen["train_fit"], seen["heldout"] + seen["dev"])

    def test_tau_thresholds_are_consistent_with_the_frozen_split(self):
        pool = set(holdout.pool_members(self.fams))
        held, dev = set(self.res["held"]), set(self.res["dev"])
        self.assertTrue(all(holdout.prio(SEED, f) <= self.tau for f in held))
        self.assertTrue(all(holdout.prio(SEED, f) > self.tau for f in pool - held))
        self.assertTrue(all(holdout.prio(SEED, f, holdout.DEV_DOMAIN) <= self.tau_dev for f in dev))
        self.assertTrue(all(holdout.prio(SEED, f, holdout.DEV_DOMAIN) > self.tau_dev for f in pool - held - dev))
        self.assertEqual(self.res["p_tau"], (self.tau + 1) / 2**256)

    def test_append_helper_refuses_recomputed_ids(self):
        existing = self.res["train_fit"][0]
        with self.assertRaises(holdout.HoldoutStop):
            holdout.append_assignments(self.res, SEED, {existing: {"member_families": [], "pool_eligible": True,
                                                                   "sealed": False, "prohibited": False}})
        out = holdout.append_assignments(self.res, SEED, {existing: {"member_families": [existing], "pool_eligible": True,
                                                                     "sealed": False, "prohibited": False}})
        self.assertEqual(out["assignment"][existing], "train_fit")

    @staticmethod
    def append(members=(), pool_eligible=True, sealed=False, prohibited=False):
        return {"member_families": list(members), "pool_eligible": pool_eligible, "sealed": sealed,
                "prohibited": prohibited}

    def test_chained_appends_resolve_in_dependency_order(self):
        """SC-09: an appended family that joins another family appended in the same call inherits its result."""
        a, b, c = fx.fid("chain-a"), fx.fid("chain-b"), fx.fid("chain-c")
        appends = {c: self.append([b]), b: self.append([a]), a: self.append(pool_eligible=False)}
        out = holdout.append_assignments(self.res, SEED, appends)
        self.assertEqual(out["assignment"], {a: "train_nonpool", b: "train_nonpool", c: "train_nonpool"})
        self.assertEqual(out["order"].index(a) < out["order"].index(b) < out["order"].index(c), True)
        pool_new = fx.fid("chain-pool")
        out = holdout.append_assignments(self.res, SEED, {pool_new: self.append(), fx.fid("chain-join"): self.append([pool_new])})
        self.assertEqual(out["assignment"][fx.fid("chain-join")], out["assignment"][pool_new])
        mixed = {a: self.append(pool_eligible=False), b: self.append([a, self.by("heldout")[0]])}
        with self.assertRaisesRegex(holdout.Contaminated, "^CONTAMINATED$"):    # never a bare KeyError
            holdout.append_assignments(self.res, SEED, mixed)

    def test_append_helper_stops_on_unknown_members_cycles_and_self_reference(self):
        x, y, z = fx.fid("cyc-x"), fx.fid("cyc-y"), fx.fid("lonely")
        cases = (({x: self.append([y]), y: self.append([x])}, "cycle"),
                 ({z: self.append([fx.fid("nowhere")])}, "neither frozen nor appended"),
                 ({z: self.append([z])}, "names itself"))
        for appends, regex in cases:
            with self.assertRaisesRegex(holdout.HoldoutStop, regex):
                holdout.append_assignments(self.res, SEED, appends)

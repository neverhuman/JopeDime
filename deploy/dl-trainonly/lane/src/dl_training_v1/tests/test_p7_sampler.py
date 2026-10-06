"""PLAN §5 P7 tests 3 and 7 for the sampler (§6.1 macro draw, groups and streams; §6.2 train-only fits; §6.6 the
counter-based sampler state), plus the shared fixture's own invariants (tests/fixtures_p7.py).

- 3a: family-uniform macro draw, and dataset-within-family and context-within-dataset uniformity (chi-square
  against hard-coded 0.999 critical values, 1e5 slot draws);
- 3b: held, dev, diag and sealed units are never in the fit universe and never drawn; pi sums to 1 and every
  journal slot records it;
- 3c: successor group composition (comparators strictly below the teacher band), the feasibility stream, the
  journal's L_feas class counts (pi_group numerators, local backend only) and the baseline-arm group;
- 3d: determinism and the sampler-state round trip;
- TrainData.content_sha256 (§6.6 identities) tracks universe encodings, labels and grammar masks;
- 7: closure poison leaves every fitted statistic, train_fit label, the training-data identity and every collated
  batch byte-identical.

CPU only; nothing is written to disk.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from dltrain import common, labels, model, sampler
from tests import fixtures_p7 as fx

SRC = Path(__file__).resolve().parents[1]
# Chi-square 0.999 quantiles by degrees of freedom (hard-coded, no scipy).
CHI2_999 = {1: 10.828, 2: 13.816, 3: 16.266, 4: 18.467, 5: 20.515, 6: 22.458, 7: 24.322, 8: 26.124, 9: 27.877,
            10: 29.588}
MACRO_STEPS = 25_000            # steps 1..25000 x 4 slots = 1e5 macro draws
FULL_STEPS = 2_000
POISON_STEPS = 200
EXCLUDED_ROLES = ("dev", "diag", "heldout", "sealed_excluded")
OTHER_SEED = hashlib.sha256(b"jope.dl-p7-fixture/v1 another run_seed").digest()
FEASIBILITY_SPECIALS = {"degenerate": "DEGENERATE_BASELINE", "few_skeletons": "FEW_SKELETONS",
                        "no_detail": "UNKNOWN_BASELINE"}
SLOT_KEYS = {"slot", "family_id", "dataset_hash", "context_sha256", "pi_c", "stream", "members", "roles",
             "class_counts", "feas_class_counts"}

_CACHE: dict = {}


def setUpModule() -> None:
    model.configure_determinism(4)


def _data() -> sampler.TrainData:
    if "data" not in _CACHE:
        _CACHE["data"] = fx.train_data()
    return _CACHE["data"]


def _macro_draws() -> list[tuple[str, str, str]]:
    if "macro" not in _CACHE:
        fit = _data().fit
        _CACHE["macro"] = [sampler.macro_draw(fit, fx.RUN_SEED, step, slot)
                           for step in range(1, MACRO_STEPS + 1) for slot in range(fit.config.b_ctx)]
    return _CACHE["macro"]


def _full_draws() -> list[sampler.StepDraw]:
    if "full" not in _CACHE:
        data = _data()
        s = sampler.MacroSampler(data.fit, fx.RUN_SEED, data.config)
        _CACHE["full"] = [s.draw_next() for _ in range(FULL_STEPS)]
    return _CACHE["full"]


def _rows_in_j_order(context_sha256: str) -> list[dict]:
    if "records" not in _CACHE:
        _CACHE["records"] = fx.records()
    return [r for r in _CACHE["records"] if r["context_sha256"] == context_sha256]


def _chi2(counts: list[int]) -> float:
    expected = sum(counts) / len(counts)
    return sum((c - expected) ** 2 for c in counts) / expected


def _journal_bytes(draws) -> bytes:
    return b"".join(common.canonical(d.journal_record()) + b"\n" for d in draws)


def _label_bytes(obj) -> bytes:
    return json.dumps(dataclasses.asdict(obj), sort_keys=True).encode()


def _batch_digest(b: sampler.Batch) -> dict:
    out = {"mode": b.mode, "step": b.step, "ctx_ids": list(b.ctx_ids), "member_ids": [list(m) for m in b.member_ids],
           "n_tokens": b.n_tokens}
    for name in sampler.BATCH_TENSOR_FIELDS:
        t = getattr(b, name)
        out[name] = [str(t.dtype), list(t.shape), hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()]
    return out


def _rng_states():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def _fixture_digest() -> dict:
    """Digest of every fixture output, the fit, 50 journal lines and one batch (compared across processes)."""
    h = hashlib.sha256()
    h.update(json.dumps([fx.contexts(), fx.records(), fx.special_contexts(), fx.assignment()], sort_keys=True).encode())
    for sha, z in sorted(fx.encodings().items()):
        h.update(sha.encode() + z.tobytes())
    data = fx.train_data()
    s = sampler.MacroSampler(data.fit, fx.RUN_SEED, data.config)
    return {"fixture": h.hexdigest(), "fit_sha256": data.fit.sha256(),
            "journal_sha256": hashlib.sha256(_journal_bytes([s.draw_next() for _ in range(50)])).hexdigest(),
            "batch_sha256": hashlib.sha256(json.dumps(_batch_digest(sampler.collate(s.draw(1), data))).encode()).hexdigest()}


class FixtureTests(unittest.TestCase):
    """The shared fixture is the specified one, is pure, and exercises the §6.3 edge cases it promises."""

    def test_vocabulary_programs_and_grammar_mask(self):
        vocab = fx.vocab()
        self.assertEqual(len(vocab), 32)
        self.assertEqual(vocab[:15], ("<PAD>", "<BOS>", "<EOS>", "(", ")", "pipe", "X", "std", "log1p", "winsor_clip",
                                      "pca", "ridge", "lasso", "alpha=", "."))
        self.assertEqual(vocab[15:25], tuple("0123456789"))
        self.assertEqual(vocab[25:], tuple(f"<UNUSED_{i}>" for i in range(7)))
        for spec, dsl in zip(fx.SEED_SPECS, fx.SEED_DSLS):
            self.assertEqual(fx.program(*spec)["dsl"], dsl)
        p = fx.program(("pca", "winsor_clip", "log1p"), "lasso", "007")
        self.assertEqual(p["dsl"], "(pipe X (pca) (winsor_clip) (log1p) (lasso alpha=0.07))")
        self.assertEqual(p["program_sha256"], hashlib.sha256(p["dsl"].encode()).hexdigest())
        self.assertEqual(p["skeleton_sha256"],
                         hashlib.sha256(b"(pipe X (pca) (winsor_clip) (log1p) (lasso alpha=#))").hexdigest())
        self.assertEqual(len(p["tokens"]), 23)
        for r in fx.records():
            toks = r["tokens"]
            self.assertIn(len(toks), (17, 20, 23))
            self.assertEqual(r["length"], len(toks))
            mask = fx.legal_mask(toks)
            self.assertEqual((mask.shape, mask.dtype), ((len(toks) - 1, 32), np.dtype(bool)))
            self.assertTrue(all(mask[i, toks[i + 1]] for i in range(len(toks) - 1)))
            self.assertFalse(mask[:, [0] + list(range(25, 32))].any())      # PAD and unused ids are never legal
        good = fx.program(("std",), "ridge", "100")["tokens"]
        four_stages = good[:4] + [3, 7, 4] * 4 + good[7:]
        for bad in (good[:-1], good + [2], [2] + good[1:], good[:5] + [11] + good[6:], [1], four_stages):
            with self.assertRaises(ValueError):
                fx.legal_mask(bad)

    def test_partitions_records_and_specials(self):
        ctxs = fx.contexts()
        shas = [c["context_sha256"] for c in ctxs]
        self.assertEqual(shas, sorted(shas))
        self.assertEqual(Counter(c["split_role"] for c in ctxs),
                         {"train_fit": 42, "diag": 8, "dev": 9, "heldout": 5, "sealed_excluded": 2})
        asg = fx.assignment()
        self.assertEqual(sorted(asg), list(fx.FAMILY_IDS))
        self.assertEqual(Counter(asg.values()), {"train_fit": 8, "train_nonpool": 1, "dev": 2, "heldout": 1,
                                                 "sealed_excluded": 1})
        for c in ctxs:
            a = asg[c["family_id"]]
            expected_role = ("diag" if c["fold"] == 3 else "train_fit") if a in ("train_fit", "train_nonpool") else a
            self.assertEqual(c["split_role"], expected_role)
            self.assertEqual(c["frozen_fold0"], c["fold"] == 0)
            self.assertEqual(len(c["y_train"]), 24)
        per_ctx = Counter(r["context_sha256"] for r in fx.records())
        for i, sha in enumerate(shas):
            self.assertEqual(per_ctx[sha], 12 + i % 9)
            rows = _rows_in_j_order(sha)
            self.assertEqual([r["shared_seed"] for r in rows[:4]], [True, True, True, False])
            self.assertEqual(len({r["program_sha256"] for r in rows}), len(rows))
        specials = fx.special_contexts()
        self.assertEqual(tuple(specials), fx.SPECIAL_NAMES)
        self.assertEqual(len(set(specials.values())), len(fx.SPECIAL_NAMES))
        for sha in specials.values():
            i = shas.index(sha)
            self.assertEqual(ctxs[i]["split_role"], "train_fit")
            self.assertGreaterEqual(12 + i % 9, 14)

    def test_special_contexts_exercise_the_label_edge_cases(self):
        corp = _data().corpus
        sp = fx.special_contexts()
        status = {name: corp.labels.contexts[sha].status for name, sha in sp.items()}
        self.assertEqual(status, {"degenerate": "DEGENERATE_BASELINE", "few_skeletons": "OK",
                                  "no_detail": "UNKNOWN_BASELINE", "exact_fit": "OK", "nonfinite": "OK", "ties": "OK",
                                  "detailless": "OK"})

        def lab(name: str, j: int) -> labels.RecordLabel:
            return corp.label(sp[name], _rows_in_j_order(sp[name])[j]["program_sha256"])

        few = corp.labels.records[sp["few_skeletons"]]
        self.assertEqual(len({r.skeleton_sha256 for r in few if r.outcome == labels.COMPLETED}), 3)
        cap = corp.labels.contexts[sp["exact_fit"]].q_cap
        self.assertTrue(math.isfinite(cap))
        self.assertEqual((lab("exact_fit", 3).q, lab("exact_fit", 3).q_status), (cap, "known"))
        for j in (3, 4):
            self.assertEqual((lab("nonfinite", j).outcome, lab("nonfinite", j).feas_class, lab("nonfinite", j).q),
                             (labels.NONFINITE, 1, None))
        t3, t4 = lab("ties", 3), lab("ties", 4)
        self.assertEqual(t3.q_grid, t4.q_grid)
        self.assertNotEqual(t3.q, t4.q)
        self.assertEqual(set(_data().fit.plans[sp["ties"]].teachers[:2]), {t3.program_sha256, t4.program_sha256})
        self.assertEqual(lab("detailless", 3).q, corp.labels.contexts[sp["detailless"]].q_cap)
        self.assertEqual(lab("detailless", 4).q_status, "unknown_detailless_nonfinite")
        self.assertTrue(all(r.q is None for r in corp.labels.records[sp["degenerate"]]))

    def test_fresh_objects_and_no_global_rng(self):
        before = _rng_states()
        a, b = fx.records(), fx.records()
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        a[0]["tokens"].append(99)
        self.assertNotEqual(fx.records()[0]["tokens"], a[0]["tokens"])
        enc = fx.encodings()
        first = next(iter(enc))
        enc[first][:] = 0.0
        self.assertTrue(np.any(fx.encodings()[first] != 0.0))
        data = fx.train_data()
        s = sampler.MacroSampler(data.fit, fx.RUN_SEED, data.config)
        for _ in range(20):
            sampler.collate(s.draw_next(), data)
        after = _rng_states()
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[1][0], after[1][0])
        self.assertTrue(np.array_equal(before[1][1], after[1][1]))
        self.assertEqual(before[1][2:], after[1][2:])
        self.assertTrue(torch.equal(before[2], after[2]))

    def test_identical_content_in_a_fresh_process(self):
        code = "import json\nfrom tests import test_p7_sampler as t\nprint(json.dumps(t._fixture_digest()))"
        env = dict(os.environ, PYTHONHASHSEED="20261005")
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(SRC), env=env, capture_output=True, text=True,
                              timeout=600)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(json.loads(proc.stdout.strip().splitlines()[-1]), _fixture_digest())

    def test_poison_touches_only_non_train_fit_units_and_never_its_inputs(self):
        ctxs, recs, enc = fx.contexts(), fx.records(), fx.encodings()
        snapshot = json.dumps([ctxs, recs], sort_keys=True)
        enc_copy = {k: v.copy() for k, v in enc.items()}
        p_ctx, p_rec, p_enc = fx.poison(ctxs, recs, enc)
        self.assertEqual(json.dumps([ctxs, recs], sort_keys=True), snapshot)
        self.assertTrue(all(np.array_equal(enc[k], enc_copy[k]) for k in enc))
        fit_set = {c["context_sha256"] for c in ctxs if c["split_role"] == "train_fit"}
        targets = [c["context_sha256"] for c in ctxs if c["split_role"] != "train_fit"]
        self.assertEqual(len(targets), 24)

        def part(rows, inside):
            return json.dumps([r for r in rows if (r["context_sha256"] in fit_set) == inside], sort_keys=True)

        self.assertEqual(part(ctxs, True), part(p_ctx, True))
        self.assertEqual(part(recs, True), part(p_rec, True))
        self.assertNotEqual(part(recs, False), part(p_rec, False))
        self.assertEqual(len(p_rec), len(recs) + 2 * len(targets))
        by_sha = {c["context_sha256"]: c for c in ctxs}
        for c in p_ctx:
            if c["context_sha256"] in targets:
                self.assertEqual(c["y_train"], [3.0 * y + 7.0 for y in by_sha[c["context_sha256"]]["y_train"]])
        for k, z in p_enc.items():
            self.assertTrue(np.array_equal(z, -enc[k] if k in targets else enc[k]))
        original = {(r["context_sha256"], r["program_sha256"]): r["outcome"] for r in recs}
        for r in p_rec:
            key = (r["context_sha256"], r["program_sha256"])
            if key[0] in fit_set or key not in original:
                continue
            was = original[key]
            expected = (fx.POISON_FAILED if was == fx.COMPLETED else fx.COMPLETED if was.startswith("failed:") else was)
            self.assertEqual(r["outcome"], expected)
        sampler.Corpus.build(p_ctx, p_rec, p_enc, fx.assignment())          # the poisoned corpus is well formed


class MacroDrawTests(unittest.TestCase):
    """P7 tests 3a and 3b (PLAN §6.1 draw steps 1-3 and pi_c; §6.2 train_fit only)."""

    def test_3a_family_uniform_chi_square(self):
        fit = _data().fit
        self.assertEqual(fit.families, tuple(f"fam-{f:02d}" for f in range(9)))
        counts = Counter(f for f, _, _ in _macro_draws())
        self.assertEqual(sum(counts.values()), 100_000)
        self.assertEqual(sorted(counts), list(fit.families))
        stat = _chi2([counts[f] for f in fit.families])
        self.assertLess(stat, CHI2_999[8], f"family chi-square {stat:.3f} (df 8)")

    def test_3a_dataset_and_context_uniform_within_their_parent(self):
        fit = _data().fit
        by_family: dict[str, Counter] = {}
        by_dataset: dict[str, Counter] = {}
        for f, d, c in _macro_draws():
            by_family.setdefault(f, Counter())[d] += 1
            by_dataset.setdefault(d, Counter())[c] += 1
        tested = 0
        for f in fit.families:
            datasets = fit.datasets[f]
            self.assertEqual(sorted(by_family[f]), list(datasets))
            if len(datasets) > 1:
                stat = _chi2([by_family[f][d] for d in datasets])
                self.assertLess(stat, CHI2_999[len(datasets) - 1], f"datasets of {f}: chi-square {stat:.3f}")
                tested += 1
        for d, ctxs in fit.contexts.items():
            self.assertEqual(sorted(by_dataset[d]), list(ctxs))
            if len(ctxs) > 1:
                stat = _chi2([by_dataset[d][c] for c in ctxs])
                self.assertLess(stat, CHI2_999[len(ctxs) - 1], f"contexts of {d}: chi-square {stat:.3f}")
                tested += 1
        self.assertEqual(tested, 5 + 12)        # 5 families with > 1 dataset, 12 datasets with > 1 train_fit context

    def test_3b_universe_excludes_held_dev_diag_and_sealed(self):
        data = _data()
        corp, fit = data.corpus, data.fit
        roles = {sha: row["split_role"] for sha, row in corp.contexts.items()}
        for role in EXCLUDED_ROLES:
            self.assertGreater(sum(r == role for r in roles.values()), 0, f"fixture has no {role} context")
        universe = set(fit.pi)
        expected = {sha for sha, row in corp.contexts.items()
                    if row["split_role"] == "train_fit" and corp.assignment[row["family_id"]] in sampler.TRAIN_ASSIGNMENTS}
        self.assertEqual(universe, expected)
        self.assertEqual(len(universe), 42)
        self.assertEqual(set(fit.plans), universe)
        self.assertEqual({c for ctxs in fit.contexts.values() for c in ctxs}, universe)
        self.assertEqual(set(fit.families), {f for f, a in corp.assignment.items() if a in sampler.TRAIN_ASSIGNMENTS})
        excluded = {sha for sha, role in roles.items() if role in EXCLUDED_ROLES}
        self.assertFalse(universe & excluded)
        self.assertEqual(fit.excluded, {"diag_contexts": 8, "unlabelled_contexts": 0, "empty_datasets": 0,
                                        "empty_families": 0})
        blob = common.canonical(fit.statistics()).decode()
        for sha in excluded:
            self.assertNotIn(sha, blob)
        for fam, assigned in corp.assignment.items():
            if assigned not in sampler.TRAIN_ASSIGNMENTS:
                self.assertNotIn(fam, blob)
                for row in corp.contexts.values():
                    if row["family_id"] == fam:
                        self.assertNotIn(row["dataset_hash"], blob)

    def test_3b_pi_sums_to_one_and_is_the_tree_formula(self):
        fit = _data().fit
        self.assertAlmostEqual(math.fsum(fit.pi.values()), 1.0, delta=1e-12)
        n_fam = len(fit.families)
        for f in fit.families:
            for d in fit.datasets[f]:
                for c in fit.contexts[d]:
                    self.assertEqual(fit.pi[c], 1.0 / (n_fam * len(fit.datasets[f]) * len(fit.contexts[d])))

    def test_3b_held_dev_diag_and_sealed_never_drawn_and_pi_journalled(self):
        data = _data()
        corp, fit = data.corpus, data.fit
        drawn = set(_macro_draws())
        for f, d, c in drawn:
            row = corp.contexts[c]
            self.assertEqual((row["family_id"], row["dataset_hash"], row["split_role"]), (f, d, "train_fit"))
            self.assertIn(c, fit.pi)
        self.assertEqual({c for _, _, c in drawn}, set(fit.pi))          # every universe context is reached
        draws = _full_draws()
        self.assertEqual([d.step for d in draws], list(range(1, FULL_STEPS + 1)))
        for step_draw in draws:
            rec = step_draw.journal_record()
            self.assertEqual(set(rec), {"schema", "step", "slots"})
            self.assertEqual(rec["schema"], "jope.dl-consumption/v1")
            common.canonical(rec)                                         # JSON-able, no NaN
            self.assertEqual(len(rec["slots"]), 4)
            for s, js in zip(step_draw.slots, rec["slots"]):
                self.assertEqual(set(js), SLOT_KEYS)
                self.assertIn(s.context_sha256, fit.pi)
                self.assertEqual(js["pi_c"], fit.pi[s.context_sha256])
                row = corp.contexts[s.context_sha256]
                self.assertEqual((row["family_id"], row["dataset_hash"], row["split_role"]),
                                 (s.family_id, s.dataset_hash, "train_fit"))
                for m in s.members:
                    self.assertTrue(corp.label(s.context_sha256, m).labelled)


class GroupCompositionTests(unittest.TestCase):
    """P7 test 3c (PLAN §6.1 groups and feasibility stream; §6.3.2 lineage-B group of the baseline arm)."""

    def _check_group(self, corp: sampler.Corpus, plan: sampler.ContextPlan, s: sampler.SlotDraw) -> None:
        ctx = s.context_sha256
        labs = [corp.label(ctx, m) for m in s.members]
        self.assertEqual(len(s.members), 8)
        self.assertEqual(len(set(s.members)), 8)
        n_fail, n_comp = s.roles.count("failure"), s.roles.count("comparator")
        self.assertEqual(s.roles, ("teacher",) * 4 + ("comparator",) * n_comp + ("failure",) * n_fail)
        self.assertEqual(n_fail, min(2, len(plan.failures)))
        self.assertEqual(n_comp, 2 + (2 - n_fail))
        teachers = labs[:4]
        self.assertEqual(tuple(t.program_sha256 for t in teachers), plan.teachers)
        self.assertEqual(len({t.skeleton_sha256 for t in teachers}), 4)
        for t in teachers:
            self.assertEqual((t.outcome, t.q_status, t.gate_pass), (labels.COMPLETED, "known", True))
        keys = [(-t.q_grid, t.program_sha256) for t in teachers]
        self.assertEqual(keys, sorted(keys))
        best: dict[str, tuple[int, str]] = {}
        for r in corp.labels.records[ctx]:
            if r.outcome == labels.COMPLETED and r.q_status == "known" and r.gate_pass is True:
                best[r.skeleton_sha256] = min(best.get(r.skeleton_sha256, (math.inf, "")), (-r.q_grid, r.program_sha256))
        for t, key in zip(teachers, keys):
            self.assertEqual(best[t.skeleton_sha256], key)              # each teacher is its skeleton's best
        teacher_skeletons = {t.skeleton_sha256 for t in teachers}
        for skeleton, key in best.items():
            if skeleton not in teacher_skeletons:
                self.assertGreater(key, keys[-1])                       # no other skeleton outranks the 4th teacher
        available = [len(plan.bands[0]), len(plan.bands[1])]
        teacher_floor = min(t.q_grid for t in teachers)
        for i, lab in enumerate(labs[4:4 + n_comp]):
            self.assertEqual((lab.outcome, lab.q_status), (labels.COMPLETED, "known"))
            self.assertNotIn(lab.program_sha256, plan.teachers)
            self.assertLess(lab.q_grid, teacher_floor, "a comparator must come from below the teacher band")
            band = i % 2 if available[i % 2] else 1 - i % 2
            available[band] -= 1
            self.assertIn(lab.program_sha256, plan.bands[band])
        for lab in labs[4 + n_comp:]:
            self.assertIn(lab.feas_class, (1, 2))
            self.assertTrue(lab.outcome.startswith("failed:") or lab.outcome == labels.TIMEOUT_OUTCOME)
        self.assertEqual(s.class_counts, tuple(sum(lab.feas_class == k for lab in labs) for k in range(3)))
        self.assertEqual(s.feas_class_counts,
                         tuple(sum(lab.feas_class == k and lab.feas_valid for lab in labs) for k in range(3)))

    def test_3c_comparator_bands_lie_strictly_below_the_teacher_band(self):
        """PLAN §6.1: comparators come from quantile bands below the teacher band. Every non-teacher completed q-known
        program is in a band iff its q_grid is below the lowest teacher's; the fixture holds contexts where a
        teacher skeleton's other program or a gate-failing program ranks inside or above the band."""
        corp, fit = _data().corpus, _data().fit
        excluded = 0
        for ctx, plan in fit.plans.items():
            if not plan.teachers:
                continue
            floor = min(corp.label(ctx, t).q_grid for t in plan.teachers)
            known = sorted((r for r in corp.labels.records[ctx]
                            if r.outcome == labels.COMPLETED and r.q_status == "known"),
                           key=lambda r: (-r.q_grid, r.program_sha256))
            below = [r.program_sha256 for r in known if r.program_sha256 not in plan.teachers and r.q_grid < floor]
            self.assertEqual(list(plan.bands[0]) + list(plan.bands[1]), below, ctx)
            self.assertEqual(len(plan.bands[0]), (len(below) + 1) // 2)
            inside = [r for r in known if r.program_sha256 not in plan.teachers and r.q_grid >= floor]
            excluded += bool(inside)
            for r in inside:
                self.assertTrue(r.gate_pass is not True or r.skeleton_sha256 in
                                {corp.label(ctx, t).skeleton_sha256 for t in plan.teachers} or r.q_grid == floor)
        self.assertGreater(excluded, 0, "the fixture never ranks a non-teacher inside or above the teacher band")

    def test_3c_too_few_comparators_below_the_band_is_a_feasibility_context(self):
        """A hand-built context whose only non-teacher programs at or above the teacher band are a teacher skeleton's
        second program and a gate-failing program with high q: neither is a comparator, so with one program below
        the band the context is TOO_FEW_LABELLED, and a second one below the band makes it a group."""
        ctx = {"context_sha256": "c" * 64, "dataset_hash": "d" * 64, "family_id": "fam-00", "split_role": "train_fit",
               "fold": 0, "frozen_fold0": True, "y_train": fx.contexts()[0]["y_train"]}
        s_c = labels.robust_scale(ctx["y_train"])
        base = [s_c, 1.2 * s_c]
        specs = [("T1", ("std",), "ridge", 0.9), ("T2", ("log1p",), "lasso", 0.8), ("T3", ("pca", "std"), "ridge", 0.7),
                 ("T4", ("winsor_clip",), "ridge", 0.6), ("S1", ("std",), "ridge", 0.85),
                 ("G1", ("pca",), "lasso", None), ("L1", ("log1p", "pca"), "lasso", 0.1)]
        rows, names = [], {}
        for j, (name, stages, head, a) in enumerate(specs):
            prog = fx.program(stages, head, 100 + j)
            names[prog["program_sha256"]] = name
            row = fx._record(ctx, prog, fx.COMPLETED, j, base, False)
            if a is None:                           # q about 1.75, arithmetic-mean ratio 1.505: gate fails
                row.update(rmse_j=[base[0] * 0.01, base[1] * 3.0], cv_mean_ratio=1.505)
            else:
                fx._set_completed_metrics(row, a, base)
            row["wall_us"] = 1_000_000
            rows.append(row)
        for j, (stages, outcome) in enumerate(((("std", "pca"), fx.FAILED), (("pca", "log1p"), fx.TIMEOUT)), start=20):
            rows.append(dict(fx._record(ctx, fx.program(stages, "ridge", j), outcome, j, base, False), wall_us=50_000))
        cfg = fx.sampler_config()

        def plan_of(records) -> sampler.ContextPlan:
            corpus = sampler.Corpus.build([ctx], records, {ctx["context_sha256"]: np.zeros(fx.ENC_DIM)}, fx.assignment())
            return sampler.context_plan(corpus, ctx["context_sha256"], cfg)

        plan = plan_of(rows)
        self.assertEqual([names[p] for p in plan.teachers], ["T1", "T2", "T3", "T4"])
        self.assertEqual(([names[p] for p in plan.bands[0]], list(plan.bands[1])), (["L1"], []))
        self.assertEqual((plan.stream, plan.reason), ("feasibility", "TOO_FEW_LABELLED"))
        prog = fx.program(("winsor_clip", "std"), "lasso", 999)
        names[prog["program_sha256"]] = "L2"
        extra = fx._record(ctx, prog, fx.COMPLETED, 30, base, False)
        fx._set_completed_metrics(extra, 0.2, base)
        plan = plan_of(rows + [dict(extra, wall_us=1_000_000)])
        self.assertEqual((plan.stream, plan.reason), ("group", None))
        self.assertEqual([[names[p] for p in band] for band in plan.bands], [["L2"], ["L1"]])

    def test_3c_feasibility_class_counts_exclude_non_local_members(self):
        """PLAN §6.3.2 pi_group: the journal's feas_class_counts count only the members that enter L_feas (the local
        backend, §6.3 backend rule); class_counts count every sampled member."""
        corp = _data().corpus
        differing = 0
        for step_draw in _full_draws():
            for s in step_draw.slots:
                labs = [corp.label(s.context_sha256, m) for m in s.members]
                self.assertEqual(s.feas_class_counts, tuple(sum(lab.feas_class == k and lab.feas_valid for lab in labs)
                                                            for k in range(3)))
                self.assertTrue(all(f <= c for f, c in zip(s.feas_class_counts, s.class_counts)))
                remote = sum(lab.backend != "local" for lab in labs)
                self.assertEqual(sum(s.class_counts) - sum(s.feas_class_counts), remote)
                differing += s.class_counts != s.feas_class_counts
        self.assertGreater(differing, 0, "no fixture slot sampled a non-local member")

    def test_3c_successor_groups(self):
        data = _data()
        corp, fit = data.corpus, data.fit
        for ctx, plan in fit.plans.items():
            if plan.stream == "group":
                self.assertEqual(len(plan.teachers), 4)
            else:
                self.assertIn(plan.reason, sampler.PLAN_REASONS)
        groups = slots = 0
        for step_draw in _full_draws():
            for s in step_draw.slots:
                slots += 1
                plan = fit.plans[s.context_sha256]
                self.assertEqual(s.stream, plan.stream)
                if s.stream == "group":
                    self._check_group(corp, plan, s)
                    groups += 1
        group_mass = math.fsum(fit.pi[c] for c, p in fit.plans.items() if p.stream == "group")
        self.assertGreater(groups, 0)
        self.assertLess(abs(groups / slots - group_mass), 0.03)       # about 7 standard errors at 8000 slots

    def test_3c_feasibility_stream_contexts(self):
        data = _data()
        corp, fit = data.corpus, data.fit
        specials = fx.special_contexts()
        for name, reason in FEASIBILITY_SPECIALS.items():
            self.assertEqual((fit.plans[specials[name]].stream, fit.plans[specials[name]].reason),
                             ("feasibility", reason))
        seen: Counter = Counter()
        collated = 0
        for step_draw in _full_draws():
            feasibility = [s for s in step_draw.slots if s.stream == "feasibility"]
            for s in feasibility:
                plan = fit.plans[s.context_sha256]
                self.assertEqual(plan.stream, "feasibility")
                self.assertEqual(len(s.members), min(8, len(plan.labelled)))
                self.assertLessEqual(len(s.members), 8)
                self.assertEqual(len(set(s.members)), len(s.members))
                self.assertEqual(s.roles, ("feasibility",) * len(s.members))
                for m in s.members:
                    self.assertIsNotNone(corp.label(s.context_sha256, m).feas_class)
                seen[s.context_sha256] += 1
            if feasibility and collated < 100:
                collated += 1
                b = sampler.collate(step_draw, data)
                for ci, s in enumerate([s for s in step_draw.slots if s.members]):
                    self.assertEqual(b.ctx_ids[ci], s.context_sha256)
                    is_group = s.stream == "group"
                    self.assertEqual(b.ctx_weight[ci].item(), 1.0 if is_group else float(np.float32(0.1)))
                    self.assertEqual(bool(b.ctx_is_group[ci]), is_group)
                    rows = (b.seq_ctx == ci).nonzero().flatten().tolist()
                    self.assertEqual([b.member_ids[r] for r in rows], [(s.context_sha256, m) for m in s.members])
                    self.assertEqual(b.role[rows].tolist(), [sampler.ROLE_CODES[r] for r in s.roles])
        for name in FEASIBILITY_SPECIALS:
            self.assertGreater(seen[specials[name]], 0, f"special {name} never drawn")
        self.assertGreater(collated, 0)

    def test_3c_baseline_mode_starts_with_the_champion_or_is_skipped(self):
        data = fx.train_data(group_mode="baseline-v1")
        corp, fit = data.corpus, data.fit
        s = sampler.MacroSampler(fit, fx.RUN_SEED, data.config)
        streams: Counter = Counter()
        skipped: set[str] = set()
        for _ in range(500):
            step_draw = s.draw_next()
            for sd in step_draw.slots:
                ctx = sd.context_sha256
                plan = fit.plans[ctx]
                rows = [r for r in corp.labels.records[ctx] if r.q_status == "known" and r.value is not None]
                streams[sd.stream] += 1
                if sd.stream == "skipped":
                    self.assertIsNone(plan.baseline)
                    self.assertEqual((sd.members, sd.roles, sd.class_counts, sd.feas_class_counts),
                                     ((), (), (0, 0, 0), (0, 0, 0)))
                    self.assertLess(len(rows), 8)
                    skipped.add(ctx)
                    continue
                self.assertEqual(sd.stream, "group")
                champion, bands = plan.baseline
                self.assertEqual(len(bands), 7)
                self.assertEqual((sd.members[0], sd.roles), (champion, ("champion",) + ("comparator",) * 7))
                self.assertEqual(corp.label(ctx, champion).value, max(r.value for r in rows))
                sizes = [len(b) for b in bands]
                self.assertEqual((sum(sizes), max(sizes) - min(sizes) <= 1), (len(rows) - 1, True))
                for upper, lower in zip(bands, bands[1:]):
                    self.assertGreaterEqual(min(labels.q_grid(corp.label(ctx, p).value) for p in upper),
                                            max(labels.q_grid(corp.label(ctx, p).value) for p in lower))
                for i, m in enumerate(sd.members[1:]):
                    self.assertIn(m, bands[i])
                self.assertEqual(sd.class_counts, (8, 0, 0))
                self.assertEqual(sd.feas_class_counts, (sum(corp.label(ctx, m).feas_valid for m in sd.members), 0, 0))
            if any(sd.members for sd in step_draw.slots):
                b = sampler.collate(step_draw, data)
                self.assertEqual(b.mode, "baseline-v1")
                self.assertTrue(bool(b.ctx_is_group.all()))
                self.assertTrue(bool((b.ctx_weight == 1.0).all()))
                self.assertEqual(b.role.tolist(), ([0] + [1] * 7) * len(b.ctx_ids))
                self.assertTrue(bool(b.q_known.all()))
        sp = fx.special_contexts()
        self.assertGreater(streams["group"], 0)
        self.assertEqual(skipped, {sp["degenerate"], sp["no_detail"]})

    def test_collate_layout(self):
        data = _data()
        corp, fit = data.corpus, data.fit
        for step_draw in _full_draws()[:60]:
            b = sampler.collate(step_draw, data)
            slots = [s for s in step_draw.slots if s.members]
            n, c = sum(len(s.members) for s in slots), len(slots)
            t_max = int(b.lengths.max())
            self.assertEqual((b.mode, b.step, len(b.ctx_ids), len(b.member_ids)), ("successor-v1", step_draw.step, c, n))
            self.assertEqual((tuple(b.tokens.shape), b.tokens.dtype), ((n, t_max), torch.int64))
            self.assertEqual((tuple(b.legal.shape), b.legal.dtype), ((n, t_max - 1, 32), torch.bool))
            self.assertEqual((tuple(b.ctx_z.shape), b.ctx_z.dtype), ((c, 8), torch.float32))
            for name in ("q", "q_opt", "midrank_pct", "value", "value_pct", "rt_target", "imitation_divisor"):
                self.assertEqual((tuple(getattr(b, name).shape), getattr(b, name).dtype), ((n,), torch.float32))
            for name in ("q_known", "gate_pass", "feas_valid", "rt_valid", "rt_censored"):
                self.assertEqual((tuple(getattr(b, name).shape), getattr(b, name).dtype), ((n,), torch.bool))
            self.assertEqual(b.n_tokens, int((b.lengths - 1).sum()))
            for ci, ctx in enumerate(b.ctx_ids):
                self.assertTrue(np.array_equal(b.ctx_z[ci].numpy(), corp.encodings[ctx]))
            for i, (ctx, prog) in enumerate(b.member_ids):
                toks = corp.tokens(ctx, prog)
                length = len(toks)
                self.assertEqual(int(b.lengths[i]), length)
                self.assertEqual(b.tokens[i, :length].tolist(), list(toks))
                self.assertFalse(bool(b.tokens[i, length:].any()))                     # PAD 0
                self.assertTrue(torch.equal(b.legal[i, :length - 1], torch.from_numpy(fx.legal_mask(toks))))
                self.assertFalse(bool(b.legal[i, length - 1:].any()))
                lab = corp.label(ctx, prog)
                known = lab.q_status == "known"
                self.assertEqual(bool(b.q_known[i]), known)
                self.assertEqual(int(b.q_grid[i]), lab.q_grid if known else 0)
                self.assertEqual(b.q_opt[i].item(), float(np.float32(lab.q_opt)) if known else 0.0)
                self.assertEqual(int(b.feas_class[i]), -1 if lab.feas_class is None else lab.feas_class)
                self.assertEqual(bool(b.gate_pass[i]), lab.gate_pass is True)
                self.assertEqual((bool(b.feas_valid[i]), bool(b.rt_valid[i]), bool(b.rt_censored[i])),
                                 (lab.feas_valid, lab.rt_valid, lab.rt_censored))
                divisor = fit.imitation_divisor.get(prog, 1) if lab.shared_seed else 1
                self.assertEqual(b.imitation_divisor[i].item(), float(divisor))
                self.assertGreaterEqual(b.imitation_divisor[i].item(), 1.0)
            b.ctx_z.zero_()                                       # augmentation edits a copy, never the corpus
            self.assertTrue(np.array_equal(sampler.collate(step_draw, data).ctx_z[0].numpy(), corp.encodings[b.ctx_ids[0]]))
        self.assertGreater(max(fit.imitation_divisor.values()), 1)


class DeterminismTests(unittest.TestCase):
    """P7 test 3d (PLAN §6.6: draws are sha256(run_seed ‖ step ‖ slot), so the cursor fixes every future draw)."""

    def test_3d_same_seed_identical_draws_and_journal(self):
        data = _data()
        rebuilt = fx.train_data()
        self.assertEqual(common.canonical(rebuilt.fit.statistics()), common.canonical(data.fit.statistics()))
        self.assertEqual(rebuilt.fit.sha256(), data.fit.sha256())
        a = sampler.MacroSampler(data.fit, fx.RUN_SEED, data.config)
        b = sampler.MacroSampler(rebuilt.fit, fx.RUN_SEED, rebuilt.config)
        da = [a.draw_next() for _ in range(300)]
        db = [b.draw_next() for _ in range(300)]
        self.assertEqual(da, db)
        self.assertEqual(_journal_bytes(da), _journal_bytes(db))
        self.assertEqual(da, _full_draws()[:300])
        self.assertEqual(a.draw(5), da[4])                      # draw() is pure and leaves the cursor alone
        self.assertEqual(a.cursor, 300)
        for step_draw in (da[0], da[6], da[299]):
            self.assertEqual(_batch_digest(sampler.collate(step_draw, data)),
                             _batch_digest(sampler.collate(step_draw, rebuilt)))

    def test_3d_another_seed_differs(self):
        data = _data()
        other = sampler.MacroSampler(data.fit, OTHER_SEED, data.config)
        draws = [other.draw_next() for _ in range(300)]
        self.assertNotEqual(_journal_bytes(draws), _journal_bytes(_full_draws()[:300]))
        self.assertEqual(sum(x != y for x, y in zip(draws, _full_draws()[:300])), 300)

    def test_3d_state_round_trip_continues_identically(self):
        data = _data()
        a = sampler.MacroSampler(data.fit, fx.RUN_SEED, data.config)
        for _ in range(37):
            a.draw_next()
        state = json.loads(common.canonical(a.state_dict()))
        self.assertEqual(state, {"schema": "jope.dl-sampler-state/v1", "cursor": 37,
                                 "run_seed_sha256": hashlib.sha256(fx.RUN_SEED).hexdigest(),
                                 "fit_sha256": data.fit.sha256(), "config_sha256": data.config.sha256()})
        b = sampler.MacroSampler(data.fit, fx.RUN_SEED, data.config)
        b.load_state_dict(state)
        self.assertEqual(b.cursor, 37)
        rest_a = [a.draw_next() for _ in range(43)]
        rest_b = [b.draw_next() for _ in range(43)]
        self.assertEqual(rest_a, rest_b)
        self.assertEqual(rest_b[0].step, 38)
        self.assertEqual(rest_b, _full_draws()[37:80])
        self.assertEqual(a.state_dict(), b.state_dict())
        c = sampler.MacroSampler(data.fit, fx.RUN_SEED, data.config)
        for cursor in (0, 1999, 12):                             # the cursor alone fixes the next draw
            c.load_state_dict(dict(state, cursor=cursor))
            self.assertEqual(c.draw_next(), _full_draws()[cursor])

    def test_3d_load_refuses_mismatched_state(self):
        data = _data()
        a = sampler.MacroSampler(data.fit, fx.RUN_SEED, data.config)
        a.draw_next()
        state = a.state_dict()

        def fresh() -> sampler.MacroSampler:
            return sampler.MacroSampler(data.fit, fx.RUN_SEED, data.config)

        bad_states = [dict(state, **{key: "0" * 64}) for key in ("fit_sha256", "config_sha256", "run_seed_sha256")]
        bad_states += [dict(state, schema="jope.dl-sampler-state/v0"), dict(state, cursor=-1), dict(state, cursor=True),
                       dict(state, cursor=1.5), {k: v for k, v in state.items() if k != "cursor"}]
        for bad in bad_states:
            s = fresh()
            with self.subTest(bad=bad), self.assertRaises(sampler.SamplerError):
                s.load_state_dict(bad)
            self.assertEqual(s.cursor, 0)
        with self.assertRaises(sampler.SamplerError):
            sampler.MacroSampler(data.fit, OTHER_SEED, data.config).load_state_dict(state)
        wider = fx.train_data(b_ctx=5)
        with self.assertRaises(sampler.SamplerError):
            sampler.MacroSampler(wider.fit, fx.RUN_SEED, wider.config).load_state_dict(state)
        ctxs, recs, enc = fx.contexts(), fx.records(), fx.encodings()
        drop = next(i for i, r in enumerate(recs)
                    if r["split_role"] == "train_fit" and r["outcome"] == "completed" and not r["shared_seed"])
        other_fit = sampler.fit(sampler.Corpus.build(ctxs, recs[:drop] + recs[drop + 1:], enc, fx.assignment()),
                                data.config)
        self.assertNotEqual(other_fit.sha256(), data.fit.sha256())
        with self.assertRaises(sampler.SamplerError):
            sampler.MacroSampler(other_fit, fx.RUN_SEED, data.config).load_state_dict(state)
        with self.assertRaises(sampler.SamplerError):
            sampler.MacroSampler(data.fit, fx.RUN_SEED, fx.sampler_config(b_ctx=5))
        with self.assertRaises(sampler.SamplerError):
            sampler.MacroSampler(data.fit, b"short seed", data.config)

    def test_uniform_index_is_the_specified_rejection_sampler(self):
        def word(seed: bytes, step: int, slot: int, purpose: bytes, k: int) -> int:
            msg = b"jope.dl-sampler/v1\0" + seed + step.to_bytes(8, "big") + slot.to_bytes(4, "big") + purpose
            return int.from_bytes(hashlib.sha256(msg + b"\0" + k.to_bytes(4, "big")).digest()[:8], "big")

        def reference(seed: bytes, step: int, slot: int, purpose: bytes, n: int) -> int:
            k = 0
            while word(seed, step, slot, purpose, k) >= (2**64 // n) * n:
                k += 1
            return word(seed, step, slot, purpose, k) % n

        for n in (1, 2, 3, 7, 9):
            for step in (0, 1, 37, 2**40):
                for slot in (0, 3):
                    got = sampler.uniform_index(fx.RUN_SEED, step, slot, b"family", n)
                    self.assertEqual(got, reference(fx.RUN_SEED, step, slot, b"family", n))
                    self.assertTrue(0 <= got < n)
        big = 2**63 + 1                                     # rejects about half of all words: exercises k > 0
        rejected = 0
        for step in range(64):
            self.assertEqual(sampler.uniform_index(fx.RUN_SEED, step, 0, b"x", big),
                             reference(fx.RUN_SEED, step, 0, b"x", big))
            rejected += word(fx.RUN_SEED, step, 0, b"x", 0) >= big
        self.assertGreater(rejected, 0)
        for args in ((fx.RUN_SEED, 1, 0, b"family", 0), (fx.RUN_SEED, -1, 0, b"family", 3),
                     (fx.RUN_SEED, 1, 2**32, b"family", 3), (b"short", 1, 0, b"family", 3),
                     (fx.RUN_SEED, 1, 0, b"fam\0ily", 3)):
            with self.assertRaises(sampler.SamplerError):
                sampler.uniform_index(*args)


class CorpusValidationTests(unittest.TestCase):
    """Corpus.build and fit refuse inconsistent inputs rather than widen the universe (PLAN §6.2)."""

    def test_build_refuses_inconsistent_inputs(self):
        ctxs, recs, enc, asg = fx.contexts(), fx.records(), fx.encodings(), fx.assignment()
        first = recs[0]
        wrong_length = dict(copy.deepcopy(first), length=first["length"] + 1)
        wrong_role = dict(copy.deepcopy(first), split_role="heldout" if first["split_role"] != "heldout" else "dev")
        orphan = dict(copy.deepcopy(first), context_sha256="0" * 64)
        narrow = dict(enc, **{first["context_sha256"]: np.zeros(7, dtype=np.float32)})
        nan_enc = dict(enc, **{first["context_sha256"]: np.full(8, np.nan, dtype=np.float32)})
        cases = {
            "duplicate context": (ctxs + [copy.deepcopy(ctxs[0])], recs, enc, asg),
            "duplicate program": (ctxs, recs + [copy.deepcopy(first)], enc, asg),
            "length mismatch": (ctxs, [wrong_length] + recs[1:], enc, asg),
            "record disagrees with its context": (ctxs, [wrong_role] + recs[1:], enc, asg),
            "record of an unknown context": (ctxs, recs + [orphan], enc, asg),
            "encoding width": (ctxs, recs, narrow, asg),
            "non-finite encoding": (ctxs, recs, nan_enc, asg),
            "encoding of an unknown context": (ctxs, recs, dict(enc, **{"0" * 64: np.zeros(8, np.float32)}), asg),
            "family without assignment": (ctxs, recs, enc, {k: v for k, v in asg.items() if k != "fam-00"}),
        }
        for name, args in cases.items():
            with self.subTest(name), self.assertRaises(sampler.SamplerError):
                sampler.Corpus.build(*args)

    def test_fit_refuses_cross_partition_split_roles_and_an_empty_universe(self):
        cfg = fx.sampler_config()

        def relabel(predicate, role: str) -> sampler.Corpus:
            ctxs, recs = fx.contexts(), fx.records()
            for row in ctxs + recs:
                if predicate(row):
                    row["split_role"] = role
            return sampler.Corpus.build(ctxs, recs, fx.encodings(), fx.assignment())

        one_train = next(c["context_sha256"] for c in fx.contexts() if c["split_role"] == "train_fit")
        one_dev = next(c["context_sha256"] for c in fx.contexts() if c["split_role"] == "dev")
        with self.assertRaises(sampler.SamplerError):
            sampler.fit(relabel(lambda r: r["context_sha256"] == one_train, "heldout"), cfg)
        with self.assertRaises(sampler.SamplerError):
            sampler.fit(relabel(lambda r: r["context_sha256"] == one_dev, "train_fit"), cfg)
        with self.assertRaises(sampler.SamplerError):
            sampler.fit(relabel(lambda r: r["split_role"] == "train_fit", "diag"), cfg)


class TrainDataIdentityTests(unittest.TestCase):
    """PLAN §6.6 identities: TrainData.content_sha256 binds what a training step reads from the train_fit universe
    (encodings, tokens, labels and grammar masks), including inputs the sampler fit does not hash."""

    @staticmethod
    def _data_from(ctxs, recs, enc) -> sampler.TrainData:
        cfg = fx.sampler_config()
        corp = sampler.Corpus.build(ctxs, recs, enc, fx.assignment())
        return sampler.TrainData(corp, sampler.fit(corp, cfg), cfg, fx.legal_mask, fx.V, fx.ENC_DIM)

    def test_content_identity_tracks_encodings_labels_and_grammar(self):
        base = _data()
        digest = base.content_sha256
        self.assertEqual(len(digest), 64)
        self.assertEqual(fx.train_data().content_sha256, digest)
        ctxs, recs, enc = fx.contexts(), fx.records(), fx.encodings()
        target = sorted(base.fit.pi)[0]
        shifted = dict(enc, **{target: enc[target] + np.float32(0.5)})
        slower = [dict(r, wall_us=2 * r["wall_us"])
                  if r["context_sha256"] in base.fit.pi and r["outcome"] == "completed" and r["wall_us"] else r
                  for r in recs]
        for what, data in (("re-encoded universe context", self._data_from(ctxs, recs, shifted)),
                           ("doubled completed runtimes", self._data_from(ctxs, slower, enc))):
            with self.subTest(what):
                self.assertEqual(data.fit.sha256(), base.fit.sha256(), "the fit alone cannot see this change")
                self.assertNotEqual(data.content_sha256, digest)

        def wider(tokens):
            mask = fx.legal_mask(tokens).copy()
            mask[:, fx.V - 1] = True
            return mask

        args = (base.corpus, base.fit, base.config)
        self.assertNotEqual(sampler.TrainData(*args, wider, fx.V, fx.ENC_DIM).content_sha256, digest)
        stamped = {"grammar": "a" * 64}
        a = sampler.TrainData(*args, fx.legal_mask, fx.V, fx.ENC_DIM, refs=stamped)
        b = sampler.TrainData(*args, wider, fx.V, fx.ENC_DIM, refs=stamped)
        self.assertEqual(a.content_sha256, b.content_sha256, "a stamped grammar ref replaces the mask hashing")
        self.assertNotEqual(a.content_sha256, digest)
        self.assertNotEqual(sampler.TrainData(*args, fx.legal_mask, fx.V, fx.ENC_DIM, refs={"grammar": "b" * 64})
                            .content_sha256, a.content_sha256)
        for bad in ({"heldout": "x"}, {"corpus": float("nan")}, ["corpus"]):
            with self.subTest(refs=bad), self.assertRaises(ValueError):
                sampler.TrainData(*args, fx.legal_mask, fx.V, fx.ENC_DIM, refs=bad)


class ClosurePoisonTests(unittest.TestCase):
    """P7 test 7 (PLAN §6.2: every fit uses train_fit only): poisoning dev, diag, heldout and sealed records, contexts
    and encodings leaves the fit, the train_fit labels and every collated batch byte-identical."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = fx.sampler_config()
        ctxs, recs, enc = fx.contexts(), fx.records(), fx.encodings()
        cls.clean = sampler.Corpus.build(ctxs, recs, enc, fx.assignment())
        cls.dirty = sampler.Corpus.build(*fx.poison(ctxs, recs, enc), fx.assignment())
        cls.fit_clean = sampler.fit(cls.clean, cls.cfg)
        cls.fit_dirty = sampler.fit(cls.dirty, cls.cfg)

    def test_7_poison_reaches_every_excluded_unit(self):
        targets = [sha for sha, row in self.clean.contexts.items() if row["split_role"] != "train_fit"]
        self.assertEqual(Counter(self.clean.contexts[t]["split_role"] for t in targets),
                         {"diag": 8, "dev": 9, "heldout": 5, "sealed_excluded": 2})
        for sha in targets:
            self.assertNotEqual(_label_bytes(self.clean.labels.contexts[sha]), _label_bytes(self.dirty.labels.contexts[sha]))
            self.assertEqual(len(self.dirty.records[sha]), len(self.clean.records[sha]) + 2)
            self.assertTrue(np.array_equal(self.dirty.encodings[sha], -self.clean.encodings[sha]))
            self.assertNotEqual(sampler.context_plan(self.clean, sha, self.cfg),
                                sampler.context_plan(self.dirty, sha, self.cfg))

    def test_7_fit_statistics_weights_and_sha_byte_identical(self):
        self.assertEqual(common.canonical(self.fit_clean.statistics()), common.canonical(self.fit_dirty.statistics()))
        self.assertEqual(self.fit_clean.weights_bytes(), self.fit_dirty.weights_bytes())
        self.assertEqual(self.fit_clean.sha256(), self.fit_dirty.sha256())

    def test_7_train_fit_labels_byte_identical(self):
        checked = 0
        for sha, row in self.clean.contexts.items():
            if row["split_role"] != "train_fit":
                continue
            checked += 1
            self.assertEqual(_label_bytes(self.clean.labels.contexts[sha]), _label_bytes(self.dirty.labels.contexts[sha]))
            self.assertEqual([_label_bytes(r) for r in self.clean.labels.records[sha]],
                             [_label_bytes(r) for r in self.dirty.labels.records[sha]])
            self.assertEqual(json.dumps(self.clean.records[sha], sort_keys=True),
                             json.dumps(self.dirty.records[sha], sort_keys=True))
            self.assertEqual(self.clean.encodings[sha].tobytes(), self.dirty.encodings[sha].tobytes())
        self.assertEqual(checked, 42)

    def test_7_train_data_content_identity_unchanged(self):
        data_clean = sampler.TrainData(self.clean, self.fit_clean, self.cfg, fx.legal_mask, fx.V, fx.ENC_DIM)
        data_dirty = sampler.TrainData(self.dirty, self.fit_dirty, self.cfg, fx.legal_mask, fx.V, fx.ENC_DIM)
        self.assertEqual(data_clean.content_sha256, data_dirty.content_sha256)

    def test_7_collated_batches_byte_identical_for_steps_1_to_200(self):
        data_clean = sampler.TrainData(self.clean, self.fit_clean, self.cfg, fx.legal_mask, fx.V, fx.ENC_DIM)
        data_dirty = sampler.TrainData(self.dirty, self.fit_dirty, self.cfg, fx.legal_mask, fx.V, fx.ENC_DIM)
        s_clean = sampler.MacroSampler(self.fit_clean, fx.RUN_SEED, self.cfg)
        s_dirty = sampler.MacroSampler(self.fit_dirty, fx.RUN_SEED, self.cfg)
        for step in range(1, POISON_STEPS + 1):
            a, b = s_clean.draw_next(), s_dirty.draw_next()
            self.assertEqual((a.step, b.step), (step, step))
            self.assertEqual(common.canonical(a.journal_record()), common.canonical(b.journal_record()))
            self.assertEqual(_batch_digest(sampler.collate(a, data_clean)), _batch_digest(sampler.collate(b, data_dirty)),
                             f"step {step}")


if __name__ == "__main__":
    unittest.main()

"""successor_targets tests on a tempdir corpus shaped like replay_policy.groups() expects."""

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import successor_targets as st
from resume_expired_cycle import atomic, sha

EVALUATOR = "9" * 64
CHECKPOINT = "c" * 64
ONESTEP = "onestep_forced_mutation_v2"
TIERB = "bounded_seed_lineage_v3_cap2s"
TIERB4 = "bounded_seed_lineage_v3_cap4s"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def context(prepared, cut=0, family=None, partition="meta_train", n=20, base=1.0, step=0.1):
    """Programs 0..12 at base + step*i (0 = champion, 1..12 = candidates); 13.. interleave into the same range (controls)."""
    return dict(
        prepared=prepared,
        cut=cut,
        family=family or "family-" + prepared,
        partition=partition,
        programs=[
            ("program-%s-%d" % (prepared, i), base + step * i if i <= 12 else base + step * (i - 12) + 0.03)
            for i in range(n)
        ],
    )


def median_of(ctx):
    import statistics

    return statistics.median(rmse for _, rmse in ctx["programs"])


def corpus(root, contexts):
    path = root / "corpus"
    path.mkdir(parents=True)
    records, labels, index, number = (
        bytearray(),
        [],
        dict(record_bytes=532, vocabulary_sha256="v", context_index_sha256="c", contexts=[], groups=[]),
        0,
    )
    for g, ctx in enumerate(contexts):
        ids = []
        for name, rmse in ctx["programs"]:
            record = bytearray(532)
            struct.pack_into("<H", record, 0, number + 1)
            struct.pack_into("<H", record, 512, 1)
            struct.pack_into("<fff", record, 520, 0.4, 0.5, 1)
            records.extend(record)
            labels.append(dict(record=number, context=g, program=name, rmse_cut_normalized=rmse))
            ids.append(number)
            number += 1
        index["contexts"].append(
            dict(
                prepared_sha256=ctx["prepared"],
                cut=ctx["cut"],
                family=ctx["family"],
                partition=ctx["partition"],
            )
        )
        index["groups"].append(dict(context=g, family=ctx["family"], partition=ctx["partition"], records=ids))
    (path / "records.bin").write_bytes(records)
    (path / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labels))
    index.update(records=number, records_sha256=sha(path / "records.bin"))
    atomic(path / "index.json", index)
    atomic(
        path / "provenance.json",
        dict(index_sha256=sha(path / "index.json"), labels_sha256=sha(path / "labels.jsonl")),
    )
    return path


def row(
    ctx,
    role,
    program,
    r0,
    child,
    arm="neural",
    replicate=0,
    replicate_pair=None,
    status="measured",
    evaluator=EVALUATOR,
    partition="meta_train",
    protocol=ONESTEP,
):
    key = "%s:%d" % (ctx["prepared"], ctx["cut"])
    identity = digest(("%s|%s|%s|%d|%s" % (key, role, program, replicate, protocol)).encode())
    return dict(
        schema="parallel.seed_value.v1",
        generation=73,
        dataset="ds-" + ctx["prepared"],
        family=ctx["family"],
        prepared_sha256=ctx["prepared"],
        cut=ctx["cut"],
        context_key=key,
        episode_id=identity,
        identity_sha256=identity,
        protocol=protocol,
        role=role,
        arm=arm,
        replicate=replicate,
        seed_origin="generic_comparison" if arm == "generic" else "neural_policy",
        seed_program_sha256=digest(program.encode()),
        seed_genome=program,
        producing_checkpoint_sha256=CHECKPOINT,
        evaluator_sha256=evaluator,
        driver_sha256=None,
        r0=r0,
        child=dict(
            program_sha256="x" * 64,
            rmse=child,
            failure=None if child is not None else "candidate_or_view_timeout",
            genetic_operator="forced_seed_mutation",
        ),
        flip=int(child is not None and child < r0),
        S_rmse=(min(r0, child) if child is not None else r0),
        replicate_pair=replicate_pair,
        status=status,
        completed_at=1789237312,
        partition=partition,
    )


def scenario(ctx, improvement=0.25, controls=4, panel=3, spread=0.001, candidates=12):
    """12 candidates (odd: child better by `improvement`, even: worse by 0.2), controls, and a replicate panel."""
    rows = []
    programs = dict(ctx["programs"])
    for i in range(1, candidates + 1):
        name = "program-%s-%d" % (ctx["prepared"], i)
        r0 = programs[name]
        child = r0 - improvement if i % 2 else r0 + 0.2
        pair = dict(child_rmse_rep0=child, child_rmse_rep1=child + spread) if i <= panel else None
        rows.append(row(ctx, "candidate", name, r0, child, replicate_pair=pair))
        if i <= panel:
            rows.append(
                row(ctx, "replicate_panel", name, r0, child + spread, replicate=1, replicate_pair=pair)
            )
    for i in range(candidates + 1, candidates + 1 + controls):
        name = "program-%s-%d" % (ctx["prepared"], i)
        rows.append(row(ctx, "control_generic", name, programs[name], programs[name] + 0.3, arm="generic"))
    return rows


def tierb_row(
    ctx,
    role,
    program,
    r0,
    lineage,
    protocol=TIERB,
    arm="neural",
    status="measured",
    termination="budget_exhausted",
    attempted=23,
):
    """A Tier-B export row as successor_ledger writes it: child = best descendant, S_rmse = S_lineage = min(r0, best)."""
    finite_lineage = [x for x in lineage if x is not None]
    best = min(finite_lineage) if finite_lineage else None
    r = row(ctx, role, program, r0, best, arm=arm, status=status, protocol=protocol)
    r["child"].update(
        genetic_operator="mutation" if best is not None else None,
        failure=None if best is not None else "no_descendant_measured",
    )
    r.update(
        attempted_descendants=attempted,
        successful_descendants=len(finite_lineage),
        termination_reason=termination,
        lineage_size=1 + len(finite_lineage),
    )
    return r


def tierb_scenario(ctx, protocol=TIERB):
    """Top-3 seeds per arm: neural candidates 1..3 (seed 3's lineage overtakes seeds 1 and 2: two flips of three pairs),
    generic controls 13..15 (best descendant r0 - 0.05), shuffled controls 16..18 (no better descendant: S = r0)."""
    programs = dict(ctx["programs"])
    rows = []
    for k, gain in ((1, 0.3), (2, 0.15), (3, 0.9)):
        name = "program-%s-%d" % (ctx["prepared"], k)
        rows.append(
            tierb_row(
                ctx,
                "candidate",
                name,
                programs[name],
                [programs[name] + 0.1, programs[name] - gain, None],
                protocol=protocol,
            )
        )
    for k in (13, 14, 15):
        name = "program-%s-%d" % (ctx["prepared"], k)
        rows.append(
            tierb_row(
                ctx,
                "control_generic",
                name,
                programs[name],
                [programs[name] + 0.2, programs[name] - 0.05],
                protocol=protocol,
                arm="generic",
            )
        )
    for k in (16, 17, 18):
        name = "program-%s-%d" % (ctx["prepared"], k)
        rows.append(
            tierb_row(
                ctx,
                "control_shuffled",
                name,
                programs[name],
                [programs[name] + 0.3],
                protocol=protocol,
                arm="shuffled",
            )
        )
    return rows


class TargetsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/dev/shm")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.c0 = context("p0")
        self.c1 = context("p1")
        self.validation = context("p2", partition="meta_validation")
        self.corpus = corpus(self.root, [self.c0, self.c1, self.validation])

    def export(self, rows, name="seed-values-73.json", **extra):
        document = dict(
            schema="parallel.seed_value.v1",
            generation=73,
            protocol="onestep_forced_mutation_v2",
            evaluator_sha256=EVALUATOR,
            rows=rows,
            counts={},
            final_tests_opened=False,
        )
        document.update(extra)
        path = self.root / name
        path.write_bytes(encoded(document))
        return path

    def test_value_conversion_ranks_and_percentile(self):
        self.assertAlmostEqual(st.value(1.95, 1.95), 0.5)
        self.assertAlmostEqual(st.value(0.0, 1.95), 1.0)
        self.assertGreater(st.value(1.0, 1.95), st.value(1.1, 1.95))
        with self.assertRaises(ValueError):
            st.value(float("nan"), 1.95)
        with self.assertRaises(ValueError):
            st.value(1.0, 0.0)
        self.assertEqual(st.ranks([3.0, 1.0, 2.0, 2.0]), [4.0, 1.0, 2.5, 2.5])
        self.assertAlmostEqual(st.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(st.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(st.spearman([1, 1, 1], [1, 2, 3]))
        self.assertEqual(st.percentile([5, 1, 3, 4, 2], 0.75), 4)
        self.assertIsNone(st.percentile([], 0.75))
        fit = st.ols3(
            [1.5, 2.0, 3.5, 4.0, 6.0], [0, 1, 2, 3, 4], [1, 0, 1, 0, 2]
        )  # y = 1 + x1 + 0.5 x2 exactly
        self.assertAlmostEqual(fit[0], 1.0, places=6)
        self.assertAlmostEqual(fit[1], 1.0, places=6)
        self.assertAlmostEqual(fit[2], 0.5, places=6)
        self.assertIsNone(st.ols3([1, 2, 3, 4], [1, 2, 3, 4], [2, 4, 6, 8]))

    def test_eps_floor_and_replicate_spread(self):
        median = 1.95
        tiny = st.eps_for(scenario(self.c0, spread=0.001), median)
        self.assertEqual(tiny, st.EPS_FLOOR)
        wide = st.eps_for(scenario(self.c0, spread=0.5), median)
        spreads = sorted(abs(st.value(c, median) - st.value(c + 0.5, median)) for c in (0.85, 1.4, 1.05))
        self.assertAlmostEqual(wide, spreads[2])
        self.assertIsNone(st.eps_for(scenario(self.c0, panel=0), median))

    def test_block_schema_groups_and_info_test(self):
        block = st.build(self.corpus, self.export(scenario(self.c0)), evaluator_sha256=EVALUATOR)
        self.assertEqual(block["schema"], "parallel.seed_values.v1")
        self.assertEqual(block["corpus_sha256"], sha(self.corpus / "index.json"))
        self.assertEqual(block["evaluator_sha256"], EVALUATOR)
        self.assertEqual(
            (block["protocol"], block["generation"], block["tie_rule"], block["per_context_cap"]),
            ("onestep_forced_mutation_v2", 73, "abs(dS_value) <= eps_c", 8),
        )
        self.assertIs(block["final_tests_opened"], False)
        self.assertEqual(block["seed_values_export_sha256"], sha(self.root / "seed-values-73.json"))
        for key in ("groups", "values", "floors", "roles", "arms", "counts", "info_test"):
            self.assertIn(key, block)
        self.assertEqual(block["counts"]["contexts"], 1)
        self.assertEqual(
            block["counts"]["groups"], 2
        )  # 16 eligible seeds -> two full groups of 7 + champion, 2 leftovers
        self.assertEqual(block["counts"]["incomplete_groups"], 1)
        self.assertEqual(block["counts"]["ties_excluded"], 0)
        self.assertEqual(block["counts"]["refused_contexts"], 0)
        self.assertEqual(block["counts"]["label_mismatch_rows"], 0)
        median = median_of(self.c0)
        self.assertAlmostEqual(block["contexts"]["p0:0"]["median_rmse"], median)
        for index, group in enumerate(block["groups"]):
            self.assertEqual(len(group), 8)
            self.assertEqual(len(set(group)), 8)
            self.assertEqual(group[0], 0)  # champion record (program-p0-0, rmse 1.0) first
            self.assertEqual(block["roles"][str(group[0])], "champion")
            values = [block["values"][str(r)] for r in group[1:]]
            self.assertEqual(values, sorted(values, reverse=True))
            self.assertEqual(block["floors"][str(index)], st.EPS_FLOOR)
        self.assertAlmostEqual(block["values"]["0"], st.value(1.0, median))
        first_seed = block["groups"][0][1]
        self.assertAlmostEqual(
            block["values"][str(first_seed)], st.value(0.85, median)
        )  # program-p0-1: r0 1.1, child 0.85
        self.assertEqual(block["roles"][str(first_seed)], "candidate")
        self.assertEqual(block["arms"][str(first_seed)], "neural")
        self.assertIn("generic", block["arms"].values())
        info = block["info_test"]
        self.assertEqual((info["p_flip_flips"], info["p_flip_pairs"]), (5, 66))
        self.assertAlmostEqual(info["p_flip"], 5 / 66)
        self.assertFalse(info["passed_a"])
        self.assertFalse(info["passed_b"])
        self.assertFalse(info["passed"])
        self.assertIsNone(info["a_own_ci"])
        self.assertEqual(info["a_own"]["reason"], "next_outcome_unavailable")
        contrast = info["conditional_contrast"]
        self.assertEqual(contrast["control_role"], "control_generic")
        self.assertEqual(contrast["contexts"], 1)
        self.assertGreater(contrast["cells"], 0)
        self.assertIsNotNone(contrast["overall"]["mean"])
        self.assertIsNone(contrast["overall"]["ci90"])  # one cluster: no bootstrap interval
        # A_ctx (plan D1): value(S) - median over the context's control_generic seeds of value(S), per record and per arm.
        a_ctx = block["a_ctx"]
        self.assertEqual(a_ctx["control_role"], "control_generic")
        self.assertEqual((a_ctx["contexts_with_control"], a_ctx["contexts_without_control"]), (1, []))
        programs = dict(self.c0["programs"])
        controls = sorted(
            st.value(programs["program-p0-%d" % i], median) for i in range(13, 17)
        )  # S = r0: children are worse
        reference = (controls[1] + controls[2]) / 2
        self.assertAlmostEqual(block["contexts"]["p0:0"]["a_ctx_control_median"], reference)
        self.assertAlmostEqual(a_ctx["by_arm"]["generic"], 0.0)
        neural = sorted(
            st.value(
                min(
                    programs["program-p0-%d" % i],
                    programs["program-p0-%d" % i] - 0.25 if i % 2 else programs["program-p0-%d" % i] + 0.2,
                ),
                median,
            )
            - reference
            for i in range(1, 13)
        )
        self.assertAlmostEqual(a_ctx["by_arm"]["neural"], (neural[5] + neural[6]) / 2)
        self.assertIsNone(a_ctx["by_arm"]["shuffled"])
        self.assertEqual(a_ctx["n_by_arm"], dict(neural=12, generic=4, shuffled=0))
        self.assertAlmostEqual(a_ctx["per_record"][str(first_seed)], st.value(0.85, median) - reference)
        self.assertEqual(
            len(a_ctx["per_record"]), 16
        )  # every joined seed, consumed or not (two leftovers never enter a group)
        self.assertTrue({str(r) for g in block["groups"] for r in g[1:]} <= set(a_ctx["per_record"]))
        uncontrolled = st.build(self.corpus, self.export(scenario(self.c0, controls=0)))
        self.assertEqual(uncontrolled["a_ctx"]["contexts_without_control"], ["p0:0"])
        self.assertEqual(uncontrolled["a_ctx"]["by_arm"], dict(neural=None, generic=None, shuffled=None))
        self.assertNotIn("a_ctx_control_median", uncontrolled["contexts"]["p0:0"])
        drifted = scenario(self.c0)
        drifted[0]["r0"] = drifted[0]["r0"] + 1e-9  # export r0 no longer bit-equal to the corpus label
        block = st.build(self.corpus, self.export(drifted))
        self.assertEqual(block["counts"]["label_mismatch_rows"], 1)
        self.assertEqual(
            block["label_mismatches"][0]["seed_program_sha256"], drifted[0]["seed_program_sha256"]
        )

    def test_ties_are_excluded_and_thin_groups_dropped(self):
        rows = scenario(self.c0)
        twin = next(r for r in rows if r["seed_genome"] == "program-p0-4")
        source = next(r for r in rows if r["seed_genome"] == "program-p0-2")
        twin["child"]["rmse"], twin["S_rmse"] = (
            source["S_rmse"],
            source["S_rmse"],
        )  # exact tie in S; own label (r0) untouched
        block = st.build(self.corpus, self.export(rows))
        self.assertEqual(block["counts"]["ties_excluded"], 1)
        self.assertEqual(block["counts"]["groups"], 2)
        flat = [r for g in block["groups"] for r in g[1:]]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertLessEqual(
            sum(1 for r in flat if block["values"][str(r)] == st.value(source["S_rmse"], median_of(self.c0))),
            1,
        )
        collapsed = scenario(self.c0)
        for r in collapsed:
            if r["role"] != "replicate_panel":
                r["r0"], r["child"]["rmse"], r["S_rmse"] = 1.5, 1.5, 1.5
        with self.assertRaisesRegex(ValueError, "no context produced a consumable group"):
            st.build(self.corpus, self.export(collapsed))

    def test_cap_of_eight_groups_per_context(self):
        big = context("p0", n=80)
        self.corpus = corpus(self.root / "big", [big])
        rows = scenario(big, controls=60)
        block = st.build(self.corpus, self.export(rows))
        self.assertEqual(block["counts"]["groups"], 8)
        self.assertEqual(len(block["floors"]), 8)

    def test_refusal_paths(self):
        rows = scenario(self.c0)
        with self.assertRaisesRegex(ValueError, "corpus mismatch"):
            st.build(self.corpus, self.export(rows), corpus_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "evaluator mismatch"):
            st.build(self.corpus, self.export(rows), evaluator_sha256="1" * 64)
        with self.assertRaisesRegex(ValueError, "replicate panel unavailable"):
            st.build(self.corpus, self.export(scenario(self.c0, panel=0)))
        with self.assertRaisesRegex(ValueError, "non-meta_train corpus group"):
            st.build(self.corpus, self.export(scenario(self.validation)))
        sealed = scenario(self.c0)
        sealed[0]["partition"] = "meta_validation"
        with self.assertRaisesRegex(ValueError, "non-meta_train seed-value rows"):
            st.build(self.corpus, self.export(sealed))
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            st.build(self.corpus, self.export(rows, schema="parallel.other.v1"))
        with self.assertRaisesRegex(ValueError, "exactly one evaluator"):
            st.build(
                self.corpus,
                self.export(rows + [row(self.c1, "candidate", "program-p1-1", 1.1, 1.0, evaluator="2" * 64)]),
            )
        with self.assertRaisesRegex(ValueError, "control role"):
            st.build(self.corpus, self.export(rows), control_role="candidate")
        tied = scenario(self.c1)
        for r in tied:
            if r["role"] != "replicate_panel":
                r["r0"], r["child"]["rmse"], r["S_rmse"] = 1.5, 1.6, 1.5
        block = st.build(self.corpus, self.export(rows + tied))
        self.assertEqual(block["counts"]["contexts"], 1)
        self.assertEqual(block["counts"]["refused_contexts"], 1)
        self.assertEqual(block["refused"][0]["context_key"], "p1:0")
        self.assertTrue(block["refused"][0]["reason"].startswith("tie_fraction 1.000"))
        self.assertEqual(block["tie_fraction"]["p1:0"]["tie_fraction"], 1.0)
        unknown = scenario(self.c0) + [
            row(
                context("p9"),
                "candidate",
                "program-p9-1",
                1.1,
                1.0,
                replicate_pair=dict(child_rmse_rep0=1.0, child_rmse_rep1=1.0),
            ),
            row(
                context("p9"),
                "replicate_panel",
                "program-p9-1",
                1.1,
                1.0,
                replicate=1,
                replicate_pair=dict(child_rmse_rep0=1.0, child_rmse_rep1=1.0),
            ),
        ]
        block = st.build(self.corpus, self.export(unknown))
        self.assertEqual(block["refused"], [dict(context_key="p9:0", reason="context_not_in_corpus")])
        unjoined = scenario(self.c0) + [row(self.c0, "candidate", "program-not-in-corpus", 1.15, 1.0)]
        block = st.build(self.corpus, self.export(unjoined))
        self.assertEqual(block["counts"]["unjoined_rows"], 1)

    def test_information_tests_pass_when_flips_and_a_own_signal_exist(self):
        rows = scenario(self.c0, improvement=0.55) + scenario(self.c1, improvement=0.55)
        outcomes = {}
        for r in rows:
            if r["role"] == "candidate":
                median = median_of(self.c0)
                a_own = st.value(r["S_rmse"], median) - st.value(r["r0"], median)
                outcomes[(r["context_key"], r["seed_program_sha256"])] = 10.0 * a_own - 0.05 * r["r0"]
        block = st.build(self.corpus, self.export(rows), next_outcome=outcomes)
        info = block["info_test"]
        self.assertEqual((info["p_flip_flips"], info["p_flip_pairs"]), (24, 132))
        self.assertTrue(info["passed_a"])
        self.assertEqual(info["a_own"]["n_rows"], 24)
        self.assertEqual(info["a_own"]["n_contexts"], 2)
        self.assertGreater(info["a_own"]["coefficient"], 0)
        self.assertGreater(info["a_own_ci"]["low"], 0)
        self.assertTrue(info["passed_b"])
        self.assertTrue(info["passed"])
        self.assertEqual(block["counts"]["contexts"], 2)
        self.assertEqual(
            block["bootstrap"], dict(resamples=2000, seed=20260912, alpha=0.10, clusters="contexts")
        )
        again = st.build(self.corpus, self.export(rows), next_outcome=outcomes)
        self.assertEqual(again["info_test"]["a_own_ci"], info["a_own_ci"])
        listed = self.root / "next-outcome.json"
        listed.write_bytes(
            encoded(
                dict(
                    schema="parallel.successor_next_outcome.v1",
                    rows=[
                        dict(context_key=k[0], seed_program_sha256=k[1], outcome=v)
                        for k, v in outcomes.items()
                    ],
                )
            )
        )
        from_file = st.build(self.corpus, self.export(rows), next_outcome=listed)
        self.assertEqual(from_file["info_test"]["a_own"], info["a_own"])

    def test_tierb_rows_are_grouped_per_protocol_with_per_protocol_info_tests(self):
        rows = scenario(self.c0) + tierb_scenario(self.c0)
        block = st.build(self.corpus, self.export(rows), evaluator_sha256=EVALUATOR)
        baseline = st.build(
            self.corpus, self.export(scenario(self.c0), name="onestep.json"), evaluator_sha256=EVALUATOR
        )
        # the existing keys keep the one-step result: groups, values, tie floors, per-context info, info tests, A_ctx
        for key in (
            "groups",
            "values",
            "floors",
            "roles",
            "arms",
            "eps",
            "tie_fraction",
            "contexts",
            "refused",
            "label_mismatches",
        ):
            self.assertEqual(block[key], baseline[key], key)
        self.assertEqual(
            {k: v for k, v in block["info_test"].items() if k != "by_protocol"},
            {k: v for k, v in baseline["info_test"].items() if k != "by_protocol"},
        )
        self.assertEqual(
            {k: v for k, v in block["a_ctx"].items() if k != "by_protocol"},
            {k: v for k, v in baseline["a_ctx"].items() if k != "by_protocol"},
        )
        self.assertEqual((block["protocol"], block["protocols"]), (ONESTEP, [TIERB, ONESTEP]))
        self.assertEqual(block["group_protocols"], {str(i): ONESTEP for i in range(len(block["groups"]))})
        self.assertEqual((block["info_test"]["p_flip_flips"], block["info_test"]["p_flip_pairs"]), (5, 66))
        self.assertEqual(block["info_test"]["by_protocol"][ONESTEP]["p_flip"], block["info_test"]["p_flip"])
        # the Tier-B info tests: three candidates, seed 3's lineage overtakes seeds 1 and 2 -> 2 flips of 3 non-tie pairs
        tierb = block["info_test"]["by_protocol"][TIERB]
        self.assertEqual((tierb["p_flip_flips"], tierb["p_flip_pairs"]), (2, 3))
        self.assertAlmostEqual(tierb["p_flip"], 2 / 3)
        self.assertTrue(tierb["passed_a"])
        self.assertFalse(tierb["passed_b"])
        self.assertFalse(tierb["passed"])
        self.assertEqual(tierb["a_own"]["reason"], "next_outcome_unavailable")
        self.assertEqual(tierb["credited_statuses"], ["measured", "incomplete"])
        self.assertEqual((tierb["contexts"], tierb["measured_rows"], tierb["incomplete_rows"]), (1, 9, 0))
        self.assertEqual(tierb["conditional_contrast"]["control_role"], "control_generic")
        # tie floors are per (context, protocol): Tier-B has no replicate panel and uses the floor alone, never the one-step eps
        space = block["by_protocol"][TIERB]
        self.assertEqual(space["eps"], {"p0:0": st.EPS_FLOOR})
        self.assertTrue(space["eps_source"]["p0:0"].startswith("floor_only"))
        self.assertEqual(block["by_protocol"][ONESTEP]["eps_source"], {"p0:0": "replicate_panel"})
        self.assertEqual(block["by_protocol"][ONESTEP]["contexts"], block["contexts"])
        self.assertEqual(
            space["counts"],
            dict(contexts=0, groups=0, measured_rows=9, incomplete_rows=0, refused_contexts=0),
        )
        self.assertEqual(space["tie_fraction"]["p0:0"]["pairs"], 36)
        # A_ctx for Tier-B: value(S_lineage) - median over the context's control_generic Tier-B seeds (never the one-step controls)
        median = median_of(self.c0)
        programs = dict(self.c0["programs"])
        a_ctx = block["a_ctx"]["by_protocol"][TIERB]
        controls = sorted(st.value(programs["program-p0-%d" % k] - 0.05, median) for k in (13, 14, 15))
        self.assertAlmostEqual(a_ctx["control_median_by_context"]["p0:0"], controls[1])
        self.assertNotAlmostEqual(
            a_ctx["control_median_by_context"]["p0:0"], block["a_ctx"]["control_median_by_context"]["p0:0"]
        )
        self.assertEqual(a_ctx["n_by_arm"], dict(neural=3, generic=3, shuffled=3))
        neural = sorted(st.value(s, median) - controls[1] for s in (1.1 - 0.3, 1.2 - 0.15, 1.3 - 0.9))
        self.assertAlmostEqual(a_ctx["by_arm"]["neural"], neural[1])
        self.assertAlmostEqual(a_ctx["by_arm"]["generic"], 0.0)
        self.assertEqual(
            space["contexts"]["p0:0"]["a_ctx_control_median"], a_ctx["control_median_by_context"]["p0:0"]
        )
        # A_own = value(S_lineage) - value(r0) is still diagnostic: the Tier-B rank regression runs per protocol on next outcomes
        three = context("p3")
        corpus_dir = corpus(self.root / "three", [self.c0, self.c1, three])
        rows = tierb_scenario(self.c0) + tierb_scenario(self.c1) + tierb_scenario(three) + scenario(self.c0)
        outcomes = {}
        for r in rows:
            if r["role"] == "candidate" and r["protocol"] == TIERB:
                a_own = st.value(r["S_rmse"], median) - st.value(r["r0"], median)
                outcomes[(r["context_key"], r["seed_program_sha256"])] = 10.0 * a_own - 0.05 * r["r0"]
        block = st.build(corpus_dir, self.export(rows, name="three.json"), next_outcome=outcomes)
        tierb = block["info_test"]["by_protocol"][TIERB]
        self.assertEqual((tierb["a_own"]["n_rows"], tierb["a_own"]["n_contexts"]), (9, 3))
        self.assertIsNotNone(tierb["a_own"]["coefficient"])
        self.assertEqual((tierb["p_flip_flips"], tierb["p_flip_pairs"]), (6, 9))
        self.assertEqual(
            block["info_test"]["a_own"]["n_contexts"], 1
        )  # the one-step regression sees only the p0 one-step seeds it shares

    def test_block_consumes_one_protocol_and_tags_every_group(self):
        rows = scenario(self.c0) + tierb_scenario(self.c0) + tierb_scenario(self.c1)
        block = st.build(self.corpus, self.export(rows), protocol=TIERB)
        self.assertEqual((block["protocol"], block["protocols"]), (TIERB, [TIERB, ONESTEP]))
        self.assertEqual(set(block["group_protocols"].values()), {TIERB})
        self.assertEqual(
            (block["counts"]["contexts"], block["counts"]["groups"], block["counts"]["incomplete_groups"]),
            (2, 2, 2),
        )
        for index, group in enumerate(block["groups"]):
            self.assertEqual((len(group), len(set(group))), (8, 8))
            self.assertEqual(block["roles"][str(group[0])], "champion")
            values = [block["values"][str(r)] for r in group[1:]]
            self.assertEqual(values, sorted(values, reverse=True))
            self.assertEqual(block["floors"][str(index)], st.EPS_FLOOR)
        median = median_of(self.c0)
        self.assertEqual(block["groups"][0][1], 3)  # program-p0-3: lineage best 0.4
        self.assertAlmostEqual(block["values"]["3"], st.value(1.3 - 0.9, median))
        self.assertAlmostEqual(
            block["values"]["1"], st.value(1.1 - 0.3, median)
        )  # S_lineage, never the seed's one-step S (0.85)
        self.assertEqual(
            (block["roles"]["3"], block["arms"]["3"], block["roles"]["13"], block["arms"]["13"]),
            ("candidate", "neural", "control_generic", "generic"),
        )
        self.assertEqual(block["info_test"]["p_flip_pairs"], 66)  # existing keys: the one-step result
        self.assertEqual(block["counts"]["by_protocol"][ONESTEP]["groups"], 0)
        self.assertEqual(
            block["counts"]["by_protocol"][TIERB],
            dict(contexts=2, groups=2, measured_rows=18, incomplete_rows=0, refused_contexts=0),
        )
        self.assertEqual(block["refused"], [])
        with self.assertRaisesRegex(ValueError, "no credited rows of protocol " + TIERB4):
            st.build(self.corpus, self.export(rows, name="four.json"), protocol=TIERB4)
        with self.assertRaisesRegex(ValueError, "unknown consumption protocol"):
            st.build(self.corpus, self.export(rows), protocol="bounded_seed_lineage_v3")
        with self.assertRaisesRegex(ValueError, "unknown seed-value protocol"):
            st.build(
                self.corpus,
                self.export(
                    rows + [row(self.c1, "candidate", "program-p1-1", 1.1, 1.0, protocol="other_v9")],
                    name="alien.json",
                ),
            )
        sys.argv = [
            "successor_targets",
            "--corpus",
            str(self.corpus),
            "--export",
            str(self.export(rows, name="cli.json")),
            "--output",
            str(self.root / "policy/tierb.json"),
            "--protocol",
            TIERB,
        ]
        st.main()
        self.assertEqual(
            json.loads((self.root / "policy/tierb.json").read_bytes())["group_protocols"],
            block["group_protocols"],
        )

    def test_cap_variants_never_mix_inside_a_context(self):
        rows = tierb_scenario(self.c0) + tierb_scenario(self.c0, protocol=TIERB4) + tierb_scenario(self.c1)
        block = st.build(self.corpus, self.export(rows), protocol=TIERB)
        refusal = dict(context_key="p0:0", reason="tierb_cap_variants_mixed: %s, %s" % (TIERB, TIERB4))
        self.assertEqual(block["by_protocol"][TIERB]["refused"], [refusal])
        self.assertEqual(block["by_protocol"][TIERB4]["refused"], [refusal])
        self.assertEqual((block["counts"]["refused_contexts"], block["counts"]["contexts"]), (2, 1))
        self.assertEqual(set(block["group_protocols"].values()), {TIERB})
        self.assertEqual(list(block["by_protocol"][TIERB]["contexts"]), ["p1:0"])
        self.assertEqual(block["info_test"]["by_protocol"][TIERB]["p_flip_pairs"], 3)  # p1 only
        self.assertEqual(block["refused"], [])  # the one-step namespace is empty here
        with self.assertRaisesRegex(
            ValueError, "no context produced a consumable group: tierb_cap_variants_mixed"
        ):
            st.build(self.corpus, self.export(rows, name="four.json"), protocol=TIERB4)
        with self.assertRaisesRegex(
            ValueError, "no context produced a consumable group: tierb_cap_variants_mixed"
        ):
            st.build(
                self.corpus,
                self.export(
                    tierb_scenario(self.c0) + tierb_scenario(self.c0, protocol=TIERB4), name="mixed.json"
                ),
                protocol=TIERB,
            )

    def test_incomplete_tierb_rows_are_credited_and_counted(self):
        rows = tierb_scenario(self.c0)
        truncated = next(r for r in rows if r["seed_genome"] == "program-p0-2")
        truncated.update(status="incomplete", termination_reason="episode_timeout", attempted_descendants=8)
        block = st.build(self.corpus, self.export(rows), protocol=TIERB)
        self.assertEqual((block["counts"]["measured_rows"], block["counts"]["incomplete_rows"]), (9, 1))
        self.assertEqual(block["info_test"]["by_protocol"][TIERB]["incomplete_rows"], 1)
        self.assertEqual(block["info_test"]["by_protocol"][TIERB]["p_flip_pairs"], 3)
        self.assertIn(2, block["groups"][0])  # program-p0-2 consumed for what it measured
        stale = scenario(self.c0)
        stale[0]["status"] = "incomplete"  # one-step rows have no incomplete status
        block = st.build(self.corpus, self.export(stale, name="stale.json"))
        self.assertEqual(
            (block["counts"]["measured_rows"], block["counts"]["incomplete_rows"]), (len(stale) - 1, 0)
        )

    def test_cli_writes_immutable_block_or_refusal(self):
        export = self.export(scenario(self.c0))
        output = self.root / "policy/seed-values.json"
        argv = [
            "successor_targets",
            "--corpus",
            str(self.corpus),
            "--export",
            str(export),
            "--output",
            str(output),
            "--evaluator-sha256",
            EVALUATOR,
        ]
        sys.argv = argv
        st.main()
        first = output.read_bytes()
        sys.argv = argv
        st.main()
        self.assertEqual(output.read_bytes(), first)
        self.assertEqual(json.loads(first)["schema"], "parallel.seed_values.v1")
        refusal = self.root / "policy/refusal.json"
        sys.argv = argv[:-2] + [
            "--evaluator-sha256",
            "1" * 64,
            "--refusal-output",
            str(refusal),
            "--output",
            str(self.root / "policy/other.json"),
        ]
        with self.assertRaises(SystemExit) as stop:
            st.main()
        self.assertEqual(stop.exception.code, 1)
        body = json.loads(refusal.read_bytes())
        self.assertEqual(body["schema"], "parallel.seed_values_refusal.v1")
        self.assertTrue(body["refused"])
        self.assertIn("evaluator mismatch", body["reason"])
        self.assertFalse((self.root / "policy/other.json").exists())


if __name__ == "__main__":
    unittest.main()

"""Lane B seal and judge-evaluator fixtures.

tab4.md is not in this checkout. The test names follow spec PR-17 in the order
written there (row quarantine, fit-only chain, Rust attestation, one-time
unlock, family crosswalk). They are equivalents, not a quotation of T4 §8.4.
Every seal in this file is created under a temporary directory.
"""

import hashlib
import threading
import unittest
from pathlib import Path

from rlfw.arena.game_spec import canonical, digest
from rlfw.arena.kpi import cluster_stat
from rlfw.judge.encoding import fit_chain
from rlfw.judge.evaluator import JudgeEvaluator
from rlfw.judge.g_judge import g_curve, g_judge, judge_claim, select_program
from rlfw.judge.seal import (
    QuarantineError,
    RealSealRefusal,
    SealSpent,
    SealStore,
    build_document,
    fit_judge_split,
    freeze,
    release,
    save_new,
    seal_id_of,
)

ARM = "lane-b"
ENCODING = digest("encoding-fixture")
EVALUATOR = digest("pinned-evaluator")
CANONICALIZER = digest("canonicalizer")
PROTOCOL = "jope.rlfw-judge-q/v1"
RECEIPT = digest("receipt-fixture")


class TracingTable(dict):
    def __init__(self, rows, forbidden):
        super().__init__(rows)
        self.forbidden = set(forbidden)

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"judge row was read: {key}")
        return dict.__getitem__(self, key)


def _family(name, *, source_ids, disclosed):
    return {
        "family_id": name,
        "judge_context_id": f"judge-{name}",
        "lineage_id": f"lineage-{name}",
        "pretraining_disclosure": {"sha256": digest("disclosure"), "source_ids": disclosed},
        "role": "rights_cleared",
        "row_ids": [f"{name}-row-{index}" for index in range(10)],
        "source_ids": source_ids,
    }


def _disclosures(family_id):
    return [
        {
            "arm": ARM,
            "context_id": f"inner-{family_id}-{index}",
            "inner_q_grid": index,
            "program_sha256": digest(f"{family_id}-program-{index}"),
        }
        for index in range(1, 9)
    ]


def _document(kind="fixture", *, overlap=False):
    raw = _family("fam", source_ids=["family-src"], disclosed=["kumo-src"])
    if overlap:
        raw["source_ids"] = ["kumo-src"]
    disclosures = {"fam": _disclosures("fam")}
    return build_document([raw], disclosures, arm=ARM, seed=7, kind=kind)


def _picks(document):
    return {
        family["family_id"]: select_program(document["disclosures"][family["family_id"]], 8)
        for family in document["families"]
    }


def _materialize(root, document, name="seal.json"):
    path = Path(root) / name
    save_new(path, document)
    return path


class SplitAndQuarantineTest(unittest.TestCase):
    def test_t06_judge_rows_stay_out_of_the_fit_chain(self):
        row_ids = [f"r{index:02d}" for index in range(10)]
        fit_ids, judge_ids = fit_judge_split(row_ids, 7)
        self.assertEqual(len(fit_ids), 8)
        self.assertEqual(len(judge_ids), 2)
        self.assertEqual(set(fit_ids).isdisjoint(judge_ids), True)
        self.assertEqual(fit_judge_split(row_ids, 7), (fit_ids, judge_ids))
        table = {
            row_id: {"features": [float(index), 0.0], "target": float(index)}
            for index, row_id in enumerate(row_ids)
        }
        traced = TracingTable(table, judge_ids)
        built = fit_chain(traced, fit_ids, judge_row_ids=judge_ids, proposals=["p"])
        self.assertEqual(built["fit_row_ids"], fit_ids)
        self.assertEqual(built["judge_row_ids_ignored"], judge_ids)
        self.assertTrue(set(built["context_row_ids"]).isdisjoint(judge_ids))
        self.assertTrue(set(built["query_row_ids"]).isdisjoint(judge_ids))

    def test_t07_chain_records_fit_row_ids_and_ignores_judge_targets(self):
        row_ids = [f"r{index:02d}" for index in range(10)]
        fit_ids, judge_ids = fit_judge_split(row_ids, 7)
        table = {
            row_id: {"features": [float(index), 0.0], "target": float(index)}
            for index, row_id in enumerate(row_ids)
        }
        base = fit_chain(table, fit_ids, judge_row_ids=judge_ids, proposals=["p"])
        self.assertEqual(base["weights"]["fit_row_ids"], fit_ids)
        poisoned = {key: dict(value) for key, value in table.items()}
        poisoned[judge_ids[0]] = {
            "features": [999.0, 999.0],
            "target": 999.0,
        }
        self.assertEqual(
            fit_chain(poisoned, fit_ids, judge_row_ids=judge_ids, proposals=["p"])["vector"],
            base["vector"],
        )
        shifted = {key: {"features": list(value["features"]), "target": value["target"]} for key, value in table.items()}
        shifted[fit_ids[0]]["target"] = -50.0
        self.assertNotEqual(
            fit_chain(shifted, fit_ids, judge_row_ids=judge_ids, proposals=["p"])["cache_key"],
            base["cache_key"],
        )

    def test_refused_roles_and_split_does_not_take_features(self):
        for role in ("development", "sealed", "rights_pending", "train"):
            raw = _family("fam", source_ids=["family-src"], disclosed=["kumo-src"])
            raw["role"] = role
            with self.assertRaises(ValueError):
                build_document([raw], {"fam": _disclosures("fam")}, arm=ARM, seed=1, kind="fixture")
        self.assertEqual(len(fit_judge_split(["a", "b"], 1)[0]), 1)


class SealSpendTest(unittest.TestCase):
    def test_t14_one_release_then_spent(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            path = _materialize(root, _document())
            freeze(path, picks=_picks(_document()), encoding_sha256=ENCODING)
            first = release(path, fixture_root=root, owner_fixture_release=True)
            self.assertEqual(first["state"], "spent")
            self.assertEqual(first["release"], {"count": 1, "spent": True})
            with self.assertRaises(SealSpent):
                release(path, fixture_root=root, owner_fixture_release=True)
            self.assertEqual(SealStore(path, fixture_root=root).load()["release"]["count"], 1)

    def test_t14_concurrent_release_allows_exactly_one(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            path = _materialize(root, _document())
            freeze(path, picks=_picks(_document()), encoding_sha256=ENCODING)
            barrier = threading.Barrier(2)
            outcomes = []

            def attempt():
                barrier.wait()
                try:
                    release(path, fixture_root=root, owner_fixture_release=True)
                    outcomes.append("released")
                except SealSpent:
                    outcomes.append("spent")

            threads = [threading.Thread(target=attempt) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(sorted(outcomes), ["released", "spent"])
            self.assertEqual(SealStore(path, fixture_root=root).load()["state"], "spent")

    def test_owner_kind_and_missing_kind_do_not_change_bytes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            path = _materialize(root, _document(kind="owner"), name="owner.json")
            before = path.read_bytes()
            with self.assertRaises(RealSealRefusal):
                release(path, fixture_root=root, owner_fixture_release=True)
            self.assertEqual(path.read_bytes(), before)
            missing = Path(root) / "missing.json"
            document = _document()
            del document["kind"]
            document["seal_id"] = seal_id_of(document)
            missing.write_text(canonical(document) + "\n")
            before = missing.read_bytes()
            with self.assertRaises(RealSealRefusal):
                release(missing, fixture_root=root, owner_fixture_release=True)
            self.assertEqual(missing.read_bytes(), before)

    def test_release_refuses_a_path_outside_the_fixture_root(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as other:
            path = _materialize(root, _document())
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(RealSealRefusal):
                release(path, fixture_root=other, owner_fixture_release=True)
            with self.assertRaises(RealSealRefusal):
                release(path, fixture_root=root, owner_fixture_release=False)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_judge_rows_and_scores_wait_for_the_one_release(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            document = _document()
            path = _materialize(root, document)
            store = SealStore(path, fixture_root=root)
            picks = _picks(document)
            with self.assertRaises(RealSealRefusal):
                store.judge_rows("fam")
            freeze(path, picks=picks, encoding_sha256=ENCODING)
            with self.assertRaises(ValueError):
                freeze(path, picks=picks, encoding_sha256=ENCODING)
            evaluator = JudgeEvaluator(
                lambda _request: (_ for _ in ()).throw(AssertionError("scored before release")),
                evaluator_sha256=EVALUATOR,
                canonicalizer_sha256=CANONICALIZER,
                protocol=PROTOCOL,
            )
            with self.assertRaises(RealSealRefusal):
                evaluator.score(
                    store,
                    family_id="fam",
                    context_id="judge-fam",
                    program_sha256=picks["fam"],
                )
            release(path, fixture_root=root, owner_fixture_release=True)
            self.assertEqual(len(store.judge_rows("fam")), 2)
            with self.assertRaises(ValueError):
                evaluator.score(
                    store,
                    family_id="fam",
                    context_id="judge-fam",
                    program_sha256=digest("not-the-pick"),
                )
            self.assertEqual(store.load()["scored"], [])

    def test_tamper_and_early_release_are_refused(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            document = _document()
            path = _materialize(root, document)
            with self.assertRaises(ValueError):
                release(path, fixture_root=root, owner_fixture_release=True)
            self.assertEqual(SealStore(path, fixture_root=root).load()["state"], "sealed")
            edited = json.loads(path.read_text())
            edited["families"][0]["judge_row_ids"] = ["forged"]
            path.write_text(canonical(edited) + "\n")
            with self.assertRaises(ValueError):
                SealStore(path, fixture_root=root).load()


class EvaluatorAttestationTest(unittest.TestCase):
    def _spent(self, root):
        document = _document()
        path = _materialize(root, document)
        picks = _picks(document)
        freeze(path, picks=picks, encoding_sha256=ENCODING)
        release(path, fixture_root=root, owner_fixture_release=True)
        return SealStore(path, fixture_root=root), picks["fam"]

    def _result(self, request, **overrides):
        result = {
            "canonicalizer_sha256": request["canonicalizer_sha256"],
            "closure_verified": True,
            "context_id": request["context_id"],
            "cpu_seconds": 0.0,
            "evaluator_sha256": request["evaluator_sha256"],
            "outcome_class": "qualified_complete",
            "parity_max_normalized_error": 0.0,
            "parity_verified": True,
            "program_sha256": request["program_sha256"],
            "protocol": request["protocol"],
            "q_grid": 1_250_000,
            "receipt_sha256": RECEIPT,
            "rust_attested": True,
            "wall_seconds": 0.0,
        }
        result.update(overrides)
        return result

    def test_t12_wrapper_passes_the_pinned_grid_and_requires_rust(self):
        import tempfile

        seen = {}

        def pinned(request):
            seen["request"] = request
            return self._result(request)

        with tempfile.TemporaryDirectory() as root:
            store, program = self._spent(root)
            evaluator = JudgeEvaluator(
                pinned,
                evaluator_sha256=EVALUATOR,
                canonicalizer_sha256=CANONICALIZER,
                protocol=PROTOCOL,
            )
            attested = evaluator.score(
                store, family_id="fam", context_id="judge-fam", program_sha256=program
            )
            self.assertEqual(attested["q_grid"], 1_250_000)
            self.assertNotIn("q", attested)
            self.assertNotIn("target", seen["request"])
            self.assertNotIn("targets", seen["request"])
            self.assertNotIn("features", seen["request"])
            self.assertEqual(seen["request"]["lane"], "B")
            self.assertEqual(seen["request"]["judge_row_ids"], store.judge_rows("fam"))
            with self.assertRaises(SealSpent):
                evaluator.score(store, family_id="fam", context_id="judge-fam", program_sha256=program)

    def test_t12_refuses_a_missing_attestation_and_a_numeric_failure(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            store, program = self._spent(root)

            def pinned(request):
                return self._result(request, rust_attested=False)

            evaluator = JudgeEvaluator(
                pinned,
                evaluator_sha256=EVALUATOR,
                canonicalizer_sha256=CANONICALIZER,
                protocol=PROTOCOL,
            )
            with self.assertRaises(ValueError):
                evaluator.score(store, family_id="fam", context_id="judge-fam", program_sha256=program)

            def failed(request):
                return self._result(
                    request,
                    outcome_class="program_failed",
                    q=None,
                    q_grid=None,
                    parity_verified=False,
                    closure_verified=False,
                )

            evaluator.pinned_evaluate = failed
            attested = evaluator.score(
                store, family_id="fam", context_id="judge-fam", program_sha256=program
            )
            self.assertIsNone(attested.get("q_grid"))
            self.assertIsNone(attested.get("q"))

        with tempfile.TemporaryDirectory() as root:
            store, program = self._spent(root)

            def leaked(request):
                return self._result(request, outcome_class="timeout", q_grid=1, q=1e-6)

            evaluator = JudgeEvaluator(
                leaked,
                evaluator_sha256=EVALUATOR,
                canonicalizer_sha256=CANONICALIZER,
                protocol=PROTOCOL,
            )
            with self.assertRaises(ValueError):
                evaluator.score(store, family_id="fam", context_id="judge-fam", program_sha256=program)
            self.assertEqual(store.load()["scored"], [])

    def test_t15_overlap_is_not_scored_and_is_not_called_unseen(self):
        import tempfile

        from rlfw.judge.crosswalk import unseen_word

        with tempfile.TemporaryDirectory() as root:
            document = _document(overlap=True)
            self.assertEqual(document["families"][0]["overlap_status"], "quarantine")
            with self.assertRaises(ValueError):
                unseen_word("quarantine")
            with self.assertRaises(ValueError):
                unseen_word("overlap_unknown")
            self.assertEqual(unseen_word("no_overlap"), "unseen")
            path = _materialize(root, document)
            freeze(path, picks=_picks(document), encoding_sha256=ENCODING)
            release(path, fixture_root=root, owner_fixture_release=True)
            evaluator = JudgeEvaluator(
                lambda _request: (_ for _ in ()).throw(AssertionError("quarantine was scored")),
                evaluator_sha256=EVALUATOR,
                canonicalizer_sha256=CANONICALIZER,
                protocol=PROTOCOL,
            )
            with self.assertRaises(QuarantineError):
                evaluator.score(
                    SealStore(path, fixture_root=root),
                    family_id="fam",
                    context_id="judge-fam",
                    program_sha256=_picks(document)["fam"],
                )


class ClaimAndCurveTest(unittest.TestCase):
    def test_winners_curse_fixture_falls_as_b_grows(self):
        contexts = []
        for lineage in range(3):
            disclosures = _disclosures(f"curve-{lineage}")
            contexts.append(
                {
                    "disclosures": disclosures,
                    "judge_q_by_program": {
                        item["program_sha256"]: 1 - item["inner_q_grid"] for item in disclosures
                    },
                    "lineage_id": f"L{lineage}",
                }
            )
        curve = g_curve(contexts, draws=30, seed=1)
        self.assertGreater(curve[1]["mean"], curve[8]["mean"])

    def test_siblings_do_not_narrow_the_lineage_interval(self):
        one = cluster_stat({"L1": [1.0], "L2": [3.0]}, draws=40, seed=3)
        many = cluster_stat({"L1": [1.0, 1.0, 1.0], "L2": [3.0, 3.0]}, draws=40, seed=3)
        self.assertEqual(one["ci95"], many["ci95"])
        self.assertEqual(one["lineages"], many["lineages"])
        self.assertGreater(many["units"], one["units"])

    def test_g4_claim_text_requires_every_gate_on_a_fixture(self):
        units = [
            {"censored": False, "judge_q": 1.0, "lineage_id": f"L{index}"} for index in range(12)
        ]
        report = g_judge(units, draws=20, seed=1)
        self.assertGreater(report["ci95"][0], 0.0)
        allowed = judge_claim(
            report,
            n_families=200,
            n_contexts=400,
            canaries_a2_a4=True,
            overlap_status="no_overlap",
            sigma_source="fixture",
            observed_sigma=1.0,
            registered_delta=0.2,
        )
        self.assertTrue(allowed["allowed"])
        self.assertEqual(allowed["claim"], "Policy-training-unseen-family lift")
        self.assertEqual(allowed["sigma_source"], "fixture")
        self.assertEqual(allowed["multiplicity"], 1)
        blocked = judge_claim(
            report,
            n_families=200,
            n_contexts=400,
            canaries_a2_a4=True,
            overlap_status="no_overlap",
            sigma_source="unmeasured",
            observed_sigma=None,
            registered_delta=0.2,
        )
        self.assertFalse(blocked["allowed"])
        self.assertIsNone(blocked["claim"])
        self.assertIn("observed noise unmeasured", blocked["reasons"])
        with self.assertRaises(ValueError):
            judge_claim(
                report,
                n_families=200,
                n_contexts=400,
                canaries_a2_a4=True,
                overlap_status="no_overlap",
                sigma_source="measured",
                observed_sigma=1.0,
                registered_delta=0.2,
            )
        underpowered = judge_claim(
            report,
            n_families=200,
            n_contexts=400,
            canaries_a2_a4=True,
            overlap_status="no_overlap",
            sigma_source="fixture",
            observed_sigma=1.0,
            registered_delta=0.1,
        )
        self.assertFalse(underpowered["allowed"])
        self.assertGreater(underpowered["power_required_families"], 200)
        censored = g_judge([{"censored": True, "judge_q": None, "lineage_id": "L0"}], draws=10, seed=1)
        refused = judge_claim(
            censored,
            n_families=200,
            n_contexts=400,
            canaries_a2_a4=True,
            overlap_status="no_overlap",
            sigma_source="fixture",
            observed_sigma=1.0,
            registered_delta=0.2,
        )
        self.assertIn("censoring can reverse the sign", refused["reasons"])
        self.assertIsNone(refused["claim"])


if __name__ == "__main__":
    unittest.main()

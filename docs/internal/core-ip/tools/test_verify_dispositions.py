import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from verify_dispositions import EVIDENCE_SCHEMA, SCHEMA, verify
from verify_findings import ROOT

PRE = "1" * 40
FIXED = "2" * 40
TARGET = "cargo:jope-parallel:gp_contracts"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    """Disposable copy of the real register with synthetic evidence only."""

    def __init__(self, directory):
        self.root = Path(directory)
        shutil.copytree(ROOT / "sources", self.root / "sources")
        shutil.copy(ROOT / "register.json", self.root / "register.json")
        self.register_sha256 = digest(self.root / "register.json")

    def evidence(self, finding, name, body):
        path = self.root / "evidence" / finding / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, sort_keys=True))
        return {"path": f"evidence/{finding}/{name}", "sha256": digest(path)}

    def body(self, finding, stage, kind, **extra):
        return {"schema": EVIDENCE_SCHEMA, "finding": finding, "stage": stage,
                "source_commit": FIXED, "evidence_kind": kind, **extra}

    def defect(self, finding="S03-G03", closed=True):
        """A complete confirmed-defect closure for a non-live package."""
        test = "fitted_predictor_roundtrip"
        source = self.evidence(finding, "source.json", self.body(
            finding, "source_implementation", "source-trace", paths=["crates/jope-engine/src/gp.rs"]))
        fixture = self.evidence(finding, "fixture.json", self.body(
            finding, "fixture_proof", "synthetic-fixture", pre_fix_commit=PRE,
            failed_tests=[f"{TARGET}::{test}_pre"],
            passed_tests=[f"{TARGET}::{test}", f"{TARGET}::{test}_negative"]))
        ci = self.evidence(finding, "ci.json", self.body(
            finding, "hosted_ci", "hosted-ci", run_id=1, attempt=1, head_sha=FIXED,
            conclusion="success"))
        return {
            "finding": finding, "sequence": 1, "disposition": "confirmed-defect",
            "scope": "regression/shared receiving boundary", "closed": closed,
            "decided_by": {"role": "independent-reviewer", "identity": "fixture"},
            "attestations": [
                {"stage": "source_implementation", "source_commit": FIXED, "evidence": [source]},
                {"stage": "fixture_proof", "source_commit": FIXED, "evidence": [fixture], "tests": [
                    {"target": TARGET, "name": f"{test}_pre", "role": "pre-fix-failure",
                     "pre_fix_commit": PRE},
                    {"target": TARGET, "name": f"{test}_negative", "role": "negative"},
                    {"target": TARGET, "name": test, "role": "positive"}]},
                {"stage": "hosted_ci", "source_commit": FIXED, "evidence": [ci]},
            ],
        }

    def write(self, records, register_sha256=None):
        (self.root / "dispositions.json").write_text(json.dumps({
            "schema": SCHEMA, "register_sha256": register_sha256 or self.register_sha256,
            "records": records}))

    def verify(self):
        return verify(self.root)


def stage(record, name):
    return next(a for a in record["attestations"] if a["stage"] == name)


class DispositionTests(unittest.TestCase):
    def test_committed_dispositions_verify(self):
        result = verify()
        self.assertEqual(result["entries"], 195)
        self.assertEqual(result["open"] + result["closed"], 195)

    def test_complete_defect_closure_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.write([fixture.defect()])
            result = fixture.verify()
            self.assertEqual((result["closed"], result["open"]), (1, 194))

    def test_open_honest_dispositions_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.write([
                {"finding": "S08-G09", "sequence": 1, "disposition": "historical-unknown",
                 "scope": "historical Elo matches", "closed": False,
                 "reason": "original matches not retained",
                 "decided_by": {"role": "implementation-owner", "identity": "fixture"}},
                {"finding": "S04-G10", "sequence": 1, "disposition": "deferred-non-regression",
                 "scope": "classification subcase", "closed": False,
                 "deferred_subcases": ["binary", "multiclass"],
                 "decided_by": {"role": "implementation-owner", "identity": "fixture"}},
            ])
            self.assertEqual(fixture.verify()["closed"], 0)

    def test_mutations_are_rejected(self):
        def pop_stage(name):
            return lambda f, r: r["attestations"].remove(stage(r, name))

        def edit_evidence(name, **changes):
            def mutate(fixture, record):
                reference = stage(record, name)["evidence"][0]
                path = fixture.root / reference["path"]
                body = json.loads(path.read_text())
                body.update(changes)
                path.write_text(json.dumps(body))
                reference["sha256"] = digest(path)
            return mutate

        def symlink_evidence(fixture, record):
            reference = stage(record, "hosted_ci")["evidence"][0]
            path = fixture.root / reference["path"]
            target = path.with_name("real.json")
            path.rename(target)
            os.symlink(target.name, path)

        def live_package(fixture, record):
            record.update(fixture.defect("S08-G03"))

        mutations = {
            "unknown finding": lambda f, r: r.update(finding="S99-G1"),
            "unsupported disposition": lambda f, r: r.update(disposition="closed"),
            "digest mismatch": lambda f, r: stage(r, "hosted_ci")["evidence"][0].update(sha256="0" * 64),
            "path traversal": lambda f, r: stage(r, "hosted_ci")["evidence"][0].update(
                path="evidence/S03-G03/../../register.json"),
            "other finding directory": lambda f, r: stage(r, "hosted_ci")["evidence"][0].update(
                path="evidence/S03-G04/ci.json"),
            "symlinked evidence": symlink_evidence,
            "evidence names another finding": edit_evidence("hosted_ci", finding="S03-G04"),
            "evidence names another stage": edit_evidence("hosted_ci", stage="fixture_proof"),
            "evidence names another commit": edit_evidence("hosted_ci", source_commit=PRE),
            "hosted CI failed": edit_evidence("hosted_ci", conclusion="failure"),
            "hosted CI other head": edit_evidence("hosted_ci", head_sha=PRE),
            "synthetic claims hosted CI": edit_evidence("hosted_ci", evidence_kind="synthetic-fixture"),
            "defect without hosted CI": pop_stage("hosted_ci"),
            "defect without source": pop_stage("source_implementation"),
            "positive not observed": edit_evidence("fixture_proof", passed_tests=[]),
            "positive also failed": edit_evidence(
                "fixture_proof", failed_tests=[f"{TARGET}::fitted_predictor_roundtrip_pre",
                                               f"{TARGET}::fitted_predictor_roundtrip"]),
            "pre-fix not observed": edit_evidence("fixture_proof", failed_tests=[]),
            "pre-fix commit equals fix": lambda f, r: stage(r, "fixture_proof")["tests"][0].update(
                pre_fix_commit=FIXED),
            "no negative test": lambda f, r: stage(r, "fixture_proof")["tests"].pop(1),
            "duplicate stage": lambda f, r: r["attestations"].append(copy.deepcopy(stage(r, "hosted_ci"))),
            "historical unknown closed": lambda f, r: r.update(disposition="historical-unknown",
                                                               reason="lost"),
            "missing evidence closed": lambda f, r: r.update(disposition="missing-execution-evidence",
                                                             reason="none"),
            "deferral of regression": lambda f, r: r.update(
                disposition="deferred-non-regression", closed=False, deferred_subcases=["regression"]),
            "live package without live stage": live_package,
            "no decision owner": lambda f, r: r.update(decided_by={}),
            "empty scope": lambda f, r: r.update(scope=" "),
        }
        # Each mutation must fail for its own reason, not an incidental earlier check.
        reasons = {
            "unknown finding": "unknown finding", "unsupported disposition": ": disposition$",
            "digest mismatch": "digest mismatch", "path traversal": "path traversal",
            "other finding directory": "outside evidence/S03-G03/",
            "symlinked evidence": "evidence symlink",
            "evidence names another finding": "another finding",
            "evidence names another stage": "another stage",
            "evidence names another commit": "another commit",
            "hosted CI failed": "hosted CI evidence binding",
            "hosted CI other head": "hosted CI evidence binding",
            "synthetic claims hosted CI": "cannot back hosted_ci",
            "defect without hosted CI": "needs source, fixture and hosted CI",
            "defect without source": "needs source, fixture and hosted CI",
            "positive not observed": "not observed passing",
            "positive also failed": "roundtrip not observed passing",
            "pre-fix not observed": "pre-fix failure not observed",
            "pre-fix commit equals fix": "distinct pre-fix commit",
            "no negative test": "pre-fix failure, negative and positive",
            "duplicate stage": "duplicate hosted_ci",
            "historical unknown closed": "historical-unknown cannot close",
            "missing evidence closed": "missing-execution-evidence cannot close",
            "deferral of regression": "deferral names only",
            "live package without live stage": "W8 requires live qualification",
            "no decision owner": "decided_by", "empty scope": ": scope$",
        }
        self.assertEqual(set(reasons), set(mutations))
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(directory)
                record = fixture.defect()
                mutate(fixture, record)
                fixture.write([record])
                with self.assertRaisesRegex(ValueError, reasons[label]):
                    fixture.verify()

    def test_live_and_promotion_stages_need_their_own_evidence(self):
        cases = {
            "synthetic live": ("live_qualification", "synthetic-fixture", {},
                               "cannot back live_qualification"),
            "promotion without live": ("promotion", "decision",
                                       {"decision_owner_role": "scientific-decision-owner"},
                                       "promotion without live qualification"),
            "promotion by implementer": ("promotion", "decision",
                                         {"decision_owner_role": "implementation-owner"},
                                         "promotion owner"),
        }
        for label, (name, kind, extra, reason) in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(directory)
                record = fixture.defect()
                reference = fixture.evidence("S03-G03", f"{name}.json",
                                             fixture.body("S03-G03", name, kind, **extra))
                record["attestations"].append(
                    {"stage": name, "source_commit": FIXED, "evidence": [reference]})
                fixture.write([record])
                with self.assertRaisesRegex(ValueError, reason):
                    fixture.verify()

    def test_register_binding_and_append_only_history(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.write([fixture.defect()], register_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "bind another register"):
                fixture.verify()
            first = fixture.defect(closed=False)
            later = fixture.defect()
            later["sequence"] = 1
            fixture.write([first, later])
            with self.assertRaisesRegex(ValueError, "append-only"):
                fixture.verify()
            later["sequence"] = 2
            fixture.write([first, later])
            self.assertEqual(fixture.verify()["closed"], 1)


if __name__ == "__main__":
    unittest.main()

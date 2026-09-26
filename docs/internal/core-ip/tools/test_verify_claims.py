import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from verify_claims import CLAIMS, SCHEMA, verify

COMMIT = "a" * 40


class Fixture:
    """A repository-local evidence graph: root -> child -> grandchild, plus an unrelated node."""

    def __init__(self, directory):
        self.root = Path(directory)
        self.path = self.root / "docs/internal/core-ip/claims/claims.json"
        self.path.parent.mkdir(parents=True)
        self.nodes = {}
        for name, kind, parents in [("E-root", "synthetic-fixture", []),
                                    ("E-child", "hosted-ci", ["E-root"]),
                                    ("E-grandchild", "live-host", ["E-child"]),
                                    ("E-other", "measurement", [])]:
            file = self.root / "evidence" / f"{name}.json"
            file.parent.mkdir(exist_ok=True)
            file.write_text(json.dumps({"node": name}))
            self.nodes[name] = {"kind": kind, "location": "repository", "path": f"evidence/{name}.json",
                                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "source_commit": COMMIT,
                                "subject": "subject", "scope": "scope", "depends_on": parents}

    def claim(self, cid, stage, evidence, status="active"):
        return {"id": cid, "statement": "s", "subject": "subject", "metric": "m", "units": "u",
                "denominator": "d", "comparison": None, "baseline": None, "scope": "scope",
                "source_commit": COMMIT, "observed_utc": "2026-09-26T00:00:00Z", "uncertainty": "exact",
                "limitations": ["fixture"], "stage": stage, "evidence": evidence, "status": status}

    def document(self):
        return {"schema": SCHEMA, "evidence": copy.deepcopy(self.nodes), "revocations": [],
                "claims": [self.claim("C-live", "live", ["E-grandchild"]),
                           self.claim("C-measure", "measurement", ["E-other"])]}

    def verify(self, document):
        self.path.write_text(json.dumps(document))
        return verify(self.path, root=self.root)


class ClaimTests(unittest.TestCase):
    def test_committed_claims_verify(self):
        result = verify(CLAIMS)
        self.assertGreaterEqual(result["claims"].get("active", 0), 1)

    def test_valid_graph_and_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            result = fixture.verify(fixture.document())
            self.assertEqual((result["verified_bytes"], result["revoked_evidence"]), (4, []))

    def test_revocation_invalidates_exactly_the_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            document = fixture.document()
            document["revocations"] = [{"node": "E-root", "issuer": "reviewer", "cause": "leak",
                                        "effective_utc": "2026-09-26T01:00:00Z", "sequence": 1}]
            with self.assertRaisesRegex(ValueError, "C-live: cites revoked evidence"):
                fixture.verify(document)
            document["claims"][0]["status"] = "revoked"
            result = fixture.verify(document)
            self.assertEqual(result["revoked_evidence"], ["E-child", "E-grandchild", "E-root"])
            self.assertEqual(result["claims"], {"revoked": 1, "active": 1})  # unrelated claim stays usable

    def test_mutations_are_rejected(self):
        def claim(field, value, index=0):
            return lambda d: d["claims"][index].__setitem__(field, value)

        mutations = {
            "altered scope": (claim("scope", "broader scope"), "subject, scope or source differs"),
            "altered subject": (claim("subject", "other model"), "subject, scope or source differs"),
            "other source": (claim("source_commit", "b" * 40), "subject, scope or source differs"),
            "stage without evidence kind": (claim("stage", "promotion"), "no promotion evidence"),
            "synthetic backs live": (claim("evidence", ["E-root", "E-grandchild"]),
                                     "synthetic evidence cannot support a live claim"),
            "unknown evidence": (claim("evidence", ["E-missing"]), "evidence references"),
            "missing denominator": (claim("denominator", ""), "incomplete binding"),
            "no limitations": (claim("limitations", []), "limitations"),
            "unknown status": (claim("status", "passed"), "status"),
            "extra field": (lambda d: d["claims"][0].__setitem__("verified", True), "claim fields"),
            "duplicate claim": (lambda d: d["claims"].append(copy.deepcopy(d["claims"][0])), "duplicate claim"),
            "digest mismatch": (lambda d: d["evidence"]["E-other"].__setitem__("sha256", "0" * 64),
                                "evidence digest mismatch"),
            "path traversal": (lambda d: d["evidence"]["E-other"].__setitem__("path", "../evidence/E-other.json"),
                               "path traversal"),
            "cycle": (lambda d: d["evidence"]["E-root"].__setitem__("depends_on", ["E-grandchild"]),
                      "dependency cycle"),
            "unknown parent": (lambda d: d["evidence"]["E-root"].__setitem__("depends_on", ["E-ghost"]),
                               "unknown parent"),
            "unknown kind": (lambda d: d["evidence"]["E-other"].__setitem__("kind", "vibes"), ": kind"),
            "revoke unknown": (lambda d: d["revocations"].append(
                {"node": "E-ghost", "issuer": "i", "cause": "c", "effective_utc": "t", "sequence": 1}),
                "revocation of unknown"),
            "revocation not append-only": (lambda d: d["revocations"].extend(
                [{"node": "E-other", "issuer": "i", "cause": "c", "effective_utc": "t", "sequence": 2},
                 {"node": "E-other", "issuer": "i", "cause": "c", "effective_utc": "t", "sequence": 1}]),
                "append-only"),
        }
        for label, (mutate, reason) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(directory)
                document = fixture.document()
                mutate(document)
                with self.assertRaisesRegex(ValueError, reason):
                    fixture.verify(document)

    def test_missing_custody_is_reported_and_can_be_required(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            document = fixture.document()
            document["evidence"]["E-other"].update(location="custody", path="/nonexistent/custody.json")
            self.assertEqual(fixture.verify(document)["custody_unverified"], 1)
            fixture.path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "custody evidence unavailable"):
                verify(fixture.path, root=fixture.root, require_custody=True)


if __name__ == "__main__":
    unittest.main()

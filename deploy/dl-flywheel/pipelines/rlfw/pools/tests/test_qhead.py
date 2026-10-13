"""CPU fixtures for the J9 q-head adapter. A stub policy replaces torch; no data, no network."""

import json
import math
import unittest

from rlfw.pools.qhead import NAME, SCORES_SCHEMA, read_vectors, read_views, score_views

CTX = ["a" * 64, "b" * 64]


def program(i):
    return {"program_sha256": f"{i:064x}", "tokens": [1, 3 + i, 2]}


def view_line(cid, n, **extra):
    view = {
        "schema": "jope.j9-pool-view/v1",
        "context_id": cid,
        "parent_id": "p",
        "pool_sha256": "c" * 64,
        "programs": [program(i) for i in range(n)],
    }
    view.update(extra)
    return json.dumps(view)


class StubPolicy:
    """q depends on the normalized context and the token row, so batching mistakes show."""

    def __init__(self, bad=None):
        self.calls = []
        self.bad = bad

    def normalize(self, vector):
        return [2.0 * v for v in vector]

    def score_heads(self, contexts, token_rows):
        self.calls.append(len(token_rows))
        out = []
        for context, row in zip(contexts, token_rows):
            q = sum(context) + 0.01 * sum(row)
            out.append({"q": self.bad if self.bad is not None else q, "p_complete": 1.0})
        return out


class QheadAdapter(unittest.TestCase):
    def setUp(self):
        self.views = read_views("\n".join([view_line(CTX[0], 5), view_line(CTX[1], 3)]))
        self.vectors = read_vectors(
            "\n".join(json.dumps({"id": c, "vector": [i + 1.0, 0.5]}) for i, c in enumerate(CTX))
        )

    def test_scores_every_program_once_in_view_order(self):
        scores = score_views(self.views, self.vectors, StubPolicy())
        self.assertEqual(len(scores), 8)
        keys = [(s["context_id"], s["program_sha256"]) for s in scores]
        self.assertEqual(len(set(keys)), 8)
        self.assertEqual(keys[0], (CTX[0], program(0)["program_sha256"]))
        # context a: normalized sum = 2*(1+0.5) = 3; program 0 tokens sum 1+3+2 = 6
        self.assertAlmostEqual(scores[0]["score"], 3.06)
        self.assertTrue(all(math.isfinite(s["score"]) for s in scores))

    def test_batching_does_not_change_scores(self):
        whole = score_views(self.views, self.vectors, StubPolicy())
        stub = StubPolicy()
        chunked = score_views(self.views, self.vectors, stub, batch_rows=2)
        self.assertEqual(whole, chunked)
        self.assertEqual(stub.calls, [2, 2, 1, 2, 1])

    def test_views_with_outcomes_are_refused(self):
        for field in ("q", "status", "incumbent_q"):
            leaked = json.loads(view_line(CTX[0], 2))
            leaked["programs"][0][field] = 0.5
            with self.assertRaisesRegex(ValueError, "outcome field"):
                read_views(json.dumps(leaked))
        with self.assertRaisesRegex(ValueError, "outcome field"):
            read_views(view_line(CTX[0], 2, incumbent_q=0.9))

    def test_missing_vector_tokens_or_schema_are_refused(self):
        with self.assertRaisesRegex(ValueError, "no context vector"):
            score_views(self.views, {CTX[0]: [1.0, 0.5]}, StubPolicy())
        bare = json.loads(view_line(CTX[0], 2))
        bare["programs"][1]["tokens"] = None
        with self.assertRaisesRegex(ValueError, "tokens"):
            read_views(json.dumps(bare))
        with self.assertRaisesRegex(ValueError, "schema"):
            read_views(view_line(CTX[0], 2, schema="other"))
        with self.assertRaisesRegex(ValueError, "no pool views"):
            read_views("\n")

    def test_non_finite_head_is_refused(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            score_views(self.views, self.vectors, StubPolicy(bad=float("nan")))

    def test_document_identity_constants(self):
        self.assertEqual(SCORES_SCHEMA, "jope.j9-external-scores/v1")
        self.assertEqual(NAME, "policylm_v1_qhead")


if __name__ == "__main__":
    unittest.main()

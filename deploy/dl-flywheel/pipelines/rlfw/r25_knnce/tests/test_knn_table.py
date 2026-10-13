"""Unit tests for the R25-kNN-CE neighbour table (ENG:watcher ASSIGN 2026-10-13T07:42:55Z).

The required assertion: neighbours are disjoint from held-out and validation contexts, and never share the anchor's
parent lineage. The fixture makes every forbidden context the *most* similar candidate, so a builder that forgot an
exclusion would select it. Run through run_r25_offline.py (needs torch only).
"""
import unittest

import torch

from rlfw.r25_knnce.knn_table import KnnTableError, build_knn_table, table_digest, verify_knn_table


def fixture():
    # Anchors a0..a7 over lineages L0..L3 (two per lineage). Each anchor's own-lineage sibling, a held-out context
    # and a validation context are placed nearer to it than any eligible neighbour.
    anchors, vectors, lineage = [], {}, {}
    for i in range(8):
        cid = f'a{i}'
        anchors.append(cid)
        lineage[cid] = f'L{i // 2}'
        base = [0.0] * 8
        base[i] = 1.0
        base[(i + 2) % 8] = 0.3  # eligible neighbours share this weaker component
        vectors[cid] = base
    for i in range(0, 8, 2):  # make each pair of lineage siblings nearly identical
        vectors[f'a{i + 1}'] = [v + (0.99 if j == i else 0.0) for j, v in enumerate(vectors[f'a{i + 1}'])]
    excluded = {}
    for i in range(8):
        excluded[f'held{i}'] = list(vectors[f'a{i}'])      # identical to anchor i (held-out lineage)
        excluded[f'val{i}'] = [v * 1.0001 for v in vectors[f'a{i}']]  # validation, nearly identical
    return anchors, vectors, lineage, excluded


class KnnTableTest(unittest.TestCase):
    def test_neighbours_exclude_heldout_validation_and_same_lineage(self):
        anchors, vectors, lineage, excluded = fixture()
        all_vectors = {**vectors, **excluded}
        for e in excluded:
            lineage[e] = 'LX'
        table = build_knn_table(anchors, all_vectors, lineage, set(excluded), k=3, torch=torch)
        self.assertEqual(set(table), set(anchors))
        for a, nbrs in table.items():
            self.assertEqual(len(nbrs), 3)
            for n in nbrs:
                self.assertIn(n, anchors)                      # only L6 train-roster contexts
                self.assertNotIn(n, excluded)                  # never held-out / validation / outer test
                self.assertNotEqual(lineage[n], lineage[a])    # never the anchor's own parent lineage
                self.assertNotEqual(n, a)

    def test_excluded_context_cannot_be_an_anchor(self):
        anchors, vectors, lineage, excluded = fixture()
        with self.assertRaises(KnnTableError):
            build_knn_table(anchors, vectors, lineage, {anchors[0]}, k=2, torch=torch)

    def test_verify_rejects_each_forbidden_neighbour(self):
        anchors, vectors, lineage, _ = fixture()
        good = build_knn_table(anchors, vectors, lineage, set(), k=2, torch=torch)
        lineage['held0'] = 'LX'
        for bad_n, why in (('held0', 'excluded'), ('a1', 'same lineage as a0'), ('zz', 'outside roster')):
            table = {a: list(n) for a, n in good.items()}
            table['a0'] = [bad_n, table['a0'][1]]
            lin = {**lineage, 'zz': 'LZ'}
            with self.assertRaises(KnnTableError, msg=why):
                verify_knn_table(table, anchors, lin, {'held0'}, k=2)

    def test_deterministic_and_tie_break_by_id(self):
        ids = ['c', 'b', 'a', 'd']
        vec = {'a': [1.0, 0.0], 'b': [1.0, 0.0], 'c': [1.0, 0.0], 'd': [0.0, 1.0]}
        lin = {'a': 'L1', 'b': 'L2', 'c': 'L3', 'd': 'L4'}
        t1 = build_knn_table(ids, vec, lin, set(), k=2, torch=torch)
        t2 = build_knn_table(list(reversed(ids)), vec, lin, set(), k=2, torch=torch)
        self.assertEqual(t1, t2)
        self.assertEqual(table_digest(t1), table_digest(t2))
        self.assertEqual(t1['a'], ['b', 'c'])  # equal similarity: ordered by id
        self.assertEqual(t1['d'], ['a', 'b'])  # all three tie at cosine 0: smallest ids first

    def test_candidates_restrict_neighbours_but_every_anchor_gets_a_row(self):
        anchors, vectors, lineage, _ = fixture()
        cands = [a for a in anchors if a not in ('a2', 'a3')]  # e.g. no trie-leaf teacher
        table = build_knn_table(anchors, vectors, lineage, set(), k=2, torch=torch, candidate_ids=cands)
        self.assertEqual(set(table), set(anchors))
        self.assertTrue(all(n in cands for nbrs in table.values() for n in nbrs))
        bad = {a: list(n) for a, n in table.items()}
        bad['a0'] = ['a2', bad['a0'][1]]
        with self.assertRaises(KnnTableError):
            verify_knn_table(bad, anchors, lineage, set(), 2, cands)
        with self.assertRaises(KnnTableError):  # candidates outside the roster are refused
            build_knn_table(anchors, {**vectors, 'zz': [1.0] * 8}, {**lineage, 'zz': 'LZ'}, set(), k=2,
                            torch=torch, candidate_ids=cands + ['zz'])

    def test_too_few_eligible_neighbours_refused(self):
        vec = {'a': [1.0, 0.0], 'b': [0.0, 1.0]}
        lin = {'a': 'L1', 'b': 'L1'}
        with self.assertRaises(KnnTableError):
            build_knn_table(['a', 'b'], vec, lin, set(), k=1, torch=torch)


if __name__ == '__main__':
    unittest.main()

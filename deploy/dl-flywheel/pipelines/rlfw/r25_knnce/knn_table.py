"""R25-kNN-CE neighbour table (lane harness; not part of any PR tree).

build_knn_table picks, for every anchor context (the L6 train roster), its k nearest *eligible* neighbours by cosine
similarity of the frozen-normalised encodings. A neighbour must be a candidate context (roster contexts whose frozen
native HWM teacher is a trie leaf; a subset of the anchors), must not be in `excluded` (held-out lineages, validation,
outer test: the caller passes every non-roster context), and must not share the anchor's parent lineage. Ties break
on the neighbour id, so the table is a pure function of its inputs.

Every exclusion is asserted again on the finished table (verify_knn_table), and the receipt pins its digest.
"""
import hashlib
import json


class KnnTableError(ValueError):
    pass


def table_digest(table):
    return hashlib.sha256(json.dumps(table, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_knn_table(anchor_ids, vectors, lineage, excluded, k, torch, candidate_ids=None, block=512):
    """Return {anchor id: [k neighbour ids, most similar first]}.

    anchor_ids: the L6 train contexts; every one gets a row.
    candidate_ids: the contexts that may be neighbours (default: all anchors); must be a subset of anchor_ids.
    vectors: {id: sequence of floats} for every anchor id (frozen-normalised encodings).
    lineage: {id: parent lineage id} for every anchor id.
    excluded: ids that may never be neighbours or anchors (held-out lineages, validation, outer test).
    """
    ids = sorted(anchor_ids)
    cands = sorted(ids if candidate_ids is None else candidate_ids)
    if len(set(ids)) != len(ids) or len(set(cands)) != len(cands):
        raise KnnTableError('duplicate ids')
    if k < 1:
        raise KnnTableError('k must be positive')
    if not set(cands) <= set(ids):
        raise KnnTableError('candidates must come from the anchor roster')
    bad = set(ids) & set(excluded)
    if bad:
        raise KnnTableError(f'{len(bad)} anchor contexts are excluded contexts')

    def unit(rows):
        x = torch.tensor([list(map(float, vectors[c])) for c in rows], dtype=torch.float64)
        norms = x.norm(dim=1, keepdim=True)
        if not bool((norms > 0).all()):
            raise KnnTableError('zero encoding vector')
        return x / norms

    xa, xc = unit(ids), unit(cands)
    lin_code = {v: i for i, v in enumerate(sorted({lineage[c] for c in ids}))}
    code_a = torch.tensor([lin_code[lineage[c]] for c in ids])
    code_c = torch.tensor([lin_code[lineage[c]] for c in cands])
    table = {}
    for start in range(0, len(ids), block):
        sim = xa[start:start + block] @ xc.T
        same = code_a[start:start + block].unsqueeze(1) == code_c.unsqueeze(0)
        sim = sim.masked_fill(same, float('-inf'))  # own lineage (and so the anchor itself) is never a neighbour
        eligible = (sim != float('-inf')).sum(dim=1)
        kk = min(k, sim.shape[1])
        kth = sim.topk(kk, dim=1).values[:, -1]
        for r in range(sim.shape[0]):
            if int(eligible[r]) < k:
                raise KnnTableError(f'anchor {ids[start + r]} has {int(eligible[r])} eligible neighbours < k={k}')
            # Exact: every candidate at least as similar as the k-th best (ties included), then (-sim, id) order.
            idx = (sim[r] >= kth[r]).nonzero().flatten().tolist()
            cand = sorted(((float(sim[r, j]), cands[j]) for j in idx), key=lambda t: (-t[0], t[1]))
            table[ids[start + r]] = [c for _, c in cand[:k]]
    verify_knn_table(table, ids, lineage, excluded, k, cands)
    return table


def verify_knn_table(table, anchor_ids, lineage, excluded, k, candidate_ids=None):
    """Assert every pre-registered exclusion on a finished table; raise KnnTableError on any violation."""
    anchors = set(anchor_ids)
    cands = anchors if candidate_ids is None else set(candidate_ids)
    excluded = set(excluded)
    if set(table) != anchors:
        raise KnnTableError('table anchors differ from the L6 train roster')
    for a, nbrs in table.items():
        if len(nbrs) != k or len(set(nbrs)) != k:
            raise KnnTableError(f'anchor {a}: expected {k} distinct neighbours')
        for n in nbrs:
            if n == a:
                raise KnnTableError(f'anchor {a} is its own neighbour')
            if n not in anchors:
                raise KnnTableError(f'neighbour {n} is outside the L6 train roster')
            if n not in cands:
                raise KnnTableError(f'neighbour {n} is not a candidate (no trie-leaf teacher)')
            if n in excluded:
                raise KnnTableError(f'neighbour {n} is an excluded (held-out/validation/outer-test) context')
            if lineage[n] == lineage[a]:
                raise KnnTableError(f'neighbour {n} shares the parent lineage of anchor {a}')
    return True

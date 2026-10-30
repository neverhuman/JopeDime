from collections import Counter

from run import family_balanced_context_batch, family_folds


def test_family_folds_are_deterministic_disjoint_and_complete():
    rows = [
        {"context_sha256": f"ctx-{family}-{i}", "family_id": f"family-{family}"}
        for family in range(11)
        for i in range(family % 3 + 1)
    ]
    first = family_folds(rows, count=5, seed=12)
    assert first == family_folds(rows, count=5, seed=12)
    assert set.union(*first) == {row["family_id"] for row in rows}
    assert sum(map(len, first)) == len(set.union(*first))


def test_family_balanced_batch_equalizes_family_sampling_mass():
    family_by_cid = {"small": "a", **{f"large-{i}": "b" for i in range(40)}}
    sampled = family_balanced_context_batch(
        list(family_by_cid), family_by_cid, size=20_000, seed=77
    )
    counts = Counter(family_by_cid[cid] for cid in sampled)
    assert abs(counts["a"] / len(sampled) - 0.5) < 0.02
    assert abs(counts["b"] / len(sampled) - 0.5) < 0.02

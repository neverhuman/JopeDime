from collections import Counter

import torch

from run import family_balanced_context_batch, family_folds, h46, save_adapter_snapshot


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


def test_adapter_snapshot_round_trips_with_identity_and_parameter_count(tmp_path):
    path = tmp_path / "selected.pt"
    save_adapter_snapshot(path, {"adapter": torch.zeros(29_211)}, "checkpoint-sha")
    saved = torch.load(path, weights_only=True)
    assert saved["base_checkpoint_sha256"] == "checkpoint-sha"
    assert saved["trainable_params"] == 29_211
    assert sum(value.numel() for value in saved["adapter_trainable_state"].values()) == 29_211


def test_h46_helpers_load_under_a_nonconflicting_module_name():
    assert h46.__name__ == "h48_h46_run"
    assert callable(h46.checkpoint_guard)
    assert callable(h46.make_tokens)

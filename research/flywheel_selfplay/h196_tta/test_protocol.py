"""Synthetic integration checks for family inference, stage order and budgets."""

import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import analysis as stat
from engine import Engine
from monitor import violations
from runner import panel, save, save_panel, utility, forbidden_training


def rows(n=205):
    return [dict(context_sha256=f"c{i}", family_id=f"f{i}") for i in range(n)]


def test_fresh_folds_and_jaccard():
    manifest = stat.fold_manifest(rows())
    assert len({json.dumps(v["families"]) for v in manifest.values()}) == 3
    assert all(
        v["coverage"]["a_exact_once"] and v["coverage"]["b_exact_once"]
        for v in manifest.values()
    )
    for seed in stat.SEEDS:
        for outer in range(5):
            fit, b, a = stat.partition(rows(), seed, outer)
            assert list(map(len, (fit, b, a))) == [123, 41, 41]
            assert not {r["family_id"] for r in a} & {r["family_id"] for r in fit + b}
    assert len(stat.jaccards(manifest)) == 3
    assert all(0 < v["diagonal_mean"] < 1 for v in stat.jaccards(manifest).values())


def lock_fixture(tmp_path):
    save_panel(tmp_path / "B-panel.npz", utility=np.zeros((3, 205, 4)))
    cells = {}
    for seed in stat.SEEDS:
        for outer in range(5):
            fit, b, a = stat.partition(rows(), seed, outer)
            cells[f"s{seed}-f{outer}"] = dict(
                seed=seed,
                outer=outer,
                scale=np.zeros((10, 32)).tolist(),
                b_family_means=[0] * 4,
                fit_families=[r["family_id"] for r in fit],
                b_families=[r["family_id"] for r in b],
                a_families=[r["family_id"] for r in a],
                noise_receipt=dict(version_id="v1"),
            )
    lock = dict(
        wave="H196",
        sigma=0.05,
        locked_before_a_and_treatment_val=True,
        fold_manifest_sha256="fold",
        adapter_sha256="adapter",
        cells=cells,
        b_panel_sha256=stat.sha(tmp_path / "B-panel.npz"),
        val_scales={str(s): np.zeros((10, 32)).tolist() for s in stat.SEEDS},
    )
    return lock


def test_complete_durable_lock_and_mutation(tmp_path):
    value = lock_fixture(tmp_path)
    path = tmp_path / "b_lock.json"
    guard = stat.StageGuard("fold", "adapter")
    with pytest.raises(RuntimeError):
        guard.require()
    save(path, value)
    guard.lock(path)
    assert guard.require() == value
    with pytest.raises(RuntimeError):
        guard.lock(path)
    with (tmp_path / "B-panel.npz").open("ab") as f:
        f.write(b"mutation")
    with pytest.raises(RuntimeError, match="panel mutation"):
        guard.require()


@pytest.mark.parametrize(
    "fault", ["missing_cell", "wrong_adapter", "overlap", "val_refit"]
)
def test_lock_fail_closed(tmp_path, fault):
    value = lock_fixture(tmp_path)
    if fault == "missing_cell":
        value["cells"].pop("s196101-f0")
    if fault == "wrong_adapter":
        value["adapter_sha256"] = "different"
    if fault == "overlap":
        value["cells"]["s196101-f0"]["a_families"] = value["cells"]["s196101-f0"][
            "b_families"
        ]
    if fault == "val_refit":
        value["val_scales"]["196101"][0][0] = 1
    path = tmp_path / "b_lock.json"
    save(path, value)
    with pytest.raises(RuntimeError):
        stat.StageGuard("fold", "adapter").lock(path)


def test_median_seed_bootstrap_family_weight_harm_and_exact_minus2():
    rs = [
        dict(context_sha256=f"c{i}", family_id="a" if i < 3 else "b") for i in range(4)
    ]
    values = np.zeros((3, 4, 4))
    values[:, :, 0] = 1
    # Seed gains have median .2, despite unequal contexts/family and a large third seed.
    for si, gain in enumerate((0.1, 0.2, 0.9)):
        values[si, :, 1] = 1 + gain
        values[si, :, 2] = 1
        values[si, :, 3] = [1, 1, 1, -2]
    out = stat.summarize(rs, values)
    p = out["contrasts"]["TTA_NOISE16"]
    assert p["per_seed_gain"] == pytest.approx([0.1, 0.2, 0.9])
    assert p["ci95"] == pytest.approx([0.2, 0.2]) and p["median_gain"] == pytest.approx(
        0.2
    )
    assert p["holm_p_m1"] == 1 / 10001 and p["harm_rate"] == 0
    assert stat.verdict(out)["verdict"] == "ADVANCE"
    assert out["contrasts"]["NOISE1"]["median_gain"] == pytest.approx(-1.5)
    assert out["contrasts"]["NOISE1"]["harm_rate"] == 0.5
    assert np.array_equal(np.asarray(out["exact_minus2_by_seed"])[:, 3], [1, 1, 1])
    assert np.array_equal(np.asarray(out["elo_by_seed"])[:, 0], [1500] * 3)


def test_panel_uses_original_float64_gates_and_paired_first_draw():
    pool = dict(
        fitness=np.array([2 + 1e-10, 2.0, -1.0]), gate=np.array([True, True, False])
    )
    eng = SimpleNamespace(pools={"c": pool})
    costs = {"c": np.array([[0.0, 1.0, 3.0], [4.0, 0.0, 3.0]])}
    result = panel(
        [dict(context_sha256="c")], eng, {"c": 5.0}, costs, reused_perm={"c": 7.0}
    )
    assert result[0].tolist() == [5.0, -2.0, 7.0, -2 - 1e-10]
    assert utility(pool, 2) == -2
    with pytest.raises(RuntimeError):
        forbidden_training()


def test_cached_batch_mismatch_is_fatal():
    engine = object.__new__(Engine)
    engine.native_costs = {"a": np.array([1.0, 2.0], np.float32)}
    engine.verify_costs(["a"], torch.tensor([[1.0, 2.0]]))
    with pytest.raises(RuntimeError, match="identity"):
        engine.verify_costs(["a"], torch.tensor([[1.0, 2.001]]))


def test_resource_boundaries():
    sample = dict(
        cpu=0,
        available=16 * 2**30,
        free=40_000_000_000,
        disk_used_fraction=0.938,
        rss=1000,
        threads=4,
    )
    assert violations(sample) == []
    for key, value in [
        ("available", 16 * 2**30 - 1),
        ("disk_used_fraction", 0.938001),
        ("rss", 6_000_000_000),
        ("threads", 5),
        ("cpu", 85.1),
    ]:
        assert violations(dict(sample, **{key: value}))

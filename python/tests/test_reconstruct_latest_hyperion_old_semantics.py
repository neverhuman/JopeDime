from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

import jope.research

SCRIPT = Path(jope.research.__file__).parent / "reconstruct_latest_hyperion_old_semantics.py"
SPEC = importlib.util.spec_from_file_location("hyperion_old_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_latest_quantization_matches_training_formula() -> None:
    ranked = np.array(
        [[25.0, 100.0], [50.0, 75.0], [75.0, 50.0], [100.0, 25.0]],
        dtype=np.float32,
    )
    actual = MODULE.robust_quantize_ranked(ranked, 1.0, 99.0, 3.0)
    lo, hi = np.percentile(ranked, [1.0, 99.0], axis=0)
    mid = (lo.astype(np.float32) + hi.astype(np.float32)) * np.float32(0.5)
    half = (hi.astype(np.float32) - lo.astype(np.float32)) * np.float32(0.5)
    expected = np.clip(
        ((((ranked - mid) / half).clip(-1, 1) * 3.0) / 6.0 + 0.5) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    np.testing.assert_array_equal(actual, expected)


def test_all100_collapse_is_old_two_stage_mean() -> None:
    rng = np.random.default_rng(7)
    stack = rng.normal(size=(100, 4, 5, 3)).astype(np.float32)
    actual = MODULE.collapse_all_sweep_mean(stack, 100)
    expected = np.mean(np.mean(stack, axis=1).transpose(1, 0, 2), axis=1)
    np.testing.assert_array_equal(actual, expected.astype(np.float32))


def test_model_u8_range_is_exact() -> None:
    values = np.array([0, 128, 255], dtype=np.uint8)
    actual = MODULE.normalize_u8_for_model(values, 3.0)
    expected = np.array([-3.0, (128.0 / 255.0 * 2.0 - 1.0) * 3.0, 3.0])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)


def test_output_write_is_exclusive_and_float32_exact(tmp_path: Path) -> None:
    matrix = np.array([[0.125, -0.25], [1.0 / 3.0, -1.0 / 7.0]], dtype=np.float32)
    path = tmp_path / "train3.vs.orig.v2"
    receipt = MODULE.write_matrix_exclusive(path, matrix)
    replay = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
    np.testing.assert_array_equal(replay, matrix)
    assert receipt["shape"] == [2, 2]
    try:
        MODULE.write_matrix_exclusive(path, matrix)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing v2 output was replaced")

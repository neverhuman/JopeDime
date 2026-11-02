"""Safety and statistical tests use fixtures only, never scientific payloads."""

import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SPEC = importlib.util.spec_from_file_location(
    "h09_run", Path(__file__).with_name("run.py")
)
h09 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h09)


def packet(tmp_path):
    source = {
        "source_manifest_sha256": h09.SOURCE,
        "excluded_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "excluded_dl_holdout_count": 93,
    }
    (tmp_path / "source_manifest.json").write_text(json.dumps(source))
    selected = {"train": [], "dev": []}
    datasets = []
    for i in range(7):
        name = f"{i:016x}"
        selected["train" if i < 6 else "dev"].append(
            {
                "dataset_id": name,
                "family_id": f"fam-{i}",
                "context_sha256": f"{i:064x}",
                "task": "regression",
                "dl_split": "DL-Train" if i < 6 else "DL-Val",
                "source_split": "inner",
            }
        )
        datasets.append(
            {
                "id": name,
                "canonical_sha256": name + "0" * 48,
                "family_key": f"key-{i}",
                "task": "regression",
                "s3_objects": {
                    "manifest.json": {"key": f"datasets/{name}/manifest.json"}
                },
            }
        )
    (tmp_path / "selected.json").write_text(json.dumps(selected))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {"datasets": datasets, "inventory": {"sealed_name_matches_excluded": []}}
        )
    )
    return selected, datasets, catalog


def test_disjoint_family_roles_and_deterministic_folds(tmp_path):
    _, _, catalog = packet(tmp_path)
    rows = h09.roster(tmp_path, catalog)
    assert rows == h09.roster(tmp_path, catalog)
    roles = {}
    for row in rows:
        roles.setdefault(row["component"], set()).add(row["group"])
    assert all(len(v) == 1 for v in roles.values())
    assert {"tree", "calibrate", "select", "validate"} == {row["group"] for row in rows}


@pytest.mark.parametrize(
    "bad",
    [
        "sealed_family_alias",
        "catalog_overlap",
        "packet_overlap",
        "holdout",
        "sealed",
        "beyondarena",
        "unkeyed",
        "binary",
        "duplicate",
    ],
)
def test_metadata_admission_refuses_protected_or_leaking_rosters_before_payloads(
    tmp_path, bad
):
    selected, datasets, catalog = packet(tmp_path)
    if bad == "catalog_overlap":
        datasets[-1]["family_key"] = datasets[0]["family_key"]
    if bad == "packet_overlap":
        selected["dev"][0]["family_id"] = selected["train"][0]["family_id"]
    if bad == "holdout":
        selected["train"][0]["dl_split"] = "DL-Holdout"
    if bad == "beyondarena":
        datasets[0]["source_url"] = "s3://x/research/BeyondArena/safe-looking"
    if bad == "unkeyed":
        datasets[0]["family_key"] = None
    if bad == "binary":
        datasets[0]["task"] = "binary"
    if bad == "duplicate":
        selected["dev"][0]["context_sha256"] = selected["train"][0]["context_sha256"]
    excluded = [datasets[0]["id"]] if bad == "sealed" else []
    if bad == "sealed_family_alias":
        excluded = ["protected-alias"]
        datasets.append(
            {"id": "protected-alias", "family_key": datasets[0]["family_key"]}
        )
    catalog.write_text(
        json.dumps(
            {
                "datasets": datasets,
                "inventory": {"sealed_name_matches_excluded": excluded},
            }
        )
    )
    (tmp_path / "selected.json").write_text(json.dumps(selected))
    with pytest.raises(ValueError):
        h09.roster(tmp_path, catalog)
    assert not (tmp_path / "pools").exists()  # rejection needs metadata only


def table_bytes(outer=0):
    values = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 2.0], [3.0, 6.0, 1.0]], dtype="<f8")
    return (
        b"JTT1"
        + struct.pack("<III", 3, outer, 2)
        + bytes([2, 0, 1, 0])
        + values.tobytes()
    )


def test_table_summary_uses_only_admitted_train_rows():
    result = h09.context_features(table_bytes())
    assert result.shape == (8,) and np.isfinite(result).all()
    with pytest.raises(ValueError):
        h09.context_features(table_bytes(outer=1))
    with pytest.raises(ValueError):
        h09.context_features(table_bytes()[:-1])


def test_residual_keeps_every_dsl_main_effect_and_interactions_are_conditional():
    rng = np.random.default_rng(3)
    x, c = rng.normal(size=(10, 412)), rng.normal(size=(10, 8))
    sx, sc = StandardScaler().fit(x), StandardScaler().fit(c)
    projection = rng.normal(size=(412, 16))
    z = h09.residual_features(x, c, sx, sc, projection)
    assert z.shape == (10, 540)
    np.testing.assert_array_equal(z[:, :412], sx.transform(x))
    changed = h09.residual_features(x, c + 1, sx, sc, projection)
    np.testing.assert_array_equal(changed[:, :412], z[:, :412])
    assert not np.array_equal(changed[:, 412:], z[:, 412:])


def test_component_weighting_does_not_favor_more_contexts():
    rows = [
        {"component": "a", "fitness": 1},
        {"component": "a", "fitness": 3},
        {"component": "b", "fitness": 8},
    ]
    assert np.mean(list(h09.family_mean(rows).values())) == 5
    assert h09.interval([0, 0, 0]) == [0, 0]
    assert h09.interval([2, 2, 2]) == [2, 2]
    assert h09.interval([-2, -2, -2]) == [-2, -2]
    with pytest.raises(ValueError):
        h09.interval([np.nan, 1])


def test_numeric_model_archive_roundtrip_and_object_payload_refusal(tmp_path):
    rng = np.random.default_rng(91)
    x = rng.normal(size=(40, 412))
    c = rng.normal(size=(40, 8))
    y = x[:, 0] - 0.25 * c[:, 0]
    tree = HistGradientBoostingRegressor(
        max_iter=4, max_leaf_nodes=31, learning_rate=0.05,
        l2_regularization=1, early_stopping=False, random_state=h09.SEED,
    ).fit(np.concatenate((x, c), axis=1), y)
    sx, sc = StandardScaler().fit(x), StandardScaler().fit(c)
    projection = rng.normal(size=(412, 16)) / np.sqrt(412)
    z = h09.residual_features(x, c, sx, sc, projection)
    residual = (Ridge(alpha=10).fit(z, y - tree.predict(np.concatenate((x, c), axis=1))),
                sx, sc, projection)
    archive = tmp_path / "models.npz"
    h09.save_models(tree, residual, archive)
    restored_tree, restored_residual = h09.load_models(archive)
    np.testing.assert_array_equal(
        restored_tree.predict(np.concatenate((x, c), axis=1)),
        tree.predict(np.concatenate((x, c), axis=1)),
    )
    np.testing.assert_array_equal(
        restored_residual[0].predict(h09.residual_features(x, c, *restored_residual[1:])),
        residual[0].predict(z),
    )
    hostile = tmp_path / "hostile.npz"
    import sklearn

    hostile_metadata = {
        "format": "jopedime-h09-numeric-v1",
        "sklearn_version": sklearn.__version__,
        "n_features": 420,
        "n_iter": 1,
        "mapper": {"n_bins": 256, "missing_values_bin_idx": 255, "n_threads": 1,
                   "subsample": 1000, "random_state": 1},
        "thresholds": [],
        "trees": [],
    }
    np.savez(hostile, metadata=np.asarray(json.dumps(hostile_metadata)),
             categorical=np.asarray([object()], dtype=object))
    with pytest.raises(ValueError):
        h09.load_models(hostile)


def test_phase_receipts_cannot_be_overwritten(tmp_path):
    path = tmp_path / "score-start.json"
    h09.write_once(path, {"terminal": 1})
    with pytest.raises(FileExistsError):
        h09.write_once(path, {"terminal": 2})
    assert json.loads(path.read_text()) == {"terminal": 1}

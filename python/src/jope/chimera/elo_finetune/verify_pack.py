"""Verify fable_real_v1.pack: goldens, determinism, loader invariants.

- goldens: N random datasets re-parsed straight from the archive (bit-exact
  X/y; utilities re-derived from raw report JSONs with an independent
  implementation of the fable-elo metric rules, compared at 1e-12).
- rebuild-check: build a second pack in a temp dir; SHA must match.
- invariants: split counts, CSR monotonicity, finiteness, schedule/collate
  determinism.

Usage: python3 -m python.elo_finetune.verify_pack [--golden 8] [--seed 7]
           [--rebuild-check] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from .pack import SPLIT_NAMES, FablePack, bucket_schedule, collate
from .pack_build import ARCHIVE_ROOT, DEFAULT_OUT, build, read_csv_f32, sha256_file

METRIC_ORDER = [
    ("rmse", False),
    ("mse", False),
    ("mae", False),
    ("median_absolute_error", False),
    ("r2", True),
    ("explained_variance", True),
]


def _finite(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def reference_utilities(dataset_dir: Path) -> dict[str, float]:
    """Independent re-implementation of fable-elo's regression utility rules."""
    utils: dict[str, float] = {}
    for path in sorted((dataset_dir / "reports").glob("*.json")):
        try:
            report = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        metrics = report.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        status = report.get("status")
        if not isinstance(status, str):
            status = "completed" if metrics else "unknown"
        if status != "completed":
            continue
        algo = report.get("algorithm")
        if not isinstance(algo, str):
            algo = path.stem
        utility = None
        for name, higher_is_better in METRIC_ORDER:
            score = _finite(metrics.get(name))
            if score is not None:
                utility = score if higher_is_better else -score
                break
        if utility is None:
            continue
        if algo not in utils or utility > utils[algo]:
            utils[algo] = utility
    return utils


def check_goldens(pack: FablePack, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(pack), size=min(n, len(pack)), replace=False)
    for i in picks.tolist():
        view = pack.dataset(i)
        root = ARCHIVE_ROOT / view.dataset_id
        train = read_csv_f32(root / "train.csv")
        test = read_csv_f32(root / "test.csv")
        assert np.array_equal(view.Xtr, train[:, :-1]), f"{view.dataset_id}: Xtr mismatch"
        assert np.array_equal(view.ytr, train[:, -1]), f"{view.dataset_id}: ytr mismatch"
        assert np.array_equal(view.Xte, test[:, :-1]), f"{view.dataset_id}: Xte mismatch"
        assert np.array_equal(view.yte, test[:, -1]), f"{view.dataset_id}: yte mismatch"

        want = reference_utilities(root)
        got = {pack.algos[a]: u for a, u in zip(view.util_algo_ids, view.util_values)}
        assert set(got) == set(want), (
            f"{view.dataset_id}: algo sets differ "
            f"(pack-only={set(got) - set(want)}, ref-only={set(want) - set(got)})"
        )
        for algo, utility in want.items():
            assert abs(got[algo] - utility) <= 1e-12, f"{view.dataset_id}/{algo}: {got[algo]} != {utility}"
    print(f"goldens ok: {len(picks)} datasets bit-exact X/y, utilities at 1e-12")


def check_invariants(pack: FablePack) -> None:
    counts = {name: len(pack.split_indices(name)) for name in SPLIT_NAMES}
    assert counts == {"train": 1629, "val": 340, "test": 353}, counts
    for name in (
        "train_row_offsets",
        "test_row_offsets",
        "x_train_offsets",
        "x_test_offsets",
        "util_offsets",
    ):
        off = pack.section(name)
        assert off[0] == 0 and (np.diff(off) >= 0).all(), f"{name} not monotone"
    for name in ("X_train", "y_train", "X_test", "y_test", "util_values"):
        assert np.isfinite(pack.section(name)).all(), f"{name} has non-finite values"
    ours = pack.algo_id.get("tabpfn_v3_regressor")
    assert ours is not None, "our algorithm missing from algo table"
    have_ours = sum(1 for i in range(len(pack)) if ours in pack.dataset(i).util_algo_ids)
    print(f"invariants ok: splits {counts}, our reports on {have_ours}/{len(pack)} datasets")


def check_schedule_determinism(pack: FablePack) -> None:
    a = bucket_schedule(pack, "train", run_seed=11, epoch=0, micro_size=8)
    b = bucket_schedule(pack, "train", run_seed=11, epoch=0, micro_size=8)
    assert [p.dataset_indices for p in a] == [p.dataset_indices for p in b]
    batch1 = collate(pack, a[0], seed_a=1001)
    batch2 = collate(pack, a[0], seed_a=1001)
    assert np.array_equal(batch1.x, batch2.x) and np.array_equal(batch1.y_qry, batch2.y_qry)
    sizes = {(p.ctx, p.qry, p.n_features) for p in a}
    print(f"schedule ok: {len(a)} train micro-plans, {len(sizes)} distinct shapes")


def check_rebuild(out_dir: Path) -> None:
    original = sha256_file(out_dir / "fable_real_v1.pack")
    with tempfile.TemporaryDirectory(dir=out_dir.parent) as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "real_split_v2.json").write_bytes((out_dir / "real_split_v2.json").read_bytes())
        (tmp_dir / "utilities.jsonl").symlink_to(out_dir / "utilities.jsonl")
        manifest = build(tmp_dir, jobs=32)
    assert manifest["pack_sha256"] == original, f"rebuild differs: {manifest['pack_sha256']} != {original}"
    print(f"rebuild ok: identical sha256 {original[:16]}…")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--golden", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rebuild-check", action="store_true")
    args = parser.parse_args()

    pack = FablePack(args.out / "fable_real_v1.pack")
    check_invariants(pack)
    check_goldens(pack, args.golden, args.seed)
    check_schedule_determinism(pack)
    if args.rebuild_check:
        check_rebuild(args.out)
    print("verify_pack: ALL OK")


if __name__ == "__main__":
    main()

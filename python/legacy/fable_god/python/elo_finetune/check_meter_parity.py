"""Golden: SplicedEloMeter must reproduce fable-elo's exact replay.

Builds the meter on a split with u_base spliced in, then asks the Rust
`fable-elo elo-json` to replay the identical table (cache + extra records +
dataset filter) and compares every rating. Also exercises the incremental
update path with perturbed utilities.

Usage: python3 -m python.elo_finetune.check_meter_parity [--split val]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .elo_meter import SplicedEloMeter
from .pack import DEFAULT_PACK_DIR, FablePack, OUR_ALGORITHM

FABLE_ELO_CANDIDATES = [
    Path("/home/ubuntu/fable_god/rust/target/release/fable-elo"),
    Path.home() / "fable_god_run" / "bin" / "fable-elo",
]


def fable_elo_bin() -> Path:
    import os

    override = os.environ.get("FABLE_ELO_BIN")
    candidates = [Path(override)] if override else []
    for candidate in candidates + FABLE_ELO_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("fable-elo binary not found")


def write_extra(path: Path, pack: FablePack, u_table: dict[int, float]) -> None:
    with open(path, "w") as f:
        for i, utility in sorted(u_table.items()):
            record = {
                "dataset_hash": pack.dataset_ids[i],
                "task_type": "regression",
                "algorithm": OUR_ALGORITHM,
                "metric_name": "rmse",
                "raw_score": -utility,
                "utility": utility,
                "status": "completed",
            }
            f.write(json.dumps(record) + "\n")


def rust_ratings(pack_dir: Path, extra: Path, datasets: Path, out: Path) -> dict[str, float]:
    subprocess.run(
        [
            str(fable_elo_bin()),
            "elo-json",
            "--cache",
            str(pack_dir / "utilities.jsonl"),
            "--extra",
            str(extra),
            "--datasets",
            str(datasets),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return json.loads(out.read_text())["ratings"]


def compare(name: str, meter: SplicedEloMeter, rust: dict[str, float]) -> float:
    ours = meter.ratings_by_name()
    assert set(ours) == set(rust), (
        f"{name}: algo sets differ: meter-only={set(ours) - set(rust)}, "
        f"rust-only={set(rust) - set(ours)}"
    )
    worst = max(abs(ours[a] - rust[a]) for a in ours)
    assert worst <= 1e-9, f"{name}: max rating diff {worst}"
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    pack = FablePack(args.pack_dir / "fable_real_v1.pack")
    proxy = np.load(args.pack_dir / "fable_base_proxy_v1.npz")
    u_base = proxy["u_base"]
    indices = pack.split_indices(args.split).tolist()
    u_init = {i: float(u_base[i]) for i in indices}

    meter = SplicedEloMeter(pack, args.split, u_init)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        datasets = tmp_dir / "datasets.txt"
        datasets.write_text("".join(pack.dataset_ids[i] + "\n" for i in indices))

        extra = tmp_dir / "extra_base.jsonl"
        write_extra(extra, pack, meter.u_table)
        worst_base = compare(
            "base", meter, rust_ratings(args.pack_dir, extra, datasets, tmp_dir / "base.json")
        )

        rng = np.random.default_rng(13)
        picks = rng.choice(indices, size=min(20, len(indices)), replace=False)
        perturbed = {int(i): float(u_base[i] + rng.normal(0, 0.02)) for i in picks}
        meter.update(perturbed)
        extra2 = tmp_dir / "extra_perturbed.jsonl"
        write_extra(extra2, pack, meter.u_table)
        worst_pert = compare(
            "perturbed",
            meter,
            rust_ratings(args.pack_dir, extra2, datasets, tmp_dir / "perturbed.json"),
        )

    print(
        json.dumps(
            {
                "split": args.split,
                "datasets": len(indices),
                "algorithms": len(meter.ratings_by_name()),
                "our_elo_base": round(meter.our_rating, 6),
                "max_diff_base": worst_base,
                "max_diff_after_update": worst_pert,
                "exchange_rate": round(meter.exchange_rate(), 2),
            }
        )
    )
    print("meter parity: OK")


if __name__ == "__main__":
    main()

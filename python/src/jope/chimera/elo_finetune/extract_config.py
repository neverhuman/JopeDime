"""Extract the architecture config dict from the provenance checkpoint.

The deployed safetensors carries weights only; the architecture hyperparams
live in provenance/regressor_source/model.ckpt ("config" key). We dump them
once to weights/starlight_config.json so no node ever needs the 222MB ckpt.

Usage: python3 -m python.elo_finetune.extract_config \
           [--ckpt PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .paths import regressor_source_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt", type=Path, default=regressor_source_dir() / "model.ckpt"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "weights" / "starlight_config.json",
    )
    args = parser.parse_args()

    obj = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    config = obj["config"]
    if not isinstance(config, dict) or not config:
        raise ValueError(f"checkpoint {args.ckpt} has no usable config dict")
    args.out.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "keys": len(config)}))


if __name__ == "__main__":
    main()

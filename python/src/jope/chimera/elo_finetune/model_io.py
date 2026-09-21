"""Strict-load the deployed safetensors regressor into the provenance arch.

The deployed weights (weights/starlight_core_regressor.safetensors) come from
a DIFFERENT source checkpoint than provenance model.ckpt — same architecture,
different values — so weights must always come from the safetensors file and
only the config comes from the extracted JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .paths import import_regressor_train

WEIGHTS_NAME = "starlight_core_regressor.safetensors"
CONFIG_NAME = "starlight_config.json"


def default_weights_dir() -> Path:
    """weights/ next to the workspace or run-dir code tree."""
    pkg_root = Path(__file__).resolve().parents[2]
    for candidate in (pkg_root / "weights", pkg_root.parent / "weights"):
        if (candidate / WEIGHTS_NAME).is_file():
            return candidate
    raise FileNotFoundError(f"{WEIGHTS_NAME} not found near {pkg_root}")


def load_regressor(
    weights_path: str | Path | None = None,
    config_path: str | Path | None = None,
    device: str | torch.device = "cpu",
):
    """Build TabPFNV3 from the extracted config and strict-load the deployed
    safetensors. Returns the model in eval mode; caller flips to train."""
    train_mod = import_regressor_train()
    weights_dir = None
    if weights_path is None or config_path is None:
        weights_dir = default_weights_dir()
    weights_path = Path(weights_path) if weights_path else weights_dir / WEIGHTS_NAME
    config_path = Path(config_path) if config_path else weights_dir / CONFIG_NAME

    config = json.loads(config_path.read_text())
    model = train_mod.get_architecture(train_mod.TabPFNV3Config.from_dict(config))
    state = load_file(str(weights_path), device="cpu")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"strict load failed: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    return model.to(device).eval()


def bucket_centres(model) -> torch.Tensor:
    """Bar-distribution bucket midpoints in z-space (predict_reg.py:49-50)."""
    borders = model.regression_borders.float()
    return borders[:-1] + (borders[1:] - borders[:-1]) / 2


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="strict-load smoke test")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = load_regressor(device=args.device)
    n_params = sum(p.numel() for p in model.parameters())
    centres = bucket_centres(model)
    print(
        json.dumps(
            {
                "params": n_params,
                "buckets": centres.numel(),
                "centre_min": float(centres.min()),
                "centre_max": float(centres.max()),
                "device": args.device,
            }
        )
    )


if __name__ == "__main__":
    main()

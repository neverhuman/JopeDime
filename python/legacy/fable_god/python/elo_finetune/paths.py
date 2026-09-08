"""Path resolution that works in both layouts:

- workspace:  /home/ubuntu/fable_god/python/elo_finetune  with sources at
              /home/ubuntu/fable_god/provenance/regressor_source
- run dir:    ~/fable_god_run/code/python/elo_finetune    with sources at
              ~/fable_god_run/code/regressor_source

FABLE_REGRESSOR_SOURCE overrides both.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def regressor_source_dir() -> Path:
    override = os.environ.get("FABLE_REGRESSOR_SOURCE")
    candidates = [Path(override)] if override else []
    candidates += [
        _PKG_DIR.parent.parent / "provenance" / "regressor_source",
        _PKG_DIR.parent.parent / "regressor_source",
    ]
    for candidate in candidates:
        if (candidate / "starlight_core_regressor_train.py").is_file():
            return candidate
    raise FileNotFoundError(
        "regressor_source not found; set FABLE_REGRESSOR_SOURCE "
        f"(tried {[str(c) for c in candidates]})"
    )


def import_regressor_train():
    """Import the provenance training module without moving or editing it."""
    source_dir = str(regressor_source_dir())
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    import starlight_core_regressor_train as train_mod

    return train_mod

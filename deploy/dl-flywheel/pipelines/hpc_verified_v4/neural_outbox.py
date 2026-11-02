"""Compatibility facade for the pinned continuous neural outbox."""
import importlib as _importlib
from pathlib import Path as _Path
import sys as _sys

_pipeline_root = str(_Path(__file__).resolve().parents[1])
_sys.path.insert(0, _pipeline_root)
try:
    _module = _importlib.import_module("continuous.neural_outbox")
finally:
    _sys.path.remove(_pipeline_root)

globals().update(
    (name, value)
    for name, value in vars(_module).items()
    if not name.startswith("_")
)
del _module, _importlib, _Path, _pipeline_root, _sys

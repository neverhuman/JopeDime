"""Compatibility module for the source-pinned prepared continuous runner."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys as _sys

_name = "_jopedime_prepared_baseline_runner"
_path = Path(__file__).resolve().parents[1] / "prepared" / "baseline_runner.py"
_spec = spec_from_file_location(_name, _path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load canonical continuous runner at {_path}")

_canonical = module_from_spec(_spec)
_sys.modules[_name] = _canonical
_spec.loader.exec_module(_canonical)

# Keep callers and patch sites on the same module object and globals.
_sys.modules[__name__] = _canonical

"""Compatibility alias for the canonical HPC native input module."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_current = sys.modules[__name__]
_shared_path = Path(__file__).resolve().parents[1] / "hpc" / "native_inputs.py"
_spec = spec_from_file_location(__name__, _shared_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"canonical HPC native inputs are unavailable: {_shared_path}")

_shared = module_from_spec(_spec)
sys.modules[__name__] = _shared
try:
    _spec.loader.exec_module(_shared)
except BaseException:
    sys.modules[__name__] = _current
    raise

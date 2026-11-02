"""Compatibility surface for the shared native-input implementation."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_CORE_PATH = _HERE / "native_inputs_core.py"
if not _CORE_PATH.is_file():
    _CORE_PATH = _HERE.parent / "hpc" / "native_inputs_core.py"

_CORE_NAME = "_jope_shared_native_inputs_core"
_CORE_SPEC = spec_from_file_location(_CORE_NAME, _CORE_PATH)
if _CORE_SPEC is None or _CORE_SPEC.loader is None:
    raise ImportError(f"cannot load shared native inputs from {_CORE_PATH}")
_CORE = module_from_spec(_CORE_SPEC)
sys.modules[_CORE_NAME] = _CORE
_CORE_SPEC.loader.exec_module(_CORE)

CANONICALIZER_SHA = _CORE.CANONICALIZER_SHA
MAX_RAW = _CORE.MAX_RAW
PROTOCOL = _CORE.PROTOCOL
WORKER_SHA = _CORE.WORKER_SHA
NativeInputs = _CORE.NativeInputs
OriginalScoreContext = _CORE.OriginalScoreContext
Unsupported = _CORE.Unsupported
bound = _CORE.bound
canonical = _CORE.canonical
digest = _CORE.digest
hash64 = _CORE.hash64
masked_training = _CORE.masked_training
need = _CORE.need
sha = _CORE.sha

# Keep compatibility with callers that persist or inspect these public types.
OriginalScoreContext.__module__ = __name__
Unsupported.__module__ = __name__
NativeInputs.__module__ = __name__

__all__ = [
    "CANONICALIZER_SHA",
    "MAX_RAW",
    "PROTOCOL",
    "WORKER_SHA",
    "NativeInputs",
    "OriginalScoreContext",
    "Unsupported",
    "bound",
    "canonical",
    "digest",
    "hash64",
    "masked_training",
    "need",
    "sha",
]

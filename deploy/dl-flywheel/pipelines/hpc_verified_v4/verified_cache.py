"""Compatibility import for the canonical bounded verified-artifact caches."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_shared_path = Path(__file__).resolve().parents[1] / "hpc" / "verified_cache.py"
if _shared_path.is_symlink() or not _shared_path.is_file():
    raise ImportError(f"canonical verified-cache implementation is unavailable: {_shared_path}")

_module_name = "_jope_hpc_verified_cache"
_shared = sys.modules.get(_module_name)
if _shared is None:
    _spec = spec_from_file_location(_module_name, _shared_path)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"cannot load canonical verified-cache implementation: {_shared_path}")
    _shared = module_from_spec(_spec)
    sys.modules[_module_name] = _shared
    try:
        _spec.loader.exec_module(_shared)
    except BaseException:
        sys.modules.pop(_module_name, None)
        raise

CachedS3Library = _shared.CachedS3Library
CachedNativeTools = _shared.CachedNativeTools
__all__ = ("CachedS3Library", "CachedNativeTools")

"""dltrain: research-only DL training lane for GP program generation (PLAN dltrain-audit-v1, route DL-D1=A)."""
from pathlib import Path as _Path

__version__ = "0.1.0"

# Keep native imports first, then continuous-owned modules, then frozen shared
# grammar modules removed from the duplicate continuous package.
_root = _Path(__file__).resolve().parents[4]
_shared = _root / "pipelines" / "continuous" / "consumer" / "dltrain"
_canonical = _root.parent / "dl-trainonly" / "lane" / "src" / "dl_training_v1" / "dltrain"
for _source in (_shared, _canonical):
    if _source.is_symlink() or not _source.is_dir():
        raise ImportError(f"shared dltrain source package is unavailable: {_source}")
    __path__.append(str(_source))

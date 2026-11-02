"""dltrain: research-only DL training lane for GP program generation (PLAN dltrain-audit-v1, route DL-D1=A)."""
from pathlib import Path as _Path

__version__ = "0.1.0"

# The native offline package shares the continuous tokenizer and sampler source.
# Keep this package's remaining native modules local, while resolving those two
# modules from the canonical consumer package in the same checked source tree.
_shared = _Path(__file__).resolve().parents[4] / "pipelines" / "continuous" / "consumer" / "dltrain"
if _shared.is_symlink() or not _shared.is_dir():
    raise ImportError(f"shared dltrain source package is unavailable: {_shared}")
__path__.append(str(_shared))

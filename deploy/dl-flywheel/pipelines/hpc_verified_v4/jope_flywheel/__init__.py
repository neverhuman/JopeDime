"""HPC v4 runtime imports from the single maintained flywheel package."""
__version__ = '0.1.0'

from pathlib import Path as _Path

_shared = _Path(__file__).resolve().parents[3] / 'jope_flywheel'
if _shared.is_symlink() or not _shared.is_dir():
    raise ImportError(f'canonical jope_flywheel package is unavailable: {_shared}')
__path__.append(str(_shared))

"""Experimental JopeDime flywheel provenance and data-contract tools."""
__version__ = '0.1.0'

# Keep the continuous runtime import entry point, but source modules shared with
# the base package from its single canonical copy.
from pathlib import Path as _Path

_shared = _Path(__file__).resolve().parents[4] / 'jope_flywheel'
if _shared.is_symlink() or not _shared.is_dir():
    raise ImportError(f'canonical jope_flywheel package is unavailable: {_shared}')
__path__.append(str(_shared))

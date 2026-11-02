"""dltrain research lane; shared grammar modules live in the frozen lane package."""
__version__ = "0.1.0"

from pathlib import Path as _Path

_shared = _Path(__file__).resolve().parents[4].parent / "dl-trainonly" / "lane" / "src" / "dl_training_v1" / "dltrain"
if _shared.is_symlink() or not _shared.is_dir():
    raise ImportError(f"canonical dltrain package is unavailable: {_shared}")
__path__.append(str(_shared))

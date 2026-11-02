"""Compatibility import for the canonical measured-timing implementation."""
import sys as _sys

from jope_flywheel import timing as _canonical

# Keep the legacy top-level module name while sharing the canonical module's
# state, including its mutable ACTIVE timing sink.
_sys.modules[__name__] = _canonical

"""Compatibility import for the canonical measured-timing implementation."""
import sys as _sys

from jope_flywheel import timing as _canonical

# Alias the module itself so assignments such as ``timing.ACTIVE = ...`` keep
# the same state as imports through ``jope_flywheel.timing``.
_sys.modules[__name__] = _canonical

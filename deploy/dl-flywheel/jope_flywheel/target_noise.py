"""Stable public API for the canonical target-noise implementation."""
from . import target_noise_core as _core

SCHEMA = _core.SCHEMA
RNG = _core.RNG
TargetNoise = _core._TargetNoiseState
TargetNoise.__name__ = "TargetNoise"
TargetNoise.__qualname__ = "TargetNoise"
TargetNoise.__module__ = __name__
need = _core._require
vector = _core._require_vector
fit = _core._fit_state
apply = _core._apply_state
for _public in (need, vector, fit, apply):
    _public.__module__ = __name__

def __getattr__(name):
    # Preserve access to the former module namespace for existing callers.
    return getattr(_core, name)

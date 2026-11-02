"""Compatibility facade for the shared continuous-runtime protocol."""
from . import protocol_core as _core

SCHEMA = _core.SCHEMA
HEX = _core.HEX
Context = _core._Context
Context.__name__ = "Context"
Context.__qualname__ = "Context"
Context.__module__ = __name__

_EXPORTS = {
    "canonical": "_canonical",
    "digest": "_digest",
    "sha256": "_sha256",
    "finite": "_finite",
    "partition": "_partition",
    "inherit_partition": "_inherit_partition",
    "normalized_quality": "_normalized_quality",
    "freeze_opponents": "_freeze_opponents",
    "compare": "_compare",
    "elo_update": "_elo_update",
}
for _name, _internal in _EXPORTS.items():
    _value = getattr(_core, _internal)
    _value.__name__ = _name
    _value.__qualname__ = _name
    _value.__module__ = __name__
    globals()[_name] = _value

# Preserve the former module namespace, including imported utility names.
for _name in ("asdict", "dataclass", "hashlib", "json", "math", "re"):
    globals()[_name] = getattr(_core, _name)

__all__ = [
    "asdict", "dataclass", "hashlib", "json", "math", "re", "SCHEMA", "HEX",
    "canonical", "digest", "sha256", "finite", "Context", "partition",
    "inherit_partition", "normalized_quality", "freeze_opponents", "compare",
    "elo_update",
]

def __getattr__(name):
    if name in _EXPORTS:
        return getattr(_core, _EXPORTS[name])
    return getattr(_core, name)

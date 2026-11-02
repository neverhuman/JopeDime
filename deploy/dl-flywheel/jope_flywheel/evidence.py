"""Compatibility API for the canonical evidence ledger implementation.

The implementation lives in :mod:`evidence_core`; this module preserves the
historical import path used by flywheel callers.
"""
from __future__ import annotations

from . import evidence_core as _evidence_core

# Preserve the old module namespace, including explicitly imported private
# helpers, while keeping dunder metadata local to this compatibility module.
for _name, _value in vars(_evidence_core).items():
    if _name not in {
        "__name__", "__package__", "__loader__", "__spec__",
        "__file__", "__cached__", "__builtins__",
    }:
        globals()[_name] = _value

"""Canonical validation primitives for flywheel evidence APIs."""
from __future__ import annotations

import math
import re

MAX_JSON_BYTES = 256 * 1024
MAX_DEPTH = 32
_IDENT = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z')
_HASH = re.compile(r'[0-9a-f]{64}\Z')


class EvidenceError(ValueError):
    """Invalid evidence, identity, path or lifecycle request."""


class CollisionError(EvidenceError):
    """An immutable identity already names different content."""


class IntegrityError(EvidenceError):
    """Stored evidence fails its declared chain or manifest binding."""


class LimitError(EvidenceError):
    """An operation exceeds a declared read/write bound."""


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not _IDENT.fullmatch(value):
        raise EvidenceError('identifier must be 1..128 safe ASCII characters')
    return value


def _digest(value: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise EvidenceError('expected lowercase SHA-256')
    return value


def _tree(value, depth=0):
    if depth > MAX_DEPTH:
        raise LimitError('JSON nesting exceeds limit')
    if value is None or type(value) in (str, bool):
        return
    if type(value) is int:
        if value.bit_length() > 256:
            raise LimitError('JSON integer exceeds 256 bits')
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise EvidenceError('nonfinite number')
        return
    if type(value) is list:
        for x in value:
            _tree(x, depth + 1)
        return
    if type(value) is dict:
        for k, x in value.items():
            if type(k) is not str:
                raise EvidenceError('JSON object keys must be strings')
            _tree(x, depth + 1)
        return
    raise EvidenceError('only JSON primitives are accepted')

from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import TextIO


def _nonspace(stream: TextIO) -> str:
    while True:
        value = stream.read(1)
        if not value:
            return ""
        if not value.isspace():
            return value


def _json_string(stream: TextIO, first: str) -> str:
    if first != '"':
        raise ValueError("expected JSON object key")
    chars = [first]
    escaped = False
    while True:
        char = stream.read(1)
        if not char:
            raise ValueError("truncated JSON key")
        chars.append(char)
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return json.loads("".join(chars))


def _json_object(stream: TextIO, first: str) -> dict:
    if first != "{":
        raise ValueError("expected a training row object")
    chars = [first]
    depth = 1
    in_string = False
    escaped = False
    while depth:
        char = stream.read(1)
        if not char:
            raise ValueError("truncated training row")
        chars.append(char)
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
    value = json.loads("".join(chars))
    if not isinstance(value, dict):
        raise ValueError("training row is not a JSON object")
    return value


def read_train_rows_only(path: Path) -> tuple[list[dict], str]:
    """Read only the first top-level `train` array; stop before the next key."""
    digest = hashlib.sha256()

    class HashingReader:
        def __init__(self, inner: TextIO):
            self.inner = inner

        def read(self, count: int = 1) -> str:
            value = self.inner.read(count)
            digest.update(value.encode("utf-8"))
            return value

    with path.open("r", encoding="utf-8") as raw:
        stream = HashingReader(raw)
        if _nonspace(stream) != "{":
            raise ValueError("selected input must start with a JSON object")
        key = _json_string(stream, _nonspace(stream))
        if key != "train":
            raise ValueError("train must be the first selected.json field")
        if _nonspace(stream) != ":" or _nonspace(stream) != "[":
            raise ValueError("selected train field must be an array")
        rows: list[dict] = []
        first = _nonspace(stream)
        if first == "]":
            return rows, digest.hexdigest()
        while True:
            rows.append(_json_object(stream, first))
            separator = _nonspace(stream)
            if separator == "]":
                break
            if separator != ",":
                raise ValueError("expected a training row separator")
            first = _nonspace(stream)
        return rows, digest.hexdigest()

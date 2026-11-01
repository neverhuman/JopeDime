"""Durable strict JSON writer shared by every H197 record path."""

import json
import os
from pathlib import Path


def save(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False)
    with Path(path).open("w") as f:
        f.write(encoded)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

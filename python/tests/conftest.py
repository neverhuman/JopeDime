"""Test configuration for the JopeDime Python package.

The package itself is importable through the editable install (`uv sync`). The research drivers that the
historical tests import as top-level modules (test_exact_train, test_exact_common, ...) live in
jope.research; expose that directory on sys.path so those tests keep working unchanged (WP6).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "src" / "jope" / "research", ROOT / "src" / "jope" / "tools"):
    p = str(extra)
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("JOPE_HOME", str(ROOT.parent))

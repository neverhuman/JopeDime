#!/usr/bin/env python3
"""Provenance bind: PR base vs exact-head for runner-change PRs."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

_ci_dir = str(Path(__file__).resolve().parent)
if _ci_dir not in sys.path:
    sys.path.insert(0, _ci_dir)

from test_provenance_manual import ManualPrBindTests as _ManualPrBindTests
from test_provenance_binding import ProvenanceBindTests as _ProvenanceBindTests


class ManualPrBindTests(_ManualPrBindTests):
    pass


class ProvenanceBindTests(_ProvenanceBindTests):
    pass


if __name__ == "__main__":
    unittest.main()

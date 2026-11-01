"""Private H193 import closure: unchanged H191 partitions and H192 native engine."""

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "_h196_parents193", ROOT / "research/flywheel_selfplay/h193_invalid_veto/parents.py"
)
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
h177, h191, h192, h188 = parent.h177, parent.h191, parent.h192, parent.h188

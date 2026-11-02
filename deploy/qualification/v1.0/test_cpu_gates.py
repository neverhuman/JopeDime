"""CPU qualification controller proofs; no Rust build, GPU, release or dataset jobs."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cpu_gates_cases_a import ControllerA
from cpu_gates_cases_b import ControllerB

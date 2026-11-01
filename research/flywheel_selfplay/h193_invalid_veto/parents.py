"""Private imports of the merged H188 kernel, H192 engine and H191/H177 statistics."""

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "_h193_imports", ROOT / "research/flywheel_selfplay/h191_nested_kernel/parents.py"
)
imports = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = imports
spec.loader.exec_module(imports)
load, aliases, h177 = imports.load, imports.aliases, imports.h177
with aliases(parents=imports):
    t191 = load(
        "h193_t191", "research/flywheel_selfplay/h191_nested_kernel/treatment.py"
    )
with aliases(parents=imports, treatment=t191):
    h191 = load(
        "h193_a191", "research/flywheel_selfplay/h191_nested_kernel/analysis.py"
    )
h188 = load("h193_t188", "research/flywheel_selfplay/h188_kernel_inputs/treatment.py")
a192 = load("h193_a192", "research/flywheel_selfplay/h192_margin_gate/analysis.py")
t192 = load("h193_t192", "research/flywheel_selfplay/h192_margin_gate/treatment.py")
with aliases(analysis=a192, treatment=t192):
    h192 = load("h193_r192", "research/flywheel_selfplay/h192_margin_gate/runner.py")

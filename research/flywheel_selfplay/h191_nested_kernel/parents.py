"""Private imports of merged H186 scoring and the unchanged H177 decision rule."""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    key = "_h191_" + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


@contextmanager
def aliases(**modules):
    previous = {key: sys.modules.get(key) for key in modules}
    try:
        sys.modules.update(modules)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


h177 = load("h177", "research/flywheel_selfplay/h177_pooled_selector/analysis.py")
h186_treatment = load(
    "h186_treatment", "research/flywheel_selfplay/h186_kernel_sweep/treatment.py"
)


def scoring_parent(stat, constants):
    original_parents = load(
        "h186_parents", "research/flywheel_selfplay/h186_kernel_sweep/parents.py"
    )
    with aliases(parents=original_parents):
        original_analysis = load(
            "h186_analysis", "research/flywheel_selfplay/h186_kernel_sweep/analysis.py"
        )
    with aliases(
        parents=original_parents,
        analysis=original_analysis,
        treatment=h186_treatment,
    ):
        module = load(
            "h186_runner", "research/flywheel_selfplay/h186_kernel_sweep/runner.py"
        )
    # The imported Engine body remains unchanged. Its grid/guard configuration is
    # private to this process; no parent file or learned tensor is modified.
    module.stat, module.constants = stat, constants
    return module

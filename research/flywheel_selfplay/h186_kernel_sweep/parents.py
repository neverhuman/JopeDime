"""Pinned parent implementations loaded privately; no parent source is changed."""

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def load(name, relative):
    key = "_h186_" + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


h177 = load("h177", "research/flywheel_selfplay/h177_pooled_selector/analysis.py")
h184_treatment = load(
    "h184_treatment", "research/flywheel_selfplay/h184_adapter_components/treatment.py"
)


def engine_parent():
    aliases = dict(
        analysis=load(
            "h184_analysis",
            "research/flywheel_selfplay/h184_adapter_components/analysis.py",
        ),
        treatment=h184_treatment,
    )
    previous = {key: sys.modules.get(key) for key in aliases}
    try:
        sys.modules.update(aliases)
        return load(
            "h184_runner",
            "research/flywheel_selfplay/h184_adapter_components/runner.py",
        )
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value

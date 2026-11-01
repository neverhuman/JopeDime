"""Pinned H182 numerical/forward helpers; no parent source is modified."""

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / "research/flywheel_selfplay/h182_ensemble_size_dose"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


@lru_cache(maxsize=1)
def numerical():
    return module("h183_h182_analysis", PARENT / "analysis.py")


@lru_cache(maxsize=1)
def runner_parent():
    treatment = module("h183_h182_treatment", PARENT / "treatment.py")
    previous = {name: sys.modules.get(name) for name in ("analysis", "treatment")}
    try:
        sys.modules["analysis"] = numerical()
        sys.modules["treatment"] = treatment
        return module("h183_h182_runner", PARENT / "runner.py")
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

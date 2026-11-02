"""Compatibility entrypoint for the canonical rnk10 acceptance runner."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

CANONICAL = Path(__file__).resolve().parents[2] / "run_acceptance.py"
SPEC = spec_from_file_location("rnk10_canonical_acceptance", CANONICAL)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load canonical acceptance runner: {CANONICAL}")
module = module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

if __name__ == "__main__":
    raise SystemExit(module.main())

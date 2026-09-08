"""Small, explicit entry point for Prime registry and CPU reference checks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from .grammars.prime import GRAMMAR, registry_hash, registry_snapshot_path
from .prime_config import load_config
from .prime_ops import fit_transform


def _load_realize_driver():
    """Load the realized-label champion driver (lives in scripts/, not a package)."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "prime_realize_champions.py"
    spec = importlib.util.spec_from_file_location("prime_realize_champions", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load realize driver at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # 'realize' owns a rich argparse of its own in the driver; forward its args verbatim
    # (peeling here sidesteps argparse.REMAINDER's mishandling of leading options).
    if raw and raw[0] == "realize":
        return _load_realize_driver().main(raw[1:])
    parser = argparse.ArgumentParser(prog="jope-prime")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("registry", help="print the Rust-owned registry snapshot")
    sub.add_parser("config", help="print resolved Prime path configuration")
    sub.add_parser("realize", help="realize champion genomes into prime/runs/realizations")
    transform = sub.add_parser("transform", help="apply one CPU reference transform to JSON rows")
    transform.add_argument("operator", choices=sorted(GRAMMAR.transforms))
    transform.add_argument("--params", default="{}", help="JSON object of operator parameters")
    transform.add_argument("--append", action="store_true")
    args = parser.parse_args(raw)
    if args.command == "registry":
        print(registry_snapshot_path().read_text(encoding="utf-8"))
        return 0
    if args.command == "config":
        config = load_config()
        print(json.dumps({key: str(value) for key, value in config.__dict__.items()}, sort_keys=True))
        return 0
    params = json.loads(args.params)
    if not isinstance(params, dict):
        parser.error("--params must be a JSON object")
    params["append"] = bool(args.append)
    rows = json.load(sys.stdin)
    out, _ = fit_transform(args.operator, rows, params)
    json.dump(out.tolist(), sys.stdout, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

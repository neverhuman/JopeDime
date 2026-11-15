#!/usr/bin/env python3
"""H458 import paths, dependency pins and path guards.

Every H458 entry point calls `setup_paths()` before importing a dependency. The wave reuses the evaluation code of
earlier waves by import (never by copy), so the exact bytes of those modules are part of the experiment: the staged
runtime copy carries `deps_h458.json` (module file name -> sha256) and `verify_deps()` refuses to run on a mismatch.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"  # frozen copies of the unpinned h452-stage modules (present in the staged runtime copy only)
H452_STAGE = Path("/home/ubuntu/h452-stage/code")
PUB_R6 = Path("/home/ubuntu/h450-stage/pub-r6/code")
FLEET_R10 = Path("/home/ubuntu/h450-stage/fleet-r10/code")
PUB_R1 = Path("/home/ubuntu/h448-stage/pub-r1/code")

QUARANTINE = Path("/home/ubuntu/jope-data/h458-quarantine")
EVIDENCE = Path("/home/ubuntu/jope_plan/evidence/H458")
MMAP_V1 = Path("/home/ubuntu/jope-data/h450-fleet-agy/banks/mmap_v1")
EVAL_BANK = Path("/home/ubuntu/jope-data/h450-quarantine/bank_d91_search_f64.npz")
POOL_DIR = QUARANTINE / "family_pool_v1"

# probe banks by content, so a copy moved away from its board directory cannot pass as an evaluation bank
KNOWN_PROBE_BANKS = {"ffe5f998181fda96b726e0293911e12d44d960b11abf41b1a43bd7ccd9cba632": "dev91-freshprobe-v1"}
SEALED_TOKENS = ("CONFIRM", "HOLDOUT", "SEALED", "DL-VAL", "DL-Val", "FMGAP")
DISCLOSURE = {"in_corpus_dev91_disclosure": True, "canonical_promotion_forbidden": True}
WRITE_ROOTS = (str(QUARANTINE) + "/", str(EVIDENCE) + "/")


def dep_dir() -> Path:
    return VENDOR if VENDOR.is_dir() else H452_STAGE


def setup_paths() -> None:
    """Puts this folder and the pinned dependency folders first on `sys.path` and blanks `JOPE_REF_CODE`."""
    os.environ["JOPE_REF_CODE"] = ""
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.dont_write_bytecode = True
    front = [str(HERE), str(dep_dir()), str(PUB_R6), str(FLEET_R10), str(PUB_R1)]
    sys.path[:] = front + [p for p in sys.path if p not in front]


def digest_file(p: Path) -> str:
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def guard_write_path(p: Path) -> Path:
    """H458 writes only inside its quarantine and evidence folders."""
    rp = str(Path(p).resolve())
    if not any((rp + "/").startswith(root) or rp.startswith(root) for root in WRITE_ROOTS):
        raise SystemExit(f"H458 QUARANTINE REFUSAL: {rp} is outside {WRITE_ROOTS}")
    return Path(p)


def loaded_dep_files() -> dict[str, str]:
    """Every already-imported module file that lives in a dependency folder, by file name."""
    roots = tuple(str(d) + "/" for d in (dep_dir(), H452_STAGE, PUB_R6, FLEET_R10, PUB_R1))
    out: dict[str, str] = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and str(Path(f).resolve()).startswith(roots):
            out[Path(f).name] = str(Path(f).resolve())
    return out


def verify_deps(strict: bool = True) -> dict[str, str]:
    """Hashes the imported dependency files; with `deps_h458.json` present they must match it exactly."""
    got = {name: digest_file(Path(fp)) for name, fp in sorted(loaded_dep_files().items())}
    pin_p = HERE / "deps_h458.json"
    if pin_p.is_file():
        pins = json.loads(pin_p.read_text())["sha256"]
        bad = {n: (h, pins.get(n)) for n, h in got.items() if pins.get(n) != h}
        if bad and strict:
            raise SystemExit(f"H458 dependency sha mismatch: {json.dumps(bad, sort_keys=True)[:800]}")
    elif strict and VENDOR.is_dir():
        raise SystemExit("staged runtime copy without deps_h458.json")
    return got

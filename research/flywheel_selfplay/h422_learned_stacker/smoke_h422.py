#!/usr/bin/env python3
"""H422 code smoke of the apply and readout steps on FIT-SEL (a design cohort): the same functions the scored run
calls on DEV, with the cohort constant overridden in this process. Its numbers are in-sample on design contexts and
are not results. Development only.

  smoke_h422.py --cache FITSEL_UNION_CACHE --frozen FROZEN.json --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sel_h422 as S  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S.APPLY_COHORT = "FIT-SEL"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(Path(a.frozen).read_bytes()).hexdigest()
    S.cmd_apply(argparse.Namespace(cache=a.cache, frozen=a.frozen, frozen_sha256=sha, out=str(out / "APPLY_SMOKE.json")))
    S.cmd_readout(argparse.Namespace(apply=str(out / "APPLY_SMOKE.json"), out=str(out / "READOUT_SMOKE.json")))


if __name__ == "__main__":
    main()

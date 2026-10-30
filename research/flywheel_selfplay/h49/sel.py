#!/usr/bin/env python3
"""Offline H49-SEL admission rule and the predeclared noise-event audit."""

from __future__ import annotations

import json
from pathlib import Path

K = 8
STRICT_MARGIN = 0.001


def accept_family_bootstrap_lower_bound(lower_bound: float, margin: float = STRICT_MARGIN) -> bool:
    """Exact ties and gains at the margin fail; lower bound must strictly exceed it."""
    return lower_bound > margin


def historical_noise_audit() -> dict:
    """Classify only the three event facts supplied by the watcher; no live state is opened."""
    events = [
        {"round": "r782", "kind": "exact_tie", "observed_gain": 0.0},
        {"round": "r882", "kind": "exact_tie", "observed_gain": 0.0},
        {"round": "r859", "kind": "single_context_move", "observed_gain": 0.000199},
    ]
    for event in events:
        event["would_pass_strict_margin"] = False
        event["reason"] = "tie" if event["observed_gain"] == 0 else "single-context gain below 0.001 margin"
    return {
        "schema": "jope.h49.sel-audit/v1",
        "buffer_k": K,
        "strict_margin": STRICT_MARGIN,
        "admission_rule": "accept iff family-bootstrap 95% lower bound of median family-mean inner-CV gain > 0.001",
        "historical_input_scope": "only watcher-reported facts; no live-loop files/process were opened",
        "events": events,
        "survived": 0,
        "total": len(events),
    }


def main() -> None:
    output = Path.home() / "jope-data/h49/runs/H49-SEL-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(historical_noise_audit(), indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()

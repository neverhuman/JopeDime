"""Frozen-expert fusion of TabPFN-v3 + TabICL.

Experts are loaded frozen; only the fusion weights (adapters, calibrators, gate,
residual head) train. See docs/best_training_strategy.md for the loss rationale.
"""
import os
import sys

# tabicl 2.1.1 lives in a py3.12 venv here (and in ~/.local on super1); append so
# our own torch/numpy still win import resolution.
_TABICL_PATHS = [
    "/home/ubuntu/quant/synthetic/.venv/lib/python3.12/site-packages",
    os.path.expanduser("~/.local/lib/python3.10/site-packages"),
    os.path.expanduser("~/.local/lib/python3.12/site-packages"),
]
for _p in _TABICL_PATHS:
    if os.path.isdir(os.path.join(_p, "tabicl")) and _p not in sys.path:
        sys.path.append(_p)

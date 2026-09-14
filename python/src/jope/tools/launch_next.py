"""Start only gp-next / controller-next. Never touches the fenced GP or campaign."""
from __future__ import annotations

import subprocess

import fence_honesty as fence

ALLOWED = (fence.NEXT_GP, fence.NEXT_CONTROLLER)


def normalize(unit):
    name = str(unit or "")
    if name and not name.endswith(".service"):
        name += ".service"
    return name


def refuse(unit, action="start", pid=None):
    name = normalize(unit)
    fence.refuse_legacy_restart(action, pid=pid, unit=name)
    if name not in ALLOWED:
        raise ValueError("only gp-next and controller-next may start")
    if fence.idle_gpus_authorize_train(0, True):
        raise ValueError("idle GPUs do not authorize work")
    return name


def start(unit, runner=subprocess.run):
    name = refuse(unit, "start")
    # A new service name does not change the mathematical work or its authority.
    # Activation belongs to the native verified handoff, never this legacy helper.
    raise ValueError("native namespace activation and qualified release required: " + name)


def stop_resident_if_no_dispatcher(gp_next_active, runner=subprocess.run):
    """A GP fence does not revoke resident recovery or qualification custody."""
    return dict(stopped=False, reason="resident cleanup requires owned task and lease reconciliation")

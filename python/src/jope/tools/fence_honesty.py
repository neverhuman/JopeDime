"""Keep the registry fence honest. Never restart legacy GP 774452."""
from __future__ import annotations

LEGACY_GP_PID = 774452
LEGACY_UNIT = "jope-parallel-gp.service"
LEGACY_CAMPAIGN = "jope-parallel-campaign.service"
HANDOFF_NAME = "REGISTRY_HANDOFF_REQUIRED"
NEXT_GP = "jope-parallel-gp-next.service"
NEXT_CONTROLLER = "jope-parallel-controller-next.service"


def refuse_legacy_restart(action, pid=None, unit=None):
    text = str(action or "")
    name = str(unit or "")
    if not name.endswith(".service") and name:
        name = name + ".service"
    if pid == LEGACY_GP_PID or "774452" in text or "774452" in name:
        raise ValueError("restarting 774452 is forbidden")
    if action in ("restart", "unfence", "start-legacy-gp", "systemctl-start-gp"):
        raise ValueError("legacy GP restart/unfence is forbidden")
    if name in (LEGACY_UNIT, LEGACY_CAMPAIGN):
        raise ValueError("legacy unit is fenced")
    return True


def idle_gpus_authorize_train(util_pct, fence_up):
    del util_pct, fence_up
    return False


def historical_failures_are_crash(failed_views, health):
    return health != "planned_migration" and bool(failed_views)


def classify_health(fence, health_receipt=None):
    """Fence is the brake. Historical 104 failures are not a live crash."""
    fence = fence or {}
    receipt = health_receipt or {}
    verified = bool(fence.get("verified"))
    fenced = bool(fence.get("legacy_dispatch_fenced"))
    import_complete = bool(fence.get("historical_import_complete"))
    active = int(receipt.get("active", fence.get("active", 0)) or 0)
    phase = receipt.get("phase") or fence.get("phase")
    if verified and fenced and active == 0:
        phase = "planned_migration"
    return dict(
        schema="parallel.fence_honesty.v1",
        phase=phase,
        verified=verified,
        legacy_dispatch_fenced=fenced,
        historical_import_complete=import_complete,
        registry_dispatch_enabled=False,
        gp_active=active,
        useful_gpu_work_established=False,
        collection_active=False,
        restart_legacy_gp=False,
        paid_idle_is_not_a_license=True,
        historical_104_is_not_a_crash=not historical_failures_are_crash(
            receipt.get("failed_views"), phase
        ),
    )


def inspect_fence(complete, override_exists, handoff_exists, gp_main_pid=0, restart="no"):
    """Pure inspection of saved fence evidence. Does not talk to systemd."""
    complete = complete or {}
    fenced = complete.get("legacy_dispatch_fenced") is True
    import_complete = complete.get("historical_import_complete") is True
    dispatch = complete.get("registry_dispatch_enabled") is True
    verified = (
        override_exists
        and not handoff_exists
        and fenced
        and not import_complete
        and not dispatch
        and int(gp_main_pid or 0) == 0
        and restart == "no"
    )
    return classify_health(
        dict(
            verified=verified,
            legacy_dispatch_fenced=fenced,
            historical_import_complete=import_complete,
            phase="planned_migration" if verified else "migration_fence_unverified",
            active=0 if int(gp_main_pid or 0) == 0 else 1,
        )
    )

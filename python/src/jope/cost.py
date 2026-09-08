from __future__ import annotations

import json
from pathlib import Path

DEFAULT_SAMPLE = {
    "evals": 11370,
    "search_secs": 300.0,
    "gpu_kernel_secs": 33.0,
    "finalist_cv_secs": 222.0,
    "total_secs": 523.0,
}


def estimate_costs(
    *,
    quant_root: str | Path | None = None,
    gpu_hourly_usd: float | None = None,
    cpu_hourly_usd: float | None = None,
) -> dict:
    observed = _observed_from_quant(quant_root) if quant_root is not None else None
    base = observed or DEFAULT_SAMPLE
    gpu_secs = float(base.get("gpu_kernel_secs") or 0.0)
    cpu_secs = float(base.get("total_secs") or 0.0) - gpu_secs
    out = {
        "source": "observed_quant_reports" if observed else "default_seed_constants",
        "evals": base.get("evals"),
        "search_secs": base.get("search_secs"),
        "gpu_kernel_secs": gpu_secs,
        "finalist_cv_secs": base.get("finalist_cv_secs"),
        "total_secs": base.get("total_secs"),
        "local_compute_units": {
            "cpu_seconds": max(0.0, cpu_secs),
            "gpu_seconds": gpu_secs,
        },
    }
    if gpu_hourly_usd is not None or cpu_hourly_usd is not None:
        out["usd"] = {
            "cpu": max(0.0, cpu_secs) / 3600.0 * float(cpu_hourly_usd or 0.0),
            "gpu": gpu_secs / 3600.0 * float(gpu_hourly_usd or 0.0),
        }
        out["usd"]["total"] = out["usd"]["cpu"] + out["usd"]["gpu"]
    return out


def _observed_from_quant(root: str | Path | None) -> dict | None:
    if root is None:
        return None
    root = Path(root)
    paths = sorted(
        {
            *root.glob("regression/*/reports/jain_regressor_300.json"),
            *root.glob("*/reports/jain_regressor_300.json"),
        }
    )
    totals: dict[str, float] = {}
    count = 0
    evals = 0.0
    for path in paths[:256]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        timings = data.get("timings") or {}
        params = data.get("model_params") or {}
        if not timings:
            continue
        count += 1
        for key in ("search_secs", "gpu_kernel_secs", "finalist_cv_secs", "total_secs"):
            if isinstance(timings.get(key), (int, float)):
                totals[key] = totals.get(key, 0.0) + float(timings[key])
        if isinstance(params.get("evals"), (int, float)):
            evals += float(params["evals"])
    if count == 0:
        return None
    observed = {key: value / count for key, value in totals.items()}
    observed["evals"] = int(evals / count) if evals else None
    observed["reports"] = count
    return observed

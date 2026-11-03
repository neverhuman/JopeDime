from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

try:  # system-load telemetry; degrade gracefully if unavailable
    import psutil
except (Exception,):  # pragma: no cover - psutil is a hard dep but keep the TUI resilient
    psutil = None  # type: ignore[assignment]

CODE_PREVIEW_CHARS = 10_000
_HEAD_OPS = {
    "elasticnet",
    "ensemble",
    "ensemble_ridge_gbm",
    "gbm",
    "gpr",
    "histgbm",
    "histgbm2",
    "huber_ridge",
    "knn",
    "krr",
    "ksmooth",
    "logistic",
    "random_forest",
    "ridge",
    "stack",
}

def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _read_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows[-limit:]


def _json_objects_from_log(path: Path) -> list[dict[str, Any]]:
    return [row for row in _read_jsonl(path, limit=200) if isinstance(row, dict)]


def _tail_lines(path: Path, count: int) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-count:]


def _file_age(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def _mount_label(path: Path) -> str:
    try:
        dev = os.stat(path).st_dev
        for part in psutil.disk_partitions(all=False) if psutil else []:
            try:
                if os.stat(part.mountpoint).st_dev == dev:
                    return part.mountpoint
            except OSError:
                continue
    except (Exception,):
        pass
    return "/"


def _gpu_states() -> list[dict[str, Any]]:
    query = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    gpus: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "index": _to_int(parts[0]),
                "name": parts[1],
                "util": _to_float(parts[2]),
                "mem_used": _to_float(parts[3]),
                "mem_total": _to_float(parts[4]),
                "temp": _to_float(parts[5]) if len(parts) > 5 else None,
                "power": _to_float(parts[6]) if len(parts) > 6 else None,
                "power_limit": _to_float(parts[7]) if len(parts) > 7 else None,
            }
        )
    return gpus


def _metric_values(rows: list[dict[str, Any]], key: str, *, fallback: str | None = None) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if not isinstance(value, (int, float)) and fallback is not None:
            value = row.get(fallback)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _latest_rank_status(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in reversed(rows):
        value = row.get("rank_status")
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _spark(values: list[float], width: int = 52, *, label: str = "loss") -> str:
    prefix = f"{label}: " if label else ""
    if not values:
        return f"{prefix}waiting" if label else ""
    values = values[-width:]
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return prefix + "▁" * len(values)
    chars = "▁▂▃▄▅▆▇█"
    out = []
    for value in values:
        idx = int((value - lo) / (hi - lo) * (len(chars) - 1))
        out.append(chars[max(0, min(idx, len(chars) - 1))])
    return prefix + "".join(out)


def _trend_arrow(values: list[float]) -> str:
    if len(values) < 2:
        return "•"
    delta = values[-1] - values[max(0, len(values) - 4)]
    if delta > 1e-4:
        return "▲"
    if delta < -1e-4:
        return "▼"
    return "→"


def _meter(value: float, scale: float, width: int = 12) -> str:
    filled = max(0, min(width, int((value / max(scale, 1e-9)) * width)))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _bar(value: float, *, scale: float, width: int = 16) -> str:
    filled = max(0, min(width, int((value / max(scale, 1e-9)) * width)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _row_genome(row: dict[str, Any]) -> str:
    return str(row.get("canonical_genome") or row.get("input_genome") or row.get("genome") or "")


def _candidate_key(row: dict[str, Any]) -> tuple[str, str] | None:
    genome = _row_genome(row)
    if not genome:
        return None
    dataset = str(row.get("dataset_hash") or row.get("dataset_id") or "")
    return (dataset, genome)


def _evaluation_maps(
    evaluations: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    by_genome: dict[str, dict[str, Any]] = {}
    for row in evaluations:
        key = _candidate_key(row)
        if key is not None:
            by_key[key] = row
            by_genome[key[1]] = row
    return by_key, by_genome


def _matching_eval(
    row: dict[str, Any],
    by_key: dict[tuple[str, str], dict[str, Any]],
    by_genome: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    key = _candidate_key(row)
    if key is not None and key in by_key:
        return by_key[key]
    genome = _row_genome(row)
    return by_genome.get(genome) if genome else None


def _gene_stream_entries(
    proposals: list[dict[str, Any]],
    candidate_audit: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    *,
    limit: int,
) -> list[tuple[str, dict[str, Any], dict[str, Any] | None]]:
    by_key, by_genome = _evaluation_maps(evaluations)
    audit_rows = [row for row in candidate_audit[-max(3, limit // 2) :] if _row_genome(row)]
    proposal_rows = [row for row in proposals[-max(4, limit - len(audit_rows)) :] if _row_genome(row)]
    combined: list[tuple[str, dict[str, Any]]] = [("audit", row) for row in audit_rows] + [
        ("gene", row) for row in proposal_rows
    ]

    seen: set[tuple[str, str, str]] = set()
    entries: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []
    for source, row in combined:
        key = _candidate_key(row)
        dedupe = (source, key[0], key[1]) if key is not None else (source, "", _row_genome(row))
        if dedupe in seen:
            continue
        seen.add(dedupe)
        entries.append((source, row, _matching_eval(row, by_key, by_genome)))
    return entries[-limit:]


def _eval_delta(row: dict[str, Any] | None) -> float | None:
    if not isinstance(row, dict):
        return None
    for key in (
        "oof_delta_vs_incumbent",
        "delta_vs_incumbent",
        "cv_delta_vs_incumbent",
        "median_delta_vs_incumbent",
        "actual_delta",
    ):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    cv = row.get("cv") if isinstance(row.get("cv"), dict) else {}
    for key in ("oof_delta_vs_incumbent", "delta_vs_incumbent", "oof_delta"):
        value = cv.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _code_token_total(row: dict[str, Any]) -> int | None:
    tokens = row.get("code_tokens")
    if isinstance(tokens, dict):
        total = tokens.get("total")
        if isinstance(total, (int, float)):
            return int(total)
        subtotal = 0
        any_value = False
        for value in tokens.values():
            if isinstance(value, (int, float)):
                subtotal += int(value)
                any_value = True
        return subtotal if any_value else None
    if isinstance(tokens, (int, float)):
        return int(tokens)
    return None


def _freshest_age(*ages: float | None) -> float | None:
    present = [age for age in ages if isinstance(age, (int, float))]
    return min(present) if present else None


def _latest_algorithm_row(state: dict[str, Any]) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for key in ("proposals", "candidate_audit", "evaluations"):
        rows.extend(reversed([row for row in state.get(key) or [] if isinstance(row, dict)]))
    preferred: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    for row in rows:
        if not _row_genome(row):
            continue
        if row.get("valid") is False or row.get("candidate_selected") is False:
            fallback.append(row)
            continue
        preferred.append(row)
    return (preferred or fallback or [None])[0]


def _code_for_row(row: dict[str, Any], lang: str, roots: list[Path]) -> tuple[str, str]:
    inline = _inline_code(row, lang)
    if inline:
        return (_clip_code(inline), "inline")
    path = _code_path(row, lang, roots)
    if path is not None:
        code = _read_code(path)
        if code:
            return (_clip_code(code), _truncate(str(path), 34))
    if lang == "rust":
        return (_rust_algorithm_preview(row), "preview")
    return (_python_algorithm_preview(row), "preview")


def _inline_code(row: dict[str, Any], lang: str) -> str | None:
    generated = row.get("generated_code")
    if isinstance(generated, dict):
        value = generated.get(lang) or generated.get(f"{lang}_code")
        if isinstance(value, str) and value.strip():
            return value
    for key in (f"{lang}_code", f"{lang}_source", f"generated_{lang}"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    code = row.get("code")
    if isinstance(code, dict):
        value = code.get(lang) or code.get(f"{lang}_code")
        if isinstance(value, str) and value.strip():
            return value
    return None


def _code_path(row: dict[str, Any], lang: str, roots: list[Path]) -> Path | None:
    values: list[Any] = []
    for key in (f"{lang}_path", f"{lang}_source_path", f"{lang}_code_path"):
        values.append(row.get(key))
    for container_key in ("code_paths", "source_paths", "artifact_paths", "artifacts"):
        container = row.get(container_key)
        if isinstance(container, dict):
            values.extend(
                [container.get(lang), container.get(f"{lang}_path"), container.get(f"{lang}_source_path")]
            )
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        path = Path(value).expanduser()
        candidates = [path] if path.is_absolute() else [root / path for root in roots] + [path]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def _read_code(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return text[:CODE_PREVIEW_CHARS]


def _clip_code(code: str) -> str:
    code = code.strip("\n")
    if len(code) <= CODE_PREVIEW_CHARS:
        return code
    return code[:CODE_PREVIEW_CHARS] + "\n# ... clipped ..."


def _genome_stages(genome: str) -> list[tuple[str, dict[str, str]]]:
    stages: list[tuple[str, dict[str, str]]] = []
    for match in re.finditer(r"\(([A-Za-z_][A-Za-z0-9_\-]*)([^()]*)\)", genome):
        op = match.group(1)
        if op in {"pipe", "sym"}:
            continue
        stages.append((op, _stage_params(match.group(2))))
    return stages


def _stage_params(raw: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for idx, token in enumerate(str(raw).strip().split()):
        if not token:
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            params[key.strip()] = value.strip()
        else:
            params[f"arg{idx}"] = token.strip('"')
    return params


def _split_transforms_head(
    row: dict[str, Any],
) -> tuple[list[tuple[str, dict[str, str]]], tuple[str, dict[str, str]]]:
    stages = _genome_stages(_row_genome(row))
    if not stages:
        head = str(row.get("head_type") or "ridge")
        return [], (head, {})
    last_op, last_params = stages[-1]
    if last_op in _HEAD_OPS:
        return stages[:-1], (last_op, last_params)
    head = str(row.get("head_type") or _head_from_genome(_row_genome(row)) or "ridge")
    return stages, (head, {})


def _python_algorithm_preview(row: dict[str, Any]) -> str:
    transforms, head = _split_transforms_head(row)
    lines = [
        "class Model:",
        "    def fit(self, X, y):",
        "        Z = X",
    ]
    if not transforms:
        lines.append("        # raw features")
    for op, params in transforms[:5]:
        lines.append(f"        Z = apply_stage(Z, {op!r}, {params!r}, y=y)")
    if len(transforms) > 5:
        lines.append(f"        # ... {len(transforms) - 5} more transform stages ...")
    lines.extend(
        [
            f"        self.head = fit_head({head[0]!r}, Z, y, {head[1]!r})",
            "        return self",
            "",
            "    def predict(self, X):",
            "        Z = X",
        ]
    )
    for index, (op, _params) in enumerate(transforms[:5]):
        lines.append(f"        Z = self.transforms[{index}].transform(Z)  # {op}")
    if len(transforms) > 5:
        lines.append("        # remaining fitted transforms replay here")
    lines.append("        return self.head.predict(Z)")
    return "\n".join(lines)


def _rust_algorithm_preview(row: dict[str, Any]) -> str:
    transforms, head = _split_transforms_head(row)
    lines = [
        "pub struct Model {",
        "    transforms: Vec<FittedStage>,",
        "    head: FittedHead,",
        "}",
        "",
        "impl Model {",
        "    pub fn fit(x: &Matrix, y: &[f64]) -> Self {",
        "        let mut z = x.clone();",
        "        let mut transforms = Vec::new();",
    ]
    for op, params in transforms[:4]:
        lines.append(
            "        z = fit_stage("
            f"&mut transforms, z, {json.dumps(op)}, {json.dumps(params, sort_keys=True)}, Some(y));"
        )
    if len(transforms) > 4:
        lines.append(f"        // ... {len(transforms) - 4} more transform stages ...")
    lines.extend(
        [
            "        let head = fit_head("
            f"{json.dumps(head[0])}, &z, y, {json.dumps(head[1], sort_keys=True)});",
            "        Self { transforms, head }",
            "    }",
            "",
            "    pub fn predict(&self, x: &Matrix) -> Vec<f64> {",
            "        let mut z = x.clone();",
        ]
    )
    for index, (op, _params) in enumerate(transforms[:4]):
        lines.append(f"        z = self.transforms[{index}].transform(z); // {op}")
    if len(transforms) > 4:
        lines.append("        // remaining fitted transforms replay here")
    lines.extend(["        self.head.predict(&z)", "    }", "}"])
    return "\n".join(lines)


def _head_from_genome(genome: str) -> str | None:
    if not genome:
        return None
    for head in sorted(_HEAD_OPS, key=len, reverse=True):
        if f"({head}" in genome:
            return head
    return None


def _load_style(pct: float) -> str:
    if pct >= 90:
        return "bold red"
    if pct >= 70:
        return "yellow"
    if pct >= 40:
        return "green"
    return "bright_green"


def _temp_style(temp: float) -> str:
    if temp >= 85:
        return "bold red"
    if temp >= 70:
        return "yellow"
    return "green"


def _ranking_style(pw: float) -> str:
    if pw >= 0.55:
        return "bold bright_green"
    if pw >= 0.52:
        return "green"
    if pw >= 0.5:
        return "yellow"
    return "red"


def _win_style(win: Any) -> str:
    if not isinstance(win, (int, float)):
        return "dim"
    if win > 0.15:
        return "bold bright_green"
    if win > 0.0:
        return "green"
    return "red"


def _freshness_style(age: float | None) -> str:
    if age is None:
        return "dim"
    if age < 15:
        return "bright_green"
    if age < 120:
        return "yellow"
    return "red"


def _fmt_age(age: float | None) -> str:
    if age is None:
        return "—"
    if age < 90:
        return f"{age:0.0f}s ago"
    if age < 5400:
        return f"{age / 60:0.0f}m ago"
    return f"{age / 3600:0.1f}h ago"


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "?"


def _fmt_float(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):+.4f}"


def _fmt_plain_float(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):0.1f}"


def _fmt_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value) * 100.0:0.1f}%"


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _truncate(value: str, width: int) -> str:
    text = str(value).replace("\n", " ")
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


def _dir_size(path: Path) -> str:
    total = 0
    try:
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
    except OSError:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024:
            return f"{total:0.1f} {unit}"
        total /= 1024
    return f"{total:0.1f} TB"

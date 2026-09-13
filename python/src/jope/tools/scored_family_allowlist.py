"""Receipt-only scored-family allowlist. Does not rewrite prepared.json or live corpus."""
from __future__ import annotations

import copy
import json
from pathlib import Path

SCHEMA = "parallel.scored_family_allowlist.v2"
KNOWN_QUARANTINE = {}  # No family-name or target-magnitude heuristics.



def inspect_prepared(prepared):
    task = prepared.get("task") or "unknown"
    family = prepared.get("family") or "unknown"
    reasons = []
    if family in KNOWN_QUARANTINE:
        reasons.append(KNOWN_QUARANTINE[family])
    if task != "regression" and family not in KNOWN_QUARANTINE:
        reasons.append(f"task={task} must not enter the regression scored panel")
    # Calendar year can be a feature or a legitimate declared numeric target.
    # Interleaved positions invalidate an asserted forecasting split, not a
    # regression interpolation problem solely because its source is temporal.
    cuts = prepared.get("cuts") or []
    if task == "timeseries" or prepared.get("prediction_contract") == "forecasting":
        for index, cut in enumerate(cuts):
            if interleaved_cut(cut):
                reasons.append(f"cut-{index} train/val indices interleave under forecasting contract")
    return dict(family=family, task=task, scored=not reasons, reasons=sorted(set(reasons)))


def interleaved_cut(cut):
    train = cut.get("train_indices") or cut.get("selected_train_indices") or []
    val = cut.get("validation_indices") or cut.get("test_indices") or []
    if not train or not val:
        return False
    return max(train) > min(val) and max(val) > min(train)


def scan(prepared_paths):
    rows = []
    for path in prepared_paths:
        prepared = json.loads(Path(path).read_text())
        row = inspect_prepared(prepared)
        row["path"] = str(path)
        rows.append(row)
    return report(rows)


def report(rows, extra_quarantine=None):
    quarantine = dict(KNOWN_QUARANTINE)
    quarantine.update(extra_quarantine or {})
    families = {}
    for row in rows:
        families.setdefault(row["family"], copy.deepcopy(row))
    for family, reason in quarantine.items():
        families.setdefault(family, dict(family=family, task="regression", scored=False,
                                         reasons=[reason], path=None))
        families[family]["scored"] = False
        families[family].setdefault("reasons", []).append(reason)
        families[family]["reasons"] = sorted(set(families[family]["reasons"]))
    scored = sorted(f for f, row in families.items() if row["scored"])
    held = sorted(f for f, row in families.items() if not row["scored"])
    return dict(schema=SCHEMA, eligibility_scope="task-screen-only; identity and contamination qualification required",
                independent_evaluation=False, scored_families=scored, quarantined_families=held,
                rows=[families[f] for f in sorted(families)])


def quality_rows_on_allowlist(quality_report, allowlist):
    allowed = set(allowlist.get("scored_families") or [])
    rows = []
    skipped = 0
    for row in quality_report.get("rows") or []:
        if row.get("family") in allowed:
            rows.append(row)
        else:
            skipped += 1
    return dict(kept=len(rows), skipped=skipped, rows=rows)

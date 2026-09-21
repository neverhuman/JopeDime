"""Freeze volatile production artifacts and harvest (proxy, honest) pairs.

Freeze first, standalone: scratch rotation is the only unrecoverable data
loss. Freezing is `cp -al` hardlinks (scratch and prime/ share one
filesystem), idempotent via `.done` markers keyed on the source inventory.

Harvest sources:
  featmath: frozen jain reports, finalists[] pair search_score/rungs/fold
            scores with honest cv.oof_r2/fold_r2.  Permanent, gold.
  newmath:  frozen scratch dirs; gp_evaluations.json holds the 10 honest
            labels, gp_candidates_*.jsonl hold every generation's proxy rows.
            Join = exact dsl_expr string match (JSON-escaped grep -F prefilter,
            then exact compare on parsed lines).
  negatives: labels with negative=1 from the DB snapshot (off-band probes).
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRATCH_TASKS = Path("/home/ubuntu/jope-scratch/runtime/xbabe2/tasks")
LEAGUE_ARTIFACTS = Path("prime/runs/league/artifacts")
LEAGUE_DB = Path("prime/runs/league/league.db")

FEATMATH_ALGO = "jain_regressor_gen1000"
NEWMATH_SCHEMA = "PrimeNewmathGpReport.v1"


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------


def _inventory_hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime)}\n".encode())
    return h.hexdigest()[:16]


def freeze_scratch(cache_dir: Path) -> dict:
    dest_root = cache_dir / "scratch_freeze"
    dest_root.mkdir(parents=True, exist_ok=True)
    stats = {"eligible": 0, "frozen": 0, "skipped": 0}
    if not SCRATCH_TASKS.is_dir():
        return {**stats, "note": "scratch root missing"}
    for d in sorted(SCRATCH_TASKS.glob("task_*")):
        ev = d / "gp_evaluations.json"
        if not ev.is_file():
            continue
        stats["eligible"] += 1
        dest = dest_root / d.name
        if (dest / ".done").exists():
            stats["skipped"] += 1
            continue
        dest.mkdir(parents=True, exist_ok=True)
        files = [ev, *sorted(d.glob("gp_candidates_*.jsonl"))]
        isl = d / "gp_islands.json"
        if isl.is_file():
            files.append(isl)
        for f in files:
            target = dest / f.name
            if not target.exists():
                subprocess.run(["cp", "-al", str(f), str(target)], check=False)
        (dest / ".done").write_text(_inventory_hash(files) + "\n")
        stats["frozen"] += 1
    return stats


def freeze_reports(cache_dir: Path) -> dict:
    dest = cache_dir / "reports_freeze"
    dest.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for f in sorted(LEAGUE_ARTIFACTS.glob("*.report.json")):
        target = dest / f.name
        if not target.exists():
            subprocess.run(["cp", "-al", str(f), str(target)], check=False)
            n_new += 1
    return {"new": n_new, "total": len(list(dest.glob("*.report.json")))}


def snapshot_db(cache_dir: Path) -> dict:
    dst_path = cache_dir / "league_snapshot.db"
    src = sqlite3.connect(f"file:{LEAGUE_DB}?mode=ro", uri=True, timeout=30)
    dst = sqlite3.connect(str(dst_path))
    src.backup(dst)
    dst.commit()
    meta = dict(
        dst.execute(
            "SELECT key, value FROM metadata WHERE key IN "
            "('completed_gp_candidates','completed_algorithms_searched')"
        )
    )
    n_labels = dst.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    dst.close()
    src.close()
    return {"labels": n_labels, "size_bytes": dst_path.stat().st_size, **meta}


# ---------------------------------------------------------------------------
# featmath harvest
# ---------------------------------------------------------------------------


def harvest_featmath(cache_dir: Path, out_path: Path) -> dict:
    rows = 0
    n_reports = 0
    n_calib_reports = 0
    n_calib_rows = 0
    with out_path.open("w") as out:
        for p in sorted((cache_dir / "reports_freeze").glob("*.report.json")):
            try:
                rep = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if rep.get("algorithm") != FEATMATH_ALGO:
                continue
            n_reports += 1
            calib = rep.get("fitness_calib") or []
            if calib:
                n_calib_reports += 1
                n_calib_rows += len(calib)
            for fin in rep.get("finalists") or []:
                cv = fin.get("cv") or {}
                if cv.get("oof_r2") is None:
                    continue
                fold = fin.get("search_fold_scores") or []
                row = {
                    "engine": "featmath",
                    "dataset": rep.get("dataset_hash"),
                    "report_file": p.name,
                    "task_key": p.name.split(".")[0],
                    "started_at_unix_ms": rep.get("started_at_unix_ms"),
                    "finalist_rank": fin.get("rank"),
                    "genome": fin.get("gene") or fin.get("genome"),
                    "genome_hash": fin.get("genome_hash"),
                    "search_score": fin.get("search_score"),
                    "search_score_rung0": fin.get("search_score_rung0"),
                    "search_score_rung1": fin.get("search_score_rung1"),
                    "search_score_rung2": fin.get("search_score_rung2"),
                    "search_fold_scores": fold,
                    "honest_oof_r2": cv.get("oof_r2"),
                    "honest_mean_r2": cv.get("mean_r2"),
                    "honest_fold_r2": cv.get("fold_r2") or [],
                    "finalist_cv_secs": (rep.get("timings") or {}).get("finalist_cv_secs"),
                    "search_secs": (rep.get("timings") or {}).get("search_secs"),
                    "total_secs": (rep.get("timings") or {}).get("total_secs"),
                }
                out.write(json.dumps(row, sort_keys=True) + "\n")
                rows += 1
    return {"reports": n_reports, "pairs": rows, "calib_reports": n_calib_reports, "calib_rows": n_calib_rows}


# ---------------------------------------------------------------------------
# newmath harvest
# ---------------------------------------------------------------------------


def _grep_candidates(task_dir: Path, genomes: list[str]) -> dict[str, list[dict]]:
    """Exact-match candidate rows per genome via grep -F prefilter."""
    patterns = set()
    for g in genomes:
        patterns.add(g)
        patterns.add(json.dumps(g)[1:-1])  # JSON-escaped form as stored
    pat_file = task_dir / ".fc_patterns"
    pat_file.write_text("\n".join(sorted(patterns)) + "\n")
    cand_files = sorted(task_dir.glob("gp_candidates_*.jsonl"))
    hits: dict[str, list[dict]] = {g: [] for g in genomes}
    if cand_files:
        proc = subprocess.run(
            ["grep", "-hF", "-f", str(pat_file), *[str(f) for f in cand_files]],
            capture_output=True,
            text=True,
        )
        for line in proc.stdout.splitlines():
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            expr = c.get("dsl_expr")
            if expr in hits:
                hits[expr].append(c)
    pat_file.unlink(missing_ok=True)
    return hits


def _population_sample(task_dir: Path, stride: int = 25) -> dict:
    """Stride-sampled predicted_score distribution + generation count."""
    preds: list[float] = []
    complexities: list[float] = []
    n_lines = 0
    max_gen = 0
    for f in sorted(task_dir.glob("gp_candidates_*.jsonl")):
        try:
            gen = int(f.stem.rsplit("_", 1)[1])
            max_gen = max(max_gen, gen)
        except ValueError:
            pass
        with f.open() as fh:
            for i, line in enumerate(fh):
                n_lines += 1
                if i % stride:
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue
                v = c.get("predicted_score")
                if isinstance(v, (int, float)) and math.isfinite(v):
                    preds.append(float(v))
                v = c.get("complexity")
                if isinstance(v, (int, float)):
                    complexities.append(float(v))
    preds.sort()
    return {
        "pred_sorted": preds,
        "n_lines": n_lines,
        "max_generation": max_gen,
        "mean_complexity": (sum(complexities) / len(complexities)) if complexities else None,
    }


def _percentile_of(sorted_vals: list[float], x: float) -> float | None:
    if not sorted_vals or x is None or not math.isfinite(x):
        return None
    import bisect

    lo = bisect.bisect_left(sorted_vals, x)
    hi = bisect.bisect_right(sorted_vals, x)
    return (lo + hi) / 2.0 / len(sorted_vals)


def _harvest_one_newmath(task_dir: Path) -> tuple[list[dict], dict]:
    ev = json.loads((task_dir / "gp_evaluations.json").read_text())
    dataset = ev.get("dataset")
    task_id = ev.get("task_id")
    labels = ev.get("labels") or []
    genomes = [l.get("semantic_genome") for l in labels if l.get("semantic_genome")]
    hits = _grep_candidates(task_dir, genomes)
    pop = _population_sample(task_dir)
    rows: list[dict] = []
    matched = 0
    for lab in labels:
        g = lab.get("semantic_genome")
        prov = lab.get("provenance") or {}
        if lab.get("robust_score") is None:
            continue
        cands = hits.get(g) or []
        if not cands:
            continue
        matched += 1
        cands.sort(key=lambda c: (c.get("generation", 0), c.get("candidate_id", "")))
        first = cands[0]
        preds = [
            c.get("predicted_score") for c in cands if isinstance(c.get("predicted_score"), (int, float))
        ]
        acqs = [
            c.get("acquisition_score") for c in cands if isinstance(c.get("acquisition_score"), (int, float))
        ]
        rows.append(
            {
                "engine": "newmath",
                "dataset": dataset,
                "task_key": task_dir.name,
                "gp_task_id": prov.get("gp_task_id") or task_id,
                "genome": g,
                "predicted_score": first.get("predicted_score"),
                "predicted_score_range": (max(preds) - min(preds)) if preds else None,
                "acquisition_score": first.get("acquisition_score"),
                "acquisition_score_max": max(acqs) if acqs else None,
                "mutation": first.get("mutation"),
                "origin": first.get("origin"),
                "complexity": first.get("complexity"),
                "novelty_hash": first.get("novelty_hash"),
                "first_generation": first.get("generation"),
                "last_generation": cands[-1].get("generation"),
                "occurrences": len(cands),
                "task_max_generation": pop["max_generation"],
                "task_n_lines": pop["n_lines"],
                "task_mean_complexity": pop["mean_complexity"],
                "within_task_pred_percentile": _percentile_of(
                    pop["pred_sorted"], first.get("predicted_score")
                ),
                "honest_robust_score": lab.get("robust_score"),
                "honest_oof_r2": prov.get("oof_r2"),
                "honest_fold_r2": prov.get("fold_r2") or [],
            }
        )
    audit = {"task": task_dir.name, "dataset": dataset, "labels": len(labels), "matched": matched}
    return rows, audit


def harvest_newmath(cache_dir: Path, out_path: Path, audit_path: Path, workers: int = 8) -> dict:
    dirs = sorted(
        d for d in (cache_dir / "scratch_freeze").glob("task_*") if (d / "gp_evaluations.json").is_file()
    )
    all_rows: list[dict] = []
    audits: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rows, audit in ex.map(_harvest_one_newmath, dirs):
            all_rows.extend(rows)
            audits.append(audit)
    all_rows.sort(key=lambda r: (r["dataset"], r["task_key"], r["genome"]))
    with out_path.open("w") as out:
        for r in all_rows:
            out.write(json.dumps(r, sort_keys=True) + "\n")
    audits.sort(key=lambda a: a["task"])
    with audit_path.open("w") as out:
        for a in audits:
            out.write(json.dumps(a, sort_keys=True) + "\n")
    n_labels = sum(a["labels"] for a in audits)
    n_matched = sum(a["matched"] for a in audits)
    low = [a for a in audits if a["labels"] and a["matched"] < 0.8 * a["labels"]]
    return {
        "dirs": len(dirs),
        "labels_seen": n_labels,
        "pairs": n_matched,
        "match_rate": round(n_matched / n_labels, 4) if n_labels else None,
        "low_match_dirs": len(low),
    }


# ---------------------------------------------------------------------------
# negatives
# ---------------------------------------------------------------------------


def harvest_negatives(cache_dir: Path, out_path: Path) -> dict:
    db = sqlite3.connect(f"file:{cache_dir / 'league_snapshot.db'}?mode=ro", uri=True)
    cur = db.execute(
        "SELECT dataset, engine, semantic_genome, "
        "json_extract(label_json,'$.evidence_tier') "
        "FROM labels WHERE negative=1"
    )
    n = 0
    with out_path.open("w") as out:
        for dataset, engine, genome, tier in cur:
            out.write(
                json.dumps(
                    {"dataset": dataset, "engine": engine, "genome": genome, "evidence_tier": tier},
                    sort_keys=True,
                )
                + "\n"
            )
            n += 1
    db.close()
    return {"rows": n}


def run_all(cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "freeze_scratch": freeze_scratch(cache_dir),
        "freeze_reports": freeze_reports(cache_dir),
        "snapshot_db": snapshot_db(cache_dir),
    }
    audit["featmath"] = harvest_featmath(cache_dir, cache_dir / "featmath_pairs.jsonl")
    audit["newmath"] = harvest_newmath(
        cache_dir, cache_dir / "newmath_pairs.jsonl", cache_dir / "newmath_match_audit.jsonl"
    )
    audit["negatives"] = harvest_negatives(cache_dir, cache_dir / "negatives.jsonl")
    return audit

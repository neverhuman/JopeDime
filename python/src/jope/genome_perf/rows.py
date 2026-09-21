"""Row table: (dataset, genome, target) plus the per-dataset resolvability weights.

The target is built here and nowhere else, so there is exactly one definition of "percentile".
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass

import numpy as np

from . import R2_CLIP_FLOOR, STRATA, split_of

LEAGUE_DB = os.environ.get("JOPE_LEAGUE_DB", "/home/ubuntu/jope-prime/prime/runs/league/league.db")


@dataclass
class RowTable:
    dataset: np.ndarray  # (N,) str -- dataset hash
    genome: np.ndarray  # (N,) str -- full S-expression text
    score: np.ndarray  # (N,) f8  -- CLIPPED robust_score
    y: np.ndarray  # (N,) f8  -- within-dataset midrank percentile in [0,1]
    oof_r2: np.ndarray  # (N,) f8  -- nan where absent
    fold_std: np.ndarray  # (N,) f8  -- population std of fold_r2, nan where absent
    se_oof: np.ndarray  # (N,) f8  -- std(fold_r2)/sqrt(k), the per-row measurement noise
    finalist_rank: np.ndarray  # (N,) f8  -- REFERENCE ONLY, never a feature
    split: np.ndarray  # (N,) str -- select | dev | test
    engine: str
    watermark: int

    def __len__(self) -> int:
        return len(self.dataset)


def within_group_midrank_pct(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Tie-aware average-rank percentile in [0,1] per group; 0.5 for singleton groups.

    Mirrors fitness_corrector.table.within_group_midrank_pct so a percentile means the same thing
    in both packages. Midrank matters because clipping at -1 creates a genuine tie mass at the
    floor -- ordinary ranking would fabricate an ordering among -1e51 and -1e11.
    """
    out = np.full(len(values), 0.5, dtype=np.float64)
    for g in np.unique(groups):
        m = groups == g
        v = values[m]
        if len(v) < 2:
            continue
        order = np.argsort(v, kind="stable")
        ranks = np.empty(len(v), dtype=np.float64)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            ranks[order[i : j + 1]] = 0.5 * (i + j)  # midrank of the tie block
            i = j + 1
        out[m] = ranks / (len(v) - 1)
    return out


def load_rows(engine: str, db_path: str = LEAGUE_DB) -> RowTable:
    """Load one honest k=5 stratum, drop negatives, build the percentile target."""
    if engine not in STRATA:
        raise ValueError(f"unknown engine {engine!r}; expected one of {sorted(STRATA)}")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        watermark = con.execute("select max(watermark) from labels").fetchone()[0]
        raw = con.execute(
            "select dataset, semantic_genome, label_json from labels where fold_plan = ? and negative = 0",
            (STRATA[engine],),
        ).fetchall()
    finally:
        con.close()

    ds, gen, sc, oof, fstd, se, frank = [], [], [], [], [], [], []
    for dataset, genome, label_json in raw:
        obj = json.loads(label_json)
        score = obj.get("robust_score")
        if score is None:
            continue  # redundant with negative=0, but the schema allows NULL independently
        prov = obj.get("provenance") or {}
        folds = prov.get("fold_r2")
        if folds:
            arr = np.asarray(folds, dtype=np.float64)
            std = float(arr.std())  # POPULATION std, matching robust_formula
            fstd.append(std)
            se.append(std / np.sqrt(len(arr)))
        else:
            fstd.append(np.nan)
            se.append(np.nan)
        ds.append(dataset)
        gen.append(genome)
        sc.append(max(float(score), R2_CLIP_FLOOR))
        oof.append(float(prov["oof_r2"]) if prov.get("oof_r2") is not None else np.nan)
        frank.append(float(prov["finalist_rank"]) if prov.get("finalist_rank") is not None else np.nan)

    dataset_arr = np.asarray(ds)
    score_arr = np.asarray(sc, dtype=np.float64)
    return RowTable(
        dataset=dataset_arr,
        genome=np.asarray(gen),
        score=score_arr,
        y=within_group_midrank_pct(score_arr, dataset_arr),
        oof_r2=np.asarray(oof, dtype=np.float64),
        fold_std=np.asarray(fstd, dtype=np.float64),
        se_oof=np.asarray(se, dtype=np.float64),
        finalist_rank=np.asarray(frank, dtype=np.float64),
        split=np.asarray([split_of(d) for d in dataset_arr]),
        engine=engine,
        watermark=int(watermark),
    )


def dataset_snr(rt: RowTable) -> dict[str, float]:
    """Per-dataset resolvability: within-dataset score spread / typical per-row measurement noise.

    SNR < 1 means the candidates on this dataset are closer together than the noise on any one of
    their scores -- ranking them is largely fitting coin flips. Two thirds of featmath datasets are
    in that regime, which is why every metric must be reported stratified by this.
    """
    out = {}
    for d in np.unique(rt.dataset):
        m = rt.dataset == d
        spread = float(rt.score[m].std())
        noise = np.nanmedian(rt.se_oof[m])
        out[d] = float(spread / noise) if noise and np.isfinite(noise) and noise > 0 else np.nan
    return out

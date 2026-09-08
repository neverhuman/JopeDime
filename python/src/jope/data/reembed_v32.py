"""V6 Lane A2: upgrade corpus rows from 512-d to 1536-d dataset embeddings.

z_new = concat[existing standardized Hyperion 512, PCA-projected foundation 1024].
Fits the projection (train datasets only), registers it in standardization.json,
then rewrites active_rows.jsonl + self_rows_g*.jsonl + champion_rows_g*.jsonl
in place (atomic tmp+rename; one-time .v31.bak backups).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..encoders.projection import FoundationProjector, fit_projection


def reembed_corpus(art_dir: str | Path, cache_dir: str | Path, log=print) -> dict:
    art_dir, cache_dir = Path(art_dir), Path(cache_dir)
    std_path = art_dir / "standardization.json"
    std = json.load(open(std_path))
    if not std.get("foundation"):
        log("[reembed] fitting PCA projection (train datasets only)...")
        fit_projection(art_dir, cache_dir)
        std = json.load(open(std_path))
    proj = FoundationProjector.from_spec(std["foundation"])
    blocks: dict[str, list[float]] = {}

    def block(h: str) -> list[float] | None:
        if h not in blocks:
            try:
                blocks[h] = proj.block_for_hash(h).tolist()
            except FileNotFoundError:
                blocks[h] = None  # type: ignore[assignment]
        return blocks[h]

    files = [art_dir / "active_rows.jsonl"]
    files += sorted(art_dir.glob("self_rows_g*.jsonl"))
    files += sorted(art_dir.glob("champion_rows_g*.jsonl"))
    stats = {"files": 0, "rows": 0, "upgraded": 0, "missing_block": 0, "already": 0}
    for f in files:
        bak = f.with_suffix(f.suffix + ".v31.bak")
        if not bak.exists():
            shutil.copy2(f, bak)
        tmp = f.with_suffix(".tmp")
        with open(f) as src, open(tmp, "w") as dst:
            for line in src:
                r = json.loads(line)
                stats["rows"] += 1
                emb = r.get("embedding") or []
                h = (r.get("metadata") or {}).get("dataset_hash") or r.get("dataset_id")
                if len(emb) >= 1536:
                    stats["already"] += 1
                elif h and (b := block(h)) is not None:
                    r["embedding"] = list(emb[:512]) + b
                    stats["upgraded"] += 1
                else:
                    stats["missing_block"] += 1
                dst.write(json.dumps(r) + "\n")
        tmp.replace(f)
        stats["files"] += 1
        log(f"[reembed] {f.name} done ({stats['rows']} rows so far)")
    return stats

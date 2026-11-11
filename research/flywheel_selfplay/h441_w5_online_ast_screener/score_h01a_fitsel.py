#!/usr/bin/env python3
"""Step 1 (inside sandbox_h335_docker.sh, H335_GROUP=AGG, zero worker runs, CPU-only):
Scores the stock H0.1-A fold-0 neural screener (`SCR` from H417 Gate G0) on the stored FIT-SEL memo candidate pools
(capped deterministically at 160 candidates per context including all H420 TBOTH shas) and writes create-once
`h01a_scores_FITSEL.npz`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_FAST_LINE_RE = re.compile(
    r'"dsl":\s*"([^"]+)".*?"kind":\s*"search".*?"sha":\s*"([0-9a-f]{64})",\s*"status":\s*"completed".*?"utility":\s*([-+0-9.eE]+)'
)
POOL_SEED = 435_101
MAX_CANDS_PER_CTX = 160


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--memo-dir", default="/home/ubuntu/jope-data/h335-cl/runs/H420_FIT-SEL/gproot_AGG/memo")
    ap.add_argument("--must-shas-json", required=True, help="JSON mapping cid -> list of TBOTH shas to always retain")
    ap.add_argument("--out", required=True, help="create-once .npz output inside /home/ubuntu/jope-data/h335-cl/")
    a = ap.parse_args()

    out_p = Path(a.out)
    if out_p.exists():
        raise SystemExit(f"{out_p} exists (create-once)")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(1)
    import apex_core as A
    import lang_h420 as LH

    A.install()
    LH.install(A.G)
    views = A.S.SearchViewProvider()
    train = A.C.train_rows()
    folds = A.C.family_folds(train)
    fam_of = {r["context_sha256"]: r["family_id"] for r in train}
    for cid, r in A.rows().items():
        fam_of[cid] = r["family_id"]
    nd = Path(A.C.DATA_ROOT) / "neural" / "runs" / "H0.1-A"
    fs = A.S.load_neural_fold_screeners(nd, folds, views, family_of=fam_of)
    f0 = fs.by_fold[0]
    man = A.L.screener_manifest(nd, fs)

    must_map = json.loads(Path(a.must_shas_json).read_text())
    cids = sorted(must_map.keys())
    memo_dir = Path(a.memo_dir)

    all_cids, all_fams = [], []
    offsets = [0]
    shas_flat, dsls_flat, search_u_flat, h01a_flat, in_tboth_flat = [], [], [], [], []
    t0 = time.time()

    for cid in cids:
        mp = memo_dir / f"{cid}.split.jsonl"
        must_set = set(must_map[cid])
        seen = {}
        with mp.open("r") as f:
            for line in f:
                if '"kind": "search"' not in line or '"status": "completed"' not in line:
                    continue
                m = _FAST_LINE_RE.search(line)
                if m:
                    dsl, sha, u_str = m.group(1), m.group(2), m.group(3)
                    u_val = float(u_str)
                    if np.isfinite(u_val) and sha not in seen:
                        seen[sha] = (dsl, u_val)

        must_items = [(sha, seen[sha][0], seen[sha][1], True) for sha in sorted(must_set) if sha in seen]
        other_shas = sorted(s for s in seen if s not in must_set)
        rem_cap = max(0, MAX_CANDS_PER_CTX - len(must_items))
        if len(other_shas) > rem_cap and rem_cap > 0:
            rng = np.random.default_rng(A.C.stream_seed(POOL_SEED, "POOL", cid))
            pick = sorted(rng.choice(len(other_shas), size=rem_cap, replace=False).tolist())
            other_shas = [other_shas[i] for i in pick]
        elif rem_cap == 0:
            other_shas = []
        other_items = [(sha, seen[sha][0], seen[sha][1], False) for sha in other_shas]
        items = must_items + other_items
        dsl_list = [it[1] for it in items]
        sc = f0.score(cid, (0, 1), dsl_list).mean(axis=0).astype(np.float64)

        all_cids.append(cid)
        all_fams.append(fam_of[cid])
        for idx_i, (sha, dsl, u_val, is_tb) in enumerate(items):
            shas_flat.append(sha)
            dsls_flat.append(dsl)
            search_u_flat.append(u_val)
            h01a_flat.append(float(sc[idx_i]))
            in_tboth_flat.append(is_tb)
        offsets.append(len(shas_flat))

    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_p.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        cids=np.array(all_cids),
        families=np.array(all_fams),
        offsets=np.array(offsets, dtype=np.int64),
        shas=np.array(shas_flat),
        dsls=np.array(dsls_flat),
        search_u=np.array(search_u_flat, dtype=np.float64),
        h01a_score=np.array(h01a_flat, dtype=np.float64),
        in_tboth=np.array(in_tboth_flat, dtype=bool),
    )
    tmp.rename(out_p)
    meta = {
        "contexts": len(all_cids),
        "total_candidates": len(shas_flat),
        "wall_s": round(time.time() - t0, 2),
        "npz_sha256": hashlib.sha256(out_p.read_bytes()).hexdigest(),
        "screener_manifest": man,
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    (out_p.parent / "h01a_scores_FITSEL.meta.json").write_text(json.dumps(meta, sort_keys=True, indent=2))
    print(json.dumps({k: meta[k] for k in ("contexts", "total_candidates", "wall_s", "npz_sha256")}), flush=True)


if __name__ == "__main__":
    main()

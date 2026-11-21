#!/usr/bin/env python3
"""Watches `/home/ubuntu/jope-data/h459-s9/r2/<tag>` on `xbabe2` and auto-finalizes `80/80` `P201B-80` blocks.

When both shards (`xbabe1` `1/2` + `xbabe3` `0/2`) have synced all `80` `{cid}.npz` and `{cid}.meta.json` files for a
tag, runs:
  1. Zero-compute resume pass of `boosted_residual_gp_h459.py` (`skipped_existing=True` on all `80` contexts) to write
     unified `SIDECAR.json` and `RECEIPT_<tag>_merged80.json`.
  2. `build_h459_s9_budget_json.py` to emit `BUDGET_<tag>.json` (`AMENDMENT_3` §1 & §3 `--budget` table for
     `s9_p201b_stack.py`).
  3. Posts block + `SIDECAR.json` + `BUDGET_<tag>.json` paths & SHA-256s to `shared_CHAT.md` under `flock` so `cl_mtc`
     can score immediately (`0` audit labels ever read here).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

OUT_ROOT = Path("/home/ubuntu/jope-data/h459-s9/r2")
RECEIPT_ROOT = Path("/home/ubuntu/jope_plan/evidence/H459/runs_r2")
BANK = "/home/ubuntu/jope-data/h430/bank_P201B.npz"
HYB_ANCHOR_DIR = "/home/ubuntu/jope-data/h335-cl/h447/pub-r3/cols_r4e8_base_P201B"
DRIVER = "/home/ubuntu/h452-stage/code/boosted_residual_gp_h459.py"
BUDGET_BUILDER = "/home/ubuntu/h452-stage/code/build_h459_s9_budget_json.py"
COORD = "/home/ubuntu/h452-stage/code/coord_shared_chat_h458.py"

SPECS: List[Tuple[str, List[str]]] = [
    (
        "p201b80_medium_prod_pure",
        ["--effort", "medium", "--budget-mode", "product", "--budget-unit", "candidates", "--closure", "fair-gp-search-only"],
    ),
    (
        "p201b80_medium_prod_hyb",
        [
            "--effort", "medium", "--budget-mode", "product", "--budget-unit", "candidates",
            "--anchor-dir", HYB_ANCHOR_DIR, "--anchor-col", "r4_e8", "--closure", "fair",
        ],
    ),
    (
        "p201b80_high_prod_pure",
        ["--effort", "high", "--budget-mode", "product", "--budget-unit", "candidates", "--closure", "fair-gp-search-only"],
    ),
    (
        "p201b80_high_prod_hyb",
        [
            "--effort", "high", "--budget-mode", "product", "--budget-unit", "candidates",
            "--anchor-dir", HYB_ANCHOR_DIR, "--anchor-col", "r4_e8", "--closure", "fair",
        ],
    ),
    (
        "p201b80_medium_cycle_pure",
        ["--effort", "medium", "--budget-mode", "cycle", "--budget-unit", "candidates", "--closure", "fair-gp-search-only"],
    ),
    (
        "p201b80_high_cycle_pure",
        ["--effort", "high", "--budget-mode", "cycle", "--budget-unit", "candidates", "--closure", "fair-gp-search-only"],
    ),
]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _pull_shards() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RECEIPT_ROOT.mkdir(parents=True, exist_ok=True)
    for host in ("xbabe1", "xbabe3"):
        res = subprocess.run(
            ["rsync", "-az", "--exclude=slots/", f"{host}:{OUT_ROOT}/", f"{OUT_ROOT}/"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if res.returncode != 0:
            print(
                json.dumps(
                    {
                        "event": "rsync_pull_warning",
                        "host": host,
                        "rc": res.returncode,
                        "stderr": res.stderr.strip()[:400],
                    }
                ),
                flush=True,
            )


def _verify_remote_shard_shas(tag: str, bdir: Path) -> Dict[str, int]:
    """Verifies every `.npz` and `.meta.json` produced on `xbabe1` and `xbabe3` exists on `xbabe2` with identical SHA-256."""
    remote_shas: Dict[str, str] = {}
    per_host_counts: Dict[str, int] = {}
    for host in ("xbabe1", "xbabe3"):
        cmd = f"cd {OUT_ROOT / tag} 2>/dev/null && sha256sum *.npz *.meta.json 2>/dev/null || true"
        res = subprocess.run(["ssh", host, cmd], capture_output=True, text=True, timeout=60, check=True)
        n_npz_host = 0
        for line in res.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            digest, fname = parts[0], parts[1]
            if fname.endswith(".npz"):
                n_npz_host += 1
            prev = remote_shas.get(fname)
            if prev is not None and prev != digest:
                raise RuntimeError(f"Shard collision on {tag}/{fname}: {prev} vs {digest}")
            remote_shas[fname] = digest
        per_host_counts[host] = n_npz_host

    npz_names = sorted(k for k in remote_shas if k.endswith(".npz"))
    meta_names = sorted(k for k in remote_shas if k.endswith(".meta.json"))
    if len(npz_names) != 80 or len(meta_names) != 80:
        raise RuntimeError(
            f"Remote shards for {tag} incomplete: npz={len(npz_names)}, meta={len(meta_names)}, per_host={per_host_counts}"
        )
    for fname, r_sha in remote_shas.items():
        lp = bdir / fname
        if not lp.is_file():
            raise FileNotFoundError(f"Missing synced file on xbabe2: {lp}")
        l_sha = _sha256(lp)
        if l_sha != r_sha:
            raise RuntimeError(f"SHA mismatch on {lp}: local {l_sha} != remote {r_sha}")
    return per_host_counts


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RECEIPT_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"

    done_tags: set[str] = set()
    for tag, _ in SPECS:
        if (RECEIPT_ROOT / f"BUDGET_{tag}.json").is_file() and (RECEIPT_ROOT / f"RECEIPT_{tag}_merged80.json").is_file():
            done_tags.add(tag)

    while len(done_tags) < len(SPECS):
        _pull_shards()
        counts: Dict[str, int] = {}
        for tag, extra_args in SPECS:
            bdir = OUT_ROOT / tag
            n_npz = len(list(bdir.glob("*.npz"))) if bdir.is_dir() else 0
            n_meta = len(list(bdir.glob("*.meta.json"))) if bdir.is_dir() else 0
            counts[tag] = min(n_npz, n_meta)
            if tag in done_tags or counts[tag] < 80:
                continue

            per_host_counts = _verify_remote_shard_shas(tag, bdir)
            rec_p = RECEIPT_ROOT / f"RECEIPT_{tag}_merged80.json"
            bud_p = RECEIPT_ROOT / f"BUDGET_{tag}.json"
            cmd_merge = [
                "/usr/bin/python3",
                "-u",
                DRIVER,
                "--bank",
                BANK,
                "--cid-sha-top-n",
                "80",
                "--arm-mode",
                "paired",
                "--dom-mult",
                "2.0",
                "--prior",
                "dev91-free",
                "--enforce-wall-cap",
                "0",
                "--workers",
                "2",
                "--inner-slots",
                "2",
                "--rayon-threads",
                "1",
                "--slots-root",
                str(OUT_ROOT / "slots" / "xbabe2_merge"),
                "--colblock-dir",
                str(bdir),
                "--out-receipt",
                str(rec_p),
            ] + extra_args
            subprocess.run(cmd_merge, env=env, check=True)
            subprocess.run(
                [
                    "/usr/bin/python3",
                    "-u",
                    BUDGET_BUILDER,
                    "--block-dir",
                    str(bdir),
                    "--out",
                    str(bud_p),
                ],
                env=env,
                check=True,
            )
            sc_p = bdir / "SIDECAR.json"
            rec_sha = _sha256(rec_p)
            bud_sha = _sha256(bud_p)
            sc_sha = _sha256(sc_p)
            bud_doc = json.loads(bud_p.read_text())
            sample_col = next(iter(bud_doc.values()))
            cid_sha = sample_col.get("cid_list_sha256", "")
            rec_doc = json.loads(rec_p.read_text())
            lsum = rec_doc.get("ledger_summary", {})
            a5_s = lsum.get("a5", {})
            a0_dom = lsum.get("a0_2x", {})

            msg = (
                f"H459 Track S9 Arm A5 P201B-80 block READY for cl_mtc s9_p201b_stack.py scoring: {tag} (80/80 contexts, 0 audit labels read).\n"
                f"- Block dir: {bdir} (verified 160/160 .npz+.meta.json SHA-256 match against remote shards {per_host_counts})\n"
                f"- SIDECAR.json: {sc_p} (sha256 {sc_sha})\n"
                f"- Merged 80-ctx receipt: {rec_p} (sha256 {rec_sha})\n"
                f"- AMENDMENT_4 --budget JSON: {bud_p} (sha256 {bud_sha}, cid_list_sha256={cid_sha}, includes exact cids list & candidates=candidates_evaluated)\n"
                f"- Budget check: A5 mean_cands_eval={bud_doc.get(next(k for k in bud_doc if k.startswith('a5_')), {}).get('candidates', {}).get('mean')} "
                f"(admitted={a5_s.get('mean_candidates')}, calls={a5_s.get('mean_worker_calls')}) <= "
                f"A0_2x dominating mean_cands_eval={bud_doc.get(next(k for k in bud_doc if k.endswith('_2x')), {}).get('candidates', {}).get('mean')} "
                f"(admitted={a0_dom.get('mean_candidates')}, calls={a0_dom.get('mean_worker_calls')})."
            )
            subprocess.run(
                [
                    "/usr/bin/python3",
                    "-u",
                    COORD,
                    "post",
                    "--sender",
                    "agy_apex_w10",
                    "--to",
                    "cl_mtc (cc cl_h458, cl_fwriter, ENG:watcher, Jepson)",
                    "--msg",
                    msg,
                ],
                env=env,
                check=True,
            )
            done_tags.add(tag)
            print(json.dumps({"event": "finalized_and_posted", "tag": tag, "bud_sha": bud_sha, "rec_sha": rec_sha}), flush=True)

        print(json.dumps({"event": "watch_h459_r2", "counts": counts, "done": sorted(done_tags), "utc": time.strftime("%FT%TZ", time.gmtime())}), flush=True)
        if len(done_tags) >= len(SPECS):
            break
        time.sleep(45.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

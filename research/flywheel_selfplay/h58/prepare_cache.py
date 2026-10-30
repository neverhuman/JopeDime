#!/usr/bin/env python3
"""Precompute frozen Stage-1A/1B tokens, dope629, lm48_z, 584D ctx_feat, and pool priors on CPU for H58."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/flywheel_selfplay"))
import run_fused_closed_loop as base  # noqa: E402

MANIFEST_FILE_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
MANIFEST_ID = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jtt1_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    nt, nq, width = struct.unpack_from("<III", raw, 4)
    if nq != 0 or raw[:4] != b"JTT1" or raw[16] != 2:
        raise RuntimeError(f"invalid train-only two-fold JTT1 table: {path}")
    values = np.frombuffer(raw, dtype="<f8", offset=17 + nt).reshape(nt, width + 1).copy()
    return values[:, :-1], values[:, -1]


def compute_candidate_priors(
    net: torch.nn.Module,
    prog_412: np.ndarray,
    lm48_z_1d: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute real base_prior_cost [M] and sos_prior [M, 2] using frozen network buffers."""
    m = prog_412.shape[0]
    sem_24 = base.compute_semantic_struct_sig_24d(prog_412)
    with torch.no_grad():
        c_t = torch.as_tensor(prog_412, dtype=torch.float32, device=device).unsqueeze(0)
        s_t = torch.as_tensor(sem_24, dtype=torch.float32, device=device).unsqueeze(0)
        l_t = torch.as_tensor(lm48_z_1d, dtype=torch.float32, device=device).unsqueeze(0)
        base_p = net._predict_calibrated_prior_batched_gpu(c_t, s_t, l_t, track_prefix="prior10")[0]
        sos_idx = torch.full((1, m), -1, dtype=torch.long, device=device)
        sos_p = net._compute_sos_prior_batched_gpu(c_t, sos_idx)[0]
    return base_p.cpu().numpy().astype(np.float32), sos_p.cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h50/input")
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / "jope-data/h50/ckpt/round-785.pt")
    parser.add_argument("--promotions", type=Path, default=Path.home() / "jope-data/h46/ckpt/promotions.jsonl")
    parser.add_argument("--out-dir", type=Path, default=Path.home() / "jope-data/h58/cache")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    t0 = time.monotonic()
    torch.set_num_threads(args.threads)

    manifest_path = args.input / "source_manifest.json"
    if sha256(manifest_path) != MANIFEST_FILE_SHA:
        raise RuntimeError("source_manifest.json file SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("source_manifest_sha256") != MANIFEST_ID
        or manifest.get("excluded_dl_holdout_count") != 93
        or manifest.get("excluded_payloads_opened") != 0
        or manifest.get("outer_rows_delivered") != 0
    ):
        raise RuntimeError("source_manifest scope/isolation check failed")

    selected = json.loads((args.input / "selected.json").read_text(encoding="utf-8"))
    train_rows = selected.get("train", [])
    val_rows = selected.get("dev", [])
    if len(train_rows) != 1219 or len({r["family_id"] for r in train_rows}) != 205:
        raise RuntimeError("DL-Train roster mismatch")
    if len(val_rows) != 91 or len({r["family_id"] for r in val_rows}) != 19:
        raise RuntimeError("DL-Val roster mismatch")
    if any(r.get("dl_split") != "DL-Train" for r in train_rows):
        raise RuntimeError("non-DL-Train row in train split")
    if any(r.get("dl_split") != "DL-Val" for r in val_rows):
        raise RuntimeError("non-DL-Val row in dev split")

    all_rows = train_rows + val_rows
    cids = [r["context_sha256"] for r in all_rows]
    cid_to_idx = {cid: i for i, cid in enumerate(cids)}

    spec = importlib.util.spec_from_file_location("h58_fused", args.input / "cpu_exact_fused.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load cpu_exact_fused.py")
    fused = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused)

    device = torch.device("cpu")
    net = fused.FusedJopeNetwork(device=device, adapter_variant="h40b")
    ckpt_state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    backbone_state = {
        k: v
        for k, v in ckpt_state["state_dict"].items()
        if not k.startswith("adapter_10tok.")
        or k.startswith("adapter_10tok.anchor.")
        or k in ("adapter_10tok.prog_mean", "adapter_10tok.prog_std")
    }
    net.load_state_dict(backbone_state, strict=False)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    base.guard.lock_dope_weights_and_eval_mode(net)

    dope_before = base.guard.snapshot_dope_weights(net).composite_sha256
    if dope_before != DOPE_SHA:
        raise RuntimeError(f"DOPE SHA mismatch before extraction: {dope_before}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / "context_features.npz"
    n = len(all_rows)

    if npz_path.exists():
        cached = np.load(npz_path, allow_pickle=False)
        if list(cached["cids"]) == cids:
            tokens_arr = cached["tokens_10x32"]
            masks_arr = cached["mask_10"]
            dope629_arr = cached["dope629"]
            lm48_z_arr = cached["lm48_z"]
            ctx_feat_arr = cached["ctx_feat_584"]
            print(f"[prepare_cache] loaded existing context_features.npz ({n} contexts)", flush=True)
        else:
            raise RuntimeError("cached cids mismatch")
    else:
        tokens_arr = np.zeros((n, 10, 32), dtype=np.float32)
        masks_arr = np.zeros((n, 10), dtype=bool)
        dope629_arr = np.zeros((n, 629), dtype=np.float32)
        lm48_z_arr = np.zeros((n, 48), dtype=np.float32)
        ctx_feat_arr = np.zeros((n, 584), dtype=np.float32)

        ws = base.FusedLandmarkWorkspace(num_threads=args.threads)
        with torch.no_grad():
            for idx, row in enumerate(all_rows):
                if (idx + 1) % 32 == 0:
                    base.headroom()
                cid = row["context_sha256"]
                table_path = args.input / "tables" / f"{cid}.jtt1"
                if sha256(table_path) != row["train_only_jtt1_sha256"]:
                    raise RuntimeError(f"table SHA mismatch for {cid}")
                x_np, y_np = read_jtt1_table(table_path)
                yd = y_np.astype(np.float64)
                yn = ((yd - yd.mean()) / max(float(yd.std()), 1e-8)).astype(np.float32)
                vec, cols, _ = ws.fit_transform_and_extract_bundle_high(
                    base.landmark_float32(x_np), np.ascontiguousarray(yn)
                )
                colbuf = np.zeros((64, 17), np.float32)
                ncols = min(len(cols), 64)
                colbuf[:ncols, : min(cols.shape[1], 17)] = cols[:ncols, :17]

                d629, lm48 = net._run_dope_mlc_v3_and_lm48(
                    torch.as_tensor(vec, device=device).reshape(1, 256),
                    torch.as_tensor(colbuf, device=device).reshape(1, 64, 17),
                    use_landmarks=True,
                )
                tok, mask, lm48_z = net._project_stage1b_tokens_batched_gpu(lm, d629)

                d629_np = d629[0].cpu().numpy().astype(np.float32)
                lm48_z_np = lm48_z[0].cpu().numpy().astype(np.float32)
                tokens_arr[idx] = tok[0].cpu().numpy().astype(np.float32)
                masks_arr[idx] = mask[0].cpu().numpy().astype(bool)
                dope629_arr[idx] = d629_np
                lm48_z_arr[idx] = lm48_z_np
                ctx_feat_arr[idx, 88:200] = d629_np[:112]
                ctx_feat_arr[idx, 536:584] = lm48_z_np

                if (idx + 1) % 200 == 0 or (idx + 1) == n:
                    print(f"[prepare_cache] processed {idx + 1}/{n} tables ({time.monotonic() - t0:.1f}s)", flush=True)

        tmp_npz = args.out_dir / "context_features.tmp.npz"
        np.savez(
            tmp_npz,
            cids=np.asarray(cids),
            tokens_10x32=tokens_arr,
            mask_10=masks_arr,
            dope629=dope629_arr,
            lm48_z=lm48_z_arr,
            ctx_feat_584=ctx_feat_arr,
        )
        tmp_npz.replace(npz_path)

    # Precompute pool priors (variable pool lengths stored via CSR offsets)
    priors_path = args.out_dir / "pool_priors.npz"
    w1_mat = base.evaluator.get_w1_sign_matrix() / np.sqrt(412.0)
    offsets = [0]
    base_prior_list = []
    sos_prior_list = []
    proj16_list = []
    scale_arr = np.zeros((n,), dtype=np.float32)

    for idx, row in enumerate(all_rows):
        cid = row["context_sha256"]
        pool_path = args.input / "pools" / f"{cid}.npz"
        if sha256(pool_path) != row["pool_sha256"]:
            raise RuntimeError(f"pool SHA mismatch for {cid}")
        with np.load(pool_path, allow_pickle=False) as src:
            prog = src["program"].astype(np.float32)
            fit = src["fitness"].astype(np.float64)
        if prog.ndim != 2 or prog.shape[1] != 412 or prog.shape[0] < 16:
            raise RuntimeError(f"unexpected pool shape {prog.shape} for {cid}")
        proj16_list.append((prog.astype(np.float64) @ w1_mat).astype(np.float32))
        scale_arr[idx] = max(float(np.std(fit[:4])), 0.10)
        bp, sp = compute_candidate_priors(net, prog, lm48_z_arr[idx], device)
        base_prior_list.append(bp)
        sos_prior_list.append(sp)
        offsets.append(offsets[-1] + prog.shape[0])
        if (idx + 1) % 300 == 0 or (idx + 1) == n:
            print(f"[prepare_cache] computed priors for {idx + 1}/{n} pools ({time.monotonic() - t0:.1f}s)", flush=True)

    tmp_priors = args.out_dir / "pool_priors.tmp.npz"
    np.savez(
        tmp_priors,
        cids=np.asarray(cids),
        offsets=np.asarray(offsets, dtype=np.int64),
        base_prior_cost=np.concatenate(base_prior_list, axis=0),
        sos_prior=np.concatenate(sos_prior_list, axis=0),
        scale=scale_arr,
        proj16=np.concatenate(proj16_list, axis=0),
    )
    tmp_priors.replace(priors_path)

    # Precompute priors for the 400 evaluated GP journal candidates (promotions.jsonl)
    train_cid_set = {r["context_sha256"] for r in train_rows}
    journal_records = []
    if args.promotions.exists():
        for line in args.promotions.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec["context_sha256"]
            if cid not in train_cid_set:
                continue
            c_idx = cid_to_idx[cid]
            prog_1 = np.asarray(rec["feat412"], dtype=np.float32).reshape(1, 412)
            bp_1, sp_1 = compute_candidate_priors(net, prog_1, lm48_z_arr[c_idx], device)
            journal_records.append(
                {
                    "context_sha256": cid,
                    "program_sha256": str(rec["program_sha256"]),
                    "kind": str(rec.get("kind", "unknown")),
                    "cv_gate_passed": bool(rec["cv_gate_passed"]),
                    "fitness": float(rec["fitness"]),
                    "logratio": float(rec["logratio"]),
                    "feat412": [ float(x) for x in rec["feat412"] ],
                    "proj16": [ float(x) for x in rec["proj16"] ],
                    "base_prior_cost": float(bp_1[0]),
                    "sos_prior": [ float(sp_1[0, 0]), float(sp_1[0, 1]) ],
                }
            )
    journal_path = args.out_dir / "journal_candidates.json"
    journal_path.write_text(json.dumps(journal_records, sort_keys=True) + "\n", encoding="utf-8")

    dope_after = base.guard.snapshot_dope_weights(net).composite_sha256
    if dope_after != DOPE_SHA:
        raise RuntimeError(f"DOPE SHA mismatch after extraction: {dope_after}")

    manifest_out = {
        "schema": "jope.h58.context-feature-cache/v1",
        "source_manifest_file_sha256": MANIFEST_FILE_SHA,
        "source_manifest_sha256": MANIFEST_ID,
        "dope_sha256_before": dope_before,
        "dope_sha256_after": dope_after,
        "cache_npz_sha256": sha256(npz_path),
        "pool_priors_npz_sha256": sha256(priors_path),
        "journal_candidates_sha256": sha256(journal_path),
        "journal_candidates_count": len(journal_records),
        "train_contexts": len(train_rows),
        "val_contexts": len(val_rows),
        "total_contexts": n,
        "excluded_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }
    (args.out_dir / "cache_manifest.json").write_text(
        json.dumps(manifest_out, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest_out), flush=True)


if __name__ == "__main__":
    main()

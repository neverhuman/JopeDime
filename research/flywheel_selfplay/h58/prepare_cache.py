#!/usr/bin/env python3
"""Precompute frozen Stage-1A/1B tokens, dope629, lm48_z, 584D ctx_feat, pool priors, and JTT1 envelopes on CPU for H58."""

from __future__ import annotations

import argparse
import gzip
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
SELECTED_FILE_SHA = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
CHAMPIONS_FILE_SHA = "a444047ffb937dc8d69012cbdbf0891faf3e7d743b127e7cac3baed114cf141d"
FUSED_SOURCE_SHA = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
WORKER_BIN_SHA = "e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
STAGE1_STAGE2_PT_SHA = "136b2381c312fd4bf267249fe0543aa44f5b780a05d57f974d43dd4277588b6d"
DOPE_MULTITASK_PT_SHA = "a36510b9e63050c01d182ccce7e4b489d9fb6ed7e39f8f2948cff420114e8952"
PROG_FEAT_NAMES_SHA = "c4e4373d1bc7e99bc38e44eb4acda22d0f9ecc9e332fa8cb9c6a0b463fe37962"
RUST_EVALUATOR_PY_SHA = "53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900"
MODEL_PY_SHA = "0aa76360a7e5e691e420bd5a88a05b343b5b83edcc645e5f50fb33a3d5756f5a"
PHASE12_MM_PY_SHA = "7f5d30280de5cd47e8af5aa6787e5cb2bff321d984f9cae7d261eabd0a9c0d7a"
MICRO_ROOT = Path("/home/ubuntu/JopeDimeMicro")


def verify_pinned_micro_assets() -> dict[str, str]:
    pins = {
        "stage1_stage2_pt_sha256": (
            MICRO_ROOT / "assets/projections/frozen_stage1_stage2_inference_params_v3.pt",
            STAGE1_STAGE2_PT_SHA,
        ),
        "dope_multitask_pt_sha256": (
            MICRO_ROOT / "assets/models/dope_mlc_v3_multitask_best.pt",
            DOPE_MULTITASK_PT_SHA,
        ),
        "prog_feat_names_sha256": (
            MICRO_ROOT / "assets/projections/program-feature-names.json",
            PROG_FEAT_NAMES_SHA,
        ),
        "rust_evaluator_py_sha256": (
            MICRO_ROOT / "python/rust_evaluator.py",
            RUST_EVALUATOR_PY_SHA,
        ),
        "model_py_sha256": (
            MICRO_ROOT / "python/model.py",
            MODEL_PY_SHA,
        ),
        "phase12_mm_py_sha256": (
            MICRO_ROOT / "python/phase12_multimodal_model.py",
            PHASE12_MM_PY_SHA,
        ),
    }
    verified: dict[str, str] = {}
    for key, (path, expected) in pins.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"pinned asset SHA256 mismatch for {key} ({path}): {actual} != {expected}")
        verified[key] = actual
    return verified


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


def extract_single_table_features(
    net: torch.nn.Module,
    ws: base.FusedLandmarkWorkspace,
    x_np: np.ndarray,
    y_np: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run frozen Stage-1A landmark + DOPE629 and Stage-1B token projection for one table."""
    yd = y_np.astype(np.float64)
    yn = ((yd - yd.mean()) / max(float(yd.std()), 1e-8)).astype(np.float32)
    vec, cols, _ = ws.fit_transform_and_extract_bundle_high(
        base.landmark_float32(x_np), np.ascontiguousarray(yn)
    )
    colbuf = np.zeros((64, 17), np.float32)
    ncols = min(len(cols), 64)
    colbuf[:ncols, : min(cols.shape[1], 17)] = cols[:ncols, :17]

    with torch.no_grad():
        d629, lm48 = net._run_dope_mlc_v3_and_lm48(
            torch.as_tensor(vec, device=device).reshape(1, 256),
            torch.as_tensor(colbuf, device=device).reshape(1, 64, 17),
            use_landmarks=True,
        )
        tok, mask, lm48_z = net._project_stage1b_tokens_batched_gpu(lm48, d629)

    d629_np = d629[0].cpu().numpy().astype(np.float32)
    lm48_z_np = lm48_z[0].cpu().numpy().astype(np.float32)
    tok_np = tok[0].cpu().numpy().astype(np.float32)
    mask_np = mask[0].cpu().numpy().astype(bool)
    ctx_584 = np.zeros((584,), dtype=np.float32)
    ctx_584[88:200] = d629_np[:112]
    ctx_584[536:584] = lm48_z_np
    return tok_np, mask_np, d629_np, lm48_z_np, ctx_584


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

    selected_path = args.input / "selected.json"
    champions_path = args.input / "champions.json"
    fused_source_path = args.input / "cpu_exact_fused.py"
    if sha256(selected_path) != SELECTED_FILE_SHA:
        raise RuntimeError("selected.json file SHA mismatch")
    if sha256(champions_path) != CHAMPIONS_FILE_SHA:
        raise RuntimeError("champions.json file SHA mismatch")
    if sha256(fused_source_path) != FUSED_SOURCE_SHA:
        raise RuntimeError("cpu_exact_fused.py file SHA mismatch")
    if sha256(base.evaluator.WORKER_BIN) != WORKER_BIN_SHA:
        raise RuntimeError("bootstrap.s1main worker binary SHA mismatch")
    pinned_micro = verify_pinned_micro_assets()

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
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
    train_cid_set = {r["context_sha256"] for r in train_rows}

    spec = importlib.util.spec_from_file_location("h58_cpu_exact_fused", fused_source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load cpu_exact_fused.py")
    fused = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused)

    device = torch.device("cpu")
    net = fused.FusedJopeNetwork(device=device, adapter_variant="h40b")
    for p in net.parameters():
        p.requires_grad_(False)
    base.guard.lock_dope_weights_and_eval_mode(net)
    dope_snap = base.guard.snapshot_dope_weights(net)
    if dope_snap.composite_sha256 != DOPE_SHA:
        raise RuntimeError(f"DOPE SHA mismatch: {dope_snap.composite_sha256} != {DOPE_SHA}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_npz_path = args.out_dir / "context_features.npz"
    pool_priors_npz_path = args.out_dir / "pool_priors.npz"

    if cache_npz_path.exists():
        existing_ctx = np.load(cache_npz_path, allow_pickle=False)
        if list(existing_ctx["cids"]) == cids and "ctx_feat_584" in existing_ctx:
            tokens_10x32 = existing_ctx["tokens_10x32"].astype(np.float32)
            mask_10 = existing_ctx["mask_10"].astype(bool)
            dope629 = existing_ctx["dope629"].astype(np.float32)
            lm48_z = existing_ctx["lm48_z"].astype(np.float32)
            ctx_feat_584 = existing_ctx["ctx_feat_584"].astype(np.float32)
        else:
            existing_ctx = None
    else:
        existing_ctx = None

    if existing_ctx is None:
        ws = base.FusedLandmarkWorkspace(num_threads=args.threads)
        n_ctx = len(all_rows)
        tokens_10x32 = np.zeros((n_ctx, 10, 32), dtype=np.float32)
        mask_10 = np.zeros((n_ctx, 10), dtype=bool)
        dope629 = np.zeros((n_ctx, 629), dtype=np.float32)
        lm48_z = np.zeros((n_ctx, 48), dtype=np.float32)
        ctx_feat_584 = np.zeros((n_ctx, 584), dtype=np.float32)

        for idx, row in enumerate(all_rows):
            if idx % 32 == 0:
                base.headroom()
            cid = row["context_sha256"]
            tpath = args.input / "tables" / f"{cid}.jtt1"
            if sha256(tpath) != row["train_only_jtt1_sha256"]:
                raise RuntimeError(f"table SHA mismatch for {cid}")
            x_np, y_np = read_jtt1_table(tpath)
            tok_i, msk_i, d629_i, lm48_i, ctx584_i = extract_single_table_features(
                net, ws, x_np, y_np, device
            )
            tokens_10x32[idx] = tok_i
            mask_10[idx] = msk_i
            dope629[idx] = d629_i
            lm48_z[idx] = lm48_i
            ctx_feat_584[idx] = ctx584_i

        np.savez(
            cache_npz_path,
            cids=np.asarray(cids),
            tokens_10x32=tokens_10x32,
            mask_10=mask_10,
            dope629=dope629,
            lm48_z=lm48_z,
            ctx_feat_584=ctx_feat_584,
        )

    if not pool_priors_npz_path.exists():
        offsets = np.zeros((len(all_rows) + 1,), dtype=np.int64)
        base_prior_chunks = []
        sos_prior_chunks = []
        proj16_chunks = []
        pool_scales = np.zeros((len(all_rows),), dtype=np.float32)

        for idx, row in enumerate(all_rows):
            if idx % 32 == 0:
                base.headroom()
            cid = row["context_sha256"]
            ppath = args.input / "pools" / f"{cid}.npz"
            if sha256(ppath) != row["pool_sha256"]:
                raise RuntimeError(f"pool SHA mismatch for {cid}")
            with np.load(ppath, allow_pickle=False) as z:
                prog = z["program"].astype(np.float32)
                fit = z["fitness"].astype(np.float64)
            bp, sp = compute_candidate_priors(net, prog, lm48_z[idx], device)
            offsets[idx + 1] = offsets[idx] + prog.shape[0]
            base_prior_chunks.append(bp)
            sos_prior_chunks.append(sp)
            proj16_chunks.append(base.project(prog).astype(np.float32))
            pool_scales[idx] = max(float(np.std(fit[:4])), 0.10)

        np.savez(
            pool_priors_npz_path,
            cids=np.asarray(cids),
            offsets=offsets,
            base_prior_cost=np.concatenate(base_prior_chunks, axis=0),
            sos_prior=np.concatenate(sos_prior_chunks, axis=0),
            scale=pool_scales,
            proj16=np.concatenate(proj16_chunks, axis=0),
        )

    # Verify all DL-Train JTT1 tables and stage compressed S1 two-fold envelopes for live GP evaluation
    jtt1_gz_dir = args.out_dir / "jtt1_gz"
    jtt1_gz_dir.mkdir(parents=True, exist_ok=True)
    jtt1_gz_sha_by_cid: dict[str, str] = {}
    for idx, row in enumerate(train_rows):
        if idx % 64 == 0:
            base.headroom()
        cid = row["context_sha256"]
        tpath = args.input / "tables" / f"{cid}.jtt1"
        raw = tpath.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["train_only_jtt1_sha256"]:
            raise RuntimeError(f"table SHA mismatch for {cid}")
        expected_env = base.training_cv_envelope(raw)
        gz_path = jtt1_gz_dir / f"{cid}.jtt1.gz"
        if not gz_path.exists():
            gz_path.write_bytes(gzip.compress(expected_env, compresslevel=1, mtime=0))
        elif gzip.decompress(gz_path.read_bytes()) != expected_env:
            raise RuntimeError(f"existing jtt1.gz envelope content mismatch for {cid}")
        jtt1_gz_sha_by_cid[cid] = sha256(gz_path)
    jtt1_gz_manifest_path = args.out_dir / "jtt1_gz_manifest.json"
    jtt1_gz_manifest_path.write_text(
        json.dumps(jtt1_gz_sha_by_cid, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Pre-featurize champions.json for DL-Train contexts
    lm_by_cid = {cid: lm48_z[i] for i, cid in enumerate(cids)}
    champions_raw = json.loads(champions_path.read_text(encoding="utf-8"))
    champions_featurized = []
    for rec in champions_raw:
        cid = rec.get("context_sha256")
        if cid not in train_cid_set:
            continue
        f412, pr16 = base.evaluator.featurize_dsl_412(rec["dsl"])
        champions_featurized.append(
            {
                "context_sha256": cid,
                "family_id": rec["family_id"],
                "dl_split": "DL-Train",
                "dsl": rec["dsl"],
                "program_sha256": rec["program_sha256"],
                "feat412": f412.astype(np.float32).tolist(),
                "proj16": pr16.astype(np.float32).tolist(),
                "landmark_summary": lm_by_cid[cid][:12].astype(np.float32).tolist(),
            }
        )
    champions_cache_path = args.out_dir / "champions_featurized.json"
    champions_cache_path.write_text(json.dumps(champions_featurized) + "\n", encoding="utf-8")

    journal_candidates = []
    if args.promotions.exists():
        seen_keys = set()
        for line in args.promotions.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            cid = ev.get("context_sha256")
            psha = ev.get("program_sha256")
            if cid not in train_cid_set or not psha:
                continue
            if (cid, psha) in seen_keys:
                continue
            seen_keys.add((cid, psha))
            feat = np.asarray(ev["feat412"], dtype=np.float32).reshape(1, 412)
            bp, sp = compute_candidate_priors(net, feat, lm_by_cid[cid], device)
            journal_candidates.append(
                {
                    "context_sha256": cid,
                    "family_id": ev.get("family_id"),
                    "program_sha256": psha,
                    "dsl": ev.get("dsl", ""),
                    "kind": ev.get("kind", "evaluation"),
                    "cv_gate_passed": bool(ev.get("cv_gate_passed", True)),
                    "fitness": float(ev["fitness"]),
                    "logratio": float(ev.get("logratio", ev["fitness"])),
                    "feat412": feat[0].tolist(),
                    "proj16": np.asarray(ev["proj16"], dtype=np.float32).tolist(),
                    "base_prior_cost": float(bp[0]),
                    "sos_prior": sp[0].tolist(),
                }
            )

    journal_path = args.out_dir / "journal_candidates.json"
    if not journal_path.exists():
        journal_path.write_text(json.dumps(journal_candidates, indent=2) + "\n", encoding="utf-8")

    after_snap = base.guard.snapshot_dope_weights(net)
    if after_snap.composite_sha256 != DOPE_SHA:
        raise RuntimeError("DOPE SHA changed during cache prep")

    receipt = {
        "schema": "jope.h58.context-feature-cache/v3",
        "source_manifest_file_sha256": MANIFEST_FILE_SHA,
        "source_manifest_sha256": MANIFEST_ID,
        "selected_file_sha256": SELECTED_FILE_SHA,
        "champions_file_sha256": CHAMPIONS_FILE_SHA,
        "fused_source_sha256": FUSED_SOURCE_SHA,
        "worker_bin_sha256": WORKER_BIN_SHA,
        "dope_sha256_before": dope_snap.composite_sha256,
        "dope_sha256_after": after_snap.composite_sha256,
        **pinned_micro,
        "train_contexts": len(train_rows),
        "val_contexts": len(val_rows),
        "total_contexts": len(all_rows),
        "cache_npz_sha256": sha256(cache_npz_path),
        "pool_priors_npz_sha256": sha256(pool_priors_npz_path),
        "journal_candidates_count": len(journal_candidates),
        "journal_candidates_sha256": sha256(journal_path),
        "champions_featurized_count": len(champions_featurized),
        "champions_featurized_sha256": sha256(champions_cache_path),
        "jtt1_gz_count": len(train_rows),
        "jtt1_gz_manifest_sha256": sha256(jtt1_gz_manifest_path),
        "excluded_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }
    (args.out_dir / "cache_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fair Same-Fold 2-Fold Inner-CV (`jope.s1-two-fold/v1`) Evaluator for Tabular Foundation Models
on `DL-Val91` (`91` Unseen Validation Contexts / `19` Unseen Dataset Families).

Evaluates the external tabular foundation models on the exact same 2-fold `JTT1` inner-CV splits
(`val91_jtt1_envelopes/{cid}.jtt1`) used by `H257`, `H256`, `R882`, and the 4 Frontier LLM regressors
(`Opus5p5_Extra`, `Gemini4_argon_high`, `GPT6p1_Sol_Max`, `GPT6_Astra_Max`):
  1. `TabFM` (`google/tabfm-1.0.0-pytorch`, `regression/model.safetensors`, ~1.6B params)
  2. `Kumo_Tabular_L` (`nvidia/Kumo-Tabular` large, `large/regressor.pt`)
  3. `Kumo_Tabular_S` (`nvidia/Kumo-Tabular` small, `small/regressor.pt`)
  4. `PriorLabs_TabPFN_3p5` (`PriorLabs/TabPFN` v3.5, `tabpfn-v3.5-20260909.safetensors`)
  5. `TabICL_v2` (`jingang/TabICL`, `tabicl-regressor-v2-20260212.ckpt`)
  6. `TabDPT_1p3` (`Layer6/TabDPT`, `tabdpt1_3.safetensors`)
  7. `Mitra_v2` (`autogluon/mitra-regressor-2`, `model.safetensors`)

Protocol (`jope.s1-two-fold/v1`):
  - Fold 0: Fit on `folds == 1`, predict `folds == 0` -> `rmse_fold0`, `base_rmse_fold0 = std(y0 around mean(y1))`
  - Fold 1: Fit on `folds == 0`, predict `folds == 1` -> `rmse_fold1`, `base_rmse_fold1 = std(y1 around mean(y0))`
  - `ratio_fold0 = rmse_fold0 / base_rmse_fold0`, `ratio_fold1 = rmse_fold1 / base_rmse_fold1`
  - `logratio = 0.5 * (log(ratio_fold0) + log(ratio_fold1))`
  - `sd_logratio = 0.5 * |log(ratio_fold0) - log(ratio_fold1)|`
  - `cv_geomean_rmse_ratio = exp(logratio)`
  - `cv_fold_wins = int(ratio_fold0 < 1.0) + int(ratio_fold1 < 1.0)`
  - `cv_gate_passed = bool(cv_geomean_rmse_ratio <= 0.90 and cv_fold_wins == 2)`
  - `s1_fitness_zero_comp = 0.9 * logratio + 0.2 * sd_logratio`
  - `unconstrained_utility = -s1_fitness_zero_comp`
  - `s1_utility_zero_comp = -s1_fitness_zero_comp if cv_gate_passed else -2.0`
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

OFM_ROOT = Path("/home/ubuntu/jope-evidence/jopedime_probe_exp/ofm-20261008")
TABPFN35_ROOT = Path("/home/ubuntu/scratch/tabpfn35-encoder-20260927")
ENV_DIR = Path("/home/ubuntu/jope-data/h257/cache/val91_jtt1_envelopes")
OUT_JSONL = Path("/home/ubuntu/jope-data/h257/cache/tabular_foundation_dl_val91_results.jsonl")
OUT_CACHE_JSON = Path("/home/ubuntu/jope-data/h257/cache/tabular_foundation_dl_val91_cache.json")


def load_hf_token_if_present() -> None:
    """Loads HF_TOKEN from /home/ubuntu/.env.jnoccio into the process environment without printing."""
    env_file = Path("/home/ubuntu/.env.jnoccio")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "HF_TOKEN":
                    os.environ["HF_TOKEN"] = v.strip().strip('"').strip("'")


def parse_jtt1_2fold(jtt1_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Parses a 2-fold JTT1 envelope into `(X0_tr, y0_tr, X0_te, y0_te, b0, b1)`."""
    raw = jtt1_path.read_bytes()
    nt, _nq, width = struct.unpack_from("<III", raw, 4)
    folds = np.frombuffer(raw, dtype=np.uint8, count=nt, offset=17)
    mat = np.frombuffer(raw, dtype="<f8", count=nt * (width + 1), offset=17 + nt).reshape(nt, width + 1)
    X = mat[:, :-1].astype(np.float64)
    y = mat[:, -1].astype(np.float64)

    X0_tr, y0_tr = X[folds == 1].copy(), y[folds == 1].copy()
    X0_te, y0_te = X[folds == 0].copy(), y[folds == 0].copy()
    b0 = float(np.sqrt(np.mean((y0_te - float(np.mean(y0_tr))) ** 2)))
    b1 = float(np.sqrt(np.mean((y0_tr - float(np.mean(y0_te))) ** 2)))
    return X0_tr, y0_tr, X0_te, y0_te, b0, b1


def compute_s1_record(
    cid: str,
    entrant: str,
    y0_te: np.ndarray,
    pred0: np.ndarray,
    b0: float,
    y1_te: np.ndarray,
    pred1: np.ndarray,
    b1: float,
    wall_ms: float,
) -> Dict[str, Any]:
    """Computes exact `jope.s1-two-fold/v1` metrics from Fold 0 and Fold 1 predictions."""
    if not (np.all(np.isfinite(pred0)) and np.all(np.isfinite(pred1))):
        return {
            "cid": cid,
            "entrant": entrant,
            "status": "failed",
            "rmse_fold0": None,
            "rmse_fold1": None,
            "base_rmse_fold0": round(b0, 10),
            "base_rmse_fold1": round(b1, 10),
            "ratio_fold0": None,
            "ratio_fold1": None,
            "cv_geomean_rmse_ratio": 1e9,
            "logratio": 0.0,
            "tail_logratio": 0.0,
            "sd_logratio": 0.0,
            "cv_fold_wins": 0,
            "cv_gate_passed": False,
            "s1_fitness_zero_comp": 2.0,
            "s1_utility_zero_comp": -2.0,
            "unconstrained_utility": -2.0,
            "wall_ms": round(wall_ms, 3),
        }

    rmse0 = float(np.sqrt(np.mean((y0_te - pred0) ** 2)))
    rmse1 = float(np.sqrt(np.mean((y1_te - pred1) ** 2)))
    r0 = rmse0 / max(b0, 1e-12)
    r1 = rmse1 / max(b1, 1e-12)
    lr0 = math.log(max(r0, 1e-12))
    lr1 = math.log(max(r1, 1e-12))
    logratio = 0.5 * (lr0 + lr1)
    tail_logratio = max(lr0, lr1)
    sd_logratio = 0.5 * abs(lr0 - lr1)
    geomean_ratio = math.exp(logratio)
    wins = int(r0 < 1.0) + int(r1 < 1.0)
    gate = bool(geomean_ratio <= 0.90 and wins == 2)
    s1_fit = 0.9 * logratio + 0.2 * sd_logratio
    unconstrained_u = -s1_fit
    s1_u = unconstrained_u if gate else -2.0

    return {
        "cid": cid,
        "entrant": entrant,
        "status": "completed",
        "rmse_fold0": round(rmse0, 10),
        "rmse_fold1": round(rmse1, 10),
        "base_rmse_fold0": round(b0, 10),
        "base_rmse_fold1": round(b1, 10),
        "ratio_fold0": round(r0, 10),
        "ratio_fold1": round(r1, 10),
        "cv_geomean_rmse_ratio": round(geomean_ratio, 10),
        "logratio": round(logratio, 10),
        "tail_logratio": round(tail_logratio, 10),
        "sd_logratio": round(sd_logratio, 10),
        "cv_fold_wins": wins,
        "cv_gate_passed": gate,
        "s1_fitness_zero_comp": round(s1_fit, 10),
        "s1_utility_zero_comp": round(s1_u, 10),
        "unconstrained_utility": round(unconstrained_u, 10),
        "wall_ms": round(wall_ms, 3),
    }


def evaluate_ofm_and_tabfm_suite(
    jtt1_paths: List[Path],
    device: str,
    existing: Dict[Tuple[str, str], Dict[str, Any]],
    out_jsonl: Path,
) -> None:
    """Evaluates `TabFM`, `Kumo_Tabular_L`, `Kumo_Tabular_S`, `TabICL_v2`, `TabDPT_1p3`, and `Mitra_v2`."""
    sys.path.insert(0, str(OFM_ROOT / "src" / "structured-data-models"))
    sys.path.insert(0, str(OFM_ROOT / "src" / "TabDPT-inference" / "src"))
    sys.path.insert(0, str(OFM_ROOT / "src" / "tabicl" / "src"))
    sys.path.insert(0, str(OFM_ROOT / "src" / "mitra-finetune"))
    sys.path.insert(0, str(OFM_ROOT / "code"))

    import ofm_models as M  # noqa: E402
    from safetensors.torch import load_file  # noqa: E402
    from sdm.models.tabfm.ckpt import remap_ckpt  # noqa: E402
    from sdm.models.tabfm.model import _TabFM  # noqa: E402

    M.W = str(OFM_ROOT)

    # 1. TabFM & TabFM_Ensemble (`google/tabfm-1.0.0-pytorch`)
    entrant_tabfm = "TabFM"
    entrant_tabfm_ens = "TabFM_Ensemble"
    missing_tabfm = [p for p in jtt1_paths if (entrant_tabfm, p.stem) not in existing]
    missing_tabfm_ens = [p for p in jtt1_paths if (entrant_tabfm_ens, p.stem) not in existing]
    if missing_tabfm or missing_tabfm_ens:
        from sklearn.preprocessing import PowerTransformer  # noqa: E402

        ckpt_path = OFM_ROOT / "models" / "google__tabfm-1.0.0-pytorch" / "regression" / "model.safetensors"
        if not ckpt_path.exists():
            load_hf_token_if_present()
            from huggingface_hub import hf_hub_download  # noqa: E402

            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id="google/tabfm-1.0.0-pytorch",
                filename="regression/model.safetensors",
                local_dir=str(ckpt_path.parent.parent),
            )
        t_load = time.time()
        ckpt = remap_ckpt(load_file(str(ckpt_path), device=device), is_classifier=False)
        tabfm_mod = _TabFM(num_classes=0, device="meta")
        tabfm_mod.load_state_dict(ckpt, assign=True)
        tabfm_mod.eval()
        print(
            f"[TabFM] Loaded on {device} in {time.time() - t_load:.2f}s "
            f"(single={len(missing_tabfm)}, ens={len(missing_tabfm_ens)} contexts to run)",
            flush=True,
        )

        def _pred_tabfm(Xc: np.ndarray, yc: np.ndarray, Xq: np.ndarray) -> np.ndarray:
            Zc, Zq = M.ctx_standardize(Xc, Xq, impute=True, soft=4.0)
            mu = float(yc.mean())
            sd = float(yc.std()) + 1e-8
            yc_z = ((yc - mu) / sd).astype(np.float32)
            x = torch.from_numpy(np.concatenate([Zc, Zq], axis=0)).to(device)[None]
            y_t = torch.from_numpy(yc_z).to(device)[None]
            cm = torch.zeros(1, x.shape[-1], dtype=torch.bool, device=device)
            with torch.inference_mode():
                if device.startswith("cuda"):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = tabfm_mod(x, y_t, cm)
                else:
                    out = tabfm_mod(x, y_t, cm)
            pred_z = out.float()[0, :, 0].cpu().numpy()
            return pred_z * sd + mu

        def _pred_tabfm_ens4(Xc: np.ndarray, yc: np.ndarray, Xq: np.ndarray) -> np.ndarray:
            Zc, Zq = M.ctx_standardize(Xc, Xq, impute=True, soft=4.0)
            try:
                pt = PowerTransformer(standardize=True)
                Zc_pt = np.clip(pt.fit_transform(Zc), -4.0, 4.0).astype(np.float32)
                Zq_pt = np.clip(pt.transform(Zq), -4.0, 4.0).astype(np.float32)
                if not (np.all(np.isfinite(Zc_pt)) and np.all(np.isfinite(Zq_pt))):
                    Zc_pt, Zq_pt = Zc, Zq
            except Exception:
                Zc_pt, Zq_pt = Zc, Zq

            mu = float(yc.mean())
            sd = float(yc.std()) + 1e-8
            yc_z = ((yc - mu) / sd).astype(np.float32)
            rng = np.random.default_rng(1729)
            d = Zc.shape[1]
            batch_x = []
            for k, (A_c, A_q) in enumerate([(Zc, Zq), (Zc_pt, Zq_pt), (Zc, Zq), (Zc_pt, Zq_pt)]):
                perm = np.arange(d) if k == 0 else rng.permutation(d)
                batch_x.append(np.concatenate([A_c[:, perm], A_q[:, perm]], axis=0))
            x = torch.from_numpy(np.stack(batch_x, axis=0)).to(device)
            y_t = torch.from_numpy(yc_z).to(device).unsqueeze(0).expand(4, -1)
            cm = torch.zeros(4, d, dtype=torch.bool, device=device)
            with torch.inference_mode():
                if device.startswith("cuda"):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = tabfm_mod(x, y_t, cm)
                else:
                    out = tabfm_mod(x, y_t, cm)
            pred_z = out.float()[:, :, 0].mean(dim=0).cpu().numpy()
            return pred_z * sd + mu

        with out_jsonl.open("a") as f_out:
            for idx, p in enumerate(missing_tabfm, 1):
                cid = p.stem
                X0_tr, y0_tr, X0_te, y0_te, b0, b1 = parse_jtt1_2fold(p)
                t0 = time.perf_counter()
                pred0 = _pred_tabfm(X0_tr, y0_tr, X0_te)
                pred1 = _pred_tabfm(X0_te, y0_te, X0_tr)
                wall_ms = (time.perf_counter() - t0) * 1000.0
                rec = compute_s1_record(cid, entrant_tabfm, y0_te, pred0, b0, y0_tr, pred1, b1, wall_ms)
                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()
                existing[(entrant_tabfm, cid)] = rec
                if idx % 15 == 0 or idx == len(missing_tabfm):
                    print(
                        f"[{entrant_tabfm}] {idx}/{len(missing_tabfm)} | last={cid[:10]} "
                        f"s1_u={rec['s1_utility_zero_comp']:.4f} -lr={-rec['logratio']:.4f} ({wall_ms:.1f}ms)",
                        flush=True,
                    )
            for idx, p in enumerate(missing_tabfm_ens, 1):
                cid = p.stem
                X0_tr, y0_tr, X0_te, y0_te, b0, b1 = parse_jtt1_2fold(p)
                t0 = time.perf_counter()
                pred0 = _pred_tabfm_ens4(X0_tr, y0_tr, X0_te)
                pred1 = _pred_tabfm_ens4(X0_te, y0_te, X0_tr)
                wall_ms = (time.perf_counter() - t0) * 1000.0
                rec = compute_s1_record(cid, entrant_tabfm_ens, y0_te, pred0, b0, y0_tr, pred1, b1, wall_ms)
                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()
                existing[(entrant_tabfm_ens, cid)] = rec
                if idx % 15 == 0 or idx == len(missing_tabfm_ens):
                    print(
                        f"[{entrant_tabfm_ens}] {idx}/{len(missing_tabfm_ens)} | last={cid[:10]} "
                        f"s1_u={rec['s1_utility_zero_comp']:.4f} -lr={-rec['logratio']:.4f} ({wall_ms:.1f}ms)",
                        flush=True,
                    )
        del tabfm_mod, ckpt
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # 2. OFM Suite (`Kumo_Tabular_L`, `Kumo_Tabular_S`, `TabICL_v2`, `TabDPT_1p3`, `Mitra_v2`)
    ofm_specs = [
        ("Kumo_Tabular_L", "kumo_l"),
        ("Kumo_Tabular_S", "kumo_s"),
        ("TabICL_v2", "tabicl"),
        ("TabDPT_1p3", "tabdpt"),
        ("Mitra_v2", "mitra"),
    ]
    for entrant, build_key in ofm_specs:
        missing = [p for p in jtt1_paths if (entrant, p.stem) not in existing]
        if not missing:
            continue
        t_load = time.time()
        mod = M.build(build_key, dev=device)
        print(f"[{entrant}] Loaded on {device} in {time.time() - t_load:.2f}s ({len(missing)} contexts to run)", flush=True)

        def _pred_ofm(Xc: np.ndarray, yc: np.ndarray, Xq: np.ndarray) -> np.ndarray:
            mu = float(yc.mean())
            sd = float(yc.std()) + 1e-8
            yc_z = (yc - mu) / sd
            _, pred_z, _, _ = mod.run(Xc, yc_z, Xq)
            return np.asarray(pred_z, dtype=np.float64).reshape(-1) * sd + mu

        with out_jsonl.open("a") as f_out:
            for idx, p in enumerate(missing, 1):
                cid = p.stem
                X0_tr, y0_tr, X0_te, y0_te, b0, b1 = parse_jtt1_2fold(p)
                t0 = time.perf_counter()
                pred0 = _pred_ofm(X0_tr, y0_tr, X0_te)
                pred1 = _pred_ofm(X0_te, y0_te, X0_tr)
                wall_ms = (time.perf_counter() - t0) * 1000.0
                rec = compute_s1_record(cid, entrant, y0_te, pred0, b0, y0_tr, pred1, b1, wall_ms)
                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()
                existing[(entrant, cid)] = rec
                if idx % 15 == 0 or idx == len(missing):
                    print(
                        f"[{entrant}] {idx}/{len(missing)} | last={cid[:10]} "
                        f"s1_u={rec['s1_utility_zero_comp']:.4f} -lr={-rec['logratio']:.4f} ({wall_ms:.1f}ms)",
                        flush=True,
                    )
        del mod
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def evaluate_tabpfn35_suite(
    jtt1_paths: List[Path],
    device: str,
    existing: Dict[Tuple[str, str], Dict[str, Any]],
    out_jsonl: Path,
) -> None:
    """Evaluates `PriorLabs_TabPFN_3p5` (`n_estimators=8` ensemble and `n_estimators=1` single pass)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
    from tabpfn import TabPFNRegressor  # noqa: E402
    from tabpfn.constants import ModelVersion  # noqa: E402

    ckpt_path = str(TABPFN35_ROOT / "models" / "tabpfn-v3.5-20260909.safetensors")
    specs = [
        ("PriorLabs_TabPFN_3p5", 8),
        ("PriorLabs_TabPFN_3p5_Single", 1),
    ]
    for entrant, n_est in specs:
        missing = [p for p in jtt1_paths if (entrant, p.stem) not in existing]
        if not missing:
            continue
        print(f"[{entrant}] Running TabPFN v3.5 (n_estimators={n_est}) on {device} ({len(missing)} contexts)", flush=True)
        with out_jsonl.open("a") as f_out:
            for idx, p in enumerate(missing, 1):
                cid = p.stem
                X0_tr, y0_tr, X0_te, y0_te, b0, b1 = parse_jtt1_2fold(p)
                t0 = time.perf_counter()

                reg0 = TabPFNRegressor.create_default_for_version(
                    ModelVersion.V3_5,
                    model_path=ckpt_path,
                    device=device,
                    n_estimators=n_est,
                    random_state=1729,
                )
                reg0.fit(X0_tr.astype(np.float32), y0_tr)
                pred0 = np.asarray(reg0.predict(X0_te.astype(np.float32)), dtype=np.float64).reshape(-1)

                reg1 = TabPFNRegressor.create_default_for_version(
                    ModelVersion.V3_5,
                    model_path=ckpt_path,
                    device=device,
                    n_estimators=n_est,
                    random_state=1729,
                )
                reg1.fit(X0_te.astype(np.float32), y0_te)
                pred1 = np.asarray(reg1.predict(X0_tr.astype(np.float32)), dtype=np.float64).reshape(-1)

                wall_ms = (time.perf_counter() - t0) * 1000.0
                rec = compute_s1_record(cid, entrant, y0_te, pred0, b0, y0_tr, pred1, b1, wall_ms)
                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()
                existing[(entrant, cid)] = rec
                if idx % 15 == 0 or idx == len(missing):
                    print(
                        f"[{entrant}] {idx}/{len(missing)} | last={cid[:10]} "
                        f"s1_u={rec['s1_utility_zero_comp']:.4f} -lr={-rec['logratio']:.4f} ({wall_ms:.1f}ms)",
                        flush=True,
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["ofm_tabfm", "tabpfn35", "summarize"], default="ofm_tabfm")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--env-dir", type=Path, default=ENV_DIR)
    parser.add_argument("--out-jsonl", type=Path, default=OUT_JSONL)
    parser.add_argument("--out-cache-json", type=Path, default=OUT_CACHE_JSON)
    args = parser.parse_args()

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    jtt1_paths = sorted(args.env_dir.glob("*.jtt1"))
    assert len(jtt1_paths) == 91, f"Expected 91 JTT1 envelopes in {args.env_dir}, found {len(jtt1_paths)}"

    existing: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if args.out_jsonl.exists():
        for line in args.out_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            existing[(r["entrant"], r["cid"])] = r

    if args.suite == "ofm_tabfm":
        evaluate_ofm_and_tabfm_suite(jtt1_paths, args.device, existing, args.out_jsonl)
    elif args.suite == "tabpfn35":
        evaluate_tabpfn35_suite(jtt1_paths, args.device, existing, args.out_jsonl)

    # Export structured cache JSON grouped by entrant -> cid -> record
    by_ent: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (ent, cid), rec in sorted(existing.items()):
        by_ent.setdefault(ent, {})[cid] = rec
    args.out_cache_json.write_text(json.dumps(by_ent, indent=2) + "\n")

    print("\n=== Summary of 2-Fold Inner-CV (`jope.s1-two-fold/v1`) Tabular Foundation Models on DL-Val91 ===")
    for ent, cmap in sorted(by_ent.items()):
        n = len(cmap)
        n_gate = sum(1 for r in cmap.values() if r["cv_gate_passed"])
        mean_u_gated = float(np.mean([r["s1_utility_zero_comp"] for r in cmap.values()]))
        mean_u_unc = float(np.mean([r["unconstrained_utility"] for r in cmap.values()]))
        mean_neg_lr = float(np.mean([-r["logratio"] for r in cmap.values()]))
        print(
            f"{ent:30s} | n={n:2d}/91 | gate_pass={n_gate:2d}/91 "
            f"| mean_s1_u(gated)={mean_u_gated:.4f} | mean_s1_u(unc)={mean_u_unc:.4f} | mean(-lr)={mean_neg_lr:.4f}"
        )


if __name__ == "__main__":
    main()

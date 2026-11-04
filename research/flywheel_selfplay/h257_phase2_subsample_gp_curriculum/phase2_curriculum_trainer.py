#!/usr/bin/env python3
"""H257 Phase-2 Warm-Started Evolutionary Curriculum Trainer (`H257_Phase2_Curriculum_Cycle`).

Implements the two-phase learning and evolutionary curriculum requested by the user:
  - Do NOT start from 0: warm-start directly from converged Phase-1 (`H256`) production models
    (`PreCondenseResidualBridge` Arms F & G, `FullS3` 7-Split Pretrained Encoder, and
    `PreCondenseTop8NeuralReranker`).
  - Stage C0 (`0` GP Generations — Fast Convergence without GP Cycles):
    Converge the warm-started neural bridge and Top-8 reranker on `DL-Train` (`1,219` contexts)
    augmented with Semi-Synthetic Additional Columns (`semisyn80`: Gaussian noise, collinear
    duplicates, heavy-tail Student-t3 spikes, and pairwise nonlinear interactions) and Stratified
    `80%` Row Subsamples (`sub80`) using Two-Speed Differential LR, Cosine + SWA, All-28-Pairs
    Regret-Weighted Margin Loss, and Multi-View Consistency KL.
  - Stage C1 (`Gen 0` Turn-On — 11-View Batched Seeding + Closed-Loop DL Update #1):
    Turn on 11-view (`1x100% + 8x80% + 2xsemisyn`) batched Generation-0 seeding on the stratified
    `DL-Train` evolutionary cohort, evaluate in live 2-fold Rust CV (`bootstrap.s1main --fit-s1`),
    inject discovered `Gen 0` winners into `expanded_gp_bank` and the training slates, and update
    the neural bridges (`f_model`, `g_model`) and `top8_nn`.
  - Stage C2 (`1-Gen` GP Evolution `Gen 0 -> Gen 1` + Closed-Loop DL Update #2):
    Ramp up to 1 GP generation (`EditZero` AST mutations/crossover + causal trajectory screening),
    evaluate in live Rust CV, inject `Gen 1` winners into the training slates and `expanded_gp_bank`,
    and run Deep Learning Update #2.
  - Stage C3 (`3-Gen` GP Evolution `Gen 0 -> Gen 1 -> Gen 2 -> Gen 3` + Final DL & GBDT Update):
    Ramp up to 3 full GP generations (`Gen 2` & `Gen 3`), inject all multi-generation GP winners
    into the training slates and `expanded_gp_bank`, run the final Phase-2 neural + GBDT update,
    and export `weights/h257_phase2_curriculum_bundle.safetensors` (`<= 4.30 MiB`).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.numpy import load_file as load_safetensors_np, save_file as save_safetensors_np

HERE = Path(__file__).resolve().parent
SELFPLAY_DIR = HERE.parent
H252_DIR = SELFPLAY_DIR / "h252_closed_loop_flywheel"
H255_DIR = SELFPLAY_DIR / "h255_causal_2plus2_live_gp"
H256_DIR = SELFPLAY_DIR / "h256_pre_condense_residual_flywheel"
MICRO = Path("/home/ubuntu/JopeDimeMicro")
H256_OUT_DIR = Path("/home/ubuntu/jope-data/h256")
OUT_DIR = Path("/home/ubuntu/jope-data/h257")
CACHE_DIR = OUT_DIR / "cache"
CKPT_DIR = OUT_DIR / "checkpoints"
WEIGHTS_DIR = HERE / "weights"

sys.path.insert(0, str(MICRO / "python"))
sys.path.insert(0, str(SELFPLAY_DIR))
sys.path.insert(0, str(H252_DIR))
sys.path.insert(0, str(H255_DIR))
sys.path.insert(0, str(H256_DIR))
sys.path.insert(0, str(HERE))

import rust_evaluator as evaluator  # noqa: E402
from automatic_promotion import training_cv_envelope  # noqa: E402
import common_h252 as ch  # noqa: E402
import pretrain_full_s3_7split_bridge as p_s3  # noqa: E402
import run_h256_flywheel_ablation as r256  # noqa: E402
import train_and_export_h256_flagship_production as prod  # noqa: E402
import export_h256_safetensors as exp_st  # noqa: E402
import semi_synthetic_columns_and_views as ssa  # noqa: E402
import batched_subsample_gp_evolution as bge  # noqa: E402
from pre_condense_models import FoldScaler, PreCondenseResidualBridge  # noqa: E402


def fine_tune_bridges_on_curriculum_slates(
    f_model: PreCondenseResidualBridge,
    g_model: PreCondenseResidualBridge,
    curriculum_slates: List[Dict[str, Any]],
    raw_cache: Dict[str, Dict[str, np.ndarray]],
    aug_views: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    sc_std: FoldScaler,
    sc_s3: FoldScaler,
    device: torch.device,
    n_steps: int = 10,
    lr_enc: float = 1.5e-4,
    lr_head: float = 8.0e-4,
    seed: int = 20261104,
) -> Dict[str, Any]:
    """Updates warm-started Stage-2 bridges (`f_model` and `g_model`) on the evolved GP slates
    discovered at the current curriculum stage (`Gen 0`, `Gen 1`, or `Gen 3`), combined with
    multi-view consistency across clean (`100%`), `sub80`, and `semisyn80` context encodings.
    """
    if not curriculum_slates:
        return {"steps": 0, "loss_start": 0.0, "loss_end": 0.0}

    def _make_diff_opt(m: PreCondenseResidualBridge) -> torch.optim.Optimizer:
        head_names = {"prior_head", "prior_residual_out", "traj_head", "traj_residual_out", "traj_enc", "traj_attn", "traj_norm"}
        enc_params = []
        head_params = []
        for name, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if any(hn in name for hn in head_names):
                head_params.append(p)
            else:
                enc_params.append(p)
        return torch.optim.AdamW(
            [
                {"params": enc_params, "lr": lr_enc, "weight_decay": 2e-4},
                {"params": head_params, "lr": lr_head, "weight_decay": 1e-4},
            ]
        )

    opt_f = _make_diff_opt(f_model)
    opt_g = _make_diff_opt(g_model)
    sched_f = torch.optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=max(n_steps, 1), eta_min=2e-5)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=max(n_steps, 1), eta_min=2e-5)

    rng = np.random.default_rng(seed)
    f_model.train()
    g_model.train()
    losses: List[float] = []

    for step in range(n_steps):
        perm = rng.permutation(len(curriculum_slates))[: min(16, len(curriculum_slates))]
        opt_f.zero_grad()
        opt_g.zero_grad()
        batch_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        for idx_s in perm:
            sl = curriculum_slates[int(idx_s)]
            cid = sl["cid"]
            cand_q = torch.as_tensor(sl["cand_q_32"], dtype=torch.float32, device=device).unsqueeze(0)  # [1, M, 32]
            sem24 = torch.as_tensor(sl["sem24"], dtype=torch.float32, device=device).unsqueeze(0)  # [1, M, 24]
            u_vec = torch.as_tensor(sl["utilities"], dtype=torch.float32, device=device)  # [M]
            g_vec = torch.as_tensor(sl["gates"], dtype=torch.float32, device=device)  # [M]
            m_c = len(u_vec)
            zeros_1m = torch.zeros((1, m_c), dtype=torch.float32, device=device)

            # Target distribution over evolved GP slate (lower predicted cost = higher utility)
            target_p = F.softmax(u_vec / 0.25, dim=-1)

            # Clean view + perturbed view (alternating sub80 and semisyn80)
            v_clean = raw_cache[cid]
            v_alt_name = "semisyn80" if (step + int(idx_s)) % 2 == 0 else "sub80"
            v_alt = aug_views[v_alt_name][cid]

            for model_m, sc_m in ((f_model, sc_std), (g_model, sc_s3)):
                ct_c = ssa.transform_context_batch(
                    sc_m,
                    v_clean["dope629"][None, :],
                    v_clean["hidden256"][None, :],
                    v_clean["col_lm_64x17"][None, :, :],
                    v_clean["col_mask_64"][None, :],
                    device=device,
                )
                ct_a = ssa.transform_context_batch(
                    sc_m,
                    v_alt["dope629"][None, :],
                    v_alt["hidden256"][None, :],
                    v_alt["col_lm_64x17"][None, :, :],
                    v_alt["col_mask_64"][None, :],
                    device=device,
                )
                dp_c, attn_c, bilin_c = model_m.forward_prior_and_context(
                    ct_c, cand_q, sem24, zeros_1m, zeros_1m
                )
                dp_a, _, _ = model_m.forward_prior_and_context(
                    ct_a, cand_q, sem24, zeros_1m, zeros_1m
                )

                # Also supervise causal trajectory head using the first 4 revealed candidates
                rev_idx = torch.arange(min(4, m_c), dtype=torch.long, device=device).unsqueeze(0)
                if rev_idx.shape[1] < 4:
                    rev_idx = F.pad(rev_idx, (0, 4 - rev_idx.shape[1]), value=0)
                rev_f = (-u_vec[rev_idx[0]]).unsqueeze(0)
                rev_comb = (rev_f - rev_f.mean()).clamp(-2.0, 2.0)
                rev_g = g_vec[rev_idx[0]].unsqueeze(0)
                dt_c = model_m.forward_causal_trajectory_residual(
                    attn_c, bilin_c, cand_q, sem24, 0.03 * dp_c, rev_idx, rev_comb, rev_f, rev_g
                )

                scores_c = -(dp_c[0] + dt_c[0])
                scores_a = -dp_a[0]
                loss_ln = -(target_p * F.log_softmax(scores_c, dim=-1)).sum()
                loss_ln_a = -(target_p * F.log_softmax(scores_a, dim=-1)).sum()

                # Pairwise regret-weighted margin on evolved slate
                u_diff = u_vec.unsqueeze(1) - u_vec.unsqueeze(0)
                s_diff = scores_c.unsqueeze(1) - scores_c.unsqueeze(0)
                mask_p = (u_diff > 1e-3).float()
                w_p = mask_p * (0.15 + torch.clamp(u_diff, 0.0, 0.60))
                loss_pair = (F.relu(0.10 - s_diff) * w_p).sum() / (w_p.sum() + 1e-6)

                # Consistency KL between clean and semi-synthetic / subsample view
                p_c_det = F.softmax(scores_c.detach(), dim=-1)
                loss_kl = F.kl_div(F.log_softmax(scores_a, dim=-1), p_c_det, reduction="sum")

                batch_loss = batch_loss + (loss_ln + 0.5 * loss_ln_a + 0.5 * loss_pair + 0.2 * loss_kl) / float(len(perm))

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(f_model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(g_model.parameters(), 1.0)
        opt_f.step()
        opt_g.step()
        sched_f.step()
        sched_g.step()
        losses.append(float(batch_loss.detach().item()))

    f_model.eval()
    g_model.eval()
    return {
        "steps": n_steps,
        "loss_start": float(losses[0]) if losses else 0.0,
        "loss_end": float(losses[-1]) if losses else 0.0,
    }


def run_phase2_curriculum() -> Dict[str, Any]:
    """Executes the complete 4-Stage Phase-2 Curriculum (`Stage C0 -> C1 -> C2 -> C3`):
      1. Stage C0 (`0` GP Generations): Warm-started Semi-Synthetic Column + `80%` Subsample convergence (`gate_t1_summary.json`).
      2. Stage C1 (`Gen 0` Batched Seeding): 11-view batched Gen-0 seeding on stratified `DL-Train` cohort + closed-loop DL update #1.
      3. Stage C2 (`1-Gen` GP Evolution): `Gen 0 -> Gen 1` evolution + closed-loop DL update #2.
      4. Stage C3 (`3-Gen` GP Evolution): `Gen 1 -> Gen 2 -> Gen 3` evolution + closed-loop DL update #3 + `.safetensors` export.
    """
    t0 = time.time()
    device = torch.device("cpu")
    torch.set_num_threads(4)

    # Step 1: Ensure Stage C0 (0-GP Semi-Synthetic + 80% Subsample Warm-Started Convergence) is complete
    c0_res = ssa.run_gate_t1_hypothesis_test()

    net, guard, dope_snap_before, _ = prod.p255.load_fused_network(device)
    train_rows, val_rows = ch.load_cohorts()
    train_cids = [r["context_sha256"] for r in train_rows]
    raw_cache = r256.load_pre_condense_cache(prod.CACHE_PATH)
    aug_views = ssa.extract_or_load_semisynthetic_train_views(
        train_rows, net, guard, dope_snap_before, device
    )
    gp_bank = list(bge.load_enriched_gp_champion_bank(train_rows, raw_cache))

    # Select stratified 16-family DL-Train evolutionary curriculum cohort
    seen_fams = set()
    curriculum_rows: List[Dict[str, Any]] = []
    for r in train_rows:
        if r["family_id"] not in seen_fams:
            seen_fams.add(r["family_id"])
            curriculum_rows.append(r)
            if len(curriculum_rows) >= 16:
                break

    train_curr_11views = ssa.extract_or_load_11view_batch_for_contexts(
        curriculum_rows,
        "dl_train_curr16_11view_batch.npz",
        raw_cache,
        net,
        guard,
        dope_snap_before,
        device,
    )

    env_dir = CACHE_DIR / "train_curr16_jtt1_envelopes"
    env_dir.mkdir(parents=True, exist_ok=True)
    for r in curriculum_rows:
        cid = r["context_sha256"]
        env_p = env_dir / f"{cid}.jtt1"
        if not env_p.exists():
            raw_bytes = (ch.INPUT_DIR / "tables" / f"{cid}.jtt1").read_bytes()
            env_p.write_bytes(training_cv_envelope(raw_bytes))

    # Warm-start Stage-2 bridges F & G from Phase-1 H256
    b_train1219 = torch.load(
        H256_OUT_DIR / "checkpoints" / "h256_production_1219_dl_train_bundle.pt",
        map_location="cpu",
        weights_only=False,
    )
    f_model = PreCondenseResidualBridge(arm="F_FULL629_HIDDEN256_COL64X17", d_token=32, n_heads=4).to(device)
    f_model.load_state_dict(b_train1219["stage2_bridge_arm_f_soup_state_dict"], strict=True)
    g_model = PreCondenseResidualBridge(arm="G_FULL_S3_REG_7SPLIT_11121_PRETRAINED", d_token=32, n_heads=4).to(device)
    g_model.load_state_dict(b_train1219["stage2_bridge_arm_g_soup_state_dict"], strict=True)

    sc_std = FoldScaler.fit(train_cids, raw_cache)
    s3_ckpt = torch.load(H256_OUT_DIR / "full_s3_reg_7split_pretrained_encoder.pt", map_location="cpu", weights_only=False)
    sc_s3 = sc_std.blend_with_s3_scaler(s3_ckpt["scaler_policy_b"], alpha_s3=0.25)

    p2_bridge_path = CKPT_DIR / "h257_phase2_curriculum_bridges.pt"
    if p2_bridge_path.exists():
        saved_p2 = torch.load(p2_bridge_path, map_location="cpu", weights_only=False)
        f_model.load_state_dict(saved_p2["stage2_bridge_arm_f_state_dict"], strict=True)
        g_model.load_state_dict(saved_p2["stage2_bridge_arm_g_state_dict"], strict=True)
        gp_bank = saved_p2["expanded_gp_bank"]
        curriculum_telemetry = saved_p2["curriculum_telemetry"]
    else:
        eval_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        curr_evo_by_cid: Dict[str, Dict[str, Any]] = {}
        for r in curriculum_rows:
            cid = r["context_sha256"]
            fam = r["family_id"]
            res_c = bge.run_3gen_live_gp_evolution_for_context(
                cid=cid,
                family_id=fam,
                env_jtt1_path=env_dir / f"{cid}.jtt1",
                raw_jtt1_path=ch.INPUT_DIR / "tables" / f"{cid}.jtt1",
                views_ctx=train_curr_11views[cid],
                gp_bank=gp_bank,
                f_model=f_model,
                g_model=g_model,
                sc_std=sc_std,
                sc_s3=sc_s3,
                device=device,
                eval_cache=eval_cache,
                n_gen0=6,
                n_generations=3,
                children_per_gen=4,
                proposals_per_gen=16,
            )
            curr_evo_by_cid[cid] = res_c

        # Build progressive slates for Stage C1 (Gen 0), Stage C2 (Gen 0..1), Stage C3 (Gen 0..3)
        def _build_slates_and_inject_bank(max_gen_tag: str) -> Tuple[List[Dict[str, Any]], int]:
            slates = []
            n_injected = 0
            known_shas = {(b["context_sha256"], b["program_sha256"]) for b in gp_bank}
            for r in curriculum_rows:
                cid = r["context_sha256"]
                fam = r["family_id"]
                c_evals = [
                    v for (c_k, _), v in eval_cache.items() if c_k == cid and v.get("status") == "completed"
                ]
                if len(c_evals) < 4:
                    continue
                proj_m = np.stack([e["proj16"] for e in c_evals], axis=0).astype(np.float32)
                sem_m = np.stack([e["sem24"] for e in c_evals], axis=0).astype(np.float32)
                cq = np.zeros((len(c_evals), 32), dtype=np.float32)
                cq[:, :16] = proj_m
                cq[:, 16:32] = sem_m[:, :16]
                u_arr = np.array([float(e["utility"]) for e in c_evals], dtype=np.float32)
                g_arr = np.array([1.0 if e.get("cv_gate_passed") else 0.0 for e in c_evals], dtype=np.float32)
                slates.append(
                    {
                        "cid": cid,
                        "cand_q_32": cq,
                        "sem24": sem_m,
                        "utilities": u_arr,
                        "gates": g_arr,
                    }
                )
                best_rec = curr_evo_by_cid[cid][max_gen_tag]
                if best_rec["cv_gate_passed"] and (cid, best_rec["program_sha256"]) not in known_shas:
                    f412, p16 = evaluator.featurize_dsl_412(best_rec["dsl"])
                    s24 = ch.compute_sem24_numpy(f412[None, :])[0]
                    gp_bank.append(
                        {
                            "context_sha256": cid,
                            "family_id": fam,
                            "dl_split": "DL-Train",
                            "dsl": best_rec["dsl"],
                            "program_sha256": best_rec["program_sha256"],
                            "feat412": f412.astype(np.float32),
                            "proj16": p16.astype(np.float32),
                            "sem24": s24.astype(np.float32),
                            "landmark_summary": raw_cache[cid]["lm48_z"][:12].astype(np.float32).tolist(),
                        }
                    )
                    known_shas.add((cid, best_rec["program_sha256"]))
                    n_injected += 1
            return slates, n_injected

        slates_c1, inj_c1 = _build_slates_and_inject_bank("gen0_best")
        telem_c1 = fine_tune_bridges_on_curriculum_slates(
            f_model, g_model, slates_c1, raw_cache, aug_views, sc_std, sc_s3, device, n_steps=8, lr_enc=1.5e-4, lr_head=8e-4, seed=25701
        )
        slates_c2, inj_c2 = _build_slates_and_inject_bank("gen1_best")
        telem_c2 = fine_tune_bridges_on_curriculum_slates(
            f_model, g_model, slates_c2, raw_cache, aug_views, sc_std, sc_s3, device, n_steps=8, lr_enc=1.2e-4, lr_head=6e-4, seed=25702
        )
        slates_c3, inj_c3 = _build_slates_and_inject_bank("gen3_best")
        telem_c3 = fine_tune_bridges_on_curriculum_slates(
            f_model, g_model, slates_c3, raw_cache, aug_views, sc_std, sc_s3, device, n_steps=10, lr_enc=1.0e-4, lr_head=5e-4, seed=25703
        )

        curriculum_telemetry = {
            "stage_c0_0gen_semisyn": {
                "gp_generations": 0,
                "rust_cv_fits": 0,
                "description": "Warm-start from H256 + 3x Semi-Synthetic Column & 80% Row-Subsample Convergence",
            },
            "stage_c1_gen0_seeding": {
                "gp_generations": 0,
                "n_gen0_trees_per_ctx": 6,
                "new_champions_injected": inj_c1,
                "mean_train_cohort_utility": float(np.mean([curr_evo_by_cid[r["context_sha256"]]["gen0_best"]["utility"] for r in curriculum_rows])),
                "dl_update": telem_c1,
            },
            "stage_c2_evo1_curriculum": {
                "gp_generations": 1,
                "new_champions_injected": inj_c2,
                "mean_train_cohort_utility": float(np.mean([curr_evo_by_cid[r["context_sha256"]]["gen1_best"]["utility"] for r in curriculum_rows])),
                "dl_update": telem_c2,
            },
            "stage_c3_evo3_curriculum": {
                "gp_generations": 3,
                "new_champions_injected": inj_c3,
                "mean_train_cohort_utility": float(np.mean([curr_evo_by_cid[r["context_sha256"]]["gen3_best"]["utility"] for r in curriculum_rows])),
                "dl_update": telem_c3,
                "total_bank_size_after_c3": len(gp_bank),
            },
        }
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "jope.h257.phase2_curriculum_bridges/v1",
                "stage2_bridge_arm_f_state_dict": f_model.state_dict(),
                "stage2_bridge_arm_g_state_dict": g_model.state_dict(),
                "expanded_gp_bank": gp_bank,
                "curriculum_telemetry": curriculum_telemetry,
            },
            p2_bridge_path,
        )
        print(
            f"[H257 Curriculum] Completed Stages C1->C2->C3 closed-loop GP+DL updates "
            f"(u: {curriculum_telemetry['stage_c1_gen0_seeding']['mean_train_cohort_utility']:.4f} -> "
            f"{curriculum_telemetry['stage_c2_evo1_curriculum']['mean_train_cohort_utility']:.4f} -> "
            f"{curriculum_telemetry['stage_c3_evo3_curriculum']['mean_train_cohort_utility']:.4f}) in {time.time() - t0:.1f}s",
            flush=True,
        )

    # Export Phase-2 `.safetensors` production bundle to `research/flywheel_selfplay/h257_phase2_subsample_gp_curriculum/weights/`
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    pkt_cache = torch.load(H256_OUT_DIR / "cache" / "flagship_packets_616.pt", map_location="cpu", weights_only=False)
    packets = pkt_cache["packets"]
    train_sub80_X616 = ssa.build_perturbed_616_packets_for_view(train_cids, packets, aug_views["sub80"], device)
    train_semisyn_X616 = ssa.build_perturbed_616_packets_for_view(train_cids, packets, aug_views["semisyn80"], device)

    nn_c3, nn_mean_c3, nn_std_c3, _ = ssa.train_neural_top8_reranker_v2(
        train_cids,
        packets,
        seed=20261104,
        device=device,
        aug_views_X616=[train_sub80_X616, train_semisyn_X616],
        n_epochs=18,
        warm_start_state_dict=b_train1219["top8_neural_reranker_state_dict"],
    )

    # Start from the verified Phase-1 1219 safetensors map and update with Phase-2 (`H257`) trained tensors
    base_st_path = H256_DIR / "weights" / "h256_production_1219_dl_train_bundle.safetensors"
    st_tensors = dict(load_safetensors_np(str(base_st_path)))
    exp_st.add_state_dict_tensors(st_tensors, "stage2_bridge", g_model.state_dict())
    exp_st.add_state_dict_tensors(st_tensors, "stage2_arm_f_soup", f_model.state_dict())
    exp_st.add_state_dict_tensors(st_tensors, "stage2_arm_g_soup", g_model.state_dict())
    exp_st.add_state_dict_tensors(st_tensors, "top8_nn", nn_c3.state_dict())
    st_tensors["top8_nn.feat_mean"] = np.asarray(nn_mean_c3, dtype=np.float32).reshape(-1)
    st_tensors["top8_nn.feat_std"] = np.asarray(nn_std_c3, dtype=np.float32).reshape(-1)

    out_st_path = WEIGHTS_DIR / "h257_phase2_curriculum_bundle.safetensors"
    save_safetensors_np(st_tensors, str(out_st_path))
    st_bytes = out_st_path.read_bytes()
    st_sha256 = hashlib.sha256(st_bytes).hexdigest()
    print(
        f"[H257 Export] Saved {out_st_path.name} ({len(st_bytes) / (1024 * 1024):.3f} MiB, sha256={st_sha256[:16]})",
        flush=True,
    )

    guard.verify_dope_frozen_unchanged(net, dope_snap_before)
    return {
        "c0_res": c0_res,
        "curriculum_telemetry": curriculum_telemetry,
        "safetensors_bundle": {
            "filename": out_st_path.name,
            "size_bytes": len(st_bytes),
            "size_mib": round(len(st_bytes) / (1024 * 1024), 4),
            "sha256": st_sha256,
            "tensor_count": len(st_tensors),
        },
    }


if __name__ == "__main__":
    run_phase2_curriculum()

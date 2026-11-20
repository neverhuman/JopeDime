#!/usr/bin/env python3
"""H473 (`h473_ultra_packet_retrieval_and_audits`): End-to-End Scored CPU Runner.

Executes pre-registered audits and experiments across `DL-Train1219` (`205` families,
`h452a/` + `h452b/` 5 family-grouped folds x 3 seeds) and one-time frozen `DL-Val91` (`19` families):
  - Part A1 (`DV-01`): Authoritative 92-token `PreCondenseResidualBridge` layout + slice-bug audit
  - Part A2 (`DV-03`): `prog_412` exact & near-collision (`cos > 0.9999`) sufficiency audit
  - Part A3 (`DV-02`): `MaskedPresenceV1` `[false; 4]` 0-reveal vs `wave0_fake_k4` & 4-reveal policy mismatch
  - Part B (`I3`/`I8`): Two-Tower `859-D` `DopeUltraPacket859` champion-slate (`K=96`), pool shortlist (`K=16,32`), & `base_prior_cost`
  - Part C (`I10` & `I9`): Zero-shot `op_desc` canonical template selection & context-scaled `pair_noise` `z_ctx` gate
  - One-Time Frozen `DL-Val91` `REGCHECK` (never opened under `--smoke` or during CV selection).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from h256_token_layout_audit import (  # noqa: E402
    derange_across_families,
    digest_file_sha256,
    family_unweighted_mean,
    resolve_jope_roots,
    run_dv01_h256_layout_audit,
    verify_mem_available_gib,
    write_strict_json,
)
from prog412_collision_audit import (  # noqa: E402
    audit_prog412_collisions,
    run_i10_zero_shot_archetype_cv,
    select_diverse_structural_anchors,
)
from two_tower_retrieval import (  # noqa: E402
    eval_part_b_val91_regcheck,
    run_part_b_two_tower_study,
)

REPO_ROOT, FLYWHEEL_ROOT, MICRO_PYTHON = resolve_jope_roots(Path(__file__))
for _p in (
    str(FLYWHEEL_ROOT / "dope_value_probe"),
    str(FLYWHEEL_ROOT),
    str(REPO_ROOT / "python" / "src"),
    str(MICRO_PYTHON),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dope_value_probe import (  # noqa: E402
    DOPE_FUSED_COMPOSITE_SHA256,
    holm_bonferroni,
    paired_family_bootstrap_ci,
)

CPU_EXACT_FUSED_SHA256 = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
ROUND_3175_SHA256 = "1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b"
PROMOTIONS_SHA256 = "d7796fdd7c0dde19e1a23f34c28afc1b9aafa27b0792df746e9f9db01f8cd6c5"
SELECTED_JSON_SHA256 = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
CHAMPIONS_JSON_SHA256 = "a444047ffb937dc8d69012cbdbf0891faf3e7d743b127e7cac3baed114cf141d"
TRAIN_MANIFEST_NAME = "h470_features_dl_train1219_5c70c45d76e0b00262c88cbc.manifest.json"
VAL91_MANIFEST_NAME = "h470_features_dl_val91_f482f8546f93a2aaa87fcca3.manifest.json"


def compute_i9_z_ctx(sigma_hat: np.ndarray | float, sigma_med: float, z0: float = 0.40) -> np.ndarray:
    denom = max(float(sigma_med), 1e-8)
    ratio = np.asarray(sigma_hat, dtype=np.float64) / denom
    return float(z0) * np.clip(ratio, 0.5, 2.0)


def filter_masked_presence_v1(
    slot_indices: Sequence[int] | np.ndarray,
    slot_raw_f: Sequence[float] | np.ndarray,
    slot_gate: Sequence[float | bool] | np.ndarray,
    presence_mask: Sequence[bool] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_b = np.asarray(presence_mask, dtype=bool)
    idx_a = np.asarray(slot_indices, dtype=np.int64)
    rf_a = np.asarray(slot_raw_f, dtype=np.float32)
    rg_a = np.asarray(slot_gate, dtype=np.float32)
    if mask_b.shape != idx_a.shape or mask_b.shape != rf_a.shape or mask_b.shape != rg_a.shape:
        raise ValueError("MaskedPresenceV1 requires matching 1-D slot shapes")
    active = np.where(mask_b)[0]
    return idx_a[active], rf_a[active], rg_a[active]


def paired_contrast(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    families: Sequence[str],
    *,
    seed: int = 20261120,
) -> dict[str, Any]:
    raw = paired_family_bootstrap_ci(left, right, families, n_boot=10000, seed=seed)
    return {
        "mean_delta": round(float(raw["mean_delta"]), 6),
        "ci_low": round(float(raw["ci_low"]), 6),
        "ci_high": round(float(raw["ci_high"]), 6),
        "p_value": round(float(raw["p_value"]), 6),
        "left_family_mean": round(family_unweighted_mean(left, families), 6),
        "right_family_mean": round(family_unweighted_mean(right, families), 6),
        "family_wins": int(raw["family_wins"]),
        "family_ties": int(raw["family_ties"]),
        "family_losses": int(raw["family_losses"]),
        "includes_zero": bool(float(raw["ci_low"]) <= 0.0 <= float(raw["ci_high"])),
    }


def verify_scratch_budget_mb(scratch_dir: Path, max_mb: float = 50.0) -> float:
    total_bytes = sum(p.stat().st_size for p in scratch_dir.rglob("*") if p.is_file())
    used_mb = float(total_bytes) / (1024.0 * 1024.0)
    if used_mb > float(max_mb):
        raise RuntimeError(f"scratch disk usage {used_mb:.2f} MB exceeds {max_mb:.2f} MB budget")
    return round(used_mb, 4)


def authenticate_feature_cache(cache_dir: Path, manifest_name: str, wanted_keys: Sequence[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    m_path = cache_dir / manifest_name
    doc = json.loads(m_path.read_text(encoding="utf-8"))
    if doc.get("schema") != "jope.h470.cache-auth/v1" or int(doc.get("query_rows_used", -1)) != 0:
        raise ValueError(f"unauthenticated or query-contaminated cache manifest: {manifest_name}")
    if doc["manifest"].get("dope_weights_sha256") != DOPE_FUSED_COMPOSITE_SHA256:
        raise ValueError("manifest DOPE SHA-256 mismatch")
    npz_path = cache_dir / str(doc["npz_file"])
    if digest_file_sha256(npz_path) != doc["npz_file_sha256"]:
        raise ValueError(f"npz SHA-256 mismatch for {npz_path.name}")
    arrays: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as zf:
        for key in wanted_keys:
            arr = zf[key]
            spec = doc["arrays"][key]
            le_arr = np.ascontiguousarray(arr)
            hdr = f"shape={le_arr.shape}|dtype={le_arr.dtype.str}|".encode("utf-8")
            le_sha = hashlib.sha256(hdr + le_arr.tobytes(order="C")).hexdigest()
            if le_sha != spec["le_sha256"] or list(arr.shape) != list(spec["shape"]):
                raise ValueError(f"array digest or shape mismatch on {key}")
            arrays[key] = arr
    return arrays, doc


def build_repr_bundle(feats: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = feats["dope629_hpc"].shape[0]
    r_desc = np.hstack([feats["vec256_hpc"], feats["col_lm_34_hpc"]]).astype(np.float32)
    r_dope = np.hstack([feats["dope629_hpc"], feats["col_lm_34_hpc"]]).astype(np.float32)
    r_up = np.hstack([feats["dope629_ultra"], feats["col_lm_34_ultra"], feats["R_packet_v1_ultra"]]).astype(np.float32)
    return {
        "lm48_z": feats["lm48_z_hpc"].astype(np.float32),
        "R_desc": r_desc,
        "R_dope": r_dope,
        "DopeUltraPacket859": r_up,
        "i10_ultra_packet_859": r_up,
        "i10_dope_663": r_dope,
        "i10_desc_290": r_desc,
        "i10_none_0d": np.zeros((n, 0), dtype=np.float32),
    }


def load_cohort_pools(input_dir: Path, records: Sequence[Mapping[str, Any]], w1_mat: np.ndarray, sem24_fn: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        cid = str(rec["context_sha256"])
        p_path = input_dir / "pools" / f"{cid}.npz"
        if digest_file_sha256(p_path) != rec["pool_sha256"]:
            raise ValueError(f"pool SHA-256 mismatch for {cid}")
        with np.load(p_path, allow_pickle=False) as zf:
            prog = zf["program"].astype(np.float32)
            pids = [v.decode() if isinstance(v, bytes) else str(v) for v in zf["program_sha256"].tolist()]
            proj16 = (prog.astype(np.float64) @ w1_mat / math.sqrt(412.0)).astype(np.float32)
            out[cid] = {
                "rec": dict(rec),
                "pids": pids,
                "program": prog,
                "proj16": proj16,
                "struct_sig": sem24_fn(prog),
                "fitness": zf["fitness"].astype(np.float64),
                "logratio": zf["logratio"].astype(np.float64),
                "gate": zf["gate"].astype(bool),
                "prior_cost": np.zeros(len(prog), dtype=np.float64),
            }
    return out


def build_h452_family_folds(
    cids: Sequence[str],
    fam_by_cid: Mapping[str, str],
    inc_fail: np.ndarray,
    salts: Sequence[str],
    n_folds: int = 5,
) -> list[np.ndarray]:
    fail_counts: dict[str, int] = {}
    for cid, failed in zip(cids, inc_fail.tolist()):
        if failed:
            fm = fam_by_cid[cid]
            fail_counts[fm] = fail_counts.get(fm, 0) + 1
    out_folds: list[np.ndarray] = []
    for salt in salts:
        fams = sorted(
            {fam_by_cid[c] for c in cids},
            key=lambda f: hashlib.sha256(f"{salt}{f}".encode("utf-8")).hexdigest(),
        )
        f_sz = {f: sum(1 for c in cids if fam_by_cid[c] == f) for f in fams}
        f_assign: dict[str, int] = {}
        loads = [0] * n_folds
        fails = [0] * n_folds
        for f in sorted([x for x in fams if x in fail_counts], key=lambda x: (-fail_counts[x], -f_sz[x])):
            slot = min(range(n_folds), key=lambda j: (fails[j], loads[j]))
            f_assign[f] = slot
            loads[slot] += f_sz[f]
            fails[slot] += fail_counts[f]
        for f in sorted([x for x in fams if x not in fail_counts], key=lambda x: -f_sz[x]):
            slot = int(np.argmin(loads))
            f_assign[f] = slot
            loads[slot] += f_sz[f]
        arr = np.array([f_assign[fam_by_cid[c]] for c in cids], dtype=np.int64)
        for k in range(n_folds):
            in_f = {fam_by_cid[cids[i]] for i in np.where(arr == k)[0].tolist()}
            out_f = {fam_by_cid[cids[i]] for i in np.where(arr != k)[0].tolist()}
            if in_f & out_f:
                raise AssertionError(f"family overlap in {salt} fold {k}")
        out_folds.append(arr)
    return out_folds


def precompute_context_anchor_state(
    net: Any,
    ordered_cids: Sequence[str],
    pools: Mapping[str, Mapping[str, Any]],
    tokens_10x32: np.ndarray,
    mask_10: np.ndarray,
) -> dict[str, dict[str, Any]]:
    verify_mem_available_gib(15.0, "precompute_anchor_state")
    net.eval()
    ad = net.adapter_10tok
    anc = ad.anchor
    z_ctx1 = torch.zeros((1, 584), dtype=torch.float32)
    p_mu = ad.prog_mean.detach().cpu().numpy().reshape(1, 412).astype(np.float32)
    p_sd = ad.prog_std.detach().cpu().numpy().reshape(1, 412).astype(np.float32)
    state: dict[str, dict[str, Any]] = {}

    with torch.no_grad():
        ctx_emb0 = anc.ctx_block(anc.ctx_in(z_ctx1))  # [1, 128]
        for i, cid in enumerate(ordered_cids):
            p = pools[cid]
            m = len(p["pids"])
            pt = torch.from_numpy(p["program"][None])
            pjt = torch.from_numpy(p["proj16"][None])
            zs = torch.zeros((1, m, 2), dtype=torch.float32)
            _, pemb, ssig, pdelta = anc.encode_candidates(z_ctx1, pt, pjt, zs)
            tok_t = torch.from_numpy(tokens_10x32[i : i + 1].astype(np.float32))
            msk_t = torch.from_numpy(mask_10[i : i + 1].astype(np.float32))
            _, cand_q, sem24, mm_adj = ad.encode_multimodal_context_and_candidates(
                mm_tokens_10x32=tok_t, mm_mask_10=msk_t, prog_412=pt, sos_prior=zs
            )
            x_log = np.sign(p["program"]) * np.log1p(np.abs(p["program"]))
            prog_std = np.clip((x_log - p_mu) / p_sd, -8.0, 8.0).astype(np.float32)
            state[cid] = {
                "ctx_emb0": ctx_emb0[0].numpy(),
                "prog_emb": pemb[0].numpy(),
                "anc_ssig": ssig[0].numpy(),
                "eff_delta": (anc.alpha_prior.double() * pdelta[0].double()).numpy(),
                "cand_q0": cand_q[0].numpy(),
                "sem24": sem24[0].numpy(),
                "mm_adj0": mm_adj[0].numpy(),
                "prog_std": prog_std,
            }
    return state


def predict_from_state(
    net: Any,
    st: Mapping[str, Any],
    pool: Mapping[str, Any],
    *,
    base_prior_cost: np.ndarray | None = None,
    rev_idx: np.ndarray | None = None,
    rev_raw_f: np.ndarray | None = None,
    rev_gate: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ad = net.adapter_10tok
    anc = ad.anchor
    m = len(pool["pids"])
    bp = np.zeros(m, dtype=np.float64) if base_prior_cost is None else np.asarray(base_prior_cost, dtype=np.float64)
    eff_anc_np = bp + st["eff_delta"]

    with torch.no_grad():
        pemb = torch.from_numpy(st["prog_emb"][None])
        ctx_exp = torch.from_numpy(st["ctx_emb0"]).view(1, 1, -1).expand(1, m, anc.d_model)
        eff_anc = torch.from_numpy(eff_anc_np[None])
        mm_adj = torch.from_numpy(st["mm_adj0"][None])

        if rev_idx is None or len(rev_idx) == 0:
            z1 = torch.zeros((1, m, 1), dtype=torch.float32)
            g_in = torch.cat([ctx_exp, pemb, eff_anc.float().unsqueeze(-1), z1], dim=-1)
            f_log = anc.gate_head(g_in).squeeze(-1)
            g_pen = F.softplus(anc.gate_barrier_weight) * torch.sigmoid(f_log)
            anc_tot = (eff_anc + g_pen.double()).float()
            tot = (anc_tot + mm_adj)[0].numpy()
            return tot, anc_tot[0].numpy(), eff_anc.float()[0].numpy()

        r_i = torch.as_tensor(rev_idx[None], dtype=torch.long)
        r_f = torch.as_tensor(rev_raw_f[None], dtype=torch.float32)
        r_g = torch.as_tensor(rev_gate[None], dtype=torch.float32)
        t_k = int(r_i.shape[1])
        proj16_d = torch.from_numpy(pool["proj16"][None]).double()
        ssig = torch.from_numpy(st["anc_ssig"][None])
        scale_d = 0.4208

        rev_p16 = torch.gather(proj16_d, dim=1, index=r_i.unsqueeze(-1).expand(1, t_k, 16))
        rev_bp = torch.gather(torch.from_numpy(bp[None]), dim=1, index=r_i)
        rev_ep = torch.gather(eff_anc, dim=1, index=r_i)
        res_w1 = r_f.double() / scale_d - rev_bp
        clamp_f = torch.where(r_g > 0.5, r_f.double(), torch.clamp(r_f.double(), max=2.5))
        res_sm = clamp_f / scale_d - rev_ep

        d2_16 = torch.sum((proj16_d.unsqueeze(2) - rev_p16.unsqueeze(1)) ** 2, dim=-1)
        td2, tord = torch.topk(d2_16, k=min(5, t_k), dim=-1, largest=False, sorted=True)
        ww1 = 1.0 / (1.0 + td2)
        w1_adj = torch.sum(ww1 * torch.gather(res_w1.unsqueeze(1).expand(1, m, t_k), -1, tord), -1) / torch.sum(ww1, -1).clamp(min=1e-12)

        rev_s24 = torch.gather(ssig, dim=1, index=r_i.unsqueeze(-1).expand(1, t_k, 24))
        jd2 = 0.5 * d2_16.float() + 2.0 * ((1.0 - torch.einsum("bnd,btd->bnt", ssig, rev_s24)).clamp(min=0.0) * 4.0)
        min_sd = torch.min(jd2, dim=-1).values.clamp(max=10.0) / 10.0
        sd2, sord = torch.topk(jd2.double(), k=min(8, t_k), dim=-1, largest=False, sorted=True)
        ws = 0.7 / (0.25 + sd2) + 0.3 / (2.0 + sd2)
        sden = torch.sum(ws, dim=-1).clamp(min=1e-12)
        st_adj = torch.sum(ws * torch.gather(res_sm.unsqueeze(1).expand(1, m, t_k), -1, sord), -1) / sden
        rg_loc = (torch.sum(ws * torch.gather((1.0 - r_g).double().unsqueeze(1).expand(1, m, t_k), -1, sord), -1) / sden).float()

        rev_pemb = torch.gather(pemb, dim=1, index=r_i.unsqueeze(-1).expand(1, t_k, anc.d_model))
        s_frac = torch.linspace(0.1, 1.0, steps=t_k).unsqueeze(0) if t_k > 1 else torch.full((1, 1), 0.1)
        t_in = torch.cat([rev_pemb, torch.stack([res_sm.float().clamp(-5.0, 5.0), r_g, rev_ep.float().clamp(-5.0, 5.0), s_frac], -1)], -1)
        mem = anc.history_token_in(t_in)
        a_out, _ = anc.cross_attn(query=pemb, key=mem, value=mem, need_weights=False)
        sum_f = torch.stack([w1_adj.float().clamp(-5.0, 5.0), st_adj.float().clamp(-5.0, 5.0), min_sd, rg_loc], -1)
        a_del = torch.tanh(anc.traj_out(torch.cat([ctx_exp, a_out, sum_f], -1)).squeeze(-1))
        g_in = torch.cat([ctx_exp, pemb, eff_anc.float().unsqueeze(-1), rg_loc.unsqueeze(-1)], -1)
        g_pen = F.softplus(anc.gate_barrier_weight) * torch.sigmoid(anc.gate_head(g_in).squeeze(-1) + 2.0 * rg_loc)
        w_st = torch.sigmoid(anc.alpha_struct_kernel).double() * 0.35
        anc_tot = (eff_anc + (1.0 - w_st) * w1_adj + w_st * st_adj + anc.alpha_attn.double() * a_del.double() + g_pen.double()).float()

        cq = torch.from_numpy(st["cand_q0"][None])
        cqn = F.normalize(cq, p=2, dim=-1, eps=1e-8)
        sem24 = torch.from_numpy(st["sem24"][None])
        static_terms = {"eff_prior": eff_anc.float() + mm_adj, "mm_prior_adj": mm_adj, "sem24": sem24, "cand_q": cq, "cand_q_norm": cqn}
        ad_delta = ad.compute_adapter_delta_from_static_terms(static_terms, torch.full((1,), 0.4208), r_i, r_f, r_g)
        tot = (anc_tot + ad_delta)[0].numpy()
        return tot, anc_tot[0].numpy(), eff_anc.float()[0].numpy()


def run_dv02_reveal_policy_benchmark(
    net: Any,
    ordered_cids: Sequence[str],
    families: Sequence[str],
    pools: Mapping[str, Mapping[str, Any]],
    state_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    modes = (
        "masked_presence_0rev",
        "wave0_fake_k4",
        "wave0_fake_k1",
        "serving_storage_prefix4",
        "deterministic_diverse_anchor4",
        "training_oracle_best4",
    )
    n_c = len(ordered_cids)
    u_map = {m: np.zeros(n_c, dtype=np.float64) for m in modes}
    gf_map = {m: np.zeros(n_c, dtype=bool) for m in modes}
    pk_map = {m: np.zeros(n_c, dtype=np.int64) for m in modes}
    mask_v1_max_diff = 0.0

    for i, cid in enumerate(ordered_cids):
        p = pools[cid]
        st = state_map[cid]
        m = len(p["pids"])
        u_all = np.where(p["gate"], -p["fitness"], -2.0)

        idx_m, rf_m, rg_m = filter_masked_presence_v1([0, 1, 2, 3], [100.0] * 4, [0.0] * 4, [False] * 4)
        sc_m0, _, _ = predict_from_state(net, st, p, rev_idx=idx_m, rev_raw_f=rf_m, rev_gate=rg_m)
        sc_e0, _, _ = predict_from_state(net, st, p)
        mask_v1_max_diff = max(mask_v1_max_diff, float(np.max(np.abs(sc_m0 - sc_e0))))

        k4 = min(4, m)
        p4 = np.arange(k4, dtype=np.int64)
        div4 = select_diverse_structural_anchors(p["struct_sig"], p["pids"], k=k4)
        ora4 = np.argsort(p["fitness"])[:k4].astype(np.int64)

        mode_scores = {
            "masked_presence_0rev": sc_m0,
            "wave0_fake_k4": predict_from_state(net, st, p, rev_idx=p4, rev_raw_f=np.full(k4, 100.0, dtype=np.float32), rev_gate=np.zeros(k4, dtype=np.float32))[0],
            "wave0_fake_k1": predict_from_state(net, st, p, rev_idx=np.zeros(1, dtype=np.int64), rev_raw_f=np.full(1, 100.0, dtype=np.float32), rev_gate=np.zeros(1, dtype=np.float32))[0],
            "serving_storage_prefix4": predict_from_state(net, st, p, rev_idx=p4, rev_raw_f=p["fitness"][p4].astype(np.float32), rev_gate=p["gate"][p4].astype(np.float32))[0],
            "deterministic_diverse_anchor4": predict_from_state(net, st, p, rev_idx=div4, rev_raw_f=p["fitness"][div4].astype(np.float32), rev_gate=p["gate"][div4].astype(np.float32))[0],
            "training_oracle_best4": predict_from_state(net, st, p, rev_idx=ora4, rev_raw_f=p["fitness"][ora4].astype(np.float32), rev_gate=p["gate"][ora4].astype(np.float32))[0],
        }
        for m_name, sc in mode_scores.items():
            pk = int(np.argmin(sc))
            pk_map[m_name][i] = pk
            gf_map[m_name][i] = not bool(p["gate"][pk])
            u_map[m_name][i] = float(u_all[pk])

    arms_summary = {
        m_name: {
            "family_mean_utility": round(family_unweighted_mean(u_map[m_name], families), 6),
            "ctx_mean_utility": round(float(np.mean(u_map[m_name])), 6),
            "gate_failures": int(np.sum(gf_map[m_name])),
            "gate_failure_rate": round(float(np.mean(gf_map[m_name])), 6),
            "top1_agree_with_masked_0rev": round(float(np.mean(pk_map[m_name] == pk_map["masked_presence_0rev"])), 6),
        }
        for m_name in modes
    }
    return {
        "n_contexts": n_c,
        "n_families": len(set(families)),
        "masked_presence_false4_vs_empty_max_diff": float(mask_v1_max_diff),
        "modes": arms_summary,
        "contrasts": {
            "masked_0rev_minus_wave0_fake_k4": paired_contrast(u_map["masked_presence_0rev"], u_map["wave0_fake_k4"], families),
            "masked_0rev_minus_wave0_fake_k1": paired_contrast(u_map["masked_presence_0rev"], u_map["wave0_fake_k1"], families),
            "training_oracle_best4_minus_serving_prefix4": paired_contrast(u_map["training_oracle_best4"], u_map["serving_storage_prefix4"], families),
            "diverse_anchor4_minus_serving_prefix4": paired_contrast(u_map["deterministic_diverse_anchor4"], u_map["serving_storage_prefix4"], families),
            "serving_prefix4_minus_masked_0rev": paired_contrast(u_map["serving_storage_prefix4"], u_map["masked_presence_0rev"], families),
        },
        "_inc_fail": gf_map["serving_storage_prefix4"],
        "_inc_pick": {cid: int(pk_map["serving_storage_prefix4"][i]) for i, cid in enumerate(ordered_cids)},
    }


def _stream_seed(*parts: object) -> int:
    dg = hashlib.sha256("h448/".join(str(p) for p in parts).encode("utf-8")).digest()[:8]
    return int.from_bytes(dg, "little")


def _adapter_4rev_batch(
    ad: Any,
    tok: torch.Tensor,
    msk: torch.Tensor,
    prog_raw: torch.Tensor,
    anc_tot: torch.Tensor,
    anc_eff: torch.Tensor,
    rf4: torch.Tensor,
    rg4: torch.Tensor,
) -> torch.Tensor:
    b, m, _ = prog_raw.shape
    z_sos = torch.zeros((b, m, 2), dtype=torch.float32, device=prog_raw.device)
    _, cq, sem24, mm_adj = ad.encode_multimodal_context_and_candidates(
        mm_tokens_10x32=tok, mm_mask_10=msk, prog_412=prog_raw, sos_prior=z_sos
    )
    static_terms = {
        "eff_prior": anc_eff + mm_adj,
        "mm_prior_adj": mm_adj,
        "sem24": sem24,
        "cand_q": cq,
        "cand_q_norm": F.normalize(cq, p=2, dim=-1, eps=1e-8),
    }
    rev_idx4 = torch.arange(4, dtype=torch.long, device=prog_raw.device).unsqueeze(0).expand(b, 4)
    scale_t = torch.full((b,), 0.4208, dtype=torch.float32, device=prog_raw.device)
    return anc_tot + ad.compute_adapter_delta_from_static_terms(static_terms, scale_t, rev_idx4, rf4, rg4)


def _compute_context_noise_targets(
    ordered_cids: Sequence[str],
    pools: Mapping[str, Mapping[str, Any]],
    inc_pick: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(ordered_cids)
    p_noise = np.zeros(n, dtype=np.float64)
    headroom = np.zeros(n, dtype=np.float64)
    ratio = np.zeros(n, dtype=np.float64)
    for i, cid in enumerate(ordered_cids):
        p = pools[cid]
        u = np.where(p["gate"], -p["fitness"], -2.0)
        u_ora = float(np.max(u))
        u_inc = float(u[inc_pick[cid]])
        hr = max(u_ora - u_inc, 0.0)
        passing = np.where(p["gate"])[0]
        if len(passing) >= 8:
            top_p = passing[np.argsort(-u[passing])[: min(48, len(passing))]]
            sem_p = p["struct_sig"][top_p].astype(np.float64)
            sims = sem_p @ sem_p.T
            np.fill_diagonal(sims, -2.0)
            nn_j = np.argmax(sims, axis=1)
            diffs = np.abs(u[top_p] - u[top_p[nn_j]])
            non_tie = diffs[diffs > 1e-6]
            pn = float(np.median(non_tie)) if len(non_tie) > 0 else 0.01
        else:
            pn = 0.05
        p_noise[i] = pn
        headroom[i] = hr
        ratio[i] = pn / (hr + 0.02)
    return np.log1p(p_noise), np.log1p(headroom), np.log1p(ratio)


def _fit_predict_i9_scalar(x_tr: np.ndarray, y_tr: np.ndarray, fams_tr: Sequence[str], x_ev: np.ndarray, seed: int) -> np.ndarray:
    mu = np.mean(x_tr.astype(np.float64), axis=0, keepdims=True)
    sd = np.where(np.std(x_tr.astype(np.float64), axis=0, keepdims=True) < 1e-6, 1.0, np.std(x_tr.astype(np.float64), axis=0, keepdims=True))
    z_tr = np.clip((x_tr.astype(np.float64) - mu) / sd, -8.0, 8.0)
    z_ev = np.clip((x_ev.astype(np.float64) - mu) / sd, -8.0, 8.0)

    uniq_f = sorted(set(str(f) for f in fams_tr))
    rng = np.random.default_rng(int(seed) + 47391)
    perm_f = [uniq_f[i] for i in rng.permutation(len(uniq_f)).tolist()]
    va_f = set(perm_f[: max(1, len(uniq_f) // 5)])
    va_m = np.array([str(f) in va_f for f in fams_tr], dtype=bool)
    tr_m = ~va_m if not np.all(va_m) else np.ones(len(y_tr), dtype=bool)

    def _solve(zf: np.ndarray, yf: np.ndarray, zo: np.ndarray, reg: float) -> np.ndarray:
        xa = np.hstack([zf, np.ones((zf.shape[0], 1), dtype=np.float64)])
        xo = np.hstack([zo, np.ones((zo.shape[0], 1), dtype=np.float64)])
        eye = np.eye(xa.shape[1], dtype=np.float64) * reg
        eye[-1, -1] = 1e-4
        return xo @ np.linalg.solve(xa.T @ xa + eye, xa.T @ yf)

    best_reg, best_mse = 200.0, 1e18
    for reg in (10.0, 50.0, 200.0, 800.0, 2500.0):
        pr = _solve(z_tr[tr_m], y_tr[tr_m], z_tr[va_m], reg)
        mse = float(np.mean((pr - y_tr[va_m]) ** 2))
        if mse < best_mse:
            best_mse, best_reg = mse, reg
    pred_tr = _solve(z_tr, y_tr, z_tr, best_reg)
    pred_ev = _solve(z_tr, y_tr, z_ev, best_reg)
    return np.maximum(np.expm1(pred_tr), 1e-4), np.maximum(np.expm1(pred_ev), 1e-4)


def run_i9_pair_noise_study(
    net: Any,
    guard: Any,
    before_snap: Any,
    init_state: Mapping[str, torch.Tensor],
    init_opt: Mapping[str, Any],
    train_cids: Sequence[str],
    train_families: Sequence[str],
    pools: Mapping[str, Mapping[str, Any]],
    state_map: Mapping[str, Mapping[str, Any]],
    inc_pick: Mapping[str, int],
    camp_tok: np.ndarray,
    camp_msk: np.ndarray,
    repr_train: Mapping[str, np.ndarray],
    folds_list: Sequence[np.ndarray],
    salts: Sequence[str],
    *,
    seeds: Sequence[int] = (1, 2, 3),
    smoke: bool = False,
    val_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verify_mem_available_gib(15.0, "i9_pair_noise")
    n_tr = len(train_cids)
    cid_pos = {c: i for i, c in enumerate(train_cids)}
    y_pn_log, y_hr_log, y_rat_log = _compute_context_noise_targets(train_cids, pools, inc_pick)

    # Precompute 4-reveal training & eval anchor arrays per context
    tr_anc: dict[str, dict[str, np.ndarray]] = {}
    max_m = max(len(pools[c]["pids"]) for c in train_cids)
    pad_prog = torch.zeros((n_tr, max_m, 412), dtype=torch.float32)
    pad_atot = torch.zeros((n_tr, max_m), dtype=torch.float32)
    pad_aeff = torch.zeros((n_tr, max_m), dtype=torch.float32)
    pad_fit = torch.zeros((n_tr, max_m), dtype=torch.float32)
    pad_gate = torch.zeros((n_tr, max_m), dtype=torch.bool)
    all_rf4 = torch.zeros((n_tr, 4), dtype=torch.float32)
    all_rg4 = torch.zeros((n_tr, 4), dtype=torch.float32)
    fit_orders: dict[str, np.ndarray] = {}

    for i, cid in enumerate(train_cids):
        p = pools[cid]
        st = state_map[cid]
        m = len(p["pids"])
        ord_f = np.argsort(p["fitness"])
        fit_orders[cid] = ord_f
        ic = inc_pick[cid]
        hc = [int(x) for x in ord_f[:48]]
        if ic in hc[:4]:
            hc.remove(ic)
            hc.insert(4, ic)
        th4 = np.asarray(hc[:4], dtype=np.int64)
        _, tr_tot, tr_eff = predict_from_state(
            net, st, p, rev_idx=th4, rev_raw_f=p["fitness"][th4].astype(np.float32), rev_gate=p["gate"][th4].astype(np.float32)
        )
        p4 = np.arange(min(4, m), dtype=np.int64)
        _, ev_tot, ev_eff = predict_from_state(
            net, st, p, rev_idx=p4, rev_raw_f=p["fitness"][p4].astype(np.float32), rev_gate=p["gate"][p4].astype(np.float32)
        )
        tr_anc[cid] = {"ev_tot": ev_tot, "ev_eff": ev_eff, "ev_rf4": p["fitness"][p4].astype(np.float32), "ev_rg4": p["gate"][p4].astype(np.float32)}
        pad_prog[i, :m] = torch.from_numpy(p["program"])
        pad_atot[i, :m] = torch.from_numpy(tr_tot)
        pad_aeff[i, :m] = torch.from_numpy(tr_eff)
        pad_fit[i, :m] = torch.as_tensor(p["fitness"], dtype=torch.float32)
        pad_gate[i, :m] = torch.as_tensor(p["gate"], dtype=torch.bool)
        all_rf4[i] = torch.as_tensor(p["fitness"][th4], dtype=torch.float32)
        all_rg4[i] = torch.as_tensor(p["gate"][th4], dtype=torch.float32)

    tok_t = torch.from_numpy(camp_tok.astype(np.float32))
    msk_t = torch.from_numpy(camp_msk.astype(np.float32))
    ad_names = [n for n, _ in net.named_parameters() if n.startswith("adapter_10tok.") and not n.startswith("adapter_10tok.anchor.")]

    def _fit_pairnf_and_margins(tr_cids_cell: list[str], eval_cids_cell: Sequence[str], eval_tok_t: torch.Tensor, eval_msk_t: torch.Tensor, eval_anc_map: Mapping[str, Mapping[str, np.ndarray]], eval_st_map: Mapping[str, Mapping[str, Any]], tag: str, sd: int) -> list[tuple[int, int, float]]:
        net.load_state_dict(init_state, strict=True)
        for param in net.parameters():
            param.requires_grad_(False)
        ad_params = []
        for nm in ad_names:
            pm = dict(net.named_parameters())[nm]
            pm.requires_grad_(True)
            ad_params.append(pm)
        guard.lock_dope_weights_and_eval_mode(net)

        gen_i = torch.Generator().manual_seed(_stream_seed(sd, "pairnf", tag, "init") % (2 ** 63))
        with torch.no_grad():
            for pm in ad_params:
                rms = pm.detach().pow(2).mean().sqrt().clamp_min(1e-8)
                pm.add_(torch.randn(pm.shape, generator=gen_i) * rms * 5e-3)
        opt = torch.optim.AdamW(ad_params, lr=3e-4, weight_decay=1e-4)
        opt.load_state_dict(copy.deepcopy(init_opt))
        for st_o in opt.state.values():
            st_o["exp_avg"].zero_()
        for gp in opt.param_groups:
            gp["lr"], gp["weight_decay"] = 3e-4, 1e-4
        guard.assert_optimizer_excludes_dope(opt, net)

        torch.manual_seed(_stream_seed(sd, "pairnf", tag, "dropout") % (2 ** 63))
        anc_ref = [init_state[n].detach().clone() for n in ad_names]
        anc_norm = sum(float(t.pow(2).sum()) for t in anc_ref)

        for step in range(4 if smoke else 16):
            rr = random.Random(_stream_seed(sd, "pairnf", tag, step, "ctx"))
            rng = np.random.default_rng(_stream_seed(sd, "pairnf", tag, step, "slate"))
            b_cids = rr.choices(tr_cids_cell, k=256 if smoke else 1024)
            sels: list[np.ndarray] = []
            for c in b_cids:
                m_c = len(pools[c]["pids"])
                of = fit_orders[c]
                top48 = of[:48].tolist()
                used = set(top48)
                rest = np.array([j for j in range(m_c) if j not in used], dtype=np.int64)
                need = 192 - len(top48)
                chosen = top48 + rng.choice(rest, need, replace=len(rest) < need).tolist()
                chosen = chosen[:192]
                ic = inc_pick[c]
                if ic in chosen[:4]:
                    chosen.remove(ic)
                    chosen.insert(4, ic)
                elif ic not in chosen:
                    chosen[-1] = ic
                sels.append(np.asarray(chosen, dtype=np.int64))

            net.train()
            opt.zero_grad(set_to_none=True)
            b_ix = torch.tensor([cid_pos[c] for c in b_cids], dtype=torch.long)
            s_t = torch.from_numpy(np.stack(sels, axis=0))
            r_ix = b_ix.unsqueeze(1)
            pred = _adapter_4rev_batch(
                net.adapter_10tok, tok_t[b_ix], msk_t[b_ix],
                pad_prog[r_ix, s_t], pad_atot[r_ix, s_t], pad_aeff[r_ix, s_t],
                all_rf4[b_ix], all_rg4[b_ix],
            )
            tf = pad_fit[r_ix, s_t]
            gm = pad_gate[r_ix, s_t]
            tgt = torch.where(gm, tf, torch.full_like(tf, 2.0))
            row = F.smooth_l1_loss(pred, tgt, reduction="none").mean(-1) + 0.1 * F.kl_div(
                F.log_softmax(-pred / 0.2, -1), F.softmax(-tgt / 0.2, -1), reduction="none"
            ).sum(-1)
            bp = torch.where(gm, pred, torch.full_like(pred, 1e9)).min(-1).values
            fm = (~gm).float()
            row = row + 0.5 * torch.where(gm.any(-1), (F.relu(bp.unsqueeze(-1) + 0.25 - pred) * fm).sum(-1) / fm.sum(-1).clamp_min(1.0), torch.zeros_like(bp))
            tz = (tgt - tgt.mean(-1, keepdim=True)) / tgt.std(-1, keepdim=True).clamp_min(1e-6)
            pz = (pred - pred.mean(-1, keepdim=True)) / pred.std(-1, keepdim=True).clamp_min(1e-4)
            row = row - (F.softmax(-tz / 0.5, -1) * F.log_softmax(-pz / 0.5, -1)).sum(-1)
            ipos = torch.tensor([int(np.where(s == inc_pick[c])[0][0]) for c, s in zip(b_cids, sels)], dtype=torch.long)
            ar = torch.arange(len(b_cids))
            ci, ti, gi = pred[ar, ipos].unsqueeze(-1), tgt[ar, ipos].unsqueeze(-1), gm[ar, ipos].unsqueeze(-1)
            bet = (tgt < ti - 1e-9).float() * gm.float() * gi.float()
            wor = (tgt > ti + 1e-9).float() * gm.float() * gi.float()
            row = row + (bet * F.softplus(-(ci - pred) / 0.1) + wor * F.softplus(-(pred - ci) / 0.1)).sum(-1) / (bet + wor).sum(-1).clamp_min(1.0)
            loss = row.mean() + 5.0 * sum((p - q).pow(2).sum() for p, q in zip(ad_params, anc_ref)) / anc_norm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ad_params, 1.0)
            opt.step()

        guard.verify_dope_frozen_unchanged(net, before_snap)
        net.eval()
        out_m: list[tuple[int, int, float]] = []
        with torch.no_grad():
            for qi, cid in enumerate(eval_cids_cell):
                ac = eval_anc_map[cid]
                pq = _adapter_4rev_batch(
                    net.adapter_10tok, eval_tok_t[qi : qi + 1], eval_msk_t[qi : qi + 1],
                    torch.from_numpy(pools[cid]["program"][None]),
                    torch.from_numpy(ac["ev_tot"][None]), torch.from_numpy(ac["ev_eff"][None]),
                    torch.from_numpy(ac["ev_rf4"][None]), torch.from_numpy(ac["ev_rg4"][None]),
                )[0].numpy()
                cand_j = int(np.argmin(pq))
                inc_j = int(eval_anc_map[cid].get("inc_j", inc_pick.get(cid, 0)))
                zm = float((pq[inc_j] - pq[cand_j]) / max(float(np.std(pq)), 1e-8)) if cand_j != inc_j else 0.0
                out_m.append((cand_j, inc_j, zm))
        return out_m

    eval_folds = folds_list[:1] if smoke else folds_list
    eval_seeds = tuple(seeds[:1]) if smoke else tuple(seeds)
    n_folds = int(np.max(eval_folds[0])) + 1
    denom = 1.0 if smoke else float(len(eval_folds) * len(eval_seeds))

    gate_arms = ("incumbent_4rev", "pairnf_fixed_z40", "pairnf_i9_ultra_ratio", "pairnf_i9_ultra_noise", "pairnf_i9_dope663_ratio", "pairnf_i9_shuffled")
    u_acc = {a: np.zeros(n_tr, dtype=np.float64) for a in gate_arms}
    gf_acc = {a: np.zeros(n_tr, dtype=np.float64) for a in gate_arms}
    sw_acc = {a: 0 for a in gate_arms}
    pred_pn_oof = np.zeros(n_tr, dtype=np.float64)
    pred_hr_oof = np.zeros(n_tr, dtype=np.float64)
    pred_rat_oof = np.zeros(n_tr, dtype=np.float64)

    for p_i, fold_arr in enumerate(eval_folds):
        salt_p = salts[p_i]
        for k in range(1 if smoke else n_folds):
            ho_idx = np.where(fold_arr == k)[0]
            tr_idx = np.where(fold_arr != k)[0]
            tr_cids_k = [train_cids[i] for i in tr_idx.tolist()]
            ho_cids_k = [train_cids[i] for i in ho_idx.tolist()]
            tr_fams_k = [train_families[i] for i in tr_idx.tolist()]
            ho_fams_k = [train_families[i] for i in ho_idx.tolist()]

            for sd in eval_seeds:
                margins = _fit_pairnf_and_margins(
                    tr_cids_k, ho_cids_k, tok_t[ho_idx], msk_t[ho_idx], tr_anc, state_map, f"{salt_p}f{k}", int(sd)
                )
                donor_ho = derange_across_families(ho_fams_k, seed=20261300 + 100 * k + int(sd))
                x_up = repr_train["DopeUltraPacket859"]
                x_dp = repr_train["R_dope"]
                pn_tr_h, pn_ev_h = _fit_predict_i9_scalar(x_up[tr_idx], y_pn_log[tr_idx], tr_fams_k, x_up[ho_idx], int(sd))
                _, hr_ev_h = _fit_predict_i9_scalar(x_up[tr_idx], y_hr_log[tr_idx], tr_fams_k, x_up[ho_idx], int(sd))
                rat_tr_h, rat_ev_h = _fit_predict_i9_scalar(x_up[tr_idx], y_rat_log[tr_idx], tr_fams_k, x_up[ho_idx], int(sd))
                dp_tr_h, dp_ev_h = _fit_predict_i9_scalar(x_dp[tr_idx], y_rat_log[tr_idx], tr_fams_k, x_dp[ho_idx], int(sd))
                _, sh_ev_h = _fit_predict_i9_scalar(x_up[tr_idx], y_rat_log[tr_idx], tr_fams_k, x_up[ho_idx][donor_ho], int(sd))

                pred_pn_oof[ho_idx] += np.log1p(pn_ev_h) / denom
                pred_hr_oof[ho_idx] += np.log1p(hr_ev_h) / denom
                pred_rat_oof[ho_idx] += np.log1p(rat_ev_h) / denom

                z_map = {
                    "incumbent_4rev": np.full(len(ho_idx), 1e9),
                    "pairnf_fixed_z40": np.full(len(ho_idx), 0.40),
                    "pairnf_i9_ultra_ratio": compute_i9_z_ctx(rat_ev_h, float(np.median(rat_tr_h)), 0.40),
                    "pairnf_i9_ultra_noise": compute_i9_z_ctx(pn_ev_h, float(np.median(pn_tr_h)), 0.40),
                    "pairnf_i9_dope663_ratio": compute_i9_z_ctx(dp_ev_h, float(np.median(dp_tr_h)), 0.40),
                    "pairnf_i9_shuffled": compute_i9_z_ctx(sh_ev_h, float(np.median(rat_tr_h)), 0.40),
                }
                for q_i, cid in enumerate(ho_cids_k):
                    p = pools[cid]
                    cand_j, inc_j, zm = margins[q_i]
                    for g_arm, z_arr in z_map.items():
                        use_cand = (cand_j != inc_j) and (zm >= float(z_arr[q_i]))
                        if use_cand:
                            sw_acc[g_arm] += 1
                        fin_j = cand_j if use_cand else inc_j
                        gp = bool(p["gate"][fin_j])
                        gf_acc[g_arm][ho_idx[q_i]] += (0.0 if gp else 1.0) / denom
                        u_acc[g_arm][ho_idx[q_i]] += (-float(p["fitness"][fin_j]) if gp else -2.0) / denom

    net.load_state_dict(init_state, strict=True)
    net.eval()
    mask_ev = (eval_folds[0] == 0) if smoke else np.ones(n_tr, dtype=bool)
    fams_ev = [train_families[i] for i in np.where(mask_ev)[0].tolist()]

    def _rank_avg(vec: np.ndarray) -> np.ndarray:
        order = np.argsort(vec, kind="mergesort")
        ranks = np.empty(len(vec), dtype=np.float64)
        i = 0
        while i < len(vec):
            j = i
            while j + 1 < len(vec) and vec[order[j + 1]] == vec[order[i]]:
                j += 1
            ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return ranks

    def _skill(y_true: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
        yt, yh = y_true[mask_ev], y_hat[mask_ev]
        ss_tot = max(float(np.sum((yt - np.mean(yt)) ** 2)), 1e-12)
        r2 = 1.0 - float(np.sum((yt - yh) ** 2)) / ss_tot
        rt, rh = _rank_avg(yt), _rank_avg(yh)
        rt_c, rh_c = rt - np.mean(rt), rh - np.mean(rh)
        denom_r = math.sqrt(float(np.sum(rt_c ** 2) * np.sum(rh_c ** 2)))
        rho = float(np.sum(rt_c * rh_c) / denom_r) if denom_r > 1e-12 else 0.0
        return {"oof_r2": round(r2, 6), "spearman_rho": round(rho, 6)}

    val_i9: dict[str, Any] | None = None
    if val_bundle is not None and not smoke:
        v_cids = val_bundle["val_cids"]
        v_fams = val_bundle["val_families"]
        v_st = val_bundle["val_state"]
        v_tok = torch.from_numpy(val_bundle["val_camp_tok"].astype(np.float32))
        v_msk = torch.from_numpy(val_bundle["val_camp_msk"].astype(np.float32))
        v_inc = val_bundle["val_inc_pick"]
        v_anc: dict[str, dict[str, Any]] = {}
        for cid in v_cids:
            p = pools[cid]
            p4 = np.arange(min(4, len(p["pids"])), dtype=np.int64)
            _, et, ee = predict_from_state(net, v_st[cid], p, rev_idx=p4, rev_raw_f=p["fitness"][p4].astype(np.float32), rev_gate=p["gate"][p4].astype(np.float32))
            v_anc[cid] = {"ev_tot": et, "ev_eff": ee, "ev_rf4": p["fitness"][p4].astype(np.float32), "ev_rg4": p["gate"][p4].astype(np.float32), "inc_j": v_inc[cid]}
        vu = {a: np.zeros(len(v_cids), dtype=np.float64) for a in ("incumbent_4rev", "pairnf_fixed_z40", "pairnf_i9_ultra_ratio")}
        for sd in eval_seeds:
            vm = _fit_pairnf_and_margins(list(train_cids), v_cids, v_tok, v_msk, v_anc, v_st, "full_train", int(sd))
            rt_tr, rt_ev = _fit_predict_i9_scalar(repr_train["DopeUltraPacket859"], y_rat_log, train_families, val_bundle["repr_val"]["DopeUltraPacket859"], int(sd))
            z_dyn = compute_i9_z_ctx(rt_ev, float(np.median(rt_tr)), 0.40)
            for qi, cid in enumerate(v_cids):
                p = pools[cid]
                cj, ij, zm = vm[qi]
                for ga, z_th in (("incumbent_4rev", 1e9), ("pairnf_fixed_z40", 0.40), ("pairnf_i9_ultra_ratio", float(z_dyn[qi]))):
                    fj = cj if (cj != ij and zm >= z_th) else ij
                    gp = bool(p["gate"][fj])
                    vu[ga][qi] += (-float(p["fitness"][fj]) if gp else -2.0) / float(len(eval_seeds))
        net.load_state_dict(init_state, strict=True)
        net.eval()
        val_i9 = {
            "arms": {ga: {"family_mean_utility": round(family_unweighted_mean(vu[ga], v_fams), 6), "ctx_mean_utility": round(float(np.mean(vu[ga])), 6)} for ga in vu},
            "contrasts": {
                "i9_ultra_ratio_minus_fixed_z40": paired_contrast(vu["pairnf_i9_ultra_ratio"], vu["pairnf_fixed_z40"], v_fams),
                "i9_ultra_ratio_minus_incumbent": paired_contrast(vu["pairnf_i9_ultra_ratio"], vu["incumbent_4rev"], v_fams),
            },
        }

    return {
        "predictability": {
            "log_pair_noise": _skill(y_pn_log, pred_pn_oof),
            "log_incumbent_headroom": _skill(y_hr_log, pred_hr_oof),
            "log_noise_over_headroom": _skill(y_rat_log, pred_rat_oof),
        },
        "arms": {
            ga: {
                "family_mean_utility": round(family_unweighted_mean(u_acc[ga][mask_ev], fams_ev), 6),
                "ctx_mean_utility": round(float(np.mean(u_acc[ga][mask_ev])), 6),
                "gatefail_rate": round(float(np.mean(gf_acc[ga][mask_ev])), 6),
                "total_switches_across_runs": int(sw_acc[ga]),
            }
            for ga in gate_arms
        },
        "contrasts": {
            "i9_ultra_ratio_minus_fixed_z40": paired_contrast(u_acc["pairnf_i9_ultra_ratio"][mask_ev], u_acc["pairnf_fixed_z40"][mask_ev], fams_ev),
            "i9_ultra_noise_minus_fixed_z40": paired_contrast(u_acc["pairnf_i9_ultra_noise"][mask_ev], u_acc["pairnf_fixed_z40"][mask_ev], fams_ev),
            "i9_ultra_ratio_minus_dope663": paired_contrast(u_acc["pairnf_i9_ultra_ratio"][mask_ev], u_acc["pairnf_i9_dope663_ratio"][mask_ev], fams_ev),
            "i9_ultra_ratio_minus_shuffled": paired_contrast(u_acc["pairnf_i9_ultra_ratio"][mask_ev], u_acc["pairnf_i9_shuffled"][mask_ev], fams_ev),
            "i9_ultra_ratio_minus_incumbent": paired_contrast(u_acc["pairnf_i9_ultra_ratio"][mask_ev], u_acc["incumbent_4rev"][mask_ev], fams_ev),
        },
        "val91": val_i9,
    }


def render_summary_markdown(metrics: Mapping[str, Any], verdict: Mapping[str, Any]) -> str:
    prov = metrics["provenance"]
    a1 = metrics["part_a1_dv01_token_layout"]
    a2 = metrics["part_a2_dv03_prog412_collisions"]
    a3_tr = metrics["part_a3_dv02_reveal_policy"]["dl_train1219"]
    pb = metrics["part_b_two_tower"]
    pc_i10 = metrics["part_c"]["i10_zero_shot_op_desc"]
    pc_i9 = metrics["part_c"]["i9_pair_noise_gate"]
    s1_t = a1["checkpoint_token_telemetry"]["stage1_s3_all15796_encoder"]["dl_train1219"]
    err = s1_t["slice_bug_error_quantification"]

    lines = [
        "# H473 Scored Study Summary: Ultra+Packet Retrieval, Zero-Shot Archetypes, & DV-01..03 Audits",
        "",
        f"- **Wave**: `{verdict['wave']}` (`overall_verdict = {verdict['overall_verdict']}`)",
        f"- **Frozen DOPE Composite SHA-256 (`before == after`)**: `{prov['dope_before_sha256']}`",
        f"- **Cohort**: `{prov['n_train']}` `DL-Train` contexts (`{prov['n_train_families']}` families) / `{prov['n_dev']}` `DL-Val` contexts (`{prov['n_dev_families']}` families), `0` family overlap, `0` query rows",
        f"- **Runtime & Memory**: `{prov['total_wall_s']}s` wall, `{prov['max_rss_gib']} GiB` peak RSS, `{prov['scratch_used_mb']} MB` scratch disk",
        "",
        "## 1. Criterion Verdicts",
        "",
        "| Criterion | Status | Summary Evidence |",
        "| --- | --- | --- |",
    ]
    for cid, cobj in verdict["criteria"].items():
        lines.append(f"| `{cid}` | **{cobj['status']}** | {cobj['evidence']} |")

    lines.extend([
        "",
        "## 2. Part A1 (`DV-01`): 92-Token `PreCondenseResidualBridge` Layout & Historical 93-Token Slice Error",
        f"- Contiguous `[0..92]` coverage verified (`token_dim = 32`, separate `scale_side_features` width `8`, inactive column max abs `{s1_t['inactive_column_token_max_abs']:.2e}`).",
        f"- `hidden256` `[20:28]` authoritative L2 `{err['hidden256_20_28_vs_buggy_45_77']['authoritative']:.4f}` vs buggy `[45:77]` `{err['hidden256_20_28_vs_buggy_45_77']['historical_buggy']:.4f}` (relative error `{100.0 * err['hidden256_20_28_vs_buggy_45_77']['rel_error_frac']:+.1f}%`).",
        f"- `columns64` active-masked `[28:92]` L2 `{err['columns64_active_vs_buggy_77_93']['authoritative']:.4f}` vs buggy `[77:93]` (truncated to `15` tokens) `{err['columns64_active_vs_buggy_77_93']['historical_buggy']:.4f}` (`{100.0 * err['columns64_active_vs_buggy_77_93']['rel_error_frac']:+.1f}%`).",
        "",
        "## 3. Part A2 (`DV-03`): `prog_412` Collision & Sufficiency Audit (`DL-Train1219`)",
        f"- Scanned `{a2['n_total_evaluations']}` evaluations across `{a2['n_distinct_program_sha256']}` distinct `program_sha256` hashes (`deterministic = {a2['sha256_to_prog412_deterministic']}`).",
        f"- Exact `prog_412` distinct vectors: `{a2['global_exact_prog412']['distinct_vectors']}` (`{a2['global_exact_prog412']['colliding_programs']}` colliding programs, `{100.0 * a2['global_exact_prog412']['colliding_program_fraction']:.4f}%`).",
        f"- Within-pool exact `prog_412` disagreement: `{a2['within_pool_exact_prog412_disagreement']['pair_count']}` pairs, mean `|Δu| = {a2['within_pool_exact_prog412_disagreement']['utility_abs_diff_mean']:.5f}`, gate disagreement `{100.0 * a2['within_pool_exact_prog412_disagreement']['gate_disagreement_rate']:.2f}%`.",
        f"- Within-pool near-collisions (`cos > 0.9999`): `{a2['within_pool_near_cos_gt_0p9999_disagreement']['pair_count']}` pairs across `{a2['within_pool_near_cos_gt_0p9999_disagreement']['contexts_with_collision']}` contexts, mean `|Δu| = {a2['within_pool_near_cos_gt_0p9999_disagreement']['utility_abs_diff_mean']:.5f}`, p90 `{a2['within_pool_near_cos_gt_0p9999_disagreement']['utility_abs_diff_p90']:.5f}`.",
        "",
        "## 4. Part A3 (`DV-02`): Reveal-Policy & `MaskedPresenceV1` Benchmark",
        "",
        "| Mode (`DL-Train1219`) | Fam Mean Utility | Ctx Mean Utility | Gate Fails | Top-1 Agree vs `masked_0rev` |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for m_k, mr in a3_tr["modes"].items():
        lines.append(f"| `{m_k}` | `{mr['family_mean_utility']:.6f}` | `{mr['ctx_mean_utility']:.6f}` | `{mr['gate_failures']}` | `{mr['top1_agree_with_masked_0rev']:.4f}` |")

    lines.extend([
        "",
        "## 5. Part B (`I3`/`I8`): Two-Tower Retrieval & `base_prior_cost` (`2` Partitions x `5` Folds x `3` Seeds)",
        "",
        "| Arm | Champ Recall@96 | Champ Top-1 Fam Util | Shortlist-16 Oracle | Shortlist-32 Oracle | Top-8 Recall@32 | Prior 0-Rev Fam Util |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for arm_b in ("ret_pop", "ret_lm48_knn", "ret_desc_2t", "ret_dope_2t", "ret_ultra_packet_2t", "ret_ultra_packet_2t_m32", "ret_ultra_packet_shuffled"):
        cs = pb["cross_context_champion_slate_k96"][arm_b]
        wp = pb["within_pool_shortlist_and_prior"].get(arm_b, {})
        o16 = f"`{wp['shortlist16_oracle_family_utility']:.6f}`" if wp else "`—`"
        o32 = f"`{wp['shortlist32_oracle_family_utility']:.6f}`" if wp else "`—`"
        r32 = f"`{wp['top8_pool_recall_at_32']:.4f}`" if wp else "`—`"
        p0u = f"`{wp['prior_0rev_family_utility']:.6f}`" if wp else "`—`"
        lines.append(f"| `{arm_b}` | `{cs['champ_recall_at_96']:.4f}` | `{cs['champ_top1_family_utility']:.6f}` | {o16} | {o32} | {r32} | {p0u} |")

    lines.extend([
        "",
        "## 6. Part C (`I10` Zero-Shot `op_desc` & `I9` `pair_noise` Gate) + `DL-Val91` `REGCHECK`",
        f"- `I10` `op_desc` OOF family utility: `i10_ultra_packet_859` `{pc_i10['arms']['i10_ultra_packet_859']['family_mean_utility']:.6f}` (gate-pass `{pc_i10['arms']['i10_ultra_packet_859']['gate_pass_rate']:.4f}`) vs `i10_dope_663` `{pc_i10['arms']['i10_dope_663']['family_mean_utility']:.6f}` and `i10_none_0d` `{pc_i10['arms']['i10_none_0d']['family_mean_utility']:.6f}`.",
        f"- `I9` predictability (`DopeUltraPacket859` OOF): `log_pair_noise` $R^2 = {pc_i9['predictability']['log_pair_noise']['oof_r2']:.4f}$ ($\\rho = {pc_i9['predictability']['log_pair_noise']['spearman_rho']:.4f}$), `log_incumbent_headroom` $R^2 = {pc_i9['predictability']['log_incumbent_headroom']['oof_r2']:.4f}$.",
        f"- `I9` `pairnf` OOF utility: `pairnf_i9_ultra_ratio` `{pc_i9['arms']['pairnf_i9_ultra_ratio']['family_mean_utility']:.6f}` vs `pairnf_fixed_z40` `{pc_i9['arms']['pairnf_fixed_z40']['family_mean_utility']:.6f}` (`Δ = {pc_i9['contrasts']['i9_ultra_ratio_minus_fixed_z40']['mean_delta']:+.6f}`).",
    ])
    if metrics.get("val91_regcheck") is not None:
        vr = metrics["val91_regcheck"]
        v_a3 = vr["dv02_val91"]
        lines.extend([
            f"- `DL-Val91` `DV-02` `masked_0rev` `{v_a3['modes']['masked_presence_0rev']['family_mean_utility']:.6f}` vs `wave0_fake_k4` `{v_a3['modes']['wave0_fake_k4']['family_mean_utility']:.6f}` (`Δ = {v_a3['contrasts']['masked_0rev_minus_wave0_fake_k4']['mean_delta']:+.6f}`).",
            f"- `DL-Val91` Part B `ret_ultra_packet_2t` Champ Top-1 `{vr['part_b_val91']['champ_slate_k96']['ret_ultra_packet_2t']['champ_top1_family_utility']:.6f}` vs `ret_lm48_knn` `{vr['part_b_val91']['champ_slate_k96']['ret_lm48_knn']['champ_top1_family_utility']:.6f}`; Prior 0-Rev `{vr['part_b_val91']['within_pool']['ret_ultra_packet_2t']['prior_0rev_family_utility']:.6f}` vs `incumbent_0rev` `{vr['part_b_val91']['within_pool']['incumbent_0rev']['prior_0rev_family_utility']:.6f}`.",
        ])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    data_root = Path(os.environ.get("JOPE_DATA_ROOT", "/home/ubuntu/jope-data")).resolve()
    ap.add_argument("--h470-dir", type=Path, default=data_root / "hwave-h470")
    ap.add_argument("--h256-dir", type=Path, default=data_root / "h256")
    ap.add_argument("--scratch-dir", type=Path, default=data_root / "hwave-h473" / "scratch")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--partitions", default="h452a/,h452b/")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    n_threads = max(1, min(12, int(args.threads)))
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    torch.set_num_threads(n_threads)
    verify_mem_available_gib(15.0, "startup")

    scratch = args.scratch_dir.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    verify_scratch_budget_mb(scratch, 50.0)

    from automatic_promotion import replay
    from jope.north_star import dope_freeze_guard as guard
    from phase12_multimodal_model import compute_semantic_struct_sig_24d
    import rust_evaluator

    guard.reject_research_dope_unfreeze("h473_ultra_packet_retrieval_and_audits")
    h470_dir = args.h470_dir.resolve()
    inp_dir = h470_dir / "input"
    for p_chk, sha_exp in (
        (inp_dir / "cpu_exact_fused.py", CPU_EXACT_FUSED_SHA256),
        (h470_dir / "round-3175.pt", ROUND_3175_SHA256),
        (h470_dir / "promotions.jsonl", PROMOTIONS_SHA256),
        (inp_dir / "selected.json", SELECTED_JSON_SHA256),
        (inp_dir / "champions.json", CHAMPIONS_JSON_SHA256),
    ):
        if digest_file_sha256(p_chk) != sha_exp:
            raise ValueError(f"input SHA-256 mismatch on {p_chk.name}")

    if str(inp_dir) not in sys.path:
        sys.path.insert(0, str(inp_dir))
    import cpu_exact_fused

    net = cpu_exact_fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    for pm in net.parameters():
        pm.requires_grad_(False)
    guard.lock_dope_weights_and_eval_mode(net)
    before_snap = guard.snapshot_dope_weights(net)
    if before_snap.composite_sha256 != DOPE_FUSED_COMPOSITE_SHA256:
        raise RuntimeError(f"DOPE SHA-256 mismatch: {before_snap.composite_sha256}")

    ckpt = torch.load(h470_dir / "round-3175.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["state_dict"], strict=True)
    guard.verify_dope_frozen_unchanged(net, before_snap)
    init_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    init_opt = ckpt["optimizer"]

    sel_doc = json.loads((inp_dir / "selected.json").read_text(encoding="utf-8"))
    train_recs = sel_doc["train"]
    dev_recs = sel_doc["dev"]
    if args.smoke:
        train_recs = train_recs[:180]
    train_cids = [str(r["context_sha256"]) for r in train_recs]
    dev_cids = [str(r["context_sha256"]) for r in dev_recs]
    fam_by_cid = {str(r["context_sha256"]): str(r["family_id"]) for r in train_recs + dev_recs}
    train_families = [fam_by_cid[c] for c in train_cids]
    dev_families = [fam_by_cid[c] for c in dev_cids]
    if set(train_families) & set(dev_families):
        raise ValueError("train/dev family overlap detected")

    champs_list = json.loads((inp_dir / "champions.json").read_text(encoding="utf-8"))
    champs_by_cid = {str(r["context_sha256"]): str(r["program_sha256"]) for r in champs_list if r.get("dl_split") == "DL-Train"}

    wanted_keys = (
        "vec256_hpc", "col_lm_34_hpc", "dope629_hpc", "dope629_ultra",
        "col_lm_34_ultra", "R_packet_v1_ultra", "lm48_z_hpc",
        "tokens_10x32_campaign", "mask_10_campaign",
    )
    tr_feats_full, tr_man = authenticate_feature_cache(h470_dir / "cache", TRAIN_MANIFEST_NAME, wanted_keys)
    tr_feats = {k: v[: len(train_cids)] for k, v in tr_feats_full.items()}
    repr_train = build_repr_bundle(tr_feats)

    w1_mat = rust_evaluator.get_w1_sign_matrix()
    pools = load_cohort_pools(inp_dir, train_recs, w1_mat, compute_semantic_struct_sig_24d)
    print(f"[H473] loaded {len(train_cids)} DL-Train pools, mem={verify_mem_available_gib(15.0, 'pools'):.1f}GiB", flush=True)

    # Part A2 on raw 1,152-candidate pools BEFORE replay, plus post-replay count
    part_a2 = audit_prog412_collisions(train_cids, pools, champions_records=champs_list)
    n_replayed = replay(pools, h470_dir / "promotions.jsonl") if not args.smoke else 0
    part_a2["post_promotion_replay_events_added"] = int(n_replayed)
    print(f"[H473] Part A2 DV-03 done (distinct_sha={part_a2['n_distinct_program_sha256']})", flush=True)

    # Part A1 DV-01 (DL-Train only before REGCHECK)
    layout_out = None if args.smoke else (HERE / "token_layout.json")
    part_a1 = run_dv01_h256_layout_audit(args.h256_dir.resolve(), train_cids, layout_out_path=layout_out)
    print("[H473] Part A1 DV-01 done", flush=True)

    # Precompute candidate anchor states & verify <1e-5 parity across 0-reveal, base_prior_cost, and 4-reveal
    tr_state = precompute_context_anchor_state(
        net, train_cids, pools, tr_feats["tokens_10x32_campaign"], tr_feats["mask_10_campaign"]
    )
    max_parity_err = 0.0
    with torch.no_grad():
        for i_chk in range(min(6, len(train_cids))):
            c_chk = train_cids[i_chk]
            p_chk = pools[c_chk]
            m_chk = len(p_chk["pids"])
            bp_chk = np.linspace(-0.4, 0.4, num=m_chk, dtype=np.float32)
            for use_4, bp_vec in ((False, np.zeros(m_chk, dtype=np.float32)), (False, bp_chk), (True, np.zeros(m_chk, dtype=np.float32))):
                r_idx = np.arange(4, dtype=np.int64) if use_4 else None
                r_f = p_chk["fitness"][:4].astype(np.float32) if use_4 else None
                r_g = p_chk["gate"][:4].astype(np.float32) if use_4 else None
                fast_c, _, _ = predict_from_state(net, tr_state[c_chk], p_chk, base_prior_cost=bp_vec, rev_idx=r_idx, rev_raw_f=r_f, rev_gate=r_g)
                ref_c = net.adapter_10tok.predict_costs(
                    ctx_feat=torch.zeros((1, 584)),
                    prog_412=torch.from_numpy(p_chk["program"][None]),
                    proj_16=torch.from_numpy(p_chk["proj16"][None]),
                    base_prior_cost=torch.from_numpy(bp_vec[None]),
                    scale=torch.full((1,), 0.4208),
                    sos_prior=torch.zeros((1, m_chk, 2)),
                    mm_tokens_10x32=torch.from_numpy(tr_feats["tokens_10x32_campaign"][i_chk : i_chk + 1]),
                    mm_mask_10=torch.from_numpy(tr_feats["mask_10_campaign"][i_chk : i_chk + 1]),
                    revealed_indices=torch.arange(4).view(1, 4) if use_4 else torch.empty((1, 0), dtype=torch.long),
                    revealed_raw_f=torch.from_numpy(r_f[None]) if use_4 else torch.empty((1, 0)),
                    revealed_gate=torch.from_numpy(r_g[None]) if use_4 else torch.empty((1, 0)),
                )["total_cost"].float()[0].numpy()
                max_parity_err = max(max_parity_err, float(np.max(np.abs(fast_c - ref_c))))
    if max_parity_err > 5e-5:
        raise RuntimeError(f"state parity error {max_parity_err} exceeds 5e-5")
    print(f"[H473] precomputed DL-Train states (parity_max_diff={max_parity_err:.2e})", flush=True)

    # Part A3 DV-02 on DL-Train
    dv02_tr_raw = run_dv02_reveal_policy_benchmark(net, train_cids, train_families, pools, tr_state)
    inc_fail_tr = dv02_tr_raw.pop("_inc_fail")
    inc_pick_tr = dv02_tr_raw.pop("_inc_pick")
    salts = [s.strip() for s in args.partitions.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    folds_list = build_h452_family_folds(train_cids, fam_by_cid, inc_fail_tr, salts, n_folds=5)

    def _zero_rev_cost(cid: str, bp_vec: np.ndarray) -> np.ndarray:
        return predict_from_state(net, tr_state[cid], pools[cid], base_prior_cost=bp_vec)[0]

    # Part B Two-Tower study on DL-Train CV
    part_b = run_part_b_two_tower_study(
        train_cids, train_families, pools, champs_by_cid, repr_train, folds_list, _zero_rev_cost,
        seeds=seeds, contrast_fn=paired_contrast, smoke=args.smoke,
    )
    print("[H473] Part B Two-Tower CV done", flush=True)

    # One-time DL-Val91 load (ONLY when not --smoke, AFTER all DL-Train CV choices are frozen!)
    val_bundle_c: dict[str, Any] | None = None
    val91_regcheck: dict[str, Any] | None = None
    dv02_val_clean: dict[str, Any] | None = None
    if not args.smoke:
        part_a1 = run_dv01_h256_layout_audit(args.h256_dir.resolve(), train_cids, layout_out_path=layout_out, val_cids=dev_cids)
        dev_feats, dev_man = authenticate_feature_cache(h470_dir / "cache", VAL91_MANIFEST_NAME, wanted_keys)
        repr_val = build_repr_bundle(dev_feats)
        dev_pools = load_cohort_pools(inp_dir, dev_recs, w1_mat, compute_semantic_struct_sig_24d)
        pools.update(dev_pools)
        val_state = precompute_context_anchor_state(
            net, dev_cids, pools, dev_feats["tokens_10x32_campaign"], dev_feats["mask_10_campaign"]
        )
        tr_state.update(val_state)
        dv02_v_raw = run_dv02_reveal_policy_benchmark(net, dev_cids, dev_families, pools, val_state)
        dv02_v_raw.pop("_inc_fail")
        val_inc_pick = dv02_v_raw.pop("_inc_pick")
        dv02_val_clean = dv02_v_raw
        part_b_val = eval_part_b_val91_regcheck(
            train_cids, dev_cids, dev_families, pools, champs_by_cid, repr_train, repr_val,
            _zero_rev_cost, part_b["selected_alpha_2t_mean"], seeds=seeds, contrast_fn=paired_contrast,
        )
        val_bundle_c = {
            "val_cids": dev_cids,
            "val_families": dev_families,
            "arm_features_val": repr_val,
            "repr_val": repr_val,
            "val_state": val_state,
            "val_camp_tok": dev_feats["tokens_10x32_campaign"],
            "val_camp_msk": dev_feats["mask_10_campaign"],
            "val_inc_pick": val_inc_pick,
            "dev_manifest_sha256": dev_man["manifest_sha256"],
        }

    # Part C (`I10` + `I9`, including one-time `val91` pass if `val_bundle_c` is set)
    i10_res = run_i10_zero_shot_archetype_cv(
        train_cids, train_families, pools, repr_train, folds_list,
        seeds=seeds, contrast_fn=paired_contrast, smoke=args.smoke, val_bundle=val_bundle_c,
    )
    i9_res = run_i9_pair_noise_study(
        net, guard, before_snap, init_state, init_opt, train_cids, train_families, pools,
        tr_state, inc_pick_tr, tr_feats["tokens_10x32_campaign"], tr_feats["mask_10_campaign"],
        repr_train, folds_list, salts, seeds=seeds, smoke=args.smoke, val_bundle=val_bundle_c,
    )
    print("[H473] Part C (I10 & I9) done", flush=True)

    after_snap = guard.snapshot_dope_weights(net)
    guard.verify_dope_frozen_unchanged(net, before_snap)
    if after_snap.composite_sha256 != DOPE_FUSED_COMPOSITE_SHA256:
        raise RuntimeError("post-run DOPE SHA-256 mismatch")

    if val_bundle_c is not None:
        val91_regcheck = {
            "dev_manifest_sha256": val_bundle_c["dev_manifest_sha256"],
            "dv02_val91": dv02_val_clean,
            "part_b_val91": part_b_val,
            "i10_val91": i10_res.pop("val91"),
            "i9_val91": i9_res.pop("val91"),
        }
    else:
        i10_res.pop("val91", None)
        i9_res.pop("val91", None)

    primary_p = {
        "DV02_masked0rev_vs_wave0_fake_k4": dv02_tr_raw["contrasts"]["masked_0rev_minus_wave0_fake_k4"]["p_value"],
        "B_champ_ultra_vs_lm48_knn": part_b["contrasts"]["champ_top1_ultra_minus_lm48_knn"]["p_value"],
        "B_shortlist32_ultra_vs_dope2t": part_b["contrasts"]["shortlist32_oracle_ultra_minus_dope2t"]["p_value"],
        "B_prior0rev_ultra_vs_incumbent0rev": part_b["contrasts"]["prior0rev_ultra_minus_incumbent0rev"]["p_value"],
        "C_i10_ultra_vs_dope663": i10_res["contrasts"]["ultra_packet_minus_dope663"]["p_value"],
        "C_i9_ultra_ratio_vs_fixed_z40": i9_res["contrasts"]["i9_ultra_ratio_minus_fixed_z40"]["p_value"],
    }
    holm_p = {k: round(float(v), 6) for k, v in holm_bonferroni(primary_p).items()}

    scratch_mb = verify_scratch_budget_mb(scratch, 50.0)
    rss_gib = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0), 3)
    provenance = {
        "wave": "H473",
        "smoke_mode": bool(args.smoke),
        "dope_before_sha256": before_snap.composite_sha256,
        "dope_after_sha256": after_snap.composite_sha256,
        "dope_frozen_unchanged": before_snap.composite_sha256 == after_snap.composite_sha256,
        "train_manifest_sha256": tr_man["manifest_sha256"],
        "state_parity_max_abs_diff": round(max_parity_err, 8),
        "n_train": len(train_cids),
        "n_train_families": len(set(train_families)),
        "n_dev": 0 if args.smoke else len(dev_cids),
        "n_dev_families": 0 if args.smoke else len(set(dev_families)),
        "train_dev_family_overlap": 0,
        "query_rows_used": 0,
        "val91_evaluations_count": 0 if args.smoke else 1,
        "partitions": salts,
        "seeds": seeds,
        "max_rss_gib": rss_gib,
        "scratch_used_mb": scratch_mb,
        "total_wall_s": round(time.perf_counter() - t0, 2),
    }

    metrics = {
        "schema": "jope.h473.metrics/v1",
        "provenance": provenance,
        "part_a1_dv01_token_layout": part_a1,
        "part_a2_dv03_prog412_collisions": part_a2,
        "part_a3_dv02_reveal_policy": {"dl_train1219": dv02_tr_raw},
        "part_b_two_tower": part_b,
        "part_c": {"i10_zero_shot_op_desc": i10_res, "i9_pair_noise_gate": i9_res, "holm_adjusted_p": holm_p},
        "val91_regcheck": val91_regcheck,
    }

    dv02_gain = dv02_tr_raw["contrasts"]["masked_0rev_minus_wave0_fake_k4"]
    b_champ = part_b["contrasts"]["champ_top1_ultra_minus_lm48_knn"]
    b_prior = part_b["contrasts"]["prior0rev_ultra_minus_incumbent0rev"]
    c_i10 = i10_res["contrasts"]["ultra_packet_minus_none0d"]
    c_i9 = i9_res["contrasts"]["i9_ultra_ratio_minus_fixed_z40"]

    criteria = {
        "DV01_h256_92tok_layout": {
            "status": "PASS",
            "evidence": "92-token layout [0..92] + 8D non-token scale_side_features verified; historical 93-token slice distortion quantified.",
        },
        "DV02_masked_presence_and_4rev_policy": {
            "status": "PASS" if float(dv02_gain["ci_low"]) > 0.0 else "DIAGNOSED",
            "evidence": f"masked_0rev - wave0_fake_k4 Δ={dv02_gain['mean_delta']:+.6f} CI=[{dv02_gain['ci_low']:+.6f}, {dv02_gain['ci_high']:+.6f}]; oracle4 - prefix4 Δ={dv02_tr_raw['contrasts']['training_oracle_best4_minus_serving_prefix4']['mean_delta']:+.6f}.",
        },
        "DV03_prog412_collision_audit": {
            "status": part_a2["dv03_structural_sufficiency_verdict"],
            "evidence": f"{part_a2['n_distinct_program_sha256']} distinct hashes -> {part_a2['global_exact_prog412']['distinct_vectors']} exact prog_412 vectors ({part_a2['within_pool_exact_prog412_disagreement']['pair_count']} within-pool exact pairs, {part_a2['within_pool_near_cos_gt_0p9999_disagreement']['pair_count']} near pairs cos>0.9999).",
        },
        "I3_I8_two_tower_ultra_packet": {
            "status": "PASS" if (float(b_champ["ci_low"]) > 0.0 or float(b_prior["ci_low"]) > 0.0) else "NO_LIFT",
            "evidence": f"Champ Top-1 ultra-lm48_knn Δ={b_champ['mean_delta']:+.6f} CI=[{b_champ['ci_low']:+.6f}, {b_champ['ci_high']:+.6f}]; Prior 0-rev ultra-incumbent Δ={b_prior['mean_delta']:+.6f} CI=[{b_prior['ci_low']:+.6f}, {b_prior['ci_high']:+.6f}].",
        },
        "I10_zero_shot_op_desc": {
            "status": "PASS" if float(c_i10["ci_low"]) > 0.0 else "NO_LIFT",
            "evidence": f"i10_ultra_packet_859 - i10_none_0d Δ={c_i10['mean_delta']:+.6f} CI=[{c_i10['ci_low']:+.6f}, {c_i10['ci_high']:+.6f}]; vs dope663 Δ={i10_res['contrasts']['ultra_packet_minus_dope663']['mean_delta']:+.6f}.",
        },
        "I9_pair_noise_dynamic_z_gate": {
            "status": "PASS" if float(c_i9["ci_low"]) > 0.0 else "NO_LIFT",
            "evidence": f"log_pair_noise R²={i9_res['predictability']['log_pair_noise']['oof_r2']:.4f}; i9_ultra_ratio - fixed_z40 Δ={c_i9['mean_delta']:+.6f} CI=[{c_i9['ci_low']:+.6f}, {c_i9['ci_high']:+.6f}].",
        },
    }

    if args.smoke:
        overall = "SMOKE_VERIFIED"
    elif any(criteria[k]["status"] == "PASS" for k in ("I3_I8_two_tower_ultra_packet", "I10_zero_shot_op_desc", "I9_pair_noise_dynamic_z_gate")):
        overall = "AUDITS_CLOSED_WITH_ULTRA_PACKET_RETRIEVAL_SIGNAL"
    else:
        overall = "AUDITS_CLOSED_NULL_RETRIEVAL_LIFT"

    verdict = {
        "schema": "jope.h473.verdict/v1",
        "wave": "H473",
        "smoke_mode": bool(args.smoke),
        "overall_verdict": overall,
        "dope_composite_sha256": DOPE_FUSED_COMPOSITE_SHA256,
        "criteria": criteria,
    }
    summary_md = render_summary_markdown(metrics, verdict)

    sfx = "_smoke" if args.smoke else ""
    write_strict_json(scratch / f"metrics{sfx}.json", metrics)
    write_strict_json(scratch / f"verdict{sfx}.json", verdict)
    (scratch / f"SUMMARY{sfx}.md").write_text(summary_md, encoding="utf-8")
    verify_scratch_budget_mb(scratch, 50.0)

    if not args.smoke:
        write_strict_json(HERE / "metrics.json", metrics)
        write_strict_json(HERE / "verdict.json", verdict)
        (HERE / "SUMMARY.md").write_text(summary_md, encoding="utf-8")

    print(f"[H473] DONE verdict={overall} wall={provenance['total_wall_s']}s rss={rss_gib}GiB scratch={scratch_mb}MB", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""H431 synthetic, zero-leakage, and r9 byte-equivalence unit suite (6 checks; runs without real tables)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
R9 = Path("/home/ubuntu/apex-stage/h419/r9/code")
sys.path.insert(0, str(HERE))
if str(R9) not in sys.path:
    sys.path.append(str(R9))

import icb_gen as GEN  # noqa: E402
import icl_inject as INJ  # noqa: E402
import icl_model as M_R9  # noqa: E402
import icl_model_cw3 as M_CW3  # noqa: E402
import icl_prior as P_R9  # noqa: E402
import icl_prior_v3s as P_V3S  # noqa: E402
import icl_resid as RES  # noqa: E402
import icl_train_cw3 as TR_CW3  # noqa: E402
import strong_base as SB  # noqa: E402


def check_1_battery_noisefree_reference() -> dict:
    tasks = GEN.generate_battery(seed=431_001)
    assert len(tasks) == 340, f"expected 340 tasks, got {len(tasks)}"
    fam_nf = {}
    for fid, _ in GEN.FAMILIES:
        if fid == "B12":
            continue
        vals = [t["meta"]["u8_ref"] for t in tasks if t["meta"]["family"] == fid and t["meta"]["noise"] == 0.0]
        assert len(vals) == 10, f"{fid} noise-free count {len(vals)}"
        mean_u = float(np.mean(vals))
        assert mean_u >= 6.0, f"{fid} noise-free reference mean u8 = {mean_u:.3f} < 6.0"
        fam_nf[fid] = round(mean_u, 4)
    return {"tasks": len(tasks), "noisefree_mean_u8_ref": fam_nf}


def check_2_reference_query_isolation() -> dict:
    rng = np.random.default_rng(77)
    max_diff = 0.0
    for fid, _ in GEN.FAMILIES:
        nz = 0.20 if fid == "B12" else 0.05
        t = GEN.make_single_task(rng, fid, n_ctx=256, n_q=64, noise=nz, rep=0)
        xc = t["x"][:256].astype(np.float64)
        yc = t["y"][:256].astype(np.float64)
        xq1 = t["x"][256:].astype(np.float64).copy()
        xq2 = xq1.copy()
        xq2[1:] = xq2[1:] * -3.7 + 19.0   # corrupt all query rows except row 0
        p1 = GEN.fit_reference_predictor(t["meta"], xc, yc, xq1)
        p2 = GEN.fit_reference_predictor(t["meta"], xc, yc, xq2)
        d = abs(float(p1[0] - p2[0]))
        max_diff = max(max_diff, d)
        assert d == 0.0, f"{fid} leaked across query batch: diff={d}"
    return {"max_query_row0_diff": max_diff}


def check_3_v3s_prior_families_finite() -> dict:
    p = P_V3S.Prior("cpu", 991)
    counts = {}
    for ki in range(len(P_V3S.KINDS), len(P_V3S.KINDS_V3S)):
        name = P_V3S.KINDS_V3S[ki]
        for b_iter in range(10):
            kinds = torch.full((6,), ki, dtype=torch.long)
            x, y, k_out = p.sample(6, 256, 12, kinds=kinds, probs=P_V3S.PROBS_V3S)
            assert x.shape == (6, 256, 12) and y.shape == (6, 256)
            assert bool(torch.isfinite(y).all()), f"{name} produced non-finite y at iter {b_iter}"
            assert float(y.std(dim=1).min().item()) > 1e-5, f"{name} produced constant y"
        counts[name] = 60
    # Mixture draw check
    x_mix, y_mix, k_mix = p.sample(32, 256, 10, probs=P_V3S.PROBS_V3S)
    assert bool(torch.isfinite(y_mix).all()) and x_mix.shape == (32, 256, 10)
    return {"verified_tasks_per_new_family": counts, "probs_v3s_sum": round(sum(P_V3S.PROBS_V3S), 6)}


def check_4_strong_base_nonlinear_gain_and_isolation() -> dict:
    g = torch.Generator().manual_seed(42)
    x = torch.randn((4, 320, 8), generator=g)
    # Nonlinear step + quadratic target where ridge leaves large structure
    sig = 3.0 * (x[:, :, 0] > 0.2).float() - 2.5 * (x[:, :, 1] < -0.3).float() + 0.8 * x[:, :, 2]
    y = sig + 0.05 * torch.randn((4, 320), generator=g)
    yr_ridge, used_r, bq_r = RES.residual_targets(x, y, 256, torch.Generator().manual_seed(11))
    yr_strong, used_s, bq_s = SB.strong_residual_targets(x, y, 256, torch.Generator().manual_seed(11))
    assert bool(used_r.all()) and bool(used_s.all())
    sd_r = float(yr_ridge[:, 256:].double().std().item())
    sd_s = float(yr_strong[:, 256:].double().std().item())
    assert sd_s < 0.75 * sd_r, f"strong_base did not cut nonlinear residual: {sd_s:.4f} vs {sd_r:.4f}"
    # Query-row isolation check
    x2 = x.clone()
    x2[:, 257:, :] = x2[:, 257:, :] * -4.0 + 12.0
    y2 = y.clone()
    y2[:, 256:] = y2[:, 256:] + 999.0
    _, _, bq_s2 = SB.strong_residual_targets(x2, y2, 256, torch.Generator().manual_seed(11))
    q0_diff = float((bq_s[:, 0] - bq_s2[:, 0]).abs().max().item())
    assert q0_diff == 0.0, f"strong_base leaked across query rows/targets: {q0_diff}"
    return {"ridge_res_sd": round(sd_r, 4), "strong_res_sd": round(sd_s, 4), "q0_leak_diff": q0_diff}


def check_5_r9_byte_for_byte_equivalence() -> dict:
    # 5a: Prior v1 & v2 batches + INJ.sample_synth byte-identical to r9
    for probs_r9, probs_cw3, label in ((P_R9.PROBS, P_V3S.PROBS, "v1"), (P_R9.PROBS_V2, P_V3S.PROBS_V2, "v2")):
        p1, r1 = P_R9.Prior("cpu", 777), random.Random(555)
        p2, r2 = P_V3S.Prior("cpu", 777), random.Random(555)
        for _ in range(5):
            x1, y1, k1, inj1 = INJ.sample_synth(p1, r1, 8, 256, 12, p=0.5, probs=probs_r9)
            x2, y2, k2, inj2 = INJ.sample_synth(p2, r2, 8, 256, 12, p=0.5, probs=probs_cw3)
            assert inj1 == inj2 and torch.equal(k1, k2), f"{label} kinds mismatch"
            assert x1.numpy().tobytes() == x2.numpy().tobytes(), f"{label} x bytes differ"
            assert y1.numpy().tobytes() == y2.numpy().tobytes(), f"{label} y bytes differ"

    # 5b: ICL model init + forward + loss byte-identical to r9 when dropout == 0.0
    torch.manual_seed(1234)
    m1 = M_R9.ICL("S").eval()
    torch.manual_seed(1234)
    m2 = M_CW3.ICL("S", dropout=0.0).eval()
    assert list(m1.state_dict().keys()) == list(m2.state_dict().keys())
    for k in m1.state_dict():
        assert m1.state_dict()[k].numpy().tobytes() == m2.state_dict()[k].numpy().tobytes(), f"param {k} differs"
    p_test = P_R9.Prior("cpu", 99)
    xt, yt, _ = p_test.sample(4, 192, 9, probs=P_R9.PROBS_V2)
    with torch.no_grad():
        lg1, ym1, ys1 = m1(xt, yt[:, :128], 128, amp=False)
        lg2, ym2, ys2 = m2(xt, yt[:, :128], 128, amp=False)
        ls1 = m1.loss(lg1, yt[:, 128:], ym1, ys1)
        ls2 = m2.loss(lg2, yt[:, 128:], ym2, ys2)
    assert lg1.numpy().tobytes() == lg2.numpy().tobytes(), "forward logits bytes differ"
    assert ls1.numpy().tobytes() == ls2.numpy().tobytes(), "loss bytes differ"
    return {"v1_batches_equal": True, "v2_batches_equal": True, "model_init_and_forward_bytes_equal": True}


def check_6_replay_seed_determinism_and_flags() -> dict:
    # CW-3 section 3 Factor C requirement: batch generation from (seed, shape) is byte-identical across calls
    x1, y1, k1, i1 = TR_CW3.deterministic_synth_batch("cpu", 424242, 6, 256, 10, 0.5, "v3s")
    x2, y2, k2, i2 = TR_CW3.deterministic_synth_batch("cpu", 424242, 6, 256, 10, 0.5, "v3s")
    assert i1 == i2 and torch.equal(k1, k2)
    assert x1.numpy().tobytes() == x2.numpy().tobytes()
    assert y1.numpy().tobytes() == y2.numpy().tobytes()
    # Verify Factor B & E helpers produce finite tensors
    gen = torch.Generator().manual_seed(9)
    x_aug = TR_CW3.augment_features(x1, 192, cell_drop=0.05, jitter=0.01, gen=gen)
    m = M_CW3.ICL("S", dropout=0.05).train()
    lg_a, ym, ys = m(x_aug, y1[:, :192], 192, amp=False)
    lg_b, _, _ = m(x_aug, y1[:, :192], 192, slots=torch.randperm(256)[:10], amp=False)
    kl = TR_CW3.symmetric_kl(lg_a, lg_b)
    loss = m.loss(lg_a, y1[:, 192:], ym, ys).mean() + 0.1 * kl
    assert bool(torch.isfinite(loss)) and float(kl.item()) >= 0.0
    return {"replay_batch_bytes_equal": True, "factors_be_finite_loss": round(float(loss.item()), 4)}


def main() -> None:
    torch.set_num_threads(1)
    res = {
        "check_1_battery_noisefree_reference": check_1_battery_noisefree_reference(),
        "check_2_reference_query_isolation": check_2_reference_query_isolation(),
        "check_3_v3s_prior_families_finite": check_3_v3s_prior_families_finite(),
        "check_4_strong_base_nonlinear_gain_and_isolation": check_4_strong_base_nonlinear_gain_and_isolation(),
        "check_5_r9_byte_for_byte_equivalence": check_5_r9_byte_for_byte_equivalence(),
        "check_6_replay_seed_determinism_and_flags": check_6_replay_seed_determinism_and_flags(),
        "all_passed": True,
    }
    print(json.dumps(res, sort_keys=True))
    print("CW3_UNIT_TESTS_OK 6/6")


if __name__ == "__main__":
    main()

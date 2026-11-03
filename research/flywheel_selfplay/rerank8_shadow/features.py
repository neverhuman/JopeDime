"""Outcome-blind H273 feature/selection functions; source SHA24606a84b2402c59eebb40ebed6e09ee4b8a51f8d43c27779ec3b25a8b0b15f2."""

import math
import numpy as np

SLICE_K = 8


def compute_sem24_numpy(prog_412: np.ndarray) -> np.ndarray:
    """Vectorized 24D structural semantic signature."""
    x = np.asarray(prog_412, dtype=np.float32)
    orig_shape = x.shape[:-1]
    flat = x.reshape(-1, 412)
    padded = np.pad(flat, ((0, 0), (0, 20)), mode="constant").reshape(-1, 18, 24)
    sig = padded.mean(axis=1) + 0.5 * padded.max(axis=1)
    norms = np.maximum(np.linalg.norm(sig, axis=-1, keepdims=True), 1e-6)
    return (sig / norms).reshape(*orig_shape, 24).astype(np.float32)


def build_topk_feature_matrix_for_context(
    rcost: np.ndarray,
    aprior: np.ndarray,
    prog_full: np.ndarray,
    sem24_full: np.ndarray,
    proj16_full: np.ndarray,
    lm48_z: np.ndarray,
    dope629: np.ndarray,
    k: int = SLICE_K,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clean-room K-slice feature builder (568 cols at K=8, 560 cols at K=4), written from
    H270 PREREG.md text specification:
      - Slice is `r882`'s `K` lowest-cost candidates (`np.argsort(rcost, kind='mergesort')[:K]`).
      - Per-context z-features (`slice z`, `slice softmax`, `slice spread = (c[K-1] - c[0]) / r_sd`)
        are computed within the `K`-slice.
      - Per-pair raw and z cost differences cover the other `K - 1` slots (`14` cols at `K=8` vs `6` at `K=4`).
      - Slot position is `j / (K - 1)`.
      - Fixed context descriptors (`top-8 sem24 centroid` and `pool rows 0..3`, `lm48_z`, `dope629`) unchanged.
    """
    assert len(rcost) >= k, f"Pool size {len(rcost)} < k={k}"
    rcost64 = rcost.astype(np.float64)
    order = np.argsort(rcost64, kind="mergesort")
    topk = order[:k]
    top8 = order[:8]
    prog_topk = prog_full[topk]

    r_mean = float(np.mean(rcost64))
    r_sd = max(float(np.std(rcost64)), 1e-12)
    r_z_all = (rcost64 - r_mean) / r_sd

    ck = rcost64[topk]
    ck_mean = float(np.mean(ck))
    ck_sd = max(float(np.std(ck)), 1e-6)
    ck_z = (ck - ck_mean) / ck_sd

    zk = r_z_all[topk]
    sm_tau1 = np.exp(-(zk - np.min(zk)))
    sm_tau1 /= np.sum(sm_tau1)
    sm_tau02 = np.exp(-(zk - np.min(zk)) / 0.2)
    sm_tau02 /= np.sum(sm_tau02)
    sm_ck = np.exp(-(ck_z - np.min(ck_z)))
    sm_ck /= np.sum(sm_ck)

    ap64 = aprior.astype(np.float64)
    ap_z_all = (ap64 - float(np.mean(ap64))) / max(float(np.std(ap64)), 1e-6)

    ctx_top2_zgap = float((ck[1] - ck[0]) / r_sd)
    ctx_topk_spread_z = float((ck[k - 1] - ck[0]) / r_sd)

    sem64 = sem24_full.astype(np.float64)
    sem_mean = sem64.mean(axis=0, keepdims=True)
    sem_std = sem64.std(axis=0, keepdims=True) + 1e-5
    sem_z_topk = np.clip((sem64[topk] - sem_mean) / sem_std, -5.0, 5.0).astype(
        np.float32
    )

    proj64 = proj16_full.astype(np.float64)
    proj_mean = proj64.mean(axis=0, keepdims=True)
    proj_std = proj64.std(axis=0, keepdims=True) + 1e-5
    proj_z_topk = np.clip((proj64[topk] - proj_mean) / proj_std, -5.0, 5.0).astype(
        np.float32
    )
    proj_unit_topk = proj_z_topk / np.maximum(
        np.linalg.norm(proj_z_topk, axis=1, keepdims=True), 1e-6
    )

    sem_topk = sem64[topk]
    sem_top8_centroid = sem64[top8].mean(axis=0)
    sem_top8_centroid /= max(float(np.linalg.norm(sem_top8_centroid)), 1e-6)
    sem_rev03 = sem64[:4]

    est_slice = prog_topk[:, 24:29]
    est_ids = np.where(
        est_slice.max(axis=1) > 0, np.argmax(est_slice, axis=1), -1
    ).astype(np.int32)
    op_counts = prog_topk[:, :23].sum(axis=1).astype(np.float64)

    d629 = dope629.astype(np.float32)
    d_summary = np.array(
        [
            float(np.mean(d629)),
            float(np.std(d629)),
            float(np.linalg.norm(d629) / math.sqrt(len(d629))),
        ],
        dtype=np.float32,
    )
    ctx_vec76 = np.concatenate(
        [lm48_z.astype(np.float32), d629[:25], d_summary], axis=0
    )

    feat_dim = 554 + 2 * (k - 1)
    rows_out = np.zeros((k, feat_dim), dtype=np.float32)
    denom_pos = float(max(k - 1, 1))
    for j in range(k):
        others = [m for m in range(k) if m != j]
        raw_diffs = [float(ck[j] - ck[m]) for m in others]
        z_diffs = [float((ck[j] - ck[m]) / r_sd) for m in others]
        block1 = np.array(
            [
                float(ck[j]),
                float(zk[j]),
                float(ck_z[j]),
                float(j),
                float(j) / denom_pos,
                1.0 if j == 0 else 0.0,
                float(ck[j] - ck[0]),
                0.0 if j == 0 else float((ck[j] - ck[0]) / r_sd),
                ctx_top2_zgap,
                ctx_topk_spread_z,
                float(sm_tau1[j]),
                float(sm_tau02[j]),
                float(sm_ck[j]),
                float(ap_z_all[topk[j]]),
                *raw_diffs,
                *z_diffs,
            ],
            dtype=np.float32,
        )

        cos_sem_other = [
            float(np.clip(np.dot(sem_topk[j], sem_topk[m]), -1.0, 1.0)) for m in others
        ]
        cos_proj_other = [
            float(np.clip(np.dot(proj_unit_topk[j], proj_unit_topk[m]), -1.0, 1.0))
            for m in others
        ]
        cos_sem_r1 = float(np.clip(np.dot(sem_topk[j], sem_topk[0]), -1.0, 1.0))
        cos_proj_r1 = float(
            np.clip(np.dot(proj_unit_topk[j], proj_unit_topk[0]), -1.0, 1.0)
        )
        cos_top8 = float(np.clip(np.dot(sem_topk[j], sem_top8_centroid), -1.0, 1.0))
        cos_rev4_mean = float(np.mean(np.clip(sem_rev03 @ sem_topk[j], -1.0, 1.0)))
        same_est_r1 = 1.0 if est_ids[j] == est_ids[0] else 0.0
        n_same_est = float(sum(1 for m in range(k) if est_ids[m] == est_ids[j]))

        block2 = np.array(
            [
                cos_sem_r1,
                float(np.mean(cos_sem_other)),
                float(np.max(cos_sem_other)),
                cos_proj_r1,
                float(np.mean(cos_proj_other)),
                float(np.max(cos_proj_other)),
                cos_top8,
                cos_rev4_mean,
                float(op_counts[j]),
                float(op_counts[j] - op_counts[0]),
                same_est_r1,
                n_same_est,
            ],
            dtype=np.float32,
        )

        rows_out[j] = np.concatenate(
            [
                block1,
                block2,
                prog_topk[j].astype(np.float32),
                sem_z_topk[j],
                proj_z_topk[j],
                ctx_vec76,
            ],
            axis=0,
        )
    return topk, rows_out


def select_from_topk_scores_with_veto(
    scores_k: np.ndarray,
    p_inv_k: np.ndarray,
    tau: float,
    margin: float,
) -> int:
    if math.isinf(margin):
        return 0
    best_j = int(np.argmax(scores_k))
    if best_j == 0:
        return 0
    if float(scores_k[best_j] - scores_k[0]) > float(margin):
        if float(p_inv_k[best_j]) > float(tau):
            return 0
        return best_j
    return 0

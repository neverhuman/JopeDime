"""H276 frozen K=8 inference feature function, copied without training dependencies."""

import math
import numpy as np


def build_topk(rcost, aprior, prog_topk, sem24_full, proj16_full, lm48_z, dope629, k):
    rcost64 = rcost.astype(np.float64)
    topk = np.argsort(rcost64, kind="mergesort")[:k]
    top8 = np.argsort(rcost64, kind="mergesort")[:8]
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
    ctx_spread_z = float((ck[k - 1] - ck[0]) / r_sd)
    sem64 = sem24_full.astype(np.float64)
    sem_z = np.clip(
        (sem64[topk] - sem64.mean(axis=0, keepdims=True))
        / (sem64.std(axis=0, keepdims=True) + 1e-5),
        -5.0,
        5.0,
    ).astype(np.float32)
    proj64 = proj16_full.astype(np.float64)
    proj_z = np.clip(
        (proj64[topk] - proj64.mean(axis=0, keepdims=True))
        / (proj64.std(axis=0, keepdims=True) + 1e-5),
        -5.0,
        5.0,
    ).astype(np.float32)
    proj_unit = proj_z / np.maximum(np.linalg.norm(proj_z, axis=1, keepdims=True), 1e-6)
    sem_topk = sem64[topk]
    centroid = sem64[top8].mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-6)
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
    width = 14 + 2 * (k - 1) + 12 + prog_topk.shape[1] + 24 + 16 + 76
    rows_out = np.zeros((k, width), dtype=np.float32)
    for j in range(k):
        others = [o for o in range(k) if o != j]
        block1 = np.array(
            [
                float(ck[j]),
                float(zk[j]),
                float(ck_z[j]),
                float(j),
                float(j) / float(k - 1),
                1.0 if j == 0 else 0.0,
                float(ck[j] - ck[0]),
                0.0 if j == 0 else float((ck[j] - ck[0]) / r_sd),
                ctx_top2_zgap,
                ctx_spread_z,
                float(sm_tau1[j]),
                float(sm_tau02[j]),
                float(sm_ck[j]),
                float(ap_z_all[topk[j]]),
                *[float(ck[j] - ck[o]) for o in others],
                *[float((ck[j] - ck[o]) / r_sd) for o in others],
            ],
            dtype=np.float32,
        )
        cos_sem_other = [
            float(np.clip(np.dot(sem_topk[j], sem_topk[o]), -1.0, 1.0)) for o in others
        ]
        cos_proj_other = [
            float(np.clip(np.dot(proj_unit[j], proj_unit[o]), -1.0, 1.0))
            for o in others
        ]
        block2 = np.array(
            [
                float(np.clip(np.dot(sem_topk[j], sem_topk[0]), -1.0, 1.0)),
                float(np.mean(cos_sem_other)),
                float(np.max(cos_sem_other)),
                float(np.clip(np.dot(proj_unit[j], proj_unit[0]), -1.0, 1.0)),
                float(np.mean(cos_proj_other)),
                float(np.max(cos_proj_other)),
                float(np.clip(np.dot(sem_topk[j], centroid), -1.0, 1.0)),
                float(np.mean(np.clip(sem_rev03 @ sem_topk[j], -1.0, 1.0))),
                float(op_counts[j]),
                float(op_counts[j] - op_counts[0]),
                1.0 if est_ids[j] == est_ids[0] else 0.0,
                float(sum(1 for o in range(k) if est_ids[o] == est_ids[j])),
            ],
            dtype=np.float32,
        )
        rows_out[j] = np.concatenate(
            [
                block1,
                block2,
                prog_topk[j].astype(np.float32),
                sem_z[j],
                proj_z[j],
                ctx_vec76,
            ],
            axis=0,
        )
    return topk, rows_out
